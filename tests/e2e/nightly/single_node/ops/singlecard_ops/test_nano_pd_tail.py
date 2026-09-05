# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual registered-host H2D through the production nano descriptor builders."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401
from memfabric_hybrid import offload

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.nano_cache import (
    CopyDescriptors,
    prepare_resident_gather,
    prepare_tail_copy,
)

BLOCK = 128
HOT = 2048
STRIDE = HOT + 2 * BLOCK


@pytest.fixture(scope="module")
def registered_pool():
    torch.npu.set_device(0)
    torch.empty(1, device="npu:0")
    config = offload.OffloadConfig()
    config.device_id, config.reserve_size, config.alloc_size = 0, 1 << 30, 1 << 30
    config.world_size, config.rank_id, config.scene = 1, 0, offload.Scene.SHARED
    assert offload.initialize(config) == 0
    yield
    torch.npu.synchronize()
    offload.uninitialize()


def caches():
    host = [offload.empty((8, BLOCK, dim), dtype=torch.bfloat16, pin_memory=True) for dim in (512, 64)]
    for index, tensor in enumerate(host):
        tensor.copy_((torch.arange(tensor.numel()).reshape_as(tensor) % 113 + index).to(tensor.dtype))
    hbm = [torch.full((3, STRIDE, dim), -17, dtype=torch.bfloat16, device="npu:0") for dim in (512, 64)]
    geometry = dict(
        block_size=BLOCK,
        hot_tokens=HOT,
        source_bases=tuple(t.data_ptr() for t in host),
        destination_bases=tuple(t.data_ptr() for t in hbm),
        token_bytes=(1024, 128),
        source_block_capacity=8,
    )
    return host, hbm, geometry


def copy(descriptors):
    result = offload.sparse_copy(*descriptors.args(), torch.device("npu:0"))
    assert result in (None, 0)


def check(hbm, expected):
    torch.npu.synchronize()
    for actual, reference in zip(hbm, expected):
        torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)


def test_tail_h2d_replays_current_metadata_for_each_layer(registered_pool):
    layers = [caches(), caches()]
    for tensor in layers[1][0]:
        tensor.add_(20)
    expected = [[t.cpu() for t in hbm] for _, hbm, _ in layers]
    batch = SimpleNamespace(
        source_block_table=torch.zeros((2, 70), dtype=torch.int32, device="npu:0"),
        seq_lens=torch.zeros(2, dtype=torch.int32, device="npu:0"),
        offload_lens=torch.zeros(2, dtype=torch.int32, device="npu:0"),
        pool_entries=torch.tensor([0, 2], dtype=torch.int32, device="npu:0"),
    )
    active = torch.zeros(2, dtype=torch.bool, device="npu:0")
    descriptors = [CopyDescriptors(8, torch.device("npu:0")) for _ in layers]
    addresses = [[t.data_ptr() for t in d.args()] for d in descriptors]

    def update(requests):
        table = torch.zeros_like(batch.source_block_table, device="cpu")
        prefixes, lengths, pools = [0, 0], [0, 0], [2, 2]
        active.zero_()
        for row, (pool, logical, physical, count) in enumerate(requests):
            prefixes[row], lengths[row], pools[row] = logical * BLOCK, logical * BLOCK + count, pool
            for part, block in enumerate(physical):
                if logical + part < table.shape[1]:
                    table[row, logical + part] = block
            active[row] = True
            for layer, (host, _, _) in enumerate(layers):
                for component in range(2):
                    for offset in range(count):
                        token = logical * BLOCK + offset
                        expected[layer][component][pool, HOT + token % 256].copy_(
                            host[component][physical[offset // BLOCK], offset % BLOCK]
                        )
        for tensor, values in ((batch.seq_lens, lengths), (batch.offload_lens, prefixes), (batch.pool_entries, pools)):
            tensor.copy_(torch.tensor(values, dtype=tensor.dtype, device=tensor.device))
        batch.source_block_table.copy_(table)

    def chain():
        for d, (_, _, geometry) in zip(descriptors, layers):
            prepare_tail_copy(d, batch, active=active, **geometry)
            copy(d)

    first = [(0, 64, [5, 2], 131), (1, 65, [7, 1], 17)]
    update(first)
    chain()
    for (_, hbm, _), reference in zip(layers, expected):
        check(hbm, reference)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        chain()
    torch.npu.synchronize()
    cases = [first, [(1, 67, [3, 6], 129), (0, 68, [4, 0], 4)], [(1, 66, [6, 3], 2)], [], [(0, 69, [1, 7], 128)]]
    for requests in cases:
        for host, _, _ in layers:
            for tensor in host:
                tensor.add_(1)
        update(requests)
        graph.replay()
        for (_, hbm, _), reference in zip(layers, expected):
            check(hbm, reference)
        assert [[t.data_ptr() for t in d.args()] for d in descriptors] == addresses
    # Last case has no second logical table column, but zero copy bytes there.
    assert descriptors[0].lengths.cpu().tolist() == [131072, 0, 0, 0, 16384, 0, 0, 0]


def test_fallback_gather_respects_causality_padding_and_nano_stride(registered_pool):
    host, hbm, geometry = caches()
    expected = [t.cpu() for t in hbm]
    selected = torch.tensor([[0, 130, -1, 131], [259, 0, 260, -1], [1, 128, 129, -1]], device="npu:0")
    table = torch.tensor([[7, 2, 5], [3, 1, 6]], dtype=torch.int32, device="npu:0")
    requests = torch.tensor([0, 1, 0], dtype=torch.int64, device="npu:0")
    visible = torch.tensor([131, 260, 129], device="npu:0")
    descriptors = CopyDescriptors(24, torch.device("npu:0"))
    slots, resident_table = prepare_resident_gather(
        descriptors,
        block_table=table,
        selected=selected,
        token_to_request=requests,
        visible_lengths=visible,
        **geometry,
    )
    copy(descriptors)
    for row, request in enumerate(requests.cpu().tolist()):
        for column, token in enumerate(selected[row].cpu().tolist()):
            if 0 <= token < visible[row].item():
                physical = table[request, token // BLOCK].item()
                for component in range(2):
                    expected[component][row, column].copy_(host[component][physical, token % BLOCK])
    check(hbm, expected)
    assert slots.cpu().tolist() == [[0, 1, -1, -1], [0, 1, -1, -1], [0, 1, -1, -1]]
    assert resident_table.cpu().tolist() == [[0], [18], [36]]


@pytest.mark.parametrize("speculative", [None, SimpleNamespace(num_speculative_tokens=3)])
def test_true_pd_selects_generalized_abi(speculative):
    from vllm_ascend.ascend_config import SparseKVOffloadConfig

    config = SimpleNamespace(
        speculative_config=speculative,
        use_v2_model_runner=False,
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(index_topk=2048), enforce_eager=True),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1, decode_context_parallel_size=1, pipeline_parallel_size=1
        ),
        kv_transfer_config=SimpleNamespace(is_kv_consumer=True),
    )
    offload_config = SparseKVOffloadConfig(config, {"enabled": True, "fused_op_type": "nano"})
    assert offload_config.generalized_mtp
    assert not offload_config.keep_device_kv_cache
    assert offload_config.topk_buffer_size >= (4 if speculative else 1) * 2048
