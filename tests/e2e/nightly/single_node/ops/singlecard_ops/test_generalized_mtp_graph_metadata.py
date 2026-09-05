# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent metadata and scratch ownership across NPU graph replays."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.indexer import AscendSFAIndexerMetadata
from vllm_ascend.attention.sfa_kv_offload import AscendSFAKVOffloadMetadata
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.generalized_mtp import (
    GeneralizedMtpRuntime,
    make_mtp_batch,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.generalized_mtp_graph import (
    MtpGraphBuffers,
    MtpGraphMetadataSet,
    eligible_graph_steps,
)


def fixture():
    device = torch.device("npu:0")
    manager = SimpleNamespace(
        block_size=128,
        topk_buffer_size=8192,
        max_num_reqs=3,
        max_num_topk_rows=12,
        topk_buffer_slot_manager=SimpleNamespace(req2slot={"a": 2, "b": 0}),
        nano_mtp_slot_generations={2: 1, 0: 1},
    )
    metadata = SimpleNamespace(
        num_prefills=0,
        num_decodes=2,
        cum_query_lens=torch.tensor([4, 8], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([8452, 8196], dtype=torch.int32, device=device),
        req_topk_buffer_slots=torch.tensor([2, 0], dtype=torch.int32, device=device),
        block_table=torch.arange(256, dtype=torch.int32, device=device).reshape(2, 128),
    )
    runtime = GeneralizedMtpRuntime(manager)
    buffers = MtpGraphBuffers(runtime, query_width=4, source_capacity=16384, device=device)
    return manager, metadata, buffers


def test_graph_padding_owns_private_rows_and_fixed_tail_storage():
    manager, metadata, buffers = fixture()
    batch = make_mtp_batch(metadata, manager)
    buffers.update(batch)
    assert buffers.batch.pool_rows == [2, 0, 5]
    assert buffers.batch.tail_sources.numel() == 3 * 256
    assert buffers.tail_valid.cpu().sum().item() == 8
    valid = buffers.tail_valid
    torch.testing.assert_close(buffers.batch.tail_sources[valid], batch.tail_sources)
    torch.testing.assert_close(buffers.batch.tail_destinations[valid], batch.tail_destinations)
    scratch_destinations = buffers.batch.tail_destinations[-256:]
    assert int(scratch_destinations.min().cpu()) >= manager.max_num_reqs * (8192 + 256)
    buffers.prepare_layer("owner")
    assert buffers.layers["owner"][1].cpu().tolist() == [-2, -2, -2]
    buffers.prepare_layer("owner")
    assert buffers.layers["owner"][1].cpu().tolist() == [-1, -1, -2]
    manager.nano_mtp_slot_generations[2] += 1
    buffers.prepare_layer("owner")
    assert buffers.layers["owner"][1].cpu().tolist() == [-2, -1, -2]


def test_graph_inputs_change_without_replacing_captured_storage():
    manager, metadata, buffers = fixture()
    buffers.update(make_mtp_batch(metadata, manager))
    watched = [
        buffers.batch.seq_lens,
        buffers.batch.pool_entries,
        buffers.batch.tail_sources,
        buffers.batch.tail_destinations,
        buffers.tail_valid,
    ]
    addresses = [tensor.data_ptr() for tensor in watched]
    snapshots = [torch.empty_like(tensor) for tensor in watched]
    graph = torch.npu.NPUGraph()
    torch.npu.synchronize()
    with torch.npu.graph(graph):
        for snapshot, value in zip(snapshots, watched):
            snapshot.copy_(value)
    graph.replay()
    torch.npu.synchronize()
    assert snapshots[0].cpu().tolist() == [8452, 8196, 8196]
    # A smaller next batch changes pool ownership and crosses a tail block.
    metadata.num_decodes = 1
    metadata.seq_lens[0] = 8580
    buffers.update(make_mtp_batch(metadata, manager))
    graph.replay()
    torch.npu.synchronize()
    assert [tensor.data_ptr() for tensor in watched] == addresses
    assert snapshots[0].cpu().tolist() == [8580, 8196, 8196]
    assert snapshots[1].cpu().tolist() == [2, 4, 5]
    assert snapshots[-1].cpu().sum().item() == 4
    for snapshot, current in zip(snapshots, watched):
        torch.testing.assert_close(snapshot, current)


def test_nonuniform_batch_does_not_reuse_uniform_graph():
    manager, metadata, buffers = fixture()
    metadata.cum_query_lens.copy_(torch.tensor([1, 5], dtype=torch.int32, device="npu:0"))
    batch = make_mtp_batch(metadata, manager)
    assert not buffers.accepts(batch)
    with pytest.raises(ValueError, match="uniform-query capacity"):
        buffers.update(batch)


def test_graph_step_uses_owned_batch_after_builder_scratch_is_reused():
    manager, source, buffers = fixture()
    batch = make_mtp_batch(source, manager)
    device = source.seq_lens.device
    metadata = AscendSFAKVOffloadMetadata(
        num_actual_tokens=8,
        num_input_tokens=16,
        slot_mapping=torch.arange(16, dtype=torch.int32, device=device),
        seq_lens=source.seq_lens,
        seq_lens_cpu=source.seq_lens.cpu(),
        cum_query_lens=source.cum_query_lens,
        block_table=source.block_table,
        sin=torch.zeros(16, 64, device=device),
        cos=torch.ones(16, 64, device=device),
        num_decodes=2,
        num_decode_tokens=8,
        mtp_batch=batch,
    )
    graphs = MtpGraphMetadataSet(buffers.runtime, [{"owner": metadata}])
    # The next draft builder reuses these SFA scratch buffers, while the
    # first step's MtpBatch still owns its rejection-adjusted snapshots.
    source.cum_query_lens.zero_()
    source.seq_lens.zero_()
    source.block_table.zero_()
    graphs.update([{"owner": metadata}])
    captured = graphs.steps[0]["owner"]
    assert captured.cum_query_lens.cpu().tolist() == [4, 8, 12]
    assert captured.seq_lens.cpu().tolist() == [8452, 8196, 8196]
    torch.testing.assert_close(captured.block_table[:2], batch.source_block_table)
    assert captured.req_topk_buffer_slots.cpu().tolist() == [2, 0, 5]


def test_graph_step_keeps_ordinary_attention_metadata():
    manager, source, buffers = fixture()
    batch = make_mtp_batch(source, manager)
    device = source.seq_lens.device
    sparse = AscendSFAKVOffloadMetadata(
        num_actual_tokens=8,
        num_input_tokens=16,
        slot_mapping=torch.arange(16, dtype=torch.int32, device=device),
        seq_lens=source.seq_lens,
        seq_lens_cpu=source.seq_lens.cpu(),
        cum_query_lens=source.cum_query_lens,
        block_table=source.block_table,
        sin=torch.zeros(16, 64, device=device),
        cos=torch.ones(16, 64, device=device),
        num_decodes=2,
        num_decode_tokens=8,
        mtp_batch=batch,
    )
    ordinary = SimpleNamespace(seq_lens=source.seq_lens)
    step = {"ordinary": ordinary, "sparse": sparse}

    assert eligible_graph_steps([step])
    assert not eligible_graph_steps([{"ordinary": ordinary}])
    graphs = MtpGraphMetadataSet(buffers.runtime, [step])
    assert graphs.steps[0]["ordinary"] is ordinary
    graphs.update([step])


def test_graph_owns_separate_indexer_mappings_and_refreshes_them():
    manager, source, buffers = fixture()
    batch = make_mtp_batch(source, manager)
    device = source.seq_lens.device
    metadata = AscendSFAKVOffloadMetadata(
        num_actual_tokens=8,
        num_input_tokens=16,
        slot_mapping=torch.arange(16, dtype=torch.int32, device=device),
        seq_lens=source.seq_lens,
        seq_lens_cpu=source.seq_lens.cpu(),
        cum_query_lens=source.cum_query_lens,
        block_table=source.block_table,
        sin=torch.zeros(16, 64, device=device),
        cos=torch.ones(16, 64, device=device),
        num_decodes=2,
        num_decode_tokens=8,
        mtp_batch=batch,
    )
    indexer = AscendSFAIndexerMetadata(source.block_table + 100, metadata.slot_mapping + 50)
    step = {"main": metadata, "indexer": indexer}
    graphs = MtpGraphMetadataSet(buffers.runtime, [step])
    owned = graphs.steps[0]["indexer"]
    assert owned.block_table.data_ptr() != indexer.block_table.data_ptr()
    pointers = [owned.block_table.data_ptr(), owned.slot_mapping.data_ptr()]
    graphs.update([step])
    snapshots = [torch.empty_like(owned.block_table), torch.empty_like(owned.slot_mapping)]
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        snapshots[0].copy_(owned.block_table)
        snapshots[1].copy_(owned.slot_mapping)
    indexer.block_table.add_(17)
    indexer.slot_mapping.add_(11)
    graphs.update([step])
    graph.replay()
    torch.npu.synchronize()
    assert pointers == [owned.block_table.data_ptr(), owned.slot_mapping.data_ptr()]
    torch.testing.assert_close(snapshots[0][:2], indexer.block_table)
    assert snapshots[0][2].count_nonzero().item() == 0
    torch.testing.assert_close(snapshots[1], indexer.slot_mapping)
    torch.testing.assert_close(graphs.steps[0]["main"].block_table[:2], batch.source_block_table)
