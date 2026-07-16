"""Unit tests for SFAKVOffloadWorker layer registration.

Covers:
- offload layer selection by tuple length (five/six-tuple in, others out)
- the mixed LIC8 / non-LIC8 guard: under offload, C8 must be uniform across
  sparse layers (the attention path gates the quant indexer on a GLOBAL flag),
  so coexisting five- and six-tuple layers must raise.

The worker module JIT-builds a C++ extension and imports memfabric_hybrid at
module load time, neither of which is available in the UT sandbox; both are
stubbed before the import below.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

# Stub heavy module-level dependencies BEFORE importing the worker.
# 1. cpu_sparse_attn cpp extension JIT build (torch.utils.cpp_extension.load).
import torch.utils.cpp_extension as _cpp_extension  # noqa: E402

_cpp_extension.load = MagicMock(return_value=MagicMock())  # noqa: E402

# 2. memfabric_hybrid.offload is not exported in the sandbox install.
import memfabric_hybrid  # noqa: E402

if not hasattr(memfabric_hybrid, "offload"):  # noqa: E402
    memfabric_hybrid.offload = MagicMock()  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from vllm_ascend.distributed.kv_transfer.sfa_kv_offload import (  # noqa: E402
    sfa_kv_offload_worker as worker_module,
)
from vllm_ascend.distributed.kv_transfer.sfa_kv_offload.sfa_kv_offload_worker import (  # noqa: E402
    SFAKVOffloadWorker,
)


def _make_worker_without_init() -> SFAKVOffloadWorker:
    """Bypass __init__ (heavy); set only the attrs _register_offload_layers reads."""
    w = SFAKVOffloadWorker.__new__(SFAKVOffloadWorker)
    w.num_target_layers = 0
    w.tp_rank = 0
    w.pending_save_layer_ids = set()
    w.submitted_save_layer_ids = set()
    return w


def _tuple(n: int) -> tuple:
    return tuple(torch.zeros(1) for _ in range(n))


def test_register_selects_offload_tuples_and_skips_others():
    # Five- and six-tuple layers are offload candidates; single tensors and
    # other lengths are skipped. (5- and 6-tuple cannot coexist — see the
    # mixed guard test below — so exercise them in separate dicts.)
    for offload_len in (5, 6):
        w = _make_worker_without_init()
        kv_caches = {
            "layer.0": _tuple(offload_len),
            "layer.1": _tuple(offload_len),
            "indexer.layer.0": torch.zeros(1),  # single tensor, not an offload tuple
            "layer.2": _tuple(3),  # neither five- nor six-tuple
        }
        w._register_offload_layers(kv_caches)
        assert w.offload_layer_names == ["layer.0", "layer.1"]
        assert w.num_offload_layers == 2


def test_register_raises_when_no_offload_layers():
    w = _make_worker_without_init()
    with pytest.raises(ValueError, match="did not find SFA KV cache layers"):
        w._register_offload_layers({"layer.0": _tuple(3)})


def test_register_all_five_tuple_passes():
    w = _make_worker_without_init()
    w._register_offload_layers({"layer.0": _tuple(5), "layer.1": _tuple(5)})
    assert w.num_offload_layers == 2


def test_register_all_six_tuple_passes():
    w = _make_worker_without_init()
    w._register_offload_layers({"layer.0": _tuple(6), "layer.1": _tuple(6)})
    assert w.num_offload_layers == 2


def test_register_rejects_mixed_five_and_six_tuple():
    """Mixed LIC8 / non-LIC8 layers under offload would route a non-C8 layer
    through the quant indexer (global flag) — must raise at registration."""
    w = _make_worker_without_init()
    kv_caches = {"layer.0": _tuple(5), "layer.1": _tuple(6)}
    with pytest.raises(ValueError, match="mixed LIC8 / non-LIC8"):
        w._register_offload_layers(kv_caches)


def _make_runtime_worker(tp_rank: int = 0, tp_size: int = 2) -> SFAKVOffloadWorker:
    worker = SFAKVOffloadWorker.__new__(SFAKVOffloadWorker)
    worker.tp_rank = tp_rank
    worker.tp_size = tp_size
    worker.allocate_dram_size = 64 * 1024 * 1024 * 1024
    worker.tp_group = MagicMock()
    worker.tp_group.all_reduce.side_effect = lambda tensor: tensor
    return worker


@pytest.mark.parametrize("use_fused_overlap_offload", [False, True])
def test_initialize_memfabric_uses_rank_aware_config(
    monkeypatch,
    use_fused_overlap_offload,
):
    worker = _make_runtime_worker(tp_rank=1, tp_size=4)
    worker.use_fused_overlap_offload = use_fused_overlap_offload
    config = SimpleNamespace()
    monkeypatch.setattr(worker_module.offload, "OffloadConfig", MagicMock(return_value=config))
    initialize = MagicMock(return_value=0)
    monkeypatch.setattr(worker_module.offload, "initialize", initialize)
    monkeypatch.setattr(worker_module.torch_npu.npu, "current_device", lambda: 3)
    real_tensor = torch.tensor

    def cpu_tensor(*args, **kwargs):
        kwargs.pop("device", None)
        return real_tensor(*args, **kwargs)

    monkeypatch.setattr(worker_module.torch, "tensor", cpu_tensor)

    worker._initialize_memfabric()

    assert config.device_id == 3
    assert config.size == worker.allocate_dram_size
    assert config.world_size == 4
    assert config.rank_id == 1
    initialize.assert_called_once_with(config)
    worker.tp_group.barrier.assert_called_once_with()


def test_initialize_memfabric_propagates_local_failure(monkeypatch):
    worker = _make_runtime_worker(tp_rank=0, tp_size=2)
    monkeypatch.setattr(worker_module.offload, "OffloadConfig", MagicMock(return_value=SimpleNamespace()))
    monkeypatch.setattr(worker_module.offload, "initialize", MagicMock(return_value=7))
    monkeypatch.setattr(worker_module.torch_npu.npu, "current_device", lambda: 0)
    real_tensor = torch.tensor

    def cpu_tensor(*args, **kwargs):
        kwargs.pop("device", None)
        return real_tensor(*args, **kwargs)

    monkeypatch.setattr(worker_module.torch, "tensor", cpu_tensor)

    with pytest.raises(RuntimeError, match=r"tp_rank=0.*ret=7"):
        worker._initialize_memfabric()

    worker.tp_group.barrier.assert_not_called()


def test_initialize_memfabric_propagates_peer_failure(monkeypatch):
    worker = _make_runtime_worker(tp_rank=1, tp_size=2)
    worker.tp_group.all_reduce.side_effect = None
    worker.tp_group.all_reduce.return_value = torch.tensor([1], dtype=torch.int32)
    monkeypatch.setattr(worker_module.offload, "OffloadConfig", MagicMock(return_value=SimpleNamespace()))
    monkeypatch.setattr(worker_module.offload, "initialize", MagicMock(return_value=0))
    monkeypatch.setattr(worker_module.torch_npu.npu, "current_device", lambda: 1)
    real_tensor = torch.tensor

    def cpu_tensor(*args, **kwargs):
        kwargs.pop("device", None)
        return real_tensor(*args, **kwargs)

    monkeypatch.setattr(worker_module.torch, "tensor", cpu_tensor)

    with pytest.raises(RuntimeError, match="another TP rank failed"):
        worker._initialize_memfabric()

    worker.tp_group.barrier.assert_not_called()


def test_validate_owner_gva_rejects_zero_and_unaligned():
    with pytest.raises(RuntimeError, match="is zero"):
        SFAKVOffloadWorker._validate_owner_gva(0, name="main_k")
    with pytest.raises(RuntimeError, match="not 2MB aligned"):
        SFAKVOffloadWorker._validate_owner_gva(4096, name="main_k")

    SFAKVOffloadWorker._validate_owner_gva(2 * 1024 * 1024, name="main_k")


def test_restore_non_owner_view_validates_tensor_metadata(monkeypatch):
    ptr = 2 * 1024 * 1024
    shape = [1, 128, 1, 512]
    view = MagicMock()
    view.data_ptr.return_value = ptr
    view.shape = torch.Size(shape)
    view.dtype = torch.bfloat16
    view.is_contiguous.return_value = True
    restore = MagicMock(return_value=view)
    monkeypatch.setattr(worker_module.cpu_sparse_attn, "restore_bfloat16_tensor", restore)

    assert SFAKVOffloadWorker._restore_bfloat16_tensor(ptr, shape) is view
    restore.assert_called_once_with(ptr, shape)


def test_owned_cpu_kv_pools_are_exposed_only_on_tp0():
    owner = _make_runtime_worker(tp_rank=0)
    owner.k_caches_cpu = [torch.empty(1)]
    owner.v_caches_cpu = [torch.empty(1)]
    assert owner.get_owned_cpu_kv_pools() == (owner.k_caches_cpu, owner.v_caches_cpu)

    non_owner = _make_runtime_worker(tp_rank=1)
    non_owner.k_caches_cpu = [MagicMock(name="non_owner_k_view")]
    non_owner.v_caches_cpu = [MagicMock(name="non_owner_v_view")]
    assert non_owner.get_owned_cpu_kv_pools() == (None, None)


def _make_d2h_worker(tp_rank: int) -> SFAKVOffloadWorker:
    worker = _make_runtime_worker(tp_rank=tp_rank)
    worker.use_fused_overlap_offload = True
    worker.fused_step_has_offload = True
    worker.d2h_status_npu = torch.zeros(1, dtype=torch.int32)
    worker._save_current_kv_tokens_on_owner = MagicMock()
    return worker


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_current_token_d2h_uses_tp0_and_all_ranks_barrier(monkeypatch, tp_rank):
    worker = _make_d2h_worker(tp_rank)
    monkeypatch.setattr(worker_module.envs, "VLLM_ASCEND_SFA_DEBUG", False)
    monkeypatch.setattr(worker_module, "_is_current_stream_capturing", lambda: False)
    tensor = torch.zeros(1, dtype=torch.int32)

    worker.save_current_kv_tokens("layer.0", tensor, tensor, tensor, 1, 1)

    if tp_rank == 0:
        worker._save_current_kv_tokens_on_owner.assert_called_once_with(
            "layer.0",
            tensor,
            tensor,
            tensor,
            1,
            1,
            capturing=False,
        )
    else:
        worker._save_current_kv_tokens_on_owner.assert_not_called()
    worker.tp_group.broadcast.assert_called_once_with(worker.d2h_status_npu, src=0)
    worker.tp_group.barrier.assert_called_once_with()


def test_current_token_d2h_eager_inactive_skips_copy_and_collectives(monkeypatch):
    worker = _make_d2h_worker(tp_rank=0)
    worker.fused_step_has_offload = False
    monkeypatch.setattr(worker_module, "_is_current_stream_capturing", lambda: False)
    tensor = torch.zeros(1, dtype=torch.int32)

    worker.save_current_kv_tokens("layer.0", tensor, tensor, tensor, 1, 1)

    worker._save_current_kv_tokens_on_owner.assert_not_called()
    worker.tp_group.broadcast.assert_not_called()
    worker.tp_group.barrier.assert_not_called()


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_current_token_d2h_capture_records_static_all_rank_path(monkeypatch, tp_rank):
    worker = _make_d2h_worker(tp_rank=tp_rank)
    worker.fused_step_has_offload = False
    monkeypatch.setattr(worker_module, "_is_current_stream_capturing", lambda: False)
    tensor = torch.zeros(1, dtype=torch.int32)

    worker.save_current_kv_tokens("layer.0", tensor, tensor, tensor, 1, 1, capturing=True)

    if tp_rank == 0:
        worker._save_current_kv_tokens_on_owner.assert_called_once_with(
            "layer.0",
            tensor,
            tensor,
            tensor,
            1,
            1,
            capturing=True,
        )
    else:
        worker._save_current_kv_tokens_on_owner.assert_not_called()
    worker.tp_group.broadcast.assert_called_once_with(worker.d2h_status_npu, src=0)
    worker.tp_group.barrier.assert_not_called()


def test_owner_zero_copy_skips_sparse_copy(monkeypatch):
    worker = _make_runtime_worker(tp_rank=0)
    worker._get_offload_layer_id = MagicMock(return_value=0)
    worker.d2h_slot_mapping_cpu = torch.empty(1, dtype=torch.int32)
    worker.d2h_token_to_req_cpu = torch.empty(1, dtype=torch.int32)
    worker.d2h_cum_query_lens_cpu = torch.empty(1, dtype=torch.int32)
    worker.cpu_block_table_host_buffer = torch.empty((1, 1), dtype=torch.int32)
    worker.cpu_block_table = SimpleNamespace(gpu=torch.empty((1, 1), dtype=torch.int32))
    worker.d2h_num_tokens_buffer_cpu = torch.zeros(1, dtype=torch.int32)
    worker._compute_step_offload_addrs_cpu = MagicMock()
    worker._fused_overlap_d2h_logged = True
    monkeypatch.setattr(worker_module.envs, "VLLM_ASCEND_SFA_DEBUG", False)
    monkeypatch.setattr(worker_module.torch_npu.npu, "current_stream", MagicMock(return_value=MagicMock()))
    sparse_copy = MagicMock()
    monkeypatch.setattr(worker_module.offload, "sparse_copy", sparse_copy)
    tensor = torch.zeros(1, dtype=torch.int32)

    worker._save_current_kv_tokens_on_owner("layer.0", tensor, tensor, tensor, 1, 1)

    sparse_copy.assert_not_called()


def test_owner_capture_zero_copy_still_records_sparse_copy(monkeypatch):
    worker = _make_runtime_worker(tp_rank=0)
    worker._get_offload_layer_id = MagicMock(return_value=0)
    worker.d2h_slot_mapping_cpu = torch.empty(1, dtype=torch.int32)
    worker.d2h_token_to_req_cpu = torch.empty(1, dtype=torch.int32)
    worker.d2h_cum_query_lens_cpu = torch.empty(1, dtype=torch.int32)
    worker.cpu_block_table_host_buffer = torch.empty((1, 1), dtype=torch.int32)
    worker.cpu_block_table = SimpleNamespace(gpu=torch.empty((1, 1), dtype=torch.int32))
    worker.d2h_num_tokens_buffer_cpu = MagicMock()
    worker.d2h_num_tokens_buffer_cpu.__getitem__.return_value.item.side_effect = AssertionError(
        "capture must not read copy_count on the host"
    )
    worker._compute_step_offload_addrs_cpu = MagicMock()
    worker.d2h_batch_copy_args_buffer_cpu = torch.zeros(4, dtype=torch.int8)
    worker.d2h_batch_copy_args_buffer_npu = MagicMock()
    worker.d2h_addr_buffer_npu = MagicMock()
    worker.d2h_gvas_buffer_npu = MagicMock()
    worker.d2h_size_buffer_npu = MagicMock()
    worker.d2h_num_tokens_buffer_npu = MagicMock()
    worker.k_caches_npu = [SimpleNamespace(device=torch.device("cpu"))]
    worker.d2h_save_event = MagicMock()
    current_stream = MagicMock()
    subscribed_streams = set()
    monkeypatch.setattr(worker_module.envs, "VLLM_ASCEND_SFA_DEBUG", False)
    monkeypatch.setattr(worker_module.torch_npu.npu, "current_stream", lambda: current_stream)
    monkeypatch.setattr(worker_module, "get_subscribed_compute_streams", lambda: subscribed_streams)
    subscribe_report = MagicMock()
    launch_host_func = MagicMock()
    monkeypatch.setattr(worker_module.torch_npu.npu, "_subscribe_report", subscribe_report)
    monkeypatch.setattr(worker_module.torch_npu.npu, "_launch_host_func", launch_host_func)
    sparse_copy = MagicMock()
    monkeypatch.setattr(worker_module.offload, "sparse_copy", sparse_copy)
    tensor = torch.zeros(1, dtype=torch.int32)

    worker._save_current_kv_tokens_on_owner(
        "layer.0",
        tensor,
        tensor,
        tensor,
        1,
        1,
        capturing=True,
    )

    subscribe_report.assert_called_once_with(current_stream)
    launch_host_func.assert_called_once_with(
        current_stream,
        worker._compute_step_offload_addrs_cpu,
        (1, 1, 0),
    )
    worker.d2h_batch_copy_args_buffer_npu.copy_.assert_called_once_with(
        worker.d2h_batch_copy_args_buffer_cpu,
        non_blocking=True,
    )
    sparse_copy.assert_called_once_with(
        worker.d2h_addr_buffer_npu,
        worker.d2h_gvas_buffer_npu,
        worker.d2h_size_buffer_npu,
        worker.d2h_num_tokens_buffer_npu,
        torch.device("cpu"),
    )
    worker.d2h_save_event.record.assert_not_called()
    worker.d2h_save_event.synchronize.assert_not_called()
