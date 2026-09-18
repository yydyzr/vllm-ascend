# SPDX-License-Identifier: Apache-2.0
"""Diagnostics for nano tail D2D and decode ring writes."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload import nano_tail_debug


@pytest.fixture
def nano_tail_debug_on(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_NANO_TAIL_DEBUG", "1")
    nano_tail_debug.reset_nano_tail_debug_counters()
    yield
    monkeypatch.delenv("VLLM_ASCEND_NANO_TAIL_DEBUG", raising=False)
    nano_tail_debug.reset_nano_tail_debug_counters()


def test_emit_is_silent_when_disabled(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_NANO_TAIL_DEBUG", raising=False)
    with patch.object(nano_tail_debug.logger, "info") as info:
        nano_tail_debug.emit_nano_tail_debug("bind", req="r0")
    info.assert_not_called()


def test_emit_writes_structured_payload(nano_tail_debug_on):
    with patch.object(nano_tail_debug.logger, "info") as info:
        nano_tail_debug.emit_nano_tail_debug("d2d_append", req="r0", tail_tokens=127)
    info.assert_called_once()
    prefix, payload = info.call_args.args[1], info.call_args.args[2]
    assert prefix == nano_tail_debug.NANO_TAIL_DEBUG_PREFIX
    assert json.loads(payload) == {"event": "d2d_append", "req": "r0", "tail_tokens": 127}


def test_exec_kv_ring_logs_circular_suffix(nano_tail_debug_on):
    torch = pytest.importorskip("torch")
    slots = torch.tensor([8192 + 127, 8192 + 128], dtype=torch.int64)
    active = torch.tensor([True, True])
    with patch.object(nano_tail_debug.logger, "info") as info:
        nano_tail_debug.emit_nano_exec_kv_ring(
            layer_name="layers.0.self_attn",
            layer_id=0,
            device_slots=slots,
            token_active=active,
            hot_tokens=8192,
        )
        nano_tail_debug.emit_nano_exec_kv_ring(
            layer_name="layers.1.self_attn",
            layer_id=1,
            device_slots=slots,
            token_active=active,
            hot_tokens=8192,
        )
    assert info.call_count == 1
    payload = json.loads(info.call_args.args[2])
    assert payload["event"] == "exec_kv_ring"
    assert payload["in_circular_tail"] == [True, True]
    assert payload["ring_offsets"] == [8319, 8320]


def _tail_verify_fixture(torch, *, host_tokens=512, dim_k=8, dim_v=4, ring_tokens=384):
    """One 112-token span seeded from host token 256 into ring token 256."""
    generator = torch.Generator().manual_seed(0)
    host_k = torch.rand(host_tokens, dim_k, generator=generator)
    host_v = torch.rand(host_tokens, dim_v, generator=generator)
    ring_k = torch.zeros(ring_tokens, dim_k)
    ring_v = torch.zeros(ring_tokens, dim_v)
    return {
        "layer_name": "layers.0.self_attn",
        "layer_id": 0,
        "host_k": host_k,
        "ring_k": ring_k,
        "host_v": host_v,
        "ring_v": ring_v,
        "tail_src": torch.tensor([[256, 0]], dtype=torch.int64),
        "tail_dst": torch.tensor([[256, 0]], dtype=torch.int64),
        "tail_lengths": torch.tensor([[112, 0]], dtype=torch.int32),
        "tp_rank": 0,
    }


def _seed_ring(kwargs, src_token):
    kwargs["ring_k"][256:368] = kwargs["host_k"][src_token : src_token + 112]
    kwargs["ring_v"][256:368] = kwargs["host_v"][src_token : src_token + 112]


def _emit_tail_verify(kwargs):
    with patch.object(nano_tail_debug.logger, "info") as info:
        nano_tail_debug.emit_nano_tail_verify(**kwargs)
    if not info.call_args_list:
        return None
    return json.loads(info.call_args.args[2])


def test_tail_verify_reports_match_when_ring_mirrors_host(nano_tail_debug_on):
    torch = pytest.importorskip("torch")
    kwargs = _tail_verify_fixture(torch)
    _seed_ring(kwargs, 256)
    payload = _emit_tail_verify(kwargs)
    assert payload["event"] == "tail_verify"
    span = payload["spans"][0]
    assert (span["src_token"], span["dst_token"], span["tokens"]) == (256, 256, 112)
    for component in ("k", "v"):
        assert span[component]["mismatch_tokens"] == 0
        assert span[component]["max_abs_diff"] == 0.0
        assert span[component]["first_mismatch"] == -1
        assert span[component]["ring_all_zero"] is False


def test_tail_verify_locates_off_by_one_block_source(nano_tail_debug_on):
    torch = pytest.importorskip("torch")
    kwargs = _tail_verify_fixture(torch)
    # Ring seeded from the block before the one the descriptors point at.
    _seed_ring(kwargs, 128)
    span = _emit_tail_verify(kwargs)["spans"][0]
    assert span["k"]["mismatch_tokens"] > 0
    assert span["k"]["first_mismatch"] == 0
    assert span["k"]["ring_source_token"] == 128


def test_tail_verify_flags_unwritten_ring(nano_tail_debug_on):
    torch = pytest.importorskip("torch")
    span = _emit_tail_verify(_tail_verify_fixture(torch))["spans"][0]
    assert span["k"]["ring_all_zero"] is True
    assert span["k"]["host_all_zero"] is False
    assert span["k"]["mismatch_tokens"] == 112
    # An all-zero ring matches no host token, so the search reports nothing.
    assert span["k"]["ring_source_token"] == -1


def test_tail_verify_skips_aligned_prefix_and_non_first_layer(nano_tail_debug_on):
    torch = pytest.importorskip("torch")
    aligned = _tail_verify_fixture(torch)
    aligned["tail_lengths"] = torch.zeros((1, 2), dtype=torch.int32)
    assert _emit_tail_verify(aligned) is None

    later_layer = _tail_verify_fixture(torch)
    later_layer["layer_id"] = 1
    assert _emit_tail_verify(later_layer) is None


def test_tail_verify_skips_ranks_without_a_cpu_mapped_host_pool(nano_tail_debug_on):
    torch = pytest.importorskip("torch")
    kwargs = _tail_verify_fixture(torch)
    _seed_ring(kwargs, 256)
    kwargs["tp_rank"] = 1
    assert _emit_tail_verify(kwargs) is None


def test_tail_verify_is_silent_when_disabled(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.delenv("VLLM_ASCEND_NANO_TAIL_DEBUG", raising=False)
    nano_tail_debug.reset_nano_tail_debug_counters()
    assert _emit_tail_verify(_tail_verify_fixture(torch)) is None


def test_tail_verify_stops_after_log_cap(nano_tail_debug_on):
    torch = pytest.importorskip("torch")
    emitted = 0
    for _ in range(nano_tail_debug._MAX_TAIL_VERIFY_LOGS + 2):
        kwargs = _tail_verify_fixture(torch)
        _seed_ring(kwargs, 256)
        emitted += _emit_tail_verify(kwargs) is not None
    assert emitted == nano_tail_debug._MAX_TAIL_VERIFY_LOGS


def _restore_probe_fixture(torch, *, ring_tokens=384, dim_k=8, dim_v=4):
    """One 112-token span at ring token 256, restored from a host stand-in."""
    generator = torch.Generator().manual_seed(1)
    host_k = torch.rand(112, dim_k, generator=generator)
    host_v = torch.rand(112, dim_v, generator=generator)
    ring_k = torch.zeros(ring_tokens, dim_k)
    ring_v = torch.zeros(ring_tokens, dim_v)

    def restore():
        ring_k[256:368] = host_k
        ring_v[256:368] = host_v

    return {
        "layer_name": "layers.0.self_attn",
        "layer_id": 0,
        "ring_k": ring_k,
        "ring_v": ring_v,
        "tail_dst": torch.tensor([[256, 0]], dtype=torch.int64),
        "tail_lengths": torch.tensor([[112, 0]], dtype=torch.int32),
        "restore": restore,
    }, (host_k, host_v)


def _probe_restore(kwargs):
    with patch.object(nano_tail_debug.logger, "info") as info:
        nano_tail_debug.probe_nano_tail_restore(**kwargs)
    if not info.call_args_list:
        return None
    return json.loads(info.call_args.args[2])


def test_restore_probe_reports_zero_diff_when_d2d_already_matches(nano_tail_debug_on):
    torch = pytest.importorskip("torch")
    kwargs, (host_k, host_v) = _restore_probe_fixture(torch)
    kwargs["ring_k"][256:368] = host_k
    kwargs["ring_v"][256:368] = host_v
    payload = _probe_restore(kwargs)
    assert payload["event"] == "tail_restore_diff"
    span = payload["spans"][0]
    assert (span["dst_token"], span["tokens"]) == (256, 112)
    for component in ("k", "v"):
        assert span[component]["max_abs_diff"] == 0.0
        assert span[component]["mismatch_tokens"] == 0
        assert span[component]["first_mismatch"] == -1
        assert span[component]["d2d_all_zero"] is False


def test_restore_probe_reports_diff_for_unseeded_ring(nano_tail_debug_on):
    torch = pytest.importorskip("torch")
    kwargs, _ = _restore_probe_fixture(torch)
    span = _probe_restore(kwargs)["spans"][0]
    assert span["k"]["mismatch_tokens"] == 112
    assert span["k"]["first_mismatch"] == 0
    assert span["k"]["d2d_all_zero"] is True
    assert span["k"]["restored_all_zero"] is False


def test_restore_probe_leaves_the_d2d_payload_in_place(nano_tail_debug_on):
    torch = pytest.importorskip("torch")
    kwargs, _ = _restore_probe_fixture(torch)
    seeded_k = torch.full((112, kwargs["ring_k"].shape[-1]), 0.5)
    kwargs["ring_k"][256:368] = seeded_k
    untouched = kwargs["ring_k"][:256].clone()
    assert _probe_restore(kwargs) is not None
    # The restore ran, but the model must still see the seeded tail.
    assert torch.equal(kwargs["ring_k"][256:368], seeded_k)
    assert torch.equal(kwargs["ring_k"][:256], untouched)
    assert torch.count_nonzero(kwargs["ring_v"][256:368]) == 0


def test_restore_probe_skips_aligned_prefix_without_restoring(nano_tail_debug_on):
    torch = pytest.importorskip("torch")
    kwargs, _ = _restore_probe_fixture(torch)
    kwargs["tail_lengths"] = torch.zeros((1, 2), dtype=torch.int32)
    assert _probe_restore(kwargs) is None
    assert torch.count_nonzero(kwargs["ring_k"]) == 0


def test_restore_probe_stops_after_log_cap(nano_tail_debug_on):
    torch = pytest.importorskip("torch")
    emitted = 0
    for _ in range(nano_tail_debug._MAX_RESTORE_DIFF_LOGS + 2):
        kwargs, _ = _restore_probe_fixture(torch)
        emitted += _probe_restore(kwargs) is not None
    assert emitted == nano_tail_debug._MAX_RESTORE_DIFF_LOGS


def test_restore_probe_is_silent_when_disabled(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.delenv("VLLM_ASCEND_NANO_TAIL_DEBUG", raising=False)
    nano_tail_debug.reset_nano_tail_debug_counters()
    kwargs, _ = _restore_probe_fixture(torch)
    assert _probe_restore(kwargs) is None
    assert torch.count_nonzero(kwargs["ring_k"]) == 0


def test_scheduler_bind_emits_geometry(nano_tail_debug_on):
    from tests.ut.kv_offload.test_sfa_pd_rd2h_connector import (
        test_consumer_scheduler_binds_nano_tail_at_alloc,
    )

    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.scheduler.emit_nano_tail_debug"
    ) as emit:
        test_consumer_scheduler_binds_nano_tail_at_alloc()
    events = [call.args[0] for call in emit.call_args_list]
    assert "bind" in events
    bind = next(call.kwargs for call in emit.call_args_list if call.args[0] == "bind")
    assert bind["tail_tokens"] == 127
    assert bind["tail_block_index"] == 80
    assert bind["aligned_prefix"] is False


def test_worker_load_dest_emits(nano_tail_debug_on):
    from tests.ut.kv_offload.test_sfa_pd_rd2h_connector import (
        test_consumer_worker_records_nano_slot_for_runner,
    )

    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.worker.emit_nano_tail_debug"
    ) as emit:
        test_consumer_worker_records_nano_slot_for_runner()
    assert emit.call_args.args[0] == "load_dest"
    assert emit.call_args.kwargs["tail_tokens"] == 7
    assert emit.call_args.kwargs["tail_block_index"] == 4


def test_d2d_append_and_chunk_skip_emit(nano_tail_debug_on):
    from tests.ut.kv_offload.test_sfa_pd_rd2h_connector import (
        test_nano_tail_d2d_appends_to_every_decode_rank,
        test_nano_tail_d2d_skips_when_last_block_is_not_in_chunk,
    )

    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.read_thread.emit_nano_tail_debug"
    ) as emit:
        test_nano_tail_d2d_appends_to_every_decode_rank()
        test_nano_tail_d2d_skips_when_last_block_is_not_in_chunk()
    events = [call.args[0] for call in emit.call_args_list]
    assert "d2d_append" in events
    assert "d2d_skip" in events
    append = next(call.kwargs for call in emit.call_args_list if call.args[0] == "d2d_append")
    assert append["dst_token"] == 384
    assert append["tail_tokens"] == 3
    skip = next(call.kwargs for call in emit.call_args_list if call.args[0] == "d2d_skip")
    assert skip["reason"] == "not_in_chunk"
