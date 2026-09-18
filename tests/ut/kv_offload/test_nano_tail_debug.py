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
    assert append["dst_token"] == 512
    assert append["tail_tokens"] == 3
    skip = next(call.kwargs for call in emit.call_args_list if call.args[0] == "d2d_skip")
    assert skip["reason"] == "not_in_chunk"
