# SPDX-License-Identifier: Apache-2.0
"""Optional long-sequence nano tail D2D / ring-write diagnostics.

Enable with ``VLLM_ASCEND_NANO_TAIL_DEBUG=1``. Disabled by default so the
decode hot path stays silent. Failures here must never affect serving.
"""

from __future__ import annotations

import json
from typing import Any

from vllm.logger import logger

from vllm_ascend import envs

# Circular tail is two 128-token pages after the hot prefix. Keep this local
# so connector import does not depend on newer nano_topk_slots symbols.
NANO_RING_TOKENS = 256
NANO_TAIL_DEBUG_PREFIX = "[NANO_TAIL_DEBUG]"
# exec_kv / attention run every layer every step. Cap so a long decode does
# not flood the log after the first few ring writes are visible.
_MAX_EXEC_KV_LOGS = 8
_MAX_ATTENTION_LOGS = 4
_exec_kv_logs = 0
_attention_logs = 0


def nano_tail_debug_enabled() -> bool:
    try:
        return bool(envs.VLLM_ASCEND_NANO_TAIL_DEBUG)
    except Exception:
        return False


def reset_nano_tail_debug_counters() -> None:
    global _exec_kv_logs, _attention_logs
    _exec_kv_logs = 0
    _attention_logs = 0


def emit_nano_tail_debug(event: str, **fields: Any) -> None:
    try:
        if not envs.VLLM_ASCEND_NANO_TAIL_DEBUG:
            return
        payload = {"event": event, **fields}
        logger.info("%s %s", NANO_TAIL_DEBUG_PREFIX, json.dumps(payload, separators=(",", ":")))
    except Exception:
        pass


def emit_nano_exec_kv_ring(
    *,
    layer_name: str,
    layer_id: int,
    device_slots,
    token_active,
    hot_tokens: int,
) -> None:
    """Log the first few decode ring writes on the first offload layer."""
    global _exec_kv_logs
    try:
        if not envs.VLLM_ASCEND_NANO_TAIL_DEBUG:
            return
        if layer_id != 0 or _exec_kv_logs >= _MAX_EXEC_KV_LOGS:
            return
        row_tokens = hot_tokens + NANO_RING_TOKENS
        n = min(int(device_slots.numel()), 8)
        slots = [int(x) for x in device_slots[:n].detach().cpu().tolist()]
        active = [bool(x) for x in token_active[:n].detach().cpu().tolist()] if token_active is not None else []
        ring = [slot % row_tokens for slot in slots]
        payload = {
            "event": "exec_kv_ring",
            "layer": layer_name,
            "layer_id": int(layer_id),
            "hot_tokens": int(hot_tokens),
            "row_tokens": int(row_tokens),
            "active_tokens": int(token_active[: device_slots.numel()].sum().item()) if token_active is not None else n,
            "device_slots": slots,
            "ring_offsets": ring,
            "in_circular_tail": [hot_tokens <= offset < row_tokens for offset in ring],
            "token_active": active,
            "seq": _exec_kv_logs,
        }
        logger.info("%s %s", NANO_TAIL_DEBUG_PREFIX, json.dumps(payload, separators=(",", ":")))
        _exec_kv_logs += 1
    except Exception:
        pass


def emit_nano_attention_restore(*, layer_name: str, skipped_graph_h2d: bool, reused_indices: bool) -> None:
    global _attention_logs
    try:
        if not envs.VLLM_ASCEND_NANO_TAIL_DEBUG:
            return
        if _attention_logs >= _MAX_ATTENTION_LOGS:
            return
        emit_nano_tail_debug(
            "nano_attention_restore",
            layer=layer_name,
            skipped_graph_h2d=bool(skipped_graph_h2d),
            reused_indices=bool(reused_indices),
            seq=_attention_logs,
        )
        _attention_logs += 1
    except Exception:
        pass
