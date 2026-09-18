# SPDX-License-Identifier: Apache-2.0
"""Optional long-sequence nano tail D2D / ring-write diagnostics.

Enable with ``VLLM_ASCEND_NANO_TAIL_DEBUG=1``. Disabled by default so the
decode hot path stays silent. Failures here must never affect serving.
"""

from __future__ import annotations

import json
import math
from typing import Any

import torch
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
# Tail checks are per step, and divergence can start a few steps in, so keep
# enough budget to watch the ring evolve instead of only its seeded state.
_MAX_TAIL_VERIFY_LOGS = 16
_MAX_RESTORE_DIFF_LOGS = 16
# One span per request per tail page. A decode batch that seeds more than this
# already shows the pattern in the first few.
_MAX_TAIL_VERIFY_SPANS = 4
# When the ring does not match its expected host tokens, look this far either
# side for the tokens it actually holds. Two pages covers an off-by-one block
# or a page-parity mistake.
_TAIL_VERIFY_SEARCH_TOKENS = 256
_exec_kv_logs = 0
_attention_logs = 0
_tail_verify_logs = 0
_restore_diff_logs = 0


def nano_tail_debug_enabled() -> bool:
    try:
        return bool(envs.VLLM_ASCEND_NANO_TAIL_DEBUG)
    except Exception:
        return False


def reset_nano_tail_debug_counters() -> None:
    global _exec_kv_logs, _attention_logs, _tail_verify_logs, _restore_diff_logs
    _exec_kv_logs = 0
    _attention_logs = 0
    _tail_verify_logs = 0
    _restore_diff_logs = 0


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


def _json_float(value: float) -> Any:
    return value if math.isfinite(value) else repr(value)


def _stream_is_capturing() -> bool:
    try:
        capturing = torch.npu.is_current_stream_capturing()
    except Exception:
        return False
    # A mocked torch_npu returns a non-bool sentinel; only a real True skips.
    return capturing is True


def _locate_ring_source(ring_row: torch.Tensor, host: torch.Tensor, expected_token: int) -> int:
    """Return the host token whose KV the ring actually holds, or -1.

    Answers "where did this data come from" when the ring does not match the
    tokens it was seeded with: an off-by-one block, the wrong page or an
    unrelated source all produce distinguishable answers.
    """
    start = max(expected_token - _TAIL_VERIFY_SEARCH_TOKENS, 0)
    stop = min(expected_token + _TAIL_VERIFY_SEARCH_TOKENS, int(host.shape[0]))
    if stop <= start:
        return -1
    window = host[start:stop].detach().to("cpu", torch.float32)
    hits = torch.nonzero((window - ring_row).abs().amax(dim=-1) == 0).flatten()
    return int(hits[0]) + start if hits.numel() else -1


def _compare_tail_span(
    *,
    src_token: int,
    dst_token: int,
    tokens: int,
    host_k: torch.Tensor,
    ring_k: torch.Tensor,
    host_v: torch.Tensor,
    ring_v: torch.Tensor,
) -> dict[str, Any]:
    span: dict[str, Any] = {"src_token": src_token, "dst_token": dst_token, "tokens": tokens}
    for name, host, ring in (("k", host_k, ring_k), ("v", host_v, ring_v)):
        expected = host[src_token : src_token + tokens].detach().to("cpu", torch.float32)
        actual = ring[dst_token : dst_token + tokens].detach().to("cpu", torch.float32)
        if expected.shape != actual.shape:
            span[name] = {
                "error": "shape_mismatch",
                "host": list(expected.shape),
                "ring": list(actual.shape),
            }
            continue
        if expected.numel() == 0:
            span[name] = {"error": "empty_span"}
            continue
        per_token = (expected - actual).abs().amax(dim=-1)
        mismatched = torch.nonzero(per_token > 0).flatten()
        result: dict[str, Any] = {
            "max_abs_diff": _json_float(float(per_token.max())),
            "mismatch_tokens": int(mismatched.numel()),
            "first_mismatch": int(mismatched[0]) if mismatched.numel() else -1,
            "ring_all_zero": bool(actual.abs().max() == 0),
            "host_all_zero": bool(expected.abs().max() == 0),
        }
        if name == "k" and mismatched.numel() and tokens:
            # Reported as an absolute host token so it can be read directly
            # against src_token; -1 means the ring holds no nearby host KV.
            result["ring_source_token"] = _locate_ring_source(actual[0], host, src_token)
        span[name] = result
    return span


def emit_nano_tail_verify(
    *,
    layer_name: str,
    layer_id: int,
    host_k: torch.Tensor,
    ring_k: torch.Tensor,
    host_v: torch.Tensor,
    ring_v: torch.Tensor,
    tail_src: torch.Tensor,
    tail_dst: torch.Tensor,
    tail_lengths: torch.Tensor,
    tp_rank: int,
) -> None:
    """Check the PD-seeded circular tail against the host KV it should mirror.

    The tail descriptors describe exactly what the graph H2D restore would
    copy, so comparing them isolates a bad D2D payload from a ring that is
    merely not refreshed. Call before any restore, otherwise the restore
    makes the two sides equal by construction.

    Only tp_rank 0 owns the host pool. Other ranks hold a raw-pointer view of
    it that is valid for DMA address translation but not mapped for CPU loads,
    so dereferencing it there faults. Use ``probe_nano_tail_restore`` for those
    ranks. See ``SparseKVOffloadManager._restore_bfloat16_tensor``.
    """
    global _tail_verify_logs
    try:
        if not envs.VLLM_ASCEND_NANO_TAIL_DEBUG:
            return
        if layer_id != 0 or _tail_verify_logs >= _MAX_TAIL_VERIFY_LOGS:
            return
        if int(tp_rank) != 0:
            return
        if _stream_is_capturing():
            return
        lengths = [int(x) for x in tail_lengths.reshape(-1).tolist()]
        sources = [int(x) for x in tail_src.reshape(-1).tolist()]
        destinations = [int(x) for x in tail_dst.reshape(-1).tolist()]
        spans = []
        for index, tokens in enumerate(lengths):
            # Padding rows and 128-aligned prefixes carry no tail to check.
            if tokens <= 0:
                continue
            spans.append(
                _compare_tail_span(
                    src_token=sources[index],
                    dst_token=destinations[index],
                    tokens=tokens,
                    host_k=host_k,
                    ring_k=ring_k,
                    host_v=host_v,
                    ring_v=ring_v,
                )
            )
            if len(spans) >= _MAX_TAIL_VERIFY_SPANS:
                break
        if not spans:
            return
        payload = {
            "event": "tail_verify",
            "layer": layer_name,
            "layer_id": int(layer_id),
            "spans": spans,
            "seq": _tail_verify_logs,
        }
        logger.info("%s %s", NANO_TAIL_DEBUG_PREFIX, json.dumps(payload, separators=(",", ":")))
        _tail_verify_logs += 1
    except Exception:
        pass


def _restore_diff_summary(d2d: torch.Tensor, restored: torch.Tensor) -> dict[str, Any]:
    if d2d.shape != restored.shape:
        return {"error": "shape_mismatch", "d2d": list(d2d.shape), "restored": list(restored.shape)}
    if d2d.numel() == 0:
        return {"error": "empty_span"}
    left = d2d.to(torch.float32)
    right = restored.to(torch.float32)
    per_token = (right - left).abs().amax(dim=-1)
    # One device-to-host copy per span component: the per-token diffs plus the
    # two all-zero probes. Keeps the probe off the per-element sync path.
    probes = torch.cat([per_token.reshape(-1), left.abs().amax().reshape(1), right.abs().amax().reshape(1)])
    values = probes.detach().to("cpu")
    diffs = values[:-2]
    mismatched = torch.nonzero(diffs > 0).flatten()
    return {
        "max_abs_diff": _json_float(float(diffs.max())),
        "mismatch_tokens": int(mismatched.numel()),
        "first_mismatch": int(mismatched[0]) if mismatched.numel() else -1,
        "d2d_all_zero": bool(values[-2] == 0),
        "restored_all_zero": bool(values[-1] == 0),
    }


def probe_nano_tail_restore(
    *,
    layer_name: str,
    layer_id: int,
    ring_k: torch.Tensor,
    ring_v: torch.Tensor,
    tail_dst: torch.Tensor,
    tail_lengths: torch.Tensor,
    restore,
) -> None:
    """Measure what the skipped H2D restore would have changed in the ring.

    Works on every TP rank because it reads the host pool through the same
    DMA path the restore uses instead of CPU loads. The pre-restore payload is
    written back afterwards so the model still consumes the D2D-seeded tail and
    the accuracy under test is unchanged.
    """
    global _restore_diff_logs
    try:
        if not envs.VLLM_ASCEND_NANO_TAIL_DEBUG:
            return
        if layer_id != 0 or _restore_diff_logs >= _MAX_RESTORE_DIFF_LOGS:
            return
        if _stream_is_capturing():
            return
        lengths = [int(x) for x in tail_lengths.reshape(-1).tolist()]
        destinations = [int(x) for x in tail_dst.reshape(-1).tolist()]
        # Snapshot every live span, not just the logged ones, so the write-back
        # covers everything the restore below touches.
        snapshots = [
            (dst, tokens, ring_k[dst : dst + tokens].clone(), ring_v[dst : dst + tokens].clone())
            for dst, tokens in zip(destinations, lengths)
            if tokens > 0
        ]
        if not snapshots:
            return
        restore()
        spans = [
            {
                "dst_token": dst,
                "tokens": tokens,
                "k": _restore_diff_summary(snap_k, ring_k[dst : dst + tokens]),
                "v": _restore_diff_summary(snap_v, ring_v[dst : dst + tokens]),
            }
            for dst, tokens, snap_k, snap_v in snapshots[:_MAX_TAIL_VERIFY_SPANS]
        ]
        for dst, tokens, snap_k, snap_v in snapshots:
            ring_k[dst : dst + tokens].copy_(snap_k)
            ring_v[dst : dst + tokens].copy_(snap_v)
        payload = {
            "event": "tail_restore_diff",
            "layer": layer_name,
            "layer_id": int(layer_id),
            "spans": spans,
            "seq": _restore_diff_logs,
        }
        logger.info("%s %s", NANO_TAIL_DEBUG_PREFIX, json.dumps(payload, separators=(",", ":")))
        _restore_diff_logs += 1
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
