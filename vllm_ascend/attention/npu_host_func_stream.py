"""Shared NPU stream subscription for ``_launch_host_func`` (ACLGraph-safe)."""

from __future__ import annotations

from typing import Any

try:
    import torch_npu
except ImportError:  # pragma: no cover
    torch_npu = None

_SUBSCRIBED_HOST_FUNC_STREAMS: set[Any] = set()


def get_subscribed_host_func_streams() -> set[Any]:
    return _SUBSCRIBED_HOST_FUNC_STREAMS


def ensure_host_func_stream_subscribed(stream: Any) -> None:
    """Subscribe ``stream`` once for host_func; safe across D2H / dump / LRU paths."""
    if torch_npu is None:
        return
    if stream in _SUBSCRIBED_HOST_FUNC_STREAMS:
        return
    torch_npu.npu._subscribe_report(stream)
    _SUBSCRIBED_HOST_FUNC_STREAMS.add(stream)
