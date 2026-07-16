from __future__ import annotations

import os
from collections.abc import Collection
from pathlib import Path
from typing import Any

import torch

try:
    import torch_npu
except ImportError:  # pragma: no cover
    torch_npu = None

from vllm_ascend.attention.npu_host_func_stream import ensure_host_func_stream_subscribed
_graph_inputs_step_count = 0
_graph_output_step_count = 0


def dump_op_inputs(
    dump_dir: str | os.PathLike[str],
    *,
    mode: str,
    layer_name: str,
    layer_id: int,
    op_name: str,
    inputs: dict[str, Any],
    rank: int,
    pid: int,
    step: int = 0,
) -> Path:
    payload = {
        "mode": mode,
        "layer_name": layer_name,
        "op_name": op_name,
        "step": step,
        "inputs": {
            name: (value.detach().cpu() if isinstance(value, torch.Tensor) else value)
            for name, value in inputs.items()
        },
    }
    output_dir = Path(dump_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / (
        f"sfa_{mode}_inputs_layer{layer_id}_step{step}_rank{rank}_pid{pid}.pt"
    )
    temporary = output.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    return output


def dump_op_output(
    dump_dir: str | os.PathLike[str],
    *,
    mode: str,
    layer_name: str,
    layer_id: int,
    op_name: str,
    output: torch.Tensor,
    rank: int,
    pid: int,
    step: int = 0,
) -> Path:
    payload = {
        "mode": mode,
        "layer_name": layer_name,
        "op_name": op_name,
        "step": step,
        "output": output.detach().cpu(),
    }
    output_dir = Path(dump_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"sfa_{mode}_output_layer{layer_id}_step{step}_rank{rank}_pid{pid}.pt"
    )
    temporary = output_path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output_path)
    return output_path


def _stage_value_for_host_dump(value: Any) -> Any:
    """Enqueue NPU→CPU copies on the current stream; leave CPU/scalars as-is."""
    if not isinstance(value, torch.Tensor):
        return value
    if value.device.type != "npu":
        return value
    try:
        host = torch.empty(value.shape, dtype=value.dtype, device="cpu", pin_memory=True)
    except RuntimeError:
        host = torch.empty(value.shape, dtype=value.dtype, device="cpu")
    host.copy_(value, non_blocking=True)
    return host


def stage_op_inputs_for_host_dump(inputs: dict[str, Any]) -> dict[str, Any]:
    return {name: _stage_value_for_host_dump(value) for name, value in inputs.items()}


def _materialize_host_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Snapshot tensors at host_func time (after prior stream ops / D2H)."""
    out: dict[str, Any] = {}
    for name, value in inputs.items():
        if isinstance(value, torch.Tensor):
            out[name] = value.detach().cpu().contiguous().clone()
        else:
            out[name] = value
    return out


def _normalize_dump_steps(dump_steps: Collection[int] | None) -> frozenset[int]:
    if not dump_steps:
        return frozenset({0})
    return frozenset(int(s) for s in dump_steps)


def _is_stream_capturing() -> bool:
    """True during ACLGraph capture; False on replay and eager."""
    if torch_npu is None:
        return False
    for npu_runtime in (getattr(torch_npu, "npu", None), getattr(torch, "npu", None)):
        if npu_runtime is None:
            continue
        for attr_name in ("is_current_stream_capturing", "_is_current_stream_capturing"):
            capture_state = getattr(npu_runtime, attr_name, None)
            if not callable(capture_state):
                continue
            try:
                if bool(capture_state()):
                    return True
            except Exception:
                continue
    return False


def _dump_fused_inputs_host_callback(args: tuple) -> None:
    """ACLGraph host callback: CPU-only save. No NPU ops / logger."""
    global _graph_inputs_step_count
    # Skip capture-time host_func runs (warmup / multi-graph capture). Only
    # count decode replays so DUMP_STEP aligns with eager.
    if _is_stream_capturing():
        return
    (
        dump_dir,
        mode,
        layer_name,
        layer_id,
        op_name,
        rank,
        pid,
        dump_steps,
        staged_inputs,
    ) = args
    step = _graph_inputs_step_count
    _graph_inputs_step_count += 1
    steps = _normalize_dump_steps(dump_steps)
    if step not in steps:
        return
    payload_inputs = _materialize_host_inputs(staged_inputs)
    dump_op_inputs(
        dump_dir,
        mode=mode,
        layer_name=layer_name,
        layer_id=layer_id,
        op_name=op_name,
        inputs=payload_inputs,
        rank=rank,
        pid=pid,
        step=step,
    )


def _dump_fused_output_host_callback(args: tuple) -> None:
    """ACLGraph host callback: CPU-only save of fused output. No NPU ops / logger."""
    global _graph_output_step_count
    if _is_stream_capturing():
        return
    (
        dump_dir,
        mode,
        layer_name,
        layer_id,
        op_name,
        rank,
        pid,
        dump_steps,
        staged_output,
    ) = args
    step = _graph_output_step_count
    _graph_output_step_count += 1
    steps = _normalize_dump_steps(dump_steps)
    if step not in steps:
        return
    if not isinstance(staged_output, torch.Tensor):
        return
    output = staged_output.detach().cpu().contiguous().clone()
    dump_op_output(
        dump_dir,
        mode=mode,
        layer_name=layer_name,
        layer_id=layer_id,
        op_name=op_name,
        output=output,
        rank=rank,
        pid=pid,
        step=step,
    )


def launch_graph_fused_inputs_dump(
    *,
    dump_dir: str,
    mode: str,
    layer_name: str,
    layer_id: int,
    op_name: str,
    inputs: dict[str, Any],
    rank: int,
    pid: int,
    dump_steps: Collection[int],
    stream: Any | None = None,
) -> None:
    """Record a graph-safe fused-inputs dump on ``stream`` (default: current).

    Uses the same ``VLLM_ASCEND_SFA_DUMP_STEP`` set as eager: each host_func
    invocation advances a step counter; only matching steps are written.
    """
    if torch_npu is None:
        return
    current = stream if stream is not None else torch_npu.npu.current_stream()
    ensure_host_func_stream_subscribed(current)
    staged = stage_op_inputs_for_host_dump(inputs)
    torch_npu.npu._launch_host_func(
        current,
        _dump_fused_inputs_host_callback,
        (
            dump_dir,
            mode,
            layer_name,
            layer_id,
            op_name,
            rank,
            pid,
            _normalize_dump_steps(dump_steps),
            staged,
        ),
    )


def launch_graph_fused_output_dump(
    *,
    dump_dir: str,
    mode: str,
    layer_name: str,
    layer_id: int,
    op_name: str,
    output: torch.Tensor,
    rank: int,
    pid: int,
    dump_steps: Collection[int],
    stream: Any | None = None,
) -> None:
    if torch_npu is None:
        return
    current = stream if stream is not None else torch_npu.npu.current_stream()
    ensure_host_func_stream_subscribed(current)
    staged = _stage_value_for_host_dump(output)
    torch_npu.npu._launch_host_func(
        current,
        _dump_fused_output_host_callback,
        (
            dump_dir,
            mode,
            layer_name,
            layer_id,
            op_name,
            rank,
            pid,
            _normalize_dump_steps(dump_steps),
            staged,
        ),
    )
