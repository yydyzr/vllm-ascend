#!/usr/bin/env python3
"""Replay a fused_overlap input dump and compare against baseline SFA output.

Steps:
  1) Align fused vs baseline *inputs* (same checks as compare_sfa_decode_dump.py)
  2) If inputs align, run fused op on NPU and compare output to baseline dump

Requires NPU + registered ``npu_fused_sparse_attention_overlap``.

Usage:
  python examples/replay_sfa_fused_dump.py \\
      --fused /tmp/sfa-dump/sfa_fused_inputs_layer0_step1_rank0_pid123.pt \\
      --sfa /tmp/sfa-dump/sfa_sfa_inputs_layer0_step1_rank0_pid456.pt \\
      --baseline /tmp/sfa-dump/sfa_sfa_output_layer0_step1_rank0_pid456.pt
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import compare_sfa_decode_dump as dump_cmp  # noqa: E402

_LAYER_STEP_RE = re.compile(r"layer(?P<layer>\d+)_step(?P<step>\d+)")


def _npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def _get_fused_op():
    for namespace_name in ("_C_ascend", "custom"):
        namespace = getattr(torch.ops, namespace_name, None)
        if namespace is None:
            continue
        op = getattr(namespace, "npu_fused_sparse_attention_overlap", None)
        if op is not None:
            return op
    torch_npu = getattr(torch, "npu", None)
    if torch_npu is not None:
        op = getattr(torch_npu, "npu_fused_sparse_attention_overlap", None)
        if op is not None:
            return op
    return None


def _infer_layer_step(path: Path) -> tuple[int | None, int | None]:
    match = _LAYER_STEP_RE.search(path.name)
    if match is None:
        return None, None
    return int(match.group("layer")), int(match.group("step"))


def _cosine_sim(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.detach().float().reshape(-1)
    right = right.detach().float().reshape(-1)
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    if left_norm <= 0 and right_norm <= 0:
        return 1.0
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return float(
        (torch.dot(left, right) / (left_norm * right_norm)).clamp(-1.0, 1.0).item()
    )


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=False).contiguous()
    return value


def _prepare_fused_kwargs(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Mirror production: query/selection on NPU, full KV stays on CPU."""
    npu_keys = {
        "query",
        "selection_k_rope",
        "selection_kv_cache",
        "selection_kv_block_table",
        "selection_kv_block_status",
        "selection_topk_indices",
        "full_kv_block_table",
        "full_kv_actual_seq",
        "full_q_actual_seq",
    }
    cpu_keys = {"full_k_rope", "full_kv_cache"}
    kwargs: dict[str, Any] = {}
    for name, value in inputs.items():
        if name in npu_keys:
            tensor = _to_device(value, device)
            if name.startswith("selection_") and isinstance(tensor, torch.Tensor):
                tensor = tensor.clone()
            kwargs[name] = tensor
        elif name in cpu_keys:
            kwargs[name] = value.contiguous() if isinstance(value, torch.Tensor) else value
        else:
            kwargs[name] = value
    return kwargs


def _print_input_align_report(
    fused_path: Path,
    sfa_path: Path,
    fused_payload: dict[str, Any],
    sfa_payload: dict[str, Any],
    results: list[dump_cmp.CheckResult],
) -> int:
    print("=" * 72)
    print("Step 1/2: fused vs baseline input alignment")
    print("=" * 72)
    print(f"fused inputs: {fused_path}")
    print(f"  layer={fused_payload.get('layer_name')} op={fused_payload.get('op_name')}")
    print(f"sfa inputs:   {sfa_path}")
    print(f"  layer={sfa_payload.get('layer_name')} op={sfa_payload.get('op_name')}")
    print("-" * 72)

    counts = {
        dump_cmp.PASS: 0,
        dump_cmp.FAIL: 0,
        dump_cmp.WARN: 0,
        dump_cmp.SKIP: 0,
    }
    for item in results:
        counts[item.status] = counts.get(item.status, 0) + 1
        print(f"[{item.status:4}] {item.name}: {item.detail}")

    print("-" * 72)
    print(
        "input summary: "
        f"pass={counts[dump_cmp.PASS]} fail={counts[dump_cmp.FAIL]} "
        f"warn={counts[dump_cmp.WARN]} skip={counts[dump_cmp.SKIP]}"
    )
    return 1 if counts[dump_cmp.FAIL] else 0


def run_fused_vs_baseline_output(
    fused_op,
    fused_inputs_path: Path,
    baseline_output_path: Path,
    *,
    atol: float,
) -> tuple[bool, str]:
    fused_payload = dump_cmp.load_input_payload(fused_inputs_path)
    baseline_payload = dump_cmp.load_output_payload(baseline_output_path)

    golden = baseline_payload["output"]
    if not isinstance(golden, torch.Tensor):
        return False, "baseline output is not a tensor"

    layer, step = _infer_layer_step(fused_inputs_path)
    layer_tag = f"layer={layer}" if layer is not None else "layer=?"
    step_tag = f"step={step}" if step is not None else "step=?"

    device = torch.device("npu")
    kwargs = _prepare_fused_kwargs(fused_payload["inputs"], device)
    attn_output = fused_op(**kwargs)
    attn_output = attn_output[..., : golden.shape[-1]].contiguous().cpu()

    if tuple(attn_output.shape) != tuple(golden.shape):
        return (
            False,
            f"FAIL {layer_tag} {step_tag}: shape fused={tuple(attn_output.shape)} "
            f"baseline={tuple(golden.shape)}",
        )

    cos = _cosine_sim(attn_output, golden)
    detail = (
        f"{layer_tag} {step_tag} cosine_sim={cos:.8f} "
        f"fused={fused_inputs_path.name} baseline={baseline_output_path.name}"
    )
    if cos >= 1.0 - atol:
        return True, f"PASS {detail}"
    return False, f"FAIL {detail}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Align fused/baseline input dumps, then replay fused op and compare "
            "to baseline output."
        )
    )
    parser.add_argument(
        "--fused",
        type=Path,
        required=True,
        help="Fused op input dump (.pt), e.g. sfa_fused_inputs_*.pt",
    )
    parser.add_argument(
        "--sfa",
        type=Path,
        required=True,
        help="Baseline (no-offload) SFA input dump (.pt), e.g. sfa_sfa_inputs_*.pt",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help=(
            "Baseline SFA output dump (.pt), e.g. sfa_sfa_output_*.pt. "
            "Default: infer from --sfa by replacing _inputs_ with _output_."
        ),
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-3,
        help="Allowed cosine distance (pass if cosine_sim >= 1-atol). Default: 1e-3.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Kept for compatibility with compare_sfa_decode_dump.py; unused by cosine compare.",
    )
    parser.add_argument(
        "--skip-input-check",
        action="store_true",
        help="Skip fused vs baseline input alignment and only replay/compare outputs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.fused.is_file():
        print(f"ERROR: --fused is not a file: {args.fused}", file=sys.stderr)
        return 2
    if not args.sfa.is_file():
        print(f"ERROR: --sfa is not a file: {args.sfa}", file=sys.stderr)
        return 2

    baseline_output = args.baseline or dump_cmp.infer_output_path(args.sfa)
    if not baseline_output.is_file():
        print(f"ERROR: baseline output dump not found: {baseline_output}", file=sys.stderr)
        print("       pass --baseline explicitly if the filename differs", file=sys.stderr)
        return 2

    fused_payload = dump_cmp.load_input_payload(args.fused)
    sfa_payload = dump_cmp.load_input_payload(args.sfa)

    if not args.skip_input_check:
        input_results = dump_cmp.compare_payloads(
            fused_payload,
            sfa_payload,
            atol=args.atol,
            rtol=args.rtol,
        )
        input_rc = _print_input_align_report(
            args.fused,
            args.sfa,
            fused_payload,
            sfa_payload,
            input_results,
        )
        if input_rc != 0:
            print("ABORT: inputs are not aligned; skip fused op replay")
            return 1
        print("Inputs aligned; continue to fused op replay")
    else:
        print("Skip input alignment (--skip-input-check)")

    if not _npu_available():
        print("ERROR: NPU is not available", file=sys.stderr)
        return 2

    fused_op = _get_fused_op()
    if fused_op is None:
        print(
            "ERROR: npu_fused_sparse_attention_overlap is not registered "
            "(import/install vllm_ascend custom ops first)",
            file=sys.stderr,
        )
        return 2

    print("=" * 72)
    print("Step 2/2: replay fused op vs baseline output")
    print("=" * 72)
    ok, message = run_fused_vs_baseline_output(
        fused_op,
        args.fused,
        baseline_output,
        atol=args.atol,
    )
    print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
