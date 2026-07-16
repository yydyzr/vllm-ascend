#!/usr/bin/env python3
"""Compare SFA decode operator dumps.

Usage:
  # dump selected decode steps (eager + graph share this env):
  #   export VLLM_ASCEND_SFA_DUMP_DIR=/tmp/sfa-dump
  #   export VLLM_ASCEND_SFA_DUMP_LAYER=0
  #   export VLLM_ASCEND_SFA_DUMP_STEP=0,1,3

  # fused_overlap vs no-offload SFA
  python examples/compare_sfa_decode_dump.py \\
      --fused /tmp/sfa-dump/sfa_fused_inputs_layer0_step1_rank0_pid123.pt \\
      --sfa   /tmp/sfa-dump/sfa_sfa_inputs_layer0_step1_rank0_pid456.pt

  # eager fused vs ACLGraph fused (same op schema / same step)
  python examples/compare_sfa_decode_dump.py \\
      --eager /tmp/sfa-dump/sfa_fused_inputs_layer0_step1_rank0_pid123.pt \\
      --graph /tmp/sfa-dump/sfa_fused_graph_inputs_layer0_step1_rank0_pid123.pt

  # auto-pick newest fused/sfa dumps in a directory
  python examples/compare_sfa_decode_dump.py --dump-dir /tmp/sfa-dump
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"
SKIP = "SKIP"


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str


def load_input_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "inputs" not in payload:
        raise ValueError(f"{path} is not an SFA op-input dump (missing 'inputs')")
    return payload


def load_output_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "output" not in payload:
        raise ValueError(f"{path} is not an SFA op-output dump (missing 'output')")
    return payload


def find_latest_input_dump(dump_dir: Path, mode: str) -> Path:
    matches = sorted(
        dump_dir.glob(f"sfa_{mode}_inputs_*.pt"),
        key=lambda p: p.stat().st_mtime,
    )
    if not matches:
        raise FileNotFoundError(f"no sfa_{mode}_inputs_*.pt found under {dump_dir}")
    return matches[-1]


def infer_output_path(input_path: Path) -> Path:
    return input_path.with_name(input_path.name.replace("_inputs_", "_output_", 1))


def _tensor_info(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return f"tensor shape={tuple(value.shape)} dtype={value.dtype}"
    return f"{type(value).__name__}({value!r})"


def compare_tensors(
    name: str,
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> CheckResult:
    del rtol  # retained for call-site compatibility; float compare uses cosine similarity
    if left.shape != right.shape:
        return CheckResult(
            name,
            FAIL,
            f"shape mismatch: left={tuple(left.shape)} right={tuple(right.shape)}",
        )
    if left.dtype != right.dtype:
        left = left.to(torch.float32)
        right = right.to(torch.float32)
    elif not torch.is_floating_point(left):
        if torch.equal(left, right):
            return CheckResult(name, PASS, "exact match")
        diff = (left != right).sum().item()
        return CheckResult(name, FAIL, f"{diff} element(s) differ")
    else:
        left = left.float()
        right = right.float()

    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    left_norm = torch.linalg.vector_norm(left_flat)
    right_norm = torch.linalg.vector_norm(right_flat)
    if left_norm <= 0 or right_norm <= 0:
        if left_norm <= 0 and right_norm <= 0:
            return CheckResult(name, PASS, "cosine_sim=1 (both zero)")
        return CheckResult(
            name,
            FAIL,
            f"cosine_sim undefined: left_norm={left_norm.item():.6g} right_norm={right_norm.item():.6g}",
        )

    cos_sim = torch.dot(left_flat, right_flat) / (left_norm * right_norm)
    cos_sim_value = float(cos_sim.clamp(-1.0, 1.0).item())
    min_cos = 1.0 - atol
    detail = f"cosine_sim={cos_sim_value:.8f} min_cos={min_cos:.8f}"
    if cos_sim_value >= min_cos:
        return CheckResult(name, PASS, detail)
    return CheckResult(name, FAIL, detail)


def compare_scalars(name: str, left: Any, right: Any) -> CheckResult:
    if left == right:
        return CheckResult(name, PASS, f"value={left!r}")
    return CheckResult(name, FAIL, f"left={left!r} right={right!r}")


def compare_same_schema_payloads(
    left_payload: dict[str, Any],
    right_payload: dict[str, Any],
    *,
    atol: float,
    rtol: float,
    left_label: str = "eager",
    right_label: str = "graph",
) -> list[CheckResult]:
    """Compare two fused-style dumps that share the same input keys."""
    left = left_payload["inputs"]
    right = right_payload["inputs"]
    results: list[CheckResult] = [
        CheckResult(
            "payload.mode",
            WARN,
            f"{left_label}.mode={left_payload.get('mode')!r} "
            f"{right_label}.mode={right_payload.get('mode')!r} "
            f"{left_label}.step={left_payload.get('step')!r} "
            f"{right_label}.step={right_payload.get('step')!r}",
        )
    ]
    keys = sorted(set(left) | set(right))
    for key in keys:
        if key not in left:
            results.append(CheckResult(key, FAIL, f"missing in {left_label}"))
            continue
        if key not in right:
            results.append(CheckResult(key, FAIL, f"missing in {right_label}"))
            continue
        lv, rv = left[key], right[key]
        if isinstance(lv, torch.Tensor) and isinstance(rv, torch.Tensor):
            results.append(compare_tensors(key, lv, rv, atol=atol, rtol=rtol))
        else:
            results.append(compare_scalars(key, lv, rv))
    return results


def build_sfa_query(sfa_inputs: dict[str, Any]) -> torch.Tensor:
    return torch.cat(
        [sfa_inputs["query"], sfa_inputs["query_rope"]],
        dim=-1,
    ).contiguous()


def infer_block_size(cache: torch.Tensor) -> int:
    if cache.dim() == 3:
        return cache.shape[1]
    if cache.dim() == 4:
        return cache.shape[1]
    raise ValueError(f"unsupported paged cache shape: {tuple(cache.shape)}")


def gather_selected_paged_cache(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    block_size: int,
    token_to_req: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather topk-selected KV rows from a paged cache."""
    if cache.dim() == 4:
        cache = cache.reshape(cache.shape[0], cache.shape[1], -1)
    flat_cache = cache
    topk = topk_indices.to(torch.long)
    table = block_table.to(torch.long)

    if token_to_req is None:
        token_to_req = torch.arange(topk.shape[0], dtype=torch.long)
    else:
        token_to_req = token_to_req.to(torch.long)

    valid = topk >= 0
    safe_topk = torch.where(valid, topk, torch.zeros_like(topk))
    logical_blocks = torch.div(safe_topk, block_size, rounding_mode="floor")
    offsets = torch.remainder(safe_topk, block_size)

    per_token_table = table.index_select(0, token_to_req)
    expanded_table = per_token_table.unsqueeze(1).expand(-1, topk.shape[1], -1)
    physical_blocks = torch.gather(expanded_table, 2, logical_blocks)
    linear_indices = physical_blocks * block_size + offsets

    values = flat_cache.reshape(-1, flat_cache.shape[-1]).index_select(
        0,
        linear_indices.reshape(-1),
    )
    values = values.reshape(*linear_indices.shape, flat_cache.shape[-1])
    values = values.masked_fill_(~valid.unsqueeze(-1), 0)
    return values


def compare_payloads(
    fused_payload: dict[str, Any],
    sfa_payload: dict[str, Any],
    *,
    atol: float,
    rtol: float,
) -> list[CheckResult]:
    fused = fused_payload["inputs"]
    sfa = sfa_payload["inputs"]
    results: list[CheckResult] = []

    if fused_payload.get("mode") != "fused":
        results.append(
            CheckResult("payload.mode", WARN, f"expected fused, got {fused_payload.get('mode')!r}")
        )
    if sfa_payload.get("mode") != "sfa":
        results.append(
            CheckResult("payload.mode", WARN, f"expected sfa, got {sfa_payload.get('mode')!r}")
        )

    results.append(
        compare_tensors(
            "query",
            fused["query"],
            build_sfa_query(sfa),
            atol=atol,
            rtol=rtol,
        )
    )
    results.append(
        compare_tensors(
            "topk_indices",
            fused["selection_topk_indices"].to(torch.long),
            sfa["sparse_indices"].to(torch.long),
            atol=atol,
            rtol=rtol,
        )
    )
    results.append(
        compare_tensors(
            "actual_seq_lengths_query",
            fused["full_q_actual_seq"].to(torch.long),
            sfa["actual_seq_lengths_query"].to(torch.long),
            atol=atol,
            rtol=rtol,
        )
    )
    results.append(
        compare_tensors(
            "actual_seq_lengths_kv",
            fused["full_kv_actual_seq"].to(torch.long),
            sfa["actual_seq_lengths_kv"].to(torch.long),
            atol=atol,
            rtol=rtol,
        )
    )

    for name in ("scale_value", "sparse_block_size", "layout_query", "layout_kv", "sparse_mode"):
        if name in fused and name in sfa:
            results.append(compare_scalars(name, fused[name], sfa[name]))

    results.append(
        CheckResult(
            "block_table",
            WARN,
            "skipped direct compare: fused uses CPU full_kv_block_table, "
            f"sfa uses NPU block_table ({_tensor_info(fused['full_kv_block_table'])} vs "
            f"{_tensor_info(sfa['block_table'])})",
        )
    )

    # Compare KV values selected by topk. Fused full_kv_cache is CPU; sfa key is NPU paged.
    try:
        fused_block_size = infer_block_size(fused["full_kv_cache"])
        sfa_block_size = infer_block_size(sfa["key"])
        if fused_block_size != sfa_block_size:
            results.append(
                CheckResult(
                    "selected_kv.block_size",
                    WARN,
                    f"fused={fused_block_size} sfa={sfa_block_size}",
                )
            )
        block_size = fused_block_size
        fused_selected_kv = gather_selected_paged_cache(
            fused["full_kv_cache"],
            fused["full_kv_block_table"],
            fused["selection_topk_indices"],
            block_size=block_size,
        )
        sfa_selected_kv = gather_selected_paged_cache(
            sfa["key"],
            sfa["block_table"],
            sfa["sparse_indices"],
            block_size=block_size,
        )
        results.append(
            compare_tensors(
                "selected_kv_from_full_cache",
                fused_selected_kv,
                sfa_selected_kv,
                atol=atol,
                rtol=rtol,
            )
        )
        fused_selected_rope = gather_selected_paged_cache(
            fused["full_k_rope"],
            fused["full_kv_block_table"],
            fused["selection_topk_indices"],
            block_size=block_size,
        )
        sfa_selected_rope = gather_selected_paged_cache(
            sfa["key_rope"],
            sfa["block_table"],
            sfa["sparse_indices"],
            block_size=block_size,
        )
        results.append(
            compare_tensors(
                "selected_rope_from_full_cache",
                fused_selected_rope,
                sfa_selected_rope,
                atol=atol,
                rtol=rtol,
            )
        )
    except Exception as exc:
        results.append(CheckResult("selected_kv", SKIP, f"compare skipped: {exc}"))

    results.append(
        CheckResult(
            "selection_buffer",
            WARN,
            "skipped: fused has selection_kv_cache/selection_k_rope buffers, "
            f"sfa uses full paged key directly ({_tensor_info(fused.get('selection_kv_cache'))} vs "
            f"{_tensor_info(sfa.get('key'))})",
        )
    )
    return results


def compare_output_payloads(
    fused_payload: dict[str, Any],
    sfa_payload: dict[str, Any],
    *,
    atol: float,
    rtol: float,
) -> list[CheckResult]:
    results: list[CheckResult] = []
    if fused_payload.get("mode") != "fused":
        results.append(
            CheckResult("output_payload.mode", WARN, f"expected fused, got {fused_payload.get('mode')!r}")
        )
    if sfa_payload.get("mode") != "sfa":
        results.append(
            CheckResult("output_payload.mode", WARN, f"expected sfa, got {sfa_payload.get('mode')!r}")
        )
    results.append(
        compare_tensors(
            "attention_output",
            fused_payload["output"],
            sfa_payload["output"],
            atol=atol,
            rtol=rtol,
        )
    )
    return results


def print_report(
    fused_path: Path,
    sfa_path: Path,
    fused_output_path: Path | None,
    sfa_output_path: Path | None,
    fused_payload: dict[str, Any],
    sfa_payload: dict[str, Any],
    results: list[CheckResult],
) -> int:
    print("=" * 72)
    print("SFA decode dump comparison")
    print("=" * 72)
    print(f"fused: {fused_path}")
    print(f"  layer={fused_payload.get('layer_name')} op={fused_payload.get('op_name')}")
    print(f"sfa:   {sfa_path}")
    print(f"  layer={sfa_payload.get('layer_name')} op={sfa_payload.get('op_name')}")
    if fused_output_path is not None:
        print(f"fused output: {fused_output_path}")
    if sfa_output_path is not None:
        print(f"sfa output:   {sfa_output_path}")
    print("-" * 72)

    counts = {PASS: 0, FAIL: 0, WARN: 0, SKIP: 0}
    for item in results:
        counts[item.status] = counts.get(item.status, 0) + 1
        print(f"[{item.status:4}] {item.name}: {item.detail}")

    print("-" * 72)
    print(
        "summary: "
        f"pass={counts[PASS]} fail={counts[FAIL]} warn={counts[WARN]} skip={counts[SKIP]}"
    )
    return 1 if counts[FAIL] else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fused", type=Path, help="path to sfa_fused_inputs_*.pt")
    parser.add_argument("--sfa", type=Path, help="path to sfa_sfa_inputs_*.pt")
    parser.add_argument("--eager", type=Path, help="path to eager sfa_fused_inputs_*.pt")
    parser.add_argument(
        "--graph",
        type=Path,
        help="path to ACLGraph sfa_fused_graph_inputs_layer*_step*_rank*_pid*.pt",
    )
    parser.add_argument("--fused-output", type=Path, help="path to sfa_fused_output_*.pt")
    parser.add_argument("--sfa-output", type=Path, help="path to sfa_sfa_output_*.pt")
    parser.add_argument(
        "--dump-dir",
        type=Path,
        help="directory containing dump files; picks newest fused/sfa dumps",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-3,
        help="float tensors pass when cosine_sim >= 1 - atol (default: 1e-3 => 0.999)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="kept for CLI compatibility; unused by cosine similarity compare",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.eager is not None and args.graph is not None:
        eager_payload = load_input_payload(args.eager)
        graph_payload = load_input_payload(args.graph)
        results = compare_same_schema_payloads(
            eager_payload,
            graph_payload,
            atol=args.atol,
            rtol=args.rtol,
            left_label="eager",
            right_label="graph",
        )
        eager_out = infer_output_path(args.eager)
        graph_out = infer_output_path(args.graph)
        if eager_out.is_file() and graph_out.is_file():
            results.extend(
                compare_output_payloads(
                    load_output_payload(eager_out),
                    load_output_payload(graph_out),
                    atol=args.atol,
                    rtol=args.rtol,
                )
            )
        else:
            results.append(
                CheckResult(
                    "attention_output",
                    WARN,
                    f"output dump missing: {[str(p) for p in (eager_out, graph_out) if not p.is_file()]}",
                )
            )
        return print_report(
            args.eager,
            args.graph,
            eager_out if eager_out.is_file() else None,
            graph_out if graph_out.is_file() else None,
            eager_payload,
            graph_payload,
            results,
        )

    if args.dump_dir is not None:
        fused_path = find_latest_input_dump(args.dump_dir, "fused")
        sfa_path = find_latest_input_dump(args.dump_dir, "sfa")
    elif args.fused is not None and args.sfa is not None:
        fused_path = args.fused
        sfa_path = args.sfa
    else:
        print(
            "error: provide --eager/--graph, --fused/--sfa, or --dump-dir",
            file=sys.stderr,
        )
        return 2

    fused_output_path = args.fused_output or infer_output_path(fused_path)
    sfa_output_path = args.sfa_output or infer_output_path(sfa_path)
    fused_payload = load_input_payload(fused_path)
    sfa_payload = load_input_payload(sfa_path)
    results = compare_payloads(
        fused_payload,
        sfa_payload,
        atol=args.atol,
        rtol=args.rtol,
    )
    missing_outputs = [
        path
        for path in (fused_output_path, sfa_output_path)
        if not path.is_file()
    ]
    if missing_outputs:
        results.append(
            CheckResult(
                "attention_output",
                WARN,
                f"output dump missing: {', '.join(str(path) for path in missing_outputs)}",
            )
        )
        fused_output_path = None
        sfa_output_path = None
    else:
        results.extend(
            compare_output_payloads(
                load_output_payload(fused_output_path),
                load_output_payload(sfa_output_path),
                atol=args.atol,
                rtol=args.rtol,
            )
        )
    return print_report(
        fused_path,
        sfa_path,
        fused_output_path,
        sfa_output_path,
        fused_payload,
        sfa_payload,
        results,
    )


if __name__ == "__main__":
    raise SystemExit(main())
