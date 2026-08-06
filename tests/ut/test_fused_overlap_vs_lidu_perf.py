"""Perf comparison: vllm-ascend fused_overlap vs nano LI_MANAGE+scatter_sfa.

Two OPP packages cannot coexist in one process, so run one approach per
process (writes JSON), then --compare reads both.

A: LI + host-side LRU planner + fused_overlap(copy+sfa in one kernel).
   LRU does not emit topk; it writes an external plan into
   ``selection_membership_map``. Fused reads that plan plus the same topk
   that was fed into the planner.
B: LightningIndexer baseline (same LI inputs as li_manage) +
   LI_MANAGE(li+lru fused) + scatter_copy_sfa(copy+sfa fused).
   Reports ``li_manage_minus_li_ms`` = li_manage - lightning_indexer.

Common compare fields written into every scenario JSON (profile or not):
  ``lru_ms``      — A: host planner cost; B: li_manage - LI   (event time)
  ``sfa_copy_ms`` — A: fused_overlap (copy+sfa); B: scatter_sfa (event time)

With ``--profile``, also emit kernel/timeline compare fields:
  ``lru_kernel_ms``      — A: HOSTFUNC_CALLBACK + EVENT_WAIT (both parts, sum);
                           B: li_manage_kernel - LI_kernel
  ``sfa_copy_kernel_ms`` — A: fused_overlap kernel; B: scatter_sfa kernel
``--compare`` prints both event and kernel pairs when present.

NOTE: Approach-A ``lru_kernel_ms`` is NEVER hostfunc alone — it is always
``hostfunc_callback_steady_us + event_wait_steady_us`` (converted to ms).

Supports a scenario matrix via multi-value ``--batch-size`` / ``--miss`` /
``--seq-len`` / ``--cache-tokens`` (cartesian product). Each approach writes
a suite JSON; ``--compare`` matches scenarios by
(batch_size, miss, seq_len, cache_tokens, topk_layout, q_heads).

Timing defaults to NPUGraph capture/replay (``--graph-mode``). Use
``--eager`` only for debugging launch/dispatch overhead. Approach A's planner
follows the production side-stream + C++ ``enqueue_lru_*`` host-callback path.

``--miss`` is the only hit/miss control for both A and B (no unused
aliases). Each warmup / timed / profile iteration rebuilds residents with
a freshly randomized hit/miss partition among the same topk set. Approach
A wires production-like planner inputs: ``stable_prefix = ak - aq``,
``visible_seq = ak``, continuing ``req_ids``, and ``--cache-tokens`` as
resident fill (capped by A physical capacity ``TOPK*2-1``).

GLM-5.2 parameters (kv_lora_rank=512, qk_rope_head_dim=64, index_n_heads=32,
index_head_dim=128, index_topk=2048, num_attention_heads=64).
Both sides use the projected MLA formulation: query_nope dim = kv_lora_rank
= 512, rope dim = 64. The fused_overlap adapter splits the concatenated
query (576 = 512 + 64) internally; the tiling's qk_head_dim check (==512)
is on kv_lora_rank, so GLM-5.2 is fully supported.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch_npu  # noqa: F401

BLOCK_SIZE = 128
# GLM-5.2 MLA parameters (from config.json):
#   kv_lora_rank=512, qk_nope_head_dim=192, qk_rope_head_dim=64,
#   qk_head_dim=256, v_head_dim=256, num_attention_heads=64,
#   index_n_heads=32, index_head_dim=128, index_topk=2048.
# The fused_overlap adapter splits the query into query_nope (first
# kv_lora_rank dims) and query_rope (last qk_rope_head_dim dims), so the
# tiling's qk_head_dim check (==512) is on kv_lora_rank, not the total
# query dim. Both sides use kv_lora_rank=512, rope=64, matching GLM-5.2.
KVD = 512          # kv_lora_rank — compressed KV cache dim (also projected nope dim)
KRD = 64           # qk_rope_head_dim — rope dim
INDEX_DIM = 128    # index_head_dim
TOPK = 2048        # index_topk
NUM_ATTENTION_HEADS = 64  # GLM-5.2 num_attention_heads (total across TP)
_GIB = 1024 * 1024 * 1024
_ALIGN = 2 * 1024 * 1024
TOPK_LAYOUTS = ("contiguous", "random")


def _default_nano_path() -> str:
    """Resolve the nano torch_extension directory.

    Priority: $NANO_PATH env > sibling-relative to this UT > hardcoded fallback.
    """
    env = os.environ.get("NANO_PATH")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    sibling = os.path.normpath(
        os.path.join(here, "..", "..", "..",
                     "nanovllm-DSA-offload", "torch_extension"))
    if os.path.isdir(os.path.join(sibling, "ops_overlap")):
        return sibling
    return "/Users/zywszr/Desktop/codes/kv_offload/nanovllm-DSA-offload/torch_extension"


def cdiv(a: int, b: int) -> int:
    return -(a // -b)


def resolve_miss_count(args) -> int:
    """Return the controlled per-row miss count from ``--miss``."""
    miss = int(args.miss)
    if miss < 0 or miss > TOPK:
        raise ValueError(f"--miss must be in [0, {TOPK}], got {miss}")
    return miss


def attach_compare_fields(
    result: dict,
    *,
    lru_ms: float,
    sfa_copy_ms: float,
) -> dict:
    """Attach the A/B-comparable event-time metric pair every scenario exposes.

    ``lru_ms``:      A host planner  ↔  B (li_manage - lightning_indexer)
    ``sfa_copy_ms``: A fused_overlap ↔  B scatter_sfa (both copy+sfa)
    """
    result["lru_ms"] = float(lru_ms)
    result["sfa_copy_ms"] = float(sfa_copy_ms)
    return result


# Substring patterns (lowercased) used to pick msprof rows for kernel compare.
_KERNEL_NAME_PATTERNS = {
    "li": (
        "npu_lightning_indexer",
        "lightning_indexer",
        "lightningindexer",
    ),
    "a_sfa_copy": (
        "npu_fused_sparse_attention_overlap",
        "fused_sparse_attention_overlap",
        "fused_sparse_attention",
    ),
    "b_li_manage": (
        "li_manage_out",
        "li_manage",
    ),
    "b_sfa_copy": (
        "sparse_and_tail_attention_and_scatter_copy",
        "sparse_and_tail_attention",
        "scatter_copy",
    ),
}


def _ops_from_profile_summary(profile_summary: dict | None) -> list[dict]:
    if not profile_summary:
        return []
    ops = profile_summary.get("ops")
    if isinstance(ops, list) and ops:
        return ops
    top = profile_summary.get("top_ops")
    return top if isinstance(top, list) else []


def _match_op_avg_ms(
    ops: list[dict],
    patterns: tuple[str, ...],
    *,
    exclude: tuple[str, ...] = (),
) -> tuple[float | None, str | None]:
    """Return (avg_ms, matched_name) for the best-matching op by total_us."""
    best: dict | None = None
    for op in ops:
        name = str(op.get("name", ""))
        lower = name.lower()
        if exclude and any(tok in lower for tok in exclude):
            continue
        if not any(tok in lower for tok in patterns):
            continue
        if best is None or float(op.get("total_us", 0.0)) > float(
                best.get("total_us", 0.0)):
            best = op
    if best is None or best.get("avg_us") is None:
        return None, None
    return float(best["avg_us"]) / 1000.0, str(best["name"])


def attach_kernel_compare_fields(
    result: dict,
    approach: str,
    *,
    quiet: bool = False,
) -> dict:
    """Derive ``lru_kernel_ms`` / ``sfa_copy_kernel_ms`` from profile_summary.

    A ``lru_kernel_ms`` = HOSTFUNC_CALLBACK + EVENT_WAIT (both required parts).
    A ``sfa_copy_kernel_ms`` = fused_overlap AI-Core kernel.
    B ``lru_kernel_ms`` = li_manage kernel - LI kernel.
    B ``sfa_copy_kernel_ms`` = scatter_sfa AI-Core kernel.
    """
    summary = result.get("profile_summary")
    if not summary:
        return result
    ops = _ops_from_profile_summary(summary)
    matched: dict[str, str | None] = {}

    if approach == "A":
        sfa_ms, sfa_name = _match_op_avg_ms(ops, _KERNEL_NAME_PATTERNS["a_sfa_copy"])
        li_ms, li_name = _match_op_avg_ms(
            ops, _KERNEL_NAME_PATTERNS["li"],
            exclude=_KERNEL_NAME_PATTERNS["b_li_manage"],
        )
        lru = summary.get("lru_planner") or {}
        # Sum of BOTH timeline markers — do not drop EVENT_WAIT.
        host_us = lru.get("hostfunc_callback_steady_us")
        wait_us = lru.get("event_wait_steady_us")
        lru_us = lru.get("lru_planner_steady_us")
        if lru_us is None and host_us is not None and wait_us is not None:
            lru_us = float(host_us) + float(wait_us)
        if lru_us is None:
            lru_us = result.get("lru_planner_steady_us")
        lru_ms = float(lru_us) / 1000.0 if lru_us is not None else None
        if host_us is not None:
            result["hostfunc_callback_kernel_ms"] = float(host_us) / 1000.0
        if wait_us is not None:
            result["event_wait_kernel_ms"] = float(wait_us) / 1000.0
        matched["sfa_copy"] = sfa_name
        matched["li"] = li_name
        matched["lru"] = (
            "HOSTFUNC_CALLBACK+EVENT_WAIT" if lru_ms is not None else None
        )
    else:
        li_ms, li_name = _match_op_avg_ms(
            ops, _KERNEL_NAME_PATTERNS["li"],
            exclude=_KERNEL_NAME_PATTERNS["b_li_manage"],
        )
        manage_ms, manage_name = _match_op_avg_ms(
            ops, _KERNEL_NAME_PATTERNS["b_li_manage"],
            exclude=_KERNEL_NAME_PATTERNS["li"],
        )
        sfa_ms, sfa_name = _match_op_avg_ms(ops, _KERNEL_NAME_PATTERNS["b_sfa_copy"])
        if manage_ms is not None and li_ms is not None:
            lru_ms = manage_ms - li_ms
        else:
            lru_ms = None
        matched["li"] = li_name
        matched["li_manage"] = manage_name
        matched["sfa_copy"] = sfa_name
        matched["lru"] = (
            f"{manage_name} - {li_name}"
            if manage_name and li_name else None
        )
        result["li_kernel_ms"] = li_ms
        result["li_manage_kernel_ms"] = manage_ms

    result["lru_kernel_ms"] = lru_ms
    result["sfa_copy_kernel_ms"] = sfa_ms
    if approach == "A":
        result["li_kernel_ms"] = li_ms
    result["kernel_match_names"] = matched
    if not quiet:
        def _fmt(v):
            return "n/a" if v is None else f"{float(v):.4f}"
        print(
            f"[{approach}-kernel] lru_kernel_ms={_fmt(lru_ms)}  "
            f"sfa_copy_kernel_ms={_fmt(sfa_ms)}  "
            f"matched={matched}",
            flush=True,
        )
    return result


def _assert_tensor_eq(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.shape != expected.shape:
        raise RuntimeError(
            f"{name} shape mismatch: actual={tuple(actual.shape)} "
            f"expected={tuple(expected.shape)}")
    if not torch.equal(actual.cpu(), expected.cpu()):
        raise RuntimeError(f"{name} value mismatch versus expected wired input")


def build_index_scores_and_topk(
    seq_len: int,
    topk_layout: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return CPU scores and the corresponding sorted physical top-k tokens."""
    if topk_layout == "contiguous":
        score_values = torch.arange(seq_len, dtype=torch.int32)
    else:
        score_values = torch.randperm(
            seq_len,
            generator=torch.Generator().manual_seed(seed),
            dtype=torch.int32,
        )
    topk_tokens = torch.argsort(score_values, descending=True)[:TOPK]
    return score_values, torch.sort(topk_tokens).values.to(torch.int32)


def bench_events(runner, warmup, iters, reset=None):
    for _ in range(max(0, warmup)):
        if reset is not None:
            reset()
            torch.npu.synchronize()
        runner()
    torch.npu.synchronize()
    times = []
    for _ in range(iters):
        if reset is not None:
            reset()
            torch.npu.synchronize()
        s = torch.npu.Event(enable_timing=True)
        e = torch.npu.Event(enable_timing=True)
        s.record()
        runner()
        e.record()
        e.synchronize()
        times.append(float(s.elapsed_time(e)))
    return statistics.mean(times)


def capture_npu_graph(runner, *, warmup: int = 3, reset=None):
    """Warm up then capture ``runner`` into an NPUGraph for replay timing.

    ``reset`` (if provided) restores miss/hit state outside the graph so each
    warmup/capture sees the same controlled miss count.
    """
    for _ in range(max(0, warmup)):
        if reset is not None:
            reset()
            torch.npu.synchronize()
        runner()
    if reset is not None:
        reset()
        torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        runner()
    torch.npu.synchronize()
    if reset is not None:
        reset()
        torch.npu.synchronize()
    return graph


def maybe_graph_runner(runner, *, use_graph: bool, warmup: int = 3, reset=None):
    """Return ``(callable, mode_name)``; graph mode wraps ``runner`` in replay."""
    if not use_graph:
        return runner, "eager"
    graph = capture_npu_graph(runner, warmup=warmup, reset=reset)
    return graph.replay, "graph"


def _path_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _newest_ascend_pt_dirs(profile_dir: str) -> list[Path]:
    """Return ``*_ascend_pt`` dirs under profile_dir, newest first."""
    root = Path(profile_dir).resolve()
    if not root.is_dir():
        return []
    dirs = [p for p in root.rglob("*_ascend_pt") if p.is_dir()]
    dirs.sort(key=_path_mtime, reverse=True)
    return dirs


def _csv_rank(path: Path) -> tuple:
    """Prefer newest ASCEND_PROFILER_OUTPUT CSV from the latest ascend_pt run."""
    parts_lower = [part.lower() for part in path.parts]
    under_output = any(part == "ascend_profiler_output" for part in parts_lower)
    under_ascend_pt = any(part.endswith("_ascend_pt") for part in parts_lower)
    return (1 if under_output else 0, 1 if under_ascend_pt else 0, _path_mtime(path))


def _profile_search_root(profile_dir: str) -> Path:
    """Newest ascend_pt tree under profile_dir, else profile_dir itself."""
    root = Path(profile_dir).resolve()
    ascend_pts = _newest_ascend_pt_dirs(str(root))
    if ascend_pts:
        print(f"[profile] using newest ascend_pt dir: {ascend_pts[0]}",
              flush=True)
        return ascend_pts[0]
    return root


def _find_named_artifacts(search_root: Path, names: tuple[str, ...]) -> dict[str, Path]:
    candidates: dict[str, list[Path]] = defaultdict(list)
    for path in search_root.rglob("*"):
        if not path.is_file():
            continue
        if path.name in names:
            candidates[path.name].append(path.resolve())
            continue
        # task_time_*.csv / api_statistic_*.csv style exports
        for name in names:
            stem = name.rsplit(".", 1)[0]
            if path.name.startswith(stem) and path.suffix == Path(name).suffix:
                candidates[name].append(path.resolve())
    found: dict[str, Path] = {}
    for name, paths in candidates.items():
        best = max(paths, key=_csv_rank)
        found[name] = best
        print(f"[profile] selected {name}: {best}", flush=True)
    return found


def _find_profile_artifacts(profile_dir: str) -> dict[str, Path]:
    """Locate summary/trace artifacts for the newest run under ``profile_dir``."""
    search_root = _profile_search_root(profile_dir)
    wanted = (
        "op_statistic.csv",
        "kernel_details.csv",
        "operator_details.csv",
        "api_statistic.csv",
        "task_time.csv",
        "trace_view.json",
    )
    return _find_named_artifacts(search_root, wanted)


def _ensure_profile_artifacts(profile_dir: str) -> dict[str, Path]:
    """Return profile artifacts, calling offline analyse() once if needed."""
    found = _find_profile_artifacts(profile_dir)
    if found:
        return found
    try:
        from torch_npu.profiler.profiler import analyse
    except Exception as exc:
        print(f"[profile] offline analyse unavailable: {exc}", flush=True)
        return found
    print("[profile] summary artifacts not ready; running offline analyse()...",
          flush=True)
    ascend_pts = _newest_ascend_pt_dirs(profile_dir)
    analyse_targets = [str(p) for p in ascend_pts[:1]] or [
        str(Path(profile_dir).resolve())]
    for target in analyse_targets:
        print(f"[profile] analyse target: {target}", flush=True)
        try:
            analyse(target)
        except Exception as exc:
            print(f"[profile] offline analyse failed for {target}: {exc}",
                  flush=True)
    return _find_profile_artifacts(profile_dir)


def _pick_csv_field(row: dict, *candidates: str) -> str | None:
    for name in candidates:
        if name in row and row[name] not in ("", None):
            return name
    lower = {k.lower(): k for k in row}
    for name in candidates:
        key = lower.get(name.lower())
        if key is not None and row[key] not in ("", None):
            return key
    return None


def _as_float(value) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text or text.upper() in ("N/A", "NA", "NONE"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


# Approach A: after LightningIndexer, host-side LRU shows up as these msprof
# markers. ``lru_kernel_ms`` ALWAYS includes BOTH parts (sum), never hostfunc
# alone: lru_kernel_ms = HOSTFUNC_CALLBACK + EVENT_WAIT (steady avg).
_LRU_PROFILE_MARKERS = (
    "HOSTFUNC_CALLBACK",
    "EVENT_WAIT",
)


def _normalize_op_name(name: str) -> str:
    return "".join(ch for ch in name.upper() if ch.isalnum() or ch == "_")


def _match_lru_marker(name: str) -> str | None:
    norm = _normalize_op_name(name)
    for marker in _LRU_PROFILE_MARKERS:
        if marker in norm:
            return marker
    return None


def _print_rows(title: str, rows: list[tuple[str, int, float, float]], top_n: int):
    print(title, flush=True)
    print(f"{'name':<48} {'count':>8} {'avg_us':>12} {'total_us':>14}",
          flush=True)
    for name, count, avg_us, total_us in rows[:top_n]:
        print(f"{name:<48} {count:8d} {avg_us:12.3f} {total_us:14.3f}",
              flush=True)


def _rows_from_duration_groups(
    groups: dict[str, list[float]],
) -> list[tuple[str, int, float, float]]:
    rows = []
    for name, durs in groups.items():
        rows.append((name, len(durs), statistics.mean(durs), sum(durs)))
    rows.sort(key=lambda x: x[3], reverse=True)
    return rows


def _parse_trace_view_lru_samples(
    trace_path: Path,
) -> dict[str, list[tuple[float, float]]]:
    """Parse LRU timeline samples as ``marker -> [(timestamp_us, dur_us), ...]``.

    These Runtime/Task-Scheduler markers appear in MindStudio timeline
    (``trace_view.json``), not in AI-Core ``op_statistic`` / ``kernel_details``.
    Chrome-trace complete events use ``dur`` / ``ts`` in microseconds.
    """
    try:
        payload = json.loads(trace_path.read_text())
    except Exception as exc:
        print(f"[profile] failed to read {trace_path}: {exc}", flush=True)
        return {}

    if isinstance(payload, dict):
        events = payload.get("traceEvents") or payload.get("events") or []
    elif isinstance(payload, list):
        events = payload
    else:
        return {}

    samples: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for event in events:
        if not isinstance(event, dict):
            continue
        name = str(event.get("name") or event.get("args", {}).get("name") or "")
        marker = _match_lru_marker(name)
        if marker is None:
            continue
        dur = _as_float(event.get("dur"))
        if dur is None:
            continue
        ts = _as_float(event.get("ts"))
        if ts is None:
            ts = float(len(samples[marker]))
        samples[marker].append((ts, dur))

    for marker in samples:
        samples[marker].sort(key=lambda item: item[0])
    return samples


def _summarize_duration_samples(
    durs: list[float],
    *,
    drop_first: int = 1,
) -> dict[str, float | int | None]:
    """Return raw and steady-state stats for one marker's durations."""
    if not durs:
        return {
            "count": 0,
            "avg_us": None,
            "total_us": None,
            "min_us": None,
            "max_us": None,
            "median_us": None,
            "steady_count": 0,
            "steady_avg_us": None,
            "first_us": None,
        }
    steady = durs[drop_first:] if len(durs) > drop_first else list(durs)
    return {
        "count": len(durs),
        "avg_us": statistics.mean(durs),
        "total_us": sum(durs),
        "min_us": min(durs),
        "max_us": max(durs),
        "median_us": statistics.median(durs),
        "steady_count": len(steady),
        "steady_avg_us": statistics.mean(steady),
        "first_us": durs[0],
    }


def _collect_lru_marker_samples(
    artifacts: dict[str, Path],
) -> tuple[dict[str, list[float]], str]:
    """Return per-marker duration samples in timeline order when possible."""
    if "trace_view.json" in artifacts:
        ordered = _parse_trace_view_lru_samples(artifacts["trace_view.json"])
        if ordered:
            samples = {
                marker: [dur for _, dur in items]
                for marker, items in ordered.items()
            }
            return samples, f"trace_view.json:{artifacts['trace_view.json']}"
    return {}, "none"


def _print_lru_planner_summary(
    artifacts: dict[str, Path],
    fallback_rows: list[tuple[str, int, float, float]],
    *,
    drop_first: int = 1,
) -> dict:
    """Average HOSTFUNC_CALLBACK + EVENT_WAIT as Approach-A LRU planner time."""
    samples, source = _collect_lru_marker_samples(artifacts)
    out: dict = {"source": source, "drop_first": drop_first, "markers": {}}

    print("----- PROFILE LRU PLANNER (Approach A proxy) -----", flush=True)
    print(
        "HOSTFUNC_CALLBACK / EVENT_WAIT are Runtime timeline markers "
        "(not AI-Core ops). Prefer trace_view.json.",
        flush=True,
    )
    print(f"[profile] LRU marker source: {source}", flush=True)
    if source == "none":
        # CSV aggregates cannot drop the cold first sample reliably.
        by_marker: dict[str, list[tuple[str, int, float, float]]] = defaultdict(list)
        for row in fallback_rows:
            marker = _match_lru_marker(row[0])
            if marker is not None:
                by_marker[marker].append(row)
        if not by_marker:
            print(
                "[profile] LRU markers not found "
                f"(looked for {_LRU_PROFILE_MARKERS} in trace_view.json).",
                flush=True,
            )
            out["found"] = False
            return out
        print(
            "[profile] warning: only aggregated CSV rows available; "
            "cannot drop the cold first sample.",
            flush=True,
        )
        for marker in _LRU_PROFILE_MARKERS:
            matched = by_marker.get(marker, [])
            if not matched:
                print(f"{marker:<48} {'MISSING':>8}", flush=True)
                continue
            name, count, avg_us, total_us = max(matched, key=lambda x: x[3])
            out["markers"][marker] = {
                "name": name, "count": count,
                "avg_us": avg_us, "total_us": total_us,
            }
            print(
                f"{name:<48} {count:8d} {avg_us:12.3f} {total_us:14.3f}",
                flush=True,
            )
        out["found"] = bool(out["markers"])
        return out

    print(
        f"[profile] steady-state avg drops the first {drop_first} sample(s) "
        "per marker (first profile iter is often cold for EVENT_WAIT).",
        flush=True,
    )
    print(
        f"{'marker':<24} {'n':>4} {'first':>10} {'avg_all':>10} "
        f"{'avg_steady':>12} {'median':>10} {'min':>10} {'max':>10}",
        flush=True,
    )

    steady_avg: dict[str, float] = {}
    for marker in _LRU_PROFILE_MARKERS:
        durs = samples.get(marker, [])
        stats = _summarize_duration_samples(durs, drop_first=drop_first)
        if not durs:
            print(f"{marker:<24} {'MISS':>4}", flush=True)
            continue
        steady_avg[marker] = float(stats["steady_avg_us"])
        out["markers"][marker] = dict(stats)
        print(
            f"{marker:<24} {stats['count']:4d} "
            f"{stats['first_us']:10.3f} {stats['avg_us']:10.3f} "
            f"{stats['steady_avg_us']:12.3f} {stats['median_us']:10.3f} "
            f"{stats['min_us']:10.3f} {stats['max_us']:10.3f}",
            flush=True,
        )

    out["found"] = bool(out["markers"])
    if len(steady_avg) == len(_LRU_PROFILE_MARKERS):
        lru_steady = sum(steady_avg.values())
        out["lru_planner_steady_us"] = lru_steady
        out["hostfunc_callback_steady_us"] = steady_avg["HOSTFUNC_CALLBACK"]
        out["event_wait_steady_us"] = steady_avg["EVENT_WAIT"]
        print(
            f"{'LRU_PLANNER_STEADY':<24} "
            f"{'':>4} {'':>10} {'':>10} {lru_steady:12.3f}",
            flush=True,
        )
        print(
            "[profile] LRU planner steady avg = "
            f"{steady_avg['HOSTFUNC_CALLBACK']:.3f} + "
            f"{steady_avg['EVENT_WAIT']:.3f} = {lru_steady:.3f} us",
            flush=True,
        )
    return out


def summarize_profile_csvs(
    profile_dir: str,
    top_n: int = 30,
    *,
    drop_first: int = 1,
    include_lru_planner: bool = True,
) -> dict:
    """Print average kernel / op times; optionally Approach-A LRU markers.

    Returns a JSON-serializable summary suitable for ``--out``.
    """
    found = _ensure_profile_artifacts(profile_dir)
    summary: dict = {
        "profile_dir": str(Path(profile_dir).resolve()),
        "artifacts": {k: str(v) for k, v in found.items()},
        "source": None,
        "top_ops": [],
        "lru_planner": None,
    }
    if not found:
        print("[profile] no profile artifacts found under "
              f"{profile_dir}", flush=True)
        return summary

    rows: list[tuple[str, int, float, float]] = []
    source = None
    if "op_statistic.csv" in found:
        path = found["op_statistic.csv"]
        source = "op_statistic"
        print(f"[profile] parsing {path}", flush=True)
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name_key = _pick_csv_field(
                    row, "OP Type", "Op Type", "OP Name", "Op Name", "Name")
                count_key = _pick_csv_field(
                    row, "Count", "Calls", "Call Count", "Total Count")
                total_key = _pick_csv_field(
                    row, "Total Time(us)", "Total Duration(us)",
                    "Total Time (us)", "Total Duration (us)", "Total Time")
                avg_key = _pick_csv_field(
                    row, "Avg Time(us)", "Avg Duration(us)",
                    "Avg Time (us)", "Avg Duration (us)", "Average Time(us)")
                if name_key is None:
                    continue
                name = str(row[name_key]).strip()
                count = _as_float(row[count_key]) if count_key else None
                total_us = _as_float(row[total_key]) if total_key else None
                avg_us = _as_float(row[avg_key]) if avg_key else None
                if avg_us is None and total_us is not None and count:
                    avg_us = total_us / count
                if avg_us is None:
                    continue
                rows.append((name, int(count or 0), avg_us, total_us or 0.0))
    elif "kernel_details.csv" in found:
        path = found["kernel_details.csv"]
        source = "kernel_details"
        print(f"[profile] parsing {path}", flush=True)
        groups: dict[str, list[float]] = defaultdict(list)
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name_key = _pick_csv_field(
                    row, "OP Name", "Op Name", "OP Type", "Op Type", "Name")
                dur_key = _pick_csv_field(
                    row, "Duration(us)", "Task Duration(us)",
                    "Task Duration (us)", "Duration (us)", "Duration")
                if name_key is None or dur_key is None:
                    continue
                duration = _as_float(row[dur_key])
                if duration is None:
                    continue
                groups[str(row[name_key]).strip()].append(duration)
        rows = _rows_from_duration_groups(groups)

    rows.sort(key=lambda x: x[3], reverse=True)
    summary["source"] = source
    # Keep the full op list for kernel-name matching in --compare; top_ops
    # remains the truncated view used for stdout / quick inspection.
    summary["ops"] = [
        {
            "name": name, "count": count,
            "avg_us": avg_us, "total_us": total_us,
        }
        for name, count, avg_us, total_us in rows
    ]
    summary["top_ops"] = summary["ops"][:top_n]
    if source == "op_statistic":
        _print_rows(
            "----- PROFILE OP STATISTIC (avg kernel time) -----", rows, top_n)
    elif source == "kernel_details":
        _print_rows(
            "----- PROFILE KERNEL DETAILS (avg over captured iters) -----",
            rows, top_n)
        print(
            "[profile] note: avg_us = mean Duration(us) across all captured "
            "kernel launches in this profile window.",
            flush=True,
        )
    else:
        print(
            "[profile] no op_statistic/kernel_details found under "
            f"{profile_dir}.",
            flush=True,
        )
    if include_lru_planner:
        summary["lru_planner"] = _print_lru_planner_summary(
            found, rows, drop_first=drop_first)
    else:
        print(
            "[profile] skip LRU planner summary "
            "(Approach B has no host LRU planner; LRU is inside li_manage).",
            flush=True,
        )
    return summary


def profile_pipeline(steps, warmup, iters, profile_dir=None, top_n: int = 30,
                     reset=None, drop_first: int = 1,
                     include_lru_planner: bool = True) -> dict:
    """Run the full pipeline under torch_npu.profiler (msprof backend).

    ``steps`` is a list of ``(name, callable)`` executed in order each iter.
    Prefer a single step that represents one production round so kernels are
    not duplicated and stay on the intended streams.
    ``reset`` runs outside the timed/profiled work each iteration.
    ``include_lru_planner`` enables Approach-A HOSTFUNC/EVENT_WAIT summary.

    Each invocation writes into a fresh ``run_<timestamp>_<pid>/`` subdirectory
    under ``profile_dir``, so parsing never picks a stale ``op_statistic.csv``
    left by a previous run that reused the same ``--profile-dir``.

    Returns a JSON-serializable profile summary (includes ``run_dir``).
    """
    import tempfile
    import torch_npu.profiler as prof

    if profile_dir is None:
        profile_dir = tempfile.mkdtemp(prefix="fused_overlap_prof_")
        print(f"[profile] no --profile-dir given, using temp dir: {profile_dir}",
              flush=True)
    os.makedirs(profile_dir, exist_ok=True)
    # Isolate this capture from older exports under the same --profile-dir.
    run_dir = Path(profile_dir).resolve() / (
        f"run_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}")
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"[profile] run dir: {run_dir}", flush=True)

    for _ in range(max(0, warmup)):
        if reset is not None:
            reset()
            torch.npu.synchronize()
        for _, fn in steps:
            fn()
        torch.npu.synchronize()

    # Level2 includes CANN Runtime / Task Scheduler data that surfaces
    # HOSTFUNC_CALLBACK and EVENT_WAIT on the timeline.
    exp = prof._ExperimentalConfig(
        export_type=prof.ExportType.Text,
        profiler_level=prof.ProfilerLevel.Level2,
        aic_metrics=prof.AiCMetrics.PipeUtilization,
        data_simplification=True,
    )
    activities = [prof.ProfilerActivity.CPU, prof.ProfilerActivity.NPU]
    trace_cb = prof.tensorboard_trace_handler(str(run_dir))
    with prof.profile(activities=activities, experimental_config=exp,
                      on_trace_ready=trace_cb):
        for _ in range(iters):
            if reset is not None:
                reset()
                torch.npu.synchronize()
            for name, fn in steps:
                # Named ranges help correlate CSV rows with pipeline stages.
                with torch.profiler.record_function(name):
                    fn()
            torch.npu.synchronize()

    print(f"PROFILE_TRACE_DIR={run_dir}", flush=True)
    summary = summarize_profile_csvs(
        str(run_dir),
        top_n=top_n,
        drop_first=drop_first,
        include_lru_planner=include_lru_planner,
    )
    summary["run_dir"] = str(run_dir)
    return summary


# ===================== Approach A: vllm-ascend =====================
def run_a(args) -> dict:
    import vllm_ascend.vllm_ascend_C  # noqa: F401
    from memfabric_hybrid import offload
    from vllm_ascend.utils import enable_custom_op
    from vllm_ascend.distributed.kv_transfer.kv_offload_decode import (
        kv_offload_decode_manager,
    )

    assert enable_custom_op()
    torch_npu.npu.set_device(0)
    torch.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})
    FUSED_OVERLAP = torch.ops._C_ascend.npu_fused_sparse_attention_overlap

    def _align(t, a):
        p = t.data_ptr()
        ap = (p + a - 1) // a * a
        return t[(ap - p) // t.element_size():]

    def _empty_cpu(shape, dtype):
        n = int(np.prod(shape))
        nb = n * torch.empty((), dtype=dtype).element_size()
        raw = offload.empty([nb + _ALIGN], dtype=torch.int8, pin_memory=True)
        return _align(raw, _ALIGN)[:nb].view(dtype).view(shape)

    dt = torch.bfloat16
    batch = args.batch_size
    q_heads = args.q_heads
    idx_heads = args.indexer_heads
    seq_len = args.seq_len
    topk_layout = args.topk_layout
    miss = resolve_miss_count(args)
    hit_count = TOPK - miss
    hit_ratio = 1.0 - (miss / float(TOPK))
    cache_tokens_req = int(args.cache_tokens)
    # Per-request CPU KV pool: batch independent sequences.
    blocks_per_seq = cdiv(seq_len, BLOCK_SIZE)
    fmbn = batch * blocks_per_seq
    smbn = cdiv(TOPK, BLOCK_SIZE)
    scale = 1.0 / math.sqrt(KVD)
    elem = torch.empty((), dtype=dt).element_size()
    cpu_kv_bytes = fmbn * BLOCK_SIZE * (KVD + KRD) * elem
    # membership_map ~ [batch, 16392] int16 + a few aligned slabs of slack.
    cpu_meta_bytes = batch * 16392 * 2 + 8 * _ALIGN
    needed_pool = cpu_kv_bytes + cpu_meta_bytes + 64 * 1024 * 1024
    pool_bytes = max(int(args.dram_size_gb * _GIB), int(needed_pool))
    cfg = offload.OffloadConfig()
    cfg.device_id = torch_npu.npu.current_device()
    cfg.size = pool_bytes
    cfg.world_size = 1
    cfg.rank_id = 0
    offload.initialize(cfg)
    print(
        f"[A-mem] CPU KV scaled by batch: blocks={fmbn} "
        f"(batch={batch} x blocks_per_seq={blocks_per_seq})  "
        f"full_kv+rope≈{cpu_kv_bytes / _GIB:.3f} GiB  "
        f"offload_pool={pool_bytes / _GIB:.3f} GiB",
        flush=True,
    )
    torch.manual_seed(910000 + batch + q_heads * 17 + seq_len)

    q = torch.randn(batch, q_heads, KVD, dtype=dt, device="npu")
    qr = torch.randn(batch, q_heads, KRD, dtype=dt, device="npu")
    q_fused = torch.cat([q, qr], dim=-1).contiguous()
    full_kv = _empty_cpu([fmbn, BLOCK_SIZE, KVD], dt)
    full_kv.zero_()
    full_rope = _empty_cpu([fmbn, BLOCK_SIZE, KRD], dt)
    full_rope.zero_()
    # Row r owns blocks [r*blocks_per_seq, (r+1)*blocks_per_seq).
    full_bt = (
        torch.arange(fmbn, dtype=torch.int32, device="npu")
        .view(batch, blocks_per_seq)
        .contiguous()
    )
    aq = torch.tensor([1] * batch, dtype=torch.int32, device="npu")
    ak = torch.tensor([seq_len] * batch, dtype=torch.int32, device="npu")

    index_seed = 910000 + batch + q_heads * 17 + seq_len
    index_scores_cpu, topk_tokens_cpu = build_index_scores_and_topk(
        seq_len, topk_layout, index_seed)
    topk = topk_tokens_cpu.to("npu").view(1, 1, 1, TOPK).expand(
        batch, 1, 1, TOPK).contiguous()
    non_topk_mask = torch.ones(seq_len, dtype=torch.bool)
    non_topk_mask[topk_tokens_cpu.to(torch.long)] = False
    non_topk = torch.arange(seq_len, dtype=torch.int32)[non_topk_mask]

    total_sel = smbn * batch
    sel_kv = torch.zeros(total_sel, BLOCK_SIZE, KVD, dtype=dt, device="npu")
    sel_rope = torch.zeros(total_sel, BLOCK_SIZE, KRD, dtype=dt, device="npu")
    sel_bt = torch.arange(total_sel, dtype=torch.int32, device="npu").reshape(batch, smbn)
    sel_status = torch.full((batch, 1, 1, TOPK + 1), -1, dtype=torch.int32, device="npu")

    li_q = torch.zeros((batch, idx_heads, INDEX_DIM), dtype=dt, device="npu")
    li_q[:, 0, 0] = 1
    li_q[:, 0, 1] = 64
    li_q[:, 0, 2] = 4096
    li_w = torch.zeros((batch, idx_heads), dtype=dt, device="npu")
    li_w[:, 0] = 1
    bpr = seq_len // BLOCK_SIZE
    li_key = torch.zeros((batch * bpr, BLOCK_SIZE, 1, INDEX_DIM), dtype=dt, device="npu")
    index_scores = index_scores_cpu.to("npu").view(1, bpr, BLOCK_SIZE)
    kr = li_key.view(batch, bpr, BLOCK_SIZE, 1, INDEX_DIM)
    kr[:, :, :, 0, 0] = (index_scores % 64).to(dt)
    kr[:, :, :, 0, 1] = ((index_scores // 64) % 64).to(dt)
    kr[:, :, :, 0, 2] = (index_scores // 4096).to(dt)
    li_bt = torch.arange(batch * bpr, dtype=torch.int32, device="npu").view(batch, bpr)
    li_ql = torch.arange(1, batch + 1, dtype=torch.int32, device="npu")
    li_cl = torch.full((batch,), seq_len, dtype=torch.int32, device="npu")

    # ---- host-side LRU planner (production path) -----------------------
    # The production A path runs the LRU/replacement planner on host
    # (kv_offload_decode.cpp::lru_resident_compact_with_plan_stable_rows)
    # BEFORE the fused kernel. The kernel then consumes the encoded plan
    # written into selection_membership_map (EXTERNAL_PLAN_READY_MARKER).
    ascend_home = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
    npu_include = os.path.join(ascend_home, "include")
    npu_lib = os.path.join(ascend_home, "lib64")
    if not os.path.exists(npu_lib):
        npu_lib = os.path.join(ascend_home, "lib")
    torch_npu_path = os.path.dirname(torch_npu.__file__)
    torch_npu_include = os.path.join(torch_npu_path, "include")
    torch_npu_lib = os.path.join(torch_npu_path, "lib")
    os.environ["TORCH_EXTENSIONS_ALWAYS_BUILD"] = "1"
    os.environ["CXX"] = "clang++"
    os.environ["CC"] = "clang"
    cpp_src = os.path.join(
        os.path.dirname(os.path.abspath(kv_offload_decode_manager.__file__)),
        "kv_offload_decode.cpp",
    )
    lru_cpp = torch.utils.cpp_extension.load(
        name="kv_offload_decode",
        sources=[cpp_src],
        extra_cflags=[
            "-O3", "-std=c++20", "-fopenmp",
            "-march=armv8.2-a+sve+fp16+bf16", "-fPIC",
            f"-I{npu_include}", f"-I{torch_npu_include}",
        ],
        extra_ldflags=[
            "-fopenmp", f"-L{npu_lib}", "-lascendcl",
            f"-L{torch_npu_lib}", "-ltorch_npu",
        ],
        verbose=False,
    )

    EXTERNAL_PLAN_READY_MARKER = 0x5A45
    PAIRED_SELECTION_COPY_MARKER = 0x5A56
    MEMBERSHIP_MAP_INT16 = 16376
    MEMBERSHIP_ALIGN_INT16 = 16
    MEMBERSHIP_CONTROL_INT16 = 8
    CONTROL_OFFSET_INT16 = (
        MEMBERSHIP_MAP_INT16 + MEMBERSHIP_ALIGN_INT16
    ) // MEMBERSHIP_ALIGN_INT16 * MEMBERSHIP_ALIGN_INT16  # 16384
    STORAGE_INT16 = (
        CONTROL_OFFSET_INT16 + MEMBERSHIP_CONTROL_INT16 + MEMBERSHIP_ALIGN_INT16
    ) // MEMBERSHIP_ALIGN_INT16 * MEMBERSHIP_ALIGN_INT16  # 16392

    topk_buffer_size = TOPK * 2
    max_rows = batch
    threads = 8
    plan_start = CONTROL_OFFSET_INT16 - TOPK
    required_columns = CONTROL_OFFSET_INT16 + MEMBERSHIP_CONTROL_INT16

    membership_map = offload.empty(
        [max_rows * STORAGE_INT16], dtype=torch.int16, pin_memory=True,
    ).view([max_rows, STORAGE_INT16])
    membership_map.fill_(-1)
    control = membership_map[
        :, CONTROL_OFFSET_INT16:CONTROL_OFFSET_INT16 + MEMBERSHIP_CONTROL_INT16]
    control[:, 1] = EXTERNAL_PLAN_READY_MARKER
    control[:, 2] = TOPK
    control[:, 3] = CONTROL_OFFSET_INT16 - TOPK
    control[:, 7] = PAIRED_SELECTION_COPY_MARKER
    plan_storage = membership_map[:max_rows, plan_start:required_columns]
    encoded_plan_stride = membership_map.stride(0)

    lru_req_ids = torch.empty([max_rows], dtype=torch.int64, device="cpu", pin_memory=True)
    lru_last_req_ids = torch.empty([max_rows], dtype=torch.int64, device="cpu", pin_memory=True)
    lru_topk_indices = torch.empty([max_rows, TOPK], dtype=torch.int32, device="cpu", pin_memory=True)
    lru_stable_prefix_lens = torch.empty([max_rows], dtype=torch.int32, device="cpu", pin_memory=True)
    lru_visible_seq_lens = torch.empty([max_rows], dtype=torch.int32, device="cpu", pin_memory=True)
    lru_slot_to_token = torch.empty(
        [max_rows, topk_buffer_size], dtype=torch.int32, device="cpu", pin_memory=True)
    lru_slots = torch.empty(
        [max_rows, topk_buffer_size], dtype=torch.int32, device="cpu", pin_memory=True)
    lru_current_slots = torch.empty([max_rows, TOPK], dtype=torch.int32, device="cpu", pin_memory=True)
    lru_miss_count = torch.empty([max_rows], dtype=torch.int32, device="cpu", pin_memory=True)
    lru_miss_tokens = torch.empty([max_rows, TOPK], dtype=torch.int32, device="cpu", pin_memory=True)
    lru_miss_slots = torch.empty([max_rows, TOPK], dtype=torch.int32, device="cpu", pin_memory=True)
    lru_token_mark = torch.zeros([threads, seq_len], dtype=torch.int32, device="cpu", pin_memory=True)
    lru_token_pos = torch.full([threads, seq_len], -1, dtype=torch.int32, device="cpu", pin_memory=True)
    lru_slot_ws = torch.empty([threads, topk_buffer_size * 3], dtype=torch.int32, device="cpu", pin_memory=True)
    lru_miss_pos_ws = torch.empty([threads, TOPK], dtype=torch.int32, device="cpu", pin_memory=True)
    lru_epochs = torch.zeros([threads], dtype=torch.int32, device="cpu", pin_memory=True)
    lru_physical_row_ws = torch.empty([max_rows * 3], dtype=torch.int32, device="cpu", pin_memory=True)

    # Stable-row planner reserves one slot for the current token, so only
    # capacity-1 slots are usable residents (see kv_offload_decode.cpp).
    resident_capacity = topk_buffer_size - 1
    fill_count = min(cache_tokens_req, resident_capacity)
    if fill_count < hit_count:
        raise ValueError(
            f"Need at least hit_count={hit_count} resident slots to honor "
            f"--miss={miss}, but fill_count={fill_count} "
            f"(cache_tokens={cache_tokens_req}, "
            f"resident_capacity={resident_capacity}).")

    # Each reset rebuilds residents so miss count stays fixed, but which
    # topk tokens are hits vs misses is freshly randomized every round.
    lru_slots_init = (
        torch.arange(topk_buffer_size, dtype=torch.int32)
        .view(1, -1).repeat(max_rows, 1).contiguous()
    )
    lru_last_req_ids_init = torch.full(
        [max_rows], -1, dtype=torch.int64)
    lru_last_req_ids_init[:batch] = torch.arange(batch, dtype=torch.int64)
    _a_reset_round = [0]

    lru_cpp.warmup_lru_resident_threads(threads)

    use_graph = bool(getattr(args, "graph_mode", True))
    topk_flat = topk.reshape(batch, TOPK).contiguous()
    req_ids_npu = torch.arange(batch, dtype=torch.int64, device="npu")
    # Production decode: stable_prefix = actual_seq_key - query_len.
    # aq/ak are therefore wired into the planner (not decorative).
    stable_prefix_npu = (ak - aq).clamp_min(0).to(torch.int32).contiguous()
    visible_seq_npu = ak.to(torch.int32).contiguous()
    stable_prefix_len = int(stable_prefix_npu[0].item())
    visible_seq_len = int(visible_seq_npu[0].item())
    if not torch.all(stable_prefix_npu == stable_prefix_len):
        raise RuntimeError("A UT expects uniform stable_prefix across the batch")
    if not torch.all(visible_seq_npu == visible_seq_len):
        raise RuntimeError("A UT expects uniform visible_seq across the batch")

    # Tokens >= stable_prefix are wiped as speculative suffix each plan call.
    # Hit prefill must therefore only use tokens in [0, stable_prefix).
    eligible_topk = topk_tokens_cpu[topk_tokens_cpu < stable_prefix_len]
    forced_suffix_miss = int((topk_tokens_cpu >= stable_prefix_len).sum().item())
    if eligible_topk.numel() < hit_count:
        raise ValueError(
            f"Only {eligible_topk.numel()} topk tokens are < stable_prefix="
            f"{stable_prefix_len}, but hit_count={hit_count} is required for "
            f"--miss={miss} (forced_suffix_miss={forced_suffix_miss}).")
    eligible_non_topk = non_topk[non_topk < stable_prefix_len]
    if eligible_non_topk.numel() < fill_count - hit_count:
        raise ValueError(
            f"Not enough non-topk tokens < stable_prefix={stable_prefix_len} "
            f"to fill residents: need {fill_count - hit_count}, "
            f"have {eligible_non_topk.numel()}.")

    print(
        f"[A-cfg] batch={batch} q_heads={q_heads} idx_heads={idx_heads} "
        f"seq_len={seq_len} topk_layout={topk_layout} miss={miss} "
        f"hit_count={hit_count} hit_ratio={hit_ratio:.4f}",
        flush=True,
    )
    print(
        f"[A-lru] wired inputs: req_ids=arange({batch}) "
        f"stable_prefix=ak-aq={stable_prefix_len} "
        f"visible_seq=ak={visible_seq_len} "
        f"topk_buffer={topk_buffer_size} resident_capacity={resident_capacity} "
        f"cache_tokens_req={cache_tokens_req} fill_count={fill_count} "
        f"forced_suffix_miss={forced_suffix_miss} "
        f"eligible_topk={eligible_topk.numel()}",
        flush=True,
    )
    if cache_tokens_req > resident_capacity:
        print(
            f"[A-lru] NOTE: --cache-tokens={cache_tokens_req} exceeds A "
            f"resident_capacity={resident_capacity}; fill_count capped to "
            f"{fill_count} (A physical buffer is TOPK*2-1).",
            flush=True,
        )

    plan_stream = torch_npu.npu.Stream()
    # Do NOT call _subscribe_report on this stream. Production enqueue uses
    # aclrtLaunchHostFunc, which creates/subscribes its own callback thread.
    # Mixing it with aclrtSubscribeReport on the same stream returns 107011
    # (ACL_ERROR_RT_STREAM_SUBSCRIBE).

    def _planner_kwargs_ptrs():
        return (
            lru_req_ids.data_ptr(),
            lru_last_req_ids.data_ptr(),
            lru_topk_indices.data_ptr(),
            lru_stable_prefix_lens.data_ptr(),
            lru_slot_to_token.data_ptr(),
            lru_slots.data_ptr(),
            lru_current_slots.data_ptr(),
            lru_miss_count.data_ptr(),
            lru_miss_tokens.data_ptr(),
            lru_miss_slots.data_ptr(),
            lru_token_mark.data_ptr(),
            lru_token_pos.data_ptr(),
            lru_slot_ws.data_ptr(),
            lru_miss_pos_ws.data_ptr(),
            lru_epochs.data_ptr(),
            lru_physical_row_ws.data_ptr(),
            max_rows,
            plan_storage.data_ptr(),
            encoded_plan_stride,
            batch,
            TOPK,
            topk_buffer_size,
            seq_len,
            threads,
            threads,
            lru_visible_seq_lens.data_ptr(),
        )

    def _copy_planner_inputs_d2h(*, non_blocking: bool = True):
        # Match production: D2H from NPU tensors on the current stream.
        lru_topk_indices.copy_(topk_flat, non_blocking=non_blocking)
        lru_req_ids.copy_(req_ids_npu, non_blocking=non_blocking)
        lru_stable_prefix_lens.copy_(stable_prefix_npu, non_blocking=non_blocking)
        lru_visible_seq_lens.copy_(visible_seq_npu, non_blocking=non_blocking)

    def _a_reset_lru_state():
        """Rebuild residents for one round: fixed miss count, fresh hit/miss set."""
        round_id = _a_reset_round[0]
        _a_reset_round[0] = round_id + 1
        lru_slot_to_token.fill_(-1)
        other_count = fill_count - hit_count
        for r in range(batch):
            gen = torch.Generator().manual_seed(1000 + round_id * 10007 + r)
            hit_perm = torch.randperm(eligible_topk.numel(), generator=gen)
            hit_tokens = eligible_topk[hit_perm[:hit_count]]
            others = eligible_non_topk[
                torch.randperm(eligible_non_topk.numel(), generator=gen)[:other_count]
            ]
            cached = torch.cat((hit_tokens, others))
            # Reject any token that production would wipe as speculative suffix.
            if torch.any(cached >= stable_prefix_len):
                raise RuntimeError(
                    "resident prefill produced token >= stable_prefix; "
                    "LRU would discard it and break --miss control")
            slots = torch.randperm(resident_capacity, generator=gen)[:fill_count]
            lru_slot_to_token[r, slots] = cached
            valid = lru_slot_to_token[r, :resident_capacity]
            valid = valid[valid >= 0]
            if int(valid.numel()) != fill_count:
                raise RuntimeError(
                    f"A row {r} resident fill={valid.numel()} != "
                    f"fill_count={fill_count} from --cache-tokens")
            hits_in_row = int(torch.isin(valid, topk_tokens_cpu).sum().item())
            if hits_in_row != hit_count:
                raise RuntimeError(
                    f"A row {r} topk hits in residents={hits_in_row}, "
                    f"expected hit_count={hit_count} from --miss={miss}")
        lru_slots.copy_(lru_slots_init)
        lru_last_req_ids.copy_(lru_last_req_ids_init)
        lru_epochs.zero_()
        membership_map.fill_(-1)
        control[:, 1] = EXTERNAL_PLAN_READY_MARKER
        control[:, 2] = TOPK
        control[:, 3] = CONTROL_OFFSET_INT16 - TOPK
        control[:, 7] = PAIRED_SELECTION_COPY_MARKER
        sel_status.fill_(-1)

    def _a_validate_lru_inputs_and_outputs():
        """Fail hard if any planner input is unwired or miss control broke."""
        _assert_tensor_eq("lru_topk_indices", lru_topk_indices[:batch], topk_flat.cpu())
        _assert_tensor_eq("lru_req_ids", lru_req_ids[:batch], req_ids_npu.cpu())
        _assert_tensor_eq(
            "lru_stable_prefix_lens",
            lru_stable_prefix_lens[:batch],
            stable_prefix_npu.cpu(),
        )
        _assert_tensor_eq(
            "lru_visible_seq_lens",
            lru_visible_seq_lens[:batch],
            visible_seq_npu.cpu(),
        )
        # Continuing-request path: last_req_ids must stay equal to req_ids.
        _assert_tensor_eq(
            "lru_last_req_ids",
            lru_last_req_ids[:batch],
            req_ids_npu.cpu(),
        )
        if int(control[0, 1].item()) != EXTERNAL_PLAN_READY_MARKER:
            raise RuntimeError("membership control EXTERNAL_PLAN_READY_MARKER not set")
        if int(control[0, 2].item()) != TOPK:
            raise RuntimeError("membership control topk field not set to TOPK")
        actual_miss = lru_miss_count[:batch].tolist()
        expected_miss = [miss] * batch
        if actual_miss != expected_miss:
            raise RuntimeError(
                f"[A] controlled miss mismatch: got {actual_miss}, "
                f"expected {expected_miss}. Check resident prefill / "
                f"stable_prefix/visible_seq wiring.")
        # Encoded plan must be populated for every topk position (hit>0 or miss<0).
        plan = plan_storage[:batch, :TOPK]
        if torch.any(plan == 0):
            raise RuntimeError(
                "encoded plan has zero entries; planner did not write a full plan")
        # Resident occupancy after plan still uses the continuing-req row mapping.
        occupied = (lru_slot_to_token[:batch, :resident_capacity] >= 0).sum(dim=1)
        if torch.any(occupied < hit_count):
            raise RuntimeError(
                f"post-plan resident occupancy {occupied.tolist()} "
                f"fell below hit_count={hit_count}")
        print(
            f"[A-lru] validated: miss_counts={actual_miss} "
            f"stable_prefix={stable_prefix_len} visible_seq={visible_seq_len} "
            f"fill_count={fill_count} occupied_after_plan={occupied.tolist()}",
            flush=True,
        )

    def _a_planner_sync():
        """Synchronous planner — used for miss verification / fused prep."""
        _copy_planner_inputs_d2h(non_blocking=False)
        lru_cpp.lru_resident_compact_with_plan_stable_rows(*_planner_kwargs_ptrs())

    def _a_reset_then_plan():
        """Reset residents then build a fresh plan (isolated fused timing)."""
        _a_reset_lru_state()
        _a_planner_sync()

    def _a_planner():
        """Production-like planner: side-stream D2H + C++ enqueue host func.

        Graph capture records the side-stream work and the main-stream wait,
        so replay no longer pays eager Python dispatch around the LRU callback.
        """
        compute = torch_npu.npu.current_stream()
        ready = compute.record_event()
        with torch_npu.npu.stream(plan_stream):
            plan_stream.wait_event(ready)
            _copy_planner_inputs_d2h()
            lru_cpp.enqueue_lru_resident_compact_with_plan_stable_rows(
                *_planner_kwargs_ptrs())

    def _a_li():
        o = torch_npu.npu_lightning_indexer(
            query=li_q, key=li_key, weights=li_w,
            actual_seq_lengths_query=li_ql, actual_seq_lengths_key=li_cl,
            block_table=li_bt, layout_query="TND", layout_key="PA_BSND",
            sparse_count=TOPK, sparse_mode=3)
        return o[0] if isinstance(o, (tuple, list)) else o

    def _a_fused():
        # Host LRU does NOT emit a new topk. Fused consumes:
        #   - selection_membership_map: LRU-encoded external plan (THE LRU output)
        #   - selection_topk_indices: the SAME topk that was fed into the planner
        # Other selection_* buffers are workspace; plan tells fused hit/miss slots.
        return FUSED_OVERLAP(
            query=q_fused,
            selection_k_rope=sel_rope,
            selection_kv_cache=sel_kv,
            selection_kv_block_table=sel_bt,
            selection_kv_block_status=sel_status,
            selection_membership_map=membership_map,
            selection_topk_indices=topk,
            full_k_rope=full_rope,
            full_kv_cache=full_kv,
            full_kv_block_table=full_bt,
            full_kv_actual_seq=ak,
            full_q_actual_seq=aq,
            scale_value=scale,
            sparse_block_size=1,
            selection_topk_block_size=1,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3)

    def _a_planner_plus_fused():
        # Production order: side-stream LRU writes plan into membership_map,
        # compute stream waits, then fused reads that plan (+ shared topk).
        _a_planner()
        torch_npu.npu.current_stream().wait_stream(plan_stream)
        _a_fused()

    def _a_full_pipeline():
        # One capture/replay unit for profiling: LI and fused share the
        # compute stream; only LRU host-callback uses plan_stream.
        _a_li()
        _a_planner_plus_fused()

    # Eager functional smoke + fail-hard validation of every wired LRU input.
    _a_reset_lru_state()
    _a_planner_sync()
    _a_validate_lru_inputs_and_outputs()
    # Fused must see the plan just written by the planner (not a stale map).
    _a_fused()
    torch.npu.synchronize()
    print(
        "[A-cfg] fused_overlap consumes LRU plan via selection_membership_map; "
        "selection_topk_indices is the shared planner input topk (LRU does not "
        "output topk).",
        flush=True,
    )
    print(f"[A-cfg] timing_mode={'graph' if use_graph else 'eager'}", flush=True)

    # Profile must capture ONE full pipeline. Capturing LI / fused as separate
    # graphs previously made them look like different streams and caused fused
    # to execute twice per profile iter (pipeline + isolated fused).
    full_runner, full_mode = maybe_graph_runner(
        _a_full_pipeline, use_graph=use_graph,
        warmup=max(1, args.warmup // 2), reset=_a_reset_lru_state)

    # Event timing breaks components out with separate graphs/runners.
    # Always measured (even under --profile) so --out has compare fields:
    #   lru_ms      = planner_ms      ↔ B li_manage_minus_li_ms
    #   sfa_copy_ms = fused_overlap_ms ↔ B scatter_sfa_ms
    li_runner, li_mode = maybe_graph_runner(
        _a_li, use_graph=use_graph, warmup=max(1, args.warmup // 2))
    pipeline_runner, pipeline_mode = maybe_graph_runner(
        _a_planner_plus_fused, use_graph=use_graph,
        warmup=max(1, args.warmup // 2), reset=_a_reset_lru_state)
    fused_runner, fused_mode = maybe_graph_runner(
        _a_fused, use_graph=use_graph, warmup=max(1, args.warmup // 2),
        reset=_a_reset_then_plan)

    profile_summary = None
    if getattr(args, "profile", False):
        profile_summary = profile_pipeline(
            [("full_pipeline", full_runner)],
            args.warmup, args.profile_iters, args.profile_dir,
            top_n=args.profile_top_n,
            reset=_a_reset_lru_state,
            drop_first=args.profile_drop_first,
            include_lru_planner=True)

    li_ms = bench_events(li_runner, args.warmup, args.iters)
    pipeline_ms = bench_events(
        pipeline_runner, args.warmup, args.iters, reset=_a_reset_lru_state)
    fused_ms = bench_events(
        fused_runner, args.warmup, args.iters, reset=_a_reset_then_plan)
    planner_ms = max(0.0, pipeline_ms - fused_ms)
    print(
        f"[A-timing] LI={li_ms:.4f} ms  planner(lru)={planner_ms:.4f} ms  "
        f"fused_overlap(sfa+copy)={fused_ms:.4f} ms  "
        f"pipeline={pipeline_ms:.4f} ms",
        flush=True,
    )
    try:
        offload.uninitialize()
    except Exception as exc:
        print(f"OFFLOAD_UNINIT_WARN {type(exc).__name__}: {exc}", flush=True)

    result = {
        "approach": "A",
        "timing_mode": full_mode,
        "batch_size": batch, "q_heads": q_heads, "indexer_heads": idx_heads,
        "seq_len": seq_len, "topk_layout": topk_layout,
        "miss": miss, "hit_count": hit_count, "hit_ratio": hit_ratio,
        "cache_tokens_req": cache_tokens_req, "fill_count": fill_count,
        "stable_prefix": stable_prefix_len, "visible_seq": visible_seq_len,
        "topk_buffer_size": topk_buffer_size,
        "resident_capacity": resident_capacity,
        "li_ms": li_ms,
        "planner_ms": planner_ms,
        "fused_overlap_ms": fused_ms,
        "pipeline_ms": pipeline_ms,
        "total_ms": li_ms + pipeline_ms,
        "warmup": args.warmup,
        "iters": args.iters,
        "li_mode": li_mode,
        "pipeline_mode": pipeline_mode,
        "fused_mode": fused_mode,
    }
    attach_compare_fields(result, lru_ms=planner_ms, sfa_copy_ms=fused_ms)
    if profile_summary is not None:
        lru = profile_summary.get("lru_planner") or {}
        result["profile"] = True
        result["profile_run_dir"] = profile_summary.get("run_dir")
        result["profile_dir"] = profile_summary.get("profile_dir")
        result["lru_planner_steady_us"] = lru.get("lru_planner_steady_us")
        result["hostfunc_callback_steady_us"] = lru.get(
            "hostfunc_callback_steady_us")
        result["event_wait_steady_us"] = lru.get("event_wait_steady_us")
        result["profile_summary"] = profile_summary
        # Profile-timeline LRU proxy (us); also promoted to lru_kernel_ms.
        if lru.get("lru_planner_steady_us") is not None:
            result["lru_profile_ms"] = float(lru["lru_planner_steady_us"]) / 1000.0
        attach_kernel_compare_fields(result, "A")
    return result


# ===================== Approach B: nano =====================
def run_b(args) -> dict:
    if args.nano_path not in sys.path:
        sys.path.insert(0, args.nano_path)

    np_dir = args.nano_path
    pkg_dir = os.path.join(np_dir, "ops_overlap")
    opapi = os.path.join(pkg_dir, "..", "..", "_custom_opp", "vendors",
                         "ops-overlap", "op_api", "lib", "libcust_opapi.so")
    so_files = [f for f in os.listdir(pkg_dir)
                if f.startswith("_C") and f.endswith(".so")] if os.path.isdir(pkg_dir) else []
    print(f"[B-diag] nano_path={np_dir!r}", flush=True)
    print(f"[B-diag] ops_overlap/ exists: {os.path.isdir(pkg_dir)}", flush=True)
    print(f"[B-diag] ops_overlap/__init__.py exists: "
          f"{os.path.isfile(os.path.join(pkg_dir, '__init__.py'))}", flush=True)
    print(f"[B-diag] _C*.so files: {so_files}", flush=True)
    print(f"[B-diag] libcust_opapi.so exists: "
          f"{os.path.isfile(opapi)} ({opapi})", flush=True)
    print(f"[B-diag] sys.path[0:3]={sys.path[:3]}", flush=True)

    try:
        import ops_overlap  # noqa: F401  sets ASCEND_CUSTOM_OPP_PATH
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise ImportError(
            f"Cannot import ops_overlap from --nano-path={np_dir!r}. "
            f"Ensure nano is built (bash build.sh in nanovllm-DSA-offload) and "
            f"the path contains ops_overlap/. Original error: {exc}"
        ) from exc

    # verify the required ops are actually registered (stale _C.so check)
    _required_ops = [
        "li_manage_out",
        "sparse_and_tail_attention_and_scatter_copy",
    ]
    _missing = [op for op in _required_ops
                if not hasattr(torch.ops.ops_overlap, op)]
    if _missing:
        raise RuntimeError(
            f"ops_overlap._C.so is stale — missing ops: {_missing}. "
            f"Rebuild nano: cd <nano-root> && rm -rf _custom_opp "
            f"torch_extension/build torch_extension/ops_overlap/_C*.so && bash build.sh"
        )

    torch_npu.npu.set_device(0)
    torch.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})

    def _swap(cpu, dev):
        t = torch_npu.empty_with_swapped_memory(cpu.shape, dtype=cpu.dtype, device=dev)
        t.fill_(0)
        t.add_(cpu.to(dev))
        return t

    dev = torch.device("npu:0")
    dt = torch.bfloat16
    batch = args.batch_size
    q_heads = args.q_heads
    idx_heads = args.indexer_heads
    seq_len = args.seq_len
    cache_tokens = args.cache_tokens
    miss = resolve_miss_count(args)
    topk_layout = args.topk_layout
    if idx_heads not in (32, 64):
        raise RuntimeError(
            f"li_manage_out requires indexer_heads in (32, 64), got {idx_heads}. "
            f"Use --indexer-heads 32 or --indexer-heads 64.")
    hit_count = TOPK - miss
    print(f"[B-cfg] batch={batch} q_heads={q_heads} idx_heads={idx_heads} "
          f"seq_len={seq_len} topk_layout={topk_layout} "
          f"cache_tokens={cache_tokens} miss={miss} hit_count={hit_count}",
          flush=True)
    print(
        f"[B-lru] wired inputs: cache_slots prefilled each reset with "
        f"exactly hit_count={hit_count} topk hits + "
        f"{cache_tokens - hit_count} non-topk; li_manage_out must report "
        f"miss_counts=={[miss]}*batch",
        flush=True,
    )
    torch.manual_seed(910000 + batch + q_heads * 17 + seq_len + 1)
    cb = math.ceil(cache_tokens / BLOCK_SIZE)
    sb = seq_len // BLOCK_SIZE

    hbm_bt = torch.empty((batch, cb), dtype=torch.int32)
    for r in range(batch):
        ph = torch.arange(r * cb, (r + 1) * cb, dtype=torch.int32)
        hbm_bt[r] = ph[torch.randperm(cb)]
    # Per-request DRAM KV pool (swapped CPU backing), scaled by batch.
    # Row r owns blocks [r*sb, (r+1)*sb); block table permutes within that range.
    total_sb = batch * sb
    dram_bt = torch.stack([
        (torch.randperm(sb, dtype=torch.int64) + r * sb).to(torch.int32)
        for r in range(batch)
    ])

    thb = batch * cb
    hbm_kpe = torch.randn((thb, BLOCK_SIZE, 1, KRD), dtype=dt, device=dev)
    hbm_ckv = torch.randn((thb, BLOCK_SIZE, 1, KVD), dtype=dt, device=dev)
    query = torch.randn((batch, q_heads, KVD), dtype=dt, device=dev)
    qrope = torch.randn((batch, q_heads, KRD), dtype=dt, device=dev)

    # ---- deterministic index key and selected top-k token set ----
    # A base-64 encoding preserves the score in the matching query below.
    tib = batch * sb
    index_seed = 910000 + batch + q_heads * 17 + seq_len
    index_scores_cpu, topk_tokens = build_index_scores_and_topk(
        seq_len, topk_layout, index_seed)
    idx_cache_cpu = torch.zeros((tib, BLOCK_SIZE, 1, INDEX_DIM), dtype=dt)
    index_scores = index_scores_cpu.view(1, sb, BLOCK_SIZE)
    idx_cache_rows = idx_cache_cpu.view(batch, sb, BLOCK_SIZE, 1, INDEX_DIM)
    idx_cache_rows[:, :, :, 0, 0] = (index_scores % 64).to(dt)
    idx_cache_rows[:, :, :, 0, 1] = ((index_scores // 64) % 64).to(dt)
    idx_cache_rows[:, :, :, 0, 2] = (index_scores // 4096).to(dt)
    idx_cache = idx_cache_cpu.to(dev)
    idx_bt = torch.arange(tib, dtype=torch.int32, device=dev).view(batch, sb)

    li_query = torch.zeros((batch, idx_heads, INDEX_DIM), dtype=dt, device=dev)
    li_query[:, 0, 0] = 1
    li_query[:, 0, 1] = 64
    li_query[:, 0, 2] = 4096

    g = torch.Generator().manual_seed(42)
    dk = torch.randn((total_sb, BLOCK_SIZE, KRD), generator=g, dtype=torch.float32).to(dt)
    dc = torch.randn((total_sb, BLOCK_SIZE, KVD), generator=g, dtype=torch.float32).to(dt)
    dram_kpe = _swap(dk, dev)
    dram_ckv = _swap(dc, dev)
    dram_bytes = (
        total_sb * BLOCK_SIZE * (KVD + KRD)
        * torch.empty((), dtype=dt).element_size()
    )
    print(
        f"[B-mem] DRAM KV scaled by batch: blocks={total_sb} "
        f"(batch={batch} x sb={sb})  dram_kv+rope≈{dram_bytes / _GIB:.3f} GiB",
        flush=True,
    )
    torch.npu.synchronize()

    # ---- cache_slots rebuilt each reset: fixed miss, random hit/miss set ----
    pool = batch + 7
    pg = torch.Generator().manual_seed(99)
    req_e = torch.randperm(pool, generator=pg)[:batch].to(torch.int32)
    non_topk_mask = torch.ones(seq_len, dtype=torch.bool)
    non_topk_mask[topk_tokens] = False
    non_topk = torch.arange(seq_len, dtype=torch.int32)[non_topk_mask]
    if cache_tokens < hit_count:
        raise ValueError(
            f"Need cache_tokens >= hit_count to honor --miss={miss}, "
            f"got cache_tokens={cache_tokens}, hit_count={hit_count}.")
    cache_slots = torch.full((pool, seq_len), -1, dtype=torch.int32, device=dev)
    _b_reset_round = [0]

    weights = torch.zeros((batch, idx_heads), dtype=dt, device=dev)
    weights[:, 0] = 1
    cache_tokens_t = torch.full((batch,), cache_tokens, dtype=torch.int32, device=dev)
    candidate_lens = torch.full((batch,), seq_len, dtype=torch.int32, device=dev)
    # li_manage_out outputs (alias caller buffers):
    #   source_ids / topk_index: miss-first then hit (not IOSORT within parts)
    #   destination_slots / topk_slots: aligned 1:1 with source_ids
    #   miss_counts
    #   cache_slots (updated)
    src_ids_t = torch.full((batch, TOPK), -1, dtype=torch.int32, device=dev)
    li_src_ids = src_ids_t.unsqueeze(1)
    li_dst_slots = torch.empty_like(li_src_ids)
    miss_counts = torch.empty((batch,), dtype=torch.int32, device=dev)
    use_graph = bool(getattr(args, "graph_mode", True))
    req_e_dev = req_e.to(dev)
    actual_q = torch.arange(1, batch + 1, dtype=torch.int32, device=dev)
    actual_kv = torch.full((batch,), cache_tokens, dtype=torch.int32, device=dev)
    hbm_bt_dev = hbm_bt.to(dev)
    dram_bt_dev = dram_bt.to(dev)
    scale = 1.0 / math.sqrt(KVD + KRD)

    def _b_reset():
        """Rebuild cache_slots: fixed miss count, freshly randomized hit/miss set."""
        round_id = _b_reset_round[0]
        _b_reset_round[0] = round_id + 1
        cs_cpu = torch.full((pool, seq_len), -1, dtype=torch.int32)
        other_count = cache_tokens - hit_count
        for r in range(batch):
            pr = int(req_e[r])
            gen = torch.Generator().manual_seed(2000 + round_id * 10007 + r)
            hit_perm = torch.randperm(TOPK, generator=gen)
            hit_tokens_r = topk_tokens[hit_perm[:hit_count]]
            others = non_topk[
                torch.randperm(non_topk.numel(), generator=gen)[:other_count]
            ]
            cached = torch.cat((hit_tokens_r, others))
            if cached.numel() != cache_tokens:
                raise RuntimeError(
                    f"B cache prefill size {cached.numel()} != "
                    f"--cache-tokens={cache_tokens}")
            slots = torch.randperm(cache_tokens, generator=gen, dtype=torch.int32)
            cs_cpu[pr, cached] = slots
            # Sanity: exactly hit_count topk tokens are present for this req.
            present = (cs_cpu[pr, topk_tokens.to(torch.long)] >= 0).sum().item()
            if present != hit_count:
                raise RuntimeError(
                    f"B row {r} cached topk hits={present}, expected {hit_count}")
        cache_slots.copy_(cs_cpu.to(dev))
        src_ids_t.fill_(-1)
        li_dst_slots.fill_(-1)
        miss_counts.fill_(-1)

    def _b_li_manage():
        # LI-part inputs match _b_lightning_indexer (query/key/weights/bt/lens);
        # extra args are the fused LRU/cache-management operands.
        return torch.ops.ops_overlap.li_manage_out.default(
            li_query, idx_cache, weights, req_e_dev, cache_slots,
            cache_tokens_t, candidate_lens, idx_bt,
            li_src_ids, li_dst_slots, miss_counts)

    def _b_lightning_indexer():
        # Same LI inputs as the indexer half of li_manage_out.
        return torch_npu.npu_lightning_indexer(
            query=li_query,
            key=idx_cache,
            weights=weights,
            actual_seq_lengths_query=actual_q,
            actual_seq_lengths_key=candidate_lens,
            block_table=idx_bt,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=TOPK,
            sparse_mode=3,
        )

    def _b_scatter_sfa():
        # Reuse li_manage outputs directly:
        #   sparse_slots      <- destination_slots (topk_slots, miss-first order)
        #   source_token_ids  <- source_ids      (topk_index, miss-first order)
        #   copy_counts       <- miss_counts
        # Do NOT re-gather with a prebuilt/unsorted topk; order must match
        # li_manage's miss-then-hit layout.
        return torch.ops.ops_overlap.sparse_and_tail_attention_and_scatter_copy.default(
            query, hbm_ckv, li_dst_slots, cache_tokens_t, hbm_bt_dev,
            actual_q, actual_kv, qrope, hbm_kpe, dram_kpe, dram_ckv,
            dram_bt_dev, src_ids_t, miss_counts, scale)

    print(
        "[B-cfg] lightning_indexer shares LI inputs with li_manage_out: "
        "query/key/weights/block_table/actual_seq_lengths_* "
        "(delta = li_manage - lightning_indexer ≈ fused LRU cost)",
        flush=True,
    )
    _b_lightning_indexer()
    _b_reset()
    _b_li_manage()
    torch.npu.synchronize()
    actual_miss = miss_counts.cpu().tolist()
    expected_miss = [miss] * batch
    if actual_miss != expected_miss:
        raise RuntimeError(
            f"[B] controlled miss mismatch: got {actual_miss}, "
            f"expected {expected_miss}. Check cache_slots prefill.")
    # li_manage lays out topk as miss-first then hit; slots alias cache_slots.
    src_cpu = src_ids_t.cpu()
    dst_cpu = li_dst_slots[:, 0, :].cpu()
    cs_cpu = cache_slots.cpu()
    for r in range(batch):
        pr = int(req_e[r])
        mapped = cs_cpu[pr, src_cpu[r].to(torch.long)]
        if not torch.equal(mapped, dst_cpu[r]):
            raise RuntimeError(
                f"[B] row {r}: cache_slots[source_ids] != destination_slots; "
                f"scatter must consume li_manage outputs in miss-first order.")
        if int((dst_cpu[r] < 0).sum().item()) != 0:
            raise RuntimeError(f"[B] row {r}: destination_slots contain -1")
    print(
        f"[B-cfg] verified miss_counts={actual_miss}; "
        f"scatter will reuse li_manage source_ids/destination_slots/miss_counts",
        flush=True,
    )
    _b_scatter_sfa()
    torch.npu.synchronize()
    print(f"[B-cfg] timing_mode={'graph' if use_graph else 'eager'}", flush=True)

    def _b_pipeline():
        # Nano production round: LI_MANAGE then scatter/SFA on the same stream.
        _b_li_manage()
        _b_scatter_sfa()

    # Capture after the functional smoke so graph replay measures steady-state
    # kernel time without eager launch/dispatch noise.
    li_runner, li_mode = maybe_graph_runner(
        _b_lightning_indexer, use_graph=use_graph, warmup=max(1, args.warmup // 2))
    li_manage_runner, li_manage_mode = maybe_graph_runner(
        _b_li_manage, use_graph=use_graph,
        warmup=max(1, args.warmup // 2), reset=_b_reset)
    full_runner, full_mode = maybe_graph_runner(
        _b_pipeline, use_graph=use_graph,
        warmup=max(1, args.warmup // 2), reset=_b_reset)

    def _b_reset_then_manage():
        _b_reset()
        _b_li_manage()
        torch.npu.synchronize()

    scatter_runner, scatter_mode = maybe_graph_runner(
        _b_scatter_sfa, use_graph=use_graph,
        warmup=max(1, args.warmup // 2), reset=_b_reset_then_manage)

    def _b_report_timing(
        lightning_indexer_ms: float,
        li_manage_ms: float,
        sfa_ms: float,
    ) -> float:
        delta_ms = li_manage_ms - lightning_indexer_ms
        print(
            f"[B-timing] lightning_indexer={lightning_indexer_ms:.4f} ms  "
            f"li_manage={li_manage_ms:.4f} ms  "
            f"delta(li_manage-LI / lru)={delta_ms:+.4f} ms  "
            f"scatter_sfa(sfa+copy)={sfa_ms:.4f} ms",
            flush=True,
        )
        return delta_ms

    profile_summary = None
    if getattr(args, "profile", False):
        # Profile LI and li_manage as separate named ranges (same LI inputs),
        # then scatter. No host-LRU planner summary for B.
        profile_summary = profile_pipeline(
            [
                ("lightning_indexer", li_runner),
                ("li_manage", li_manage_runner),
                ("scatter_sfa", scatter_runner),
            ],
            args.warmup, args.profile_iters, args.profile_dir,
            top_n=args.profile_top_n,
            reset=_b_reset,
            drop_first=args.profile_drop_first,
            include_lru_planner=False)

    # Always event-time comparable metrics for --out / --compare:
    #   lru_ms      = li_manage - LI  ↔  A planner_ms
    #   sfa_copy_ms = scatter_sfa    ↔  A fused_overlap_ms
    lightning_indexer_ms = bench_events(li_runner, args.warmup, args.iters)
    li_manage_ms = bench_events(
        li_manage_runner, args.warmup, args.iters, reset=_b_reset)
    sfa_ms = bench_events(
        scatter_runner, args.warmup, args.iters, reset=_b_reset_then_manage)
    delta_ms = _b_report_timing(lightning_indexer_ms, li_manage_ms, sfa_ms)

    result = {
        "approach": "B",
        "timing_mode": full_mode,
        "batch_size": batch, "q_heads": q_heads, "indexer_heads": idx_heads,
        "seq_len": seq_len, "topk_layout": topk_layout,
        "cache_tokens": cache_tokens, "miss": miss,
        "lightning_indexer_ms": lightning_indexer_ms,
        "li_manage_ms": li_manage_ms,
        "li_manage_minus_li_ms": delta_ms,
        "scatter_sfa_ms": sfa_ms,
        "total_ms": li_manage_ms + sfa_ms,
        "warmup": args.warmup,
        "iters": args.iters,
        "li_mode": li_mode,
        "li_manage_mode": li_manage_mode,
        "scatter_mode": scatter_mode,
    }
    attach_compare_fields(result, lru_ms=delta_ms, sfa_copy_ms=sfa_ms)
    if profile_summary is not None:
        result["profile"] = True
        result["profile_run_dir"] = profile_summary.get("run_dir")
        result["profile_dir"] = profile_summary.get("profile_dir")
        result["profile_summary"] = profile_summary
        attach_kernel_compare_fields(result, "B")
    return result


# ===================== Multi-scenario suite / Compare =====================
def scenario_key(result: dict) -> tuple:
    """Identity used to align A/B scenarios for compare."""
    cache_tokens = result.get("cache_tokens", result.get("cache_tokens_req", -1))
    return (
        int(result["batch_size"]),
        int(result["miss"]),
        int(result.get("seq_len", -1)),
        int(cache_tokens),
        str(result.get("topk_layout", "contiguous")),
        int(result.get("q_heads", -1)),
    )


def load_scenarios(path: str) -> tuple[dict, list[dict]]:
    with open(path) as f:
        payload = json.load(f)
    if isinstance(payload, dict) and isinstance(payload.get("scenarios"), list):
        return payload, payload["scenarios"]
    if isinstance(payload, dict):
        return payload, [payload]
    raise ValueError(f"Unsupported result JSON format in {path}")


def run_scenario_suite(args, runner) -> dict:
    """Run ``runner`` over the cartesian product of sweep knobs."""
    batch_sizes = [int(x) for x in args.batch_size]
    misses = [int(x) for x in args.miss]
    seq_lens = [int(x) for x in args.seq_len]
    cache_tokens_list = [int(x) for x in args.cache_tokens]
    if not batch_sizes:
        raise ValueError("--batch-size requires at least one value")
    if not misses:
        raise ValueError("--miss requires at least one value")
    if not seq_lens:
        raise ValueError("--seq-len requires at least one value")
    if not cache_tokens_list:
        raise ValueError("--cache-tokens requires at least one value")
    for m in misses:
        if m < 0 or m > TOPK:
            raise ValueError(f"--miss values must be in [0, {TOPK}], got {m}")
    for sl in seq_lens:
        if sl <= 0:
            raise ValueError(f"--seq-len values must be > 0, got {sl}")
    for ct in cache_tokens_list:
        if ct <= 0:
            raise ValueError(f"--cache-tokens values must be > 0, got {ct}")

    n_scenarios = (
        len(batch_sizes) * len(misses) * len(seq_lens) * len(cache_tokens_list)
    )
    print(
        f"[suite] approach={args.approach} batch_sizes={batch_sizes} "
        f"misses={misses} seq_lens={seq_lens} "
        f"cache_tokens={cache_tokens_list}  ({n_scenarios} scenarios)",
        flush=True,
    )
    base_profile_dir = args.profile_dir
    if getattr(args, "profile", False) and not base_profile_dir:
        base_profile_dir = tempfile.mkdtemp(prefix="fused_overlap_suite_prof_")
        print(f"[suite] no --profile-dir given, using {base_profile_dir}",
              flush=True)
    scenarios: list[dict] = []
    for bs in batch_sizes:
        for miss in misses:
            for seq_len in seq_lens:
                for cache_tokens in cache_tokens_list:
                    print(
                        f"\n===== scenario approach={args.approach} "
                        f"batch_size={bs} miss={miss} "
                        f"seq_len={seq_len} cache_tokens={cache_tokens} =====",
                        flush=True,
                    )
                    args.batch_size = bs
                    args.miss = miss
                    args.seq_len = seq_len
                    args.cache_tokens = cache_tokens
                    if getattr(args, "profile", False):
                        # Always isolate profiles by the full scenario key.
                        args.profile_dir = os.path.join(
                            base_profile_dir,
                            f"bs{bs}_miss{miss}_seq{seq_len}_ct{cache_tokens}",
                        )
                    result = runner(args)
                    # Normalize cache_tokens field so A/B compare keys align.
                    if "cache_tokens" not in result and "cache_tokens_req" in result:
                        result["cache_tokens"] = result["cache_tokens_req"]
                    scenarios.append(result)
                    gc.collect()
                    try:
                        torch.npu.empty_cache()
                    except Exception:
                        pass

    args.batch_size = batch_sizes
    args.miss = misses
    args.seq_len = seq_lens
    args.cache_tokens = cache_tokens_list
    args.profile_dir = base_profile_dir
    return {
        "approach": args.approach,
        "suite": True,
        "batch_sizes": batch_sizes,
        "misses": misses,
        "seq_lens": seq_lens,
        "cache_tokens_list": cache_tokens_list,
        "topk_layout": args.topk_layout,
        "q_heads": args.q_heads,
        "scenarios": scenarios,
    }


def _fmt_ms(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _scenario_lru_ms(result: dict) -> float | None:
    """A planner / B (li_manage - LI) — event time."""
    if result.get("lru_ms") is not None:
        return float(result["lru_ms"])
    if result.get("approach") == "A" and result.get("planner_ms") is not None:
        return float(result["planner_ms"])
    if result.get("li_manage_minus_li_ms") is not None:
        return float(result["li_manage_minus_li_ms"])
    if result.get("li_manage_ms") is not None \
            and result.get("lightning_indexer_ms") is not None:
        return float(result["li_manage_ms"]) - float(result["lightning_indexer_ms"])
    return None


def _scenario_sfa_copy_ms(result: dict) -> float | None:
    """A fused_overlap / B scatter_sfa — event time."""
    if result.get("sfa_copy_ms") is not None:
        return float(result["sfa_copy_ms"])
    if result.get("fused_overlap_ms") is not None:
        return float(result["fused_overlap_ms"])
    if result.get("scatter_sfa_ms") is not None:
        return float(result["scatter_sfa_ms"])
    return None


def _ensure_kernel_compare_fields(result: dict) -> None:
    """Fill kernel compare fields in-place from profile_summary if missing."""
    if result.get("lru_kernel_ms") is not None \
            and result.get("sfa_copy_kernel_ms") is not None:
        return
    if not result.get("profile_summary"):
        return
    approach = str(result.get("approach", "")).upper()
    if approach in ("A", "B"):
        attach_kernel_compare_fields(result, approach, quiet=True)


def _scenario_lru_kernel_ms(result: dict) -> float | None:
    """A: HOSTFUNC_CALLBACK + EVENT_WAIT (sum). B: li_manage_k - LI_k."""
    _ensure_kernel_compare_fields(result)
    if result.get("lru_kernel_ms") is not None:
        return float(result["lru_kernel_ms"])
    return None


def _scenario_sfa_copy_kernel_ms(result: dict) -> float | None:
    """A fused_overlap kernel / B scatter_sfa kernel."""
    _ensure_kernel_compare_fields(result)
    if result.get("sfa_copy_kernel_ms") is not None:
        return float(result["sfa_copy_kernel_ms"])
    return None


def _print_pair_delta(label: str, a_ms, b_ms) -> None:
    if a_ms is None or b_ms is None:
        print(f"{label}: A={_fmt_ms(a_ms)}  B={_fmt_ms(b_ms)}  delta=n/a",
              flush=True)
        return
    diff = float(a_ms) - float(b_ms)
    who = "A faster" if diff < 0 else "B faster"
    print(
        f"{label}: A={_fmt_ms(a_ms)}  B={_fmt_ms(b_ms)}  "
        f"A-B={diff:+.4f} ms ({who})",
        flush=True,
    )


def _print_scenario_compare(ra: dict, rb: dict) -> None:
    key = scenario_key(ra)
    print(
        f"----- SCENARIO batch={key[0]} miss={key[1]} "
        f"seq_len={key[2]} cache_tokens={key[3]} "
        f"topk_layout={key[4]} q_heads={key[5]} -----",
        flush=True,
    )
    print(
        f"Timing mode: A={ra.get('timing_mode', 'unknown')} "
        f"B={rb.get('timing_mode', 'unknown')}"
        f"{'  (profile+event)' if ra.get('profile') or rb.get('profile') else ''}",
        flush=True,
    )

    a_lru = _scenario_lru_ms(ra)
    b_lru = _scenario_lru_ms(rb)
    a_sfa = _scenario_sfa_copy_ms(ra)
    b_sfa = _scenario_sfa_copy_ms(rb)
    print("--- EVENT TIME ---", flush=True)
    _print_pair_delta("lru_ms      (A planner ↔ B LI_MANAGE-LI)", a_lru, b_lru)
    _print_pair_delta("sfa_copy_ms (A fused  ↔ B scatter_sfa)", a_sfa, b_sfa)

    a_lru_k = _scenario_lru_kernel_ms(ra)
    b_lru_k = _scenario_lru_kernel_ms(rb)
    a_sfa_k = _scenario_sfa_copy_kernel_ms(ra)
    b_sfa_k = _scenario_sfa_copy_kernel_ms(rb)
    print("--- KERNEL / PROFILE TIME ---", flush=True)
    if all(v is None for v in (a_lru_k, b_lru_k, a_sfa_k, b_sfa_k)):
        print(
            "kernel times n/a (re-run A/B with --profile so out JSON "
            "contains profile_summary / lru_kernel_ms / sfa_copy_kernel_ms)",
            flush=True,
        )
    else:
        _print_pair_delta(
            "lru_kernel_ms      (A HOSTFUNC_CALLBACK+EVENT_WAIT ↔ B LI_MANAGE_k-LI_k)",
            a_lru_k, b_lru_k,
        )
        if ra.get("hostfunc_callback_kernel_ms") is not None \
                or ra.get("event_wait_kernel_ms") is not None:
            print(
                f"  A lru parts: HOSTFUNC_CALLBACK="
                f"{_fmt_ms(ra.get('hostfunc_callback_kernel_ms'))} + "
                f"EVENT_WAIT={_fmt_ms(ra.get('event_wait_kernel_ms'))} = "
                f"{_fmt_ms(a_lru_k)}",
                flush=True,
            )
        _print_pair_delta(
            "sfa_copy_kernel_ms (A fused_k ↔ B scatter_k)",
            a_sfa_k, b_sfa_k,
        )
        if ra.get("kernel_match_names") or rb.get("kernel_match_names"):
            print(
                f"  matched A={ra.get('kernel_match_names')}  "
                f"B={rb.get('kernel_match_names')}",
                flush=True,
            )

    print(
        f"Approach A detail: LI={_fmt_ms(ra.get('li_ms'))}  "
        f"planner={_fmt_ms(ra.get('planner_ms', a_lru))}  "
        f"fused_overlap={_fmt_ms(ra.get('fused_overlap_ms', a_sfa))}  "
        f"pipeline={_fmt_ms(ra.get('pipeline_ms'))}  "
        f"total={_fmt_ms(ra.get('total_ms'))}",
        flush=True,
    )
    print(
        f"Approach B detail: LI={_fmt_ms(rb.get('lightning_indexer_ms'))}  "
        f"LI_MANAGE={_fmt_ms(rb.get('li_manage_ms'))}  "
        f"delta(LI_MANAGE-LI)={_fmt_ms(b_lru)}  "
        f"scatter_sfa={_fmt_ms(rb.get('scatter_sfa_ms', b_sfa))}  "
        f"total={_fmt_ms(rb.get('total_ms'))}",
        flush=True,
    )
    if ra.get("total_ms") is None or rb.get("total_ms") is None:
        print("Delta total event (A-B): n/a (missing total_ms)", flush=True)
        return
    diff = float(ra["total_ms"]) - float(rb["total_ms"])
    who = "A faster" if diff < 0 else "B faster"
    print(
        f"Delta total event (A-B): {diff:+.4f} ms  "
        f"({who} by {abs(diff):.4f} ms)",
        flush=True,
    )


def compare(args) -> None:
    _, scenarios_a = load_scenarios(args.out_a)
    _, scenarios_b = load_scenarios(args.out_b)
    map_a = {scenario_key(s): s for s in scenarios_a}
    map_b = {scenario_key(s): s for s in scenarios_b}
    common = sorted(set(map_a) & set(map_b))
    only_a = sorted(set(map_a) - set(map_b))
    only_b = sorted(set(map_b) - set(map_a))

    print("----- TIMING SUMMARY (ms, mean over iters) -----", flush=True)
    print(
        f"Scenarios: A={len(scenarios_a)} B={len(scenarios_b)} "
        f"common={len(common)}",
        flush=True,
    )
    if only_a:
        print(f"[compare] only in A: {only_a}", flush=True)
    if only_b:
        print(f"[compare] only in B: {only_b}", flush=True)
    if not common:
        raise ValueError(
            "No matching scenarios between --out-a and --out-b. "
            "Match key is (batch_size, miss, seq_len, cache_tokens, "
            "topk_layout, q_heads).")

    summary_rows: list[tuple] = []
    for key in common:
        ra = map_a[key]
        rb = map_b[key]
        _print_scenario_compare(ra, rb)
        summary_rows.append((
            key[0], key[1], key[2], key[3],
            _scenario_lru_ms(ra), _scenario_lru_ms(rb),
            _scenario_sfa_copy_ms(ra), _scenario_sfa_copy_ms(rb),
            _scenario_lru_kernel_ms(ra), _scenario_lru_kernel_ms(rb),
            _scenario_sfa_copy_kernel_ms(ra), _scenario_sfa_copy_kernel_ms(rb),
            ra.get("total_ms"), rb.get("total_ms"),
        ))

    print("\n----- COMPARE TABLE (EVENT ms) -----", flush=True)
    print(
        f"{'batch':>6} {'miss':>6} {'seq':>8} {'cache':>8} "
        f"{'A_lru':>8} {'B_lru':>8} {'A_sfa':>8} {'B_sfa':>8} "
        f"{'A_tot':>8} {'B_tot':>8}",
        flush=True,
    )
    for row in summary_rows:
        (batch, miss, seq_len, cache_tokens,
         a_lru, b_lru, a_sfa, b_sfa, *_rest) = row
        a_total, b_total = row[-2], row[-1]
        print(
            f"{batch:6d} {miss:6d} {seq_len:8d} {cache_tokens:8d} "
            f"{_fmt_ms(a_lru):>8} {_fmt_ms(b_lru):>8} "
            f"{_fmt_ms(a_sfa):>8} {_fmt_ms(b_sfa):>8} "
            f"{_fmt_ms(a_total):>8} {_fmt_ms(b_total):>8}",
            flush=True,
        )

    print("\n----- COMPARE TABLE (KERNEL ms) -----", flush=True)
    print(
        f"{'batch':>6} {'miss':>6} {'seq':>8} {'cache':>8} "
        f"{'A_lru_k':>8} {'B_lru_k':>8} {'A_sfa_k':>8} {'B_sfa_k':>8}",
        flush=True,
    )
    for row in summary_rows:
        (batch, miss, seq_len, cache_tokens,
         _a_lru, _b_lru, _a_sfa, _b_sfa,
         a_lru_k, b_lru_k, a_sfa_k, b_sfa_k, _a_tot, _b_tot) = row
        print(
            f"{batch:6d} {miss:6d} {seq_len:8d} {cache_tokens:8d} "
            f"{_fmt_ms(a_lru_k):>8} {_fmt_ms(b_lru_k):>8} "
            f"{_fmt_ms(a_sfa_k):>8} {_fmt_ms(b_sfa_k):>8}",
            flush=True,
        )
    print("UT_OK", flush=True)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--approach", choices=("A", "B"), default=None,
                   help="Run one approach in this process (writes --out).")
    p.add_argument("--compare", action="store_true",
                   help="Compare --out-a and --out-b JSON results "
                        "(matches all common batch×miss×seq×cache scenarios).")
    p.add_argument("--out", default=None, help="Output JSON path for --approach.")
    p.add_argument("--out-a", default="result_a.json")
    p.add_argument("--out-b", default="result_b.json")
    p.add_argument("--nano-path", default=_default_nano_path(),
                   help="Path to nano torch_extension dir (contains ops_overlap/). "
                        "Override via $NANO_PATH env or --nano-path.")
    p.add_argument(
        "--batch-size", type=int, nargs="+", default=[24],
        help="One or more batch sizes to sweep, e.g. --batch-size 8 16 24.",
    )
    p.add_argument("--tp", type=int, default=16,
                   help="Tensor parallel size. q_heads = NUM_ATTENTION_HEADS / tp "
                        "(GLM-5.2: 64 / 16 = 4).")
    p.add_argument("--q-heads", type=int, default=None,
                   help="Attention heads per rank. Overrides --tp derivation. "
                        "Default: 64 / tp.")
    p.add_argument("--indexer-heads", type=int, default=32,
                   help="Indexer heads (GLM-5.2 index_n_heads=32).")
    p.add_argument(
        "--seq-len", type=int, nargs="+", default=[65536],
        help="One or more sequence lengths to sweep, e.g. --seq-len 32768 65536.",
    )
    p.add_argument(
        "--cache-tokens", type=int, nargs="+", default=[8192],
        help="One or more resident-token counts to sweep, e.g. "
             "--cache-tokens 4096 8192. Used by both A and B. Approach A caps "
             "at resident_capacity=TOPK*2-1 (physical host LRU buffer size).",
    )
    p.add_argument(
        "--topk-layout",
        choices=TOPK_LAYOUTS,
        default="contiguous",
        help=(
            "Physical top-k distribution: contiguous selects the final TOPK "
            "tokens; random scatters them with a deterministic permutation."
        ),
    )
    p.add_argument(
        "--miss", type=int, nargs="+", default=[300],
        help="One or more miss counts per row (0..TOPK), e.g. --miss 100 300 800. "
             "Combined with --batch-size/--seq-len/--cache-tokens as a "
             "cartesian scenario matrix. Each timed/profile iter rebuilds "
             "residents with a freshly randomized hit/miss partition.",
    )
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--dram-size-gb", type=float, default=2.0)
    p.add_argument(
        "--graph-mode",
        dest="graph_mode",
        action="store_true",
        default=True,
        help="Measure NPUGraph capture/replay times (default). Avoids eager "
             "dispatch inflating host-callback / LRU planner latency.",
    )
    p.add_argument(
        "--eager",
        dest="graph_mode",
        action="store_false",
        help="Disable NPUGraph and measure eager launch times instead.",
    )
    p.add_argument("--profile", action="store_true",
                   help="Profile the full pipeline with torch_npu.profiler (msprof) "
                        "and print a kernel-level op table instead of Event timing.")
    p.add_argument("--profile-iters", type=int, default=5,
                   help="Iterations to capture under --profile.")
    p.add_argument("--profile-dir", default=None,
                   help="Directory to export the full msprof trace to (optional).")
    p.add_argument("--profile-top-n", type=int, default=30,
                   help="How many top kernels/ops to print after --profile.")
    p.add_argument(
        "--profile-drop-first",
        type=int,
        default=1,
        help="When averaging LRU timeline markers from trace_view.json, drop "
             "this many leading samples per marker (default 1). The first "
             "profiled EVENT_WAIT is often a cold outlier.",
    )
    p.add_argument(
        "--parse-profile-dir",
        default=None,
        help="Only parse an existing msprof/torch_npu profile directory and "
             "print average kernel times; do not run Approach A/B.",
    )
    p.add_argument(
        "--include-lru-planner",
        dest="include_lru_planner",
        action="store_true",
        default=None,
        help="When parsing a profile dir, also print Approach-A LRU planner "
             "timeline markers (HOSTFUNC_CALLBACK + EVENT_WAIT).",
    )
    p.add_argument(
        "--no-lru-planner",
        dest="include_lru_planner",
        action="store_false",
        help="When parsing a profile dir, skip LRU planner timeline summary "
             "(use for Approach B profiles).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.parse_profile_dir is not None:
        # Default: include LRU planner markers unless explicitly disabled.
        include_lru = True if args.include_lru_planner is None else args.include_lru_planner
        summarize_profile_csvs(
            args.parse_profile_dir,
            top_n=args.profile_top_n,
            drop_first=args.profile_drop_first,
            include_lru_planner=include_lru,
        )
        return
    if args.q_heads is None:
        if NUM_ATTENTION_HEADS % args.tp != 0:
            raise SystemExit(
                f"NUM_ATTENTION_HEADS={NUM_ATTENTION_HEADS} not divisible by "
                f"--tp={args.tp}. Use --q-heads to override.")
        args.q_heads = NUM_ATTENTION_HEADS // args.tp
        print(f"[cfg] tp={args.tp}  q_heads={args.q_heads} "
              f"(={NUM_ATTENTION_HEADS}/{args.tp})", flush=True)
    if args.compare:
        compare(args)
        return
    if args.approach is None:
        raise SystemExit("Use --approach A|B or --compare. See --help.")
    runner = run_a if args.approach == "A" else run_b
    result = run_scenario_suite(args, runner)
    out = args.out or f"result_{args.approach.lower()}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    n = len(result["scenarios"])
    print(
        f"RESULT approach={args.approach} out={out} "
        f"scenarios={n} batch_sizes={result['batch_sizes']} "
        f"misses={result['misses']} seq_lens={result['seq_lens']} "
        f"cache_tokens={result['cache_tokens_list']}",
        flush=True,
    )
    for sc in result["scenarios"]:
        cache_tokens = sc.get("cache_tokens", sc.get("cache_tokens_req"))
        print(
            f"  - batch={sc['batch_size']} miss={sc['miss']} "
            f"seq_len={sc['seq_len']} cache_tokens={cache_tokens} "
            f"lru_ms={_fmt_ms(sc.get('lru_ms'))} "
            f"sfa_copy_ms={_fmt_ms(sc.get('sfa_copy_ms'))} "
            f"lru_kernel_ms={_fmt_ms(sc.get('lru_kernel_ms'))} "
            f"sfa_copy_kernel_ms={_fmt_ms(sc.get('sfa_copy_kernel_ms'))} "
            f"total_ms={_fmt_ms(sc.get('total_ms'))}"
            f"{' profile=True' if sc.get('profile') else ''}",
            flush=True,
        )
    print("UT_OK", flush=True)


if __name__ == "__main__":
    main()
