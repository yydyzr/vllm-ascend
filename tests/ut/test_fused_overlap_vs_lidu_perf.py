"""Perf comparison: vllm-ascend fused_overlap vs nano LI_MANAGE+scatter_sfa.

Two OPP packages cannot coexist in one process, so run one approach per
process (writes JSON), then --compare reads both.

A: LI + host-side LRU planner + fused_overlap(copy+sfa in one kernel,
   consuming the host-encoded external plan).
B: LightningIndexer baseline + LI_MANAGE(li+lru fused) +
   scatter_copy_sfa(copy+sfa fused).

Timing defaults to NPUGraph capture/replay (``--graph-mode``). Use
``--eager`` only for debugging launch/dispatch overhead. Approach A's planner
follows the production side-stream + C++ ``enqueue_lru_*`` host-callback path.

``--miss`` fixes the per-row miss count for both A and B. Each warmup /
timed / profile iteration rebuilds residents with a freshly randomized
hit/miss partition among the same topk set, so miss stays fixed but the
concrete hit/miss tokens change every round.

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
    """Return the controlled per-row miss count from CLI.

    ``--miss`` is the source of truth. Deprecated ``--hit-ratio`` still
    overrides when explicitly provided.
    """
    if getattr(args, "hit_ratio", None) is not None:
        miss = int(round(TOPK * (1.0 - float(args.hit_ratio))))
    else:
        miss = int(args.miss)
    if miss < 0 or miss > TOPK:
        raise ValueError(f"--miss must be in [0, {TOPK}], got {miss}")
    return miss


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
# markers. Their avg sum is the planner/LRU proxy used in this UT.
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
) -> None:
    """Average HOSTFUNC_CALLBACK + EVENT_WAIT as Approach-A LRU planner time."""
    samples, source = _collect_lru_marker_samples(artifacts)

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
            return
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
            print(
                f"{name:<48} {count:8d} {avg_us:12.3f} {total_us:14.3f}",
                flush=True,
            )
        return

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
        print(
            f"{marker:<24} {stats['count']:4d} "
            f"{stats['first_us']:10.3f} {stats['avg_us']:10.3f} "
            f"{stats['steady_avg_us']:12.3f} {stats['median_us']:10.3f} "
            f"{stats['min_us']:10.3f} {stats['max_us']:10.3f}",
            flush=True,
        )

    if len(steady_avg) == len(_LRU_PROFILE_MARKERS):
        lru_steady = sum(steady_avg.values())
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


def summarize_profile_csvs(
    profile_dir: str,
    top_n: int = 30,
    *,
    drop_first: int = 1,
) -> None:
    """Print average kernel / op times and LRU timeline markers."""
    found = _ensure_profile_artifacts(profile_dir)
    if not found:
        print("[profile] no profile artifacts found under "
              f"{profile_dir}", flush=True)
        return

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
            "[profile] no op_statistic/kernel_details; still trying LRU "
            "markers from timeline/runtime exports.",
            flush=True,
        )
    _print_lru_planner_summary(found, rows, drop_first=drop_first)


def profile_pipeline(steps, warmup, iters, profile_dir=None, top_n: int = 30,
                     reset=None, drop_first: int = 1):
    """Run the full pipeline under torch_npu.profiler (msprof backend).

    ``steps`` is a list of ``(name, callable)`` executed in order each iter.
    Prefer a single step that represents one production round so kernels are
    not duplicated and stay on the intended streams.
    ``reset`` runs outside the timed/profiled work each iteration.

    Each invocation writes into a fresh ``run_<timestamp>_<pid>/`` subdirectory
    under ``profile_dir``, so parsing never picks a stale ``op_statistic.csv``
    left by a previous run that reused the same ``--profile-dir``.
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
    summarize_profile_csvs(str(run_dir), top_n=top_n, drop_first=drop_first)


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

    cfg = offload.OffloadConfig()
    cfg.device_id = torch_npu.npu.current_device()
    cfg.size = int(args.dram_size_gb * _GIB)
    cfg.world_size = 1
    cfg.rank_id = 0
    offload.initialize(cfg)

    dt = torch.bfloat16
    batch = args.batch_size
    q_heads = args.q_heads
    idx_heads = args.indexer_heads
    seq_len = args.seq_len
    topk_layout = args.topk_layout
    miss = resolve_miss_count(args)
    hit_count = TOPK - miss
    hit_ratio = 1.0 - (miss / float(TOPK))
    fmbn = cdiv(seq_len, BLOCK_SIZE)
    smbn = cdiv(TOPK, BLOCK_SIZE)
    scale = 1.0 / math.sqrt(KVD)
    torch.manual_seed(910000 + batch + q_heads * 17 + seq_len)
    print(
        f"[A-cfg] batch={batch} q_heads={q_heads} idx_heads={idx_heads} "
        f"seq_len={seq_len} topk_layout={topk_layout} miss={miss} "
        f"hit_count={hit_count} hit_ratio={hit_ratio:.4f}",
        flush=True,
    )

    q = torch.randn(batch, q_heads, KVD, dtype=dt, device="npu")
    qr = torch.randn(batch, q_heads, KRD, dtype=dt, device="npu")
    q_fused = torch.cat([q, qr], dim=-1).contiguous()
    full_kv = _empty_cpu([fmbn, BLOCK_SIZE, KVD], dt)
    full_kv.zero_()
    full_rope = _empty_cpu([fmbn, BLOCK_SIZE, KRD], dt)
    full_rope.zero_()
    full_bt = torch.arange(fmbn, dtype=torch.int32, device="npu").unsqueeze(0).expand(batch, -1).contiguous()
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
    fill_count = min(int(args.cache_tokens), resident_capacity)
    if fill_count < hit_count:
        raise ValueError(
            f"Need at least hit_count={hit_count} resident slots to honor "
            f"--miss={miss}, but fill_count={fill_count} "
            f"(cache_tokens={args.cache_tokens}, "
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
    # Production decode uses ~seq_len-1; use seq_len so no resident is wiped
    # as a speculative suffix while we measure controlled miss counts.
    stable_prefix_npu = torch.full((batch,), seq_len, dtype=torch.int32, device="npu")
    visible_seq_npu = torch.full((batch,), seq_len, dtype=torch.int32, device="npu")
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

    def _copy_planner_inputs_d2h():
        # Match production: D2H from NPU tensors on the current stream.
        lru_topk_indices.copy_(topk_flat, non_blocking=True)
        lru_req_ids.copy_(req_ids_npu, non_blocking=True)
        lru_stable_prefix_lens.copy_(stable_prefix_npu, non_blocking=True)
        lru_visible_seq_lens.copy_(visible_seq_npu, non_blocking=True)

    def _a_reset_lru_state():
        """Rebuild residents for one round: fixed miss count, fresh hit/miss set."""
        round_id = _a_reset_round[0]
        _a_reset_round[0] = round_id + 1
        lru_slot_to_token.fill_(-1)
        for r in range(batch):
            gen = torch.Generator().manual_seed(1000 + round_id * 10007 + r)
            hit_perm = torch.randperm(TOPK, generator=gen)
            hit_tokens = topk_tokens_cpu[hit_perm[:hit_count]]
            other_count = fill_count - hit_count
            others = non_topk[
                torch.randperm(non_topk.numel(), generator=gen)[:other_count]
            ]
            cached = torch.cat((hit_tokens, others))
            slots = torch.randperm(resident_capacity, generator=gen)[:fill_count]
            lru_slot_to_token[r, slots] = cached
        lru_slots.copy_(lru_slots_init)
        lru_last_req_ids.copy_(lru_last_req_ids_init)
        lru_epochs.zero_()
        membership_map.fill_(-1)
        control[:, 1] = EXTERNAL_PLAN_READY_MARKER
        control[:, 2] = TOPK
        control[:, 3] = CONTROL_OFFSET_INT16 - TOPK
        control[:, 7] = PAIRED_SELECTION_COPY_MARKER
        sel_status.fill_(-1)

    def _a_planner_sync():
        """Synchronous planner — used for miss verification / fused prep."""
        lru_topk_indices.copy_(topk_flat)
        lru_req_ids.copy_(req_ids_npu)
        lru_stable_prefix_lens.copy_(stable_prefix_npu)
        lru_visible_seq_lens.copy_(visible_seq_npu)
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
        # Production order on the compute stream: wait for side-stream LRU,
        # then fused. LI and fused both stay on the compute stream.
        _a_planner()
        torch_npu.npu.current_stream().wait_stream(plan_stream)
        _a_fused()

    def _a_full_pipeline():
        # One capture/replay unit for profiling: LI and fused share the
        # compute stream; only LRU host-callback uses plan_stream.
        _a_li()
        _a_planner_plus_fused()

    # Eager functional smoke + verify controlled miss before capture/timing.
    _a_reset_lru_state()
    _a_planner_sync()
    actual_miss = lru_miss_count[:batch].tolist()
    expected_miss = [miss] * batch
    if actual_miss != expected_miss:
        raise RuntimeError(
            f"[A] controlled miss mismatch: got {actual_miss}, "
            f"expected {expected_miss}. Check resident prefill / "
            f"stable_prefix_lens.")
    print(f"[A-cfg] verified miss_counts={actual_miss}", flush=True)
    _a_fused()
    torch.npu.synchronize()
    print(f"[A-cfg] timing_mode={'graph' if use_graph else 'eager'}", flush=True)

    # Profile must capture ONE full pipeline. Capturing LI / fused as separate
    # graphs previously made them look like different streams and caused fused
    # to execute twice per profile iter (pipeline + isolated fused).
    full_runner, full_mode = maybe_graph_runner(
        _a_full_pipeline, use_graph=use_graph,
        warmup=max(1, args.warmup // 2), reset=_a_reset_lru_state)

    if getattr(args, "profile", False):
        profile_pipeline(
            [("full_pipeline", full_runner)],
            args.warmup, args.profile_iters, args.profile_dir,
            top_n=args.profile_top_n,
            reset=_a_reset_lru_state,
            drop_first=args.profile_drop_first)
        try:
            offload.uninitialize()
        except Exception as exc:
            print(f"OFFLOAD_UNINIT_WARN {type(exc).__name__}: {exc}", flush=True)
        return {
            "approach": "A", "profile": True,
            "timing_mode": full_mode,
            "batch_size": batch, "q_heads": q_heads, "indexer_heads": idx_heads,
            "seq_len": seq_len, "topk_layout": topk_layout,
            "miss": miss, "hit_ratio": hit_ratio,
            "warmup": args.warmup, "iters": args.profile_iters,
        }

    # Event timing still breaks components out with separate graphs/runners.
    li_runner, li_mode = maybe_graph_runner(
        _a_li, use_graph=use_graph, warmup=max(1, args.warmup // 2))
    pipeline_runner, pipeline_mode = maybe_graph_runner(
        _a_planner_plus_fused, use_graph=use_graph,
        warmup=max(1, args.warmup // 2), reset=_a_reset_lru_state)
    fused_runner, fused_mode = maybe_graph_runner(
        _a_fused, use_graph=use_graph, warmup=max(1, args.warmup // 2),
        reset=_a_reset_then_plan)

    li_ms = bench_events(li_runner, args.warmup, args.iters)
    pipeline_ms = bench_events(
        pipeline_runner, args.warmup, args.iters, reset=_a_reset_lru_state)
    fused_ms = bench_events(
        fused_runner, args.warmup, args.iters, reset=_a_reset_then_plan)
    planner_ms = max(0.0, pipeline_ms - fused_ms)
    try:
        offload.uninitialize()
    except Exception as exc:
        print(f"OFFLOAD_UNINIT_WARN {type(exc).__name__}: {exc}", flush=True)

    return {
        "approach": "A",
        "timing_mode": full_mode,
        "batch_size": batch, "q_heads": q_heads, "indexer_heads": idx_heads,
        "seq_len": seq_len, "topk_layout": topk_layout,
        "miss": miss, "hit_ratio": hit_ratio,
        "li_ms": li_ms, "planner_ms": planner_ms, "fused_overlap_ms": fused_ms,
        "pipeline_ms": pipeline_ms,
        "total_ms": li_ms + pipeline_ms,
        "warmup": args.warmup, "iters": args.iters,
        "li_mode": li_mode, "pipeline_mode": pipeline_mode, "fused_mode": fused_mode,
    }


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
    print(f"[B-cfg] batch={batch} q_heads={q_heads} idx_heads={idx_heads} "
          f"seq_len={seq_len} topk_layout={topk_layout} "
          f"cache_tokens={cache_tokens} miss={miss}",
          flush=True)
    torch.manual_seed(910000 + batch + q_heads * 17 + seq_len + 1)
    cb = math.ceil(cache_tokens / BLOCK_SIZE)
    sb = seq_len // BLOCK_SIZE

    hbm_bt = torch.empty((batch, cb), dtype=torch.int32)
    for r in range(batch):
        ph = torch.arange(r * cb, (r + 1) * cb, dtype=torch.int32)
        hbm_bt[r] = ph[torch.randperm(cb)]
    dram_bt = torch.stack([torch.randperm(sb, dtype=torch.int64).to(torch.int32) for _ in range(batch)])

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
    hit_count = TOPK - miss
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
    dk = torch.randn((sb, BLOCK_SIZE, KRD), generator=g, dtype=torch.float32).to(dt)
    dc = torch.randn((sb, BLOCK_SIZE, KVD), generator=g, dtype=torch.float32).to(dt)
    dram_kpe = _swap(dk, dev)
    dram_ckv = _swap(dc, dev)
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
    # src_ids and miss_counts are OUTPUTS of li_manage_out, pre-allocate with -1
    src_ids_t = torch.full((batch, TOPK), -1, dtype=torch.int32, device=dev)
    li_src_ids = src_ids_t.unsqueeze(1)
    li_dst_slots = torch.empty_like(li_src_ids)
    miss_counts = torch.empty((batch,), dtype=torch.int32, device=dev)
    # sparse_slots: physical slot positions for top-k tokens, extracted
    # from cache_slots AFTER li_manage updates it with miss destinations
    use_graph = bool(getattr(args, "graph_mode", True))
    topk_tokens_dev = topk_tokens.to(dev)
    req_e_dev = req_e.to(dev)
    req_e_long = req_e_dev.to(torch.long)
    topk_long = topk_tokens_dev.to(torch.long)
    sparse_slots = torch.empty((batch, 1, TOPK), dtype=torch.int32, device=dev)
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
            slots = torch.randperm(cache_tokens, generator=gen, dtype=torch.int32)
            cs_cpu[pr, cached] = slots
        cache_slots.copy_(cs_cpu.to(dev))
        src_ids_t.fill_(-1)

    def _b_extract_slots():
        # Device-only gather so scatter can be captured into an NPUGraph.
        sparse_slots[:, 0, :] = cache_slots[req_e_long][:, topk_long]

    def _b_li_manage():
        return torch.ops.ops_overlap.li_manage_out.default(
            li_query, idx_cache, weights, req_e_dev, cache_slots,
            cache_tokens_t, candidate_lens, idx_bt,
            li_src_ids, li_dst_slots, miss_counts)

    def _b_lightning_indexer():
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
        _b_extract_slots()
        return torch.ops.ops_overlap.sparse_and_tail_attention_and_scatter_copy.default(
            query, hbm_ckv, sparse_slots, cache_tokens_t, hbm_bt_dev,
            actual_q, actual_kv, qrope, hbm_kpe, dram_kpe, dram_ckv,
            dram_bt_dev, src_ids_t, miss_counts, scale)

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
    print(f"[B-cfg] verified miss_counts={actual_miss}", flush=True)
    _b_scatter_sfa()
    torch.npu.synchronize()
    print(f"[B-cfg] timing_mode={'graph' if use_graph else 'eager'}", flush=True)

    def _b_pipeline():
        # Nano production round: LI_MANAGE then scatter/SFA on the same stream.
        _b_li_manage()
        _b_scatter_sfa()

    # Capture after the functional smoke so graph replay measures steady-state
    # kernel time without eager launch/dispatch noise.
    full_runner, full_mode = maybe_graph_runner(
        _b_pipeline, use_graph=use_graph,
        warmup=max(1, args.warmup // 2), reset=_b_reset)

    if getattr(args, "profile", False):
        # One pipeline per iter. Reset stays outside the profiled work.
        profile_pipeline(
            [("full_pipeline", full_runner)],
            args.warmup, args.profile_iters, args.profile_dir,
            top_n=args.profile_top_n,
            reset=_b_reset,
            drop_first=args.profile_drop_first)
        return {
            "approach": "B", "profile": True,
            "timing_mode": full_mode,
            "batch_size": batch, "q_heads": q_heads, "indexer_heads": idx_heads,
            "seq_len": seq_len, "topk_layout": topk_layout,
            "cache_tokens": cache_tokens, "miss": miss,
            "warmup": args.warmup, "iters": args.profile_iters,
        }

    li_runner, li_mode = maybe_graph_runner(
        _b_lightning_indexer, use_graph=use_graph, warmup=max(1, args.warmup // 2))
    li_manage_runner, li_manage_mode = maybe_graph_runner(
        _b_li_manage, use_graph=use_graph,
        warmup=max(1, args.warmup // 2), reset=_b_reset)

    def _b_reset_then_manage():
        _b_reset()
        _b_li_manage()
        torch.npu.synchronize()

    scatter_runner, scatter_mode = maybe_graph_runner(
        _b_scatter_sfa, use_graph=use_graph,
        warmup=max(1, args.warmup // 2), reset=_b_reset_then_manage)

    lightning_indexer_ms = bench_events(li_runner, args.warmup, args.iters)
    li_manage_ms = bench_events(
        li_manage_runner, args.warmup, args.iters, reset=_b_reset)
    sfa_ms = bench_events(
        scatter_runner, args.warmup, args.iters, reset=_b_reset_then_manage)

    return {
        "approach": "B",
        "timing_mode": full_mode,
        "batch_size": batch, "q_heads": q_heads, "indexer_heads": idx_heads,
        "seq_len": seq_len, "topk_layout": topk_layout,
        "cache_tokens": cache_tokens, "miss": miss,
        "lightning_indexer_ms": lightning_indexer_ms,
        "li_manage_ms": li_manage_ms, "scatter_sfa_ms": sfa_ms,
        "total_ms": li_manage_ms + sfa_ms, "warmup": args.warmup, "iters": args.iters,
        "li_mode": li_mode, "li_manage_mode": li_manage_mode,
        "scatter_mode": scatter_mode,
    }


# ===================== Compare =====================
def compare(args) -> None:
    with open(args.out_a) as f:
        ra = json.load(f)
    with open(args.out_b) as f:
        rb = json.load(f)
    print("----- TIMING SUMMARY (ms, mean over iters) -----", flush=True)
    if ra["topk_layout"] != rb["topk_layout"]:
        raise ValueError(
            "Cannot compare runs with different top-k layouts: "
            f"A={ra['topk_layout']}, B={rb['topk_layout']}.")
    print(f"Top-k layout: {ra['topk_layout']}", flush=True)
    if "miss" in ra or "miss" in rb:
        print(
            f"Miss: A={ra.get('miss', 'n/a')} B={rb.get('miss', 'n/a')}",
            flush=True)
    print(
        f"Timing mode: A={ra.get('timing_mode', 'unknown')} "
        f"B={rb.get('timing_mode', 'unknown')}",
        flush=True)
    print(f"Approach A (vllm-ascend):  LI={ra['li_ms']:.4f}  "
          f"planner(launch_host)={ra.get('planner_ms', 0.0):.4f}  "
          f"fused_overlap={ra['fused_overlap_ms']:.4f}  "
          f"pipeline={ra.get('pipeline_ms', 0.0):.4f}  "
          f"total={ra['total_ms']:.4f}",
          flush=True)
    print(f"Approach B (nano):        LI={rb['lightning_indexer_ms']:.4f}  "
          f"LI_MANAGE={rb['li_manage_ms']:.4f}  "
          f"scatter_sfa={rb['scatter_sfa_ms']:.4f}  total={rb['total_ms']:.4f}",
          flush=True)
    diff = ra["total_ms"] - rb["total_ms"]
    who = "A faster" if diff < 0 else "B faster"
    print(f"Delta (A-B): {diff:+.4f} ms  ({who} by {abs(diff):.4f} ms)", flush=True)
    print("UT_OK", flush=True)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--approach", choices=("A", "B"), default=None,
                   help="Run one approach in this process (writes --out).")
    p.add_argument("--compare", action="store_true",
                   help="Compare --out-a and --out-b JSON results.")
    p.add_argument("--out", default=None, help="Output JSON path for --approach.")
    p.add_argument("--out-a", default="result_a.json")
    p.add_argument("--out-b", default="result_b.json")
    p.add_argument("--nano-path", default=_default_nano_path(),
                   help="Path to nano torch_extension dir (contains ops_overlap/). "
                        "Override via $NANO_PATH env or --nano-path.")
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--tp", type=int, default=16,
                   help="Tensor parallel size. q_heads = NUM_ATTENTION_HEADS / tp "
                        "(GLM-5.2: 64 / 16 = 4).")
    p.add_argument("--q-heads", type=int, default=None,
                   help="Attention heads per rank. Overrides --tp derivation. "
                        "Default: 64 / tp.")
    p.add_argument("--indexer-heads", type=int, default=32,
                   help="Indexer heads (GLM-5.2 index_n_heads=32).")
    p.add_argument("--seq-len", type=int, default=65536)
    p.add_argument("--cache-tokens", type=int, default=8192)
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
        "--miss", type=int, default=300,
        help="Miss count per row for both A and B (0..TOPK). "
             "Each timed/profile iter rebuilds residents with a freshly "
             "randomized hit/miss partition among the same topk set.",
    )
    p.add_argument(
        "--hit-ratio", type=float, default=None,
        help="Deprecated. If set, overrides --miss as "
             "miss = round(TOPK * (1 - hit_ratio)). Prefer --miss.",
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
    return p.parse_args()


def main():
    args = parse_args()
    if args.parse_profile_dir is not None:
        summarize_profile_csvs(
            args.parse_profile_dir,
            top_n=args.profile_top_n,
            drop_first=args.profile_drop_first,
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
    if args.approach == "A":
        result = run_a(args)
    else:
        result = run_b(args)
    out = args.out or f"result_{args.approach.lower()}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    if result.get("profile"):
        print(f"RESULT approach={args.approach} out={out} profile=True",
              flush=True)
    else:
        print(f"RESULT approach={args.approach} out={out} total_ms={result['total_ms']:.4f}",
              flush=True)
    print("UT_OK", flush=True)


if __name__ == "__main__":
    main()
