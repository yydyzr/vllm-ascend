#!/usr/bin/env python3
"""Plot per-layer avg_hit_ratio curves from SFA fused_overlap debug logs (TP0).

Reads Worker_TP0 ``[fused_overlap_offload][selection][hit]`` lines and writes
one PNG per layer: x=step, y=avg_hit_ratio.

Supports both log formats:

- old: ``avg_hit_ratio=... (steps=...)``
- new: ``layer_avg=... (layer_steps=...)``  (preferred for per-layer curves;
  process-wide ``avg_hit_ratio`` is ignored when ``layer_avg`` is present)

Usage:
    python3 examples/plot_tp0_avg_hit_ratio.py server.log -o hit_ratio_plots
    python3 examples/plot_tp0_avg_hit_ratio.py -o hit_ratio_plots < server.log
    cat server.log | python3 examples/plot_tp0_avg_hit_ratio.py
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path


LAYER_ID_RE = re.compile(r"layers\.(\d+)\.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse TP0 fused_overlap selection hit logs and plot "
            "avg_hit_ratio vs step for each layer."
        )
    )
    parser.add_argument(
        "log_file",
        nargs="?",
        default="-",
        help="Log file path, or '-' / omit to read stdin",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="hit_ratio_plots_tp0",
        help="Directory to write per-layer PNG files (default: hit_ratio_plots_tp0)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also show plots interactively (requires GUI backend)",
    )
    return parser.parse_args()


def iter_log_lines(path: str):
    if path == "-":
        yield from sys.stdin
        return
    with open(path, encoding="utf-8", errors="replace") as f:
        yield from f


def layer_sort_key(layer_name: str) -> tuple[int, str]:
    m = LAYER_ID_RE.search(layer_name)
    return (int(m.group(1)), layer_name) if m else (10**9, layer_name)


def layer_label(layer_name: str) -> str:
    m = LAYER_ID_RE.search(layer_name)
    return f"layer_{m.group(1)}" if m else re.sub(r"[^\w.-]+", "_", layer_name)


def parse_records(lines) -> dict[str, list[tuple[int, float]]]:
    """Return {layer_name: [(step, avg_hit_ratio), ...]} sorted by step."""
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    skipped = 0
    for line in lines:
        if "Worker_TP0" not in line or "[selection][hit]" not in line:
            continue
        if "avg_hit_ratio=" not in line:
            continue

        layer_m = re.search(r"layer=(\S+)", line)
        if not layer_m:
            skipped += 1
            continue
        layer = layer_m.group(1)

        # Prefer per-layer fields from the new log format.
        layer_avg_m = re.search(r"layer_avg=([0-9.]+)", line)
        layer_steps_m = re.search(r"layer_steps=(\d+)", line)
        if layer_avg_m and layer_steps_m:
            series[layer].append((int(layer_steps_m.group(1)), float(layer_avg_m.group(1))))
            continue

        avg_m = re.search(r"avg_hit_ratio=([0-9.]+)", line)
        steps_m = re.search(r"\(steps=(\d+)\b", line)
        if avg_m and steps_m:
            series[layer].append((int(steps_m.group(1)), float(avg_m.group(1))))
            continue
        skipped += 1

    for layer, points in series.items():
        # Keep chronological order; if duplicate steps appear, keep last value.
        by_step: dict[int, float] = {}
        for step, value in points:
            by_step[step] = value
        series[layer] = sorted(by_step.items())

    if skipped:
        print(f"[warn] skipped {skipped} hit line(s) with incomplete fields", file=sys.stderr)
    return series


def plot_series(series: dict[str, list[tuple[int, float]]], output_dir: Path, show: bool) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(
            "matplotlib is required: pip install matplotlib\n" + str(e)
        ) from e

    if not series:
        raise SystemExit("No Worker_TP0 selection hit records found.")

    output_dir.mkdir(parents=True, exist_ok=True)
    layers = sorted(series.keys(), key=layer_sort_key)
    print(f"Found {len(layers)} layer(s); writing plots to {output_dir.resolve()}")

    for layer in layers:
        points = series[layer]
        steps = [p[0] for p in points]
        values = [p[1] for p in points]
        label = layer_label(layer)
        out_path = output_dir / f"{label}_avg_hit_ratio.png"

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(steps, values, linewidth=1.5)
        ax.set_xlabel("step")
        ax.set_ylabel("avg_hit_ratio")
        ax.set_title(f"TP0 {layer}")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        if show:
            plt.show()
        plt.close(fig)
        print(f"  wrote {out_path} ({len(points)} points)")


def main() -> None:
    args = parse_args()
    series = parse_records(iter_log_lines(args.log_file))
    plot_series(series, Path(args.output_dir), show=args.show)


if __name__ == "__main__":
    main()
