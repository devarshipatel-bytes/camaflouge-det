#!/usr/bin/env python3
"""Compare every run that has already been evaluated.

Reads ``reports/eval/*/summary.json`` and emits one grouped bar chart per
metric plus a tidy CSV. This script runs no models — it only aggregates
completed evaluations, which is what keeps it consistent with the
one-run-per-invocation rule for ``08_evaluate.py``.

It states which runs it found, so a missing dataset (mhcd, until it is
trained) is visible rather than silently absent from the chart.

Example
-------
    python scripts/08_evaluate.py --run acd1k
    python scripts/08_evaluate.py --run cpd1k
    python scripts/10_compare_runs.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chd.eval.report import MASK_METRICS  # noqa: E402
from chd.viz.colors import DATASET_COLOR, DATASET_LABEL, INK  # noqa: E402
from chd.viz.panels import save_figure  # noqa: E402

#: Lower is better for these, so charts label the direction explicitly.
LOWER_IS_BETTER = {"MAE"}
DEFAULT_METRICS = ("S_alpha", "MAE", "F_beta_mean", "E_phi_mean", "F_bd", "IoU")


def load_summaries(eval_root: Path) -> list[dict]:
    summaries = []
    for path in sorted(eval_root.glob("*/summary.json")):
        data = json.loads(path.read_text())
        if not data.get("mask"):
            print(f"[compare] skipping {path}: no positives, so no mask metrics")
            continue
        data["_run"] = data.get("run", path.parent.name)
        summaries.append(data)
    return summaries


def write_comparison_csv(summaries: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["run", "dataset", "architecture", "backbone", "img_size", "weights",
              "n_positives", *MASK_METRICS, "presence_accuracy", "presence_auc"]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            presence = summary.get("presence", {})
            # Single-class splits (every dataset except mhcd/combined) have an
            # empty half of the confusion matrix, so their gate rates are not
            # measurable. Leave the cells blank rather than writing a number
            # that would sit in the same column as mhcd's genuine values.
            single_class = bool(presence.get("presence_single_class"))
            row = {
                "run": summary["_run"],
                "dataset": summary.get("dataset"),
                "architecture": summary.get("architecture"),
                "backbone": summary.get("backbone"),
                "img_size": summary.get("img_size"),
                "weights": summary.get("weights"),
                "n_positives": summary.get("n_positives"),
                "presence_accuracy": None if single_class else presence.get("presence_accuracy"),
                "presence_auc": None if single_class else presence.get("presence_auc"),
            }
            row.update({m: summary["mask"].get(m) for m in MASK_METRICS})
            writer.writerow(row)


def bar_chart(summaries: list[dict], metric: str, out: Path) -> None:
    labels, values, colors = [], [], []
    for summary in summaries:
        dataset = summary.get("dataset", "?")
        labels.append(f"{summary['_run']}\n{DATASET_LABEL.get(dataset, dataset)}")
        values.append(summary["mask"].get(metric, float("nan")))
        colors.append(DATASET_COLOR.get(dataset, INK["secondary"]))

    fig, ax = plt.subplots(figsize=(max(5.0, 1.5 * len(labels)), 4.0))
    positions = np.arange(len(labels))
    ax.bar(positions, values, color=colors, width=0.65)
    for x, value in zip(positions, values):
        ax.text(x, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8, color=INK["primary"])

    direction = "lower is better" if metric in LOWER_IS_BETTER else "higher is better"
    ax.set_title(f"{metric} by run  ({direction})", color=INK["primary"])
    ax.set_ylabel(metric)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_facecolor("white")
    ax.grid(True, axis="y", color=INK["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    save_figure(fig, out, metric)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-root", type=Path, default=Path("reports/eval"))
    parser.add_argument("--out", type=Path, default=Path("reports/comparison"))
    parser.add_argument("--metrics", nargs="*", default=list(DEFAULT_METRICS))
    args = parser.parse_args()

    summaries = load_summaries(args.eval_root)
    if not summaries:
        raise SystemExit(
            f"no evaluated runs under {args.eval_root}. "
            "Run scripts/08_evaluate.py --run <name> first."
        )

    print(f"[compare] {len(summaries)} evaluated run(s):")
    for summary in summaries:
        gate = "" if not summary.get("presence", {}).get("presence_single_class") else \
            "  (single-class split: presence gate n/a)"
        print(f"    {summary['_run']:<24} dataset={summary.get('dataset') or '?':<12} "
              f"S_alpha={summary['mask'].get('S_alpha', float('nan')):.4f}{gate}")

    write_comparison_csv(summaries, args.out / "comparison.csv")
    for metric in args.metrics:
        if metric not in MASK_METRICS:
            print(f"[compare] skipping unknown metric {metric!r}")
            continue
        bar_chart(summaries, metric, args.out)

    print(f"[compare] wrote {args.out}/comparison.csv and one chart per metric")


if __name__ == "__main__":
    main()
