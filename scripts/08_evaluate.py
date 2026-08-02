#!/usr/bin/env python3
"""Evaluate one trained run against its dataset's test split.

Everything about the model — dataset, architecture, backbone, img_size, pose
setting — is recovered from the checkpoint's own stored ``args``, so you name
a run folder and nothing else:

    python scripts/08_evaluate.py --run camo-human-final

That matters because the run folders are named inconsistently
(``camo-human-final`` holds the ``camo_human`` dataset) and because runs were
trained at ``img_size`` 640, not the 352 CLI default — evaluating at the
wrong size would silently degrade every reported number.

Protocol notes (see docs/superpowers/specs/2026-08-02-evaluation-visualization-design.md):

  - Predictions are scored against the ground truth at its **native**
    resolution, which is what the COD literature does and what makes these
    numbers comparable to the baselines in the paper's tables.
  - Mask metrics average over **positives only**; presence-gate metrics cover
    every image, including negatives.
  - Only ``mhcd`` (and hence ``combined``) actually has negatives — 376, of
    which 69 are in test. On an all-positive split the presence gate has no
    measurable rates, and this script says so loudly rather than printing the
    vacuous ``precision = 1.0`` a one-class confusion matrix produces.

Examples
--------
    # full test split
    python scripts/08_evaluate.py --run acd1k

    # quick smoke check, keep the probability maps for manual inspection
    python scripts/08_evaluate.py --run acd1k --limit 8 --save-preds

    # score the val split instead
    python scripts/08_evaluate.py --run acd1k --split val
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chd.eval.predict import Prediction, predict_run  # noqa: E402
from chd.eval.report import (  # noqa: E402
    aggregate,
    metric_row,
    single_class_note,
    write_failures_csv,
    write_metrics_md,
    write_per_image_csv,
    write_summary_json,
)
from chd.eval.runs import DEFAULT_RUNS_ROOT, load_run  # noqa: E402


def _metric_job(stem: str, prob: np.ndarray, gt: np.ndarray, presence: float, is_negative: bool) -> dict:
    """Worker-process entry point: metrics for one image.

    Takes plain arrays rather than a Prediction so the pickled payload stays
    minimal, and rebuilds the dataclass on the far side.
    """
    return metric_row(Prediction(stem=stem, prob=prob, gt=gt,
                                 presence=presence, is_negative=is_negative))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run folder name under --runs-root, e.g. camo-human-final")
    p.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="explicit checkpoint path, bypassing --run lookup")
    p.add_argument("--prefer", choices=("best", "last"), default="best")
    p.add_argument("--split", default="test")
    p.add_argument("--data-root", type=Path, default=None,
                   help="default: the data_root stored in the checkpoint")
    p.add_argument("--dataset", default=None, help="override the checkpoint's dataset")
    p.add_argument("--img-size", type=int, default=None, help="override the checkpoint's img_size")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=None, help="evaluate only the first N images")
    p.add_argument("--workers", type=int, default=None,
                   help="metric worker processes; 0 runs inline. Default: cpu_count - 2")
    p.add_argument("--save-preds", action="store_true",
                   help="write uint8-quantized probability maps to <out>/preds/ for manual "
                        "inspection or offline re-analysis. The figure scripts do NOT read "
                        "these — 09_visualize_predictions.py recomputes from the checkpoint.")
    p.add_argument("--out", type=Path, default=None, help="default: reports/eval/<run>")
    return p


def default_workers(requested: int | None) -> int:
    import os

    if requested is not None:
        return max(0, requested)
    return max(1, (os.cpu_count() or 2) - 2)


def main() -> None:
    args = build_parser().parse_args()

    bundle = load_run(
        args.run, runs_root=args.runs_root, device=args.device, prefer=args.prefer,
        overrides={"dataset": args.dataset, "img_size": args.img_size},
        checkpoint=args.checkpoint,
    )
    out = args.out or Path("reports/eval") / args.run
    out.mkdir(parents=True, exist_ok=True)
    preds_dir = out / "preds"
    if args.save_preds:
        preds_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"  run          {bundle.name}  ({bundle.checkpoint_path})")
    print(f"  dataset      {bundle.config.dataset}  split={args.split}")
    print(f"  architecture {getattr(bundle.config, 'architecture', 'chdnet')}"
          f" + {getattr(bundle.config, 'backbone', '?')}")
    print(f"  weights      {bundle.weights}  (epoch {bundle.epoch},"
          f" best S_alpha={bundle.best_s_alpha})")
    print(f"  img_size     {bundle.config.img_size}  (scored at native GT resolution)")
    print(f"  out          {out}")
    print("=" * 78)

    workers = default_workers(args.workers)
    started = time.time()
    rows: list[dict] = []

    def record(prediction: Prediction, row: dict) -> None:
        rows.append(row)
        if args.save_preds:
            cv2.imwrite(str(preds_dir / f"{prediction.stem}.png"),
                        (np.clip(prediction.prob, 0, 1) * 255).astype(np.uint8))
        if len(rows) % 25 == 0:
            print(f"  [{len(rows)}] {time.time() - started:.0f}s elapsed", flush=True)

    stream = predict_run(bundle, split=args.split, data_root=args.data_root,
                         device=args.device, limit=args.limit)

    def evaluate_all_images() -> None:
        if workers == 0:
            for prediction in stream:
                record(prediction, metric_row(prediction))
            return
        # Bounded in-flight futures: the metric stage is the slow one (two
        # 255-threshold curves per image), but queueing all 1150 native-
        # resolution maps at once would cost several GB of pickled payload.
        pending: dict = {}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for prediction in stream:
                future = pool.submit(_metric_job, prediction.stem, prediction.prob,
                                     prediction.gt, prediction.presence, prediction.is_negative)
                pending[future] = prediction
                while len(pending) >= workers * 3:
                    done = next(as_completed(pending))
                    record(pending.pop(done), done.result())
            for future in as_completed(list(pending)):
                record(pending[future], future.result())

    try:
        evaluate_all_images()
    except BaseException as error:
        # A worker dying at image 1100 of 1150 must not throw away 1100
        # images' worth of metrics — a full combined-split evaluation is ~40
        # CPU-minutes. Flush whatever completed, then re-raise: the partial
        # file is explicitly labelled partial so it can never be mistaken for
        # a finished run (no summary.json / metrics.md is written).
        partial = out / "per_image.csv"
        if rows:
            rows.sort(key=lambda r: r["stem"])
            write_per_image_csv(rows, partial)
        print("!" * 78, file=sys.stderr)
        print(f"  evaluation FAILED after {len(rows)} of the split's images: "
              f"{type(error).__name__}: {error}", file=sys.stderr)
        if rows:
            print(f"  PARTIAL results flushed to {partial} ({len(rows)} row(s)).", file=sys.stderr)
            print("  No summary.json/metrics.md was written — this run is incomplete "
                  "and must not be reported.", file=sys.stderr)
        else:
            print("  No rows completed, so nothing was written.", file=sys.stderr)
        print("!" * 78, file=sys.stderr)
        raise

    if not rows:
        raise SystemExit(f"no images evaluated for split {args.split!r} — is the split file empty?")

    rows.sort(key=lambda r: r["stem"])  # process pool completes out of order

    summary = aggregate(rows)
    summary.update({
        "run": bundle.name,
        "checkpoint": str(bundle.checkpoint_path),
        "dataset": bundle.config.dataset,
        "split": args.split,
        "architecture": getattr(bundle.config, "architecture", "chdnet"),
        "backbone": getattr(bundle.config, "backbone", None),
        "img_size": bundle.config.img_size,
        "weights": bundle.weights,
        "epoch": bundle.epoch,
        "best_s_alpha_from_training": bundle.best_s_alpha,
        "eval_seconds": round(time.time() - started, 1),
    })

    write_per_image_csv(rows, out / "per_image.csv")
    write_failures_csv(rows, out / "failures.csv")
    write_summary_json(summary, out / "summary.json")
    write_metrics_md(summary, out / "metrics.md")

    print("-" * 78)
    for key, value in summary["mask"].items():
        print(f"  {key:<18} {value:.4f}")
    presence_accuracy = summary["presence"]["presence_accuracy"]
    if presence_accuracy is not None:
        print(f"  {'presence_acc':<18} {presence_accuracy:.4f}")
    print("-" * 78)
    if summary["presence"].get("presence_single_class"):
        print("  !! WARNING: single-class split — presence gate NOT measurable")
        for chunk in textwrap.wrap(single_class_note(summary), 72):
            print(f"     {chunk}")
        print("-" * 78)
    print(f"  wrote {out}/per_image.csv, failures.csv, summary.json, metrics.md")
    if args.save_preds:
        print(f"  wrote {preds_dir}/*.png")
    print(f"  took {summary['eval_seconds']}s over {summary['n_images']} image(s)")


if __name__ == "__main__":
    main()
