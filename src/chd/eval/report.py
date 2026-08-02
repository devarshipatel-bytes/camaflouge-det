"""Aggregate per-image metrics into the numbers and files the paper needs.

Two aggregation rules here are decisions, not conveniences:

1. **Mask metrics average over positives only.** An empty ground truth sends
   ``s_measure`` down its ``y == 0`` branch, which returns ``1 - pred.mean()``
   — a presence score, not a segmentation score. Averaging that together with
   real segmentation scores would inflate S_alpha, and ``camo_human`` has
   1024 negatives.
2. **Presence metrics cover every image.** That is the whole point of the
   presence gate, and it is what the paper's target-free-frame claim rests on.

AUC is computed from ranks (Mann-Whitney U) using ``scipy.stats.rankdata`` so
ties are handled correctly and no new dependency is needed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from chd.eval.predict import Prediction
from chd.metrics import evaluate_all

#: Keys returned by ``chd.metrics.evaluate_all``, in report order.
MASK_METRICS = (
    "MAE", "F_beta_mean", "F_beta_max", "F_beta_adaptive", "S_alpha",
    "E_phi_mean", "E_phi_max", "E_phi_adaptive", "F_bd", "IoU", "Dice",
)

ROW_FIELDS = ("stem", "is_negative", "presence_prob", "height", "width", *MASK_METRICS)


def metric_row(pred: Prediction) -> dict:
    """One flat CSV row: identity, presence, and every mask metric."""
    scores = evaluate_all(pred.prob, pred.gt)
    height, width = pred.gt.shape
    return {
        "stem": pred.stem,
        "is_negative": int(pred.is_negative),
        "presence_prob": float(pred.presence),
        "height": int(height),
        "width": int(width),
        **{key: float(scores[key]) for key in MASK_METRICS},
    }


def _rank_auc(probs: np.ndarray, labels: np.ndarray) -> float | None:
    """Mann-Whitney U AUC; ``None`` unless both classes are present."""
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(probs)  # average ranks for ties
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def presence_metrics(
    presence_probs: list[float], is_negative: list[bool], threshold: float = 0.5,
) -> dict:
    """Confusion matrix, accuracy/precision/recall/F1 and AUC for the presence gate.

    Positive class = a target IS present, i.e. ``not is_negative``.
    """
    if not presence_probs:
        return {
            "presence_tp": 0, "presence_fp": 0, "presence_tn": 0, "presence_fn": 0,
            "presence_accuracy": None, "presence_precision": None,
            "presence_recall": None, "presence_f1": None, "presence_auc": None,
            "presence_threshold": threshold,
        }

    probs = np.asarray(presence_probs, dtype=np.float64)
    labels = (~np.asarray(is_negative, dtype=bool)).astype(int)
    predicted = (probs >= threshold).astype(int)

    tp = int(((predicted == 1) & (labels == 1)).sum())
    fp = int(((predicted == 1) & (labels == 0)).sum())
    tn = int(((predicted == 0) & (labels == 0)).sum())
    fn = int(((predicted == 0) & (labels == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    return {
        "presence_tp": tp, "presence_fp": fp, "presence_tn": tn, "presence_fn": fn,
        "presence_accuracy": (tp + tn) / len(probs),
        "presence_precision": precision,
        "presence_recall": recall,
        "presence_f1": f1,
        "presence_auc": _rank_auc(probs, labels),
        "presence_threshold": threshold,
    }


def aggregate(rows: list[dict]) -> dict:
    """Positives-only mask means plus an all-images presence block."""
    positives = [r for r in rows if not r["is_negative"]]
    mask_means = (
        {key: float(np.mean([r[key] for r in positives])) for key in MASK_METRICS}
        if positives else {}
    )
    return {
        "n_images": len(rows),
        "n_positives": len(positives),
        "n_negatives": len(rows) - len(positives),
        "mask": mask_means,
        "presence": presence_metrics(
            [r["presence_prob"] for r in rows],
            [bool(r["is_negative"]) for r in rows],
        ),
        "notes": (
            "Mask metrics are averaged over positives only; an empty ground truth "
            "makes S_alpha degenerate to a presence score. Presence metrics cover "
            "all images. Predictions were scored at native ground-truth resolution."
        ),
    }


def write_per_image_csv(rows: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ROW_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def write_failures_csv(rows: list[dict], path: Path) -> None:
    """Positives sorted worst-first by S_alpha, tie-broken by IoU."""
    positives = [r for r in rows if not r["is_negative"]]
    ordered = sorted(positives, key=lambda r: (r["S_alpha"], r["IoU"]))
    write_per_image_csv(ordered, path)


def write_summary_json(summary: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, default=str))


def write_metrics_md(summary: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Evaluation — {summary.get('run', '(unnamed run)')}",
        "",
        f"- dataset: `{summary.get('dataset', '?')}`",
        f"- split: `{summary.get('split', '?')}`",
        f"- architecture: `{summary.get('architecture', '?')}`",
        f"- weights: `{summary.get('weights', '?')}`"
        f" (epoch {summary.get('epoch', '?')})",
        f"- img_size: `{summary.get('img_size', '?')}`",
        f"- images: {summary['n_images']} "
        f"({summary['n_positives']} positive, {summary['n_negatives']} negative)",
        "",
        "## Mask metrics (positives only, native resolution)",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for key in MASK_METRICS:
        value = summary["mask"].get(key)
        lines.append(f"| {key} | {'n/a' if value is None else f'{value:.4f}'} |")

    lines += ["", "## Presence gate (all images)", "", "| Metric | Value |", "| --- | --- |"]
    for key, value in summary["presence"].items():
        formatted = "n/a" if value is None else (f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append(f"| {key} | {formatted} |")

    lines += ["", f"> {summary['notes']}", ""]
    path.write_text("\n".join(lines))
