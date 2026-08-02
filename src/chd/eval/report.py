"""Aggregate per-image metrics into the numbers and files the paper needs.

Two aggregation rules here are decisions, not conveniences:

1. **Mask metrics average over positives only.** An empty ground truth sends
   ``s_measure`` down its ``y == 0`` branch, which returns ``1 - pred.mean()``
   — a presence score, not a segmentation score. Averaging that together with
   real segmentation scores would inflate S_alpha, and ``mhcd`` (and therefore
   ``combined``) carries 376 negatives.
2. **Presence metrics cover every image.** That is the whole point of the
   presence gate, and it is what the paper's target-free-frame claim rests on.
3. **Single-class splits report no presence rates.** ``acd1k``, ``cpd1k`` and
   ``camo_human`` have zero negatives in ``meta.csv``, so ``fp == tn == 0`` and
   the naive precision is a vacuous ``1.0`` while "accuracy" is really just
   recall. Those four rates plus AUC are returned as ``None`` with
   ``presence_single_class = True``; only the raw tp/fp/tn/fn counts survive,
   because those are still factual.

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

ROW_FIELDS = ("stem", "is_negative", "gt_positive", "presence_prob", "height", "width", *MASK_METRICS)


def metric_row(pred: Prediction) -> dict:
    """One flat CSV row: identity, presence, and every mask metric.

    ``gt_positive`` is measured from the mask itself rather than trusted from
    the manifest's ``is_negative`` flag. If the two ever disagree — a
    mislabelled empty ground truth — ``aggregate`` must not let that image into
    the mask means, where ``s_measure``'s ``1 - pred.mean()`` branch would
    inflate S_alpha.
    """
    scores = evaluate_all(pred.prob, pred.gt)
    height, width = pred.gt.shape
    return {
        "stem": pred.stem,
        "is_negative": int(pred.is_negative),
        "gt_positive": int(bool(pred.gt.sum() > 0)),
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

    **Single-class splits report no rates.** Most of our datasets are
    all-positive (acd1k, cpd1k and camo_human have zero negatives in
    ``meta.csv``; only mhcd, and therefore combined, carries any). With one
    class absent the confusion matrix has an empty half: precision collapses to
    ``tp / (tp + 0) == 1.0`` no matter how bad the gate is, and "accuracy"
    silently equals recall. Publishing those as measured numbers would be a
    fabrication, so precision/recall/F1/accuracy/AUC are all ``None`` and
    ``presence_single_class`` is ``True``. The raw counts are kept — they are
    still facts about what happened.
    """
    if not presence_probs:
        return {
            "presence_tp": 0, "presence_fp": 0, "presence_tn": 0, "presence_fn": 0,
            "presence_accuracy": None, "presence_precision": None,
            "presence_recall": None, "presence_f1": None, "presence_auc": None,
            "presence_single_class": True,
            "presence_threshold": threshold,
        }

    probs = np.asarray(presence_probs, dtype=np.float64)
    labels = (~np.asarray(is_negative, dtype=bool)).astype(int)
    predicted = (probs >= threshold).astype(int)

    tp = int(((predicted == 1) & (labels == 1)).sum())
    fp = int(((predicted == 1) & (labels == 0)).sum())
    tn = int(((predicted == 0) & (labels == 0)).sum())
    fn = int(((predicted == 0) & (labels == 1)).sum())

    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return {
            "presence_tp": tp, "presence_fp": fp, "presence_tn": tn, "presence_fn": fn,
            "presence_accuracy": None, "presence_precision": None,
            "presence_recall": None, "presence_f1": None, "presence_auc": None,
            "presence_single_class": True,
            "presence_threshold": threshold,
        }

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
        "presence_single_class": False,
        "presence_threshold": threshold,
    }


def _is_positive(row: dict) -> bool:
    """A row counts as a positive only if the manifest AND the mask agree.

    ``is_negative`` comes from ``meta.csv``; ``gt_positive`` is measured from
    the mask that was actually scored. Requiring both means a mislabelled
    empty ground truth is excluded from the mask means instead of dragging
    ``s_measure``'s ``1 - pred.mean()`` branch into them. Rows written before
    ``gt_positive`` existed fall back to the manifest flag alone.
    """
    return not row["is_negative"] and bool(int(row.get("gt_positive", 1)))


def aggregate(rows: list[dict]) -> dict:
    """Positives-only mask means plus an all-images presence block."""
    positives = [r for r in rows if _is_positive(r)]
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


def single_class_note(summary: dict) -> str:
    """One line naming which class is missing and why the gate rates are absent.

    Shared by ``write_metrics_md`` and ``scripts/08_evaluate.py`` so the paper
    table and the console say exactly the same thing.
    """
    presence = summary.get("presence", {})
    # Counted from the presence block's own labels, not from n_negatives:
    # n_negatives also absorbs any row whose mask disagreed with the manifest,
    # which is not what the gate was scored on.
    n_neg = presence.get("presence_fp", 0) + presence.get("presence_tn", 0)
    n_pos = presence.get("presence_tp", 0) + presence.get("presence_fn", 0)
    absent = "negative (target-free)" if n_neg == 0 else "positive (target-present)"
    return (
        f"Presence gate not measurable: the {summary.get('split', '?')} split of "
        f"`{summary.get('dataset', '?')}` contains only one class — no {absent} images "
        f"({n_pos} positive, {n_neg} negative). "
        "Accuracy, precision, recall, F1 and AUC are reported as n/a rather than as "
        "the degenerate values a one-class confusion matrix produces (precision is "
        "trivially 1.0 with no negatives, and accuracy would just be recall). The raw "
        "tp/fp/tn/fn counts above are still exact."
    )


def write_per_image_csv(rows: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ROW_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def write_failures_csv(rows: list[dict], path: Path) -> None:
    """Positives sorted worst-first by S_alpha, tie-broken by IoU."""
    positives = [r for r in rows if _is_positive(r)]
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

    presence = summary["presence"]
    lines += ["", "## Presence gate (all images)", "", "| Metric | Value |", "| --- | --- |"]
    for key, value in presence.items():
        formatted = "n/a" if value is None else (f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append(f"| {key} | {formatted} |")

    if presence.get("presence_single_class"):
        lines += ["", f"> **{single_class_note(summary)}**"]

    lines += ["", f"> {summary['notes']}", ""]
    path.write_text("\n".join(lines))
