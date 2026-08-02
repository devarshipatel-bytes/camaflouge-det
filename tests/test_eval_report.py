"""Tests for chd.eval.report — aggregation rules that affect published numbers.

The rule being pinned hardest: mask metrics average over positives only.
An empty ground truth sends s_measure down its y == 0 branch, where it
returns 1 - pred.mean() — a presence score, not a segmentation score.
Averaging that in would inflate S_alpha, and camo_human has 1024 negatives.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chd.eval import report  # noqa: E402
from chd.eval.predict import Prediction  # noqa: E402


def make_pred(stem: str, *, is_negative: bool = False, presence: float = 0.9,
              perfect: bool = True) -> Prediction:
    gt = np.zeros((32, 32), dtype=np.float32)
    if not is_negative:
        gt[8:24, 8:24] = 1.0
    prob = gt.copy() if perfect else np.zeros_like(gt)
    return Prediction(stem=stem, prob=prob, gt=gt, presence=presence, is_negative=is_negative)


class TestMetricRow:
    def test_row_carries_every_mask_metric_plus_identity(self) -> None:
        row = report.metric_row(make_pred("a"))
        for key in report.MASK_METRICS:
            assert key in row
        assert row["stem"] == "a"
        assert row["is_negative"] == 0
        assert row["height"] == 32 and row["width"] == 32

    def test_perfect_prediction_scores_perfectly(self) -> None:
        row = report.metric_row(make_pred("a"))
        assert row["IoU"] == pytest.approx(1.0)
        assert row["MAE"] == pytest.approx(0.0)


class TestPresenceMetrics:
    def test_hand_computed_confusion_matrix(self) -> None:
        """probs .9/.8 on positives, .1/.6 on negatives, threshold .5:
        TP=2, FN=0, FP=1 (the .6 negative), TN=1."""
        out = report.presence_metrics([0.9, 0.8, 0.1, 0.6], [False, False, True, True])
        assert out["presence_tp"] == 2
        assert out["presence_fn"] == 0
        assert out["presence_fp"] == 1
        assert out["presence_tn"] == 1
        assert out["presence_accuracy"] == pytest.approx(0.75)
        assert out["presence_precision"] == pytest.approx(2 / 3)
        assert out["presence_recall"] == pytest.approx(1.0)
        assert out["presence_f1"] == pytest.approx(0.8)

    def test_auc_is_one_for_perfectly_separated_scores(self) -> None:
        out = report.presence_metrics([0.9, 0.8, 0.2, 0.1], [False, False, True, True])
        assert out["presence_auc"] == pytest.approx(1.0)

    def test_auc_is_none_without_both_classes(self) -> None:
        out = report.presence_metrics([0.9, 0.8], [False, False])
        assert out["presence_auc"] is None

    def test_empty_input_does_not_divide_by_zero(self) -> None:
        out = report.presence_metrics([], [])
        assert out["presence_accuracy"] is None
        assert np.isfinite(out["presence_tp"])


class TestAggregate:
    def test_mask_means_exclude_negatives(self) -> None:
        """The negative row is a perfect all-zero prediction on an empty GT,
        which scores IoU 1.0. Including it would mask a bad positive."""
        rows = [
            report.metric_row(make_pred("pos", perfect=False)),
            report.metric_row(make_pred("neg", is_negative=True)),
        ]
        out = report.aggregate(rows)
        assert out["n_positives"] == 1
        assert out["n_negatives"] == 1
        assert out["mask"]["IoU"] == pytest.approx(rows[0]["IoU"])

    def test_presence_block_covers_all_images(self) -> None:
        rows = [
            report.metric_row(make_pred("pos", presence=0.9)),
            report.metric_row(make_pred("neg", is_negative=True, presence=0.1)),
        ]
        out = report.aggregate(rows)
        assert out["presence"]["presence_accuracy"] == pytest.approx(1.0)

    def test_all_negative_split_reports_no_mask_metrics(self) -> None:
        rows = [report.metric_row(make_pred("n1", is_negative=True))]
        out = report.aggregate(rows)
        assert out["mask"] == {}
        assert out["n_positives"] == 0


class TestWriters:
    def test_per_image_csv_round_trips(self, tmp_path: Path) -> None:
        rows = [report.metric_row(make_pred("a")), report.metric_row(make_pred("b"))]
        path = tmp_path / "per_image.csv"
        report.write_per_image_csv(rows, path)
        with path.open() as fh:
            back = list(csv.DictReader(fh))
        assert [r["stem"] for r in back] == ["a", "b"]
        assert "S_alpha" in back[0]

    def test_failures_are_sorted_worst_first_and_exclude_negatives(self, tmp_path: Path) -> None:
        rows = [
            report.metric_row(make_pred("good")),
            report.metric_row(make_pred("bad", perfect=False)),
            report.metric_row(make_pred("neg", is_negative=True)),
        ]
        path = tmp_path / "failures.csv"
        report.write_failures_csv(rows, path)
        with path.open() as fh:
            back = list(csv.DictReader(fh))
        assert [r["stem"] for r in back] == ["bad", "good"]

    def test_summary_json_is_valid_json(self, tmp_path: Path) -> None:
        summary = report.aggregate([report.metric_row(make_pred("a"))])
        summary["run"] = "toy"
        path = tmp_path / "summary.json"
        report.write_summary_json(summary, path)
        assert json.loads(path.read_text())["run"] == "toy"

    def test_metrics_md_names_the_metrics(self, tmp_path: Path) -> None:
        summary = report.aggregate([report.metric_row(make_pred("a"))])
        summary["run"] = "toy"
        path = tmp_path / "metrics.md"
        report.write_metrics_md(summary, path)
        text = path.read_text()
        assert "S_alpha" in text and "toy" in text
