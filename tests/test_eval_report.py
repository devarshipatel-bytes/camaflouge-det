"""Tests for chd.eval.report — aggregation rules that affect published numbers.

Two rules are pinned hardest here:

1. Mask metrics average over positives only. An empty ground truth sends
   s_measure down its y == 0 branch, where it returns 1 - pred.mean() — a
   presence score, not a segmentation score. Averaging that in would inflate
   S_alpha. mhcd (and therefore combined) carries 376 such negatives.
2. A single-class split reports no presence rates. acd1k, cpd1k and camo_human
   have zero negatives in meta.csv, so fp == tn == 0 and the naive precision is
   a vacuous 1.0. Reporting that in a paper table would be a fabrication.
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

    def test_gt_positive_is_measured_from_the_mask_not_the_manifest(self) -> None:
        assert report.metric_row(make_pred("pos"))["gt_positive"] == 1
        assert report.metric_row(make_pred("neg", is_negative=True))["gt_positive"] == 0
        # Manifest says positive, mask is empty: the measurement wins.
        empty = make_pred("mislabelled", is_negative=True)
        mislabelled = Prediction(stem="mislabelled", prob=empty.prob, gt=empty.gt,
                                 presence=empty.presence, is_negative=False)
        assert report.metric_row(mislabelled)["gt_positive"] == 0

    def test_gt_positive_reaches_the_csv(self) -> None:
        assert "gt_positive" in report.ROW_FIELDS

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
        assert out["presence_single_class"] is False

    def test_auc_is_one_for_perfectly_separated_scores(self) -> None:
        out = report.presence_metrics([0.9, 0.8, 0.2, 0.1], [False, False, True, True])
        assert out["presence_auc"] == pytest.approx(1.0)

    def test_all_positive_split_reports_no_rates_and_never_a_fake_precision(self) -> None:
        """acd1k / cpd1k / camo_human are all-positive in meta.csv.

        With no negatives, fp == tn == 0, so the naive precision is tp/(tp+0)
        == 1.0 for *any* gate however bad, and "accuracy" is just recall.
        Printing those into a paper table would be a fabrication, so every rate
        is None and the split is flagged.
        """
        out = report.presence_metrics([0.9, 0.8, 0.1], [False, False, False])
        assert out["presence_single_class"] is True
        assert out["presence_precision"] is None
        assert out["presence_precision"] != 1.0
        for key in ("presence_recall", "presence_f1", "presence_accuracy", "presence_auc"):
            assert out[key] is None, key
        # The raw counts are still facts about what happened, so they survive.
        assert (out["presence_tp"], out["presence_fn"]) == (2, 1)
        assert (out["presence_fp"], out["presence_tn"]) == (0, 0)

    def test_all_negative_split_is_also_single_class(self) -> None:
        out = report.presence_metrics([0.1, 0.9], [True, True])
        assert out["presence_single_class"] is True
        assert out["presence_precision"] is None
        assert (out["presence_fp"], out["presence_tn"]) == (1, 1)

    def test_empty_input_does_not_divide_by_zero(self) -> None:
        out = report.presence_metrics([], [])
        assert out["presence_accuracy"] is None
        assert out["presence_single_class"] is True
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

    def test_mislabelled_empty_gt_is_excluded_from_mask_means(self) -> None:
        """is_negative says positive but the mask is empty — both flags must agree.

        Otherwise s_measure's 1 - pred.mean() branch would join the positives
        mean and inflate S_alpha.
        """
        empty = make_pred("mislabelled", is_negative=True)
        rows = [
            report.metric_row(make_pred("pos", perfect=False)),
            report.metric_row(Prediction(stem="mislabelled", prob=empty.prob, gt=empty.gt,
                                         presence=0.9, is_negative=False)),
        ]
        out = report.aggregate(rows)
        assert out["n_positives"] == 1
        assert out["mask"]["S_alpha"] == pytest.approx(rows[0]["S_alpha"])

    def test_single_class_split_is_flagged_in_the_summary(self) -> None:
        rows = [report.metric_row(make_pred(s)) for s in ("a", "b")]
        assert report.aggregate(rows)["presence"]["presence_single_class"] is True


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

    def test_metrics_md_flags_a_single_class_split_and_names_the_missing_class(
        self, tmp_path: Path,
    ) -> None:
        summary = report.aggregate([report.metric_row(make_pred(s)) for s in ("a", "b")])
        summary.update({"run": "toy", "dataset": "acd1k", "split": "test"})
        path = tmp_path / "metrics.md"
        report.write_metrics_md(summary, path)
        text = path.read_text()
        assert "only one class" in text
        assert "negative (target-free)" in text
        assert "| presence_precision | n/a |" in text
        assert "| presence_precision | 1.0000 |" not in text

    def test_metrics_md_has_no_single_class_note_when_both_classes_present(
        self, tmp_path: Path,
    ) -> None:
        summary = report.aggregate([
            report.metric_row(make_pred("pos", presence=0.9)),
            report.metric_row(make_pred("neg", is_negative=True, presence=0.1)),
        ])
        summary.update({"run": "toy", "dataset": "mhcd", "split": "test"})
        path = tmp_path / "metrics.md"
        report.write_metrics_md(summary, path)
        assert "only one class" not in path.read_text()
