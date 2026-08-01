"""Correctness tests for chd.metrics, against hand-derivable cases.

Perfect prediction must hit the optimum of every metric (checked
analytically for S-measure and E-measure in the comments below, not just
asserted by faith). Degenerate cases (all-zero prediction, empty GT) are
pinned to their actual — sometimes counter-intuitive — behavior so a future
change to the implementation shows up as a diff, not a surprise in training.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chd import metrics  # noqa: E402

RNG = np.random.default_rng(0)


def make_mask(h: int = 40, w: int = 40, box: tuple[int, int, int, int] = (10, 10, 30, 30)) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.float64)
    y1, x1, y2, x2 = box
    mask[y1:y2, x1:x2] = 1.0
    return mask


# --------------------------------------------------------------------------
# perfect prediction -> optimum of every metric
# --------------------------------------------------------------------------

class TestPerfectPrediction:
    """gt is a 20x20 box in a 40x40 image (mu_gt = 0.25, strictly between 0
    and 1), so none of S-measure/E-measure's degenerate special-cases fire —
    the identity pred == gt is being tested on its own general-case formula.
    """

    def setup_method(self) -> None:
        self.gt = make_mask()
        self.pred = self.gt.copy()

    def test_mae_zero(self) -> None:
        assert metrics.mae(self.pred, self.gt) == 0.0

    def test_f_measure_all_variants_are_one(self) -> None:
        assert metrics.f_measure_mean(self.pred, self.gt) == pytest.approx(1.0)
        assert metrics.f_measure_max(self.pred, self.gt) == pytest.approx(1.0)
        assert metrics.f_measure_adaptive(self.pred, self.gt) == pytest.approx(1.0)

    def test_s_measure_is_one(self) -> None:
        # Derivation: pred == gt binary means, for any quadrant, x == y (same
        # mean) and cov(pred, gt) == var(pred), so alpha/beta in _ssim_patch
        # reduces to 4x^2*var / (2x^2 * 2var) == 1 exactly; the object score
        # similarly reduces to 2x/(x^2+1+2*sigma) with x=1, sigma=0 -> 1.
        assert metrics.s_measure(self.pred, self.gt) == pytest.approx(1.0, abs=1e-6)

    def test_e_measure_all_variants_are_one(self) -> None:
        # Derivation: 0 < mu_gt < 1, so align_gt is never exactly 0; with
        # pred == gt, align_fm == align_gt everywhere, so the alignment ratio
        # 2*a^2 / (2*a^2) == 1 at every pixel with no eps involved.
        assert metrics.e_measure_mean(self.pred, self.gt) == pytest.approx(1.0, abs=1e-6)
        assert metrics.e_measure_max(self.pred, self.gt) == pytest.approx(1.0, abs=1e-6)
        assert metrics.e_measure_adaptive(self.pred, self.gt) == pytest.approx(1.0, abs=1e-6)

    def test_boundary_f_measure_is_one(self) -> None:
        assert metrics.boundary_f_measure(self.pred, self.gt) == pytest.approx(1.0)

    def test_iou_and_dice_are_one(self) -> None:
        assert metrics.iou(self.pred, self.gt) == pytest.approx(1.0)
        assert metrics.dice(self.pred, self.gt) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# fully wrong prediction (inverted mask) -> should be near worst-case
# --------------------------------------------------------------------------

class TestInvertedPrediction:
    def setup_method(self) -> None:
        self.gt = make_mask()
        self.pred = 1.0 - self.gt

    def test_mae_is_one(self) -> None:
        assert metrics.mae(self.pred, self.gt) == pytest.approx(1.0)

    def test_f_measure_is_zero(self) -> None:
        assert metrics.f_measure_mean(self.pred, self.gt) == pytest.approx(0.0)
        assert metrics.f_measure_max(self.pred, self.gt) == pytest.approx(0.0)

    def test_iou_is_zero(self) -> None:
        assert metrics.iou(self.pred, self.gt) == pytest.approx(0.0)

    def test_s_measure_is_low(self) -> None:
        # not analytically pinned to a single value, but must be far from 1
        assert metrics.s_measure(self.pred, self.gt) < 0.3


# --------------------------------------------------------------------------
# known-geometry IoU / Dice (hand-computable)
# --------------------------------------------------------------------------

class TestKnownGeometry:
    def test_half_overlapping_boxes(self) -> None:
        # gt: rows/cols [0,20); pred: rows/cols [10,30) in a 40x40 canvas.
        # intersection = 10x20 = 200 (rows 10-20 overlap, cols 0-20 overlap
        # for gt's box, but pred spans cols 10-30) -> recompute precisely:
        gt = make_mask(40, 40, box=(0, 0, 20, 20))
        pred = make_mask(40, 40, box=(10, 10, 30, 30))
        inter = 10 * 10  # overlap region rows[10,20) x cols[10,20)
        union = 20 * 20 + 20 * 20 - inter
        assert metrics.iou(pred, gt) == pytest.approx(inter / union)
        assert metrics.dice(pred, gt) == pytest.approx(2 * inter / (20 * 20 + 20 * 20))

    def test_disjoint_boxes_zero_iou(self) -> None:
        gt = make_mask(40, 40, box=(0, 0, 10, 10))
        pred = make_mask(40, 40, box=(30, 30, 40, 40))
        assert metrics.iou(pred, gt) == 0.0
        assert metrics.dice(pred, gt) == 0.0


# --------------------------------------------------------------------------
# degenerate cases: pinned to actual (documented) behavior
# --------------------------------------------------------------------------

class TestDegenerateCases:
    def test_empty_gt_all_zero_pred_is_perfect(self) -> None:
        gt = np.zeros((20, 20))
        pred = np.zeros((20, 20))
        assert metrics.mae(pred, gt) == 0.0
        assert metrics.s_measure(pred, gt) == pytest.approx(1.0)
        assert metrics.e_measure_mean(pred, gt) == pytest.approx(1.0)

    def test_empty_gt_nonzero_pred_is_penalised(self) -> None:
        gt = np.zeros((20, 20))
        pred = np.full((20, 20), 0.8)
        assert metrics.s_measure(pred, gt) == pytest.approx(1 - 0.8)
        assert metrics.e_measure_mean(pred, gt) == pytest.approx(1 - 0.8)

    def test_all_zero_prediction_mae_equals_gt_mean(self) -> None:
        gt = make_mask()  # fg fraction = 0.25
        pred = np.zeros_like(gt)
        assert metrics.mae(pred, gt) == pytest.approx(gt.mean())

    def test_all_zero_prediction_f_measure_curve_is_zero_at_positive_thresholds(self) -> None:
        gt = make_mask()
        pred = np.zeros_like(gt)
        # every threshold > 0 makes pred_bin all-False -> precision=recall=0
        assert metrics.f_measure_mean(pred, gt) == 0.0
        assert metrics.f_measure_max(pred, gt) == 0.0


# --------------------------------------------------------------------------
# input handling
# --------------------------------------------------------------------------

class TestInputNormalisation:
    def test_accepts_0_255_gt(self) -> None:
        gt_float = make_mask()
        gt_255 = (gt_float * 255).astype(np.uint8)
        assert metrics.mae(gt_float, gt_255) == pytest.approx(0.0)

    def test_accepts_0_255_pred(self) -> None:
        gt = make_mask()
        pred_255 = (gt * 255).astype(np.uint8)
        assert metrics.mae(pred_255, gt) == pytest.approx(0.0)

    def test_all_metrics_bounded_zero_one_on_random_input(self) -> None:
        gt = (RNG.random((30, 30)) > 0.7).astype(np.float64)
        pred = RNG.random((30, 30))
        report = metrics.evaluate_all(pred, gt)
        for name, value in report.items():
            assert 0.0 - 1e-6 <= value <= 1.0 + 1e-6, f"{name}={value} out of [0,1]"


# --------------------------------------------------------------------------
# boundary F-measure specifics
# --------------------------------------------------------------------------

class TestBoundaryFMeasure:
    def test_one_pixel_shift_still_scores_high_with_default_tolerance(self) -> None:
        gt = make_mask(200, 200, box=(50, 50, 150, 150))
        pred = make_mask(200, 200, box=(51, 50, 151, 150))  # shifted by 1px
        # default tolerance ~0.75% of a 200*sqrt(2)~283 diagonal -> ~2px, so
        # a 1px shift should still score near-perfectly.
        assert metrics.boundary_f_measure(pred, gt) > 0.95

    def test_large_shift_scores_low(self) -> None:
        gt = make_mask(200, 200, box=(50, 50, 150, 150))
        pred = make_mask(200, 200, box=(120, 120, 180, 180))  # mostly non-overlapping
        assert metrics.boundary_f_measure(pred, gt) < 0.5

    def test_full_image_masks_have_no_boundary(self) -> None:
        gt = np.ones((20, 20))
        pred = np.ones((20, 20))
        assert metrics.boundary_f_measure(pred, gt) == 1.0
