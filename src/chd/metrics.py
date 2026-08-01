"""COD/CHD evaluation metrics: MAE, F-beta, S-alpha, E-phi, boundary F-measure.

Operates on numpy arrays: ``pred`` is a continuous saliency map in [0, 1],
``gt`` is a binary {0, 1} (or {0, 255}) ground-truth mask, same shape.
Implements the standard formulas used across the COD/SOD literature (Fan et
al. 2017 S-measure, Fan et al. 2018 E-measure), not a novel variant, so
numbers are comparable to published baselines.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

EPS = 1e-8


def _binarize_gt(gt: np.ndarray) -> np.ndarray:
    gt = np.asarray(gt, dtype=np.float64)
    if gt.max() > 1.0:
        gt = gt / 255.0
    return (gt > 0.5).astype(np.float64)


def _as_prob(pred: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)
    if pred.max() > 1.0:
        pred = pred / 255.0
    return np.clip(pred, 0.0, 1.0)


# --------------------------------------------------------------------------
# MAE
# --------------------------------------------------------------------------

def mae(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.abs(_as_prob(pred) - _binarize_gt(gt))))


# --------------------------------------------------------------------------
# F-beta
# --------------------------------------------------------------------------

def _f_measure_at_threshold(pred_bin: np.ndarray, gt: np.ndarray, beta_sq: float = 0.3) -> float:
    tp = float(np.sum(pred_bin * gt))
    precision = tp / (pred_bin.sum() + EPS)
    recall = tp / (gt.sum() + EPS)
    if precision + recall == 0:
        return 0.0
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall + EPS)


def f_measure_curve(pred: np.ndarray, gt: np.ndarray, beta_sq: float = 0.3, n_thresholds: int = 255) -> np.ndarray:
    pred, gt = _as_prob(pred), _binarize_gt(gt)
    thresholds = np.linspace(0, 1, n_thresholds + 2)[1:-1]
    return np.array([_f_measure_at_threshold(pred >= t, gt, beta_sq) for t in thresholds])


def f_measure_mean(pred: np.ndarray, gt: np.ndarray, beta_sq: float = 0.3) -> float:
    return float(f_measure_curve(pred, gt, beta_sq).mean())


def f_measure_max(pred: np.ndarray, gt: np.ndarray, beta_sq: float = 0.3) -> float:
    return float(f_measure_curve(pred, gt, beta_sq).max())


def f_measure_adaptive(pred: np.ndarray, gt: np.ndarray, beta_sq: float = 0.3) -> float:
    pred, gt = _as_prob(pred), _binarize_gt(gt)
    threshold = min(1.0, 2.0 * pred.mean())
    return _f_measure_at_threshold(pred >= threshold, gt, beta_sq)


# --------------------------------------------------------------------------
# S-measure (Fan et al., 2017)
# --------------------------------------------------------------------------

def _object_score(pred_region: np.ndarray) -> float:
    x = pred_region.mean()
    sigma = pred_region.std()
    return float(2 * x / (x ** 2 + 1 + 2 * sigma + EPS))


def _s_object(pred: np.ndarray, gt: np.ndarray) -> float:
    u = gt.mean()
    fg = pred[gt > 0.5]
    bg = pred[gt <= 0.5]
    o_fg = _object_score(fg) if fg.size else 0.0
    o_bg = _object_score(1 - bg) if bg.size else 0.0
    return u * o_fg + (1 - u) * o_bg


def _centroid(gt: np.ndarray) -> tuple[int, int]:
    if gt.sum() == 0:
        h, w = gt.shape
        return h // 2, w // 2
    ys, xs = np.nonzero(gt)
    return int(round(ys.mean())), int(round(xs.mean()))


def _ssim_patch(pred: np.ndarray, gt: np.ndarray) -> float:
    h, w = pred.shape
    n = h * w
    if n <= 1:
        return 1.0 if np.array_equal(pred, gt) else 0.0
    x, y = pred.mean(), gt.mean()
    vx = pred.var(ddof=1) if n > 1 else 0.0
    vy = gt.var(ddof=1) if n > 1 else 0.0
    cov = np.mean((pred - x) * (gt - y)) * n / (n - 1)

    if x == 0 and y == 0:
        return 1.0
    if x == 0 or y == 0:
        return 0.0
    alpha = 4 * x * y * cov
    beta = (x ** 2 + y ** 2) * (vx + vy)
    return float(alpha / (beta + EPS)) if beta != 0 else 1.0


def _s_region(pred: np.ndarray, gt: np.ndarray) -> float:
    h, w = gt.shape
    cy, cx = _centroid(gt)
    cy, cx = max(1, min(h - 1, cy)), max(1, min(w - 1, cx))

    area = h * w
    weights = [
        (cy * cx) / area,
        (cy * (w - cx)) / area,
        ((h - cy) * cx) / area,
        ((h - cy) * (w - cx)) / area,
    ]
    slices = [
        (slice(0, cy), slice(0, cx)), (slice(0, cy), slice(cx, w)),
        (slice(cy, h), slice(0, cx)), (slice(cy, h), slice(cx, w)),
    ]
    score = 0.0
    for weight, sl in zip(weights, slices, strict=True):
        score += weight * _ssim_patch(pred[sl], gt[sl])
    return score


def s_measure(pred: np.ndarray, gt: np.ndarray, alpha: float = 0.5) -> float:
    pred, gt = _as_prob(pred), _binarize_gt(gt)
    y = gt.mean()
    if y == 0:
        return float(1 - pred.mean())
    if y == 1:
        return float(pred.mean())
    return float(alpha * _s_object(pred, gt) + (1 - alpha) * _s_region(pred, gt))


# --------------------------------------------------------------------------
# E-measure (Fan et al., 2018)
# --------------------------------------------------------------------------

def _enhanced_alignment(pred_bin: np.ndarray, gt: np.ndarray) -> float:
    mu_fm, mu_gt = pred_bin.mean(), gt.mean()
    align_fm, align_gt = pred_bin - mu_fm, gt - mu_gt
    numerator = 2 * align_gt * align_fm
    denominator = align_gt ** 2 + align_fm ** 2 + EPS
    align_matrix = numerator / denominator
    enhanced = ((align_matrix + 1) ** 2) / 4
    return float(enhanced.mean())


def e_measure_curve(pred: np.ndarray, gt: np.ndarray, n_thresholds: int = 255) -> np.ndarray:
    pred, gt = _as_prob(pred), _binarize_gt(gt)
    if gt.sum() == 0:
        return np.array([1.0 - pred.mean()])
    thresholds = np.linspace(0, 1, n_thresholds + 2)[1:-1]
    return np.array([_enhanced_alignment(pred >= t, gt) for t in thresholds])


def e_measure_mean(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(e_measure_curve(pred, gt).mean())


def e_measure_max(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(e_measure_curve(pred, gt).max())


def e_measure_adaptive(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = _as_prob(pred), _binarize_gt(gt)
    if gt.sum() == 0:
        return float(1.0 - pred.mean())
    threshold = min(1.0, 2.0 * pred.mean())
    return _enhanced_alignment(pred >= threshold, gt)


# --------------------------------------------------------------------------
# boundary F-measure (F^bd)
# --------------------------------------------------------------------------

def _boundary_of(mask: np.ndarray) -> np.ndarray:
    solid = mask > 0.5
    if not solid.any() or solid.all():
        return np.zeros_like(solid)
    inner = ndimage.binary_erosion(solid, border_value=0)
    return solid & ~inner


def boundary_f_measure(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5,
                       tolerance_frac: float = 0.0075) -> float:
    """Relaxed boundary F-measure. ``tolerance_frac`` is a fraction of the image diagonal."""
    gt = _binarize_gt(gt)
    pred_bin = _as_prob(pred) >= threshold

    gt_boundary = _boundary_of(gt)
    pred_boundary = _boundary_of(pred_bin)
    if not gt_boundary.any() and not pred_boundary.any():
        return 1.0
    if not gt_boundary.any() or not pred_boundary.any():
        return 0.0

    h, w = gt.shape
    tol = max(1, round(tolerance_frac * float(np.hypot(h, w))))

    dist_to_gt = ndimage.distance_transform_edt(~gt_boundary)
    dist_to_pred = ndimage.distance_transform_edt(~pred_boundary)

    precision = float((dist_to_gt[pred_boundary] <= tol).mean())
    recall = float((dist_to_pred[gt_boundary] <= tol).mean())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# --------------------------------------------------------------------------
# IoU / Dice (auxiliary, not in the paper's headline table but useful)
# --------------------------------------------------------------------------

def iou(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> float:
    pred_bin, gt = _as_prob(pred) >= threshold, _binarize_gt(gt) > 0.5
    inter = float(np.logical_and(pred_bin, gt).sum())
    union = float(np.logical_or(pred_bin, gt).sum())
    return inter / union if union > 0 else 1.0


def dice(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> float:
    pred_bin, gt = _as_prob(pred) >= threshold, _binarize_gt(gt) > 0.5
    inter = float(np.logical_and(pred_bin, gt).sum())
    denom = pred_bin.sum() + gt.sum()
    return 2 * inter / denom if denom > 0 else 1.0


# --------------------------------------------------------------------------


def evaluate_all(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """One-shot report of every metric used in the paper's comparison table."""
    return {
        "MAE": mae(pred, gt),
        "F_beta_mean": f_measure_mean(pred, gt),
        "F_beta_max": f_measure_max(pred, gt),
        "F_beta_adaptive": f_measure_adaptive(pred, gt),
        "S_alpha": s_measure(pred, gt),
        "E_phi_mean": e_measure_mean(pred, gt),
        "E_phi_max": e_measure_max(pred, gt),
        "E_phi_adaptive": e_measure_adaptive(pred, gt),
        "F_bd": boundary_f_measure(pred, gt),
        "IoU": iou(pred, gt),
        "Dice": dice(pred, gt),
    }
