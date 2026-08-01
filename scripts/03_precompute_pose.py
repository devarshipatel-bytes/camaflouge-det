#!/usr/bin/env python3
"""Cache anatomical keypoint-heatmap priors for the AER module.

Runs a pose model once per image and renders 17 COCO-keypoint Gaussian
heatmaps, confidence-weighted (``c_k`` in the paper's ``P_ana`` notation), to
``data/<name>/pose/<stem>.npy`` as ``float16 [17, H/4, W/4]`` at each image's
**native** resolution — img-size stays a training-time knob, so this cache is
never invalidated by a resolution change.

This is intentionally the cheapest correct choice, not the paper's own setup:
the paper distills from an offline HRNet-W48 teacher; this uses YOLO11-pose,
which is far lighter and has no training-time cost since it's fully cached.
Camouflaged subjects naturally yield low keypoint confidence — that's
expected, not a bug, and is exactly why every heatmap is confidence-weighted
rather than binary: the AER module learns to trust the prior only where it's
actually informative.

An image with no detected person (below ``--conf``) gets an all-zero
heatmap, which is correct for presence-gate negatives and for camouflaged
subjects the pose model simply misses — the mask/edge losses still supervise
those pixels normally, only the anatomical prior is absent.

Example
-------
    python scripts/03_precompute_pose.py --dataset acd1k
    python scripts/03_precompute_pose.py --dataset all --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chd.data.manifest import SPLITS, image_size, read_split  # noqa: E402

DATASETS = ("acd1k", "cpd1k", "camo_human", "mhcd")
N_KEYPOINTS = 17
DOWNSAMPLE = 4


def render_heatmaps(keypoints_xy: np.ndarray, keypoints_conf: np.ndarray,
                    orig_h: int, orig_w: int, sigma: float = 2.0) -> np.ndarray:
    """``(n_people, 17, 2)`` xy + ``(n_people, 17)`` conf -> ``[17, H/4, W/4]`` float16.

    Multiple detected people are combined by taking, per keypoint channel and
    pixel, the max response across people — this is a scene-level anatomical
    prior, not a per-instance one, matching how the AER module consumes it
    (concatenated once per spatial location, not per detected identity).
    """
    out_h, out_w = max(1, round(orig_h / DOWNSAMPLE)), max(1, round(orig_w / DOWNSAMPLE))
    heat = np.zeros((N_KEYPOINTS, out_h, out_w), dtype=np.float32)
    if keypoints_xy.size == 0:
        return heat.astype(np.float16)

    scale_x, scale_y = out_w / orig_w, out_h / orig_h
    yy, xx = np.mgrid[0:out_h, 0:out_w]
    for person_xy, person_conf in zip(keypoints_xy, keypoints_conf, strict=True):
        for k in range(N_KEYPOINTS):
            conf = float(person_conf[k])
            if conf <= 0.05:
                continue
            cx, cy = person_xy[k, 0] * scale_x, person_xy[k, 1] * scale_y
            gaussian = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)) * conf
            np.maximum(heat[k], gaussian, out=heat[k])
    return heat.astype(np.float16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=(*DATASETS, "all"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--pose-model", default="yolo11n-pose.pt")
    parser.add_argument("--conf", type=float, default=0.10, help="minimum person-detection confidence")
    parser.add_argument("--sigma", type=float, default=2.0, help="Gaussian sigma in output-resolution pixels")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, help="process only the first N images per dataset (dry run)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.pose_model)

    targets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for name in targets:
        root = args.data_root / name
        if not (root / "meta.csv").exists():
            print(f"[pose] skipping {name}: {root}/meta.csv missing, run its prepare script first")
            continue

        stems = sorted({s for split in SPLITS for s in read_split(root, split)})
        if args.limit:
            stems = stems[: args.limit]
        pending = [s for s in stems if args.overwrite or not (root / "pose" / f"{s}.npy").exists()]
        (root / "pose").mkdir(parents=True, exist_ok=True)
        print(f"[pose] {name}: {len(stems)} images, {len(pending)} to render "
              f"({len(stems) - len(pending)} already cached)")

        n_detected = 0
        for start in tqdm(range(0, len(pending), args.batch_size), desc=f"pose:{name}", unit="batch"):
            batch = pending[start:start + args.batch_size]
            paths = [root / "images" / f"{s}.jpg" for s in batch]
            results = model(
                [str(p) for p in paths], conf=args.conf, classes=[0],
                device=args.device, verbose=False,
            )
            for stem, path, result in zip(batch, paths, results, strict=True):
                h, w = image_size(path)
                if result.keypoints is not None and len(result.keypoints.xy):
                    xy = result.keypoints.xy.cpu().numpy()
                    conf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None \
                        else np.ones(xy.shape[:2], dtype=np.float32)
                    n_detected += 1
                else:
                    xy = np.zeros((0, N_KEYPOINTS, 2), dtype=np.float32)
                    conf = np.zeros((0, N_KEYPOINTS), dtype=np.float32)
                heat = render_heatmaps(xy, conf, h, w, sigma=args.sigma)
                np.save(root / "pose" / f"{stem}.npy", heat)

        print(f"[pose] {name}: person detected in {n_detected}/{len(pending)} rendered images "
              f"({n_detected / len(pending):.1%})" if pending else f"[pose] {name}: nothing to do")


if __name__ == "__main__":
    main()
