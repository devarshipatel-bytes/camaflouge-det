"""Inference that produces probability maps at the ground truth's own resolution.

This is the one place the evaluation protocol deliberately differs from
``train.py``'s ``run_validation``. Training-time validation scores at
``img_size`` because it only needs a comparable number epoch to epoch. The
COD/SOD literature — and therefore the paper's comparison tables — scores
against the ground-truth mask at its **native** resolution. Predicting at
``img_size`` and then resizing the probability map back up is what makes our
numbers comparable to the published baselines.

The mask is re-read from disk rather than taken from the dataset item, because
``CHDDataset`` resizes it to ``img_size`` on load.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from chd.data.dataset import AugmentConfig, CHDDataset
from chd.data.manifest import load_gray
from chd.eval.runs import RunBundle


@dataclass
class Prediction:
    """One test image's prediction, at the ground truth's native resolution."""

    stem: str
    prob: np.ndarray  # (H, W) float32 in [0, 1]
    gt: np.ndarray  # (H, W) float32 in {0, 1}
    presence: float
    is_negative: bool


def resize_prob(prob: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    """Bilinearly resize a probability map to ``(H, W)``, clipped to [0, 1]."""
    height, width = size_hw
    if prob.shape == (height, width):
        return prob
    resized = cv2.resize(prob, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.clip(resized, 0.0, 1.0).astype(np.float32)


@torch.no_grad()
def predict_run(
    bundle: RunBundle,
    split: str = "test",
    data_root: str | Path | None = None,
    device: str = "cpu",
    limit: int | None = None,
    stems: list[str] | None = None,
) -> Iterator[Prediction]:
    """Yield one ``Prediction`` per image, streaming so memory stays flat.

    Streaming matters: the combined test split is 1150 images, and holding
    every native-resolution probability map and mask in memory at once would
    cost several GB.

    Output order follows the split file's stem order, not ``stems``' order:
    ``predict_run(bundle, stems=["b", "a"])`` still yields ``a`` before ``b``
    if that is the split file's order. ``stems`` only filters which images are
    used, it does not reorder them.
    """
    root = Path(data_root or getattr(bundle.config, "data_root", "data")) / bundle.config.dataset
    zero_pose = bool(getattr(bundle.config, "no_pose", False))
    dataset = CHDDataset(root, split, img_size=bundle.config.img_size,
                         augment=AugmentConfig(enabled=False), require_pose=not zero_pose)

    wanted = set(stems) if stems else None
    indices = [i for i, stem in enumerate(dataset.stems) if wanted is None or stem in wanted]
    if limit is not None:
        indices = indices[:limit]

    bundle.model.to(device)

    for index in indices:
        item = dataset[index]
        stem = item["stem"]

        image = item["image"].unsqueeze(0).to(device)
        pose = item["pose"].unsqueeze(0)
        if zero_pose:
            pose = torch.zeros_like(pose)
        pose = pose.to(device)

        outputs = bundle.model(image, pose)
        prob = bundle.model.predict_mask(outputs)[0, 0].float().cpu().numpy()
        presence = float(torch.sigmoid(outputs["presence_logit"]).flatten()[0].item())

        gt = (load_gray(root / "masks" / f"{stem}.png") > 127).astype(np.float32)
        yield Prediction(
            stem=stem,
            prob=resize_prob(prob.astype(np.float32), gt.shape),
            gt=gt,
            presence=presence,
            is_negative=bool(dataset.is_negative[stem]),
        )
