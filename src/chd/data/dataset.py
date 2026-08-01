"""torch Dataset over the canonical prepared-dataset layout.

Reads ``data/<name>/{images,masks,edges,pose}`` + ``splits/<split>.txt`` +
``meta.csv``. Every image, mask, edge and pose heatmap is stored at its
**native resolution** (see ``chd.data.manifest``); resizing to ``img_size``
happens here, at load time, which is what keeps ``--img-size`` a pure
training-time knob with zero preprocessing cost to change.

Augmentation is hand-rolled with cv2 rather than a library pipeline so the
*exact same* random geometric transform (flip, rotation, scale, crop) is
guaranteed to apply identically to the image, mask, edge and the 17-channel
pose heatmap in one pass — including the pose channels, which most
off-the-shelf segmentation-augmentation pipelines aren't set up to carry
through. Color jitter is applied to the image only, never to mask/edge/pose.

Multi-scale training is deliberately NOT done per-item here: every item this
Dataset returns is a fixed ``img_size``, so a batch is always stackable.
Picking a random scale per training *step* (not per sample) and resizing the
whole already-collated batch is the training loop's job — see
``batch_resize`` below, called from ``train.py``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from chd.data.manifest import read_split

N_KEYPOINTS = 17


@dataclass
class AugmentConfig:
    hflip_prob: float = 0.5
    rotate_deg: float = 15.0
    scale_range: tuple[float, float] = (0.75, 1.25)
    color_jitter: float = 0.2  # brightness/contrast/saturation jitter strength, 0 disables
    enabled: bool = True


def _load_gray01(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise FileNotFoundError(path)
    return (arr > 127).astype(np.float32)


def _load_rgb(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def _resize(arr: np.ndarray, size: int, interpolation: int) -> np.ndarray:
    return cv2.resize(arr, (size, size), interpolation=interpolation)


def _color_jitter(image: np.ndarray, strength: float, rng: random.Random) -> np.ndarray:
    """Brightness/contrast/saturation only — never touches mask/edge/pose."""
    out = image.astype(np.float32)
    brightness = 1.0 + rng.uniform(-strength, strength)
    contrast = 1.0 + rng.uniform(-strength, strength)
    out = out * brightness
    mean = out.mean(axis=(0, 1), keepdims=True)
    out = (out - mean) * contrast + mean
    if strength > 0:
        gray = out.mean(axis=2, keepdims=True)
        saturation = 1.0 + rng.uniform(-strength, strength)
        out = gray + (out - gray) * saturation
    return np.clip(out, 0, 255).astype(np.uint8)


class CHDDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        img_size: int = 352,
        augment: AugmentConfig | None = None,
        require_pose: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.augment = augment if augment is not None else AugmentConfig(enabled=(split == "train"))
        self.require_pose = require_pose

        self.stems = read_split(self.root, split)
        if not self.stems:
            raise ValueError(f"no stems found for split {split!r} under {self.root}")

        negatives = set()
        meta_path = self.root / "meta.csv"
        if meta_path.exists():
            import csv
            with meta_path.open() as fh:
                for row in csv.DictReader(fh):
                    if row.get("is_negative", "0") == "1":
                        negatives.add(row["stem"])
        self.is_negative = {stem: (stem in negatives) for stem in self.stems}

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        stem = self.stems[index]
        rng = random.Random()  # per-call RNG: safe under multi-worker DataLoader forking

        image = _load_rgb(self.root / "images" / f"{stem}.jpg")
        mask = _load_gray01(self.root / "masks" / f"{stem}.png")
        edge_path = self.root / "edges" / f"{stem}.png"
        edge = _load_gray01(edge_path) if edge_path.exists() else np.zeros_like(mask)

        pose_path = self.root / "pose" / f"{stem}.npy"
        if pose_path.exists():
            pose = np.load(pose_path).astype(np.float32).transpose(1, 2, 0)  # H,W,17
        elif self.require_pose:
            raise FileNotFoundError(f"pose cache missing for {stem} — run scripts/03_precompute_pose.py")
        else:
            h, w = mask.shape
            pose = np.zeros((h, w, N_KEYPOINTS), dtype=np.float32)
        if pose.shape[:2] != mask.shape:
            pose = cv2.resize(pose, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)

        if self.augment.enabled:
            image, mask, edge, pose = self._augment_geometric(image, mask, edge, pose, rng)
            if self.augment.color_jitter > 0:
                image = _color_jitter(image, self.augment.color_jitter, rng)

        image = _resize(image, self.img_size, cv2.INTER_LINEAR)
        mask = _resize(mask, self.img_size, cv2.INTER_NEAREST)
        edge = _resize(edge, self.img_size, cv2.INTER_NEAREST)
        pose = _resize(pose, max(1, self.img_size // 4), cv2.INTER_LINEAR)

        return {
            "stem": stem,
            "image": torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255.0,
            "mask": torch.from_numpy(mask.copy()).float().unsqueeze(0),
            "edge": torch.from_numpy(edge.copy()).float().unsqueeze(0),
            "pose": torch.from_numpy(pose.transpose(2, 0, 1).copy()).float(),
            "is_negative": torch.tensor(float(self.is_negative[stem])),
        }

    def _augment_geometric(
        self, image: np.ndarray, mask: np.ndarray, edge: np.ndarray, pose: np.ndarray, rng: random.Random,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.augment
        h, w = mask.shape

        if rng.random() < cfg.hflip_prob:
            image, mask, edge, pose = image[:, ::-1], mask[:, ::-1], edge[:, ::-1], pose[:, ::-1]

        angle = rng.uniform(-cfg.rotate_deg, cfg.rotate_deg)
        scale = rng.uniform(*cfg.scale_range)
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)

        image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        mask = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        edge = cv2.warpAffine(edge, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        pose = cv2.warpAffine(pose, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if pose.ndim == 2:  # cv2 squeezes a single remaining channel
            pose = pose[:, :, None]
        return np.ascontiguousarray(image), np.ascontiguousarray(mask), np.ascontiguousarray(edge), np.ascontiguousarray(pose)


def collate(batch: list[dict]) -> dict:
    out = {}
    for key in batch[0]:
        if key == "stem":
            out[key] = [item[key] for item in batch]
        else:
            out[key] = torch.stack([item[key] for item in batch])
    return out


def batch_resize(batch: dict, scale: float) -> dict:
    """Resize an already-collated batch by ``scale``, rounded to a multiple of 32.

    This is how multi-scale training is applied: once per step, on the whole
    batch, after collation — never per-item (see the module docstring for why
    per-item scaling would break ``torch.stack``).
    """
    if scale == 1.0:
        return batch
    import torch.nn.functional as F

    _, _, h, w = batch["image"].shape
    new_h = max(32, int(round(h * scale / 32.0)) * 32)
    new_w = max(32, int(round(w * scale / 32.0)) * 32)
    if (new_h, new_w) == (h, w):
        return batch

    resized = dict(batch)
    resized["image"] = F.interpolate(batch["image"], size=(new_h, new_w), mode="bilinear", align_corners=False)
    resized["mask"] = F.interpolate(batch["mask"], size=(new_h, new_w), mode="nearest")
    resized["edge"] = F.interpolate(batch["edge"], size=(new_h, new_w), mode="nearest")
    resized["pose"] = F.interpolate(batch["pose"], size=(new_h // 4, new_w // 4), mode="bilinear", align_corners=False)
    return resized
