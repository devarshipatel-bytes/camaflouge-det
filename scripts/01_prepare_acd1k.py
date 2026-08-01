#!/usr/bin/env python3
"""Prepare ACD1K into the canonical layout.

ACD1K already ships 1:1 image/mask pairs with no train/test overlap — the
only real issue is that the **training** ground truth is saved as JPEG, so it
carries 248+ gray levels instead of two (measured: 0.11% of pixels ambiguous
at the mean, 0.47% at the max — thresholding at 127 throws away negligible
information). Test GT is already clean PNG.

ACD1K ships no validation split, so one is carved out of training.

Example
-------
    python scripts/01_prepare_acd1k.py --src dataset/ACD1K/dataset-splitM --out data/acd1k
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chd.data.manifest import DatasetWriter, ambiguity, binarize, image_size, load_gray  # noqa: E402


def discover(src: Path) -> list[dict]:
    samples = []
    for official, folder in (("train", "Training"), ("test", "Testing")):
        image_dir = src / folder / "images"
        gt_dir = src / folder / "GT"
        if not image_dir.is_dir():
            raise SystemExit(f"expected {image_dir} — is --src pointing at ACD1K/dataset-splitM?")
        for image_path in sorted(image_dir.glob("*.jpg")):
            gt_path = gt_dir / f"{image_path.stem}.png"
            if not gt_path.exists():
                gt_path = gt_dir / f"{image_path.stem}.jpg"
            if not gt_path.exists():
                raise SystemExit(f"{image_path.stem}: no GT found in {gt_dir}")
            samples.append({"stem": image_path.stem, "image": image_path, "mask": gt_path, "official": official})
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=Path("dataset/ACD1K/dataset-splitM"))
    parser.add_argument("--out", type=Path, default=Path("data/acd1k"))
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="process only the first N images (dry run)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    samples = discover(args.src)
    if args.limit:
        samples = samples[: args.limit]
    print(f"[acd1k] discovered {len(samples)} image/GT pairs")

    train_pool = [s for s in samples if s["official"] == "train"]
    test = [s for s in samples if s["official"] == "test"]
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(train_pool))
    n_val = max(1, round(len(train_pool) * args.val_frac))
    val_idx, train_idx = set(order[:n_val].tolist()), set(order[n_val:].tolist())

    writer = DatasetWriter(args.out, "acd1k", overwrite=args.overwrite)
    high_ambiguity: list[tuple[str, float]] = []
    size_mismatch: list[tuple[str, tuple[int, int], tuple[int, int]]] = []

    def add_one(sample: dict, split: str) -> None:
        stem = sample["stem"]
        if writer.has(stem) and not args.overwrite:
            return
        img_hw, gt_hw = image_size(sample["image"]), image_size(sample["mask"])
        if img_hw != gt_hw:
            # Width matches, only height differs by ~9% in the one case seen
            # (ACD1K/Testing/image785) — a corrupted pair shipped upstream,
            # not something we can safely repair: we don't know whether the
            # mismatch is a top- or bottom-crop, and guessing wrong would
            # silently misalign a mask that otherwise looks perfectly clean.
            size_mismatch.append((stem, img_hw, gt_hw))
            return
        gray = load_gray(sample["mask"])
        frac = ambiguity(gray)
        if frac > 0.01:
            high_ambiguity.append((stem, frac))
        writer.add(stem=stem, split=split, image_src=sample["image"], mask=binarize(gray), source="acd1k")

    for i, sample in enumerate(tqdm(train_pool, desc="acd1k train/val", unit="img")):
        add_one(sample, "val" if i in val_idx else "train")
    for sample in tqdm(test, desc="acd1k test", unit="img"):
        add_one(sample, "test")

    summary = writer.finalize()
    print(f"\n[acd1k] {summary}")
    if high_ambiguity:
        print(f"[acd1k] {len(high_ambiguity)} masks had >1% ambiguous pixels after thresholding, "
              f"worst: {max(high_ambiguity, key=lambda x: x[1])}")
    else:
        print("[acd1k] binarisation was clean everywhere (<1% ambiguous pixels per mask)")
    if size_mismatch:
        print(f"[acd1k] SKIPPED {len(size_mismatch)} pair(s) with image/mask size mismatch "
              "(upstream data defect, not repaired):")
        for stem, img_hw, gt_hw in size_mismatch:
            print(f"           {stem}: image {img_hw} vs GT {gt_hw}")


if __name__ == "__main__":
    main()
