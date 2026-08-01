#!/usr/bin/env python3
"""Integrity gate for every prepared dataset. Training refuses to start until this passes.

Checks, per dataset:
  * every stem listed in a split has an image, a mask and an edge on disk
  * masks are strictly binary {0, 255} and match their image's dimensions
  * splits are pairwise disjoint and cover exactly the rows in meta.csv
  * no positive sample has an empty mask (that would be a silent negative)
  * negatives are explicitly flagged, not merely empty
  * pose cache, if present, is complete and correctly shaped

Example
-------
    python scripts/05_verify_datasets.py --datasets camo_human mhcd
    python scripts/05_verify_datasets.py --all
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chd.data.manifest import SPLITS, image_size, read_split  # noqa: E402

KNOWN = ("acd1k", "camo_human", "cpd1k", "mhcd", "combined")


class Report:
    def __init__(self, name: str) -> None:
        self.name = name
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.stats: dict[str, object] = {}

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    @property
    def ok(self) -> bool:
        return not self.errors


def verify(root: Path, name: str, *, sample: int | None, check_pose: bool) -> Report:
    report = Report(name)
    if not root.is_dir():
        report.error(f"{root} does not exist — has it been prepared?")
        return report

    meta_path = root / "meta.csv"
    if not meta_path.exists():
        report.error("meta.csv missing")
        return report
    with meta_path.open() as fh:
        meta = {row["stem"]: row for row in csv.DictReader(fh)}
    if not meta:
        report.error("meta.csv is empty")
        return report

    # ---- splits -----------------------------------------------------------
    splits = {s: read_split(root, s) for s in SPLITS}
    for split, stems in splits.items():
        if len(stems) != len(set(stems)):
            report.error(f"splits/{split}.txt contains duplicate stems")
    names = list(SPLITS)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            clash = set(splits[a]) & set(splits[b])
            if clash:
                report.error(f"splits {a} and {b} overlap on {len(clash)} stems, "
                             f"e.g. {sorted(clash)[:3]}")

    listed = {stem for stems in splits.values() for stem in stems}
    if listed - meta.keys():
        report.error(f"{len(listed - meta.keys())} stems in splits are absent from meta.csv")
    if meta.keys() - listed:
        report.error(f"{len(meta.keys() - listed)} stems in meta.csv are in no split")
    if not splits["train"]:
        report.error("train split is empty")

    # ---- files ------------------------------------------------------------
    stems = sorted(listed)
    if sample and len(stems) > sample:
        rng = np.random.default_rng(0)
        inspect = sorted(rng.choice(stems, size=sample, replace=False).tolist())
        report.warn(f"pixel checks ran on a {sample}-image sample of {len(stems)}")
    else:
        inspect = stems

    empty_positive, non_binary, size_mismatch, missing = 0, 0, 0, 0
    fg_fracs: list[float] = []

    for stem in tqdm(inspect, desc=f"verify {name}", unit="img", leave=False):
        image_path = root / "images" / f"{stem}.jpg"
        mask_path = root / "masks" / f"{stem}.png"
        edge_path = root / "edges" / f"{stem}.png"
        if not (image_path.exists() and mask_path.exists() and edge_path.exists()):
            missing += 1
            continue

        h, w = image_size(image_path)
        with Image.open(mask_path) as im:
            mask = np.array(im.convert("L"))
        if mask.shape != (h, w):
            size_mismatch += 1
            continue
        levels = np.unique(mask)
        if not np.isin(levels, (0, 255)).all():
            non_binary += 1
            continue

        fraction = float((mask > 0).mean())
        is_negative = meta[stem].get("is_negative", "0") == "1"
        if fraction == 0.0 and not is_negative:
            empty_positive += 1
        if not is_negative:
            fg_fracs.append(fraction)

    for count, message in (
        (missing, "stems missing an image/mask/edge file"),
        (size_mismatch, "masks whose size differs from their image"),
        (non_binary, "masks that are not strictly binary"),
        (empty_positive, "empty masks not flagged as negatives"),
    ):
        if count:
            report.error(f"{count} {message}")

    # ---- pose cache -------------------------------------------------------
    pose_dir = root / "pose"
    if pose_dir.is_dir():
        cached = {p.stem for p in pose_dir.glob("*.npy")}
        if missing_pose := listed - cached:
            report.warn(f"pose cache incomplete: {len(missing_pose)} stems unrendered")
        elif inspect:
            probe = np.load(pose_dir / f"{inspect[0]}.npy")
            if probe.ndim != 3 or probe.shape[0] != 17:
                report.error(f"pose cache has shape {probe.shape}, expected [17, H/4, W/4]")
    elif check_pose:
        report.error("pose/ directory missing — run scripts/03_precompute_pose.py")
    else:
        report.warn("pose cache not built yet (fine until training)")

    negatives = sum(row.get("is_negative", "0") == "1" for row in meta.values())
    report.stats = {
        "total": len(meta),
        "train/val/test": "/".join(str(len(splits[s])) for s in SPLITS),
        "negatives": negatives,
        "mean_fg": round(float(np.mean(fg_fracs)), 4) if fg_fracs else 0.0,
        "checked": len(inspect),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--datasets", nargs="+", default=None, help=f"subset of {KNOWN}")
    parser.add_argument("--all", action="store_true", help="verify every dataset under --data-root")
    parser.add_argument("--sample", type=int, default=None,
                        help="only pixel-check N random images per dataset (faster)")
    parser.add_argument("--require-pose", action="store_true")
    args = parser.parse_args()

    if args.all:
        targets = [p.name for p in sorted(args.data_root.iterdir()) if p.is_dir()]
    elif args.datasets:
        targets = args.datasets
    else:
        parser.error("pass --datasets NAME [NAME ...] or --all")

    reports = [
        verify(args.data_root / name, name, sample=args.sample, check_pose=args.require_pose)
        for name in targets
    ]

    width = max((len(r.name) for r in reports), default=10)
    print()
    for report in reports:
        status = "PASS" if report.ok else "FAIL"
        details = "  ".join(f"{k}={v}" for k, v in report.stats.items())
        print(f"[{status}] {report.name:<{width}}  {details}")
        for message in report.warnings:
            print(f"         warn: {message}")
        for message in report.errors:
            print(f"         ERROR: {message}")

    failed = [r.name for r in reports if not r.ok]
    print()
    if failed:
        print(f"{len(failed)} dataset(s) failed verification: {', '.join(failed)}")
        raise SystemExit(1)
    print(f"all {len(reports)} dataset(s) passed")


if __name__ == "__main__":
    main()
