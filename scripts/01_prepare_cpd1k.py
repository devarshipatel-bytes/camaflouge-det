#!/usr/bin/env python3
"""Prepare CPD1K into the canonical layout.

CPD1K is already the best-prepared of the four sources: clean binary masks,
precomputed edges, and a clip-aware 657/128/215 split with zero clips
straddling train/val/test (verified in ``prepare_stats.json``). Because it is
video frames (260 clips, 1-12 frames each), that split must be reused
**verbatim** — reshuffling at the frame level would leak near-duplicate
frames from the same clip across splits.

The per-image camouflage family label (woodland/desert/digital/urban/snow)
from ``pattern_labels.csv`` is carried into ``meta.csv`` for the dataset
visualisations, since that is exactly where CPD1K's composition is most
interesting to plot.

Example
-------
    python scripts/01_prepare_cpd1k.py --src dataset/cpd1k --out data/cpd1k
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chd.data.manifest import DatasetWriter, binarize, load_gray  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=Path("dataset/cpd1k"))
    parser.add_argument("--out", type=Path, default=Path("data/cpd1k"))
    parser.add_argument("--limit", type=int, help="process only the first N images (dry run)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    splits_dir = args.src / "splits"
    split_of: dict[str, str] = {}
    for split in ("train", "val", "test"):
        path = splits_dir / f"{split}.txt"
        if not path.exists():
            raise SystemExit(f"expected {path} — is --src pointing at the cpd1k root?")
        for stem in path.read_text().splitlines():
            stem = stem.strip()
            if stem:
                split_of[stem] = split
    print(f"[cpd1k] loaded shipped split for {len(split_of)} stems "
          f"(train={sum(v=='train' for v in split_of.values())} "
          f"val={sum(v=='val' for v in split_of.values())} "
          f"test={sum(v=='test' for v in split_of.values())})")

    labels_path = args.src / "pattern_labels.csv"
    family: dict[str, str] = {}
    if labels_path.exists():
        with labels_path.open() as fh:
            for row in csv.DictReader(fh):
                family[row["stem"]] = row.get("family_name", "")

    stems = sorted(split_of)
    if args.limit:
        stems = stems[: args.limit]

    writer = DatasetWriter(args.out, "cpd1k", overwrite=args.overwrite)
    unmatched_clip_check = 0
    for stem in tqdm(stems, desc="cpd1k", unit="img"):
        if writer.has(stem) and not args.overwrite:
            continue
        image_path = args.src / "images" / f"{stem}.jpg"
        mask_path = args.src / "masks" / f"{stem}.png"
        if not (image_path.exists() and mask_path.exists()):
            unmatched_clip_check += 1
            continue
        writer.add(
            stem=stem,
            split=split_of[stem],
            image_src=image_path,
            mask=binarize(load_gray(mask_path)),
            source="cpd1k",
            extra=f"family={family.get(stem, '')}",
        )

    summary = writer.finalize()
    print(f"\n[cpd1k] {summary}")
    if unmatched_clip_check:
        print(f"[cpd1k] {unmatched_clip_check} stems in the split files had no image/mask on disk")

    # writer.add() already derived an edge from each binarized mask, but
    # CPD1K ships its own precomputed edges — always prefer those over the
    # derived ones, regardless of --overwrite (which only gates re-copying
    # the image/mask, not this cheap final pass).
    edge_src = args.src / "edges"
    edge_dst = args.out / "edges"
    copied = 0
    for stem in stems:
        src_edge = edge_src / f"{stem}.png"
        if src_edge.exists():
            (edge_dst / f"{stem}.png").write_bytes(src_edge.read_bytes())
            copied += 1
    print(f"[cpd1k] replaced {copied} derived edges with the dataset's own precomputed edges")


if __name__ == "__main__":
    main()
