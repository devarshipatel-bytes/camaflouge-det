#!/usr/bin/env python3
"""Build the combined dataset that unions ACD1K, CPD1K, CAMO-Human and MHCD.

The combined dataset is a **manifest that indexes the other four**, not a
copy: every image/mask/edge is a symlink back to the already-prepared source
dataset, so it costs almost no disk space and stays in sync if a source is
rebuilt. Splits are the straightforward union of each source's own split —
combined train = union of the four training splits, and so on — so the
cross-dataset evaluation matrix compares against the *same* test sets every
other model is scored on.

Two safety checks matter more here than in any single-source script:

1. **Stem collisions.** ACD1K and MHCD both use bare numeric-ish stems
   (``image100``, ``000001``); left alone they could collide. Every combined
   stem is prefixed with its source (``acd1k__image100``).

2. **Cross-dataset near-duplicates.** COD/CHD benchmarks are frequently built
   from overlapping stock-photo pools, so the same underlying photograph can
   end up in two different "independent" datasets. If a copy of a test image
   from any of the four sources also sits in another source's train or val
   split, the combined model would train on data leaked from a test set it
   is later evaluated on. This is checked with a difference-hash (dHash) over
   every train/val image against every test image across all four sources —
   not just within the combined set — because the leak is real regardless of
   whether the combined model is the one being scored.

Example
-------
    python scripts/02_build_combined.py --data-root data --out data/combined
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chd.data.manifest import SPLITS, read_split  # noqa: E402

SOURCES = ("acd1k", "cpd1k", "camo_human", "mhcd")

#: Two images with a dHash Hamming distance at or below this are treated as
#: *candidate* near-duplicates. dHash alone is not reliable at this margin —
#: verified by hand: a fighter jet over desert mountains and an unrelated
#: snowy mountain scene hit Hamming=5 purely because both are a light sky
#: over a ridge silhouette. Every candidate is re-checked against a color
#: thumbnail below before being trusted (see ``looks_like_duplicate``).
DUPLICATE_HAMMING_THRESHOLD = 5

#: Second-signal confirmation: mean absolute difference between two 24x24
#: RGB thumbnails, normalised to [0, 1]. Verified pairs: four genuine
#: duplicates (same ghillie-suit photo, same video frame, re-crops of the
#: same stock photo) all scored <= 0.05; the jet/mountain false positive
#: scored ~0.22. The gap is wide, so a lenient cutoff well below the false
#: positive is enough without needing a tighter dHash threshold that would
#: risk missing genuine but differently-exposed re-uploads.
COLOR_DIFF_THRESHOLD = 0.10

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def dhash(path: Path, hash_size: int = 8) -> np.ndarray:
    """64-bit difference hash, packed into 8 bytes."""
    with Image.open(path) as im:
        small = np.asarray(im.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    return np.packbits(bits)


def color_thumb(path: Path, size: int = 24) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize((size, size), Image.LANCZOS), dtype=np.float32) / 255.0


def looks_like_duplicate(path_a: Path, path_b: Path) -> tuple[bool, float]:
    diff = float(np.abs(color_thumb(path_a) - color_thumb(path_b)).mean())
    return diff <= COLOR_DIFF_THRESHOLD, diff


def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distance between two (N, 8) and (M, 8) packed-bit arrays."""
    xor = a[:, None, :] ^ b[None, :, :]
    return _POPCOUNT[xor].sum(axis=-1)


def load_meta(root: Path) -> dict[str, dict]:
    with (root / "meta.csv").open() as fh:
        return {row["stem"]: row for row in csv.DictReader(fh)}


def relative_symlink(dst: Path, src: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(os.path.relpath(src, dst.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("data/combined"))
    parser.add_argument("--sources", nargs="+", default=list(SOURCES))
    parser.add_argument("--skip-dedup", action="store_true", help="skip the cross-dataset dHash pass (faster)")
    args = parser.parse_args()

    per_source: dict[str, dict[str, dict]] = {}
    per_source_splits: dict[str, dict[str, list[str]]] = {}
    for name in args.sources:
        root = args.data_root / name
        if not (root / "meta.csv").exists():
            raise SystemExit(f"{root}/meta.csv missing — run 01_prepare_{name}.py first")
        per_source[name] = load_meta(root)
        per_source_splits[name] = {s: read_split(root, s) for s in SPLITS}
        print(f"[combined] {name}: "
              f"{sum(len(v) for v in per_source_splits[name].values())} images "
              f"({', '.join(f'{s}={len(v)}' for s, v in per_source_splits[name].items())})")

    # ---- stem collision guard (belt-and-braces; prefixing already prevents
    # this, but a bug in the prefix scheme should fail loudly, not silently
    # merge two unrelated images) --------------------------------------------
    combined_stem = lambda name, stem: f"{name}__{stem}"  # noqa: E731
    seen: set[str] = set()
    for name, meta in per_source.items():
        for stem in meta:
            cstem = combined_stem(name, stem)
            if cstem in seen:
                raise SystemExit(f"stem collision: {cstem} already exists")
            seen.add(cstem)

    # ---- cross-dataset near-duplicate check --------------------------------
    dropped: list[tuple[str, str, int]] = []
    dropped_train_keys: set[str] = set()
    if not args.skip_dedup:
        trainval: list[tuple[str, str]] = []  # (source, stem)
        test: list[tuple[str, str]] = []
        for name, splits in per_source_splits.items():
            trainval += [(name, s) for s in splits["train"] + splits["val"]]
            test += [(name, s) for s in splits["test"]]
        print(f"\n[combined] hashing {len(trainval)} train/val + {len(test)} test images for cross-dataset dupes...")

        def hash_all(items: list[tuple[str, str]]) -> np.ndarray:
            out = np.zeros((len(items), 8), dtype=np.uint8)
            for i, (name, stem) in enumerate(tqdm(items, desc="dhash", unit="img")):
                out[i] = dhash(args.data_root / name / "images" / f"{stem}.jpg")
            return out

        trainval_hashes = hash_all(trainval)
        test_hashes = hash_all(test)

        # Chunk over test images to bound peak memory on the (N, M, 8) XOR buffer.
        candidates: list[tuple[int, int, int]] = []  # (trainval_idx, test_idx, hamming)
        chunk = 200
        for start in range(0, len(test), chunk):
            block = test_hashes[start:start + chunk]
            dist = hamming_matrix(trainval_hashes, block)
            hits = np.argwhere(dist <= DUPLICATE_HAMMING_THRESHOLD)
            for ti, bj in hits:
                test_name, _ = test[start + bj]
                train_name, _ = trainval[ti]
                if train_name == test_name:
                    continue  # within-dataset train/test overlap is that dataset's own concern, already split-checked
                candidates.append((int(ti), start + int(bj), int(dist[ti, bj])))

        print(f"[combined] {len(candidates)} dHash candidate(s), confirming with a color-thumbnail check...")
        rejected_false_positive = 0
        drop_indices: set[int] = set()
        for ti, tj, d in tqdm(candidates, desc="confirm", unit="pair"):
            train_name, train_stem = trainval[ti]
            test_name, test_stem = test[tj]
            train_path = args.data_root / train_name / "images" / f"{train_stem}.jpg"
            test_path = args.data_root / test_name / "images" / f"{test_stem}.jpg"
            is_dup, color_diff = looks_like_duplicate(train_path, test_path)
            if is_dup:
                drop_indices.add(ti)
                dropped.append((f"{train_name}__{train_stem}", f"{test_name}__{test_stem}", d, color_diff))
            else:
                rejected_false_positive += 1
        dropped_train_keys = {f"{trainval[i][0]}__{trainval[i][1]}" for i in drop_indices}

        if rejected_false_positive:
            print(f"[combined] dHash flagged {rejected_false_positive} pair(s) that the color check "
                  "ruled out as false positives (not dropped)")
        if dropped:
            print(f"[combined] confirmed {len(dropped)} cross-dataset near-duplicate pair(s):")
            for train_key, test_key, d, cdiff in dropped[:20]:
                print(f"           {train_key}  ~=  {test_key}   (hamming={d}, color_diff={cdiff:.3f})")
            if len(dropped) > 20:
                print(f"           ... and {len(dropped) - 20} more (see console/log above for full detail)")
        else:
            print("[combined] no confirmed cross-dataset near-duplicates")

    # ---- assemble splits ----------------------------------------------------
    out_splits: dict[str, list[str]] = {s: [] for s in SPLITS}
    meta_rows: list[dict] = []
    for name in args.sources:
        meta = per_source[name]
        for split, stems in per_source_splits[name].items():
            for stem in stems:
                cstem = combined_stem(name, stem)
                if split != "test" and cstem in dropped_train_keys:
                    continue
                out_splits[split].append(cstem)
                row = meta[stem]
                meta_rows.append({**row, "stem": cstem, "split": split, "source": name,
                                  "extra": f"source_stem={stem};{row.get('extra', '')}"})

    (args.out / "splits").mkdir(parents=True, exist_ok=True)
    for split, stems in out_splits.items():
        (args.out / "splits" / f"{split}.txt").write_text("\n".join(sorted(stems)) + ("\n" if stems else ""))
    with (args.out / "meta.csv").open("w", newline="") as fh:
        fieldnames = ["stem", "split", "h", "w", "fg_frac", "n_components", "source", "is_negative", "extra"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(meta_rows, key=lambda r: r["stem"]):
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    # ---- symlink images/masks/edges (skip the ones we dropped) -------------
    n_links = 0
    for name in args.sources:
        source_root = args.data_root / name
        for split, stems in per_source_splits[name].items():
            for stem in stems:
                cstem = combined_stem(name, stem)
                if split != "test" and cstem in dropped_train_keys:
                    continue
                for sub, ext in (("images", "jpg"), ("masks", "png"), ("edges", "png"), ("pose", "npy")):
                    src = source_root / sub / f"{stem}.{ext}"
                    if src.exists():
                        relative_symlink(args.out / sub / f"{cstem}.{ext}", src)
                        n_links += 1

    print(f"\n[combined] wrote {sum(len(v) for v in out_splits.values())} entries "
          f"({n_links} symlinks) to {args.out}")
    print(f"[combined] splits: " + ", ".join(f"{s}={len(v)}" for s, v in out_splits.items()))
    if dropped:
        with (args.out / "dedup_dropped.csv").open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["train_key", "test_key", "hamming", "color_diff"])
            writer.writerows(dropped)
        print(f"[combined] excluded {len(dropped_train_keys)} train/val image(s) that duplicated a test image "
              f"from another source (full list: {args.out / 'dedup_dropped.csv'})")


if __name__ == "__main__":
    main()
