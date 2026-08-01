#!/usr/bin/env python3
"""Zip a prepared dataset into a self-contained archive (symlinks dereferenced).

``data/combined/`` is a manifest of symlinks pointing back into the other
four prepared datasets — 1.2MB of symlinks standing in for ~5.2GB of real
image/mask/edge/pose data. A plain ``zip -r`` (or Windows' built-in
"Compress to zip") stores those symlinks as-is: on a filesystem/tool that
doesn't preserve or resolve them (which is most of the time on Windows
without Developer Mode), extraction produces a handful of KB of broken
shortcuts, not a training-ready dataset. This script always dereferences,
so the output zip needs nothing special to extract anywhere.

Example
-------
    python scripts/zip_dataset.py --dataset combined --out data/combined.zip
    python scripts/zip_dataset.py --dataset acd1k --out data/acd1k.zip
"""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="name under --data-root, e.g. combined, acd1k")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=None, help="default: <data-root>/<dataset>.zip")
    args = parser.parse_args()

    src = args.data_root / args.dataset
    if not src.is_dir():
        raise SystemExit(f"{src} does not exist")
    out = args.out or (args.data_root / f"{args.dataset}.zip")
    out.parent.mkdir(parents=True, exist_ok=True)

    files = [p for p in src.rglob("*") if p.is_file() or p.is_symlink()]
    print(f"[zip] {src} -> {out}  ({len(files)} entries, dereferencing symlinks)")

    n_symlinks = 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for i, path in enumerate(files):
            real = Path(os.path.realpath(path))  # resolves symlinks to the actual file on disk
            if path.is_symlink():
                n_symlinks += 1
            arcname = Path(args.dataset) / path.relative_to(src)
            zf.write(real, arcname)
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(files)}")

    size_gb = out.stat().st_size / 1e9
    print(f"[zip] done: {out}  ({size_gb:.2f} GB, {n_symlinks} symlinks dereferenced)")


if __name__ == "__main__":
    main()
