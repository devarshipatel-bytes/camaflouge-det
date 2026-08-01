"""Canonical on-disk dataset format shared by every ``scripts/01_prepare_*`` script.

Every prepared dataset lands in the same shape so the training pipeline never
needs to know which corpus a sample came from::

    data/<name>/
      images/<stem>.jpg     RGB, NATIVE resolution (--img-size resizes at load time)
      masks/<stem>.png      strictly binary {0, 255}
      edges/<stem>.png      strictly binary {0, 255}, derived from the mask
      pose/<stem>.npy       float16 [17, H/4, W/4]  (written later by 03_precompute_pose)
      splits/{train,val,test}.txt
      meta.csv

Only numpy / Pillow / scipy are imported here, so the manifest layer stays
usable on a machine that has not installed torch, ultralytics or cv2.
"""

from __future__ import annotations

import csv
import shutil
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

SPLITS = ("train", "val", "test")

#: Ground truth in ACD1K is JPEG-compressed and CAMO's is partly anti-aliased,
#: so both arrive with hundreds of gray levels instead of two.
MASK_THRESHOLD = 127

#: Half-width of the edge band drawn around each mask boundary, in pixels.
EDGE_WIDTH = 4


# --------------------------------------------------------------------------
# mask utilities
# --------------------------------------------------------------------------


def binarize(mask: np.ndarray, threshold: int = MASK_THRESHOLD) -> np.ndarray:
    """Collapse a possibly-lossy grayscale mask to strict ``{0, 255}`` uint8."""
    return ((np.asarray(mask) > threshold).astype(np.uint8)) * 255


def ambiguity(mask: np.ndarray, lo: int = 40, hi: int = 215) -> float:
    """Fraction of pixels that are neither clearly foreground nor background.

    High values mean thresholding is throwing information away. Measured at
    0.0011 mean / 0.0047 max on ACD1K's JPEG ground truth, i.e. negligible.
    """
    arr = np.asarray(mask)
    return float(((arr > lo) & (arr < hi)).mean())


def mask_to_edge(mask_bin: np.ndarray, width: int = EDGE_WIDTH) -> np.ndarray:
    """Boundary band of a binary mask, as strict ``{0, 255}`` uint8."""
    solid = np.asarray(mask_bin) > 0
    if not solid.any():
        return np.zeros(solid.shape, dtype=np.uint8)
    inner = ndimage.binary_erosion(solid, border_value=0)
    edge = solid & ~inner
    iterations = max(0, width // 2 - 1)
    if iterations:
        edge = ndimage.binary_dilation(edge, iterations=iterations, border_value=0)
    return edge.astype(np.uint8) * 255


def count_components(mask_bin: np.ndarray) -> int:
    """Number of 8-connected foreground blobs."""
    _, n = ndimage.label(np.asarray(mask_bin) > 0, structure=np.ones((3, 3)))
    return int(n)


def foreground_fraction(mask_bin: np.ndarray) -> float:
    return float((np.asarray(mask_bin) > 0).mean())


def mask_bbox(mask_bin: np.ndarray) -> tuple[int, int, int, int] | None:
    """Tight ``(x1, y1, x2, y2)`` around the foreground, or ``None`` if empty."""
    solid = np.asarray(mask_bin) > 0
    if not solid.any():
        return None
    rows = np.where(solid.any(axis=1))[0]
    cols = np.where(solid.any(axis=0))[0]
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


# --------------------------------------------------------------------------
# io helpers
# --------------------------------------------------------------------------


def load_gray(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("L"))


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def image_size(path: str | Path) -> tuple[int, int]:
    """``(height, width)`` read from the header, without decoding pixels."""
    with Image.open(path) as im:
        w, h = im.size
    return h, w


def save_png(path: str | Path, arr: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(path, optimize=True)


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------


@dataclass
class Record:
    stem: str
    split: str
    h: int
    w: int
    fg_frac: float
    n_components: int
    source: str
    is_negative: int = 0
    extra: str = ""


class DatasetWriter:
    """Accumulates samples and writes the canonical layout on ``finalize()``.

    Idempotent: ``has(stem)`` lets a caller skip work already on disk so an
    interrupted preprocessing pass can be resumed without ``--overwrite``.
    """

    def __init__(self, root: str | Path, name: str, *, overwrite: bool = False) -> None:
        self.root = Path(root)
        self.name = name
        self.overwrite = overwrite
        self.records: list[Record] = []
        for sub in ("images", "masks", "edges", "splits"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    # -- paths ------------------------------------------------------------
    def image_path(self, stem: str) -> Path:
        return self.root / "images" / f"{stem}.jpg"

    def mask_path(self, stem: str) -> Path:
        return self.root / "masks" / f"{stem}.png"

    def edge_path(self, stem: str) -> Path:
        return self.root / "edges" / f"{stem}.png"

    def has(self, stem: str) -> bool:
        return self.image_path(stem).exists() and self.mask_path(stem).exists()

    # -- writing ----------------------------------------------------------
    def add(
        self,
        *,
        stem: str,
        split: str,
        image_src: str | Path,
        mask: np.ndarray,
        source: str,
        extra: str = "",
        is_negative: bool = False,
    ) -> Record:
        """Register one sample, copying the image and writing mask + edge.

        ``mask`` must already be binarized; an all-zero mask marks a
        presence-gate negative (no edge is written for it).
        """
        if split not in SPLITS:
            raise ValueError(f"{stem}: unknown split {split!r}, expected one of {SPLITS}")

        mask = np.asarray(mask, dtype=np.uint8)
        levels = np.unique(mask)
        if not np.isin(levels, (0, 255)).all():
            raise ValueError(f"{stem}: mask is not binary, found levels {levels[:8]}")

        self._copy_image(image_src, self.image_path(stem))

        h, w = image_size(self.image_path(stem))
        if mask.shape != (h, w):
            raise ValueError(f"{stem}: mask {mask.shape} does not match image {(h, w)}")

        save_png(self.mask_path(stem), mask)
        save_png(self.edge_path(stem), mask_to_edge(mask))

        record = Record(
            stem=stem,
            split=split,
            h=h,
            w=w,
            fg_frac=round(foreground_fraction(mask), 6),
            n_components=count_components(mask),
            source=source,
            is_negative=int(is_negative),
            extra=extra,
        )
        self.records.append(record)
        return record

    @staticmethod
    def _copy_image(src: str | Path, dst: Path) -> None:
        """Copy JPEG bytes verbatim; re-encode anything else at quality 95.

        Verbatim copying matters: re-encoding an already-lossy JPEG would add a
        second generation of compression artefacts to exactly the
        high-frequency texture the frequency-decomposition module reads.
        """
        src = Path(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix.lower() in {".jpg", ".jpeg"}:
            shutil.copyfile(src, dst)
        else:
            with Image.open(src) as im:
                im.convert("RGB").save(dst, quality=95, subsampling=0)

    # -- finalisation -----------------------------------------------------
    def finalize(self) -> dict:
        """Write ``splits/*.txt`` and ``meta.csv``; return a summary dict."""
        if not self.records:
            raise RuntimeError(f"{self.name}: nothing to write, zero records collected")

        by_split: dict[str, list[str]] = {s: [] for s in SPLITS}
        for record in self.records:
            by_split[record.split].append(record.stem)

        seen: dict[str, str] = {}
        for split, stems in by_split.items():
            for stem in stems:
                if stem in seen:
                    raise RuntimeError(
                        f"{self.name}: stem {stem!r} appears in both "
                        f"{seen[stem]!r} and {split!r} — splits must be disjoint"
                    )
                seen[stem] = split

        for split, stems in by_split.items():
            (self.root / "splits" / f"{split}.txt").write_text(
                "\n".join(sorted(stems)) + ("\n" if stems else "")
            )

        meta = self.root / "meta.csv"
        with meta.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=[f.name for f in fields(Record)])
            writer.writeheader()
            for record in sorted(self.records, key=lambda r: r.stem):
                writer.writerow(asdict(record))

        negatives = sum(r.is_negative for r in self.records)
        positives = [r for r in self.records if not r.is_negative]
        return {
            "name": self.name,
            "root": str(self.root),
            "total": len(self.records),
            "negatives": negatives,
            "splits": {s: len(v) for s, v in by_split.items()},
            "mean_fg_frac": (
                round(float(np.mean([r.fg_frac for r in positives])), 5) if positives else 0.0
            ),
        }


def read_split(root: str | Path, split: str) -> list[str]:
    path = Path(root) / "splits" / f"{split}.txt"
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]
