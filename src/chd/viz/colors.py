"""Shared palette for the dataset/architecture visualisations.

Fixed categorical order (Okabe-Ito, colorblind-safe) — one color per dataset,
assigned by identity and never re-cycled or re-ranked. Validated with the
dataviz skill's palette checker (light mode, the target here since these are
static PNG/SVG reports on a fixed light background, not a theme-adaptive
artifact): all four categorical checks pass. The CVD-separation and
surface-contrast checks land in the legal-with-secondary-encoding band, which
every chart here already satisfies via direct labels/legends, never color
alone.

"combined" is deliberately NOT a fifth categorical peer — it's the union of
the other four, so it gets a neutral gray plus a hatch texture rather than a
hue, which also keeps it from crowding the validated 4-color set.
"""

from __future__ import annotations

DATASET_ORDER = ("acd1k", "cpd1k", "camo_human", "mhcd", "combined")

DATASET_COLOR = {
    "acd1k": "#0072B2",       # blue
    "cpd1k": "#E69F00",       # orange
    "camo_human": "#009E73",  # green
    "mhcd": "#CC79A7",        # reddish purple
    "combined": "#595959",    # neutral gray, paired with a hatch — not a hue identity
}

COMBINED_HATCH = "///"

DATASET_LABEL = {
    "acd1k": "ACD1K",
    "cpd1k": "CPD1K",
    "camo_human": "CAMO-Human",
    "mhcd": "MHCD",
    "combined": "Combined",
}

SPLIT_COLOR = {"train": "#0072B2", "val": "#E69F00", "test": "#009E73"}

HUMAN_PAIR = ("#0072B2", "#B0B0B0")  # human, non-human/excluded

SEQUENTIAL = "Blues"

INK = {"primary": "#1a1a1a", "secondary": "#595959", "muted": "#8c8c8c", "grid": "#e0e0e0"}

#: False-positive / false-negative / true-positive colors for the error panel
#: in the qualitative figure. FP is Okabe-Ito vermillion, FN is the same blue
#: already used for acd1k/train, TP is a neutral gray so correct pixels never
#: compete for attention with mistakes.
ERROR_COLOR = {"fp": "#D55E00", "fn": "#0072B2", "tp": "#B0B0B0"}
