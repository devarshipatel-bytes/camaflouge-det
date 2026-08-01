"""Compatibility shims for running on older Python.

The training machine seen in practice runs Python 3.9, where
``zip(*iterables, strict=True)`` isn't available (added in 3.10). Every
call site in this project uses ``strict=True`` purely as a safety net over
small, finite, already-in-memory collections (model params, per-level
feature lists, CLI arg tuples) — never a large or infinite stream — so
materialising to lists first is free and exactly equivalent.
"""

from __future__ import annotations

from typing import Iterable


def zip_strict(*iterables: Iterable):
    """Like Python 3.10+'s ``zip(*iterables, strict=True)``: raises
    ``ValueError`` if the iterables have different lengths."""
    lists = [list(it) for it in iterables]
    lengths = {len(lst) for lst in lists}
    if len(lengths) > 1:
        raise ValueError(f"zip_strict() argument lengths differ: {[len(lst) for lst in lists]}")
    return zip(*lists)
