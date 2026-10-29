"""Hand-written colour rules for the T1 bottle and the T2 mug.

All rules work on gray-world-normalised chromaticity, which removes the
eval's per-channel ambient cast (the table fills most of the view). Thresholds
were checked against simulator segmentation under clean light and the eight
corners of the ambient cube (dev/eval_rules.py).
"""
from __future__ import annotations

import numpy as np

from . import color


def features(img: np.ndarray) -> dict:
    f = color.gray_world(img)
    c = color.chromaticity(f)
    bright = f.mean(axis=-1)
    return {
        "c": c,
        "bright_rel": bright / (np.median(bright) + 1e-6),
        "sat": (f.max(axis=-1) - f.min(axis=-1)) / (f.max(axis=-1) + 1e-6),
    }


def bottle_body(F: dict) -> np.ndarray:
    """Magenta bottle body (link_1 of 3558)."""
    c = F["c"]
    return (c[..., 2] - c[..., 1] > 0.08) & (c[..., 0] - c[..., 1] > 0.10) & (F["sat"] > 0.3)


def bottle_cap(F: dict) -> np.ndarray:
    """Beige cap (link_0): after gray-world it is less red than the wood table."""
    c = F["c"]
    return (c[..., 0] < 0.30) & (c[..., 1] > 0.32) & (c[..., 2] > 0.32) & (F["sat"] > 0.15)


def mug_lid(F: dict) -> np.ndarray:
    """Dark bluish lid / rim of the mug (8848)."""
    c = F["c"]
    return (c[..., 2] - c[..., 1] > 0.05) & (c[..., 2] > 0.40) & (F["bright_rel"] < 0.85)


def mug_handle(F: dict) -> np.ndarray:
    """Saturated blue handle."""
    c = F["c"]
    return (c[..., 2] > 0.55) & (c[..., 2] - np.maximum(c[..., 0], c[..., 1]) > 0.5)
