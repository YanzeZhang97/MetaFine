"""Colour cues that survive the eval's per-channel ambient-light casts.

Gray-world normalisation divides out the global cast (the table fills most of
the hand-camera view), then chromaticity separates saturated primaries from the
wood table and the tan boxes.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

PRIMARIES = ("red", "green", "blue")
_CHANNEL = {"red": 0, "green": 1, "blue": 2}


def gray_world(img: np.ndarray) -> np.ndarray:
    f = np.asarray(img, dtype=np.float32)
    means = f.reshape(-1, 3).mean(axis=0) + 1e-6
    return f * (means.mean() / means)


def chromaticity(f: np.ndarray) -> np.ndarray:
    return f / (f.sum(axis=-1, keepdims=True) + 1e-6)


def primary_masks(img: np.ndarray, min_chroma: float = 0.55, min_margin: float = 0.5,
                  min_brightness: float = 30.0) -> dict[str, np.ndarray]:
    """Boolean masks of strongly red / green / blue pixels after gray-world correction."""
    f = gray_world(img)
    c = chromaticity(f)
    bright = f.sum(axis=-1) > min_brightness
    out = {}
    for name in PRIMARIES:
        k = _CHANNEL[name]
        others = np.delete(c, k, axis=-1).max(axis=-1)
        out[name] = bright & (c[..., k] > min_chroma) & (c[..., k] - others > min_margin)
    return out


def components(mask: np.ndarray, min_area: int = 100) -> list[np.ndarray]:
    """Connected components (8-connectivity) as (N, 2) arrays of (row, col), largest first.

    ``min_area`` is in pixels of the 512x512 hand camera. Other image sizes scale it by
    (height / 512)^2: the focal length is proportional to the image height (fixed vertical FOV),
    so that keeps the same physical size."""
    scale = (mask.shape[0] / 512.0) ** 2
    if scale != 1.0:
        min_area = max(1, int(round(min_area * scale)))
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    if n == 0:
        return []
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    order = np.argsort(-sizes)
    out = []
    for i in order:
        if sizes[i] < min_area:
            break
        out.append(np.argwhere(lab == i + 1))
    return out


def convex_fill(pix_rc: np.ndarray, shape) -> np.ndarray:
    """Filled convex hull of a set of (row, col) pixels: robust to gaps in a ring-shaped blob."""
    from scipy.spatial import ConvexHull
    out = np.zeros(shape[:2], dtype=bool)
    out[pix_rc[:, 0], pix_rc[:, 1]] = True
    try:
        hull = ConvexHull(pix_rc[:, ::-1].astype(float) + 0.5)
    except Exception:  # degenerate blob (QhullError's import path differs across scipy versions)
        return out
    r0, c0 = pix_rc.min(axis=0)
    r1, c1 = pix_rc.max(axis=0)
    rr, cc = np.mgrid[r0:r1 + 1, c0:c1 + 1]
    pts = np.stack([cc.ravel() + 0.5, rr.ravel() + 0.5], axis=1)
    inside = np.all(pts @ hull.equations[:, :2].T + hull.equations[:, 2] <= 1e-6, axis=1)
    out[rr.ravel()[inside], cc.ravel()[inside]] = True
    return out
