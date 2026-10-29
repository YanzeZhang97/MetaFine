"""Hand-camera pixels covered by the robot's own hand and fingers.

The hand and fingers are rigid with respect to the wrist camera, so their image
footprint depends only on the finger opening (rendered by dev/make_self_mask.py).
A task YAML's sensor size applies to the hand camera too. 512x512 images use the
512x512 render as is; other sizes get the 1024x512 render (same vertical FOV, so it
covers images up to 2:1 wide) mapped through the pinhole model: the vertical FOV
stays pi/2 and pixels stay square (``camera.intrinsics``).
"""
from __future__ import annotations

import os

import numpy as np

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "hand_self_mask.npz")
_PATH_WIDE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "hand_self_mask_wide.npz")
_CACHE: dict = {}
_RESAMPLED: dict = {}


def _unpack(path):
    d = np.load(path)
    shape = tuple(int(v) for v in d["shape"])
    return np.unpackbits(d["masks"], axis=-1)[..., :shape[-1]].astype(bool).reshape(shape), d["widths"]


def _load():
    if "masks" not in _CACHE:
        _CACHE["masks"], _CACHE["widths"] = _unpack(_PATH)
    return _CACHE["masks"], _CACHE["widths"]


def _load_wide():
    """The 1024x512 renders (same openings as the 512x512 ones), or None if unavailable."""
    if "wide" not in _CACHE:
        try:
            wide, widths = _unpack(_PATH_WIDE)
            _CACHE["wide"] = wide if np.array_equal(widths, _load()[1]) else None
        except Exception:
            _CACHE["wide"] = None
    return _CACHE["wide"]


def _resample(src: np.ndarray, shape) -> np.ndarray:
    """Render (Hs x Ws) -> H x W image with the same vertical FOV and centred principal points.
    Every output pixel takes the source pixel on its ray, with out-of-view columns repeating
    the edge (the hand continues past it); every source pixel also marks the output pixel it
    falls in, so thin fingers survive downsampling."""
    Hs, Ws = src.shape
    H, W = shape
    k = Hs / H  # source pixels per output pixel
    rows = np.clip(np.floor(Hs / 2 + (np.arange(H) + 0.5 - H / 2) * k).astype(int), 0, Hs - 1)
    cols = np.clip(np.floor(Ws / 2 + (np.arange(W) + 0.5 - W / 2) * k).astype(int), 0, Ws - 1)
    out = src[np.ix_(rows, cols)]
    r, c = np.nonzero(src)
    rd = np.floor((r + 0.5 - Hs / 2) / k + H / 2).astype(int)
    cd = np.floor((c + 0.5 - Ws / 2) / k + W / 2).astype(int)
    ok = (rd >= 0) & (rd < H) & (cd >= 0) & (cd < W)
    out[rd[ok], cd[ok]] = True
    return out


def self_mask(q, shape=(512, 512)) -> np.ndarray:
    """Boolean mask of hand/finger pixels for the current finger opening (q[7])."""
    shape = tuple(int(v) for v in shape[:2])
    try:
        masks, widths = _load()
        width = float(q[7]) if len(q) > 7 else 0.04  # finger joint position (one finger)
        i = int(np.argmin(np.abs(widths - width)))
        m = masks[i]
        if m.shape == shape:
            return m
        if (i, shape) not in _RESAMPLED:
            if len(_RESAMPLED) > 64:
                _RESAMPLED.clear()
            wide = _load_wide()
            _RESAMPLED[(i, shape)] = _resample(m if wide is None else wide[i], shape)
        return _RESAMPLED[(i, shape)]
    except Exception:
        return np.zeros(shape, dtype=bool)
