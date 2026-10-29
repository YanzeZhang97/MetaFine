"""Tiny per-pixel MLP for the T5 scene (table / board / C / o / R / L / robot).

Features per pixel: raw RGB, the image's median RGB (≈ the table under the
current cast) and the table-referenced white-balanced RGB. Trained on rendered,
labelled pixels (dev/gen_pixels.py, dev/train_pixel_mlp.py); runs in numpy.
"""
from __future__ import annotations

import os

import numpy as np

CLASSES = ["table", "board", "C", "o", "R", "L", "robot"]
TABLE_RGB = np.array([176.0, 104.0, 64.0], dtype=np.float32)
_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "t5_pixel_mlp.npz")
_W: dict = {}


def pixel_features(img: np.ndarray) -> np.ndarray:
    f = np.asarray(img, dtype=np.float32)
    med = np.median(f.reshape(-1, 3), axis=0) + 1e-3
    wb = np.clip(f * (TABLE_RGB / med) / 255.0, 0.0, 2.0)
    medb = np.broadcast_to(med / 255.0, f.shape)
    return np.concatenate([f / 255.0, medb, wb], axis=-1).astype(np.float32)


def _load():
    if not _W:
        d = np.load(_PATH)
        _W.update({k: d[k].astype(np.float32) for k in d.files})
    return _W


def classify(img: np.ndarray, stride: int = 1) -> np.ndarray:
    """(H, W) class indices; with stride > 1 the result is upsampled by repetition."""
    W = _load()
    x = pixel_features(img[::stride, ::stride])
    h, w = x.shape[:2]
    z = x.reshape(-1, x.shape[-1])
    z = (z - W["mu"]) / W["sd"]
    z = np.maximum(z @ W["w1"] + W["b1"], 0.0)
    z = np.maximum(z @ W["w2"] + W["b2"], 0.0)
    z = z @ W["w3"] + W["b3"]
    lab = np.argmax(z, axis=-1).reshape(h, w)
    if stride > 1:
        lab = np.repeat(np.repeat(lab, stride, axis=0), stride, axis=1)[:img.shape[0], :img.shape[1]]
    return lab
