"""T5: find a letter peg by colour and fit its planar pose by template matching.

Mask pixels are back-projected onto the peg's top plane (z = 36 mm) and matched
against the glyph (union of stroke rectangles) with an FFT cross-correlation
per candidate yaw, which gives the best translation for each yaw in one shot.
From a view straight above the peg the side faces add only a thin inner band.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .. import camera, letters
from . import color, pixelmlp
from .selfmask import self_mask


# Clean-light appearance of horizontal surfaces (hand camera, ambient 0.3): the table,
# the board and the four pegs' top faces.
TABLE_RGB = np.array([176.0, 104.0, 64.0])
REF_RGB = {"C": (198.0, 96.0, 96.0), "o": (94.0, 135.0, 195.0), "R": (190.0, 174.0, 83.0),
           "L": (106.0, 195.0, 128.0), "table": tuple(TABLE_RGB), "board": (99.0, 99.0, 104.0)}
_CLASSES = list(REF_RGB)
_REF_CHROMA = np.array([np.asarray(REF_RGB[k]) / sum(REF_RGB[k]) for k in _CLASSES])


def white_balance(img: np.ndarray) -> np.ndarray:
    """Undo the per-channel ambient cast with the table as the reference surface.

    The table fills most of every hand-camera view, so the median colour is the
    table's; table and peg tops are both horizontal and lit alike, so scaling
    each channel by (clean table / observed table) restores their clean colours.
    """
    f = np.asarray(img, dtype=np.float32)
    med = np.median(f.reshape(-1, 3), axis=0) + 1e-3
    return f * (TABLE_RGB / med)


def classify(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel nearest reference class (index into _CLASSES) and chroma distance."""
    f = white_balance(img)
    c = f / (f.sum(axis=-1, keepdims=True) + 1e-3)
    d = np.linalg.norm(c[..., None, :] - _REF_CHROMA[None, None], axis=-1)
    return np.argmin(d, axis=-1), np.min(d, axis=-1)


def letter_mask(img, q, letter: str, labels=None) -> np.ndarray:
    """Pixels of the given letter peg, from the per-pixel MLP (robust to the ambient casts)."""
    lab = pixelmlp.classify(img) if labels is None else labels
    m = ndimage.binary_opening(lab == pixelmlp.CLASSES.index(letter), iterations=1)
    return m & ~self_mask(q, img.shape[:2])


def _raster_glyph(letter: str, yaw: float, n: int, res: float) -> np.ndarray:
    """Glyph rotated by yaw, origin at the grid centre, as an (n, n) occupancy grid."""
    ax = (np.arange(n) - n // 2) * res
    X, Y = np.meshgrid(ax, ax, indexing="xy")  # X: columns (world x), Y: rows (world y)
    c, s = np.cos(yaw), np.sin(yaw)
    xl, yl = c * X + s * Y, -s * X + c * Y
    occ = np.zeros((n, n), dtype=bool)
    for x0, y0, x1, y1 in letters.strokes_m(letter):
        occ |= (xl >= x0) & (xl <= x1) & (yl >= y0) & (yl <= y1)
    return occ


def _match(points: np.ndarray, letter: str, yaws, res: float, n: int):
    """Best (x, y, yaw, iou) of the glyph origin for the given world points on the top plane."""
    centre = np.round(points.mean(axis=0) / res) * res
    ij = np.round((points - centre) / res).astype(int) + n // 2
    ok = (ij >= 0).all(axis=1) & (ij < n).all(axis=1)
    obs = np.zeros((n, n), dtype=bool)
    obs[ij[ok, 1], ij[ok, 0]] = True
    # Close pixel-sampling gaps (R's and o's counters are much larger than this).
    obs = ndimage.binary_closing(obs, iterations=max(1, int(round(0.0015 / res))))
    F_obs = np.fft.rfft2(obs.astype(np.float32))
    n_obs = obs.sum()
    best = (-1.0, 0.0, 0.0, 0.0)
    for yaw in yaws:
        tpl = _raster_glyph(letter, yaw, n, res)
        corr = np.fft.irfft2(F_obs * np.conj(np.fft.rfft2(tpl.astype(np.float32))), s=(n, n))
        iou = corr / np.maximum(n_obs + tpl.sum() - corr, 1.0)
        k = int(np.argmax(iou))
        if iou.flat[k] > best[0]:
            dr, dc = np.unravel_index(k, iou.shape)
            dr = dr - n if dr > n // 2 else dr
            dc = dc - n if dc > n // 2 else dc
            best = (float(iou.flat[k]), centre[0] + dc * res, centre[1] + dr * res, float(yaw))
    return best


# Blobs are back-projected onto the peg-top plane (z = PEG_H). In that plane the board
# (y 0.10..0.18, fixed by the env code) and its front face land beyond this line, while the
# pegs spawn at y <= -0.10. A strong colour cast can make board surfaces look peg-coloured.
BOARD_ZONE_Y = 0.06


def fit_letter(img, q, letter: str, min_area: int = 80, yaw_prior=None, labels=None,
               near_xy=None, near_tol: float = 0.03) -> dict | None:
    """Planar pose (x, y, yaw) of the glyph origin of ``letter``, fitted to the largest blob of
    its colour that is not on the board. With ``near_xy`` (an earlier estimate of the same peg),
    a fit farther than ``near_tol`` from it is replaced by the fit of the nearest blob that lands
    within ``near_tol``, or None: under a strong cast another surface can outgrow the peg's blob,
    while pegs are >= 8 cm apart."""
    comps = color.components(letter_mask(img, q, letter, labels), min_area=min_area)
    h, w = img.shape[:2]
    K, c2w = camera.intrinsics(w, h), camera.cam2world(q)

    def plane_points(pix):
        uv = np.stack([pix[:, 1] + 0.5, pix[:, 0] + 0.5], axis=1)
        return camera.backproject_to_plane(K, c2w, uv, letters.PEG_H)[:, :2]

    blobs = [pts for pts in map(plane_points, comps) if pts[:, 1].mean() <= BOARD_ZONE_Y]
    if not blobs:
        return None
    est = _fit_points(blobs[0], letter, yaw_prior, len(blobs[0]))
    if near_xy is None:
        return est
    near_xy = np.asarray(near_xy, dtype=float)
    if np.hypot(est["x"] - near_xy[0], est["y"] - near_xy[1]) <= near_tol:
        return est
    for pts in sorted(blobs[1:], key=lambda p: np.hypot(*(p.mean(axis=0) - near_xy))):
        alt = _fit_points(pts, letter, yaw_prior, len(pts))
        if np.hypot(alt["x"] - near_xy[0], alt["y"] - near_xy[1]) <= near_tol:
            return alt
    return None


def _fit_points(pts, letter: str, yaw_prior, area: int) -> dict:
    period = letters.SYMMETRY[letter]
    coarse = np.arange(0.0, period, np.radians(4.0)) if yaw_prior is None else \
        yaw_prior + np.radians(np.arange(-40.0, 40.1, 4.0))
    iou, x, y, yaw = _match(pts, letter, coarse, res=0.001, n=128)
    fine = yaw + np.radians(np.arange(-3.0, 3.01, 0.5))
    iou, x, y, yaw = _match(pts, letter, fine, res=0.0005, n=256)
    yaw = float(np.angle(np.exp(1j * yaw)))
    if letter == "o":
        # The o is a square ring: its outline repeats every 90 degrees about the ring
        # centre, which is not the glyph origin. The eval spawns pegs within +-30 deg
        # (and scores o modulo 180), so pick the quarter turn closest to yaw 0 / 180.
        c = np.array([x, y]) + _rot(yaw) @ O_CENTRE
        k = int(np.round(-yaw / (np.pi / 2)))
        yaw = float(np.angle(np.exp(1j * (yaw + k * np.pi / 2))))
        x, y = c - _rot(yaw) @ O_CENTRE
    if period < 2 * np.pi - 1e-6:
        yaw = float(np.mod(yaw + period / 2, period) - period / 2)
    return {"x": float(x), "y": float(y), "yaw": yaw, "iou": iou, "area": int(area)}


O_CENTRE = np.array([0.0, -0.010])  # centre of the o ring in its glyph frame


def _rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def fit_all(img, q) -> dict:
    """Coarse poses of every letter peg visible in ``img`` (one classifier pass)."""
    lab = pixelmlp.classify(img)
    out = {}
    for L in ("C", "o", "R", "L"):
        est = fit_letter(img, q, L, labels=lab)
        if est is not None:
            out[L] = est
    return out
