"""Planar pose of a known box-shaped object from its silhouette in the hand camera.

Hypotheses (x, y, yaw) of a box resting on a horizontal plane are projected with
the exact camera model; the silhouette (union of the six projected faces) is
rasterised over a region of interest and scored by IoU with the observed mask.
A coarse-to-fine grid search keeps it numpy-only and well under a second.
"""
from __future__ import annotations

import numpy as np

from .. import camera

MAX_PIXELS = 120_000  # per search stage; in-spec regions of interest are < 40k pixels

# Corner indices of the six faces of a box, each listed as a cyclic quad.
_FACES = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
_SIGNS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)


def _box_corners(hyp: np.ndarray, half: np.ndarray, z_center: float) -> np.ndarray:
    """(H, 3) hypotheses (x, y, yaw) -> (H, 8, 3) world corners."""
    local = _SIGNS * half  # (8, 3)
    c, s = np.cos(hyp[:, 2]), np.sin(hyp[:, 2])
    x = c[:, None] * local[None, :, 0] - s[:, None] * local[None, :, 1] + hyp[:, 0:1]
    y = s[:, None] * local[None, :, 0] + c[:, None] * local[None, :, 1] + hyp[:, 1:2]
    z = np.broadcast_to(local[None, :, 2] + z_center, x.shape)
    return np.stack([x, y, z], axis=-1)


def _silhouettes(uv: np.ndarray, px: np.ndarray) -> np.ndarray:
    """(H, 8, 2) projected corners, (P, 2) pixel centres -> (H, P) inside-silhouette mask.

    A pixel is inside if it lies inside any of the six projected faces (edges inclusive).
    float32 with preallocated temporaries: the cost is memory-bound, H x P elements per edge.
    """
    uv = uv.astype(np.float32)
    pxu = px[None, :, 0].astype(np.float32)
    pxv = px[None, :, 1].astype(np.float32)
    shape = (uv.shape[0], px.shape[0])
    inside_any = np.zeros(shape, dtype=bool)
    cross = np.empty(shape, dtype=np.float32)
    tmp = np.empty(shape, dtype=np.float32)
    pos = np.empty(shape, dtype=bool)
    neg = np.empty(shape, dtype=bool)
    test = np.empty(shape, dtype=bool)
    for face in _FACES:
        quad = uv[:, face, :]  # (H, 4, 2)
        pos.fill(True)
        neg.fill(True)
        for k in range(4):
            a, b = quad[:, k, :], quad[:, (k + 1) % 4, :]
            np.subtract(pxv, a[:, None, 1], out=cross)
            cross *= (b[:, 0] - a[:, 0])[:, None]
            np.subtract(pxu, a[:, None, 0], out=tmp)
            tmp *= (b[:, 1] - a[:, 1])[:, None]
            cross -= tmp
            np.greater_equal(cross, 0, out=test)
            pos &= test
            np.less_equal(cross, 0, out=test)
            neg &= test
        inside_any |= pos
        inside_any |= neg
    return inside_any


def fit_box_pose(pix_rc: np.ndarray, K: np.ndarray, c2w: np.ndarray, half_extents, z_center: float,
                 yaw_period: float = np.pi / 2, xy_init=None, margin_px: int = 8,
                 image_shape=(512, 512)) -> dict:
    """Fit (x, y, yaw) of a box whose observed silhouette pixels are ``pix_rc`` (row, col).

    ``yaw_period`` is the box's rotational symmetry (pi/2 for a square footprint).
    Returns dict(x, y, yaw, iou).
    """
    half = np.asarray(half_extents, dtype=float)
    r0, c0 = pix_rc.min(axis=0) - margin_px
    r1, c1 = pix_rc.max(axis=0) + margin_px
    r0, c0 = max(int(r0), 0), max(int(c0), 0)
    r1, c1 = min(int(r1), image_shape[0] - 1), min(int(c1), image_shape[1] - 1)
    full = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    full[pix_rc[:, 0] - r0, pix_rc[:, 1] - c0] = True

    def grid(stride):
        # Pixel centres (u, v) and the observed mask, subsampled for the coarse stages.
        rr, cc = np.mgrid[r0:r1 + 1:stride, c0:c1 + 1:stride]
        px = np.stack([cc.ravel() + 0.5, rr.ravel() + 0.5], axis=1)
        return px, full[::stride, ::stride].ravel()

    if xy_init is None:
        uv_c = np.array([[pix_rc[:, 1].mean() + 0.5, pix_rc[:, 0].mean() + 0.5]])
        xy_init = camera.backproject_to_plane(K, c2w, uv_c, z_center)[0, :2]
    best = (float(xy_init[0]), float(xy_init[1]), 0.0)

    def search(cx, cy, cyaw, dxy, nxy, dyaw, nyaw, full_yaw=False, stride=1):
        px, obs = grid(stride)
        offs = (np.arange(nxy) - (nxy - 1) / 2) * dxy
        if full_yaw:
            yaws = np.arange(nyaw) * (yaw_period / nyaw)
        else:
            yaws = cyaw + (np.arange(nyaw) - (nyaw - 1) / 2) * dyaw
        X, Y, A = np.meshgrid(cx + offs, cy + offs, yaws, indexing="ij")
        hyp = np.stack([X.ravel(), Y.ravel(), A.ravel()], axis=1)
        corners = _box_corners(hyp, half, z_center)
        uv, _ = camera.project(K, c2w, corners.reshape(-1, 3))
        sil = _silhouettes(uv.reshape(-1, 8, 2), px)
        inter = (sil & obs[None, :]).sum(axis=1)
        union = (sil | obs[None, :]).sum(axis=1)
        iou = inter / np.maximum(union, 1)
        i = int(np.argmax(iou))
        return (hyp[i, 0], hyp[i, 1], hyp[i, 2]), float(iou[i])

    n_yaw = int(round(yaw_period / np.radians(5.0)))
    # Bound the cost on huge (false-positive) regions: no stage looks at more than MAX_PIXELS
    # pixels. In-spec regions are far smaller, so their strides are unchanged.
    floor = max(1, int(np.ceil(np.sqrt(full.size / MAX_PIXELS))))
    stride = max(floor, int(np.sqrt(full.size / 2500)))  # ~2500 pixels in the coarse stage
    best, iou = search(*best, dxy=0.004, nxy=7, dyaw=0.0, nyaw=n_yaw, full_yaw=True, stride=stride)
    best, iou = search(*best, dxy=0.001, nxy=7, dyaw=np.radians(1.0), nyaw=9, stride=max(floor, stride // 2))
    best, iou = search(*best, dxy=0.00025, nxy=5, dyaw=np.radians(0.25), nyaw=7, stride=floor)
    yaw = float(np.mod(best[2], yaw_period))
    return {"x": float(best[0]), "y": float(best[1]), "yaw": yaw, "iou": iou}
