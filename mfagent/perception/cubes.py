"""T4: find the red / green / blue cubes in a hand-camera image and fit their planar poses."""
from __future__ import annotations

import numpy as np

from .. import camera
from . import color, fit

CUBE_HALF = 0.02


def detect_cubes(img: np.ndarray, q, min_area: int = 150, min_iou: float = 0.5,
                 max_candidates: int = 3) -> dict[str, dict]:
    """Return {colour: dict(x, y, yaw, iou, area)} for every cube found in ``img``."""
    h, w = img.shape[:2]
    K = camera.intrinsics(w, h)
    c2w = camera.cam2world(q)
    out = {}
    for name, mask in color.primary_masks(img).items():
        best = None
        # A strong cast can leave background blobs in the mask: fit the few largest
        # components and keep the one that looks most like a cube.
        for pix in color.components(mask, min_area=min_area)[:max_candidates]:
            res = fit.fit_box_pose(pix, K, c2w, (CUBE_HALF,) * 3, z_center=CUBE_HALF, image_shape=(h, w))
            res["area"] = int(len(pix))
            if best is None or res["iou"] > best["iou"]:
                best = res
        if best is not None and best["iou"] >= min_iou:
            out[name] = best
    return out
