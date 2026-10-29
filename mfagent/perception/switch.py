"""T3: the red dual-slider switch (PartNet-Mobility 100920, lying on its side).

Geometry (switch root frame, from dev/dump_t3_geometry.py): the red body spans
10.9 cm along the slider stroke and 7.1 cm across, top face at world z = 0.084;
slider tops sit at z = 0.096. The root is at z = 0.05 and the rotation is
Rz(theta) @ Ry(90 deg), so the stroke axis points along world yaw theta.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .. import camera
from . import color, fit
from .selfmask import self_mask

ROOT_Z = 0.05
BODY_TOP_Z = 0.0842
BODY_CENTER_Z = 0.0512
BODY_HALF = (0.0545, 0.0355, 0.0330)        # along stroke, across, vertical
BODY_CENTER_LOCAL = np.array([0.0003, 0.0007])  # body centre - root, (along stroke, across)
SLIDER_TOP_Z = 0.0963
SLIDER_ACROSS = (-0.0150, 0.0150)           # link_0, link_1 lateral offsets from the root
SLIDER_ALONG = -0.0101                      # slider centre along the stroke at qpos = 0
BLUE_SCORE_MIN = 0.2                        # slider >= 0.28, body pixels < 0.26 at p99 under casts
BODY_RED_MARGIN = 0.25


def body_mask(img, q) -> np.ndarray:
    """Red switch body: red margin > 0.25 after gray-world (body >= 0.39 median, table <= 0.2;
    checked with dev/eval_rules.py under the 8 ambient corners), minus the robot's own fingers."""
    c = color.chromaticity(color.gray_world(img))
    m = c[..., 0] - np.maximum(c[..., 1], c[..., 2]) > BODY_RED_MARGIN
    return m & ~self_mask(q, img.shape[:2])


def _red_body_pixels(img, q, min_area=150):
    comps = color.components(body_mask(img, q), min_area=min_area)
    return comps[0] if comps else None


def find_coarse(img, q) -> np.ndarray | None:
    """World xy of the switch from a distant view (centroid of the largest red blob)."""
    pix = _red_body_pixels(img, q)
    if pix is None:
        return None
    h, w = img.shape[:2]
    uv = np.array([[pix[:, 1].mean() + 0.5, pix[:, 0].mean() + 0.5]])
    return camera.backproject_to_plane(camera.intrinsics(w, h), camera.cam2world(q), uv, BODY_TOP_Z)[0, :2]


def _rot2(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def fit_switch(img, q) -> dict | None:
    """Root pose (x, y, theta) of the switch and which slider link is blue, from a close view."""
    h, w = img.shape[:2]
    K, c2w = camera.intrinsics(w, h), camera.cam2world(q)
    comps = color.components(body_mask(img, q), min_area=300)
    if not comps:
        return None
    body = np.zeros((h, w), dtype=bool)
    body[comps[0][:, 0], comps[0][:, 1]] = True
    body = ndimage.binary_fill_holes(body)  # white channels, label and sliders sit inside the body
    res = fit.fit_box_pose(np.argwhere(body), K, c2w, BODY_HALF, BODY_CENTER_Z,
                           yaw_period=np.pi, image_shape=(h, w))
    theta = float(np.angle(np.exp(1j * res["yaw"] * 2)) / 2)  # long axis direction in (-90, 90] deg
    centre = np.array([res["x"], res["y"]])
    root = centre - _rot2(theta) @ BODY_CENTER_LOCAL
    out = {"x": float(root[0]), "y": float(root[1]), "theta": theta, "iou": res["iou"],
           "blue_link": None, "flip_evidence": None}

    # The blue slider is light blue (about 90/140/220 RGB): too pale for the strict
    # primary-colour mask, but it is the only clearly bluish thing on the body.
    c = color.chromaticity(color.gray_world(img))
    blue_score = c[..., 2] - np.maximum(c[..., 0], c[..., 1])
    blues = color.components((blue_score > BLUE_SCORE_MIN) & ndimage.binary_dilation(body, iterations=3),
                             min_area=60)
    if blues:
        pix = blues[0]
        uv = np.array([[pix[:, 1].mean() + 0.5, pix[:, 0].mean() + 0.5]])
        p = camera.backproject_to_plane(K, c2w, uv, SLIDER_TOP_Z)[0, :2]
        along, across = _rot2(-theta) @ (p - root)
        if along > 0.005:
            # The slider sits on the far half: the body is turned the other way round.
            theta = float(np.angle(np.exp(1j * (theta + np.pi))))
            root = centre - _rot2(theta) @ BODY_CENTER_LOCAL
            along, across = _rot2(-theta) @ (p - root)
            out.update(x=float(root[0]), y=float(root[1]), theta=theta)
        out["flip_evidence"] = float(along)
        out["blue_link"] = 0 if across < 0 else 1
        out["blue_across"] = float(across)
    return out
