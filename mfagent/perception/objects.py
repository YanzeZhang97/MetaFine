"""T1 bottle (3558) and T2 mug (8848) localisation from the hand camera.

Both objects stand upright on the table, so a view from straight above makes
their silhouettes symmetric about the vertical axis; the filled silhouette's
centroid then back-projects onto that axis at any height. We iterate: look,
re-centre the camera over the estimate, look again.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .. import camera
from . import color, rules
from .selfmask import self_mask

# T1 bottle: body link_1 spans z 0..0.238, cap link_0 0.238..0.257 (root at z 0.1072).
BOTTLE_ROOT_Z = 0.1072
BOTTLE_REF_Z = 0.20
# T2 mug: root at z 0.0674, rim/lid top ~0.123; handle grasp point at local (-0.0626, 0.0132).
MUG_ROOT_Z = 0.0674
MUG_LID_Z = 0.123
MUG_HANDLE_Z = 0.105
MUG_HANDLE_ANGLE = float(np.arctan2(0.0132, -0.0626))  # handle direction in the mug frame
MUG_LID_OFFSET = np.array([0.005, 0.001])  # lid (link_0) origin in the mug root frame


def _blob_xy(mask, img_shape, q, z, near_uv=None, min_area=20, convex=False):
    comps = color.components(mask, min_area=min_area)
    if not comps:
        return None, None
    if near_uv is not None:
        comps = sorted(comps, key=lambda p: np.hypot(p[:, 1].mean() - near_uv[0], p[:, 0].mean() - near_uv[1]))
    pix = comps[0]
    if convex:
        filled = color.convex_fill(pix, img_shape)
    else:
        filled = np.zeros(img_shape[:2], dtype=bool)
        filled[pix[:, 0], pix[:, 1]] = True
        filled = ndimage.binary_fill_holes(filled)
    rc = np.argwhere(filled)
    uv = np.array([[rc[:, 1].mean() + 0.5, rc[:, 0].mean() + 0.5]])
    h, w = img_shape[:2]
    xy = camera.backproject_to_plane(camera.intrinsics(w, h), camera.cam2world(q), uv, z)[0, :2]
    return xy, {"area": int(len(rc)), "uv": uv[0]}


def find_bottle(img, q) -> dict | None:
    F = rules.features(img)
    m = rules.bottle_body(F) & ~self_mask(q, img.shape[:2])
    m = ndimage.binary_closing(m, iterations=2)
    xy, info = _blob_xy(m, img.shape, q, BOTTLE_REF_Z, convex=True)
    if xy is None:
        return None
    return {"x": float(xy[0]), "y": float(xy[1]), **info}


def find_mug(img, q) -> dict | None:
    F = rules.features(img)
    sm = self_mask(q, img.shape[:2])
    handle = rules.mug_handle(F) & ~sm
    handle_xy, hinfo = _blob_xy(handle, img.shape, q, MUG_HANDLE_Z, min_area=15)
    # The dark-blue handle also passes the lid rule: remove it, then take the largest blob.
    lid = rules.mug_lid(F) & ~sm & ~ndimage.binary_dilation(handle, iterations=3)
    lid = ndimage.binary_opening(lid, iterations=1)
    lid_xy, linfo = _blob_xy(lid, img.shape, q, MUG_LID_Z, min_area=150)
    if lid_xy is None:
        return None
    out = {"x": float(lid_xy[0]), "y": float(lid_xy[1]), "yaw": None, "area": linfo["area"]}
    if handle_xy is not None:
        d = handle_xy - lid_xy
        yaw = float(np.angle(np.exp(1j * (np.arctan2(d[1], d[0]) - MUG_HANDLE_ANGLE))))
        # The lid (link_0) sits 5 mm off the mug root along the mug's x axis.
        c, s = np.cos(yaw), np.sin(yaw)
        root = lid_xy - np.array([[c, -s], [s, c]]) @ MUG_LID_OFFSET
        out.update(x=float(root[0]), y=float(root[1]), yaw=yaw, handle_dist=float(np.hypot(*d)))
    return out


# Lying bottle (knocked over): body radius 0.03 m, so its axis is ~0.03 m above the table and
# the visible upper surface is ~0.045 m; the magenta body ends at the neck, where the cap begins.
LYING_SURFACE_Z = 0.045
BOTTOM_TO_CAP = 0.247
BOTTOM_TO_BODY = 0.119


def bottle_shape(img, q) -> dict | None:
    """From a view straight above the bottle: is it lying down, and if so where are its
    axis, body centre and cap? Uses the magenta body pixels back-projected to the table."""
    F = rules.features(img)
    m = ndimage.binary_closing(rules.bottle_body(F) & ~self_mask(q, img.shape[:2]), iterations=2)
    comps = color.components(m, min_area=40)
    if not comps:
        return None
    h, w = img.shape[:2]
    K, c2w = camera.intrinsics(w, h), camera.cam2world(q)

    def to_table(pix):
        uv = np.stack([pix[:, 1] + 0.5, pix[:, 0] + 0.5], axis=1)
        return camera.backproject_to_plane(K, c2w, uv, LYING_SURFACE_Z)[:, :2]

    # The bottle's label can cut the magenta body into pieces: merge every blob near the largest.
    pts = to_table(comps[0])
    c0 = pts.mean(axis=0)
    for pix in comps[1:]:
        p = to_table(pix)
        if np.linalg.norm(p.mean(axis=0) - c0) < 0.25:
            pts = np.concatenate([pts, p])
    c = pts.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov((pts - c).T))
    axis = evecs[:, 1]
    elong = float(np.sqrt(evals[1] / max(evals[0], 1e-12)))
    s = (pts - c) @ axis
    lo, hi = s.min(), s.max()
    out = {"elongation": elong, "lying": bool(elong > 2.2 and hi - lo > 0.12)}
    if out["lying"]:
        perp = (pts - c) @ np.array([-axis[1], axis[0]])
        band = 0.025
        w_lo = np.ptp(perp[s < lo + band]) if (s < lo + band).sum() > 5 else 1.0
        w_hi = np.ptp(perp[s > hi - band]) if (s > hi - band).sum() > 5 else 1.0
        sign = 1.0 if w_hi < w_lo else -1.0   # the neck end is the narrower one
        # The thin neck is under-detected, so measure from the flat bottom instead:
        # bottom at local z -0.107, body centre +0.012, cap centre +0.140 (along the axis).
        bottom = c + axis * (lo if sign > 0 else hi)
        u_cap = axis * sign
        out["axis"] = u_cap.tolist()                                 # unit vector pointing to the cap
        out["cap_xy"] = (bottom + u_cap * BOTTOM_TO_CAP).tolist()
        out["body_xy"] = (bottom + u_cap * BOTTOM_TO_BODY).tolist()
    return out
