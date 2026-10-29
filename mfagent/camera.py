"""Hand-camera geometry: intrinsics, pose from joint angles, projection and
back-projection onto horizontal planes.

The hand camera is rigidly mounted on the Panda hand, so its pose follows from
forward kinematics exactly; the eval's camera jitter only moves the base camera.
"""
from __future__ import annotations

import numpy as np

from . import robot

# camera_link (x forward, y left, z up) -> OpenCV camera (x right, y down, z forward)
_LINK_TO_CV = np.array([[0.0, 0.0, 1.0, 0.0],
                        [-1.0, 0.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]])

HAND_FOV = np.pi / 2


def intrinsics(width: int = 512, height: int = 512, fov: float = HAND_FOV) -> np.ndarray:
    f = 0.5 * height / np.tan(0.5 * fov)
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])


def cam2world(q) -> np.ndarray:
    """World pose of the hand camera in the OpenCV convention."""
    return robot.camera_link_pose(q) @ _LINK_TO_CV


def project(K, c2w, pts) -> tuple[np.ndarray, np.ndarray]:
    """World points (N, 3) -> pixels (N, 2) and depths (N,)."""
    w2c = np.linalg.inv(c2w)
    P = np.asarray(pts, dtype=float) @ w2c[:3, :3].T + w2c[:3, 3]
    uv = P @ np.asarray(K).T
    return uv[:, :2] / uv[:, 2:3], P[:, 2]


def backproject_to_plane(K, c2w, uv, z_plane: float) -> np.ndarray:
    """Pixels (N, 2) -> intersection of their rays with the plane z = z_plane (world)."""
    uv = np.atleast_2d(np.asarray(uv, dtype=float))
    d_cam = np.linalg.solve(K, np.vstack([uv.T, np.ones(len(uv))]))
    d = c2w[:3, :3] @ d_cam
    o = c2w[:3, 3:4]
    t = (z_plane - o[2]) / d[2]
    return (o + d * t).T
