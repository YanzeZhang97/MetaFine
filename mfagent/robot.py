"""Panda (``panda_wristcam``) kinematics in plain numpy: forward kinematics, the
TCP Jacobian and damped-least-squares IK.

The joint constants are copied from mani_skill's ``panda_v3.urdf`` (printed by
``dev/extract_panda_chain.py``), so the policy server needs neither the URDF nor
SAPIEN. Every competition env puts the robot base at world (-0.615, 0, 0) with
identity rotation.
"""
from __future__ import annotations

import numpy as np

BASE_POS = np.array([-0.615, 0.0, 0.0])

# Origin (xyz, rpy) of panda_joint1..7; every arm joint turns about its local z axis.
_ARM_ORIGINS = [
    ((0.0, 0.0, 0.333), (0.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (-np.pi / 2, 0.0, 0.0)),
    ((0.0, -0.316, 0.0), (np.pi / 2, 0.0, 0.0)),
    ((0.0825, 0.0, 0.0), (np.pi / 2, 0.0, 0.0)),
    ((-0.0825, 0.384, 0.0), (-np.pi / 2, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (np.pi / 2, 0.0, 0.0)),
    ((0.088, 0.0, 0.0), (np.pi / 2, 0.0, 0.0)),
]
Q_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

# Initial arm pose of every competition env (before the 0.02 rad init noise).
Q_HOME = np.array([0.0, np.pi / 8, 0.0, -np.pi * 5 / 8, 0.0, np.pi * 3 / 4, np.pi / 4])


def tf(xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)) -> np.ndarray:
    """4x4 transform from a URDF origin (rotation = Rz(yaw) @ Ry(pitch) @ Rx(roll))."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    T = np.eye(4)
    T[:3, :3] = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                 [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                 [-sp, cp * sr, cp * cr]]
    T[:3, 3] = xyz
    return T


_ARM_T = [tf(xyz, rpy) for xyz, rpy in _ARM_ORIGINS]
# link7 -> panda_link8 (panda_joint8) -> panda_hand (panda_hand_joint)
_FLANGE_TO_HAND = tf((0.0, 0.0, 0.107)) @ tf(rpy=(0.0, 0.0, -np.pi / 4))
_HAND_TO_TCP = tf((0.0, 0.0, 0.1034))
# panda_hand -> camera_base_link (realsense_joint) -> camera_link; the URDF rounds pi/2, pi.
_HAND_TO_CAMERA = tf((0.035, 0.0, 0.036), (0.0, -1.5707, 3.1415)) @ tf((0.0, 0.02, 0.0115))


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    T = np.eye(4)
    T[0, 0], T[0, 1], T[1, 0], T[1, 1] = c, -s, s, c
    return T


def _arm_frames(q):
    """World frame of each joint (before its rotation) and the link7 frame."""
    T = np.eye(4)
    T[:3, 3] = BASE_POS
    frames = []
    for i in range(7):
        T = T @ _ARM_T[i]
        frames.append(T)
        T = T @ _rz(q[i])
    return frames, T


def hand_pose(q) -> np.ndarray:
    return _arm_frames(q)[1] @ _FLANGE_TO_HAND


def tcp_pose(q) -> np.ndarray:
    """World pose of ``panda_hand_tcp``: z = approach direction, y = finger closing axis."""
    return hand_pose(q) @ _HAND_TO_TCP


def camera_link_pose(q) -> np.ndarray:
    """World pose of the hand camera in SAPIEN's convention (x forward, y left, z up)."""
    return hand_pose(q) @ _HAND_TO_CAMERA


def tcp_jacobian(q):
    """Geometric 6x7 Jacobian (linear; angular, world frame) at the TCP, and the TCP pose."""
    frames, T7 = _arm_frames(q)
    T = T7 @ _FLANGE_TO_HAND @ _HAND_TO_TCP
    p = T[:3, 3]
    J = np.empty((6, 7))
    for i, F in enumerate(frames):
        # z x (p - o), written out: np.cross on 3-vectors costs more than the rest of IK
        z0, z1, z2 = F[0, 2], F[1, 2], F[2, 2]
        d0, d1, d2 = p[0] - F[0, 3], p[1] - F[1, 3], p[2] - F[2, 3]
        J[0, i] = z1 * d2 - z2 * d1
        J[1, i] = z2 * d0 - z0 * d2
        J[2, i] = z0 * d1 - z1 * d0
        J[3, i], J[4, i], J[5, i] = z0, z1, z2
    return J, T


def skew(w) -> np.ndarray:
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def rot_exp(w) -> np.ndarray:
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    K = skew(np.asarray(w) / th)
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * K @ K


def rot_log(R) -> np.ndarray:
    """Axis-angle vector of a rotation matrix."""
    ang = float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if ang < 1e-9:
        return 0.5 * v
    if np.pi - ang < 1e-6:
        A = (R + np.eye(3)) / 2.0
        k = int(np.argmax(np.diag(A)))
        axis = A[:, k] / np.sqrt(max(A[k, k], 1e-12))
        return axis * ang
    return v / (2.0 * np.sin(ang)) * ang


def ik_tcp(T_goal, q_init, max_iter: int = 300, tol_p: float = 1e-5, tol_r: float = 1e-4,
           damping: float = 1e-3, max_step: float = 0.2):
    """Damped-least-squares IK for the TCP. Returns (q[7], converged)."""
    q = np.clip(np.asarray(q_init, dtype=float)[:7].copy(), Q_LOWER, Q_UPPER)
    for _ in range(max_iter):
        J, T = tcp_jacobian(q)
        e = np.concatenate([T_goal[:3, 3] - T[:3, 3], rot_log(T_goal[:3, :3] @ T[:3, :3].T)])
        if np.linalg.norm(e[:3]) < tol_p and np.linalg.norm(e[3:]) < tol_r:
            return q, True
        dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), e)
        n = np.abs(dq).max()
        if n > max_step:
            dq *= max_step / n
        q = np.clip(q + dq, Q_LOWER + 1e-4, Q_UPPER - 1e-4)
    return q, False


def tcp_pose_for_camera(target_xy, cam_z: float, closing_yaw: float) -> np.ndarray:
    """Top-down TCP pose that puts the hand camera at (x, y, cam_z), looking straight down.

    The camera's optical axis is parallel to the approach axis, so the camera
    origin is a fixed offset from the TCP in the TCP frame.
    """
    T = top_down_pose([0.0, 0.0, 0.0], closing_yaw)
    d = (np.linalg.inv(_HAND_TO_TCP) @ _HAND_TO_CAMERA)[:3, 3]
    T[:3, 3] = np.array([target_xy[0], target_xy[1], cam_z]) - T[:3, :3] @ d
    return T


def tcp_yaw(q) -> float:
    """World yaw of the finger closing axis (TCP y axis)."""
    T = tcp_pose(q)
    return float(np.arctan2(T[1, 1], T[0, 1]))


def top_down_pose(center, closing_yaw: float) -> np.ndarray:
    """TCP pose pointing straight down with the fingers closing along ``closing_yaw`` (world)."""
    approaching = np.array([0.0, 0.0, -1.0])
    closing = np.array([np.cos(closing_yaw), np.sin(closing_yaw), 0.0])
    T = np.eye(4)
    T[:3, :3] = np.stack([np.cross(closing, approaching), closing, approaching], axis=1)
    T[:3, 3] = center
    return T
