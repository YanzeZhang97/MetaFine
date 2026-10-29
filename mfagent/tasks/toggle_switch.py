"""T3 toggle_switch_table: push the red or blue slider of the lying switch and hold it.

Mirrors core/skill.py:toggle_switch: close the gripper, come down along the
slider's annotated approach axis, push 2.2 cm along the stroke and keep holding
(the sliders are springy and snap back when released).
"""
from __future__ import annotations

import numpy as np

from .. import motion, robot
from ..motion import Ctx, MotionError
from ..perception import switch

SURVEY_Q = np.array([-0.0198, 0.0634, -0.0622, -1.4623, 0.004, 1.5257, 0.7038])
CLOSE_VIEW_Z = 0.30   # hand-camera height for the close look at the switch
PRE_DIST = 0.04       # back-off along the approach axis before pushing
PUSH_DIST = 0.022     # push length along the stroke (stroke is 2.05 cm)

# grasp_poses.json "button" matrices (translation scaled by 0.08), switch root frame.
GRASP_LOCAL = [
    np.array([[0.00701, 0.07328, 0.99729, -0.04532],
              [-0.01602, -0.99718, 0.07339, -0.01612],
              [0.99985, -0.01649, -0.00582, -0.03472],
              [0.0, 0.0, 0.0, 1.0]]),
    np.array([[0.0, 0.08715, 0.9962, -0.04619],
              [0.0, -0.9962, 0.08715, 0.01535],
              [1.0, 0.0, 0.0, -0.03667],
              [0.0, 0.0, 0.0, 1.0]]),
]


def _orthonormal(T):
    U, _, Vt = np.linalg.svd(T[:3, :3])
    T = T.copy()
    T[:3, :3] = U @ Vt
    return T


def root_pose(x, y, theta) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    T = np.eye(4)
    T[:3, :3] = Rz @ Ry
    T[:3, 3] = [x, y, switch.ROOT_Z]
    return T


def _push_poses(est: dict, link: int, q):
    """Pre-push, contact and end-of-push TCP poses for slider ``link``."""
    G = root_pose(est["x"], est["y"], est["theta"]) @ _orthonormal(GRASP_LOCAL[link])
    push_dir = G[:3, 0].copy()
    approach = G[:3, 2].copy()
    # The closed fingers are symmetric under a half turn about the approach axis:
    # use whichever version needs less wrist rotation.
    flipped = G.copy()
    flipped[:3, 0], flipped[:3, 1] = -G[:3, 0], -G[:3, 1]
    cur = robot.tcp_pose(q)[:3, :3]
    if np.linalg.norm(robot.rot_log(cur.T @ flipped[:3, :3])) < np.linalg.norm(robot.rot_log(cur.T @ G[:3, :3])):
        G = flipped
    pre, end = G.copy(), G.copy()
    pre[:3, 3] = G[:3, 3] - approach * PRE_DIST
    end[:3, 3] = G[:3, 3] + push_dir * PUSH_DIST
    return pre, G, end


def routine(ctx: Ctx):
    target = ctx.variant if ctx.variant in ("red", "blue") else "red"
    yield from motion.move_joints(ctx, SURVEY_Q, v_max=0.06, settle_tol=1e-3, max_settle=20)
    xy = switch.find_coarse(ctx.images["hand_camera"], ctx.q)
    if xy is None:
        ctx.notes["error"] = "switch not found"
        yield from motion.hold(ctx)
        return
    view = robot.tcp_pose_for_camera(xy, CLOSE_VIEW_Z, robot.tcp_yaw(ctx.q))
    yield from motion.move_line(ctx, view, v_max=0.05, settle_tol=1e-3, max_settle=20)
    est = switch.fit_switch(ctx.images["hand_camera"], ctx.q)
    ctx.notes["switch"] = est
    if est is None:
        yield from motion.hold(ctx)
        return
    blue = est["blue_link"] if est["blue_link"] is not None else 1
    link = blue if target == "blue" else 1 - blue
    pre, contact, end = _push_poses(est, link, ctx.q)
    yield from motion.set_gripper(ctx, motion.GRIPPER_CLOSED, steps=6)
    try:
        yield from motion.move_line(ctx, pre, v_max=0.05)
        yield from motion.move_line(ctx, contact, v_max=0.02, settle_tol=1e-3)
        yield from motion.move_line(ctx, end, v_max=0.015, settle_tol=5e-4, max_settle=30)
    except MotionError as exc:
        ctx.notes["error"] = str(exc)
    yield from motion.hold(ctx)
