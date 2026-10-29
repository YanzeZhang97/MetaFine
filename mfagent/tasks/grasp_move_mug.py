"""T2 grasp_move_mug: pinch the mug handle from above, lift, and carry it (up to 20 cm).

Success (configs/t2_mug_move_*.yaml): still grasping, and since the first grasp
the TCP moved >= 7 cm and the mug >= 5 cm along the commanded axis. Left is
world -Y, right is world +Y, forward is world +X.
"""
from __future__ import annotations

import numpy as np

from .. import motion, robot
from ..motion import Ctx, MotionError
from ..perception.objects import MUG_ROOT_Z, find_mug

SURVEY_Q = np.array([-0.0027, -0.3705, -0.034, -1.9331, -0.0123, 1.563, 0.751])
CLOSE_Z = 0.40
# The next episode re-drops the mug from z = 0.2 m before the robot is reset, so the
# episode should end (on success, mid-carry) with the arm well above the drop zone:
# carry at the highest of these TCP heights that IK can follow.
LIFT_ZS = (0.40, 0.32, 0.25)
# Success fires 7-11 cm into the carry: carried forward, the mug pivots on the pinch and
# lags the TCP. Aim well past that; the shorter carries are fallbacks near the workspace edge.
MOVE_DISTS = (0.20, 0.16, 0.12)
FAST_DIST = 0.062     # success needs >= 7 cm of TCP travel; cross it slowly
DIRECTION = {"left": np.array([0.0, -1.0, 0.0]), "right": np.array([0.0, 1.0, 0.0]),
             "forward": np.array([1.0, 0.0, 0.0])}

# grasp_poses.json "handle" (scale 0.08): top-down pinch on the outer C-handle, mug frame.
_HANDLE_P = np.array([-0.0626, 0.0132, 0.0365])
_HANDLE_CLOSING = np.array([0.219, 0.9757, 0.0]) / np.linalg.norm([0.219, 0.9757, 0.0])


def handle_grasp_pose(x, y, yaw) -> np.ndarray:
    a = np.array([0.0, 0.0, -1.0])
    G = np.eye(4)
    G[:3, :3] = np.stack([np.cross(_HANDLE_CLOSING, a), _HANDLE_CLOSING, a], axis=1)
    G[:3, 3] = _HANDLE_P
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.eye(4)
    R[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    R[:3, 3] = [x, y, MUG_ROOT_Z]
    return R @ G


def _closer_half_turn(T, q):
    """The gripper is symmetric under a half turn about the approach axis; pick the cheaper one."""
    F = T.copy()
    F[:3, 0], F[:3, 1] = -T[:3, 0], -T[:3, 1]
    cur = robot.tcp_pose(q)[:3, :3]
    if np.linalg.norm(robot.rot_log(cur.T @ F[:3, :3])) < np.linalg.norm(robot.rot_log(cur.T @ T[:3, :3])):
        return F
    return T


def _carry_plan(q, G, direction):
    """(lift height, carry distance) whose three straight TCP lines (lift, fast, creep) IK can
    follow from the grasp configuration ``q``: carries of at least 16 cm from the highest lift
    down, then 12 cm carries; None when nothing is reachable."""
    order = [(z, d) for z in LIFT_ZS for d in MOVE_DISTS[:-1]] + [(z, MOVE_DISTS[-1]) for z in LIFT_ZS]
    reach = {}
    for z, dist in order:
        if z not in reach:
            lift = G.copy()
            lift[2, 3] = z
            fast = lift.copy()
            fast[:3, 3] += direction * FAST_DIST
            try:
                q_lift = motion.line_waypoints(q, lift)[-1]
                reach[z] = (lift, motion.line_waypoints(q_lift, fast)[-1])
            except MotionError:
                reach[z] = None
        if reach[z] is None:
            continue
        lift, q_fast = reach[z]
        carry = lift.copy()
        carry[:3, 3] += direction * dist
        try:
            motion.line_waypoints(q_fast, carry)
        except MotionError:
            continue
        return z, dist
    return None


def routine(ctx: Ctx):
    direction = DIRECTION.get(ctx.variant, DIRECTION["left"])
    yield from motion.move_joints(ctx, SURVEY_Q, v_max=0.06, settle_tol=1e-3, max_settle=20)
    ctx.notes["views"] = [ctx.images["hand_camera"]]
    est = find_mug(ctx.images["hand_camera"], ctx.q)
    if est is None:
        ctx.notes["error"] = "mug not found"
        yield from motion.hold(ctx)
        return
    for _ in range(2):
        ok = yield from motion.look_from_above(ctx, [est["x"], est["y"]], CLOSE_Z)
        if not ok:
            break
        ctx.notes["views"].append(ctx.images["hand_camera"])
        new = find_mug(ctx.images["hand_camera"], ctx.q)
        if new is not None and new.get("yaw") is not None:
            est = new
    ctx.notes["mug"] = est
    if est.get("yaw") is None:
        ctx.notes["error"] = "handle not found"
        yield from motion.hold(ctx)
        return
    G = _closer_half_turn(handle_grasp_pose(est["x"], est["y"], est["yaw"]), ctx.q)
    pre = G.copy()
    pre[2, 3] += 0.08
    try:
        yield from motion.move_to_pose(ctx, pre, v_max=0.05)
        yield from motion.move_line(ctx, G, v_max=0.02, settle_tol=1e-3)
        yield from motion.set_gripper(ctx, motion.GRIPPER_CLOSED, steps=10)
        lift_z, move_dist = _carry_plan(ctx.q[:7], G, direction) or (LIFT_ZS[0], MOVE_DISTS[0])
        ctx.notes["carry"] = (lift_z, move_dist)
        lift = G.copy()
        lift[2, 3] = lift_z
        yield from motion.move_line(ctx, lift, v_max=0.025)
        # The harness ends the episode on the first successful step (TCP >= 7 cm since the
        # grasp): carry most of the way, let the swinging mug settle, then creep across.
        fast = lift.copy()
        fast[:3, 3] += direction * FAST_DIST
        yield from motion.move_line(ctx, fast, v_max=0.03, settle_tol=1e-3)
        yield from motion.hold(ctx, steps=12)
        carry = lift.copy()
        carry[:3, 3] += direction * move_dist
        yield from motion.move_line(ctx, carry, v_max=0.004, settle_tol=1e-3)
    except MotionError as exc:
        ctx.notes["error"] = str(exc)
    yield from motion.hold(ctx)
