"""T1 grasp_part: grasp the cap (top-down) or the body (side grasp) of bottle 3558.

Grasp heights follow the asset annotations (grasp_poses.json): cap at the root
+ 0.13 m, body at the root height. Success only needs a held grasp on the right
part for 5 steps, so we grasp and keep holding; after a top-down grasp the arm
rises at full speed meanwhile, to clear the next episode's bottle drop (DASH).
"""
from __future__ import annotations

import numpy as np

from .. import motion, robot
from ..motion import Ctx, MotionError
from ..perception.objects import BOTTLE_ROOT_Z, bottle_shape, find_bottle

SURVEY_Q = np.array([-0.051, -0.2903, -0.087, -1.585, -0.0258, 1.2959, 0.6441])
CLOSE_Z = 0.45            # camera height for the close look (cap top is at 0.257)
CAP_TCP_Z = BOTTLE_ROOT_Z + 0.13
BODY_TCP_Z = BOTTLE_ROOT_Z
BODY_BACKOFF = 0.01       # annotation puts the TCP 1 cm short of the bottle axis
BASE_XY = np.array([-0.615, 0.0])
SIDE_ENTRY_Z = 0.25       # TCP height where the hand turns sideways, well above the bottle
LYING_TCP_Z = 0.03        # bottle axis height when lying on its side
# Success fires 5 steps into a held grasp and ends the episode with the arm where it stopped. The
# next episode (same env) teleports a new bottle to z 0.2 and lets it fall while the drives still
# hold the last targets; the robot goes home only afterwards. A hand left low knocks over or throws
# the new bottle: 4.3 % of re-drops after a top-down cap grasp, 12.1 % after a side body grasp.
# After a top-down grasp we therefore rise at full joint speed (DASH): the arm ends 15-25 cm higher
# and no re-drop in 1600 simulated was disturbed. A side body grasp cannot rise far enough in 5
# steps (hand still below 0.30 m), so it keeps the plain close.
DASH = True
DASH_DELAY = 2   # closing steps first: the cap is in the fingers by then (1 step was 11/20)
DASH_STEPS = 8


def locate(ctx: Ctx):
    """Survey, then two close looks re-centred over the estimate. Returns (x, y) or None."""
    yield from motion.move_joints(ctx, SURVEY_Q, v_max=0.06, settle_tol=1e-3, max_settle=20)
    ctx.notes["views"] = [ctx.images["hand_camera"]]
    est = find_bottle(ctx.images["hand_camera"], ctx.q)
    if est is None:
        return None
    for _ in range(2):
        ok = yield from motion.look_from_above(ctx, [est["x"], est["y"]], CLOSE_Z)
        if not ok:
            break
        ctx.notes["views"].append(ctx.images["hand_camera"])
        new = find_bottle(ctx.images["hand_camera"], ctx.q)
        if new is None:
            break
        est = new
    ctx.notes["bottle"] = est
    ctx.notes["shape"] = bottle_shape(ctx.images["hand_camera"], ctx.q)
    return np.array([est["x"], est["y"]])


def choose_side_approach(xy, q):
    """Horizontal approach direction (radial from the base, then rotated) whose pre-grasp and
    grasp poses are both reachable; returns (approach_xy, closing_sign) or None."""
    base = np.arctan2(*(xy - BASE_XY)[::-1])
    for d in np.radians([0, 30, -30, 60, -60, 90, -90]):
        a = np.array([np.cos(base + d), np.sin(base + d)])
        for sign in (1.0, -1.0):
            pre = side_grasp_pose(xy - 0.13 * a, BODY_TCP_Z, a, sign)
            q_pre = motion.ik_multi(pre, q, n_random=8)
            if q_pre is None:
                continue
            q_g, ok = robot.ik_tcp(side_grasp_pose(xy - BODY_BACKOFF * a, BODY_TCP_Z, a, sign), q_pre)
            if not ok or np.abs(q_g - q_pre).max() > 0.6:
                continue
            # Same orientation 14 cm higher, seeded from the low pre-grasp: a free-space entry point.
            q_high, ok = robot.ik_tcp(side_grasp_pose(xy - 0.13 * a, SIDE_ENTRY_Z, a, sign), q_pre)
            if not (ok and np.all(q_high > robot.Q_LOWER + 0.03) and np.all(q_high < robot.Q_UPPER - 0.03)):
                continue
            # The descent and the horizontal approach must both be followable as IK chains.
            try:
                wps = motion.line_waypoints(q_high, side_grasp_pose(xy - 0.13 * a, BODY_TCP_Z, a, sign))
                motion.line_waypoints(wps[-1], side_grasp_pose(xy - BODY_BACKOFF * a, BODY_TCP_Z, a, sign))
            except motion.MotionError:
                continue
            return a, sign, q_high
    return None


def cap_grasp_pose(xy, q):
    """Top-down cap grasp; for far bottles, tilt the approach toward the robot until reachable."""
    yaw = robot.tcp_yaw(q)
    radial = np.arctan2(*(xy - BASE_XY)[::-1])
    for tilt in np.radians([0, 15, 30, 45]):
        for dyaw in (0.0, np.pi / 2, -np.pi / 2):
            T = robot.top_down_pose([xy[0], xy[1], CAP_TCP_Z], yaw + dyaw)
            if tilt > 0:
                # rotate about the horizontal axis perpendicular to the radial direction,
                # so the gripper leans back toward the robot base
                axis = np.array([-np.sin(radial), np.cos(radial), 0.0])
                T[:3, :3] = robot.rot_exp(-axis * tilt) @ T[:3, :3]
            pre = T.copy()
            pre[:3, 3] -= T[:3, 2] * 0.08
            if motion.ik_multi(pre, q, n_random=6) is not None and motion.ik_multi(T, q, n_random=6) is not None:
                return T
    return None


def lying_grasp(target_xy, yaw, q):
    """Pinch across a lying bottle from above; for far targets lean the gripper back toward
    the robot about its closing axis. Returns (grasp, pre-grasp) or (None, None)."""
    radial = np.arctan2(*(np.asarray(target_xy) - BASE_XY)[::-1])
    for tilt in np.radians([0, 20, 35, 50]):
        for flip in (0.0, np.pi):
            G = robot.top_down_pose([target_xy[0], target_xy[1], LYING_TCP_Z], yaw + flip)
            if tilt > 0:
                # lean about the closing axis, choosing the sign that tips the hand toward the base
                for sgn in (1.0, -1.0):
                    M = G.copy()
                    M[:3, :3] = robot.rot_exp(G[:3, 1] * sgn * tilt) @ G[:3, :3]
                    away = M[:3, 2][:2] @ np.array([np.cos(radial), np.sin(radial)])
                    if away > 0:  # approach axis points away from the base => hand is on the base side
                        G = M
                        break
            P = G.copy()
            P[:3, 3] -= G[:3, 2] * 0.08
            if motion.ik_multi(P, q, n_random=6) is not None and motion.ik_multi(G, q, n_random=6) is not None:
                return G, P
    return None, None


def side_grasp_pose(xy, z, approach_xy, closing_sign: float = 1.0) -> np.ndarray:
    a = np.array([approach_xy[0], approach_xy[1], 0.0])
    c = closing_sign * np.array([-a[1], a[0], 0.0])
    T = np.eye(4)
    T[:3, :3] = np.stack([np.cross(c, a), c, a], axis=1)
    T[:3, 3] = [xy[0], xy[1], z]
    return T


def close_and_dash(ctx: Ctx):
    """Close the gripper, then drive every joint at full speed in the direction that raises the
    TCP, orientation free (about 5 cm per step; a straight lift that keeps the orientation
    manages 1.7-2.3 cm). See DASH."""
    ctx.gripper = motion.GRIPPER_CLOSED
    q0 = ctx.q[:7].copy()
    for _ in range(DASH_DELAY):
        yield ctx.arm_action(q0)
    for _ in range(DASH_STEPS):
        J, _ = robot.tcp_jacobian(ctx.q[:7])
        d = np.where(np.abs(J[2]) > 0.02, np.sign(J[2]), 0.0)
        yield ctx.arm_action(np.clip(ctx.q[:7] + motion.ARM_DELTA_LIMIT * d, robot.Q_LOWER + 0.05,
                                     robot.Q_UPPER - 0.05))


def routine(ctx: Ctx):
    part = ctx.variant if ctx.variant in ("cap", "body") else "cap"
    xy = yield from locate(ctx)
    if xy is None:
        ctx.notes["error"] = "bottle not found"
        yield from motion.hold(ctx)
        return
    shape = ctx.notes.get("shape")
    top_down = True
    try:
        if shape and shape.get("lying"):
            # Knocked over (e.g. during the env's re-drop at reset): pinch the cap or the
            # body from above, fingers closing across the bottle axis.
            target = np.array(shape["cap_xy"] if part == "cap" else shape["body_xy"])
            ax = np.array(shape["axis"])
            yaw = float(np.arctan2(ax[0], -ax[1]))  # closing axis perpendicular to the bottle axis
            grasp, pre = lying_grasp(target, yaw, ctx.q[:7])
            if grasp is None:
                raise MotionError("lying bottle out of reach")
            yield from motion.move_to_pose(ctx, pre, v_max=0.05)
            yield from motion.move_line(ctx, grasp, v_max=0.02, settle_tol=1e-3)
        elif part == "cap":
            grasp = cap_grasp_pose(xy, ctx.q[:7])
            if grasp is None:
                raise MotionError("no reachable cap grasp")
            pre = grasp.copy()
            pre[:3, 3] -= grasp[:3, 2] * 0.08  # back off along the approach axis
            yield from motion.move_to_pose(ctx, pre, v_max=0.05)
            yield from motion.move_line(ctx, grasp, v_max=0.02, settle_tol=1e-3)
        else:
            choice = choose_side_approach(xy, ctx.q[:7])
            if choice is None:
                raise MotionError("no reachable side approach")
            a, sign, q_high = choice
            top_down = False
            grasp_xy = xy - BODY_BACKOFF * a
            pre_xy = xy - 0.13 * a
            yield from motion.move_joints(ctx, q_high, v_max=0.05, settle_tol=2e-3)
            yield from motion.move_line(ctx, side_grasp_pose(pre_xy, BODY_TCP_Z, a, sign), v_max=0.04)
            yield from motion.move_line(ctx, side_grasp_pose(grasp_xy, BODY_TCP_Z, a, sign), v_max=0.02,
                                        settle_tol=1e-3)
        if DASH and top_down:
            yield from close_and_dash(ctx)
        else:
            yield from motion.set_gripper(ctx, motion.GRIPPER_CLOSED, steps=10)
    except MotionError as exc:
        ctx.notes["error"] = str(exc)
    yield from motion.hold(ctx)
