"""T5 insert_letter: pick the instructed letter peg and seat it in its slot.

Sequence:
  1. survey, then a close look straight above the target peg -> planar pose;
  2. squeeze-grasp: fingers close along the glyph x axis on a row whose span is
     symmetric about the TCP (C, L: the stem; o, R: the full width), which also
     squares the peg's yaw to the finger pads;
  3. carry above the slot (fixed board) with the peg's goal yaw;
  4. descend until the peg bottom touches the board, then press and spiral until
     the peg drops through the hole (detected from the measured TCP height);
  5. push it home, release, back off.
"""
from __future__ import annotations

import math

import numpy as np

from .. import letters, motion, robot
from ..motion import Ctx, MotionError
from ..perception.letters import fit_all, fit_letter

SURVEY_Q = np.array([-0.2162, 0.3495, -0.0819, -1.5842, 0.0301, 1.9325, -1.0891])
CLOSE_Z = 0.29            # hand-camera height for the close look (peg top is at 0.036)
LIFT_Z = 0.12             # TCP height while carrying the peg
HOVER_GAP = 0.02          # peg bottom above the board top before the final descent
PRESS_DEPTH = 0.004       # commanded depth below the contact height while searching
SPIRAL_PITCH = 0.0015
SPIRAL_MAX_R = 0.007
SEATED_DROP = 0.0025      # TCP this far below the contact height => peg is in the hole (< PRESS_DEPTH)
MIN_HOLD_WIDTH = 0.006    # both fingers closed below this => nothing between them (strokes are 10 mm)


def grasp_pose_for(letter: str, x: float, y: float, yaw: float, axis: str = "x", c: float | None = None) -> np.ndarray:
    """TCP pose for a squeeze grasp of a peg at (x, y, yaw): fingers close along the glyph
    x axis (row grasp) or glyph y axis (column grasp)."""
    if c is None:
        axis, c = letters.GRASPS[letter][0]
    g = letters.grasp_point(letter, axis, c)
    cy, sy = math.cos(yaw), math.sin(yaw)
    p = np.array([x + cy * g[0] - sy * g[1], y + sy * g[0] + cy * g[1], letters.GRASP_TCP_Z])
    return robot.top_down_pose(p, yaw if axis == "x" else yaw + math.pi / 2)


def gripper_action_for_opening(per_finger: float) -> float:
    """Gripper action for a finger joint target (metres per finger, range -0.01..0.04)."""
    t = min(max(per_finger, -0.01), 0.04)
    return 2.0 * (t + 0.01) / 0.05 - 1.0


def _finger_clearance(G, opening, others: dict) -> float:
    """Smallest gap between the open fingers and any other peg (pegs as 33 mm discs)."""
    c = G[:3, 1]
    gap = 1.0
    for est in others.values():
        centre = np.array([est["x"], est["y"]])
        for side in (1.0, -1.0):
            finger = G[:2, 3] + side * c[:2] * (opening + 0.008)
            gap = min(gap, float(np.linalg.norm(finger - centre)) - 0.033)
    return gap


def choose_grasps(letter: str, est: dict, others: dict) -> list:
    """Grasp options ordered by preference: collision-free first, then by clearance."""
    opts = []
    for i, (axis, c) in enumerate(letters.GRASPS[letter]):
        G = grasp_pose_for(letter, est["x"], est["y"], est["yaw"], axis, c)
        opening = letters.grasp_width(letter, axis, c) / 2 + 0.012
        clear = _finger_clearance(G, opening, others)
        opts.append((clear < 0.0, -clear if clear < 0 else i, axis, c, opening))
    opts.sort(key=lambda o: (o[0], o[1]))
    return [(axis, c, opening) for _, _, axis, c, opening in opts]


def _tilted(T, angle):
    """Rotate a grasp about its finger closing axis (keeps the pads parallel to the squeezed faces)."""
    M = T.copy()
    M[:3, :3] = robot.rot_exp(T[:3, 1] * angle) @ T[:3, :3]
    return M


def _pregrasp(G, dist=0.09):
    P = G.copy()
    P[:3, 3] = G[:3, 3] - G[:3, 2] * dist  # back off along the approach axis
    return P


def _reachable_grasp(T, q):
    """Top-down grasps are symmetric under a half turn about the approach axis: prefer the
    version needing less wrist rotation; if neither is reachable (far corners), lean the
    gripper about its closing axis."""
    F = T.copy()
    F[:3, 0], F[:3, 1] = -T[:3, 0], -T[:3, 1]
    cur = robot.tcp_pose(q)[:3, :3]
    cands = sorted([T, F], key=lambda M: np.linalg.norm(robot.rot_log(cur.T @ M[:3, :3])))
    for tilt in np.radians([0.0, 15.0, -15.0, 30.0, -30.0]):
        for M0 in cands:
            M = _tilted(M0, tilt) if tilt else M0
            if motion.ik_multi(_pregrasp(M), q, n_random=6) is not None and motion.ik_multi(M, q, n_random=6) is not None:
                return M
    return cands[0]


def peg_hold_pose(letter: str, est: dict, G: np.ndarray, axis: str = "x", c: float | None = None) -> np.ndarray:
    """Peg-origin pose in the TCP frame predicted by the squeeze model: the fingers square the
    grasped glyph axis to the (horizontal) closing axis and centre the line's span between them."""
    if c is None:
        axis, c = letters.GRASPS[letter][0]
    cl = G[:3, 1]
    yaw_c = math.atan2(cl[1], cl[0]) - (0.0 if axis == "x" else math.pi / 2)
    yaw = min((yaw_c, yaw_c + math.pi), key=lambda a: abs(math.remainder(a - est["yaw"], 2 * math.pi)))
    g = letters.grasp_point(letter, axis, c)
    cy, sy = math.cos(yaw), math.sin(yaw)
    T_peg = np.eye(4)
    T_peg[:3, :3] = [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]]
    T_peg[:2, 3] = G[:2, 3] - np.array([cy * g[0] - sy * g[1], sy * g[0] + cy * g[1]])
    return np.linalg.inv(G) @ T_peg


def locate(ctx: Ctx, letter: str):
    oracle = ctx.notes.get("oracle_peg")  # dev-only hook (never set in the submission)
    if oracle is not None:
        return dict(oracle)
    yield from motion.move_joints(ctx, SURVEY_Q, v_max=0.06, settle_tol=1e-3, max_settle=20)
    ctx.notes["views"] = [ctx.images["hand_camera"]]
    found = fit_all(ctx.images["hand_camera"], ctx.q)
    ctx.notes["others"] = {L: v for L, v in found.items() if L != letter}
    est = found.get(letter)
    if est is None:
        return None
    ok = yield from motion.look_from_above(ctx, [est["x"], est["y"]], CLOSE_Z)
    if ok:
        ctx.notes["views"].append(ctx.images["hand_camera"])
        close = fit_letter(ctx.images["hand_camera"], ctx.q, letter, yaw_prior=est["yaw"],
                           near_xy=(est["x"], est["y"]))
        if close is not None:
            est = close
    return est


def insert(ctx: Ctx, letter: str, T_hold: np.ndarray):
    """Carry the held peg over its slot and seat it. ``T_hold``: peg-origin pose in the TCP frame."""
    slot = letters.slot_pose(letter)
    period = letters.SYMMETRY[letter]
    # Goal pose of the peg origin; for the 180-degree-symmetric o pick the closer equivalent yaw.
    goal_yaw = slot[2]
    if period < 2 * math.pi - 1e-6:
        T_peg_now = robot.tcp_pose(ctx.q) @ T_hold
        peg_yaw_now = math.atan2(T_peg_now[1, 0], T_peg_now[0, 0])
        k = round((peg_yaw_now - goal_yaw) / period)
        goal_yaw += k * period
    T_peg_goal = np.eye(4)
    c, s = math.cos(goal_yaw), math.sin(goal_yaw)
    T_peg_goal[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    T_peg_goal[:3, 3] = [slot[0], slot[1], 0.0]
    T_tcp_seated = T_peg_goal @ np.linalg.inv(T_hold)
    z_seated = T_tcp_seated[2, 3]
    z_contact = z_seated + letters.BOARD_H        # peg bottom on the board top

    above = T_tcp_seated.copy()
    above[2, 3] = LIFT_Z
    yield from motion.move_to_pose(ctx, above, v_max=0.04, settle_tol=1e-3)
    hover = T_tcp_seated.copy()
    hover[2, 3] = z_contact + HOVER_GAP
    yield from motion.move_line(ctx, hover, v_max=0.02, settle_tol=5e-4, max_settle=25)

    # Press down at the nominal xy; if the peg stops on the board, spiral while pressing.
    press = T_tcp_seated.copy()
    press[2, 3] = z_contact - PRESS_DEPTH
    yield from motion.move_line(ctx, press, v_max=0.01, settle_tol=5e-4, max_settle=10)
    seated = robot.tcp_pose(ctx.q)[2, 3] < z_contact - SEATED_DROP
    if not seated:
        r_steps = np.arange(SPIRAL_PITCH / 4, SPIRAL_MAX_R, SPIRAL_PITCH / 8)
        for i, r in enumerate(r_steps):
            ang = 2 * math.pi * r / SPIRAL_PITCH
            T = press.copy()
            T[0, 3] += r * math.cos(ang)
            T[1, 3] += r * math.sin(ang)
            q_t, ok = robot.ik_tcp(T, ctx.q[:7])
            if not ok:
                continue
            for _ in range(2):
                yield ctx.arm_action(q_t)
            if robot.tcp_pose(ctx.q)[2, 3] < z_contact - SEATED_DROP:
                seated = True
                break
    ctx.notes["seated"] = bool(seated)
    # Push the peg all the way down (its origin must end below 3 mm), then let go.
    home = robot.tcp_pose(ctx.q).copy()
    home[2, 3] = z_seated - 0.002
    home[:3, :3] = T_tcp_seated[:3, :3]
    try:
        yield from motion.move_line(ctx, home, v_max=0.01, settle_tol=5e-4, max_settle=15)
    except MotionError:
        pass
    yield from motion.set_gripper(ctx, motion.GRIPPER_OPEN, steps=8)
    up = robot.tcp_pose(ctx.q).copy()
    up[2, 3] = LIFT_Z
    yield from motion.move_line(ctx, up, v_max=0.02)


def routine(ctx: Ctx):
    letter = ctx.variant if ctx.variant in letters.GLYPHS else "C"
    est = yield from locate(ctx, letter)
    ctx.notes["peg"] = est
    if est is None:
        ctx.notes["error"] = "peg not found"
        yield from motion.hold(ctx)
        return
    try:
        held = False
        for axis, c, opening in choose_grasps(letter, est, ctx.notes.get("others", {})):
            for attempt in range(2):
                G = _reachable_grasp(grasp_pose_for(letter, est["x"], est["y"], est["yaw"], axis, c), ctx.q[:7])
                pre = _pregrasp(G)
                ctx.gripper = gripper_action_for_opening(opening)
                yield from motion.move_to_pose(ctx, pre, v_max=0.05)
                yield from motion.move_to_pose(ctx, G, v_max=0.02, settle_tol=5e-4)
                yield from motion.set_gripper(ctx, motion.GRIPPER_CLOSED, steps=10)
                if ctx.finger_width > MIN_HOLD_WIDTH:
                    held = True
                    break
                # Fingers closed on nothing: back off, look again.
                yield from motion.set_gripper(ctx, gripper_action_for_opening(opening), steps=6)
                yield from motion.move_line(ctx, pre, v_max=0.05)
                again = yield from locate(ctx, letter)
                if again is None:
                    raise MotionError("peg lost after a missed grasp")
                est = again
            if held:
                break
        if not held:
            raise MotionError("could not grasp the peg")
        T_hold = peg_hold_pose(letter, est, G, axis, c)
        ctx.notes["T_hold"] = T_hold
        ctx.notes["grasp"] = (axis, c)
        lift = G.copy()
        lift[2, 3] = LIFT_Z
        yield from motion.move_line(ctx, lift, v_max=0.03)
        yield from insert(ctx, letter, T_hold)
    except MotionError as exc:
        ctx.notes["error"] = str(exc)
    yield from motion.hold(ctx)
