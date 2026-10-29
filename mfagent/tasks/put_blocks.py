"""T4 put_blocks_into_boxes: special cube -> left box, the other two -> right box.

The env scores the stages in a fixed order (special, then others[0], then
others[1], with others in red/blue/green order), so we place them in that order.
Round 1 uses the official box positions (core/env.py BOX_X/BOX_*_Y/BOX_Z).
"""
from __future__ import annotations

import numpy as np

from .. import motion, robot
from ..motion import Ctx, MotionError
from ..perception.cubes import CUBE_HALF, detect_cubes

SURVEY_Q = np.array([-0.0407, 0.0594, 0.0056, -1.982, -0.0004, 2.0415, 0.7506])
BOX_CENTER = {"left": np.array([0.10, -0.20, 0.05]), "right": np.array([0.10, 0.20, 0.05])}
# Drop spots inside the boxes (x, y offsets from the centre) for the 1st / 2nd / 3rd cube.
DROP_OFFSETS = [np.array([0.0, 0.0]), np.array([-0.035, 0.0]), np.array([0.035, 0.0])]
# Spawn windows (x range, y range) per colour, used only as a last-resort guess.
SLOT_PRIOR = {"red": (-0.075, -0.14), "blue": (-0.075, 0.0), "green": (-0.075, 0.14)}

CARRY_Z = 0.20       # TCP height while moving between cube and box
PREGRASP_DZ = 0.08   # TCP height above the cube centre before descending
RELEASE_Z = 0.105    # TCP height when opening over a box (cube bottom ~1 cm above the rim)
MIN_GRASP_WIDTH = 0.03  # finger opening below this after closing => missed the 4 cm cube


def _closing_yaw(cube_yaw: float, q) -> float:
    """Pick the cube face pair whose normal is closest to the current finger closing axis."""
    y_axis = robot.tcp_pose(q)[:3, 1]
    cur = np.arctan2(y_axis[1], y_axis[0])
    cands = cube_yaw + np.arange(4) * (np.pi / 2)
    diff = np.angle(np.exp(1j * (cands - cur)))
    return float(cands[int(np.argmin(np.abs(diff)))])


def _survey(ctx: Ctx):
    yield from motion.move_joints(ctx, SURVEY_Q, v_max=0.06, settle_tol=1e-3, max_settle=20)


def _pick(ctx: Ctx, cube: dict):
    yaw = _closing_yaw(cube["yaw"], ctx.q)
    above = robot.top_down_pose([cube["x"], cube["y"], CUBE_HALF + PREGRASP_DZ], yaw)
    grasp = robot.top_down_pose([cube["x"], cube["y"], CUBE_HALF], yaw)
    yield from motion.move_line(ctx, above, v_max=0.05)
    yield from motion.move_line(ctx, grasp, v_max=0.025, settle_tol=1e-3)
    yield from motion.set_gripper(ctx, motion.GRIPPER_CLOSED, steps=8)
    lift = robot.top_down_pose([cube["x"], cube["y"], CARRY_Z], yaw)
    yield from motion.move_line(ctx, lift, v_max=0.05)
    return ctx.finger_width > MIN_GRASP_WIDTH


def _place(ctx: Ctx, box: str, slot: int):
    T = robot.tcp_pose(ctx.q)
    yaw = float(np.arctan2(T[1, 1], T[0, 1]))
    xy = BOX_CENTER[box][:2] + DROP_OFFSETS[slot]
    yield from motion.move_line(ctx, robot.top_down_pose([xy[0], xy[1], CARRY_Z], yaw), v_max=0.05)
    yield from motion.move_line(ctx, robot.top_down_pose([xy[0], xy[1], RELEASE_Z], yaw), v_max=0.04)
    yield from motion.set_gripper(ctx, motion.GRIPPER_OPEN, steps=6)
    yield from motion.move_line(ctx, robot.top_down_pose([xy[0], xy[1], CARRY_Z], yaw), v_max=0.05)


def routine(ctx: Ctx):
    special = ctx.variant if ctx.variant in ("red", "blue", "green") else "green"
    others = [c for c in ("red", "blue", "green") if c != special]
    plan = [(special, "left", 0), (others[0], "right", 1), (others[1], "right", 2)]
    for colour, box, slot in plan:
        for attempt in range(2):
            yield from _survey(ctx)
            cubes = detect_cubes(ctx.images["hand_camera"], ctx.q)
            ctx.notes.setdefault("detections", []).append({k: dict(v) for k, v in cubes.items()})
            cube = cubes.get(colour)
            if cube is None:
                x, y = SLOT_PRIOR[colour]
                cube = {"x": x, "y": y, "yaw": 0.0}
            try:
                grasped = yield from _pick(ctx, cube)
            except MotionError:
                grasped = False
            if grasped:
                yield from _place(ctx, box, slot)
                break
            yield from motion.set_gripper(ctx, motion.GRIPPER_OPEN, steps=4)
    yield from motion.hold(ctx)
