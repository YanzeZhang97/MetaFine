"""Motion primitives for ``pd_joint_delta_pos``.

Task routines are generators: they ``yield`` one 8-dim action per control step
and read the newest observation from the shared :class:`Ctx`, which the policy
refreshes before resuming them. Arm actions are closed-loop: the env sets the
joint target to ``measured qpos + 0.1 * action``, so we send
``(q_ref - q_meas) / 0.1`` and let the reference walk along a timed path.
"""
from __future__ import annotations

import math

import numpy as np

from . import robot

ARM_DELTA_LIMIT = 0.1  # rad per control step at |action| = 1
GRIPPER_OPEN, GRIPPER_CLOSED = 1.0, -1.0
MAX_PATH_RAD = 50.0    # inf-norm length of any sane joint path is a few rad
MAX_REFS = 20_000      # per-step references in one timed path (episodes are <= 800 steps)


class MotionError(RuntimeError):
    """A primitive could not produce a feasible motion (IK failure, bad target)."""


class Ctx:
    """Per-episode state shared between the policy and the running routine."""

    def __init__(self, task_id: str, variant: str, instruction: str):
        self.task_id = task_id
        self.variant = variant
        self.instruction = instruction
        self.q = np.zeros(9)
        self.images: dict = {}
        self.step = 0
        self.gripper = GRIPPER_OPEN
        self.notes: dict = {}

    def arm_action(self, q_ref) -> np.ndarray:
        d = (np.asarray(q_ref, dtype=float)[:7] - self.q[:7]) / ARM_DELTA_LIMIT
        return np.concatenate([np.clip(d, -1.0, 1.0), [self.gripper]]).astype(np.float32)

    def hold_action(self) -> np.ndarray:
        return np.concatenate([np.zeros(7), [self.gripper]]).astype(np.float32)

    @property
    def finger_width(self) -> float:
        return float(self.q[7] + self.q[8])


def time_parameterize(q_start, waypoints, v_max: float = 0.05, acc: float = 0.006):
    """Resample a joint path into per-step references with a trapezoidal speed
    profile on the inf-norm arc length (rad per control step)."""
    pts = [np.asarray(q_start, dtype=float)[:7]] + [np.asarray(w, dtype=float)[:7] for w in waypoints]
    if not all(np.all(np.isfinite(p)) for p in pts):
        raise MotionError("non-finite joint path")
    seg = np.array([np.abs(pts[i + 1] - pts[i]).max() for i in range(len(pts) - 1)])
    total = float(seg.sum())
    if total < 1e-9:
        return [pts[-1]]
    if total > MAX_PATH_RAD or not v_max > 0:
        raise MotionError(f"implausible joint path: length {total:.3g} rad, v_max {v_max}")
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    refs, s = [], 0.0
    while s < total - 1e-9:
        if len(refs) >= MAX_REFS:
            raise MotionError("timed path too long")
        v = min(v_max, math.sqrt(2 * acc * max(s, 1e-4)) + 0.004, math.sqrt(2 * acc * (total - s)) + 0.004)
        s = min(total, s + v)
        k = min(int(np.searchsorted(cum, s, side="right")) - 1, len(seg) - 1)
        t = 0.0 if seg[k] < 1e-12 else (s - cum[k]) / seg[k]
        refs.append(pts[k] + t * (pts[k + 1] - pts[k]))
    return refs


def follow(ctx: Ctx, refs, settle_tol: float = 2e-3, max_settle: int = 15):
    """Track per-step joint references, then wait until the arm settles on the last one."""
    for q_ref in refs:
        yield ctx.arm_action(q_ref)
    q_end = refs[-1]
    for _ in range(max_settle):
        if np.abs(ctx.q[:7] - q_end).max() < settle_tol:
            break
        yield ctx.arm_action(q_end)


def move_joints(ctx: Ctx, q_goal, v_max: float = 0.05, **kw):
    yield from follow(ctx, time_parameterize(ctx.q[:7], [q_goal], v_max), **kw)


def line_waypoints(q_start, T_goal, step_m: float = 0.005, step_rad: float = math.radians(2.0)):
    """IK-chained joint waypoints for a straight TCP line (position lerp, rotation slerp)."""
    T0 = robot.tcp_pose(q_start)
    dp = T_goal[:3, 3] - T0[:3, 3]
    w = robot.rot_log(T0[:3, :3].T @ T_goal[:3, :3])
    n = max(1, int(math.ceil(max(np.linalg.norm(dp) / step_m, np.linalg.norm(w) / step_rad))))
    q_prev, out = np.asarray(q_start, dtype=float)[:7], []
    for i in range(1, n + 1):
        s = i / n
        T = np.eye(4)
        T[:3, :3] = T0[:3, :3] @ robot.rot_exp(w * s)
        T[:3, 3] = T0[:3, 3] + dp * s
        q, ok = robot.ik_tcp(T, q_prev)
        if not ok:
            raise MotionError(f"IK failed at {i}/{n} towards {np.round(T_goal[:3, 3], 3)}")
        out.append(q)
        q_prev = q
    return out


def move_line(ctx: Ctx, T_goal, v_max: float = 0.04, acc: float = 0.006, **kw):
    """Move the TCP along a straight line to ``T_goal``."""
    yield from follow(ctx, time_parameterize(ctx.q[:7], line_waypoints(ctx.q[:7], T_goal), v_max, acc), **kw)


def ik_multi(T_goal, q_ref, n_random: int = 24, seed: int = 0):
    """IK from ``q_ref`` first, then random restarts; returns the solution closest to
    ``q_ref`` that keeps a margin from the joint limits, or None."""
    rng = np.random.default_rng(seed)
    q_ref = np.asarray(q_ref, dtype=float)[:7]
    seeds = [q_ref, robot.Q_HOME] + [robot.Q_HOME + rng.uniform(-1.2, 1.2, 7) for _ in range(n_random)]
    sols = []
    for s in seeds:
        q, ok = robot.ik_tcp(T_goal, np.clip(s, robot.Q_LOWER, robot.Q_UPPER))
        if ok and np.all(q > robot.Q_LOWER + 0.03) and np.all(q < robot.Q_UPPER - 0.03):
            sols.append(q)
            if len(sols) >= 4:
                break
    if not sols:
        return None
    return min(sols, key=lambda q: np.abs(q - q_ref).max())


def move_to_pose(ctx: Ctx, T_goal, v_max: float = 0.05, **kw):
    """Straight TCP line when IK can follow it, otherwise a joint-space move to an IK
    solution (use only in free space: the joint path does not keep the TCP on a line)."""
    try:
        wps = line_waypoints(ctx.q[:7], T_goal)
    except MotionError:
        q = ik_multi(T_goal, ctx.q[:7])
        if q is None:
            raise MotionError(f"no IK solution for {np.round(T_goal[:3, 3], 3)}")
        wps = [q]
    yield from follow(ctx, time_parameterize(ctx.q[:7], wps, v_max), **kw)


def look_from_above(ctx: Ctx, xy, cam_z: float, heights=(0.0, -0.05, -0.10)):
    """Put the hand camera straight above ``xy``; tries other wrist yaws / lower heights when
    the first choice is out of reach. Returns True when a view was reached."""
    yaw0 = robot.tcp_yaw(ctx.q)
    for dz in heights:
        for dyaw in (0.0, np.pi / 2, -np.pi / 2, np.pi):
            T = robot.tcp_pose_for_camera(xy, cam_z + dz, yaw0 + dyaw)
            if ik_multi(T, ctx.q[:7], n_random=4) is None:
                continue
            try:
                yield from move_to_pose(ctx, T, v_max=0.05, settle_tol=1e-3, max_settle=20)
                return True
            except MotionError:
                continue
    return False


def set_gripper(ctx: Ctx, value: float, steps: int = 6):
    """Command the gripper and hold the arm where it is for ``steps`` control steps."""
    ctx.gripper = float(value)
    q_hold = ctx.q[:7].copy()
    for _ in range(steps):
        yield ctx.arm_action(q_hold)


def hold(ctx: Ctx, steps: int | None = None):
    """Keep the arm still (forever when ``steps`` is None)."""
    q_hold = ctx.q[:7].copy()
    n = 0
    while steps is None or n < steps:
        yield ctx.arm_action(q_hold)
        n += 1
