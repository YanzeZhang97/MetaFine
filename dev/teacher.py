"""Dev: privileged teacher for the learned policy's demonstrations.

The mfagent routines with their perception calls answered from the simulator state. Every
observation move (survey pose, close looks) is kept, so the student sees the views it needs;
only the estimates come from the env. Never shipped.

  teacher = Teacher(env, task, variant, instruction)   # per episode, after env.reset
  action = teacher.act(state, images, step)            # teacher.failed is set after an exception
  teacher.aux()                                        # ground-truth targets for auxiliary heads
  teacher.goals(variants)                              # per-variant target positions (goal head)
"""
from __future__ import annotations

import numpy as np

from dev.simenv import to_np
from mfagent import motion, robot
from mfagent.motion import MotionError
from mfagent.tasks import get_routine


def _pose(actor) -> np.ndarray:
    return to_np(actor.pose.to_transformation_matrix()).reshape(4, 4)


class _Proxy:
    """A perception module with some functions replaced (other attributes pass through)."""

    def __init__(self, module, **overrides):
        self._module = module
        self.__dict__.update(overrides)

    def __getattr__(self, name):
        return getattr(self._module, name)


def _capped(gen, cap: int):
    """Run a motion generator with every settle in it (mfagent.motion.follow's max_settle) capped at ``cap`` steps.
    follow is looked up when a primitive starts, so it is patched only while the wrapped generator runs."""
    orig = motion.follow

    def follow(ctx, refs, settle_tol=2e-3, max_settle=15):
        return orig(ctx, refs, settle_tol=settle_tol, max_settle=min(max_settle, cap))
    while True:
        motion.follow = follow
        try:
            a = next(gen)
        except StopIteration as stop:
            return stop.value
        finally:
            motion.follow = orig
        yield a


class _ToggleSwitch:
    AUX_DIM = 6  # root x, y, cos, sin of the stroke yaw; xy of the instructed slider's push contact
    # DEEP_PUSH (m, default 0 = the l1-l6 teacher): move the contact and end-of-push poses this far further
    # along the approach axis (down, toward the switch body). The official contact gate needs the TCP within
    # 6 cm of the slider's link origin (at the switch root); the default push passes it by only ~4 mm, so a
    # student that pushes a few mm higher toggles the slider without ever latching contact.
    DEEP_PUSH = 0.0
    # Still phases a memoryless student can get stuck in (t3_c, fresh seeds 2,780,001+: hovering at the close
    # view for 190 steps, or never leaving the survey pose). SETTLE_CAP (steps, default None = the l1-l6
    # teacher) caps the settles at the survey, close-view and pre-push poses; the settle onto the push contact
    # stays complete so that the push height is exact. NO_CLOSE_HOLD: close the fingers while moving to the
    # pre-push pose instead of holding the arm still for 6 steps at the close view.
    SETTLE_CAP = None
    NO_CLOSE_HOLD = False
    # PRE_RANGE (m, (lo, hi), default None = the fixed 4 cm): per-episode distance of the pre-push pose back along
    # the approach axis; PRE_JITTER (m): per-episode offset of the pre-push pose across the approach axis. The
    # fixed pre pose let t3_c learn "descend 4 cm, then push": from a pre pose 10 mm too high it pushed 11 mm too
    # high (seed 2,780,021), toggled the slider and never latched the 6 cm contact gate. Varied starts make the
    # push height something the student must read from qpos, not a memorised stroke.
    PRE_RANGE = None
    PRE_JITTER = 0.0

    def __init__(self, env, variant):
        self.u, self.variant = env.unwrapped, variant

    def est(self) -> dict:
        T = _pose(self.u.target_object)
        return {"x": float(T[0, 3]), "y": float(T[1, 3]), "theta": float(np.arctan2(T[1, 2], T[0, 2])),
                "iou": 1.0, "blue_link": int(self.u.scene_builder.switch_color["blue"][-1]),
                "flip_evidence": None}

    def install(self):
        from mfagent.perception import switch
        from mfagent.tasks import toggle_switch as ts
        ts.switch = _Proxy(switch, find_coarse=lambda img, q: np.array([self.est()["x"], self.est()["y"]]),
                           fit_switch=lambda img, q: self.est())
        if not hasattr(ts, "_push_poses_orig"):
            ts._push_poses_orig = ts._push_poses
        cls = type(self)

        def push_poses(est, link, q):
            pre, contact, end = ts._push_poses_orig(est, link, q)
            if cls.DEEP_PUSH:
                shift = contact[:3, 2] * float(cls.DEEP_PUSH)  # the approach axis
                contact, end = contact.copy(), end.copy()
                contact[:3, 3] += shift
                end[:3, 3] += shift
            return pre, contact, end
        ts._push_poses = push_poses

    def routine(self, ctx):
        from mfagent.tasks import toggle_switch as ts
        if self.SETTLE_CAP is None and not self.NO_CLOSE_HOLD and self.PRE_RANGE is None and not self.PRE_JITTER:
            return ts.routine(ctx)  # the l1-l6 teacher
        return self._routine(ctx, ts)

    def _routine(self, ctx, ts):
        """mfagent's toggle_switch.routine (ground-truth estimates) with the still phases capped."""
        # Drawn here, not in __init__, so that the default teacher leaves NumPy's global RNG untouched. The global
        # RNG is seeded per episode (gen_demos, the eval loops) and the env has drawn its scene already.
        self.rng = np.random.default_rng([np.random.randint(2**31), 3])
        cap = 20 if self.SETTLE_CAP is None else int(self.SETTLE_CAP)
        target = ctx.variant if ctx.variant in ("red", "blue") else "red"
        yield from motion.move_joints(ctx, ts.SURVEY_Q, v_max=0.06, settle_tol=1e-3, max_settle=cap)
        xy = ts.switch.find_coarse(ctx.images["hand_camera"], ctx.q)
        view = robot.tcp_pose_for_camera(xy, ts.CLOSE_VIEW_Z, robot.tcp_yaw(ctx.q))
        yield from motion.move_line(ctx, view, v_max=0.05, settle_tol=1e-3, max_settle=cap)
        est = ts.switch.fit_switch(ctx.images["hand_camera"], ctx.q)
        ctx.notes["switch"] = est
        blue = est["blue_link"] if est["blue_link"] is not None else 1
        link = blue if target == "blue" else 1 - blue
        pre, contact, end = ts._push_poses(est, link, ctx.q)
        if self.PRE_RANGE is not None or self.PRE_JITTER:
            dist = self.rng.uniform(*self.PRE_RANGE) if self.PRE_RANGE is not None else ts.PRE_DIST
            pre = contact.copy()
            pre[:3, 3] = contact[:3, 3] - contact[:3, 2] * dist
            if self.PRE_JITTER:
                pre[:3, 3] += (contact[:3, :2] @ self.rng.uniform(-self.PRE_JITTER, self.PRE_JITTER, 2))
            ctx.notes["pre"] = [round(float(dist), 4)]
        if self.NO_CLOSE_HOLD:
            ctx.gripper = motion.GRIPPER_CLOSED
        else:
            yield from motion.set_gripper(ctx, motion.GRIPPER_CLOSED, steps=6)
        try:
            yield from motion.move_line(ctx, pre, v_max=0.05, max_settle=min(15, cap))
            yield from motion.move_line(ctx, contact, v_max=0.02, settle_tol=1e-3)
            yield from motion.move_line(ctx, end, v_max=0.015, settle_tol=5e-4, max_settle=30)
        except MotionError as exc:
            ctx.notes["error"] = str(exc)
        yield from motion.hold(ctx)

    def slider_travel(self) -> float:
        """How far (m) the instructed slider has moved from its start."""
        u = self.u
        j = u._LINK_TO_JOINT_ID[u.scene_builder.switch_color[self.variant]]
        return float(np.abs(to_np(u.target_object.get_qpos()).reshape(-1)[j] - to_np(u.init_qpos).reshape(-1)[j]))

    def resume(self, ctx):
        """Takeover from whatever state a student left (dev/gen_takeover.py), fingers closed, short settles.
        In the push corridor behind the instructed slider (within 2 cm of its push line across the stroke, at most
        6 cm above the push height, not over the slider knob): drop onto the push line at the current point of the
        stroke and push on. That is the fix for a student pushing too high (the fingertips slide down the back of
        the knob). Anywhere else: rise to the pre-push height when low near the switch, then pre-push pose,
        contact, push, hold."""
        from mfagent.tasks import toggle_switch as ts
        return self._resume(ctx, ts)

    def _resume(self, ctx, ts):
        est = self.est()
        link = est["blue_link"] if self.variant == "blue" else 1 - est["blue_link"]
        pre, contact, end = ts._push_poses(est, link, ctx.q)
        ctx.gripper = motion.GRIPPER_CLOSED
        tcp = robot.tcp_pose(ctx.q[:7])
        s, lat, a = contact[:3, 0], contact[:3, 1], contact[:3, 2]  # push, across, approach (down)
        r = tcp[:3, 3] - contact[:3, 3]
        ds, dl, da = float(r @ s), float(r @ lat), float(r @ a)
        # The TCP meets the knob 6.7 mm along the stroke from the contact pose (traces: the slider starts moving
        # after 6.7 mm of push), and the knob has moved with the slider since; beyond that the TCP is over the knob.
        behind_knob = ds < 0.011 + self.slider_travel()
        try:
            if abs(dl) < 0.02 and -0.06 < da < 0.004 and -0.04 < ds and behind_knob:
                ctx.notes["resume"] = "corridor"
                p = contact.copy()
                p[:3, 3] = contact[:3, 3] + s * ds
                yield from motion.move_line(ctx, p, v_max=0.02, settle_tol=1e-3)
            else:
                ctx.notes["resume"] = "approach"
                if np.hypot(*(tcp[:2, 3] - contact[:2, 3])) < 0.08 and tcp[2, 3] < pre[2, 3]:
                    up = tcp.copy()
                    up[2, 3] = pre[2, 3]
                    yield from motion.move_line(ctx, up, v_max=0.03, max_settle=3)
                yield from motion.move_line(ctx, pre, v_max=0.05, max_settle=3)
                yield from motion.move_line(ctx, contact, v_max=0.02, settle_tol=1e-3)
            yield from motion.move_line(ctx, end, v_max=0.015, settle_tol=5e-4, max_settle=30)
        except MotionError as exc:
            ctx.notes["error"] = str(exc)
        yield from motion.hold(ctx)

    def contact_xy(self) -> np.ndarray:
        from mfagent.tasks import toggle_switch as ts
        e = self.est()
        link = e["blue_link"] if self.variant == "blue" else 1 - e["blue_link"]
        G = ts.root_pose(e["x"], e["y"], e["theta"]) @ ts._orthonormal(ts.GRASP_LOCAL[link])
        return G[:2, 3]

    def aux(self) -> np.ndarray:
        e = self.est()
        return np.array([e["x"], e["y"], np.cos(e["theta"]), np.sin(e["theta"]), *self.contact_xy()],
                        dtype=np.float32)

    def goals(self, variants) -> np.ndarray:
        """Push contact point of the slider each variant names."""
        from mfagent.tasks import toggle_switch as ts
        e = self.est()
        out = []
        for v in variants:
            link = e["blue_link"] if v == "blue" else 1 - e["blue_link"]
            out.append((ts.root_pose(e["x"], e["y"], e["theta"]) @ ts._orthonormal(ts.GRASP_LOCAL[link]))[:3, 3])
        return np.array(out, dtype=np.float32)

    def noise_scale(self, ctx) -> float:
        """DART noise is cut to 30 % once the fingers are within 6 cm (xy) and 20 cm (z) of the push."""
        p = robot.tcp_pose(ctx.q[:7])[:3, 3]
        near = np.hypot(*(p[:2] - self.contact_xy())) < 0.06 and p[2] < 0.20
        return 0.3 if near else 1.0

    def on_step(self, t, obs, info):
        pass


class _GraspMoveMug:
    AUX_DIM = 6  # mug root x, y, cos, sin of its yaw; xy of the handle pinch point

    def __init__(self, env, variant):
        from dev.simenv import load_cfg
        from eval.eval_grasp_move_mug import _episode_probe
        self.u, self.variant = env.unwrapped, variant
        # The official eval re-arms the move origin at the first real grasp; success depends on it.
        self.on_step, _ = _episode_probe(env, variant, part_name=load_cfg("grasp_move_mug").get("part_name", "handle"))

    def est(self) -> dict:
        T = _pose(self.u.target_object)
        return {"x": float(T[0, 3]), "y": float(T[1, 3]), "yaw": float(np.arctan2(T[1, 0], T[0, 0])), "area": 1e4}

    # The mfagent routine crosses the 7 cm success line at a creep (0.004 rad/step after a 12-step
    # hold): actions of ~0.05 that a student cannot tell from holding still (its T2 failures stall
    # 3-7 cm into the carry). The teacher creeps at CREEP_V instead (None keeps the mfagent value).
    CREEP_V = None
    # DIRECT: after the lift, one straight carry at 0.03 rad/step through the success line (no
    # fast leg / hold / creep): a constant-velocity motion the student can imitate.
    DIRECT = False
    # BLEND (with DIRECT): lift and carry as one timed path, so the arm never comes to rest at the top
    # of the lift (t2_b's remaining failures stall there, creeping at a tenth of the teacher's speed).
    BLEND = False

    def install(self):
        from mfagent.tasks import grasp_move_mug as gm
        gm.find_mug = lambda img, q: self.est()

    def routine(self, ctx):
        """mfagent's grasp_move_mug.routine with the creep speed taken from CREEP_V."""
        from mfagent.tasks import grasp_move_mug as gm
        if self.CREEP_V is None and not self.DIRECT:
            return gm.routine(ctx)
        return self._routine(ctx, gm)

    def _routine(self, ctx, gm):
        direction = gm.DIRECTION.get(ctx.variant, gm.DIRECTION["left"])
        yield from motion.move_joints(ctx, gm.SURVEY_Q, v_max=0.06, settle_tol=1e-3, max_settle=20)
        est = gm.find_mug(ctx.images["hand_camera"], ctx.q)
        for _ in range(2):
            ok = yield from motion.look_from_above(ctx, [est["x"], est["y"]], gm.CLOSE_Z)
            if not ok:
                break
            est = gm.find_mug(ctx.images["hand_camera"], ctx.q)
        G = gm._closer_half_turn(gm.handle_grasp_pose(est["x"], est["y"], est["yaw"]), ctx.q)
        pre = G.copy()
        pre[2, 3] += 0.08
        try:
            yield from motion.move_to_pose(ctx, pre, v_max=0.05)
            yield from motion.move_line(ctx, G, v_max=0.02, settle_tol=1e-3)
            yield from motion.set_gripper(ctx, motion.GRIPPER_CLOSED, steps=10)
            lift_z, move_dist = gm._carry_plan(ctx.q[:7], G, direction) or (gm.LIFT_ZS[0], gm.MOVE_DISTS[0])
            lift = G.copy()
            lift[2, 3] = lift_z
            carry = lift.copy()
            carry[:3, 3] += direction * move_dist
            if self.DIRECT and self.BLEND:
                up = motion.line_waypoints(ctx.q[:7], lift)
                path = up + motion.line_waypoints(up[-1], carry)
                yield from motion.follow(ctx, motion.time_parameterize(ctx.q[:7], path, v_max=0.03), settle_tol=1e-3)
            elif self.DIRECT:
                yield from motion.move_line(ctx, lift, v_max=0.025)
                yield from motion.move_line(ctx, carry, v_max=0.03, settle_tol=1e-3)
            else:
                yield from motion.move_line(ctx, lift, v_max=0.025)
                fast = lift.copy()
                fast[:3, 3] += direction * gm.FAST_DIST
                yield from motion.move_line(ctx, fast, v_max=0.03, settle_tol=1e-3)
                yield from motion.hold(ctx, steps=12)
                yield from motion.move_line(ctx, carry, v_max=self.CREEP_V, settle_tol=1e-3)
        except MotionError as exc:
            ctx.notes["error"] = str(exc)
        yield from motion.hold(ctx)

    def grasp_xy(self) -> np.ndarray:
        from mfagent.tasks import grasp_move_mug as gm
        e = self.est()
        return gm.handle_grasp_pose(e["x"], e["y"], e["yaw"])[:2, 3]

    def aux(self) -> np.ndarray:
        e = self.est()
        return np.array([e["x"], e["y"], np.cos(e["yaw"]), np.sin(e["yaw"]), *self.grasp_xy()], dtype=np.float32)

    def goals(self, variants) -> np.ndarray:
        """Where the carry ends for each direction: 20 cm from the handle pinch point, 40 cm up."""
        from mfagent.tasks import grasp_move_mug as gm
        g = np.array([*self.grasp_xy(), 0.40])
        return np.array([g + 0.20 * gm.DIRECTION[v] for v in variants], dtype=np.float32)

    def noise_scale(self, ctx) -> float:
        """Half noise while carrying (closed gripper); 30 % within 5 cm (xy) and 20 cm (z) of the handle."""
        if ctx.gripper < 0:
            return 0.5
        p = robot.tcp_pose(ctx.q[:7])[:3, 3]
        near = np.hypot(*(p[:2] - self.grasp_xy())) < 0.05 and p[2] < 0.20
        return 0.3 if near else 1.0


class _PutBlocks:
    AUX_DIM = 6  # xy of the special cube, then of the other two in the env's stage order

    def __init__(self, env, variant):
        self.u, self.variant = env.unwrapped, variant
        self.order = [variant] + [c for c in ("red", "blue", "green") if c != variant]

    def cubes(self) -> dict:
        out = {}
        for name in ("red", "green", "blue"):
            T = _pose(getattr(self.u, f"{name}_cube"))
            yaw = float(np.mod(np.arctan2(T[1, 0], T[0, 0]), np.pi / 2))
            out[name] = {"x": float(T[0, 3]), "y": float(T[1, 3]), "yaw": yaw, "iou": 1.0, "area": 1000}
        return out

    def install(self):
        from mfagent.tasks import put_blocks as pb
        pb.detect_cubes = lambda img, q, **kw: self.cubes()

    def aux(self) -> np.ndarray:
        c = self.cubes()
        return np.array([v for name in self.order for v in (c[name]["x"], c[name]["y"])], dtype=np.float32)

    BOXES = (np.array([0.10, -0.20]), np.array([0.10, 0.20]))  # left, right (core/env.py)

    def goals(self, variants) -> np.ndarray:
        """For each variant, the position of the next cube its plan picks (special cube first, then the
        other two in red/blue/green order), i.e. the first one not yet dropped into a box (inside a box
        footprint and below 6 cm: a cube carried over the box is ~9 cm up or more); once all three are
        in, the left box."""
        P = {n: _pose(getattr(self.u, f"{n}_cube"))[:3, 3] for n in ("red", "green", "blue")}

        def boxed(p):
            return any(abs(p[0] - b[0]) < 0.12 and abs(p[1] - b[1]) < 0.12 for b in self.BOXES) and p[2] < 0.06

        out = []
        for v in variants:
            order = [v] + [c for c in ("red", "blue", "green") if c != v]
            nxt = next((P[c] for c in order if not boxed(P[c])), np.array([*self.BOXES[0], 0.05]))
            out.append(nxt)
        return np.array(out, dtype=np.float32)

    def noise_scale(self, ctx) -> float:
        """Half noise while carrying; 30 % within 4 cm (xy) and 15 cm (z) of any cube."""
        if ctx.gripper < 0:
            return 0.5
        p = robot.tcp_pose(ctx.q[:7])[:3, 3]
        near = any(np.hypot(p[0] - c["x"], p[1] - c["y"]) < 0.04 for c in self.cubes().values()) and p[2] < 0.15
        return 0.3 if near else 1.0

    def on_step(self, t, obs, info):
        pass


class _GraspPart:
    AUX_DIM = 7  # bottle root x, y; horizontal axis direction (0 when upright); lying flag; target part xy
    CAP_OFF, BODY_OFF = 0.140, 0.012  # part centres along the bottle axis from its root

    # The t1_b student's cap failures close the gripper beside or above the cap, dash up empty and
    # then drift: the demos never show a missed grasp. In MISS_P of the upright-cap episodes the
    # teacher first misses on purpose (masked: no labels), either closing MISS_HIGH (m) above the
    # grasp or MISS_SIDE (m) off along the hand's x axis, dashes up as after any top-down close, then
    # opens and runs the whole routine again from the survey (labelled): the recovery to learn.
    MISS_P = 0.0
    MISS_HIGH = (0.02, 0.05)
    MISS_SIDE = (0.025, 0.04)

    def __init__(self, env, variant):
        self.u, self.variant = env.unwrapped, variant
        self.rng = np.random.default_rng([np.random.randint(2**31), sum(map(ord, variant))])

    def routine(self, ctx):
        from mfagent.tasks import grasp_part as gp
        if self.variant == "body" and self.BODY_DASH:
            return self._body_dash_routine(ctx, gp)
        if self.variant != "cap" or not self.rng.random() < self.MISS_P:
            return gp.routine(ctx)
        return self._miss_then_routine(ctx, gp)

    # BODY_DASH: after an upright side grasp, close and rise at full joint speed with the bottle (as mfagent
    # does after top-down grasps) instead of closing in place. The episode ends 5 steps into the hold, so a
    # body grasp otherwise leaves the hand low, where the next episode's re-dropped bottle hits it: on seeds
    # 2,710,001-060 most of t1_b's body failures were bottles lying or thrown off the table.
    BODY_DASH = False

    def _body_dash_routine(self, ctx, gp):
        """grasp_part.routine for the body, with close_and_dash after side grasps too."""
        xy = yield from gp.locate(ctx)
        if xy is None:
            ctx.notes["error"] = "bottle not found"
            yield from motion.hold(ctx)
            return
        shape = ctx.notes.get("shape") or {}
        try:
            if shape.get("lying"):
                ax = np.array(shape["axis"])
                grasp, pre = gp.lying_grasp(np.array(shape["body_xy"]), float(np.arctan2(ax[0], -ax[1])), ctx.q[:7])
                if grasp is None:
                    raise MotionError("lying bottle out of reach")
                yield from motion.move_to_pose(ctx, pre, v_max=0.05)
                yield from motion.move_line(ctx, grasp, v_max=0.02, settle_tol=1e-3)
            else:
                choice = gp.choose_side_approach(xy, ctx.q[:7])
                if choice is None:
                    raise MotionError("no reachable side approach")
                a, sign, q_high = choice
                yield from motion.move_joints(ctx, q_high, v_max=0.05, settle_tol=2e-3)
                yield from motion.move_line(ctx, gp.side_grasp_pose(xy - 0.13 * a, gp.BODY_TCP_Z, a, sign), v_max=0.04)
                yield from motion.move_line(ctx, gp.side_grasp_pose(xy - gp.BODY_BACKOFF * a, gp.BODY_TCP_Z, a, sign),
                                            v_max=0.02, settle_tol=1e-3)
            yield from gp.close_and_dash(ctx)
        except MotionError as exc:
            ctx.notes["error"] = str(exc)
        yield from motion.hold(ctx)

    def _miss_then_routine(self, ctx, gp):
        xy = yield from gp.locate(ctx)
        if xy is None:
            yield from motion.hold(ctx)
            return
        shape = ctx.notes.get("shape") or {}
        grasp = None if shape.get("lying") else gp.cap_grasp_pose(xy, ctx.q[:7])
        if grasp is not None:
            miss = grasp.copy()
            if self.rng.random() < 0.5:
                miss[:3, 3] -= grasp[:3, 2] * self.rng.uniform(*self.MISS_HIGH)  # back along the approach
            else:
                miss[:3, 3] += grasp[:3, 0] * self.rng.choice([-1.0, 1.0]) * self.rng.uniform(*self.MISS_SIDE)
            pre = miss.copy()
            pre[:3, 3] -= miss[:3, 2] * 0.08
            ctx.notes["mask"] = True  # unlabelled, so it may as well be quick
            try:
                yield from motion.move_to_pose(ctx, pre, v_max=0.08)
                yield from motion.move_line(ctx, miss, v_max=0.04, settle_tol=2e-3, max_settle=5)
                yield from gp.close_and_dash(ctx)
            except MotionError:
                pass
            ctx.notes["mask"] = False
            ctx.notes["missed"] = True
            # Recover within the 300 steps: open, straight back to the survey pose, one close look, grasp.
            ctx.gripper = motion.GRIPPER_OPEN
            yield from self._cap(motion.move_joints(ctx, gp.SURVEY_Q, v_max=0.1, settle_tol=2e-3, max_settle=10))
            est = gp.find_bottle(ctx.images["hand_camera"], ctx.q)
            yield from self._cap(motion.look_from_above(ctx, [est["x"], est["y"]], gp.CLOSE_Z))
            est = gp.find_bottle(ctx.images["hand_camera"], ctx.q)
            ctx.notes["shape"] = gp.bottle_shape(ctx.images["hand_camera"], ctx.q)
            xy = np.array([est["x"], est["y"]])
        yield from self._cap_grasp(ctx, gp, xy)
        yield from motion.hold(ctx)

    @staticmethod
    def _cap_grasp(ctx, gp, xy, from_here: bool = False):
        """grasp_part.routine's cap grasp after locate (lying bottles: pinch the cap from above).
        from_here: when the TCP is already on the approach line between the pre-grasp and the grasp
        (within 1.5 cm of it), go straight down instead of backing up to the pre-grasp first."""
        shape = ctx.notes.get("shape") or {}
        try:
            if shape.get("lying"):
                ax = np.array(shape["axis"])
                grasp, pre = gp.lying_grasp(np.array(shape["cap_xy"]), float(np.arctan2(ax[0], -ax[1])), ctx.q[:7])
                if grasp is None:
                    raise MotionError("lying bottle out of reach")
            else:
                grasp = gp.cap_grasp_pose(xy, ctx.q[:7])
                if grasp is None:
                    raise MotionError("no reachable cap grasp")
                pre = grasp.copy()
                pre[:3, 3] -= grasp[:3, 2] * 0.08
            p = robot.tcp_pose(ctx.q[:7])[:3, 3]
            d = grasp[:3, 3] - p
            along = float(d @ grasp[:3, 2])  # distance still to go along the approach axis
            on_line = 0.0 <= along <= 0.085 and np.linalg.norm(d - along * grasp[:3, 2]) < 0.015
            if not (from_here and on_line):
                yield from motion.move_to_pose(ctx, pre, v_max=0.05)
            yield from motion.move_line(ctx, grasp, v_max=0.02, settle_tol=1e-3)
            if gp.DASH:
                yield from gp.close_and_dash(ctx)
            else:
                yield from motion.set_gripper(ctx, motion.GRIPPER_CLOSED, steps=10)
        except MotionError as exc:
            ctx.notes["error"] = str(exc)

    def resume(self, ctx):
        """Takeover from a student's state (dev/gen_takeover.py). Holding something: hold. Cap: fingers
        closed on nothing -> open, survey, one close look, grasp; open -> grasp from here. Body: open,
        then the routine's side grasp from here (straight in when already on the approach line)."""
        from mfagent.tasks import grasp_part as gp
        return self._resume(ctx, gp)

    def _resume(self, ctx, gp):
        holding = len(self.u._grasped_links()) > 0  # the bottle is an articulation (cap and body links)
        if holding:
            ctx.gripper = motion.GRIPPER_CLOSED
            yield from motion.hold(ctx)
            return
        if self.variant == "body":
            yield from self._resume_body(ctx, gp)
            return
        if ctx.finger_width < 0.07:  # closed or closing on nothing
            ctx.gripper = motion.GRIPPER_OPEN
            yield from self._cap(motion.move_joints(ctx, gp.SURVEY_Q, v_max=0.1, settle_tol=2e-3, max_settle=10))
            yield from self._cap(motion.look_from_above(ctx, self.target_xy(), gp.CLOSE_Z))
            from_here = False
        else:
            ctx.gripper = motion.GRIPPER_OPEN
            from_here = True
        ctx.notes["shape"] = self.bottle_shape(None, ctx.q)
        T, _, _ = self.state()
        yield from self._cap_grasp(ctx, gp, T[:2, 3].copy(), from_here=from_here)
        yield from motion.hold(ctx)

    def _resume_body(self, ctx, gp):
        T, lying, ax = self.state()
        if ctx.finger_width < 0.07:
            yield from motion.set_gripper(ctx, motion.GRIPPER_OPEN, steps=6)
        ctx.gripper = motion.GRIPPER_OPEN
        try:
            if lying:
                target = (T[:3, 3] + T[:3, 2] * self.BODY_OFF)[:2]
                grasp, pre = gp.lying_grasp(target, float(np.arctan2(ax[0], -ax[1])), ctx.q[:7])
                if grasp is None:
                    raise MotionError("lying bottle out of reach")
                yield from motion.move_to_pose(ctx, pre, v_max=0.05)
                yield from motion.move_line(ctx, grasp, v_max=0.02, settle_tol=1e-3)
            else:
                xy = T[:2, 3].copy()
                choice = gp.choose_side_approach(xy, ctx.q[:7])
                if choice is None:
                    raise MotionError("no reachable side approach")
                a, sign, q_high = choice
                grasp = gp.side_grasp_pose(xy - gp.BODY_BACKOFF * a, gp.BODY_TCP_Z, a, sign)
                pre = gp.side_grasp_pose(xy - 0.13 * a, gp.BODY_TCP_Z, a, sign)
                p = robot.tcp_pose(ctx.q[:7])[:3, 3]
                d = grasp[:3, 3] - p
                along = float(d @ grasp[:3, 2])
                if not (0.0 <= along <= 0.135 and np.linalg.norm(d - along * grasp[:3, 2]) < 0.02):
                    if p[2] < gp.SIDE_ENTRY_Z - 0.02 and np.hypot(*(p[:2] - xy)) < 0.15:
                        up = robot.tcp_pose(ctx.q[:7]).copy()  # low beside the bottle: rise clear of it first
                        up[2, 3] = gp.SIDE_ENTRY_Z
                        yield from motion.move_line(ctx, up, v_max=0.04)
                    yield from motion.move_joints(ctx, q_high, v_max=0.05, settle_tol=2e-3)
                    yield from motion.move_line(ctx, pre, v_max=0.04)
                yield from motion.move_line(ctx, grasp, v_max=0.02, settle_tol=1e-3)
            if self.BODY_DASH or lying:
                yield from gp.close_and_dash(ctx)
            else:
                yield from motion.set_gripper(ctx, motion.GRIPPER_CLOSED, steps=10)
        except MotionError as exc:
            ctx.notes["error"] = str(exc)
        yield from motion.hold(ctx)

    def state(self):
        T = _pose(self.u.target_object)
        lying = T[2, 2] < 0.866
        ax = T[:3, 2].copy()
        ax[2] = 0.0
        ax = ax / (np.linalg.norm(ax) + 1e-12) if lying else np.zeros(3)
        return T, lying, ax

    def find_bottle(self, img, q):
        T, _, _ = self.state()
        return {"x": float(T[0, 3]), "y": float(T[1, 3]), "area": 1000, "uv": np.array([256.0, 256.0])}

    def bottle_shape(self, img, q):
        T, lying, ax = self.state()
        out = {"elongation": 4.0 if lying else 1.0, "lying": bool(lying)}
        if lying:
            out["axis"] = ax[:2].tolist()
            out["cap_xy"] = (T[:3, 3] + T[:3, 2] * self.CAP_OFF)[:2].tolist()
            out["body_xy"] = (T[:3, 3] + T[:3, 2] * self.BODY_OFF)[:2].tolist()
        return out

    # mfagent picks the side-grasp approach as the first feasible of 0, +-30, +-60, +-90 deg (and either
    # closing sign): over the workspace that is +30 (~70 %), +60 (~24 %), -30 or a flipped hand (~6 %),
    # modes a regression student averages into a stall at the close look. SMOOTH_SIDE instead takes
    # the feasible direction closest to +30 deg on a 5 deg grid, closing sign +1 first, so the choice
    # changes in small steps across the workspace.
    SMOOTH_SIDE = False

    # LOCATE_CAP (steps, default None = the l1-l6 teacher): cap the settles of the survey pose and the close looks
    # (gp.locate, and the miss recovery's survey and look). t1_g, fresh seeds 2,795,001-040: 6/120 light-cap
    # episodes hovered at the close-look height (TCP z 0.36-0.39, the look pose is 0.383) and never approached,
    # the same still-phase stall as t3_c at its close view. Grasp-approach settles stay complete.
    LOCATE_CAP = None

    def _cap(self, gen):
        return _capped(gen, int(self.LOCATE_CAP)) if self.LOCATE_CAP is not None else gen

    def install(self):
        from mfagent.tasks import grasp_part as gp
        gp.find_bottle = self.find_bottle
        gp.bottle_shape = self.bottle_shape
        if not hasattr(gp, "_locate_mfagent"):
            gp._locate_mfagent = gp.locate
        gp.locate = (lambda ctx: self._cap(gp._locate_mfagent(ctx))) if self.LOCATE_CAP is not None \
            else gp._locate_mfagent
        if self.SMOOTH_SIDE:
            if not hasattr(gp, "_choose_side_approach_mfagent"):
                gp._choose_side_approach_mfagent = gp.choose_side_approach
            gp.choose_side_approach = self.smooth_side_approach
        if self.EXT_LYING:
            if not hasattr(gp, "_lying_grasp_mfagent"):
                gp._lying_grasp_mfagent = gp.lying_grasp
            gp.lying_grasp = self.extended_lying_grasp

    # A knocked-over bottle (the env's re-drop) is out of reach for mfagent's lying grasp in ~5 % of
    # body episodes. EXT_LYING also tries tilts up to 65 deg either way about the closing axis, small
    # tilts about the bottle axis and both hand flips, and requires the descent line to be followable
    # (offline: all realistic lying poses become reachable).
    EXT_LYING = False

    @staticmethod
    def extended_lying_grasp(target_xy, yaw, q):
        from mfagent.tasks import grasp_part as gp
        G, P = gp._lying_grasp_mfagent(target_xy, yaw, q)
        if G is not None:
            return G, P
        cands, seen = [], set()
        for tilt in np.radians([0, 20, 35, 50, 65]):
            for flip in (0.0, np.pi):
                G0 = robot.top_down_pose([target_xy[0], target_xy[1], gp.LYING_TCP_Z], yaw + flip)
                for axis_idx in (1, 0):
                    if axis_idx == 0 and tilt > np.radians(35):
                        continue
                    for sgn in ((1.0,) if tilt == 0 else (1.0, -1.0)):
                        M = G0.copy()
                        M[:3, :3] = robot.rot_exp(G0[:3, axis_idx] * sgn * tilt) @ G0[:3, :3]
                        key = tuple(np.round(M[:3, :3], 3).ravel())
                        if key not in seen:
                            seen.add(key)
                            cands.append(M)
        for G in cands:
            P = G.copy()
            P[:3, 3] -= G[:3, 2] * 0.08
            qp = motion.ik_multi(P, q, n_random=6)
            if qp is None:
                continue
            try:
                motion.line_waypoints(qp, G)
            except MotionError:
                continue
            return G, P
        return None, None

    @staticmethod
    def smooth_side_approach(xy, q):
        from mfagent.tasks import grasp_part as gp
        base = np.arctan2(*(xy - gp.BASE_XY)[::-1])
        order = sorted(range(-60, 95, 5), key=lambda d: (abs(d - 30), d))
        for sign in (1.0, -1.0):
            for dd in order:
                a = np.array([np.cos(base + np.radians(dd)), np.sin(base + np.radians(dd))])
                pre = gp.side_grasp_pose(xy - 0.13 * a, gp.BODY_TCP_Z, a, sign)
                q_pre = motion.ik_multi(pre, q, n_random=8)
                if q_pre is None:
                    continue
                q_g, ok = robot.ik_tcp(gp.side_grasp_pose(xy - gp.BODY_BACKOFF * a, gp.BODY_TCP_Z, a, sign), q_pre)
                if not ok or np.abs(q_g - q_pre).max() > 0.6:
                    continue
                q_high, ok = robot.ik_tcp(gp.side_grasp_pose(xy - 0.13 * a, gp.SIDE_ENTRY_Z, a, sign), q_pre)
                if not (ok and np.all(q_high > robot.Q_LOWER + 0.03) and np.all(q_high < robot.Q_UPPER - 0.03)):
                    continue
                try:
                    wps = motion.line_waypoints(q_high, gp.side_grasp_pose(xy - 0.13 * a, gp.BODY_TCP_Z, a, sign))
                    motion.line_waypoints(wps[-1], gp.side_grasp_pose(xy - gp.BODY_BACKOFF * a, gp.BODY_TCP_Z, a, sign))
                except MotionError:
                    continue
                return a, sign, q_high
        return gp._choose_side_approach_mfagent(xy, q)

    def target_xy(self) -> np.ndarray:
        T, lying, _ = self.state()
        if not lying:
            return T[:2, 3].copy()
        return (T[:3, 3] + T[:3, 2] * (self.CAP_OFF if self.variant == "cap" else self.BODY_OFF))[:2]

    def aux(self) -> np.ndarray:
        T, lying, ax = self.state()
        return np.array([T[0, 3], T[1, 3], ax[0], ax[1], float(lying), *self.target_xy()], dtype=np.float32)

    def goals(self, variants) -> np.ndarray:
        """Centre of the part each variant names (cap 14 cm, body 1.2 cm along the axis from the root)."""
        T, _, _ = self.state()
        return np.array([T[:3, 3] + T[:3, 2] * (self.CAP_OFF if v == "cap" else self.BODY_OFF) for v in variants],
                        dtype=np.float32)

    def noise_scale(self, ctx) -> float:
        """30 % within 6 cm (xy) and 30 cm (z) of the bottle; none once the gripper closes."""
        if ctx.gripper < 0:
            return 0.0
        p = robot.tcp_pose(ctx.q[:7])[:3, 3]
        near = np.hypot(*(p[:2] - self.target_xy())) < 0.06 and p[2] < 0.30
        return 0.3 if near else 1.0

    def on_step(self, t, obs, info):
        pass


class _InsertLetter:
    # instructed peg x, y, cos, sin of its yaw; xy of the other three pegs (C, o, R, L order). V2 adds the
    # instructed peg's origin in the TCP frame (3) and its xy offset from its slot (2).
    AUX_DIM = 10

    # V2 teacher (off: the mfagent routine with ground-truth perception). The mfagent routine grasps
    # exactly where it planned and inserts with the squeeze model's hold pose, so its demos never show a
    # peg held off-centre; a student whose grasp lands a few mm along the stroke then carries the peg to
    # the nominal pose and presses it onto the board (t5_a: most failures, 6-13 mm off). V2:
    # - moves the grasp line up to GRASP_JITTER along the free direction (same squeeze span), so the
    #   in-hand offset varies and has to be read from the image;
    # - seats from the true in-hand pose, re-measured before every descent (no spiral search: a timed
    #   spiral cannot be imitated without memory);
    # - in MISALIGN_P of the episodes, offsets the first descent by MISALIGN (m), with those steps
    #   marked as masked (ctx.notes["mask"]: no training labels), then recovers: lift the peg
    #   REALIGN_GAP above the board, align from the true pose, press again (up to 4 times).
    V2 = False
    GRASP_JITTER = 0.006
    MISALIGN_P = 0.5
    MISALIGN = (0.003, 0.010)
    REALIGN_GAP = 0.006
    # DART noise scale while a held peg is lowered (z < 11 cm). 0 keeps the old rule (no noise there): then the
    # student never sees a descent drifting off the slot, and t5_b drifted 1-1.5 cm on the way down and stalled
    # 9 mm above the board. The seating loop re-measures the true pose, so drift is recoverable.
    DESCENT_NOISE = 0.0
    SEATED_Z = 0.0115  # peg origin below this: in the hole (on the board top it is at 0.014; a press goes 4 mm)
    # Let go the first step the peg is home (origin below HOME_Z, over its slot) instead of after the push's
    # settle. The settle held a home peg still with the gripper closed for 10-15 steps before opening; a
    # policy without memory cannot time that, and students learned to hold on forever (t5_e2, t5_d2 on o).
    QUICK_RELEASE = False
    HOME_Z = 0.002
    # The same holds stall students elsewhere (l4 traces, seeds 2,740,001-040): the settles at the grasp, over the
    # board and in the seating loop last up to 13-25 steps, longer than the 8 actions a student executes per
    # chunk, so a student that reaches such a still state predicts "hold" again and never leaves it.
    # SETTLE_CAP (steps) caps every settle of this teacher; it also ends a grasp's close once the fingers stop.
    SETTLE_CAP = None
    # Takeover with the peg not held and the TCP below 15 cm within RESUME_NEAR (m, xy) of the peg: grasp from the
    # current height (pre-shaped fingers, no rise to 15 cm). The old resume opened fully and rose first; from a
    # student hovering over its grasp that taught "rise and retry", the opposite of the plain demos' "go down and
    # close", and students looped between the two (l4: C 2740005, R 2740005).
    RESUME_NEAR = 0.0
    # The first RESUME_NEAR still rose to 2 cm over the grasp before going down: from the students stalled at the
    # grasp height (95 % of the low near-resumes in round 5) that taught "rise 2 cm" against the plain demos'
    # "go down", and t5_f stalled where the two balance (TCP 1.5-2 cm over the grasp, 48 of its 83 fresh-seed
    # failures). NEAR_NO_RISE aligns at the current height instead (see _grasp_insert).
    NEAR_NO_RISE = False
    # RELEASE_RAMP (steps, default 0 = the instant open): open the gripper over this many steps with the arm held.
    # Students let go one step early, with the peg still 2-3 mm up and the arm moving down (the teacher's last push
    # runs at ~3 mm per step, so the release window is one step). An instant open then kicks the o 10-15 deg in that
    # step (t5_j traces 2,793,006/009, 2,796,011: 1.7-2.8 deg held -> 12.4-15.1 deg), and the pocket's 4 mm clearance
    # lets it stay there. A ramp fades the squeeze so the peg slides down between the pads instead.
    RELEASE_RAMP = 0

    def __init__(self, env, variant):
        self.u, self.variant = env.unwrapped, variant
        self.ctx = None
        # The global RNG is seeded per episode; the variant is mixed in so that the letters of one scene
        # get different jitters and misalignments.
        self.rng = np.random.default_rng([np.random.randint(2**31), sum(map(ord, variant))])

    def peg(self, letter) -> dict:
        T = _pose(self.u.pegs[letter])
        return {"x": float(T[0, 3]), "y": float(T[1, 3]), "yaw": float(np.arctan2(T[1, 0], T[0, 0])),
                "iou": 1.0, "area": 1000}

    def install(self):
        from mfagent.tasks import insert_letter as il
        il.fit_all = lambda img, q: {L: self.peg(L) for L in self.u.pegs}
        il.fit_letter = lambda img, q, letter, **kw: self.peg(letter)
        if self.SETTLE_CAP is not None:
            # mfagent's motion primitives (and il.locate) look follow() up at call time; cap it for this process
            # (generators run one task each).
            orig = getattr(motion.follow, "uncapped", motion.follow)
            cap = int(self.SETTLE_CAP)

            def follow(ctx, refs, settle_tol=2e-3, max_settle=15):
                return orig(ctx, refs, settle_tol=settle_tol, max_settle=min(max_settle, cap))
            follow.uncapped = orig
            motion.follow = follow

    def aux(self) -> np.ndarray:
        e = self.peg(self.variant)
        others = [v for L in ("C", "o", "R", "L") if L != self.variant for v in (self.peg(L)["x"], self.peg(L)["y"])]
        out = [e["x"], e["y"], np.cos(e["yaw"]), np.sin(e["yaw"]), *others]
        if self.V2:
            from mfagent import letters
            T = _pose(self.u.pegs[self.variant])
            rel = np.linalg.inv(robot.tcp_pose(self.ctx.q[:7])) @ T
            out += [*rel[:3, 3], *(T[:2, 3] - letters.slot_pose(self.variant)[:2])]
        return np.array(out, dtype=np.float32)

    def routine(self, ctx):
        from mfagent.tasks import insert_letter as il
        self.ctx = ctx
        if not self.V2:
            return il.routine(ctx)
        return self._routine(ctx, il)

    def _jitter(self, letter, axis, c):
        """The grasp line moved along the free direction, keeping the squeezed span (also 3 mm either side)."""
        from mfagent import letters

        def span(cc):
            try:
                return letters.line_span(letter, axis, cc)
            except ValueError:  # the line misses the glyph
                return None
        ref = span(c)
        for _ in range(30):
            c2 = c + self.rng.uniform(-self.GRASP_JITTER, self.GRASP_JITTER)
            if all(span(c2 + d) == ref for d in (-0.003, 0.0, 0.003)):
                return c2
        return c

    def _seat_pose(self, ctx, letter) -> np.ndarray:
        """TCP pose that puts the peg origin on its slot centre (z = 0) at the goal yaw, from the true
        in-hand pose."""
        import math

        from mfagent import letters
        T_peg = _pose(self.u.pegs[letter])
        T_hold = np.linalg.inv(robot.tcp_pose(ctx.q[:7])) @ T_peg
        slot = letters.slot_pose(letter)
        goal_yaw, period = slot[2], letters.SYMMETRY[letter]
        if period < 2 * math.pi - 1e-6:
            yaw_now = math.atan2(T_peg[1, 0], T_peg[0, 0])
            goal_yaw += round((yaw_now - goal_yaw) / period) * period
        G = np.eye(4)
        G[:3, :3] = [[math.cos(goal_yaw), -math.sin(goal_yaw), 0.0], [math.sin(goal_yaw), math.cos(goal_yaw), 0.0],
                     [0.0, 0.0, 1.0]]
        G[:3, 3] = [slot[0], slot[1], 0.0]
        return G @ np.linalg.inv(T_hold)

    def _routine(self, ctx, il):
        from mfagent import letters
        letter = ctx.variant if ctx.variant in letters.GLYPHS else "C"
        est = yield from il.locate(ctx, letter)
        ctx.notes["peg"] = est
        try:
            yield from self._grasp_insert(ctx, il, letter, est)
        except MotionError as exc:
            ctx.notes["error"] = str(exc)
        ctx.notes["mask"] = False
        yield from motion.hold(ctx)

    def _grasp_insert(self, ctx, il, letter, est, from_here: bool = False):
        """from_here: the first approach starts at the current height (at least 2 cm, at most the usual 9 cm
        above the grasp), rising straight up first when the TCP is lower than that."""
        held, G = False, None
        for axis, c, opening in il.choose_grasps(letter, est, ctx.notes.get("others", {})):
            c = self._jitter(letter, axis, c)
            for _ in range(2):
                G = il._reachable_grasp(il.grasp_pose_for(letter, est["x"], est["y"], est["yaw"], axis, c), ctx.q[:7])
                if G is None:
                    raise MotionError("no reachable grasp")
                pre = il._pregrasp(G)
                ctx.gripper = il.gripper_action_for_opening(opening)
                if from_here and self.NEAR_NO_RISE:
                    # Align at the current height and go down: rise (to 6 mm over the peg top) only when the
                    # fingers are already down at the peg and more than 4 mm / 8 deg off the grasp.
                    from mfagent import letters
                    from_here = False
                    tcp = robot.tcp_pose(ctx.q[:7])
                    pre = il._pregrasp(G, dist=float(np.clip(tcp[2, 3] - G[2, 3], 0.0, 0.09)))
                    top = _pose(self.u.pegs[letter])[2, 3] + letters.PEG_H
                    off = np.hypot(*(tcp[:2, 3] - G[:2, 3]))
                    turn = np.linalg.norm(robot.rot_log(tcp[:3, :3].T @ G[:3, :3]))
                    if tcp[2, 3] < top + 0.003 and (off > 0.004 or turn > np.radians(8)):
                        up = tcp.copy()
                        up[2, 3] = top + 0.006
                        yield from motion.move_line(ctx, up, v_max=0.03)
                        pre = il._pregrasp(G, dist=float(np.clip(up[2, 3] - G[2, 3], 0.0, 0.09)))
                elif from_here:
                    from_here = False
                    tcp = robot.tcp_pose(ctx.q[:7])
                    pre = il._pregrasp(G, dist=float(np.clip(tcp[2, 3] - G[2, 3], 0.02, 0.09)))
                    if tcp[2, 3] < pre[2, 3] - 0.003:
                        up = tcp.copy()
                        up[2, 3] = pre[2, 3]
                        yield from motion.move_line(ctx, up, v_max=0.03)
                yield from motion.move_to_pose(ctx, pre, v_max=0.05)
                yield from motion.move_to_pose(ctx, G, v_max=0.02, settle_tol=5e-4)
                yield from self._close(ctx)
                if ctx.finger_width > il.MIN_HOLD_WIDTH:
                    held = True
                    break
                yield from motion.set_gripper(ctx, il.gripper_action_for_opening(opening), steps=6)
                yield from motion.move_line(ctx, pre, v_max=0.05)
                est = yield from il.locate(ctx, letter)
            if held:
                break
        if not held:
            raise MotionError("could not grasp the peg")
        lift = G.copy()
        lift[2, 3] = il.LIFT_Z
        yield from motion.move_line(ctx, lift, v_max=0.03)
        yield from self._insert(ctx, letter, il)

    def _insert(self, ctx, letter, il, misalign: bool = True):
        from mfagent import letters
        T = self._seat_pose(ctx, letter)
        above = T.copy()
        above[2, 3] = il.LIFT_Z
        yield from motion.move_to_pose(ctx, above, v_max=0.04, settle_tol=1e-3)
        off = np.zeros(2)
        if misalign and self.rng.random() < self.MISALIGN_P:
            a = self.rng.uniform(0.0, 2 * np.pi)
            off = self.rng.uniform(*self.MISALIGN) * np.array([np.cos(a), np.sin(a)])
        T = self._seat_pose(ctx, letter)
        z_contact = T[2, 3] + letters.BOARD_H
        hover, press = T.copy(), T.copy()
        hover[2, 3], press[2, 3] = z_contact + il.HOVER_GAP, z_contact - il.PRESS_DEPTH
        hover[:2, 3] += off
        press[:2, 3] += off
        ctx.notes["mask"] = bool(np.any(off))
        yield from motion.move_line(ctx, hover, v_max=0.02, settle_tol=5e-4, max_settle=25)
        yield from motion.move_line(ctx, press, v_max=0.01, settle_tol=5e-4, max_settle=10)
        ctx.notes["mask"] = False
        yield from self._seat(ctx, letter, il)

    def _seat(self, ctx, letter, il):
        """From a held peg pressed on (or just above) the board: lift it REALIGN_GAP clear, align from the
        true pose and press, until it is in the hole (4 tries); push it home, release, back off."""
        from mfagent import letters
        for _ in range(4):
            if _pose(self.u.pegs[letter])[2, 3] < self.SEATED_Z:
                break
            T = self._seat_pose(ctx, letter)
            z_contact = T[2, 3] + letters.BOARD_H
            up = robot.tcp_pose(ctx.q[:7]).copy()
            up[2, 3] = z_contact + self.REALIGN_GAP
            yield from motion.move_line(ctx, up, v_max=0.01, settle_tol=5e-4, max_settle=10)
            T = self._seat_pose(ctx, letter)
            align, press = T.copy(), T.copy()
            align[2, 3], press[2, 3] = z_contact + self.REALIGN_GAP, z_contact - il.PRESS_DEPTH
            yield from motion.move_line(ctx, align, v_max=0.01, settle_tol=5e-4, max_settle=15)
            yield from motion.move_line(ctx, press, v_max=0.01, settle_tol=5e-4, max_settle=10)
        ctx.notes["seated"] = bool(_pose(self.u.pegs[letter])[2, 3] < self.SEATED_Z)
        # Push the peg all the way down (its origin must end below 3 mm), then let go.
        T = self._seat_pose(ctx, letter)
        home = robot.tcp_pose(ctx.q[:7]).copy()
        home[2, 3] = T[2, 3] - 0.002
        home[:3, :3] = T[:3, :3]
        try:
            for a in motion.move_line(ctx, home, v_max=0.01, settle_tol=5e-4, max_settle=15):
                if self.QUICK_RELEASE and self._home(letter):
                    break
                yield a
        except MotionError:
            pass
        if self.RELEASE_RAMP:
            q_hold = ctx.q[:7].copy()
            n = int(self.RELEASE_RAMP)
            for k in range(1, n + 1):
                ctx.gripper = motion.GRIPPER_CLOSED + (motion.GRIPPER_OPEN - motion.GRIPPER_CLOSED) * k / n
                yield ctx.arm_action(q_hold)
        yield from motion.set_gripper(ctx, motion.GRIPPER_OPEN, steps=8)
        up = robot.tcp_pose(ctx.q[:7]).copy()
        up[2, 3] = il.LIFT_Z
        yield from motion.move_line(ctx, up, v_max=0.02)

    def _close(self, ctx):
        """Close the gripper holding the arm: 10 steps, or with SETTLE_CAP until the fingers have stopped for
        2 steps (they stop on the stroke after 3-4)."""
        if self.SETTLE_CAP is None:
            yield from motion.set_gripper(ctx, motion.GRIPPER_CLOSED, steps=10)
            return
        ctx.gripper = motion.GRIPPER_CLOSED
        q_hold = ctx.q[:7].copy()
        prev, still = ctx.finger_width, 0
        for _ in range(10):
            yield ctx.arm_action(q_hold)
            w = ctx.finger_width
            still = still + 1 if abs(w - prev) < 2e-4 else 0
            prev = w
            if still >= 2:
                break

    def _home(self, letter) -> bool:
        """The peg is all the way down in its own slot."""
        from mfagent import letters
        p = _pose(self.u.pegs[letter])[:3, 3]
        return bool(p[2] < self.HOME_Z and np.hypot(*(p[:2] - letters.slot_pose(letter)[:2])) < 0.008)

    def resume(self, ctx):
        """Takeover from whatever state a student left (dev/gen_takeover.py). With the peg held: seat it,
        re-aligning from close by when it is already low over its slot. Otherwise: open, rise to 15 cm
        and grasp it again (true pose, no survey), then insert."""
        from mfagent.tasks import insert_letter as il
        self.ctx = ctx
        return self._resume(ctx, il)

    def _resume(self, ctx, il):
        from mfagent import letters
        letter = self.variant
        held = bool(np.asarray(to_np(self.u.agent.is_grasping(self.u.pegs[letter]))).reshape(-1)[0])
        try:
            if held:
                ctx.gripper = motion.GRIPPER_CLOSED
                T = self._seat_pose(ctx, letter)
                tcp = robot.tcp_pose(ctx.q[:7])
                if tcp[2, 3] < T[2, 3] + letters.BOARD_H + 0.03 and np.hypot(*(tcp[:2, 3] - T[:2, 3])) < 0.03:
                    yield from self._seat(ctx, letter, il)
                else:
                    yield from self._insert(ctx, letter, il, misalign=False)
            else:
                ctx.notes["others"] = {L: self.peg(L) for L in self.u.pegs if L != letter}
                tcp = robot.tcp_pose(ctx.q[:7])
                est = self.peg(letter)
                if tcp[2, 3] < 0.15 and np.hypot(tcp[0, 3] - est["x"], tcp[1, 3] - est["y"]) < self.RESUME_NEAR:
                    ctx.notes["resume"] = "near"
                    yield from self._grasp_insert(ctx, il, letter, est, from_here=True)
                else:
                    ctx.gripper = motion.GRIPPER_OPEN
                    up = tcp.copy()
                    if up[2, 3] < 0.15:
                        up[2, 3] = 0.15
                        yield from motion.move_line(ctx, up, v_max=0.03)
                    yield from self._grasp_insert(ctx, il, letter, est)
        except MotionError as exc:
            ctx.notes["error"] = str(exc)
        ctx.notes["mask"] = False
        yield from motion.hold(ctx)

    def goals(self, variants) -> np.ndarray:
        """Position of the peg each variant names."""
        return np.array([_pose(self.u.pegs[v])[:3, 3] for v in variants], dtype=np.float32)

    def noise_scale(self, ctx) -> float:
        """DESCENT_NOISE while lowering a held peg (z < 11 cm); 30 % while carrying; 20 % below 10 cm."""
        z = robot.tcp_pose(ctx.q[:7])[2, 3]
        if ctx.gripper < -0.5:
            return self.DESCENT_NOISE if z < 0.11 else 0.3
        return 0.2 if z < 0.10 else 1.0

    def on_step(self, t, obs, info):
        pass


HOOKS = {"toggle_switch_table": _ToggleSwitch, "grasp_move_mug": _GraspMoveMug,
         "put_blocks_into_boxes": _PutBlocks, "grasp_part": _GraspPart, "insert_letter": _InsertLetter}


class Teacher:
    def __init__(self, env, task: str, variant: str, instruction: str, resume: bool = False):
        """resume: take over mid-episode from whatever state the env is in (the hook's resume routine)."""
        self.ctx = motion.Ctx(task, variant, instruction)
        self.hooks = HOOKS[task](env, variant)
        self.hooks.install()
        own = getattr(self.hooks, "resume" if resume else "routine", None)
        if resume and own is None:
            raise ValueError(f"{task}: the teacher hook has no resume routine")
        self.routine = own(self.ctx) if own else get_routine(task)(self.ctx)
        self.failed = None

    def act(self, state, images, step: int) -> np.ndarray:
        ctx = self.ctx
        ctx.q = np.asarray(state, dtype=float).reshape(-1)[:9]
        ctx.images, ctx.step = images, int(step)
        if self.failed is None:
            try:
                return np.asarray(next(self.routine), dtype=np.float32)
            except StopIteration:
                self.routine = motion.hold(ctx)
                return np.asarray(next(self.routine), dtype=np.float32)
            except Exception as exc:  # noqa: BLE001 - the episode is dropped
                self.failed = repr(exc)
        return ctx.hold_action()

    def aux(self) -> np.ndarray:
        return self.hooks.aux()

    def goals(self, variants) -> np.ndarray:
        """Per-variant target positions (n_variants x 3): what each instruction would refer to in this
        scene. Supervises the policy's goal head under every variant code, not only the true one."""
        return self.hooks.goals(variants)

    def noise_scale(self) -> float:
        return self.hooks.noise_scale(self.ctx)

    def on_step(self, t, obs, info) -> None:
        """Call after every env.step, before reading success (mirrors run_episode's on_step)."""
        self.hooks.on_step(t, obs, info)
