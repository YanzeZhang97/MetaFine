"""Dev: takeover demos (HG-DAgger style) for the learned policy.

A trained student drives the episode until a trigger; then the privileged teacher takes over from the
state the student left (dev/teacher.py, Teacher(resume=True)) and only the teacher's steps are
recorded, the way dev/gen_demos.py records them (ambient light DR, DART noise, POST steps after
success, second-stream crop, masks). The demos so start from the student's own mistakes: an
off-centre grasp, a peg pressed beside its slot, a peg pushed away.

Triggers, drawn per episode. insert_letter:
  - "low"   (50 %): the first step the held peg's bottom is within 3 cm of the board top and the peg
                    within 4 cm of its slot;
  - "carry" (30 %): a uniform step in [lift + 5, lift + 120] (lift: first step the held peg is 1 cm up);
  - "early" (20 %): a uniform step in [40, 150] (approach and grasp states);
falling back to step 160 when the student has not lifted the peg by then, and to the step it drops a
lifted peg. --triggers sets other weights and can add "stall": from step 30 on, the first step the student's TCP
has moved under 3 mm over 12 steps (holding still over a grasp, over the board, or with a home peg still held);
fallback step 300; "seat": the held peg in its hole (origin 1.5-12 mm up, within 6 mm of the slot) but not home;
"home": the peg home and still held for 4 steps. grasp_part (cap): "near" (50 %): the TCP first within 8 cm of the cap; "early" (20 %):
a uniform step in [20, 80]; and always "miss": the student has commanded the gripper closed for 6
steps without holding the bottle (30 % of the episodes wait for this one); fallback step 200.
Episodes the student finishes before the trigger, or that the teacher does not finish within
HORIZON steps, are not written.

  python dev/gen_takeover.py insert_letter OUT FIRST_SEED N --student RUN/weights --set V2=true [--crop grip]
"""
import argparse
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import torch

torch.set_num_threads(1)

from dev.gen_demos import CROP_TASKS, POST, RHO, far_letters
from dev.simenv import load_cfg, make_task_env, obs_to_policy_inputs, to_np
from dev.teacher import HOOKS, Teacher
from lpolicy.policy import LearnedPolicy
from lpolicy.preprocess import crop, resize_rgb

HORIZON = 600  # student + teacher steps; the eval allows 400, but late corrections are still good labels
TRIGGERS = (("low", 0.5), ("carry", 0.3), ("early", 0.2))


class InsertLetterTrigger:
    def __init__(self, env, variant, rng):
        from mfagent import letters
        self.u, self.letter = env.unwrapped, variant
        self.path = []
        self.slot = letters.slot_pose(variant)[:2]
        self.kind = rng.choice([k for k, _ in TRIGGERS], p=[p for _, p in TRIGGERS])
        self.rng = rng
        self.t_early = int(rng.integers(40, 151))
        self.t_lift = None
        self.t_carry = None
        self.n_home = 0

    def __call__(self, t, state, last_action) -> str | None:
        from mfagent import robot
        self.path.append(robot.tcp_pose(np.asarray(state, dtype=float).reshape(-1)[:7])[:3, 3])
        peg = self.u.pegs[self.letter]
        p = to_np(peg.pose.p).reshape(-1)
        held = bool(np.asarray(to_np(self.u.agent.is_grasping(peg))).reshape(-1)[0])
        lifted = held and p[2] > 0.01
        if self.t_lift is None and lifted:
            self.t_lift = t
            self.t_carry = t + int(self.rng.integers(5, 121))
        if self.t_lift is not None and not held and p[2] > 0.02:
            return "drop"
        if self.kind == "early" and t >= self.t_early:
            return "early"
        if self.kind == "stall":
            if t >= 30 and len(self.path) > 12 and np.linalg.norm(self.path[-1] - self.path[-13]) < 0.003:
                return "stall"
            if t >= 300:
                return "late"
        if self.t_lift is None:
            return "no_lift" if t >= 160 else None
        if self.kind == "carry" and t >= self.t_carry:
            return "carry"
        if self.kind == "low" and held and p[2] < 0.014 + 0.03 and np.hypot(*(p[:2] - self.slot)) < 0.04:
            return "low"
        # "seat": the held peg is in its hole but not home (t5_g on o let go 1.4-5 mm up and the o landed 9-14 deg
        # off, runs/diag/t5_o_yaw); the teacher re-centres, pushes it home, then lets go.
        if self.kind == "seat" and held and 0.0015 < p[2] < 0.012 and np.hypot(*(p[:2] - self.slot)) < 0.006:
            return "seat"
        # "home": the peg is home and still held for 4 steps (students that never let go).
        if self.kind == "home":
            self.n_home = self.n_home + 1 if held and p[2] < 0.0025 and np.hypot(*(p[:2] - self.slot)) < 0.008 else 0
            if self.n_home >= 4:
                return "home"
        return None


class GraspPartTrigger:
    """cap: "near" (the TCP within 8 cm of the cap), "miss", "early"; body: "stall" (the TCP moved under 3 mm
    over 15 steps with the gripper open, after step 40), "near" (within 10 cm of the body grasp point), "miss",
    "early". Fallback: step 200."""
    KINDS = {"cap": (("near", 0.5), ("miss", 0.3), ("early", 0.2)),
             "body": (("stall", 0.5), ("near", 0.3), ("early", 0.2))}

    def __init__(self, env, variant, rng):
        self.u, self.variant = env.unwrapped, variant
        kinds = self.KINDS[variant]
        self.kind = rng.choice([k for k, _ in kinds], p=[p for _, p in kinds])
        self.t_early = int(rng.integers(20, 81))
        self.closed = 0
        self.path = []

    def __call__(self, t, state, last_action) -> str | None:
        from mfagent import robot
        T = to_np(self.u.target_object.pose.to_transformation_matrix()).reshape(4, 4)
        part = T[:3, 3] + T[:3, 2] * (0.140 if self.variant == "cap" else 0.012)
        p = robot.tcp_pose(np.asarray(state, dtype=float).reshape(-1)[:7])[:3, 3]
        self.path.append(p)
        holding = len(self.u._grasped_links()) > 0  # the bottle is an articulation (cap and body links)
        self.closed = self.closed + 1 if last_action is not None and last_action[7] < 0 and not holding else 0
        if self.closed >= 6:
            return "miss"
        if self.kind == "early" and t >= self.t_early:
            return "early"
        if self.kind == "near" and np.linalg.norm(p - part) < (0.08 if self.variant == "cap" else 0.10):
            return "near"
        if (self.kind == "stall" and t >= 40 and len(self.path) > 15 and not self.closed
                and np.linalg.norm(self.path[-1] - self.path[-16]) < 0.003):
            return "stall"
        return "late" if t >= 200 else None


class ToggleSwitchTrigger:
    """Always: "wrong" (the other slider moved past 15 % of its stroke), "high" (the instructed slider moved past
    15 % with the contact gate never latched: pushing too high; the teacher latches at <= 8 %). Drawn per episode
    (--triggers sets weights): "near" (the TCP first within 4 cm of the push contact), "stall" (from step 20, the
    TCP moved under 3 mm over 12 steps before the slider moved), "early" (a uniform step in [15, 120]).
    Fallback: step 200 ("late")."""
    KINDS = (("near", 0.35), ("stall", 0.35), ("early", 0.3))

    def __init__(self, env, variant, rng):
        self.u, self.variant = env.unwrapped, variant
        kinds = TRIGGERS if TRIGGERS[0][0] in dict(self.KINDS) else self.KINDS
        self.kind = rng.choice([k for k, _ in kinds], p=[p for _, p in kinds])
        self.t_early = int(rng.integers(15, 121))
        self.path = []
        self.hook = HOOKS["toggle_switch_table"](env, variant)

    def __call__(self, t, state, last_action) -> str | None:
        from mfagent import robot
        u = self.u
        names = u.scene_builder.switch_color
        jt = u._LINK_TO_JOINT_ID[names[self.variant]]
        jo = u._LINK_TO_JOINT_ID[names["blue" if self.variant == "red" else "red"]]
        q, q0 = to_np(u.target_object.get_qpos()).reshape(-1), to_np(u.init_qpos).reshape(-1)
        joints = u.target_object.get_active_joints()
        stroke = [float(j.get_limits()[0][1] - j.get_limits()[0][0]) for j in joints]
        moved_t, moved_o = abs(q[jt] - q0[jt]) / stroke[jt], abs(q[jo] - q0[jo]) / stroke[jo]
        p = robot.tcp_pose(np.asarray(state, dtype=float).reshape(-1)[:7])[:3, 3]
        self.path.append(p)
        if moved_o > 0.15:
            return "wrong"
        if moved_t > 0.15 and not u._had_target_contact:
            return "high"
        if moved_t > 0.15:
            return None  # pushing with the contact latched: leave it to the student
        if self.kind == "early" and t >= self.t_early:
            return "early"
        if self.kind == "near" and np.linalg.norm(p - self.hook.goals([self.variant])[0]) < 0.04:
            return "near"
        if (self.kind == "stall" and t >= 20 and len(self.path) > 12
                and np.linalg.norm(self.path[-1] - self.path[-13]) < 0.003):
            return "stall"
        return "late" if t >= 200 else None


TRIGGER_CLASSES = {"insert_letter": InsertLetterTrigger, "grasp_part": GraspPartTrigger,
                   "toggle_switch_table": ToggleSwitchTrigger}


def run(env, task, variant, instruction, seed, student, rng, clean_p, noise_p, img_stride, crop_kind):
    from utils.eval_common import _extract_success
    u = env.unwrapped
    np.random.seed(seed % 2**32)
    obs, _ = env.reset(seed=seed)
    ambient = [0.3] * 3 if rng.random() < clean_p else rng.uniform(0.05, 0.95, 3).round(3).tolist()
    for sc in u.scene.sub_scenes:
        sc.ambient_light = ambient
    obs = u.get_obs()
    trigger = TRIGGER_CLASSES[task](env, variant, rng)
    student.reset(task, instruction, 8)
    teacher, why, t_take, last_a = None, None, -1, None
    sigma = float(rng.uniform(0.05, 0.25)) if rng.random() < noise_p else 0.0
    noise = np.zeros(7)
    rec = {k: [] for k in ("hand", "img_t", "qpos", "action", "exec", "aux", "goal", "mask")}
    if crop_kind:
        rec["crop"] = []
    variants = list(load_cfg(task)["variants"])
    succ = -1
    for t in range(HORIZON + POST):
        state, images = obs_to_policy_inputs(obs)
        if teacher is None:
            why = trigger(t, state, last_a)
            if why is not None:
                teacher, t_take = Teacher(env, task, variant, instruction, resume=True), t
        if teacher is None:
            e = last_a = np.asarray(student.act(state, images, instruction, t), dtype=np.float32)
        else:
            a = teacher.act(state, images, t)
            if teacher.failed:
                break
            e = a.copy()
            if sigma > 0 and succ < 0:
                noise = RHO * noise + np.sqrt(1 - RHO ** 2) * sigma * rng.standard_normal(7)
                e[:7] = np.clip(a[:7] + teacher.noise_scale() * noise, -1.0, 1.0)
            k = t - t_take
            if k % img_stride == 0:
                rec["hand"].append(resize_rgb(images["hand_camera"]))
                rec["img_t"].append(k)
                if "crop" in rec:
                    rec["crop"].append(crop(images["hand_camera"], crop_kind))
            rec["qpos"].append(state[:9].astype(np.float32))
            rec["action"].append(a)
            rec["exec"].append(e.astype(np.float32))
            rec["aux"].append(teacher.aux())
            rec["goal"].append(teacher.goals(variants))
            rec["mask"].append(bool(teacher.ctx.notes.get("mask", False)))
        obs, _, _, _, info = env.step(e)
        if teacher is not None:
            teacher.on_step(t, obs, info)
        if succ < 0 and _extract_success(info):
            succ = t
            if teacher is None:  # the student finished on its own: nothing to label
                break
        if succ >= 0 and t >= succ + POST:
            break
        if succ < 0 and t >= HORIZON - 1:
            break
    meta = {"task": task, "variant": variant, "seed": seed, "instruction": instruction, "ambient": ambient,
            "sigma": round(sigma, 4), "success_step": succ - t_take if succ >= 0 and teacher else -1,
            "steps": len(rec["action"]), "takeover_t": t_take, "trigger": why, "trigger_kind": trigger.kind,
            "student_success": bool(succ >= 0 and teacher is None),
            "error": (teacher.failed or teacher.ctx.notes.get("error")) if teacher else None, "crop": crop_kind,
            "resume": teacher.ctx.notes.get("resume") if teacher else None}
    return ({k: np.stack(v) for k, v in rec.items()} if rec["action"] else None), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=sorted(TRIGGER_CLASSES))
    ap.add_argument("out")
    ap.add_argument("first_seed", type=int)
    ap.add_argument("n_seeds", type=int)
    ap.add_argument("--student", required=True, help="weights dir with <task>.pt (the policy to take over from)")
    ap.add_argument("--clean-p", type=float, default=0.35)
    ap.add_argument("--noise-p", type=float, default=0.5)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--img-stride", type=int, default=1)
    ap.add_argument("--crop", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="ATTR=VALUE")
    ap.add_argument("--triggers", default=None, help='insert_letter trigger weights, e.g. "low=0.3,stall=0.4,carry=0.15,early=0.15"'
                    ' (toggle_switch_table: near / stall / early)')
    ap.add_argument("--min-x", type=float, default=None,
                    help="insert_letter: only the variants whose peg starts at x >= MIN_X (m); see gen_demos.far_letters")
    args = ap.parse_args()
    if args.triggers:
        global TRIGGERS
        TRIGGERS = tuple((k, float(w)) for k, w in (kv.split("=") for kv in args.triggers.split(",")))
        assert abs(sum(w for _, w in TRIGGERS) - 1.0) < 1e-6, TRIGGERS
    assert args.first_seed >= 2_000_000, "training seeds must stay outside the eval range"
    for kv in args.set:
        k, v = kv.split("=", 1)
        setattr(HOOKS[args.task], k, json.loads(v))
    cfg = load_cfg(args.task)
    crop_kind = args.crop or CROP_TASKS.get(args.task)
    variants = args.variants or list(cfg["variants"])
    strings = {s: v for v, s in cfg["variants"].items()}
    student = LearnedPolicy(weights_dir=os.path.abspath(args.student),
                            variant_of=lambda task, instruction, names: strings.get(instruction))
    envs = {v: make_task_env(args.task, v, "clean")[0] for v in variants}
    os.makedirs(args.out, exist_ok=True)
    n_ok = n_all = 0
    t_start = time.time()
    for seed in range(args.first_seed, args.first_seed + args.n_seeds):
        far = far_letters(envs[variants[0]], seed, args.min_x) if args.min_x is not None else None
        for vi, v in enumerate(variants):
            path = os.path.join(args.out, f"{v}_{seed}.npz")
            if os.path.exists(path) or (far is not None and v not in far):
                continue
            rng = np.random.default_rng([seed, vi, 13])
            t0 = time.time()
            data, meta = run(envs[v], args.task, v, cfg["variants"][v], seed, student, rng, args.clean_p,
                             args.noise_p, args.img_stride, crop_kind)
            meta["set"] = args.set
            meta["triggers"] = TRIGGERS
            meta["student_dir"] = os.path.abspath(args.student)
            ok = data is not None and meta["success_step"] >= 0
            if ok:
                with open(path + ".tmp", "wb") as f:
                    np.savez_compressed(f, **data, success_step=meta["success_step"], meta=json.dumps(meta))
                os.replace(path + ".tmp", path)
            meta.update(ok=ok, wall_s=round(time.time() - t0, 2))
            with open(os.path.join(args.out, "log.jsonl"), "a") as f:
                f.write(json.dumps(meta) + "\n")
            n_ok += ok
            n_all += 1
            print(f"{v} {seed} ok={ok} trigger={meta['trigger']}@{meta['takeover_t']} steps={meta['steps']} "
                  f"succ@{meta['success_step']} student_success={meta['student_success']} sigma={meta['sigma']} "
                  f"{meta['wall_s']}s err={meta['error']}", flush=True)
    print(f"DONE {args.task}: {n_ok}/{n_all} kept, {time.time() - t_start:.0f} s", flush=True)
    for e in envs.values():
        e.close()


if __name__ == "__main__":
    main()
