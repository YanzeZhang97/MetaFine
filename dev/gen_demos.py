"""Dev: render teacher demonstrations for the learned policy (privileged teacher: dev/teacher.py).

Mirrors the official loop: one env per variant, reused across seeds; NumPy's global RNG seeded
with the episode seed; the policy sees 9-dim qpos and uint8 RGB. Every seed is run under every
variant, so the variant is the only thing that tells those trajectories apart.

Per episode:
- ambient light: the eval's clean grey 0.3 with probability --clean-p, else U[0.05, 0.95]^3
  (light_l3 draws U[0.1, 0.9]^3);
- DART noise on a share (--noise-p) of the episodes: AR(1) noise on the 7 arm dims of the
  executed action, sigma drawn per episode, scaled down near the target (Teacher.noise_scale).
  The label is always the teacher's clean action, which corrects the drift on the next step;
- recording continues POST steps past success (noise off, the teacher holds), so the student
  learns to stay put.
Only successful episodes are written, as OUT/<variant>_<seed>.npz; every episode gets a line in
OUT/log.jsonl. Seeds must stay outside the eval range (use >= 5,000,000 for training data).

  python dev/gen_demos.py toggle_switch_table OUT_DIR FIRST_SEED N_SEEDS
"""
import argparse
import json
import os
import sys
import time

# One core per generator: torch / BLAS would otherwise start a thread per core for tiny tensors,
# and several generators then oversubscribe the machine (seen: 8-11 cores each, all slower).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import torch

torch.set_num_threads(1)

from dev.simenv import load_cfg, make_task_env, obs_to_policy_inputs, to_np
from dev.teacher import Teacher
from lpolicy.preprocess import crop, resize_rgb

POST = 20      # recorded steps after success
RHO = 0.9      # AR(1) coefficient of the DART noise
# Tasks that also store a second, higher-resolution crop of the hand image (the peg strokes are ~4 px
# wide at 224 from 512), and its default kind (lpolicy.preprocess.crop; --crop overrides).
CROP_TASKS = {"insert_letter": "center"}


def far_letters(env, seed, min_x):
    """insert_letter: the letters whose peg starts at x >= min_x (m) for this seed. The layout depends only on the
    seed (ManiSkill seeds _initialize_episode from it), so any variant's env gives it. t5_j failed 24 % of the
    episodes with the instructed peg at x >= 0.12 (the far right, near the arm's reach) against 4 % elsewhere."""
    env.reset(seed=seed)
    u = env.unwrapped
    return {L for L, peg in u.pegs.items() if float(np.asarray(to_np(peg.pose.p)).reshape(-1)[0]) >= min_x}


def run(env, task, variant, instruction, seed, max_steps, rng, clean_p, noise_p, img_stride=1, crop_kind=None):
    from utils.eval_common import _extract_success
    u = env.unwrapped
    np.random.seed(seed % 2**32)
    obs, _ = env.reset(seed=seed)
    ambient = [0.3] * 3 if rng.random() < clean_p else rng.uniform(0.05, 0.95, 3).round(3).tolist()
    for sc in u.scene.sub_scenes:
        sc.ambient_light = ambient
    obs = u.get_obs()
    teacher = Teacher(env, task, variant, instruction)
    sigma = float(rng.uniform(0.05, 0.25)) if rng.random() < noise_p else 0.0
    noise = np.zeros(7)
    rec = {k: [] for k in ("hand", "img_t", "qpos", "action", "exec", "aux", "goal", "mask")}
    if crop_kind:
        rec["crop"] = []
    variants = list(load_cfg(task)["variants"])
    succ = -1
    for t in range(max_steps + POST):
        state, images = obs_to_policy_inputs(obs)
        a = teacher.act(state, images, t)
        if teacher.failed:
            break
        e = a.copy()
        if sigma > 0 and succ < 0:
            noise = RHO * noise + np.sqrt(1 - RHO ** 2) * sigma * rng.standard_normal(7)
            e[:7] = np.clip(a[:7] + teacher.noise_scale() * noise, -1.0, 1.0)
        if t % img_stride == 0:
            rec["hand"].append(resize_rgb(images["hand_camera"]))
            rec["img_t"].append(t)
            if "crop" in rec:
                rec["crop"].append(crop(images["hand_camera"], crop_kind))
        rec["qpos"].append(state[:9].astype(np.float32))
        rec["action"].append(a)
        rec["exec"].append(e.astype(np.float32))
        rec["aux"].append(teacher.aux())
        rec["goal"].append(teacher.goals(variants))
        rec["mask"].append(bool(teacher.ctx.notes.get("mask", False)))  # a deliberately wrong action: no label
        obs, _, _, _, info = env.step(e)
        teacher.on_step(t, obs, info)
        if succ < 0 and _extract_success(info):
            succ = t
        if succ >= 0 and t >= succ + POST:
            break
        if succ < 0 and t >= max_steps - 1:
            break
    meta = {"task": task, "variant": variant, "seed": seed, "instruction": instruction, "ambient": ambient,
            "sigma": round(sigma, 4), "success_step": succ, "steps": len(rec["action"]),
            "error": teacher.failed or teacher.ctx.notes.get("error"), "crop": crop_kind}
    return {k: np.stack(v) for k, v in rec.items()} if rec["action"] else None, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("first_seed", type=int)
    ap.add_argument("n_seeds", type=int)
    ap.add_argument("--clean-p", type=float, default=0.35)
    ap.add_argument("--noise-p", type=float, default=0.5)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--img-stride", type=int, default=1, help="store the hand image every K steps (long tasks)")
    ap.add_argument("--crop", default=None, help="second-stream crop kind (default: CROP_TASKS[task] or none)")
    ap.add_argument("--set", action="append", default=[], metavar="ATTR=VALUE",
                    help="set a teacher-hook class attribute, e.g. CREEP_V=0.012 (grasp_move_mug)")
    ap.add_argument("--min-x", type=float, default=None,
                    help="insert_letter: only the variants whose peg starts at x >= MIN_X (m); see far_letters")
    args = ap.parse_args()
    assert args.first_seed >= 2_000_000, "training seeds must stay outside the eval range"
    from dev.teacher import HOOKS
    for kv in args.set:
        k, v = kv.split("=", 1)
        setattr(HOOKS[args.task], k, json.loads(v))
    cfg = load_cfg(args.task)
    crop_kind = args.crop or CROP_TASKS.get(args.task)
    variants = args.variants or list(cfg["variants"])
    max_steps = int(cfg["max_episode_steps"])
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
            rng = np.random.default_rng([seed, vi, 7])
            t0 = time.time()
            data, meta = run(envs[v], args.task, v, cfg["variants"][v], seed, max_steps, rng,
                             args.clean_p, args.noise_p, args.img_stride, crop_kind)
            meta["set"] = args.set
            ok = data is not None and 0 <= meta["success_step"] < max_steps
            if ok:  # written under a temporary name first: a trainer may be reading the directory
                with open(path + ".tmp", "wb") as f:
                    np.savez_compressed(f, **data, success_step=meta["success_step"], meta=json.dumps(meta))
                os.replace(path + ".tmp", path)
            meta.update(ok=ok, wall_s=round(time.time() - t0, 2))
            with open(os.path.join(args.out, "log.jsonl"), "a") as f:
                f.write(json.dumps(meta) + "\n")
            n_ok += ok
            n_all += 1
            print(f"{v} {seed} ok={ok} steps={meta['steps']} succ@{meta['success_step']} sigma={meta['sigma']} "
                  f"amb={meta['ambient']} {meta['wall_s']}s err={meta['error']}", flush=True)
    print(f"DONE {args.task}: {n_ok}/{n_all} kept, {time.time() - t_start:.0f} s", flush=True)
    for e in envs.values():
        e.close()


if __name__ == "__main__":
    main()
