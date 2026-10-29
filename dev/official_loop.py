"""Dev: replay the official eval loop in-process (one env per profile, seeds in sequence,
official run_episode + probes), with MFPolicy behind the same batch round-trip as HTTP.

  python dev/official_loop.py grasp_move_mug left 1000411 1000412 1000413

Environment variables:
  PROFILE        perception profile (default clean)
  MFAGENT_ROOT   import mfagent from this checkout instead (e.g. a worktree of a released tag)
  SEED_NP        1 (default): seed NumPy's global RNG with the episode seed before each episode.
                 The official envs draw some object yaws from it unseeded, so without this the
                 episodes (and their action hashes) are not reproducible.
  RESULTS_JSONL  append one JSON line per episode (task, variant, seed, success, len, hash, ...)
  MF_SENSOR_WH   camera resolution override, e.g. 224x224 (see dev/simenv.py)
  ACTIONS_DIR    save each episode's states and actions as <task>_<variant>_<seed>.npz
  POLICY         lpolicy: run the learned policy (lpolicy/) instead of MFPolicy
  LPOLICY_WEIGHTS  directory holding <task>.pt for POLICY=lpolicy (default lpolicy/weights)
  ORACLE_VARIANT 1: map the instruction to its variant through the task config (bypasses the
                 learned instruction classifier; isolates execution quality)
  LPOLICY_EXEC   actions executed per predicted chunk (default: each network's own, else 8)
"""
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np

from dev.simenv import REPO, load_cfg

if os.environ.get("MFAGENT_ROOT"):
    sys.path.insert(0, os.path.abspath(os.environ["MFAGENT_ROOT"]))
sys.path.insert(0, REPO)
os.chdir(REPO)
from utils.eval_common import RemotePolicy, run_episode  # noqa: E402

from mfagent.policy import MFPolicy  # noqa: E402


def _make_impl(task_id):
    if os.environ.get("POLICY") == "lpolicy":
        from lpolicy.policy import LearnedPolicy
        variant_of = None
        if os.environ.get("ORACLE_VARIANT") == "1":
            strings = {s: v for v, s in load_cfg(task_id)["variants"].items()}
            variant_of = lambda task, instruction, names: strings.get(instruction)  # noqa: E731
        return LearnedPolicy(weights_dir=os.environ.get("LPOLICY_WEIGHTS"), variant_of=variant_of,
                             exec_steps=int(os.environ["LPOLICY_EXEC"]) if os.environ.get("LPOLICY_EXEC") else None)
    return MFPolicy()


class InProcessPolicy:
    """RemotePolicy semantics (deferred /reset, uint8 round trip) without HTTP."""

    def __init__(self, task_id, action_dim=8):
        self.impl = _make_impl(task_id)
        self.task_id, self.action_dim = task_id, action_dim
        self._pending, self._step = True, 0
        self.digest = hashlib.sha1()
        self.trace = ([], [])

    def reset(self):
        self._pending, self._step = True, 0
        self.digest = hashlib.sha1()
        self.trace = ([], [])

    def select_action(self, batch):
        state, images, instruction = RemotePolicy._split_batch(batch)
        if self._pending:
            self.impl.reset(self.task_id, instruction, self.action_dim)
            self._pending = False
        a = self.impl.act(state, images, instruction, self._step)
        self.digest.update(np.asarray(a, dtype=np.float32).tobytes())
        if os.environ.get("ACTIONS_DIR"):
            self.trace[0].append(np.asarray(state, dtype=np.float32).reshape(-1).copy())
            self.trace[1].append(np.asarray(a, dtype=np.float32).copy())
        self._step += 1
        import torch
        return torch.as_tensor(np.asarray(a, dtype=np.float32)[None])


def _seed_globals(seed):
    if os.environ.get("SEED_NP", "1") == "1":
        np.random.seed(int(seed) % 2**32)


def _record(task, variant, seed, ep, policy, **extra):
    """Action-hash tag for the result line; also appends a JSON line when RESULTS_JSONL is set."""
    h = policy.digest.hexdigest()[:16]
    if os.environ.get("ACTIONS_DIR"):
        np.savez(f"{os.environ['ACTIONS_DIR']}/{task}_{variant}_{seed}.npz",
                 states=np.array(policy.trace[0]), actions=np.array(policy.trace[1]))
    path = os.environ.get("RESULTS_JSONL")
    if path:
        row = {"task": task, "variant": variant, "seed": int(seed), "success": bool(ep["success"]),
               "len": int(ep["episode_length"]), "hash": h, "profile": os.environ.get("PROFILE", "clean"),
               "sensor": os.environ.get("MF_SENSOR_WH", ""),
               "error": policy.impl.ctx.notes.get("error"), "read_variant": policy.impl.ctx.notes.get("variant"),
               **extra}
        with open(path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
    return h


def main():
    task, variant, seeds = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3:]]
    cfg = load_cfg(task)
    instruction = cfg["variants"][variant]
    ident = lambda x: x  # noqa: E731
    policy = InProcessPolicy(task)
    profile = os.environ.get("PROFILE", "clean")
    if task == "grasp_move_mug":
        from eval.eval_grasp_move_mug import _episode_probe, _make_env
        from utils.eval_common import build_base_env_kwargs, build_perception_profiles
        prof = build_perception_profiles(build_base_env_kwargs(cfg), cfg)[profile]
        env = _make_env(cfg, variant, extra={k: v for k, v in prof.items() if k.startswith("eval_")})
        from mfagent import robot as _robot
        from dev.simenv import to_np as _to_np
        totals = []
        for seed in seeds:
            on_step0, finish = _episode_probe(env, variant, part_name=cfg.get("part_name", "handle"))
            trace = []

            def on_step(t, obs, info, _f=on_step0):
                _f(t, obs, info)
                q = _to_np(obs["agent"]["qpos"]).reshape(-1)
                if t % 5 == 0:
                    trace.append((t, _robot.tcp_pose(q)[:3, 3].round(3).tolist(),
                                  _to_np(env.unwrapped.target_object.pose.p).reshape(-1).round(3).tolist(),
                                  round(float(q[7] + q[8]), 4)))
            _seed_globals(seed)
            ep = run_episode(env, policy, ident, ident, task=instruction, seed=seed,
                             max_steps=int(cfg["max_episode_steps"]), on_step=on_step)
            diag = finish()
            ev = ep["extras"].get("evaluate") or {}
            h = _record(task, variant, seed, ep, policy, moved_tcp=ev.get("moved_tcp"),
                        moved_obj=ev.get("moved_object"))
            print(seed, "success", ep["success"], "len", ep["episode_length"], "hash", h, "grasp_step", diag["grasp_step"],
                  "moved_tcp", ev.get("moved_tcp"), "moved_obj", ev.get("moved_object"),
                  "obj_delta", np.round(diag["obj_delta"], 3) if diag["obj_delta"] else None,
                  "notes", {k: v for k, v in policy.impl.ctx.notes.items() if k in ("error",)}, flush=True)
            totals.append(bool(ep["success"]))
            if os.environ.get("TRACE"):
                for row in trace:
                    print("   ", *row)
            views = policy.impl.ctx.notes.get("views")
            if views and os.environ.get("VIEWS_DIR"):
                from PIL import Image
                Image.fromarray(np.concatenate(views, axis=1)).save(f"{os.environ['VIEWS_DIR']}/t2_{seed}.png")
        print(f"TOTAL {task} {variant}: {sum(totals)}/{len(totals)}")
    else:
        from dev.simenv import make_task_env
        env, _, _ = make_task_env(task, variant, profile)  # one env for all seeds, like the official loop
        n_ok = 0
        for seed in seeds:
            _seed_globals(seed)
            ep = run_episode(env, policy, ident, ident, task=instruction, seed=seed,
                             max_steps=int(cfg["max_episode_steps"]))
            n_ok += ep["success"]
            ev = ep["extras"].get("evaluate") or {}
            keep = ("red_toggled", "blue_toggled", "had_contact", "in_slot", "pos_diff_norm", "rot_diff")
            h = _record(task, variant, seed, ep, policy, stage=ep["extras"].get("stage_id"),
                        grasped=ep["extras"].get("grasped_part"), ev={k: v for k, v in ev.items() if k in keep})
            views = policy.impl.ctx.notes.get("views")
            if views and os.environ.get("VIEWS_DIR"):
                from PIL import Image
                Image.fromarray(np.concatenate(views, axis=1)).save(f"{os.environ['VIEWS_DIR']}/{task}_{variant}_{seed}.png")
            obj = getattr(env.unwrapped, "target_object", None)
            if obj is not None:
                from dev.simenv import to_np as _tn
                Tm = _tn(obj.pose.to_transformation_matrix()).reshape(4, 4)
                print("   object at end: p", Tm[:3, 3].round(3), "up-axis z", round(float(Tm[2, 2]), 3))
            print(seed, "success", ep["success"], "len", ep["episode_length"], "hash", h, "stage", ep["extras"].get("stage_id"),
                  "grasped", ep["extras"].get("grasped_part"),
                  "notes", {k: v for k, v in policy.impl.ctx.notes.items() if k in ("error",)}, flush=True)
        print(f"TOTAL {task} {variant}: {n_ok}/{len(seeds)}")


if __name__ == "__main__":
    main()
