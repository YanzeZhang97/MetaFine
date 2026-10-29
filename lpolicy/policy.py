"""Runtime: per-task ACT policies behind the harness interface (reset / act).

Each /act either pops the next action of the current chunk or, when the chunk is used up,
runs the task's network on the newest observation (hand camera + qpos + variant code) and
queues the first EXEC_STEPS actions. The gripper is sent as exactly +1 or -1 (classification
head), or, for a regression head (T5 pre-grasp openings), as its value snapped to +-1 near the ends.

Nothing here may raise: the harness aborts a whole task run on any server error, so every
failure is answered with a hold (zero arm delta, last gripper command).
"""
from __future__ import annotations

import os
import traceback
from types import SimpleNamespace

import numpy as np
import torch

from .model import ACTPolicy
from .preprocess import crop, resize_rgb

EXEC_STEPS = 8
GRIP_SNAP = 0.8  # regression gripper outputs beyond +-0.8 are sent as exactly +-1
WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")


class LearnedPolicy:
    name = "lpolicy"

    def __init__(self, weights_dir: str | None = None, device: str | None = None, variant_of=None,
                 exec_steps: int | None = None):
        self.dir = weights_dir or WEIGHTS
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.variant_of = variant_of  # (task_id, instruction, variant names) -> variant name
        if self.variant_of is None and os.path.isdir(os.path.join(self.dir, "text")):
            from .text import InstructionReader
            reader = InstructionReader(os.path.join(self.dir, "text"))
            self.variant_of = lambda task, instruction, names: reader.variant(task, instruction)
        self.exec_steps = None if exec_steps is None else int(exec_steps)  # None: each network's own setting
        self.models: dict = {}
        self.task, self.model, self.v = None, None, 0
        self.queue: list = []
        self.gripper = 1.0
        self.action_dim = 8
        self.ctx = SimpleNamespace(notes={})  # dev loops read ctx.notes
        # Every network is loaded here, once (the competition guide); one that fails is retried, and its error
        # reported, at /reset.
        for f in sorted(os.listdir(self.dir)) if os.path.isdir(self.dir) else []:
            if f.endswith(".pt"):
                try:
                    self._load(f[:-3])
                except Exception:
                    pass

    def _load(self, name: str):
        """Network ``<name>.pt``: a task (``grasp_part``) or a task's variant (``grasp_part__cap``)."""
        if name not in self.models:
            path = os.path.join(self.dir, f"{name}.pt")
            ck = torch.load(path, map_location="cpu", weights_only=False)
            m = ACTPolicy(**ck["model_cfg"])
            m.load_state_dict(ck["state_dict"])
            m.grip_mode = ck.get("grip_mode", "bce")
            m.crop_kind = ck.get("crop") or "center"
            m.exec_steps = int(ck.get("exec") or EXEC_STEPS)
            self.models[name] = (m.eval().to(self.device), ck["variants"])
        return self.models[name]

    def reset(self, task_id: str, instruction: str, action_dim: int) -> None:
        self.queue, self.gripper = [], 1.0
        self.ctx = SimpleNamespace(notes={})
        try:
            self.action_dim = int(action_dim) if action_dim else 8
            self.task = str(task_id)
            self.model, variants = self._load(self.task)
            name = self.variant_of(self.task, instruction, variants) if self.variant_of else variants[0]
            self.v = variants.index(name) if name in variants else 0
            self.ctx.notes["variant"] = variants[self.v]
            # A variant can have its own network (<task>__<variant>.pt), with the same variant list.
            special = f"{self.task}__{variants[self.v]}"
            if os.path.exists(os.path.join(self.dir, f"{special}.pt")):
                try:  # keep the task network if the variant's own one cannot be used
                    model, v2 = self._load(special)
                    if list(v2) == list(variants):
                        self.model = model
                except Exception:
                    self.ctx.notes["error"] = f"{special} failed: " + traceback.format_exc()[-400:]
        except Exception:
            self.model = None
            self.ctx.notes["error"] = "reset failed: " + traceback.format_exc()[-600:]

    @torch.inference_mode()
    def _plan(self, state, images) -> list:
        hand = images["hand_camera"]
        img = torch.from_numpy(resize_rgb(hand)).permute(2, 0, 1)[None].to(self.device)
        second = None
        if self.model.n_img == 2:
            second = torch.from_numpy(crop(hand, self.model.crop_kind)).permute(2, 0, 1)[None].to(self.device)
        q = torch.as_tensor(np.asarray(state, dtype=np.float32).reshape(-1)[:9])[None].to(self.device)
        v = torch.tensor([self.v], device=self.device)
        arm, grip = self.model(img, q, v, second)[:2]
        arm = arm[0].float().clamp(-1, 1).cpu().numpy()
        g = grip[0].float().cpu().numpy()
        if getattr(self.model, "grip_mode", "bce") == "reg":
            # continuous openings, but snapped to the ends: a half-open command drops a held object
            g = np.tanh(g)
            grip = np.where(g > GRIP_SNAP, 1.0, np.where(g < -GRIP_SNAP, -1.0, g))
        else:
            grip = np.where(g > 0, 1.0, -1.0)
        chunk = np.concatenate([arm, grip[:, None]], axis=1).astype(np.float32)
        return list(chunk[:self.exec_steps or self.model.exec_steps])

    def act(self, state, images, instruction: str, step: int) -> np.ndarray:
        try:
            s = np.asarray(state, dtype=np.float32).reshape(-1)
            if self.model is None or s.size < 9 or not np.all(np.isfinite(s[:9])):
                return self._hold()
            if not self.queue:
                self.queue = self._plan(s, images)
            a = self.queue.pop(0)
            self.gripper = float(a[7])
            return self._out(a)
        except Exception:
            if "error" not in self.ctx.notes:
                self.ctx.notes["error"] = f"act failed at step {step}: " + traceback.format_exc()[-600:]
            self.queue = []
            return self._hold()

    def _hold(self) -> np.ndarray:
        return self._out(np.concatenate([np.zeros(7), [self.gripper]]))

    def _out(self, a) -> np.ndarray:
        a = np.nan_to_num(np.asarray(a, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
        out = np.zeros(self.action_dim, dtype=np.float32)
        n = min(len(a), self.action_dim)
        out[:n] = a[:n]
        return np.clip(out, -1.0, 1.0)
