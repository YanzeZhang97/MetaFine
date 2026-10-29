"""Crash-proof policy: one generator routine per episode, driven one action per /act.

The official harness aborts a whole task run on any server exception, a reply
slower than 120 s, or a non-finite action, so nothing here may raise or hang:

- errors switch the episode to holding still with the last gripper command;
- a non-finite or implausible joint state is answered with a hold, and the routine
  is not advanced;
- each routine step runs in a worker thread with a time budget. If a step overruns,
  the reply is a hold and the step keeps running; later calls wait for it (again with
  a budget) and then send its action, which still applies because the arm held still.
"""
from __future__ import annotations

import threading
import time
import traceback

import numpy as np

from . import motion
from .instructions import parse_instruction
from .log import log as _log
from .tasks import get_routine, preload

STATE_LIMIT = 10.0     # |q| above this (rad or m) is not a Panda state
STEP_BUDGET_S = 40.0   # longest wait for one routine step within a request
CALL_BUDGET_S = 80.0   # longest total wait within one request (the harness allows 120 s)
MIN_STEP_BUDGET_S = 5.0


class MFPolicy:
    name = "mfagent"

    def __init__(self) -> None:
        preload()  # raises here, not mid-evaluation, if anything the routines need is missing
        self.action_dim = 8
        self.ctx = motion.Ctx("unknown", "", "")
        self._routine = None
        self._routine_fn = get_routine("unknown")
        self._failed = False
        self._busy = None  # (thread, result box) of a step that overran its budget
        self._last_reset = None
        self._warned = set()
        self._unparsed = set()  # (task, instruction) already logged as not recognised
        self.last_act_seconds = 0.0

    def _warn_once(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            _log(msg)

    def reset(self, task_id: str, instruction: str, action_dim: int) -> None:
        self._busy = None  # abandon a late step of the previous episode (it keeps the old ctx)
        self._routine, self._failed, self._warned = None, False, set()
        try:
            self.action_dim = int(action_dim) if action_dim else 8
            variant, how = parse_instruction(task_id, instruction)
            self.ctx = motion.Ctx(task_id, variant, instruction)
            self._routine_fn = get_routine(task_id)
            if (task_id, variant) != self._last_reset:
                self._last_reset = (task_id, variant)
                _log(f"reset task={task_id} variant={variant!r}")
        except Exception:
            _log("reset failed:\n" + traceback.format_exc()[-1200:])
            self.ctx = motion.Ctx(str(task_id), "", str(instruction))
            self._routine_fn = get_routine("unknown")
            return
        try:
            key = (str(task_id), str(instruction)[:300])
            if how == "default" and key not in self._unparsed and len(self._unparsed) < 50:
                self._unparsed.add(key)
                _log(f"instruction not recognised for task={key[0]}, using the default variant {variant!r}: {key[1]!r}")
        except Exception:
            pass

    def _start_step(self, budget: float):
        gen, box = self._routine, {}

        def run():
            try:
                box["a"] = gen.send(None)
            except StopIteration:
                box["stop"] = True
            except BaseException as exc:  # noqa: BLE001 - re-raised in the request thread
                box["exc"] = exc
                box["tb"] = traceback.format_exc()

        th = threading.Thread(target=run, daemon=True, name="mfagent-step")
        th.start()
        th.join(budget)
        return th, box

    def _result(self, box):
        """Action of a finished step; switches to holding when the routine ended."""
        if "exc" in box:
            raise RuntimeError(box.get("tb", repr(box["exc"])))
        if box.get("stop"):
            self._routine = motion.hold(self.ctx)
            return self._routine.send(None)
        return box["a"]

    def act(self, state, images, instruction: str, step: int) -> np.ndarray:
        t0 = time.time()
        ctx = self.ctx
        try:
            if self._busy is not None:
                th, box = self._busy
                th.join(STEP_BUDGET_S)
                if th.is_alive():  # still late: keep holding, and leave ctx alone while it runs
                    self.last_act_seconds = time.time() - t0
                    return self._sanitize(ctx.hold_action())
                # The arm held still while the step ran, so its action still applies: send it now.
                self._busy = None
                action = self._result(box)
                self.last_act_seconds = time.time() - t0
                return self._sanitize(action)
            q = np.zeros(9)
            s = np.asarray(state, dtype=float).reshape(-1)[:9]
            q[:len(s)] = s
            if not np.all(np.isfinite(q)) or np.abs(q).max() > STATE_LIMIT:
                self._warn_once("state", f"implausible joint state at step {step}, holding: {np.round(s, 3).tolist()}")
                self.last_act_seconds = time.time() - t0
                return self._sanitize(ctx.hold_action())
            ctx.q, ctx.images, ctx.step = q, images, int(step)
            if self._routine is None:
                self._routine = self._routine_fn(ctx)
            budget = min(STEP_BUDGET_S, max(MIN_STEP_BUDGET_S, CALL_BUDGET_S - (time.time() - t0)))
            th, box = self._start_step(budget)
            if th.is_alive():
                self._busy = (th, box)
                self._warn_once("late", f"step {step} exceeded {budget:.0f} s: holding until it finishes")
                action = ctx.hold_action()
            else:
                action = self._result(box)
        except Exception:
            if not self._failed:
                _log(f"routine failed at step {step}, holding still:\n" + traceback.format_exc()[-1200:])
            self._failed = True
            self._routine = motion.hold(ctx)
            action = ctx.hold_action()
        self.last_act_seconds = time.time() - t0
        return self._sanitize(action)

    def _sanitize(self, action) -> np.ndarray:
        try:
            a = np.nan_to_num(np.asarray(action, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            a = self.ctx.hold_action()
        out = np.zeros(self.action_dim, dtype=np.float32)
        n = min(len(a), self.action_dim)
        out[:n] = a[:n]
        if self.action_dim >= 8 and len(a) < 8:
            out[7] = self.ctx.gripper
        return np.clip(out, -1.0, 1.0)
