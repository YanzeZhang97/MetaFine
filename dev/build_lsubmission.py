"""Build the HF submission folder for the learned policy (lpolicy/) from the official template.

The template's protocol plumbing is kept byte-identical; only the ``Policy`` class is replaced by
a thin, crash-proof wrapper around ``lpolicy.policy.LearnedPolicy``. Weights: one ACT checkpoint
per task (tasks without one hold still) and the instruction classifier.

  python dev/build_lsubmission.py --weights toggle_switch_table=runs/t3_c/last.pt [--weights ...] \
      --text runs/text_clf_v1 [--out hf_submission_l] [--team NAME] [--email ADDR] [--no-evaluate]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.environ.get("METAFINE_REPO") or os.path.join(ROOT, "MetaFine-Challenge_CoRL")  # official repository
TEMPLATE = os.path.join(REPO, "submission", "policy_server.py")
TASKS = ("grasp_part", "grasp_move_mug", "toggle_switch_table", "put_blocks_into_boxes", "insert_letter")
TEXT_FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.txt",
              "special_tokens_map.json", "heads.pt", "tasks.json")

POLICY_CLASS = '''class Policy:
    """MetaFine entry: learned visuomotor policies (code in lpolicy/).

    One ACT-style network per task maps the hand-camera image, the joint state and the instruction
    (read by a learned text classifier) to joint-delta action chunks. Every call is guarded: the
    official harness aborts a task run on any server error, so failures fall back to holding still
    with the last gripper command. If lpolicy cannot be loaded, /health reports "lpolicy-LOAD-FAILED".
    """

    name = "lpolicy"

    def __init__(self) -> None:
        import os
        import sys
        import warnings
        warnings.simplefilter("ignore")  # a warning written to a dead stderr pipe must not fail a request
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        self.action_dim = 8
        self._gripper = 1.0
        self._impl = None
        self._said = 0
        try:
            from lpolicy.policy import LearnedPolicy
            self._impl = LearnedPolicy()
        except Exception as exc:  # an import problem must not take the server down
            self.name = "lpolicy-LOAD-FAILED"
            self._say(f"failed to load lpolicy: {exc!r}")

    def _say(self, msg: str) -> None:
        # Never raises and writes at most a few kB: a closed or unread log pipe must not
        # turn into a failed or hung request.
        try:
            if self._said < 4000:
                self._said += len(msg) + 10
                print(f"[policy] {msg[:1000]}", flush=True)
        except Exception:
            pass

    def reset(self, task_id: str, instruction: str, action_dim: int) -> None:
        try:
            self.action_dim = int(action_dim)
        except Exception:
            self.action_dim = 8
        self._gripper = 1.0
        if self._impl is not None:
            try:
                self._impl.reset(task_id, instruction, self.action_dim)
                err = self._impl.ctx.notes.get("error")
                if err:
                    self._say(err)
            except Exception as exc:
                self._say(f"reset error: {exc!r}")

    def act(self, state: np.ndarray, images: dict[str, np.ndarray],
            instruction: str, step: int) -> np.ndarray:
        action = None
        if self._impl is not None:
            try:
                action = np.asarray(self._impl.act(state, images, instruction, step),
                                    dtype=np.float32).reshape(-1)
            except Exception as exc:
                self._say(f"act error: {exc!r}")
        if action is None or action.shape[0] != self.action_dim or not np.all(np.isfinite(action)):
            action = np.zeros(self.action_dim, dtype=np.float32)
            if self.action_dim >= 8:
                action[7] = self._gripper
        elif self.action_dim >= 8:
            self._gripper = float(action[7])
        return np.clip(action, -1.0, 1.0).astype(np.float32)


'''

START = "class Policy:"
END = "# --------------------------------------------------------------------------- #\n# Protocol plumbing below"


def build(out: str, weights: dict, text: str, team: str, email: str, evaluate: bool, execs: dict) -> None:
    src = open(TEMPLATE).read()
    i, j = src.index(START), src.index(END)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    with open(os.path.join(out, "policy_server.py"), "w") as f:
        f.write(src[:i] + POLICY_CLASS + src[j:])
    pkg = os.path.join(out, "lpolicy")
    shutil.copytree(os.path.join(ROOT, "lpolicy"), pkg,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "weights"))
    wdir = os.path.join(pkg, "weights")
    os.makedirs(os.path.join(wdir, "text"))
    for task, path in weights.items():
        ck = torch.load(path, map_location="cpu", weights_only=False)
        keep = {k: ck[k] for k in ("state_dict", "model_cfg", "variants", "task") if k in ck}
        keep["grip_mode"] = ck.get("grip_mode", "bce")
        keep["crop"] = ck.get("crop") or "center"
        if task.split("@")[0] in execs:
            keep["exec"] = execs[task.split("@")[0]]
        torch.save(keep, os.path.join(wdir, f"{task.replace('@', '__')}.pt"))  # TASK@VARIANT -> TASK__VARIANT.pt
    for name in TEXT_FILES:
        shutil.copy(os.path.join(text, name), os.path.join(wdir, "text", name))
    with open(os.path.join(out, "requirements.txt"), "w") as f:
        f.write("# numpy and torch come with the official base image.\ntransformers\ntokenizers\nsafetensors\n")
    with open(os.path.join(out, "metadata.json"), "w") as f:
        json.dump({"team_name": team, "contact_email": email, "evaluate": evaluate}, f, indent=2)
        f.write("\n")
    with open(os.path.join(out, ".gitignore"), "w") as f:
        f.write("__pycache__/\n*.pyc\n")
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write("---\ntags:\n- robotics\n---\n\n"
                "# MetaFine CoRL 2026 entry (learned policy)\n\nPolicy server for the MetaFine Open Competition. "
                "Entry point: `policy_server.py` (official template); code in `lpolicy/`: one ACT-style "
                "visuomotor network per task (hand camera + joint state + instruction -> joint-delta action "
                "chunks) and a learned instruction classifier. Weights in `lpolicy/weights/`.\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "hf_submission_l"))
    ap.add_argument("--weights", action="append", default=[],
                    help="TASK=checkpoint.pt, or TASK@VARIANT=checkpoint.pt for one variant's own network (repeatable)")
    ap.add_argument("--text", required=True, help="instruction classifier dir (dev/train_text.py output)")
    ap.add_argument("--exec", action="append", default=[], metavar="TASK=N",
                    help="actions executed per predicted chunk for TASK's networks (default 8; repeatable)")
    ap.add_argument("--team", default="TBD")
    ap.add_argument("--email", default="TBD")
    ap.add_argument("--no-evaluate", action="store_true")
    a = ap.parse_args()
    weights = dict(w.split("=", 1) for w in a.weights)
    bad = [t for t in weights if t.split("@")[0] not in TASKS]  # TASK or TASK@VARIANT
    if bad:
        sys.exit(f"unknown task(s): {bad}")
    execs = {t: int(n) for t, n in (e.split("=", 1) for e in a.exec)}
    build(a.out, weights, a.text, a.team, a.email, not a.no_evaluate, execs)
    print(f"built {a.out}: tasks {sorted(weights)}")
    r = subprocess.run([sys.executable, "-B", "policy_server.py", "--self-test", "--port", "18766"],
                       cwd=a.out, capture_output=True, text=True, timeout=600)
    print(r.stdout[-2000:], r.stderr[-2000:])
    for dirpath, dirnames, _ in os.walk(a.out):
        for d in list(dirnames):
            if d == "__pycache__":
                shutil.rmtree(os.path.join(dirpath, d))
    if r.returncode == 0 and "/health ok — policy: lpolicy\n" not in r.stdout:
        print("BUILD FAILED: lpolicy did not load inside the built server")
        sys.exit(1)
    if r.returncode:
        sys.exit(r.returncode)
    # The template's self-test only exercises grasp_part: also run every packed task's network
    # (official instruction, random images) from the built folder, and time it.
    instr = {}
    import yaml
    for t in {w.split("@")[0] for w in weights}:
        cfg = yaml.safe_load(open(os.path.join(REPO, "eval", "configs", f"{t}.yaml")))
        instr[t] = (cfg["variants"], cfg["perception"]["default_variant"])
    code = f"""
import sys, time, numpy as np
sys.path.insert(0, ".")
from lpolicy.policy import LearnedPolicy
p = LearnedPolicy()
rng = np.random.default_rng(0)
for task, (variants, default) in {instr!r}.items():
    for v, text in variants.items():
        p.reset(task, text, 8)
        assert p.ctx.notes.get("variant") == v, (task, v, p.ctx.notes)
    t0 = time.time()
    for k in range(16):
        a = p.act(rng.uniform(-1, 1, 9).astype(np.float32), {{"hand_camera": rng.integers(0, 255, (512, 512, 3), dtype=np.uint8),
                  "base_camera": rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)}}, "", k)
        assert a.shape == (8,) and np.all(np.isfinite(a)) and abs(a[7]) <= 1, a
    assert "error" not in p.ctx.notes, p.ctx.notes
    print(f"{{task}}: variants read OK, 16 acts in {{time.time() - t0:.2f}} s on {{p.device}}")
"""
    r = subprocess.run([sys.executable, "-B", "-c", code], cwd=a.out, capture_output=True, text=True, timeout=600)
    print(r.stdout[-2000:], r.stderr[-1500:] if r.returncode else "")
    for dirpath, dirnames, _ in os.walk(a.out):
        for d in list(dirnames):
            if d == "__pycache__":
                shutil.rmtree(os.path.join(dirpath, d))
    if r.returncode:
        print("BUILD FAILED: a packed task did not run")
    sys.exit(r.returncode)
