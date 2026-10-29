"""Dev: train one ACT policy (lpolicy/model.py) per task on teacher demos from dev/gen_demos.py.

  CUDA_VISIBLE_DEVICES=0 python dev/train_act.py TASK DATA_DIR RUN_DIR [--steps 40000] [--batch 64]

The whole dataset stays in RAM as uint8 (about 150 KB per frame). Batches are gathered on the CPU
by a prefetch thread and augmented on the GPU (random shift up to 8 px). Episodes whose seed is a
multiple of 20 are held out for the validation loss. EMA weights are what gets saved.
Chunk targets past the end of an episode repeat its last action (the teacher's hold).
"""
import argparse
import copy
import glob
import json
import math
import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import torch
import torch.nn.functional as F

from dev.simenv import load_cfg
from lpolicy.model import ACTPolicy

RESNET18 = os.path.expanduser("~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")


def load_demos(task, data_dir, crop=False, mask_release=False, img_every=1, img_every_ramp=None, window_prefix=(),
               release_window=24):
    """Every frame keeps its qpos / action / aux; training samples are the frames that have an image
    (all of them, unless the demos were rendered with --img-stride) and a label. Images of unlabelled frames are not
    loaded. img_every: keep only every K-th stored image of each episode (actions stay dense), to fit more episodes in
    memory; img_every_ramp: the same for RELEASE_RAMP episodes (default img_every).
    mask_release: in episodes whose teacher opened the gripper at once (no RELEASE_RAMP in meta["set"]), mask the
    frames from the last closed -> open switch to the end, so that they teach neither the instant open nor what
    follows it; the RELEASE_RAMP episodes then alone show how to let go.
    window_prefix: RELEASE_RAMP episodes whose file name starts with one of these prefixes teach only the release:
    every frame before (ramp start - release_window) is masked."""
    variants = list(load_cfg(task)["variants"])
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    every_ramp = img_every if img_every_ramp is None else img_every_ramp

    def labels(path, d, meta, n):
        """Per-frame mask (True = no label)."""
        m = np.asarray(d["mask"], dtype=bool).copy() if "mask" in d.files else np.zeros(n, dtype=bool)
        ramp = any(str(x).startswith("RELEASE_RAMP=") for x in meta.get("set", []))
        g = d["action"][:, 7]
        if mask_release and not ramp:
            opens = np.flatnonzero((g[:-1] < -0.5) & (g[1:] > 0.5)) + 1
            if len(opens):
                m[opens[-1]:] = True
        if ramp and window_prefix and os.path.basename(path).startswith(tuple(window_prefix)):
            # the ramp starts where a closed gripper first takes an intermediate value (last such switch)
            r = np.flatnonzero((g[:-1] <= -0.9) & (g[1:] > -0.9) & (g[1:] < 0.9)) + 1
            if len(r):
                m[:max(0, int(r[-1]) - release_window)] = True
        return m, ramp

    def head(path):
        try:
            with np.load(path) as d:
                meta = json.loads(str(d["meta"]))
                n = len(d["action"])
                img_t = d["img_t"] if "img_t" in d.files else np.arange(n)
                m, ramp = labels(path, d, meta, n)
                idx = np.arange(len(img_t))[::every_ramp if ramp else img_every]
                idx = idx[~m[img_t[idx]]]
                return n, idx, img_t[idx], meta, m
        except Exception:  # unreadable (e.g. still being written): skip
            return None

    with ThreadPoolExecutor(16) as ex:
        heads = list(ex.map(head, files))
    files = [f for f, h in zip(files, heads) if h is not None]
    heads = [h for h in heads if h is not None]
    lens = np.array([h[0] for h in heads])
    nimg = np.array([len(h[1]) for h in heads])
    N, M = int(lens.sum()), int(nimg.sum())
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    istarts = np.concatenate([[0], np.cumsum(nimg)[:-1]])
    with np.load(files[0]) as d:
        aux_dim = d["aux"].shape[1]
    n_var = len(variants)
    hand = np.empty((M, 224, 224, 3), dtype=np.uint8)
    crops = np.empty((M, 224, 224, 3), dtype=np.uint8) if crop else None
    frame = np.empty(M, dtype=np.int64)       # sample -> frame
    val = np.empty(M, dtype=bool)
    qpos = np.empty((N, 9), dtype=np.float32)
    action = np.empty((N, 8), dtype=np.float32)
    aux = np.empty((N, aux_dim), dtype=np.float32)
    goal = np.empty((N, n_var, 3), dtype=np.float32)
    var = np.empty(N, dtype=np.int64)
    ep_end = np.empty(N, dtype=np.int64)
    mask = np.zeros(N, dtype=bool)  # deliberately wrong teacher actions (dev/teacher.py): no labels

    def fill(i):
        s, n, si, m = starts[i], lens[i], istarts[i], nimg[i]
        idx, img_t, meta, lab = heads[i][1], heads[i][2], heads[i][3], heads[i][4]
        with np.load(files[i]) as d:
            hand[si:si + m] = d["hand"][idx]
            if crop:
                crops[si:si + m] = d["crop"][idx]
            qpos[s:s + n] = d["qpos"]
            action[s:s + n] = d["action"]
            mask[s:s + n] = lab
            aux[s:s + n] = d["aux"]
            if "goal" in d.files:
                goal[s:s + n] = d["goal"]
            elif task == "toggle_switch_table":
                goal[s:s + n] = t3_goals(d["aux"], meta["variant"], variants)
            elif task == "grasp_move_mug":
                goal[s:s + n] = t2_goals(d["aux"], variants)
            else:
                raise KeyError(f"{files[i]} has no per-variant goals")
        frame[si:si + m] = s + img_t
        val[si:si + m] = meta["seed"] % 20 == 0
        var[s:s + n] = variants.index(meta["variant"])
        ep_end[s:s + n] = s + n - 1

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(fill, range(len(files))))
    kinds = {h[3].get("crop") or "center" for h in heads}
    if crop and len(kinds) != 1:
        raise ValueError(f"demos mix second-stream crops {kinds}")
    return dict(hand=hand, crop=crops, frame=frame, val=val, qpos=qpos, action=action, aux=aux, goal=goal, var=var,
                ep_end=ep_end, mask=mask, variants=variants, n_episodes=len(files),
                crop_kind=kinds.pop() if crop else None)


def t2_goals(aux, variants):
    """Per-direction carry end points for T2 demos recorded before goals were stored (same formula as
    dev/teacher.py: the handle pinch point, 40 cm up, moved 20 cm along the direction)."""
    from mfagent.tasks import grasp_move_mug as gm
    base = np.stack([aux[:, 4], aux[:, 5], np.full(len(aux), 0.40)], axis=1)
    return np.stack([base + 0.20 * gm.DIRECTION[v] for v in variants], axis=1).astype(np.float32)


def t3_goals(aux, variant, variants):
    """Per-variant push contacts for T3 demos recorded before goals were stored: the aux holds the
    switch root (x, y, cos, sin) and the instructed slider's contact xy, which tells the link layout."""
    from mfagent.tasks import toggle_switch as ts
    out = np.empty((len(aux), len(variants), 3), dtype=np.float32)
    for i, a in enumerate(aux):
        R = ts.root_pose(a[0], a[1], float(np.arctan2(a[3], a[2])))
        c = [(R @ ts._orthonormal(ts.GRASP_LOCAL[k]))[:3, 3] for k in (0, 1)]
        inst = int(np.argmin([np.hypot(*(ck[:2] - a[4:6])) for ck in c]))
        blue = inst if variant == "blue" else 1 - inst
        for j, v in enumerate(variants):
            out[i, j] = c[blue if v == "blue" else 1 - blue]
    return out


def random_shift(x, pad=8):
    """DrQ-style shift: replicate-pad then crop back at a random integer offset, per sample."""
    n, _, h, w = x.shape
    x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    eps = 1.0 / (h + 2 * pad)
    ar = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * pad, device=x.device)[:h]
    ar = ar[:, None].repeat(1, h)[..., None]
    base = torch.cat([ar.transpose(1, 0), ar], dim=2)[None].repeat(n, 1, 1, 1)
    shift = torch.randint(0, 2 * pad + 1, (n, 1, 1, 2), device=x.device, dtype=x.dtype) * (2.0 / (h + 2 * pad))
    return F.grid_sample(x, base + shift, padding_mode="zeros", align_corners=False)


class Batches:
    """Background thread gathering pinned CPU batches."""

    def __init__(self, D, idx_pool, batch, chunk, seed):
        self.D, self.pool, self.batch, self.chunk = D, idx_pool, batch, chunk
        self.rng = np.random.default_rng(seed)
        self.q = queue.Queue(maxsize=6)
        threading.Thread(target=self._run, daemon=True).start()

    def make(self, idx):
        D = self.D
        f = D["frame"][idx]
        t = np.minimum(f[:, None] + np.arange(self.chunk)[None], D["ep_end"][f][:, None])
        pin = (lambda x: x.pin_memory()) if torch.cuda.is_available() else (lambda x: x)
        b = {} if D["crop"] is None else {"crop": pin(torch.from_numpy(D["crop"][idx]))}
        return {**b, "img": pin(torch.from_numpy(D["hand"][idx])),
                "qpos": pin(torch.from_numpy(D["qpos"][f])),
                "var": pin(torch.from_numpy(D["var"][f])),
                "act": pin(torch.from_numpy(D["action"][t])),
                "mask": pin(torch.from_numpy(D["mask"][t])),
                "aux": pin(torch.from_numpy(D["aux"][f])),
                "goal": pin(torch.from_numpy(D["goal"][f]))}

    def _run(self):
        while True:
            self.q.put(self.make(self.rng.choice(self.pool, self.batch)))

    def get(self):
        return self.q.get()


def to_dev(b, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in b.items()}


def losses(model, b, aux_mean, aux_std, goal_mean, goal_std, train, grip_mode="bce", p_swap=0.0):
    """Action / gripper / aux losses on the true variant. With a goal head, a share p_swap of the samples
    is run under another variant code; those only train the goal head (its target is where that
    variant's object is in the same image), the rest train everything."""
    img = b["img"].permute(0, 3, 1, 2)
    crop = b["crop"].permute(0, 3, 1, 2) if "crop" in b else None
    if train:
        img = random_shift(img.float()).round().clamp(0, 255).to(torch.uint8)
        if crop is not None:
            crop = random_shift(crop.float()).round().clamp(0, 255).to(torch.uint8)
    if img.device.type == "cpu":  # oneDNN convolutions run ~20 % faster on channels-last tensors
        img = img.contiguous(memory_format=torch.channels_last)
        crop = None if crop is None else crop.contiguous(memory_format=torch.channels_last)
    v_true = b["var"]
    n_var = b["goal"].shape[1]
    v_in = v_true
    if p_swap > 0 and n_var > 1:
        other = (v_true + torch.randint(1, n_var, v_true.shape, device=v_true.device)) % n_var
        v_in = torch.where(torch.rand(v_true.shape, device=v_true.device) < p_swap, other, v_true)
    keep = (v_in == v_true).float()
    with torch.autocast(img.device.type, dtype=torch.bfloat16):
        arm, grip, aux, goal = model(img, b["qpos"], v_in, crop)
    arm, grip = arm.float(), grip.float()
    w = keep / keep.sum().clamp(min=1.0)
    m = 1.0 - b["mask"].float()  # chunk steps with a label
    m = m / m.sum(1, keepdim=True).clamp(min=1.0)
    l_arm = (((arm - b["act"][..., :7]).abs().mean(2) * m).sum(1) * w).sum()
    if grip_mode == "reg":  # continuous openings (T5 pre-grasp widths): L1 on tanh
        l_grip = (((torch.tanh(grip) - b["act"][..., 7]).abs() * m).sum(1) * w).sum()
    else:
        l_grip = ((F.binary_cross_entropy_with_logits(grip, (b["act"][..., 7] > 0).float(), reduction="none")
                   * m).sum(1) * w).sum()
    l_aux = (((aux.float() - (b["aux"] - aux_mean) / aux_std) ** 2).mean(1) * w).sum() if aux is not None \
        else torch.zeros((), device=arm.device)
    if goal is not None:
        tgt = b["goal"][torch.arange(len(v_in), device=v_in.device), v_in]
        l_goal = F.mse_loss(goal.float(), (tgt - goal_mean) / goal_std)
    else:
        l_goal = torch.zeros((), device=arm.device)
    grip_acc = (((((grip > 0) == (b["act"][..., 7] > 0)).float() * m).sum(1)) * w).sum()
    return l_arm, l_grip, l_aux, l_goal, grip_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("data")
    ap.add_argument("run")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--w-grip", type=float, default=0.5)
    ap.add_argument("--w-aux", type=float, default=0.1)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--goal", action="store_true", help="goal path: predict the instructed object's position")
    ap.add_argument("--crop", action="store_true", help="second image stream: native-resolution centre crop")
    ap.add_argument("--w-goal", type=float, default=1.0)
    ap.add_argument("--p-swap", type=float, default=0.3, help="share of samples run under another variant code")
    ap.add_argument("--grip-mode", choices=["bce", "reg"], default="bce",
                    help="bce: gripper is +-1 only; reg: continuous gripper targets (T5)")
    ap.add_argument("--device", default="cuda", help="cuda, or cpu (bf16 autocast on the CPU; set "
                    "OMP_NUM_THREADS for the thread count) when no GPU memory is free")
    ap.add_argument("--init", default=None, help="start from this checkpoint's (EMA) weights, e.g. runs/t5_g/last.pt "
                    "(same model config; its qpos normalisation is kept)")
    ap.add_argument("--mask-release", action="store_true", help="mask the instant-open releases (and what follows) "
                    "of episodes without RELEASE_RAMP; see load_demos")
    ap.add_argument("--img-every", type=int, default=1, help="use every K-th stored image of each episode")
    ap.add_argument("--img-every-ramp", type=int, default=None, help="the same for RELEASE_RAMP episodes")
    ap.add_argument("--window-prefix", default="", help="comma-separated file-name prefixes of RELEASE_RAMP episodes "
                    "that teach only the release (see load_demos)")
    ap.add_argument("--release-window", type=int, default=24, help="frames before the ramp start kept by --window-prefix")
    args = ap.parse_args()
    os.makedirs(args.run, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    dev = torch.device(args.device)

    t0 = time.time()
    D = load_demos(args.task, args.data, crop=args.crop, mask_release=args.mask_release, img_every=args.img_every,
                   img_every_ramp=args.img_every_ramp, window_prefix=tuple(p for p in args.window_prefix.split(",") if p),
                   release_window=args.release_window)
    labelled = ~D["mask"][D["frame"]]  # images whose own action is a deliberate error are not samples
    train_idx, val_idx = np.flatnonzero(~D["val"] & labelled), np.flatnonzero(D["val"] & labelled)
    print(f"loaded {D['n_episodes']} episodes, {len(D['action'])} frames, {len(D['hand'])} images "
          f"({len(train_idx)} train / {len(val_idx)} val) in {time.time() - t0:.0f} s; variants {D['variants']}",
          flush=True)
    train_f = D["frame"][train_idx]
    aux_mean = torch.tensor(D["aux"][train_f].mean(0), device=dev)
    aux_std = torch.tensor(D["aux"][train_f].std(0) + 1e-3, device=dev)
    g = D["goal"][train_f].reshape(-1, 3)
    goal_mean = torch.tensor(g.mean(0), device=dev)
    goal_std = torch.tensor(g.std(0) + 1e-3, device=dev)

    model = ACTPolicy(len(D["variants"]), chunk=args.chunk, aux_dim=D["aux"].shape[1],
                      goal_dim=3 if args.goal else 0, n_img=2 if args.crop else 1, pretrained=RESNET18)
    model.q_mean.copy_(torch.tensor(D["qpos"][train_f].mean(0)))
    model.q_std.copy_(torch.tensor(D["qpos"][train_f].std(0) + 1e-3))
    if args.init:
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        if ck["model_cfg"] != model.cfg:
            raise ValueError(f"--init model config {ck['model_cfg']} != {model.cfg}")
        model.load_state_dict(ck["state_dict"])
        print(f"initialised from {args.init}", flush=True)
    model.to(dev)
    if dev.type == "cpu":
        model.to(memory_format=torch.channels_last)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    backbone = [p for n, p in model.named_parameters() if n.startswith(("stem", "layer"))]
    rest = [p for n, p in model.named_parameters() if not n.startswith(("stem", "layer"))]
    opt = torch.optim.AdamW([{"params": backbone, "lr": args.lr_backbone}, {"params": rest, "lr": args.lr}],
                            weight_decay=1e-4)
    base_lrs = [args.lr_backbone, args.lr]
    warm = 1000
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M", flush=True)

    train_b = Batches(D, train_idx, args.batch, args.chunk, seed=0)
    val_b = Batches(D, val_idx, args.batch, args.chunk, seed=1) if len(val_idx) else None
    log = open(os.path.join(args.run, "train.jsonl"), "a")
    meta = {"task": args.task, "variants": D["variants"], "model_cfg": model.cfg, "args": vars(args),
            "grip_mode": args.grip_mode, "crop": D["crop_kind"],
            "goal_mean": goal_mean.tolist(), "goal_std": goal_std.tolist(),
            "aux_mean": aux_mean.tolist(), "aux_std": aux_std.tolist(), "n_episodes": D["n_episodes"]}
    json.dump(meta, open(os.path.join(args.run, "meta.json"), "w"), indent=1)

    def save(tag):
        torch.save({"state_dict": ema.state_dict(), **meta}, os.path.join(args.run, f"{tag}.pt"))

    t0, acc = time.time(), []
    for step in range(1, args.steps + 1):
        scale = min(1.0, step / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / args.steps))))
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * scale
        model.train()
        l_arm, l_grip, l_aux, l_goal, g_acc = losses(model, to_dev(train_b.get(), dev), aux_mean, aux_std,
                                                     goal_mean, goal_std, True, args.grip_mode,
                                                     args.p_swap if args.goal else 0.0)
        loss = l_arm + args.w_grip * l_grip + args.w_aux * l_aux + args.w_goal * l_goal
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - args.ema)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)
        acc.append([l_arm.item(), l_grip.item(), l_aux.item(), l_goal.item(), g_acc.item()])
        if step % 250 == 0:
            row = {"step": step, "s": round(time.time() - t0), "lr": opt.param_groups[1]["lr"],
                   **dict(zip(["arm", "grip", "aux", "goal", "grip_acc"], np.mean(acc, 0).round(4).tolist()))}
            acc = []
            if step % 1000 == 0 and val_b is not None:
                with torch.no_grad():
                    v = np.mean([[x.item() for x in losses(ema, to_dev(val_b.get(), dev), aux_mean, aux_std, goal_mean,
                                                           goal_std, False, args.grip_mode,
                                                           0.5 if args.goal else 0.0)]
                                 for _ in range(16)], 0)
                row.update(dict(zip(["val_arm", "val_grip", "val_aux", "val_goal", "val_grip_acc"], v.round(4).tolist())))
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
        if step % args.save_every == 0 or step == args.steps:
            save(f"step{step}")
            save("last")
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
