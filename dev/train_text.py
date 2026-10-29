"""Dev: fine-tune the instruction classifier (lpolicy/text.py) on dev/gen_instructions.py output.

Evaluates on the official instruction strings (eval/configs/*.yaml, must all be right) and on the
226 held-out paraphrases of dev/test_instructions.py, which are never used for training.

  CUDA_VISIBLE_DEVICES=0 python dev/train_text.py data/instructions_train.jsonl runs/text_clf
"""
import argparse
import glob
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.environ.get("METAFINE_REPO") or os.path.join(ROOT, "MetaFine-Challenge_CoRL")  # official repository
sys.path.insert(0, ROOT)
import torch
import torch.nn.functional as F
import yaml

from dev.gen_instructions import TASKS
from lpolicy.text import MAX_TOKENS, InstructionClassifier

BASE = os.path.join(ROOT, "data", "models", "all-MiniLM-L6-v2")


def official():
    rows = []
    for path in sorted(glob.glob(os.path.join(REPO, "eval", "configs", "*.yaml"))):
        cfg = yaml.safe_load(open(path))
        rows += [(cfg["task_id"], text, v) for v, text in (cfg.get("variants") or {}).items()]
    return rows


def held_out():
    from dev.test_instructions import CASES
    seen, rows = set(), []
    for task, text, want in CASES:
        if (task, text) not in seen:
            seen.add((task, text))
            rows.append((task, text, want))
    return rows


@torch.no_grad()
def predict(model, tok, rows, dev):
    model.eval()
    out = []
    for task, text, _ in rows:
        enc = tok([text], truncation=True, max_length=MAX_TOKENS, return_tensors="pt").to(dev)
        out.append(TASKS[task][int(model(task, enc["input_ids"], enc["attention_mask"]).argmax())])
    return out


def report(model, tok, dev, verbose=False):
    res = {}
    for name, rows in (("official", official()), ("held_out", held_out())):
        pred = predict(model, tok, rows, dev)
        wrong = [(t, s, w, p) for (t, s, w), p in zip(rows, pred) if p != w]
        res[name] = f"{len(rows) - len(wrong)}/{len(rows)}"
        if verbose:
            for t, s, w, p in wrong:
                print(f"  WRONG {name} {t:22s} want={w:7s} got={p:7s} | {s}")
    return res


def main():
    from transformers import AutoModel, AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(BASE)
    model = InstructionClassifier(AutoModel.from_pretrained(BASE), TASKS).to(dev)
    rows = [json.loads(line) for line in open(args.data)]
    by_task = {t: [r for r in rows if r["task"] == t] for t in TASKS}
    print({t: len(v) for t, v in by_task.items()}, "before:", report(model, tok, dev), flush=True)
    opt = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": args.lr},
                             {"params": model.heads.parameters(), "lr": 1e-3}], weight_decay=0.01)
    batches = []
    for _ in range(args.epochs):
        for t, rs in by_task.items():
            rs = rs[:]
            random.shuffle(rs)
            batches += [(t, rs[i:i + args.batch]) for i in range(0, len(rs), args.batch)]
    random.shuffle(batches)
    total = len(batches)
    for i, (t, b) in enumerate(batches, 1):
        for g, lr in zip(opt.param_groups, (args.lr, 1e-3)):
            g["lr"] = lr * min(1.0, i / 100) * max(0.0, 1 - i / total)
        model.train()
        enc = tok([r["text"] for r in b], truncation=True, max_length=MAX_TOKENS, padding=True,
                  return_tensors="pt").to(dev)
        y = torch.tensor([TASKS[t].index(r["label"]) for r in b], device=dev)
        loss = F.cross_entropy(model(t, enc["input_ids"], enc["attention_mask"]), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % 200 == 0 or i == total:
            print(f"batch {i}/{total} loss {loss.item():.4f}", report(model, tok, dev), flush=True)
    print("final:", report(model, tok, dev, verbose=True), flush=True)
    model.cpu().save(args.out, tok)
    print("saved", args.out)


if __name__ == "__main__":
    main()
