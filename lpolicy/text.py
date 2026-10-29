"""Learned instruction classifier: instruction text -> the task's variant.

A small pretrained sentence encoder (all-MiniLM-L6-v2, fine-tuned) with mean pooling and one
linear head per task. It runs once per episode, at reset. Unusable input (not text, empty,
or a task without a head) gets the task's first variant; nothing here raises.
"""
from __future__ import annotations

import json
import os

import torch
import torch.nn as nn

MAX_TOKENS = 64


class InstructionClassifier(nn.Module):
    def __init__(self, encoder, tasks: dict):
        super().__init__()
        self.encoder = encoder
        self.tasks = {t: list(v) for t, v in tasks.items()}
        d = encoder.config.hidden_size
        self.heads = nn.ModuleDict({t: nn.Linear(d, len(v)) for t, v in self.tasks.items()})

    def embed(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask[..., None].to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp(min=1.0)

    def forward(self, task: str, input_ids, attention_mask):
        return self.heads[task](self.embed(input_ids, attention_mask))

    def save(self, out_dir: str, tokenizer) -> None:
        os.makedirs(out_dir, exist_ok=True)
        self.encoder.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        torch.save({t: h.state_dict() for t, h in self.heads.items()}, os.path.join(out_dir, "heads.pt"))
        json.dump(self.tasks, open(os.path.join(out_dir, "tasks.json"), "w"), indent=1)


class InstructionReader:
    """Runtime wrapper: load once, then ``variant(task_id, text)``."""

    def __init__(self, path: str, device: str = "cpu"):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path)
        enc = AutoModel.from_pretrained(path)
        tasks = json.load(open(os.path.join(path, "tasks.json")))
        self.model = InstructionClassifier(enc, tasks)
        heads = torch.load(os.path.join(path, "heads.pt"), map_location="cpu")
        for t, sd in heads.items():
            self.model.heads[t].load_state_dict(sd)
        self.model.eval().to(device)
        self.device = device
        self.cache: dict = {}

    @torch.inference_mode()
    def probs(self, task_id: str, text: str):
        enc = self.tok([text], truncation=True, max_length=MAX_TOKENS, return_tensors="pt").to(self.device)
        return torch.softmax(self.model(task_id, enc["input_ids"], enc["attention_mask"]), -1)[0].cpu()

    def variant(self, task_id: str, text) -> str | None:
        try:
            names = self.model.tasks.get(str(task_id))
            if not names:
                return None
            if not isinstance(text, str) or not text.strip():
                return names[0]
            key = (str(task_id), text[:2000])
            if key not in self.cache:
                if len(self.cache) > 1000:
                    self.cache.clear()
                self.cache[key] = names[int(self.probs(key[0], key[1]).argmax())]
            return self.cache[key]
        except Exception:
            return None
