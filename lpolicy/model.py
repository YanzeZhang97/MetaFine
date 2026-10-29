"""ACT-style visuomotor policy (no CVAE): hand-camera image + qpos + variant code -> action chunk.

- ResNet18 trunk; FiLM from the variant embedding after layers 2, 3 and 4, so the instruction
  can steer what the visual features pick out (which slider, which cube, which letter).
- Transformer encoder over the 7x7 image tokens, a qpos token and a variant token; decoder with
  CHUNK learned queries -> CHUNK x (7 arm deltas, trained on targets in [-1, 1], + 1 gripper logit).
- Optional auxiliary head (training only) regressing ground-truth targets from the variant token.
- Optional goal path: a head predicts, from the variant token, where the instructed object is
  (normalised xyz); its embedding is appended to the decoder memory and added to every query, so
  the actions are conditioned on "where to go". Training supervises it under every variant code.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .resnet import ResNet18

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FiLM(nn.Module):
    def __init__(self, d_cond: int, channels: int):
        super().__init__()
        self.lin = nn.Linear(d_cond, 2 * channels)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x, cond):
        g, b = self.lin(cond).chunk(2, dim=-1)
        return x * (1 + g[:, :, None, None]) + b[:, :, None, None]


class ACTPolicy(nn.Module):
    def __init__(self, n_variants: int, chunk: int = 32, d: int = 256, heads: int = 8, enc_layers: int = 4,
                 dec_layers: int = 4, ff: int = 1024, dropout: float = 0.1, aux_dim: int = 0,
                 qpos_dim: int = 9, goal_dim: int = 0, n_img: int = 1, pretrained: str | None = None):
        super().__init__()
        self.cfg = dict(n_variants=n_variants, chunk=chunk, d=d, heads=heads, enc_layers=enc_layers,
                        dec_layers=dec_layers, ff=ff, dropout=dropout, aux_dim=aux_dim, qpos_dim=qpos_dim,
                        goal_dim=goal_dim, n_img=n_img)
        self.n_img = n_img
        r = ResNet18()
        if pretrained:
            r.load_imagenet(pretrained)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1)
        self.layer2, self.layer3, self.layer4 = r.layer2, r.layer3, r.layer4
        self.var_emb = nn.Embedding(n_variants, d)
        self.film2, self.film3, self.film4 = FiLM(d, 128), FiLM(d, 256), FiLM(d, 512)
        self.proj = nn.Conv2d(512, d, 1)
        self.qpos_proj = nn.Linear(qpos_dim, d)
        self.pos = nn.Parameter(torch.zeros(1, 49 * n_img + 2, d))
        nn.init.normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True, activation="gelu")
        dec = nn.TransformerDecoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, enc_layers, enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(dec, dec_layers)
        self.enc_norm, self.dec_norm = nn.LayerNorm(d), nn.LayerNorm(d)
        self.queries = nn.Parameter(torch.zeros(1, chunk, d))
        nn.init.normal_(self.queries, std=0.02)
        self.arm_head = nn.Linear(d, 7)
        self.grip_head = nn.Linear(d, 1)
        self.aux_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, aux_dim)) if aux_dim else None
        self.goal_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, goal_dim)) if goal_dim else None
        self.goal_emb = nn.Sequential(nn.Linear(goal_dim, d), nn.GELU(), nn.Linear(d, d)) if goal_dim else None
        self.register_buffer("img_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        # qpos normalisation (set from the training data, saved with the weights)
        self.register_buffer("q_mean", torch.zeros(qpos_dim))
        self.register_buffer("q_std", torch.ones(qpos_dim))

    def forward(self, img_u8, qpos, variant, crop_u8=None):
        """img_u8: B x 3 x 224 x 224 uint8; qpos: B x 9; variant: B (long); crop_u8: the native-resolution
        centre crop (B x 3 x 224 x 224) when the model has two image streams (same trunk, own positions).
        Returns arm (B x chunk x 7, linear; clip to [-1, 1] when acting), gripper logits (B x chunk),
        aux (B x aux_dim or None), goal (B x goal_dim, normalised, or None)."""
        c = self.var_emb(variant)
        imgs = [img_u8] if self.n_img == 1 else [img_u8, crop_u8]
        x = (torch.cat(imgs, 0).float() / 255.0 - self.img_mean) / self.img_std
        cc = c.repeat(len(imgs), 1)
        x = self.stem(x)
        x = self.film2(self.layer2(x), cc)
        x = self.film3(self.layer3(x), cc)
        x = self.film4(self.layer4(x), cc)
        tok = self.proj(x).flatten(2).transpose(1, 2)                      # (n_img B) x 49 x d
        tok = torch.cat(tok.chunk(len(imgs), 0), dim=1)                     # B x 49 n_img x d
        q = self.qpos_proj((qpos - self.q_mean) / self.q_std)[:, None]      # B x 1 x d
        mem = torch.cat([c[:, None], q, tok], dim=1) + self.pos
        mem = self.enc_norm(self.encoder(mem))
        queries = self.queries.expand(mem.shape[0], -1, -1)
        goal = None
        if self.goal_head is not None:
            goal = self.goal_head(mem[:, 0])
            g = self.goal_emb(goal)
            mem = torch.cat([mem, g[:, None]], dim=1)
            queries = queries + g[:, None]
        h = self.dec_norm(self.decoder(queries, mem))
        arm = self.arm_head(h)
        grip = self.grip_head(h)[..., 0]
        aux = self.aux_head(mem[:, 0]) if self.aux_head is not None else None
        return arm, grip, aux, goal
