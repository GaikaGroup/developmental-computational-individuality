from __future__ import annotations
import torch
from torch import nn

class TinySoftMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(18, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU())
        self.router = nn.Linear(128, 2)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 128)) for _ in range(2)])
        self.norm, self.head = nn.LayerNorm(128), nn.Linear(128, 1)
    def forward(self, x, knockout=None, return_router=False, knockout_mode="renormalized"):
        h = self.encoder(x); p = self.router(h).softmax(dim=-1)
        if knockout is None: weights = p
        else:
            if knockout not in (0, 1): raise ValueError(knockout)
            if knockout_mode not in ("renormalized", "non_renormalized"): raise ValueError(knockout_mode)
            weights = p.clone(); weights[:, knockout] = 0.0
            if knockout_mode == "renormalized": weights[:, 1 - knockout] = 1.0
        m = sum(weights[:, e:e + 1] * self.experts[e](h) for e in range(2))
        logits = self.head(self.norm(h + m)).squeeze(-1)
        return (logits, p) if return_router else logits

def parameter_count(model): return sum(p.numel() for p in model.parameters())
