from __future__ import annotations
import torch

def routing_specialization(a, b): return float(0.5 * (a - b).abs().sum().item())
def causal_specialization(c): return float(0.5 * (c[:, 0] - c[:, 1]).abs().sum().item())
def router_entropy(p):
    p = p.clamp_min(1e-8); return float(-(p * p.log()).sum(dim=-1).mean().item())
def perm_matrix_distance(a, b):
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] != 2: raise ValueError("expected (2,n)")
    return min(float(torch.linalg.matrix_norm(a - b)), float(torch.linalg.matrix_norm(a - b.flip(0))))

