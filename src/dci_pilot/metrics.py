from __future__ import annotations
import torch

def routing_specialization(a, b): return float(0.5 * (a - b).abs().sum().item())
def causal_specialization(c): return float(0.5 * (c[:, 0] - c[:, 1]).abs().sum().item())
def router_entropy(p):
    p = p.clamp_min(1e-8); return float(-(p * p.log()).sum(dim=-1).mean().item())
def prolonged_router_collapse(history, threshold=.95, min_consecutive=3, min_span_steps=2000):
    streak = []
    for point in history:
        if "routing_A" not in point or "routing_B" not in point: continue
        mean = [(a + b) / 2 for a, b in zip(point["routing_A"], point["routing_B"])]
        expert = max(range(len(mean)), key=mean.__getitem__)
        if mean[expert] > threshold:
            streak = streak + [(point["step"], expert, mean[expert])] if not streak or streak[-1][1] == expert else [(point["step"], expert, mean[expert])]
            if len(streak) >= min_consecutive and streak[-1][0] - streak[0][0] >= min_span_steps:
                return {"collapsed": True, "expert": expert + 1, "start_step": streak[0][0], "end_step": streak[-1][0], "peak_probability": max(x[2] for x in streak)}
        else: streak = []
    return {"collapsed": False, "expert": None, "start_step": None, "end_step": None, "peak_probability": None}
def perm_matrix_distance(a, b):
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] != 2: raise ValueError("expected (2,n)")
    return min(float(torch.linalg.matrix_norm(a - b)), float(torch.linalg.matrix_norm(a - b.flip(0))))
