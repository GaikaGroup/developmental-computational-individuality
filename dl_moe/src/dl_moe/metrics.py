from __future__ import annotations

from itertools import permutations
import numpy as np
import torch


def causal_specialization_general(causal_effects: torch.Tensor) -> float:
    """Mean per-expert mean absolute deviation across task effects."""
    if causal_effects.ndim != 2 or causal_effects.shape[1] < 2:
        raise ValueError("expected [experts, tasks] matrix with at least two tasks")
    return float((causal_effects - causal_effects.mean(dim=1, keepdim=True)).abs().mean().item())


def s_expert_causal(causal_effects: torch.Tensor) -> float:
    """Preregistered DL-MoE-01 endpoint: mean absolute A/B contrast per expert."""
    if causal_effects.ndim != 2 or causal_effects.shape[1] != 2:
        raise ValueError("s_expert_causal requires [experts, 2] effects")
    return float((causal_effects[:, 0] - causal_effects[:, 1]).abs().mean().item())


def population_specialization(causal_effects: torch.Tensor) -> float:
    if tuple(causal_effects.shape) != (2, 2):
        raise ValueError("population metric requires [2, 2]")
    return float(0.5 * (causal_effects[:, 0] - causal_effects[:, 1]).abs().sum().item())


def routing_entropy(probabilities: torch.Tensor) -> float:
    p = probabilities.clamp_min(1e-8)
    return float(-(p * p.log()).sum(dim=-1).mean().item())


def _best_assignment(a: np.ndarray, b: np.ndarray, allow_swap: bool) -> tuple[float, tuple[int, ...]]:
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("causal profile matrices must have equal 2-D shapes")
    best = (float("inf"), tuple(range(len(b))))
    for order in permutations(range(len(b))):
        if not allow_swap and len(order) > 0 and order != tuple(range(len(b))):
            continue
        distance = float(np.linalg.norm(a - b[list(order)]))
        if distance < best[0]:
            best = (distance, order)
    return best


def organizational_distance(
    a: torch.Tensor,
    b: torch.Tensor,
    experts_per_population: int | None = None,
    population_swap_allowed: bool = True,
    mode: str = "global",
) -> float:
    """Normalized Frobenius distance after allowed expert/population alignment."""
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("organizational matrices must have equal 2-D shapes")
    if mode not in {"global", "within_population"}:
        raise ValueError(mode)
    aa, bb = a.detach().cpu().numpy(), b.detach().cpu().numpy()
    if mode == "global" or experts_per_population is None:
        distance, _ = _best_assignment(aa, bb, allow_swap=True)
    else:
        k = experts_per_population
        if len(aa) != 2 * k:
            raise ValueError("within_population alignment expects two equal populations")
        candidates = []
        for swap in (False, True) if population_swap_allowed else (False,):
            order = list(range(k, 2 * k)) + list(range(0, k)) if swap else list(range(2 * k))
            left = aa.reshape(2, k, -1)
            right = bb[order].reshape(2, k, -1)
            d = sum(_best_assignment(left[p], right[p], True)[0] ** 2 for p in range(2)) ** 0.5
            candidates.append(d)
        distance = min(candidates)
    return float(distance / max(1.0, np.sqrt(a.numel())))
