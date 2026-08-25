"""Explicit compatibility contracts with the historical DCI pilot."""

from __future__ import annotations

import torch


def make_dci_task(task: str, n: int, seed: int, device: str = "cpu"):
    """Independent copy of the DCI A/B synthetic task contract."""
    if task not in {"A", "B"}:
        raise ValueError(task)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n, 16, generator=generator)
    if task == "A":
        score = x[:, 0] * x[:, 1] + x[:, 2] * x[:, 3]
        indicator = torch.tensor([1.0, 0.0]).expand(n, -1)
    else:
        score = x[:, 8] * x[:, 9] - x[:, 10] * x[:, 11]
        indicator = torch.tensor([0.0, 1.0]).expand(n, -1)
    return torch.cat((x, indicator), dim=1).to(device), (score > 0).float().to(device)


def s_causal_original(causal_effects: torch.Tensor) -> float:
    """Legacy DCI metric for exactly 2 experts x 2 tasks."""
    if tuple(causal_effects.shape) != (2, 2):
        raise ValueError("legacy metric requires a [2, 2] matrix")
    return float(0.5 * (causal_effects[:, 0] - causal_effects[:, 1]).abs().sum().item())
