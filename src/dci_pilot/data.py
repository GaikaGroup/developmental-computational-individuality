from __future__ import annotations
import torch

def make_task(task: str, n: int, seed: int, device: str = "cpu"):
    if task not in ("A", "B"): raise ValueError(task)
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n, 16, generator=g)
    if task == "A":
        score = x[:, 0] * x[:, 1] + x[:, 2] * x[:, 3]
        indicator = torch.tensor([1.0, 0.0]).expand(n, -1)
    else:
        score = x[:, 8] * x[:, 9] - x[:, 10] * x[:, 11]
        indicator = torch.tensor([0.0, 1.0]).expand(n, -1)
    return torch.cat((x, indicator), dim=1).to(device), (score > 0).float().to(device)

class BatchStream:
    def __init__(self, seed: int, batch_size: int, device: str = "cpu"):
        self.seed, self.batch_size, self.device = seed, batch_size, device
        self.counts = {"A": 0, "B": 0}
    def next(self, task: str):
        ordinal = self.counts[task]; self.counts[task] += 1
        offset = 0 if task == "A" else 1_000_003
        return make_task(task, self.batch_size, self.seed + offset + ordinal, self.device)

