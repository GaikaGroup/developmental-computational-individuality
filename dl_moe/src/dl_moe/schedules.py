from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class DevelopmentalSchedule:
    early: tuple[str, ...]
    mature: tuple[str, ...]

    @property
    def tasks(self) -> tuple[str, ...]:
        return self.early + self.mature

    def task_at(self, step: int) -> str:
        if step < 0 or step >= len(self.tasks):
            raise IndexError(step)
        return self.tasks[step]


def _balanced(n: int, seed: int) -> tuple[str, ...]:
    if n % 2:
        raise ValueError("balanced phase requires an even number of steps")
    values = ["A" if i % 2 == 0 else "B" for i in range(n)]
    random.Random(seed).shuffle(values)
    return tuple(values)


def make_schedule(name: str, early_steps: int = 4000, total_steps: int = 20000, seed: int = 0) -> DevelopmentalSchedule:
    if name not in {"interleaved", "blocked_ab", "blocked_ba"}:
        raise ValueError(f"unknown developmental schedule: {name}")
    if early_steps <= 0 or early_steps > total_steps or early_steps % 2 or (total_steps - early_steps) % 2:
        raise ValueError("early_steps and mature steps must be positive and even")
    if name == "interleaved":
        early = _balanced(early_steps, seed)
    elif name == "blocked_ab":
        half = early_steps // 2
        early = ("A",) * half + ("B",) * half
    else:
        half = early_steps // 2
        early = ("B",) * half + ("A",) * half
    return DevelopmentalSchedule(tuple(early), _balanced(total_steps - early_steps, seed + 1))
