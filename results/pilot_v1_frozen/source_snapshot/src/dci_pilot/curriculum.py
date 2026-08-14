from __future__ import annotations
from dataclasses import dataclass
import random

@dataclass(frozen=True)
class Curriculum:
    early: tuple[str, ...]
    mature: tuple[str, ...]
    @property
    def tasks(self): return self.early + self.mature

def _balanced(n: int, seed: int):
    values = ["A" if i % 2 == 0 else "B" for i in range(n)]
    random.Random(seed).shuffle(values)
    return tuple(values)

def build_curriculum(condition: str, early_steps=4000, steps=20000, seed=0):
    if condition not in ("INTERLEAVED", "BLOCKED_AB", "BLOCKED_BA"): raise ValueError(condition)
    if early_steps % 2 or steps < early_steps: raise ValueError("invalid step counts")
    half = early_steps // 2
    early = _balanced(early_steps, seed) if condition == "INTERLEAVED" else (("A",) * half + ("B",) * half if condition == "BLOCKED_AB" else ("B",) * half + ("A",) * half)
    return Curriculum(tuple(early), _balanced(steps - early_steps, seed + 1))

