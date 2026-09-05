from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class DLMoEConfig:
    input_dim: int = 18
    model_dim: int = 128
    num_populations: int = 2
    experts_per_population: int = 4
    expert_hidden_dim: int = 256
    comm_dim: int = 16
    routing_mode: str = "soft"
    top_k: int = 1
    communication_enabled: bool = True
    communication_scale: float = 1.0
    learned_communication_gate: bool = True
    dropout: float = 0.0
    output_dim: int = 1

    def __post_init__(self) -> None:
        if self.num_populations != 2:
            raise ValueError("v1.0 requires exactly two symmetric populations")
        if self.input_dim < 1 or self.model_dim < 1 or self.experts_per_population < 1:
            raise ValueError("dimensions and expert count must be positive")
        if not 0 <= self.comm_dim <= self.model_dim:
            raise ValueError("comm_dim must be in [0, model_dim]")
        if self.routing_mode not in {"soft", "top1", "top2"}:
            raise ValueError(f"unknown routing mode: {self.routing_mode}")
        if self.routing_mode == "top1" and self.top_k != 1:
            raise ValueError("top1 routing requires top_k=1")
        if self.routing_mode == "top2" and self.top_k != 2:
            raise ValueError("top2 routing requires top_k=2")
        if self.routing_mode == "soft" and self.top_k not in {1, 2}:
            raise ValueError("soft routing top_k is unused and must be 1 or 2")
        if not 0 <= self.communication_scale <= 1:
            raise ValueError("communication_scale must be between 0 and 1")


@dataclass(frozen=True)
class Intervention:
    disabled_experts: tuple[tuple[int, int], ...] = ()
    disabled_populations: tuple[int, ...] = ()
    disable_comm_0_to_1: bool = False
    disable_comm_1_to_0: bool = False
    routing_mode_override: Optional[str] = None
    uniform_population_routing: bool = False
    uniform_expert_routing: bool = False
    renormalize: bool = True

    def __post_init__(self) -> None:
        if any(p not in (0, 1) for p in self.disabled_populations):
            raise ValueError("population index must be 0 or 1")
        if any(p not in (0, 1) or e < 0 for p, e in self.disabled_experts):
            raise ValueError("expert index must be (population, non-negative expert)")
        if self.routing_mode_override not in {None, "soft", "top1", "top2"}:
            raise ValueError("invalid routing_mode_override")


@dataclass
class DLMoEOutput:
    logits: Tensor
    population_probs: Tensor
    expert_probs: Tensor
    expert_outputs: Optional[Tensor] = None
    population_outputs: Optional[Tensor] = None
    communication_messages: Optional[Tensor] = None
    communication_gates: Optional[Tensor] = None


class _Expert(nn.Module):
    def __init__(self, cfg: DLMoEConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.model_dim, cfg.expert_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.expert_hidden_dim, cfg.model_dim),
        )

    def forward(self, h: Tensor) -> Tensor:
        return self.net(h)


class DLMoE(nn.Module):
    """Symmetric two-population hierarchical Mixture-of-Experts model."""

    def __init__(self, config: DLMoEConfig = DLMoEConfig()) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, config.model_dim),
            nn.LayerNorm(config.model_dim),
        )
        self.population_router = nn.Linear(config.model_dim, config.num_populations)
        self.expert_routers = nn.ModuleList(
            [nn.Linear(config.model_dim, config.experts_per_population) for _ in range(config.num_populations)]
        )
        self.experts = nn.ModuleList(
            [nn.ModuleList([_Expert(config) for _ in range(config.experts_per_population)]) for _ in range(config.num_populations)]
        )
        if config.comm_dim:
            self.comm_down = nn.ModuleList([nn.Linear(config.model_dim, config.comm_dim) for _ in range(2)])
            self.comm_up = nn.ModuleList([nn.Linear(config.comm_dim, config.model_dim) for _ in range(2)])
        else:
            self.comm_down = nn.ModuleList([nn.Identity(), nn.Identity()])
            self.comm_up = nn.ModuleList([nn.Identity(), nn.Identity()])
        self.communication_gate = nn.Parameter(torch.zeros(2)) if config.learned_communication_gate else None
        self.population_norms = nn.ModuleList([nn.LayerNorm(config.model_dim) for _ in range(2)])
        self.output_norm = nn.LayerNorm(config.model_dim)
        self.head = nn.Linear(config.model_dim, config.output_dim)

    @property
    def total_experts(self) -> int:
        return self.config.num_populations * self.config.experts_per_population

    def _route(self, logits: Tensor, mode: str, k: int) -> Tensor:
        if mode == "soft":
            return logits.softmax(dim=-1)
        values, indices = logits.topk(min(k, logits.shape[-1]), dim=-1)
        weights = torch.zeros_like(logits).scatter(-1, indices, values.softmax(dim=-1))
        return weights

    @staticmethod
    def _apply_disabled(probs: Tensor, disabled: Sequence[int], renormalize: bool) -> Tensor:
        if not disabled:
            return probs
        result = probs.clone()
        result[:, list(disabled)] = 0
        if renormalize:
            denom = result.sum(dim=-1, keepdim=True)
            if torch.any(denom <= 0):
                raise ValueError("intervention disables all routing choices")
            result = result / denom
        return result

    def forward(
        self,
        x: Tensor,
        intervention: Optional[Intervention] = None,
        return_diagnostics: bool = False,
    ) -> Tensor | DLMoEOutput:
        if x.ndim != 2 or x.shape[-1] != self.config.input_dim:
            raise ValueError(f"expected [batch, {self.config.input_dim}] input, got {tuple(x.shape)}")
        intervention = intervention or Intervention()
        h = self.encoder(x)
        mode = intervention.routing_mode_override or self.config.routing_mode
        k = 1 if mode == "top1" else 2 if mode == "top2" else self.config.top_k
        pop_probs = torch.full_like(self.population_router(h), 1 / 2) if intervention.uniform_population_routing else self._route(self.population_router(h), "soft", 1)
        pop_probs = self._apply_disabled(pop_probs, intervention.disabled_populations, intervention.renormalize)
        expert_probs = []
        expert_outputs = []
        population_outputs = []
        for p in range(2):
            probs = torch.full_like(self.expert_routers[p](h), 1 / self.config.experts_per_population) if intervention.uniform_expert_routing else self._route(self.expert_routers[p](h), mode, k)
            disabled = [e for pp, e in intervention.disabled_experts if pp == p]
            probs = self._apply_disabled(probs, disabled, intervention.renormalize)
            if p in intervention.disabled_populations:
                probs = torch.zeros_like(probs)
            outputs = torch.stack([expert(h) for expert in self.experts[p]], dim=1)
            aggregate = (probs.unsqueeze(-1) * outputs).sum(dim=1)
            expert_probs.append(probs)
            expert_outputs.append(outputs)
            population_outputs.append(aggregate)
        populations = torch.stack(population_outputs, dim=1)
        if self.config.communication_enabled and self.config.comm_dim > 0 and self.config.communication_scale > 0:
            messages = torch.stack([self.comm_down[p](populations[:, p]) for p in range(2)], dim=1)
            incoming = torch.stack([self.comm_up[1](messages[:, 1]), self.comm_up[0](messages[:, 0])], dim=1)
            if intervention.disable_comm_1_to_0:
                incoming[:, 0] = 0
            if intervention.disable_comm_0_to_1:
                incoming[:, 1] = 0
            scale = self.config.communication_scale
            gates = torch.sigmoid(self.communication_gate) if self.communication_gate is not None else torch.ones(2, device=x.device)
            gates = gates * scale
            populations = torch.stack([self.population_norms[p](populations[:, p] + gates[p] * incoming[:, p]) for p in range(2)], dim=1)
        else:
            messages = None
            gates = None
            populations = torch.stack([self.population_norms[p](populations[:, p]) for p in range(2)], dim=1)
        mixed = (pop_probs.unsqueeze(-1) * populations).sum(dim=1)
        logits = self.head(self.output_norm(h + mixed))
        if self.config.output_dim == 1:
            logits = logits.squeeze(-1)
        if not return_diagnostics:
            return logits
        return DLMoEOutput(
            logits=logits,
            population_probs=pop_probs,
            expert_probs=torch.stack(expert_probs, dim=1),
            expert_outputs=torch.stack(expert_outputs, dim=1),
            population_outputs=populations,
            communication_messages=messages,
            communication_gates=gates,
        )


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class FlatMoE(nn.Module):
    """Contemporary parameter-comparable flat control with the same expert count."""

    def __init__(self, config: DLMoEConfig = DLMoEConfig()) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.model_dim), nn.GELU(),
            nn.Linear(config.model_dim, config.model_dim), nn.LayerNorm(config.model_dim),
        )
        self.router = nn.Linear(config.model_dim, config.num_populations * config.experts_per_population)
        self.experts = nn.ModuleList([_Expert(config) for _ in range(config.num_populations * config.experts_per_population)])
        self.norm = nn.LayerNorm(config.model_dim)
        self.head = nn.Linear(config.model_dim, config.output_dim)

    @property
    def total_experts(self) -> int:
        return len(self.experts)

    def forward(self, x: Tensor, intervention: Optional[Intervention] = None, return_diagnostics: bool = False) -> Tensor | DLMoEOutput:
        if x.ndim != 2 or x.shape[-1] != self.config.input_dim:
            raise ValueError(f"expected [batch, {self.config.input_dim}] input")
        intervention = intervention or Intervention()
        h = self.encoder(x)
        probs = self.router(h).softmax(-1)
        if intervention.uniform_expert_routing:
            probs = torch.full_like(probs, 1 / self.total_experts)
        disabled = [p * self.config.experts_per_population + e for p, e in intervention.disabled_experts]
        disabled += [p * self.config.experts_per_population + e for p in intervention.disabled_populations for e in range(self.config.experts_per_population)]
        probs = DLMoE._apply_disabled(probs, disabled, intervention.renormalize)
        outputs = torch.stack([expert(h) for expert in self.experts], dim=1)
        mixed = (probs.unsqueeze(-1) * outputs).sum(dim=1)
        logits = self.head(self.norm(h + mixed))
        if self.config.output_dim == 1:
            logits = logits.squeeze(-1)
        if not return_diagnostics:
            return logits
        return DLMoEOutput(logits=logits, population_probs=probs[:, :2], expert_probs=probs.unsqueeze(1), expert_outputs=outputs.unsqueeze(1), population_outputs=mixed.unsqueeze(1))
