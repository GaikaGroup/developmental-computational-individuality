from __future__ import annotations

import torch
from torch import Tensor

from .compatibility import make_dci_task, s_causal_original
from .metrics import causal_specialization_general, population_specialization, routing_entropy
from .model import DLMoE, FlatMoE, Intervention


@torch.no_grad()
def evaluate_behavior(model: DLMoE | FlatMoE, device: str, n: int, seed: int, intervention: Intervention | None = None) -> dict:
    model.eval()
    result: dict = {}
    routes = []
    for task, offset in (("A", 0), ("B", 1)):
        x, y = make_dci_task(task, n, seed + offset, device)
        output = model(x, intervention=intervention, return_diagnostics=True)
        logits = output.logits
        result[f"accuracy_{task}"] = float(((logits > 0).float() == y).float().mean().item())
        result[f"bce_{task}"] = float(torch.nn.functional.binary_cross_entropy_with_logits(logits, y).item())
        routes.append(output.population_probs.mean(0).cpu())
    result["population_routing_A"] = routes[0]
    result["population_routing_B"] = routes[1]
    result["population_routing_entropy"] = routing_entropy(torch.stack(routes))
    return result


@torch.no_grad()
def evaluate_causal(model: DLMoE | FlatMoE, device: str, n: int, seed: int, renormalize: bool = True) -> dict:
    normal = evaluate_behavior(model, device, n, seed)
    expert_effects = torch.zeros(model.total_experts, 2)
    for p in range(2):
        for e in range(model.config.experts_per_population):
            ablated = evaluate_behavior(model, device, n, seed, Intervention(disabled_experts=((p, e),), renormalize=renormalize))
            index = p * model.config.experts_per_population + e
            expert_effects[index] = torch.tensor([normal["accuracy_A"] - ablated["accuracy_A"], normal["accuracy_B"] - ablated["accuracy_B"]])
    population_effects = torch.zeros(2, 2)
    for p in range(2):
        ablated = evaluate_behavior(model, device, n, seed, Intervention(disabled_populations=(p,), renormalize=renormalize))
        population_effects[p] = torch.tensor([normal["accuracy_A"] - ablated["accuracy_A"], normal["accuracy_B"] - ablated["accuracy_B"]])
    comm_effects = {}
    for name, intervention in {
        "0_to_1": Intervention(disable_comm_0_to_1=True),
        "1_to_0": Intervention(disable_comm_1_to_0=True),
        "both": Intervention(disable_comm_0_to_1=True, disable_comm_1_to_0=True),
    }.items():
        ablated = evaluate_behavior(model, device, n, seed, intervention)
        comm_effects[name] = {task: normal[f"accuracy_{task}"] - ablated[f"accuracy_{task}"] for task in ("A", "B")}
    return {
        **normal,
        "expert_causal_effects": expert_effects,
        "population_causal_effects": population_effects,
        "communication_effects": comm_effects,
        "s_causal_original": s_causal_original(expert_effects) if expert_effects.shape == (2, 2) else None,
        "causal_specialization_general": causal_specialization_general(expert_effects),
        "population_specialization": population_specialization(population_effects),
        "renormalized": renormalize,
    }


def serialize_tensors(value):
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: serialize_tensors(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_tensors(v) for v in value]
    return value
