from __future__ import annotations
import torch
from .data import make_task
from .metrics import causal_specialization, router_entropy, routing_specialization

@torch.no_grad()
def evaluate(model, device, n, seed, knockout=None, knockout_mode="renormalized"):
    model.eval(); result = {}; routes = []; entropies = []
    for task, offset in (("A", 0), ("B", 1)):
        x, y = make_task(task, n, seed + offset, device)
        logits, p = model(x, knockout=knockout, return_router=True, knockout_mode=knockout_mode)
        result[f"accuracy_{task}"] = float(((logits > 0).float() == y).float().mean())
        result[f"bce_{task}"] = float(torch.nn.functional.binary_cross_entropy_with_logits(logits, y))
        routes.append(p.mean(0).cpu()); entropies.append(router_entropy(p))
    result["routing_A"], result["routing_B"] = routes
    result["routing_specialization"] = None if knockout is not None else routing_specialization(*routes)
    result["router_entropy"] = sum(entropies) / len(entropies)
    result["router_entropy_of_task_means"] = router_entropy(torch.stack(routes))
    return result

@torch.no_grad()
def causal_evaluation(model, device, n, seed, knockout_modes=("renormalized", "non_renormalized")):
    normal = evaluate(model, device, n, seed); knockouts = {}
    for mode in knockout_modes:
        ablation = {}
        for expert in (0, 1):
            ablated = evaluate(model, device, n, seed, knockout=expert, knockout_mode=mode)
            ablation[f"expert_{expert + 1}"] = {f"delta_{t}": normal[f"accuracy_{t}"] - ablated[f"accuracy_{t}"] for t in ("A", "B")}
        causal = torch.tensor([[ablation["expert_1"]["delta_A"], ablation["expert_1"]["delta_B"]], [ablation["expert_2"]["delta_A"], ablation["expert_2"]["delta_B"]]])
        knockouts[mode] = {"ablation": ablation, "causal_specialization": causal_specialization(causal)}
    primary = knockouts["renormalized"]
    normal["ablation"], normal["causal_specialization"] = primary["ablation"], primary["causal_specialization"]
    normal["knockout_renormalized"] = primary
    if "non_renormalized" in knockouts: normal["knockout_non_renormalized"] = knockouts["non_renormalized"]
    return normal
