"""Protocol-specific balanced-accuracy interventions; historical evaluators stay unchanged."""
from __future__ import annotations

import torch

from ...compatibility import make_dci_task
from ...model import Intervention


def classification_metrics(logits, labels):
    if not torch.isfinite(logits).all():
        raise FloatingPointError('nonfinite evaluation logits')
    predicted = logits > 0
    recalls = []
    for label in (0, 1):
        mask = labels == label
        if not mask.any():
            raise ValueError('evaluation requires both classes')
        recalls.append(float((predicted[mask] == labels[mask]).float().mean()))
    return dict(accuracy=float((predicted == labels).float().mean()),
                balanced_accuracy=sum(recalls)/2,
                loss=float(torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)),
                n_eval_examples=len(labels))


@torch.no_grad()
def evaluate_checkpoint(model, n, seed, batch_size=512):
    model.eval()
    device = next(model.parameters()).device
    tasks = {t:make_dci_task(t,n,seed+i,'cpu') for i,t in enumerate(('A','B'))}

    def measure(intervention=None, diagnostics=False):
        behavior, routes, communication = {}, [], []
        for task,(x,y) in tasks.items():
            logits, probabilities, messages, population_routes = [], [], [], []
            gates = None
            for batch in x.split(batch_size):
                result = model(batch.to(device),intervention=intervention,return_diagnostics=True)
                logits.append(result.logits.cpu())
                if diagnostics:
                    if result.expert_probs.shape[1] == 2:
                        population_routes.append(result.population_probs.cpu())
                        probs = result.population_probs.unsqueeze(-1)*result.expert_probs
                        probs = probs.flatten(1)
                    else:
                        probs = result.expert_probs[:,0]
                    probabilities.append(probs.cpu())
                    if result.communication_messages is not None:
                        messages.append(result.communication_messages.norm(dim=-1).cpu())
                        gates = result.communication_gates.cpu()
            behavior[task] = classification_metrics(torch.cat(logits),y)
            if diagnostics:
                probs = torch.cat(probabilities)
                entropy = float(-(probs*probs.clamp_min(1e-12).log()).sum(-1).mean())
                winners = probs.argmax(-1)
                for e in range(model.total_experts):
                    routes.append(dict(task=task,expert_id=e,population_id=e//model.config.experts_per_population
                                       if model.__class__.__name__=='DLMoE' else '',
                        mean_routing_probability=float(probs[:,e].mean()),
                        utilization=float((winners==e).float().mean()),routing_entropy=entropy,
                        router_collapse_flag=bool(probs[:,e].mean()>.95)))
                if population_routes:
                    population_probs = torch.cat(population_routes)
                    population_entropy = float(-(population_probs*population_probs.clamp_min(1e-12).log()).sum(-1).mean())
                    for p in (0,1):
                        routes.append(dict(task=task,expert_id='',population_id=p,
                            mean_routing_probability=float(population_probs[:,p].mean()),
                            utilization=float((population_probs.argmax(-1)==p).float().mean()),
                            routing_entropy=population_entropy,router_collapse_flag=bool(population_probs[:,p].mean()>.95)))
                if messages:
                    norms = torch.cat(messages)
                    for source in (0,1):
                        communication.append(dict(task=task,direction=f'{source}_to_{1-source}',
                            mean_message_norm=float(norms[:,source].mean()),
                            median_message_norm=float(norms[:,source].quantile(.5)),
                            gate_value=float(gates[1-source]/model.config.communication_scale),
                            communication_scale=model.config.communication_scale))
        return behavior,routes,communication

    normal,routes,communication = measure(diagnostics=True)
    result = dict(behavior=normal,routing=routes,communication=communication,expert={},population={},communication_knockout={})
    for mode,renormalize in [('renormalized',True),('unrenormalized',False)]:
        effects=[]
        for e in range(model.total_experts):
            p,local = divmod(e,model.config.experts_per_population)
            ablated,_,_ = measure(Intervention(disabled_experts=((p,local),),renormalize=renormalize))
            effects.append({t:normal[t]['balanced_accuracy']-ablated[t]['balanced_accuracy'] for t in tasks})
        result['expert'][mode]=effects
        if model.__class__.__name__=='DLMoE':
            effects=[]
            for p in (0,1):
                ablated,_,_ = measure(Intervention(disabled_populations=(p,),renormalize=renormalize))
                effects.append({t:normal[t]['balanced_accuracy']-ablated[t]['balanced_accuracy'] for t in tasks})
            result['population'][mode]=effects
    if model.__class__.__name__=='DLMoE':
        for name,intervention in [('0_to_1',Intervention(disable_comm_0_to_1=True)),
                                 ('1_to_0',Intervention(disable_comm_1_to_0=True)),
                                 ('both',Intervention(disable_comm_0_to_1=True,disable_comm_1_to_0=True))]:
            ablated,_,_ = measure(intervention)
            result['communication_knockout'][name]={t:normal[t]['balanced_accuracy']-ablated[t]['balanced_accuracy'] for t in tasks}
    return result
