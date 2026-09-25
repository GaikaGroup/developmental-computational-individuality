from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import torch

from ...compatibility import s_causal_original
from ...evaluation import evaluate_behavior, serialize_tensors
from ...metrics import causal_specialization_general, population_specialization, s_expert_causal
from ...model import DLMoE, DLMoEConfig, Intervention
from ...training import config_hash, load_config


INTERVENTIONS = {
    "intact": Intervention(),
    "disable_0_to_1": Intervention(disable_comm_0_to_1=True),
    "disable_1_to_0": Intervention(disable_comm_1_to_0=True),
    "disable_both": Intervention(disable_comm_0_to_1=True, disable_comm_1_to_0=True),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_dict(payload):
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
    return payload


def compose_intervention(base: Intervention, **changes) -> Intervention:
    """Compose a mechanism intervention with an inner causal ablation."""
    return replace(base, **changes)


@torch.no_grad()
def evaluate_causal_conditioned(
    model: DLMoE,
    device: str,
    n: int,
    seed: int,
    outer_intervention: Intervention,
    renormalize: bool = True,
) -> dict:
    """Evaluate causal specialization while holding an outer intervention fixed.

    Every expert/population ablation inherits the same communication condition,
    so differences across conditions measure changes in the causal endpoint
    rather than accidentally returning to intact communication.
    """
    normal = evaluate_behavior(model, device, n, seed, intervention=outer_intervention)
    expert_effects = torch.zeros(model.total_experts, 2)
    for p in range(2):
        for e in range(model.config.experts_per_population):
            intervention = compose_intervention(
                outer_intervention,
                disabled_experts=((p, e),),
                renormalize=renormalize,
            )
            ablated = evaluate_behavior(model, device, n, seed, intervention=intervention)
            index = p * model.config.experts_per_population + e
            expert_effects[index] = torch.tensor([
                normal["accuracy_A"] - ablated["accuracy_A"],
                normal["accuracy_B"] - ablated["accuracy_B"],
            ])

    population_effects = torch.zeros(2, 2)
    for p in range(2):
        intervention = compose_intervention(
            outer_intervention,
            disabled_populations=(p,),
            renormalize=renormalize,
        )
        ablated = evaluate_behavior(model, device, n, seed, intervention=intervention)
        population_effects[p] = torch.tensor([
            normal["accuracy_A"] - ablated["accuracy_A"],
            normal["accuracy_B"] - ablated["accuracy_B"],
        ])

    return {
        **normal,
        "expert_causal_effects": expert_effects,
        "population_causal_effects": population_effects,
        "s_causal_original": s_causal_original(expert_effects) if expert_effects.shape == (2, 2) else None,
        "s_expert_causal": s_expert_causal(expert_effects),
        "causal_specialization_general": causal_specialization_general(expert_effects),
        "population_specialization": population_specialization(population_effects),
        "renormalized": renormalize,
    }


def run_checkpoint(checkpoint: str | Path, config_path: str | Path, output: str | Path,
                   schedule: str, seed: int, examples: int | None = None,
                   device: str = "cpu") -> dict:
    checkpoint = Path(checkpoint)
    config = load_config(config_path)
    model = DLMoE(DLMoEConfig(**config["model"])).to(device)
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(_state_dict(payload))
    n = examples or int(config["evaluation"]["examples"])
    eval_seed = int(config["seeds"]["eval_seed"]) + seed

    conditions = {}
    for name, intervention in INTERVENTIONS.items():
        conditions[name] = serialize_tensors(
            evaluate_causal_conditioned(model, device, n, eval_seed, intervention)
        )

    intact = conditions["intact"]
    paired_deltas = {}
    endpoint_keys = (
        "accuracy_A",
        "accuracy_B",
        "bce_A",
        "bce_B",
        "s_expert_causal",
        "causal_specialization_general",
        "population_specialization",
    )
    for name in ("disable_0_to_1", "disable_1_to_0", "disable_both"):
        paired_deltas[name] = {
            metric: intact[metric] - conditions[name][metric]
            for metric in endpoint_keys
        }

    result = {
        "experiment": "dl_moe_mechanism_01",
        "status": "exploratory_diagnostic",
        "schedule": schedule,
        "seed": seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config_path": str(config_path),
        "config_hash": config_hash(config),
        "examples_per_task": n,
        "eval_seed": eval_seed,
        "conditions": conditions,
        "paired_deltas_vs_intact": paired_deltas,
        "interpretation_guardrail": (
            "Communication-conditioned causal effects indicate pathway participation in the "
            "measured organization; they do not by themselves establish developmental causation "
            "or computational individuality."
        ),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Exploratory DL-MoE communication diagnostic")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="dl_moe/configs/confirmatory/dl_moe_v1.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--schedule", required=True, choices=("blocked_ab", "blocked_ba", "interleaved"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--examples", type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    result = run_checkpoint(**vars(args))
    print(json.dumps({
        "output": args.output,
        "checkpoint_sha256": result["checkpoint_sha256"],
        "schedule": result["schedule"],
        "seed": result["seed"],
    }, indent=2))


if __name__ == "__main__":
    main()
