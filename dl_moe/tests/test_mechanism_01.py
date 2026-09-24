from pathlib import Path

import torch
import yaml

from dl_moe.experiments.dl_moe_mechanism_01.run import INTERVENTIONS, run_checkpoint
from dl_moe.model import DLMoE, DLMoEConfig


def test_intervention_matrix_is_paired_and_complete():
    assert set(INTERVENTIONS) == {
        "intact", "disable_0_to_1", "disable_1_to_0", "disable_both"
    }
    assert INTERVENTIONS["disable_0_to_1"].disable_comm_0_to_1
    assert INTERVENTIONS["disable_1_to_0"].disable_comm_1_to_0
    assert INTERVENTIONS["disable_both"].disable_comm_0_to_1
    assert INTERVENTIONS["disable_both"].disable_comm_1_to_0


def test_runner_writes_provenance_complete_artifact(tmp_path: Path):
    cfg = {
        "model": {
            "input_dim": 18,
            "model_dim": 16,
            "num_populations": 2,
            "experts_per_population": 2,
            "expert_hidden_dim": 16,
            "comm_dim": 4,
            "routing_mode": "soft",
            "top_k": 1,
            "communication_enabled": True,
            "communication_scale": 1.0,
            "learned_communication_gate": True,
            "dropout": 0.0,
            "output_dim": 1,
        },
        "evaluation": {"examples": 32},
        "seeds": {"eval_seed": 9000},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    checkpoint = tmp_path / "model.pt"
    torch.save(DLMoE(DLMoEConfig(**cfg["model"])).state_dict(), checkpoint)
    output = tmp_path / "artifact.json"

    result = run_checkpoint(
        checkpoint, config_path, output, schedule="interleaved", seed=7, examples=16
    )

    assert output.exists()
    assert len(result["checkpoint_sha256"]) == 64
    assert len(result["config_hash"]) == 64
    assert result["status"] == "exploratory_diagnostic"
    assert set(result["conditions"]) == set(INTERVENTIONS)
    assert set(result["paired_deltas_vs_intact"]) == {
        "disable_0_to_1", "disable_1_to_0", "disable_both"
    }
