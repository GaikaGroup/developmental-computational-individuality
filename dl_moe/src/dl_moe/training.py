from __future__ import annotations

import copy
import hashlib
import json
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from .compatibility import make_dci_task
from .evaluation import evaluate_causal, serialize_tensors
from .model import DLMoE, DLMoEConfig, FlatMoE, parameter_count
from .schedules import make_schedule


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def run_smoke(config: dict, output_dir: str | Path, architecture: str = "dl_moe", condition: str = "interleaved", seed: int = 0) -> dict:
    """Run a short software-validation experiment, never a confirmatory run."""
    seed_all(config["seeds"]["model_init_seed"] + seed)
    device = config.get("device", "cpu")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model_class = FlatMoE if architecture == "flat_moe_control" else DLMoE
    model = model_class(DLMoEConfig(**config["model"])).to(device)
    schedule = make_schedule(condition, config["training"]["early_steps"], config["training"]["total_steps"], config["seeds"]["order_seed"] + seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"].get("weight_decay", 0.0))
    data_seed = config["seeds"]["train_data_seed"] + seed
    history = []
    for step, task in enumerate(schedule.tasks, 1):
        x, y = make_dci_task(task, config["training"]["batch_size"], data_seed + (0 if task == "A" else 1_000_003) + step, device)
        model.train()
        logits = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in config["evaluation"]["checkpoints"]:
            causal = evaluate_causal(model, device, config["evaluation"]["examples"], config["seeds"]["eval_seed"] + seed)
            history.append({"step": step, **serialize_tensors(causal)})
    result = {
        "experiment_id": f"{architecture}_{condition}_seed_{seed:03d}_{config_hash(config)[:12]}",
        "architecture": architecture,
        "condition": condition,
        "seed_roles": {k: v + seed for k, v in config["seeds"].items() if isinstance(v, int)},
        "config_hash": config_hash(config),
        "git_commit": _git_commit(),
        "python": sys.version,
        "pytorch": torch.__version__,
        "platform": platform.platform(),
        "device": str(next(model.parameters()).device),
        "parameter_count": parameter_count(model),
        "history": history,
        "final": history[-1] if history else {},
        "task_counts": {task: schedule.tasks.count(task) for task in ("A", "B")},
        "smoke_only": True,
    }
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / f"{result['experiment_id']}.json").write_text(json.dumps(result, indent=2, default=serialize_tensors), encoding="utf-8")
    return result


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)
