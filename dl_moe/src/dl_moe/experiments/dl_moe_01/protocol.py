from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from ...model import DLMoE, DLMoEConfig, FlatMoE, parameter_count

ROOT_STRING = "DL-MOE-01-CONFIRMATORY-v1"
PRIMARY_BLOCKS = range(1, 31)
RESERVE_BLOCKS = range(31, 41)
SEED_FIELDS = ("model_init_seed", "development_data_seed", "common_data_seed", "common_order_seed", "eval_seed", "bootstrap_seed")
MANIFEST_COLUMNS = ("seed_block_id", "seed_role", *SEED_FIELDS)


def deterministic_seed(block: int, field: str) -> int:
    payload = f"{ROOT_STRING}|{block:03d}|{field}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return 1 + (int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**31 - 2))


def seed_rows() -> list[dict[str, int | str]]:
    rows = []
    for block in (*PRIMARY_BLOCKS, *RESERVE_BLOCKS):
        row: dict[str, int | str] = {"seed_block_id": f"{block:03d}", "seed_role": "primary" if block <= 30 else "reserve"}
        row.update({field: deterministic_seed(block, field) for field in SEED_FIELDS})
        rows.append(row)
    return rows


def write_seed_manifest(path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(seed_rows())
    return sha256_file(path)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def parameter_audit(config: dict) -> dict:
    model_config = DLMoEConfig(**config["model"])
    dl = DLMoE(model_config)
    flat = FlatMoE(model_config)
    dl_count, flat_count = parameter_count(dl), parameter_count(flat)
    ratio = abs(dl_count - flat_count) / max(dl_count, flat_count)
    return {"dl_params": dl_count, "flat_params": flat_count, "ratio": ratio, "threshold": 0.05, "passes": ratio <= 0.05}


def experience_rows(block: int, seeds: dict, early_steps: int = 4000, common_steps: int = 16000) -> list[dict]:
    """Create compact deterministic experience identities, not tensor payloads."""
    if early_steps != 4000 or common_steps != 16000:
        raise ValueError("DL-MoE-01 preregistration fixes 4000/16000 steps")
    rows = []
    schedules = {
        "INT": [(task, i) for i in range(2000) for task in ("A", "B")],
        "AB": [("A", i) for i in range(2000)] + [("B", i) for i in range(2000)],
        "BA": [("B", i) for i in range(2000)] + [("A", i) for i in range(2000)],
    }
    for condition, tasks in schedules.items():
        step = 0
        for task, ordinal in tasks:
            step += 1
            batch_seed = seeds["development_data_seed"] + (0 if task == "A" else 1_000_003) + ordinal
            rows.append({"seed_block_id": f"{block:03d}", "phase": "developmental", "condition": condition, "step": step, "task": task, "batch_id": f"{task}{ordinal + 1}", "batch_seed": batch_seed})
    for step in range(common_steps):
        task = "A" if step % 2 == 0 else "B"
        batch_seed = seeds["common_data_seed"] + (0 if task == "A" else 1_000_003) + step // 2
        rows.append({"seed_block_id": f"{block:03d}", "phase": "common", "condition": "ALL", "step": step + 1, "task": task, "batch_id": f"C{step + 1:05d}", "batch_seed": batch_seed})
    return rows


def validate_experience_rows(rows: list[dict]) -> None:
    development = [row for row in rows if row["phase"] == "developmental"]
    assert len(development) == 12000
    by_condition = {condition: [row for row in development if row["condition"] == condition] for condition in ("INT", "AB", "BA")}
    assert all(len(rows_) == 4000 for rows_ in by_condition.values())
    assert all(sorted(row["batch_id"] for row in rows_ if row["task"] == "A") == sorted(by_condition["INT"][i]["batch_id"] for i in range(0, 4000, 2)) for rows_ in by_condition.values())
    common = [row for row in rows if row["phase"] == "common"]
    assert len(common) == 16000 and all(row["condition"] == "ALL" for row in common)
