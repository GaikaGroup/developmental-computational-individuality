from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ...training import load_config
from .protocol import MANIFEST_COLUMNS, PRIMARY_BLOCKS, RESERVE_BLOCKS, parameter_audit, seed_rows, sha256_file


def validate(config_path: str, manifest_path: str, prereg_path: str) -> list[str]:
    errors = []
    config = load_config(config_path)
    audit = parameter_audit(config)
    if not audit["passes"]:
        errors.append(f"parameter ratio {audit['ratio']:.6f} > 0.05")
    manifest = Path(manifest_path)
    if not manifest.exists():
        errors.append("seed_manifest.csv missing")
    else:
        with manifest.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if tuple(rows[0]) != MANIFEST_COLUMNS:
            errors.append("seed manifest columns mismatch")
        if len(rows) != 40 or [row["seed_block_id"] for row in rows[:30]] != [f"{i:03d}" for i in PRIMARY_BLOCKS]:
            errors.append("seed manifest must contain blocks 001-040")
        expected = [{k: str(v) for k, v in row.items()} for row in seed_rows()]
        if rows != expected:
            errors.append("seed manifest values do not match deterministic generator")
    if not Path(prereg_path).exists():
        errors.append("preregistration YAML missing")
    return errors


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="dl_moe/configs/confirmatory/dl_moe_v1.yaml")
    parser.add_argument("--manifest", default="dl_moe/experiments/dl_moe_01/seed_manifest.csv")
    parser.add_argument("--preregistration", default="dl_moe/experiments/dl_moe_01/preregistration.yaml")
    args = parser.parse_args(argv)
    errors = validate(args.config, args.manifest, args.preregistration)
    if errors:
        print("DL-MoE-01 NOT READY")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("DL-MoE-01 READY FOR PREREGISTRATION FREEZE")


if __name__ == "__main__":
    main()
