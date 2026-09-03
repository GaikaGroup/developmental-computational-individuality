from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ...training import load_config
from .protocol import parameter_audit, seed_rows, write_seed_manifest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="DL-MoE-01 explicit runner")
    parser.add_argument("--preregistered", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config", default="dl_moe/configs/confirmatory/dl_moe_v1.yaml")
    parser.add_argument("--manifest", default="dl_moe/experiments/dl_moe_01/seed_manifest.csv")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    audit = parameter_audit(config)
    if not audit["passes"]:
        raise SystemExit(f"PROTOCOL BLOCKER: parameter ratio {audit['ratio']:.6f} exceeds 0.05")
    if args.dry_run:
        manifest_hash = write_seed_manifest(args.manifest)
        print("DL-MoE-01 DRY RUN")
        print("30 primary seed blocks")
        print("6 models per block")
        print("180 planned models")
        print(f"DL params: {audit['dl_params']}; Flat params: {audit['flat_params']}; ratio: {audit['ratio']:.6f}")
        print(f"primary endpoint: {config['evaluation']['primary_endpoint']}")
        print(f"seed manifest SHA256: {manifest_hash}")
        return
    if not args.preregistered:
        raise SystemExit("Refusing to train: use --preregistered explicitly after review; this task supports --dry-run only.")
    raise SystemExit("Full DL-MoE-01 training is intentionally not launched by the preregistration task.")


if __name__ == "__main__":
    main()
