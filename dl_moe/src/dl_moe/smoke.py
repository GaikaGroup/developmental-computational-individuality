from __future__ import annotations

import argparse
import json

from .training import load_config, run_smoke


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="DL-MoE v1.0 software-validation smoke run")
    parser.add_argument("--config", default="dl_moe/configs/smoke.yaml")
    parser.add_argument("--output", default="dl_moe/results/smoke")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    for architecture in ("dl_moe", "flat_moe_control"):
        for condition in ("interleaved", "blocked_ab"):
            result = run_smoke(config, args.output, architecture, condition, args.seed)
            print(json.dumps({k: result[k] for k in ("experiment_id", "architecture", "condition", "parameter_count", "smoke_only")}, indent=2))


if __name__ == "__main__":
    main()
