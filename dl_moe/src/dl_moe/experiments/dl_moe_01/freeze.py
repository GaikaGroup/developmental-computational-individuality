from __future__ import annotations

import argparse
import csv
import datetime as dt
import subprocess
from pathlib import Path

from ...training import load_config
from .protocol import sha256_file, parameter_audit, write_seed_manifest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Create DL-MoE-01 protocol hash record")
    parser.add_argument("--output", default="dl_moe/experiments/dl_moe_01/protocol_hashes.csv")
    parser.add_argument("--manifest", default="dl_moe/experiments/dl_moe_01/seed_manifest.csv")
    parser.add_argument("--config", default="dl_moe/configs/confirmatory/dl_moe_v1.yaml")
    parser.add_argument("--prereg", default="dl_moe/experiments/dl_moe_01/preregistration.yaml")
    parser.add_argument("--prereg-md", default="docs/preregistration/DL_MOE_01_PREREGISTRATION.md")
    args = parser.parse_args(argv)
    write_seed_manifest(args.manifest)
    audit = parameter_audit(load_config(args.config))
    if not audit["passes"]:
        raise SystemExit("PROTOCOL BLOCKER: parameter comparability failed")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    rows = []
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    for label, path in (("preregistration.yaml", args.prereg), ("preregistration.md", args.prereg_md), ("seed_manifest.csv", args.manifest), ("confirmatory_config", args.config)):
        rows.append({"artifact": label, "path": path, "sha256": sha256_file(path), "created_at": timestamp})
    rows.append({"artifact": "code_commit", "path": "git:HEAD", "sha256": commit, "created_at": timestamp})
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("artifact", "path", "sha256", "created_at")); writer.writeheader(); writer.writerows(rows)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
