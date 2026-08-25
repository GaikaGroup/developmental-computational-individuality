from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_results(directory: str | Path) -> list[dict]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(Path(directory).glob("*.json"))]


def developmental_effects(results: list[dict], final_only: bool = True) -> dict:
    grouped: dict[tuple[str, str], list[float]] = {}
    for result in results:
        record = result.get("final", {}) if final_only else result
        value = record.get("causal_specialization_general")
        if value is not None:
            grouped.setdefault((result["architecture"], result["condition"]), []).append(float(value))
    out = {}
    for architecture in sorted({key[0] for key in grouped}):
        interleaved = np.asarray(grouped.get((architecture, "interleaved"), []), dtype=float)
        blocked = np.concatenate([grouped.get((architecture, "blocked_ab"), []), grouped.get((architecture, "blocked_ba"), [])]).astype(float)
        if len(interleaved) and len(blocked):
            out[architecture] = {"interleaved_mean": float(interleaved.mean()), "blocked_mean": float(blocked.mean()), "blocked_minus_interleaved": float(blocked.mean() - interleaved.mean()), "n_interleaved": int(len(interleaved)), "n_blocked": int(len(blocked))}
    return out


def main(argv=None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Analyze DL-MoE smoke or experimental artifacts")
    parser.add_argument("directory")
    args = parser.parse_args(argv)
    print(json.dumps(developmental_effects(load_results(args.directory)), indent=2))


if __name__ == "__main__":
    main()
