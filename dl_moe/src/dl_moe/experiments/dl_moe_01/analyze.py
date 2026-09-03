from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .schemas import write_empty_schema_files
from .statistics import one_sided_t, paired_tost, sign_flip


def _stats(values: np.ndarray, bootstrap_seed: int = 0, samples: int = 10000) -> dict:
    if len(values) == 0:
        raise ValueError("no seed-level values")
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return {"n": int(len(values)), "mean": float(values.mean()), "median": float(np.median(values)), "sd": sd, "se": float(sd / np.sqrt(len(values))), "ci95_lower": float(np.quantile(bootstrap, .025)), "ci95_upper": float(np.quantile(bootstrap, .975)), "cohens_dz": float(values.mean() / sd) if sd else 0.0, "positive": int((values > 0).sum()), "zero": int((values == 0).sum()), "negative": int((values < 0).sum())}


def analyze(results: str | Path, output: str | Path, preregistration: str | Path) -> dict:
    results, output = Path(results), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_empty_schema_files(output)
    source = results / "seed_effects.csv"
    if not source.exists():
        raise FileNotFoundError("seed_effects.csv is required; analysis never trains models")
    with source.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    primary = [row for row in rows if row.get("primary_endpoint_available", "").lower() == "true"]
    if len(primary) < 28:
        raise ValueError(f"confirmatory analysis incomplete: {len(primary)} complete blocks, minimum is 28")
    values = np.asarray([float(row["architecture_amplification_a"]) for row in primary], dtype=float)
    primary_stats = _stats(values)
    primary_stats.update({"t_test": one_sided_t(values), "sign_flip_p": sign_flip(values, 0)})
    summary = {"experiment_id": "DL-MoE-01", "preregistration": str(preregistration), "primary": primary_stats, "analysis_set_n": len(values), "trained": False}
    statistical_rows = []
    for hypothesis, column in (("H2", "architecture_amplification_a"), ("H1", "d_dl"), ("H4", "d_pop_dl"), ("H5", "architecture_amplification_a_unrenormalized")):
        available = [row for row in primary if row.get(column, "") not in ("", "NA", "None")]
        if not available:
            continue
        vector = np.asarray([float(row[column]) for row in available], dtype=float)
        stats = _stats(vector)
        test = one_sided_t(vector)
        statistical_rows.append({"hypothesis": hypothesis, "endpoint": column, "analysis_set_n": len(vector), "mean": stats["mean"], "median": stats["median"], "sd": stats["sd"], "standard_error": stats["se"], "ci95_lower": stats["ci95_lower"], "ci95_upper": stats["ci95_upper"], "effect_size_name": "cohens_dz", "effect_size": stats["cohens_dz"], "test_name": "one_sample_t", "alternative": "greater", "test_statistic": test["statistic"], "p_value_raw": test["p_value"], "p_value_adjusted": test["p_value"], "positive_count": stats["positive"], "zero_count": stats["zero"], "negative_count": stats["negative"], "supported": stats["mean"] > 0 and test["p_value"] < .05, "notes": "Seed-block analysis; no outcome exclusions."})
    with (output / "statistical_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        import csv
        writer = csv.DictWriter(handle, fieldnames=statistical_rows[0]); writer.writeheader(); writer.writerows(statistical_rows)
    behavior = results / "behavior_metrics.csv"
    if behavior.exists():
        with behavior.open(encoding="utf-8") as handle:
            behavior_rows = list(csv.DictReader(handle))
        summary["behavior_rows_available"] = len(behavior_rows)
        behavior_rows = [row for row in behavior_rows if row.get("checkpoint") == "20000"]
        by_key = {(row["architecture"], row["seed_block_id"], row["developmental_condition"], row["task"]): float(row["balanced_accuracy"]) for row in behavior_rows}
        equivalence_rows = []
        for architecture in ("dl_moe", "flat_moe"):
            for task in ("A", "B"):
                for blocked in ("AB", "BA"):
                    differences = [by_key[(architecture, block, blocked, task)] - by_key[(architecture, block, "INT", task)] for block in {key[1] for key in by_key if key[0] == architecture} if (architecture, block, blocked, task) in by_key and (architecture, block, "INT", task) in by_key]
                    if differences:
                        tost = paired_tost(np.asarray(differences))
                        equivalence_rows.append({"architecture": architecture, "task": task, "comparison": f"{blocked}_minus_INT", **tost, "sd_difference": float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0, "equivalence_margin_lower": -.02, "equivalence_margin_upper": .02})
        if equivalence_rows:
            with (output / "behavioral_equivalence.csv").open("w", newline="", encoding="utf-8") as handle:
                import csv
                writer = csv.DictWriter(handle, fieldnames=equivalence_rows[0]); writer.writeheader(); writer.writerows(equivalence_rows)
    if statistical_rows:
        with (output / "statistical_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            import csv
            writer = csv.DictWriter(handle, fieldnames=statistical_rows[0]); writer.writeheader(); writer.writerows(statistical_rows)
    (output / "DL_MOE_01_INTERPRETATION.md").write_text("# DL-MoE-01 interpretation\n\nThis report is generated from all complete seed blocks. Primary and behavioral gates must be read together.\n", encoding="utf-8")
    (output / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--output", default="results/dl_moe_01/analysis")
    args = parser.parse_args(argv)
    print(json.dumps(analyze(args.results, args.output, args.preregistration), indent=2))


if __name__ == "__main__":
    main()
