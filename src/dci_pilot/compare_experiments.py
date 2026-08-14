from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from . import CONDITIONS
from .curriculum import build_curriculum
from .metrics import prolonged_router_collapse, perm_matrix_distance
from .statistics import paired_statistics, summary

EXPERIMENTS = ("Pilot 1", "Replication 1")
LABELS = {"INTERLEAVED": "INTERLEAVED", "BLOCKED_AB": "BLOCKED_AB", "BLOCKED_BA": "BLOCKED_BA"}
COLORS = {"INTERLEAVED": "#3274A1", "BLOCKED_AB": "#E1812C", "BLOCKED_BA": "#3A923A"}
MODES = ("renormalized", "non_renormalized")


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_raw(root):
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "seed_*", "*", "result.json"))):
        row = _read_json(path)
        row["_source_file"] = path
        rows.append(row)
    return rows


def _load_posthoc(path):
    return _read_json(path)["records"]


def _mode_key(mode):
    return "knockout_renormalized" if mode == "renormalized" else "knockout_non_renormalized"


def _causal(row, mode):
    value = row.get(_mode_key(mode), {})
    if value:
        return float(value["causal_specialization"])
    if mode == "renormalized":
        return float(row["causal_specialization"])
    return None


def _ablation(row, mode):
    value = row.get(_mode_key(mode), {})
    if value:
        return value["ablation"]
    return row["ablation"] if mode == "renormalized" else None


def _merge_final(raw_rows, posthoc_rows):
    final = {(r["seed"], r["condition"]): r for r in raw_rows}
    for posthoc in posthoc_rows:
        if posthoc["step"] != 20000:
            continue
        row = final[(posthoc["seed"], posthoc["condition"])]
        for key in ("accuracy_A", "accuracy_B", "bce_A", "bce_B", "routing_A", "routing_B", "routing_specialization", "router_entropy", "router_entropy_of_task_means", "knockout_renormalized", "knockout_non_renormalized", "ablation", "causal_specialization"):
            if key in posthoc:
                row[key] = posthoc[key]
        row["_posthoc_source_file"] = "results/pilot_v1_posthoc/analysis.json"
    for row in final.values():
        if "router_collapse" not in row:
            row["router_collapse"] = prolonged_router_collapse(row["history"])
    return list(final.values())


def _group(rows):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[row["seed"]][row["condition"]] = row
    return grouped


def _paired_rows(rows, mode="renormalized"):
    grouped = _group(rows)
    output = []
    for seed in sorted(grouped):
        values = grouped[seed]
        if set(values) != set(CONDITIONS):
            continue
        i = _causal(values["INTERLEAVED"], mode)
        ab = _causal(values["BLOCKED_AB"], mode)
        ba = _causal(values["BLOCKED_BA"], mode)
        output.append({"seed": seed, "interleaved": i, "blocked_ab": ab, "blocked_ba": ba, "blocked_mean": (ab + ba) / 2, "D_s": (ab + ba) / 2 - i})
    return output


def _stats(rows, mode):
    paired = _paired_rows(rows, mode)
    stats = paired_statistics([r["D_s"] for r in paired])
    values = np.asarray([r["D_s"] for r in paired], dtype=float)
    stats.update({"min_D_s": float(values.min()), "max_D_s": float(values.max()), "q1": float(np.quantile(values, .25)), "q3": float(np.quantile(values, .75))})
    stats["source_n_models"] = len(rows)
    stats["seed_values"] = [{"seed": r["seed"], "D_s": r["D_s"]} for r in paired]
    return stats


def _condition_metrics(rows, mode):
    result = []
    for condition in CONDITIONS:
        selected = [r for r in rows if r["condition"] == condition]
        def metric(name, values):
            s = summary(values)
            return {f"{name}_mean": s["mean"], f"{name}_sd": s["sd"]}
        row = {"condition": condition, "n": len(selected)}
        for name, values in (
            ("accuracy_A", [r["accuracy_A"] for r in selected]),
            ("accuracy_B", [r["accuracy_B"] for r in selected]),
            ("balanced_accuracy", [(r["accuracy_A"] + r["accuracy_B"]) / 2 for r in selected]),
            ("routing_specialization", [r["routing_specialization"] for r in selected]),
            ("router_entropy", [r["router_entropy"] for r in selected]),
            ("causal_specialization", [_causal(r, mode) for r in selected]),
        ):
            row.update(metric(name, values))
        row["router_collapse_count"] = sum(bool(r["router_collapse"]["collapsed"]) for r in selected)
        row["router_collapse_rate"] = row["router_collapse_count"] / len(selected) if selected else None
        max_effects = []
        for r in selected:
            ablation = _ablation(r, mode)
            if ablation:
                max_effects.append(max(abs(float(value)) for expert in ablation.values() for value in expert.values()))
        row.update(metric("max_absolute_knockout_effect", max_effects))
        result.append(row)
    return result


def _accuracy_comparison(rows):
    grouped = {name: _group(values) for name, values in rows.items()}
    output = []
    for experiment, values in grouped.items():
        for condition in CONDITIONS:
            selected = values
            row_values = [selected[seed][condition] for seed in selected if condition in selected[seed]]
            row = {"experiment": experiment, "condition": condition, "n": len(row_values)}
            for task in ("A", "B"):
                s = summary([r[f"accuracy_{task}"] for r in row_values])
                row.update({f"accuracy_{task}_mean": s["mean"], f"accuracy_{task}_sd": s["sd"], f"accuracy_{task}_min": min(r[f"accuracy_{task}"] for r in row_values)})
            balanced = [(r["accuracy_A"] + r["accuracy_B"]) / 2 for r in row_values]
            s = summary(balanced)
            row.update({"balanced_accuracy_mean": s["mean"], "balanced_accuracy_sd": s["sd"], "balanced_accuracy_min": min(balanced)})
            output.append(row)
    for experiment in rows:
        selected = [r for r in output if r["experiment"] == experiment]
        interleaved = next(r for r in selected if r["condition"] == "INTERLEAVED")
        blocked = [r for r in selected if r["condition"] in ("BLOCKED_AB", "BLOCKED_BA")]
        blocked_a = float(np.mean([r["accuracy_A_mean"] for r in blocked])); blocked_b = float(np.mean([r["accuracy_B_mean"] for r in blocked])); blocked_bal = float(np.mean([r["balanced_accuracy_mean"] for r in blocked]))
        for row in selected:
            if row["condition"] == "INTERLEAVED":
                row["blocked_minus_interleaved_accuracy_A"] = blocked_a - interleaved["accuracy_A_mean"]
                row["blocked_minus_interleaved_accuracy_B"] = blocked_b - interleaved["accuracy_B_mean"]
                row["blocked_minus_interleaved_balanced_accuracy"] = blocked_bal - interleaved["balanced_accuracy_mean"]
            else:
                row["blocked_minus_interleaved_accuracy_A"] = None; row["blocked_minus_interleaved_accuracy_B"] = None; row["blocked_minus_interleaved_balanced_accuracy"] = None
    return output


def _canonical_configuration(pilot_config, replication_config):
    pilot_test_seed = pilot_config["data_seed"] + 9_000_000
    common = {
        "architecture": "18→128→128 shared encoder; Linear(128,2) softmax router; two Linear(128,256)→128 experts; residual LayerNorm; Linear(128,1)",
        "parameter_count": 151427,
        "Task A": "1[x0*x1 + x2*x3 > 0]",
        "Task B": "1[x8*x9 - x10*x11 > 0]",
        "input_dimensions": "16 Gaussian features + 2D one-hot task indicator = 18",
        "router_architecture": "Linear(128,2), Softmax(dim=-1)",
        "expert_architecture": "two identical Linear(128,256)→GELU→Linear(256,128)",
        "optimizer": "AdamW",
        "learning_rate": 0.0003,
        "lr_schedule": "500-step linear warmup then cosine decay",
        "weight_decay": 0.0001,
        "batch_size": 256,
        "training_steps": 20000,
        "early_phase_steps": 4000,
        "common_phase_steps": 16000,
        "lambda_balance": 0.01,
        "evaluation_examples_per_task": 50000,
        "S_route_definition": "0.5 * sum_e abs(q_A,e - q_B,e)",
        "S_causal_definition": "0.5 * sum_e abs(Delta_e,A - Delta_e,B)",
        "D_s_definition": "mean(S_causal_BLOCKED_AB_s, S_causal_BLOCKED_BA_s) - S_causal_INTERLEAVED_s",
        "primary_knockout": "renormalized: survivor weight 1",
        "curricula": "INTERLEAVED, BLOCKED_AB, BLOCKED_BA with identical 16,000-step common phase",
    }
    pilot = dict(common); replication = dict(common)
    pilot.update({"training_data_seed": pilot_config["data_seed"], "test_data_seed": pilot_test_seed, "requested_device": pilot_config.get("device", "auto"), "router_entropy_definition": "entropy of task-mean routing vectors in original result JSON", "deterministic_algorithms": "not enabled in original Pilot", "metadata_hashes": "config/source hashes unavailable in original result metadata", "knockout_variants": "renormalized primary only"})
    replication.update({"training_data_seed": replication_config["data_seed"], "test_data_seed": replication_config["test_data_seed"], "requested_device": replication_config.get("device", "auto"), "router_entropy_definition": "true mean per-example entropy", "deterministic_algorithms": "enabled with strict unsupported-operation failure", "metadata_hashes": "config hash and source hash recorded per run", "knockout_variants": "renormalized primary and non-renormalized secondary"})
    diff = []
    for field in sorted(set(pilot) | set(replication)):
        p = pilot.get(field); r = replication.get(field); diff.append({"field": field, "pilot_1": p, "replication_1": r, "status": "MATCH" if p == r else "DIFFERENCE"})
    return {"pilot_1": pilot, "replication_1": replication, "diff": diff}


def _curriculum_integrity(config, seeds):
    checks = {"task_counts_balanced": True, "mature_schedule_shared_within_triplet": True}
    for seed in seeds:
        mature = []
        for condition in CONDITIONS:
            curriculum = build_curriculum(condition, config["early_steps"], config["steps"], config["data_seed"] + seed)
            checks["task_counts_balanced"] &= curriculum.tasks.count("A") == curriculum.tasks.count("B")
            mature.append(curriculum.mature)
        checks["mature_schedule_shared_within_triplet"] &= len(set(mature)) == 1
    return {key: bool(value) for key, value in checks.items()}


def _finite(value):
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    return True


def _checkpoint_integrity(root, seeds):
    expected = {"initial.pt", "step_4000.pt", "step_10000.pt", "step_20000.pt"}
    observed = []
    for seed in seeds:
        for condition in CONDITIONS:
            paths = {os.path.basename(path) for path in glob.glob(os.path.join(root, f"seed_{seed}", condition, "*.pt"))}
            observed.append(paths == expected)
    return all(observed), len(observed) * 4 if all(observed) else sum(len({os.path.basename(p) for p in glob.glob(os.path.join(root, f"seed_{s}", c, "*.pt"))}) for s in seeds for c in CONDITIONS)


def _integrity(rows_by_experiment, configs, roots):
    result = {}
    for experiment, rows in rows_by_experiment.items():
        seeds = sorted({r["seed"] for r in rows})
        grouped = _group(rows)
        complete = sorted(seed for seed, conditions in grouped.items() if set(conditions) == set(CONDITIONS))
        checks = {
            "expected_run_files": len(rows) == 60,
            "complete_seed_triplets": len(complete) == 20 and complete == seeds,
            "no_duplicate_seed_condition_runs": len({(r["seed"], r["condition"]) for r in rows}) == len(rows),
            "no_missing_seeds": seeds == (list(range(20)) if experiment == "Pilot 1" else list(range(100, 120))),
            "initial_checksum_matching": all(len({grouped[s][c]["initial_checksum"] for c in CONDITIONS}) == 1 for s in complete),
            "no_nan_or_inf_metrics": all(_finite(r) for r in rows),
        }
        curriculum = _curriculum_integrity(configs[experiment], seeds)
        checks.update({"exposure_matching": curriculum["task_counts_balanced"], "common_mature_schedule_matching": curriculum["mature_schedule_shared_within_triplet"]})
        checkpoint_ok, checkpoint_count = _checkpoint_integrity(roots[experiment], seeds)
        checks["checkpoint_completeness"] = checkpoint_ok
        metadata = [r.get("metadata", {}) for r in rows]
        commits = {m.get("git_commit") for m in metadata}; config_hashes = {m.get("config_hash") for m in metadata}; source_hashes = {m.get("source_hash") for m in metadata}
        checks["configuration_hash_consistent"] = len(config_hashes) == 1 and None not in config_hashes
        checks["source_hash_consistent"] = len(source_hashes) == 1 and None not in source_hashes
        checks["source_commit_available"] = len(commits) == 1 and "unknown" not in commits and None not in commits
        statuses = {key: ("PASS" if value else "FAIL") for key, value in checks.items()}
        if experiment == "Pilot 1":
            statuses["configuration_hash_consistent"] = "WARN"; statuses["source_hash_consistent"] = "WARN"; statuses["source_commit_available"] = "WARN"
        result[experiment] = {"checks": checks, "statuses": statuses, "pass": all(value for key, value in checks.items() if key not in ("configuration_hash_consistent", "source_hash_consistent", "source_commit_available") or experiment != "Pilot 1"), "n_runs": len(rows), "n_paired_seeds": len(complete), "checkpoint_count": checkpoint_count, "metadata_values": {"git_commit": sorted(map(str, commits)), "config_hash": sorted(map(str, config_hashes)), "source_hash": sorted(map(str, source_hashes))}}
    result["cross_experiment_sources_separate"] = True
    return result


def _permutation_distance(rows):
    output = []
    for pair in _paired_rows(rows):
        seed = pair["seed"]
        grouped = _group(rows)[seed]
        matrices = {}
        for condition, row in grouped.items():
            ablation = _ablation(row, "renormalized")
            matrices[condition] = torch.tensor([[row["routing_A"][0], row["routing_B"][0], ablation["expert_1"]["delta_A"], ablation["expert_1"]["delta_B"]], [row["routing_A"][1], row["routing_B"][1], ablation["expert_2"]["delta_A"], ablation["expert_2"]["delta_B"]]], dtype=torch.float32)
        output.append({"seed": seed, "interleaved_blocked_ab": perm_matrix_distance(matrices["INTERLEAVED"], matrices["BLOCKED_AB"]), "interleaved_blocked_ba": perm_matrix_distance(matrices["INTERLEAVED"], matrices["BLOCKED_BA"])})
    return output


def _dynamics(rows, checkpoint_records):
    routing = []
    for experiment, values in rows.items():
        for condition in CONDITIONS:
            histories = [r["history"] for r in values if r["condition"] == condition]
            steps = sorted({point["step"] for history in histories for point in history})
            for step in steps:
                points = [point for history in histories for point in history if point["step"] == step]
                routing.append({"experiment": experiment, "condition": condition, "step": step, "S_route_mean": float(np.mean([point["routing_specialization"] for point in points]))})
    causal = []
    for experiment, records in checkpoint_records.items():
        for step in (4000, 10000, 20000):
            for condition in CONDITIONS:
                values = [r for r in records if r["step"] == step and r["condition"] == condition]
                causal.append({"experiment": experiment, "condition": condition, "step": step, "S_causal_mean": float(np.mean([_causal(r, "renormalized") for r in values])), "n": len(values), "label": "POST-HOC EXPLORATORY ANALYSIS" if experiment == "Pilot 1" and step in (4000, 10000) else "PRIMARY/SECONDARY CHECKPOINT ANALYSIS"})
    return {"routing": routing, "causal": causal}


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def _seed_comparison(rows_by_experiment):
    grouped = {name: _group(rows) for name, rows in rows_by_experiment.items()}
    output = []
    for seed in sorted(set(grouped["Pilot 1"]) & set(grouped["Replication 1"])):
        row = {"pilot_seed": seed, "replication_seed": seed + 100, "pilot_D_s": _paired_rows(rows_by_experiment["Pilot 1"])[seed]["D_s"] if False else None}
        # Pair by within-experiment ordinal, not by reusing numeric seed identities.
        output.append(row)
    pilot_pairs = _paired_rows(rows_by_experiment["Pilot 1"]); replication_pairs = _paired_rows(rows_by_experiment["Replication 1"])
    output = []
    for pilot, replication in zip(pilot_pairs, replication_pairs):
        row = {"pilot_seed": pilot["seed"], "replication_seed": replication["seed"], "pilot_D_s": pilot["D_s"], "replication_D_s": replication["D_s"]}
        for mode in MODES:
            pp = _paired_rows(rows_by_experiment["Pilot 1"], mode)[len(output)]
            rr = _paired_rows(rows_by_experiment["Replication 1"], mode)[len(output)]
            row[f"pilot_D_s_{mode}"] = pp["D_s"]; row[f"replication_D_s_{mode}"] = rr["D_s"]
        for condition in CONDITIONS:
            p = grouped["Pilot 1"][pilot["seed"]][condition]; r = grouped["Replication 1"][replication["seed"]][condition]
            for field in ("accuracy_A", "accuracy_B", "routing_specialization"):
                row[f"pilot_{condition}_{field}"] = p[field]; row[f"replication_{condition}_{field}"] = r[field]
            for mode in MODES:
                row[f"pilot_{condition}_S_causal_{mode}"] = _causal(p, mode); row[f"replication_{condition}_S_causal_{mode}"] = _causal(r, mode)
        output.append(row)
    return output


def _figures(rows_by_experiment, accuracy, stats_by_experiment, dynamics, output_dir):
    os.makedirs(output_dir, exist_ok=True); experiments = list(EXPERIMENTS); x = np.arange(3); width = .35
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, task in zip(axes, ("A", "B")):
        for j, experiment in enumerate(experiments):
            selected = [r for r in accuracy if r["experiment"] == experiment]
            means = [next(r for r in selected if r["condition"] == c)[f"accuracy_{task}_mean"] for c in CONDITIONS]
            ax.bar(x + (j-.5)*width, means, width, label=experiment, alpha=.75)
        ax.set_title(f"Task {task}"); ax.set_xticks(x, list(CONDITIONS)); ax.set_ylim(.94, 1); ax.set_ylabel("Accuracy"); ax.grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False); fig.suptitle("Pilot 1 vs Replication 1: final accuracy"); fig.tight_layout(); fig.savefig(os.path.join(output_dir, "pilot_vs_replication_accuracy.png"), dpi=220); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); positions = np.arange(2); vals = [[r["D_s"] for r in _paired_rows(rows_by_experiment[e])] for e in experiments]; ax.boxplot(vals, tick_labels=experiments, showfliers=False)
    for i, values in enumerate(vals, 1): ax.scatter(np.full(len(values), i), values, color="#333", alpha=.65, s=20)
    ax.axhline(0, color="#222", linewidth=1); ax.set_ylabel("D_s"); ax.set_title("Primary paired endpoint distributions"); ax.grid(axis="y", alpha=.2); fig.tight_layout(); fig.savefig(os.path.join(output_dir, "pilot_vs_replication_Ds.png"), dpi=220); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5)); offsets = [-.18, .18]
    for j, experiment in enumerate(experiments):
        selected = _condition_metrics(rows_by_experiment[experiment], "renormalized")
        means = [next(r for r in selected if r["condition"] == c)["causal_specialization_mean"] for c in CONDITIONS]
        ax.bar(x + offsets[j], means, .34, label=experiment, alpha=.75)
    ax.set_xticks(x, list(CONDITIONS)); ax.set_ylabel("Mean S_causal"); ax.set_title("Renormalized causal specialization"); ax.legend(frameon=False); ax.grid(axis="y", alpha=.2); fig.tight_layout(); fig.savefig(os.path.join(output_dir, "pilot_vs_replication_Scausal.png"), dpi=220); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for j, experiment in enumerate(experiments):
        selected = _condition_metrics(rows_by_experiment[experiment], "renormalized")
        means = [next(r for r in selected if r["condition"] == c)["routing_specialization_mean"] for c in CONDITIONS]
        ax.bar(x + offsets[j], means, .34, label=experiment, alpha=.75)
    ax.set_xticks(x, list(CONDITIONS)); ax.set_ylabel("Mean S_route"); ax.set_title("Routing specialization"); ax.legend(frameon=False); ax.grid(axis="y", alpha=.2); fig.tight_layout(); fig.savefig(os.path.join(output_dir, "pilot_vs_replication_Sroute.png"), dpi=220); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5)); bins = np.linspace(min(min(r["D_s"] for r in _paired_rows(rows_by_experiment[e])) for e in experiments), max(max(r["D_s"] for r in _paired_rows(rows_by_experiment[e])) for e in experiments), 10)
    for experiment, color in zip(experiments, ("#777777", "#3274A1")): ax.hist([r["D_s"] for r in _paired_rows(rows_by_experiment[experiment])], bins=bins, alpha=.6, label=experiment, color=color)
    ax.axvline(0, color="#222", linewidth=1); ax.set_xlabel("D_s"); ax.set_ylabel("Number of seeds"); ax.set_title("Seed-level D_s distributions"); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(os.path.join(output_dir, "pilot_vs_replication_seed_distribution.png"), dpi=220); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for experiment, linestyle in zip(experiments, ("--", "-")):
        for condition, color in COLORS.items():
            points = [r for r in dynamics["routing"] if r["experiment"] == experiment and r["condition"] == condition]
            points.sort(key=lambda r: r["step"]); axes[0].plot([r["step"] for r in points], [r["S_route_mean"] for r in points], linestyle=linestyle, color=color, label=f"{experiment} {condition}")
            causal = [r for r in dynamics["causal"] if r["experiment"] == experiment and r["condition"] == condition]
            causal.sort(key=lambda r: r["step"]); axes[1].plot([r["step"] for r in causal], [r["S_causal_mean"] for r in causal], linestyle=linestyle, color=color, marker="o", label=f"{experiment} {condition}")
    axes[0].axvline(4000, color="#222", linestyle=":"); axes[1].axvline(4000, color="#222", linestyle=":"); axes[0].set_title("S_route(t)"); axes[1].set_title("S_causal(t), renormalized"); axes[0].set_xlabel("Step"); axes[1].set_xlabel("Step"); axes[0].set_ylabel("Score"); axes[1].set_ylabel("Score"); axes[1].legend(fontsize=7, frameon=False, ncol=2)
    for ax in axes: ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(os.path.join(output_dir, "pilot_vs_replication_dynamics.png"), dpi=220); plt.close(fig)


def _md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(str(v) for v in row) + " |" for row in rows)
    return "\n".join(lines)


def _report(config_diff, integrity, accuracy, stats, causal, routing, discrepancies, source_inventory):
    config_rows = [(d["field"], d["pilot_1"], d["replication_1"], d["status"]) for d in config_diff["diff"]]
    accuracy_rows = [(r["experiment"], r["condition"], f'{r["accuracy_A_mean"]:.6f}', f'{r["accuracy_B_mean"]:.6f}', f'{r["balanced_accuracy_mean"]:.6f}', f'{r["balanced_accuracy_min"]:.6f}') for r in accuracy]
    d_rows = [(e, stats[e]["n"], f'{stats[e]["mean_difference"]:.6f}', f'{stats[e]["median_difference"]:.6f}', f'{stats[e]["sd_difference"]:.6f}', f'[{stats[e]["ci_low"]:.6f}, {stats[e]["ci_high"]:.6f}]', f'{stats[e]["paired_t_p"]:.4g}', f'{stats[e]["wilcoxon_p"]:.4g}', f'{stats[e]["cohens_dz"]:.4f}', f'{stats[e]["positive_seeds"]}/{stats[e]["negative_seeds"]}', f'[{stats[e]["min_D_s"]:.6f}, {stats[e]["max_D_s"]:.6f}]') for e in EXPERIMENTS]
    causal_rows = []
    for e in EXPERIMENTS:
        for row in causal[e]["renormalized"]:
            causal_rows.append((e, row["condition"], f'{row["causal_specialization_mean"]:.6f}', f'{row["causal_specialization_sd"]:.6f}', f'{row["max_absolute_knockout_effect_mean"]:.6f}', f'{row["max_absolute_knockout_effect_sd"]:.6f}'))
    routing_rows = [(e, r["condition"], f'{r["routing_specialization_mean"]:.6f}', f'{r["routing_specialization_sd"]:.6f}', f'{r["router_entropy_mean"]:.6f}', f'{r["router_collapse_count"]}/{r["n"]}') for e in EXPERIMENTS for r in routing[e]]
    integrity_rows = [(e, key, value, integrity[e]["statuses"][key]) for e in EXPERIMENTS for key, value in integrity[e]["checks"].items()]
    return f"""# Pilot vs Replication Quantitative Comparison

This report compares the completed Pilot 1 and Replication 1 artifacts without new training, pooling, tuning, or modification of existing result files. All inferential statistics are computed separately with `n=20` paired seeds per experiment.

## Experimental configuration comparison

{_md_table(["Field", "Pilot 1", "Replication 1", "Status"], config_rows)}

Configuration differences are reported explicitly. The training architecture, parameter count, tasks, optimizer, schedule, curriculum definitions, primary metric, and primary knockout definition match. The data seeds, diagnostic instrumentation, and availability of the secondary knockout differ as documented above.

## Data integrity

{_md_table(["Experiment", "Check", "Value", "Status"], integrity_rows)}

## Behavioral results

{_md_table(["Experiment", "Condition", "Task A mean", "Task B mean", "Balanced mean", "Minimum balanced"], accuracy_rows)}

The blocked-minus-interleaved mean accuracy differences are reported per experiment in `comparison.json`; no accuracy values were normalized across experiments.

## Primary D_s results

{_md_table(["Experiment", "n", "Mean", "Median", "SD", "95% bootstrap CI", "t p", "Wilcoxon p", "dz", "+/- seeds", "Min/Max"], d_rows)}

The Pilot and Replication samples are separate. Negative seed-level values remain in the machine-readable comparison and figures.

## Seed-level distributions

The seed-level comparison aligns seeds by within-experiment ordinal position: Pilot seeds 0–19 and Replication seeds 100–119 are not treated as the same initialization seeds. The full values, quartiles, positive/negative counts, and extremes are in `seed_level_comparison.csv` and `comparison.json`.

![Seed-level D_s distributions](../../figures/pilot_vs_replication_seed_distribution.png)

## Causal specialization

{_md_table(["Experiment", "Condition", "S_causal mean", "S_causal SD", "Max |knockout Δ| mean", "Max |knockout Δ| SD"], causal_rows)}

The table above uses the renormalized knockout. Replication non-renormalized results are retained separately in `comparison.json`, `comparison.csv`, and `seed_level_comparison.csv`.

![Causal specialization](../../figures/pilot_vs_replication_Scausal.png)

## Routing specialization

{_md_table(["Experiment", "Condition", "S_route mean", "S_route SD", "Router entropy mean", "Collapse events"], routing_rows)}

Router entropy is not treated as interchangeable across implementations: Pilot original result JSON stored entropy of task-mean routing vectors, while the post-hoc/final comparison uses the checkpoint-derived true per-example entropy where available. Collapse events are reported, not excluded.

![Routing specialization](../../figures/pilot_vs_replication_Sroute.png)

## Developmental dynamics

Both experiments have routing history checkpoints and checkpoint-derived causal measurements at steps 4,000, 10,000, and 20,000. Pilot steps 4,000 and 10,000 are labeled `POST-HOC EXPLORATORY ANALYSIS`; they are not part of Pilot 1's original confirmatory endpoint.

![Developmental dynamics](../../figures/pilot_vs_replication_dynamics.png)

## Knockout analysis

The primary renormalized knockout removes the selected expert and assigns the surviving expert weight 1. The replication's secondary non-renormalized knockout zeros the removed contribution while retaining the survivor's original probability. The two procedures are kept separate in all artifacts; neither is silently substituted for the other.

## Quantitative discrepancies

{_md_table(["Metric", "Value"], [(k, f'{v}') for k, v in discrepancies.items()])}

These are measured differences only. This comparison does not assign causes to them.

## Machine-readable artifact inventory

{_md_table(["Artifact", "Purpose"], source_inventory)}

Factual summary: both experiments used the same recorded training specification and showed high, closely matched final task accuracy. Replication 1 had a positive primary paired `D_s` for all 20 seeds; Pilot 1 had a positive `D_s` for 16 of 20 seeds. Mean primary `D_s` was 0.148851 in Pilot 1 and 0.123148 in Replication 1. Replication 1 also reports the separately defined non-renormalized knockout results. No general-hypothesis conclusion is made here.
"""


def compare(output_root="results/pilot_vs_replication", figure_root="figures"):
    output = Path(output_root); output.mkdir(parents=True, exist_ok=True)
    pilot_config = yaml.safe_load(open("configs/default.yaml", encoding="utf-8")); replication_config = yaml.safe_load(open("replication_config.yaml", encoding="utf-8"))
    pilot_raw = _load_raw("results/pilot_v1_frozen/raw"); replication_raw = _load_raw("results/replication/raw")
    pilot = _merge_final(pilot_raw, _load_posthoc("results/pilot_v1_posthoc/analysis.json")); replication = replication_raw
    rows = {"Pilot 1": pilot, "Replication 1": replication}; configs = {"Pilot 1": pilot_config, "Replication 1": replication_config}; roots = {"Pilot 1": "results/pilot_v1_frozen/raw", "Replication 1": "results/replication/raw"}
    config_diff = _canonical_configuration(pilot_config, replication_config); integrity = _integrity(rows, configs, roots); accuracy = _accuracy_comparison(rows)
    stats = {experiment: _stats(values, "renormalized") for experiment, values in rows.items()}
    causal = {experiment: {mode: _condition_metrics(values, mode) for mode in MODES} for experiment, values in rows.items()}
    routing = {experiment: causal[experiment]["renormalized"] for experiment in EXPERIMENTS}
    checkpoint_files = {"Pilot 1": "results/pilot_v1_posthoc/analysis.json", "Replication 1": "results/replication/checkpoint_analysis/analysis.json"}
    checkpoint_records = {experiment: _load_posthoc(path) for experiment, path in checkpoint_files.items()}
    dynamics = _dynamics(rows, checkpoint_records); distances = {experiment: _permutation_distance(values) for experiment, values in rows.items()}
    seed_comparison = _seed_comparison(rows)
    pilot_stats = stats["Pilot 1"]; replication_stats = stats["Replication 1"]
    discrepancies = {"mean_D_s_replication_minus_pilot": replication_stats["mean_difference"] - pilot_stats["mean_difference"], "cohens_dz_replication_minus_pilot": replication_stats["cohens_dz"] - pilot_stats["cohens_dz"], "bootstrap_CI_width_pilot": pilot_stats["ci_high"] - pilot_stats["ci_low"], "bootstrap_CI_width_replication": replication_stats["ci_high"] - replication_stats["ci_low"], "positive_seed_proportion_pilot": pilot_stats["positive_seeds"] / pilot_stats["n"], "positive_seed_proportion_replication": replication_stats["positive_seeds"] / replication_stats["n"], "mean_balanced_accuracy_pilot": float(np.mean([(r["accuracy_A"] + r["accuracy_B"]) / 2 for r in pilot])), "mean_balanced_accuracy_replication": float(np.mean([(r["accuracy_A"] + r["accuracy_B"]) / 2 for r in replication])), "pilot_replication_data_seed_difference": replication_config["data_seed"] - pilot_config["data_seed"], "pilot_replication_test_seed_difference": replication_config["test_data_seed"] - (pilot_config["data_seed"] + 9_000_000)}
    source_inventory = [("Pilot raw results", "results/pilot_v1_frozen/raw/"), ("Pilot checkpoint ablation", "results/pilot_v1_posthoc/analysis.json"), ("Pilot configuration", "configs/default.yaml"), ("Replication raw results", "results/replication/raw/"), ("Replication checkpoint ablation", "results/replication/checkpoint_analysis/analysis.json"), ("Replication analysis", "results/replication/analysis.json"), ("Replication manifest", "replication_manifest.md"), ("Replication config", "replication_config.yaml")]
    comparison = {"source_files": {"pilot_raw": "results/pilot_v1_frozen/raw/", "pilot_posthoc": "results/pilot_v1_posthoc/analysis.json", "pilot_config": "configs/default.yaml", "replication_raw": "results/replication/raw/", "replication_checkpoint_analysis": "results/replication/checkpoint_analysis/analysis.json", "replication_config": "replication_config.yaml"}, "configuration": config_diff, "integrity": integrity, "accuracy": accuracy, "primary_D_s": stats, "causal_specialization": causal, "routing": routing, "dynamics": dynamics, "permutation_invariant_distance": distances, "seed_level_comparison": seed_comparison, "quantitative_discrepancies": discrepancies, "artifacts": source_inventory}
    with open(output / "comparison.json", "w", encoding="utf-8") as f: json.dump(comparison, f, indent=2, allow_nan=False)
    with open(output / "integrity_audit.json", "w", encoding="utf-8") as f: json.dump(integrity, f, indent=2, allow_nan=False)
    with open(output / "config_diff.json", "w", encoding="utf-8") as f: json.dump(config_diff, f, indent=2, allow_nan=False)
    config_md = "# Pilot 1 vs Replication 1 Configuration Diff\n\n" + _md_table(["Field", "Pilot 1", "Replication 1", "Status"], [(d["field"], d["pilot_1"], d["replication_1"], d["status"]) for d in config_diff["diff"]]) + "\n"
    with open(output / "config_diff.md", "w", encoding="utf-8") as f: f.write(config_md)
    _write_csv(output / "comparison.csv", accuracy)
    _write_csv(output / "seed_level_comparison.csv", seed_comparison)
    _figures(rows, accuracy, stats, dynamics, figure_root)
    report = _report(config_diff, integrity, accuracy, stats, causal, routing, discrepancies, source_inventory)
    with open(output / "analysis_report.md", "w", encoding="utf-8") as f: f.write(report)
    return comparison


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--output", default="results/pilot_vs_replication"); parser.add_argument("--figures", default="figures"); args = parser.parse_args(argv); print(json.dumps(compare(args.output, args.figures), indent=2))


if __name__ == "__main__":
    main()
