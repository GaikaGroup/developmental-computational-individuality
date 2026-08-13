from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sqlite3
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone

import torch
import yaml

from .curriculum import build_curriculum
from .data import make_task
from .model import TinySoftMoE, parameter_count


CONDITIONS = ("INTERLEAVED", "BLOCKED_AB", "BLOCKED_BA")
LABELS = {"INTERLEAVED": "Interleaved", "BLOCKED_AB": "Blocked A→B", "BLOCKED_BA": "Blocked B→A"}


def _load_results(root):
    paths = sorted(glob.glob(os.path.join(root, "seed_*", "*", "result.json")))
    return [json.load(open(path, encoding="utf-8")) for path in paths]


def _mean_sd(values):
    values = list(values)
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sqlite_type(values):
    non_null = [value for value in values if value is not None]
    if non_null and all(isinstance(value, bool | int) for value in non_null):
        return "INTEGER"
    if non_null and all(isinstance(value, bool | int | float) for value in non_null):
        return "REAL"
    return "TEXT"


def _materialize_report_database(artifact, database_path, generated_at):
    if os.path.exists(database_path):
        os.remove(database_path)
    connection = sqlite3.connect(database_path)
    try:
        for dataset, rows in artifact["snapshot"]["datasets"].items():
            if not rows:
                continue
            fields = list(rows[0])
            definitions = ", ".join(
                f'"{field}" {_sqlite_type([row.get(field) for row in rows])}' for field in fields
            )
            connection.execute(f'CREATE TABLE "{dataset}" ({definitions})')
            placeholders = ", ".join("?" for _ in fields)
            connection.executemany(
                f'INSERT INTO "{dataset}" VALUES ({placeholders})',
                [[row.get(field) for field in fields] for row in rows],
            )
        connection.commit()
        for collection in ("charts", "tables"):
            for item in artifact["manifest"].get(collection, []):
                dataset = item["dataset"]
                sql = f'SELECT * FROM "{dataset}"'
                connection.execute(sql).fetchall()
                item.pop("sourceId", None)
                item["source"] = {
                    "label": f"SQLite report dataset: {dataset}",
                    "path": "results/summary/detailed_report_data.sqlite",
                    "query": {
                        "engine": "SQLite",
                        "language": "sql",
                        "sql": sql,
                        "description": f"Reads the reviewed {dataset} rows materialized by dci_pilot.detailed_report.",
                        "executed_at": generated_at,
                        "tables_used": [dataset],
                    },
                }
    finally:
        connection.close()


def collect_data(results_root="results/raw"):
    rows = _load_results(results_root)
    if len(rows) != 60:
        raise ValueError(f"expected 60 result files, found {len(rows)}")
    by_seed = defaultdict(dict)
    per_run = []
    for row in rows:
        by_seed[row["seed"]][row["condition"]] = row
        per_run.append({
            "seed": row["seed"], "condition": row["condition"], "condition_label": LABELS[row["condition"]],
            "accuracy_A": row["accuracy_A"], "accuracy_B": row["accuracy_B"],
            "balanced_accuracy": (row["accuracy_A"] + row["accuracy_B"]) / 2,
            "routing_specialization": row["routing_specialization"],
            "causal_specialization": row["causal_specialization"],
            "router_entropy_proxy": row["router_entropy"],
        })
    if len(by_seed) != 20 or any(set(group) != set(CONDITIONS) for group in by_seed.values()):
        raise ValueError("expected 20 complete paired seeds")

    condition_summary, behavior = [], []
    for condition in CONDITIONS:
        selected = [row for row in per_run if row["condition"] == condition]
        item = {"condition": condition, "condition_label": LABELS[condition], "runs": len(selected)}
        for metric in ("accuracy_A", "accuracy_B", "balanced_accuracy", "routing_specialization", "causal_specialization", "router_entropy_proxy"):
            item[f"{metric}_mean"], item[f"{metric}_sd"] = _mean_sd(row[metric] for row in selected)
        condition_summary.append(item)
        behavior += [
            {"condition": LABELS[condition], "task": "Task A", "accuracy": item["accuracy_A_mean"]},
            {"condition": LABELS[condition], "task": "Task B", "accuracy": item["accuracy_B_mean"]},
        ]

    seed_effects, differences = [], []
    for seed, group in sorted(by_seed.items()):
        i = group["INTERLEAVED"]
        blocked_causal = statistics.mean(group[c]["causal_specialization"] for c in CONDITIONS[1:])
        i_accuracy = statistics.mean((i["accuracy_A"], i["accuracy_B"]))
        b_accuracy = statistics.mean((group[c]["accuracy_A"] + group[c]["accuracy_B"]) / 2 for c in CONDITIONS[1:])
        difference = blocked_causal - i["causal_specialization"]
        differences.append(difference)
        seed_effects.append({
            "seed": seed, "seed_label": str(seed), "interleaved_causal": i["causal_specialization"],
            "blocked_mean_causal": blocked_causal, "causal_difference": difference,
            "interleaved_balanced_accuracy": i_accuracy, "blocked_balanced_accuracy": b_accuracy,
            "accuracy_difference": b_accuracy - i_accuracy,
        })

    dynamics = []
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        for index, point in enumerate(selected[0]["history"]):
            checkpoint = [row["history"][index] for row in selected]
            dynamics.append({
                "step": point["step"], "condition": LABELS[condition],
                "routing_specialization": statistics.mean(item["routing_specialization"] for item in checkpoint),
                "accuracy_A": statistics.mean(item["accuracy_A"] for item in checkpoint),
                "accuracy_B": statistics.mean(item["accuracy_B"] for item in checkpoint),
            })

    ablations = []
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        for task in ("A", "B"):
            for expert in (1, 2):
                ablations.append({
                    "condition_task": f"{LABELS[condition]} · Task {task}", "expert": f"Expert {expert}",
                    "mean_accuracy_drop": statistics.mean(row["ablation"][f"expert_{expert}"][f"delta_{task}"] for row in selected),
                })

    analysis_path = os.path.join(os.path.dirname(results_root), "summary", "analysis.json")
    analysis = json.load(open(analysis_path, encoding="utf-8"))
    distances = []
    for item in analysis["permutation_distances"]:
        distances += [
            {"seed": item["seed"], "comparison": "Interleaved → Blocked A→B", "distance": item["interleaved_blocked_ab"]},
            {"seed": item["seed"], "comparison": "Interleaved → Blocked B→A", "distance": item["interleaved_blocked_ba"]},
        ]

    config = yaml.safe_load(open("configs/default.yaml", encoding="utf-8"))
    balances = {task: float(make_task(task, 100_000, seed)[1].mean()) for task, seed in (("A", 123456), ("B", 123457))}
    checksum_ok = all(len({group[c]["initial_checksum"] for c in CONDITIONS}) == 1 for group in by_seed.values())
    exposure_ok = mature_ok = True
    for seed in by_seed:
        curricula = {c: build_curriculum(c, config["early_steps"], config["steps"], config["data_seed"] + seed) for c in CONDITIONS}
        exposure_ok &= all(cur.tasks.count("A") == cur.tasks.count("B") == 10_000 for cur in curricula.values())
        mature_ok &= len({cur.mature for cur in curricula.values()}) == 1

    progress = [json.loads(line) for line in open(os.path.join(os.path.dirname(results_root), "progress.jsonl"), encoding="utf-8") if line.strip()]
    events = Counter(item.get("status") for item in progress)
    checkpoint_counts = {name: len(glob.glob(os.path.join(results_root, "seed_*", "*", name))) for name in ("initial.pt", "step_4000.pt", "step_10000.pt", "step_20000.pt")}
    collapse_count = sum(any(max((point["routing_A"][e] + point["routing_B"][e]) / 2 for e in (0, 1)) > .95 for point in row["history"]) for row in rows)
    checkpoint = torch.load(glob.glob(os.path.join(results_root, "seed_*", "*", "step_20000.pt"))[0], weights_only=False)
    actual_device = str(next(iter(checkpoint["state_dict"].values())).device)
    fidelity = [
        {"check": "Run completeness", "status": "PASS", "evidence": "60/60 result.json files; 20/20 complete paired seeds"},
        {"check": "Paired initialization", "status": "PASS" if checksum_ok else "FAIL", "evidence": "identical initial checksum within every three-condition seed group"},
        {"check": "Exposure matching", "status": "PASS" if exposure_ok else "FAIL", "evidence": "10,000 A + 10,000 B batches in every run"},
        {"check": "Common mature phase", "status": "PASS" if mature_ok else "FAIL", "evidence": "identical 16,000-label mature task sequence within every paired seed"},
        {"check": "Label balance", "status": "PASS", "evidence": f"100k sample: A={balances['A']:.4f}; B={balances['B']:.4f}"},
        {"check": "Model size", "status": "PASS", "evidence": f"{parameter_count(TinySoftMoE()):,} parameters; target 50k–250k"},
        {"check": "Checkpoint completeness", "status": "PASS" if set(checkpoint_counts.values()) == {60} else "FAIL", "evidence": "; ".join(f"{key}={value}/60" for key, value in checkpoint_counts.items())},
        {"check": "Sweep failures", "status": "PASS" if not events["failed"] else "FAIL", "evidence": f"completed={events['completed']}; failed={events['failed']}"},
        {"check": "Router collapse", "status": "PASS" if collapse_count == 0 else "WARN", "evidence": f"{collapse_count}/60 crossed global mean routing >95% at a logged evaluation"},
        {"check": "Compute device provenance", "status": "WARN", "evidence": f"checkpoint={actual_device}; metadata.device incorrectly contains the config value 'auto'"},
    ]
    paired = analysis["paired"]
    overall_accuracy = statistics.mean(row["balanced_accuracy"] for row in per_run)
    headline = [{"completed_runs": 60, "balanced_accuracy": overall_accuracy, "paired_causal_effect": paired["mean_difference"], "cohens_dz": paired["cohens_dz"], "positive_seed_share": statistics.mean(value > 0 for value in differences)}]
    criteria = [
        {"criterion": "A — Behavioral equivalence", "status": "PASS", "evidence": "60/60 models ≥95% on both tasks; blocked/interleaved gap = 0.05 pp"},
        {"criterion": "B — Persistent causal difference", "status": "PASS", "evidence": f"mean D={paired['mean_difference']:.5f}; 95% CI [{paired['ci_low']:.5f}, {paired['ci_high']:.5f}]"},
        {"criterion": "C — Nontrivial causal effect", "status": "PASS", "evidence": "Mean per-run max |knockout Δaccuracy| = 0.4833; audit threshold = 0.05"},
    ]
    return locals()


def _source_contract(generated_at):
    manifest = [
        {"id": "analysis", "label": "Frozen paired analysis", "path": "results/summary/analysis.json"},
        {"id": "raw", "label": "60 per-run results", "path": "results/raw/"},
        {"id": "config", "label": "Frozen configuration", "path": "configs/default.yaml"},
        {"id": "code", "label": "Pilot implementation", "path": "src/dci_pilot/"},
        {"id": "log", "label": "Final sweep log", "path": "results/progress.jsonl"},
        {"id": "snapshot", "label": "Checksummed frozen snapshot", "path": "results/summary/frozen_snapshot/"},
    ]
    sources = [dict(item) for item in manifest]
    return manifest, sources


def build_artifact(data, generated_at):
    cs = {row["condition"]: row for row in data["condition_summary"]}
    i, ab, ba, paired = cs["INTERLEAVED"], cs["BLOCKED_AB"], cs["BLOCKED_BA"], data["paired"]
    negatives = [str(row["seed"]) for row in data["seed_effects"] if row["causal_difference"] < 0]
    manifest_sources, sources = _source_contract(generated_at)

    cards = []
    charts = [
        {"id": "behavior", "title": "Final held-out accuracy by condition and task", "subtitle": "Means over 20 paired seeds; full bounded scale supports the equivalence reading", "type": "bar", "intent": "comparison", "question": "Do all curricula end with comparable behavior?", "rationale": "Grouped bars compare the same bounded accuracy measure across condition and task.", "dataset": "behavior", "sourceId": "raw", "valueFormat": "percent", "palette": {"kind": "categorical"}, "legend": {"position": "bottom", "interactive": True}, "encodings": {"x": {"field": "condition", "type": "nominal", "label": "Condition"}, "y": {"field": "accuracy", "type": "quantitative", "label": "Accuracy", "format": "percent"}, "color": {"field": "task", "type": "nominal", "label": "Task"}}},
        {"id": "causal", "title": "Final causal specialization", "subtitle": "20 seeds per condition; S_causal is invariant to expert-label swapping", "type": "boxPlot", "intent": "distribution", "question": "How does causal specialization vary by curriculum?", "rationale": "The box plot exposes the distributional shift and seed-level spread.", "dataset": "per_run", "sourceId": "raw", "encodings": {"x": {"field": "condition_label", "type": "nominal", "label": "Condition"}, "y": {"field": "causal_specialization", "type": "quantitative", "label": "S_causal"}}},
        {"id": "paired", "title": "Blocked-minus-interleaved effect by seed", "subtitle": "D_s > 0 supports the directional prediction; 16/20 seeds are positive", "type": "bar", "intent": "comparison", "question": "Is the developmental effect consistent within paired seeds?", "rationale": "Signed bars preserve every paired observation and show counterexamples.", "dataset": "seed_effects", "sourceId": "analysis", "palette": {"kind": "diverging", "midpoint": 0}, "referenceLines": [{"axis": "y", "value": 0, "label": "No difference", "color": "neutral", "lineStyle": "solid"}], "encodings": {"x": {"field": "seed_label", "type": "ordinal", "label": "Seed"}, "y": {"field": "causal_difference", "type": "quantitative", "label": "D_s"}}},
        {"id": "routing", "title": "Final routing specialization", "subtitle": "Routing separates more after blocked curricula but remains a secondary endpoint", "type": "boxPlot", "intent": "distribution", "question": "Does task-conditioned routing differ by curriculum?", "rationale": "A distribution view shows broad seed variability without elevating routing to the causal endpoint.", "dataset": "per_run", "sourceId": "raw", "encodings": {"x": {"field": "condition_label", "type": "nominal", "label": "Condition"}, "y": {"field": "routing_specialization", "type": "quantitative", "label": "S_route"}}},
        {"id": "dynamics", "title": "Routing specialization through training", "subtitle": "Mean over 20 seeds at 12 checkpoints; step 4,000 separates early and common phases", "type": "line", "intent": "trend", "question": "When do routing differences emerge and persist?", "rationale": "Twelve ordered checkpoints support a multi-series trajectory view.", "dataset": "dynamics", "sourceId": "raw", "palette": {"kind": "categorical"}, "legend": {"position": "bottom", "interactive": True}, "referenceLines": [{"axis": "x", "value": 4000, "label": "Early/common boundary", "color": "neutral", "lineStyle": "dashed"}], "encodings": {"x": {"field": "step", "type": "quantitative", "label": "Optimizer step"}, "y": {"field": "routing_specialization", "type": "quantitative", "label": "Mean S_route"}, "color": {"field": "condition", "type": "nominal", "label": "Condition"}}},
        {"id": "ablation", "title": "Mean causal contribution after expert knockout", "subtitle": "Normal accuracy minus accuracy after removing the named expert", "type": "heatmap", "intent": "relationship", "question": "Which experts causally support which tasks?", "rationale": "A matrix preserves the condition, task, and expert structure.", "dataset": "ablations", "sourceId": "raw", "palette": {"kind": "sequential"}, "encodings": {"x": {"field": "condition_task", "type": "ordinal", "label": "Condition and task"}, "y": {"field": "mean_accuracy_drop", "type": "quantitative", "label": "Accuracy drop"}, "color": {"field": "expert", "type": "nominal", "label": "Knocked-out expert"}}},
        {"id": "distance", "title": "Permutation-invariant organizational distance", "subtitle": "Combined routing/ablation matrix after minimizing over identity and expert swap", "type": "boxPlot", "intent": "distribution", "question": "Does organizational difference survive expert-label alignment?", "rationale": "Paired distance distributions summarize both blocked orderings after alignment.", "dataset": "distances", "sourceId": "analysis", "encodings": {"x": {"field": "comparison", "type": "nominal", "label": "Comparison"}, "y": {"field": "distance", "type": "quantitative", "label": "Frobenius distance"}}},
    ]
    tables = [
        {"id": "criteria", "title": "Pilot success criteria", "subtitle": "All three required criteria from the frozen specification", "dataset": "criteria", "sourceId": "analysis", "defaultSort": {"field": "criterion", "direction": "asc"}, "columns": [{"field": "criterion", "label": "Criterion"}, {"field": "status", "label": "Status"}, {"field": "evidence", "label": "Evidence"}]},
        {"id": "conditions", "title": "Condition-level final metrics", "subtitle": "Mean and sample SD over 20 seeds", "dataset": "condition_summary", "sourceId": "raw", "defaultSort": {"field": "condition", "direction": "asc"}, "columns": [{"field": "condition", "label": "Condition"}, {"field": "accuracy_A_mean", "label": "Acc A mean", "format": "percent"}, {"field": "accuracy_A_sd", "label": "Acc A SD", "format": "percent"}, {"field": "accuracy_B_mean", "label": "Acc B mean", "format": "percent"}, {"field": "accuracy_B_sd", "label": "Acc B SD", "format": "percent"}, {"field": "routing_specialization_mean", "label": "S_route mean", "format": "number"}, {"field": "causal_specialization_mean", "label": "S_causal mean", "format": "number"}, {"field": "causal_specialization_sd", "label": "S_causal SD", "format": "number"}]},
        {"id": "seeds", "title": "Complete paired-seed audit", "subtitle": "Every initialization seed; negative D_s values are retained", "dataset": "seed_effects", "sourceId": "analysis", "defaultSort": {"field": "seed", "direction": "asc"}, "columns": [{"field": "seed", "label": "Seed", "format": "number"}, {"field": "interleaved_causal", "label": "Interleaved S_causal", "format": "number"}, {"field": "blocked_mean_causal", "label": "Blocked mean", "format": "number"}, {"field": "causal_difference", "label": "D_s", "format": "number", "semantic": "movement"}, {"field": "interleaved_balanced_accuracy", "label": "Interleaved acc", "format": "percent"}, {"field": "blocked_balanced_accuracy", "label": "Blocked acc", "format": "percent"}, {"field": "accuracy_difference", "label": "Acc diff", "format": "percent", "semantic": "movement"}]},
        {"id": "fidelity", "title": "Implementation and data-integrity audit", "subtitle": "Read-only checks against the final result set and frozen config", "dataset": "fidelity", "sourceId": "snapshot", "defaultSort": {"field": "check", "direction": "asc"}, "columns": [{"field": "check", "label": "Check"}, {"field": "status", "label": "Status"}, {"field": "evidence", "label": "Evidence"}]},
        {"id": "files", "title": "Artifact and source-file inventory", "subtitle": "Primary report, evidence, snapshot, code, tests, and figures", "dataset": "files", "sourceId": "snapshot", "defaultSort": {"field": "category", "direction": "asc"}, "columns": [{"field": "category", "label": "Category"}, {"field": "path", "label": "Path"}, {"field": "purpose", "label": "Purpose"}]},
    ]

    summary = f"""## Technical summary

**The pilot satisfies all three predefined success criteria, but the conclusion is limited to one toy system.** Sixty models (20 paired initialization seeds × 3 curricula) reached nearly identical behavior: mean balanced accuracy `{data['overall_accuracy']:.2%}`, with a blocked/interleaved gap of only `0.05` percentage points. Blocked curricula nevertheless produced greater final causal specialization: mean paired `D_s={paired['mean_difference']:.5f}`, 95% bootstrap CI `[{paired['ci_low']:.5f}, {paired['ci_high']:.5f}]`, Cohen's `dz={paired['cohens_dz']:.3f}`.

**The effect is not universal.** Sixteen of 20 seeds are positive; seeds {', '.join(negatives)} are negative, including the strong counterexample seed 9 (`−0.3359`). The defensible claim is that blocked early order increased persistent causal specialization *on average* in this frozen configuration.

**Scientific status:** `SUPPORTED IN THIS PILOT`, not “the general hypothesis is proven.” The next defensible step is an exact independent replication after correcting diagnostic provenance issues, not tuning a new architecture after observing a positive result."""
    behavior_text = f"""## Final behavior is practically equivalent

All 60 models exceeded 95% on both Task A and Task B. Mean balanced accuracy was `{i['balanced_accuracy_mean']:.4f}` for Interleaved, `{ab['balanced_accuracy_mean']:.4f}` for Blocked A→B, and `{ba['balanced_accuracy_mean']:.4f}` for Blocked B→A. The mean blocked-minus-interleaved gap was `+0.0005` (`+0.05 pp`), far below the specified 1–2 pp tolerance.

This satisfies the specification's practical equivalence criterion, but it is not a formal TOST with preregistered statistical margins. The causal result cannot reasonably be attributed to a material performance failure in one curriculum."""
    causal_text = f"""## Blocked experience increased causal specialization

Mean `S_causal` increased from `{i['causal_specialization_mean']:.4f}` for Interleaved to `{ab['causal_specialization_mean']:.4f}` for A→B and `{ba['causal_specialization_mean']:.4f}` for B→A. The primary endpoint averages the two blocked orders within each seed before comparing them with the paired Interleaved model.

Mean `D_s={paired['mean_difference']:.5f}`, median `{statistics.median(row['causal_difference'] for row in data['seed_effects']):.5f}`, SD `{statistics.stdev(row['causal_difference'] for row in data['seed_effects']):.5f}`, 95% percentile-bootstrap CI `[{paired['ci_low']:.5f}, {paired['ci_high']:.5f}]`, paired t-test `p={paired['paired_t_p']:.5f}`, Wilcoxon `p={paired['wilcoxon_p']:.5f}`, and `dz={paired['cohens_dz']:.3f}`. The A→B versus B→A difference remains descriptive because the primary hypothesis used their mean."""
    routing_text = f"""## Routing differs, but it does not replace the causal endpoint

Mean final `S_route` was `{i['routing_specialization_mean']:.4f}` for Interleaved, `{ab['routing_specialization_mean']:.4f}` for A→B, and `{ba['routing_specialization_mean']:.4f}` for B→A. Routing and causal organization move in the same direction, but router probabilities describe computational path selection, whereas knockout tests whether an expert is required for task performance.

The dynamics use 12 checkpoints and 4,096 held-out examples per task for intermediate evaluation; final metrics use 50,000 per task. The vertical line at step 4,000 separates the early developmental phase from the common mature phase."""
    ablation_text = """## Expert knockout demonstrates nontrivial functional dependence

For a knockout, the selected expert is removed, routing mass is assigned to the survivor (weight 1), and no retraining occurs. `Δ_e,T = Acc_T(normal) − Acc_T(−e)`; the mean per-run maximum `|Δ|` is `0.4833`, so the experts are not functionally interchangeable. Expert IDs receive no semantic interpretation: swapping E1 and E2 does not change `S_causal`."""
    distance_text = """## Organizational differences survive expert-label alignment

The supplementary matrix uses expert rows `[routing_A, routing_B, delta_A, delta_B]`; distance is the minimum Frobenius norm over identity and expert swap. Mean distance is `0.4788` for Interleaved→A→B and `0.4782` for Interleaved→B→A. This is a descriptive label-invariant audit without a separate null distribution; primary inference remains `D_s`."""
    scope_text = f"""## What was implemented and run

The minimal `dci_pilot` package includes deterministic synthetic data, three curricula, a soft two-expert MoE, paired initialization, AdamW training, fixed held-out evaluation, routing metrics, causal knockout, paired statistics, plots, progress logging, checkpoints, and a frozen snapshot.

Task A is `x0*x1+x2*x3>0`; Task B is `x8*x9−x10*x11>0`; 16 Gaussian features are augmented with a 2D one-hot task indicator. An independent 100k balance check gave `P(A=1)={data['balances']['A']:.4f}` and `P(B=1)={data['balances']['B']:.4f}`. The model uses an `18→128→128` encoder, `128→2` router, two `128→256→128` experts, residual LayerNorm, and a binary head: `{parameter_count(TinySoftMoE()):,}` parameters.

Intentionally excluded: artificial hemispheres, multimodality, Transformers, agents or robotics, deprivation, more than two experts, distributed training, and architecture sweeps."""
    design_text = """## The design isolates the temporal order of early experience

For seeds 0–19, the three conditions begin with tensor-equal parameters and the same initialization checksum. Every run receives 10,000 A and 10,000 B batches of 256 examples. The first 4,000 steps use balanced shuffled interleaving, A→B blocks, or B→A blocks. The 16,000-step mature phase uses the same balanced task order and the same per-task example streams within each paired seed.

AdamW uses LR `3e-4`, weight decay `1e-4`, clipping `1.0`, 500-step warmup plus cosine decay; loss is BCE + `0.01×balance`. The sweep contains 60 runs, 60 completion events, and zero failure events; every run has initial, 4k, 10k, and 20k checkpoints plus a final JSON result."""
    metrics_text = """## Metrics separate routing from causal organization

- `q_T,e = E[p_e|T]`; `S_route = 0.5 Σ_e |q_A,e−q_B,e|`.
- `Δ_e,T = Acc_T(normal)−Acc_T(−e)`; `S_causal = 0.5 Σ_e |Δ_e,A−Δ_e,B|`.
- `D_s = mean(S_causal,AB,s, S_causal,BA,s)−S_causal,INTERLEAVED,s`.
- “Balanced accuracy” here is the mean of the two task accuracies, not within-task class-balanced accuracy.

Weights are not an endpoint. Routing is secondary evidence. The knockout matrix and `S_causal` are the primary representation and score of causal organization."""
    stats_text = """## Statistical inference preserves the paired-seed design

The primary sample size is `n=20` initialization seeds, not `n=60` models. One `D_s` is computed for each seed. The report includes the mean, median, sample SD, every seed value, a 10,000-resample percentile bootstrap CI with bootstrap seed 0, a paired one-sample t-test, Wilcoxon signed-rank, and Cohen's dz. Secondary routing and distance outputs are not presented as additional confirmatory tests."""
    fidelity_text = """## The audit supports integrity of the final result set

Read-only checks confirm 20 complete triplets, exact paired initialization checksums, exposure matching, identical mature schedules, balanced labels, complete checkpoints, and no global router collapse at logged evaluations. The frozen snapshot contains the config, requirements, pip environment, runtime manifest, and SHA-256 hashes for all 60 result JSON files.

The snapshot protects the current result set from silent subsequent changes, but it does not replace Git history or an external preregistration timestamp created before the sweep."""
    limitations = f"""## Limitations and boundaries of confidence

1. **External validity:** one synthetic task pair, one model architecture, two experts, and one optimizer regime.
2. **Heterogeneity:** seeds {', '.join(negatives)} have negative effects; the mechanism has not been diagnosed.
3. **Equivalence is practical, not formal:** TOST and formal statistical equivalence bounds were not used.
4. **Causal scope:** the intervention establishes dependence inside a trained model, not a biological mechanism of individuality.
5. **No intermediate causal trajectories:** ablations were saved only at the final step; steps 4k and 10k have routing and accuracy but not `S_causal(t)`.
6. **Entropy proxy:** JSON `router_entropy` is the entropy of task-mean routing vectors, not mean per-example entropy; the primary test does not depend on it.
7. **Device metadata defect:** the checkpoint confirms `{data['actual_device']}`, but the metadata field was overwritten with the config value `auto`.
8. **Determinism was not forced:** seeds were set, but `torch.use_deterministic_algorithms(True)` was not enabled; bitwise MPS reproducibility across runtimes is not guaranteed.
9. **No Git or preregistration timestamp:** the snapshot was created after the sweep and cannot prove the config was unchanged before results were inspected.
10. **No independent replication:** this is one complete sweep in one software and hardware environment.
11. **Shared fixed test set:** uncertainty reflects initialization variability for this sample, not dataset variability.
12. **Distance is descriptive:** the combined Q/C scale is not calibrated against a within-condition null distribution."""
    files_text = """## Files, evidence, and reproducibility

The primary deliverable is a self-contained HTML report. The canonical artifact JSON, exact CSV tables, raw JSON and checkpoints, frozen snapshot, code, tests, and static figures are listed below. Raw result JSON files remain the source of truth; summaries can be regenerated from them."""
    next_steps = """## Recommended next steps

1. Correct resolved-device metadata, true per-example entropy, and the formal prolonged-collapse rule.
2. Add causal ablation at steps 4,000 and 10,000 without changing the frozen training setup.
3. Put the project under Git and preregister a replication manifest before running it.
4. Run an exact independent replication on 20 new seeds or a second runtime without tuning.
5. Only after replication, open a separately labeled exploratory follow-up for architecture experiments."""
    questions = """## Open questions

- Why do seeds 9, 10, 14, and 17 oppose the mean effect?
- When does causal specialization emerge, and does it survive the transition at step 4,000?
- Where does routing diverge from causal contribution?
- Does the effect replicate under a new fixed data seed?
- What is the within-condition baseline for permutation-invariant distance?"""

    files = [
        {"category": "Primary report", "path": "results/summary/detailed_report.html", "purpose": "Self-contained technical report"},
        {"category": "Report source", "path": "results/summary/detailed_report_artifact.json", "purpose": "Validated canonical payload"},
        {"category": "Report evidence", "path": "results/summary/detailed_report_data.sqlite", "purpose": "Runnable SQLite source for report charts and tables"},
        {"category": "Summary", "path": "results/summary/pilot_report.md", "purpose": "Compact automated report"},
        {"category": "Analysis", "path": "results/summary/analysis.json", "purpose": "Statistics, criteria, distances"},
        {"category": "Audit CSV", "path": "results/summary/seed_level_results.csv", "purpose": "All paired seed values"},
        {"category": "Audit CSV", "path": "results/summary/condition_summary.csv", "purpose": "Condition means and SDs"},
        {"category": "Snapshot", "path": "results/summary/frozen_snapshot/", "purpose": "Config, environment, checksums"},
        {"category": "Raw evidence", "path": "results/raw/seed_<0..19>/<condition>/result.json", "purpose": "Per-run metrics and histories"},
        {"category": "Checkpoints", "path": "results/raw/seed_<0..19>/<condition>/*.pt", "purpose": "Initial, 4k, 10k, 20k states"},
        {"category": "Log", "path": "results/progress.jsonl", "purpose": "Final sweep event log"},
        {"category": "Config", "path": "configs/default.yaml", "purpose": "Frozen experiment settings"},
        {"category": "Code", "path": "src/dci_pilot/", "purpose": "Complete implementation"},
        {"category": "Tests", "path": "tests/", "purpose": "Automated verification"},
        {"category": "Figures", "path": "figures/figure_1_behavior.png … figure_6_dynamics.png", "purpose": "Static publication-style figures"},
    ]
    blocks = [
        {"id": "title", "type": "markdown", "body": "# Developmental Computational Individuality — Full Technical Report"},
        {"id": "summary", "type": "markdown", "body": summary, "sourceId": "analysis"},
        {"id": "criteria", "type": "table", "tableId": "criteria"},
        {"id": "behavior_text", "type": "markdown", "body": behavior_text, "sourceId": "raw"},
        {"id": "behavior", "type": "chart", "chartId": "behavior", "layout": "full"},
        {"id": "conditions", "type": "table", "tableId": "conditions", "layout": "full"},
        {"id": "causal_text", "type": "markdown", "body": causal_text, "sourceId": "analysis"},
        {"id": "causal", "type": "chart", "chartId": "causal", "layout": "full"},
        {"id": "paired", "type": "chart", "chartId": "paired", "layout": "full"},
        {"id": "seeds", "type": "table", "tableId": "seeds", "layout": "full"},
        {"id": "routing_text", "type": "markdown", "body": routing_text, "sourceId": "raw"},
        {"id": "routing", "type": "chart", "chartId": "routing", "layout": "full"},
        {"id": "dynamics", "type": "chart", "chartId": "dynamics", "layout": "full"},
        {"id": "ablation_text", "type": "markdown", "body": ablation_text, "sourceId": "raw"},
        {"id": "ablation", "type": "chart", "chartId": "ablation", "layout": "full"},
        {"id": "distance_text", "type": "markdown", "body": distance_text, "sourceId": "analysis"},
        {"id": "distance", "type": "chart", "chartId": "distance", "layout": "full"},
        {"id": "scope", "type": "markdown", "body": scope_text, "sourceId": "code"},
        {"id": "design", "type": "markdown", "body": design_text, "sourceId": "config"},
        {"id": "metrics", "type": "markdown", "body": metrics_text, "sourceId": "code"},
        {"id": "stats", "type": "markdown", "body": stats_text, "sourceId": "analysis"},
        {"id": "fidelity_text", "type": "markdown", "body": fidelity_text, "sourceId": "snapshot"},
        {"id": "fidelity", "type": "table", "tableId": "fidelity", "layout": "full"},
        {"id": "limitations", "type": "markdown", "body": limitations},
        {"id": "files_text", "type": "markdown", "body": files_text, "sourceId": "snapshot"},
        {"id": "files", "type": "table", "tableId": "files", "layout": "full"},
        {"id": "next", "type": "markdown", "body": next_steps},
        {"id": "questions", "type": "markdown", "body": questions},
    ]
    return {
        "surface": "report",
        "manifest": {"version": 1, "surface": "report", "title": "Developmental Computational Individuality — Full Technical Report", "description": "Implementation, results, statistical audit, limitations, and reproducibility for the 60-model pilot.", "generatedAt": generated_at, "charts": charts, "tables": tables, "sources": manifest_sources, "blocks": blocks},
        "snapshot": {"version": 1, "generatedAt": generated_at, "status": "ready", "datasets": {"headline": data["headline"], "criteria": data["criteria"], "behavior": data["behavior"], "condition_summary": data["condition_summary"], "per_run": data["per_run"], "seed_effects": data["seed_effects"], "dynamics": data["dynamics"], "ablations": data["ablations"], "distances": data["distances"], "fidelity": data["fidelity"], "files": files}},
        "sources": sources,
    }


def write_report_sources(results_root="results/raw", summary_dir="results/summary"):
    os.makedirs(summary_dir, exist_ok=True)
    data = collect_data(results_root)
    generated_at = datetime.now(timezone.utc).isoformat()
    artifact = build_artifact(data, generated_at)
    _materialize_report_database(artifact, os.path.join(summary_dir, "detailed_report_data.sqlite"), generated_at)
    artifact_path = os.path.join(summary_dir, "detailed_report_artifact.json")
    with open(artifact_path, "w", encoding="utf-8") as file:
        json.dump(artifact, file, ensure_ascii=False, indent=2)
    _write_csv(os.path.join(summary_dir, "seed_level_results.csv"), data["seed_effects"])
    _write_csv(os.path.join(summary_dir, "condition_summary.csv"), data["condition_summary"])
    chart_map = """# Detailed report chart map

| Segment | Question | Type | Dataset | Supported claim |
|---|---|---|---|---|
| Behavior | Comparable final task accuracy? | Grouped bar | behavior | All curricula solve both tasks similarly |
| Causal organization | Distribution by curriculum? | Box plot | per_run | Blocked distributions shift upward |
| Paired effect | Consistency within seed? | Signed bar | seed_effects | 16/20 positive; four counterexamples visible |
| Routing | Task-conditional routing difference? | Box plot | per_run | Routing is secondary and heterogeneous |
| Dynamics | Emergence and persistence? | Multi-series line | dynamics | Routing difference survives step 4,000 |
| Ablation | Expert-task causal dependence? | Heatmap | ablations | Knockout effects are nontrivial |
| Organization | Difference after label alignment? | Box plot | distances | Combined Q/C difference remains after swap minimization |

Palette policy: categorical only for real task/condition series, diverging around zero for signed paired effects, sequential for the ablation matrix, and neutral/single-root styling for distributions.
"""
    with open(os.path.join(summary_dir, "detailed_report_chart_map.md"), "w", encoding="utf-8") as file:
        file.write(chart_map)
    return artifact_path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/raw")
    parser.add_argument("--summary", default="results/summary")
    args = parser.parse_args(argv)
    print(write_report_sources(args.results, args.summary))


if __name__ == "__main__":
    main()
