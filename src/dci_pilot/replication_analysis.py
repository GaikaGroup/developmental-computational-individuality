from __future__ import annotations
import argparse, csv, html, json, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from . import CONDITIONS
from .statistics import paired_statistics, summary

LABELS = {"INTERLEAVED": "Interleaved", "BLOCKED_AB": "Blocked A→B", "BLOCKED_BA": "Blocked B→A"}
COLORS = {"INTERLEAVED": "#3274A1", "BLOCKED_AB": "#E1812C", "BLOCKED_BA": "#3A923A"}

def _load_results(root):
    rows = []
    for seed_dir in sorted(os.scandir(root), key=lambda x: x.name):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"): continue
        for condition in CONDITIONS:
            path = os.path.join(seed_dir.path, condition, "result.json")
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f: rows.append(json.load(f))
    return rows

def _write_csv(path, rows):
    if not rows: return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

def _causal(row, mode):
    key = "knockout_renormalized" if mode == "renormalized" else "knockout_non_renormalized"
    return row.get(key, {}).get("causal_specialization", row["causal_specialization"] if mode == "renormalized" else None)

def _ablation(row, mode):
    key = "knockout_renormalized" if mode == "renormalized" else "knockout_non_renormalized"
    return row.get(key, {}).get("ablation", row["ablation"] if mode == "renormalized" else None)

def _paired(rows, mode):
    grouped = defaultdict(dict)
    for r in rows: grouped[r["seed"]][r["condition"]] = r
    out = []
    for seed in sorted(grouped):
        triplet = grouped[seed]
        if set(triplet) != set(CONDITIONS): continue
        i = _causal(triplet["INTERLEAVED"], mode)
        ab = _causal(triplet["BLOCKED_AB"], mode); ba = _causal(triplet["BLOCKED_BA"], mode)
        out.append((seed, i, (ab + ba) / 2, (ab + ba) / 2 - i))
    return out

def _condition_summary(rows):
    result = []
    for condition in CONDITIONS:
        rs = [r for r in rows if r["condition"] == condition]
        def add(name, values, row):
            stats = summary(values); row[f"{name}_mean"] = stats["mean"]; row[f"{name}_sd"] = stats["sd"]
        row = {"condition": condition, "n": len(rs)}
        add("accuracy_A", [r["accuracy_A"] for r in rs], row); add("accuracy_B", [r["accuracy_B"] for r in rs], row)
        add("balanced_accuracy", [(r["accuracy_A"] + r["accuracy_B"]) / 2 for r in rs], row)
        add("routing_specialization", [r["routing_specialization"] for r in rs], row)
        add("router_entropy", [r["router_entropy"] for r in rs], row)
        add("causal_specialization_renormalized", [_causal(r, "renormalized") for r in rs], row)
        add("causal_specialization_non_renormalized", [_causal(r, "non_renormalized") for r in rs], row)
        row["router_collapse_count"] = sum(bool(r["router_collapse"]["collapsed"]) for r in rs)
        row["router_collapse_rate"] = row["router_collapse_count"] / len(rs) if rs else None
        result.append(row)
    return result

def _seed_rows(rows):
    grouped = defaultdict(dict)
    for r in rows: grouped[r["seed"]][r["condition"]] = r
    output = []
    for seed in sorted(grouped):
        if set(grouped[seed]) != set(CONDITIONS): continue
        v = grouped[seed]; row = {"seed": seed}
        for condition in CONDITIONS:
            r = v[condition]; prefix = condition.lower()
            row.update({f"{prefix}_accuracy_A": r["accuracy_A"], f"{prefix}_accuracy_B": r["accuracy_B"], f"{prefix}_balanced_accuracy": (r["accuracy_A"] + r["accuracy_B"]) / 2, f"{prefix}_routing_specialization": r["routing_specialization"], f"{prefix}_causal_renormalized": _causal(r, "renormalized"), f"{prefix}_causal_non_renormalized": _causal(r, "non_renormalized")})
        for mode in ("renormalized", "non_renormalized"):
            row[f"blocked_mean_causal_{mode}"] = (row[f"blocked_ab_causal_{mode}"] + row[f"blocked_ba_causal_{mode}"]) / 2
            row[f"D_s_{mode}"] = row[f"blocked_mean_causal_{mode}"] - row[f"interleaved_causal_{mode}"]
        row["blocked_mean_balanced_accuracy"] = (row["blocked_ab_balanced_accuracy"] + row["blocked_ba_balanced_accuracy"]) / 2
        row["blocked_minus_interleaved_balanced_accuracy"] = row["blocked_mean_balanced_accuracy"] - row["interleaved_balanced_accuracy"]
        output.append(row)
    return output

def _integrity(rows, expected_seeds):
    grouped = defaultdict(dict)
    for r in rows: grouped[r["seed"]][r["condition"]] = r
    complete = sorted(seed for seed, values in grouped.items() if set(values) == set(CONDITIONS))
    checksums = all(len({grouped[s][c]["initial_checksum"] for c in CONDITIONS}) == 1 for s in complete)
    fields = ("git_commit", "config_hash", "source_hash", "test_data_seed")
    consistent = {field: sorted({str(r["metadata"].get(field)) for r in rows}) for field in fields}
    checks = {
        "expected_60_runs": len(rows) == 60,
        "expected_20_complete_seed_triplets": complete == list(expected_seeds),
        "identical_initial_weights_within_triplet": checksums and len(complete) == len(expected_seeds),
        "resolved_device_recorded": all(r["metadata"].get("device") not in (None, "auto") for r in rows),
        "deterministic_algorithms_enabled": all(r["metadata"].get("determinism", {}).get("enabled") is True for r in rows),
        "single_git_commit": len(consistent["git_commit"]) == 1 and consistent["git_commit"] != ["unknown"],
        "single_config_hash": len(consistent["config_hash"]) == 1,
        "single_source_hash": len(consistent["source_hash"]) == 1,
        "single_fixed_test_seed": len(consistent["test_data_seed"]) == 1,
        "both_knockouts_present": all("knockout_renormalized" in r and "knockout_non_renormalized" in r for r in rows),
    }
    return {"pass": all(checks.values()), "checks": checks, "values": consistent, "complete_seeds": complete}

def _pilot_comparison(pilot_rows, replication_rows):
    output = {}
    for name, rows in (("Pilot V1", pilot_rows), ("Replication", replication_rows)):
        paired = _paired(rows, "renormalized"); stats = paired_statistics([x[3] for x in paired])
        output[name] = {"n_paired_seeds": len(paired), "mean_D_s": stats["mean_difference"], "ci_low": stats["ci_low"], "ci_high": stats["ci_high"], "cohens_dz": stats["cohens_dz"], "mean_balanced_accuracy": float(np.mean([(r["accuracy_A"] + r["accuracy_B"]) / 2 for r in rows]))}
    return output

def _style(ax):
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=.2)

def _figures(rows, seed_rows, checkpoint_records, pilot_comparison, figure_dir):
    os.makedirs(figure_dir, exist_ok=True); rng = np.random.default_rng(4)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(3); width = .34
    for j, task in enumerate(("A", "B")):
        values = [[r[f"accuracy_{task}"] for r in rows if r["condition"] == c] for c in CONDITIONS]
        means = [np.mean(v) for v in values]; ax.bar(x + (j-.5)*width, means, width, label=f"Task {task}", alpha=.75)
        for k, v in enumerate(values): ax.scatter(np.full(len(v), x[k] + (j-.5)*width) + rng.normal(0, .025, len(v)), v, s=12, color="black", alpha=.5)
    ax.set_xticks(x, [LABELS[c] for c in CONDITIONS]); ax.set_ylim(.94, 1); ax.set_ylabel("Held-out accuracy"); ax.legend(frameon=False); _style(ax); fig.tight_layout(); fig.savefig(os.path.join(figure_dir, "replication_behavior.png"), dpi=220); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, mode, title in zip(axes, ("renormalized", "non_renormalized"), ("Renormalized knockout (primary)", "Non-renormalized knockout (secondary)")):
        vals = [[_causal(r, mode) for r in rows if r["condition"] == c] for c in CONDITIONS]
        ax.boxplot(vals, tick_labels=[LABELS[c] for c in CONDITIONS], showfliers=False)
        for k, v in enumerate(vals, 1): ax.scatter(np.full(len(v), k)+rng.normal(0,.045,len(v)), v, s=18, color=COLORS[CONDITIONS[k-1]], alpha=.7)
        ax.set_title(title); ax.tick_params(axis="x", rotation=15); _style(ax)
    axes[0].set_ylabel("Causal specialization (S_causal)"); fig.tight_layout(); fig.savefig(os.path.join(figure_dir, "replication_causal_specialization.png"), dpi=220); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, mode, title in zip(axes, ("renormalized", "non_renormalized"), ("Primary", "Secondary")):
        for row in seed_rows: ax.plot([0,1], [row[f"interleaved_causal_{mode}"], row[f"blocked_mean_causal_{mode}"]], marker="o", color="#555", alpha=.55)
        ax.set_xticks([0,1], ["Interleaved", "Mean blocked"]); ax.set_title(f"{title}: {mode.replace('_',' ')}"); _style(ax)
    axes[0].set_ylabel("Causal specialization (S_causal)"); fig.tight_layout(); fig.savefig(os.path.join(figure_dir, "replication_paired_effect.png"), dpi=220); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, mode, title in zip(axes, ("renormalized", "non_renormalized"), ("Renormalized (primary at step 20,000)", "Non-renormalized (secondary)")):
        for condition in CONDITIONS:
            pts=[]
            for step in (4000,10000,20000):
                values=[r[f"knockout_{mode}"]["causal_specialization"] for r in checkpoint_records if r["condition"]==condition and r["step"]==step]
                pts.append(np.mean(values))
            ax.plot((4000,10000,20000), pts, marker="o", label=LABELS[condition], color=COLORS[condition])
        ax.axvline(4000, color="#222", linestyle="--", alpha=.6); ax.set_title(title); ax.set_xlabel("Optimizer step"); _style(ax)
    axes[0].set_ylabel("Mean S_causal"); axes[1].legend(frameon=False); fig.tight_layout(); fig.savefig(os.path.join(figure_dir, "replication_causal_dynamics.png"), dpi=220); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,5)); names=list(pilot_comparison); means=[pilot_comparison[n]["mean_D_s"] for n in names]
    lows=[means[i]-pilot_comparison[n]["ci_low"] for i,n in enumerate(names)]; highs=[pilot_comparison[n]["ci_high"]-means[i] for i,n in enumerate(names)]
    ax.bar(names, means, color=["#7A7A7A", "#3274A1"], alpha=.8); ax.errorbar(names, means, yerr=[lows, highs], fmt="none", color="black", capsize=5); ax.axhline(0,color="#222",linewidth=1); ax.set_ylabel("Mean paired D_s (95% bootstrap CI)"); _style(ax); fig.tight_layout(); fig.savefig(os.path.join(figure_dir, "pilot_vs_replication.png"), dpi=220); plt.close(fig)

def _md_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] + ["---:" for _ in headers[1:]]) + "|"] + ["| " + " | ".join(map(str,row)) + " |" for row in rows])

def _report(analysis, condition_rows, seed_rows):
    p = analysis["primary"]; s = analysis["secondary_non_renormalized"]; b = analysis["behavioral_equivalence"]
    cond_table = _md_table(["Condition", "Acc A", "Acc B", "Balanced", "S_route", "S_causal renorm", "S_causal non-renorm", "Collapse"], [[LABELS[r["condition"]], f'{r["accuracy_A_mean"]:.4f}', f'{r["accuracy_B_mean"]:.4f}', f'{r["balanced_accuracy_mean"]:.4f}', f'{r["routing_specialization_mean"]:.4f}', f'{r["causal_specialization_renormalized_mean"]:.4f}', f'{r["causal_specialization_non_renormalized_mean"]:.4f}', f'{r["router_collapse_count"]}/{r["n"]}'] for r in condition_rows])
    seed_table = _md_table(["Seed", "Interleaved", "Mean blocked", "D_s", "D_s non-renorm"], [[r["seed"], f'{r["interleaved_causal_renormalized"]:.5f}', f'{r["blocked_mean_causal_renormalized"]:.5f}', f'{r["D_s_renormalized"]:+.5f}', f'{r["D_s_non_renormalized"]:+.5f}'] for r in seed_rows])
    return f"""# Exact Independent Replication Report

## Technical summary

**Classification: {analysis['classification']}.** The preregistered primary endpoint used the final step-20,000 renormalized knockout across {p['n']} paired initialization seeds. Mean `D_s` was {p['mean_difference']:.5f} (median {p['median_difference']:.5f}, SD {p['sd_difference']:.5f}), with a 95% paired bootstrap CI [{p['ci_low']:.5f}, {p['ci_high']:.5f}]. The paired t-test gave p={p['paired_t_p']:.6g}, Wilcoxon p={p['wilcoxon_p']:.6g}, and Cohen's dz={p['cohens_dz']:.3f}. Positive/negative/zero seeds: {p['positive_seeds']}/{p['negative_seeds']}/{p['zero_seeds']}.

This result is evidence only for this frozen tiny-MoE configuration. It does not prove the general Developmental Computational Individuality Hypothesis.

## Behavioral equivalence

The preregistered behavioral criterion {'passed' if b['pass'] else 'did not pass'}. Mean blocked-minus-interleaved differences were {b['blocked_minus_interleaved_accuracy_A']:+.5f} for Task A and {b['blocked_minus_interleaved_accuracy_B']:+.5f} for Task B; the frozen tolerance was ±0.02, with every condition required to average at least 0.95 on both tasks.

{cond_table}

![Behavioral accuracy](../../figures/replication_behavior.png)

## Primary renormalized knockout result

For knockout of expert e, its contribution is removed and the surviving expert receives weight 1. `Delta_e,T = Accuracy_T(normal) - Accuracy_T(-e)` and `S_causal = 0.5 * sum_e |Delta_e,A - Delta_e,B|`. The paired endpoint is the mean of the two blocked conditions minus Interleaved within each initialization seed.

![Causal specialization](../../figures/replication_causal_specialization.png)

![Paired developmental effect](../../figures/replication_paired_effect.png)

{seed_table}

## Secondary non-renormalized knockout

The secondary intervention zeroes the removed expert while retaining the survivor's original routing probability. Its independent paired result was mean `D_s={s['mean_difference']:.5f}`, 95% bootstrap CI [{s['ci_low']:.5f}, {s['ci_high']:.5f}], t-test p={s['paired_t_p']:.6g}, Wilcoxon p={s['wilcoxon_p']:.6g}, and dz={s['cohens_dz']:.3f}. It does not replace the preregistered primary metric.

## Causal dynamics

Steps 4,000 and 10,000 are preregistered secondary dynamics analyses in the replication; only the renormalized step-20,000 value is the primary endpoint.

![Causal dynamics](../../figures/replication_causal_dynamics.png)

## Pilot V1 versus replication

![Pilot and replication comparison](../../figures/pilot_vs_replication.png)

Pilot V1 was analyzed from its immutable frozen results. Its 4,000/10,000 checkpoint causal analyses are labeled **POST-HOC EXPLORATORY ANALYSIS** and are not part of Pilot V1's original confirmatory test.

## Reproducibility and diagnostics

All integrity checks {'passed' if analysis['integrity']['pass'] else 'did not pass'}. Each run records the resolved device, Python, PyTorch, OS, Git commit, config hash, source hash, deterministic-algorithm setting, true mean per-example router entropy, and formal prolonged-collapse outcome. Unsupported nondeterministic backend operations recorded: {analysis['nondeterministic_backend_operations']}.

## Interpretation boundary

The architecture, tasks, optimizer, schedule, curriculum, loss coefficient, duration, data-generating process, primary metric, and decision rule were frozen in Git before replication. No DL-MoE, multimodal, hemispheric, bottleneck, or architecture experiment was performed.
"""

def analyze(replication_root="results/replication", pilot_root="results/pilot_v1_frozen/raw", figure_dir="figures"):
    rows = _load_results(os.path.join(replication_root, "raw")); pilot_rows = _load_results(pilot_root)
    with open("replication_config.yaml", encoding="utf-8") as f:
        import yaml; config = yaml.safe_load(f)
    expected_seeds = config["seeds"]; integrity = _integrity(rows, expected_seeds)
    if len(rows) != 60: raise ValueError(f"replication incomplete: expected 60 results, found {len(rows)}")
    seed_rows = _seed_rows(rows); condition_rows = _condition_summary(rows)
    primary = paired_statistics([r["D_s_renormalized"] for r in seed_rows], config["bootstrap_samples"], config["bootstrap_seed"])
    secondary = paired_statistics([r["D_s_non_renormalized"] for r in seed_rows], config["bootstrap_samples"], config["bootstrap_seed"])
    by_condition = {r["condition"]: r for r in condition_rows}
    blocked_A = np.mean([by_condition[c]["accuracy_A_mean"] for c in ("BLOCKED_AB", "BLOCKED_BA")]); blocked_B = np.mean([by_condition[c]["accuracy_B_mean"] for c in ("BLOCKED_AB", "BLOCKED_BA")])
    behavior = {"pass": bool(all(by_condition[c][f"accuracy_{t}_mean"] >= .95 for c in CONDITIONS for t in ("A","B")) and abs(blocked_A-by_condition["INTERLEAVED"]["accuracy_A_mean"]) <= .02 and abs(blocked_B-by_condition["INTERLEAVED"]["accuracy_B_mean"]) <= .02), "blocked_minus_interleaved_accuracy_A": float(blocked_A-by_condition["INTERLEAVED"]["accuracy_A_mean"]), "blocked_minus_interleaved_accuracy_B": float(blocked_B-by_condition["INTERLEAVED"]["accuracy_B_mean"]), "tolerance": .02}
    max_deltas = [max(abs(v) for e in _ablation(r,"renormalized").values() for v in e.values()) for r in rows]
    nontrivial = {"pass": float(np.mean(max_deltas)) > .05, "mean_per_run_max_absolute_delta": float(np.mean(max_deltas)), "threshold": .05}
    if not integrity["pass"]: classification = "INCONCLUSIVE"
    elif behavior["pass"] and nontrivial["pass"] and primary["ci_low"] > 0: classification = "REPLICATED IN THIS CONFIGURATION"
    elif behavior["pass"] and primary["mean_difference"] > 0 and (primary["paired_t_p"] < .05 or primary["wilcoxon_p"] < .05): classification = "PARTIALLY REPLICATED"
    else: classification = "NOT REPLICATED"
    checkpoint_path = os.path.join(replication_root, "checkpoint_analysis", "analysis.json")
    with open(checkpoint_path, encoding="utf-8") as f: checkpoint_records = json.load(f)["records"]
    pilot_comparison = _pilot_comparison(pilot_rows, rows)
    unsupported = sorted({op for r in rows for op in r["metadata"].get("determinism", {}).get("unsupported_operations", [])})
    analysis = {"classification": classification, "n_models": len(rows), "n_paired_seeds": len(seed_rows), "primary_definition": "mean(S_causal_BLOCKED_AB, S_causal_BLOCKED_BA) - S_causal_INTERLEAVED at step 20000 using renormalized knockout", "primary": primary, "secondary_non_renormalized": secondary, "behavioral_equivalence": behavior, "nontrivial_causal_effect": nontrivial, "condition_summary": condition_rows, "pilot_vs_replication": pilot_comparison, "integrity": integrity, "nondeterministic_backend_operations": unsupported, "router_collapse_total": sum(r["router_collapse_count"] for r in condition_rows)}
    _figures(rows, seed_rows, checkpoint_records, pilot_comparison, figure_dir)
    _write_csv(os.path.join(replication_root, "seed_level_results.csv"), seed_rows); _write_csv(os.path.join(replication_root, "condition_summary.csv"), condition_rows)
    with open(os.path.join(replication_root, "analysis.json"), "w", encoding="utf-8") as f: json.dump(analysis, f, indent=2)
    report = _report(analysis, condition_rows, seed_rows)
    with open(os.path.join(replication_root, "replication_report.md"), "w", encoding="utf-8") as f: f.write(report)
    body = html.escape(report)
    for name in ("replication_behavior.png", "replication_causal_specialization.png", "replication_paired_effect.png", "replication_causal_dynamics.png", "pilot_vs_replication.png"):
        body = body.replace(html.escape(f"![{ {'replication_behavior.png':'Behavioral accuracy','replication_causal_specialization.png':'Causal specialization','replication_paired_effect.png':'Paired developmental effect','replication_causal_dynamics.png':'Causal dynamics','pilot_vs_replication.png':'Pilot and replication comparison'}[name] }](../../figures/{name})"), f'<img src="../../figures/{name}" alt="{name}" loading="lazy">')
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Exact Independent Replication Report</title><style>body{{font:16px/1.55 system-ui;max-width:1040px;margin:auto;padding:32px;color:#17212b}}pre{{white-space:pre-wrap;font:inherit}}img{{display:block;max-width:100%;margin:24px auto;border:1px solid #ddd}}@media(prefers-color-scheme:dark){{body{{background:#101418;color:#e8edf2}}img{{background:white}}}}</style></head><body><pre>{body}</pre></body></html>'''
    with open(os.path.join(replication_root, "replication_report.html"), "w", encoding="utf-8") as f: f.write(document)
    return analysis

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--replication", default="results/replication"); p.add_argument("--pilot", default="results/pilot_v1_frozen/raw"); p.add_argument("--figures", default="figures"); a=p.parse_args(); print(json.dumps(analyze(a.replication,a.pilot,a.figures), indent=2))

if __name__ == "__main__": main()
