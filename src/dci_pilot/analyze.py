from __future__ import annotations
import argparse, glob, hashlib, json, os, platform, shutil, subprocess, sys
import numpy as np
import torch
from .plots import make_figures
from .statistics import paired_statistics, summary
from .metrics import perm_matrix_distance

def load_rows(root):
    rows = []
    for path in glob.glob(os.path.join(root, "seed_*", "*", "result.json")):
        with open(path, encoding="utf-8") as f: rows.append(json.load(f))
    return rows

def _organization_matrix(row):
    return torch.tensor([
        [row["routing_A"][0], row["routing_B"][0], row["ablation"]["expert_1"]["delta_A"], row["ablation"]["expert_1"]["delta_B"]],
        [row["routing_A"][1], row["routing_B"][1], row["ablation"]["expert_2"]["delta_A"], row["ablation"]["expert_2"]["delta_B"]],
    ])

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()

def _snapshot(root, summary_dir, rows):
    snapshot_dir = os.path.join(summary_dir, "frozen_snapshot"); os.makedirs(snapshot_dir, exist_ok=True)
    shutil.copyfile("configs/default.yaml", os.path.join(snapshot_dir, "default.yaml"))
    shutil.copyfile("requirements.txt", os.path.join(snapshot_dir, "requirements.txt"))
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    with open(os.path.join(snapshot_dir, "pip_freeze.txt"), "w", encoding="utf-8") as f: f.write(freeze)
    paths = sorted(glob.glob(os.path.join(root, "seed_*", "*", "result.json")))
    checksums = {os.path.relpath(p, root): _sha256(p) for p in paths}
    with open(os.path.join(snapshot_dir, "result_checksums.json"), "w", encoding="utf-8") as f: json.dump(checksums, f, indent=2)
    manifest = {"valid_runs": len(rows), "result_files": len(paths), "python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__, "git_commit": "unavailable: repository is not under git", "config_sha256": _sha256("configs/default.yaml"), "requirements_sha256": _sha256("requirements.txt"), "result_checksums_file": "result_checksums.json"}
    with open(os.path.join(snapshot_dir, "manifest.json"), "w", encoding="utf-8") as f: json.dump(manifest, f, indent=2)
    return manifest

def analyze(root="results/raw", summary_dir="results/summary", figure_dir="figures"):
    rows = load_rows(root); os.makedirs(summary_dir, exist_ok=True)
    if not rows: raise ValueError(f"no result.json files under {root}")
    make_figures(rows, figure_dir)
    by_seed = {}
    for r in rows: by_seed.setdefault(r["seed"], {})[r["condition"]] = r
    complete_seeds = {s: v for s, v in by_seed.items() if set(v) == {"INTERLEAVED", "BLOCKED_AB", "BLOCKED_BA"}}
    lines = ["# DCI pilot report", "", f"Valid run files: **{len(rows)}**; complete paired seeds: **{len(complete_seeds)}**", "", "## Performance", "", "| condition | A mean (sd) | B mean (sd) | balanced mean (sd) |", "|---|---:|---:|---:|"]
    for c in ("INTERLEAVED", "BLOCKED_AB", "BLOCKED_BA"):
        rs = [r for r in rows if r["condition"] == c]; a = summary([r["accuracy_A"] for r in rs]); b = summary([r["accuracy_B"] for r in rs]); bal = summary([(r["accuracy_A"] + r["accuracy_B"]) / 2 for r in rs]); lines.append(f"| {c} | {a['mean']:.3f} ({a['sd']:.3f}) | {b['mean']:.3f} ({b['sd']:.3f}) | {bal['mean']:.3f} ({bal['sd']:.3f}) |")
    diffs = [((v["BLOCKED_AB"]["causal_specialization"] + v["BLOCKED_BA"]["causal_specialization"]) / 2 - v["INTERLEAVED"]["causal_specialization"]) for v in complete_seeds.values()]
    stats = paired_statistics(diffs) if diffs else {}
    lines += ["", "## Organization", "", "| condition | routing specialization mean | causal specialization mean | router entropy mean | collapse rate |", "|---|---:|---:|---:|---:|"]
    for c in ("INTERLEAVED", "BLOCKED_AB", "BLOCKED_BA"):
        rs = [r for r in rows if r["condition"] == c]; lines.append(f"| {c} | {np.mean([r['routing_specialization'] for r in rs]):.4f} | {np.mean([r['causal_specialization'] for r in rs]):.4f} | {np.mean([r['router_entropy'] for r in rs]):.4f} | {np.mean([max(r['routing_A'] + r['routing_B']) > .95 for r in rs]):.3f} |")
    lines += ["", "## Primary paired analysis", "", f"Blocked minus interleaved causal specialization: mean={stats.get('mean_difference', float('nan')):.5f}, 95% bootstrap CI=({stats.get('ci_low', float('nan')):.5f}, {stats.get('ci_high', float('nan')):.5f}), Cohen's dz={stats.get('cohens_dz', float('nan')):.3f}, paired t p={stats.get('paired_t_p')}, Wilcoxon p={stats.get('wilcoxon_p')}."]
    behavior = {}
    for c in ("INTERLEAVED", "BLOCKED_AB", "BLOCKED_BA"):
        rs = [r for r in rows if r["condition"] == c]
        behavior[c] = {"pass_rate_95_each_task": float(np.mean([(r["accuracy_A"] >= .95 and r["accuracy_B"] >= .95) for r in rs])), "balanced_mean": float(np.mean([(r["accuracy_A"] + r["accuracy_B"]) / 2 for r in rs]))}
    blocked_bal = np.mean([behavior["BLOCKED_AB"]["balanced_mean"], behavior["BLOCKED_BA"]["balanced_mean"]]); behavior_gap = abs(blocked_bal - behavior["INTERLEAVED"]["balanced_mean"])
    causal_abs = [max(abs(r["ablation"][f"expert_{e}"][f"delta_{t}"]) for e in (1, 2) for t in ("A", "B")) for r in rows]
    criteria = {"A_behavioral_equivalence": {"pass": bool(min(v["pass_rate_95_each_task"] for v in behavior.values()) >= .5 and behavior_gap <= .02), "details": f"all-condition >=95% rate min={min(v['pass_rate_95_each_task'] for v in behavior.values()):.3f}; blocked/interleaved balanced-accuracy gap={behavior_gap:.4f}"}, "B_persistent_organizational_difference": {"pass": bool(stats) and bool(stats["ci_low"] > 0), "details": f"paired mean={stats.get('mean_difference')}; CI=({stats.get('ci_low')}, {stats.get('ci_high')})"}, "C_nontrivial_causal_effect": {"pass": bool(float(np.mean(causal_abs)) > .05), "details": f"mean per-run maximum absolute knockout delta={float(np.mean(causal_abs)):.4f}; operational threshold=0.05"}}
    lines += ["", "## Success criteria", "", "| criterion | status | evidence |", "|---|---|---|"]
    for name, value in criteria.items(): lines.append(f"| {name} | {'PASS' if value['pass'] else 'FAIL'} | {value['details']} |")
    lines += ["", "## Seed-level paired results", "", "| seed | interleaved causal | blocked mean causal | difference | interleaved balanced acc. | blocked balanced acc. |", "|---:|---:|---:|---:|---:|---:|"]
    for seed in sorted(complete_seeds):
        v = complete_seeds[seed]; blocked = (v["BLOCKED_AB"]["causal_specialization"] + v["BLOCKED_BA"]["causal_specialization"]) / 2; ia = (v["INTERLEAVED"]["accuracy_A"] + v["INTERLEAVED"]["accuracy_B"]) / 2; ba = np.mean([(v[c]["accuracy_A"] + v[c]["accuracy_B"]) / 2 for c in ("BLOCKED_AB", "BLOCKED_BA")]); lines.append(f"| {seed} | {v['INTERLEAVED']['causal_specialization']:.4f} | {blocked:.4f} | {blocked-v['INTERLEAVED']['causal_specialization']:.4f} | {ia:.4f} | {ba:.4f} |")
    distances = []
    for seed, v in complete_seeds.items(): distances.append({"seed": seed, "interleaved_blocked_ab": perm_matrix_distance(_organization_matrix(v["INTERLEAVED"]), _organization_matrix(v["BLOCKED_AB"])), "interleaved_blocked_ba": perm_matrix_distance(_organization_matrix(v["INTERLEAVED"]), _organization_matrix(v["BLOCKED_BA"]))})
    lines += ["", "## Permutation-invariant organizational audit", "", "The organization matrix has expert rows and columns `[routing_A, routing_B, delta_A, delta_B]`; distances minimize over identity and expert swap.", "", f"Mean distance INTERLEAVED→BLOCKED_AB: **{np.mean([d['interleaved_blocked_ab'] for d in distances]):.4f}**; INTERLEAVED→BLOCKED_BA: **{np.mean([d['interleaved_blocked_ba'] for d in distances]):.4f}**.", "", "## Interpretation", "", "This automated report is descriptive and does not prove the general hypothesis."]
    if all(v["pass"] for v in criteria.values()): lines.append("Classification: **SUPPORTED IN THIS PILOT**.")
    elif criteria["B_persistent_organizational_difference"]["pass"]: lines.append("Classification: **PARTIALLY SUPPORTED**.")
    else: lines.append("Classification: **INCONCLUSIVE** or not supported by the current result files.")
    with open(os.path.join(summary_dir, "pilot_report.md"), "w", encoding="utf-8") as f: f.write("\n".join(lines) + "\n")
    manifest = _snapshot(root, summary_dir, rows)
    with open(os.path.join(summary_dir, "analysis.json"), "w", encoding="utf-8") as f: json.dump({"n": len(rows), "complete_paired_seeds": len(complete_seeds), "paired": stats, "criteria": criteria, "behavior": behavior, "permutation_distances": distances, "snapshot": manifest}, f, indent=2)

def main(argv=None):
    p = argparse.ArgumentParser(); p.add_argument("--results", default="results/raw"); p.add_argument("--summary", default="results/summary"); p.add_argument("--figures", default="figures"); a = p.parse_args(); analyze(a.results, a.summary, a.figures)
if __name__ == "__main__": main()
