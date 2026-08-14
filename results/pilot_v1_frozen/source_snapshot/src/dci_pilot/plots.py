from __future__ import annotations
import os
import matplotlib.pyplot as plt
import numpy as np

def _group(rows, key):
    return {c: [r[key] for r in rows if r["condition"] == c] for c in ("INTERLEAVED", "BLOCKED_AB", "BLOCKED_BA")}

def make_figures(rows, out_dir="figures"):
    os.makedirs(out_dir, exist_ok=True); groups = _group(rows, "causal_specialization")
    conditions = list(groups)
    fig, ax = plt.subplots();
    for task, color in (("A", "tab:blue"), ("B", "tab:orange")):
        vals = _group(rows, f"accuracy_{task}")
        ax.boxplot([vals[c] for c in conditions], tick_labels=conditions, positions=np.arange(3) + (.18 if task == "B" else -.18), widths=.3, patch_artist=True, boxprops={"facecolor": color}, manage_ticks=False)
    ax.set_ylabel("accuracy"); ax.set_title("Final behavioral performance"); fig.savefig(os.path.join(out_dir, "figure_1_behavior.png"), dpi=180, bbox_inches="tight"); plt.close(fig)
    for number, key, title, filename in ((2, "routing_specialization", "Routing specialization", "figure_2_routing.png"), (3, "causal_specialization", "Causal specialization", "figure_3_causal.png")):
        fig, ax = plt.subplots(); ax.boxplot([_group(rows, key)[c] for c in conditions], tick_labels=conditions); ax.set_ylabel(key); ax.set_title(title); fig.savefig(os.path.join(out_dir, filename), dpi=180, bbox_inches="tight"); plt.close(fig)
    by_seed = {s: {r["condition"]: r[key] for r in rows if r["seed"] == s} for s in sorted({r["seed"] for r in rows}) for key in ["causal_specialization"]}
    fig, ax = plt.subplots()
    for s, values in by_seed.items():
        if "INTERLEAVED" in values and "BLOCKED_AB" in values and "BLOCKED_BA" in values:
            ax.plot([0, 1], [values["INTERLEAVED"], (values["BLOCKED_AB"] + values["BLOCKED_BA"]) / 2], marker="o", color="0.5")
    ax.set_xticks([0, 1], ["INTERLEAVED", "mean BLOCKED"]); ax.set_ylabel("causal specialization"); ax.set_title("Paired developmental effect"); fig.savefig(os.path.join(out_dir, "figure_4_paired.png"), dpi=180, bbox_inches="tight"); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for ax, c in zip(axes, conditions):
        matrices = [np.array([[r["ablation"]["expert_1"]["delta_A"], r["ablation"]["expert_2"]["delta_A"]], [r["ablation"]["expert_1"]["delta_B"], r["ablation"]["expert_2"]["delta_B"]]]) for r in rows if r["condition"] == c]
        ax.imshow(np.mean(matrices, axis=0), cmap="coolwarm"); ax.set_title(c); ax.set_xticks([0, 1], ["E1", "E2"]); ax.set_yticks([0, 1], ["A", "B"])
    fig.suptitle("Mean causal contribution matrices"); fig.savefig(os.path.join(out_dir, "figure_5_ablation.png"), dpi=180, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots()
    for c in conditions:
        histories = [r["history"] for r in rows if r["condition"] == c]
        steps = [h["step"] for h in histories[0]]; means = [np.mean([h["routing_specialization"] for h in hs]) for hs in zip(*histories)]
        ax.plot(steps, means, label=c)
    ax.axvline(4000, color="black", linestyle="--", label="early boundary"); ax.set_xlabel("step"); ax.set_ylabel("routing specialization"); ax.legend(); fig.savefig(os.path.join(out_dir, "figure_6_dynamics.png"), dpi=180, bbox_inches="tight"); plt.close(fig)
