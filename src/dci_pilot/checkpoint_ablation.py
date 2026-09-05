from __future__ import annotations
import argparse, csv, glob, json, os, time
import torch, yaml
from . import CONDITIONS
from .evaluate import causal_evaluation
from .model import TinySoftMoE
from .utils import choose_device, configure_determinism, json_dump

STEPS = (4000, 10000, 20000)

def run(checkpoint_root="results/pilot_v1_frozen/raw", output="results/pilot_v1_posthoc", config_path="results/pilot_v1_frozen/default.yaml", device_value="auto", context="pilot_posthoc"):
    with open(config_path, encoding="utf-8") as f: config = yaml.safe_load(f)
    device = choose_device(device_value); configure_determinism(); test_seed = config.get("test_data_seed", config["data_seed"] + 9_000_000)
    paths = sorted(glob.glob(os.path.join(checkpoint_root, "seed_*", "*", "step_*.pt")))
    paths = [p for p in paths if int(os.path.basename(p)[5:-3]) in STEPS]
    seeds = {int(os.path.basename(os.path.dirname(os.path.dirname(p))).split("_")[-1]) for p in paths}
    expected = len(seeds) * len(CONDITIONS) * len(STEPS)
    if not seeds or len(paths) != expected: raise ValueError(f"expected {expected} complete checkpoints, found {len(paths)}")
    os.makedirs(output, exist_ok=True); records = []; started = time.time()
    progress_path = os.path.join(output, "progress.jsonl")
    with open(progress_path, "w", encoding="utf-8") as f: f.write(json.dumps({"status": "started", "total_checkpoints": len(paths)}) + "\n")
    for index, path in enumerate(paths, 1):
        seed = int(os.path.basename(os.path.dirname(os.path.dirname(path))).split("_")[-1])
        condition = os.path.basename(os.path.dirname(path)); step = int(os.path.basename(path)[5:-3])
        checkpoint = torch.load(path, map_location=device, weights_only=False); model = TinySoftMoE().to(device); model.load_state_dict(checkpoint["state_dict"])
        metrics = causal_evaluation(model, device, config["test_examples"], test_seed)
        if context == "pilot_posthoc":
            label = "PILOT V1 CONFIRMATORY ENDPOINT (RENORMALIZED ONLY)" if step == 20000 else "POST-HOC EXPLORATORY ANALYSIS"
            secondary_label = "POST-HOC SECONDARY ANALYSIS"
        else:
            label = "PREREGISTERED PRIMARY ENDPOINT (RENORMALIZED)" if step == 20000 else "PREREGISTERED SECONDARY DYNAMICS ANALYSIS"
            secondary_label = "PREREGISTERED SECONDARY ANALYSIS"
        record = {"seed": seed, "condition": condition, "step": step, "analysis_label": label, "non_renormalized_label": secondary_label, **metrics}
        records.append(record)
        event = {"status": "evaluated", "checkpoint_index": index, "total_checkpoints": len(paths), "seed": seed, "condition": condition, "step": step, "elapsed_seconds": round(time.time() - started, 1)}
        with open(progress_path, "a", encoding="utf-8") as f: f.write(json.dumps(event) + "\n")
        print(f"checkpoint {index}/{len(paths)} | seed={seed} {condition} step={step}", flush=True)
    json_dump({"scope": "Pilot V1 frozen checkpoints only", "confirmatory_endpoint_step": 20000, "exploratory_steps": [4000, 10000], "records": records}, os.path.join(output, "analysis.json"))
    fields = ["seed", "condition", "step", "analysis_label", "non_renormalized_label", "accuracy_A", "accuracy_B", "routing_specialization", "router_entropy", "causal_specialization_renormalized", "causal_specialization_non_renormalized"]
    with open(os.path.join(output, "checkpoint_results.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for r in records:
            writer.writerow({**{k: r[k] for k in fields[:9]}, "causal_specialization_renormalized": r["knockout_renormalized"]["causal_specialization"], "causal_specialization_non_renormalized": r["knockout_non_renormalized"]["causal_specialization"]})
    with open(progress_path, "a", encoding="utf-8") as f: f.write(json.dumps({"status": "completed", "total_checkpoints": len(paths), "elapsed_seconds": round(time.time() - started, 1)}) + "\n")
    return records

def main(argv=None):
    p = argparse.ArgumentParser(); p.add_argument("--checkpoints", default="results/pilot_v1_frozen/raw"); p.add_argument("--output", default="results/pilot_v1_posthoc"); p.add_argument("--config", default="results/pilot_v1_frozen/default.yaml"); p.add_argument("--device", default="auto"); p.add_argument("--context", choices=("pilot_posthoc", "replication"), default="pilot_posthoc"); a = p.parse_args()
    run(a.checkpoints, a.output, a.config, a.device, a.context)

if __name__ == "__main__": main()
