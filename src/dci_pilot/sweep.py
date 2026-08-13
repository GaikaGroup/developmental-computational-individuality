from __future__ import annotations
import argparse, json, os, traceback, yaml
from .train import run
from . import CONDITIONS

def parse_seeds(value):
    if ":" in value:
        start, end = map(int, value.split(":", 1)); return range(start, end + 1)
    return [int(x) for x in value.split(",")]

def main(argv=None):
    p = argparse.ArgumentParser(); p.add_argument("--seeds", default="0:19"); p.add_argument("--config", default="configs/default.yaml"); p.add_argument("--output", default="results"); args = p.parse_args()
    with open(args.config, encoding="utf-8") as f: config = yaml.safe_load(f)
    seeds = list(parse_seeds(args.seeds)); total_runs = len(seeds) * len(CONDITIONS)
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "progress.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps({"status": "sweep_started", "total_runs": total_runs, "seeds": seeds, "conditions": CONDITIONS}) + "\n")
    print(f"ЭТАП 1/2 — обучение: {total_runs} моделей | seeds={len(seeds)} | conditions={len(CONDITIONS)}", flush=True)
    run_index = 0
    for seed in seeds:
        for condition in CONDITIONS:
            run_index += 1
            print(f"\n--- запуск {run_index}/{total_runs}: seed={seed}, condition={condition} ---", flush=True)
            try:
                run(condition, seed, config, args.output, run_index, total_runs)
            except Exception as exc:
                os.makedirs(args.output, exist_ok=True)
                event = {"seed": seed, "condition": condition, "status": "failed", "error": repr(exc), "traceback": traceback.format_exc()}
                with open(os.path.join(args.output, "progress.jsonl"), "a", encoding="utf-8") as f: f.write(json.dumps(event) + "\n")
                raise
    print(f"ЭТАП 2/2 — анализ и отчёт: {total_runs}/{total_runs} моделей завершено", flush=True)

if __name__ == "__main__": main()
