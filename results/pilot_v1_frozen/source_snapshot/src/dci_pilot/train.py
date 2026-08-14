from __future__ import annotations
import argparse, copy, json, os, time, yaml, torch
from .curriculum import build_curriculum
from .data import BatchStream
from .evaluate import causal_evaluation, evaluate
from .model import TinySoftMoE
from .utils import choose_device, json_dump, metadata, seed_everything, state_checksum

def run(condition, seed, config, output_root="results", run_index=1, total_runs=1):
    device = choose_device(config.get("device", "auto")); seed_everything(seed)
    model = TinySoftMoE().to(device); initial = copy.deepcopy(model.state_dict()); checksum = state_checksum(initial)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    curriculum = build_curriculum(condition, config["early_steps"], config["steps"], config["data_seed"] + seed)
    stream = BatchStream(config["data_seed"], config["batch_size"], device)
    run_dir = os.path.join(output_root, "raw", f"seed_{seed}", condition); os.makedirs(run_dir, exist_ok=True)
    progress_path = os.path.join(output_root, "progress.jsonl"); started = time.time()
    def progress(step, status):
        elapsed = time.time() - started
        event = {"run_index": run_index, "total_runs": total_runs, "seed": seed, "condition": condition, "step": step, "total_steps": config["steps"], "status": status, "elapsed_seconds": round(elapsed, 1)}
        with open(progress_path, "a", encoding="utf-8") as f: f.write(json.dumps(event) + "\n")
        fraction = ((run_index - 1) + step / config["steps"]) / total_runs
        width = 28; filled = int(width * fraction)
        bar = "█" * filled + "░" * (width - filled)
        remaining = (elapsed / step * (config["steps"] - step)) if step else 0
        eta = f"{int(remaining // 60)}м {int(remaining % 60):02d}с"
        print(f"ЭТАП 1/2 — обучение {bar} {fraction * 100:5.1f}% {run_index}/{total_runs} | seed={seed} {condition} | step {step}/{config['steps']} | ETA текущего run {eta} | {status}", flush=True)
    torch.save({"state_dict": initial, "checksum": checksum}, os.path.join(run_dir, "initial.pt"))
    checkpoints, eval_steps, history = set(config["checkpoints"]), set(config["eval_steps"]), []
    def lr_at(step):
        if step <= config["warmup_steps"]: return config["learning_rate"] * step / max(1, config["warmup_steps"])
        progress = (step - config["warmup_steps"]) / max(1, config["steps"] - config["warmup_steps"])
        return config["learning_rate"] * .5 * (1 + torch.cos(torch.tensor(progress * torch.pi)).item())
    frozen = dict(config, optimizer="AdamW", training_steps=config["steps"], curriculum_definition=condition)
    if 0 in eval_steps:
        metrics = evaluate(model, device, min(config["test_examples"], 4096), config["data_seed"] + 9_000_000)
        metrics.update(step=0, training_loss=None, mean_router=None); history.append(metrics)
        progress(0, "started")
    for step, task in enumerate(curriculum.tasks, 1):
        model.train(); x, y = stream.next(task); optimizer.zero_grad(set_to_none=True)
        logits, p = model(x, return_router=True)
        task_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        balance_loss = ((p.mean(0) - .5) ** 2).sum(); loss = task_loss + config["balance_lambda"] * balance_loss
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
        for group in optimizer.param_groups: group["lr"] = lr_at(step)
        optimizer.step()
        if step % 500 == 0: progress(step, "training")
        if step in eval_steps:
            metrics = evaluate(model, device, min(config["test_examples"], 4096), config["data_seed"] + 9_000_000)
            metrics.update(step=step, training_loss=float(loss.detach()), mean_router=p.mean(0).detach().cpu().tolist()); history.append(metrics)
        if step in checkpoints:
            torch.save({"state_dict": model.state_dict(), "metadata": frozen, "initial_checksum": checksum}, os.path.join(run_dir, f"step_{step}.pt"))
            progress(step, "checkpoint")
    final = causal_evaluation(model, device, config["test_examples"], config["data_seed"] + 9_000_000)
    result = {"seed": seed, "condition": condition, "initial_checksum": checksum, **final, "history": history, "metadata": metadata(seed, config["data_seed"], condition, device, model, frozen)}
    json_dump(result, os.path.join(run_dir, "result.json")); progress(config["steps"], "completed"); return result

def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--condition", required=True, choices=("INTERLEAVED", "BLOCKED_AB", "BLOCKED_BA")); parser.add_argument("--seed", type=int, default=0); parser.add_argument("--config", default="configs/default.yaml"); parser.add_argument("--output", default="results")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f: config = yaml.safe_load(f)
    run(args.condition, args.seed, config, args.output)

if __name__ == "__main__": main()
