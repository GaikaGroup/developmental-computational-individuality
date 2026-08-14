from __future__ import annotations
import hashlib, json, os, platform, random, subprocess, sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch

def _json_default(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray): return value.tolist()
    if hasattr(value, "item"): return value.item()
    return str(value)

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
def configure_determinism():
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return {"enabled": torch.are_deterministic_algorithms_enabled(), "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(), "unsupported_operations": []}
def choose_device(value):
    if value != "auto": return value
    if torch.cuda.is_available(): return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    return "cpu"
def state_checksum(state):
    h = hashlib.sha256()
    for key in sorted(state): h.update(key.encode()); h.update(state[key].detach().cpu().numpy().tobytes())
    return h.hexdigest()
def git_commit():
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return "unknown"
def value_hash(value): return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default).encode()).hexdigest()
def source_hash():
    h = hashlib.sha256(); root = Path(__file__).resolve().parent
    for path in sorted(root.glob("*.py")):
        h.update(path.name.encode()); h.update(path.read_bytes())
    return h.hexdigest()
def json_dump(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f: json.dump(data, f, indent=2, default=_json_default)
def metadata(seed, data_seed, condition, device, model, cfg):
    frozen = dict(cfg); configured_device = frozen.pop("device", None)
    config_hash = frozen.pop("experiment_config_hash", value_hash(cfg))
    return {**frozen, "model_seed": seed, "data_seed": data_seed, "condition": condition, "git_commit": git_commit(), "config_hash": config_hash, "source_hash": source_hash(), "PyTorch_version": torch.__version__, "Python_version": sys.version, "OS": platform.platform(), "configured_device": configured_device, "device": str(next(model.parameters()).device), "parameter_count": sum(p.numel() for p in model.parameters()), "timestamp": datetime.now(timezone.utc).isoformat()}
