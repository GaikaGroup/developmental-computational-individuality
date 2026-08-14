from __future__ import annotations
import hashlib, json, os, random, subprocess, sys
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
def json_dump(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f: json.dump(data, f, indent=2, default=_json_default)
def metadata(seed, data_seed, condition, device, model, cfg):
    return {"model_seed": seed, "data_seed": data_seed, "condition": condition, "git_commit": git_commit(), "PyTorch_version": torch.__version__, "Python_version": sys.version, "device": device, "parameter_count": sum(p.numel() for p in model.parameters()), **cfg, "timestamp": datetime.now(timezone.utc).isoformat()}
