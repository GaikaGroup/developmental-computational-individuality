from __future__ import annotations
import numpy as np

def summary(values):
    x = np.asarray(values, dtype=float); return {"mean": float(x.mean()), "median": float(np.median(x)), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0}

def paired_statistics(differences, bootstrap_samples=10000, seed=0):
    x = np.asarray(differences, dtype=float); rng = np.random.default_rng(seed)
    means = rng.choice(x, (bootstrap_samples, len(x)), replace=True).mean(1)
    dz = float(x.mean() / x.std(ddof=1)) if len(x) > 1 and x.std(ddof=1) else 0.0
    out = {"mean_difference": float(x.mean()), "ci_low": float(np.quantile(means, .025)), "ci_high": float(np.quantile(means, .975)), "cohens_dz": dz}
    try:
        from scipy.stats import ttest_1samp, wilcoxon
        out["paired_t_p"] = float(ttest_1samp(x, 0).pvalue); out["wilcoxon_p"] = float(wilcoxon(x).pvalue)
    except ImportError: out["paired_t_p"] = out["wilcoxon_p"] = None
    return out

