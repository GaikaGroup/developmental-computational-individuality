from __future__ import annotations

import numpy as np


def one_sided_t(values: np.ndarray) -> dict:
    from scipy.stats import ttest_1samp
    values = np.asarray(values, dtype=float)
    result = ttest_1samp(values, 0.0)
    p = float(result.pvalue / 2 if result.statistic >= 0 else 1 - result.pvalue / 2)
    return {"statistic": float(result.statistic), "p_value": p, "alternative": "greater"}


def sign_flip(values: np.ndarray, seed: int, samples: int = 100000) -> float:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    observed = abs(values.mean())
    signs = rng.choice(np.array([-1.0, 1.0]), size=(samples, len(values)))
    return float((np.abs((signs * values).mean(axis=1)) >= observed).mean())


def paired_tost(differences: np.ndarray, margin: float = 0.02) -> dict:
    from scipy.stats import ttest_1samp
    differences = np.asarray(differences, dtype=float)
    lower = ttest_1samp(differences, -margin)
    upper = ttest_1samp(differences, margin)
    p_lower = float(lower.pvalue / 2 if lower.statistic >= 0 else 1 - lower.pvalue / 2)
    p_upper = float(upper.pvalue / 2 if upper.statistic <= 0 else 1 - upper.pvalue / 2)
    mean = float(differences.mean())
    se = float(differences.std(ddof=1) / np.sqrt(len(differences))) if len(differences) > 1 else 0.0
    from scipy.stats import t
    critical = float(t.ppf(.95, len(differences) - 1)) if len(differences) > 1 else 0.0
    return {"mean_difference": mean, "ci90_lower": mean - critical * se, "ci90_upper": mean + critical * se, "tost_p_lower": p_lower, "tost_p_upper": p_upper, "tost_p_max": max(p_lower, p_upper), "equivalent": bool(p_lower < .05 and p_upper < .05)}
