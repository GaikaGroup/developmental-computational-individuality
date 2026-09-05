from __future__ import annotations

import numpy as np
from scipy.stats import t


def _vector(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('expected a nonempty finite vector')
    return values


def one_sided_t(values: np.ndarray) -> dict:
    values = _vector(values)
    if len(values) < 2:
        raise ValueError('t-test requires at least two seed blocks')
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    if sd == 0:
        statistic = None  # Infinite/undefined statistic is represented explicitly in JSON.
        p = 0.0 if mean > 0 else 1.0
    else:
        statistic = mean / (sd / np.sqrt(len(values)))
        p = float(t.sf(statistic, len(values) - 1))
    return {'statistic': statistic, 'p_value': p, 'alternative': 'greater'}


def holm(p_values):
    values = np.asarray(p_values, dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('invalid p-values')
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    return adjusted.tolist()


def sign_flip(values: np.ndarray, seed: int, samples: int = 100000) -> float:
    """Deterministic two-sided robustness test, preserving the candidate implementation."""
    values = _vector(values)
    rng = np.random.default_rng(seed)
    observed = abs(values.mean())
    count = 0
    for start in range(0, samples, 1000):
        signs = rng.choice(np.array([-1.0, 1.0]), size=(min(1000, samples-start), len(values)))
        count += int((np.abs((signs * values).mean(axis=1)) >= observed).sum())
    return float(count / samples)


def paired_tost(differences: np.ndarray, margin: float = 0.02) -> dict:
    differences = _vector(differences)
    mean = float(differences.mean())
    n = len(differences)
    sd = float(differences.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n)
    if n < 2:
        lo = hi = None
        p_lower = p_upper = 1.0
    elif sd == 0:
        lo = hi = mean
        p_lower = 0.0 if mean > -margin else 1.0
        p_upper = 0.0 if mean < margin else 1.0
    else:
        critical = float(t.ppf(.95, n-1))
        lo, hi = mean-critical*se, mean+critical*se
        p_lower = float(t.sf((mean+margin)/se, n-1))
        p_upper = float(t.cdf((mean-margin)/se, n-1))
    return {'mean_difference': mean, 'ci90_lower': lo, 'ci90_upper': hi,
            'tost_p_lower': p_lower, 'tost_p_upper': p_upper,
            'tost_p_max': max(p_lower, p_upper),
            'equivalent': bool(n > 1 and p_lower < .05 and p_upper < .05)}
