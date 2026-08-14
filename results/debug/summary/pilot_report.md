# DCI pilot report

Valid run files: **3**

## Performance

| condition | A mean (sd) | B mean (sd) | balanced mean (sd) |
|---|---:|---:|---:|
| INTERLEAVED | 0.484 (0.000) | 0.453 (0.000) | 0.469 (0.000) |
| BLOCKED_AB | 0.492 (0.000) | 0.461 (0.000) | 0.477 (0.000) |
| BLOCKED_BA | 0.438 (0.000) | 0.477 (0.000) | 0.457 (0.000) |

## Organization

| condition | routing specialization mean | causal specialization mean | router entropy mean | collapse rate |
|---|---:|---:|---:|---:|
| INTERLEAVED | 0.0018 | 0.0156 | 0.6925 | 0.000 |
| BLOCKED_AB | 0.0016 | 0.0078 | 0.6925 | 0.000 |
| BLOCKED_BA | 0.0019 | 0.0117 | 0.6925 | 0.000 |

## Primary paired analysis

Blocked minus interleaved causal specialization: mean=-0.00586, 95% bootstrap CI=(-0.00586, -0.00586), Cohen's dz=0.000, paired t p=nan, Wilcoxon p=1.0.

## Interpretation

This automated report is descriptive and does not prove the general hypothesis.
Classification: **INCONCLUSIVE** or not supported by the current result files.
