# DCI pilot report

Valid run files: **60**; complete paired seeds: **20**

## Performance

| condition | A mean (sd) | B mean (sd) | balanced mean (sd) |
|---|---:|---:|---:|
| INTERLEAVED | 0.988 (0.001) | 0.988 (0.001) | 0.988 (0.001) |
| BLOCKED_AB | 0.989 (0.000) | 0.988 (0.001) | 0.989 (0.000) |
| BLOCKED_BA | 0.988 (0.001) | 0.989 (0.001) | 0.988 (0.001) |

## Organization

| condition | routing specialization mean | causal specialization mean | router entropy mean | collapse rate |
|---|---:|---:|---:|---:|
| INTERLEAVED | 0.4459 | 0.2471 | 0.5433 | 0.000 |
| BLOCKED_AB | 0.6786 | 0.4238 | 0.4063 | 0.000 |
| BLOCKED_BA | 0.5986 | 0.3681 | 0.4365 | 0.000 |

## Primary paired analysis

Blocked minus interleaved causal specialization: mean=0.14885, 95% bootstrap CI=(0.06372, 0.22642), Cohen's dz=0.771, paired t p=0.0026911198596382534, Wilcoxon p=0.0031528472900390625.

## Success criteria

| criterion | status | evidence |
|---|---|---|
| A_behavioral_equivalence | PASS | all-condition >=95% rate min=1.000; blocked/interleaved balanced-accuracy gap=0.0005 |
| B_persistent_organizational_difference | PASS | paired mean=0.1488509975373745; CI=(0.06371528539806603, 0.22641508547589181) |
| C_nontrivial_causal_effect | PASS | mean per-run maximum absolute knockout delta=0.4833; operational threshold=0.05 |

## Seed-level paired results

| seed | interleaved causal | blocked mean causal | difference | interleaved balanced acc. | blocked balanced acc. |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.3280 | 0.4705 | 0.1425 | 0.9887 | 0.9886 |
| 1 | 0.0339 | 0.4680 | 0.4342 | 0.9879 | 0.9889 |
| 2 | 0.1575 | 0.3929 | 0.2354 | 0.9871 | 0.9892 |
| 3 | 0.3769 | 0.4821 | 0.1052 | 0.9882 | 0.9886 |
| 4 | 0.2890 | 0.3287 | 0.0397 | 0.9883 | 0.9873 |
| 5 | 0.2995 | 0.4614 | 0.1619 | 0.9873 | 0.9890 |
| 6 | 0.1297 | 0.4718 | 0.3421 | 0.9875 | 0.9883 |
| 7 | 0.3959 | 0.4397 | 0.0438 | 0.9881 | 0.9879 |
| 8 | 0.4739 | 0.4860 | 0.0121 | 0.9888 | 0.9889 |
| 9 | 0.4337 | 0.0977 | -0.3359 | 0.9882 | 0.9884 |
| 10 | 0.3105 | 0.2654 | -0.0451 | 0.9879 | 0.9887 |
| 11 | 0.0391 | 0.4692 | 0.4301 | 0.9881 | 0.9889 |
| 12 | 0.0925 | 0.4737 | 0.3813 | 0.9887 | 0.9893 |
| 13 | 0.1223 | 0.4093 | 0.2869 | 0.9882 | 0.9884 |
| 14 | 0.3585 | 0.3322 | -0.0263 | 0.9882 | 0.9886 |
| 15 | 0.1929 | 0.4552 | 0.2623 | 0.9878 | 0.9885 |
| 16 | 0.0676 | 0.2967 | 0.2291 | 0.9882 | 0.9883 |
| 17 | 0.3803 | 0.2955 | -0.0848 | 0.9878 | 0.9878 |
| 18 | 0.1032 | 0.3411 | 0.2379 | 0.9883 | 0.9882 |
| 19 | 0.3573 | 0.4820 | 0.1247 | 0.9868 | 0.9884 |

## Permutation-invariant organizational audit

The organization matrix has expert rows and columns `[routing_A, routing_B, delta_A, delta_B]`; distances minimize over identity and expert swap.

Mean distance INTERLEAVED→BLOCKED_AB: **0.4788**; INTERLEAVED→BLOCKED_BA: **0.4782**.

## Interpretation

This automated report is descriptive and does not prove the general hypothesis.
Classification: **SUPPORTED IN THIS PILOT**.
