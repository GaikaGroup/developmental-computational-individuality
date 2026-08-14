# Detailed report chart map

| Segment | Question | Type | Dataset | Supported claim |
|---|---|---|---|---|
| Behavior | Comparable final task accuracy? | Grouped bar | behavior | All curricula solve both tasks similarly |
| Causal organization | Distribution by curriculum? | Box plot | per_run | Blocked distributions shift upward |
| Paired effect | Consistency within seed? | Signed bar | seed_effects | 16/20 positive; four counterexamples visible |
| Routing | Task-conditional routing difference? | Box plot | per_run | Routing is secondary and heterogeneous |
| Dynamics | Emergence and persistence? | Multi-series line | dynamics | Routing difference survives step 4,000 |
| Ablation | Expert-task causal dependence? | Heatmap | ablations | Knockout effects are nontrivial |
| Organization | Difference after label alignment? | Box plot | distances | Combined Q/C difference remains after swap minimization |

Palette policy: categorical only for real task/condition series, diverging around zero for signed paired effects, sequential for the ablation matrix, and neutral/single-root styling for distributions.
