# Preregistered Exact Independent Replication Manifest

Status: **FROZEN BEFORE REPLICATION**. This manifest is to be committed to Git before any seed 100–119 replication model is trained. No listed choice may be changed after replication outcomes are inspected.

## Scope and independence

- Pilot V1 comprises initialization seeds 0–19 and is preserved under `results/pilot_v1_frozen/`; it will not be retrained.
- Replication initialization seeds are exactly 100–119 (20 paired seeds; 60 models).
- Training-data seed is 271828 and held-out test-data seed is 314159. Both differ from Pilot V1 while retaining the identical data-generating process.
- Each seed triplet starts from identical initial weights. The three conditions consume identical per-task example streams and have identical total Task A/Task B exposure. Their mature-phase task schedule is identical.

## Frozen model and tasks

- Architecture (151,427 trainable parameters): input 18 → Linear(18,128) → GELU → Linear(128,128) → GELU; Linear(128,2)+softmax router; two independent Linear(128,256) → GELU → Linear(256,128) experts; probability-weighted soft mixture; residual `h + mixture`; LayerNorm(128); Linear(128,1).
- Task A: draw 16 independent standard-normal features; `y = 1[x0*x1 + x2*x3 > 0]`.
- Task B: draw 16 independent standard-normal features; `y = 1[x8*x9 - x10*x11 > 0]`.
- Task indicator appended to the 16 features: A=`[1,0]`, B=`[0,1]`.

## Frozen training protocol

- Loss: BCE-with-logits plus `lambda_balance * sum_e(mean_batch(p_e)-0.5)^2`; `lambda_balance = 0.01`. No specialization loss.
- Optimizer: AdamW; learning rate 0.0003; weight decay 0.0001; batch size 256; gradient clipping norm 1.0.
- Schedule: 500-step linear warmup followed by cosine decay through step 20,000.
- Duration: exactly 20,000 optimizer steps. Early phase: steps 1–4,000. Common mature phase: steps 4,001–20,000.
- INTERLEAVED: deterministic randomized 50/50 ordering in the early phase (2,000 A and 2,000 B batches).
- BLOCKED_AB: 2,000 A then 2,000 B batches in the early phase.
- BLOCKED_BA: 2,000 B then 2,000 A batches in the early phase.
- Common phase: deterministic randomized 50/50 ordering, shared exactly within every paired triplet.
- Evaluation: fixed 50,000-example held-out dataset per task.

## Frozen measurements

- Accuracy: threshold the single logit at zero. Balanced accuracy reported here means the arithmetic mean of Task A and Task B accuracies.
- Routing specialization: `S_route = 0.5 * sum_e |q_A,e - q_B,e|`, where `q_T,e` is mean routing probability over examples from task T.
- Router entropy: true mean per-example entropy, averaged equally over Task A and Task B evaluation sets.
- Primary causal metric uses the RENORMALIZED knockout: remove expert e and route the surviving expert with weight 1. Define `Delta_e,T = Accuracy_T(normal) - Accuracy_T(-e)` and `S_causal = 0.5 * sum_e |Delta_e,A - Delta_e,B|`.
- Secondary causal metric uses the NON-RENORMALIZED knockout: zero the removed expert contribution while retaining the survivor's original routing probability, without probability renormalization. Its `Delta` and `S_causal` are computed independently.
- Primary paired endpoint: `D_s = mean(S_causal_BLOCKED_AB,s, S_causal_BLOCKED_BA,s) - S_causal_INTERLEAVED,s`, using the renormalized knockout at step 20,000.
- Intermediate `S_causal(4,000)` and `S_causal(10,000)` are secondary dynamics analyses and are not substituted for the final primary endpoint. Pilot V1 intermediate causal analyses are explicitly POST-HOC EXPLORATORY ANALYSIS.

## Frozen criteria and statistics

- Behavioral equivalence: each condition's mean accuracy is at least 0.95 on each task, and the absolute mean blocked-minus-interleaved difference is at most 0.02 separately for Task A and Task B.
- Nontrivial causal effect: mean across runs of the per-run maximum absolute renormalized knockout delta exceeds 0.05.
- Bootstrap: 10,000 paired-seed resamples of `D_s`, sampling seeds with replacement; NumPy generator seed 0; percentile 2.5% and 97.5% limits.
- Paired t-test: two-sided one-sample t-test of paired `D_s` values against zero.
- Wilcoxon signed-rank: two-sided test of paired `D_s` values against zero using SciPy defaults.
- Cohen's dz: mean(`D_s`) / sample SD(`D_s`, ddof=1).
- Prolonged router collapse: at least three consecutive scheduled evaluations for the same expert where its mean routing probability across Tasks A and B is >0.95, with at least 2,000 optimizer steps between the first and last qualifying evaluation. Collapsed runs remain in all analyses.

## Frozen interpretation rule

- **REPLICATED IN THIS CONFIGURATION**: all 20 paired seeds are complete, behavioral equivalence and nontrivial causal-effect criteria pass, and the primary 95% bootstrap CI is strictly above zero.
- **PARTIALLY REPLICATED**: all 20 pairs and behavioral equivalence are present, the primary mean is positive, and at least one preregistered inferential test has p<0.05, but the full replication rule is not met.
- **NOT REPLICATED**: the complete, behaviorally equivalent replication does not meet either rule above.
- **INCONCLUSIVE**: missing paired runs, failed behavioral equivalence, invalid data integrity, or an execution failure prevents a valid test.

No DL-MoE, multimodal, hemispheric, bottleneck, or other architecture experiment is authorized as part of this workflow.
