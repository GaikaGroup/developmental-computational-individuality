# DL-MoE Mechanism-01 — Communication Pathway Ablation

Status: EXPLORATORY PROTOCOL / NOT CONFIRMATORY / NOT PREREGISTERED

## Scientific question

DL-MoE-01 did not support the predicted positive Architecture × Development amplification. The next question is narrower: does the explicit cross-population communication pathway contribute causally to developmental-history-dependent functional organization within the hierarchical DL-MoE implementation?

This experiment is mechanism isolation, not a rescue test of the rejected DL-MoE-01 positive-amplification hypothesis.

## Fixed implementation basis

Base implementation: `codex/dl-moe-v1` at commit `8f307e310cfafb377e86ab55da44e6793bf158c2`.

The implementation already exposes directional communication interventions (`disable_comm_0_to_1`, `disable_comm_1_to_0`) and a configuration-level communication switch/scale. No architecture redesign is required for the first diagnostic.

## Experimental contrast

Use the same trained hierarchical model state and evaluation examples for paired interventions:

1. intact communication;
2. disable population 0 -> population 1 communication;
3. disable population 1 -> population 0 communication;
4. disable both directions simultaneously.

The primary diagnostic quantity is the paired change in the existing causal-specialization endpoint under each intervention relative to intact evaluation. Behavioral performance must be reported alongside causal organization to detect destructive interventions.

## Developmental conditions

Evaluate the communication interventions separately for the existing developmental schedules used by the implementation (BLOCKED_AB, BLOCKED_BA, INTERLEAVED where available). Do not pool schedule identities before inspecting direction and symmetry.

## Symmetry handling

Population labels are exchangeable. Direction-specific effects must therefore be interpreted only after label alignment or through label-invariant summaries. A raw difference between population 0 and population 1 is not evidence of a biologically or computationally privileged population identity.

## Interpretation contract

- A routing shift alone is not causal specialization.
- A performance difference alone is not computational individuality.
- A communication-ablation effect on the causal endpoint is evidence that the communication pathway participates in the measured organization, not that communication alone created it.
- If disabling communication destroys task performance, the specialization endpoint is not interpretable as a selective mechanism effect without a behavioral-equivalence qualification.
- Null or opposite-direction effects are retained and reported.

## First executable diagnostic

The first run should be deliberately small: one existing trained model/checkpoint per available developmental schedule, evaluated under the four paired communication conditions above. It is a smoke/diagnostic run used to validate intervention semantics, endpoint extraction, label handling and artifact schema before any multi-seed mechanism study.

## Required artifact schema

For every evaluated checkpoint record:

- source commit and checkpoint identity/hash;
- developmental schedule and seed;
- intervention condition;
- causal-specialization endpoint components and aggregate;
- task-performance metrics used by the existing evaluation contract;
- routing diagnostics, explicitly marked diagnostic-only;
- population-label alignment/invariant summary;
- software commit and config hash.

## Gate to a larger study

A multi-seed mechanism experiment may be designed only after the diagnostic confirms that (a) communication interventions execute as intended, (b) endpoint extraction is stable, (c) checkpoint provenance is complete, and (d) behavioral-equivalence interpretation is defined. Any later confirmatory mechanism claim requires a separately frozen protocol, seeds, analysis plan and prospective registration/authorization.
