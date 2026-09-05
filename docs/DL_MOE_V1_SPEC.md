# DL-MoE v1.0 specification

## Scope

`dci_pilot` is a frozen historical scientific artifact. `dl_moe` is a new,
independent experimental platform. Historical results are never pooled with
DL-MoE runs.

The primary scientific question is whether architecture and developmental
history interact to shape mature causal organization:

\[
O = F(A, D, S), \qquad A \times D \rightarrow O.
\]

The architecture provides conditions for specialization; it never assigns a
task to a population or expert in advance.

## Forward computation

For input `x`, a shared encoder produces `h ∈ R^model_dim`. A population
router produces `r_pop ∈ simplex(2)`. Each population `p` has an independent
expert router `r_exp[p] ∈ simplex(K)` and experts `E[p,e]` with identical
architecture. The population aggregate is

\[
u_p = \sum_e r_{p,e}^{exp} E_{p,e}(h).
\]

When communication is enabled, each `u_p` is projected down to `comm_dim`,
sent to the other population, projected back, gated, and added residually:

\[
v_0 = LN(u_0 + g_{10} C_{up,0}(C_{down,1}(u_1))),
\quad
v_1 = LN(u_1 + g_{01} C_{up,1}(C_{down,0}(u_0))).
\]

The final mixture is `z = r_pop[0] v_0 + r_pop[1] v_1`, followed by a residual
connection from `h`, normalization, and an output head. Communication output
never bypasses a population into the classifier.

Defaults are two populations, four experts per population, model dimension
128, expert hidden dimension 256, `comm_dim=16`, soft routing, shared encoder,
and no task-specialization loss. `comm_dim=0` disables communication.

## Routing and interventions

Routing supports `soft`, `top1`, and `top2`. Top-k modes use differentiable
masked softmax weights and are exploratory in v1.0. `Intervention` is
functional and supports expert, population, and directional communication
knockouts, plus routing overrides. Evaluation never mutates model weights.

## Development

The schedule interface exposes `task_at(step)` and `tasks`. v1.0 implements
`interleaved`, `blocked_ab`, and `blocked_ba`, with an early manipulation phase
followed by a common balanced phase. Schedule generation is deterministic and
paired conditions have identical task counts.

## Causal endpoints

The original two-expert/two-task metric is preserved exactly as
`s_causal_original = 1/2 Σ_e |Δ_e,A - Δ_e,B|`. It is not generalized by
overloading its name. The generalized metric is `causal_specialization_general`,
defined as the mean per-expert mean absolute deviation of task-wise causal
effects from that expert's task mean.

Population specialization uses the same two-population/two-task form. General
organizational comparisons use permutation-aligned causal profiles, with
optional within-population alignment and population swap symmetry.

## Primary and secondary endpoints

The confirmatory candidate primary endpoint is final-checkpoint renormalized
causal knockout. For paired seeds, `D_s` is blocked mean specialization minus
interleaved specialization. Architecture amplification is
`A_s = D_s(dl_moe) - D_s(flat_moe)` and must be tested as an interaction, not
inferred from an absolute DL-MoE score.

Behavioral accuracy and loss are reported separately and must satisfy the
configured equivalence bound before organizational claims are interpreted.
Routing and communication diagnostics are secondary observations, never causal
evidence by themselves.

## Reproducibility contract

Each run records the full config, config hash, Git commit, four seed roles,
device/runtime metadata, parameter count, checkpoints, routing diagnostics,
causal profiles, and intervention settings. Development/tuning seeds and
confirmatory seeds are separate configuration concerns. No confirmatory grid
is launched by the smoke command.
