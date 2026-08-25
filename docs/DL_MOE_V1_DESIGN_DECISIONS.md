# DL-MoE v1.0 design decisions

## Decision: separate package

**Alternatives:** mutate `dci_pilot`; create a shared mutable core; create a
new package. **Decision:** new `dl_moe` package. **Rationale:** preserves
scientific provenance and makes baseline compatibility explicit. **Risk:** a
small amount of adapter code is duplicated. **Test:** historical imports and
frozen artifact paths remain untouched.

## Decision: two symmetric populations

**Rationale:** provides a minimal population-level intervention while keeping
the symmetry group explicit. P0/P1 have the same module types and dimensions;
labels have no biological interpretation.

## Decision: shared encoder

**Rationale:** prevents specialization from being trivially assigned by
population-specific input extractors. Population-specific encoders remain a
future ablation.

## Decision: soft routing default

**Rationale:** maximizes continuity with the proof-of-concept and avoids adding
discrete routing noise to the first architecture test. `top1` and `top2` remain
configurable exploratory factors.

## Decision: `comm_dim=16`

**Rationale:** a deliberately narrow channel relative to model dimension 128.
It is configurable and can be ablated at 0, 8, 16, and 128.

## Decision: no specialization loss

**Rationale:** a task-to-population assignment loss would encode the desired
result. Load balancing, when enabled, is task-agnostic and separately logged.

## Decision: functional interventions

**Rationale:** knockout must not mutate weights or leave evaluation state behind.
The model receives an immutable intervention object at forward time.

## Decision: contemporary flat baseline

**Rationale:** historical DCI results are not a valid architecture control for
an 8-expert system. A new flat 8-expert baseline must run under the same new
protocol and seeds.
