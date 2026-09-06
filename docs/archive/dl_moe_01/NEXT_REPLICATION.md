# DL-MoE-01-R2 prospective replication planning draft

Status: planning draft, not registered, not authorized for scientific launch.
Prepared after observing the complete first series. No R2 training outcomes
have been generated as part of this archival task. This document must not be
cited as an already completed preregistration.

## Prior knowledge

The first series has n=30, mean A=−0.0120438891, 95% bootstrap CI
[−0.0205617259, −0.0032389058], one-sided positive p=0.993725449 and eight
passed behavioral TOST comparisons. The first series and historical pilot
will not be combined with the R2 primary sample. Repeating the study will not
retroactively preregister the first series.

## Proposed question and fixed design

Perform a direct replication of the original positive amplification prediction,
H2: E[A]>0, using 30 new paired blocks, six models per block, the same eight-
expert Flat-MoE and two-by-four DL-MoE, the same two tasks, 4,000 early steps
and 16,000 common steps, optimizer, precision, interventions and final endpoint.
Do not replace the original direction with a negative hypothesis after seeing
the first result. Report the signed estimate and 95% interval irrespective of
whether the positive hypothesis is supported. A separate test of the newly
suggested negative direction would require an explicitly different study plan.

Primary A, S and D formulas, α=.05 one-sided t test, 10,000 block bootstrap
resamples, two-sided 100,000 sign flips and H1/H4/H5 Holm family remain those
of the original frozen implementation. Analysis random seed is 0. H3 remains
all eight paired TOST comparisons within ±.02 and all 12 mean accuracy floors
above .95. Support requires the original positive mean, p<.05, positive lower
bootstrap bound and behavioral gate. Failed gate means no comparable-behavior
support claim. Intermediate checkpoints and diagnostics remain secondary.

No optional stopping or outcome exclusions. Infrastructure interruptions resume
the same seed. Numerical failures remain flagged and un-replaced. Analyze only
complete finite primary blocks; fewer than 28 prohibits the confirmatory
conclusion. Report all missing blocks and failures. The fixed 30-block size is
for direct comparability, not a new claim of adequate power for all small effects.

## Seed identity and anti-reuse rule

Proposed root: `DL-MOE-01-REPLICATION-R2-v1`; fields and hash mapping match the
original protocol: SHA256 of UTF-8 `{root}|{block:03d}|{field}`, first eight
bytes unsigned big-endian, modulo (2^31−2), plus 1. Use primary blocks 001–030;
reserve rows 031–040 are documented but never substituted automatically.
Before registration verify no equality with first-series model/evaluation seeds
and no overlap in actual per-task batch-seed ranges. Publish the resulting
manifest and its hash; do not regenerate it opportunistically after outcomes.

## Required before registration and launch

The original runner hardcodes the first-series seed generator and rejects a
new manifest. Therefore this draft is NOT executable merely by changing a CLI
flag. A separate R2 runner/namespace must first be implemented and tested while
preserving the original source. Validate manifest propagation into execution,
checkpoints, evaluation and analysis. Use tiny, explicitly technical fixtures
for testing; do not inspect R2 scientific outcomes before registration.

After technical review, freeze the exact R2 code commit, configuration, seed
manifest, computational environment, deviations from the original, and this
analysis plan. Attach all of them to an OSF registration (public or embargoed)
and confirm the completed server-issued registration and timestamp. Save that
identifier and its relationship to the first-series release. Only then may a
separately authorized full run start. Any later amendments must be disclosed.

No registration ID, timestamp, public archive DOI or R2 implementation commit
is invented in this planning draft.
