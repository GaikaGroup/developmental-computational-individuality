# DL-MoE-01 execution and validation

The scientific candidate YAML, Markdown, seed manifest and confirmatory config
remain byte-identical to their recorded hashes. This implementation completes
the runner and analysis; it does not constitute a confirmatory result or a
formal preregistration release. Run commands from a repository checkout.

## Readiness checks

```sh
make ci
```

This runs both packages' tests in separate processes, scientific-artifact
validation, a read-only 180-model dry run, an offline wheel build and Python
bytecode compilation. There is no configured formatter, linter or static type
checker. SciPy and Matplotlib were already in the root environment; they are
now declared by the DL-MoE package because its analysis and plots import them.

`dl_moe/environment.cpu-macos-py314.lock.txt` records the tested local package
versions, including transitive dependencies. It is a CPU/macOS/Python 3.14
version snapshot, not a portable wheel-hash lock. A different compute platform
must establish and record its own environment before scientific execution.
Every execution records the actual installed versions and rejects resume if
its source, config, runtime, Git state or block inventory changes.

## Technical end-to-end validation

Use a new output directory. The technical mode runs two blocks, both
architectures and all three conditions, with a tiny CPU model, 8 training
steps and 40 evaluation examples. These artifacts carry `technical=true` and
cannot enter ordinary confirmatory analysis.

```sh
PYTHONPATH=dl_moe/src .venv/bin/python -m dl_moe.experiments.dl_moe_01.run \
  --technical --output results/validation/dl_moe_01/raw
PYTHONPATH=dl_moe/src .venv/bin/python -m dl_moe.experiments.dl_moe_01.artifacts \
  --raw results/validation/dl_moe_01/raw --freeze \
  --output results/validation/dl_moe_01/tables
PYTHONPATH=dl_moe/src .venv/bin/python -m dl_moe.experiments.dl_moe_01.analyze \
  --results results/validation/dl_moe_01/tables \
  --preregistration dl_moe/experiments/dl_moe_01/preregistration.yaml \
  --technical --output results/validation/dl_moe_01/analysis
```

Tests additionally inject an interruption and compare resumed weights bit for
bit with an uninterrupted run. Other cases cover numerical failures, duplicate
blocks and behavior rows, missing data, negative effects, performance below the
floor, Holm correction, constant paired differences and artifact tampering.

## Scientific execution, after review and tagging

First commit the reviewed implementation, scientific artifacts, tests and
environment snapshot, and create the agreed preregistration tag. Repeat CI
from a clean checkout of that tag. The runner requires a clean Git tree,
including untracked files, and a tag pointing to HEAD. This preparation task
does not create that release or launch its 180 models.

An optional separate freeze record references the final commit without trying
to embed a commit's own hash in that commit:

```sh
PYTHONPATH=dl_moe/src .venv/bin/python -m dl_moe.experiments.dl_moe_01.freeze \
  --output results/dl_moe_01/execution_freeze.json
PYTHONPATH=dl_moe/src .venv/bin/python -m dl_moe.experiments.dl_moe_01.run --dry-run
```

On the reviewed machine, start the complete series explicitly:

```sh
PYTHONPATH=dl_moe/src .venv/bin/python -m dl_moe.experiments.dl_moe_01.run \
  --preregistered --output results/dl_moe_01/confirmatory_v1/raw
```

`auto` selects CUDA when available and CPU otherwise. MPS is deliberately not
an execution backend for this validated path. CUDA requires setting
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before starting Python; deterministic
algorithms are enabled and TF32 is disabled. CUDA execution still requires
validation on the target hardware; the current verification used CPU only.

The series always plans blocks 001–030, six models per block. Reserve blocks
are never substituted automatically. Each architecture/seed block initializes
its three conditions identically. Batch identities follow the frozen
`experience_rows` implementation: the same 2,000 A and 2,000 B batches under
different developmental orders, then a shared alternating A/B common phase.
The generated `common_order_seed` remains recorded; the frozen alternating
common order does not consume it.

## Operational monitoring and recovery

`progress.json` contains the current run, training step and time, without
scientific effect estimates. `run_manifest.csv` is refreshed during execution;
`exclusions_and_incidents.csv` records incidents as they occur. Checkpoints
include model, AdamW state and Torch RNG states. `latest.pt` is written every
500 steps and at 4k/10k/20k; the latter checkpoints are retained separately.
Evaluation is chunked without changing the fixed evaluation sample.

After an infrastructure interruption, rerun the identical command with
`--resume`. An OS lock prevents simultaneous writers and releases on process
exit. Atomic file replacement prevents partially written final artifacts.
Recovery replays from the most recent durable checkpoint using the same batch
identities. No new seed is chosen. A numerical failure is retained as a
terminal outcome, including its earlier valid artifacts, and is not retried
or replaced by `--resume`. Hardware or source changes require explicit review
and a separately documented recovery/deviation; the runner refuses silent
identity changes.

## Raw freeze, extraction and canonical analysis

Wait until all planned runs have a terminal status (completed or numerical
failure). Then hash raw evidence before computing group statistics:

```sh
PYTHONPATH=dl_moe/src .venv/bin/python -m dl_moe.experiments.dl_moe_01.artifacts \
  --raw results/dl_moe_01/confirmatory_v1/raw --freeze \
  --output results/dl_moe_01/confirmatory_v1/tables
PYTHONPATH=dl_moe/src .venv/bin/python -m dl_moe.experiments.dl_moe_01.analyze \
  --results results/dl_moe_01/confirmatory_v1/tables \
  --preregistration dl_moe/experiments/dl_moe_01/preregistration.yaml \
  --output results/dl_moe_01/confirmatory_v1/analysis
```

Raw freeze rejects incomplete inventories and completed runs missing required
checkpoints/evaluations. It records every raw file's SHA-256. The runner
refuses to modify a frozen run directory. Extraction verifies that inventory
and writes derived tables outside it. The analyzer verifies extracted table
hashes before use. Low-level `analyze()` also accepts explicitly supplied
fixture tables for software tests; use the frozen-raw extraction workflow for
canonical scientific analysis. A raw hash manifest is an integrity record,
not filesystem write protection: independent backup is still needed.

Balanced accuracy is the mean of positive- and negative-class recalls within
each task, distinct from ordinary accuracy (both are saved). Knockout deltas
and specialization use balanced accuracy. Renormalized and unrenormalized
profiles are retained at every checkpoint. Schema suffix `_unrenorm` maps to
the preregistration's unrenormalized H5 endpoint.

H2 uses the one-sided t-test and 10,000 seed-block bootstrap samples. The
existing deterministic, two-sided 100,000-draw sign-flip robustness test and
analysis random seed 0 are preserved and explicitly labeled. H1/H4/H5 form
one Holm family. Degenerate zero-variance effect sizes/statistics are reported
as null rather than nonstandard JSON infinity; directional p-values are
handled explicitly. Every finite complete primary block is retained; fewer
than 28 prevents confirmatory analysis. Missing blocks remain listed.

The behavioral gate requires all eight paired TOST comparisons and all twelve
architecture/condition/task mean performance floors. Comparisons use the
complete endpoint blocks; performance floors additionally retain available
final outcomes from incomplete blocks. Strong support also requires positive
mean A_s, primary p < .05 and a positive bootstrap lower bound. A failed
behavioral gate prevents the comparable-behavior claim. Technical analyses
can never declare strong confirmatory support.

Organization distances use the renormalized expert causal A/B profiles,
normalized Frobenius distance, and optimal assignment. Flat models allow
all expert permutations. DL models report both global and within-population
alignment, allowing a population swap. Routing tables contain global expert
weights plus population rows for DL; blank expert IDs identify population
rows. The descriptive collapse flag means a task-average routing probability
above .95 at an evaluated checkpoint; it is never an exclusion rule.
Communication message norms describe the transmitted bottleneck vector;
gate values refer to its receiving population. Parameter-count `head`
includes the final and population normalization modules so categories sum
to the total.
