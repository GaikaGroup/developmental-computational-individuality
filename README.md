# DCI pilot

Minimal PyTorch implementation of the attached developmental computational
individuality pilot. The frozen architecture, curriculum, optimizer and
metrics are recorded in `configs/default.yaml` and the specification.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=src pytest
PYTHONPATH=src python -m dci_pilot.train --condition INTERLEAVED --seed 0
PYTHONPATH=src python -m dci_pilot.sweep --seeds 0:19
```

Debugging runs must not be included in confirmatory statistics.

## Monitoring a background sweep

Start without occupying the terminal:

```bash
make pilot-background
```

Monitor live stdout/stderr and structured progress independently:

```bash
tail -f results/logs/sweep.log
tail -f results/progress.jsonl
pgrep -af "dci_pilot.sweep|dci_pilot.train"
find results/raw -name result.json | wc -l
```

`progress.jsonl` records `started`, every 500 training steps, checkpoint
creation, `completed`, and `failed` events. Each completed run also writes its
full metrics and metadata to `results/raw/seed_<seed>/<condition>/result.json`.

## Frozen pilot audit

After the sweep, `dci-analyze` regenerates the report and writes:

- explicit pass/fail status for behavioral equivalence, persistent causal difference, and nontrivial causal effect;
- paired seed-level results and permutation-invariant organizational distances;
- `results/summary/frozen_snapshot/` with the frozen config, requirements, `pip freeze`, runtime metadata, and SHA-256 checksums of all result files.

The snapshot records that this workspace is not under Git, so the result is
identified by its file checksums rather than a commit hash.

Generate the canonical source payload for the full English technical report:

```bash
.venv/bin/python -m dci_pilot.detailed_report
```

The validated portable report is stored at
`results/summary/detailed_report.html`; exact condition-level and seed-level
tables are stored alongside it as CSV files.

## Exact independent replication

Pilot V1 is preserved read-only under `results/pilot_v1_frozen/`. The frozen
replication protocol and data seeds are in `replication_manifest.md` and
`replication_config.yaml`; they must be committed before starting the sweep.

```bash
make replication-background
tail -f results/replication/logs/replication.log
tail -f results/replication/progress.jsonl
```

The workflow trains seeds 100–119 in all three conditions, evaluates both
renormalized and non-renormalized knockouts at steps 4k/10k/20k, and writes the
preregistered analysis and reports under `results/replication/`.
