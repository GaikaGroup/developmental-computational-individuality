# Pilot V1 Frozen Snapshot

Pilot V1 is preserved at `results/pilot_v1_frozen/` as a read-only snapshot.
It contains all 60 result JSON files, 60 initial states, 180 trained
checkpoints, the original summaries and figures, the original configuration,
and a source/test snapshot. `SHA256SUMS` covers every file in the snapshot.

The snapshot was created before replication code was introduced. Seeds 0–19
must not be retrained or replaced. New checkpoint analyses read only from this
snapshot and write to separate result directories.
