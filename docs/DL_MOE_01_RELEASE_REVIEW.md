# DL-MoE-01 v1.0 release review

This release records the scientific candidate and its implementation before
confirmatory data collection. The YAML/Markdown scientific protocol, seed
manifest and confirmatory config retain the four original scientific hashes.
The historical `protocol_hashes.csv` code-commit row identifies the earlier
preparation state; the separate execution-freeze JSON records this release's
actual commit, tag, source hash and environment without rewriting that file.

## Review scope and corrections

Reviewed execution order and paired initialization, batch identities,
balanced-accuracy interventions, seed-level endpoints, Holm correction,
behavioral gates, artifact extraction and integrity, recovery, environment
capture and explicit release/run separation.

Two additional infrastructure interruptions were reproduced by regression
tests and fixed before tagging:

- If `latest.pt` is durable but the retained checkpoint write is interrupted,
  resume now restores that retained checkpoint from the same saved state
  before evaluation. It does not retrain or change the seed.
- If execution identity was written but initial protocol artifacts were not
  all created, resume now completes missing artifacts and rejects mismatched
  existing files. The original scientific bytes are preserved.

These corrections change recovery behavior only. They do not change the
architecture, optimizer, sample size, data ordering, endpoints or statistical
criteria. No confirmatory outcomes were used in review.

## Validation and limitations

`make ci` covers both packages, scientific hashes, the read-only 180-model dry
run, offline package build and bytecode compilation. The release must also
pass these checks from a detached clean checkout of the annotated tag.
Technical tests exercise reduced models; the local preflight separately
checks full-size model training operations and interventions on CPU using
non-confirmatory data. Neither is a full 20,000-step scientific run.

The current host is macOS arm64 with Python 3.14.6 and PyTorch 2.13.0. CUDA is
unavailable. MPS is available but is not selected by this execution path;
`auto` resolves to CPU. The environment snapshot is platform-specific. A
move to a different machine/backend requires its own preflight and environment
record before training. Formatter, linter and static type checks are not
configured. Local Git tags and hash manifests provide local provenance,
not an external timestamped preregistration publication or remote backup.

The 180-model confirmatory series remains a separate explicit execution step.

Scientific artifacts are marked `-text` in `.gitattributes` to preserve their
bytes under different Git line-ending settings. Frozen CSVs retain their
original CRLF endings, and historical files retain existing trailing blank
lines. Release whitespace verification allows those existing representations
instead of changing scientific hashes for cosmetic formatting.
