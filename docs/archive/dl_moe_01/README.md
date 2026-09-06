# DL-MoE-01 completed study archive

This is a retrospective release of a completed experiment, not a public
preregistration and not a Registered Report. The protocol was frozen locally
before the full run according to local records. Git timestamps and checksums
alone do not independently establish that chronology.

## Scientific outcome

All 30 primary blocks (180 models) completed, with no numerical failures or
outcome-based exclusions. Primary architecture amplification A = D_DL − D_Flat
has mean −0.01204388908421, 95% bootstrap CI [−0.0205617259, −0.0032389058],
t(29) = −2.661546, one-sided greater p = 0.993725449, Cohen's dz = −0.485930.
The predicted positive amplification was not supported. All eight within-
architecture behavioral TOST comparisons passed the ±0.02 margins. The
negative contrast does not establish general superiority of either architecture.

## Release files and reconstruction

Download the compact `dl-moe-01-evidence.zip`, the frozen-source ZIP, this
README, the checksums and the manuscript. For complete checkpoint recovery,
also download each `dl-moe-01-checkpoints-*.zip` file. These are independent
ZIP archives, not binary fragments. Extract all ZIPs into the same empty
working directory. They preserve original repository-relative paths. The
compact evidence archive deliberately omits `.pt` checkpoint tensors; the
three checkpoint ZIPs provide every such file in the original raw manifest.

The original execution code is commit
`e57710fcb7b2e15b448ae2438f041d2a98fd6929`. The release's newer commit contains
only archival documentation and packaging tooling. Use the frozen-source ZIP
for exact analysis reproduction; do not substitute an unrelated default branch.
Historical pilot results are not included in this DL-MoE-01 release dataset.

Check downloaded asset bytes with `shasum -a 256 -c SHA256SUMS.txt`.
After extracting, run `python3 verify_archive.py` to check every packaged
file against ARCHIVE_MANIFEST.json without loading model checkpoints. With
all checkpoint ZIPs extracted, the original raw/raw_manifest.json inventory
can also be fully verified. A missing checkpoint is an intentional limitation
of the compact-only download, not evidence of a complete raw verification.

## Reproduce the primary analysis without training

Create a Python environment using the frozen source requirements. The supplied
CPU/macOS/Python 3.14 package snapshot records the original environment;
other platforms need their own validated installation. From the reconstructed
repository root (install packages only in the new environment):

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e ./dl_moe
PYTHONPATH=dl_moe/src .venv/bin/python -m dl_moe.experiments.dl_moe_01.analyze \
  --results results/dl_moe_01/confirmatory_v1/tables \
  --preregistration dl_moe/experiments/dl_moe_01/preregistration.yaml \
  --output results/archive_reanalysis
```

The compact archive contains all 13 extracted CSV tables and original manifest,
canonical analysis outputs, raw non-tensor evidence, post-run validation scripts
and reports. Its SHA manifest covers these files separately from the immutable
original raw manifest. Do not overwrite the original canonical outputs.

## Scope of validation

The release packaging checks every original raw checksum and every archived
member. It independently recomputes the primary analysis from extracted tables.
Earlier post-run audit recalculated 1,080 specialization values and all eight
TOST intervals. Re-evaluation from tensors covered six final models in the
first block and expert 0 in two modes; it was not a second execution of every
intervention. “Independent” here describes a separate calculation, not an audit
by an unaffiliated research team.

The manuscript is an author-review draft. Public-release metadata supersedes
its preparation-time availability statement; a public release timestamp records
availability from release time, not preregistration before data generation.
No archive DOI has yet been assigned. Licensing status is stated in RIGHTS.md.
