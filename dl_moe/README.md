# DL-MoE v1.0

This is an independent scientific-experimental project next to the frozen
`dci_pilot` artifact. It tests architecture × developmental-history
interactions using symmetric populations, hierarchical routing, explicit
communication bottlenecks, and causal interventions.

```bash
PYTHONPATH=dl_moe/src pytest -q dl_moe/tests
PYTHONPATH=dl_moe/src python -m dl_moe.smoke
```

The smoke experiment is software validation, not scientific evidence. The
candidate confirmatory protocol is in `configs/confirmatory/dl_moe_v1.yaml`.

## DL-MoE-01 execution

The complete runner, checkpoint recovery, raw freeze, table extraction and
preregistered analysis are implemented. See
[`../docs/DL_MOE_01_EXECUTION.md`](../docs/DL_MOE_01_EXECUTION.md) for the
technical validation workflow and the separately authorized scientific run.
Run `make ci` from the repository root. Full confirmatory execution requires
a reviewed clean tagged checkout; preparing this code does not launch it.
