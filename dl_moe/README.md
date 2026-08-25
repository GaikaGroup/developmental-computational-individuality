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
