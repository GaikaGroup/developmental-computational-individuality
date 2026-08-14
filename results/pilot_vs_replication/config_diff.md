# Pilot 1 vs Replication 1 Configuration Diff

| Field | Pilot 1 | Replication 1 | Status |
|---|---|---|---|
| D_s_definition | mean(S_causal_BLOCKED_AB_s, S_causal_BLOCKED_BA_s) - S_causal_INTERLEAVED_s | mean(S_causal_BLOCKED_AB_s, S_causal_BLOCKED_BA_s) - S_causal_INTERLEAVED_s | MATCH |
| S_causal_definition | 0.5 * sum_e abs(Delta_e,A - Delta_e,B) | 0.5 * sum_e abs(Delta_e,A - Delta_e,B) | MATCH |
| S_route_definition | 0.5 * sum_e abs(q_A,e - q_B,e) | 0.5 * sum_e abs(q_A,e - q_B,e) | MATCH |
| Task A | 1[x0*x1 + x2*x3 > 0] | 1[x0*x1 + x2*x3 > 0] | MATCH |
| Task B | 1[x8*x9 - x10*x11 > 0] | 1[x8*x9 - x10*x11 > 0] | MATCH |
| architecture | 18→128→128 shared encoder; Linear(128,2) softmax router; two Linear(128,256)→128 experts; residual LayerNorm; Linear(128,1) | 18→128→128 shared encoder; Linear(128,2) softmax router; two Linear(128,256)→128 experts; residual LayerNorm; Linear(128,1) | MATCH |
| batch_size | 256 | 256 | MATCH |
| common_phase_steps | 16000 | 16000 | MATCH |
| curricula | INTERLEAVED, BLOCKED_AB, BLOCKED_BA with identical 16,000-step common phase | INTERLEAVED, BLOCKED_AB, BLOCKED_BA with identical 16,000-step common phase | MATCH |
| deterministic_algorithms | not enabled in original Pilot | enabled with strict unsupported-operation failure | DIFFERENCE |
| early_phase_steps | 4000 | 4000 | MATCH |
| evaluation_examples_per_task | 50000 | 50000 | MATCH |
| expert_architecture | two identical Linear(128,256)→GELU→Linear(256,128) | two identical Linear(128,256)→GELU→Linear(256,128) | MATCH |
| input_dimensions | 16 Gaussian features + 2D one-hot task indicator = 18 | 16 Gaussian features + 2D one-hot task indicator = 18 | MATCH |
| knockout_variants | renormalized primary only | renormalized primary and non-renormalized secondary | DIFFERENCE |
| lambda_balance | 0.01 | 0.01 | MATCH |
| learning_rate | 0.0003 | 0.0003 | MATCH |
| lr_schedule | 500-step linear warmup then cosine decay | 500-step linear warmup then cosine decay | MATCH |
| metadata_hashes | config/source hashes unavailable in original result metadata | config hash and source hash recorded per run | DIFFERENCE |
| optimizer | AdamW | AdamW | MATCH |
| parameter_count | 151427 | 151427 | MATCH |
| primary_knockout | renormalized: survivor weight 1 | renormalized: survivor weight 1 | MATCH |
| requested_device | auto | auto | MATCH |
| router_architecture | Linear(128,2), Softmax(dim=-1) | Linear(128,2), Softmax(dim=-1) | MATCH |
| router_entropy_definition | entropy of task-mean routing vectors in original result JSON | true mean per-example entropy | DIFFERENCE |
| test_data_seed | 9001729 | 314159 | DIFFERENCE |
| training_data_seed | 1729 | 271828 | DIFFERENCE |
| training_steps | 20000 | 20000 | MATCH |
| weight_decay | 0.0001 | 0.0001 | MATCH |
