from __future__ import annotations

import csv
from pathlib import Path

SCHEMAS = {
    "run_manifest.csv": "experiment_id preregistration_version seed_block_id seed_role architecture developmental_condition run_id model_init_seed development_data_seed common_data_seed common_order_seed eval_seed experience_manifest_sha256 initial_state_sha256 config_sha256 code_commit parameter_count_total parameter_count_encoder parameter_count_experts parameter_count_routers parameter_count_communication parameter_count_head device torch_version cuda_version python_version run_status start_time end_time wall_time_seconds final_checkpoint_available primary_endpoint_available router_collapse_flag numerical_failure_flag infrastructure_incident_flag notes",
    "behavior_metrics.csv": "experiment_id seed_block_id architecture developmental_condition checkpoint task accuracy balanced_accuracy loss n_eval_examples eval_seed run_id",
    "expert_knockout.csv": "experiment_id seed_block_id architecture developmental_condition checkpoint knockout_mode expert_id population_id task performance_normal performance_knockout delta_balanced_accuracy renormalized run_id",
    "population_knockout.csv": "experiment_id seed_block_id developmental_condition checkpoint population_id task knockout_mode performance_normal performance_knockout delta_balanced_accuracy run_id",
    "communication_knockout.csv": "experiment_id seed_block_id developmental_condition checkpoint intervention task performance_normal performance_intervention delta_balanced_accuracy run_id",
    "routing_diagnostics.csv": "experiment_id seed_block_id architecture developmental_condition checkpoint task population_id expert_id mean_routing_probability utilization routing_entropy router_collapse_flag run_id",
    "communication_diagnostics.csv": "experiment_id seed_block_id developmental_condition checkpoint task direction mean_message_norm median_message_norm gate_value communication_scale run_id",
    "run_summary.csv": "experiment_id seed_block_id architecture developmental_condition balanced_accuracy_A balanced_accuracy_B mean_balanced_accuracy s_expert_causal_renorm s_expert_causal_unrenorm s_population_causal organizational_profile_hash checkpoint run_id",
    "seed_effects.csv": "experiment_id seed_block_id s_dl_int s_dl_ab s_dl_ba s_flat_int s_flat_ab s_flat_ba d_dl d_flat architecture_amplification_a s_dl_int_unrenorm s_dl_ab_unrenorm s_dl_ba_unrenorm s_flat_int_unrenorm s_flat_ab_unrenorm s_flat_ba_unrenorm d_dl_unrenorm d_flat_unrenorm architecture_amplification_a_unrenorm s_pop_dl_int s_pop_dl_ab s_pop_dl_ba d_pop_dl primary_endpoint_available behavior_equivalence_complete",
    "behavioral_equivalence.csv": "architecture task comparison mean_difference sd_difference ci90_lower ci90_upper equivalence_margin_lower equivalence_margin_upper tost_p_lower tost_p_upper tost_p_max equivalent",
    "organizational_distance.csv": "experiment_id seed_block_id architecture comparison checkpoint raw_distance aligned_distance alignment_mode population_swap_allowed model_a_run_id model_b_run_id",
    "statistical_summary.csv": "hypothesis endpoint analysis_set_n mean median sd standard_error ci95_lower ci95_upper effect_size_name effect_size test_name alternative test_statistic p_value_raw p_value_adjusted positive_count zero_count negative_count supported notes",
    "exclusions_and_incidents.csv": "seed_block_id run_id architecture developmental_condition incident_type incident_time scientific_or_infrastructure excluded_from_primary reason action_taken replacement_seed_used deviation_id",
    "protocol_hashes.csv": "artifact path sha256 created_at",
}


def write_empty_schema_files(directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for filename, columns in SCHEMAS.items():
        path = directory / filename
        if not path.exists():
            with path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(columns.split())
