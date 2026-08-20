"""Generate the minimal tracked checkpoint-panel locks from frozen local inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from tools.gate13_causal_return.checkpoint_panel.panel import (
    BASELINE_EXECUTION_ID,
    BASELINE_FAILURE_LOCALIZATION_COMMIT,
    BASELINE_MODEL,
    BASELINE_OPERATOR_LOCK_SHA256,
    BASELINE_REVISION,
    CHECKPOINTS,
    EXECUTION_ORDER,
    FORECAST_CONTINGENCY_MULTIPLIER,
    MAX_FRESH_OPERATOR_FORWARD_COUNT,
    MAX_TRACK_A_FORWARD_COUNT_PER_CHECKPOINT,
    M1_FORWARD_COUNT_PER_CHECKPOINT,
    NORMALIZED_OPERATOR_DEPTHS,
    OPERATOR_PRIORITY,
    OPERATIONAL_EXECUTION_ATTEMPTS,
    PANEL_BINDING_ID,
    PANEL_CREDIT_RESERVE_USD,
    PANEL_SPEND_CEILING_USD,
    PANEL_TARGET_MAX_SPEND_USD,
    QUALIFICATION_THRESHOLDS,
    SELECTED_INSTRUMENT,
    TRANSFORMERS_COMMIT,
    TRANSFORMERS_RELEASE,
    TRANSFORMERS_SOURCE,
    canonical_json,
    derive_operator_layers,
    execution_identity,
    image_definition_payload,
    image_definition_sha256,
    model_volume_name,
    qualification_contract,
    result_volume_name,
    score_slot_record,
    sha256_file,
    validate_provider_metadata,
)


class FreezeCommandError(RuntimeError):
    """Raised when a tracked panel artifact cannot be generated exactly."""


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _tokenizer_report(assets_root: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer

    asset_specs = {
        "qwen3_8b_baseline": {
            "asset_slug": "qwen3-8b",
            "repo_id": BASELINE_MODEL,
            "revision": BASELINE_REVISION,
        },
        **CHECKPOINTS,
    }
    rows: dict[str, Any] = {}
    for key, spec in asset_specs.items():
        root = assets_root / str(spec["asset_slug"])
        tokenizer = AutoTokenizer.from_pretrained(
            root,
            local_files_only=True,
            use_fast=True,
            trust_remote_code=False,
        )
        labels = []
        from tools.gate13_causal_return.stepwise.compiler import BANK_SLICES, OPAQUE_LABELS

        for index, label in enumerate(OPAQUE_LABELS):
            ids = [int(value) for value in tokenizer.encode(label, add_special_tokens=False)]
            labels.append(
                {
                    "index": index,
                    "label": label,
                    "banks": [
                        bank
                        for bank, (start, stop) in BANK_SLICES.items()
                        if start <= index < stop
                    ],
                    "token_ids": ids,
                    "one_token": len(ids) == 1,
                    "non_special": len(ids) == 1 and ids[0] not in tokenizer.all_special_ids,
                    "round_trip": tokenizer.decode(ids, skip_special_tokens=False) == label,
                }
            )
        transfer = [row for row in labels if int(row["index"]) >= 16]
        if not all(
            row["one_token"] and row["non_special"] and row["round_trip"]
            for row in transfer
        ):
            raise FreezeCommandError(f"transfer codebook is invalid for {key}")
        rows[key] = {
            "repo_id": spec["repo_id"],
            "revision": spec["revision"],
            "tokenizer_json_sha256": sha256_file(root / "tokenizer.json"),
            "tokenizer_config_sha256": sha256_file(root / "tokenizer_config.json"),
            "vocab_size": len(tokenizer),
            "special_token_ids": sorted(int(value) for value in tokenizer.all_special_ids),
            "effective_chat_template_sha256": hashlib.sha256(
                str(tokenizer.chat_template).encode("utf-8")
            ).hexdigest(),
            "labels": labels,
            "transfer_labels_all_one_non_special_token": True,
            "score_slot": score_slot_record(tokenizer, key),
        }
    baseline = rows["qwen3_8b_baseline"]
    fourteen = rows["qwen3_14b"]
    panel_s_identical = all(
        baseline[field] == fourteen[field]
        for field in (
            "tokenizer_json_sha256",
            "tokenizer_config_sha256",
            "vocab_size",
            "special_token_ids",
            "effective_chat_template_sha256",
        )
    ) and [row["token_ids"] for row in baseline["labels"]] == [
        row["token_ids"] for row in fourteen["labels"]
    ]
    if not panel_s_identical:
        raise FreezeCommandError("Panel S exact 8B codebook transfer is not tokenizer-identical")
    panel_g_keys = ("qwen3_5_27b", "qwen3_6_27b", "qwen3_8_27b")
    common_literals = []
    for index in range(16, 80):
        source = rows[panel_g_keys[0]]["labels"][index]
        common_literals.append(
            {
                "index": index,
                "label": source["label"],
                "banks": source["banks"],
                "token_ids": {
                    key: rows[key]["labels"][index]["token_ids"][0]
                    for key in panel_g_keys
                },
            }
        )
    report = {
        "schema_version": "gate13_checkpoint_panel_tokenizer_codebook_report_v1",
        "campaign_binding_id": PANEL_BINDING_ID,
        "selection_used_model_outputs": False,
        "candidate_source": "existing deterministic OPAQUE_LABELS ordered tuple",
        "filter": "indices 16..79 used by qualification and conditionally reserved operator halves; exact one-token, non-special, byte-round-trip",
        "ordering": "existing tuple order",
        "tie_break": "not applicable",
        "Panel_S": {
            "status": "EXACT_8B_CODEBOOK_REUSED",
            "tokenizer_assets_byte_identical": True,
            "same_literal_strings": True,
            "same_one_token_ids": True,
        },
        "Panel_G": {
            "status": "COMMON_LITERAL_CODEBOOK",
            "priority": 2,
            "tokenizer_assets_all_byte_identical": False,
            "same_literal_labels_one_non_special_token_all_checkpoints": True,
            "per_checkpoint_bank_forbidden": True,
            "common_transfer_labels": common_literals,
            "unused_development_only_incompatibilities": [
                {
                    "label": rows["qwen3_5_27b"]["labels"][10]["label"],
                    "index": 10,
                    "reason": "two tokens in all Panel G tokenizers; outside qualification and operator banks",
                }
            ],
        },
        "checkpoints": rows,
    }
    report["report_payload_sha256"] = hashlib.sha256(
        canonical_json(report).encode("utf-8")
    ).hexdigest()
    return report


def _runtime_manifest(transformers_wheel: Path) -> dict[str, Any]:
    wheel_sha = sha256_file(transformers_wheel)
    return {
        "schema_version": "gate13_checkpoint_panel_common_runtime_v1",
        "campaign_binding_id": PANEL_BINDING_ID,
        "Panel_S": {
            "status": "FROZEN",
            "reuse_prior_exact_runtime": True,
            **image_definition_payload("Panel S"),
            "driver": "580.95.05",
            "dtype": "bfloat16",
            "quantization": False,
            "batch": 1,
            "minimum_free_vram_margin_mib": 8192,
            "model_class": "Qwen3ForCausalLM",
            "processor_class": "Qwen2Tokenizer",
        },
        "Panel_G": {
            "status": "FROZEN_COMMON_RUNTIME_CANDIDATE_1_PENDING_MODAL_M0_M1",
            "candidate_selection_rule": "latest official release commit supporting all three repositories",
            "candidate_commits": [
                {
                    "candidate": 1,
                    "release": TRANSFORMERS_RELEASE,
                    "commit": TRANSFORMERS_COMMIT,
                    "source": TRANSFORMERS_SOURCE,
                    "selected": True,
                    "qualification_status": "PENDING_FORWARD_ZERO_MODEL_LOAD_TEXT_ONLY_PROCESSOR_AND_SCORE_SLOT_SMOKE",
                },
                {"candidate": 2, "status": "UNOPENED_FIRST_CANDIDATE_QUALIFIED_FORWARD_ZERO"},
            ],
            "maximum_candidate_commits": 2,
            **image_definition_payload("Panel G", wheel_sha),
            "driver": "580.95.05",
            "dtype": "bfloat16",
            "quantization": False,
            "batch": 1,
            "minimum_free_vram_margin_mib": 8192,
            "model_class": "Qwen3_5ForConditionalGeneration",
            "processor_class": "Qwen3VLProcessor",
            "text_only": True,
            "image_video_input": False,
            "cpu_offload": False,
            "disk_offload": False,
            "multi_gpu": False,
            "tensor_parallel": False,
            "H100_fallback": "FORBIDDEN",
        },
    }


def _operator_reservations() -> dict[str, Any]:
    rows = {}
    for key, spec in CHECKPOINTS.items():
        rows[key] = {
            "repo_id": spec["repo_id"],
            "revision": spec["revision"],
            "node_conditions": [
                "phase0_state0",
                "phase0_state1",
                "phase1_state0",
                "phase1_state1",
                "phase1_state1_broken",
            ],
            "text_token_position": "LAST_PROMPT_TOKEN_BEFORE_FORCED_CHOICE",
            "activation_tensor": "language-model residual hidden_states[layer][0,-1]",
            "normalized_layer_depths": list(NORMALIZED_OPERATOR_DEPTHS),
            "layer_derivation": "round_half_up(normalized_depth * num_hidden_layers), clamped to [1,L-1]",
            "number_of_layers": spec["layers"],
            "actual_hidden_state_indices": derive_operator_layers(int(spec["layers"])),
            "independent_nuisance_halves": {
                "half_1_seed_range": [810000, 810999],
                "half_2_seed_range": [910000, 910999],
                "samples_per_node_per_half": 24,
                "disjoint_opaque_codebooks": True,
                "disjoint_templates": True,
                "disjoint_demonstration_instances": True,
                "disjoint_episode_seeds": True,
                "bootstrap": "SECONDARY_ONLY_NOT_A_SUBSTITUTE",
            },
        }
    return {
        "schema_version": "gate13_checkpoint_panel_fresh_square_reservations_v1",
        "campaign_binding_id": PANEL_BINDING_ID,
        "authority_status": "CONDITIONALLY_AUTHORIZED_AFTER_PANEL_A2_PASS",
        "inherited_operator_lock_sha256": BASELINE_OPERATOR_LOCK_SHA256,
        "scientific_question": "Can the operator packet be re-estimated on independent nuisance halves, with broken-square response above the split-half self-estimation floor?",
        "selection_priority": list(OPERATOR_PRIORITY),
        "selection_uses_track_a_only_as_resource_gate": True,
        "track_a_effect_size_or_failure_strata_used_for_surface_selection": False,
        "per_checkpoint": rows,
        "frame_estimator": "CENTERED_SVD_WITH_DETERMINISTIC_COLUMN_SIGN",
        "frame_rank": 4,
        "frame_relative_singular_tolerance": 1e-6,
        "half_2_alignment": "NODE_WISE_RIGHT_ORTHOGONAL_PROCRUSTES_TO_HALF_1",
        "edge_estimator": "PAIRED_RIDGE_LINEAR_MAP_IN_NODE_LOCAL_COORDINATES",
        "edge_ridge_relative": 0.001,
        "edge_rank_tolerance": 1e-8,
        "edge_condition_ceiling": 1_000_000.0,
        "minimum_node_support_per_half": 24,
        "exact_square_registry": {
            "path_p": ["phase0_state0", "phase0_state1", "phase1_state1"],
            "path_q": ["phase0_state0", "phase1_state0", "phase1_state1"],
            "external_relation": "J_(t+1) tau = tau J_t",
        },
        "broken_square_registry": {
            "path_p": ["phase0_state0", "phase0_state1", "phase1_state1"],
            "path_q": ["phase0_state0", "phase1_state0", "phase1_state1_broken"],
            "corruption": "one frozen transition demonstration output inverted without changing row count or choice set",
        },
        "operator_packet": [
            "P_p",
            "P_q",
            "Delta_pq",
            "singular spectra",
            "S_p",
            "S_q",
            "H_path where qualified",
            "H_edge where qualified",
            "rank and conditioning",
            "split-half disagreement",
            "exact-square response",
            "broken-square response",
        ],
        "qualification": {
            "split_half_singular_floor_max": 0.2,
            "broken_sensitivity_rule": "minimum broken-square normalized Delta across halves exceeds max(2*split-half floor, floor+0.05, maximum exact-square normalized Delta+0.05)",
            "minimum_qualified_layer_count": 2,
            "fixed_layer_count": 3,
        },
        "maximum_forward_count": MAX_FRESH_OPERATOR_FORWARD_COUNT,
        "forbidden_pass_conditions": [
            "nonzero holonomy alone",
            "operator existence alone",
            "difference from legacy scalar alone",
        ],
        "historical_B2A": "TERMINATED_SUBSTRATE_INADEQUATE",
        "legacy_scalar_fresh_B2B": "RESERVED_NOT_PART_OF_THIS_CAMPAIGN",
        "TRACK_C": "CLOSED",
    }


def _panel_lock(provider: Mapping[str, Any], runtime: Mapping[str, Any], tokenizers: Mapping[str, Any]) -> dict[str, Any]:
    registry = validate_provider_metadata(provider)
    contract = qualification_contract()
    return {
        "schema_version": "gate13_checkpoint_transfer_panel_lock_v1",
        "authority_status": "FROZEN_BEFORE_NEW_CHECKPOINT_SCIENTIFIC_RESPONSE",
        "campaign_binding_id": PANEL_BINDING_ID,
        "objective": "Describe checkpoint-level movement of one frozen stepwise visible-state qualification boundary; not an isolated causal effect of size, generation, architecture, or training.",
        "immutable_8B_baseline": {
            "repo_id": BASELINE_MODEL,
            "revision": BASELINE_REVISION,
            "failure_localization_commit": BASELINE_FAILURE_LOCALIZATION_COMMIT,
            "execution_identity": BASELINE_EXECUTION_ID,
            "selected_instrument": SELECTED_INSTRUMENT,
            "STREAM-A0": "PASS",
            "STREAM-A0_ONE_STEP": "32/32",
            "STREAM-A0_MIN_STRATA": "8/8",
            "STREAM-A0_SELF_FED_ROLLOUT": "8/8",
            "STREAM-A1": "FAIL_TO_QUALIFY",
            "STREAM-A1_CORRECT": "13/16",
            "STREAM-A1_CORRECT_MINUS_STRONGEST_CONTROL": 0.3125,
            "STREAM-A1_MINIMUM_STRATUM": "2/4",
            "STREAM-A2": "UNOPENED_STREAM_A1_FAIL",
            "FRESH_OPERATOR_QUALIFICATION": "UNOPENED",
            "rerun_or_rejudgment": "FORBIDDEN",
        },
        "panels": {
            "Panel_S": {
                "name": "PROSPECTIVE_SAME_GENERATION_NEXT_CHECKPOINT_TRANSFER",
                "existing_baseline_no_rerun": BASELINE_MODEL,
                "required_new_checkpoints": ["Qwen/Qwen3-14B"],
                "causal_parameter_count_experiment": False,
            },
            "Panel_G": {
                "name": "PROSPECTIVE_NEAR_MATCHED_27B_GENERATIONAL_CHECKPOINT_PANEL",
                "required_checkpoints": [
                    "Qwen/Qwen3.5-27B",
                    "Qwen/Qwen3.6-27B",
                    "Qwen/Qwen3.8-27B",
                ],
                "conditional_checkpoint": None,
                "common_runtime": runtime["Panel_G"],
            },
        },
        "official_model_registry": registry,
        "instrument_transfer": {
            **contract,
            "external_binary_state_algebra": "XOR identity/flip",
            "stepwise_visible_state_bottleneck": True,
            "full_history_exclusion": True,
            "self_fed_rollout_semantics": "previous forced-choice prediction becomes next current visible state",
            "canonical_message_content": "UNCHANGED",
            "qualification_episode_seed_demonstration_control_identities": "UNCHANGED",
            "controls": ["correct", "label_shuffled", "corrupted", "format_matched", "marker_only"],
            "output": "raw next-token forced-choice logits",
            "generation": False,
            "sampling": False,
            "transfer_surface_already_observed_on_8B": True,
            "panel_wide_held_out_claim": "FORBIDDEN",
            "checkpoint_specific_tuning": "FORBIDDEN",
        },
        "tokenizer_and_codebook": {
            "report_file": "panel_tokenizer_codebook_report.json",
            "Panel_S_status": tokenizers["Panel_S"]["status"],
            "Panel_G_status": tokenizers["Panel_G"]["status"],
            "model_output_used_for_selection": False,
        },
        "score_slot_equivalence": {
            "rule": "logits at final prompt tensor index predict the first semantic assistant-answer token after the official closed empty non-thinking protocol boundary",
            "active_think_prefix": "FORBIDDEN",
            "reasoning_channel_token_at_slot": "FORBIDDEN",
            "forced_assistant_prose": "FORBIDDEN",
            "vision_video_dummy_placeholder": "FORBIDDEN",
            "tool_prefix": "FORBIDDEN",
            "per_checkpoint_forward_zero_records": {
                key: tokenizers["checkpoints"][key]["score_slot"] for key in CHECKPOINTS
            },
        },
        "thresholds": QUALIFICATION_THRESHOLDS,
        "conditional_ladder": [
            "CHECKPOINT-A0 FAIL -> checkpoint stop; A1/A2 unopened",
            "CHECKPOINT-A0 PASS -> A1",
            "CHECKPOINT-A1 FAIL -> checkpoint stop; A2 unopened",
            "CHECKPOINT-A1 PASS -> A2",
            "CHECKPOINT-A2 PASS or FAIL -> checkpoint stop",
        ],
        "checkpoint_independence": True,
        "execution_order": list(EXECUTION_ORDER),
        "per_checkpoint_maximum_track_a_forwards": MAX_TRACK_A_FORWARD_COUNT_PER_CHECKPOINT,
        "preflight_forward_count_per_checkpoint": M1_FORWARD_COUNT_PER_CHECKPOINT,
        "fresh_operator": {
            "reservation_file": "fresh_square_operator_reservations.json",
            "execution_gate": "after all Track A checkpoints; at least one A2 PASS",
            "priority": list(OPERATOR_PRIORITY),
            "maximum_executions": 1,
            "no_A2_status": "UNOPENED_NO_A2_QUALIFIED_CHECKPOINT",
        },
        "budget": {
            "hard_incremental_modal_spend_ceiling_usd": PANEL_SPEND_CEILING_USD,
            "target_max_spend_preserving_reserve_usd": PANEL_TARGET_MAX_SPEND_USD,
            "credit_reserve_usd": PANEL_CREDIT_RESERVE_USD,
            "forecast_contingency_multiplier": FORECAST_CONTINGENCY_MULTIPLIER,
            "automatic_retries": 0,
            "maximum_active_gpu_containers": 1,
            "region_premium": "FORBIDDEN",
            "non_preemptible_premium": "FORBIDDEN",
        },
        "gpu_policy": {
            "qwen3_14b": {"primary": "L40S", "fallback": None},
            "Panel_G": {
                "primary": "A100-80GB",
                "fallback": "H200_ONLY_AFTER_PREFLIGHT_OOM_OR_FROZEN_MARGIN_FAILURE",
                "H100": "FORBIDDEN",
            },
            "quantization": False,
            "cpu_disk_offload": False,
            "multi_gpu_tensor_parallel": False,
        },
        "terminal_states": [
            "PANEL_NO_A1_PASS",
            "PANEL_A1_PASS_NO_A2_PASS",
            "PANEL_A2_PASS_B_UNOPENED_BLOCKED",
            "PANEL_A2_PASS_B_FAIL",
            "PANEL_A2_AND_B_PASS",
            "CAMPAIGN_BUDGET_BLOCK_BEFORE_PANEL",
            "CAMPAIGN_BUDGET_BLOCK_DURING_PANEL",
            "PANEL_G_COMMON_RUNTIME_BLOCK",
            "CAMPAIGN_INFRASTRUCTURE_BLOCK",
        ],
        "closed_surfaces": {
            "A3": "CLOSED",
            "TRACK_C": "CLOSED",
            "FORMAL_GATE13": "CLOSED",
            "historical_B2A": "TERMINATED_SUBSTRATE_INADEQUATE",
            "legacy_scalar_fresh_B2B": "RESERVED_NOT_PART_OF_THIS_CAMPAIGN",
            "public_docs": "UNCHANGED",
            "other_model_families_32B_MoE_API": "FORBIDDEN",
        },
    }


def freeze_locks(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    provider = _load_json(args.provider_metadata.resolve())
    runtime = _runtime_manifest(args.transformers_wheel.resolve())
    tokenizers = _tokenizer_report(args.assets_root.resolve())
    reservations = _operator_reservations()
    _write_json(output / "panel_g_common_runtime.json", runtime)
    _write_json(output / "panel_tokenizer_codebook_report.json", tokenizers)
    _write_json(output / "fresh_square_operator_reservations.json", reservations)
    _write_json(output / "checkpoint_transfer_panel_lock.json", _panel_lock(provider, runtime, tokenizers))
    print(
        json.dumps(
            {
                name: sha256_file(output / name)
                for name in (
                    "checkpoint_transfer_panel_lock.json",
                    "panel_g_common_runtime.json",
                    "panel_tokenizer_codebook_report.json",
                    "fresh_square_operator_reservations.json",
                )
            },
            sort_keys=True,
        )
    )


def freeze_authorization(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    filenames = (
        "checkpoint_transfer_panel_lock.json",
        "panel_g_common_runtime.json",
        "panel_tokenizer_codebook_report.json",
        "fresh_square_operator_reservations.json",
    )
    runtime = _load_json(output / "panel_g_common_runtime.json")
    wheel_sha = runtime["Panel_G"]["transformers_wheel_sha256"]
    bindings = {name: sha256_file(output / name) for name in filenames}
    executions = {}
    for key, spec in CHECKPOINTS.items():
        identity = execution_identity(key)
        executions[key] = {
            "execution_identity": identity,
            "operational_attempt": OPERATIONAL_EXECUTION_ATTEMPTS[key],
            "repo_id": spec["repo_id"],
            "revision": spec["revision"],
            "model_volume_name": model_volume_name(key),
            "result_volume_name": result_volume_name(key),
            "automatic_retries": 0,
            "maximum_active_gpu_containers": 1,
        }
    authorization = {
        "schema_version": "gate13_checkpoint_panel_execution_authorization_v1",
        "campaign_binding_id": PANEL_BINDING_ID,
        "execution_authorized": True,
        "implementation_commit": args.implementation_commit,
        "freeze_commit": args.freeze_commit,
        "tracked_bindings": bindings,
        "image_definitions": {
            "Panel_S": {
                "sha256": image_definition_sha256("Panel S"),
                "payload": image_definition_payload("Panel S"),
            },
            "Panel_G": {
                "sha256": image_definition_sha256("Panel G", wheel_sha),
                "payload": image_definition_payload("Panel G", wheel_sha),
            },
        },
        "checkpoint_executions": executions,
        "execution_order": list(EXECUTION_ORDER),
        "fresh_operator_priority": list(OPERATOR_PRIORITY),
        "maximum_fresh_operator_executions": 1,
        "hard_incremental_modal_spend_ceiling_usd": PANEL_SPEND_CEILING_USD,
        "target_max_spend_preserving_reserve_usd": PANEL_TARGET_MAX_SPEND_USD,
        "credit_reserve_usd": PANEL_CREDIT_RESERVE_USD,
        "forecast_contingency_multiplier": FORECAST_CONTINGENCY_MULTIPLIER,
        "automatic_retries": 0,
        "maximum_active_gpu_containers": 1,
        "region_premium": "FORBIDDEN",
        "non_preemptible_premium": "FORBIDDEN",
        "old_8B_model_volume": "PRESERVE_UNCHANGED",
        "new_model_and_result_volumes": "DELETE_AFTER_HASH_VERIFIED_RETRIEVAL",
        "push": False,
        "closed_surfaces": {
            "A3": "CLOSED",
            "TRACK_C": "CLOSED",
            "FORMAL_GATE13": "CLOSED",
            "historical_B2A": "TERMINATED_SUBSTRATE_INADEQUATE",
            "legacy_scalar_fresh_B2B": "RESERVED_NOT_PART_OF_THIS_CAMPAIGN",
        },
    }
    _write_json(output / "panel_execution_authorization.json", authorization)
    print(json.dumps({"panel_execution_authorization.json": sha256_file(output / "panel_execution_authorization.json")}, sort_keys=True))


def validate_frozen_artifacts(output: Path, *, require_authorization: bool) -> dict[str, Any]:
    output = output.resolve()
    panel = _load_json(output / "checkpoint_transfer_panel_lock.json")
    runtime = _load_json(output / "panel_g_common_runtime.json")
    tokenizers = _load_json(output / "panel_tokenizer_codebook_report.json")
    operator = _load_json(output / "fresh_square_operator_reservations.json")
    values = (panel, runtime, tokenizers, operator)
    if any(value.get("campaign_binding_id") != PANEL_BINDING_ID for value in values):
        raise FreezeCommandError("campaign binding ID mismatch")
    if panel["execution_order"] != list(EXECUTION_ORDER):
        raise FreezeCommandError("checkpoint execution order drift")
    if panel["instrument_transfer"]["qualification_ledger_sha256"] != qualification_contract()[
        "qualification_ledger_sha256"
    ]:
        raise FreezeCommandError("qualification ledger drift")
    if panel["thresholds"] != QUALIFICATION_THRESHOLDS:
        raise FreezeCommandError("qualification threshold drift")
    registry = validate_provider_metadata(
        {"checkpoints": [{key: value for key, value in row.items() if key != "checkpoint_key"} for row in panel["official_model_registry"]]}
    )
    if [row["checkpoint_key"] for row in registry] != list(EXECUTION_ORDER):
        raise FreezeCommandError("official model registry drift")
    if runtime["Panel_G"]["transformers_commit"] != TRANSFORMERS_COMMIT:
        raise FreezeCommandError("Panel G Transformers commit drift")
    if runtime["Panel_G"]["candidate_commits"][0]["qualification_status"] != (
        "PENDING_FORWARD_ZERO_MODEL_LOAD_TEXT_ONLY_PROCESSOR_AND_SCORE_SLOT_SMOKE"
    ):
        raise FreezeCommandError("Panel G preflight state is not fail-closed")
    if tokenizers["Panel_S"]["status"] != "EXACT_8B_CODEBOOK_REUSED":
        raise FreezeCommandError("Panel S tokenizer status drift")
    if tokenizers["Panel_G"]["status"] != "COMMON_LITERAL_CODEBOOK":
        raise FreezeCommandError("Panel G common codebook drift")
    for key in CHECKPOINTS:
        slot = tokenizers["checkpoints"][key]["score_slot"]
        if slot["status"] != "PASS" or slot["active_think_prefix_before_score_slot"]:
            raise FreezeCommandError(f"score-slot drift: {key}")
        expected_layers = derive_operator_layers(int(CHECKPOINTS[key]["layers"]))
        if operator["per_checkpoint"][key]["actual_hidden_state_indices"] != expected_layers:
            raise FreezeCommandError(f"operator layer drift: {key}")
    hashes = {
        name: sha256_file(output / name)
        for name in (
            "checkpoint_transfer_panel_lock.json",
            "panel_g_common_runtime.json",
            "panel_tokenizer_codebook_report.json",
            "fresh_square_operator_reservations.json",
        )
    }
    if require_authorization:
        authorization = _load_json(output / "panel_execution_authorization.json")
        if not authorization.get("execution_authorized"):
            raise FreezeCommandError("panel execution is not authorized")
        if authorization["tracked_bindings"] != hashes:
            raise FreezeCommandError("authorization tracked hash binding mismatch")
        wheel_sha = runtime["Panel_G"]["transformers_wheel_sha256"]
        expected_images = {
            "Panel_S": image_definition_sha256("Panel S"),
            "Panel_G": image_definition_sha256("Panel G", wheel_sha),
        }
        if {
            name: row["sha256"] for name, row in authorization["image_definitions"].items()
        } != expected_images:
            raise FreezeCommandError("authorization image binding mismatch")
        for key in CHECKPOINTS:
            entry = authorization["checkpoint_executions"][key]
            if entry["execution_identity"] != execution_identity(key):
                raise FreezeCommandError(f"execution identity drift: {key}")
            if entry["operational_attempt"] != OPERATIONAL_EXECUTION_ATTEMPTS[key]:
                raise FreezeCommandError(f"operational attempt drift: {key}")
            if entry["model_volume_name"] != model_volume_name(key):
                raise FreezeCommandError(f"model Volume drift: {key}")
            if entry["result_volume_name"] != result_volume_name(key):
                raise FreezeCommandError(f"result Volume drift: {key}")
        hashes["panel_execution_authorization.json"] = sha256_file(
            output / "panel_execution_authorization.json"
        )
    return {"status": "PASS", "require_authorization": require_authorization, "sha256": hashes}


def validate_command(args: argparse.Namespace) -> None:
    print(
        json.dumps(
            validate_frozen_artifacts(
                args.output_dir,
                require_authorization=args.require_authorization,
            ),
            sort_keys=True,
        )
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    subparsers = value.add_subparsers(dest="command", required=True)
    locks = subparsers.add_parser("locks")
    locks.add_argument("--provider-metadata", type=Path, required=True)
    locks.add_argument("--assets-root", type=Path, required=True)
    locks.add_argument("--transformers-wheel", type=Path, required=True)
    locks.add_argument("--output-dir", type=Path, required=True)
    authorization = subparsers.add_parser("authorization")
    authorization.add_argument("--output-dir", type=Path, required=True)
    authorization.add_argument("--implementation-commit", required=True)
    authorization.add_argument("--freeze-commit", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--output-dir", type=Path, required=True)
    validate.add_argument("--require-authorization", action="store_true")
    return value


def main() -> None:
    args = parser().parse_args()
    if args.command == "locks":
        freeze_locks(args)
    elif args.command == "authorization":
        freeze_authorization(args)
    else:
        validate_command(args)


if __name__ == "__main__":
    main()
