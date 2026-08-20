"""Pure helpers for freezing and validating the Gate13 checkpoint panel.

This module does not load scientific weights or execute a model.  It binds the
existing ``natural_rule_v1`` surface to named immutable repositories, validates
tokenizer/score-slot equivalence, and derives the predeclared operator layers.
"""

from __future__ import annotations

import hashlib
import json
import struct
import uuid
from pathlib import Path
from typing import Any, Mapping

from tools.gate13_causal_return.stepwise.compiler import (
    BANK_SLICES,
    OPAQUE_LABELS,
    codebook_bank,
    compile_qualification_ledgers,
    prompt_contract_payload,
    render_step_prompt,
    sha256_json,
    validate_codebook_partition,
)


PANEL_BINDING_ID = "47171e04-dc39-4fda-b28d-24fe0b2f57eb"
SELECTED_INSTRUMENT = "natural_rule_v1"
BASELINE_MODEL = "Qwen/Qwen3-8B"
BASELINE_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
BASELINE_QUALIFICATION_LEDGER_SHA256 = (
    "f5e436496b3f607b6e0e51fb9856e04fa5b0beafc6011fb44a5dd0d84a1c898f"
)
BASELINE_OPERATOR_LOCK_SHA256 = (
    "f684788571150683eeca57295d53537bc728d7d80e3e229e05d9094010a1530b"
)
BASELINE_FAILURE_LOCALIZATION_COMMIT = "ab047352ff03cb0ce409664470cb633a9ea35ccc"
BASELINE_EXECUTION_ID = "e941e509-ab69-4965-85a2-f48a622d89b7"

TRANSFORMERS_RELEASE = "v5.15.1"
TRANSFORMERS_COMMIT = "550d7b3834670483a4df436541272c055dc364bf"
TRANSFORMERS_SOURCE = (
    "https://github.com/huggingface/transformers/archive/"
    f"{TRANSFORMERS_COMMIT}.tar.gz"
)

BASE_IMAGE = (
    "python:3.11.2-slim-bullseye@"
    "sha256:2f749ef90f54fd4b3c77cde78eec23ab5b8199d9ac84e4ced6ae523ef223ef7b"
)
BASE_IMAGE_MANIFEST_DIGEST = (
    "sha256:2f749ef90f54fd4b3c77cde78eec23ab5b8199d9ac84e4ced6ae523ef223ef7b"
)
BASE_IMAGE_LINUX_AMD64_DIGEST = (
    "sha256:9ad4ffc502779e5508f7ac1eccab4a22786b80bd53d721d735f6de0840b245a1"
)

CHECKPOINTS: dict[str, dict[str, Any]] = {
    "qwen3_14b": {
        "display_name": "Qwen3-14B",
        "repo_id": "Qwen/Qwen3-14B",
        "revision": "40c069824f4251a91eefaf281ebe4c544efd3e18",
        "asset_slug": "qwen3-14b",
        "panel": "PROSPECTIVE_SAME_GENERATION_NEXT_CHECKPOINT_TRANSFER",
        "model_class": "Qwen3ForCausalLM",
        "processor_class": "Qwen2Tokenizer",
        "layers": 40,
        "primary_gpu": "L40S",
        "fallback_gpu": None,
    },
    "qwen3_5_27b": {
        "display_name": "Qwen3.5-27B",
        "repo_id": "Qwen/Qwen3.5-27B",
        "revision": "fc05daec18b0a78c049392ed2e771dde82bdf654",
        "asset_slug": "qwen3_5-27b",
        "panel": "PROSPECTIVE_NEAR_MATCHED_27B_GENERATIONAL_CHECKPOINT_PANEL",
        "model_class": "Qwen3_5ForConditionalGeneration",
        "processor_class": "Qwen3VLProcessor",
        "layers": 64,
        "primary_gpu": "A100-80GB",
        "fallback_gpu": "H200",
    },
    "qwen3_6_27b": {
        "display_name": "Qwen3.6-27B",
        "repo_id": "Qwen/Qwen3.6-27B",
        "revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
        "asset_slug": "qwen3_6-27b",
        "panel": "PROSPECTIVE_NEAR_MATCHED_27B_GENERATIONAL_CHECKPOINT_PANEL",
        "model_class": "Qwen3_5ForConditionalGeneration",
        "processor_class": "Qwen3VLProcessor",
        "layers": 64,
        "primary_gpu": "A100-80GB",
        "fallback_gpu": "H200",
    },
    "qwen3_8_27b": {
        "display_name": "Qwen3.8-27B",
        "repo_id": "Qwen/Qwen3.8-27B",
        "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "asset_slug": "qwen3_8-27b",
        "panel": "PROSPECTIVE_NEAR_MATCHED_27B_GENERATIONAL_CHECKPOINT_PANEL",
        "model_class": "Qwen3_5ForConditionalGeneration",
        "processor_class": "Qwen3VLProcessor",
        "layers": 64,
        "primary_gpu": "A100-80GB",
        "fallback_gpu": "H200",
    },
}
EXECUTION_ORDER = tuple(CHECKPOINTS)
OPERATOR_PRIORITY = (
    "qwen3_8_27b",
    "qwen3_6_27b",
    "qwen3_5_27b",
    "qwen3_14b",
)

QUALIFICATION_THRESHOLDS = {
    "STREAM-A0": {
        "one_step_accuracy_min": 0.90,
        "minimum_transition_cell_accuracy_min": 0.80,
        "self_fed_rollout_exact_accuracy_min": 0.75,
    },
    "STREAM-A1": {
        "correct_demonstration_accuracy_min": 0.85,
        "correct_minus_strongest_control_min": 0.20,
        "minimum_correct_transition_cell_accuracy_min": 0.75,
        "each_control_independent_qualification_ceiling_exclusive": 0.85,
    },
    "STREAM-A2": {
        "edited_immediate_successor_accuracy_min": 0.90,
        "edited_downstream_step_accuracy_min": 0.75,
        "paired_final_state_flip_rate_min": 0.80,
        "marker_only_no_change_accuracy_min": 0.90,
    },
}

PANEL_SPEND_CEILING_USD = 22.0
PANEL_TARGET_MAX_SPEND_USD = 16.0
PANEL_CREDIT_RESERVE_USD = 6.0
FORECAST_CONTINGENCY_MULTIPLIER = 1.25
M1_FORWARD_COUNT_PER_CHECKPOINT = 1
MAX_TRACK_A_FORWARD_COUNT_PER_CHECKPOINT = 232
MAX_FRESH_OPERATOR_FORWARD_COUNT = 240

NORMALIZED_OPERATOR_DEPTHS = (1 / 3, 2 / 3, 35 / 36)


class PanelFreezeError(ValueError):
    """Raised when a prospective panel surface cannot be frozen exactly."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derive_operator_layers(number_of_layers: int) -> list[int]:
    if number_of_layers < 4:
        raise PanelFreezeError("operator layer derivation requires at least four layers")
    rows = [int(depth * number_of_layers + 0.5) for depth in NORMALIZED_OPERATOR_DEPTHS]
    rows = [max(1, min(number_of_layers - 1, row)) for row in rows]
    if len(set(rows)) != len(rows):
        raise PanelFreezeError("normalized operator layers collapsed")
    return rows


def execution_identity(checkpoint_key: str) -> str:
    if checkpoint_key not in CHECKPOINTS:
        raise PanelFreezeError(f"unknown checkpoint: {checkpoint_key}")
    return str(uuid.uuid5(uuid.UUID(PANEL_BINDING_ID), f"checkpoint-transfer:{checkpoint_key}"))


def model_volume_name(checkpoint_key: str) -> str:
    spec = CHECKPOINTS[checkpoint_key]
    short = str(spec["revision"])[:7]
    return f"gate13-panel-{checkpoint_key.replace('_', '-')}-{short}-model"


def result_volume_name(checkpoint_key: str) -> str:
    return (
        f"gate13-panel-{checkpoint_key.replace('_', '-')}-"
        f"{execution_identity(checkpoint_key)[:8]}-results"
    )


def qualification_contract() -> dict[str, Any]:
    ledger = compile_qualification_ledgers(SELECTED_INSTRUMENT)
    if ledger["sha256"] != BASELINE_QUALIFICATION_LEDGER_SHA256:
        raise PanelFreezeError("existing qualification ledger SHA drift")
    if ledger["forward_counts"]["maximum_conditional_total"] != 232:
        raise PanelFreezeError("existing qualification forward count drift")
    prompt = prompt_contract_payload(SELECTED_INSTRUMENT)
    partition = validate_codebook_partition()
    return {
        "selected_instrument": SELECTED_INSTRUMENT,
        "qualification_ledger_sha256": ledger["sha256"],
        "forward_counts": ledger["forward_counts"],
        "prompt_contract_sha256": prompt["sha256"],
        "qualification_codebook_bank_sha256": partition["banks"]["qualification"]["sha256"],
        "track_b_half_1_codebook_bank_sha256": partition["banks"]["track_b_half_1"]["sha256"],
        "track_b_half_2_codebook_bank_sha256": partition["banks"]["track_b_half_2"]["sha256"],
    }


def _tensor_int64_sha(values: list[int]) -> str:
    return hashlib.sha256(b"".join(struct.pack("<q", int(value)) for value in values)).hexdigest()


def _first_transfer_prompt() -> tuple[str, tuple[str, str]]:
    ledger = compile_qualification_ledgers(SELECTED_INSTRUMENT)
    row = ledger["teacher_forced"][0]
    codebooks = {row.codebook_id: row for row in codebook_bank("qualification")}
    codebook = codebooks[str(row["codebook_id"])]
    prompt = render_step_prompt(
        variant_id=SELECTED_INSTRUMENT,
        surface="STREAM-A0",
        codebook=codebook,
        current_state=int(row["current_state"]),
        action=int(row["action"]),
    )
    return prompt, codebook.state_labels


def score_slot_record(tokenizer: Any, checkpoint_key: str) -> dict[str, Any]:
    prompt, candidates = _first_transfer_prompt()
    messages = [{"role": "user", "content": prompt}]
    kwargs = {
        "add_generation_prompt": True,
        "enable_thinking": False,
        "preserve_thinking": False,
    }
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True, **kwargs)
    input_ids = [int(value) for value in encoded["input_ids"]]
    attention_mask = [int(value) for value in encoded["attention_mask"]]
    if len(input_ids) != len(attention_mask) or set(attention_mask) != {1}:
        raise PanelFreezeError("score-slot attention mask is not the exact all-visible prompt mask")
    candidate_rows = []
    for label in candidates:
        ids = [int(value) for value in tokenizer.encode(label, add_special_tokens=False)]
        if len(ids) != 1 or ids[0] in tokenizer.all_special_ids:
            raise PanelFreezeError(f"transfer candidate is not one non-special token: {label!r}")
        appended = tokenizer.encode(rendered + label, add_special_tokens=False)
        if [int(value) for value in appended] != input_ids + ids:
            raise PanelFreezeError("candidate is not the direct first semantic assistant token")
        candidate_rows.append({"label": label, "token_id": ids[0]})
    active_think = rendered.rfind("<think>") > rendered.rfind("</think>")
    forbidden_placeholders = (
        "<|image_pad|>",
        "<|video_pad|>",
        "<|vision_start|>",
        "<tool_call>",
    )
    if active_think or any(value in rendered for value in forbidden_placeholders):
        raise PanelFreezeError("score slot is preceded by an active reasoning/tool/vision channel")
    if not rendered.endswith("<think>\n\n</think>\n\n"):
        raise PanelFreezeError("official non-thinking assistant boundary changed")
    return {
        "checkpoint": checkpoint_key,
        "canonical_message_content_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "official_chat_template_sha256": hashlib.sha256(
            str(tokenizer.chat_template).encode("utf-8")
        ).hexdigest(),
        "rendered_prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "rendered_final_token_suffix": input_ids[-12:],
        "assistant_generation_prompt_suffix": rendered[-80:],
        "thinking_arguments": {
            "enable_thinking": False,
            "preserve_thinking": False,
            "reasoning_effort": "NOT_APPLICABLE_WHEN_ENABLE_THINKING_FALSE",
        },
        "input_ids_sha256": _tensor_int64_sha(input_ids),
        "attention_mask_sha256": _tensor_int64_sha(attention_mask),
        "input_length": len(input_ids),
        "score_tensor_index": len(input_ids) - 1,
        "semantic_answer_token_index": len(input_ids),
        "candidate_labels": candidate_rows,
        "candidate_labels_one_non_special_token": True,
        "candidate_append_is_direct_assistant_answer": True,
        "empty_closed_think_protocol_before_slot": True,
        "active_think_prefix_before_score_slot": False,
        "reasoning_channel_token_at_score_slot": False,
        "forced_assistant_prose_before_score_slot": False,
        "vision_or_video_placeholder": False,
        "tool_call_prefix": False,
        "status": "PASS",
    }


def validate_provider_metadata(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_repo = {str(row["repo_id"]): dict(row) for row in payload.get("checkpoints", [])}
    rows = []
    forbidden_names = ("fp8", "awq", "gptq", "gguf", "int8", "int4")
    for key, spec in CHECKPOINTS.items():
        row = by_repo.get(str(spec["repo_id"]))
        if row is None:
            raise PanelFreezeError(f"repository metadata unavailable: {spec['repo_id']}")
        if row.get("resolved_revision") != spec["revision"]:
            raise PanelFreezeError(f"immutable revision drift: {spec['repo_id']}")
        if row.get("model_class") != spec["model_class"]:
            raise PanelFreezeError(f"model class drift: {spec['repo_id']}")
        if row.get("license") != "apache-2.0":
            raise PanelFreezeError(f"license mismatch: {spec['repo_id']}")
        if int(row.get("number_of_layers", 0)) != int(spec["layers"]):
            raise PanelFreezeError(f"layer count drift: {spec['repo_id']}")
        files = row.get("file_inventory") or []
        if int(row.get("file_count", 0)) != len(files) or not files:
            raise PanelFreezeError(f"incomplete file inventory: {spec['repo_id']}")
        weight_files = [item for item in files if str(item["path"]).endswith(".safetensors")]
        if not weight_files or any(not item.get("lfs_sha256") for item in weight_files):
            raise PanelFreezeError(f"weight shard identity missing: {spec['repo_id']}")
        lowered = " ".join(str(item["path"]).lower() for item in files)
        if any(name in lowered for name in forbidden_names):
            raise PanelFreezeError(f"quantized or substituted artifact present: {spec['repo_id']}")
        identity_rows = [
            {
                "path": item["path"],
                "bytes": item["bytes"],
                "content_identity": item.get("lfs_sha256") or item.get("git_blob_id"),
            }
            for item in files
        ]
        row["model_directory_identity_sha256"] = sha256_json(identity_rows)
        rows.append({"checkpoint_key": key, **row})
    return rows


def image_definition_payload(panel: str, transformers_wheel_sha256: str | None = None) -> dict[str, Any]:
    if panel not in {"Panel S", "Panel G"}:
        raise PanelFreezeError(f"unknown runtime panel: {panel}")
    payload = {
        "schema_version": "gate13_checkpoint_panel_image_definition_v1",
        "panel": panel,
        "base_image": BASE_IMAGE,
        "base_image_manifest_digest": BASE_IMAGE_MANIFEST_DIGEST,
        "base_image_linux_amd64_digest": BASE_IMAGE_LINUX_AMD64_DIGEST,
        "python": "3.11.2",
        "torch": "2.7.1+cu126",
        "cuda_runtime": "12.6",
        "tokenizers": "0.22.2",
        "safetensors": "0.8.0",
        "accelerate": "1.14.0",
        "automatic_retries": 0,
        "maximum_active_gpu_containers": 1,
    }
    if panel == "Panel S":
        payload.update({"transformers": "5.15.0", "gpu": "L40S"})
    else:
        if not transformers_wheel_sha256:
            raise PanelFreezeError("Panel G Transformers wheel SHA is required")
        payload.update(
            {
                "transformers": TRANSFORMERS_RELEASE,
                "transformers_commit": TRANSFORMERS_COMMIT,
                "transformers_source": TRANSFORMERS_SOURCE,
                "transformers_wheel_sha256": transformers_wheel_sha256,
                "torchvision": "0.22.1+cu126",
                "Pillow": "12.1.1",
                "primary_gpu": "A100-80GB",
                "conditional_fallback_gpu": "H200",
            }
        )
    return payload


def image_definition_sha256(panel: str, transformers_wheel_sha256: str | None = None) -> str:
    return sha256_json(image_definition_payload(panel, transformers_wheel_sha256))
