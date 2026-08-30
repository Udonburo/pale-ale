"""Compile and analyze the frozen Review 2.1 Track C campaign.

This module has no Modal or model-loading capability.  It freezes the complete
twenty-block map/behavior surface, validates the deterministic P/Q renderer,
implements exact-resume bookkeeping, and applies the already-frozen Review 2.1
map and final-analysis functions.  Scientific prompts come only from the
selected ``natural_rule_v1`` stepwise compiler.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from analysis.gate13_causal_return.review2_1.track_c_review2_1_validator import (
    AnalysisTerminal,
    BEHAVIOR_EPISODES_PER_BLOCK,
    BROKEN_MAP_NODE,
    CHAT_TEMPLATE_SHA256,
    DEPTH_LEVELS,
    EXACT_MAP_NODES,
    EXECUTION_ORDER_SEED,
    FROZEN_LAYERS,
    FROZEN_TEMPLATE,
    MAP_HALVES,
    MAP_SAMPLES_PER_NODE_PER_HALF,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    RUNTIME_IMAGE_DEFINITION_SHA256,
    SCIENTIFIC_PERMUTATIONS,
    SCIENTIFIC_PERMUTATION_SEED,
    TOKENIZER_JSON_SHA256,
    amplitude_representation_observable,
    block_behavioral_outcome,
    build_analysis_geometry,
    canonical_sha256,
    crossfit_return_energy,
    derived_schedule_seed,
    evaluate_map_stage_predictor_gates,
    generate_stratified_permutation_schedule,
    map_derived_competence,
    run_primary_pipeline,
    validate_frozen_campaign_manifest,
    validate_path_surface_ledger,
)
from tools.gate13_causal_return.stepwise.compiler import (
    OPAQUE_LABELS,
    Codebook,
    render_step_prompt,
    transition,
)
from tools.gate13_causal_return.stepwise.operator_qualification import (
    BROKEN_SENSITIVITY_ABSOLUTE_MARGIN,
    BROKEN_SENSITIVITY_MULTIPLIER,
    EDGE_CONDITION_CEILING,
    EXACT_PATH_P,
    EXACT_PATH_Q,
    FRAME_RANK,
    SPLIT_HALF_SINGULAR_FLOOR_MAX,
    build_half_packets,
    estimate_frame,
    qualify_layer,
)


SCHEMA_VERSION = "gate13_track_c_frozen_campaign_v1"
CAMPAIGN_ROOT_SEED = 20_260_826_05
CODEBOOK_ROOT_SEED = 20_260_826_06
MAP_HALF_1_ROOT_SEED = 20_260_826_07
MAP_HALF_2_ROOT_SEED = 20_260_826_08
BEHAVIOR_ROOT_SEED = 20_260_826_09
EXECUTION_ID_NAMESPACE = "gate13-track-c-review2-1"
MODEL_INVENTORY_IDENTITY = "70abf71fc4b9c8b25c061043023e3d04528585c4d554ad7538ada2145c2c9e4a"
MODEL_FILE_COUNT = 29
MODEL_TOTAL_BYTES = 55_586_107_940
GPU_TYPE = "A100-80GB"
DTYPE = "bfloat16"
QUANTIZATION = False
SCORE_POSITION = "FIRST_SEMANTIC_ASSISTANT_ANSWER_TOKEN"
# Review 2.1 inherits the Panel G deterministic common literal bank.  The
# development-only labels at indices 0..15 are excluded because one of them is
# not a single token in the Panel G tokenizers.  Codebooks are fresh mappings;
# literal reuse across independent blocks is allowed by the frozen validator.
TRACK_C_LABELS = OPAQUE_LABELS[16:80]
CHAT_ASSISTANT_PREFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
RENDERED_USER_PREFIX = "<|im_start|>user\n"
RENDERED_USER_SUFFIX = "<|im_end|>\n"
MAP_NODE_VALUES = {
    "phase0_state0": (0, 0, False),
    "phase0_state1": (0, 1, False),
    "phase1_state0": (1, 0, False),
    "phase1_state1": (1, 1, False),
    "phase1_state1_broken": (1, 1, True),
}


class TrackCCampaignError(ValueError):
    """Fail-closed error for frozen campaign construction or execution data."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(dict(value)), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TrackCCampaignError(f"JSON root must be an object: {path}")
    return value


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


@dataclass(frozen=True)
class ExactTokenizer:
    tokenizer: Any
    special_ids: frozenset[int]

    @classmethod
    def from_file(cls, path: Path) -> "ExactTokenizer":
        try:
            from tokenizers import Tokenizer
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency preflight
            raise TrackCCampaignError("tokenizers is required to compile the frozen surface") from exc
        tokenizer = Tokenizer.from_file(str(path))
        raw = json.loads(path.read_text(encoding="utf-8"))
        special_ids = frozenset(
            int(row["id"])
            for row in raw.get("added_tokens", [])
            if bool(row.get("special"))
        )
        return cls(tokenizer=tokenizer, special_ids=special_ids)

    def encode(self, text: str) -> list[int]:
        return [int(value) for value in self.tokenizer.encode(text).ids]

    def one_token(self, label: str) -> int:
        ids = self.encode(label)
        if len(ids) != 1 or ids[0] in self.special_ids:
            raise TrackCCampaignError(f"opaque label is not one non-special token: {label!r} -> {ids}")
        if self.tokenizer.decode(ids, skip_special_tokens=False) != label:
            raise TrackCCampaignError(f"opaque label does not round-trip exactly: {label!r}")
        return ids[0]


def official_text_chat_render(prompt: str) -> str:
    """Exact frozen Qwen3.6 text-only non-thinking chat serialization."""

    return f"{RENDERED_USER_PREFIX}{prompt.strip()}{RENDERED_USER_SUFFIX}{CHAT_ASSISTANT_PREFIX}"


def _block_codebook(index: int) -> Codebook:
    labels = list(TRACK_C_LABELS)
    random.Random(stable_seed(CODEBOOK_ROOT_SEED, index)).shuffle(labels)
    labels = labels[:4]
    return Codebook(
        codebook_id=f"track-c-cb-{index:02d}",
        bank="track_c_frozen_twenty_block_bank",
        state_labels=(labels[0], labels[1]),
        action_labels=(labels[2], labels[3]),
    )


def _prompt_binding(prompt: str, tokenizer: ExactTokenizer) -> dict[str, Any]:
    rendered = official_text_chat_render(prompt)
    return {
        "canonical_message_sha256": sha256_text(prompt),
        "rendered_prompt_sha256": sha256_text(rendered),
        "input_ids_sha256": canonical_sha256(tokenizer.encode(rendered)),
        "input_token_count": len(tokenizer.encode(rendered)),
    }


def _interleaved_order(
    rows_by_block: Mapping[str, Sequence[str]],
    *,
    seed: int,
    shuffle_within_block: bool = True,
) -> list[str]:
    queues: dict[str, list[str]] = {}
    for block_id, values in rows_by_block.items():
        queue = list(values)
        if shuffle_within_block:
            random.Random(stable_seed(seed, block_id)).shuffle(queue)
        else:
            queue.reverse()
        queues[block_id] = queue
    rng = random.Random(seed)
    result: list[str] = []
    previous: str | None = None
    while any(queues.values()):
        active = [block_id for block_id, values in queues.items() if values]
        rng.shuffle(active)
        if len(active) > 1 and active[0] == previous:
            active[0], active[1] = active[1], active[0]
        for block_id in active:
            if block_id == previous and len(active) > 1:
                raise TrackCCampaignError("interleaving algorithm produced adjacent block reuse")
            result.append(queues[block_id].pop())
            previous = block_id
    return result


def _surface_operation(action_label: str) -> str:
    return f"Apply one frozen state update using action {action_label.strip()}."


def _surface_tau() -> str:
    return "Advance the protocol phase without changing the visible state."


def _surface_path(
    *,
    tokenizer: ExactTokenizer,
    block_id: str,
    episode_id: str,
    codebook: Codebook,
    operations: Sequence[str],
) -> dict[str, Any]:
    context = (
        f"Frozen natural_rule_v1 path surface. Block {block_id}; episode {episode_id}. "
        f"States {codebook.state_labels[0].strip()}, {codebook.state_labels[1].strip()}; "
        f"actions {codebook.action_labels[0].strip()}, {codebook.action_labels[1].strip()}."
    )
    message_contents = [context, *operations]
    rendered_segments = [
        f"{RENDERED_USER_PREFIX}{content}{RENDERED_USER_SUFFIX}" for content in message_contents
    ]
    rendered = "".join(rendered_segments) + CHAT_ASSISTANT_PREFIX
    token_ids = tokenizer.encode(rendered)
    operation_token_ids = [tokenizer.encode(content) for content in operations]
    wrapper_prefix = tokenizer.encode(RENDERED_USER_PREFIX)
    wrapper_suffix = tokenizer.encode(RENDERED_USER_SUFFIX)
    non_operation = [
        *wrapper_prefix,
        *tokenizer.encode(context),
        *wrapper_suffix,
    ]
    for _operation in operations:
        non_operation.extend(wrapper_prefix)
        non_operation.extend(wrapper_suffix)
    non_operation.extend(tokenizer.encode(CHAT_ASSISTANT_PREFIX))
    return {
        "canonical_template_id": FROZEN_TEMPLATE,
        "canonical_template_sha256": CHAT_TEMPLATE_SHA256,
        "message_count": len(message_contents),
        "operation_count": len(operations),
        "operation_token_ids": operation_token_ids,
        "codebook_token_ids": [
            tokenizer.one_token(label)
            for label in (*codebook.state_labels, *codebook.action_labels)
        ],
        "special_token_count": sum(token in tokenizer.special_ids for token in token_ids),
        "answer_prefix_utf8_hex": CHAT_ASSISTANT_PREFIX.encode("utf-8").hex(),
        "score_slot": SCORE_POSITION,
        "rendered_utf8_hex": rendered.encode("utf-8").hex(),
        "token_ids": token_ids,
        "non_operation_token_ids": non_operation,
    }


def _compile_path_pair(
    *,
    tokenizer: ExactTokenizer,
    block_id: str,
    episode_id: str,
    codebook: Codebook,
    actions: Sequence[int],
) -> dict[str, Any]:
    steps = [_surface_operation(codebook.action(action)) for action in actions]
    tau = _surface_tau()
    return {
        "pair_id": f"tc-pair-{block_id}-{episode_id}",
        "block_id": block_id,
        "episode_id": episode_id,
        "path_p": _surface_path(
            tokenizer=tokenizer,
            block_id=block_id,
            episode_id=episode_id,
            codebook=codebook,
            operations=[*steps, tau],
        ),
        "path_q": _surface_path(
            tokenizer=tokenizer,
            block_id=block_id,
            episode_id=episode_id,
            codebook=codebook,
            operations=[tau, *steps],
        ),
    }


def _behavior_prompt_variants(
    *,
    tokenizer: ExactTokenizer,
    codebook: Codebook,
    phase: int,
    action: int,
    demo_seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for current in (0, 1):
        prompt = render_step_prompt(
            variant_id=FROZEN_TEMPLATE,
            surface="TRACK-B",
            codebook=codebook,
            current_state=current,
            action=action,
            demo_seed=demo_seed,
            template_flavor=0,
            phase_index=phase,
            broken_context=False,
        )
        result[str(current)] = _prompt_binding(prompt, tokenizer)
    return result


def codebook_from_block(block: Mapping[str, Any]) -> Codebook:
    raw = block["codebook"]
    return Codebook(
        codebook_id=str(raw["codebook_id"]),
        bank=str(raw["bank"]),
        state_labels=tuple(map(str, raw["state_labels"])),
        action_labels=tuple(map(str, raw["action_labels"])),
    )


def render_map_case(case: Mapping[str, Any], block: Mapping[str, Any]) -> str:
    return render_step_prompt(
        variant_id=FROZEN_TEMPLATE,
        surface="TRACK-B",
        codebook=codebook_from_block(block),
        current_state=int(case["current_state"]),
        action=int(case["action"]),
        demo_seed=int(case["episode_seed"]),
        template_flavor=0,
        phase_index=int(case["phase_index"]),
        broken_context=bool(case["broken_context"]),
    )


def render_behavior_case(
    case: Mapping[str, Any],
    block: Mapping[str, Any],
    *,
    current_state: int,
) -> str:
    if current_state not in (0, 1):
        raise TrackCCampaignError("self-fed state must be binary")
    return render_step_prompt(
        variant_id=FROZEN_TEMPLATE,
        surface="TRACK-B",
        codebook=codebook_from_block(block),
        current_state=current_state,
        action=int(case["action"]),
        demo_seed=int(case["episode_seed"]),
        template_flavor=0,
        phase_index=int(case["phase_index"]),
        broken_context=False,
    )


def compile_campaign(tokenizer_path: Path) -> dict[str, dict[str, Any]]:
    tokenizer = ExactTokenizer.from_file(tokenizer_path)
    if sha256_file(tokenizer_path) != TOKENIZER_JSON_SHA256:
        raise TrackCCampaignError("exact tokenizer.json hash mismatch")
    label_bindings = [
        {"label": label, "token_id": tokenizer.one_token(label)} for label in TRACK_C_LABELS
    ]
    blocks: list[dict[str, Any]] = []
    map_cases: list[dict[str, Any]] = []
    behavior_cases: list[dict[str, Any]] = []
    path_pairs: list[dict[str, Any]] = []
    map_by_block: dict[str, list[str]] = {}
    behavior_by_block: dict[str, list[str]] = {}
    depths = [level for level in DEPTH_LEVELS for _ in range(5)]
    for block_index, depth in enumerate(depths):
        block_id = f"tc-b{block_index:02d}-d{depth}"
        codebook = _block_codebook(block_index)
        codebook_payload = codebook.as_json()
        map_ids: list[str] = []
        behavior_ids: list[str] = []
        demonstration_ids: list[str] = []
        map_roots = {
            "half_1": stable_seed(MAP_HALF_1_ROOT_SEED, block_id),
            "half_2": stable_seed(MAP_HALF_2_ROOT_SEED, block_id),
        }
        behavior_root = stable_seed(BEHAVIOR_ROOT_SEED, block_id)
        for half_id in MAP_HALVES:
            for sample_index in range(MAP_SAMPLES_PER_NODE_PER_HALF):
                sample_seed = map_roots[half_id] + sample_index
                sample_id = f"{block_id}-{half_id}-s{sample_index:02d}"
                demo_id = f"tc-demo-map-{sample_id}"
                demonstration_ids.append(demo_id)
                action = int(stable_seed(sample_seed, "map-action") % 2)
                for node_id, (phase, current_state, broken) in MAP_NODE_VALUES.items():
                    case_id = f"tc-m-{sample_id}-{node_id}"
                    prompt = render_step_prompt(
                        variant_id=FROZEN_TEMPLATE,
                        surface="TRACK-B",
                        codebook=codebook,
                        current_state=current_state,
                        action=action,
                        demo_seed=sample_seed,
                        template_flavor=0,
                        phase_index=phase,
                        broken_context=broken,
                    )
                    map_cases.append(
                        {
                            "case_id": case_id,
                            "block_id": block_id,
                            "half_id": half_id,
                            "sample_id": sample_id,
                            "node_id": node_id,
                            "episode_seed": sample_seed,
                            "demonstration_id": demo_id,
                            "codebook_id": codebook.codebook_id,
                            "phase_index": phase,
                            "current_state": current_state,
                            "action": action,
                            "target_state": transition(current_state, action),
                            "broken_context": broken,
                            "candidate_labels": list(codebook.state_labels),
                            "prompt_binding": _prompt_binding(prompt, tokenizer),
                        }
                    )
                    map_ids.append(case_id)
        for episode_index in range(BEHAVIOR_EPISODES_PER_BLOCK):
            episode_seed = behavior_root + episode_index
            episode_id = f"{block_id}-e{episode_index:02d}"
            demo_id = f"tc-demo-behavior-{episode_id}"
            demonstration_ids.append(demo_id)
            rng = random.Random(stable_seed(episode_seed, "behavior"))
            initial_state = rng.randrange(2)
            actions = [rng.randrange(2) for _ in range(depth)]
            final_oracle = initial_state
            for action in actions:
                final_oracle = transition(final_oracle, action)
            for path_id, phase in (("P", 0), ("Q", 1)):
                oracle_state = initial_state
                for step_index, action in enumerate(actions):
                    case_id = f"tc-e-{episode_id}-{path_id}-s{step_index:02d}"
                    behavior_cases.append(
                        {
                            "case_id": case_id,
                            "block_id": block_id,
                            "episode_id": episode_id,
                            "path_id": path_id,
                            "call_index": step_index,
                            "call_role": "SELF_FED_TRANSITION",
                            "episode_seed": episode_seed,
                            "demonstration_id": demo_id,
                            "codebook_id": codebook.codebook_id,
                            "phase_index": phase,
                            "action": action,
                            "oracle_state_before": oracle_state,
                            "oracle_state_after": transition(oracle_state, action),
                            "candidate_labels": list(codebook.state_labels),
                            "prompt_variants": _behavior_prompt_variants(
                                tokenizer=tokenizer,
                                codebook=codebook,
                                phase=phase,
                                action=action,
                                demo_seed=episode_seed,
                            ),
                        }
                    )
                    behavior_ids.append(case_id)
                    oracle_state = transition(oracle_state, action)
                probe_id = f"tc-e-{episode_id}-{path_id}-probe"
                behavior_cases.append(
                    {
                        "case_id": probe_id,
                        "block_id": block_id,
                        "episode_id": episode_id,
                        "path_id": path_id,
                        "call_index": depth,
                        "call_role": "COMMON_ENDPOINT_PROBE",
                        "episode_seed": episode_seed,
                        "demonstration_id": demo_id,
                        "codebook_id": codebook.codebook_id,
                        "phase_index": 1,
                        "action": 0,
                        "oracle_state_before": final_oracle,
                        "oracle_state_after": final_oracle,
                        "target_state": final_oracle,
                        "candidate_labels": list(codebook.state_labels),
                        "prompt_variants": _behavior_prompt_variants(
                            tokenizer=tokenizer,
                            codebook=codebook,
                            phase=1,
                            action=0,
                            demo_seed=episode_seed,
                        ),
                    }
                )
                behavior_ids.append(probe_id)
            path_pairs.append(
                _compile_path_pair(
                    tokenizer=tokenizer,
                    block_id=block_id,
                    episode_id=episode_id,
                    codebook=codebook,
                    actions=actions,
                )
            )
        seeds = {
            "codebook_seed": stable_seed(CODEBOOK_ROOT_SEED, block_index),
            "map_half_1_seed_root": map_roots["half_1"],
            "map_half_2_seed_root": map_roots["half_2"],
            "behavior_seed_root": behavior_root,
            "block_order_seed": stable_seed(EXECUTION_ORDER_SEED, block_id),
        }
        block = {
            "block_id": block_id,
            "rollout_depth": depth,
            "template": FROZEN_TEMPLATE,
            "codebook_id": codebook.codebook_id,
            "codebook": codebook_payload,
            "codebook_sha256": canonical_sha256(codebook_payload),
            "demonstration_ids": demonstration_ids,
            "seeds": seeds,
            "map_half_ids": list(MAP_HALVES),
            "map_case_ids": map_ids,
            "behavior_case_ids": behavior_ids,
        }
        blocks.append(block)
        map_by_block[block_id] = map_ids
        behavior_by_block[block_id] = behavior_ids
    stage_m_order = _interleaved_order(map_by_block, seed=EXECUTION_ORDER_SEED)
    stage_e_order = _interleaved_order(
        behavior_by_block,
        seed=EXECUTION_ORDER_SEED + 1,
        shuffle_within_block=False,
    )
    plan = {
        "schema_version": "gate13_track_c_campaign_plan_v1",
        "campaign_schema": SCHEMA_VERSION,
        "scientific_authority": "analysis/gate13_causal_return/review2_1",
        "review2_1_accepted_by_human": True,
        "track_c_authorized_under_frozen_conditional_protocol": True,
        "model": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION},
        "instrument": FROZEN_TEMPLATE,
        "external_algebra": "BINARY_XOR_WITH_PHASE_NATURALITY_SQUARE",
        "behavior_path_semantics": {
            "path_p": "SELF_FED_J_ACTIONS_IN_PHASE_0_THEN_EXTERNAL_TAU_TO_PHASE_1",
            "path_q": "EXTERNAL_TAU_TO_PHASE_1_THEN_SAME_SELF_FED_J_ACTIONS",
            "tau_changes_visible_phase_only": True,
            "full_history_excluded": True,
            "endpoint": "PHASE_1_IDENTITY_ACTION_FORCED_CHOICE_PROBE",
        },
        "stages": {
            "M": {"forwards": 4_800, "paid_forecast_usd": 20.62321418},
            "E": {"maximum_forwards": 5_760, "paid_forecast_usd": 24.59943168},
        },
        "absolute_campaign_ceiling_usd": 65.0,
        "automatic_retries": 0,
        "maximum_active_gpu_containers": 1,
        "a3": "CLOSED",
        "formal_gate13": "CLOSED",
    }
    ledger = {
        "schema_version": "gate13_track_c_complete_case_ledger_v1",
        "map_cases": map_cases,
        "behavior_cases": behavior_cases,
        "counts": {
            "map": len(map_cases),
            "behavior_maximum": len(behavior_cases),
            "path_pairs": len(path_pairs),
        },
    }
    manifest = {
        "schema_version": "gate13_track_c_campaign_manifest_v1",
        "authority": {"track_c_authorized": False, "reason": "PLANNING_MANIFEST_SEPARATE_AUTHORITY"},
        "model": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "tokenizer_repository": MODEL_REPOSITORY,
            "tokenizer_revision": MODEL_REVISION,
            "inventory_identity": MODEL_INVENTORY_IDENTITY,
            "file_count": MODEL_FILE_COUNT,
            "total_bytes": MODEL_TOTAL_BYTES,
        },
        "runtime": {
            "image_definition_sha256": RUNTIME_IMAGE_DEFINITION_SHA256,
            "chat_template_sha256": CHAT_TEMPLATE_SHA256,
            "tokenizer_json_sha256": TOKENIZER_JSON_SHA256,
            "gpu": GPU_TYPE,
            "dtype": DTYPE,
            "quantization": QUANTIZATION,
            "dependency_versions": {
                "python": "3.11.2",
                "torch": "2.7.1+cu126",
                "cuda": "12.6",
                "transformers": "5.15.1@550d7b3834670483a4df436541272c055dc364bf",
                "tokenizers": "0.22.2",
                "accelerate": "1.14.0",
                "safetensors": "0.8.0",
            },
        },
        "scoring": {
            "score_position": SCORE_POSITION,
            "correct_is_single_token": True,
            "other_is_single_token": True,
            "correct_token_id": label_bindings[0]["token_id"],
            "other_token_id": label_bindings[1]["token_id"],
            "all_label_bindings": label_bindings,
            "readout": "RAW_NEXT_TOKEN_FORCED_CHOICE_LOGITS",
            "generation": False,
            "sampling": False,
        },
        "blocks": blocks,
        "execution": {
            "order_seed": EXECUTION_ORDER_SEED,
            "order_algorithm": "SEEDED_BLOCK_INTERLEAVED_SHUFFLE_V1",
            "stage_m_order": stage_m_order,
            "stage_e_order": stage_e_order,
            "accepted_ids_may_be_duplicated_or_replaced": False,
            "exact_resume_missing_ids_only": True,
        },
        "analysis": {
            "permutation_root_seed": SCIENTIFIC_PERMUTATION_SEED,
            "permutations": SCIENTIFIC_PERMUTATIONS,
            "schedule_family_algorithm": "SHA256_ROOT_SEED_AND_ORDERED_QUALIFIED_BLOCK_IDS_V1",
        },
        "path_surface_pairs": path_pairs,
        "case_ledger_sha256": canonical_sha256(ledger),
        "campaign_plan_sha256": canonical_sha256(plan),
    }
    manifest_validation = validate_frozen_campaign_manifest(manifest)
    if manifest_validation["status"] != "PASS":
        raise TrackCCampaignError("frozen campaign manifest did not validate")
    return {"plan": plan, "manifest": manifest, "ledger": ledger, "validation": manifest_validation}


def compact_manifest_validation(validation: Mapping[str, Any]) -> dict[str, Any]:
    surface = validation["path_surface_validation"]
    return {
        "status": validation["status"],
        "campaign_manifest_sha256": validation["campaign_manifest_sha256"],
        "block_count": validation["block_count"],
        "map_case_count": validation["map_case_count"],
        "behavior_case_count": validation["behavior_case_count"],
        "stage_m_order_sha256": validation["stage_m_order_sha256"],
        "stage_e_order_sha256": validation["stage_e_order_sha256"],
        "path_surface_validation": {
            "status": surface["status"],
            "pair_count": surface["pair_count"],
            "input_ledger_sha256": surface["input_ledger_sha256"],
            "mismatch_action": surface["mismatch_action"],
        },
        "track_c_authorized": False,
    }


def write_campaign(output_dir: Path, tokenizer_path: Path) -> dict[str, Any]:
    compiled = compile_campaign(tokenizer_path)
    compact_validation = compact_manifest_validation(compiled["validation"])
    names = {
        "plan": "track_c_campaign_plan.json",
        "manifest": "track_c_campaign_manifest.json",
        "ledger": "track_c_execution_ledger.json",
        "validation": "track_c_preflight_validation.json",
    }
    hashes: dict[str, str] = {}
    for key, filename in names.items():
        path = output_dir / filename
        value = compact_validation if key == "validation" else compiled[key]
        atomic_json(path, value)
        hashes[filename] = sha256_file(path)
    return {"status": "PASS", "files_sha256": hashes, **compact_validation}


def ledger_indexes(ledger: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    map_rows = {str(row["case_id"]): dict(row) for row in ledger["map_cases"]}
    behavior_rows = {str(row["case_id"]): dict(row) for row in ledger["behavior_cases"]}
    if len(map_rows) != 4_800 or len(behavior_rows) != 5_760:
        raise TrackCCampaignError("case ledger coverage drift")
    return map_rows, behavior_rows


class AtomicCaseStore:
    """Immutable per-case records with accepted-ID exact resume semantics."""

    def __init__(self, root: Path, stage: str):
        self.root = root / stage
        self.attempt_root = self.root / "attempts"
        self.response_root = self.root / "responses"

    def _path(self, root: Path, case_id: str) -> Path:
        if not case_id or any(value in case_id for value in ("/", "\\", "..")):
            raise TrackCCampaignError("unsafe case identity")
        return root / f"{case_id}.json"

    def accepted(self, case_id: str) -> dict[str, Any] | None:
        path = self._path(self.response_root, case_id)
        return load_json(path) if path.exists() else None

    def record_attempt(self, case_id: str, binding: Mapping[str, Any]) -> None:
        path = self._path(self.attempt_root, case_id)
        payload = {"case_id": case_id, "binding_sha256": canonical_sha256(dict(binding))}
        if path.exists():
            if load_json(path) != payload:
                raise TrackCCampaignError(f"attempt binding drift: {case_id}")
            return
        atomic_json(path, payload)

    def accept(self, case_id: str, response: Mapping[str, Any]) -> None:
        path = self._path(self.response_root, case_id)
        payload = dict(response)
        if payload.get("case_id") != case_id:
            raise TrackCCampaignError("response case identity mismatch")
        if path.exists():
            if load_json(path) != payload:
                raise TrackCCampaignError(f"accepted response is immutable: {case_id}")
            return
        atomic_json(path, payload)

    def accepted_ids(self) -> set[str]:
        if not self.response_root.exists():
            return set()
        return {path.stem for path in self.response_root.glob("*.json")}


def _array(value: Any, *, context: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or not np.all(np.isfinite(result)):
        raise TrackCCampaignError(f"{context} must be a finite matrix")
    return result


def analyze_map_block(
    *,
    block: Mapping[str, Any],
    response_rows: Sequence[Mapping[str, Any]],
    activation_loader: Any,
) -> dict[str, Any]:
    block_id = str(block["block_id"])
    expected = set(map(str, block["map_case_ids"]))
    rows = [dict(row) for row in response_rows if str(row.get("case_id")) in expected]
    if len(rows) != 240 or {str(row["case_id"]) for row in rows} != expected:
        raise TrackCCampaignError(f"incomplete map packet: {block_id}")
    activations: dict[str, dict[int, dict[str, dict[str, np.ndarray]]]] = {
        half: {layer: {node: {} for node in MAP_NODE_VALUES} for layer in FROZEN_LAYERS}
        for half in MAP_HALVES
    }
    exact_margin_rows: list[dict[str, Any]] = []
    for row in rows:
        vectors = activation_loader(row)
        half = str(row["half_id"])
        node = str(row["node_id"])
        sample_id = str(row["sample_id"])
        for layer in FROZEN_LAYERS:
            node_values = activations[half][layer][node]
            if sample_id in node_values:
                raise TrackCCampaignError(f"duplicate paired activation: {half}:{node}:{sample_id}")
            node_values[sample_id] = _array([vectors[layer]], context="activation")[0]
        if node in EXACT_MAP_NODES:
            exact_margin_rows.append(row)
    competence = map_derived_competence(exact_margin_rows)
    energy_components: dict[int, list[float]] = {}
    layers: list[dict[str, Any]] = []
    block_qualified = True
    for layer in FROZEN_LAYERS:
        half_arrays = {
            half: {
                node: np.asarray(
                    [values[sample_id] for sample_id in sorted(values)],
                    dtype=np.float64,
                )
                for node, values in activations[half][layer].items()
            }
            for half in MAP_HALVES
        }
        for half in MAP_HALVES:
            sample_sets = [set(values) for values in activations[half][layer].values()]
            if any(len(values) != MAP_SAMPLES_PER_NODE_PER_HALF for values in sample_sets):
                raise TrackCCampaignError(f"map node support mismatch: {block_id}:{half}:{layer}")
            if any(values != sample_sets[0] for values in sample_sets[1:]):
                raise TrackCCampaignError(f"paired map sample identity mismatch: {block_id}:{half}:{layer}")
        validity = qualify_layer(half_arrays["half_1"], half_arrays["half_2"], layer=layer)
        native_frames = {
            half: {node: estimate_frame(values) for node, values in half_arrays[half].items()}
            for half in MAP_HALVES
        }
        native_packets = {
            half: build_half_packets(native_frames[half], half_id=half) for half in MAP_HALVES
        }
        components: list[float] = []
        for training_half, opposite_half in (("half_1", "half_2"), ("half_2", "half_1")):
            packet = native_packets[training_half]["exact_square"]
            source_record = native_frames[training_half][EXACT_PATH_P[0]]
            if packet.get("status") != "QUALIFIED" or source_record.get("status") != "QUALIFIED":
                block_qualified = False
                components.append(float("nan"))
                continue
            source_frame = source_record["frame"]
            components.append(
                crossfit_return_energy(
                    packet["raw"]["Delta_pq"],
                    half_arrays[opposite_half][EXACT_PATH_P[0]],
                    source_frame,
                )
            )
        if validity.get("status") != "PASS" or not np.all(np.isfinite(components)):
            block_qualified = False
        energy_components[int(layer)] = components
        compact_validity = {
            key: value
            for key, value in validity.items()
            if key
            in {
                "layer",
                "status",
                "reason",
                "split_half_singular_floor",
                "split_half_floor_ceiling",
                "exact_square_normalized_delta_by_half",
                "broken_square_normalized_delta_by_half",
                "broken_sensitivity_threshold",
                "reproducible",
                "broken_square_sensitive",
                "packet_disagreement",
                "half_1",
                "half_2",
            }
        }
        layers.append(
            json_ready(
                {
                    "layer": int(layer),
                    "validity": compact_validity,
                    "native_packets": native_packets,
                    "crossfit_energy_by_training_half": components,
                }
            )
        )
    representation = (
        amplitude_representation_observable(energy_components)
        if block_qualified
        else {"primary_amplitude": 0.0, "unqualified_no_imputation": True}
    )
    return {
        "block_id": block_id,
        "rollout_depth": int(block["rollout_depth"]),
        "qualified": block_qualified,
        "map_competence": competence,
        "representation": representation,
        "layers": layers,
    }


def evaluate_map_campaign(
    *,
    manifest: Mapping[str, Any],
    block_results: Sequence[Mapping[str, Any]],
    artifact_complete: bool,
) -> dict[str, Any]:
    by_id = {str(row["block_id"]): dict(row) for row in block_results}
    block_ids = [str(block["block_id"]) for block in manifest["blocks"]]
    if set(by_id) != set(block_ids):
        raise TrackCCampaignError("map analysis block coverage drift")
    ordered = [by_id[block_id] for block_id in block_ids]
    qualified = np.asarray([bool(row["qualified"]) for row in ordered], dtype=bool)
    representation = np.asarray(
        [float(row["representation"]["primary_amplitude"]) for row in ordered],
        dtype=np.float64,
    )
    competence = np.asarray(
        [float(row["map_competence"]["map_derived_competence"]) for row in ordered],
        dtype=np.float64,
    )
    return evaluate_map_stage_predictor_gates(
        block_ids=block_ids,
        rollout_depth=[int(row["rollout_depth"]) for row in ordered],
        map_competence=competence,
        representation_feature=representation,
        qualification_mask=qualified,
        split_half_valid=bool(np.all([row["qualified"] for row in ordered if row["qualified"]])),
        frame_rank_valid=True,
        conditioning_valid=True,
        exact_square_reproducibility_valid=True,
        broken_square_sensitivity_valid=True,
        path_surface_valid=validate_path_surface_ledger(manifest["path_surface_pairs"])["status"] == "PASS",
        artifact_complete=artifact_complete,
    )


def analyze_behavior_and_primary(
    *,
    manifest: Mapping[str, Any],
    map_block_results: Sequence[Mapping[str, Any]],
    behavior_responses: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    map_by_id = {str(row["block_id"]): dict(row) for row in map_block_results}
    endpoint = [row for row in behavior_responses if row.get("call_role") == "COMMON_ENDPOINT_PROBE"]
    margins: dict[tuple[str, str, str], float] = {}
    for row in endpoint:
        target = int(row["target_state"])
        logits = np.asarray(row["candidate_logits"], dtype=np.float64)
        if logits.shape != (2,) or target not in (0, 1):
            raise TrackCCampaignError("malformed endpoint score row")
        key = (str(row["block_id"]), str(row["episode_id"]), str(row["path_id"]))
        if key in margins:
            raise TrackCCampaignError("duplicate behavior endpoint")
        margins[key] = float(logits[target] - logits[1 - target])
    qualified_ids = [
        str(block["block_id"])
        for block in manifest["blocks"]
        if bool(map_by_id[str(block["block_id"])]["qualified"])
    ]
    outcomes: list[float] = []
    competence: list[float] = []
    representation: list[float] = []
    depth: list[int] = []
    block_packets: list[dict[str, Any]] = []
    for block in manifest["blocks"]:
        block_id = str(block["block_id"])
        if block_id not in qualified_ids:
            continue
        p_values: list[float] = []
        q_values: list[float] = []
        for episode_index in range(BEHAVIOR_EPISODES_PER_BLOCK):
            episode_id = f"{block_id}-e{episode_index:02d}"
            try:
                p_values.append(margins[(block_id, episode_id, "P")])
                q_values.append(margins[(block_id, episode_id, "Q")])
            except KeyError as exc:
                raise TrackCCampaignError(f"incomplete behavior packet: {block_id}") from exc
        y_value = block_behavioral_outcome(p_values, q_values)
        row = map_by_id[block_id]
        outcomes.append(y_value)
        competence.append(float(row["map_competence"]["map_derived_competence"]))
        representation.append(float(row["representation"]["primary_amplitude"]))
        depth.append(int(block["rollout_depth"]))
        block_packets.append(
            {
                "block_id": block_id,
                "rollout_depth": int(block["rollout_depth"]),
                "Y_b": y_value,
                "C_b_M": competence[-1],
                "R_b": representation[-1],
                "path_p_margins": p_values,
                "path_q_margins": q_values,
            }
        )
    geometry = build_analysis_geometry(depth, competence, representation)
    schedule_seed = derived_schedule_seed(
        root_seed=SCIENTIFIC_PERMUTATION_SEED,
        qualified_block_ids=qualified_ids,
    )
    schedule = generate_stratified_permutation_schedule(
        depth,
        permutations=SCIENTIFIC_PERMUTATIONS,
        seed=schedule_seed,
    )
    primary = run_primary_pipeline(outcome=outcomes, geometry=geometry, schedule=schedule)
    return {
        "schema_version": "gate13_track_c_primary_result_v1",
        "qualified_block_ids": qualified_ids,
        "block_packets": block_packets,
        "permutation_schedule_seed": schedule_seed,
        "permutation_schedule_sha256": canonical_sha256(schedule.tolist()),
        "primary": primary,
    }


def synthetic_preflight(manifest: Mapping[str, Any], ledger: Mapping[str, Any]) -> dict[str, Any]:
    validation = validate_frozen_campaign_manifest(manifest)
    map_index, behavior_index = ledger_indexes(ledger)
    if set(manifest["execution"]["stage_m_order"]) != set(map_index):
        raise TrackCCampaignError("Stage M order/ledger mismatch")
    if set(manifest["execution"]["stage_e_order"]) != set(behavior_index):
        raise TrackCCampaignError("Stage E order/ledger mismatch")
    depth = np.asarray([2] * 5 + [4] * 5 + [6] * 5 + [8] * 5, dtype=np.int64)
    competence = np.linspace(-1.1, 1.3, 20) + 0.03 * np.sin(np.arange(20))
    representation = np.cos(np.arange(20) * 0.71) + 0.08 * competence
    geometry = build_analysis_geometry(depth, competence, representation)
    synthetic_y = 0.15 * representation + np.sin(np.arange(20) * 0.37)
    seed = derived_schedule_seed(
        root_seed=SCIENTIFIC_PERMUTATION_SEED,
        qualified_block_ids=[str(block["block_id"]) for block in manifest["blocks"]],
    )
    schedule = generate_stratified_permutation_schedule(depth, permutations=999, seed=seed)
    primary = run_primary_pipeline(outcome=synthetic_y, geometry=geometry, schedule=schedule)
    return {
        "status": "PASS",
        "manifest_validation": compact_manifest_validation(validation),
        "map_case_count": len(map_index),
        "behavior_case_count": len(behavior_index),
        "model_free_pipeline_terminal": primary["terminal_state"],
        "synthetic_only": True,
        "scientific_model_forward_count": 0,
    }


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = write_campaign(args.output_dir, args.tokenizer)
    manifest = load_json(args.output_dir / "track_c_campaign_manifest.json")
    ledger = load_json(args.output_dir / "track_c_execution_ledger.json")
    preflight = synthetic_preflight(manifest, ledger)
    atomic_json(args.output_dir / "track_c_model_free_preflight.json", preflight)
    print(json.dumps({"compile": result, "preflight": preflight}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
