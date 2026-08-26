"""Model-free validator and final null audit for Track C Review 2.1.

The module has no model, tokenizer, GPU, or Modal capability.  It validates
prospective ledgers, implements the exact block-level OLS/LOBO/permutation
pipeline, and runs synthetic calibration only.  It never accepts a Track C
outcome file or activation directory.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "gate13_track_c_review2_1_validator_v1"
MODEL_REPOSITORY = "Qwen/Qwen3.6-27B"
MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
TOKENIZER_REPOSITORY = MODEL_REPOSITORY
TOKENIZER_REVISION = MODEL_REVISION
FROZEN_TEMPLATE = "natural_rule_v1"
FROZEN_LAYERS = (21, 43, 62)
FRAME_RANK = 4
DEPTH_LEVELS = (2, 4, 6, 8)
PLANNED_BLOCKS = 20
PLANNED_BLOCKS_PER_DEPTH = 5
MINIMUM_QUALIFIED_BLOCKS = 16
MINIMUM_QUALIFIED_PER_DEPTH = 4
MAP_HALVES = ("half_1", "half_2")
MAP_SAMPLES_PER_NODE_PER_HALF = 24
EXACT_MAP_NODES = (
    "phase0_state0",
    "phase0_state1",
    "phase1_state0",
    "phase1_state1",
)
BROKEN_MAP_NODE = "phase1_state1_broken"
BEHAVIOR_EPISODES_PER_BLOCK = 24

RUNTIME_IMAGE_DEFINITION_SHA256 = (
    "61e2bb1ecf850a7e106799f1aa4c0b5447dbc9daf5bcb63175f0d690cd469d31"
)
CHAT_TEMPLATE_SHA256 = (
    "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259"
)
TOKENIZER_JSON_SHA256 = (
    "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
)

SCIENTIFIC_PERMUTATIONS = 99_999
SCIENTIFIC_PERMUTATION_SEED = 13_602_027
DEVELOPMENT_SEED = 20_260_826_01
CALIBRATION_PERMUTATION_SEED = 20_260_826_02
FINAL_AUDIT_SEED = 20_260_826_03
EXECUTION_ORDER_SEED = 20_260_826_04
FINAL_AUDIT_DATASETS_PER_SCENARIO = 2_000
FINAL_AUDIT_PERMUTATIONS = 999
FINAL_AUDIT_ALPHA = 0.05
FINAL_AUDIT_FPR_INTERVAL = (0.035, 0.065)

NUMERICAL_RELATIVE_FLOOR = 1.0e-12
DESIGN_CONDITION_CEILING = 1.0e6
LEVERAGE_AVERAGE_MULTIPLIER = 3.0
LEVERAGE_ABSOLUTE_CEILING = 0.80
MINIMUM_PRESS_DENOMINATOR = 1.0 - LEVERAGE_ABSOLUTE_CEILING

EXISTING_B_OPERATOR_COST_USD = 1.02497632
EXISTING_B_OPERATOR_FORWARD_COUNT = 240
EXISTING_ACQUISITION_AND_PREFLIGHT_COST_USD = 0.12368778
COST_CONTINGENCY_MULTIPLIER = 1.25
FUTURE_MODAL_BUDGET_CEILING_USD = 65.0

REVIEW2_COMMIT = "4818d0cd313f09e9ae6304bae7c647424dcfcb9f"
REVIEW2_HASHES = {
    "PANEL_CLOSEOUT_BINDING.md": "c650b26d6766e12cb6968f7b851e001eaf0ede6ce7fe1da62bf1c858e756f177",
    "TRACK_C_REVIEW2_PROTOCOL.md": "8043ccf7fc053b5bd0089e504469fe33908fe5ae178e4b0e548d85a2a5405a4c",
    "track_c_estimand_lock_candidate.json": "d2cc4a17a2be5768857d649f09a3a007f60a13f5edca450b707d47d310884d11",
    "track_c_sensitivity_and_cost.json": "cd40fcde024d064d19850a37cb24328c222f450a9362f63e3b3fa7b4be985f63",
    "track_c_prior_art_collision_matrix.md": "ab48f2d43f42b59dbbbf39541a419fb5e72e66b0a4e019acd07be2d52f9b4fce",
    "track_c_review2_validator.py": "61a5742a794d353bb4a68341c212f1a45a5f55a7473b9e411dbc00b61639b7ce",
    "tests/test_track_c_review2_validator.py": "906a7719e69c2acddc03a1f193cdc9ea38b7cf14134bb9bceb45b54fd001499e",
}
PANEL_CLOSEOUT_HASHES = {
    "checkpoint_transfer_panel_lock.json": "27972f4ba4920c45b272fa7ea6360cbae2fb4cc748a1ef9ededa681a5dad8526",
    "fresh_square_operator_reservations.json": "22f875050a16a0ad0f170539cdf99bc145fc555c5908ba647c632fb4a86d9e24",
    "panel_execution_authorization.json": "ed71079bf9905ce3cec5a1d6ecd7c515cbda0f3464d86e4bbb69db1358a745f5",
}
PLANNING_NOTE_SHA256 = (
    "4677d79cedc9671219c7c8fd693d08188d771d43f1c70b2b3885fd66eec2997a"
)

REQUIRED_FILES = (
    "TRACK_C_REVIEW2_1_AMENDMENT.md",
    "track_c_estimand_lock_candidate_v2.json",
    "track_c_review2_1_validator.py",
    "track_c_null_calibration.json",
    "track_c_credit_staged_execution_plan.md",
    "REVIEW2_1_REPORT.md",
)

REQUIRED_ANALYSIS_TERMINALS = {
    "NO_POSITIVE_INCREMENT",
    "WRONG_DIRECTION",
    "NOT_SIGNIFICANT",
    "NO_OUTCOME_VARIANCE",
    "NO_REPRESENTATION_FEATURE_VARIANCE",
    "INSUFFICIENT_QUALIFIED_BLOCKS",
    "INSUFFICIENT_DEPTH_STRATUM",
    "RANK_DEFICIENT_DESIGN",
    "DEGENERATE_NUISANCE_SSE",
    "EXCESSIVE_LEVERAGE",
    "INVALID_PERMUTATION_SUPPORT",
}


class Review21ValidationError(ValueError):
    """Raised for malformed prospective inputs or package drift."""


class AnalysisTerminal(RuntimeError):
    """Fail-closed scientific terminal with structured diagnostics."""

    def __init__(self, state: str, message: str, **details: Any):
        super().__init__(message)
        self.state = state
        self.details = details


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Review21ValidationError(f"JSON root must be an object: {path}")
    return value


def _finite_vector(value: Any, *, context: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        raise Review21ValidationError(f"{context} must be a finite vector")
    return vector


def _relative_variance_floor(vector: np.ndarray) -> float:
    scale = max(1.0, float(np.mean(np.square(vector))))
    return NUMERICAL_RELATIVE_FLOOR * scale


def amplitude_representation_observable(
    energy_components: Mapping[int | str, Sequence[float]] | Any,
) -> dict[str, Any]:
    """Aggregate six gauge-invariant energy components then take one square root."""

    if isinstance(energy_components, Mapping):
        keys = {int(key) for key in energy_components}
        if keys != set(FROZEN_LAYERS):
            raise Review21ValidationError("energy components must use the frozen layers")
        matrix = np.asarray(
            [energy_components.get(layer, energy_components.get(str(layer))) for layer in FROZEN_LAYERS],
            dtype=np.float64,
        )
    else:
        matrix = np.asarray(energy_components, dtype=np.float64)
    if matrix.shape != (len(FROZEN_LAYERS), 2) or not np.all(np.isfinite(matrix)):
        raise Review21ValidationError("energy components must be a finite 3x2 matrix")
    if np.any(matrix < 0.0):
        raise Review21ValidationError("unsquared energy components must be nonnegative")
    mean_energy = float(np.mean(matrix))
    amplitude = math.sqrt(mean_energy)
    return {
        "primary_amplitude": amplitude,
        "mean_unsquared_energy": mean_energy,
        "unsquared_energy_components": {
            str(layer): [float(value) for value in matrix[index]]
            for index, layer in enumerate(FROZEN_LAYERS)
        },
        "component_order": [
            {"layer": layer, "training_half": half}
            for layer in FROZEN_LAYERS
            for half in (1, 2)
        ],
    }


def crossfit_return_energy(
    delta_in_training_half_gauges: Any,
    opposite_half_source_activations: Any,
    training_half_source_frame: Any,
) -> float:
    """Compute one held-out energy component in one compatible source gauge."""

    delta = np.asarray(delta_in_training_half_gauges, dtype=np.float64)
    activations = np.asarray(opposite_half_source_activations, dtype=np.float64)
    frame = np.asarray(training_half_source_frame, dtype=np.float64)
    if delta.shape != (FRAME_RANK, FRAME_RANK) or not np.all(np.isfinite(delta)):
        raise Review21ValidationError("Delta must be a finite rank-four square operator")
    if (
        activations.ndim != 2
        or activations.shape[0] < 2
        or not np.all(np.isfinite(activations))
    ):
        raise Review21ValidationError("opposite-half source activations are malformed")
    if (
        frame.shape != (activations.shape[1], FRAME_RANK)
        or not np.all(np.isfinite(frame))
        or np.linalg.matrix_rank(frame) != FRAME_RANK
    ):
        raise Review21ValidationError("training-half source frame must have rank four")
    gram = frame.T @ frame
    if not np.allclose(gram, np.eye(FRAME_RANK), atol=1.0e-8, rtol=1.0e-8):
        raise Review21ValidationError("training-half source frame must be orthonormal")
    projected = (activations - np.mean(activations, axis=0, keepdims=True)) @ frame
    covariance = np.cov(projected, rowvar=False, ddof=1)
    denominator = float(np.trace(covariance))
    floor = NUMERICAL_RELATIVE_FLOOR * max(
        1.0,
        float(np.mean(np.square(projected))),
    )
    if not np.isfinite(denominator) or denominator <= floor:
        raise Review21ValidationError("held-out source covariance is degenerate")
    numerator = float(np.trace(delta @ covariance @ delta.T))
    if numerator < -1.0e-10 or not np.isfinite(numerator):
        raise Review21ValidationError("return-action energy is not finite/nonnegative")
    return max(0.0, numerator) / denominator


def map_derived_competence(
    map_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute C_b^M only from required exact-square map-stage logits."""

    expected_count = len(MAP_HALVES) * MAP_SAMPLES_PER_NODE_PER_HALF * len(EXACT_MAP_NODES)
    if len(map_rows) != expected_count:
        raise Review21ValidationError(
            f"map competence requires exactly {expected_count} exact-square rows"
        )
    seen: set[tuple[str, str, str]] = set()
    by_half: dict[str, list[float]] = {half: [] for half in MAP_HALVES}
    by_half_node: Counter[tuple[str, str]] = Counter()
    for row in map_rows:
        half = str(row.get("half_id"))
        sample_id = str(row.get("sample_id"))
        node = str(row.get("node_id"))
        if half not in MAP_HALVES or node not in EXACT_MAP_NODES:
            raise Review21ValidationError(
                "map competence may use only exact-square nodes from the two map halves"
            )
        identity = (half, sample_id, node)
        if identity in seen:
            raise Review21ValidationError(f"duplicate map competence row: {identity}")
        seen.add(identity)
        target = int(row.get("target_state", -1))
        logits = np.asarray(row.get("candidate_logits"), dtype=np.float64)
        if target not in (0, 1) or logits.shape != (2,) or not np.all(np.isfinite(logits)):
            raise Review21ValidationError("invalid target/logit map competence row")
        margin = float(logits[target] - logits[1 - target])
        by_half[half].append(margin)
        by_half_node[(half, node)] += 1
    expected_per_node = MAP_SAMPLES_PER_NODE_PER_HALF
    if any(
        by_half_node[(half, node)] != expected_per_node
        for half in MAP_HALVES
        for node in EXACT_MAP_NODES
    ):
        raise Review21ValidationError("map competence half/node support drift")
    half_means = {half: float(np.mean(by_half[half])) for half in MAP_HALVES}
    return {
        "map_derived_competence": 0.5 * sum(half_means.values()),
        "half_means": half_means,
        "row_count": expected_count,
        "broken_square_rows_used": 0,
        "behavior_rows_used": 0,
        "formula": "0.5*(mean_exact_map_margin_half_1+mean_exact_map_margin_half_2)",
    }


def block_behavioral_outcome(
    path_p_margins: Any,
    path_q_margins: Any,
    *,
    expected_episodes: int = BEHAVIOR_EPISODES_PER_BLOCK,
) -> float:
    path_p = _finite_vector(path_p_margins, context="path P margins")
    path_q = _finite_vector(path_q_margins, context="path Q margins")
    if path_p.shape != path_q.shape or path_p.size != expected_episodes:
        raise Review21ValidationError("behavior paths require one complete matched episode ledger")
    return float(np.sqrt(np.mean(np.square(path_p - path_q))))


def _decode_hex(value: Any, *, context: str) -> bytes:
    if not isinstance(value, str):
        raise Review21ValidationError(f"{context} must be hexadecimal text")
    try:
        return bytes.fromhex(value)
    except ValueError as exc:
        raise Review21ValidationError(f"invalid hexadecimal bytes: {context}") from exc


def validate_path_surface_pair(pair: Mapping[str, Any]) -> dict[str, Any]:
    """Require P/Q to differ only by the order of matched operation segments."""

    pair_id = str(pair.get("pair_id", ""))
    block_id = str(pair.get("block_id", ""))
    episode_id = str(pair.get("episode_id", ""))
    if not pair_id or not block_id or not episode_id:
        raise Review21ValidationError("path-surface pair lacks pair/block/episode identity")
    paths = {name: pair.get(name) for name in ("path_p", "path_q")}
    if not all(isinstance(value, Mapping) for value in paths.values()):
        raise Review21ValidationError(f"path-surface pair is incomplete: {pair_id}")

    normalized: dict[str, dict[str, Any]] = {}
    for name, raw_value in paths.items():
        value = dict(raw_value)  # type: ignore[arg-type]
        rendered = _decode_hex(value.get("rendered_utf8_hex"), context=f"{pair_id}:{name}:rendered")
        answer_prefix = _decode_hex(
            value.get("answer_prefix_utf8_hex"),
            context=f"{pair_id}:{name}:answer_prefix",
        )
        try:
            rendered.decode("utf-8")
            answer_prefix.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise Review21ValidationError(f"path bytes are not UTF-8: {pair_id}:{name}") from exc
        token_ids = tuple(int(token) for token in value.get("token_ids", []))
        non_operation = tuple(int(token) for token in value.get("non_operation_token_ids", []))
        operation_segments = tuple(
            tuple(int(token) for token in segment)
            for segment in value.get("operation_token_ids", [])
        )
        codebook_tokens = tuple(int(token) for token in value.get("codebook_token_ids", []))
        if not token_ids or any(token < 0 for token in token_ids):
            raise Review21ValidationError(f"invalid token ids: {pair_id}:{name}")
        if not operation_segments or any(not segment for segment in operation_segments):
            raise Review21ValidationError(f"operation segmentation missing: {pair_id}:{name}")
        normalized[name] = {
            "canonical_template_id": str(value.get("canonical_template_id")),
            "canonical_template_sha256": str(value.get("canonical_template_sha256")),
            "message_count": int(value.get("message_count", -1)),
            "operation_count": int(value.get("operation_count", -1)),
            "operation_segments": operation_segments,
            "operation_segment_multiset": Counter(operation_segments),
            "operation_token_multiset": Counter(
                token for segment in operation_segments for token in segment
            ),
            "codebook_token_multiset": Counter(codebook_tokens),
            "special_token_count": int(value.get("special_token_count", -1)),
            "answer_prefix": answer_prefix,
            "score_slot": value.get("score_slot"),
            "token_ids": token_ids,
            "non_operation_token_ids": non_operation,
            "rendered_bytes_sha256": hashlib.sha256(rendered).hexdigest(),
            "rendered_utf8_hex": rendered.hex(),
            "token_ids": list(token_ids),
            "token_ids_sha256": canonical_sha256(list(token_ids)),
            "total_rendered_input_token_count": len(token_ids),
        }

    left = normalized["path_p"]
    right = normalized["path_q"]
    checks = {
        "canonical_template": (
            left["canonical_template_id"] == FROZEN_TEMPLATE
            and right["canonical_template_id"] == FROZEN_TEMPLATE
            and left["canonical_template_sha256"] == right["canonical_template_sha256"]
            and re.fullmatch(r"[0-9a-f]{64}", left["canonical_template_sha256"])
            is not None
        ),
        "message_count": left["message_count"] == right["message_count"] and left["message_count"] > 0,
        "operation_count": (
            left["operation_count"] == right["operation_count"]
            and left["operation_count"] == len(left["operation_segments"])
            and right["operation_count"] == len(right["operation_segments"])
        ),
        "operation_token_multiset": left["operation_segment_multiset"] == right["operation_segment_multiset"],
        "individual_operation_token_multiset": (
            left["operation_token_multiset"] == right["operation_token_multiset"]
        ),
        "operation_order_is_the_only_intended_difference": (
            left["operation_segments"] != right["operation_segments"]
        ),
        "codebook_tokens": left["codebook_token_multiset"] == right["codebook_token_multiset"],
        "special_token_count": left["special_token_count"] == right["special_token_count"],
        "answer_prefix": left["answer_prefix"] == right["answer_prefix"],
        "score_slot": left["score_slot"] == right["score_slot"],
        "total_rendered_input_token_count": (
            left["total_rendered_input_token_count"]
            == right["total_rendered_input_token_count"]
        ),
        "non_operation_token_sequence": (
            left["non_operation_token_ids"] == right["non_operation_token_ids"]
        ),
    }
    return {
        "pair_id": pair_id,
        "block_id": block_id,
        "episode_id": episode_id,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "path_p": {
            key: value
            for key, value in left.items()
            if key
            in {
                "rendered_bytes_sha256",
                "rendered_utf8_hex",
                "token_ids",
                "token_ids_sha256",
                "total_rendered_input_token_count",
                "message_count",
                "operation_count",
                "special_token_count",
                "score_slot",
            }
        },
        "path_q": {
            key: value
            for key, value in right.items()
            if key
            in {
                "rendered_bytes_sha256",
                "rendered_utf8_hex",
                "token_ids",
                "token_ids_sha256",
                "total_rendered_input_token_count",
                "message_count",
                "operation_count",
                "special_token_count",
                "score_slot",
            }
        },
        "mismatch_status": "NONE" if all(checks.values()) else "PATH_SURFACE_MISMATCH",
    }


def validate_path_surface_ledger(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not pairs:
        raise Review21ValidationError("path-surface ledger is empty")
    results = [validate_path_surface_pair(pair) for pair in pairs]
    identities = [result["pair_id"] for result in results]
    if len(set(identities)) != len(identities):
        raise Review21ValidationError("duplicate path-surface pair identity")
    block_episode_identities = [
        (result["block_id"], result["episode_id"]) for result in results
    ]
    if len(set(block_episode_identities)) != len(block_episode_identities):
        raise Review21ValidationError("duplicate block/episode path-surface identity")
    return {
        "schema_version": "gate13_track_c_path_surface_validation_v1",
        "status": "PASS" if all(result["status"] == "PASS" for result in results) else "FAIL",
        "pair_count": len(results),
        "pair_results": results,
        "input_ledger_sha256": canonical_sha256(list(pairs)),
        "mismatch_action": "REVIEW2_1_BLOCKED_BEFORE_ANY_FORWARD",
        "broken_square_is_serialization_control": False,
    }


def _depth_vector(value: Any, *, expected_size: int | None = None) -> np.ndarray:
    depth = np.asarray(value, dtype=np.int64)
    if depth.ndim != 1 or (expected_size is not None and depth.size != expected_size):
        raise Review21ValidationError("rollout depth must be a one-dimensional block vector")
    if set(depth.tolist()) - set(DEPTH_LEVELS):
        raise Review21ValidationError("rollout depth left the frozen categorical levels")
    return depth


def _depth_dummies(depth: np.ndarray) -> np.ndarray:
    """Reference-code the frozen four-level factor with depth 2 as reference."""

    return np.column_stack([(depth == level).astype(np.float64) for level in DEPTH_LEVELS[1:]])


def _training_scale(
    vector: np.ndarray,
    training: np.ndarray,
    *,
    context: str,
    terminal: str,
) -> tuple[float, float]:
    values = vector[training]
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    variance_floor = _relative_variance_floor(values)
    if not np.isfinite(sd) or sd * sd <= variance_floor:
        raise AnalysisTerminal(
            terminal,
            f"{context} has no usable training-fold variance",
            variance=float(sd * sd),
            variance_floor=variance_floor,
        )
    return mean, sd


def _check_rank_condition(matrix: np.ndarray, *, context: str) -> float:
    rank = int(np.linalg.matrix_rank(matrix))
    if rank != matrix.shape[1] or matrix.shape[0] <= matrix.shape[1]:
        raise AnalysisTerminal(
            "RANK_DEFICIENT_DESIGN",
            f"{context} is rank deficient or lacks residual degrees of freedom",
            rows=int(matrix.shape[0]),
            columns=int(matrix.shape[1]),
            rank=rank,
        )
    condition = float(np.linalg.cond(matrix))
    if not np.isfinite(condition) or condition > DESIGN_CONDITION_CEILING:
        raise AnalysisTerminal(
            "RANK_DEFICIENT_DESIGN",
            f"{context} exceeds the frozen condition-number ceiling",
            condition=condition,
            ceiling=DESIGN_CONDITION_CEILING,
        )
    return condition


def leverage_threshold(column_count: int, row_count: int) -> float:
    """Three times average leverage, capped to keep PRESS amplification <= 5."""

    if column_count <= 0 or row_count <= column_count:
        raise Review21ValidationError("invalid design dimensions for leverage threshold")
    return min(
        LEVERAGE_ABSOLUTE_CEILING,
        LEVERAGE_AVERAGE_MULTIPLIER * column_count / row_count,
    )


def _check_leverage(matrix: np.ndarray, *, context: str) -> dict[str, float]:
    inverse = np.linalg.inv(matrix.T @ matrix)
    leverage = np.einsum("ij,jk,ik->i", matrix, inverse, matrix)
    maximum = float(np.max(leverage))
    threshold = leverage_threshold(matrix.shape[1], matrix.shape[0])
    if maximum > threshold + 1.0e-12:
        raise AnalysisTerminal(
            "EXCESSIVE_LEVERAGE",
            f"{context} has a block above the frozen leverage threshold",
            maximum_leverage=maximum,
            threshold=threshold,
            column_count=int(matrix.shape[1]),
            row_count=int(matrix.shape[0]),
        )
    return {
        "maximum": maximum,
        "threshold": threshold,
        "minimum_press_denominator": 1.0 - maximum,
    }


def _scaled_design(
    depth: np.ndarray,
    competence: np.ndarray,
    representation: np.ndarray,
    *,
    competence_mean: float,
    competence_sd: float,
    representation_mean: float | None,
    representation_sd: float | None,
    include_representation: bool,
) -> np.ndarray:
    columns = [
        np.ones(depth.size, dtype=np.float64),
        _depth_dummies(depth),
        ((competence - competence_mean) / competence_sd)[:, None],
    ]
    if include_representation:
        if representation_mean is None or representation_sd is None:
            raise Review21ValidationError("full design lacks representation scaling")
        columns.append(((representation - representation_mean) / representation_sd)[:, None])
    return np.column_stack(columns)


@dataclass(frozen=True)
class AnalysisGeometry:
    depth: np.ndarray
    competence: np.ndarray
    representation: np.ndarray
    nuisance_lobo_operator: np.ndarray
    full_lobo_operator: np.ndarray
    nuisance_full_projection: np.ndarray
    beta_r_weight: np.ndarray
    diagnostics: dict[str, Any]


def _lobo_residual_operator(
    depth: np.ndarray,
    competence: np.ndarray,
    representation: np.ndarray,
    *,
    include_representation: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    n = depth.size
    operator = np.zeros((n, n), dtype=np.float64)
    maximum_condition = 0.0
    maximum_leverage = 0.0
    minimum_leverage_threshold = math.inf
    for held_out in range(n):
        training = np.arange(n) != held_out
        competence_mean, competence_sd = _training_scale(
            competence,
            training,
            context="map-derived competence",
            terminal="RANK_DEFICIENT_DESIGN",
        )
        if include_representation:
            representation_mean, representation_sd = _training_scale(
                representation,
                training,
                context="representation feature",
                terminal="NO_REPRESENTATION_FEATURE_VARIANCE",
            )
        else:
            representation_mean = None
            representation_sd = None
        training_matrix = _scaled_design(
            depth[training],
            competence[training],
            representation[training],
            competence_mean=competence_mean,
            competence_sd=competence_sd,
            representation_mean=representation_mean,
            representation_sd=representation_sd,
            include_representation=include_representation,
        )
        held_out_matrix = _scaled_design(
            depth[[held_out]],
            competence[[held_out]],
            representation[[held_out]],
            competence_mean=competence_mean,
            competence_sd=competence_sd,
            representation_mean=representation_mean,
            representation_sd=representation_sd,
            include_representation=include_representation,
        )
        condition = _check_rank_condition(
            training_matrix,
            context=("full" if include_representation else "nuisance")
            + f" LOBO training fold {held_out}",
        )
        leverage = _check_leverage(
            training_matrix,
            context=("full" if include_representation else "nuisance")
            + f" LOBO training fold {held_out}",
        )
        solve = np.linalg.solve(
            training_matrix.T @ training_matrix,
            training_matrix.T,
        )
        prediction_weights = held_out_matrix @ solve
        operator[held_out, held_out] = 1.0
        operator[held_out, training] = -prediction_weights[0]
        maximum_condition = max(maximum_condition, condition)
        maximum_leverage = max(maximum_leverage, leverage["maximum"])
        minimum_leverage_threshold = min(
            minimum_leverage_threshold,
            leverage["threshold"],
        )
    return operator, {
        "maximum_training_fold_condition": maximum_condition,
        "maximum_training_fold_leverage": maximum_leverage,
        "minimum_training_fold_leverage_threshold": minimum_leverage_threshold,
        "scaling": "FIT_INDEPENDENTLY_INSIDE_EACH_LOBO_TRAINING_FOLD",
    }


def build_analysis_geometry(
    rollout_depth: Any,
    map_competence: Any,
    representation_feature: Any,
) -> AnalysisGeometry:
    competence = _finite_vector(map_competence, context="map-derived competence")
    representation = _finite_vector(
        representation_feature,
        context="representation feature",
    )
    if competence.size != representation.size:
        raise Review21ValidationError("predictor vectors have unequal block counts")
    depth = _depth_vector(rollout_depth, expected_size=competence.size)
    if representation.size < MINIMUM_QUALIFIED_BLOCKS:
        raise AnalysisTerminal(
            "INSUFFICIENT_QUALIFIED_BLOCKS",
            "fewer than sixteen qualified blocks remain",
            qualified_blocks=int(representation.size),
        )
    counts = {level: int(np.sum(depth == level)) for level in DEPTH_LEVELS}
    if any(counts[level] < MINIMUM_QUALIFIED_PER_DEPTH for level in DEPTH_LEVELS):
        raise AnalysisTerminal(
            "INSUFFICIENT_DEPTH_STRATUM",
            "a rollout-depth stratum has fewer than four qualified blocks",
            depth_counts=counts,
        )
    representation_variance = float(np.var(representation, ddof=1))
    representation_floor = _relative_variance_floor(representation)
    if representation_variance <= representation_floor:
        raise AnalysisTerminal(
            "NO_REPRESENTATION_FEATURE_VARIANCE",
            "qualified blocks have no usable representation-feature variance",
            variance=representation_variance,
            variance_floor=representation_floor,
        )

    full_index = np.ones(representation.size, dtype=bool)
    competence_mean, competence_sd = _training_scale(
        competence,
        full_index,
        context="map-derived competence",
        terminal="RANK_DEFICIENT_DESIGN",
    )
    representation_mean, representation_sd = _training_scale(
        representation,
        full_index,
        context="representation feature",
        terminal="NO_REPRESENTATION_FEATURE_VARIANCE",
    )
    nuisance_scaled = _scaled_design(
        depth,
        competence,
        representation,
        competence_mean=competence_mean,
        competence_sd=competence_sd,
        representation_mean=None,
        representation_sd=None,
        include_representation=False,
    )
    full_scaled = _scaled_design(
        depth,
        competence,
        representation,
        competence_mean=competence_mean,
        competence_sd=competence_sd,
        representation_mean=representation_mean,
        representation_sd=representation_sd,
        include_representation=True,
    )
    nuisance_condition = _check_rank_condition(
        nuisance_scaled,
        context="full-cohort nuisance design",
    )
    full_condition = _check_rank_condition(
        full_scaled,
        context="full-cohort full design",
    )
    nuisance_leverage = _check_leverage(
        nuisance_scaled,
        context="full-cohort nuisance design",
    )
    full_leverage = _check_leverage(
        full_scaled,
        context="full-cohort full design",
    )
    nuisance_operator, nuisance_fold_diagnostics = _lobo_residual_operator(
        depth,
        competence,
        representation,
        include_representation=False,
    )
    full_operator, full_fold_diagnostics = _lobo_residual_operator(
        depth,
        competence,
        representation,
        include_representation=True,
    )

    nuisance_raw = np.column_stack(
        [np.ones(depth.size), _depth_dummies(depth), competence]
    )
    full_raw = np.column_stack([nuisance_raw, representation])
    if np.linalg.matrix_rank(full_raw) != full_raw.shape[1]:
        raise AnalysisTerminal(
            "RANK_DEFICIENT_DESIGN",
            "raw-scale directional full model is rank deficient",
        )
    nuisance_projection = nuisance_raw @ np.linalg.pinv(nuisance_raw)
    beta_r_weight = np.linalg.pinv(full_raw)[-1]
    return AnalysisGeometry(
        depth=depth,
        competence=competence,
        representation=representation,
        nuisance_lobo_operator=nuisance_operator,
        full_lobo_operator=full_operator,
        nuisance_full_projection=nuisance_projection,
        beta_r_weight=beta_r_weight,
        diagnostics={
            "qualified_blocks": int(depth.size),
            "depth_counts": {str(key): value for key, value in counts.items()},
            "nuisance_columns": [
                "INTERCEPT",
                "DEPTH_4_INDICATOR",
                "DEPTH_6_INDICATOR",
                "DEPTH_8_INDICATOR",
                "MAP_DERIVED_COMPETENCE",
            ],
            "full_additional_column": "REPRESENTATION_AMPLITUDE_R_B",
            "depth_reference": 2,
            "full_cohort_condition": {
                "nuisance": nuisance_condition,
                "full": full_condition,
                "ceiling": DESIGN_CONDITION_CEILING,
            },
            "full_cohort_leverage": {
                "nuisance": nuisance_leverage,
                "full": full_leverage,
            },
            "lobo_folds": {
                "nuisance": nuisance_fold_diagnostics,
                "full": full_fold_diagnostics,
            },
            "leverage_rule": "h_max <= min(0.80, 3*p/n)",
            "press_amplification_ceiling": 1.0 / MINIMUM_PRESS_DENOMINATOR,
            "beta_r_definition": (
                "RAW_SCALE_R_B_COEFFICIENT_FROM_THE_FULL_COHORT_FULL_OLS_MODEL; "
                "SIGN_IS_INVARIANT_TO_POSITIVE_CENTERING_AND_SCALING"
            ),
        },
    )


def permutation_support_size(depth: Any) -> int:
    values = _depth_vector(depth)
    counts = [int(np.sum(values == level)) for level in DEPTH_LEVELS]
    return math.prod(math.factorial(count) for count in counts) - 1


def generate_stratified_permutation_schedule(
    depth: Any,
    *,
    permutations: int,
    seed: int,
) -> np.ndarray:
    values = _depth_vector(depth)
    if permutations <= 0:
        raise Review21ValidationError("permutation count must be positive")
    support = permutation_support_size(values)
    if support < permutations:
        raise AnalysisTerminal(
            "INVALID_PERMUTATION_SUPPORT",
            "frozen depth strata cannot support the requested unique permutations",
            support=support,
            requested=permutations,
        )
    counts = [int(np.sum(values == level)) for level in DEPTH_LEVELS]
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), *counts, int(permutations)])
    )
    identity = tuple(range(values.size))
    seen: set[tuple[int, ...]] = set()
    rows: list[tuple[int, ...]] = []
    strata = [np.flatnonzero(values == level) for level in DEPTH_LEVELS]
    while len(rows) < permutations:
        candidate = np.arange(values.size, dtype=np.int64)
        for indices in strata:
            candidate[indices] = rng.permutation(indices)
        key = tuple(int(value) for value in candidate)
        if key == identity or key in seen:
            continue
        seen.add(key)
        rows.append(key)
    return np.asarray(rows, dtype=np.int64)


def enumerate_stratified_permutations(depth: Any) -> np.ndarray:
    """Enumerate every non-identity within-depth permutation for small tests."""

    values = _depth_vector(depth)
    strata = [tuple(int(index) for index in np.flatnonzero(values == level)) for level in DEPTH_LEVELS]
    per_stratum = [tuple(itertools.permutations(indices)) for indices in strata]
    identity = tuple(range(values.size))
    rows = []
    for combination in itertools.product(*per_stratum):
        candidate = list(range(values.size))
        for indices, replacement in zip(strata, combination):
            for output, source in zip(indices, replacement):
                candidate[output] = source
        key = tuple(candidate)
        if key != identity:
            rows.append(key)
    return np.asarray(rows, dtype=np.int64)


def validate_permutation_schedule(
    depth: Any,
    schedule: Any,
    *,
    expected_count: int,
) -> dict[str, Any]:
    values = _depth_vector(depth)
    matrix = np.asarray(schedule, dtype=np.int64)
    if matrix.shape != (expected_count, values.size):
        raise AnalysisTerminal(
            "INVALID_PERMUTATION_SUPPORT",
            "permutation schedule has the wrong shape",
            observed_shape=list(matrix.shape),
            expected_shape=[expected_count, int(values.size)],
        )
    identity = tuple(range(values.size))
    keys = [tuple(int(value) for value in row) for row in matrix]
    valid_indices = all(sorted(row) == list(range(values.size)) for row in keys)
    preserves_depth = all(np.array_equal(values[row], values) for row in matrix)
    unique = len(set(keys)) == len(keys)
    excludes_identity = identity not in set(keys)
    support = permutation_support_size(values)
    if not (valid_indices and preserves_depth and unique and excludes_identity and support >= expected_count):
        raise AnalysisTerminal(
            "INVALID_PERMUTATION_SUPPORT",
            "permutation schedule violates the frozen support contract",
            valid_indices=valid_indices,
            preserves_depth=preserves_depth,
            unique=unique,
            excludes_identity=excludes_identity,
            support=support,
        )
    return {
        "count": expected_count,
        "support_excluding_identity": support,
        "schedule_sha256": canonical_sha256(matrix.tolist()),
        "unique": True,
        "identity_excluded": True,
        "depth_strata_preserved": True,
    }


def qualified_block_indices(
    rollout_depth: Any,
    qualification_mask: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the frozen outcome-blind eligibility floor to the planned blocks."""

    depth = _depth_vector(rollout_depth, expected_size=PLANNED_BLOCKS)
    planned_counts = {level: int(np.sum(depth == level)) for level in DEPTH_LEVELS}
    if any(planned_counts[level] != PLANNED_BLOCKS_PER_DEPTH for level in DEPTH_LEVELS):
        raise Review21ValidationError(
            "the planned ledger must contain exactly five blocks at each depth"
        )
    raw_mask = np.asarray(qualification_mask)
    if raw_mask.shape != (PLANNED_BLOCKS,):
        raise Review21ValidationError("qualification mask must cover all twenty blocks")
    if raw_mask.dtype.kind not in "biu" or not np.all(np.isin(raw_mask, [0, 1])):
        raise Review21ValidationError("qualification mask must be Boolean")
    mask = raw_mask.astype(bool)
    indices = np.flatnonzero(mask)
    counts = {level: int(np.sum(mask & (depth == level))) for level in DEPTH_LEVELS}
    if indices.size < MINIMUM_QUALIFIED_BLOCKS:
        raise AnalysisTerminal(
            "INSUFFICIENT_QUALIFIED_BLOCKS",
            "the outcome-blind map qualification left fewer than sixteen blocks",
            qualified_blocks=int(indices.size),
            depth_counts=counts,
        )
    if any(counts[level] < MINIMUM_QUALIFIED_PER_DEPTH for level in DEPTH_LEVELS):
        raise AnalysisTerminal(
            "INSUFFICIENT_DEPTH_STRATUM",
            "the outcome-blind map qualification left fewer than four blocks in a depth stratum",
            qualified_blocks=int(indices.size),
            depth_counts=counts,
        )
    return indices, {
        "qualified_blocks": int(indices.size),
        "depth_counts": {str(level): counts[level] for level in DEPTH_LEVELS},
        "eligibility_floor_only": True,
        "all_remaining_predictor_and_analysis_gates_still_required": True,
    }


def derived_schedule_seed(
    *,
    root_seed: int,
    qualified_block_ids: Sequence[str],
) -> int:
    """Select one member of the ex-ante schedule family without outcome input."""

    if not qualified_block_ids or len(set(qualified_block_ids)) != len(qualified_block_ids):
        raise Review21ValidationError("qualified block IDs must be nonempty and unique")
    digest = hashlib.sha256(
        json.dumps(
            {
                "algorithm": "SHA256_ROOT_SEED_AND_ORDERED_QUALIFIED_BLOCK_IDS_V1",
                "root_seed": int(root_seed),
                "qualified_block_ids": list(qualified_block_ids),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def evaluate_map_stage_predictor_gates(
    *,
    block_ids: Sequence[str],
    rollout_depth: Any,
    map_competence: Any,
    representation_feature: Any,
    qualification_mask: Any,
    split_half_valid: bool,
    frame_rank_valid: bool,
    conditioning_valid: bool,
    exact_square_reproducibility_valid: bool,
    broken_square_sensitivity_valid: bool,
    path_surface_valid: bool,
    artifact_complete: bool,
) -> dict[str, Any]:
    """Evaluate every predictor-side gate before behavior collection can open.

    The returned public summary deliberately excludes block-level representation
    values, layer packets, spectra, and predictor plots.
    """

    if len(block_ids) != PLANNED_BLOCKS or len(set(block_ids)) != PLANNED_BLOCKS:
        raise Review21ValidationError("the Stage M ledger requires twenty unique block IDs")
    depth = _depth_vector(rollout_depth, expected_size=PLANNED_BLOCKS)
    competence = _finite_vector(map_competence, context="map-derived competence")
    representation = _finite_vector(
        representation_feature,
        context="representation feature",
    )
    if competence.size != PLANNED_BLOCKS or representation.size != PLANNED_BLOCKS:
        raise Review21ValidationError("Stage M predictors must cover all planned blocks")
    external_gates = {
        "split_half_valid": bool(split_half_valid),
        "frame_rank_valid": bool(frame_rank_valid),
        "conditioning_valid": bool(conditioning_valid),
        "exact_square_reproducibility_valid": bool(exact_square_reproducibility_valid),
        "broken_square_sensitivity_valid": bool(broken_square_sensitivity_valid),
        "path_surface_valid": bool(path_surface_valid),
        "artifact_complete": bool(artifact_complete),
    }
    failed_external = [name for name, passed in external_gates.items() if not passed]
    if failed_external:
        raise AnalysisTerminal(
            "MAP_COMPLETE_NOT_QUALIFIED",
            "one or more frozen predictor-side gates failed",
            failed_gates=failed_external,
        )
    indices, qualification = qualified_block_indices(depth, qualification_mask)
    geometry = build_analysis_geometry(
        depth[indices],
        competence[indices],
        representation[indices],
    )
    support = permutation_support_size(depth[indices])
    if support < SCIENTIFIC_PERMUTATIONS:
        raise AnalysisTerminal(
            "INVALID_PERMUTATION_SUPPORT",
            "qualified depth strata do not support the frozen scientific schedule",
            support=support,
            requested=SCIENTIFIC_PERMUTATIONS,
        )
    qualified_ids = [str(block_ids[index]) for index in indices]
    schedule_seed = derived_schedule_seed(
        root_seed=SCIENTIFIC_PERMUTATION_SEED,
        qualified_block_ids=qualified_ids,
    )
    return {
        "state": "MAP_COMPLETE_AND_QUALIFIED",
        "qualified_indices": indices,
        "geometry": geometry,
        "sealed_public_summary": {
            "qualification_state": "MAP_COMPLETE_AND_QUALIFIED",
            "qualified_blocks": qualification["qualified_blocks"],
            "depth_counts": qualification["depth_counts"],
        },
        "sealed_private_analysis_metadata": {
            "permutation_support_excluding_identity": support,
            "schedule_family_member_seed": schedule_seed,
        },
    }


def _outcome_statistics(
    outcome: np.ndarray,
    geometry: AnalysisGeometry,
) -> dict[str, float]:
    nuisance_residual = geometry.nuisance_lobo_operator @ outcome
    full_residual = geometry.full_lobo_operator @ outcome
    nuisance_sse = float(nuisance_residual @ nuisance_residual)
    full_sse = float(full_residual @ full_residual)
    nuisance_floor = NUMERICAL_RELATIVE_FLOOR * max(
        1.0,
        float(outcome @ outcome),
    )
    if not np.isfinite(nuisance_sse) or nuisance_sse <= nuisance_floor:
        raise AnalysisTerminal(
            "DEGENERATE_NUISANCE_SSE",
            "observed LOBO nuisance SSE is at or below the frozen numerical floor",
            nuisance_sse=nuisance_sse,
            numerical_floor=nuisance_floor,
        )
    return {
        "sse_nuisance_lobo": nuisance_sse,
        "sse_full_lobo": full_sse,
        "t_lobo": 1.0 - full_sse / nuisance_sse,
        "beta_r": float(geometry.beta_r_weight @ outcome),
        "nuisance_sse_floor": nuisance_floor,
    }


def run_primary_pipeline(
    *,
    outcome: Any,
    geometry: AnalysisGeometry,
    schedule: Any,
    alpha: float = FINAL_AUDIT_ALPHA,
    validate_schedule: bool = True,
) -> dict[str, Any]:
    """Run the exact directional LOBO/Freedman--Lane primary pipeline."""

    y = _finite_vector(outcome, context="block-level behavioral outcome")
    if y.size != geometry.depth.size:
        raise Review21ValidationError("outcome and qualified predictor counts differ")
    outcome_variance = float(np.var(y, ddof=1))
    outcome_floor = _relative_variance_floor(y)
    if outcome_variance <= outcome_floor:
        raise AnalysisTerminal(
            "NO_OUTCOME_VARIANCE",
            "qualified block outcomes have no usable variance",
            variance=outcome_variance,
            variance_floor=outcome_floor,
        )
    permutation_matrix = np.asarray(schedule, dtype=np.int64)
    permutation_count = int(permutation_matrix.shape[0]) if permutation_matrix.ndim == 2 else -1
    if validate_schedule:
        schedule_diagnostics = validate_permutation_schedule(
            geometry.depth,
            permutation_matrix,
            expected_count=permutation_count,
        )
    else:
        schedule_diagnostics = {
            "count": permutation_count,
            "support_excluding_identity": permutation_support_size(geometry.depth),
            "schedule_sha256": hashlib.sha256(
                permutation_matrix.astype("<i8", copy=False).tobytes(order="C")
            ).hexdigest(),
            "schedule_encoding": "LITTLE_ENDIAN_INT64_ROW_MAJOR",
            "schedule_validation": "PREVALIDATED_BY_TRUSTED_GENERATOR_OR_EXACT_ENUMERATOR",
        }

    observed = _outcome_statistics(y, geometry)
    nuisance_fit = geometry.nuisance_full_projection @ y
    nuisance_residual = y - nuisance_fit
    permuted_outcome = nuisance_fit[None, :] + nuisance_residual[permutation_matrix]
    nuisance_lobo_residual = permuted_outcome @ geometry.nuisance_lobo_operator.T
    full_lobo_residual = permuted_outcome @ geometry.full_lobo_operator.T
    nuisance_sse = np.einsum("ij,ij->i", nuisance_lobo_residual, nuisance_lobo_residual)
    full_sse = np.einsum("ij,ij->i", full_lobo_residual, full_lobo_residual)
    permutation_floor = NUMERICAL_RELATIVE_FLOOR * np.maximum(
        1.0,
        np.einsum("ij,ij->i", permuted_outcome, permuted_outcome),
    )
    if np.any(~np.isfinite(nuisance_sse)) or np.any(nuisance_sse <= permutation_floor):
        raise AnalysisTerminal(
            "DEGENERATE_NUISANCE_SSE",
            "a refitted permutation pipeline has degenerate nuisance SSE",
            degenerate_permutations=int(np.sum(nuisance_sse <= permutation_floor)),
        )
    permutation_t = 1.0 - full_sse / nuisance_sse
    permutation_beta = permuted_outcome @ geometry.beta_r_weight
    direction_admissible = permutation_beta > 0.0
    exceedances = int(
        np.sum(direction_admissible & (permutation_t >= observed["t_lobo"]))
    )
    p_value = (1.0 + exceedances) / (1.0 + permutation_count)

    if observed["t_lobo"] <= 0.0:
        terminal = "NO_POSITIVE_INCREMENT"
    elif observed["beta_r"] <= 0.0:
        terminal = "WRONG_DIRECTION"
    elif p_value > alpha:
        terminal = "NOT_SIGNIFICANT"
    else:
        terminal = "PRIMARY_POSITIVE"
    return {
        "terminal_state": terminal,
        **observed,
        "one_sided_permutation_p": p_value,
        "exceedances": exceedances,
        "permutations": permutation_count,
        "alpha": alpha,
        "positive_rule": {
            "all_predictor_and_qualification_gates_pass": True,
            "t_lobo_gt_zero": observed["t_lobo"] > 0.0,
            "beta_r_gt_zero": observed["beta_r"] > 0.0,
            "one_sided_p_lte_alpha": p_value <= alpha,
            "joint_positive": terminal == "PRIMARY_POSITIVE",
        },
        "directional_permutation_statistic": (
            "T_PERM_IF_REFITTED_BETA_R_GT_ZERO_ELSE_NEGATIVE_INFINITY"
        ),
        "schedule": schedule_diagnostics,
        "complete_refit": (
            "EVERY_PERMUTED_OUTCOME_IS_APPLIED_TO_DESIGN_FIXED_LINEAR_LOBO_"
            "OPERATORS_DERIVED_BY_EXPLICIT_FOLDWISE_REFITS"
        ),
    }


def brute_force_lobo_statistics(
    *,
    outcome: Any,
    rollout_depth: Any,
    map_competence: Any,
    representation_feature: Any,
) -> dict[str, float]:
    """Slow reference implementation used only to verify the linear operators."""

    y = _finite_vector(outcome, context="reference outcome")
    depth = _depth_vector(rollout_depth, expected_size=y.size)
    competence = _finite_vector(map_competence, context="reference competence")
    representation = _finite_vector(representation_feature, context="reference representation")
    nuisance_errors: list[float] = []
    full_errors: list[float] = []
    for held_out in range(y.size):
        training = np.arange(y.size) != held_out
        competence_mean, competence_sd = _training_scale(
            competence,
            training,
            context="reference competence",
            terminal="RANK_DEFICIENT_DESIGN",
        )
        representation_mean, representation_sd = _training_scale(
            representation,
            training,
            context="reference representation",
            terminal="NO_REPRESENTATION_FEATURE_VARIANCE",
        )
        nuisance_train = _scaled_design(
            depth[training],
            competence[training],
            representation[training],
            competence_mean=competence_mean,
            competence_sd=competence_sd,
            representation_mean=None,
            representation_sd=None,
            include_representation=False,
        )
        nuisance_test = _scaled_design(
            depth[[held_out]],
            competence[[held_out]],
            representation[[held_out]],
            competence_mean=competence_mean,
            competence_sd=competence_sd,
            representation_mean=None,
            representation_sd=None,
            include_representation=False,
        )
        full_train = _scaled_design(
            depth[training],
            competence[training],
            representation[training],
            competence_mean=competence_mean,
            competence_sd=competence_sd,
            representation_mean=representation_mean,
            representation_sd=representation_sd,
            include_representation=True,
        )
        full_test = _scaled_design(
            depth[[held_out]],
            competence[[held_out]],
            representation[[held_out]],
            competence_mean=competence_mean,
            competence_sd=competence_sd,
            representation_mean=representation_mean,
            representation_sd=representation_sd,
            include_representation=True,
        )
        nuisance_beta = np.linalg.solve(
            nuisance_train.T @ nuisance_train,
            nuisance_train.T @ y[training],
        )
        full_beta = np.linalg.solve(
            full_train.T @ full_train,
            full_train.T @ y[training],
        )
        nuisance_errors.append(float((y[held_out] - nuisance_test @ nuisance_beta)[0]))
        full_errors.append(float((y[held_out] - full_test @ full_beta)[0]))
    nuisance_sse = float(np.sum(np.square(nuisance_errors)))
    full_sse = float(np.sum(np.square(full_errors)))
    raw_full = np.column_stack(
        [np.ones(y.size), _depth_dummies(depth), competence, representation]
    )
    return {
        "sse_nuisance_lobo": nuisance_sse,
        "sse_full_lobo": full_sse,
        "t_lobo": 1.0 - full_sse / nuisance_sse,
        "beta_r": float(np.linalg.solve(raw_full.T @ raw_full, raw_full.T @ y)[-1]),
    }


def forward_and_cost_forecast(
    *,
    qualified_blocks: int = PLANNED_BLOCKS,
    qualified_depth_counts: Mapping[int, int] | None = None,
) -> dict[str, Any]:
    if qualified_depth_counts is None:
        if qualified_blocks == MINIMUM_QUALIFIED_BLOCKS:
            counts = {level: MINIMUM_QUALIFIED_PER_DEPTH for level in DEPTH_LEVELS}
        elif qualified_blocks == PLANNED_BLOCKS:
            counts = {level: PLANNED_BLOCKS_PER_DEPTH for level in DEPTH_LEVELS}
        else:
            raise Review21ValidationError(
                "an intermediate qualified-block forecast requires exact depth counts"
            )
    else:
        counts = {int(level): int(count) for level, count in qualified_depth_counts.items()}
        if set(counts) != set(DEPTH_LEVELS):
            raise Review21ValidationError("qualified depth-count forecast is incomplete")
        qualified_blocks = sum(counts.values())
    if (
        qualified_blocks < MINIMUM_QUALIFIED_BLOCKS
        or qualified_blocks > PLANNED_BLOCKS
        or any(
            counts[level] < MINIMUM_QUALIFIED_PER_DEPTH
            or counts[level] > PLANNED_BLOCKS_PER_DEPTH
            for level in DEPTH_LEVELS
        )
    ):
        raise Review21ValidationError("qualified depth counts violate the frozen design")
    rate = EXISTING_B_OPERATOR_COST_USD / EXISTING_B_OPERATOR_FORWARD_COUNT
    stage_m_forwards = PLANNED_BLOCKS * len(MAP_HALVES) * 5 * MAP_SAMPLES_PER_NODE_PER_HALF
    stage_e_forwards = sum(
        counts[level] * 2 * BEHAVIOR_EPISODES_PER_BLOCK * (level + 1)
        for level in DEPTH_LEVELS
    )
    stage_m_variable = stage_m_forwards * rate
    stage_m_total = stage_m_variable + EXISTING_ACQUISITION_AND_PREFLIGHT_COST_USD
    stage_e_total = stage_e_forwards * rate
    expected = stage_m_total + stage_e_total
    return {
        "historical_linear_rate_usd_per_forward": rate,
        "fixed_acquisition_and_preflight_usd": EXISTING_ACQUISITION_AND_PREFLIGHT_COST_USD,
        "stage_m": {
            "forwards": stage_m_forwards,
            "variable_usd": stage_m_variable,
            "planning_usd": stage_m_total,
        },
        "stage_e": {
            "qualified_blocks": qualified_blocks,
            "qualified_depth_counts": {str(level): counts[level] for level in DEPTH_LEVELS},
            "forwards": stage_e_forwards,
            "planning_usd": stage_e_total,
        },
        "total": {
            "forwards": stage_m_forwards + stage_e_forwards,
            "expected_usd": expected,
            "contingency_usd": expected * COST_CONTINGENCY_MULTIPLIER,
            "ceiling_usd": FUTURE_MODAL_BUDGET_CEILING_USD,
            "forecast_not_charge_or_authorization": True,
        },
    }


def _maximum_same_block_run(order: Sequence[str], ownership: Mapping[str, str]) -> int:
    maximum = 0
    current = 0
    previous: str | None = None
    for case_id in order:
        owner = ownership[case_id]
        current = current + 1 if owner == previous else 1
        maximum = max(maximum, current)
        previous = owner
    return maximum


def validate_frozen_campaign_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the all-at-once freeze required before any Stage M forward."""

    authority = manifest.get("authority")
    if not isinstance(authority, Mapping) or authority.get("track_c_authorized") is not False:
        raise Review21ValidationError("planning manifest must not contain Track C authority")
    model = manifest.get("model")
    if not isinstance(model, Mapping) or (
        model.get("repository") != MODEL_REPOSITORY
        or model.get("revision") != MODEL_REVISION
        or model.get("tokenizer_repository") != TOKENIZER_REPOSITORY
        or model.get("tokenizer_revision") != TOKENIZER_REVISION
    ):
        raise Review21ValidationError("model/tokenizer revision drift in campaign manifest")
    runtime = manifest.get("runtime")
    if not isinstance(runtime, Mapping) or (
        runtime.get("image_definition_sha256") != RUNTIME_IMAGE_DEFINITION_SHA256
        or runtime.get("chat_template_sha256") != CHAT_TEMPLATE_SHA256
        or runtime.get("tokenizer_json_sha256") != TOKENIZER_JSON_SHA256
        or not isinstance(runtime.get("dependency_versions"), Mapping)
        or not runtime.get("dependency_versions")
    ):
        raise Review21ValidationError("runtime/rendering provenance is incomplete or drifted")
    scoring = manifest.get("scoring")
    if not isinstance(scoring, Mapping):
        raise Review21ValidationError("scoring contract is missing")
    correct_id = scoring.get("correct_token_id")
    other_id = scoring.get("other_token_id")
    if (
        not isinstance(scoring.get("score_position"), str)
        or not scoring.get("score_position")
        or scoring.get("correct_is_single_token") is not True
        or scoring.get("other_is_single_token") is not True
        or not isinstance(correct_id, int)
        or not isinstance(other_id, int)
        or correct_id == other_id
    ):
        raise Review21ValidationError("score position or single-token checks are not frozen")

    blocks = manifest.get("blocks")
    if not isinstance(blocks, Sequence) or isinstance(blocks, (str, bytes)) or len(blocks) != PLANNED_BLOCKS:
        raise Review21ValidationError("campaign manifest requires exactly twenty block records")
    block_ids: list[str] = []
    codebook_ids: list[str] = []
    codebook_hashes: list[str] = []
    demonstration_ids: list[str] = []
    seed_hashes: list[str] = []
    seed_values: list[str] = []
    map_case_ids: list[str] = []
    behavior_case_ids: list[str] = []
    map_ownership: dict[str, str] = {}
    behavior_ownership: dict[str, str] = {}
    depth_values: list[int] = []
    for raw_block in blocks:
        if not isinstance(raw_block, Mapping):
            raise Review21ValidationError("block record must be an object")
        block_id = str(raw_block.get("block_id", ""))
        codebook_id = str(raw_block.get("codebook_id", ""))
        codebook_sha256 = str(raw_block.get("codebook_sha256", ""))
        demos = raw_block.get("demonstration_ids")
        seeds = raw_block.get("seeds")
        half_ids = raw_block.get("map_half_ids")
        block_map_ids = raw_block.get("map_case_ids")
        block_behavior_ids = raw_block.get("behavior_case_ids")
        if (
            not block_id
            or not codebook_id
            or re.fullmatch(r"[0-9a-f]{64}", codebook_sha256) is None
        ):
            raise Review21ValidationError("block/codebook identity is missing")
        if raw_block.get("template") != FROZEN_TEMPLATE:
            raise Review21ValidationError("fresh template variants are forbidden")
        if not isinstance(demos, Sequence) or isinstance(demos, (str, bytes)) or not demos:
            raise Review21ValidationError("demonstration identities are missing")
        if not isinstance(seeds, Mapping) or not seeds:
            raise Review21ValidationError("episode and construction seeds are missing")
        if not isinstance(half_ids, Sequence) or len(half_ids) != 2 or len(set(half_ids)) != 2:
            raise Review21ValidationError("two frozen nuisance/map halves are required")
        depth_value = int(raw_block.get("rollout_depth", -1))
        if depth_value not in DEPTH_LEVELS:
            raise Review21ValidationError("block rollout depth left the frozen levels")
        if (
            not isinstance(block_map_ids, Sequence)
            or isinstance(block_map_ids, (str, bytes))
            or len(block_map_ids) != 240
            or len(set(block_map_ids)) != 240
        ):
            raise Review21ValidationError("each block requires 240 unique frozen Stage M case IDs")
        if (
            not isinstance(block_behavior_ids, Sequence)
            or isinstance(block_behavior_ids, (str, bytes))
            or len(block_behavior_ids)
            != 2 * BEHAVIOR_EPISODES_PER_BLOCK * (depth_value + 1)
            or len(set(block_behavior_ids)) != len(block_behavior_ids)
        ):
            raise Review21ValidationError(
                "each block requires its depth-specific unique frozen Stage E case IDs"
            )
        block_ids.append(block_id)
        codebook_ids.append(codebook_id)
        codebook_hashes.append(codebook_sha256)
        demonstration_ids.extend(str(value) for value in demos)
        seed_hashes.append(canonical_sha256(dict(seeds)))
        seed_values.extend(
            json.dumps(value, sort_keys=True, separators=(",", ":"))
            for value in seeds.values()
        )
        depth_values.append(depth_value)
        for case_id in map(str, block_map_ids):
            map_case_ids.append(case_id)
            map_ownership[case_id] = block_id
        for case_id in map(str, block_behavior_ids):
            behavior_case_ids.append(case_id)
            behavior_ownership[case_id] = block_id
    if len(set(block_ids)) != PLANNED_BLOCKS:
        raise Review21ValidationError("block IDs are not unique")
    if len(set(codebook_ids)) != PLANNED_BLOCKS:
        raise Review21ValidationError("codebooks are not fresh across blocks")
    if len(set(codebook_hashes)) != PLANNED_BLOCKS:
        raise Review21ValidationError("codebook payloads are not fresh across blocks")
    if len(set(demonstration_ids)) != len(demonstration_ids):
        raise Review21ValidationError("demonstration identities are reused across blocks")
    if len(set(seed_hashes)) != PLANNED_BLOCKS:
        raise Review21ValidationError("block seed packets are not fresh")
    if len(set(seed_values)) != len(seed_values):
        raise Review21ValidationError("individual episode/construction seeds are reused")
    depth = _depth_vector(depth_values, expected_size=PLANNED_BLOCKS)
    if any(int(np.sum(depth == level)) != PLANNED_BLOCKS_PER_DEPTH for level in DEPTH_LEVELS):
        raise Review21ValidationError("depth allocation is not five blocks per level")
    if len(set(map_case_ids)) != 4_800 or len(set(behavior_case_ids)) != 5_760:
        raise Review21ValidationError("case IDs are duplicated across blocks")
    if set(map_case_ids) & set(behavior_case_ids):
        raise Review21ValidationError("map and behavior case ledgers are not disjoint")

    execution = manifest.get("execution")
    if not isinstance(execution, Mapping):
        raise Review21ValidationError("execution order contract is missing")
    stage_m_order = [str(value) for value in execution.get("stage_m_order", [])]
    stage_e_order = [str(value) for value in execution.get("stage_e_order", [])]
    if len(stage_m_order) != 4_800 or set(stage_m_order) != set(map_case_ids):
        raise Review21ValidationError("Stage M order is not an exact permutation of frozen IDs")
    if len(stage_e_order) != 5_760 or set(stage_e_order) != set(behavior_case_ids):
        raise Review21ValidationError("Stage E order is not an exact permutation of frozen IDs")
    if len(set(stage_m_order)) != len(stage_m_order) or len(set(stage_e_order)) != len(stage_e_order):
        raise Review21ValidationError("execution order duplicates an accepted case ID")
    if _maximum_same_block_run(stage_m_order, map_ownership) > 1:
        raise Review21ValidationError("Stage M is not block-interleaved")
    if _maximum_same_block_run(stage_e_order, behavior_ownership) > 1:
        raise Review21ValidationError("Stage E is not block-interleaved")
    if (
        execution.get("order_seed") != EXECUTION_ORDER_SEED
        or execution.get("order_algorithm") != "SEEDED_BLOCK_INTERLEAVED_SHUFFLE_V1"
        or execution.get("accepted_ids_may_be_duplicated_or_replaced") is not False
        or execution.get("exact_resume_missing_ids_only") is not True
    ):
        raise Review21ValidationError("ordering or exact-resume contract drift")

    analysis = manifest.get("analysis")
    if not isinstance(analysis, Mapping) or (
        analysis.get("permutation_root_seed") != SCIENTIFIC_PERMUTATION_SEED
        or analysis.get("permutations") != SCIENTIFIC_PERMUTATIONS
        or analysis.get("schedule_family_algorithm")
        != "SHA256_ROOT_SEED_AND_ORDERED_QUALIFIED_BLOCK_IDS_V1"
    ):
        raise Review21ValidationError("analysis seed/schedule family is not frozen")
    surface_pairs = manifest.get("path_surface_pairs")
    if (
        not isinstance(surface_pairs, Sequence)
        or isinstance(surface_pairs, (str, bytes))
        or len(surface_pairs) != PLANNED_BLOCKS * BEHAVIOR_EPISODES_PER_BLOCK
    ):
        raise Review21ValidationError("all 480 prospective P/Q episode pairs must be rendered")
    surface_result = validate_path_surface_ledger(surface_pairs)
    surface_block_counts = Counter(result["block_id"] for result in surface_result["pair_results"])
    if set(surface_block_counts) != set(block_ids) or any(
        surface_block_counts[block_id] != BEHAVIOR_EPISODES_PER_BLOCK
        for block_id in block_ids
    ):
        raise Review21ValidationError("path-surface ledger is not exactly 24 episodes per block")
    if surface_result["status"] != "PASS":
        raise AnalysisTerminal(
            "MAP_COMPLETE_NOT_QUALIFIED",
            "exact P/Q surface matching failed before the first forward",
            mismatch_action="REVIEW2_1_BLOCKED_BEFORE_ANY_FORWARD",
        )
    return {
        "status": "PASS",
        "campaign_manifest_sha256": canonical_sha256(dict(manifest)),
        "block_count": PLANNED_BLOCKS,
        "map_case_count": len(map_case_ids),
        "behavior_case_count": len(behavior_case_ids),
        "stage_m_order_sha256": canonical_sha256(stage_m_order),
        "stage_e_order_sha256": canonical_sha256(stage_e_order),
        "path_surface_validation": surface_result,
        "track_c_authorized": False,
    }


CALIBRATION_GENERATOR_VERSION = "review2_1_null_generator_v1"
FINAL_AUDIT_ELIGIBLE_SCENARIOS = (
    "BALANCED_20_NUISANCE_CORRELATED",
    "BALANCED_16_NUISANCE_CORRELATED",
    "INDEPENDENT_ELIGIBLE_QUALIFICATION_MASKS",
    "HETEROSKEDASTIC_WITHIN_FROZEN_DEPTH_STRATA",
    "LEVERAGE_STRESS_BELOW_FROZEN_GATE",
)


def calibration_threshold_lock() -> dict[str, Any]:
    payload = {
        "generator_version": CALIBRATION_GENERATOR_VERSION,
        "depth_encoding": "FOUR_LEVEL_CATEGORICAL_REFERENCE_2",
        "nuisance_columns": [
            "INTERCEPT",
            "DEPTH_4_INDICATOR",
            "DEPTH_6_INDICATOR",
            "DEPTH_8_INDICATOR",
            "MAP_DERIVED_COMPETENCE",
        ],
        "full_additional_column": "REPRESENTATION_AMPLITUDE_R_B",
        "fold_scaling": "TRAINING_FOLD_ONLY",
        "directional_permutation_statistic": (
            "T_PERM_IF_REFITTED_BETA_R_GT_ZERO_ELSE_NEGATIVE_INFINITY"
        ),
        "numerical_relative_floor": NUMERICAL_RELATIVE_FLOOR,
        "condition_ceiling": DESIGN_CONDITION_CEILING,
        "leverage_average_multiplier": LEVERAGE_AVERAGE_MULTIPLIER,
        "leverage_absolute_ceiling": LEVERAGE_ABSOLUTE_CEILING,
        "minimum_press_denominator": MINIMUM_PRESS_DENOMINATOR,
        "minimum_qualified_blocks": MINIMUM_QUALIFIED_BLOCKS,
        "minimum_qualified_per_depth": MINIMUM_QUALIFIED_PER_DEPTH,
        "final_audit_datasets_per_scenario": FINAL_AUDIT_DATASETS_PER_SCENARIO,
        "final_audit_permutations": FINAL_AUDIT_PERMUTATIONS,
        "final_audit_alpha": FINAL_AUDIT_ALPHA,
        "final_audit_fpr_interval": list(FINAL_AUDIT_FPR_INTERVAL),
        "development_seed": DEVELOPMENT_SEED,
        "calibration_permutation_seed": CALIBRATION_PERMUTATION_SEED,
        "final_audit_seed": FINAL_AUDIT_SEED,
        "eligible_scenarios": list(FINAL_AUDIT_ELIGIBLE_SCENARIOS),
        "terminals": sorted(REQUIRED_ANALYSIS_TERMINALS),
    }
    return {"payload": payload, "sha256": canonical_sha256(payload)}


def _planned_depth_vector() -> np.ndarray:
    return np.repeat(np.asarray(DEPTH_LEVELS, dtype=np.int64), PLANNED_BLOCKS_PER_DEPTH)


def _balanced_sixteen_mask() -> np.ndarray:
    mask = np.zeros(PLANNED_BLOCKS, dtype=bool)
    for level in DEPTH_LEVELS:
        indices = np.flatnonzero(_planned_depth_vector() == level)
        mask[indices[:MINIMUM_QUALIFIED_PER_DEPTH]] = True
    return mask


def _independent_eligible_mask(rng: np.random.Generator) -> np.ndarray:
    depth = _planned_depth_vector()
    mask = np.zeros(PLANNED_BLOCKS, dtype=bool)
    for level in DEPTH_LEVELS:
        indices = np.flatnonzero(depth == level)
        qualified_count = int(rng.integers(MINIMUM_QUALIFIED_PER_DEPTH, 6))
        chosen = rng.choice(indices, size=qualified_count, replace=False)
        mask[chosen] = True
    return mask


def _independent_ineligible_mask(rng: np.random.Generator) -> np.ndarray:
    depth = _planned_depth_vector()
    mask = _independent_eligible_mask(rng)
    failed_level = int(rng.choice(np.asarray(DEPTH_LEVELS)))
    indices = np.flatnonzero((depth == failed_level) & mask)
    rng.shuffle(indices)
    mask[indices[: max(1, indices.size - (MINIMUM_QUALIFIED_PER_DEPTH - 1))]] = False
    return mask


def _reported_maximum_leverage(geometry: AnalysisGeometry) -> float:
    diagnostics = geometry.diagnostics
    values = [
        diagnostics["full_cohort_leverage"]["nuisance"]["maximum"],
        diagnostics["full_cohort_leverage"]["full"]["maximum"],
        diagnostics["lobo_folds"]["nuisance"]["maximum_training_fold_leverage"],
        diagnostics["lobo_folds"]["full"]["maximum_training_fold_leverage"],
    ]
    return float(max(values))


def _synthetic_predictor_candidate(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Bounded map-side geometry with competence/representation correlation."""

    competence = np.empty(PLANNED_BLOCKS, dtype=np.float64)
    representation = np.empty(PLANNED_BLOCKS, dtype=np.float64)
    depth = _planned_depth_vector()
    for depth_index, level in enumerate(DEPTH_LEVELS):
        indices = np.flatnonzero(depth == level)
        phase = float(rng.uniform(-math.pi, math.pi))
        angles = 2.0 * math.pi * np.arange(indices.size) / indices.size + phase
        cosine = np.cos(angles) + rng.normal(0.0, 0.035, size=indices.size)
        sine = np.sin(angles) + rng.normal(0.0, 0.035, size=indices.size)
        competence[indices] = 0.30 * depth_index + cosine
        representation[indices] = 4.0 + 0.62 * cosine + 0.78 * sine
    if np.any(representation <= 0.0):
        raise AssertionError("synthetic amplitude generator left the positive domain")
    return competence, representation


def _eligible_synthetic_geometry(
    *,
    predictor_rng: np.random.Generator,
    qualification_mask: np.ndarray,
    leverage_stress: bool,
) -> tuple[np.ndarray, np.ndarray, AnalysisGeometry, int]:
    depth = _planned_depth_vector()
    indices, _ = qualified_block_indices(depth, qualification_mask)
    for attempt in range(1, 401):
        competence, representation = _synthetic_predictor_candidate(predictor_rng)
        if leverage_stress:
            target = int(indices[0])
            severity = float(predictor_rng.uniform(1.70, 3.10))
            competence[target] += severity
            representation[target] += 1.15 * severity
        try:
            geometry = build_analysis_geometry(
                depth[indices],
                competence[indices],
                representation[indices],
            )
        except AnalysisTerminal:
            continue
        maximum_leverage = _reported_maximum_leverage(geometry)
        if leverage_stress and maximum_leverage < 0.68:
            continue
        return competence, representation, geometry, attempt
    raise Review21ValidationError(
        "synthetic predictor generator could not produce an eligible frozen design"
    )


def _scenario_mask(
    scenario: str,
    qualification_rng: np.random.Generator,
) -> np.ndarray:
    if scenario in {
        "BALANCED_20_NUISANCE_CORRELATED",
        "HETEROSKEDASTIC_WITHIN_FROZEN_DEPTH_STRATA",
        "LEVERAGE_STRESS_BELOW_FROZEN_GATE",
    }:
        return np.ones(PLANNED_BLOCKS, dtype=bool)
    if scenario == "BALANCED_16_NUISANCE_CORRELATED":
        return _balanced_sixteen_mask()
    if scenario == "INDEPENDENT_ELIGIBLE_QUALIFICATION_MASKS":
        return _independent_eligible_mask(qualification_rng)
    raise Review21ValidationError(f"unknown calibration scenario: {scenario}")


def _synthetic_null_outcome(
    *,
    scenario: str,
    depth: np.ndarray,
    competence: np.ndarray,
    outcome_rng: np.random.Generator,
) -> np.ndarray:
    depth_effects = {2: -0.45, 4: -0.10, 6: 0.20, 8: 0.55}
    mean = np.asarray([depth_effects[int(level)] for level in depth]) + 0.85 * competence
    if scenario == "HETEROSKEDASTIC_WITHIN_FROZEN_DEPTH_STRATA":
        scales = np.asarray([{2: 0.55, 4: 0.80, 6: 1.10, 8: 1.45}[int(level)] for level in depth])
    else:
        scales = np.ones(depth.size, dtype=np.float64)
    return mean + scales * outcome_rng.normal(0.0, 1.0, size=depth.size)


def _schedule_for_depth(
    depth: np.ndarray,
    *,
    permutations: int,
    cache: dict[tuple[tuple[int, ...], tuple[int, ...]], np.ndarray],
    qualified_positions: Sequence[int],
) -> np.ndarray:
    depth_key = tuple(int(value) for value in depth)
    position_key = tuple(int(value) for value in qualified_positions)
    if len(position_key) != depth.size or len(set(position_key)) != depth.size:
        raise Review21ValidationError("calibration qualification positions are invalid")
    key = (depth_key, position_key)
    if key not in cache:
        seed = derived_schedule_seed(
            root_seed=CALIBRATION_PERMUTATION_SEED,
            qualified_block_ids=[
                f"calibration_block_{position:02d}_depth_{value}"
                for position, value in zip(position_key, depth_key)
            ],
        )
        cache[key] = generate_stratified_permutation_schedule(
            depth,
            permutations=permutations,
            seed=seed,
        )
        validate_permutation_schedule(
            depth,
            cache[key],
            expected_count=permutations,
        )
    return cache[key]


def _run_calibration_scenario(
    *,
    scenario: str,
    datasets: int,
    permutations: int,
    seed: int,
    progress: bool,
) -> dict[str, Any]:
    positives = 0
    terminal_counts: Counter[str] = Counter()
    mask_counts: Counter[tuple[int, ...]] = Counter()
    generator_attempts = 0
    maximum_leverage = 0.0
    schedule_cache: dict[tuple[tuple[int, ...], tuple[int, ...]], np.ndarray] = {}
    depth_all = _planned_depth_vector()
    scenario_index = FINAL_AUDIT_ELIGIBLE_SCENARIOS.index(scenario)
    for dataset_index in range(datasets):
        root = np.random.SeedSequence([int(seed), scenario_index, dataset_index])
        qualification_seed, predictor_seed, outcome_seed = root.spawn(3)
        qualification_rng = np.random.default_rng(qualification_seed)
        predictor_rng = np.random.default_rng(predictor_seed)
        outcome_rng = np.random.default_rng(outcome_seed)
        mask = _scenario_mask(scenario, qualification_rng)
        indices, qualification = qualified_block_indices(depth_all, mask)
        mask_counts[tuple(qualification["depth_counts"].values())] += 1
        competence, _, geometry, attempts = _eligible_synthetic_geometry(
            predictor_rng=predictor_rng,
            qualification_mask=mask,
            leverage_stress=scenario == "LEVERAGE_STRESS_BELOW_FROZEN_GATE",
        )
        generator_attempts += attempts
        maximum_leverage = max(maximum_leverage, _reported_maximum_leverage(geometry))
        outcome = _synthetic_null_outcome(
            scenario=scenario,
            depth=depth_all[indices],
            competence=competence[indices],
            outcome_rng=outcome_rng,
        )
        schedule = _schedule_for_depth(
            geometry.depth,
            permutations=permutations,
            cache=schedule_cache,
            qualified_positions=indices,
        )
        try:
            result = run_primary_pipeline(
                outcome=outcome,
                geometry=geometry,
                schedule=schedule,
                alpha=FINAL_AUDIT_ALPHA,
                validate_schedule=False,
            )
            terminal = str(result["terminal_state"])
        except AnalysisTerminal as exc:
            terminal = exc.state
        terminal_counts[terminal] += 1
        positives += int(terminal == "PRIMARY_POSITIVE")
        if progress and (dataset_index + 1) % max(1, min(100, datasets)) == 0:
            print(
                f"{scenario}: {dataset_index + 1}/{datasets}",
                file=sys.stderr,
                flush=True,
            )
    fpr = positives / datasets
    standard_error = math.sqrt(max(fpr * (1.0 - fpr), 0.0) / datasets)
    lower, upper = FINAL_AUDIT_FPR_INTERVAL
    return {
        "scenario": scenario,
        "datasets": datasets,
        "permutations_per_dataset": permutations,
        "positive_results": positives,
        "empirical_false_positive_rate": fpr,
        "binomial_standard_error": standard_error,
        "acceptance_interval": [lower, upper],
        "status": "PASS" if lower <= fpr <= upper else "FAIL",
        "joint_positive_rule_evaluated": True,
        "terminal_counts": dict(sorted(terminal_counts.items())),
        "qualification_depth_count_patterns": {
            "/".join(str(value) for value in key): count
            for key, count in sorted(mask_counts.items())
        },
        "qualification_rng_separate_from_outcome_rng": True,
        "predictor_generator_attempts": generator_attempts,
        "maximum_eligible_leverage_observed": maximum_leverage,
        "schedule_family_members_used": len(schedule_cache),
    }


def _run_ineligible_mask_audit(*, datasets: int, seed: int) -> dict[str, Any]:
    depth = _planned_depth_vector()
    terminal_counts: Counter[str] = Counter()
    tests_run = 0
    for dataset_index in range(datasets):
        qualification_rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), 99, dataset_index])
        )
        mask = _independent_ineligible_mask(qualification_rng)
        try:
            qualified_block_indices(depth, mask)
        except AnalysisTerminal as exc:
            terminal_counts[exc.state] += 1
        else:
            tests_run += 1
    return {
        "datasets": datasets,
        "outcome_generated": False,
        "primary_tests_run": tests_run,
        "terminal_counts": dict(sorted(terminal_counts.items())),
        "status": "PASS" if tests_run == 0 and sum(terminal_counts.values()) == datasets else "FAIL",
    }


def _run_deterministic_reproducibility_check(*, permutations: int) -> dict[str, Any]:
    def one_run() -> dict[str, Any]:
        root = np.random.SeedSequence([FINAL_AUDIT_SEED, 777, 1])
        qualification_seed, predictor_seed, outcome_seed = root.spawn(3)
        mask = _independent_eligible_mask(np.random.default_rng(qualification_seed))
        indices, _ = qualified_block_indices(_planned_depth_vector(), mask)
        competence, _, geometry, _ = _eligible_synthetic_geometry(
            predictor_rng=np.random.default_rng(predictor_seed),
            qualification_mask=mask,
            leverage_stress=False,
        )
        outcome = _synthetic_null_outcome(
            scenario="INDEPENDENT_ELIGIBLE_QUALIFICATION_MASKS",
            depth=_planned_depth_vector()[indices],
            competence=competence[indices],
            outcome_rng=np.random.default_rng(outcome_seed),
        )
        schedule = _schedule_for_depth(
            geometry.depth,
            permutations=permutations,
            cache={},
            qualified_positions=indices,
        )
        return run_primary_pipeline(
            outcome=outcome,
            geometry=geometry,
            schedule=schedule,
            validate_schedule=False,
        )

    first = one_run()
    second = one_run()
    first_hash = canonical_sha256(first)
    second_hash = canonical_sha256(second)
    return {
        "fixed_seed": FINAL_AUDIT_SEED,
        "first_result_sha256": first_hash,
        "second_result_sha256": second_hash,
        "status": "PASS" if first_hash == second_hash else "FAIL",
    }


def _run_terminal_case_checks() -> dict[str, Any]:
    depth = _planned_depth_vector()
    competence, representation = _synthetic_predictor_candidate(
        np.random.default_rng(np.random.SeedSequence([DEVELOPMENT_SEED, 901]))
    )
    geometry = build_analysis_geometry(depth, competence, representation)
    schedule = generate_stratified_permutation_schedule(depth, permutations=19, seed=901)
    observed: dict[str, str] = {}

    try:
        run_primary_pipeline(
            outcome=np.ones(PLANNED_BLOCKS),
            geometry=geometry,
            schedule=schedule,
        )
    except AnalysisTerminal as exc:
        observed["zero_outcome_variance"] = exc.state

    try:
        build_analysis_geometry(depth, competence, np.ones(PLANNED_BLOCKS))
    except AnalysisTerminal as exc:
        observed["zero_representation_variance"] = exc.state

    try:
        build_analysis_geometry(depth, depth.astype(np.float64), representation)
    except AnalysisTerminal as exc:
        observed["rank_deficient_design"] = exc.state

    exact_nuisance_outcome = (
        0.4
        + 0.2 * (depth == 4)
        - 0.3 * (depth == 6)
        + 0.5 * (depth == 8)
        + 0.7 * competence
    )
    try:
        run_primary_pipeline(
            outcome=exact_nuisance_outcome,
            geometry=geometry,
            schedule=schedule,
        )
    except AnalysisTerminal as exc:
        observed["degenerate_nuisance_sse"] = exc.state

    high_leverage = representation.copy()
    high_leverage[0] += 100.0
    try:
        build_analysis_geometry(depth, competence, high_leverage)
    except AnalysisTerminal as exc:
        observed["excessive_leverage"] = exc.state

    small_depth = np.repeat(np.asarray(DEPTH_LEVELS), 2)
    try:
        generate_stratified_permutation_schedule(
            small_depth,
            permutations=permutation_support_size(small_depth) + 1,
            seed=902,
        )
    except AnalysisTerminal as exc:
        observed["invalid_permutation_support"] = exc.state

    insufficient_depth_mask = np.ones(PLANNED_BLOCKS, dtype=bool)
    insufficient_depth_mask[:2] = False
    try:
        qualified_block_indices(depth, insufficient_depth_mask)
    except AnalysisTerminal as exc:
        observed["insufficient_depth_stratum"] = exc.state

    insufficient_total_mask = np.zeros(PLANNED_BLOCKS, dtype=bool)
    insufficient_total_mask[:15] = True
    try:
        qualified_block_indices(depth, insufficient_total_mask)
    except AnalysisTerminal as exc:
        observed["insufficient_qualified_blocks"] = exc.state

    expected = {
        "zero_outcome_variance": "NO_OUTCOME_VARIANCE",
        "zero_representation_variance": "NO_REPRESENTATION_FEATURE_VARIANCE",
        "rank_deficient_design": "RANK_DEFICIENT_DESIGN",
        "degenerate_nuisance_sse": "DEGENERATE_NUISANCE_SSE",
        "excessive_leverage": "EXCESSIVE_LEVERAGE",
        "invalid_permutation_support": "INVALID_PERMUTATION_SUPPORT",
        "insufficient_depth_stratum": "INSUFFICIENT_DEPTH_STRATUM",
        "insufficient_qualified_blocks": "INSUFFICIENT_QUALIFIED_BLOCKS",
    }
    return {
        "expected": expected,
        "observed": observed,
        "status": "PASS" if observed == expected else "FAIL",
    }


def _run_exact_enumeration_checks() -> dict[str, Any]:
    """Enumerate the entire minimal-eligible 4x4 permutation support."""

    depth_all = _planned_depth_vector()
    mask = _balanced_sixteen_mask()
    indices, _ = qualified_block_indices(depth_all, mask)
    competence, _, geometry, _ = _eligible_synthetic_geometry(
        predictor_rng=np.random.default_rng(np.random.SeedSequence([DEVELOPMENT_SEED, 808])),
        qualification_mask=mask,
        leverage_stress=False,
    )
    schedule = enumerate_stratified_permutations(geometry.depth)
    support = permutation_support_size(geometry.depth)
    if schedule.shape[0] != support:
        raise AssertionError("exact enumeration did not cover the full support")
    case_results = []
    for case_index in (1, 2):
        outcome_rng = np.random.default_rng(
            np.random.SeedSequence([DEVELOPMENT_SEED, 809, case_index])
        )
        outcome = _synthetic_null_outcome(
            scenario="BALANCED_16_NUISANCE_CORRELATED",
            depth=depth_all[indices],
            competence=competence[indices],
            outcome_rng=outcome_rng,
        )
        result = run_primary_pipeline(
            outcome=outcome,
            geometry=geometry,
            schedule=schedule,
            validate_schedule=False,
        )
        reference = brute_force_lobo_statistics(
            outcome=outcome,
            rollout_depth=geometry.depth,
            map_competence=geometry.competence,
            representation_feature=geometry.representation,
        )
        operator_match = all(
            math.isclose(float(result[key]), float(reference[key]), rel_tol=1.0e-10, abs_tol=1.0e-10)
            for key in ("sse_nuisance_lobo", "sse_full_lobo", "t_lobo", "beta_r")
        )
        case_results.append(
            {
                "case": case_index,
                "support_excluding_identity": support,
                "all_nonidentity_permutations_enumerated": True,
                "exact_one_sided_p": result["one_sided_permutation_p"],
                "terminal_state": result["terminal_state"],
                "explicit_refit_matches_linear_operator": operator_match,
                "status": "PASS" if operator_match else "FAIL",
            }
        )
    return {
        "minimal_eligible_structure": "16_BLOCKS_EXACTLY_4_PER_DEPTH",
        "support_excluding_identity": support,
        "cases": case_results,
        "status": "PASS" if all(case["status"] == "PASS" for case in case_results) else "FAIL",
    }


def run_null_calibration(
    *,
    datasets_per_scenario: int,
    permutations: int,
    seed: int,
    final_audit: bool,
    progress: bool = False,
) -> dict[str, Any]:
    if final_audit and (
        datasets_per_scenario != FINAL_AUDIT_DATASETS_PER_SCENARIO
        or permutations != FINAL_AUDIT_PERMUTATIONS
        or seed != FINAL_AUDIT_SEED
    ):
        raise Review21ValidationError("final audit parameters differ from the frozen lock")
    if datasets_per_scenario <= 0 or permutations <= 0:
        raise Review21ValidationError("calibration dimensions must be positive")
    scenario_results = [
        _run_calibration_scenario(
            scenario=scenario,
            datasets=datasets_per_scenario,
            permutations=permutations,
            seed=seed,
            progress=progress,
        )
        for scenario in FINAL_AUDIT_ELIGIBLE_SCENARIOS
    ]
    ineligible = _run_ineligible_mask_audit(
        datasets=datasets_per_scenario,
        seed=seed,
    )
    reproducibility = _run_deterministic_reproducibility_check(
        permutations=permutations,
    )
    terminal_cases = _run_terminal_case_checks()
    exact = _run_exact_enumeration_checks() if final_audit else {
        "status": "NOT_RUN_IN_DEVELOPMENT_MODE",
        "final_audit_only": True,
    }
    required_passes = [result["status"] == "PASS" for result in scenario_results]
    required_passes.extend(
        [
            ineligible["status"] == "PASS",
            reproducibility["status"] == "PASS",
            terminal_cases["status"] == "PASS",
            exact["status"] == "PASS" if final_audit else True,
        ]
    )
    overall_pass = all(required_passes)
    threshold_lock = calibration_threshold_lock()
    return {
        "schema_version": "gate13_track_c_review2_1_null_calibration_v1",
        "audit_mode": "FINAL_AUDIT_RUN_EXACTLY_ONCE" if final_audit else "DEVELOPMENT_ONLY",
        "final_audit_read_only_after_run": bool(final_audit),
        "generator_version": CALIBRATION_GENERATOR_VERSION,
        "datasets_per_eligible_scenario": datasets_per_scenario,
        "permutations_per_dataset": permutations,
        "nominal_alpha": FINAL_AUDIT_ALPHA,
        "frozen_acceptance_interval": list(FINAL_AUDIT_FPR_INTERVAL),
        "audit_seed": seed,
        "qualification_masks_outcome_independent": True,
        "eligible_scenarios": scenario_results,
        "ineligible_mask_audit": ineligible,
        "exact_enumeration_checks": exact,
        "deterministic_reproducibility": reproducibility,
        "deterministic_terminal_case_checks": terminal_cases,
        "threshold_lock": threshold_lock,
        "validator_source_sha256": sha256_file(Path(__file__).resolve()),
        "all_required_conditions_pass": overall_pass,
        "terminal_review2_1_state": (
            "REVIEW2_1_READY_FOR_HUMAN_AUTHORIZATION"
            if overall_pass
            else "REVIEW2_1_BLOCKED"
        ),
        "failure_is_not_replaceable_or_repairable": bool(final_audit),
    }


def _verify_historical_bindings(repo_root: Path) -> dict[str, str]:
    review2_dir = repo_root / "analysis" / "gate13_causal_return" / "review2"
    verified: dict[str, str] = {}
    for relative, expected in REVIEW2_HASHES.items():
        observed = sha256_file(review2_dir / relative)
        if observed != expected:
            raise Review21ValidationError(
                f"historical Review 2 drift: {relative}: {observed} != {expected}"
            )
        verified[f"review2/{relative}"] = observed
    panel_dir = repo_root / "analysis" / "gate13_causal_return" / "checkpoint_panel"
    for relative, expected in PANEL_CLOSEOUT_HASHES.items():
        observed = sha256_file(panel_dir / relative)
        if observed != expected:
            raise Review21ValidationError(
                f"panel closeout drift: {relative}: {observed} != {expected}"
            )
        verified[f"checkpoint_panel/{relative}"] = observed
    return verified


def validate_package(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    package = repo_root / "analysis" / "gate13_causal_return" / "review2_1"
    missing = [relative for relative in REQUIRED_FILES if not (package / relative).is_file()]
    test_path = package / "tests" / "test_track_c_review2_1_validator.py"
    if not test_path.is_file():
        missing.append("tests/test_track_c_review2_1_validator.py")
    if missing:
        raise Review21ValidationError(f"Review 2.1 package is incomplete: {missing}")
    historical = _verify_historical_bindings(repo_root)
    lock = _load_json(package / "track_c_estimand_lock_candidate_v2.json")
    calibration = _load_json(package / "track_c_null_calibration.json")
    if lock.get("schema_version") != "gate13_track_c_estimand_lock_candidate_v2":
        raise Review21ValidationError("estimand lock schema drift")
    if lock.get("historical_review2_binding", {}).get("commit") != REVIEW2_COMMIT:
        raise Review21ValidationError("estimand lock lost the historical Review 2 commit")
    model_target = lock.get("model_target", {})
    if (
        model_target.get("repository") != MODEL_REPOSITORY
        or model_target.get("revision") != MODEL_REVISION
        or model_target.get("tokenizer_revision") != TOKENIZER_REVISION
        or model_target.get("layers") != list(FROZEN_LAYERS)
    ):
        raise Review21ValidationError("estimand lock model/layer target drift")
    design = lock.get("prospective_design", {})
    if (
        design.get("analysis_unit") != "ONE_INDEPENDENT_FRESH_NATURALITY_SQUARE_BLOCK"
        or design.get("planned_blocks") != PLANNED_BLOCKS
        or design.get("planned_blocks_per_depth") != PLANNED_BLOCKS_PER_DEPTH
        or design.get("minimum_qualified_blocks") != MINIMUM_QUALIFIED_BLOCKS
        or design.get("minimum_qualified_per_depth") != MINIMUM_QUALIFIED_PER_DEPTH
        or design.get("rollout_depth_levels") != list(DEPTH_LEVELS)
    ):
        raise Review21ValidationError("estimand lock block/depth design drift")
    representation = lock.get("representation_observable", {})
    if (
        representation.get("primary_feature") != "SQUARE_ROOT_AMPLITUDE_R_B"
        or representation.get("frame_rank") != FRAME_RANK
        or representation.get("layers") != list(FROZEN_LAYERS)
        or representation.get("opposite_half_source_activations_required") is not True
        or representation.get("incompatible_half_gauge_multiplication") != "FORBIDDEN"
    ):
        raise Review21ValidationError("estimand lock representation observable drift")
    nuisance = lock.get("map_derived_competence", {})
    if (
        nuisance.get("symbol") != "C_b^M"
        or nuisance.get("behavior_ledger_rows_used") != 0
        or nuisance.get("broken_square_rows_used") != 0
        or nuisance.get("required_exact_map_rows_per_block") != 192
    ):
        raise Review21ValidationError("estimand lock nuisance leakage or schema drift")
    analysis_lock = lock.get("analysis", {})
    if (
        analysis_lock.get("depth_encoding") != "FOUR_LEVEL_CATEGORICAL_REFERENCE_2"
        or analysis_lock.get("fold_scaling") != "FIT_ON_EACH_LOBO_TRAINING_FOLD_ONLY"
        or analysis_lock.get("permutations") != SCIENTIFIC_PERMUTATIONS
        or analysis_lock.get("permutation_root_seed") != SCIENTIFIC_PERMUTATION_SEED
        or analysis_lock.get("positive_rule")
        != "ALL_GATES_PASS_AND_T_LOBO_GT_0_AND_BETA_R_GT_0_AND_ONE_SIDED_P_LTE_0.05"
    ):
        raise Review21ValidationError("estimand lock primary analysis drift")
    configured_terminals = set(lock.get("analysis", {}).get("terminal_states", []))
    if not REQUIRED_ANALYSIS_TERMINALS.issubset(configured_terminals):
        raise Review21ValidationError("estimand lock omits a required fail-closed terminal")
    authority = lock.get("authority", {})
    forbidden_flags = (
        "track_c_authorized",
        "modal_called",
        "gpu_allocated",
        "model_loaded",
        "model_forward_performed",
        "activation_collection_performed",
        "track_c_outcome_inspected",
    )
    if any(authority.get(flag) is not False for flag in forbidden_flags):
        raise Review21ValidationError("estimand lock contains execution authority or activity")
    if calibration.get("audit_mode") != "FINAL_AUDIT_RUN_EXACTLY_ONCE":
        raise Review21ValidationError("the committed calibration is not the frozen final audit")
    if (
        calibration.get("datasets_per_eligible_scenario")
        != FINAL_AUDIT_DATASETS_PER_SCENARIO
        or calibration.get("permutations_per_dataset") != FINAL_AUDIT_PERMUTATIONS
        or calibration.get("audit_seed") != FINAL_AUDIT_SEED
        or calibration.get("frozen_acceptance_interval")
        != list(FINAL_AUDIT_FPR_INTERVAL)
    ):
        raise Review21ValidationError("final audit dimensions or seed drift")
    if calibration.get("validator_source_sha256") != sha256_file(Path(__file__).resolve()):
        raise Review21ValidationError("validator changed after the final audit")
    threshold_lock = calibration_threshold_lock()
    if calibration.get("threshold_lock", {}).get("sha256") != threshold_lock["sha256"]:
        raise Review21ValidationError("calibration threshold lock drift")
    scenario_results = calibration.get("eligible_scenarios")
    if not isinstance(scenario_results, list) or [
        result.get("scenario") for result in scenario_results if isinstance(result, Mapping)
    ] != list(FINAL_AUDIT_ELIGIBLE_SCENARIOS):
        raise Review21ValidationError("final audit scenario inventory drift")
    scenario_pass = all(
        isinstance(result, Mapping)
        and result.get("status") == "PASS"
        and FINAL_AUDIT_FPR_INTERVAL[0]
        <= float(result.get("empirical_false_positive_rate", math.nan))
        <= FINAL_AUDIT_FPR_INTERVAL[1]
        for result in scenario_results
    )
    ineligible = calibration.get("ineligible_mask_audit", {})
    ancillary_pass = all(
        calibration.get(key, {}).get("status") == "PASS"
        for key in (
            "exact_enumeration_checks",
            "deterministic_reproducibility",
            "deterministic_terminal_case_checks",
        )
    )
    ancillary_pass = ancillary_pass and (
        ineligible.get("status") == "PASS"
        and ineligible.get("primary_tests_run") == 0
        and ineligible.get("outcome_generated") is False
    )
    recomputed_audit_pass = scenario_pass and ancillary_pass
    if calibration.get("all_required_conditions_pass") is not recomputed_audit_pass:
        raise Review21ValidationError("final audit pass flag does not match its results")
    expected_state = (
        "REVIEW2_1_READY_FOR_HUMAN_AUTHORIZATION"
        if recomputed_audit_pass
        else "REVIEW2_1_BLOCKED"
    )
    if calibration.get("terminal_review2_1_state") != expected_state:
        raise Review21ValidationError("calibration terminal state is internally inconsistent")
    if lock.get("terminal_review2_1_state") != expected_state:
        raise Review21ValidationError("estimand lock and final audit terminal states disagree")
    inventory = {
        relative: sha256_file(package / relative)
        for relative in (*REQUIRED_FILES, "tests/test_track_c_review2_1_validator.py")
    }
    return {
        "status": "PASS",
        "terminal_review2_1_state": expected_state,
        "historical_bindings": historical,
        "artifact_inventory": inventory,
        "artifact_count": len(inventory),
    }


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    development = subparsers.add_parser("development-calibration")
    development.add_argument("--datasets", type=int, default=200)
    development.add_argument("--permutations", type=int, default=199)
    development.add_argument("--progress", action="store_true")
    final = subparsers.add_parser("final-audit")
    final.add_argument("--progress", action="store_true")
    package = subparsers.add_parser("validate-package")
    package.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args(argv)
    try:
        if args.command == "development-calibration":
            result = run_null_calibration(
                datasets_per_scenario=args.datasets,
                permutations=args.permutations,
                seed=DEVELOPMENT_SEED,
                final_audit=False,
                progress=args.progress,
            )
        elif args.command == "final-audit":
            result = run_null_calibration(
                datasets_per_scenario=FINAL_AUDIT_DATASETS_PER_SCENARIO,
                permutations=FINAL_AUDIT_PERMUTATIONS,
                seed=FINAL_AUDIT_SEED,
                final_audit=True,
                progress=args.progress,
            )
        else:
            result = validate_package(args.repo_root)
    except (Review21ValidationError, AnalysisTerminal, OSError, json.JSONDecodeError) as exc:
        payload = {
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        if isinstance(exc, AnalysisTerminal):
            payload["terminal_state"] = exc.state
            payload["details"] = exc.details
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
