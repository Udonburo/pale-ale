#!/usr/bin/env python3
"""Build Gate4RunInputV1 using a native local span boundary candidate."""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import build_gate4_input as packer


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = "runs/native_local_span_boundary"
DEFAULT_GATE4_INPUT = "gate4_input.json"
DEFAULT_SELECTION_MANIFEST = "native_local_span_selection_manifest.json"
DEFAULT_BUILD_MANIFEST = "native_local_span_build_manifest.json"
DEFAULT_BOUNDARY_STEPS = "native_local_span_boundary_steps.ndjson"

BOUNDARY_ID_ANCHORED_V0 = "native_local_span_anchored_projection_v0"
BOUNDARY_ID_CENTERED_AFFINE_V1 = "native_local_span_centered_affine_v1"
BOUNDARY_ID_ORIGIN_SPAN_V2 = "native_local_span_origin_span_v2"
BOUNDARY_ID_RELATION_AFFINE_LIFT_V0 = "local_relation_affine_lift_v0"
BOUNDARY_ID_RELATION_AFFINE_LIFT_V1 = "local_relation_affine_lift_v1"
BOUNDARY_ID_RELATION_AFFINE_LIFT_V2 = "local_relation_affine_lift_v2"
BOUNDARY_ID_RELATION_AFFINE_LIFT_V3 = "local_relation_affine_lift_v3"
BOUNDARY_ID_RELATION_AFFINE_LIFT_V4 = "local_relation_affine_lift_v4"
FRAME_CONSTRUCTION_ID_ANCHORED = "anchor_v_diff_gram_schmidt_v0"
FRAME_CONSTRUCTION_ID_ORIGIN_SPAN = "raw_triplet_origin_gram_schmidt_v1"
FRAME_CONSTRUCTION_ID_RELATION_AFFINE = "relation_affine_lift_chart_v0"
FRAME_CONSTRUCTION_ID_RELATION_AFFINE_V1 = "relation_affine_origin_span_midrange_lift_v1"
FRAME_CONSTRUCTION_ID_RELATION_AFFINE_V2 = "relation_affine_angle_profile_origin_span_modulation_v2"
FRAME_CONSTRUCTION_ID_RELATION_AFFINE_V3 = (
    "relation_affine_angle_profile_origin_span_modulation_gated_v3"
)
FRAME_CONSTRUCTION_ID_RELATION_AFFINE_V4 = (
    "relation_affine_angle_profile_origin_span_modulation_capped_v4"
)
SOURCE_TENSOR_ID = "triality_raw_triplet_preprojection_v1"
COORDINATE_SPACE_ID_ANCHORED_V0 = "native_local_span_coordinates_pad8_v0"
COORDINATE_SPACE_ID_CENTERED_AFFINE_V1 = "native_local_span_coordinates_centered_affine_pad8_v1"
COORDINATE_SPACE_ID_ORIGIN_SPAN_V2 = "native_local_span_coordinates_origin_span_pad8_v2"
COORDINATE_SPACE_ID_RELATION_AFFINE_LIFT_V0 = "local_relation_affine_lift_coordinates_pad8_v0"
COORDINATE_SPACE_ID_RELATION_AFFINE_LIFT_V1 = "local_relation_affine_lift_coordinates_pad8_v1"
COORDINATE_SPACE_ID_RELATION_AFFINE_LIFT_V2 = "local_relation_affine_lift_coordinates_pad8_v2"
COORDINATE_SPACE_ID_RELATION_AFFINE_LIFT_V3 = "local_relation_affine_lift_coordinates_pad8_v3"
COORDINATE_SPACE_ID_RELATION_AFFINE_LIFT_V4 = "local_relation_affine_lift_coordinates_pad8_v4"
COORDINATE_RULE_ANCHORED_V0 = "anchored_projection_v0"
COORDINATE_RULE_CENTERED_AFFINE_V1 = "centered_affine_local_span_v1"
COORDINATE_RULE_ORIGIN_SPAN_V2 = "origin_span_projection_v2"
COORDINATE_RULE_RELATION_AFFINE_LIFT_V0 = "local_relation_affine_lift_v0"
COORDINATE_RULE_RELATION_AFFINE_LIFT_V1 = "local_relation_affine_lift_v1"
COORDINATE_RULE_RELATION_AFFINE_LIFT_V2 = "local_relation_affine_lift_v2"
COORDINATE_RULE_RELATION_AFFINE_LIFT_V3 = "local_relation_affine_lift_v3"
COORDINATE_RULE_RELATION_AFFINE_LIFT_V4 = "local_relation_affine_lift_v4"
BASIS_RULE_ID_ANCHORED = "v_anchor_diff_gram_schmidt_v0"
BASIS_RULE_ID_ORIGIN_SPAN = "raw_triplet_origin_gram_schmidt_v1"
BASIS_RULE_ID_RELATION_AFFINE = "v_anchor_relation_affine_lift_v0"
BASIS_RULE_ID_RELATION_AFFINE_V1 = "v_anchor_relation_chart_plus_origin_span_e3_v1"
BASIS_RULE_ID_RELATION_AFFINE_V2 = (
    "v_anchor_relation_chart_angle_profile_plus_origin_span_modulation_v2"
)
BASIS_RULE_ID_RELATION_AFFINE_V3 = (
    "v_anchor_relation_chart_angle_profile_plus_origin_span_modulation_gated_v3"
)
BASIS_RULE_ID_RELATION_AFFINE_V4 = (
    "v_anchor_relation_chart_angle_profile_plus_origin_span_modulation_capped_v4"
)
PROJECTION_OR_SPAN_RULE_ANCHORED_V0 = "unit_source_projection_onto_local_span_pad8_v0"
PROJECTION_OR_SPAN_RULE_CENTERED_AFFINE_V1 = "centered_affine_projection_onto_local_span_pad8_v1"
PROJECTION_OR_SPAN_RULE_ORIGIN_SPAN_V2 = (
    "unit_source_projection_onto_raw_triplet_origin_span_pad8_v2"
)
PROJECTION_OR_SPAN_RULE_RELATION_AFFINE_LIFT_V0 = (
    "canonical_triangle_centroid_altitude_lift_pad8_v0"
)
PROJECTION_OR_SPAN_RULE_RELATION_AFFINE_LIFT_V1 = (
    "canonical_triangle_centroid_midrange_centered_origin_span_e3_pad8_v1"
)
PROJECTION_OR_SPAN_RULE_RELATION_AFFINE_LIFT_V2 = (
    "canonical_triangle_centroid_angle_profile_origin_span_modulation_pad8_v2"
)
PROJECTION_OR_SPAN_RULE_RELATION_AFFINE_LIFT_V3 = (
    "canonical_triangle_centroid_angle_profile_origin_span_modulation_gated_pad8_v3"
)
PROJECTION_OR_SPAN_RULE_RELATION_AFFINE_LIFT_V4 = (
    "canonical_triangle_centroid_angle_profile_origin_span_modulation_capped_pad8_v4"
)
BASIS_SIGN_RULE_ID = "first_non_negligible_positive_v0"
BASIS_ORDER_RULE_ID = "construction_order_v0"
ORIENTATION_RULE_ID = "construction_order_parity_v0"
DEGENERACY_POLICY_ID = "honest_variable_rank_no_fake_completion_v0"
RAW_NATIVE_SCHEMA_ID = "triality_raw_native_v1"
RAW_KEYS = ("V_raw_native", "Splus_raw_native", "Sminus_raw_native")
FRAME_EPS = 1e-6
SIGN_STABILITY_EPS = 1e-4
RELATION_AREA_EPS = 1e-6
EMIT_DIM = 8
RAW_SPAN_MODULATION_ALPHA_V2 = 0.25
RAW_SPAN_MODULATION_ALPHA_V3 = 0.25
RAW_SPAN_MODULATION_ALPHA_V4 = 0.25
RAW_SPAN_Z_CAP_MULTIPLIER_V4 = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transform raw native triality triplets into a Gate4-compatible "
            "native-local-span boundary candidate."
        )
    )
    parser.add_argument("--samples-root", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--sample-ids", nargs="+", type=int)
    selection.add_argument("--sample-id-file")
    selection.add_argument("--all-samples", action="store_true")
    parser.add_argument(
        "--variant",
        choices=("consistent", "frustrated", "unknown"),
        help="Optional variant filter applied after sample discovery.",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--coordinate-rule",
        choices=(
            COORDINATE_RULE_ANCHORED_V0,
            COORDINATE_RULE_CENTERED_AFFINE_V1,
            COORDINATE_RULE_ORIGIN_SPAN_V2,
            COORDINATE_RULE_RELATION_AFFINE_LIFT_V0,
            COORDINATE_RULE_RELATION_AFFINE_LIFT_V1,
            COORDINATE_RULE_RELATION_AFFINE_LIFT_V2,
            COORDINATE_RULE_RELATION_AFFINE_LIFT_V3,
            COORDINATE_RULE_RELATION_AFFINE_LIFT_V4,
        ),
        default=COORDINATE_RULE_ANCHORED_V0,
    )
    parser.add_argument("--perm-r", type=int, default=2000)
    parser.add_argument("--primary-score", default="E")
    parser.add_argument("--script-extract", default="tools/extract_triality_triplets.py")
    parser.add_argument("--script-eval", default="tools/eval_triality_token.py")
    return parser.parse_args()


def repo_relative_or_posix(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def ensure_finite_vector(values: Sequence[float], label: str) -> List[float]:
    out = [float(value) for value in values]
    for idx, value in enumerate(out):
        if not math.isfinite(value):
            raise ValueError(f"{label}[{idx}] is non-finite")
    return out


def l2_norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def subtract(lhs: Sequence[float], rhs: Sequence[float]) -> List[float]:
    if len(lhs) != len(rhs):
        raise ValueError("subtract requires equal lengths")
    return [float(a) - float(b) for a, b in zip(lhs, rhs)]


def dot(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    if len(lhs) != len(rhs):
        raise ValueError("dot requires equal lengths")
    return sum(float(a) * float(b) for a, b in zip(lhs, rhs))


def scale(values: Sequence[float], scalar: float) -> List[float]:
    return [float(scalar) * float(value) for value in values]


def add(lhs: Sequence[float], rhs: Sequence[float]) -> List[float]:
    if len(lhs) != len(rhs):
        raise ValueError("add requires equal lengths")
    return [float(a) + float(b) for a, b in zip(lhs, rhs)]


def normalize(values: Sequence[float]) -> Optional[List[float]]:
    norm = l2_norm(values)
    if not math.isfinite(norm) or norm <= FRAME_EPS:
        return None
    return [float(value) / norm for value in values]


def fix_sign(values: Sequence[float]) -> Tuple[List[float], int]:
    out = [float(value) for value in values]
    for value in out:
        if abs(value) > FRAME_EPS:
            if value < 0.0:
                return ([-entry for entry in out], -1)
            return (out, 1)
    return (out, 1)


def fix_sign_with_stability(
    values: Sequence[float],
) -> Tuple[List[float], int, bool, Optional[int], float]:
    out = [float(value) for value in values]
    for idx, value in enumerate(out):
        magnitude = abs(value)
        if magnitude > FRAME_EPS:
            sign = 1
            if value < 0.0:
                out = [-entry for entry in out]
                sign = -1
            return (out, sign, magnitude >= SIGN_STABILITY_EPS, idx, magnitude)
    return (out, 1, False, None, 0.0)


def orthogonalize(
    direction: Sequence[float], basis: Sequence[Sequence[float]]
) -> Tuple[Optional[List[float]], int]:
    residual = [float(value) for value in direction]
    for axis in basis:
        residual = add(residual, scale(axis, -dot(residual, axis)))
    normalized = normalize(residual)
    if normalized is None:
        return (None, 0)
    return fix_sign(normalized)


def orthogonalize_with_stability(
    direction: Sequence[float], basis: Sequence[Sequence[float]]
) -> Tuple[Optional[List[float]], int, bool, Optional[int], float]:
    residual = [float(value) for value in direction]
    for axis in basis:
        residual = add(residual, scale(axis, -dot(residual, axis)))
    normalized = normalize(residual)
    if normalized is None:
        return (None, 0, False, None, 0.0)
    return fix_sign_with_stability(normalized)


def pad8(values: Sequence[float]) -> List[float]:
    out = [0.0] * EMIT_DIM
    for idx, value in enumerate(values[:EMIT_DIM]):
        out[idx] = float(value)
    return out


def project_into_basis(source: Sequence[float], basis: Sequence[Sequence[float]]) -> List[float]:
    return [dot(source, axis) for axis in basis]


def boundary_outcome_from_rank(rank: int) -> str:
    if rank == 1:
        return "materialized_rank1"
    if rank == 2:
        return "materialized_rank2"
    if rank == 3:
        return "materialized_rank3"
    raise ValueError(f"unsupported materialized rank: {rank}")


def coordinate_rule_metadata(rule: str) -> Dict[str, str]:
    if rule == COORDINATE_RULE_ANCHORED_V0:
        return {
            "boundary_id": BOUNDARY_ID_ANCHORED_V0,
            "coordinate_space_id": COORDINATE_SPACE_ID_ANCHORED_V0,
            "projection_or_span_rule": PROJECTION_OR_SPAN_RULE_ANCHORED_V0,
            "frame_construction_id": FRAME_CONSTRUCTION_ID_ANCHORED,
            "basis_rule_id": BASIS_RULE_ID_ANCHORED,
        }
    if rule == COORDINATE_RULE_CENTERED_AFFINE_V1:
        return {
            "boundary_id": BOUNDARY_ID_CENTERED_AFFINE_V1,
            "coordinate_space_id": COORDINATE_SPACE_ID_CENTERED_AFFINE_V1,
            "projection_or_span_rule": PROJECTION_OR_SPAN_RULE_CENTERED_AFFINE_V1,
            "frame_construction_id": FRAME_CONSTRUCTION_ID_ANCHORED,
            "basis_rule_id": BASIS_RULE_ID_ANCHORED,
        }
    if rule == COORDINATE_RULE_ORIGIN_SPAN_V2:
        return {
            "boundary_id": BOUNDARY_ID_ORIGIN_SPAN_V2,
            "coordinate_space_id": COORDINATE_SPACE_ID_ORIGIN_SPAN_V2,
            "projection_or_span_rule": PROJECTION_OR_SPAN_RULE_ORIGIN_SPAN_V2,
            "frame_construction_id": FRAME_CONSTRUCTION_ID_ORIGIN_SPAN,
            "basis_rule_id": BASIS_RULE_ID_ORIGIN_SPAN,
        }
    if rule == COORDINATE_RULE_RELATION_AFFINE_LIFT_V0:
        return {
            "boundary_id": BOUNDARY_ID_RELATION_AFFINE_LIFT_V0,
            "coordinate_space_id": COORDINATE_SPACE_ID_RELATION_AFFINE_LIFT_V0,
            "projection_or_span_rule": PROJECTION_OR_SPAN_RULE_RELATION_AFFINE_LIFT_V0,
            "frame_construction_id": FRAME_CONSTRUCTION_ID_RELATION_AFFINE,
            "basis_rule_id": BASIS_RULE_ID_RELATION_AFFINE,
        }
    if rule == COORDINATE_RULE_RELATION_AFFINE_LIFT_V1:
        return {
            "boundary_id": BOUNDARY_ID_RELATION_AFFINE_LIFT_V1,
            "coordinate_space_id": COORDINATE_SPACE_ID_RELATION_AFFINE_LIFT_V1,
            "projection_or_span_rule": PROJECTION_OR_SPAN_RULE_RELATION_AFFINE_LIFT_V1,
            "frame_construction_id": FRAME_CONSTRUCTION_ID_RELATION_AFFINE_V1,
            "basis_rule_id": BASIS_RULE_ID_RELATION_AFFINE_V1,
        }
    if rule == COORDINATE_RULE_RELATION_AFFINE_LIFT_V2:
        return {
            "boundary_id": BOUNDARY_ID_RELATION_AFFINE_LIFT_V2,
            "coordinate_space_id": COORDINATE_SPACE_ID_RELATION_AFFINE_LIFT_V2,
            "projection_or_span_rule": PROJECTION_OR_SPAN_RULE_RELATION_AFFINE_LIFT_V2,
            "frame_construction_id": FRAME_CONSTRUCTION_ID_RELATION_AFFINE_V2,
            "basis_rule_id": BASIS_RULE_ID_RELATION_AFFINE_V2,
        }
    if rule == COORDINATE_RULE_RELATION_AFFINE_LIFT_V3:
        return {
            "boundary_id": BOUNDARY_ID_RELATION_AFFINE_LIFT_V3,
            "coordinate_space_id": COORDINATE_SPACE_ID_RELATION_AFFINE_LIFT_V3,
            "projection_or_span_rule": PROJECTION_OR_SPAN_RULE_RELATION_AFFINE_LIFT_V3,
            "frame_construction_id": FRAME_CONSTRUCTION_ID_RELATION_AFFINE_V3,
            "basis_rule_id": BASIS_RULE_ID_RELATION_AFFINE_V3,
        }
    if rule == COORDINATE_RULE_RELATION_AFFINE_LIFT_V4:
        return {
            "boundary_id": BOUNDARY_ID_RELATION_AFFINE_LIFT_V4,
            "coordinate_space_id": COORDINATE_SPACE_ID_RELATION_AFFINE_LIFT_V4,
            "projection_or_span_rule": PROJECTION_OR_SPAN_RULE_RELATION_AFFINE_LIFT_V4,
            "frame_construction_id": FRAME_CONSTRUCTION_ID_RELATION_AFFINE_V4,
            "basis_rule_id": BASIS_RULE_ID_RELATION_AFFINE_V4,
        }
    raise ValueError(f"unsupported coordinate rule: {rule}")


def centroid(values: Sequence[Sequence[float]]) -> List[float]:
    if not values:
        raise ValueError("centroid requires non-empty input")
    width = len(values[0])
    out = [0.0] * width
    for value in values:
        if len(value) != width:
            raise ValueError("centroid requires equal-length vectors")
        for idx, entry in enumerate(value):
            out[idx] += float(entry)
    inv = 1.0 / float(len(values))
    return [entry * inv for entry in out]


def centered_energy(values: Sequence[Sequence[float]], center: Sequence[float]) -> float:
    if not values:
        return 0.0
    total = 0.0
    for value in values:
        total += l2_norm(subtract(value, center))
    return total / float(len(values))


def gram_schmidt_rank(values: Sequence[Sequence[float]]) -> int:
    basis: List[List[float]] = []
    for value in values:
        axis, _ = orthogonalize(value, basis)
        if axis is None:
            continue
        basis.append(axis)
    return len(basis)


def clamped_cosine(left: Sequence[float], right: Sequence[float]) -> float:
    left_norm = l2_norm(left)
    right_norm = l2_norm(right)
    if left_norm <= FRAME_EPS or right_norm <= FRAME_EPS:
        return 0.0
    value = dot(left, right) / (left_norm * right_norm)
    return max(-1.0, min(1.0, float(value)))


def relation_ambiguity_gate(angle_values: Sequence[float]) -> float:
    if not angle_values:
        return 0.0
    spread = max(float(value) for value in angle_values) - min(float(value) for value in angle_values)
    normalized_spread = max(0.0, min(1.0, 0.5 * spread))
    return 1.0 - normalized_spread


def clamp_symmetric(value: float, abs_limit: float) -> float:
    if abs_limit <= 0.0:
        return 0.0
    return max(-abs_limit, min(abs_limit, float(value)))


def load_raw_triplet(row: Dict[str, Any], key: str) -> List[float]:
    if key not in row:
        raise KeyError(f"triplet row missing required raw native field {key!r}")
    raw = row.get(key)
    if not isinstance(raw, list):
        raise TypeError(f"triplet row field {key!r} is not a list")
    return ensure_finite_vector(raw, key)


def build_v_anchor_basis(
    v_unit: Sequence[float],
    splus_unit: Sequence[float],
    sminus_unit: Sequence[float],
) -> Tuple[List[List[float]], List[str], int]:
    basis: List[List[float]] = []
    basis_sources: List[str] = []
    orientation_parity = 1
    for label, direction in (
        ("d1", subtract(splus_unit, v_unit)),
        ("d2", subtract(sminus_unit, v_unit)),
    ):
        axis, sign = orthogonalize(direction, basis)
        if axis is None:
            continue
        basis.append(axis)
        basis_sources.append(label)
        orientation_parity *= sign
    return basis, basis_sources, orientation_parity


def build_origin_span_basis(
    v_unit: Sequence[float],
    splus_unit: Sequence[float],
    sminus_unit: Sequence[float],
) -> Tuple[List[List[float]], List[str], int]:
    basis: List[List[float]] = []
    basis_sources: List[str] = []
    orientation_parity = 1
    for label, direction in (
        ("v", v_unit),
        ("splus", splus_unit),
        ("sminus", sminus_unit),
    ):
        axis, sign = orthogonalize(direction, basis)
        if axis is None:
            continue
        basis.append(axis)
        basis_sources.append(label)
        orientation_parity *= sign
    return basis, basis_sources, orientation_parity


def build_origin_span_e3_axis(
    v_unit: Sequence[float],
    splus_unit: Sequence[float],
    sminus_unit: Sequence[float],
) -> Tuple[Optional[List[float]], int, bool, Optional[int], float]:
    e_origin_v, _sign_v = orthogonalize(v_unit, [])
    if e_origin_v is None:
        return (None, 0, False, None, 0.0)
    e_origin_splus, _sign_splus = orthogonalize(splus_unit, [e_origin_v])
    if e_origin_splus is None:
        return (None, 0, False, None, 0.0)
    return orthogonalize_with_stability(sminus_unit, [e_origin_v, e_origin_splus])


def build_relation_affine_lift_coordinates_with_mode(
    v_unit: Sequence[float],
    splus_unit: Sequence[float],
    sminus_unit: Sequence[float],
    *,
    lift_mode: str,
) -> Dict[str, Any]:
    edge_len_v_splus = float(l2_norm(subtract(splus_unit, v_unit)))
    edge_len_v_sminus = float(l2_norm(subtract(sminus_unit, v_unit)))
    edge_len_splus_sminus = float(l2_norm(subtract(splus_unit, sminus_unit)))
    zero_result = {
        "coords_v": [0.0] * EMIT_DIM,
        "coords_splus": [0.0] * EMIT_DIM,
        "coords_sminus": [0.0] * EMIT_DIM,
        "frame_rank": 0,
        "orientation_parity": 0,
        "basis_sources": [],
        "projected_norm_v": 0.0,
        "projected_norm_splus": 0.0,
        "projected_norm_sminus": 0.0,
        "raw_triplet_centroid_norm": 0.0,
        "emitted_coord_centroid_norm": 0.0,
        "emitted_centered_energy": 0.0,
        "relation_signed_area2": 0.0,
        "relation_plane_height_signed": 0.0,
        "relation_edge_len_v_splus": edge_len_v_splus,
        "relation_edge_len_v_sminus": edge_len_v_sminus,
        "relation_edge_len_splus_sminus": edge_len_splus_sminus,
        "relation_altitude_v": 0.0,
        "relation_altitude_splus": 0.0,
        "relation_altitude_sminus": 0.0,
        "relation_angle_cos_v": 0.0,
        "relation_angle_cos_splus": 0.0,
        "relation_angle_cos_sminus": 0.0,
        "relation_lift_rank": 0,
        "sign_anchor_index_e1": None,
        "sign_anchor_index_e2": None,
        "sign_anchor_index_e3": None,
        "sign_anchor_abs_e1": 0.0,
        "sign_anchor_abs_e2": 0.0,
        "sign_anchor_abs_e3": 0.0,
        "raw_span_lift_center": 0.0,
        "raw_span_lift_range": 0.0,
        "raw_span_axis_available": False,
        "raw_span_modulation_alpha": 0.0,
    }

    def set_partial_state(
        *,
        frame_rank: int,
        basis_sources: Sequence[str],
        orientation_parity: int,
        altitude_values: Optional[Tuple[float, float, float]] = None,
        angle_values: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        zero_result["frame_rank"] = int(frame_rank)
        zero_result["basis_sources"] = list(basis_sources)
        zero_result["orientation_parity"] = int(orientation_parity)
        if altitude_values is not None:
            zero_result["relation_altitude_v"] = float(altitude_values[0])
            zero_result["relation_altitude_splus"] = float(altitude_values[1])
            zero_result["relation_altitude_sminus"] = float(altitude_values[2])
        if angle_values is not None:
            zero_result["relation_angle_cos_v"] = float(angle_values[0])
            zero_result["relation_angle_cos_splus"] = float(angle_values[1])
            zero_result["relation_angle_cos_sminus"] = float(angle_values[2])

    if edge_len_v_splus <= FRAME_EPS or edge_len_v_sminus <= FRAME_EPS:
        zero_result["boundary_outcome"] = "near_collinear"
        return zero_result

    e1, sign_e1, stable_e1, anchor_idx_e1, anchor_abs_e1 = orthogonalize_with_stability(
        subtract(splus_unit, v_unit), []
    )
    zero_result["sign_anchor_index_e1"] = anchor_idx_e1
    zero_result["sign_anchor_abs_e1"] = float(anchor_abs_e1)
    if e1 is None:
        zero_result["boundary_outcome"] = "near_collinear"
        return zero_result
    if not stable_e1:
        zero_result["boundary_outcome"] = "sign_unstable"
        return zero_result
    set_partial_state(frame_rank=1, basis_sources=["d1"], orientation_parity=0)

    e2, sign_e2, stable_e2, anchor_idx_e2, anchor_abs_e2 = orthogonalize_with_stability(
        subtract(sminus_unit, v_unit), [e1]
    )
    zero_result["sign_anchor_index_e2"] = anchor_idx_e2
    zero_result["sign_anchor_abs_e2"] = float(anchor_abs_e2)
    if e2 is None:
        zero_result["boundary_outcome"] = "zero_area"
        return zero_result
    if not stable_e2:
        zero_result["boundary_outcome"] = "sign_unstable"
        return zero_result

    signed_height = float(dot(subtract(sminus_unit, v_unit), e2))
    signed_area2 = float(edge_len_v_splus * signed_height)
    zero_result["relation_plane_height_signed"] = signed_height
    zero_result["relation_signed_area2"] = signed_area2
    set_partial_state(
        frame_rank=2,
        basis_sources=["d1", "d2_residual"],
        orientation_parity=int(sign_e1 * sign_e2),
    )
    if abs(signed_area2) <= RELATION_AREA_EPS:
        zero_result["boundary_outcome"] = "zero_area"
        return zero_result

    p_v = [0.0, 0.0]
    p_splus = [edge_len_v_splus, 0.0]
    p_sminus = [
        float(dot(subtract(sminus_unit, v_unit), e1)),
        signed_height,
    ]
    xy_center = centroid([p_v, p_splus, p_sminus])
    q_v = subtract(p_v, xy_center)
    q_splus = subtract(p_splus, xy_center)
    q_sminus = subtract(p_sminus, xy_center)

    altitude_v = float(signed_area2 / max(edge_len_splus_sminus, FRAME_EPS))
    altitude_splus = float(signed_area2 / max(edge_len_v_sminus, FRAME_EPS))
    altitude_sminus = float(signed_area2 / max(edge_len_v_splus, FRAME_EPS))
    angle_cos_v = clamped_cosine(subtract(splus_unit, v_unit), subtract(sminus_unit, v_unit))
    angle_cos_splus = clamped_cosine(
        subtract(v_unit, splus_unit), subtract(sminus_unit, splus_unit)
    )
    angle_cos_sminus = clamped_cosine(
        subtract(v_unit, sminus_unit), subtract(splus_unit, sminus_unit)
    )
    third_basis_source = "signed_angle_profile"
    orientation_parity = int(sign_e1 * sign_e2)
    z_v = signed_height * angle_cos_v
    z_splus = signed_height * angle_cos_splus
    z_sminus = signed_height * angle_cos_sminus

    if lift_mode in (
        COORDINATE_RULE_RELATION_AFFINE_LIFT_V1,
        COORDINATE_RULE_RELATION_AFFINE_LIFT_V2,
        COORDINATE_RULE_RELATION_AFFINE_LIFT_V3,
        COORDINATE_RULE_RELATION_AFFINE_LIFT_V4,
    ):
        e3_origin, sign_e3, stable_e3, anchor_idx_e3, anchor_abs_e3 = build_origin_span_e3_axis(
            v_unit=v_unit,
            splus_unit=splus_unit,
            sminus_unit=sminus_unit,
        )
        zero_result["sign_anchor_index_e3"] = anchor_idx_e3
        zero_result["sign_anchor_abs_e3"] = float(anchor_abs_e3)
        if e3_origin is None or not stable_e3:
            if lift_mode == COORDINATE_RULE_RELATION_AFFINE_LIFT_V1:
                zero_result["boundary_outcome"] = "raw_span_axis_collapse"
                return zero_result
        else:
            raw_z_values = [
                0.0,
                float(dot(subtract(splus_unit, v_unit), e3_origin)),
                float(dot(subtract(sminus_unit, v_unit), e3_origin)),
            ]
            raw_span_lift_center = 0.5 * (max(raw_z_values) + min(raw_z_values))
            centered_raw_z = [float(value - raw_span_lift_center) for value in raw_z_values]
            zero_result["raw_span_lift_center"] = float(raw_span_lift_center)
            zero_result["raw_span_lift_range"] = float(max(raw_z_values) - min(raw_z_values))
            zero_result["raw_span_axis_available"] = True
            if lift_mode == COORDINATE_RULE_RELATION_AFFINE_LIFT_V1:
                z_v = centered_raw_z[0]
                z_splus = centered_raw_z[1]
                z_sminus = centered_raw_z[2]
                third_basis_source = "origin_span_e3"
                orientation_parity = int(sign_e1 * sign_e2 * sign_e3)
            else:
                max_abs_centered = max(abs(value) for value in centered_raw_z)
                if max_abs_centered > FRAME_EPS:
                    raw_span_modulation = [
                        float(value / max_abs_centered) for value in centered_raw_z
                    ]
                    if lift_mode == COORDINATE_RULE_RELATION_AFFINE_LIFT_V3:
                        alpha = (
                            RAW_SPAN_MODULATION_ALPHA_V3
                            * relation_ambiguity_gate(
                                [angle_cos_v, angle_cos_splus, angle_cos_sminus]
                            )
                        )
                        third_basis_source = (
                            "signed_angle_profile_origin_span_modulation_gated"
                        )
                    elif lift_mode == COORDINATE_RULE_RELATION_AFFINE_LIFT_V4:
                        alpha = RAW_SPAN_MODULATION_ALPHA_V4
                        third_basis_source = (
                            "signed_angle_profile_origin_span_modulation_capped"
                        )
                    else:
                        alpha = RAW_SPAN_MODULATION_ALPHA_V2
                        third_basis_source = "signed_angle_profile_origin_span_modulation"
                    uncapped_z_v = signed_height * (
                        angle_cos_v + alpha * raw_span_modulation[0]
                    )
                    uncapped_z_splus = signed_height * (
                        angle_cos_splus + alpha * raw_span_modulation[1]
                    )
                    uncapped_z_sminus = signed_height * (
                        angle_cos_sminus + alpha * raw_span_modulation[2]
                    )
                    if lift_mode == COORDINATE_RULE_RELATION_AFFINE_LIFT_V4:
                        z_abs_cap = abs(signed_height) * RAW_SPAN_Z_CAP_MULTIPLIER_V4
                        z_v = clamp_symmetric(uncapped_z_v, z_abs_cap)
                        z_splus = clamp_symmetric(uncapped_z_splus, z_abs_cap)
                        z_sminus = clamp_symmetric(uncapped_z_sminus, z_abs_cap)
                    else:
                        z_v = uncapped_z_v
                        z_splus = uncapped_z_splus
                        z_sminus = uncapped_z_sminus
                    zero_result["raw_span_modulation_alpha"] = float(alpha)
                    orientation_parity = int(sign_e1 * sign_e2 * sign_e3)

    relation_v = [float(q_v[0]), float(q_v[1]), float(z_v)]
    relation_splus = [float(q_splus[0]), float(q_splus[1]), float(z_splus)]
    relation_sminus = [float(q_sminus[0]), float(q_sminus[1]), float(z_sminus)]
    relation_rank = gram_schmidt_rank([relation_v, relation_splus, relation_sminus])
    if relation_rank < 3:
        zero_result["boundary_outcome"] = "lift_axis_collapse"
        set_partial_state(
            frame_rank=relation_rank,
            basis_sources=["d1", "d2_residual", third_basis_source],
            orientation_parity=orientation_parity,
            altitude_values=(altitude_v, altitude_splus, altitude_sminus),
            angle_values=(angle_cos_v, angle_cos_splus, angle_cos_sminus),
        )
        zero_result["relation_lift_rank"] = int(relation_rank)
        return zero_result

    coords_v = pad8(relation_v)
    coords_splus = pad8(relation_splus)
    coords_sminus = pad8(relation_sminus)
    projected_norm_v = l2_norm(coords_v)
    projected_norm_splus = l2_norm(coords_splus)
    projected_norm_sminus = l2_norm(coords_sminus)
    raw_triplet_centroid = centroid([v_unit, splus_unit, sminus_unit])
    emitted_coord_centroid = centroid([coords_v, coords_splus, coords_sminus])
    return {
        "coords_v": coords_v,
        "coords_splus": coords_splus,
        "coords_sminus": coords_sminus,
        "boundary_outcome": "materialized_rank3",
        "frame_rank": 3,
        "orientation_parity": orientation_parity,
        "basis_sources": ["d1", "d2_residual", third_basis_source],
        "projected_norm_v": float(projected_norm_v),
        "projected_norm_splus": float(projected_norm_splus),
        "projected_norm_sminus": float(projected_norm_sminus),
        "raw_triplet_centroid_norm": float(l2_norm(raw_triplet_centroid)),
        "emitted_coord_centroid_norm": float(l2_norm(emitted_coord_centroid)),
        "emitted_centered_energy": float(
            centered_energy([coords_v, coords_splus, coords_sminus], emitted_coord_centroid)
        ),
        "relation_signed_area2": signed_area2,
        "relation_plane_height_signed": signed_height,
        "relation_edge_len_v_splus": edge_len_v_splus,
        "relation_edge_len_v_sminus": edge_len_v_sminus,
        "relation_edge_len_splus_sminus": edge_len_splus_sminus,
        "relation_altitude_v": altitude_v,
        "relation_altitude_splus": altitude_splus,
        "relation_altitude_sminus": altitude_sminus,
        "relation_angle_cos_v": angle_cos_v,
        "relation_angle_cos_splus": angle_cos_splus,
        "relation_angle_cos_sminus": angle_cos_sminus,
        "relation_lift_rank": int(relation_rank),
        "sign_anchor_index_e1": anchor_idx_e1,
        "sign_anchor_index_e2": anchor_idx_e2,
        "sign_anchor_index_e3": zero_result["sign_anchor_index_e3"],
        "sign_anchor_abs_e1": float(anchor_abs_e1),
        "sign_anchor_abs_e2": float(anchor_abs_e2),
        "sign_anchor_abs_e3": float(zero_result["sign_anchor_abs_e3"]),
        "raw_span_lift_center": float(zero_result["raw_span_lift_center"]),
        "raw_span_lift_range": float(zero_result["raw_span_lift_range"]),
        "raw_span_axis_available": bool(zero_result["raw_span_axis_available"]),
        "raw_span_modulation_alpha": float(zero_result["raw_span_modulation_alpha"]),
    }


def build_relation_affine_lift_coordinates(
    v_unit: Sequence[float],
    splus_unit: Sequence[float],
    sminus_unit: Sequence[float],
) -> Dict[str, Any]:
    return build_relation_affine_lift_coordinates_with_mode(
        v_unit=v_unit,
        splus_unit=splus_unit,
        sminus_unit=sminus_unit,
        lift_mode=COORDINATE_RULE_RELATION_AFFINE_LIFT_V0,
    )


def build_relation_affine_lift_coordinates_v1(
    v_unit: Sequence[float],
    splus_unit: Sequence[float],
    sminus_unit: Sequence[float],
) -> Dict[str, Any]:
    return build_relation_affine_lift_coordinates_with_mode(
        v_unit=v_unit,
        splus_unit=splus_unit,
        sminus_unit=sminus_unit,
        lift_mode=COORDINATE_RULE_RELATION_AFFINE_LIFT_V1,
    )


def build_relation_affine_lift_coordinates_v2(
    v_unit: Sequence[float],
    splus_unit: Sequence[float],
    sminus_unit: Sequence[float],
) -> Dict[str, Any]:
    return build_relation_affine_lift_coordinates_with_mode(
        v_unit=v_unit,
        splus_unit=splus_unit,
        sminus_unit=sminus_unit,
        lift_mode=COORDINATE_RULE_RELATION_AFFINE_LIFT_V2,
    )


def build_relation_affine_lift_coordinates_v3(
    v_unit: Sequence[float],
    splus_unit: Sequence[float],
    sminus_unit: Sequence[float],
) -> Dict[str, Any]:
    return build_relation_affine_lift_coordinates_with_mode(
        v_unit=v_unit,
        splus_unit=splus_unit,
        sminus_unit=sminus_unit,
        lift_mode=COORDINATE_RULE_RELATION_AFFINE_LIFT_V3,
    )


def build_relation_affine_lift_coordinates_v4(
    v_unit: Sequence[float],
    splus_unit: Sequence[float],
    sminus_unit: Sequence[float],
) -> Dict[str, Any]:
    return build_relation_affine_lift_coordinates_with_mode(
        v_unit=v_unit,
        splus_unit=splus_unit,
        sminus_unit=sminus_unit,
        lift_mode=COORDINATE_RULE_RELATION_AFFINE_LIFT_V4,
    )


def build_native_local_span_step(
    row: Dict[str, Any], sample_id: int, coordinate_rule: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    rule_meta = coordinate_rule_metadata(coordinate_rule)
    step = int(row["step"])
    relation_signed_area2 = 0.0
    relation_plane_height_signed = 0.0
    relation_edge_len_v_splus = 0.0
    relation_edge_len_v_sminus = 0.0
    relation_edge_len_splus_sminus = 0.0
    relation_altitude_v = 0.0
    relation_altitude_splus = 0.0
    relation_altitude_sminus = 0.0
    relation_angle_cos_v = 0.0
    relation_angle_cos_splus = 0.0
    relation_angle_cos_sminus = 0.0
    relation_lift_rank = 0
    sign_anchor_index_e1: Optional[int] = None
    sign_anchor_index_e2: Optional[int] = None
    sign_anchor_index_e3: Optional[int] = None
    sign_anchor_abs_e1 = 0.0
    sign_anchor_abs_e2 = 0.0
    sign_anchor_abs_e3 = 0.0
    raw_span_lift_center = 0.0
    raw_span_lift_range = 0.0
    raw_span_axis_available = False
    raw_span_modulation_alpha = 0.0
    try:
        v_raw = load_raw_triplet(row, "V_raw_native")
        splus_raw = load_raw_triplet(row, "Splus_raw_native")
        sminus_raw = load_raw_triplet(row, "Sminus_raw_native")
    except ValueError:
        coords_v = [0.0] * EMIT_DIM
        coords_splus = [0.0] * EMIT_DIM
        coords_sminus = [0.0] * EMIT_DIM
        boundary_outcome = "source_nonfinite"
        frame_rank = 0
        basis_sources: List[str] = []
        projected_norm_v = 0.0
        projected_norm_splus = 0.0
        projected_norm_sminus = 0.0
        orientation_parity = 0
        raw_triplet_centroid_norm = 0.0
        emitted_coord_centroid_norm = 0.0
        emitted_centered_energy = 0.0
    else:
        v_unit = normalize(v_raw)
        splus_unit = normalize(splus_raw)
        sminus_unit = normalize(sminus_raw)
        if v_unit is None or splus_unit is None or sminus_unit is None:
            coords_v = [0.0] * EMIT_DIM
            coords_splus = [0.0] * EMIT_DIM
            coords_sminus = [0.0] * EMIT_DIM
            boundary_outcome = "source_zero_or_nonfinite_norm"
            frame_rank = 0
            basis_sources = []
            projected_norm_v = 0.0
            projected_norm_splus = 0.0
            projected_norm_sminus = 0.0
            orientation_parity = 0
            raw_triplet_centroid_norm = 0.0
            emitted_coord_centroid_norm = 0.0
            emitted_centered_energy = 0.0
        else:
            relation_edge_len_v_splus = float(l2_norm(subtract(splus_unit, v_unit)))
            relation_edge_len_v_sminus = float(l2_norm(subtract(sminus_unit, v_unit)))
            relation_edge_len_splus_sminus = float(l2_norm(subtract(splus_unit, sminus_unit)))

            if coordinate_rule in (
                COORDINATE_RULE_RELATION_AFFINE_LIFT_V0,
                COORDINATE_RULE_RELATION_AFFINE_LIFT_V1,
                COORDINATE_RULE_RELATION_AFFINE_LIFT_V2,
                COORDINATE_RULE_RELATION_AFFINE_LIFT_V3,
                COORDINATE_RULE_RELATION_AFFINE_LIFT_V4,
            ):
                relation = (
                    build_relation_affine_lift_coordinates_v4
                    if coordinate_rule == COORDINATE_RULE_RELATION_AFFINE_LIFT_V4
                    else (
                        build_relation_affine_lift_coordinates_v3
                        if coordinate_rule == COORDINATE_RULE_RELATION_AFFINE_LIFT_V3
                        else (
                            build_relation_affine_lift_coordinates_v2
                            if coordinate_rule == COORDINATE_RULE_RELATION_AFFINE_LIFT_V2
                            else (
                                build_relation_affine_lift_coordinates_v1
                                if coordinate_rule == COORDINATE_RULE_RELATION_AFFINE_LIFT_V1
                                else build_relation_affine_lift_coordinates
                            )
                        )
                    )
                )(
                    v_unit=v_unit,
                    splus_unit=splus_unit,
                    sminus_unit=sminus_unit,
                )
                coords_v = relation["coords_v"]
                coords_splus = relation["coords_splus"]
                coords_sminus = relation["coords_sminus"]
                boundary_outcome = str(relation["boundary_outcome"])
                frame_rank = int(relation["frame_rank"])
                basis_sources = list(relation["basis_sources"])
                projected_norm_v = float(relation["projected_norm_v"])
                projected_norm_splus = float(relation["projected_norm_splus"])
                projected_norm_sminus = float(relation["projected_norm_sminus"])
                orientation_parity = int(relation["orientation_parity"])
                raw_triplet_centroid_norm = float(relation["raw_triplet_centroid_norm"])
                emitted_coord_centroid_norm = float(relation["emitted_coord_centroid_norm"])
                emitted_centered_energy = float(relation["emitted_centered_energy"])
                relation_signed_area2 = float(relation["relation_signed_area2"])
                relation_plane_height_signed = float(relation["relation_plane_height_signed"])
                relation_edge_len_v_splus = float(relation["relation_edge_len_v_splus"])
                relation_edge_len_v_sminus = float(relation["relation_edge_len_v_sminus"])
                relation_edge_len_splus_sminus = float(relation["relation_edge_len_splus_sminus"])
                relation_altitude_v = float(relation["relation_altitude_v"])
                relation_altitude_splus = float(relation["relation_altitude_splus"])
                relation_altitude_sminus = float(relation["relation_altitude_sminus"])
                relation_angle_cos_v = float(relation["relation_angle_cos_v"])
                relation_angle_cos_splus = float(relation["relation_angle_cos_splus"])
                relation_angle_cos_sminus = float(relation["relation_angle_cos_sminus"])
                relation_lift_rank = int(relation["relation_lift_rank"])
                sign_anchor_index_e1 = relation["sign_anchor_index_e1"]
                sign_anchor_index_e2 = relation["sign_anchor_index_e2"]
                sign_anchor_index_e3 = relation["sign_anchor_index_e3"]
                sign_anchor_abs_e1 = float(relation["sign_anchor_abs_e1"])
                sign_anchor_abs_e2 = float(relation["sign_anchor_abs_e2"])
                sign_anchor_abs_e3 = float(relation["sign_anchor_abs_e3"])
                raw_span_lift_center = float(relation["raw_span_lift_center"])
                raw_span_lift_range = float(relation["raw_span_lift_range"])
                raw_span_axis_available = bool(relation["raw_span_axis_available"])
                raw_span_modulation_alpha = float(relation["raw_span_modulation_alpha"])
            else:
                if coordinate_rule == COORDINATE_RULE_ORIGIN_SPAN_V2:
                    basis, basis_sources, orientation_parity = build_origin_span_basis(
                        v_unit=v_unit,
                        splus_unit=splus_unit,
                        sminus_unit=sminus_unit,
                    )
                else:
                    basis, basis_sources, orientation_parity = build_v_anchor_basis(
                        v_unit=v_unit,
                        splus_unit=splus_unit,
                        sminus_unit=sminus_unit,
                    )

                frame_rank = len(basis)
                if frame_rank <= 0:
                    coords_v = [0.0] * EMIT_DIM
                    coords_splus = [0.0] * EMIT_DIM
                    coords_sminus = [0.0] * EMIT_DIM
                    boundary_outcome = "frame_rank_collapse"
                    projected_norm_v = 0.0
                    projected_norm_splus = 0.0
                    projected_norm_sminus = 0.0
                    orientation_parity = 0
                    raw_triplet_centroid_norm = 0.0
                    emitted_coord_centroid_norm = 0.0
                    emitted_centered_energy = 0.0
                else:
                    raw_triplet_centroid = centroid([v_unit, splus_unit, sminus_unit])
                    raw_triplet_centroid_norm = l2_norm(raw_triplet_centroid)
                    if coordinate_rule == COORDINATE_RULE_CENTERED_AFFINE_V1:
                        source_v = subtract(v_unit, raw_triplet_centroid)
                        source_splus = subtract(splus_unit, raw_triplet_centroid)
                        source_sminus = subtract(sminus_unit, raw_triplet_centroid)
                    else:
                        source_v = list(v_unit)
                        source_splus = list(splus_unit)
                        source_sminus = list(sminus_unit)

                    coords_v = pad8(project_into_basis(source_v, basis))
                    coords_splus = pad8(project_into_basis(source_splus, basis))
                    coords_sminus = pad8(project_into_basis(source_sminus, basis))
                    projected_norm_v = l2_norm(coords_v)
                    projected_norm_splus = l2_norm(coords_splus)
                    projected_norm_sminus = l2_norm(coords_sminus)
                    emitted_coord_centroid = centroid([coords_v, coords_splus, coords_sminus])
                    emitted_coord_centroid_norm = l2_norm(emitted_coord_centroid)
                    emitted_centered_energy = centered_energy(
                        [coords_v, coords_splus, coords_sminus], emitted_coord_centroid
                    )
                    if (
                        projected_norm_v <= FRAME_EPS
                        or projected_norm_splus <= FRAME_EPS
                        or projected_norm_sminus <= FRAME_EPS
                    ):
                        boundary_outcome = "coordinate_projection_zero_or_nonfinite_norm"
                    else:
                        boundary_outcome = boundary_outcome_from_rank(frame_rank)

    token_step = {
        "step": step,
        "absolute_pos": int(row["absolute_pos"]),
        "answer_char_start": row.get("answer_char_start"),
        "answer_char_end": row.get("answer_char_end"),
        "token_id": int(row["token_id"]),
        "token_str": str(row["token_str"]),
        "V_8d": coords_v,
        "Splus_8d": coords_splus,
        "Sminus_8d": coords_sminus,
        "baseline_logprob": float(row["baseline_logprob"]),
        "baseline_entropy": float(row["baseline_entropy"]),
    }
    boundary_step = {
        "sample_id": int(sample_id),
        "step": step,
        "boundary_id": rule_meta["boundary_id"],
        "frame_construction_id": rule_meta["frame_construction_id"],
        "source_tensor_id": SOURCE_TENSOR_ID,
        "coordinate_space_id": rule_meta["coordinate_space_id"],
        "coordinate_rule_id": coordinate_rule,
        "basis_rule_id": rule_meta["basis_rule_id"],
        "projection_or_span_rule": rule_meta["projection_or_span_rule"],
        "basis_sign_rule_id": BASIS_SIGN_RULE_ID,
        "basis_order_rule_id": BASIS_ORDER_RULE_ID,
        "orientation_rule_id": ORIENTATION_RULE_ID,
        "degeneracy_policy_id": DEGENERACY_POLICY_ID,
        "boundary_outcome": boundary_outcome,
        "frame_rank": int(frame_rank),
        "frame_dim_emitted": EMIT_DIM,
        "orientation_parity": int(orientation_parity),
        "basis_sources": basis_sources,
        "projected_norm_v": float(projected_norm_v),
        "projected_norm_splus": float(projected_norm_splus),
        "projected_norm_sminus": float(projected_norm_sminus),
        "raw_triplet_centroid_norm": float(raw_triplet_centroid_norm),
        "emitted_coord_centroid_norm": float(emitted_coord_centroid_norm),
        "emitted_centered_energy": float(emitted_centered_energy),
        "relation_signed_area2": float(relation_signed_area2),
        "relation_plane_height_signed": float(relation_plane_height_signed),
        "relation_edge_len_v_splus": float(relation_edge_len_v_splus),
        "relation_edge_len_v_sminus": float(relation_edge_len_v_sminus),
        "relation_edge_len_splus_sminus": float(relation_edge_len_splus_sminus),
        "relation_altitude_v": float(relation_altitude_v),
        "relation_altitude_splus": float(relation_altitude_splus),
        "relation_altitude_sminus": float(relation_altitude_sminus),
        "relation_angle_cos_v": float(relation_angle_cos_v),
        "relation_angle_cos_splus": float(relation_angle_cos_splus),
        "relation_angle_cos_sminus": float(relation_angle_cos_sminus),
        "relation_lift_rank": int(relation_lift_rank),
        "sign_anchor_index_e1": sign_anchor_index_e1,
        "sign_anchor_index_e2": sign_anchor_index_e2,
        "sign_anchor_index_e3": sign_anchor_index_e3,
        "sign_anchor_abs_e1": float(sign_anchor_abs_e1),
        "sign_anchor_abs_e2": float(sign_anchor_abs_e2),
        "sign_anchor_abs_e3": float(sign_anchor_abs_e3),
        "raw_span_lift_center": float(raw_span_lift_center),
        "raw_span_lift_range": float(raw_span_lift_range),
        "raw_span_axis_available": bool(raw_span_axis_available),
        "raw_span_modulation_alpha": float(raw_span_modulation_alpha),
    }
    return token_step, boundary_step


def validate_upstream_metadata(sample_dirs: Sequence[Path]) -> Dict[str, Any]:
    keys = (
        "model_id",
        "model_revision",
        "seed",
        "splus_def_id",
        "sminus_def_id",
    )
    first_meta = packer.load_json(sample_dirs[0] / "meta.json")
    first_view = {key: first_meta.get(key) for key in keys}
    for sample_dir in sample_dirs[1:]:
        current_meta = packer.load_json(sample_dir / "meta.json")
        current_view = {key: current_meta.get(key) for key in keys}
        if current_view != first_view:
            raise ValueError(
                f"sample metadata mismatch between {sample_dirs[0] / 'meta.json'} and "
                f"{sample_dir / 'meta.json'} for keys={list(keys)}: "
                f"expected={first_view!r} actual={current_view!r}"
            )
    return first_meta


def build_sample_payload(
    sample_dir: Path,
    sample_id: int,
    coordinate_rule: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    triplets_path = sample_dir / "triplets.ndjson"
    meta_path = sample_dir / "meta.json"
    labels_path = sample_dir / "labels.jsonl"
    labels_meta_path = sample_dir / "labels_meta.json"

    triplets = packer.load_jsonl(triplets_path)
    if not triplets:
        raise ValueError(f"empty triplets: {triplets_path}")
    triplets.sort(key=lambda row: int(row["step"]))
    expected_steps = list(range(len(triplets)))
    actual_steps = [int(row["step"]) for row in triplets]
    if actual_steps != expected_steps:
        raise ValueError(
            f"triplets steps for sample {sample_id} are not contiguous 0..N-1: {actual_steps[:16]}"
        )

    meta = packer.load_json(meta_path)
    labels_meta = packer.load_json(labels_meta_path)
    labels = packer.load_step_labels(labels_path, n_steps=len(triplets))
    triplets_file_sha256 = packer.sha256_file(triplets_path)
    meta_triplets_sha256 = str(meta.get("output_ndjson_sha256") or "")
    if triplets_file_sha256 != meta_triplets_sha256:
        raise ValueError(
            f"triplets SHA mismatch for sample {sample_id}: "
            f"file={triplets_file_sha256} meta={meta_triplets_sha256}"
        )

    native_schema_id = meta.get("native_raw_schema_id")
    if native_schema_id not in (RAW_NATIVE_SCHEMA_ID, None):
        raise ValueError(
            f"unsupported native raw schema for sample {sample_id}: {native_schema_id!r}"
        )
    if not all(all(key in row for key in RAW_KEYS) for row in triplets):
        raise ValueError(
            f"sample {sample_id} triplets do not contain raw native vectors; "
            "rerun extraction with --emit-native-raw"
        )

    token_steps: List[Dict[str, Any]] = []
    boundary_steps: List[Dict[str, Any]] = []
    for idx, row in enumerate(triplets):
        token_step, boundary_step = build_native_local_span_step(
            row=row, sample_id=sample_id, coordinate_rule=coordinate_rule
        )
        token_step["label_token"] = int(labels[idx])
        token_step["defect_span_id"] = None
        token_steps.append(token_step)
        boundary_steps.append(boundary_step)

    return (
        {
            "sample_id": int(sample_id),
            "variant": str(labels_meta.get("variant") or "unknown"),
            "world_type": labels_meta.get("world_type"),
            "exact_token_match_ratio": float(meta["exact_token_match_ratio"]),
            "label_coverage_ratio": float(labels_meta["final_alignment_coverage_ratio"]),
            "triplets_sha256": triplets_file_sha256,
            "labels_sha256": packer.sha256_file(labels_path),
            "token_steps": token_steps,
        },
        boundary_steps,
    )


def build_metadata(
    first_meta: Dict[str, Any],
    script_extract: Path,
    script_boundary_builder: Path,
    script_eval: Path,
    perm_r: int,
    primary_score: str,
    coordinate_rule: str,
) -> Dict[str, Any]:
    boundary_meta = dict(first_meta)
    boundary_meta["proj_id"] = coordinate_rule_metadata(coordinate_rule)["boundary_id"]
    metadata = packer.build_metadata(
        first_meta=boundary_meta,
        script_extract=script_extract,
        script_eval=script_eval,
        perm_r=perm_r,
        primary_score=primary_score,
    )
    metadata["script_sha256_boundary_builder"] = packer.sha256_file(script_boundary_builder)
    return metadata


def count_by_key(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        value = str(row[key])
        counts[value] = counts.get(value, 0) + 1
    return counts


def count_by_classifier(
    rows: Sequence[Dict[str, Any]], classifier: Any
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        value = classifier(row)
        if value is None:
            continue
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def raw_span_path_key(row: Dict[str, Any]) -> Optional[str]:
    if row.get("coordinate_rule_id") not in (
        COORDINATE_RULE_RELATION_AFFINE_LIFT_V2,
        COORDINATE_RULE_RELATION_AFFINE_LIFT_V3,
        COORDINATE_RULE_RELATION_AFFINE_LIFT_V4,
    ):
        return None
    boundary_outcome = str(row.get("boundary_outcome") or "")
    is_materialized = boundary_outcome.startswith("materialized_rank")
    axis_available = bool(row.get("raw_span_axis_available"))
    if is_materialized:
        if axis_available:
            return "modulated"
        return "fallback_materialized"
    if axis_available:
        return "axis_available_nonmaterialized"
    return "axis_unavailable_nonmaterialized"


def main() -> int:
    args = parse_args()
    samples_root = (REPO_ROOT / args.samples_root).resolve()
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    gate4_input_path = out_dir / DEFAULT_GATE4_INPUT
    selection_manifest_path = out_dir / DEFAULT_SELECTION_MANIFEST
    build_manifest_path = out_dir / DEFAULT_BUILD_MANIFEST
    boundary_steps_path = out_dir / DEFAULT_BOUNDARY_STEPS

    selected_ids = packer.resolve_selected_sample_ids(
        samples_root=samples_root,
        sample_ids=args.sample_ids,
        sample_id_file=Path(args.sample_id_file) if args.sample_id_file else None,
        all_samples=bool(args.all_samples),
        variant=args.variant,
        offset=args.offset,
        limit=args.limit,
    )
    sample_dirs = [samples_root / packer.sample_dir_name(sample_id) for sample_id in selected_ids]
    first_meta = validate_upstream_metadata(sample_dirs)

    samples: List[Dict[str, Any]] = []
    boundary_steps: List[Dict[str, Any]] = []
    for sample_dir, sample_id in zip(sample_dirs, selected_ids):
        sample_payload, sample_boundary_steps = build_sample_payload(
            sample_dir=sample_dir,
            sample_id=sample_id,
            coordinate_rule=args.coordinate_rule,
        )
        samples.append(sample_payload)
        boundary_steps.extend(sample_boundary_steps)

    payload = {
        "metadata": build_metadata(
            first_meta=first_meta,
            script_extract=(REPO_ROOT / args.script_extract).resolve(),
            script_boundary_builder=Path(__file__).resolve(),
            script_eval=(REPO_ROOT / args.script_eval).resolve(),
            perm_r=args.perm_r,
            primary_score=args.primary_score,
            coordinate_rule=args.coordinate_rule,
        ),
        "samples": samples,
    }
    write_json(gate4_input_path, payload)
    write_jsonl(boundary_steps_path, boundary_steps)

    selection_manifest = packer.build_selection_manifest(
        samples_root=samples_root,
        sample_ids=selected_ids,
        variant_filter=args.variant,
        offset=args.offset,
        limit=args.limit,
        out_path=gate4_input_path,
    )
    rule_meta = coordinate_rule_metadata(args.coordinate_rule)
    selection_manifest["boundary_id"] = rule_meta["boundary_id"]
    selection_manifest["coordinate_rule_id"] = args.coordinate_rule
    selection_manifest["boundary_steps_jsonl"] = repo_relative_or_posix(boundary_steps_path)
    write_json(selection_manifest_path, selection_manifest)

    build_manifest = {
        "mode": "native_local_span_gate4_builder_v0",
        "samples_root": repo_relative_or_posix(samples_root),
        "gate4_input_json": repo_relative_or_posix(gate4_input_path),
        "selection_manifest_json": repo_relative_or_posix(selection_manifest_path),
        "boundary_steps_jsonl": repo_relative_or_posix(boundary_steps_path),
        "n_samples": len(selected_ids),
        "sample_ids": [int(sample_id) for sample_id in selected_ids],
        "n_boundary_steps": len(boundary_steps),
        "boundary_id": rule_meta["boundary_id"],
        "frame_construction_id": rule_meta["frame_construction_id"],
        "source_tensor_id": SOURCE_TENSOR_ID,
        "coordinate_space_id": rule_meta["coordinate_space_id"],
        "coordinate_rule_id": args.coordinate_rule,
        "basis_rule_id": rule_meta["basis_rule_id"],
        "projection_or_span_rule": rule_meta["projection_or_span_rule"],
        "basis_sign_rule_id": BASIS_SIGN_RULE_ID,
        "basis_order_rule_id": BASIS_ORDER_RULE_ID,
        "orientation_rule_id": ORIENTATION_RULE_ID,
        "degeneracy_policy_id": DEGENERACY_POLICY_ID,
        "frame_eps": FRAME_EPS,
        "frame_dim_emitted": EMIT_DIM,
        "boundary_outcome_counts": count_by_key(boundary_steps, "boundary_outcome"),
        "frame_rank_counts": count_by_key(boundary_steps, "frame_rank"),
        "raw_span_axis_available_counts": count_by_key(boundary_steps, "raw_span_axis_available"),
        "raw_span_path_counts": count_by_classifier(boundary_steps, raw_span_path_key),
        "model_id": first_meta.get("model_id"),
        "model_revision": first_meta.get("model_revision"),
        "seed": first_meta.get("seed"),
        "upstream_proj_id": first_meta.get("proj_id"),
        "upstream_native_raw_schema_id": first_meta.get("native_raw_schema_id"),
        "script_sha256_boundary_builder": packer.sha256_file(Path(__file__).resolve()),
        "script_sha256_extract": packer.sha256_file((REPO_ROOT / args.script_extract).resolve()),
        "script_sha256_eval": packer.sha256_file((REPO_ROOT / args.script_eval).resolve()),
    }
    write_json(build_manifest_path, build_manifest)

    print(f"gate4_input_json={repo_relative_or_posix(gate4_input_path)}")
    print(f"selection_manifest_json={repo_relative_or_posix(selection_manifest_path)}")
    print(f"build_manifest_json={repo_relative_or_posix(build_manifest_path)}")
    print(f"boundary_steps_jsonl={repo_relative_or_posix(boundary_steps_path)}")
    print(f"n_samples={len(selected_ids)}")
    print(f"n_boundary_steps={len(boundary_steps)}")
    print(f"sample_ids={','.join(str(sample_id) for sample_id in selected_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
