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
FRAME_CONSTRUCTION_ID_ANCHORED = "anchor_v_diff_gram_schmidt_v0"
FRAME_CONSTRUCTION_ID_ORIGIN_SPAN = "raw_triplet_origin_gram_schmidt_v1"
SOURCE_TENSOR_ID = "triality_raw_triplet_preprojection_v1"
COORDINATE_SPACE_ID_ANCHORED_V0 = "native_local_span_coordinates_pad8_v0"
COORDINATE_SPACE_ID_CENTERED_AFFINE_V1 = "native_local_span_coordinates_centered_affine_pad8_v1"
COORDINATE_SPACE_ID_ORIGIN_SPAN_V2 = "native_local_span_coordinates_origin_span_pad8_v2"
COORDINATE_RULE_ANCHORED_V0 = "anchored_projection_v0"
COORDINATE_RULE_CENTERED_AFFINE_V1 = "centered_affine_local_span_v1"
COORDINATE_RULE_ORIGIN_SPAN_V2 = "origin_span_projection_v2"
BASIS_RULE_ID_ANCHORED = "v_anchor_diff_gram_schmidt_v0"
BASIS_RULE_ID_ORIGIN_SPAN = "raw_triplet_origin_gram_schmidt_v1"
PROJECTION_OR_SPAN_RULE_ANCHORED_V0 = "unit_source_projection_onto_local_span_pad8_v0"
PROJECTION_OR_SPAN_RULE_CENTERED_AFFINE_V1 = "centered_affine_projection_onto_local_span_pad8_v1"
PROJECTION_OR_SPAN_RULE_ORIGIN_SPAN_V2 = (
    "unit_source_projection_onto_raw_triplet_origin_span_pad8_v2"
)
BASIS_SIGN_RULE_ID = "first_non_negligible_positive_v0"
BASIS_ORDER_RULE_ID = "construction_order_v0"
ORIENTATION_RULE_ID = "construction_order_parity_v0"
DEGENERACY_POLICY_ID = "honest_variable_rank_no_fake_completion_v0"
RAW_NATIVE_SCHEMA_ID = "triality_raw_native_v1"
RAW_KEYS = ("V_raw_native", "Splus_raw_native", "Sminus_raw_native")
FRAME_EPS = 1e-6
EMIT_DIM = 8


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


def build_native_local_span_step(
    row: Dict[str, Any], sample_id: int, coordinate_rule: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    rule_meta = coordinate_rule_metadata(coordinate_rule)
    step = int(row["step"])
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
