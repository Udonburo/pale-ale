#!/usr/bin/env python3
"""Run a Gate6 object-native consumer based on local ray-projector holonomy."""

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import build_gate6_native_local_span as gate6_builder
import run_gate6_native_object_consumer as base_consumer


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MANIFEST = "manifest.json"
DEFAULT_STEP_INDEX = "step_index.jsonl"
DEFAULT_ARRAYS = "native_object_arrays.npz"
DEFAULT_CHECKSUMS = "checksums.json"
DEFAULT_TOKEN_CSV = "gate6c_token_telemetry.csv"
DEFAULT_SAMPLE_CSV = "gate6c_sample_summary.csv"
DEFAULT_AGGREGATE = "gate6c_aggregate_summary.md"

SCHEMA_VERSION = "gate6_native_ray_projector_consumer_artifacts_v1"
METHOD_ID = "gate6_native_object_consumer_ray_projector_v1"
PRIMARY_METRIC_ID = "ray_projector_loop_projective_chordal_v1"
PRIMARY_AUX_METRIC_ID = "ray_projector_loop_geodesic_angle_v1"
PRIMARY_LEAKAGE_METRIC_ID = "ray_projector_loop_frob_identity_v1"
BASELINE_METRIC_ID = "score_F_gram_loop_v1"

RAY_NAMES = ("v", "splus", "sminus")
PAIR_NAMES = (
    "v_to_splus",
    "splus_to_sminus",
    "sminus_to_v",
)
PAIR_INDICES = (
    (0, 1),
    (1, 2),
    (2, 0),
)

TOKEN_COLUMNS = (
    "run_id",
    "sample_id",
    "step",
    "token_text",
    "label_token",
    "rank_local",
    "flags_compact",
    "ray_outcome_v",
    "ray_outcome_splus",
    "ray_outcome_sminus",
    "transport_outcome_v_to_splus",
    "transport_outcome_splus_to_sminus",
    "transport_outcome_sminus_to_v",
    "loop_outcome",
    "score_F_gram_loop_v1",
    PRIMARY_METRIC_ID,
    PRIMARY_AUX_METRIC_ID,
    PRIMARY_LEAKAGE_METRIC_ID,
    "ray_projector_loop_det_v1",
    "baseline_logprob",
    "baseline_entropy",
)

SAMPLE_COLUMNS = (
    "run_id",
    "sample_id",
    "n_token_steps",
    "n_loop_steps_valid",
    "n_loop_steps_missing",
    "positive_token_count",
    "auprc_F_gram_loop_v1",
    f"auprc_{PRIMARY_METRIC_ID}",
    f"delta_auprc_{PRIMARY_METRIC_ID}_vs_F_gram_loop_v1",
    "hit_at_10_F_gram_loop_v1",
    f"hit_at_10_{PRIMARY_METRIC_ID}",
    f"mean_{PRIMARY_METRIC_ID}",
    f"max_{PRIMARY_METRIC_ID}",
    f"p90_{PRIMARY_METRIC_ID}",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Gate6 object-native consumer that computes projective holonomy "
            "over the three local observable rays."
        )
    )
    parser.add_argument("--gate6-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-id", default="")
    return parser.parse_args()


def projective_ray_from_column(column: np.ndarray) -> Tuple[str, Optional[np.ndarray]]:
    unit = base_consumer.normalize3(np.asarray(column, dtype=np.float64))
    if unit is None:
        return ("zero_ray", None)
    fixed, _flipped, _anchor_index = gate6_builder.sign_fix_column(unit)
    return ("materialized", fixed)


def projective_transport_matrix(left: np.ndarray, right: np.ndarray) -> Tuple[str, Optional[np.ndarray]]:
    dot_value = float(np.clip(np.dot(left, right), -1.0, 1.0))
    target = right if dot_value >= 0.0 else -right
    prefix = "projective_flip_" if dot_value < 0.0 else ""
    outcome, rotation = base_consumer.minimal_rotation_matrix(left, target)
    return (prefix + outcome, rotation)


def compute_ray_projector_loop_metrics(
    coords_local: np.ndarray,
    gram_raw: np.ndarray,
) -> Dict[str, Any]:
    coords = np.asarray(coords_local, dtype=np.float64)
    if coords.shape != (3, 3):
        raise ValueError(f"coords_local must be 3x3, got {coords.shape}")

    ray_outcomes: List[str] = []
    rays: List[Optional[np.ndarray]] = []
    for ray_idx in range(3):
        outcome, ray = projective_ray_from_column(coords[:, ray_idx])
        ray_outcomes.append(outcome)
        rays.append(ray)

    score_f = base_consumer.build_score_f_gram_loop(np.asarray(gram_raw, dtype=np.float64))
    if any(ray is None for ray in rays):
        return {
            "ray_outcomes": ray_outcomes,
            "transport_outcomes": ["skipped_missing_ray"] * 3,
            "loop_outcome": "partial_loop_missing",
            BASELINE_METRIC_ID: score_f,
            PRIMARY_METRIC_ID: None,
            PRIMARY_AUX_METRIC_ID: None,
            PRIMARY_LEAKAGE_METRIC_ID: None,
            "ray_projector_loop_det_v1": None,
        }

    materialized_rays = [ray for ray in rays if ray is not None]
    transport_outcomes: List[str] = []
    rotations: List[Optional[np.ndarray]] = []
    for left_idx, right_idx in PAIR_INDICES:
        outcome, rotation = projective_transport_matrix(
            materialized_rays[left_idx],
            materialized_rays[right_idx],
        )
        transport_outcomes.append(outcome)
        rotations.append(rotation)

    if any(rotation is None for rotation in rotations):
        return {
            "ray_outcomes": ray_outcomes,
            "transport_outcomes": transport_outcomes,
            "loop_outcome": "partial_loop_missing",
            BASELINE_METRIC_ID: score_f,
            PRIMARY_METRIC_ID: None,
            PRIMARY_AUX_METRIC_ID: None,
            PRIMARY_LEAKAGE_METRIC_ID: None,
            "ray_projector_loop_det_v1": None,
        }

    holonomy = rotations[2] @ rotations[1] @ rotations[0]
    quat_abs = base_consumer.quaternion_scalar_abs_from_rotation(holonomy)
    chordal = math.sqrt(max(0.0, 2.0 * (1.0 - min(1.0, abs(quat_abs)))))
    geodesic = base_consumer.geodesic_angle_from_rotation(holonomy)
    fro_identity = float(np.linalg.norm(holonomy - np.eye(3, dtype=np.float64), ord="fro"))
    det_value = float(np.linalg.det(holonomy))
    if not all(math.isfinite(value) for value in (chordal, geodesic, fro_identity, det_value)):
        return {
            "ray_outcomes": ray_outcomes,
            "transport_outcomes": transport_outcomes,
            "loop_outcome": "invalid_holonomy",
            BASELINE_METRIC_ID: score_f,
            PRIMARY_METRIC_ID: None,
            PRIMARY_AUX_METRIC_ID: None,
            PRIMARY_LEAKAGE_METRIC_ID: None,
            "ray_projector_loop_det_v1": None,
        }

    return {
        "ray_outcomes": ray_outcomes,
        "transport_outcomes": transport_outcomes,
        "loop_outcome": "none",
        BASELINE_METRIC_ID: score_f,
        PRIMARY_METRIC_ID: chordal,
        PRIMARY_AUX_METRIC_ID: geodesic,
        PRIMARY_LEAKAGE_METRIC_ID: fro_identity,
        "ray_projector_loop_det_v1": det_value,
    }


def load_rows(step_index_path: Path) -> List[Dict[str, Any]]:
    rows = base_consumer.read_jsonl(step_index_path)
    rows.sort(key=lambda row: (int(row["sample_id"]), int(row["step"])))
    return rows


def build_token_rows(
    run_id: str,
    step_rows: Sequence[Dict[str, Any]],
    arrays: Dict[str, np.ndarray],
) -> List[Dict[str, Any]]:
    token_rows: List[Dict[str, Any]] = []
    for step_row in step_rows:
        array_row_index = int(step_row["array_row_index"])
        metrics = compute_ray_projector_loop_metrics(
            coords_local=arrays["coords_local"][array_row_index],
            gram_raw=arrays["gram_raw"][array_row_index],
        )
        token_rows.append(
            {
                "run_id": run_id,
                "sample_id": int(step_row["sample_id"]),
                "step": int(step_row["step"]),
                "token_text": str(step_row["token_text"]),
                "label_token": int(step_row["label_token"]),
                "rank_local": int(step_row["rank_local"]),
                "flags_compact": str(step_row["flags_compact"]),
                "ray_outcome_v": metrics["ray_outcomes"][0],
                "ray_outcome_splus": metrics["ray_outcomes"][1],
                "ray_outcome_sminus": metrics["ray_outcomes"][2],
                "transport_outcome_v_to_splus": metrics["transport_outcomes"][0],
                "transport_outcome_splus_to_sminus": metrics["transport_outcomes"][1],
                "transport_outcome_sminus_to_v": metrics["transport_outcomes"][2],
                "loop_outcome": metrics["loop_outcome"],
                BASELINE_METRIC_ID: metrics[BASELINE_METRIC_ID],
                PRIMARY_METRIC_ID: metrics[PRIMARY_METRIC_ID],
                PRIMARY_AUX_METRIC_ID: metrics[PRIMARY_AUX_METRIC_ID],
                PRIMARY_LEAKAGE_METRIC_ID: metrics[PRIMARY_LEAKAGE_METRIC_ID],
                "ray_projector_loop_det_v1": metrics["ray_projector_loop_det_v1"],
                "baseline_logprob": float(step_row["baseline_logprob"]),
                "baseline_entropy": float(step_row["baseline_entropy"]),
            }
        )
    return token_rows


def build_sample_rows(run_id: str, token_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in token_rows:
        grouped[int(row["sample_id"])].append(row)

    sample_rows: List[Dict[str, Any]] = []
    for sample_id in sorted(grouped):
        rows = sorted(grouped[sample_id], key=lambda row: int(row["step"]))
        labels = [int(row["label_token"]) for row in rows]
        scores_f = [float(row[BASELINE_METRIC_ID]) for row in rows]
        scores_primary = [
            None if row[PRIMARY_METRIC_ID] is None else float(row[PRIMARY_METRIC_ID]) for row in rows
        ]
        auprc_f = base_consumer.average_precision(labels, scores_f)
        auprc_primary = base_consumer.average_precision_optional(labels, scores_primary)
        delta_auprc = None if auprc_f is None or auprc_primary is None else float(auprc_primary - auprc_f)
        valid_primary = [score for score in scores_primary if score is not None]
        sample_rows.append(
            {
                "run_id": run_id,
                "sample_id": sample_id,
                "n_token_steps": len(rows),
                "n_loop_steps_valid": sum(1 for row in rows if row["loop_outcome"] == "none"),
                "n_loop_steps_missing": sum(1 for row in rows if row["loop_outcome"] != "none"),
                "positive_token_count": sum(1 for label in labels if label == 1),
                "auprc_F_gram_loop_v1": auprc_f,
                f"auprc_{PRIMARY_METRIC_ID}": auprc_primary,
                f"delta_auprc_{PRIMARY_METRIC_ID}_vs_F_gram_loop_v1": delta_auprc,
                "hit_at_10_F_gram_loop_v1": base_consumer.hit_at_k_optional(labels, scores_f, 10),
                f"hit_at_10_{PRIMARY_METRIC_ID}": base_consumer.hit_at_k_optional(labels, scores_primary, 10),
                f"mean_{PRIMARY_METRIC_ID}": None if not valid_primary else float(np.mean(valid_primary)),
                f"max_{PRIMARY_METRIC_ID}": None if not valid_primary else float(np.max(valid_primary)),
                f"p90_{PRIMARY_METRIC_ID}": base_consumer.percentile(valid_primary, 90.0),
            }
        )
    return sample_rows


def build_aggregate_summary(
    run_id: str,
    gate6_manifest: Dict[str, Any],
    token_rows: Sequence[Dict[str, Any]],
    sample_rows: Sequence[Dict[str, Any]],
) -> str:
    labels = [int(row["label_token"]) for row in token_rows]
    scores_f = [float(row[BASELINE_METRIC_ID]) for row in token_rows]
    scores_primary = [
        None if row[PRIMARY_METRIC_ID] is None else float(row[PRIMARY_METRIC_ID]) for row in token_rows
    ]
    global_auprc_f = base_consumer.average_precision(labels, scores_f)
    global_auprc_primary = base_consumer.average_precision_optional(labels, scores_primary)
    mean_sample_auprc_f_values = [
        float(row["auprc_F_gram_loop_v1"])
        for row in sample_rows
        if row["auprc_F_gram_loop_v1"] is not None
    ]
    mean_sample_auprc_primary_values = [
        float(row[f"auprc_{PRIMARY_METRIC_ID}"])
        for row in sample_rows
        if row[f"auprc_{PRIMARY_METRIC_ID}"] is not None
    ]
    mean_hit_f_values = [
        int(row["hit_at_10_F_gram_loop_v1"])
        for row in sample_rows
        if row["hit_at_10_F_gram_loop_v1"] is not None
    ]
    mean_hit_primary_values = [
        int(row[f"hit_at_10_{PRIMARY_METRIC_ID}"])
        for row in sample_rows
        if row[f"hit_at_10_{PRIMARY_METRIC_ID}"] is not None
    ]
    first_hit_primary_values = [
        value
        for value in (
            base_consumer.first_hit_distance(
                [int(row["label_token"]) for row in sample_token_rows],
                [
                    None if metric_row[PRIMARY_METRIC_ID] is None else float(metric_row[PRIMARY_METRIC_ID])
                    for metric_row in sample_token_rows
                ],
            )
            for sample_token_rows in (
                [row for row in token_rows if int(row["sample_id"]) == int(sample_row["sample_id"])]
                for sample_row in sample_rows
            )
        )
        if value is not None
    ]
    primary_values = [float(score) for score in scores_primary if score is not None]

    lines = [
        "# Gate6 Ray Projector Consumer Summary",
        "",
        f"run_id: {run_id}",
        f"method_id: {METHOD_ID}",
        f"input_gate6_run_id: {gate6_manifest.get('run_id', '')}",
        f"n_samples_total: {len(sample_rows)}",
        f"n_token_rows_total: {len(token_rows)}",
        f"n_loop_rows_valid: {sum(1 for row in token_rows if row['loop_outcome'] == 'none')}",
        f"n_loop_rows_missing: {sum(1 for row in token_rows if row['loop_outcome'] != 'none')}",
        "",
        "## Headline Metrics",
        "",
        f"- global_auprc_{BASELINE_METRIC_ID}: {base_consumer.render_float(global_auprc_f)}",
        f"- global_auprc_{PRIMARY_METRIC_ID}: {base_consumer.render_float(global_auprc_primary)}",
        f"- mean_sample_auprc_{BASELINE_METRIC_ID}: {base_consumer.render_float(None if not mean_sample_auprc_f_values else float(np.mean(mean_sample_auprc_f_values)))}",
        f"- mean_sample_auprc_{PRIMARY_METRIC_ID}: {base_consumer.render_float(None if not mean_sample_auprc_primary_values else float(np.mean(mean_sample_auprc_primary_values)))}",
        f"- mean_hit@10_{BASELINE_METRIC_ID}: {base_consumer.render_float(None if not mean_hit_f_values else float(np.mean(mean_hit_f_values)))}",
        f"- mean_hit@10_{PRIMARY_METRIC_ID}: {base_consumer.render_float(None if not mean_hit_primary_values else float(np.mean(mean_hit_primary_values)))}",
        f"- mean_first_hit_{PRIMARY_METRIC_ID}: {base_consumer.render_float(None if not first_hit_primary_values else float(np.mean(first_hit_primary_values)))}",
        f"- mean_{PRIMARY_METRIC_ID}: {base_consumer.render_float(None if not primary_values else float(np.mean(primary_values)))}",
        f"- p90_{PRIMARY_METRIC_ID}: {base_consumer.render_float(base_consumer.percentile(primary_values, 90.0))}",
        f"- max_{PRIMARY_METRIC_ID}: {base_consumer.render_float(None if not primary_values else float(np.max(primary_values)))}",
    ]
    return "\n".join(lines) + "\n"


def build_manifest(
    run_id: str,
    gate6_dir: Path,
    gate6_manifest_path: Path,
    step_index_path: Path,
    arrays_path: Path,
    token_rows: Sequence[Dict[str, Any]],
    sample_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "primary_metric_id": PRIMARY_METRIC_ID,
        "baseline_metric_id": BASELINE_METRIC_ID,
        "input_gate6_dir": base_consumer.repo_relative_or_posix(gate6_dir),
        "input_gate6_manifest_path": base_consumer.repo_relative_or_posix(gate6_manifest_path),
        "input_gate6_manifest_sha256": base_consumer.sha256_file(gate6_manifest_path),
        "input_gate6_step_index_path": base_consumer.repo_relative_or_posix(step_index_path),
        "input_gate6_step_index_sha256": base_consumer.sha256_file(step_index_path),
        "input_gate6_arrays_path": base_consumer.repo_relative_or_posix(arrays_path),
        "input_gate6_arrays_sha256": base_consumer.sha256_file(arrays_path),
        "input_gate6_method_id": base_consumer.read_json(gate6_manifest_path).get("method_id", ""),
        "code_git_commit": gate6_builder.current_git_commit(),
        "normal_sign_fix_mode": gate6_builder.SIGN_FIX_MODE,
        "n_samples_total": len(sample_rows),
        "n_token_rows_total": len(token_rows),
        "n_loop_rows_valid": sum(1 for row in token_rows if row["loop_outcome"] == "none"),
        "n_loop_rows_missing": sum(1 for row in token_rows if row["loop_outcome"] != "none"),
    }


def main() -> int:
    args = parse_args()
    gate6_dir = (REPO_ROOT / args.gate6_dir).resolve()
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    run_id = args.run_id or out_dir.name

    gate6_manifest_path = gate6_dir / DEFAULT_MANIFEST
    step_index_path = gate6_dir / DEFAULT_STEP_INDEX
    arrays_path = gate6_dir / DEFAULT_ARRAYS

    gate6_manifest = base_consumer.read_json(gate6_manifest_path)
    step_rows = load_rows(step_index_path)
    with np.load(arrays_path) as npz_handle:
        arrays = {
            "coords_local": np.asarray(npz_handle["coords_local"], dtype=np.float64),
            "gram_raw": np.asarray(npz_handle["gram_raw"], dtype=np.float64),
        }

    token_rows = build_token_rows(run_id=run_id, step_rows=step_rows, arrays=arrays)
    sample_rows = build_sample_rows(run_id=run_id, token_rows=token_rows)
    aggregate_summary = build_aggregate_summary(
        run_id=run_id,
        gate6_manifest=gate6_manifest,
        token_rows=token_rows,
        sample_rows=sample_rows,
    )
    manifest = build_manifest(
        run_id=run_id,
        gate6_dir=gate6_dir,
        gate6_manifest_path=gate6_manifest_path,
        step_index_path=step_index_path,
        arrays_path=arrays_path,
        token_rows=token_rows,
        sample_rows=sample_rows,
    )

    manifest_path = out_dir / DEFAULT_MANIFEST
    token_csv_path = out_dir / DEFAULT_TOKEN_CSV
    sample_csv_path = out_dir / DEFAULT_SAMPLE_CSV
    aggregate_path = out_dir / DEFAULT_AGGREGATE
    checksums_path = out_dir / DEFAULT_CHECKSUMS

    base_consumer.write_json(manifest_path, manifest)
    base_consumer.write_csv(token_csv_path, TOKEN_COLUMNS, token_rows)
    base_consumer.write_csv(sample_csv_path, SAMPLE_COLUMNS, sample_rows)
    base_consumer.write_text(aggregate_path, aggregate_summary)
    base_consumer.write_checksums(
        checksums_path,
        (
            ("manifest_json", manifest_path),
            ("token_telemetry_csv", token_csv_path),
            ("sample_summary_csv", sample_csv_path),
            ("aggregate_summary_md", aggregate_path),
        ),
    )

    print(f"manifest_json={base_consumer.repo_relative_or_posix(manifest_path)}")
    print(f"token_telemetry_csv={base_consumer.repo_relative_or_posix(token_csv_path)}")
    print(f"sample_summary_csv={base_consumer.repo_relative_or_posix(sample_csv_path)}")
    print(f"aggregate_summary_md={base_consumer.repo_relative_or_posix(aggregate_path)}")
    print(f"checksums_json={base_consumer.repo_relative_or_posix(checksums_path)}")
    print(f"n_samples_total={len(sample_rows)}")
    print(f"n_token_rows_total={len(token_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
