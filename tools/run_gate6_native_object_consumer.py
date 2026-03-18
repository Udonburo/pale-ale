#!/usr/bin/env python3
"""Run a minimal Gate6-B object-native holonomy consumer on Gate6 artifacts."""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import build_gate6_native_local_span as gate6_builder


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MANIFEST = "manifest.json"
DEFAULT_STEP_INDEX = "step_index.jsonl"
DEFAULT_ARRAYS = "native_object_arrays.npz"
DEFAULT_CHECKSUMS = "checksums.json"
DEFAULT_TOKEN_CSV = "gate6b_token_telemetry.csv"
DEFAULT_SAMPLE_CSV = "gate6b_sample_summary.csv"
DEFAULT_AGGREGATE = "gate6b_aggregate_summary.md"

SCHEMA_VERSION = "gate6_native_object_consumer_artifacts_v1"
METHOD_ID = "gate6_native_object_consumer_edge_plane_v1"
PRIMARY_METRIC_ID = "edge_plane_loop_projective_chordal_v1"
PRIMARY_AUX_METRIC_ID = "edge_plane_loop_geodesic_angle_v1"
PRIMARY_LEAKAGE_METRIC_ID = "edge_plane_loop_frob_identity_v1"
BASELINE_METRIC_ID = "score_F_gram_loop_v1"

TAU_NORMAL_ABS = 1e-12
TAU_ROTATION_CROSS_ABS = 1e-12
TAU_ROTATION_ANTIPODAL_DOT = -1.0 + 1e-12

EDGE_NAMES = (
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
    "edge_plane_outcome_v_to_splus",
    "edge_plane_outcome_splus_to_sminus",
    "edge_plane_outcome_sminus_to_v",
    "transport_outcome_vp_to_pm",
    "transport_outcome_pm_to_mv",
    "transport_outcome_mv_to_vp",
    "loop_outcome",
    "score_F_gram_loop_v1",
    "edge_plane_loop_projective_chordal_v1",
    "edge_plane_loop_geodesic_angle_v1",
    "edge_plane_loop_frob_identity_v1",
    "edge_plane_loop_det_v1",
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
    "auprc_edge_plane_loop_projective_chordal_v1",
    "delta_auprc_edge_plane_loop_projective_chordal_v1_vs_F_gram_loop_v1",
    "hit_at_10_F_gram_loop_v1",
    "hit_at_10_edge_plane_loop_projective_chordal_v1",
    "mean_edge_plane_loop_projective_chordal_v1",
    "max_edge_plane_loop_projective_chordal_v1",
    "p90_edge_plane_loop_projective_chordal_v1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a minimal object-native Gate6 consumer that computes edge-plane "
            "holonomy directly from coords_local and gram_raw."
        )
    )
    parser.add_argument("--gate6-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-id", default="")
    return parser.parse_args()


def repo_relative_or_posix(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_text(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8", newline="\n")


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def d_proj_from_dot(dot_value: float) -> float:
    return math.sqrt(max(0.0, 2.0 * (1.0 - min(1.0, abs(dot_value)))))


def normalize3(vector: np.ndarray, tau_abs: float = TAU_NORMAL_ABS) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= tau_abs:
        return None
    return vector / norm


def plane_normal_from_pair(left: np.ndarray, right: np.ndarray) -> Tuple[str, Optional[np.ndarray]]:
    raw_normal = np.cross(left, right)
    normal = normalize3(raw_normal)
    if normal is None:
        return ("collinear_pair", None)
    fixed, _flipped, _anchor_index = gate6_builder.sign_fix_column(normal)
    return ("materialized", fixed)


def skew_symmetric(axis: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )


def minimal_rotation_matrix(left: np.ndarray, right: np.ndarray) -> Tuple[str, Optional[np.ndarray]]:
    dot_value = float(np.clip(np.dot(left, right), -1.0, 1.0))
    cross = np.cross(left, right)
    cross_norm = float(np.linalg.norm(cross))
    if not math.isfinite(dot_value) or not math.isfinite(cross_norm):
        return ("invalid_transport", None)
    if cross_norm <= TAU_ROTATION_CROSS_ABS:
        if dot_value <= TAU_ROTATION_ANTIPODAL_DOT:
            return ("antipodal_normals", None)
        return ("collinear_identity", np.eye(3, dtype=np.float64))

    axis = cross / cross_norm
    angle = math.atan2(cross_norm, dot_value)
    k_matrix = skew_symmetric(axis)
    rotation = (
        np.eye(3, dtype=np.float64)
        + math.sin(angle) * k_matrix
        + (1.0 - math.cos(angle)) * (k_matrix @ k_matrix)
    )
    return ("materialized", rotation)


def quaternion_scalar_abs_from_rotation(rotation: np.ndarray) -> float:
    trace_value = float(np.trace(rotation))
    return math.sqrt(max(0.0, 1.0 + trace_value)) / 2.0


def geodesic_angle_from_rotation(rotation: np.ndarray) -> float:
    cos_theta = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return math.acos(cos_theta)


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    positive_count = sum(1 for label in labels if int(label) == 1)
    if positive_count == 0:
        return None
    order = sorted(range(len(scores)), key=lambda idx: (-float(scores[idx]), idx))
    tp = 0
    fp = 0
    prev_recall = 0.0
    ap = 0.0
    for idx in order:
        if int(labels[idx]) == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / positive_count
        precision = tp / (tp + fp)
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return ap


def average_precision_optional(labels: Sequence[int], scores: Sequence[Optional[float]]) -> Optional[float]:
    filtered_labels: List[int] = []
    filtered_scores: List[float] = []
    for label, score in zip(labels, scores):
        if score is None:
            continue
        filtered_labels.append(int(label))
        filtered_scores.append(float(score))
    return average_precision(filtered_labels, filtered_scores)


def hit_at_k_optional(labels: Sequence[int], scores: Sequence[Optional[float]], k: int) -> Optional[int]:
    indexed = [(idx, float(score)) for idx, score in enumerate(scores) if score is not None]
    if not indexed or all(int(label) == 0 for label in labels):
        return None
    indexed.sort(key=lambda pair: (-pair[1], pair[0]))
    return sum(1 for idx, _score in indexed[:k] if int(labels[idx]) == 1)


def first_hit_distance(labels: Sequence[int], scores: Sequence[Optional[float]]) -> Optional[int]:
    indexed = [(idx, float(score)) for idx, score in enumerate(scores) if score is not None]
    if not indexed or all(int(label) == 0 for label in labels):
        return None
    indexed.sort(key=lambda pair: (-pair[1], pair[0]))
    for rank, (idx, _score) in enumerate(indexed):
        if int(labels[idx]) == 1:
            return rank
    return None


def build_score_f_gram_loop(gram_raw: np.ndarray) -> float:
    score = 0.0
    for left_idx, right_idx in PAIR_INDICES:
        score += d_proj_from_dot(float(gram_raw[left_idx, right_idx]))
    return score


def compute_edge_plane_loop_metrics(coords_local: np.ndarray, gram_raw: np.ndarray) -> Dict[str, Any]:
    coords = np.asarray(coords_local, dtype=np.float64)
    if coords.shape != (3, 3):
        raise ValueError(f"coords_local must be 3x3, got {coords.shape}")

    observables = [coords[:, 0], coords[:, 1], coords[:, 2]]
    edge_outcomes: List[str] = []
    normals: List[Optional[np.ndarray]] = []
    for left_idx, right_idx in PAIR_INDICES:
        outcome, normal = plane_normal_from_pair(observables[left_idx], observables[right_idx])
        edge_outcomes.append(outcome)
        normals.append(normal)

    score_f = build_score_f_gram_loop(np.asarray(gram_raw, dtype=np.float64))
    if any(normal is None for normal in normals):
        return {
            "edge_plane_outcomes": edge_outcomes,
            "transport_outcomes": ["skipped_missing_plane"] * 3,
            "loop_outcome": "partial_loop_missing",
            "score_F_gram_loop_v1": score_f,
            PRIMARY_METRIC_ID: None,
            PRIMARY_AUX_METRIC_ID: None,
            PRIMARY_LEAKAGE_METRIC_ID: None,
            "edge_plane_loop_det_v1": None,
        }

    transport_outcomes: List[str] = []
    rotations: List[Optional[np.ndarray]] = []
    normal_triplet = [normal for normal in normals if normal is not None]
    for left_normal, right_normal in (
        (normal_triplet[0], normal_triplet[1]),
        (normal_triplet[1], normal_triplet[2]),
        (normal_triplet[2], normal_triplet[0]),
    ):
        outcome, rotation = minimal_rotation_matrix(left_normal, right_normal)
        transport_outcomes.append(outcome)
        rotations.append(rotation)

    if any(rotation is None for rotation in rotations):
        return {
            "edge_plane_outcomes": edge_outcomes,
            "transport_outcomes": transport_outcomes,
            "loop_outcome": "partial_loop_missing",
            "score_F_gram_loop_v1": score_f,
            PRIMARY_METRIC_ID: None,
            PRIMARY_AUX_METRIC_ID: None,
            PRIMARY_LEAKAGE_METRIC_ID: None,
            "edge_plane_loop_det_v1": None,
        }

    holonomy = rotations[2] @ rotations[1] @ rotations[0]
    quat_abs = quaternion_scalar_abs_from_rotation(holonomy)
    chordal = math.sqrt(max(0.0, 2.0 * (1.0 - min(1.0, abs(quat_abs)))))
    geodesic = geodesic_angle_from_rotation(holonomy)
    fro_identity = float(np.linalg.norm(holonomy - np.eye(3, dtype=np.float64), ord="fro"))
    det_value = float(np.linalg.det(holonomy))
    if not all(math.isfinite(value) for value in (chordal, geodesic, fro_identity, det_value)):
        return {
            "edge_plane_outcomes": edge_outcomes,
            "transport_outcomes": transport_outcomes,
            "loop_outcome": "invalid_holonomy",
            "score_F_gram_loop_v1": score_f,
            PRIMARY_METRIC_ID: None,
            PRIMARY_AUX_METRIC_ID: None,
            PRIMARY_LEAKAGE_METRIC_ID: None,
            "edge_plane_loop_det_v1": None,
        }

    return {
        "edge_plane_outcomes": edge_outcomes,
        "transport_outcomes": transport_outcomes,
        "loop_outcome": "none",
        "score_F_gram_loop_v1": score_f,
        PRIMARY_METRIC_ID: chordal,
        PRIMARY_AUX_METRIC_ID: geodesic,
        PRIMARY_LEAKAGE_METRIC_ID: fro_identity,
        "edge_plane_loop_det_v1": det_value,
    }


def load_rows(step_index_path: Path) -> List[Dict[str, Any]]:
    rows = read_jsonl(step_index_path)
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
        metrics = compute_edge_plane_loop_metrics(
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
                "edge_plane_outcome_v_to_splus": metrics["edge_plane_outcomes"][0],
                "edge_plane_outcome_splus_to_sminus": metrics["edge_plane_outcomes"][1],
                "edge_plane_outcome_sminus_to_v": metrics["edge_plane_outcomes"][2],
                "transport_outcome_vp_to_pm": metrics["transport_outcomes"][0],
                "transport_outcome_pm_to_mv": metrics["transport_outcomes"][1],
                "transport_outcome_mv_to_vp": metrics["transport_outcomes"][2],
                "loop_outcome": metrics["loop_outcome"],
                "score_F_gram_loop_v1": metrics["score_F_gram_loop_v1"],
                PRIMARY_METRIC_ID: metrics[PRIMARY_METRIC_ID],
                PRIMARY_AUX_METRIC_ID: metrics[PRIMARY_AUX_METRIC_ID],
                PRIMARY_LEAKAGE_METRIC_ID: metrics[PRIMARY_LEAKAGE_METRIC_ID],
                "edge_plane_loop_det_v1": metrics["edge_plane_loop_det_v1"],
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
        scores_f = [float(row["score_F_gram_loop_v1"]) for row in rows]
        scores_primary = [
            None if row[PRIMARY_METRIC_ID] is None else float(row[PRIMARY_METRIC_ID]) for row in rows
        ]
        auprc_f = average_precision(labels, scores_f)
        auprc_primary = average_precision_optional(labels, scores_primary)
        delta_auprc = (
            None
            if auprc_f is None or auprc_primary is None
            else float(auprc_primary - auprc_f)
        )
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
                "auprc_edge_plane_loop_projective_chordal_v1": auprc_primary,
                "delta_auprc_edge_plane_loop_projective_chordal_v1_vs_F_gram_loop_v1": delta_auprc,
                "hit_at_10_F_gram_loop_v1": hit_at_k_optional(labels, [float(v) for v in scores_f], 10),
                "hit_at_10_edge_plane_loop_projective_chordal_v1": hit_at_k_optional(
                    labels, scores_primary, 10
                ),
                "mean_edge_plane_loop_projective_chordal_v1": (
                    None if not valid_primary else float(np.mean(valid_primary))
                ),
                "max_edge_plane_loop_projective_chordal_v1": (
                    None if not valid_primary else float(np.max(valid_primary))
                ),
                "p90_edge_plane_loop_projective_chordal_v1": percentile(valid_primary, 90.0),
            }
        )
    return sample_rows


def render_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


def render_int(value: Optional[int]) -> str:
    if value is None:
        return ""
    return str(int(value))


def build_aggregate_summary(
    run_id: str,
    gate6_manifest: Dict[str, Any],
    token_rows: Sequence[Dict[str, Any]],
    sample_rows: Sequence[Dict[str, Any]],
) -> str:
    labels = [int(row["label_token"]) for row in token_rows]
    scores_f = [float(row["score_F_gram_loop_v1"]) for row in token_rows]
    scores_primary = [
        None if row[PRIMARY_METRIC_ID] is None else float(row[PRIMARY_METRIC_ID]) for row in token_rows
    ]
    global_auprc_f = average_precision(labels, scores_f)
    global_auprc_primary = average_precision_optional(labels, scores_primary)
    mean_sample_auprc_f_values = [
        float(row["auprc_F_gram_loop_v1"])
        for row in sample_rows
        if row["auprc_F_gram_loop_v1"] is not None
    ]
    mean_sample_auprc_primary_values = [
        float(row["auprc_edge_plane_loop_projective_chordal_v1"])
        for row in sample_rows
        if row["auprc_edge_plane_loop_projective_chordal_v1"] is not None
    ]
    mean_hit_f_values = [
        int(row["hit_at_10_F_gram_loop_v1"])
        for row in sample_rows
        if row["hit_at_10_F_gram_loop_v1"] is not None
    ]
    mean_hit_primary_values = [
        int(row["hit_at_10_edge_plane_loop_projective_chordal_v1"])
        for row in sample_rows
        if row["hit_at_10_edge_plane_loop_projective_chordal_v1"] is not None
    ]
    first_hit_primary_values = [
        value
        for value in (
            first_hit_distance(
                [int(row["label_token"]) for row in sample_token_rows],
                [
                    None
                    if metric_row[PRIMARY_METRIC_ID] is None
                    else float(metric_row[PRIMARY_METRIC_ID])
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
        "# Gate6 Native Object Consumer Summary",
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
        f"- global_auprc_{BASELINE_METRIC_ID}: {render_float(global_auprc_f)}",
        f"- global_auprc_{PRIMARY_METRIC_ID}: {render_float(global_auprc_primary)}",
        f"- mean_sample_auprc_{BASELINE_METRIC_ID}: {render_float(None if not mean_sample_auprc_f_values else float(np.mean(mean_sample_auprc_f_values)))}",
        f"- mean_sample_auprc_{PRIMARY_METRIC_ID}: {render_float(None if not mean_sample_auprc_primary_values else float(np.mean(mean_sample_auprc_primary_values)))}",
        f"- mean_hit@10_{BASELINE_METRIC_ID}: {render_float(None if not mean_hit_f_values else float(np.mean(mean_hit_f_values)))}",
        f"- mean_hit@10_{PRIMARY_METRIC_ID}: {render_float(None if not mean_hit_primary_values else float(np.mean(mean_hit_primary_values)))}",
        f"- mean_first_hit_{PRIMARY_METRIC_ID}: {render_float(None if not first_hit_primary_values else float(np.mean(first_hit_primary_values)))}",
        f"- mean_{PRIMARY_METRIC_ID}: {render_float(None if not primary_values else float(np.mean(primary_values)))}",
        f"- p90_{PRIMARY_METRIC_ID}: {render_float(percentile(primary_values, 90.0))}",
        f"- max_{PRIMARY_METRIC_ID}: {render_float(None if not primary_values else float(np.max(primary_values)))}",
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
        "input_gate6_dir": repo_relative_or_posix(gate6_dir),
        "input_gate6_manifest_path": repo_relative_or_posix(gate6_manifest_path),
        "input_gate6_manifest_sha256": sha256_file(gate6_manifest_path),
        "input_gate6_step_index_path": repo_relative_or_posix(step_index_path),
        "input_gate6_step_index_sha256": sha256_file(step_index_path),
        "input_gate6_arrays_path": repo_relative_or_posix(arrays_path),
        "input_gate6_arrays_sha256": sha256_file(arrays_path),
        "input_gate6_method_id": read_json(gate6_manifest_path).get("method_id", ""),
        "code_git_commit": gate6_builder.current_git_commit(),
        "tau_normal_abs": TAU_NORMAL_ABS,
        "tau_rotation_cross_abs": TAU_ROTATION_CROSS_ABS,
        "tau_rotation_antipodal_dot": TAU_ROTATION_ANTIPODAL_DOT,
        "normal_sign_fix_mode": gate6_builder.SIGN_FIX_MODE,
        "n_samples_total": len(sample_rows),
        "n_token_rows_total": len(token_rows),
        "n_loop_rows_valid": sum(1 for row in token_rows if row["loop_outcome"] == "none"),
        "n_loop_rows_missing": sum(1 for row in token_rows if row["loop_outcome"] != "none"),
    }


def write_checksums(path: Path, artifact_paths: Sequence[Tuple[str, Path]]) -> None:
    payload = {
        name: {
            "path": repo_relative_or_posix(artifact_path),
            "sha256": sha256_file(artifact_path),
        }
        for name, artifact_path in artifact_paths
    }
    write_json(path, payload)


def main() -> int:
    args = parse_args()
    gate6_dir = (REPO_ROOT / args.gate6_dir).resolve()
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    run_id = args.run_id or out_dir.name

    gate6_manifest_path = gate6_dir / DEFAULT_MANIFEST
    step_index_path = gate6_dir / DEFAULT_STEP_INDEX
    arrays_path = gate6_dir / DEFAULT_ARRAYS

    gate6_manifest = read_json(gate6_manifest_path)
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

    write_json(manifest_path, manifest)
    write_csv(token_csv_path, TOKEN_COLUMNS, token_rows)
    write_csv(sample_csv_path, SAMPLE_COLUMNS, sample_rows)
    write_text(aggregate_path, aggregate_summary)
    write_checksums(
        checksums_path,
        (
            ("manifest_json", manifest_path),
            ("token_telemetry_csv", token_csv_path),
            ("sample_summary_csv", sample_csv_path),
            ("aggregate_summary_md", aggregate_path),
        ),
    )

    print(f"manifest_json={repo_relative_or_posix(manifest_path)}")
    print(f"token_telemetry_csv={repo_relative_or_posix(token_csv_path)}")
    print(f"sample_summary_csv={repo_relative_or_posix(sample_csv_path)}")
    print(f"aggregate_summary_md={repo_relative_or_posix(aggregate_path)}")
    print(f"checksums_json={repo_relative_or_posix(checksums_path)}")
    print(f"n_samples_total={len(sample_rows)}")
    print(f"n_token_rows_total={len(token_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
