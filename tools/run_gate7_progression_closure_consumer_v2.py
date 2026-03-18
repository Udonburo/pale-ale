#!/usr/bin/env python3
"""Run a Gate7 two-step projector-closure consumer from Gate6 native objects."""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import build_gate6_native_local_span as gate6_builder
import run_gate6_native_object_consumer as base_consumer


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MANIFEST = "manifest.json"
DEFAULT_STEP_INDEX = "step_index.jsonl"
DEFAULT_ARRAYS = "native_object_arrays.npz"
DEFAULT_CHECKSUMS = "checksums.json"
DEFAULT_TOKEN_CSV = "gate7b_token_telemetry.csv"
DEFAULT_SAMPLE_CSV = "gate7b_sample_summary.csv"
DEFAULT_AGGREGATE = "gate7b_aggregate_summary.md"

SCHEMA_VERSION = "gate7_progression_closure_consumer_artifacts_v2"
METHOD_ID = "gate7_progression_closure_consumer_v2"
PRIMARY_METRIC_ID = "progression_closure_v2"
PRIMARY_AUX_METRIC_ID = "progression_closure_energy_ratio_v2"
PRIMARY_LEAKAGE_METRIC_ID = "progression_closure_projected_norm_v2"
BASELINE_METRIC_ID = "score_F_gram_loop_v1"
STRUCTURAL_NO_SUCCESSOR_OUTCOME = "final_step_no_successor"

TOKEN_COLUMNS = (
    "run_id",
    "sample_id",
    "step",
    "token_text",
    "label_token",
    "label_transition",
    "rank_local",
    "rank_local_next",
    "flags_compact",
    "loop_outcome",
    BASELINE_METRIC_ID,
    PRIMARY_METRIC_ID,
    PRIMARY_AUX_METRIC_ID,
    PRIMARY_LEAKAGE_METRIC_ID,
    "baseline_logprob",
    "baseline_entropy",
)

SAMPLE_COLUMNS = (
    "run_id",
    "sample_id",
    "n_token_steps",
    "n_loop_steps_valid",
    "n_loop_steps_structural_no_successor",
    "n_loop_steps_missing",
    "positive_transition_count",
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
            "Run a Gate7 projector-native progression closure consumer that measures "
            "how much of v_{t+1} survives two-step closure through P_t and P_{t+1}."
        )
    )
    parser.add_argument("--gate6-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-id", default="")
    return parser.parse_args()


def load_rows(step_index_path: Path) -> List[Dict[str, Any]]:
    rows = base_consumer.read_jsonl(step_index_path)
    rows.sort(key=lambda row: (int(row["sample_id"]), int(row["step"])))
    return rows


def reconstruct_v(
    basis: np.ndarray,
    coords_local: np.ndarray,
    rank_local: int,
) -> Optional[np.ndarray]:
    if rank_local <= 0:
        return None
    basis_slice = np.asarray(basis[:, :rank_local], dtype=np.float64)
    coord_slice = np.asarray(coords_local[:rank_local, 0], dtype=np.float64)
    vector = basis_slice @ coord_slice
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        return None
    return vector


def compute_progression_metrics(
    current_basis: np.ndarray,
    current_rank: int,
    current_gram_raw: np.ndarray,
    next_basis: np.ndarray,
    next_coords_local: np.ndarray,
    next_rank: int,
) -> Dict[str, Any]:
    score_f = base_consumer.build_score_f_gram_loop(np.asarray(current_gram_raw, dtype=np.float64))
    if current_rank <= 0:
        return {
            "loop_outcome": "invalid_current_projector",
            BASELINE_METRIC_ID: None,
            PRIMARY_METRIC_ID: None,
            PRIMARY_AUX_METRIC_ID: None,
            PRIMARY_LEAKAGE_METRIC_ID: None,
        }
    if next_rank <= 0:
        return {
            "loop_outcome": "invalid_next_projector",
            BASELINE_METRIC_ID: None,
            PRIMARY_METRIC_ID: None,
            PRIMARY_AUX_METRIC_ID: None,
            PRIMARY_LEAKAGE_METRIC_ID: None,
        }

    next_v = reconstruct_v(next_basis, next_coords_local, next_rank)
    if next_v is None:
        return {
            "loop_outcome": "invalid_next_vector",
            BASELINE_METRIC_ID: None,
            PRIMARY_METRIC_ID: None,
            PRIMARY_AUX_METRIC_ID: None,
            PRIMARY_LEAKAGE_METRIC_ID: None,
        }

    current_basis_slice = np.asarray(current_basis[:, :current_rank], dtype=np.float64)
    next_basis_slice = np.asarray(next_basis[:, :next_rank], dtype=np.float64)

    current_projected_coeffs = current_basis_slice.T @ next_v
    current_projected = current_basis_slice @ current_projected_coeffs
    closure_projected_coeffs = next_basis_slice.T @ current_projected
    closure_projected = next_basis_slice @ closure_projected_coeffs

    next_norm_sq = float(np.dot(next_v, next_v))
    closure_projected_norm_sq = float(np.dot(closure_projected, closure_projected))
    if (
        not np.isfinite(next_norm_sq)
        or next_norm_sq <= 1e-12
        or not np.isfinite(closure_projected_norm_sq)
    ):
        return {
            "loop_outcome": "invalid_closure_energy",
            BASELINE_METRIC_ID: None,
            PRIMARY_METRIC_ID: None,
            PRIMARY_AUX_METRIC_ID: None,
            PRIMARY_LEAKAGE_METRIC_ID: None,
        }

    closure_energy_ratio = float(np.clip(closure_projected_norm_sq / next_norm_sq, 0.0, 1.0))
    progression_closure = float(np.clip(1.0 - closure_energy_ratio, 0.0, 1.0))
    closure_projected_norm = float(np.sqrt(max(0.0, closure_projected_norm_sq)))
    return {
        "loop_outcome": "none",
        BASELINE_METRIC_ID: score_f,
        PRIMARY_METRIC_ID: progression_closure,
        PRIMARY_AUX_METRIC_ID: closure_energy_ratio,
        PRIMARY_LEAKAGE_METRIC_ID: closure_projected_norm,
    }


def build_token_rows(
    run_id: str,
    step_rows: Sequence[Dict[str, Any]],
    arrays: Dict[str, np.ndarray],
) -> List[Dict[str, Any]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in step_rows:
        grouped[int(row["sample_id"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["step"]))

    token_rows: List[Dict[str, Any]] = []
    for sample_id in sorted(grouped):
        rows = grouped[sample_id]
        for idx, step_row in enumerate(rows):
            array_row_index = int(step_row["array_row_index"])
            is_final = idx + 1 >= len(rows)
            next_row = None if is_final else rows[idx + 1]
            if is_final:
                metrics = {
                    "loop_outcome": STRUCTURAL_NO_SUCCESSOR_OUTCOME,
                    BASELINE_METRIC_ID: None,
                    PRIMARY_METRIC_ID: None,
                    PRIMARY_AUX_METRIC_ID: None,
                    PRIMARY_LEAKAGE_METRIC_ID: None,
                }
                label_transition = None
                rank_local_next = None
            else:
                next_row_index = int(next_row["array_row_index"])
                metrics = compute_progression_metrics(
                    current_basis=arrays["basis"][array_row_index],
                    current_rank=int(arrays["rank_local"][array_row_index]),
                    current_gram_raw=arrays["gram_raw"][array_row_index],
                    next_basis=arrays["basis"][next_row_index],
                    next_coords_local=arrays["coords_local"][next_row_index],
                    next_rank=int(arrays["rank_local"][next_row_index]),
                )
                label_transition = max(int(step_row["label_token"]), int(next_row["label_token"]))
                rank_local_next = int(arrays["rank_local"][next_row_index])

            token_rows.append(
                {
                    "run_id": run_id,
                    "sample_id": int(step_row["sample_id"]),
                    "step": int(step_row["step"]),
                    "token_text": str(step_row["token_text"]),
                    "label_token": int(step_row["label_token"]),
                    "label_transition": label_transition,
                    "rank_local": int(step_row["rank_local"]),
                    "rank_local_next": rank_local_next,
                    "flags_compact": str(step_row["flags_compact"]),
                    "loop_outcome": metrics["loop_outcome"],
                    BASELINE_METRIC_ID: metrics[BASELINE_METRIC_ID],
                    PRIMARY_METRIC_ID: metrics[PRIMARY_METRIC_ID],
                    PRIMARY_AUX_METRIC_ID: metrics[PRIMARY_AUX_METRIC_ID],
                    PRIMARY_LEAKAGE_METRIC_ID: metrics[PRIMARY_LEAKAGE_METRIC_ID],
                    "baseline_logprob": float(step_row["baseline_logprob"]),
                    "baseline_entropy": float(step_row["baseline_entropy"]),
                }
            )
    return token_rows


def count_valid_rows(rows: Sequence[Dict[str, Any]]) -> int:
    return sum(1 for row in rows if row["loop_outcome"] == "none")


def count_structural_no_successor_rows(rows: Sequence[Dict[str, Any]]) -> int:
    return sum(1 for row in rows if row["loop_outcome"] == STRUCTURAL_NO_SUCCESSOR_OUTCOME)


def count_invalid_missing_rows(rows: Sequence[Dict[str, Any]]) -> int:
    return sum(
        1
        for row in rows
        if row["loop_outcome"] not in ("none", STRUCTURAL_NO_SUCCESSOR_OUTCOME)
    )


def average_precision_on_valid_rows(
    rows: Sequence[Dict[str, Any]],
    label_key: str,
    score_key: str,
) -> Optional[float]:
    labels: List[int] = []
    scores: List[float] = []
    for row in rows:
        score = row[score_key]
        label = row[label_key]
        if score is None or label is None:
            continue
        labels.append(int(label))
        scores.append(float(score))
    return base_consumer.average_precision(labels, scores)


def hit_at_k_on_valid_rows(
    rows: Sequence[Dict[str, Any]],
    label_key: str,
    score_key: str,
    k: int,
) -> Optional[int]:
    labels: List[int] = []
    scores: List[Optional[float]] = []
    for row in rows:
        label = row[label_key]
        if label is None:
            continue
        labels.append(int(label))
        scores.append(None if row[score_key] is None else float(row[score_key]))
    return base_consumer.hit_at_k_optional(labels, scores, k)


def first_hit_distance_on_valid_rows(
    rows: Sequence[Dict[str, Any]],
    label_key: str,
    score_key: str,
) -> Optional[int]:
    labels: List[int] = []
    scores: List[Optional[float]] = []
    for row in rows:
        label = row[label_key]
        if label is None:
            continue
        labels.append(int(label))
        scores.append(None if row[score_key] is None else float(row[score_key]))
    return base_consumer.first_hit_distance(labels, scores)


def build_sample_rows(run_id: str, token_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in token_rows:
        grouped[int(row["sample_id"])].append(row)

    sample_rows: List[Dict[str, Any]] = []
    for sample_id in sorted(grouped):
        rows = sorted(grouped[sample_id], key=lambda row: int(row["step"]))
        valid_primary = [
            float(row[PRIMARY_METRIC_ID]) for row in rows if row[PRIMARY_METRIC_ID] is not None
        ]
        auprc_f = average_precision_on_valid_rows(rows, "label_transition", BASELINE_METRIC_ID)
        auprc_primary = average_precision_on_valid_rows(rows, "label_transition", PRIMARY_METRIC_ID)
        delta_auprc = None if auprc_f is None or auprc_primary is None else float(auprc_primary - auprc_f)
        sample_rows.append(
            {
                "run_id": run_id,
                "sample_id": sample_id,
                "n_token_steps": len(rows),
                "n_loop_steps_valid": count_valid_rows(rows),
                "n_loop_steps_structural_no_successor": count_structural_no_successor_rows(rows),
                "n_loop_steps_missing": count_invalid_missing_rows(rows),
                "positive_transition_count": sum(
                    1 for row in rows if row["label_transition"] is not None and int(row["label_transition"]) == 1
                ),
                "auprc_F_gram_loop_v1": auprc_f,
                f"auprc_{PRIMARY_METRIC_ID}": auprc_primary,
                f"delta_auprc_{PRIMARY_METRIC_ID}_vs_F_gram_loop_v1": delta_auprc,
                "hit_at_10_F_gram_loop_v1": hit_at_k_on_valid_rows(rows, "label_transition", BASELINE_METRIC_ID, 10),
                f"hit_at_10_{PRIMARY_METRIC_ID}": hit_at_k_on_valid_rows(rows, "label_transition", PRIMARY_METRIC_ID, 10),
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
    valid_rows = [row for row in token_rows if row["label_transition"] is not None]
    labels = [int(row["label_transition"]) for row in valid_rows]
    scores_f = [None if row[BASELINE_METRIC_ID] is None else float(row[BASELINE_METRIC_ID]) for row in valid_rows]
    scores_primary = [None if row[PRIMARY_METRIC_ID] is None else float(row[PRIMARY_METRIC_ID]) for row in valid_rows]
    global_auprc_f = base_consumer.average_precision_optional(labels, scores_f)
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
    n_loop_rows_valid = count_valid_rows(token_rows)
    n_loop_rows_structural_no_successor = count_structural_no_successor_rows(token_rows)
    n_loop_rows_missing = count_invalid_missing_rows(token_rows)
    first_hit_primary_values = [
        value
        for value in (
            first_hit_distance_on_valid_rows(
                [row for row in token_rows if int(row["sample_id"]) == int(sample_row["sample_id"])],
                "label_transition",
                PRIMARY_METRIC_ID,
            )
            for sample_row in sample_rows
        )
        if value is not None
    ]
    primary_values = [float(score) for score in scores_primary if score is not None]
    lines = [
        "# Gate7 Progression Closure Consumer Summary",
        "",
        f"run_id: {run_id}",
        f"method_id: {METHOD_ID}",
        f"input_gate6_run_id: {gate6_manifest.get('run_id', '')}",
        f"n_samples_total: {len(sample_rows)}",
        f"n_token_rows_total: {len(token_rows)}",
        f"n_loop_rows_valid: {n_loop_rows_valid}",
        f"n_loop_rows_structural_no_successor: {n_loop_rows_structural_no_successor}",
        f"n_loop_rows_missing: {n_loop_rows_missing}",
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
        "n_samples_total": len(sample_rows),
        "n_token_rows_total": len(token_rows),
        "n_loop_rows_valid": count_valid_rows(token_rows),
        "n_loop_rows_structural_no_successor": count_structural_no_successor_rows(token_rows),
        "n_loop_rows_missing": count_invalid_missing_rows(token_rows),
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
            "basis": np.asarray(npz_handle["basis"], dtype=np.float64),
            "coords_local": np.asarray(npz_handle["coords_local"], dtype=np.float64),
            "gram_raw": np.asarray(npz_handle["gram_raw"], dtype=np.float64),
            "rank_local": np.asarray(npz_handle["rank_local"], dtype=np.int64),
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
