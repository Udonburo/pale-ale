#!/usr/bin/env python3
"""Run a Gate6 object-native consumer from singular spectrum and gram invariants."""

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
DEFAULT_TOKEN_CSV = "gate6e_token_telemetry.csv"
DEFAULT_SAMPLE_CSV = "gate6e_sample_summary.csv"
DEFAULT_AGGREGATE = "gate6e_aggregate_summary.md"

SCHEMA_VERSION = "gate6_native_sigma_gram_consumer_artifacts_v1"
METHOD_ID = "gate6_native_object_consumer_sigma_gap_gram_v1"
PRIMARY_METRIC_ID = "sigma_gap_weighted_gram_loop_v1"
PRIMARY_AUX_METRIC_ID = "sigma_gap_rel_v1"
PRIMARY_LEAKAGE_METRIC_ID = "sigma_tail_rel_v1"
BASELINE_METRIC_ID = "score_F_gram_loop_v1"

TOKEN_COLUMNS = (
    "run_id",
    "sample_id",
    "step",
    "token_text",
    "label_token",
    "rank_local",
    "flags_compact",
    "loop_outcome",
    BASELINE_METRIC_ID,
    PRIMARY_METRIC_ID,
    PRIMARY_AUX_METRIC_ID,
    PRIMARY_LEAKAGE_METRIC_ID,
    "sigma_spread_rel_v1",
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
            "Run a Gate6 object-native consumer from gram_raw and singular_values "
            "using spectral-gap weighting."
        )
    )
    parser.add_argument("--gate6-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-id", default="")
    return parser.parse_args()


def sigma_ratios(singular_values: np.ndarray) -> tuple[Optional[float], Optional[float], Optional[float]]:
    sigma = np.asarray(singular_values, dtype=np.float64).reshape(-1)
    if sigma.size < 3:
        raise ValueError(f"singular_values must have at least 3 entries, got {sigma.shape}")
    sigma_1 = float(sigma[0])
    if not np.isfinite(sigma_1) or sigma_1 <= 0.0:
        return (None, None, None)
    sigma_2_rel = float(sigma[1] / sigma_1)
    sigma_3_rel = float(sigma[2] / sigma_1)
    sigma_gap_rel = max(0.0, sigma_2_rel - sigma_3_rel)
    return (sigma_2_rel, sigma_3_rel, sigma_gap_rel)


def compute_sigma_gap_gram_metrics(
    gram_raw: np.ndarray,
    singular_values: np.ndarray,
) -> Dict[str, Any]:
    score_f = base_consumer.build_score_f_gram_loop(np.asarray(gram_raw, dtype=np.float64))
    sigma_2_rel, sigma_3_rel, sigma_gap_rel = sigma_ratios(singular_values)
    if sigma_gap_rel is None or sigma_2_rel is None or sigma_3_rel is None:
        return {
            "loop_outcome": "invalid_singular_values",
            BASELINE_METRIC_ID: score_f,
            PRIMARY_METRIC_ID: None,
            PRIMARY_AUX_METRIC_ID: None,
            PRIMARY_LEAKAGE_METRIC_ID: None,
            "sigma_spread_rel_v1": None,
        }
    return {
        "loop_outcome": "none",
        BASELINE_METRIC_ID: score_f,
        PRIMARY_METRIC_ID: float(score_f * sigma_gap_rel),
        PRIMARY_AUX_METRIC_ID: sigma_gap_rel,
        PRIMARY_LEAKAGE_METRIC_ID: sigma_3_rel,
        "sigma_spread_rel_v1": sigma_2_rel,
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
        metrics = compute_sigma_gap_gram_metrics(
            gram_raw=arrays["gram_raw"][array_row_index],
            singular_values=arrays["singular_values"][array_row_index],
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
                "loop_outcome": metrics["loop_outcome"],
                BASELINE_METRIC_ID: metrics[BASELINE_METRIC_ID],
                PRIMARY_METRIC_ID: metrics[PRIMARY_METRIC_ID],
                PRIMARY_AUX_METRIC_ID: metrics[PRIMARY_AUX_METRIC_ID],
                PRIMARY_LEAKAGE_METRIC_ID: metrics[PRIMARY_LEAKAGE_METRIC_ID],
                "sigma_spread_rel_v1": metrics["sigma_spread_rel_v1"],
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
        "# Gate6 Sigma Gram Consumer Summary",
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
            "gram_raw": np.asarray(npz_handle["gram_raw"], dtype=np.float64),
            "singular_values": np.asarray(npz_handle["singular_values"], dtype=np.float64),
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
