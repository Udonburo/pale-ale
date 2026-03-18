#!/usr/bin/env python3
"""Evaluate fixed Gate8 candidates on conflict cells and quietness pairs."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from aggregate_gate5_spike import mean, parse_float, robust_normalize, write_csv
import build_gate6_native_local_span as gate6_builder
from evaluate_gate6_native_object_seam_pairs import delta, metric_topk_rows, sample_metric_stats
import run_gate6_native_object_consumer as base_consumer


REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "gate8_candidate_standing_eval_v1"
METHOD_ID = "gate8_candidate_standing_eval_v1"
DEFAULT_TOPK = 10
CONFLICT_CELLS = ("direct_contradiction", "distributed_incompatibility")
QUIET_CLEAN_CELL = "clean_support"
QUIET_PERTURBED_CELL = "surface_noisy_clean"
QUIETNESS_PAIRING_RULE = "shared_world_id_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one fixed Gate8 candidate on the frozen conflict cells and "
            "quietness pair surface."
        )
    )
    parser.add_argument("--sample-registry-jsonl", required=True)
    parser.add_argument("--token-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--metric-id", required=True)
    parser.add_argument("--label-key", default="label_token")
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
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
        newline="\n",
    )


def write_text(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8", newline="\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_optional_label(value: Any) -> Optional[int]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    return int(raw)


def load_sample_registry(path: Path) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    rows = read_jsonl(path)
    by_execution_id: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        execution_sample_id = int(row["execution_sample_id"])
        if execution_sample_id in by_execution_id:
            raise ValueError(f"duplicate execution_sample_id in registry: {execution_sample_id}")
        by_execution_id[execution_sample_id] = row
    return rows, by_execution_id


def attach_registry_metadata(
    token_rows: Sequence[Dict[str, str]],
    registry_by_execution_id: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged_rows: List[Dict[str, Any]] = []
    for row in token_rows:
        execution_sample_id = int(row["sample_id"])
        registry_row = registry_by_execution_id.get(execution_sample_id)
        if registry_row is None:
            raise ValueError(
                f"token telemetry references sample_id={execution_sample_id} missing from registry"
            )
        merged_rows.append({**row, **registry_row})
    return merged_rows


def labels_and_scores(
    rows: Sequence[Dict[str, Any]],
    label_key: str,
    metric_id: str,
) -> Tuple[List[int], List[Optional[float]]]:
    labels: List[int] = []
    scores: List[Optional[float]] = []
    for row in rows:
        label = parse_optional_label(row.get(label_key))
        if label is None:
            continue
        labels.append(int(label))
        scores.append(parse_float(row.get(metric_id)))
    return labels, scores


def sample_metrics(
    rows: Sequence[Dict[str, Any]],
    label_key: str,
    metric_id: str,
    topk: int,
) -> Dict[str, Optional[float]]:
    labels, scores = labels_and_scores(rows, label_key=label_key, metric_id=metric_id)
    return {
        "positive_count": float(sum(labels)),
        "auprc": base_consumer.average_precision_optional(labels, scores),
        "hit_at_10": (
            None if topk != 10 else base_consumer.hit_at_k_optional(labels, scores, 10)
        ),
        "hit_at_k": base_consumer.hit_at_k_optional(labels, scores, topk),
        "first_hit_distance": base_consumer.first_hit_distance(labels, scores),
    }


def grouped_rows_by_sample(rows: Sequence[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["execution_sample_id"])].append(row)
    for sample_rows in grouped.values():
        sample_rows.sort(key=lambda row: int(row["step"]))
    return grouped


def build_conflict_cell_summary_rows(
    token_rows: Sequence[Dict[str, Any]],
    label_key: str,
    metric_id: str,
    topk: int,
) -> List[Dict[str, Any]]:
    grouped = grouped_rows_by_sample(token_rows)
    summary_rows: List[Dict[str, Any]] = []
    for cell_id in CONFLICT_CELLS:
        cell_rows = [row for row in token_rows if str(row["cell_id"]) == cell_id]
        labels, scores = labels_and_scores(cell_rows, label_key=label_key, metric_id=metric_id)
        cell_sample_ids = sorted(
            {
                int(row["execution_sample_id"])
                for row in cell_rows
                if parse_optional_label(row.get(label_key)) is not None
            }
        )
        per_sample = [
            sample_metrics(grouped[sample_id], label_key=label_key, metric_id=metric_id, topk=topk)
            for sample_id in cell_sample_ids
        ]
        summary_rows.append(
            {
                "cell_id": cell_id,
                "n_samples_total": len(cell_sample_ids),
                "n_samples_with_positive_labels": sum(
                    1 for row in per_sample if row["positive_count"] is not None and row["positive_count"] > 0
                ),
                "n_rows_total": len(
                    [row for row in cell_rows if parse_optional_label(row.get(label_key)) is not None]
                ),
                "n_positive_labels_total": sum(labels),
                "global_auprc": base_consumer.average_precision_optional(labels, scores),
                "mean_sample_auprc": mean(
                    float(row["auprc"]) for row in per_sample if row["auprc"] is not None
                ),
                "mean_hit_at_10": mean(
                    float(row["hit_at_10"]) for row in per_sample if row["hit_at_10"] is not None
                ),
                "mean_first_hit_distance": mean(
                    float(row["first_hit_distance"])
                    for row in per_sample
                    if row["first_hit_distance"] is not None
                ),
            }
        )
    return summary_rows


def build_quietness_pair_rows(
    token_rows: Sequence[Dict[str, Any]],
    registry_rows: Sequence[Dict[str, Any]],
    metric_id: str,
    topk: int,
) -> List[Dict[str, Any]]:
    grouped_tokens = grouped_rows_by_sample(token_rows)
    pair_bindings: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in registry_rows:
        pair_id = str(row.get("quietness_pair_id") or "")
        if not pair_id:
            continue
        pair_bindings[pair_id][str(row["cell_id"])] = row

    pair_rows: List[Dict[str, Any]] = []
    for pair_id in sorted(pair_bindings):
        pair_binding = pair_bindings[pair_id]
        clean = pair_binding.get(QUIET_CLEAN_CELL)
        perturbed = pair_binding.get(QUIET_PERTURBED_CELL)
        if clean is None or perturbed is None:
            continue
        clean_rows = grouped_tokens.get(int(clean["execution_sample_id"]), [])
        perturbed_rows = grouped_tokens.get(int(perturbed["execution_sample_id"]), [])
        if not clean_rows or not perturbed_rows:
            continue

        clean_stats = sample_metric_stats(clean_rows, metric_id)
        perturbed_stats = sample_metric_stats(perturbed_rows, metric_id)
        clean_p90 = clean_stats["p90"]
        perturbed_top = metric_topk_rows(perturbed_rows, metric_id, topk)

        pair_rows.append(
            {
                "pair_id": pair_id,
                "world_type": str(clean["world_type"]),
                "clean_execution_sample_id": int(clean["execution_sample_id"]),
                "clean_benchmark_sample_id": str(clean["benchmark_sample_id"]),
                "perturbed_execution_sample_id": int(perturbed["execution_sample_id"]),
                "perturbed_benchmark_sample_id": str(perturbed["benchmark_sample_id"]),
                "delta_max": delta(perturbed_stats["max"], clean_stats["max"]),
                "delta_mean": delta(perturbed_stats["mean"], clean_stats["mean"]),
                "delta_p90": delta(perturbed_stats["p90"], clean_stats["p90"]),
                "iqr_normalized_delta_max": robust_normalize(
                    delta(perturbed_stats["max"], clean_stats["max"]),
                    clean_stats["iqr"],
                ),
                "topk_inflation": None
                if clean_p90 is None
                else sum(
                    1
                    for row in perturbed_top
                    if float(parse_float(row.get(metric_id))) >= float(clean_p90)
                ),
            }
        )
    return pair_rows


def summarize_quietness_rows(
    pair_rows: Sequence[Dict[str, Any]],
    topk: int,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    grouped["all"] = list(pair_rows)
    for row in pair_rows:
        grouped[str(row["world_type"])].append(row)

    summary_rows: List[Dict[str, Any]] = []
    for bucket in ("all", "genealogy", "temporal", "reachability"):
        rows = grouped.get(bucket, [])
        if not rows:
            continue
        summary_rows.append(
            {
                "bucket": bucket,
                "n_pairs": len(rows),
                "mean_delta_max": mean(
                    float(row["delta_max"]) for row in rows if row["delta_max"] is not None
                ),
                "mean_delta_mean": mean(
                    float(row["delta_mean"]) for row in rows if row["delta_mean"] is not None
                ),
                "mean_delta_p90": mean(
                    float(row["delta_p90"]) for row in rows if row["delta_p90"] is not None
                ),
                "mean_iqr_normalized_delta_max": mean(
                    float(row["iqr_normalized_delta_max"])
                    for row in rows
                    if row["iqr_normalized_delta_max"] is not None
                ),
                f"mean_top{topk}_inflation": mean(
                    float(row["topk_inflation"]) for row in rows if row["topk_inflation"] is not None
                ),
            }
        )
    return summary_rows


def render_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


def render_int(value: Optional[int]) -> str:
    if value is None:
        return ""
    return str(int(value))


def build_report(
    run_id: str,
    candidate_id: str,
    metric_id: str,
    label_key: str,
    conflict_rows: Sequence[Dict[str, Any]],
    quietness_rows: Sequence[Dict[str, Any]],
    topk: int,
) -> str:
    lines = [
        "# Gate8 Candidate Standing Evaluation",
        "",
        f"run_id: {run_id}",
        f"candidate_id: {candidate_id}",
        f"metric_id: {metric_id}",
        f"label_key: {label_key}",
        f"quietness_pairing_rule: {QUIETNESS_PAIRING_RULE}",
        "",
        "## Conflict Cells",
        "",
        "| cell_id | n_samples | n_positive_labels | global_auprc | mean_sample_auprc | mean_hit@10 | mean_first_hit_distance |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in conflict_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["cell_id"]),
                    str(int(row["n_samples_total"])),
                    str(int(row["n_positive_labels_total"])),
                    render_float(row["global_auprc"]),
                    render_float(row["mean_sample_auprc"]),
                    render_float(row["mean_hit_at_10"]),
                    render_float(row["mean_first_hit_distance"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Quietness",
            "",
            f"| bucket | n_pairs | mean_delta_max | mean_delta_p90 | mean_iqr_normalized_delta_max | mean_top{topk}_inflation |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in quietness_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["bucket"]),
                    render_int(row["n_pairs"]),
                    render_float(row["mean_delta_max"]),
                    render_float(row["mean_delta_p90"]),
                    render_float(row["mean_iqr_normalized_delta_max"]),
                    render_float(row[f"mean_top{topk}_inflation"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def build_manifest(
    run_id: str,
    candidate_id: str,
    metric_id: str,
    label_key: str,
    sample_registry_path: Path,
    token_csv_path: Path,
    topk: int,
    conflict_summary_rows: Sequence[Dict[str, Any]],
    quietness_pair_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "candidate_id": candidate_id,
        "metric_id": metric_id,
        "label_key": label_key,
        "topk": int(topk),
        "conflict_cells": list(CONFLICT_CELLS),
        "quiet_clean_cell": QUIET_CLEAN_CELL,
        "quiet_perturbed_cell": QUIET_PERTURBED_CELL,
        "quietness_pairing_rule": QUIETNESS_PAIRING_RULE,
        "sample_registry_path": repo_relative_or_posix(sample_registry_path),
        "sample_registry_sha256": sha256_file(sample_registry_path),
        "token_csv_path": repo_relative_or_posix(token_csv_path),
        "token_csv_sha256": sha256_file(token_csv_path),
        "code_git_commit": gate6_builder.current_git_commit(),
        "n_conflict_cell_rows": len(conflict_summary_rows),
        "n_quietness_pairs": len(quietness_pair_rows),
    }


def main() -> int:
    args = parse_args()

    sample_registry_path = Path(args.sample_registry_jsonl)
    token_csv_path = Path(args.token_csv)
    out_dir = Path(args.out_dir)
    run_id = args.run_id or out_dir.name

    registry_rows, registry_by_execution_id = load_sample_registry(sample_registry_path)
    token_rows = attach_registry_metadata(
        token_rows=read_csv(token_csv_path),
        registry_by_execution_id=registry_by_execution_id,
    )

    conflict_summary_rows = build_conflict_cell_summary_rows(
        token_rows=token_rows,
        label_key=args.label_key,
        metric_id=args.metric_id,
        topk=args.topk,
    )
    quietness_pair_rows = build_quietness_pair_rows(
        token_rows=token_rows,
        registry_rows=registry_rows,
        metric_id=args.metric_id,
        topk=args.topk,
    )
    quietness_summary_rows = summarize_quietness_rows(
        pair_rows=quietness_pair_rows,
        topk=args.topk,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    conflict_summary_path = out_dir / "conflict_cell_summary.csv"
    quiet_pair_path = out_dir / "quietness_pair_summary.csv"
    quiet_summary_path = out_dir / "quietness_summary.csv"
    report_path = out_dir / "report.md"
    checksums_path = out_dir / "checksums.json"

    manifest = build_manifest(
        run_id=run_id,
        candidate_id=args.candidate_id,
        metric_id=args.metric_id,
        label_key=args.label_key,
        sample_registry_path=sample_registry_path,
        token_csv_path=token_csv_path,
        topk=args.topk,
        conflict_summary_rows=conflict_summary_rows,
        quietness_pair_rows=quietness_pair_rows,
    )
    report = build_report(
        run_id=run_id,
        candidate_id=args.candidate_id,
        metric_id=args.metric_id,
        label_key=args.label_key,
        conflict_rows=conflict_summary_rows,
        quietness_rows=quietness_summary_rows,
        topk=args.topk,
    )

    write_json(manifest_path, manifest)
    write_csv(
        conflict_summary_path,
        (
            "cell_id",
            "n_samples_total",
            "n_samples_with_positive_labels",
            "n_rows_total",
            "n_positive_labels_total",
            "global_auprc",
            "mean_sample_auprc",
            "mean_hit_at_10",
            "mean_first_hit_distance",
        ),
        conflict_summary_rows,
    )
    write_csv(
        quiet_pair_path,
        (
            "pair_id",
            "world_type",
            "clean_execution_sample_id",
            "clean_benchmark_sample_id",
            "perturbed_execution_sample_id",
            "perturbed_benchmark_sample_id",
            "delta_max",
            "delta_mean",
            "delta_p90",
            "iqr_normalized_delta_max",
            "topk_inflation",
        ),
        quietness_pair_rows,
    )
    write_csv(
        quiet_summary_path,
        (
            "bucket",
            "n_pairs",
            "mean_delta_max",
            "mean_delta_mean",
            "mean_delta_p90",
            "mean_iqr_normalized_delta_max",
            f"mean_top{args.topk}_inflation",
        ),
        quietness_summary_rows,
    )
    write_text(report_path, report)
    write_json(
        checksums_path,
        {
            "manifest.json": sha256_file(manifest_path),
            "conflict_cell_summary.csv": sha256_file(conflict_summary_path),
            "quietness_pair_summary.csv": sha256_file(quiet_pair_path),
            "quietness_summary.csv": sha256_file(quiet_summary_path),
            "report.md": sha256_file(report_path),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
