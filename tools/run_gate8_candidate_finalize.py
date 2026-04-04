#!/usr/bin/env python3
"""Finalize a partially completed Gate8 candidate execution bundle."""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import run_gate8_candidate_batch as batch
import run_gate7_progression_anisotropic_consumer_v3 as gate7c_consumer


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Finish the late-stage evaluation and manifest closeout for an existing "
            "Gate8 candidate execution bundle that already contains samples, gate6, "
            "gate7, and diagnostic bridge artifacts."
        )
    )
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--execution-dir", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", default="unknown")
    parser.add_argument("--device", default="unknown")
    parser.add_argument("--topk", type=int, default=batch.DEFAULT_TOPK)
    parser.add_argument("--seed", type=int, default=batch.DEFAULT_SEED)
    return parser.parse_args()


def run_subprocess(command: Sequence[str]) -> None:
    completed = subprocess.run(
        list(command),
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed rc={completed.returncode}: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"required file not found: {path}")


def require_dir(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"required directory not found: {path}")


def load_anchor_objects(
    extraction_rows: Sequence[Dict[str, Any]],
) -> tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    support_anchor_objects: Dict[int, Dict[str, Any]] = {}
    conflict_anchor_objects: Dict[int, Dict[str, Any]] = {}

    for row in extraction_rows:
        execution_sample_id = int(row["execution_sample_id"])
        sample_dir = REPO_ROOT / str(row["sample_dir"])

        support_triplets_path = sample_dir / batch.SUPPORT_ANCHOR_TRIPLETS_FILENAME
        support_meta_path = sample_dir / batch.SUPPORT_ANCHOR_META_FILENAME
        require_file(support_triplets_path)
        require_file(support_meta_path)
        support_triplet_rows = batch.read_jsonl(support_triplets_path)
        support_meta = batch.read_json(support_meta_path)
        support_anchor_object = batch.build_support_anchor_object(support_triplet_rows)
        support_anchor_object["support_claim"] = str(support_meta.get("support_claim") or "")
        exact_match = support_meta.get("exact_token_match_ratio")
        support_anchor_object["exact_token_match_ratio"] = (
            None if exact_match in (None, "") else float(exact_match)
        )
        support_anchor_object["support_anchor_triplets_path"] = batch.repo_relative_or_posix(
            support_triplets_path
        )
        support_anchor_object["support_anchor_triplets_sha256"] = batch.sha256_file(support_triplets_path)
        support_anchor_objects[execution_sample_id] = support_anchor_object

        conflict_triplets_path = sample_dir / batch.CONFLICT_ANCHOR_TRIPLETS_FILENAME
        conflict_meta_path = sample_dir / batch.CONFLICT_ANCHOR_META_FILENAME
        if conflict_triplets_path.is_file() and conflict_meta_path.is_file():
            conflict_triplet_rows = batch.read_jsonl(conflict_triplets_path)
            conflict_meta = batch.read_json(conflict_meta_path)
            conflict_anchor_object = batch.build_anchor_object(conflict_triplet_rows)
            conflict_anchor_object["wrong_claim"] = str(conflict_meta.get("wrong_claim") or "")
            conflict_match = conflict_meta.get("exact_token_match_ratio")
            conflict_anchor_object["exact_token_match_ratio"] = (
                None if conflict_match in (None, "") else float(conflict_match)
            )
            conflict_anchor_object["conflict_anchor_triplets_path"] = batch.repo_relative_or_posix(
                conflict_triplets_path
            )
            conflict_anchor_object["conflict_anchor_triplets_sha256"] = batch.sha256_file(
                conflict_triplets_path
            )
            conflict_anchor_objects[execution_sample_id] = conflict_anchor_object

    return support_anchor_objects, conflict_anchor_objects


def ensure_diagnostics(
    *,
    execution_dir: Path,
    sample_registry_rows: Sequence[Dict[str, Any]],
    extraction_rows: Sequence[Dict[str, Any]],
    diagnostics_dir: Path,
    support_anchor_objects: Dict[int, Dict[str, Any]],
    conflict_anchor_objects: Dict[int, Dict[str, Any]],
) -> None:
    bridge_per_sample_path = diagnostics_dir / batch.BRIDGE_PER_SAMPLE_FILENAME
    bridge_by_cell_path = diagnostics_dir / batch.BRIDGE_BY_CELL_FILENAME
    bridge_report_path = diagnostics_dir / batch.BRIDGE_REPORT_FILENAME
    support_bridge_per_sample_path = diagnostics_dir / batch.SUPPORT_BRIDGE_PER_SAMPLE_FILENAME
    support_bridge_by_cell_path = diagnostics_dir / batch.SUPPORT_BRIDGE_BY_CELL_FILENAME
    support_bridge_report_path = diagnostics_dir / batch.SUPPORT_BRIDGE_REPORT_FILENAME
    direct_bridge_per_sample_path = diagnostics_dir / batch.DIRECT_BRIDGE_PER_SAMPLE_FILENAME
    direct_bridge_by_target_path = diagnostics_dir / batch.DIRECT_BRIDGE_BY_TARGET_FILENAME
    direct_bridge_report_path = diagnostics_dir / batch.DIRECT_BRIDGE_REPORT_FILENAME

    gate6_dir = execution_dir / "gate6_native"
    gate6_step_rows = gate7c_consumer.load_rows(gate6_dir / gate7c_consumer.DEFAULT_STEP_INDEX)
    with np.load(gate6_dir / gate7c_consumer.DEFAULT_ARRAYS) as npz_handle:
        gate6_arrays = {
            "basis": np.asarray(npz_handle["basis"], dtype=np.float64),
            "coords_local": np.asarray(npz_handle["coords_local"], dtype=np.float64),
            "singular_values": np.asarray(npz_handle["singular_values"], dtype=np.float64),
            "rank_local": np.asarray(npz_handle["rank_local"], dtype=np.int64),
        }

    run_id = execution_dir.name

    if not bridge_per_sample_path.is_file() or not bridge_by_cell_path.is_file() or not bridge_report_path.is_file():
        bridge_transition_rows = batch.build_rotation_leakage_transition_rows(
            step_rows=gate6_step_rows,
            arrays=gate6_arrays,
        )
        bridge_per_sample_rows = batch.build_rotation_leakage_per_sample_rows(
            sample_registry_rows=sample_registry_rows,
            transition_rows=bridge_transition_rows,
        )
        bridge_by_cell_rows = batch.build_rotation_leakage_by_cell_rows(bridge_per_sample_rows)
        batch.write_csv(
            bridge_per_sample_path,
            (
                "execution_sample_id",
                "benchmark_sample_id",
                "cell_id",
                "world_id",
                "rendering_id",
                "target_id",
                "world_type",
                "answer_target_type",
                "quietness_pair_id",
                "n_transition_rows_total",
                "n_transition_rows_valid",
                "n_transition_rows_missing",
                "positive_transition_count",
                "mean_rotation_only",
                "p90_rotation_only",
                "max_rotation_only",
                "mean_leakage_only",
                "p90_leakage_only",
                "max_leakage_only",
                "mean_closure_defect",
                "p90_closure_defect",
                "max_closure_defect",
            ),
            bridge_per_sample_rows,
        )
        batch.write_csv(
            bridge_by_cell_path,
            (
                "cell_id",
                "n_samples",
                "n_transition_rows_total",
                "n_transition_rows_valid",
                "n_transition_rows_missing",
                "mean_sample_mean_rotation_only",
                "mean_sample_p90_rotation_only",
                "mean_sample_max_rotation_only",
                "mean_sample_mean_leakage_only",
                "mean_sample_p90_leakage_only",
                "mean_sample_max_leakage_only",
                "mean_sample_mean_closure_defect",
                "mean_sample_p90_closure_defect",
                "mean_sample_max_closure_defect",
            ),
            bridge_by_cell_rows,
        )
        batch.write_text(
            bridge_report_path,
            batch.build_rotation_leakage_bridge_report(
                run_id=run_id,
                per_sample_rows=bridge_per_sample_rows,
                by_cell_rows=bridge_by_cell_rows,
            ),
        )

    if (
        not support_bridge_per_sample_path.is_file()
        or not support_bridge_by_cell_path.is_file()
        or not support_bridge_report_path.is_file()
    ):
        support_bridge_transition_rows = batch.build_support_closure_transition_rows(
            step_rows=gate6_step_rows,
            arrays=gate6_arrays,
            support_anchor_objects=support_anchor_objects,
        )
        support_bridge_per_sample_rows = batch.build_support_closure_per_sample_rows(
            sample_registry_rows=sample_registry_rows,
            transition_rows=support_bridge_transition_rows,
            support_anchor_objects=support_anchor_objects,
        )
        support_bridge_by_cell_rows = batch.build_support_closure_by_cell_rows(
            support_bridge_per_sample_rows
        )
        batch.write_csv(
            support_bridge_per_sample_path,
            (
                "execution_sample_id",
                "benchmark_sample_id",
                "cell_id",
                "world_id",
                "rendering_id",
                "target_id",
                "world_type",
                "answer_target_type",
                "quietness_pair_id",
                "support_anchor_rank",
                "support_anchor_steps",
                "support_anchor_exact_token_match_ratio",
                "n_transition_rows_total",
                "n_transition_rows_anchor_valid",
                "n_transition_rows_closure_valid",
                "n_transition_rows_missing",
                "positive_transition_count",
                "mean_support_anchor_coverage",
                "p90_support_anchor_coverage",
                "max_support_anchor_coverage",
                "mean_support_reanchor_cost",
                "p90_support_reanchor_cost",
                "max_support_reanchor_cost",
                "mean_support_conditioned_closure",
                "p90_support_conditioned_closure",
                "max_support_conditioned_closure",
            ),
            support_bridge_per_sample_rows,
        )
        batch.write_csv(
            support_bridge_by_cell_path,
            (
                "cell_id",
                "n_samples",
                "n_transition_rows_total",
                "n_transition_rows_anchor_valid",
                "n_transition_rows_closure_valid",
                "n_transition_rows_missing",
                "mean_support_anchor_rank",
                "mean_support_anchor_steps",
                "mean_sample_mean_support_anchor_coverage",
                "mean_sample_p90_support_anchor_coverage",
                "mean_sample_mean_support_reanchor_cost",
                "mean_sample_p90_support_reanchor_cost",
                "mean_sample_mean_support_conditioned_closure",
                "mean_sample_p90_support_conditioned_closure",
            ),
            support_bridge_by_cell_rows,
        )
        batch.write_text(
            support_bridge_report_path,
            batch.build_support_closure_bridge_report(
                run_id=run_id,
                per_sample_rows=support_bridge_per_sample_rows,
                by_cell_rows=support_bridge_by_cell_rows,
            ),
        )

    if (
        not direct_bridge_per_sample_path.is_file()
        or not direct_bridge_by_target_path.is_file()
        or not direct_bridge_report_path.is_file()
    ):
        direct_bridge_transition_rows = batch.build_direct_contradiction_transition_rows(
            step_rows=gate6_step_rows,
            arrays=gate6_arrays,
            sample_registry_rows=sample_registry_rows,
            support_anchor_objects=support_anchor_objects,
            conflict_anchor_objects=conflict_anchor_objects,
        )
        direct_bridge_per_sample_rows = batch.build_direct_contradiction_per_sample_rows(
            sample_registry_rows=sample_registry_rows,
            transition_rows=direct_bridge_transition_rows,
            support_anchor_objects=support_anchor_objects,
            conflict_anchor_objects=conflict_anchor_objects,
        )
        direct_bridge_by_target_rows = batch.build_direct_contradiction_by_answer_target_rows(
            direct_bridge_per_sample_rows
        )
        batch.write_csv(
            direct_bridge_per_sample_path,
            (
                "execution_sample_id",
                "benchmark_sample_id",
                "cell_id",
                "world_id",
                "rendering_id",
                "target_id",
                "world_type",
                "answer_target_type",
                "support_anchor_rank",
                "support_anchor_steps",
                "support_anchor_exact_token_match_ratio",
                "conflict_anchor_rank",
                "conflict_anchor_steps",
                "conflict_anchor_exact_token_match_ratio",
                "n_transition_rows_total",
                "n_transition_rows_support_anchor_valid",
                "n_transition_rows_conflict_anchor_valid",
                "n_transition_rows_gap_valid",
                "n_transition_rows_missing",
                "positive_transition_count",
                "mean_support_anchor_coverage",
                "p90_support_anchor_coverage",
                "mean_conflict_anchor_coverage",
                "p90_conflict_anchor_coverage",
                "mean_dual_anchor_contradiction_gap",
                "p90_dual_anchor_contradiction_gap",
                "max_dual_anchor_contradiction_gap",
            ),
            direct_bridge_per_sample_rows,
        )
        batch.write_csv(
            direct_bridge_by_target_path,
            (
                "answer_target_type",
                "n_samples",
                "n_transition_rows_total",
                "n_transition_rows_gap_valid",
                "n_transition_rows_missing",
                "mean_sample_mean_support_anchor_coverage",
                "mean_sample_mean_conflict_anchor_coverage",
                "mean_sample_mean_dual_anchor_contradiction_gap",
                "mean_sample_p90_dual_anchor_contradiction_gap",
            ),
            direct_bridge_by_target_rows,
        )
        batch.write_text(
            direct_bridge_report_path,
            batch.build_direct_contradiction_bridge_report(
                run_id=run_id,
                per_sample_rows=direct_bridge_per_sample_rows,
                by_answer_target_rows=direct_bridge_by_target_rows,
            ),
        )


def finalize_candidate_execution(
    *,
    benchmark_dir: Path,
    execution_dir: Path,
    model_id: str,
    model_revision: str,
    device: str,
    topk: int,
    seed: int,
) -> int:
    repo_root = REPO_ROOT
    benchmark_dir = Path(benchmark_dir)
    execution_dir = Path(execution_dir)
    run_id = execution_dir.name

    evaluations_root = execution_dir / "evaluations"
    diagnostics_dir = execution_dir / "diagnostics"
    manifest_path = execution_dir / "manifest.json"
    sample_registry_path = execution_dir / "sample_registry.jsonl"
    quietness_pairs_path = execution_dir / "quietness_pairs.jsonl"
    extraction_results_path = execution_dir / "extraction_results.jsonl"
    summary_csv_path = execution_dir / "candidate_summary.csv"
    report_path = execution_dir / "gate8a_standing_summary.md"
    checksums_path = execution_dir / "checksums.json"
    bridge_per_sample_path = diagnostics_dir / batch.BRIDGE_PER_SAMPLE_FILENAME
    bridge_by_cell_path = diagnostics_dir / batch.BRIDGE_BY_CELL_FILENAME
    bridge_report_path = diagnostics_dir / batch.BRIDGE_REPORT_FILENAME
    support_bridge_per_sample_path = diagnostics_dir / batch.SUPPORT_BRIDGE_PER_SAMPLE_FILENAME
    support_bridge_by_cell_path = diagnostics_dir / batch.SUPPORT_BRIDGE_BY_CELL_FILENAME
    support_bridge_report_path = diagnostics_dir / batch.SUPPORT_BRIDGE_REPORT_FILENAME
    direct_bridge_per_sample_path = diagnostics_dir / batch.DIRECT_BRIDGE_PER_SAMPLE_FILENAME
    direct_bridge_by_target_path = diagnostics_dir / batch.DIRECT_BRIDGE_BY_TARGET_FILENAME
    direct_bridge_report_path = diagnostics_dir / batch.DIRECT_BRIDGE_REPORT_FILENAME

    require_file(benchmark_dir / "manifest.json")
    require_file(benchmark_dir / "benchmark_rows.jsonl")
    require_file(sample_registry_path)
    require_file(quietness_pairs_path)
    require_file(extraction_results_path)
    require_dir(execution_dir / "samples")
    require_dir(execution_dir / "gate6_native")
    require_dir(execution_dir / "gate6f")
    require_dir(execution_dir / "gate6h")
    require_dir(execution_dir / "gate7c")
    require_dir(diagnostics_dir)

    benchmark_manifest = batch.read_json(benchmark_dir / "manifest.json")
    batch.validate_benchmark_manifest(benchmark_manifest)
    sample_registry_rows = batch.read_jsonl(sample_registry_path)
    extraction_rows = batch.read_jsonl(extraction_results_path)
    diagnostics_complete = all(
        path.is_file()
        for path in (
            bridge_per_sample_path,
            bridge_by_cell_path,
            bridge_report_path,
            support_bridge_per_sample_path,
            support_bridge_by_cell_path,
            support_bridge_report_path,
            direct_bridge_per_sample_path,
            direct_bridge_by_target_path,
            direct_bridge_report_path,
        )
    )
    if not diagnostics_complete:
        support_anchor_objects, conflict_anchor_objects = load_anchor_objects(extraction_rows)
        ensure_diagnostics(
            execution_dir=execution_dir,
            sample_registry_rows=sample_registry_rows,
            extraction_rows=extraction_rows,
            diagnostics_dir=diagnostics_dir,
            support_anchor_objects=support_anchor_objects,
            conflict_anchor_objects=conflict_anchor_objects,
        )
    require_file(bridge_per_sample_path)
    require_file(bridge_by_cell_path)
    require_file(bridge_report_path)
    require_file(support_bridge_per_sample_path)
    require_file(support_bridge_by_cell_path)
    require_file(support_bridge_report_path)
    require_file(direct_bridge_per_sample_path)
    require_file(direct_bridge_by_target_path)
    require_file(direct_bridge_report_path)
    rendering_family_id = batch.resolve_execution_rendering_family_id(
        benchmark_manifest=benchmark_manifest,
        sample_registry_rows=sample_registry_rows,
    )

    for candidate in batch.FIXED_CANDIDATES:
        token_csv = execution_dir / candidate["token_csv_relpath"]
        require_file(token_csv)
        run_subprocess(
            [
                sys.executable,
                str((repo_root / "tools" / "evaluate_gate8_standing.py").resolve()),
                "--sample-registry-jsonl",
                str(sample_registry_path),
                "--token-csv",
                str(token_csv),
                "--out-dir",
                str(evaluations_root / candidate["candidate_id"]),
                "--run-id",
                f"{run_id}_{candidate['candidate_id']}",
                "--candidate-id",
                candidate["candidate_id"],
                "--metric-id",
                candidate["metric_id"],
                "--label-key",
                candidate["label_key"],
                "--label-granularity",
                candidate["label_granularity"],
                "--topk",
                "10",
            ]
        )

    candidate_summary_rows = batch.build_candidate_summary(evaluations_root)
    batch.write_csv(
        summary_csv_path,
        (
            "candidate_id",
            "label_key",
            "label_granularity",
            "metric_id",
            "direct_global_auprc",
            "direct_mean_sample_auprc",
            "direct_mean_hit_at_10",
            "direct_mean_first_hit_distance",
            "distributed_global_auprc",
            "distributed_mean_sample_auprc",
            "distributed_mean_hit_at_10",
            "distributed_mean_first_hit_distance",
            "quiet_mean_delta_max",
            "quiet_mean_delta_p90",
            "quiet_mean_iqr_normalized_delta_max",
            "quiet_mean_top10_inflation",
        ),
        candidate_summary_rows,
    )

    report = batch.build_standing_report(
        run_id=run_id,
        benchmark_manifest=benchmark_manifest,
        rendering_family_id=rendering_family_id,
        sample_registry_rows=sample_registry_rows,
        extraction_rows=extraction_rows,
        candidate_summary_rows=candidate_summary_rows,
        model_id=model_id,
        model_revision=model_revision,
    )
    batch.write_text(report_path, report)

    batch.write_json(
        manifest_path,
        {
            "run_id": run_id,
            "schema_version": batch.SCHEMA_VERSION,
            "method_id": batch.METHOD_ID,
            "benchmark_dir": batch.repo_relative_or_posix(benchmark_dir),
            "benchmark_manifest_path": batch.repo_relative_or_posix(benchmark_dir / "manifest.json"),
            "benchmark_manifest_sha256": batch.sha256_file(benchmark_dir / "manifest.json"),
            "rendering_family_id": rendering_family_id,
            "benchmark_rows_path": batch.repo_relative_or_posix(benchmark_dir / "benchmark_rows.jsonl"),
            "benchmark_rows_sha256": batch.sha256_file(benchmark_dir / "benchmark_rows.jsonl"),
            "sample_registry_path": batch.repo_relative_or_posix(sample_registry_path),
            "sample_registry_sha256": batch.sha256_file(sample_registry_path),
            "quietness_pairs_path": batch.repo_relative_or_posix(quietness_pairs_path),
            "quietness_pairs_sha256": batch.sha256_file(quietness_pairs_path),
            "extraction_results_path": batch.repo_relative_or_posix(extraction_results_path),
            "extraction_results_sha256": batch.sha256_file(extraction_results_path),
            "candidate_summary_path": batch.repo_relative_or_posix(summary_csv_path),
            "candidate_summary_sha256": batch.sha256_file(summary_csv_path),
            "diagnostic_bridge": {
                "method_id": batch.BRIDGE_METHOD_ID,
                "status": batch.BRIDGE_STATUS,
                "source_candidate_id": batch.BRIDGE_SOURCE_CANDIDATE_ID,
                "doc_path": batch.repo_relative_or_posix(repo_root / batch.BRIDGE_DOC_PATH),
                "per_sample_path": batch.repo_relative_or_posix(bridge_per_sample_path),
                "per_sample_sha256": batch.sha256_file(bridge_per_sample_path),
                "by_cell_path": batch.repo_relative_or_posix(bridge_by_cell_path),
                "by_cell_sha256": batch.sha256_file(bridge_by_cell_path),
                "report_path": batch.repo_relative_or_posix(bridge_report_path),
                "report_sha256": batch.sha256_file(bridge_report_path),
            },
            "support_closure_bridge": {
                "method_id": batch.SUPPORT_BRIDGE_METHOD_ID,
                "status": batch.SUPPORT_BRIDGE_STATUS,
                "source_candidate_id": batch.SUPPORT_BRIDGE_SOURCE_CANDIDATE_ID,
                "doc_path": batch.repo_relative_or_posix(repo_root / batch.SUPPORT_BRIDGE_DOC_PATH),
                "per_sample_path": batch.repo_relative_or_posix(support_bridge_per_sample_path),
                "per_sample_sha256": batch.sha256_file(support_bridge_per_sample_path),
                "by_cell_path": batch.repo_relative_or_posix(support_bridge_by_cell_path),
                "by_cell_sha256": batch.sha256_file(support_bridge_by_cell_path),
                "report_path": batch.repo_relative_or_posix(support_bridge_report_path),
                "report_sha256": batch.sha256_file(support_bridge_report_path),
            },
            "direct_contradiction_bridge": {
                "method_id": batch.DIRECT_BRIDGE_METHOD_ID,
                "status": batch.DIRECT_BRIDGE_STATUS,
                "source_candidate_id": batch.DIRECT_BRIDGE_SOURCE_CANDIDATE_ID,
                "doc_path": batch.repo_relative_or_posix(repo_root / batch.DIRECT_BRIDGE_DOC_PATH),
                "per_sample_path": batch.repo_relative_or_posix(direct_bridge_per_sample_path),
                "per_sample_sha256": batch.sha256_file(direct_bridge_per_sample_path),
                "by_answer_target_path": batch.repo_relative_or_posix(direct_bridge_by_target_path),
                "by_answer_target_sha256": batch.sha256_file(direct_bridge_by_target_path),
                "report_path": batch.repo_relative_or_posix(direct_bridge_report_path),
                "report_sha256": batch.sha256_file(direct_bridge_report_path),
            },
            "model_id": model_id,
            "model_revision": model_revision,
            "device": str(device),
            "topk_requested": int(topk),
            "seed": int(seed),
            "quietness_pairing_rule": batch.QUIETNESS_PAIRING_RULE,
            "candidate_granularity_status": batch.GRANULARITY_COURT_STATUS,
            "candidate_granularity_note": batch.GRANULARITY_COURT_NOTE,
            "candidate_set": batch.fixed_candidate_contract_rows(),
            "code_git_commit": batch.gate6_builder.current_git_commit(),
            "n_samples_total": len(sample_registry_rows),
        },
    )
    batch.write_json(
        checksums_path,
        {
            "manifest.json": batch.sha256_file(manifest_path),
            "sample_registry.jsonl": batch.sha256_file(sample_registry_path),
            "quietness_pairs.jsonl": batch.sha256_file(quietness_pairs_path),
            "extraction_results.jsonl": batch.sha256_file(extraction_results_path),
            "candidate_summary.csv": batch.sha256_file(summary_csv_path),
            "gate8a_standing_summary.md": batch.sha256_file(report_path),
            batch.repo_relative_or_posix(bridge_per_sample_path): batch.sha256_file(bridge_per_sample_path),
            batch.repo_relative_or_posix(bridge_by_cell_path): batch.sha256_file(bridge_by_cell_path),
            batch.repo_relative_or_posix(bridge_report_path): batch.sha256_file(bridge_report_path),
            batch.repo_relative_or_posix(support_bridge_per_sample_path): batch.sha256_file(
                support_bridge_per_sample_path
            ),
            batch.repo_relative_or_posix(support_bridge_by_cell_path): batch.sha256_file(
                support_bridge_by_cell_path
            ),
            batch.repo_relative_or_posix(support_bridge_report_path): batch.sha256_file(
                support_bridge_report_path
            ),
            batch.repo_relative_or_posix(direct_bridge_per_sample_path): batch.sha256_file(
                direct_bridge_per_sample_path
            ),
            batch.repo_relative_or_posix(direct_bridge_by_target_path): batch.sha256_file(
                direct_bridge_by_target_path
            ),
            batch.repo_relative_or_posix(direct_bridge_report_path): batch.sha256_file(
                direct_bridge_report_path
            ),
        },
    )
    return 0


def main() -> int:
    args = parse_args()
    return finalize_candidate_execution(
        benchmark_dir=Path(args.benchmark_dir),
        execution_dir=Path(args.execution_dir),
        model_id=args.model_id,
        model_revision=args.model_revision,
        device=args.device,
        topk=args.topk,
        seed=args.seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
