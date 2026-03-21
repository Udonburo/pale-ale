#!/usr/bin/env python3
"""Run a Gate9D conflict-motif coverage audit on Gate9C outputs."""

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import run_gate9a_graph_gauge_consumer as gate9a
import run_gate9b_small_cycle_holonomy_study as gate9b
import run_gate9c_missingness_topology_audit as gate9c


REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "gate9d_conflict_motif_coverage_audit_v1"
METHOD_ID = "gate9d_conflict_motif_coverage_audit_v1"
FOCUS_CYCLE_TYPE = "conflict_answer_terminal_token_cycle"

ALREADY_COVERED = "already_covered"
NOT_APPLICABLE_STRUCTURAL = "not_applicable_structural"
RECOVERABLE_CANDIDATE = "recoverable_under_frozen_law_candidate"
BLOCKED_WITHOUT_LAW_CHANGE = "blocked_without_law_change"
IMPLEMENTATION_BOUND_GAP = "implementation_bound_gap"
TAXONOMIC_GAP = "taxonomic_branch_gap"

DEFAULT_REGISTRY = "conflict_motif_coverage_registry.jsonl"
DEFAULT_SUMMARY = "conflict_motif_coverage_by_cell_answer_target.csv"
DEFAULT_CELL_SUMMARY = "conflict_motif_coverage_by_cell.csv"
DEFAULT_STATUS = "coverage_recovery_status.json"
DEFAULT_REPORT = "gate9d_conflict_coverage_read.md"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Gate9D conflict-motif coverage audit over an existing Gate9C "
            "missingness bundle without reopening the frozen graph-gauge law."
        )
    )
    parser.add_argument("--gate9c-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def sample_dir_for_execution_id(source_gate8_dir: Path, execution_sample_id: int) -> Path:
    return source_gate8_dir / "samples" / f"sample_{execution_sample_id:06d}"


def derive_source_dirs(source_gate9c_dir: Path) -> Tuple[Path, Dict[str, Any], Path, Dict[str, Any], Path, Dict[str, Any]]:
    gate9c_manifest = gate9a.read_json(source_gate9c_dir / gate9a.DEFAULT_MANIFEST)
    source_gate9b_dir = REPO_ROOT / str(gate9c_manifest["source_gate9b_dir"])
    gate9b_manifest = gate9a.read_json(source_gate9b_dir / gate9b.DEFAULT_MANIFEST)
    source_gate9a_dir = REPO_ROOT / str(gate9b_manifest["source_gate9a_dir"])
    gate9a_manifest = gate9a.read_json(source_gate9a_dir / gate9a.DEFAULT_MANIFEST)
    source_gate8_dir = REPO_ROOT / str(gate9a_manifest["source_gate8_execution_dir"])
    return source_gate9b_dir, gate9b_manifest, source_gate9a_dir, gate9a_manifest, source_gate8_dir, gate9c_manifest


def classify_recovery_path(
    cycle_outcome: str,
    absence_class: str,
    *,
    is_conflict_intended: bool,
    has_conflict_chunk_declared: bool,
    has_conflict_anchor_materialized: bool,
) -> Tuple[str, str]:
    if cycle_outcome == "none":
        return ALREADY_COVERED, "cycle_already_available"
    if absence_class == gate9c.STRUCTURAL:
        return NOT_APPLICABLE_STRUCTURAL, "cleaner_side_conflict_cycle_not_licensed"
    if absence_class == gate9c.IMPLEMENTATION_BOUND:
        return IMPLEMENTATION_BOUND_GAP, "execution_or_registry_gap"
    if absence_class == gate9c.TAXONOMIC:
        return TAXONOMIC_GAP, "answer_target_branch_specific_gap"
    if absence_class == gate9c.BUNDLE_SPECIFIC:
        if not is_conflict_intended:
            return NOT_APPLICABLE_STRUCTURAL, "non_conflict_row_cannot_recover_conflict_cycle"
        if not has_conflict_chunk_declared:
            return BLOCKED_WITHOUT_LAW_CHANGE, "no_declared_conflict_chunk_upstream"
        if not has_conflict_anchor_materialized:
            return RECOVERABLE_CANDIDATE, "declared_conflict_chunk_without_materialized_conflict_anchor"
        return RECOVERABLE_CANDIDATE, "bundle_specific_gap_with_materialized_anchor_still_missing"
    return IMPLEMENTATION_BOUND_GAP, "unclassified_gap_defaults_to_execution_audit"


def build_registry_rows(
    focus_rows: Sequence[Dict[str, Any]],
    missingness_by_cycle_id: Dict[str, Dict[str, Any]],
    source_gate8_dir: Path,
) -> List[Dict[str, Any]]:
    registry_rows: List[Dict[str, Any]] = []
    for row in focus_rows:
        execution_sample_id = int(row["execution_sample_id"])
        sample_dir = sample_dir_for_execution_id(source_gate8_dir, execution_sample_id)
        benchmark_row = gate9a.read_json(sample_dir / "benchmark_row.json")
        missingness_row = missingness_by_cycle_id.get(str(row["cycle_id"]), {})

        retrieval_conflict_chunk_ids = list(benchmark_row.get("retrieval_conflict_chunk_ids") or [])
        retrieval_support_chunk_ids = list(benchmark_row.get("retrieval_support_chunk_ids") or [])
        has_conflict_anchor_text = (sample_dir / "conflict_anchor.txt").exists()
        has_conflict_anchor_meta = (sample_dir / "conflict_anchor_meta.json").exists()
        has_conflict_anchor_triplets = (sample_dir / "conflict_anchor_triplets.ndjson").exists()
        has_conflict_anchor_materialized = (
            has_conflict_anchor_text or has_conflict_anchor_meta or has_conflict_anchor_triplets
        )

        recovery_path_status, recovery_reason = classify_recovery_path(
            str(row["cycle_outcome"]),
            str(missingness_row.get("absence_class") or ""),
            is_conflict_intended=bool(row.get("is_conflict_intended", False)),
            has_conflict_chunk_declared=bool(retrieval_conflict_chunk_ids),
            has_conflict_anchor_materialized=has_conflict_anchor_materialized,
        )

        cleaner_side_risk_status = "not_cleaner_side"
        if str(row["cell_id"]) in gate9b.CLEANER_CELL_IDS:
            cleaner_side_risk_status = (
                "pollution_risk" if recovery_path_status == RECOVERABLE_CANDIDATE else "clear"
            )

        registry_rows.append(
            {
                "cycle_id": str(row["cycle_id"]),
                "cycle_type": str(row["cycle_type"]),
                "execution_sample_id": execution_sample_id,
                "benchmark_sample_id": str(row["benchmark_sample_id"]),
                "cell_id": str(row["cell_id"]),
                "cell_bucket": str(row["cell_bucket"]),
                "world_id": str(row["world_id"]),
                "world_type": str(row["world_type"]),
                "answer_target_type": str(row["answer_target_type"]),
                "rendering_family_id": str(row["rendering_family_id"]),
                "cycle_outcome": str(row["cycle_outcome"]),
                "absence_class": str(missingness_row.get("absence_class") or ""),
                "classification_reason": str(missingness_row.get("classification_reason") or ""),
                "is_conflict_intended": bool(row.get("is_conflict_intended", False)),
                "retrieval_conflict_chunk_count": len(retrieval_conflict_chunk_ids),
                "retrieval_support_chunk_count": len(retrieval_support_chunk_ids),
                "has_conflict_chunk_declared": bool(retrieval_conflict_chunk_ids),
                "has_support_chunk_declared": bool(retrieval_support_chunk_ids),
                "has_conflict_anchor_text": has_conflict_anchor_text,
                "has_conflict_anchor_meta": has_conflict_anchor_meta,
                "has_conflict_anchor_triplets": has_conflict_anchor_triplets,
                "has_conflict_anchor_materialized": has_conflict_anchor_materialized,
                "has_answer_triplets": (sample_dir / "triplets.ndjson").exists(),
                "recovery_path_status": recovery_path_status,
                "recovery_reason": recovery_reason,
                "cleaner_side_risk_status": cleaner_side_risk_status,
            }
        )
    return registry_rows


def summarize_by_cell_answer_target(registry_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in registry_rows:
        grouped[(str(row["cell_id"]), str(row["answer_target_type"]))].append(row)

    out_rows: List[Dict[str, Any]] = []
    for cell_id, answer_target_type in sorted(grouped):
        rows = grouped[(cell_id, answer_target_type)]
        status_counter = Counter(str(row["recovery_path_status"]) for row in rows)
        n_rows = len(rows)
        n_available = sum(1 for row in rows if str(row["cycle_outcome"]) == "none")
        n_missing = n_rows - n_available
        coverage_rate = float(n_available / n_rows) if n_rows else None
        out_rows.append(
            {
                "cell_id": cell_id,
                "answer_target_type": answer_target_type,
                "n_rows": n_rows,
                "n_available": n_available,
                "n_missing": n_missing,
                "coverage_rate": coverage_rate,
                "already_covered_count": int(status_counter[ALREADY_COVERED]),
                "not_applicable_structural_count": int(status_counter[NOT_APPLICABLE_STRUCTURAL]),
                "recoverable_under_frozen_law_candidate_count": int(status_counter[RECOVERABLE_CANDIDATE]),
                "blocked_without_law_change_count": int(status_counter[BLOCKED_WITHOUT_LAW_CHANGE]),
                "implementation_bound_gap_count": int(status_counter[IMPLEMENTATION_BOUND_GAP]),
                "taxonomic_branch_gap_count": int(status_counter[TAXONOMIC_GAP]),
            }
        )
    return out_rows


def summarize_by_cell(registry_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in registry_rows:
        grouped[str(row["cell_id"])].append(row)

    out_rows: List[Dict[str, Any]] = []
    for cell_id in sorted(grouped):
        rows = grouped[cell_id]
        status_counter = Counter(str(row["recovery_path_status"]) for row in rows)
        n_rows = len(rows)
        n_available = sum(1 for row in rows if str(row["cycle_outcome"]) == "none")
        n_missing = n_rows - n_available
        coverage_rate = float(n_available / n_rows) if n_rows else None
        out_rows.append(
            {
                "cell_id": cell_id,
                "n_rows": n_rows,
                "n_available": n_available,
                "n_missing": n_missing,
                "coverage_rate": coverage_rate,
                "already_covered_count": int(status_counter[ALREADY_COVERED]),
                "not_applicable_structural_count": int(status_counter[NOT_APPLICABLE_STRUCTURAL]),
                "recoverable_under_frozen_law_candidate_count": int(status_counter[RECOVERABLE_CANDIDATE]),
                "blocked_without_law_change_count": int(status_counter[BLOCKED_WITHOUT_LAW_CHANGE]),
                "implementation_bound_gap_count": int(status_counter[IMPLEMENTATION_BOUND_GAP]),
                "taxonomic_branch_gap_count": int(status_counter[TAXONOMIC_GAP]),
            }
        )
    return out_rows


def build_status_payload(registry_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    candidate_rows = [row for row in registry_rows if str(row["recovery_path_status"]) == RECOVERABLE_CANDIDATE]
    cleaner_side_pollution_rows = [
        row for row in registry_rows if str(row["cleaner_side_risk_status"]) == "pollution_risk"
    ]
    implementation_rows = [
        row for row in registry_rows if str(row["recovery_path_status"]) == IMPLEMENTATION_BOUND_GAP
    ]
    law_change_rows = [
        row for row in registry_rows if str(row["recovery_path_status"]) == BLOCKED_WITHOUT_LAW_CHANGE
    ]
    not_recovered_rows = [
        row for row in registry_rows
        if str(row["cell_id"]) == "distributed_incompatibility" and str(row["cycle_outcome"]) != "none"
    ]
    status_counter = Counter(str(row["recovery_path_status"]) for row in registry_rows)
    return {
        "focus_cycle_type": FOCUS_CYCLE_TYPE,
        "coverage_recovery_status": "not_yet_recovered" if not_recovered_rows else "recovered",
        "frozen_law_recovery_candidate_status": "candidate_present" if candidate_rows else "denied",
        "cleaner_side_pollution_status": "triggered" if cleaner_side_pollution_rows else "clear",
        "implementation_bound_gap_status": "triggered" if implementation_rows else "clear",
        "law_change_required_status": "triggered" if law_change_rows else "clear",
        "recovery_path_counts": dict(status_counter),
        "candidate_rows": [
            {
                "benchmark_sample_id": str(row["benchmark_sample_id"]),
                "cell_id": str(row["cell_id"]),
                "answer_target_type": str(row["answer_target_type"]),
                "recovery_reason": str(row["recovery_reason"]),
            }
            for row in candidate_rows
        ],
    }


def build_report(
    run_id: str,
    source_gate9c_manifest: Dict[str, Any],
    source_gate9b_manifest: Dict[str, Any],
    cell_summary_rows: Sequence[Dict[str, Any]],
    registry_rows: Sequence[Dict[str, Any]],
    status_payload: Dict[str, Any],
) -> str:
    lines = [
        "# Gate9D Conflict Motif Coverage Read",
        "",
        f"run_id: {run_id}",
        f"source_gate9c_run_id: {source_gate9c_manifest.get('run_id', '')}",
        f"source_gate9c_code_git_commit: {source_gate9c_manifest.get('code_git_commit', '')}",
        f"source_gate9b_run_id: {source_gate9b_manifest.get('run_id', '')}",
        f"source_gate9b_code_git_commit: {source_gate9b_manifest.get('code_git_commit', '')}",
        "",
        "## Discipline",
        "",
        "- focus motif only: `conflict_answer_terminal_token_cycle`",
        "- operator remains closed",
        "- no anchor redesign is introduced",
        "- the question is recovery candidacy under the frozen law, not recovery by new machinery",
        "",
        "## Coverage By Cell",
        "",
        "| cell_id | n_rows | n_available | n_missing | coverage_rate | already_covered | structural | recoverable_candidate | law_change_blocked | implementation_bound | taxonomic |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cell_summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["cell_id"]),
                    str(row["n_rows"]),
                    str(row["n_available"]),
                    str(row["n_missing"]),
                    "" if row["coverage_rate"] in (None, "") else f"{float(row['coverage_rate']):.6f}",
                    str(row["already_covered_count"]),
                    str(row["not_applicable_structural_count"]),
                    str(row["recoverable_under_frozen_law_candidate_count"]),
                    str(row["blocked_without_law_change_count"]),
                    str(row["implementation_bound_gap_count"]),
                    str(row["taxonomic_branch_gap_count"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Recovery Candidates",
            "",
        ]
    )
    candidate_rows = [row for row in registry_rows if str(row["recovery_path_status"]) == RECOVERABLE_CANDIDATE]
    if candidate_rows:
        for row in candidate_rows:
            lines.append(
                "- "
                + f"{row['cell_id']} / {row['answer_target_type']} / {row['benchmark_sample_id']}: "
                + f"{row['recovery_reason']}"
            )
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Status",
            "",
            f"- coverage_recovery_status: `{status_payload['coverage_recovery_status']}`",
            f"- frozen_law_recovery_candidate_status: `{status_payload['frozen_law_recovery_candidate_status']}`",
            f"- cleaner_side_pollution_status: `{status_payload['cleaner_side_pollution_status']}`",
            f"- implementation_bound_gap_status: `{status_payload['implementation_bound_gap_status']}`",
            f"- law_change_required_status: `{status_payload['law_change_required_status']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()
    source_gate9c_dir = Path(args.gate9c_dir)
    out_dir = Path(args.out_dir)
    run_id = out_dir.name

    source_gate9b_dir, source_gate9b_manifest, _source_gate9a_dir, source_gate9a_manifest, source_gate8_dir, source_gate9c_manifest = derive_source_dirs(
        source_gate9c_dir
    )
    cycle_focus_rows = gate9a.read_jsonl(source_gate9b_dir / gate9b.DEFAULT_CYCLE_FOCUS)
    missingness_rows = gate9a.read_jsonl(source_gate9c_dir / gate9c.DEFAULT_MISSINGNESS_REGISTRY)

    focus_rows = [
        row for row in cycle_focus_rows if str(row["cycle_type"]) == FOCUS_CYCLE_TYPE
    ]
    missingness_by_cycle_id = {
        str(row["cycle_id"]): row for row in missingness_rows
    }

    registry_rows = build_registry_rows(focus_rows, missingness_by_cycle_id, source_gate8_dir)
    summary_rows = summarize_by_cell_answer_target(registry_rows)
    cell_summary_rows = summarize_by_cell(registry_rows)
    status_payload = build_status_payload(registry_rows)

    registry_path = out_dir / DEFAULT_REGISTRY
    summary_path = out_dir / DEFAULT_SUMMARY
    cell_summary_path = out_dir / DEFAULT_CELL_SUMMARY
    status_path = out_dir / DEFAULT_STATUS
    report_path = out_dir / DEFAULT_REPORT
    manifest_path = out_dir / DEFAULT_MANIFEST
    checksums_path = out_dir / DEFAULT_CHECKSUMS

    gate9a.write_jsonl(registry_path, registry_rows)
    gate9a.write_csv(
        summary_path,
        (
            "cell_id",
            "answer_target_type",
            "n_rows",
            "n_available",
            "n_missing",
            "coverage_rate",
            "already_covered_count",
            "not_applicable_structural_count",
            "recoverable_under_frozen_law_candidate_count",
            "blocked_without_law_change_count",
            "implementation_bound_gap_count",
            "taxonomic_branch_gap_count",
        ),
        summary_rows,
    )
    gate9a.write_csv(
        cell_summary_path,
        (
            "cell_id",
            "n_rows",
            "n_available",
            "n_missing",
            "coverage_rate",
            "already_covered_count",
            "not_applicable_structural_count",
            "recoverable_under_frozen_law_candidate_count",
            "blocked_without_law_change_count",
            "implementation_bound_gap_count",
            "taxonomic_branch_gap_count",
        ),
        cell_summary_rows,
    )
    gate9a.write_json(status_path, status_payload)
    gate9a.write_text(
        report_path,
        build_report(
            run_id=run_id,
            source_gate9c_manifest=source_gate9c_manifest,
            source_gate9b_manifest=source_gate9b_manifest,
            cell_summary_rows=cell_summary_rows,
            registry_rows=registry_rows,
            status_payload=status_payload,
        ),
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "run_id": run_id,
        "code_git_commit": gate9a.current_git_commit(),
        "source_gate9c_dir": gate9a.repo_relative_or_posix(source_gate9c_dir),
        "source_gate9c_run_id": str(source_gate9c_manifest.get("run_id") or ""),
        "source_gate9c_code_git_commit": str(source_gate9c_manifest.get("code_git_commit") or ""),
        "source_gate9b_run_id": str(source_gate9b_manifest.get("run_id") or ""),
        "source_gate9a_run_id": str(source_gate9a_manifest.get("run_id") or ""),
        "source_gate8_run_id": str(source_gate9a_manifest.get("source_gate8_run_id") or ""),
        "focus_cycle_type": FOCUS_CYCLE_TYPE,
        "paths": {
            DEFAULT_REGISTRY: gate9a.repo_relative_or_posix(registry_path),
            DEFAULT_SUMMARY: gate9a.repo_relative_or_posix(summary_path),
            DEFAULT_CELL_SUMMARY: gate9a.repo_relative_or_posix(cell_summary_path),
            DEFAULT_STATUS: gate9a.repo_relative_or_posix(status_path),
            DEFAULT_REPORT: gate9a.repo_relative_or_posix(report_path),
        },
    }
    gate9a.write_json(manifest_path, manifest)
    gate9a.write_json(
        checksums_path,
        {
            DEFAULT_MANIFEST: sha256_file(manifest_path),
            DEFAULT_REGISTRY: sha256_file(registry_path),
            DEFAULT_SUMMARY: sha256_file(summary_path),
            DEFAULT_CELL_SUMMARY: sha256_file(cell_summary_path),
            DEFAULT_STATUS: sha256_file(status_path),
            DEFAULT_REPORT: sha256_file(report_path),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
