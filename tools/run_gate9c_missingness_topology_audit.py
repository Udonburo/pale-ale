#!/usr/bin/env python3
"""Run a Gate9C missingness-topology admission audit on Gate9B outputs."""

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import run_gate9a_graph_gauge_consumer as gate9a
import run_gate9b_small_cycle_holonomy_study as gate9b


REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "gate9c_missingness_topology_audit_v1"
METHOD_ID = "gate9c_missingness_topology_audit_v1"

STRUCTURAL = "structural"
TAXONOMIC = "taxonomic"
BUNDLE_SPECIFIC = "bundle_specific"
IMPLEMENTATION_BOUND = "implementation_bound"

MISSING_OUTCOMES = (
    "missing_support_anchor",
    "missing_conflict_anchor",
    "missing_cycle_edge",
    "missing_terminal_token",
)

DEFAULT_MISSINGNESS_REGISTRY = "missingness_registry.jsonl"
DEFAULT_COVERAGE_BY_TARGET = "missingness_by_cell_motif_answer_target.csv"
DEFAULT_COVERAGE_BY_CELL = "usable_motif_coverage_by_cell.csv"
DEFAULT_CLASS_SUMMARY = "missingness_class_summary.csv"
DEFAULT_ADMISSION_STATUS = "admission_slice_status.json"
DEFAULT_REPORT = "gate9c_missingness_read.md"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Gate9C admission audit over missingness topology and usable motif "
            "coverage from an existing Gate9B execution bundle."
        )
    )
    parser.add_argument("--gate9b-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def is_missing_outcome(cycle_outcome: str) -> bool:
    return cycle_outcome != "none"


def is_implementation_bound_outcome(cycle_outcome: str) -> bool:
    return cycle_outcome in {"missing_cycle_edge", "missing_terminal_token", "missing_cycle_row"} or cycle_outcome.startswith(
        "edge_failure:"
    ) or cycle_outcome in {"missing_coord_isometry", "invalid_root_rank"}


def build_answer_target_availability(
    focus_rows: Sequence[Dict[str, Any]],
) -> Dict[Tuple[str, str, str], Dict[str, int]]:
    grouped: Dict[Tuple[str, str, str], Dict[str, int]] = defaultdict(lambda: {"n_rows": 0, "n_available": 0, "n_missing": 0})
    for row in focus_rows:
        key = (str(row["cell_id"]), str(row["cycle_type"]), str(row["answer_target_type"]))
        grouped[key]["n_rows"] += 1
        if str(row["cycle_outcome"]) == "none":
            grouped[key]["n_available"] += 1
        else:
            grouped[key]["n_missing"] += 1
    return grouped


def classify_missingness_row(
    row: Dict[str, Any],
    availability: Dict[Tuple[str, str, str], Dict[str, int]],
) -> Tuple[str, str]:
    cycle_outcome = str(row["cycle_outcome"])
    cycle_type = str(row["cycle_type"])
    cell_id = str(row["cell_id"])
    answer_target_type = str(row["answer_target_type"])

    if is_implementation_bound_outcome(cycle_outcome):
        return IMPLEMENTATION_BOUND, "execution_or_registry_gap"

    if cycle_outcome == "missing_conflict_anchor" and not bool(row.get("is_conflict_intended", False)):
        return STRUCTURAL, "conflict_motif_not_licensed_on_non_conflict_cell"

    current_key = (cell_id, cycle_type, answer_target_type)
    current_stats = availability.get(current_key, {"n_available": 0})
    other_has_available = any(
        stats["n_available"] > 0
        for key, stats in availability.items()
        if key[0] == cell_id and key[1] == cycle_type and key[2] != answer_target_type
    )
    if current_stats["n_available"] == 0 and other_has_available:
        return TAXONOMIC, "availability_varies_by_answer_target"

    if cycle_outcome in MISSING_OUTCOMES:
        return BUNDLE_SPECIFIC, "motif_allowed_but_uninstantiated_in_current_bundle"

    return IMPLEMENTATION_BOUND, "non_success_outcome_not_in_missing_audit_core"


def build_missingness_rows(focus_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    availability = build_answer_target_availability(focus_rows)
    missingness_rows: List[Dict[str, Any]] = []
    for row in focus_rows:
        cycle_outcome = str(row["cycle_outcome"])
        if not is_missing_outcome(cycle_outcome):
            continue
        absence_class, classification_reason = classify_missingness_row(row, availability)
        missingness_rows.append(
            {
                "cycle_id": str(row["cycle_id"]),
                "cycle_type": str(row["cycle_type"]),
                "execution_sample_id": int(row["execution_sample_id"]),
                "benchmark_sample_id": str(row["benchmark_sample_id"]),
                "cell_id": str(row["cell_id"]),
                "cell_bucket": str(row["cell_bucket"]),
                "world_id": str(row["world_id"]),
                "world_type": str(row["world_type"]),
                "answer_target_type": str(row["answer_target_type"]),
                "quietness_pair_id": str(row.get("quietness_pair_id") or ""),
                "rendering_family_id": str(row["rendering_family_id"]),
                "cycle_outcome": cycle_outcome,
                "absence_class": absence_class,
                "classification_reason": classification_reason,
                "is_conflict_intended": bool(row.get("is_conflict_intended", False)),
                "is_surface_noise_only": bool(row.get("is_surface_noise_only", False)),
            }
        )
    return missingness_rows


def summarize_coverage_by_target(
    focus_rows: Sequence[Dict[str, Any]],
    missingness_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    grouped_rows: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in focus_rows:
        grouped_rows[(str(row["cell_id"]), str(row["cycle_type"]), str(row["answer_target_type"]))].append(row)

    class_counter_by_key: Dict[Tuple[str, str, str], Counter] = defaultdict(Counter)
    outcome_counter_by_key: Dict[Tuple[str, str, str], Counter] = defaultdict(Counter)
    for row in missingness_rows:
        key = (str(row["cell_id"]), str(row["cycle_type"]), str(row["answer_target_type"]))
        class_counter_by_key[key][str(row["absence_class"])] += 1
        outcome_counter_by_key[key][str(row["cycle_outcome"])] += 1

    out_rows: List[Dict[str, Any]] = []
    for key in sorted(grouped_rows):
        cell_id, cycle_type, answer_target_type = key
        rows = grouped_rows[key]
        n_rows = len(rows)
        n_available = sum(1 for row in rows if str(row["cycle_outcome"]) == "none")
        n_missing = n_rows - n_available
        outcome_counts = outcome_counter_by_key.get(key, Counter())
        dominant_missing_outcome = ""
        if outcome_counts:
            dominant_missing_outcome = sorted(outcome_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        class_counts = class_counter_by_key.get(key, Counter())
        out_rows.append(
            {
                "cell_id": cell_id,
                "cycle_type": cycle_type,
                "answer_target_type": answer_target_type,
                "n_rows": n_rows,
                "n_available": n_available,
                "n_missing": n_missing,
                "coverage_rate": float(n_available / n_rows) if n_rows else None,
                "dominant_missing_outcome": dominant_missing_outcome,
                "structural_count": int(class_counts[STRUCTURAL]),
                "taxonomic_count": int(class_counts[TAXONOMIC]),
                "bundle_specific_count": int(class_counts[BUNDLE_SPECIFIC]),
                "implementation_bound_count": int(class_counts[IMPLEMENTATION_BOUND]),
            }
        )
    return out_rows


def summarize_coverage_by_cell(
    coverage_by_target_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in coverage_by_target_rows:
        grouped[(str(row["cell_id"]), str(row["cycle_type"]))].append(row)

    out_rows: List[Dict[str, Any]] = []
    for cell_id, cycle_type in sorted(grouped):
        rows = grouped[(cell_id, cycle_type)]
        n_rows = sum(int(row["n_rows"]) for row in rows)
        n_available = sum(int(row["n_available"]) for row in rows)
        n_missing = sum(int(row["n_missing"]) for row in rows)
        structural_count = sum(int(row["structural_count"]) for row in rows)
        taxonomic_count = sum(int(row["taxonomic_count"]) for row in rows)
        bundle_specific_count = sum(int(row["bundle_specific_count"]) for row in rows)
        implementation_bound_count = sum(int(row["implementation_bound_count"]) for row in rows)
        coverage_rate = float(n_available / n_rows) if n_rows else None
        usable_status = "usable" if n_rows and n_available > 0 and n_available > n_missing else "not_yet_usable"
        out_rows.append(
            {
                "cell_id": cell_id,
                "cycle_type": cycle_type,
                "n_rows": n_rows,
                "n_available": n_available,
                "n_missing": n_missing,
                "coverage_rate": coverage_rate,
                "usable_status": usable_status,
                "structural_count": structural_count,
                "taxonomic_count": taxonomic_count,
                "bundle_specific_count": bundle_specific_count,
                "implementation_bound_count": implementation_bound_count,
            }
        )
    return out_rows


def summarize_class_counts(missingness_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], int] = defaultdict(int)
    for row in missingness_rows:
        grouped[(str(row["cycle_type"]), str(row["cycle_outcome"]), str(row["absence_class"]))] += 1
    out_rows: List[Dict[str, Any]] = []
    for cycle_type, cycle_outcome, absence_class in sorted(grouped):
        out_rows.append(
            {
                "cycle_type": cycle_type,
                "cycle_outcome": cycle_outcome,
                "absence_class": absence_class,
                "n_rows": grouped[(cycle_type, cycle_outcome, absence_class)],
            }
        )
    return out_rows


def build_admission_slice(
    coverage_by_cell_rows: Sequence[Dict[str, Any]],
    missingness_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    conflict_rows = [
        row for row in coverage_by_cell_rows
        if str(row["cell_id"]) in gate9b.CONFLICT_CELL_IDS
    ]
    unusable_conflict_rows = [
        row for row in conflict_rows
        if str(row["usable_status"]) != "usable"
    ]
    usable_motif_coverage_status = "denied" if unusable_conflict_rows else "provisionally_clear"

    unclassified_rows = [row for row in missingness_rows if not str(row.get("absence_class") or "")]
    missingness_topology_accounted_status = "clear" if not unclassified_rows else "denied"

    class_counter = Counter(str(row["absence_class"]) for row in missingness_rows)
    return {
        "usable_motif_coverage_status": usable_motif_coverage_status,
        "missingness_topology_accounted_status": missingness_topology_accounted_status,
        "operator_admission_status": "denied",
        "unusable_conflict_rows": [
            {
                "cell_id": str(row["cell_id"]),
                "cycle_type": str(row["cycle_type"]),
                "coverage_rate": row["coverage_rate"],
                "usable_status": str(row["usable_status"]),
            }
            for row in unusable_conflict_rows
        ],
        "absence_class_counts": dict(class_counter),
    }


def build_report(
    run_id: str,
    source_gate9b_manifest: Dict[str, Any],
    coverage_by_cell_rows: Sequence[Dict[str, Any]],
    class_summary_rows: Sequence[Dict[str, Any]],
    admission_slice: Dict[str, Any],
) -> str:
    lines = [
        "# Gate9C Missingness Topology Read",
        "",
        f"run_id: {run_id}",
        f"source_gate9b_run_id: {source_gate9b_manifest.get('run_id', '')}",
        f"source_gate9b_code_git_commit: {source_gate9b_manifest.get('code_git_commit', '')}",
        "",
        "## Discipline",
        "",
        "- operator still unopened",
        "- admission audit only",
        "- missingness and motif coverage are first-class",
        "- no smoothing, spectral, or field layer is introduced",
        "",
        "## Usable Motif Coverage By Cell",
        "",
        "| cell_id | cycle_type | n_rows | n_available | n_missing | coverage_rate | usable_status | structural | taxonomic | bundle_specific | implementation_bound |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in coverage_by_cell_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["cell_id"]),
                    str(row["cycle_type"]),
                    str(row["n_rows"]),
                    str(row["n_available"]),
                    str(row["n_missing"]),
                    "" if row["coverage_rate"] in (None, "") else f"{float(row['coverage_rate']):.6f}",
                    str(row["usable_status"]),
                    str(row["structural_count"]),
                    str(row["taxonomic_count"]),
                    str(row["bundle_specific_count"]),
                    str(row["implementation_bound_count"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Missingness Class Summary",
            "",
            "| cycle_type | cycle_outcome | absence_class | n_rows |",
            "|---|---|---|---:|",
        ]
    )
    for row in class_summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["cycle_type"]),
                    str(row["cycle_outcome"]),
                    str(row["absence_class"]),
                    str(row["n_rows"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Admission Slice",
            "",
            f"- usable_motif_coverage_status: `{admission_slice['usable_motif_coverage_status']}`",
            f"- missingness_topology_accounted_status: `{admission_slice['missingness_topology_accounted_status']}`",
            f"- operator_admission_status: `{admission_slice['operator_admission_status']}`",
        ]
    )
    if admission_slice["unusable_conflict_rows"]:
        lines.extend(["", "### Unusable Conflict Rows", ""])
        for row in admission_slice["unusable_conflict_rows"]:
            coverage = row["coverage_rate"]
            coverage_text = "" if coverage in (None, "") else f"{float(coverage):.6f}"
            lines.append(
                f"- {row['cell_id']} / {row['cycle_type']}: usable_status={row['usable_status']}, coverage_rate={coverage_text}"
            )
    return "\n".join(lines) + "\n"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()
    source_dir = Path(args.gate9b_dir)
    out_dir = Path(args.out_dir)
    run_id = out_dir.name

    source_manifest = gate9a.read_json(source_dir / gate9b.DEFAULT_MANIFEST)
    cycle_focus_rows = gate9a.read_jsonl(source_dir / gate9b.DEFAULT_CYCLE_FOCUS)

    missingness_rows = build_missingness_rows(cycle_focus_rows)
    coverage_by_target_rows = summarize_coverage_by_target(cycle_focus_rows, missingness_rows)
    coverage_by_cell_rows = summarize_coverage_by_cell(coverage_by_target_rows)
    class_summary_rows = summarize_class_counts(missingness_rows)
    admission_slice = build_admission_slice(coverage_by_cell_rows, missingness_rows)

    missingness_registry_path = out_dir / DEFAULT_MISSINGNESS_REGISTRY
    coverage_by_target_path = out_dir / DEFAULT_COVERAGE_BY_TARGET
    coverage_by_cell_path = out_dir / DEFAULT_COVERAGE_BY_CELL
    class_summary_path = out_dir / DEFAULT_CLASS_SUMMARY
    admission_status_path = out_dir / DEFAULT_ADMISSION_STATUS
    report_path = out_dir / DEFAULT_REPORT
    manifest_path = out_dir / DEFAULT_MANIFEST
    checksums_path = out_dir / DEFAULT_CHECKSUMS

    gate9a.write_jsonl(missingness_registry_path, missingness_rows)
    gate9a.write_csv(
        coverage_by_target_path,
        (
            "cell_id",
            "cycle_type",
            "answer_target_type",
            "n_rows",
            "n_available",
            "n_missing",
            "coverage_rate",
            "dominant_missing_outcome",
            "structural_count",
            "taxonomic_count",
            "bundle_specific_count",
            "implementation_bound_count",
        ),
        coverage_by_target_rows,
    )
    gate9a.write_csv(
        coverage_by_cell_path,
        (
            "cell_id",
            "cycle_type",
            "n_rows",
            "n_available",
            "n_missing",
            "coverage_rate",
            "usable_status",
            "structural_count",
            "taxonomic_count",
            "bundle_specific_count",
            "implementation_bound_count",
        ),
        coverage_by_cell_rows,
    )
    gate9a.write_csv(
        class_summary_path,
        ("cycle_type", "cycle_outcome", "absence_class", "n_rows"),
        class_summary_rows,
    )
    gate9a.write_json(admission_status_path, admission_slice)
    gate9a.write_text(
        report_path,
        build_report(
            run_id=run_id,
            source_gate9b_manifest=source_manifest,
            coverage_by_cell_rows=coverage_by_cell_rows,
            class_summary_rows=class_summary_rows,
            admission_slice=admission_slice,
        ),
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "run_id": run_id,
        "code_git_commit": gate9a.current_git_commit(),
        "source_gate9b_dir": gate9a.repo_relative_or_posix(source_dir),
        "source_gate9b_run_id": str(source_manifest.get("run_id") or ""),
        "source_gate9b_code_git_commit": str(source_manifest.get("code_git_commit") or ""),
        "source_gate9a_run_id": str(source_manifest.get("source_gate9a_run_id") or ""),
        "source_gate8_run_id": str(source_manifest.get("source_gate8_run_id") or ""),
        "missing_outcomes": list(MISSING_OUTCOMES),
        "absence_classes": [STRUCTURAL, TAXONOMIC, BUNDLE_SPECIFIC, IMPLEMENTATION_BOUND],
        "paths": {
            DEFAULT_MISSINGNESS_REGISTRY: gate9a.repo_relative_or_posix(missingness_registry_path),
            DEFAULT_COVERAGE_BY_TARGET: gate9a.repo_relative_or_posix(coverage_by_target_path),
            DEFAULT_COVERAGE_BY_CELL: gate9a.repo_relative_or_posix(coverage_by_cell_path),
            DEFAULT_CLASS_SUMMARY: gate9a.repo_relative_or_posix(class_summary_path),
            DEFAULT_ADMISSION_STATUS: gate9a.repo_relative_or_posix(admission_status_path),
            DEFAULT_REPORT: gate9a.repo_relative_or_posix(report_path),
        },
    }
    gate9a.write_json(manifest_path, manifest)
    gate9a.write_json(
        checksums_path,
        {
            DEFAULT_MANIFEST: sha256_file(manifest_path),
            DEFAULT_MISSINGNESS_REGISTRY: sha256_file(missingness_registry_path),
            DEFAULT_COVERAGE_BY_TARGET: sha256_file(coverage_by_target_path),
            DEFAULT_COVERAGE_BY_CELL: sha256_file(coverage_by_cell_path),
            DEFAULT_CLASS_SUMMARY: sha256_file(class_summary_path),
            DEFAULT_ADMISSION_STATUS: sha256_file(admission_status_path),
            DEFAULT_REPORT: sha256_file(report_path),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
