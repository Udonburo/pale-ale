#!/usr/bin/env python3
"""Run a Gate10C second trusted-tree settlement comparison on Gate10B outputs."""

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import run_gate10b_trusted_tree_settlement_comparison as gate10b
import run_gate9a_graph_gauge_consumer as gate9a


REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "gate10c_second_settlement_comparison_v1"
METHOD_ID = "gate10c_second_settlement_comparison_v1"

DEFAULT_REGISTRY = "trusted_tree_second_settlement_comparison_registry.jsonl"
DEFAULT_POLICY_COMPARE = "trusted_tree_second_settlement_comparison_policy_compare.csv"
DEFAULT_STATUS = "trusted_tree_second_settlement_comparison_status.json"
DEFAULT_REPORT = "gate10c_trusted_tree_second_settlement_comparison_read.md"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"

BASELINE_LANE = gate10b.BASELINE_LANE
SOURCE_CANDIDATE_LANE = gate10b.CANDIDATE_LANE
SECOND_CANDIDATE_LANE = "distributed_incompatibility"
SECOND_CANDIDATE_CELL_ID = "distributed_incompatibility"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Gate10C second trusted-tree settlement comparison for the "
            "declaratively extracted distributed_incompatibility sublane against "
            "the preserved Gate9Q baseline."
        )
    )
    parser.add_argument("--gate10b-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def lane_rows(
    registry_rows: Sequence[Dict[str, Any]], lane_name: str
) -> List[Dict[str, Any]]:
    return [row for row in registry_rows if str(row["comparison_lane"]) == lane_name]


def lane_joined_values(rows: Sequence[Dict[str, Any]], key: str) -> str:
    values = sorted({str(row[key]) for row in rows})
    return "|".join(values)


def is_second_candidate_row(row: Dict[str, Any]) -> bool:
    return (
        str(row["broader_candidate_class"]) == SOURCE_CANDIDATE_LANE
        and str(row["cell_class"]) == "conflict"
        and str(row["cell_id"]) == SECOND_CANDIDATE_CELL_ID
    )


def build_comparison_registry(
    source_registry_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    registry_rows: List[Dict[str, Any]] = []
    for row in source_registry_rows:
        if str(row["broader_candidate_class"]) == BASELINE_LANE:
            comparison_lane = BASELINE_LANE
            second_candidate_declaration = "not_applicable"
        elif is_second_candidate_row(row):
            comparison_lane = SECOND_CANDIDATE_LANE
            second_candidate_declaration = "declaratively_extracted"
        else:
            continue

        registry_rows.append(
            {
                "edge_id": str(row["edge_id"]),
                "execution_sample_id": int(row["execution_sample_id"]),
                "benchmark_sample_id": str(row["benchmark_sample_id"]),
                "cell_id": str(row["cell_id"]),
                "cell_class": str(row["cell_class"]),
                "world_id": str(row["world_id"]),
                "world_type": str(row["world_type"]),
                "answer_target_type": str(row["answer_target_type"]),
                "edge_transport_defect": row.get("edge_transport_defect"),
                "historical_role": str(row["historical_role"]),
                "forward_basis_role": str(row["forward_basis_role"]),
                "role_changed_by_adoption": bool(row["role_changed_by_adoption"]),
                "broader_candidate_class": str(row["broader_candidate_class"]),
                "comparison_lane": comparison_lane,
                "second_candidate_declaration": second_candidate_declaration,
                "forward_basis_adoption_preserved": bool(
                    row["forward_basis_adoption_preserved"]
                ),
                "requires_retroactive_reinterpretation": bool(
                    row["requires_retroactive_reinterpretation"]
                ),
                "implies_operator_admission_open": bool(
                    row["implies_operator_admission_open"]
                ),
                "implies_broader_tree_settlement": bool(
                    row["implies_broader_tree_settlement"]
                ),
                "widens_doctrine": bool(row["widens_doctrine"]),
            }
        )
    return registry_rows


def build_policy_compare(
    registry_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in registry_rows:
        grouped[str(row["comparison_lane"])].append(row)

    out_rows: List[Dict[str, Any]] = []
    for lane_name in (BASELINE_LANE, SECOND_CANDIDATE_LANE):
        rows = grouped.get(lane_name, [])
        defects = [
            float(row["edge_transport_defect"])
            for row in rows
            if row["edge_transport_defect"] not in (None, "")
        ]
        out_rows.append(
            {
                "comparison_lane": lane_name,
                "source_broader_candidate_class": lane_joined_values(
                    rows, "broader_candidate_class"
                ),
                "cell_class": lane_joined_values(rows, "cell_class"),
                "cell_id": lane_joined_values(rows, "cell_id"),
                "n_edges": len(rows),
                "mean_edge_transport_defect": mean_or_none(defects),
                "n_role_changed_by_adoption": sum(
                    1 for row in rows if row["role_changed_by_adoption"]
                ),
                "n_forward_basis_preserved": sum(
                    1 for row in rows if row["forward_basis_adoption_preserved"]
                ),
            }
        )
    return out_rows


def build_status_payload(
    registry_rows: Sequence[Dict[str, Any]],
    source_gate10b_status: Dict[str, Any],
) -> Dict[str, Any]:
    baseline_rows = lane_rows(registry_rows, BASELINE_LANE)
    candidate_rows = lane_rows(registry_rows, SECOND_CANDIDATE_LANE)
    comparison_slice_complete = bool(baseline_rows) and bool(candidate_rows)

    source_gate10b_preserved = (
        str(source_gate10b_status.get("forward_basis_baseline_preservation_status", ""))
        == "clear"
        and str(source_gate10b_status.get("non_retroactive_memory_preservation_status", ""))
        == "clear"
        and str(source_gate10b_status.get("comparison_outcome_status", ""))
        == "settled"
        and str(source_gate10b_status.get("operator_admission_still_denied_status", ""))
        == "confirmed"
        and str(source_gate10b_status.get("broader_tree_settlement_non_promotion_status", ""))
        == "clear"
    )

    forward_basis_baseline_preservation_status = (
        "clear"
        if source_gate10b_preserved
        and baseline_rows
        and all(
            bool(row["forward_basis_adoption_preserved"])
            and bool(row["role_changed_by_adoption"])
            and str(row["cell_class"]) == "cleaner"
            and str(row["forward_basis_role"]) == "closure_return_leg_auxiliary"
            for row in baseline_rows
        )
        else "denied"
    )

    gate10b_slice_non_retroactive_preservation_status = (
        "clear"
        if source_gate10b_preserved
        and all(
            str(row["comparison_lane"]) != SECOND_CANDIDATE_LANE
            or str(row["broader_candidate_class"]) == SOURCE_CANDIDATE_LANE
            for row in registry_rows
        )
        else "denied"
    )

    second_candidate_declaration_status = (
        "clear"
        if candidate_rows
        and all(
            str(row["second_candidate_declaration"]) == "declaratively_extracted"
            and str(row["cell_id"]) == SECOND_CANDIDATE_CELL_ID
            and str(row["cell_class"]) == "conflict"
            and str(row["broader_candidate_class"]) == SOURCE_CANDIDATE_LANE
            for row in candidate_rows
        )
        else "denied"
    )

    conflict_side_bridge_preservation_status = (
        "clear"
        if comparison_slice_complete
        and all(
            not bool(row["role_changed_by_adoption"])
            and str(row["cell_class"]) == "conflict"
            and str(row["forward_basis_role"]) == str(row["historical_role"])
            and row["edge_transport_defect"] not in (None, "")
            for row in candidate_rows
        )
        else "denied"
    )

    any_retroactive = any(
        bool(row["requires_retroactive_reinterpretation"]) for row in registry_rows
    )
    non_retroactive_memory_preservation_status = (
        "clear"
        if str(source_gate10b_status.get("non_retroactive_memory_preservation_status", ""))
        == "clear"
        and not any_retroactive
        else "denied"
    )

    any_operator_open = any(
        bool(row["implies_operator_admission_open"]) for row in registry_rows
    )
    operator_adjacent_rescue_pressure_status = (
        "clear"
        if str(source_gate10b_status.get("operator_adjacent_rescue_pressure_status", ""))
        == "clear"
        and not any_operator_open
        else "triggered"
    )

    any_semantics_broadening = any(bool(row["widens_doctrine"]) for row in registry_rows)
    trusted_tree_semantics_broadening_pressure_status = (
        "clear"
        if str(
            source_gate10b_status.get(
                "trusted_tree_semantics_broadening_pressure_status", ""
            )
        )
        == "clear"
        and not any_semantics_broadening
        else "triggered"
    )

    broader_tree_settlement_non_promotion_status = (
        "clear"
        if str(
            source_gate10b_status.get(
                "broader_tree_settlement_non_promotion_status", ""
            )
        )
        == "clear"
        and not any(bool(row["implies_broader_tree_settlement"]) for row in registry_rows)
        else "violated"
    )

    operator_admission_still_denied_status = (
        "confirmed" if not any_operator_open else "violated"
    )

    baseline_edge_ids = {str(row["edge_id"]) for row in baseline_rows}
    candidate_edge_ids = {str(row["edge_id"]) for row in candidate_rows}
    decision_relevant_gain_beyond_baseline_status = (
        "present"
        if comparison_slice_complete
        and second_candidate_declaration_status == "clear"
        and conflict_side_bridge_preservation_status == "clear"
        and bool(candidate_edge_ids - baseline_edge_ids)
        else "absent"
    )

    if not source_gate10b_preserved:
        comparison_outcome_status = "deferred"
        next_named_blocker = str(source_gate10b_status.get("next_named_blocker", ""))
    elif not comparison_slice_complete:
        comparison_outcome_status = "deferred"
        next_named_blocker = "comparison_slice_incomplete"
    elif forward_basis_baseline_preservation_status != "clear":
        comparison_outcome_status = "denied"
        next_named_blocker = "forward_basis_baseline_preservation_fails"
    elif gate10b_slice_non_retroactive_preservation_status != "clear":
        comparison_outcome_status = "denied"
        next_named_blocker = "gate10b_slice_reinterpretation_pressure"
    elif second_candidate_declaration_status != "clear":
        comparison_outcome_status = "denied"
        next_named_blocker = "second_candidate_declaration_integrity_fails"
    elif conflict_side_bridge_preservation_status != "clear":
        comparison_outcome_status = "denied"
        next_named_blocker = "conflict_side_bridge_degrades"
    elif non_retroactive_memory_preservation_status != "clear":
        comparison_outcome_status = "denied"
        next_named_blocker = "retroactive_reinterpretation_pressure"
    elif operator_adjacent_rescue_pressure_status != "clear":
        comparison_outcome_status = "denied"
        next_named_blocker = "operator_adjacent_rescue_pressure"
    elif trusted_tree_semantics_broadening_pressure_status != "clear":
        comparison_outcome_status = "denied"
        next_named_blocker = "silent_tree_semantics_broadening"
    elif broader_tree_settlement_non_promotion_status != "clear":
        comparison_outcome_status = "denied"
        next_named_blocker = "broader_tree_settlement_leak"
    elif decision_relevant_gain_beyond_baseline_status != "present":
        comparison_outcome_status = "bounded keep"
        next_named_blocker = "decision_relevant_gain_beyond_baseline_absent"
    else:
        comparison_outcome_status = "settled"
        next_named_blocker = ""

    return {
        "forward_basis_baseline_preservation_status": forward_basis_baseline_preservation_status,
        "gate10b_slice_non_retroactive_preservation_status": gate10b_slice_non_retroactive_preservation_status,
        "second_candidate_declaration_status": second_candidate_declaration_status,
        "conflict_side_bridge_preservation_status": conflict_side_bridge_preservation_status,
        "non_retroactive_memory_preservation_status": non_retroactive_memory_preservation_status,
        "operator_adjacent_rescue_pressure_status": operator_adjacent_rescue_pressure_status,
        "trusted_tree_semantics_broadening_pressure_status": trusted_tree_semantics_broadening_pressure_status,
        "decision_relevant_gain_beyond_baseline_status": decision_relevant_gain_beyond_baseline_status,
        "comparison_outcome_status": comparison_outcome_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "broader_tree_settlement_non_promotion_status": broader_tree_settlement_non_promotion_status,
        "next_named_blocker": next_named_blocker,
    }


def build_report(
    run_id: str,
    source_gate10b_manifest: Dict[str, Any],
    policy_compare_rows: Sequence[Dict[str, Any]],
    status_payload: Dict[str, Any],
) -> str:
    lines = [
        "# Gate10C Second Trusted-Tree Settlement Comparison Read",
        "",
        f"run_id: {run_id}",
        f"source_gate10b_run_id: {source_gate10b_manifest.get('run_id', '')}",
        f"source_gate10b_code_git_commit: {source_gate10b_manifest.get('code_git_commit', '')}",
        f"source_gate10a_run_id: {source_gate10b_manifest.get('source_gate10a_run_id', '')}",
        f"source_gate9q_run_id: {source_gate10b_manifest.get('source_gate9q_run_id', '')}",
        "",
        "## Discipline",
        "",
        "- one narrow second comparison only",
        "- Gate9Q forward-basis split remains the preserved baseline",
        "- second candidate is the declaratively extracted distributed_incompatibility sublane only",
        "- Gate10B settled slice remains preserved, operator admission remains denied, and no scalar masking is used",
        "",
        "## Comparison Summary",
        "",
        "| comparison_lane | source_broader_candidate_class | cell_class | cell_id | n_edges | mean_defect | n_role_changed | n_forward_basis_preserved |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in policy_compare_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["comparison_lane"]),
                    str(row["source_broader_candidate_class"]),
                    str(row["cell_class"]),
                    str(row["cell_id"]),
                    str(row["n_edges"]),
                    ""
                    if row["mean_edge_transport_defect"] in (None, "")
                    else f"{float(row['mean_edge_transport_defect']):.6f}",
                    str(row["n_role_changed_by_adoption"]),
                    str(row["n_forward_basis_preserved"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Status",
            "",
            f"- forward_basis_baseline_preservation_status: `{status_payload['forward_basis_baseline_preservation_status']}`",
            f"- gate10b_slice_non_retroactive_preservation_status: `{status_payload['gate10b_slice_non_retroactive_preservation_status']}`",
            f"- second_candidate_declaration_status: `{status_payload['second_candidate_declaration_status']}`",
            f"- conflict_side_bridge_preservation_status: `{status_payload['conflict_side_bridge_preservation_status']}`",
            f"- non_retroactive_memory_preservation_status: `{status_payload['non_retroactive_memory_preservation_status']}`",
            f"- operator_adjacent_rescue_pressure_status: `{status_payload['operator_adjacent_rescue_pressure_status']}`",
            f"- trusted_tree_semantics_broadening_pressure_status: `{status_payload['trusted_tree_semantics_broadening_pressure_status']}`",
            f"- decision_relevant_gain_beyond_baseline_status: `{status_payload['decision_relevant_gain_beyond_baseline_status']}`",
            f"- comparison_outcome_status: `{status_payload['comparison_outcome_status']}`",
            f"- operator_admission_still_denied_status: `{status_payload['operator_admission_still_denied_status']}`",
            f"- broader_tree_settlement_non_promotion_status: `{status_payload['broader_tree_settlement_non_promotion_status']}`",
            f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
        ]
    )

    outcome = str(status_payload["comparison_outcome_status"])
    if outcome == "settled":
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- the declared distributed_incompatibility second candidate survives the narrow comparison against the preserved baseline",
                "- this remains a second slice-local settlement sentence only and does not declare broader Gate10 settlement",
            ]
        )
    elif outcome == "bounded keep":
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- the second candidate remains informative relative to baseline but does not earn doctrine-safe settlement",
                f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
            ]
        )
    elif outcome == "denied":
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- the second candidate fails the declared second settlement comparison",
                f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- the second comparison remains incomplete for settlement",
                f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    source_gate10b_dir = Path(args.gate10b_dir)
    out_dir = Path(args.out_dir)
    run_id = out_dir.name

    source_gate10b_manifest = gate9a.read_json(
        source_gate10b_dir / gate10b.DEFAULT_MANIFEST
    )
    source_registry_rows = gate9a.read_jsonl(
        source_gate10b_dir / gate10b.DEFAULT_REGISTRY
    )
    source_gate10b_status = gate9a.read_json(
        source_gate10b_dir / gate10b.DEFAULT_STATUS
    )

    registry_rows = build_comparison_registry(source_registry_rows)
    policy_compare_rows = build_policy_compare(registry_rows)
    status_payload = build_status_payload(registry_rows, source_gate10b_status)

    registry_path = out_dir / DEFAULT_REGISTRY
    policy_compare_path = out_dir / DEFAULT_POLICY_COMPARE
    status_path = out_dir / DEFAULT_STATUS
    report_path = out_dir / DEFAULT_REPORT
    manifest_path = out_dir / DEFAULT_MANIFEST
    checksums_path = out_dir / DEFAULT_CHECKSUMS

    gate9a.write_jsonl(registry_path, registry_rows)
    gate9a.write_csv(
        policy_compare_path,
        (
            "comparison_lane",
            "source_broader_candidate_class",
            "cell_class",
            "cell_id",
            "n_edges",
            "mean_edge_transport_defect",
            "n_role_changed_by_adoption",
            "n_forward_basis_preserved",
        ),
        policy_compare_rows,
    )
    gate9a.write_json(status_path, status_payload)
    gate9a.write_text(
        report_path,
        build_report(
            run_id=run_id,
            source_gate10b_manifest=source_gate10b_manifest,
            policy_compare_rows=policy_compare_rows,
            status_payload=status_payload,
        ),
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "run_id": run_id,
        "code_git_commit": gate9a.current_git_commit(),
        "source_gate10b_dir": gate9a.repo_relative_or_posix(source_gate10b_dir),
        "source_gate10b_run_id": str(source_gate10b_manifest.get("run_id") or ""),
        "source_gate10b_code_git_commit": str(
            source_gate10b_manifest.get("code_git_commit") or ""
        ),
        "source_gate10a_run_id": str(
            source_gate10b_manifest.get("source_gate10a_run_id") or ""
        ),
        "source_gate10a_code_git_commit": str(
            source_gate10b_manifest.get("source_gate10a_code_git_commit") or ""
        ),
        "source_gate9q_run_id": str(
            source_gate10b_manifest.get("source_gate9q_run_id") or ""
        ),
        "source_gate9q_code_git_commit": str(
            source_gate10b_manifest.get("source_gate9q_code_git_commit") or ""
        ),
        "paths": {
            DEFAULT_REGISTRY: gate9a.repo_relative_or_posix(registry_path),
            DEFAULT_POLICY_COMPARE: gate9a.repo_relative_or_posix(policy_compare_path),
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
            DEFAULT_POLICY_COMPARE: sha256_file(policy_compare_path),
            DEFAULT_STATUS: sha256_file(status_path),
            DEFAULT_REPORT: sha256_file(report_path),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())