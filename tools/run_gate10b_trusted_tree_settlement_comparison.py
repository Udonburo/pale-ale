#!/usr/bin/env python3
"""Run a Gate10B trusted-tree settlement comparison on Gate10A outputs."""

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import run_gate10a_trusted_tree_generalization_eligibility as gate10a
import run_gate9a_graph_gauge_consumer as gate9a


REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "gate10b_trusted_tree_settlement_comparison_v1"
METHOD_ID = "gate10b_trusted_tree_settlement_comparison_v1"

DEFAULT_REGISTRY = "trusted_tree_settlement_comparison_registry.jsonl"
DEFAULT_POLICY_COMPARE = "trusted_tree_settlement_comparison_policy_compare.csv"
DEFAULT_STATUS = "trusted_tree_settlement_comparison_status.json"
DEFAULT_REPORT = "gate10b_trusted_tree_settlement_comparison_read.md"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"

BASELINE_LANE = "adopted_split_baseline"
CANDIDATE_LANE = "broader_candidate_opening_lane"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Gate10B trusted-tree settlement comparison for one narrow "
            "broader-candidate lane against the preserved Gate9Q baseline."
        )
    )
    parser.add_argument("--gate10a-dir", required=True)
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
    return [row for row in registry_rows if str(row["broader_candidate_class"]) == lane_name]


def lane_cell_class(rows: Sequence[Dict[str, Any]]) -> str:
    classes = sorted({str(row["cell_class"]) for row in rows})
    return "|".join(classes)


def build_comparison_registry(
    source_registry_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    registry_rows: List[Dict[str, Any]] = []
    for row in source_registry_rows:
        lane_name = str(row["broader_candidate_class"])
        if lane_name not in (BASELINE_LANE, CANDIDATE_LANE):
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
                "broader_candidate_class": lane_name,
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
        grouped[str(row["broader_candidate_class"])].append(row)

    out_rows: List[Dict[str, Any]] = []
    for lane_name in (BASELINE_LANE, CANDIDATE_LANE):
        rows = grouped.get(lane_name, [])
        defects = [
            float(row["edge_transport_defect"])
            for row in rows
            if row["edge_transport_defect"] not in (None, "")
        ]
        out_rows.append(
            {
                "broader_candidate_class": lane_name,
                "cell_class": lane_cell_class(rows),
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
    source_gate10a_status: Dict[str, Any],
) -> Dict[str, Any]:
    baseline_rows = lane_rows(registry_rows, BASELINE_LANE)
    candidate_rows = lane_rows(registry_rows, CANDIDATE_LANE)
    comparison_slice_complete = bool(baseline_rows) and bool(candidate_rows)

    source_gate10a_ready = (
        str(source_gate10a_status.get("integrated_baseline_source_status", ""))
        == "clear"
        and str(
            source_gate10a_status.get(
                "forward_basis_adoption_preservation_status", ""
            )
        )
        == "clear"
        and str(
            source_gate10a_status.get(
                "broader_candidate_eligibility_status", ""
            )
        )
        == "eligible"
        and str(
            source_gate10a_status.get(
                "settlement_comparison_permission_status", ""
            )
        )
        == "permitted"
    )

    forward_basis_baseline_preservation_status = (
        "clear"
        if source_gate10a_ready
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
        if str(
            source_gate10a_status.get(
                "non_retroactive_memory_preservation_status", ""
            )
        )
        == "clear"
        and not any_retroactive
        else "denied"
    )

    any_operator_open = any(
        bool(row["implies_operator_admission_open"]) for row in registry_rows
    )
    operator_adjacent_rescue_pressure_status = (
        "clear"
        if str(
            source_gate10a_status.get(
                "operator_adjacent_rescue_pressure_status", ""
            )
        )
        == "clear"
        and not any_operator_open
        else "triggered"
    )

    any_semantics_broadening = any(bool(row["widens_doctrine"]) for row in registry_rows)
    trusted_tree_semantics_broadening_pressure_status = (
        "clear"
        if str(
            source_gate10a_status.get(
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
            source_gate10a_status.get(
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
        and conflict_side_bridge_preservation_status == "clear"
        and bool(candidate_edge_ids - baseline_edge_ids)
        else "absent"
    )

    if not source_gate10a_ready:
        comparison_outcome_status = "deferred"
        next_named_blocker = str(source_gate10a_status.get("next_named_blocker", ""))
    elif not comparison_slice_complete:
        comparison_outcome_status = "deferred"
        next_named_blocker = "comparison_slice_incomplete"
    elif forward_basis_baseline_preservation_status != "clear":
        comparison_outcome_status = "denied"
        next_named_blocker = "forward_basis_baseline_preservation_fails"
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
    source_gate10a_manifest: Dict[str, Any],
    policy_compare_rows: Sequence[Dict[str, Any]],
    status_payload: Dict[str, Any],
) -> str:
    lines = [
        "# Gate10B Trusted-Tree Settlement Comparison Read",
        "",
        f"run_id: {run_id}",
        f"source_gate10a_run_id: {source_gate10a_manifest.get('run_id', '')}",
        f"source_gate10a_code_git_commit: {source_gate10a_manifest.get('code_git_commit', '')}",
        f"source_gate9q_run_id: {source_gate10a_manifest.get('source_gate9q_run_id', '')}",
        "",
        "## Discipline",
        "",
        "- one narrow comparison only",
        "- Gate9Q forward-basis split remains the preserved baseline",
        "- operator admission remains denied",
        "- no retroactive rewrite, no semantic broadening, no scalar masking",
        "",
        "## Comparison Summary",
        "",
        "| broader_candidate_class | cell_class | n_edges | mean_defect | n_role_changed | n_forward_basis_preserved |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in policy_compare_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["broader_candidate_class"]),
                    str(row["cell_class"]),
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
                "- the declared broader candidate survives the narrow comparison against the preserved baseline",
                "- this settlement sentence is slice-local only and does not reopen operator admission or broader promotion",
            ]
        )
    elif outcome == "bounded keep":
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- the broader candidate remains informative relative to baseline but does not earn doctrine-safe settlement",
                f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
            ]
        )
    elif outcome == "denied":
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- the broader candidate fails the declared settlement comparison",
                f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- the comparison remains incomplete for settlement",
                f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    source_gate10a_dir = Path(args.gate10a_dir)
    out_dir = Path(args.out_dir)
    run_id = out_dir.name

    source_gate10a_manifest = gate9a.read_json(
        source_gate10a_dir / gate10a.DEFAULT_MANIFEST
    )
    source_registry_rows = gate9a.read_jsonl(
        source_gate10a_dir / gate10a.DEFAULT_REGISTRY
    )
    source_gate10a_status = gate9a.read_json(
        source_gate10a_dir / gate10a.DEFAULT_STATUS
    )

    registry_rows = build_comparison_registry(source_registry_rows)
    policy_compare_rows = build_policy_compare(registry_rows)
    status_payload = build_status_payload(registry_rows, source_gate10a_status)

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
            "broader_candidate_class",
            "cell_class",
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
            source_gate10a_manifest=source_gate10a_manifest,
            policy_compare_rows=policy_compare_rows,
            status_payload=status_payload,
        ),
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "run_id": run_id,
        "code_git_commit": gate9a.current_git_commit(),
        "source_gate10a_dir": gate9a.repo_relative_or_posix(source_gate10a_dir),
        "source_gate10a_run_id": str(source_gate10a_manifest.get("run_id") or ""),
        "source_gate10a_code_git_commit": str(
            source_gate10a_manifest.get("code_git_commit") or ""
        ),
        "source_gate9q_run_id": str(
            source_gate10a_manifest.get("source_gate9q_run_id") or ""
        ),
        "source_gate9q_code_git_commit": str(
            source_gate10a_manifest.get("source_gate9q_code_git_commit") or ""
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