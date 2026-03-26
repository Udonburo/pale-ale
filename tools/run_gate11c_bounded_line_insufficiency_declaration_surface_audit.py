#!/usr/bin/env python3
"""Run a Gate11C bounded-line insufficiency declaration-surface audit on Gate11B outputs."""

import argparse
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

import run_gate11b_bounded_line_insufficiency_declarability as gate11b
import run_gate9a_graph_gauge_consumer as gate9a


REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "gate11c_bounded_line_insufficiency_declaration_surface_audit_v1"
METHOD_ID = "gate11c_bounded_line_insufficiency_declaration_surface_audit_v1"

DEFAULT_REGISTRY = "bounded_line_insufficiency_declaration_surface_registry.jsonl"
DEFAULT_POLICY_COMPARE = "bounded_line_insufficiency_declaration_surface_policy_compare.csv"
DEFAULT_STATUS = "bounded_line_insufficiency_declaration_surface_status.json"
DEFAULT_REPORT = "gate11c_bounded_line_insufficiency_declaration_surface_read.md"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"

DECLARATION_CLASSES = (
    "tree_choice_instability",
    "current_bounded_line_insufficiency",
    "nonlocal_reconciliation_pressure",
    "narrow_reopening_pressure_without_graph_wide_leap",
)

REQUIRED_GATE11B_STATUS_KEYS = (
    "gate10_closeout_preservation_status",
    "gate11a_absence_result_preservation_status",
    "bounded_line_insufficiency_candidate_status",
    "bounded_line_insufficiency_class_status",
    "settlement_inflation_pressure_status",
    "graph_wide_operator_leap_pressure_status",
    "bounded_line_insufficiency_declarability_status",
    "broader_trusted_tree_settlement_still_unearned_status",
    "operator_admission_still_denied_status",
    "retroactive_reinterpretation_forbidden_status",
)

DECLARATION_PREFIX = "one bounded-line insufficiency candidate is explicitly declared:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Gate11C bounded-line insufficiency declaration-surface audit "
            "from the frozen Gate11B declarability run without declaring a candidate "
            "or deciding reopening eligibility."
        )
    )
    parser.add_argument("--gate11b-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_status_value(status_payload: Dict[str, Any], key: str) -> str:
    return str(status_payload.get(key, ""))


def source_is_incomplete(
    source_gate11b_manifest: Dict[str, Any], source_gate11b_status: Dict[str, Any], report_text: str
) -> bool:
    if not str(source_gate11b_manifest.get("run_id") or ""):
        return True
    if not report_text.strip():
        return True
    return any(not source_status_value(source_gate11b_status, key) for key in REQUIRED_GATE11B_STATUS_KEYS)


def extract_explicit_marker_values(report_text: str, marker: str, value_pattern: str = r"[a-z0-9_]+") -> List[str]:
    pattern = re.compile(rf"(?im)^\s*{re.escape(marker)}\s*[:=]\s*({value_pattern})\s*$")
    return [match.group(1).lower() for match in pattern.finditer(report_text)]


def inspect_declaration_surface_conflict(report_text: str) -> str:
    declaration_statuses = extract_explicit_marker_values(
        report_text,
        "bounded_line_insufficiency_candidate_declaration_status",
    )
    candidate_ids = extract_explicit_marker_values(
        report_text,
        "bounded_line_insufficiency_candidate_id",
    )
    class_statuses = extract_explicit_marker_values(
        report_text,
        "bounded_line_insufficiency_class_status",
    )
    host_failure_statuses = extract_explicit_marker_values(
        report_text,
        "bounded_line_host_failure_status",
    )
    declaration_prefix_count = sum(
        1 for line in report_text.lower().splitlines() if DECLARATION_PREFIX in line
    )

    any_surface_marker = any(
        [
            declaration_statuses,
            candidate_ids,
            class_statuses,
            host_failure_statuses,
            declaration_prefix_count,
        ]
    )
    if not any_surface_marker:
        return ""

    if any(status != "declared" for status in declaration_statuses):
        return "deferred"
    if any(status != "explicit" for status in host_failure_statuses):
        return "deferred"
    if any(value not in DECLARATION_CLASSES for value in class_statuses):
        return "deferred"
    if len(set(candidate_ids)) > 1 or len(set(class_statuses)) > 1 or declaration_prefix_count > 1:
        return "deferred"

    # Any declaration-surface marker in the frozen no-candidate source would require
    # worker-side resolution against the preserved absence state.
    return "deferred"


def build_registry(
    source_gate11b_manifest: Dict[str, Any],
    source_gate11b_status: Dict[str, Any],
    status_payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return [
        {
            "source_gate11b_run_id": str(source_gate11b_manifest.get("run_id") or ""),
            "source_gate11b_code_git_commit": str(
                source_gate11b_manifest.get("code_git_commit") or ""
            ),
            "gate10_closeout_preservation_status": str(
                status_payload["gate10_closeout_preservation_status"]
            ),
            "gate11a_absence_result_preservation_status": str(
                status_payload["gate11a_absence_result_preservation_status"]
            ),
            "gate11b_no_candidate_state_preservation_status": str(
                status_payload["gate11b_no_candidate_state_preservation_status"]
            ),
            "broader_trusted_tree_settlement_still_unearned_status": str(
                status_payload["broader_trusted_tree_settlement_still_unearned_status"]
            ),
            "operator_admission_still_denied_status": str(
                status_payload["operator_admission_still_denied_status"]
            ),
            "retroactive_reinterpretation_forbidden_status": str(
                status_payload["retroactive_reinterpretation_forbidden_status"]
            ),
            "bounded_line_insufficiency_candidate_status": source_status_value(
                source_gate11b_status, "bounded_line_insufficiency_candidate_status"
            ),
            "bounded_line_insufficiency_class_status": source_status_value(
                source_gate11b_status, "bounded_line_insufficiency_class_status"
            ),
            "bounded_line_insufficiency_declarability_status": source_status_value(
                source_gate11b_status, "bounded_line_insufficiency_declarability_status"
            ),
            "explicit_marker_shape_status": str(status_payload["explicit_marker_shape_status"]),
            "single_candidate_singularity_status": str(
                status_payload["single_candidate_singularity_status"]
            ),
            "bounded_line_insufficiency_evidence_shape_status": str(
                status_payload["bounded_line_insufficiency_evidence_shape_status"]
            ),
            "anti_inflation_boundary_status": str(
                status_payload["anti_inflation_boundary_status"]
            ),
            "bounded_line_insufficiency_declaration_surface_status": str(
                status_payload["bounded_line_insufficiency_declaration_surface_status"]
            ),
        }
    ]


def build_policy_compare(
    registry_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [
        {
            "source_gate11b_run_id": str(row["source_gate11b_run_id"]),
            "gate10_closeout_preservation_status": str(
                row["gate10_closeout_preservation_status"]
            ),
            "gate11a_absence_result_preservation_status": str(
                row["gate11a_absence_result_preservation_status"]
            ),
            "gate11b_no_candidate_state_preservation_status": str(
                row["gate11b_no_candidate_state_preservation_status"]
            ),
            "broader_trusted_tree_settlement_still_unearned_status": str(
                row["broader_trusted_tree_settlement_still_unearned_status"]
            ),
            "operator_admission_still_denied_status": str(
                row["operator_admission_still_denied_status"]
            ),
            "retroactive_reinterpretation_forbidden_status": str(
                row["retroactive_reinterpretation_forbidden_status"]
            ),
            "explicit_marker_shape_status": str(row["explicit_marker_shape_status"]),
            "single_candidate_singularity_status": str(
                row["single_candidate_singularity_status"]
            ),
            "bounded_line_insufficiency_evidence_shape_status": str(
                row["bounded_line_insufficiency_evidence_shape_status"]
            ),
            "anti_inflation_boundary_status": str(row["anti_inflation_boundary_status"]),
            "bounded_line_insufficiency_declaration_surface_status": str(
                row["bounded_line_insufficiency_declaration_surface_status"]
            ),
        }
        for row in registry_rows
    ]


def build_status_payload(
    source_gate11b_manifest: Dict[str, Any],
    source_gate11b_status: Dict[str, Any],
    report_text: str,
) -> Dict[str, Any]:
    incomplete = source_is_incomplete(source_gate11b_manifest, source_gate11b_status, report_text)
    source_surface_conflict = inspect_declaration_surface_conflict(report_text)

    gate10_closeout_preservation_status = (
        "preserved"
        if source_status_value(source_gate11b_status, "gate10_closeout_preservation_status")
        == "preserved"
        else "not_preserved"
    )
    gate11a_absence_result_preservation_status = (
        "preserved"
        if source_status_value(source_gate11b_status, "gate11a_absence_result_preservation_status")
        == "preserved"
        else "not_preserved"
    )
    gate11b_no_candidate_state_preservation_status = (
        "preserved"
        if source_status_value(source_gate11b_status, "bounded_line_insufficiency_candidate_status")
        == "absent"
        and source_status_value(source_gate11b_status, "bounded_line_insufficiency_class_status")
        == "none"
        and source_status_value(
            source_gate11b_status, "bounded_line_insufficiency_declarability_status"
        )
        == "not_yet_declarable"
        else "not_preserved"
    )
    broader_trusted_tree_settlement_still_unearned_status = (
        "confirmed"
        if source_status_value(
            source_gate11b_status, "broader_trusted_tree_settlement_still_unearned_status"
        )
        == "confirmed"
        else "not_confirmed"
    )
    operator_admission_still_denied_status = (
        "confirmed"
        if source_status_value(source_gate11b_status, "operator_admission_still_denied_status")
        == "confirmed"
        else "not_confirmed"
    )
    retroactive_reinterpretation_forbidden_status = (
        "confirmed"
        if source_status_value(
            source_gate11b_status, "retroactive_reinterpretation_forbidden_status"
        )
        == "confirmed"
        else "not_confirmed"
    )

    if incomplete or source_surface_conflict:
        explicit_marker_shape_status = "deferred"
        single_candidate_singularity_status = "deferred"
        bounded_line_insufficiency_evidence_shape_status = "deferred"
        anti_inflation_boundary_status = "deferred"
    else:
        explicit_marker_shape_status = "defined"
        single_candidate_singularity_status = "defined"
        bounded_line_insufficiency_evidence_shape_status = "defined"
        if (
            broader_trusted_tree_settlement_still_unearned_status != "confirmed"
            or operator_admission_still_denied_status != "confirmed"
            or retroactive_reinterpretation_forbidden_status != "confirmed"
            or source_status_value(source_gate11b_status, "settlement_inflation_pressure_status")
            == "present"
            or source_status_value(source_gate11b_status, "graph_wide_operator_leap_pressure_status")
            == "present"
        ):
            anti_inflation_boundary_status = "denied"
        else:
            anti_inflation_boundary_status = "defined"

    if incomplete or source_surface_conflict:
        bounded_line_insufficiency_declaration_surface_status = "deferred"
    elif (
        gate10_closeout_preservation_status != "preserved"
        or gate11a_absence_result_preservation_status != "preserved"
        or gate11b_no_candidate_state_preservation_status != "preserved"
        or broader_trusted_tree_settlement_still_unearned_status != "confirmed"
        or operator_admission_still_denied_status != "confirmed"
        or retroactive_reinterpretation_forbidden_status != "confirmed"
        or anti_inflation_boundary_status == "denied"
    ):
        bounded_line_insufficiency_declaration_surface_status = "denied"
    elif (
        explicit_marker_shape_status == "defined"
        and single_candidate_singularity_status == "defined"
        and bounded_line_insufficiency_evidence_shape_status == "defined"
        and anti_inflation_boundary_status == "defined"
    ):
        bounded_line_insufficiency_declaration_surface_status = "surface_defined"
    else:
        bounded_line_insufficiency_declaration_surface_status = "not_yet_defined"

    if gate10_closeout_preservation_status != "preserved":
        next_named_blocker = "gate10_closeout_not_preserved"
    elif gate11a_absence_result_preservation_status != "preserved":
        next_named_blocker = "gate11a_absence_result_not_preserved"
    elif gate11b_no_candidate_state_preservation_status != "preserved":
        next_named_blocker = "gate11b_no_candidate_state_not_preserved"
    elif broader_trusted_tree_settlement_still_unearned_status != "confirmed":
        next_named_blocker = "broader_trusted_tree_settlement_not_explicitly_unearned"
    elif operator_admission_still_denied_status != "confirmed":
        next_named_blocker = "operator_admission_not_denied"
    elif retroactive_reinterpretation_forbidden_status != "confirmed":
        next_named_blocker = "retroactive_reinterpretation_pressure"
    elif bounded_line_insufficiency_declaration_surface_status == "deferred":
        if incomplete:
            next_named_blocker = "controlling_source_incomplete"
        else:
            next_named_blocker = "declaration_surface_requires_worker_resolution"
    elif anti_inflation_boundary_status == "denied":
        if source_status_value(source_gate11b_status, "settlement_inflation_pressure_status") == "present":
            next_named_blocker = "settlement_inflation_pressure"
        elif source_status_value(source_gate11b_status, "graph_wide_operator_leap_pressure_status") == "present":
            next_named_blocker = "graph_wide_operator_leap_pressure"
        else:
            next_named_blocker = "anti_inflation_boundary_not_defined"
    else:
        next_named_blocker = ""

    return {
        "gate10_closeout_preservation_status": gate10_closeout_preservation_status,
        "gate11a_absence_result_preservation_status": gate11a_absence_result_preservation_status,
        "gate11b_no_candidate_state_preservation_status": gate11b_no_candidate_state_preservation_status,
        "broader_trusted_tree_settlement_still_unearned_status": broader_trusted_tree_settlement_still_unearned_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "retroactive_reinterpretation_forbidden_status": retroactive_reinterpretation_forbidden_status,
        "explicit_marker_shape_status": explicit_marker_shape_status,
        "single_candidate_singularity_status": single_candidate_singularity_status,
        "bounded_line_insufficiency_evidence_shape_status": bounded_line_insufficiency_evidence_shape_status,
        "anti_inflation_boundary_status": anti_inflation_boundary_status,
        "bounded_line_insufficiency_declaration_surface_status": bounded_line_insufficiency_declaration_surface_status,
        "next_named_blocker": next_named_blocker,
    }


def build_report(
    run_id: str,
    source_gate11b_manifest: Dict[str, Any],
    policy_compare_rows: Sequence[Dict[str, Any]],
    status_payload: Dict[str, Any],
) -> str:
    lines = [
        "# Gate11C Bounded-Line Insufficiency Declaration-Surface Read",
        "",
        f"run_id: {run_id}",
        f"source_gate11b_run_id: {source_gate11b_manifest.get('run_id', '')}",
        f"source_gate11b_code_git_commit: {source_gate11b_manifest.get('code_git_commit', '')}",
        "",
        "## Discipline",
        "",
        "- Gate11C asks only what would count as a valid explicit declaration surface later",
        "- Gate11C does not declare a candidate",
        "- Gate10 closeout remains bounded and preserved",
        "- Gate11A absence and Gate11B no-candidate state remain preserved",
        "- broader trusted-tree settlement remains unearned",
        "- operator admission remains denied",
        "- no retroactive rewrite of Gate9, Gate10, Gate11A, or Gate11B memory is permitted",
        "",
        "## Source Summary",
        "",
        "| source_gate11b_run_id | gate10_closeout_preservation_status | gate11a_absence_result_preservation_status | gate11b_no_candidate_state_preservation_status | broader_trusted_tree_settlement_still_unearned_status | operator_admission_still_denied_status | retroactive_reinterpretation_forbidden_status | explicit_marker_shape_status | single_candidate_singularity_status | bounded_line_insufficiency_evidence_shape_status | anti_inflation_boundary_status | bounded_line_insufficiency_declaration_surface_status |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in policy_compare_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["source_gate11b_run_id"]),
                    str(row["gate10_closeout_preservation_status"]),
                    str(row["gate11a_absence_result_preservation_status"]),
                    str(row["gate11b_no_candidate_state_preservation_status"]),
                    str(row["broader_trusted_tree_settlement_still_unearned_status"]),
                    str(row["operator_admission_still_denied_status"]),
                    str(row["retroactive_reinterpretation_forbidden_status"]),
                    str(row["explicit_marker_shape_status"]),
                    str(row["single_candidate_singularity_status"]),
                    str(row["bounded_line_insufficiency_evidence_shape_status"]),
                    str(row["anti_inflation_boundary_status"]),
                    str(row["bounded_line_insufficiency_declaration_surface_status"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Status",
            "",
            f"- gate10_closeout_preservation_status: `{status_payload['gate10_closeout_preservation_status']}`",
            f"- gate11a_absence_result_preservation_status: `{status_payload['gate11a_absence_result_preservation_status']}`",
            f"- gate11b_no_candidate_state_preservation_status: `{status_payload['gate11b_no_candidate_state_preservation_status']}`",
            f"- broader_trusted_tree_settlement_still_unearned_status: `{status_payload['broader_trusted_tree_settlement_still_unearned_status']}`",
            f"- operator_admission_still_denied_status: `{status_payload['operator_admission_still_denied_status']}`",
            f"- retroactive_reinterpretation_forbidden_status: `{status_payload['retroactive_reinterpretation_forbidden_status']}`",
            f"- explicit_marker_shape_status: `{status_payload['explicit_marker_shape_status']}`",
            f"- single_candidate_singularity_status: `{status_payload['single_candidate_singularity_status']}`",
            f"- bounded_line_insufficiency_evidence_shape_status: `{status_payload['bounded_line_insufficiency_evidence_shape_status']}`",
            f"- anti_inflation_boundary_status: `{status_payload['anti_inflation_boundary_status']}`",
            f"- bounded_line_insufficiency_declaration_surface_status: `{status_payload['bounded_line_insufficiency_declaration_surface_status']}`",
            f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
        ]
    )

    if status_payload["bounded_line_insufficiency_declaration_surface_status"] == "surface_defined":
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- the future declaration surface for one bounded-line insufficiency candidate is now fixed narrowly enough to audit later",
                "- no candidate is declared here and no reopening-eligibility judgment is made here",
            ]
        )
    elif status_payload["bounded_line_insufficiency_declaration_surface_status"] == "deferred":
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- the declaration-surface audit remains deferred because the frozen controlling source is incomplete or would require worker-side resolution",
                f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
            ]
        )
    elif status_payload["bounded_line_insufficiency_declaration_surface_status"] == "denied":
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- the proposed declaration surface is denied under the frozen Gate11C boundary",
                f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- the declaration surface is not yet fixed narrowly enough to audit later candidate declaration honestly",
                f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    source_gate11b_dir = Path(args.gate11b_dir)
    out_dir = Path(args.out_dir)
    run_id = out_dir.name

    out_dir.mkdir(parents=True, exist_ok=True)

    source_gate11b_manifest = gate9a.read_json(source_gate11b_dir / gate11b.DEFAULT_MANIFEST)
    source_gate11b_status = gate9a.read_json(source_gate11b_dir / gate11b.DEFAULT_STATUS)
    source_gate11b_report = (source_gate11b_dir / gate11b.DEFAULT_REPORT).read_text(
        encoding="utf-8"
    )

    status_payload = build_status_payload(
        source_gate11b_manifest, source_gate11b_status, source_gate11b_report
    )
    registry_rows = build_registry(source_gate11b_manifest, source_gate11b_status, status_payload)
    policy_compare_rows = build_policy_compare(registry_rows)

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
            "source_gate11b_run_id",
            "gate10_closeout_preservation_status",
            "gate11a_absence_result_preservation_status",
            "gate11b_no_candidate_state_preservation_status",
            "broader_trusted_tree_settlement_still_unearned_status",
            "operator_admission_still_denied_status",
            "retroactive_reinterpretation_forbidden_status",
            "explicit_marker_shape_status",
            "single_candidate_singularity_status",
            "bounded_line_insufficiency_evidence_shape_status",
            "anti_inflation_boundary_status",
            "bounded_line_insufficiency_declaration_surface_status",
        ),
        policy_compare_rows,
    )
    gate9a.write_json(status_path, status_payload)
    gate9a.write_text(
        report_path,
        build_report(
            run_id=run_id,
            source_gate11b_manifest=source_gate11b_manifest,
            policy_compare_rows=policy_compare_rows,
            status_payload=status_payload,
        ),
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "run_id": run_id,
        "code_git_commit": gate9a.current_git_commit(),
        "source_gate11b_dir": gate9a.repo_relative_or_posix(source_gate11b_dir),
        "source_gate11b_run_id": str(source_gate11b_manifest.get("run_id") or ""),
        "source_gate11b_code_git_commit": str(
            source_gate11b_manifest.get("code_git_commit") or ""
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