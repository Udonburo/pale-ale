#!/usr/bin/env python3
"""Run a Gate11E explicit-declaration instantiation-path audit on Gate11D outputs."""

import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Sequence

import run_gate11d_one_bounded_line_insufficiency_explicit_declaration_audit as gate11d
import run_gate9a_graph_gauge_consumer as gate9a


SCHEMA_VERSION = "gate11e_explicit_declaration_instantiation_path_audit_v1"
METHOD_ID = "gate11e_explicit_declaration_instantiation_path_audit_v1"

DEFAULT_REGISTRY = "explicit_declaration_instantiation_path_registry.jsonl"
DEFAULT_POLICY_COMPARE = "explicit_declaration_instantiation_path_policy_compare.csv"
DEFAULT_STATUS = "explicit_declaration_instantiation_path_status.json"
DEFAULT_REPORT = "gate11e_explicit_declaration_instantiation_path_read.md"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"

REQUIRED_GATE11D_STATUS_KEYS = (
    "gate10_closeout_preservation_status",
    "gate11a_absence_result_preservation_status",
    "gate11c_declaration_surface_preservation_status",
    "broader_trusted_tree_settlement_still_unearned_status",
    "operator_admission_still_denied_status",
    "retroactive_reinterpretation_forbidden_status",
    "anti_inflation_boundary_status",
    "bounded_line_insufficiency_explicit_declaration_marker_status",
    "bounded_line_insufficiency_candidate_id_singularity_status",
    "bounded_line_insufficiency_class_singularity_status",
    "bounded_line_host_failure_statement_status",
    "one_bounded_line_insufficiency_explicit_declaration_status",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Gate11E explicit-declaration instantiation-path audit from the frozen "
            "Gate11D explicit-declaration run without declaring a candidate or deciding reopening eligibility."
        )
    )
    parser.add_argument("--gate11d-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_status_value(status_payload: Dict[str, Any], key: str) -> str:
    return str(status_payload.get(key, ""))


def source_is_incomplete(
    source_gate11d_manifest: Dict[str, Any], source_gate11d_status: Dict[str, Any], report_text: str
) -> bool:
    if not str(source_gate11d_manifest.get("run_id") or ""):
        return True
    if not report_text.strip():
        return True
    return any(not source_status_value(source_gate11d_status, key) for key in REQUIRED_GATE11D_STATUS_KEYS)


def build_registry(
    source_gate11d_manifest: Dict[str, Any],
    source_gate11d_status: Dict[str, Any],
    status_payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return [
        {
            "source_gate11d_run_id": str(source_gate11d_manifest.get("run_id") or ""),
            "source_gate11d_code_git_commit": str(
                source_gate11d_manifest.get("code_git_commit") or ""
            ),
            "gate10_closeout_preservation_status": str(
                status_payload["gate10_closeout_preservation_status"]
            ),
            "gate11a_absence_result_preservation_status": str(
                status_payload["gate11a_absence_result_preservation_status"]
            ),
            "gate11c_declaration_surface_preservation_status": str(
                status_payload["gate11c_declaration_surface_preservation_status"]
            ),
            "gate11d_not_yet_declared_state_preservation_status": str(
                status_payload["gate11d_not_yet_declared_state_preservation_status"]
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
            "missing_surface_component_naming_status": str(
                status_payload["missing_surface_component_naming_status"]
            ),
            "minimal_later_source_instantiation_rule_status": str(
                status_payload["minimal_later_source_instantiation_rule_status"]
            ),
            "anti_shortcut_boundary_status": str(
                status_payload["anti_shortcut_boundary_status"]
            ),
            "explicit_declaration_instantiation_path_status": str(
                status_payload["explicit_declaration_instantiation_path_status"]
            ),
            "source_marker_status": source_status_value(
                source_gate11d_status,
                "bounded_line_insufficiency_explicit_declaration_marker_status",
            ),
            "source_candidate_id_status": source_status_value(
                source_gate11d_status,
                "bounded_line_insufficiency_candidate_id_singularity_status",
            ),
            "source_class_status": source_status_value(
                source_gate11d_status,
                "bounded_line_insufficiency_class_singularity_status",
            ),
            "source_host_failure_status": source_status_value(
                source_gate11d_status,
                "bounded_line_host_failure_statement_status",
            ),
        }
    ]


def build_policy_compare(registry_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "source_gate11d_run_id": str(row["source_gate11d_run_id"]),
            "gate10_closeout_preservation_status": str(row["gate10_closeout_preservation_status"]),
            "gate11a_absence_result_preservation_status": str(
                row["gate11a_absence_result_preservation_status"]
            ),
            "gate11c_declaration_surface_preservation_status": str(
                row["gate11c_declaration_surface_preservation_status"]
            ),
            "gate11d_not_yet_declared_state_preservation_status": str(
                row["gate11d_not_yet_declared_state_preservation_status"]
            ),
            "broader_trusted_tree_settlement_still_unearned_status": str(
                row["broader_trusted_tree_settlement_still_unearned_status"]
            ),
            "operator_admission_still_denied_status": str(row["operator_admission_still_denied_status"]),
            "retroactive_reinterpretation_forbidden_status": str(
                row["retroactive_reinterpretation_forbidden_status"]
            ),
            "missing_surface_component_naming_status": str(
                row["missing_surface_component_naming_status"]
            ),
            "minimal_later_source_instantiation_rule_status": str(
                row["minimal_later_source_instantiation_rule_status"]
            ),
            "anti_shortcut_boundary_status": str(row["anti_shortcut_boundary_status"]),
            "explicit_declaration_instantiation_path_status": str(
                row["explicit_declaration_instantiation_path_status"]
            ),
        }
        for row in registry_rows
    ]


def build_status_payload(
    source_gate11d_manifest: Dict[str, Any],
    source_gate11d_status: Dict[str, Any],
    report_text: str,
) -> Dict[str, Any]:
    incomplete = source_is_incomplete(source_gate11d_manifest, source_gate11d_status, report_text)

    gate10_closeout_preservation_status = (
        "preserved"
        if source_status_value(source_gate11d_status, "gate10_closeout_preservation_status") == "preserved"
        else "not_preserved"
    )
    gate11a_absence_result_preservation_status = (
        "preserved"
        if source_status_value(source_gate11d_status, "gate11a_absence_result_preservation_status")
        == "preserved"
        else "not_preserved"
    )
    gate11c_declaration_surface_preservation_status = (
        "preserved"
        if source_status_value(
            source_gate11d_status, "gate11c_declaration_surface_preservation_status"
        )
        == "preserved"
        else "not_preserved"
    )
    gate11d_not_yet_declared_state_preservation_status = (
        "preserved"
        if source_status_value(
            source_gate11d_status, "one_bounded_line_insufficiency_explicit_declaration_status"
        )
        == "not_yet_declared"
        else "not_preserved"
    )
    broader_trusted_tree_settlement_still_unearned_status = (
        "confirmed"
        if source_status_value(
            source_gate11d_status, "broader_trusted_tree_settlement_still_unearned_status"
        )
        == "confirmed"
        else "not_confirmed"
    )
    operator_admission_still_denied_status = (
        "confirmed"
        if source_status_value(source_gate11d_status, "operator_admission_still_denied_status")
        == "confirmed"
        else "not_confirmed"
    )
    retroactive_reinterpretation_forbidden_status = (
        "confirmed"
        if source_status_value(
            source_gate11d_status, "retroactive_reinterpretation_forbidden_status"
        )
        == "confirmed"
        else "not_confirmed"
    )

    if incomplete:
        missing_surface_component_naming_status = "deferred"
    else:
        marker_absent = source_status_value(
            source_gate11d_status, "bounded_line_insufficiency_explicit_declaration_marker_status"
        ) == "absent"
        candidate_absent = source_status_value(
            source_gate11d_status, "bounded_line_insufficiency_candidate_id_singularity_status"
        ) == "absent"
        class_none = source_status_value(
            source_gate11d_status, "bounded_line_insufficiency_class_singularity_status"
        ) == "none"
        host_failure_absent = source_status_value(
            source_gate11d_status, "bounded_line_host_failure_statement_status"
        ) == "absent"
        if marker_absent and candidate_absent and class_none and host_failure_absent:
            missing_surface_component_naming_status = "named"
        elif (
            source_status_value(
                source_gate11d_status, "one_bounded_line_insufficiency_explicit_declaration_status"
            )
            == "declared"
        ):
            missing_surface_component_naming_status = "denied"
        else:
            missing_surface_component_naming_status = "not_yet_named"

    if incomplete:
        anti_shortcut_boundary_status = "deferred"
    elif (
        broader_trusted_tree_settlement_still_unearned_status != "confirmed"
        or operator_admission_still_denied_status != "confirmed"
        or retroactive_reinterpretation_forbidden_status != "confirmed"
        or source_status_value(source_gate11d_status, "anti_inflation_boundary_status") != "confirmed"
    ):
        anti_shortcut_boundary_status = "denied"
    else:
        anti_shortcut_boundary_status = "confirmed"

    if incomplete:
        minimal_later_source_instantiation_rule_status = "deferred"
    elif anti_shortcut_boundary_status == "denied":
        minimal_later_source_instantiation_rule_status = "denied"
    elif (
        gate11c_declaration_surface_preservation_status != "preserved"
        or gate11d_not_yet_declared_state_preservation_status != "preserved"
    ):
        minimal_later_source_instantiation_rule_status = "not_yet_defined"
    else:
        minimal_later_source_instantiation_rule_status = "defined"

    if incomplete:
        explicit_declaration_instantiation_path_status = "deferred"
    elif (
        gate10_closeout_preservation_status != "preserved"
        or gate11a_absence_result_preservation_status != "preserved"
        or gate11c_declaration_surface_preservation_status != "preserved"
        or gate11d_not_yet_declared_state_preservation_status != "preserved"
        or anti_shortcut_boundary_status == "denied"
        or broader_trusted_tree_settlement_still_unearned_status != "confirmed"
        or operator_admission_still_denied_status != "confirmed"
        or retroactive_reinterpretation_forbidden_status != "confirmed"
    ):
        explicit_declaration_instantiation_path_status = "denied"
    elif (
        missing_surface_component_naming_status == "named"
        and minimal_later_source_instantiation_rule_status == "defined"
        and anti_shortcut_boundary_status == "confirmed"
    ):
        explicit_declaration_instantiation_path_status = "path_defined"
    elif (
        missing_surface_component_naming_status == "deferred"
        or minimal_later_source_instantiation_rule_status == "deferred"
        or anti_shortcut_boundary_status == "deferred"
    ):
        explicit_declaration_instantiation_path_status = "deferred"
    else:
        explicit_declaration_instantiation_path_status = "not_yet_defined"

    if gate10_closeout_preservation_status != "preserved":
        next_named_blocker = "gate10_closeout_not_preserved"
    elif gate11a_absence_result_preservation_status != "preserved":
        next_named_blocker = "gate11a_absence_result_not_preserved"
    elif gate11c_declaration_surface_preservation_status != "preserved":
        next_named_blocker = "gate11c_declaration_surface_not_preserved"
    elif gate11d_not_yet_declared_state_preservation_status != "preserved":
        next_named_blocker = "gate11d_not_yet_declared_state_not_preserved"
    elif broader_trusted_tree_settlement_still_unearned_status != "confirmed":
        next_named_blocker = "broader_trusted_tree_settlement_not_explicitly_unearned"
    elif operator_admission_still_denied_status != "confirmed":
        next_named_blocker = "operator_admission_not_denied"
    elif retroactive_reinterpretation_forbidden_status != "confirmed":
        next_named_blocker = "retroactive_reinterpretation_pressure"
    elif anti_shortcut_boundary_status == "deferred":
        next_named_blocker = "controlling_source_incomplete"
    elif anti_shortcut_boundary_status == "denied":
        next_named_blocker = "anti_shortcut_boundary_not_intact"
    elif missing_surface_component_naming_status == "denied":
        next_named_blocker = "source_already_instantiates_declaration"
    elif missing_surface_component_naming_status == "not_yet_named":
        next_named_blocker = "missing_components_not_explicitly_named"
    elif minimal_later_source_instantiation_rule_status == "not_yet_defined":
        next_named_blocker = "same_source_instantiation_rule_not_defined"
    else:
        next_named_blocker = ""

    return {
        "gate10_closeout_preservation_status": gate10_closeout_preservation_status,
        "gate11a_absence_result_preservation_status": gate11a_absence_result_preservation_status,
        "gate11c_declaration_surface_preservation_status": gate11c_declaration_surface_preservation_status,
        "gate11d_not_yet_declared_state_preservation_status": gate11d_not_yet_declared_state_preservation_status,
        "broader_trusted_tree_settlement_still_unearned_status": broader_trusted_tree_settlement_still_unearned_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "retroactive_reinterpretation_forbidden_status": retroactive_reinterpretation_forbidden_status,
        "missing_surface_component_naming_status": missing_surface_component_naming_status,
        "minimal_later_source_instantiation_rule_status": minimal_later_source_instantiation_rule_status,
        "anti_shortcut_boundary_status": anti_shortcut_boundary_status,
        "explicit_declaration_instantiation_path_status": explicit_declaration_instantiation_path_status,
        "next_named_blocker": next_named_blocker,
    }


def build_report(
    run_id: str,
    source_gate11d_manifest: Dict[str, Any],
    policy_compare_rows: Sequence[Dict[str, Any]],
    status_payload: Dict[str, Any],
) -> str:
    lines = [
        "# Gate11E Explicit-Declaration Instantiation Path Read",
        "",
        f"run_id: {run_id}",
        f"source_gate11d_run_id: {source_gate11d_manifest.get('run_id', '')}",
        f"source_gate11d_code_git_commit: {source_gate11d_manifest.get('code_git_commit', '')}",
        "",
        "## Discipline",
        "",
        "- Gate11E asks only what the minimum later-source path would be from surface_defined but not_yet_declared to one honest explicit declaration",
        "- Gate11E does not declare a candidate",
        "- Gate11E does not re-audit declaration existence",
        "- Gate11E does not decide reopening eligibility",
        "- Gate10 closeout remains bounded and preserved",
        "- Gate11A absence remains preserved",
        "- Gate11C surface-defined result remains preserved",
        "- Gate11D not-yet-declared result remains preserved",
        "- broader trusted-tree settlement remains unearned",
        "- operator admission remains denied",
        "- no retroactive rewrite of Gate9, Gate10, Gate11A, Gate11B, Gate11C, or Gate11D memory is permitted",
        "",
        "## Source Summary",
        "",
        "| source_gate11d_run_id | gate10_closeout_preservation_status | gate11a_absence_result_preservation_status | gate11c_declaration_surface_preservation_status | gate11d_not_yet_declared_state_preservation_status | broader_trusted_tree_settlement_still_unearned_status | operator_admission_still_denied_status | retroactive_reinterpretation_forbidden_status | missing_surface_component_naming_status | minimal_later_source_instantiation_rule_status | anti_shortcut_boundary_status | explicit_declaration_instantiation_path_status |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in policy_compare_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["source_gate11d_run_id"]),
                    str(row["gate10_closeout_preservation_status"]),
                    str(row["gate11a_absence_result_preservation_status"]),
                    str(row["gate11c_declaration_surface_preservation_status"]),
                    str(row["gate11d_not_yet_declared_state_preservation_status"]),
                    str(row["broader_trusted_tree_settlement_still_unearned_status"]),
                    str(row["operator_admission_still_denied_status"]),
                    str(row["retroactive_reinterpretation_forbidden_status"]),
                    str(row["missing_surface_component_naming_status"]),
                    str(row["minimal_later_source_instantiation_rule_status"]),
                    str(row["anti_shortcut_boundary_status"]),
                    str(row["explicit_declaration_instantiation_path_status"]),
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
            f"- gate11c_declaration_surface_preservation_status: `{status_payload['gate11c_declaration_surface_preservation_status']}`",
            f"- gate11d_not_yet_declared_state_preservation_status: `{status_payload['gate11d_not_yet_declared_state_preservation_status']}`",
            f"- broader_trusted_tree_settlement_still_unearned_status: `{status_payload['broader_trusted_tree_settlement_still_unearned_status']}`",
            f"- operator_admission_still_denied_status: `{status_payload['operator_admission_still_denied_status']}`",
            f"- retroactive_reinterpretation_forbidden_status: `{status_payload['retroactive_reinterpretation_forbidden_status']}`",
            f"- missing_surface_component_naming_status: `{status_payload['missing_surface_component_naming_status']}`",
            f"- minimal_later_source_instantiation_rule_status: `{status_payload['minimal_later_source_instantiation_rule_status']}`",
            f"- anti_shortcut_boundary_status: `{status_payload['anti_shortcut_boundary_status']}`",
            f"- explicit_declaration_instantiation_path_status: `{status_payload['explicit_declaration_instantiation_path_status']}`",
            f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
            "",
            "## Minimum Later-Source Rule",
            "",
            "- one later frozen source must instantiate the declaration marker under the fixed Gate11C surface",
            "- that same source must carry one and only one bounded_line_insufficiency_candidate_id",
            "- that same source must carry one and only one Gate11B class for that same candidate id",
            "- that same source must explicitly state: the current bounded line cannot honestly host <candidate_id>",
            "- that same source must match status payload, registry row, and read sentence on the same single candidate id",
            "- no cross-run stitching, worker-side synthesis, broader-settlement promotion, retroactive rewrite, or graph-wide leap is admissible",
        ]
    )

    if status_payload["explicit_declaration_instantiation_path_status"] == "path_defined":
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- the minimum same-source additions required for one later honest explicit declaration are now fixed narrowly enough",
                "- Gate11E does not instantiate that declaration path here",
            ]
        )
    elif status_payload["explicit_declaration_instantiation_path_status"] == "deferred":
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- the instantiation-path audit remains deferred because the frozen source is incomplete or contradictory",
                f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
            ]
        )
    elif status_payload["explicit_declaration_instantiation_path_status"] == "denied":
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- the proposed instantiation path is denied under the frozen Gate11E boundary",
                f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- the minimum later-source instantiation path is not yet fixed narrowly enough",
                f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    source_gate11d_dir = Path(args.gate11d_dir)
    out_dir = Path(args.out_dir)
    run_id = out_dir.name

    out_dir.mkdir(parents=True, exist_ok=True)

    source_gate11d_manifest = gate9a.read_json(source_gate11d_dir / gate11d.DEFAULT_MANIFEST)
    source_gate11d_status = gate9a.read_json(source_gate11d_dir / gate11d.DEFAULT_STATUS)
    source_gate11d_report = (source_gate11d_dir / gate11d.DEFAULT_REPORT).read_text(encoding="utf-8")

    status_payload = build_status_payload(
        source_gate11d_manifest,
        source_gate11d_status,
        source_gate11d_report,
    )
    registry_rows = build_registry(source_gate11d_manifest, source_gate11d_status, status_payload)
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
            "source_gate11d_run_id",
            "gate10_closeout_preservation_status",
            "gate11a_absence_result_preservation_status",
            "gate11c_declaration_surface_preservation_status",
            "gate11d_not_yet_declared_state_preservation_status",
            "broader_trusted_tree_settlement_still_unearned_status",
            "operator_admission_still_denied_status",
            "retroactive_reinterpretation_forbidden_status",
            "missing_surface_component_naming_status",
            "minimal_later_source_instantiation_rule_status",
            "anti_shortcut_boundary_status",
            "explicit_declaration_instantiation_path_status",
        ),
        policy_compare_rows,
    )
    gate9a.write_json(status_path, status_payload)
    gate9a.write_text(
        report_path,
        build_report(
            run_id=run_id,
            source_gate11d_manifest=source_gate11d_manifest,
            policy_compare_rows=policy_compare_rows,
            status_payload=status_payload,
        ),
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "run_id": run_id,
        "code_git_commit": gate9a.current_git_commit(),
        "source_gate11d_dir": gate9a.repo_relative_or_posix(source_gate11d_dir),
        "source_gate11d_run_id": str(source_gate11d_manifest.get("run_id") or ""),
        "source_gate11d_code_git_commit": str(
            source_gate11d_manifest.get("code_git_commit") or ""
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