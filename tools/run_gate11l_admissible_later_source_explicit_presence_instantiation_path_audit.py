#!/usr/bin/env python3
"""Run a Gate11L admissible later-source explicit-presence instantiation-path audit on Gate11K outputs."""

import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Sequence

import run_gate11k_one_admissible_later_source_explicit_presence_audit as gate11k
import run_gate9a_graph_gauge_consumer as gate9a


SCHEMA_VERSION = "gate11l_admissible_later_source_explicit_presence_instantiation_path_audit_v1"
METHOD_ID = "gate11l_admissible_later_source_explicit_presence_instantiation_path_audit_v1"

DEFAULT_REGISTRY = "admissible_later_source_explicit_presence_instantiation_path_registry.jsonl"
DEFAULT_POLICY_COMPARE = "admissible_later_source_explicit_presence_instantiation_path_policy_compare.csv"
DEFAULT_STATUS = "admissible_later_source_explicit_presence_instantiation_path_status.json"
DEFAULT_REPORT = "gate11l_admissible_later_source_explicit_presence_instantiation_path_read.md"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"

REQUIRED_GATE11K_STATUS_KEYS = (
    "gate10_closeout_preservation_status",
    "gate11a_absence_result_preservation_status",
    "gate11c_declaration_surface_preservation_status",
    "gate11d_not_yet_declared_state_preservation_status",
    "gate11e_path_defined_state_preservation_status",
    "gate11f_not_yet_admissible_state_preservation_status",
    "gate11g_naming_surface_preservation_status",
    "gate11h_not_yet_named_state_preservation_status",
    "gate11i_path_defined_state_preservation_status",
    "gate11j_not_yet_admissible_state_preservation_status",
    "broader_trusted_tree_settlement_still_unearned_status",
    "operator_admission_still_denied_status",
    "retroactive_reinterpretation_forbidden_status",
    "explicit_later_source_marker_status",
    "later_source_singularity_status",
    "same_source_path_attachment_status",
    "admissibility_boundary_status",
    "one_admissible_later_source_explicit_presence_status",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Gate11L admissible later-source explicit-presence instantiation-path audit "
            "from the frozen Gate11K explicit-presence run without deciding later-source admissibility."
        )
    )
    parser.add_argument("--gate11k-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_status_value(status_payload: Dict[str, Any], key: str) -> str:
    return str(status_payload.get(key, ""))


def source_is_incomplete(
    source_gate11k_manifest: Dict[str, Any], source_gate11k_status: Dict[str, Any], report_text: str
) -> bool:
    if not str(source_gate11k_manifest.get("run_id") or ""):
        return True
    if not report_text.strip():
        return True
    return any(not source_status_value(source_gate11k_status, key) for key in REQUIRED_GATE11K_STATUS_KEYS)


def build_registry(
    source_gate11k_manifest: Dict[str, Any],
    source_gate11k_status: Dict[str, Any],
    status_payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return [
        {
            "source_gate11k_run_id": str(source_gate11k_manifest.get("run_id") or ""),
            "source_gate11k_code_git_commit": str(
                source_gate11k_manifest.get("code_git_commit") or ""
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
            "gate11e_path_defined_state_preservation_status": str(
                status_payload["gate11e_path_defined_state_preservation_status"]
            ),
            "gate11f_not_yet_admissible_state_preservation_status": str(
                status_payload["gate11f_not_yet_admissible_state_preservation_status"]
            ),
            "gate11g_naming_surface_preservation_status": str(
                status_payload["gate11g_naming_surface_preservation_status"]
            ),
            "gate11h_not_yet_named_state_preservation_status": str(
                status_payload["gate11h_not_yet_named_state_preservation_status"]
            ),
            "gate11i_path_defined_state_preservation_status": str(
                status_payload["gate11i_path_defined_state_preservation_status"]
            ),
            "gate11j_not_yet_admissible_state_preservation_status": str(
                status_payload["gate11j_not_yet_admissible_state_preservation_status"]
            ),
            "gate11k_not_yet_present_state_preservation_status": str(
                status_payload["gate11k_not_yet_present_state_preservation_status"]
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
            "missing_explicit_presence_component_naming_status": str(
                status_payload["missing_explicit_presence_component_naming_status"]
            ),
            "minimal_same_source_admissible_presence_instantiation_rule_status": str(
                status_payload["minimal_same_source_admissible_presence_instantiation_rule_status"]
            ),
            "admissibility_boundary_status": str(
                status_payload["admissibility_boundary_status"]
            ),
            "admissible_later_source_explicit_presence_instantiation_path_status": str(
                status_payload["admissible_later_source_explicit_presence_instantiation_path_status"]
            ),
            "source_explicit_later_source_marker_status": source_status_value(
                source_gate11k_status,
                "explicit_later_source_marker_status",
            ),
            "source_later_source_singularity_status": source_status_value(
                source_gate11k_status,
                "later_source_singularity_status",
            ),
            "source_same_source_path_attachment_status": source_status_value(
                source_gate11k_status,
                "same_source_path_attachment_status",
            ),
        }
    ]


def build_policy_compare(registry_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "source_gate11k_run_id": str(row["source_gate11k_run_id"]),
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
            "gate11e_path_defined_state_preservation_status": str(
                row["gate11e_path_defined_state_preservation_status"]
            ),
            "gate11f_not_yet_admissible_state_preservation_status": str(
                row["gate11f_not_yet_admissible_state_preservation_status"]
            ),
            "gate11g_naming_surface_preservation_status": str(
                row["gate11g_naming_surface_preservation_status"]
            ),
            "gate11h_not_yet_named_state_preservation_status": str(
                row["gate11h_not_yet_named_state_preservation_status"]
            ),
            "gate11i_path_defined_state_preservation_status": str(
                row["gate11i_path_defined_state_preservation_status"]
            ),
            "gate11j_not_yet_admissible_state_preservation_status": str(
                row["gate11j_not_yet_admissible_state_preservation_status"]
            ),
            "gate11k_not_yet_present_state_preservation_status": str(
                row["gate11k_not_yet_present_state_preservation_status"]
            ),
            "broader_trusted_tree_settlement_still_unearned_status": str(
                row["broader_trusted_tree_settlement_still_unearned_status"]
            ),
            "operator_admission_still_denied_status": str(row["operator_admission_still_denied_status"]),
            "retroactive_reinterpretation_forbidden_status": str(
                row["retroactive_reinterpretation_forbidden_status"]
            ),
            "missing_explicit_presence_component_naming_status": str(
                row["missing_explicit_presence_component_naming_status"]
            ),
            "minimal_same_source_admissible_presence_instantiation_rule_status": str(
                row["minimal_same_source_admissible_presence_instantiation_rule_status"]
            ),
            "admissibility_boundary_status": str(row["admissibility_boundary_status"]),
            "admissible_later_source_explicit_presence_instantiation_path_status": str(
                row["admissible_later_source_explicit_presence_instantiation_path_status"]
            ),
        }
        for row in registry_rows
    ]


def build_status_payload(
    source_gate11k_manifest: Dict[str, Any],
    source_gate11k_status: Dict[str, Any],
    report_text: str,
) -> Dict[str, Any]:
    incomplete = source_is_incomplete(source_gate11k_manifest, source_gate11k_status, report_text)

    gate10_closeout_preservation_status = (
        "preserved"
        if source_status_value(source_gate11k_status, "gate10_closeout_preservation_status") == "preserved"
        else "not_preserved"
    )
    gate11a_absence_result_preservation_status = (
        "preserved"
        if source_status_value(source_gate11k_status, "gate11a_absence_result_preservation_status")
        == "preserved"
        else "not_preserved"
    )
    gate11c_declaration_surface_preservation_status = (
        "preserved"
        if source_status_value(
            source_gate11k_status, "gate11c_declaration_surface_preservation_status"
        )
        == "preserved"
        else "not_preserved"
    )
    gate11d_not_yet_declared_state_preservation_status = (
        "preserved"
        if source_status_value(
            source_gate11k_status, "gate11d_not_yet_declared_state_preservation_status"
        )
        == "preserved"
        else "not_preserved"
    )
    gate11e_path_defined_state_preservation_status = (
        "preserved"
        if source_status_value(
            source_gate11k_status, "gate11e_path_defined_state_preservation_status"
        )
        == "preserved"
        else "not_preserved"
    )
    gate11f_not_yet_admissible_state_preservation_status = (
        "preserved"
        if source_status_value(
            source_gate11k_status, "gate11f_not_yet_admissible_state_preservation_status"
        )
        == "preserved"
        else "not_preserved"
    )
    gate11g_naming_surface_preservation_status = (
        "preserved"
        if source_status_value(source_gate11k_status, "gate11g_naming_surface_preservation_status")
        == "preserved"
        else "not_preserved"
    )
    gate11h_not_yet_named_state_preservation_status = (
        "preserved"
        if source_status_value(
            source_gate11k_status, "gate11h_not_yet_named_state_preservation_status"
        )
        == "preserved"
        else "not_preserved"
    )
    gate11i_path_defined_state_preservation_status = (
        "preserved"
        if source_status_value(
            source_gate11k_status, "gate11i_path_defined_state_preservation_status"
        )
        == "preserved"
        else "not_preserved"
    )
    gate11j_not_yet_admissible_state_preservation_status = (
        "preserved"
        if source_status_value(
            source_gate11k_status, "gate11j_not_yet_admissible_state_preservation_status"
        )
        == "preserved"
        else "not_preserved"
    )
    gate11k_not_yet_present_state_preservation_status = (
        "preserved"
        if source_status_value(
            source_gate11k_status, "one_admissible_later_source_explicit_presence_status"
        )
        == "not_yet_present"
        else "not_preserved"
    )
    broader_trusted_tree_settlement_still_unearned_status = (
        "confirmed"
        if source_status_value(
            source_gate11k_status, "broader_trusted_tree_settlement_still_unearned_status"
        )
        == "confirmed"
        else "not_confirmed"
    )
    operator_admission_still_denied_status = (
        "confirmed"
        if source_status_value(source_gate11k_status, "operator_admission_still_denied_status")
        == "confirmed"
        else "not_confirmed"
    )
    retroactive_reinterpretation_forbidden_status = (
        "confirmed"
        if source_status_value(
            source_gate11k_status, "retroactive_reinterpretation_forbidden_status"
        )
        == "confirmed"
        else "not_confirmed"
    )

    if incomplete:
        missing_explicit_presence_component_naming_status = "deferred"
    else:
        marker_absent = source_status_value(
            source_gate11k_status, "explicit_later_source_marker_status"
        ) == "absent"
        singularity_none = source_status_value(
            source_gate11k_status, "later_source_singularity_status"
        ) == "none"
        path_not_attached = source_status_value(
            source_gate11k_status, "same_source_path_attachment_status"
        ) == "not_attached"
        if marker_absent and singularity_none and path_not_attached:
            missing_explicit_presence_component_naming_status = "named"
        else:
            missing_explicit_presence_component_naming_status = "not_named"

    if incomplete:
        admissibility_boundary_status = "deferred"
    elif (
        broader_trusted_tree_settlement_still_unearned_status != "confirmed"
        or operator_admission_still_denied_status != "confirmed"
        or retroactive_reinterpretation_forbidden_status != "confirmed"
        or source_status_value(source_gate11k_status, "admissibility_boundary_status") != "confirmed"
    ):
        admissibility_boundary_status = "denied"
    else:
        admissibility_boundary_status = "confirmed"

    if incomplete:
        minimal_same_source_admissible_presence_instantiation_rule_status = "deferred"
    elif admissibility_boundary_status == "denied":
        minimal_same_source_admissible_presence_instantiation_rule_status = "not_defined"
    elif (
        gate11j_not_yet_admissible_state_preservation_status != "preserved"
        or gate11k_not_yet_present_state_preservation_status != "preserved"
    ):
        minimal_same_source_admissible_presence_instantiation_rule_status = "not_defined"
    else:
        minimal_same_source_admissible_presence_instantiation_rule_status = "defined"

    if incomplete:
        admissible_later_source_explicit_presence_instantiation_path_status = "deferred"
    elif (
        gate10_closeout_preservation_status != "preserved"
        or gate11a_absence_result_preservation_status != "preserved"
        or gate11c_declaration_surface_preservation_status != "preserved"
        or gate11d_not_yet_declared_state_preservation_status != "preserved"
        or gate11e_path_defined_state_preservation_status != "preserved"
        or gate11f_not_yet_admissible_state_preservation_status != "preserved"
        or gate11g_naming_surface_preservation_status != "preserved"
        or gate11h_not_yet_named_state_preservation_status != "preserved"
        or gate11i_path_defined_state_preservation_status != "preserved"
        or gate11j_not_yet_admissible_state_preservation_status != "preserved"
        or gate11k_not_yet_present_state_preservation_status != "preserved"
        or admissibility_boundary_status == "denied"
        or broader_trusted_tree_settlement_still_unearned_status != "confirmed"
        or operator_admission_still_denied_status != "confirmed"
        or retroactive_reinterpretation_forbidden_status != "confirmed"
    ):
        admissible_later_source_explicit_presence_instantiation_path_status = "denied"
    elif (
        missing_explicit_presence_component_naming_status == "named"
        and minimal_same_source_admissible_presence_instantiation_rule_status == "defined"
        and admissibility_boundary_status == "confirmed"
    ):
        admissible_later_source_explicit_presence_instantiation_path_status = "path_defined"
    elif (
        missing_explicit_presence_component_naming_status == "deferred"
        or minimal_same_source_admissible_presence_instantiation_rule_status == "deferred"
        or admissibility_boundary_status == "deferred"
    ):
        admissible_later_source_explicit_presence_instantiation_path_status = "deferred"
    else:
        admissible_later_source_explicit_presence_instantiation_path_status = "not_yet_defined"

    if gate10_closeout_preservation_status != "preserved":
        next_named_blocker = "gate10_closeout_not_preserved"
    elif gate11a_absence_result_preservation_status != "preserved":
        next_named_blocker = "gate11a_absence_result_not_preserved"
    elif gate11c_declaration_surface_preservation_status != "preserved":
        next_named_blocker = "gate11c_declaration_surface_not_preserved"
    elif gate11d_not_yet_declared_state_preservation_status != "preserved":
        next_named_blocker = "gate11d_not_yet_declared_state_not_preserved"
    elif gate11e_path_defined_state_preservation_status != "preserved":
        next_named_blocker = "gate11e_path_defined_state_not_preserved"
    elif gate11f_not_yet_admissible_state_preservation_status != "preserved":
        next_named_blocker = "gate11f_not_yet_admissible_state_not_preserved"
    elif gate11g_naming_surface_preservation_status != "preserved":
        next_named_blocker = "gate11g_naming_surface_not_preserved"
    elif gate11h_not_yet_named_state_preservation_status != "preserved":
        next_named_blocker = "gate11h_not_yet_named_state_not_preserved"
    elif gate11i_path_defined_state_preservation_status != "preserved":
        next_named_blocker = "gate11i_path_defined_state_not_preserved"
    elif gate11j_not_yet_admissible_state_preservation_status != "preserved":
        next_named_blocker = "gate11j_not_yet_admissible_state_not_preserved"
    elif gate11k_not_yet_present_state_preservation_status != "preserved":
        next_named_blocker = "gate11k_not_yet_present_state_not_preserved"
    elif broader_trusted_tree_settlement_still_unearned_status != "confirmed":
        next_named_blocker = "broader_trusted_tree_settlement_not_explicitly_unearned"
    elif operator_admission_still_denied_status != "confirmed":
        next_named_blocker = "operator_admission_not_denied"
    elif retroactive_reinterpretation_forbidden_status != "confirmed":
        next_named_blocker = "retroactive_reinterpretation_pressure"
    elif admissibility_boundary_status == "deferred":
        next_named_blocker = "controlling_source_incomplete"
    elif admissibility_boundary_status == "denied":
        next_named_blocker = "admissibility_boundary_not_intact"
    elif missing_explicit_presence_component_naming_status == "not_named":
        next_named_blocker = "missing_explicit_presence_components_not_named"
    elif minimal_same_source_admissible_presence_instantiation_rule_status == "not_defined":
        next_named_blocker = "same_source_admissible_presence_instantiation_rule_not_defined"
    else:
        next_named_blocker = ""

    return {
        "gate10_closeout_preservation_status": gate10_closeout_preservation_status,
        "gate11a_absence_result_preservation_status": gate11a_absence_result_preservation_status,
        "gate11c_declaration_surface_preservation_status": gate11c_declaration_surface_preservation_status,
        "gate11d_not_yet_declared_state_preservation_status": gate11d_not_yet_declared_state_preservation_status,
        "gate11e_path_defined_state_preservation_status": gate11e_path_defined_state_preservation_status,
        "gate11f_not_yet_admissible_state_preservation_status": gate11f_not_yet_admissible_state_preservation_status,
        "gate11g_naming_surface_preservation_status": gate11g_naming_surface_preservation_status,
        "gate11h_not_yet_named_state_preservation_status": gate11h_not_yet_named_state_preservation_status,
        "gate11i_path_defined_state_preservation_status": gate11i_path_defined_state_preservation_status,
        "gate11j_not_yet_admissible_state_preservation_status": gate11j_not_yet_admissible_state_preservation_status,
        "gate11k_not_yet_present_state_preservation_status": gate11k_not_yet_present_state_preservation_status,
        "broader_trusted_tree_settlement_still_unearned_status": broader_trusted_tree_settlement_still_unearned_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "retroactive_reinterpretation_forbidden_status": retroactive_reinterpretation_forbidden_status,
        "missing_explicit_presence_component_naming_status": missing_explicit_presence_component_naming_status,
        "minimal_same_source_admissible_presence_instantiation_rule_status": minimal_same_source_admissible_presence_instantiation_rule_status,
        "admissibility_boundary_status": admissibility_boundary_status,
        "admissible_later_source_explicit_presence_instantiation_path_status": admissible_later_source_explicit_presence_instantiation_path_status,
        "next_named_blocker": next_named_blocker,
    }


def build_report(
    run_id: str,
    source_gate11k_manifest: Dict[str, Any],
    policy_compare_rows: Sequence[Dict[str, Any]],
    status_payload: Dict[str, Any],
) -> str:
    lines = [
        "# Gate11L Admissible Later-Source Explicit-Presence Instantiation Path Read",
        "",
        f"run_id: {run_id}",
        f"source_gate11k_run_id: {source_gate11k_manifest.get('run_id', '')}",
        f"source_gate11k_code_git_commit: {source_gate11k_manifest.get('code_git_commit', '')}",
        "",
        "## Discipline",
        "",
        "- Gate11L asks only what the minimum admissible path would be from not_yet_present to one future honest explicit admissible later-source presence",
        "- Gate11L does not admit a later source",
        "- Gate11L does not create explicit presence",
        "- Gate11L does not declare a candidate",
        "- Gate11L does not declare that explicit declaration already exists",
        "- Gate11L does not decide reopening eligibility",
        "- Gate10 closeout remains bounded and preserved",
        "- Gate11A absence remains preserved",
        "- Gate11C surface-defined result remains preserved",
        "- Gate11D not-yet-declared result remains preserved",
        "- Gate11E path-defined result remains preserved",
        "- Gate11F not-yet-admissible result remains preserved",
        "- Gate11G surface-defined result remains preserved",
        "- Gate11H not-yet-named result remains preserved",
        "- Gate11I path-defined result remains preserved",
        "- Gate11J not-yet-admissible result remains preserved",
        "- Gate11K not-yet-present result remains preserved",
        "- broader trusted-tree settlement remains unearned",
        "- operator admission remains denied",
        "- no retroactive rewrite of Gate9, Gate10, Gate11A, Gate11B, Gate11C, Gate11D, Gate11E, Gate11F, Gate11G, Gate11H, Gate11I, Gate11J, or Gate11K memory is permitted",
        "",
        "## Source Summary",
        "",
        "| source_gate11k_run_id | gate10_closeout_preservation_status | gate11a_absence_result_preservation_status | gate11c_declaration_surface_preservation_status | gate11d_not_yet_declared_state_preservation_status | gate11e_path_defined_state_preservation_status | gate11f_not_yet_admissible_state_preservation_status | gate11g_naming_surface_preservation_status | gate11h_not_yet_named_state_preservation_status | gate11i_path_defined_state_preservation_status | gate11j_not_yet_admissible_state_preservation_status | gate11k_not_yet_present_state_preservation_status | broader_trusted_tree_settlement_still_unearned_status | operator_admission_still_denied_status | retroactive_reinterpretation_forbidden_status | missing_explicit_presence_component_naming_status | minimal_same_source_admissible_presence_instantiation_rule_status | admissibility_boundary_status | admissible_later_source_explicit_presence_instantiation_path_status |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in policy_compare_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["source_gate11k_run_id"]),
                    str(row["gate10_closeout_preservation_status"]),
                    str(row["gate11a_absence_result_preservation_status"]),
                    str(row["gate11c_declaration_surface_preservation_status"]),
                    str(row["gate11d_not_yet_declared_state_preservation_status"]),
                    str(row["gate11e_path_defined_state_preservation_status"]),
                    str(row["gate11f_not_yet_admissible_state_preservation_status"]),
                    str(row["gate11g_naming_surface_preservation_status"]),
                    str(row["gate11h_not_yet_named_state_preservation_status"]),
                    str(row["gate11i_path_defined_state_preservation_status"]),
                    str(row["gate11j_not_yet_admissible_state_preservation_status"]),
                    str(row["gate11k_not_yet_present_state_preservation_status"]),
                    str(row["broader_trusted_tree_settlement_still_unearned_status"]),
                    str(row["operator_admission_still_denied_status"]),
                    str(row["retroactive_reinterpretation_forbidden_status"]),
                    str(row["missing_explicit_presence_component_naming_status"]),
                    str(row["minimal_same_source_admissible_presence_instantiation_rule_status"]),
                    str(row["admissibility_boundary_status"]),
                    str(row["admissible_later_source_explicit_presence_instantiation_path_status"]),
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
            f"- gate11e_path_defined_state_preservation_status: `{status_payload['gate11e_path_defined_state_preservation_status']}`",
            f"- gate11f_not_yet_admissible_state_preservation_status: `{status_payload['gate11f_not_yet_admissible_state_preservation_status']}`",
            f"- gate11g_naming_surface_preservation_status: `{status_payload['gate11g_naming_surface_preservation_status']}`",
            f"- gate11h_not_yet_named_state_preservation_status: `{status_payload['gate11h_not_yet_named_state_preservation_status']}`",
            f"- gate11i_path_defined_state_preservation_status: `{status_payload['gate11i_path_defined_state_preservation_status']}`",
            f"- gate11j_not_yet_admissible_state_preservation_status: `{status_payload['gate11j_not_yet_admissible_state_preservation_status']}`",
            f"- gate11k_not_yet_present_state_preservation_status: `{status_payload['gate11k_not_yet_present_state_preservation_status']}`",
            f"- broader_trusted_tree_settlement_still_unearned_status: `{status_payload['broader_trusted_tree_settlement_still_unearned_status']}`",
            f"- operator_admission_still_denied_status: `{status_payload['operator_admission_still_denied_status']}`",
            f"- retroactive_reinterpretation_forbidden_status: `{status_payload['retroactive_reinterpretation_forbidden_status']}`",
            f"- missing_explicit_presence_component_naming_status: `{status_payload['missing_explicit_presence_component_naming_status']}`",
            f"- minimal_same_source_admissible_presence_instantiation_rule_status: `{status_payload['minimal_same_source_admissible_presence_instantiation_rule_status']}`",
            f"- admissibility_boundary_status: `{status_payload['admissibility_boundary_status']}`",
            f"- admissible_later_source_explicit_presence_instantiation_path_status: `{status_payload['admissible_later_source_explicit_presence_instantiation_path_status']}`",
            f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
            "",
            "## Minimum Later-Source Presence Path",
            "",
            "- the current frozen source still names the missing explicit-presence components explicitly: no explicit later-source marker, no single later-source identity, and no explicit same-source path attachment on that later source",
            "- the minimum same-source admissible-presence instantiation rule is fixed narrowly: one same later source must carry one explicit later_source_id or later_frozen_run_id, one later source and only one later source, one declaration marker, one candidate id, one class, one explicit host-failure sentence, and matched status, registry, and read surfaces",
            "- the path remains bounded by the admissibility boundary: no shortcut, no inflation, no retroactive rewrite, no graph-wide leap, and no worker-side synthesis",
            "",
            "## Judgment",
            "",
        ]
    )

    if status_payload["admissible_later_source_explicit_presence_instantiation_path_status"] == "path_defined":
        lines.append(
            "- the minimum admissible path by which one explicit admissible later-source presence could later be honestly instantiated is now fixed, while the current source still does not admit a later source or create explicit presence"
        )
    elif status_payload["admissible_later_source_explicit_presence_instantiation_path_status"] == "not_yet_defined":
        lines.append(
            "- the preserved Gate11K line remains bounded, but the admissible later-source explicit-presence instantiation path is not yet fixed narrowly enough"
        )
    elif status_payload["admissible_later_source_explicit_presence_instantiation_path_status"] == "denied":
        lines.append(
            "- the attempted admissible later-source explicit-presence instantiation path is denied because it would require shortcut, inflation, rewrite, leap, or synthesis pressure"
        )
    else:
        lines.append(
            "- the frozen source is incomplete for an admissible later-source explicit-presence instantiation-path judgment"
        )

    if status_payload["next_named_blocker"]:
        lines.append(f"- next_named_blocker: `{status_payload['next_named_blocker']}`")

    lines.extend(
        [
            "",
            "## Memory Hook",
            "",
            "- Gate11L does not admit a later source; it fixes the minimum admissible path by which one explicit admissible later-source presence could later become honestly instantiated under the preserved Gate11K line.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    source_gate11k_dir = Path(args.gate11k_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = out_dir.name

    source_gate11k_manifest = gate9a.read_json(source_gate11k_dir / gate11k.DEFAULT_MANIFEST)
    source_gate11k_status = gate9a.read_json(source_gate11k_dir / gate11k.DEFAULT_STATUS)
    source_gate11k_report = (source_gate11k_dir / gate11k.DEFAULT_REPORT).read_text(encoding="utf-8")

    status_payload = build_status_payload(
        source_gate11k_manifest,
        source_gate11k_status,
        source_gate11k_report,
    )
    registry_rows = build_registry(source_gate11k_manifest, source_gate11k_status, status_payload)
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
            "source_gate11k_run_id",
            "gate10_closeout_preservation_status",
            "gate11a_absence_result_preservation_status",
            "gate11c_declaration_surface_preservation_status",
            "gate11d_not_yet_declared_state_preservation_status",
            "gate11e_path_defined_state_preservation_status",
            "gate11f_not_yet_admissible_state_preservation_status",
            "gate11g_naming_surface_preservation_status",
            "gate11h_not_yet_named_state_preservation_status",
            "gate11i_path_defined_state_preservation_status",
            "gate11j_not_yet_admissible_state_preservation_status",
            "gate11k_not_yet_present_state_preservation_status",
            "broader_trusted_tree_settlement_still_unearned_status",
            "operator_admission_still_denied_status",
            "retroactive_reinterpretation_forbidden_status",
            "missing_explicit_presence_component_naming_status",
            "minimal_same_source_admissible_presence_instantiation_rule_status",
            "admissibility_boundary_status",
            "admissible_later_source_explicit_presence_instantiation_path_status",
        ),
        policy_compare_rows,
    )
    gate9a.write_json(status_path, status_payload)
    gate9a.write_text(
        report_path,
        build_report(
            run_id=run_id,
            source_gate11k_manifest=source_gate11k_manifest,
            policy_compare_rows=policy_compare_rows,
            status_payload=status_payload,
        ),
    )

    manifest = {
        "run_id": run_id,
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "code_git_commit": gate9a.current_git_commit(),
        "source_gate11k_run_id": str(source_gate11k_manifest.get("run_id") or ""),
        "source_gate11k_code_git_commit": str(
            source_gate11k_manifest.get("code_git_commit") or ""
        ),
        "inputs": {
            "gate11k_dir": str(source_gate11k_dir),
        },
        "outputs": {
            "registry": str(registry_path),
            "policy_compare": str(policy_compare_path),
            "status": str(status_path),
            "report": str(report_path),
        },
    }
    gate9a.write_json(manifest_path, manifest)

    checksums = {
        DEFAULT_REGISTRY: sha256_file(registry_path),
        DEFAULT_POLICY_COMPARE: sha256_file(policy_compare_path),
        DEFAULT_STATUS: sha256_file(status_path),
        DEFAULT_REPORT: sha256_file(report_path),
        DEFAULT_MANIFEST: sha256_file(manifest_path),
    }
    gate9a.write_json(checksums_path, checksums)


if __name__ == "__main__":
    main()