#!/usr/bin/env python3
"""Run a Gate11W named residual marker-carrier completion audit on Gate11V outputs."""

import argparse
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

import run_gate11v_explicit_residual_completion_marker_carrier_completion_instantiation_path_audit as gate11v
import run_gate9a_graph_gauge_consumer as gate9a


SCHEMA_VERSION = "gate11w_named_residual_marker_carrier_completion_audit_v1"
METHOD_ID = "gate11w_named_residual_marker_carrier_completion_audit_v1"

DEFAULT_REGISTRY = "named_residual_marker_carrier_completion_registry.jsonl"
DEFAULT_POLICY_COMPARE = "named_residual_marker_carrier_completion_policy_compare.csv"
DEFAULT_STATUS = "named_residual_marker_carrier_completion_status.json"
DEFAULT_REPORT = "gate11w_named_residual_marker_carrier_completion_read.md"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"

REQUIRED_GATE11V_STATUS_KEYS = (
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
    "gate11l_path_defined_state_preservation_status",
    "gate11m_not_yet_present_state_preservation_status",
    "gate11n_residual_named_state_preservation_status",
    "gate11o_path_defined_state_preservation_status",
    "gate11p_not_yet_completed_state_preservation_status",
    "gate11q_surface_defined_state_preservation_status",
    "gate11r_not_yet_present_state_preservation_status",
    "gate11s_path_defined_state_preservation_status",
    "gate11t_not_yet_present_state_preservation_status",
    "gate11u_residual_named_state_preservation_status",
    "broader_trusted_tree_settlement_still_unearned_status",
    "operator_admission_still_denied_status",
    "retroactive_reinterpretation_forbidden_status",
    "named_residual_marker_carrier_condition_preservation_status",
    "minimum_same_source_carrier_completion_rule_status",
    "bounded_read_prefix_completion_requirement_status",
    "carrier_completion_boundary_status",
    "explicit_residual_completion_marker_carrier_completion_instantiation_path_status",
    "next_named_blocker",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Gate11W named residual marker-carrier completion audit "
            "from the frozen Gate11V path run without widening beyond the fixed Gate11V line."
        )
    )
    parser.add_argument("--gate11v-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_status_value(status_payload: Dict[str, Any], key: str) -> str:
    return str(status_payload.get(key, ""))


def source_is_incomplete(
    source_gate11v_manifest: Dict[str, Any], source_gate11v_status: Dict[str, Any], report_text: str
) -> bool:
    if not str(source_gate11v_manifest.get("run_id") or ""):
        return True
    if not report_text.strip():
        return True
    return any(key not in source_gate11v_status for key in REQUIRED_GATE11V_STATUS_KEYS)


def extract_later_source_ids(report_text: str) -> List[str]:
    patterns = [
        re.compile(r"(?im)^\s*residual_completion_later_source_id\s*[:=]\s*([a-z0-9_./-]+)\s*$"),
        re.compile(r"(?im)^\s*residual_completion_later_frozen_run_id\s*[:=]\s*([a-z0-9_./-]+)\s*$"),
    ]
    values: List[str] = []
    for pattern in patterns:
        values.extend(match.group(1).strip().lower() for match in pattern.finditer(report_text))
    return values


def explicit_carrier_completion_marker_present(report_text: str) -> bool:
    return bool(
        re.search(r"(?im)^\s*residual_completion_marker_status\s*[:=]\s*present\s*$", report_text)
    )


def extract_bounded_completion_surfaces(report_text: str) -> List[str]:
    pattern = re.compile(r"(?im)^\s*residual_completion_surface\s*[:=]\s*(.+?)\s*$")
    return [match.group(1).strip().lower() for match in pattern.finditer(report_text)]


def same_source_carrier_completion_instantiated(report_text: str) -> bool:
    required_phrases = (
        "one explicit residual completion marker",
        "one explicit later_source_id or later_frozen_run_id",
        "one marker and only one marker",
        "one explicit same-source path-attachment status",
        "one bounded read-prefix declaration for the marker",
        "one explicit admissible later-source presence marker",
        "one declaration marker",
        "one candidate id",
        "one class",
        "one explicit host-failure sentence",
        "matched status, registry, and read surfaces",
    )
    surfaces = set(extract_bounded_completion_surfaces(report_text))
    same_source_status_completed = bool(
        re.search(r"(?im)^\s*residual_completion_same_source_status\s*[:=]\s*completed\s*$", report_text)
    )
    return same_source_status_completed and all(phrase in surfaces for phrase in required_phrases)


def build_registry(
    source_gate11v_manifest: Dict[str, Any],
    status_payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return [
        {
            "source_gate11v_run_id": str(source_gate11v_manifest.get("run_id") or ""),
            "source_gate11v_code_git_commit": str(source_gate11v_manifest.get("code_git_commit") or ""),
            "gate10_closeout_preservation_status": str(status_payload["gate10_closeout_preservation_status"]),
            "gate11a_absence_result_preservation_status": str(status_payload["gate11a_absence_result_preservation_status"]),
            "gate11c_declaration_surface_preservation_status": str(status_payload["gate11c_declaration_surface_preservation_status"]),
            "gate11d_not_yet_declared_state_preservation_status": str(status_payload["gate11d_not_yet_declared_state_preservation_status"]),
            "gate11e_path_defined_state_preservation_status": str(status_payload["gate11e_path_defined_state_preservation_status"]),
            "gate11f_not_yet_admissible_state_preservation_status": str(status_payload["gate11f_not_yet_admissible_state_preservation_status"]),
            "gate11g_naming_surface_preservation_status": str(status_payload["gate11g_naming_surface_preservation_status"]),
            "gate11h_not_yet_named_state_preservation_status": str(status_payload["gate11h_not_yet_named_state_preservation_status"]),
            "gate11i_path_defined_state_preservation_status": str(status_payload["gate11i_path_defined_state_preservation_status"]),
            "gate11j_not_yet_admissible_state_preservation_status": str(status_payload["gate11j_not_yet_admissible_state_preservation_status"]),
            "gate11k_not_yet_present_state_preservation_status": str(status_payload["gate11k_not_yet_present_state_preservation_status"]),
            "gate11l_path_defined_state_preservation_status": str(status_payload["gate11l_path_defined_state_preservation_status"]),
            "gate11m_not_yet_present_state_preservation_status": str(status_payload["gate11m_not_yet_present_state_preservation_status"]),
            "gate11n_residual_named_state_preservation_status": str(status_payload["gate11n_residual_named_state_preservation_status"]),
            "gate11o_path_defined_state_preservation_status": str(status_payload["gate11o_path_defined_state_preservation_status"]),
            "gate11p_not_yet_completed_state_preservation_status": str(status_payload["gate11p_not_yet_completed_state_preservation_status"]),
            "gate11q_surface_defined_state_preservation_status": str(status_payload["gate11q_surface_defined_state_preservation_status"]),
            "gate11r_not_yet_present_state_preservation_status": str(status_payload["gate11r_not_yet_present_state_preservation_status"]),
            "gate11s_path_defined_state_preservation_status": str(status_payload["gate11s_path_defined_state_preservation_status"]),
            "gate11t_not_yet_present_state_preservation_status": str(status_payload["gate11t_not_yet_present_state_preservation_status"]),
            "gate11u_residual_named_state_preservation_status": str(status_payload["gate11u_residual_named_state_preservation_status"]),
            "gate11v_path_defined_state_preservation_status": str(status_payload["gate11v_path_defined_state_preservation_status"]),
            "broader_trusted_tree_settlement_still_unearned_status": str(status_payload["broader_trusted_tree_settlement_still_unearned_status"]),
            "operator_admission_still_denied_status": str(status_payload["operator_admission_still_denied_status"]),
            "retroactive_reinterpretation_forbidden_status": str(status_payload["retroactive_reinterpretation_forbidden_status"]),
            "named_residual_marker_carrier_condition_preservation_status": str(status_payload["named_residual_marker_carrier_condition_preservation_status"]),
            "explicit_carrier_completion_marker_status": str(status_payload["explicit_carrier_completion_marker_status"]),
            "same_source_carrier_completion_status": str(status_payload["same_source_carrier_completion_status"]),
            "carrier_completion_boundary_status": str(status_payload["carrier_completion_boundary_status"]),
            "named_residual_marker_carrier_completion_status": str(status_payload["named_residual_marker_carrier_completion_status"]),
            "next_named_blocker": str(status_payload["next_named_blocker"]),
        }
    ]


def build_policy_compare(registry_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "source_gate11v_run_id": str(row["source_gate11v_run_id"]),
            "gate10_closeout_preservation_status": str(row["gate10_closeout_preservation_status"]),
            "gate11a_absence_result_preservation_status": str(row["gate11a_absence_result_preservation_status"]),
            "gate11c_declaration_surface_preservation_status": str(row["gate11c_declaration_surface_preservation_status"]),
            "gate11d_not_yet_declared_state_preservation_status": str(row["gate11d_not_yet_declared_state_preservation_status"]),
            "gate11e_path_defined_state_preservation_status": str(row["gate11e_path_defined_state_preservation_status"]),
            "gate11f_not_yet_admissible_state_preservation_status": str(row["gate11f_not_yet_admissible_state_preservation_status"]),
            "gate11g_naming_surface_preservation_status": str(row["gate11g_naming_surface_preservation_status"]),
            "gate11h_not_yet_named_state_preservation_status": str(row["gate11h_not_yet_named_state_preservation_status"]),
            "gate11i_path_defined_state_preservation_status": str(row["gate11i_path_defined_state_preservation_status"]),
            "gate11j_not_yet_admissible_state_preservation_status": str(row["gate11j_not_yet_admissible_state_preservation_status"]),
            "gate11k_not_yet_present_state_preservation_status": str(row["gate11k_not_yet_present_state_preservation_status"]),
            "gate11l_path_defined_state_preservation_status": str(row["gate11l_path_defined_state_preservation_status"]),
            "gate11m_not_yet_present_state_preservation_status": str(row["gate11m_not_yet_present_state_preservation_status"]),
            "gate11n_residual_named_state_preservation_status": str(row["gate11n_residual_named_state_preservation_status"]),
            "gate11o_path_defined_state_preservation_status": str(row["gate11o_path_defined_state_preservation_status"]),
            "gate11p_not_yet_completed_state_preservation_status": str(row["gate11p_not_yet_completed_state_preservation_status"]),
            "gate11q_surface_defined_state_preservation_status": str(row["gate11q_surface_defined_state_preservation_status"]),
            "gate11r_not_yet_present_state_preservation_status": str(row["gate11r_not_yet_present_state_preservation_status"]),
            "gate11s_path_defined_state_preservation_status": str(row["gate11s_path_defined_state_preservation_status"]),
            "gate11t_not_yet_present_state_preservation_status": str(row["gate11t_not_yet_present_state_preservation_status"]),
            "gate11u_residual_named_state_preservation_status": str(row["gate11u_residual_named_state_preservation_status"]),
            "gate11v_path_defined_state_preservation_status": str(row["gate11v_path_defined_state_preservation_status"]),
            "broader_trusted_tree_settlement_still_unearned_status": str(row["broader_trusted_tree_settlement_still_unearned_status"]),
            "operator_admission_still_denied_status": str(row["operator_admission_still_denied_status"]),
            "retroactive_reinterpretation_forbidden_status": str(row["retroactive_reinterpretation_forbidden_status"]),
            "named_residual_marker_carrier_condition_preservation_status": str(row["named_residual_marker_carrier_condition_preservation_status"]),
            "explicit_carrier_completion_marker_status": str(row["explicit_carrier_completion_marker_status"]),
            "same_source_carrier_completion_status": str(row["same_source_carrier_completion_status"]),
            "carrier_completion_boundary_status": str(row["carrier_completion_boundary_status"]),
            "named_residual_marker_carrier_completion_status": str(row["named_residual_marker_carrier_completion_status"]),
            "next_named_blocker": str(row["next_named_blocker"]),
        }
        for row in registry_rows
    ]


def build_status_payload(
    source_gate11v_manifest: Dict[str, Any],
    source_gate11v_status: Dict[str, Any],
    report_text: str,
) -> Dict[str, Any]:
    incomplete = source_is_incomplete(source_gate11v_manifest, source_gate11v_status, report_text)

    gate10_closeout_preservation_status = "preserved" if source_status_value(source_gate11v_status, "gate10_closeout_preservation_status") == "preserved" else "not_preserved"
    gate11a_absence_result_preservation_status = "preserved" if source_status_value(source_gate11v_status, "gate11a_absence_result_preservation_status") == "preserved" else "not_preserved"
    gate11c_declaration_surface_preservation_status = "preserved" if source_status_value(source_gate11v_status, "gate11c_declaration_surface_preservation_status") == "preserved" else "not_preserved"
    gate11d_not_yet_declared_state_preservation_status = "preserved" if source_status_value(source_gate11v_status, "gate11d_not_yet_declared_state_preservation_status") == "preserved" else "not_preserved"
    gate11e_path_defined_state_preservation_status = "preserved" if source_status_value(source_gate11v_status, "gate11e_path_defined_state_preservation_status") == "preserved" else "not_preserved"
    gate11f_not_yet_admissible_state_preservation_status = "preserved" if source_status_value(source_gate11v_status, "gate11f_not_yet_admissible_state_preservation_status") == "preserved" else "not_preserved"
    gate11g_naming_surface_preservation_status = "preserved" if source_status_value(source_gate11v_status, "gate11g_naming_surface_preservation_status") == "preserved" else "not_preserved"
    gate11h_not_yet_named_state_preservation_status = "preserved" if source_status_value(source_gate11v_status, "gate11h_not_yet_named_state_preservation_status") == "preserved" else "not_preserved"
    gate11i_path_defined_state_preservation_status = "preserved" if source_status_value(source_gate11v_status, "gate11i_path_defined_state_preservation_status") == "preserved" else "not_preserved"
    gate11j_not_yet_admissible_state_preservation_status = "preserved" if source_status_value(source_gate11v_status, "gate11j_not_yet_admissible_state_preservation_status") == "preserved" else "not_preserved"
    gate11k_not_yet_present_state_preservation_status = "preserved" if source_status_value(source_gate11v_status, "gate11k_not_yet_present_state_preservation_status") == "preserved" else "not_preserved"
    gate11l_path_defined_state_preservation_status = "preserved" if source_status_value(source_gate11v_status, "gate11l_path_defined_state_preservation_status") == "preserved" else "not_preserved"
    gate11m_not_yet_present_state_preservation_status = "preserved" if source_status_value(source_gate11v_status, "gate11m_not_yet_present_state_preservation_status") == "preserved" else "not_preserved"
    gate11n_residual_named_state_preservation_status = "preserved" if source_status_value(source_gate11v_status, "gate11n_residual_named_state_preservation_status") == "preserved" else "not_preserved"
    gate11o_path_defined_state_preservation_status = "preserved" if source_status_value(source_gate11v_status, "gate11o_path_defined_state_preservation_status") == "preserved" else "not_preserved"
    gate11p_not_yet_completed_state_preservation_status = "preserved" if source_status_value(source_gate11v_status, "gate11p_not_yet_completed_state_preservation_status") == "preserved" else "not_preserved"
    gate11q_surface_defined_state_preservation_status = "preserved" if source_status_value(source_gate11v_status, "gate11q_surface_defined_state_preservation_status") == "preserved" else "not_preserved"
    gate11r_not_yet_present_state_preservation_status = "preserved" if source_status_value(source_gate11v_status, "gate11r_not_yet_present_state_preservation_status") == "preserved" else "not_preserved"
    gate11s_path_defined_state_preservation_status = "preserved" if source_status_value(source_gate11v_status, "gate11s_path_defined_state_preservation_status") == "preserved" else "not_preserved"
    gate11t_not_yet_present_state_preservation_status = "preserved" if source_status_value(source_gate11v_status, "gate11t_not_yet_present_state_preservation_status") == "preserved" else "not_preserved"
    gate11u_residual_named_state_preservation_status = "preserved" if source_status_value(source_gate11v_status, "gate11u_residual_named_state_preservation_status") == "preserved" else "not_preserved"
    source_gate11v_status_value = source_status_value(source_gate11v_status, "explicit_residual_completion_marker_carrier_completion_instantiation_path_status")
    if source_gate11v_status_value == "path_defined":
        gate11v_path_defined_state_preservation_status = "preserved"
    elif source_gate11v_status_value == "deferred":
        gate11v_path_defined_state_preservation_status = "deferred"
    else:
        gate11v_path_defined_state_preservation_status = "not_preserved"
    broader_trusted_tree_settlement_still_unearned_status = "confirmed" if source_status_value(source_gate11v_status, "broader_trusted_tree_settlement_still_unearned_status") == "confirmed" else "not_confirmed"
    operator_admission_still_denied_status = "confirmed" if source_status_value(source_gate11v_status, "operator_admission_still_denied_status") == "confirmed" else "not_confirmed"
    retroactive_reinterpretation_forbidden_status = "confirmed" if source_status_value(source_gate11v_status, "retroactive_reinterpretation_forbidden_status") == "confirmed" else "not_confirmed"

    if incomplete:
        named_residual_marker_carrier_condition_preservation_status = "deferred"
    elif source_status_value(source_gate11v_status, "named_residual_marker_carrier_condition_preservation_status") == "preserved":
        named_residual_marker_carrier_condition_preservation_status = "preserved"
    else:
        named_residual_marker_carrier_condition_preservation_status = "not_preserved"

    if incomplete:
        carrier_completion_boundary_status = "deferred"
    elif (
        broader_trusted_tree_settlement_still_unearned_status != "confirmed"
        or operator_admission_still_denied_status != "confirmed"
        or retroactive_reinterpretation_forbidden_status != "confirmed"
        or source_status_value(source_gate11v_status, "carrier_completion_boundary_status") != "confirmed"
    ):
        carrier_completion_boundary_status = "denied"
    else:
        carrier_completion_boundary_status = "confirmed"

    later_source_ids = extract_later_source_ids(report_text)
    unique_later_source_ids = sorted(set(later_source_ids))

    if incomplete:
        explicit_carrier_completion_marker_status = "deferred"
    elif explicit_carrier_completion_marker_present(report_text):
        explicit_carrier_completion_marker_status = "present"
    else:
        explicit_carrier_completion_marker_status = "absent"

    if incomplete:
        same_source_carrier_completion_status = "deferred"
    elif len(unique_later_source_ids) > 1:
        same_source_carrier_completion_status = "deferred"
    elif (
        explicit_carrier_completion_marker_status == "present"
        and len(unique_later_source_ids) == 1
        and same_source_carrier_completion_instantiated(report_text)
    ):
        same_source_carrier_completion_status = "completed"
    else:
        same_source_carrier_completion_status = "not_completed"

    if incomplete:
        named_residual_marker_carrier_completion_status = "deferred"
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
        or gate11l_path_defined_state_preservation_status != "preserved"
        or gate11m_not_yet_present_state_preservation_status != "preserved"
        or gate11n_residual_named_state_preservation_status != "preserved"
        or gate11o_path_defined_state_preservation_status != "preserved"
        or gate11p_not_yet_completed_state_preservation_status != "preserved"
        or gate11q_surface_defined_state_preservation_status != "preserved"
        or gate11r_not_yet_present_state_preservation_status != "preserved"
        or gate11s_path_defined_state_preservation_status != "preserved"
        or gate11t_not_yet_present_state_preservation_status != "preserved"
        or gate11u_residual_named_state_preservation_status != "preserved"
        or gate11v_path_defined_state_preservation_status == "not_preserved"
        or carrier_completion_boundary_status == "denied"
        or broader_trusted_tree_settlement_still_unearned_status != "confirmed"
        or operator_admission_still_denied_status != "confirmed"
        or retroactive_reinterpretation_forbidden_status != "confirmed"
    ):
        named_residual_marker_carrier_completion_status = "denied"
    elif (
        gate11v_path_defined_state_preservation_status == "deferred"
        or explicit_carrier_completion_marker_status == "deferred"
        or same_source_carrier_completion_status == "deferred"
        or carrier_completion_boundary_status == "deferred"
    ):
        named_residual_marker_carrier_completion_status = "deferred"
    elif (
        named_residual_marker_carrier_condition_preservation_status == "preserved"
        and explicit_carrier_completion_marker_status == "present"
        and same_source_carrier_completion_status == "completed"
        and carrier_completion_boundary_status == "confirmed"
    ):
        named_residual_marker_carrier_completion_status = "completed"
    else:
        named_residual_marker_carrier_completion_status = "not_yet_completed"

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
    elif gate11l_path_defined_state_preservation_status != "preserved":
        next_named_blocker = "gate11l_path_defined_state_not_preserved"
    elif gate11m_not_yet_present_state_preservation_status != "preserved":
        next_named_blocker = "gate11m_not_yet_present_state_not_preserved"
    elif gate11n_residual_named_state_preservation_status != "preserved":
        next_named_blocker = "gate11n_residual_named_state_not_preserved"
    elif gate11o_path_defined_state_preservation_status != "preserved":
        next_named_blocker = "gate11o_path_defined_state_not_preserved"
    elif gate11p_not_yet_completed_state_preservation_status != "preserved":
        next_named_blocker = "gate11p_not_yet_completed_state_not_preserved"
    elif gate11q_surface_defined_state_preservation_status != "preserved":
        next_named_blocker = "gate11q_surface_defined_state_not_preserved"
    elif gate11r_not_yet_present_state_preservation_status != "preserved":
        next_named_blocker = "gate11r_not_yet_present_state_not_preserved"
    elif gate11s_path_defined_state_preservation_status != "preserved":
        next_named_blocker = "gate11s_path_defined_state_not_preserved"
    elif gate11t_not_yet_present_state_preservation_status != "preserved":
        next_named_blocker = "gate11t_not_yet_present_state_not_preserved"
    elif gate11u_residual_named_state_preservation_status != "preserved":
        next_named_blocker = "gate11u_residual_named_state_not_preserved"
    elif gate11v_path_defined_state_preservation_status == "not_preserved":
        next_named_blocker = "gate11v_path_defined_state_not_preserved"
    elif broader_trusted_tree_settlement_still_unearned_status != "confirmed":
        next_named_blocker = "broader_trusted_tree_settlement_not_explicitly_unearned"
    elif operator_admission_still_denied_status != "confirmed":
        next_named_blocker = "operator_admission_not_denied"
    elif retroactive_reinterpretation_forbidden_status != "confirmed":
        next_named_blocker = "retroactive_reinterpretation_pressure"
    elif carrier_completion_boundary_status == "deferred":
        next_named_blocker = "controlling_source_incomplete"
    elif carrier_completion_boundary_status == "denied":
        next_named_blocker = "carrier_completion_boundary_not_intact"
    elif gate11v_path_defined_state_preservation_status == "deferred":
        next_named_blocker = "upstream_path_definition_deferred"
    elif named_residual_marker_carrier_condition_preservation_status != "preserved":
        next_named_blocker = "named_residual_marker_carrier_condition_not_preserved"
    elif explicit_carrier_completion_marker_status == "absent":
        next_named_blocker = "no_explicit_residual_completion_marker"
    elif same_source_carrier_completion_status == "deferred":
        next_named_blocker = "multiple_candidate_carriers"
    elif same_source_carrier_completion_status != "completed":
        next_named_blocker = "same_source_carrier_completion_not_instantiated"
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
        "gate11l_path_defined_state_preservation_status": gate11l_path_defined_state_preservation_status,
        "gate11m_not_yet_present_state_preservation_status": gate11m_not_yet_present_state_preservation_status,
        "gate11n_residual_named_state_preservation_status": gate11n_residual_named_state_preservation_status,
        "gate11o_path_defined_state_preservation_status": gate11o_path_defined_state_preservation_status,
        "gate11p_not_yet_completed_state_preservation_status": gate11p_not_yet_completed_state_preservation_status,
        "gate11q_surface_defined_state_preservation_status": gate11q_surface_defined_state_preservation_status,
        "gate11r_not_yet_present_state_preservation_status": gate11r_not_yet_present_state_preservation_status,
        "gate11s_path_defined_state_preservation_status": gate11s_path_defined_state_preservation_status,
        "gate11t_not_yet_present_state_preservation_status": gate11t_not_yet_present_state_preservation_status,
        "gate11u_residual_named_state_preservation_status": gate11u_residual_named_state_preservation_status,
        "gate11v_path_defined_state_preservation_status": gate11v_path_defined_state_preservation_status,
        "broader_trusted_tree_settlement_still_unearned_status": broader_trusted_tree_settlement_still_unearned_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "retroactive_reinterpretation_forbidden_status": retroactive_reinterpretation_forbidden_status,
        "named_residual_marker_carrier_condition_preservation_status": named_residual_marker_carrier_condition_preservation_status,
        "explicit_carrier_completion_marker_status": explicit_carrier_completion_marker_status,
        "same_source_carrier_completion_status": same_source_carrier_completion_status,
        "carrier_completion_boundary_status": carrier_completion_boundary_status,
        "named_residual_marker_carrier_completion_status": named_residual_marker_carrier_completion_status,
        "next_named_blocker": next_named_blocker,
    }


def build_report(
    run_id: str,
    source_gate11v_manifest: Dict[str, Any],
    policy_compare_rows: Sequence[Dict[str, Any]],
    status_payload: Dict[str, Any],
) -> str:
    lines = [
        "# Gate11W Named Residual Marker-Carrier Completion Read",
        "",
        f"run_id: {run_id}",
        f"source_gate11v_run_id: {source_gate11v_manifest.get('run_id', '')}",
        f"source_gate11v_code_git_commit: {source_gate11v_manifest.get('code_git_commit', '')}",
        "",
        "## Discipline",
        "",
        "- Gate11W asks only whether the named residual marker-carrier condition now actually counts as completed under the fixed Gate11V path",
        "- Gate11W audits completion only",
        "- Gate11W does not widen the line beyond the fixed Gate11V path",
        "- Gate11W does not instantiate a marker by leap",
        "- Gate11W does not admit a later source",
        "- Gate11W does not decide one-admissible-later-source explicit-presence judgment",
        "- Gate11W does not declare a bounded-line insufficiency candidate",
        "- Gate11W does not declare that explicit declaration already exists",
        "- Gate11W does not decide reopening eligibility",
        "- Gate11W does not reopen operator admission",
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
        "- Gate11L path-defined result remains preserved",
        "- Gate11M not-yet-present result remains preserved",
        "- Gate11N residual-named result remains preserved",
        "- Gate11O path-defined result remains preserved",
        "- Gate11P not-yet-completed result remains preserved",
        "- Gate11Q surface-defined result remains preserved",
        "- Gate11R not-yet-present result remains preserved",
        "- Gate11S path-defined result remains preserved",
        "- Gate11T not-yet-present result remains preserved",
        "- Gate11U residual-named result remains preserved",
        "- Gate11V path-defined result remains preserved",
        "- broader trusted-tree settlement remains unearned",
        "- operator admission remains denied",
        "- no retroactive rewrite of Gate9, Gate10, Gate11A, Gate11B, Gate11C, Gate11D, Gate11E, Gate11F, Gate11G, Gate11H, Gate11I, Gate11J, Gate11K, Gate11L, Gate11M, Gate11N, Gate11O, Gate11P, Gate11Q, Gate11R, Gate11S, Gate11T, Gate11U, or Gate11V memory is permitted",
        "",
        "## Source Summary",
        "",
        "| source_gate11v_run_id | gate10_closeout_preservation_status | gate11a_absence_result_preservation_status | gate11c_declaration_surface_preservation_status | gate11d_not_yet_declared_state_preservation_status | gate11e_path_defined_state_preservation_status | gate11f_not_yet_admissible_state_preservation_status | gate11g_naming_surface_preservation_status | gate11h_not_yet_named_state_preservation_status | gate11i_path_defined_state_preservation_status | gate11j_not_yet_admissible_state_preservation_status | gate11k_not_yet_present_state_preservation_status | gate11l_path_defined_state_preservation_status | gate11m_not_yet_present_state_preservation_status | gate11n_residual_named_state_preservation_status | gate11o_path_defined_state_preservation_status | gate11p_not_yet_completed_state_preservation_status | gate11q_surface_defined_state_preservation_status | gate11r_not_yet_present_state_preservation_status | gate11s_path_defined_state_preservation_status | gate11t_not_yet_present_state_preservation_status | gate11u_residual_named_state_preservation_status | gate11v_path_defined_state_preservation_status | broader_trusted_tree_settlement_still_unearned_status | operator_admission_still_denied_status | retroactive_reinterpretation_forbidden_status | named_residual_marker_carrier_condition_preservation_status | explicit_carrier_completion_marker_status | same_source_carrier_completion_status | carrier_completion_boundary_status | named_residual_marker_carrier_completion_status | next_named_blocker |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in policy_compare_rows:
        lines.append("| " + " | ".join([
            str(row["source_gate11v_run_id"]),
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
            str(row["gate11l_path_defined_state_preservation_status"]),
            str(row["gate11m_not_yet_present_state_preservation_status"]),
            str(row["gate11n_residual_named_state_preservation_status"]),
            str(row["gate11o_path_defined_state_preservation_status"]),
            str(row["gate11p_not_yet_completed_state_preservation_status"]),
            str(row["gate11q_surface_defined_state_preservation_status"]),
            str(row["gate11r_not_yet_present_state_preservation_status"]),
            str(row["gate11s_path_defined_state_preservation_status"]),
            str(row["gate11t_not_yet_present_state_preservation_status"]),
            str(row["gate11u_residual_named_state_preservation_status"]),
            str(row["gate11v_path_defined_state_preservation_status"]),
            str(row["broader_trusted_tree_settlement_still_unearned_status"]),
            str(row["operator_admission_still_denied_status"]),
            str(row["retroactive_reinterpretation_forbidden_status"]),
            str(row["named_residual_marker_carrier_condition_preservation_status"]),
            str(row["explicit_carrier_completion_marker_status"]),
            str(row["same_source_carrier_completion_status"]),
            str(row["carrier_completion_boundary_status"]),
            str(row["named_residual_marker_carrier_completion_status"]),
            str(row["next_named_blocker"]),
        ]) + " |")

    lines.extend([
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
        f"- gate11l_path_defined_state_preservation_status: `{status_payload['gate11l_path_defined_state_preservation_status']}`",
        f"- gate11m_not_yet_present_state_preservation_status: `{status_payload['gate11m_not_yet_present_state_preservation_status']}`",
        f"- gate11n_residual_named_state_preservation_status: `{status_payload['gate11n_residual_named_state_preservation_status']}`",
        f"- gate11o_path_defined_state_preservation_status: `{status_payload['gate11o_path_defined_state_preservation_status']}`",
        f"- gate11p_not_yet_completed_state_preservation_status: `{status_payload['gate11p_not_yet_completed_state_preservation_status']}`",
        f"- gate11q_surface_defined_state_preservation_status: `{status_payload['gate11q_surface_defined_state_preservation_status']}`",
        f"- gate11r_not_yet_present_state_preservation_status: `{status_payload['gate11r_not_yet_present_state_preservation_status']}`",
        f"- gate11s_path_defined_state_preservation_status: `{status_payload['gate11s_path_defined_state_preservation_status']}`",
        f"- gate11t_not_yet_present_state_preservation_status: `{status_payload['gate11t_not_yet_present_state_preservation_status']}`",
        f"- gate11u_residual_named_state_preservation_status: `{status_payload['gate11u_residual_named_state_preservation_status']}`",
        f"- gate11v_path_defined_state_preservation_status: `{status_payload['gate11v_path_defined_state_preservation_status']}`",
        f"- broader_trusted_tree_settlement_still_unearned_status: `{status_payload['broader_trusted_tree_settlement_still_unearned_status']}`",
        f"- operator_admission_still_denied_status: `{status_payload['operator_admission_still_denied_status']}`",
        f"- retroactive_reinterpretation_forbidden_status: `{status_payload['retroactive_reinterpretation_forbidden_status']}`",
        f"- named_residual_marker_carrier_condition_preservation_status: `{status_payload['named_residual_marker_carrier_condition_preservation_status']}`",
        f"- explicit_carrier_completion_marker_status: `{status_payload['explicit_carrier_completion_marker_status']}`",
        f"- same_source_carrier_completion_status: `{status_payload['same_source_carrier_completion_status']}`",
        f"- carrier_completion_boundary_status: `{status_payload['carrier_completion_boundary_status']}`",
        f"- named_residual_marker_carrier_completion_status: `{status_payload['named_residual_marker_carrier_completion_status']}`",
        f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
        "",
        "## Completion Slice",
        "",
        "- Gate11W counts only bounded same-source completion evidence under the fixed Gate11V path",
        "- path prose, hypothetical examples, and generic read narrative are not completion",
        "- worker-side inference or synthesis does not count",
        "",
        "## Judgment",
        "",
    ])

    outcome = status_payload["named_residual_marker_carrier_completion_status"]
    if outcome == "completed":
        lines.append("- the named residual marker-carrier condition is explicitly completed under the fixed Gate11V path")
    elif outcome == "not_yet_completed":
        lines.append("- the fixed Gate11V path remains preserved, but explicit same-source completion evidence is still absent")
    elif outcome == "denied":
        lines.append("- the attempted completion is denied because it would depend on shortcut, inflation, rewrite, graph-wide leap, or worker-side synthesis")
    else:
        lines.append("- the frozen source is incomplete or contradictory for a named residual marker-carrier completion judgment")
    if status_payload["next_named_blocker"]:
        lines.append(f"- next_named_blocker: `{status_payload['next_named_blocker']}`")
    lines.extend([
        "",
        "## Memory Hook",
        "",
        "- Gate11W does not widen the line beyond the fixed Gate11V path; it asks whether the named residual marker-carrier condition now actually counts as completed under that path.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    source_gate11v_dir = Path(args.gate11v_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = out_dir.name

    source_gate11v_manifest = gate9a.read_json(source_gate11v_dir / gate11v.DEFAULT_MANIFEST)
    source_gate11v_status = gate9a.read_json(source_gate11v_dir / gate11v.DEFAULT_STATUS)
    source_gate11v_report = (source_gate11v_dir / gate11v.DEFAULT_REPORT).read_text(encoding="utf-8")

    status_payload = build_status_payload(source_gate11v_manifest, source_gate11v_status, source_gate11v_report)
    registry_rows = build_registry(source_gate11v_manifest, status_payload)
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
            "source_gate11v_run_id",
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
            "gate11l_path_defined_state_preservation_status",
            "gate11m_not_yet_present_state_preservation_status",
            "gate11n_residual_named_state_preservation_status",
            "gate11o_path_defined_state_preservation_status",
            "gate11p_not_yet_completed_state_preservation_status",
            "gate11q_surface_defined_state_preservation_status",
            "gate11r_not_yet_present_state_preservation_status",
            "gate11s_path_defined_state_preservation_status",
            "gate11t_not_yet_present_state_preservation_status",
            "gate11u_residual_named_state_preservation_status",
            "gate11v_path_defined_state_preservation_status",
            "broader_trusted_tree_settlement_still_unearned_status",
            "operator_admission_still_denied_status",
            "retroactive_reinterpretation_forbidden_status",
            "named_residual_marker_carrier_condition_preservation_status",
            "explicit_carrier_completion_marker_status",
            "same_source_carrier_completion_status",
            "carrier_completion_boundary_status",
            "named_residual_marker_carrier_completion_status",
            "next_named_blocker",
        ),
        policy_compare_rows,
    )
    gate9a.write_json(status_path, status_payload)
    gate9a.write_text(report_path, build_report(run_id, source_gate11v_manifest, policy_compare_rows, status_payload))

    manifest = {
        "run_id": run_id,
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "code_git_commit": gate9a.current_git_commit(),
        "source_gate11v_run_id": str(source_gate11v_manifest.get("run_id") or ""),
        "source_gate11v_code_git_commit": str(source_gate11v_manifest.get("code_git_commit") or ""),
        "inputs": {"gate11v_dir": str(source_gate11v_dir)},
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