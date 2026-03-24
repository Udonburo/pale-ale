#!/usr/bin/env python3
"""Run a Gate11V explicit residual completion-marker carrier-completion instantiation-path audit on Gate11U outputs."""

import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Sequence

import run_gate11u_explicit_residual_completion_marker_carrier_completion_audit as gate11u
import run_gate9a_graph_gauge_consumer as gate9a


SCHEMA_VERSION = "gate11v_explicit_residual_completion_marker_carrier_completion_instantiation_path_audit_v1"
METHOD_ID = "gate11v_explicit_residual_completion_marker_carrier_completion_instantiation_path_audit_v1"

DEFAULT_REGISTRY = "explicit_residual_completion_marker_carrier_completion_instantiation_path_registry.jsonl"
DEFAULT_POLICY_COMPARE = "explicit_residual_completion_marker_carrier_completion_instantiation_path_policy_compare.csv"
DEFAULT_STATUS = "explicit_residual_completion_marker_carrier_completion_instantiation_path_status.json"
DEFAULT_REPORT = "gate11v_explicit_residual_completion_marker_carrier_completion_instantiation_path_read.md"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"

REQUIRED_GATE11U_STATUS_KEYS = (
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
    "broader_trusted_tree_settlement_still_unearned_status",
    "operator_admission_still_denied_status",
    "retroactive_reinterpretation_forbidden_status",
    "explicit_marker_carrier_completion_status",
    "marker_singularity_carrier_completion_status",
    "same_source_path_attachment_carrier_completion_status",
    "carrier_completion_boundary_status",
    "explicit_residual_completion_marker_carrier_completion_status",
    "next_named_blocker",
)

KNOWN_RESIDUAL_BLOCKERS = gate11u.KNOWN_RESIDUAL_BLOCKERS
PATH_DEFINABLE_RESIDUAL_BLOCKERS = {
    "no_explicit_residual_completion_marker",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Gate11V explicit residual completion-marker carrier-completion instantiation-path audit "
            "from the frozen Gate11U carrier-completion run without deciding completion or marker existence."
        )
    )
    parser.add_argument("--gate11u-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_status_value(status_payload: Dict[str, Any], key: str) -> str:
    return str(status_payload.get(key, ""))


def source_is_incomplete(
    source_gate11u_manifest: Dict[str, Any], source_gate11u_status: Dict[str, Any], report_text: str
) -> bool:
    if not str(source_gate11u_manifest.get("run_id") or ""):
        return True
    if not report_text.strip():
        return True
    return any(key not in source_gate11u_status for key in REQUIRED_GATE11U_STATUS_KEYS)


def blocker_sentence(blocker: str) -> str:
    mapping = {
        "no_explicit_residual_completion_marker": "the minimum honest path is fixed narrowly: one same later source must carry one explicit residual completion marker, one explicit later_source_id or later_frozen_run_id, one marker and only one marker, one explicit same-source path-attachment status, one bounded read-prefix declaration for the marker, repeated bounded residual_completion_surface rows for the required same-source elements, one explicit admissible later-source presence marker, one declaration marker, one candidate id, one class, one explicit host-failure sentence, and matched status, registry, and read surfaces",
        "minimum_same_source_carrier_completion_rule_not_fixed": "the named residual marker-carrier condition remains preserved, but the minimum same-source carrier-completion rule is not yet fixed narrowly enough",
        "bounded_read_prefix_completion_requirement_not_fixed": "the named residual marker-carrier condition remains preserved, but the bounded read-prefix completion requirement is not yet fixed narrowly enough",
        "named_residual_marker_carrier_condition_not_preserved": "the controlling source no longer preserves the named residual marker-carrier condition narrowly enough for path definition",
    }
    return mapping.get(blocker, "the explicit residual completion-marker carrier-completion instantiation path is not yet fixed narrowly enough")


def build_registry(
    source_gate11u_manifest: Dict[str, Any],
    status_payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return [
        {
            "source_gate11u_run_id": str(source_gate11u_manifest.get("run_id") or ""),
            "source_gate11u_code_git_commit": str(source_gate11u_manifest.get("code_git_commit") or ""),
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
            "broader_trusted_tree_settlement_still_unearned_status": str(status_payload["broader_trusted_tree_settlement_still_unearned_status"]),
            "operator_admission_still_denied_status": str(status_payload["operator_admission_still_denied_status"]),
            "retroactive_reinterpretation_forbidden_status": str(status_payload["retroactive_reinterpretation_forbidden_status"]),
            "named_residual_marker_carrier_condition_preservation_status": str(status_payload["named_residual_marker_carrier_condition_preservation_status"]),
            "minimum_same_source_carrier_completion_rule_status": str(status_payload["minimum_same_source_carrier_completion_rule_status"]),
            "bounded_read_prefix_completion_requirement_status": str(status_payload["bounded_read_prefix_completion_requirement_status"]),
            "carrier_completion_boundary_status": str(status_payload["carrier_completion_boundary_status"]),
            "explicit_residual_completion_marker_carrier_completion_instantiation_path_status": str(status_payload["explicit_residual_completion_marker_carrier_completion_instantiation_path_status"]),
            "next_named_blocker": str(status_payload["next_named_blocker"]),
        }
    ]


def build_policy_compare(registry_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "source_gate11u_run_id": str(row["source_gate11u_run_id"]),
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
            "broader_trusted_tree_settlement_still_unearned_status": str(row["broader_trusted_tree_settlement_still_unearned_status"]),
            "operator_admission_still_denied_status": str(row["operator_admission_still_denied_status"]),
            "retroactive_reinterpretation_forbidden_status": str(row["retroactive_reinterpretation_forbidden_status"]),
            "named_residual_marker_carrier_condition_preservation_status": str(row["named_residual_marker_carrier_condition_preservation_status"]),
            "minimum_same_source_carrier_completion_rule_status": str(row["minimum_same_source_carrier_completion_rule_status"]),
            "bounded_read_prefix_completion_requirement_status": str(row["bounded_read_prefix_completion_requirement_status"]),
            "carrier_completion_boundary_status": str(row["carrier_completion_boundary_status"]),
            "explicit_residual_completion_marker_carrier_completion_instantiation_path_status": str(row["explicit_residual_completion_marker_carrier_completion_instantiation_path_status"]),
            "next_named_blocker": str(row["next_named_blocker"]),
        }
        for row in registry_rows
    ]


def build_status_payload(
    source_gate11u_manifest: Dict[str, Any],
    source_gate11u_status: Dict[str, Any],
    report_text: str,
) -> Dict[str, Any]:
    incomplete = source_is_incomplete(source_gate11u_manifest, source_gate11u_status, report_text)

    gate10_closeout_preservation_status = "preserved" if source_status_value(source_gate11u_status, "gate10_closeout_preservation_status") == "preserved" else "not_preserved"
    gate11a_absence_result_preservation_status = "preserved" if source_status_value(source_gate11u_status, "gate11a_absence_result_preservation_status") == "preserved" else "not_preserved"
    gate11c_declaration_surface_preservation_status = "preserved" if source_status_value(source_gate11u_status, "gate11c_declaration_surface_preservation_status") == "preserved" else "not_preserved"
    gate11d_not_yet_declared_state_preservation_status = "preserved" if source_status_value(source_gate11u_status, "gate11d_not_yet_declared_state_preservation_status") == "preserved" else "not_preserved"
    gate11e_path_defined_state_preservation_status = "preserved" if source_status_value(source_gate11u_status, "gate11e_path_defined_state_preservation_status") == "preserved" else "not_preserved"
    gate11f_not_yet_admissible_state_preservation_status = "preserved" if source_status_value(source_gate11u_status, "gate11f_not_yet_admissible_state_preservation_status") == "preserved" else "not_preserved"
    gate11g_naming_surface_preservation_status = "preserved" if source_status_value(source_gate11u_status, "gate11g_naming_surface_preservation_status") == "preserved" else "not_preserved"
    gate11h_not_yet_named_state_preservation_status = "preserved" if source_status_value(source_gate11u_status, "gate11h_not_yet_named_state_preservation_status") == "preserved" else "not_preserved"
    gate11i_path_defined_state_preservation_status = "preserved" if source_status_value(source_gate11u_status, "gate11i_path_defined_state_preservation_status") == "preserved" else "not_preserved"
    gate11j_not_yet_admissible_state_preservation_status = "preserved" if source_status_value(source_gate11u_status, "gate11j_not_yet_admissible_state_preservation_status") == "preserved" else "not_preserved"
    gate11k_not_yet_present_state_preservation_status = "preserved" if source_status_value(source_gate11u_status, "gate11k_not_yet_present_state_preservation_status") == "preserved" else "not_preserved"
    gate11l_path_defined_state_preservation_status = "preserved" if source_status_value(source_gate11u_status, "gate11l_path_defined_state_preservation_status") == "preserved" else "not_preserved"
    gate11m_not_yet_present_state_preservation_status = "preserved" if source_status_value(source_gate11u_status, "gate11m_not_yet_present_state_preservation_status") == "preserved" else "not_preserved"
    gate11n_residual_named_state_preservation_status = "preserved" if source_status_value(source_gate11u_status, "gate11n_residual_named_state_preservation_status") == "preserved" else "not_preserved"
    gate11o_path_defined_state_preservation_status = "preserved" if source_status_value(source_gate11u_status, "gate11o_path_defined_state_preservation_status") == "preserved" else "not_preserved"
    gate11p_not_yet_completed_state_preservation_status = "preserved" if source_status_value(source_gate11u_status, "gate11p_not_yet_completed_state_preservation_status") == "preserved" else "not_preserved"
    gate11q_surface_defined_state_preservation_status = "preserved" if source_status_value(source_gate11u_status, "gate11q_surface_defined_state_preservation_status") == "preserved" else "not_preserved"
    gate11r_not_yet_present_state_preservation_status = "preserved" if source_status_value(source_gate11u_status, "gate11r_not_yet_present_state_preservation_status") == "preserved" else "not_preserved"
    gate11s_path_defined_state_preservation_status = "preserved" if source_status_value(source_gate11u_status, "gate11s_path_defined_state_preservation_status") == "preserved" else "not_preserved"
    gate11t_not_yet_present_state_preservation_status = "preserved" if source_status_value(source_gate11u_status, "gate11t_not_yet_present_state_preservation_status") == "preserved" else "not_preserved"
    source_gate11u_status_value = source_status_value(source_gate11u_status, "explicit_residual_completion_marker_carrier_completion_status")
    if source_gate11u_status_value == "residual_named":
        gate11u_residual_named_state_preservation_status = "preserved"
    elif source_gate11u_status_value == "deferred":
        gate11u_residual_named_state_preservation_status = "deferred"
    else:
        gate11u_residual_named_state_preservation_status = "not_preserved"
    broader_trusted_tree_settlement_still_unearned_status = "confirmed" if source_status_value(source_gate11u_status, "broader_trusted_tree_settlement_still_unearned_status") == "confirmed" else "not_confirmed"
    operator_admission_still_denied_status = "confirmed" if source_status_value(source_gate11u_status, "operator_admission_still_denied_status") == "confirmed" else "not_confirmed"
    retroactive_reinterpretation_forbidden_status = "confirmed" if source_status_value(source_gate11u_status, "retroactive_reinterpretation_forbidden_status") == "confirmed" else "not_confirmed"

    source_blocker = source_status_value(source_gate11u_status, "next_named_blocker")

    if incomplete:
        named_residual_marker_carrier_condition_preservation_status = "deferred"
    elif (
        source_status_value(source_gate11u_status, "explicit_marker_carrier_completion_status") == "missing"
        and source_status_value(source_gate11u_status, "marker_singularity_carrier_completion_status") == "missing"
        and source_status_value(source_gate11u_status, "same_source_path_attachment_carrier_completion_status") == "missing"
        and source_blocker in KNOWN_RESIDUAL_BLOCKERS
    ):
        named_residual_marker_carrier_condition_preservation_status = "preserved"
    else:
        named_residual_marker_carrier_condition_preservation_status = "not_preserved"

    if incomplete:
        carrier_completion_boundary_status = "deferred"
    elif (
        broader_trusted_tree_settlement_still_unearned_status != "confirmed"
        or operator_admission_still_denied_status != "confirmed"
        or retroactive_reinterpretation_forbidden_status != "confirmed"
        or source_status_value(source_gate11u_status, "carrier_completion_boundary_status") != "confirmed"
    ):
        carrier_completion_boundary_status = "denied"
    else:
        carrier_completion_boundary_status = "confirmed"

    if incomplete:
        minimum_same_source_carrier_completion_rule_status = "deferred"
    elif carrier_completion_boundary_status == "denied":
        minimum_same_source_carrier_completion_rule_status = "not_defined"
    elif gate11u_residual_named_state_preservation_status == "deferred":
        minimum_same_source_carrier_completion_rule_status = "deferred"
    elif gate11u_residual_named_state_preservation_status != "preserved":
        minimum_same_source_carrier_completion_rule_status = "not_defined"
    elif named_residual_marker_carrier_condition_preservation_status != "preserved":
        minimum_same_source_carrier_completion_rule_status = "not_defined"
    elif source_blocker in PATH_DEFINABLE_RESIDUAL_BLOCKERS:
        minimum_same_source_carrier_completion_rule_status = "defined"
    else:
        minimum_same_source_carrier_completion_rule_status = "not_defined"

    if incomplete:
        bounded_read_prefix_completion_requirement_status = "deferred"
    elif carrier_completion_boundary_status == "denied":
        bounded_read_prefix_completion_requirement_status = "not_defined"
    elif gate11u_residual_named_state_preservation_status == "deferred":
        bounded_read_prefix_completion_requirement_status = "deferred"
    elif minimum_same_source_carrier_completion_rule_status != "defined":
        bounded_read_prefix_completion_requirement_status = "not_defined"
    elif source_blocker in PATH_DEFINABLE_RESIDUAL_BLOCKERS:
        bounded_read_prefix_completion_requirement_status = "defined"
    else:
        bounded_read_prefix_completion_requirement_status = "not_defined"

    if incomplete:
        explicit_residual_completion_marker_carrier_completion_instantiation_path_status = "deferred"
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
        or gate11u_residual_named_state_preservation_status == "not_preserved"
        or carrier_completion_boundary_status == "denied"
        or broader_trusted_tree_settlement_still_unearned_status != "confirmed"
        or operator_admission_still_denied_status != "confirmed"
        or retroactive_reinterpretation_forbidden_status != "confirmed"
    ):
        explicit_residual_completion_marker_carrier_completion_instantiation_path_status = "denied"
    elif (
        gate11u_residual_named_state_preservation_status == "deferred"
        or named_residual_marker_carrier_condition_preservation_status == "deferred"
        or minimum_same_source_carrier_completion_rule_status == "deferred"
        or bounded_read_prefix_completion_requirement_status == "deferred"
        or carrier_completion_boundary_status == "deferred"
    ):
        explicit_residual_completion_marker_carrier_completion_instantiation_path_status = "deferred"
    elif (
        named_residual_marker_carrier_condition_preservation_status == "preserved"
        and minimum_same_source_carrier_completion_rule_status == "defined"
        and bounded_read_prefix_completion_requirement_status == "defined"
        and carrier_completion_boundary_status == "confirmed"
    ):
        explicit_residual_completion_marker_carrier_completion_instantiation_path_status = "path_defined"
    else:
        explicit_residual_completion_marker_carrier_completion_instantiation_path_status = "not_yet_defined"

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
    elif gate11u_residual_named_state_preservation_status == "not_preserved":
        next_named_blocker = "gate11u_residual_named_state_not_preserved"
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
    elif gate11u_residual_named_state_preservation_status == "deferred":
        next_named_blocker = "upstream_residual_naming_deferred"
    elif named_residual_marker_carrier_condition_preservation_status != "preserved":
        next_named_blocker = "named_residual_marker_carrier_condition_not_preserved"
    elif minimum_same_source_carrier_completion_rule_status != "defined":
        next_named_blocker = "minimum_same_source_carrier_completion_rule_not_fixed"
    elif bounded_read_prefix_completion_requirement_status != "defined":
        next_named_blocker = "bounded_read_prefix_completion_requirement_not_fixed"
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
        "broader_trusted_tree_settlement_still_unearned_status": broader_trusted_tree_settlement_still_unearned_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "retroactive_reinterpretation_forbidden_status": retroactive_reinterpretation_forbidden_status,
        "named_residual_marker_carrier_condition_preservation_status": named_residual_marker_carrier_condition_preservation_status,
        "minimum_same_source_carrier_completion_rule_status": minimum_same_source_carrier_completion_rule_status,
        "bounded_read_prefix_completion_requirement_status": bounded_read_prefix_completion_requirement_status,
        "carrier_completion_boundary_status": carrier_completion_boundary_status,
        "explicit_residual_completion_marker_carrier_completion_instantiation_path_status": explicit_residual_completion_marker_carrier_completion_instantiation_path_status,
        "next_named_blocker": next_named_blocker,
    }


def build_report(
    run_id: str,
    source_gate11u_manifest: Dict[str, Any],
    policy_compare_rows: Sequence[Dict[str, Any]],
    status_payload: Dict[str, Any],
) -> str:
    lines = [
        "# Gate11V Explicit Residual Completion-Marker Carrier-Completion Instantiation Path Read",
        "",
        f"run_id: {run_id}",
        f"source_gate11u_run_id: {source_gate11u_manifest.get('run_id', '')}",
        f"source_gate11u_code_git_commit: {source_gate11u_manifest.get('code_git_commit', '')}",
        "",
        "## Discipline",
        "",
        "- Gate11V asks only what is the minimum honest path by which the named residual marker-carrier condition could later be completed under the fixed Gate11U line",
        "- Gate11V defines path only",
        "- Gate11V does not satisfy the residual marker-carrier condition",
        "- Gate11V does not instantiate a marker",
        "- Gate11V does not complete the residual",
        "- Gate11V does not admit a later source",
        "- Gate11V does not decide one-admissible-later-source explicit-presence judgment",
        "- Gate11V does not declare a bounded-line insufficiency candidate",
        "- Gate11V does not declare that explicit declaration already exists",
        "- Gate11V does not decide reopening eligibility",
        "- Gate11V does not reopen operator admission",
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
        "- broader trusted-tree settlement remains unearned",
        "- operator admission remains denied",
        "- no retroactive rewrite of Gate9, Gate10, Gate11A, Gate11B, Gate11C, Gate11D, Gate11E, Gate11F, Gate11G, Gate11H, Gate11I, Gate11J, Gate11K, Gate11L, Gate11M, Gate11N, Gate11O, Gate11P, Gate11Q, Gate11R, Gate11S, Gate11T, or Gate11U memory is permitted",
        "",
        "## Source Summary",
        "",
        "| source_gate11u_run_id | gate10_closeout_preservation_status | gate11a_absence_result_preservation_status | gate11c_declaration_surface_preservation_status | gate11d_not_yet_declared_state_preservation_status | gate11e_path_defined_state_preservation_status | gate11f_not_yet_admissible_state_preservation_status | gate11g_naming_surface_preservation_status | gate11h_not_yet_named_state_preservation_status | gate11i_path_defined_state_preservation_status | gate11j_not_yet_admissible_state_preservation_status | gate11k_not_yet_present_state_preservation_status | gate11l_path_defined_state_preservation_status | gate11m_not_yet_present_state_preservation_status | gate11n_residual_named_state_preservation_status | gate11o_path_defined_state_preservation_status | gate11p_not_yet_completed_state_preservation_status | gate11q_surface_defined_state_preservation_status | gate11r_not_yet_present_state_preservation_status | gate11s_path_defined_state_preservation_status | gate11t_not_yet_present_state_preservation_status | gate11u_residual_named_state_preservation_status | broader_trusted_tree_settlement_still_unearned_status | operator_admission_still_denied_status | retroactive_reinterpretation_forbidden_status | named_residual_marker_carrier_condition_preservation_status | minimum_same_source_carrier_completion_rule_status | bounded_read_prefix_completion_requirement_status | carrier_completion_boundary_status | explicit_residual_completion_marker_carrier_completion_instantiation_path_status | next_named_blocker |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in policy_compare_rows:
        lines.append("| " + " | ".join([
            str(row["source_gate11u_run_id"]),
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
            str(row["broader_trusted_tree_settlement_still_unearned_status"]),
            str(row["operator_admission_still_denied_status"]),
            str(row["retroactive_reinterpretation_forbidden_status"]),
            str(row["named_residual_marker_carrier_condition_preservation_status"]),
            str(row["minimum_same_source_carrier_completion_rule_status"]),
            str(row["bounded_read_prefix_completion_requirement_status"]),
            str(row["carrier_completion_boundary_status"]),
            str(row["explicit_residual_completion_marker_carrier_completion_instantiation_path_status"]),
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
        f"- broader_trusted_tree_settlement_still_unearned_status: `{status_payload['broader_trusted_tree_settlement_still_unearned_status']}`",
        f"- operator_admission_still_denied_status: `{status_payload['operator_admission_still_denied_status']}`",
        f"- retroactive_reinterpretation_forbidden_status: `{status_payload['retroactive_reinterpretation_forbidden_status']}`",
        f"- named_residual_marker_carrier_condition_preservation_status: `{status_payload['named_residual_marker_carrier_condition_preservation_status']}`",
        f"- minimum_same_source_carrier_completion_rule_status: `{status_payload['minimum_same_source_carrier_completion_rule_status']}`",
        f"- bounded_read_prefix_completion_requirement_status: `{status_payload['bounded_read_prefix_completion_requirement_status']}`",
        f"- carrier_completion_boundary_status: `{status_payload['carrier_completion_boundary_status']}`",
        f"- explicit_residual_completion_marker_carrier_completion_instantiation_path_status: `{status_payload['explicit_residual_completion_marker_carrier_completion_instantiation_path_status']}`",
        f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
        "",
        "## Path Slice",
        "",
        "- Gate11V defines only the minimum same-source path by which the named residual marker-carrier condition could later be completed",
        "- path prose, hypothetical examples, and generic read narrative are not completion",
        "- worker-side inference or synthesis does not count",
        "",
        "## Judgment",
        "",
    ])

    outcome = status_payload["explicit_residual_completion_marker_carrier_completion_instantiation_path_status"]
    if outcome == "path_defined":
        lines.append(f"- {blocker_sentence('no_explicit_residual_completion_marker')}")
    elif outcome == "not_yet_defined":
        lines.append(f"- {blocker_sentence(status_payload['next_named_blocker'])}")
    elif outcome == "denied":
        lines.append("- the attempted path is denied because it would depend on shortcut, inflation, rewrite, graph-wide leap, or worker-side synthesis")
    else:
        lines.append("- the frozen source is incomplete or contradictory for an explicit residual completion-marker carrier-completion instantiation-path judgment")
    if status_payload["next_named_blocker"]:
        lines.append(f"- next_named_blocker: `{status_payload['next_named_blocker']}`")
    lines.extend([
        "",
        "## Memory Hook",
        "",
        "- Gate11V does not say the residual is completed; it asks what is the minimum honest path by which the named residual marker-carrier condition could later be completed under the fixed Gate11U line.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    source_gate11u_dir = Path(args.gate11u_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = out_dir.name

    source_gate11u_manifest = gate9a.read_json(source_gate11u_dir / gate11u.DEFAULT_MANIFEST)
    source_gate11u_status = gate9a.read_json(source_gate11u_dir / gate11u.DEFAULT_STATUS)
    source_gate11u_report = (source_gate11u_dir / gate11u.DEFAULT_REPORT).read_text(encoding="utf-8")

    status_payload = build_status_payload(source_gate11u_manifest, source_gate11u_status, source_gate11u_report)
    registry_rows = build_registry(source_gate11u_manifest, status_payload)
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
            "source_gate11u_run_id",
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
        ),
        policy_compare_rows,
    )
    gate9a.write_json(status_path, status_payload)
    gate9a.write_text(report_path, build_report(run_id, source_gate11u_manifest, policy_compare_rows, status_payload))

    manifest = {
        "run_id": run_id,
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "code_git_commit": gate9a.current_git_commit(),
        "source_gate11u_run_id": str(source_gate11u_manifest.get("run_id") or ""),
        "source_gate11u_code_git_commit": str(source_gate11u_manifest.get("code_git_commit") or ""),
        "inputs": {"gate11u_dir": str(source_gate11u_dir)},
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