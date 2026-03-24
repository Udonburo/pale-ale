#!/usr/bin/env python3
"""Run a Gate11R one explicit residual completion-marker audit on Gate11Q outputs."""

import argparse
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

import run_gate11q_named_residual_completion_marker_surface_audit as gate11q
import run_gate9a_graph_gauge_consumer as gate9a


SCHEMA_VERSION = "gate11r_one_explicit_residual_completion_marker_audit_v1"
METHOD_ID = "gate11r_one_explicit_residual_completion_marker_audit_v1"

DEFAULT_REGISTRY = "one_explicit_residual_completion_marker_registry.jsonl"
DEFAULT_POLICY_COMPARE = "one_explicit_residual_completion_marker_policy_compare.csv"
DEFAULT_STATUS = "one_explicit_residual_completion_marker_status.json"
DEFAULT_REPORT = "gate11r_one_explicit_residual_completion_marker_read.md"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"

REQUIRED_GATE11Q_STATUS_KEYS = (
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
    "broader_trusted_tree_settlement_still_unearned_status",
    "operator_admission_still_denied_status",
    "retroactive_reinterpretation_forbidden_status",
    "bounded_marker_surface_rows_status",
    "same_source_binding_requirement_status",
    "bounded_read_prefix_requirement_status",
    "residual_completion_boundary_status",
    "named_residual_completion_marker_surface_status",
    "next_named_blocker",
)

REQUIRED_COMPLETION_SURFACES = (
    "one explicit admissible later-source presence marker",
    "one explicit later_source_id or later_frozen_run_id",
    "one later source and only one later source",
    "one declaration marker",
    "one candidate id",
    "one class",
    "one explicit host-failure sentence",
    "matched status, registry, and read surfaces",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Gate11R one explicit residual completion-marker audit from the frozen "
            "Gate11Q surface run without deciding residual completion itself."
        )
    )
    parser.add_argument("--gate11q-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_status_value(status_payload: Dict[str, Any], key: str) -> str:
    return str(status_payload.get(key, ""))


def source_is_incomplete(
    source_gate11q_manifest: Dict[str, Any], source_gate11q_status: Dict[str, Any], report_text: str
) -> bool:
    if not str(source_gate11q_manifest.get("run_id") or ""):
        return True
    if not report_text.strip():
        return True
    return any(key not in source_gate11q_status for key in REQUIRED_GATE11Q_STATUS_KEYS)


def extract_later_source_ids(report_text: str) -> List[str]:
    patterns = [
        re.compile(r"(?im)^\s*residual_completion_later_source_id\s*[:=]\s*([a-z0-9_./-]+)\s*$"),
        re.compile(r"(?im)^\s*residual_completion_later_frozen_run_id\s*[:=]\s*([a-z0-9_./-]+)\s*$"),
    ]
    values: List[str] = []
    for pattern in patterns:
        values.extend(match.group(1).strip().lower() for match in pattern.finditer(report_text))
    return values


def explicit_residual_completion_marker_present(report_text: str) -> bool:
    return bool(
        re.search(r"(?im)^\s*residual_completion_marker_status\s*[:=]\s*present\s*$", report_text)
    )


def extract_bounded_completion_surfaces(report_text: str) -> List[str]:
    pattern = re.compile(r"(?im)^\s*residual_completion_surface\s*[:=]\s*(.+?)\s*$")
    return [match.group(1).strip().lower() for match in pattern.finditer(report_text)]


def same_source_residual_completion_marker_binding_explicit(report_text: str) -> bool:
    same_source_status_completed = bool(
        re.search(r"(?im)^\s*residual_completion_same_source_status\s*[:=]\s*completed\s*$", report_text)
    )
    unique_later_source_ids = sorted(set(extract_later_source_ids(report_text)))
    return same_source_status_completed and len(unique_later_source_ids) == 1


def bounded_read_prefix_attached(report_text: str) -> bool:
    surfaces = set(extract_bounded_completion_surfaces(report_text))
    return all(phrase in surfaces for phrase in REQUIRED_COMPLETION_SURFACES)


def build_registry(
    source_gate11q_manifest: Dict[str, Any],
    status_payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return [
        {
            "source_gate11q_run_id": str(source_gate11q_manifest.get("run_id") or ""),
            "source_gate11q_code_git_commit": str(source_gate11q_manifest.get("code_git_commit") or ""),
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
            "broader_trusted_tree_settlement_still_unearned_status": str(status_payload["broader_trusted_tree_settlement_still_unearned_status"]),
            "operator_admission_still_denied_status": str(status_payload["operator_admission_still_denied_status"]),
            "retroactive_reinterpretation_forbidden_status": str(status_payload["retroactive_reinterpretation_forbidden_status"]),
            "explicit_residual_completion_marker_status": str(status_payload["explicit_residual_completion_marker_status"]),
            "residual_completion_marker_singularity_status": str(status_payload["residual_completion_marker_singularity_status"]),
            "same_source_residual_completion_marker_binding_status": str(status_payload["same_source_residual_completion_marker_binding_status"]),
            "bounded_read_prefix_attachment_status": str(status_payload["bounded_read_prefix_attachment_status"]),
            "residual_completion_marker_boundary_status": str(status_payload["residual_completion_marker_boundary_status"]),
            "one_explicit_residual_completion_marker_status": str(status_payload["one_explicit_residual_completion_marker_status"]),
            "next_named_blocker": str(status_payload["next_named_blocker"]),
        }
    ]


def build_policy_compare(registry_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "source_gate11q_run_id": str(row["source_gate11q_run_id"]),
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
            "broader_trusted_tree_settlement_still_unearned_status": str(row["broader_trusted_tree_settlement_still_unearned_status"]),
            "operator_admission_still_denied_status": str(row["operator_admission_still_denied_status"]),
            "retroactive_reinterpretation_forbidden_status": str(row["retroactive_reinterpretation_forbidden_status"]),
            "explicit_residual_completion_marker_status": str(row["explicit_residual_completion_marker_status"]),
            "residual_completion_marker_singularity_status": str(row["residual_completion_marker_singularity_status"]),
            "same_source_residual_completion_marker_binding_status": str(row["same_source_residual_completion_marker_binding_status"]),
            "bounded_read_prefix_attachment_status": str(row["bounded_read_prefix_attachment_status"]),
            "residual_completion_marker_boundary_status": str(row["residual_completion_marker_boundary_status"]),
            "one_explicit_residual_completion_marker_status": str(row["one_explicit_residual_completion_marker_status"]),
            "next_named_blocker": str(row["next_named_blocker"]),
        }
        for row in registry_rows
    ]


def build_status_payload(
    source_gate11q_manifest: Dict[str, Any],
    source_gate11q_status: Dict[str, Any],
    report_text: str,
) -> Dict[str, Any]:
    incomplete = source_is_incomplete(source_gate11q_manifest, source_gate11q_status, report_text)

    gate10_closeout_preservation_status = "preserved" if source_status_value(source_gate11q_status, "gate10_closeout_preservation_status") == "preserved" else "not_preserved"
    gate11a_absence_result_preservation_status = "preserved" if source_status_value(source_gate11q_status, "gate11a_absence_result_preservation_status") == "preserved" else "not_preserved"
    gate11c_declaration_surface_preservation_status = "preserved" if source_status_value(source_gate11q_status, "gate11c_declaration_surface_preservation_status") == "preserved" else "not_preserved"
    gate11d_not_yet_declared_state_preservation_status = "preserved" if source_status_value(source_gate11q_status, "gate11d_not_yet_declared_state_preservation_status") == "preserved" else "not_preserved"
    gate11e_path_defined_state_preservation_status = "preserved" if source_status_value(source_gate11q_status, "gate11e_path_defined_state_preservation_status") == "preserved" else "not_preserved"
    gate11f_not_yet_admissible_state_preservation_status = "preserved" if source_status_value(source_gate11q_status, "gate11f_not_yet_admissible_state_preservation_status") == "preserved" else "not_preserved"
    gate11g_naming_surface_preservation_status = "preserved" if source_status_value(source_gate11q_status, "gate11g_naming_surface_preservation_status") == "preserved" else "not_preserved"
    gate11h_not_yet_named_state_preservation_status = "preserved" if source_status_value(source_gate11q_status, "gate11h_not_yet_named_state_preservation_status") == "preserved" else "not_preserved"
    gate11i_path_defined_state_preservation_status = "preserved" if source_status_value(source_gate11q_status, "gate11i_path_defined_state_preservation_status") == "preserved" else "not_preserved"
    gate11j_not_yet_admissible_state_preservation_status = "preserved" if source_status_value(source_gate11q_status, "gate11j_not_yet_admissible_state_preservation_status") == "preserved" else "not_preserved"
    gate11k_not_yet_present_state_preservation_status = "preserved" if source_status_value(source_gate11q_status, "gate11k_not_yet_present_state_preservation_status") == "preserved" else "not_preserved"
    gate11l_path_defined_state_preservation_status = "preserved" if source_status_value(source_gate11q_status, "gate11l_path_defined_state_preservation_status") == "preserved" else "not_preserved"
    gate11m_not_yet_present_state_preservation_status = "preserved" if source_status_value(source_gate11q_status, "gate11m_not_yet_present_state_preservation_status") == "preserved" else "not_preserved"
    gate11n_residual_named_state_preservation_status = "preserved" if source_status_value(source_gate11q_status, "gate11n_residual_named_state_preservation_status") == "preserved" else "not_preserved"
    gate11o_path_defined_state_preservation_status = "preserved" if source_status_value(source_gate11q_status, "gate11o_path_defined_state_preservation_status") == "preserved" else "not_preserved"
    gate11p_not_yet_completed_state_preservation_status = "preserved" if source_status_value(source_gate11q_status, "gate11p_not_yet_completed_state_preservation_status") == "preserved" else "not_preserved"
    gate11q_surface_defined_state_preservation_status = "preserved" if source_status_value(source_gate11q_status, "named_residual_completion_marker_surface_status") == "surface_defined" else "not_preserved"
    broader_trusted_tree_settlement_still_unearned_status = "confirmed" if source_status_value(source_gate11q_status, "broader_trusted_tree_settlement_still_unearned_status") == "confirmed" else "not_confirmed"
    operator_admission_still_denied_status = "confirmed" if source_status_value(source_gate11q_status, "operator_admission_still_denied_status") == "confirmed" else "not_confirmed"
    retroactive_reinterpretation_forbidden_status = "confirmed" if source_status_value(source_gate11q_status, "retroactive_reinterpretation_forbidden_status") == "confirmed" else "not_confirmed"

    if incomplete:
        residual_completion_marker_boundary_status = "deferred"
    elif (
        broader_trusted_tree_settlement_still_unearned_status != "confirmed"
        or operator_admission_still_denied_status != "confirmed"
        or retroactive_reinterpretation_forbidden_status != "confirmed"
        or source_status_value(source_gate11q_status, "residual_completion_boundary_status") != "confirmed"
    ):
        residual_completion_marker_boundary_status = "denied"
    else:
        residual_completion_marker_boundary_status = "confirmed"

    later_source_ids = extract_later_source_ids(report_text)
    unique_later_source_ids = sorted(set(later_source_ids))
    marker_present = explicit_residual_completion_marker_present(report_text)
    binding_explicit = same_source_residual_completion_marker_binding_explicit(report_text)
    read_prefix_attached = bounded_read_prefix_attached(report_text)

    if incomplete:
        explicit_residual_completion_marker_status = "deferred"
    elif marker_present:
        explicit_residual_completion_marker_status = "present"
    else:
        explicit_residual_completion_marker_status = "absent"

    if incomplete:
        residual_completion_marker_singularity_status = "deferred"
    elif not unique_later_source_ids:
        residual_completion_marker_singularity_status = "none"
    elif len(unique_later_source_ids) == 1:
        residual_completion_marker_singularity_status = "single"
    else:
        residual_completion_marker_singularity_status = "multiple"

    if incomplete:
        same_source_residual_completion_marker_binding_status = "deferred"
    elif residual_completion_marker_singularity_status == "multiple":
        same_source_residual_completion_marker_binding_status = "deferred"
    elif binding_explicit:
        same_source_residual_completion_marker_binding_status = "explicit"
    else:
        same_source_residual_completion_marker_binding_status = "not_explicit"

    if incomplete:
        bounded_read_prefix_attachment_status = "deferred"
    elif residual_completion_marker_singularity_status == "multiple":
        bounded_read_prefix_attachment_status = "deferred"
    elif read_prefix_attached:
        bounded_read_prefix_attachment_status = "attached"
    else:
        bounded_read_prefix_attachment_status = "not_attached"

    if incomplete:
        one_explicit_residual_completion_marker_status = "deferred"
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
        or residual_completion_marker_boundary_status == "denied"
        or broader_trusted_tree_settlement_still_unearned_status != "confirmed"
        or operator_admission_still_denied_status != "confirmed"
        or retroactive_reinterpretation_forbidden_status != "confirmed"
    ):
        one_explicit_residual_completion_marker_status = "denied"
    elif (
        explicit_residual_completion_marker_status == "present"
        and residual_completion_marker_singularity_status == "single"
        and same_source_residual_completion_marker_binding_status == "explicit"
        and bounded_read_prefix_attachment_status == "attached"
        and residual_completion_marker_boundary_status == "confirmed"
    ):
        one_explicit_residual_completion_marker_status = "present"
    elif (
        residual_completion_marker_singularity_status == "multiple"
        or same_source_residual_completion_marker_binding_status == "deferred"
        or bounded_read_prefix_attachment_status == "deferred"
        or residual_completion_marker_boundary_status == "deferred"
    ):
        one_explicit_residual_completion_marker_status = "deferred"
    else:
        one_explicit_residual_completion_marker_status = "not_yet_present"

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
    elif broader_trusted_tree_settlement_still_unearned_status != "confirmed":
        next_named_blocker = "broader_trusted_tree_settlement_not_explicitly_unearned"
    elif operator_admission_still_denied_status != "confirmed":
        next_named_blocker = "operator_admission_not_denied"
    elif retroactive_reinterpretation_forbidden_status != "confirmed":
        next_named_blocker = "retroactive_reinterpretation_pressure"
    elif residual_completion_marker_boundary_status == "deferred":
        next_named_blocker = "controlling_source_incomplete"
    elif residual_completion_marker_boundary_status == "denied":
        next_named_blocker = "residual_completion_marker_boundary_not_intact"
    elif residual_completion_marker_singularity_status == "multiple":
        next_named_blocker = "multiple_candidate_markers"
    elif explicit_residual_completion_marker_status != "present":
        next_named_blocker = "no_explicit_residual_completion_marker"
    elif same_source_residual_completion_marker_binding_status != "explicit":
        next_named_blocker = "same_source_residual_completion_marker_binding_not_explicit"
    elif bounded_read_prefix_attachment_status != "attached":
        next_named_blocker = "bounded_read_prefix_not_attached"
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
        "broader_trusted_tree_settlement_still_unearned_status": broader_trusted_tree_settlement_still_unearned_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "retroactive_reinterpretation_forbidden_status": retroactive_reinterpretation_forbidden_status,
        "explicit_residual_completion_marker_status": explicit_residual_completion_marker_status,
        "residual_completion_marker_singularity_status": residual_completion_marker_singularity_status,
        "same_source_residual_completion_marker_binding_status": same_source_residual_completion_marker_binding_status,
        "bounded_read_prefix_attachment_status": bounded_read_prefix_attachment_status,
        "residual_completion_marker_boundary_status": residual_completion_marker_boundary_status,
        "one_explicit_residual_completion_marker_status": one_explicit_residual_completion_marker_status,
        "next_named_blocker": next_named_blocker,
    }


def build_report(
    run_id: str,
    source_gate11q_manifest: Dict[str, Any],
    policy_compare_rows: Sequence[Dict[str, Any]],
    status_payload: Dict[str, Any],
) -> str:
    lines = [
        "# Gate11R One Explicit Residual Completion-Marker Read",
        "",
        f"run_id: {run_id}",
        f"source_gate11q_run_id: {source_gate11q_manifest.get('run_id', '')}",
        f"source_gate11q_code_git_commit: {source_gate11q_manifest.get('code_git_commit', '')}",
        "",
        "## Discipline",
        "",
        "- Gate11R asks only whether one explicit residual completion marker now exists under the fixed Gate11Q surface",
        "- Gate11R audits marker existence only",
        "- Gate11R does not complete the residual",
        "- Gate11R does not admit a later source",
        "- Gate11R does not decide one-admissible-later-source explicit-presence judgment",
        "- Gate11R does not declare a bounded-line insufficiency candidate",
        "- Gate11R does not declare that explicit declaration already exists",
        "- Gate11R does not decide reopening eligibility",
        "- Gate11R does not reopen operator admission",
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
        "- broader trusted-tree settlement remains unearned",
        "- operator admission remains denied",
        "- no retroactive rewrite of Gate9, Gate10, Gate11A, Gate11B, Gate11C, Gate11D, Gate11E, Gate11F, Gate11G, Gate11H, Gate11I, Gate11J, Gate11K, Gate11L, Gate11M, Gate11N, Gate11O, Gate11P, or Gate11Q memory is permitted",
        "",
        "## Source Summary",
        "",
        "| source_gate11q_run_id | gate10_closeout_preservation_status | gate11a_absence_result_preservation_status | gate11c_declaration_surface_preservation_status | gate11d_not_yet_declared_state_preservation_status | gate11e_path_defined_state_preservation_status | gate11f_not_yet_admissible_state_preservation_status | gate11g_naming_surface_preservation_status | gate11h_not_yet_named_state_preservation_status | gate11i_path_defined_state_preservation_status | gate11j_not_yet_admissible_state_preservation_status | gate11k_not_yet_present_state_preservation_status | gate11l_path_defined_state_preservation_status | gate11m_not_yet_present_state_preservation_status | gate11n_residual_named_state_preservation_status | gate11o_path_defined_state_preservation_status | gate11p_not_yet_completed_state_preservation_status | gate11q_surface_defined_state_preservation_status | broader_trusted_tree_settlement_still_unearned_status | operator_admission_still_denied_status | retroactive_reinterpretation_forbidden_status | explicit_residual_completion_marker_status | residual_completion_marker_singularity_status | same_source_residual_completion_marker_binding_status | bounded_read_prefix_attachment_status | residual_completion_marker_boundary_status | one_explicit_residual_completion_marker_status | next_named_blocker |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in policy_compare_rows:
        lines.append("| " + " | ".join([
            str(row["source_gate11q_run_id"]),
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
            str(row["broader_trusted_tree_settlement_still_unearned_status"]),
            str(row["operator_admission_still_denied_status"]),
            str(row["retroactive_reinterpretation_forbidden_status"]),
            str(row["explicit_residual_completion_marker_status"]),
            str(row["residual_completion_marker_singularity_status"]),
            str(row["same_source_residual_completion_marker_binding_status"]),
            str(row["bounded_read_prefix_attachment_status"]),
            str(row["residual_completion_marker_boundary_status"]),
            str(row["one_explicit_residual_completion_marker_status"]),
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
        f"- broader_trusted_tree_settlement_still_unearned_status: `{status_payload['broader_trusted_tree_settlement_still_unearned_status']}`",
        f"- operator_admission_still_denied_status: `{status_payload['operator_admission_still_denied_status']}`",
        f"- retroactive_reinterpretation_forbidden_status: `{status_payload['retroactive_reinterpretation_forbidden_status']}`",
        f"- explicit_residual_completion_marker_status: `{status_payload['explicit_residual_completion_marker_status']}`",
        f"- residual_completion_marker_singularity_status: `{status_payload['residual_completion_marker_singularity_status']}`",
        f"- same_source_residual_completion_marker_binding_status: `{status_payload['same_source_residual_completion_marker_binding_status']}`",
        f"- bounded_read_prefix_attachment_status: `{status_payload['bounded_read_prefix_attachment_status']}`",
        f"- residual_completion_marker_boundary_status: `{status_payload['residual_completion_marker_boundary_status']}`",
        f"- one_explicit_residual_completion_marker_status: `{status_payload['one_explicit_residual_completion_marker_status']}`",
        f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
        "",
        "## Marker-Existence Slice",
        "",
        "- Gate11R asks only whether one explicit residual completion marker now exists under the fixed Gate11Q surface",
        "- it does not convert surface definition into marker existence",
        "- it does not convert path-definition prose, hypothetical examples, or generic read narrative into a marker",
        "- worker-side inference, synthesis, or multi-candidate resolution does not count",
        "",
        "## Judgment",
        "",
    ])

    outcome = status_payload["one_explicit_residual_completion_marker_status"]
    if outcome == "present":
        lines.append("- one explicit residual completion marker is now present under the fixed Gate11Q surface, without completing the residual itself")
    elif outcome == "not_yet_present":
        lines.append("- the fixed Gate11Q surface remains preserved, but no explicit residual completion marker is yet instantiated there")
    elif outcome == "denied":
        lines.append("- the attempted marker-existence judgment is denied because it would depend on shortcut, inflation, rewrite, graph-wide leap, or worker-side synthesis")
    else:
        lines.append("- the frozen source is incomplete or contradictory for a one explicit residual completion-marker judgment")
    if status_payload["next_named_blocker"]:
        lines.append(f"- next_named_blocker: `{status_payload['next_named_blocker']}`")
    lines.extend([
        "",
        "## Memory Hook",
        "",
        "- Gate11R does not complete the residual; it asks whether one explicit residual completion marker now exists under the fixed Gate11Q surface.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    source_gate11q_dir = Path(args.gate11q_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = out_dir.name

    source_gate11q_manifest = gate9a.read_json(source_gate11q_dir / gate11q.DEFAULT_MANIFEST)
    source_gate11q_status = gate9a.read_json(source_gate11q_dir / gate11q.DEFAULT_STATUS)
    source_gate11q_report = (source_gate11q_dir / gate11q.DEFAULT_REPORT).read_text(encoding="utf-8")

    status_payload = build_status_payload(source_gate11q_manifest, source_gate11q_status, source_gate11q_report)
    registry_rows = build_registry(source_gate11q_manifest, status_payload)
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
            "source_gate11q_run_id",
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
            "broader_trusted_tree_settlement_still_unearned_status",
            "operator_admission_still_denied_status",
            "retroactive_reinterpretation_forbidden_status",
            "explicit_residual_completion_marker_status",
            "residual_completion_marker_singularity_status",
            "same_source_residual_completion_marker_binding_status",
            "bounded_read_prefix_attachment_status",
            "residual_completion_marker_boundary_status",
            "one_explicit_residual_completion_marker_status",
            "next_named_blocker",
        ),
        policy_compare_rows,
    )
    gate9a.write_json(status_path, status_payload)
    gate9a.write_text(report_path, build_report(run_id, source_gate11q_manifest, policy_compare_rows, status_payload))

    manifest = {
        "run_id": run_id,
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "code_git_commit": gate9a.current_git_commit(),
        "source_gate11q_run_id": str(source_gate11q_manifest.get("run_id") or ""),
        "source_gate11q_code_git_commit": str(source_gate11q_manifest.get("code_git_commit") or ""),
        "inputs": {"gate11q_dir": str(source_gate11q_dir)},
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