#!/usr/bin/env python3
"""Run a Gate11H one later-source explicit-naming audit on Gate11G outputs."""

import argparse
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

import run_gate11g_later_source_naming_surface_audit as gate11g
import run_gate9a_graph_gauge_consumer as gate9a


SCHEMA_VERSION = "gate11h_one_later_source_explicit_naming_audit_v1"
METHOD_ID = "gate11h_one_later_source_explicit_naming_audit_v1"

DEFAULT_REGISTRY = "one_later_source_explicit_naming_registry.jsonl"
DEFAULT_POLICY_COMPARE = "one_later_source_explicit_naming_policy_compare.csv"
DEFAULT_STATUS = "one_later_source_explicit_naming_status.json"
DEFAULT_REPORT = "gate11h_one_later_source_explicit_naming_read.md"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"

REQUIRED_GATE11G_STATUS_KEYS = (
    "gate10_closeout_preservation_status",
    "gate11a_absence_result_preservation_status",
    "gate11c_declaration_surface_preservation_status",
    "gate11d_not_yet_declared_state_preservation_status",
    "gate11e_path_defined_state_preservation_status",
    "gate11f_not_yet_admissible_state_preservation_status",
    "broader_trusted_tree_settlement_still_unearned_status",
    "operator_admission_still_denied_status",
    "retroactive_reinterpretation_forbidden_status",
    "explicit_later_source_marker_shape_status",
    "single_later_source_singularity_status",
    "full_path_attachment_shape_status",
    "anti_shortcut_boundary_status",
    "later_source_naming_surface_status",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Gate11H one later-source explicit-naming audit from the frozen Gate11G "
            "naming-surface run without deciding later-source admissibility."
        )
    )
    parser.add_argument("--gate11g-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_status_value(status_payload: Dict[str, Any], key: str) -> str:
    return str(status_payload.get(key, ""))


def source_is_incomplete(
    source_gate11g_manifest: Dict[str, Any], source_gate11g_status: Dict[str, Any], report_text: str
) -> bool:
    if not str(source_gate11g_manifest.get("run_id") or ""):
        return True
    if not report_text.strip():
        return True
    return any(not source_status_value(source_gate11g_status, key) for key in REQUIRED_GATE11G_STATUS_KEYS)


def extract_later_source_ids(report_text: str) -> List[str]:
    patterns = [
        re.compile(r"(?im)^\s*later_source_id\s*[:=]\s*([a-z0-9_./-]+)\s*$"),
        re.compile(r"(?im)^\s*later_frozen_run_id\s*[:=]\s*([a-z0-9_./-]+)\s*$"),
        re.compile(r"(?im)^\s*(?:[-*]\s*)?one later source is explicitly named:\s*([a-z0-9_./-]+)\s*$"),
        re.compile(r"(?im)^\s*(?:[-*]\s*)?one later frozen run is explicitly named:\s*([a-z0-9_./-]+)\s*$"),
    ]
    values: List[str] = []
    for pattern in patterns:
        values.extend(match.group(1).strip().lower() for match in pattern.finditer(report_text))
    return values


def later_source_path_attached(report_text: str) -> bool:
    required_phrases = (
        "one declaration marker",
        "one candidate id",
        "one class",
        "one explicit host-failure sentence",
        "matched status, registry, and read surfaces",
    )
    lowered = report_text.lower()
    return all(phrase in lowered for phrase in required_phrases)


def build_registry(
    source_gate11g_manifest: Dict[str, Any],
    source_gate11g_status: Dict[str, Any],
    status_payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return [
        {
            "source_gate11g_run_id": str(source_gate11g_manifest.get("run_id") or ""),
            "source_gate11g_code_git_commit": str(
                source_gate11g_manifest.get("code_git_commit") or ""
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
            "broader_trusted_tree_settlement_still_unearned_status": str(
                status_payload["broader_trusted_tree_settlement_still_unearned_status"]
            ),
            "operator_admission_still_denied_status": str(
                status_payload["operator_admission_still_denied_status"]
            ),
            "retroactive_reinterpretation_forbidden_status": str(
                status_payload["retroactive_reinterpretation_forbidden_status"]
            ),
            "explicit_later_source_marker_status": str(
                status_payload["explicit_later_source_marker_status"]
            ),
            "later_source_singularity_status": str(
                status_payload["later_source_singularity_status"]
            ),
            "full_path_attachment_status": str(status_payload["full_path_attachment_status"]),
            "anti_shortcut_boundary_status": str(status_payload["anti_shortcut_boundary_status"]),
            "one_later_source_explicit_naming_status": str(
                status_payload["one_later_source_explicit_naming_status"]
            ),
        }
    ]


def build_policy_compare(registry_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "source_gate11g_run_id": str(row["source_gate11g_run_id"]),
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
            "broader_trusted_tree_settlement_still_unearned_status": str(
                row["broader_trusted_tree_settlement_still_unearned_status"]
            ),
            "operator_admission_still_denied_status": str(row["operator_admission_still_denied_status"]),
            "retroactive_reinterpretation_forbidden_status": str(
                row["retroactive_reinterpretation_forbidden_status"]
            ),
            "explicit_later_source_marker_status": str(row["explicit_later_source_marker_status"]),
            "later_source_singularity_status": str(row["later_source_singularity_status"]),
            "full_path_attachment_status": str(row["full_path_attachment_status"]),
            "anti_shortcut_boundary_status": str(row["anti_shortcut_boundary_status"]),
            "one_later_source_explicit_naming_status": str(
                row["one_later_source_explicit_naming_status"]
            ),
        }
        for row in registry_rows
    ]


def build_status_payload(
    source_gate11g_manifest: Dict[str, Any],
    source_gate11g_status: Dict[str, Any],
    report_text: str,
) -> Dict[str, Any]:
    incomplete = source_is_incomplete(source_gate11g_manifest, source_gate11g_status, report_text)

    gate10_closeout_preservation_status = (
        "preserved"
        if source_status_value(source_gate11g_status, "gate10_closeout_preservation_status") == "preserved"
        else "not_preserved"
    )
    gate11a_absence_result_preservation_status = (
        "preserved"
        if source_status_value(source_gate11g_status, "gate11a_absence_result_preservation_status")
        == "preserved"
        else "not_preserved"
    )
    gate11c_declaration_surface_preservation_status = (
        "preserved"
        if source_status_value(
            source_gate11g_status, "gate11c_declaration_surface_preservation_status"
        )
        == "preserved"
        else "not_preserved"
    )
    gate11d_not_yet_declared_state_preservation_status = (
        "preserved"
        if source_status_value(
            source_gate11g_status, "gate11d_not_yet_declared_state_preservation_status"
        )
        == "preserved"
        else "not_preserved"
    )
    gate11e_path_defined_state_preservation_status = (
        "preserved"
        if source_status_value(
            source_gate11g_status, "gate11e_path_defined_state_preservation_status"
        )
        == "preserved"
        else "not_preserved"
    )
    gate11f_not_yet_admissible_state_preservation_status = (
        "preserved"
        if source_status_value(
            source_gate11g_status, "gate11f_not_yet_admissible_state_preservation_status"
        )
        == "preserved"
        else "not_preserved"
    )
    gate11g_naming_surface_preservation_status = (
        "preserved"
        if source_status_value(source_gate11g_status, "later_source_naming_surface_status")
        == "surface_defined"
        else "not_preserved"
    )
    broader_trusted_tree_settlement_still_unearned_status = (
        "confirmed"
        if source_status_value(
            source_gate11g_status, "broader_trusted_tree_settlement_still_unearned_status"
        )
        == "confirmed"
        else "not_confirmed"
    )
    operator_admission_still_denied_status = (
        "confirmed"
        if source_status_value(source_gate11g_status, "operator_admission_still_denied_status")
        == "confirmed"
        else "not_confirmed"
    )
    retroactive_reinterpretation_forbidden_status = (
        "confirmed"
        if source_status_value(
            source_gate11g_status, "retroactive_reinterpretation_forbidden_status"
        )
        == "confirmed"
        else "not_confirmed"
    )

    if incomplete:
        anti_shortcut_boundary_status = "deferred"
    elif (
        broader_trusted_tree_settlement_still_unearned_status != "confirmed"
        or operator_admission_still_denied_status != "confirmed"
        or retroactive_reinterpretation_forbidden_status != "confirmed"
        or source_status_value(source_gate11g_status, "anti_shortcut_boundary_status") != "confirmed"
    ):
        anti_shortcut_boundary_status = "denied"
    else:
        anti_shortcut_boundary_status = "confirmed"

    later_source_ids = extract_later_source_ids(report_text)
    unique_later_source_ids = sorted(set(later_source_ids))

    if incomplete:
        explicit_later_source_marker_status = "deferred"
    elif not later_source_ids:
        explicit_later_source_marker_status = "absent"
    else:
        explicit_later_source_marker_status = "present"

    if incomplete:
        later_source_singularity_status = "deferred"
    elif not unique_later_source_ids:
        later_source_singularity_status = "none"
    elif len(unique_later_source_ids) == 1:
        later_source_singularity_status = "single"
    else:
        later_source_singularity_status = "multiple"

    if incomplete:
        full_path_attachment_status = "deferred"
    elif later_source_singularity_status == "multiple":
        full_path_attachment_status = "deferred"
    elif later_source_singularity_status != "single":
        full_path_attachment_status = "not_attached"
    elif later_source_path_attached(report_text):
        full_path_attachment_status = "attached"
    else:
        full_path_attachment_status = "not_attached"

    if incomplete:
        one_later_source_explicit_naming_status = "deferred"
    elif (
        gate10_closeout_preservation_status != "preserved"
        or gate11a_absence_result_preservation_status != "preserved"
        or gate11c_declaration_surface_preservation_status != "preserved"
        or gate11d_not_yet_declared_state_preservation_status != "preserved"
        or gate11e_path_defined_state_preservation_status != "preserved"
        or gate11f_not_yet_admissible_state_preservation_status != "preserved"
        or gate11g_naming_surface_preservation_status != "preserved"
        or broader_trusted_tree_settlement_still_unearned_status != "confirmed"
        or operator_admission_still_denied_status != "confirmed"
        or retroactive_reinterpretation_forbidden_status != "confirmed"
        or anti_shortcut_boundary_status == "denied"
    ):
        one_later_source_explicit_naming_status = "denied"
    elif (
        explicit_later_source_marker_status == "present"
        and later_source_singularity_status == "single"
        and full_path_attachment_status == "attached"
        and anti_shortcut_boundary_status == "confirmed"
    ):
        one_later_source_explicit_naming_status = "named"
    elif (
        explicit_later_source_marker_status == "deferred"
        or later_source_singularity_status == "deferred"
        or full_path_attachment_status == "deferred"
        or anti_shortcut_boundary_status == "deferred"
    ):
        one_later_source_explicit_naming_status = "deferred"
    else:
        one_later_source_explicit_naming_status = "not_yet_named"

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
    elif later_source_singularity_status == "multiple":
        next_named_blocker = "multiple_later_sources"
    elif explicit_later_source_marker_status == "absent":
        next_named_blocker = "no_explicit_later_source_marker"
    elif full_path_attachment_status == "not_attached":
        next_named_blocker = "full_path_not_attached_to_later_source"
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
        "broader_trusted_tree_settlement_still_unearned_status": broader_trusted_tree_settlement_still_unearned_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "retroactive_reinterpretation_forbidden_status": retroactive_reinterpretation_forbidden_status,
        "explicit_later_source_marker_status": explicit_later_source_marker_status,
        "later_source_singularity_status": later_source_singularity_status,
        "full_path_attachment_status": full_path_attachment_status,
        "anti_shortcut_boundary_status": anti_shortcut_boundary_status,
        "one_later_source_explicit_naming_status": one_later_source_explicit_naming_status,
        "next_named_blocker": next_named_blocker,
    }


def build_report(
    run_id: str,
    source_gate11g_manifest: Dict[str, Any],
    policy_compare_rows: Sequence[Dict[str, Any]],
    status_payload: Dict[str, Any],
) -> str:
    lines = [
        "# Gate11H One Later-Source Explicit-Naming Read",
        "",
        f"run_id: {run_id}",
        f"source_gate11g_run_id: {source_gate11g_manifest.get('run_id', '')}",
        f"source_gate11g_code_git_commit: {source_gate11g_manifest.get('code_git_commit', '')}",
        "",
        "## Discipline",
        "",
        "- Gate11H asks only whether one explicit later-source naming instance now exists under the fixed Gate11G surface",
        "- Gate11H does not decide later-source admissibility",
        "- Gate11H does not declare a candidate",
        "- Gate11H does not declare that explicit declaration already exists",
        "- Gate11H does not decide reopening eligibility",
        "- Gate10 closeout remains bounded and preserved",
        "- Gate11A absence remains preserved",
        "- Gate11C surface-defined result remains preserved",
        "- Gate11D not-yet-declared result remains preserved",
        "- Gate11E path-defined result remains preserved",
        "- Gate11F not-yet-admissible result remains preserved",
        "- Gate11G surface-defined result remains preserved",
        "- broader trusted-tree settlement remains unearned",
        "- operator admission remains denied",
        "- no retroactive rewrite of Gate9, Gate10, Gate11A, Gate11B, Gate11C, Gate11D, Gate11E, Gate11F, or Gate11G memory is permitted",
        "",
        "## Source Summary",
        "",
        "| source_gate11g_run_id | gate10_closeout_preservation_status | gate11a_absence_result_preservation_status | gate11c_declaration_surface_preservation_status | gate11d_not_yet_declared_state_preservation_status | gate11e_path_defined_state_preservation_status | gate11f_not_yet_admissible_state_preservation_status | gate11g_naming_surface_preservation_status | broader_trusted_tree_settlement_still_unearned_status | operator_admission_still_denied_status | retroactive_reinterpretation_forbidden_status | explicit_later_source_marker_status | later_source_singularity_status | full_path_attachment_status | anti_shortcut_boundary_status | one_later_source_explicit_naming_status |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in policy_compare_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["source_gate11g_run_id"]),
                    str(row["gate10_closeout_preservation_status"]),
                    str(row["gate11a_absence_result_preservation_status"]),
                    str(row["gate11c_declaration_surface_preservation_status"]),
                    str(row["gate11d_not_yet_declared_state_preservation_status"]),
                    str(row["gate11e_path_defined_state_preservation_status"]),
                    str(row["gate11f_not_yet_admissible_state_preservation_status"]),
                    str(row["gate11g_naming_surface_preservation_status"]),
                    str(row["broader_trusted_tree_settlement_still_unearned_status"]),
                    str(row["operator_admission_still_denied_status"]),
                    str(row["retroactive_reinterpretation_forbidden_status"]),
                    str(row["explicit_later_source_marker_status"]),
                    str(row["later_source_singularity_status"]),
                    str(row["full_path_attachment_status"]),
                    str(row["anti_shortcut_boundary_status"]),
                    str(row["one_later_source_explicit_naming_status"]),
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
            f"- broader_trusted_tree_settlement_still_unearned_status: `{status_payload['broader_trusted_tree_settlement_still_unearned_status']}`",
            f"- operator_admission_still_denied_status: `{status_payload['operator_admission_still_denied_status']}`",
            f"- retroactive_reinterpretation_forbidden_status: `{status_payload['retroactive_reinterpretation_forbidden_status']}`",
            f"- explicit_later_source_marker_status: `{status_payload['explicit_later_source_marker_status']}`",
            f"- later_source_singularity_status: `{status_payload['later_source_singularity_status']}`",
            f"- full_path_attachment_status: `{status_payload['full_path_attachment_status']}`",
            f"- anti_shortcut_boundary_status: `{status_payload['anti_shortcut_boundary_status']}`",
            f"- one_later_source_explicit_naming_status: `{status_payload['one_later_source_explicit_naming_status']}`",
            f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
        ]
    )

    if status_payload["one_later_source_explicit_naming_status"] == "named":
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- one later source is now explicitly named under the fixed Gate11G surface",
                "- Gate11H does not decide later-source admissibility here",
            ]
        )
    elif status_payload["one_later_source_explicit_naming_status"] == "deferred":
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- the one-later-source explicit-naming audit remains deferred because the frozen source is incomplete or contradictory",
                f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
            ]
        )
    elif status_payload["one_later_source_explicit_naming_status"] == "denied":
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- the proposed later-source naming instance is denied under the frozen Gate11H boundary",
                f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- the fixed Gate11G naming surface exists, but no one explicit later-source naming instance yet exists under that surface",
                f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    source_gate11g_dir = Path(args.gate11g_dir)
    out_dir = Path(args.out_dir)
    run_id = out_dir.name

    out_dir.mkdir(parents=True, exist_ok=True)

    source_gate11g_manifest = gate9a.read_json(source_gate11g_dir / gate11g.DEFAULT_MANIFEST)
    source_gate11g_status = gate9a.read_json(source_gate11g_dir / gate11g.DEFAULT_STATUS)
    source_gate11g_report = (source_gate11g_dir / gate11g.DEFAULT_REPORT).read_text(encoding="utf-8")

    status_payload = build_status_payload(
        source_gate11g_manifest,
        source_gate11g_status,
        source_gate11g_report,
    )
    registry_rows = build_registry(source_gate11g_manifest, source_gate11g_status, status_payload)
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
            "source_gate11g_run_id",
            "gate10_closeout_preservation_status",
            "gate11a_absence_result_preservation_status",
            "gate11c_declaration_surface_preservation_status",
            "gate11d_not_yet_declared_state_preservation_status",
            "gate11e_path_defined_state_preservation_status",
            "gate11f_not_yet_admissible_state_preservation_status",
            "gate11g_naming_surface_preservation_status",
            "broader_trusted_tree_settlement_still_unearned_status",
            "operator_admission_still_denied_status",
            "retroactive_reinterpretation_forbidden_status",
            "explicit_later_source_marker_status",
            "later_source_singularity_status",
            "full_path_attachment_status",
            "anti_shortcut_boundary_status",
            "one_later_source_explicit_naming_status",
        ),
        policy_compare_rows,
    )
    gate9a.write_json(status_path, status_payload)
    gate9a.write_text(
        report_path,
        build_report(
            run_id=run_id,
            source_gate11g_manifest=source_gate11g_manifest,
            policy_compare_rows=policy_compare_rows,
            status_payload=status_payload,
        ),
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "run_id": run_id,
        "code_git_commit": gate9a.current_git_commit(),
        "source_gate11g_dir": gate9a.repo_relative_or_posix(source_gate11g_dir),
        "source_gate11g_run_id": str(source_gate11g_manifest.get("run_id") or ""),
        "source_gate11g_code_git_commit": str(
            source_gate11g_manifest.get("code_git_commit") or ""
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