#!/usr/bin/env python3
"""Run a Gate11I later-source explicit-naming instantiation-path audit on Gate11H outputs."""

import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Sequence

import run_gate11h_one_later_source_explicit_naming_audit as gate11h
import run_gate9a_graph_gauge_consumer as gate9a


SCHEMA_VERSION = "gate11i_later_source_explicit_naming_instantiation_path_audit_v1"
METHOD_ID = "gate11i_later_source_explicit_naming_instantiation_path_audit_v1"

DEFAULT_REGISTRY = "later_source_explicit_naming_instantiation_path_registry.jsonl"
DEFAULT_POLICY_COMPARE = "later_source_explicit_naming_instantiation_path_policy_compare.csv"
DEFAULT_STATUS = "later_source_explicit_naming_instantiation_path_status.json"
DEFAULT_REPORT = "gate11i_later_source_explicit_naming_instantiation_path_read.md"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"

REQUIRED_GATE11H_STATUS_KEYS = (
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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Gate11I later-source explicit-naming instantiation-path audit from the "
            "frozen Gate11H explicit-naming run without deciding later-source admissibility."
        )
    )
    parser.add_argument("--gate11h-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_status_value(status_payload: Dict[str, Any], key: str) -> str:
    return str(status_payload.get(key, ""))


def source_is_incomplete(
    source_gate11h_manifest: Dict[str, Any], source_gate11h_status: Dict[str, Any], report_text: str
) -> bool:
    if not str(source_gate11h_manifest.get("run_id") or ""):
        return True
    if not report_text.strip():
        return True
    return any(not source_status_value(source_gate11h_status, key) for key in REQUIRED_GATE11H_STATUS_KEYS)


def build_registry(
    source_gate11h_manifest: Dict[str, Any],
    source_gate11h_status: Dict[str, Any],
    status_payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return [
        {
            "source_gate11h_run_id": str(source_gate11h_manifest.get("run_id") or ""),
            "source_gate11h_code_git_commit": str(
                source_gate11h_manifest.get("code_git_commit") or ""
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
            "broader_trusted_tree_settlement_still_unearned_status": str(
                status_payload["broader_trusted_tree_settlement_still_unearned_status"]
            ),
            "operator_admission_still_denied_status": str(
                status_payload["operator_admission_still_denied_status"]
            ),
            "retroactive_reinterpretation_forbidden_status": str(
                status_payload["retroactive_reinterpretation_forbidden_status"]
            ),
            "missing_naming_component_naming_status": str(
                status_payload["missing_naming_component_naming_status"]
            ),
            "minimal_same_source_later_source_instantiation_rule_status": str(
                status_payload["minimal_same_source_later_source_instantiation_rule_status"]
            ),
            "anti_shortcut_boundary_status": str(
                status_payload["anti_shortcut_boundary_status"]
            ),
            "later_source_explicit_naming_instantiation_path_status": str(
                status_payload["later_source_explicit_naming_instantiation_path_status"]
            ),
            "source_explicit_later_source_marker_status": source_status_value(
                source_gate11h_status,
                "explicit_later_source_marker_status",
            ),
            "source_later_source_singularity_status": source_status_value(
                source_gate11h_status,
                "later_source_singularity_status",
            ),
            "source_full_path_attachment_status": source_status_value(
                source_gate11h_status,
                "full_path_attachment_status",
            ),
        }
    ]


def build_policy_compare(registry_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "source_gate11h_run_id": str(row["source_gate11h_run_id"]),
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
            "broader_trusted_tree_settlement_still_unearned_status": str(
                row["broader_trusted_tree_settlement_still_unearned_status"]
            ),
            "operator_admission_still_denied_status": str(row["operator_admission_still_denied_status"]),
            "retroactive_reinterpretation_forbidden_status": str(
                row["retroactive_reinterpretation_forbidden_status"]
            ),
            "missing_naming_component_naming_status": str(
                row["missing_naming_component_naming_status"]
            ),
            "minimal_same_source_later_source_instantiation_rule_status": str(
                row["minimal_same_source_later_source_instantiation_rule_status"]
            ),
            "anti_shortcut_boundary_status": str(row["anti_shortcut_boundary_status"]),
            "later_source_explicit_naming_instantiation_path_status": str(
                row["later_source_explicit_naming_instantiation_path_status"]
            ),
        }
        for row in registry_rows
    ]


def build_status_payload(
    source_gate11h_manifest: Dict[str, Any],
    source_gate11h_status: Dict[str, Any],
    report_text: str,
) -> Dict[str, Any]:
    incomplete = source_is_incomplete(source_gate11h_manifest, source_gate11h_status, report_text)

    gate10_closeout_preservation_status = (
        "preserved"
        if source_status_value(source_gate11h_status, "gate10_closeout_preservation_status") == "preserved"
        else "not_preserved"
    )
    gate11a_absence_result_preservation_status = (
        "preserved"
        if source_status_value(source_gate11h_status, "gate11a_absence_result_preservation_status")
        == "preserved"
        else "not_preserved"
    )
    gate11c_declaration_surface_preservation_status = (
        "preserved"
        if source_status_value(
            source_gate11h_status, "gate11c_declaration_surface_preservation_status"
        )
        == "preserved"
        else "not_preserved"
    )
    gate11d_not_yet_declared_state_preservation_status = (
        "preserved"
        if source_status_value(
            source_gate11h_status, "gate11d_not_yet_declared_state_preservation_status"
        )
        == "preserved"
        else "not_preserved"
    )
    gate11e_path_defined_state_preservation_status = (
        "preserved"
        if source_status_value(
            source_gate11h_status, "gate11e_path_defined_state_preservation_status"
        )
        == "preserved"
        else "not_preserved"
    )
    gate11f_not_yet_admissible_state_preservation_status = (
        "preserved"
        if source_status_value(
            source_gate11h_status, "gate11f_not_yet_admissible_state_preservation_status"
        )
        == "preserved"
        else "not_preserved"
    )
    gate11g_naming_surface_preservation_status = (
        "preserved"
        if source_status_value(source_gate11h_status, "gate11g_naming_surface_preservation_status")
        == "preserved"
        else "not_preserved"
    )
    gate11h_not_yet_named_state_preservation_status = (
        "preserved"
        if source_status_value(
            source_gate11h_status, "one_later_source_explicit_naming_status"
        )
        == "not_yet_named"
        else "not_preserved"
    )
    broader_trusted_tree_settlement_still_unearned_status = (
        "confirmed"
        if source_status_value(
            source_gate11h_status, "broader_trusted_tree_settlement_still_unearned_status"
        )
        == "confirmed"
        else "not_confirmed"
    )
    operator_admission_still_denied_status = (
        "confirmed"
        if source_status_value(source_gate11h_status, "operator_admission_still_denied_status")
        == "confirmed"
        else "not_confirmed"
    )
    retroactive_reinterpretation_forbidden_status = (
        "confirmed"
        if source_status_value(
            source_gate11h_status, "retroactive_reinterpretation_forbidden_status"
        )
        == "confirmed"
        else "not_confirmed"
    )

    if incomplete:
        missing_naming_component_naming_status = "deferred"
    else:
        marker_absent = source_status_value(
            source_gate11h_status, "explicit_later_source_marker_status"
        ) == "absent"
        singularity_none = source_status_value(
            source_gate11h_status, "later_source_singularity_status"
        ) == "none"
        full_path_not_attached = source_status_value(
            source_gate11h_status, "full_path_attachment_status"
        ) == "not_attached"
        if marker_absent and singularity_none and full_path_not_attached:
            missing_naming_component_naming_status = "named"
        else:
            missing_naming_component_naming_status = "not_named"

    if incomplete:
        anti_shortcut_boundary_status = "deferred"
    elif (
        broader_trusted_tree_settlement_still_unearned_status != "confirmed"
        or operator_admission_still_denied_status != "confirmed"
        or retroactive_reinterpretation_forbidden_status != "confirmed"
        or source_status_value(source_gate11h_status, "anti_shortcut_boundary_status") != "confirmed"
    ):
        anti_shortcut_boundary_status = "denied"
    else:
        anti_shortcut_boundary_status = "confirmed"

    if incomplete:
        minimal_same_source_later_source_instantiation_rule_status = "deferred"
    elif anti_shortcut_boundary_status == "denied":
        minimal_same_source_later_source_instantiation_rule_status = "not_defined"
    elif (
        gate11g_naming_surface_preservation_status != "preserved"
        or gate11h_not_yet_named_state_preservation_status != "preserved"
    ):
        minimal_same_source_later_source_instantiation_rule_status = "not_defined"
    else:
        minimal_same_source_later_source_instantiation_rule_status = "defined"

    if incomplete:
        later_source_explicit_naming_instantiation_path_status = "deferred"
    elif (
        gate10_closeout_preservation_status != "preserved"
        or gate11a_absence_result_preservation_status != "preserved"
        or gate11c_declaration_surface_preservation_status != "preserved"
        or gate11d_not_yet_declared_state_preservation_status != "preserved"
        or gate11e_path_defined_state_preservation_status != "preserved"
        or gate11f_not_yet_admissible_state_preservation_status != "preserved"
        or gate11g_naming_surface_preservation_status != "preserved"
        or gate11h_not_yet_named_state_preservation_status != "preserved"
        or anti_shortcut_boundary_status == "denied"
        or broader_trusted_tree_settlement_still_unearned_status != "confirmed"
        or operator_admission_still_denied_status != "confirmed"
        or retroactive_reinterpretation_forbidden_status != "confirmed"
    ):
        later_source_explicit_naming_instantiation_path_status = "denied"
    elif (
        missing_naming_component_naming_status == "named"
        and minimal_same_source_later_source_instantiation_rule_status == "defined"
        and anti_shortcut_boundary_status == "confirmed"
    ):
        later_source_explicit_naming_instantiation_path_status = "path_defined"
    elif (
        missing_naming_component_naming_status == "deferred"
        or minimal_same_source_later_source_instantiation_rule_status == "deferred"
        or anti_shortcut_boundary_status == "deferred"
    ):
        later_source_explicit_naming_instantiation_path_status = "deferred"
    else:
        later_source_explicit_naming_instantiation_path_status = "not_yet_defined"

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
    elif missing_naming_component_naming_status == "not_named":
        next_named_blocker = "missing_naming_components_not_explicitly_named"
    elif minimal_same_source_later_source_instantiation_rule_status == "not_defined":
        next_named_blocker = "same_source_later_source_instantiation_rule_not_defined"
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
        "broader_trusted_tree_settlement_still_unearned_status": broader_trusted_tree_settlement_still_unearned_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "retroactive_reinterpretation_forbidden_status": retroactive_reinterpretation_forbidden_status,
        "missing_naming_component_naming_status": missing_naming_component_naming_status,
        "minimal_same_source_later_source_instantiation_rule_status": minimal_same_source_later_source_instantiation_rule_status,
        "anti_shortcut_boundary_status": anti_shortcut_boundary_status,
        "later_source_explicit_naming_instantiation_path_status": later_source_explicit_naming_instantiation_path_status,
        "next_named_blocker": next_named_blocker,
    }


def build_report(
    run_id: str,
    source_gate11h_manifest: Dict[str, Any],
    policy_compare_rows: Sequence[Dict[str, Any]],
    status_payload: Dict[str, Any],
) -> str:
    lines = [
        "# Gate11I Later-Source Explicit-Naming Instantiation Path Read",
        "",
        f"run_id: {run_id}",
        f"source_gate11h_run_id: {source_gate11h_manifest.get('run_id', '')}",
        f"source_gate11h_code_git_commit: {source_gate11h_manifest.get('code_git_commit', '')}",
        "",
        "## Discipline",
        "",
        "- Gate11I asks only what the minimum admissible path would be from surface_defined but not_yet_named to one future honest later-source explicit-naming instance",
        "- Gate11I does not admit a later source",
        "- Gate11I does not create a naming instance",
        "- Gate11I does not declare a candidate",
        "- Gate11I does not declare that explicit declaration already exists",
        "- Gate11I does not decide reopening eligibility",
        "- Gate10 closeout remains bounded and preserved",
        "- Gate11A absence remains preserved",
        "- Gate11C surface-defined result remains preserved",
        "- Gate11D not-yet-declared result remains preserved",
        "- Gate11E path-defined result remains preserved",
        "- Gate11F not-yet-admissible result remains preserved",
        "- Gate11G surface-defined result remains preserved",
        "- Gate11H not-yet-named result remains preserved",
        "- broader trusted-tree settlement remains unearned",
        "- operator admission remains denied",
        "- no retroactive rewrite of Gate9, Gate10, Gate11A, Gate11B, Gate11C, Gate11D, Gate11E, Gate11F, Gate11G, or Gate11H memory is permitted",
        "",
        "## Source Summary",
        "",
        "| source_gate11h_run_id | gate10_closeout_preservation_status | gate11a_absence_result_preservation_status | gate11c_declaration_surface_preservation_status | gate11d_not_yet_declared_state_preservation_status | gate11e_path_defined_state_preservation_status | gate11f_not_yet_admissible_state_preservation_status | gate11g_naming_surface_preservation_status | gate11h_not_yet_named_state_preservation_status | broader_trusted_tree_settlement_still_unearned_status | operator_admission_still_denied_status | retroactive_reinterpretation_forbidden_status | missing_naming_component_naming_status | minimal_same_source_later_source_instantiation_rule_status | anti_shortcut_boundary_status | later_source_explicit_naming_instantiation_path_status |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in policy_compare_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["source_gate11h_run_id"]),
                    str(row["gate10_closeout_preservation_status"]),
                    str(row["gate11a_absence_result_preservation_status"]),
                    str(row["gate11c_declaration_surface_preservation_status"]),
                    str(row["gate11d_not_yet_declared_state_preservation_status"]),
                    str(row["gate11e_path_defined_state_preservation_status"]),
                    str(row["gate11f_not_yet_admissible_state_preservation_status"]),
                    str(row["gate11g_naming_surface_preservation_status"]),
                    str(row["gate11h_not_yet_named_state_preservation_status"]),
                    str(row["broader_trusted_tree_settlement_still_unearned_status"]),
                    str(row["operator_admission_still_denied_status"]),
                    str(row["retroactive_reinterpretation_forbidden_status"]),
                    str(row["missing_naming_component_naming_status"]),
                    str(row["minimal_same_source_later_source_instantiation_rule_status"]),
                    str(row["anti_shortcut_boundary_status"]),
                    str(row["later_source_explicit_naming_instantiation_path_status"]),
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
            f"- broader_trusted_tree_settlement_still_unearned_status: `{status_payload['broader_trusted_tree_settlement_still_unearned_status']}`",
            f"- operator_admission_still_denied_status: `{status_payload['operator_admission_still_denied_status']}`",
            f"- retroactive_reinterpretation_forbidden_status: `{status_payload['retroactive_reinterpretation_forbidden_status']}`",
            f"- missing_naming_component_naming_status: `{status_payload['missing_naming_component_naming_status']}`",
            f"- minimal_same_source_later_source_instantiation_rule_status: `{status_payload['minimal_same_source_later_source_instantiation_rule_status']}`",
            f"- anti_shortcut_boundary_status: `{status_payload['anti_shortcut_boundary_status']}`",
            f"- later_source_explicit_naming_instantiation_path_status: `{status_payload['later_source_explicit_naming_instantiation_path_status']}`",
            f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
            "",
            "## Minimum Later-Source Naming Path",
            "",
            "- the current frozen source still names the missing naming components explicitly: no explicit later-source marker, no single later-source identity, and no explicit full-path attachment on that same later source",
            "- the minimum same-source later-source instantiation rule is fixed narrowly: one same later source must carry one explicit later_source_id or later_frozen_run_id, one later source and only one later source, one declaration marker, one candidate id, one class, one explicit host-failure sentence, and matched status, registry, and read surfaces",
            "- the path remains bounded by the anti-shortcut boundary: no broader-settlement promotion, no retroactive rewrite, no graph-wide leap, and no worker-side synthesis",
            "",
            "## Judgment",
            "",
        ]
    )

    if status_payload["later_source_explicit_naming_instantiation_path_status"] == "path_defined":
        lines.append(
            "- the minimum later-source path by which one explicit later-source naming instance could later be honestly instantiated is now fixed, while the current source still does not admit a later source or create a naming instance"
        )
    elif status_payload["later_source_explicit_naming_instantiation_path_status"] == "not_yet_defined":
        lines.append(
            "- the preserved Gate11H line remains bounded, but the minimum later-source explicit-naming instantiation path is not yet fixed narrowly enough"
        )
    elif status_payload["later_source_explicit_naming_instantiation_path_status"] == "denied":
        lines.append(
            "- the attempted later-source explicit-naming instantiation path is denied because it would require shortcut, inflation, rewrite, leap, or synthesis pressure"
        )
    else:
        lines.append(
            "- the frozen source is incomplete for a later-source explicit-naming instantiation-path judgment"
        )

    if status_payload["next_named_blocker"]:
        lines.append(
            f"- next_named_blocker: `{status_payload['next_named_blocker']}`"
        )

    lines.extend(
        [
            "",
            "## Memory Hook",
            "",
            "- Gate11I does not admit a later source; it fixes the minimum later-source path by which one explicit later-source naming instance could later become honestly instantiated under the preserved Gate11H line.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    source_gate11h_dir = Path(args.gate11h_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_gate11h_manifest = gate9a.read_json(source_gate11h_dir / gate11h.DEFAULT_MANIFEST)
    source_gate11h_status = gate9a.read_json(source_gate11h_dir / gate11h.DEFAULT_STATUS)
    report_text = (source_gate11h_dir / gate11h.DEFAULT_REPORT).read_text(encoding="utf-8")

    run_id = out_dir.name
    status_payload = build_status_payload(source_gate11h_manifest, source_gate11h_status, report_text)
    registry_rows = build_registry(source_gate11h_manifest, source_gate11h_status, status_payload)
    policy_compare_rows = build_policy_compare(registry_rows)
    report_text = build_report(run_id, source_gate11h_manifest, policy_compare_rows, status_payload)

    registry_path = out_dir / DEFAULT_REGISTRY
    policy_compare_path = out_dir / DEFAULT_POLICY_COMPARE
    status_path = out_dir / DEFAULT_STATUS
    report_path = out_dir / DEFAULT_REPORT

    gate9a.write_jsonl(registry_path, registry_rows)
    gate9a.write_csv(
        policy_compare_path,
        (
            "source_gate11h_run_id",
            "gate10_closeout_preservation_status",
            "gate11a_absence_result_preservation_status",
            "gate11c_declaration_surface_preservation_status",
            "gate11d_not_yet_declared_state_preservation_status",
            "gate11e_path_defined_state_preservation_status",
            "gate11f_not_yet_admissible_state_preservation_status",
            "gate11g_naming_surface_preservation_status",
            "gate11h_not_yet_named_state_preservation_status",
            "broader_trusted_tree_settlement_still_unearned_status",
            "operator_admission_still_denied_status",
            "retroactive_reinterpretation_forbidden_status",
            "missing_naming_component_naming_status",
            "minimal_same_source_later_source_instantiation_rule_status",
            "anti_shortcut_boundary_status",
            "later_source_explicit_naming_instantiation_path_status",
        ),
        policy_compare_rows,
    )
    gate9a.write_json(status_path, status_payload)
    gate9a.write_text(report_path, report_text)

    manifest = {
        "run_id": run_id,
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "code_git_commit": gate9a.current_git_commit(),
        "source_gate11h_run_id": str(source_gate11h_manifest.get("run_id") or ""),
        "source_gate11h_code_git_commit": str(
            source_gate11h_manifest.get("code_git_commit") or ""
        ),
        "inputs": {
            "gate11h_dir": str(source_gate11h_dir),
        },
        "outputs": {
            "registry": str(registry_path),
            "policy_compare": str(policy_compare_path),
            "status": str(status_path),
            "report": str(report_path),
        },
    }
    manifest_path = out_dir / DEFAULT_MANIFEST
    gate9a.write_json(manifest_path, manifest)

    checksums = {
        DEFAULT_REGISTRY: sha256_file(registry_path),
        DEFAULT_POLICY_COMPARE: sha256_file(policy_compare_path),
        DEFAULT_STATUS: sha256_file(status_path),
        DEFAULT_REPORT: sha256_file(report_path),
        DEFAULT_MANIFEST: sha256_file(manifest_path),
    }
    gate9a.write_json(out_dir / DEFAULT_CHECKSUMS, checksums)


if __name__ == "__main__":
    main()