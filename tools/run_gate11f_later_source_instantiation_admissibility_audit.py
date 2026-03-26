#!/usr/bin/env python3
"""Run a Gate11F later-source instantiation admissibility audit on Gate11E outputs."""

import argparse
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

import run_gate11e_explicit_declaration_instantiation_path_audit as gate11e
import run_gate9a_graph_gauge_consumer as gate9a


SCHEMA_VERSION = "gate11f_later_source_instantiation_admissibility_audit_v1"
METHOD_ID = "gate11f_later_source_instantiation_admissibility_audit_v1"

DEFAULT_REGISTRY = "later_source_instantiation_admissibility_registry.jsonl"
DEFAULT_POLICY_COMPARE = "later_source_instantiation_admissibility_policy_compare.csv"
DEFAULT_STATUS = "later_source_instantiation_admissibility_status.json"
DEFAULT_REPORT = "gate11f_later_source_instantiation_admissibility_read.md"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"

REQUIRED_GATE11E_STATUS_KEYS = (
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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Gate11F later-source instantiation admissibility audit from the frozen "
            "Gate11E path-definition run without declaring a candidate or declaration."
        )
    )
    parser.add_argument("--gate11e-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_status_value(status_payload: Dict[str, Any], key: str) -> str:
    return str(status_payload.get(key, ""))


def source_is_incomplete(
    source_gate11e_manifest: Dict[str, Any], source_gate11e_status: Dict[str, Any], report_text: str
) -> bool:
    if not str(source_gate11e_manifest.get("run_id") or ""):
        return True
    if not report_text.strip():
        return True
    return any(not source_status_value(source_gate11e_status, key) for key in REQUIRED_GATE11E_STATUS_KEYS)


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
        "one and only one bounded_line_insufficiency_candidate_id",
        "one and only one gate11b class",
        "the current bounded line cannot honestly host <candidate_id>",
        "match status payload, registry row, and read sentence",
    )
    lowered = report_text.lower()
    return all(phrase in lowered for phrase in required_phrases)


def build_registry(
    source_gate11e_manifest: Dict[str, Any],
    source_gate11e_status: Dict[str, Any],
    status_payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return [
        {
            "source_gate11e_run_id": str(source_gate11e_manifest.get("run_id") or ""),
            "source_gate11e_code_git_commit": str(
                source_gate11e_manifest.get("code_git_commit") or ""
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
            "broader_trusted_tree_settlement_still_unearned_status": str(
                status_payload["broader_trusted_tree_settlement_still_unearned_status"]
            ),
            "operator_admission_still_denied_status": str(
                status_payload["operator_admission_still_denied_status"]
            ),
            "retroactive_reinterpretation_forbidden_status": str(
                status_payload["retroactive_reinterpretation_forbidden_status"]
            ),
            "later_source_naming_status": str(status_payload["later_source_naming_status"]),
            "later_source_cardinality_status": str(
                status_payload["later_source_cardinality_status"]
            ),
            "same_source_path_attachment_status": str(
                status_payload["same_source_path_attachment_status"]
            ),
            "anti_shortcut_boundary_status": str(status_payload["anti_shortcut_boundary_status"]),
            "later_source_instantiation_admissibility_status": str(
                status_payload["later_source_instantiation_admissibility_status"]
            ),
        }
    ]


def build_policy_compare(registry_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "source_gate11e_run_id": str(row["source_gate11e_run_id"]),
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
            "broader_trusted_tree_settlement_still_unearned_status": str(
                row["broader_trusted_tree_settlement_still_unearned_status"]
            ),
            "operator_admission_still_denied_status": str(row["operator_admission_still_denied_status"]),
            "retroactive_reinterpretation_forbidden_status": str(
                row["retroactive_reinterpretation_forbidden_status"]
            ),
            "later_source_naming_status": str(row["later_source_naming_status"]),
            "later_source_cardinality_status": str(row["later_source_cardinality_status"]),
            "same_source_path_attachment_status": str(row["same_source_path_attachment_status"]),
            "anti_shortcut_boundary_status": str(row["anti_shortcut_boundary_status"]),
            "later_source_instantiation_admissibility_status": str(
                row["later_source_instantiation_admissibility_status"]
            ),
        }
        for row in registry_rows
    ]


def build_status_payload(
    source_gate11e_manifest: Dict[str, Any],
    source_gate11e_status: Dict[str, Any],
    report_text: str,
) -> Dict[str, Any]:
    incomplete = source_is_incomplete(source_gate11e_manifest, source_gate11e_status, report_text)

    gate10_closeout_preservation_status = (
        "preserved"
        if source_status_value(source_gate11e_status, "gate10_closeout_preservation_status") == "preserved"
        else "not_preserved"
    )
    gate11a_absence_result_preservation_status = (
        "preserved"
        if source_status_value(source_gate11e_status, "gate11a_absence_result_preservation_status")
        == "preserved"
        else "not_preserved"
    )
    gate11c_declaration_surface_preservation_status = (
        "preserved"
        if source_status_value(
            source_gate11e_status, "gate11c_declaration_surface_preservation_status"
        )
        == "preserved"
        else "not_preserved"
    )
    gate11d_not_yet_declared_state_preservation_status = (
        "preserved"
        if source_status_value(
            source_gate11e_status, "gate11d_not_yet_declared_state_preservation_status"
        )
        == "preserved"
        else "not_preserved"
    )
    gate11e_path_defined_state_preservation_status = (
        "preserved"
        if source_status_value(
            source_gate11e_status, "explicit_declaration_instantiation_path_status"
        )
        == "path_defined"
        else "not_preserved"
    )
    broader_trusted_tree_settlement_still_unearned_status = (
        "confirmed"
        if source_status_value(
            source_gate11e_status, "broader_trusted_tree_settlement_still_unearned_status"
        )
        == "confirmed"
        else "not_confirmed"
    )
    operator_admission_still_denied_status = (
        "confirmed"
        if source_status_value(source_gate11e_status, "operator_admission_still_denied_status")
        == "confirmed"
        else "not_confirmed"
    )
    retroactive_reinterpretation_forbidden_status = (
        "confirmed"
        if source_status_value(
            source_gate11e_status, "retroactive_reinterpretation_forbidden_status"
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
        or source_status_value(source_gate11e_status, "anti_shortcut_boundary_status") != "confirmed"
    ):
        anti_shortcut_boundary_status = "denied"
    else:
        anti_shortcut_boundary_status = "confirmed"

    later_source_ids = extract_later_source_ids(report_text)
    unique_later_source_ids = sorted(set(later_source_ids))

    if incomplete:
        later_source_naming_status = "deferred"
    elif not later_source_ids:
        later_source_naming_status = "absent"
    else:
        later_source_naming_status = "present"

    if incomplete:
        later_source_cardinality_status = "deferred"
    elif not unique_later_source_ids:
        later_source_cardinality_status = "none"
    elif len(unique_later_source_ids) == 1:
        later_source_cardinality_status = "single"
    else:
        later_source_cardinality_status = "multiple"

    if incomplete:
        same_source_path_attachment_status = "deferred"
    elif later_source_cardinality_status == "multiple":
        same_source_path_attachment_status = "deferred"
    elif later_source_cardinality_status != "single":
        same_source_path_attachment_status = "not_attached"
    elif later_source_path_attached(report_text):
        same_source_path_attachment_status = "attached"
    else:
        same_source_path_attachment_status = "not_attached"

    if incomplete:
        later_source_instantiation_admissibility_status = "deferred"
    elif (
        gate10_closeout_preservation_status != "preserved"
        or gate11a_absence_result_preservation_status != "preserved"
        or gate11c_declaration_surface_preservation_status != "preserved"
        or gate11d_not_yet_declared_state_preservation_status != "preserved"
        or gate11e_path_defined_state_preservation_status != "preserved"
        or anti_shortcut_boundary_status == "denied"
        or broader_trusted_tree_settlement_still_unearned_status != "confirmed"
        or operator_admission_still_denied_status != "confirmed"
        or retroactive_reinterpretation_forbidden_status != "confirmed"
    ):
        later_source_instantiation_admissibility_status = "denied"
    elif (
        later_source_naming_status == "present"
        and later_source_cardinality_status == "single"
        and same_source_path_attachment_status == "attached"
        and anti_shortcut_boundary_status == "confirmed"
    ):
        later_source_instantiation_admissibility_status = "instantiation_admissible"
    elif (
        later_source_naming_status == "deferred"
        or later_source_cardinality_status == "deferred"
        or same_source_path_attachment_status == "deferred"
        or anti_shortcut_boundary_status == "deferred"
    ):
        later_source_instantiation_admissibility_status = "deferred"
    else:
        later_source_instantiation_admissibility_status = "not_yet_admissible"

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
    elif later_source_cardinality_status == "multiple":
        next_named_blocker = "multiple_later_sources"
    elif later_source_naming_status == "absent":
        next_named_blocker = "no_later_source_named"
    elif same_source_path_attachment_status == "not_attached":
        next_named_blocker = "later_source_path_not_attached"
    else:
        next_named_blocker = ""

    return {
        "gate10_closeout_preservation_status": gate10_closeout_preservation_status,
        "gate11a_absence_result_preservation_status": gate11a_absence_result_preservation_status,
        "gate11c_declaration_surface_preservation_status": gate11c_declaration_surface_preservation_status,
        "gate11d_not_yet_declared_state_preservation_status": gate11d_not_yet_declared_state_preservation_status,
        "gate11e_path_defined_state_preservation_status": gate11e_path_defined_state_preservation_status,
        "broader_trusted_tree_settlement_still_unearned_status": broader_trusted_tree_settlement_still_unearned_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "retroactive_reinterpretation_forbidden_status": retroactive_reinterpretation_forbidden_status,
        "later_source_naming_status": later_source_naming_status,
        "later_source_cardinality_status": later_source_cardinality_status,
        "same_source_path_attachment_status": same_source_path_attachment_status,
        "anti_shortcut_boundary_status": anti_shortcut_boundary_status,
        "later_source_instantiation_admissibility_status": later_source_instantiation_admissibility_status,
        "next_named_blocker": next_named_blocker,
    }


def build_report(
    run_id: str,
    source_gate11e_manifest: Dict[str, Any],
    policy_compare_rows: Sequence[Dict[str, Any]],
    status_payload: Dict[str, Any],
) -> str:
    lines = [
        "# Gate11F Later-Source Instantiation Admissibility Read",
        "",
        f"run_id: {run_id}",
        f"source_gate11e_run_id: {source_gate11e_manifest.get('run_id', '')}",
        f"source_gate11e_code_git_commit: {source_gate11e_manifest.get('code_git_commit', '')}",
        "",
        "## Discipline",
        "",
        "- Gate11F asks only whether one later source is now admissibly present as the carrier of the fixed Gate11E path",
        "- Gate11F does not declare a candidate",
        "- Gate11F does not declare that explicit declaration already exists",
        "- Gate11F does not decide reopening eligibility",
        "- Gate10 closeout remains bounded and preserved",
        "- Gate11A absence remains preserved",
        "- Gate11C surface-defined result remains preserved",
        "- Gate11D not-yet-declared result remains preserved",
        "- Gate11E path-defined result remains preserved",
        "- broader trusted-tree settlement remains unearned",
        "- operator admission remains denied",
        "- no retroactive rewrite of Gate9, Gate10, Gate11A, Gate11B, Gate11C, Gate11D, or Gate11E memory is permitted",
        "",
        "## Source Summary",
        "",
        "| source_gate11e_run_id | gate10_closeout_preservation_status | gate11a_absence_result_preservation_status | gate11c_declaration_surface_preservation_status | gate11d_not_yet_declared_state_preservation_status | gate11e_path_defined_state_preservation_status | broader_trusted_tree_settlement_still_unearned_status | operator_admission_still_denied_status | retroactive_reinterpretation_forbidden_status | later_source_naming_status | later_source_cardinality_status | same_source_path_attachment_status | anti_shortcut_boundary_status | later_source_instantiation_admissibility_status |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in policy_compare_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["source_gate11e_run_id"]),
                    str(row["gate10_closeout_preservation_status"]),
                    str(row["gate11a_absence_result_preservation_status"]),
                    str(row["gate11c_declaration_surface_preservation_status"]),
                    str(row["gate11d_not_yet_declared_state_preservation_status"]),
                    str(row["gate11e_path_defined_state_preservation_status"]),
                    str(row["broader_trusted_tree_settlement_still_unearned_status"]),
                    str(row["operator_admission_still_denied_status"]),
                    str(row["retroactive_reinterpretation_forbidden_status"]),
                    str(row["later_source_naming_status"]),
                    str(row["later_source_cardinality_status"]),
                    str(row["same_source_path_attachment_status"]),
                    str(row["anti_shortcut_boundary_status"]),
                    str(row["later_source_instantiation_admissibility_status"]),
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
            f"- broader_trusted_tree_settlement_still_unearned_status: `{status_payload['broader_trusted_tree_settlement_still_unearned_status']}`",
            f"- operator_admission_still_denied_status: `{status_payload['operator_admission_still_denied_status']}`",
            f"- retroactive_reinterpretation_forbidden_status: `{status_payload['retroactive_reinterpretation_forbidden_status']}`",
            f"- later_source_naming_status: `{status_payload['later_source_naming_status']}`",
            f"- later_source_cardinality_status: `{status_payload['later_source_cardinality_status']}`",
            f"- same_source_path_attachment_status: `{status_payload['same_source_path_attachment_status']}`",
            f"- anti_shortcut_boundary_status: `{status_payload['anti_shortcut_boundary_status']}`",
            f"- later_source_instantiation_admissibility_status: `{status_payload['later_source_instantiation_admissibility_status']}`",
            f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
        ]
    )

    if status_payload["later_source_instantiation_admissibility_status"] == "instantiation_admissible":
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- one later source is now admissibly present as the carrier of the fixed Gate11E path",
                "- Gate11F does not instantiate declaration itself here",
            ]
        )
    elif status_payload["later_source_instantiation_admissibility_status"] == "deferred":
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- the later-source admissibility audit remains deferred because the frozen source is incomplete or contradictory",
                f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
            ]
        )
    elif status_payload["later_source_instantiation_admissibility_status"] == "denied":
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- the proposed later-source admissibility path is denied under the frozen Gate11F boundary",
                f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- the fixed Gate11E path exists, but no one later source is yet admissibly present as its carrier",
                f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    source_gate11e_dir = Path(args.gate11e_dir)
    out_dir = Path(args.out_dir)
    run_id = out_dir.name

    out_dir.mkdir(parents=True, exist_ok=True)

    source_gate11e_manifest = gate9a.read_json(source_gate11e_dir / gate11e.DEFAULT_MANIFEST)
    source_gate11e_status = gate9a.read_json(source_gate11e_dir / gate11e.DEFAULT_STATUS)
    source_gate11e_report = (source_gate11e_dir / gate11e.DEFAULT_REPORT).read_text(encoding="utf-8")

    status_payload = build_status_payload(
        source_gate11e_manifest,
        source_gate11e_status,
        source_gate11e_report,
    )
    registry_rows = build_registry(source_gate11e_manifest, source_gate11e_status, status_payload)
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
            "source_gate11e_run_id",
            "gate10_closeout_preservation_status",
            "gate11a_absence_result_preservation_status",
            "gate11c_declaration_surface_preservation_status",
            "gate11d_not_yet_declared_state_preservation_status",
            "gate11e_path_defined_state_preservation_status",
            "broader_trusted_tree_settlement_still_unearned_status",
            "operator_admission_still_denied_status",
            "retroactive_reinterpretation_forbidden_status",
            "later_source_naming_status",
            "later_source_cardinality_status",
            "same_source_path_attachment_status",
            "anti_shortcut_boundary_status",
            "later_source_instantiation_admissibility_status",
        ),
        policy_compare_rows,
    )
    gate9a.write_json(status_path, status_payload)
    gate9a.write_text(
        report_path,
        build_report(
            run_id=run_id,
            source_gate11e_manifest=source_gate11e_manifest,
            policy_compare_rows=policy_compare_rows,
            status_payload=status_payload,
        ),
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "run_id": run_id,
        "code_git_commit": gate9a.current_git_commit(),
        "source_gate11e_dir": gate9a.repo_relative_or_posix(source_gate11e_dir),
        "source_gate11e_run_id": str(source_gate11e_manifest.get("run_id") or ""),
        "source_gate11e_code_git_commit": str(
            source_gate11e_manifest.get("code_git_commit") or ""
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