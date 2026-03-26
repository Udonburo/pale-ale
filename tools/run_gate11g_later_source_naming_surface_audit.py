#!/usr/bin/env python3
"""Run a Gate11G later-source naming surface audit on Gate11F outputs."""

import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Sequence

import run_gate11f_later_source_instantiation_admissibility_audit as gate11f
import run_gate9a_graph_gauge_consumer as gate9a


SCHEMA_VERSION = "gate11g_later_source_naming_surface_audit_v1"
METHOD_ID = "gate11g_later_source_naming_surface_audit_v1"

DEFAULT_REGISTRY = "later_source_naming_surface_registry.jsonl"
DEFAULT_POLICY_COMPARE = "later_source_naming_surface_policy_compare.csv"
DEFAULT_STATUS = "later_source_naming_surface_status.json"
DEFAULT_REPORT = "gate11g_later_source_naming_surface_read.md"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"

REQUIRED_GATE11F_STATUS_KEYS = (
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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Gate11G later-source naming surface audit from the frozen Gate11F "
            "later-source admissibility run without deciding admissibility itself."
        )
    )
    parser.add_argument("--gate11f-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_status_value(status_payload: Dict[str, Any], key: str) -> str:
    return str(status_payload.get(key, ""))


def source_is_incomplete(
    source_gate11f_manifest: Dict[str, Any], source_gate11f_status: Dict[str, Any], report_text: str
) -> bool:
    if not str(source_gate11f_manifest.get("run_id") or ""):
        return True
    if not report_text.strip():
        return True
    return any(not source_status_value(source_gate11f_status, key) for key in REQUIRED_GATE11F_STATUS_KEYS)


def build_registry(
    source_gate11f_manifest: Dict[str, Any],
    source_gate11f_status: Dict[str, Any],
    status_payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return [
        {
            "source_gate11f_run_id": str(source_gate11f_manifest.get("run_id") or ""),
            "source_gate11f_code_git_commit": str(
                source_gate11f_manifest.get("code_git_commit") or ""
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
            "broader_trusted_tree_settlement_still_unearned_status": str(
                status_payload["broader_trusted_tree_settlement_still_unearned_status"]
            ),
            "operator_admission_still_denied_status": str(
                status_payload["operator_admission_still_denied_status"]
            ),
            "retroactive_reinterpretation_forbidden_status": str(
                status_payload["retroactive_reinterpretation_forbidden_status"]
            ),
            "explicit_later_source_marker_shape_status": str(
                status_payload["explicit_later_source_marker_shape_status"]
            ),
            "single_later_source_singularity_status": str(
                status_payload["single_later_source_singularity_status"]
            ),
            "full_path_attachment_shape_status": str(
                status_payload["full_path_attachment_shape_status"]
            ),
            "anti_shortcut_boundary_status": str(status_payload["anti_shortcut_boundary_status"]),
            "later_source_naming_surface_status": str(
                status_payload["later_source_naming_surface_status"]
            ),
        }
    ]


def build_policy_compare(registry_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "source_gate11f_run_id": str(row["source_gate11f_run_id"]),
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
            "broader_trusted_tree_settlement_still_unearned_status": str(
                row["broader_trusted_tree_settlement_still_unearned_status"]
            ),
            "operator_admission_still_denied_status": str(row["operator_admission_still_denied_status"]),
            "retroactive_reinterpretation_forbidden_status": str(
                row["retroactive_reinterpretation_forbidden_status"]
            ),
            "explicit_later_source_marker_shape_status": str(
                row["explicit_later_source_marker_shape_status"]
            ),
            "single_later_source_singularity_status": str(
                row["single_later_source_singularity_status"]
            ),
            "full_path_attachment_shape_status": str(row["full_path_attachment_shape_status"]),
            "anti_shortcut_boundary_status": str(row["anti_shortcut_boundary_status"]),
            "later_source_naming_surface_status": str(row["later_source_naming_surface_status"]),
        }
        for row in registry_rows
    ]


def build_status_payload(
    source_gate11f_manifest: Dict[str, Any],
    source_gate11f_status: Dict[str, Any],
    report_text: str,
) -> Dict[str, Any]:
    incomplete = source_is_incomplete(source_gate11f_manifest, source_gate11f_status, report_text)

    gate10_closeout_preservation_status = (
        "preserved"
        if source_status_value(source_gate11f_status, "gate10_closeout_preservation_status") == "preserved"
        else "not_preserved"
    )
    gate11a_absence_result_preservation_status = (
        "preserved"
        if source_status_value(source_gate11f_status, "gate11a_absence_result_preservation_status")
        == "preserved"
        else "not_preserved"
    )
    gate11c_declaration_surface_preservation_status = (
        "preserved"
        if source_status_value(
            source_gate11f_status, "gate11c_declaration_surface_preservation_status"
        )
        == "preserved"
        else "not_preserved"
    )
    gate11d_not_yet_declared_state_preservation_status = (
        "preserved"
        if source_status_value(
            source_gate11f_status, "gate11d_not_yet_declared_state_preservation_status"
        )
        == "preserved"
        else "not_preserved"
    )
    gate11e_path_defined_state_preservation_status = (
        "preserved"
        if source_status_value(
            source_gate11f_status, "gate11e_path_defined_state_preservation_status"
        )
        == "preserved"
        else "not_preserved"
    )
    gate11f_not_yet_admissible_state_preservation_status = (
        "preserved"
        if source_status_value(
            source_gate11f_status, "later_source_instantiation_admissibility_status"
        )
        == "not_yet_admissible"
        else "not_preserved"
    )
    broader_trusted_tree_settlement_still_unearned_status = (
        "confirmed"
        if source_status_value(
            source_gate11f_status, "broader_trusted_tree_settlement_still_unearned_status"
        )
        == "confirmed"
        else "not_confirmed"
    )
    operator_admission_still_denied_status = (
        "confirmed"
        if source_status_value(source_gate11f_status, "operator_admission_still_denied_status")
        == "confirmed"
        else "not_confirmed"
    )
    retroactive_reinterpretation_forbidden_status = (
        "confirmed"
        if source_status_value(
            source_gate11f_status, "retroactive_reinterpretation_forbidden_status"
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
        or source_status_value(source_gate11f_status, "anti_shortcut_boundary_status") != "confirmed"
    ):
        anti_shortcut_boundary_status = "denied"
    else:
        anti_shortcut_boundary_status = "confirmed"

    if incomplete:
        explicit_later_source_marker_shape_status = "deferred"
        single_later_source_singularity_status = "deferred"
        full_path_attachment_shape_status = "deferred"
    else:
        explicit_later_source_marker_shape_status = "defined"
        single_later_source_singularity_status = "defined"
        full_path_attachment_shape_status = "defined"

    if incomplete:
        later_source_naming_surface_status = "deferred"
    elif (
        gate10_closeout_preservation_status != "preserved"
        or gate11a_absence_result_preservation_status != "preserved"
        or gate11c_declaration_surface_preservation_status != "preserved"
        or gate11d_not_yet_declared_state_preservation_status != "preserved"
        or gate11e_path_defined_state_preservation_status != "preserved"
        or gate11f_not_yet_admissible_state_preservation_status != "preserved"
        or broader_trusted_tree_settlement_still_unearned_status != "confirmed"
        or operator_admission_still_denied_status != "confirmed"
        or retroactive_reinterpretation_forbidden_status != "confirmed"
        or anti_shortcut_boundary_status == "denied"
    ):
        later_source_naming_surface_status = "denied"
    elif (
        explicit_later_source_marker_shape_status == "defined"
        and single_later_source_singularity_status == "defined"
        and full_path_attachment_shape_status == "defined"
        and anti_shortcut_boundary_status == "confirmed"
    ):
        later_source_naming_surface_status = "surface_defined"
    else:
        later_source_naming_surface_status = "not_yet_defined"

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
    else:
        next_named_blocker = ""

    return {
        "gate10_closeout_preservation_status": gate10_closeout_preservation_status,
        "gate11a_absence_result_preservation_status": gate11a_absence_result_preservation_status,
        "gate11c_declaration_surface_preservation_status": gate11c_declaration_surface_preservation_status,
        "gate11d_not_yet_declared_state_preservation_status": gate11d_not_yet_declared_state_preservation_status,
        "gate11e_path_defined_state_preservation_status": gate11e_path_defined_state_preservation_status,
        "gate11f_not_yet_admissible_state_preservation_status": gate11f_not_yet_admissible_state_preservation_status,
        "broader_trusted_tree_settlement_still_unearned_status": broader_trusted_tree_settlement_still_unearned_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "retroactive_reinterpretation_forbidden_status": retroactive_reinterpretation_forbidden_status,
        "explicit_later_source_marker_shape_status": explicit_later_source_marker_shape_status,
        "single_later_source_singularity_status": single_later_source_singularity_status,
        "full_path_attachment_shape_status": full_path_attachment_shape_status,
        "anti_shortcut_boundary_status": anti_shortcut_boundary_status,
        "later_source_naming_surface_status": later_source_naming_surface_status,
        "next_named_blocker": next_named_blocker,
    }


def build_report(
    run_id: str,
    source_gate11f_manifest: Dict[str, Any],
    policy_compare_rows: Sequence[Dict[str, Any]],
    status_payload: Dict[str, Any],
) -> str:
    lines = [
        "# Gate11G Later-Source Naming Surface Read",
        "",
        f"run_id: {run_id}",
        f"source_gate11f_run_id: {source_gate11f_manifest.get('run_id', '')}",
        f"source_gate11f_code_git_commit: {source_gate11f_manifest.get('code_git_commit', '')}",
        "",
        "## Discipline",
        "",
        "- Gate11G asks only what would count as an explicit naming surface for one later source to carry the fixed Gate11E path",
        "- Gate11G does not admit a later source",
        "- Gate11G does not declare a candidate",
        "- Gate11G does not declare that explicit declaration already exists",
        "- Gate11G does not decide reopening eligibility",
        "- Gate10 closeout remains bounded and preserved",
        "- Gate11A absence remains preserved",
        "- Gate11C surface-defined result remains preserved",
        "- Gate11D not-yet-declared result remains preserved",
        "- Gate11E path-defined result remains preserved",
        "- Gate11F not-yet-admissible result remains preserved",
        "- broader trusted-tree settlement remains unearned",
        "- operator admission remains denied",
        "- no retroactive rewrite of Gate9, Gate10, Gate11A, Gate11B, Gate11C, Gate11D, Gate11E, or Gate11F memory is permitted",
        "",
        "## Source Summary",
        "",
        "| source_gate11f_run_id | gate10_closeout_preservation_status | gate11a_absence_result_preservation_status | gate11c_declaration_surface_preservation_status | gate11d_not_yet_declared_state_preservation_status | gate11e_path_defined_state_preservation_status | gate11f_not_yet_admissible_state_preservation_status | broader_trusted_tree_settlement_still_unearned_status | operator_admission_still_denied_status | retroactive_reinterpretation_forbidden_status | explicit_later_source_marker_shape_status | single_later_source_singularity_status | full_path_attachment_shape_status | anti_shortcut_boundary_status | later_source_naming_surface_status |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in policy_compare_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["source_gate11f_run_id"]),
                    str(row["gate10_closeout_preservation_status"]),
                    str(row["gate11a_absence_result_preservation_status"]),
                    str(row["gate11c_declaration_surface_preservation_status"]),
                    str(row["gate11d_not_yet_declared_state_preservation_status"]),
                    str(row["gate11e_path_defined_state_preservation_status"]),
                    str(row["gate11f_not_yet_admissible_state_preservation_status"]),
                    str(row["broader_trusted_tree_settlement_still_unearned_status"]),
                    str(row["operator_admission_still_denied_status"]),
                    str(row["retroactive_reinterpretation_forbidden_status"]),
                    str(row["explicit_later_source_marker_shape_status"]),
                    str(row["single_later_source_singularity_status"]),
                    str(row["full_path_attachment_shape_status"]),
                    str(row["anti_shortcut_boundary_status"]),
                    str(row["later_source_naming_surface_status"]),
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
            f"- broader_trusted_tree_settlement_still_unearned_status: `{status_payload['broader_trusted_tree_settlement_still_unearned_status']}`",
            f"- operator_admission_still_denied_status: `{status_payload['operator_admission_still_denied_status']}`",
            f"- retroactive_reinterpretation_forbidden_status: `{status_payload['retroactive_reinterpretation_forbidden_status']}`",
            f"- explicit_later_source_marker_shape_status: `{status_payload['explicit_later_source_marker_shape_status']}`",
            f"- single_later_source_singularity_status: `{status_payload['single_later_source_singularity_status']}`",
            f"- full_path_attachment_shape_status: `{status_payload['full_path_attachment_shape_status']}`",
            f"- anti_shortcut_boundary_status: `{status_payload['anti_shortcut_boundary_status']}`",
            f"- later_source_naming_surface_status: `{status_payload['later_source_naming_surface_status']}`",
            f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
            "",
            "## Surface Conditions",
            "",
            "- one explicit later_source_id or later_frozen_run_id must count as the machine-readable naming marker",
            "- one later source and only one later source may be named in a single run",
            "- that same later source must be the carrier of the full Gate11E path: declaration marker, candidate id, class, explicit host-failure sentence, and matched status/registry/read surfaces",
            "- broader-settlement promotion, retroactive rewrite, graph-wide leap, and worker-side synthesis remain forbidden shortcuts",
        ]
    )

    if status_payload["later_source_naming_surface_status"] == "surface_defined":
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- the later-source naming surface is now fixed narrowly enough to audit later",
                "- Gate11G does not admit a later source here",
            ]
        )
    elif status_payload["later_source_naming_surface_status"] == "deferred":
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- the later-source naming-surface audit remains deferred because the frozen source is incomplete or contradictory",
                f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
            ]
        )
    elif status_payload["later_source_naming_surface_status"] == "denied":
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- the proposed later-source naming surface is denied under the frozen Gate11G boundary",
                f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- the later-source naming surface is not yet fixed narrowly enough to audit later",
                f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    source_gate11f_dir = Path(args.gate11f_dir)
    out_dir = Path(args.out_dir)
    run_id = out_dir.name

    out_dir.mkdir(parents=True, exist_ok=True)

    source_gate11f_manifest = gate9a.read_json(source_gate11f_dir / gate11f.DEFAULT_MANIFEST)
    source_gate11f_status = gate9a.read_json(source_gate11f_dir / gate11f.DEFAULT_STATUS)
    source_gate11f_report = (source_gate11f_dir / gate11f.DEFAULT_REPORT).read_text(encoding="utf-8")

    status_payload = build_status_payload(
        source_gate11f_manifest,
        source_gate11f_status,
        source_gate11f_report,
    )
    registry_rows = build_registry(source_gate11f_manifest, source_gate11f_status, status_payload)
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
            "source_gate11f_run_id",
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
        ),
        policy_compare_rows,
    )
    gate9a.write_json(status_path, status_payload)
    gate9a.write_text(
        report_path,
        build_report(
            run_id=run_id,
            source_gate11f_manifest=source_gate11f_manifest,
            policy_compare_rows=policy_compare_rows,
            status_payload=status_payload,
        ),
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "run_id": run_id,
        "code_git_commit": gate9a.current_git_commit(),
        "source_gate11f_dir": gate9a.repo_relative_or_posix(source_gate11f_dir),
        "source_gate11f_run_id": str(source_gate11f_manifest.get("run_id") or ""),
        "source_gate11f_code_git_commit": str(
            source_gate11f_manifest.get("code_git_commit") or ""
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