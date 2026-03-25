#!/usr/bin/env python3
"""Run a Gate11AC explicit blocker-resolution marker instantiation-path audit on Gate11AB outputs."""

import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Sequence

import run_gate11ab_one_explicit_blocker_resolution_marker_audit as gate11ab
import run_gate9a_graph_gauge_consumer as gate9a


SCHEMA_VERSION = "gate11ac_explicit_blocker_resolution_marker_instantiation_path_audit_v1"
METHOD_ID = "gate11ac_explicit_blocker_resolution_marker_instantiation_path_audit_v1"
DEFAULT_REGISTRY = "explicit_blocker_resolution_marker_instantiation_path_registry.jsonl"
DEFAULT_POLICY_COMPARE = "explicit_blocker_resolution_marker_instantiation_path_policy_compare.csv"
DEFAULT_STATUS = "explicit_blocker_resolution_marker_instantiation_path_status.json"
DEFAULT_REPORT = "gate11ac_explicit_blocker_resolution_marker_instantiation_path_read.md"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"

SOURCE_PRESERVATION_KEYS = tuple(
    """
gate10_closeout_preservation_status
gate11a_absence_result_preservation_status
gate11c_declaration_surface_preservation_status
gate11d_not_yet_declared_state_preservation_status
gate11e_path_defined_state_preservation_status
gate11f_not_yet_admissible_state_preservation_status
gate11g_naming_surface_preservation_status
gate11h_not_yet_named_state_preservation_status
gate11i_path_defined_state_preservation_status
gate11j_not_yet_admissible_state_preservation_status
gate11k_not_yet_present_state_preservation_status
gate11l_path_defined_state_preservation_status
gate11m_not_yet_present_state_preservation_status
gate11n_residual_named_state_preservation_status
gate11o_path_defined_state_preservation_status
gate11p_not_yet_completed_state_preservation_status
gate11q_surface_defined_state_preservation_status
gate11r_not_yet_present_state_preservation_status
gate11s_path_defined_state_preservation_status
gate11t_not_yet_present_state_preservation_status
gate11u_residual_named_state_preservation_status
gate11v_path_defined_state_preservation_status
gate11w_not_yet_completed_state_preservation_status
gate11x_blocker_named_state_preservation_status
gate11y_path_defined_state_preservation_status
gate11z_not_yet_resolved_state_preservation_status
gate11aa_surface_defined_state_preservation_status
""".split()
)
CONFIRMED_KEYS = tuple(
    """
broader_trusted_tree_settlement_still_unearned_status
operator_admission_still_denied_status
retroactive_reinterpretation_forbidden_status
""".split()
)
REQUIRED_GATE11AB_STATUS_KEYS = (
    *SOURCE_PRESERVATION_KEYS,
    *CONFIRMED_KEYS,
    "explicit_blocker_resolution_marker_status",
    "blocker_resolution_marker_singularity_status",
    "same_source_blocker_resolution_marker_binding_status",
    "blocker_resolution_marker_boundary_status",
    "one_explicit_blocker_resolution_marker_status",
    "next_named_blocker",
)
PATH_DEFINABLE_BLOCKER_MARKERS = {
    "no_explicit_blocker_resolution_marker",
    "same_source_blocker_resolution_marker_binding_not_explicit",
}
STATUS_FIELDS = (
    *SOURCE_PRESERVATION_KEYS,
    "gate11ab_not_yet_present_state_preservation_status",
    *CONFIRMED_KEYS,
    "missing_blocker_resolution_marker_instantiation_components_status",
    "minimum_same_source_blocker_resolution_marker_instantiation_rule_status",
    "bounded_read_prefix_instantiation_requirement_status",
    "blocker_resolution_marker_boundary_status",
    "explicit_blocker_resolution_marker_instantiation_path_status",
    "next_named_blocker",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Gate11AC explicit blocker-resolution marker instantiation-path audit from the frozen Gate11AB run."
    )
    parser.add_argument("--gate11ab-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_status_value(status_payload: Dict[str, Any], key: str) -> str:
    return str(status_payload.get(key, ""))


def source_is_incomplete(manifest: Dict[str, Any], status_payload: Dict[str, Any], report_text: str) -> bool:
    if not str(manifest.get("run_id") or "") or not report_text.strip():
        return True
    return any(key not in status_payload for key in REQUIRED_GATE11AB_STATUS_KEYS)


def blocker_sentence(blocker: str) -> str:
    mapping = {
        "": "the blocker-resolution marker-not-yet-present line remains preserved, and the minimum honest path is fixed narrowly enough for one explicit blocker-resolution marker to later be instantiated on one same source without widening the line",
        "no_explicit_blocker_resolution_marker": "the minimum honest path is fixed narrowly: one same later source must carry one explicit blocker-resolution marker, one explicit later_source_id or later_frozen_run_id, one later source and only one later source, one explicit same-source completion status, one bounded read-prefix declaration for the marker, repeated bounded residual_completion_surface rows for the required same-source elements, one explicit residual completion marker, one explicit admissible later-source presence marker, one declaration marker, one candidate id, one class, one explicit host-failure sentence, and matched status, registry, and read surfaces",
        "same_source_blocker_resolution_marker_binding_not_explicit": "the blocker-resolution marker line remains not yet present, and the missing same-source additions are fixed narrowly: one same later source must carry one explicit later_source_id or later_frozen_run_id, one later source and only one later source, one explicit same-source completion status, and the bounded same-source rows required to bind the marker without widening the line",
        "minimum_same_source_blocker_resolution_marker_instantiation_rule_not_fixed": "the blocker-resolution marker-not-yet-present line remains preserved, but the minimum same-source marker-instantiation additions are not yet fixed narrowly enough",
    }
    return mapping.get(blocker, "the minimum explicit blocker-resolution marker instantiation path is not yet fixed narrowly enough")


def preservation_status(status_payload: Dict[str, Any], key: str) -> str:
    return "preserved" if source_status_value(status_payload, key) == "preserved" else "not_preserved"


def build_registry(source_manifest: Dict[str, Any], status_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    row = {
        "source_gate11ab_run_id": str(source_manifest.get("run_id") or ""),
        "source_gate11ab_code_git_commit": str(source_manifest.get("code_git_commit") or ""),
    }
    row.update({key: str(status_payload[key]) for key in STATUS_FIELDS})
    return [row]


def build_policy_compare(registry_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"source_gate11ab_run_id": str(row["source_gate11ab_run_id"]), **{key: str(row[key]) for key in STATUS_FIELDS}} for row in registry_rows]


def build_status_payload(source_manifest: Dict[str, Any], source_status: Dict[str, Any], report_text: str) -> Dict[str, Any]:
    incomplete = source_is_incomplete(source_manifest, source_status, report_text)
    status_payload = {key: preservation_status(source_status, key) for key in SOURCE_PRESERVATION_KEYS}

    source_marker_status = source_status_value(source_status, "one_explicit_blocker_resolution_marker_status")
    if source_marker_status == "not_yet_present":
        gate11ab_preservation = "preserved"
    elif source_marker_status == "deferred":
        gate11ab_preservation = "deferred"
    else:
        gate11ab_preservation = "not_preserved"
    status_payload["gate11ab_not_yet_present_state_preservation_status"] = gate11ab_preservation

    for key in CONFIRMED_KEYS:
        status_payload[key] = "confirmed" if source_status_value(source_status, key) == "confirmed" else "not_confirmed"

    source_blocker = source_status_value(source_status, "next_named_blocker")
    source_boundary = source_status_value(source_status, "blocker_resolution_marker_boundary_status")
    if incomplete:
        boundary = "deferred"
    elif any(status_payload[key] != "confirmed" for key in CONFIRMED_KEYS) or source_boundary != "confirmed":
        boundary = "denied"
    else:
        boundary = "confirmed"
    status_payload["blocker_resolution_marker_boundary_status"] = boundary

    if incomplete or source_marker_status == "deferred":
        missing = "deferred"
    elif (
        source_status_value(source_status, "explicit_blocker_resolution_marker_status") == "absent"
        and source_status_value(source_status, "blocker_resolution_marker_singularity_status") == "none"
        and source_status_value(source_status, "same_source_blocker_resolution_marker_binding_status") == "not_explicit"
        and source_blocker in PATH_DEFINABLE_BLOCKER_MARKERS
    ):
        missing = "named"
    elif source_blocker in PATH_DEFINABLE_BLOCKER_MARKERS:
        missing = "partially_named"
    else:
        missing = "not_named"
    status_payload["missing_blocker_resolution_marker_instantiation_components_status"] = missing

    if incomplete or source_marker_status == "deferred":
        minimum_rule = "deferred"
    elif boundary == "denied" or gate11ab_preservation != "preserved":
        minimum_rule = "not_defined"
    elif source_blocker in PATH_DEFINABLE_BLOCKER_MARKERS:
        minimum_rule = "defined"
    else:
        minimum_rule = "not_defined"
    status_payload["minimum_same_source_blocker_resolution_marker_instantiation_rule_status"] = minimum_rule

    if incomplete or source_marker_status == "deferred":
        bounded_prefix = "deferred"
    elif source_blocker in PATH_DEFINABLE_BLOCKER_MARKERS:
        bounded_prefix = "defined"
    else:
        bounded_prefix = "not_defined"
    status_payload["bounded_read_prefix_instantiation_requirement_status"] = bounded_prefix

    if incomplete:
        overall = "deferred"
    elif (
        any(status_payload[key] != "preserved" for key in SOURCE_PRESERVATION_KEYS)
        or gate11ab_preservation == "not_preserved"
        or any(status_payload[key] != "confirmed" for key in CONFIRMED_KEYS)
        or boundary == "denied"
    ):
        overall = "denied"
    elif missing in {"named", "partially_named"} and minimum_rule == "defined" and bounded_prefix == "defined" and boundary == "confirmed" and gate11ab_preservation == "preserved":
        overall = "path_defined"
    elif gate11ab_preservation == "deferred" or missing == "deferred" or minimum_rule == "deferred" or bounded_prefix == "deferred" or boundary == "deferred":
        overall = "deferred"
    else:
        overall = "not_yet_defined"
    status_payload["explicit_blocker_resolution_marker_instantiation_path_status"] = overall

    next_named_blocker = ""
    for key in (*SOURCE_PRESERVATION_KEYS, "gate11ab_not_yet_present_state_preservation_status"):
        if status_payload[key] == "not_preserved":
            next_named_blocker = key.replace("_preservation_status", "_not_preserved")
            break
    if not next_named_blocker:
        if status_payload["broader_trusted_tree_settlement_still_unearned_status"] != "confirmed":
            next_named_blocker = "broader_trusted_tree_settlement_not_explicitly_unearned"
        elif status_payload["operator_admission_still_denied_status"] != "confirmed":
            next_named_blocker = "operator_admission_not_denied"
        elif status_payload["retroactive_reinterpretation_forbidden_status"] != "confirmed":
            next_named_blocker = "retroactive_reinterpretation_pressure"
        elif boundary == "deferred":
            next_named_blocker = "controlling_source_incomplete"
        elif boundary == "denied":
            next_named_blocker = "blocker_resolution_marker_boundary_not_intact"
        elif gate11ab_preservation == "deferred" or missing == "deferred":
            next_named_blocker = "multiple_future_blocker_markers"
        elif missing == "not_named":
            next_named_blocker = "missing_blocker_resolution_marker_instantiation_components_not_named"
        elif minimum_rule != "defined":
            next_named_blocker = "minimum_same_source_blocker_resolution_marker_instantiation_rule_not_fixed"
        elif bounded_prefix != "defined":
            next_named_blocker = "bounded_read_prefix_instantiation_requirement_not_fixed"
    status_payload["next_named_blocker"] = next_named_blocker
    return status_payload


def build_report(run_id: str, source_manifest: Dict[str, Any], policy_compare_rows: Sequence[Dict[str, Any]], status_payload: Dict[str, Any]) -> str:
    lines = [
        "# Gate11AC Explicit Blocker-Resolution Marker Instantiation Path Read",
        "",
        f"run_id: {run_id}",
        f"source_gate11ab_run_id: {source_manifest.get('run_id', '')}",
        f"source_gate11ab_code_git_commit: {source_manifest.get('code_git_commit', '')}",
        "",
        "## Discipline",
        "",
        "- Gate11AC asks only what is the minimum honest path by which one explicit blocker-resolution marker could later be instantiated under the fixed Gate11AB line",
        "- Gate11AC defines path only",
        "- Gate11AC does not instantiate a marker",
        "- Gate11AC does not judge blocker resolution or residual completion",
        "- Gate11AC does not admit a later source or decide reopening eligibility",
        "",
        "## Status",
        "",
    ]
    lines.extend([f"- {key}: `{status_payload[key]}`" for key in STATUS_FIELDS])
    lines.extend(
        [
            "",
            "## Source Summary",
            "",
            f"- source rows compared: `{len(policy_compare_rows)}`",
            f"- source_gate11ab_run_id: `{source_manifest.get('run_id', '')}`",
            "",
            "## Judgment",
            "",
        ]
    )
    outcome = status_payload["explicit_blocker_resolution_marker_instantiation_path_status"]
    if outcome == "path_defined":
        lines.append(f"- {blocker_sentence(status_payload['next_named_blocker'])}")
        lines.append("- the minimum same-source blocker-resolution marker-instantiation path is fixed narrowly enough for a later audit, without instantiating a marker now")
    elif outcome == "not_yet_defined":
        lines.append("- the blocker-resolution marker-not-yet-present line remains preserved, but the minimum same-source marker-instantiation path is not yet fixed narrowly enough")
    elif outcome == "denied":
        lines.append("- the attempted blocker-resolution marker-instantiation path is denied because it would depend on shortcut, inflation, rewrite, graph-wide leap, or worker-side synthesis")
    else:
        lines.append("- the frozen source is incomplete or contradictory for a blocker-resolution marker-instantiation-path judgment")
    if status_payload["next_named_blocker"]:
        lines.append(f"- next_named_blocker: `{status_payload['next_named_blocker']}`")
    lines.extend(
        [
            "",
            "## Memory Hook",
            "",
            "- Gate11AC does not say a blocker-resolution marker exists; it fixes only the minimum honest path by which one explicit blocker-resolution marker could later be instantiated under the fixed Gate11AB line.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    source_dir = Path(args.gate11ab_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = out_dir.name

    source_manifest = gate9a.read_json(source_dir / gate11ab.DEFAULT_MANIFEST)
    source_status = gate9a.read_json(source_dir / gate11ab.DEFAULT_STATUS)
    source_report = (source_dir / gate11ab.DEFAULT_REPORT).read_text(encoding="utf-8")

    status_payload = build_status_payload(source_manifest, source_status, source_report)
    registry_rows = build_registry(source_manifest, status_payload)
    policy_compare_rows = build_policy_compare(registry_rows)

    registry_path = out_dir / DEFAULT_REGISTRY
    policy_compare_path = out_dir / DEFAULT_POLICY_COMPARE
    status_path = out_dir / DEFAULT_STATUS
    report_path = out_dir / DEFAULT_REPORT
    manifest_path = out_dir / DEFAULT_MANIFEST
    checksums_path = out_dir / DEFAULT_CHECKSUMS

    gate9a.write_jsonl(registry_path, registry_rows)
    gate9a.write_csv(policy_compare_path, ("source_gate11ab_run_id", *STATUS_FIELDS), policy_compare_rows)
    gate9a.write_json(status_path, status_payload)
    gate9a.write_text(report_path, build_report(run_id, source_manifest, policy_compare_rows, status_payload))
    gate9a.write_json(
        manifest_path,
        {
            "run_id": run_id,
            "schema_version": SCHEMA_VERSION,
            "method_id": METHOD_ID,
            "code_git_commit": gate9a.current_git_commit(),
            "source_gate11ab_run_id": str(source_manifest.get("run_id") or ""),
            "source_gate11ab_code_git_commit": str(source_manifest.get("code_git_commit") or ""),
            "inputs": {"gate11ab_dir": str(source_dir)},
            "outputs": {
                "registry": str(registry_path),
                "policy_compare": str(policy_compare_path),
                "status": str(status_path),
                "report": str(report_path),
            },
        },
    )
    gate9a.write_json(
        checksums_path,
        {
            DEFAULT_REGISTRY: sha256_file(registry_path),
            DEFAULT_POLICY_COMPARE: sha256_file(policy_compare_path),
            DEFAULT_STATUS: sha256_file(status_path),
            DEFAULT_REPORT: sha256_file(report_path),
            DEFAULT_MANIFEST: sha256_file(manifest_path),
        },
    )


if __name__ == "__main__":
    main()
