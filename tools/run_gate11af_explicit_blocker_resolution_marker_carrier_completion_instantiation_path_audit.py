#!/usr/bin/env python3
"""Run a Gate11AF explicit blocker-resolution marker carrier-completion instantiation-path audit on Gate11AE outputs."""

import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Sequence

import run_gate11ae_explicit_blocker_resolution_marker_carrier_completion_audit as gate11ae
import run_gate9a_graph_gauge_consumer as gate9a


SCHEMA_VERSION = "gate11af_explicit_blocker_resolution_marker_carrier_completion_instantiation_path_audit_v1"
METHOD_ID = "gate11af_explicit_blocker_resolution_marker_carrier_completion_instantiation_path_audit_v1"
DEFAULT_REGISTRY = "explicit_blocker_resolution_marker_carrier_completion_instantiation_path_registry.jsonl"
DEFAULT_POLICY_COMPARE = "explicit_blocker_resolution_marker_carrier_completion_instantiation_path_policy_compare.csv"
DEFAULT_STATUS = "explicit_blocker_resolution_marker_carrier_completion_instantiation_path_status.json"
DEFAULT_REPORT = "gate11af_explicit_blocker_resolution_marker_carrier_completion_instantiation_path_read.md"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"

SOURCE_PRESERVATION_KEYS = (*gate11ae.SOURCE_PRESERVATION_KEYS, "gate11ad_not_yet_present_state_preservation_status")
CONFIRMED_KEYS = gate11ae.CONFIRMED_KEYS
REQUIRED_GATE11AE_STATUS_KEYS = gate11ae.STATUS_FIELDS
STATUS_FIELDS = (
    *SOURCE_PRESERVATION_KEYS,
    "gate11ae_residual_named_state_preservation_status",
    *CONFIRMED_KEYS,
    "named_residual_marker_carrier_condition_preservation_status",
    "minimum_same_source_carrier_completion_rule_status",
    "bounded_read_prefix_completion_requirement_status",
    "carrier_completion_boundary_status",
    "explicit_blocker_resolution_marker_carrier_completion_instantiation_path_status",
    "next_named_blocker",
)

KNOWN_RESIDUAL_BLOCKERS = gate11ae.KNOWN_RESIDUAL_BLOCKERS
PATH_DEFINABLE_RESIDUAL_BLOCKERS = {
    "no_explicit_blocker_resolution_marker",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Gate11AF explicit blocker-resolution marker carrier-completion instantiation-path audit "
            "from the frozen Gate11AE carrier-completion run without deciding completion or blocker resolution."
        )
    )
    parser.add_argument("--gate11ae-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_status_value(status_payload: Dict[str, Any], key: str) -> str:
    return str(status_payload.get(key, ""))


def source_is_incomplete(source_manifest: Dict[str, Any], source_status: Dict[str, Any], report_text: str) -> bool:
    if not str(source_manifest.get("run_id") or ""):
        return True
    if not report_text.strip():
        return True
    return any(key not in source_status for key in REQUIRED_GATE11AE_STATUS_KEYS)


def blocker_sentence(blocker: str) -> str:
    mapping = {
        "no_explicit_blocker_resolution_marker": (
            "the minimum honest path is fixed narrowly: one same later source must carry one explicit blocker-resolution marker, one explicit later_source_id or later_frozen_run_id, one blocker-resolution marker and only one blocker-resolution marker, one explicit same-source path-attachment status marked completed, one bounded read-prefix declaration for the blocker-resolution marker, repeated bounded residual_completion_surface rows for the required same-source elements, one explicit residual completion marker, one explicit admissible later-source presence marker, one declaration marker, one candidate id, one class, one explicit host-failure sentence, and matched status, registry, and read surfaces"
        ),
        "minimum_same_source_carrier_completion_rule_not_fixed": (
            "the named blocker-resolution marker carrier condition remains preserved, but the minimum same-source carrier-completion rule is not yet fixed narrowly enough"
        ),
        "bounded_read_prefix_completion_requirement_not_fixed": (
            "the named blocker-resolution marker carrier condition remains preserved, but the bounded read-prefix completion requirement is not yet fixed narrowly enough"
        ),
        "named_residual_marker_carrier_condition_not_preserved": (
            "the controlling source no longer preserves the named blocker-resolution marker carrier condition narrowly enough for path definition"
        ),
    }
    return mapping.get(blocker, "the explicit blocker-resolution marker carrier-completion instantiation path is not yet fixed narrowly enough")


def build_registry(source_manifest: Dict[str, Any], status_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    row = {
        "source_gate11ae_run_id": str(source_manifest.get("run_id") or ""),
        "source_gate11ae_code_git_commit": str(source_manifest.get("code_git_commit") or ""),
    }
    row.update({key: str(status_payload[key]) for key in STATUS_FIELDS})
    return [row]


def build_policy_compare(registry_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"source_gate11ae_run_id": str(row["source_gate11ae_run_id"]), **{key: str(row[key]) for key in STATUS_FIELDS}} for row in registry_rows]


def build_status_payload(source_manifest: Dict[str, Any], source_status: Dict[str, Any], report_text: str) -> Dict[str, Any]:
    incomplete = source_is_incomplete(source_manifest, source_status, report_text)
    status_payload = {
        key: ("preserved" if source_status_value(source_status, key) == "preserved" else "not_preserved")
        for key in SOURCE_PRESERVATION_KEYS
    }

    source_gate11ae_status = source_status_value(source_status, "explicit_blocker_resolution_marker_carrier_completion_status")
    if source_gate11ae_status == "residual_named":
        gate11ae_residual_named_state_preservation_status = "preserved"
    elif source_gate11ae_status == "deferred":
        gate11ae_residual_named_state_preservation_status = "deferred"
    else:
        gate11ae_residual_named_state_preservation_status = "not_preserved"
    status_payload["gate11ae_residual_named_state_preservation_status"] = gate11ae_residual_named_state_preservation_status

    for key in CONFIRMED_KEYS:
        status_payload[key] = "confirmed" if source_status_value(source_status, key) == "confirmed" else "not_confirmed"

    source_blocker = source_status_value(source_status, "next_named_blocker")

    if incomplete:
        named_residual_marker_carrier_condition_preservation_status = "deferred"
    elif (
        source_status_value(source_status, "explicit_marker_carrier_completion_status") == "missing"
        and source_status_value(source_status, "marker_singularity_carrier_completion_status") == "missing"
        and source_status_value(source_status, "same_source_path_attachment_carrier_completion_status") == "missing"
        and source_blocker in KNOWN_RESIDUAL_BLOCKERS
    ):
        named_residual_marker_carrier_condition_preservation_status = "preserved"
    else:
        named_residual_marker_carrier_condition_preservation_status = "not_preserved"
    status_payload["named_residual_marker_carrier_condition_preservation_status"] = named_residual_marker_carrier_condition_preservation_status

    if incomplete:
        carrier_completion_boundary_status = "deferred"
    elif (
        any(status_payload[key] != "confirmed" for key in CONFIRMED_KEYS)
        or source_status_value(source_status, "carrier_completion_boundary_status") != "confirmed"
    ):
        carrier_completion_boundary_status = "denied"
    else:
        carrier_completion_boundary_status = "confirmed"
    status_payload["carrier_completion_boundary_status"] = carrier_completion_boundary_status

    if incomplete:
        minimum_same_source_carrier_completion_rule_status = "deferred"
    elif carrier_completion_boundary_status == "denied":
        minimum_same_source_carrier_completion_rule_status = "not_defined"
    elif gate11ae_residual_named_state_preservation_status == "deferred":
        minimum_same_source_carrier_completion_rule_status = "deferred"
    elif gate11ae_residual_named_state_preservation_status != "preserved":
        minimum_same_source_carrier_completion_rule_status = "not_defined"
    elif named_residual_marker_carrier_condition_preservation_status != "preserved":
        minimum_same_source_carrier_completion_rule_status = "not_defined"
    elif source_blocker in PATH_DEFINABLE_RESIDUAL_BLOCKERS:
        minimum_same_source_carrier_completion_rule_status = "defined"
    else:
        minimum_same_source_carrier_completion_rule_status = "not_defined"
    status_payload["minimum_same_source_carrier_completion_rule_status"] = minimum_same_source_carrier_completion_rule_status

    if incomplete:
        bounded_read_prefix_completion_requirement_status = "deferred"
    elif carrier_completion_boundary_status == "denied":
        bounded_read_prefix_completion_requirement_status = "not_defined"
    elif gate11ae_residual_named_state_preservation_status == "deferred":
        bounded_read_prefix_completion_requirement_status = "deferred"
    elif minimum_same_source_carrier_completion_rule_status != "defined":
        bounded_read_prefix_completion_requirement_status = "not_defined"
    elif source_blocker in PATH_DEFINABLE_RESIDUAL_BLOCKERS:
        bounded_read_prefix_completion_requirement_status = "defined"
    else:
        bounded_read_prefix_completion_requirement_status = "not_defined"
    status_payload["bounded_read_prefix_completion_requirement_status"] = bounded_read_prefix_completion_requirement_status

    if incomplete:
        overall = "deferred"
    elif (
        any(status_payload[key] != "preserved" for key in SOURCE_PRESERVATION_KEYS)
        or gate11ae_residual_named_state_preservation_status == "not_preserved"
        or any(status_payload[key] != "confirmed" for key in CONFIRMED_KEYS)
        or carrier_completion_boundary_status == "denied"
    ):
        overall = "denied"
    elif (
        gate11ae_residual_named_state_preservation_status == "deferred"
        or named_residual_marker_carrier_condition_preservation_status == "deferred"
        or minimum_same_source_carrier_completion_rule_status == "deferred"
        or bounded_read_prefix_completion_requirement_status == "deferred"
        or carrier_completion_boundary_status == "deferred"
    ):
        overall = "deferred"
    elif (
        named_residual_marker_carrier_condition_preservation_status == "preserved"
        and minimum_same_source_carrier_completion_rule_status == "defined"
        and bounded_read_prefix_completion_requirement_status == "defined"
        and carrier_completion_boundary_status == "confirmed"
    ):
        overall = "path_defined"
    else:
        overall = "not_yet_defined"
    status_payload["explicit_blocker_resolution_marker_carrier_completion_instantiation_path_status"] = overall

    next_named_blocker = ""
    for key in SOURCE_PRESERVATION_KEYS:
        if status_payload[key] == "not_preserved":
            next_named_blocker = key.replace("_preservation_status", "_not_preserved")
            break
    if not next_named_blocker and gate11ae_residual_named_state_preservation_status == "not_preserved":
        next_named_blocker = "gate11ae_residual_named_state_not_preserved"
    if not next_named_blocker:
        if status_payload["broader_trusted_tree_settlement_still_unearned_status"] != "confirmed":
            next_named_blocker = "broader_trusted_tree_settlement_not_explicitly_unearned"
        elif status_payload["operator_admission_still_denied_status"] != "confirmed":
            next_named_blocker = "operator_admission_not_denied"
        elif status_payload["retroactive_reinterpretation_forbidden_status"] != "confirmed":
            next_named_blocker = "retroactive_reinterpretation_pressure"
        elif carrier_completion_boundary_status == "deferred":
            next_named_blocker = "controlling_source_incomplete"
        elif carrier_completion_boundary_status == "denied":
            next_named_blocker = "carrier_completion_boundary_not_intact"
        elif gate11ae_residual_named_state_preservation_status == "deferred":
            next_named_blocker = source_blocker or "upstream_carrier_completion_deferred"
        elif named_residual_marker_carrier_condition_preservation_status != "preserved":
            next_named_blocker = "named_residual_marker_carrier_condition_not_preserved"
        elif minimum_same_source_carrier_completion_rule_status != "defined":
            next_named_blocker = "minimum_same_source_carrier_completion_rule_not_fixed"
        elif bounded_read_prefix_completion_requirement_status != "defined":
            next_named_blocker = "bounded_read_prefix_completion_requirement_not_fixed"
        elif overall == "path_defined":
            next_named_blocker = ""
        else:
            next_named_blocker = "carrier_completion_instantiation_path_not_yet_defined"
    status_payload["next_named_blocker"] = next_named_blocker
    return status_payload


def build_report(run_id: str, source_manifest: Dict[str, Any], status_payload: Dict[str, Any]) -> str:
    lines = [
        "# Gate11AF Explicit Blocker-Resolution Marker Carrier-Completion Instantiation-Path Read",
        "",
        f"run_id: {run_id}",
        f"source_gate11ae_run_id: {source_manifest.get('run_id', '')}",
        f"source_gate11ae_code_git_commit: {source_manifest.get('code_git_commit', '')}",
        "",
        "## Discipline",
        "",
        "- Gate11AF asks only what is the minimum honest path by which the named blocker-resolution marker carrier condition could later be completed under the fixed Gate11AE line",
        "- Gate11AF defines path only",
        "- Gate11AF does not instantiate a marker, resolve the blocker, or complete the residual",
        "- Gate11AF does not admit a later source, decide explicit presence, or reopen operator admission",
        "",
        "## Status",
        "",
    ]
    lines.extend([f"- {key}: `{status_payload[key]}`" for key in STATUS_FIELDS])
    lines.extend(["", "## Judgment", ""])

    outcome = status_payload["explicit_blocker_resolution_marker_carrier_completion_instantiation_path_status"]
    if outcome == "path_defined":
        lines.append(f"- {blocker_sentence(status_payload['next_named_blocker'] or 'no_explicit_blocker_resolution_marker')}")
    elif outcome == "not_yet_defined":
        lines.append("- the fixed Gate11AE line still does not fix the minimum explicit blocker-resolution marker carrier-completion path narrowly enough")
    elif outcome == "denied":
        lines.append("- the attempted carrier-completion path read is denied because it would depend on shortcut, inflation, rewrite, graph-wide leap, or worker-side synthesis")
    else:
        lines.append("- the frozen source is incomplete or contradictory for an explicit blocker-resolution marker carrier-completion instantiation-path judgment")
    if status_payload["next_named_blocker"]:
        lines.append(f"- next_named_blocker: `{status_payload['next_named_blocker']}`")
    lines.extend(
        [
            "",
            "## Memory Hook",
            "",
            "- Gate11AF does not say the blocker-resolution marker carrier condition is completed; it fixes only the minimum honest path by which the named blocker-resolution marker carrier condition could later be completed under the fixed Gate11AE line.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    source_dir = Path(args.gate11ae_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = out_dir.name

    source_manifest = gate9a.read_json(source_dir / gate11ae.DEFAULT_MANIFEST)
    source_status = gate9a.read_json(source_dir / gate11ae.DEFAULT_STATUS)
    source_report = (source_dir / gate11ae.DEFAULT_REPORT).read_text(encoding="utf-8")

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
    gate9a.write_csv(policy_compare_path, ("source_gate11ae_run_id", *STATUS_FIELDS), policy_compare_rows)
    gate9a.write_json(status_path, status_payload)
    gate9a.write_text(report_path, build_report(run_id, source_manifest, status_payload))
    gate9a.write_json(
        manifest_path,
        {
            "run_id": run_id,
            "schema_version": SCHEMA_VERSION,
            "method_id": METHOD_ID,
            "code_git_commit": gate9a.current_git_commit(),
            "source_gate11ae_run_id": str(source_manifest.get("run_id") or ""),
            "source_gate11ae_code_git_commit": str(source_manifest.get("code_git_commit") or ""),
            "inputs": {"gate11ae_dir": str(source_dir)},
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