#!/usr/bin/env python3
"""Run a Gate11AD one explicit blocker-resolution marker path-instantiation audit on Gate11AC outputs."""

import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Sequence

import run_gate11ab_one_explicit_blocker_resolution_marker_audit as gate11ab
import run_gate11ac_explicit_blocker_resolution_marker_instantiation_path_audit as gate11ac
import run_gate11r_one_explicit_residual_completion_marker_audit as gate11r
import run_gate9a_graph_gauge_consumer as gate9a


SCHEMA_VERSION = "gate11ad_one_explicit_blocker_resolution_marker_path_instantiation_audit_v1"
METHOD_ID = "gate11ad_one_explicit_blocker_resolution_marker_path_instantiation_audit_v1"
DEFAULT_REGISTRY = "one_explicit_blocker_resolution_marker_path_instantiation_registry.jsonl"
DEFAULT_POLICY_COMPARE = "one_explicit_blocker_resolution_marker_path_instantiation_policy_compare.csv"
DEFAULT_STATUS = "one_explicit_blocker_resolution_marker_path_instantiation_status.json"
DEFAULT_REPORT = "gate11ad_one_explicit_blocker_resolution_marker_path_instantiation_read.md"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"

SOURCE_PRESERVATION_KEYS = (*gate11ac.SOURCE_PRESERVATION_KEYS, "gate11ab_not_yet_present_state_preservation_status")
CONFIRMED_KEYS = gate11ac.CONFIRMED_KEYS
REQUIRED_GATE11AC_STATUS_KEYS = gate11ac.STATUS_FIELDS
STATUS_FIELDS = (
    *SOURCE_PRESERVATION_KEYS,
    "gate11ac_path_defined_state_preservation_status",
    *CONFIRMED_KEYS,
    "explicit_blocker_resolution_marker_status",
    "blocker_resolution_marker_singularity_status",
    "same_source_marker_path_attachment_status",
    "blocker_resolution_marker_boundary_status",
    "one_explicit_blocker_resolution_marker_path_instantiation_status",
    "next_named_blocker",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Gate11AD one explicit blocker-resolution marker path-instantiation audit "
            "from the frozen Gate11AC path-definition run."
        )
    )
    parser.add_argument("--gate11ac-dir", required=True)
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
    return any(key not in source_status for key in REQUIRED_GATE11AC_STATUS_KEYS)


def same_source_marker_path_attached(report_text: str) -> bool:
    return gate11ab.same_source_blocker_resolution_marker_binding_explicit(report_text) and gate11r.bounded_read_prefix_attached(report_text)


def build_registry(source_manifest: Dict[str, Any], status_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    row = {
        "source_gate11ac_run_id": str(source_manifest.get("run_id") or ""),
        "source_gate11ac_code_git_commit": str(source_manifest.get("code_git_commit") or ""),
    }
    row.update({key: str(status_payload[key]) for key in STATUS_FIELDS})
    return [row]


def build_policy_compare(registry_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"source_gate11ac_run_id": str(row["source_gate11ac_run_id"]), **{key: str(row[key]) for key in STATUS_FIELDS}} for row in registry_rows]


def build_status_payload(source_manifest: Dict[str, Any], source_status: Dict[str, Any], report_text: str) -> Dict[str, Any]:
    incomplete = source_is_incomplete(source_manifest, source_status, report_text)
    status_payload = {
        key: ("preserved" if source_status_value(source_status, key) == "preserved" else "not_preserved")
        for key in SOURCE_PRESERVATION_KEYS
    }

    source_path_status = source_status_value(source_status, "explicit_blocker_resolution_marker_instantiation_path_status")
    if source_path_status == "path_defined":
        gate11ac_path_defined_state_preservation_status = "preserved"
    elif source_path_status == "deferred":
        gate11ac_path_defined_state_preservation_status = "deferred"
    else:
        gate11ac_path_defined_state_preservation_status = "not_preserved"
    status_payload["gate11ac_path_defined_state_preservation_status"] = gate11ac_path_defined_state_preservation_status

    for key in CONFIRMED_KEYS:
        status_payload[key] = "confirmed" if source_status_value(source_status, key) == "confirmed" else "not_confirmed"

    if incomplete:
        blocker_resolution_marker_boundary_status = "deferred"
    elif (
        any(status_payload[key] != "confirmed" for key in CONFIRMED_KEYS)
        or source_status_value(source_status, "blocker_resolution_marker_boundary_status") != "confirmed"
    ):
        blocker_resolution_marker_boundary_status = "denied"
    else:
        blocker_resolution_marker_boundary_status = "confirmed"
    status_payload["blocker_resolution_marker_boundary_status"] = blocker_resolution_marker_boundary_status

    later_source_ids = gate11ab.extract_later_source_ids(report_text)
    unique_later_source_ids = sorted(set(later_source_ids))
    marker_present = gate11ab.explicit_blocker_resolution_marker_present(report_text)

    if incomplete:
        explicit_blocker_resolution_marker_status = "deferred"
    elif marker_present:
        explicit_blocker_resolution_marker_status = "present"
    else:
        explicit_blocker_resolution_marker_status = "absent"
    status_payload["explicit_blocker_resolution_marker_status"] = explicit_blocker_resolution_marker_status

    if incomplete:
        blocker_resolution_marker_singularity_status = "deferred"
    elif not marker_present:
        blocker_resolution_marker_singularity_status = "none"
    elif not unique_later_source_ids:
        blocker_resolution_marker_singularity_status = "none"
    elif len(unique_later_source_ids) == 1:
        blocker_resolution_marker_singularity_status = "single"
    else:
        blocker_resolution_marker_singularity_status = "multiple"
    status_payload["blocker_resolution_marker_singularity_status"] = blocker_resolution_marker_singularity_status

    if incomplete:
        same_source_marker_path_attachment_status = "deferred"
    elif not marker_present:
        same_source_marker_path_attachment_status = "not_instantiated"
    elif blocker_resolution_marker_singularity_status == "multiple":
        same_source_marker_path_attachment_status = "deferred"
    elif blocker_resolution_marker_singularity_status != "single":
        same_source_marker_path_attachment_status = "not_instantiated"
    elif same_source_marker_path_attached(report_text):
        same_source_marker_path_attachment_status = "instantiated"
    else:
        same_source_marker_path_attachment_status = "not_instantiated"
    status_payload["same_source_marker_path_attachment_status"] = same_source_marker_path_attachment_status

    if incomplete:
        overall = "deferred"
    elif (
        any(status_payload[key] != "preserved" for key in SOURCE_PRESERVATION_KEYS)
        or gate11ac_path_defined_state_preservation_status == "not_preserved"
        or any(status_payload[key] != "confirmed" for key in CONFIRMED_KEYS)
        or blocker_resolution_marker_boundary_status == "denied"
    ):
        overall = "denied"
    elif (
        explicit_blocker_resolution_marker_status == "present"
        and blocker_resolution_marker_singularity_status == "single"
        and same_source_marker_path_attachment_status == "instantiated"
        and blocker_resolution_marker_boundary_status == "confirmed"
    ):
        overall = "present"
    elif (
        gate11ac_path_defined_state_preservation_status == "deferred"
        or explicit_blocker_resolution_marker_status == "deferred"
        or blocker_resolution_marker_singularity_status == "deferred"
        or same_source_marker_path_attachment_status == "deferred"
        or blocker_resolution_marker_boundary_status == "deferred"
    ):
        overall = "deferred"
    else:
        overall = "not_yet_present"
    status_payload["one_explicit_blocker_resolution_marker_path_instantiation_status"] = overall

    next_named_blocker = ""
    for key in SOURCE_PRESERVATION_KEYS:
        if status_payload[key] == "not_preserved":
            next_named_blocker = key.replace("_preservation_status", "_not_preserved")
            break
    if not next_named_blocker and gate11ac_path_defined_state_preservation_status == "not_preserved":
        next_named_blocker = "gate11ac_path_defined_state_not_preserved"
    if not next_named_blocker:
        if status_payload["broader_trusted_tree_settlement_still_unearned_status"] != "confirmed":
            next_named_blocker = "broader_trusted_tree_settlement_not_explicitly_unearned"
        elif status_payload["operator_admission_still_denied_status"] != "confirmed":
            next_named_blocker = "operator_admission_not_denied"
        elif status_payload["retroactive_reinterpretation_forbidden_status"] != "confirmed":
            next_named_blocker = "retroactive_reinterpretation_pressure"
        elif blocker_resolution_marker_boundary_status == "deferred":
            next_named_blocker = "controlling_source_incomplete"
        elif blocker_resolution_marker_boundary_status == "denied":
            next_named_blocker = "blocker_resolution_marker_boundary_not_intact"
        elif gate11ac_path_defined_state_preservation_status == "deferred":
            next_named_blocker = "upstream_marker_instantiation_path_deferred"
        elif blocker_resolution_marker_singularity_status == "multiple":
            next_named_blocker = "multiple_candidate_markers"
        elif same_source_marker_path_attachment_status == "deferred":
            next_named_blocker = "multiple_candidate_markers"
        elif explicit_blocker_resolution_marker_status == "absent":
            next_named_blocker = "no_explicit_blocker_resolution_marker"
        elif same_source_marker_path_attachment_status != "instantiated":
            next_named_blocker = "same_source_marker_path_not_instantiated"
    status_payload["next_named_blocker"] = next_named_blocker
    return status_payload


def build_report(run_id: str, source_manifest: Dict[str, Any], policy_compare_rows: Sequence[Dict[str, Any]], status_payload: Dict[str, Any]) -> str:
    lines = [
        "# Gate11AD One Explicit Blocker-Resolution Marker Path-Instantiation Read",
        "",
        f"run_id: {run_id}",
        f"source_gate11ac_run_id: {source_manifest.get('run_id', '')}",
        f"source_gate11ac_code_git_commit: {source_manifest.get('code_git_commit', '')}",
        "",
        "## Discipline",
        "",
        "- Gate11AD asks only whether one explicit blocker-resolution marker now exists that instantiates the fixed Gate11AC path",
        "- Gate11AD audits path-instantiation only",
        "- Gate11AD does not judge blocker resolution or residual completion",
        "- Gate11AD does not admit a later source or decide reopening eligibility",
        "",
        "## Status",
        "",
    ]
    lines.extend([f"- {key}: `{status_payload[key]}`" for key in STATUS_FIELDS])
    lines.extend(["", "## Judgment", ""])

    outcome = status_payload["one_explicit_blocker_resolution_marker_path_instantiation_status"]
    if outcome == "present":
        lines.append("- one explicit blocker-resolution marker is now instantiated under the fixed Gate11AC path")
    elif outcome == "not_yet_present":
        lines.append("- the fixed Gate11AC path remains preserved, but no explicit blocker-resolution marker is yet instantiated there")
    elif outcome == "denied":
        lines.append("- the attempted blocker-resolution marker-instantiation read is denied because it would depend on shortcut, inflation, rewrite, graph-wide leap, or worker-side synthesis")
    else:
        lines.append("- the frozen source is incomplete or contradictory for a blocker-resolution marker path-instantiation judgment")
    if status_payload["next_named_blocker"]:
        lines.append(f"- next_named_blocker: `{status_payload['next_named_blocker']}`")
    lines.extend(["", "## Memory Hook", "", "- Gate11AD does not say the blocker is resolved; it asks only whether one explicit blocker-resolution marker now exists that instantiates the fixed Gate11AC path."])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    source_dir = Path(args.gate11ac_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = out_dir.name

    source_manifest = gate9a.read_json(source_dir / gate11ac.DEFAULT_MANIFEST)
    source_status = gate9a.read_json(source_dir / gate11ac.DEFAULT_STATUS)
    source_report = (source_dir / gate11ac.DEFAULT_REPORT).read_text(encoding="utf-8")

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
    gate9a.write_csv(policy_compare_path, ("source_gate11ac_run_id", *STATUS_FIELDS), policy_compare_rows)
    gate9a.write_json(status_path, status_payload)
    gate9a.write_text(report_path, build_report(run_id, source_manifest, policy_compare_rows, status_payload))
    gate9a.write_json(
        manifest_path,
        {
            "run_id": run_id,
            "schema_version": SCHEMA_VERSION,
            "method_id": METHOD_ID,
            "code_git_commit": gate9a.current_git_commit(),
            "source_gate11ac_run_id": str(source_manifest.get("run_id") or ""),
            "source_gate11ac_code_git_commit": str(source_manifest.get("code_git_commit") or ""),
            "inputs": {"gate11ac_dir": str(source_dir)},
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
