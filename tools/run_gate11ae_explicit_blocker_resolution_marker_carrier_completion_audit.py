#!/usr/bin/env python3
"""Run a Gate11AE explicit blocker-resolution marker carrier-completion audit on Gate11AD outputs."""

import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Sequence

import run_gate11ad_one_explicit_blocker_resolution_marker_path_instantiation_audit as gate11ad
import run_gate9a_graph_gauge_consumer as gate9a


SCHEMA_VERSION = "gate11ae_explicit_blocker_resolution_marker_carrier_completion_audit_v1"
METHOD_ID = "gate11ae_explicit_blocker_resolution_marker_carrier_completion_audit_v1"
DEFAULT_REGISTRY = "explicit_blocker_resolution_marker_carrier_completion_registry.jsonl"
DEFAULT_POLICY_COMPARE = "explicit_blocker_resolution_marker_carrier_completion_policy_compare.csv"
DEFAULT_STATUS = "explicit_blocker_resolution_marker_carrier_completion_status.json"
DEFAULT_REPORT = "gate11ae_explicit_blocker_resolution_marker_carrier_completion_read.md"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"

SOURCE_PRESERVATION_KEYS = (*gate11ad.SOURCE_PRESERVATION_KEYS, "gate11ac_path_defined_state_preservation_status")
CONFIRMED_KEYS = gate11ad.CONFIRMED_KEYS
REQUIRED_GATE11AD_STATUS_KEYS = gate11ad.STATUS_FIELDS
STATUS_FIELDS = (
    *SOURCE_PRESERVATION_KEYS,
    "gate11ad_not_yet_present_state_preservation_status",
    *CONFIRMED_KEYS,
    "explicit_marker_carrier_completion_status",
    "marker_singularity_carrier_completion_status",
    "same_source_path_attachment_carrier_completion_status",
    "carrier_completion_boundary_status",
    "explicit_blocker_resolution_marker_carrier_completion_status",
    "next_named_blocker",
)

KNOWN_RESIDUAL_BLOCKERS = {
    "no_explicit_blocker_resolution_marker",
    "same_source_marker_path_not_instantiated",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Gate11AE explicit blocker-resolution marker carrier-completion audit "
            "from the frozen Gate11AD path-instantiation run without deciding marker existence or blocker resolution."
        )
    )
    parser.add_argument("--gate11ad-dir", required=True)
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
    return any(key not in source_status for key in REQUIRED_GATE11AD_STATUS_KEYS)


def blocker_sentence(blocker: str) -> str:
    mapping = {
        "no_explicit_blocker_resolution_marker": (
            "the residual carrier condition is named narrowly: no explicit blocker-resolution marker is yet carried under the fixed Gate11AD line"
        ),
        "same_source_marker_path_not_instantiated": (
            "the residual carrier condition is named narrowly: same-source path attachment to the fixed Gate11AC line is still incomplete"
        ),
    }
    return mapping.get(blocker, "the explicit blocker-resolution marker carrier condition is not yet named narrowly enough")


def build_registry(source_manifest: Dict[str, Any], status_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    row = {
        "source_gate11ad_run_id": str(source_manifest.get("run_id") or ""),
        "source_gate11ad_code_git_commit": str(source_manifest.get("code_git_commit") or ""),
    }
    row.update({key: str(status_payload[key]) for key in STATUS_FIELDS})
    return [row]


def build_policy_compare(registry_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"source_gate11ad_run_id": str(row["source_gate11ad_run_id"]), **{key: str(row[key]) for key in STATUS_FIELDS}} for row in registry_rows]


def build_status_payload(source_manifest: Dict[str, Any], source_status: Dict[str, Any], report_text: str) -> Dict[str, Any]:
    incomplete = source_is_incomplete(source_manifest, source_status, report_text)
    status_payload = {
        key: ("preserved" if source_status_value(source_status, key) == "preserved" else "not_preserved")
        for key in SOURCE_PRESERVATION_KEYS
    }

    source_gate11ad_status = source_status_value(source_status, "one_explicit_blocker_resolution_marker_path_instantiation_status")
    if source_gate11ad_status == "not_yet_present":
        gate11ad_not_yet_present_state_preservation_status = "preserved"
    elif source_gate11ad_status == "deferred":
        gate11ad_not_yet_present_state_preservation_status = "deferred"
    else:
        gate11ad_not_yet_present_state_preservation_status = "not_preserved"
    status_payload["gate11ad_not_yet_present_state_preservation_status"] = gate11ad_not_yet_present_state_preservation_status

    for key in CONFIRMED_KEYS:
        status_payload[key] = "confirmed" if source_status_value(source_status, key) == "confirmed" else "not_confirmed"

    if incomplete:
        carrier_completion_boundary_status = "deferred"
    elif (
        any(status_payload[key] != "confirmed" for key in CONFIRMED_KEYS)
        or source_status_value(source_status, "blocker_resolution_marker_boundary_status") != "confirmed"
    ):
        carrier_completion_boundary_status = "denied"
    else:
        carrier_completion_boundary_status = "confirmed"
    status_payload["carrier_completion_boundary_status"] = carrier_completion_boundary_status

    if incomplete:
        explicit_marker_carrier_completion_status = "deferred"
    elif source_status_value(source_status, "explicit_blocker_resolution_marker_status") == "present":
        explicit_marker_carrier_completion_status = "complete"
    else:
        explicit_marker_carrier_completion_status = "missing"
    status_payload["explicit_marker_carrier_completion_status"] = explicit_marker_carrier_completion_status

    if incomplete:
        marker_singularity_carrier_completion_status = "deferred"
    elif source_status_value(source_status, "blocker_resolution_marker_singularity_status") == "single":
        marker_singularity_carrier_completion_status = "complete"
    else:
        marker_singularity_carrier_completion_status = "missing"
    status_payload["marker_singularity_carrier_completion_status"] = marker_singularity_carrier_completion_status

    if incomplete:
        same_source_path_attachment_carrier_completion_status = "deferred"
    elif source_status_value(source_status, "same_source_marker_path_attachment_status") == "instantiated":
        same_source_path_attachment_carrier_completion_status = "complete"
    else:
        same_source_path_attachment_carrier_completion_status = "missing"
    status_payload["same_source_path_attachment_carrier_completion_status"] = same_source_path_attachment_carrier_completion_status

    source_blocker = source_status_value(source_status, "next_named_blocker")
    residual_condition_explicitly_named = source_blocker in KNOWN_RESIDUAL_BLOCKERS

    if incomplete:
        overall = "deferred"
    elif (
        any(status_payload[key] != "preserved" for key in SOURCE_PRESERVATION_KEYS)
        or gate11ad_not_yet_present_state_preservation_status == "not_preserved"
        or any(status_payload[key] != "confirmed" for key in CONFIRMED_KEYS)
        or carrier_completion_boundary_status == "denied"
    ):
        overall = "denied"
    elif (
        gate11ad_not_yet_present_state_preservation_status == "deferred"
        or explicit_marker_carrier_completion_status == "deferred"
        or marker_singularity_carrier_completion_status == "deferred"
        or same_source_path_attachment_carrier_completion_status == "deferred"
        or carrier_completion_boundary_status == "deferred"
    ):
        overall = "deferred"
    elif residual_condition_explicitly_named:
        overall = "residual_named"
    else:
        overall = "not_yet_named"
    status_payload["explicit_blocker_resolution_marker_carrier_completion_status"] = overall

    next_named_blocker = ""
    for key in SOURCE_PRESERVATION_KEYS:
        if status_payload[key] == "not_preserved":
            next_named_blocker = key.replace("_preservation_status", "_not_preserved")
            break
    if not next_named_blocker and gate11ad_not_yet_present_state_preservation_status == "not_preserved":
        next_named_blocker = "gate11ad_not_yet_present_state_not_preserved"
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
        elif gate11ad_not_yet_present_state_preservation_status == "deferred":
            next_named_blocker = source_blocker or "upstream_marker_path_instantiation_deferred"
        elif overall == "not_yet_named":
            next_named_blocker = "no_residual_carrier_condition_explicitly_named"
        else:
            next_named_blocker = source_blocker
    status_payload["next_named_blocker"] = next_named_blocker
    return status_payload


def build_report(
    run_id: str,
    source_manifest: Dict[str, Any],
    status_payload: Dict[str, Any],
) -> str:
    lines = [
        "# Gate11AE Explicit Blocker-Resolution Marker Carrier-Completion Read",
        "",
        f"run_id: {run_id}",
        f"source_gate11ad_run_id: {source_manifest.get('run_id', '')}",
        f"source_gate11ad_code_git_commit: {source_manifest.get('code_git_commit', '')}",
        "",
        "## Discipline",
        "",
        "- Gate11AE asks only which explicit blocker-resolution marker carrier condition still blocks one marker from being actually present under the fixed Gate11AD line",
        "- Gate11AE names residual carrier conditions only",
        "- Gate11AE does not instantiate a marker, resolve the blocker, or judge residual completion",
        "- Gate11AE does not admit a later source, decide explicit presence, or reopen operator admission",
        "",
        "## Status",
        "",
    ]
    lines.extend([f"- {key}: `{status_payload[key]}`" for key in STATUS_FIELDS])
    lines.extend(["", "## Judgment", ""])

    outcome = status_payload["explicit_blocker_resolution_marker_carrier_completion_status"]
    if outcome == "residual_named":
        lines.append(f"- {blocker_sentence(status_payload['next_named_blocker'])}")
    elif outcome == "not_yet_named":
        lines.append("- the fixed Gate11AD line still does not name the remaining explicit blocker-resolution marker carrier condition narrowly enough")
    elif outcome == "denied":
        lines.append("- the attempted carrier-completion read is denied because it would depend on shortcut, inflation, rewrite, graph-wide leap, or worker-side synthesis")
    else:
        lines.append("- the frozen source is incomplete or contradictory for an explicit blocker-resolution marker carrier-completion judgment")
    if status_payload["next_named_blocker"]:
        lines.append(f"- next_named_blocker: `{status_payload['next_named_blocker']}`")

    lines.extend(
        [
            "",
            "## Memory Hook",
            "",
            "- Gate11AE does not say a blocker-resolution marker exists; it asks which explicit blocker-resolution marker carrier condition still blocks one marker from being actually present under the fixed Gate11AD line.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    source_dir = Path(args.gate11ad_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = out_dir.name

    source_manifest = gate9a.read_json(source_dir / gate11ad.DEFAULT_MANIFEST)
    source_status = gate9a.read_json(source_dir / gate11ad.DEFAULT_STATUS)
    source_report = (source_dir / gate11ad.DEFAULT_REPORT).read_text(encoding="utf-8")

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
    gate9a.write_csv(policy_compare_path, ("source_gate11ad_run_id", *STATUS_FIELDS), policy_compare_rows)
    gate9a.write_json(status_path, status_payload)
    gate9a.write_text(report_path, build_report(run_id, source_manifest, status_payload))
    gate9a.write_json(
        manifest_path,
        {
            "run_id": run_id,
            "schema_version": SCHEMA_VERSION,
            "method_id": METHOD_ID,
            "code_git_commit": gate9a.current_git_commit(),
            "source_gate11ad_run_id": str(source_manifest.get("run_id") or ""),
            "source_gate11ad_code_git_commit": str(source_manifest.get("code_git_commit") or ""),
            "inputs": {"gate11ad_dir": str(source_dir)},
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