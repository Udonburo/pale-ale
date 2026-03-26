#!/usr/bin/env python3
"""Run a Gate11AG named blocker-resolution marker carrier-completion audit on Gate11AF outputs."""

import argparse
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

import run_gate11af_explicit_blocker_resolution_marker_carrier_completion_instantiation_path_audit as gate11af
import run_gate9a_graph_gauge_consumer as gate9a


SCHEMA_VERSION = "gate11ag_named_blocker_resolution_marker_carrier_completion_audit_v1"
METHOD_ID = "gate11ag_named_blocker_resolution_marker_carrier_completion_audit_v1"
DEFAULT_REGISTRY = "named_blocker_resolution_marker_carrier_completion_registry.jsonl"
DEFAULT_POLICY_COMPARE = "named_blocker_resolution_marker_carrier_completion_policy_compare.csv"
DEFAULT_STATUS = "named_blocker_resolution_marker_carrier_completion_status.json"
DEFAULT_REPORT = "gate11ag_named_blocker_resolution_marker_carrier_completion_read.md"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"

SOURCE_PRESERVATION_KEYS = (*gate11af.SOURCE_PRESERVATION_KEYS, "gate11ae_residual_named_state_preservation_status")
CONFIRMED_KEYS = gate11af.CONFIRMED_KEYS
REQUIRED_GATE11AF_STATUS_KEYS = gate11af.STATUS_FIELDS
STATUS_FIELDS = (
    *SOURCE_PRESERVATION_KEYS,
    "gate11af_path_defined_state_preservation_status",
    *CONFIRMED_KEYS,
    "named_blocker_resolution_marker_carrier_condition_preservation_status",
    "explicit_blocker_resolution_marker_status",
    "blocker_resolution_marker_singularity_status",
    "same_source_carrier_completion_status",
    "carrier_completion_boundary_status",
    "named_blocker_resolution_marker_carrier_completion_status",
    "next_named_blocker",
)

REQUIRED_COMPLETION_SURFACES = (
    "one explicit blocker-resolution marker",
    "one explicit later_source_id or later_frozen_run_id",
    "one blocker-resolution marker and only one blocker-resolution marker",
    "one explicit same-source path-attachment status marked completed",
    "one bounded read-prefix declaration for the blocker-resolution marker",
    "one explicit residual completion marker",
    "one explicit admissible later-source presence marker",
    "one declaration marker",
    "one candidate id",
    "one class",
    "one explicit host-failure sentence",
    "matched status, registry, and read surfaces",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Gate11AG named blocker-resolution marker carrier-completion audit "
            "from the frozen Gate11AF path run without widening beyond the fixed Gate11AF line."
        )
    )
    parser.add_argument("--gate11af-dir", required=True)
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
    return any(key not in source_status for key in REQUIRED_GATE11AF_STATUS_KEYS)


def extract_later_source_ids(report_text: str) -> List[str]:
    patterns = [
        re.compile(r"(?im)^\s*residual_completion_later_source_id\s*[:=]\s*([a-z0-9_./-]+)\s*$"),
        re.compile(r"(?im)^\s*residual_completion_later_frozen_run_id\s*[:=]\s*([a-z0-9_./-]+)\s*$"),
    ]
    values: List[str] = []
    for pattern in patterns:
        values.extend(match.group(1).strip().lower() for match in pattern.finditer(report_text))
    return values


def explicit_blocker_resolution_marker_present(report_text: str) -> bool:
    return bool(
        re.search(r"(?im)^\s*residual_completion_blocker_resolution_marker_status\s*[:=]\s*present\s*$", report_text)
    )


def explicit_blocker_resolution_marker_count(report_text: str) -> int:
    return len(
        re.findall(r"(?im)^\s*residual_completion_blocker_resolution_marker_status\s*[:=]\s*present\s*$", report_text)
    )


def explicit_residual_completion_marker_present(report_text: str) -> bool:
    return bool(re.search(r"(?im)^\s*residual_completion_marker_status\s*[:=]\s*present\s*$", report_text))


def extract_bounded_completion_surfaces(report_text: str) -> List[str]:
    pattern = re.compile(r"(?im)^\s*residual_completion_surface\s*[:=]\s*(.+?)\s*$")
    return [match.group(1).strip().lower() for match in pattern.finditer(report_text)]


def same_source_carrier_completion_instantiated(report_text: str) -> bool:
    surfaces = set(extract_bounded_completion_surfaces(report_text))
    same_source_status_completed = bool(
        re.search(r"(?im)^\s*residual_completion_same_source_status\s*[:=]\s*completed\s*$", report_text)
    )
    return (
        same_source_status_completed
        and explicit_residual_completion_marker_present(report_text)
        and all(phrase in surfaces for phrase in REQUIRED_COMPLETION_SURFACES)
    )


def build_registry(source_manifest: Dict[str, Any], status_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    row = {
        "source_gate11af_run_id": str(source_manifest.get("run_id") or ""),
        "source_gate11af_code_git_commit": str(source_manifest.get("code_git_commit") or ""),
    }
    row.update({key: str(status_payload[key]) for key in STATUS_FIELDS})
    return [row]


def build_policy_compare(registry_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"source_gate11af_run_id": str(row["source_gate11af_run_id"]), **{key: str(row[key]) for key in STATUS_FIELDS}} for row in registry_rows]


def build_status_payload(source_manifest: Dict[str, Any], source_status: Dict[str, Any], report_text: str) -> Dict[str, Any]:
    incomplete = source_is_incomplete(source_manifest, source_status, report_text)
    status_payload = {
        key: ("preserved" if source_status_value(source_status, key) == "preserved" else "not_preserved")
        for key in SOURCE_PRESERVATION_KEYS
    }

    source_gate11af_status = source_status_value(source_status, "explicit_blocker_resolution_marker_carrier_completion_instantiation_path_status")
    if source_gate11af_status == "path_defined":
        gate11af_path_defined_state_preservation_status = "preserved"
    elif source_gate11af_status == "deferred":
        gate11af_path_defined_state_preservation_status = "deferred"
    else:
        gate11af_path_defined_state_preservation_status = "not_preserved"
    status_payload["gate11af_path_defined_state_preservation_status"] = gate11af_path_defined_state_preservation_status

    for key in CONFIRMED_KEYS:
        status_payload[key] = "confirmed" if source_status_value(source_status, key) == "confirmed" else "not_confirmed"

    if incomplete:
        named_blocker_resolution_marker_carrier_condition_preservation_status = "deferred"
    elif source_status_value(source_status, "named_residual_marker_carrier_condition_preservation_status") == "preserved":
        named_blocker_resolution_marker_carrier_condition_preservation_status = "preserved"
    else:
        named_blocker_resolution_marker_carrier_condition_preservation_status = "not_preserved"
    status_payload["named_blocker_resolution_marker_carrier_condition_preservation_status"] = named_blocker_resolution_marker_carrier_condition_preservation_status

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

    later_source_ids = extract_later_source_ids(report_text)
    unique_later_source_ids = sorted(set(later_source_ids))
    explicit_blocker_marker_count = explicit_blocker_resolution_marker_count(report_text)

    if incomplete:
        explicit_blocker_resolution_marker_status = "deferred"
    elif explicit_blocker_resolution_marker_present(report_text):
        explicit_blocker_resolution_marker_status = "present"
    else:
        explicit_blocker_resolution_marker_status = "absent"
    status_payload["explicit_blocker_resolution_marker_status"] = explicit_blocker_resolution_marker_status

    if incomplete:
        blocker_resolution_marker_singularity_status = "deferred"
    elif explicit_blocker_resolution_marker_status == "absent":
        blocker_resolution_marker_singularity_status = "none"
    elif explicit_blocker_marker_count == 1:
        blocker_resolution_marker_singularity_status = "single"
    else:
        blocker_resolution_marker_singularity_status = "multiple"
    status_payload["blocker_resolution_marker_singularity_status"] = blocker_resolution_marker_singularity_status

    if incomplete:
        same_source_carrier_completion_status = "deferred"
    elif explicit_blocker_resolution_marker_status == "absent":
        same_source_carrier_completion_status = "not_completed"
    elif not unique_later_source_ids:
        same_source_carrier_completion_status = "not_completed"
    elif len(unique_later_source_ids) > 1:
        same_source_carrier_completion_status = "deferred"
    elif blocker_resolution_marker_singularity_status == "multiple":
        same_source_carrier_completion_status = "deferred"
    elif blocker_resolution_marker_singularity_status != "single":
        same_source_carrier_completion_status = "not_completed"
    elif same_source_carrier_completion_instantiated(report_text):
        same_source_carrier_completion_status = "completed"
    else:
        same_source_carrier_completion_status = "not_completed"
    status_payload["same_source_carrier_completion_status"] = same_source_carrier_completion_status

    if incomplete:
        overall = "deferred"
    elif (
        any(status_payload[key] != "preserved" for key in SOURCE_PRESERVATION_KEYS)
        or gate11af_path_defined_state_preservation_status == "not_preserved"
        or named_blocker_resolution_marker_carrier_condition_preservation_status == "not_preserved"
        or any(status_payload[key] != "confirmed" for key in CONFIRMED_KEYS)
        or carrier_completion_boundary_status == "denied"
    ):
        overall = "denied"
    elif (
        gate11af_path_defined_state_preservation_status == "deferred"
        or named_blocker_resolution_marker_carrier_condition_preservation_status == "deferred"
        or explicit_blocker_resolution_marker_status == "deferred"
        or blocker_resolution_marker_singularity_status == "deferred"
        or same_source_carrier_completion_status == "deferred"
        or carrier_completion_boundary_status == "deferred"
    ):
        overall = "deferred"
    elif (
        explicit_blocker_resolution_marker_status == "present"
        and blocker_resolution_marker_singularity_status == "single"
        and same_source_carrier_completion_status == "completed"
        and carrier_completion_boundary_status == "confirmed"
    ):
        overall = "completed"
    else:
        overall = "not_yet_completed"
    status_payload["named_blocker_resolution_marker_carrier_completion_status"] = overall

    next_named_blocker = ""
    for key in SOURCE_PRESERVATION_KEYS:
        if status_payload[key] == "not_preserved":
            next_named_blocker = key.replace("_preservation_status", "_not_preserved")
            break
    if not next_named_blocker and gate11af_path_defined_state_preservation_status == "not_preserved":
        next_named_blocker = "gate11af_path_defined_state_not_preserved"
    if not next_named_blocker:
        if named_blocker_resolution_marker_carrier_condition_preservation_status == "not_preserved":
            next_named_blocker = "named_blocker_resolution_marker_carrier_condition_not_preserved"
        elif status_payload["broader_trusted_tree_settlement_still_unearned_status"] != "confirmed":
            next_named_blocker = "broader_trusted_tree_settlement_not_explicitly_unearned"
        elif status_payload["operator_admission_still_denied_status"] != "confirmed":
            next_named_blocker = "operator_admission_not_denied"
        elif status_payload["retroactive_reinterpretation_forbidden_status"] != "confirmed":
            next_named_blocker = "retroactive_reinterpretation_pressure"
        elif carrier_completion_boundary_status == "deferred":
            next_named_blocker = "controlling_source_incomplete"
        elif carrier_completion_boundary_status == "denied":
            next_named_blocker = "carrier_completion_boundary_not_intact"
        elif gate11af_path_defined_state_preservation_status == "deferred":
            next_named_blocker = "upstream_carrier_completion_path_deferred"
        elif len(unique_later_source_ids) > 1 or blocker_resolution_marker_singularity_status == "multiple" or same_source_carrier_completion_status == "deferred":
            next_named_blocker = "multiple_candidate_carriers"
        elif explicit_blocker_resolution_marker_status == "absent":
            next_named_blocker = "no_explicit_blocker_resolution_marker"
        elif same_source_carrier_completion_status != "completed":
            next_named_blocker = "same_source_carrier_completion_not_completed"
    status_payload["next_named_blocker"] = next_named_blocker
    return status_payload


def build_report(run_id: str, source_manifest: Dict[str, Any], status_payload: Dict[str, Any]) -> str:
    lines = [
        "# Gate11AG Named Blocker-Resolution Marker Carrier-Completion Read",
        "",
        f"run_id: {run_id}",
        f"source_gate11af_run_id: {source_manifest.get('run_id', '')}",
        f"source_gate11af_code_git_commit: {source_manifest.get('code_git_commit', '')}",
        "",
        "## Discipline",
        "",
        "- Gate11AG asks only whether the named blocker-resolution marker carrier condition now counts as completed under the fixed Gate11AF path",
        "- Gate11AG judges completion only from bounded same-source completion evidence",
        "- Gate11AG does not widen into blocker resolution, residual completion beyond the fixed path, later-source admission, or reopening eligibility",
        "- Gate11AG does not treat path prose, hypothetical examples, or generic read narrative as completion evidence",
        "",
        "## Status",
        "",
    ]
    lines.extend([f"- {key}: `{status_payload[key]}`" for key in STATUS_FIELDS])
    lines.extend(["", "## Judgment", ""])

    outcome = status_payload["named_blocker_resolution_marker_carrier_completion_status"]
    if outcome == "completed":
        lines.append("- the named blocker-resolution marker carrier condition is now completed explicitly under the fixed Gate11AF path")
    elif outcome == "not_yet_completed":
        lines.append("- the fixed Gate11AF path remains preserved, but explicit same-source completion evidence is still absent")
    elif outcome == "denied":
        lines.append("- the attempted carrier-completion read is denied because it would depend on shortcut, inflation, rewrite, graph-wide leap, or worker-side synthesis")
    else:
        lines.append("- the frozen source is incomplete or contradictory for a blocker-resolution marker carrier-completion judgment")
    if status_payload["next_named_blocker"]:
        lines.append(f"- next_named_blocker: `{status_payload['next_named_blocker']}`")
    lines.extend(
        [
            "",
            "## Memory Hook",
            "",
            "- Gate11AG does not say the blocker-resolution marker carrier condition is completed; it asks whether that named condition now actually counts as completed under the fixed Gate11AF path.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    source_dir = Path(args.gate11af_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = out_dir.name

    source_manifest = gate9a.read_json(source_dir / gate11af.DEFAULT_MANIFEST)
    source_status = gate9a.read_json(source_dir / gate11af.DEFAULT_STATUS)
    source_report = (source_dir / gate11af.DEFAULT_REPORT).read_text(encoding="utf-8")

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
    gate9a.write_csv(policy_compare_path, ("source_gate11af_run_id", *STATUS_FIELDS), policy_compare_rows)
    gate9a.write_json(status_path, status_payload)
    gate9a.write_text(report_path, build_report(run_id, source_manifest, status_payload))
    gate9a.write_json(
        manifest_path,
        {
            "run_id": run_id,
            "schema_version": SCHEMA_VERSION,
            "method_id": METHOD_ID,
            "code_git_commit": gate9a.current_git_commit(),
            "source_gate11af_run_id": str(source_manifest.get("run_id") or ""),
            "source_gate11af_code_git_commit": str(source_manifest.get("code_git_commit") or ""),
            "inputs": {"gate11af_dir": str(source_dir)},
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