#!/usr/bin/env python3
"""Run a Gate11BB named blocker-resolution marker carrier-completion blocker-resolution blocker-resolution blocker-resolution blocker-resolution blocker-resolution blocker-resolution blocker-resolution audit on Gate11BA outputs."""

import argparse
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

import run_gate11ah_blocker_resolution_marker_carrier_completion_blocker_audit as gate11ah
import run_gate11ba_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_path_audit as gate11ba
import run_gate9a_graph_gauge_consumer as gate9a


SCHEMA_VERSION = "gate11bb_named_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_audit_v1"
METHOD_ID = "gate11bb_named_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_audit_v1"
DEFAULT_REGISTRY = "named_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_registry.jsonl"
DEFAULT_POLICY_COMPARE = "named_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_policy_compare.csv"
DEFAULT_STATUS = "named_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_status.json"
DEFAULT_REPORT = "gate11bb_named_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_read.md"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"

SOURCE_PRESERVATION_KEYS = (*gate11ba.SOURCE_PRESERVATION_KEYS, "gate11az_blocker_named_state_preservation_status")
CONFIRMED_KEYS = gate11ah.CONFIRMED_KEYS
REQUIRED_GATE11BA_STATUS_KEYS = gate11ba.STATUS_FIELDS
STATUS_FIELDS = (
    *SOURCE_PRESERVATION_KEYS,
    "gate11ba_path_defined_state_preservation_status",
    *CONFIRMED_KEYS,
    "named_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_condition_preservation_status",
    "explicit_blocker_resolution_marker_status",
    "same_source_blocker_resolution_status",
    "blocker_resolution_boundary_status",
    "named_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_status",
    "next_named_blocker",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Gate11BB named blocker-resolution marker carrier-completion blocker-resolution blocker-resolution blocker-resolution blocker-resolution blocker-resolution blocker-resolution blocker-resolution audit "
            "from the frozen Gate11BA path run without converting path definition into resolution by prose."
        )
    )
    parser.add_argument("--gate11ba-dir", required=True)
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
    return any(key not in source_status for key in REQUIRED_GATE11BA_STATUS_KEYS)


def extract_later_source_ids(report_text: str) -> List[str]:
    patterns = [
        re.compile(r"(?im)^\s*(?:blocker_resolution_)?later_source_id\s*[:=]\s*([a-z0-9_./-]+)\s*$"),
        re.compile(r"(?im)^\s*(?:blocker_resolution_)?later_frozen_run_id\s*[:=]\s*([a-z0-9_./-]+)\s*$"),
    ]
    values: List[str] = []
    for pattern in patterns:
        values.extend(match.group(1).strip().lower() for match in pattern.finditer(report_text))
    return values


def extract_explicit_blocker_resolution_markers(report_text: str) -> List[str]:
    pattern = re.compile(r"(?im)^\s*(?:explicit_)?blocker_resolution_marker_status\s*[:=]\s*(present)\s*$")
    return [match.group(1).strip().lower() for match in pattern.finditer(report_text)]


def extract_bounded_resolution_surfaces(report_text: str) -> List[str]:
    pattern = re.compile(r"(?im)^\s*(?:residual_completion|blocker_resolution)_surface\s*[:=]\s*(.+?)\s*$")
    return [match.group(1).strip().lower() for match in pattern.finditer(report_text)]


def explicit_status_equals(report_text: str, field_name: str, expected_value: str) -> bool:
    pattern = re.compile(rf"(?im)^\s*{re.escape(field_name)}\s*[:=]\s*{re.escape(expected_value)}\s*$")
    return bool(pattern.search(report_text))


def explicit_scalar_present(report_text: str, field_name: str) -> bool:
    pattern = re.compile(rf"(?im)^\s*{re.escape(field_name)}\s*[:=]\s*(.+?)\s*$")
    match = pattern.search(report_text)
    return bool(match and match.group(1).strip())


def same_source_blocker_resolution_completed(report_text: str) -> bool:
    surfaces = extract_bounded_resolution_surfaces(report_text)
    required_surface_phrases = (
        "one explicit blocker-resolution marker",
        "one explicit later-source identifier",
        "one blocker-resolution marker and only one blocker-resolution marker",
        "one explicit same-source blocker-resolution status marked resolved",
        "one bounded read-prefix declaration for the blocker-resolution marker",
        "repeated bounded residual_completion_surface rows for the required same-source elements",
        "one explicit residual completion marker",
        "one explicit admissible later-source presence marker",
        "one declaration marker",
        "one candidate id",
        "one class",
        "one explicit host-failure sentence",
        "matched status, registry, and read surfaces",
    )
    later_source_present = explicit_scalar_present(report_text, "later_source_id") or explicit_scalar_present(
        report_text, "later_frozen_run_id"
    )
    explicit_requirements_met = all(
        (
            explicit_status_equals(report_text, "blocker_resolution_marker_status", "present"),
            later_source_present,
            explicit_status_equals(report_text, "same_source_blocker_resolution_status", "resolved"),
            explicit_status_equals(report_text, "bounded_read_prefix_declaration_status", "present"),
            explicit_status_equals(report_text, "residual_completion_marker_status", "present"),
            explicit_status_equals(report_text, "admissible_later_source_presence_status", "present"),
            explicit_status_equals(report_text, "declaration_marker_status", "present"),
            explicit_scalar_present(report_text, "candidate_id"),
            explicit_scalar_present(report_text, "class"),
            explicit_status_equals(report_text, "host_failure_sentence_status", "present"),
            explicit_status_equals(report_text, "matched_status_registry_read_surfaces_status", "matched"),
        )
    )
    return explicit_requirements_met and all(phrase in surfaces for phrase in required_surface_phrases)


def build_registry(source_manifest: Dict[str, Any], status_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    row = {
        "source_gate11ba_run_id": str(source_manifest.get("run_id") or ""),
        "source_gate11ba_code_git_commit": str(source_manifest.get("code_git_commit") or ""),
    }
    row.update({key: str(status_payload[key]) for key in STATUS_FIELDS})
    return [row]


def build_policy_compare(registry_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"source_gate11ba_run_id": str(row["source_gate11ba_run_id"]), **{key: str(row[key]) for key in STATUS_FIELDS}} for row in registry_rows]


def build_status_payload(source_manifest: Dict[str, Any], source_status: Dict[str, Any], report_text: str) -> Dict[str, Any]:
    incomplete = source_is_incomplete(source_manifest, source_status, report_text)
    status_payload = {
        key: ("preserved" if source_status_value(source_status, key) == "preserved" else "not_preserved")
        for key in SOURCE_PRESERVATION_KEYS
    }

    source_gate11ba_status = source_status_value(
        source_status,
        "blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_path_status",
    )
    if source_gate11ba_status == "path_defined":
        gate11ba_path_defined_state_preservation_status = "preserved"
    elif source_gate11ba_status == "deferred":
        gate11ba_path_defined_state_preservation_status = "deferred"
    else:
        gate11ba_path_defined_state_preservation_status = "not_preserved"
    status_payload["gate11ba_path_defined_state_preservation_status"] = gate11ba_path_defined_state_preservation_status

    for key in CONFIRMED_KEYS:
        status_payload[key] = "confirmed" if source_status_value(source_status, key) == "confirmed" else "not_confirmed"

    if incomplete:
        named_blocker_preservation_status = "deferred"
    elif source_status_value(
        source_status,
        "named_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_condition_preservation_status",
    ) == "preserved":
        named_blocker_preservation_status = "preserved"
    else:
        named_blocker_preservation_status = "not_preserved"
    status_payload[
        "named_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_condition_preservation_status"
    ] = named_blocker_preservation_status

    if incomplete:
        blocker_resolution_boundary_status = "deferred"
    elif (
        any(status_payload[key] != "confirmed" for key in CONFIRMED_KEYS)
        or source_status_value(source_status, "blocker_resolution_boundary_status") != "confirmed"
    ):
        blocker_resolution_boundary_status = "denied"
    else:
        blocker_resolution_boundary_status = "confirmed"
    status_payload["blocker_resolution_boundary_status"] = blocker_resolution_boundary_status

    later_source_ids = extract_later_source_ids(report_text)
    unique_later_source_ids = sorted(set(later_source_ids))
    explicit_marker_rows = extract_explicit_blocker_resolution_markers(report_text)

    if incomplete:
        explicit_blocker_resolution_marker_status = "deferred"
    elif len(explicit_marker_rows) != 1:
        explicit_blocker_resolution_marker_status = "deferred" if len(explicit_marker_rows) > 1 else "absent"
    else:
        explicit_blocker_resolution_marker_status = "present"
    status_payload["explicit_blocker_resolution_marker_status"] = explicit_blocker_resolution_marker_status

    if incomplete:
        same_source_blocker_resolution_status = "deferred"
    elif len(explicit_marker_rows) > 1 or len(unique_later_source_ids) > 1:
        same_source_blocker_resolution_status = "deferred"
    elif (
        explicit_blocker_resolution_marker_status == "present"
        and len(unique_later_source_ids) == 1
        and same_source_blocker_resolution_completed(report_text)
    ):
        same_source_blocker_resolution_status = "resolved"
    else:
        same_source_blocker_resolution_status = "not_resolved"
    status_payload["same_source_blocker_resolution_status"] = same_source_blocker_resolution_status

    if incomplete:
        overall = "deferred"
    elif (
        any(status_payload[key] != "preserved" for key in SOURCE_PRESERVATION_KEYS)
        or gate11ba_path_defined_state_preservation_status == "not_preserved"
        or any(status_payload[key] != "confirmed" for key in CONFIRMED_KEYS)
        or blocker_resolution_boundary_status == "denied"
    ):
        overall = "denied"
    elif (
        gate11ba_path_defined_state_preservation_status == "deferred"
        or named_blocker_preservation_status == "deferred"
        or explicit_blocker_resolution_marker_status == "deferred"
        or same_source_blocker_resolution_status == "deferred"
        or blocker_resolution_boundary_status == "deferred"
    ):
        overall = "deferred"
    elif (
        named_blocker_preservation_status == "preserved"
        and explicit_blocker_resolution_marker_status == "present"
        and same_source_blocker_resolution_status == "resolved"
        and blocker_resolution_boundary_status == "confirmed"
    ):
        overall = "resolved"
    else:
        overall = "not_yet_resolved"
    status_payload[
        "named_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_status"
    ] = overall

    next_named_blocker = ""
    for key in SOURCE_PRESERVATION_KEYS:
        if status_payload[key] == "not_preserved":
            next_named_blocker = key.replace("_preservation_status", "_not_preserved")
            break
    if not next_named_blocker and gate11ba_path_defined_state_preservation_status == "not_preserved":
        next_named_blocker = "gate11ba_path_defined_state_not_preserved"
    if not next_named_blocker:
        if status_payload["broader_trusted_tree_settlement_still_unearned_status"] != "confirmed":
            next_named_blocker = "broader_trusted_tree_settlement_not_explicitly_unearned"
        elif status_payload["operator_admission_still_denied_status"] != "confirmed":
            next_named_blocker = "operator_admission_not_denied"
        elif status_payload["retroactive_reinterpretation_forbidden_status"] != "confirmed":
            next_named_blocker = "retroactive_reinterpretation_pressure"
        elif blocker_resolution_boundary_status == "deferred":
            next_named_blocker = "controlling_source_incomplete"
        elif blocker_resolution_boundary_status == "denied":
            next_named_blocker = "blocker_resolution_boundary_not_intact"
        elif gate11ba_path_defined_state_preservation_status == "deferred":
            next_named_blocker = "upstream_path_definition_deferred"
        elif named_blocker_preservation_status != "preserved":
            next_named_blocker = "named_blocker_not_preserved"
        elif explicit_blocker_resolution_marker_status == "deferred" or same_source_blocker_resolution_status == "deferred":
            next_named_blocker = "multiple_candidate_resolutions"
        elif explicit_blocker_resolution_marker_status != "present":
            next_named_blocker = "no_explicit_blocker_resolution_marker"
        elif same_source_blocker_resolution_status != "resolved":
            next_named_blocker = "same_source_blocker_resolution_not_completed"
    status_payload["next_named_blocker"] = next_named_blocker
    return status_payload


def build_report(
    run_id: str,
    source_manifest: Dict[str, Any],
    policy_compare_rows: Sequence[Dict[str, Any]],
    status_payload: Dict[str, Any],
) -> str:
    lines = [
        "# Gate11BB Named Blocker-Resolution Marker Carrier-Completion Blocker-Resolution Blocker-Resolution Blocker-Resolution Blocker-Resolution Blocker-Resolution Blocker-Resolution Blocker-Resolution Read",
        "",
        f"run_id: {run_id}",
        f"source_gate11ba_run_id: {source_manifest.get('run_id', '')}",
        f"source_gate11ba_code_git_commit: {source_manifest.get('code_git_commit', '')}",
        "",
        "## Discipline",
        "",
        "- Gate11BB asks only whether the named blocker-resolution marker carrier-completion blocker-resolution blocker-resolution blocker-resolution blocker-resolution blocker-resolution blocker-resolution blocker now actually counts as resolved under the fixed Gate11BA path",
        "- Gate11BB resolves only from bounded same-source evidence",
        "- Gate11BB does not let Gate11BA path-definition prose, hypotheticals, or summaries count as blocker-resolution evidence",
        "- Gate11BB does not widen into blocker-resolution marker carrier-completion blocker-resolution blocker path judgment, blocker-resolution marker carrier-completion blocker-resolution blocker judgment, blocker-resolution marker carrier-completion blocker-resolution path judgment, blocker-resolution marker carrier-completion blocker-resolution judgment, blocker-resolution marker carrier-completion judgment, blocker-resolution judgment, residual completion judgment, later-source admission, explicit-presence judgment, candidate declaration, reopening eligibility, or operator reopening",
        "",
        "## Source Summary",
        "",
        "| source_gate11ba_run_id | " + " | ".join(STATUS_FIELDS) + " |",
        "|---|" + "---|" * len(STATUS_FIELDS),
    ]
    for row in policy_compare_rows:
        lines.append("| " + " | ".join([str(row["source_gate11ba_run_id"]), *[str(row[key]) for key in STATUS_FIELDS]]) + " |")
    lines.extend(["", "## Status", ""])
    lines.extend([f"- {key}: `{status_payload[key]}`" for key in STATUS_FIELDS])
    lines.extend(["", "## Judgment", ""])

    outcome = status_payload[
        "named_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_status"
    ]
    if outcome == "resolved":
        lines.append("- the fixed Gate11BA path is now backed by bounded same-source blocker-resolution evidence, so the named blocker counts as resolved")
    elif outcome == "not_yet_resolved":
        lines.append("- the fixed Gate11BA path remains only a path definition; the named blocker is not yet resolved on bounded same-source evidence")
    elif outcome == "denied":
        lines.append("- the attempted blocker-resolution read is denied because it would depend on shortcut, inflation, rewrite, graph-wide leap, or worker-side synthesis")
    else:
        lines.append("- the frozen source is incomplete or contradictory for a blocker-resolution judgment")
    if status_payload["next_named_blocker"]:
        lines.append(f"- next_named_blocker: `{status_payload['next_named_blocker']}`")
    lines.extend(
        [
            "",
            "## Memory Hook",
            "",
            "- Gate11BB does not let Gate11BA path-definition prose earn resolved; only bounded same-source blocker-resolution evidence can do that.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    source_dir = Path(args.gate11ba_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = out_dir.name

    source_manifest = gate9a.read_json(source_dir / gate11ba.DEFAULT_MANIFEST)
    source_status = gate9a.read_json(source_dir / gate11ba.DEFAULT_STATUS)
    source_report = (source_dir / gate11ba.DEFAULT_REPORT).read_text(encoding="utf-8")

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
    gate9a.write_csv(policy_compare_path, ("source_gate11ba_run_id", *STATUS_FIELDS), policy_compare_rows)
    gate9a.write_json(status_path, status_payload)
    gate9a.write_text(report_path, build_report(run_id, source_manifest, policy_compare_rows, status_payload))
    gate9a.write_json(
        manifest_path,
        {
            "run_id": run_id,
            "schema_version": SCHEMA_VERSION,
            "method_id": METHOD_ID,
            "code_git_commit": gate9a.current_git_commit(),
            "source_gate11ba_run_id": str(source_manifest.get("run_id") or ""),
            "source_gate11ba_code_git_commit": str(source_manifest.get("code_git_commit") or ""),
            "inputs": {"gate11ba_dir": str(source_dir)},
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