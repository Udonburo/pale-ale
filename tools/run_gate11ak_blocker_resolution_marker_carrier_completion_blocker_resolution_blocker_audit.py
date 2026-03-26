#!/usr/bin/env python3
"""Run a Gate11AK blocker-resolution marker carrier-completion blocker-resolution blocker audit on Gate11AJ outputs."""

import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Sequence

import run_gate11ah_blocker_resolution_marker_carrier_completion_blocker_audit as gate11ah
import run_gate11aj_named_blocker_resolution_marker_carrier_completion_blocker_resolution_audit as gate11aj
import run_gate9a_graph_gauge_consumer as gate9a


SCHEMA_VERSION = "gate11ak_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_audit_v1"
METHOD_ID = "gate11ak_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_audit_v1"
DEFAULT_REGISTRY = "blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_registry.jsonl"
DEFAULT_POLICY_COMPARE = "blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_policy_compare.csv"
DEFAULT_STATUS = "blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_status.json"
DEFAULT_REPORT = "gate11ak_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_read.md"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"

SOURCE_PRESERVATION_KEYS = (*gate11aj.SOURCE_PRESERVATION_KEYS, "gate11ai_path_defined_state_preservation_status")
CONFIRMED_KEYS = gate11ah.CONFIRMED_KEYS
REQUIRED_GATE11AJ_STATUS_KEYS = gate11aj.STATUS_FIELDS
STATUS_FIELDS = (
    *SOURCE_PRESERVATION_KEYS,
    "gate11aj_not_yet_resolved_state_preservation_status",
    *CONFIRMED_KEYS,
    "named_blocker_resolution_marker_carrier_completion_blocker_resolution_condition_preservation_status",
    "explicit_blocker_resolution_marker_blocker_status",
    "same_source_blocker_resolution_blocker_status",
    "blocker_resolution_blocker_boundary_status",
    "blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_status",
    "next_named_blocker",
)

KNOWN_BLOCKERS = {
    "no_explicit_blocker_resolution_marker",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Gate11AK blocker-resolution marker carrier-completion blocker-resolution blocker audit "
            "from the frozen Gate11AJ resolution run without widening into resolution, admission, or reopening."
        )
    )
    parser.add_argument("--gate11aj-dir", required=True)
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
    return any(key not in source_status for key in REQUIRED_GATE11AJ_STATUS_KEYS)


def blocker_sentence(blocker: str) -> str:
    mapping = {
        "no_explicit_blocker_resolution_marker": (
            "the blocker-resolution marker carrier-completion blocker-resolution blocker is named narrowly: no explicit blocker-resolution marker still blocks resolution under the fixed Gate11AJ line"
        ),
    }
    return mapping.get(blocker, "the blocker-resolution marker carrier-completion blocker-resolution blocker is not yet named narrowly enough")


def build_registry(source_manifest: Dict[str, Any], status_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    row = {
        "source_gate11aj_run_id": str(source_manifest.get("run_id") or ""),
        "source_gate11aj_code_git_commit": str(source_manifest.get("code_git_commit") or ""),
    }
    row.update({key: str(status_payload[key]) for key in STATUS_FIELDS})
    return [row]


def build_policy_compare(registry_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"source_gate11aj_run_id": str(row["source_gate11aj_run_id"]), **{key: str(row[key]) for key in STATUS_FIELDS}} for row in registry_rows]


def build_status_payload(source_manifest: Dict[str, Any], source_status: Dict[str, Any], report_text: str) -> Dict[str, Any]:
    incomplete = source_is_incomplete(source_manifest, source_status, report_text)
    status_payload = {
        key: ("preserved" if source_status_value(source_status, key) == "preserved" else "not_preserved")
        for key in SOURCE_PRESERVATION_KEYS
    }

    source_gate11aj_status = source_status_value(
        source_status, "named_blocker_resolution_marker_carrier_completion_blocker_resolution_status"
    )
    if source_gate11aj_status == "not_yet_resolved":
        gate11aj_not_yet_resolved_state_preservation_status = "preserved"
    elif source_gate11aj_status == "deferred":
        gate11aj_not_yet_resolved_state_preservation_status = "deferred"
    else:
        gate11aj_not_yet_resolved_state_preservation_status = "not_preserved"
    status_payload["gate11aj_not_yet_resolved_state_preservation_status"] = gate11aj_not_yet_resolved_state_preservation_status

    for key in CONFIRMED_KEYS:
        status_payload[key] = "confirmed" if source_status_value(source_status, key) == "confirmed" else "not_confirmed"

    if incomplete:
        named_condition = "deferred"
    elif source_status_value(source_status, "named_blocker_preservation_status") == "preserved":
        named_condition = "preserved"
    else:
        named_condition = "not_preserved"
    status_payload["named_blocker_resolution_marker_carrier_completion_blocker_resolution_condition_preservation_status"] = named_condition

    if incomplete:
        explicit_blocker_resolution_marker_blocker_status = "deferred"
    elif source_status_value(source_status, "explicit_blocker_resolution_marker_status") == "absent":
        explicit_blocker_resolution_marker_blocker_status = "named"
    else:
        explicit_blocker_resolution_marker_blocker_status = "not_named"
    status_payload["explicit_blocker_resolution_marker_blocker_status"] = explicit_blocker_resolution_marker_blocker_status

    if incomplete:
        same_source_blocker_resolution_blocker_status = "deferred"
    elif source_status_value(source_status, "same_source_blocker_resolution_status") == "not_resolved":
        same_source_blocker_resolution_blocker_status = "named"
    elif source_status_value(source_status, "same_source_blocker_resolution_status") == "deferred":
        same_source_blocker_resolution_blocker_status = "deferred"
    else:
        same_source_blocker_resolution_blocker_status = "not_named"
    status_payload["same_source_blocker_resolution_blocker_status"] = same_source_blocker_resolution_blocker_status

    if incomplete:
        blocker_resolution_blocker_boundary_status = "deferred"
    elif (
        any(status_payload[key] != "confirmed" for key in CONFIRMED_KEYS)
        or source_status_value(source_status, "blocker_resolution_boundary_status") != "confirmed"
    ):
        blocker_resolution_blocker_boundary_status = "denied"
    else:
        blocker_resolution_blocker_boundary_status = "confirmed"
    status_payload["blocker_resolution_blocker_boundary_status"] = blocker_resolution_blocker_boundary_status

    source_blocker = source_status_value(source_status, "next_named_blocker")
    blocker_explicitly_named = source_blocker in KNOWN_BLOCKERS

    if incomplete:
        overall = "deferred"
    elif (
        any(status_payload[key] != "preserved" for key in SOURCE_PRESERVATION_KEYS)
        or gate11aj_not_yet_resolved_state_preservation_status == "not_preserved"
        or named_condition == "not_preserved"
        or any(status_payload[key] != "confirmed" for key in CONFIRMED_KEYS)
        or blocker_resolution_blocker_boundary_status == "denied"
    ):
        overall = "denied"
    elif (
        gate11aj_not_yet_resolved_state_preservation_status == "deferred"
        or named_condition == "deferred"
        or explicit_blocker_resolution_marker_blocker_status == "deferred"
        or same_source_blocker_resolution_blocker_status == "deferred"
        or blocker_resolution_blocker_boundary_status == "deferred"
    ):
        overall = "deferred"
    elif (
        named_condition == "preserved"
        and explicit_blocker_resolution_marker_blocker_status == "named"
        and same_source_blocker_resolution_blocker_status == "named"
        and blocker_resolution_blocker_boundary_status == "confirmed"
        and blocker_explicitly_named
    ):
        overall = "blocker_named"
    else:
        overall = "not_yet_named"
    status_payload["blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_status"] = overall

    next_named_blocker = ""
    for key in SOURCE_PRESERVATION_KEYS:
        if status_payload[key] == "not_preserved":
            next_named_blocker = key.replace("_preservation_status", "_not_preserved")
            break
    if not next_named_blocker and gate11aj_not_yet_resolved_state_preservation_status == "not_preserved":
        next_named_blocker = "gate11aj_not_yet_resolved_state_not_preserved"
    if not next_named_blocker:
        if named_condition == "not_preserved":
            next_named_blocker = "named_blocker_resolution_marker_carrier_completion_blocker_resolution_condition_not_preserved"
        elif status_payload["broader_trusted_tree_settlement_still_unearned_status"] != "confirmed":
            next_named_blocker = "broader_trusted_tree_settlement_not_explicitly_unearned"
        elif status_payload["operator_admission_still_denied_status"] != "confirmed":
            next_named_blocker = "operator_admission_not_denied"
        elif status_payload["retroactive_reinterpretation_forbidden_status"] != "confirmed":
            next_named_blocker = "retroactive_reinterpretation_pressure"
        elif blocker_resolution_blocker_boundary_status == "deferred":
            next_named_blocker = "controlling_source_incomplete"
        elif blocker_resolution_blocker_boundary_status == "denied":
            next_named_blocker = "blocker_resolution_boundary_not_intact"
        elif gate11aj_not_yet_resolved_state_preservation_status == "deferred":
            next_named_blocker = "upstream_resolution_judgment_deferred"
        elif overall == "not_yet_named":
            next_named_blocker = "no_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_explicitly_named"
        else:
            next_named_blocker = source_blocker
    status_payload["next_named_blocker"] = next_named_blocker
    return status_payload


def build_report(run_id: str, source_manifest: Dict[str, Any], status_payload: Dict[str, Any]) -> str:
    lines = [
        "# Gate11AK Blocker-Resolution Marker Carrier-Completion Blocker-Resolution Blocker Read",
        "",
        f"run_id: {run_id}",
        f"source_gate11aj_run_id: {source_manifest.get('run_id', '')}",
        f"source_gate11aj_code_git_commit: {source_manifest.get('code_git_commit', '')}",
        "",
        "## Discipline",
        "",
        "- Gate11AK asks only which blocker-resolution marker carrier-completion blocker-resolution blocker still blocks resolution under the fixed Gate11AJ line",
        "- Gate11AK names blockers only",
        "- Gate11AK does not convert not-yet-resolved into resolution",
        "- Gate11AK does not widen into blocker-resolution marker carrier completion, blocker resolution, residual completion, later-source admission, reopening eligibility, or operator reopening",
        "- Gate11AK does not inherit worker-side interpretations from path prose or generic narrative",
        "",
        "## Status",
        "",
    ]
    lines.extend([f"- {key}: `{status_payload[key]}`" for key in STATUS_FIELDS])
    lines.extend(["", "## Judgment", ""])

    outcome = status_payload["blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_status"]
    if outcome == "blocker_named":
        lines.append(f"- {blocker_sentence(status_payload['next_named_blocker'])}")
    elif outcome == "not_yet_named":
        lines.append("- the fixed Gate11AJ line still does not name the blocker-resolution marker carrier-completion blocker-resolution blocker narrowly enough")
    elif outcome == "denied":
        lines.append("- the attempted blocker read is denied because it would depend on shortcut, inflation, rewrite, graph-wide leap, or worker-side synthesis")
    else:
        lines.append("- the frozen source is incomplete or contradictory for a blocker-resolution marker carrier-completion blocker-resolution blocker judgment")
    if status_payload["next_named_blocker"]:
        lines.append(f"- next_named_blocker: `{status_payload['next_named_blocker']}`")
    lines.extend(
        [
            "",
            "## Memory Hook",
            "",
            "- Gate11AK does not say the named blocker is resolved; it asks which blocker-resolution marker carrier-completion blocker-resolution blocker still blocks resolution under the fixed Gate11AJ line.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    source_dir = Path(args.gate11aj_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = out_dir.name

    source_manifest = gate9a.read_json(source_dir / gate11aj.DEFAULT_MANIFEST)
    source_status = gate9a.read_json(source_dir / gate11aj.DEFAULT_STATUS)
    source_report = (source_dir / gate11aj.DEFAULT_REPORT).read_text(encoding="utf-8")

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
    gate9a.write_csv(policy_compare_path, ("source_gate11aj_run_id", *STATUS_FIELDS), policy_compare_rows)
    gate9a.write_json(status_path, status_payload)
    gate9a.write_text(report_path, build_report(run_id, source_manifest, status_payload))
    gate9a.write_json(
        manifest_path,
        {
            "run_id": run_id,
            "schema_version": SCHEMA_VERSION,
            "method_id": METHOD_ID,
            "code_git_commit": gate9a.current_git_commit(),
            "source_gate11aj_run_id": str(source_manifest.get("run_id") or ""),
            "source_gate11aj_code_git_commit": str(source_manifest.get("code_git_commit") or ""),
            "inputs": {"gate11aj_dir": str(source_dir)},
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