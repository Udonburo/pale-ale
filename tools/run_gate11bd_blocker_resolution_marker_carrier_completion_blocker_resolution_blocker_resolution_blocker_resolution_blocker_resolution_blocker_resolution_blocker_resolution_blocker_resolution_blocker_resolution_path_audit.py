#!/usr/bin/env python3
"""Run a Gate11BD blocker-resolution marker carrier-completion blocker-resolution blocker-resolution blocker-resolution blocker-resolution blocker-resolution blocker-resolution blocker-resolution blocker-resolution path audit on Gate11BC outputs."""

import argparse
import hashlib
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate11ah_blocker_resolution_marker_carrier_completion_blocker_audit as gate11ah
import run_gate11bc_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_audit as gate11bc
import run_gate9a_graph_gauge_consumer as gate9a


SCHEMA_VERSION = "gate11bd_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_path_audit_v1"
METHOD_ID = "gate11bd_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_path_audit_v1"
DEFAULT_REGISTRY = "blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_path_registry.jsonl"
DEFAULT_POLICY_COMPARE = "blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_path_policy_compare.csv"
DEFAULT_STATUS = "blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_path_status.json"
DEFAULT_REPORT = "gate11bd_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_path_read.md"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"

SOURCE_PRESERVATION_KEYS = (*gate11bc.SOURCE_PRESERVATION_KEYS, "gate11bb_not_yet_resolved_state_preservation_status")
CONFIRMED_KEYS = gate11ah.CONFIRMED_KEYS
REQUIRED_GATE11BC_STATUS_KEYS = gate11bc.STATUS_FIELDS
PATH_DEFINABLE_BLOCKERS = {
    "no_explicit_blocker_resolution_marker",
}
STATUS_FIELDS = (
    *SOURCE_PRESERVATION_KEYS,
    "gate11bc_blocker_named_state_preservation_status",
    *CONFIRMED_KEYS,
    "named_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_condition_preservation_status",
    "minimum_same_source_blocker_resolution_rule_status",
    "bounded_read_prefix_resolution_requirement_status",
    "blocker_resolution_boundary_status",
    "blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_path_status",
    "next_named_blocker",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Gate11BD blocker-resolution marker carrier-completion blocker-resolution blocker-resolution blocker-resolution blocker-resolution blocker-resolution blocker-resolution blocker_resolution blocker-resolution path audit "
            "from the frozen Gate11BC blocker-naming run without converting path definition into actual blocker resolution."
        )
    )
    parser.add_argument("--gate11bc-dir", required=True)
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
    return any(key not in source_status for key in REQUIRED_GATE11BC_STATUS_KEYS)


def blocker_sentence(blocker: str) -> str:
    mapping = {
        "no_explicit_blocker_resolution_marker": (
            "the minimum honest path is fixed narrowly: one explicit blocker-resolution marker, one explicit later_source_id or later_frozen_run_id, one blocker-resolution marker and only one blocker-resolution marker, one explicit same-source blocker-resolution status marked resolved, one bounded read-prefix declaration for the blocker-resolution marker, repeated bounded residual_completion_surface rows for the required same-source elements, one explicit residual completion marker, one explicit admissible later-source presence marker, one declaration marker, one candidate id, one class, one explicit host-failure sentence, and matched status, registry, and read surfaces"
        ),
        "minimum_same_source_blocker_resolution_rule_not_fixed": (
            "the named blocker-resolution marker carrier-completion blocker-resolution blocker-resolution blocker-resolution blocker-resolution blocker-resolution blocker-resolution blocker_resolution blocker remains preserved, but the minimum same-source blocker-resolution rule is not yet fixed narrowly enough"
        ),
        "bounded_read_prefix_resolution_requirement_not_fixed": (
            "the named blocker-resolution marker carrier-completion blocker-resolution blocker-resolution blocker-resolution blocker-resolution blocker_resolution blocker_resolution blocker_resolution blocker remains preserved, but the bounded read-prefix resolution requirement is not yet fixed narrowly enough"
        ),
        "named_blocker_not_preserved": (
            "the controlling source no longer preserves the named blocker-resolution marker carrier-completion blocker-resolution blocker-resolution blocker-resolution blocker-resolution blocker_resolution blocker_resolution blocker_resolution blocker narrowly enough for path definition"
        ),
    }
    return mapping.get(
        blocker,
        "the blocker-resolution marker carrier-completion blocker-resolution blocker-resolution blocker-resolution blocker-resolution blocker-resolution blocker-resolution blocker_resolution blocker-resolution path is not yet fixed narrowly enough",
    )


def build_registry(source_manifest: Dict[str, Any], status_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    row = {
        "source_gate11bc_run_id": str(source_manifest.get("run_id") or ""),
        "source_gate11bc_code_git_commit": str(source_manifest.get("code_git_commit") or ""),
    }
    row.update({key: str(status_payload[key]) for key in STATUS_FIELDS})
    return [row]


def build_policy_compare(registry_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"source_gate11bc_run_id": str(row["source_gate11bc_run_id"]), **{key: str(row[key]) for key in STATUS_FIELDS}} for row in registry_rows]


def build_status_payload(source_manifest: Dict[str, Any], source_status: Dict[str, Any], report_text: str) -> Dict[str, Any]:
    incomplete = source_is_incomplete(source_manifest, source_status, report_text)
    status_payload = {
        key: ("preserved" if source_status_value(source_status, key) == "preserved" else "not_preserved")
        for key in SOURCE_PRESERVATION_KEYS
    }

    source_gate11bc_status = source_status_value(
        source_status,
        "blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_status",
    )
    if source_gate11bc_status == "blocker_named":
        gate11bc_blocker_named_state_preservation_status = "preserved"
    elif source_gate11bc_status == "deferred":
        gate11bc_blocker_named_state_preservation_status = "deferred"
    else:
        gate11bc_blocker_named_state_preservation_status = "not_preserved"
    status_payload["gate11bc_blocker_named_state_preservation_status"] = gate11bc_blocker_named_state_preservation_status

    for key in CONFIRMED_KEYS:
        status_payload[key] = "confirmed" if source_status_value(source_status, key) == "confirmed" else "not_confirmed"

    source_blocker = source_status_value(source_status, "next_named_blocker")
    if incomplete:
        named_blocker_preservation_status = "deferred"
    elif (
        source_status_value(
            source_status,
            "named_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_condition_preservation_status",
        )
        == "preserved"
        and source_status_value(source_status, "explicit_blocker_resolution_marker_blocker_status") == "named"
        and source_status_value(source_status, "same_source_blocker_resolution_blocker_status") == "named"
        and source_blocker in gate11bc.KNOWN_BLOCKERS
    ):
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
        or source_status_value(source_status, "blocker_resolution_blocker_boundary_status") != "confirmed"
    ):
        blocker_resolution_boundary_status = "denied"
    else:
        blocker_resolution_boundary_status = "confirmed"
    status_payload["blocker_resolution_boundary_status"] = blocker_resolution_boundary_status

    if incomplete:
        minimum_same_source_blocker_resolution_rule_status = "deferred"
    elif blocker_resolution_boundary_status == "denied":
        minimum_same_source_blocker_resolution_rule_status = "denied"
    elif gate11bc_blocker_named_state_preservation_status == "deferred":
        minimum_same_source_blocker_resolution_rule_status = "deferred"
    elif gate11bc_blocker_named_state_preservation_status != "preserved":
        minimum_same_source_blocker_resolution_rule_status = "not_yet_defined"
    elif named_blocker_preservation_status != "preserved":
        minimum_same_source_blocker_resolution_rule_status = "not_yet_defined"
    elif source_blocker in PATH_DEFINABLE_BLOCKERS:
        minimum_same_source_blocker_resolution_rule_status = "defined"
    else:
        minimum_same_source_blocker_resolution_rule_status = "not_yet_defined"
    status_payload["minimum_same_source_blocker_resolution_rule_status"] = minimum_same_source_blocker_resolution_rule_status

    if incomplete:
        bounded_read_prefix_resolution_requirement_status = "deferred"
    elif blocker_resolution_boundary_status == "denied":
        bounded_read_prefix_resolution_requirement_status = "denied"
    elif gate11bc_blocker_named_state_preservation_status == "deferred":
        bounded_read_prefix_resolution_requirement_status = "deferred"
    elif minimum_same_source_blocker_resolution_rule_status != "defined":
        bounded_read_prefix_resolution_requirement_status = "not_yet_defined"
    elif source_blocker in PATH_DEFINABLE_BLOCKERS:
        bounded_read_prefix_resolution_requirement_status = "defined"
    else:
        bounded_read_prefix_resolution_requirement_status = "not_yet_defined"
    status_payload["bounded_read_prefix_resolution_requirement_status"] = bounded_read_prefix_resolution_requirement_status

    if incomplete:
        overall = "deferred"
    elif (
        any(status_payload[key] != "preserved" for key in SOURCE_PRESERVATION_KEYS)
        or gate11bc_blocker_named_state_preservation_status == "not_preserved"
        or any(status_payload[key] != "confirmed" for key in CONFIRMED_KEYS)
        or blocker_resolution_boundary_status == "denied"
    ):
        overall = "denied"
    elif (
        gate11bc_blocker_named_state_preservation_status == "deferred"
        or named_blocker_preservation_status == "deferred"
        or minimum_same_source_blocker_resolution_rule_status == "deferred"
        or bounded_read_prefix_resolution_requirement_status == "deferred"
        or blocker_resolution_boundary_status == "deferred"
    ):
        overall = "deferred"
    elif (
        named_blocker_preservation_status == "preserved"
        and minimum_same_source_blocker_resolution_rule_status == "defined"
        and bounded_read_prefix_resolution_requirement_status == "defined"
        and blocker_resolution_boundary_status == "confirmed"
    ):
        overall = "path_defined"
    else:
        overall = "not_yet_defined"
    status_payload[
        "blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_path_status"
    ] = overall

    next_named_blocker = ""
    for key in SOURCE_PRESERVATION_KEYS:
        if status_payload[key] == "not_preserved":
            next_named_blocker = key.replace("_preservation_status", "_not_preserved")
            break
    if not next_named_blocker and gate11bc_blocker_named_state_preservation_status == "not_preserved":
        next_named_blocker = "gate11bc_blocker_named_state_not_preserved"
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
        elif gate11bc_blocker_named_state_preservation_status == "deferred":
            next_named_blocker = "upstream_blocker_naming_deferred"
        elif named_blocker_preservation_status != "preserved":
            next_named_blocker = "named_blocker_not_preserved"
        elif minimum_same_source_blocker_resolution_rule_status != "defined":
            next_named_blocker = "minimum_same_source_blocker_resolution_rule_not_fixed"
        elif bounded_read_prefix_resolution_requirement_status != "defined":
            next_named_blocker = "bounded_read_prefix_resolution_requirement_not_fixed"
    status_payload["next_named_blocker"] = next_named_blocker
    return status_payload


def build_report(
    run_id: str,
    source_manifest: Dict[str, Any],
    policy_compare_rows: Sequence[Dict[str, Any]],
    status_payload: Dict[str, Any],
) -> str:
    lines = [
        "# Gate11BD Blocker-Resolution Marker Carrier-Completion Blocker-Resolution Blocker-Resolution Blocker-Resolution Blocker-Resolution Blocker-Resolution Blocker-Resolution Blocker-Resolution Blocker-Resolution Path Read",
        "",
        f"run_id: {run_id}",
        f"source_gate11bc_run_id: {source_manifest.get('run_id', '')}",
        f"source_gate11bc_code_git_commit: {source_manifest.get('code_git_commit', '')}",
        "",
        "## Discipline",
        "",
        "- Gate11BD asks only what is the minimum honest path by which the named blocker-resolution marker carrier-completion blocker-resolution blocker-resolution blocker-resolution blocker-resolution blocker-resolution blocker-resolution blocker_resolution blocker-resolution blocker could later be resolved under the fixed Gate11BC line",
        "- Gate11BD defines only a path",
        "- Gate11BD does not convert blocker naming into blocker resolution",
        "- Gate11BD does not widen into blocker-resolution marker carrier-completion blocker-resolution blocker_resolution blocker judgment, blocker-resolution marker carrier-completion blocker-resolution blocker_resolution judgment, blocker-resolution marker carrier-completion blocker_resolution judgment, blocker_resolution judgment, residual completion judgment, later-source admission, explicit-presence judgment, candidate declaration, reopening eligibility, or operator reopening",
        "- Gate11BD does not inherit worker-side interpretations from generic prose or examples",
        "",
        "## Source Summary",
        "",
        "| source_gate11bc_run_id | " + " | ".join(STATUS_FIELDS) + " |",
        "|---|" + "---|" * len(STATUS_FIELDS),
    ]
    for row in policy_compare_rows:
        lines.append("| " + " | ".join([str(row["source_gate11bc_run_id"]), *[str(row[key]) for key in STATUS_FIELDS]]) + " |")
    lines.extend(["", "## Status", ""])
    lines.extend([f"- {key}: `{status_payload[key]}`" for key in STATUS_FIELDS])
    lines.extend(["", "## Judgment", ""])

    outcome = status_payload[
        "blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_path_status"
    ]
    if outcome == "path_defined":
        if status_payload["next_named_blocker"]:
            lines.append(f"- {blocker_sentence(status_payload['next_named_blocker'])}")
        else:
            lines.append(
                "- the minimum honest same-source blocker-resolution path is fixed narrowly enough for a later actual-resolution slice without resolving the blocker here"
            )
    elif outcome == "not_yet_defined":
        lines.append("- the fixed Gate11BC line still does not define the blocker-resolution path narrowly enough for a later actual-resolution slice")
    elif outcome == "denied":
        lines.append("- the attempted blocker-resolution path read is denied because it would depend on shortcut, inflation, rewrite, graph-wide leap, or worker-side synthesis")
    else:
        lines.append("- the frozen source is incomplete or contradictory for a blocker-resolution path judgment")
    if status_payload["next_named_blocker"]:
        lines.append(f"- next_named_blocker: `{status_payload['next_named_blocker']}`")
    lines.extend(
        [
            "",
            "## Memory Hook",
            "",
            "- Gate11BD defines only the minimum honest path by which the named Gate11BC blocker could later be resolved; it does not resolve that blocker here.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    source_dir = Path(args.gate11bc_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = out_dir.name

    source_manifest = gate9a.read_json(source_dir / gate11bc.DEFAULT_MANIFEST)
    source_status = gate9a.read_json(source_dir / gate11bc.DEFAULT_STATUS)
    source_report = (source_dir / gate11bc.DEFAULT_REPORT).read_text(encoding="utf-8")

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
    gate9a.write_csv(policy_compare_path, ("source_gate11bc_run_id", *STATUS_FIELDS), policy_compare_rows)
    gate9a.write_json(status_path, status_payload)
    gate9a.write_text(report_path, build_report(run_id, source_manifest, policy_compare_rows, status_payload))
    gate9a.write_json(
        manifest_path,
        {
            "run_id": run_id,
            "schema_version": SCHEMA_VERSION,
            "method_id": METHOD_ID,
            "code_git_commit": gate9a.current_git_commit(),
            "source_gate11bc_run_id": str(source_manifest.get("run_id") or ""),
            "source_gate11bc_code_git_commit": str(source_manifest.get("code_git_commit") or ""),
            "inputs": {"gate11bc_dir": str(source_dir)},
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