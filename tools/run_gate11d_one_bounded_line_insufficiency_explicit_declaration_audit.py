#!/usr/bin/env python3
"""Run a Gate11D one bounded-line insufficiency explicit-declaration audit on Gate11C outputs."""

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

import run_gate11c_bounded_line_insufficiency_declaration_surface_audit as gate11c
import run_gate9a_graph_gauge_consumer as gate9a


SCHEMA_VERSION = "gate11d_one_bounded_line_insufficiency_explicit_declaration_audit_v1"
METHOD_ID = "gate11d_one_bounded_line_insufficiency_explicit_declaration_audit_v1"

DEFAULT_REGISTRY = "one_bounded_line_insufficiency_explicit_declaration_registry.jsonl"
DEFAULT_POLICY_COMPARE = "one_bounded_line_insufficiency_explicit_declaration_policy_compare.csv"
DEFAULT_STATUS = "one_bounded_line_insufficiency_explicit_declaration_status.json"
DEFAULT_REPORT = "gate11d_one_bounded_line_insufficiency_explicit_declaration_read.md"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"

REQUIRED_GATE11C_STATUS_KEYS = (
    "gate10_closeout_preservation_status",
    "gate11a_absence_result_preservation_status",
    "broader_trusted_tree_settlement_still_unearned_status",
    "operator_admission_still_denied_status",
    "retroactive_reinterpretation_forbidden_status",
    "explicit_marker_shape_status",
    "single_candidate_singularity_status",
    "bounded_line_insufficiency_evidence_shape_status",
    "anti_inflation_boundary_status",
    "bounded_line_insufficiency_declaration_surface_status",
)

DECLARATION_PREFIX = "one bounded-line insufficiency candidate is explicitly declared:"
HOST_FAILURE_PREFIX = "the current bounded line cannot honestly host "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Gate11D one bounded-line insufficiency explicit-declaration audit "
            "from the frozen Gate11C declaration-surface run without deciding reopening eligibility."
        )
    )
    parser.add_argument("--gate11c-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_status_value(status_payload: Dict[str, Any], key: str) -> str:
    return str(status_payload.get(key, ""))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        rows.append(json.loads(stripped))
    return rows


def source_is_incomplete(
    source_gate11c_manifest: Dict[str, Any], source_gate11c_status: Dict[str, Any], report_text: str
) -> bool:
    if not str(source_gate11c_manifest.get("run_id") or ""):
        return True
    if not report_text.strip():
        return True
    return any(not source_status_value(source_gate11c_status, key) for key in REQUIRED_GATE11C_STATUS_KEYS)


def extract_explicit_marker_values(report_text: str, marker: str, value_pattern: str = r"[^\r\n]+") -> List[str]:
    pattern = re.compile(rf"(?im)^\s*{re.escape(marker)}\s*[:=]\s*({value_pattern})\s*$")
    return [match.group(1).strip() for match in pattern.finditer(report_text)]


def extract_prefixed_values(report_text: str, prefix: str) -> List[str]:
    pattern = re.compile(rf"(?im)^\s*(?:[-*]\s*)?{re.escape(prefix)}(.+?)\s*$")
    return [match.group(1).strip() for match in pattern.finditer(report_text)]


def collect_declaration_evidence(
    source_gate11c_status: Dict[str, Any], report_text: str, registry_rows: Sequence[Dict[str, Any]]
) -> Dict[str, List[str]]:
    status_declaration = [
        value.lower()
        for value in extract_explicit_marker_values(
            report_text, "bounded_line_insufficiency_candidate_declaration_status", r"[a-z_]+"
        )
    ]
    status_candidate_ids = [
        value.lower()
        for value in extract_explicit_marker_values(
            report_text, "bounded_line_insufficiency_candidate_id", r"[a-z0-9_]+"
        )
    ]
    status_classes = [
        value.lower()
        for value in extract_explicit_marker_values(
            report_text, "bounded_line_insufficiency_class_status", r"[a-z0-9_]+"
        )
    ]
    host_failure_statuses = [
        value.lower()
        for value in extract_explicit_marker_values(
            report_text, "bounded_line_host_failure_status", r"[a-z_]+"
        )
    ]
    prefixed_candidate_ids = [value.lower() for value in extract_prefixed_values(report_text, DECLARATION_PREFIX)]
    host_failure_candidates = [
        value.lower() for value in extract_prefixed_values(report_text, HOST_FAILURE_PREFIX)
    ]
    declaration_registry_rows = [
        row
        for row in registry_rows
        if str(row.get("bounded_line_insufficiency_candidate_id") or "").strip()
    ]
    registry_candidate_ids = [
        str(row.get("bounded_line_insufficiency_candidate_id") or "").strip().lower()
        for row in declaration_registry_rows
    ]
    registry_classes = [
        str(row.get("bounded_line_insufficiency_class_status") or "").strip().lower()
        for row in declaration_registry_rows
        if str(row.get("bounded_line_insufficiency_class_status") or "").strip()
    ]

    if "bounded_line_insufficiency_candidate_declaration_status" in source_gate11c_status:
        value = str(source_gate11c_status.get("bounded_line_insufficiency_candidate_declaration_status") or "").strip()
        if value:
            status_declaration.append(value.lower())
    if "bounded_line_insufficiency_candidate_id" in source_gate11c_status:
        value = str(source_gate11c_status.get("bounded_line_insufficiency_candidate_id") or "").strip()
        if value:
            status_candidate_ids.append(value.lower())
    if "bounded_line_insufficiency_class_status" in source_gate11c_status:
        value = str(source_gate11c_status.get("bounded_line_insufficiency_class_status") or "").strip()
        if value:
            status_classes.append(value.lower())
    if "bounded_line_host_failure_status" in source_gate11c_status:
        value = str(source_gate11c_status.get("bounded_line_host_failure_status") or "").strip()
        if value:
            host_failure_statuses.append(value.lower())

    return {
        "declaration_statuses": status_declaration,
        "candidate_ids": status_candidate_ids,
        "prefixed_candidate_ids": prefixed_candidate_ids,
        "registry_candidate_ids": registry_candidate_ids,
        "class_statuses": status_classes,
        "registry_classes": registry_classes,
        "host_failure_statuses": host_failure_statuses,
        "host_failure_candidates": host_failure_candidates,
    }


def build_registry(
    source_gate11c_manifest: Dict[str, Any],
    source_gate11c_status: Dict[str, Any],
    status_payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return [
        {
            "source_gate11c_run_id": str(source_gate11c_manifest.get("run_id") or ""),
            "source_gate11c_code_git_commit": str(
                source_gate11c_manifest.get("code_git_commit") or ""
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
            "broader_trusted_tree_settlement_still_unearned_status": str(
                status_payload["broader_trusted_tree_settlement_still_unearned_status"]
            ),
            "operator_admission_still_denied_status": str(
                status_payload["operator_admission_still_denied_status"]
            ),
            "retroactive_reinterpretation_forbidden_status": str(
                status_payload["retroactive_reinterpretation_forbidden_status"]
            ),
            "anti_inflation_boundary_status": str(status_payload["anti_inflation_boundary_status"]),
            "bounded_line_insufficiency_explicit_declaration_marker_status": str(
                status_payload["bounded_line_insufficiency_explicit_declaration_marker_status"]
            ),
            "bounded_line_insufficiency_candidate_id_singularity_status": str(
                status_payload["bounded_line_insufficiency_candidate_id_singularity_status"]
            ),
            "bounded_line_insufficiency_class_singularity_status": str(
                status_payload["bounded_line_insufficiency_class_singularity_status"]
            ),
            "bounded_line_host_failure_statement_status": str(
                status_payload["bounded_line_host_failure_statement_status"]
            ),
            "one_bounded_line_insufficiency_explicit_declaration_status": str(
                status_payload["one_bounded_line_insufficiency_explicit_declaration_status"]
            ),
            "source_bounded_line_insufficiency_declaration_surface_status": source_status_value(
                source_gate11c_status, "bounded_line_insufficiency_declaration_surface_status"
            ),
        }
    ]


def build_policy_compare(registry_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "source_gate11c_run_id": str(row["source_gate11c_run_id"]),
            "gate10_closeout_preservation_status": str(row["gate10_closeout_preservation_status"]),
            "gate11a_absence_result_preservation_status": str(
                row["gate11a_absence_result_preservation_status"]
            ),
            "gate11c_declaration_surface_preservation_status": str(
                row["gate11c_declaration_surface_preservation_status"]
            ),
            "broader_trusted_tree_settlement_still_unearned_status": str(
                row["broader_trusted_tree_settlement_still_unearned_status"]
            ),
            "operator_admission_still_denied_status": str(row["operator_admission_still_denied_status"]),
            "retroactive_reinterpretation_forbidden_status": str(
                row["retroactive_reinterpretation_forbidden_status"]
            ),
            "anti_inflation_boundary_status": str(row["anti_inflation_boundary_status"]),
            "bounded_line_insufficiency_explicit_declaration_marker_status": str(
                row["bounded_line_insufficiency_explicit_declaration_marker_status"]
            ),
            "bounded_line_insufficiency_candidate_id_singularity_status": str(
                row["bounded_line_insufficiency_candidate_id_singularity_status"]
            ),
            "bounded_line_insufficiency_class_singularity_status": str(
                row["bounded_line_insufficiency_class_singularity_status"]
            ),
            "bounded_line_host_failure_statement_status": str(
                row["bounded_line_host_failure_statement_status"]
            ),
            "one_bounded_line_insufficiency_explicit_declaration_status": str(
                row["one_bounded_line_insufficiency_explicit_declaration_status"]
            ),
        }
        for row in registry_rows
    ]


def build_status_payload(
    source_gate11c_manifest: Dict[str, Any],
    source_gate11c_status: Dict[str, Any],
    report_text: str,
    registry_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    incomplete = source_is_incomplete(source_gate11c_manifest, source_gate11c_status, report_text)
    evidence = collect_declaration_evidence(source_gate11c_status, report_text, registry_rows)

    gate10_closeout_preservation_status = (
        "preserved"
        if source_status_value(source_gate11c_status, "gate10_closeout_preservation_status") == "preserved"
        else "not_preserved"
    )
    gate11a_absence_result_preservation_status = (
        "preserved"
        if source_status_value(source_gate11c_status, "gate11a_absence_result_preservation_status")
        == "preserved"
        else "not_preserved"
    )
    gate11c_declaration_surface_preservation_status = (
        "preserved"
        if source_status_value(
            source_gate11c_status, "bounded_line_insufficiency_declaration_surface_status"
        )
        == "surface_defined"
        else "not_preserved"
    )
    broader_trusted_tree_settlement_still_unearned_status = (
        "confirmed"
        if source_status_value(
            source_gate11c_status, "broader_trusted_tree_settlement_still_unearned_status"
        )
        == "confirmed"
        else "not_confirmed"
    )
    operator_admission_still_denied_status = (
        "confirmed"
        if source_status_value(source_gate11c_status, "operator_admission_still_denied_status")
        == "confirmed"
        else "not_confirmed"
    )
    retroactive_reinterpretation_forbidden_status = (
        "confirmed"
        if source_status_value(
            source_gate11c_status, "retroactive_reinterpretation_forbidden_status"
        )
        == "confirmed"
        else "not_confirmed"
    )
    anti_inflation_boundary_status = (
        "confirmed"
        if source_status_value(source_gate11c_status, "anti_inflation_boundary_status") == "defined"
        and broader_trusted_tree_settlement_still_unearned_status == "confirmed"
        and operator_admission_still_denied_status == "confirmed"
        and retroactive_reinterpretation_forbidden_status == "confirmed"
        else "not_confirmed"
    )

    declaration_statuses = evidence["declaration_statuses"]
    marker_candidate_ids = evidence["candidate_ids"]
    prefixed_candidate_ids = evidence["prefixed_candidate_ids"]
    registry_candidate_ids = evidence["registry_candidate_ids"]
    class_statuses = evidence["class_statuses"]
    registry_classes = evidence["registry_classes"]
    host_failure_statuses = evidence["host_failure_statuses"]
    host_failure_candidates = evidence["host_failure_candidates"]

    if incomplete:
        bounded_line_insufficiency_explicit_declaration_marker_status = "deferred"
    else:
        has_any_marker_fragment = any(
            [declaration_statuses, marker_candidate_ids, prefixed_candidate_ids, registry_candidate_ids]
        )
        if not has_any_marker_fragment:
            bounded_line_insufficiency_explicit_declaration_marker_status = "absent"
        elif any(status != "declared" for status in declaration_statuses):
            bounded_line_insufficiency_explicit_declaration_marker_status = "deferred"
        elif not declaration_statuses or not marker_candidate_ids or not prefixed_candidate_ids or not registry_candidate_ids:
            bounded_line_insufficiency_explicit_declaration_marker_status = "deferred"
        elif len(set(marker_candidate_ids)) != 1 or len(set(prefixed_candidate_ids)) != 1 or len(set(registry_candidate_ids)) != 1:
            bounded_line_insufficiency_explicit_declaration_marker_status = "deferred"
        elif (
            set(marker_candidate_ids) == set(prefixed_candidate_ids)
            and set(marker_candidate_ids) == set(registry_candidate_ids)
        ):
            bounded_line_insufficiency_explicit_declaration_marker_status = "present"
        else:
            bounded_line_insufficiency_explicit_declaration_marker_status = "deferred"

    if incomplete:
        bounded_line_insufficiency_candidate_id_singularity_status = "deferred"
    else:
        candidate_pool = set(marker_candidate_ids) | set(prefixed_candidate_ids) | set(registry_candidate_ids) | set(host_failure_candidates)
        if not candidate_pool:
            bounded_line_insufficiency_candidate_id_singularity_status = "absent"
        elif len(candidate_pool) > 1:
            bounded_line_insufficiency_candidate_id_singularity_status = "multiple"
        elif bounded_line_insufficiency_explicit_declaration_marker_status == "deferred":
            bounded_line_insufficiency_candidate_id_singularity_status = "deferred"
        else:
            bounded_line_insufficiency_candidate_id_singularity_status = "single"

    if incomplete:
        bounded_line_insufficiency_class_singularity_status = "deferred"
    else:
        class_pool = set(class_statuses) | set(registry_classes)
        if not class_pool:
            bounded_line_insufficiency_class_singularity_status = "none"
        elif any(value not in gate11c.DECLARATION_CLASSES for value in class_pool):
            bounded_line_insufficiency_class_singularity_status = "deferred"
        elif len(class_pool) > 1:
            bounded_line_insufficiency_class_singularity_status = "multiple"
        elif (
            class_statuses
            and registry_classes
            and set(class_statuses) != set(registry_classes)
        ):
            bounded_line_insufficiency_class_singularity_status = "deferred"
        else:
            bounded_line_insufficiency_class_singularity_status = "single"

    if incomplete:
        bounded_line_host_failure_statement_status = "deferred"
    else:
        candidate_pool = set(marker_candidate_ids) | set(prefixed_candidate_ids) | set(registry_candidate_ids)
        if not host_failure_statuses and not host_failure_candidates:
            bounded_line_host_failure_statement_status = "absent"
        elif any(status != "explicit" for status in host_failure_statuses):
            bounded_line_host_failure_statement_status = "deferred"
        elif not host_failure_statuses or not host_failure_candidates:
            bounded_line_host_failure_statement_status = "deferred"
        elif len(set(host_failure_candidates)) != 1:
            bounded_line_host_failure_statement_status = "deferred"
        elif candidate_pool and set(host_failure_candidates) != candidate_pool:
            bounded_line_host_failure_statement_status = "deferred"
        else:
            bounded_line_host_failure_statement_status = "explicit"

    if incomplete:
        one_bounded_line_insufficiency_explicit_declaration_status = "deferred"
    elif (
        gate10_closeout_preservation_status != "preserved"
        or gate11a_absence_result_preservation_status != "preserved"
        or gate11c_declaration_surface_preservation_status != "preserved"
        or broader_trusted_tree_settlement_still_unearned_status != "confirmed"
        or operator_admission_still_denied_status != "confirmed"
        or retroactive_reinterpretation_forbidden_status != "confirmed"
        or anti_inflation_boundary_status != "confirmed"
    ):
        one_bounded_line_insufficiency_explicit_declaration_status = "denied"
    elif (
        bounded_line_insufficiency_explicit_declaration_marker_status == "deferred"
        or bounded_line_insufficiency_candidate_id_singularity_status in {"deferred", "multiple"}
        or bounded_line_insufficiency_class_singularity_status in {"deferred", "multiple"}
        or bounded_line_host_failure_statement_status == "deferred"
    ):
        one_bounded_line_insufficiency_explicit_declaration_status = "deferred"
    elif (
        bounded_line_insufficiency_explicit_declaration_marker_status == "present"
        and bounded_line_insufficiency_candidate_id_singularity_status == "single"
        and bounded_line_insufficiency_class_singularity_status == "single"
        and bounded_line_host_failure_statement_status == "explicit"
    ):
        one_bounded_line_insufficiency_explicit_declaration_status = "declared"
    else:
        one_bounded_line_insufficiency_explicit_declaration_status = "not_yet_declared"

    if gate10_closeout_preservation_status != "preserved":
        next_named_blocker = "gate10_closeout_not_preserved"
    elif gate11a_absence_result_preservation_status != "preserved":
        next_named_blocker = "gate11a_absence_result_not_preserved"
    elif gate11c_declaration_surface_preservation_status != "preserved":
        next_named_blocker = "gate11c_declaration_surface_not_preserved"
    elif broader_trusted_tree_settlement_still_unearned_status != "confirmed":
        next_named_blocker = "broader_trusted_tree_settlement_not_explicitly_unearned"
    elif operator_admission_still_denied_status != "confirmed":
        next_named_blocker = "operator_admission_not_denied"
    elif retroactive_reinterpretation_forbidden_status != "confirmed":
        next_named_blocker = "retroactive_reinterpretation_pressure"
    elif anti_inflation_boundary_status != "confirmed":
        next_named_blocker = "anti_inflation_boundary_not_confirmed"
    elif one_bounded_line_insufficiency_explicit_declaration_status == "deferred":
        if incomplete:
            next_named_blocker = "controlling_source_incomplete"
        elif bounded_line_insufficiency_candidate_id_singularity_status == "multiple":
            next_named_blocker = "multiple_candidate_ids"
        elif bounded_line_insufficiency_class_singularity_status == "multiple":
            next_named_blocker = "multiple_classes"
        elif bounded_line_insufficiency_explicit_declaration_marker_status == "deferred":
            next_named_blocker = "explicit_declaration_marker_ambiguous"
        elif bounded_line_host_failure_statement_status == "deferred":
            next_named_blocker = "host_failure_statement_ambiguous"
        else:
            next_named_blocker = "worker_side_resolution_required"
    elif one_bounded_line_insufficiency_explicit_declaration_status == "not_yet_declared":
        if bounded_line_insufficiency_explicit_declaration_marker_status == "absent":
            next_named_blocker = "no_explicit_declaration_marker"
        elif bounded_line_insufficiency_candidate_id_singularity_status == "absent":
            next_named_blocker = "no_candidate_id"
        elif bounded_line_insufficiency_class_singularity_status == "none":
            next_named_blocker = "no_single_class"
        elif bounded_line_host_failure_statement_status == "absent":
            next_named_blocker = "no_explicit_host_failure_statement"
        else:
            next_named_blocker = "explicit_declaration_not_instantiated"
    else:
        next_named_blocker = ""

    return {
        "gate10_closeout_preservation_status": gate10_closeout_preservation_status,
        "gate11a_absence_result_preservation_status": gate11a_absence_result_preservation_status,
        "gate11c_declaration_surface_preservation_status": gate11c_declaration_surface_preservation_status,
        "broader_trusted_tree_settlement_still_unearned_status": broader_trusted_tree_settlement_still_unearned_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "retroactive_reinterpretation_forbidden_status": retroactive_reinterpretation_forbidden_status,
        "anti_inflation_boundary_status": anti_inflation_boundary_status,
        "bounded_line_insufficiency_explicit_declaration_marker_status": bounded_line_insufficiency_explicit_declaration_marker_status,
        "bounded_line_insufficiency_candidate_id_singularity_status": bounded_line_insufficiency_candidate_id_singularity_status,
        "bounded_line_insufficiency_class_singularity_status": bounded_line_insufficiency_class_singularity_status,
        "bounded_line_host_failure_statement_status": bounded_line_host_failure_statement_status,
        "one_bounded_line_insufficiency_explicit_declaration_status": one_bounded_line_insufficiency_explicit_declaration_status,
        "next_named_blocker": next_named_blocker,
    }


def build_report(
    run_id: str,
    source_gate11c_manifest: Dict[str, Any],
    policy_compare_rows: Sequence[Dict[str, Any]],
    status_payload: Dict[str, Any],
) -> str:
    lines = [
        "# Gate11D One Bounded-Line Insufficiency Explicit-Declaration Read",
        "",
        f"run_id: {run_id}",
        f"source_gate11c_run_id: {source_gate11c_manifest.get('run_id', '')}",
        f"source_gate11c_code_git_commit: {source_gate11c_manifest.get('code_git_commit', '')}",
        "",
        "## Discipline",
        "",
        "- Gate11D asks only whether one bounded-line insufficiency candidate is explicitly declared now",
        "- Gate11D does not redefine declaration surface",
        "- Gate11D does not decide reopening eligibility",
        "- Gate10 closeout remains bounded and preserved",
        "- Gate11A absence remains preserved",
        "- Gate11C surface-defined result remains preserved unless the same frozen source fully instantiates one declaration",
        "- broader trusted-tree settlement remains unearned",
        "- operator admission remains denied",
        "- no retroactive rewrite of Gate9, Gate10, Gate11A, Gate11B, or Gate11C memory is permitted",
        "",
        "## Source Summary",
        "",
        "| source_gate11c_run_id | gate10_closeout_preservation_status | gate11a_absence_result_preservation_status | gate11c_declaration_surface_preservation_status | broader_trusted_tree_settlement_still_unearned_status | operator_admission_still_denied_status | retroactive_reinterpretation_forbidden_status | anti_inflation_boundary_status | bounded_line_insufficiency_explicit_declaration_marker_status | bounded_line_insufficiency_candidate_id_singularity_status | bounded_line_insufficiency_class_singularity_status | bounded_line_host_failure_statement_status | one_bounded_line_insufficiency_explicit_declaration_status |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in policy_compare_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["source_gate11c_run_id"]),
                    str(row["gate10_closeout_preservation_status"]),
                    str(row["gate11a_absence_result_preservation_status"]),
                    str(row["gate11c_declaration_surface_preservation_status"]),
                    str(row["broader_trusted_tree_settlement_still_unearned_status"]),
                    str(row["operator_admission_still_denied_status"]),
                    str(row["retroactive_reinterpretation_forbidden_status"]),
                    str(row["anti_inflation_boundary_status"]),
                    str(row["bounded_line_insufficiency_explicit_declaration_marker_status"]),
                    str(row["bounded_line_insufficiency_candidate_id_singularity_status"]),
                    str(row["bounded_line_insufficiency_class_singularity_status"]),
                    str(row["bounded_line_host_failure_statement_status"]),
                    str(row["one_bounded_line_insufficiency_explicit_declaration_status"]),
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
            f"- broader_trusted_tree_settlement_still_unearned_status: `{status_payload['broader_trusted_tree_settlement_still_unearned_status']}`",
            f"- operator_admission_still_denied_status: `{status_payload['operator_admission_still_denied_status']}`",
            f"- retroactive_reinterpretation_forbidden_status: `{status_payload['retroactive_reinterpretation_forbidden_status']}`",
            f"- anti_inflation_boundary_status: `{status_payload['anti_inflation_boundary_status']}`",
            f"- bounded_line_insufficiency_explicit_declaration_marker_status: `{status_payload['bounded_line_insufficiency_explicit_declaration_marker_status']}`",
            f"- bounded_line_insufficiency_candidate_id_singularity_status: `{status_payload['bounded_line_insufficiency_candidate_id_singularity_status']}`",
            f"- bounded_line_insufficiency_class_singularity_status: `{status_payload['bounded_line_insufficiency_class_singularity_status']}`",
            f"- bounded_line_host_failure_statement_status: `{status_payload['bounded_line_host_failure_statement_status']}`",
            f"- one_bounded_line_insufficiency_explicit_declaration_status: `{status_payload['one_bounded_line_insufficiency_explicit_declaration_status']}`",
            f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
        ]
    )

    if status_payload["one_bounded_line_insufficiency_explicit_declaration_status"] == "declared":
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- one bounded-line insufficiency candidate is now explicitly declared under the fixed Gate11C surface",
                "- Gate11D makes no reopening-eligibility judgment here",
            ]
        )
    elif status_payload["one_bounded_line_insufficiency_explicit_declaration_status"] == "deferred":
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- the explicit-declaration audit remains deferred because the frozen source is incomplete or would require worker-side resolution",
                f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
            ]
        )
    elif status_payload["one_bounded_line_insufficiency_explicit_declaration_status"] == "denied":
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- the explicit-declaration audit is denied under the frozen Gate11D boundary",
                f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- the fixed Gate11C declaration surface exists, but one full explicit bounded-line insufficiency declaration is not yet instantiated in the controlling source",
                f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    source_gate11c_dir = Path(args.gate11c_dir)
    out_dir = Path(args.out_dir)
    run_id = out_dir.name

    out_dir.mkdir(parents=True, exist_ok=True)

    source_gate11c_manifest = gate9a.read_json(source_gate11c_dir / gate11c.DEFAULT_MANIFEST)
    source_gate11c_status = gate9a.read_json(source_gate11c_dir / gate11c.DEFAULT_STATUS)
    source_gate11c_report = (source_gate11c_dir / gate11c.DEFAULT_REPORT).read_text(encoding="utf-8")
    source_gate11c_registry = read_jsonl(source_gate11c_dir / gate11c.DEFAULT_REGISTRY)

    status_payload = build_status_payload(
        source_gate11c_manifest,
        source_gate11c_status,
        source_gate11c_report,
        source_gate11c_registry,
    )
    registry_rows = build_registry(source_gate11c_manifest, source_gate11c_status, status_payload)
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
            "source_gate11c_run_id",
            "gate10_closeout_preservation_status",
            "gate11a_absence_result_preservation_status",
            "gate11c_declaration_surface_preservation_status",
            "broader_trusted_tree_settlement_still_unearned_status",
            "operator_admission_still_denied_status",
            "retroactive_reinterpretation_forbidden_status",
            "anti_inflation_boundary_status",
            "bounded_line_insufficiency_explicit_declaration_marker_status",
            "bounded_line_insufficiency_candidate_id_singularity_status",
            "bounded_line_insufficiency_class_singularity_status",
            "bounded_line_host_failure_statement_status",
            "one_bounded_line_insufficiency_explicit_declaration_status",
        ),
        policy_compare_rows,
    )
    gate9a.write_json(status_path, status_payload)
    gate9a.write_text(
        report_path,
        build_report(
            run_id=run_id,
            source_gate11c_manifest=source_gate11c_manifest,
            policy_compare_rows=policy_compare_rows,
            status_payload=status_payload,
        ),
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "run_id": run_id,
        "code_git_commit": gate9a.current_git_commit(),
        "source_gate11c_dir": gate9a.repo_relative_or_posix(source_gate11c_dir),
        "source_gate11c_run_id": str(source_gate11c_manifest.get("run_id") or ""),
        "source_gate11c_code_git_commit": str(
            source_gate11c_manifest.get("code_git_commit") or ""
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