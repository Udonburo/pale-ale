#!/usr/bin/env python3
"""Run a Gate11B bounded-line insufficiency declarability audit on Gate11A outputs."""

import argparse
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

import run_gate11a_named_operator_pressure_admissibility as gate11a
import run_gate9a_graph_gauge_consumer as gate9a


REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "gate11b_bounded_line_insufficiency_declarability_v1"
METHOD_ID = "gate11b_bounded_line_insufficiency_declarability_v1"

DEFAULT_REGISTRY = "bounded_line_insufficiency_declarability_registry.jsonl"
DEFAULT_POLICY_COMPARE = "bounded_line_insufficiency_declarability_policy_compare.csv"
DEFAULT_STATUS = "bounded_line_insufficiency_declarability_status.json"
DEFAULT_REPORT = "gate11b_bounded_line_insufficiency_declarability_read.md"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"

INSUFFICIENCY_CLASSES = (
    "tree_choice_instability",
    "current_bounded_line_insufficiency",
    "nonlocal_reconciliation_pressure",
    "narrow_reopening_pressure_without_graph_wide_leap",
)

REQUIRED_GATE11A_STATUS_KEYS = (
    "gate10_closeout_preservation_status",
    "named_operator_pressure_case_status",
    "admissible_pressure_class_status",
    "named_operator_pressure_admissibility_status",
    "broader_trusted_tree_settlement_still_unearned_status",
    "operator_admission_still_denied_status",
    "retroactive_reinterpretation_forbidden_status",
    "graph_wide_operator_leap_pressure_status",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Gate11B bounded-line insufficiency declarability audit from the "
            "frozen Gate11A admissibility run without deciding reopening eligibility "
            "or reopening the operator line."
        )
    )
    parser.add_argument("--gate11a-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_status_value(status_payload: Dict[str, Any], key: str) -> str:
    return str(status_payload.get(key, ""))


def source_is_incomplete(
    source_gate11a_manifest: Dict[str, Any], source_gate11a_status: Dict[str, Any], report_text: str
) -> bool:
    if not str(source_gate11a_manifest.get("run_id") or ""):
        return True
    if not report_text.strip():
        return True
    return any(not source_status_value(source_gate11a_status, key) for key in REQUIRED_GATE11A_STATUS_KEYS)


def extract_explicit_marker_values(report_text: str, marker: str) -> List[str]:
    pattern = re.compile(rf"(?im)^\s*{re.escape(marker)}\s*[:=]\s*([a-z_]+)\s*$")
    return [match.group(1).lower() for match in pattern.finditer(report_text)]


def extract_single_explicit_status(
    report_text: str, marker: str, allowed_values: Sequence[str]
) -> str:
    values = extract_explicit_marker_values(report_text, marker)
    if not values:
        return ""
    if any(value not in allowed_values for value in values):
        return "deferred"
    distinct = list(dict.fromkeys(values))
    if len(distinct) > 1:
        return "deferred"
    return distinct[0]


def extract_candidate_declaration(report_text: str) -> Dict[str, str]:
    candidate_values = []
    candidate_values.extend(
        extract_explicit_marker_values(report_text, "bounded_line_insufficiency_candidate")
    )
    candidate_values.extend(
        extract_explicit_marker_values(report_text, "bounded_line_insufficiency_class")
    )
    candidate_values.extend(
        extract_explicit_marker_values(report_text, "bounded_line_insufficiency_class_status")
    )

    candidate_status_marker = extract_single_explicit_status(
        report_text,
        "bounded_line_insufficiency_candidate_status",
        ("present", "absent", "deferred"),
    )
    settlement_inflation_pressure_status = extract_single_explicit_status(
        report_text,
        "settlement_inflation_pressure_status",
        ("absent", "present"),
    )
    graph_wide_operator_leap_pressure_status = extract_single_explicit_status(
        report_text,
        "graph_wide_operator_leap_pressure_status",
        ("absent", "present"),
    )

    if (
        candidate_status_marker == "deferred"
        or settlement_inflation_pressure_status == "deferred"
        or graph_wide_operator_leap_pressure_status == "deferred"
    ):
        return {
            "bounded_line_insufficiency_candidate_status": "deferred",
            "bounded_line_insufficiency_class_status": "deferred",
            "settlement_inflation_pressure_status": "absent",
            "graph_wide_operator_leap_pressure_status": "absent",
        }

    if not settlement_inflation_pressure_status:
        settlement_inflation_pressure_status = "absent"
    if not graph_wide_operator_leap_pressure_status:
        graph_wide_operator_leap_pressure_status = "absent"

    filtered_candidate_values: List[str] = []
    for value in candidate_values:
        if value in INSUFFICIENCY_CLASSES:
            filtered_candidate_values.append(value)
        elif value in {"none", "absent"}:
            filtered_candidate_values.append("none")
        else:
            return {
                "bounded_line_insufficiency_candidate_status": "deferred",
                "bounded_line_insufficiency_class_status": "deferred",
                "settlement_inflation_pressure_status": settlement_inflation_pressure_status,
                "graph_wide_operator_leap_pressure_status": graph_wide_operator_leap_pressure_status,
            }

    distinct_candidates = list(dict.fromkeys(filtered_candidate_values))

    if len([value for value in distinct_candidates if value != "none"]) > 1:
        return {
            "bounded_line_insufficiency_candidate_status": "deferred",
            "bounded_line_insufficiency_class_status": "deferred",
            "settlement_inflation_pressure_status": settlement_inflation_pressure_status,
            "graph_wide_operator_leap_pressure_status": graph_wide_operator_leap_pressure_status,
        }

    if distinct_candidates and distinct_candidates[0] in INSUFFICIENCY_CLASSES:
        if candidate_status_marker == "absent":
            return {
                "bounded_line_insufficiency_candidate_status": "deferred",
                "bounded_line_insufficiency_class_status": "deferred",
                "settlement_inflation_pressure_status": settlement_inflation_pressure_status,
                "graph_wide_operator_leap_pressure_status": graph_wide_operator_leap_pressure_status,
            }
        return {
            "bounded_line_insufficiency_candidate_status": "present",
            "bounded_line_insufficiency_class_status": distinct_candidates[0],
            "settlement_inflation_pressure_status": settlement_inflation_pressure_status,
            "graph_wide_operator_leap_pressure_status": graph_wide_operator_leap_pressure_status,
        }

    if distinct_candidates == ["none"]:
        return {
            "bounded_line_insufficiency_candidate_status": "absent",
            "bounded_line_insufficiency_class_status": "none",
            "settlement_inflation_pressure_status": settlement_inflation_pressure_status,
            "graph_wide_operator_leap_pressure_status": graph_wide_operator_leap_pressure_status,
        }

    if candidate_status_marker == "present":
        return {
            "bounded_line_insufficiency_candidate_status": "deferred",
            "bounded_line_insufficiency_class_status": "deferred",
            "settlement_inflation_pressure_status": settlement_inflation_pressure_status,
            "graph_wide_operator_leap_pressure_status": graph_wide_operator_leap_pressure_status,
        }

    return {
        "bounded_line_insufficiency_candidate_status": "absent",
        "bounded_line_insufficiency_class_status": "none",
        "settlement_inflation_pressure_status": settlement_inflation_pressure_status,
        "graph_wide_operator_leap_pressure_status": graph_wide_operator_leap_pressure_status,
    }


def build_registry(
    source_gate11a_manifest: Dict[str, Any],
    source_gate11a_status: Dict[str, Any],
    status_payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return [
        {
            "source_gate11a_run_id": str(source_gate11a_manifest.get("run_id") or ""),
            "source_gate11a_code_git_commit": str(
                source_gate11a_manifest.get("code_git_commit") or ""
            ),
            "gate10_closeout_preservation_status": str(
                status_payload["gate10_closeout_preservation_status"]
            ),
            "gate11a_absence_result_preservation_status": str(
                status_payload["gate11a_absence_result_preservation_status"]
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
            "named_operator_pressure_case_status": source_status_value(
                source_gate11a_status, "named_operator_pressure_case_status"
            ),
            "admissible_pressure_class_status": source_status_value(
                source_gate11a_status, "admissible_pressure_class_status"
            ),
            "named_operator_pressure_admissibility_status": source_status_value(
                source_gate11a_status, "named_operator_pressure_admissibility_status"
            ),
            "bounded_line_insufficiency_candidate_status": str(
                status_payload["bounded_line_insufficiency_candidate_status"]
            ),
            "bounded_line_insufficiency_class_status": str(
                status_payload["bounded_line_insufficiency_class_status"]
            ),
            "settlement_inflation_pressure_status": str(
                status_payload["settlement_inflation_pressure_status"]
            ),
            "graph_wide_operator_leap_pressure_status": str(
                status_payload["graph_wide_operator_leap_pressure_status"]
            ),
            "bounded_line_insufficiency_declarability_status": str(
                status_payload["bounded_line_insufficiency_declarability_status"]
            ),
        }
    ]


def build_policy_compare(
    registry_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [
        {
            "source_gate11a_run_id": str(row["source_gate11a_run_id"]),
            "gate10_closeout_preservation_status": str(
                row["gate10_closeout_preservation_status"]
            ),
            "gate11a_absence_result_preservation_status": str(
                row["gate11a_absence_result_preservation_status"]
            ),
            "broader_trusted_tree_settlement_still_unearned_status": str(
                row["broader_trusted_tree_settlement_still_unearned_status"]
            ),
            "operator_admission_still_denied_status": str(
                row["operator_admission_still_denied_status"]
            ),
            "retroactive_reinterpretation_forbidden_status": str(
                row["retroactive_reinterpretation_forbidden_status"]
            ),
            "bounded_line_insufficiency_candidate_status": str(
                row["bounded_line_insufficiency_candidate_status"]
            ),
            "bounded_line_insufficiency_class_status": str(
                row["bounded_line_insufficiency_class_status"]
            ),
            "settlement_inflation_pressure_status": str(
                row["settlement_inflation_pressure_status"]
            ),
            "graph_wide_operator_leap_pressure_status": str(
                row["graph_wide_operator_leap_pressure_status"]
            ),
            "bounded_line_insufficiency_declarability_status": str(
                row["bounded_line_insufficiency_declarability_status"]
            ),
        }
        for row in registry_rows
    ]


def build_status_payload(
    source_gate11a_manifest: Dict[str, Any],
    source_gate11a_status: Dict[str, Any],
    report_text: str,
) -> Dict[str, Any]:
    incomplete = source_is_incomplete(source_gate11a_manifest, source_gate11a_status, report_text)

    gate10_closeout_preservation_status = (
        "preserved"
        if source_status_value(source_gate11a_status, "gate10_closeout_preservation_status")
        == "preserved"
        else "not_preserved"
    )
    gate11a_absence_result_preservation_status = (
        "preserved"
        if source_status_value(source_gate11a_status, "named_operator_pressure_case_status")
        == "absent"
        and source_status_value(source_gate11a_status, "admissible_pressure_class_status")
        == "none"
        and source_status_value(source_gate11a_status, "named_operator_pressure_admissibility_status")
        == "not_yet_admissible"
        else "not_preserved"
    )
    broader_trusted_tree_settlement_still_unearned_status = (
        "confirmed"
        if source_status_value(
            source_gate11a_status, "broader_trusted_tree_settlement_still_unearned_status"
        )
        == "confirmed"
        else "not_confirmed"
    )
    operator_admission_still_denied_status = (
        "confirmed"
        if source_status_value(source_gate11a_status, "operator_admission_still_denied_status")
        == "confirmed"
        else "not_confirmed"
    )
    retroactive_reinterpretation_forbidden_status = (
        "confirmed"
        if source_status_value(
            source_gate11a_status, "retroactive_reinterpretation_forbidden_status"
        )
        == "confirmed"
        else "not_confirmed"
    )

    if incomplete:
        bounded_line_insufficiency_candidate_status = "deferred"
        bounded_line_insufficiency_class_status = "deferred"
        settlement_inflation_pressure_status = "absent"
        graph_wide_operator_leap_pressure_status = "absent"
    else:
        extracted = extract_candidate_declaration(report_text)
        bounded_line_insufficiency_candidate_status = extracted[
            "bounded_line_insufficiency_candidate_status"
        ]
        bounded_line_insufficiency_class_status = extracted[
            "bounded_line_insufficiency_class_status"
        ]
        settlement_inflation_pressure_status = extracted[
            "settlement_inflation_pressure_status"
        ]
        graph_wide_operator_leap_pressure_status = extracted[
            "graph_wide_operator_leap_pressure_status"
        ]

    if broader_trusted_tree_settlement_still_unearned_status != "confirmed":
        settlement_inflation_pressure_status = "present"

    if incomplete or bounded_line_insufficiency_candidate_status == "deferred":
        bounded_line_insufficiency_declarability_status = "deferred"
    elif (
        settlement_inflation_pressure_status == "present"
        or retroactive_reinterpretation_forbidden_status != "confirmed"
        or graph_wide_operator_leap_pressure_status == "present"
    ):
        bounded_line_insufficiency_declarability_status = "denied"
    elif (
        gate10_closeout_preservation_status == "preserved"
        and gate11a_absence_result_preservation_status == "preserved"
        and bounded_line_insufficiency_candidate_status == "present"
        and bounded_line_insufficiency_class_status in INSUFFICIENCY_CLASSES
    ):
        bounded_line_insufficiency_declarability_status = "declarable"
    else:
        bounded_line_insufficiency_declarability_status = "not_yet_declarable"

    if gate10_closeout_preservation_status != "preserved":
        next_named_blocker = "gate10_closeout_not_preserved"
    elif gate11a_absence_result_preservation_status != "preserved":
        next_named_blocker = "gate11a_absence_result_not_preserved"
    elif broader_trusted_tree_settlement_still_unearned_status != "confirmed":
        next_named_blocker = "broader_trusted_tree_settlement_not_explicitly_unearned"
    elif operator_admission_still_denied_status != "confirmed":
        next_named_blocker = "operator_admission_not_denied"
    elif retroactive_reinterpretation_forbidden_status != "confirmed":
        next_named_blocker = "retroactive_reinterpretation_pressure"
    elif bounded_line_insufficiency_declarability_status == "deferred":
        if incomplete:
            next_named_blocker = "controlling_source_incomplete"
        elif bounded_line_insufficiency_candidate_status == "deferred":
            next_named_blocker = "multiple_bounded_line_insufficiency_candidates"
        else:
            next_named_blocker = "controlling_source_incomplete"
    elif settlement_inflation_pressure_status == "present":
        next_named_blocker = "settlement_inflation_pressure"
    elif graph_wide_operator_leap_pressure_status == "present":
        next_named_blocker = "graph_wide_operator_leap_pressure"
    elif bounded_line_insufficiency_candidate_status == "absent":
        next_named_blocker = "no_bounded_line_insufficiency_candidate"
    else:
        next_named_blocker = ""

    return {
        "gate10_closeout_preservation_status": gate10_closeout_preservation_status,
        "gate11a_absence_result_preservation_status": gate11a_absence_result_preservation_status,
        "broader_trusted_tree_settlement_still_unearned_status": broader_trusted_tree_settlement_still_unearned_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "retroactive_reinterpretation_forbidden_status": retroactive_reinterpretation_forbidden_status,
        "bounded_line_insufficiency_candidate_status": bounded_line_insufficiency_candidate_status,
        "bounded_line_insufficiency_class_status": bounded_line_insufficiency_class_status,
        "settlement_inflation_pressure_status": settlement_inflation_pressure_status,
        "graph_wide_operator_leap_pressure_status": graph_wide_operator_leap_pressure_status,
        "bounded_line_insufficiency_declarability_status": bounded_line_insufficiency_declarability_status,
        "next_named_blocker": next_named_blocker,
    }


def build_report(
    run_id: str,
    source_gate11a_manifest: Dict[str, Any],
    policy_compare_rows: Sequence[Dict[str, Any]],
    status_payload: Dict[str, Any],
) -> str:
    lines = [
        "# Gate11B Bounded-Line Insufficiency Declarability Read",
        "",
        f"run_id: {run_id}",
        f"source_gate11a_run_id: {source_gate11a_manifest.get('run_id', '')}",
        f"source_gate11a_code_git_commit: {source_gate11a_manifest.get('code_git_commit', '')}",
        "",
        "## Discipline",
        "",
        "- Gate11B asks only whether one bounded-line insufficiency candidate may be declared honestly",
        "- Gate10 closeout remains bounded and preserved",
        "- Gate11A absence remains preserved unless one explicit declaration already exists in the same frozen run",
        "- broader trusted-tree settlement remains unearned",
        "- operator admission remains denied",
        "- no retroactive rewrite of Gate9 or Gate10 memory is permitted",
        "",
        "## Source Summary",
        "",
        "| source_gate11a_run_id | gate10_closeout_preservation_status | gate11a_absence_result_preservation_status | broader_trusted_tree_settlement_still_unearned_status | operator_admission_still_denied_status | retroactive_reinterpretation_forbidden_status | bounded_line_insufficiency_candidate_status | bounded_line_insufficiency_class_status | settlement_inflation_pressure_status | graph_wide_operator_leap_pressure_status | bounded_line_insufficiency_declarability_status |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in policy_compare_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["source_gate11a_run_id"]),
                    str(row["gate10_closeout_preservation_status"]),
                    str(row["gate11a_absence_result_preservation_status"]),
                    str(row["broader_trusted_tree_settlement_still_unearned_status"]),
                    str(row["operator_admission_still_denied_status"]),
                    str(row["retroactive_reinterpretation_forbidden_status"]),
                    str(row["bounded_line_insufficiency_candidate_status"]),
                    str(row["bounded_line_insufficiency_class_status"]),
                    str(row["settlement_inflation_pressure_status"]),
                    str(row["graph_wide_operator_leap_pressure_status"]),
                    str(row["bounded_line_insufficiency_declarability_status"]),
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
            f"- broader_trusted_tree_settlement_still_unearned_status: `{status_payload['broader_trusted_tree_settlement_still_unearned_status']}`",
            f"- operator_admission_still_denied_status: `{status_payload['operator_admission_still_denied_status']}`",
            f"- retroactive_reinterpretation_forbidden_status: `{status_payload['retroactive_reinterpretation_forbidden_status']}`",
            f"- bounded_line_insufficiency_candidate_status: `{status_payload['bounded_line_insufficiency_candidate_status']}`",
            f"- bounded_line_insufficiency_class_status: `{status_payload['bounded_line_insufficiency_class_status']}`",
            f"- settlement_inflation_pressure_status: `{status_payload['settlement_inflation_pressure_status']}`",
            f"- graph_wide_operator_leap_pressure_status: `{status_payload['graph_wide_operator_leap_pressure_status']}`",
            f"- bounded_line_insufficiency_declarability_status: `{status_payload['bounded_line_insufficiency_declarability_status']}`",
            f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
        ]
    )

    if status_payload["bounded_line_insufficiency_declarability_status"] == "declarable":
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- one bounded-line insufficiency candidate is explicitly present in the controlling Gate11A source and may be named honestly in a later slice",
                "- this does not decide reopening eligibility or reopen operator admission",
            ]
        )
    elif status_payload["bounded_line_insufficiency_declarability_status"] == "deferred":
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- the declarability audit remains deferred because the frozen controlling source record is incomplete or requires worker-side selection",
                f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
            ]
        )
    elif status_payload["bounded_line_insufficiency_declarability_status"] == "denied":
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- the proposed bounded-line insufficiency declaration is denied under the frozen Gate11B boundary",
                f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- no single explicit bounded-line insufficiency declaration is yet present in the frozen post-Gate10 line",
                f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    source_gate11a_dir = Path(args.gate11a_dir)
    out_dir = Path(args.out_dir)
    run_id = out_dir.name

    out_dir.mkdir(parents=True, exist_ok=True)

    source_gate11a_manifest = gate9a.read_json(source_gate11a_dir / gate11a.DEFAULT_MANIFEST)
    source_gate11a_status = gate9a.read_json(source_gate11a_dir / gate11a.DEFAULT_STATUS)
    source_gate11a_report = (source_gate11a_dir / gate11a.DEFAULT_REPORT).read_text(
        encoding="utf-8"
    )

    status_payload = build_status_payload(
        source_gate11a_manifest, source_gate11a_status, source_gate11a_report
    )
    registry_rows = build_registry(source_gate11a_manifest, source_gate11a_status, status_payload)
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
            "source_gate11a_run_id",
            "gate10_closeout_preservation_status",
            "gate11a_absence_result_preservation_status",
            "broader_trusted_tree_settlement_still_unearned_status",
            "operator_admission_still_denied_status",
            "retroactive_reinterpretation_forbidden_status",
            "bounded_line_insufficiency_candidate_status",
            "bounded_line_insufficiency_class_status",
            "settlement_inflation_pressure_status",
            "graph_wide_operator_leap_pressure_status",
            "bounded_line_insufficiency_declarability_status",
        ),
        policy_compare_rows,
    )
    gate9a.write_json(status_path, status_payload)
    gate9a.write_text(
        report_path,
        build_report(
            run_id=run_id,
            source_gate11a_manifest=source_gate11a_manifest,
            policy_compare_rows=policy_compare_rows,
            status_payload=status_payload,
        ),
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "run_id": run_id,
        "code_git_commit": gate9a.current_git_commit(),
        "source_gate11a_dir": gate9a.repo_relative_or_posix(source_gate11a_dir),
        "source_gate11a_run_id": str(source_gate11a_manifest.get("run_id") or ""),
        "source_gate11a_code_git_commit": str(
            source_gate11a_manifest.get("code_git_commit") or ""
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