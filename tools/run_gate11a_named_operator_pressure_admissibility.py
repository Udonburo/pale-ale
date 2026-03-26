#!/usr/bin/env python3
"""Run a Gate11A named operator-pressure admissibility audit on Gate10F outputs."""

import argparse
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

import run_gate10f_pre_closeout_judgment as gate10f
import run_gate9a_graph_gauge_consumer as gate9a


REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "gate11a_named_operator_pressure_admissibility_v1"
METHOD_ID = "gate11a_named_operator_pressure_admissibility_v1"

DEFAULT_REGISTRY = "named_operator_pressure_admissibility_registry.jsonl"
DEFAULT_POLICY_COMPARE = "named_operator_pressure_admissibility_policy_compare.csv"
DEFAULT_STATUS = "named_operator_pressure_admissibility_status.json"
DEFAULT_REPORT = "gate11a_named_operator_pressure_admissibility_read.md"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"

ADMISSIBLE_PRESSURE_CLASSES = (
    "tree_choice_instability",
    "current_bounded_line_insufficiency",
    "nonlocal_reconciliation_pressure",
    "narrow_reopening_pressure_without_graph_wide_leap",
)

REQUIRED_GATE10F_STATUS_KEYS = (
    "closeout_judgment_outcome_status",
    "closeout_sentence_support_status",
    "broader_trusted_tree_settlement_status",
    "operator_admission_still_denied_status",
    "retroactive_reinterpretation_forbidden_status",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Gate11A named operator-pressure admissibility audit from the "
            "frozen Gate10F closeout-support line without deciding reopening "
            "eligibility or reopening the operator line."
        )
    )
    parser.add_argument("--gate10f-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_status_value(status_payload: Dict[str, Any], key: str) -> str:
    return str(status_payload.get(key, ""))


def source_is_incomplete(
    source_gate10f_manifest: Dict[str, Any], source_gate10f_status: Dict[str, Any], report_text: str
) -> bool:
    if not str(source_gate10f_manifest.get("run_id") or ""):
        return True
    if not report_text.strip():
        return True
    return any(not source_status_value(source_gate10f_status, key) for key in REQUIRED_GATE10F_STATUS_KEYS)


def extract_explicit_marker_values(report_text: str, marker: str) -> List[str]:
    pattern = re.compile(
        rf"(?im)^\s*{re.escape(marker)}\s*[:=]\s*([a-z_]+)\s*$"
    )
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


def extract_pressure_case(report_text: str) -> Dict[str, str]:
    named_case_marker = extract_single_explicit_status(
        report_text,
        "named_operator_pressure_case",
        ADMISSIBLE_PRESSURE_CLASSES + ("none", "absent", "deferred"),
    )
    named_case_status_marker = extract_single_explicit_status(
        report_text,
        "named_operator_pressure_case_status",
        ("present", "absent", "deferred"),
    )

    if named_case_marker == "deferred" or named_case_status_marker == "deferred":
        return {
            "named_operator_pressure_case_status": "deferred",
            "admissible_pressure_class_status": "deferred",
            "bounded_line_insufficiency_evidence_status": "deferred",
            "graph_wide_operator_leap_pressure_status": "absent",
        }

    if named_case_marker in ADMISSIBLE_PRESSURE_CLASSES:
        named_operator_pressure_case_status = "present"
        admissible_pressure_class_status = named_case_marker
    elif named_case_marker in {"none", "absent"}:
        named_operator_pressure_case_status = "absent"
        admissible_pressure_class_status = "none"
    elif named_case_status_marker == "present":
        return {
            "named_operator_pressure_case_status": "deferred",
            "admissible_pressure_class_status": "deferred",
            "bounded_line_insufficiency_evidence_status": "deferred",
            "graph_wide_operator_leap_pressure_status": "absent",
        }
    else:
        named_operator_pressure_case_status = "absent"
        admissible_pressure_class_status = "none"

    graph_wide_operator_leap_pressure_status = extract_single_explicit_status(
        report_text,
        "graph_wide_operator_leap_pressure_status",
        ("present", "absent"),
    )
    if graph_wide_operator_leap_pressure_status == "deferred":
        return {
            "named_operator_pressure_case_status": "deferred",
            "admissible_pressure_class_status": "deferred",
            "bounded_line_insufficiency_evidence_status": "deferred",
            "graph_wide_operator_leap_pressure_status": "absent",
        }
    if not graph_wide_operator_leap_pressure_status:
        graph_wide_operator_leap_pressure_status = "absent"

    bounded_line_insufficiency_evidence_status = extract_single_explicit_status(
        report_text,
        "bounded_line_insufficiency_evidence_status",
        ("present", "absent", "deferred"),
    )
    if bounded_line_insufficiency_evidence_status == "deferred":
        bounded_line_insufficiency_evidence_status = "deferred"
    elif bounded_line_insufficiency_evidence_status:
        bounded_line_insufficiency_evidence_status = bounded_line_insufficiency_evidence_status
    else:
        bounded_line_insufficiency_evidence_status = "absent"

    if named_operator_pressure_case_status == "absent":
        bounded_line_insufficiency_evidence_status = "absent"

    return {
        "named_operator_pressure_case_status": named_operator_pressure_case_status,
        "admissible_pressure_class_status": admissible_pressure_class_status,
        "bounded_line_insufficiency_evidence_status": bounded_line_insufficiency_evidence_status,
        "graph_wide_operator_leap_pressure_status": graph_wide_operator_leap_pressure_status,
    }


def build_registry(
    source_gate10f_manifest: Dict[str, Any],
    source_gate10f_status: Dict[str, Any],
    status_payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return [
        {
            "source_gate10f_run_id": str(source_gate10f_manifest.get("run_id") or ""),
            "source_gate10f_code_git_commit": str(
                source_gate10f_manifest.get("code_git_commit") or ""
            ),
            "gate10_closeout_preservation_status": str(
                status_payload["gate10_closeout_preservation_status"]
            ),
            "bounded_closeout_support_preservation_status": str(
                status_payload["bounded_closeout_support_preservation_status"]
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
            "closeout_judgment_outcome_status": source_status_value(
                source_gate10f_status, "closeout_judgment_outcome_status"
            ),
            "closeout_sentence_support_status": source_status_value(
                source_gate10f_status, "closeout_sentence_support_status"
            ),
            "named_operator_pressure_case_status": str(
                status_payload["named_operator_pressure_case_status"]
            ),
            "admissible_pressure_class_status": str(
                status_payload["admissible_pressure_class_status"]
            ),
            "bounded_line_insufficiency_evidence_status": str(
                status_payload["bounded_line_insufficiency_evidence_status"]
            ),
            "graph_wide_operator_leap_pressure_status": str(
                status_payload["graph_wide_operator_leap_pressure_status"]
            ),
            "named_operator_pressure_admissibility_status": str(
                status_payload["named_operator_pressure_admissibility_status"]
            ),
        }
    ]


def build_policy_compare(
    registry_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [
        {
            "source_gate10f_run_id": str(row["source_gate10f_run_id"]),
            "gate10_closeout_preservation_status": str(
                row["gate10_closeout_preservation_status"]
            ),
            "bounded_closeout_support_preservation_status": str(
                row["bounded_closeout_support_preservation_status"]
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
            "named_operator_pressure_case_status": str(
                row["named_operator_pressure_case_status"]
            ),
            "admissible_pressure_class_status": str(
                row["admissible_pressure_class_status"]
            ),
            "bounded_line_insufficiency_evidence_status": str(
                row["bounded_line_insufficiency_evidence_status"]
            ),
            "graph_wide_operator_leap_pressure_status": str(
                row["graph_wide_operator_leap_pressure_status"]
            ),
            "named_operator_pressure_admissibility_status": str(
                row["named_operator_pressure_admissibility_status"]
            ),
        }
        for row in registry_rows
    ]


def build_status_payload(
    source_gate10f_manifest: Dict[str, Any],
    source_gate10f_status: Dict[str, Any],
    report_text: str,
) -> Dict[str, Any]:
    incomplete = source_is_incomplete(source_gate10f_manifest, source_gate10f_status, report_text)

    gate10_closeout_preservation_status = (
        "preserved"
        if source_status_value(source_gate10f_status, "closeout_judgment_outcome_status")
        == "closeout_supported"
        else "not_preserved"
    )
    bounded_closeout_support_preservation_status = (
        "preserved"
        if source_status_value(source_gate10f_status, "closeout_sentence_support_status")
        == "supported"
        else "not_preserved"
    )
    broader_trusted_tree_settlement_still_unearned_status = (
        "confirmed"
        if source_status_value(source_gate10f_status, "broader_trusted_tree_settlement_status")
        == "unearned"
        else "not_confirmed"
    )
    operator_admission_still_denied_status = (
        "confirmed"
        if source_status_value(source_gate10f_status, "operator_admission_still_denied_status")
        == "confirmed"
        else "not_confirmed"
    )
    retroactive_reinterpretation_forbidden_status = (
        "confirmed"
        if source_status_value(
            source_gate10f_status, "retroactive_reinterpretation_forbidden_status"
        )
        == "confirmed"
        else "not_confirmed"
    )

    if incomplete:
        named_operator_pressure_case_status = "deferred"
        admissible_pressure_class_status = "deferred"
        bounded_line_insufficiency_evidence_status = "deferred"
        graph_wide_operator_leap_pressure_status = "absent"
    else:
        extracted = extract_pressure_case(report_text)
        named_operator_pressure_case_status = extracted["named_operator_pressure_case_status"]
        admissible_pressure_class_status = extracted["admissible_pressure_class_status"]
        bounded_line_insufficiency_evidence_status = extracted[
            "bounded_line_insufficiency_evidence_status"
        ]
        graph_wide_operator_leap_pressure_status = extracted[
            "graph_wide_operator_leap_pressure_status"
        ]

    if incomplete:
        named_operator_pressure_admissibility_status = "deferred"
    elif (
        broader_trusted_tree_settlement_still_unearned_status != "confirmed"
        or operator_admission_still_denied_status != "confirmed"
        or retroactive_reinterpretation_forbidden_status != "confirmed"
        or graph_wide_operator_leap_pressure_status == "present"
    ):
        named_operator_pressure_admissibility_status = "denied"
    elif (
        gate10_closeout_preservation_status == "preserved"
        and bounded_closeout_support_preservation_status == "preserved"
        and named_operator_pressure_case_status == "present"
        and admissible_pressure_class_status not in {"none", "deferred"}
        and bounded_line_insufficiency_evidence_status == "present"
    ):
        named_operator_pressure_admissibility_status = "admissible"
    else:
        named_operator_pressure_admissibility_status = "not_yet_admissible"

    if gate10_closeout_preservation_status != "preserved":
        next_named_blocker = "gate10_closeout_not_preserved"
    elif bounded_closeout_support_preservation_status != "preserved":
        next_named_blocker = "bounded_closeout_support_not_preserved"
    elif broader_trusted_tree_settlement_still_unearned_status != "confirmed":
        next_named_blocker = "broader_trusted_tree_settlement_not_explicitly_unearned"
    elif operator_admission_still_denied_status != "confirmed":
        next_named_blocker = "operator_admission_not_denied"
    elif retroactive_reinterpretation_forbidden_status != "confirmed":
        next_named_blocker = "retroactive_reinterpretation_pressure"
    elif named_operator_pressure_admissibility_status == "deferred":
        next_named_blocker = "controlling_source_incomplete"
    elif graph_wide_operator_leap_pressure_status == "present":
        next_named_blocker = "graph_wide_operator_leap_pressure"
    elif named_operator_pressure_case_status == "absent":
        next_named_blocker = "no_named_operator_pressure_case"
    elif bounded_line_insufficiency_evidence_status == "absent":
        next_named_blocker = "bounded_line_insufficiency_evidence_absent"
    else:
        next_named_blocker = ""

    return {
        "gate10_closeout_preservation_status": gate10_closeout_preservation_status,
        "bounded_closeout_support_preservation_status": bounded_closeout_support_preservation_status,
        "broader_trusted_tree_settlement_still_unearned_status": broader_trusted_tree_settlement_still_unearned_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "retroactive_reinterpretation_forbidden_status": retroactive_reinterpretation_forbidden_status,
        "named_operator_pressure_case_status": named_operator_pressure_case_status,
        "admissible_pressure_class_status": admissible_pressure_class_status,
        "bounded_line_insufficiency_evidence_status": bounded_line_insufficiency_evidence_status,
        "graph_wide_operator_leap_pressure_status": graph_wide_operator_leap_pressure_status,
        "named_operator_pressure_admissibility_status": named_operator_pressure_admissibility_status,
        "next_named_blocker": next_named_blocker,
    }


def build_report(
    run_id: str,
    source_gate10f_manifest: Dict[str, Any],
    policy_compare_rows: Sequence[Dict[str, Any]],
    status_payload: Dict[str, Any],
) -> str:
    lines = [
        "# Gate11A Named Operator-Pressure Admissibility Read",
        "",
        f"run_id: {run_id}",
        f"source_gate10f_run_id: {source_gate10f_manifest.get('run_id', '')}",
        f"source_gate10f_code_git_commit: {source_gate10f_manifest.get('code_git_commit', '')}",
        "",
        "## Discipline",
        "",
        "- Gate11A asks only whether any real named operator-pressure case exists at all",
        "- Gate10 closeout remains bounded and preserved",
        "- broader trusted-tree settlement remains unearned",
        "- operator admission remains denied",
        "- no retroactive rewrite of Gate9 or Gate10 memory is permitted",
        "",
        "## Source Summary",
        "",
        "| source_gate10f_run_id | gate10_closeout_preservation_status | bounded_closeout_support_preservation_status | broader_trusted_tree_settlement_still_unearned_status | operator_admission_still_denied_status | retroactive_reinterpretation_forbidden_status | named_operator_pressure_case_status | admissible_pressure_class_status | bounded_line_insufficiency_evidence_status | graph_wide_operator_leap_pressure_status | named_operator_pressure_admissibility_status |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in policy_compare_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["source_gate10f_run_id"]),
                    str(row["gate10_closeout_preservation_status"]),
                    str(row["bounded_closeout_support_preservation_status"]),
                    str(row["broader_trusted_tree_settlement_still_unearned_status"]),
                    str(row["operator_admission_still_denied_status"]),
                    str(row["retroactive_reinterpretation_forbidden_status"]),
                    str(row["named_operator_pressure_case_status"]),
                    str(row["admissible_pressure_class_status"]),
                    str(row["bounded_line_insufficiency_evidence_status"]),
                    str(row["graph_wide_operator_leap_pressure_status"]),
                    str(row["named_operator_pressure_admissibility_status"]),
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
            f"- bounded_closeout_support_preservation_status: `{status_payload['bounded_closeout_support_preservation_status']}`",
            f"- broader_trusted_tree_settlement_still_unearned_status: `{status_payload['broader_trusted_tree_settlement_still_unearned_status']}`",
            f"- operator_admission_still_denied_status: `{status_payload['operator_admission_still_denied_status']}`",
            f"- retroactive_reinterpretation_forbidden_status: `{status_payload['retroactive_reinterpretation_forbidden_status']}`",
            f"- named_operator_pressure_case_status: `{status_payload['named_operator_pressure_case_status']}`",
            f"- admissible_pressure_class_status: `{status_payload['admissible_pressure_class_status']}`",
            f"- bounded_line_insufficiency_evidence_status: `{status_payload['bounded_line_insufficiency_evidence_status']}`",
            f"- graph_wide_operator_leap_pressure_status: `{status_payload['graph_wide_operator_leap_pressure_status']}`",
            f"- named_operator_pressure_admissibility_status: `{status_payload['named_operator_pressure_admissibility_status']}`",
            f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
        ]
    )

    if status_payload["named_operator_pressure_admissibility_status"] == "admissible":
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- a real named operator-pressure case is explicitly present in the controlling Gate10F source and is admissible for a later eligibility slice",
                "- this does not reopen operator admission or decide reopening eligibility",
            ]
        )
    elif status_payload["named_operator_pressure_admissibility_status"] == "deferred":
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- the admissibility audit remains deferred because the frozen controlling source record is incomplete or contradictory",
                f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
            ]
        )
    elif status_payload["named_operator_pressure_admissibility_status"] == "denied":
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- the proposed operator-pressure line is denied under the frozen Gate11A boundary",
                f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- no real admissible named operator-pressure case is yet explicit in the frozen post-Gate10 line",
                f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    source_gate10f_dir = Path(args.gate10f_dir)
    out_dir = Path(args.out_dir)
    run_id = out_dir.name

    out_dir.mkdir(parents=True, exist_ok=True)

    source_gate10f_manifest = gate9a.read_json(source_gate10f_dir / gate10f.DEFAULT_MANIFEST)
    source_gate10f_status = gate9a.read_json(source_gate10f_dir / gate10f.DEFAULT_STATUS)
    source_gate10f_report = (source_gate10f_dir / gate10f.DEFAULT_REPORT).read_text(
        encoding="utf-8"
    )

    status_payload = build_status_payload(
        source_gate10f_manifest, source_gate10f_status, source_gate10f_report
    )
    registry_rows = build_registry(source_gate10f_manifest, source_gate10f_status, status_payload)
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
            "source_gate10f_run_id",
            "gate10_closeout_preservation_status",
            "bounded_closeout_support_preservation_status",
            "broader_trusted_tree_settlement_still_unearned_status",
            "operator_admission_still_denied_status",
            "retroactive_reinterpretation_forbidden_status",
            "named_operator_pressure_case_status",
            "admissible_pressure_class_status",
            "bounded_line_insufficiency_evidence_status",
            "graph_wide_operator_leap_pressure_status",
            "named_operator_pressure_admissibility_status",
        ),
        policy_compare_rows,
    )
    gate9a.write_json(status_path, status_payload)
    gate9a.write_text(
        report_path,
        build_report(
            run_id=run_id,
            source_gate10f_manifest=source_gate10f_manifest,
            policy_compare_rows=policy_compare_rows,
            status_payload=status_payload,
        ),
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "run_id": run_id,
        "code_git_commit": gate9a.current_git_commit(),
        "source_gate10f_dir": gate9a.repo_relative_or_posix(source_gate10f_dir),
        "source_gate10f_run_id": str(source_gate10f_manifest.get("run_id") or ""),
        "source_gate10f_code_git_commit": str(
            source_gate10f_manifest.get("code_git_commit") or ""
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
