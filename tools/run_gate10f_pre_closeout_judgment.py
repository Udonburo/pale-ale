#!/usr/bin/env python3
"""Run a Gate10F pre-closeout / closeout judgment on Gate10E outputs."""

import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Sequence

import run_gate10e_interim_broader_judgment as gate10e
import run_gate9a_graph_gauge_consumer as gate9a


REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "gate10f_pre_closeout_judgment_v1"
METHOD_ID = "gate10f_pre_closeout_judgment_v1"

DEFAULT_REGISTRY = "gate10_pre_closeout_judgment_registry.jsonl"
DEFAULT_POLICY_COMPARE = "gate10_pre_closeout_judgment_policy_compare.csv"
DEFAULT_STATUS = "gate10_pre_closeout_judgment_status.json"
DEFAULT_REPORT = "gate10f_pre_closeout_judgment_read.md"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Gate10F pre-closeout / closeout judgment from the frozen "
            "Gate10E interim broader judgment without promoting it into broader "
            "settlement or operator reopening."
        )
    )
    parser.add_argument("--gate10e-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_status_value(status_payload: Dict[str, Any], key: str) -> str:
    return str(status_payload.get(key, ""))


def build_registry(
    source_gate10e_manifest: Dict[str, Any],
    source_gate10e_status: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return [
        {
            "source_gate10e_run_id": str(source_gate10e_manifest.get("run_id") or ""),
            "source_gate10e_code_git_commit": str(
                source_gate10e_manifest.get("code_git_commit") or ""
            ),
            "gate10b_slice_settled_status": source_status_value(
                source_gate10e_status, "gate10b_slice_settled_status"
            ),
            "gate10c_slice_settled_status": source_status_value(
                source_gate10e_status, "gate10c_slice_settled_status"
            ),
            "gate10d_slice_settled_status": source_status_value(
                source_gate10e_status, "gate10d_slice_settled_status"
            ),
            "three_slice_pattern_status": source_status_value(
                source_gate10e_status, "three_slice_pattern_status"
            ),
            "interim_broader_judgment_status": source_status_value(
                source_gate10e_status, "interim_broader_judgment_status"
            ),
            "pre_closeout_readiness_status": source_status_value(
                source_gate10e_status, "pre_closeout_readiness_status"
            ),
            "operator_admission_still_denied_status": source_status_value(
                source_gate10e_status, "operator_admission_still_denied_status"
            ),
            "retroactive_reinterpretation_forbidden_status": source_status_value(
                source_gate10e_status, "retroactive_reinterpretation_forbidden_status"
            ),
            "broader_trusted_tree_settlement_status": source_status_value(
                source_gate10e_status, "broader_trusted_tree_settlement_status"
            ),
        }
    ]


def build_policy_compare(
    registry_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [
        {
            "source_gate10e_run_id": str(row["source_gate10e_run_id"]),
            "three_slice_pattern_status": str(row["three_slice_pattern_status"]),
            "interim_broader_judgment_status": str(row["interim_broader_judgment_status"]),
            "pre_closeout_readiness_status": str(row["pre_closeout_readiness_status"]),
            "operator_admission_still_denied_status": str(
                row["operator_admission_still_denied_status"]
            ),
            "retroactive_reinterpretation_forbidden_status": str(
                row["retroactive_reinterpretation_forbidden_status"]
            ),
            "broader_trusted_tree_settlement_status": str(
                row["broader_trusted_tree_settlement_status"]
            ),
        }
        for row in registry_rows
    ]


def build_status_payload(source_gate10e_status: Dict[str, Any]) -> Dict[str, Any]:
    three_slice_pattern_preserved = (
        source_status_value(source_gate10e_status, "three_slice_pattern_status")
        == "supported"
        and source_status_value(source_gate10e_status, "gate10b_slice_settled_status")
        == "preserved"
        and source_status_value(source_gate10e_status, "gate10c_slice_settled_status")
        == "preserved"
        and source_status_value(source_gate10e_status, "gate10d_slice_settled_status")
        == "preserved"
    )

    bounded_support_preservation_status = (
        "preserved"
        if source_status_value(source_gate10e_status, "interim_broader_judgment_status")
        == "bounded_support"
        else "not_preserved"
    )

    pre_closeout_readiness_preservation_status = (
        "preserved"
        if source_status_value(source_gate10e_status, "pre_closeout_readiness_status")
        == "ready"
        else "not_preserved"
    )

    broader_trusted_tree_settlement_status = (
        "unearned"
        if source_status_value(
            source_gate10e_status, "broader_trusted_tree_settlement_status"
        )
        == "unearned"
        else "pressure_to_overclaim"
    )

    operator_admission_still_denied_status = source_status_value(
        source_gate10e_status, "operator_admission_still_denied_status"
    )
    retroactive_reinterpretation_forbidden_status = source_status_value(
        source_gate10e_status, "retroactive_reinterpretation_forbidden_status"
    )

    overclaim_pressure_status = (
        "present"
        if broader_trusted_tree_settlement_status != "unearned"
        or operator_admission_still_denied_status != "confirmed"
        or retroactive_reinterpretation_forbidden_status != "confirmed"
        else "absent"
    )

    closeout_sentence_support_status = (
        "supported"
        if three_slice_pattern_preserved
        and bounded_support_preservation_status == "preserved"
        and pre_closeout_readiness_preservation_status == "preserved"
        and overclaim_pressure_status == "absent"
        else "not_supported"
    )

    closeout_judgment_outcome_status = (
        "closeout_supported"
        if closeout_sentence_support_status == "supported"
        else "not_yet_closeable"
    )
    post_closeout_memory_readiness_status = (
        "ready" if closeout_judgment_outcome_status == "closeout_supported" else "not_ready"
    )

    if not three_slice_pattern_preserved:
        next_named_blocker = "three_slice_pattern_not_preserved"
    elif bounded_support_preservation_status != "preserved":
        next_named_blocker = "bounded_broader_support_not_preserved"
    elif pre_closeout_readiness_preservation_status != "preserved":
        next_named_blocker = "pre_closeout_readiness_not_preserved"
    elif broader_trusted_tree_settlement_status != "unearned":
        next_named_blocker = "broader_trusted_tree_settlement_overclaim_pressure"
    elif operator_admission_still_denied_status != "confirmed":
        next_named_blocker = "operator_admission_pressure"
    elif retroactive_reinterpretation_forbidden_status != "confirmed":
        next_named_blocker = "retroactive_reinterpretation_pressure"
    elif closeout_sentence_support_status != "supported":
        next_named_blocker = "closeout_sentence_not_supported"
    elif overclaim_pressure_status != "absent":
        next_named_blocker = "closeout_overclaim_pressure"
    else:
        next_named_blocker = ""

    return {
        "bounded_support_preservation_status": bounded_support_preservation_status,
        "pre_closeout_readiness_preservation_status": pre_closeout_readiness_preservation_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "retroactive_reinterpretation_forbidden_status": retroactive_reinterpretation_forbidden_status,
        "broader_trusted_tree_settlement_status": broader_trusted_tree_settlement_status,
        "closeout_sentence_support_status": closeout_sentence_support_status,
        "overclaim_pressure_status": overclaim_pressure_status,
        "closeout_judgment_outcome_status": closeout_judgment_outcome_status,
        "post_closeout_memory_readiness_status": post_closeout_memory_readiness_status,
        "next_named_blocker": next_named_blocker,
    }


def build_report(
    run_id: str,
    source_gate10e_manifest: Dict[str, Any],
    policy_compare_rows: Sequence[Dict[str, Any]],
    status_payload: Dict[str, Any],
) -> str:
    lines = [
        "# Gate10F Pre-Closeout Judgment Read",
        "",
        f"run_id: {run_id}",
        f"source_gate10e_run_id: {source_gate10e_manifest.get('run_id', '')}",
        f"source_gate10e_code_git_commit: {source_gate10e_manifest.get('code_git_commit', '')}",
        "",
        "## Discipline",
        "",
        "- Gate10F decides only whether a bounded closeout sentence is now honest",
        "- broader trusted-tree settlement remains unearned",
        "- operator admission remains denied",
        "- no retroactive rewrite of Gate9 or Gate10A-E is permitted",
        "",
        "## Source Summary",
        "",
        "| source_gate10e_run_id | three_slice_pattern_status | interim_broader_judgment_status | pre_closeout_readiness_status | operator_admission_still_denied_status | retroactive_reinterpretation_forbidden_status | broader_trusted_tree_settlement_status |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in policy_compare_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["source_gate10e_run_id"]),
                    str(row["three_slice_pattern_status"]),
                    str(row["interim_broader_judgment_status"]),
                    str(row["pre_closeout_readiness_status"]),
                    str(row["operator_admission_still_denied_status"]),
                    str(row["retroactive_reinterpretation_forbidden_status"]),
                    str(row["broader_trusted_tree_settlement_status"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Status",
            "",
            f"- bounded_support_preservation_status: `{status_payload['bounded_support_preservation_status']}`",
            f"- pre_closeout_readiness_preservation_status: `{status_payload['pre_closeout_readiness_preservation_status']}`",
            f"- operator_admission_still_denied_status: `{status_payload['operator_admission_still_denied_status']}`",
            f"- retroactive_reinterpretation_forbidden_status: `{status_payload['retroactive_reinterpretation_forbidden_status']}`",
            f"- broader_trusted_tree_settlement_status: `{status_payload['broader_trusted_tree_settlement_status']}`",
            f"- closeout_sentence_support_status: `{status_payload['closeout_sentence_support_status']}`",
            f"- overclaim_pressure_status: `{status_payload['overclaim_pressure_status']}`",
            f"- closeout_judgment_outcome_status: `{status_payload['closeout_judgment_outcome_status']}`",
            f"- post_closeout_memory_readiness_status: `{status_payload['post_closeout_memory_readiness_status']}`",
            f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
        ]
    )

    if status_payload["closeout_judgment_outcome_status"] == "closeout_supported":
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- the preserved Gate10E read is sufficient to support the bounded Gate10 closeout sentence allowed by the frozen spec",
                "- broader trusted-tree settlement remains unearned, operator admission remains denied, and prior memory remains non-retroactive",
            ]
        )
    elif status_payload["closeout_judgment_outcome_status"] == "deferred":
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- the closeout judgment remains deferred pending preservation of the controlling Gate10E record",
                f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- Gate10 is not yet closeable under the frozen Gate10F boundary",
                f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    source_gate10e_dir = Path(args.gate10e_dir)
    out_dir = Path(args.out_dir)
    run_id = out_dir.name

    source_gate10e_manifest = gate9a.read_json(
        source_gate10e_dir / gate10e.DEFAULT_MANIFEST
    )
    source_gate10e_status = gate9a.read_json(
        source_gate10e_dir / gate10e.DEFAULT_STATUS
    )

    registry_rows = build_registry(source_gate10e_manifest, source_gate10e_status)
    policy_compare_rows = build_policy_compare(registry_rows)
    status_payload = build_status_payload(source_gate10e_status)

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
            "source_gate10e_run_id",
            "three_slice_pattern_status",
            "interim_broader_judgment_status",
            "pre_closeout_readiness_status",
            "operator_admission_still_denied_status",
            "retroactive_reinterpretation_forbidden_status",
            "broader_trusted_tree_settlement_status",
        ),
        policy_compare_rows,
    )
    gate9a.write_json(status_path, status_payload)
    gate9a.write_text(
        report_path,
        build_report(
            run_id=run_id,
            source_gate10e_manifest=source_gate10e_manifest,
            policy_compare_rows=policy_compare_rows,
            status_payload=status_payload,
        ),
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "run_id": run_id,
        "code_git_commit": gate9a.current_git_commit(),
        "source_gate10e_dir": gate9a.repo_relative_or_posix(source_gate10e_dir),
        "source_gate10e_run_id": str(source_gate10e_manifest.get("run_id") or ""),
        "source_gate10e_code_git_commit": str(
            source_gate10e_manifest.get("code_git_commit") or ""
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