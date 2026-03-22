#!/usr/bin/env python3
"""Run a Gate10E interim broader judgment on Gate10B/C/D outputs."""

import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Sequence

import run_gate10b_trusted_tree_settlement_comparison as gate10b
import run_gate10c_second_settlement_comparison as gate10c
import run_gate10d_third_settlement_comparison as gate10d
import run_gate9a_graph_gauge_consumer as gate9a


REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "gate10e_interim_broader_judgment_v1"
METHOD_ID = "gate10e_interim_broader_judgment_v1"

DEFAULT_REGISTRY = "gate10_interim_broader_judgment_registry.jsonl"
DEFAULT_POLICY_COMPARE = "gate10_interim_broader_judgment_policy_compare.csv"
DEFAULT_STATUS = "gate10_interim_broader_judgment_status.json"
DEFAULT_REPORT = "gate10e_interim_broader_judgment_read.md"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Gate10E interim broader judgment from the first three "
            "declared Gate10 slice-local settled runs without promoting them into "
            "broader settlement or closeout."
        )
    )
    parser.add_argument("--gate10b-dir", required=True)
    parser.add_argument("--gate10c-dir", required=True)
    parser.add_argument("--gate10d-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_status_value(status_payload: Dict[str, Any], key: str) -> str:
    return str(status_payload.get(key, ""))


def build_registry(
    source_runs: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    registry_rows: List[Dict[str, Any]] = []
    for source_run in source_runs:
        status_payload = source_run["status_payload"]
        registry_rows.append(
            {
                "slice_id": str(source_run["slice_id"]),
                "source_dir": str(source_run["source_dir"]),
                "source_run_id": str(source_run["source_run_id"]),
                "source_code_git_commit": str(source_run["source_code_git_commit"]),
                "comparison_outcome_status": source_status_value(
                    status_payload, "comparison_outcome_status"
                ),
                "forward_basis_baseline_preservation_status": source_status_value(
                    status_payload, "forward_basis_baseline_preservation_status"
                ),
                "operator_admission_still_denied_status": source_status_value(
                    status_payload, "operator_admission_still_denied_status"
                ),
                "retroactive_guard_status": source_run["retroactive_guard_status"],
                "broader_non_promotion_status": source_status_value(
                    status_payload, "broader_tree_settlement_non_promotion_status"
                ),
                "slice_settled_status": source_run["slice_settled_status"],
            }
        )
    return registry_rows


def build_policy_compare(
    registry_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out_rows: List[Dict[str, Any]] = []
    for row in registry_rows:
        out_rows.append(
            {
                "slice_id": str(row["slice_id"]),
                "comparison_outcome_status": str(row["comparison_outcome_status"]),
                "slice_settled_status": str(row["slice_settled_status"]),
                "forward_basis_baseline_preservation_status": str(
                    row["forward_basis_baseline_preservation_status"]
                ),
                "operator_admission_still_denied_status": str(
                    row["operator_admission_still_denied_status"]
                ),
                "retroactive_guard_status": str(row["retroactive_guard_status"]),
                "broader_non_promotion_status": str(row["broader_non_promotion_status"]),
            }
        )
    return out_rows


def build_status_payload(
    source_runs: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    by_slice = {str(source_run["slice_id"]): source_run for source_run in source_runs}

    gate10b_slice_settled_status = str(by_slice["gate10b"]["slice_settled_status"])
    gate10c_slice_settled_status = str(by_slice["gate10c"]["slice_settled_status"])
    gate10d_slice_settled_status = str(by_slice["gate10d"]["slice_settled_status"])

    all_sources_present = all(
        bool(source_run["source_run_id"]) for source_run in source_runs
    )
    all_slices_preserved = all(
        str(source_run["slice_settled_status"]) == "preserved"
        for source_run in source_runs
    )
    all_operator_denied = all(
        source_status_value(
            source_run["status_payload"], "operator_admission_still_denied_status"
        )
        == "confirmed"
        for source_run in source_runs
    )
    all_retroactive_forbidden = all(
        str(source_run["retroactive_guard_status"]) == "confirmed"
        for source_run in source_runs
    )
    all_non_promotional = all(
        source_status_value(
            source_run["status_payload"], "broader_tree_settlement_non_promotion_status"
        )
        == "clear"
        for source_run in source_runs
    )
    all_baselines_preserved = all(
        source_status_value(
            source_run["status_payload"], "forward_basis_baseline_preservation_status"
        )
        == "clear"
        for source_run in source_runs
    )

    if not all_sources_present:
        three_slice_pattern_status = "deferred"
    elif all_slices_preserved and all_baselines_preserved and all_operator_denied and all_retroactive_forbidden and all_non_promotional:
        three_slice_pattern_status = "supported"
    else:
        three_slice_pattern_status = "not_supported"

    operator_admission_still_denied_status = (
        "confirmed" if all_operator_denied else "violated"
    )
    retroactive_reinterpretation_forbidden_status = (
        "confirmed" if all_retroactive_forbidden else "violated"
    )
    broader_trusted_tree_settlement_status = (
        "unearned" if all_non_promotional else "pressure_to_overclaim"
    )

    if three_slice_pattern_status == "supported" and broader_trusted_tree_settlement_status == "unearned":
        interim_broader_judgment_status = "bounded_support"
    elif three_slice_pattern_status == "deferred":
        interim_broader_judgment_status = "deferred"
    else:
        interim_broader_judgment_status = "not_yet_supported"

    pre_closeout_readiness_status = (
        "ready" if interim_broader_judgment_status == "bounded_support" else "not_ready"
    )

    if gate10b_slice_settled_status != "preserved":
        next_named_blocker = str(by_slice["gate10b"]["next_named_blocker"])
        if not next_named_blocker:
            next_named_blocker = "gate10b_slice_not_preserved"
    elif gate10c_slice_settled_status != "preserved":
        next_named_blocker = str(by_slice["gate10c"]["next_named_blocker"])
        if not next_named_blocker:
            next_named_blocker = "gate10c_slice_not_preserved"
    elif gate10d_slice_settled_status != "preserved":
        next_named_blocker = str(by_slice["gate10d"]["next_named_blocker"])
        if not next_named_blocker:
            next_named_blocker = "gate10d_slice_not_preserved"
    elif operator_admission_still_denied_status != "confirmed":
        next_named_blocker = "operator_admission_pressure"
    elif retroactive_reinterpretation_forbidden_status != "confirmed":
        next_named_blocker = "retroactive_reinterpretation_pressure"
    elif broader_trusted_tree_settlement_status != "unearned":
        next_named_blocker = "broader_trusted_tree_settlement_overclaim_pressure"
    elif interim_broader_judgment_status == "deferred":
        next_named_blocker = "controlling_source_incomplete"
    elif three_slice_pattern_status == "not_supported":
        next_named_blocker = "three_slice_pattern_not_supported"
    else:
        next_named_blocker = ""

    return {
        "gate10b_slice_settled_status": gate10b_slice_settled_status,
        "gate10c_slice_settled_status": gate10c_slice_settled_status,
        "gate10d_slice_settled_status": gate10d_slice_settled_status,
        "three_slice_pattern_status": three_slice_pattern_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "retroactive_reinterpretation_forbidden_status": retroactive_reinterpretation_forbidden_status,
        "broader_trusted_tree_settlement_status": broader_trusted_tree_settlement_status,
        "interim_broader_judgment_status": interim_broader_judgment_status,
        "pre_closeout_readiness_status": pre_closeout_readiness_status,
        "next_named_blocker": next_named_blocker,
    }


def build_report(
    run_id: str,
    source_runs: Sequence[Dict[str, Any]],
    policy_compare_rows: Sequence[Dict[str, Any]],
    status_payload: Dict[str, Any],
) -> str:
    lines = [
        "# Gate10E Interim Broader Judgment Read",
        "",
        f"run_id: {run_id}",
    ]
    for source_run in source_runs:
        lines.append(
            f"{source_run['slice_id']}_run_id: {source_run['source_run_id']}"
        )
    lines.extend(
        [
            "",
            "## Discipline",
            "",
            "- three declared slice-local results are aggregated only as bounded interim support",
            "- broader trusted-tree settlement remains unearned",
            "- operator admission remains denied",
            "- Gate10 closeout is not declared here",
            "",
            "## Slice Summary",
            "",
            "| slice_id | comparison_outcome_status | slice_settled_status | forward_basis_baseline_preservation_status | operator_admission_still_denied_status | retroactive_guard_status | broader_non_promotion_status |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for row in policy_compare_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["slice_id"]),
                    str(row["comparison_outcome_status"]),
                    str(row["slice_settled_status"]),
                    str(row["forward_basis_baseline_preservation_status"]),
                    str(row["operator_admission_still_denied_status"]),
                    str(row["retroactive_guard_status"]),
                    str(row["broader_non_promotion_status"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Status",
            "",
            f"- gate10b_slice_settled_status: `{status_payload['gate10b_slice_settled_status']}`",
            f"- gate10c_slice_settled_status: `{status_payload['gate10c_slice_settled_status']}`",
            f"- gate10d_slice_settled_status: `{status_payload['gate10d_slice_settled_status']}`",
            f"- three_slice_pattern_status: `{status_payload['three_slice_pattern_status']}`",
            f"- operator_admission_still_denied_status: `{status_payload['operator_admission_still_denied_status']}`",
            f"- retroactive_reinterpretation_forbidden_status: `{status_payload['retroactive_reinterpretation_forbidden_status']}`",
            f"- broader_trusted_tree_settlement_status: `{status_payload['broader_trusted_tree_settlement_status']}`",
            f"- interim_broader_judgment_status: `{status_payload['interim_broader_judgment_status']}`",
            f"- pre_closeout_readiness_status: `{status_payload['pre_closeout_readiness_status']}`",
            f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
        ]
    )

    if status_payload["interim_broader_judgment_status"] == "bounded_support":
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- the first three declared Gate10 slices are preserved as slice-local settled results",
                "- together they support a bounded broader trusted-tree pattern under the preserved Gate10 court",
                "- broader trusted-tree settlement and Gate10 closeout remain explicitly unearned here",
            ]
        )
    elif status_payload["interim_broader_judgment_status"] == "deferred":
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- the interim broader judgment remains deferred pending complete controlling source preservation",
                f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- the three-slice line does not yet support a bounded broader judgment under the preserved court",
                f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    source_gate10b_dir = Path(args.gate10b_dir)
    source_gate10c_dir = Path(args.gate10c_dir)
    source_gate10d_dir = Path(args.gate10d_dir)
    out_dir = Path(args.out_dir)
    run_id = out_dir.name

    source_runs = [
        {
            "slice_id": "gate10b",
            "source_dir": gate9a.repo_relative_or_posix(source_gate10b_dir),
            "source_run_id": str(
                gate9a.read_json(source_gate10b_dir / gate10b.DEFAULT_MANIFEST).get("run_id")
                or ""
            ),
            "source_code_git_commit": str(
                gate9a.read_json(source_gate10b_dir / gate10b.DEFAULT_MANIFEST).get(
                    "code_git_commit"
                )
                or ""
            ),
            "status_payload": gate9a.read_json(source_gate10b_dir / gate10b.DEFAULT_STATUS),
        },
        {
            "slice_id": "gate10c",
            "source_dir": gate9a.repo_relative_or_posix(source_gate10c_dir),
            "source_run_id": str(
                gate9a.read_json(source_gate10c_dir / gate10c.DEFAULT_MANIFEST).get("run_id")
                or ""
            ),
            "source_code_git_commit": str(
                gate9a.read_json(source_gate10c_dir / gate10c.DEFAULT_MANIFEST).get(
                    "code_git_commit"
                )
                or ""
            ),
            "status_payload": gate9a.read_json(source_gate10c_dir / gate10c.DEFAULT_STATUS),
        },
        {
            "slice_id": "gate10d",
            "source_dir": gate9a.repo_relative_or_posix(source_gate10d_dir),
            "source_run_id": str(
                gate9a.read_json(source_gate10d_dir / gate10d.DEFAULT_MANIFEST).get("run_id")
                or ""
            ),
            "source_code_git_commit": str(
                gate9a.read_json(source_gate10d_dir / gate10d.DEFAULT_MANIFEST).get(
                    "code_git_commit"
                )
                or ""
            ),
            "status_payload": gate9a.read_json(source_gate10d_dir / gate10d.DEFAULT_STATUS),
        },
    ]

    for source_run in source_runs:
        status_payload = source_run["status_payload"]
        outcome_status = source_status_value(status_payload, "comparison_outcome_status")
        source_run["slice_settled_status"] = (
            "preserved" if outcome_status == "settled" else "not_preserved"
        )
        if source_run["slice_id"] == "gate10b":
            retroactive_key = "non_retroactive_memory_preservation_status"
        else:
            retroactive_key = "non_retroactive_memory_preservation_status"
        source_run["retroactive_guard_status"] = (
            "confirmed"
            if source_status_value(status_payload, retroactive_key) == "clear"
            else "violated"
        )
        source_run["next_named_blocker"] = source_status_value(
            status_payload, "next_named_blocker"
        )

    registry_rows = build_registry(source_runs)
    policy_compare_rows = build_policy_compare(registry_rows)
    status_payload = build_status_payload(source_runs)

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
            "slice_id",
            "comparison_outcome_status",
            "slice_settled_status",
            "forward_basis_baseline_preservation_status",
            "operator_admission_still_denied_status",
            "retroactive_guard_status",
            "broader_non_promotion_status",
        ),
        policy_compare_rows,
    )
    gate9a.write_json(status_path, status_payload)
    gate9a.write_text(
        report_path,
        build_report(
            run_id=run_id,
            source_runs=source_runs,
            policy_compare_rows=policy_compare_rows,
            status_payload=status_payload,
        ),
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "run_id": run_id,
        "code_git_commit": gate9a.current_git_commit(),
        "source_gate10b_dir": gate9a.repo_relative_or_posix(source_gate10b_dir),
        "source_gate10b_run_id": source_runs[0]["source_run_id"],
        "source_gate10b_code_git_commit": source_runs[0]["source_code_git_commit"],
        "source_gate10c_dir": gate9a.repo_relative_or_posix(source_gate10c_dir),
        "source_gate10c_run_id": source_runs[1]["source_run_id"],
        "source_gate10c_code_git_commit": source_runs[1]["source_code_git_commit"],
        "source_gate10d_dir": gate9a.repo_relative_or_posix(source_gate10d_dir),
        "source_gate10d_run_id": source_runs[2]["source_run_id"],
        "source_gate10d_code_git_commit": source_runs[2]["source_code_git_commit"],
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