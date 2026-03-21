#!/usr/bin/env python3
"""Run a Gate9J distributed-underactivation audit on Gate9I outputs."""

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import run_gate9a_graph_gauge_consumer as gate9a
import run_gate9h_anchor_coverage_gap_redesign_audit as gate9h
import run_gate9i_support_anchor_cleaner_cell_dominance_audit as gate9i


REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "gate9j_distributed_underactivation_audit_v1"
METHOD_ID = "gate9j_distributed_underactivation_audit_v1"
DEFAULT_REGISTRY = "distributed_underactivation_registry.jsonl"
DEFAULT_CELL_SUMMARY = "distributed_underactivation_by_cell.csv"
DEFAULT_BRANCH_SUMMARY = "distributed_underactivation_by_branch.csv"
DEFAULT_STATUS = "distributed_underactivation_status.json"
DEFAULT_REPORT = "gate9j_distributed_underactivation_read.md"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Gate9J audit over support-anchor distributed underactivation "
            "on the Gate9H redesign line."
        )
    )
    parser.add_argument("--gate9i-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive_source_context(source_gate9i_dir: Path) -> Tuple[Dict[str, Any], Path, Dict[str, Any]]:
    source_gate9i_manifest = gate9a.read_json(source_gate9i_dir / gate9a.DEFAULT_MANIFEST)
    source_gate9h_dir = REPO_ROOT / str(source_gate9i_manifest["source_gate9h_dir"])
    source_gate9h_manifest = gate9a.read_json(source_gate9h_dir / gate9a.DEFAULT_MANIFEST)
    return source_gate9i_manifest, source_gate9h_dir, source_gate9h_manifest


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def branch_kind(answer_target_type: str) -> str:
    return "consistent_answer_branch" if answer_target_type == "consistent_answer" else "nonconsistent_answer_branch"


def build_registry_rows(gate9h_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    registry_rows: List[Dict[str, Any]] = []
    for row in gate9h_rows:
        if str(row["anchor_kind"]) != "support":
            continue
        if str(row["cell_id"]) not in {"direct_contradiction", "distributed_incompatibility"}:
            continue
        registry_rows.append(
            {
                "closure_id": str(row["closure_id"]),
                "benchmark_sample_id": str(row["benchmark_sample_id"]),
                "execution_sample_id": int(row["execution_sample_id"]),
                "cell_id": str(row["cell_id"]),
                "world_id": str(row["world_id"]),
                "world_type": str(row["world_type"]),
                "answer_target_type": str(row["answer_target_type"]),
                "branch_kind": branch_kind(str(row["answer_target_type"])),
                "coverage_gap_abs": row.get("coverage_gap_abs"),
                "anchor_answer_coverage": row.get("anchor_answer_coverage"),
                "anchor_token_coverage": row.get("anchor_token_coverage"),
                "candidate_status": str(row["candidate_status"]),
            }
        )
    return registry_rows


def summarize_by_key(
    registry_rows: Sequence[Dict[str, Any]],
    key_name: str,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in registry_rows:
        grouped[str(row[key_name])].append(row)

    out_rows: List[Dict[str, Any]] = []
    for key in sorted(grouped):
        rows = grouped[key]
        gap_values = [
            float(row["coverage_gap_abs"])
            for row in rows
            if row["candidate_status"] == "nontrivial_gap_candidate" and row["coverage_gap_abs"] not in (None, "")
        ]
        answer_values = [
            float(row["anchor_answer_coverage"])
            for row in rows
            if row["candidate_status"] == "nontrivial_gap_candidate" and row["anchor_answer_coverage"] not in (None, "")
        ]
        token_values = [
            float(row["anchor_token_coverage"])
            for row in rows
            if row["candidate_status"] == "nontrivial_gap_candidate" and row["anchor_token_coverage"] not in (None, "")
        ]
        out_rows.append(
            {
                key_name: key,
                "n_rows": len(rows),
                "mean_coverage_gap_abs": mean_or_none(gap_values),
                "mean_anchor_answer_coverage": mean_or_none(answer_values),
                "mean_anchor_token_coverage": mean_or_none(token_values),
            }
        )
    return out_rows


def first_row(
    registry_rows: Sequence[Dict[str, Any]],
    *,
    cell_id: str,
    answer_target_type: Optional[str] = None,
    branch_kind_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    for row in registry_rows:
        if str(row["cell_id"]) != cell_id:
            continue
        if answer_target_type is not None and str(row["answer_target_type"]) != answer_target_type:
            continue
        if branch_kind_name is not None and str(row["branch_kind"]) != branch_kind_name:
            continue
        return row
    return None


def build_status_payload(registry_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    direct_values = [
        float(row["coverage_gap_abs"])
        for row in registry_rows
        if row["cell_id"] == "direct_contradiction" and row["coverage_gap_abs"] not in (None, "")
    ]
    distributed_values = [
        float(row["coverage_gap_abs"])
        for row in registry_rows
        if row["cell_id"] == "distributed_incompatibility" and row["coverage_gap_abs"] not in (None, "")
    ]
    direct_mean = mean_or_none(direct_values)
    distributed_mean = mean_or_none(distributed_values)

    direct_consistent = first_row(
        registry_rows,
        cell_id="direct_contradiction",
        answer_target_type="consistent_answer",
    )
    distributed_consistent = first_row(
        registry_rows,
        cell_id="distributed_incompatibility",
        answer_target_type="consistent_answer",
    )
    distributed_nonconsistent = first_row(
        registry_rows,
        cell_id="distributed_incompatibility",
        branch_kind_name="nonconsistent_answer_branch",
    )

    distributed_underactivation_status = "insufficient_data"
    if direct_mean is not None and distributed_mean is not None:
        distributed_underactivation_status = "triggered" if distributed_mean < direct_mean else "clear"

    distributed_answer_target_split_status = "insufficient_data"
    if distributed_consistent and distributed_nonconsistent:
        distributed_answer_target_split_status = (
            "triggered"
            if float(distributed_consistent["coverage_gap_abs"]) < float(distributed_nonconsistent["coverage_gap_abs"])
            else "clear"
        )

    distributed_consistent_branch_status = "insufficient_data"
    if direct_consistent and distributed_consistent and distributed_nonconsistent:
        distributed_consistent_branch_status = (
            "underactivated"
            if float(distributed_consistent["coverage_gap_abs"]) < min(
                float(direct_consistent["coverage_gap_abs"]),
                float(distributed_nonconsistent["coverage_gap_abs"]),
            )
            else "clear"
        )

    direct_baseline_answer_suppression_status = "insufficient_data"
    direct_to_distributed_consistent_answer_delta = None
    direct_to_distributed_consistent_token_delta = None
    direct_to_distributed_consistent_gap_delta = None
    if direct_consistent and distributed_consistent:
        direct_to_distributed_consistent_answer_delta = (
            float(distributed_consistent["anchor_answer_coverage"]) - float(direct_consistent["anchor_answer_coverage"])
        )
        direct_to_distributed_consistent_token_delta = (
            float(distributed_consistent["anchor_token_coverage"]) - float(direct_consistent["anchor_token_coverage"])
        )
        direct_to_distributed_consistent_gap_delta = (
            float(distributed_consistent["coverage_gap_abs"]) - float(direct_consistent["coverage_gap_abs"])
        )
        direct_baseline_answer_suppression_status = (
            "triggered"
            if float(distributed_consistent["anchor_answer_coverage"]) < float(direct_consistent["anchor_answer_coverage"])
            else "clear"
        )

    gap_loss_explained_as_token_only_status = "insufficient_data"
    if (
        direct_to_distributed_consistent_answer_delta is not None
        and direct_to_distributed_consistent_token_delta is not None
    ):
        gap_loss_explained_as_token_only_status = (
            "denied"
            if abs(direct_to_distributed_consistent_answer_delta)
            > abs(direct_to_distributed_consistent_token_delta)
            else "not_yet_denied"
        )

    return {
        "distributed_underactivation_status": distributed_underactivation_status,
        "distributed_answer_target_split_status": distributed_answer_target_split_status,
        "distributed_consistent_branch_status": distributed_consistent_branch_status,
        "direct_baseline_answer_suppression_status": direct_baseline_answer_suppression_status,
        "gap_loss_explained_as_token_only_status": gap_loss_explained_as_token_only_status,
        "direct_mean_gap": direct_mean,
        "distributed_mean_gap": distributed_mean,
        "direct_consistent_gap": None if direct_consistent is None else float(direct_consistent["coverage_gap_abs"]),
        "distributed_consistent_gap": (
            None if distributed_consistent is None else float(distributed_consistent["coverage_gap_abs"])
        ),
        "distributed_nonconsistent_gap": (
            None if distributed_nonconsistent is None else float(distributed_nonconsistent["coverage_gap_abs"])
        ),
        "direct_to_distributed_consistent_gap_delta": direct_to_distributed_consistent_gap_delta,
        "direct_to_distributed_consistent_answer_delta": direct_to_distributed_consistent_answer_delta,
        "direct_to_distributed_consistent_token_delta": direct_to_distributed_consistent_token_delta,
        "next_named_subblocker": (
            "distributed_consistent_answer_compression"
            if distributed_consistent_branch_status == "underactivated"
            else ""
        ),
    }


def build_report(
    run_id: str,
    source_manifest: Dict[str, Any],
    cell_summary_rows: Sequence[Dict[str, Any]],
    branch_summary_rows: Sequence[Dict[str, Any]],
    status_payload: Dict[str, Any],
) -> str:
    lines = [
        "# Gate9J Distributed Underactivation Read",
        "",
        f"run_id: {run_id}",
        f"source_gate9i_run_id: {source_manifest.get('run_id', '')}",
        f"source_gate9i_code_git_commit: {source_manifest.get('code_git_commit', '')}",
        "",
        "## Discipline",
        "",
        "- support-anchor conflict rows only",
        "- no new metric family beyond `anchor_coverage_gap_abs_v1`",
        "- the question is where distributed underactivation lives, not how to rescue it yet",
        "",
        "## Means By Cell",
        "",
        "| cell_id | n_rows | mean_coverage_gap_abs | mean_anchor_answer_coverage | mean_anchor_token_coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in cell_summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["cell_id"]),
                    str(row["n_rows"]),
                    "" if row["mean_coverage_gap_abs"] in (None, "") else f"{float(row['mean_coverage_gap_abs']):.6f}",
                    "" if row["mean_anchor_answer_coverage"] in (None, "") else f"{float(row['mean_anchor_answer_coverage']):.6f}",
                    "" if row["mean_anchor_token_coverage"] in (None, "") else f"{float(row['mean_anchor_token_coverage']):.6f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Means By Branch",
            "",
            "| branch_kind | n_rows | mean_coverage_gap_abs | mean_anchor_answer_coverage | mean_anchor_token_coverage |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in branch_summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["branch_kind"]),
                    str(row["n_rows"]),
                    "" if row["mean_coverage_gap_abs"] in (None, "") else f"{float(row['mean_coverage_gap_abs']):.6f}",
                    "" if row["mean_anchor_answer_coverage"] in (None, "") else f"{float(row['mean_anchor_answer_coverage']):.6f}",
                    "" if row["mean_anchor_token_coverage"] in (None, "") else f"{float(row['mean_anchor_token_coverage']):.6f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Status",
            "",
            f"- distributed_underactivation_status: `{status_payload['distributed_underactivation_status']}`",
            f"- distributed_answer_target_split_status: `{status_payload['distributed_answer_target_split_status']}`",
            f"- distributed_consistent_branch_status: `{status_payload['distributed_consistent_branch_status']}`",
            f"- direct_baseline_answer_suppression_status: `{status_payload['direct_baseline_answer_suppression_status']}`",
            f"- gap_loss_explained_as_token_only_status: `{status_payload['gap_loss_explained_as_token_only_status']}`",
        ]
    )
    if status_payload.get("next_named_subblocker"):
        lines.extend(
            [
                "",
                "## Next Subblocker",
                "",
                f"- `{status_payload['next_named_subblocker']}`",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    source_gate9i_dir = Path(args.gate9i_dir)
    out_dir = Path(args.out_dir)
    run_id = out_dir.name

    source_gate9i_manifest, source_gate9h_dir, source_gate9h_manifest = derive_source_context(source_gate9i_dir)
    gate9h_rows = gate9a.read_jsonl(source_gate9h_dir / gate9h.DEFAULT_REGISTRY)

    registry_rows = build_registry_rows(gate9h_rows)
    cell_summary_rows = summarize_by_key(registry_rows, "cell_id")
    branch_summary_rows = summarize_by_key(registry_rows, "branch_kind")
    status_payload = build_status_payload(registry_rows)

    registry_path = out_dir / DEFAULT_REGISTRY
    cell_summary_path = out_dir / DEFAULT_CELL_SUMMARY
    branch_summary_path = out_dir / DEFAULT_BRANCH_SUMMARY
    status_path = out_dir / DEFAULT_STATUS
    report_path = out_dir / DEFAULT_REPORT
    manifest_path = out_dir / DEFAULT_MANIFEST
    checksums_path = out_dir / DEFAULT_CHECKSUMS

    gate9a.write_jsonl(registry_path, registry_rows)
    gate9a.write_csv(
        cell_summary_path,
        ("cell_id", "n_rows", "mean_coverage_gap_abs", "mean_anchor_answer_coverage", "mean_anchor_token_coverage"),
        cell_summary_rows,
    )
    gate9a.write_csv(
        branch_summary_path,
        ("branch_kind", "n_rows", "mean_coverage_gap_abs", "mean_anchor_answer_coverage", "mean_anchor_token_coverage"),
        branch_summary_rows,
    )
    gate9a.write_json(status_path, status_payload)
    gate9a.write_text(
        report_path,
        build_report(
            run_id=run_id,
            source_manifest=source_gate9i_manifest,
            cell_summary_rows=cell_summary_rows,
            branch_summary_rows=branch_summary_rows,
            status_payload=status_payload,
        ),
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "run_id": run_id,
        "code_git_commit": gate9a.current_git_commit(),
        "source_gate9i_dir": gate9a.repo_relative_or_posix(source_gate9i_dir),
        "source_gate9i_run_id": str(source_gate9i_manifest.get("run_id") or ""),
        "source_gate9i_code_git_commit": str(source_gate9i_manifest.get("code_git_commit") or ""),
        "source_gate9h_dir": gate9a.repo_relative_or_posix(source_gate9h_dir),
        "source_gate9h_run_id": str(source_gate9h_manifest.get("run_id") or ""),
        "source_gate9h_code_git_commit": str(source_gate9h_manifest.get("code_git_commit") or ""),
        "paths": {
            DEFAULT_REGISTRY: gate9a.repo_relative_or_posix(registry_path),
            DEFAULT_CELL_SUMMARY: gate9a.repo_relative_or_posix(cell_summary_path),
            DEFAULT_BRANCH_SUMMARY: gate9a.repo_relative_or_posix(branch_summary_path),
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
            DEFAULT_CELL_SUMMARY: sha256_file(cell_summary_path),
            DEFAULT_BRANCH_SUMMARY: sha256_file(branch_summary_path),
            DEFAULT_STATUS: sha256_file(status_path),
            DEFAULT_REPORT: sha256_file(report_path),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
