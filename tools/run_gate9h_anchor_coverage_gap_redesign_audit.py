#!/usr/bin/env python3
"""Run a Gate9H anchor-coverage-gap redesign audit on Gate9G outputs."""

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import run_gate9a_graph_gauge_consumer as gate9a
import run_gate9b_small_cycle_holonomy_study as gate9b
import run_gate9g_anchor_conditioned_triviality_audit as gate9g


REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "gate9h_anchor_coverage_gap_redesign_audit_v1"
METHOD_ID = "gate9h_anchor_coverage_gap_redesign_audit_v1"
NONTRIVIAL_GAP_TOLERANCE = 1e-6
DEFAULT_REGISTRY = "anchor_coverage_gap_redesign_registry.jsonl"
DEFAULT_SUMMARY = "anchor_coverage_gap_redesign_by_cell_anchor.csv"
DEFAULT_STATUS = "anchor_coverage_gap_redesign_status.json"
DEFAULT_REPORT = "gate9h_anchor_coverage_gap_redesign_read.md"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Gate9H redesign audit that replaces collapsed anchor-conditioned "
            "closure defect with coverage-gap under the frozen law."
        )
    )
    parser.add_argument("--gate9g-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_status_for_row(row: Dict[str, Any]) -> Tuple[str, str]:
    if str(row["closure_outcome"]) != "none":
        return "missing_or_insufficient", str(row["closure_outcome"])
    gap = row.get("coverage_gap_abs")
    if gap in (None, ""):
        return "missing_or_insufficient", "missing_coverage_gap"
    if float(gap) > NONTRIVIAL_GAP_TOLERANCE:
        return "nontrivial_gap_candidate", "coverage_gap_exceeds_tolerance"
    return "collapsed_gap_candidate", "coverage_gap_below_tolerance"


def build_registry_rows(gate9g_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    registry_rows: List[Dict[str, Any]] = []
    for row in gate9g_rows:
        candidate_status, candidate_reason = candidate_status_for_row(row)
        registry_rows.append(
            {
                "closure_id": str(row["closure_id"]),
                "anchor_kind": str(row["anchor_kind"]),
                "execution_sample_id": int(row["execution_sample_id"]),
                "benchmark_sample_id": str(row["benchmark_sample_id"]),
                "cell_id": str(row["cell_id"]),
                "world_id": str(row["world_id"]),
                "world_type": str(row["world_type"]),
                "answer_target_type": str(row["answer_target_type"]),
                "rendering_family_id": str(row["rendering_family_id"]),
                "closure_outcome": str(row["closure_outcome"]),
                "legacy_triviality_status": str(row["triviality_status"]),
                "anchor_rank": row.get("anchor_rank"),
                "anchor_answer_coverage": row.get("anchor_answer_coverage"),
                "anchor_token_coverage": row.get("anchor_token_coverage"),
                "coverage_gap_abs": row.get("coverage_gap_abs"),
                "candidate_metric_id": "anchor_coverage_gap_abs_v1",
                "candidate_status": candidate_status,
                "candidate_reason": candidate_reason,
            }
        )
    return registry_rows


def summarize_registry_rows(registry_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in registry_rows:
        grouped[(str(row["cell_id"]), str(row["anchor_kind"]), str(row["candidate_status"]))].append(row)

    out_rows: List[Dict[str, Any]] = []
    for cell_id, anchor_kind, candidate_status in sorted(grouped):
        rows = grouped[(cell_id, anchor_kind, candidate_status)]
        gaps = [
            float(row["coverage_gap_abs"])
            for row in rows
            if row["coverage_gap_abs"] not in (None, "")
        ]
        out_rows.append(
            {
                "cell_id": cell_id,
                "anchor_kind": anchor_kind,
                "candidate_status": candidate_status,
                "n_rows": len(rows),
                "mean_coverage_gap_abs": None if not gaps else float(sum(gaps) / len(gaps)),
                "max_coverage_gap_abs": None if not gaps else float(max(gaps)),
            }
        )
    return out_rows


def mean_gap_for_cell_anchor(
    registry_rows: Sequence[Dict[str, Any]],
    *,
    cell_id: str,
    anchor_kind: str,
) -> float | None:
    values = [
        float(row["coverage_gap_abs"])
        for row in registry_rows
        if str(row["cell_id"]) == cell_id
        and str(row["anchor_kind"]) == anchor_kind
        and str(row["candidate_status"]) == "nontrivial_gap_candidate"
        and row["coverage_gap_abs"] not in (None, "")
    ]
    if not values:
        return None
    return float(sum(values) / len(values))


def build_status_payload(registry_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    candidate_rows = [
        row for row in registry_rows if str(row["candidate_status"]) == "nontrivial_gap_candidate"
    ]
    collapsed_rows = [
        row for row in registry_rows if str(row["candidate_status"]) == "collapsed_gap_candidate"
    ]
    support_cleaner_means = [
        mean_gap_for_cell_anchor(registry_rows, cell_id=cell_id, anchor_kind="support")
        for cell_id in gate9b.CLEANER_CELL_IDS
    ]
    support_conflict_means = [
        mean_gap_for_cell_anchor(registry_rows, cell_id=cell_id, anchor_kind="support")
        for cell_id in gate9b.CONFLICT_CELL_IDS
    ]
    support_cleaner_values = [value for value in support_cleaner_means if value is not None]
    support_conflict_values = [value for value in support_conflict_means if value is not None]
    support_anchor_cleaner_dominance_status = "insufficient_data"
    if support_cleaner_values and support_conflict_values:
        support_anchor_cleaner_dominance_status = (
            "triggered"
            if max(support_cleaner_values) >= max(support_conflict_values)
            else "clear"
        )
    conflict_direct = mean_gap_for_cell_anchor(
        registry_rows,
        cell_id="direct_contradiction",
        anchor_kind="conflict",
    )
    conflict_distributed = mean_gap_for_cell_anchor(
        registry_rows,
        cell_id="distributed_incompatibility",
        anchor_kind="conflict",
    )
    conflict_anchor_availability_status = (
        "clear" if conflict_direct is not None and conflict_distributed is not None else "denied"
    )
    redesign_candidate_nontriviality_status = "provisionally_clear" if candidate_rows else "denied"
    redesign_admission_readiness_status = (
        "denied" if support_anchor_cleaner_dominance_status == "triggered" or not candidate_rows else "provisionally_clear"
    )
    return {
        "candidate_metric_id": "anchor_coverage_gap_abs_v1",
        "redesign_candidate_nontriviality_status": redesign_candidate_nontriviality_status,
        "support_anchor_cleaner_dominance_status": support_anchor_cleaner_dominance_status,
        "conflict_anchor_availability_status": conflict_anchor_availability_status,
        "redesign_admission_readiness_status": redesign_admission_readiness_status,
        "next_named_blocker": (
            "cleaner_cell_dominance"
            if support_anchor_cleaner_dominance_status == "triggered"
            else ""
        ),
        "n_nontrivial_gap_candidate_rows": len(candidate_rows),
        "n_collapsed_gap_candidate_rows": len(collapsed_rows),
        "support_cleaner_max_mean_gap": None if not support_cleaner_values else float(max(support_cleaner_values)),
        "support_conflict_max_mean_gap": None if not support_conflict_values else float(max(support_conflict_values)),
        "conflict_direct_mean_gap": conflict_direct,
        "conflict_distributed_mean_gap": conflict_distributed,
    }


def build_report(
    run_id: str,
    source_manifest: Dict[str, Any],
    summary_rows: Sequence[Dict[str, Any]],
    status_payload: Dict[str, Any],
) -> str:
    lines = [
        "# Gate9H Anchor-Coverage-Gap Redesign Read",
        "",
        f"run_id: {run_id}",
        f"source_gate9g_run_id: {source_manifest.get('run_id', '')}",
        f"source_gate9g_code_git_commit: {source_manifest.get('code_git_commit', '')}",
        f"source_gate9a_run_id: {source_manifest.get('source_gate9a_run_id', '')}",
        "",
        "## Discipline",
        "",
        "- the redesign stays on the same anchor-conditioned bundle",
        "- the redesign replaces collapsed closure defect with absolute coverage gap only",
        "- no new anchor semantics or closure convention are introduced",
        "",
        "## Summary By Cell And Anchor",
        "",
        "| cell_id | anchor_kind | candidate_status | n_rows | mean_coverage_gap_abs | max_coverage_gap_abs |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["cell_id"]),
                    str(row["anchor_kind"]),
                    str(row["candidate_status"]),
                    str(row["n_rows"]),
                    "" if row["mean_coverage_gap_abs"] in (None, "") else f"{float(row['mean_coverage_gap_abs']):.6f}",
                    "" if row["max_coverage_gap_abs"] in (None, "") else f"{float(row['max_coverage_gap_abs']):.6f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Status",
            "",
            f"- redesign_candidate_nontriviality_status: `{status_payload['redesign_candidate_nontriviality_status']}`",
            f"- support_anchor_cleaner_dominance_status: `{status_payload['support_anchor_cleaner_dominance_status']}`",
            f"- conflict_anchor_availability_status: `{status_payload['conflict_anchor_availability_status']}`",
            f"- redesign_admission_readiness_status: `{status_payload['redesign_admission_readiness_status']}`",
        ]
    )
    if status_payload.get("next_named_blocker"):
        lines.extend(
            [
                "",
                "## Next Blocker",
                "",
                f"- `{status_payload['next_named_blocker']}`",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    source_gate9g_dir = Path(args.gate9g_dir)
    out_dir = Path(args.out_dir)
    run_id = out_dir.name

    source_manifest = gate9a.read_json(source_gate9g_dir / gate9a.DEFAULT_MANIFEST)
    gate9g_rows = gate9a.read_jsonl(source_gate9g_dir / gate9g.DEFAULT_REGISTRY)
    registry_rows = build_registry_rows(gate9g_rows)
    summary_rows = summarize_registry_rows(registry_rows)
    status_payload = build_status_payload(registry_rows)

    registry_path = out_dir / DEFAULT_REGISTRY
    summary_path = out_dir / DEFAULT_SUMMARY
    status_path = out_dir / DEFAULT_STATUS
    report_path = out_dir / DEFAULT_REPORT
    manifest_path = out_dir / DEFAULT_MANIFEST
    checksums_path = out_dir / DEFAULT_CHECKSUMS

    gate9a.write_jsonl(registry_path, registry_rows)
    gate9a.write_csv(
        summary_path,
        (
            "cell_id",
            "anchor_kind",
            "candidate_status",
            "n_rows",
            "mean_coverage_gap_abs",
            "max_coverage_gap_abs",
        ),
        summary_rows,
    )
    gate9a.write_json(status_path, status_payload)
    gate9a.write_text(
        report_path,
        build_report(
            run_id=run_id,
            source_manifest=source_manifest,
            summary_rows=summary_rows,
            status_payload=status_payload,
        ),
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "run_id": run_id,
        "code_git_commit": gate9a.current_git_commit(),
        "source_gate9g_dir": gate9a.repo_relative_or_posix(source_gate9g_dir),
        "source_gate9g_run_id": str(source_manifest.get("run_id") or ""),
        "source_gate9g_code_git_commit": str(source_manifest.get("code_git_commit") or ""),
        "source_gate9a_run_id": str(source_manifest.get("source_gate9a_run_id") or ""),
        "paths": {
            DEFAULT_REGISTRY: gate9a.repo_relative_or_posix(registry_path),
            DEFAULT_SUMMARY: gate9a.repo_relative_or_posix(summary_path),
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
            DEFAULT_SUMMARY: sha256_file(summary_path),
            DEFAULT_STATUS: sha256_file(status_path),
            DEFAULT_REPORT: sha256_file(report_path),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
