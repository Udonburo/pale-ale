#!/usr/bin/env python3
"""Run a Gate9G anchor-conditioned triviality audit on Gate9A outputs."""

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import run_gate9a_graph_gauge_consumer as gate9a


REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "gate9g_anchor_conditioned_triviality_audit_v1"
METHOD_ID = "gate9g_anchor_conditioned_triviality_audit_v1"
TRIVIALITY_TOLERANCE = 1e-12
MISSING_OR_INSUFFICIENT = "missing_or_insufficient"
FULL_ANCHOR_SPAN_COLLAPSE = "full_anchor_span_collapse"
NEAR_ZERO_OTHER = "near_zero_other"
NONTRIVIAL_SIGNAL_CANDIDATE = "nontrivial_signal_candidate"

DEFAULT_REGISTRY = "anchor_conditioned_triviality_registry.jsonl"
DEFAULT_SUMMARY = "anchor_conditioned_triviality_by_cell_anchor.csv"
DEFAULT_STATUS = "anchor_conditioned_triviality_status.json"
DEFAULT_REPORT = "gate9g_anchor_conditioned_triviality_read.md"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Gate9G anchor-conditioned triviality audit over an existing "
            "Gate9A bundle without reopening the graph-gauge law."
        )
    )
    parser.add_argument("--gate9a-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify_triviality(
    *,
    closure_outcome: str,
    closure_defect: Any,
    anchor_rank: Any,
    answer_conditioned_rank: Any,
    token_conditioned_rank: Any,
) -> Tuple[str, str]:
    if closure_outcome != "none":
        return MISSING_OR_INSUFFICIENT, str(closure_outcome)
    if closure_defect in (None, ""):
        return MISSING_OR_INSUFFICIENT, "missing_closure_defect"
    defect_value = abs(float(closure_defect))
    if defect_value <= TRIVIALITY_TOLERANCE:
        if (
            anchor_rank not in (None, "")
            and answer_conditioned_rank not in (None, "")
            and token_conditioned_rank not in (None, "")
            and int(anchor_rank) > 0
            and int(answer_conditioned_rank) == int(anchor_rank)
            and int(token_conditioned_rank) == int(anchor_rank)
        ):
            return FULL_ANCHOR_SPAN_COLLAPSE, "conditioned_ranks_saturate_anchor_rank"
        return NEAR_ZERO_OTHER, "near_zero_defect_without_full_anchor_saturation"
    return NONTRIVIAL_SIGNAL_CANDIDATE, "closure_defect_exceeds_triviality_tolerance"


def build_registry_rows(
    anchor_rows: Sequence[Dict[str, Any]],
    node_rows_by_id: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    registry_rows: List[Dict[str, Any]] = []
    for row in anchor_rows:
        anchor_node_id = str(row.get("anchor_node_id") or "")
        anchor_node = node_rows_by_id.get(anchor_node_id, {})
        anchor_rank = anchor_node.get("rank_local")
        answer_coverage = row.get("anchor_answer_coverage")
        token_coverage = row.get("anchor_token_coverage")
        closure_defect = row.get("anchor_conditioned_closure_defect")
        triviality_status, triviality_reason = classify_triviality(
            closure_outcome=str(row["closure_outcome"]),
            closure_defect=closure_defect,
            anchor_rank=anchor_rank,
            answer_conditioned_rank=row.get("answer_conditioned_rank"),
            token_conditioned_rank=row.get("token_conditioned_rank"),
        )
        coverage_gap_abs = None
        if answer_coverage not in (None, "") and token_coverage not in (None, ""):
            coverage_gap_abs = abs(float(answer_coverage) - float(token_coverage))
        registry_rows.append(
            {
                "closure_id": str(row["closure_id"]),
                "anchor_kind": str(row["anchor_kind"]),
                "anchor_node_id": anchor_node_id,
                "execution_sample_id": int(row["execution_sample_id"]),
                "benchmark_sample_id": str(row["benchmark_sample_id"]),
                "cell_id": str(row["cell_id"]),
                "world_id": str(row["world_id"]),
                "world_type": str(row["world_type"]),
                "answer_target_type": str(row["answer_target_type"]),
                "rendering_family_id": str(row["rendering_family_id"]),
                "closure_outcome": str(row["closure_outcome"]),
                "anchor_rank": anchor_rank,
                "answer_conditioned_rank": row.get("answer_conditioned_rank"),
                "token_conditioned_rank": row.get("token_conditioned_rank"),
                "anchor_answer_coverage": answer_coverage,
                "anchor_token_coverage": token_coverage,
                "coverage_gap_abs": coverage_gap_abs,
                "anchor_conditioned_closure_defect": closure_defect,
                "triviality_status": triviality_status,
                "triviality_reason": triviality_reason,
            }
        )
    return registry_rows


def summarize_registry_rows(registry_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in registry_rows:
        grouped[(str(row["cell_id"]), str(row["anchor_kind"]), str(row["triviality_status"]))].append(row)

    out_rows: List[Dict[str, Any]] = []
    for cell_id, anchor_kind, triviality_status in sorted(grouped):
        rows = grouped[(cell_id, anchor_kind, triviality_status)]
        defects = [
            float(row["anchor_conditioned_closure_defect"])
            for row in rows
            if row["anchor_conditioned_closure_defect"] not in (None, "")
        ]
        coverage_gaps = [
            float(row["coverage_gap_abs"])
            for row in rows
            if row["coverage_gap_abs"] not in (None, "")
        ]
        out_rows.append(
            {
                "cell_id": cell_id,
                "anchor_kind": anchor_kind,
                "triviality_status": triviality_status,
                "n_rows": len(rows),
                "mean_closure_defect": None if not defects else float(sum(defects) / len(defects)),
                "max_closure_defect": None if not defects else float(max(defects)),
                "mean_coverage_gap_abs": None if not coverage_gaps else float(sum(coverage_gaps) / len(coverage_gaps)),
            }
        )
    return out_rows


def build_status_payload(registry_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    non_missing_rows = [
        row for row in registry_rows if str(row["closure_outcome"]) == "none"
    ]
    collapse_rows = [
        row for row in non_missing_rows if str(row["triviality_status"]) == FULL_ANCHOR_SPAN_COLLAPSE
    ]
    candidate_rows = [
        row for row in non_missing_rows if str(row["triviality_status"]) == NONTRIVIAL_SIGNAL_CANDIDATE
    ]
    near_zero_other_rows = [
        row for row in non_missing_rows if str(row["triviality_status"]) == NEAR_ZERO_OTHER
    ]
    status_counter = Counter(str(row["triviality_status"]) for row in registry_rows)
    full_collapse_status = "clear"
    if non_missing_rows and len(collapse_rows) == len(non_missing_rows):
        full_collapse_status = "triggered"
    elif collapse_rows:
        full_collapse_status = "partial"
    return {
        "non_trivial_anchor_conditioned_read_status": "denied" if not candidate_rows else "candidate_present",
        "full_anchor_span_collapse_status": full_collapse_status,
        "near_zero_other_status": "triggered" if near_zero_other_rows else "clear",
        "operator_admission_blocker_status": "triggered" if not candidate_rows else "clear",
        "n_non_missing_rows": len(non_missing_rows),
        "n_full_anchor_span_collapse_rows": len(collapse_rows),
        "n_nontrivial_signal_candidate_rows": len(candidate_rows),
        "triviality_status_counts": dict(status_counter),
    }


def build_report(
    run_id: str,
    source_manifest: Dict[str, Any],
    summary_rows: Sequence[Dict[str, Any]],
    status_payload: Dict[str, Any],
) -> str:
    lines = [
        "# Gate9G Anchor-Conditioned Triviality Read",
        "",
        f"run_id: {run_id}",
        f"source_gate9a_run_id: {source_manifest.get('run_id', '')}",
        f"source_gate9a_code_git_commit: {source_manifest.get('code_git_commit', '')}",
        f"source_gate8_run_id: {source_manifest.get('source_gate8_run_id', '')}",
        "",
        "## Discipline",
        "",
        "- the question is not anchor redesign yet",
        "- the question is whether the current anchor-conditioned read is non-trivial at all",
        "- missingness stays explicit and is not folded into triviality",
        "",
        "## Summary By Cell And Anchor",
        "",
        "| cell_id | anchor_kind | triviality_status | n_rows | mean_closure_defect | max_closure_defect | mean_coverage_gap_abs |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["cell_id"]),
                    str(row["anchor_kind"]),
                    str(row["triviality_status"]),
                    str(row["n_rows"]),
                    "" if row["mean_closure_defect"] in (None, "") else f"{float(row['mean_closure_defect']):.6f}",
                    "" if row["max_closure_defect"] in (None, "") else f"{float(row['max_closure_defect']):.6f}",
                    "" if row["mean_coverage_gap_abs"] in (None, "") else f"{float(row['mean_coverage_gap_abs']):.6f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Status",
            "",
            f"- non_trivial_anchor_conditioned_read_status: `{status_payload['non_trivial_anchor_conditioned_read_status']}`",
            f"- full_anchor_span_collapse_status: `{status_payload['full_anchor_span_collapse_status']}`",
            f"- near_zero_other_status: `{status_payload['near_zero_other_status']}`",
            f"- operator_admission_blocker_status: `{status_payload['operator_admission_blocker_status']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    source_gate9a_dir = Path(args.gate9a_dir)
    out_dir = Path(args.out_dir)
    run_id = out_dir.name

    source_manifest = gate9a.read_json(source_gate9a_dir / gate9a.DEFAULT_MANIFEST)
    anchor_rows = gate9a.read_jsonl(source_gate9a_dir / gate9a.DEFAULT_ANCHOR_CLOSURE)
    node_rows = gate9a.read_jsonl(source_gate9a_dir / gate9a.DEFAULT_NODE_REGISTRY)
    node_rows_by_id = {str(row["node_id"]): row for row in node_rows}

    registry_rows = build_registry_rows(anchor_rows, node_rows_by_id)
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
            "triviality_status",
            "n_rows",
            "mean_closure_defect",
            "max_closure_defect",
            "mean_coverage_gap_abs",
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
        "source_gate9a_dir": gate9a.repo_relative_or_posix(source_gate9a_dir),
        "source_gate9a_run_id": str(source_manifest.get("run_id") or ""),
        "source_gate9a_code_git_commit": str(source_manifest.get("code_git_commit") or ""),
        "source_gate8_run_id": str(source_manifest.get("source_gate8_run_id") or ""),
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
