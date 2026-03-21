#!/usr/bin/env python3
"""Run a Gate9K trusted-tree / residual-chord logging audit on Gate9J outputs."""

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import run_gate9a_graph_gauge_consumer as gate9a
import run_gate9j_distributed_underactivation_audit as gate9j


REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "gate9k_trusted_tree_residual_chord_logging_v1"
METHOD_ID = "gate9k_trusted_tree_residual_chord_logging_v1"
TRUSTED_EDGE_POLICY_ID = "gate9k_trusted_edges_temporal_support_v1"
RESIDUAL_CHORD_POLICY_ID = "gate9k_residual_chords_conflict_answer_v1"
TRUSTED_EDGE_TYPES = ("temporal_transition", "support_anchor")
RESIDUAL_CHORD_EDGE_TYPES = ("conflict_anchor", "answer_projection")
EXCLUDED_EDGE_TYPES = ("quietness_pair",)

DEFAULT_REGISTRY = "trusted_tree_residual_chord_registry.jsonl"
DEFAULT_ROLE_SUMMARY = "trusted_tree_residual_chord_by_role_type.csv"
DEFAULT_CELL_SUMMARY = "trusted_tree_residual_chord_by_cell_role.csv"
DEFAULT_STATUS = "trusted_tree_residual_chord_status.json"
DEFAULT_REPORT = "gate9k_trusted_tree_residual_chord_read.md"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Gate9K trusted-tree / residual-chord logging audit over the "
            "recovered Gate9A graph bound through Gate9J."
        )
    )
    parser.add_argument("--gate9j-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive_source_context(
    source_gate9j_dir: Path,
) -> Tuple[Dict[str, Any], Path, Dict[str, Any], Path, Dict[str, Any], Path]:
    source_gate9j_manifest = gate9a.read_json(source_gate9j_dir / gate9a.DEFAULT_MANIFEST)
    source_gate9i_dir = REPO_ROOT / str(source_gate9j_manifest["source_gate9i_dir"])
    source_gate9i_manifest = gate9a.read_json(source_gate9i_dir / gate9a.DEFAULT_MANIFEST)
    source_gate9h_dir = REPO_ROOT / str(source_gate9i_manifest["source_gate9h_dir"])
    source_gate9h_manifest = gate9a.read_json(source_gate9h_dir / gate9a.DEFAULT_MANIFEST)
    source_gate9g_dir = REPO_ROOT / str(source_gate9h_manifest["source_gate9g_dir"])
    source_gate9g_manifest = gate9a.read_json(source_gate9g_dir / gate9a.DEFAULT_MANIFEST)
    source_gate9a_dir = REPO_ROOT / str(source_gate9g_manifest["source_gate9a_dir"])
    return (
        source_gate9j_manifest,
        source_gate9i_dir,
        source_gate9i_manifest,
        source_gate9h_dir,
        source_gate9g_manifest,
        source_gate9a_dir,
    )


def role_for_edge_type(edge_type: str) -> Tuple[str, str]:
    if edge_type in TRUSTED_EDGE_TYPES:
        return "trusted_tree_candidate", "trusted_edge_policy"
    if edge_type in RESIDUAL_CHORD_EDGE_TYPES:
        return "residual_chord_candidate", "residual_chord_policy"
    if edge_type in EXCLUDED_EDGE_TYPES:
        return "excluded_nonstructural", "excluded_from_decomposition"
    return "excluded_unknown", "unknown_edge_type"


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def build_registry_rows(edge_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    registry_rows: List[Dict[str, Any]] = []
    for row in edge_rows:
        role, reason = role_for_edge_type(str(row["edge_type"]))
        registry_rows.append(
            {
                "edge_id": str(row["edge_id"]),
                "edge_type": str(row["edge_type"]),
                "decomposition_role": role,
                "decomposition_reason": reason,
                "execution_sample_id": int(row["execution_sample_id"]),
                "benchmark_sample_id": str(row["benchmark_sample_id"]),
                "cell_id": str(row["cell_id"]),
                "world_id": str(row["world_id"]),
                "world_type": str(row["world_type"]),
                "answer_target_type": str(row["answer_target_type"]),
                "quietness_pair_id": str(row.get("quietness_pair_id") or ""),
                "rendering_family_id": str(row["rendering_family_id"]),
                "source_node_type": str(row["source_node_type"]),
                "target_node_type": str(row["target_node_type"]),
                "edge_outcome": str(row["edge_outcome"]),
                "edge_transport_defect": row.get("edge_transport_defect"),
                "transport_mode": str(row["transport_mode"]),
            }
        )
    return registry_rows


def summarize_by_role_type(registry_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in registry_rows:
        grouped[(str(row["decomposition_role"]), str(row["edge_type"]))].append(row)

    out_rows: List[Dict[str, Any]] = []
    for role_edge in sorted(grouped):
        rows = grouped[role_edge]
        defects = [
            float(row["edge_transport_defect"])
            for row in rows
            if row["edge_transport_defect"] not in (None, "")
        ]
        out_rows.append(
            {
                "decomposition_role": role_edge[0],
                "edge_type": role_edge[1],
                "n_edges": len(rows),
                "n_nonmissing_defects": len(defects),
                "mean_edge_transport_defect": mean_or_none(defects),
            }
        )
    return out_rows


def summarize_by_cell_role(registry_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in registry_rows:
        grouped[(str(row["cell_id"]), str(row["decomposition_role"]))].append(row)

    out_rows: List[Dict[str, Any]] = []
    for cell_role in sorted(grouped):
        rows = grouped[cell_role]
        defects = [
            float(row["edge_transport_defect"])
            for row in rows
            if row["edge_transport_defect"] not in (None, "")
        ]
        out_rows.append(
            {
                "cell_id": cell_role[0],
                "decomposition_role": cell_role[1],
                "n_edges": len(rows),
                "n_nonmissing_defects": len(defects),
                "mean_edge_transport_defect": mean_or_none(defects),
            }
        )
    return out_rows


def count_by_role(registry_rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for row in registry_rows:
        counts[str(row["decomposition_role"])] += 1
    return dict(counts)


def build_status_payload(
    registry_rows: Sequence[Dict[str, Any]],
    gate9i_status: Dict[str, Any],
    gate9j_status: Dict[str, Any],
) -> Dict[str, Any]:
    role_counts = count_by_role(registry_rows)
    return {
        "trusted_edge_policy_id": TRUSTED_EDGE_POLICY_ID,
        "residual_chord_policy_id": RESIDUAL_CHORD_POLICY_ID,
        "tree_construction_status": "declared_not_built",
        "tree_choice_dependence_status": "not_yet_executed",
        "scalar_masking_violation_status": "clear",
        "operator_admission_non_promotion_status": "enforced",
        "trusted_tree_candidate_edge_count": int(role_counts.get("trusted_tree_candidate", 0)),
        "residual_chord_candidate_edge_count": int(role_counts.get("residual_chord_candidate", 0)),
        "excluded_edge_count": int(role_counts.get("excluded_nonstructural", 0) + role_counts.get("excluded_unknown", 0)),
        "support_anchor_cleaner_dominance_status_at_bind": str(
            gate9i_status.get("support_anchor_cleaner_dominance_status") or ""
        ),
        "distributed_underactivation_status_at_bind": str(
            gate9j_status.get("distributed_underactivation_status") or ""
        ),
        "distributed_consistent_branch_status_at_bind": str(
            gate9j_status.get("distributed_consistent_branch_status") or ""
        ),
        "decomposition_hypothesis_execution_status": "not_yet_executed",
    }


def build_report(
    run_id: str,
    source_gate9j_manifest: Dict[str, Any],
    role_summary_rows: Sequence[Dict[str, Any]],
    cell_summary_rows: Sequence[Dict[str, Any]],
    status_payload: Dict[str, Any],
) -> str:
    lines = [
        "# Gate9K Trusted-Tree / Residual-Chord Read",
        "",
        f"run_id: {run_id}",
        f"source_gate9j_run_id: {source_gate9j_manifest.get('run_id', '')}",
        f"source_gate9j_code_git_commit: {source_gate9j_manifest.get('code_git_commit', '')}",
        "",
        "## Discipline",
        "",
        "- trusted-edge policy is declared, not yet executed as a tree",
        "- residual chord set is declared, not yet scored as a bypass verdict",
        "- scalar masking remains forbidden",
        "- operator admission remains denied",
        "",
        "## Role By Edge Type",
        "",
        "| decomposition_role | edge_type | n_edges | n_nonmissing_defects | mean_edge_transport_defect |",
        "|---|---|---:|---:|---:|",
    ]
    for row in role_summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["decomposition_role"]),
                    str(row["edge_type"]),
                    str(row["n_edges"]),
                    str(row["n_nonmissing_defects"]),
                    "" if row["mean_edge_transport_defect"] in (None, "") else f"{float(row['mean_edge_transport_defect']):.6f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Role By Cell",
            "",
            "| cell_id | decomposition_role | n_edges | n_nonmissing_defects | mean_edge_transport_defect |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in cell_summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["cell_id"]),
                    str(row["decomposition_role"]),
                    str(row["n_edges"]),
                    str(row["n_nonmissing_defects"]),
                    "" if row["mean_edge_transport_defect"] in (None, "") else f"{float(row['mean_edge_transport_defect']):.6f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Status",
            "",
            f"- trusted_edge_policy_id: `{status_payload['trusted_edge_policy_id']}`",
            f"- residual_chord_policy_id: `{status_payload['residual_chord_policy_id']}`",
            f"- tree_construction_status: `{status_payload['tree_construction_status']}`",
            f"- tree_choice_dependence_status: `{status_payload['tree_choice_dependence_status']}`",
            f"- scalar_masking_violation_status: `{status_payload['scalar_masking_violation_status']}`",
            f"- operator_admission_non_promotion_status: `{status_payload['operator_admission_non_promotion_status']}`",
            f"- support_anchor_cleaner_dominance_status_at_bind: `{status_payload['support_anchor_cleaner_dominance_status_at_bind']}`",
            f"- distributed_underactivation_status_at_bind: `{status_payload['distributed_underactivation_status_at_bind']}`",
            f"- distributed_consistent_branch_status_at_bind: `{status_payload['distributed_consistent_branch_status_at_bind']}`",
            f"- decomposition_hypothesis_execution_status: `{status_payload['decomposition_hypothesis_execution_status']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    source_gate9j_dir = Path(args.gate9j_dir)
    out_dir = Path(args.out_dir)
    run_id = out_dir.name

    (
        source_gate9j_manifest,
        source_gate9i_dir,
        _source_gate9i_manifest,
        _source_gate9h_dir,
        _source_gate9g_manifest,
        source_gate9a_dir,
    ) = derive_source_context(source_gate9j_dir)

    edge_rows = gate9a.read_jsonl(source_gate9a_dir / gate9a.DEFAULT_EDGE_REGISTRY)
    gate9i_status = gate9a.read_json(source_gate9i_dir / gate9i.DEFAULT_STATUS)
    gate9j_status = gate9a.read_json(source_gate9j_dir / gate9j.DEFAULT_STATUS)

    registry_rows = build_registry_rows(edge_rows)
    role_summary_rows = summarize_by_role_type(registry_rows)
    cell_summary_rows = summarize_by_cell_role(registry_rows)
    status_payload = build_status_payload(registry_rows, gate9i_status, gate9j_status)

    registry_path = out_dir / DEFAULT_REGISTRY
    role_summary_path = out_dir / DEFAULT_ROLE_SUMMARY
    cell_summary_path = out_dir / DEFAULT_CELL_SUMMARY
    status_path = out_dir / DEFAULT_STATUS
    report_path = out_dir / DEFAULT_REPORT
    manifest_path = out_dir / DEFAULT_MANIFEST
    checksums_path = out_dir / DEFAULT_CHECKSUMS

    gate9a.write_jsonl(registry_path, registry_rows)
    gate9a.write_csv(
        role_summary_path,
        ("decomposition_role", "edge_type", "n_edges", "n_nonmissing_defects", "mean_edge_transport_defect"),
        role_summary_rows,
    )
    gate9a.write_csv(
        cell_summary_path,
        ("cell_id", "decomposition_role", "n_edges", "n_nonmissing_defects", "mean_edge_transport_defect"),
        cell_summary_rows,
    )
    gate9a.write_json(status_path, status_payload)
    gate9a.write_text(
        report_path,
        build_report(
            run_id=run_id,
            source_gate9j_manifest=source_gate9j_manifest,
            role_summary_rows=role_summary_rows,
            cell_summary_rows=cell_summary_rows,
            status_payload=status_payload,
        ),
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "run_id": run_id,
        "code_git_commit": gate9a.current_git_commit(),
        "source_gate9j_dir": gate9a.repo_relative_or_posix(source_gate9j_dir),
        "source_gate9j_run_id": str(source_gate9j_manifest.get("run_id") or ""),
        "source_gate9j_code_git_commit": str(source_gate9j_manifest.get("code_git_commit") or ""),
        "source_gate9a_dir": gate9a.repo_relative_or_posix(source_gate9a_dir),
        "paths": {
            DEFAULT_REGISTRY: gate9a.repo_relative_or_posix(registry_path),
            DEFAULT_ROLE_SUMMARY: gate9a.repo_relative_or_posix(role_summary_path),
            DEFAULT_CELL_SUMMARY: gate9a.repo_relative_or_posix(cell_summary_path),
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
            DEFAULT_ROLE_SUMMARY: sha256_file(role_summary_path),
            DEFAULT_CELL_SUMMARY: sha256_file(cell_summary_path),
            DEFAULT_STATUS: sha256_file(status_path),
            DEFAULT_REPORT: sha256_file(report_path),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
