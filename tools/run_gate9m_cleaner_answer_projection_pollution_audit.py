#!/usr/bin/env python3
"""Run a Gate9M cleaner-side answer-projection pollution audit on Gate9L outputs."""

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import run_gate9a_graph_gauge_consumer as gate9a
import run_gate9l_first_tree_answer_projection_pollution_audit as gate9l


REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "gate9m_cleaner_answer_projection_pollution_audit_v1"
METHOD_ID = "gate9m_cleaner_answer_projection_pollution_audit_v1"

DEFAULT_REGISTRY = "cleaner_answer_projection_pollution_registry.jsonl"
DEFAULT_POLICY_SUMMARY = "cleaner_answer_projection_policy_compare.csv"
DEFAULT_STATUS = "cleaner_answer_projection_pollution_status.json"
DEFAULT_REPORT = "gate9m_cleaner_answer_projection_pollution_read.md"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"

CLEANER_CELLS = {"clean_support", "surface_noisy_clean"}
CONFLICT_CELLS = {"direct_contradiction", "distributed_incompatibility"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Gate9M audit over cleaner-side answer-projection pollution "
            "using the fixed first forest from Gate9L."
        )
    )
    parser.add_argument("--gate9l-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive_source_context(
    source_gate9l_dir: Path,
) -> Tuple[Dict[str, Any], Path, Dict[str, Any], Path]:
    source_gate9l_manifest = gate9a.read_json(source_gate9l_dir / gate9a.DEFAULT_MANIFEST)
    source_gate9k_dir = REPO_ROOT / str(source_gate9l_manifest["source_gate9k_dir"])
    source_gate9k_manifest = gate9a.read_json(source_gate9k_dir / gate9a.DEFAULT_MANIFEST)
    source_gate9a_dir = REPO_ROOT / str(source_gate9l_manifest["source_gate9a_dir"])
    return source_gate9l_manifest, source_gate9k_dir, source_gate9k_manifest, source_gate9a_dir


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def collect_cycle_answer_projection_sets(cycle_rows: Sequence[Dict[str, Any]]) -> Tuple[Set[str], Set[str]]:
    support_edges: Set[str] = set()
    conflict_edges: Set[str] = set()
    for row in cycle_rows:
        if str(row.get("cycle_outcome")) != "none":
            continue
        cycle_type = str(row.get("cycle_type"))
        for edge_id in row.get("edge_ids") or []:
            if ":answer_projection" not in str(edge_id):
                continue
            if cycle_type == "support_answer_terminal_token_cycle":
                support_edges.add(str(edge_id))
            elif cycle_type == "conflict_answer_terminal_token_cycle":
                conflict_edges.add(str(edge_id))
    return support_edges, conflict_edges


def build_registry_rows(
    residual_rows: Sequence[Dict[str, Any]],
    support_cycle_answer_projection_edges: Set[str],
    conflict_cycle_answer_projection_edges: Set[str],
) -> List[Dict[str, Any]]:
    registry_rows: List[Dict[str, Any]] = []
    for row in residual_rows:
        if str(row["edge_type"]) != "answer_projection":
            continue
        cell_id = str(row["cell_id"])
        cell_class = "cleaner" if cell_id in CLEANER_CELLS else "conflict" if cell_id in CONFLICT_CELLS else "other"
        edge_id = str(row["edge_id"])
        in_support_cycle = edge_id in support_cycle_answer_projection_edges
        in_conflict_cycle = edge_id in conflict_cycle_answer_projection_edges
        split_policy_role = (
            "closure_return_leg_auxiliary"
            if cell_class == "cleaner"
            else "residual_chord_candidate"
        )
        registry_rows.append(
            {
                "edge_id": edge_id,
                "execution_sample_id": int(row["execution_sample_id"]),
                "benchmark_sample_id": str(row["benchmark_sample_id"]),
                "cell_id": cell_id,
                "cell_class": cell_class,
                "world_id": str(row["world_id"]),
                "world_type": str(row["world_type"]),
                "answer_target_type": str(row["answer_target_type"]),
                "edge_transport_defect": row.get("edge_transport_defect"),
                "baseline_residual_role": "residual_chord_candidate",
                "split_policy_role": split_policy_role,
                "participates_in_support_cycle": in_support_cycle,
                "participates_in_conflict_cycle": in_conflict_cycle,
                "structural_return_leg_candidate": cell_class == "cleaner" and in_support_cycle,
                "policy_mixing_candidate": cell_class == "cleaner" and in_support_cycle and split_policy_role == "closure_return_leg_auxiliary",
            }
        )
    return registry_rows


def summarize_policy_compare(registry_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in registry_rows:
        grouped[(str(row["cell_class"]), str(row["split_policy_role"]))].append(row)

    out_rows: List[Dict[str, Any]] = []
    for key in sorted(grouped):
        rows = grouped[key]
        defects = [
            float(row["edge_transport_defect"])
            for row in rows
            if row["edge_transport_defect"] not in (None, "")
        ]
        out_rows.append(
            {
                "cell_class": key[0],
                "split_policy_role": key[1],
                "n_edges": len(rows),
                "mean_edge_transport_defect": mean_or_none(defects),
            }
        )
    return out_rows


def build_status_payload(registry_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    cleaner_rows = [row for row in registry_rows if row["cell_class"] == "cleaner"]
    conflict_rows = [row for row in registry_rows if row["cell_class"] == "conflict"]
    cleaner_support_cycle_rows = [row for row in cleaner_rows if row["participates_in_support_cycle"]]
    cleaner_conflict_cycle_rows = [row for row in cleaner_rows if row["participates_in_conflict_cycle"]]
    split_policy_cleaner_residual_rows = [row for row in cleaner_rows if row["split_policy_role"] == "residual_chord_candidate"]
    split_policy_conflict_rows = [row for row in conflict_rows if row["split_policy_role"] == "residual_chord_candidate"]
    split_policy_conflict_support_cycle_rows = [row for row in split_policy_conflict_rows if row["participates_in_support_cycle"]]
    split_policy_conflict_conflict_cycle_rows = [row for row in split_policy_conflict_rows if row["participates_in_conflict_cycle"]]

    structural_return_leg_pollution_status = (
        "triggered" if cleaner_support_cycle_rows and not cleaner_conflict_cycle_rows else "clear"
    )
    policy_mixing_pollution_status = (
        "triggered"
        if not split_policy_cleaner_residual_rows
        and split_policy_conflict_conflict_cycle_rows
        and split_policy_conflict_support_cycle_rows
        else "clear"
    )
    removing_cleaner_answer_projection_breaks_closure_doctrine_status = (
        "triggered" if cleaner_support_cycle_rows else "clear"
    )
    suppression_requires_scalar_masking_status = "denied"
    undeclared_role_surgery_required_status = "denied"
    return {
        "baseline_cleaner_residual_answer_projection_edge_count": len(cleaner_rows),
        "cleaner_support_cycle_answer_projection_edge_count": len(cleaner_support_cycle_rows),
        "cleaner_conflict_cycle_answer_projection_edge_count": len(cleaner_conflict_cycle_rows),
        "split_policy_cleaner_residual_answer_projection_edge_count": len(split_policy_cleaner_residual_rows),
        "split_policy_conflict_residual_answer_projection_edge_count": len(split_policy_conflict_rows),
        "split_policy_conflict_bridge_preservation_status": (
            "clear"
            if split_policy_conflict_conflict_cycle_rows and split_policy_conflict_support_cycle_rows
            else "denied"
        ),
        "structural_return_leg_pollution_status": structural_return_leg_pollution_status,
        "policy_mixing_pollution_status": policy_mixing_pollution_status,
        "removing_cleaner_answer_projection_breaks_closure_doctrine_status": removing_cleaner_answer_projection_breaks_closure_doctrine_status,
        "suppression_requires_scalar_masking_status": suppression_requires_scalar_masking_status,
        "undeclared_role_surgery_required_status": undeclared_role_surgery_required_status,
        "next_named_blocker": (
            "cleaner_answer_projection_role_coupling"
            if structural_return_leg_pollution_status == "triggered" and policy_mixing_pollution_status == "triggered"
            else ""
        ),
    }


def build_report(
    run_id: str,
    source_gate9l_manifest: Dict[str, Any],
    policy_summary_rows: Sequence[Dict[str, Any]],
    status_payload: Dict[str, Any],
) -> str:
    lines = [
        "# Gate9M Cleaner Answer-Projection Pollution Read",
        "",
        f"run_id: {run_id}",
        f"source_gate9l_run_id: {source_gate9l_manifest.get('run_id', '')}",
        f"source_gate9l_code_git_commit: {source_gate9l_manifest.get('code_git_commit', '')}",
        "",
        "## Discipline",
        "",
        "- cleaner-side answer_projection edges only",
        "- first forest remains fixed",
        "- role split is declared as audit only, not yet executed as recovery",
        "- operator admission remains denied",
        "",
        "## Split Policy Summary",
        "",
        "| cell_class | split_policy_role | n_edges | mean_edge_transport_defect |",
        "|---|---|---:|---:|",
    ]
    for row in policy_summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["cell_class"]),
                    str(row["split_policy_role"]),
                    str(row["n_edges"]),
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
            f"- structural_return_leg_pollution_status: `{status_payload['structural_return_leg_pollution_status']}`",
            f"- policy_mixing_pollution_status: `{status_payload['policy_mixing_pollution_status']}`",
            f"- split_policy_conflict_bridge_preservation_status: `{status_payload['split_policy_conflict_bridge_preservation_status']}`",
            f"- removing_cleaner_answer_projection_breaks_closure_doctrine_status: `{status_payload['removing_cleaner_answer_projection_breaks_closure_doctrine_status']}`",
            f"- suppression_requires_scalar_masking_status: `{status_payload['suppression_requires_scalar_masking_status']}`",
            f"- undeclared_role_surgery_required_status: `{status_payload['undeclared_role_surgery_required_status']}`",
        ]
    )
    if status_payload.get("next_named_blocker"):
        lines.extend(["", "## Next Blocker", "", f"- `{status_payload['next_named_blocker']}`"])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    source_gate9l_dir = Path(args.gate9l_dir)
    out_dir = Path(args.out_dir)
    run_id = out_dir.name

    source_gate9l_manifest, _source_gate9k_dir, _source_gate9k_manifest, source_gate9a_dir = derive_source_context(
        source_gate9l_dir
    )
    residual_rows = gate9a.read_jsonl(source_gate9l_dir / gate9l.DEFAULT_RESIDUAL_REGISTRY)
    cycle_rows = gate9a.read_jsonl(source_gate9a_dir / gate9a.DEFAULT_CYCLE_REGISTRY)

    support_cycle_answer_projection_edges, conflict_cycle_answer_projection_edges = collect_cycle_answer_projection_sets(cycle_rows)
    registry_rows = build_registry_rows(
        residual_rows,
        support_cycle_answer_projection_edges=support_cycle_answer_projection_edges,
        conflict_cycle_answer_projection_edges=conflict_cycle_answer_projection_edges,
    )
    policy_summary_rows = summarize_policy_compare(registry_rows)
    status_payload = build_status_payload(registry_rows)

    registry_path = out_dir / DEFAULT_REGISTRY
    policy_summary_path = out_dir / DEFAULT_POLICY_SUMMARY
    status_path = out_dir / DEFAULT_STATUS
    report_path = out_dir / DEFAULT_REPORT
    manifest_path = out_dir / DEFAULT_MANIFEST
    checksums_path = out_dir / DEFAULT_CHECKSUMS

    gate9a.write_jsonl(registry_path, registry_rows)
    gate9a.write_csv(
        policy_summary_path,
        ("cell_class", "split_policy_role", "n_edges", "mean_edge_transport_defect"),
        policy_summary_rows,
    )
    gate9a.write_json(status_path, status_payload)
    gate9a.write_text(
        report_path,
        build_report(
            run_id=run_id,
            source_gate9l_manifest=source_gate9l_manifest,
            policy_summary_rows=policy_summary_rows,
            status_payload=status_payload,
        ),
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "run_id": run_id,
        "code_git_commit": gate9a.current_git_commit(),
        "source_gate9l_dir": gate9a.repo_relative_or_posix(source_gate9l_dir),
        "source_gate9l_run_id": str(source_gate9l_manifest.get("run_id") or ""),
        "source_gate9l_code_git_commit": str(source_gate9l_manifest.get("code_git_commit") or ""),
        "source_gate9a_dir": gate9a.repo_relative_or_posix(source_gate9a_dir),
        "paths": {
            DEFAULT_REGISTRY: gate9a.repo_relative_or_posix(registry_path),
            DEFAULT_POLICY_SUMMARY: gate9a.repo_relative_or_posix(policy_summary_path),
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
            DEFAULT_POLICY_SUMMARY: sha256_file(policy_summary_path),
            DEFAULT_STATUS: sha256_file(status_path),
            DEFAULT_REPORT: sha256_file(report_path),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
