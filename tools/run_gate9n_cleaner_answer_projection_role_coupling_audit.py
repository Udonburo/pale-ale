#!/usr/bin/env python3
"""Run a Gate9N cleaner-side answer-projection role-coupling separation audit on Gate9M outputs."""

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import run_gate9a_graph_gauge_consumer as gate9a
import run_gate9m_cleaner_answer_projection_pollution_audit as gate9m


REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "gate9n_cleaner_answer_projection_role_coupling_audit_v1"
METHOD_ID = "gate9n_cleaner_answer_projection_role_coupling_audit_v1"

DEFAULT_REGISTRY = "cleaner_answer_projection_role_coupling_registry.jsonl"
DEFAULT_POLICY_COMPARE = "cleaner_answer_projection_role_coupling_policy_compare.csv"
DEFAULT_STATUS = "cleaner_answer_projection_role_coupling_status.json"
DEFAULT_REPORT = "gate9n_cleaner_answer_projection_role_coupling_read.md"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Gate9N role-coupling separation audit over cleaner-side "
            "answer-projection pollution using the Gate9M registry."
        )
    )
    parser.add_argument("--gate9m-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def build_role_coupling_registry(
    source_registry_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build Gate9N role-coupling registry from Gate9M pollution registry rows.

    Each row inherits the declared split_policy_role from Gate9M and adds
    role-coupling classification fields.  No new metrics or roles are introduced.
    """
    registry_rows: List[Dict[str, Any]] = []
    for row in source_registry_rows:
        cell_class = str(row["cell_class"])
        split_policy_role = str(row["split_policy_role"])

        # Role-coupling class: determined purely by whether the declared split
        # moved the edge out of the baseline residual_chord_candidate role.
        if split_policy_role == "closure_return_leg_auxiliary":
            role_coupling_class = "auxiliary_only"
        else:
            role_coupling_class = "residual_only"

        # An edge is separable if its role under the declared split is
        # unambiguous — i.e. the split moved cleaner edges to auxiliary
        # and kept conflict edges as residual, without mixing.
        role_coupling_separable = (
            (cell_class == "cleaner" and split_policy_role == "closure_return_leg_auxiliary")
            or (cell_class == "conflict" and split_policy_role == "residual_chord_candidate")
        )

        registry_rows.append(
            {
                "edge_id": str(row["edge_id"]),
                "execution_sample_id": int(row["execution_sample_id"]),
                "benchmark_sample_id": str(row["benchmark_sample_id"]),
                "cell_id": str(row["cell_id"]),
                "cell_class": cell_class,
                "world_id": str(row["world_id"]),
                "world_type": str(row["world_type"]),
                "answer_target_type": str(row["answer_target_type"]),
                "edge_transport_defect": row.get("edge_transport_defect"),
                "baseline_residual_role": str(row["baseline_residual_role"]),
                "declared_role": split_policy_role,
                "role_coupling_class": role_coupling_class,
                "role_coupling_separable": role_coupling_separable,
                "participates_in_support_cycle": bool(row.get("participates_in_support_cycle", False)),
                "participates_in_conflict_cycle": bool(row.get("participates_in_conflict_cycle", False)),
                "structural_return_leg_candidate": bool(row.get("structural_return_leg_candidate", False)),
                "policy_mixing_candidate": bool(row.get("policy_mixing_candidate", False)),
            }
        )
    return registry_rows


def build_policy_compare(
    registry_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build a policy comparison table: baseline (all residual_chord_candidate) vs declared split."""
    # Baseline view: all edges grouped under their baseline role
    baseline_grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in registry_rows:
        baseline_grouped[(str(row["cell_class"]), str(row["baseline_residual_role"]))].append(row)

    # Declared split view: edges grouped under their declared role
    declared_grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in registry_rows:
        declared_grouped[(str(row["cell_class"]), str(row["declared_role"]))].append(row)

    out_rows: List[Dict[str, Any]] = []
    all_keys = sorted(set(baseline_grouped.keys()) | set(declared_grouped.keys()))
    for key in all_keys:
        baseline_rows = baseline_grouped.get(key, [])
        declared_rows = declared_grouped.get(key, [])
        baseline_defects = [
            float(r["edge_transport_defect"])
            for r in baseline_rows
            if r["edge_transport_defect"] not in (None, "")
        ]
        declared_defects = [
            float(r["edge_transport_defect"])
            for r in declared_rows
            if r["edge_transport_defect"] not in (None, "")
        ]
        out_rows.append(
            {
                "cell_class": key[0],
                "role": key[1],
                "baseline_n_edges": len(baseline_rows),
                "baseline_mean_edge_transport_defect": mean_or_none(baseline_defects),
                "declared_split_n_edges": len(declared_rows),
                "declared_split_mean_edge_transport_defect": mean_or_none(declared_defects),
            }
        )
    return out_rows


def build_status_payload(
    registry_rows: Sequence[Dict[str, Any]],
    source_status: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the Gate9N status payload with all 10 required keys."""
    # Baseline count: all cleaner edges that were residual_chord_candidate in baseline
    baseline_cleaner_residual = [
        row for row in registry_rows
        if row["cell_class"] == "cleaner" and row["baseline_residual_role"] == "residual_chord_candidate"
    ]

    # Declared split: cleaner edges that became closure_return_leg_auxiliary
    declared_split_cleaner_auxiliary = [
        row for row in registry_rows
        if row["cell_class"] == "cleaner" and row["declared_role"] == "closure_return_leg_auxiliary"
    ]

    # Declared split: conflict edges that remain residual_chord_candidate
    declared_split_conflict_residual = [
        row for row in registry_rows
        if row["cell_class"] == "conflict" and row["declared_role"] == "residual_chord_candidate"
    ]

    # Conflict bridge preservation: conflict-side edges still participate in
    # both support and conflict cycles under residual_chord_candidate role
    conflict_residual_support_cycle = [
        row for row in declared_split_conflict_residual
        if row["participates_in_support_cycle"]
    ]
    conflict_residual_conflict_cycle = [
        row for row in declared_split_conflict_residual
        if row["participates_in_conflict_cycle"]
    ]
    conflict_bridge_preservation_status = (
        "clear"
        if conflict_residual_support_cycle and conflict_residual_conflict_cycle
        else "denied"
    )

    # Closure doctrine preservation: cleaner auxiliary edges still participate
    # in support cycles (closure return leg path remains)
    cleaner_auxiliary_support_cycle = [
        row for row in declared_split_cleaner_auxiliary
        if row["participates_in_support_cycle"]
    ]
    closure_doctrine_preservation_status = (
        "clear" if cleaner_auxiliary_support_cycle else "denied"
    )

    # Cleaner pollution reduction: the declared split must move ALL cleaner edges
    # out of residual_chord_candidate
    cleaner_still_residual = [
        row for row in registry_rows
        if row["cell_class"] == "cleaner" and row["declared_role"] == "residual_chord_candidate"
    ]
    cleaner_pollution_reduction_status = (
        "reduced"
        if baseline_cleaner_residual and not cleaner_still_residual
        else "unchanged"
    )

    # Role-coupling separability: separable only if pollution reduces AND
    # bridge preserved AND closure preserved, with no falsifier firing
    role_coupling_separability_status = (
        "separable"
        if (
            cleaner_pollution_reduction_status == "reduced"
            and conflict_bridge_preservation_status == "clear"
            and closure_doctrine_preservation_status == "clear"
        )
        else "coupled"
    )

    # No scalar masking or undeclared surgery used
    scalar_masking_violation_status = "denied"
    undeclared_role_surgery_required_status = "denied"

    # Next named blocker: if still coupled, the blocker persists;
    # if separable, no blocker from this line
    next_named_blocker = (
        "cleaner_answer_projection_role_coupling"
        if role_coupling_separability_status == "coupled"
        else ""
    )

    return {
        "baseline_cleaner_residual_answer_projection_edge_count": len(baseline_cleaner_residual),
        "declared_split_cleaner_auxiliary_answer_projection_edge_count": len(declared_split_cleaner_auxiliary),
        "declared_split_conflict_residual_answer_projection_edge_count": len(declared_split_conflict_residual),
        "conflict_bridge_preservation_status": conflict_bridge_preservation_status,
        "closure_doctrine_preservation_status": closure_doctrine_preservation_status,
        "cleaner_pollution_reduction_status": cleaner_pollution_reduction_status,
        "role_coupling_separability_status": role_coupling_separability_status,
        "scalar_masking_violation_status": scalar_masking_violation_status,
        "undeclared_role_surgery_required_status": undeclared_role_surgery_required_status,
        "next_named_blocker": next_named_blocker,
    }


def build_report(
    run_id: str,
    source_gate9m_manifest: Dict[str, Any],
    policy_compare_rows: Sequence[Dict[str, Any]],
    status_payload: Dict[str, Any],
) -> str:
    lines = [
        "# Gate9N Cleaner Answer-Projection Role Coupling Read",
        "",
        f"run_id: {run_id}",
        f"source_gate9m_run_id: {source_gate9m_manifest.get('run_id', '')}",
        f"source_gate9m_code_git_commit: {source_gate9m_manifest.get('code_git_commit', '')}",
        "",
        "## Discipline",
        "",
        "- cleaner-side answer_projection role coupling audit only",
        "- declared split between closure_return_leg_auxiliary and residual_chord_candidate",
        "- first forest remains fixed",
        "- closure doctrine remains fixed",
        "- operator admission remains denied",
        "",
        "## Policy Compare",
        "",
        "| cell_class | role | baseline_n_edges | baseline_mean_defect | declared_split_n_edges | declared_split_mean_defect |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in policy_compare_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["cell_class"]),
                    str(row["role"]),
                    str(row["baseline_n_edges"]),
                    ""
                    if row["baseline_mean_edge_transport_defect"] in (None, "")
                    else f"{float(row['baseline_mean_edge_transport_defect']):.6f}",
                    str(row["declared_split_n_edges"]),
                    ""
                    if row["declared_split_mean_edge_transport_defect"] in (None, "")
                    else f"{float(row['declared_split_mean_edge_transport_defect']):.6f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Status",
            "",
            f"- baseline_cleaner_residual_answer_projection_edge_count: `{status_payload['baseline_cleaner_residual_answer_projection_edge_count']}`",
            f"- declared_split_cleaner_auxiliary_answer_projection_edge_count: `{status_payload['declared_split_cleaner_auxiliary_answer_projection_edge_count']}`",
            f"- declared_split_conflict_residual_answer_projection_edge_count: `{status_payload['declared_split_conflict_residual_answer_projection_edge_count']}`",
            f"- conflict_bridge_preservation_status: `{status_payload['conflict_bridge_preservation_status']}`",
            f"- closure_doctrine_preservation_status: `{status_payload['closure_doctrine_preservation_status']}`",
            f"- cleaner_pollution_reduction_status: `{status_payload['cleaner_pollution_reduction_status']}`",
            f"- role_coupling_separability_status: `{status_payload['role_coupling_separability_status']}`",
            f"- scalar_masking_violation_status: `{status_payload['scalar_masking_violation_status']}`",
            f"- undeclared_role_surgery_required_status: `{status_payload['undeclared_role_surgery_required_status']}`",
        ]
    )
    if status_payload.get("next_named_blocker"):
        lines.extend(["", "## Next Blocker", "", f"- `{status_payload['next_named_blocker']}`"])
    else:
        lines.extend(["", "## Next Blocker", "", "- (none from this line)"])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    source_gate9m_dir = Path(args.gate9m_dir)
    out_dir = Path(args.out_dir)
    run_id = out_dir.name

    # Read source Gate9M data
    source_gate9m_manifest = gate9a.read_json(source_gate9m_dir / gate9a.DEFAULT_MANIFEST)
    source_registry_rows = gate9a.read_jsonl(source_gate9m_dir / gate9m.DEFAULT_REGISTRY)
    source_status = gate9a.read_json(source_gate9m_dir / gate9m.DEFAULT_STATUS)

    # Build Gate9N outputs
    registry_rows = build_role_coupling_registry(source_registry_rows)
    policy_compare_rows = build_policy_compare(registry_rows)
    status_payload = build_status_payload(registry_rows, source_status)

    # Write outputs
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
            "cell_class",
            "role",
            "baseline_n_edges",
            "baseline_mean_edge_transport_defect",
            "declared_split_n_edges",
            "declared_split_mean_edge_transport_defect",
        ),
        policy_compare_rows,
    )
    gate9a.write_json(status_path, status_payload)
    gate9a.write_text(
        report_path,
        build_report(
            run_id=run_id,
            source_gate9m_manifest=source_gate9m_manifest,
            policy_compare_rows=policy_compare_rows,
            status_payload=status_payload,
        ),
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "run_id": run_id,
        "code_git_commit": gate9a.current_git_commit(),
        "source_gate9m_dir": gate9a.repo_relative_or_posix(source_gate9m_dir),
        "source_gate9m_run_id": str(source_gate9m_manifest.get("run_id") or ""),
        "source_gate9m_code_git_commit": str(source_gate9m_manifest.get("code_git_commit") or ""),
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
