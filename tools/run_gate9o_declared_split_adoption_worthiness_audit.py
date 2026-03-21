#!/usr/bin/env python3
"""Run a Gate9O declared-split adoption-worthiness audit on Gate9N outputs."""

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import run_gate9a_graph_gauge_consumer as gate9a
import run_gate9n_cleaner_answer_projection_role_coupling_audit as gate9n


REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "gate9o_declared_split_adoption_worthiness_audit_v1"
METHOD_ID = "gate9o_declared_split_adoption_worthiness_audit_v1"

DEFAULT_REGISTRY = "declared_split_adoption_worthiness_registry.jsonl"
DEFAULT_POLICY_COMPARE = "declared_split_adoption_worthiness_policy_compare.csv"
DEFAULT_STATUS = "declared_split_adoption_worthiness_status.json"
DEFAULT_REPORT = "gate9o_declared_split_adoption_worthiness_read.md"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Gate9O adoption-worthiness audit for the declared split "
            "using the Gate9N role-coupling registry."
        )
    )
    parser.add_argument("--gate9n-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def build_adoption_worthiness_registry(
    source_registry_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build Gate9O adoption-worthiness registry from Gate9N role-coupling registry.

    Each row inherits all Gate9N fields and adds adoption-worthiness
    classification.  No new metrics or roles are introduced.
    """
    registry_rows: List[Dict[str, Any]] = []
    for row in source_registry_rows:
        cell_class = str(row["cell_class"])
        baseline_role = str(row["baseline_residual_role"])
        declared_role = str(row["declared_role"])

        # Baseline bypass contribution: under baseline policy, cleaner edges
        # in residual_chord_candidate block bypass.
        baseline_blocks_bypass = (
            cell_class == "cleaner" and baseline_role == "residual_chord_candidate"
        )

        # Declared split bypass contribution: under declared split, cleaner
        # edges are moved to closure_return_leg_auxiliary, removing the block.
        declared_split_blocks_bypass = (
            cell_class == "cleaner" and declared_role == "residual_chord_candidate"
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
                "baseline_residual_role": baseline_role,
                "declared_role": declared_role,
                "role_coupling_class": str(row["role_coupling_class"]),
                "role_coupling_separable": bool(row.get("role_coupling_separable", False)),
                "participates_in_support_cycle": bool(row.get("participates_in_support_cycle", False)),
                "participates_in_conflict_cycle": bool(row.get("participates_in_conflict_cycle", False)),
                "baseline_blocks_bypass": baseline_blocks_bypass,
                "declared_split_blocks_bypass": declared_split_blocks_bypass,
            }
        )
    return registry_rows


def build_policy_compare(
    registry_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build policy comparison table for adoption-worthiness: baseline vs declared split."""
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in registry_rows:
        grouped[(
            str(row["cell_class"]),
            str(row["baseline_residual_role"]),
            str(row["declared_role"]),
        )].append(row)

    out_rows: List[Dict[str, Any]] = []
    for key in sorted(grouped):
        rows = grouped[key]
        defects = [
            float(r["edge_transport_defect"])
            for r in rows
            if r["edge_transport_defect"] not in (None, "")
        ]
        baseline_bypass_blockers = sum(1 for r in rows if r["baseline_blocks_bypass"])
        declared_split_bypass_blockers = sum(1 for r in rows if r["declared_split_blocks_bypass"])
        out_rows.append(
            {
                "cell_class": key[0],
                "baseline_role": key[1],
                "declared_role": key[2],
                "n_edges": len(rows),
                "mean_edge_transport_defect": mean_or_none(defects),
                "baseline_bypass_blockers": baseline_bypass_blockers,
                "declared_split_bypass_blockers": declared_split_bypass_blockers,
            }
        )
    return out_rows


def build_status_payload(
    registry_rows: Sequence[Dict[str, Any]],
    source_gate9n_status: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the Gate9O status payload with all 10 required keys."""
    # Baseline bypass readiness: denied if any cleaner edge blocks bypass
    # under baseline policy (cleaner residual_chord_candidate exists).
    baseline_bypass_blockers = [
        row for row in registry_rows
        if row["baseline_blocks_bypass"]
    ]
    baseline_bypass_readiness_status = (
        "denied" if baseline_bypass_blockers else "clear"
    )

    # Declared split bypass readiness: denied if any cleaner edge still blocks
    # bypass under declared split.
    declared_split_bypass_blockers = [
        row for row in registry_rows
        if row["declared_split_blocks_bypass"]
    ]
    declared_split_bypass_readiness_status = (
        "denied" if declared_split_bypass_blockers else "clear"
    )

    # Carry forward Gate9N preservation statuses (do not recompute — they are
    # fixed from the upstream audit and must not be silently changed).
    conflict_bridge_preservation_status = str(
        source_gate9n_status.get("conflict_bridge_preservation_status", "")
    )
    closure_doctrine_preservation_status = str(
        source_gate9n_status.get("closure_doctrine_preservation_status", "")
    )
    cleaner_pollution_reduction_status = str(
        source_gate9n_status.get("cleaner_pollution_reduction_status", "")
    )

    # Decision-relevant reduction: the split not only reduces cleaner pollution
    # in absolute terms (already shown by Gate9N) but also changes bypass
    # readiness from denied to clear — that is the decision-relevant threshold.
    decision_relevant_cleaner_pollution_reduction_status = (
        "decision_relevant"
        if (
            baseline_bypass_readiness_status == "denied"
            and declared_split_bypass_readiness_status == "clear"
            and cleaner_pollution_reduction_status == "reduced"
        )
        else "not_decision_relevant"
    )

    # Operator admission is not promoted by this audit.
    operator_admission_non_promotion_status = "confirmed"

    # Scalar masking: carried from Gate9N, must remain denied.
    scalar_masking_violation_status = str(
        source_gate9n_status.get("scalar_masking_violation_status", "denied")
    )

    # Adoption-worthiness: the split is adoption-worthy only if ALL of:
    # 1. it reduces cleaner pollution in a decision-relevant way
    # 2. conflict bridge is preserved
    # 3. closure doctrine is preserved
    # 4. no scalar masking
    # 5. operator admission is not promoted
    adoption_worthiness_status = (
        "adoption_worthy"
        if (
            decision_relevant_cleaner_pollution_reduction_status == "decision_relevant"
            and conflict_bridge_preservation_status == "clear"
            and closure_doctrine_preservation_status == "clear"
            and scalar_masking_violation_status == "denied"
        )
        else "not_adoption_worthy"
    )

    # Next named blocker: if not adoption-worthy, carry the coupling blocker;
    # if adoption-worthy, no blocker from this line.
    next_named_blocker = (
        "cleaner_answer_projection_role_coupling"
        if adoption_worthiness_status == "not_adoption_worthy"
        else ""
    )

    return {
        "baseline_bypass_readiness_status": baseline_bypass_readiness_status,
        "declared_split_bypass_readiness_status": declared_split_bypass_readiness_status,
        "conflict_bridge_preservation_status": conflict_bridge_preservation_status,
        "closure_doctrine_preservation_status": closure_doctrine_preservation_status,
        "cleaner_pollution_reduction_status": cleaner_pollution_reduction_status,
        "decision_relevant_cleaner_pollution_reduction_status": decision_relevant_cleaner_pollution_reduction_status,
        "adoption_worthiness_status": adoption_worthiness_status,
        "operator_admission_non_promotion_status": operator_admission_non_promotion_status,
        "scalar_masking_violation_status": scalar_masking_violation_status,
        "next_named_blocker": next_named_blocker,
    }


def build_report(
    run_id: str,
    source_gate9n_manifest: Dict[str, Any],
    policy_compare_rows: Sequence[Dict[str, Any]],
    status_payload: Dict[str, Any],
) -> str:
    lines = [
        "# Gate9O Declared Split Adoption-Worthiness Read",
        "",
        f"run_id: {run_id}",
        f"source_gate9n_run_id: {source_gate9n_manifest.get('run_id', '')}",
        f"source_gate9n_code_git_commit: {source_gate9n_manifest.get('code_git_commit', '')}",
        "",
        "## Discipline",
        "",
        "- adoption-worthiness audit only, not mainline adoption",
        "- baseline: Gate9L/Gate9M residual-role policy",
        "- declared split: Gate9N closure_return_leg_auxiliary / residual_chord_candidate",
        "- first forest remains fixed",
        "- closure doctrine remains fixed",
        "- operator admission remains denied",
        "",
        "## Policy Compare",
        "",
        "| cell_class | baseline_role | declared_role | n_edges | mean_defect | baseline_bypass_blockers | declared_split_bypass_blockers |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in policy_compare_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["cell_class"]),
                    str(row["baseline_role"]),
                    str(row["declared_role"]),
                    str(row["n_edges"]),
                    ""
                    if row["mean_edge_transport_defect"] in (None, "")
                    else f"{float(row['mean_edge_transport_defect']):.6f}",
                    str(row["baseline_bypass_blockers"]),
                    str(row["declared_split_bypass_blockers"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Status",
            "",
            f"- baseline_bypass_readiness_status: `{status_payload['baseline_bypass_readiness_status']}`",
            f"- declared_split_bypass_readiness_status: `{status_payload['declared_split_bypass_readiness_status']}`",
            f"- conflict_bridge_preservation_status: `{status_payload['conflict_bridge_preservation_status']}`",
            f"- closure_doctrine_preservation_status: `{status_payload['closure_doctrine_preservation_status']}`",
            f"- cleaner_pollution_reduction_status: `{status_payload['cleaner_pollution_reduction_status']}`",
            f"- decision_relevant_cleaner_pollution_reduction_status: `{status_payload['decision_relevant_cleaner_pollution_reduction_status']}`",
            f"- adoption_worthiness_status: `{status_payload['adoption_worthiness_status']}`",
            f"- operator_admission_non_promotion_status: `{status_payload['operator_admission_non_promotion_status']}`",
            f"- scalar_masking_violation_status: `{status_payload['scalar_masking_violation_status']}`",
        ]
    )
    if status_payload.get("next_named_blocker"):
        lines.extend(["", "## Next Blocker", "", f"- `{status_payload['next_named_blocker']}`"])
    else:
        lines.extend(["", "## Next Blocker", "", "- (none from this line)"])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    source_gate9n_dir = Path(args.gate9n_dir)
    out_dir = Path(args.out_dir)
    run_id = out_dir.name

    # Read source Gate9N data
    source_gate9n_manifest = gate9a.read_json(source_gate9n_dir / gate9a.DEFAULT_MANIFEST)
    source_registry_rows = gate9a.read_jsonl(source_gate9n_dir / gate9n.DEFAULT_REGISTRY)
    source_gate9n_status = gate9a.read_json(source_gate9n_dir / gate9n.DEFAULT_STATUS)

    # Build Gate9O outputs
    registry_rows = build_adoption_worthiness_registry(source_registry_rows)
    policy_compare_rows = build_policy_compare(registry_rows)
    status_payload = build_status_payload(registry_rows, source_gate9n_status)

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
            "baseline_role",
            "declared_role",
            "n_edges",
            "mean_edge_transport_defect",
            "baseline_bypass_blockers",
            "declared_split_bypass_blockers",
        ),
        policy_compare_rows,
    )
    gate9a.write_json(status_path, status_payload)
    gate9a.write_text(
        report_path,
        build_report(
            run_id=run_id,
            source_gate9n_manifest=source_gate9n_manifest,
            policy_compare_rows=policy_compare_rows,
            status_payload=status_payload,
        ),
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "run_id": run_id,
        "code_git_commit": gate9a.current_git_commit(),
        "source_gate9n_dir": gate9a.repo_relative_or_posix(source_gate9n_dir),
        "source_gate9n_run_id": str(source_gate9n_manifest.get("run_id") or ""),
        "source_gate9n_code_git_commit": str(source_gate9n_manifest.get("code_git_commit") or ""),
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
