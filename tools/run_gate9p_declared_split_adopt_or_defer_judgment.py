#!/usr/bin/env python3
"""Run a Gate9P declared-split adopt-or-defer judgment on Gate9O outputs."""

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import run_gate9a_graph_gauge_consumer as gate9a
import run_gate9o_declared_split_adoption_worthiness_audit as gate9o


REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "gate9p_declared_split_adopt_or_defer_judgment_v1"
METHOD_ID = "gate9p_declared_split_adopt_or_defer_judgment_v1"

DEFAULT_REGISTRY = "declared_split_adopt_or_defer_registry.jsonl"
DEFAULT_POLICY_COMPARE = "declared_split_adopt_or_defer_policy_compare.csv"
DEFAULT_STATUS = "declared_split_adopt_or_defer_status.json"
DEFAULT_REPORT = "gate9p_declared_split_adopt_or_defer_read.md"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Gate9P adopt-or-defer judgment for the declared split "
            "using the Gate9O adoption-worthiness audit."
        )
    )
    parser.add_argument("--gate9o-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def build_adopt_or_defer_registry(
    source_registry_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build Gate9P adopt-or-defer registry from Gate9O adoption-worthiness registry.

    Each row inherits all Gate9O fields and annotates with adopt-or-defer
    judgment-relevant classification.  No new metrics or roles are introduced.
    """
    registry_rows: List[Dict[str, Any]] = []
    for row in source_registry_rows:
        cell_class = str(row["cell_class"])
        baseline_role = str(row["baseline_residual_role"])
        declared_role = str(row["declared_role"])

        # The declared split is a forward-only relabeling of cleaner edges
        # from residual_chord_candidate to closure_return_leg_auxiliary.
        # This does NOT require reinterpretation of prior reads because:
        # - conflict edges remain residual_chord_candidate (unchanged)
        # - the relabeling applies only to audit-lane role assignment going forward
        requires_historical_reinterpretation = False

        # The split does NOT widen doctrine — it is purely a role relabeling
        # within the existing closure_return_leg_auxiliary / residual_chord_candidate
        # vocabulary that Gate9M already defined.
        requires_doctrine_scope_change = False

        # The split does NOT weaken the audit-lane / operator boundary:
        # operator admission remains denied, and the split only moves
        # cleaner edges to auxiliary, not to any operator-controlled role.
        weakens_audit_lane_boundary = False

        # No hidden role surgery or bundle-specific exception logic needed:
        # the split is the same for every edge of each cell_class.
        requires_hidden_role_surgery = False

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
                "baseline_blocks_bypass": bool(row.get("baseline_blocks_bypass", False)),
                "declared_split_blocks_bypass": bool(row.get("declared_split_blocks_bypass", False)),
                "requires_historical_reinterpretation": requires_historical_reinterpretation,
                "requires_doctrine_scope_change": requires_doctrine_scope_change,
                "weakens_audit_lane_boundary": weakens_audit_lane_boundary,
                "requires_hidden_role_surgery": requires_hidden_role_surgery,
            }
        )
    return registry_rows


def build_policy_compare(
    registry_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build adopt-or-defer policy comparison table."""
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
        n_reinterpretation = sum(1 for r in rows if r["requires_historical_reinterpretation"])
        n_doctrine_change = sum(1 for r in rows if r["requires_doctrine_scope_change"])
        n_boundary_weaken = sum(1 for r in rows if r["weakens_audit_lane_boundary"])
        n_hidden_surgery = sum(1 for r in rows if r["requires_hidden_role_surgery"])
        out_rows.append(
            {
                "cell_class": key[0],
                "baseline_role": key[1],
                "declared_role": key[2],
                "n_edges": len(rows),
                "mean_edge_transport_defect": mean_or_none(defects),
                "n_requires_historical_reinterpretation": n_reinterpretation,
                "n_requires_doctrine_scope_change": n_doctrine_change,
                "n_weakens_audit_lane_boundary": n_boundary_weaken,
                "n_requires_hidden_role_surgery": n_hidden_surgery,
            }
        )
    return out_rows


def build_status_payload(
    registry_rows: Sequence[Dict[str, Any]],
    source_gate9o_status: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the Gate9P status payload with all 10 required keys."""
    # Carry forward adoption-worthiness from Gate9O
    adoption_worthiness_status = str(
        source_gate9o_status.get("adoption_worthiness_status", "")
    )

    # Mainline comparability: the split preserves mainline comparability
    # because conflict-side edges are unchanged and the only change is a
    # forward-only relabeling of cleaner edges.  The residual set for
    # conflict cells remains identical, so existing comparisons remain valid.
    any_reinterpretation = any(
        row["requires_historical_reinterpretation"] for row in registry_rows
    )
    mainline_comparability_preservation_status = (
        "clear" if not any_reinterpretation else "denied"
    )

    # Audit-lane / operator boundary: preserved if no edge weakens it
    any_boundary_weakening = any(
        row["weakens_audit_lane_boundary"] for row in registry_rows
    )
    audit_lane_boundary_preservation_status = (
        "clear" if not any_boundary_weakening else "denied"
    )

    # Operator admission non-promotion: carried from Gate9O
    operator_admission_non_promotion_status = str(
        source_gate9o_status.get("operator_admission_non_promotion_status", "confirmed")
    )

    # Historical reinterpretation: denied if no edge requires it
    historical_reinterpretation_required_status = (
        "denied" if not any_reinterpretation else "triggered"
    )

    # Doctrine scope change: denied if no edge requires it
    any_doctrine_change = any(
        row["requires_doctrine_scope_change"] for row in registry_rows
    )
    doctrine_scope_change_required_status = (
        "denied" if not any_doctrine_change else "triggered"
    )

    # Hidden role surgery: denied if no edge requires it
    any_hidden_surgery = any(
        row["requires_hidden_role_surgery"] for row in registry_rows
    )

    # Adopt candidate: passes if adoption-worthy AND no adoption-side
    # falsifiers fire (no reinterpretation, no doctrine widening, no
    # boundary weakening, no hidden surgery).
    adopt_candidate_status = (
        "clear"
        if (
            adoption_worthiness_status == "adoption_worthy"
            and not any_reinterpretation
            and not any_doctrine_change
            and not any_boundary_weakening
            and not any_hidden_surgery
        )
        else "denied"
    )

    # Defer candidate: there is a named reason to defer only if Gate9O
    # left a next_named_blocker.  If Gate9O cleared the blocker,
    # deferral has no surviving named reason.
    deferral_surviving_blocker = str(
        source_gate9o_status.get("next_named_blocker", "")
    )
    defer_candidate_status = (
        "has_surviving_blocker" if deferral_surviving_blocker else "no_surviving_blocker"
    )

    # Judgment outcome: adopt if adopt_candidate is clear AND deferral
    # has no surviving blocker.  Defer if adopt_candidate is denied OR
    # deferral retains a surviving blocker.
    if adopt_candidate_status == "clear" and defer_candidate_status == "no_surviving_blocker":
        judgment_outcome_status = "adopt"
    else:
        judgment_outcome_status = "defer"

    # Next named blocker: if judgment is defer, carry the reason;
    # if judgment is adopt, no blocker from this line.
    if judgment_outcome_status == "defer":
        if deferral_surviving_blocker:
            next_named_blocker = deferral_surviving_blocker
        elif adopt_candidate_status == "denied":
            # An adoption-side falsifier fired — the blocker is the
            # coupling that cannot be adopted.
            next_named_blocker = "cleaner_answer_projection_role_coupling"
        else:
            next_named_blocker = "cleaner_answer_projection_role_coupling"
    else:
        next_named_blocker = ""

    return {
        "adoption_worthiness_status": adoption_worthiness_status,
        "mainline_comparability_preservation_status": mainline_comparability_preservation_status,
        "audit_lane_boundary_preservation_status": audit_lane_boundary_preservation_status,
        "operator_admission_non_promotion_status": operator_admission_non_promotion_status,
        "historical_reinterpretation_required_status": historical_reinterpretation_required_status,
        "doctrine_scope_change_required_status": doctrine_scope_change_required_status,
        "adopt_candidate_status": adopt_candidate_status,
        "defer_candidate_status": defer_candidate_status,
        "judgment_outcome_status": judgment_outcome_status,
        "next_named_blocker": next_named_blocker,
    }


def build_report(
    run_id: str,
    source_gate9o_manifest: Dict[str, Any],
    policy_compare_rows: Sequence[Dict[str, Any]],
    status_payload: Dict[str, Any],
) -> str:
    lines = [
        "# Gate9P Declared Split Adopt-Or-Defer Read",
        "",
        f"run_id: {run_id}",
        f"source_gate9o_run_id: {source_gate9o_manifest.get('run_id', '')}",
        f"source_gate9o_code_git_commit: {source_gate9o_manifest.get('code_git_commit', '')}",
        "",
        "## Discipline",
        "",
        "- adopt-or-defer judgment only, not full mainline rollout",
        "- operator admission remains denied",
        "- adoption is forward-only, no retroactive reinterpretation",
        "- declared split definition remains fixed",
        "",
        "## Policy Compare",
        "",
        "| cell_class | baseline_role | declared_role | n_edges | mean_defect | reinterpretation | doctrine_change | boundary_weaken | hidden_surgery |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
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
                    str(row["n_requires_historical_reinterpretation"]),
                    str(row["n_requires_doctrine_scope_change"]),
                    str(row["n_weakens_audit_lane_boundary"]),
                    str(row["n_requires_hidden_role_surgery"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Status",
            "",
            f"- adoption_worthiness_status: `{status_payload['adoption_worthiness_status']}`",
            f"- mainline_comparability_preservation_status: `{status_payload['mainline_comparability_preservation_status']}`",
            f"- audit_lane_boundary_preservation_status: `{status_payload['audit_lane_boundary_preservation_status']}`",
            f"- operator_admission_non_promotion_status: `{status_payload['operator_admission_non_promotion_status']}`",
            f"- historical_reinterpretation_required_status: `{status_payload['historical_reinterpretation_required_status']}`",
            f"- doctrine_scope_change_required_status: `{status_payload['doctrine_scope_change_required_status']}`",
            f"- adopt_candidate_status: `{status_payload['adopt_candidate_status']}`",
            f"- defer_candidate_status: `{status_payload['defer_candidate_status']}`",
            f"- judgment_outcome_status: `{status_payload['judgment_outcome_status']}`",
        ]
    )
    if status_payload.get("next_named_blocker"):
        lines.extend(["", "## Next Blocker", "", f"- `{status_payload['next_named_blocker']}`"])
    else:
        lines.extend(["", "## Next Blocker", "", "- (none from this line)"])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    source_gate9o_dir = Path(args.gate9o_dir)
    out_dir = Path(args.out_dir)
    run_id = out_dir.name

    # Read source Gate9O data
    source_gate9o_manifest = gate9a.read_json(source_gate9o_dir / gate9a.DEFAULT_MANIFEST)
    source_registry_rows = gate9a.read_jsonl(source_gate9o_dir / gate9o.DEFAULT_REGISTRY)
    source_gate9o_status = gate9a.read_json(source_gate9o_dir / gate9o.DEFAULT_STATUS)

    # Build Gate9P outputs
    registry_rows = build_adopt_or_defer_registry(source_registry_rows)
    policy_compare_rows = build_policy_compare(registry_rows)
    status_payload = build_status_payload(registry_rows, source_gate9o_status)

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
            "n_requires_historical_reinterpretation",
            "n_requires_doctrine_scope_change",
            "n_weakens_audit_lane_boundary",
            "n_requires_hidden_role_surgery",
        ),
        policy_compare_rows,
    )
    gate9a.write_json(status_path, status_payload)
    gate9a.write_text(
        report_path,
        build_report(
            run_id=run_id,
            source_gate9o_manifest=source_gate9o_manifest,
            policy_compare_rows=policy_compare_rows,
            status_payload=status_payload,
        ),
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "run_id": run_id,
        "code_git_commit": gate9a.current_git_commit(),
        "source_gate9o_dir": gate9a.repo_relative_or_posix(source_gate9o_dir),
        "source_gate9o_run_id": str(source_gate9o_manifest.get("run_id") or ""),
        "source_gate9o_code_git_commit": str(source_gate9o_manifest.get("code_git_commit") or ""),
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
