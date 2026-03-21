#!/usr/bin/env python3
"""Run a Gate9Q post-adoption integration on Gate9P outputs."""

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import run_gate9a_graph_gauge_consumer as gate9a
import run_gate9p_declared_split_adopt_or_defer_judgment as gate9p


REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "gate9q_post_adoption_integration_v1"
METHOD_ID = "gate9q_post_adoption_integration_v1"

DEFAULT_REGISTRY = "post_adoption_integration_registry.jsonl"
DEFAULT_POLICY_COMPARE = "post_adoption_integration_policy_compare.csv"
DEFAULT_STATUS = "post_adoption_integration_status.json"
DEFAULT_REPORT = "gate9q_post_adoption_integration_read.md"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Gate9Q post-adoption integration for the forward-basis "
            "adopted split using the Gate9P judgment."
        )
    )
    parser.add_argument("--gate9p-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def build_integration_registry(
    source_registry_rows: Sequence[Dict[str, Any]],
    judgment_outcome: str,
) -> List[Dict[str, Any]]:
    """Build Gate9Q integration registry from Gate9P adopt-or-defer registry.

    Each row inherits Gate9P fields and adds integration-relevant annotations.
    No new metrics or roles are introduced.
    """
    registry_rows: List[Dict[str, Any]] = []
    for row in source_registry_rows:
        cell_class = str(row["cell_class"])
        baseline_role = str(row["baseline_residual_role"])
        declared_role = str(row["declared_role"])

        # The adopted forward-basis role: if judgment is adopt, edges use
        # declared_role going forward; if defer, they remain at baseline.
        forward_basis_role = declared_role if judgment_outcome == "adopt" else baseline_role

        # Integration is forward-only: no prior read is changed.
        # The historical role for this edge remains the baseline.
        historical_role = baseline_role

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
                "historical_role": historical_role,
                "forward_basis_role": forward_basis_role,
                "role_changed_by_adoption": historical_role != forward_basis_role,
                "requires_retroactive_reinterpretation": False,
                "implies_operator_admission_open": False,
                "implies_broader_tree_settlement": False,
                "widens_doctrine": False,
            }
        )
    return registry_rows


def build_policy_compare(
    registry_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build integration policy comparison table: historical vs forward-basis."""
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in registry_rows:
        grouped[(
            str(row["cell_class"]),
            str(row["historical_role"]),
            str(row["forward_basis_role"]),
        )].append(row)

    out_rows: List[Dict[str, Any]] = []
    for key in sorted(grouped):
        rows = grouped[key]
        defects = [
            float(r["edge_transport_defect"])
            for r in rows
            if r["edge_transport_defect"] not in (None, "")
        ]
        n_changed = sum(1 for r in rows if r["role_changed_by_adoption"])
        out_rows.append(
            {
                "cell_class": key[0],
                "historical_role": key[1],
                "forward_basis_role": key[2],
                "n_edges": len(rows),
                "mean_edge_transport_defect": mean_or_none(defects),
                "n_role_changed_by_adoption": n_changed,
            }
        )
    return out_rows


def build_status_payload(
    registry_rows: Sequence[Dict[str, Any]],
    source_gate9p_status: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the Gate9Q status payload with all 10 required keys."""
    judgment_outcome = str(source_gate9p_status.get("judgment_outcome_status", ""))

    # Forward-basis adoption: adopted if Gate9P judged adopt
    forward_basis_adoption_status = (
        "adopted" if judgment_outcome == "adopt" else "deferred"
    )

    # Mainline memory update: ready if adoption is adopted and no
    # falsifiers fire
    any_retroactive = any(row["requires_retroactive_reinterpretation"] for row in registry_rows)
    any_operator_open = any(row["implies_operator_admission_open"] for row in registry_rows)
    any_broader_settlement = any(row["implies_broader_tree_settlement"] for row in registry_rows)
    any_doctrine_widen = any(row["widens_doctrine"] for row in registry_rows)

    # Operator admission: must remain denied
    operator_admission_still_denied_status = (
        "confirmed" if not any_operator_open else "violated"
    )

    # Retroactive reinterpretation: must remain forbidden
    retroactive_reinterpretation_forbidden_status = (
        "confirmed" if not any_retroactive else "violated"
    )

    # Broader tree settlement: must remain explicitly unresolved
    broader_tree_settlement_unresolved_status = (
        "confirmed" if not any_broader_settlement else "violated"
    )

    # Historical lane preservation: prior reads are not rewritten
    historical_lane_preservation_status = (
        "clear" if not any_retroactive else "denied"
    )

    # Integration scope preservation: integration does not widen doctrine
    integration_scope_preservation_status = (
        "clear" if not any_doctrine_widen else "denied"
    )

    # Can we state what remains unresolved?  Yes — operator admission
    # and broader tree settlement remain explicitly unresolved.
    can_state_unresolved = (
        operator_admission_still_denied_status == "confirmed"
        and broader_tree_settlement_unresolved_status == "confirmed"
    )

    # Post-adoption integration readiness: ready if all guard checks pass
    post_adoption_integration_readiness_status = (
        "ready"
        if (
            forward_basis_adoption_status == "adopted"
            and not any_retroactive
            and not any_operator_open
            and not any_broader_settlement
            and not any_doctrine_widen
            and can_state_unresolved
        )
        else "not_ready"
    )

    # Mainline memory update
    mainline_memory_update_status = (
        "updated"
        if post_adoption_integration_readiness_status == "ready"
        else "deferred"
    )

    # Integration outcome
    integration_outcome_status = (
        "integrated"
        if post_adoption_integration_readiness_status == "ready"
        else "blocked"
    )

    # Next named blocker: empty if integrated, otherwise carry reason
    if integration_outcome_status == "integrated":
        next_named_blocker = ""
    else:
        if forward_basis_adoption_status == "deferred":
            next_named_blocker = "cleaner_answer_projection_role_coupling"
        elif any_retroactive:
            next_named_blocker = "retroactive_reinterpretation_required"
        elif any_operator_open:
            next_named_blocker = "operator_admission_leaked"
        elif any_broader_settlement:
            next_named_blocker = "broader_tree_settlement_leaked"
        elif any_doctrine_widen:
            next_named_blocker = "doctrine_scope_widened"
        else:
            next_named_blocker = "integration_readiness_unknown"

    return {
        "forward_basis_adoption_status": forward_basis_adoption_status,
        "mainline_memory_update_status": mainline_memory_update_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "retroactive_reinterpretation_forbidden_status": retroactive_reinterpretation_forbidden_status,
        "broader_tree_settlement_unresolved_status": broader_tree_settlement_unresolved_status,
        "historical_lane_preservation_status": historical_lane_preservation_status,
        "integration_scope_preservation_status": integration_scope_preservation_status,
        "post_adoption_integration_readiness_status": post_adoption_integration_readiness_status,
        "integration_outcome_status": integration_outcome_status,
        "next_named_blocker": next_named_blocker,
    }


def build_report(
    run_id: str,
    source_gate9p_manifest: Dict[str, Any],
    policy_compare_rows: Sequence[Dict[str, Any]],
    status_payload: Dict[str, Any],
) -> str:
    lines = [
        "# Gate9Q Post-Adoption Integration Read",
        "",
        f"run_id: {run_id}",
        f"source_gate9p_run_id: {source_gate9p_manifest.get('run_id', '')}",
        f"source_gate9p_code_git_commit: {source_gate9p_manifest.get('code_git_commit', '')}",
        "",
        "## Discipline",
        "",
        "- post-adoption integration only, not operator opening",
        "- forward-basis only, no retroactive reinterpretation",
        "- broader trusted-tree settlement remains explicitly unresolved",
        "- operator admission remains denied",
        "",
        "## Integration Summary",
        "",
        "| cell_class | historical_role | forward_basis_role | n_edges | mean_defect | n_role_changed |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in policy_compare_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["cell_class"]),
                    str(row["historical_role"]),
                    str(row["forward_basis_role"]),
                    str(row["n_edges"]),
                    ""
                    if row["mean_edge_transport_defect"] in (None, "")
                    else f"{float(row['mean_edge_transport_defect']):.6f}",
                    str(row["n_role_changed_by_adoption"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Status",
            "",
            f"- forward_basis_adoption_status: `{status_payload['forward_basis_adoption_status']}`",
            f"- mainline_memory_update_status: `{status_payload['mainline_memory_update_status']}`",
            f"- operator_admission_still_denied_status: `{status_payload['operator_admission_still_denied_status']}`",
            f"- retroactive_reinterpretation_forbidden_status: `{status_payload['retroactive_reinterpretation_forbidden_status']}`",
            f"- broader_tree_settlement_unresolved_status: `{status_payload['broader_tree_settlement_unresolved_status']}`",
            f"- historical_lane_preservation_status: `{status_payload['historical_lane_preservation_status']}`",
            f"- integration_scope_preservation_status: `{status_payload['integration_scope_preservation_status']}`",
            f"- post_adoption_integration_readiness_status: `{status_payload['post_adoption_integration_readiness_status']}`",
            f"- integration_outcome_status: `{status_payload['integration_outcome_status']}`",
        ]
    )

    lines.extend(
        [
            "",
            "## What Remains Unresolved",
            "",
            "- operator admission: denied",
            "- broader trusted-tree settlement: unresolved",
            "- retroactive reinterpretation: forbidden",
        ]
    )

    if status_payload.get("next_named_blocker"):
        lines.extend(["", "## Next Blocker", "", f"- `{status_payload['next_named_blocker']}`"])
    else:
        lines.extend(["", "## Next Blocker", "", "- (none from this line)"])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    source_gate9p_dir = Path(args.gate9p_dir)
    out_dir = Path(args.out_dir)
    run_id = out_dir.name

    # Read source Gate9P data
    source_gate9p_manifest = gate9a.read_json(source_gate9p_dir / gate9a.DEFAULT_MANIFEST)
    source_registry_rows = gate9a.read_jsonl(source_gate9p_dir / gate9p.DEFAULT_REGISTRY)
    source_gate9p_status = gate9a.read_json(source_gate9p_dir / gate9p.DEFAULT_STATUS)

    judgment_outcome = str(source_gate9p_status.get("judgment_outcome_status", ""))

    # Build Gate9Q outputs
    registry_rows = build_integration_registry(source_registry_rows, judgment_outcome)
    policy_compare_rows = build_policy_compare(registry_rows)
    status_payload = build_status_payload(registry_rows, source_gate9p_status)

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
            "historical_role",
            "forward_basis_role",
            "n_edges",
            "mean_edge_transport_defect",
            "n_role_changed_by_adoption",
        ),
        policy_compare_rows,
    )
    gate9a.write_json(status_path, status_payload)
    gate9a.write_text(
        report_path,
        build_report(
            run_id=run_id,
            source_gate9p_manifest=source_gate9p_manifest,
            policy_compare_rows=policy_compare_rows,
            status_payload=status_payload,
        ),
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "run_id": run_id,
        "code_git_commit": gate9a.current_git_commit(),
        "source_gate9p_dir": gate9a.repo_relative_or_posix(source_gate9p_dir),
        "source_gate9p_run_id": str(source_gate9p_manifest.get("run_id") or ""),
        "source_gate9p_code_git_commit": str(source_gate9p_manifest.get("code_git_commit") or ""),
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
