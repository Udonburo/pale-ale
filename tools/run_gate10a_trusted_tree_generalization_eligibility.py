#!/usr/bin/env python3
"""Run a Gate10A trusted-tree generalization eligibility audit on Gate9Q outputs."""

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import run_gate9a_graph_gauge_consumer as gate9a
import run_gate9q_post_adoption_integration as gate9q


REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "gate10a_trusted_tree_generalization_eligibility_v1"
METHOD_ID = "gate10a_trusted_tree_generalization_eligibility_v1"

DEFAULT_REGISTRY = "trusted_tree_generalization_eligibility_registry.jsonl"
DEFAULT_POLICY_COMPARE = "trusted_tree_generalization_eligibility_policy_compare.csv"
DEFAULT_STATUS = "trusted_tree_generalization_eligibility_status.json"
DEFAULT_REPORT = "gate10a_trusted_tree_generalization_eligibility_read.md"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Gate10A eligibility audit for broader trusted-tree settlement "
            "entry using the Gate9Q integrated bundle."
        )
    )
    parser.add_argument("--gate9q-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def build_eligibility_registry(
    source_registry_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    registry_rows: List[Dict[str, Any]] = []
    for row in source_registry_rows:
        historical_role = str(row["historical_role"])
        forward_basis_role = str(row["forward_basis_role"])
        role_changed = bool(row["role_changed_by_adoption"])

        broader_candidate_class = (
            "adopted_split_baseline"
            if role_changed
            else "broader_candidate_opening_lane"
        )
        forward_basis_adoption_preserved = (
            (role_changed and forward_basis_role == "closure_return_leg_auxiliary")
            or ((not role_changed) and forward_basis_role == historical_role)
        )

        registry_rows.append(
            {
                "edge_id": str(row["edge_id"]),
                "execution_sample_id": int(row["execution_sample_id"]),
                "benchmark_sample_id": str(row["benchmark_sample_id"]),
                "cell_id": str(row["cell_id"]),
                "cell_class": str(row["cell_class"]),
                "world_id": str(row["world_id"]),
                "world_type": str(row["world_type"]),
                "answer_target_type": str(row["answer_target_type"]),
                "edge_transport_defect": row.get("edge_transport_defect"),
                "historical_role": historical_role,
                "forward_basis_role": forward_basis_role,
                "role_changed_by_adoption": role_changed,
                "broader_candidate_class": broader_candidate_class,
                "forward_basis_adoption_preserved": forward_basis_adoption_preserved,
                "requires_retroactive_reinterpretation": bool(
                    row["requires_retroactive_reinterpretation"]
                ),
                "implies_operator_admission_open": bool(
                    row["implies_operator_admission_open"]
                ),
                "implies_broader_tree_settlement": bool(
                    row["implies_broader_tree_settlement"]
                ),
                "widens_doctrine": bool(row["widens_doctrine"]),
            }
        )
    return registry_rows


def build_policy_compare(
    registry_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in registry_rows:
        grouped[(str(row["cell_class"]), str(row["broader_candidate_class"]))].append(row)

    out_rows: List[Dict[str, Any]] = []
    for key in sorted(grouped):
        rows = grouped[key]
        defects = [
            float(r["edge_transport_defect"])
            for r in rows
            if r["edge_transport_defect"] not in (None, "")
        ]
        out_rows.append(
            {
                "cell_class": key[0],
                "broader_candidate_class": key[1],
                "n_edges": len(rows),
                "mean_edge_transport_defect": mean_or_none(defects),
                "n_role_changed_by_adoption": sum(
                    1 for r in rows if r["role_changed_by_adoption"]
                ),
                "n_forward_basis_preserved": sum(
                    1 for r in rows if r["forward_basis_adoption_preserved"]
                ),
            }
        )
    return out_rows


def build_status_payload(
    registry_rows: Sequence[Dict[str, Any]],
    source_gate9q_status: Dict[str, Any],
) -> Dict[str, Any]:
    source_integrated = (
        str(source_gate9q_status.get("forward_basis_adoption_status", "")) == "adopted"
        and str(source_gate9q_status.get("mainline_memory_update_status", "")) == "updated"
        and str(source_gate9q_status.get("integration_outcome_status", "")) == "integrated"
    )
    integrated_baseline_source_status = "clear" if source_integrated else "denied"

    any_forward_basis_break = any(
        not bool(row["forward_basis_adoption_preserved"]) for row in registry_rows
    )
    forward_basis_adoption_preservation_status = (
        "clear" if source_integrated and not any_forward_basis_break else "denied"
    )

    any_retroactive = any(
        bool(row["requires_retroactive_reinterpretation"]) for row in registry_rows
    )
    non_retroactive_memory_preservation_status = (
        "clear"
        if (
            str(
                source_gate9q_status.get(
                    "retroactive_reinterpretation_forbidden_status", ""
                )
            )
            == "confirmed"
            and not any_retroactive
        )
        else "denied"
    )

    any_operator_open = any(
        bool(row["implies_operator_admission_open"]) for row in registry_rows
    )
    operator_adjacent_rescue_pressure_status = (
        "clear"
        if (
            str(source_gate9q_status.get("operator_admission_still_denied_status", ""))
            == "confirmed"
            and not any_operator_open
        )
        else "triggered"
    )

    any_semantics_broadening = any(bool(row["widens_doctrine"]) for row in registry_rows)
    trusted_tree_semantics_broadening_pressure_status = (
        "clear"
        if (
            str(source_gate9q_status.get("integration_scope_preservation_status", ""))
            == "clear"
            and not any_semantics_broadening
        )
        else "triggered"
    )

    any_broader_settlement = any(
        bool(row["implies_broader_tree_settlement"]) for row in registry_rows
    )
    broader_tree_settlement_non_promotion_status = (
        "clear"
        if (
            str(
                source_gate9q_status.get(
                    "broader_tree_settlement_unresolved_status", ""
                )
            )
            == "confirmed"
            and not any_broader_settlement
        )
        else "violated"
    )

    operator_admission_still_denied_status = (
        "confirmed" if not any_operator_open else "violated"
    )

    broader_candidate_eligibility_status = (
        "eligible"
        if (
            integrated_baseline_source_status == "clear"
            and forward_basis_adoption_preservation_status == "clear"
            and non_retroactive_memory_preservation_status == "clear"
            and operator_adjacent_rescue_pressure_status == "clear"
            and trusted_tree_semantics_broadening_pressure_status == "clear"
            and broader_tree_settlement_non_promotion_status == "clear"
        )
        else "not_yet_eligible"
    )

    settlement_comparison_permission_status = (
        "permitted"
        if broader_candidate_eligibility_status == "eligible"
        else "withheld"
    )

    if broader_candidate_eligibility_status == "eligible":
        next_named_blocker = ""
    elif integrated_baseline_source_status != "clear":
        next_named_blocker = "gate9q_integrated_baseline_missing"
    elif forward_basis_adoption_preservation_status != "clear":
        next_named_blocker = "forward_basis_adoption_not_preserved"
    elif non_retroactive_memory_preservation_status != "clear":
        next_named_blocker = "retroactive_reinterpretation_pressure"
    elif operator_adjacent_rescue_pressure_status != "clear":
        next_named_blocker = "operator_adjacent_rescue_pressure"
    elif trusted_tree_semantics_broadening_pressure_status != "clear":
        next_named_blocker = "silent_tree_semantics_broadening"
    elif broader_tree_settlement_non_promotion_status != "clear":
        next_named_blocker = "broader_tree_settlement_leak"
    else:
        next_named_blocker = "eligibility_unknown"

    return {
        "integrated_baseline_source_status": integrated_baseline_source_status,
        "forward_basis_adoption_preservation_status": forward_basis_adoption_preservation_status,
        "non_retroactive_memory_preservation_status": non_retroactive_memory_preservation_status,
        "operator_adjacent_rescue_pressure_status": operator_adjacent_rescue_pressure_status,
        "trusted_tree_semantics_broadening_pressure_status": trusted_tree_semantics_broadening_pressure_status,
        "broader_tree_settlement_non_promotion_status": broader_tree_settlement_non_promotion_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "broader_candidate_eligibility_status": broader_candidate_eligibility_status,
        "settlement_comparison_permission_status": settlement_comparison_permission_status,
        "next_named_blocker": next_named_blocker,
    }


def build_report(
    run_id: str,
    source_gate9q_manifest: Dict[str, Any],
    policy_compare_rows: Sequence[Dict[str, Any]],
    status_payload: Dict[str, Any],
) -> str:
    lines = [
        "# Gate10A Trusted-Tree Generalization Eligibility Read",
        "",
        f"run_id: {run_id}",
        f"source_gate9q_run_id: {source_gate9q_manifest.get('run_id', '')}",
        f"source_gate9q_code_git_commit: {source_gate9q_manifest.get('code_git_commit', '')}",
        "",
        "## Discipline",
        "",
        "- eligibility only, not settlement",
        "- forward-basis split remains baseline, not relitigation target",
        "- operator admission remains denied",
        "- broader trusted-tree settlement remains non-promoted at this stage",
        "",
        "## Eligibility Summary",
        "",
        "| cell_class | broader_candidate_class | n_edges | mean_defect | n_role_changed | n_forward_basis_preserved |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in policy_compare_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["cell_class"]),
                    str(row["broader_candidate_class"]),
                    str(row["n_edges"]),
                    ""
                    if row["mean_edge_transport_defect"] in (None, "")
                    else f"{float(row['mean_edge_transport_defect']):.6f}",
                    str(row["n_role_changed_by_adoption"]),
                    str(row["n_forward_basis_preserved"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Status",
            "",
            f"- integrated_baseline_source_status: `{status_payload['integrated_baseline_source_status']}`",
            f"- forward_basis_adoption_preservation_status: `{status_payload['forward_basis_adoption_preservation_status']}`",
            f"- non_retroactive_memory_preservation_status: `{status_payload['non_retroactive_memory_preservation_status']}`",
            f"- operator_adjacent_rescue_pressure_status: `{status_payload['operator_adjacent_rescue_pressure_status']}`",
            f"- trusted_tree_semantics_broadening_pressure_status: `{status_payload['trusted_tree_semantics_broadening_pressure_status']}`",
            f"- broader_tree_settlement_non_promotion_status: `{status_payload['broader_tree_settlement_non_promotion_status']}`",
            f"- operator_admission_still_denied_status: `{status_payload['operator_admission_still_denied_status']}`",
            f"- broader_candidate_eligibility_status: `{status_payload['broader_candidate_eligibility_status']}`",
            f"- settlement_comparison_permission_status: `{status_payload['settlement_comparison_permission_status']}`",
        ]
    )

    if status_payload["broader_candidate_eligibility_status"] == "eligible":
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- broader trusted-tree candidates may now enter settlement comparison",
                "- this does not grant settlement, operator reopening, or retroactive reinterpretation",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Judgment",
                "",
                "- broader trusted-tree candidates are not yet eligible to enter settlement comparison",
                f"- next_named_blocker: `{status_payload['next_named_blocker']}`",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    source_gate9q_dir = Path(args.gate9q_dir)
    out_dir = Path(args.out_dir)
    run_id = out_dir.name

    source_gate9q_manifest = gate9a.read_json(source_gate9q_dir / gate9a.DEFAULT_MANIFEST)
    source_registry_rows = gate9a.read_jsonl(source_gate9q_dir / gate9q.DEFAULT_REGISTRY)
    source_gate9q_status = gate9a.read_json(source_gate9q_dir / gate9q.DEFAULT_STATUS)

    registry_rows = build_eligibility_registry(source_registry_rows)
    policy_compare_rows = build_policy_compare(registry_rows)
    status_payload = build_status_payload(registry_rows, source_gate9q_status)

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
            "broader_candidate_class",
            "n_edges",
            "mean_edge_transport_defect",
            "n_role_changed_by_adoption",
            "n_forward_basis_preserved",
        ),
        policy_compare_rows,
    )
    gate9a.write_json(status_path, status_payload)
    gate9a.write_text(
        report_path,
        build_report(
            run_id=run_id,
            source_gate9q_manifest=source_gate9q_manifest,
            policy_compare_rows=policy_compare_rows,
            status_payload=status_payload,
        ),
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "run_id": run_id,
        "code_git_commit": gate9a.current_git_commit(),
        "source_gate9q_dir": gate9a.repo_relative_or_posix(source_gate9q_dir),
        "source_gate9q_run_id": str(source_gate9q_manifest.get("run_id") or ""),
        "source_gate9q_code_git_commit": str(
            source_gate9q_manifest.get("code_git_commit") or ""
        ),
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
