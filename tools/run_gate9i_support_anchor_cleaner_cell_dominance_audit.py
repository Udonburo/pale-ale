#!/usr/bin/env python3
"""Run a Gate9I support-anchor cleaner-cell dominance audit on Gate9H outputs."""

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import run_gate9a_graph_gauge_consumer as gate9a
import run_gate9b_small_cycle_holonomy_study as gate9b
import run_gate9h_anchor_coverage_gap_redesign_audit as gate9h


REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "gate9i_support_anchor_cleaner_cell_dominance_audit_v1"
METHOD_ID = "gate9i_support_anchor_cleaner_cell_dominance_audit_v1"
DEFAULT_REGISTRY = "support_anchor_cleaner_dominance_registry.jsonl"
DEFAULT_PAIR_SUMMARY = "support_anchor_quietness_pairs.csv"
DEFAULT_CELL_SUMMARY = "support_anchor_cleaner_dominance_by_cell.csv"
DEFAULT_STATUS = "support_anchor_cleaner_dominance_status.json"
DEFAULT_REPORT = "gate9i_support_anchor_cleaner_dominance_read.md"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Gate9I audit over support-anchor cleaner-cell dominance on the "
            "Gate9H redesign line."
        )
    )
    parser.add_argument("--gate9h-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive_source_context(source_gate9h_dir: Path) -> Tuple[Dict[str, Any], Path, Dict[str, Any], Path]:
    source_gate9h_manifest = gate9a.read_json(source_gate9h_dir / gate9a.DEFAULT_MANIFEST)
    source_gate9g_dir = REPO_ROOT / str(source_gate9h_manifest["source_gate9g_dir"])
    source_gate9g_manifest = gate9a.read_json(source_gate9g_dir / gate9a.DEFAULT_MANIFEST)
    source_gate9a_dir = REPO_ROOT / str(source_gate9g_manifest["source_gate9a_dir"])
    source_gate9a_manifest = gate9a.read_json(source_gate9a_dir / gate9a.DEFAULT_MANIFEST)
    source_gate8_dir = REPO_ROOT / str(source_gate9a_manifest["source_gate8_execution_dir"])
    return source_gate9h_manifest, source_gate9g_dir, source_gate9g_manifest, source_gate8_dir


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def build_registry_rows(
    gate9h_rows: Sequence[Dict[str, Any]],
    sample_registry_by_benchmark: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    registry_rows: List[Dict[str, Any]] = []
    for row in gate9h_rows:
        if str(row["anchor_kind"]) != "support":
            continue
        sample_registry_row = sample_registry_by_benchmark.get(str(row["benchmark_sample_id"]), {})
        registry_rows.append(
            {
                "closure_id": str(row["closure_id"]),
                "benchmark_sample_id": str(row["benchmark_sample_id"]),
                "execution_sample_id": int(row["execution_sample_id"]),
                "cell_id": str(row["cell_id"]),
                "world_id": str(row["world_id"]),
                "world_type": str(row["world_type"]),
                "answer_target_type": str(row["answer_target_type"]),
                "rendering_family_id": str(row["rendering_family_id"]),
                "candidate_status": str(row["candidate_status"]),
                "coverage_gap_abs": row.get("coverage_gap_abs"),
                "quietness_pair_id": str(sample_registry_row.get("quietness_pair_id") or ""),
                "is_surface_noise_only": bool(sample_registry_row.get("is_surface_noise_only", False)),
                "is_conflict_intended": bool(sample_registry_row.get("is_conflict_intended", False)),
            }
        )
    return registry_rows


def summarize_by_cell(registry_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in registry_rows:
        grouped[str(row["cell_id"])].append(row)

    out_rows: List[Dict[str, Any]] = []
    for cell_id in sorted(grouped):
        rows = grouped[cell_id]
        values = [
            float(row["coverage_gap_abs"])
            for row in rows
            if row["candidate_status"] == "nontrivial_gap_candidate" and row["coverage_gap_abs"] not in (None, "")
        ]
        out_rows.append(
            {
                "cell_id": cell_id,
                "n_rows": len(rows),
                "n_nontrivial_rows": len(values),
                "mean_coverage_gap_abs": mean_or_none(values),
                "max_coverage_gap_abs": None if not values else float(max(values)),
            }
        )
    return out_rows


def build_quietness_pair_rows(
    registry_rows: Sequence[Dict[str, Any]],
    quietness_pair_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    support_row_by_benchmark = {
        str(row["benchmark_sample_id"]): row for row in registry_rows
    }
    pair_rows: List[Dict[str, Any]] = []
    for pair in quietness_pair_rows:
        clean_id = str(pair["clean_benchmark_sample_id"])
        noisy_id = str(pair["surface_noisy_benchmark_sample_id"])
        clean_row = support_row_by_benchmark.get(clean_id)
        noisy_row = support_row_by_benchmark.get(noisy_id)
        pair_status = "available"
        if clean_row is None or noisy_row is None:
            pair_status = "missing_support_row"
        clean_gap = None if clean_row is None else clean_row["coverage_gap_abs"]
        noisy_gap = None if noisy_row is None else noisy_row["coverage_gap_abs"]
        noisy_minus_clean = None
        abs_delta = None
        if pair_status == "available" and clean_gap not in (None, "") and noisy_gap not in (None, ""):
            noisy_minus_clean = float(noisy_gap) - float(clean_gap)
            abs_delta = abs(noisy_minus_clean)
        pair_rows.append(
            {
                "quietness_pair_id": str(pair["quietness_pair_id"]),
                "world_id": str(pair["world_id"]),
                "world_type": str(pair["world_type"]),
                "clean_benchmark_sample_id": clean_id,
                "surface_noisy_benchmark_sample_id": noisy_id,
                "pair_status": pair_status,
                "clean_gap_abs": clean_gap,
                "surface_noisy_gap_abs": noisy_gap,
                "surface_noisy_minus_clean_gap": noisy_minus_clean,
                "abs_pair_gap_delta": abs_delta,
            }
        )
    return pair_rows


def mean_gap_by_cell(registry_rows: Sequence[Dict[str, Any]], cell_id: str) -> Optional[float]:
    values = [
        float(row["coverage_gap_abs"])
        for row in registry_rows
        if str(row["cell_id"]) == cell_id
        and str(row["candidate_status"]) == "nontrivial_gap_candidate"
        and row["coverage_gap_abs"] not in (None, "")
    ]
    return mean_or_none(values)


def build_status_payload(
    registry_rows: Sequence[Dict[str, Any]],
    pair_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    clean_mean = mean_gap_by_cell(registry_rows, "clean_support")
    noisy_mean = mean_gap_by_cell(registry_rows, "surface_noisy_clean")
    direct_mean = mean_gap_by_cell(registry_rows, "direct_contradiction")
    distributed_mean = mean_gap_by_cell(registry_rows, "distributed_incompatibility")
    cleaner_values = [value for value in (clean_mean, noisy_mean) if value is not None]
    conflict_values = [value for value in (direct_mean, distributed_mean) if value is not None]
    support_anchor_cleaner_dominance_status = "insufficient_data"
    if cleaner_values and conflict_values:
        support_anchor_cleaner_dominance_status = (
            "triggered" if max(cleaner_values) >= max(conflict_values) else "clear"
        )
    surface_noisy_corroboration_status = "insufficient_data"
    if noisy_mean is not None and conflict_values:
        surface_noisy_corroboration_status = (
            "corroborated" if noisy_mean > max(conflict_values) else "not_corroborated"
        )
    distributed_underactivation_status = "insufficient_data"
    if distributed_mean is not None and direct_mean is not None:
        distributed_underactivation_status = (
            "triggered" if distributed_mean < direct_mean else "clear"
        )
    pair_deltas = [
        float(row["abs_pair_gap_delta"])
        for row in pair_rows
        if row["abs_pair_gap_delta"] not in (None, "")
    ]
    return {
        "support_anchor_cleaner_dominance_status": support_anchor_cleaner_dominance_status,
        "surface_noisy_corroboration_status": surface_noisy_corroboration_status,
        "distributed_underactivation_status": distributed_underactivation_status,
        "dominance_explained_as_quietness_noise_status": (
            "denied" if surface_noisy_corroboration_status == "corroborated" else "not_yet_denied"
        ),
        "mean_abs_quietness_pair_gap_delta": mean_or_none(pair_deltas),
        "support_clean_mean_gap": clean_mean,
        "support_surface_noisy_mean_gap": noisy_mean,
        "support_direct_mean_gap": direct_mean,
        "support_distributed_mean_gap": distributed_mean,
        "next_named_subblocker": (
            "distributed_underactivation"
            if distributed_underactivation_status == "triggered"
            else ""
        ),
    }


def build_report(
    run_id: str,
    source_manifest: Dict[str, Any],
    cell_summary_rows: Sequence[Dict[str, Any]],
    pair_rows: Sequence[Dict[str, Any]],
    status_payload: Dict[str, Any],
) -> str:
    lines = [
        "# Gate9I Support-Anchor Cleaner-Cell Dominance Read",
        "",
        f"run_id: {run_id}",
        f"source_gate9h_run_id: {source_manifest.get('run_id', '')}",
        f"source_gate9h_code_git_commit: {source_manifest.get('code_git_commit', '')}",
        "",
        "## Discipline",
        "",
        "- support-anchor redesign candidate only",
        "- no new metric family beyond `anchor_coverage_gap_abs_v1`",
        "- the question is why cleaner-cell dominance still triggers, not how to rescue it yet",
        "",
        "## Support Means By Cell",
        "",
        "| cell_id | n_rows | n_nontrivial_rows | mean_coverage_gap_abs | max_coverage_gap_abs |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in cell_summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["cell_id"]),
                    str(row["n_rows"]),
                    str(row["n_nontrivial_rows"]),
                    "" if row["mean_coverage_gap_abs"] in (None, "") else f"{float(row['mean_coverage_gap_abs']):.6f}",
                    "" if row["max_coverage_gap_abs"] in (None, "") else f"{float(row['max_coverage_gap_abs']):.6f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Quietness Pairs",
            "",
            "| quietness_pair_id | pair_status | clean_gap_abs | surface_noisy_gap_abs | abs_pair_gap_delta |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in pair_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["quietness_pair_id"]),
                    str(row["pair_status"]),
                    "" if row["clean_gap_abs"] in (None, "") else f"{float(row['clean_gap_abs']):.6f}",
                    "" if row["surface_noisy_gap_abs"] in (None, "") else f"{float(row['surface_noisy_gap_abs']):.6f}",
                    "" if row["abs_pair_gap_delta"] in (None, "") else f"{float(row['abs_pair_gap_delta']):.6f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Status",
            "",
            f"- support_anchor_cleaner_dominance_status: `{status_payload['support_anchor_cleaner_dominance_status']}`",
            f"- surface_noisy_corroboration_status: `{status_payload['surface_noisy_corroboration_status']}`",
            f"- distributed_underactivation_status: `{status_payload['distributed_underactivation_status']}`",
            f"- dominance_explained_as_quietness_noise_status: `{status_payload['dominance_explained_as_quietness_noise_status']}`",
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
    source_gate9h_dir = Path(args.gate9h_dir)
    out_dir = Path(args.out_dir)
    run_id = out_dir.name

    source_manifest, _source_gate9g_dir, _source_gate9g_manifest, source_gate8_dir = derive_source_context(
        source_gate9h_dir
    )
    gate9h_rows = gate9a.read_jsonl(source_gate9h_dir / gate9h.DEFAULT_REGISTRY)
    sample_registry_rows = gate9a.read_jsonl(source_gate8_dir / "sample_registry.jsonl")
    quietness_pair_rows = gate9a.read_jsonl(source_gate8_dir / "quietness_pairs.jsonl")
    sample_registry_by_benchmark = {
        str(row["benchmark_sample_id"]): row for row in sample_registry_rows
    }

    registry_rows = build_registry_rows(gate9h_rows, sample_registry_by_benchmark)
    cell_summary_rows = summarize_by_cell(registry_rows)
    pair_rows = build_quietness_pair_rows(registry_rows, quietness_pair_rows)
    status_payload = build_status_payload(registry_rows, pair_rows)

    registry_path = out_dir / DEFAULT_REGISTRY
    pair_summary_path = out_dir / DEFAULT_PAIR_SUMMARY
    cell_summary_path = out_dir / DEFAULT_CELL_SUMMARY
    status_path = out_dir / DEFAULT_STATUS
    report_path = out_dir / DEFAULT_REPORT
    manifest_path = out_dir / DEFAULT_MANIFEST
    checksums_path = out_dir / DEFAULT_CHECKSUMS

    gate9a.write_jsonl(registry_path, registry_rows)
    gate9a.write_csv(
        pair_summary_path,
        (
            "quietness_pair_id",
            "pair_status",
            "clean_gap_abs",
            "surface_noisy_gap_abs",
            "surface_noisy_minus_clean_gap",
            "abs_pair_gap_delta",
        ),
        pair_rows,
    )
    gate9a.write_csv(
        cell_summary_path,
        (
            "cell_id",
            "n_rows",
            "n_nontrivial_rows",
            "mean_coverage_gap_abs",
            "max_coverage_gap_abs",
        ),
        cell_summary_rows,
    )
    gate9a.write_json(status_path, status_payload)
    gate9a.write_text(
        report_path,
        build_report(
            run_id=run_id,
            source_manifest=source_manifest,
            cell_summary_rows=cell_summary_rows,
            pair_rows=pair_rows,
            status_payload=status_payload,
        ),
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "run_id": run_id,
        "code_git_commit": gate9a.current_git_commit(),
        "source_gate9h_dir": gate9a.repo_relative_or_posix(source_gate9h_dir),
        "source_gate9h_run_id": str(source_manifest.get("run_id") or ""),
        "source_gate9h_code_git_commit": str(source_manifest.get("code_git_commit") or ""),
        "paths": {
            DEFAULT_REGISTRY: gate9a.repo_relative_or_posix(registry_path),
            DEFAULT_PAIR_SUMMARY: gate9a.repo_relative_or_posix(pair_summary_path),
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
            DEFAULT_PAIR_SUMMARY: sha256_file(pair_summary_path),
            DEFAULT_CELL_SUMMARY: sha256_file(cell_summary_path),
            DEFAULT_STATUS: sha256_file(status_path),
            DEFAULT_REPORT: sha256_file(report_path),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
