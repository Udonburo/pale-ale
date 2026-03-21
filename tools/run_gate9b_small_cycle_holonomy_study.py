#!/usr/bin/env python3
"""Run a narrow Gate9B small-cycle holonomy study on Gate9A outputs."""

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import run_gate9a_graph_gauge_consumer as gate9a


REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "gate9b_small_cycle_holonomy_study_v1"
METHOD_ID = "gate9b_small_cycle_holonomy_study_v1"
ALLOWED_CYCLE_TYPES = (
    "support_answer_terminal_token_cycle",
    "conflict_answer_terminal_token_cycle",
)
CLEANER_CELL_IDS = ("clean_support", "surface_noisy_clean")
CONFLICT_CELL_IDS = ("direct_contradiction", "distributed_incompatibility")
FALSIFIER_IDS = (
    "cleaner_cell_dominance",
    "direct_contradiction_escape",
    "distributed_incompatibility_failure",
    "missing_anchor_collapse",
)

DEFAULT_CYCLE_FOCUS = "cycle_focus_registry.jsonl"
DEFAULT_PAIR_REGISTRY = "quietness_cycle_pair_registry.jsonl"
DEFAULT_CELL_SUMMARY = "cycle_motif_by_cell.csv"
DEFAULT_PAIR_SUMMARY = "quietness_cycle_pairs_by_type.csv"
DEFAULT_FALSIFIER_STATUS = "falsifier_status.json"
DEFAULT_REPORT = "gate9b_holonomy_read.md"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a narrow Gate9B small-cycle holonomy comparison layer over an "
            "existing Gate9A execution bundle without reopening the graph-gauge law."
        )
    )
    parser.add_argument("--gate9a-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def cell_bucket(cell_id: str) -> str:
    if cell_id in CLEANER_CELL_IDS:
        return "cleaner_cell"
    if cell_id in CONFLICT_CELL_IDS:
        return "conflict_cell"
    return "other"


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def median_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.median(values))


def build_cycle_focus_rows(
    cycle_rows: Sequence[Dict[str, Any]],
    sample_registry_by_benchmark: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    focus_rows: List[Dict[str, Any]] = []
    for row in cycle_rows:
        cycle_type = str(row.get("cycle_type") or "")
        if cycle_type not in ALLOWED_CYCLE_TYPES:
            continue
        registry_row = sample_registry_by_benchmark.get(str(row["benchmark_sample_id"]), {})
        focus_rows.append(
            {
                "cycle_id": str(row["cycle_id"]),
                "cycle_type": cycle_type,
                "execution_sample_id": int(row["execution_sample_id"]),
                "benchmark_sample_id": str(row["benchmark_sample_id"]),
                "cell_id": str(row["cell_id"]),
                "cell_bucket": cell_bucket(str(row["cell_id"])),
                "world_id": str(row["world_id"]),
                "world_type": str(row["world_type"]),
                "answer_target_type": str(row["answer_target_type"]),
                "rendering_family_id": str(row["rendering_family_id"]),
                "cycle_outcome": str(row["cycle_outcome"]),
                "holonomy_defect": row.get("holonomy_defect"),
                "holonomy_trace": row.get("holonomy_trace"),
                "quietness_pair_id": str(registry_row.get("quietness_pair_id") or ""),
                "is_conflict_intended": bool(registry_row.get("is_conflict_intended", False)),
                "is_surface_noise_only": bool(registry_row.get("is_surface_noise_only", False)),
                "edge_ids": list(row.get("edge_ids") or []),
                "metadata": dict(row.get("metadata") or {}),
            }
        )
    return focus_rows


def summarize_cycle_focus_rows(focus_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in focus_rows:
        grouped[(str(row["cycle_type"]), str(row["cell_id"]), str(row["cycle_outcome"]))].append(row)

    out_rows: List[Dict[str, Any]] = []
    for cycle_type, cell_id, cycle_outcome in sorted(grouped):
        rows = grouped[(cycle_type, cell_id, cycle_outcome)]
        defects = [
            float(row["holonomy_defect"])
            for row in rows
            if row["holonomy_defect"] not in (None, "")
        ]
        out_rows.append(
            {
                "cycle_type": cycle_type,
                "cell_id": cell_id,
                "cycle_outcome": cycle_outcome,
                "n_cycles": len(rows),
                "mean_holonomy_defect": mean_or_none(defects),
                "median_holonomy_defect": median_or_none(defects),
                "max_holonomy_defect": None if not defects else float(max(defects)),
            }
        )
    return out_rows


def build_quietness_pair_rows(
    focus_rows: Sequence[Dict[str, Any]],
    quietness_pair_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    cycle_by_benchmark_and_type = {
        (str(row["benchmark_sample_id"]), str(row["cycle_type"])): row for row in focus_rows
    }
    pair_rows: List[Dict[str, Any]] = []
    for pair in quietness_pair_rows:
        clean_benchmark_sample_id = str(pair["clean_benchmark_sample_id"])
        surface_noisy_benchmark_sample_id = str(pair["surface_noisy_benchmark_sample_id"])
        for cycle_type in ALLOWED_CYCLE_TYPES:
            clean_row = cycle_by_benchmark_and_type.get((clean_benchmark_sample_id, cycle_type))
            noisy_row = cycle_by_benchmark_and_type.get((surface_noisy_benchmark_sample_id, cycle_type))
            pair_outcome = "none"
            if clean_row is None or noisy_row is None:
                pair_outcome = "missing_cycle_row"
            elif str(clean_row["cycle_outcome"]) != "none" or str(noisy_row["cycle_outcome"]) != "none":
                pair_outcome = (
                    f"paired_cycle_failure:{clean_row['cycle_outcome']}"
                    if str(clean_row["cycle_outcome"]) == str(noisy_row["cycle_outcome"])
                    else "paired_cycle_failure_mixed"
                )
            clean_defect = None if clean_row is None else clean_row["holonomy_defect"]
            noisy_defect = None if noisy_row is None else noisy_row["holonomy_defect"]
            noisy_minus_clean = None
            abs_quietness_delta = None
            if pair_outcome == "none" and clean_defect not in (None, "") and noisy_defect not in (None, ""):
                noisy_minus_clean = float(noisy_defect) - float(clean_defect)
                abs_quietness_delta = abs(noisy_minus_clean)
            pair_rows.append(
                {
                    "quietness_pair_id": str(pair["quietness_pair_id"]),
                    "world_id": str(pair["world_id"]),
                    "world_type": str(pair["world_type"]),
                    "rendering_family_id": str(pair["rendering_family_id"]),
                    "cycle_type": cycle_type,
                    "clean_benchmark_sample_id": clean_benchmark_sample_id,
                    "surface_noisy_benchmark_sample_id": surface_noisy_benchmark_sample_id,
                    "clean_cycle_outcome": "" if clean_row is None else str(clean_row["cycle_outcome"]),
                    "surface_noisy_cycle_outcome": "" if noisy_row is None else str(noisy_row["cycle_outcome"]),
                    "pair_outcome": pair_outcome,
                    "clean_holonomy_defect": clean_defect,
                    "surface_noisy_holonomy_defect": noisy_defect,
                    "surface_noisy_minus_clean_defect": noisy_minus_clean,
                    "abs_quietness_delta": abs_quietness_delta,
                }
            )
    return pair_rows


def summarize_quietness_pair_rows(pair_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        grouped[(str(row["cycle_type"]), str(row["pair_outcome"]))].append(row)

    out_rows: List[Dict[str, Any]] = []
    for cycle_type, pair_outcome in sorted(grouped):
        rows = grouped[(cycle_type, pair_outcome)]
        deltas = [
            float(row["abs_quietness_delta"])
            for row in rows
            if row["abs_quietness_delta"] not in (None, "")
        ]
        signed = [
            float(row["surface_noisy_minus_clean_defect"])
            for row in rows
            if row["surface_noisy_minus_clean_defect"] not in (None, "")
        ]
        out_rows.append(
            {
                "cycle_type": cycle_type,
                "pair_outcome": pair_outcome,
                "n_pairs": len(rows),
                "mean_abs_quietness_delta": mean_or_none(deltas),
                "mean_surface_noisy_minus_clean_defect": mean_or_none(signed),
            }
        )
    return out_rows


def none_outcome_mean_by_cell(
    focus_rows: Sequence[Dict[str, Any]],
    cycle_type: str,
) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in focus_rows:
        if str(row["cycle_type"]) != cycle_type or str(row["cycle_outcome"]) != "none":
            continue
        defect = row["holonomy_defect"]
        if defect in (None, ""):
            continue
        grouped[str(row["cell_id"])].append(float(defect))
    return {cell_id: float(sum(values) / len(values)) for cell_id, values in grouped.items()}


def evaluate_falsifiers(focus_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out_rows: List[Dict[str, Any]] = []
    for cycle_type in ALLOWED_CYCLE_TYPES:
        means = none_outcome_mean_by_cell(focus_rows, cycle_type)
        cleaner_means = [means[cell_id] for cell_id in CLEANER_CELL_IDS if cell_id in means]
        conflict_means = [means[cell_id] for cell_id in CONFLICT_CELL_IDS if cell_id in means]
        max_cleaner = None if not cleaner_means else float(max(cleaner_means))
        max_conflict = None if not conflict_means else float(max(conflict_means))
        direct_mean = means.get("direct_contradiction")
        distributed_mean = means.get("distributed_incompatibility")

        cleaner_status = "insufficient_data"
        if max_cleaner is not None and max_conflict is not None:
            cleaner_status = "triggered" if max_cleaner >= max_conflict else "clear"
        out_rows.append(
            {
                "cycle_type": cycle_type,
                "falsifier_id": "cleaner_cell_dominance",
                "status": cleaner_status,
                "max_cleaner_mean_holonomy_defect": max_cleaner,
                "max_conflict_mean_holonomy_defect": max_conflict,
                "direct_contradiction_mean_holonomy_defect": direct_mean,
                "distributed_incompatibility_mean_holonomy_defect": distributed_mean,
            }
        )

        direct_escape_status = "insufficient_data"
        if max_cleaner is not None and direct_mean is not None:
            if direct_mean > max_cleaner and (distributed_mean is None or distributed_mean <= max_cleaner):
                direct_escape_status = "triggered"
            else:
                direct_escape_status = "clear"
        out_rows.append(
            {
                "cycle_type": cycle_type,
                "falsifier_id": "direct_contradiction_escape",
                "status": direct_escape_status,
                "max_cleaner_mean_holonomy_defect": max_cleaner,
                "max_conflict_mean_holonomy_defect": max_conflict,
                "direct_contradiction_mean_holonomy_defect": direct_mean,
                "distributed_incompatibility_mean_holonomy_defect": distributed_mean,
            }
        )

        distributed_status = "insufficient_data"
        if max_cleaner is not None:
            if distributed_mean is None or distributed_mean <= max_cleaner:
                distributed_status = "triggered"
            else:
                distributed_status = "clear"
        out_rows.append(
            {
                "cycle_type": cycle_type,
                "falsifier_id": "distributed_incompatibility_failure",
                "status": distributed_status,
                "max_cleaner_mean_holonomy_defect": max_cleaner,
                "max_conflict_mean_holonomy_defect": max_conflict,
                "direct_contradiction_mean_holonomy_defect": direct_mean,
                "distributed_incompatibility_mean_holonomy_defect": distributed_mean,
            }
        )

        conflict_rows = [
            row for row in focus_rows
            if str(row["cycle_type"]) == cycle_type and str(row["cell_id"]) in CONFLICT_CELL_IDS
        ]
        missing_anchor_status = "insufficient_data"
        if conflict_rows:
            conflict_success_by_cell = {
                cell_id: any(
                    str(row["cycle_outcome"]) == "none"
                    for row in conflict_rows
                    if str(row["cell_id"]) == cell_id
                )
                for cell_id in CONFLICT_CELL_IDS
            }
            missing_anchor_status = (
                "triggered" if not all(conflict_success_by_cell.values()) else "clear"
            )
        out_rows.append(
            {
                "cycle_type": cycle_type,
                "falsifier_id": "missing_anchor_collapse",
                "status": missing_anchor_status,
                "max_cleaner_mean_holonomy_defect": max_cleaner,
                "max_conflict_mean_holonomy_defect": max_conflict,
                "direct_contradiction_mean_holonomy_defect": direct_mean,
                "distributed_incompatibility_mean_holonomy_defect": distributed_mean,
            }
        )
    return out_rows


def build_report(
    run_id: str,
    source_gate9a_manifest: Dict[str, Any],
    cycle_focus_rows: Sequence[Dict[str, Any]],
    pair_rows: Sequence[Dict[str, Any]],
    cell_summary_rows: Sequence[Dict[str, Any]],
    falsifier_rows: Sequence[Dict[str, Any]],
) -> str:
    cycle_counts: Dict[str, int] = defaultdict(int)
    for row in cycle_focus_rows:
        cycle_counts[str(row["cycle_type"])] += 1

    lines = [
        "# Gate9B Small-Cycle Holonomy Read",
        "",
        f"run_id: {run_id}",
        f"source_gate9a_run_id: {source_gate9a_manifest.get('run_id', '')}",
        f"source_gate9a_code_git_commit: {source_gate9a_manifest.get('code_git_commit', '')}",
        f"source_gate8_run_id: {source_gate9a_manifest.get('source_gate8_run_id', '')}",
        "",
        "## Discipline",
        "",
        "- existing two cycle motifs only",
        "- per-cycle registry is primary",
        "- per-cell aggregate is supplementary",
        "- distributed_incompatibility remains the main proving ground",
        "",
        "## Cycle Counts",
        "",
    ]
    for cycle_type in ALLOWED_CYCLE_TYPES:
        lines.append(f"- {cycle_type}: {cycle_counts.get(cycle_type, 0)}")

    lines.extend(
        [
            "",
            "## Quietness Pair Comparison",
            "",
            "| cycle_type | pair_outcome | n_pairs | mean_abs_quietness_delta | mean_surface_noisy_minus_clean_defect |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in summarize_quietness_pair_rows(pair_rows):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["cycle_type"]),
                    str(row["pair_outcome"]),
                    str(row["n_pairs"]),
                    ""
                    if row["mean_abs_quietness_delta"] in (None, "")
                    else f"{float(row['mean_abs_quietness_delta']):.6f}",
                    ""
                    if row["mean_surface_noisy_minus_clean_defect"] in (None, "")
                    else f"{float(row['mean_surface_noisy_minus_clean_defect']):.6f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Cycle Summary By Cell",
            "",
            "| cycle_type | cell_id | cycle_outcome | n_cycles | mean_holonomy_defect | median_holonomy_defect | max_holonomy_defect |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in cell_summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["cycle_type"]),
                    str(row["cell_id"]),
                    str(row["cycle_outcome"]),
                    str(row["n_cycles"]),
                    "" if row["mean_holonomy_defect"] in (None, "") else f"{float(row['mean_holonomy_defect']):.6f}",
                    "" if row["median_holonomy_defect"] in (None, "") else f"{float(row['median_holonomy_defect']):.6f}",
                    "" if row["max_holonomy_defect"] in (None, "") else f"{float(row['max_holonomy_defect']):.6f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Falsifier Status",
            "",
            "| cycle_type | falsifier_id | status | max_cleaner_mean | max_conflict_mean | direct_mean | distributed_mean |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in falsifier_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["cycle_type"]),
                    str(row["falsifier_id"]),
                    str(row["status"]),
                    "" if row["max_cleaner_mean_holonomy_defect"] in (None, "") else f"{float(row['max_cleaner_mean_holonomy_defect']):.6f}",
                    "" if row["max_conflict_mean_holonomy_defect"] in (None, "") else f"{float(row['max_conflict_mean_holonomy_defect']):.6f}",
                    "" if row["direct_contradiction_mean_holonomy_defect"] in (None, "") else f"{float(row['direct_contradiction_mean_holonomy_defect']):.6f}",
                    "" if row["distributed_incompatibility_mean_holonomy_defect"] in (None, "") else f"{float(row['distributed_incompatibility_mean_holonomy_defect']):.6f}",
                ]
            )
            + " |"
        )

    return "\n".join(lines) + "\n"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()
    source_dir = Path(args.gate9a_dir)
    out_dir = Path(args.out_dir)
    run_id = out_dir.name

    source_manifest = gate9a.read_json(source_dir / gate9a.DEFAULT_MANIFEST)
    cycle_rows = gate9a.read_jsonl(source_dir / gate9a.DEFAULT_CYCLE_REGISTRY)

    source_gate8_dir = REPO_ROOT / str(source_manifest["source_gate8_execution_dir"])
    sample_registry_rows = gate9a.read_jsonl(source_gate8_dir / "sample_registry.jsonl")
    quietness_pair_rows = gate9a.read_jsonl(source_gate8_dir / "quietness_pairs.jsonl")
    sample_registry_by_benchmark = {
        str(row["benchmark_sample_id"]): row for row in sample_registry_rows
    }

    cycle_focus_rows = build_cycle_focus_rows(cycle_rows, sample_registry_by_benchmark)
    pair_rows = build_quietness_pair_rows(cycle_focus_rows, quietness_pair_rows)
    cell_summary_rows = summarize_cycle_focus_rows(cycle_focus_rows)
    falsifier_rows = evaluate_falsifiers(cycle_focus_rows)

    cycle_focus_path = out_dir / DEFAULT_CYCLE_FOCUS
    pair_registry_path = out_dir / DEFAULT_PAIR_REGISTRY
    cell_summary_path = out_dir / DEFAULT_CELL_SUMMARY
    pair_summary_path = out_dir / DEFAULT_PAIR_SUMMARY
    falsifier_status_path = out_dir / DEFAULT_FALSIFIER_STATUS
    report_path = out_dir / DEFAULT_REPORT
    manifest_path = out_dir / DEFAULT_MANIFEST
    checksums_path = out_dir / DEFAULT_CHECKSUMS

    gate9a.write_jsonl(cycle_focus_path, cycle_focus_rows)
    gate9a.write_jsonl(pair_registry_path, pair_rows)
    gate9a.write_csv(
        cell_summary_path,
        (
            "cycle_type",
            "cell_id",
            "cycle_outcome",
            "n_cycles",
            "mean_holonomy_defect",
            "median_holonomy_defect",
            "max_holonomy_defect",
        ),
        cell_summary_rows,
    )
    gate9a.write_csv(
        pair_summary_path,
        (
            "cycle_type",
            "pair_outcome",
            "n_pairs",
            "mean_abs_quietness_delta",
            "mean_surface_noisy_minus_clean_defect",
        ),
        summarize_quietness_pair_rows(pair_rows),
    )
    gate9a.write_json(
        falsifier_status_path,
        {
            "falsifier_rows": falsifier_rows,
            "overall_status": (
                "triggered"
                if any(str(row["status"]) == "triggered" for row in falsifier_rows)
                else "clear"
            ),
        },
    )
    gate9a.write_text(
        report_path,
        build_report(
            run_id=run_id,
            source_gate9a_manifest=source_manifest,
            cycle_focus_rows=cycle_focus_rows,
            pair_rows=pair_rows,
            cell_summary_rows=cell_summary_rows,
            falsifier_rows=falsifier_rows,
        ),
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "run_id": run_id,
        "code_git_commit": gate9a.current_git_commit(),
        "source_gate9a_dir": gate9a.repo_relative_or_posix(source_dir),
        "source_gate9a_run_id": str(source_manifest.get("run_id") or ""),
        "source_gate9a_code_git_commit": str(source_manifest.get("code_git_commit") or ""),
        "source_gate8_run_id": str(source_manifest.get("source_gate8_run_id") or ""),
        "source_gate8_code_git_commit": str(source_manifest.get("source_gate8_code_git_commit") or ""),
        "source_rendering_family_id": str(source_manifest.get("source_rendering_family_id") or ""),
        "allowed_cycle_types": list(ALLOWED_CYCLE_TYPES),
        "cycle_registry_primary": True,
        "per_cell_aggregate_supplemental": True,
        "falsifier_ids": list(FALSIFIER_IDS),
        "paths": {
            DEFAULT_CYCLE_FOCUS: gate9a.repo_relative_or_posix(cycle_focus_path),
            DEFAULT_PAIR_REGISTRY: gate9a.repo_relative_or_posix(pair_registry_path),
            DEFAULT_CELL_SUMMARY: gate9a.repo_relative_or_posix(cell_summary_path),
            DEFAULT_PAIR_SUMMARY: gate9a.repo_relative_or_posix(pair_summary_path),
            DEFAULT_FALSIFIER_STATUS: gate9a.repo_relative_or_posix(falsifier_status_path),
            DEFAULT_REPORT: gate9a.repo_relative_or_posix(report_path),
        },
    }
    gate9a.write_json(manifest_path, manifest)
    gate9a.write_json(
        checksums_path,
        {
            DEFAULT_MANIFEST: sha256_file(manifest_path),
            DEFAULT_CYCLE_FOCUS: sha256_file(cycle_focus_path),
            DEFAULT_PAIR_REGISTRY: sha256_file(pair_registry_path),
            DEFAULT_CELL_SUMMARY: sha256_file(cell_summary_path),
            DEFAULT_PAIR_SUMMARY: sha256_file(pair_summary_path),
            DEFAULT_FALSIFIER_STATUS: sha256_file(falsifier_status_path),
            DEFAULT_REPORT: sha256_file(report_path),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
