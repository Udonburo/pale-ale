#!/usr/bin/env python3
"""Evaluate genealogy-specific reader candidates on fixed Gate5 FWHT artifacts."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]

BASELINE = "first_after_defect_score_ranknorm"
GENEALOGY_POST_START_MASS_W1 = "genealogy_post_start_mass_w1_ranknorm"
GENEALOGY_POST_START_MEAN_W2 = "genealogy_post_start_mean_w2_ranknorm"
GENEALOGY_BEFORE_PENALIZED_FIRST_AFTER_W1 = "genealogy_before_penalized_first_after_w1_ranknorm"

CANDIDATE_ORDER = [
    BASELINE,
    GENEALOGY_POST_START_MASS_W1,
    GENEALOGY_POST_START_MEAN_W2,
    GENEALOGY_BEFORE_PENALIZED_FIRST_AFTER_W1,
]

TEMPORAL_K0_MAX_EXTRA_DEGRADATION = 0.02
DEFAULT_SELECTED_WINS = [137, 147, 149, 11, 167]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare narrow genealogy-specific reader candidates on an existing Gate5 FWHT "
            "baseline run without changing the boundary or comparator."
        )
    )
    parser.add_argument("--gate5-out-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sample-ids", nargs="+", type=int, default=DEFAULT_SELECTED_WINS)
    parser.add_argument("--k-values", nargs="+", type=int, default=[0, 3])
    parser.add_argument("--genealogy-worst-count", type=int, default=5)
    return parser.parse_args()


def parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        out = float(raw)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            encoded: Dict[str, Any] = {}
            for field in fieldnames:
                value = row.get(field)
                if isinstance(value, float):
                    encoded[field] = f"{value:.17e}"
                elif value is None:
                    encoded[field] = ""
                else:
                    encoded[field] = value
            writer.writerow(encoded)


def mean(values: Iterable[float]) -> Optional[float]:
    arr = list(values)
    if not arr:
        return None
    return sum(arr) / float(len(arr))


def render_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def group_token_rows(rows: Sequence[Dict[str, str]]) -> Dict[int, List[Dict[str, str]]]:
    grouped: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["sample_id"])].append(row)
    for sample_rows in grouped.values():
        sample_rows.sort(key=lambda row: int(row["step"]))
    return grouped


def dilate_labels(labels: Sequence[int], k: int) -> List[int]:
    out = [0] * len(labels)
    positive_steps = [idx for idx, label in enumerate(labels) if label == 1]
    for step in positive_steps:
        lo = max(0, step - k)
        hi = min(len(labels) - 1, step + k)
        for idx in range(lo, hi + 1):
            out[idx] = 1
    return out


def defect_span(labels: Sequence[int]) -> Tuple[Optional[int], Optional[int]]:
    steps = [idx for idx, label in enumerate(labels) if label == 1]
    if not steps:
        return (None, None)
    return (steps[0], steps[-1])


def rank_normalize_scores(scores: Sequence[Optional[float]]) -> List[Optional[float]]:
    present = [(idx, float(score)) for idx, score in enumerate(scores) if score is not None]
    if not present:
        return [None] * len(scores)

    sorted_present = sorted(present, key=lambda item: (item[1], item[0]))
    ranks: Dict[int, float] = {}
    total = len(sorted_present)
    cursor = 0
    while cursor < total:
        next_cursor = cursor + 1
        while next_cursor < total and sorted_present[next_cursor][1] == sorted_present[cursor][1]:
            next_cursor += 1
        average_rank = ((cursor + 1) + next_cursor) / 2.0
        normalized = 0.0 if total == 1 else (average_rank - 1.0) / float(total - 1)
        for tied_idx in range(cursor, next_cursor):
            ranks[sorted_present[tied_idx][0]] = normalized
        cursor = next_cursor

    out: List[Optional[float]] = [None] * len(scores)
    for idx, _score in present:
        out[idx] = ranks[idx]
    return out


def value_at(scores: Sequence[Optional[float]], index: int) -> Optional[float]:
    if not scores:
        return None
    index = max(0, min(len(scores) - 1, index))
    return scores[index]


def candidate_label(candidate_id: str) -> str:
    return {
        BASELINE: "first_after_defect_score_ranknorm",
        GENEALOGY_POST_START_MASS_W1: "genealogy_post_start_mass_w1_ranknorm",
        GENEALOGY_POST_START_MEAN_W2: "genealogy_post_start_mean_w2_ranknorm",
        GENEALOGY_BEFORE_PENALIZED_FIRST_AFTER_W1: "genealogy_before_penalized_first_after_w1_ranknorm",
    }[candidate_id]


def compute_candidate_values(labels: Sequence[int], scores: Sequence[Optional[float]]) -> Dict[str, Optional[float]]:
    start, end = defect_span(labels)
    if start is None or end is None:
        return {
            BASELINE: None,
            GENEALOGY_POST_START_MASS_W1: None,
            GENEALOGY_POST_START_MEAN_W2: None,
            GENEALOGY_BEFORE_PENALIZED_FIRST_AFTER_W1: None,
        }

    rank_scores = rank_normalize_scores(scores)
    first_after = value_at(rank_scores, start)
    post_start_1 = value_at(rank_scores, start + 1)
    before_1 = value_at(rank_scores, start - 1)

    post_start_mass_w1 = None
    if first_after is not None or post_start_1 is not None:
        post_start_mass_w1 = float(first_after or 0.0) + float(post_start_1 or 0.0)

    post_start_mean_w2 = None
    values = [value for value in (first_after, post_start_1) if value is not None]
    if values:
        post_start_mean_w2 = sum(values) / float(len(values))

    before_penalized = None
    if first_after is not None:
        before_penalized = first_after - float(before_1 or 0.0)

    return {
        BASELINE: first_after,
        GENEALOGY_POST_START_MASS_W1: post_start_mass_w1,
        GENEALOGY_POST_START_MEAN_W2: post_start_mean_w2,
        GENEALOGY_BEFORE_PENALIZED_FIRST_AFTER_W1: before_penalized,
    }


def build_sample_rows(
    sample_summary: Dict[str, str],
    token_rows: Sequence[Dict[str, str]],
    k: int,
) -> List[Dict[str, Any]]:
    base_labels = [1 if row.get("label_token") == "1" else 0 for row in token_rows]
    labels = dilate_labels(base_labels, k)
    defect_start, defect_end = defect_span(labels)
    scores_f = [parse_float(row.get("score_F_loop")) for row in token_rows]
    scores_rotor = [parse_float(row.get("rotor_loop_chordal_v1")) for row in token_rows]
    values_f = compute_candidate_values(labels, scores_f)
    values_rotor = compute_candidate_values(labels, scores_rotor)

    rows: List[Dict[str, Any]] = []
    for candidate_id in CANDIDATE_ORDER:
        metric_f = values_f[candidate_id]
        metric_rotor = values_rotor[candidate_id]
        rows.append(
            {
                "sample_id": int(sample_summary["sample_id"]),
                "variant": sample_summary.get("variant", ""),
                "world_type": sample_summary.get("world_type", ""),
                "k": k,
                "candidate_id": candidate_id,
                "defect_start_step_dilated": defect_start,
                "defect_end_step_dilated": defect_end,
                "metric_F": metric_f,
                "metric_rotor": metric_rotor,
                "delta_rotor_vs_F": (
                    None if metric_f is None or metric_rotor is None else metric_rotor - metric_f
                ),
                "rotor_gt_F": (
                    None if metric_f is None or metric_rotor is None else (1 if metric_rotor > metric_f else 0)
                ),
            }
        )
    return rows


def summarize_rows(group_id: str, rows: Sequence[Dict[str, Any]], selected_ids: Sequence[int]) -> Dict[str, Any]:
    selected_set = {int(sample_id) for sample_id in selected_ids}
    selected_rows = [row for row in rows if int(row["sample_id"]) in selected_set]
    return {
        "group_id": group_id,
        "n_samples": len(rows),
        "mean_metric_F": mean(row["metric_F"] for row in rows if row["metric_F"] is not None),
        "mean_metric_rotor": mean(row["metric_rotor"] for row in rows if row["metric_rotor"] is not None),
        "mean_delta_rotor_vs_F": mean(
            row["delta_rotor_vs_F"] for row in rows if row["delta_rotor_vs_F"] is not None
        ),
        "rotor_win_rate": mean(float(row["rotor_gt_F"]) for row in rows if row["rotor_gt_F"] is not None),
        "selected_win_preservation_rate": mean(
            float(row["rotor_gt_F"]) for row in selected_rows if row["rotor_gt_F"] is not None
        ),
    }


def build_case_rows(
    sample_rows: Sequence[Dict[str, Any]],
    selected_ids: Sequence[int],
    genealogy_worst_count: int,
) -> List[Dict[str, Any]]:
    grouped: Dict[int, Dict[str, Any]] = {}
    for row in sample_rows:
        sample_id = int(row["sample_id"])
        current = grouped.setdefault(
            sample_id,
            {
                "sample_id": sample_id,
                "variant": row["variant"],
                "world_type": row["world_type"],
            },
        )
        key_prefix = f"{row['candidate_id']}_k{row['k']}"
        current[f"{key_prefix}_metric_F"] = row["metric_F"]
        current[f"{key_prefix}_metric_rotor"] = row["metric_rotor"]
        current[f"{key_prefix}_delta_rotor_vs_F"] = row["delta_rotor_vs_F"]
        current[f"{key_prefix}_rotor_gt_F"] = row["rotor_gt_F"]
        current[f"defect_start_step_k{row['k']}"] = row["defect_start_step_dilated"]
        current[f"defect_end_step_k{row['k']}"] = row["defect_end_step_dilated"]

    selected_set = {int(sample_id) for sample_id in selected_ids}
    selected_rows: List[Dict[str, Any]] = []
    for row in grouped.values():
        if int(row["sample_id"]) in selected_set:
            copied = dict(row)
            copied["case_group"] = "existing_global_win"
            selected_rows.append(copied)

    genealogy_rows = [
        row
        for row in grouped.values()
        if row["world_type"] == "genealogy"
        and row.get(f"{BASELINE}_k0_delta_rotor_vs_F") is not None
    ]
    genealogy_rows.sort(key=lambda row: row[f"{BASELINE}_k0_delta_rotor_vs_F"])
    worst_rows = []
    for row in genealogy_rows[:genealogy_worst_count]:
        copied = dict(row)
        copied["case_group"] = "baseline_genealogy_k0_worst"
        worst_rows.append(copied)

    case_rows = selected_rows + worst_rows
    case_rows.sort(key=lambda row: (str(row["case_group"]), int(row["sample_id"])))
    return case_rows


def choose_decision(global_rows: Sequence[Dict[str, Any]], world_rows: Sequence[Dict[str, Any]]) -> Tuple[str, str]:
    global_index = {(int(row["k"]), str(row["candidate_id"])): row for row in global_rows}
    world_index = {
        (int(row["k"]), str(row["candidate_id"]), str(row["group_id"])): row
        for row in world_rows
    }

    baseline_genealogy_k0 = world_index[(0, BASELINE, "genealogy")]["mean_delta_rotor_vs_F"]
    baseline_genealogy_k3 = world_index[(3, BASELINE, "genealogy")]["mean_delta_rotor_vs_F"]
    current_post_start_temporal_k0 = world_index[(0, GENEALOGY_POST_START_MASS_W1, "temporal")]["mean_delta_rotor_vs_F"]
    current_post_start_selected_k3 = global_index[(3, GENEALOGY_POST_START_MASS_W1)]["selected_win_preservation_rate"]

    def meets_guardrail(candidate_id: str) -> bool:
        temporal_k0 = world_index[(0, candidate_id, "temporal")]["mean_delta_rotor_vs_F"]
        selected_k3 = global_index[(3, candidate_id)]["selected_win_preservation_rate"]
        if temporal_k0 is None or current_post_start_temporal_k0 is None or selected_k3 is None or current_post_start_selected_k3 is None:
            return False
        return temporal_k0 >= (current_post_start_temporal_k0 - TEMPORAL_K0_MAX_EXTRA_DEGRADATION) and selected_k3 >= current_post_start_selected_k3

    genealogy_improvers: List[Tuple[str, float, float, bool]] = []
    for candidate_id in CANDIDATE_ORDER:
        if candidate_id == BASELINE:
            continue
        genealogy_k0 = world_index[(0, candidate_id, "genealogy")]["mean_delta_rotor_vs_F"]
        genealogy_k3 = world_index[(3, candidate_id, "genealogy")]["mean_delta_rotor_vs_F"]
        if genealogy_k0 is None or genealogy_k3 is None or baseline_genealogy_k0 is None or baseline_genealogy_k3 is None:
            continue
        if genealogy_k0 > baseline_genealogy_k0:
            genealogy_improvers.append((candidate_id, genealogy_k0, genealogy_k3, meets_guardrail(candidate_id)))

    for candidate_id, genealogy_k0, genealogy_k3, guard_ok in genealogy_improvers:
        if guard_ok and genealogy_k3 >= baseline_genealogy_k3:
            return (
                "genealogy-reader-still-live",
                f"{candidate_label(candidate_id)} improves genealogy k=0 ({render_float(genealogy_k0)} vs baseline {render_float(baseline_genealogy_k0)}) "
                f"without violating the temporal k=0 guardrail (no worse than current post-start by more than {TEMPORAL_K0_MAX_EXTRA_DEGRADATION:.2f}) "
                f"and preserves genealogy k=3 ({render_float(genealogy_k3)}).",
            )

    if genealogy_improvers:
        best_candidate, best_genealogy_k0, best_genealogy_k3, guard_ok = max(
            genealogy_improvers, key=lambda item: item[1]
        )
        temporal_k0 = world_index[(0, best_candidate, "temporal")]["mean_delta_rotor_vs_F"]
        return (
            "genealogy-reader-no-clear-candidate",
            f"{candidate_label(best_candidate)} improves genealogy k=0 ({render_float(best_genealogy_k0)} vs baseline {render_float(baseline_genealogy_k0)}) "
            f"but fails the guardrails: genealogy k3={render_float(best_genealogy_k3)}, temporal k0={render_float(temporal_k0)}, "
            f"guardrail floor={render_float((current_post_start_temporal_k0 or 0.0) - TEMPORAL_K0_MAX_EXTRA_DEGRADATION)}.",
        )

    return (
        "genealogy-line-not-worth-pursuing",
        "none of the genealogy-specific candidates improves genealogy k=0 over the baseline reader.",
    )


def write_report(
    out_path: Path,
    manifest: Dict[str, Any],
    global_rows: Sequence[Dict[str, Any]],
    world_rows: Sequence[Dict[str, Any]],
    case_rows: Sequence[Dict[str, Any]],
    decision: str,
    rationale: str,
) -> None:
    lines = [
        "# Gate5 Genealogy-Specific Reader Lab",
        "",
        f"Run ID: {manifest.get('run_id', '')}",
        f"Method ID: {manifest.get('method_id', '')}",
        "",
        "Fixed inputs:",
        "- boundary fixed to FWHT baseline",
        "- comparator fixed to `rotor_loop_chordal_v1`",
        "- labels fixed to `k=0,3`",
        "- focus restricted to genealogy-specific readers",
        "",
        "Candidate definitions:",
        f"- `{candidate_label(BASELINE)}`: `rank(defect_start)`",
        f"- `{candidate_label(GENEALOGY_POST_START_MASS_W1)}`: `rank(defect_start) + rank(defect_start+1)`",
        f"- `{candidate_label(GENEALOGY_POST_START_MEAN_W2)}`: `mean(rank(defect_start), rank(defect_start+1))`",
        f"- `{candidate_label(GENEALOGY_BEFORE_PENALIZED_FIRST_AFTER_W1)}`: `rank(defect_start) - rank(defect_start-1)`",
        "",
        f"Temporal k=0 guardrail: candidate temporal k=0 mean_delta_rotor_vs_F must not be worse than the current `{candidate_label(GENEALOGY_POST_START_MASS_W1)}` temporal k=0 by more than {TEMPORAL_K0_MAX_EXTRA_DEGRADATION:.2f}.",
        "Worst genealogy failures: baseline reader genealogy k=0 `delta_rotor_vs_F` ascending, top 5.",
        "",
        "## Global Summary",
        "",
        "| k | candidate | n | mean_metric_F | mean_metric_rotor | mean_delta_rotor_vs_F | rotor_win_rate | selected_win_preservation_rate |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in global_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["k"]),
                    candidate_label(str(row["candidate_id"])),
                    str(row["n_samples"]),
                    render_float(row["mean_metric_F"]),
                    render_float(row["mean_metric_rotor"]),
                    render_float(row["mean_delta_rotor_vs_F"]),
                    render_float(row["rotor_win_rate"]),
                    render_float(row["selected_win_preservation_rate"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## World-Type Summary",
            "",
            "| k | candidate | world_type | n | mean_delta_rotor_vs_F | rotor_win_rate | selected_win_preservation_rate |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in world_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["k"]),
                    candidate_label(str(row["candidate_id"])),
                    str(row["group_id"]),
                    str(row["n_samples"]),
                    render_float(row["mean_delta_rotor_vs_F"]),
                    render_float(row["rotor_win_rate"]),
                    render_float(row["selected_win_preservation_rate"]),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Selected Cases", ""])
    current_group = None
    for row in case_rows:
        if current_group != row["case_group"]:
            current_group = row["case_group"]
            lines.extend([f"### {current_group}", ""])
        lines.append(
            f"- sample_id={row['sample_id']} world_type={row['world_type']} "
            f"baseline_k0={render_float(row.get(f'{BASELINE}_k0_delta_rotor_vs_F'))} "
            f"mass_w1_k0={render_float(row.get(f'{GENEALOGY_POST_START_MASS_W1}_k0_delta_rotor_vs_F'))} "
            f"mean_w2_k0={render_float(row.get(f'{GENEALOGY_POST_START_MEAN_W2}_k0_delta_rotor_vs_F'))} "
            f"before_penalized_k0={render_float(row.get(f'{GENEALOGY_BEFORE_PENALIZED_FIRST_AFTER_W1}_k0_delta_rotor_vs_F'))}"
        )

    lines.extend(["", "## Decision", "", f"- decision: `{decision}`", f"- {rationale}"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_decision(out_path: Path, decision: str, rationale: str, world_rows: Sequence[Dict[str, Any]]) -> None:
    world_index = {
        (int(row["k"]), str(row["candidate_id"]), str(row["group_id"])): row
        for row in world_rows
    }
    lines = [
        "# Gate5 Genealogy-Specific Reader Decision",
        "",
        f"- decision: `{decision}`",
        "",
        "## Key Basis",
        "",
        f"- baseline genealogy k0: {render_float(world_index[(0, BASELINE, 'genealogy')]['mean_delta_rotor_vs_F'])}",
        f"- mass_w1 genealogy k0: {render_float(world_index[(0, GENEALOGY_POST_START_MASS_W1, 'genealogy')]['mean_delta_rotor_vs_F'])}",
        f"- mean_w2 genealogy k0: {render_float(world_index[(0, GENEALOGY_POST_START_MEAN_W2, 'genealogy')]['mean_delta_rotor_vs_F'])}",
        f"- before_penalized genealogy k0: {render_float(world_index[(0, GENEALOGY_BEFORE_PENALIZED_FIRST_AFTER_W1, 'genealogy')]['mean_delta_rotor_vs_F'])}",
        f"- current post-start temporal k0: {render_float(world_index[(0, GENEALOGY_POST_START_MASS_W1, 'temporal')]['mean_delta_rotor_vs_F'])}",
        "",
        f"- {rationale}",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    gate5_out_dir = (REPO_ROOT / args.gate5_out_dir).resolve()
    out_dir = (REPO_ROOT / args.out_dir).resolve()

    if set(args.k_values) != {0, 3}:
        raise ValueError("--k-values must be exactly 0 and 3 for genealogy-specific reader")

    token_rows = read_csv(gate5_out_dir / "gate5_token_telemetry.csv")
    sample_rows = read_csv(gate5_out_dir / "gate5_sample_summary.csv")
    manifest = json.loads((gate5_out_dir / "manifest.json").read_text(encoding="utf-8"))
    token_grouped = group_token_rows(token_rows)

    frustrated_rows = [row for row in sample_rows if row.get("variant") == "frustrated"]
    genealogy_frustrated_rows = [row for row in frustrated_rows if row.get("world_type") == "genealogy"]
    if not frustrated_rows:
        raise ValueError("genealogy-specific reader found zero frustrated samples")
    if not genealogy_frustrated_rows:
        raise ValueError("genealogy-specific reader found zero frustrated genealogy samples")

    present_ids = {int(row["sample_id"]) for row in frustrated_rows}
    missing_ids = [sample_id for sample_id in args.sample_ids if sample_id not in present_ids]
    if missing_ids:
        raise ValueError(
            "requested sample_ids are not present in genealogy-specific reader frustrated population: "
            + ",".join(str(sample_id) for sample_id in missing_ids)
        )

    sample_candidate_rows: List[Dict[str, Any]] = []
    for summary_row in frustrated_rows:
        sample_id = int(summary_row["sample_id"])
        token_sample_rows = token_grouped[sample_id]
        for k in sorted(args.k_values):
            sample_candidate_rows.extend(build_sample_rows(summary_row, token_sample_rows, k))

    sample_candidate_rows.sort(
        key=lambda row: (int(row["k"]), CANDIDATE_ORDER.index(str(row["candidate_id"])), int(row["sample_id"]))
    )

    global_rows: List[Dict[str, Any]] = []
    world_rows: List[Dict[str, Any]] = []
    for k in sorted(args.k_values):
        k_rows = [row for row in sample_candidate_rows if int(row["k"]) == k]
        for candidate_id in CANDIDATE_ORDER:
            candidate_rows = [row for row in k_rows if row["candidate_id"] == candidate_id]
            global_row = summarize_rows("all", candidate_rows, args.sample_ids)
            global_row["k"] = k
            global_row["candidate_id"] = candidate_id
            global_rows.append(global_row)
            for world_type in sorted({str(row["world_type"]) for row in candidate_rows}):
                grouped = [row for row in candidate_rows if row["world_type"] == world_type]
                world_row = summarize_rows(world_type, grouped, args.sample_ids)
                world_row["k"] = k
                world_row["candidate_id"] = candidate_id
                world_rows.append(world_row)

    case_rows = build_case_rows(sample_candidate_rows, args.sample_ids, args.genealogy_worst_count)

    decision, rationale = choose_decision(global_rows, world_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    sample_csv = out_dir / "gate5_genealogy_specific_reader_sample_summary.csv"
    world_csv = out_dir / "gate5_genealogy_specific_reader_world_summary.csv"
    selected_csv = out_dir / "gate5_genealogy_specific_reader_selected_cases.csv"
    report_md = out_dir / "gate5_genealogy_specific_reader_report.md"
    decision_md = out_dir / "gate5_genealogy_specific_reader_decision.md"

    write_csv(
        sample_csv,
        [
            "sample_id",
            "variant",
            "world_type",
            "k",
            "candidate_id",
            "defect_start_step_dilated",
            "defect_end_step_dilated",
            "metric_F",
            "metric_rotor",
            "delta_rotor_vs_F",
            "rotor_gt_F",
        ],
        sample_candidate_rows,
    )
    write_csv(
        world_csv,
        [
            "k",
            "candidate_id",
            "group_id",
            "n_samples",
            "mean_metric_F",
            "mean_metric_rotor",
            "mean_delta_rotor_vs_F",
            "rotor_win_rate",
            "selected_win_preservation_rate",
        ],
        global_rows + world_rows,
    )

    case_fieldnames = [
        "case_group",
        "sample_id",
        "variant",
        "world_type",
        "defect_start_step_k0",
        "defect_end_step_k0",
        "defect_start_step_k3",
        "defect_end_step_k3",
    ]
    for candidate_id in CANDIDATE_ORDER:
        for k in (0, 3):
            case_fieldnames.extend(
                [
                    f"{candidate_id}_k{k}_metric_F",
                    f"{candidate_id}_k{k}_metric_rotor",
                    f"{candidate_id}_k{k}_delta_rotor_vs_F",
                    f"{candidate_id}_k{k}_rotor_gt_F",
                ]
            )

    write_csv(selected_csv, case_fieldnames, case_rows)
    write_report(report_md, manifest, global_rows, world_rows, case_rows, decision, rationale)
    write_decision(decision_md, decision, rationale, world_rows)

    print(f"sample_summary_csv={sample_csv.as_posix()}")
    print(f"world_summary_csv={world_csv.as_posix()}")
    print(f"selected_cases_csv={selected_csv.as_posix()}")
    print(f"report_md={report_md.as_posix()}")
    print(f"decision_md={decision_md.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
