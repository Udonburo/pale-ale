#!/usr/bin/env python3
"""Autopsy k=0 failure modes for post_start_mass_w1_ranknorm on fixed Gate5 artifacts."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]

BASELINE = "first_after_defect_score_ranknorm"
POST_START_MASS_W1 = "post_start_mass_w1_ranknorm"
POST_START_MASS_W1_PREFIX_PENALIZED = "post_start_mass_w1_prefix_penalized_ranknorm"
POST_START_MEAN_W2 = "post_start_mean_w2_ranknorm"

CANDIDATE_ORDER = [
    BASELINE,
    POST_START_MASS_W1,
    POST_START_MASS_W1_PREFIX_PENALIZED,
    POST_START_MEAN_W2,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose why post_start_mass_w1_ranknorm fails at k=0, with explicit genealogy "
            "worst failures and temporal side-effect cuts on a fixed Gate5 FWHT baseline."
        )
    )
    parser.add_argument("--gate5-out-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sample-ids", nargs="+", type=int, default=[137, 147, 149, 11, 167])
    parser.add_argument("--k-values", nargs="+", type=int, default=[0, 3])
    parser.add_argument("--genealogy-worst-count", type=int, default=5)
    parser.add_argument("--temporal-side-effect-count", type=int, default=5)
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


def sum_window(scores: Sequence[Optional[float]], start: int, end: int) -> float:
    lo = max(0, start)
    hi = min(len(scores) - 1, end)
    if hi < lo:
        return 0.0
    total = 0.0
    for idx in range(lo, hi + 1):
        score = scores[idx]
        if score is not None:
            total += float(score)
    return total


def mean_window(scores: Sequence[Optional[float]], start: int, end: int) -> Optional[float]:
    lo = max(0, start)
    hi = min(len(scores) - 1, end)
    if hi < lo:
        return None
    values = [float(scores[idx]) for idx in range(lo, hi + 1) if scores[idx] is not None]
    if not values:
        return None
    return sum(values) / float(len(values))


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
        avg_rank = ((cursor + 1) + next_cursor) / 2.0
        normalized = 0.0 if total == 1 else (avg_rank - 1.0) / float(total - 1)
        for tied_idx in range(cursor, next_cursor):
            ranks[sorted_present[tied_idx][0]] = normalized
        cursor = next_cursor

    out: List[Optional[float]] = [None] * len(scores)
    for idx, _score in present:
        out[idx] = ranks[idx]
    return out


def candidate_label(candidate_id: str) -> str:
    return {
        BASELINE: "first_after_defect_score_ranknorm",
        POST_START_MASS_W1: "post_start_mass_w1_ranknorm",
        POST_START_MASS_W1_PREFIX_PENALIZED: "post_start_mass_w1_prefix_penalized_ranknorm",
        POST_START_MEAN_W2: "post_start_mean_w2_ranknorm",
    }[candidate_id]


def compute_candidate_values(labels: Sequence[int], scores: Sequence[Optional[float]]) -> Dict[str, Optional[float]]:
    start, end = defect_span(labels)
    if start is None or end is None:
        return {
            BASELINE: None,
            POST_START_MASS_W1: None,
            POST_START_MASS_W1_PREFIX_PENALIZED: None,
            POST_START_MEAN_W2: None,
        }

    rank_scores = rank_normalize_scores(scores)
    baseline = rank_scores[start]
    post_start_mass_w1 = sum_window(rank_scores, start, start + 1)
    before_penalized = None
    if baseline is not None:
        before_penalized = post_start_mass_w1 - sum_window(rank_scores, start - 1, start - 1)
    post_start_mean_w2 = mean_window(rank_scores, start, start + 1)
    return {
        BASELINE: baseline,
        POST_START_MASS_W1: post_start_mass_w1,
        POST_START_MASS_W1_PREFIX_PENALIZED: before_penalized,
        POST_START_MEAN_W2: post_start_mean_w2,
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


def build_wide_rows(sample_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
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

    out = list(grouped.values())
    for row in out:
        baseline_k0 = row.get(f"{BASELINE}_k0_delta_rotor_vs_F")
        post_start_k0 = row.get(f"{POST_START_MASS_W1}_k0_delta_rotor_vs_F")
        baseline_k3 = row.get(f"{BASELINE}_k3_delta_rotor_vs_F")
        post_start_k3 = row.get(f"{POST_START_MASS_W1}_k3_delta_rotor_vs_F")
        row["post_start_vs_baseline_delta_gap_k0"] = (
            None if baseline_k0 is None or post_start_k0 is None else post_start_k0 - baseline_k0
        )
        row["post_start_vs_baseline_delta_gap_k3"] = (
            None if baseline_k3 is None or post_start_k3 is None else post_start_k3 - baseline_k3
        )
    out.sort(key=lambda item: int(item["sample_id"]))
    return out


def choose_decision(world_rows: Sequence[Dict[str, Any]], global_rows: Sequence[Dict[str, Any]]) -> Tuple[str, str]:
    world_index = {
        (int(row["k"]), str(row["candidate_id"]), str(row["group_id"])): row
        for row in world_rows
    }
    global_index = {(int(row["k"]), str(row["candidate_id"])): row for row in global_rows}

    baseline_genealogy_k0 = world_index[(0, BASELINE, "genealogy")]
    post_start_genealogy_k0 = world_index[(0, POST_START_MASS_W1, "genealogy")]
    prefix_genealogy_k0 = world_index[(0, POST_START_MASS_W1_PREFIX_PENALIZED, "genealogy")]
    mean_w2_genealogy_k0 = world_index[(0, POST_START_MEAN_W2, "genealogy")]
    post_start_temporal_k0 = world_index[(0, POST_START_MASS_W1, "temporal")]
    prefix_temporal_k0 = world_index[(0, POST_START_MASS_W1_PREFIX_PENALIZED, "temporal")]
    mean_w2_temporal_k0 = world_index[(0, POST_START_MEAN_W2, "temporal")]

    post_start_selected_k3 = global_index[(3, POST_START_MASS_W1)]["selected_win_preservation_rate"] or 0.0
    prefix_selected_k3 = global_index[(3, POST_START_MASS_W1_PREFIX_PENALIZED)]["selected_win_preservation_rate"] or 0.0
    mean_w2_selected_k3 = global_index[(3, POST_START_MEAN_W2)]["selected_win_preservation_rate"] or 0.0

    candidates = [
        (
            POST_START_MASS_W1,
            post_start_genealogy_k0,
            post_start_temporal_k0,
            global_index[(3, POST_START_MASS_W1)],
            post_start_selected_k3,
        ),
        (
            POST_START_MASS_W1_PREFIX_PENALIZED,
            prefix_genealogy_k0,
            prefix_temporal_k0,
            global_index[(3, POST_START_MASS_W1_PREFIX_PENALIZED)],
            prefix_selected_k3,
        ),
        (
            POST_START_MEAN_W2,
            mean_w2_genealogy_k0,
            mean_w2_temporal_k0,
            global_index[(3, POST_START_MEAN_W2)],
            mean_w2_selected_k3,
        ),
    ]

    def better_than(value: Optional[float], reference: Optional[float]) -> bool:
        return value is not None and reference is not None and value > reference

    if any(
        better_than(candidate_genealogy_k0["mean_delta_rotor_vs_F"], baseline_genealogy_k0["mean_delta_rotor_vs_F"])
        and (
            candidate_temporal_k0["mean_delta_rotor_vs_F"] is not None
            and post_start_temporal_k0["mean_delta_rotor_vs_F"] is not None
            and candidate_temporal_k0["mean_delta_rotor_vs_F"] >= post_start_temporal_k0["mean_delta_rotor_vs_F"]
        )
        and (candidate_global_k3["mean_delta_rotor_vs_F"] or float("-inf"))
        >= (global_index[(3, POST_START_MASS_W1)]["mean_delta_rotor_vs_F"] or float("-inf"))
        and selected_k3 >= post_start_selected_k3
        for _candidate_id, candidate_genealogy_k0, candidate_temporal_k0, candidate_global_k3, selected_k3 in candidates
    ):
        return (
            "refinement-still-live",
            "a post-start family variant improves genealogy k=0 over the baseline reader while at least "
            "matching the current post-start reader on temporal k=0, k=3 support, and selected wins.",
        )

    if any(
        better_than(candidate_genealogy_k0["mean_delta_rotor_vs_F"], baseline_genealogy_k0["mean_delta_rotor_vs_F"])
        for _candidate_id, candidate_genealogy_k0, _candidate_temporal_k0, _candidate_global_k3, _selected_k3 in candidates
    ):
        return (
            "needs-genealogy-specific-reader",
            "at least one post-start family variant helps genealogy k=0 versus the baseline reader, but only by "
            "paying back the gain in temporal k=0, k=3 support, or selected wins. The next step should stay "
            "genealogy-specific rather than promoting a global reader.",
        )

    return (
        "post-start-family-not-salvageable",
        "none of the tested post-start family variants improves genealogy k=0 over the baseline reader without "
        "breaking temporal k=0 or k=3 support. The family does not currently justify further broad refinement.",
    )


def write_report(
    out_path: Path,
    manifest: Dict[str, Any],
    global_rows: Sequence[Dict[str, Any]],
    world_rows: Sequence[Dict[str, Any]],
    selected_rows: Sequence[Dict[str, Any]],
    genealogy_failures: Sequence[Dict[str, Any]],
    temporal_side_effects: Sequence[Dict[str, Any]],
    decision: str,
    rationale: str,
) -> None:
    lines = [
        "# Gate5 Reader Failure Autopsy",
        "",
        f"Run ID: {manifest.get('run_id', '')}",
        f"Method ID: {manifest.get('method_id', '')}",
        "",
        "Fixed inputs:",
        "- boundary fixed to FWHT baseline",
        "- comparator fixed to `rotor_loop_chordal_v1`",
        "- focus reader family fixed to `post_start_mass_w1_ranknorm`",
        "- labels fixed to `k=0,3`",
        "",
        "Candidate definitions:",
        f"- `{candidate_label(BASELINE)}`: rank at `defect_start`",
        f"- `{candidate_label(POST_START_MASS_W1)}`: `rank(defect_start) + rank(defect_start+1)`",
        f"- `{candidate_label(POST_START_MASS_W1_PREFIX_PENALIZED)}`: `rank(defect_start) + rank(defect_start+1) - rank(defect_start-1)`",
        f"- `{candidate_label(POST_START_MEAN_W2)}`: `mean(rank(defect_start), rank(defect_start+1))`",
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

    lines.extend(["", "## Selected Existing Wins", ""])
    for row in selected_rows:
        lines.append(
            f"- sample_id={row['sample_id']} world_type={row['world_type']} "
            f"baseline_k0={render_float(row.get(f'{BASELINE}_k0_delta_rotor_vs_F'))} "
            f"post_start_k0={render_float(row.get(f'{POST_START_MASS_W1}_k0_delta_rotor_vs_F'))} "
            f"post_start_k3={render_float(row.get(f'{POST_START_MASS_W1}_k3_delta_rotor_vs_F'))}"
        )

    lines.extend(["", "## Genealogy Worst Failures (post_start_mass_w1_ranknorm, k=0)", ""])
    for row in genealogy_failures:
        lines.append(
            f"- sample_id={row['sample_id']} "
            f"baseline_k0={render_float(row.get(f'{BASELINE}_k0_delta_rotor_vs_F'))} "
            f"post_start_k0={render_float(row.get(f'{POST_START_MASS_W1}_k0_delta_rotor_vs_F'))} "
            f"prefix_penalized_k0={render_float(row.get(f'{POST_START_MASS_W1_PREFIX_PENALIZED}_k0_delta_rotor_vs_F'))} "
            f"mean_w2_k0={render_float(row.get(f'{POST_START_MEAN_W2}_k0_delta_rotor_vs_F'))}"
        )

    lines.extend(["", "## Temporal Side Effects (post_start_mass_w1_ranknorm vs baseline at k=0)", ""])
    for row in temporal_side_effects:
        lines.append(
            f"- sample_id={row['sample_id']} "
            f"baseline_k0={render_float(row.get(f'{BASELINE}_k0_delta_rotor_vs_F'))} "
            f"post_start_k0={render_float(row.get(f'{POST_START_MASS_W1}_k0_delta_rotor_vs_F'))} "
            f"gap={render_float(row.get('post_start_vs_baseline_delta_gap_k0'))} "
            f"post_start_k3={render_float(row.get(f'{POST_START_MASS_W1}_k3_delta_rotor_vs_F'))}"
        )

    lines.extend(["", "## Decision", "", f"- decision: `{decision}`", f"- {rationale}"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_decision(out_path: Path, decision: str, rationale: str, global_rows: Sequence[Dict[str, Any]], world_rows: Sequence[Dict[str, Any]]) -> None:
    global_index = {(int(row["k"]), str(row["candidate_id"])): row for row in global_rows}
    world_index = {
        (int(row["k"]), str(row["candidate_id"]), str(row["group_id"])): row
        for row in world_rows
    }
    lines = [
        "# Gate5 Reader Failure Autopsy Decision",
        "",
        f"- decision: `{decision}`",
        "",
        "## Key Basis",
        "",
        f"- baseline genealogy k0: {render_float(world_index[(0, BASELINE, 'genealogy')]['mean_delta_rotor_vs_F'])}",
        f"- post_start_mass_w1 genealogy k0: {render_float(world_index[(0, POST_START_MASS_W1, 'genealogy')]['mean_delta_rotor_vs_F'])}",
        f"- prefix_penalized genealogy k0: {render_float(world_index[(0, POST_START_MASS_W1_PREFIX_PENALIZED, 'genealogy')]['mean_delta_rotor_vs_F'])}",
        f"- mean_w2 genealogy k0: {render_float(world_index[(0, POST_START_MEAN_W2, 'genealogy')]['mean_delta_rotor_vs_F'])}",
        f"- post_start_mass_w1 temporal k0: {render_float(world_index[(0, POST_START_MASS_W1, 'temporal')]['mean_delta_rotor_vs_F'])}",
        f"- post_start_mass_w1 k3 selected wins: {render_float(global_index[(3, POST_START_MASS_W1)]['selected_win_preservation_rate'])}",
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
        raise ValueError("--k-values must be exactly 0 and 3 for reader failure autopsy")

    token_rows = read_csv(gate5_out_dir / "gate5_token_telemetry.csv")
    sample_rows = read_csv(gate5_out_dir / "gate5_sample_summary.csv")
    manifest = json.loads((gate5_out_dir / "manifest.json").read_text(encoding="utf-8"))
    token_grouped = group_token_rows(token_rows)

    frustrated_rows = [row for row in sample_rows if row.get("variant") == "frustrated"]
    if not frustrated_rows:
        raise ValueError("reader failure autopsy found zero frustrated samples")

    present_ids = {int(row["sample_id"]) for row in frustrated_rows}
    missing_ids = [sample_id for sample_id in args.sample_ids if sample_id not in present_ids]
    if missing_ids:
        raise ValueError(
            "requested sample_ids are not present in reader failure autopsy frustrated population: "
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

    wide_rows = build_wide_rows(sample_candidate_rows)
    selected_set = {int(sample_id) for sample_id in args.sample_ids}
    selected_rows = [row for row in wide_rows if int(row["sample_id"]) in selected_set]
    selected_rows.sort(key=lambda row: args.sample_ids.index(int(row["sample_id"])))

    genealogy_failures = [
        row
        for row in wide_rows
        if row["world_type"] == "genealogy" and row.get(f"{POST_START_MASS_W1}_k0_delta_rotor_vs_F") is not None
    ]
    genealogy_failures.sort(key=lambda row: row[f"{POST_START_MASS_W1}_k0_delta_rotor_vs_F"])
    genealogy_failures = genealogy_failures[: args.genealogy_worst_count]

    temporal_side_effects = [
        row
        for row in wide_rows
        if row["world_type"] == "temporal" and row.get("post_start_vs_baseline_delta_gap_k0") is not None
    ]
    temporal_side_effects.sort(key=lambda row: row["post_start_vs_baseline_delta_gap_k0"])
    temporal_side_effects = temporal_side_effects[: args.temporal_side_effect_count]

    decision, rationale = choose_decision(world_rows, global_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    sample_csv = out_dir / "gate5_reader_failure_autopsy_sample_summary.csv"
    world_csv = out_dir / "gate5_reader_failure_autopsy_world_summary.csv"
    selected_csv = out_dir / "gate5_reader_failure_autopsy_selected_wins.csv"
    genealogy_csv = out_dir / "gate5_reader_failure_autopsy_genealogy_cases.csv"
    temporal_csv = out_dir / "gate5_reader_failure_autopsy_temporal_cases.csv"
    report_md = out_dir / "gate5_reader_failure_autopsy_report.md"
    decision_md = out_dir / "gate5_reader_failure_autopsy_decision.md"

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

    wide_fieldnames = [
        "sample_id",
        "variant",
        "world_type",
        "defect_start_step_k0",
        "defect_end_step_k0",
        "defect_start_step_k3",
        "defect_end_step_k3",
        "post_start_vs_baseline_delta_gap_k0",
        "post_start_vs_baseline_delta_gap_k3",
    ]
    for candidate_id in CANDIDATE_ORDER:
        for k in (0, 3):
            wide_fieldnames.extend(
                [
                    f"{candidate_id}_k{k}_metric_F",
                    f"{candidate_id}_k{k}_metric_rotor",
                    f"{candidate_id}_k{k}_delta_rotor_vs_F",
                    f"{candidate_id}_k{k}_rotor_gt_F",
                ]
            )

    write_csv(selected_csv, wide_fieldnames, selected_rows)
    write_csv(genealogy_csv, wide_fieldnames, genealogy_failures)
    write_csv(temporal_csv, wide_fieldnames, temporal_side_effects)
    write_report(
        report_md,
        manifest,
        global_rows,
        world_rows,
        selected_rows,
        genealogy_failures,
        temporal_side_effects,
        decision,
        rationale,
    )
    write_decision(decision_md, decision, rationale, global_rows, world_rows)

    print(f"sample_summary_csv={sample_csv.as_posix()}")
    print(f"world_summary_csv={world_csv.as_posix()}")
    print(f"selected_wins_csv={selected_csv.as_posix()}")
    print(f"genealogy_cases_csv={genealogy_csv.as_posix()}")
    print(f"temporal_cases_csv={temporal_csv.as_posix()}")
    print(f"report_md={report_md.as_posix()}")
    print(f"decision_md={decision_md.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
