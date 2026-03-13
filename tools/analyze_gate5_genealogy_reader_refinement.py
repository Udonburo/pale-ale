#!/usr/bin/env python3
"""Compare genealogy-focused reader refinements on a fixed Gate5 FWHT baseline."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-12

BASELINE = "first_after_defect_score_ranknorm"
POST_START_MASS_W1 = "post_start_mass_w1_ranknorm"
POST_START_MEAN_W3 = "post_start_mean_w3_ranknorm"
BEFORE_PENALIZED_FIRST_AFTER_W1 = "before_penalized_first_after_w1_ranknorm"

CANDIDATE_ORDER = [
    BASELINE,
    POST_START_MASS_W1,
    POST_START_MEAN_W3,
    BEFORE_PENALIZED_FIRST_AFTER_W1,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare genealogy-focused reader refinements on an existing Gate5 FWHT baseline "
            "run without changing the boundary or comparator."
        )
    )
    parser.add_argument("--gate5-out-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sample-ids", nargs="+", type=int, default=[137, 147, 149, 11, 167])
    parser.add_argument("--k-values", nargs="+", type=int, default=[0, 3])
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
        average_rank = ((cursor + 1) + next_cursor) / 2.0
        normalized = 0.0 if total == 1 else (average_rank - 1.0) / float(total - 1)
        for tied_idx in range(cursor, next_cursor):
            ranks[sorted_present[tied_idx][0]] = normalized
        cursor = next_cursor

    out: List[Optional[float]] = [None] * len(scores)
    for idx, _score in present:
        out[idx] = ranks[idx]
    return out


def compute_candidate_values(labels: Sequence[int], scores: Sequence[Optional[float]]) -> Dict[str, Optional[float]]:
    start, end = defect_span(labels)
    if start is None or end is None:
        return {
            BASELINE: None,
            POST_START_MASS_W1: None,
            POST_START_MEAN_W3: None,
            BEFORE_PENALIZED_FIRST_AFTER_W1: None,
        }
    rank_scores = rank_normalize_scores(scores)
    baseline = rank_scores[start]
    post_start_mass_w1 = sum_window(rank_scores, start, start + 1)
    post_start_mean_w3 = mean_window(rank_scores, start, start + 2)
    before_penalized_first_after_w1 = None
    if baseline is not None:
        before_w1 = sum_window(rank_scores, start - 1, start - 1)
        before_penalized_first_after_w1 = baseline - before_w1
    return {
        BASELINE: baseline,
        POST_START_MASS_W1: post_start_mass_w1,
        POST_START_MEAN_W3: post_start_mean_w3,
        BEFORE_PENALIZED_FIRST_AFTER_W1: before_penalized_first_after_w1,
    }


def candidate_label(candidate_id: str) -> str:
    labels = {
        BASELINE: "first_after_defect_score_ranknorm",
        POST_START_MASS_W1: "post_start_mass_w1_ranknorm",
        POST_START_MEAN_W3: "post_start_mean_w3_ranknorm",
        BEFORE_PENALIZED_FIRST_AFTER_W1: "before_penalized_first_after_w1_ranknorm",
    }
    return labels[candidate_id]


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

    out: List[Dict[str, Any]] = []
    for candidate_id in CANDIDATE_ORDER:
        metric_f = values_f[candidate_id]
        metric_rotor = values_rotor[candidate_id]
        out.append(
            {
                "k": k,
                "candidate_id": candidate_id,
                "sample_id": int(sample_summary["sample_id"]),
                "variant": sample_summary.get("variant", ""),
                "world_type": sample_summary.get("world_type", ""),
                "positive_token_count_dilated": sum(labels),
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
    return out


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


def render_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def choose_decision(global_rows: Sequence[Dict[str, Any]], world_rows: Sequence[Dict[str, Any]]) -> Tuple[str, str]:
    global_index = {(int(row["k"]), str(row["candidate_id"])): row for row in global_rows}
    world_index = {
        (int(row["k"]), str(row["candidate_id"]), str(row["group_id"])): row for row in world_rows
    }

    baseline_genealogy_k0 = world_index[(0, BASELINE, "genealogy")]
    baseline_genealogy_k3 = world_index[(3, BASELINE, "genealogy")]
    baseline_reachability_k3 = world_index[(3, BASELINE, "reachability")]
    baseline_temporal_k3 = world_index[(3, BASELINE, "temporal")]
    baseline_selected_k3 = global_index[(3, BASELINE)]["selected_win_preservation_rate"] or 0.0

    best_candidate = None
    best_score = float("-inf")
    best_reason = ""
    for candidate_id in CANDIDATE_ORDER:
        if candidate_id == BASELINE:
            continue
        g0 = world_index[(0, candidate_id, "genealogy")]
        g3 = world_index[(3, candidate_id, "genealogy")]
        r3 = world_index[(3, candidate_id, "reachability")]
        t3 = world_index[(3, candidate_id, "temporal")]
        global3 = global_index[(3, candidate_id)]
        score = 0.0
        if (g0["mean_delta_rotor_vs_F"] or float("-inf")) > (baseline_genealogy_k0["mean_delta_rotor_vs_F"] or float("-inf")):
            score += 1.0
        if (g3["mean_delta_rotor_vs_F"] or float("-inf")) >= (baseline_genealogy_k3["mean_delta_rotor_vs_F"] or float("-inf")):
            score += 1.0
        if (r3["mean_delta_rotor_vs_F"] or float("-inf")) >= (baseline_reachability_k3["mean_delta_rotor_vs_F"] or float("-inf")):
            score += 1.0
        if (t3["mean_delta_rotor_vs_F"] or float("-inf")) >= (baseline_temporal_k3["mean_delta_rotor_vs_F"] or float("-inf")):
            score += 1.0
        if (global3["selected_win_preservation_rate"] or 0.0) >= baseline_selected_k3:
            score += 1.0
        if score > best_score:
            best_score = score
            best_candidate = candidate_id
            best_reason = (
                f"{candidate_label(candidate_id)} score={score:.1f} "
                f"genealogy_k0={render_float(g0['mean_delta_rotor_vs_F'])} "
                f"genealogy_k3={render_float(g3['mean_delta_rotor_vs_F'])} "
                f"reachability_k3={render_float(r3['mean_delta_rotor_vs_F'])} "
                f"temporal_k3={render_float(t3['mean_delta_rotor_vs_F'])} "
                f"selected_k3={render_float(global3['selected_win_preservation_rate'])}"
            )

    if best_candidate is None:
        return ("no-clear-candidate", "no non-baseline genealogy refinement candidate was evaluated")
    if best_score >= 4.0:
        return (
            "genealogy-refinement-still-live",
            best_reason
            + " It improves genealogy at k=0 and/or k=3 without breaking reachability/temporal k=3 or selected-case preservation.",
        )
    return (
        "no-clear-candidate",
        best_reason
        + " It does not improve genealogy enough without paying back the gain in reachability/temporal or selected wins.",
    )


def write_report(
    out_path: Path,
    manifest: Dict[str, Any],
    global_rows: Sequence[Dict[str, Any]],
    world_rows: Sequence[Dict[str, Any]],
    selected_rows: Sequence[Dict[str, Any]],
    decision: str,
    rationale: str,
) -> None:
    lines = [
        "# Gate5 Genealogy Reader Refinement",
        "",
        f"Run ID: {manifest.get('run_id', '')}",
        f"Method ID: {manifest.get('method_id', '')}",
        "",
        "Fixed inputs:",
        "- FWHT baseline only",
        "- rotor comparator fixed to `rotor_loop_chordal_v1`",
        "- genealogy-focused reader refinement only",
        "- label diagnostics fixed to `k=0,3`",
        "",
        "Candidate definitions:",
        "- `first_after_defect_score_ranknorm`: rank-normalized score at defect start",
        "- `post_start_mass_w1_ranknorm`: sum over `[defect_start, defect_start+1]`",
        "- `post_start_mean_w3_ranknorm`: mean over `[defect_start, defect_start+2]`",
        "- `before_penalized_first_after_w1_ranknorm`: `first_after - previous_step`",
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
    current_k = None
    current_candidate = None
    for row in selected_rows:
        if current_k != row["k"] or current_candidate != row["candidate_id"]:
            current_k = row["k"]
            current_candidate = row["candidate_id"]
            lines.extend([f"### k={current_k} candidate={candidate_label(str(current_candidate))}", ""])
        lines.append(
            f"- sample_id={row['sample_id']} world_type={row['world_type']} "
            f"metric_F={render_float(row['metric_F'])} metric_rotor={render_float(row['metric_rotor'])} "
            f"delta={render_float(row['delta_rotor_vs_F'])} rotor_gt_F={row['rotor_gt_F']}"
        )

    lines.extend(["", "## Decision", "", f"- decision: `{decision}`", f"- {rationale}"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_decision(out_path: Path, decision: str, rationale: str, global_rows: Sequence[Dict[str, Any]]) -> None:
    global_index = {(int(row["k"]), str(row["candidate_id"])): row for row in global_rows}
    lines = [
        "# Gate5 Genealogy Reader Refinement Decision",
        "",
        f"- decision: `{decision}`",
        "",
        "## Baseline Reader",
        "",
        f"- k=0 baseline mean_delta_rotor_vs_F: {render_float(global_index[(0, BASELINE)]['mean_delta_rotor_vs_F'])}",
        f"- k=3 baseline mean_delta_rotor_vs_F: {render_float(global_index[(3, BASELINE)]['mean_delta_rotor_vs_F'])}",
        f"- k=3 baseline selected_win_preservation_rate: {render_float(global_index[(3, BASELINE)]['selected_win_preservation_rate'])}",
        "",
        "## Decision Basis",
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
        raise ValueError("--k-values must be exactly 0 and 3 for genealogy reader refinement")

    token_rows = read_csv(gate5_out_dir / "gate5_token_telemetry.csv")
    sample_rows = read_csv(gate5_out_dir / "gate5_sample_summary.csv")
    manifest = json.loads((gate5_out_dir / "manifest.json").read_text(encoding="utf-8"))
    token_grouped = group_token_rows(token_rows)

    frustrated_rows = [row for row in sample_rows if row.get("variant") == "frustrated"]
    if not frustrated_rows:
        raise ValueError("genealogy reader refinement found zero frustrated samples")

    present_ids = {int(row["sample_id"]) for row in frustrated_rows}
    missing_ids = [sample_id for sample_id in args.sample_ids if sample_id not in present_ids]
    if missing_ids:
        raise ValueError(
            "requested sample_ids are not present in genealogy reader refinement frustrated population: "
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

    selected_rows = [row for row in sample_candidate_rows if int(row["sample_id"]) in set(args.sample_ids)]
    selected_rows.sort(
        key=lambda row: (
            int(row["k"]),
            CANDIDATE_ORDER.index(str(row["candidate_id"])),
            args.sample_ids.index(int(row["sample_id"])),
        )
    )

    decision, rationale = choose_decision(global_rows, world_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    sample_csv = out_dir / "gate5_genealogy_reader_refinement_sample_summary.csv"
    world_csv = out_dir / "gate5_genealogy_reader_refinement_world_summary.csv"
    selected_csv = out_dir / "gate5_genealogy_reader_refinement_selected_cases.csv"
    report_md = out_dir / "gate5_genealogy_reader_refinement_report.md"
    decision_md = out_dir / "gate5_genealogy_reader_refinement_decision.md"

    write_csv(
        sample_csv,
        [
            "k",
            "candidate_id",
            "sample_id",
            "variant",
            "world_type",
            "positive_token_count_dilated",
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
    write_csv(
        selected_csv,
        [
            "k",
            "candidate_id",
            "sample_id",
            "variant",
            "world_type",
            "positive_token_count_dilated",
            "defect_start_step_dilated",
            "defect_end_step_dilated",
            "metric_F",
            "metric_rotor",
            "delta_rotor_vs_F",
            "rotor_gt_F",
        ],
        selected_rows,
    )
    write_report(report_md, manifest, global_rows, world_rows, selected_rows, decision, rationale)
    write_decision(decision_md, decision, rationale, global_rows)

    print(f"sample_summary_csv={sample_csv.as_posix()}")
    print(f"world_summary_csv={world_csv.as_posix()}")
    print(f"selected_cases_csv={selected_csv.as_posix()}")
    print(f"report_md={report_md.as_posix()}")
    print(f"decision_md={decision_md.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
