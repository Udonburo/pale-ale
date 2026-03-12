#!/usr/bin/env python3
"""Compare field-side aggregator candidates on a fixed Gate5 FWHT baseline."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-12

RAW_TOKEN = "raw_token_auprc"
FIRST_AFTER_RANK = "first_after_defect_score_ranknorm"
INSIDE_AFTER = "inside_to_after_ratio"
PREFIX_PENALIZED_W1_RANK = "prefix_penalized_inside_mass_w1_ranknorm"
PREFIX_PENALIZED_W3_RANK = "prefix_penalized_inside_mass_w3_ranknorm"

CANDIDATE_ORDER = [
    RAW_TOKEN,
    INSIDE_AFTER,
    FIRST_AFTER_RANK,
    PREFIX_PENALIZED_W1_RANK,
    PREFIX_PENALIZED_W3_RANK,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate field-side aggregator candidates on an existing Gate5 CFA run "
            "without changing the boundary or comparator."
        )
    )
    parser.add_argument("--gate5-out-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--k-values", nargs="+", type=int, default=[0, 3])
    parser.add_argument("--sample-ids", nargs="+", type=int, default=[137, 147, 149, 11, 167])
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
    positive_steps = [idx for idx, label in enumerate(labels) if label == 1]
    if not positive_steps:
        return (None, None)
    return (positive_steps[0], positive_steps[-1])


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


def average_precision(labels: Sequence[int], scores: Sequence[Optional[float]]) -> Optional[float]:
    filtered = [(idx, float(score)) for idx, score in enumerate(scores) if score is not None]
    if not filtered:
        return None
    n_pos = sum(labels[idx] for idx, _ in filtered)
    if n_pos == 0:
        return None
    filtered.sort(key=lambda item: (-item[1], item[0]))
    tp = 0
    fp = 0
    prev_recall = 0.0
    ap = 0.0
    for idx, _score in filtered:
        if labels[idx] == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / float(n_pos)
        precision = tp / float(tp + fp)
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return ap


def rank_normalize_scores(scores: Sequence[Optional[float]]) -> List[Optional[float]]:
    present = [(idx, float(score)) for idx, score in enumerate(scores) if score is not None]
    if not present:
        return [None] * len(scores)

    sorted_present = sorted(present, key=lambda item: (item[1], item[0]))
    ranks: Dict[int, float] = {}
    cursor = 0
    total = len(sorted_present)
    while cursor < total:
        next_cursor = cursor + 1
        while next_cursor < total and sorted_present[next_cursor][1] == sorted_present[cursor][1]:
            next_cursor += 1
        # Average rank for tied values, then scale to [0, 1].
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
            RAW_TOKEN: None,
            INSIDE_AFTER: None,
            FIRST_AFTER_RANK: None,
            PREFIX_PENALIZED_W1_RANK: None,
            PREFIX_PENALIZED_W3_RANK: None,
        }
    rank_scores = rank_normalize_scores(scores)
    inside = sum_window(scores, start, end)
    after = sum_window(scores, end + 1, len(scores) - 1)
    rank_inside = sum_window(rank_scores, start, end)
    rank_prefix_w1 = sum_window(rank_scores, start - 1, start - 1)
    rank_prefix_w3 = sum_window(rank_scores, start - 3, start - 1)
    return {
        RAW_TOKEN: average_precision(labels, scores),
        INSIDE_AFTER: inside / max(after, EPS),
        FIRST_AFTER_RANK: rank_scores[start] if rank_scores[start] is not None else None,
        PREFIX_PENALIZED_W1_RANK: rank_inside - rank_prefix_w1,
        PREFIX_PENALIZED_W3_RANK: rank_inside - rank_prefix_w3,
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
    candidate_values_f = compute_candidate_values(labels, scores_f)
    candidate_values_rotor = compute_candidate_values(labels, scores_rotor)

    out: List[Dict[str, Any]] = []
    for candidate_id in CANDIDATE_ORDER:
        f_value = candidate_values_f[candidate_id]
        rotor_value = candidate_values_rotor[candidate_id]
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
                "metric_F": f_value,
                "metric_rotor": rotor_value,
                "delta_rotor_vs_F": None if f_value is None or rotor_value is None else rotor_value - f_value,
                "rotor_gt_F": (
                    None if f_value is None or rotor_value is None else (1 if rotor_value > f_value else 0)
                ),
            }
        )
    return out


def summarize_rows(
    group_id: str,
    rows: Sequence[Dict[str, Any]],
    selected_ids: Sequence[int],
) -> Dict[str, Any]:
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


def candidate_label(candidate_id: str) -> str:
    labels = {
        RAW_TOKEN: "raw_token_rotor",
        INSIDE_AFTER: "inside_to_after_ratio",
        FIRST_AFTER_RANK: "first_after_defect_score_ranknorm",
        PREFIX_PENALIZED_W1_RANK: "prefix_penalized_inside_mass_w1_ranknorm",
        PREFIX_PENALIZED_W3_RANK: "prefix_penalized_inside_mass_w3_ranknorm",
    }
    return labels[candidate_id]


def write_report(
    out_path: Path,
    manifest: Dict[str, Any],
    global_rows: Sequence[Dict[str, Any]],
    world_rows: Sequence[Dict[str, Any]],
    genealogy_rows: Sequence[Dict[str, Any]],
    selected_rows: Sequence[Dict[str, Any]],
) -> None:
    lines = [
        "# Gate5 Aggregation Candidates",
        "",
        f"Run ID: {manifest.get('run_id', '')}",
        f"Method ID: {manifest.get('method_id', '')}",
        "",
        "Fixed inputs:",
        "- FWHT baseline only",
        "- rotor comparator fixed to `rotor_loop_chordal_v1`",
        "- label diagnostics fixed to `k=0,3`",
        "- scale-free comparison: ratio candidate uses raw scores; scalar and mass candidates use sample-wise rank-normalized scores",
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

    lines.extend(
        [
            "",
            "## Genealogy Summary",
            "",
            "| k | candidate | mean_delta_rotor_vs_F | rotor_win_rate | selected_win_preservation_rate |",
            "| ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in genealogy_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["k"]),
                    candidate_label(str(row["candidate_id"])),
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
            lines.extend(
                [
                    f"### k={current_k} candidate={candidate_label(str(current_candidate))}",
                    "",
                ]
            )
        lines.append(
            f"- sample_id={row['sample_id']} world_type={row['world_type']} "
            f"metric_F={render_float(row['metric_F'])} metric_rotor={render_float(row['metric_rotor'])} "
            f"delta={render_float(row['delta_rotor_vs_F'])} rotor_gt_F={row['rotor_gt_F']}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def choose_decision(
    global_rows: Sequence[Dict[str, Any]],
    world_rows: Sequence[Dict[str, Any]],
    selected_rows: Sequence[Dict[str, Any]],
) -> Tuple[str, str]:
    global_index = {
        (int(row["k"]), str(row["candidate_id"])): row for row in global_rows
    }
    world_index = {
        (int(row["k"]), str(row["candidate_id"]), str(row["group_id"])): row for row in world_rows
    }

    raw_k0 = global_index[(0, RAW_TOKEN)]["mean_delta_rotor_vs_F"] or 0.0
    raw_k3 = global_index[(3, RAW_TOKEN)]["mean_delta_rotor_vs_F"] or 0.0
    raw_gap = abs(raw_k3 - raw_k0)
    raw_genealogy = world_index[(3, RAW_TOKEN, "genealogy")]["mean_delta_rotor_vs_F"] or float("-inf")
    raw_reachability = world_index[(3, RAW_TOKEN, "reachability")]["mean_delta_rotor_vs_F"] or float("-inf")
    raw_temporal = world_index[(3, RAW_TOKEN, "temporal")]["mean_delta_rotor_vs_F"] or float("-inf")
    raw_selected = global_index[(3, RAW_TOKEN)]["selected_win_preservation_rate"] or 0.0

    promotion_reason = ""
    strongest_k3_candidate = None
    strongest_k3_delta = float("-inf")
    strongest_k3_reason = ""
    for candidate_id in CANDIDATE_ORDER:
        if candidate_id == RAW_TOKEN:
            continue
        k0 = global_index[(0, candidate_id)]
        k3 = global_index[(3, candidate_id)]
        reach3 = world_index[(3, candidate_id, "reachability")]
        temp3 = world_index[(3, candidate_id, "temporal")]
        genealogy3 = world_index[(3, candidate_id, "genealogy")]
        k0_delta = k0["mean_delta_rotor_vs_F"] or 0.0
        k3_delta = k3["mean_delta_rotor_vs_F"] or 0.0
        gap = abs(k3_delta - k0_delta)
        k0_selected = k0["selected_win_preservation_rate"] or 0.0
        k3_selected = k3["selected_win_preservation_rate"] or 0.0
        k0_win = k0["rotor_win_rate"] or 0.0
        k3_win = k3["rotor_win_rate"] or 0.0
        reach3_delta = reach3["mean_delta_rotor_vs_F"] or float("-inf")
        temp3_delta = temp3["mean_delta_rotor_vs_F"] or float("-inf")
        genealogy3_delta = genealogy3["mean_delta_rotor_vs_F"] or float("-inf")

        if k3_delta > strongest_k3_delta:
            strongest_k3_delta = k3_delta
            strongest_k3_candidate = candidate_id
            strongest_k3_reason = (
                f"{candidate_label(candidate_id)} is the strongest k=3 non-baseline reader "
                f"(k3_delta={render_float(k3_delta)}, k0_delta={render_float(k0_delta)}, "
                f"reachability_k3={render_float(reach3_delta)}, temporal_k3={render_float(temp3_delta)}, "
                f"genealogy_k3={render_float(genealogy3_delta)}, gap={render_float(gap)}, "
                f"selected_k3={render_float(k3_selected)})."
            )

        if (
            k3_delta > raw_k3
            and k0_delta >= raw_k0
            and reach3_delta >= raw_reachability
            and temp3_delta >= raw_temporal
            and genealogy3_delta >= raw_genealogy
            and k3_selected >= raw_selected
            and gap <= raw_gap
        ):
            promotion_reason = (
                f"{candidate_label(candidate_id)} improves raw rotor at both k=0 ({render_float(k0_delta)} vs "
                f"{render_float(raw_k0)}) and k=3 ({render_float(k3_delta)} vs {render_float(raw_k3)}), "
                f"preserves selected-case wins ({render_float(k3_selected)}), does not worsen genealogy "
                f"({render_float(genealogy3_delta)} vs {render_float(raw_genealogy)}), and does not widen the "
                f"k-gap ({render_float(gap)} vs raw {render_float(raw_gap)})."
            )
            return ("aggregation-first", promotion_reason)

    if strongest_k3_candidate is None:
        return ("no-clear-candidate", "no non-baseline aggregation candidate was evaluated")
    return (
        "no-clear-candidate",
        strongest_k3_reason
        + " It improves late-dilated reading but does not satisfy the full promotion rule "
        + "(raw-k0 parity and/or gap shrink).",
    )


def write_decision(
    out_path: Path,
    decision: str,
    rationale: str,
    global_rows: Sequence[Dict[str, Any]],
) -> None:
    global_index = {(int(row["k"]), str(row["candidate_id"])): row for row in global_rows}
    lines = [
        "# Gate5 Aggregation Candidate Decision",
        "",
        f"- decision: `{decision}`",
        "",
        "## Raw Baseline",
        "",
        f"- k=0 raw_token_rotor mean_delta_rotor_vs_F: {render_float(global_index[(0, RAW_TOKEN)]['mean_delta_rotor_vs_F'])}",
        f"- k=3 raw_token_rotor mean_delta_rotor_vs_F: {render_float(global_index[(3, RAW_TOKEN)]['mean_delta_rotor_vs_F'])}",
        f"- k=0 raw_token_rotor selected_win_preservation_rate: {render_float(global_index[(0, RAW_TOKEN)]['selected_win_preservation_rate'])}",
        f"- k=3 raw_token_rotor selected_win_preservation_rate: {render_float(global_index[(3, RAW_TOKEN)]['selected_win_preservation_rate'])}",
        "",
        "## Decision Basis",
        "",
        f"- {rationale}",
        "",
        "This is an aggregation-reader comparison only. Boundary and residual remain fixed.",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    gate5_out_dir = (REPO_ROOT / args.gate5_out_dir).resolve()
    out_dir = (REPO_ROOT / args.out_dir).resolve()

    token_rows = read_csv(gate5_out_dir / "gate5_token_telemetry.csv")
    sample_rows = read_csv(gate5_out_dir / "gate5_sample_summary.csv")
    manifest = json.loads((gate5_out_dir / "manifest.json").read_text(encoding="utf-8"))
    token_grouped = group_token_rows(token_rows)

    if set(args.k_values) != {0, 3}:
        raise ValueError("--k-values must be exactly 0 and 3 for aggregation candidate comparison")

    frustrated_rows = [row for row in sample_rows if row.get("variant") == "frustrated"]
    if not frustrated_rows:
        raise ValueError("aggregation candidate analysis found zero frustrated samples")

    sample_ids_present = {int(row["sample_id"]) for row in frustrated_rows}
    missing_ids = [sample_id for sample_id in args.sample_ids if sample_id not in sample_ids_present]
    if missing_ids:
        raise ValueError(
            "requested sample_ids are not present in frustrated aggregation-candidate population: "
            + ",".join(str(sample_id) for sample_id in missing_ids)
        )

    sample_candidate_rows: List[Dict[str, Any]] = []
    for summary_row in frustrated_rows:
        sample_id = int(summary_row["sample_id"])
        token_sample_rows = token_grouped[sample_id]
        for k in args.k_values:
            sample_candidate_rows.extend(build_sample_rows(summary_row, token_sample_rows, k))

    sample_candidate_rows.sort(
        key=lambda row: (int(row["k"]), CANDIDATE_ORDER.index(str(row["candidate_id"])), int(row["sample_id"]))
    )

    global_rows: List[Dict[str, Any]] = []
    world_rows: List[Dict[str, Any]] = []
    genealogy_rows: List[Dict[str, Any]] = []
    for k in sorted(args.k_values):
        k_rows = [row for row in sample_candidate_rows if int(row["k"]) == k]
        for candidate_id in CANDIDATE_ORDER:
            candidate_rows = [row for row in k_rows if row["candidate_id"] == candidate_id]
            global_row = summarize_rows(f"all", candidate_rows, args.sample_ids)
            global_row["k"] = k
            global_row["candidate_id"] = candidate_id
            global_rows.append(global_row)
            for world_type in sorted({str(row["world_type"]) for row in candidate_rows}):
                grouped = [row for row in candidate_rows if row["world_type"] == world_type]
                world_row = summarize_rows(world_type, grouped, args.sample_ids)
                world_row["k"] = k
                world_row["candidate_id"] = candidate_id
                world_rows.append(world_row)
                if world_type == "genealogy":
                    genealogy_rows.append(world_row)

    selected_rows = [
        row for row in sample_candidate_rows if int(row["sample_id"]) in set(args.sample_ids)
    ]
    selected_rows.sort(
        key=lambda row: (
            int(row["k"]),
            CANDIDATE_ORDER.index(str(row["candidate_id"])),
            args.sample_ids.index(int(row["sample_id"])),
        )
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    global_csv = out_dir / "gate5_aggregation_candidates_global_summary.csv"
    world_csv = out_dir / "gate5_aggregation_candidates_world_summary.csv"
    genealogy_csv = out_dir / "gate5_aggregation_candidates_genealogy_summary.csv"
    selected_csv = out_dir / "gate5_aggregation_candidates_selected_cases.csv"
    report_md = out_dir / "gate5_aggregation_candidates_report.md"
    decision_md = out_dir / "gate5_aggregation_candidates_decision.md"

    common_fields = [
        "k",
        "candidate_id",
        "group_id",
        "n_samples",
        "mean_metric_F",
        "mean_metric_rotor",
        "mean_delta_rotor_vs_F",
        "rotor_win_rate",
        "selected_win_preservation_rate",
    ]
    write_csv(global_csv, common_fields, global_rows)
    write_csv(world_csv, common_fields, world_rows)
    write_csv(genealogy_csv, common_fields, genealogy_rows)
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
    write_report(report_md, manifest, global_rows, world_rows, genealogy_rows, selected_rows)
    decision, rationale = choose_decision(global_rows, world_rows, selected_rows)
    write_decision(decision_md, decision, rationale, global_rows)

    print(f"global_summary_csv={global_csv.as_posix()}")
    print(f"world_summary_csv={world_csv.as_posix()}")
    print(f"genealogy_summary_csv={genealogy_csv.as_posix()}")
    print(f"selected_cases_csv={selected_csv.as_posix()}")
    print(f"report_md={report_md.as_posix()}")
    print(f"decision_md={decision_md.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
