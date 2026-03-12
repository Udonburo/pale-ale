#!/usr/bin/env python3
"""Diagnose why first_after_defect_score_ranknorm is strong at k=3 but weak at k=0."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the failure mode of first_after_defect_score_ranknorm on an existing "
            "Gate5 FWHT baseline run using k=0 and k=3 only."
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


def sign_of(value: Optional[float]) -> int:
    if value is None:
        return 0
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def compute_k_metrics(
    labels: Sequence[int],
    scores_f: Sequence[Optional[float]],
    scores_rotor: Sequence[Optional[float]],
) -> Dict[str, Any]:
    start, end = defect_span(labels)
    if start is None or end is None:
        raise ValueError("aggregation failure-mode analysis requires a labeled defect span")
    rank_f = rank_normalize_scores(scores_f)
    rank_rotor = rank_normalize_scores(scores_rotor)
    raw_token_ap_f = average_precision(labels, scores_f)
    raw_token_ap_rotor = average_precision(labels, scores_rotor)
    raw_first_after_f = scores_f[start]
    raw_first_after_rotor = scores_rotor[start]
    rank_first_after_f = rank_f[start]
    rank_first_after_rotor = rank_rotor[start]
    return {
        "defect_start_step": start,
        "defect_end_step": end,
        "raw_token_ap_F": raw_token_ap_f,
        "raw_token_ap_rotor": raw_token_ap_rotor,
        "raw_token_ap_delta_rotor_vs_F": (
            None
            if raw_token_ap_f is None or raw_token_ap_rotor is None
            else raw_token_ap_rotor - raw_token_ap_f
        ),
        "raw_first_after_F": raw_first_after_f,
        "raw_first_after_rotor": raw_first_after_rotor,
        "rank_first_after_F": rank_first_after_f,
        "rank_first_after_rotor": rank_first_after_rotor,
        "rank_first_after_delta_rotor_vs_F": (
            None
            if rank_first_after_f is None or rank_first_after_rotor is None
            else rank_first_after_rotor - rank_first_after_f
        ),
        "rank_first_after_rotor_gt_F": (
            None
            if rank_first_after_f is None or rank_first_after_rotor is None
            else (1 if rank_first_after_rotor > rank_first_after_f else 0)
        ),
    }


def build_sample_row(sample_summary: Dict[str, str], token_rows: Sequence[Dict[str, str]]) -> Dict[str, Any]:
    base_labels = [1 if row.get("label_token") == "1" else 0 for row in token_rows]
    scores_f = [parse_float(row.get("score_F_loop")) for row in token_rows]
    scores_rotor = [parse_float(row.get("rotor_loop_chordal_v1")) for row in token_rows]
    metrics_by_k: Dict[int, Dict[str, Any]] = {}
    for k in (0, 3):
        metrics_by_k[k] = compute_k_metrics(dilate_labels(base_labels, k), scores_f, scores_rotor)

    k0 = metrics_by_k[0]
    k3 = metrics_by_k[3]
    raw_k0 = k0["raw_token_ap_delta_rotor_vs_F"]
    raw_k3 = k3["raw_token_ap_delta_rotor_vs_F"]
    rank_k0 = k0["rank_first_after_delta_rotor_vs_F"]
    rank_k3 = k3["rank_first_after_delta_rotor_vs_F"]

    row: Dict[str, Any] = {
        "sample_id": int(sample_summary["sample_id"]),
        "variant": sample_summary.get("variant", ""),
        "world_type": sample_summary.get("world_type", ""),
        "positive_token_count": int(sample_summary["positive_token_count"]),
        "delta_auprc_rotor_vs_F": parse_float(sample_summary.get("delta_auprc_rotor_loop_chordal_v1_vs_F")),
        "auprc_F": parse_float(sample_summary.get("auprc_F")),
        "auprc_rotor": parse_float(sample_summary.get("auprc_rotor_loop_chordal_v1")),
    }
    for k in (0, 3):
        metrics = metrics_by_k[k]
        for field_name, value in metrics.items():
            row[f"{field_name}_k{k}"] = value

    row["defect_start_shift_k3_minus_k0"] = (
        None
        if k0["defect_start_step"] is None or k3["defect_start_step"] is None
        else k3["defect_start_step"] - k0["defect_start_step"]
    )
    row["rank_first_after_lift_F"] = None if k0["rank_first_after_F"] is None or k3["rank_first_after_F"] is None else k3["rank_first_after_F"] - k0["rank_first_after_F"]
    row["rank_first_after_lift_rotor"] = None if k0["rank_first_after_rotor"] is None or k3["rank_first_after_rotor"] is None else k3["rank_first_after_rotor"] - k0["rank_first_after_rotor"]
    row["raw_first_after_lift_F"] = None if k0["raw_first_after_F"] is None or k3["raw_first_after_F"] is None else k3["raw_first_after_F"] - k0["raw_first_after_F"]
    row["raw_first_after_lift_rotor"] = None if k0["raw_first_after_rotor"] is None or k3["raw_first_after_rotor"] is None else k3["raw_first_after_rotor"] - k0["raw_first_after_rotor"]
    row["rank_delta_lift_rotor_vs_F"] = None if rank_k0 is None or rank_k3 is None else rank_k3 - rank_k0
    row["raw_token_delta_lift_rotor_vs_F"] = None if raw_k0 is None or raw_k3 is None else raw_k3 - raw_k0
    row["ranknorm_sign_flip_between_k0_k3"] = 0 if sign_of(rank_k0) == sign_of(rank_k3) else 1
    row["label_mismatch_support"] = (
        1
        if (rank_k0 is not None and rank_k3 is not None and rank_k0 <= 0.0 and rank_k3 > 0.0 and (row["rank_first_after_lift_rotor"] or 0.0) > 0.0)
        else 0
    )
    row["ranknorm_artifact_risk"] = (
        1
        if (
            rank_k3 is not None
            and rank_k3 > 0.0
            and row["rank_first_after_lift_rotor"] is not None
            and row["raw_first_after_lift_rotor"] is not None
            and row["rank_first_after_lift_rotor"] > 0.0
            and row["raw_first_after_lift_rotor"] <= 0.0
        )
        else 0
    )
    return row


def summarize_group(group_id: str, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "group_id": group_id,
        "n_samples": len(rows),
        "mean_delta_auprc_rotor_vs_F": mean(
            row["delta_auprc_rotor_vs_F"] for row in rows if row["delta_auprc_rotor_vs_F"] is not None
        ),
        "mean_raw_token_ap_delta_rotor_vs_F_k0": mean(
            row["raw_token_ap_delta_rotor_vs_F_k0"] for row in rows if row["raw_token_ap_delta_rotor_vs_F_k0"] is not None
        ),
        "mean_raw_token_ap_delta_rotor_vs_F_k3": mean(
            row["raw_token_ap_delta_rotor_vs_F_k3"] for row in rows if row["raw_token_ap_delta_rotor_vs_F_k3"] is not None
        ),
        "mean_rank_first_after_delta_rotor_vs_F_k0": mean(
            row["rank_first_after_delta_rotor_vs_F_k0"] for row in rows if row["rank_first_after_delta_rotor_vs_F_k0"] is not None
        ),
        "mean_rank_first_after_delta_rotor_vs_F_k3": mean(
            row["rank_first_after_delta_rotor_vs_F_k3"] for row in rows if row["rank_first_after_delta_rotor_vs_F_k3"] is not None
        ),
        "mean_rank_delta_lift_rotor_vs_F": mean(
            row["rank_delta_lift_rotor_vs_F"] for row in rows if row["rank_delta_lift_rotor_vs_F"] is not None
        ),
        "mean_rank_first_after_lift_F": mean(
            row["rank_first_after_lift_F"] for row in rows if row["rank_first_after_lift_F"] is not None
        ),
        "mean_rank_first_after_lift_rotor": mean(
            row["rank_first_after_lift_rotor"] for row in rows if row["rank_first_after_lift_rotor"] is not None
        ),
        "mean_raw_first_after_lift_F": mean(
            row["raw_first_after_lift_F"] for row in rows if row["raw_first_after_lift_F"] is not None
        ),
        "mean_raw_first_after_lift_rotor": mean(
            row["raw_first_after_lift_rotor"] for row in rows if row["raw_first_after_lift_rotor"] is not None
        ),
        "rank_first_after_win_rate_k0": mean(
            float(row["rank_first_after_rotor_gt_F_k0"]) for row in rows if row["rank_first_after_rotor_gt_F_k0"] is not None
        ),
        "rank_first_after_win_rate_k3": mean(
            float(row["rank_first_after_rotor_gt_F_k3"]) for row in rows if row["rank_first_after_rotor_gt_F_k3"] is not None
        ),
        "label_mismatch_support_rate": mean(float(row["label_mismatch_support"]) for row in rows),
        "ranknorm_artifact_risk_rate": mean(float(row["ranknorm_artifact_risk"]) for row in rows),
        "mean_defect_start_shift_k3_minus_k0": mean(
            float(row["defect_start_shift_k3_minus_k0"]) for row in rows if row["defect_start_shift_k3_minus_k0"] is not None
        ),
    }
    return summary


def render_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def write_report(
    out_path: Path,
    manifest: Dict[str, Any],
    all_summary: Dict[str, Any],
    world_rows: Sequence[Dict[str, Any]],
    selected_rows: Sequence[Dict[str, Any]],
    decision: str,
    rationale: str,
) -> None:
    lines = [
        "# Gate5 Aggregation Failure-Mode Analysis",
        "",
        f"Run ID: {manifest.get('run_id', '')}",
        f"Method ID: {manifest.get('method_id', '')}",
        "",
        "Focus:",
        "- boundary fixed to FWHT baseline",
        "- comparator fixed to `rotor_loop_chordal_v1`",
        "- reader fixed to `first_after_defect_score_ranknorm`",
        "- compare `k=0` vs `k=3` only",
        "",
        "## All Frustrated Summary",
        "",
        f"- mean_raw_token_ap_delta_rotor_vs_F_k0: {render_float(all_summary['mean_raw_token_ap_delta_rotor_vs_F_k0'])}",
        f"- mean_raw_token_ap_delta_rotor_vs_F_k3: {render_float(all_summary['mean_raw_token_ap_delta_rotor_vs_F_k3'])}",
        f"- mean_rank_first_after_delta_rotor_vs_F_k0: {render_float(all_summary['mean_rank_first_after_delta_rotor_vs_F_k0'])}",
        f"- mean_rank_first_after_delta_rotor_vs_F_k3: {render_float(all_summary['mean_rank_first_after_delta_rotor_vs_F_k3'])}",
        f"- mean_rank_delta_lift_rotor_vs_F: {render_float(all_summary['mean_rank_delta_lift_rotor_vs_F'])}",
        f"- mean_rank_first_after_lift_rotor: {render_float(all_summary['mean_rank_first_after_lift_rotor'])}",
        f"- mean_raw_first_after_lift_rotor: {render_float(all_summary['mean_raw_first_after_lift_rotor'])}",
        f"- rank_first_after_win_rate_k0: {render_float(all_summary['rank_first_after_win_rate_k0'])}",
        f"- rank_first_after_win_rate_k3: {render_float(all_summary['rank_first_after_win_rate_k3'])}",
        f"- label_mismatch_support_rate: {render_float(all_summary['label_mismatch_support_rate'])}",
        f"- ranknorm_artifact_risk_rate: {render_float(all_summary['ranknorm_artifact_risk_rate'])}",
        "",
        "## World-Type Summary",
        "",
        "| world_type | n | rank_delta_k0 | rank_delta_k3 | rank_lift_rotor | raw_lift_rotor | mismatch_support | artifact_risk |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in world_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["group_id"]),
                    str(row["n_samples"]),
                    render_float(row["mean_rank_first_after_delta_rotor_vs_F_k0"]),
                    render_float(row["mean_rank_first_after_delta_rotor_vs_F_k3"]),
                    render_float(row["mean_rank_first_after_lift_rotor"]),
                    render_float(row["mean_raw_first_after_lift_rotor"]),
                    render_float(row["label_mismatch_support_rate"]),
                    render_float(row["ranknorm_artifact_risk_rate"]),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Selected Cases", ""])
    for row in selected_rows:
        lines.append(
            f"- sample_id={row['sample_id']} world_type={row['world_type']} "
            f"rank_delta_k0={render_float(row['rank_first_after_delta_rotor_vs_F_k0'])} "
            f"rank_delta_k3={render_float(row['rank_first_after_delta_rotor_vs_F_k3'])} "
            f"rank_lift_rotor={render_float(row['rank_first_after_lift_rotor'])} "
            f"raw_lift_rotor={render_float(row['raw_first_after_lift_rotor'])} "
            f"mismatch_support={row['label_mismatch_support']} "
            f"artifact_risk={row['ranknorm_artifact_risk']}"
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- decision: `{decision}`",
            f"- {rationale}",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_decision(out_path: Path, decision: str, rationale: str, all_summary: Dict[str, Any]) -> None:
    lines = [
        "# Gate5 Aggregation Failure-Mode Decision",
        "",
        f"- decision: `{decision}`",
        "",
        "## Decision Basis",
        "",
        f"- mean_rank_first_after_delta_rotor_vs_F_k0: {render_float(all_summary['mean_rank_first_after_delta_rotor_vs_F_k0'])}",
        f"- mean_rank_first_after_delta_rotor_vs_F_k3: {render_float(all_summary['mean_rank_first_after_delta_rotor_vs_F_k3'])}",
        f"- mean_rank_first_after_lift_rotor: {render_float(all_summary['mean_rank_first_after_lift_rotor'])}",
        f"- mean_raw_first_after_lift_rotor: {render_float(all_summary['mean_raw_first_after_lift_rotor'])}",
        f"- label_mismatch_support_rate: {render_float(all_summary['label_mismatch_support_rate'])}",
        f"- ranknorm_artifact_risk_rate: {render_float(all_summary['ranknorm_artifact_risk_rate'])}",
        "",
        f"- {rationale}",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def choose_decision(all_summary: Dict[str, Any], world_rows: Sequence[Dict[str, Any]]) -> Tuple[str, str]:
    world_index = {str(row["group_id"]): row for row in world_rows}
    genealogy = world_index["genealogy"]
    reachability = world_index["reachability"]
    temporal = world_index["temporal"]

    rank_k0 = all_summary["mean_rank_first_after_delta_rotor_vs_F_k0"] or 0.0
    rank_k3 = all_summary["mean_rank_first_after_delta_rotor_vs_F_k3"] or 0.0
    raw_lift_rotor = all_summary["mean_raw_first_after_lift_rotor"] or 0.0
    mismatch_rate = all_summary["label_mismatch_support_rate"] or 0.0
    artifact_rate = all_summary["ranknorm_artifact_risk_rate"] or 0.0

    if (
        rank_k3 > 0.0
        and rank_k0 <= 0.0
        and raw_lift_rotor > 0.0
        and (reachability["mean_rank_first_after_delta_rotor_vs_F_k3"] or 0.0) > 0.0
        and (temporal["mean_rank_first_after_delta_rotor_vs_F_k3"] or 0.0) > 0.0
    ):
        return (
            "reader-refinement-still-live",
            "k=3 strength is consistent with an early-signal / label-mismatch story: "
            f"overall rank delta flips from {render_float(rank_k0)} to {render_float(rank_k3)}, "
            f"rotor raw first-after also rises ({render_float(raw_lift_rotor)}), "
            f"reachability/temporal stay positive at k=3, while genealogy remains the unresolved failure.",
        )

    if artifact_rate >= 0.5 and mismatch_rate < 0.5:
        return (
            "ranknorm-artifact-risk",
            "the k=3 gain appears too dependent on rank normalization without matching raw first-after lift; "
            "reader refinement should pause unless a non-rank artifact explanation appears.",
        )

    if (genealogy["mean_rank_first_after_delta_rotor_vs_F_k3"] or 0.0) < 0.0:
        return (
            "reader-refinement-still-live",
            "k=3 remains promising outside genealogy, but genealogy is still negative; continue only as targeted "
            "reader failure-mode analysis rather than promotion work.",
        )

    return (
        "pause-aggregation-line",
        "the current reader does not separate a usable early-signal story from normalization-driven behavior.",
    )


def main() -> int:
    args = parse_args()
    gate5_out_dir = (REPO_ROOT / args.gate5_out_dir).resolve()
    out_dir = (REPO_ROOT / args.out_dir).resolve()

    if set(args.k_values) != {0, 3}:
        raise ValueError("--k-values must be exactly 0 and 3 for aggregation failure-mode analysis")

    token_rows = read_csv(gate5_out_dir / "gate5_token_telemetry.csv")
    sample_rows = read_csv(gate5_out_dir / "gate5_sample_summary.csv")
    manifest = json.loads((gate5_out_dir / "manifest.json").read_text(encoding="utf-8"))
    token_grouped = group_token_rows(token_rows)

    frustrated_rows = [row for row in sample_rows if row.get("variant") == "frustrated"]
    if not frustrated_rows:
        raise ValueError("aggregation failure-mode analysis found zero frustrated samples")

    sample_ids_present = {int(row["sample_id"]) for row in frustrated_rows}
    missing_ids = [sample_id for sample_id in args.sample_ids if sample_id not in sample_ids_present]
    if missing_ids:
        raise ValueError(
            "requested sample_ids are not present in aggregation failure-mode frustrated population: "
            + ",".join(str(sample_id) for sample_id in missing_ids)
        )

    sample_summary_rows: List[Dict[str, Any]] = []
    for sample_row in frustrated_rows:
        sample_id = int(sample_row["sample_id"])
        sample_summary_rows.append(build_sample_row(sample_row, token_grouped[sample_id]))
    sample_summary_rows.sort(key=lambda row: (str(row["world_type"]), int(row["sample_id"])))

    all_summary = summarize_group("all", sample_summary_rows)
    world_rows: List[Dict[str, Any]] = []
    for world_type in sorted({str(row["world_type"]) for row in sample_summary_rows}):
        grouped = [row for row in sample_summary_rows if str(row["world_type"]) == world_type]
        world_rows.append(summarize_group(world_type, grouped))

    selected_rows = [row for row in sample_summary_rows if int(row["sample_id"]) in set(args.sample_ids)]
    selected_rows.sort(key=lambda row: args.sample_ids.index(int(row["sample_id"])))

    decision, rationale = choose_decision(all_summary, world_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    sample_csv = out_dir / "gate5_aggregation_failure_mode_sample_summary.csv"
    world_csv = out_dir / "gate5_aggregation_failure_mode_world_summary.csv"
    selected_csv = out_dir / "gate5_aggregation_failure_mode_selected_cases.csv"
    report_md = out_dir / "gate5_aggregation_failure_mode_report.md"
    decision_md = out_dir / "gate5_aggregation_failure_mode_decision.md"

    write_csv(
        sample_csv,
        [
            "sample_id",
            "variant",
            "world_type",
            "positive_token_count",
            "delta_auprc_rotor_vs_F",
            "auprc_F",
            "auprc_rotor",
            "defect_start_step_k0",
            "defect_start_step_k3",
            "defect_start_shift_k3_minus_k0",
            "raw_token_ap_delta_rotor_vs_F_k0",
            "raw_token_ap_delta_rotor_vs_F_k3",
            "raw_first_after_F_k0",
            "raw_first_after_rotor_k0",
            "raw_first_after_F_k3",
            "raw_first_after_rotor_k3",
            "rank_first_after_F_k0",
            "rank_first_after_rotor_k0",
            "rank_first_after_delta_rotor_vs_F_k0",
            "rank_first_after_F_k3",
            "rank_first_after_rotor_k3",
            "rank_first_after_delta_rotor_vs_F_k3",
            "rank_first_after_lift_F",
            "rank_first_after_lift_rotor",
            "raw_first_after_lift_F",
            "raw_first_after_lift_rotor",
            "rank_delta_lift_rotor_vs_F",
            "raw_token_delta_lift_rotor_vs_F",
            "ranknorm_sign_flip_between_k0_k3",
            "label_mismatch_support",
            "ranknorm_artifact_risk",
        ],
        sample_summary_rows,
    )
    write_csv(
        world_csv,
        [
            "group_id",
            "n_samples",
            "mean_delta_auprc_rotor_vs_F",
            "mean_raw_token_ap_delta_rotor_vs_F_k0",
            "mean_raw_token_ap_delta_rotor_vs_F_k3",
            "mean_rank_first_after_delta_rotor_vs_F_k0",
            "mean_rank_first_after_delta_rotor_vs_F_k3",
            "mean_rank_delta_lift_rotor_vs_F",
            "mean_rank_first_after_lift_F",
            "mean_rank_first_after_lift_rotor",
            "mean_raw_first_after_lift_F",
            "mean_raw_first_after_lift_rotor",
            "rank_first_after_win_rate_k0",
            "rank_first_after_win_rate_k3",
            "label_mismatch_support_rate",
            "ranknorm_artifact_risk_rate",
            "mean_defect_start_shift_k3_minus_k0",
        ],
        [all_summary] + world_rows,
    )
    write_csv(
        selected_csv,
        [
            "sample_id",
            "world_type",
            "raw_token_ap_delta_rotor_vs_F_k0",
            "raw_token_ap_delta_rotor_vs_F_k3",
            "rank_first_after_delta_rotor_vs_F_k0",
            "rank_first_after_delta_rotor_vs_F_k3",
            "rank_first_after_lift_rotor",
            "raw_first_after_lift_rotor",
            "label_mismatch_support",
            "ranknorm_artifact_risk",
        ],
        selected_rows,
    )
    write_report(report_md, manifest, all_summary, world_rows, selected_rows, decision, rationale)
    write_decision(decision_md, decision, rationale, all_summary)

    print(f"sample_summary_csv={sample_csv.as_posix()}")
    print(f"world_summary_csv={world_csv.as_posix()}")
    print(f"selected_cases_csv={selected_csv.as_posix()}")
    print(f"report_md={report_md.as_posix()}")
    print(f"decision_md={decision_md.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
