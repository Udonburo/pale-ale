#!/usr/bin/env python3
"""Quantify genealogy residual-side remainder after policy-adjusted interpretation."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
GEOMETRIES = ("inside_span", "prefix_only_w3")
EPS = 1e-12
MIN_POSITIVE_PREFIX_DELTA = 0.10
MIN_GEOMETRY_GAIN = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure how much genealogy residual-side weakness remains after "
            "accepting prefix_only_w3 as a diagnostic-only geometry."
        )
    )
    parser.add_argument("--gate5-out-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--percentile", type=float, default=0.90)
    parser.add_argument("--borderline-threshold", type=float, default=0.03)
    return parser.parse_args()


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


def mean(values: Iterable[float]) -> Optional[float]:
    arr = list(values)
    if not arr:
        return None
    return sum(arr) / float(len(arr))


def render_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def percentile_nearest_rank(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    arr = sorted(values)
    rank = int(math.ceil(q * len(arr))) - 1
    rank = max(0, min(rank, len(arr) - 1))
    return arr[rank]


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


def group_token_rows(rows: Sequence[Dict[str, str]]) -> Dict[int, List[Dict[str, str]]]:
    grouped: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["sample_id"])].append(row)
    for sample_rows in grouped.values():
        sample_rows.sort(key=lambda row: int(row["step"]))
    return grouped


def defect_span(labels: Sequence[int]) -> Tuple[Optional[int], Optional[int]]:
    steps = [idx for idx, label in enumerate(labels) if label == 1]
    if not steps:
        return (None, None)
    return (steps[0], steps[-1])


def build_geometry_labels(base_labels: Sequence[int], geometry_id: str) -> List[int]:
    if geometry_id == "inside_span":
        return list(base_labels)
    if geometry_id != "prefix_only_w3":
        raise ValueError(f"unsupported geometry id: {geometry_id}")

    n = len(base_labels)
    defect_start, _defect_end = defect_span(base_labels)
    if defect_start is None:
        return [0] * n

    out = [0] * n
    lo = max(0, defect_start - 3)
    hi = defect_start - 1
    if hi < 0 or lo > hi:
        return out
    for idx in range(lo, min(hi, n - 1) + 1):
        out[idx] = 1
    return out


def first_hit_step(scores: Sequence[Optional[float]], percentile: float) -> Optional[int]:
    valid_scores = [score for score in scores if score is not None]
    threshold = percentile_nearest_rank(valid_scores, percentile)
    if threshold is None:
        return None
    return next(
        (idx for idx, score in enumerate(scores) if score is not None and score >= threshold),
        None,
    )


def top1_step(scores: Sequence[Optional[float]]) -> Optional[int]:
    ranked = [(idx, float(score)) for idx, score in enumerate(scores) if score is not None]
    if not ranked:
        return None
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked[0][0]


def sum_mass(scores: Sequence[Optional[float]], indices: Iterable[int]) -> float:
    total = 0.0
    for idx in indices:
        if 0 <= idx < len(scores) and scores[idx] is not None:
            total += float(scores[idx])
    return total


def build_per_sample_row(
    sample_summary: Dict[str, str],
    token_rows: Sequence[Dict[str, str]],
    geometry_id: str,
    percentile: float,
) -> Dict[str, Any]:
    base_labels = [1 if row.get("label_token") == "1" else 0 for row in token_rows]
    labels = build_geometry_labels(base_labels, geometry_id)
    defect_start, defect_end = defect_span(labels)
    scores_f = [parse_float(row.get("score_F_loop")) for row in token_rows]
    scores_rotor = [parse_float(row.get("rotor_loop_chordal_v1")) for row in token_rows]
    ap_f = average_precision(labels, scores_f)
    ap_rotor = average_precision(labels, scores_rotor)
    first_hit_rotor = first_hit_step(scores_rotor, percentile)
    top1_rotor = top1_step(scores_rotor)
    before_ix = range(0, defect_start if defect_start is not None else 0)
    inside_ix = range(
        defect_start if defect_start is not None else 0,
        (defect_end + 1) if defect_end is not None else 0,
    )
    after_ix = range((defect_end + 1) if defect_end is not None else len(scores_rotor), len(scores_rotor))
    before_mass = sum_mass(scores_rotor, before_ix)
    inside_mass = sum_mass(scores_rotor, inside_ix)
    after_mass = sum_mass(scores_rotor, after_ix)
    return {
        "sample_id": int(sample_summary["sample_id"]),
        "variant": sample_summary.get("variant", ""),
        "world_type": sample_summary.get("world_type", ""),
        "geometry_id": geometry_id,
        "defect_start_step": defect_start,
        "defect_end_step": defect_end,
        "auprc_F": ap_f,
        "auprc_rotor": ap_rotor,
        "delta_rotor_vs_F": None if ap_f is None or ap_rotor is None else ap_rotor - ap_f,
        "first_hit_rotor_distance_signed": None
        if defect_start is None or first_hit_rotor is None
        else int(first_hit_rotor) - int(defect_start),
        "top1_rotor_distance_signed": None
        if defect_start is None or top1_rotor is None
        else int(top1_rotor) - int(defect_start),
        "before_span_mass_rotor": before_mass,
        "inside_span_mass_rotor": inside_mass,
        "after_span_mass_rotor": after_mass,
        "before_to_inside_ratio_rotor": before_mass / max(inside_mass, EPS),
        "inside_to_after_ratio_rotor": inside_mass / max(after_mass, EPS),
    }


def summarize_rows(rows: Sequence[Dict[str, Any]], geometry_id: str) -> Dict[str, Any]:
    return {
        "geometry_id": geometry_id,
        "n_samples": len(rows),
        "mean_auprc_F": mean(row["auprc_F"] for row in rows if row["auprc_F"] is not None),
        "mean_auprc_rotor": mean(row["auprc_rotor"] for row in rows if row["auprc_rotor"] is not None),
        "mean_delta_rotor_vs_F": mean(
            row["delta_rotor_vs_F"] for row in rows if row["delta_rotor_vs_F"] is not None
        ),
        "mean_first_hit_rotor_distance_signed": mean(
            float(row["first_hit_rotor_distance_signed"])
            for row in rows
            if row["first_hit_rotor_distance_signed"] is not None
        ),
        "mean_top1_rotor_distance_signed": mean(
            float(row["top1_rotor_distance_signed"])
            for row in rows
            if row["top1_rotor_distance_signed"] is not None
        ),
        "mean_before_span_mass_rotor": mean(row["before_span_mass_rotor"] for row in rows),
        "mean_inside_span_mass_rotor": mean(row["inside_span_mass_rotor"] for row in rows),
        "mean_after_span_mass_rotor": mean(row["after_span_mass_rotor"] for row in rows),
        "mean_before_to_inside_ratio_rotor": mean(row["before_to_inside_ratio_rotor"] for row in rows),
        "mean_inside_to_after_ratio_rotor": mean(row["inside_to_after_ratio_rotor"] for row in rows),
        "still_negative_rate": mean(
            1.0 if row["delta_rotor_vs_F"] is not None and row["delta_rotor_vs_F"] < 0.0 else 0.0
            for row in rows
        ),
    }


def build_cases(
    inside_rows: Sequence[Dict[str, Any]],
    prefix_rows: Sequence[Dict[str, Any]],
    top_n: int,
    borderline_threshold: float,
) -> List[Dict[str, Any]]:
    by_inside = {int(row["sample_id"]): row for row in inside_rows}
    by_prefix = {int(row["sample_id"]): row for row in prefix_rows}
    combined: List[Dict[str, Any]] = []
    for sample_id, inside_row in by_inside.items():
        prefix_row = by_prefix.get(sample_id)
        if prefix_row is None:
            continue
        inside_delta = inside_row["delta_rotor_vs_F"]
        prefix_delta = prefix_row["delta_rotor_vs_F"]
        combined.append(
            {
                "sample_id": sample_id,
                "inside_delta_rotor_vs_F": inside_delta,
                "prefix_delta_rotor_vs_F": prefix_delta,
                "geometry_recovery": None
                if inside_delta is None or prefix_delta is None
                else prefix_delta - inside_delta,
                "still_negative_under_prefix_only_w3": 1
                if prefix_delta is not None and prefix_delta < 0.0
                else 0,
                "inside_first_hit_rotor_distance_signed": inside_row["first_hit_rotor_distance_signed"],
                "prefix_first_hit_rotor_distance_signed": prefix_row["first_hit_rotor_distance_signed"],
                "inside_top1_rotor_distance_signed": inside_row["top1_rotor_distance_signed"],
                "prefix_top1_rotor_distance_signed": prefix_row["top1_rotor_distance_signed"],
                "inside_before_to_inside_ratio_rotor": inside_row["before_to_inside_ratio_rotor"],
                "prefix_before_to_inside_ratio_rotor": prefix_row["before_to_inside_ratio_rotor"],
                "inside_inside_to_after_ratio_rotor": inside_row["inside_to_after_ratio_rotor"],
                "prefix_inside_to_after_ratio_rotor": prefix_row["inside_to_after_ratio_rotor"],
            }
        )

    selected: List[Dict[str, Any]] = []

    still_negative = sorted(
        [row for row in combined if row["still_negative_under_prefix_only_w3"] == 1],
        key=lambda row: (
            float("inf") if row["prefix_delta_rotor_vs_F"] is None else row["prefix_delta_rotor_vs_F"],
            row["sample_id"],
        ),
    )[:top_n]
    for row in still_negative:
        selected.append({**row, "selection_reason": "still_negative_under_prefix_only_w3"})

    recovered = sorted(
        [row for row in combined if row["prefix_delta_rotor_vs_F"] is not None and row["prefix_delta_rotor_vs_F"] >= 0.0],
        key=lambda row: (
            float("-inf") if row["geometry_recovery"] is None else row["geometry_recovery"],
            -row["sample_id"],
        ),
        reverse=True,
    )[:top_n]
    for row in recovered:
        selected.append({**row, "selection_reason": "recovered_by_prefix_only_w3"})

    borderline = sorted(
        combined,
        key=lambda row: (
            float("inf")
            if row["prefix_delta_rotor_vs_F"] is None
            else abs(row["prefix_delta_rotor_vs_F"]),
            row["sample_id"],
        ),
    )[: max(top_n, 1)]
    borderline = [
        row
        for row in borderline
        if row["prefix_delta_rotor_vs_F"] is not None
        and abs(row["prefix_delta_rotor_vs_F"]) <= borderline_threshold
    ][:top_n]
    for row in borderline:
        selected.append({**row, "selection_reason": "borderline_under_prefix_only_w3"})

    selected.sort(key=lambda row: (row["selection_reason"], int(row["sample_id"])))
    return selected


def decide(
    inside_summary: Dict[str, Any],
    prefix_summary: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    inside_delta = inside_summary["mean_delta_rotor_vs_F"]
    prefix_delta = prefix_summary["mean_delta_rotor_vs_F"]
    gain = None
    if inside_delta is not None and prefix_delta is not None:
        gain = prefix_delta - inside_delta
    meta = {
        "inside_delta": inside_delta,
        "prefix_delta": prefix_delta,
        "gain": gain,
        "still_negative_rate": prefix_summary["still_negative_rate"],
        "prefix_before_to_inside_ratio": prefix_summary["mean_before_to_inside_ratio_rotor"],
        "prefix_inside_to_after_ratio": prefix_summary["mean_inside_to_after_ratio_rotor"],
    }

    if (
        prefix_delta is not None
        and prefix_delta >= MIN_POSITIVE_PREFIX_DELTA
        and gain is not None
        and gain >= MIN_GEOMETRY_GAIN
        and prefix_summary["still_negative_rate"] is not None
        and prefix_summary["still_negative_rate"] <= 0.10
        and prefix_summary["mean_before_to_inside_ratio_rotor"] is not None
        and prefix_summary["mean_before_to_inside_ratio_rotor"] <= 1.0
        and prefix_summary["mean_inside_to_after_ratio_rotor"] is not None
        and prefix_summary["mean_inside_to_after_ratio_rotor"] >= 0.20
    ):
        return ("little-residual-remainder", meta)

    if (
        prefix_delta is None
        or prefix_delta <= 0.0
        or (
            prefix_summary["still_negative_rate"] is not None
            and prefix_summary["still_negative_rate"] >= 0.35
        )
        or (
            prefix_summary["mean_before_to_inside_ratio_rotor"] is not None
            and prefix_summary["mean_before_to_inside_ratio_rotor"] > 1.20
        )
        or (
            prefix_summary["mean_inside_to_after_ratio_rotor"] is not None
            and prefix_summary["mean_inside_to_after_ratio_rotor"] < 0.10
        )
    ):
        return ("residual-remainder-still-material", meta)

    return ("mixed", meta)


def build_decision_text(decision: str, meta: Dict[str, Any]) -> str:
    lines = [
        "# Genealogy Residual Remainder Decision",
        "",
        f"- decision: `{decision}`",
        "",
        "## Key Basis",
        "",
        f"- inside_span delta: {render_float(meta.get('inside_delta'))}",
        f"- prefix_only_w3 delta: {render_float(meta.get('prefix_delta'))}",
        f"- geometry gain: {render_float(meta.get('gain'))}",
        f"- little-residual thresholds: prefix_delta>={MIN_POSITIVE_PREFIX_DELTA:.2f}, gain>={MIN_GEOMETRY_GAIN:.2f}",
        f"- prefix_only_w3 still-negative rate: {render_float(meta.get('still_negative_rate'))}",
        f"- prefix_only_w3 before_to_inside ratio: {render_float(meta.get('prefix_before_to_inside_ratio'))}",
        f"- prefix_only_w3 inside_to_after ratio: {render_float(meta.get('prefix_inside_to_after_ratio'))}",
        "",
        "This diagnosis keeps canonical `inside_span` unchanged and treats `prefix_only_w3` as diagnostic-only.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    gate5_out_dir = (REPO_ROOT / Path(args.gate5_out_dir)).resolve()
    out_dir = (REPO_ROOT / Path(args.out_dir)).resolve()

    sample_rows_all = read_csv(gate5_out_dir / "gate5_sample_summary.csv")
    token_rows_all = read_csv(gate5_out_dir / "gate5_token_telemetry.csv")
    token_grouped = group_token_rows(token_rows_all)

    sample_rows = [
        row
        for row in sample_rows_all
        if row.get("variant") == "frustrated" and row.get("world_type") == "genealogy"
    ]
    if not sample_rows:
        raise SystemExit("zero frustrated genealogy rows")

    per_sample_rows: List[Dict[str, Any]] = []
    for sample_row in sample_rows:
        sample_id = int(sample_row["sample_id"])
        token_rows = token_grouped[sample_id]
        for geometry_id in GEOMETRIES:
            per_sample_rows.append(
                build_per_sample_row(sample_row, token_rows, geometry_id, args.percentile)
            )

    by_geometry = {
        geometry_id: [row for row in per_sample_rows if row["geometry_id"] == geometry_id]
        for geometry_id in GEOMETRIES
    }

    summary_rows: List[Dict[str, Any]] = []
    for geometry_id in GEOMETRIES:
        summary_rows.append(summarize_rows(by_geometry[geometry_id], geometry_id))

    inside_summary = next(row for row in summary_rows if row["geometry_id"] == "inside_span")
    prefix_summary = next(row for row in summary_rows if row["geometry_id"] == "prefix_only_w3")
    decision, meta = decide(inside_summary, prefix_summary)

    cases = build_cases(
        inside_rows=by_geometry["inside_span"],
        prefix_rows=by_geometry["prefix_only_w3"],
        top_n=args.top_n,
        borderline_threshold=args.borderline_threshold,
    )

    summary_fields = [
        "geometry_id",
        "n_samples",
        "mean_auprc_F",
        "mean_auprc_rotor",
        "mean_delta_rotor_vs_F",
        "mean_first_hit_rotor_distance_signed",
        "mean_top1_rotor_distance_signed",
        "mean_before_span_mass_rotor",
        "mean_inside_span_mass_rotor",
        "mean_after_span_mass_rotor",
        "mean_before_to_inside_ratio_rotor",
        "mean_inside_to_after_ratio_rotor",
        "still_negative_rate",
    ]
    case_fields = [
        "selection_reason",
        "sample_id",
        "inside_delta_rotor_vs_F",
        "prefix_delta_rotor_vs_F",
        "geometry_recovery",
        "still_negative_under_prefix_only_w3",
        "inside_first_hit_rotor_distance_signed",
        "prefix_first_hit_rotor_distance_signed",
        "inside_top1_rotor_distance_signed",
        "prefix_top1_rotor_distance_signed",
        "inside_before_to_inside_ratio_rotor",
        "prefix_before_to_inside_ratio_rotor",
        "inside_inside_to_after_ratio_rotor",
        "prefix_inside_to_after_ratio_rotor",
    ]
    write_csv(out_dir / "genealogy_residual_remainder_summary.csv", summary_fields, summary_rows)
    write_csv(out_dir / "genealogy_residual_remainder_cases.csv", case_fields, cases)
    (out_dir / "genealogy_residual_remainder_decision.md").write_text(
        build_decision_text(decision, meta),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
