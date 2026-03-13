#!/usr/bin/env python3
"""Diagnose genealogy residual-side failure on a fixed Gate5 CFA run."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GEOMETRIES = ["inside_span", "prefix_only_w3"]
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether genealogy failure is mostly benchmark geometry "
            "mismatch or residual weakness on an existing Gate5 FWHT run."
        )
    )
    parser.add_argument("--gate5-out-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--geometries", nargs="+", default=DEFAULT_GEOMETRIES)
    parser.add_argument("--genealogy-top-n", type=int, default=5)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--percentile", type=float, default=0.90)
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

    n = len(base_labels)
    defect_start, defect_end = defect_span(base_labels)
    if defect_start is None or defect_end is None:
        return [0] * n

    out = [0] * n

    def mark(lo: int, hi: int) -> None:
        lo = max(0, lo)
        hi = min(n - 1, hi)
        for idx in range(lo, hi + 1):
            out[idx] = 1

    if geometry_id == "prefix_only_w3":
        mark(defect_start - 3, defect_start - 1)
    else:
        raise ValueError(f"unsupported geometry id: {geometry_id}")
    return out


def top1_step(scores: Sequence[Optional[float]]) -> Optional[int]:
    ranked = [(idx, float(score)) for idx, score in enumerate(scores) if score is not None]
    if not ranked:
        return None
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked[0][0]


def first_hit_step(scores: Sequence[Optional[float]], percentile: float) -> Optional[int]:
    valid_scores = [score for score in scores if score is not None]
    threshold = percentile_nearest_rank(valid_scores, percentile)
    if threshold is None:
        return None
    return next(
        (idx for idx, score in enumerate(scores) if score is not None and score >= threshold),
        None,
    )


def sum_mass(scores: Sequence[Optional[float]], indices: Iterable[int]) -> float:
    total = 0.0
    for idx in indices:
        if 0 <= idx < len(scores) and scores[idx] is not None:
            total += float(scores[idx])
    return total


def build_per_sample_geometry_row(
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
        "first_hit_rotor_step": first_hit_rotor,
        "first_hit_rotor_distance_signed": None if defect_start is None or first_hit_rotor is None else int(first_hit_rotor) - int(defect_start),
        "top1_rotor_step": top1_rotor,
        "top1_rotor_distance_signed": None if defect_start is None or top1_rotor is None else int(top1_rotor) - int(defect_start),
        "before_span_mass_rotor": before_mass,
        "inside_span_mass_rotor": inside_mass,
        "after_span_mass_rotor": after_mass,
        "before_to_inside_ratio_rotor": before_mass / max(inside_mass, EPS),
        "inside_to_after_ratio_rotor": inside_mass / max(after_mass, EPS),
    }


def summarize_rows(rows: Sequence[Dict[str, Any]], geometry_id: str, world_type: str) -> Dict[str, Any]:
    return {
        "geometry_id": geometry_id,
        "world_type": world_type,
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


def build_selected_failures(
    genealogy_inside_rows: Sequence[Dict[str, Any]],
    genealogy_prefix_rows: Sequence[Dict[str, Any]],
    top_n: int,
) -> List[Dict[str, Any]]:
    by_sample_inside = {int(row["sample_id"]): row for row in genealogy_inside_rows}
    by_sample_prefix = {int(row["sample_id"]): row for row in genealogy_prefix_rows}
    combined: Dict[int, Dict[str, Any]] = {}

    for sample_id, inside_row in by_sample_inside.items():
        prefix_row = by_sample_prefix.get(sample_id)
        if prefix_row is None:
            continue
        combined[sample_id] = {
            "sample_id": sample_id,
            "world_type": inside_row["world_type"],
            "inside_delta_rotor_vs_F": inside_row["delta_rotor_vs_F"],
            "prefix_delta_rotor_vs_F": prefix_row["delta_rotor_vs_F"],
            "geometry_recovery": None
            if inside_row["delta_rotor_vs_F"] is None or prefix_row["delta_rotor_vs_F"] is None
            else prefix_row["delta_rotor_vs_F"] - inside_row["delta_rotor_vs_F"],
            "still_negative_under_prefix_only_w3": 1
            if prefix_row["delta_rotor_vs_F"] is not None and prefix_row["delta_rotor_vs_F"] < 0.0
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

    selected_rows: List[Dict[str, Any]] = []

    worst_inside = sorted(
        combined.values(),
        key=lambda row: (
            float("inf") if row["inside_delta_rotor_vs_F"] is None else row["inside_delta_rotor_vs_F"],
            int(row["sample_id"]),
        ),
    )[:top_n]
    for row in worst_inside:
        selected_rows.append({**row, "selection_reason": "inside_span_worst_genealogy"})

    still_negative = sorted(
        [row for row in combined.values() if row["still_negative_under_prefix_only_w3"] == 1],
        key=lambda row: (
            float("inf") if row["prefix_delta_rotor_vs_F"] is None else row["prefix_delta_rotor_vs_F"],
            int(row["sample_id"]),
        ),
    )[:top_n]
    for row in still_negative:
        selected_rows.append({**row, "selection_reason": "prefix_only_w3_still_negative"})

    recovery_rows = sorted(
        combined.values(),
        key=lambda row: (
            float("-inf") if row["geometry_recovery"] is None else row["geometry_recovery"],
            -int(row["sample_id"]),
        ),
        reverse=True,
    )[:top_n]
    for row in recovery_rows:
        selected_rows.append({**row, "selection_reason": "geometry_recovery_large"})

    selected_rows.sort(key=lambda row: (row["selection_reason"], int(row["sample_id"])))
    return selected_rows


def decide(
    genealogy_inside: Dict[str, Any],
    genealogy_prefix: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    gain = None
    if (
        genealogy_inside["mean_delta_rotor_vs_F"] is not None
        and genealogy_prefix["mean_delta_rotor_vs_F"] is not None
    ):
        gain = genealogy_prefix["mean_delta_rotor_vs_F"] - genealogy_inside["mean_delta_rotor_vs_F"]

    meta = {
        "genealogy_inside_delta": genealogy_inside["mean_delta_rotor_vs_F"],
        "genealogy_prefix_delta": genealogy_prefix["mean_delta_rotor_vs_F"],
        "genealogy_gain": gain,
        "prefix_still_negative_rate": genealogy_prefix["still_negative_rate"],
        "inside_before_to_inside_ratio": genealogy_inside["mean_before_to_inside_ratio_rotor"],
        "prefix_before_to_inside_ratio": genealogy_prefix["mean_before_to_inside_ratio_rotor"],
        "inside_inside_to_after_ratio": genealogy_inside["mean_inside_to_after_ratio_rotor"],
        "prefix_inside_to_after_ratio": genealogy_prefix["mean_inside_to_after_ratio_rotor"],
    }

    if (
        gain is not None
        and gain >= 0.10
        and genealogy_prefix["mean_delta_rotor_vs_F"] is not None
        and genealogy_prefix["mean_delta_rotor_vs_F"] > 0.0
        and genealogy_prefix["still_negative_rate"] is not None
        and genealogy_prefix["still_negative_rate"] <= 0.35
    ):
        return ("benchmark-mismatch-explains-most", meta)

    shape_still_bad = False
    if (
        genealogy_prefix["mean_before_to_inside_ratio_rotor"] is not None
        and genealogy_inside["mean_before_to_inside_ratio_rotor"] is not None
        and genealogy_prefix["mean_inside_to_after_ratio_rotor"] is not None
        and genealogy_inside["mean_inside_to_after_ratio_rotor"] is not None
    ):
        shape_still_bad = (
            genealogy_prefix["mean_before_to_inside_ratio_rotor"]
            >= genealogy_inside["mean_before_to_inside_ratio_rotor"]
            or genealogy_prefix["mean_inside_to_after_ratio_rotor"]
            <= genealogy_inside["mean_inside_to_after_ratio_rotor"]
        )

    if (
        genealogy_prefix["mean_delta_rotor_vs_F"] is None
        or genealogy_prefix["mean_delta_rotor_vs_F"] <= 0.0
        or (
            genealogy_prefix["still_negative_rate"] is not None
            and genealogy_prefix["still_negative_rate"] >= 0.60
            and shape_still_bad
        )
    ):
        return ("residual-genealogy-weakness-remains", meta)

    return ("mixed", meta)


def build_decision_text(decision: str, meta: Dict[str, Any]) -> str:
    lines = [
        "# Genealogy Residual Autopsy Decision",
        "",
        f"- decision: `{decision}`",
        "",
        "## Key Basis",
        "",
        f"- genealogy inside_span delta: {render_float(meta.get('genealogy_inside_delta'))}",
        f"- genealogy prefix_only_w3 delta: {render_float(meta.get('genealogy_prefix_delta'))}",
        f"- geometry gain: {render_float(meta.get('genealogy_gain'))}",
        f"- prefix_only_w3 still-negative rate: {render_float(meta.get('prefix_still_negative_rate'))}",
        f"- inside before_to_inside ratio: {render_float(meta.get('inside_before_to_inside_ratio'))}",
        f"- prefix before_to_inside ratio: {render_float(meta.get('prefix_before_to_inside_ratio'))}",
        f"- inside inside_to_after ratio: {render_float(meta.get('inside_inside_to_after_ratio'))}",
        f"- prefix inside_to_after ratio: {render_float(meta.get('prefix_inside_to_after_ratio'))}",
        "",
        "This diagnosis keeps canonical `inside_span` fixed and uses `prefix_only_w3` only as a diagnostic view.",
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
        if row.get("variant") == "frustrated"
        and parse_float(row.get("delta_auprc_rotor_loop_chordal_v1_vs_F")) is not None
    ]
    if not sample_rows:
        raise SystemExit("no frustrated rows found in gate5_sample_summary.csv")

    per_sample_rows: List[Dict[str, Any]] = []
    for sample_row in sample_rows:
        sample_id = int(sample_row["sample_id"])
        token_rows = token_grouped[sample_id]
        for geometry_id in args.geometries:
            per_sample_rows.append(
                build_per_sample_geometry_row(
                    sample_summary=sample_row,
                    token_rows=token_rows,
                    geometry_id=geometry_id,
                    percentile=args.percentile,
                )
            )

    genealogy_inside_rows = [
        row
        for row in per_sample_rows
        if row["world_type"] == "genealogy" and row["geometry_id"] == "inside_span"
    ]
    genealogy_prefix_rows = [
        row
        for row in per_sample_rows
        if row["world_type"] == "genealogy" and row["geometry_id"] == "prefix_only_w3"
    ]
    if not genealogy_inside_rows or not genealogy_prefix_rows:
        raise SystemExit("genealogy rows missing for inside_span or prefix_only_w3")

    shape_summary_rows: List[Dict[str, Any]] = []
    world_types = sorted({row["world_type"] for row in per_sample_rows})
    for geometry_id in args.geometries:
        for world_type in world_types:
            rows = [
                row
                for row in per_sample_rows
                if row["geometry_id"] == geometry_id and row["world_type"] == world_type
            ]
            if rows:
                shape_summary_rows.append(summarize_rows(rows, geometry_id, world_type))

    selected_rows = build_selected_failures(
        genealogy_inside_rows=genealogy_inside_rows,
        genealogy_prefix_rows=genealogy_prefix_rows,
        top_n=args.genealogy_top_n,
    )

    by_world_geometry = {
        (row["geometry_id"], row["world_type"]): row for row in shape_summary_rows
    }
    decision, meta = decide(
        genealogy_inside=by_world_geometry[("inside_span", "genealogy")],
        genealogy_prefix=by_world_geometry[("prefix_only_w3", "genealogy")],
    )

    summary_fields = [
        "geometry_id",
        "world_type",
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
    selected_fields = [
        "selection_reason",
        "sample_id",
        "world_type",
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

    write_csv(out_dir / "genealogy_residual_shape_summary.csv", summary_fields, shape_summary_rows)
    write_csv(out_dir / "genealogy_residual_selected_failures.csv", selected_fields, selected_rows)
    (out_dir / "genealogy_residual_autopsy_decision.md").write_text(
        build_decision_text(decision, meta),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
