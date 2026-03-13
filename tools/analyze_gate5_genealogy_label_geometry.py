#!/usr/bin/env python3
"""Compare diagnostic label geometries on an existing Gate5 CFA run."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTED_WINS = [137, 147, 149, 11, 167]
DEFAULT_GEOMETRIES = [
    "inside_span",
    "onset_only",
    "start_neighborhood_w1",
    "start_neighborhood_w3",
    "prefix_only_w1",
    "prefix_only_w3",
    "symmetric_dilation_k1",
    "symmetric_dilation_k3",
]
DIAGNOSTIC_GEOMETRIES = [
    "onset_only",
    "start_neighborhood_w1",
    "start_neighborhood_w3",
    "prefix_only_w1",
    "prefix_only_w3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare diagnostic label geometries on an existing Gate5 FWHT run "
            "without changing the boundary, residual, or canonical labels."
        )
    )
    parser.add_argument("--gate5-out-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sample-ids", nargs="+", type=int, default=DEFAULT_SELECTED_WINS)
    parser.add_argument("--geometries", nargs="+", default=DEFAULT_GEOMETRIES)
    parser.add_argument("--genealogy-worst-count", type=int, default=5)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--percentile", type=float, default=0.90)
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


def rate(rows: Sequence[Dict[str, Any]], predicate) -> Optional[float]:
    if not rows:
        return None
    return sum(1.0 for row in rows if predicate(row)) / float(len(rows))


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


def hit_at_k(labels: Sequence[int], scores: Sequence[Optional[float]], k: int) -> Optional[int]:
    filtered = [(idx, float(score)) for idx, score in enumerate(scores) if score is not None]
    if not filtered or sum(labels) == 0:
        return None
    filtered.sort(key=lambda item: (-item[1], item[0]))
    return sum(labels[idx] for idx, _ in filtered[:k])


def group_token_rows(rows: Sequence[Dict[str, str]]) -> Dict[int, List[Dict[str, str]]]:
    grouped: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["sample_id"])].append(row)
    for sample_rows in grouped.values():
        sample_rows.sort(key=lambda row: int(row["step"]))
    return grouped


def zone_for_step(step: Optional[int], defect_start: Optional[int], defect_end: Optional[int]) -> str:
    if step is None or defect_start is None or defect_end is None:
        return ""
    if step < defect_start:
        return "before"
    if step > defect_end:
        return "after"
    return "inside"


def defect_span(labels: Sequence[int]) -> Tuple[Optional[int], Optional[int]]:
    steps = [idx for idx, label in enumerate(labels) if label == 1]
    if not steps:
        return (None, None)
    return (steps[0], steps[-1])


def dilate_labels(labels: Sequence[int], k: int) -> List[int]:
    n = len(labels)
    out = [0] * n
    positive_steps = [idx for idx, label in enumerate(labels) if label == 1]
    for step in positive_steps:
        lo = max(0, step - k)
        hi = min(n - 1, step + k)
        for idx in range(lo, hi + 1):
            out[idx] = 1
    return out


def build_geometry_labels(base_labels: Sequence[int], geometry_id: str) -> List[int]:
    n = len(base_labels)
    if geometry_id == "inside_span":
        return list(base_labels)

    defect_start, defect_end = defect_span(base_labels)
    if defect_start is None or defect_end is None:
        return [0] * n

    out = [0] * n

    def mark(lo: int, hi: int) -> None:
        lo = max(0, lo)
        hi = min(n - 1, hi)
        for idx in range(lo, hi + 1):
            out[idx] = 1

    if geometry_id == "onset_only":
        mark(defect_start, defect_start)
    elif geometry_id == "start_neighborhood_w1":
        mark(defect_start, defect_start + 1)
    elif geometry_id == "start_neighborhood_w3":
        mark(defect_start, defect_start + 3)
    elif geometry_id == "prefix_only_w1":
        mark(defect_start - 1, defect_start - 1)
    elif geometry_id == "prefix_only_w3":
        mark(defect_start - 3, defect_start - 1)
    elif geometry_id == "symmetric_dilation_k1":
        return dilate_labels(base_labels, 1)
    elif geometry_id == "symmetric_dilation_k3":
        return dilate_labels(base_labels, 3)
    else:
        raise ValueError(f"unknown geometry id: {geometry_id}")

    return out


def first_hit_stats(
    labels: Sequence[int],
    scores: Sequence[Optional[float]],
    percentile: float,
) -> Dict[str, Any]:
    defect_start, defect_end = defect_span(labels)
    valid_scores = [score for score in scores if score is not None]
    threshold = percentile_nearest_rank(valid_scores, percentile)
    first_hit = next(
        (idx for idx, score in enumerate(scores) if threshold is not None and score is not None and score >= threshold),
        None,
    )
    signed = None
    if defect_start is not None and first_hit is not None:
        signed = int(first_hit) - int(defect_start)
    return {
        "defect_start": defect_start,
        "defect_end": defect_end,
        "first_hit_step": first_hit,
        "first_hit_distance_signed": signed,
        "first_hit_zone": zone_for_step(first_hit, defect_start, defect_end),
    }


def build_per_sample_row(
    sample_summary: Dict[str, str],
    token_rows: Sequence[Dict[str, str]],
    geometry_id: str,
    topk: int,
    percentile: float,
) -> Dict[str, Any]:
    base_labels = [1 if row.get("label_token") == "1" else 0 for row in token_rows]
    labels = build_geometry_labels(base_labels, geometry_id)
    scores_f = [parse_float(row.get("score_F_loop")) for row in token_rows]
    scores_rotor = [parse_float(row.get("rotor_loop_chordal_v1")) for row in token_rows]
    ap_f = average_precision(labels, scores_f)
    ap_rotor = average_precision(labels, scores_rotor)
    hit_f = hit_at_k(labels, scores_f, topk)
    hit_rotor = hit_at_k(labels, scores_rotor, topk)
    first_f = first_hit_stats(labels, scores_f, percentile)
    first_rotor = first_hit_stats(labels, scores_rotor, percentile)
    defect_start, defect_end = defect_span(labels)
    return {
        "sample_id": int(sample_summary["sample_id"]),
        "variant": sample_summary.get("variant", ""),
        "world_type": sample_summary.get("world_type", ""),
        "geometry_id": geometry_id,
        "positive_token_count": sum(labels),
        "defect_start_step": defect_start,
        "defect_end_step": defect_end,
        "auprc_F": ap_f,
        "auprc_rotor": ap_rotor,
        "delta_rotor_vs_F": None if ap_f is None or ap_rotor is None else ap_rotor - ap_f,
        "hit_at_10_F": hit_f,
        "hit_at_10_rotor": hit_rotor,
        "first_hit_F_step": first_f["first_hit_step"],
        "first_hit_rotor_step": first_rotor["first_hit_step"],
        "first_hit_F_distance_signed": first_f["first_hit_distance_signed"],
        "first_hit_rotor_distance_signed": first_rotor["first_hit_distance_signed"],
        "first_hit_F_zone": first_f["first_hit_zone"],
        "first_hit_rotor_zone": first_rotor["first_hit_zone"],
    }


def summarize_rows(group_id: str, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "group_id": group_id,
        "n_samples": len(rows),
        "mean_auprc_F": mean(row["auprc_F"] for row in rows if row["auprc_F"] is not None),
        "mean_auprc_rotor": mean(row["auprc_rotor"] for row in rows if row["auprc_rotor"] is not None),
        "mean_delta_rotor_vs_F": mean(
            row["delta_rotor_vs_F"] for row in rows if row["delta_rotor_vs_F"] is not None
        ),
        "mean_hit_at_10_F": mean(float(row["hit_at_10_F"]) for row in rows if row["hit_at_10_F"] is not None),
        "mean_hit_at_10_rotor": mean(
            float(row["hit_at_10_rotor"]) for row in rows if row["hit_at_10_rotor"] is not None
        ),
        "rotor_first_hit_before_rate": rate(rows, lambda row: row["first_hit_rotor_zone"] == "before"),
        "f_first_hit_before_rate": rate(rows, lambda row: row["first_hit_F_zone"] == "before"),
        "mean_first_hit_rotor_distance_signed": mean(
            float(row["first_hit_rotor_distance_signed"])
            for row in rows
            if row["first_hit_rotor_distance_signed"] is not None
        ),
        "mean_first_hit_F_distance_signed": mean(
            float(row["first_hit_F_distance_signed"])
            for row in rows
            if row["first_hit_F_distance_signed"] is not None
        ),
    }


def build_selected_cases(
    sample_rows: Sequence[Dict[str, Any]],
    selected_ids: Sequence[int],
    genealogy_worst_count: int,
) -> List[Dict[str, Any]]:
    selected_set = {int(sample_id) for sample_id in selected_ids}
    inside_genealogy = [
        row
        for row in sample_rows
        if row["geometry_id"] == "inside_span" and row["world_type"] == "genealogy"
    ]
    worst_genealogy_ids = [
        int(row["sample_id"])
        for row in sorted(
            inside_genealogy,
            key=lambda row: (
                float("inf") if row["delta_rotor_vs_F"] is None else row["delta_rotor_vs_F"],
                int(row["sample_id"]),
            ),
        )[:genealogy_worst_count]
    ]
    out: List[Dict[str, Any]] = []
    for row in sample_rows:
        sample_id = int(row["sample_id"])
        tag = None
        if sample_id in selected_set:
            tag = "existing_global_win"
        elif sample_id in worst_genealogy_ids:
            tag = "baseline_genealogy_k0_worst"
        if tag is None:
            continue
        copied = dict(row)
        copied["selected_case_type"] = tag
        out.append(copied)
    out.sort(key=lambda row: (row["selected_case_type"], int(row["sample_id"]), row["geometry_id"]))
    return out


def validate_sample_ids(sample_rows: Sequence[Dict[str, Any]], requested_ids: Sequence[int]) -> None:
    available = {int(row["sample_id"]) for row in sample_rows}
    missing = [sample_id for sample_id in requested_ids if sample_id not in available]
    if missing:
        missing_str = ", ".join(str(sample_id) for sample_id in missing)
        raise SystemExit(f"requested sample ids not found in frustrated population: {missing_str}")


def decision_from_world_summary(world_rows: Sequence[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    geometry_world = {
        (str(row["geometry_id"]), str(row["world_type"])): row for row in world_rows
    }
    baseline_genealogy = geometry_world.get(("inside_span", "genealogy"))
    if baseline_genealogy is None:
        return ("inconclusive", {})

    baseline_delta = baseline_genealogy["mean_delta_rotor_vs_F"]
    best_row = None
    for geometry_id in DIAGNOSTIC_GEOMETRIES:
        row = geometry_world.get((geometry_id, "genealogy"))
        if row is None:
            continue
        if best_row is None or (
            row["mean_delta_rotor_vs_F"] is not None
            and best_row["mean_delta_rotor_vs_F"] is not None
            and row["mean_delta_rotor_vs_F"] > best_row["mean_delta_rotor_vs_F"]
        ):
            best_row = row

    if best_row is None or baseline_delta is None or best_row["mean_delta_rotor_vs_F"] is None:
        return ("inconclusive", {})

    genealogy_gain = best_row["mean_delta_rotor_vs_F"] - baseline_delta
    temporal_gain = None
    reachability_gain = None
    temporal_row = geometry_world.get((best_row["geometry_id"], "temporal"))
    temporal_base = geometry_world.get(("inside_span", "temporal"))
    reach_row = geometry_world.get((best_row["geometry_id"], "reachability"))
    reach_base = geometry_world.get(("inside_span", "reachability"))
    if temporal_row and temporal_base:
        if temporal_row["mean_delta_rotor_vs_F"] is not None and temporal_base["mean_delta_rotor_vs_F"] is not None:
            temporal_gain = temporal_row["mean_delta_rotor_vs_F"] - temporal_base["mean_delta_rotor_vs_F"]
    if reach_row and reach_base:
        if reach_row["mean_delta_rotor_vs_F"] is not None and reach_base["mean_delta_rotor_vs_F"] is not None:
            reachability_gain = reach_row["mean_delta_rotor_vs_F"] - reach_base["mean_delta_rotor_vs_F"]

    meta = {
        "baseline_genealogy_delta": baseline_delta,
        "best_geometry_id": best_row["geometry_id"],
        "best_genealogy_delta": best_row["mean_delta_rotor_vs_F"],
        "genealogy_gain": genealogy_gain,
        "temporal_gain_same_geometry": temporal_gain,
        "reachability_gain_same_geometry": reachability_gain,
    }

    if genealogy_gain < 0.01:
        return ("residual-still-weak-on-genealogy", meta)

    temporal_ok = temporal_gain is None or genealogy_gain > temporal_gain
    reach_ok = reachability_gain is None or genealogy_gain > reachability_gain
    if genealogy_gain >= 0.03 and (temporal_ok or reach_ok):
        return ("label-geometry-mismatch-supported", meta)
    return ("inconclusive", meta)


def build_report(
    decision: str,
    decision_meta: Dict[str, Any],
    global_rows: Sequence[Dict[str, Any]],
    world_rows: Sequence[Dict[str, Any]],
    selected_rows: Sequence[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append("# Gate5 Genealogy Label Geometry Report")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"- decision: `{decision}`")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- fixed boundary: FWHT baseline")
    lines.append("- fixed comparator: `rotor_loop_chordal_v1`")
    lines.append("- canonical CFA labels remain unchanged")
    lines.append("- diagnostic geometries only")
    lines.append("")
    lines.append("## All Frustrated Summary")
    lines.append("")
    lines.append("| geometry | mean_auprc_F | mean_auprc_rotor | mean_delta_rotor_vs_F | mean_hit_at_10_F | mean_hit_at_10_rotor | rotor_before_rate |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in global_rows:
        lines.append(
            f"| {row['geometry_id']} | {render_float(row['mean_auprc_F'])} | "
            f"{render_float(row['mean_auprc_rotor'])} | {render_float(row['mean_delta_rotor_vs_F'])} | "
            f"{render_float(row['mean_hit_at_10_F'])} | {render_float(row['mean_hit_at_10_rotor'])} | "
            f"{render_float(row['rotor_first_hit_before_rate'])} |"
        )
    lines.append("")
    lines.append("## World-Type Summary")
    lines.append("")
    lines.append("| geometry | world_type | mean_delta_rotor_vs_F | rotor_before_rate | f_before_rate | mean_first_hit_rotor_distance_signed |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for row in world_rows:
        lines.append(
            f"| {row['geometry_id']} | {row['world_type']} | "
            f"{render_float(row['mean_delta_rotor_vs_F'])} | "
            f"{render_float(row['rotor_first_hit_before_rate'])} | "
            f"{render_float(row['f_first_hit_before_rate'])} | "
            f"{render_float(row['mean_first_hit_rotor_distance_signed'])} |"
        )
    lines.append("")
    lines.append("## Selected Cases")
    lines.append("")
    lines.append("| case_type | sample_id | world_type | geometry | delta_rotor_vs_F | first_hit_rotor_zone | first_hit_rotor_distance_signed |")
    lines.append("| --- | ---: | --- | --- | ---: | --- | ---: |")
    for row in selected_rows:
        lines.append(
            f"| {row['selected_case_type']} | {row['sample_id']} | {row['world_type']} | "
            f"{row['geometry_id']} | {render_float(row['delta_rotor_vs_F'])} | "
            f"{row['first_hit_rotor_zone']} | {render_float(row['first_hit_rotor_distance_signed'])} |"
        )
    lines.append("")
    lines.append("## Key Basis")
    lines.append("")
    if decision_meta:
        lines.append(f"- baseline genealogy inside-span delta: {render_float(decision_meta.get('baseline_genealogy_delta'))}")
        lines.append(f"- best diagnostic geometry: `{decision_meta.get('best_geometry_id', '')}`")
        lines.append(f"- best genealogy delta: {render_float(decision_meta.get('best_genealogy_delta'))}")
        lines.append(f"- genealogy gain vs inside-span: {render_float(decision_meta.get('genealogy_gain'))}")
        lines.append(
            f"- temporal gain on same geometry: {render_float(decision_meta.get('temporal_gain_same_geometry'))}"
        )
        lines.append(
            f"- reachability gain on same geometry: {render_float(decision_meta.get('reachability_gain_same_geometry'))}"
        )
    lines.append("")
    lines.append("This report answers only whether genealogy's persistent failure is better explained by label geometry mismatch or by residual weakness under the fixed FWHT baseline.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    gate5_out_dir = (REPO_ROOT / Path(args.gate5_out_dir)).resolve()
    out_dir = (REPO_ROOT / Path(args.out_dir)).resolve()
    sample_summary_path = gate5_out_dir / "gate5_sample_summary.csv"
    token_path = gate5_out_dir / "gate5_token_telemetry.csv"

    sample_rows_all = read_csv(sample_summary_path)
    token_rows_all = read_csv(token_path)

    sample_rows = [row for row in sample_rows_all if row.get("variant") == "frustrated"]
    if not sample_rows:
        raise SystemExit("no frustrated rows found in gate5_sample_summary.csv")

    genealogy_rows = [row for row in sample_rows if row.get("world_type") == "genealogy"]
    if not genealogy_rows:
        raise SystemExit("zero genealogy frustrated rows in gate5_sample_summary.csv")

    token_grouped = group_token_rows(token_rows_all)

    sample_summary_rows: List[Dict[str, Any]] = []
    for sample_row in sample_rows:
        sample_id = int(sample_row["sample_id"])
        token_rows = token_grouped[sample_id]
        for geometry_id in args.geometries:
            sample_summary_rows.append(
                build_per_sample_row(
                    sample_row,
                    token_rows,
                    geometry_id=geometry_id,
                    topk=args.topk,
                    percentile=args.percentile,
                )
            )

    validate_sample_ids(sample_summary_rows, args.sample_ids)

    global_rows: List[Dict[str, Any]] = []
    for geometry_id in args.geometries:
        rows = [row for row in sample_summary_rows if row["geometry_id"] == geometry_id]
        summary = summarize_rows(geometry_id, rows)
        summary["geometry_id"] = geometry_id
        global_rows.append(summary)

    world_rows: List[Dict[str, Any]] = []
    for geometry_id in args.geometries:
        for world_type in sorted({row["world_type"] for row in sample_rows}):
            rows = [
                row
                for row in sample_summary_rows
                if row["geometry_id"] == geometry_id and row["world_type"] == world_type
            ]
            summary = summarize_rows(f"{geometry_id}:{world_type}", rows)
            summary["geometry_id"] = geometry_id
            summary["world_type"] = world_type
            world_rows.append(summary)

    selected_rows = build_selected_cases(
        sample_summary_rows,
        selected_ids=args.sample_ids,
        genealogy_worst_count=args.genealogy_worst_count,
    )
    decision, decision_meta = decision_from_world_summary(world_rows)

    sample_summary_fields = [
        "sample_id",
        "variant",
        "world_type",
        "geometry_id",
        "positive_token_count",
        "defect_start_step",
        "defect_end_step",
        "auprc_F",
        "auprc_rotor",
        "delta_rotor_vs_F",
        "hit_at_10_F",
        "hit_at_10_rotor",
        "first_hit_F_step",
        "first_hit_rotor_step",
        "first_hit_F_distance_signed",
        "first_hit_rotor_distance_signed",
        "first_hit_F_zone",
        "first_hit_rotor_zone",
    ]
    global_fields = [
        "geometry_id",
        "n_samples",
        "mean_auprc_F",
        "mean_auprc_rotor",
        "mean_delta_rotor_vs_F",
        "mean_hit_at_10_F",
        "mean_hit_at_10_rotor",
        "rotor_first_hit_before_rate",
        "f_first_hit_before_rate",
        "mean_first_hit_rotor_distance_signed",
        "mean_first_hit_F_distance_signed",
    ]
    world_fields = ["geometry_id", "world_type"] + global_fields[1:]
    selected_fields = ["selected_case_type"] + sample_summary_fields

    write_csv(out_dir / "gate5_genealogy_label_geometry_sample_summary.csv", sample_summary_fields, sample_summary_rows)
    write_csv(out_dir / "gate5_genealogy_label_geometry_global_summary.csv", global_fields, global_rows)
    write_csv(out_dir / "gate5_genealogy_label_geometry_world_summary.csv", world_fields, world_rows)
    write_csv(out_dir / "gate5_genealogy_label_geometry_selected_cases.csv", selected_fields, selected_rows)

    report = build_report(decision, decision_meta, global_rows, world_rows, selected_rows)
    (out_dir / "gate5_genealogy_label_geometry_report.md").write_text(report, encoding="utf-8")

    decision_lines = [
        "# Gate5 Genealogy Label Geometry Decision",
        "",
        f"- decision: `{decision}`",
        "",
        "## Key Basis",
        "",
        f"- baseline genealogy inside-span delta: {render_float(decision_meta.get('baseline_genealogy_delta'))}",
        f"- best diagnostic geometry: `{decision_meta.get('best_geometry_id', '')}`",
        f"- best genealogy delta: {render_float(decision_meta.get('best_genealogy_delta'))}",
        f"- genealogy gain vs inside-span: {render_float(decision_meta.get('genealogy_gain'))}",
        f"- temporal gain on same geometry: {render_float(decision_meta.get('temporal_gain_same_geometry'))}",
        f"- reachability gain on same geometry: {render_float(decision_meta.get('reachability_gain_same_geometry'))}",
        "",
        "- canonical CFA labels were not modified; this is a diagnostic label-geometry comparison only.",
    ]
    (out_dir / "gate5_genealogy_label_geometry_decision.md").write_text(
        "\n".join(decision_lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
