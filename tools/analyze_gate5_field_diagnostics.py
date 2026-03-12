#!/usr/bin/env python3
"""Analyze field-side Gate5 diagnostics on an existing FWHT baseline run."""

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
            "Analyze field-side quantities for Gate5 on an existing gate5_out directory "
            "without rerunning extraction, boundaries, or motifs."
        )
    )
    parser.add_argument("--gate5-out-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sample-ids", nargs="+", type=int, default=[137, 147, 149, 11, 167])
    parser.add_argument("--top-cases", type=int, default=5)
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


def compute_metric_fields(labels: Sequence[int], scores: Sequence[Optional[float]]) -> Dict[str, Any]:
    start, end = defect_span(labels)
    if start is None or end is None:
        raise ValueError("field diagnostics require labeled defect spans")
    inside = sum_window(scores, start, end)
    before = sum_window(scores, 0, start - 1)
    after = sum_window(scores, end + 1, len(scores) - 1)
    return {
        "prefix_band_energy_w1": sum_window(scores, start - 1, start - 1),
        "prefix_band_energy_w3": sum_window(scores, start - 3, start - 1),
        "defect_start_neighborhood_mass_w1": sum_window(scores, start - 1, start + 1),
        "defect_start_neighborhood_mass_w3": sum_window(scores, start - 3, start + 3),
        "first_after_defect_score": scores[start] if scores[start] is not None else None,
        "inside_span_mass": inside,
        "before_span_mass": before,
        "after_span_mass": after,
        "span_relative_decay": (inside - after) / max(inside, EPS),
        "before_to_inside_ratio": before / max(inside, EPS),
        "inside_to_after_ratio": inside / max(after, EPS),
    }


def build_sample_row(sample_summary: Dict[str, str], token_rows: Sequence[Dict[str, str]]) -> Dict[str, Any]:
    labels = [1 if row.get("label_token") == "1" else 0 for row in token_rows]
    start, end = defect_span(labels)
    scores_f = [parse_float(row.get("score_F_loop")) for row in token_rows]
    scores_rotor = [parse_float(row.get("rotor_loop_chordal_v1")) for row in token_rows]
    fields_f = compute_metric_fields(labels, scores_f)
    fields_rotor = compute_metric_fields(labels, scores_rotor)
    aggregation_win_count = 0
    if fields_rotor["defect_start_neighborhood_mass_w1"] > fields_f["defect_start_neighborhood_mass_w1"]:
        aggregation_win_count += 1
    if fields_rotor["defect_start_neighborhood_mass_w3"] > fields_f["defect_start_neighborhood_mass_w3"]:
        aggregation_win_count += 1
    rotor_first_after = fields_rotor["first_after_defect_score"]
    f_first_after = fields_f["first_after_defect_score"]
    if rotor_first_after is not None and f_first_after is not None and rotor_first_after > f_first_after:
        aggregation_win_count += 1
    if fields_rotor["inside_to_after_ratio"] > fields_f["inside_to_after_ratio"]:
        aggregation_win_count += 1
    if fields_rotor["span_relative_decay"] > fields_f["span_relative_decay"]:
        aggregation_win_count += 1
    if fields_rotor["before_to_inside_ratio"] < fields_f["before_to_inside_ratio"]:
        aggregation_win_count += 1
    row: Dict[str, Any] = {
        "sample_id": int(sample_summary["sample_id"]),
        "variant": sample_summary.get("variant", ""),
        "world_type": sample_summary.get("world_type", ""),
        "positive_token_count": int(sample_summary["positive_token_count"]),
        "delta_auprc_rotor_vs_F": parse_float(sample_summary.get("delta_auprc_rotor_loop_chordal_v1_vs_F")),
        "auprc_F": parse_float(sample_summary.get("auprc_F")),
        "auprc_rotor": parse_float(sample_summary.get("auprc_rotor_loop_chordal_v1")),
        "defect_start_step": start,
        "defect_end_step": end,
        "aggregation_win_count_rotor": aggregation_win_count,
    }
    for base_name, value in fields_f.items():
        row[f"{base_name}_F"] = value
    for base_name, value in fields_rotor.items():
        row[f"{base_name}_rotor"] = value
    for base_name in (
        "prefix_band_energy_w1",
        "prefix_band_energy_w3",
        "defect_start_neighborhood_mass_w1",
        "defect_start_neighborhood_mass_w3",
        "first_after_defect_score",
        "inside_span_mass",
        "before_span_mass",
        "after_span_mass",
        "span_relative_decay",
        "before_to_inside_ratio",
        "inside_to_after_ratio",
    ):
        f_value = row[f"{base_name}_F"]
        rotor_value = row[f"{base_name}_rotor"]
        row[f"delta_{base_name}_rotor_vs_F"] = None if f_value is None or rotor_value is None else rotor_value - f_value
    return row


def summarize_group(group_id: str, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "group_id": group_id,
        "n_samples": len(rows),
        "mean_delta_auprc_rotor_vs_F": mean(
            row["delta_auprc_rotor_vs_F"]
            for row in rows
            if row["delta_auprc_rotor_vs_F"] is not None
        ),
        "mean_aggregation_win_count_rotor": mean(
            float(row["aggregation_win_count_rotor"]) for row in rows
        ),
    }
    for base_name in (
        "prefix_band_energy_w1",
        "prefix_band_energy_w3",
        "defect_start_neighborhood_mass_w1",
        "defect_start_neighborhood_mass_w3",
        "first_after_defect_score",
        "inside_span_mass",
        "before_span_mass",
        "after_span_mass",
        "span_relative_decay",
        "before_to_inside_ratio",
        "inside_to_after_ratio",
    ):
        summary[f"mean_{base_name}_F"] = mean(
            row[f"{base_name}_F"] for row in rows if row[f"{base_name}_F"] is not None
        )
        summary[f"mean_{base_name}_rotor"] = mean(
            row[f"{base_name}_rotor"] for row in rows if row[f"{base_name}_rotor"] is not None
        )
        summary[f"mean_delta_{base_name}_rotor_vs_F"] = mean(
            row[f"delta_{base_name}_rotor_vs_F"]
            for row in rows
            if row[f"delta_{base_name}_rotor_vs_F"] is not None
        )
    return summary


def render_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def top_before_low_inside(rows: Sequence[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    candidates = [row for row in rows if row["before_to_inside_ratio_rotor"] is not None]
    candidates.sort(
        key=lambda row: (
            -float(row["before_to_inside_ratio_rotor"]),
            float(row["inside_span_mass_rotor"]),
            int(row["sample_id"]),
        )
    )
    return candidates[:limit]


def top_aggregation_wins(rows: Sequence[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    candidates = [row for row in rows if row["aggregation_win_count_rotor"] > 0]
    candidates.sort(
        key=lambda row: (
            -int(row["aggregation_win_count_rotor"]),
            -(row["delta_inside_to_after_ratio_rotor_vs_F"] or float("-inf")),
            -(row["delta_span_relative_decay_rotor_vs_F"] or float("-inf")),
            int(row["sample_id"]),
        )
    )
    return candidates[:limit]


def write_report(
    out_path: Path,
    manifest: Dict[str, Any],
    all_summary: Dict[str, Any],
    world_rows: Sequence[Dict[str, Any]],
    genealogy_summary: Optional[Dict[str, Any]],
    selected_rows: Sequence[Dict[str, Any]],
    high_before_rows: Sequence[Dict[str, Any]],
    aggregation_win_rows: Sequence[Dict[str, Any]],
) -> None:
    lines = [
        "# Gate5 Field Diagnostics",
        "",
        f"Run ID: {manifest.get('run_id', '')}",
        f"Method ID: {manifest.get('method_id', '')}",
        "",
        "## All Frustrated Summary",
        "",
        f"- n_samples: {all_summary['n_samples']}",
        f"- mean_delta_auprc_rotor_vs_F: {render_float(all_summary['mean_delta_auprc_rotor_vs_F'])}",
        f"- mean_before_span_mass_F: {render_float(all_summary['mean_before_span_mass_F'])}",
        f"- mean_before_span_mass_rotor: {render_float(all_summary['mean_before_span_mass_rotor'])}",
        f"- mean_inside_span_mass_F: {render_float(all_summary['mean_inside_span_mass_F'])}",
        f"- mean_inside_span_mass_rotor: {render_float(all_summary['mean_inside_span_mass_rotor'])}",
        f"- mean_after_span_mass_F: {render_float(all_summary['mean_after_span_mass_F'])}",
        f"- mean_after_span_mass_rotor: {render_float(all_summary['mean_after_span_mass_rotor'])}",
        f"- mean_before_to_inside_ratio_F: {render_float(all_summary['mean_before_to_inside_ratio_F'])}",
        f"- mean_before_to_inside_ratio_rotor: {render_float(all_summary['mean_before_to_inside_ratio_rotor'])}",
        f"- mean_inside_to_after_ratio_F: {render_float(all_summary['mean_inside_to_after_ratio_F'])}",
        f"- mean_inside_to_after_ratio_rotor: {render_float(all_summary['mean_inside_to_after_ratio_rotor'])}",
        f"- mean_span_relative_decay_F: {render_float(all_summary['mean_span_relative_decay_F'])}",
        f"- mean_span_relative_decay_rotor: {render_float(all_summary['mean_span_relative_decay_rotor'])}",
        "",
        "## World-Type Summary",
        "",
        "| world_type | n | delta_auprc | before_to_inside_F | before_to_inside_rotor | inside_to_after_F | inside_to_after_rotor | span_decay_F | span_decay_rotor | mean_aggregation_wins |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in world_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["group_id"]),
                    str(row["n_samples"]),
                    render_float(row["mean_delta_auprc_rotor_vs_F"]),
                    render_float(row["mean_before_to_inside_ratio_F"]),
                    render_float(row["mean_before_to_inside_ratio_rotor"]),
                    render_float(row["mean_inside_to_after_ratio_F"]),
                    render_float(row["mean_inside_to_after_ratio_rotor"]),
                    render_float(row["mean_span_relative_decay_F"]),
                    render_float(row["mean_span_relative_decay_rotor"]),
                    render_float(row["mean_aggregation_win_count_rotor"]),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Genealogy-Only Summary", ""])
    if genealogy_summary is None:
        lines.append("- no genealogy frustrated samples")
    else:
        lines.extend(
            [
                f"- mean_delta_auprc_rotor_vs_F: {render_float(genealogy_summary['mean_delta_auprc_rotor_vs_F'])}",
                f"- mean_before_to_inside_ratio_F: {render_float(genealogy_summary['mean_before_to_inside_ratio_F'])}",
                f"- mean_before_to_inside_ratio_rotor: {render_float(genealogy_summary['mean_before_to_inside_ratio_rotor'])}",
                f"- mean_inside_to_after_ratio_F: {render_float(genealogy_summary['mean_inside_to_after_ratio_F'])}",
                f"- mean_inside_to_after_ratio_rotor: {render_float(genealogy_summary['mean_inside_to_after_ratio_rotor'])}",
                f"- mean_span_relative_decay_F: {render_float(genealogy_summary['mean_span_relative_decay_F'])}",
                f"- mean_span_relative_decay_rotor: {render_float(genealogy_summary['mean_span_relative_decay_rotor'])}",
            ]
        )

    lines.extend(["", "## Selected Cases", ""])
    for row in selected_rows:
        lines.append(
            f"- sample_id={row['sample_id']} world_type={row['world_type']} "
            f"delta_auprc_rotor_vs_F={render_float(row['delta_auprc_rotor_vs_F'])} "
            f"before_to_inside_F={render_float(row['before_to_inside_ratio_F'])} "
            f"before_to_inside_rotor={render_float(row['before_to_inside_ratio_rotor'])} "
            f"inside_to_after_F={render_float(row['inside_to_after_ratio_F'])} "
            f"inside_to_after_rotor={render_float(row['inside_to_after_ratio_rotor'])} "
            f"span_decay_F={render_float(row['span_relative_decay_F'])} "
            f"span_decay_rotor={render_float(row['span_relative_decay_rotor'])} "
            f"aggregation_win_count_rotor={row['aggregation_win_count_rotor']}"
        )

    lines.extend(["", "## Representative Cases: High Before Mass / Low Inside Mass", ""])
    for row in high_before_rows:
        lines.append(
            f"- sample_id={row['sample_id']} world_type={row['world_type']} "
            f"before_to_inside_rotor={render_float(row['before_to_inside_ratio_rotor'])} "
            f"inside_span_mass_rotor={render_float(row['inside_span_mass_rotor'])} "
            f"delta_auprc_rotor_vs_F={render_float(row['delta_auprc_rotor_vs_F'])}"
        )

    lines.extend(["", "## Representative Cases: Rotor Aggregation-Style Wins", ""])
    for row in aggregation_win_rows:
        lines.append(
            f"- sample_id={row['sample_id']} world_type={row['world_type']} "
            f"aggregation_win_count_rotor={row['aggregation_win_count_rotor']} "
            f"delta_inside_to_after={render_float(row['delta_inside_to_after_ratio_rotor_vs_F'])} "
            f"delta_span_decay={render_float(row['delta_span_relative_decay_rotor_vs_F'])} "
            f"delta_auprc_rotor_vs_F={render_float(row['delta_auprc_rotor_vs_F'])}"
        )

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

    frustrated_rows = [row for row in sample_rows if row.get("variant") == "frustrated"]
    if not frustrated_rows:
        raise ValueError("field diagnostics found zero frustrated samples")

    sample_ids_present = {int(row["sample_id"]) for row in frustrated_rows}
    missing_ids = [sample_id for sample_id in args.sample_ids if sample_id not in sample_ids_present]
    if missing_ids:
        raise ValueError(
            "requested sample_ids are not present in frustrated field-diagnostic population: "
            + ",".join(str(sample_id) for sample_id in missing_ids)
        )

    sample_field_rows: List[Dict[str, Any]] = []
    for summary_row in frustrated_rows:
        sample_id = int(summary_row["sample_id"])
        sample_field_rows.append(build_sample_row(summary_row, token_grouped[sample_id]))
    sample_field_rows.sort(key=lambda row: int(row["sample_id"]))

    all_summary = summarize_group("all", sample_field_rows)
    world_rows = []
    for world_type in sorted({str(row["world_type"]) for row in sample_field_rows}):
        rows = [row for row in sample_field_rows if row["world_type"] == world_type]
        world_rows.append(summarize_group(world_type, rows))
    genealogy_rows = [row for row in sample_field_rows if row["world_type"] == "genealogy"]
    genealogy_summary = summarize_group("genealogy", genealogy_rows) if genealogy_rows else None
    selected_rows = [row for row in sample_field_rows if int(row["sample_id"]) in args.sample_ids]
    selected_rows.sort(key=lambda row: args.sample_ids.index(int(row["sample_id"])))

    high_before_rows = top_before_low_inside(sample_field_rows, args.top_cases)
    aggregation_win_rows = top_aggregation_wins(sample_field_rows, args.top_cases)

    out_dir.mkdir(parents=True, exist_ok=True)
    sample_csv = out_dir / "gate5_field_diagnostics_sample_summary.csv"
    world_csv = out_dir / "gate5_field_diagnostics_world_summary.csv"
    selected_csv = out_dir / "gate5_field_diagnostics_selected_cases.csv"
    report_md = out_dir / "gate5_field_diagnostics_report.md"

    sample_fieldnames = [
        "sample_id",
        "variant",
        "world_type",
        "positive_token_count",
        "delta_auprc_rotor_vs_F",
        "auprc_F",
        "auprc_rotor",
        "defect_start_step",
        "defect_end_step",
        "aggregation_win_count_rotor",
    ]
    for base_name in (
        "prefix_band_energy_w1",
        "prefix_band_energy_w3",
        "defect_start_neighborhood_mass_w1",
        "defect_start_neighborhood_mass_w3",
        "first_after_defect_score",
        "inside_span_mass",
        "before_span_mass",
        "after_span_mass",
        "span_relative_decay",
        "before_to_inside_ratio",
        "inside_to_after_ratio",
    ):
        sample_fieldnames.extend(
            [f"{base_name}_F", f"{base_name}_rotor", f"delta_{base_name}_rotor_vs_F"]
        )
    write_csv(sample_csv, sample_fieldnames, sample_field_rows)

    world_fieldnames = [
        "group_id",
        "n_samples",
        "mean_delta_auprc_rotor_vs_F",
        "mean_aggregation_win_count_rotor",
    ]
    for base_name in (
        "prefix_band_energy_w1",
        "prefix_band_energy_w3",
        "defect_start_neighborhood_mass_w1",
        "defect_start_neighborhood_mass_w3",
        "first_after_defect_score",
        "inside_span_mass",
        "before_span_mass",
        "after_span_mass",
        "span_relative_decay",
        "before_to_inside_ratio",
        "inside_to_after_ratio",
    ):
        world_fieldnames.extend(
            [f"mean_{base_name}_F", f"mean_{base_name}_rotor", f"mean_delta_{base_name}_rotor_vs_F"]
        )
    world_output_rows = [all_summary, *world_rows]
    write_csv(world_csv, world_fieldnames, world_output_rows)
    write_csv(selected_csv, sample_fieldnames, selected_rows)

    write_report(
        report_md,
        manifest,
        all_summary,
        world_rows,
        genealogy_summary,
        selected_rows,
        high_before_rows,
        aggregation_win_rows,
    )

    print(f"sample_summary_csv={sample_csv.as_posix()}")
    print(f"world_summary_csv={world_csv.as_posix()}")
    print(f"selected_cases_csv={selected_csv.as_posix()}")
    print(f"report_md={report_md.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
