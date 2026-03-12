#!/usr/bin/env python3
"""Run span-dilation sensitivity diagnostics on an existing Gate5 CFA run."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate label-dilation sensitivity on an existing gate5_out directory "
            "without changing scores, boundaries, or motifs."
        )
    )
    parser.add_argument("--gate5-out-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--k-values", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--sample-ids", nargs="+", type=int, default=[137, 147, 149, 11, 167])
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


def parse_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    return int(raw)


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


def defect_span(labels: Sequence[int]) -> Tuple[Optional[int], Optional[int]]:
    steps = [idx for idx, label in enumerate(labels) if label == 1]
    if not steps:
        return (None, None)
    return (steps[0], steps[-1])


def zone_for_step(step: Optional[int], defect_start: Optional[int], defect_end: Optional[int]) -> str:
    if step is None or defect_start is None or defect_end is None:
        return ""
    if step < defect_start:
        return "before"
    if step > defect_end:
        return "after"
    return "inside"


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
    first_hit_after = None
    if threshold is not None and defect_start is not None:
        for idx in range(defect_start, len(scores)):
            score = scores[idx]
            if score is not None and score >= threshold:
                first_hit_after = idx
                break
    signed = None
    if defect_start is not None and first_hit is not None:
        signed = int(first_hit) - int(defect_start)
    after_dist = None
    if defect_start is not None and first_hit_after is not None:
        after_dist = int(first_hit_after) - int(defect_start)
    return {
        "threshold": threshold,
        "defect_start": defect_start,
        "defect_end": defect_end,
        "first_hit_step": first_hit,
        "first_hit_distance_signed": signed,
        "first_hit_after_defect_distance": after_dist,
        "first_hit_zone": zone_for_step(first_hit, defect_start, defect_end),
    }


def group_token_rows(rows: Sequence[Dict[str, str]]) -> Dict[int, List[Dict[str, str]]]:
    grouped: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["sample_id"])].append(row)
    for sample_rows in grouped.values():
        sample_rows.sort(key=lambda row: int(row["step"]))
    return grouped


def build_per_sample_row(
    sample_summary: Dict[str, str],
    token_rows: Sequence[Dict[str, str]],
    k: int,
    topk: int,
    percentile: float,
) -> Dict[str, Any]:
    base_labels = [1 if row.get("label_token") == "1" else 0 for row in token_rows]
    labels = dilate_labels(base_labels, k)
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
        "k": k,
        "sample_id": int(sample_summary["sample_id"]),
        "variant": sample_summary.get("variant", ""),
        "world_type": sample_summary.get("world_type", ""),
        "positive_token_count_dilated": sum(labels),
        "auprc_F_dilated": ap_f,
        "auprc_rotor_dilated": ap_rotor,
        "delta_auprc_rotor_vs_F_dilated": None if ap_f is None or ap_rotor is None else ap_rotor - ap_f,
        "hit_at_10_F_dilated": hit_f,
        "hit_at_10_rotor_dilated": hit_rotor,
        "defect_start_step_dilated": defect_start,
        "defect_end_step_dilated": defect_end,
        "first_hit_F_step": first_f["first_hit_step"],
        "first_hit_rotor_step": first_rotor["first_hit_step"],
        "first_hit_F_distance_signed": first_f["first_hit_distance_signed"],
        "first_hit_rotor_distance_signed": first_rotor["first_hit_distance_signed"],
        "first_hit_F_after_defect_distance": first_f["first_hit_after_defect_distance"],
        "first_hit_rotor_after_defect_distance": first_rotor["first_hit_after_defect_distance"],
        "first_hit_F_zone": first_f["first_hit_zone"],
        "first_hit_rotor_zone": first_rotor["first_hit_zone"],
    }


def summarize_global(
    token_grouped: Dict[int, List[Dict[str, str]]],
    sample_rows: Sequence[Dict[str, str]],
    k: int,
    topk: int,
    percentile: float,
) -> Dict[str, Any]:
    all_labels: List[int] = []
    all_scores_f: List[Optional[float]] = []
    all_scores_rotor: List[Optional[float]] = []
    per_sample: List[Dict[str, Any]] = []
    for summary_row in sample_rows:
        sample_id = int(summary_row["sample_id"])
        token_rows = token_grouped[sample_id]
        sample_result = build_per_sample_row(summary_row, token_rows, k, topk, percentile)
        per_sample.append(sample_result)
        labels = dilate_labels([1 if row.get("label_token") == "1" else 0 for row in token_rows], k)
        all_labels.extend(labels)
        all_scores_f.extend(parse_float(row.get("score_F_loop")) for row in token_rows)
        all_scores_rotor.extend(parse_float(row.get("rotor_loop_chordal_v1")) for row in token_rows)

    return {
        "summary": {
            "k": k,
            "global_auprc_F": average_precision(all_labels, all_scores_f),
            "global_auprc_rotor": average_precision(all_labels, all_scores_rotor),
            "mean_sample_auprc_F": mean(
                row["auprc_F_dilated"] for row in per_sample if row["auprc_F_dilated"] is not None
            ),
            "mean_sample_auprc_rotor": mean(
                row["auprc_rotor_dilated"] for row in per_sample if row["auprc_rotor_dilated"] is not None
            ),
            "mean_first_hit_before_rate_F": mean(
                1.0 if row["first_hit_F_zone"] == "before" else 0.0
                for row in per_sample
                if row["first_hit_F_zone"]
            ),
            "mean_first_hit_before_rate_rotor": mean(
                1.0 if row["first_hit_rotor_zone"] == "before" else 0.0
                for row in per_sample
                if row["first_hit_rotor_zone"]
            ),
        },
        "per_sample": per_sample,
    }


def summarize_world_types(
    per_sample_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in per_sample_rows:
        if row["variant"] != "frustrated":
            continue
        grouped[(int(row["k"]), str(row["world_type"]))].append(row)
    out: List[Dict[str, Any]] = []
    for (k, world_type) in sorted(grouped):
        rows = grouped[(k, world_type)]
        out.append(
            {
                "k": k,
                "world_type": world_type,
                "n_samples": len(rows),
                "mean_delta_auprc_rotor_vs_F": mean(
                    row["delta_auprc_rotor_vs_F_dilated"]
                    for row in rows
                    if row["delta_auprc_rotor_vs_F_dilated"] is not None
                ),
                "mean_first_hit_distance_F": mean(
                    row["first_hit_F_distance_signed"]
                    for row in rows
                    if row["first_hit_F_distance_signed"] is not None
                ),
                "mean_first_hit_distance_rotor": mean(
                    row["first_hit_rotor_distance_signed"]
                    for row in rows
                    if row["first_hit_rotor_distance_signed"] is not None
                ),
                "first_hit_before_rate_F": mean(
                    1.0 if row["first_hit_F_zone"] == "before" else 0.0
                    for row in rows
                    if row["first_hit_F_zone"]
                ),
                "first_hit_before_rate_rotor": mean(
                    1.0 if row["first_hit_rotor_zone"] == "before" else 0.0
                    for row in rows
                    if row["first_hit_rotor_zone"]
                ),
            }
        )
    return out


def render_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def write_report(
    out_path: Path,
    manifest: Dict[str, Any],
    global_rows: Sequence[Dict[str, Any]],
    world_rows: Sequence[Dict[str, Any]],
    selected_rows: Sequence[Dict[str, Any]],
) -> None:
    genealogy_rows = [row for row in world_rows if str(row["world_type"]) == "genealogy"]
    lines = [
        "# Gate5 Span-Dilation Sensitivity",
        "",
        f"Run ID: {manifest.get('run_id', '')}",
        f"Method ID: {manifest.get('method_id', '')}",
        "",
        "Span dilation is a calibration / signal-shape diagnostic only and does not by itself justify comparator promotion.",
        "",
        "## Global Summary",
        "",
        "| k | global_AUPRC_F | global_AUPRC_rotor | mean_sample_AUPRC_F | mean_sample_AUPRC_rotor | first_hit_before_F | first_hit_before_rotor |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in global_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["k"]),
                    render_float(row["global_auprc_F"]),
                    render_float(row["global_auprc_rotor"]),
                    render_float(row["mean_sample_auprc_F"]),
                    render_float(row["mean_sample_auprc_rotor"]),
                    render_float(row["mean_first_hit_before_rate_F"]),
                    render_float(row["mean_first_hit_before_rate_rotor"]),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Frustrated World-Type Summary", "", "| k | world_type | n | mean_delta_rotor_vs_F | mean_first_hit_F | mean_first_hit_rotor | first_hit_before_F | first_hit_before_rotor |", "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in world_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["k"]),
                    str(row["world_type"]),
                    str(row["n_samples"]),
                    render_float(row["mean_delta_auprc_rotor_vs_F"]),
                    render_float(row["mean_first_hit_distance_F"]),
                    render_float(row["mean_first_hit_distance_rotor"]),
                    render_float(row["first_hit_before_rate_F"]),
                    render_float(row["first_hit_before_rate_rotor"]),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Genealogy-Only Summary", ""])
    if not genealogy_rows:
        lines.append("- no genealogy frustrated rows")
    else:
        lines.extend(
            [
                "| k | mean_delta_rotor_vs_F | mean_first_hit_F | mean_first_hit_rotor | first_hit_before_F | first_hit_before_rotor |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in genealogy_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["k"]),
                        render_float(row["mean_delta_auprc_rotor_vs_F"]),
                        render_float(row["mean_first_hit_distance_F"]),
                        render_float(row["mean_first_hit_distance_rotor"]),
                        render_float(row["first_hit_before_rate_F"]),
                        render_float(row["first_hit_before_rate_rotor"]),
                    ]
                )
                + " |"
            )

    lines.extend(["", "## Selected Cases", ""])
    current_k = None
    for row in selected_rows:
        if current_k != row["k"]:
            current_k = row["k"]
            lines.extend([f"### k={current_k}", ""])
        lines.extend(
            [
                f"- sample_id={row['sample_id']} world_type={row['world_type']} delta_rotor_vs_F={render_float(row['delta_auprc_rotor_vs_F_dilated'])} first_hit_F={row['first_hit_F_distance_signed']} first_hit_rotor={row['first_hit_rotor_distance_signed']} zone_F={row['first_hit_F_zone']} zone_rotor={row['first_hit_rotor_zone']}",
            ]
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

    sample_ids_present = {int(row["sample_id"]) for row in sample_rows}
    missing_ids = [sample_id for sample_id in args.sample_ids if sample_id not in sample_ids_present]
    if missing_ids:
        raise ValueError(
            "requested sample_ids are not present in gate5 sample summary: "
            + ",".join(str(sample_id) for sample_id in missing_ids)
        )

    all_per_sample: List[Dict[str, Any]] = []
    global_rows: List[Dict[str, Any]] = []
    for k in args.k_values:
        if k < 0:
            raise ValueError("--k-values must be >= 0")
        result = summarize_global(
            token_grouped=token_grouped,
            sample_rows=sample_rows,
            k=k,
            topk=args.topk,
            percentile=args.percentile,
        )
        global_rows.append(result["summary"])
        all_per_sample.extend(result["per_sample"])

    world_rows = summarize_world_types(all_per_sample)
    selected_rows = [
        row for row in all_per_sample if int(row["sample_id"]) in {int(sample_id) for sample_id in args.sample_ids}
    ]
    selected_rows.sort(key=lambda row: (int(row["k"]), args.sample_ids.index(int(row["sample_id"]))))

    out_dir.mkdir(parents=True, exist_ok=True)
    global_csv = out_dir / "gate5_span_dilation_global_summary.csv"
    world_csv = out_dir / "gate5_span_dilation_world_summary.csv"
    selected_csv = out_dir / "gate5_span_dilation_selected_cases.csv"
    report_md = out_dir / "gate5_span_dilation_report.md"

    write_csv(
        global_csv,
        fieldnames=[
            "k",
            "global_auprc_F",
            "global_auprc_rotor",
            "mean_sample_auprc_F",
            "mean_sample_auprc_rotor",
            "mean_first_hit_before_rate_F",
            "mean_first_hit_before_rate_rotor",
        ],
        rows=global_rows,
    )
    write_csv(
        world_csv,
        fieldnames=[
            "k",
            "world_type",
            "n_samples",
            "mean_delta_auprc_rotor_vs_F",
            "mean_first_hit_distance_F",
            "mean_first_hit_distance_rotor",
            "first_hit_before_rate_F",
            "first_hit_before_rate_rotor",
        ],
        rows=world_rows,
    )
    write_csv(
        selected_csv,
        fieldnames=[
            "k",
            "sample_id",
            "variant",
            "world_type",
            "positive_token_count_dilated",
            "auprc_F_dilated",
            "auprc_rotor_dilated",
            "delta_auprc_rotor_vs_F_dilated",
            "hit_at_10_F_dilated",
            "hit_at_10_rotor_dilated",
            "defect_start_step_dilated",
            "defect_end_step_dilated",
            "first_hit_F_step",
            "first_hit_rotor_step",
            "first_hit_F_distance_signed",
            "first_hit_rotor_distance_signed",
            "first_hit_F_after_defect_distance",
            "first_hit_rotor_after_defect_distance",
            "first_hit_F_zone",
            "first_hit_rotor_zone",
        ],
        rows=selected_rows,
    )
    write_report(report_md, manifest, global_rows, world_rows, selected_rows)

    print(f"global_summary_csv={global_csv.as_posix()}")
    print(f"world_summary_csv={world_csv.as_posix()}")
    print(f"selected_cases_csv={selected_csv.as_posix()}")
    print(f"report_md={report_md.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
