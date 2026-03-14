#!/usr/bin/env python3
"""Aggregate Gate5 artifacts into CFA or Seam reports."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

GENEALOGY_DUAL_VIEW_GEOMETRIES = ("inside_span", "prefix_only_w3")
GENEALOGY_DUAL_VIEW_PERCENTILE = 0.90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Gate5 spike artifacts.")
    parser.add_argument("--gate5-out-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--surface", choices=("auto", "cfa", "seam"), default="auto")
    parser.add_argument("--cfa-jsonl")
    parser.add_argument("--seam-jsonl")
    parser.add_argument("--seam-pair-summary-out", default="")
    parser.add_argument("--seam-family-summary-out", default="")
    parser.add_argument("--topk", type=int, default=10)
    return parser.parse_args()


def parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    out = float(raw)
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


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


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


def percentile_nearest_rank(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    arr = sorted(values)
    rank = int(math.ceil(q * len(arr))) - 1
    rank = max(0, min(rank, len(arr) - 1))
    return arr[rank]


def iqr(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    q75 = percentile_nearest_rank(values, 0.75)
    q25 = percentile_nearest_rank(values, 0.25)
    if q75 is None or q25 is None:
        return None
    return float(q75) - float(q25)


def mean(values: Iterable[float]) -> Optional[float]:
    arr = list(values)
    if not arr:
        return None
    return sum(arr) / float(len(arr))


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


def first_hit_distance(rows: Sequence[Dict[str, str]], score_col: str) -> Optional[int]:
    labels = [1 if row.get("label_token") == "1" else 0 for row in rows]
    if sum(labels) == 0:
        return None
    scores = [parse_float(row.get(score_col)) for row in rows]
    valid_scores = [score for score in scores if score is not None]
    threshold = percentile_nearest_rank(valid_scores, 0.90)
    if threshold is None:
        return None
    defect_start = next((idx for idx, lab in enumerate(labels) if lab == 1), None)
    first_hit = next(
        (idx for idx, score in enumerate(scores) if score is not None and score >= threshold),
        None,
    )
    if defect_start is None or first_hit is None:
        return None
    return int(first_hit) - int(defect_start)


def sample_metric_stats(rows: Sequence[Dict[str, str]], score_col: str) -> Dict[str, Optional[float]]:
    values = [parse_float(row.get(score_col)) for row in rows]
    clean = [value for value in values if value is not None]
    return {
        "max": max(clean) if clean else None,
        "mean": mean(clean),
        "p90": percentile_nearest_rank(clean, 0.90),
        "iqr": iqr(clean),
    }


def group_token_rows(rows: Sequence[Dict[str, str]]) -> Dict[int, List[Dict[str, str]]]:
    grouped: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["sample_id"])].append(row)
    for sample_rows in grouped.values():
        sample_rows.sort(key=lambda row: int(row["step"]))
    return grouped


def defect_span(labels: Sequence[int]) -> Tuple[Optional[int], Optional[int]]:
    positive = [idx for idx, label in enumerate(labels) if label == 1]
    if not positive:
        return (None, None)
    return (positive[0], positive[-1])


def build_genealogy_geometry_labels(base_labels: Sequence[int], geometry_id: str) -> List[int]:
    if geometry_id == "inside_span":
        return list(base_labels)
    if geometry_id != "prefix_only_w3":
        raise ValueError(f"unsupported genealogy geometry id: {geometry_id}")

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


def first_hit_distance_for_labels(
    labels: Sequence[int],
    scores: Sequence[Optional[float]],
    percentile: float = GENEALOGY_DUAL_VIEW_PERCENTILE,
) -> Optional[int]:
    if sum(labels) == 0:
        return None
    valid_scores = [score for score in scores if score is not None]
    threshold = percentile_nearest_rank(valid_scores, percentile)
    if threshold is None:
        return None
    defect_start, _defect_end = defect_span(labels)
    first_hit = next(
        (idx for idx, score in enumerate(scores) if score is not None and score >= threshold),
        None,
    )
    if defect_start is None or first_hit is None:
        return None
    return int(first_hit) - int(defect_start)


def summarize_genealogy_dual_view(
    token_rows: Sequence[Dict[str, str]],
    sample_rows: Sequence[Dict[str, str]],
) -> List[Dict[str, Any]]:
    grouped = group_token_rows(token_rows)
    genealogy_sample_ids = [
        int(row["sample_id"])
        for row in sample_rows
        if row.get("variant") == "frustrated" and row.get("world_type") == "genealogy"
    ]
    if not genealogy_sample_ids:
        return []

    summary_rows: List[Dict[str, Any]] = []
    for geometry_id in GENEALOGY_DUAL_VIEW_GEOMETRIES:
        per_sample_rows: List[Dict[str, Any]] = []
        for sample_id in genealogy_sample_ids:
            sample_token_rows = grouped.get(sample_id, [])
            if not sample_token_rows:
                continue
            base_labels = [1 if row.get("label_token") == "1" else 0 for row in sample_token_rows]
            labels = build_genealogy_geometry_labels(base_labels, geometry_id)
            if sum(labels) == 0:
                continue
            scores_f = [parse_float(row.get("score_F_loop")) for row in sample_token_rows]
            scores_rotor = [
                parse_float(row.get("rotor_loop_chordal_v1")) for row in sample_token_rows
            ]
            ap_f = average_precision(labels, scores_f)
            ap_rotor = average_precision(labels, scores_rotor)
            first_hit_rotor = first_hit_distance_for_labels(labels, scores_rotor)
            per_sample_rows.append(
                {
                    "auprc_F": ap_f,
                    "auprc_rotor": ap_rotor,
                    "delta_rotor_vs_F": None
                    if ap_f is None or ap_rotor is None
                    else ap_rotor - ap_f,
                    "first_hit_rotor_distance_signed": first_hit_rotor,
                    "rotor_first_hit_before": 1.0
                    if first_hit_rotor is not None and first_hit_rotor < 0
                    else 0.0,
                }
            )

        if not per_sample_rows:
            continue

        summary_rows.append(
            {
                "geometry_id": geometry_id,
                "role": "canonical" if geometry_id == "inside_span" else "diagnostic-only",
                "n_samples": len(per_sample_rows),
                "mean_auprc_F": mean(
                    row["auprc_F"] for row in per_sample_rows if row["auprc_F"] is not None
                ),
                "mean_auprc_rotor": mean(
                    row["auprc_rotor"]
                    for row in per_sample_rows
                    if row["auprc_rotor"] is not None
                ),
                "mean_delta_rotor_vs_F": mean(
                    row["delta_rotor_vs_F"]
                    for row in per_sample_rows
                    if row["delta_rotor_vs_F"] is not None
                ),
                "mean_first_hit_rotor_distance_signed": mean(
                    float(row["first_hit_rotor_distance_signed"])
                    for row in per_sample_rows
                    if row["first_hit_rotor_distance_signed"] is not None
                ),
                "rotor_before_rate": mean(
                    row["rotor_first_hit_before"] for row in per_sample_rows
                ),
            }
        )
    return summary_rows


def detect_surface(sample_rows: Sequence[Dict[str, str]], surface: str, seam_jsonl: Optional[str]) -> str:
    if surface != "auto":
        return surface
    if seam_jsonl:
        return "seam"
    return "cfa" if any(parse_int(row.get("positive_token_count")) for row in sample_rows) else "seam"


def render_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def robust_normalize(delta: Optional[float], scale: Optional[float]) -> Optional[float]:
    if delta is None or scale is None or scale <= 0.0:
        return None
    return float(delta) / float(scale)


def default_seam_sidecar_path(out_path: Path, suffix: str) -> Path:
    return out_path.with_name(f"{out_path.stem}_{suffix}.csv")


def summarize_seam_families(
    paired_rows: Sequence[Dict[str, Any]], topk: int
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in paired_rows:
        grouped[str(row["family"])].append(row)
    summary_rows: List[Dict[str, Any]] = []
    for family in sorted(grouped):
        family_rows = grouped[family]
        summary_rows.append(
            {
                "family": family,
                "n_pairs": len(family_rows),
                "mean_delta_max_f": mean(
                    row["delta_max_f"] for row in family_rows if row["delta_max_f"] is not None
                ),
                "mean_delta_max_rotor": mean(
                    row["delta_max_rotor"]
                    for row in family_rows
                    if row["delta_max_rotor"] is not None
                ),
                "mean_delta_p90_f": mean(
                    row["delta_p90_f"] for row in family_rows if row["delta_p90_f"] is not None
                ),
                "mean_delta_p90_rotor": mean(
                    row["delta_p90_rotor"]
                    for row in family_rows
                    if row["delta_p90_rotor"] is not None
                ),
                "mean_delta_mean_f": mean(
                    row["delta_mean_f"] for row in family_rows if row["delta_mean_f"] is not None
                ),
                "mean_delta_mean_rotor": mean(
                    row["delta_mean_rotor"]
                    for row in family_rows
                    if row["delta_mean_rotor"] is not None
                ),
                "mean_iqr_normalized_delta_max_f": mean(
                    row["iqr_normalized_delta_max_f"]
                    for row in family_rows
                    if row["iqr_normalized_delta_max_f"] is not None
                ),
                "mean_iqr_normalized_delta_max_rotor": mean(
                    row["iqr_normalized_delta_max_rotor"]
                    for row in family_rows
                    if row["iqr_normalized_delta_max_rotor"] is not None
                ),
                f"mean_top{topk}_inflation_f": mean(
                    row["topk_inflation_f"]
                    for row in family_rows
                    if row["topk_inflation_f"] is not None
                ),
                f"mean_top{topk}_inflation_rotor": mean(
                    row["topk_inflation_rotor"]
                    for row in family_rows
                    if row["topk_inflation_rotor"] is not None
                ),
                f"mean_top{topk}_perturbation_overlap_f": mean(
                    row["perturbation_overlap_topk_f"] for row in family_rows
                ),
                f"mean_top{topk}_perturbation_overlap_rotor": mean(
                    row["perturbation_overlap_topk_rotor"] for row in family_rows
                ),
                "rotor_better_delta_max_count": sum(
                    1
                    for row in family_rows
                    if row["delta_max_f"] is not None
                    and row["delta_max_rotor"] is not None
                    and float(row["delta_max_rotor"]) < float(row["delta_max_f"])
                ),
                "rotor_better_delta_p90_count": sum(
                    1
                    for row in family_rows
                    if row["delta_p90_f"] is not None
                    and row["delta_p90_rotor"] is not None
                    and float(row["delta_p90_rotor"]) < float(row["delta_p90_f"])
                ),
            }
        )
    return summary_rows


def build_cfa_report(
    manifest: Dict[str, Any],
    token_rows: Sequence[Dict[str, str]],
    sample_rows: Sequence[Dict[str, str]],
    topk: int,
) -> str:
    grouped = group_token_rows(token_rows)
    genealogy_dual_view_rows = summarize_genealogy_dual_view(token_rows, sample_rows)
    labels = [1 if row.get("label_token") == "1" else 0 for row in token_rows]
    ap_f = average_precision(labels, [parse_float(row.get("score_F_loop")) for row in token_rows])
    ap_rotor = average_precision(
        labels, [parse_float(row.get("rotor_loop_chordal_v1")) for row in token_rows]
    )
    per_sample_delta = [
        (
            int(row["sample_id"]),
            parse_float(row.get("delta_auprc_rotor_loop_chordal_v1_vs_F")),
            row.get("variant", ""),
            row.get("world_type", ""),
        )
        for row in sample_rows
    ]
    per_sample_delta = [row for row in per_sample_delta if row[1] is not None]
    per_sample_delta.sort(key=lambda item: (-float(item[1]), item[0]))

    f_hit_values = [parse_int(row.get("hit_at_10_F")) for row in sample_rows]
    rotor_hit_values = [parse_int(row.get("hit_at_10_rotor_loop_chordal_v1")) for row in sample_rows]
    first_hit_f = [first_hit_distance(grouped[sample_id], "score_F_loop") for sample_id in grouped]
    first_hit_rotor = [
        first_hit_distance(grouped[sample_id], "rotor_loop_chordal_v1") for sample_id in grouped
    ]
    mean_auprc_f = mean(
        value
        for value in (parse_float(row.get("auprc_F")) for row in sample_rows)
        if value is not None
    )
    mean_auprc_rotor = mean(
        value
        for value in (
            parse_float(row.get("auprc_rotor_loop_chordal_v1")) for row in sample_rows
        )
        if value is not None
    )
    mean_auprc_e = mean(
        value for value in (parse_float(row.get("auprc_E")) for row in sample_rows) if value is not None
    )
    lines = [
        "# Gate5 Aggregate Report",
        "",
        "Surface: CFA",
        f"Run ID: {manifest.get('run_id', '')}",
        f"Method ID: {manifest.get('method_id', '')}",
        "",
        "## Primary Question",
        "",
        "Does `rotor_loop_chordal_v1` preserve useful defect localization while remaining a seam-oriented transport residual rather than a distance sum?",
        "",
        "## Token-Level Summary",
        "",
        f"- global_auprc_F: {render_float(ap_f)}",
        f"- global_auprc_rotor_loop_chordal_v1: {render_float(ap_rotor)}",
        f"- mean_sample_auprc_F: {render_float(mean_auprc_f)}",
        f"- mean_sample_auprc_rotor_loop_chordal_v1: {render_float(mean_auprc_rotor)}",
        f"- mean_sample_hit_at_{topk}_F: {render_float(mean(v for v in f_hit_values if v is not None))}",
        f"- mean_sample_hit_at_{topk}_rotor: {render_float(mean(v for v in rotor_hit_values if v is not None))}",
        f"- mean_first_hit_distance_F: {render_float(mean(v for v in first_hit_f if v is not None))}",
        f"- mean_first_hit_distance_rotor: {render_float(mean(v for v in first_hit_rotor if v is not None))}",
        "",
        "## Alignment Caveat",
        "",
        f"- mean_sample_auprc_E_transition_aligned: {render_float(mean_auprc_e)}",
        "- `E` is transition-aligned and should not be read as token-aligned evidence.",
        "",
    ]
    if genealogy_dual_view_rows:
        lines.extend(
            [
                "## Genealogy Dual-View",
                "",
                "- Headline genealogy reporting remains canonical `inside_span`.",
                "- `prefix_only_w3` is diagnostic-only and must not replace the canonical leaderboard.",
                "",
                "| geometry | role | n_samples | mean_auprc_F | mean_auprc_rotor | mean_delta_rotor_vs_F | mean_first_hit_rotor_distance_signed | rotor_before_rate |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in genealogy_dual_view_rows:
            lines.append(
                f"| {row['geometry_id']} | {row['role']} | {row['n_samples']} | "
                f"{render_float(row['mean_auprc_F'])} | {render_float(row['mean_auprc_rotor'])} | "
                f"{render_float(row['mean_delta_rotor_vs_F'])} | "
                f"{render_float(row['mean_first_hit_rotor_distance_signed'])} | "
                f"{render_float(row['rotor_before_rate'])} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Representative Samples",
            "",
        ]
    )
    if not per_sample_delta:
        lines.append("- No per-sample delta rows were available.")
    else:
        for sample_id, delta, variant, world_type in per_sample_delta[:5]:
            lines.append(
                f"- sample_id={sample_id} delta_auprc_rotor_vs_F={render_float(delta)} variant={variant} world_type={world_type}"
            )
    lines.extend(
        [
            "",
            "## Conclusion Envelope",
            "",
            "- Allowed reading: transport residual may help, may fail, or may trade quietness against localization.",
        ]
    )
    return "\n".join(lines) + "\n"


def overlap_with_spans(row: Dict[str, str], spans: Sequence[Dict[str, Any]]) -> bool:
    start = parse_int(row.get("answer_char_start"))
    end = parse_int(row.get("answer_char_end"))
    if start is None or end is None:
        return False
    for span in spans:
        span_start = int(span["start"])
        span_end = int(span["end"])
        if end > span_start and start < span_end:
            return True
    return False


def build_seam_report(
    manifest: Dict[str, Any],
    token_rows: Sequence[Dict[str, str]],
    seam_rows: Sequence[Dict[str, Any]],
    topk: int,
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped = group_token_rows(token_rows)
    pairs: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in seam_rows:
        pairs[int(row["pair_id"])][str(row["challenge_class"])] = row

    paired_rows = []
    representative: List[Tuple[float, int, str]] = []
    for pair_id, pair in sorted(pairs.items()):
        clean = pair.get("clean_consistent")
        perturbed = pair.get("seam_perturbed_consistent")
        if clean is None or perturbed is None:
            continue
        expected_clean_id = int(perturbed.get("source_sample_id", clean["sample_id"]))
        if expected_clean_id != int(clean["sample_id"]):
            continue
        clean_rows = grouped.get(int(clean["sample_id"]), [])
        perturbed_rows = grouped.get(int(perturbed["sample_id"]), [])
        if not clean_rows or not perturbed_rows:
            continue
        clean_f = sample_metric_stats(clean_rows, "score_F_loop")
        pert_f = sample_metric_stats(perturbed_rows, "score_F_loop")
        clean_r = sample_metric_stats(clean_rows, "rotor_loop_chordal_v1")
        pert_r = sample_metric_stats(perturbed_rows, "rotor_loop_chordal_v1")
        clean_f_p90 = clean_f["p90"]
        clean_r_p90 = clean_r["p90"]
        pert_spans = perturbed.get("perturbation_spans", [])
        perturbed_sorted_f = sorted(
            [row for row in perturbed_rows if parse_float(row.get("score_F_loop")) is not None],
            key=lambda row: (-float(parse_float(row.get("score_F_loop"))), int(row["step"])),
        )
        perturbed_sorted_r = sorted(
            [row for row in perturbed_rows if parse_float(row.get("rotor_loop_chordal_v1")) is not None],
            key=lambda row: (-float(parse_float(row.get("rotor_loop_chordal_v1"))), int(row["step"])),
        )
        f_top = perturbed_sorted_f[:topk]
        r_top = perturbed_sorted_r[:topk]
        paired = {
            "pair_id": pair_id,
            "family": str(perturbed["perturbation_family"]),
            "clean_sample_id": int(clean["sample_id"]),
            "perturbed_sample_id": int(perturbed["sample_id"]),
            "delta_max_f": None if pert_f["max"] is None or clean_f["max"] is None else float(pert_f["max"]) - float(clean_f["max"]),
            "delta_max_rotor": None if pert_r["max"] is None or clean_r["max"] is None else float(pert_r["max"]) - float(clean_r["max"]),
            "delta_p90_f": None if pert_f["p90"] is None or clean_f["p90"] is None else float(pert_f["p90"]) - float(clean_f["p90"]),
            "delta_p90_rotor": None if pert_r["p90"] is None or clean_r["p90"] is None else float(pert_r["p90"]) - float(clean_r["p90"]),
            "delta_mean_f": None if pert_f["mean"] is None or clean_f["mean"] is None else float(pert_f["mean"]) - float(clean_f["mean"]),
            "delta_mean_rotor": None if pert_r["mean"] is None or clean_r["mean"] is None else float(pert_r["mean"]) - float(clean_r["mean"]),
            "iqr_normalized_delta_max_f": robust_normalize(
                None if pert_f["max"] is None or clean_f["max"] is None else float(pert_f["max"]) - float(clean_f["max"]),
                clean_f["iqr"],
            ),
            "iqr_normalized_delta_max_rotor": robust_normalize(
                None if pert_r["max"] is None or clean_r["max"] is None else float(pert_r["max"]) - float(clean_r["max"]),
                clean_r["iqr"],
            ),
            "topk_inflation_f": None if clean_f_p90 is None else sum(
                1 for row in f_top if float(parse_float(row.get("score_F_loop"))) >= float(clean_f_p90)
            ),
            "topk_inflation_rotor": None if clean_r_p90 is None else sum(
                1 for row in r_top if float(parse_float(row.get("rotor_loop_chordal_v1"))) >= float(clean_r_p90)
            ),
            "perturbation_overlap_topk_f": sum(1 for row in f_top if overlap_with_spans(row, pert_spans)),
            "perturbation_overlap_topk_rotor": sum(1 for row in r_top if overlap_with_spans(row, pert_spans)),
        }
        paired_rows.append(paired)
        if paired["delta_max_f"] is not None and paired["delta_max_rotor"] is not None:
            representative.append(
                (
                    float(paired["delta_max_f"]) - float(paired["delta_max_rotor"]),
                    pair_id,
                    str(perturbed["perturbation_family"]),
                )
            )

    if not paired_rows:
        raise RuntimeError(
            "seam aggregation found zero complete linked pairs; "
            "check that --seam-jsonl matches the Gate5 telemetry sample ids and pair linkage"
        )

    representative.sort(reverse=True)
    family_summary_rows = summarize_seam_families(paired_rows, topk=topk)
    lines = [
        "# Gate5 Aggregate Report",
        "",
        "Surface: Seam Challenge v0",
        f"Run ID: {manifest.get('run_id', '')}",
        f"Method ID: {manifest.get('method_id', '')}",
        "",
        "## Paired Quietness Summary",
        "",
        f"- mean_delta_max_F: {render_float(mean(row['delta_max_f'] for row in paired_rows if row['delta_max_f'] is not None))}",
        f"- mean_delta_max_rotor: {render_float(mean(row['delta_max_rotor'] for row in paired_rows if row['delta_max_rotor'] is not None))}",
        f"- mean_delta_p90_F: {render_float(mean(row['delta_p90_f'] for row in paired_rows if row['delta_p90_f'] is not None))}",
        f"- mean_delta_p90_rotor: {render_float(mean(row['delta_p90_rotor'] for row in paired_rows if row['delta_p90_rotor'] is not None))}",
        f"- mean_delta_mean_F: {render_float(mean(row['delta_mean_f'] for row in paired_rows if row['delta_mean_f'] is not None))}",
        f"- mean_delta_mean_rotor: {render_float(mean(row['delta_mean_rotor'] for row in paired_rows if row['delta_mean_rotor'] is not None))}",
        f"- mean_iqr_normalized_delta_max_F: {render_float(mean(row['iqr_normalized_delta_max_f'] for row in paired_rows if row['iqr_normalized_delta_max_f'] is not None))}",
        f"- mean_iqr_normalized_delta_max_rotor: {render_float(mean(row['iqr_normalized_delta_max_rotor'] for row in paired_rows if row['iqr_normalized_delta_max_rotor'] is not None))}",
        "- `iqr_normalized_delta_max_*` is the required robust scale-normalized quietness summary.",
        "",
        "## Spike Inflation",
        "",
        f"- mean_top{topk}_inflation_F_vs_clean_p90: {render_float(mean(row['topk_inflation_f'] for row in paired_rows if row['topk_inflation_f'] is not None))}",
        f"- mean_top{topk}_inflation_rotor_vs_clean_p90: {render_float(mean(row['topk_inflation_rotor'] for row in paired_rows if row['topk_inflation_rotor'] is not None))}",
        f"- mean_top{topk}_perturbation_overlap_F: {render_float(mean(row['perturbation_overlap_topk_f'] for row in paired_rows))}",
        f"- mean_top{topk}_perturbation_overlap_rotor: {render_float(mean(row['perturbation_overlap_topk_rotor'] for row in paired_rows))}",
        "",
        "## Representative Pairs",
        "",
    ]
    if not representative:
        lines.append("- No complete pairs were available.")
    else:
        for delta_gap, pair_id, family in representative[:5]:
            lines.append(
                f"- pair_id={pair_id} family={family} delta_max_F_minus_rotor={render_float(delta_gap)}"
            )
    lines.extend(
        [
            "",
            "## Conclusion Envelope",
            "",
            "- Seam is evaluated as quietness, not as contradiction-positive detection.",
        ]
    )
    return "\n".join(lines) + "\n", paired_rows, family_summary_rows


def main() -> int:
    args = parse_args()
    gate5_out = Path(args.gate5_out_dir)
    manifest = read_json(gate5_out / "manifest.json")
    token_rows = read_csv(gate5_out / "gate5_token_telemetry.csv")
    sample_rows = read_csv(gate5_out / "gate5_sample_summary.csv")
    surface = detect_surface(sample_rows, args.surface, args.seam_jsonl)
    out_path = Path(args.out)

    if surface == "seam":
        if not args.seam_jsonl:
            raise RuntimeError(
                "seam aggregation requires --seam-jsonl for pair linkage and perturbation spans"
            )
        seam_rows = read_jsonl(Path(args.seam_jsonl))
        report, paired_rows, family_summary_rows = build_seam_report(
            manifest, token_rows, seam_rows, topk=args.topk
        )
        pair_summary_out = (
            Path(args.seam_pair_summary_out)
            if args.seam_pair_summary_out
            else default_seam_sidecar_path(out_path, "seam_pair_summary")
        )
        family_summary_out = (
            Path(args.seam_family_summary_out)
            if args.seam_family_summary_out
            else default_seam_sidecar_path(out_path, "seam_family_summary")
        )
        write_csv(
            pair_summary_out,
            fieldnames=[
                "pair_id",
                "family",
                "clean_sample_id",
                "perturbed_sample_id",
                "delta_max_f",
                "delta_max_rotor",
                "delta_p90_f",
                "delta_p90_rotor",
                "delta_mean_f",
                "delta_mean_rotor",
                "iqr_normalized_delta_max_f",
                "iqr_normalized_delta_max_rotor",
                "topk_inflation_f",
                "topk_inflation_rotor",
                "perturbation_overlap_topk_f",
                "perturbation_overlap_topk_rotor",
            ],
            rows=paired_rows,
        )
        write_csv(
            family_summary_out,
            fieldnames=[
                "family",
                "n_pairs",
                "mean_delta_max_f",
                "mean_delta_max_rotor",
                "mean_delta_p90_f",
                "mean_delta_p90_rotor",
                "mean_delta_mean_f",
                "mean_delta_mean_rotor",
                "mean_iqr_normalized_delta_max_f",
                "mean_iqr_normalized_delta_max_rotor",
                f"mean_top{args.topk}_inflation_f",
                f"mean_top{args.topk}_inflation_rotor",
                f"mean_top{args.topk}_perturbation_overlap_f",
                f"mean_top{args.topk}_perturbation_overlap_rotor",
                "rotor_better_delta_max_count",
                "rotor_better_delta_p90_count",
            ],
            rows=family_summary_rows,
        )
    else:
        report = build_cfa_report(manifest, token_rows, sample_rows, topk=args.topk)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8", newline="\n")
    print(out_path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
