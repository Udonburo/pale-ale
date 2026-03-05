#!/usr/bin/env python3
"""Compute label-wise effect stats from per-sample CSV files (pure Python)."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute label-wise effect stats for CSV metrics.")
    parser.add_argument("--csv", dest="csv_path", help="Input per-sample CSV path.")
    parser.add_argument("--label-col", default="sample_label", help="Label column name.")
    parser.add_argument(
        "--metrics",
        help="Comma-separated metric columns to evaluate.",
    )
    # Backward-compatible alias used by older scripts.
    parser.add_argument("--gate2-samples-csv", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not args.csv_path and args.gate2_samples_csv:
        args.csv_path = args.gate2_samples_csv
        args.metrics = args.metrics or "h1b_closure_error,h3_ratio_total_product"
    if not args.csv_path:
        parser.error("--csv is required")
    if not args.metrics:
        parser.error("--metrics is required")
    return args


def parse_float(value: str) -> Optional[float]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        out = float(raw)
    except ValueError:
        return None
    if not math.isfinite(out):
        return None
    return out


def parse_label01(value: str) -> Optional[int]:
    if value is None:
        return None
    raw = str(value).strip()
    if raw == "":
        return None
    if raw in ("0", "1"):
        return int(raw)
    num = parse_float(raw)
    if num is None:
        return None
    if abs(num - 0.0) < 1e-12:
        return 0
    if abs(num - 1.0) < 1e-12:
        return 1
    return None


def cliffs_delta(group1: List[float], group0: List[float]) -> Optional[float]:
    if not group1 or not group0:
        return None
    more = 0
    less = 0
    for a in group1:
        for b in group0:
            if a > b:
                more += 1
            elif a < b:
                less += 1
    denom = len(group1) * len(group0)
    if denom == 0:
        return None
    return (more - less) / float(denom)


def mann_whitney_u_two_sided(group0: List[float], group1: List[float]) -> Dict[str, Optional[float]]:
    n0 = len(group0)
    n1 = len(group1)
    if n0 == 0 or n1 == 0:
        return {"u_stat_min": None, "z": None, "p_two_sided": None}

    pooled: List[Tuple[float, int]] = [(v, 0) for v in group0] + [(v, 1) for v in group1]
    pooled.sort(key=lambda item: item[0])

    tie_sizes: List[int] = []
    rank_sum_0 = 0.0
    i = 0
    n = len(pooled)
    while i < n:
        j = i + 1
        while j < n and pooled[j][0] == pooled[i][0]:
            j += 1
        tie_len = j - i
        tie_sizes.append(tie_len)
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            if pooled[k][1] == 0:
                rank_sum_0 += avg_rank
        i = j

    u0 = rank_sum_0 - (n0 * (n0 + 1)) / 2.0
    u1 = (n0 * n1) - u0
    u_min = min(u0, u1)

    mean_u = (n0 * n1) / 2.0
    if n <= 1:
        return {"u_stat_min": u_min, "z": 0.0, "p_two_sided": 1.0}

    tie_term = sum((t * t * t) - t for t in tie_sizes)
    var_u = (n0 * n1 / 12.0) * ((n + 1) - (tie_term / (n * (n - 1))))
    if var_u <= 0.0 or not math.isfinite(var_u):
        return {"u_stat_min": u_min, "z": 0.0, "p_two_sided": 1.0}

    if u_min > mean_u:
        cc = -0.5
    elif u_min < mean_u:
        cc = 0.5
    else:
        cc = 0.0
    z = (u_min - mean_u + cc) / math.sqrt(var_u)
    p_two_sided = math.erfc(abs(z) / math.sqrt(2.0))
    return {"u_stat_min": u_min, "z": z, "p_two_sided": p_two_sided}


def summarize_metric(values0: List[float], values1: List[float]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "n0": len(values0),
        "n1": len(values1),
        "median_label0": statistics.median(values0) if values0 else None,
        "median_label1": statistics.median(values1) if values1 else None,
        "cliffs_delta_label1_vs_label0": cliffs_delta(values1, values0),
    }
    out.update(mann_whitney_u_two_sided(values0, values1))
    return out


def collect_metric_by_label(
    rows: Iterable[Dict[str, str]], label_col: str, metric_col: str
) -> Tuple[List[float], List[float]]:
    values0: List[float] = []
    values1: List[float] = []
    for row in rows:
        label_val = parse_label01(row.get(label_col, ""))
        metric_value = parse_float(row.get(metric_col, ""))
        if metric_value is None or label_val is None:
            continue
        if label_val == 0:
            values0.append(metric_value)
        elif label_val == 1:
            values1.append(metric_value)
    return values0, values1


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv_path)
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    metrics = [m.strip() for m in str(args.metrics).split(",") if m.strip()]
    available_cols = set(rows[0].keys()) if rows else set()
    out_metrics: Dict[str, Dict[str, Optional[float]]] = {}
    missing_metrics: List[str] = []

    for metric in metrics:
        if metric not in available_cols:
            missing_metrics.append(metric)
            continue
        v0, v1 = collect_metric_by_label(rows, args.label_col, metric)
        out_metrics[metric] = summarize_metric(v0, v1)

    label0_count = 0
    label1_count = 0
    for row in rows:
        label_val = parse_label01(row.get(args.label_col, ""))
        if label_val == 0:
            label0_count += 1
        elif label_val == 1:
            label1_count += 1

    payload = {
        "source": str(csv_path.as_posix()),
        "label_col": args.label_col,
        "n_rows": len(rows),
        "label0_rows": label0_count,
        "label1_rows": label1_count,
        "metrics": out_metrics,
        "missing_metrics": missing_metrics,
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

