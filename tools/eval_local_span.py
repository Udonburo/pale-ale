#!/usr/bin/env python3
"""Local span metrics for CFA token-level case tables."""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute local span metrics (Hit@K / First-hit distance) from token table CSV."
    )
    parser.add_argument("--token-table-csv", required=True)
    parser.add_argument("--score-col", default="score_E")
    parser.add_argument("--label-col", default="in_defect_span")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--percentile", type=float, default=0.90)
    parser.add_argument("--out-json")
    args = parser.parse_args()

    if args.topk <= 0:
        parser.error("--topk must be > 0")
    if args.percentile <= 0.0 or args.percentile > 1.0:
        parser.error("--percentile must be in (0,1]")
    return args


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


def parse_label(value: Any) -> int:
    raw = str(value).strip()
    return 1 if raw == "1" else 0


def percentile_nearest_rank(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    arr = sorted(float(v) for v in values)
    rank = int(math.ceil(q * len(arr))) - 1
    rank = max(0, min(rank, len(arr) - 1))
    return arr[rank]


def compute_metrics(
    score_values: Sequence[Optional[float]],
    defect_labels: Sequence[int],
    topk: int,
    percentile: float,
) -> Dict[str, Any]:
    if len(score_values) != len(defect_labels):
        raise ValueError("score_values and defect_labels length mismatch")

    indexed = [
        (idx, float(score))
        for idx, score in enumerate(score_values)
        if score is not None
    ]
    indexed.sort(key=lambda t: (-t[1], t[0]))
    top = indexed[:topk]

    hit_count = sum(1 for idx, _ in top if int(defect_labels[idx]) == 1)
    hit_rate = hit_count / float(topk) if topk > 0 else None

    valid_scores = [score for _, score in indexed]
    threshold = percentile_nearest_rank(valid_scores, percentile)

    defect_start = None
    for i, lab in enumerate(defect_labels):
        if int(lab) == 1:
            defect_start = i
            break

    first_hit_step = None
    if threshold is not None:
        for i, score in enumerate(score_values):
            if score is not None and float(score) >= threshold:
                first_hit_step = i
                break

    signed_dist = None
    abs_dist = None
    if defect_start is not None and first_hit_step is not None:
        signed_dist = int(first_hit_step) - int(defect_start)
        abs_dist = abs(signed_dist)

    first_hit_after_defect = None
    if threshold is not None and defect_start is not None:
        for i in range(defect_start, len(score_values)):
            score = score_values[i]
            if score is not None and float(score) >= threshold:
                first_hit_after_defect = i
                break

    after_defect_dist = None
    if first_hit_after_defect is not None and defect_start is not None:
        after_defect_dist = int(first_hit_after_defect) - int(defect_start)

    return {
        "topk": int(topk),
        "percentile": float(percentile),
        "valid_score_count": len(valid_scores),
        "defect_token_count": int(sum(1 for x in defect_labels if int(x) == 1)),
        "hit_at_k_count": int(hit_count),
        "hit_at_k_rate": hit_rate,
        "threshold_value": threshold,
        "defect_start_step": defect_start,
        "first_hit_step": first_hit_step,
        "first_hit_distance_signed": signed_dist,
        "first_hit_distance_abs": abs_dist,
        "first_hit_after_defect_step": first_hit_after_defect,
        "first_hit_after_defect_distance": after_defect_dist,
    }


def load_table(path: Path, score_col: str, label_col: str) -> Dict[str, Any]:
    score_values: List[Optional[float]] = []
    labels: List[int] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        if score_col not in reader.fieldnames:
            raise ValueError(f"score column not found: {score_col}")
        if label_col not in reader.fieldnames:
            raise ValueError(f"label column not found: {label_col}")
        for row in reader:
            score_values.append(parse_float(row.get(score_col)))
            labels.append(parse_label(row.get(label_col)))
    return {
        "score_values": score_values,
        "labels": labels,
        "n_rows": len(labels),
    }


def main() -> int:
    args = parse_args()
    table_path = Path(args.token_table_csv)
    loaded = load_table(table_path, score_col=args.score_col, label_col=args.label_col)
    metrics = compute_metrics(
        loaded["score_values"],
        loaded["labels"],
        topk=args.topk,
        percentile=args.percentile,
    )
    out = {
        "token_table_csv": table_path.as_posix(),
        "score_col": args.score_col,
        "label_col": args.label_col,
        "n_rows": loaded["n_rows"],
        "metrics": metrics,
    }

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
