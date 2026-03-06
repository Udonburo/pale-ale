#!/usr/bin/env python3
"""Aggregate CFA batch results (Primary=E) with preregistered GO/NO-GO gate."""

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate per-sample CFA triality results and compute preregistered "
            "dataset-level verdict."
        )
    )
    parser.add_argument("--results-jsonl", required=True)
    parser.add_argument("--cfa-jsonl", required=True)
    parser.add_argument(
        "--out",
        default=f"attestations/triality/{dt.date.today().isoformat()}_cfa_batch_primaryE_report.txt",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--perm-r", type=int, default=2000)
    parser.add_argument("--primary-score", default="E")
    parser.add_argument("--min-per-class", type=int, default=15)
    parser.add_argument("--median-delta-threshold", type=float, default=0.02)
    parser.add_argument("--p-threshold", type=float, default=0.05)
    args = parser.parse_args()

    if args.primary_score != "E":
        parser.error("--primary-score is locked to E for this prereg run.")
    if args.seed != 7:
        parser.error("--seed is locked to 7 for this prereg run.")
    if args.perm_r != 2000:
        parser.error("--perm-r is locked to 2000 for this prereg run.")
    if args.min_per_class < 1:
        parser.error("--min-per-class must be >= 1")
    if args.p_threshold <= 0.0 or args.p_threshold >= 1.0:
        parser.error("--p-threshold must be in (0,1)")
    return args


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"non-object row at {path}:{line_no}")
            rows.append(obj)
    return rows


def parse_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    arr = sorted(float(v) for v in values)
    n = len(arr)
    mid = n // 2
    if n % 2 == 1:
        return arr[mid]
    return 0.5 * (arr[mid - 1] + arr[mid])


def mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return sum(float(v) for v in values) / float(len(values))


def fmt(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.17e}"


def mann_whitney_u(x: Sequence[float], y: Sequence[float]) -> float:
    u = 0.0
    for xv in x:
        for yv in y:
            if xv > yv:
                u += 1.0
            elif xv == yv:
                u += 0.5
    return u


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if not x or not y:
        return None
    gt = 0
    lt = 0
    for xv in x:
        for yv in y:
            if xv > yv:
                gt += 1
            elif xv < yv:
                lt += 1
    denom = len(x) * len(y)
    if denom == 0:
        return None
    return (gt - lt) / float(denom)


def empirical_p_mwu_two_sided(
    x: Sequence[float], y: Sequence[float], seed: int, r: int
) -> Optional[float]:
    if not x or not y:
        return None
    n_x = len(x)
    n_y = len(y)
    denom = float(n_x * n_y)
    obs_u = mann_whitney_u(x, y)
    obs_auc = obs_u / denom
    obs_stat = abs(obs_auc - 0.5)

    rng = random.Random(seed)
    pooled = list(float(v) for v in x) + list(float(v) for v in y)
    labels = [1] * n_x + [0] * n_y
    count_ge = 0

    for _ in range(r):
        rng.shuffle(labels)
        gx: List[float] = []
        gy: List[float] = []
        for value, lab in zip(pooled, labels):
            if lab == 1:
                gx.append(value)
            else:
                gy.append(value)
        u = mann_whitney_u(gx, gy)
        auc = u / denom
        stat = abs(auc - 0.5)
        if stat >= obs_stat:
            count_ge += 1

    return (count_ge + 1) / float(r + 1)


def collect_script_hashes() -> Dict[str, str]:
    tools_dir = Path(__file__).resolve().parent
    targets = [
        tools_dir / "run_cfa_batch_primaryE.py",
        tools_dir / "aggregate_cfa_batch.py",
        tools_dir / "extract_triality_triplets.py",
        tools_dir / "labels_from_cfa_spans.py",
        tools_dir / "eval_triality_token.py",
    ]
    out: Dict[str, str] = {}
    for path in targets:
        if path.exists():
            out[path.as_posix()] = sha256_file(path)
    return out


def summarize_status_counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for row in rows:
        key = str(row.get("status", "unknown"))
        out[key] = out.get(key, 0) + 1
    return out


def main() -> int:
    args = parse_args()
    results_path = Path(args.results_jsonl)
    cfa_path = Path(args.cfa_jsonl)
    out_path = Path(args.out)

    rows = load_jsonl(results_path)
    if not rows:
        raise ValueError(f"results JSONL is empty: {results_path}")

    status_counts = summarize_status_counts(rows)
    ok_rows = [r for r in rows if str(r.get("status")) == "ok"]
    no_positive_imputed_count = sum(
        1 for r in ok_rows if bool(r.get("no_positive_imputed", False))
    )

    ok_consistent = [
        r for r in ok_rows if str(r.get("variant", "")).strip().lower() == "consistent"
    ]
    ok_frustrated = [
        r for r in ok_rows if str(r.get("variant", "")).strip().lower() == "frustrated"
    ]

    def values(group: Sequence[Dict[str, Any]], key: str) -> List[float]:
        out_vals: List[float] = []
        for row in group:
            v = parse_float(row.get(key))
            if v is not None:
                out_vals.append(v)
        return out_vals

    c_auprc_e = values(ok_consistent, "AUPRC_E")
    f_auprc_e = values(ok_frustrated, "AUPRC_E")
    c_base = values(ok_consistent, "AUPRC_best_baseline")
    f_base = values(ok_frustrated, "AUPRC_best_baseline")
    c_delta = values(ok_consistent, "delta")
    f_delta = values(ok_frustrated, "delta")

    n_consistent = len(c_delta)
    n_frustrated = len(f_delta)

    med_c_auprc_e = median(c_auprc_e)
    med_f_auprc_e = median(f_auprc_e)
    med_c_base = median(c_base)
    med_f_base = median(f_base)
    med_c_delta = median(c_delta)
    med_f_delta = median(f_delta)

    mean_c_delta = mean(c_delta)
    mean_f_delta = mean(f_delta)

    mwu_u = mann_whitney_u(f_delta, c_delta) if (f_delta and c_delta) else None
    mwu_auc = (
        mwu_u / float(n_frustrated * n_consistent)
        if (mwu_u is not None and n_frustrated > 0 and n_consistent > 0)
        else None
    )
    cliffs = cliffs_delta(f_delta, c_delta)
    p_emp = empirical_p_mwu_two_sided(
        f_delta, c_delta, seed=args.seed, r=args.perm_r
    )

    passes_n = n_consistent >= args.min_per_class and n_frustrated >= args.min_per_class
    passes_effect = med_f_delta is not None and med_f_delta >= args.median_delta_threshold
    passes_p = p_emp is not None and p_emp < args.p_threshold

    verdict = "GO" if (passes_n and passes_effect and passes_p) else "NO-GO"

    dataset_sha = sha256_file(cfa_path)
    results_sha = sha256_file(results_path)
    script_hashes = collect_script_hashes()
    model_pairs = sorted(
        {
            (
                str(r.get("model_id", "NA")),
                str(r.get("model_revision", "NA")),
            )
            for r in ok_rows
        }
    )

    lines: List[str] = []
    lines.append(f"date={dt.date.today().isoformat()}")
    lines.append("experiment=cfa_batch_primaryE_preregistered")
    lines.append(f"primary_score={args.primary_score}")
    lines.append(f"seed={args.seed}")
    lines.append(f"perm_r={args.perm_r}")
    lines.append(f"cfa_jsonl={cfa_path.as_posix()}")
    lines.append(f"results_jsonl={results_path.as_posix()}")
    lines.append(f"cfa_sha256={dataset_sha}")
    lines.append(f"results_sha256={results_sha}")
    lines.append("")
    lines.append("status_counts:")
    lines.append(f"  total_rows={len(rows)}")
    for key in sorted(status_counts):
        lines.append(f"  status_{key}={status_counts[key]}")
    lines.append(f"  ok_rows={len(ok_rows)}")
    lines.append(f"  ok_rows_no_positive_imputed={no_positive_imputed_count}")
    lines.append("")
    lines.append("class_valid_counts:")
    lines.append(f"  consistent={n_consistent}")
    lines.append(f"  frustrated={n_frustrated}")
    lines.append("")
    lines.append("class_descriptive_stats:")
    lines.append(f"  consistent_median_auprc_e={fmt(med_c_auprc_e)}")
    lines.append(f"  frustrated_median_auprc_e={fmt(med_f_auprc_e)}")
    lines.append(f"  consistent_median_best_baseline_auprc={fmt(med_c_base)}")
    lines.append(f"  frustrated_median_best_baseline_auprc={fmt(med_f_base)}")
    lines.append(f"  consistent_median_delta_auprc={fmt(med_c_delta)}")
    lines.append(f"  frustrated_median_delta_auprc={fmt(med_f_delta)}")
    lines.append(f"  consistent_mean_delta_auprc={fmt(mean_c_delta)}")
    lines.append(f"  frustrated_mean_delta_auprc={fmt(mean_f_delta)}")
    lines.append("")
    lines.append("group_contrast_on_delta_auprc:")
    lines.append("  direction=frustrated_minus_consistent")
    lines.append(f"  mwu_u={fmt(mwu_u)}")
    lines.append(f"  mwu_auc={fmt(mwu_auc)}")
    lines.append(f"  cliffs_delta={fmt(cliffs)}")
    lines.append(f"  empirical_p_value={fmt(p_emp)}")
    lines.append("")
    lines.append("preregistered_gate:")
    lines.append(f"  condition_n_per_class_min_{args.min_per_class}={passes_n}")
    lines.append(
        f"  condition_median_delta_frustrated_ge_{args.median_delta_threshold:.2f}={passes_effect}"
    )
    lines.append(f"  condition_empirical_p_lt_{args.p_threshold:.2f}={passes_p}")
    lines.append(f"  VERDICT={verdict}")
    lines.append("")
    lines.append("model_details:")
    if model_pairs:
        for model_id, model_rev in model_pairs:
            lines.append(f"  - model_id={model_id} revision={model_rev}")
    else:
        lines.append("  - none")
    lines.append("")
    lines.append("script_sha256:")
    for script_path in sorted(script_hashes):
        lines.append(f"  {script_path}={script_hashes[script_path]}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")

    print(f"report={out_path.as_posix()}")
    print(f"verdict={verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
