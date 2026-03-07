#!/usr/bin/env python3
"""Negative stability diagnostics for CFA consistent samples under score E."""

import argparse
import datetime as dt
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import eval_triality_token as evaltok


REPORT_NAME = "consistent100_scoreE_report.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute negative-side stability diagnostics for score E using existing "
            "CFA batch artifacts."
        )
    )
    parser.add_argument("--results-jsonl", default="runs/cfa_batch_primaryE/results.jsonl")
    parser.add_argument("--cfa-jsonl", default="data/cfa/cfa_v1.jsonl")
    parser.add_argument(
        "--out",
        default=f"attestations/triality/negative_stability/{REPORT_NAME}",
    )
    parser.add_argument(
        "--top-samples",
        type=int,
        default=10,
        help="Number of consistent spike samples to inspect.",
    )
    parser.add_argument(
        "--top-transitions",
        type=int,
        default=5,
        help="Number of top score_E transitions to inspect per sample.",
    )
    args = parser.parse_args()
    if args.top_samples <= 0:
        parser.error("--top-samples must be > 0")
    if args.top_transitions <= 0:
        parser.error("--top-transitions must be > 0")
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


def quantile_nearest_rank(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("quantile_nearest_rank requires non-empty input")
    idx = int(math.ceil(q * len(sorted_values))) - 1
    idx = max(0, min(idx, len(sorted_values) - 1))
    return float(sorted_values[idx])


def summarize(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        raise ValueError("summarize requires non-empty input")
    vals = sorted(float(v) for v in values)
    mean = statistics.fmean(vals)
    std = statistics.pstdev(vals) if len(vals) >= 2 else 0.0
    return {
        "count": float(len(vals)),
        "mean": mean,
        "std": std,
        "p50": quantile_nearest_rank(vals, 0.50),
        "p90": quantile_nearest_rank(vals, 0.90),
        "p99": quantile_nearest_rank(vals, 0.99),
        "max": vals[-1],
    }


def rankdata_average(values: Sequence[float]) -> List[float]:
    indexed = sorted((float(v), i) for i, v in enumerate(values))
    ranks = [0.0] * len(indexed)
    i = 0
    n = len(indexed)
    while i < n:
        j = i + 1
        while j < n and indexed[j][0] == indexed[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            _, original_index = indexed[k]
            ranks[original_index] = avg_rank
        i = j
    return ranks


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) != len(y) or len(x) < 2:
        return None
    rx = rankdata_average(x)
    ry = rankdata_average(y)
    mx = statistics.fmean(rx)
    my = statistics.fmean(ry)
    num = 0.0
    den_x = 0.0
    den_y = 0.0
    for a, b in zip(rx, ry):
        da = a - mx
        db = b - my
        num += da * db
        den_x += da * da
        den_y += db * db
    if den_x <= 0.0 or den_y <= 0.0:
        return None
    return num / math.sqrt(den_x * den_y)


def fmt_float(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.17e}"


def repo_rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def transition_scores(rows: Sequence[Dict[str, Any]]) -> Tuple[List[float], List[float], List[float]]:
    score_e: List[float] = []
    score_a: List[float] = []
    score_b: List[float] = []
    for t in range(len(rows) - 1):
        v_t = rows[t]["V_8d"]
        v_tp1 = rows[t + 1]["V_8d"]
        sm_t = rows[t]["Sminus_8d"]
        score_e.append(evaltok.d_proj(v_t, sm_t) + evaltok.d_proj(sm_t, v_tp1))
        score_a.append(-float(rows[t]["baseline_logprob"]))
        score_b.append(float(rows[t]["baseline_entropy"]))
    return score_e, score_a, score_b


def load_cfa_index(path: Path) -> Dict[int, Dict[str, Any]]:
    rows = load_jsonl(path)
    out: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        sid = int(row["sample_id"])
        out[sid] = row
    return out


def build_sample_summary(
    result_row: Dict[str, Any],
    cfa_row: Dict[str, Any],
    repo_root: Path,
) -> Dict[str, Any]:
    sample_dir = repo_root / str(result_row["sample_dir"])
    triplets_path = sample_dir / "triplets.ndjson"
    rows = evaltok.load_ndjson(triplets_path)
    score_e, score_a, score_b = transition_scores(rows)
    if not score_e:
        raise ValueError(f"sample {result_row['sample_id']} has <2 triplet rows")
    score_e_sorted = sorted(score_e)
    return {
        "sample_id": int(result_row["sample_id"]),
        "variant": str(result_row["variant"]),
        "world_type": str(result_row.get("world_type", "")),
        "contrast_sample_id": int(cfa_row.get("contrast_sample_id", -1)),
        "sample_dir": sample_dir,
        "triplets_path": triplets_path,
        "score_e": score_e,
        "score_a": score_a,
        "score_b": score_b,
        "max_e": score_e_sorted[-1],
        "p90_e": quantile_nearest_rank(score_e_sorted, 0.90),
        "mean_e": statistics.fmean(score_e),
        "triplet_rows": rows,
        "prompt": str(cfa_row.get("prompt", "")),
        "answer": str(cfa_row.get("answer", "")),
        "defect_spans": cfa_row.get("defect_spans", []),
    }


def inspect_top_transitions(
    sample: Dict[str, Any],
    n: int,
) -> List[Dict[str, Any]]:
    rows = sample["triplet_rows"]
    score_e = sample["score_e"]
    indexed = list(range(len(score_e)))
    indexed.sort(key=lambda i: (-float(score_e[i]), i))
    out: List[Dict[str, Any]] = []
    for rank, idx in enumerate(indexed[:n], start=1):
        current_row = rows[idx]
        next_row = rows[idx + 1]
        out.append(
            {
                "rank": rank,
                "transition_step": idx,
                "score_E": float(score_e[idx]),
                "score_A": float(sample["score_a"][idx]),
                "score_B": float(sample["score_b"][idx]),
                "token_t": str(current_row.get("token_str", "")),
                "token_t1": str(next_row.get("token_str", "")),
                "char_t": (
                    int(current_row.get("answer_char_start", -1)),
                    int(current_row.get("answer_char_end", -1)),
                ),
                "char_t1": (
                    int(next_row.get("answer_char_start", -1)),
                    int(next_row.get("answer_char_end", -1)),
                ),
            }
        )
    return out


def verdict(cons_below_ratio: float) -> str:
    if cons_below_ratio >= 0.90:
        return "Green"
    if cons_below_ratio >= 0.80:
        return "Yellow"
    return "Red"


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    results_path = repo_root / args.results_jsonl
    cfa_path = repo_root / args.cfa_jsonl
    out_path = repo_root / args.out

    results_rows = load_jsonl(results_path)
    cfa_index = load_cfa_index(cfa_path)

    ok_rows = [row for row in results_rows if str(row.get("status")) == "ok"]
    consistent_rows = [row for row in ok_rows if str(row.get("variant")) == "consistent"]
    frustrated_rows = [row for row in ok_rows if str(row.get("variant")) == "frustrated"]

    consistent = [
        build_sample_summary(row, cfa_index[int(row["sample_id"])], repo_root)
        for row in consistent_rows
    ]
    frustrated = [
        build_sample_summary(row, cfa_index[int(row["sample_id"])], repo_root)
        for row in frustrated_rows
    ]

    pooled_consistent_e: List[float] = []
    pooled_consistent_a: List[float] = []
    pooled_consistent_b: List[float] = []
    for sample in consistent:
        pooled_consistent_e.extend(sample["score_e"])
        pooled_consistent_a.extend(sample["score_a"])
        pooled_consistent_b.extend(sample["score_b"])

    consistent_max = [float(sample["max_e"]) for sample in consistent]
    consistent_p90 = [float(sample["p90_e"]) for sample in consistent]
    consistent_mean = [float(sample["mean_e"]) for sample in consistent]
    frustrated_max = [float(sample["max_e"]) for sample in frustrated]
    median_frustrated_max_e = summarize(frustrated_max)["p50"]

    ratio_below = (
        sum(1 for value in consistent_max if value < median_frustrated_max_e)
        / float(len(consistent_max))
    )

    rho_e_a = spearman_rho(pooled_consistent_e, pooled_consistent_a)
    rho_e_b = spearman_rho(pooled_consistent_e, pooled_consistent_b)

    top_samples = sorted(
        consistent,
        key=lambda sample: (-float(sample["max_e"]), int(sample["sample_id"])),
    )[: args.top_samples]

    lines: List[str] = []
    lines.append(f"date={dt.date.today().isoformat()}")
    lines.append("experiment=cfa_negative_stability_consistent_scoreE")
    lines.append(f"results_jsonl={repo_rel(results_path, repo_root)}")
    lines.append(f"cfa_jsonl={repo_rel(cfa_path, repo_root)}")
    lines.append(f"results_sha256={sha256_file(results_path)}")
    lines.append(f"cfa_sha256={sha256_file(cfa_path)}")
    lines.append(f"script_sha256={sha256_file(Path(__file__))}")
    lines.append("")
    lines.append("population:")
    lines.append(f"  consistent_samples={len(consistent)}")
    lines.append(f"  frustrated_samples={len(frustrated)}")
    lines.append(f"  consistent_transitions={len(pooled_consistent_e)}")
    lines.append("")
    lines.append("score_E_run_level_distribution_consistent:")
    for key, value in summarize(pooled_consistent_e).items():
        if key == "count":
            lines.append(f"  {key}={int(value)}")
        else:
            lines.append(f"  {key}={fmt_float(value)}")
    lines.append("")
    lines.append("sample_wise_distributions_consistent:")
    lines.append("  max_E:")
    for key, value in summarize(consistent_max).items():
        if key == "count":
            lines.append(f"    {key}={int(value)}")
        else:
            lines.append(f"    {key}={fmt_float(value)}")
    lines.append("  p90_E:")
    for key, value in summarize(consistent_p90).items():
        if key == "count":
            lines.append(f"    {key}={int(value)}")
        else:
            lines.append(f"    {key}={fmt_float(value)}")
    lines.append("  mean_E:")
    for key, value in summarize(consistent_mean).items():
        if key == "count":
            lines.append(f"    {key}={int(value)}")
        else:
            lines.append(f"    {key}={fmt_float(value)}")
    lines.append("")
    lines.append("baseline_correlation_consistent_pooled:")
    lines.append(f"  spearman_rho_E_vs_A={fmt_float(rho_e_a)}")
    lines.append(f"  spearman_rho_E_vs_B={fmt_float(rho_e_b)}")
    lines.append("")
    lines.append("decision_reference:")
    lines.append(f"  median_frustrated_max_E={fmt_float(median_frustrated_max_e)}")
    lines.append(
        "  criterion=share of consistent samples with max_E < median_frustrated_max_E"
    )
    lines.append(f"  consistent_share_below_reference={fmt_float(ratio_below)}")
    lines.append(f"  verdict={verdict(ratio_below)}")
    lines.append("")
    lines.append("top_consistent_spike_samples:")
    for sample in top_samples:
        lines.append(
            "  - "
            f"sample_id={sample['sample_id']} "
            f"world_type={sample['world_type']} "
            f"contrast_sample_id={sample['contrast_sample_id']} "
            f"max_E={fmt_float(sample['max_e'])} "
            f"p90_E={fmt_float(sample['p90_e'])} "
            f"mean_E={fmt_float(sample['mean_e'])} "
            f"triplets={repo_rel(sample['triplets_path'], repo_root)}"
        )
        for row in inspect_top_transitions(sample, args.top_transitions):
            lines.append(
                "    "
                f"rank={row['rank']} "
                f"transition_step={row['transition_step']} "
                f"score_E={fmt_float(row['score_E'])} "
                f"score_A={fmt_float(row['score_A'])} "
                f"score_B={fmt_float(row['score_B'])} "
                f"token_t={json.dumps(row['token_t'], ensure_ascii=False)} "
                f"token_t1={json.dumps(row['token_t1'], ensure_ascii=False)} "
                f"char_t={row['char_t']} "
                f"char_t1={row['char_t1']}"
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"report={repo_rel(out_path, repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
