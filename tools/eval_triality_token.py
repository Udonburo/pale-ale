#!/usr/bin/env python3
"""Token/step-level evaluation harness for triality NDJSON outputs."""

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Go/No-Go thresholds (fixed for comparable runs)
GO_MAX_P_EMP = 0.05
GO_MIN_DELTA_AUPRC = 0.02
GO_MIN_PRIMARY_AUPRC = 0.15

# For transition scores C/D/E, label aggregation mode:
# - "max_pair": y_t = max(label_t, label_t+1)
# - "next":     y_t = label_t+1
TRANSITION_LABEL_MODE = "max_pair"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate triality NDJSON token/step scores.")
    parser.add_argument("--ndjson", required=True)
    parser.add_argument("--labels-jsonl", required=True)
    parser.add_argument("--labels-meta-json", help="Optional labels meta JSON for coverage gating.")
    parser.add_argument(
        "--min-label-coverage",
        type=float,
        default=0.0,
        help="If >0 and labels meta contains final_alignment_coverage_ratio below threshold, force NO-GO.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--perm-R", type=int, default=2000)
    parser.add_argument(
        "--primary-score",
        choices=("A", "B", "C", "D", "E", "F"),
        default="F",
        help="Primary endpoint score key for permutation + GO/NO-GO.",
    )
    parser.add_argument("--out")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def parse_label01(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value in (0, 1) else None
    raw = str(value).strip()
    if raw in ("0", "1"):
        return int(raw)
    return None


def load_ndjson(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid NDJSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object row at {path}:{line_no}")
            rows.append(row)
    if not rows:
        raise ValueError(f"empty NDJSON: {path}")
    return sorted(rows, key=lambda r: int(r.get("step", 0)))


def load_step_labels(path: Path, n_steps: int) -> Tuple[List[int], Dict[str, int]]:
    labels = [0] * n_steps
    assigned = [False] * n_steps
    stats = {
        "input_rows": 0,
        "assigned_steps": 0,
        "range_rows": 0,
        "point_rows": 0,
        "ignored_rows": 0,
    }

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            stats["input_rows"] += 1
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid labels JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                stats["ignored_rows"] += 1
                continue

            label = parse_label01(row.get("label"))
            if label is None:
                stats["ignored_rows"] += 1
                continue

            if "step" in row:
                try:
                    step = int(row["step"])
                except Exception:
                    stats["ignored_rows"] += 1
                    continue
                if 0 <= step < n_steps:
                    labels[step] = label
                    if not assigned[step]:
                        assigned[step] = True
                        stats["assigned_steps"] += 1
                    stats["point_rows"] += 1
                else:
                    stats["ignored_rows"] += 1
                continue

            if "step_start" in row and "step_end" in row:
                try:
                    step_start = int(row["step_start"])
                    step_end = int(row["step_end"])
                except Exception:
                    stats["ignored_rows"] += 1
                    continue
                start = max(0, step_start)
                end = min(n_steps, step_end)
                if end <= start:
                    stats["ignored_rows"] += 1
                    continue
                for step in range(start, end):
                    labels[step] = label
                    if not assigned[step]:
                        assigned[step] = True
                        stats["assigned_steps"] += 1
                stats["range_rows"] += 1
                continue

            stats["ignored_rows"] += 1

    stats["positive_count"] = sum(1 for x in labels if x == 1)
    stats["negative_count"] = sum(1 for x in labels if x == 0)
    return labels, stats


def load_labels_meta(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def dot_abs_clamped(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != 8 or len(y) != 8:
        raise ValueError("d_proj expects 8D vectors")
    acc = 0.0
    for a, b in zip(x, y):
        fa = parse_float(a)
        fb = parse_float(b)
        if fa is None or fb is None:
            raise ValueError("non-finite vector entry in d_proj")
        acc += fa * fb
    return min(1.0, abs(acc))


def d_proj(x: Sequence[float], y: Sequence[float]) -> float:
    inner = dot_abs_clamped(x, y)
    return math.sqrt(max(0.0, 2.0 * (1.0 - inner)))


def mann_whitney_u_auc(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    pos = [s for y, s in zip(labels, scores) if y == 1]
    neg = [s for y, s in zip(labels, scores) if y == 0]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return None

    pooled = [(float(s), int(y)) for y, s in zip(labels, scores)]
    pooled.sort(key=lambda t: t[0])  # ascending

    rank_sum_pos = 0.0
    i = 0
    n = len(pooled)
    while i < n:
        j = i + 1
        while j < n and pooled[j][0] == pooled[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            if pooled[k][1] == 1:
                rank_sum_pos += avg_rank
        i = j

    u_pos = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    return u_pos / float(n_pos * n_neg)


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    n_pos = sum(1 for y in labels if y == 1)
    if n_pos == 0:
        return None
    indexed = list(range(len(scores)))
    indexed.sort(key=lambda i: (-float(scores[i]), i))  # deterministic tie break

    tp = 0
    fp = 0
    prev_recall = 0.0
    ap = 0.0
    for idx in indexed:
        if labels[idx] == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / float(n_pos)
        precision = tp / float(tp + fp)
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return ap


def permutation_test_auprc(
    labels: Sequence[int], scores: Sequence[float], seed: int, r: int
) -> Dict[str, Optional[float]]:
    obs = average_precision(labels, scores)
    if obs is None:
        return {
            "obs_auprc": None,
            "perm_count": 0,
            "p_empirical": None,
            "perm_min": None,
            "perm_p5": None,
            "perm_p50": None,
            "perm_p95": None,
        }

    rng = random.Random(seed)
    labels_perm = list(labels)
    perms: List[float] = []
    count_ge = 0
    for _ in range(r):
        rng.shuffle(labels_perm)
        p = average_precision(labels_perm, scores)
        if p is None:
            p = 0.0
        perms.append(p)
        if p >= obs:
            count_ge += 1

    perms.sort()

    def q(values: Sequence[float], qv: float) -> float:
        rank = int(math.ceil(qv * len(values))) - 1
        rank = max(0, min(rank, len(values) - 1))
        return values[rank]

    p_emp = (count_ge + 1) / float(r + 1)
    return {
        "obs_auprc": obs,
        "perm_count": r,
        "p_empirical": p_emp,
        "perm_min": perms[0],
        "perm_p5": q(perms, 0.05),
        "perm_p50": q(perms, 0.50),
        "perm_p95": q(perms, 0.95),
    }


def fmt_float(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.17e}"


def metric_row(
    name: str,
    labels: Sequence[int],
    scores: Sequence[float],
    best_baseline_auprc: Optional[float],
) -> Dict[str, Any]:
    auroc = mann_whitney_u_auc(labels, scores)
    auprc = average_precision(labels, scores)
    delta = None
    if auprc is not None and best_baseline_auprc is not None:
        delta = auprc - best_baseline_auprc
    return {
        "name": name,
        "n": len(scores),
        "pos": sum(1 for y in labels if y == 1),
        "neg": sum(1 for y in labels if y == 0),
        "auroc": auroc,
        "auprc": auprc,
        "delta_auprc_vs_best_baseline": delta,
    }


def get_model_meta(ndjson_path: Path) -> Tuple[Optional[str], Optional[str]]:
    meta_path = ndjson_path.parent / "meta.json"
    if not meta_path.exists():
        return None, None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None, None
    if not isinstance(meta, dict):
        return None, None
    return (
        str(meta.get("model_id")) if meta.get("model_id") is not None else None,
        str(meta.get("model_revision")) if meta.get("model_revision") is not None else None,
    )


def to_repo_relative(path: Path) -> str:
    try:
        rel = os.path.relpath(str(path.resolve()), start=os.getcwd())
    except Exception:
        return path.as_posix()
    return Path(rel).as_posix()


def main() -> int:
    args = parse_args()
    ndjson_path = Path(args.ndjson)
    labels_path = Path(args.labels_jsonl)
    labels_meta_path = Path(args.labels_meta_json) if args.labels_meta_json else None
    out_path = (
        Path(args.out)
        if args.out
        else Path("attestations")
        / "triality"
        / f"{dt.date.today().isoformat()}_eval_report.txt"
    )

    rows = load_ndjson(ndjson_path)
    n_steps = len(rows)
    labels_step, label_stats = load_step_labels(labels_path, n_steps=n_steps)
    labels_meta = load_labels_meta(labels_meta_path)
    if len(labels_step) != n_steps:
        raise ValueError("internal error: labels length mismatch")

    # Scores A/B (token-step aligned: length N)
    score_a = [-float(row["baseline_logprob"]) for row in rows]
    score_b = [float(row["baseline_entropy"]) for row in rows]
    labels_token = labels_step[:]

    # Scores C/D/E (transition aligned: length N-1)
    labels_trans: List[int] = []
    score_c: List[float] = []
    score_d: List[float] = []
    score_e: List[float] = []
    for t in range(n_steps - 1):
        y0 = labels_step[t]
        y1 = labels_step[t + 1]
        yt = y1 if TRANSITION_LABEL_MODE == "next" else max(y0, y1)
        labels_trans.append(yt)

        v_t = rows[t]["V_8d"]
        v_tp1 = rows[t + 1]["V_8d"]
        sp_t = rows[t]["Splus_8d"]
        sm_t = rows[t]["Sminus_8d"]
        score_c.append(d_proj(v_t, v_tp1))
        score_d.append(d_proj(v_t, sp_t) + d_proj(sp_t, v_tp1))
        score_e.append(d_proj(v_t, sm_t) + d_proj(sm_t, v_tp1))

    # Score F (loop, token-step aligned: length N)
    score_f: List[float] = []
    for t in range(n_steps):
        v_t = rows[t]["V_8d"]
        sp_t = rows[t]["Splus_8d"]
        sm_t = rows[t]["Sminus_8d"]
        score_f.append(d_proj(v_t, sp_t) + d_proj(sp_t, sm_t) + d_proj(sm_t, v_t))

    baseline_auprcs = [
        average_precision(labels_token, score_a),
        average_precision(labels_token, score_b),
    ]
    non_null = [x for x in baseline_auprcs if x is not None]
    best_baseline_auprc = max(non_null) if non_null else None
    best_baseline_name = (
        "A:-logprob"
        if baseline_auprcs and baseline_auprcs[0] == best_baseline_auprc
        else "B:entropy"
    )

    metrics: List[Dict[str, Any]] = []
    metrics.append(metric_row("A:-logprob", labels_token, score_a, best_baseline_auprc))
    metrics.append(metric_row("B:entropy", labels_token, score_b, best_baseline_auprc))
    metrics.append(metric_row("C:V_curvature", labels_trans, score_c, best_baseline_auprc))
    metrics.append(metric_row("D:V_Splus_Vnext", labels_trans, score_d, best_baseline_auprc))
    metrics.append(metric_row("E:V_Sminus_Vnext", labels_trans, score_e, best_baseline_auprc))
    metrics.append(metric_row("F:loop_V_Splus_Sminus_V", labels_token, score_f, best_baseline_auprc))

    primary_map: Dict[str, Tuple[str, List[int], List[float]]] = {
        "A": ("A:-logprob", labels_token, score_a),
        "B": ("B:entropy", labels_token, score_b),
        "C": ("C:V_curvature", labels_trans, score_c),
        "D": ("D:V_Splus_Vnext", labels_trans, score_d),
        "E": ("E:V_Sminus_Vnext", labels_trans, score_e),
        "F": ("F:loop_V_Splus_Sminus_V", labels_token, score_f),
    }
    primary_name, primary_labels, primary_scores = primary_map[args.primary_score]

    primary_perm = permutation_test_auprc(
        primary_labels,
        primary_scores,
        seed=args.seed,
        r=args.perm_R,
    )
    primary_auprc = primary_perm["obs_auprc"]
    primary_delta = (
        (primary_auprc - best_baseline_auprc)
        if (primary_auprc is not None and best_baseline_auprc is not None)
        else None
    )
    p_emp = primary_perm["p_empirical"]

    coverage_ratio = parse_float(labels_meta.get("final_alignment_coverage_ratio"))
    coverage_gate_pass = True
    if args.min_label_coverage > 0.0:
        coverage_gate_pass = (
            coverage_ratio is not None and coverage_ratio >= args.min_label_coverage
        )

    go = (
        primary_auprc is not None
        and p_emp is not None
        and best_baseline_auprc is not None
        and primary_auprc >= GO_MIN_PRIMARY_AUPRC
        and primary_delta is not None
        and primary_delta >= GO_MIN_DELTA_AUPRC
        and p_emp <= GO_MAX_P_EMP
        and coverage_gate_pass
    )
    verdict = "GO" if go else "NO-GO"

    ndjson_sha = sha256_file(ndjson_path)
    labels_sha = sha256_file(labels_path)
    model_id, model_rev = get_model_meta(ndjson_path)

    header = (
        f"{'Score':30} {'N':>6} {'Pos':>6} {'Neg':>6} "
        f"{'AUROC':>14} {'AUPRC':>14} {'DeltaAUPRC':>14}"
    )
    print(header)
    print("-" * len(header))
    for row in metrics:
        print(
            f"{row['name'][:30]:30} "
            f"{row['n']:6d} {row['pos']:6d} {row['neg']:6d} "
            f"{fmt_float(row['auroc']):>14} {fmt_float(row['auprc']):>14} "
            f"{fmt_float(row['delta_auprc_vs_best_baseline']):>14}"
        )
    print("")
    print(f"Primary Endpoint: {primary_name} (key={args.primary_score})")
    print(f"Primary endpoint AUPRC: {fmt_float(primary_auprc)}")
    print(f"Delta AUPRC (Primary vs Best Baseline): {fmt_float(primary_delta)}")
    print(f"Primary permutation p_emp:  {fmt_float(p_emp)}")
    if args.min_label_coverage > 0.0:
        print(f"Label coverage ratio:      {fmt_float(coverage_ratio)}")
        print(f"Label coverage gate pass:  {coverage_gate_pass}")
    print(f"Verdict: {verdict}")

    lines: List[str] = []
    lines.append(f"date={dt.date.today().isoformat()}")
    lines.append(f"ndjson_path={to_repo_relative(ndjson_path)}")
    lines.append(f"labels_jsonl_path={to_repo_relative(labels_path)}")
    lines.append(
        f"labels_meta_json_path={to_repo_relative(labels_meta_path) if labels_meta_path else 'NA'}"
    )
    lines.append(f"ndjson_sha256={ndjson_sha}")
    lines.append(f"labels_sha256={labels_sha}")
    lines.append(f"model_id={model_id}")
    lines.append(f"model_revision={model_rev}")
    lines.append(f"seed={args.seed}")
    lines.append(f"perm_R={args.perm_R}")
    lines.append(f"transition_label_mode={TRANSITION_LABEL_MODE}")
    lines.append(f"primary_score_key={args.primary_score}")
    lines.append(f"primary_score_name={primary_name}")
    lines.append(f"min_label_coverage={args.min_label_coverage}")
    lines.append(f"label_alignment_coverage_ratio={fmt_float(coverage_ratio)}")
    lines.append(f"label_coverage_gate_pass={coverage_gate_pass}")
    lines.append("")
    lines.append("thresholds:")
    lines.append(f"  GO_MAX_P_EMP={GO_MAX_P_EMP}")
    lines.append(f"  GO_MIN_DELTA_AUPRC={GO_MIN_DELTA_AUPRC}")
    lines.append(f"  GO_MIN_PRIMARY_AUPRC={GO_MIN_PRIMARY_AUPRC}")
    lines.append("")
    lines.append("label_input_stats:")
    for key in sorted(label_stats.keys()):
        lines.append(f"  {key}={label_stats[key]}")
    lines.append("")
    lines.append(f"best_baseline={best_baseline_name}")
    lines.append(f"best_baseline_auprc={fmt_float(best_baseline_auprc)}")
    lines.append(f"delta_auprc_primary_vs_best_baseline={fmt_float(primary_delta)}")
    lines.append("")
    lines.append("per_score_metrics:")
    lines.append("score,n,pos,neg,auroc,auprc,delta_auprc_vs_best_baseline")
    for row in metrics:
        lines.append(
            f"{row['name']},{row['n']},{row['pos']},{row['neg']},"
            f"{fmt_float(row['auroc'])},{fmt_float(row['auprc'])},"
            f"{fmt_float(row['delta_auprc_vs_best_baseline'])}"
        )
    lines.append("")
    lines.append("primary_endpoint:")
    lines.append(f"  name={primary_name}")
    lines.append(f"  key={args.primary_score}")
    lines.append(f"  auprc={fmt_float(primary_auprc)}")
    lines.append(f"  delta_vs_best_baseline={fmt_float(primary_delta)}")
    lines.append(f"  perm_p_empirical={fmt_float(p_emp)}")
    lines.append(f"  perm_min={fmt_float(primary_perm['perm_min'])}")
    lines.append(f"  perm_p5={fmt_float(primary_perm['perm_p5'])}")
    lines.append(f"  perm_p50={fmt_float(primary_perm['perm_p50'])}")
    lines.append(f"  perm_p95={fmt_float(primary_perm['perm_p95'])}")
    lines.append("")
    lines.append(f"verdict={verdict}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"report={to_repo_relative(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
