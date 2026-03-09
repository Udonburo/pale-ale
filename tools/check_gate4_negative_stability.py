#!/usr/bin/env python3
"""Negative-stability diagnostics using Gate4 artifacts only."""

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


REPORT_NAME = "gate4_negative_stability_report.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute negative-side stability diagnostics from canonical Gate4 artifacts "
            "without consulting raw triplets or CFA source rows."
        )
    )
    parser.add_argument(
        "--gate4-out-dir",
        default="runs/gate4_artifact_sufficiency/gate4_out",
        help="Gate4 output directory containing manifest.json and CSV artifacts.",
    )
    parser.add_argument(
        "--out",
        default=(
            f"attestations/triality/gate4_validation/"
            f"{dt.date.today().isoformat()}_{REPORT_NAME}"
        ),
    )
    parser.add_argument("--top-samples", type=int, default=10)
    parser.add_argument("--top-transitions", type=int, default=5)
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


def load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected object JSON at {path}")
    return data


def load_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_float(value: str) -> Optional[float]:
    raw = value.strip()
    if raw == "":
        return None
    parsed = float(raw)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite float {value!r}")
    return parsed


def parse_int(value: str) -> int:
    return int(value.strip())


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


def verdict(cons_below_ratio: float) -> str:
    if cons_below_ratio >= 0.90:
        return "Green"
    if cons_below_ratio >= 0.80:
        return "Yellow"
    return "Red"


def group_token_rows(rows: Sequence[Dict[str, str]]) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        sample_id = parse_int(row["sample_id"])
        grouped.setdefault(sample_id, []).append(
            {
                "sample_id": sample_id,
                "variant": row["variant"],
                "world_type": row["world_type"],
                "step": parse_int(row["step"]),
                "absolute_pos": parse_int(row["absolute_pos"]),
                "token_text": row["token_text"],
                "answer_char_start": parse_int(row["answer_char_start"]),
                "answer_char_end": parse_int(row["answer_char_end"]),
                "label_token": parse_int(row["label_token"]),
                "label_transition": parse_int(row["label_transition"]),
                "transition_missing_reason": row["transition_missing_reason"],
                "score_A": parse_float(row["score_A_logprob"]),
                "score_B": parse_float(row["score_B_entropy"]),
                "score_E": parse_float(row["score_E_v_sminus_vnext"]),
            }
        )
    for sample_rows in grouped.values():
        sample_rows.sort(key=lambda item: item["step"])
    return grouped


def load_sample_summary(path: Path) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for row in load_csv(path):
        sample_id = parse_int(row["sample_id"])
        out[sample_id] = {
            "sample_id": sample_id,
            "run_id": row["run_id"],
            "variant": row["variant"],
            "world_type": row["world_type"],
            "positive_token_count": parse_int(row["positive_token_count"]),
            "positive_transition_count": parse_int(row["positive_transition_count"]),
            "hit_at_10_E": parse_int(row["hit_at_10_E"]),
            "delta_auprc_E_vs_best_baseline": parse_float(
                row["delta_auprc_E_vs_best_baseline"]
            ),
            "best_baseline_name": row["best_baseline_name"],
        }
    return out


def build_sample_stats(
    grouped_rows: Dict[int, List[Dict[str, Any]]],
    sample_summary: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for sample_id in sorted(grouped_rows):
        rows = grouped_rows[sample_id]
        transition_rows = [row for row in rows if row["score_E"] is not None]
        if not transition_rows:
            raise ValueError(f"sample {sample_id} has no transition rows with score_E")
        score_e = [float(row["score_E"]) for row in transition_rows]
        score_a = [float(row["score_A"]) for row in transition_rows if row["score_A"] is not None]
        score_b = [float(row["score_B"]) for row in transition_rows if row["score_B"] is not None]
        if len(score_a) != len(score_e) or len(score_b) != len(score_e):
            raise ValueError(f"sample {sample_id} has missing baseline scores in transition rows")
        sorted_e = sorted(score_e)
        summary_row = sample_summary.get(sample_id)
        if summary_row is None:
            raise ValueError(f"sample {sample_id} missing from gate4_sample_summary.csv")
        samples.append(
            {
                "sample_id": sample_id,
                "run_id": summary_row["run_id"],
                "variant": summary_row["variant"],
                "world_type": summary_row["world_type"],
                "rows": rows,
                "transition_rows": transition_rows,
                "score_a_rows": score_a,
                "score_b_rows": score_b,
                "max_e": sorted_e[-1],
                "p90_e": quantile_nearest_rank(sorted_e, 0.90),
                "mean_e": statistics.fmean(sorted_e),
                "hit_at_10_E": summary_row["hit_at_10_E"],
                "delta_auprc_E_vs_best_baseline": summary_row[
                    "delta_auprc_E_vs_best_baseline"
                ],
                "best_baseline_name": summary_row["best_baseline_name"],
            }
        )
    return samples


def validate_manifest_artifacts(
    manifest: Dict[str, Any],
    token_features_path: Path,
    sample_summary_path: Path,
    run_summary_path: Path,
    token_rows: Sequence[Dict[str, str]],
    sample_summary: Dict[int, Dict[str, Any]],
    run_summary_rows: Sequence[Dict[str, str]],
) -> None:
    expected_hashes = {
        "token_features_sha256": sha256_file(token_features_path),
        "sample_summary_sha256": sha256_file(sample_summary_path),
        "run_summary_sha256": sha256_file(run_summary_path),
    }
    for key, actual in expected_hashes.items():
        manifest_value = str(manifest.get(key, ""))
        if manifest_value != actual:
            raise ValueError(
                f"manifest {key} mismatch: manifest={manifest_value} actual={actual}"
            )

    if len(run_summary_rows) != 1:
        raise ValueError("gate4_run_summary.csv must contain exactly one data row")
    run_summary = run_summary_rows[0]
    manifest_run_id = str(manifest["run_id"])
    if run_summary["run_id"] != manifest_run_id:
        raise ValueError(
            f"run summary run_id mismatch: {run_summary['run_id']} != {manifest_run_id}"
        )
    for row in token_rows:
        if row["run_id"] != manifest_run_id:
            raise ValueError(
                f"token row run_id mismatch for sample_id={row['sample_id']} step={row['step']}"
            )
    for sample_id, row in sample_summary.items():
        if row["run_id"] != manifest_run_id:
            raise ValueError(f"sample summary run_id mismatch for sample_id={sample_id}")

    n_samples_total = len(sample_summary)
    n_token_rows_total = len(token_rows)
    n_transition_rows_total = sum(
        1 for row in token_rows if row["transition_missing_reason"] == "none"
    )
    n_samples_with_positive_tokens = sum(
        1 for row in sample_summary.values() if row["positive_token_count"] > 0
    )
    n_samples_with_positive_transitions = sum(
        1 for row in sample_summary.values() if row["positive_transition_count"] > 0
    )
    expected_counts = {
        "n_samples_total": n_samples_total,
        "n_token_rows_total": n_token_rows_total,
        "n_transition_rows_total": n_transition_rows_total,
        "n_samples_with_positive_tokens": n_samples_with_positive_tokens,
        "n_samples_with_positive_transitions": n_samples_with_positive_transitions,
    }
    for key, actual in expected_counts.items():
        manifest_value = int(manifest[key])
        run_summary_value = int(run_summary[key])
        if manifest_value != actual:
            raise ValueError(
                f"manifest {key} mismatch: manifest={manifest_value} actual={actual}"
            )
        if run_summary_value != actual:
            raise ValueError(
                f"run summary {key} mismatch: run_summary={run_summary_value} actual={actual}"
            )


def inspect_top_transitions(sample: Dict[str, Any], n: int) -> List[Dict[str, Any]]:
    rows = sample["rows"]
    indexed = [idx for idx, row in enumerate(rows[:-1]) if row["score_E"] is not None]
    indexed.sort(key=lambda idx: (-float(rows[idx]["score_E"]), idx))
    out: List[Dict[str, Any]] = []
    for rank, idx in enumerate(indexed[:n], start=1):
        row = rows[idx]
        next_row = rows[idx + 1]
        out.append(
            {
                "rank": rank,
                "transition_step": int(row["step"]),
                "score_E": float(row["score_E"]),
                "score_A": float(row["score_A"]),
                "score_B": float(row["score_B"]),
                "token_t": row["token_text"],
                "token_t1": next_row["token_text"],
                "char_t": (int(row["answer_char_start"]), int(row["answer_char_end"])),
                "char_t1": (
                    int(next_row["answer_char_start"]),
                    int(next_row["answer_char_end"]),
                ),
            }
        )
    return out


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    gate4_out_dir = repo_root / args.gate4_out_dir
    manifest_path = gate4_out_dir / "manifest.json"
    token_features_path = gate4_out_dir / "gate4_token_features.csv"
    sample_summary_path = gate4_out_dir / "gate4_sample_summary.csv"
    run_summary_path = gate4_out_dir / "gate4_run_summary.csv"
    out_path = repo_root / args.out

    manifest = load_json(manifest_path)
    token_rows = load_csv(token_features_path)
    sample_summary = load_sample_summary(sample_summary_path)
    run_summary_rows = load_csv(run_summary_path)
    validate_manifest_artifacts(
        manifest=manifest,
        token_features_path=token_features_path,
        sample_summary_path=sample_summary_path,
        run_summary_path=run_summary_path,
        token_rows=token_rows,
        sample_summary=sample_summary,
        run_summary_rows=run_summary_rows,
    )

    grouped = group_token_rows(token_rows)
    samples = build_sample_stats(grouped, sample_summary)

    consistent = [sample for sample in samples if sample["variant"] == "consistent"]
    frustrated = [sample for sample in samples if sample["variant"] == "frustrated"]
    if not consistent or not frustrated:
        raise ValueError("artifact set must contain both consistent and frustrated samples")

    consistent_run_values = [
        float(row["score_E"])
        for sample in consistent
        for row in sample["transition_rows"]
    ]
    consistent_max = [float(sample["max_e"]) for sample in consistent]
    consistent_p90 = [float(sample["p90_e"]) for sample in consistent]
    consistent_mean = [float(sample["mean_e"]) for sample in consistent]
    frustrated_max = [float(sample["max_e"]) for sample in frustrated]
    median_frustrated_max_e = summarize(frustrated_max)["p50"]
    ratio_below = (
        sum(1 for value in consistent_max if value < median_frustrated_max_e)
        / float(len(consistent_max))
    )
    pooled_e = [
        float(row["score_E"])
        for sample in consistent
        for row in sample["transition_rows"]
    ]
    pooled_a = [
        float(value) for sample in consistent for value in sample["score_a_rows"]
    ]
    pooled_b = [
        float(value) for sample in consistent for value in sample["score_b_rows"]
    ]
    rho_e_a = spearman_rho(pooled_e, pooled_a)
    rho_e_b = spearman_rho(pooled_e, pooled_b)

    top_samples = sorted(
        consistent,
        key=lambda sample: (-float(sample["max_e"]), int(sample["sample_id"])),
    )[: args.top_samples]

    lines: List[str] = []
    lines.append(f"date={dt.date.today().isoformat()}")
    lines.append("experiment=gate4_artifact_negative_stability_consistent_scoreE")
    lines.append(f"gate4_out_dir={repo_rel(gate4_out_dir, repo_root)}")
    lines.append(f"manifest_json={repo_rel(manifest_path, repo_root)}")
    lines.append(f"token_features_csv={repo_rel(token_features_path, repo_root)}")
    lines.append(f"sample_summary_csv={repo_rel(sample_summary_path, repo_root)}")
    lines.append(f"run_summary_csv={repo_rel(run_summary_path, repo_root)}")
    lines.append(f"manifest_sha256={sha256_file(manifest_path)}")
    lines.append(f"token_features_sha256={sha256_file(token_features_path)}")
    lines.append(f"sample_summary_sha256={sha256_file(sample_summary_path)}")
    lines.append(f"run_summary_sha256={sha256_file(run_summary_path)}")
    lines.append(f"script_sha256={sha256_file(Path(__file__))}")
    lines.append("artifact_integrity=PASS")
    lines.append(f"run_id={manifest['run_id']}")
    lines.append(f"dataset_revision_id={manifest['dataset_revision_id']}")
    lines.append(f"dataset_hash_blake3={manifest['dataset_hash_blake3']}")
    lines.append(f"spec_hash_raw_blake3={manifest['spec_hash_raw_blake3']}")
    lines.append(f"spec_hash_blake3={manifest['spec_hash_blake3']}")
    lines.append(f"model_id={manifest['model_id']}")
    lines.append(f"model_revision={manifest['model_revision']}")
    lines.append(f"seed={manifest['seed']}")
    lines.append(f"primary_score={manifest['primary_score']}")
    lines.append("")
    lines.append("population:")
    lines.append(f"  consistent_samples={len(consistent)}")
    lines.append(f"  frustrated_samples={len(frustrated)}")
    lines.append(f"  consistent_transitions={len(consistent_run_values)}")
    lines.append("")
    lines.append("score_E_run_level_distribution_consistent:")
    for key, value in summarize(consistent_run_values).items():
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
    lines.append("  scope=consistent_only_transition_rows")
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
            f"max_E={fmt_float(sample['max_e'])} "
            f"p90_E={fmt_float(sample['p90_e'])} "
            f"mean_E={fmt_float(sample['mean_e'])} "
            f"hit_at_10_E={sample['hit_at_10_E']} "
            f"delta_auprc_E_vs_best_baseline={fmt_float(sample['delta_auprc_E_vs_best_baseline'])} "
            f"best_baseline_name={sample['best_baseline_name']}"
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
