#!/usr/bin/env python3
"""Evaluate Gate6 native object consumer metrics on Seam clean/perturbed pairs."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from aggregate_gate5_spike import (
    mean,
    parse_float,
    percentile_nearest_rank,
    read_csv,
    read_jsonl,
    robust_normalize,
    write_csv,
)
import build_gate6_native_local_span as gate6_builder


REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "gate6_native_object_seam_pairs_v1"
METHOD_ID = "gate6_native_object_seam_pairs_v1"
PRIMARY_METRIC = "edge_plane_loop_projective_chordal_v1"
GUARDRAIL_METRIC = "score_F_gram_loop_v1"
DEFAULT_TOPK = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Gate6 native-object token telemetry on seam clean/perturbed pairs "
            "using Gate5-style quietness summaries."
        )
    )
    parser.add_argument("--token-csv", required=True)
    parser.add_argument("--seam-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    return parser.parse_args()


def repo_relative_or_posix(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_text(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8", newline="\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_metric_values(rows: Sequence[Dict[str, str]], metric_key: str) -> List[float]:
    out: List[float] = []
    for row in rows:
        if str(row.get("loop_outcome") or "") != "none":
            continue
        value = parse_float(row.get(metric_key))
        if value is not None:
            out.append(float(value))
    return out


def sample_metric_stats(rows: Sequence[Dict[str, str]], metric_key: str) -> Dict[str, Optional[float]]:
    values = valid_metric_values(rows, metric_key)
    return {
        "max": max(values) if values else None,
        "mean": mean(values),
        "p90": percentile_nearest_rank(values, 0.90),
        "iqr": None
        if not values
        else (
            percentile_nearest_rank(values, 0.75) - percentile_nearest_rank(values, 0.25)
            if percentile_nearest_rank(values, 0.75) is not None
            and percentile_nearest_rank(values, 0.25) is not None
            else None
        ),
    }


def grouped_token_rows(rows: Sequence[Dict[str, str]]) -> Dict[int, List[Dict[str, str]]]:
    grouped: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["sample_id"])].append(row)
    for sample_rows in grouped.values():
        sample_rows.sort(key=lambda row: int(row["step"]))
    return grouped


def metric_topk_rows(
    rows: Sequence[Dict[str, str]],
    metric_key: str,
    topk: int,
) -> List[Dict[str, str]]:
    filtered = [
        row
        for row in rows
        if str(row.get("loop_outcome") or "") == "none" and parse_float(row.get(metric_key)) is not None
    ]
    filtered.sort(key=lambda row: (-float(parse_float(row.get(metric_key))), int(row["step"])))
    return filtered[:topk]


def delta(left: Optional[float], right: Optional[float]) -> Optional[float]:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def build_pair_rows(
    token_rows: Sequence[Dict[str, str]],
    seam_rows: Sequence[Dict[str, Any]],
    topk: int,
) -> List[Dict[str, Any]]:
    grouped = grouped_token_rows(token_rows)
    pairs: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in seam_rows:
        pairs[int(row["pair_id"])][str(row["challenge_class"])] = row

    pair_rows: List[Dict[str, Any]] = []
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

        clean_guardrail = sample_metric_stats(clean_rows, GUARDRAIL_METRIC)
        pert_guardrail = sample_metric_stats(perturbed_rows, GUARDRAIL_METRIC)
        clean_primary = sample_metric_stats(clean_rows, PRIMARY_METRIC)
        pert_primary = sample_metric_stats(perturbed_rows, PRIMARY_METRIC)
        clean_guardrail_p90 = clean_guardrail["p90"]
        clean_primary_p90 = clean_primary["p90"]

        guardrail_top = metric_topk_rows(perturbed_rows, GUARDRAIL_METRIC, topk)
        primary_top = metric_topk_rows(perturbed_rows, PRIMARY_METRIC, topk)

        pair_rows.append(
            {
                "pair_id": pair_id,
                "family": str(perturbed["perturbation_family"]),
                "clean_sample_id": int(clean["sample_id"]),
                "perturbed_sample_id": int(perturbed["sample_id"]),
                "delta_max_f_gram": delta(pert_guardrail["max"], clean_guardrail["max"]),
                "delta_max_edge_plane": delta(pert_primary["max"], clean_primary["max"]),
                "delta_p90_f_gram": delta(pert_guardrail["p90"], clean_guardrail["p90"]),
                "delta_p90_edge_plane": delta(pert_primary["p90"], clean_primary["p90"]),
                "delta_mean_f_gram": delta(pert_guardrail["mean"], clean_guardrail["mean"]),
                "delta_mean_edge_plane": delta(pert_primary["mean"], clean_primary["mean"]),
                "iqr_normalized_delta_max_f_gram": robust_normalize(
                    delta(pert_guardrail["max"], clean_guardrail["max"]),
                    clean_guardrail["iqr"],
                ),
                "iqr_normalized_delta_max_edge_plane": robust_normalize(
                    delta(pert_primary["max"], clean_primary["max"]),
                    clean_primary["iqr"],
                ),
                "topk_inflation_f_gram": None
                if clean_guardrail_p90 is None
                else sum(
                    1
                    for row in guardrail_top
                    if float(parse_float(row.get(GUARDRAIL_METRIC))) >= float(clean_guardrail_p90)
                ),
                "topk_inflation_edge_plane": None
                if clean_primary_p90 is None
                else sum(
                    1
                    for row in primary_top
                    if float(parse_float(row.get(PRIMARY_METRIC))) >= float(clean_primary_p90)
                ),
            }
        )
    if not pair_rows:
        raise RuntimeError(
            "seam pair evaluation found zero complete linked pairs; "
            "check token telemetry sample ids and seam pair linkage"
        )
    return pair_rows


def summarize_families(pair_rows: Sequence[Dict[str, Any]], topk: int) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        grouped[str(row["family"])].append(row)

    out: List[Dict[str, Any]] = []
    for family in sorted(grouped):
        family_rows = grouped[family]
        out.append(
            {
                "family": family,
                "n_pairs": len(family_rows),
                "mean_delta_max_f_gram": mean(
                    row["delta_max_f_gram"] for row in family_rows if row["delta_max_f_gram"] is not None
                ),
                "mean_delta_max_edge_plane": mean(
                    row["delta_max_edge_plane"]
                    for row in family_rows
                    if row["delta_max_edge_plane"] is not None
                ),
                "mean_delta_p90_f_gram": mean(
                    row["delta_p90_f_gram"] for row in family_rows if row["delta_p90_f_gram"] is not None
                ),
                "mean_delta_p90_edge_plane": mean(
                    row["delta_p90_edge_plane"]
                    for row in family_rows
                    if row["delta_p90_edge_plane"] is not None
                ),
                "mean_delta_mean_f_gram": mean(
                    row["delta_mean_f_gram"] for row in family_rows if row["delta_mean_f_gram"] is not None
                ),
                "mean_delta_mean_edge_plane": mean(
                    row["delta_mean_edge_plane"]
                    for row in family_rows
                    if row["delta_mean_edge_plane"] is not None
                ),
                "mean_iqr_normalized_delta_max_f_gram": mean(
                    row["iqr_normalized_delta_max_f_gram"]
                    for row in family_rows
                    if row["iqr_normalized_delta_max_f_gram"] is not None
                ),
                "mean_iqr_normalized_delta_max_edge_plane": mean(
                    row["iqr_normalized_delta_max_edge_plane"]
                    for row in family_rows
                    if row["iqr_normalized_delta_max_edge_plane"] is not None
                ),
                f"mean_top{topk}_inflation_f_gram": mean(
                    row["topk_inflation_f_gram"] for row in family_rows if row["topk_inflation_f_gram"] is not None
                ),
                f"mean_top{topk}_inflation_edge_plane": mean(
                    row["topk_inflation_edge_plane"]
                    for row in family_rows
                    if row["topk_inflation_edge_plane"] is not None
                ),
                "edge_plane_better_delta_max_count": sum(
                    1
                    for row in family_rows
                    if row["delta_max_f_gram"] is not None
                    and row["delta_max_edge_plane"] is not None
                    and float(row["delta_max_edge_plane"]) < float(row["delta_max_f_gram"])
                ),
                "edge_plane_better_delta_p90_count": sum(
                    1
                    for row in family_rows
                    if row["delta_p90_f_gram"] is not None
                    and row["delta_p90_edge_plane"] is not None
                    and float(row["delta_p90_edge_plane"]) < float(row["delta_p90_f_gram"])
                ),
            }
        )
    return out


def render_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


def build_report(
    run_id: str,
    pair_rows: Sequence[Dict[str, Any]],
    topk: int,
) -> str:
    representative: List[Tuple[float, int, str]] = []
    for row in pair_rows:
        if row["delta_max_f_gram"] is None or row["delta_max_edge_plane"] is None:
            continue
        representative.append(
            (
                float(row["delta_max_f_gram"]) - float(row["delta_max_edge_plane"]),
                int(row["pair_id"]),
                str(row["family"]),
            )
        )
    representative.sort(reverse=True)

    lines = [
        "# Gate6 Native Object Seam Pair Report",
        "",
        f"run_id: {run_id}",
        f"method_id: {METHOD_ID}",
        "",
        "## Paired Quietness Summary",
        "",
        f"- mean_delta_max_{GUARDRAIL_METRIC}: {render_float(mean(row['delta_max_f_gram'] for row in pair_rows if row['delta_max_f_gram'] is not None))}",
        f"- mean_delta_max_{PRIMARY_METRIC}: {render_float(mean(row['delta_max_edge_plane'] for row in pair_rows if row['delta_max_edge_plane'] is not None))}",
        f"- mean_delta_p90_{GUARDRAIL_METRIC}: {render_float(mean(row['delta_p90_f_gram'] for row in pair_rows if row['delta_p90_f_gram'] is not None))}",
        f"- mean_delta_p90_{PRIMARY_METRIC}: {render_float(mean(row['delta_p90_edge_plane'] for row in pair_rows if row['delta_p90_edge_plane'] is not None))}",
        f"- mean_iqr_normalized_delta_max_{GUARDRAIL_METRIC}: {render_float(mean(row['iqr_normalized_delta_max_f_gram'] for row in pair_rows if row['iqr_normalized_delta_max_f_gram'] is not None))}",
        f"- mean_iqr_normalized_delta_max_{PRIMARY_METRIC}: {render_float(mean(row['iqr_normalized_delta_max_edge_plane'] for row in pair_rows if row['iqr_normalized_delta_max_edge_plane'] is not None))}",
        "",
        "## Spike Inflation",
        "",
        f"- mean_top{topk}_inflation_{GUARDRAIL_METRIC}_vs_clean_p90: {render_float(mean(row['topk_inflation_f_gram'] for row in pair_rows if row['topk_inflation_f_gram'] is not None))}",
        f"- mean_top{topk}_inflation_{PRIMARY_METRIC}_vs_clean_p90: {render_float(mean(row['topk_inflation_edge_plane'] for row in pair_rows if row['topk_inflation_edge_plane'] is not None))}",
        "",
        "## Representative Pairs",
        "",
    ]
    if not representative:
        lines.append("- No complete comparable pairs were available.")
    else:
        for delta_gap, pair_id, family in representative[:5]:
            lines.append(
                f"- pair_id={pair_id} family={family} delta_max_{GUARDRAIL_METRIC}_minus_{PRIMARY_METRIC}={render_float(delta_gap)}"
            )
    lines.extend(
        [
            "",
            "## Conclusion Envelope",
            "",
            "- Seam is evaluated as quietness under clean-vs-perturbed paired comparison.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_manifest(
    run_id: str,
    token_csv_path: Path,
    seam_jsonl_path: Path,
    pair_rows: Sequence[Dict[str, Any]],
    topk: int,
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "primary_metric_id": PRIMARY_METRIC,
        "guardrail_metric_id": GUARDRAIL_METRIC,
        "token_csv_path": repo_relative_or_posix(token_csv_path),
        "token_csv_sha256": sha256_file(token_csv_path),
        "seam_jsonl_path": repo_relative_or_posix(seam_jsonl_path),
        "seam_jsonl_sha256": sha256_file(seam_jsonl_path),
        "code_git_commit": gate6_builder.current_git_commit(),
        "n_pairs_total": len(pair_rows),
        "topk": int(topk),
    }


def write_checksums(path: Path, artifact_paths: Sequence[Tuple[str, Path]]) -> None:
    payload = {
        name: {
            "path": repo_relative_or_posix(artifact_path),
            "sha256": sha256_file(artifact_path),
        }
        for name, artifact_path in artifact_paths
    }
    write_json(path, payload)


def main() -> int:
    args = parse_args()
    token_csv_path = (REPO_ROOT / args.token_csv).resolve()
    seam_jsonl_path = (REPO_ROOT / args.seam_jsonl).resolve()
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    run_id = args.run_id or out_dir.name

    token_rows = read_csv(token_csv_path)
    seam_rows = read_jsonl(seam_jsonl_path)
    pair_rows = build_pair_rows(token_rows, seam_rows, topk=args.topk)
    family_rows = summarize_families(pair_rows, topk=args.topk)
    report = build_report(run_id, pair_rows, topk=args.topk)
    manifest = build_manifest(run_id, token_csv_path, seam_jsonl_path, pair_rows, topk=args.topk)

    manifest_path = out_dir / "manifest.json"
    pair_summary_path = out_dir / "gate6b_seam_pair_summary.csv"
    family_summary_path = out_dir / "gate6b_seam_family_summary.csv"
    report_path = out_dir / "gate6b_seam_report.md"
    checksums_path = out_dir / "checksums.json"

    write_json(manifest_path, manifest)
    write_csv(
        pair_summary_path,
        fieldnames=[
            "pair_id",
            "family",
            "clean_sample_id",
            "perturbed_sample_id",
            "delta_max_f_gram",
            "delta_max_edge_plane",
            "delta_p90_f_gram",
            "delta_p90_edge_plane",
            "delta_mean_f_gram",
            "delta_mean_edge_plane",
            "iqr_normalized_delta_max_f_gram",
            "iqr_normalized_delta_max_edge_plane",
            "topk_inflation_f_gram",
            "topk_inflation_edge_plane",
        ],
        rows=pair_rows,
    )
    write_csv(
        family_summary_path,
        fieldnames=[
            "family",
            "n_pairs",
            "mean_delta_max_f_gram",
            "mean_delta_max_edge_plane",
            "mean_delta_p90_f_gram",
            "mean_delta_p90_edge_plane",
            "mean_delta_mean_f_gram",
            "mean_delta_mean_edge_plane",
            "mean_iqr_normalized_delta_max_f_gram",
            "mean_iqr_normalized_delta_max_edge_plane",
            f"mean_top{args.topk}_inflation_f_gram",
            f"mean_top{args.topk}_inflation_edge_plane",
            "edge_plane_better_delta_max_count",
            "edge_plane_better_delta_p90_count",
        ],
        rows=family_rows,
    )
    write_text(report_path, report)
    write_checksums(
        checksums_path,
        (
            ("manifest_json", manifest_path),
            ("pair_summary_csv", pair_summary_path),
            ("family_summary_csv", family_summary_path),
            ("report_md", report_path),
        ),
    )

    print(f"manifest_json={repo_relative_or_posix(manifest_path)}")
    print(f"pair_summary_csv={repo_relative_or_posix(pair_summary_path)}")
    print(f"family_summary_csv={repo_relative_or_posix(family_summary_path)}")
    print(f"report_md={repo_relative_or_posix(report_path)}")
    print(f"checksums_json={repo_relative_or_posix(checksums_path)}")
    print(f"n_pairs_total={len(pair_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
