#!/usr/bin/env python3
"""Run representative CFA case-study set (Top/Median/Bottom frustrated deltas)."""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import eval_local_span as localeval
import extract_triality_triplets as extractor
import plot_cfa_case_pair as caseplot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select representative frustrated samples from CFA batch results, run pair "
            "case-study visualizations, and write summary/index manifests."
        )
    )
    parser.add_argument("--results-jsonl", default="runs/cfa_batch_primaryE/results.jsonl")
    parser.add_argument("--cfa-jsonl", default="data/cfa/cfa_v1.jsonl")
    parser.add_argument(
        "--batch-report",
        default=f"attestations/triality/{dt.date.today().isoformat()}_cfa_batch_primaryE_report.txt",
    )
    parser.add_argument("--group-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--perm-R", type=int, default=0)
    parser.add_argument("--min-coverage", type=float, default=0.30)
    parser.add_argument("--out-root", default="runs/cfa_case_study")
    parser.add_argument("--attest-root", default="attestations/triality/case_study")
    parser.add_argument("--percentile", type=float, default=0.90)
    args = parser.parse_args()
    if args.group_size <= 0:
        parser.error("--group-size must be > 0")
    if args.topk <= 0:
        parser.error("--topk must be > 0")
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


def fmt(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.17e}"


def load_results(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"non-object row at {path}:{line_no}")
            rows.append(row)
    return rows


def select_representative(
    frustrated_rows: Sequence[Dict[str, Any]], group_size: int
) -> Dict[str, List[Dict[str, Any]]]:
    ordered = sorted(
        frustrated_rows,
        key=lambda r: (float(r["delta"]), int(r["sample_id"])),
    )
    if len(ordered) < group_size * 3:
        raise RuntimeError(
            f"Not enough frustrated rows for representative selection: {len(ordered)}"
        )

    bottom = ordered[:group_size]
    top = list(reversed(ordered[-group_size:]))

    n = len(ordered)
    mid = n // 2
    left = max(0, mid - (group_size // 2))
    right = left + group_size
    if right > n:
        right = n
        left = right - group_size
    median_band = ordered[left:right]

    return {"top": top, "median": median_band, "bottom": bottom}


def collect_script_hashes(repo_root: Path) -> Dict[str, str]:
    paths = [
        repo_root / "tools" / "run_cfa_case_representative.py",
        repo_root / "tools" / "plot_cfa_case_pair.py",
        repo_root / "tools" / "eval_local_span.py",
        repo_root / "tools" / "extract_triality_triplets.py",
        repo_root / "tools" / "labels_from_cfa_spans.py",
        repo_root / "tools" / "eval_triality_token.py",
    ]
    out: Dict[str, str] = {}
    for path in paths:
        if path.exists():
            out[path.as_posix()] = sha256_file(path)
    return out


def run_one_pair(
    sample_id: int,
    contrast_id: int,
    cfa_rows: Dict[int, Dict[str, Any]],
    model: Any,
    tokenizer: Any,
    device: Any,
    topk: int,
    min_coverage: float,
    out_root: Path,
    attest_root: Path,
) -> Dict[str, Any]:
    sample_row = cfa_rows[sample_id]
    contrast_row = cfa_rows[contrast_id]
    pair_id = f"sample{sample_id}_vs_{contrast_id}"
    run_dir = out_root / pair_id
    run_dir.mkdir(parents=True, exist_ok=True)
    attest_root.mkdir(parents=True, exist_ok=True)

    sample_pack = caseplot.extract_and_label_sample(
        sample_row=sample_row,
        sample_id=sample_id,
        model=model,
        tokenizer=tokenizer,
        device=device,
        topk=topk,
        min_coverage=min_coverage,
        out_dir=run_dir,
    )
    contrast_pack = caseplot.extract_and_label_sample(
        sample_row=contrast_row,
        sample_id=contrast_id,
        model=model,
        tokenizer=tokenizer,
        device=device,
        topk=topk,
        min_coverage=min_coverage,
        out_dir=run_dir,
    )

    token_table_sample = run_dir / f"token_table_{sample_id}.csv"
    token_table_contrast = run_dir / f"token_table_{contrast_id}.csv"
    overlay_csv = run_dir / f"pair_overlay_{sample_id}_vs_{contrast_id}.csv"
    plot_case = run_dir / f"plot_case_{sample_id}.png"
    plot_pair = run_dir / f"plot_pair_compare_{sample_id}_{contrast_id}.png"
    summary_md = attest_root / f"case_summary_{sample_id}_{contrast_id}.md"

    caseplot.write_token_table_csv(
        token_table_sample,
        sample_pack["triplets"],
        sample_pack["labels"],
        sample_pack["scores"],
    )
    caseplot.write_token_table_csv(
        token_table_contrast,
        contrast_pack["triplets"],
        contrast_pack["labels"],
        contrast_pack["scores"],
    )
    caseplot.write_pair_overlay_csv(
        overlay_csv,
        sample_pack["triplets"],
        contrast_pack["triplets"],
        sample_pack["labels"],
        contrast_pack["labels"],
        sample_pack["scores"],
        contrast_pack["scores"],
    )
    caseplot.plot_single_case(
        plot_case,
        sample_id,
        sample_row,
        sample_pack["triplets"],
        sample_pack["labels"],
        sample_pack["scores"],
        sample_pack["exact_token_match_ratio"],
        sample_pack["coverage"],
    )
    caseplot.plot_pair_compare(
        plot_pair,
        sample_id,
        contrast_id,
        sample_pack["triplets"],
        contrast_pack["triplets"],
        sample_pack["labels"],
        sample_pack["scores"],
        contrast_pack["scores"],
    )

    top10 = caseplot.top_k_e_tokens(
        sample_pack["triplets"],
        sample_pack["scores"]["labels_trans"],
        sample_pack["scores"]["score_e_trans"],
        k=10,
    )
    overlap = sum(1 for row in top10 if int(row["transition_label"]) == 1)
    caseplot.write_case_summary_md(
        summary_md,
        sample_id,
        contrast_id,
        sample_row,
        contrast_row,
        sample_pack["scores"],
        contrast_pack["scores"],
        top10,
        overlap_count=overlap,
        overlap_k=10,
        token_table_fr=token_table_sample,
        token_table_co=token_table_contrast,
        overlay_csv=overlay_csv,
        plot_fr=plot_case,
        plot_pair=plot_pair,
    )

    local_loaded = localeval.load_table(
        token_table_sample, score_col="score_E", label_col="in_defect_span"
    )
    local_metrics = localeval.compute_metrics(
        local_loaded["score_values"],
        local_loaded["labels"],
        topk=10,
        percentile=0.90,
    )

    return {
        "sample_id": sample_id,
        "contrast_sample_id": contrast_id,
        "world_type": sample_row.get("world_type"),
        "delta": float(sample_pack["scores"]["delta_e_vs_best"] or 0.0),
        "auprc_A": sample_pack["scores"]["auprc_a"],
        "auprc_B": sample_pack["scores"]["auprc_b"],
        "auprc_E": sample_pack["scores"]["auprc_e"],
        "best_baseline": sample_pack["scores"]["best_baseline_name"],
        "exact_token_match_ratio": sample_pack["exact_token_match_ratio"],
        "coverage": sample_pack["coverage"],
        "hit_at_10": local_metrics["hit_at_k_count"],
        "hit_rate_10": local_metrics["hit_at_k_rate"],
        "first_hit_distance": local_metrics["first_hit_distance_signed"],
        "first_hit_after_defect_distance": local_metrics["first_hit_after_defect_distance"],
        "defect_start_step": local_metrics["defect_start_step"],
        "first_hit_step": local_metrics["first_hit_step"],
        "pair_id": pair_id,
        "token_table": token_table_sample.as_posix(),
        "plot_case": plot_case.as_posix(),
        "plot_pair": plot_pair.as_posix(),
        "summary_md": summary_md.as_posix(),
    }


def write_representative_summary(
    out_path: Path, grouped: Dict[str, List[Dict[str, Any]]]
) -> None:
    lines: List[str] = []
    lines.append("# Representative Set Summary (CFA Primary=E)")
    lines.append("")
    lines.append(
        "| bucket | sample_id | contrast_id | world | delta_AUPRC | AUPRC_E | best_baseline | hit@10 | hit_rate@10 | first_hit_distance | first_hit_after_defect | defect_start | first_hit_step | coverage | exact_match | outputs |"
    )
    lines.append(
        "|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    )
    for bucket in ("top", "median", "bottom"):
        for row in grouped[bucket]:
            outputs = f"[pair]({row['plot_pair']}) / [case]({row['plot_case']}) / [summary]({row['summary_md']})"
            lines.append(
                f"| {bucket} | {row['sample_id']} | {row['contrast_sample_id']} | {row['world_type']} | "
                f"{fmt(row['delta'])} | {fmt(row['auprc_E'])} | {row['best_baseline']} | "
                f"{row['hit_at_10']} | {fmt(row['hit_rate_10'])} | "
                f"{row['first_hit_distance']} | {row['first_hit_after_defect_distance']} | "
                f"{row['defect_start_step']} | {row['first_hit_step']} | "
                f"{fmt(row['coverage'])} | {fmt(row['exact_token_match_ratio'])} | {outputs} |"
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def write_index(
    out_path: Path,
    args: argparse.Namespace,
    dataset_sha: str,
    script_hashes: Dict[str, str],
    selected: Dict[str, List[Dict[str, Any]]],
) -> None:
    lines: List[str] = []
    lines.append("# CFA Case Study Index")
    lines.append("")
    lines.append(f"- date: `{dt.date.today().isoformat()}`")
    lines.append(f"- dataset: `{Path(args.cfa_jsonl).as_posix()}`")
    lines.append(f"- dataset_sha256: `{dataset_sha}`")
    batch_report_path = Path(args.batch_report).as_posix()
    lines.append(
        f"- batch_report: [{Path(args.batch_report).name}]({batch_report_path})"
    )
    lines.append("")
    lines.append("## Commands")
    lines.append("```bash")
    lines.append(
        f"python tools/run_cfa_case_representative.py --results-jsonl {Path(args.results_jsonl).as_posix()} "
        f"--cfa-jsonl {Path(args.cfa_jsonl).as_posix()} --batch-report {Path(args.batch_report).as_posix()} "
        f"--group-size {args.group_size} --seed {args.seed} --model-id {args.model_id} --device {args.device} "
        f"--topk {args.topk} --perm-R {args.perm_R} --min-coverage {args.min_coverage}"
    )
    lines.append("```")
    lines.append("")
    lines.append("## Selected Samples")
    for bucket in ("top", "median", "bottom"):
        ids = [int(r["sample_id"]) for r in selected[bucket]]
        lines.append(f"- {bucket}: `{','.join(str(x) for x in ids)}`")
    lines.append("")
    lines.append("## Script SHA256")
    for path in sorted(script_hashes):
        lines.append(f"- `{path}`: `{script_hashes[path]}`")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(str(repo_root))

    results_path = Path(args.results_jsonl)
    cfa_path = Path(args.cfa_jsonl)
    batch_report = Path(args.batch_report)
    out_root = Path(args.out_root)
    attest_root = Path(args.attest_root)
    index_path = attest_root / "index.md"
    summary_path = attest_root / "representative_set_summary.md"

    results = load_results(results_path)
    frustrated = [
        r
        for r in results
        if r.get("status") == "ok"
        and r.get("variant") == "frustrated"
        and isinstance(r.get("delta"), (int, float))
    ]
    selected = select_representative(frustrated, group_size=args.group_size)

    cfa_rows = caseplot.load_cfa_rows(cfa_path)

    extractor.configure_reproducibility(args.seed, deterministic=True)
    device = extractor.resolve_device(args.device)
    candidates = extractor.build_model_candidates(args.model_id)
    model_id, tokenizer, model, model_revision = extractor.load_first_available_model(
        model_candidates=candidates,
        device=device,
    )
    print(f"model_id={model_id}")
    print(f"model_revision={model_revision}")

    grouped_outputs: Dict[str, List[Dict[str, Any]]] = {
        "top": [],
        "median": [],
        "bottom": [],
    }
    for bucket in ("top", "median", "bottom"):
        for row in selected[bucket]:
            sid = int(row["sample_id"])
            if sid not in cfa_rows:
                raise ValueError(f"sample_id={sid} missing in CFA dataset")
            contrast = cfa_rows[sid].get("contrast_sample_id")
            if contrast is None:
                raise ValueError(f"sample_id={sid} missing contrast_sample_id")
            cid = int(contrast)
            print(f"[{bucket}] sample_id={sid} contrast={cid}")
            out = run_one_pair(
                sample_id=sid,
                contrast_id=cid,
                cfa_rows=cfa_rows,
                model=model,
                tokenizer=tokenizer,
                device=device,
                topk=args.topk,
                min_coverage=args.min_coverage,
                out_root=out_root,
                attest_root=attest_root,
            )
            grouped_outputs[bucket].append(out)

    write_representative_summary(summary_path, grouped_outputs)

    dataset_sha = sha256_file(cfa_path)
    script_hashes = collect_script_hashes(repo_root)
    write_index(index_path, args, dataset_sha, script_hashes, selected)

    print(f"summary_md={summary_path.as_posix()}")
    print(f"index_md={index_path.as_posix()}")
    print(f"batch_report={batch_report.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
