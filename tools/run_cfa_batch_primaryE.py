#!/usr/bin/env python3
"""Run preregistered CFA batch evaluation (Primary=E, seed=7, perm-R=2000)."""

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import extract_triality_triplets as extractor
import labels_from_cfa_spans as cfa_labels


PRIMARY_SCORE = "E"
PREREG_SEED = 7
PREREG_PERM_R = 2000
MIN_TOKEN_MATCH = 0.98
DEFAULT_MIN_COVERAGE = 0.30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute CFA batch pipeline (extract -> labels -> eval) and aggregate "
            "results under preregistered constraints."
        )
    )
    parser.add_argument("--cfa-jsonl", default="data/cfa/cfa_v1.jsonl")
    parser.add_argument("--out-dir", default="runs/cfa_batch_primaryE")
    parser.add_argument("--attest-dir", default="attestations/triality")
    parser.add_argument("--model-id", help="Optional model id override.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE)
    parser.add_argument("--seed", type=int, default=PREREG_SEED)
    parser.add_argument("--perm-r", type=int, default=PREREG_PERM_R)
    parser.add_argument("--sample-id-min", type=int)
    parser.add_argument("--sample-id-max", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Skip aggregate step and only emit sample-level results.",
    )
    args = parser.parse_args()

    if args.seed != PREREG_SEED:
        parser.error(f"--seed is locked to {PREREG_SEED} for preregistered run.")
    if args.perm_r != PREREG_PERM_R:
        parser.error(f"--perm-r is locked to {PREREG_PERM_R} for preregistered run.")
    if args.topk <= 0:
        parser.error("--topk must be > 0")
    if args.min_coverage < 0.0 or args.min_coverage > 1.0:
        parser.error("--min-coverage must be in [0,1]")
    if args.sample_id_min is not None and args.sample_id_min < 0:
        parser.error("--sample-id-min must be >= 0")
    if args.sample_id_max is not None and args.sample_id_max < 0:
        parser.error("--sample-id-max must be >= 0")
    if (
        args.sample_id_min is not None
        and args.sample_id_max is not None
        and args.sample_id_max < args.sample_id_min
    ):
        parser.error("--sample-id-max must be >= --sample-id-min")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be > 0")
    return args


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def load_cfa_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                raise ValueError(f"non-object row at {path}:{line_no}")
            rows.append(obj)
    rows.sort(key=lambda r: int(r.get("sample_id", -1)))
    return rows


def filter_rows(
    rows: Sequence[Dict[str, Any]],
    sample_id_min: Optional[int],
    sample_id_max: Optional[int],
    max_samples: Optional[int],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        sid = int(row.get("sample_id", -1))
        if sid < 0:
            continue
        if sample_id_min is not None and sid < sample_id_min:
            continue
        if sample_id_max is not None and sid > sample_id_max:
            continue
        out.append(row)
    if max_samples is not None:
        return out[:max_samples]
    return out


def parse_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not (out == out and out != float("inf") and out != float("-inf")):
        return None
    return out


def parse_eval_report(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    in_per_score = False
    in_primary = False
    auprc_e: Optional[float] = None
    primary_auprc: Optional[float] = None
    best_baseline: Optional[float] = None
    delta: Optional[float] = None
    p_emp: Optional[float] = None
    verdict: Optional[str] = None

    for line in lines:
        s = line.strip()
        if s == "per_score_metrics:":
            in_per_score = True
            in_primary = False
            continue
        if s == "primary_endpoint:":
            in_primary = True
            in_per_score = False
            continue
        if not s:
            in_per_score = False
            in_primary = False
            continue

        if s.startswith("best_baseline_auprc="):
            best_baseline = parse_float(s.split("=", 1)[1])
            continue
        if s.startswith("delta_auprc_primary_vs_best_baseline="):
            delta = parse_float(s.split("=", 1)[1])
            continue
        if s.startswith("verdict="):
            verdict = s.split("=", 1)[1].strip()
            continue

        if in_per_score:
            if s.startswith("score,"):
                continue
            parts = s.split(",")
            if len(parts) >= 7 and parts[0] == "E:V_Sminus_Vnext":
                auprc_e = parse_float(parts[5])
            continue

        if in_primary:
            if s.startswith("auprc="):
                primary_auprc = parse_float(s.split("=", 1)[1])
            elif s.startswith("perm_p_empirical="):
                p_emp = parse_float(s.split("=", 1)[1])

    return {
        "AUPRC_E": auprc_e,
        "AUPRC_primary": primary_auprc,
        "AUPRC_best_baseline": best_baseline,
        "delta": delta,
        "p_emp": p_emp,
        "verdict": verdict,
    }


def run_eval(
    repo_root: Path,
    ndjson_path: Path,
    labels_path: Path,
    labels_meta_path: Path,
    out_report: Path,
    min_coverage: float,
    seed: int,
    perm_r: int,
) -> None:
    cmd = [
        sys.executable,
        str((repo_root / "tools" / "eval_triality_token.py").resolve()),
        "--ndjson",
        str(ndjson_path),
        "--labels-jsonl",
        str(labels_path),
        "--labels-meta-json",
        str(labels_meta_path),
        "--min-label-coverage",
        str(min_coverage),
        "--perm-R",
        str(perm_r),
        "--seed",
        str(seed),
        "--primary-score",
        PRIMARY_SCORE,
        "--out",
        str(out_report),
    ]
    completed = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"eval_triality_token.py failed rc={completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(str(repo_root))

    cfa_path = Path(args.cfa_jsonl)
    out_dir = Path(args.out_dir)
    attest_dir = Path(args.attest_dir)
    batch_attest_dir = attest_dir / "cfa_batch"
    results_path = out_dir / "results.jsonl"
    date_stamp = dt.date.today().isoformat()
    aggregate_report = attest_dir / f"{date_stamp}_cfa_batch_primaryE_report.txt"

    out_dir.mkdir(parents=True, exist_ok=True)
    batch_attest_dir.mkdir(parents=True, exist_ok=True)
    if results_path.exists():
        results_path.unlink()

    rows = load_cfa_rows(cfa_path)
    selected = filter_rows(
        rows=rows,
        sample_id_min=args.sample_id_min,
        sample_id_max=args.sample_id_max,
        max_samples=args.max_samples,
    )
    if not selected:
        raise RuntimeError("No CFA rows selected after filters.")

    extractor.configure_reproducibility(args.seed, deterministic=True)
    device = extractor.resolve_device(args.device)
    model_candidates = extractor.build_model_candidates(args.model_id)
    model_id, tokenizer, model, model_revision = extractor.load_first_available_model(
        model_candidates=model_candidates,
        device=device,
    )

    print(f"model_id={model_id}")
    print(f"model_revision={model_revision}")
    print(f"selected_rows={len(selected)}")

    for index, sample in enumerate(selected, start=1):
        sample_id = int(sample.get("sample_id", -1))
        variant = str(sample.get("variant", "unknown"))
        world_type = str(sample.get("world_type", "unknown"))
        sample_dir = out_dir / "samples" / f"sample_{sample_id:06d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        prompt = str(sample.get("prompt", ""))
        answer = str(sample.get("answer", ""))
        prompt_path = sample_dir / "prompt.txt"
        answer_path = sample_dir / "answer.txt"
        triplets_path = sample_dir / "triplets.ndjson"
        meta_path = sample_dir / "meta.json"
        labels_path = sample_dir / "labels.jsonl"
        labels_meta_path = sample_dir / "labels_meta.json"
        eval_report_path = batch_attest_dir / f"{date_stamp}_cfa_eval_sample{sample_id:06d}_E.txt"

        result_row: Dict[str, Any] = {
            "sample_id": sample_id,
            "variant": variant,
            "world_type": world_type,
            "primary_score": PRIMARY_SCORE,
            "seed": args.seed,
            "perm_r": args.perm_r,
            "status": "error",
            "reason": None,
            "exact_token_match_ratio": None,
            "coverage": None,
            "AUPRC_E": None,
            "AUPRC_best_baseline": None,
            "delta": None,
            "p_emp": None,
            "no_positive_imputed": False,
            "sample_dir": sample_dir.as_posix(),
            "eval_report": eval_report_path.as_posix(),
            "model_id": model_id,
            "model_revision": model_revision,
        }

        try:
            print(f"[{index}/{len(selected)}] sample_id={sample_id} variant={variant}")
            if not prompt or not answer:
                result_row["status"] = "skip_empty_text"
                result_row["reason"] = "empty_prompt_or_answer"
                append_jsonl(results_path, result_row)
                continue

            write_text(prompt_path, prompt)
            write_text(answer_path, answer)

            triplet_rows, triplet_meta = extractor.run_teacher_forcing_extraction(
                prompt=prompt,
                target_answer=answer,
                model=model,
                tokenizer=tokenizer,
                device=device,
                topk=args.topk,
            )
            mode_details = triplet_meta["mode_details"]
            ratio = parse_float(mode_details.get("exact_token_match_ratio"))
            result_row["exact_token_match_ratio"] = ratio

            ndjson_sha = extractor.write_ndjson(triplets_path, triplet_rows)
            meta_payload = {
                "model_id": model_id,
                "model_revision": model_revision,
                "seed": args.seed,
                "topk_requested": args.topk,
                "topk_effective": int(triplet_meta["topk_effective"]),
                "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
                "target_answer_sha256": sha256_bytes(answer.encode("utf-8")),
                "output_ndjson_sha256": ndjson_sha,
                "output_ndjson_path": triplets_path.as_posix(),
                "device": str(device),
                "deterministic_requested": True,
                "n_steps_written": len(triplet_rows),
                "extraction_mode": mode_details.get("mode"),
                "alignment_method": mode_details.get("alignment_method"),
                "target_token_count_expected": mode_details.get("target_token_count_expected"),
                "target_token_count_extracted": mode_details.get("target_token_count_extracted"),
                "exact_token_match_ratio": mode_details.get("exact_token_match_ratio"),
                "target_token_indices_count": mode_details.get("target_token_indices_count"),
                "target_only_token_count": mode_details.get("target_only_token_count"),
                "boundary_merge_token_delta": mode_details.get("boundary_merge_token_delta"),
                "bos_prepended_for_teacher_forcing": mode_details.get(
                    "bos_prepended_for_teacher_forcing"
                ),
            }
            extractor.write_meta_json(meta_path, meta_payload)

            if ratio is None or ratio < MIN_TOKEN_MATCH:
                result_row["status"] = "skip_token_match"
                result_row["reason"] = f"exact_token_match_ratio_below_{MIN_TOKEN_MATCH:.2f}"
                append_jsonl(results_path, result_row)
                continue

            defect_spans = cfa_labels.normalize_spans(
                sample.get("defect_spans", []), answer_len=len(answer)
            )
            mapped = cfa_labels.map_using_triplet_char_offsets(triplet_rows, defect_spans)
            labels = mapped["labels"]
            coverage = parse_float(mapped.get("coverage"))
            result_row["coverage"] = coverage

            token_ids = [int(r["token_id"]) for r in triplet_rows]
            cfa_labels.write_labels_jsonl(labels_path, labels=labels, token_ids=token_ids)
            labels_meta_payload = {
                "label_source": "cfa_defect_spans_v1",
                "cfa_jsonl": cfa_path.as_posix(),
                "sample_id": sample_id,
                "variant": variant,
                "world_type": world_type,
                "triplets_path": triplets_path.as_posix(),
                "label_mapping_mode": mapped.get("mode"),
                "n_triplet_steps": len(token_ids),
                "n_defect_spans": len(defect_spans),
                "mapped_positive_tokens": int(mapped["mapped_positive_tokens"]),
                "total_positive_tokens": int(mapped["total_positive_tokens"]),
                "equal_blocks": int(mapped["equal_blocks"]),
                "final_alignment_coverage_ratio": coverage,
                "min_coverage_threshold": float(args.min_coverage),
                "fail_below_coverage": True,
                "final_positive_steps": int(sum(1 for x in labels if x == 1)),
                "final_negative_steps": int(sum(1 for x in labels if x == 0)),
                "labels_out": labels_path.as_posix(),
            }
            cfa_labels.write_meta_json(labels_meta_path, labels_meta_payload)
            final_positive_steps = int(labels_meta_payload["final_positive_steps"])

            if coverage is None or coverage < args.min_coverage:
                result_row["status"] = "skip_coverage"
                result_row["reason"] = f"coverage_below_{args.min_coverage:.2f}"
                append_jsonl(results_path, result_row)
                continue

            run_eval(
                repo_root=repo_root,
                ndjson_path=triplets_path,
                labels_path=labels_path,
                labels_meta_path=labels_meta_path,
                out_report=eval_report_path,
                min_coverage=args.min_coverage,
                seed=args.seed,
                perm_r=args.perm_r,
            )
            eval_parsed = parse_eval_report(eval_report_path)
            if final_positive_steps == 0:
                # AUPRC is undefined when there are no positive labels; for dataset-level
                # contrast we pin these rows to neutral zero gain.
                if eval_parsed["AUPRC_E"] is None:
                    eval_parsed["AUPRC_E"] = 0.0
                    result_row["no_positive_imputed"] = True
                if eval_parsed["AUPRC_best_baseline"] is None:
                    eval_parsed["AUPRC_best_baseline"] = 0.0
                    result_row["no_positive_imputed"] = True
                if eval_parsed["delta"] is None:
                    eval_parsed["delta"] = 0.0
                    result_row["no_positive_imputed"] = True
                if eval_parsed["p_emp"] is None:
                    eval_parsed["p_emp"] = 1.0
                    result_row["no_positive_imputed"] = True

            result_row["status"] = "ok"
            result_row["AUPRC_E"] = eval_parsed["AUPRC_E"]
            result_row["AUPRC_best_baseline"] = eval_parsed["AUPRC_best_baseline"]
            result_row["delta"] = eval_parsed["delta"]
            result_row["p_emp"] = eval_parsed["p_emp"]
            result_row["eval_verdict"] = eval_parsed["verdict"]
            result_row["reason"] = "ok"
            append_jsonl(results_path, result_row)
        except Exception as exc:
            result_row["status"] = "error"
            result_row["reason"] = str(exc)
            result_row["traceback"] = traceback.format_exc(limit=8)
            append_jsonl(results_path, result_row)
            print(f"  error sample_id={sample_id}: {exc}")

    print(f"results_jsonl={results_path.as_posix()}")

    if args.skip_aggregate:
        return 0

    aggregate_cmd = [
        sys.executable,
        str((repo_root / "tools" / "aggregate_cfa_batch.py").resolve()),
        "--results-jsonl",
        str(results_path),
        "--cfa-jsonl",
        str(cfa_path),
        "--seed",
        str(args.seed),
        "--perm-r",
        str(args.perm_r),
        "--primary-score",
        PRIMARY_SCORE,
        "--out",
        str(aggregate_report),
    ]
    completed = subprocess.run(
        aggregate_cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"aggregate_cfa_batch.py failed rc={completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    if completed.stdout.strip():
        print(completed.stdout.strip())
    print(f"aggregate_report={aggregate_report.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
