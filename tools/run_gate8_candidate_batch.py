#!/usr/bin/env python3
"""Run the fixed Gate8 candidate batch on a materialized Gate8 benchmark."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from aggregate_gate5_spike import read_csv, write_csv
import build_gate6_native_local_span as gate6_builder
import extract_triality_triplets as extractor
import labels_from_cfa_spans as cfa_labels


REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "gate8_candidate_batch_v1"
METHOD_ID = "gate8_candidate_batch_v1"
QUIETNESS_PAIRING_RULE = "world_type_occurrence_index_v1"
DEFAULT_TOPK = 128
DEFAULT_SEED = 7
FIXED_CANDIDATES = (
    {
        "candidate_id": "F",
        "metric_id": "score_F_gram_loop_v1",
        "label_key": "label_token",
        "token_csv_relpath": "gate6f/gate6f_token_telemetry.csv",
    },
    {
        "candidate_id": "gate6f",
        "metric_id": "sigma_gap_tailkeep_weighted_gram_loop_v2",
        "label_key": "label_token",
        "token_csv_relpath": "gate6f/gate6f_token_telemetry.csv",
    },
    {
        "candidate_id": "gate6h",
        "metric_id": "sigma_sqrtgap_tailkeep_object_v2",
        "label_key": "label_token",
        "token_csv_relpath": "gate6h/gate6h_token_telemetry.csv",
    },
    {
        "candidate_id": "gate7c",
        "metric_id": "progression_anisotropic_closure_v3",
        "label_key": "label_transition",
        "token_csv_relpath": "gate7c/gate7c_token_telemetry.csv",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the fixed Gate8 candidate set on a materialized Gate8 benchmark "
            "without reopening candidate or evaluator scope."
        )
    )
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-id", help="Optional explicit HF model id.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sample-limit", type=int)
    return parser.parse_args()


def repo_relative_or_posix(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def write_text(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8", newline="\n")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def fixed_candidate_ids() -> List[str]:
    return [entry["metric_id"] for entry in FIXED_CANDIDATES]


def validate_benchmark_manifest(manifest: Dict[str, Any]) -> None:
    candidate_ids = [entry["metric_id"] for entry in manifest.get("candidate_set", [])]
    if candidate_ids != fixed_candidate_ids():
        raise ValueError(
            "benchmark candidate_set does not match frozen Gate8 set: "
            f"{candidate_ids!r} != {fixed_candidate_ids()!r}"
        )
    if not bool(manifest.get("aggregation_ban", False)):
        raise ValueError("Gate8 execution requires aggregation_ban=true in benchmark manifest")


def quietness_pair_bindings(
    benchmark_rows: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    quiet_cells = ("clean_support", "surface_noisy_clean")
    grouped: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    benchmark_lookup: Dict[str, Dict[str, Any]] = {}
    for row in benchmark_rows:
        benchmark_sample_id = str(row["sample_id"])
        benchmark_lookup[benchmark_sample_id] = row
        cell_id = str(row["cell_id"])
        if cell_id not in quiet_cells:
            continue
        if str(row["answer_target_type"]) != "consistent_answer":
            continue
        grouped[(cell_id, str(row["world_type"]))].append(benchmark_sample_id)

    out_bindings: Dict[str, str] = {}
    pair_rows: List[Dict[str, Any]] = []
    all_world_types = sorted({world_type for _cell_id, world_type in grouped.keys()})
    for world_type in all_world_types:
        clean_ids = sorted(grouped.get(("clean_support", world_type), []))
        noisy_ids = sorted(grouped.get(("surface_noisy_clean", world_type), []))
        if len(clean_ids) != len(noisy_ids):
            raise ValueError(
                f"quietness pairing mismatch for world_type={world_type}: "
                f"{len(clean_ids)} clean vs {len(noisy_ids)} noisy"
            )
        for occurrence_index, (clean_id, noisy_id) in enumerate(zip(clean_ids, noisy_ids)):
            pair_id = f"quiet_pair_{world_type}_{occurrence_index:03d}"
            out_bindings[clean_id] = pair_id
            out_bindings[noisy_id] = pair_id
            pair_rows.append(
                {
                    "quietness_pair_id": pair_id,
                    "pairing_rule": QUIETNESS_PAIRING_RULE,
                    "world_type": world_type,
                    "occurrence_index": occurrence_index,
                    "clean_benchmark_sample_id": clean_id,
                    "clean_world_id": str(benchmark_lookup[clean_id]["world_id"]),
                    "surface_noisy_benchmark_sample_id": noisy_id,
                    "surface_noisy_world_id": str(benchmark_lookup[noisy_id]["world_id"]),
                }
            )
    return out_bindings, pair_rows


def build_sample_registry(
    benchmark_rows: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    sorted_rows = sorted(benchmark_rows, key=lambda row: str(row["sample_id"]))
    if not sorted_rows:
        raise ValueError("benchmark_rows is empty")

    quietness_pair_map, quietness_pairs = quietness_pair_bindings(sorted_rows)
    registry_rows: List[Dict[str, Any]] = []
    for execution_sample_id, row in enumerate(sorted_rows, start=1):
        benchmark_sample_id = str(row["sample_id"])
        registry_rows.append(
            {
                "execution_sample_id": execution_sample_id,
                "benchmark_sample_id": benchmark_sample_id,
                "cell_id": str(row["cell_id"]),
                "world_id": str(row["world_id"]),
                "rendering_id": str(row["rendering_id"]),
                "target_id": str(row["target_id"]),
                "answer_target_type": str(row["answer_target_type"]),
                "world_ordinal": int(row["world_ordinal"]),
                "world_type": str(row["world_type"]),
                "is_conflict_intended": bool(row["is_conflict_intended"]),
                "is_surface_noise_only": bool(row["is_surface_noise_only"]),
                "quietness_pair_id": quietness_pair_map.get(benchmark_sample_id, ""),
            }
        )
    return registry_rows, quietness_pairs


def build_labels_for_benchmark_row(
    benchmark_row: Dict[str, Any],
    triplet_rows: Sequence[Dict[str, Any]],
    labels_path: Path,
) -> Dict[str, Any]:
    answer_text = str(benchmark_row["answer_text"])
    defect_spans = cfa_labels.normalize_spans(
        benchmark_row.get("label_span_defect", []),
        answer_len=len(answer_text),
    )
    mapped = cfa_labels.map_using_triplet_char_offsets(triplet_rows, defect_spans)
    token_ids = [int(row["token_id"]) for row in triplet_rows]
    cfa_labels.write_labels_jsonl(labels_path, labels=mapped["labels"], token_ids=token_ids)
    labels_meta_path = labels_path.with_name(labels_path.stem + "_meta.json")
    labels_meta = {
        "label_source": "gate8_defect_spans_v1",
        "benchmark_sample_id": str(benchmark_row["sample_id"]),
        "cell_id": str(benchmark_row["cell_id"]),
        "world_type": str(benchmark_row["world_type"]),
        "answer_target_type": str(benchmark_row["answer_target_type"]),
        "triplets_path": labels_path.parent.joinpath("triplets.ndjson").as_posix(),
        "label_mapping_mode": str(mapped["mode"]),
        "n_triplet_steps": len(token_ids),
        "n_defect_spans": len(defect_spans),
        "mapped_positive_tokens": int(mapped["mapped_positive_tokens"]),
        "total_positive_tokens": int(mapped["total_positive_tokens"]),
        "equal_blocks": int(mapped["equal_blocks"]),
        "final_alignment_coverage_ratio": float(mapped["coverage"]),
        "min_coverage_threshold": 1.0,
        "fail_below_coverage": True,
        "final_positive_steps": int(sum(1 for label in mapped["labels"] if int(label) == 1)),
        "final_negative_steps": int(sum(1 for label in mapped["labels"] if int(label) == 0)),
        "variant": "frustrated" if defect_spans else "consistent",
        "labels_out": labels_path.as_posix(),
    }
    cfa_labels.write_meta_json(labels_meta_path, labels_meta)
    return labels_meta


def run_subprocess(command: Sequence[str]) -> None:
    completed = subprocess.run(
        list(command),
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed rc={completed.returncode}: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def materialize_samples(
    benchmark_rows: Sequence[Dict[str, Any]],
    registry_rows: Sequence[Dict[str, Any]],
    samples_root: Path,
    model_id: str,
    model_revision: Optional[str],
    tokenizer: Any,
    model: Any,
    device: Any,
    topk: int,
    seed: int,
) -> List[Dict[str, Any]]:
    registry_by_benchmark_id = {
        str(row["benchmark_sample_id"]): row for row in registry_rows
    }
    extraction_rows: List[Dict[str, Any]] = []
    for benchmark_row in sorted(benchmark_rows, key=lambda row: str(row["sample_id"])):
        benchmark_sample_id = str(benchmark_row["sample_id"])
        registry_row = registry_by_benchmark_id[benchmark_sample_id]
        execution_sample_id = int(registry_row["execution_sample_id"])
        sample_dir = samples_root / f"sample_{execution_sample_id:06d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        prompt = str(benchmark_row["prompt"])
        answer_text = str(benchmark_row["answer_text"])
        prompt_path = sample_dir / "prompt.txt"
        answer_path = sample_dir / "answer.txt"
        triplets_path = sample_dir / "triplets.ndjson"
        meta_path = sample_dir / "meta.json"
        labels_path = sample_dir / "labels.jsonl"
        benchmark_row_path = sample_dir / "benchmark_row.json"

        write_text(prompt_path, prompt)
        write_text(answer_path, answer_text)
        write_json(benchmark_row_path, benchmark_row)

        triplet_rows, triplet_meta = extractor.run_teacher_forcing_extraction(
            prompt=prompt,
            target_answer=answer_text,
            model=model,
            tokenizer=tokenizer,
            device=device,
            topk=topk,
            emit_native_raw=True,
        )
        ndjson_sha = extractor.write_ndjson(triplets_path, triplet_rows)
        mode_details = triplet_meta["mode_details"]
        meta_payload = {
            "model_id": model_id,
            "model_revision": model_revision,
            "seed": int(seed),
            "topk_requested": int(topk),
            "topk_effective": int(triplet_meta["topk_effective"]),
            "native_raw_emitted": True,
            "native_raw_schema_id": extractor.RAW_NATIVE_SCHEMA_ID,
            "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
            "target_answer_sha256": sha256_bytes(answer_text.encode("utf-8")),
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
            "bos_prepended_for_teacher_forcing": mode_details.get("bos_prepended_for_teacher_forcing"),
            "proj_id": extractor.PROJ_ID,
            "splus_def_id": extractor.SPLUS_DEF_ID,
            "sminus_def_id": extractor.SMINUS_DEF_ID_TEMPLATE.format(
                topk=int(triplet_meta["topk_effective"])
            ),
            "benchmark_sample_id": benchmark_sample_id,
            "cell_id": str(benchmark_row["cell_id"]),
            "world_type": str(benchmark_row["world_type"]),
            "answer_target_type": str(benchmark_row["answer_target_type"]),
        }
        extractor.write_meta_json(meta_path, meta_payload)
        labels_meta = build_labels_for_benchmark_row(
            benchmark_row=benchmark_row,
            triplet_rows=triplet_rows,
            labels_path=labels_path,
        )
        extraction_rows.append(
            {
                "execution_sample_id": execution_sample_id,
                "benchmark_sample_id": benchmark_sample_id,
                "cell_id": str(benchmark_row["cell_id"]),
                "world_type": str(benchmark_row["world_type"]),
                "answer_target_type": str(benchmark_row["answer_target_type"]),
                "n_steps_written": len(triplet_rows),
                "exact_token_match_ratio": float(meta_payload["exact_token_match_ratio"]),
                "label_coverage_ratio": float(labels_meta["final_alignment_coverage_ratio"]),
                "quietness_pair_id": str(registry_row.get("quietness_pair_id") or ""),
                "sample_dir": repo_relative_or_posix(sample_dir),
            }
        )
    return extraction_rows


def build_candidate_summary(
    evaluations_root: Path,
) -> List[Dict[str, Any]]:
    summary_rows: List[Dict[str, Any]] = []
    for candidate in FIXED_CANDIDATES:
        candidate_dir = evaluations_root / candidate["candidate_id"]
        conflict_rows = read_csv(candidate_dir / "conflict_cell_summary.csv")
        quiet_rows = read_csv(candidate_dir / "quietness_summary.csv")
        quiet_all = next((row for row in quiet_rows if row["bucket"] == "all"), None)
        direct = next((row for row in conflict_rows if row["cell_id"] == "direct_contradiction"), None)
        distributed = next(
            (row for row in conflict_rows if row["cell_id"] == "distributed_incompatibility"),
            None,
        )
        summary_rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "metric_id": candidate["metric_id"],
                "direct_global_auprc": None if direct is None else direct.get("global_auprc"),
                "direct_mean_sample_auprc": None
                if direct is None
                else direct.get("mean_sample_auprc"),
                "direct_mean_hit_at_10": None if direct is None else direct.get("mean_hit_at_10"),
                "direct_mean_first_hit_distance": None
                if direct is None
                else direct.get("mean_first_hit_distance"),
                "distributed_global_auprc": None
                if distributed is None
                else distributed.get("global_auprc"),
                "distributed_mean_sample_auprc": None
                if distributed is None
                else distributed.get("mean_sample_auprc"),
                "distributed_mean_hit_at_10": None
                if distributed is None
                else distributed.get("mean_hit_at_10"),
                "distributed_mean_first_hit_distance": None
                if distributed is None
                else distributed.get("mean_first_hit_distance"),
                "quiet_mean_delta_max": None if quiet_all is None else quiet_all.get("mean_delta_max"),
                "quiet_mean_delta_p90": None if quiet_all is None else quiet_all.get("mean_delta_p90"),
                "quiet_mean_iqr_normalized_delta_max": None
                if quiet_all is None
                else quiet_all.get("mean_iqr_normalized_delta_max"),
                "quiet_mean_top10_inflation": None
                if quiet_all is None
                else quiet_all.get("mean_top10_inflation"),
            }
        )
    return summary_rows


def build_standing_report(
    run_id: str,
    benchmark_manifest: Dict[str, Any],
    sample_registry_rows: Sequence[Dict[str, Any]],
    extraction_rows: Sequence[Dict[str, Any]],
    candidate_summary_rows: Sequence[Dict[str, Any]],
    model_id: str,
    model_revision: Optional[str],
) -> str:
    lines = [
        "# Gate8 Candidate Execution Summary",
        "",
        f"run_id: {run_id}",
        f"benchmark_run_id: {benchmark_manifest.get('run_id', '')}",
        f"model_id: {model_id}",
        f"model_revision: {model_revision or ''}",
        f"n_samples_total: {len(sample_registry_rows)}",
        f"n_quietness_pairs: {sum(1 for row in sample_registry_rows if row.get('quietness_pair_id')) // 2}",
        f"quietness_pairing_rule: {QUIETNESS_PAIRING_RULE}",
        "",
        "## Candidate Summary",
        "",
        "| candidate_id | direct_global_auprc | direct_mean_sample_auprc | direct_mean_hit@10 | distributed_global_auprc | distributed_mean_sample_auprc | distributed_mean_hit@10 | quiet_mean_delta_p90 | quiet_mean_top10_inflation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in candidate_summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["candidate_id"]),
                    "" if row["direct_global_auprc"] in (None, "") else f"{float(row['direct_global_auprc']):.6f}",
                    "" if row["direct_mean_sample_auprc"] in (None, "") else f"{float(row['direct_mean_sample_auprc']):.6f}",
                    "" if row["direct_mean_hit_at_10"] in (None, "") else f"{float(row['direct_mean_hit_at_10']):.6f}",
                    "" if row["distributed_global_auprc"] in (None, "") else f"{float(row['distributed_global_auprc']):.6f}",
                    "" if row["distributed_mean_sample_auprc"] in (None, "") else f"{float(row['distributed_mean_sample_auprc']):.6f}",
                    "" if row["distributed_mean_hit_at_10"] in (None, "") else f"{float(row['distributed_mean_hit_at_10']):.6f}",
                    "" if row["quiet_mean_delta_p90"] in (None, "") else f"{float(row['quiet_mean_delta_p90']):.6f}",
                    "" if row["quiet_mean_top10_inflation"] in (None, "") else f"{float(row['quiet_mean_top10_inflation']):.6f}",
                ]
            )
            + " |"
        )
    if extraction_rows:
        min_match = min(float(row["exact_token_match_ratio"]) for row in extraction_rows)
        min_coverage = min(float(row["label_coverage_ratio"]) for row in extraction_rows)
        lines.extend(
            [
                "",
                "## Extraction Hygiene",
                "",
                f"- min_exact_token_match_ratio: {min_match:.6f}",
                f"- min_label_coverage_ratio: {min_coverage:.6f}",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    repo_root = REPO_ROOT
    os.chdir(str(repo_root))

    benchmark_dir = Path(args.benchmark_dir)
    out_dir = Path(args.out_dir)
    run_id = out_dir.name
    samples_root = out_dir / "samples"
    gate6_dir = out_dir / "gate6_native"
    gate6f_dir = out_dir / "gate6f"
    gate6h_dir = out_dir / "gate6h"
    gate7c_dir = out_dir / "gate7c"
    evaluations_root = out_dir / "evaluations"
    manifest_path = out_dir / "manifest.json"
    sample_registry_path = out_dir / "sample_registry.jsonl"
    quietness_pairs_path = out_dir / "quietness_pairs.jsonl"
    extraction_results_path = out_dir / "extraction_results.jsonl"
    summary_csv_path = out_dir / "candidate_summary.csv"
    report_path = out_dir / "gate8a_standing_summary.md"
    checksums_path = out_dir / "checksums.json"

    benchmark_manifest = read_json(benchmark_dir / "manifest.json")
    validate_benchmark_manifest(benchmark_manifest)
    benchmark_rows = read_jsonl(benchmark_dir / "benchmark_rows.jsonl")
    if args.sample_limit is not None:
        benchmark_rows = sorted(benchmark_rows, key=lambda row: str(row["sample_id"]))[: args.sample_limit]
    sample_registry_rows, quietness_pair_rows = build_sample_registry(benchmark_rows)
    write_jsonl(sample_registry_path, sample_registry_rows)
    write_jsonl(quietness_pairs_path, quietness_pair_rows)

    extractor.configure_reproducibility(args.seed, deterministic=True)
    device = extractor.resolve_device(args.device)
    model_candidates = extractor.build_model_candidates(args.model_id)
    model_id, tokenizer, model, model_revision = extractor.load_first_available_model(
        model_candidates=model_candidates,
        device=device,
    )

    extraction_rows = materialize_samples(
        benchmark_rows=benchmark_rows,
        registry_rows=sample_registry_rows,
        samples_root=samples_root,
        model_id=model_id,
        model_revision=model_revision,
        tokenizer=tokenizer,
        model=model,
        device=device,
        topk=args.topk,
        seed=args.seed,
    )
    write_jsonl(extraction_results_path, extraction_rows)

    run_subprocess(
        [
            sys.executable,
            str((repo_root / "tools" / "build_gate6_native_local_span.py").resolve()),
            "--samples-root",
            str(samples_root),
            "--all-samples",
            "--out-dir",
            str(gate6_dir),
        ]
    )
    run_subprocess(
        [
            sys.executable,
            str((repo_root / "tools" / "run_gate6_sigma_gram_consumer_v2.py").resolve()),
            "--gate6-dir",
            str(gate6_dir),
            "--out-dir",
            str(gate6f_dir),
            "--run-id",
            f"{run_id}_gate6f",
        ]
    )
    run_subprocess(
        [
            sys.executable,
            str((repo_root / "tools" / "run_gate6_sigma_object_consumer_v2.py").resolve()),
            "--gate6-dir",
            str(gate6_dir),
            "--out-dir",
            str(gate6h_dir),
            "--run-id",
            f"{run_id}_gate6h",
        ]
    )
    run_subprocess(
        [
            sys.executable,
            str((repo_root / "tools" / "run_gate7_progression_anisotropic_consumer_v3.py").resolve()),
            "--gate6-dir",
            str(gate6_dir),
            "--out-dir",
            str(gate7c_dir),
            "--run-id",
            f"{run_id}_gate7c",
        ]
    )

    for candidate in FIXED_CANDIDATES:
        run_subprocess(
            [
                sys.executable,
                str((repo_root / "tools" / "evaluate_gate8_standing.py").resolve()),
                "--sample-registry-jsonl",
                str(sample_registry_path),
                "--token-csv",
                str(out_dir / candidate["token_csv_relpath"]),
                "--out-dir",
                str(evaluations_root / candidate["candidate_id"]),
                "--run-id",
                f"{run_id}_{candidate['candidate_id']}",
                "--candidate-id",
                candidate["candidate_id"],
                "--metric-id",
                candidate["metric_id"],
                "--label-key",
                candidate["label_key"],
                "--topk",
                "10",
            ]
        )

    candidate_summary_rows = build_candidate_summary(evaluations_root)
    write_csv(
        summary_csv_path,
        (
            "candidate_id",
            "metric_id",
            "direct_global_auprc",
            "direct_mean_sample_auprc",
            "direct_mean_hit_at_10",
            "direct_mean_first_hit_distance",
            "distributed_global_auprc",
            "distributed_mean_sample_auprc",
            "distributed_mean_hit_at_10",
            "distributed_mean_first_hit_distance",
            "quiet_mean_delta_max",
            "quiet_mean_delta_p90",
            "quiet_mean_iqr_normalized_delta_max",
            "quiet_mean_top10_inflation",
        ),
        candidate_summary_rows,
    )
    report = build_standing_report(
        run_id=run_id,
        benchmark_manifest=benchmark_manifest,
        sample_registry_rows=sample_registry_rows,
        extraction_rows=extraction_rows,
        candidate_summary_rows=candidate_summary_rows,
        model_id=model_id,
        model_revision=model_revision,
    )
    write_text(report_path, report)

    write_json(
        manifest_path,
        {
            "run_id": run_id,
            "schema_version": SCHEMA_VERSION,
            "method_id": METHOD_ID,
            "benchmark_dir": repo_relative_or_posix(benchmark_dir),
            "benchmark_manifest_path": repo_relative_or_posix(benchmark_dir / "manifest.json"),
            "benchmark_manifest_sha256": sha256_file(benchmark_dir / "manifest.json"),
            "benchmark_rows_path": repo_relative_or_posix(benchmark_dir / "benchmark_rows.jsonl"),
            "benchmark_rows_sha256": sha256_file(benchmark_dir / "benchmark_rows.jsonl"),
            "sample_registry_path": repo_relative_or_posix(sample_registry_path),
            "sample_registry_sha256": sha256_file(sample_registry_path),
            "quietness_pairs_path": repo_relative_or_posix(quietness_pairs_path),
            "quietness_pairs_sha256": sha256_file(quietness_pairs_path),
            "extraction_results_path": repo_relative_or_posix(extraction_results_path),
            "extraction_results_sha256": sha256_file(extraction_results_path),
            "candidate_summary_path": repo_relative_or_posix(summary_csv_path),
            "candidate_summary_sha256": sha256_file(summary_csv_path),
            "model_id": model_id,
            "model_revision": model_revision,
            "device": str(device),
            "topk_requested": int(args.topk),
            "seed": int(args.seed),
            "quietness_pairing_rule": QUIETNESS_PAIRING_RULE,
            "candidate_set": [
                {"candidate_id": row["candidate_id"], "metric_id": row["metric_id"]}
                for row in candidate_summary_rows
            ],
            "code_git_commit": gate6_builder.current_git_commit(),
            "n_samples_total": len(sample_registry_rows),
        },
    )
    write_json(
        checksums_path,
        {
            "manifest.json": sha256_file(manifest_path),
            "sample_registry.jsonl": sha256_file(sample_registry_path),
            "quietness_pairs.jsonl": sha256_file(quietness_pairs_path),
            "extraction_results.jsonl": sha256_file(extraction_results_path),
            "candidate_summary.csv": sha256_file(summary_csv_path),
            "gate8a_standing_summary.md": sha256_file(report_path),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
