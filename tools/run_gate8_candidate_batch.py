#!/usr/bin/env python3
"""Run the fixed Gate8 candidate batch on a materialized Gate8 benchmark."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from aggregate_gate5_spike import read_csv, write_csv
import build_gate6_native_local_span as gate6_builder
import extract_triality_triplets as extractor
import labels_from_cfa_spans as cfa_labels
import run_gate7_progression_anisotropic_consumer_v3 as gate7c_consumer


REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "gate8_candidate_batch_v1"
METHOD_ID = "gate8_candidate_batch_v1"
QUIETNESS_PAIRING_RULE = "shared_world_id_v1"
DEFAULT_TOPK = 128
DEFAULT_SEED = 7
FIXED_CANDIDATES = (
    {
        "candidate_id": "F",
        "role": "legacy_guardrail",
        "metric_id": "score_F_gram_loop_v1",
        "label_key": "label_token",
        "label_granularity": "token",
        "token_csv_relpath": "gate6f/gate6f_token_telemetry.csv",
    },
    {
        "candidate_id": "gate6f",
        "role": "operational_candidate",
        "metric_id": "sigma_gap_tailkeep_weighted_gram_loop_v2",
        "label_key": "label_token",
        "label_granularity": "token",
        "token_csv_relpath": "gate6f/gate6f_token_telemetry.csv",
    },
    {
        "candidate_id": "gate6h",
        "role": "research_north_star",
        "metric_id": "sigma_sqrtgap_tailkeep_object_v2",
        "label_key": "label_token",
        "label_granularity": "token",
        "token_csv_relpath": "gate6h/gate6h_token_telemetry.csv",
    },
    {
        "candidate_id": "gate7c",
        "role": "dynamic_candidate",
        "metric_id": "progression_anisotropic_closure_v3",
        "label_key": "label_transition",
        "label_granularity": "transition",
        "token_csv_relpath": "gate7c/gate7c_token_telemetry.csv",
    },
)
GRANULARITY_COURT_STATUS = "mixed_candidate_label_granularity_v1"
GRANULARITY_COURT_NOTE = (
    "Gate8 fixed standing is regime-consistent but not same-granularity: "
    "F/gate6f/gate6h use label_token while gate7c uses label_transition."
)
BRIDGE_METHOD_ID = "gate8_rotation_leakage_bridge_v1"
BRIDGE_STATUS = "diagnostic_only"
BRIDGE_SOURCE_CANDIDATE_ID = "gate7c"
BRIDGE_DOC_PATH = "18_GATE8_ROTATION_LEAKAGE_BRIDGE.md"
BRIDGE_PER_SAMPLE_FILENAME = "rotation_leakage_per_sample.csv"
BRIDGE_BY_CELL_FILENAME = "rotation_leakage_by_cell.csv"
BRIDGE_REPORT_FILENAME = "rotation_leakage_bridge_report.md"


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


def fixed_candidate_contract_rows() -> List[Dict[str, str]]:
    return [
        {
            "candidate_id": str(entry["candidate_id"]),
            "role": str(entry["role"]),
            "metric_id": str(entry["metric_id"]),
            "label_key": str(entry["label_key"]),
            "label_granularity": str(entry["label_granularity"]),
        }
        for entry in FIXED_CANDIDATES
    ]


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    if q < 0.0 or q > 100.0:
        raise ValueError(f"percentile q must be in [0, 100], got {q}")
    arr = sorted(float(value) for value in values)
    rank = int(math.ceil((q / 100.0) * len(arr))) - 1
    rank = max(0, min(rank, len(arr) - 1))
    return float(arr[rank])


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(float(value) for value in values) / float(len(values)))


def projector_gap(
    current_basis: np.ndarray,
    current_rank: int,
    next_basis: np.ndarray,
    next_rank: int,
) -> Optional[float]:
    if current_rank <= 0 or next_rank <= 0:
        return None
    current_slice = np.asarray(current_basis[:, :current_rank], dtype=np.float64)
    next_slice = np.asarray(next_basis[:, :next_rank], dtype=np.float64)
    overlap_matrix = current_slice.T @ next_slice
    overlap_trace = float(np.sum(np.square(overlap_matrix)))
    denom = float(max(1, min(current_rank, next_rank)))
    return float(np.clip(1.0 - (overlap_trace / denom), 0.0, 1.0))


def build_orthogonal_projector(
    basis: np.ndarray,
    rank_local: int,
) -> Optional[np.ndarray]:
    if rank_local <= 0:
        return None
    basis_slice = np.asarray(basis[:, :rank_local], dtype=np.float64)
    return basis_slice @ basis_slice.T


def compute_rotation_leakage_bridge_metrics(
    current_basis: np.ndarray,
    current_singular_values: np.ndarray,
    current_rank: int,
    next_basis: np.ndarray,
    next_singular_values: np.ndarray,
    next_coords_local: np.ndarray,
    next_rank: int,
) -> Dict[str, Any]:
    current_projector = build_orthogonal_projector(
        current_basis,
        current_rank,
    )
    if current_projector is None:
        return {
            "bridge_outcome": "invalid_current_projector",
            "rotation_only": None,
            "leakage_only": None,
            "closure_defect": None,
        }

    current_operator = gate7c_consumer.build_anisotropic_operator(
        current_basis,
        current_singular_values,
        current_rank,
    )
    if current_operator is None:
        return {
            "bridge_outcome": "invalid_current_operator",
            "rotation_only": None,
            "leakage_only": None,
            "closure_defect": None,
        }

    next_operator = gate7c_consumer.build_anisotropic_operator(
        next_basis,
        next_singular_values,
        next_rank,
    )
    if next_operator is None:
        return {
            "bridge_outcome": "invalid_next_operator",
            "rotation_only": None,
            "leakage_only": None,
            "closure_defect": None,
        }

    next_v = gate7c_consumer.reconstruct_v(next_basis, next_coords_local, next_rank)
    if next_v is None:
        return {
            "bridge_outcome": "invalid_next_vector",
            "rotation_only": None,
            "leakage_only": None,
            "closure_defect": None,
        }

    rotation_only = projector_gap(
        current_basis=current_basis,
        current_rank=current_rank,
        next_basis=next_basis,
        next_rank=next_rank,
    )
    if rotation_only is None:
        return {
            "bridge_outcome": "invalid_projector_gap",
            "rotation_only": None,
            "leakage_only": None,
            "closure_defect": None,
        }

    projected_current = current_projector @ next_v
    current_applied = current_operator @ next_v
    closure_applied = next_operator @ current_applied
    next_norm_sq = float(np.dot(next_v, next_v))
    projected_norm_sq = float(np.dot(projected_current, projected_current))
    current_norm_sq = float(np.dot(current_applied, current_applied))
    closure_norm_sq = float(np.dot(closure_applied, closure_applied))
    if (
        not np.isfinite(next_norm_sq)
        or next_norm_sq <= 1e-12
        or not np.isfinite(projected_norm_sq)
        or not np.isfinite(current_norm_sq)
        or not np.isfinite(closure_norm_sq)
    ):
        return {
            "bridge_outcome": "invalid_bridge_energy",
            "rotation_only": None,
            "leakage_only": None,
            "closure_defect": None,
        }

    projector_energy_ratio = float(np.clip(projected_norm_sq / next_norm_sq, 0.0, 1.0))
    closure_energy_ratio = float(np.clip(closure_norm_sq / next_norm_sq, 0.0, 1.0))
    leakage_only = float(np.clip(1.0 - projector_energy_ratio, 0.0, 1.0))
    closure_defect = float(np.clip(projector_energy_ratio - closure_energy_ratio, 0.0, 1.0))
    return {
        "bridge_outcome": "none",
        "rotation_only": rotation_only,
        "leakage_only": leakage_only,
        "closure_defect": closure_defect,
    }


def validate_benchmark_manifest(manifest: Dict[str, Any]) -> None:
    candidate_rows = list(manifest.get("candidate_set", []))
    candidate_ids = [entry["metric_id"] for entry in candidate_rows]
    if candidate_ids != fixed_candidate_ids():
        raise ValueError(
            "benchmark candidate_set does not match frozen Gate8 set: "
            f"{candidate_ids!r} != {fixed_candidate_ids()!r}"
        )
    required_fields = {"candidate_id", "role", "metric_id", "label_key", "label_granularity"}
    if candidate_rows and all(required_fields.issubset(entry) for entry in candidate_rows):
        normalized = [
            {
                "candidate_id": str(entry["candidate_id"]),
                "role": str(entry["role"]),
                "metric_id": str(entry["metric_id"]),
                "label_key": str(entry["label_key"]),
                "label_granularity": str(entry["label_granularity"]),
            }
            for entry in candidate_rows
        ]
        if normalized != fixed_candidate_contract_rows():
            raise ValueError("benchmark candidate_set metadata does not match frozen Gate8 contract")
    if not bool(manifest.get("aggregation_ban", False)):
        raise ValueError("Gate8 execution requires aggregation_ban=true in benchmark manifest")


def quietness_pair_bindings(
    benchmark_rows: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    quiet_cells = ("clean_support", "surface_noisy_clean")
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in benchmark_rows:
        cell_id = str(row["cell_id"])
        if cell_id not in quiet_cells:
            continue
        if str(row["answer_target_type"]) != "consistent_answer":
            continue
        world_id = str(row["world_id"])
        if cell_id in grouped[world_id]:
            raise ValueError(
                f"duplicate quietness row for world_id={world_id} cell_id={cell_id}"
            )
        grouped[world_id][cell_id] = row

    out_bindings: Dict[str, str] = {}
    pair_rows: List[Dict[str, Any]] = []
    for world_id in sorted(grouped):
        pair_binding = grouped[world_id]
        clean_row = pair_binding.get("clean_support")
        noisy_row = pair_binding.get("surface_noisy_clean")
        if clean_row is None or noisy_row is None:
            raise ValueError(
                f"quietness pairing requires shared clean/noisy rows for world_id={world_id}"
            )
        if str(clean_row["world_id"]) != str(noisy_row["world_id"]):
            raise ValueError(f"quietness pair world_id mismatch for world_id={world_id}")
        pair_id = f"quiet_pair_{world_id}"
        clean_id = str(clean_row["sample_id"])
        noisy_id = str(noisy_row["sample_id"])
        out_bindings[clean_id] = pair_id
        out_bindings[noisy_id] = pair_id
        pair_rows.append(
            {
                "quietness_pair_id": pair_id,
                "pairing_rule": QUIETNESS_PAIRING_RULE,
                "world_id": world_id,
                "world_type": str(clean_row["world_type"]),
                "clean_benchmark_sample_id": clean_id,
                "clean_rendering_id": str(clean_row["rendering_id"]),
                "surface_noisy_benchmark_sample_id": noisy_id,
                "surface_noisy_rendering_id": str(noisy_row["rendering_id"]),
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
                "label_key": candidate["label_key"],
                "label_granularity": candidate["label_granularity"],
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


def build_rotation_leakage_transition_rows(
    step_rows: Sequence[Dict[str, Any]],
    arrays: Dict[str, np.ndarray],
) -> List[Dict[str, Any]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in step_rows:
        grouped[int(row["sample_id"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["step"]))

    transition_rows: List[Dict[str, Any]] = []
    for sample_id in sorted(grouped):
        rows = grouped[sample_id]
        for idx, step_row in enumerate(rows[:-1]):
            next_row = rows[idx + 1]
            current_row_index = int(step_row["array_row_index"])
            next_row_index = int(next_row["array_row_index"])
            bridge_metrics = compute_rotation_leakage_bridge_metrics(
                current_basis=np.asarray(arrays["basis"][current_row_index], dtype=np.float64),
                current_singular_values=np.asarray(
                    arrays["singular_values"][current_row_index],
                    dtype=np.float64,
                ),
                current_rank=int(arrays["rank_local"][current_row_index]),
                next_basis=np.asarray(arrays["basis"][next_row_index], dtype=np.float64),
                next_singular_values=np.asarray(
                    arrays["singular_values"][next_row_index],
                    dtype=np.float64,
                ),
                next_coords_local=np.asarray(arrays["coords_local"][next_row_index], dtype=np.float64),
                next_rank=int(arrays["rank_local"][next_row_index]),
            )
            transition_rows.append(
                {
                    "execution_sample_id": int(step_row["sample_id"]),
                    "step": int(step_row["step"]),
                    "token_text": str(step_row["token_text"]),
                    "label_transition": max(int(step_row["label_token"]), int(next_row["label_token"])),
                    "bridge_outcome": str(bridge_metrics["bridge_outcome"]),
                    "rotation_only": bridge_metrics["rotation_only"],
                    "leakage_only": bridge_metrics["leakage_only"],
                    "closure_defect": bridge_metrics["closure_defect"],
                }
            )
    return transition_rows


def build_rotation_leakage_per_sample_rows(
    sample_registry_rows: Sequence[Dict[str, Any]],
    transition_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    transitions_by_sample: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in transition_rows:
        transitions_by_sample[int(row["execution_sample_id"])].append(row)

    sample_rows: List[Dict[str, Any]] = []
    for registry_row in sorted(
        sample_registry_rows,
        key=lambda row: int(row["execution_sample_id"]),
    ):
        execution_sample_id = int(registry_row["execution_sample_id"])
        rows = sorted(
            transitions_by_sample.get(execution_sample_id, []),
            key=lambda row: int(row["step"]),
        )
        valid_rows = [row for row in rows if row["bridge_outcome"] == "none"]
        rotation_values = [float(row["rotation_only"]) for row in valid_rows if row["rotation_only"] is not None]
        leakage_values = [float(row["leakage_only"]) for row in valid_rows if row["leakage_only"] is not None]
        closure_values = [float(row["closure_defect"]) for row in valid_rows if row["closure_defect"] is not None]
        sample_rows.append(
            {
                "execution_sample_id": execution_sample_id,
                "benchmark_sample_id": str(registry_row["benchmark_sample_id"]),
                "cell_id": str(registry_row["cell_id"]),
                "world_id": str(registry_row["world_id"]),
                "rendering_id": str(registry_row["rendering_id"]),
                "target_id": str(registry_row["target_id"]),
                "world_type": str(registry_row["world_type"]),
                "answer_target_type": str(registry_row["answer_target_type"]),
                "quietness_pair_id": str(registry_row.get("quietness_pair_id") or ""),
                "n_transition_rows_total": len(rows),
                "n_transition_rows_valid": len(valid_rows),
                "n_transition_rows_missing": len(rows) - len(valid_rows),
                "positive_transition_count": sum(int(row["label_transition"]) for row in rows),
                "mean_rotation_only": mean_or_none(rotation_values),
                "p90_rotation_only": percentile(rotation_values, 90.0),
                "max_rotation_only": None if not rotation_values else float(max(rotation_values)),
                "mean_leakage_only": mean_or_none(leakage_values),
                "p90_leakage_only": percentile(leakage_values, 90.0),
                "max_leakage_only": None if not leakage_values else float(max(leakage_values)),
                "mean_closure_defect": mean_or_none(closure_values),
                "p90_closure_defect": percentile(closure_values, 90.0),
                "max_closure_defect": None if not closure_values else float(max(closure_values)),
            }
        )
    return sample_rows


def build_rotation_leakage_by_cell_rows(
    per_sample_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in per_sample_rows:
        grouped[str(row["cell_id"])].append(row)

    cell_rows: List[Dict[str, Any]] = []
    for cell_id in sorted(grouped):
        rows = grouped[cell_id]
        cell_rows.append(
            {
                "cell_id": cell_id,
                "n_samples": len(rows),
                "n_transition_rows_total": sum(int(row["n_transition_rows_total"]) for row in rows),
                "n_transition_rows_valid": sum(int(row["n_transition_rows_valid"]) for row in rows),
                "n_transition_rows_missing": sum(int(row["n_transition_rows_missing"]) for row in rows),
                "mean_sample_mean_rotation_only": mean_or_none(
                    [float(row["mean_rotation_only"]) for row in rows if row["mean_rotation_only"] not in (None, "")]
                ),
                "mean_sample_p90_rotation_only": mean_or_none(
                    [float(row["p90_rotation_only"]) for row in rows if row["p90_rotation_only"] not in (None, "")]
                ),
                "mean_sample_max_rotation_only": mean_or_none(
                    [float(row["max_rotation_only"]) for row in rows if row["max_rotation_only"] not in (None, "")]
                ),
                "mean_sample_mean_leakage_only": mean_or_none(
                    [float(row["mean_leakage_only"]) for row in rows if row["mean_leakage_only"] not in (None, "")]
                ),
                "mean_sample_p90_leakage_only": mean_or_none(
                    [float(row["p90_leakage_only"]) for row in rows if row["p90_leakage_only"] not in (None, "")]
                ),
                "mean_sample_max_leakage_only": mean_or_none(
                    [float(row["max_leakage_only"]) for row in rows if row["max_leakage_only"] not in (None, "")]
                ),
                "mean_sample_mean_closure_defect": mean_or_none(
                    [float(row["mean_closure_defect"]) for row in rows if row["mean_closure_defect"] not in (None, "")]
                ),
                "mean_sample_p90_closure_defect": mean_or_none(
                    [float(row["p90_closure_defect"]) for row in rows if row["p90_closure_defect"] not in (None, "")]
                ),
                "mean_sample_max_closure_defect": mean_or_none(
                    [float(row["max_closure_defect"]) for row in rows if row["max_closure_defect"] not in (None, "")]
                ),
            }
        )
    return cell_rows


def build_rotation_leakage_bridge_report(
    run_id: str,
    per_sample_rows: Sequence[Dict[str, Any]],
    by_cell_rows: Sequence[Dict[str, Any]],
) -> str:
    by_cell_ordered = list(by_cell_rows)
    rotation_ranked = sorted(
        [
            row
            for row in by_cell_ordered
            if row["mean_sample_mean_rotation_only"] not in (None, "")
        ],
        key=lambda row: float(row["mean_sample_mean_rotation_only"]),
        reverse=True,
    )
    leakage_ranked = sorted(
        [
            row
            for row in by_cell_ordered
            if row["mean_sample_mean_leakage_only"] not in (None, "")
        ],
        key=lambda row: float(row["mean_sample_mean_leakage_only"]),
    )
    closure_p90_ranked = sorted(
        [
            row
            for row in by_cell_ordered
            if row["mean_sample_p90_closure_defect"] not in (None, "")
        ],
        key=lambda row: float(row["mean_sample_p90_closure_defect"]),
        reverse=True,
    )
    lines = [
        "# Gate8 Rotation/Leakage Bridge Diagnostics",
        "",
        f"run_id: {run_id}",
        f"method_id: {BRIDGE_METHOD_ID}",
        f"status: {BRIDGE_STATUS}",
        f"source_candidate_id: {BRIDGE_SOURCE_CANDIDATE_ID}",
        f"bridge_doc: {BRIDGE_DOC_PATH}",
        "",
        "## Scope",
        "",
        "- fixed standing court remains unchanged",
        "- diagnostics are emitted beside standing outputs, not inside them",
        "- rotation_only measures consecutive local-frame projector gap",
        "- leakage_only measures energy outside the current unweighted local span before anisotropic weighting",
        "- closure_defect measures residual loss after span survival under the current dynamic law and next-frame re-entry",
        "",
        f"- n_samples_total: {len(per_sample_rows)}",
        f"- n_cells_total: {len(by_cell_rows)}",
    ]
    if rotation_ranked and leakage_ranked and closure_p90_ranked:
        closure_runner_up = (
            None
            if len(closure_p90_ranked) < 2
            else closure_p90_ranked[1]
        )
        lines.extend(
            [
                "",
                "## Failure Read",
                "",
                (
                    "- rotation_only does not isolate conflict cells: "
                    f"highest mean is {rotation_ranked[0]['cell_id']}="
                    f"{float(rotation_ranked[0]['mean_sample_mean_rotation_only']):.6f}"
                ),
                (
                    "- leakage_only does not behave like a clean-negative-friendly span-escape cut: "
                    f"lowest mean is {leakage_ranked[0]['cell_id']}="
                    f"{float(leakage_ranked[0]['mean_sample_mean_leakage_only']):.6f}"
                ),
                (
                    "- closure_defect does not cleanly isolate distributed incompatibility: "
                    f"highest p90 is {closure_p90_ranked[0]['cell_id']}="
                    f"{float(closure_p90_ranked[0]['mean_sample_p90_closure_defect']):.6f}"
                    + (
                        ""
                        if closure_runner_up is None
                        else (
                            f", runner-up is {closure_runner_up['cell_id']}="
                            f"{float(closure_runner_up['mean_sample_p90_closure_defect']):.6f}"
                        )
                    )
                ),
                "- bridge v1 should be read as an explanatory-cut failure, not as a validated separation.",
            ]
        )
    lines.extend(
        [
        "",
        "## Cell Aggregates",
        "",
        "| cell_id | n_samples | n_transition_rows_valid | mean_sample_mean_rotation_only | mean_sample_mean_leakage_only | mean_sample_mean_closure_defect | mean_sample_p90_rotation_only | mean_sample_p90_leakage_only | mean_sample_p90_closure_defect |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in by_cell_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["cell_id"]),
                    str(row["n_samples"]),
                    str(row["n_transition_rows_valid"]),
                    "" if row["mean_sample_mean_rotation_only"] in (None, "") else f"{float(row['mean_sample_mean_rotation_only']):.6f}",
                    "" if row["mean_sample_mean_leakage_only"] in (None, "") else f"{float(row['mean_sample_mean_leakage_only']):.6f}",
                    "" if row["mean_sample_mean_closure_defect"] in (None, "") else f"{float(row['mean_sample_mean_closure_defect']):.6f}",
                    "" if row["mean_sample_p90_rotation_only"] in (None, "") else f"{float(row['mean_sample_p90_rotation_only']):.6f}",
                    "" if row["mean_sample_p90_leakage_only"] in (None, "") else f"{float(row['mean_sample_p90_leakage_only']):.6f}",
                    "" if row["mean_sample_p90_closure_defect"] in (None, "") else f"{float(row['mean_sample_p90_closure_defect']):.6f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def granularity_buckets() -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = defaultdict(list)
    for candidate in FIXED_CANDIDATES:
        grouped[str(candidate["label_granularity"])].append(str(candidate["candidate_id"]))
    return {key: sorted(value) for key, value in grouped.items()}


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
        f"candidate_granularity_status: {GRANULARITY_COURT_STATUS}",
        "",
        "## Candidate Summary",
        "",
        "| candidate_id | label_granularity | direct_global_auprc | direct_mean_sample_auprc | direct_mean_hit@10 | distributed_global_auprc | distributed_mean_sample_auprc | distributed_mean_hit@10 | quiet_mean_delta_p90 | quiet_mean_top10_inflation |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in candidate_summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["candidate_id"]),
                    str(row["label_granularity"]),
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
    buckets = granularity_buckets()
    lines.extend(
        [
            "",
            "## Label Granularity Court",
            "",
            f"- status: {GRANULARITY_COURT_STATUS}",
            f"- note: {GRANULARITY_COURT_NOTE}",
            f"- token candidates: {', '.join(buckets.get('token', []))}",
            f"- transition candidates: {', '.join(buckets.get('transition', []))}",
        ]
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
    diagnostics_dir = out_dir / "diagnostics"
    manifest_path = out_dir / "manifest.json"
    sample_registry_path = out_dir / "sample_registry.jsonl"
    quietness_pairs_path = out_dir / "quietness_pairs.jsonl"
    extraction_results_path = out_dir / "extraction_results.jsonl"
    summary_csv_path = out_dir / "candidate_summary.csv"
    report_path = out_dir / "gate8a_standing_summary.md"
    bridge_per_sample_path = diagnostics_dir / BRIDGE_PER_SAMPLE_FILENAME
    bridge_by_cell_path = diagnostics_dir / BRIDGE_BY_CELL_FILENAME
    bridge_report_path = diagnostics_dir / BRIDGE_REPORT_FILENAME
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

    gate6_step_rows = gate7c_consumer.load_rows(gate6_dir / gate7c_consumer.DEFAULT_STEP_INDEX)
    with np.load(gate6_dir / gate7c_consumer.DEFAULT_ARRAYS) as npz_handle:
        gate6_arrays = {
            "basis": np.asarray(npz_handle["basis"], dtype=np.float64),
            "coords_local": np.asarray(npz_handle["coords_local"], dtype=np.float64),
            "singular_values": np.asarray(npz_handle["singular_values"], dtype=np.float64),
            "rank_local": np.asarray(npz_handle["rank_local"], dtype=np.int64),
        }
    bridge_transition_rows = build_rotation_leakage_transition_rows(
        step_rows=gate6_step_rows,
        arrays=gate6_arrays,
    )
    bridge_per_sample_rows = build_rotation_leakage_per_sample_rows(
        sample_registry_rows=sample_registry_rows,
        transition_rows=bridge_transition_rows,
    )
    bridge_by_cell_rows = build_rotation_leakage_by_cell_rows(bridge_per_sample_rows)
    write_csv(
        bridge_per_sample_path,
        (
            "execution_sample_id",
            "benchmark_sample_id",
            "cell_id",
            "world_id",
            "rendering_id",
            "target_id",
            "world_type",
            "answer_target_type",
            "quietness_pair_id",
            "n_transition_rows_total",
            "n_transition_rows_valid",
            "n_transition_rows_missing",
            "positive_transition_count",
            "mean_rotation_only",
            "p90_rotation_only",
            "max_rotation_only",
            "mean_leakage_only",
            "p90_leakage_only",
            "max_leakage_only",
            "mean_closure_defect",
            "p90_closure_defect",
            "max_closure_defect",
        ),
        bridge_per_sample_rows,
    )
    write_csv(
        bridge_by_cell_path,
        (
            "cell_id",
            "n_samples",
            "n_transition_rows_total",
            "n_transition_rows_valid",
            "n_transition_rows_missing",
            "mean_sample_mean_rotation_only",
            "mean_sample_p90_rotation_only",
            "mean_sample_max_rotation_only",
            "mean_sample_mean_leakage_only",
            "mean_sample_p90_leakage_only",
            "mean_sample_max_leakage_only",
            "mean_sample_mean_closure_defect",
            "mean_sample_p90_closure_defect",
            "mean_sample_max_closure_defect",
        ),
        bridge_by_cell_rows,
    )
    write_text(
        bridge_report_path,
        build_rotation_leakage_bridge_report(
            run_id=run_id,
            per_sample_rows=bridge_per_sample_rows,
            by_cell_rows=bridge_by_cell_rows,
        ),
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
                "--label-granularity",
                candidate["label_granularity"],
                "--topk",
                "10",
            ]
        )

    candidate_summary_rows = build_candidate_summary(evaluations_root)
    write_csv(
        summary_csv_path,
        (
            "candidate_id",
            "label_key",
            "label_granularity",
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
            "diagnostic_bridge": {
                "method_id": BRIDGE_METHOD_ID,
                "status": BRIDGE_STATUS,
                "source_candidate_id": BRIDGE_SOURCE_CANDIDATE_ID,
                "doc_path": repo_relative_or_posix(repo_root / BRIDGE_DOC_PATH),
                "per_sample_path": repo_relative_or_posix(bridge_per_sample_path),
                "per_sample_sha256": sha256_file(bridge_per_sample_path),
                "by_cell_path": repo_relative_or_posix(bridge_by_cell_path),
                "by_cell_sha256": sha256_file(bridge_by_cell_path),
                "report_path": repo_relative_or_posix(bridge_report_path),
                "report_sha256": sha256_file(bridge_report_path),
            },
            "model_id": model_id,
            "model_revision": model_revision,
            "device": str(device),
            "topk_requested": int(args.topk),
            "seed": int(args.seed),
            "quietness_pairing_rule": QUIETNESS_PAIRING_RULE,
            "candidate_granularity_status": GRANULARITY_COURT_STATUS,
            "candidate_granularity_note": GRANULARITY_COURT_NOTE,
            "candidate_set": fixed_candidate_contract_rows(),
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
            repo_relative_or_posix(bridge_per_sample_path): sha256_file(bridge_per_sample_path),
            repo_relative_or_posix(bridge_by_cell_path): sha256_file(bridge_by_cell_path),
            repo_relative_or_posix(bridge_report_path): sha256_file(bridge_report_path),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
