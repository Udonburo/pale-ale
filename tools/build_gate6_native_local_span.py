#!/usr/bin/env python3
"""Build Gate6-A native local span artifacts from raw native triplet samples."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import build_gate4_input as packer


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUT_DIR = "runs/gate6_native_local_span"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_STEP_INDEX = "step_index.jsonl"
DEFAULT_ARRAYS = "native_object_arrays.npz"
DEFAULT_COMPAT_INPUT = "compatibility_input.json"
DEFAULT_CHECKSUMS = "checksums.json"

METHOD_ID = "native_local_span_gate6a_v1"
SCHEMA_VERSION = "gate6_native_local_span_artifacts_v1"
CONSTRUCTION_MODE = "anchor_plus_relations_svd_v1"
SIGN_FIX_MODE = "largest_abs_component_positive_first_index_tie_v1"
COMPATIBILITY_EMBEDDING = "local_rank_to_local8_zero_pad_v1"
BOUNDARY_ORIGIN = "gate6_native_local_span_local8_v1"
COMPAT_INPUT_SCHEMA_ID = "gate6_local8_compat_input_v1"
PROJ_ID = BOUNDARY_ORIGIN
SPLUS_DEF_ID = "gate6_local_span_coord_splus_v1"
SMINUS_DEF_ID = "gate6_local_span_coord_sminus_v1"
SOURCE_TENSOR_ID = "triality_raw_native_v1"
RAW_NATIVE_SCHEMA_ID = "triality_raw_native_v1"
RAW_KEYS = ("V_raw_native", "Splus_raw_native", "Sminus_raw_native")

TAU_NORM_ABS = 1e-12
TAU_RANK_ABS = 1e-10
TAU_RANK_REL = 1e-6
TAU_SIGN_TIE_ABS = 1e-15
TAU_WARN_REL = 1e-4
COMPAT_DIM = 8
LOCAL_DIM_MAX = 3
FLAG_FIELDS = (
    "all_finite",
    "used_sign_fix",
    "rank_drop_to_2",
    "rank_drop_to_1",
    "near_degenerate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Gate6-A native local span artifacts and compatibility input from "
            "raw native triality triplets."
        )
    )
    parser.add_argument("--samples-root", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--sample-ids", nargs="+", type=int)
    selection.add_argument("--sample-id-file")
    selection.add_argument("--all-samples", action="store_true")
    parser.add_argument(
        "--variant",
        choices=("consistent", "frustrated", "unknown"),
        help="Optional variant filter applied after sample discovery.",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--perm-r", type=int, default=2000)
    parser.add_argument("--primary-score", default="E")
    parser.add_argument("--script-extract", default="tools/extract_triality_triplets.py")
    parser.add_argument("--script-eval", default="tools/eval_triality_token.py")
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
    )


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def current_git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def validate_upstream_metadata(sample_dirs: Sequence[Path]) -> Dict[str, Any]:
    keys = (
        "model_id",
        "model_revision",
        "seed",
        "proj_id",
        "splus_def_id",
        "sminus_def_id",
    )
    return packer.validate_homogeneous_metadata(sample_dirs, keys=keys)


def load_raw_vector(row: Dict[str, Any], key: str) -> np.ndarray:
    if key not in row:
        raise ValueError(f"missing raw vector key: {key}")
    values = np.asarray(row[key], dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"{key} must be 1-D")
    if values.size == 0:
        raise ValueError(f"{key} must be non-empty")
    if not np.isfinite(values).all():
        raise ValueError(f"{key} contains non-finite values")
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= TAU_NORM_ABS:
        raise ValueError(f"{key} norm is not usable: {norm}")
    return values


def normalize_raw_vector(values: np.ndarray) -> Tuple[np.ndarray, float]:
    norm = float(np.linalg.norm(values))
    return (values / norm, norm)


def sign_fix_column(column: np.ndarray, tau_sign_tie_abs: float = TAU_SIGN_TIE_ABS) -> Tuple[np.ndarray, bool, int]:
    abs_values = np.abs(column)
    max_abs = float(np.max(abs_values))
    candidates = np.flatnonzero(np.abs(abs_values - max_abs) <= tau_sign_tie_abs)
    anchor_index = int(candidates[0])
    if float(column[anchor_index]) < 0.0:
        return (-column, True, anchor_index)
    return (column.copy(), False, anchor_index)


def compact_flags(flags: Dict[str, bool]) -> str:
    active = [name for name in FLAG_FIELDS if flags.get(name)]
    if not active:
        return "none"
    return "|".join(active)


def build_local_object(
    v_raw: Sequence[float],
    splus_raw: Sequence[float],
    sminus_raw: Sequence[float],
) -> Dict[str, Any]:
    v_raw_array = load_raw_vector({"V_raw_native": v_raw}, "V_raw_native")
    splus_raw_array = load_raw_vector({"Splus_raw_native": splus_raw}, "Splus_raw_native")
    sminus_raw_array = load_raw_vector({"Sminus_raw_native": sminus_raw}, "Sminus_raw_native")

    v_unit, norm_v_raw = normalize_raw_vector(v_raw_array)
    splus_unit, norm_splus_raw = normalize_raw_vector(splus_raw_array)
    sminus_unit, norm_sminus_raw = normalize_raw_vector(sminus_raw_array)

    d_plus = splus_unit - v_unit
    d_minus = sminus_unit - v_unit
    construction = np.stack((v_unit, d_plus, d_minus), axis=1)

    # SVD gives the deterministic span; sign-fix removes the remaining gauge flip.
    u_matrix, singular_values, _vt = np.linalg.svd(construction, full_matrices=False)
    sigma_1 = float(singular_values[0]) if singular_values.size else 0.0
    rank_cutoff = max(TAU_RANK_ABS, TAU_RANK_REL * sigma_1)
    rank_local = int(np.sum(singular_values >= rank_cutoff))

    basis = np.zeros((construction.shape[0], LOCAL_DIM_MAX), dtype=np.float64)
    sign_anchor_indices = [-1, -1, -1]
    used_sign_fix = False
    for axis_idx in range(rank_local):
        fixed_column, flipped, anchor_index = sign_fix_column(u_matrix[:, axis_idx])
        basis[:, axis_idx] = fixed_column
        sign_anchor_indices[axis_idx] = anchor_index
        used_sign_fix = used_sign_fix or flipped

    observables = np.stack((v_unit, splus_unit, sminus_unit), axis=1)
    coords_local = np.zeros((LOCAL_DIM_MAX, 3), dtype=np.float64)
    if rank_local > 0:
        coords_local[:rank_local, :] = basis[:, :rank_local].T @ observables

    gram_raw = observables.T @ observables
    compat_local8 = np.zeros((3, COMPAT_DIM), dtype=np.float64)
    compat_local8[0, :LOCAL_DIM_MAX] = coords_local[:, 0]
    compat_local8[1, :LOCAL_DIM_MAX] = coords_local[:, 1]
    compat_local8[2, :LOCAL_DIM_MAX] = coords_local[:, 2]

    singular_values_padded = np.zeros(LOCAL_DIM_MAX, dtype=np.float64)
    singular_values_padded[: singular_values.shape[0]] = singular_values[:LOCAL_DIM_MAX]
    flags = {
        "all_finite": True,
        "used_sign_fix": used_sign_fix,
        "rank_drop_to_2": rank_local == 2,
        "rank_drop_to_1": rank_local == 1,
        "near_degenerate": bool(
            singular_values_padded[0] > 0.0
            and singular_values_padded[2] / singular_values_padded[0] < TAU_WARN_REL
        ),
    }

    return {
        "basis": basis,
        "projector_factor": basis.copy(),
        "coords_local": coords_local,
        "gram_raw": gram_raw,
        "singular_values": singular_values_padded,
        "rank_local": rank_local,
        "norms_raw": np.asarray(
            (norm_v_raw, norm_splus_raw, norm_sminus_raw), dtype=np.float64
        ),
        "flags": flags,
        "flags_vector": np.asarray([1 if flags[name] else 0 for name in FLAG_FIELDS], dtype=np.uint8),
        "compat_local8": compat_local8,
        "sign_anchor_indices": sign_anchor_indices,
        "reconstruction_v": basis[:, :rank_local] @ coords_local[:rank_local, 0],
        "reconstruction_splus": basis[:, :rank_local] @ coords_local[:rank_local, 1],
        "reconstruction_sminus": basis[:, :rank_local] @ coords_local[:rank_local, 2],
        "normalized_v": v_unit,
        "normalized_splus": splus_unit,
        "normalized_sminus": sminus_unit,
    }


def build_compat_step(
    row: Dict[str, Any],
    label_token: int,
    array_row_index: int,
    local_object: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "step": int(row["step"]),
        "absolute_pos": int(row["absolute_pos"]),
        "answer_char_start": row.get("answer_char_start"),
        "answer_char_end": row.get("answer_char_end"),
        "token_id": int(row["token_id"]),
        "token_text": str(row["token_str"]),
        "label_token": int(label_token),
        "defect_span_id": None,
        "baseline_logprob": float(row["baseline_logprob"]),
        "baseline_entropy": float(row["baseline_entropy"]),
        "rank_local": int(local_object["rank_local"]),
        "array_row_index": int(array_row_index),
        "flags_compact": compact_flags(local_object["flags"]),
        "boundary_origin": BOUNDARY_ORIGIN,
        "compat_vectors": {
            "V_local8": [float(x) for x in local_object["compat_local8"][0].tolist()],
            "Splus_local8": [float(x) for x in local_object["compat_local8"][1].tolist()],
            "Sminus_local8": [float(x) for x in local_object["compat_local8"][2].tolist()],
        },
    }


def build_step_index_row(
    sample_id: int,
    row: Dict[str, Any],
    label_token: int,
    array_row_index: int,
    local_object: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "sample_id": int(sample_id),
        "step": int(row["step"]),
        "token_text": str(row["token_str"]),
        "label_token": int(label_token),
        "baseline_logprob": float(row["baseline_logprob"]),
        "baseline_entropy": float(row["baseline_entropy"]),
        "offset_start": row.get("answer_char_start"),
        "offset_end": row.get("answer_char_end"),
        "array_row_index": int(array_row_index),
        "rank_local": int(local_object["rank_local"]),
        "flags_compact": compact_flags(local_object["flags"]),
    }


def build_sample_artifacts(
    sample_dir: Path,
    sample_id: int,
    start_array_index: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, List[np.ndarray]], Dict[str, Any]]:
    triplets_path = sample_dir / "triplets.ndjson"
    meta_path = sample_dir / "meta.json"
    labels_path = sample_dir / "labels.jsonl"
    labels_meta_path = sample_dir / "labels_meta.json"

    triplets = packer.load_jsonl(triplets_path)
    if not triplets:
        raise ValueError(f"empty triplets: {triplets_path}")
    triplets.sort(key=lambda row: int(row["step"]))
    expected_steps = list(range(len(triplets)))
    actual_steps = [int(row["step"]) for row in triplets]
    if actual_steps != expected_steps:
        raise ValueError(
            f"triplets steps for sample {sample_id} are not contiguous 0..N-1: {actual_steps[:16]}"
        )

    meta = packer.load_json(meta_path)
    labels_meta = packer.load_json(labels_meta_path)
    labels = packer.load_step_labels(labels_path, n_steps=len(triplets))
    triplets_file_sha256 = packer.sha256_file(triplets_path)
    meta_triplets_sha256 = str(meta.get("output_ndjson_sha256") or "")
    if triplets_file_sha256 != meta_triplets_sha256:
        raise ValueError(
            f"triplets SHA mismatch for sample {sample_id}: "
            f"file={triplets_file_sha256} meta={meta_triplets_sha256}"
        )

    native_schema_id = meta.get("native_raw_schema_id")
    if native_schema_id not in (RAW_NATIVE_SCHEMA_ID, None):
        raise ValueError(
            f"unsupported native raw schema for sample {sample_id}: {native_schema_id!r}"
        )
    if not all(all(key in row for key in RAW_KEYS) for row in triplets):
        raise ValueError(
            f"sample {sample_id} triplets do not contain raw native vectors; "
            "rerun extraction with --emit-native-raw"
        )

    arrays: Dict[str, List[np.ndarray]] = {
        "basis": [],
        "projector_factor": [],
        "coords_local": [],
        "gram_raw": [],
        "singular_values": [],
        "rank_local": [],
        "norms_raw": [],
        "flags": [],
        "compat_local8": [],
    }
    compat_steps: List[Dict[str, Any]] = []
    step_index_rows: List[Dict[str, Any]] = []
    rank_counts = {1: 0, 2: 0, 3: 0}
    near_degenerate_count = 0

    for local_offset, row in enumerate(triplets):
        array_row_index = start_array_index + local_offset
        local_object = build_local_object(
            v_raw=row["V_raw_native"],
            splus_raw=row["Splus_raw_native"],
            sminus_raw=row["Sminus_raw_native"],
        )
        for key in arrays:
            if key == "rank_local":
                arrays[key].append(np.asarray(local_object[key], dtype=np.int64))
            elif key == "flags":
                arrays[key].append(np.asarray(local_object["flags_vector"], dtype=np.uint8))
            else:
                arrays[key].append(np.asarray(local_object[key]))
        compat_steps.append(
            build_compat_step(
                row=row,
                label_token=labels[local_offset],
                array_row_index=array_row_index,
                local_object=local_object,
            )
        )
        step_index_rows.append(
            build_step_index_row(
                sample_id=sample_id,
                row=row,
                label_token=labels[local_offset],
                array_row_index=array_row_index,
                local_object=local_object,
            )
        )
        rank_counts[int(local_object["rank_local"])] += 1
        if local_object["flags"]["near_degenerate"]:
            near_degenerate_count += 1

    sample_payload = {
        "sample_id": int(sample_id),
        "variant": str(labels_meta.get("variant") or "unknown"),
        "world_type": labels_meta.get("world_type"),
        "exact_token_match_ratio": float(meta["exact_token_match_ratio"]),
        "label_coverage_ratio": float(labels_meta["final_alignment_coverage_ratio"]),
        "triplets_sha256": triplets_file_sha256,
        "labels_sha256": packer.sha256_file(labels_path),
        "token_steps": compat_steps,
    }
    summary = {
        "triplets_sha256": triplets_file_sha256,
        "labels_sha256": packer.sha256_file(labels_path),
        "n_token_steps": len(triplets),
        "rank_counts": rank_counts,
        "near_degenerate_count": near_degenerate_count,
    }
    return (sample_payload, step_index_rows, arrays, summary)


def stack_array_rows(rows: Sequence[np.ndarray], dtype: Any) -> np.ndarray:
    if not rows:
        raise ValueError("cannot stack empty array rows")
    return np.stack([np.asarray(row, dtype=dtype) for row in rows], axis=0)


def build_compatibility_metadata(
    first_meta: Dict[str, Any],
    script_extract: Path,
    script_eval: Path,
    builder_script: Path,
    perm_r: int,
    primary_score: str,
) -> Dict[str, Any]:
    return {
        "model_id": str(first_meta["model_id"]),
        "model_revision": str(first_meta.get("model_revision") or ""),
        "seed": int(first_meta["seed"]),
        "perm_r": int(perm_r),
        "primary_score": str(primary_score),
        "proj_id": PROJ_ID,
        "splus_def_id": SPLUS_DEF_ID,
        "sminus_def_id": SMINUS_DEF_ID,
        "script_sha256_extract": packer.sha256_file(script_extract),
        "script_sha256_eval": packer.sha256_file(script_eval),
        "script_sha256_gate6_builder": packer.sha256_file(builder_script),
        "boundary_origin": BOUNDARY_ORIGIN,
        "compatibility_schema_id": COMPAT_INPUT_SCHEMA_ID,
        "local_object_method_id": METHOD_ID,
        "source_tensor_id": SOURCE_TENSOR_ID,
    }


def build_input_sha256(sample_ids: Sequence[int], sample_summaries: Sequence[Dict[str, Any]]) -> str:
    payload = [
        {
            "sample_id": int(sample_id),
            "triplets_sha256": str(summary["triplets_sha256"]),
            "labels_sha256": str(summary["labels_sha256"]),
        }
        for sample_id, summary in zip(sample_ids, sample_summaries)
    ]
    return sha256_bytes(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def count_rows_by_rank(rank_values: Iterable[int]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for rank in rank_values:
        key = str(int(rank))
        counts[key] = counts.get(key, 0) + 1
    return counts


def build_boundary_outcome_counts(rank_local_counts: Dict[str, int]) -> Dict[str, int]:
    return {
        "materialized_rank3": int(rank_local_counts.get("3", 0)),
        "materialized_rank2": int(rank_local_counts.get("2", 0)),
        "materialized_rank1": int(rank_local_counts.get("1", 0)),
        "sign_unstable": 0,
        "raw_span_axis_collapse": 0,
    }


def build_manifest(
    samples_root: Path,
    sample_ids: Sequence[int],
    first_meta: Dict[str, Any],
    sample_summaries: Sequence[Dict[str, Any]],
    arrays: Dict[str, np.ndarray],
    script_extract: Path,
    script_eval: Path,
    builder_script: Path,
    out_dir: Path,
) -> Dict[str, Any]:
    flags = arrays["flags"]
    near_degenerate_count = int(flags[:, FLAG_FIELDS.index("near_degenerate")].sum())
    rank_local_counts = count_rows_by_rank(arrays["rank_local"].tolist())
    return {
        "run_id": out_dir.name,
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "source_model_id": str(first_meta["model_id"]),
        "source_model_revision": str(first_meta.get("model_revision") or ""),
        "source_layer_id": str(first_meta.get("layer_id") or "unknown"),
        "input_source_path": repo_relative_or_posix(samples_root),
        "input_sha256": build_input_sha256(sample_ids, sample_summaries),
        "code_git_commit": current_git_commit(),
        "builder_script_sha256": packer.sha256_file(builder_script),
        "tau_norm_abs": TAU_NORM_ABS,
        "tau_rank_abs": TAU_RANK_ABS,
        "tau_rank_rel": TAU_RANK_REL,
        "tau_sign_tie_abs": TAU_SIGN_TIE_ABS,
        "normalization_mode": "float64_l2_unit_v1",
        "construction_mode": CONSTRUCTION_MODE,
        "sign_fix_mode": SIGN_FIX_MODE,
        "compatibility_embedding": COMPATIBILITY_EMBEDDING,
        "n_samples_total": len(sample_ids),
        "n_token_steps_total": int(arrays["rank_local"].shape[0]),
        "sample_ids": [int(sample_id) for sample_id in sample_ids],
        "samples_root": repo_relative_or_posix(samples_root),
        "source_tensor_id": SOURCE_TENSOR_ID,
        "proj_id": PROJ_ID,
        "splus_def_id": SPLUS_DEF_ID,
        "sminus_def_id": SMINUS_DEF_ID,
        "seed": int(first_meta["seed"]),
        "perm_r": int(first_meta.get("perm_r") or 2000),
        "rank_local_counts": rank_local_counts,
        "boundary_outcome_counts": build_boundary_outcome_counts(rank_local_counts),
        "near_degenerate_count": near_degenerate_count,
        "flag_fields": list(FLAG_FIELDS),
        "script_sha256_extract": packer.sha256_file(script_extract),
        "script_sha256_eval": packer.sha256_file(script_eval),
    }


def write_npz(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        basis=arrays["basis"],
        projector_factor=arrays["projector_factor"],
        coords_local=arrays["coords_local"],
        gram_raw=arrays["gram_raw"],
        singular_values=arrays["singular_values"],
        rank_local=arrays["rank_local"],
        norms_raw=arrays["norms_raw"],
        flags=arrays["flags"],
        compat_local8=arrays["compat_local8"],
    )


def write_checksums(path: Path, artifact_paths: Sequence[Tuple[str, Path]]) -> None:
    payload = {
        name: {
            "path": repo_relative_or_posix(artifact_path),
            "sha256": packer.sha256_file(artifact_path),
        }
        for name, artifact_path in artifact_paths
    }
    write_json(path, payload)


def main() -> int:
    args = parse_args()
    samples_root = (REPO_ROOT / args.samples_root).resolve()
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    manifest_path = out_dir / DEFAULT_MANIFEST
    step_index_path = out_dir / DEFAULT_STEP_INDEX
    arrays_path = out_dir / DEFAULT_ARRAYS
    compatibility_input_path = out_dir / DEFAULT_COMPAT_INPUT
    checksums_path = out_dir / DEFAULT_CHECKSUMS

    selected_ids = packer.resolve_selected_sample_ids(
        samples_root=samples_root,
        sample_ids=args.sample_ids,
        sample_id_file=Path(args.sample_id_file) if args.sample_id_file else None,
        all_samples=bool(args.all_samples),
        variant=args.variant,
        offset=args.offset,
        limit=args.limit,
    )
    sample_dirs = [samples_root / packer.sample_dir_name(sample_id) for sample_id in selected_ids]
    first_meta = validate_upstream_metadata(sample_dirs)

    compatibility_samples: List[Dict[str, Any]] = []
    step_index_rows: List[Dict[str, Any]] = []
    sample_summaries: List[Dict[str, Any]] = []
    array_lists: Dict[str, List[np.ndarray]] = {
        "basis": [],
        "projector_factor": [],
        "coords_local": [],
        "gram_raw": [],
        "singular_values": [],
        "rank_local": [],
        "norms_raw": [],
        "flags": [],
        "compat_local8": [],
    }

    next_array_index = 0
    for sample_dir, sample_id in zip(sample_dirs, selected_ids):
        sample_payload, sample_step_index, sample_arrays, sample_summary = build_sample_artifacts(
            sample_dir=sample_dir,
            sample_id=sample_id,
            start_array_index=next_array_index,
        )
        compatibility_samples.append(sample_payload)
        step_index_rows.extend(sample_step_index)
        sample_summaries.append(sample_summary)
        for key in array_lists:
            array_lists[key].extend(sample_arrays[key])
        next_array_index += sample_summary["n_token_steps"]

    arrays = {
        "basis": stack_array_rows(array_lists["basis"], np.float64),
        "projector_factor": stack_array_rows(array_lists["projector_factor"], np.float64),
        "coords_local": stack_array_rows(array_lists["coords_local"], np.float64),
        "gram_raw": stack_array_rows(array_lists["gram_raw"], np.float64),
        "singular_values": stack_array_rows(array_lists["singular_values"], np.float64),
        "rank_local": stack_array_rows(array_lists["rank_local"], np.int64).reshape(-1),
        "norms_raw": stack_array_rows(array_lists["norms_raw"], np.float64),
        "flags": stack_array_rows(array_lists["flags"], np.uint8),
        "compat_local8": stack_array_rows(array_lists["compat_local8"], np.float64),
    }

    builder_script = Path(__file__).resolve()
    compatibility_payload = {
        "metadata": build_compatibility_metadata(
            first_meta=first_meta,
            script_extract=(REPO_ROOT / args.script_extract).resolve(),
            script_eval=(REPO_ROOT / args.script_eval).resolve(),
            builder_script=builder_script,
            perm_r=args.perm_r,
            primary_score=args.primary_score,
        ),
        "samples": compatibility_samples,
    }
    manifest = build_manifest(
        samples_root=samples_root,
        sample_ids=selected_ids,
        first_meta=first_meta,
        sample_summaries=sample_summaries,
        arrays=arrays,
        script_extract=(REPO_ROOT / args.script_extract).resolve(),
        script_eval=(REPO_ROOT / args.script_eval).resolve(),
        builder_script=builder_script,
        out_dir=out_dir,
    )

    write_json(manifest_path, manifest)
    write_jsonl(step_index_path, step_index_rows)
    write_npz(arrays_path, arrays)
    write_json(compatibility_input_path, compatibility_payload)
    write_checksums(
        checksums_path,
        (
            ("manifest_json", manifest_path),
            ("step_index_jsonl", step_index_path),
            ("native_object_arrays_npz", arrays_path),
            ("compatibility_input_json", compatibility_input_path),
        ),
    )

    print(f"manifest_json={repo_relative_or_posix(manifest_path)}")
    print(f"step_index_jsonl={repo_relative_or_posix(step_index_path)}")
    print(f"native_object_arrays_npz={repo_relative_or_posix(arrays_path)}")
    print(f"compatibility_input_json={repo_relative_or_posix(compatibility_input_path)}")
    print(f"checksums_json={repo_relative_or_posix(checksums_path)}")
    print(f"n_samples_total={len(selected_ids)}")
    print(f"n_token_steps_total={int(arrays['rank_local'].shape[0])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
