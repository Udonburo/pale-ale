#!/usr/bin/env python3
"""Generate a deterministic Gate8 semi-closed conflict benchmark scaffold."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]

SCAFFOLD_SCHEMA_VERSION = "gate8_semiclosed_conflict_skeleton_v1"
TAXONOMY_SCHEMA_VERSION = "gate8_conflict_taxonomy_v1"
LABEL_CONTRACT_VERSION = "gate8_label_contract_v1"
WORLD_PLAN_SCHEMA_VERSION = "gate8_world_plan_placeholder_v1"
RENDERING_PLAN_SCHEMA_VERSION = "gate8_rendering_plan_placeholder_v1"
TARGET_PLAN_SCHEMA_VERSION = "gate8_target_plan_placeholder_v1"
METHOD_ID = "gate8_semiclosed_conflict_skeleton_v1"
GENERATION_STAGE = "constitution_scaffold"

DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CONFLICT_PLAN = "conflict_plan.json"
DEFAULT_LABEL_CONTRACT = "label_contract.json"
DEFAULT_WORLD_PLAN = "world_plan.json"
DEFAULT_RENDERING_PLAN = "rendering_plan.json"
DEFAULT_TARGET_PLAN = "target_plan.json"
DEFAULT_SAMPLE_INDEX = "sample_index.jsonl"
DEFAULT_CHECKSUMS = "checksums.json"


CELL_DEFS: Sequence[Dict[str, Any]] = (
    {
        "cell_id": "clean_support",
        "is_conflict_intended": False,
        "is_surface_noise_only": False,
        "default_answer_target_types": ["consistent_answer"],
        "description": "Retrieval is semantically clean and mutually supportive.",
    },
    {
        "cell_id": "direct_contradiction",
        "is_conflict_intended": True,
        "is_surface_noise_only": False,
        "default_answer_target_types": ["consistent_answer", "conflict_following_wrong_answer"],
        "description": "A retrieval chunk explicitly contradicts world truth or the dominant support set.",
    },
    {
        "cell_id": "distributed_incompatibility",
        "is_conflict_intended": True,
        "is_surface_noise_only": False,
        "default_answer_target_types": ["consistent_answer", "unsupported_bridge_answer"],
        "description": "Conflict emerges only when multiple retrieval chunks are glued together.",
    },
    {
        "cell_id": "surface_noisy_clean",
        "is_conflict_intended": False,
        "is_surface_noise_only": True,
        "default_answer_target_types": ["consistent_answer"],
        "description": "Surface is noisy but semantics remain aligned with world truth.",
    },
)

CANDIDATE_SET: Sequence[Dict[str, str]] = (
    {"role": "legacy_guardrail", "metric_id": "score_F_gram_loop_v1"},
    {"role": "operational_candidate", "metric_id": "sigma_gap_tailkeep_weighted_gram_loop_v2"},
    {"role": "research_north_star", "metric_id": "sigma_sqrtgap_tailkeep_object_v2"},
    {"role": "dynamic_candidate", "metric_id": "progression_anisotropic_closure_v3"},
)

HEADLINE_METRICS: Sequence[str] = (
    "global_auprc",
    "mean_sample_auprc",
    "hit@10",
    "first_hit_distance",
    "mean_delta_max",
    "mean_delta_p90",
    "mean_iqr_normalized_delta_max",
    "mean_top10_inflation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a deterministic Gate8 benchmark scaffold that freezes conflict taxonomy, "
            "candidate set, and label/provenance contract before full data generation."
        )
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--samples-per-cell", type=int, default=8)
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


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_git_commit() -> str:
    head_path = REPO_ROOT / ".git" / "HEAD"
    if not head_path.exists():
        return ""
    head = head_path.read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    ref = head[5:]
    ref_path = REPO_ROOT / ".git" / ref
    if ref_path.exists():
        return ref_path.read_text(encoding="utf-8").strip()
    return ""


def build_conflict_plan(samples_per_cell: int) -> Dict[str, Any]:
    return {
        "schema_version": TAXONOMY_SCHEMA_VERSION,
        "samples_per_cell": samples_per_cell,
        "cells": list(CELL_DEFS),
        "candidate_set": list(CANDIDATE_SET),
        "headline_metrics": list(HEADLINE_METRICS),
        "aggregation_ban": True,
        "purpose": "standing preservation under more natural conflict geometry",
    }


def build_label_contract() -> Dict[str, Any]:
    return {
        "schema_version": LABEL_CONTRACT_VERSION,
        "required_sample_fields": [
            "sample_id",
            "cell_id",
            "world_id",
            "rendering_id",
            "target_id",
            "answer_target_type",
            "is_conflict_intended",
            "is_surface_noise_only",
            "retrieval_chunk_ids",
            "retrieval_conflict_chunk_ids",
            "retrieval_support_chunk_ids",
        ],
        "required_label_fields": [
            "label_token",
            "label_span_conflict",
            "label_span_support",
            "label_span_defect",
        ],
        "layer_separation": [
            "world truth",
            "retrieval rendering",
            "answer target",
            "defect span labeling",
        ],
        "candidate_freeze": list(CANDIDATE_SET),
    }


def build_world_plan(samples_per_cell: int) -> Dict[str, Any]:
    return {
        "schema_version": WORLD_PLAN_SCHEMA_VERSION,
        "stage": GENERATION_STAGE,
        "binding_status": "constitution_only_placeholder",
        "world_generation_mode": "not_materialized",
        "samples_per_cell": samples_per_cell,
        "cell_ids": [str(cell["cell_id"]) for cell in CELL_DEFS],
        "notes": (
            "Placeholder world specification plan for constitution-stage provenance binding. "
            "Concrete world truth artifacts are emitted only in later benchmark generation stages."
        ),
    }


def build_rendering_plan(samples_per_cell: int) -> Dict[str, Any]:
    return {
        "schema_version": RENDERING_PLAN_SCHEMA_VERSION,
        "stage": GENERATION_STAGE,
        "binding_status": "constitution_only_placeholder",
        "rendering_generation_mode": "not_materialized",
        "samples_per_cell": samples_per_cell,
        "cell_ids": [str(cell["cell_id"]) for cell in CELL_DEFS],
        "notes": (
            "Placeholder retrieval rendering plan for constitution-stage provenance binding. "
            "Final retrieval passages and chunk layouts are emitted only in later stages."
        ),
    }


def build_target_plan(samples_per_cell: int) -> Dict[str, Any]:
    answer_target_types = sorted(
        {
            str(target_type)
            for cell in CELL_DEFS
            for target_type in cell["default_answer_target_types"]
        }
    )
    return {
        "schema_version": TARGET_PLAN_SCHEMA_VERSION,
        "stage": GENERATION_STAGE,
        "binding_status": "constitution_only_placeholder",
        "target_generation_mode": "not_materialized",
        "samples_per_cell": samples_per_cell,
        "answer_target_types": answer_target_types,
        "notes": (
            "Placeholder answer target plan for constitution-stage provenance binding. "
            "Final answer strings and span targets are emitted only in later stages."
        ),
    }


def build_sample_index(samples_per_cell: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    sample_counter = 0
    for cell in CELL_DEFS:
        cell_id = str(cell["cell_id"])
        target_types = list(cell["default_answer_target_types"])
        for i in range(samples_per_cell):
            answer_target_type = target_types[i % len(target_types)]
            sample_counter += 1
            rows.append(
                {
                    "sample_id": f"gate8_plan_{sample_counter:05d}",
                    "cell_id": cell_id,
                    "world_id": f"{cell_id}_world_{i:03d}",
                    "rendering_id": f"{cell_id}_render_{i:03d}",
                    "target_id": f"{cell_id}_{answer_target_type}_{i:03d}",
                    "answer_target_type": answer_target_type,
                    "is_conflict_intended": bool(cell["is_conflict_intended"]),
                    "is_surface_noise_only": bool(cell["is_surface_noise_only"]),
                    "retrieval_chunk_ids": [],
                    "retrieval_conflict_chunk_ids": [],
                    "retrieval_support_chunk_ids": [],
                    "status": "planned",
                }
            )
    return rows


def build_manifest(
    run_id: str,
    samples_per_cell: int,
    n_samples_total: int,
    world_plan_path: Path,
    rendering_plan_path: Path,
    target_plan_path: Path,
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "schema_version": SCAFFOLD_SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "generation_stage": GENERATION_STAGE,
        "provenance_binding_mode": "constitution_only_placeholders",
        "samples_per_cell": samples_per_cell,
        "n_cells_total": len(CELL_DEFS),
        "n_samples_total": n_samples_total,
        "candidate_set": list(CANDIDATE_SET),
        "headline_metrics": list(HEADLINE_METRICS),
        "aggregation_ban": True,
        "semi_closed_layers": [
            "world truth",
            "retrieval rendering",
            "answer target",
            "defect span labeling",
        ],
        "taxonomy_schema_version": TAXONOMY_SCHEMA_VERSION,
        "label_contract_version": LABEL_CONTRACT_VERSION,
        "world_plan_schema_version": WORLD_PLAN_SCHEMA_VERSION,
        "rendering_plan_schema_version": RENDERING_PLAN_SCHEMA_VERSION,
        "target_plan_schema_version": TARGET_PLAN_SCHEMA_VERSION,
        "code_git_commit": current_git_commit(),
        "generator_script_path": repo_relative_or_posix(Path(__file__)),
        "generator_script_sha256": sha256_file(Path(__file__)),
        "world_plan_path": world_plan_path.name,
        "world_plan_sha256": sha256_file(world_plan_path),
        "rendering_plan_path": rendering_plan_path.name,
        "rendering_plan_sha256": sha256_file(rendering_plan_path),
        "target_plan_path": target_plan_path.name,
        "target_plan_sha256": sha256_file(target_plan_path),
    }


def build_checksums(entries: Sequence[Tuple[str, Path]]) -> Dict[str, str]:
    return {name: sha256_file(path) for name, path in entries}


def main() -> int:
    args = parse_args()
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    run_id = args.run_id or out_dir.name
    samples_per_cell = int(args.samples_per_cell)
    if samples_per_cell < 1:
        raise SystemExit("--samples-per-cell must be >= 1")

    manifest_path = out_dir / DEFAULT_MANIFEST
    conflict_plan_path = out_dir / DEFAULT_CONFLICT_PLAN
    label_contract_path = out_dir / DEFAULT_LABEL_CONTRACT
    world_plan_path = out_dir / DEFAULT_WORLD_PLAN
    rendering_plan_path = out_dir / DEFAULT_RENDERING_PLAN
    target_plan_path = out_dir / DEFAULT_TARGET_PLAN
    sample_index_path = out_dir / DEFAULT_SAMPLE_INDEX
    checksums_path = out_dir / DEFAULT_CHECKSUMS

    conflict_plan = build_conflict_plan(samples_per_cell)
    label_contract = build_label_contract()
    world_plan = build_world_plan(samples_per_cell)
    rendering_plan = build_rendering_plan(samples_per_cell)
    target_plan = build_target_plan(samples_per_cell)
    sample_rows = build_sample_index(samples_per_cell)
    write_json(conflict_plan_path, conflict_plan)
    write_json(label_contract_path, label_contract)
    write_json(world_plan_path, world_plan)
    write_json(rendering_plan_path, rendering_plan)
    write_json(target_plan_path, target_plan)
    manifest = build_manifest(
        run_id,
        samples_per_cell,
        len(sample_rows),
        world_plan_path,
        rendering_plan_path,
        target_plan_path,
    )

    write_json(manifest_path, manifest)
    write_jsonl(sample_index_path, sample_rows)
    write_json(
        checksums_path,
        build_checksums(
            (
                ("manifest_json", manifest_path),
                ("conflict_plan_json", conflict_plan_path),
                ("label_contract_json", label_contract_path),
                ("world_plan_json", world_plan_path),
                ("rendering_plan_json", rendering_plan_path),
                ("target_plan_json", target_plan_path),
                ("sample_index_jsonl", sample_index_path),
            )
        ),
    )

    print(f"manifest_json={repo_relative_or_posix(manifest_path)}")
    print(f"conflict_plan_json={repo_relative_or_posix(conflict_plan_path)}")
    print(f"label_contract_json={repo_relative_or_posix(label_contract_path)}")
    print(f"world_plan_json={repo_relative_or_posix(world_plan_path)}")
    print(f"rendering_plan_json={repo_relative_or_posix(rendering_plan_path)}")
    print(f"target_plan_json={repo_relative_or_posix(target_plan_path)}")
    print(f"sample_index_jsonl={repo_relative_or_posix(sample_index_path)}")
    print(f"checksums_json={repo_relative_or_posix(checksums_path)}")
    print(f"n_samples_total={len(sample_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
