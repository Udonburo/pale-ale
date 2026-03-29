#!/usr/bin/env python3
"""Export a Gate12A node-local-object family from recovered Gate8 / Gate9A context."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

import run_gate9a_graph_gauge_consumer as gate9a


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()

SCHEMA_VERSION = "gate12a_node_local_object_family_v1"
METHOD_ID = "gate12a_node_local_object_export_v1"

DEFAULT_MANIFEST = "manifest.json"
DEFAULT_NODE_REGISTRY = "node_local_object_registry.jsonl"
DEFAULT_NODE_ARRAYS = "node_local_object_arrays.npz"
DEFAULT_CHECKSUMS = "checksums.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the Gate12A node-local-object artifact family from a recovered Gate8 "
            "execution directory using the same local-object reconstruction discipline as Gate9A."
        )
    )
    parser.add_argument("--gate8-execution-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n")


def build_node_rows(source_dir: Path) -> List[gate9a.LocalObject]:
    source_manifest = gate9a.read_json(source_dir / "manifest.json")
    _ = source_manifest
    sample_registry_rows = gate9a.read_jsonl(source_dir / "sample_registry.jsonl")
    gate6_dir = source_dir / "gate6_native"
    step_rows, arrays = gate9a.load_gate6_arrays(gate6_dir)
    registry_by_execution_id = {
        int(row["execution_sample_id"]): row for row in sample_registry_rows
    }
    sample_steps: Dict[int, List[Dict[str, Any]]] = {}
    for row in step_rows:
        sample_steps.setdefault(int(row["sample_id"]), []).append(row)
    for rows in sample_steps.values():
        rows.sort(key=lambda row: int(row["step"]))

    local_objects: List[gate9a.LocalObject] = []
    for execution_sample_id in sorted(registry_by_execution_id):
        registry_row = registry_by_execution_id[execution_sample_id]
        sample_dir = source_dir / "samples" / f"sample_{execution_sample_id:06d}"
        token_step_rows = sample_steps.get(execution_sample_id, [])

        for step_row in token_step_rows:
            local_objects.append(gate9a.build_token_local_object(registry_row, step_row, arrays))

        answer_triplet_rows = gate9a.read_jsonl(sample_dir / "triplets.ndjson")
        local_objects.append(
            gate9a.build_anchor_local_object(
                registry_row=registry_row,
                node_type="answer_state",
                node_suffix="answer_state",
                triplet_rows=answer_triplet_rows,
                extra_meta={"source_triplets": "triplets.ndjson"},
            )
        )

        support_triplet_rows = gate9a.read_jsonl_if_exists(sample_dir / "support_anchor_triplets.ndjson")
        if support_triplet_rows is not None:
            local_objects.append(
                gate9a.build_anchor_local_object(
                    registry_row=registry_row,
                    node_type="support_chunk",
                    node_suffix="support_chunk",
                    triplet_rows=support_triplet_rows,
                    extra_meta={"source_triplets": "support_anchor_triplets.ndjson"},
                )
            )

        conflict_triplet_rows = gate9a.read_jsonl_if_exists(sample_dir / "conflict_anchor_triplets.ndjson")
        if conflict_triplet_rows is not None:
            local_objects.append(
                gate9a.build_anchor_local_object(
                    registry_row=registry_row,
                    node_type="conflict_chunk",
                    node_suffix="conflict_chunk",
                    triplet_rows=conflict_triplet_rows,
                    extra_meta={"source_triplets": "conflict_anchor_triplets.ndjson"},
                )
            )
    return local_objects


def normalized_node_label(local_object: gate9a.LocalObject) -> str:
    return f"{local_object.node_type}:{local_object.execution_sample_id}"


def export_node_family(source_dir: Path, out_dir: Path) -> Dict[str, Any]:
    source_dir = Path(source_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = gate9a.read_json(source_dir / "manifest.json")
    local_objects = sorted(build_node_rows(source_dir), key=lambda obj: obj.node_id)
    if not local_objects:
        raise ValueError("no local objects were reconstructed from the source execution")

    basis_factor = np.asarray([np.asarray(obj.basis, dtype=np.float64) for obj in local_objects], dtype=np.float64)
    rank_active = np.asarray([int(obj.rank_local) for obj in local_objects], dtype=np.int64)

    node_rows = [
        {
            "node_id": obj.node_id,
            "node_label": normalized_node_label(obj),
            "basis_array_index": index,
            "projector_rank": int(obj.rank_local),
            "local_object_status": "defined",
        }
        for index, obj in enumerate(local_objects)
    ]

    manifest_path = out_dir / DEFAULT_MANIFEST
    registry_path = out_dir / DEFAULT_NODE_REGISTRY
    arrays_path = out_dir / DEFAULT_NODE_ARRAYS
    checksums_path = out_dir / DEFAULT_CHECKSUMS

    write_jsonl(registry_path, node_rows)
    np.savez(arrays_path, basis_factor=basis_factor, rank_active=rank_active)

    manifest = {
        "run_id": out_dir.name,
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "code_git_commit": gate9a.current_git_commit(),
        "builder_script_sha256": sha256_file(SCRIPT_PATH),
        "source_gate8_execution_dir": gate9a.repo_relative_or_posix(source_dir),
        "source_gate8_run_id": str(source_manifest.get("run_id") or ""),
        "source_gate8_code_git_commit": str(source_manifest.get("code_git_commit") or ""),
        "source_rendering_family_id": str(source_manifest.get("rendering_family_id") or ""),
        "source_local_object_discipline": "gate9a_reconstruction_v1",
        "n_nodes_total": len(node_rows),
        "paths": {
            DEFAULT_NODE_REGISTRY: gate9a.repo_relative_or_posix(registry_path),
            DEFAULT_NODE_ARRAYS: gate9a.repo_relative_or_posix(arrays_path),
        },
    }
    write_json(manifest_path, manifest)
    write_json(
        checksums_path,
        {
            DEFAULT_MANIFEST: sha256_file(manifest_path),
            DEFAULT_NODE_REGISTRY: sha256_file(registry_path),
            DEFAULT_NODE_ARRAYS: sha256_file(arrays_path),
        },
    )
    return {
        "manifest": manifest,
        "node_rows": node_rows,
    }


def main() -> int:
    args = parse_args()
    export_node_family(
        source_dir=Path(args.gate8_execution_dir),
        out_dir=Path(args.out_dir),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
