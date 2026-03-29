#!/usr/bin/env python3
"""Export a Gate12A explicit relation-seed family from Gate9K / Gate9A context."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import run_gate9a_graph_gauge_consumer as gate9a


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()

SCHEMA_VERSION = "gate12a_relation_seed_family_v1"
METHOD_ID = "gate12a_relation_seed_export_v1"
RELATION_SEED_MODE = "explicit_edge_seed_v1"

DEFAULT_MANIFEST = "manifest.json"
DEFAULT_REGISTRY = "explicit_relation_seed_registry.jsonl"
DEFAULT_CHECKSUMS = "checksums.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the Gate12A explicit relation-seed family from a Gate9K trusted-tree / "
            "residual-chord logging run."
        )
    )
    parser.add_argument("--gate9k-dir", required=True)
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


def relation_kind_from_role(decomposition_role: str) -> str | None:
    if decomposition_role == "trusted_tree_candidate":
        return "trusted_tree"
    if decomposition_role == "residual_chord_candidate":
        return "residual_chord"
    return None


def export_relation_seed_family(source_dir: Path, out_dir: Path) -> Dict[str, Any]:
    source_dir = Path(source_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = gate9a.read_json(source_dir / "manifest.json")
    source_gate9a_dir = REPO_ROOT / str(source_manifest["source_gate9a_dir"])
    gate9a_manifest = gate9a.read_json(source_gate9a_dir / gate9a.DEFAULT_MANIFEST)
    gate9a_edge_rows = gate9a.read_jsonl(source_gate9a_dir / gate9a.DEFAULT_EDGE_REGISTRY)
    gate9a_edge_map = {str(row["edge_id"]): row for row in gate9a_edge_rows}
    gate9k_registry_rows = gate9a.read_jsonl(source_dir / "trusted_tree_residual_chord_registry.jsonl")

    relation_rows = []
    for row in sorted(gate9k_registry_rows, key=lambda item: str(item.get("edge_id") or "")):
        relation_kind = relation_kind_from_role(str(row.get("decomposition_role") or ""))
        if relation_kind is None:
            continue
        edge_id = str(row["edge_id"])
        gate9a_row = gate9a_edge_map.get(edge_id)
        if gate9a_row is None:
            raise ValueError(f"Gate9K row references missing Gate9A edge: {edge_id}")
        edge_type = str(gate9a_row["edge_type"])
        anchor_qualified = edge_type in {"support_anchor", "conflict_anchor"}
        relation_rows.append(
            {
                "edge_id": edge_id,
                "source_node_id": str(gate9a_row["source_node_id"]),
                "target_node_id": str(gate9a_row["target_node_id"]),
                "relation_kind": relation_kind,
                "anchor_qualified": anchor_qualified,
                "anchor_relation_id": f"anchor:{edge_id}" if anchor_qualified else "",
            }
        )

    manifest_path = out_dir / DEFAULT_MANIFEST
    registry_path = out_dir / DEFAULT_REGISTRY
    checksums_path = out_dir / DEFAULT_CHECKSUMS

    write_jsonl(registry_path, relation_rows)
    manifest = {
        "run_id": out_dir.name,
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "relation_seed_mode": RELATION_SEED_MODE,
        "code_git_commit": gate9a.current_git_commit(),
        "builder_script_sha256": sha256_file(SCRIPT_PATH),
        "source_gate9k_dir": gate9a.repo_relative_or_posix(source_dir),
        "source_gate9k_run_id": str(source_manifest.get("run_id") or ""),
        "source_gate9k_code_git_commit": str(source_manifest.get("code_git_commit") or ""),
        "source_gate9a_dir": gate9a.repo_relative_or_posix(source_gate9a_dir),
        "source_gate9a_run_id": str(gate9a_manifest.get("run_id") or ""),
        "n_relation_rows_total": len(relation_rows),
        "paths": {
            DEFAULT_REGISTRY: gate9a.repo_relative_or_posix(registry_path),
        },
    }
    write_json(manifest_path, manifest)
    write_json(
        checksums_path,
        {
            DEFAULT_MANIFEST: sha256_file(manifest_path),
            DEFAULT_REGISTRY: sha256_file(registry_path),
        },
    )
    return {
        "manifest": manifest,
        "relation_rows": relation_rows,
    }


def main() -> int:
    args = parse_args()
    export_relation_seed_family(
        source_dir=Path(args.gate9k_dir),
        out_dir=Path(args.out_dir),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
