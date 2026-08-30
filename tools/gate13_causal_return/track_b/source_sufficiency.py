"""Pre-outcome B2a split-half source-sufficiency gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.gate13_causal_return.phase2_common import read_json, sha256_file, write_json


SOURCE_ARTIFACTS = {
    "manifest_sha256": "manifest.json",
    "node_registry_sha256": "node_local_object_registry.jsonl",
    "node_artifact_sha256": "node_local_object_arrays.npz",
    "edge_artifact_sha256": "transport_relation_registry.jsonl",
    "triangle_registry_sha256": "explicit_triangle_cycle_registry.jsonl",
    "holonomy_registry_sha256": "triangle_holonomy_registry.jsonl",
    "operator_array_sha256": "transport_operator_arrays.npz",
    "holonomy_array_sha256": "triangle_holonomy_arrays.npz",
}


def _verify_run_hashes(run: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    manifest_path = Path(str(run["source_manifest_path"]))
    source_dir = manifest_path.parent
    for field, filename in SOURCE_ARTIFACTS.items():
        path = source_dir / filename
        if not path.is_file():
            failures.append(f"missing:{filename}")
            continue
        if sha256_file(path) != str(run[field]):
            failures.append(f"hash_mismatch:{filename}")
    return failures


def assess_source_sufficiency(lock: Mapping[str, Any]) -> dict[str, Any]:
    """Inspect provenance/schema only; never open operator or holonomy result rows."""
    run_reports: list[dict[str, Any]] = []
    sufficient_run_ids: list[str] = []
    for run in lock.get("source_runs") or []:
        run_id = str(run["run_id"])
        failures = _verify_run_hashes(run)
        node_manifest_path = Path(str(run["source_node_manifest_path"]))
        if not node_manifest_path.is_file():
            failures.append("source_node_manifest_missing")
            node_manifest = None
        elif sha256_file(node_manifest_path) != str(run["source_node_manifest_sha256"]):
            failures.append("source_node_manifest_hash_mismatch")
            node_manifest = None
        else:
            node_manifest = read_json(node_manifest_path)

        referenced_path = Path(str(run["referenced_gate8_sample_source_path"]))
        required_source_files = [
            referenced_path / "manifest.json",
            referenced_path / "sample_registry.jsonl",
            referenced_path / "gate6_native",
            referenced_path / "samples",
        ]
        source_present = referenced_path.is_dir() and all(
            path.exists() for path in required_source_files
        )
        if not source_present:
            failures.append("retained_underlying_sample_rows_unavailable")
        if node_manifest is not None:
            declared = str(node_manifest.get("source_gate8_execution_dir") or "")
            if not declared:
                failures.append("frame_reconstruction_provenance_missing")

        status = "PASS" if not failures else "SPLIT_HALF_SOURCE_UNAVAILABLE"
        if status == "PASS":
            sufficient_run_ids.append(run_id)
        run_reports.append(
            {
                "run_id": run_id,
                "status": status,
                "hash_binding_status": (
                    "PASS" if not any("hash" in item or item.startswith("missing:") for item in failures) else "FAIL"
                ),
                "referenced_sample_source": str(referenced_path),
                "retained_underlying_sample_rows": source_present,
                "deterministic_split_key": lock["source_sufficiency"]["deterministic_split_key"],
                "frame_reconstruction_provenance": (
                    str(node_manifest.get("source_local_object_discipline") or "")
                    if node_manifest is not None
                    else None
                ),
                "failures": sorted(set(failures)),
            }
        )

    minimum = int(lock["source_sufficiency"]["minimum_source_sufficient_runs"])
    status = "PASS" if len(sufficient_run_ids) >= minimum else "SPLIT_HALF_SOURCE_UNAVAILABLE"
    return {
        "schema_version": "gate13_b2a_source_sufficiency_v1",
        "status": status,
        "B2A_SOURCE_SUFFICIENCY": status,
        "B2A": "READY" if status == "PASS" else "NOT_EXECUTED",
        "candidate_run_count": len(run_reports),
        "source_sufficient_run_count": len(sufficient_run_ids),
        "minimum_source_sufficient_runs": minimum,
        "source_sufficient_run_ids": sufficient_run_ids,
        "operator_outcomes_read": False,
        "runs": run_reports,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = assess_source_sufficiency(read_json(args.lock))
    write_json(args.out, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
