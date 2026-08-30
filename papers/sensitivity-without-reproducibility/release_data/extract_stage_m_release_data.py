#!/usr/bin/env python3
"""Export compact public Stage M records from the frozen sealed result.

This is a model-forward-free projection. It neither recomputes the scientific
operators nor opens Stage E. The source hash and every reported aggregate are
checked before any release file is written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


SOURCE_SHA256 = "8c56c308856d486b77356a42fe54ef0bea8f91e486dc5dd3820c3a60db5be772"
RETRIEVAL_INVENTORY_SHA256 = (
    "611ef9e5075c2698e93a42d2ab4d9e7ff3c2b719864d931ea1e01b18934513ae"
)
EXECUTION_ID = "bf41b049-f04b-442e-b0bd-05c8adbd4944"
FROZEN_LAYERS = (21, 43, 62)

FIELDNAMES = (
    "public_block_label",
    "block_id",
    "rollout_depth",
    "layer",
    "block_qualified",
    "layer_pass",
    "packets_qualified",
    "finite_crossfit",
    "reproducible",
    "broken_square_sensitive",
    "split_half_singular_floor",
    "split_half_floor_ceiling",
    "exact_response_half_1",
    "exact_response_half_2",
    "broken_response_half_1",
    "broken_response_half_2",
    "broken_sensitivity_threshold",
    "crossfit_energy_training_half_1",
    "crossfit_energy_training_half_2",
    "maximum_edge_condition",
    "maximum_path_condition",
    "minimum_edge_rank",
    "minimum_path_rank",
    "packet_disagreement_delta_fro",
    "packet_disagreement_path_p_fro",
    "packet_disagreement_path_q_fro",
    "packet_disagreement_h_path_fro",
    "packet_disagreement_h_edge_fro",
    "map_derived_competence",
    "qualified_block_primary_amplitude",
    "amplitude_status",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def packet_conditions(half: Mapping[str, Any]) -> tuple[list[float], list[int], list[float], list[int]]:
    edge_conditions: list[float] = []
    edge_ranks: list[int] = []
    for edge in half["edges"].values():
        edge_conditions.append(float(edge["condition"]))
        edge_ranks.append(int(edge["rank"]))

    path_conditions: list[float] = []
    path_ranks: list[int] = []
    for packet_name in ("exact_square", "broken_square"):
        raw = half[packet_name]["raw"]
        path_conditions.extend((float(raw["condition_p"]), float(raw["condition_q"])))
        path_ranks.extend((int(raw["rank_p"]), int(raw["rank_q"])))
    return edge_conditions, edge_ranks, path_conditions, path_ranks


def build_records(source: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if source.get("schema_version") != "gate13_track_c_sealed_map_result_v1":
        raise ValueError("unexpected sealed-map schema")
    if int(source.get("stage_m_forwards", -1)) != 4_800:
        raise ValueError("Stage M forward count drift")
    blocks = list(source.get("block_results", []))
    if len(blocks) != 20:
        raise ValueError("expected exactly 20 Stage M blocks")

    records: list[dict[str, Any]] = []
    for block_index, block in enumerate(blocks, start=1):
        layers = list(block["layers"])
        if [int(row["layer"]) for row in layers] != list(FROZEN_LAYERS):
            raise ValueError(f"frozen layer drift in {block['block_id']}")
        block_qualified = bool(block["qualified"])
        representation = dict(block["representation"])
        amplitude = (
            float(representation["primary_amplitude"])
            if block_qualified and not representation.get("unqualified_no_imputation", False)
            else None
        )
        amplitude_status = "QUALIFIED_VALUE" if amplitude is not None else "UNQUALIFIED_NO_IMPUTATION"
        competence = float(block["map_competence"]["map_derived_competence"])

        for layer in layers:
            validity = dict(layer["validity"])
            energies = [float(value) for value in layer["crossfit_energy_by_training_half"]]
            finite_crossfit = len(energies) == 2 and all(math.isfinite(value) for value in energies)
            conditions: list[float] = []
            edge_ranks: list[int] = []
            path_conditions: list[float] = []
            path_ranks: list[int] = []
            packets_qualified = True
            for half_name in ("half_1", "half_2"):
                half = validity[half_name]
                packets_qualified = packets_qualified and all(
                    half[name]["status"] == "QUALIFIED"
                    for name in ("exact_square", "broken_square")
                )
                ec, er, pc, pr = packet_conditions(half)
                conditions.extend(ec)
                edge_ranks.extend(er)
                path_conditions.extend(pc)
                path_ranks.extend(pr)
            if not all(math.isfinite(value) for value in (*conditions, *path_conditions)):
                raise ValueError(f"nonfinite conditioning record in {block['block_id']}")

            exact = [float(value) for value in validity["exact_square_normalized_delta_by_half"]]
            broken = [float(value) for value in validity["broken_square_normalized_delta_by_half"]]
            disagreement = dict(validity["packet_disagreement"])
            record = {
                "public_block_label": f"B{block_index:02d}",
                "block_id": str(block["block_id"]),
                "rollout_depth": int(block["rollout_depth"]),
                "layer": int(layer["layer"]),
                "block_qualified": block_qualified,
                "layer_pass": validity["status"] == "PASS",
                "packets_qualified": packets_qualified,
                "finite_crossfit": finite_crossfit,
                "reproducible": bool(validity["reproducible"]),
                "broken_square_sensitive": bool(validity["broken_square_sensitive"]),
                "split_half_singular_floor": float(validity["split_half_singular_floor"]),
                "split_half_floor_ceiling": float(validity["split_half_floor_ceiling"]),
                "exact_response_half_1": exact[0],
                "exact_response_half_2": exact[1],
                "broken_response_half_1": broken[0],
                "broken_response_half_2": broken[1],
                "broken_sensitivity_threshold": float(validity["broken_sensitivity_threshold"]),
                "crossfit_energy_training_half_1": energies[0],
                "crossfit_energy_training_half_2": energies[1],
                "maximum_edge_condition": max(conditions),
                "maximum_path_condition": max(path_conditions),
                "minimum_edge_rank": min(edge_ranks),
                "minimum_path_rank": min(path_ranks),
                "packet_disagreement_delta_fro": float(disagreement["Delta_pq_fro"]),
                "packet_disagreement_path_p_fro": float(disagreement["P_p_fro"]),
                "packet_disagreement_path_q_fro": float(disagreement["P_q_fro"]),
                "packet_disagreement_h_path_fro": float(disagreement["H_path_fro"]),
                "packet_disagreement_h_edge_fro": float(disagreement["H_edge_fro"]),
                "map_derived_competence": competence,
                "qualified_block_primary_amplitude": amplitude,
                "amplitude_status": amplitude_status,
            }
            records.append(record)

    layer_pass = Counter(int(row["layer"]) for row in records if row["layer_pass"])
    depth_pass = Counter(
        int(block["rollout_depth"]) for block in blocks if bool(block["qualified"])
    )
    checks = {
        "record_count": len(records),
        "block_count": len(blocks),
        "stage_m_forwards": int(source["stage_m_forwards"]),
        "finite_crossfit_layer_blocks": sum(bool(row["finite_crossfit"]) for row in records),
        "packet_qualified_layer_blocks": sum(bool(row["packets_qualified"]) for row in records),
        "broken_sensitive_layer_blocks": sum(bool(row["broken_square_sensitive"]) for row in records),
        "reproducible_layer_blocks": sum(bool(row["reproducible"]) for row in records),
        "layer_pass_counts": {str(layer): layer_pass[layer] for layer in FROZEN_LAYERS},
        "joint_block_pass_count": sum(bool(block["qualified"]) for block in blocks),
        "joint_block_pass_by_depth": {str(depth): depth_pass[depth] for depth in (2, 4, 6, 8)},
    }
    expected = {
        "record_count": 60,
        "block_count": 20,
        "stage_m_forwards": 4_800,
        "finite_crossfit_layer_blocks": 60,
        "packet_qualified_layer_blocks": 60,
        "broken_sensitive_layer_blocks": 59,
        "reproducible_layer_blocks": 35,
        "layer_pass_counts": {"21": 11, "43": 15, "62": 9},
        "joint_block_pass_count": 5,
        "joint_block_pass_by_depth": {"2": 0, "4": 2, "6": 2, "8": 1},
    }
    if checks != expected:
        raise ValueError(f"frozen aggregate mismatch: {checks}")
    return records, checks


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="frozen sealed_map_private_result.json")
    parser.add_argument("output_dir", type=Path, help="public release-data directory")
    args = parser.parse_args()

    if sha256_file(args.source) != SOURCE_SHA256:
        raise ValueError("sealed map source SHA-256 mismatch")
    source = load_json(args.source)
    records, checks = build_records(source)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / "stage_m_layer_block_records.csv"
    json_path = args.output_dir / "stage_m_layer_block_records.json"
    manifest_path = args.output_dir / "stage_m_release_data_manifest.json"
    write_csv(csv_path, records)
    json_path.write_bytes(
        json_bytes(
            {
                "schema_version": "sensitivity_without_reproducibility_stage_m_records_v1",
                "execution_id": EXECUTION_ID,
                "source_sealed_map_sha256": SOURCE_SHA256,
                "records": records,
                "verified_aggregates": checks,
            }
        )
    )
    manifest = {
        "schema_version": "sensitivity_without_reproducibility_release_data_manifest_v1",
        "execution_id": EXECUTION_ID,
        "source_execution_commit": "ea4558f0fcfd8944499407747615ec5bf41c1542",
        "tracked_closeout_commit": "407d20fd4f074b9ef4524c82c8136874efaec476",
        "source_sealed_map_sha256": SOURCE_SHA256,
        "retrieval_inventory_sha256": RETRIEVAL_INVENTORY_SHA256,
        "stage_e_state": "UNOPENED_MAP_NOT_QUALIFIED",
        "primary_analysis_state": "UNEXECUTED",
        "known_stale_receipt": {
            "object": "qwen3_6_27b Track A root artifact_manifest.json::execution_claim.json",
            "reason": "later fresh-operator terminal stage appended FRESH_SQUARE_OPERATOR_TERMINAL",
            "declared_pre_operator_sha256": "93d837b5e51562d6d3876f30bb1330cff54002f8a368245443846e0999197274",
            "final_actual_sha256": "53d6c77b74bc73ca2a7abfc2b88cb6158316095dbbe3497ee712c999187ed804",
            "remote_redownload_sha256": "53d6c77b74bc73ca2a7abfc2b88cb6158316095dbbe3497ee712c999187ed804",
            "scientific_payload_affected": False,
        },
        "verified_aggregates": checks,
        "files": {
            csv_path.name: sha256_file(csv_path),
            json_path.name: sha256_file(json_path),
        },
    }
    manifest_path.write_bytes(json_bytes(manifest))
    print(json.dumps({"status": "PASS", **manifest["files"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
