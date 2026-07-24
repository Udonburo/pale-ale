#!/usr/bin/env python3
"""Deterministic development-only sharding for Gate12C-2 outer experiments.

The scientific plan is independent of worker count and execution order.  Each
outer experiment is stored as one deterministic gzip shard and merged through
an index sorted by outer-experiment index.  Locked surfaces are intentionally
unrepresentable.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import platform
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import gate12c2_synthetic_lab as lab


PLAN_SCHEMA_VERSION = "gate12c2_development_shard_plan_v0.1"
SHARD_SCHEMA_VERSION = "gate12c2_development_outer_shard_v0.1"
INDEX_SCHEMA_VERSION = "gate12c2_development_shard_index_v0.1"
ALLOWED_REGIMES = frozenset(
    {
        "S0_true_null",
        "S1_known_reverse_shared_node_coupling",
        "S2_null_inflation",
    }
)


class Gate12C2ShardError(ValueError):
    """Raised when a development shard plan or artifact is inconsistent."""


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_hashes() -> dict[str, str]:
    return {
        "gate12c2_synthetic_lab.py": _sha256_file(
            Path(lab.__file__).resolve()
        ),
        "gate12c2_development_shards.py": _sha256_file(
            Path(__file__).resolve()
        ),
    }


def build_development_shard_plan(
    *,
    regime_id: str,
    master_seed: str,
    outer_experiment_indices: Sequence[int],
    block_count: int | Mapping[str, int],
    inner_valid_draw_count: int,
    effect_strength: float | None = None,
    max_draw_attempts: int | None = None,
    minimum_log_null_inflation: float = lab.S2_MIN_LOG_NULL_INFLATION,
    diagnostic_kernel: str = lab.BATCHED_DIAGNOSTIC_KERNEL,
) -> dict[str, Any]:
    """Freeze one development-only, scheduling-independent shard plan."""

    if regime_id not in ALLOWED_REGIMES:
        raise Gate12C2ShardError(f"unsupported regime: {regime_id!r}")
    indices = tuple(int(value) for value in outer_experiment_indices)
    if not indices or any(value < 0 for value in indices):
        raise Gate12C2ShardError(
            "outer_experiment_indices must be nonempty and nonnegative"
        )
    if len(set(indices)) != len(indices):
        raise Gate12C2ShardError("outer experiment indices must be unique")
    indices = tuple(sorted(indices))
    if not str(master_seed).strip():
        raise Gate12C2ShardError("master_seed must be nonempty")
    if inner_valid_draw_count <= 0:
        raise Gate12C2ShardError(
            "inner_valid_draw_count must be positive"
        )
    if diagnostic_kernel != lab.BATCHED_DIAGNOSTIC_KERNEL:
        raise Gate12C2ShardError(
            "the shard runner admits only the validated batched FP64 kernel"
        )
    if regime_id == "S1_known_reverse_shared_node_coupling":
        if (
            effect_strength is None
            or not math.isfinite(float(effect_strength))
            or float(effect_strength) <= 0.0
        ):
            raise Gate12C2ShardError(
                "S1 shard plans require a positive effect strength"
            )
    elif effect_strength not in {None, 0.0}:
        raise Gate12C2ShardError(
            "S0 and S2 shard plans do not accept an effect strength"
        )
    if max_draw_attempts is not None and int(max_draw_attempts) < (
        inner_valid_draw_count
    ):
        raise Gate12C2ShardError(
            "max_draw_attempts cannot be below inner_valid_draw_count"
        )
    if (
        not math.isfinite(float(minimum_log_null_inflation))
        or float(minimum_log_null_inflation) < 0.0
    ):
        raise Gate12C2ShardError(
            "minimum_log_null_inflation must be finite and nonnegative"
        )
    schedule = lab._resolve_block_count_schedule(block_count)
    payload: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "epistemic_status": "development_execution_plan_only",
        "contract_version": lab.C2_CONTRACT_VERSION,
        "surface_id": "development",
        "locked_execution_authorized": False,
        "real_held_out_execution_authorized": False,
        "regime_id": regime_id,
        "master_seed": str(master_seed),
        "outer_experiment_indices": list(indices),
        "block_count_schedule": lab._block_count_receipt(schedule),
        "inner_valid_draw_count": int(inner_valid_draw_count),
        "effect_strength": (
            None if effect_strength is None else float(effect_strength)
        ),
        "max_draw_attempts": (
            None
            if max_draw_attempts is None
            else int(max_draw_attempts)
        ),
        "minimum_log_null_inflation": float(
            minimum_log_null_inflation
        ),
        "diagnostic_kernel": diagnostic_kernel,
        "accepted_valid_draw_storage": (
            lab.COMPACT_ACCEPTED_PREFIX_STORAGE_ID
        ),
        "outer_experiment_schema": lab.OUTER_EXPERIMENT_SCHEMA_VERSION,
        "seed_namespace_schema": lab.SEED_NAMESPACE_SCHEMA_VERSION,
        "implementation_sha256": _implementation_hashes(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "N2_open": False,
        "N3_open": False,
        "public_claim": False,
    }
    payload["plan_payload_sha256"] = _sha256_bytes(
        _canonical_json_bytes(payload)
    )
    return payload


def _verified_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(plan)
    claimed = payload.pop("plan_payload_sha256", None)
    actual = _sha256_bytes(_canonical_json_bytes(payload))
    if claimed != actual:
        raise Gate12C2ShardError("development shard plan hash mismatch")
    if payload.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise Gate12C2ShardError("unsupported development shard plan schema")
    if payload.get("surface_id") != "development":
        raise Gate12C2ShardError("only the development surface is admitted")
    if payload.get("locked_execution_authorized") is not False:
        raise Gate12C2ShardError("locked execution must remain closed")
    current_hashes = _implementation_hashes()
    if payload.get("implementation_sha256") != current_hashes:
        raise Gate12C2ShardError(
            "implementation hashes no longer match the shard plan"
        )
    payload["plan_payload_sha256"] = str(claimed)
    return payload


def _plan_block_schedule(plan: Mapping[str, Any]) -> dict[str, int]:
    receipt = plan.get("block_count_schedule")
    if not isinstance(receipt, Mapping):
        raise Gate12C2ShardError("plan block schedule is missing")
    raw = receipt.get("block_count_by_case")
    if not isinstance(raw, Mapping):
        raise Gate12C2ShardError("plan case block counts are missing")
    return {str(key): int(value) for key, value in raw.items()}


def run_planned_outer_experiment(
    plan: Mapping[str, Any],
    *,
    outer_experiment_index: int,
) -> dict[str, Any]:
    """Run one outer unit after validating the immutable development plan."""

    verified = _verified_plan(plan)
    index = int(outer_experiment_index)
    admitted = tuple(
        int(value) for value in verified["outer_experiment_indices"]
    )
    if index not in admitted:
        raise Gate12C2ShardError(
            f"outer experiment {index} is not admitted by the plan"
        )
    common = {
        "master_seed": str(verified["master_seed"]),
        "outer_experiment_index": index,
        "block_count": _plan_block_schedule(verified),
        "inner_valid_draw_count": int(
            verified["inner_valid_draw_count"]
        ),
        "max_draw_attempts": verified["max_draw_attempts"],
        "diagnostic_kernel": str(verified["diagnostic_kernel"]),
    }
    regime_id = str(verified["regime_id"])
    if regime_id == "S2_null_inflation":
        result = lab.run_development_s2_identification_experiment(
            **common,
            minimum_log_null_inflation=float(
                verified["minimum_log_null_inflation"]
            ),
        )
    else:
        result = lab.run_development_outer_experiment(
            **common,
            regime_id=regime_id,
            effect_strength=verified["effect_strength"],
        )
    if result.get("surface_id") != "development":
        raise Gate12C2ShardError("runner returned a non-development result")
    if result.get("locked_execution_authorized") is not False:
        raise Gate12C2ShardError("runner opened a locked surface")
    if int(result["outer_experiment_index"]) != index:
        raise Gate12C2ShardError("runner returned the wrong outer index")
    return result


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _shard_path(output_dir: Path, outer_experiment_index: int) -> Path:
    return output_dir / "shards" / f"outer-{outer_experiment_index:06d}.json.gz"


def _read_shard(path: Path) -> dict[str, Any]:
    try:
        raw = gzip.decompress(path.read_bytes())
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise Gate12C2ShardError(f"could not read shard {path}: {exc}") from exc
    claimed = payload.get("shard_payload_sha256")
    projection = dict(payload)
    projection.pop("shard_payload_sha256", None)
    actual = _sha256_bytes(_canonical_json_bytes(projection))
    if claimed != actual:
        raise Gate12C2ShardError(f"shard payload hash mismatch: {path}")
    result_actual = _sha256_bytes(
        _canonical_json_bytes(payload.get("result"))
    )
    if payload.get("result_payload_sha256") != result_actual:
        raise Gate12C2ShardError(f"shard result hash mismatch: {path}")
    return payload


def _write_or_verify_shard(
    plan: Mapping[str, Any],
    output_dir: str,
    outer_experiment_index: int,
) -> dict[str, Any]:
    verified = _verified_plan(plan)
    destination = _shard_path(
        Path(output_dir).resolve(),
        int(outer_experiment_index),
    )
    if destination.exists():
        payload = _read_shard(destination)
        if payload.get("plan_payload_sha256") != verified[
            "plan_payload_sha256"
        ]:
            raise Gate12C2ShardError(
                f"existing shard belongs to a different plan: {destination}"
            )
        if int(payload.get("outer_experiment_index", -1)) != int(
            outer_experiment_index
        ):
            raise Gate12C2ShardError(
                f"existing shard has the wrong outer index: {destination}"
            )
        return _shard_index_row(destination, payload, reused=True)

    result = run_planned_outer_experiment(
        verified,
        outer_experiment_index=int(outer_experiment_index),
    )
    result_bytes = _canonical_json_bytes(result)
    payload: dict[str, Any] = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "epistemic_status": "development_outer_shard_only",
        "surface_id": "development",
        "locked_execution_authorized": False,
        "plan_payload_sha256": verified["plan_payload_sha256"],
        "outer_experiment_index": int(outer_experiment_index),
        "result_payload_sha256": _sha256_bytes(result_bytes),
        "result": result,
    }
    payload["shard_payload_sha256"] = _sha256_bytes(
        _canonical_json_bytes(payload)
    )
    raw = _canonical_json_bytes(payload)
    compressed = gzip.compress(raw, compresslevel=6, mtime=0)
    _atomic_write(destination, compressed)
    return _shard_index_row(destination, payload, reused=False)


def _shard_index_row(
    path: Path,
    payload: Mapping[str, Any],
    *,
    reused: bool,
) -> dict[str, Any]:
    result = payload["result"]
    if result["regime_id"] == "S2_null_inflation":
        decision = {
            "identification_success": bool(
                result["identification_success"]
            ),
            "identified_case_count": int(result["identified_case_count"]),
            "breadth_pass": bool(result["breadth_pass"]),
        }
    else:
        pipeline = result["pipeline_decision"]
        decision = {
            "claim_promotion": bool(pipeline["claim_promotion"]),
            "grid_outcome": str(pipeline["grid_outcome"]),
            "any_endpoint_support": bool(
                pipeline["any_endpoint_support"]
            ),
            "any_run_support": bool(pipeline["any_run_support"]),
            "q_directional_support_count": int(
                pipeline["q_directional_support_count"]
            ),
            "supporting_run_count": int(
                pipeline["supporting_run_count"]
            ),
            "q_discordant_run_count": int(
                pipeline["q_discordant_run_count"]
            ),
        }
    return {
        "outer_experiment_index": int(payload["outer_experiment_index"]),
        "relative_path": f"shards/{path.name}",
        "compressed_file_sha256": _sha256_file(path),
        "compressed_bytes": int(path.stat().st_size),
        "shard_payload_sha256": str(payload["shard_payload_sha256"]),
        "result_payload_sha256": str(payload["result_payload_sha256"]),
        "reused_existing_shard": bool(reused),
        "decision": decision,
    }


def execute_development_shard_plan(
    plan: Mapping[str, Any],
    *,
    output_dir: Path,
    worker_count: int = 1,
) -> dict[str, Any]:
    """Execute or resume a plan and write a deterministic merge index."""

    verified = _verified_plan(plan)
    if worker_count <= 0:
        raise Gate12C2ShardError("worker_count must be positive")
    destination = Path(output_dir).resolve()
    plan_path = destination / "plan.json"
    plan_bytes = _canonical_json_bytes(verified)
    if plan_path.exists():
        if plan_path.read_bytes() != plan_bytes:
            raise Gate12C2ShardError(
                "output directory already contains a different plan"
            )
    else:
        _atomic_write(plan_path, plan_bytes)

    indices = [
        int(value) for value in verified["outer_experiment_indices"]
    ]
    if worker_count == 1:
        rows = [
            _write_or_verify_shard(verified, str(destination), index)
            for index in indices
        ]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _write_or_verify_shard,
                    verified,
                    str(destination),
                    index,
                )
                for index in indices
            ]
            rows = [future.result() for future in futures]
    rows = sorted(rows, key=lambda row: row["outer_experiment_index"])
    index_payload: dict[str, Any] = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "epistemic_status": "development_shard_index_only",
        "surface_id": "development",
        "locked_execution_authorized": False,
        "plan_payload_sha256": verified["plan_payload_sha256"],
        "worker_count_operational_only": int(worker_count),
        "merge_order": "ascending_outer_experiment_index",
        "outer_experiment_count": len(rows),
        "all_outer_indices_present": [
            row["outer_experiment_index"] for row in rows
        ]
        == indices,
        "shards": rows,
    }
    payload_projection = {
        key: value
        for key, value in index_payload.items()
        if key != "worker_count_operational_only"
    }
    payload_projection["shards"] = [
        {
            key: value
            for key, value in row.items()
            if key != "reused_existing_shard"
        }
        for row in rows
    ]
    index_payload["scientific_projection_sha256"] = _sha256_bytes(
        _canonical_json_bytes(payload_projection)
    )
    index_payload["index_payload_sha256"] = _sha256_bytes(
        _canonical_json_bytes(index_payload)
    )
    _atomic_write(
        destination / "index.json",
        _canonical_json_bytes(index_payload),
    )
    return index_payload
