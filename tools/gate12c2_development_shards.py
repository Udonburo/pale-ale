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
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence


SINGLE_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
for _thread_environment_key, _thread_environment_value in (
    SINGLE_THREAD_ENVIRONMENT.items()
):
    os.environ[_thread_environment_key] = _thread_environment_value

import numpy as np
import threadpoolctl
from threadpoolctl import threadpool_info, threadpool_limits

import gate12c2_synthetic_lab as lab


PLAN_SCHEMA_VERSION = "gate12c2_development_shard_plan_v0.2"
SHARD_SCHEMA_VERSION = "gate12c2_development_outer_shard_v0.2"
INDEX_SCHEMA_VERSION = "gate12c2_development_shard_index_v0.2"
SHARD_SET_VERIFICATION_SCHEMA_VERSION = (
    "gate12c2_development_shard_set_verification_v0.1"
)
SCIENTIFIC_PROJECTION_SCHEMA_VERSION = (
    "gate12c2_development_scientific_projection_v0.2"
)
BLAS_THREAD_LIMIT = 1
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


def _blas_backend_receipt(
    *,
    include_active_threads: bool = True,
) -> list[dict[str, Any]]:
    """Return path-free BLAS metadata suitable for an environment contract."""

    rows = []
    for raw in threadpool_info():
        if raw.get("user_api") != "blas":
            continue
        row = {
                "user_api": str(raw.get("user_api")),
                "internal_api": str(raw.get("internal_api")),
                "prefix": str(raw.get("prefix")),
                "version": (
                    None
                    if raw.get("version") is None
                    else str(raw.get("version"))
                ),
                "threading_layer": (
                    None
                    if raw.get("threading_layer") is None
                    else str(raw.get("threading_layer"))
                ),
                "architecture": (
                    None
                    if raw.get("architecture") is None
                    else str(raw.get("architecture"))
                ),
        }
        if include_active_threads:
            row["active_num_threads"] = int(raw.get("num_threads", 0))
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            row["internal_api"],
            row["prefix"],
            str(row["version"]),
        ),
    )


def _numpy_build_receipt() -> dict[str, Any]:
    config = getattr(np.__config__, "CONFIG", {})
    dependencies = config.get("Build Dependencies", {})

    def dependency(name: str) -> dict[str, Any]:
        raw = dependencies.get(name, {})
        return {
            "name": raw.get("name"),
            "found": raw.get("found"),
            "version": raw.get("version"),
            "openblas_configuration": raw.get(
                "openblas configuration"
            ),
        }

    simd = config.get("SIMD Extensions", {})
    machine = config.get("Machine Information", {})
    return {
        "host_machine": dict(machine.get("host", {})),
        "blas": dependency("blas"),
        "lapack": dependency("lapack"),
        "simd_baseline": list(simd.get("baseline", [])),
        "simd_found": list(simd.get("found", [])),
    }


def _numerical_environment_receipt() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "threadpoolctl": threadpoolctl.__version__,
        "operating_system": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "thread_environment": dict(sorted(SINGLE_THREAD_ENVIRONMENT.items())),
        "blas_thread_limit": BLAS_THREAD_LIMIT,
        "numpy_build": _numpy_build_receipt(),
    }


def _assert_active_blas_limit(limit: int) -> list[dict[str, Any]]:
    rows = _blas_backend_receipt(include_active_threads=True)
    if not rows:
        raise Gate12C2ShardError(
            "no BLAS backend was visible to the shard runner"
        )
    if any(int(row["active_num_threads"]) != int(limit) for row in rows):
        raise Gate12C2ShardError(
            "active BLAS thread count does not match the frozen limit"
        )
    return rows


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
        "numerical_environment": _numerical_environment_receipt(),
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
    current_environment = _numerical_environment_receipt()
    if payload.get("numerical_environment") != current_environment:
        raise Gate12C2ShardError(
            "numerical environment no longer matches the shard plan"
        )
    if int(current_environment["blas_thread_limit"]) != BLAS_THREAD_LIMIT:
        raise Gate12C2ShardError("the frozen BLAS thread limit changed")
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
    with threadpool_limits(limits=BLAS_THREAD_LIMIT, user_api="blas"):
        _assert_active_blas_limit(BLAS_THREAD_LIMIT)
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
    result["numerical_execution_contract"] = {
        "blas_thread_limit": BLAS_THREAD_LIMIT,
        "thread_environment": dict(
            sorted(SINGLE_THREAD_ENVIRONMENT.items())
        ),
        "active_blas_thread_limit_verified": True,
        "numpy_build": verified["numerical_environment"]["numpy_build"],
        "guarantee_scope": (
            "same_frozen_software_and_numerical_environment"
        ),
        "cross_environment_bitwise_determinism_claimed": False,
    }
    return result


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise Gate12C2ShardError(
            f"stale partial write requires explicit removal: {temporary}"
        )
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _partial_artifact_paths(output_dir: Path) -> list[Path]:
    if not output_dir.exists():
        return []
    return sorted(
        (
            path
            for path in output_dir.rglob(".*.tmp")
            if path.is_file()
        ),
        key=lambda path: path.as_posix(),
    )


def _assert_no_partial_artifacts(output_dir: Path) -> None:
    partials = _partial_artifact_paths(output_dir)
    if partials:
        relative = [
            path.relative_to(output_dir).as_posix() for path in partials
        ]
        raise Gate12C2ShardError(
            "partial atomic-write artifacts require explicit review and "
            f"removal before resume: {relative}"
        )


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
    if payload.get("schema_version") != SHARD_SCHEMA_VERSION:
        raise Gate12C2ShardError(f"unsupported shard schema: {path}")
    if payload.get("surface_id") != "development":
        raise Gate12C2ShardError(f"non-development shard rejected: {path}")
    if payload.get("locked_execution_authorized") is not False:
        raise Gate12C2ShardError(f"locked shard rejected: {path}")
    return payload


def _draw_throughput_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    if result.get("regime_id") == "S2_null_inflation":
        endpoint_rows = result.get("endpoint_rows", [])
    else:
        endpoint_rows = result.get("endpoint_receipts", [])
    attempt_count = 0
    accepted_count = 0
    required_count = 0
    exhausted_incomplete_count = 0
    rejections: dict[str, int] = {}
    audit_count = 0
    for endpoint in endpoint_rows:
        for block in endpoint.get("block_rows", []):
            audit = block.get("inner_draw_audit")
            if not isinstance(audit, Mapping):
                raise Gate12C2ShardError(
                    "outer result is missing a block draw audit"
                )
            audit_count += 1
            attempts = int(audit["attempt_count"])
            accepted = int(audit["accepted_count"])
            attempt_count += attempts
            accepted_count += accepted
            required_count += int(audit["required_valid_count"])
            exhausted_incomplete_count += int(
                audit["max_attempt_status"] == "exhausted_incomplete"
            )
            block_rejections = {
                str(reason): int(count)
                for reason, count in audit[
                    "rejection_reason_counts"
                ].items()
            }
            if attempts != accepted + sum(block_rejections.values()):
                raise Gate12C2ShardError(
                    "block draw audit has unaccounted attempts"
                )
            for reason, count in block_rejections.items():
                rejections[reason] = rejections.get(reason, 0) + count

    case_rows = (
        result.get("case_rows", [])
        if result.get("regime_id") == "S2_null_inflation"
        else result.get("case_receipts", [])
    )
    generator_attempt_count = 0
    for case in case_rows:
        seed_audit = case.get("inner_draw_seed_stream_audit")
        if not isinstance(seed_audit, Mapping):
            raise Gate12C2ShardError(
                "outer result is missing a case seed-stream audit"
            )
        generator_attempt_count += int(seed_audit["attempt_count"])
    return {
        "block_q_audit_count": audit_count,
        "endpoint_draw_attempts": attempt_count,
        "endpoint_draw_acceptances": accepted_count,
        "endpoint_draw_required": required_count,
        "attempts_per_accepted_draw": (
            attempt_count / accepted_count
            if accepted_count > 0
            else None
        ),
        "rejection_reason_counts": dict(sorted(rejections.items())),
        "unaccounted_rejection_count": 0,
        "generator_attempt_count_across_cases": generator_attempt_count,
        "exhausted_incomplete_stream_count": exhausted_incomplete_count,
    }


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
        verification_started = time.perf_counter()
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
        return _shard_index_row(
            destination,
            payload,
            reused=True,
            operational_metrics={
                "mode": "verify_existing",
                "verification_wall_seconds": (
                    time.perf_counter() - verification_started
                ),
            },
        )

    total_started = time.perf_counter()
    compute_started = time.perf_counter()
    compute_cpu_started = time.process_time()
    result = run_planned_outer_experiment(
        verified,
        outer_experiment_index=int(outer_experiment_index),
    )
    compute_wall_seconds = time.perf_counter() - compute_started
    compute_cpu_seconds = time.process_time() - compute_cpu_started
    serialization_started = time.perf_counter()
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
    serialization_wall_seconds = time.perf_counter() - serialization_started
    draw_summary = _draw_throughput_summary(result)
    total_wall_seconds = time.perf_counter() - total_started
    accepted = int(draw_summary["endpoint_draw_acceptances"])
    return _shard_index_row(
        destination,
        payload,
        reused=False,
        operational_metrics={
            "mode": "execute_new",
            "compute_wall_seconds": compute_wall_seconds,
            "compute_cpu_seconds": compute_cpu_seconds,
            "serialization_write_wall_seconds": (
                serialization_wall_seconds
            ),
            "total_wall_seconds": total_wall_seconds,
            "endpoint_valid_draws_per_compute_second": (
                accepted / compute_wall_seconds
                if compute_wall_seconds > 0.0
                else None
            ),
            **draw_summary,
        },
    )


def _shard_index_row(
    path: Path,
    payload: Mapping[str, Any],
    *,
    reused: bool,
    operational_metrics: Mapping[str, Any] | None = None,
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
    row = {
        "outer_experiment_index": int(payload["outer_experiment_index"]),
        "relative_path": f"shards/{path.name}",
        "compressed_file_sha256": _sha256_file(path),
        "compressed_bytes": int(path.stat().st_size),
        "shard_payload_sha256": str(payload["shard_payload_sha256"]),
        "result_payload_sha256": str(payload["result_payload_sha256"]),
        "reused_existing_shard": bool(reused),
        "decision": decision,
    }
    if operational_metrics is not None:
        row["operational_metrics"] = dict(operational_metrics)
    return row


def _scientific_projection(
    plan: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Project only scientific/audit commitments, never runtime packaging."""

    return {
        "schema_version": SCIENTIFIC_PROJECTION_SCHEMA_VERSION,
        "plan_payload_sha256": str(plan["plan_payload_sha256"]),
        "outer_results": [
            {
                "outer_experiment_index": int(
                    row["outer_experiment_index"]
                ),
                "result_payload_sha256": str(
                    row["result_payload_sha256"]
                ),
                "decision": dict(row["decision"]),
            }
            for row in sorted(
                rows,
                key=lambda row: int(row["outer_experiment_index"]),
            )
        ],
        "scope": {
            "included_by_result_commitment": [
                "scientific_values",
                "statuses",
                "eligibility",
                "accepted_attempt_mapping",
                "rejection_classifications",
                "outer_experiment_identity",
                "endpoint_hierarchy",
                "seed_stream_commitments",
            ],
            "excluded": [
                "wall_time",
                "process_id",
                "absolute_path",
                "worker_completion_order",
                "temporary_filename",
                "timestamp",
                "compressed_file_size",
            ],
        },
    }


def _nonoperational_shard_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"reused_existing_shard", "operational_metrics"}
    }


def verify_development_shard_set(
    plan: Mapping[str, Any],
    *,
    output_dir: Path,
    candidate_paths: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Fail closed on any missing, duplicate, unexpected, or mixed shard."""

    verified = _verified_plan(plan)
    destination = Path(output_dir).resolve()
    _assert_no_partial_artifacts(destination)
    plan_path = destination / "plan.json"
    expected_plan_bytes = _canonical_json_bytes(verified)
    if not plan_path.exists() or plan_path.read_bytes() != expected_plan_bytes:
        raise Gate12C2ShardError(
            "output plan is missing or differs from the admitted plan"
        )

    expected_paths = {
        _shard_path(destination, int(index)).resolve()
        for index in verified["outer_experiment_indices"]
    }
    shard_dir = destination / "shards"
    actual_paths = {
        path.resolve()
        for path in (
            shard_dir.glob("*.json.gz") if shard_dir.exists() else ()
        )
        if path.is_file()
    }
    missing = sorted(
        path.relative_to(destination).as_posix()
        for path in expected_paths - actual_paths
    )
    unexpected = sorted(
        path.relative_to(destination).as_posix()
        for path in actual_paths - expected_paths
    )
    if missing or unexpected:
        raise Gate12C2ShardError(
            "shard set is incomplete or unexpected: "
            f"missing={missing}, unexpected={unexpected}"
        )

    if candidate_paths is None:
        ordered_paths = sorted(
            actual_paths,
            key=lambda path: path.as_posix(),
        )
    else:
        ordered_paths = [
            Path(path).resolve() for path in candidate_paths
        ]
        if len(set(ordered_paths)) != len(ordered_paths):
            raise Gate12C2ShardError(
                "duplicate shard path supplied to merge verification"
            )
        if set(ordered_paths) != expected_paths:
            raise Gate12C2ShardError(
                "candidate shard paths do not equal the frozen shard set"
            )

    rows: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for path in ordered_paths:
        payload = _read_shard(path)
        if payload.get("plan_payload_sha256") != verified[
            "plan_payload_sha256"
        ]:
            raise Gate12C2ShardError(
                f"shard belongs to a different plan: {path}"
            )
        index = int(payload.get("outer_experiment_index", -1))
        if index in seen_indices:
            raise Gate12C2ShardError(
                f"duplicate outer experiment index in shard set: {index}"
            )
        seen_indices.add(index)
        expected_path = _shard_path(destination, index).resolve()
        if path != expected_path:
            raise Gate12C2ShardError(
                f"shard path does not match its outer index: {path}"
            )
        result = payload.get("result")
        if not isinstance(result, Mapping):
            raise Gate12C2ShardError(f"shard result is missing: {path}")
        if result.get("schema_version") != lab.OUTER_EXPERIMENT_SCHEMA_VERSION:
            raise Gate12C2ShardError(
                f"outer experiment schema mismatch: {path}"
            )
        if result.get("surface_id") != "development":
            raise Gate12C2ShardError(
                f"outer result is not development-only: {path}"
            )
        if result.get("locked_execution_authorized") is not False:
            raise Gate12C2ShardError(
                f"outer result opened the locked surface: {path}"
            )
        if int(result.get("outer_experiment_index", -1)) != index:
            raise Gate12C2ShardError(
                f"outer result index mismatch: {path}"
            )
        rows.append(_shard_index_row(path, payload, reused=False))

    canonical_rows = sorted(
        rows,
        key=lambda row: int(row["outer_experiment_index"]),
    )
    expected_indices = [
        int(index) for index in verified["outer_experiment_indices"]
    ]
    actual_indices = [
        int(row["outer_experiment_index"]) for row in canonical_rows
    ]
    if actual_indices != expected_indices:
        raise Gate12C2ShardError(
            "verified outer experiment IDs do not match the frozen plan"
        )
    projection = _scientific_projection(verified, canonical_rows)
    return {
        "schema_version": SHARD_SET_VERIFICATION_SCHEMA_VERSION,
        "epistemic_status": "development_shard_set_verification_only",
        "surface_id": "development",
        "locked_execution_authorized": False,
        "plan_payload_sha256": verified["plan_payload_sha256"],
        "outer_experiment_count": len(canonical_rows),
        "missing_outer_ids": [],
        "duplicate_outer_ids": [],
        "unexpected_outer_ids": [],
        "partial_artifacts": [],
        "canonical_rows": canonical_rows,
        "scientific_projection": projection,
        "scientific_projection_sha256": _sha256_bytes(
            _canonical_json_bytes(projection)
        ),
    }


def _build_index_payload(
    plan: Mapping[str, Any],
    *,
    rows: Sequence[Mapping[str, Any]],
    worker_count: int,
    operational_execution_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    canonical_rows = sorted(
        (dict(row) for row in rows),
        key=lambda row: int(row["outer_experiment_index"]),
    )
    expected_indices = [
        int(value) for value in plan["outer_experiment_indices"]
    ]
    projection = _scientific_projection(plan, canonical_rows)
    payload: dict[str, Any] = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "epistemic_status": "development_shard_index_only",
        "surface_id": "development",
        "locked_execution_authorized": False,
        "plan_payload_sha256": plan["plan_payload_sha256"],
        "worker_count_operational_only": int(worker_count),
        "merge_order": "ascending_outer_experiment_index",
        "outer_experiment_count": len(canonical_rows),
        "all_outer_indices_present": [
            row["outer_experiment_index"] for row in canonical_rows
        ]
        == expected_indices,
        "shards": canonical_rows,
        "scientific_projection_schema_version": (
            SCIENTIFIC_PROJECTION_SCHEMA_VERSION
        ),
        "scientific_projection_sha256": _sha256_bytes(
            _canonical_json_bytes(projection)
        ),
    }
    if operational_execution_metrics is not None:
        payload["operational_execution_metrics"] = dict(
            operational_execution_metrics
        )
    payload["index_payload_sha256"] = _sha256_bytes(
        _canonical_json_bytes(payload)
    )
    return payload


def verify_development_shard_index(
    plan: Mapping[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Verify the merge index against the current complete shard set."""

    destination = Path(output_dir).resolve()
    index_path = destination / "index.json"
    if not index_path.exists():
        raise Gate12C2ShardError("development shard index is missing")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise Gate12C2ShardError(
            f"could not read development shard index: {exc}"
        ) from exc
    claimed = index.get("index_payload_sha256")
    projection = dict(index)
    projection.pop("index_payload_sha256", None)
    actual = _sha256_bytes(_canonical_json_bytes(projection))
    if claimed != actual:
        raise Gate12C2ShardError(
            "development shard index payload hash mismatch"
        )
    if index.get("schema_version") != INDEX_SCHEMA_VERSION:
        raise Gate12C2ShardError(
            "unsupported development shard index schema"
        )
    verification = verify_development_shard_set(
        plan,
        output_dir=destination,
    )
    if index.get("plan_payload_sha256") != verification[
        "plan_payload_sha256"
    ]:
        raise Gate12C2ShardError(
            "development shard index belongs to a different plan"
        )
    if index.get("scientific_projection_sha256") != verification[
        "scientific_projection_sha256"
    ]:
        raise Gate12C2ShardError(
            "development shard index scientific projection mismatch"
        )
    indexed_rows = [
        _nonoperational_shard_row(row)
        for row in index.get("shards", [])
    ]
    verified_rows = [
        _nonoperational_shard_row(row)
        for row in verification["canonical_rows"]
    ]
    if indexed_rows != verified_rows:
        raise Gate12C2ShardError(
            "development shard index rows do not match current shards"
        )
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "status": "pass",
        "plan_payload_sha256": verification["plan_payload_sha256"],
        "outer_experiment_count": verification[
            "outer_experiment_count"
        ],
        "scientific_projection_sha256": verification[
            "scientific_projection_sha256"
        ],
        "index_payload_sha256": str(claimed),
    }


def execute_development_shard_plan(
    plan: Mapping[str, Any],
    *,
    output_dir: Path,
    worker_count: int = 1,
) -> dict[str, Any]:
    """Execute or resume a plan and write a deterministic merge index."""

    verified = _verified_plan(plan)
    execution_started = time.perf_counter()
    if worker_count <= 0:
        raise Gate12C2ShardError("worker_count must be positive")
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    _assert_no_partial_artifacts(destination)
    plan_path = destination / "plan.json"
    plan_bytes = _canonical_json_bytes(verified)
    if plan_path.exists():
        if plan_path.read_bytes() != plan_bytes:
            raise Gate12C2ShardError(
                "output directory already contains a different plan"
            )
    else:
        _atomic_write(plan_path, plan_bytes)
    if (destination / "index.json").exists():
        verify_development_shard_index(
            verified,
            output_dir=destination,
        )

    indices = [
        int(value) for value in verified["outer_experiment_indices"]
    ]
    shard_phase_started = time.perf_counter()
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
    shard_phase_wall_seconds = time.perf_counter() - shard_phase_started
    reuse_by_index = {
        int(row["outer_experiment_index"]): bool(
            row["reused_existing_shard"]
        )
        for row in rows
    }
    operational_by_index = {
        int(row["outer_experiment_index"]): dict(
            row.get("operational_metrics", {})
        )
        for row in rows
    }
    merge_started = time.perf_counter()
    verification = verify_development_shard_set(
        verified,
        output_dir=destination,
    )
    verified_rows = []
    for raw_row in verification["canonical_rows"]:
        row = dict(raw_row)
        row["reused_existing_shard"] = reuse_by_index[
            int(row["outer_experiment_index"])
        ]
        row["operational_metrics"] = operational_by_index[
            int(row["outer_experiment_index"])
        ]
        verified_rows.append(row)
    index_payload = _build_index_payload(
        verified,
        rows=verified_rows,
        worker_count=worker_count,
        operational_execution_metrics={
            "shard_phase_wall_seconds": shard_phase_wall_seconds,
            "merge_validation_before_write_wall_seconds": (
                time.perf_counter() - merge_started
            ),
            "execution_before_index_write_wall_seconds": (
                time.perf_counter() - execution_started
            ),
        },
    )
    _atomic_write(
        destination / "index.json",
        _canonical_json_bytes(index_payload),
    )
    verify_development_shard_index(
        verified,
        output_dir=destination,
    )
    return index_payload
