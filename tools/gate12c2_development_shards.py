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


PLAN_SCHEMA_VERSION = "gate12c2_development_shard_plan_v0.3"
SHARD_SCHEMA_VERSION = "gate12c2_development_outer_shard_v0.3"
INDEX_SCHEMA_VERSION = "gate12c2_development_shard_index_v0.3"
SHARD_SET_VERIFICATION_SCHEMA_VERSION = (
    "gate12c2_development_shard_set_verification_v0.1"
)
SCIENTIFIC_PROJECTION_SCHEMA_VERSION = (
    "gate12c2_development_scientific_projection_v0.3"
)
RESULT_EXECUTION_CONTRACT_SCHEMA_VERSION = (
    "gate12c2_result_execution_contract_v0.1"
)
NO_OUTCOME_PREFLIGHT_SCHEMA_VERSION = (
    "gate12c2_no_outcome_preflight_v0.1"
)
EXECUTION_AUTHORIZATION_SCHEMA_VERSION = (
    "gate12c2_development_execution_authorization_v0.1"
)
BLAS_THREAD_LIMIT = 1
ALLOWED_REGIMES = frozenset(
    {
        "S0_true_null",
        "S1_known_reverse_shared_node_coupling",
        "S2_null_inflation",
    }
)
REQUIRED_PREFLIGHT_CHECKS = (
    "plan_hash_verified",
    "implementation_hashes_verified",
    "numerical_environment_verified",
    "closed_boundaries_verified",
    "output_root_verified",
    "outer_ids_verified",
    "no_scientific_outcomes_inspected",
    "recovery_requirement_verified",
)


class Gate12C2ShardError(ValueError):
    """Raised when a development shard plan or artifact is inconsistent."""


def _canonical_json_bytes(payload: object) -> bytes:
    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise Gate12C2ShardError(
            f"canonical JSON requires finite, serializable values: {exc}"
        ) from exc
    return encoded.encode("utf-8")


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
        "development_execution_requires_external_authorization": True,
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
        "epsilon": float(lab.DEFAULT_LOG_EPSILON),
        "diagnostic_kernel": diagnostic_kernel,
        "accepted_valid_draw_storage": (
            lab.COMPACT_ACCEPTED_PREFIX_STORAGE_ID
        ),
        "outer_experiment_schema": lab.OUTER_EXPERIMENT_SCHEMA_VERSION,
        "seed_namespace_schema": lab.SEED_NAMESPACE_SCHEMA_VERSION,
        "scientific_execution_parameters": {
            "reference_dtype": lab.REFERENCE_DTYPE,
            "numeric_atol": float(lab.DEFAULT_NUMERIC_ATOL),
            "degeneracy_atol": float(lab.DEFAULT_DEGENERACY_ATOL),
            "relative_gap_min": float(lab.DEFAULT_RELATIVE_GAP_MIN),
            "holm_alpha": float(lab.DEFAULT_HOLM_ALPHA),
            "primary_zero_tolerance": float(
                lab.DEFAULT_PRIMARY_ZERO_TOLERANCE
            ),
            "log_epsilon": float(lab.DEFAULT_LOG_EPSILON),
        },
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


def _plan_block_schedule(plan: Mapping[str, Any]) -> dict[str, int]:
    receipt = plan.get("block_count_schedule")
    if not isinstance(receipt, Mapping):
        raise Gate12C2ShardError("plan block schedule is missing")
    raw = receipt.get("block_count_by_case")
    if not isinstance(raw, Mapping):
        raise Gate12C2ShardError("plan case block counts are missing")
    return {str(key): int(value) for key, value in raw.items()}


def _verified_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild and compare the complete admitted plan, never just its hash."""

    if not isinstance(plan, Mapping):
        raise Gate12C2ShardError("development shard plan must be a mapping")
    supplied = dict(plan)
    claimed = supplied.get("plan_payload_sha256")
    payload = dict(supplied)
    payload.pop("plan_payload_sha256", None)
    actual = _sha256_bytes(_canonical_json_bytes(payload))
    if claimed != actual:
        raise Gate12C2ShardError("development shard plan hash mismatch")
    if payload.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise Gate12C2ShardError("unsupported development shard plan schema")
    try:
        expected = build_development_shard_plan(
            regime_id=str(payload["regime_id"]),
            master_seed=str(payload["master_seed"]),
            outer_experiment_indices=[
                int(value)
                for value in payload["outer_experiment_indices"]
            ],
            block_count=_plan_block_schedule(payload),
            inner_valid_draw_count=int(
                payload["inner_valid_draw_count"]
            ),
            effect_strength=payload["effect_strength"],
            max_draw_attempts=payload["max_draw_attempts"],
            minimum_log_null_inflation=float(
                payload["minimum_log_null_inflation"]
            ),
            diagnostic_kernel=str(payload["diagnostic_kernel"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, Gate12C2ShardError):
            raise
        raise Gate12C2ShardError(
            f"development shard plan cannot be reconstructed: {exc}"
        ) from exc
    if supplied != expected:
        unexpected = sorted(set(supplied) - set(expected))
        missing = sorted(set(expected) - set(supplied))
        changed = sorted(
            key
            for key in set(supplied) & set(expected)
            if supplied[key] != expected[key]
        )
        raise Gate12C2ShardError(
            "development shard plan differs from the complete builder "
            f"contract: missing={missing}, unexpected={unexpected}, "
            f"changed={changed}"
        )
    if int(expected["numerical_environment"]["blas_thread_limit"]) != (
        BLAS_THREAD_LIMIT
    ):
        raise Gate12C2ShardError("the frozen BLAS thread limit changed")
    return expected


def _normalized_output_root(output_dir: Path) -> str:
    return Path(output_dir).resolve().as_posix()


def _require_exact_keys(
    payload: Mapping[str, Any],
    expected: set[str],
    *,
    context: str,
) -> None:
    actual = set(payload)
    if actual != expected:
        raise Gate12C2ShardError(
            f"{context} keys differ from the frozen schema: "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def build_no_outcome_preflight_receipt(
    plan: Mapping[str, Any],
    *,
    output_dir: Path,
    worker_count: int,
    preflight_id: str,
    checks: Mapping[str, bool],
) -> dict[str, Any]:
    """Build a plan-bound, outcome-blind preflight receipt."""

    verified = _verified_plan(plan)
    if worker_count <= 0:
        raise Gate12C2ShardError("preflight worker count must be positive")
    if not str(preflight_id).strip():
        raise Gate12C2ShardError("preflight_id must be nonempty")
    normalized_checks = {
        str(key): bool(value) for key, value in checks.items()
    }
    if set(normalized_checks) != set(REQUIRED_PREFLIGHT_CHECKS):
        raise Gate12C2ShardError(
            "preflight checks do not match the required closed allowlist"
        )
    if not all(normalized_checks.values()):
        raise Gate12C2ShardError(
            "a passing preflight requires every frozen check to pass"
        )
    payload: dict[str, Any] = {
        "schema_version": NO_OUTCOME_PREFLIGHT_SCHEMA_VERSION,
        "preflight_id": str(preflight_id),
        "epistemic_status": "development_no_outcome_preflight_only",
        "surface_id": "development",
        "preflight_status": "pass",
        "development_execution_authorized": False,
        "locked_execution_authorized": False,
        "real_held_out_execution_authorized": False,
        "N2_open": False,
        "N3_open": False,
        "public_claim": False,
        "scientific_outcomes_inspected": False,
        "plan_payload_sha256": verified["plan_payload_sha256"],
        "implementation_sha256": dict(
            verified["implementation_sha256"]
        ),
        "numerical_environment_sha256": _sha256_bytes(
            _canonical_json_bytes(verified["numerical_environment"])
        ),
        "output_root": _normalized_output_root(output_dir),
        "worker_count": int(worker_count),
        "checks": dict(sorted(normalized_checks.items())),
    }
    payload["preflight_receipt_payload_sha256"] = _sha256_bytes(
        _canonical_json_bytes(payload)
    )
    return payload


def _verified_no_outcome_preflight(
    plan: Mapping[str, Any],
    preflight_receipt: Mapping[str, Any],
    *,
    output_dir: Path,
    worker_count: int,
) -> dict[str, Any]:
    verified = _verified_plan(plan)
    if not isinstance(preflight_receipt, Mapping):
        raise Gate12C2ShardError("preflight receipt must be a mapping")
    supplied = dict(preflight_receipt)
    expected_keys = {
        "schema_version",
        "preflight_id",
        "epistemic_status",
        "surface_id",
        "preflight_status",
        "development_execution_authorized",
        "locked_execution_authorized",
        "real_held_out_execution_authorized",
        "N2_open",
        "N3_open",
        "public_claim",
        "scientific_outcomes_inspected",
        "plan_payload_sha256",
        "implementation_sha256",
        "numerical_environment_sha256",
        "output_root",
        "worker_count",
        "checks",
        "preflight_receipt_payload_sha256",
    }
    _require_exact_keys(
        supplied,
        expected_keys,
        context="no-outcome preflight receipt",
    )
    claimed = supplied["preflight_receipt_payload_sha256"]
    unhashed = dict(supplied)
    unhashed.pop("preflight_receipt_payload_sha256")
    if claimed != _sha256_bytes(_canonical_json_bytes(unhashed)):
        raise Gate12C2ShardError("preflight receipt hash mismatch")
    closed_values = {
        "schema_version": NO_OUTCOME_PREFLIGHT_SCHEMA_VERSION,
        "epistemic_status": "development_no_outcome_preflight_only",
        "surface_id": "development",
        "preflight_status": "pass",
        "development_execution_authorized": False,
        "locked_execution_authorized": False,
        "real_held_out_execution_authorized": False,
        "N2_open": False,
        "N3_open": False,
        "public_claim": False,
        "scientific_outcomes_inspected": False,
        "plan_payload_sha256": verified["plan_payload_sha256"],
        "implementation_sha256": verified["implementation_sha256"],
        "numerical_environment_sha256": _sha256_bytes(
            _canonical_json_bytes(verified["numerical_environment"])
        ),
        "output_root": _normalized_output_root(output_dir),
        "worker_count": int(worker_count),
    }
    for key, expected_value in closed_values.items():
        if supplied[key] != expected_value:
            raise Gate12C2ShardError(
                f"preflight receipt changed frozen field {key!r}"
            )
    checks = supplied["checks"]
    if not isinstance(checks, Mapping):
        raise Gate12C2ShardError("preflight checks must be a mapping")
    _require_exact_keys(
        checks,
        set(REQUIRED_PREFLIGHT_CHECKS),
        context="preflight checks",
    )
    if any(value is not True for value in checks.values()):
        raise Gate12C2ShardError("preflight receipt contains a failed check")
    if not str(supplied["preflight_id"]).strip():
        raise Gate12C2ShardError("preflight_id must be nonempty")
    return supplied


def build_development_execution_authorization(
    plan: Mapping[str, Any],
    preflight_receipt: Mapping[str, Any],
    *,
    output_dir: Path,
    worker_count: int,
    authorization_id: str,
    purpose: str,
) -> dict[str, Any]:
    """Explicitly authorize one plan/output/worker execution after preflight."""

    verified = _verified_plan(plan)
    preflight = _verified_no_outcome_preflight(
        verified,
        preflight_receipt,
        output_dir=output_dir,
        worker_count=worker_count,
    )
    if not str(authorization_id).strip() or not str(purpose).strip():
        raise Gate12C2ShardError(
            "authorization_id and purpose must be nonempty"
        )
    payload: dict[str, Any] = {
        "schema_version": EXECUTION_AUTHORIZATION_SCHEMA_VERSION,
        "authorization_id": str(authorization_id),
        "epistemic_status": "development_execution_authorization_only",
        "surface_id": "development",
        "development_execution_authorized": True,
        "locked_execution_authorized": False,
        "real_held_out_execution_authorized": False,
        "N2_open": False,
        "N3_open": False,
        "public_claim": False,
        "plan_payload_sha256": verified["plan_payload_sha256"],
        "preflight_receipt_payload_sha256": preflight[
            "preflight_receipt_payload_sha256"
        ],
        "implementation_sha256": dict(
            verified["implementation_sha256"]
        ),
        "output_root": _normalized_output_root(output_dir),
        "worker_count": int(worker_count),
        "purpose": str(purpose),
    }
    payload["authorization_receipt_payload_sha256"] = _sha256_bytes(
        _canonical_json_bytes(payload)
    )
    return payload


def _verified_execution_authorization(
    plan: Mapping[str, Any],
    preflight_receipt: Mapping[str, Any],
    authorization_receipt: Mapping[str, Any],
    *,
    output_dir: Path,
    worker_count: int,
) -> dict[str, Any]:
    verified = _verified_plan(plan)
    preflight = _verified_no_outcome_preflight(
        verified,
        preflight_receipt,
        output_dir=output_dir,
        worker_count=worker_count,
    )
    if not isinstance(authorization_receipt, Mapping):
        raise Gate12C2ShardError(
            "execution authorization receipt must be a mapping"
        )
    supplied = dict(authorization_receipt)
    expected_keys = {
        "schema_version",
        "authorization_id",
        "epistemic_status",
        "surface_id",
        "development_execution_authorized",
        "locked_execution_authorized",
        "real_held_out_execution_authorized",
        "N2_open",
        "N3_open",
        "public_claim",
        "plan_payload_sha256",
        "preflight_receipt_payload_sha256",
        "implementation_sha256",
        "output_root",
        "worker_count",
        "purpose",
        "authorization_receipt_payload_sha256",
    }
    _require_exact_keys(
        supplied,
        expected_keys,
        context="development execution authorization",
    )
    claimed = supplied["authorization_receipt_payload_sha256"]
    unhashed = dict(supplied)
    unhashed.pop("authorization_receipt_payload_sha256")
    if claimed != _sha256_bytes(_canonical_json_bytes(unhashed)):
        raise Gate12C2ShardError(
            "development execution authorization hash mismatch"
        )
    expected_values = {
        "schema_version": EXECUTION_AUTHORIZATION_SCHEMA_VERSION,
        "epistemic_status": "development_execution_authorization_only",
        "surface_id": "development",
        "development_execution_authorized": True,
        "locked_execution_authorized": False,
        "real_held_out_execution_authorized": False,
        "N2_open": False,
        "N3_open": False,
        "public_claim": False,
        "plan_payload_sha256": verified["plan_payload_sha256"],
        "preflight_receipt_payload_sha256": preflight[
            "preflight_receipt_payload_sha256"
        ],
        "implementation_sha256": verified["implementation_sha256"],
        "output_root": _normalized_output_root(output_dir),
        "worker_count": int(worker_count),
    }
    for key, expected_value in expected_values.items():
        if supplied[key] != expected_value:
            raise Gate12C2ShardError(
                f"execution authorization changed frozen field {key!r}"
            )
    if not str(supplied["authorization_id"]).strip():
        raise Gate12C2ShardError("authorization_id must be nonempty")
    if not str(supplied["purpose"]).strip():
        raise Gate12C2ShardError("authorization purpose must be nonempty")
    return supplied


def _resolved_max_draw_attempts(plan: Mapping[str, Any]) -> int:
    inner = int(plan["inner_valid_draw_count"])
    configured = plan["max_draw_attempts"]
    return (
        max(inner * 4, inner + 8)
        if configured is None
        else int(configured)
    )


def _numerical_execution_contract(
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "blas_thread_limit": BLAS_THREAD_LIMIT,
        "thread_environment": dict(
            sorted(SINGLE_THREAD_ENVIRONMENT.items())
        ),
        "active_blas_thread_limit_verified": True,
        "numpy_build": plan["numerical_environment"]["numpy_build"],
        "scientific_execution_parameters": dict(
            plan["scientific_execution_parameters"]
        ),
        "guarantee_scope": (
            "same_frozen_software_and_numerical_environment"
        ),
        "cross_environment_bitwise_determinism_claimed": False,
    }


def _result_execution_configuration(
    plan: Mapping[str, Any],
    *,
    outer_experiment_index: int,
) -> dict[str, Any]:
    return {
        "schema_version": RESULT_EXECUTION_CONTRACT_SCHEMA_VERSION,
        "plan_payload_sha256": str(plan["plan_payload_sha256"]),
        "contract_version": str(plan["contract_version"]),
        "surface_id": "development",
        "locked_execution_authorized": False,
        "real_held_out_execution_authorized": False,
        "N2_open": False,
        "N3_open": False,
        "public_claim": False,
        "regime_id": str(plan["regime_id"]),
        "master_seed_sha256": _sha256_bytes(
            str(plan["master_seed"]).encode("utf-8")
        ),
        "outer_experiment_index": int(outer_experiment_index),
        "block_count_schedule": dict(plan["block_count_schedule"]),
        "inner_valid_draw_count": int(
            plan["inner_valid_draw_count"]
        ),
        "effect_strength": plan["effect_strength"],
        "configured_max_draw_attempts": plan["max_draw_attempts"],
        "resolved_max_draw_attempts": _resolved_max_draw_attempts(plan),
        "minimum_log_null_inflation": float(
            plan["minimum_log_null_inflation"]
        ),
        "epsilon": float(plan["epsilon"]),
        "diagnostic_kernel": str(plan["diagnostic_kernel"]),
        "accepted_valid_draw_storage": str(
            plan["accepted_valid_draw_storage"]
        ),
        "outer_experiment_schema": str(
            plan["outer_experiment_schema"]
        ),
        "seed_namespace_schema": str(plan["seed_namespace_schema"]),
        "scientific_execution_parameters": dict(
            plan["scientific_execution_parameters"]
        ),
        "implementation_sha256": dict(plan["implementation_sha256"]),
        "numerical_environment_sha256": _sha256_bytes(
            _canonical_json_bytes(plan["numerical_environment"])
        ),
    }


def _verify_result_against_plan(
    plan: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    outer_experiment_index: int,
) -> None:
    """Reject self-consistent result hashes with a contradictory configuration."""

    expected = _result_execution_configuration(
        plan,
        outer_experiment_index=outer_experiment_index,
    )
    if result.get("execution_configuration_contract") != expected:
        raise Gate12C2ShardError(
            "outer result execution configuration differs from the plan"
        )
    top_level_expected = {
        "schema_version": plan["outer_experiment_schema"],
        "contract_version": plan["contract_version"],
        "surface_id": "development",
        "locked_execution_authorized": False,
        "regime_id": plan["regime_id"],
        "outer_experiment_index": int(outer_experiment_index),
        "block_count_schedule": plan["block_count_schedule"],
        "inner_valid_draw_count": int(
            plan["inner_valid_draw_count"]
        ),
        "max_draw_attempts": _resolved_max_draw_attempts(plan),
        "diagnostic_kernel": plan["diagnostic_kernel"],
        "accepted_valid_draw_storage": plan[
            "accepted_valid_draw_storage"
        ],
    }
    for key, expected_value in top_level_expected.items():
        if result.get(key) != expected_value:
            raise Gate12C2ShardError(
                f"outer result changed plan-bound field {key!r}"
            )
    effect_strength = result.get("effect_strength")
    if effect_strength != plan["effect_strength"]:
        raise Gate12C2ShardError(
            "outer result effect strength differs from the plan"
        )
    if result.get("numerical_execution_contract") != (
        _numerical_execution_contract(plan)
    ):
        raise Gate12C2ShardError(
            "outer result numerical execution contract differs from the plan"
        )
    if plan["regime_id"] == "S2_null_inflation":
        if result.get("observed_process_modified") is not False:
            raise Gate12C2ShardError(
                "S2 outer result modified the observed process"
            )
        if result.get("paired_null_arms") != [
            lab.N1_ID,
            lab.S2_UNCONSTRAINED_ORIENTATION_ID,
        ]:
            raise Gate12C2ShardError(
                "S2 outer result changed its paired null arms"
            )
        endpoint_rows = result.get("endpoint_rows")
        if not isinstance(endpoint_rows, list) or len(endpoint_rows) != 24:
            raise Gate12C2ShardError(
                "S2 outer result must contain exactly 24 endpoints"
            )
        for endpoint in endpoint_rows:
            if endpoint.get("minimum_log_null_inflation") != float(
                plan["minimum_log_null_inflation"]
            ):
                raise Gate12C2ShardError(
                    "S2 endpoint changed the minimum inflation threshold"
                )


def run_planned_outer_experiment(
    plan: Mapping[str, Any],
    *,
    outer_experiment_index: int,
    output_dir: Path,
    worker_count: int,
    preflight_receipt: Mapping[str, Any],
    authorization_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Run one outer unit after validating the immutable development plan."""

    verified = _verified_plan(plan)
    _verified_execution_authorization(
        verified,
        preflight_receipt,
        authorization_receipt,
        output_dir=output_dir,
        worker_count=worker_count,
    )
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
        "epsilon": float(verified["epsilon"]),
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
    result["numerical_execution_contract"] = (
        _numerical_execution_contract(verified)
    )
    result["execution_configuration_contract"] = (
        _result_execution_configuration(
            verified,
            outer_experiment_index=index,
        )
    )
    _verify_result_against_plan(
        verified,
        result,
        outer_experiment_index=index,
    )
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
    worker_count: int,
    preflight_receipt: Mapping[str, Any],
    authorization_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    verified = _verified_plan(plan)
    resolved_output_dir = Path(output_dir).resolve()
    _verified_execution_authorization(
        verified,
        preflight_receipt,
        authorization_receipt,
        output_dir=resolved_output_dir,
        worker_count=worker_count,
    )
    destination = _shard_path(
        resolved_output_dir,
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
        _verify_result_against_plan(
            verified,
            payload["result"],
            outer_experiment_index=int(outer_experiment_index),
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
        output_dir=resolved_output_dir,
        worker_count=worker_count,
        preflight_receipt=preflight_receipt,
        authorization_receipt=authorization_receipt,
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
        _verify_result_against_plan(
            verified,
            result,
            outer_experiment_index=index,
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
    preflight_receipt: Mapping[str, Any] | None = None,
    authorization_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute or resume a plan and write a deterministic merge index."""

    verified = _verified_plan(plan)
    execution_started = time.perf_counter()
    if worker_count <= 0:
        raise Gate12C2ShardError("worker_count must be positive")
    destination = Path(output_dir).resolve()
    if preflight_receipt is None or authorization_receipt is None:
        raise Gate12C2ShardError(
            "execution requires an exact no-outcome preflight receipt and "
            "an explicit plan-bound authorization receipt"
        )
    _verified_execution_authorization(
        verified,
        preflight_receipt,
        authorization_receipt,
        output_dir=destination,
        worker_count=worker_count,
    )
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
            _write_or_verify_shard(
                verified,
                str(destination),
                index,
                worker_count,
                preflight_receipt,
                authorization_receipt,
            )
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
                    worker_count,
                    preflight_receipt,
                    authorization_receipt,
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
