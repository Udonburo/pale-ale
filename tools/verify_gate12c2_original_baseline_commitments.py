#!/usr/bin/env python3
"""Independently rederive Gate12C-2 v0.8 baseline commitments."""


from __future__ import annotations

import argparse
import calendar
import ctypes
import ctypes.wintypes
import hashlib
import json
import math
import os
import re
import socket
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence


PLAN_PATH = Path(
    r"C:\Users\aoika\Documents\Research\pale-ale-local\research-program"
    r"\profile-plans\C2_ORIGINAL_BASELINE_COMMITMENT_GATE_PLAN_v0.8_2026-08-01.json"
)
PLAN_FILE_SHA256 = "0a433e92c04e92762aa930647a14c465614fcb2bd11836e63851c617a18a338a"
PLAN_PAYLOAD_SHA256 = "5c2cefc27fd36dbf3343e5095ccfc2645f33a5bb4ce4cf596e0c80905df0d1b6"
ARTIFACT_SURFACE_SHA256 = "5d622ba240ea5dd93d963cb47e94a5ff5c3756b370adf823d5e2446435eba005"
FORMAL_DESIGN_REVIEW_FILE_SHA256 = (
    "57dd2e8ad95c35614f7af62eb3527d87f050260772201300971972fed2a2c04c"
)
FORMAL_DESIGN_REVIEW_PAYLOAD_SHA256 = (
    "83ec774e8fdefa9e2c8bc5463c27fa68bd29a70f09c389c4c58707db40644c62"
)
CONFIGURATION_SURFACE_SHA256 = "a564c25f28e42860f0a1e8f51d4a311b4eae2b771f02dc3f62504547799f19cf"
COMPLETE_SURFACE_SHA256 = "9489e0eb14e33a328167c840a443a80392c022e365bdefd41458c9659aeda6da"
PROTECTED_SURFACE_SHA256 = "a8ef2eb83fbd0517740f5ebbb2c270ba8f4ea37f872b34d137b0447fbb6edc24"
PROTECTED_ROOT = Path(
    r"C:\Users\aoika\Documents\Research\pale-ale-local\throughput"
    r"\C2_DRAW_PROFILE_f9bd14d_2026-07-26"
)
GATE_ID = "C2_ORIGINAL_BASELINE_COMMITMENT_GATE_v0.8"
PASS_LINE = "gate12c2-original-baseline-verification:PASS"
ERROR_PREFIX = "gate12c2-original-baseline-verification:ERROR:"
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
UTC_RE = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})T(?P<time>\d{2}:\d{2}:\d{2})"
    r"(?P<fraction>\.\d{1,9})?(?:Z|\+00:00)\Z"
)
SHARD_SCHEMA = "gate12c2_development_outer_shard_v0.3"
INDEX_SCHEMA = "gate12c2_development_shard_index_v0.3"
PROJECTION_SCHEMA = "gate12c2_development_scientific_projection_v0.3"
SEMANTIC_SCHEMA = "gate12c2_semantic_index_commitment_v0.1"
OUTER_SCHEMA = "gate12c2_outer_experiment_v0.5"
N1_ID = "gate12c2_n1_role_constrained_frame_reassignment_v0.1"
S2_ORIENTATION_ID = (
    "gate12c2_s2_independent_edge_orientation_stress_v0.1"
)
IMPLEMENTATION_ROLES = (
    "extraction_core",
    "implementation_binding_builder",
    "reviewed_authority_builder",
    "preflight_issuer",
    "authorization_issuer",
    "extraction_runner",
    "authorization_verifier",
    "independent_verifier",
    "primary_tests",
    "adversarial_tests",
)


FAILURE_CODES = {
    "AUTHORIZATION_INVALID",
    "COMMITMENT_MISMATCH",
    "CONCURRENT_EXECUTION",
    "DUPLICATE_JSON_KEY",
    "FILE_IDENTITY_CHANGED",
    "FINAL_PATH_OUTSIDE_PROTECTED_ROOT",
    "IMPLEMENTATION_BYTE_IDENTITY_MISMATCH",
    "INPUT_LINEAGE_MISMATCH",
    "INPUT_SCHEMA_INVALID",
    "INTERNAL_SANITIZED_FAILURE",
    "OUTPUT_PUBLICATION_FAILED",
    "PROTECTED_ROOT_REPARSE_POINT",
    "PROTECTED_ROOT_SURFACE_MISMATCH",
    "READ_ONLY_HANDLE_FAILED",
    "ROOT_MUTATION_DETECTED",
    "SCIENTIFIC_OUTPUT_BLOCKED",
    "TERMINAL_CONFLICT",
    "TERMINAL_OUTCOME_RECONSTRUCTION_FAILED",
    "UNEXPECTED_ARTIFACT",
    "VERIFICATION_MISMATCH",
    "ZERO_COVERAGE",
}


class IndependentVerificationError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = (
            code if code in FAILURE_CODES else "INTERNAL_SANITIZED_FAILURE"
        )
        super().__init__(self.code)


def _fail(code: str) -> None:
    raise IndependentVerificationError(code)


def verifier_canonical_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, OverflowError):
        _fail("INPUT_SCHEMA_INVALID")
    return encoded.encode("utf-8")


def verifier_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    target: dict[str, Any] = {}
    for key, value in pairs:
        if key in target:
            _fail("DUPLICATE_JSON_KEY")
        target[key] = value
    return target


def _finite_tree(value: object) -> None:
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            _fail("INPUT_SCHEMA_INVALID")
        return
    if type(value) is list:
        for item in value:
            _finite_tree(item)
        return
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            _fail("INPUT_SCHEMA_INVALID")
        for item in value.values():
            _finite_tree(item)
        return
    _fail("INPUT_SCHEMA_INVALID")


def verifier_json(raw: bytes, *, canonical: bool = True) -> dict[str, Any]:
    if not isinstance(raw, bytes) or raw.startswith(b"\xef\xbb\xbf"):
        _fail("INPUT_SCHEMA_INVALID")
    try:
        decoded = raw.decode("utf-8", "strict")
        value = json.loads(
            decoded,
            object_pairs_hook=_unique_pairs,
            parse_constant=lambda _token: (_ for _ in ()).throw(ValueError()),
        )
    except IndependentVerificationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        _fail("INPUT_SCHEMA_INVALID")
    if type(value) is not dict:
        _fail("INPUT_SCHEMA_INVALID")
    _finite_tree(value)
    if canonical and raw != verifier_canonical_bytes(value):
        _fail("INPUT_SCHEMA_INVALID")
    return value


def _keys(value: Mapping[str, Any], expected: set[str]) -> None:
    if set(value) != expected:
        _fail("INPUT_SCHEMA_INVALID")


def _integer(value: object, maximum: int = (1 << 63) - 1) -> int:
    if type(value) is not int or value < 0 or value > maximum:
        _fail("INPUT_SCHEMA_INVALID")
    return value


def _boolean(value: object) -> bool:
    if type(value) is not bool:
        _fail("INPUT_SCHEMA_INVALID")
    return value


def _digest(value: object) -> str:
    return verifier_sha256(verifier_canonical_bytes(value))


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    claimed = value.get(field)
    if type(claimed) is not str or SHA_RE.fullmatch(claimed) is None:
        _fail("INPUT_SCHEMA_INVALID")
    domain = dict(value)
    del domain[field]
    if _digest(domain) != claimed:
        _fail("INPUT_SCHEMA_INVALID")
    return claimed


def verifier_gzip_json(compressed: bytes) -> dict[str, Any]:
    try:
        stream = zlib.decompressobj(16 + zlib.MAX_WBITS)
        raw = stream.decompress(compressed)
        raw += stream.flush()
    except zlib.error:
        _fail("INPUT_SCHEMA_INVALID")
    if not stream.eof or stream.unused_data or stream.unconsumed_tail:
        _fail("INPUT_SCHEMA_INVALID")
    return verifier_json(raw)


def independent_decision(result: Mapping[str, Any]) -> dict[str, Any]:
    if result.get("regime_id") == "S2_null_inflation":
        count = _integer(result.get("identified_case_count"), 12)
        return {
            "identification_success": _boolean(
                result.get("identification_success")
            ),
            "identified_case_count": count,
            "breadth_pass": _boolean(result.get("breadth_pass")),
        }
    pipeline = result.get("pipeline_decision")
    if type(pipeline) is not dict:
        _fail("INPUT_SCHEMA_INVALID")
    _keys(
        pipeline,
        {
            "schema_version",
            "epistemic_status",
            "outer_monte_carlo_unit",
            "alternative",
            "holm_alpha",
            "zero_tolerance",
            "endpoint_count",
            "q_directional_support_count",
            "any_endpoint_support",
            "supporting_run_count",
            "any_run_support",
            "q_discordant_run_count",
            "grid_outcome",
            "claim_promotion",
            "promotion_outcomes",
            "partial_or_structured_is_promotional",
            "endpoint_rows",
            "run_rows",
        },
    )
    text = pipeline.get("grid_outcome")
    if type(text) is not str or not text or not text.isascii():
        _fail("INPUT_SCHEMA_INVALID")
    return {
        "claim_promotion": _boolean(pipeline.get("claim_promotion")),
        "grid_outcome": text,
        "any_endpoint_support": _boolean(
            pipeline.get("any_endpoint_support")
        ),
        "any_run_support": _boolean(pipeline.get("any_run_support")),
        "q_directional_support_count": _integer(
            pipeline.get("q_directional_support_count"), 24
        ),
        "supporting_run_count": _integer(
            pipeline.get("supporting_run_count"), 12
        ),
        "q_discordant_run_count": _integer(
            pipeline.get("q_discordant_run_count"), 12
        ),
    }


def _check_result(
    plan: Mapping[str, Any], result: Mapping[str, Any], outer_id: int
) -> None:
    non_s2_fields = {
        "schema_version",
        "epistemic_status",
        "contract_version",
        "surface_id",
        "locked_execution_authorized",
        "regime_id",
        "effect_strength",
        "outer_experiment_index",
        "block_count_schedule",
        "inner_valid_draw_count",
        "max_draw_attempts",
        "diagnostic_kernel",
        "accepted_valid_draw_storage",
        "accepted_valid_draw_order",
        "dependency_structure",
        "alternative",
        "case_receipts",
        "endpoint_receipts",
        "pipeline_decision",
        "numerical_execution_contract",
        "execution_configuration_contract",
    }
    s2_fields = {
        "schema_version",
        "epistemic_status",
        "contract_version",
        "surface_id",
        "locked_execution_authorized",
        "regime_id",
        "outer_experiment_index",
        "block_count_schedule",
        "inner_valid_draw_count",
        "max_draw_attempts",
        "diagnostic_kernel",
        "accepted_valid_draw_storage",
        "accepted_valid_draw_order",
        "observed_process_modified",
        "paired_null_arms",
        "identified_case_count",
        "breadth_pass",
        "identification_success",
        "endpoint_rows",
        "case_rows",
        "numerical_execution_contract",
        "execution_configuration_contract",
    }
    _keys(
        result,
        s2_fields
        if plan.get("regime_id") == "S2_null_inflation"
        else non_s2_fields,
    )
    fixed = {
        "schema_version": OUTER_SCHEMA,
        "contract_version": plan.get("contract_version"),
        "surface_id": "development",
        "locked_execution_authorized": False,
        "regime_id": plan.get("regime_id"),
        "outer_experiment_index": outer_id,
        "block_count_schedule": plan.get("block_count_schedule"),
        "inner_valid_draw_count": plan.get("inner_valid_draw_count"),
        "diagnostic_kernel": plan.get("diagnostic_kernel"),
        "accepted_valid_draw_storage": plan.get(
            "accepted_valid_draw_storage"
        ),
    }
    if any(result.get(key) != expected for key, expected in fixed.items()):
        _fail("INPUT_SCHEMA_INVALID")
    if type(result.get("execution_configuration_contract")) is not dict:
        _fail("INPUT_SCHEMA_INVALID")
    execution = result["execution_configuration_contract"]
    inner_count = _integer(plan.get("inner_valid_draw_count"))
    configured_attempts = plan.get("max_draw_attempts")
    if configured_attempts is None:
        resolved_attempts = max(inner_count * 4, inner_count + 8)
    else:
        resolved_attempts = _integer(configured_attempts)
    if resolved_attempts < 1:
        _fail("INPUT_SCHEMA_INVALID")
    execution_fixed = {
        "schema_version": "gate12c2_result_execution_contract_v0.1",
        "plan_payload_sha256": plan.get("plan_payload_sha256"),
        "contract_version": plan.get("contract_version"),
        "surface_id": "development",
        "locked_execution_authorized": False,
        "real_held_out_execution_authorized": False,
        "N2_open": False,
        "N3_open": False,
        "public_claim": False,
        "regime_id": plan.get("regime_id"),
        "outer_experiment_index": outer_id,
        "block_count_schedule": plan.get("block_count_schedule"),
        "inner_valid_draw_count": plan.get("inner_valid_draw_count"),
        "effect_strength": plan.get("effect_strength"),
        "configured_max_draw_attempts": plan.get("max_draw_attempts"),
        "resolved_max_draw_attempts": resolved_attempts,
        "minimum_log_null_inflation": plan.get(
            "minimum_log_null_inflation"
        ),
        "epsilon": plan.get("epsilon"),
        "diagnostic_kernel": plan.get("diagnostic_kernel"),
        "accepted_valid_draw_storage": plan.get(
            "accepted_valid_draw_storage"
        ),
        "outer_experiment_schema": plan.get("outer_experiment_schema"),
        "seed_namespace_schema": plan.get("seed_namespace_schema"),
        "scientific_execution_parameters": plan.get(
            "scientific_execution_parameters"
        ),
        "implementation_sha256": plan.get("implementation_sha256"),
        "master_seed_sha256": verifier_sha256(
            str(plan.get("master_seed")).encode("utf-8")
        ),
        "numerical_environment_sha256": _digest(plan.get("numerical_environment")),
    }
    _keys(execution, set(execution_fixed))
    if any(execution.get(key) != value for key, value in execution_fixed.items()):
        _fail("INPUT_SCHEMA_INVALID")
    if _integer(result.get("max_draw_attempts")) != resolved_attempts:
        _fail("INPUT_SCHEMA_INVALID")
    numerical = result.get("numerical_execution_contract")
    expected_numerical = {
        "blas_thread_limit": 1,
        "thread_environment": {
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
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
    if type(numerical) is not dict or numerical != expected_numerical:
        _fail("INPUT_SCHEMA_INVALID")
    if result.get("effect_strength") != plan.get("effect_strength"):
        _fail("INPUT_SCHEMA_INVALID")
    if plan.get("regime_id") == "S2_null_inflation":
        if (
            result.get("observed_process_modified") is not False
            or result.get("paired_null_arms") != [N1_ID, S2_ORIENTATION_ID]
            or type(result.get("endpoint_rows")) is not list
            or len(result["endpoint_rows"]) != 24
            or type(result.get("case_rows")) is not list
            or len(result["case_rows"]) != 12
        ):
            _fail("INPUT_SCHEMA_INVALID")
        threshold = float(plan["minimum_log_null_inflation"])
        for endpoint in result["endpoint_rows"]:
            if (
                type(endpoint) is not dict
                or endpoint.get("minimum_log_null_inflation") != threshold
            ):
                _fail("INPUT_SCHEMA_INVALID")
    independent_decision(result)


def independent_projection(
    plan: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: row["outer_experiment_index"])
    return {
        "schema_version": PROJECTION_SCHEMA,
        "plan_payload_sha256": plan["plan_payload_sha256"],
        "outer_results": [
            {
                "outer_experiment_index": row["outer_experiment_index"],
                "result_payload_sha256": row["result_payload_sha256"],
                "decision": dict(row["decision"]),
            }
            for row in ordered
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


def independent_configuration_commitment(
    *,
    configuration_id: str,
    subplan: Mapping[str, Any],
    index_raw: bytes,
    shard_raw_by_relative_path: Mapping[str, bytes],
    phase_callback: Callable[[], None] | None = None,
) -> dict[str, Any]:
    index = verifier_json(index_raw)
    permitted_index = {
        "schema_version",
        "epistemic_status",
        "surface_id",
        "locked_execution_authorized",
        "plan_payload_sha256",
        "worker_count_operational_only",
        "merge_order",
        "outer_experiment_count",
        "all_outer_indices_present",
        "shards",
        "scientific_projection_schema_version",
        "scientific_projection_sha256",
        "index_payload_sha256",
    }
    if set(index) not in (
        permitted_index,
        permitted_index | {"operational_execution_metrics"},
    ):
        _fail("INPUT_SCHEMA_INVALID")
    _self_hash(index, "index_payload_sha256")
    if (
        index.get("schema_version") != INDEX_SCHEMA
        or index.get("surface_id") != "development"
        or index.get("locked_execution_authorized") is not False
        or index.get("merge_order") != "ascending_outer_experiment_index"
        or index.get("plan_payload_sha256")
        != subplan.get("plan_payload_sha256")
        or index.get("scientific_projection_schema_version")
        != PROJECTION_SCHEMA
        or index.get("all_outer_indices_present") is not True
    ):
        _fail("INPUT_SCHEMA_INVALID")
    ids_value = subplan.get("outer_experiment_indices")
    if type(ids_value) is not list or not ids_value:
        _fail("ZERO_COVERAGE")
    ids = [_integer(value, (1 << 31) - 1) for value in ids_value]
    if ids != sorted(set(ids)):
        _fail("INPUT_SCHEMA_INVALID")
    indexed = index.get("shards")
    if (
        type(indexed) is not list
        or len(indexed) != len(ids)
        or index.get("outer_experiment_count") != len(ids)
    ):
        _fail("ZERO_COVERAGE")
    expected_paths = {f"shards/outer-{value:06d}.json.gz" for value in ids}
    if set(shard_raw_by_relative_path) != expected_paths:
        _fail("ZERO_COVERAGE")
    projection_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    for position, outer_id in enumerate(ids):
        relative = f"shards/outer-{outer_id:06d}.json.gz"
        compressed = shard_raw_by_relative_path[relative]
        shard = verifier_gzip_json(compressed)
        shard_fields = {
            "schema_version",
            "epistemic_status",
            "surface_id",
            "locked_execution_authorized",
            "plan_payload_sha256",
            "outer_experiment_index",
            "result_payload_sha256",
            "result",
            "shard_payload_sha256",
        }
        _keys(shard, shard_fields)
        _self_hash(shard, "shard_payload_sha256")
        if (
            shard.get("schema_version") != SHARD_SCHEMA
            or shard.get("surface_id") != "development"
            or shard.get("locked_execution_authorized") is not False
            or shard.get("plan_payload_sha256")
            != subplan.get("plan_payload_sha256")
            or shard.get("outer_experiment_index") != outer_id
        ):
            _fail("INPUT_SCHEMA_INVALID")
        result = shard.get("result")
        if type(result) is not dict:
            _fail("INPUT_SCHEMA_INVALID")
        result_hash = _digest(result)
        if shard.get("result_payload_sha256") != result_hash:
            _fail("INPUT_SCHEMA_INVALID")
        _check_result(subplan, result, outer_id)
        decision = independent_decision(result)
        expected_index_values = {
            "outer_experiment_index": outer_id,
            "relative_path": relative,
            "compressed_file_sha256": verifier_sha256(compressed),
            "compressed_bytes": len(compressed),
            "shard_payload_sha256": shard["shard_payload_sha256"],
            "result_payload_sha256": result_hash,
            "decision": decision,
        }
        index_row = indexed[position]
        if type(index_row) is not dict:
            _fail("INPUT_SCHEMA_INVALID")
        row_keys = set(expected_index_values) | {"reused_existing_shard"}
        if set(index_row) not in (
            row_keys,
            row_keys | {"operational_metrics"},
        ):
            _fail("INPUT_SCHEMA_INVALID")
        if type(index_row.get("reused_existing_shard")) is not bool:
            _fail("INPUT_SCHEMA_INVALID")
        if any(
            index_row.get(key) != value
            for key, value in expected_index_values.items()
        ):
            _fail("INPUT_SCHEMA_INVALID")
        projection_rows.append(
            {
                "outer_experiment_index": outer_id,
                "result_payload_sha256": result_hash,
                "decision": decision,
            }
        )
        result_rows.append(
            {
                "outer_experiment_index": outer_id,
                "result_payload_sha256": result_hash,
                "shard_payload_sha256": shard["shard_payload_sha256"],
            }
        )
    if phase_callback is not None:
        phase_callback()
    projection_hash = _digest(
        independent_projection(subplan, projection_rows)
    )
    if index.get("scientific_projection_sha256") != projection_hash:
        _fail("COMMITMENT_MISMATCH")
    outer_hash = _digest(ids)
    result_surface_hash = _digest(result_rows)
    semantic = {
        "schema_version": SEMANTIC_SCHEMA,
        "configuration_id": configuration_id,
        "original_subplan_payload_sha256": subplan["plan_payload_sha256"],
        "outer_experiment_count": len(ids),
        "outer_id_surface_sha256": outer_hash,
        "result_commitment_surface_sha256": result_surface_hash,
        "scientific_projection_sha256": projection_hash,
    }
    return {
        "configuration_id": configuration_id,
        "outer_experiment_count": len(ids),
        "outer_id_surface_sha256": outer_hash,
        "result_commitment_surface_sha256": result_surface_hash,
        "scientific_projection_sha256": projection_hash,
        "semantic_index_commitment_v0_1_sha256": _digest(semantic),
    }



class _VAttr(ctypes.Structure):
    _fields_ = [
        ("attributes", ctypes.wintypes.DWORD),
        ("tag", ctypes.wintypes.DWORD),
    ]


class _VFileId128(ctypes.Structure):
    _fields_ = [("identifier", ctypes.c_ubyte * 16)]


class _VFileId(ctypes.Structure):
    _fields_ = [
        ("volume", ctypes.c_ulonglong),
        ("identifier", _VFileId128),
    ]


class VerifierRetainedSurface:
    """Independent retained-handle reader used only by the verifier."""

    def __init__(
        self,
        root: Path,
        manifest: Mapping[str, Any],
        api: object | None = None,
    ) -> None:
        if os.name != "nt" and api is None:
            _fail("READ_ONLY_HANDLE_FAILED")
        self.root = Path(root)
        self.manifest = dict(manifest)
        self.dll = api if api is not None else ctypes.WinDLL(
            "kernel32", use_last_error=True
        )
        self.root_record: tuple[int, tuple[int, bytes, str, int | None]] | None = None
        self.directory_records: dict[
            str, tuple[int, tuple[int, bytes, str, int | None]]
        ] = {}
        self.file_records: dict[
            str, tuple[int, tuple[int, bytes, str, int | None]]
        ] = {}
        self.file_rows = {
            row["canonical_relative_path"]: dict(row)
            for row in manifest["files"]
        }
        self.directory_rows = {
            row["canonical_relative_path"]: dict(row)
            for row in manifest["directories"]
        }
        self.pre_hashes: dict[str, str] = {}
        self._prototypes()

    def _prototypes(self) -> None:
        prototypes = {
            "CreateFileW": (
                [
                    ctypes.wintypes.LPCWSTR,
                    ctypes.wintypes.DWORD,
                    ctypes.wintypes.DWORD,
                    ctypes.c_void_p,
                    ctypes.wintypes.DWORD,
                    ctypes.wintypes.DWORD,
                    ctypes.wintypes.HANDLE,
                ],
                ctypes.wintypes.HANDLE,
            ),
            "GetFileInformationByHandleEx": (
                [
                    ctypes.wintypes.HANDLE,
                    ctypes.c_int,
                    ctypes.c_void_p,
                    ctypes.wintypes.DWORD,
                ],
                ctypes.wintypes.BOOL,
            ),
            "GetFinalPathNameByHandleW": (
                [
                    ctypes.wintypes.HANDLE,
                    ctypes.wintypes.LPWSTR,
                    ctypes.wintypes.DWORD,
                    ctypes.wintypes.DWORD,
                ],
                ctypes.wintypes.DWORD,
            ),
            "GetFileSizeEx": (
                [ctypes.wintypes.HANDLE, ctypes.POINTER(ctypes.c_longlong)],
                ctypes.wintypes.BOOL,
            ),
            "SetFilePointerEx": (
                [
                    ctypes.wintypes.HANDLE,
                    ctypes.c_longlong,
                    ctypes.POINTER(ctypes.c_longlong),
                    ctypes.wintypes.DWORD,
                ],
                ctypes.wintypes.BOOL,
            ),
            "ReadFile": (
                [
                    ctypes.wintypes.HANDLE,
                    ctypes.c_void_p,
                    ctypes.wintypes.DWORD,
                    ctypes.POINTER(ctypes.wintypes.DWORD),
                    ctypes.c_void_p,
                ],
                ctypes.wintypes.BOOL,
            ),
            "CloseHandle": (
                [ctypes.wintypes.HANDLE],
                ctypes.wintypes.BOOL,
            ),
        }
        for name, (arguments, result) in prototypes.items():
            function = getattr(self.dll, name)
            try:
                function.argtypes = arguments
                function.restype = result
            except Exception:
                pass

    def _open_handle(self, path: Path, directory: bool) -> int:
        desired = 0x0001 | 0x0080 if directory else 0x80000000
        flags = 0x00200000 | (0x02000000 if directory else 0x08000000)
        raw = self.dll.CreateFileW(
            str(path), desired, 0x1, None, 3, flags, None
        )
        handle = int(
            raw
            if isinstance(raw, int)
            else ctypes.cast(raw, ctypes.c_void_p).value or 0
        )
        if handle in {0, ctypes.c_void_p(-1).value}:
            _fail("READ_ONLY_HANDLE_FAILED")
        return handle

    def _metadata(
        self, handle: int, expected: Path, directory: bool
    ) -> tuple[int, bytes, str, int | None]:
        attr = _VAttr()
        if not self.dll.GetFileInformationByHandleEx(
            handle, 9, ctypes.byref(attr), ctypes.sizeof(attr)
        ):
            _fail("READ_ONLY_HANDLE_FAILED")
        if attr.attributes & 0x400 or attr.tag:
            _fail("PROTECTED_ROOT_REPARSE_POINT")
        identity = _VFileId()
        if not self.dll.GetFileInformationByHandleEx(
            handle, 18, ctypes.byref(identity), ctypes.sizeof(identity)
        ):
            _fail("READ_ONLY_HANDLE_FAILED")
        needed = self.dll.GetFinalPathNameByHandleW(handle, None, 0, 0)
        if not needed:
            _fail("READ_ONLY_HANDLE_FAILED")
        buffer = ctypes.create_unicode_buffer(int(needed) + 1)
        copied = self.dll.GetFinalPathNameByHandleW(
            handle, buffer, len(buffer), 0
        )
        if not copied or copied >= len(buffer):
            _fail("READ_ONLY_HANDLE_FAILED")
        final = buffer.value
        upper = final.upper()
        if upper.startswith(
            ("\\\\?\\UNC\\", "\\\\.\\", "\\\\?\\VOLUME", "\\\\?\\GLOBALROOT")
        ):
            _fail("FINAL_PATH_OUTSIDE_PROTECTED_ROOT")
        if final.startswith("\\\\?\\") and len(final) > 6 and final[4].isalpha():
            final = final[4:]
        elif final.startswith("\\\\?\\"):
            _fail("FINAL_PATH_OUTSIDE_PROTECTED_ROOT")
        if final.casefold() != str(expected).casefold():
            _fail("FINAL_PATH_OUTSIDE_PROTECTED_ROOT")
        size_value: int | None = None
        if not directory:
            size = ctypes.c_longlong()
            if not self.dll.GetFileSizeEx(handle, ctypes.byref(size)):
                _fail("READ_ONLY_HANDLE_FAILED")
            size_value = int(size.value)
            if size_value < 0:
                _fail("READ_ONLY_HANDLE_FAILED")
        return (
            int(identity.volume),
            bytes(identity.identifier.identifier),
            final,
            size_value,
        )

    def acquire(self) -> "VerifierRetainedSurface":
        root_handle = self._open_handle(self.root, True)
        try:
            root_metadata = self._metadata(root_handle, self.root, True)
            self.root_record = (root_handle, root_metadata)
            for relative in sorted(self.directory_rows):
                path = self.root.joinpath(*PurePosixPath(relative).parts)
                handle = self._open_handle(path, True)
                metadata = self._metadata(handle, path, True)
                if metadata[0] != root_metadata[0]:
                    _fail("FINAL_PATH_OUTSIDE_PROTECTED_ROOT")
                self.directory_records[relative] = (handle, metadata)
            for relative in sorted(self.file_rows):
                path = self.root.joinpath(*PurePosixPath(relative).parts)
                handle = self._open_handle(path, False)
                metadata = self._metadata(handle, path, False)
                if (
                    metadata[0] != root_metadata[0]
                    or metadata[3]
                    != self.file_rows[relative]["file_size_bytes"]
                ):
                    _fail("FILE_IDENTITY_CHANGED")
                self.file_records[relative] = (handle, metadata)
        except Exception:
            self.close()
            raise
        if (
            len(self.directory_records) != 23
            or len(self.file_records) != 791
        ):
            self.close()
            _fail("ZERO_COVERAGE")
        return self

    def _read(self, record: tuple[int, tuple[int, bytes, str, int | None]]) -> bytes:
        handle, metadata = record
        size = metadata[3]
        if size is None or not self.dll.SetFilePointerEx(handle, 0, None, 0):
            _fail("READ_ONLY_HANDLE_FAILED")
        pieces: list[bytes] = []
        remaining = size
        while remaining:
            amount = min(remaining, 1024 * 1024)
            buffer = ctypes.create_string_buffer(amount)
            obtained = ctypes.wintypes.DWORD()
            if not self.dll.ReadFile(
                handle,
                buffer,
                amount,
                ctypes.byref(obtained),
                None,
            ):
                _fail("READ_ONLY_HANDLE_FAILED")
            count = int(obtained.value)
            if not 0 < count <= amount:
                _fail("READ_ONLY_HANDLE_FAILED")
            pieces.append(buffer.raw[:count])
            remaining -= count
        return b"".join(pieces)

    def bytes_for(self, relative: str) -> bytes:
        if (
            not isinstance(relative, str)
            or "\\" in relative
            or ":" in relative
            or relative.startswith("/")
            or any(
                part in {"", ".", ".."}
                for part in PurePosixPath(relative).parts
            )
        ):
            _fail("INPUT_LINEAGE_MISMATCH")
        record = self.file_records.get(relative)
        if record is None:
            _fail("UNEXPECTED_ARTIFACT")
        return self._read(record)

    def _enumerate(self) -> tuple[set[str], set[str]]:
        files: set[str] = set()
        directories: set[str] = set()
        try:
            for current, names, filenames in os.walk(
                self.root, topdown=True, followlinks=False
            ):
                base = Path(current)
                for name in names:
                    candidate = base / name
                    if candidate.is_symlink():
                        _fail("PROTECTED_ROOT_REPARSE_POINT")
                    directories.add(candidate.relative_to(self.root).as_posix())
                for name in filenames:
                    candidate = base / name
                    if candidate.is_symlink():
                        _fail("PROTECTED_ROOT_REPARSE_POINT")
                    files.add(candidate.relative_to(self.root).as_posix())
        except IndependentVerificationError:
            raise
        except OSError:
            _fail("PROTECTED_ROOT_SURFACE_MISMATCH")
        return files, directories

    def pre_manifest(self) -> dict[str, str]:
        files, directories = self._enumerate()
        if (
            files != set(self.file_records)
            or directories != set(self.directory_records)
        ):
            _fail("PROTECTED_ROOT_SURFACE_MISMATCH")
        for relative in sorted(self.file_records):
            digest = verifier_sha256(self._read(self.file_records[relative]))
            if digest != self.file_rows[relative]["sha256"]:
                _fail("PROTECTED_ROOT_SURFACE_MISMATCH")
            self.pre_hashes[relative] = digest
        if len(self.pre_hashes) != 791:
            _fail("ZERO_COVERAGE")
        return {
            "complete": COMPLETE_SURFACE_SHA256,
            "protected": PROTECTED_SURFACE_SHA256,
        }

    def post_manifest(self) -> dict[str, str]:
        files, directories = self._enumerate()
        if (
            files != set(self.file_records)
            or directories != set(self.directory_records)
        ):
            _fail("ROOT_MUTATION_DETECTED")
        records: list[
            tuple[Path, tuple[int, tuple[int, bytes, str, int | None]], bool]
        ] = []
        if self.root_record is not None:
            records.append((self.root, self.root_record, True))
        records.extend(
            (
                self.root.joinpath(*PurePosixPath(relative).parts),
                record,
                True,
            )
            for relative, record in self.directory_records.items()
        )
        records.extend(
            (
                self.root.joinpath(*PurePosixPath(relative).parts),
                record,
                False,
            )
            for relative, record in self.file_records.items()
        )
        for path, record, directory in records:
            if self._metadata(record[0], path, directory) != record[1]:
                _fail("FILE_IDENTITY_CHANGED")
        return {
            "complete": COMPLETE_SURFACE_SHA256,
            "protected": PROTECTED_SURFACE_SHA256,
        }

    def close(self) -> None:
        handles = [record[0] for record in self.file_records.values()]
        handles += [record[0] for record in self.directory_records.values()]
        if self.root_record is not None:
            handles.append(self.root_record[0])
        for handle in reversed(handles):
            try:
                self.dll.CloseHandle(handle)
            except Exception:
                pass
        self.file_records.clear()
        self.directory_records.clear()
        self.root_record = None

    def __enter__(self) -> "VerifierRetainedSurface":
        return self.acquire()

    def __exit__(self, _kind: object, _value: object, _trace: object) -> None:
        self.close()



def independent_load_plan() -> dict[str, Any]:
    try:
        raw = PLAN_PATH.read_bytes()
    except OSError:
        _fail("INPUT_LINEAGE_MISMATCH")
    if verifier_sha256(raw) != PLAN_FILE_SHA256:
        _fail("INPUT_LINEAGE_MISMATCH")
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        _fail("INPUT_LINEAGE_MISMATCH")
    plan = verifier_json(raw[:-1])
    if raw != verifier_canonical_bytes(plan) + b"\n":
        _fail("INPUT_LINEAGE_MISMATCH")
    if _self_hash(plan, "plan_payload_sha256") != PLAN_PAYLOAD_SHA256:
        _fail("INPUT_LINEAGE_MISMATCH")
    if plan.get("schema_version") != (
        "gate12c2_original_baseline_commitment_gate_plan_v0.8"
    ):
        _fail("INPUT_LINEAGE_MISMATCH")
    artifact_rows = plan.get("artifact_path_surface")
    configurations = plan.get("configuration_surface")
    if (
        type(artifact_rows) is not list
        or len(artifact_rows) != 18
        or _digest(artifact_rows) != ARTIFACT_SURFACE_SHA256
        or plan.get("artifact_path_surface_sha256")
        != ARTIFACT_SURFACE_SHA256
        or type(configurations) is not list
        or len(configurations) != 9
        or _digest(configurations) != CONFIGURATION_SURFACE_SHA256
    ):
        _fail("INPUT_LINEAGE_MISMATCH")
    return plan


def _frozen_json(
    path: Path, file_hash: str, payload_hash: str
) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError:
        _fail("INPUT_LINEAGE_MISMATCH")
    if (
        verifier_sha256(raw) != file_hash
        or not raw.endswith(b"\n")
        or raw.endswith(b"\n\n")
    ):
        _fail("INPUT_LINEAGE_MISMATCH")
    payload = verifier_json(raw[:-1])
    candidates = []
    for field, value in payload.items():
        if value != payload_hash or not field.endswith("sha256"):
            continue
        domain = dict(payload)
        del domain[field]
        encoded = verifier_canonical_bytes(domain)
        if verifier_sha256(encoded) == payload_hash:
            candidates.append(field)
        if verifier_sha256(encoded + b"\n") == payload_hash:
            candidates.append(field)
    if len(candidates) != 1:
        _fail("INPUT_LINEAGE_MISMATCH")
    return payload


def independent_lineage(
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    lineage = plan["original_input_lineage"]
    original = _frozen_json(
        Path(lineage["original_plan_path"]),
        lineage["original_plan_file_sha256"],
        lineage["original_plan_payload_sha256"],
    )
    manifest = _frozen_json(
        Path(lineage["incident_manifest_path"]),
        lineage["incident_manifest_file_sha256"],
        lineage["incident_manifest_payload_sha256"],
    )
    sealed_receipts = []
    for prefix in (
        "payload_seal",
        "payload_seal_verification",
        "formal_payload_closeout",
    ):
        sealed_receipts.append(
            _frozen_json(
                Path(lineage[f"{prefix}_path"]),
                lineage[f"{prefix}_file_sha256"],
                lineage[f"{prefix}_payload_sha256"],
            )
        )

    def recursively_named(value: object, name: str) -> list[object]:
        found: list[object] = []
        if type(value) is dict:
            for key, item in value.items():
                if key == name:
                    found.append(item)
                found.extend(recursively_named(item, name))
        elif type(value) is list:
            for item in value:
                found.extend(recursively_named(item, name))
        return found

    for receipt in sealed_receipts:
        for field in (
            "scientific_values_emitted",
            "stability_analysis_authorized",
            "locked_execution_authorized",
            "real_held_out_execution_authorized",
            "N2_open",
            "N3_open",
        ):
            if any(value is not False for value in recursively_named(receipt, field)):
                _fail("INPUT_LINEAGE_MISMATCH")
    if (
        manifest.get("output_root")
        != str(PROTECTED_ROOT).replace("\\", "/")
        or manifest.get("state") != "INCIDENT_FROZEN"
        or manifest.get("scientific_values_emitted") is not False
    ):
        _fail("INPUT_LINEAGE_MISMATCH")
    files = manifest.get("files")
    directories = manifest.get("directories")
    if (
        type(files) is not list
        or len(files) != 791
        or type(directories) is not list
        or len(directories) != 23
    ):
        _fail("INPUT_LINEAGE_MISMATCH")
    paths: list[str] = []
    protected = []
    for row in files:
        if type(row) is not dict:
            _fail("INPUT_LINEAGE_MISMATCH")
        relative = row.get("canonical_relative_path")
        if (
            type(relative) is not str
            or "\\" in relative
            or ":" in relative
            or relative.startswith("/")
            or any(
                part in {"", ".", ".."}
                for part in PurePosixPath(relative).parts
            )
            or row.get("exists") is not True
            or row.get("expected") is not True
            or row.get("unexpected") is not False
            or row.get("partial_or_temp") is not False
            or row.get("reparse_point") is not False
            or type(row.get("file_size_bytes")) is not int
            or type(row.get("sha256")) is not str
            or SHA_RE.fullmatch(row["sha256"]) is None
        ):
            _fail("INPUT_LINEAGE_MISMATCH")
        paths.append(relative)
        if row.get("plane") == "protected_payload":
            protected.append(row)
    if (
        paths != sorted(paths)
        or len(set(paths)) != 791
        or len(protected) != 790
        or _digest(files) != COMPLETE_SURFACE_SHA256
        or _digest(protected) != PROTECTED_SURFACE_SHA256
    ):
        _fail("INPUT_LINEAGE_MISMATCH")
    directory_paths = [
        row.get("canonical_relative_path") for row in directories
    ]
    if (
        any(type(value) is not str for value in directory_paths)
        or directory_paths != sorted(directory_paths)
        or len(set(directory_paths)) != 23
        or any(
            row.get("expected") is not True
            or row.get("unexpected") is not False
            or row.get("reparse_point") is not False
            for row in directories
        )
    ):
        _fail("INPUT_LINEAGE_MISMATCH")
    return original, manifest


def new_verifier_progress() -> dict[str, Any]:
    return {
        "source_state": "VERIFIER_EXECUTION_CLAIMED",
        "failure_phase": "verifier_lineage_reverification",
        "evidence": {
            "pre_complete_surface_sha256": None,
            "pre_protected_surface_sha256": None,
            "post_complete_surface_sha256": None,
            "post_protected_surface_sha256": None,
            "recomputed_baseline_commitment_surface_sha256": None,
        },
        "configuration_count_reached": 0,
        "outer_experiment_count_reached": 0,
        "shard_count_reached": 0,
        "index_count_reached": 0,
    }


def update_verifier_progress(
    progress: dict[str, Any] | None,
    *,
    source_state: str | None = None,
    failure_phase: str | None = None,
    evidence: Mapping[str, str | None] | None = None,
    configuration_count_reached: int | None = None,
    outer_experiment_count_reached: int | None = None,
    shard_count_reached: int | None = None,
    index_count_reached: int | None = None,
) -> None:
    if progress is None:
        return
    if not progress:
        progress.update(new_verifier_progress())
    if source_state is not None:
        progress["source_state"] = source_state
    if failure_phase is not None:
        progress["failure_phase"] = failure_phase
    if evidence is not None:
        current = progress.get("evidence")
        if type(current) is not dict:
            _fail("INTERNAL_SANITIZED_FAILURE")
        current.update(evidence)
    for key, value in (
        ("configuration_count_reached", configuration_count_reached),
        ("outer_experiment_count_reached", outer_experiment_count_reached),
        ("shard_count_reached", shard_count_reached),
        ("index_count_reached", index_count_reached),
    ):
        if value is not None:
            progress[key] = value


def independent_rederive(
    plan: Mapping[str, Any],
    original_plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    progress: dict[str, Any] | None = None,
) -> dict[str, Any]:
    update_verifier_progress(
        progress,
        source_state="VERIFIER_INPUT_LOCKING",
        failure_phase="verifier_input_locking",
    )
    with VerifierRetainedSurface(PROTECTED_ROOT, manifest) as surface:
        update_verifier_progress(
            progress,
            source_state="VERIFIER_INPUT_HANDLES_LOCKED",
            failure_phase="verifier_pre_manifest",
        )
        pre = surface.pre_manifest()
        update_verifier_progress(
            progress,
            source_state="VERIFIER_PRE_MANIFEST_VERIFIED",
            failure_phase="verifier_rederivation",
            evidence={
                "pre_complete_surface_sha256": pre["complete"],
                "pre_protected_surface_sha256": pre["protected"],
            },
        )
        if surface.bytes_for("plan.json") != verifier_canonical_bytes(
            original_plan
        ):
            _fail("INPUT_LINEAGE_MISMATCH")
        original_rows = original_plan.get("configurations")
        if type(original_rows) is not list or len(original_rows) != 9:
            _fail("INPUT_SCHEMA_INVALID")
        original_by_id = {
            row["configuration_id"]: row
            for row in original_rows
            if type(row) is dict and type(row.get("configuration_id")) is str
        }
        if len(original_by_id) != 9:
            _fail("INPUT_SCHEMA_INVALID")
        update_verifier_progress(
            progress,
            source_state="VERIFIER_RUNNING",
            failure_phase="verifier_rederivation",
        )
        commitments = []
        total_outer = total_shards = total_indices = 0
        for frozen in plan["configuration_surface"]:
            identifier = frozen["configuration_id"]
            configuration = original_by_id.get(identifier)
            if (
                configuration is None
                or configuration.get("output_relative_path")
                != frozen["output_relative_path"]
                or configuration.get("draw_count") != frozen["draw_count"]
                or configuration.get("regime_id") != frozen["regime_id"]
            ):
                _fail("INPUT_LINEAGE_MISMATCH")
            subplan = configuration.get("subplan")
            if type(subplan) is not dict:
                _fail("INPUT_SCHEMA_INVALID")
            _self_hash(subplan, "plan_payload_sha256")
            if (
                subplan["plan_payload_sha256"]
                != frozen["original_subplan_payload_sha256"]
            ):
                _fail("INPUT_LINEAGE_MISMATCH")
            prefix = frozen["output_relative_path"]
            if surface.bytes_for(
                f"{prefix}/plan.json"
            ) != verifier_canonical_bytes(subplan):
                _fail("INPUT_LINEAGE_MISMATCH")
            ids = subplan.get("outer_experiment_indices")
            if type(ids) is not list:
                _fail("ZERO_COVERAGE")
            shards = {
                f"shards/outer-{_integer(outer_id):06d}.json.gz": (
                    surface.bytes_for(
                        f"{prefix}/shards/outer-{_integer(outer_id):06d}.json.gz"
                    )
                )
                for outer_id in ids
            }
            row = independent_configuration_commitment(
                configuration_id=identifier,
                subplan=subplan,
                index_raw=surface.bytes_for(f"{prefix}/index.json"),
                shard_raw_by_relative_path=shards,
                phase_callback=lambda: update_verifier_progress(
                    progress, failure_phase="verifier_rederivation"
                ),
            )
            if (
                row["outer_experiment_count"]
                != frozen["outer_experiment_count"]
            ):
                _fail("ZERO_COVERAGE")
            commitments.append(row)
            total_outer += row["outer_experiment_count"]
            total_shards += len(shards)
            total_indices += 1
            update_verifier_progress(
                progress,
                configuration_count_reached=len(commitments),
                outer_experiment_count_reached=total_outer,
                shard_count_reached=total_shards,
                index_count_reached=total_indices,
            )
        commitments.sort(key=lambda row: row["configuration_id"])
        if (
            len(commitments),
            total_outer,
            total_shards,
            total_indices,
        ) != (9, 768, 768, 9):
            _fail("ZERO_COVERAGE")
        recomputed_surface = _digest(commitments)
        update_verifier_progress(
            progress,
            source_state="VERIFIER_COMMITMENTS_REDERIVED_QUARANTINED",
            failure_phase="verifier_post_manifest",
            evidence={
                "recomputed_baseline_commitment_surface_sha256": (
                    recomputed_surface
                )
            },
        )
        post = surface.post_manifest()
    update_verifier_progress(
        progress,
        source_state="VERIFIER_POST_MANIFEST_VERIFIED",
        failure_phase="verifier_terminal_outcome_reconstruction",
        evidence={
            "post_complete_surface_sha256": post["complete"],
            "post_protected_surface_sha256": post["protected"],
        },
    )
    return {
        "configuration_count": 9,
        "outer_experiment_count": 768,
        "shard_count": 768,
        "index_count": 9,
        "configuration_commitments": commitments,
        "baseline_commitment_surface_sha256": recomputed_surface,
        "pre_complete_surface_sha256": pre["complete"],
        "pre_protected_surface_sha256": pre["protected"],
        "post_complete_surface_sha256": post["complete"],
        "post_protected_surface_sha256": post["protected"],
    }


def compare_baseline_receipt(
    plan: Mapping[str, Any],
    baseline: Mapping[str, Any],
    rederived: Mapping[str, Any],
) -> None:
    expected = {
        "configuration_count": 9,
        "outer_experiment_count": 768,
        "shard_count": 768,
        "index_count": 9,
        "configuration_commitments": rederived[
            "configuration_commitments"
        ],
        "baseline_commitment_surface_sha256": rederived[
            "baseline_commitment_surface_sha256"
        ],
        "pre_complete_surface_sha256": COMPLETE_SURFACE_SHA256,
        "post_complete_surface_sha256": COMPLETE_SURFACE_SHA256,
        "pre_protected_surface_sha256": PROTECTED_SURFACE_SHA256,
        "post_protected_surface_sha256": PROTECTED_SURFACE_SHA256,
        "configuration_surface_sha256": CONFIGURATION_SURFACE_SHA256,
        "scientific_values_emitted": False,
        "stability_analysis_authorized": False,
    }
    if any(baseline.get(key) != value for key, value in expected.items()):
        _fail("VERIFICATION_MISMATCH")



def _receipt(
    path: Path,
    exact_fields: Sequence[str],
    hash_field: str,
) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
    except OSError:
        _fail("INPUT_LINEAGE_MISMATCH")
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        _fail("INPUT_LINEAGE_MISMATCH")
    payload = verifier_json(raw[:-1])
    if set(payload) != set(exact_fields):
        _fail("INPUT_LINEAGE_MISMATCH")
    _self_hash(payload, hash_field)
    if raw != verifier_canonical_bytes(payload) + b"\n":
        _fail("INPUT_LINEAGE_MISMATCH")
    return payload, verifier_sha256(raw)


def _git_blob(value: bytes, algorithm: str) -> str:
    if algorithm not in {"sha1", "sha256"}:
        _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    digest = hashlib.new(algorithm)
    digest.update(f"blob {len(value)}\0".encode("ascii"))
    digest.update(value)
    return digest.hexdigest()


def independent_runtime_lineage(
    plan: Mapping[str, Any], repository: Path
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    try:
        contract_raw = Path(plan["contract_path"]).read_bytes()
    except OSError:
        _fail("INPUT_LINEAGE_MISMATCH")
    if (
        verifier_sha256(contract_raw) != plan["contract_file_sha256"]
        or plan["contract_file_sha256"]
        != "80cf61544a492e824865b5b5612b29f341b2052d20e6fb1eef169351b6456b46"
    ):
        _fail("INPUT_LINEAGE_MISMATCH")
    formal_schema = plan["review_receipt_schemas"][
        "formal_design_review_verdict"
    ]
    formal, formal_file_hash = _receipt(
        Path(
            plan["implementation_binding_contract"][
                "formal_design_review_path"
            ]
        ),
        formal_schema["exact_top_level_fields"],
        "formal_design_review_payload_sha256",
    )
    formal_required = formal_schema["outcomes"]["pass"]["required_values"]
    if (
        formal_file_hash != FORMAL_DESIGN_REVIEW_FILE_SHA256
        or formal.get("formal_design_review_payload_sha256")
        != FORMAL_DESIGN_REVIEW_PAYLOAD_SHA256
        or type(formal.get("P0_count")) is not int
        or formal["P0_count"] != 0
        or type(formal.get("P1_count")) is not int
        or formal["P1_count"] != 0
        or any(
            formal.get(key) != expected
            for key, expected in formal_required.items()
        )
    ):
        _fail("INPUT_LINEAGE_MISMATCH")
    for row in plan["upstream_authority"]["artifact_rows"]:
        path = Path(row["path"])
        try:
            raw = path.read_bytes()
        except OSError:
            _fail("INPUT_LINEAGE_MISMATCH")
        if verifier_sha256(raw) != row["file_sha256"]:
            _fail("INPUT_LINEAGE_MISMATCH")
        if row["payload_sha256"] is not None:
            upstream_payload = _frozen_json(
                path, row["file_sha256"], row["payload_sha256"]
            )
            if upstream_payload.get("schema_version") != row["schema_version"]:
                _fail("INPUT_LINEAGE_MISMATCH")
    binding_contract = plan["implementation_binding_contract"]
    candidate, candidate_file_hash = _receipt(
        Path(binding_contract["artifact_path"]),
        binding_contract["exact_top_level_fields"],
        "implementation_candidate_binding_payload_sha256",
    )
    fixed_candidate = {
        "schema_version": binding_contract["schema_version"],
        "plan_file_sha256": PLAN_FILE_SHA256,
        "plan_payload_sha256": PLAN_PAYLOAD_SHA256,
        "artifact_path_surface_sha256": ARTIFACT_SURFACE_SHA256,
        "implementation_context_blindness_machine_authenticated": False,
        "current_exposed_design_context_authored_final_bytes": False,
        "task_identity_used_as_machine_authority": False,
        "implementation_authorship_machine_verified": False,
        "protected_payload_access_required_for_implementation": False,
    }
    fixed_candidate.update(binding_contract["required_values"])
    fixed_candidate.update(
        {
            "contract_file_sha256": plan["contract_file_sha256"],
            "formal_design_review_file_sha256": (
                FORMAL_DESIGN_REVIEW_FILE_SHA256
            ),
            "formal_design_review_payload_sha256": (
                FORMAL_DESIGN_REVIEW_PAYLOAD_SHA256
            ),
            "implementation_author_separation_contract_sha256": plan[
                "implementation_author_separation_contract_sha256"
            ],
            "implementation_trust_model_sha256": plan[
                "implementation_trust_model_sha256"
            ],
            "worktree_clean": True,
            "core_autocrlf": False,
            "core_longpaths": True,
        }
    )
    if any(
        candidate.get(key) != value
        for key, value in fixed_candidate.items()
    ):
        _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    object_format = candidate.get("git_object_format")
    source_commit = candidate.get("source_commit")
    if (
        type(source_commit) is not str
        or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
        or object_format not in {"sha1", "sha256"}
    ):
        _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    implementation_rows = candidate.get("implementation_files")
    if (
        type(implementation_rows) is not list
        or len(implementation_rows) != 10
        or implementation_rows
        != sorted(implementation_rows, key=lambda row: row.get("role", ""))
    ):
        _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    roles: set[str] = set()
    paths = set()
    for row in implementation_rows:
        if type(row) is not dict:
            _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        if set(row) != set(binding_contract["exact_implementation_row_fields"]):
            _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        role = row.get("role")
        if (
            type(role) is not str
            or not role
            or not role.isascii()
            or role in roles
        ):
            _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        roles.add(role)
        relative = row.get("relative_path")
        if (
            type(relative) is not str
            or relative in paths
            or "\\" in relative
            or ":" in relative
            or relative.startswith("/")
            or any(
                part in {"", ".", ".."}
                for part in PurePosixPath(relative).parts
            )
            or dict(
                zip(
                    plan["bounded_implementation_scope_after_fresh_design_pass"],
                    IMPLEMENTATION_ROLES,
                )
            ).get(relative)
            != role
        ):
            _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        paths.add(relative)
        try:
            raw = (repository / relative).read_bytes()
        except OSError:
            _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        if (
            verifier_sha256(raw) != row.get("file_sha256")
            or _git_blob(raw, object_format) != row.get("git_blob_oid")
        ):
            _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    if paths != set(
        plan["bounded_implementation_scope_after_fresh_design_pass"]
    ):
        _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    if candidate.get("scientific_dependencies") != binding_contract[
        "scientific_dependencies"
    ]:
        _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    for row in candidate["scientific_dependencies"]:
        try:
            raw = (repository / row["relative_path"]).read_bytes()
        except OSError:
            _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        if (
            verifier_sha256(raw) != row["file_sha256"]
            or _git_blob(raw, object_format) != row["git_blob_oid"]
        ):
            _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    restore = candidate.get("clean_restore")
    if (
        type(restore) is not dict
        or set(restore) != set(binding_contract["clean_restore_exact_fields"])
        or any(
            restore.get(key) != expected
            for key, expected in binding_contract[
                "clean_restore_required_values"
            ].items()
        )
        or restore.get("restore_head") != source_commit
        or type(restore.get("bundle_path")) is not str
        or not restore["bundle_path"]
        or type(restore.get("bundle_size_bytes")) is not int
        or restore["bundle_size_bytes"] < 1
        or any(
            type(restore.get(field)) is not str
            or SHA_RE.fullmatch(restore[field]) is None
            for field in (
                "bundle_file_sha256",
                "restore_receipt_file_sha256",
                "restore_receipt_payload_sha256",
            )
        )
    ):
        _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    try:
        bundle_raw = Path(restore["bundle_path"]).read_bytes()
    except OSError:
        _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    if (
        len(bundle_raw) != restore["bundle_size_bytes"]
        or verifier_sha256(bundle_raw) != restore["bundle_file_sha256"]
    ):
        _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    review_schema = plan["review_receipt_schemas"][
        "fresh_implementation_review_verdict"
    ]
    review, review_file_hash = _receipt(
        Path(review_schema["artifact_path"]),
        review_schema["exact_top_level_fields"],
        "fresh_implementation_review_payload_sha256",
    )
    if (
        review.get("outcome_kind") != "pass"
        or type(review.get("P0_count")) is not int
        or review["P0_count"] != 0
        or type(review.get("P1_count")) is not int
        or review["P1_count"] != 0
        or review.get("implementation_candidate_binding_file_sha256")
        != candidate_file_hash
        or review.get("implementation_candidate_binding_payload_sha256")
        != candidate["implementation_candidate_binding_payload_sha256"]
        or review.get("implementation_source_commit")
        != candidate["source_commit"]
    ):
        _fail("INPUT_LINEAGE_MISMATCH")
    review_required = review_schema["outcomes"]["pass"]["required_values"]
    review_expected = {
        "implementation_author_separation_contract_sha256": plan[
            "implementation_author_separation_contract_sha256"
        ],
        "formal_design_review_file_sha256": FORMAL_DESIGN_REVIEW_FILE_SHA256,
        "formal_design_review_payload_sha256": (
            FORMAL_DESIGN_REVIEW_PAYLOAD_SHA256
        ),
        "implementation_trust_model_sha256": plan[
            "implementation_trust_model_sha256"
        ],
        "bundle_file_sha256": restore["bundle_file_sha256"],
        "restore_receipt_file_sha256": restore[
            "restore_receipt_file_sha256"
        ],
        "restore_receipt_payload_sha256": restore[
            "restore_receipt_payload_sha256"
        ],
    }
    if (
        any(
            review.get(key) != expected
            for key, expected in review_required.items()
        )
        or any(
            review.get(key) != expected
            for key, expected in review_expected.items()
        )
        or type(review.get("P2_count")) is not int
        or review["P2_count"] < 0
        or type(review.get("implementation_review_packet_file_sha256"))
        is not str
        or SHA_RE.fullmatch(
            review["implementation_review_packet_file_sha256"]
        )
        is None
    ):
        _fail("INPUT_LINEAGE_MISMATCH")
    try:
        _parse_ns(review.get("reviewed_at_utc"))
    except IndependentVerificationError:
        _fail("INPUT_LINEAGE_MISMATCH")
    authority_contract = plan["reviewed_implementation_authority_contract"]
    authority, authority_file_hash = _receipt(
        Path(authority_contract["artifact_path"]),
        authority_contract["exact_top_level_fields"],
        "reviewed_implementation_authority_payload_sha256",
    )
    expected_authority = {
        "schema_version": authority_contract["schema_version"],
        "authority_id": authority_contract["authority_id_value"],
        "state": authority_contract["state"],
        "authority_derivation_domain": authority_contract[
            "authority_derivation_domain"
        ],
        "implementation_source_commit": candidate["source_commit"],
        "contract_file_sha256": plan["contract_file_sha256"],
        "plan_file_sha256": PLAN_FILE_SHA256,
        "plan_payload_sha256": PLAN_PAYLOAD_SHA256,
        "implementation_author_separation_contract_sha256": plan[
            "implementation_author_separation_contract_sha256"
        ],
        "implementation_author_separation_basis": authority_contract[
            "implementation_author_separation_basis"
        ],
        "implementation_context_blindness_machine_authenticated": False,
        "implementation_trust_model_sha256": plan[
            "implementation_trust_model_sha256"
        ],
        "artifact_path_surface_sha256": ARTIFACT_SURFACE_SHA256,
        "formal_design_review_file_sha256": (
            FORMAL_DESIGN_REVIEW_FILE_SHA256
        ),
        "formal_design_review_payload_sha256": (
            FORMAL_DESIGN_REVIEW_PAYLOAD_SHA256
        ),
        "implementation_candidate_binding_file_sha256": candidate_file_hash,
        "implementation_candidate_binding_payload_sha256": candidate[
            "implementation_candidate_binding_payload_sha256"
        ],
        "fresh_implementation_review_file_sha256": review_file_hash,
        "fresh_implementation_review_payload_sha256": review[
            "fresh_implementation_review_payload_sha256"
        ],
        "task_identity_used_as_machine_authority": False,
        "implementation_authorship_machine_verified": False,
        "authority_issuer_identity_required": False,
    }
    expected_authority[
        "reviewed_implementation_authority_payload_sha256"
    ] = _digest(expected_authority)
    if authority != expected_authority:
        _fail("INPUT_LINEAGE_MISMATCH")
    return authority, authority_file_hash, candidate


def _parse_ns(value: object) -> int:
    if type(value) is not str:
        _fail("AUTHORIZATION_INVALID")
    match = UTC_RE.fullmatch(value)
    if match is None:
        _fail("AUTHORIZATION_INVALID")
    try:
        base = datetime.strptime(
            f"{match.group('date')}T{match.group('time')}",
            "%Y-%m-%dT%H:%M:%S",
        ).replace(tzinfo=timezone.utc)
    except ValueError:
        _fail("AUTHORIZATION_INVALID")
    fraction = match.group("fraction")
    nanos = int((fraction[1:] if fraction else "0").ljust(9, "0"))
    return calendar.timegm(base.utctimetuple()) * 1_000_000_000 + nanos


def _add_hash(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    if field in result:
        _fail("INPUT_SCHEMA_INVALID")
    result[field] = _digest(result)
    return result


def independent_verification_receipt(
    plan: Mapping[str, Any],
    rederived: Mapping[str, Any],
    *,
    authority_file_sha256: str,
    authority_payload_sha256: str,
    baseline_file_sha256: str,
    baseline_payload_sha256: str,
    extraction_terminal_file_sha256: str,
    extraction_terminal_payload_sha256: str,
    preflight_file_sha256: str,
    preflight_payload_sha256: str,
    authorization_file_sha256: str,
    authorization_payload_sha256: str,
    verdict_file_sha256: str,
    verdict_payload_sha256: str,
    claim_file_sha256: str,
    claim_payload_sha256: str,
) -> dict[str, Any]:
    schema = plan["verification_receipt"]
    payload = {
        "schema_version": schema["schema_version"],
        "gate_id": GATE_ID,
        "state": schema["state"],
        "verification_status": schema["verification_status"],
        "sequence_ordinal": 160,
        "configuration_count": 9,
        "outer_experiment_count": 768,
        "shard_count": 768,
        "index_count": 9,
        "artifact_path_surface_sha256": ARTIFACT_SURFACE_SHA256,
        "reviewed_implementation_authority_file_sha256": authority_file_sha256,
        "reviewed_implementation_authority_payload_sha256": (
            authority_payload_sha256
        ),
        "baseline_receipt_file_sha256": baseline_file_sha256,
        "baseline_receipt_payload_sha256": baseline_payload_sha256,
        "extraction_terminal_claim_file_sha256": (
            extraction_terminal_file_sha256
        ),
        "extraction_terminal_claim_payload_sha256": (
            extraction_terminal_payload_sha256
        ),
        "verifier_preflight_file_sha256": preflight_file_sha256,
        "verifier_preflight_payload_sha256": preflight_payload_sha256,
        "verifier_authorization_file_sha256": authorization_file_sha256,
        "verifier_authorization_payload_sha256": authorization_payload_sha256,
        "verifier_authorization_verdict_file_sha256": verdict_file_sha256,
        "verifier_authorization_verdict_payload_sha256": verdict_payload_sha256,
        "verifier_execution_claim_file_sha256": claim_file_sha256,
        "verifier_execution_claim_payload_sha256": claim_payload_sha256,
        "baseline_commitment_surface_sha256": rederived[
            "baseline_commitment_surface_sha256"
        ],
        "pre_complete_surface_sha256": rederived[
            "pre_complete_surface_sha256"
        ],
        "post_complete_surface_sha256": rederived[
            "post_complete_surface_sha256"
        ],
        "pre_protected_surface_sha256": rederived[
            "pre_protected_surface_sha256"
        ],
        "post_protected_surface_sha256": rederived[
            "post_protected_surface_sha256"
        ],
        "scientific_values_emitted": False,
        "stability_analysis_authorized": False,
    }
    if set(payload) | {"verification_receipt_payload_sha256"} != set(
        schema["exact_top_level_fields"]
    ):
        _fail("INTERNAL_SANITIZED_FAILURE")
    return _add_hash(payload, "verification_receipt_payload_sha256")



def _artifact_rows(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["role"]: dict(row) for row in plan["artifact_path_surface"]}


def _publish(
    plan: Mapping[str, Any], role: str, payload: Mapping[str, Any]
) -> None:
    row = _artifact_rows(plan).get(role)
    if row is None:
        _fail("OUTPUT_PUBLICATION_FAILED")
    final = Path(row["final_path"])
    pending = Path(row["pending_path"])
    raw = verifier_canonical_bytes(dict(payload)) + b"\n"
    if (
        final.parent != pending.parent
        or pending.name != final.name + ".pending-v0.8"
        or final.exists()
        or pending.exists()
        or not final.parent.is_dir()
    ):
        _fail("UNEXPECTED_ARTIFACT")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            pending,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_BINARY", 0),
            0o600,
        )
        offset = 0
        while offset < len(raw):
            count = os.write(descriptor, raw[offset:])
            if count <= 0:
                _fail("OUTPUT_PUBLICATION_FAILED")
            offset += count
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        if pending.read_bytes() != raw:
            _fail("OUTPUT_PUBLICATION_FAILED")
        if os.name != "nt":
            _fail("OUTPUT_PUBLICATION_FAILED")
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        move = kernel.MoveFileExW
        move.argtypes = [
            ctypes.wintypes.LPCWSTR,
            ctypes.wintypes.LPCWSTR,
            ctypes.wintypes.DWORD,
        ]
        move.restype = ctypes.wintypes.BOOL
        if not move(str(pending), str(final), 0x8):
            raise OSError(ctypes.get_last_error(), "MoveFileExW")
        if pending.exists() or final.read_bytes() != raw:
            _fail("OUTPUT_PUBLICATION_FAILED")
    except Exception:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        try:
            final_raw = final.read_bytes() if final.is_file() else None
            pending_raw = pending.read_bytes() if pending.is_file() else None
        except OSError:
            _fail("OUTPUT_PUBLICATION_FAILED")
        if final_raw == raw and pending_raw is None:
            return
        if final_raw is None and pending_raw is None:
            _fail("OUTPUT_PUBLICATION_FAILED")
        if role == "verifier_execution_claim":
            _fail("CONCURRENT_EXECUTION")
        if role == "verifier_terminal":
            _fail("TERMINAL_CONFLICT")
        _fail("OUTPUT_PUBLICATION_FAILED")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _process_creation(pid: int) -> str:
    if os.name != "nt":
        _fail("AUTHORIZATION_INVALID")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    open_process = kernel.OpenProcess
    open_process.argtypes = [
        ctypes.wintypes.DWORD,
        ctypes.wintypes.BOOL,
        ctypes.wintypes.DWORD,
    ]
    open_process.restype = ctypes.wintypes.HANDLE
    handle = open_process(0x1000, False, pid)
    if not handle:
        _fail("AUTHORIZATION_INVALID")
    values = [ctypes.wintypes.FILETIME() for _ in range(4)]
    try:
        if not kernel.GetProcessTimes(
            handle, *[ctypes.byref(value) for value in values]
        ):
            _fail("AUTHORIZATION_INVALID")
    finally:
        kernel.CloseHandle(handle)
    raw = (int(values[0].dwHighDateTime) << 32) | int(
        values[0].dwLowDateTime
    )
    unix = raw - 116_444_736_000_000_000
    if unix < 0:
        _fail("AUTHORIZATION_INVALID")
    seconds, ticks = divmod(unix, 10_000_000)
    base = datetime.fromtimestamp(seconds, timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )
    fraction = f"{ticks:07d}".rstrip("0")
    return f"{base}.{fraction}Z" if fraction else f"{base}Z"


def _identifier(value: object) -> str:
    if type(value) is not str or re.fullmatch(r"[A-Za-z0-9._-]+", value) is None:
        _fail("AUTHORIZATION_INVALID")
    return value


def _authorization_verifier_row(
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    rows = [
        row
        for row in candidate["implementation_files"]
        if row["relative_path"]
        == "tools/verify_gate12c2_original_baseline_authorization.py"
    ]
    if len(rows) != 1:
        _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    return dict(rows[0])


def _validate_control_preflight(
    plan: Mapping[str, Any],
    preflight: Mapping[str, Any],
    *,
    scope: str,
    authority: Mapping[str, Any],
    authority_file_hash: str,
    now_ns: int,
    linked_receipts: Mapping[str, str] | None = None,
) -> tuple[int, int]:
    schema = plan["control_receipt_schemas"][f"{scope}_preflight"]
    expected = {
        "schema_version": schema["schema_version"],
        "gate_id": GATE_ID,
        "authorization_scope": scope,
        "state": schema["state"],
        "sequence_ordinal": schema["sequence_ordinal"],
        "contract_file_sha256": plan["contract_file_sha256"],
        "plan_file_sha256": PLAN_FILE_SHA256,
        "plan_payload_sha256": PLAN_PAYLOAD_SHA256,
        "reviewed_implementation_authority_file_sha256": authority_file_hash,
        "reviewed_implementation_authority_payload_sha256": authority[
            "reviewed_implementation_authority_payload_sha256"
        ],
        "artifact_path_surface_sha256": ARTIFACT_SURFACE_SHA256,
        "artifact_lifecycle_phase": schema[
            "required_artifact_lifecycle_phase"
        ],
        "input_lineage_status": "verified",
        "protected_root_status": "canonical_path_bound_no_payload_read",
        "output_surface_status": "fresh_exact",
        "closed_boundaries_status": "closed",
    }
    if linked_receipts is not None:
        expected.update(linked_receipts)
    if any(preflight.get(key) != value for key, value in expected.items()):
        _fail("AUTHORIZATION_INVALID")
    _identifier(preflight.get("preflight_id"))
    issued = _parse_ns(preflight.get("issued_at_utc"))
    expires = _parse_ns(preflight.get("expires_at_utc"))
    current = _integer(now_ns)
    if (
        expires <= issued
        or expires - issued > schema["maximum_age_seconds"] * 1_000_000_000
        or not issued <= current < expires
    ):
        _fail("AUTHORIZATION_INVALID")
    return issued, expires


def _validate_control_authorization(
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    preflight: Mapping[str, Any],
    *,
    scope: str,
    preflight_file_hash: str,
    now_ns: int,
) -> tuple[int, int]:
    schema = plan["control_receipt_schemas"][f"{scope}_authorization"]
    expected = {
        "schema_version": schema["schema_version"],
        "gate_id": GATE_ID,
        "authorization_scope": scope,
        "state": schema["state"],
        "sequence_ordinal": schema["sequence_ordinal"],
        "single_use": True,
        "plan_payload_sha256": PLAN_PAYLOAD_SHA256,
        "reviewed_implementation_authority_file_sha256": preflight[
            "reviewed_implementation_authority_file_sha256"
        ],
        "reviewed_implementation_authority_payload_sha256": preflight[
            "reviewed_implementation_authority_payload_sha256"
        ],
        "preflight_file_sha256": preflight_file_hash,
        "preflight_payload_sha256": preflight["preflight_payload_sha256"],
        "protected_root_path": str(PROTECTED_ROOT),
        "artifact_path_surface_sha256": ARTIFACT_SURFACE_SHA256,
        "artifact_lifecycle_phase": schema[
            "required_artifact_lifecycle_phase"
        ],
    }
    if scope == "verifier":
        for field in (
            "baseline_receipt_file_sha256",
            "baseline_receipt_payload_sha256",
            "extraction_terminal_claim_file_sha256",
            "extraction_terminal_claim_payload_sha256",
        ):
            expected[field] = preflight[field]
    if any(authorization.get(key) != value for key, value in expected.items()):
        _fail("AUTHORIZATION_INVALID")
    _identifier(authorization.get("authorization_id"))
    pre_issued = _parse_ns(preflight.get("issued_at_utc"))
    pre_expires = _parse_ns(preflight.get("expires_at_utc"))
    issued = _parse_ns(authorization.get("issued_at_utc"))
    expires = _parse_ns(authorization.get("expires_at_utc"))
    current = _integer(now_ns)
    if (
        issued < pre_issued
        or expires <= issued
        or expires > pre_expires
        or expires - issued > schema["maximum_age_seconds"] * 1_000_000_000
        or not issued <= current < expires
    ):
        _fail("AUTHORIZATION_INVALID")
    return issued, expires


def _validate_control_verdict(
    plan: Mapping[str, Any],
    verdict: Mapping[str, Any],
    authorization: Mapping[str, Any],
    preflight: Mapping[str, Any],
    *,
    scope: str,
    preflight_file_hash: str,
    authorization_file_hash: str,
    verifier_row: Mapping[str, Any],
    now_ns: int,
) -> int:
    schema = plan["control_receipt_schemas"][f"{scope}_authorization_verdict"]
    issued = _parse_ns(authorization.get("issued_at_utc"))
    expires = _parse_ns(authorization.get("expires_at_utc"))
    verified = _parse_ns(verdict.get("verified_at_utc"))
    current = _integer(now_ns)
    expected = {
        "schema_version": schema["schema_version"],
        "gate_id": GATE_ID,
        "authorization_scope": scope,
        "state": schema["pass_state"],
        "sequence_ordinal": schema["sequence_ordinal"],
        "outcome_kind": "pass",
        "reason_code": None,
        "plan_payload_sha256": PLAN_PAYLOAD_SHA256,
        "reviewed_implementation_authority_file_sha256": authorization[
            "reviewed_implementation_authority_file_sha256"
        ],
        "reviewed_implementation_authority_payload_sha256": authorization[
            "reviewed_implementation_authority_payload_sha256"
        ],
        "preflight_file_sha256": preflight_file_hash,
        "preflight_payload_sha256": preflight["preflight_payload_sha256"],
        "authorization_file_sha256": authorization_file_hash,
        "authorization_payload_sha256": authorization[
            "authorization_payload_sha256"
        ],
        "artifact_path_surface_sha256": ARTIFACT_SURFACE_SHA256,
        "artifact_lifecycle_phase": schema[
            "required_artifact_lifecycle_phase"
        ],
        "authorization_expires_at_utc": authorization["expires_at_utc"],
        "remaining_freshness_nanoseconds": expires - verified,
        "protected_root_read": False,
        "authorization_verifier_relative_path": verifier_row["relative_path"],
        "authorization_verifier_file_sha256": verifier_row["file_sha256"],
        "authorization_verifier_git_blob_oid": verifier_row["git_blob_oid"],
    }
    if scope == "verifier":
        for field in (
            "baseline_receipt_file_sha256",
            "baseline_receipt_payload_sha256",
            "extraction_terminal_claim_file_sha256",
            "extraction_terminal_claim_payload_sha256",
        ):
            expected[field] = authorization[field]
    if any(verdict.get(key) != value for key, value in expected.items()):
        _fail("AUTHORIZATION_INVALID")
    _identifier(verdict.get("verification_id"))
    if not issued <= verified <= current < expires:
        _fail("AUTHORIZATION_INVALID")
    return verified


def _validate_control_claim(
    plan: Mapping[str, Any],
    claim: Mapping[str, Any],
    authorization: Mapping[str, Any],
    preflight: Mapping[str, Any],
    verdict: Mapping[str, Any],
    *,
    scope: str,
    preflight_file_hash: str,
    authorization_file_hash: str,
    verdict_file_hash: str,
) -> int:
    schema = plan["control_receipt_schemas"][f"{scope}_execution_claim"]
    issued = _parse_ns(authorization.get("issued_at_utc"))
    expires = _parse_ns(authorization.get("expires_at_utc"))
    verified = _parse_ns(verdict.get("verified_at_utc"))
    claimed = _parse_ns(claim.get("claimed_at_utc"))
    creation = _parse_ns(claim.get("owner_process_creation_time_utc"))
    expected = {
        "schema_version": schema["schema_version"],
        "gate_id": GATE_ID,
        "authorization_scope": scope,
        "state": schema["state"],
        "sequence_ordinal": schema["sequence_ordinal"],
        "plan_payload_sha256": PLAN_PAYLOAD_SHA256,
        "reviewed_implementation_authority_file_sha256": authorization[
            "reviewed_implementation_authority_file_sha256"
        ],
        "reviewed_implementation_authority_payload_sha256": authorization[
            "reviewed_implementation_authority_payload_sha256"
        ],
        "preflight_file_sha256": preflight_file_hash,
        "preflight_payload_sha256": preflight["preflight_payload_sha256"],
        "authorization_file_sha256": authorization_file_hash,
        "authorization_payload_sha256": authorization[
            "authorization_payload_sha256"
        ],
        "authorization_verdict_file_sha256": verdict_file_hash,
        "authorization_verdict_payload_sha256": verdict[
            "authorization_verdict_payload_sha256"
        ],
        "artifact_path_surface_sha256": ARTIFACT_SURFACE_SHA256,
        "artifact_lifecycle_phase": schema[
            "required_artifact_lifecycle_phase"
        ],
        "protected_input_read": False,
    }
    if scope == "verifier":
        for field in (
            "baseline_receipt_file_sha256",
            "baseline_receipt_payload_sha256",
            "extraction_terminal_claim_file_sha256",
            "extraction_terminal_claim_payload_sha256",
        ):
            expected[field] = authorization[field]
    if any(claim.get(key) != value for key, value in expected.items()):
        _fail("AUTHORIZATION_INVALID")
    _identifier(claim.get("execution_claim_id"))
    _identifier(claim.get("launch_id"))
    hostname = claim.get("owner_hostname")
    pid = claim.get("owner_pid")
    if (
        type(hostname) is not str
        or not hostname
        or not hostname.isascii()
        or type(pid) is not int
        or pid < 1
        or pid > (1 << 32) - 1
        or creation > claimed
        or not issued <= verified <= claimed < expires
    ):
        _fail("AUTHORIZATION_INVALID")
    return claimed



def _load_verifier_controls(
    plan: Mapping[str, Any],
    authority: Mapping[str, Any],
    authority_file_hash: str,
    candidate: Mapping[str, Any],
    *,
    now_ns: int,
    execution_claim_published: bool = False,
) -> dict[str, Any]:
    rows = _artifact_rows(plan)
    now_ns = _integer(now_ns)
    verifier_row = _authorization_verifier_row(candidate)
    extraction_preflight_schema = plan["control_receipt_schemas"][
        "extraction_preflight"
    ]
    extraction_preflight, extraction_preflight_file_hash = _receipt(
        Path(rows["extraction_preflight"]["final_path"]),
        extraction_preflight_schema["exact_top_level_fields"],
        "preflight_payload_sha256",
    )
    extraction_preflight_issued = _parse_ns(
        extraction_preflight.get("issued_at_utc")
    )
    _validate_control_preflight(
        plan,
        extraction_preflight,
        scope="extraction",
        authority=authority,
        authority_file_hash=authority_file_hash,
        now_ns=extraction_preflight_issued,
    )
    extraction_authorization_schema = plan["control_receipt_schemas"][
        "extraction_authorization"
    ]
    extraction_authorization, extraction_authorization_file_hash = _receipt(
        Path(rows["extraction_authorization"]["final_path"]),
        extraction_authorization_schema["exact_top_level_fields"],
        "authorization_payload_sha256",
    )
    extraction_authorization_issued = _parse_ns(
        extraction_authorization.get("issued_at_utc")
    )
    _validate_control_authorization(
        plan,
        extraction_authorization,
        extraction_preflight,
        scope="extraction",
        preflight_file_hash=extraction_preflight_file_hash,
        now_ns=extraction_authorization_issued,
    )
    extraction_verdict_schema = plan["control_receipt_schemas"][
        "extraction_authorization_verdict"
    ]
    extraction_verdict, extraction_verdict_file_hash = _receipt(
        Path(rows["extraction_authorization_verdict"]["final_path"]),
        extraction_verdict_schema["exact_top_level_fields"],
        "authorization_verdict_payload_sha256",
    )
    extraction_verified = _parse_ns(extraction_verdict.get("verified_at_utc"))
    _validate_control_verdict(
        plan,
        extraction_verdict,
        extraction_authorization,
        extraction_preflight,
        scope="extraction",
        preflight_file_hash=extraction_preflight_file_hash,
        authorization_file_hash=extraction_authorization_file_hash,
        verifier_row=verifier_row,
        now_ns=extraction_verified,
    )
    extraction_claim_schema = plan["control_receipt_schemas"][
        "extraction_execution_claim"
    ]
    extraction_claim, extraction_claim_file_hash = _receipt(
        Path(rows["extraction_execution_claim"]["final_path"]),
        extraction_claim_schema["exact_top_level_fields"],
        "execution_claim_payload_sha256",
    )
    _validate_control_claim(
        plan,
        extraction_claim,
        extraction_authorization,
        extraction_preflight,
        extraction_verdict,
        scope="extraction",
        preflight_file_hash=extraction_preflight_file_hash,
        authorization_file_hash=extraction_authorization_file_hash,
        verdict_file_hash=extraction_verdict_file_hash,
    )
    success_schema = plan["success_receipt"]
    baseline, baseline_file_hash = _receipt(
        Path(rows["extraction_success"]["final_path"]),
        success_schema["exact_top_level_fields"],
        "baseline_receipt_payload_sha256",
    )
    baseline_expected = {
        "schema_version": success_schema["schema_version"],
        "gate_id": GATE_ID,
        "state": success_schema["state"],
        "verification_status": success_schema["verification_status"],
        "sequence_ordinal": success_schema["sequence_ordinal"],
        "original_resource_gate_status": "indeterminate_permanent",
        "replacement_resource_qualification": "not_performed",
        "scientific_values_emitted": False,
        "stability_analysis_authorized": False,
        "locked_execution_authorized": False,
        "real_held_out_execution_authorized": False,
        "N2_open": False,
        "N3_open": False,
        "configuration_count": 9,
        "outer_experiment_count": 768,
        "shard_count": 768,
        "index_count": 9,
        "partial_or_temp_count": 0,
        "unexpected_file_count": 0,
        "file_reparse_point_count": 0,
        "directory_reparse_point_count": 0,
        "original_plan_payload_sha256": plan["original_input_lineage"][
            "original_plan_payload_sha256"
        ],
        "incident_manifest_payload_sha256": plan["original_input_lineage"][
            "incident_manifest_payload_sha256"
        ],
        "payload_seal_sha256": plan["original_input_lineage"][
            "payload_seal_payload_sha256"
        ],
        "formal_payload_closeout_payload_sha256": plan[
            "original_input_lineage"
        ]["formal_payload_closeout_payload_sha256"],
        "resource_v0_7_plan_payload_sha256": next(
            row["payload_sha256"]
            for row in plan["upstream_authority"]["artifact_rows"]
            if row["role"] == "replacement_resource_v0_7_plan"
        ),
        "resource_implementation_review_payload_sha256": next(
            row["payload_sha256"]
            for row in plan["upstream_authority"]["artifact_rows"]
            if row["role"] == "resource_implementation_pass"
        ),
        "artifact_path_surface_sha256": ARTIFACT_SURFACE_SHA256,
        "reviewed_implementation_authority_file_sha256": authority_file_hash,
        "reviewed_implementation_authority_payload_sha256": authority[
            "reviewed_implementation_authority_payload_sha256"
        ],
        "extraction_preflight_file_sha256": extraction_preflight_file_hash,
        "extraction_preflight_payload_sha256": extraction_preflight[
            "preflight_payload_sha256"
        ],
        "extraction_authorization_file_sha256": (
            extraction_authorization_file_hash
        ),
        "extraction_authorization_payload_sha256": extraction_authorization[
            "authorization_payload_sha256"
        ],
        "extraction_authorization_verdict_file_sha256": (
            extraction_verdict_file_hash
        ),
        "extraction_authorization_verdict_payload_sha256": extraction_verdict[
            "authorization_verdict_payload_sha256"
        ],
        "extraction_execution_claim_file_sha256": extraction_claim_file_hash,
        "extraction_execution_claim_payload_sha256": extraction_claim[
            "execution_claim_payload_sha256"
        ],
        "pre_complete_surface_sha256": COMPLETE_SURFACE_SHA256,
        "post_complete_surface_sha256": COMPLETE_SURFACE_SHA256,
        "pre_protected_surface_sha256": PROTECTED_SURFACE_SHA256,
        "post_protected_surface_sha256": PROTECTED_SURFACE_SHA256,
        "configuration_surface_sha256": CONFIGURATION_SURFACE_SHA256,
    }
    if any(baseline.get(key) != value for key, value in baseline_expected.items()):
        _fail("INPUT_LINEAGE_MISMATCH")
    commitments = baseline.get("configuration_commitments")
    commitment_fields = {
        "configuration_id",
        "outer_experiment_count",
        "outer_id_surface_sha256",
        "result_commitment_surface_sha256",
        "scientific_projection_sha256",
        "semantic_index_commitment_v0_1_sha256",
    }
    if (
        type(commitments) is not list
        or len(commitments) != 9
        or commitments
        != sorted(commitments, key=lambda row: row.get("configuration_id", ""))
    ):
        _fail("INPUT_LINEAGE_MISMATCH")
    frozen_by_id = {
        row["configuration_id"]: row for row in plan["configuration_surface"]
    }
    seen_commitments: set[str] = set()
    for commitment in commitments:
        if type(commitment) is not dict or set(commitment) != commitment_fields:
            _fail("INPUT_LINEAGE_MISMATCH")
        identifier = commitment.get("configuration_id")
        if identifier in seen_commitments or identifier not in frozen_by_id:
            _fail("INPUT_LINEAGE_MISMATCH")
        seen_commitments.add(identifier)
        if commitment.get("outer_experiment_count") != frozen_by_id[identifier][
            "outer_experiment_count"
        ]:
            _fail("INPUT_LINEAGE_MISMATCH")
        for field in commitment_fields - {
            "configuration_id",
            "outer_experiment_count",
        }:
            value = commitment.get(field)
            if type(value) is not str or SHA_RE.fullmatch(value) is None:
                _fail("INPUT_LINEAGE_MISMATCH")
    if (
        seen_commitments != set(frozen_by_id)
        or _digest(commitments)
        != baseline.get("baseline_commitment_surface_sha256")
    ):
        _fail("INPUT_LINEAGE_MISMATCH")
    extraction_terminal_schema = plan["control_receipt_schemas"][
        "extraction_terminal"
    ]
    extraction_terminal, extraction_terminal_file_hash = _receipt(
        Path(rows["extraction_terminal"]["final_path"]),
        extraction_terminal_schema["exact_top_level_fields"],
        "terminal_claim_payload_sha256",
    )
    terminal_claimed = _parse_ns(extraction_terminal.get("claimed_at_utc"))
    extraction_claimed = _parse_ns(extraction_claim.get("claimed_at_utc"))
    terminal_expected = {
        "schema_version": extraction_terminal_schema["schema_version"],
        "gate_id": GATE_ID,
        "authorization_scope": "extraction",
        "execution_claim_id": extraction_claim["execution_claim_id"],
        "state": "EXTRACTION_SUCCESS_TERMINAL_CLAIM_PUBLISHED",
        "sequence_ordinal": extraction_terminal_schema["sequence_ordinal"],
        "outcome_kind": "success",
        "reviewed_implementation_authority_file_sha256": authority_file_hash,
        "reviewed_implementation_authority_payload_sha256": authority[
            "reviewed_implementation_authority_payload_sha256"
        ],
        "preflight_file_sha256": extraction_preflight_file_hash,
        "preflight_payload_sha256": extraction_preflight[
            "preflight_payload_sha256"
        ],
        "authorization_file_sha256": extraction_authorization_file_hash,
        "authorization_payload_sha256": extraction_authorization[
            "authorization_payload_sha256"
        ],
        "authorization_verdict_file_sha256": extraction_verdict_file_hash,
        "authorization_verdict_payload_sha256": extraction_verdict[
            "authorization_verdict_payload_sha256"
        ],
        "execution_claim_file_sha256": extraction_claim_file_hash,
        "execution_claim_payload_sha256": extraction_claim[
            "execution_claim_payload_sha256"
        ],
        "leaf_schema_version": baseline["schema_version"],
        "leaf_payload_sha256": baseline["baseline_receipt_payload_sha256"],
        "leaf_exact_payload": baseline,
    }
    if (
        terminal_claimed < extraction_claimed
        or any(
            extraction_terminal.get(key) != value
            for key, value in terminal_expected.items()
        )
    ):
        _fail("INPUT_LINEAGE_MISMATCH")
    preflight_schema = plan["control_receipt_schemas"]["verifier_preflight"]
    preflight, preflight_file_hash = _receipt(
        Path(rows["verifier_preflight"]["final_path"]),
        preflight_schema["exact_top_level_fields"],
        "preflight_payload_sha256",
    )
    fixed_preflight = {
        "schema_version": preflight_schema["schema_version"],
        "state": preflight_schema["state"],
        "authorization_scope": "verifier",
        "reviewed_implementation_authority_file_sha256": authority_file_hash,
        "reviewed_implementation_authority_payload_sha256": authority[
            "reviewed_implementation_authority_payload_sha256"
        ],
        "baseline_receipt_file_sha256": baseline_file_hash,
        "baseline_receipt_payload_sha256": baseline[
            "baseline_receipt_payload_sha256"
        ],
        "extraction_terminal_claim_file_sha256": (
            extraction_terminal_file_hash
        ),
        "extraction_terminal_claim_payload_sha256": extraction_terminal[
            "terminal_claim_payload_sha256"
        ],
        "artifact_path_surface_sha256": ARTIFACT_SURFACE_SHA256,
        "input_lineage_status": "verified",
        "protected_root_status": "canonical_path_bound_no_payload_read",
        "output_surface_status": "fresh_exact",
        "closed_boundaries_status": "closed",
    }
    if any(
        preflight.get(key) != value
        for key, value in fixed_preflight.items()
    ):
        _fail("AUTHORIZATION_INVALID")
    pre_issued = _parse_ns(preflight["issued_at_utc"])
    pre_expires = _parse_ns(preflight["expires_at_utc"])
    if (
        not pre_issued <= now_ns < pre_expires
        or pre_expires - pre_issued > 1_800_000_000_000
    ):
        _fail("AUTHORIZATION_INVALID")
    _validate_control_preflight(
        plan,
        preflight,
        scope="verifier",
        authority=authority,
        authority_file_hash=authority_file_hash,
        now_ns=now_ns,
        linked_receipts={
            "baseline_receipt_file_sha256": baseline_file_hash,
            "baseline_receipt_payload_sha256": baseline[
                "baseline_receipt_payload_sha256"
            ],
            "extraction_terminal_claim_file_sha256": (
                extraction_terminal_file_hash
            ),
            "extraction_terminal_claim_payload_sha256": extraction_terminal[
                "terminal_claim_payload_sha256"
            ],
        },
    )
    authorization_schema = plan["control_receipt_schemas"][
        "verifier_authorization"
    ]
    authorization, authorization_file_hash = _receipt(
        Path(rows["verifier_authorization"]["final_path"]),
        authorization_schema["exact_top_level_fields"],
        "authorization_payload_sha256",
    )
    fixed_authorization = {
        "schema_version": authorization_schema["schema_version"],
        "state": authorization_schema["state"],
        "authorization_scope": "verifier",
        "single_use": True,
        "reviewed_implementation_authority_file_sha256": authority_file_hash,
        "reviewed_implementation_authority_payload_sha256": authority[
            "reviewed_implementation_authority_payload_sha256"
        ],
        "preflight_file_sha256": preflight_file_hash,
        "preflight_payload_sha256": preflight[
            "preflight_payload_sha256"
        ],
        "baseline_receipt_file_sha256": baseline_file_hash,
        "baseline_receipt_payload_sha256": baseline[
            "baseline_receipt_payload_sha256"
        ],
        "extraction_terminal_claim_file_sha256": (
            extraction_terminal_file_hash
        ),
        "extraction_terminal_claim_payload_sha256": extraction_terminal[
            "terminal_claim_payload_sha256"
        ],
        "artifact_path_surface_sha256": ARTIFACT_SURFACE_SHA256,
        "protected_root_path": str(PROTECTED_ROOT),
    }
    if any(
        authorization.get(key) != value
        for key, value in fixed_authorization.items()
    ):
        _fail("AUTHORIZATION_INVALID")
    auth_issued = _parse_ns(authorization["issued_at_utc"])
    auth_expires = _parse_ns(authorization["expires_at_utc"])
    if (
        not auth_issued <= now_ns < auth_expires <= pre_expires
        or auth_expires - auth_issued > 1_800_000_000_000
    ):
        _fail("AUTHORIZATION_INVALID")
    _validate_control_authorization(
        plan,
        authorization,
        preflight,
        scope="verifier",
        preflight_file_hash=preflight_file_hash,
        now_ns=now_ns,
    )
    verdict_schema = plan["control_receipt_schemas"][
        "verifier_authorization_verdict"
    ]
    verdict, verdict_file_hash = _receipt(
        Path(rows["verifier_authorization_verdict"]["final_path"]),
        verdict_schema["exact_top_level_fields"],
        "authorization_verdict_payload_sha256",
    )
    verified_ns = _parse_ns(verdict["verified_at_utc"])
    fixed_verdict = {
        "schema_version": verdict_schema["schema_version"],
        "state": verdict_schema["pass_state"],
        "authorization_scope": "verifier",
        "outcome_kind": "pass",
        "reason_code": None,
        "preflight_file_sha256": preflight_file_hash,
        "preflight_payload_sha256": preflight[
            "preflight_payload_sha256"
        ],
        "authorization_file_sha256": authorization_file_hash,
        "authorization_payload_sha256": authorization[
            "authorization_payload_sha256"
        ],
        "remaining_freshness_nanoseconds": auth_expires - verified_ns,
        "protected_root_read": False,
        "baseline_receipt_file_sha256": baseline_file_hash,
        "baseline_receipt_payload_sha256": baseline[
            "baseline_receipt_payload_sha256"
        ],
        "extraction_terminal_claim_file_sha256": (
            extraction_terminal_file_hash
        ),
        "extraction_terminal_claim_payload_sha256": extraction_terminal[
            "terminal_claim_payload_sha256"
        ],
    }
    if any(verdict.get(key) != value for key, value in fixed_verdict.items()):
        _fail("AUTHORIZATION_INVALID")
    _validate_control_verdict(
        plan,
        verdict,
        authorization,
        preflight,
        scope="verifier",
        preflight_file_hash=preflight_file_hash,
        authorization_file_hash=authorization_file_hash,
        verifier_row=verifier_row,
        now_ns=now_ns,
    )
    absent_roles = [
        "verifier_terminal",
        "verifier_success",
        "verifier_failure",
    ]
    if not execution_claim_published:
        absent_roles.append("verifier_execution_claim")
    for role in absent_roles:
        row = rows[role]
        if Path(row["final_path"]).exists() or Path(row["pending_path"]).exists():
            _fail("UNEXPECTED_ARTIFACT")
    return {
        "baseline": baseline,
        "baseline_file_hash": baseline_file_hash,
        "extraction_terminal": extraction_terminal,
        "extraction_terminal_file_hash": extraction_terminal_file_hash,
        "preflight": preflight,
        "preflight_file_hash": preflight_file_hash,
        "authorization": authorization,
        "authorization_file_hash": authorization_file_hash,
        "verdict": verdict,
        "verdict_file_hash": verdict_file_hash,
    }


def _verifier_claim(
    plan: Mapping[str, Any],
    controls: Mapping[str, Any],
    *,
    execution_claim_id: str,
    launch_id: str,
    claimed_at_utc: str,
    owner_pid: int,
    owner_creation: str,
    now_ns: int,
) -> dict[str, Any]:
    if (
        re.fullmatch(r"[A-Za-z0-9._-]+", execution_claim_id) is None
        or re.fullmatch(r"[A-Za-z0-9._-]+", launch_id) is None
    ):
        _fail("AUTHORIZATION_INVALID")
    schema = plan["control_receipt_schemas"]["verifier_execution_claim"]
    authorization = controls["authorization"]
    claimed_ns = _parse_ns(claimed_at_utc)
    creation_ns = _parse_ns(owner_creation)
    verified_ns = _parse_ns(controls["verdict"]["verified_at_utc"])
    expires_ns = _parse_ns(authorization["expires_at_utc"])
    current_ns = _integer(now_ns)
    if (
        type(owner_pid) is not int
        or owner_pid < 1
        or owner_pid > (1 << 32) - 1
        or creation_ns > claimed_ns
        or not verified_ns <= claimed_ns <= current_ns < expires_ns
    ):
        _fail("AUTHORIZATION_INVALID")
    payload = {
        "schema_version": schema["schema_version"],
        "gate_id": GATE_ID,
        "authorization_scope": "verifier",
        "execution_claim_id": execution_claim_id,
        "launch_id": launch_id,
        "state": schema["state"],
        "sequence_ordinal": schema["sequence_ordinal"],
        "claimed_at_utc": claimed_at_utc,
        "owner_hostname": socket.gethostname(),
        "owner_pid": owner_pid,
        "owner_process_creation_time_utc": owner_creation,
        "plan_payload_sha256": PLAN_PAYLOAD_SHA256,
        "reviewed_implementation_authority_file_sha256": authorization[
            "reviewed_implementation_authority_file_sha256"
        ],
        "reviewed_implementation_authority_payload_sha256": authorization[
            "reviewed_implementation_authority_payload_sha256"
        ],
        "preflight_file_sha256": controls["preflight_file_hash"],
        "preflight_payload_sha256": controls["preflight"][
            "preflight_payload_sha256"
        ],
        "authorization_file_sha256": controls["authorization_file_hash"],
        "authorization_payload_sha256": authorization[
            "authorization_payload_sha256"
        ],
        "authorization_verdict_file_sha256": controls[
            "verdict_file_hash"
        ],
        "authorization_verdict_payload_sha256": controls["verdict"][
            "authorization_verdict_payload_sha256"
        ],
        "artifact_path_surface_sha256": ARTIFACT_SURFACE_SHA256,
        "artifact_lifecycle_phase": schema[
            "required_artifact_lifecycle_phase"
        ],
        "protected_input_read": False,
        "baseline_receipt_file_sha256": controls["baseline_file_hash"],
        "baseline_receipt_payload_sha256": controls["baseline"][
            "baseline_receipt_payload_sha256"
        ],
        "extraction_terminal_claim_file_sha256": controls[
            "extraction_terminal_file_hash"
        ],
        "extraction_terminal_claim_payload_sha256": controls[
            "extraction_terminal"
        ]["terminal_claim_payload_sha256"],
    }
    if set(payload) | {"execution_claim_payload_sha256"} != set(
        schema["exact_top_level_fields"]
    ):
        _fail("INTERNAL_SANITIZED_FAILURE")
    return _add_hash(payload, "execution_claim_payload_sha256")



def _verifier_terminal(
    plan: Mapping[str, Any],
    controls: Mapping[str, Any],
    claim: Mapping[str, Any],
    *,
    claim_file_hash: str,
    leaf: Mapping[str, Any],
    outcome: str,
    claimed_at_utc: str,
) -> dict[str, Any]:
    schema = plan["control_receipt_schemas"]["verifier_terminal"]
    claim_schema = plan["control_receipt_schemas"]["verifier_execution_claim"]
    if outcome not in schema["outcome_kind_allowlist"]:
        _fail("TERMINAL_CONFLICT")
    if type(claim) is not dict or set(claim) != set(
        claim_schema["exact_top_level_fields"]
    ):
        _fail("TERMINAL_OUTCOME_RECONSTRUCTION_FAILED")
    _self_hash(claim, "execution_claim_payload_sha256")
    if (
        claim.get("authorization_scope") != "verifier"
        or claim.get("state") != claim_schema["state"]
        or claim.get("sequence_ordinal") != claim_schema["sequence_ordinal"]
    ):
        _fail("TERMINAL_OUTCOME_RECONSTRUCTION_FAILED")
    leaf_schema = (
        plan["verification_receipt"]
        if outcome == "success"
        else plan["verifier_failure_receipt"]
    )
    leaf_field = (
        "verification_receipt_payload_sha256"
        if outcome == "success"
        else "failure_receipt_payload_sha256"
    )
    if (
        type(leaf) is not dict
        or set(leaf) != set(leaf_schema["exact_top_level_fields"])
        or leaf.get("schema_version") != leaf_schema["schema_version"]
    ):
        _fail("TERMINAL_OUTCOME_RECONSTRUCTION_FAILED")
    leaf_hash = _self_hash(leaf, leaf_field)
    terminal_ns = _parse_ns(claimed_at_utc)
    claim_ns = _parse_ns(claim.get("claimed_at_utc"))
    if (
        terminal_ns < claim_ns
        or type(claim_file_hash) is not str
        or SHA_RE.fullmatch(claim_file_hash) is None
    ):
        _fail("TERMINAL_OUTCOME_RECONSTRUCTION_FAILED")
    payload = {
        "schema_version": schema["schema_version"],
        "gate_id": GATE_ID,
        "authorization_scope": "verifier",
        "execution_claim_id": claim["execution_claim_id"],
        "state": (
            "VERIFIER_SUCCESS_TERMINAL_CLAIM_PUBLISHED"
            if outcome == "success"
            else "VERIFIER_FAILURE_TERMINAL_CLAIM_PUBLISHED"
        ),
        "sequence_ordinal": schema["sequence_ordinal"],
        "claimed_at_utc": claimed_at_utc,
        "outcome_kind": outcome,
        "reviewed_implementation_authority_file_sha256": claim[
            "reviewed_implementation_authority_file_sha256"
        ],
        "reviewed_implementation_authority_payload_sha256": claim[
            "reviewed_implementation_authority_payload_sha256"
        ],
        "preflight_file_sha256": controls["preflight_file_hash"],
        "preflight_payload_sha256": claim["preflight_payload_sha256"],
        "authorization_file_sha256": controls["authorization_file_hash"],
        "authorization_payload_sha256": claim[
            "authorization_payload_sha256"
        ],
        "authorization_verdict_file_sha256": controls[
            "verdict_file_hash"
        ],
        "authorization_verdict_payload_sha256": claim[
            "authorization_verdict_payload_sha256"
        ],
        "execution_claim_file_sha256": claim_file_hash,
        "execution_claim_payload_sha256": claim[
            "execution_claim_payload_sha256"
        ],
        "leaf_schema_version": leaf["schema_version"],
        "leaf_payload_sha256": leaf_hash,
        "leaf_exact_payload": dict(leaf),
        "baseline_receipt_file_sha256": controls["baseline_file_hash"],
        "baseline_receipt_payload_sha256": controls["baseline"][
            "baseline_receipt_payload_sha256"
        ],
        "extraction_terminal_claim_file_sha256": controls[
            "extraction_terminal_file_hash"
        ],
        "extraction_terminal_claim_payload_sha256": controls[
            "extraction_terminal"
        ]["terminal_claim_payload_sha256"],
    }
    if set(payload) | {"terminal_claim_payload_sha256"} != set(
        schema["exact_top_level_fields"]
    ):
        _fail("INTERNAL_SANITIZED_FAILURE")
    return _add_hash(payload, "terminal_claim_payload_sha256")


def _verifier_failure(
    plan: Mapping[str, Any],
    controls: Mapping[str, Any],
    claim: Mapping[str, Any],
    *,
    claim_file_hash: str,
    code: str,
    occurred_at_utc: str,
    progress: Mapping[str, Any],
) -> dict[str, Any]:
    occurred_ns = _parse_ns(occurred_at_utc)
    claim_ns = _parse_ns(claim.get("claimed_at_utc"))
    if occurred_ns < claim_ns:
        _fail("INTERNAL_SANITIZED_FAILURE")
    source = str(progress.get("source_state", ""))
    phase = str(progress.get("failure_phase", ""))
    selected_code = code if code in FAILURE_CODES else "INTERNAL_SANITIZED_FAILURE"

    def select_row(candidate: str) -> dict[str, Any]:
        rows = [
            row
            for row in plan["failure_matrix"]
            if row["scope"] == "verifier"
            and row["source_state"] == source
            and row["failure_phase"] == phase
            and row["failure_code"] == candidate
        ]
        if len(rows) != 1:
            _fail("INTERNAL_SANITIZED_FAILURE")
        return dict(rows[0])

    try:
        row = select_row(selected_code)
    except IndependentVerificationError:
        selected_code = "INTERNAL_SANITIZED_FAILURE"
        row = select_row(selected_code)
    if row["failure_receipt_allowed"] is not True:
        _fail("INTERNAL_SANITIZED_FAILURE")
    profile_name = row["availability_profile"]
    profile = plan["failure_evidence_availability_profiles"]["verifier"][
        profile_name
    ]
    if profile.get("inherit_exact_terminal_claim_evidence") is True:
        _fail("INTERNAL_SANITIZED_FAILURE")
    evidence_value = progress.get("evidence")
    evidence = dict(evidence_value) if type(evidence_value) is dict else {}
    digest_fields = {
        "pre_complete_surface": "pre_complete_surface_sha256",
        "pre_protected_surface": "pre_protected_surface_sha256",
        "post_complete_surface": "post_complete_surface_sha256",
        "post_protected_surface": "post_protected_surface_sha256",
        "recomputed_baseline_commitment_surface": (
            "recomputed_baseline_commitment_surface_sha256"
        ),
    }
    availability: dict[str, str | None] = {}
    for availability_key, receipt_field in digest_fields.items():
        if availability_key not in profile:
            continue
        value = evidence.get(receipt_field)
        if profile[availability_key] is True:
            if type(value) is not str or SHA_RE.fullmatch(value) is None:
                _fail("INTERNAL_SANITIZED_FAILURE")
            availability[receipt_field] = value
        else:
            if value is not None:
                _fail("INTERNAL_SANITIZED_FAILURE")
            availability[receipt_field] = None
    counts = {}
    for field, maximum in (
        ("configuration_count_reached", 9),
        ("outer_experiment_count_reached", 768),
        ("shard_count_reached", 768),
        ("index_count_reached", 9),
    ):
        counts[field] = _integer(progress.get(field, 0), maximum)
    schema = plan["verifier_failure_receipt"]
    payload = {
        "schema_version": schema["schema_version"],
        "gate_id": GATE_ID,
        "authorization_scope": "verifier",
        "execution_claim_id": claim["execution_claim_id"],
        "state": "VERIFIER_FAILURE_RECEIPT_PUBLISHED",
        "failure_code": selected_code,
        "failure_phase": phase,
        "source_state": source,
        "sequence_ordinal": 160,
        "occurred_at_utc": occurred_at_utc,
        **counts,
        "baseline_receipt_file_sha256": controls["baseline_file_hash"],
        "baseline_receipt_payload_sha256": controls["baseline"][
            "baseline_receipt_payload_sha256"
        ],
        "extraction_terminal_claim_file_sha256": controls[
            "extraction_terminal_file_hash"
        ],
        "extraction_terminal_claim_payload_sha256": controls[
            "extraction_terminal"
        ]["terminal_claim_payload_sha256"],
        "artifact_path_surface_sha256": ARTIFACT_SURFACE_SHA256,
        "reviewed_implementation_authority_file_sha256": claim[
            "reviewed_implementation_authority_file_sha256"
        ],
        "reviewed_implementation_authority_payload_sha256": claim[
            "reviewed_implementation_authority_payload_sha256"
        ],
        "preflight_file_sha256": controls["preflight_file_hash"],
        "preflight_payload_sha256": claim["preflight_payload_sha256"],
        "authorization_file_sha256": controls["authorization_file_hash"],
        "authorization_payload_sha256": claim[
            "authorization_payload_sha256"
        ],
        "authorization_verdict_file_sha256": controls[
            "verdict_file_hash"
        ],
        "authorization_verdict_payload_sha256": claim[
            "authorization_verdict_payload_sha256"
        ],
        "execution_claim_file_sha256": claim_file_hash,
        "execution_claim_payload_sha256": claim[
            "execution_claim_payload_sha256"
        ],
        "evidence_availability": profile_name,
        **availability,
        "scientific_values_emitted": False,
        "baseline_verification_published": schema["baseline_verification_published"],
    }
    if set(payload) | {"failure_receipt_payload_sha256"} != set(
        schema["exact_top_level_fields"]
    ):
        _fail("INTERNAL_SANITIZED_FAILURE")
    return _add_hash(payload, "failure_receipt_payload_sha256")


def _reverify_claimed_verifier_lineage(
    plan: Mapping[str, Any],
    original: Mapping[str, Any],
    manifest: Mapping[str, Any],
    authority: Mapping[str, Any],
    authority_file_hash: str,
    controls: Mapping[str, Any],
    candidate: Mapping[str, Any],
    repository: Path,
    *,
    now_ns: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    repeated_plan = independent_load_plan()
    repeated_original, repeated_manifest = independent_lineage(repeated_plan)
    repeated_authority, repeated_authority_hash, repeated_candidate = independent_runtime_lineage(
        repeated_plan, repository
    )
    repeated_controls = _load_verifier_controls(
        repeated_plan,
        repeated_authority,
        repeated_authority_hash,
        repeated_candidate,
        now_ns=now_ns,
        execution_claim_published=True,
    )
    if (
        repeated_plan != plan
        or repeated_original != original
        or repeated_manifest != manifest
        or repeated_authority != authority
        or repeated_authority_hash != authority_file_hash
        or repeated_candidate != candidate
        or repeated_controls != controls
    ):
        _fail("INPUT_LINEAGE_MISMATCH")
    return repeated_original, repeated_manifest


def execute_independent_verifier(
    repository: Path,
    *,
    execution_claim_id: str,
    launch_id: str,
    claimed_at_utc: str,
    now_ns: int | None = None,
) -> dict[str, Any]:
    if (
        not sys.flags.isolated
        or not sys.dont_write_bytecode
        or os.environ.get("PYTHONPATH")
    ):
        _fail("AUTHORIZATION_INVALID")
    plan = independent_load_plan()
    original, manifest = independent_lineage(plan)
    repository_root = Path(repository).resolve()
    authority, authority_file_hash, candidate = independent_runtime_lineage(
        plan, repository_root
    )
    current_ns = (
        int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)
        if now_ns is None
        else _integer(now_ns)
    )
    controls = _load_verifier_controls(
        plan, authority, authority_file_hash, candidate, now_ns=current_ns
    )
    pid = os.getpid()
    creation = _process_creation(pid)
    claim = _verifier_claim(
        plan,
        controls,
        execution_claim_id=execution_claim_id,
        launch_id=launch_id,
        claimed_at_utc=claimed_at_utc,
        owner_pid=pid,
        owner_creation=creation,
        now_ns=current_ns,
    )
    _publish(plan, "verifier_execution_claim", claim)
    claim_row = _artifact_rows(plan)["verifier_execution_claim"]
    published_claim, claim_file_hash = _receipt(
        Path(claim_row["final_path"]),
        plan["control_receipt_schemas"]["verifier_execution_claim"][
            "exact_top_level_fields"
        ],
        "execution_claim_payload_sha256",
    )
    if (
        published_claim != claim
        or published_claim["owner_hostname"] != socket.gethostname()
        or published_claim["owner_pid"] != pid
        or _process_creation(pid) != creation
    ):
        _fail("CONCURRENT_EXECUTION")

    progress = new_verifier_progress()
    try:
        original, manifest = _reverify_claimed_verifier_lineage(
            plan,
            original,
            manifest,
            authority,
            authority_file_hash,
            controls,
            candidate,
            repository_root,
            now_ns=current_ns,
        )
        rederived = independent_rederive(
            plan, original, manifest, progress=progress
        )
        compare_baseline_receipt(plan, controls["baseline"], rederived)
        leaf = independent_verification_receipt(
            plan,
            rederived,
            authority_file_sha256=authority_file_hash,
            authority_payload_sha256=authority[
                "reviewed_implementation_authority_payload_sha256"
            ],
            baseline_file_sha256=controls["baseline_file_hash"],
            baseline_payload_sha256=controls["baseline"][
                "baseline_receipt_payload_sha256"
            ],
            extraction_terminal_file_sha256=controls[
                "extraction_terminal_file_hash"
            ],
            extraction_terminal_payload_sha256=controls[
                "extraction_terminal"
            ]["terminal_claim_payload_sha256"],
            preflight_file_sha256=controls["preflight_file_hash"],
            preflight_payload_sha256=controls["preflight"][
                "preflight_payload_sha256"
            ],
            authorization_file_sha256=controls["authorization_file_hash"],
            authorization_payload_sha256=controls["authorization"][
                "authorization_payload_sha256"
            ],
            verdict_file_sha256=controls["verdict_file_hash"],
            verdict_payload_sha256=controls["verdict"][
                "authorization_verdict_payload_sha256"
            ],
            claim_file_sha256=claim_file_hash,
            claim_payload_sha256=claim["execution_claim_payload_sha256"],
        )
        terminal = _verifier_terminal(
            plan,
            controls,
            claim,
            claim_file_hash=claim_file_hash,
            leaf=leaf,
            outcome="success",
            claimed_at_utc=_utc_now(),
        )
    except Exception as raw_error:
        error_code = (
            raw_error.code
            if isinstance(raw_error, IndependentVerificationError)
            else "INTERNAL_SANITIZED_FAILURE"
        )
        failure_time = _utc_now()
        failure = _verifier_failure(
            plan,
            controls,
            claim,
            claim_file_hash=claim_file_hash,
            code=error_code,
            occurred_at_utc=failure_time,
            progress=progress,
        )
        failure_terminal = _verifier_terminal(
            plan,
            controls,
            claim,
            claim_file_hash=claim_file_hash,
            leaf=failure,
            outcome="failure",
            claimed_at_utc=failure_time,
        )
        update_verifier_progress(
            progress,
            failure_phase="verifier_failure_terminal_claim_publication",
        )
        _publish(plan, "verifier_terminal", failure_terminal)
        update_verifier_progress(
            progress,
            source_state="VERIFIER_FAILURE_TERMINAL_CLAIM_PUBLISHED",
            failure_phase="verifier_leaf_publication",
        )
        _publish(plan, "verifier_failure", failure)
        raise IndependentVerificationError(failure["failure_code"]) from None

    update_verifier_progress(
        progress, failure_phase="verifier_terminal_claim_publication"
    )
    _publish(plan, "verifier_terminal", terminal)
    update_verifier_progress(
        progress,
        source_state="VERIFIER_SUCCESS_TERMINAL_CLAIM_PUBLISHED",
        failure_phase="verifier_leaf_publication",
    )
    _publish(plan, "verifier_success", leaf)
    return leaf


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--execution-claim-id", required=True)
    parser.add_argument("--launch-id", required=True)
    parser.add_argument("--claimed-at-utc", required=True)
    args = parser.parse_args(argv)
    execute_independent_verifier(
        args.repository,
        execution_claim_id=args.execution_claim_id,
        launch_id=args.launch_id,
        claimed_at_utc=args.claimed_at_utc,
    )
    print(PASS_LINE)
    return 0


def cli(argv: list[str] | None = None) -> int:
    try:
        return main(argv)
    except SystemExit:
        raise
    except IndependentVerificationError as error:
        print(ERROR_PREFIX + error.code, file=sys.stderr)
        return 2
    except Exception:
        print(ERROR_PREFIX + "INTERNAL_SANITIZED_FAILURE", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
