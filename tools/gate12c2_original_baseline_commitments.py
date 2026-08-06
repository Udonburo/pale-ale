#!/usr/bin/env python3
"""Gate12C-2 v0.9 original-baseline commitment extraction core.

This module deliberately separates control-plane validation from the only
code path that may parse the protected payload.  Importing it performs no
filesystem access.  Protected input is opened only by ``extract_commitments``
after a caller has published and re-read an exact execution claim.
"""
from __future__ import annotations

import calendar
import copy
import ctypes
import ctypes.wintypes
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import socket
import stat
import subprocess
import sys
import types
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence


GATE_ID = "C2_ORIGINAL_BASELINE_COMMITMENT_GATE_v0.9"
PLAN_SCHEMA = "gate12c2_original_baseline_commitment_gate_plan_v0.9"
PLAN_PATH = Path(
    r"C:\Users\aoika\Documents\Research\pale-ale-local\research-program"
    r"\profile-plans\C2_ORIGINAL_BASELINE_COMMITMENT_GATE_PLAN_v0.9_2026-08-01.json"
)
PLAN_FILE_SHA256 = "ae7979779675fc032b2919a4db0887d60011dcdc510a43deef43fc090e5ef03c"
PLAN_PAYLOAD_SHA256 = "2bfaeb778494772a21975d0de22c6cc5221edbb742787fb1a34db5544eb24621"
CONTRACT_PATH = Path(
    r"C:\Users\aoika\Documents\Research\pale-ale-local\research-program"
    r"\contracts\C2_ORIGINAL_BASELINE_COMMITMENT_GATE_CONTRACT_v0.9_2026-08-01.md"
)
CONTRACT_FILE_SHA256 = "794cf22c89375424a3485a2f441ec53fc930f016d470e4cc95778cb2528ae82d"
FORMAL_DESIGN_REVIEW_PATH = Path(
    r"C:\Users\aoika\Documents\Research\pale-ale-local\research-program"
    r"\receipts\C2_ORIGINAL_BASELINE_COMMITMENT_GATE_FORMAL_FRESH_DESIGN_REVIEW_VERDICT_v0.9_2026-08-01.json"
)
FORMAL_DESIGN_REVIEW_FILE_SHA256 = (
    "5a0ba0d6ad6b5b79df819e73d7ab15831c081ad5e4e44ca6b8195e59bc97cc1e"
)
FORMAL_DESIGN_REVIEW_PAYLOAD_SHA256 = (
    "fafcbb29e998e2e1a7d9c6351e0d2a925e9c652d01dd3236c13e041ac5eac3e2"
)
ARTIFACT_PATH_SURFACE_SHA256 = (
    "7018f1a71ca8f41783f6228a7b73d25a28a37ba05462fbe7477f3fc76d1f5e2e"
)
R2_AUTHORITY_NAMESPACE_ID = "R2_6e92079"
R2_ACTIVATION_PLAN_PATH = Path(__file__).with_name(
    "gate12c2_original_baseline_r2_activation_plan.json"
)
R2_ACTIVATION_PLAN_FILE_SHA256 = (
    "ae9a9a7f46660e3ec846767c8cc01ecf7ecd7486a823b1b1491fd0189bf8e02a"
)
R2_ACTIVATION_PLAN_PAYLOAD_SHA256 = (
    "bef6f4e29f9fc61b95b40f3a3daa8b332779a4a26f7d04fa6bfac3edd3492c00"
)
R2_ARTIFACT_PATH_SURFACE_SHA256 = (
    "23c1595bc504ae3e500695679cfdb2f060370b3b6bb56652eee9d5aac0637c0b"
)
R2_OCCUPIED_V0_9_SURFACE_SHA256 = (
    "2291c234118e033d60faacfc7181c7a5b8ba4890eb0af7cc9a43f61975111438"
)
R2_TASK1_COMMIT = "6e92079cc962a9748d2c69147186aed2a59da8d0"
R2_TASK1_PARENT = "2e51e0727d456792a474e38d67b1e3ebc605a8aa"
R2R1_AUTHORITY_NAMESPACE_ID = "R2R1_20260803"
R2R1_REMEDIATION_PLAN_PATH = Path(__file__).with_name(
    "gate12c2_original_baseline_r2r1_remediation_plan.json"
)
R2R1_REMEDIATION_PLAN_FILE_SHA256 = (
    "7ccdeb61d7aad087dbf2b0d6def5d43ed3efca6a5c22852b50a9426d499d7e36"
)
R2R1_REMEDIATION_PLAN_PAYLOAD_SHA256 = (
    "8a4f0d39a4b4c1d01181b35c8e42ba2d6601e3e46f42ce98f35ccb7dceff1eb3"
)
R2R1_ARTIFACT_PATH_SURFACE_SHA256 = (
    "17fcf271d542a2b9305d3ae1b029a2e0b07c46fad596117db0aa4ffc18bef6b0"
)
R2R1_OCCUPIED_R2_SURFACE_SHA256 = (
    "f592f536259e82ae624e7e9080901fee68cf7aa9ecb276e898fa0d7b99a125a8"
)
R2R1_PARENT_COMMIT = "29058cd9289a9ccb6656878f10c8cbe3d19f11ba"
R2R1_GRANDPARENT_COMMIT = R2_TASK1_COMMIT
R2R2_HISTORICAL_CANDIDATE_COMMIT = (
    "88ad45d9c7e516e4d4fbaa2054a4ccf850dbbbf2"
)
R2R2_BASE_COMMIT = "e8bdb16e0e47296dbe4f7c04bc7ba52db8766f78"
R2_ACTIVATION_PLAN_RELATIVE_PATH = (
    "tools/gate12c2_original_baseline_r2_activation_plan.json"
)
R2_ACTIVATION_PLAN_HISTORICAL_DECLARED_PATH = Path(
    r"C:\Users\aoika\Documents\GitHub\pale-ale\tools"
    r"\gate12c2_original_baseline_r2_activation_plan.json"
)
R2_ACTIVATION_PLAN_BASE_BLOB_OID = (
    "d0bf666e6fbfc8b9a5c333a3480aea4626884343"
)
R2R1_REMEDIATION_PLAN_RELATIVE_PATH = (
    "tools/gate12c2_original_baseline_r2r1_remediation_plan.json"
)
R2R1_REMEDIATION_PLAN_HISTORICAL_DECLARED_PATH = Path(
    r"C:\Users\aoika\Documents\GitHub\pale-ale\tools"
    r"\gate12c2_original_baseline_r2r1_remediation_plan.json"
)
R2R1_REMEDIATION_PLAN_BASE_BLOB_OID = (
    "be34a081f52916d8ad9f5ed80758562143b7031c"
)
R2R2_AUTHORITY_NAMESPACE_ID = "R2R3_20260807"
R2R2_PORTABILITY_PLAN_RELATIVE_PATH = (
    "tools/gate12c2_original_baseline_r2r2_portability_plan.json"
)
R2R2_PORTABILITY_PLAN_HISTORICAL_DECLARED_PATH = Path(
    r"C:\Users\aoika\Documents\GitHub\pale-ale\tools"
    r"\gate12c2_original_baseline_r2r2_portability_plan.json"
)
# Frozen after the final focused collection; updated exactly once before the
# bounded candidate commit is created.
R2R2_PORTABILITY_PLAN_FILE_SHA256 = (
    "91c574b99ae3a48a257e75531a1f4d2569a657c3c10352c27efc551f80460874"
)
R2R2_PORTABILITY_PLAN_PAYLOAD_SHA256 = (
    "1e3f2879062b17cecff57738bdc4888a0c09de5ab88bc5e653c704dd0eadca45"
)
R2R2_ARTIFACT_PATH_SURFACE_SHA256 = (
    "3c9055988a4836524d57ff2c73af1092f6e35f55c26be3dd0493c2aae6d5fdf1"
)
R2R2_OCCUPIED_R2R1_SURFACE_SHA256 = (
    "bb67a1f98feda109f7243bc4a7a1a4d9b03244f74a005471bdad09a0526d6621"
)
R2R2_REPOSITORY_LOCAL_SURFACE_SHA256 = (
    "2bf9b5c8e7ba11738dc8d02dffa8964026787d73c602e32589df86137c1ffe61"
)
R2R2_UPSTREAM_FRAMING_SURFACE_SHA256 = (
    "c88d2a6618b5a9c1e4fd38e9c4143da955d1dcd7a7aaf0a76cffe746a2feac4b"
)
CONFIGURATION_SURFACE_SHA256 = (
    "a564c25f28e42860f0a1e8f51d4a311b4eae2b771f02dc3f62504547799f19cf"
)
IMPLEMENTATION_AUTHOR_SEPARATION_SHA256 = (
    "317fff23c2fcccfb9a6ae4a40c4d5eb77297b734f392f2642167936afca9e3c8"
)
IMPLEMENTATION_TRUST_MODEL_SHA256 = (
    "41247cd55b90fb4dc7dfb0d37f0154e7847e32a336fa0fcdb6e1d5bd6b5b944f"
)
AUTHORIZED_IMPLEMENTATION_REPOSITORY = Path(
    r"C:\Users\aoika\Documents\GitHub\pale-ale"
)
REMEDIATION_BASE_COMMIT = "2e51e0727d456792a474e38d67b1e3ebc605a8aa"
REMEDIATION_BASE_PARENT = "7f7a7a212364232e063774bcfdabb5d3f4f303b6"
PROTECTED_ROOT = Path(
    r"C:\Users\aoika\Documents\Research\pale-ale-local\throughput"
    r"\C2_DRAW_PROFILE_f9bd14d_2026-07-26"
)
EXPECTED_COMPLETE_SURFACE_SHA256 = (
    "9489e0eb14e33a328167c840a443a80392c022e365bdefd41458c9659aeda6da"
)
EXPECTED_PROTECTED_SURFACE_SHA256 = (
    "a8ef2eb83fbd0517740f5ebbb2c270ba8f4ea37f872b34d137b0447fbb6edc24"
)
EXPECTED_FILE_COUNT = 791
EXPECTED_DIRECTORY_COUNT = 23
EXPECTED_SHARD_COUNT = 768
EXPECTED_INDEX_COUNT = 9
MAXIMUM_FRESHNESS_SECONDS = 1800

SHARD_SCHEMA = "gate12c2_development_outer_shard_v0.3"
INDEX_SCHEMA = "gate12c2_development_shard_index_v0.3"
SCIENTIFIC_PROJECTION_SCHEMA = "gate12c2_development_scientific_projection_v0.3"
SEMANTIC_INDEX_SCHEMA = "gate12c2_semantic_index_commitment_v0.1"
OUTER_EXPERIMENT_SCHEMA = "gate12c2_outer_experiment_v0.5"
N1_NULL_ARM_ID = "gate12c2_n1_role_constrained_frame_reassignment_v0.1"
S2_NULL_ARM_ID = (
    "gate12c2_s2_independent_edge_orientation_stress_v0.1"
)

S2_COMPONENT_ARMS = (
    "observed",
    "N1",
    "graph_unconstrained_stressor",
)
S2_ALWAYS_DEFINED_COMPONENT_FIELDS = (
    "a_q",
    "u_q",
    "v_q",
    "x_q",
    "y_q",
)
S2_CONDITIONALLY_DEFINED_COMPONENT_FIELDS = (
    "c_q",
    "p_L_q",
    "p_R_q",
)
S2_COMPONENT_FIELDS = (
    S2_ALWAYS_DEFINED_COMPONENT_FIELDS
    + S2_CONDITIONALLY_DEFINED_COMPONENT_FIELDS
)
S2_COMPONENT_COUNT_FIELDS = (
    "expected_count",
    "defined_count",
    "degenerate_count",
    "unexpected_missing_count",
    "nonfinite_count",
)

# This surface is a code-level strengthening of the frozen receipt schemas.  It
# is reconstructed by every consumer and propagated literally through the
# candidate -> review -> authority chain.  The row digests were independently
# derived from the two frozen scientific source blobs without reading payloads.
REVIEW_SURFACE_SOURCE_DEPENDENCIES_SHA256 = (
    "3231bfaffc1b0cb9c7a24f6f48d2d5b14da7f4b1014a0728caa35e4ef4062857"
)
REVIEW_SURFACE_COMPATIBILITY_ROWS_SHA256 = (
    "73de8992e52506b83eee075c6d3bee6ee2accd0c9ee255d4059ed68b40951ccc"
)
REVIEW_SURFACE_NORMATIVE_ROWS_SHA256 = (
    "79a0d652c3adcb3986dc0ec3ad2c1fb0b83fe94803469a686383664f7e53689c"
)
REVIEW_SURFACE_MUTATION_APPLICABILITY_SHA256 = (
    "b42f7050c489350f8c87ac31c34752ca35731dd90139495df904b0e624fd9dea"
)
REVIEW_SURFACE_REQUIRED_MUTATIONS_SHA256 = (
    "e4f093941689e76f1acba490b36cd15fe3bffdb3ecdb835e8cc35f47994beb56"
)
REVIEW_SURFACE_IDENTITY_SHA256 = (
    "4aedbc28d2314a51403cc5e6c01b93629553716d85045a24daf074f985d7dbf9"
)
REVIEW_SURFACE_MUTATION_DIMENSIONS = (
    "missing_field",
    "extra_sibling_field",
    "wrong_primitive_or_container_type",
    "bool_as_integer",
    "bool_as_float",
    "wrong_enum_or_value",
    "below_lower_bound",
    "above_upper_bound",
    "nonfinite_number",
    "forbidden_negative_zero",
    "wrong_list_cardinality",
    "wrong_list_order",
    "duplicate_list_item",
    "conditional_presence_violation",
    "nested_row_violation",
    "cross_link_hash_consistent_adversarial_mutation",
)
# Declarative audit surface only.  The verifier carries its own byte-for-byte
# copy while implementing every check independently.
EXTRACTOR_SCHEMA_INVARIANT_MANIFEST = (
    "json.utf8_no_bom",
    "json.duplicate_keys_rejected",
    "json.nonfinite_numbers_rejected",
    "json.canonical_bytes_required",
    "json.boolean_is_not_integer",
    "index.exact_required_or_optional_operational_fields",
    "index.schema_version_exact",
    "index.epistemic_status_development_shard_index_only",
    "index.surface_id_development",
    "index.locked_execution_authorized_false",
    "index.plan_payload_sha256_exact",
    "index.worker_count_integer_minimum_one",
    "index.merge_order_exact",
    "index.outer_experiment_count_exact",
    "index.all_outer_indices_present_boolean_true",
    "index.shards_nonempty_exact_coverage",
    "index.scientific_projection_schema_version_exact",
    "index.scientific_projection_sha256_exact",
    "index.self_hash_exact",
    "index.operational_execution_metrics_optional_object_finite",
    "shard.relative_paths_exact_unique_coverage",
    "shard.gzip_single_member",
    "shard.canonical_json",
    "shard.exact_fields",
    "shard.schema_version_exact",
    "shard.epistemic_status_development_outer_shard_only",
    "shard.surface_id_development",
    "shard.locked_execution_authorized_false",
    "shard.plan_payload_sha256_exact",
    "shard.outer_experiment_index_exact",
    "shard.self_hash_exact",
    "shard.result_object",
    "shard.result_payload_sha256_exact",
    "result.exact_fields_by_regime",
    "result.fixed_envelope_values",
    "result.max_draw_attempts_positive_exact",
    "result.execution_contract_exact",
    "result.numerical_contract_exact",
    "result.effect_strength_exact",
    "result.s2_fixed_arms_and_counts",
    "pipeline.exact_fields",
    "pipeline.boolean_fields_strict",
    "pipeline.endpoint_count_integer_0_to_24",
    "pipeline.directional_count_integer_0_to_24",
    "pipeline.supporting_run_count_integer_0_to_12",
    "pipeline.discordant_run_count_integer_0_to_12",
    "pipeline.grid_outcome_nonempty_ascii",
    "index_row.exact_required_or_optional_operational_fields",
    "index_row.reused_existing_shard_boolean",
    "index_row.operational_metrics_optional_object_finite",
    "index_row.rebuilt_values_exact",
    "projection.payload_exact",
    "projection.sha256_exact",
    "commitment.outer_id_surface_sha256_exact",
    "commitment.result_surface_sha256_exact",
    "commitment.semantic_index_sha256_exact",
    "coverage.zero_coverage_rejected",
)

EXTRACTION_PASS_LINE = "gate12c2-original-baseline:PASS"
VERIFICATION_PASS_LINE = "gate12c2-original-baseline-verification:PASS"
AUTHORIZATION_VERIFICATION_PASS_LINE = (
    "gate12c2-original-baseline-authorization-verification:PASS"
)
EXTRACTION_ERROR_PREFIX = "gate12c2-original-baseline:ERROR:"
VERIFICATION_ERROR_PREFIX = "gate12c2-original-baseline-verification:ERROR:"
AUTHORIZATION_ERROR_PREFIX = (
    "gate12c2-original-baseline-authorization-verification:ERROR:"
)

FAILURE_CODES = frozenset(
    {
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
)

IMPLEMENTATION_PATHS = (
    "tools/gate12c2_original_baseline_commitments.py",
    "tools/build_gate12c2_original_baseline_implementation_binding.py",
    "tools/build_gate12c2_original_baseline_reviewed_authority.py",
    "tools/issue_gate12c2_original_baseline_preflight.py",
    "tools/issue_gate12c2_original_baseline_authorization.py",
    "tools/run_gate12c2_original_baseline_extraction.py",
    "tools/verify_gate12c2_original_baseline_authorization.py",
    "tools/verify_gate12c2_original_baseline_commitments.py",
    "tools/test_gate12c2_original_baseline_commitments.py",
    "tools/test_gate12c2_original_baseline_commitments_adversarial.py",
)
IMPLEMENTATION_ROLE_BY_PATH = {
    "tools/gate12c2_original_baseline_commitments.py": "extraction_core",
    "tools/build_gate12c2_original_baseline_implementation_binding.py": (
        "implementation_binding_builder"
    ),
    "tools/build_gate12c2_original_baseline_reviewed_authority.py": (
        "reviewed_authority_builder"
    ),
    "tools/issue_gate12c2_original_baseline_preflight.py": "preflight_issuer",
    "tools/issue_gate12c2_original_baseline_authorization.py": (
        "authorization_issuer"
    ),
    "tools/run_gate12c2_original_baseline_extraction.py": "extraction_runner",
    "tools/verify_gate12c2_original_baseline_authorization.py": (
        "authorization_verifier"
    ),
    "tools/verify_gate12c2_original_baseline_commitments.py": (
        "independent_verifier"
    ),
    "tools/test_gate12c2_original_baseline_commitments.py": "primary_tests",
    "tools/test_gate12c2_original_baseline_commitments_adversarial.py": (
        "adversarial_tests"
    ),
}


SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
GIT_OID_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
IDENTIFIER_RE = re.compile(r"[A-Za-z0-9._-]+\Z")
UTC_RE = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})T(?P<time>\d{2}:\d{2}:\d{2})"
    r"(?P<fraction>\.\d{1,9})?(?P<zone>Z|\+00:00)\Z"
)


class Gate12C2OriginalBaselineError(ValueError):
    """Closed, public-safe gate failure carrying only an allowlisted code."""

    def __init__(self, code: str) -> None:
        selected = code if code in FAILURE_CODES else "INTERNAL_SANITIZED_FAILURE"
        self.code = selected
        super().__init__(selected)


class DuplicateKeyError(ValueError):
    """Internal marker used by the duplicate-key rejecting JSON loader."""


def _raise(code: str) -> None:
    raise Gate12C2OriginalBaselineError(code)


def canonical_json_bytes(value: object) -> bytes:
    """Return the frozen RFC8259 canonical byte representation."""

    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, OverflowError):
        _raise("INPUT_SCHEMA_INVALID")
    return text.encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_blob_oid(value: bytes, object_format: str = "sha1") -> str:
    if object_format not in {"sha1", "sha256"}:
        _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    digest = hashlib.new(object_format)
    digest.update(f"blob {len(value)}\0".encode("ascii"))
    digest.update(value)
    return digest.hexdigest()


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def is_git_oid(value: object) -> bool:
    return isinstance(value, str) and GIT_OID_RE.fullmatch(value) is not None


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError
        result[key] = value
    return result


def _validate_json_domain(value: object) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            _raise("INPUT_SCHEMA_INVALID")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_domain(item)
        return
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            _raise("INPUT_SCHEMA_INVALID")
        for item in value.values():
            _validate_json_domain(item)
        return
    _raise("INPUT_SCHEMA_INVALID")


def strict_json_loads(raw: bytes, *, canonical: bool = False) -> Any:
    """Decode UTF-8 JSON, rejecting duplicate keys and non-finite values."""

    if not isinstance(raw, bytes) or raw.startswith(b"\xef\xbb\xbf"):
        _raise("INPUT_SCHEMA_INVALID")
    try:
        text = raw.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except DuplicateKeyError:
        _raise("DUPLICATE_JSON_KEY")
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
        _raise("INPUT_SCHEMA_INVALID")
    _validate_json_domain(value)
    if canonical and raw != canonical_json_bytes(value):
        _raise("INPUT_SCHEMA_INVALID")
    return value


def require_mapping(value: object, *, code: str = "INPUT_SCHEMA_INVALID") -> dict[str, Any]:
    if not isinstance(value, dict):
        _raise(code)
    return value


def require_exact_keys(
    value: Mapping[str, Any], expected: Iterable[str], *, code: str = "INPUT_SCHEMA_INVALID"
) -> None:
    if set(value) != set(expected):
        _raise(code)


REVIEW_SURFACE_BOUND_ROLES = frozenset(
    {
        "implementation_candidate_binding",
        "fresh_implementation_review_verdict",
        "reviewed_implementation_authority",
    }
)


def recompute_implementation_trust_model_sha256(
    plan: Mapping[str, Any],
    *,
    code: str = "INPUT_LINEAGE_MISMATCH",
) -> str:
    trust = require_mapping(
        plan.get("implementation_trust_model_contract"),
        code=code,
    )
    computed = sha256_bytes(canonical_json_bytes(trust))
    if (
        computed != IMPLEMENTATION_TRUST_MODEL_SHA256
        or plan.get("implementation_trust_model_sha256") != computed
    ):
        _raise(code)
    return computed


def review_surface_identity(
    plan: Mapping[str, Any],
    *,
    code: str = "INPUT_LINEAGE_MISMATCH",
) -> dict[str, Any]:
    binding = require_mapping(
        plan.get("implementation_binding_contract"),
        code=code,
    )
    dependencies = binding.get("scientific_dependencies")
    if not isinstance(dependencies, list):
        _raise(code)
    dependency_digest = sha256_bytes(canonical_json_bytes(dependencies))
    if dependency_digest != REVIEW_SURFACE_SOURCE_DEPENDENCIES_SHA256:
        _raise(code)
    payload = {
        "schema_version": (
            "gate12c2_original_baseline_review_surface_identity_v0.1"
        ),
        "source_dependency_surface_sha256": dependency_digest,
        "canonical_row_order": (
            "compatibility: schema_id then field_path; normative: row_id; "
            "mutation: row_id then declared mutation_dimensions order"
        ),
        "compatibility_row_count": 662,
        "compatibility_surface_sha256": (
            REVIEW_SURFACE_COMPATIBILITY_ROWS_SHA256
        ),
        "compatibility_digest_domain": (
            "sha256(canonical_json([{schema_id,field_path,json_type,presence}])); "
            "no LF"
        ),
        "normative_row_count": 841,
        "normative_surface_sha256": REVIEW_SURFACE_NORMATIVE_ROWS_SHA256,
        "normative_digest_domain": (
            "sha256(canonical_json(normative rows sorted by row_id)); no LF"
        ),
        "mutation_dimensions": list(REVIEW_SURFACE_MUTATION_DIMENSIONS),
        "mutation_applicability_cell_count": 13456,
        "mutation_applicability_surface_sha256": (
            REVIEW_SURFACE_MUTATION_APPLICABILITY_SHA256
        ),
        "mutation_applicability_digest_domain": (
            "sha256(canonical_json([{row_id,dimension,required}])); row_id "
            "then declared dimension order; no LF"
        ),
        "required_mutation_count": 6487,
        "required_mutation_surface_sha256": (
            REVIEW_SURFACE_REQUIRED_MUTATIONS_SHA256
        ),
        "required_mutation_digest_domain": (
            "sha256(canonical_json([{row_id,dimension}] required=true "
            "subset)); inherited order; no LF"
        ),
    }
    computed = sha256_bytes(canonical_json_bytes(payload))
    if computed != REVIEW_SURFACE_IDENTITY_SHA256:
        _raise(code)
    return {
        **payload,
        "review_surface_identity_sha256": computed,
    }


def validate_review_surface_identity(
    plan: Mapping[str, Any],
    supplied: object,
    *,
    code: str = "INPUT_LINEAGE_MISMATCH",
) -> dict[str, Any]:
    value = require_mapping(supplied, code=code)
    expected = review_surface_identity(plan, code=code)
    require_exact_keys(value, expected, code=code)
    if value != expected:
        _raise(code)
    return value


def artifact_exact_fields(
    plan: Mapping[str, Any],
    role: str,
) -> tuple[str, ...]:
    if role == "implementation_candidate_binding":
        fields = plan["implementation_binding_contract"][
            "exact_top_level_fields"
        ]
    elif role == "fresh_implementation_review_verdict":
        fields = plan["review_receipt_schemas"][role]["exact_top_level_fields"]
    elif role == "reviewed_implementation_authority":
        fields = plan["reviewed_implementation_authority_contract"][
            "exact_top_level_fields"
        ]
    else:
        _raise("INPUT_LINEAGE_MISMATCH")
    result = list(fields)
    if "review_surface_identity" in result:
        _raise("INPUT_LINEAGE_MISMATCH")
    self_hash = result.pop()
    if not self_hash.endswith("_payload_sha256"):
        _raise("INPUT_LINEAGE_MISMATCH")
    result.extend(("review_surface_identity", self_hash))
    return tuple(result)


def require_bool(value: object, *, code: str = "INPUT_SCHEMA_INVALID") -> bool:
    if type(value) is not bool:
        _raise(code)
    return value


def require_int(
    value: object,
    *,
    minimum: int = 0,
    maximum: int = (1 << 63) - 1,
    code: str = "INPUT_SCHEMA_INVALID",
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _raise(code)
    return value


def require_text(
    value: object,
    *,
    allow_empty: bool = False,
    ascii_only: bool = False,
    code: str = "INPUT_SCHEMA_INVALID",
) -> str:
    if not isinstance(value, str) or (not allow_empty and value == ""):
        _raise(code)
    if "\x00" in value or (ascii_only and not value.isascii()):
        _raise(code)
    return value


def parse_utc_ns(value: object, *, code: str = "INPUT_SCHEMA_INVALID") -> int:
    text = require_text(value, ascii_only=True, code=code)
    match = UTC_RE.fullmatch(text)
    if match is None:
        _raise(code)
    try:
        base = datetime.strptime(
            f"{match.group('date')}T{match.group('time')}", "%Y-%m-%dT%H:%M:%S"
        ).replace(tzinfo=timezone.utc)
    except ValueError:
        _raise(code)
    fraction = match.group("fraction")
    nanos = int((fraction[1:] if fraction else "0").ljust(9, "0"))
    return calendar.timegm(base.utctimetuple()) * 1_000_000_000 + nanos


def utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def require_fresh_interval(
    issued_at_utc: object,
    expires_at_utc: object,
    *,
    now_ns: int | None = None,
    maximum_age_seconds: int = MAXIMUM_FRESHNESS_SECONDS,
    code: str = "AUTHORIZATION_INVALID",
) -> int:
    if now_ns is not None and (isinstance(now_ns, bool) or not isinstance(now_ns, int)):
        _raise(code)
    if isinstance(maximum_age_seconds, bool) or not isinstance(
        maximum_age_seconds, int
    ):
        _raise(code)
    issued = parse_utc_ns(issued_at_utc, code=code)
    expires = parse_utc_ns(expires_at_utc, code=code)
    current = (
        int(now_ns)
        if now_ns is not None
        else int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)
    )
    if expires <= issued or expires - issued > maximum_age_seconds * 1_000_000_000:
        _raise(code)
    if not issued <= current < expires:
        _raise("AUTHORIZATION_INVALID")
    return expires - current


def require_identifier(value: object, *, code: str = "INPUT_SCHEMA_INVALID") -> str:
    text = require_text(value, ascii_only=True, code=code)
    if IDENTIFIER_RE.fullmatch(text) is None:
        _raise(code)
    return text


def canonical_receipt_bytes(payload: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(dict(payload)) + b"\n"


def add_self_hash(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in payload:
        _raise("INPUT_SCHEMA_INVALID")
    result = dict(payload)
    result[field] = sha256_bytes(canonical_json_bytes(result))
    return result


def verify_self_hash(
    payload: Mapping[str, Any],
    field: str,
    *,
    include_lf: bool = False,
    code: str = "INPUT_SCHEMA_INVALID",
) -> str:
    claimed = payload.get(field)
    if not is_sha256(claimed):
        _raise(code)
    unhashed = dict(payload)
    unhashed.pop(field, None)
    domain = canonical_json_bytes(unhashed) + (b"\n" if include_lf else b"")
    if claimed != sha256_bytes(domain):
        _raise(code)
    return str(claimed)


def read_exact_bytes(path: Path, expected_sha256: str, *, code: str) -> bytes:
    try:
        raw = Path(path).read_bytes()
    except OSError:
        _raise(code)
    if sha256_bytes(raw) != expected_sha256:
        _raise(code)
    return raw


def read_canonical_receipt(
    path: Path,
    *,
    expected_file_sha256: str | None = None,
    hash_field: str,
    expected_payload_sha256: str | None = None,
    payload_hash_includes_lf: bool = False,
    code: str = "INPUT_LINEAGE_MISMATCH",
) -> dict[str, Any]:
    try:
        raw = Path(path).read_bytes()
    except OSError:
        _raise(code)
    if expected_file_sha256 is not None and sha256_bytes(raw) != expected_file_sha256:
        _raise(code)
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        _raise(code)
    payload = require_mapping(strict_json_loads(raw[:-1], canonical=True), code=code)
    digest = verify_self_hash(
        payload, hash_field, include_lf=payload_hash_includes_lf, code=code
    )
    if expected_payload_sha256 is not None and digest != expected_payload_sha256:
        _raise(code)
    if raw != canonical_receipt_bytes(payload):
        _raise(code)
    return payload


def _walk_named_values(value: object, name: str, *, active: bool = True) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child_active = active and not key.startswith("historical_v0_")
            if child_active and key == name:
                found.append(item)
            found.extend(_walk_named_values(item, name, active=child_active))
    elif isinstance(value, list):
        for item in value:
            found.extend(_walk_named_values(item, name, active=active))
    return found


def validate_frozen_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Validate all machine-checkable frozen v0.9 design invariants."""

    supplied = dict(plan)
    if supplied.get("schema_version") != PLAN_SCHEMA:
        _raise("INPUT_LINEAGE_MISMATCH")
    verify_self_hash(
        supplied,
        "plan_payload_sha256",
        code="INPUT_LINEAGE_MISMATCH",
    )
    if supplied["plan_payload_sha256"] != PLAN_PAYLOAD_SHA256:
        _raise("INPUT_LINEAGE_MISMATCH")
    if supplied.get("contract_file_sha256") != CONTRACT_FILE_SHA256:
        _raise("INPUT_LINEAGE_MISMATCH")
    if supplied.get("contract_path") != str(CONTRACT_PATH):
        _raise("INPUT_LINEAGE_MISMATCH")
    if supplied.get("artifact_path_surface_sha256") != ARTIFACT_PATH_SURFACE_SHA256:
        _raise("INPUT_LINEAGE_MISMATCH")
    rows = supplied.get("artifact_path_surface")
    if not isinstance(rows, list) or len(rows) != 18:
        _raise("INPUT_LINEAGE_MISMATCH")
    if rows != sorted(rows, key=lambda row: row.get("role", "")):
        _raise("INPUT_LINEAGE_MISMATCH")
    if sha256_bytes(canonical_json_bytes(rows)) != ARTIFACT_PATH_SURFACE_SHA256:
        _raise("INPUT_LINEAGE_MISMATCH")
    roles: set[str] = set()
    paths: set[str] = set()
    for row_value in rows:
        row = require_mapping(row_value, code="INPUT_LINEAGE_MISMATCH")
        require_exact_keys(
            row,
            {"role", "final_path", "pending_path", "publication_mode", "lifecycle_scope"},
            code="INPUT_LINEAGE_MISMATCH",
        )
        role = require_text(row["role"], ascii_only=True, code="INPUT_LINEAGE_MISMATCH")
        final_path = require_text(row["final_path"], code="INPUT_LINEAGE_MISMATCH")
        pending_path = require_text(row["pending_path"], code="INPUT_LINEAGE_MISMATCH")
        if (
            role in roles
            or final_path in paths
            or pending_path in paths
            or pending_path != final_path + ".pending-v0.9"
            or row["publication_mode"] != "MoveFileExW_nonreplace_write_through"
        ):
            _raise("INPUT_LINEAGE_MISMATCH")
        roles.add(role)
        paths.update({final_path, pending_path})
    nested = _walk_named_values(supplied, "artifact_path_surface_sha256")
    if not nested or any(value != ARTIFACT_PATH_SURFACE_SHA256 for value in nested):
        _raise("INPUT_LINEAGE_MISMATCH")
    configurations = supplied.get("configuration_surface")
    if not isinstance(configurations, list) or len(configurations) != 9:
        _raise("INPUT_LINEAGE_MISMATCH")
    if configurations != sorted(
        configurations, key=lambda row: row.get("configuration_id", "")
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    if sha256_bytes(canonical_json_bytes(configurations)) != CONFIGURATION_SURFACE_SHA256:
        _raise("INPUT_LINEAGE_MISMATCH")
    if supplied.get("configuration_surface_sha256") != CONFIGURATION_SURFACE_SHA256:
        _raise("INPUT_LINEAGE_MISMATCH")
    expected_outer = 0
    configuration_ids: set[str] = set()
    output_paths: set[str] = set()
    for value in configurations:
        row = require_mapping(value, code="INPUT_LINEAGE_MISMATCH")
        require_exact_keys(
            row,
            {
                "configuration_id",
                "draw_count",
                "original_subplan_payload_sha256",
                "outer_experiment_count",
                "output_relative_path",
                "regime_id",
            },
            code="INPUT_LINEAGE_MISMATCH",
        )
        identifier = require_text(
            row["configuration_id"], ascii_only=True, code="INPUT_LINEAGE_MISMATCH"
        )
        relative = validate_relative_manifest_path(
            row["output_relative_path"], allow_directory=True
        )
        if identifier in configuration_ids or relative in output_paths:
            _raise("INPUT_LINEAGE_MISMATCH")
        configuration_ids.add(identifier)
        output_paths.add(relative)
        require_int(row["draw_count"], minimum=1, code="INPUT_LINEAGE_MISMATCH")
        expected_outer += require_int(
            row["outer_experiment_count"], minimum=1, code="INPUT_LINEAGE_MISMATCH"
        )
        if not is_sha256(row["original_subplan_payload_sha256"]):
            _raise("INPUT_LINEAGE_MISMATCH")
    totals = require_mapping(supplied.get("surface_totals"), code="INPUT_LINEAGE_MISMATCH")
    if totals != {
        "configuration_count": 9,
        "index_count": 9,
        "outer_experiment_count": 768,
        "shard_count": 768,
    } or expected_outer != 768:
        _raise("INPUT_LINEAGE_MISMATCH")
    author_separation = require_mapping(
        supplied.get("implementation_author_separation_contract"),
        code="INPUT_LINEAGE_MISMATCH",
    )
    if sha256_bytes(canonical_json_bytes(author_separation)) != (
        IMPLEMENTATION_AUTHOR_SEPARATION_SHA256
    ) or supplied.get("implementation_author_separation_contract_sha256") != (
        IMPLEMENTATION_AUTHOR_SEPARATION_SHA256
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    recompute_implementation_trust_model_sha256(supplied)
    review_surface_identity(supplied)
    if tuple(supplied.get("bounded_implementation_scope_after_fresh_design_pass", ())) != IMPLEMENTATION_PATHS:
        _raise("INPUT_LINEAGE_MISMATCH")
    _validate_design_algebras(supplied)
    return supplied


def _validate_design_algebras(plan: Mapping[str, Any]) -> None:
    codes = plan.get("failure_codes")
    if not isinstance(codes, list) or frozenset(codes) != FAILURE_CODES or len(codes) != 21:
        _raise("INPUT_LINEAGE_MISMATCH")
    matrix = plan.get("failure_matrix")
    if not isinstance(matrix, list) or len(matrix) != 94:
        _raise("INPUT_LINEAGE_MISMATCH")
    matrix_keys: set[bytes] = set()
    used_codes: set[str] = set()
    used_phases: set[str] = set()
    profiles = require_mapping(
        plan.get("failure_evidence_availability_profiles"),
        code="INPUT_LINEAGE_MISMATCH",
    )
    for row_value in matrix:
        row = require_mapping(row_value, code="INPUT_LINEAGE_MISMATCH")
        require_exact_keys(
            row,
            {
                "availability_profile",
                "failure_code",
                "failure_phase",
                "failure_receipt_allowed",
                "scope",
                "source_state",
                "terminal_action",
            },
            code="INPUT_LINEAGE_MISMATCH",
        )
        key = canonical_json_bytes(row)
        if key in matrix_keys:
            _raise("INPUT_LINEAGE_MISMATCH")
        matrix_keys.add(key)
        scope = row["scope"]
        if scope not in {"extraction", "verifier"}:
            _raise("INPUT_LINEAGE_MISMATCH")
        if row["availability_profile"] not in require_mapping(
            profiles.get(scope), code="INPUT_LINEAGE_MISMATCH"
        ):
            _raise("INPUT_LINEAGE_MISMATCH")
        require_bool(row["failure_receipt_allowed"], code="INPUT_LINEAGE_MISMATCH")
        used_codes.add(str(row["failure_code"]))
        used_phases.add(str(row["failure_phase"]))
    if used_codes != FAILURE_CODES or used_phases != set(plan.get("failure_phases", ())):
        _raise("INPUT_LINEAGE_MISMATCH")
    state_model = require_mapping(plan.get("state_model"), code="INPUT_LINEAGE_MISMATCH")
    states = set(state_model.get("states", ()))
    events = set(state_model.get("events", ()))
    transitions = state_model.get("transitions")
    terminal = set(state_model.get("terminal_states", ()))
    if not isinstance(transitions, list) or not states or not events:
        _raise("INPUT_LINEAGE_MISMATCH")
    transition_keys: set[tuple[str, str]] = set()
    used_events: set[str] = set()
    for value in transitions:
        if not isinstance(value, list) or len(value) != 3:
            _raise("INPUT_LINEAGE_MISMATCH")
        source, event, target = value
        if source not in states or target not in states or event not in events:
            _raise("INPUT_LINEAGE_MISMATCH")
        if source in terminal or (source, event) in transition_keys:
            _raise("INPUT_LINEAGE_MISMATCH")
        transition_keys.add((source, event))
        used_events.add(event)
    if used_events != events:
        _raise("INPUT_LINEAGE_MISMATCH")
    reachable = {state_model.get("initial_state")}
    changed = True
    while changed:
        changed = False
        for source, _event, target in transitions:
            if source in reachable and target not in reachable:
                reachable.add(target)
                changed = True
    if reachable != states:
        _raise("INPUT_LINEAGE_MISMATCH")
    lifecycle = require_mapping(
        plan.get("artifact_lifecycle_contract"), code="INPUT_LINEAGE_MISMATCH"
    )
    phases = lifecycle.get("stable_phases")
    roles = lifecycle.get("roles")
    if (
        not isinstance(phases, list)
        or not isinstance(roles, list)
        or len(phases) != 33
        or len(roles) != 18
        or lifecycle.get("cell_count") != 594
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    signatures: set[bytes] = set()
    for phase_value in phases:
        phase = require_mapping(phase_value, code="INPUT_LINEAGE_MISMATCH")
        must_exist = set(phase.get("must_exist", ()))
        must_absent = set(phase.get("must_be_absent", ()))
        if must_exist & must_absent or must_exist | must_absent != set(roles):
            _raise("INPUT_LINEAGE_MISMATCH")
        signature = canonical_json_bytes(
            {
                "must_exist": sorted(must_exist),
                "must_absent": sorted(must_absent),
                "required_outcomes": phase.get("required_outcomes"),
                "temporal_predicate": phase.get("temporal_predicate"),
                "liveness_predicate": phase.get("liveness_predicate"),
            }
        )
        if signature in signatures:
            _raise("INPUT_LINEAGE_MISMATCH")
        signatures.add(signature)


def load_frozen_plan(path: Path = PLAN_PATH) -> dict[str, Any]:
    raw = read_exact_bytes(path, PLAN_FILE_SHA256, code="INPUT_LINEAGE_MISMATCH")
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        _raise("INPUT_LINEAGE_MISMATCH")
    value = require_mapping(
        strict_json_loads(raw[:-1], canonical=True), code="INPUT_LINEAGE_MISMATCH"
    )
    if raw != canonical_receipt_bytes(value):
        _raise("INPUT_LINEAGE_MISMATCH")
    return validate_frozen_plan(value)


def artifact_surface_sha256(
    plan: Mapping[str, Any],
    *,
    code: str = "INPUT_LINEAGE_MISMATCH",
) -> str:
    rows = plan.get("artifact_path_surface")
    claimed = plan.get("artifact_path_surface_sha256")
    if (
        not isinstance(rows, list)
        or not is_sha256(claimed)
        or sha256_bytes(canonical_json_bytes(rows)) != claimed
    ):
        _raise(code)
    return str(claimed)


def _merge_control_delta(
    target: MutableMapping[str, Any],
    delta: Mapping[str, Any],
) -> None:
    for key, value in delta.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_control_delta(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _replace_active_string(
    value: object,
    old: str,
    new: str,
    *,
    active: bool = True,
) -> object:
    if isinstance(value, dict):
        return {
            key: (
                copy.deepcopy(item)
                if key.startswith("historical_v0_")
                else _replace_active_string(item, old, new, active=active)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _replace_active_string(item, old, new, active=active)
            for item in value
        ]
    return new if active and value == old else copy.deepcopy(value)


def _validate_r2_legacy_occupancy(
    base_plan: Mapping[str, Any],
    r2_plan: Mapping[str, Any],
) -> None:
    occupied = require_mapping(
        r2_plan.get("occupied_v0_9"),
        code="INPUT_LINEAGE_MISMATCH",
    )
    candidate_row = require_mapping(
        occupied.get("candidate_binding"),
        code="INPUT_LINEAGE_MISMATCH",
    )
    review_row = require_mapping(
        occupied.get("review_verdict"),
        code="INPUT_LINEAGE_MISMATCH",
    )
    candidate = read_canonical_receipt(
        Path(candidate_row["path"]),
        expected_file_sha256=candidate_row["file_sha256"],
        hash_field="implementation_candidate_binding_payload_sha256",
        expected_payload_sha256=candidate_row["payload_sha256"],
    )
    review = read_canonical_receipt(
        Path(review_row["path"]),
        expected_file_sha256=review_row["file_sha256"],
        hash_field="fresh_implementation_review_payload_sha256",
        expected_payload_sha256=review_row["payload_sha256"],
    )
    if (
        candidate.get("source_commit") != candidate_row["source_commit"]
        or review.get("implementation_source_commit")
        != review_row["source_commit"]
        or review.get("state") != review_row["state"]
        or review.get("outcome_kind") != review_row["outcome_kind"]
        or any(
            review.get(field) != review_row[field]
            for field in ("P0_count", "P1_count", "P2_count")
        )
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    old_rows = {
        row["role"]: row
        for row in base_plan["artifact_path_surface"]
    }
    for role in occupied["required_absent_final_roles"]:
        if role not in old_rows or Path(old_rows[role]["final_path"]).exists():
            _raise("UNEXPECTED_ARTIFACT")
    for role in occupied["required_absent_pending_roles"]:
        if role not in old_rows or Path(old_rows[role]["pending_path"]).exists():
            _raise("UNEXPECTED_ARTIFACT")


def validate_r2_activation_plan(
    base_plan: Mapping[str, Any],
    r2_plan: Mapping[str, Any],
    *,
    check_legacy_occupancy: bool = True,
) -> dict[str, Any]:
    supplied = dict(r2_plan)
    require_exact_keys(
        supplied,
        {
            "activation_lineage",
            "activation_plan_relative_path",
            "artifact_path_surface",
            "artifact_path_surface_sha256",
            "base_contract",
            "base_plan",
            "candidate_manifest_contract",
            "clean_restore_receipt_contract",
            "formal_design_pass",
            "fresh_review_contract_overlay",
            "fresh_review_evidence_contract",
            "fresh_review_packet_path",
            "implementation_binding_contract_overlay",
            "namespace_id",
            "occupied_v0_9",
            "occupied_v0_9_surface_sha256",
            "preserved_identities",
            "protected_surface_policy",
            "publication_policy",
            "purpose",
            "r2_activation_plan_payload_sha256",
            "reviewed_authority_contract_overlay",
            "schema_version",
            "state",
        },
        code="INPUT_LINEAGE_MISMATCH",
    )
    verify_self_hash(
        supplied,
        "r2_activation_plan_payload_sha256",
        code="INPUT_LINEAGE_MISMATCH",
    )
    if (
        supplied["r2_activation_plan_payload_sha256"]
        != R2_ACTIVATION_PLAN_PAYLOAD_SHA256
        or supplied.get("schema_version")
        != "gate12c2_original_baseline_r2_activation_plan_v0.1"
        or supplied.get("namespace_id") != R2_AUTHORITY_NAMESPACE_ID
        or supplied.get("state") != "R2_CONTROL_LINEAGE_FROZEN"
        or supplied.get("activation_plan_relative_path")
        != "tools/gate12c2_original_baseline_r2_activation_plan.json"
        or supplied.get("artifact_path_surface_sha256")
        != R2_ARTIFACT_PATH_SURFACE_SHA256
        or supplied.get("occupied_v0_9_surface_sha256")
        != R2_OCCUPIED_V0_9_SURFACE_SHA256
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    if sha256_bytes(
        canonical_json_bytes(supplied["occupied_v0_9"])
    ) != R2_OCCUPIED_V0_9_SURFACE_SHA256:
        _raise("INPUT_LINEAGE_MISMATCH")
    expected_lineage = {
        "activation_commit_parent": R2_TASK1_COMMIT,
        "activation_commit_parent_count": 1,
        "task1_commit": R2_TASK1_COMMIT,
        "task1_parent": R2_TASK1_PARENT,
        "task1_parent_count": 1,
    }
    if supplied.get("activation_lineage") != expected_lineage:
        _raise("INPUT_LINEAGE_MISMATCH")
    if supplied.get("base_plan") != {
        "file_sha256": PLAN_FILE_SHA256,
        "path": str(PLAN_PATH),
        "payload_sha256": PLAN_PAYLOAD_SHA256,
    } or supplied.get("base_contract") != {
        "file_sha256": CONTRACT_FILE_SHA256,
        "path": str(CONTRACT_PATH),
    } or supplied.get("formal_design_pass") != {
        "file_sha256": FORMAL_DESIGN_REVIEW_FILE_SHA256,
        "path": str(FORMAL_DESIGN_REVIEW_PATH),
        "payload_sha256": FORMAL_DESIGN_REVIEW_PAYLOAD_SHA256,
    }:
        _raise("INPUT_LINEAGE_MISMATCH")
    preserved = supplied.get("preserved_identities")
    if preserved != {
        "compatibility_row_count": 662,
        "mutation_applicability_cell_count": 13456,
        "normative_row_count": 841,
        "required_mutation_count": 6487,
        "review_surface_identity_sha256": REVIEW_SURFACE_IDENTITY_SHA256,
        "trust_model_sha256": IMPLEMENTATION_TRUST_MODEL_SHA256,
    }:
        _raise("INPUT_LINEAGE_MISMATCH")
    rows = supplied.get("artifact_path_surface")
    base_rows = {
        row["role"]: row for row in base_plan["artifact_path_surface"]
    }
    if (
        not isinstance(rows, list)
        or len(rows) != 18
        or rows != sorted(rows, key=lambda row: row.get("role", ""))
        or sha256_bytes(canonical_json_bytes(rows))
        != R2_ARTIFACT_PATH_SURFACE_SHA256
        or {row.get("role") for row in rows} != set(base_rows)
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    seen_paths: set[str] = set()
    old_paths = {
        value
        for row in base_rows.values()
        for value in (row["final_path"], row["pending_path"])
    }
    for row_value in rows:
        row = require_mapping(row_value, code="INPUT_LINEAGE_MISMATCH")
        require_exact_keys(
            row,
            {
                "role",
                "final_path",
                "pending_path",
                "publication_mode",
                "lifecycle_scope",
            },
            code="INPUT_LINEAGE_MISMATCH",
        )
        role = row["role"]
        final_path = require_text(
            row["final_path"], code="INPUT_LINEAGE_MISMATCH"
        )
        pending_path = require_text(
            row["pending_path"], code="INPUT_LINEAGE_MISMATCH"
        )
        if (
            final_path in seen_paths
            or pending_path in seen_paths
            or row["publication_mode"]
            != "MoveFileExW_nonreplace_write_through"
            or row["lifecycle_scope"] != base_rows[role]["lifecycle_scope"]
        ):
            _raise("INPUT_LINEAGE_MISMATCH")
        seen_paths.update((final_path, pending_path))
        if role == "formal_design_review_verdict":
            if row != base_rows[role]:
                _raise("INPUT_LINEAGE_MISMATCH")
        elif (
            R2_AUTHORITY_NAMESPACE_ID not in Path(final_path).name
            or pending_path
            != final_path + f".pending-{R2_AUTHORITY_NAMESPACE_ID}"
            or final_path in old_paths
            or pending_path in old_paths
        ):
            _raise("INPUT_LINEAGE_MISMATCH")
    extra_contract_names = (
        "candidate_manifest_contract",
        "clean_restore_receipt_contract",
        "fresh_review_evidence_contract",
    )
    for name in extra_contract_names:
        contract = require_mapping(
            supplied.get(name), code="INPUT_LINEAGE_MISMATCH"
        )
        final_path = require_text(
            contract.get("artifact_path"), code="INPUT_LINEAGE_MISMATCH"
        )
        pending_path = require_text(
            contract.get("pending_path"), code="INPUT_LINEAGE_MISMATCH"
        )
        if (
            final_path in seen_paths
            or pending_path in seen_paths
            or R2_AUTHORITY_NAMESPACE_ID not in Path(final_path).name
            or pending_path
            != final_path + f".pending-{R2_AUTHORITY_NAMESPACE_ID}"
        ):
            _raise("INPUT_LINEAGE_MISMATCH")
        seen_paths.update((final_path, pending_path))
    packet_path = require_text(
        supplied.get("fresh_review_packet_path"),
        code="INPUT_LINEAGE_MISMATCH",
    )
    if (
        packet_path in seen_paths
        or R2_AUTHORITY_NAMESPACE_ID not in Path(packet_path).name
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    binding_delta = require_mapping(
        supplied.get("implementation_binding_contract_overlay"),
        code="INPUT_LINEAGE_MISMATCH",
    )
    binding_required = require_mapping(
        binding_delta.get("required_values"),
        code="INPUT_LINEAGE_MISMATCH",
    )
    review_delta = require_mapping(
        supplied.get("fresh_review_contract_overlay"),
        code="INPUT_LINEAGE_MISMATCH",
    )
    review_pass = require_mapping(
        require_mapping(
            require_mapping(
                review_delta.get("outcomes"),
                code="INPUT_LINEAGE_MISMATCH",
            ).get("pass"),
            code="INPUT_LINEAGE_MISMATCH",
        ).get("required_values"),
        code="INPUT_LINEAGE_MISMATCH",
    )
    authority_delta = require_mapping(
        supplied.get("reviewed_authority_contract_overlay"),
        code="INPUT_LINEAGE_MISMATCH",
    )
    role_rows = {row["role"]: row for row in rows}
    if (
        binding_delta.get("artifact_path")
        != role_rows["implementation_candidate_binding"]["final_path"]
        or review_delta.get("artifact_path")
        != role_rows["fresh_implementation_review_verdict"]["final_path"]
        or authority_delta.get("artifact_path")
        != role_rows["reviewed_implementation_authority"]["final_path"]
        or authority_delta.get("fresh_implementation_review_path")
        != review_delta.get("artifact_path")
        or any(
            delta.get("artifact_path_surface_sha256")
            != R2_ARTIFACT_PATH_SURFACE_SHA256
            for delta in (binding_delta, authority_delta)
        )
        or binding_required.get(
            "procedural_author_separation_precondition_satisfied"
        )
        is not False
        or binding_required.get(
            "current_exposed_design_context_authored_final_bytes"
        )
        is not True
        or review_pass.get(
            "procedural_author_separation_precondition_satisfied"
        )
        is not False
        or review_pass.get(
            "current_exposed_design_context_authored_final_bytes"
        )
        is not True
        or "task_provenance"
        not in authority_delta.get(
            "implementation_author_separation_basis", ""
        )
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    policy = supplied.get("protected_surface_policy")
    if policy != {
        "phase_a_protected_root_reads_allowed": False,
        "phase_a_runtime_artifacts_allowed": False,
        "scientific_values_inspected": False,
    }:
        _raise("INPUT_LINEAGE_MISMATCH")
    if check_legacy_occupancy:
        _validate_r2_legacy_occupancy(base_plan, supplied)
    return supplied


def load_r2_activation_plan(
    path: Path = R2_ACTIVATION_PLAN_PATH,
    *,
    base_plan: Mapping[str, Any] | None = None,
    check_legacy_occupancy: bool = True,
) -> dict[str, Any]:
    raw = read_exact_bytes(
        path,
        R2_ACTIVATION_PLAN_FILE_SHA256,
        code="INPUT_LINEAGE_MISMATCH",
    )
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        _raise("INPUT_LINEAGE_MISMATCH")
    value = require_mapping(
        strict_json_loads(raw[:-1], canonical=True),
        code="INPUT_LINEAGE_MISMATCH",
    )
    if raw != canonical_receipt_bytes(value):
        _raise("INPUT_LINEAGE_MISMATCH")
    frozen = load_frozen_plan() if base_plan is None else dict(base_plan)
    return validate_r2_activation_plan(
        frozen,
        value,
        check_legacy_occupancy=check_legacy_occupancy,
    )


def build_r2_active_plan(
    base_plan: Mapping[str, Any],
    r2_plan: Mapping[str, Any],
) -> dict[str, Any]:
    frozen = validate_frozen_plan(base_plan)
    overlay = validate_r2_activation_plan(
        frozen,
        r2_plan,
        check_legacy_occupancy=False,
    )
    active = _replace_active_string(
        frozen,
        ARTIFACT_PATH_SURFACE_SHA256,
        R2_ARTIFACT_PATH_SURFACE_SHA256,
    )
    if not isinstance(active, dict):
        _raise("INPUT_LINEAGE_MISMATCH")
    active["artifact_path_surface"] = copy.deepcopy(
        overlay["artifact_path_surface"]
    )
    active["artifact_path_surface_sha256"] = (
        R2_ARTIFACT_PATH_SURFACE_SHA256
    )
    pending_by_role = {
        row["role"]: row["pending_path"]
        for row in overlay["artifact_path_surface"]
    }
    for test_row in active["artifact_lifecycle_contract"][
        "full_surface_checkpoint_contract"
    ]["pending_injection_tests"]:
        test_row["injected_pending_path"] = pending_by_role[test_row["role"]]
    _merge_control_delta(
        active["implementation_binding_contract"],
        overlay["implementation_binding_contract_overlay"],
    )
    _merge_control_delta(
        active["review_receipt_schemas"][
            "fresh_implementation_review_verdict"
        ],
        overlay["fresh_review_contract_overlay"],
    )
    _merge_control_delta(
        active["reviewed_implementation_authority_contract"],
        overlay["reviewed_authority_contract_overlay"],
    )
    active["r2_activation_control"] = {
        "authority_namespace_id": R2_AUTHORITY_NAMESPACE_ID,
        "activation_plan_path": str(R2_ACTIVATION_PLAN_PATH),
        "activation_plan_file_sha256": R2_ACTIVATION_PLAN_FILE_SHA256,
        "activation_plan_payload_sha256": (
            R2_ACTIVATION_PLAN_PAYLOAD_SHA256
        ),
        "occupied_v0_9_surface_sha256": (
            R2_OCCUPIED_V0_9_SURFACE_SHA256
        ),
        "activation_lineage": copy.deepcopy(
            overlay["activation_lineage"]
        ),
        "candidate_manifest_contract": copy.deepcopy(
            overlay["candidate_manifest_contract"]
        ),
        "clean_restore_receipt_contract": copy.deepcopy(
            overlay["clean_restore_receipt_contract"]
        ),
        "fresh_review_evidence_contract": copy.deepcopy(
            overlay["fresh_review_evidence_contract"]
        ),
        "fresh_review_packet_path": overlay["fresh_review_packet_path"],
    }
    if (
        artifact_surface_sha256(active)
        != R2_ARTIFACT_PATH_SURFACE_SHA256
        or recompute_implementation_trust_model_sha256(active)
        != IMPLEMENTATION_TRUST_MODEL_SHA256
        or review_surface_identity(active)[
            "review_surface_identity_sha256"
        ]
        != REVIEW_SURFACE_IDENTITY_SHA256
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    nested = _walk_named_values(active, "artifact_path_surface_sha256")
    if not nested or any(
        value != R2_ARTIFACT_PATH_SURFACE_SHA256 for value in nested
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    return active


def load_r2_active_plan(
    *, repository_root: Path | None = None
) -> dict[str, Any]:
    root = explicit_repository_root(repository_root)
    materialized_plan = root.joinpath(
        *PurePosixPath(R2_ACTIVATION_PLAN_RELATIVE_PATH).parts
    )
    base_plan = load_frozen_plan()
    r2_plan = load_r2_activation_plan(
        path=materialized_plan,
        base_plan=base_plan,
        check_legacy_occupancy=True,
    )
    return build_r2_active_plan(base_plan, r2_plan)


def _validate_r2r1_occupied_r2(
    r2_active_plan: Mapping[str, Any],
    remediation_plan: Mapping[str, Any],
) -> None:
    occupied = require_mapping(
        remediation_plan.get("occupied_r2"),
        code="INPUT_LINEAGE_MISMATCH",
    )
    if sha256_bytes(canonical_json_bytes(occupied)) != (
        R2R1_OCCUPIED_R2_SURFACE_SHA256
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    receipts = (
        (
            "clean_restore_receipt",
            "restore_receipt_payload_sha256",
        ),
        (
            "candidate_manifest",
            "candidate_manifest_payload_sha256",
        ),
        (
            "candidate_binding",
            "implementation_candidate_binding_payload_sha256",
        ),
    )
    for name, hash_field in receipts:
        row = require_mapping(
            occupied.get(name), code="INPUT_LINEAGE_MISMATCH"
        )
        receipt = read_canonical_receipt(
            Path(row["path"]),
            expected_file_sha256=row["file_sha256"],
            hash_field=hash_field,
            expected_payload_sha256=row["payload_sha256"],
        )
        if name == "candidate_binding" and receipt.get(
            "source_commit"
        ) != occupied.get("candidate_commit"):
            _raise("INPUT_LINEAGE_MISMATCH")
    packet = require_mapping(
        occupied.get("review_packet"), code="INPUT_LINEAGE_MISMATCH"
    )
    packet_raw = read_exact_bytes(
        Path(packet["path"]),
        packet["file_sha256"],
        code="INPUT_LINEAGE_MISMATCH",
    )
    if not packet_raw.endswith(b"\n") or packet_raw.endswith(b"\n\n"):
        _raise("INPUT_LINEAGE_MISMATCH")
    rows = {
        row["role"]: row for row in r2_active_plan["artifact_path_surface"]
    }
    for role in occupied["required_absent_final_roles"]:
        if role not in rows or Path(rows[role]["final_path"]).exists():
            _raise("UNEXPECTED_ARTIFACT")
    for role in occupied["required_absent_pending_roles"]:
        if role not in rows or Path(rows[role]["pending_path"]).exists():
            _raise("UNEXPECTED_ARTIFACT")
    for path_text in occupied["required_absent_control_paths"]:
        if Path(path_text).exists():
            _raise("UNEXPECTED_ARTIFACT")


def validate_r2r1_remediation_plan(
    r2_active_plan: Mapping[str, Any],
    remediation_plan: Mapping[str, Any],
    *,
    repository_root: Path | None = None,
    check_r2_occupancy: bool = True,
) -> dict[str, Any]:
    supplied = dict(remediation_plan)
    require_exact_keys(
        supplied,
        {
            "allowed_changed_paths",
            "artifact_path_surface",
            "artifact_path_surface_sha256",
            "candidate_manifest_contract",
            "candidate_selection_contract",
            "clean_restore_receipt_contract",
            "fresh_review_contract_overlay",
            "fresh_review_evidence_contract",
            "fresh_review_packet_path",
            "implementation_binding_contract_overlay",
            "namespace_id",
            "occupied_r2",
            "occupied_r2_surface_sha256",
            "parent_lineage",
            "preserved_identities",
            "protected_surface_policy",
            "publication_policy",
            "purpose",
            "r2_activation_plan",
            "r2r1_remediation_plan_payload_sha256",
            "remediation_plan_relative_path",
            "review_coverage_identity",
            "review_input_freeze_contract",
            "reviewed_authority_contract_overlay",
            "schema_version",
            "state",
        },
        code="INPUT_LINEAGE_MISMATCH",
    )
    verify_self_hash(
        supplied,
        "r2r1_remediation_plan_payload_sha256",
        code="INPUT_LINEAGE_MISMATCH",
    )
    if (
        supplied["r2r1_remediation_plan_payload_sha256"]
        != R2R1_REMEDIATION_PLAN_PAYLOAD_SHA256
        or supplied.get("schema_version")
        != "gate12c2_original_baseline_r2r1_remediation_plan_v0.1"
        or supplied.get("namespace_id")
        != R2R1_AUTHORITY_NAMESPACE_ID
        or supplied.get("state") != "R2R1_CONTROL_LINEAGE_FROZEN"
        or supplied.get("remediation_plan_relative_path")
        != "tools/gate12c2_original_baseline_r2r1_remediation_plan.json"
        or supplied.get("artifact_path_surface_sha256")
        != R2R1_ARTIFACT_PATH_SURFACE_SHA256
        or supplied.get("occupied_r2_surface_sha256")
        != R2R1_OCCUPIED_R2_SURFACE_SHA256
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    if supplied.get("r2_activation_plan") != {
        "file_sha256": R2_ACTIVATION_PLAN_FILE_SHA256,
        "path": str(R2_ACTIVATION_PLAN_HISTORICAL_DECLARED_PATH),
        "payload_sha256": R2_ACTIVATION_PLAN_PAYLOAD_SHA256,
    } or supplied.get("parent_lineage") != {
        "remediation_parent": R2R1_PARENT_COMMIT,
        "remediation_parent_count": 1,
        "remediation_grandparent": R2R1_GRANDPARENT_COMMIT,
        "remediation_grandparent_count": 1,
    }:
        _raise("INPUT_LINEAGE_MISMATCH")
    validate_repository_local_artifact(
        repository_root,
        historical_declared_path=supplied["r2_activation_plan"]["path"],
        expected_historical_declared_path=(
            R2_ACTIVATION_PLAN_HISTORICAL_DECLARED_PATH
        ),
        canonical_repository_relative_path=(
            R2_ACTIVATION_PLAN_RELATIVE_PATH
        ),
        expected_file_sha256=R2_ACTIVATION_PLAN_FILE_SHA256,
        bound_commit=R2R2_BASE_COMMIT,
        expected_git_blob_oid=R2_ACTIVATION_PLAN_BASE_BLOB_OID,
    )
    if supplied.get("preserved_identities") != {
        "compatibility_row_count": 662,
        "mutation_applicability_cell_count": 13456,
        "normative_row_count": 841,
        "required_mutation_count": 6487,
        "review_surface_identity_sha256": REVIEW_SURFACE_IDENTITY_SHA256,
        "trust_model_sha256": IMPLEMENTATION_TRUST_MODEL_SHA256,
    }:
        _raise("INPUT_LINEAGE_MISMATCH")
    coverage = require_mapping(
        supplied.get("review_coverage_identity"),
        code="INPUT_LINEAGE_MISMATCH",
    )
    require_exact_keys(
        coverage,
        {
            "changed_file_manifest_domain",
            "full_suite_test_count",
            "full_suite_test_node_id_sha256",
            "node_id_domain",
            "targeted_test_count",
            "targeted_test_node_id_sha256",
        },
        code="INPUT_LINEAGE_MISMATCH",
    )
    for count_field in ("targeted_test_count", "full_suite_test_count"):
        require_int(
            coverage.get(count_field),
            minimum=2,
            code="INPUT_LINEAGE_MISMATCH",
        )
    for digest_field in (
        "targeted_test_node_id_sha256",
        "full_suite_test_node_id_sha256",
    ):
        digest = coverage.get(digest_field)
        if not is_sha256(digest) or digest == "0" * 64:
            _raise("INPUT_LINEAGE_MISMATCH")
    allowed = supplied.get("allowed_changed_paths")
    if (
        not isinstance(allowed, list)
        or not allowed
        or allowed != sorted(allowed)
        or len(allowed) != len(set(allowed))
        or any(
            validate_relative_manifest_path(path) != path
            for path in allowed
        )
        or supplied["remediation_plan_relative_path"] not in allowed
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    rows = supplied.get("artifact_path_surface")
    old_rows = {
        row["role"]: row for row in r2_active_plan["artifact_path_surface"]
    }
    if (
        not isinstance(rows, list)
        or len(rows) != 18
        or rows != sorted(rows, key=lambda row: row.get("role", ""))
        or {row.get("role") for row in rows} != set(old_rows)
        or sha256_bytes(canonical_json_bytes(rows))
        != R2R1_ARTIFACT_PATH_SURFACE_SHA256
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    old_paths = {
        value
        for row in old_rows.values()
        for value in (row["final_path"], row["pending_path"])
    }
    seen: set[str] = set()
    role_rows: dict[str, Mapping[str, Any]] = {}
    for row_value in rows:
        row = require_mapping(row_value, code="INPUT_LINEAGE_MISMATCH")
        require_exact_keys(
            row,
            {
                "role",
                "final_path",
                "pending_path",
                "publication_mode",
                "lifecycle_scope",
            },
            code="INPUT_LINEAGE_MISMATCH",
        )
        role = row["role"]
        role_rows[role] = row
        final_path = require_text(
            row["final_path"], code="INPUT_LINEAGE_MISMATCH"
        )
        pending_path = require_text(
            row["pending_path"], code="INPUT_LINEAGE_MISMATCH"
        )
        if (
            final_path in seen
            or pending_path in seen
            or row["publication_mode"]
            != "MoveFileExW_nonreplace_write_through"
            or row["lifecycle_scope"] != old_rows[role]["lifecycle_scope"]
        ):
            _raise("INPUT_LINEAGE_MISMATCH")
        seen.update((final_path, pending_path))
        if role == "formal_design_review_verdict":
            if row != old_rows[role]:
                _raise("INPUT_LINEAGE_MISMATCH")
        elif (
            R2R1_AUTHORITY_NAMESPACE_ID not in Path(final_path).name
            or pending_path
            != final_path + ".pending-" + R2R1_AUTHORITY_NAMESPACE_ID
            or final_path in old_paths
            or pending_path in old_paths
        ):
            _raise("INPUT_LINEAGE_MISMATCH")
    contract_names = (
        "candidate_manifest_contract",
        "candidate_selection_contract",
        "clean_restore_receipt_contract",
        "fresh_review_evidence_contract",
        "review_input_freeze_contract",
    )
    for name in contract_names:
        contract = require_mapping(
            supplied.get(name), code="INPUT_LINEAGE_MISMATCH"
        )
        final_path = require_text(
            contract.get("artifact_path"), code="INPUT_LINEAGE_MISMATCH"
        )
        pending_path = require_text(
            contract.get("pending_path"), code="INPUT_LINEAGE_MISMATCH"
        )
        if (
            final_path in seen
            or pending_path in seen
            or R2R1_AUTHORITY_NAMESPACE_ID not in Path(final_path).name
            or pending_path
            != final_path + ".pending-" + R2R1_AUTHORITY_NAMESPACE_ID
        ):
            _raise("INPUT_LINEAGE_MISMATCH")
        seen.update((final_path, pending_path))
    packet_path = require_text(
        supplied.get("fresh_review_packet_path"),
        code="INPUT_LINEAGE_MISMATCH",
    )
    if (
        packet_path in seen
        or R2R1_AUTHORITY_NAMESPACE_ID not in Path(packet_path).name
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    binding = require_mapping(
        supplied.get("implementation_binding_contract_overlay"),
        code="INPUT_LINEAGE_MISMATCH",
    )
    review = require_mapping(
        supplied.get("fresh_review_contract_overlay"),
        code="INPUT_LINEAGE_MISMATCH",
    )
    authority = require_mapping(
        supplied.get("reviewed_authority_contract_overlay"),
        code="INPUT_LINEAGE_MISMATCH",
    )
    if (
        binding.get("artifact_path")
        != role_rows["implementation_candidate_binding"]["final_path"]
        or review.get("artifact_path")
        != role_rows["fresh_implementation_review_verdict"]["final_path"]
        or authority.get("artifact_path")
        != role_rows["reviewed_implementation_authority"]["final_path"]
        or authority.get("fresh_implementation_review_path")
        != review.get("artifact_path")
        or any(
            value.get("artifact_path_surface_sha256")
            != R2R1_ARTIFACT_PATH_SURFACE_SHA256
            for value in (binding, authority)
        )
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    if supplied.get("protected_surface_policy") != {
        "phase_a_protected_root_reads_allowed": False,
        "phase_a_runtime_artifacts_allowed": False,
        "scientific_values_inspected": False,
    }:
        _raise("INPUT_LINEAGE_MISMATCH")
    if check_r2_occupancy:
        _validate_r2r1_occupied_r2(r2_active_plan, supplied)
    return supplied


def load_r2r1_remediation_plan(
    *,
    repository_root: Path | None = None,
    r2_active_plan: Mapping[str, Any] | None = None,
    check_r2_occupancy: bool = True,
) -> dict[str, Any]:
    root = explicit_repository_root(repository_root)
    path = root.joinpath(
        *PurePosixPath(R2R1_REMEDIATION_PLAN_RELATIVE_PATH).parts
    )
    raw = read_exact_bytes(
        path,
        R2R1_REMEDIATION_PLAN_FILE_SHA256,
        code="INPUT_LINEAGE_MISMATCH",
    )
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        _raise("INPUT_LINEAGE_MISMATCH")
    value = require_mapping(
        strict_json_loads(raw[:-1], canonical=True),
        code="INPUT_LINEAGE_MISMATCH",
    )
    if raw != canonical_receipt_bytes(value):
        _raise("INPUT_LINEAGE_MISMATCH")
    materialized = validate_repository_local_artifact(
        root,
        historical_declared_path=str(
            R2R1_REMEDIATION_PLAN_HISTORICAL_DECLARED_PATH
        ),
        expected_historical_declared_path=(
            R2R1_REMEDIATION_PLAN_HISTORICAL_DECLARED_PATH
        ),
        canonical_repository_relative_path=(
            R2R1_REMEDIATION_PLAN_RELATIVE_PATH
        ),
        expected_file_sha256=R2R1_REMEDIATION_PLAN_FILE_SHA256,
        bound_commit=R2R2_BASE_COMMIT,
        expected_git_blob_oid=R2R1_REMEDIATION_PLAN_BASE_BLOB_OID,
    )
    if materialized != raw:
        _raise("INPUT_LINEAGE_MISMATCH")
    active_r2 = (
        load_r2_active_plan(repository_root=root)
        if r2_active_plan is None
        else dict(r2_active_plan)
    )
    return validate_r2r1_remediation_plan(
        active_r2,
        value,
        repository_root=repository_root,
        check_r2_occupancy=check_r2_occupancy,
    )


def build_r2r1_active_plan(
    r2_active_plan: Mapping[str, Any],
    remediation_plan: Mapping[str, Any],
    *,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    overlay = validate_r2r1_remediation_plan(
        r2_active_plan,
        remediation_plan,
        repository_root=repository_root,
        check_r2_occupancy=False,
    )
    active = _replace_active_string(
        r2_active_plan,
        R2_ARTIFACT_PATH_SURFACE_SHA256,
        R2R1_ARTIFACT_PATH_SURFACE_SHA256,
    )
    if not isinstance(active, dict):
        _raise("INPUT_LINEAGE_MISMATCH")
    active["artifact_path_surface"] = copy.deepcopy(
        overlay["artifact_path_surface"]
    )
    active["artifact_path_surface_sha256"] = (
        R2R1_ARTIFACT_PATH_SURFACE_SHA256
    )
    pending_by_role = {
        row["role"]: row["pending_path"]
        for row in overlay["artifact_path_surface"]
    }
    for test_row in active["artifact_lifecycle_contract"][
        "full_surface_checkpoint_contract"
    ]["pending_injection_tests"]:
        test_row["injected_pending_path"] = pending_by_role[test_row["role"]]
    _merge_control_delta(
        active["implementation_binding_contract"],
        overlay["implementation_binding_contract_overlay"],
    )
    _merge_control_delta(
        active["review_receipt_schemas"][
            "fresh_implementation_review_verdict"
        ],
        overlay["fresh_review_contract_overlay"],
    )
    _merge_control_delta(
        active["reviewed_implementation_authority_contract"],
        overlay["reviewed_authority_contract_overlay"],
    )
    active["r2r1_remediation_control"] = {
        "authority_namespace_id": R2R1_AUTHORITY_NAMESPACE_ID,
        "remediation_plan_path": R2R1_REMEDIATION_PLAN_RELATIVE_PATH,
        "remediation_plan_file_sha256": (
            R2R1_REMEDIATION_PLAN_FILE_SHA256
        ),
        "remediation_plan_payload_sha256": (
            R2R1_REMEDIATION_PLAN_PAYLOAD_SHA256
        ),
        "occupied_r2_surface_sha256": R2R1_OCCUPIED_R2_SURFACE_SHA256,
        "parent_lineage": copy.deepcopy(overlay["parent_lineage"]),
        "allowed_changed_paths": copy.deepcopy(
            overlay["allowed_changed_paths"]
        ),
        "review_coverage_identity": copy.deepcopy(
            overlay["review_coverage_identity"]
        ),
        "candidate_selection_contract": copy.deepcopy(
            overlay["candidate_selection_contract"]
        ),
        "candidate_manifest_contract": copy.deepcopy(
            overlay["candidate_manifest_contract"]
        ),
        "clean_restore_receipt_contract": copy.deepcopy(
            overlay["clean_restore_receipt_contract"]
        ),
        "review_input_freeze_contract": copy.deepcopy(
            overlay["review_input_freeze_contract"]
        ),
        "fresh_review_evidence_contract": copy.deepcopy(
            overlay["fresh_review_evidence_contract"]
        ),
        "fresh_review_packet_path": overlay["fresh_review_packet_path"],
    }
    if (
        artifact_surface_sha256(active)
        != R2R1_ARTIFACT_PATH_SURFACE_SHA256
        or recompute_implementation_trust_model_sha256(active)
        != IMPLEMENTATION_TRUST_MODEL_SHA256
        or review_surface_identity(active)[
            "review_surface_identity_sha256"
        ]
        != REVIEW_SURFACE_IDENTITY_SHA256
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    nested = _walk_named_values(active, "artifact_path_surface_sha256")
    if not nested or any(
        value != R2R1_ARTIFACT_PATH_SURFACE_SHA256 for value in nested
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    return active


def _validate_r2r2_occupied_r2r1(
    r2r1_active_plan: Mapping[str, Any],
    portability_plan: Mapping[str, Any],
) -> None:
    occupied = require_mapping(
        portability_plan.get("occupied_r2r1"),
        code="INPUT_LINEAGE_MISMATCH",
    )
    if (
        sha256_bytes(canonical_json_bytes(occupied))
        != R2R2_OCCUPIED_R2R1_SURFACE_SHA256
        or occupied.get("candidate_commit")
        != R2R2_HISTORICAL_CANDIDATE_COMMIT
        or occupied.get("stage1_failure_codes")
        != [
            "CLEAN_RESTORE_INPUT_LINEAGE_MISMATCH",
            "INDEPENDENT_FRAMING_INPUT_LINEAGE_MISMATCH",
        ]
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    receipt_specs = {
        "candidate_selection": "candidate_selection_payload_sha256",
        "candidate_manifest": "candidate_manifest_payload_sha256",
        "candidate_binding": (
            "implementation_candidate_binding_payload_sha256"
        ),
        "clean_restore_receipt": "restore_receipt_payload_sha256",
        "review_input_freeze": "review_input_freeze_payload_sha256",
        "review_evidence": "review_evidence_payload_sha256",
        "review_verdict": "fresh_implementation_review_payload_sha256",
        "reviewed_authority": (
            "reviewed_implementation_authority_payload_sha256"
        ),
    }
    for name, hash_field in receipt_specs.items():
        row = require_mapping(
            occupied.get(name), code="INPUT_LINEAGE_MISMATCH"
        )
        allowed = {"path", "file_sha256", "payload_sha256"}
        if name == "reviewed_authority":
            allowed.add("execution_status")
            if row.get("execution_status") != (
                "HISTORICAL_NON_EXECUTABLE_AFTER_FAILED_MANDATORY_STAGE1"
            ):
                _raise("INPUT_LINEAGE_MISMATCH")
        require_exact_keys(row, allowed, code="INPUT_LINEAGE_MISMATCH")
        receipt = read_canonical_receipt(
            Path(row["path"]),
            expected_file_sha256=row["file_sha256"],
            hash_field=hash_field,
            expected_payload_sha256=row["payload_sha256"],
        )
        if (
            name == "candidate_binding"
            and receipt.get("source_commit")
            != R2R2_HISTORICAL_CANDIDATE_COMMIT
        ):
            _raise("INPUT_LINEAGE_MISMATCH")
    packet = require_mapping(
        occupied.get("review_packet"), code="INPUT_LINEAGE_MISMATCH"
    )
    require_exact_keys(
        packet,
        {"path", "file_sha256", "size_bytes"},
        code="INPUT_LINEAGE_MISMATCH",
    )
    packet_raw = read_exact_bytes(
        Path(packet["path"]),
        packet["file_sha256"],
        code="INPUT_LINEAGE_MISMATCH",
    )
    if (
        len(packet_raw) != packet["size_bytes"]
        or not packet_raw.endswith(b"\n")
        or packet_raw.endswith(b"\n\n")
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    roles = {
        row["role"]: row
        for row in r2r1_active_plan["artifact_path_surface"]
    }
    for role in occupied["required_absent_final_roles"]:
        if role not in roles or Path(roles[role]["final_path"]).exists():
            _raise("UNEXPECTED_ARTIFACT")
    for role in occupied["required_absent_pending_roles"]:
        if role not in roles or Path(roles[role]["pending_path"]).exists():
            _raise("UNEXPECTED_ARTIFACT")


def validate_r2r2_portability_plan(
    r2r1_active_plan: Mapping[str, Any],
    portability_plan: Mapping[str, Any],
    *,
    repository_root: Path | None = None,
    check_r2r1_occupancy: bool = True,
) -> dict[str, Any]:
    supplied = dict(portability_plan)
    require_exact_keys(
        supplied,
        {
            "allowed_changed_paths",
            "artifact_path_surface",
            "artifact_path_surface_sha256",
            "candidate_manifest_contract",
            "candidate_selection_contract",
            "clean_restore_receipt_contract",
            "fresh_review_contract_overlay",
            "fresh_review_evidence_contract",
            "fresh_review_packet_path",
            "implementation_binding_contract_overlay",
            "namespace_id",
            "occupied_r2r1",
            "occupied_r2r1_surface_sha256",
            "parent_lineage",
            "preserved_identities",
            "protected_surface_policy",
            "publication_policy",
            "purpose",
            "r2r2_portability_plan_payload_sha256",
            "remediation_plan_relative_path",
            "repository_local_artifact_surface_sha256",
            "repository_local_artifacts",
            "repository_path_policy",
            "review_coverage_identity",
            "review_input_freeze_contract",
            "reviewed_authority_contract_overlay",
            "schema_version",
            "state",
            "upstream_json_framing",
            "upstream_json_framing_surface_sha256",
        },
        code="INPUT_LINEAGE_MISMATCH",
    )
    verify_self_hash(
        supplied,
        "r2r2_portability_plan_payload_sha256",
        code="INPUT_LINEAGE_MISMATCH",
    )
    if (
        supplied["r2r2_portability_plan_payload_sha256"]
        != R2R2_PORTABILITY_PLAN_PAYLOAD_SHA256
        or supplied.get("schema_version")
        != "gate12c2_original_baseline_r2r2_portability_plan_v0.2"
        or supplied.get("namespace_id") != R2R2_AUTHORITY_NAMESPACE_ID
        or supplied.get("state")
        != "R2R3_ROOT_PORTABILITY_AND_NAMESPACE_FROZEN"
        or supplied.get("remediation_plan_relative_path")
        != R2R2_PORTABILITY_PLAN_RELATIVE_PATH
        or supplied.get("artifact_path_surface_sha256")
        != R2R2_ARTIFACT_PATH_SURFACE_SHA256
        or supplied.get("occupied_r2r1_surface_sha256")
        != R2R2_OCCUPIED_R2R1_SURFACE_SHA256
        or supplied.get("repository_local_artifact_surface_sha256")
        != R2R2_REPOSITORY_LOCAL_SURFACE_SHA256
        or supplied.get("upstream_json_framing_surface_sha256")
        != R2R2_UPSTREAM_FRAMING_SURFACE_SHA256
        or supplied.get("parent_lineage")
        != {
            "remediation_parent": R2R2_BASE_COMMIT,
            "remediation_parent_count": 1,
            "remediation_grandparent": R2R2_HISTORICAL_CANDIDATE_COMMIT,
            "remediation_grandparent_count": 1,
        }
        or supplied.get("preserved_identities")
        != {
            "compatibility_row_count": 662,
            "mutation_applicability_cell_count": 13456,
            "normative_row_count": 841,
            "required_mutation_count": 6487,
            "review_surface_identity_sha256": (
                REVIEW_SURFACE_IDENTITY_SHA256
            ),
            "trust_model_sha256": IMPLEMENTATION_TRUST_MODEL_SHA256,
        }
        or supplied.get("repository_path_policy")
        != {
            "absolute_override_allowed": False,
            "canonical_repository_relative_path_in_identity": True,
            "current_materialized_path_in_identity": False,
            "git_blob_verification_required": True,
            "historical_declared_path_in_identity": True,
            "parent_traversal_allowed": False,
            "reparse_escape_allowed": False,
        }
        or supplied.get("protected_surface_policy")
        != {
            "phase_a_protected_root_reads_allowed": False,
            "phase_a_runtime_artifacts_allowed": False,
            "scientific_values_inspected": False,
        }
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    publication = supplied.get("publication_policy")
    if publication != {
        "active_role_count": 18,
        "active_role_publication_mode": (
            "MoveFileExW_nonreplace_write_through"
        ),
        "final_pending_collisions_allowed": False,
        "legacy_v0_9_write_allowed": False,
        "mixed_v0_9_r2_r2r1_lineage_allowed": False,
        "pending_suffix": ".pending-" + R2R2_AUTHORITY_NAMESPACE_ID,
        "publication_is_atomic_nonreplace": True,
        "r2_write_allowed": False,
        "unknown_or_duplicate_roles_allowed": False,
    }:
        _raise("INPUT_LINEAGE_MISMATCH")
    allowed = supplied.get("allowed_changed_paths")
    if (
        not isinstance(allowed, list)
        or allowed != sorted(allowed)
        or len(allowed) != 4
        or len(allowed) != len(set(allowed))
        or any(validate_relative_manifest_path(path) != path for path in allowed)
        or R2R2_PORTABILITY_PLAN_RELATIVE_PATH not in allowed
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    coverage = require_mapping(
        supplied.get("review_coverage_identity"),
        code="INPUT_LINEAGE_MISMATCH",
    )
    require_exact_keys(
        coverage,
        {
            "changed_file_manifest_domain",
            "full_suite_test_count",
            "full_suite_test_node_id_sha256",
            "node_id_domain",
            "targeted_test_count",
            "targeted_test_node_id_sha256",
        },
        code="INPUT_LINEAGE_MISMATCH",
    )
    for field in ("targeted_test_count", "full_suite_test_count"):
        require_int(
            coverage.get(field),
            minimum=2,
            code="INPUT_LINEAGE_MISMATCH",
        )
    for field in (
        "targeted_test_node_id_sha256",
        "full_suite_test_node_id_sha256",
    ):
        if not is_sha256(coverage.get(field)) or coverage.get(field) == "0" * 64:
            _raise("INPUT_LINEAGE_MISMATCH")
    repository_rows = supplied.get("repository_local_artifacts")
    if (
        not isinstance(repository_rows, list)
        or len(repository_rows) != 3
        or repository_rows
        != sorted(repository_rows, key=lambda row: row.get("role", ""))
        or sha256_bytes(canonical_json_bytes(repository_rows))
        != R2R2_REPOSITORY_LOCAL_SURFACE_SHA256
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    by_role = {
        require_text(row.get("role"), code="INPUT_LINEAGE_MISMATCH"): row
        for row in repository_rows
    }
    expected_local = {
        "r2_activation_plan": (
            R2_ACTIVATION_PLAN_HISTORICAL_DECLARED_PATH,
            R2_ACTIVATION_PLAN_RELATIVE_PATH,
            R2_ACTIVATION_PLAN_FILE_SHA256,
            R2_ACTIVATION_PLAN_PAYLOAD_SHA256,
            R2_ACTIVATION_PLAN_BASE_BLOB_OID,
        ),
        "r2r1_remediation_plan": (
            R2R1_REMEDIATION_PLAN_HISTORICAL_DECLARED_PATH,
            R2R1_REMEDIATION_PLAN_RELATIVE_PATH,
            R2R1_REMEDIATION_PLAN_FILE_SHA256,
            R2R1_REMEDIATION_PLAN_PAYLOAD_SHA256,
            R2R1_REMEDIATION_PLAN_BASE_BLOB_OID,
        ),
    }
    for role, values in expected_local.items():
        historical, relative, file_hash, payload_hash, blob = values
        row = require_mapping(by_role.get(role), code="INPUT_LINEAGE_MISMATCH")
        if row != {
            "bound_commit": R2R2_BASE_COMMIT,
            "canonical_repository_relative_path": relative,
            "file_sha256": file_hash,
            "git_blob_oid": blob,
            "historical_declared_path": str(historical),
            "identity_source": "frozen_parent_commit",
            "payload_sha256": payload_hash,
            "role": role,
        }:
            _raise("INPUT_LINEAGE_MISMATCH")
        validate_repository_local_artifact(
            repository_root,
            historical_declared_path=row["historical_declared_path"],
            expected_historical_declared_path=historical,
            canonical_repository_relative_path=(
                row["canonical_repository_relative_path"]
            ),
            expected_file_sha256=file_hash,
            bound_commit=R2R2_BASE_COMMIT,
            expected_git_blob_oid=blob,
        )
    if by_role.get("r2r2_portability_plan") != {
        "bound_commit": None,
        "canonical_repository_relative_path": (
            R2R2_PORTABILITY_PLAN_RELATIVE_PATH
        ),
        "file_sha256": None,
        "git_blob_oid": None,
        "historical_declared_path": str(
            R2R2_PORTABILITY_PLAN_HISTORICAL_DECLARED_PATH
        ),
        "identity_source": (
            "compiled_constants_and_candidate_selection_exact_commit"
        ),
        "payload_sha256": None,
        "role": "r2r2_portability_plan",
    }:
        _raise("INPUT_LINEAGE_MISMATCH")
    framing = supplied.get("upstream_json_framing")
    if (
        framing
        != r2r1_active_plan["upstream_authority"]["artifact_rows"]
        or sha256_bytes(canonical_json_bytes(framing))
        != R2R2_UPSTREAM_FRAMING_SURFACE_SHA256
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    rows = supplied.get("artifact_path_surface")
    old_rows = {
        row["role"]: row
        for row in r2r1_active_plan["artifact_path_surface"]
    }
    if (
        not isinstance(rows, list)
        or len(rows) != 18
        or rows != sorted(rows, key=lambda row: row.get("role", ""))
        or {row.get("role") for row in rows} != set(old_rows)
        or sha256_bytes(canonical_json_bytes(rows))
        != R2R2_ARTIFACT_PATH_SURFACE_SHA256
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    old_paths = {
        value
        for row in old_rows.values()
        for value in (row["final_path"], row["pending_path"])
    }
    seen: set[str] = set()
    role_rows: dict[str, Mapping[str, Any]] = {}
    for row_value in rows:
        row = require_mapping(row_value, code="INPUT_LINEAGE_MISMATCH")
        require_exact_keys(
            row,
            {
                "role",
                "final_path",
                "pending_path",
                "publication_mode",
                "lifecycle_scope",
            },
            code="INPUT_LINEAGE_MISMATCH",
        )
        role = row["role"]
        role_rows[role] = row
        final_path = require_text(
            row["final_path"], code="INPUT_LINEAGE_MISMATCH"
        )
        pending_path = require_text(
            row["pending_path"], code="INPUT_LINEAGE_MISMATCH"
        )
        if (
            final_path in seen
            or pending_path in seen
            or row["publication_mode"]
            != "MoveFileExW_nonreplace_write_through"
            or row["lifecycle_scope"] != old_rows[role]["lifecycle_scope"]
        ):
            _raise("INPUT_LINEAGE_MISMATCH")
        seen.update((final_path, pending_path))
        if (
            R2R2_AUTHORITY_NAMESPACE_ID not in Path(final_path).name
            or pending_path
            != final_path + ".pending-" + R2R2_AUTHORITY_NAMESPACE_ID
            or final_path in old_paths
            or pending_path in old_paths
        ):
            _raise("INPUT_LINEAGE_MISMATCH")
    for name in (
        "candidate_manifest_contract",
        "candidate_selection_contract",
        "clean_restore_receipt_contract",
        "fresh_review_evidence_contract",
        "review_input_freeze_contract",
    ):
        contract = require_mapping(
            supplied.get(name), code="INPUT_LINEAGE_MISMATCH"
        )
        final_path = require_text(
            contract.get("artifact_path"), code="INPUT_LINEAGE_MISMATCH"
        )
        pending_path = require_text(
            contract.get("pending_path"), code="INPUT_LINEAGE_MISMATCH"
        )
        if (
            final_path in seen
            or pending_path in seen
            or R2R2_AUTHORITY_NAMESPACE_ID not in Path(final_path).name
            or pending_path
            != final_path + ".pending-" + R2R2_AUTHORITY_NAMESPACE_ID
        ):
            _raise("INPUT_LINEAGE_MISMATCH")
        seen.update((final_path, pending_path))
    packet_path = require_text(
        supplied.get("fresh_review_packet_path"),
        code="INPUT_LINEAGE_MISMATCH",
    )
    if (
        packet_path in seen
        or R2R2_AUTHORITY_NAMESPACE_ID not in Path(packet_path).name
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    binding = require_mapping(
        supplied.get("implementation_binding_contract_overlay"),
        code="INPUT_LINEAGE_MISMATCH",
    )
    review = require_mapping(
        supplied.get("fresh_review_contract_overlay"),
        code="INPUT_LINEAGE_MISMATCH",
    )
    authority = require_mapping(
        supplied.get("reviewed_authority_contract_overlay"),
        code="INPUT_LINEAGE_MISMATCH",
    )
    if (
        binding.get("artifact_path")
        != role_rows["implementation_candidate_binding"]["final_path"]
        or review.get("artifact_path")
        != role_rows["fresh_implementation_review_verdict"]["final_path"]
        or authority.get("artifact_path")
        != role_rows["reviewed_implementation_authority"]["final_path"]
        or authority.get("fresh_implementation_review_path")
        != review.get("artifact_path")
        or any(
            value.get("artifact_path_surface_sha256")
            != R2R2_ARTIFACT_PATH_SURFACE_SHA256
            for value in (binding, authority)
        )
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    if check_r2r1_occupancy:
        _validate_r2r2_occupied_r2r1(r2r1_active_plan, supplied)
    return supplied


def load_r2r2_portability_plan(
    *,
    repository_root: Path | None = None,
    r2r1_active_plan: Mapping[str, Any] | None = None,
    check_r2r1_occupancy: bool = True,
) -> dict[str, Any]:
    root = explicit_repository_root(repository_root)
    path = root.joinpath(
        *PurePosixPath(R2R2_PORTABILITY_PLAN_RELATIVE_PATH).parts
    )
    raw = read_exact_bytes(
        path,
        R2R2_PORTABILITY_PLAN_FILE_SHA256,
        code="INPUT_LINEAGE_MISMATCH",
    )
    if (
        not raw.endswith(b"\n")
        or raw.endswith((b"\r\n", b"\n\n"))
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    value = require_mapping(
        strict_json_loads(raw[:-1], canonical=True),
        code="INPUT_LINEAGE_MISMATCH",
    )
    if raw != canonical_receipt_bytes(value):
        _raise("INPUT_LINEAGE_MISMATCH")
    active = (
        build_r2r1_active_plan(
            load_r2_active_plan(repository_root=root),
            load_r2r1_remediation_plan(
                repository_root=root,
                check_r2_occupancy=check_r2r1_occupancy,
            ),
            repository_root=root,
        )
        if r2r1_active_plan is None
        else dict(r2r1_active_plan)
    )
    return validate_r2r2_portability_plan(
        active,
        value,
        repository_root=root,
        check_r2r1_occupancy=check_r2r1_occupancy,
    )


def build_r2r2_active_plan(
    r2r1_active_plan: Mapping[str, Any],
    portability_plan: Mapping[str, Any],
    *,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    overlay = validate_r2r2_portability_plan(
        r2r1_active_plan,
        portability_plan,
        repository_root=repository_root,
        check_r2r1_occupancy=False,
    )
    active = _replace_active_string(
        r2r1_active_plan,
        R2R1_ARTIFACT_PATH_SURFACE_SHA256,
        R2R2_ARTIFACT_PATH_SURFACE_SHA256,
    )
    if not isinstance(active, dict):
        _raise("INPUT_LINEAGE_MISMATCH")
    active["artifact_path_surface"] = copy.deepcopy(
        overlay["artifact_path_surface"]
    )
    active["artifact_path_surface_sha256"] = (
        R2R2_ARTIFACT_PATH_SURFACE_SHA256
    )
    role_rows = {
        row["role"]: row for row in overlay["artifact_path_surface"]
    }
    formal_input_path = role_rows["formal_design_review_verdict"][
        "final_path"
    ]
    active["future_artifact_paths"]["formal_design_review_verdict"] = (
        formal_input_path
    )
    active["implementation_binding_contract"][
        "formal_design_review_path"
    ] = formal_input_path
    active["review_receipt_schemas"]["formal_design_review_verdict"][
        "artifact_path"
    ] = formal_input_path
    active["reviewed_implementation_authority_contract"][
        "formal_design_review_path"
    ] = formal_input_path
    pending_by_role = {
        row["role"]: row["pending_path"]
        for row in overlay["artifact_path_surface"]
    }
    for test_row in active["artifact_lifecycle_contract"][
        "full_surface_checkpoint_contract"
    ]["pending_injection_tests"]:
        test_row["injected_pending_path"] = pending_by_role[test_row["role"]]
    _merge_control_delta(
        active["implementation_binding_contract"],
        overlay["implementation_binding_contract_overlay"],
    )
    _merge_control_delta(
        active["review_receipt_schemas"][
            "fresh_implementation_review_verdict"
        ],
        overlay["fresh_review_contract_overlay"],
    )
    _merge_control_delta(
        active["reviewed_implementation_authority_contract"],
        overlay["reviewed_authority_contract_overlay"],
    )
    active["r2_activation_control"]["activation_plan_path"] = (
        R2_ACTIVATION_PLAN_RELATIVE_PATH
    )
    active["r2r2_portability_control"] = {
        "authority_namespace_id": R2R2_AUTHORITY_NAMESPACE_ID,
        "portability_plan_path": R2R2_PORTABILITY_PLAN_RELATIVE_PATH,
        "portability_plan_file_sha256": (
            R2R2_PORTABILITY_PLAN_FILE_SHA256
        ),
        "portability_plan_payload_sha256": (
            R2R2_PORTABILITY_PLAN_PAYLOAD_SHA256
        ),
        "occupied_r2r1_surface_sha256": (
            R2R2_OCCUPIED_R2R1_SURFACE_SHA256
        ),
        "repository_local_artifact_surface_sha256": (
            R2R2_REPOSITORY_LOCAL_SURFACE_SHA256
        ),
        "upstream_json_framing_surface_sha256": (
            R2R2_UPSTREAM_FRAMING_SURFACE_SHA256
        ),
        "parent_lineage": copy.deepcopy(overlay["parent_lineage"]),
        "allowed_changed_paths": copy.deepcopy(
            overlay["allowed_changed_paths"]
        ),
        "review_coverage_identity": copy.deepcopy(
            overlay["review_coverage_identity"]
        ),
        "candidate_selection_contract": copy.deepcopy(
            overlay["candidate_selection_contract"]
        ),
        "candidate_manifest_contract": copy.deepcopy(
            overlay["candidate_manifest_contract"]
        ),
        "clean_restore_receipt_contract": copy.deepcopy(
            overlay["clean_restore_receipt_contract"]
        ),
        "review_input_freeze_contract": copy.deepcopy(
            overlay["review_input_freeze_contract"]
        ),
        "fresh_review_evidence_contract": copy.deepcopy(
            overlay["fresh_review_evidence_contract"]
        ),
        "fresh_review_packet_path": overlay["fresh_review_packet_path"],
        "repository_local_artifacts": copy.deepcopy(
            overlay["repository_local_artifacts"]
        ),
        "upstream_json_framing": copy.deepcopy(
            overlay["upstream_json_framing"]
        ),
    }
    if (
        artifact_surface_sha256(active)
        != R2R2_ARTIFACT_PATH_SURFACE_SHA256
        or recompute_implementation_trust_model_sha256(active)
        != IMPLEMENTATION_TRUST_MODEL_SHA256
        or review_surface_identity(active)[
            "review_surface_identity_sha256"
        ]
        != REVIEW_SURFACE_IDENTITY_SHA256
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    nested = _walk_named_values(active, "artifact_path_surface_sha256")
    if not nested or any(
        value != R2R2_ARTIFACT_PATH_SURFACE_SHA256 for value in nested
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    return active

def load_active_plan(
    *, repository_root: Path | None = None
) -> dict[str, Any]:
    root = explicit_repository_root(repository_root)
    r2_active = load_r2_active_plan(repository_root=root)
    remediation = load_r2r1_remediation_plan(
        repository_root=root,
        r2_active_plan=r2_active,
        check_r2_occupancy=True,
    )
    r2r1_active = build_r2r1_active_plan(
        r2_active, remediation, repository_root=root
    )
    portability = load_r2r2_portability_plan(
        repository_root=root,
        r2r1_active_plan=r2r1_active,
        check_r2r1_occupancy=True,
    )
    return build_r2r2_active_plan(
        r2r1_active, portability, repository_root=root
    )

def validate_relative_manifest_path(value: object, *, allow_directory: bool = False) -> str:
    text = require_text(value, code="INPUT_LINEAGE_MISMATCH")
    upper = text.upper()
    if (
        "\\" in text
        or text.startswith("/")
        or text.endswith("/")
        or "//" in text
        or ":" in text
        or upper.startswith(("UNC", "GLOBALROOT", "DEVICE"))
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    pure = PurePosixPath(text)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        _raise("INPUT_LINEAGE_MISMATCH")
    if not allow_directory and pure.name == "":
        _raise("INPUT_LINEAGE_MISMATCH")
    return pure.as_posix()


def explicit_repository_root(value: Path | None) -> Path:
    supplied = (
        Path(__file__).resolve().parent.parent
        if value is None
        else Path(value)
    )
    if not supplied.is_absolute():
        _raise("INPUT_LINEAGE_MISMATCH")
    try:
        root = supplied.resolve(strict=True)
    except OSError:
        _raise("INPUT_LINEAGE_MISMATCH")
    if not root.is_dir():
        _raise("INPUT_LINEAGE_MISMATCH")
    return root


def validate_repository_local_artifact(
    repository_root: Path | None,
    *,
    historical_declared_path: object,
    expected_historical_declared_path: Path,
    canonical_repository_relative_path: object,
    expected_file_sha256: str,
    bound_commit: str,
    expected_git_blob_oid: str,
) -> bytes:
    root = explicit_repository_root(repository_root)
    historical = require_text(
        historical_declared_path, code="INPUT_LINEAGE_MISMATCH"
    )
    if (
        historical != str(expected_historical_declared_path)
        or not Path(historical).is_absolute()
        or not is_sha256(expected_file_sha256)
        or re.fullmatch(r"[0-9a-f]{40,64}", expected_git_blob_oid)
        is None
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    relative = validate_relative_manifest_path(
        canonical_repository_relative_path
    )
    materialized = root.joinpath(*PurePosixPath(relative).parts)
    current = root
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    try:
        for part in PurePosixPath(relative).parts:
            current = current / part
            metadata = os.lstat(current)
            if getattr(metadata, "st_file_attributes", 0) & reparse_flag:
                _raise("INPUT_LINEAGE_MISMATCH")
        resolved = materialized.resolve(strict=True)
        common = Path(os.path.commonpath((str(root), str(resolved))))
        if (
            os.path.normcase(str(common)) != os.path.normcase(str(root))
            or not resolved.is_file()
            or resolved.is_symlink()
        ):
            _raise("INPUT_LINEAGE_MISMATCH")
        raw = resolved.read_bytes()
    except Gate12C2OriginalBaselineError:
        raise
    except (OSError, ValueError):
        _raise("INPUT_LINEAGE_MISMATCH")
    if (
        sha256_bytes(raw) != expected_file_sha256
        or git_path_blob_oid(
            root,
            bound_commit,
            relative,
            code="INPUT_LINEAGE_MISMATCH",
        )
        != expected_git_blob_oid
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    return raw


def validate_formal_design_pass(
    plan: Mapping[str, Any], path: Path = FORMAL_DESIGN_REVIEW_PATH
) -> dict[str, Any]:
    schema = require_mapping(
        require_mapping(plan.get("review_receipt_schemas"), code="INPUT_LINEAGE_MISMATCH").get(
            "formal_design_review_verdict"
        ),
        code="INPUT_LINEAGE_MISMATCH",
    )
    payload = read_canonical_receipt(
        path,
        expected_file_sha256=FORMAL_DESIGN_REVIEW_FILE_SHA256,
        hash_field="formal_design_review_payload_sha256",
        expected_payload_sha256=FORMAL_DESIGN_REVIEW_PAYLOAD_SHA256,
        code="INPUT_LINEAGE_MISMATCH",
    )
    require_exact_keys(payload, schema["exact_top_level_fields"], code="INPUT_LINEAGE_MISMATCH")
    required = schema["outcomes"]["pass"]["required_values"]
    if any(payload.get(key) != value for key, value in required.items()):
        _raise("INPUT_LINEAGE_MISMATCH")
    p0_count = require_int(
        payload.get("P0_count"), minimum=0, code="INPUT_LINEAGE_MISMATCH"
    )
    p1_count = require_int(
        payload.get("P1_count"), minimum=0, code="INPUT_LINEAGE_MISMATCH"
    )
    p2_count = require_int(
        payload.get("P2_count"), minimum=0, code="INPUT_LINEAGE_MISMATCH"
    )
    if p0_count != 0 or p1_count != 0 or p2_count != 0:
        _raise("INPUT_LINEAGE_MISMATCH")
    expected = {
        "contract_file_sha256": CONTRACT_FILE_SHA256,
        "plan_file_sha256": PLAN_FILE_SHA256,
        "plan_payload_sha256": PLAN_PAYLOAD_SHA256,
        "implementation_author_separation_contract_sha256": (
            IMPLEMENTATION_AUTHOR_SEPARATION_SHA256
        ),
        "implementation_trust_model_sha256": (
            recompute_implementation_trust_model_sha256(plan)
        ),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        _raise("INPUT_LINEAGE_MISMATCH")
    parse_utc_ns(payload.get("reviewed_at_utc"), code="INPUT_LINEAGE_MISMATCH")
    return payload


def artifact_rows_by_role(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row["role"]): dict(row)
        for row in plan["artifact_path_surface"]
    }


@dataclass(frozen=True)
class ArtifactObservation:
    final_exists: bool
    pending_exists: bool = False
    outcome: str | None = None
    final_valid: bool = True


def _artifact_schema_descriptor(
    plan: Mapping[str, Any], role: str
) -> tuple[Mapping[str, Any], str]:
    control = plan["control_receipt_schemas"]
    review = plan["review_receipt_schemas"]
    descriptors = {
        "extraction_authorization": (
            control["extraction_authorization"],
            "authorization_payload_sha256",
        ),
        "extraction_authorization_verdict": (
            control["extraction_authorization_verdict"],
            "authorization_verdict_payload_sha256",
        ),
        "extraction_execution_claim": (
            control["extraction_execution_claim"],
            "execution_claim_payload_sha256",
        ),
        "extraction_failure": (
            plan["extraction_failure_receipt"],
            "failure_receipt_payload_sha256",
        ),
        "extraction_preflight": (
            control["extraction_preflight"],
            "preflight_payload_sha256",
        ),
        "extraction_success": (
            plan["success_receipt"],
            "baseline_receipt_payload_sha256",
        ),
        "extraction_terminal": (
            control["extraction_terminal"],
            "terminal_claim_payload_sha256",
        ),
        "formal_design_review_verdict": (
            review["formal_design_review_verdict"],
            "formal_design_review_payload_sha256",
        ),
        "fresh_implementation_review_verdict": (
            review["fresh_implementation_review_verdict"],
            "fresh_implementation_review_payload_sha256",
        ),
        "implementation_candidate_binding": (
            plan["implementation_binding_contract"],
            "implementation_candidate_binding_payload_sha256",
        ),
        "reviewed_implementation_authority": (
            plan["reviewed_implementation_authority_contract"],
            "reviewed_implementation_authority_payload_sha256",
        ),
        "verifier_authorization": (
            control["verifier_authorization"],
            "authorization_payload_sha256",
        ),
        "verifier_authorization_verdict": (
            control["verifier_authorization_verdict"],
            "authorization_verdict_payload_sha256",
        ),
        "verifier_execution_claim": (
            control["verifier_execution_claim"],
            "execution_claim_payload_sha256",
        ),
        "verifier_failure": (
            plan["verifier_failure_receipt"],
            "failure_receipt_payload_sha256",
        ),
        "verifier_preflight": (
            control["verifier_preflight"],
            "preflight_payload_sha256",
        ),
        "verifier_success": (
            plan["verification_receipt"],
            "verification_receipt_payload_sha256",
        ),
        "verifier_terminal": (
            control["verifier_terminal"],
            "terminal_claim_payload_sha256",
        ),
    }
    try:
        schema, hash_field = descriptors[role]
    except KeyError:
        _raise("UNEXPECTED_ARTIFACT")
    if role in REVIEW_SURFACE_BOUND_ROLES:
        effective = dict(schema)
        effective["exact_top_level_fields"] = artifact_exact_fields(plan, role)
        schema = effective
    return schema, hash_field


def _artifact_link_target(role: str, prefix: str) -> str | None:
    explicit = {
        "formal_design_review": "formal_design_review_verdict",
        "implementation_candidate_binding": "implementation_candidate_binding",
        "fresh_implementation_review": "fresh_implementation_review_verdict",
        "reviewed_implementation_authority": (
            "reviewed_implementation_authority"
        ),
        "extraction_preflight": "extraction_preflight",
        "extraction_authorization": "extraction_authorization",
        "extraction_authorization_verdict": (
            "extraction_authorization_verdict"
        ),
        "extraction_execution_claim": "extraction_execution_claim",
        "verifier_preflight": "verifier_preflight",
        "verifier_authorization": "verifier_authorization",
        "verifier_authorization_verdict": "verifier_authorization_verdict",
        "verifier_execution_claim": "verifier_execution_claim",
        "baseline_receipt": "extraction_success",
        "extraction_terminal_claim": "extraction_terminal",
    }
    target = explicit.get(prefix)
    if target is not None:
        return target
    if prefix not in {
        "preflight",
        "authorization",
        "authorization_verdict",
        "execution_claim",
    }:
        return None
    scope = "verifier" if role.startswith("verifier_") else "extraction"
    return f"{scope}_{prefix}"


def _artifact_cross_links_valid(
    plan: Mapping[str, Any],
    role: str,
    payload: Mapping[str, Any],
    payloads: Mapping[str, Mapping[str, Any]],
    file_hashes: Mapping[str, str],
    payload_hashes: Mapping[str, str],
    validity: Mapping[str, bool],
) -> bool:
    for field, value in payload.items():
        suffix = "_file_sha256"
        if not field.endswith(suffix):
            continue
        prefix = field[: -len(suffix)]
        target = _artifact_link_target(role, prefix)
        if target is None:
            continue
        payload_field = prefix + "_payload_sha256"
        if (
            target not in payloads
            or validity.get(target) is not True
            or value != file_hashes.get(target)
            or payload_field not in payload
            or payload[payload_field] != payload_hashes.get(target)
        ):
            return False
    if role not in {"extraction_terminal", "verifier_terminal"}:
        return True
    scope = "verifier" if role == "verifier_terminal" else "extraction"
    outcome = payload.get("outcome_kind")
    if outcome not in {"success", "failure"}:
        return False
    target = f"{scope}_{outcome}"
    try:
        leaf_schema, leaf_hash_field = _artifact_schema_descriptor(
            plan, target
        )
        leaf = require_mapping(
            payload.get("leaf_exact_payload"), code="UNEXPECTED_ARTIFACT"
        )
        require_exact_keys(
            leaf,
            leaf_schema["exact_top_level_fields"],
            code="UNEXPECTED_ARTIFACT",
        )
        verify_self_hash(
            leaf, leaf_hash_field, code="UNEXPECTED_ARTIFACT"
        )
    except Exception:
        return False
    if (
        payload.get("leaf_schema_version") != leaf.get("schema_version")
        or payload.get("leaf_payload_sha256") != leaf.get(leaf_hash_field)
    ):
        return False
    return target not in payloads or (
        validity.get(target) is True and payloads[target] == leaf
    )


def observe_artifact_surface(
    plan: Mapping[str, Any],
) -> dict[str, ArtifactObservation]:
    """Strictly classify all 18 final paths and all 18 pending paths."""

    outcome_fields = plan["artifact_lifecycle_contract"]["outcome_field_by_role"]
    observations: dict[str, ArtifactObservation] = {}
    payloads: dict[str, Mapping[str, Any]] = {}
    file_hashes: dict[str, str] = {}
    payload_hashes: dict[str, str] = {}
    for row in plan["artifact_path_surface"]:
        role = str(row["role"])
        final = Path(row["final_path"])
        pending = Path(row["pending_path"])
        final_exists = os.path.lexists(final)
        valid = not final_exists
        outcome: str | None = None
        if final_exists:
            try:
                schema, hash_field = _artifact_schema_descriptor(plan, role)
                if final.is_symlink() or not final.is_file():
                    _raise("UNEXPECTED_ARTIFACT")
                raw = final.read_bytes()
                if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
                    _raise("UNEXPECTED_ARTIFACT")
                payload = require_mapping(
                    strict_json_loads(raw[:-1], canonical=True),
                    code="UNEXPECTED_ARTIFACT",
                )
                require_exact_keys(
                    payload,
                    schema["exact_top_level_fields"],
                    code="UNEXPECTED_ARTIFACT",
                )
                verify_self_hash(payload, hash_field, code="UNEXPECTED_ARTIFACT")
                if role in REVIEW_SURFACE_BOUND_ROLES:
                    validate_review_surface_identity(
                        plan,
                        payload.get("review_surface_identity"),
                        code="UNEXPECTED_ARTIFACT",
                    )
                if raw != canonical_receipt_bytes(payload):
                    _raise("UNEXPECTED_ARTIFACT")
                if role in outcome_fields:
                    value = payload.get(outcome_fields[role])
                    outcome = value if type(value) is str else "__invalid__"
                payloads[role] = payload
                file_hashes[role] = sha256_bytes(raw)
                payload_hashes[role] = str(payload[hash_field])
                valid = True
            except Exception:
                outcome = "__invalid__"
                valid = False
        observations[role] = ArtifactObservation(
            final_exists=final_exists,
            pending_exists=os.path.lexists(pending),
            outcome=outcome,
            final_valid=valid,
        )
    validity = {
        role: observation.final_valid
        for role, observation in observations.items()
    }
    changed = True
    while changed:
        changed = False
        for role, payload in payloads.items():
            if validity[role] and not _artifact_cross_links_valid(
                plan,
                role,
                payload,
                payloads,
                file_hashes,
                payload_hashes,
                validity,
            ):
                validity[role] = False
                changed = True
    for role, valid in validity.items():
        if observations[role].final_exists and not valid:
            observations[role] = ArtifactObservation(
                final_exists=True,
                pending_exists=observations[role].pending_exists,
                outcome="__invalid__",
                final_valid=False,
            )
    return observations


def classify_lifecycle_surface(
    plan: Mapping[str, Any],
    observations: Mapping[str, ArtifactObservation],
    *,
    temporal_predicate: str = "not_applicable",
    liveness: str = "not_applicable",
) -> str:
    """Classify a fully observed 18-role surface against the 33 frozen phases."""

    roles = tuple(plan["artifact_lifecycle_contract"]["roles"])
    if set(observations) != set(roles):
        return "HOLD_new_review"
    if any(value.pending_exists for value in observations.values()):
        return "HOLD_new_review"
    if any(value.final_exists and not value.final_valid for value in observations.values()):
        return "HOLD_new_review"
    matches: list[str] = []
    for phase in plan["artifact_lifecycle_contract"]["stable_phases"]:
        if any(not observations[role].final_exists for role in phase["must_exist"]):
            continue
        if any(observations[role].final_exists for role in phase["must_be_absent"]):
            continue
        if any(
            (
                observations[role].outcome not in {"success", "failure"}
                if expected == "success_or_failure"
                else observations[role].outcome != expected
            )
            for role, expected in phase["required_outcomes"].items()
        ):
            continue
        required_time = phase["temporal_predicate"]
        if required_time != "not_applicable" and required_time != temporal_predicate:
            continue
        required_live = phase["liveness_predicate"]
        if required_live == "ACTIVE_exact_owner" and liveness != "ACTIVE":
            continue
        if required_live == "DEAD_or_UNKNOWN" and liveness not in {"DEAD", "UNKNOWN"}:
            continue
        if required_live == "not_applicable" and liveness != "not_applicable":
            continue
        matches.append(str(phase["phase"]))
    return matches[0] if len(matches) == 1 else "HOLD_new_review"


def require_full_surface_checkpoint(
    plan: Mapping[str, Any],
    *,
    scope: str,
    checkpoint: str,
    state: str,
    temporal_predicate: str = "not_applicable",
    liveness: str = "not_applicable",
) -> str:
    contract = plan["artifact_lifecycle_contract"][
        "full_surface_checkpoint_contract"
    ]
    rows = list(contract["checkpoint_rows"]) + list(
        contract["failure_publication_checkpoint_contract"][
            "failure_checkpoint_rows"
        ]
    )
    matches = [
        row
        for row in rows
        if row["scope"] == scope
        and row["checkpoint"] == checkpoint
        and row["expected_state"] == state
    ]
    if len(matches) != 1:
        _raise("UNEXPECTED_ARTIFACT")
    row = matches[0]
    if any(
        row[key] != 18
        for key in (
            "role_count_classified",
            "final_path_count_classified",
            "pending_path_count_classified",
        )
    ):
        _raise("UNEXPECTED_ARTIFACT")
    observed = observe_artifact_surface(plan)
    phase = classify_lifecycle_surface(
        plan,
        observed,
        temporal_predicate=temporal_predicate,
        liveness=liveness,
    )
    if phase != row["expected_artifact_phase"]:
        _raise("UNEXPECTED_ARTIFACT")
    return phase


def transition_state(plan: Mapping[str, Any], source: str, event: str) -> str:
    matches = [
        target
        for candidate_source, candidate_event, target in plan["state_model"]["transitions"]
        if candidate_source == source and candidate_event == event
    ]
    if len(matches) != 1:
        _raise("UNEXPECTED_ARTIFACT")
    return str(matches[0])


@dataclass(frozen=True)
class PublicationResult:
    state: str
    final_path: Path
    pending_path: Path


def _path_bytes_if_regular(path: Path) -> bytes | None:
    try:
        if path.is_symlink() or not path.is_file():
            return None
        return path.read_bytes()
    except OSError:
        return None


def classify_publication_after_exception(
    final_path: Path, pending_path: Path, expected: bytes
) -> PublicationResult:
    final = _path_bytes_if_regular(final_path)
    pending = _path_bytes_if_regular(pending_path)
    if final == expected and pending is None:
        return PublicationResult("published_exact", final_path, pending_path)
    if final is None and pending is None:
        return PublicationResult("no_durable_transition", final_path, pending_path)
    return PublicationResult("ambiguous_hold_new_review", final_path, pending_path)


def atomic_publish_exact(
    final_path: Path,
    payload: bytes,
    *,
    pending_path: Path | None = None,
) -> PublicationResult:
    """Publish canonical bytes through CREATE_NEW and a nonreplace rename."""

    final = Path(final_path)
    pending = (
        Path(pending_path)
        if pending_path is not None
        else final.with_name(final.name + ".pending-v0.9")
    )
    allowed_pending_names = {
        final.name + ".pending-v0.9",
        final.name + ".pending-" + R2_AUTHORITY_NAMESPACE_ID,
        final.name + ".pending-" + R2R1_AUTHORITY_NAMESPACE_ID,
        final.name + ".pending-" + R2R2_AUTHORITY_NAMESPACE_ID,
    }
    if final.parent != pending.parent or pending.name not in allowed_pending_names:
        _raise("OUTPUT_PUBLICATION_FAILED")
    if not final.parent.is_dir() or final.exists() or pending.exists():
        _raise("UNEXPECTED_ARTIFACT")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            pending,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                _raise("OUTPUT_PUBLICATION_FAILED")
            written += count
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        if _path_bytes_if_regular(pending) != payload:
            _raise("OUTPUT_PUBLICATION_FAILED")
        if os.name == "nt":
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            move = kernel32.MoveFileExW
            move.argtypes = [
                ctypes.wintypes.LPCWSTR,
                ctypes.wintypes.LPCWSTR,
                ctypes.wintypes.DWORD,
            ]
            move.restype = ctypes.wintypes.BOOL
            if not move(str(pending), str(final), 0x00000008):
                raise OSError(ctypes.get_last_error(), "MoveFileExW")
        else:  # Test-only portability. Production is frozen to Windows.
            os.link(pending, final)
            os.unlink(pending)
        if _path_bytes_if_regular(final) != payload or pending.exists():
            _raise("OUTPUT_PUBLICATION_FAILED")
        return PublicationResult("published_exact", final, pending)
    except Gate12C2OriginalBaselineError:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        classified = classify_publication_after_exception(final, pending, payload)
        if classified.state == "published_exact":
            return classified
        raise
    except Exception:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        classified = classify_publication_after_exception(final, pending, payload)
        if classified.state == "published_exact":
            return classified
        _raise("OUTPUT_PUBLICATION_FAILED")


def publish_role(
    plan: Mapping[str, Any], role: str, payload: Mapping[str, Any]
) -> PublicationResult:
    rows = artifact_rows_by_role(plan)
    if role not in rows:
        _raise("UNEXPECTED_ARTIFACT")
    row = rows[role]
    remediation_control = active_remediation_control(plan)
    if remediation_control is not None:
        namespace = remediation_control["authority_namespace_id"]
        if (
            role == "formal_design_review_verdict"
            or namespace not in Path(row["final_path"]).name
            or namespace not in Path(row["pending_path"]).name
        ):
            _raise("UNEXPECTED_ARTIFACT")
    elif r2_activation_control(plan) is not None:
        if (
            role == "formal_design_review_verdict"
            or R2_AUTHORITY_NAMESPACE_ID not in Path(row["final_path"]).name
            or R2_AUTHORITY_NAMESPACE_ID not in Path(row["pending_path"]).name
        ):
            _raise("UNEXPECTED_ARTIFACT")
    return atomic_publish_exact(
        Path(row["final_path"]),
        canonical_receipt_bytes(payload),
        pending_path=Path(row["pending_path"]),
    )



def publish_r2_control_receipt(
    plan: Mapping[str, Any],
    contract_name: str,
    payload: Mapping[str, Any],
) -> PublicationResult:
    control = r2_activation_control(plan)
    if (
        control is None
        or contract_name
        not in {
            "candidate_manifest_contract",
            "clean_restore_receipt_contract",
            "fresh_review_evidence_contract",
        }
    ):
        _raise("UNEXPECTED_ARTIFACT")
    contract = require_mapping(
        control.get(contract_name),
        code="UNEXPECTED_ARTIFACT",
    )
    exact_fields = contract["exact_top_level_fields"]
    require_exact_keys(payload, exact_fields, code="INPUT_LINEAGE_MISMATCH")
    self_hash_field = exact_fields[-1]
    verify_self_hash(payload, self_hash_field, code="INPUT_LINEAGE_MISMATCH")
    return atomic_publish_exact(
        Path(contract["artifact_path"]),
        canonical_receipt_bytes(payload),
        pending_path=Path(contract["pending_path"]),
    )

def _find_self_hash_field(
    payload: Mapping[str, Any], expected_payload_sha256: str
) -> tuple[str, bool]:
    candidates: list[tuple[str, bool]] = []
    for field, value in payload.items():
        if value != expected_payload_sha256 or not field.endswith("sha256"):
            continue
        unhashed = dict(payload)
        unhashed.pop(field)
        canonical = canonical_json_bytes(unhashed)
        if sha256_bytes(canonical) == expected_payload_sha256:
            candidates.append((field, False))
        if sha256_bytes(canonical + b"\n") == expected_payload_sha256:
            candidates.append((field, True))
    if len(candidates) != 1:
        _raise("INPUT_LINEAGE_MISMATCH")
    return candidates[0]


def read_frozen_json_artifact(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_payload_sha256: str,
) -> dict[str, Any]:
    raw = read_exact_bytes(path, expected_file_sha256, code="INPUT_LINEAGE_MISMATCH")
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        _raise("INPUT_LINEAGE_MISMATCH")
    payload = require_mapping(
        strict_json_loads(raw[:-1], canonical=True), code="INPUT_LINEAGE_MISMATCH"
    )
    field, includes_lf = _find_self_hash_field(payload, expected_payload_sha256)
    verify_self_hash(
        payload,
        field,
        include_lf=includes_lf,
        code="INPUT_LINEAGE_MISMATCH",
    )
    return payload


FROZEN_JSON_WITHOUT_LF = (
    "canonical_JSON_without_self_hash_no_terminating_LF"
)
FROZEN_JSON_WITH_SINGLE_LF = (
    "canonical_JSON_without_self_hash_plus_single_LF"
)


def read_declared_frozen_json_artifact(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_payload_sha256: str,
    payload_hash_domain: object,
    self_hash_field: object,
    expected_schema_version: object,
) -> dict[str, Any]:
    domain = require_text(
        payload_hash_domain, code="INPUT_LINEAGE_MISMATCH"
    )
    field = require_text(self_hash_field, code="INPUT_LINEAGE_MISMATCH")
    schema = require_text(
        expected_schema_version, code="INPUT_LINEAGE_MISMATCH"
    )
    if domain not in {FROZEN_JSON_WITHOUT_LF, FROZEN_JSON_WITH_SINGLE_LF}:
        _raise("INPUT_LINEAGE_MISMATCH")
    raw = read_exact_bytes(
        path, expected_file_sha256, code="INPUT_LINEAGE_MISMATCH"
    )
    if domain == FROZEN_JSON_WITHOUT_LF:
        if raw.endswith((b"\n", b"\r")):
            _raise("INPUT_LINEAGE_MISMATCH")
        encoded = raw
        suffix = b""
    else:
        if (
            not raw.endswith(b"\n")
            or raw.endswith((b"\r\n", b"\n\n"))
        ):
            _raise("INPUT_LINEAGE_MISMATCH")
        encoded = raw[:-1]
        suffix = b"\n"
    payload = require_mapping(
        strict_json_loads(encoded, canonical=True),
        code="INPUT_LINEAGE_MISMATCH",
    )
    if (
        payload.get("schema_version") != schema
        or field not in payload
        or payload.get(field) != expected_payload_sha256
        or canonical_json_bytes(payload) + suffix != raw
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    unhashed = dict(payload)
    del unhashed[field]
    if (
        sha256_bytes(canonical_json_bytes(unhashed) + suffix)
        != expected_payload_sha256
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    return payload


def validate_upstream_authority(
    plan: Mapping[str, Any],
) -> dict[str, dict[str, Any] | bytes]:
    authority = require_mapping(
        plan.get("upstream_authority"), code="INPUT_LINEAGE_MISMATCH"
    )
    rows = authority.get("artifact_rows")
    if not isinstance(rows, list) or len(rows) != 4:
        _raise("INPUT_LINEAGE_MISMATCH")
    result: dict[str, dict[str, Any] | bytes] = {}
    for row_value in rows:
        row = require_mapping(row_value, code="INPUT_LINEAGE_MISMATCH")
        require_exact_keys(
            row,
            {
                "file_sha256",
                "format",
                "path",
                "payload_hash_domain",
                "payload_sha256",
                "role",
                "schema_version",
                "self_hash_field",
            },
            code="INPUT_LINEAGE_MISMATCH",
        )
        path = Path(
            require_text(row.get("path"), code="INPUT_LINEAGE_MISMATCH")
        )
        role = require_text(
            row.get("role"), ascii_only=True, code="INPUT_LINEAGE_MISMATCH"
        )
        file_hash = row.get("file_sha256")
        if not is_sha256(file_hash):
            _raise("INPUT_LINEAGE_MISMATCH")
        if row.get("format") == "markdown_contract":
            if any(
                row.get(field) is not None
                for field in (
                    "payload_hash_domain",
                    "payload_sha256",
                    "schema_version",
                    "self_hash_field",
                )
            ):
                _raise("INPUT_LINEAGE_MISMATCH")
            result[role] = read_exact_bytes(
                path, str(file_hash), code="INPUT_LINEAGE_MISMATCH"
            )
            continue
        if row.get("format") != "canonical_self_hashed_JSON":
            _raise("INPUT_LINEAGE_MISMATCH")
        payload_hash = row.get("payload_sha256")
        if not is_sha256(payload_hash):
            _raise("INPUT_LINEAGE_MISMATCH")
        result[role] = read_declared_frozen_json_artifact(
            path,
            expected_file_sha256=str(file_hash),
            expected_payload_sha256=str(payload_hash),
            payload_hash_domain=row.get("payload_hash_domain"),
            self_hash_field=row.get("self_hash_field"),
            expected_schema_version=row.get("schema_version"),
        )
    return result


def _recursive_values(value: object, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for candidate, item in value.items():
            if candidate == key:
                found.append(item)
            found.extend(_recursive_values(item, key))
    elif isinstance(value, list):
        for item in value:
            found.extend(_recursive_values(item, key))
    return found


def validate_original_input_lineage(
    plan: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    lineage = require_mapping(
        plan.get("original_input_lineage"), code="INPUT_LINEAGE_MISMATCH"
    )
    specifications = (
        (
            "original_plan",
            "original_plan_path",
            "original_plan_file_sha256",
            "original_plan_payload_sha256",
        ),
        (
            "incident_manifest",
            "incident_manifest_path",
            "incident_manifest_file_sha256",
            "incident_manifest_payload_sha256",
        ),
        (
            "payload_seal",
            "payload_seal_path",
            "payload_seal_file_sha256",
            "payload_seal_payload_sha256",
        ),
        (
            "payload_seal_verification",
            "payload_seal_verification_path",
            "payload_seal_verification_file_sha256",
            "payload_seal_verification_payload_sha256",
        ),
        (
            "formal_payload_closeout",
            "formal_payload_closeout_path",
            "formal_payload_closeout_file_sha256",
            "formal_payload_closeout_payload_sha256",
        ),
    )
    result: dict[str, dict[str, Any]] = {}
    for role, path_key, file_key, payload_key in specifications:
        path = Path(
            require_text(lineage.get(path_key), code="INPUT_LINEAGE_MISMATCH")
        )
        file_hash = lineage.get(file_key)
        payload_hash = lineage.get(payload_key)
        if not is_sha256(file_hash) or not is_sha256(payload_hash):
            _raise("INPUT_LINEAGE_MISMATCH")
        result[role] = read_frozen_json_artifact(
            path,
            expected_file_sha256=str(file_hash),
            expected_payload_sha256=str(payload_hash),
        )
    original_plan = result["original_plan"]
    if (
        original_plan.get("plan_payload_sha256")
        != lineage["original_plan_payload_sha256"]
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    for role in (
        "payload_seal",
        "payload_seal_verification",
        "formal_payload_closeout",
    ):
        receipt = result[role]
        for field in (
            "scientific_values_emitted",
            "stability_analysis_authorized",
            "locked_execution_authorized",
            "real_held_out_execution_authorized",
            "N2_open",
            "N3_open",
        ):
            values = _recursive_values(receipt, field)
            if values and any(value is not False for value in values):
                _raise("INPUT_LINEAGE_MISMATCH")
    validate_incident_manifest(plan, result["incident_manifest"])
    return result


INCIDENT_MANIFEST_FIELDS = {
    "schema_version",
    "incident_id",
    "epistemic_status",
    "state",
    "observed_at_utc",
    "output_root",
    "original_source_commit",
    "original_plan_payload_sha256",
    "inspection_contract",
    "summary",
    "files",
    "directories",
    "payload_presence_observed",
    "payload_integrity_status",
    "index_integrity_status",
    "original_execution_closeout_status",
    "original_resource_evidence_status",
    "resource_gate_status",
    "scientific_values_emitted",
    "stability_analysis_authorized",
    "locked_execution_authorized",
    "real_held_out_execution_authorized",
    "N2_open",
    "N3_open",
    "incident_manifest_payload_sha256",
}
MANIFEST_FILE_FIELDS = {
    "canonical_relative_path",
    "file_size_bytes",
    "sha256",
    "file_class",
    "plane",
    "exists",
    "expected",
    "unexpected",
    "partial_or_temp",
    "reparse_point",
}
MANIFEST_DIRECTORY_FIELDS = {
    "canonical_relative_path",
    "expected",
    "unexpected",
    "reparse_point",
}


def validate_incident_manifest(
    plan: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    supplied = dict(manifest)
    require_exact_keys(
        supplied, INCIDENT_MANIFEST_FIELDS, code="INPUT_LINEAGE_MISMATCH"
    )
    verify_self_hash(
        supplied,
        "incident_manifest_payload_sha256",
        code="INPUT_LINEAGE_MISMATCH",
    )
    lineage = plan["original_input_lineage"]
    expected_values = {
        "schema_version": "gate12c2_closeout_incident_byte_manifest_v0.1",
        "state": "INCIDENT_FROZEN",
        "output_root": str(PROTECTED_ROOT).replace("\\", "/"),
        "original_source_commit": lineage["original_source_commit"],
        "original_plan_payload_sha256": lineage["original_plan_payload_sha256"],
        "payload_presence_observed": "768/768",
        "scientific_values_emitted": False,
        "stability_analysis_authorized": False,
        "locked_execution_authorized": False,
        "real_held_out_execution_authorized": False,
        "N2_open": False,
        "N3_open": False,
    }
    if any(supplied.get(key) != value for key, value in expected_values.items()):
        _raise("INPUT_LINEAGE_MISMATCH")
    parse_utc_ns(
        supplied.get("observed_at_utc"), code="INPUT_LINEAGE_MISMATCH"
    )
    inspection = require_mapping(
        supplied.get("inspection_contract"), code="INPUT_LINEAGE_MISMATCH"
    )
    if any(
        inspection.get(key) is not False
        for key in (
            "json_parsed",
            "gzip_parsed",
            "npz_parsed",
            "scientific_values_inspected",
        )
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    rows = supplied.get("files")
    directories = supplied.get("directories")
    if not isinstance(rows, list) or len(rows) != EXPECTED_FILE_COUNT:
        _raise("INPUT_LINEAGE_MISMATCH")
    if (
        not isinstance(directories, list)
        or len(directories) != EXPECTED_DIRECTORY_COUNT
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    relative_paths: list[str] = []
    shard_count = index_count = protected_count = 0
    for value in rows:
        row = require_mapping(value, code="INPUT_LINEAGE_MISMATCH")
        require_exact_keys(
            row, MANIFEST_FILE_FIELDS, code="INPUT_LINEAGE_MISMATCH"
        )
        relative = validate_relative_manifest_path(
            row["canonical_relative_path"]
        )
        relative_paths.append(relative)
        require_int(
            row["file_size_bytes"], minimum=0, code="INPUT_LINEAGE_MISMATCH"
        )
        if not is_sha256(row["sha256"]):
            _raise("INPUT_LINEAGE_MISMATCH")
        required_flags = {
            "exists": True,
            "expected": True,
            "unexpected": False,
            "partial_or_temp": False,
            "reparse_point": False,
        }
        if any(
            row[field] is not expected
            for field, expected in required_flags.items()
        ):
            _raise("INPUT_LINEAGE_MISMATCH")
        if row["file_class"] == "shard":
            shard_count += 1
        if row["file_class"] == "index":
            index_count += 1
        if row["plane"] == "protected_payload":
            protected_count += 1
    if (
        relative_paths != sorted(relative_paths)
        or len(set(relative_paths)) != len(relative_paths)
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    if (shard_count, index_count, protected_count) != (768, 9, 790):
        _raise("INPUT_LINEAGE_MISMATCH")
    directory_paths: list[str] = []
    for value in directories:
        row = require_mapping(value, code="INPUT_LINEAGE_MISMATCH")
        require_exact_keys(
            row, MANIFEST_DIRECTORY_FIELDS, code="INPUT_LINEAGE_MISMATCH"
        )
        directory_paths.append(
            validate_relative_manifest_path(
                row["canonical_relative_path"], allow_directory=True
            )
        )
        if (
            row["expected"] is not True
            or row["unexpected"] is not False
            or row["reparse_point"] is not False
        ):
            _raise("INPUT_LINEAGE_MISMATCH")
    if (
        directory_paths != sorted(directory_paths)
        or len(set(directory_paths)) != len(directory_paths)
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    protected_rows = [
        row for row in rows if row["plane"] == "protected_payload"
    ]
    summary = require_mapping(
        supplied.get("summary"), code="INPUT_LINEAGE_MISMATCH"
    )
    expected_summary = {
        "expected_file_count": 791,
        "existing_file_count": 791,
        "protected_expected_count": 790,
        "protected_existing_count": 790,
        "shard_existing_count": 768,
        "index_existing_count": 9,
        "frozen_lineage_existing_count": 13,
        "control_existing_count": 1,
        "missing_expected_count": 0,
        "unexpected_file_count": 0,
        "partial_or_temp_count": 0,
        "reparse_file_count": 0,
        "unexpected_directory_count": 0,
        "reparse_directory_count": 0,
        "protected_surface_sha256": EXPECTED_PROTECTED_SURFACE_SHA256,
        "complete_surface_sha256": EXPECTED_COMPLETE_SURFACE_SHA256,
    }
    if any(summary.get(key) != value for key, value in expected_summary.items()):
        _raise("INPUT_LINEAGE_MISMATCH")
    if (
        sha256_bytes(canonical_json_bytes(rows))
        != EXPECTED_COMPLETE_SURFACE_SHA256
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    if (
        sha256_bytes(canonical_json_bytes(protected_rows))
        != EXPECTED_PROTECTED_SURFACE_SHA256
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    return supplied



class _FILE_ATTRIBUTE_TAG_INFO(ctypes.Structure):
    _fields_ = [
        ("FileAttributes", ctypes.wintypes.DWORD),
        ("ReparseTag", ctypes.wintypes.DWORD),
    ]


class _FILE_ID_128(ctypes.Structure):
    _fields_ = [("Identifier", ctypes.c_ubyte * 16)]


class _FILE_ID_INFO(ctypes.Structure):
    _fields_ = [
        ("VolumeSerialNumber", ctypes.c_ulonglong),
        ("FileId", _FILE_ID_128),
    ]


@dataclass(frozen=True)
class HandleIdentity:
    volume_serial: int
    file_id: bytes
    final_path: str
    size: int | None


@dataclass
class RetainedHandle:
    handle: int
    identity: HandleIdentity


def windows_ordinal_equal(left: str, right: str, api: object) -> bool:
    """Compare Windows paths with the frozen ordinal-ignore-case primitive."""

    compare = getattr(api, "CompareStringOrdinal", None)
    if compare is None:
        _raise("READ_ONLY_HANDLE_FAILED")
    try:
        compare.argtypes = [
            ctypes.wintypes.LPCWSTR,
            ctypes.c_int,
            ctypes.wintypes.LPCWSTR,
            ctypes.c_int,
            ctypes.wintypes.BOOL,
        ]
        compare.restype = ctypes.c_int
    except Exception:
        pass
    try:
        result = int(compare(left, -1, right, -1, True))
    except Exception:
        _raise("READ_ONLY_HANDLE_FAILED")
    if result == 0:
        _raise("READ_ONLY_HANDLE_FAILED")
    return result == 2


class RetainedProtectedSurface:
    """Own the frozen root, 23 directory, and 791 file handles."""

    GENERIC_READ = 0x80000000
    FILE_LIST_DIRECTORY = 0x0001
    FILE_READ_ATTRIBUTES = 0x0080
    FILE_SHARE_READ = 0x00000001
    OPEN_EXISTING = 3
    FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
    FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
    FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000
    FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
    FILE_ATTRIBUTE_TAG_INFO_CLASS = 9
    FILE_ID_INFO_CLASS = 18
    FILE_NAME_NORMALIZED = 0
    VOLUME_NAME_DOS = 0
    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

    def __init__(
        self,
        root: Path,
        manifest: Mapping[str, Any],
        *,
        api: object | None = None,
    ) -> None:
        if os.name != "nt" and api is None:
            _raise("READ_ONLY_HANDLE_FAILED")
        self.root = Path(root)
        self.manifest = dict(manifest)
        self.api = api if api is not None else ctypes.WinDLL(
            "kernel32", use_last_error=True
        )
        self.root_handle: RetainedHandle | None = None
        self.directories: dict[str, RetainedHandle] = {}
        self.files: dict[str, RetainedHandle] = {}
        self._manifest_rows = {
            str(row["canonical_relative_path"]): dict(row)
            for row in self.manifest["files"]
        }
        self._manifest_directories = {
            str(row["canonical_relative_path"]): dict(row)
            for row in self.manifest["directories"]
        }
        self._pre_hashes: dict[str, str] = {}
        self._configure_api()

    def _configure_api(self) -> None:
        for name, argtypes, restype in (
            (
                "CreateFileW",
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
            (
                "GetFileInformationByHandleEx",
                [
                    ctypes.wintypes.HANDLE,
                    ctypes.c_int,
                    ctypes.c_void_p,
                    ctypes.wintypes.DWORD,
                ],
                ctypes.wintypes.BOOL,
            ),
            (
                "GetFinalPathNameByHandleW",
                [
                    ctypes.wintypes.HANDLE,
                    ctypes.wintypes.LPWSTR,
                    ctypes.wintypes.DWORD,
                    ctypes.wintypes.DWORD,
                ],
                ctypes.wintypes.DWORD,
            ),
            (
                "GetFileSizeEx",
                [ctypes.wintypes.HANDLE, ctypes.POINTER(ctypes.c_longlong)],
                ctypes.wintypes.BOOL,
            ),
            (
                "SetFilePointerEx",
                [
                    ctypes.wintypes.HANDLE,
                    ctypes.c_longlong,
                    ctypes.POINTER(ctypes.c_longlong),
                    ctypes.wintypes.DWORD,
                ],
                ctypes.wintypes.BOOL,
            ),
            (
                "ReadFile",
                [
                    ctypes.wintypes.HANDLE,
                    ctypes.c_void_p,
                    ctypes.wintypes.DWORD,
                    ctypes.POINTER(ctypes.wintypes.DWORD),
                    ctypes.c_void_p,
                ],
                ctypes.wintypes.BOOL,
            ),
            (
                "CloseHandle",
                [ctypes.wintypes.HANDLE],
                ctypes.wintypes.BOOL,
            ),
        ):
            function = getattr(self.api, name)
            try:
                function.argtypes = argtypes
                function.restype = restype
            except Exception:
                pass

    def _open(self, path: Path, *, directory: bool) -> int:
        access = (
            self.FILE_LIST_DIRECTORY | self.FILE_READ_ATTRIBUTES
            if directory
            else self.GENERIC_READ
        )
        flags = self.FILE_FLAG_OPEN_REPARSE_POINT | (
            self.FILE_FLAG_BACKUP_SEMANTICS
            if directory
            else self.FILE_FLAG_SEQUENTIAL_SCAN
        )
        handle = self.api.CreateFileW(
            str(path),
            access,
            self.FILE_SHARE_READ,
            None,
            self.OPEN_EXISTING,
            flags,
            None,
        )
        numeric = int(
            handle
            if isinstance(handle, int)
            else ctypes.cast(handle, ctypes.c_void_p).value or 0
        )
        if numeric in {0, self.INVALID_HANDLE_VALUE}:
            _raise("READ_ONLY_HANDLE_FAILED")
        return numeric

    def _tag(self, handle: int) -> _FILE_ATTRIBUTE_TAG_INFO:
        info = _FILE_ATTRIBUTE_TAG_INFO()
        if not self.api.GetFileInformationByHandleEx(
            handle,
            self.FILE_ATTRIBUTE_TAG_INFO_CLASS,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ):
            _raise("READ_ONLY_HANDLE_FAILED")
        if (
            info.FileAttributes & self.FILE_ATTRIBUTE_REPARSE_POINT
            or info.ReparseTag != 0
        ):
            _raise("PROTECTED_ROOT_REPARSE_POINT")
        return info

    def _file_id(self, handle: int) -> tuple[int, bytes]:
        info = _FILE_ID_INFO()
        if not self.api.GetFileInformationByHandleEx(
            handle,
            self.FILE_ID_INFO_CLASS,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ):
            _raise("READ_ONLY_HANDLE_FAILED")
        return int(info.VolumeSerialNumber), bytes(info.FileId.Identifier)

    def _final_path(self, handle: int) -> str:
        required = self.api.GetFinalPathNameByHandleW(
            handle, None, 0, self.FILE_NAME_NORMALIZED | self.VOLUME_NAME_DOS
        )
        if not required:
            _raise("READ_ONLY_HANDLE_FAILED")
        buffer = ctypes.create_unicode_buffer(int(required) + 1)
        written = self.api.GetFinalPathNameByHandleW(
            handle,
            buffer,
            len(buffer),
            self.FILE_NAME_NORMALIZED | self.VOLUME_NAME_DOS,
        )
        if not written or written >= len(buffer):
            _raise("READ_ONLY_HANDLE_FAILED")
        value = buffer.value
        upper = value.upper()
        if upper.startswith("\\\\?\\UNC\\") or upper.startswith(
            ("\\\\.\\", "\\\\?\\VOLUME", "\\\\?\\GLOBALROOT")
        ):
            _raise("FINAL_PATH_OUTSIDE_PROTECTED_ROOT")
        if value.startswith("\\\\?\\") and len(value) >= 7 and value[4].isalpha():
            value = value[4:]
        elif value.startswith("\\\\?\\"):
            _raise("FINAL_PATH_OUTSIDE_PROTECTED_ROOT")
        return value

    def _same_path(self, left: str, right: str) -> bool:
        return windows_ordinal_equal(left, right, self.api)

    def _size(self, handle: int) -> int:
        size = ctypes.c_longlong()
        if not self.api.GetFileSizeEx(handle, ctypes.byref(size)):
            _raise("READ_ONLY_HANDLE_FAILED")
        if size.value < 0:
            _raise("READ_ONLY_HANDLE_FAILED")
        return int(size.value)

    def _identity(
        self, handle: int, expected_path: Path, *, directory: bool
    ) -> HandleIdentity:
        self._tag(handle)
        volume, file_id = self._file_id(handle)
        final_path = self._final_path(handle)
        if not self._same_path(final_path, str(expected_path)):
            _raise("FINAL_PATH_OUTSIDE_PROTECTED_ROOT")
        return HandleIdentity(
            volume_serial=volume,
            file_id=file_id,
            final_path=final_path,
            size=None if directory else self._size(handle),
        )

    def acquire(self) -> "RetainedProtectedSurface":
        root_handle = self._open(self.root, directory=True)
        try:
            root_identity = self._identity(
                root_handle, self.root, directory=True
            )
            self.root_handle = RetainedHandle(root_handle, root_identity)
            for relative in sorted(self._manifest_directories):
                expected = self.root.joinpath(*PurePosixPath(relative).parts)
                handle = self._open(expected, directory=True)
                identity = self._identity(handle, expected, directory=True)
                if identity.volume_serial != root_identity.volume_serial:
                    _raise("FINAL_PATH_OUTSIDE_PROTECTED_ROOT")
                self.directories[relative] = RetainedHandle(handle, identity)
            for relative in sorted(self._manifest_rows):
                expected = self.root.joinpath(*PurePosixPath(relative).parts)
                handle = self._open(expected, directory=False)
                identity = self._identity(handle, expected, directory=False)
                row = self._manifest_rows[relative]
                if (
                    identity.volume_serial != root_identity.volume_serial
                    or identity.size != row["file_size_bytes"]
                ):
                    _raise("FILE_IDENTITY_CHANGED")
                self.files[relative] = RetainedHandle(handle, identity)
        except Exception:
            self.close()
            raise
        if (
            len(self.directories) != EXPECTED_DIRECTORY_COUNT
            or len(self.files) != EXPECTED_FILE_COUNT
            or 1 + len(self.directories) + len(self.files) != 815
        ):
            self.close()
            _raise("ZERO_COVERAGE")
        return self

    def _read_all(self, retained: RetainedHandle) -> bytes:
        if retained.identity.size is None:
            _raise("READ_ONLY_HANDLE_FAILED")
        if not self.api.SetFilePointerEx(
            retained.handle, 0, None, 0
        ):
            _raise("READ_ONLY_HANDLE_FAILED")
        remaining = retained.identity.size
        chunks: list[bytes] = []
        while remaining:
            requested = min(remaining, 1024 * 1024)
            buffer = ctypes.create_string_buffer(requested)
            received = ctypes.wintypes.DWORD()
            if not self.api.ReadFile(
                retained.handle,
                buffer,
                requested,
                ctypes.byref(received),
                None,
            ):
                _raise("READ_ONLY_HANDLE_FAILED")
            count = int(received.value)
            if count <= 0 or count > requested:
                _raise("READ_ONLY_HANDLE_FAILED")
            chunks.append(buffer.raw[:count])
            remaining -= count
        return b"".join(chunks)

    def read(self, relative: str) -> bytes:
        normalized = validate_relative_manifest_path(relative)
        retained = self.files.get(normalized)
        if retained is None:
            _raise("UNEXPECTED_ARTIFACT")
        return self._read_all(retained)

    def _enumerated_names(self) -> tuple[set[str], set[str]]:
        files: set[str] = set()
        directories: set[str] = set()
        try:
            for current, dir_names, file_names in os.walk(
                self.root, topdown=True, followlinks=False
            ):
                current_path = Path(current)
                for name in dir_names:
                    candidate = current_path / name
                    relative = candidate.relative_to(self.root).as_posix()
                    if candidate.is_symlink():
                        _raise("PROTECTED_ROOT_REPARSE_POINT")
                    directories.add(relative)
                for name in file_names:
                    candidate = current_path / name
                    relative = candidate.relative_to(self.root).as_posix()
                    if candidate.is_symlink():
                        _raise("PROTECTED_ROOT_REPARSE_POINT")
                    files.add(relative)
        except Gate12C2OriginalBaselineError:
            raise
        except OSError:
            _raise("PROTECTED_ROOT_SURFACE_MISMATCH")
        return files, directories

    def verify_pre_manifest(self) -> dict[str, str]:
        files, directories = self._enumerated_names()
        if files != set(self.files) or directories != set(self.directories):
            _raise("PROTECTED_ROOT_SURFACE_MISMATCH")
        for relative in sorted(self.files):
            raw = self._read_all(self.files[relative])
            digest = sha256_bytes(raw)
            if digest != self._manifest_rows[relative]["sha256"]:
                _raise("PROTECTED_ROOT_SURFACE_MISMATCH")
            self._pre_hashes[relative] = digest
        if len(self._pre_hashes) != EXPECTED_FILE_COUNT:
            _raise("ZERO_COVERAGE")
        return {
            "complete_surface_sha256": EXPECTED_COMPLETE_SURFACE_SHA256,
            "protected_surface_sha256": EXPECTED_PROTECTED_SURFACE_SHA256,
        }

    def verify_post_manifest(self) -> dict[str, str]:
        files, directories = self._enumerated_names()
        if files != set(self.files) or directories != set(self.directories):
            _raise("ROOT_MUTATION_DETECTED")
        retained_values: list[tuple[Path, RetainedHandle, bool]] = [
            (self.root, self.root_handle, True)
        ] if self.root_handle is not None else []
        retained_values.extend(
            (
                self.root.joinpath(*PurePosixPath(relative).parts),
                retained,
                True,
            )
            for relative, retained in self.directories.items()
        )
        retained_values.extend(
            (
                self.root.joinpath(*PurePosixPath(relative).parts),
                retained,
                False,
            )
            for relative, retained in self.files.items()
        )
        for path, retained, directory in retained_values:
            current = self._identity(retained.handle, path, directory=directory)
            if current != retained.identity:
                _raise("FILE_IDENTITY_CHANGED")
        if len(self._pre_hashes) != EXPECTED_FILE_COUNT:
            _raise("ZERO_COVERAGE")
        return {
            "complete_surface_sha256": EXPECTED_COMPLETE_SURFACE_SHA256,
            "protected_surface_sha256": EXPECTED_PROTECTED_SURFACE_SHA256,
        }

    def close(self) -> None:
        handles = [
            retained.handle for retained in self.files.values()
        ] + [
            retained.handle for retained in self.directories.values()
        ]
        if self.root_handle is not None:
            handles.append(self.root_handle.handle)
        for handle in reversed(handles):
            try:
                self.api.CloseHandle(handle)
            except Exception:
                pass
        self.files.clear()
        self.directories.clear()
        self.root_handle = None

    def __enter__(self) -> "RetainedProtectedSurface":
        return self.acquire()

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()


class _RetainedSourceLoader:
    def __init__(self, name: str, origin: str, raw: bytes) -> None:
        self.name = name
        self.origin = origin
        self.raw = raw

    def create_module(self, _spec: object) -> None:
        return None

    def exec_module(self, module: types.ModuleType) -> None:
        code = compile(self.raw, self.origin, "exec", dont_inherit=True)
        exec(code, module.__dict__)

    def get_filename(self, _fullname: str) -> str:
        return self.origin


class ExecutingCodeIdentity:
    """Retain and revalidate the exact source bytes that the runner executes."""

    def __init__(
        self,
        plan: Mapping[str, Any],
        candidate: Mapping[str, Any],
        *,
        entry_path: Path,
        repository_argument: Path,
        loaded_modules: Mapping[str, object],
        module_registry: MutableMapping[str, object] | None = None,
        authorized_root: Path = AUTHORIZED_IMPLEMENTATION_REPOSITORY,
        api: object | None = None,
        git_head_reader: Callable[[Path], str] | None = None,
        bootstrap_records: Mapping[str, Mapping[str, object]] | None = None,
    ) -> None:
        self.plan = dict(plan)
        self.candidate = dict(candidate)
        self.registry = sys.modules if module_registry is None else module_registry
        self.loaded_modules: dict[str, object] = dict(loaded_modules)
        self.object_format = str(candidate.get("git_object_format"))
        self.source_commit = str(candidate.get("source_commit"))
        self.git_head_reader = git_head_reader or self._git_head
        self.bootstrap_records = (
            {} if bootstrap_records is None else dict(bootstrap_records)
        )
        self.io = RetainedProtectedSurface(
            Path(entry_path).parent,
            {"files": [], "directories": []},
            api=api,
        )
        self.root: Path | None = None
        self.root_record: RetainedHandle | None = None
        self.sources: dict[str, RetainedHandle] = {}
        self.source_rows: dict[str, dict[str, Any]] = {}
        self.module_name_by_relative: dict[str, str] = {}
        self.loader_ids: set[int] = set()
        self._owned_handles: set[int] = set()
        self._checkpoint_index = 0
        self._closed = False
        contract = plan["executing_code_identity_contract"]
        self.checkpoints = tuple(contract["checkpoints"])
        try:
            self._initialize(
                Path(entry_path), Path(repository_argument), Path(authorized_root)
            )
        except Exception:
            self.close()
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")

    @staticmethod
    def _git_head(root: Path) -> str:
        def read_control(path: Path) -> bytes:
            try:
                metadata = os.lstat(path)
                if getattr(metadata, "st_file_attributes", 0) & 0x400:
                    _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
                raw = path.read_bytes()
            except OSError:
                _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
            if len(raw) > 1 << 20:
                _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
            return raw

        try:
            git_directory = root / ".git"
            git_metadata = os.lstat(git_directory)
            if (
                not git_directory.is_dir()
                or getattr(git_metadata, "st_file_attributes", 0) & 0x400
            ):
                _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
            head_raw = read_control(git_directory / "HEAD")
            if (
                not head_raw.endswith(b"\n")
                or head_raw.count(b"\n") != 1
                or b"\r" in head_raw
            ):
                _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
            head_text = head_raw[:-1].decode("ascii", "strict")
        except (OSError, UnicodeDecodeError):
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        if re.fullmatch(r"[0-9a-f]{40}", head_text) is not None:
            return head_text
        if not head_text.startswith("ref: refs/"):
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        reference = head_text[5:]
        reference_path = PurePosixPath(reference)
        if (
            reference_path.is_absolute()
            or any(part in {"", ".", ".."} for part in reference_path.parts)
            or "\\" in reference
        ):
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        loose_path = git_directory.joinpath(*reference_path.parts)
        if loose_path.exists():
            loose_raw = read_control(loose_path)
            if (
                not loose_raw.endswith(b"\n")
                or loose_raw.count(b"\n") != 1
                or b"\r" in loose_raw
            ):
                _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
            try:
                value = loose_raw[:-1].decode("ascii", "strict")
            except UnicodeDecodeError:
                _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        else:
            packed_raw = read_control(git_directory / "packed-refs")
            try:
                packed_lines = packed_raw.decode("ascii", "strict").splitlines()
            except UnicodeDecodeError:
                _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
            matches = [
                line.split(" ", 1)[0]
                for line in packed_lines
                if " " in line and line.split(" ", 1)[1] == reference
            ]
            if len(matches) != 1:
                _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
            value = matches[0]
        if re.fullmatch(r"[0-9a-f]{40}", value) is None:
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        return value

    def _same_path(self, left: str | Path, right: str | Path) -> bool:
        return windows_ordinal_equal(str(left), str(right), self.io.api)

    def _bootstrap_handle(
        self, expected: Path, *, directory: bool
    ) -> int | None:
        matches = [
            record
            for record in self.bootstrap_records.values()
            if record.get("directory") is directory
            and self._same_path(str(record.get("final_path")), expected)
        ]
        if not matches:
            return None
        if len(matches) != 1 or type(matches[0].get("handle")) is not int:
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        return int(matches[0]["handle"])

    def _candidate_rows(self) -> dict[str, dict[str, Any]]:
        rows = self.candidate.get("implementation_files")
        scientific = self.candidate.get("scientific_dependencies")
        if type(rows) is not list or type(scientific) is not list:
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        combined = [dict(row) for row in rows] + [dict(row) for row in scientific]
        by_path = {str(row.get("relative_path")): row for row in combined}
        if len(by_path) != len(combined):
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        return by_path

    def _relative_for_final_path(self, final_path: str) -> str:
        if self.root is None:
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        matches = [
            relative
            for relative in self.source_rows
            if self._same_path(final_path, self.root / relative)
        ]
        if len(matches) != 1:
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        return matches[0]

    def _open_source(
        self, relative: str, *, existing_handle: int | None = None
    ) -> tuple[RetainedHandle, bytes]:
        if self.root is None:
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        validate_relative_manifest_path(relative)
        expected = self.root / relative
        handle = (
            existing_handle
            if existing_handle is not None
            else (
                self._bootstrap_handle(expected, directory=False)
                or self.io._open(expected, directory=False)
            )
        )
        self._owned_handles.add(handle)
        try:
            identity = self.io._identity(handle, expected, directory=False)
            if (
                self.root_record is None
                or identity.volume_serial
                != self.root_record.identity.volume_serial
            ):
                _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
            retained = RetainedHandle(handle, identity)
            raw = self.io._read_all(retained)
            row = self.source_rows.get(relative)
            if row is None or (
                sha256_bytes(raw) != row.get("file_sha256")
                or git_blob_oid(raw, self.object_format) != row.get("git_blob_oid")
            ):
                _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
            if any(
                previous.identity.file_id == identity.file_id
                and previous.identity.volume_serial == identity.volume_serial
                for name, previous in self.sources.items()
                if name != relative
            ):
                _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
            self.sources[relative] = retained
            return retained, raw
        except Exception:
            if existing_handle is None:
                try:
                    self.io.api.CloseHandle(handle)
                except Exception:
                    pass
                self._owned_handles.discard(handle)
            raise

    def _verify_module(
        self, name: str, module: object, relative: str
    ) -> None:
        if self.root is None:
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        if self.registry.get(name) is not module:
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        module_file = getattr(module, "__file__", None)
        spec = getattr(module, "__spec__", None)
        origin = getattr(spec, "origin", None)
        loader = getattr(spec, "loader", None)
        expected = self.root / relative
        if (
            type(module_file) is not str
            or type(origin) is not str
            or loader is None
            or not self._same_path(module_file, expected)
            or not self._same_path(origin, expected)
            or id(loader) in self.loader_ids
        ):
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        self.loader_ids.add(id(loader))
        self.module_name_by_relative[relative] = name

    def _scan_aliases(self) -> None:
        if self.root is None:
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        for name, module in tuple(self.registry.items()):
            module_file = getattr(module, "__file__", None)
            if type(module_file) is not str:
                continue
            matches = [
                relative
                for relative in self.source_rows
                if self._same_path(module_file, self.root / relative)
            ]
            if not matches:
                continue
            if len(matches) != 1:
                _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
            relative = matches[0]
            expected_name = self.module_name_by_relative.get(relative)
            if expected_name != name or self.loaded_modules.get(name) is not module:
                _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")

    def _initialize(
        self, entry_path: Path, repository_argument: Path, authorized_root: Path
    ) -> None:
        if (
            re.fullmatch(r"[0-9a-f]{40}", self.source_commit) is None
            or self.object_format not in {"sha1", "sha256"}
        ):
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        self.source_rows = self._candidate_rows()
        entry_handle = self._bootstrap_handle(
            entry_path, directory=False
        ) or self.io._open(entry_path, directory=False)
        self._owned_handles.add(entry_handle)
        entry_identity = self.io._identity(
            entry_handle, entry_path, directory=False
        )
        derived_root = Path(entry_identity.final_path).parent.parent
        if (
            not self._same_path(derived_root, authorized_root)
            or not self._same_path(repository_argument, derived_root)
            or self.candidate.get("authorized_implementation_repository")
            != str(AUTHORIZED_IMPLEMENTATION_REPOSITORY)
        ):
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        self.root = derived_root
        root_handle = self._bootstrap_handle(
            self.root, directory=True
        ) or self.io._open(self.root, directory=True)
        self._owned_handles.add(root_handle)
        root_identity = self.io._identity(
            root_handle, self.root, directory=True
        )
        self.root_record = RetainedHandle(root_handle, root_identity)
        entry_relative = self._relative_for_final_path(entry_identity.final_path)
        entry_record, _raw = self._open_source(
            entry_relative, existing_handle=entry_handle
        )
        self.sources[entry_relative] = entry_record
        expected_names = {
            entry_relative: "__main__",
            "tools/gate12c2_original_baseline_commitments.py": (
                "gate12c2_original_baseline_commitments"
            ),
        }
        for name, module in self.loaded_modules.items():
            module_path = getattr(module, "__file__", None)
            if type(module_path) is not str:
                _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
            relative = self._relative_for_final_path(module_path)
            if expected_names.get(relative) != name:
                _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
            if relative not in self.sources:
                self._open_source(relative)
            self._verify_module(name, module, relative)
        if set(self.module_name_by_relative) != set(expected_names):
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        self._scan_aliases()

    def load_scientific_dependencies(self) -> None:
        names = (
            ("gate12c2_synthetic_lab", "tools/gate12c2_synthetic_lab.py"),
            (
                "gate12c2_development_shards",
                "tools/gate12c2_development_shards.py",
            ),
        )
        for name, relative in names:
            if name in self.loaded_modules or name in self.registry:
                _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
            retained, raw = self._open_source(relative)
            loader = _RetainedSourceLoader(name, retained.identity.final_path, raw)
            spec = importlib.util.spec_from_loader(
                name, loader, origin=retained.identity.final_path
            )
            if spec is None:
                _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
            module = importlib.util.module_from_spec(spec)
            module.__file__ = retained.identity.final_path
            self.registry[name] = module
            try:
                loader.exec_module(module)
            except Exception:
                self.registry.pop(name, None)
                _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
            self.loaded_modules[name] = module
            self._verify_module(name, module, relative)
        if getattr(
            self.loaded_modules["gate12c2_development_shards"], "lab", None
        ) is not self.loaded_modules["gate12c2_synthetic_lab"]:
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        self._scan_aliases()

    def module(self, name: str) -> object:
        module = self.loaded_modules.get(name)
        if module is None or self.registry.get(name) is not module:
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        return module

    def checkpoint(self, checkpoint: str) -> dict[str, str]:
        try:
            if (
                self.root is None
                or self.root_record is None
                or self._checkpoint_index >= len(self.checkpoints)
                or checkpoint != self.checkpoints[self._checkpoint_index]
            ):
                _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
            root_identity = self.io._identity(
                self.root_record.handle, self.root, directory=True
            )
            if root_identity != self.root_record.identity:
                _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
            surface_rows: list[dict[str, Any]] = []
            self.loader_ids.clear()
            for relative in sorted(self.sources):
                retained = self.sources[relative]
                identity = self.io._identity(
                    retained.handle, self.root / relative, directory=False
                )
                raw = self.io._read_all(retained)
                row = self.source_rows[relative]
                if (
                    identity != retained.identity
                    or sha256_bytes(raw) != row["file_sha256"]
                    or git_blob_oid(raw, self.object_format) != row["git_blob_oid"]
                ):
                    _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
                module_name = self.module_name_by_relative.get(relative)
                if module_name is None:
                    _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
                self._verify_module(
                    module_name, self.loaded_modules[module_name], relative
                )
                surface_rows.append(
                    {
                        "module_name": module_name,
                        "role": row["role"],
                        "relative_path": relative,
                        "file_id_sha256": sha256_bytes(identity.file_id),
                        "file_sha256": row["file_sha256"],
                        "git_blob_oid": row["git_blob_oid"],
                    }
                )
            self._scan_aliases()
            head = self.git_head_reader(self.root)
            if head != self.source_commit:
                _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
            digest = sha256_bytes(canonical_json_bytes(surface_rows))
            self._checkpoint_index += 1
            return {
                "git_head": head,
                "executing_code_identity_surface_sha256": digest,
            }
        except Exception:
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        handles = list(self._owned_handles)
        handles.extend(record.handle for record in self.sources.values())
        if self.root_record is not None:
            handles.append(self.root_record.handle)
        for handle in dict.fromkeys(handles):
            try:
                self.io.api.CloseHandle(handle)
            except Exception:
                pass
        self._owned_handles.clear()


_ACTIVE_EXECUTING_CODE_IDENTITY: ExecutingCodeIdentity | None = None


def install_executing_code_identity(identity: ExecutingCodeIdentity) -> None:
    global _ACTIVE_EXECUTING_CODE_IDENTITY
    if _ACTIVE_EXECUTING_CODE_IDENTITY is not None:
        _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    _ACTIVE_EXECUTING_CODE_IDENTITY = identity


def clear_executing_code_identity(identity: ExecutingCodeIdentity) -> None:
    global _ACTIVE_EXECUTING_CODE_IDENTITY
    if _ACTIVE_EXECUTING_CODE_IDENTITY is not identity:
        _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    _ACTIVE_EXECUTING_CODE_IDENTITY = None



SHARD_FIELDS = {
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
INDEX_REQUIRED_FIELDS = {
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
INDEX_ROW_REQUIRED_FIELDS = {
    "outer_experiment_index",
    "relative_path",
    "compressed_file_sha256",
    "compressed_bytes",
    "shard_payload_sha256",
    "result_payload_sha256",
    "reused_existing_shard",
    "decision",
}
PIPELINE_DECISION_FIELDS = {
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
}
NON_S2_RESULT_FIELDS = {
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
S2_RESULT_FIELDS = {
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


def strict_gzip_json(raw: bytes) -> dict[str, Any]:
    """Decode exactly one gzip member containing exact canonical JSON."""

    if not isinstance(raw, bytes) or len(raw) < 18:
        _raise("INPUT_SCHEMA_INVALID")
    try:
        decoder = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)
        decompressed = decoder.decompress(raw)
        decompressed += decoder.flush()
    except zlib.error:
        _raise("INPUT_SCHEMA_INVALID")
    if (
        not decoder.eof
        or decoder.unused_data != b""
        or decoder.unconsumed_tail != b""
    ):
        _raise("INPUT_SCHEMA_INVALID")
    return require_mapping(
        strict_json_loads(decompressed, canonical=True),
        code="INPUT_SCHEMA_INVALID",
    )


def validate_s2_component_surface(
    endpoint: Mapping[str, Any],
    *,
    expected_count: int,
) -> None:
    expected = require_int(expected_count, minimum=1)
    medians = require_mapping(endpoint.get("component_medians"))
    coverage = require_mapping(endpoint.get("component_coverage"))
    require_exact_keys(medians, S2_COMPONENT_ARMS)
    require_exact_keys(coverage, S2_COMPONENT_ARMS)
    for arm in S2_COMPONENT_ARMS:
        arm_medians = require_mapping(medians.get(arm))
        arm_coverage = require_mapping(coverage.get(arm))
        require_exact_keys(arm_medians, S2_COMPONENT_FIELDS)
        require_exact_keys(arm_coverage, S2_COMPONENT_FIELDS)
        for field_name in S2_COMPONENT_FIELDS:
            median = arm_medians[field_name]
            if median is not None and (
                type(median) is not float
                or not math.isfinite(median)
                or (
                    median == 0.0
                    and math.copysign(1.0, median) < 0.0
                )
            ):
                _raise("INPUT_SCHEMA_INVALID")
            counts = require_mapping(arm_coverage.get(field_name))
            require_exact_keys(counts, S2_COMPONENT_COUNT_FIELDS)
            checked = {
                name: require_int(counts.get(name))
                for name in S2_COMPONENT_COUNT_FIELDS
            }
            defined = checked["defined_count"]
            degenerate = checked["degenerate_count"]
            if checked["expected_count"] != expected:
                _raise("INPUT_SCHEMA_INVALID")
            if (
                defined
                + degenerate
                + checked["unexpected_missing_count"]
                + checked["nonfinite_count"]
                != expected
                or checked["unexpected_missing_count"] != 0
                or checked["nonfinite_count"] != 0
            ):
                _raise("INPUT_SCHEMA_INVALID")
            if field_name in S2_ALWAYS_DEFINED_COMPONENT_FIELDS:
                if (
                    defined != expected
                    or degenerate != 0
                    or median is None
                ):
                    _raise("INPUT_SCHEMA_INVALID")
            elif defined == 0:
                if degenerate != expected or median is not None:
                    _raise("INPUT_SCHEMA_INVALID")
            elif defined + degenerate != expected or median is None:
                _raise("INPUT_SCHEMA_INVALID")


def validate_result_envelope(
    subplan: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    outer_index: int,
) -> None:
    regime = subplan.get("regime_id")
    expected_fields = (
        S2_RESULT_FIELDS if regime == "S2_null_inflation" else NON_S2_RESULT_FIELDS
    )
    require_exact_keys(result, expected_fields)
    expected_values = {
        "schema_version": OUTER_EXPERIMENT_SCHEMA,
        "contract_version": subplan.get("contract_version"),
        "surface_id": "development",
        "locked_execution_authorized": False,
        "regime_id": regime,
        "outer_experiment_index": outer_index,
        "block_count_schedule": subplan.get("block_count_schedule"),
        "inner_valid_draw_count": subplan.get("inner_valid_draw_count"),
        "diagnostic_kernel": subplan.get("diagnostic_kernel"),
        "accepted_valid_draw_storage": subplan.get(
            "accepted_valid_draw_storage"
        ),
    }
    if any(result.get(key) != value for key, value in expected_values.items()):
        _raise("INPUT_SCHEMA_INVALID")
    require_int(result.get("max_draw_attempts"), minimum=1)
    if not isinstance(result.get("execution_configuration_contract"), dict):
        _raise("INPUT_SCHEMA_INVALID")
    inner_count = require_int(subplan.get("inner_valid_draw_count"), minimum=1)
    configured_attempts = subplan.get("max_draw_attempts")
    resolved_attempts = (
        max(inner_count * 4, inner_count + 8)
        if configured_attempts is None
        else require_int(configured_attempts, minimum=1)
    )
    if result.get("max_draw_attempts") != resolved_attempts:
        _raise("INPUT_SCHEMA_INVALID")
    execution = require_mapping(result["execution_configuration_contract"])
    expected_execution = {
        "schema_version": "gate12c2_result_execution_contract_v0.1",
        "plan_payload_sha256": subplan.get("plan_payload_sha256"),
        "contract_version": subplan.get("contract_version"),
        "surface_id": "development",
        "locked_execution_authorized": False,
        "real_held_out_execution_authorized": False,
        "N2_open": False,
        "N3_open": False,
        "public_claim": False,
        "regime_id": subplan.get("regime_id"),
        "master_seed_sha256": sha256_bytes(
            str(subplan.get("master_seed")).encode("utf-8")
        ),
        "outer_experiment_index": outer_index,
        "block_count_schedule": subplan.get("block_count_schedule"),
        "inner_valid_draw_count": subplan.get("inner_valid_draw_count"),
        "effect_strength": subplan.get("effect_strength"),
        "configured_max_draw_attempts": subplan.get("max_draw_attempts"),
        "resolved_max_draw_attempts": resolved_attempts,
        "minimum_log_null_inflation": subplan.get(
            "minimum_log_null_inflation"
        ),
        "epsilon": subplan.get("epsilon"),
        "diagnostic_kernel": subplan.get("diagnostic_kernel"),
        "accepted_valid_draw_storage": subplan.get(
            "accepted_valid_draw_storage"
        ),
        "outer_experiment_schema": subplan.get("outer_experiment_schema"),
        "seed_namespace_schema": subplan.get("seed_namespace_schema"),
        "scientific_execution_parameters": subplan.get(
            "scientific_execution_parameters"
        ),
        "implementation_sha256": subplan.get("implementation_sha256"),
        "numerical_environment_sha256": sha256_bytes(
            canonical_json_bytes(subplan.get("numerical_environment"))
        ),
    }
    require_exact_keys(execution, expected_execution)
    if execution != expected_execution:
        _raise("INPUT_SCHEMA_INVALID")
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
        "numpy_build": require_mapping(
            subplan.get("numerical_environment")
        ).get("numpy_build"),
        "scientific_execution_parameters": dict(
            require_mapping(subplan.get("scientific_execution_parameters"))
        ),
        "guarantee_scope": (
            "same_frozen_software_and_numerical_environment"
        ),
        "cross_environment_bitwise_determinism_claimed": False,
    }
    if not isinstance(numerical, dict) or numerical != expected_numerical:
        _raise("INPUT_SCHEMA_INVALID")
    if result.get("effect_strength") != subplan.get("effect_strength"):
        _raise("INPUT_SCHEMA_INVALID")
    if regime == "S2_null_inflation":
        if (
            result.get("observed_process_modified") is not False
            or result.get("paired_null_arms")
            != [N1_NULL_ARM_ID, S2_NULL_ARM_ID]
        ):
            _raise("INPUT_SCHEMA_INVALID")
        require_bool(result.get("breadth_pass"))
        require_bool(result.get("identification_success"))
        require_int(result.get("identified_case_count"), maximum=12)
        if (
            not isinstance(result.get("endpoint_rows"), list)
            or len(result["endpoint_rows"]) != 24
            or not isinstance(result.get("case_rows"), list)
            or len(result["case_rows"]) != 12
        ):
            _raise("INPUT_SCHEMA_INVALID")
        threshold = float(subplan.get("minimum_log_null_inflation"))
        for endpoint_value in result["endpoint_rows"]:
            endpoint = require_mapping(endpoint_value)
            if endpoint.get("minimum_log_null_inflation") != threshold:
                _raise("INPUT_SCHEMA_INVALID")
            expected_blocks = require_int(
                endpoint.get("expected_block_count"), minimum=1
            )
            validate_s2_component_surface(
                endpoint, expected_count=expected_blocks * inner_count
            )
    else:
        pipeline = require_mapping(result.get("pipeline_decision"))
        require_exact_keys(pipeline, PIPELINE_DECISION_FIELDS)
        for key in (
            "claim_promotion",
            "any_endpoint_support",
            "any_run_support",
            "partial_or_structured_is_promotional",
        ):
            require_bool(pipeline.get(key))
        for key in (
            "endpoint_count",
            "q_directional_support_count",
            "supporting_run_count",
            "q_discordant_run_count",
        ):
            require_int(pipeline.get(key), maximum=24)
        require_text(pipeline.get("grid_outcome"), ascii_only=True)


def reconstruct_decision(result: Mapping[str, Any]) -> dict[str, Any]:
    if result.get("regime_id") == "S2_null_inflation":
        return {
            "identification_success": require_bool(
                result.get("identification_success")
            ),
            "identified_case_count": require_int(
                result.get("identified_case_count"), maximum=12
            ),
            "breadth_pass": require_bool(result.get("breadth_pass")),
        }
    pipeline = require_mapping(result.get("pipeline_decision"))
    return {
        "claim_promotion": require_bool(pipeline.get("claim_promotion")),
        "grid_outcome": require_text(
            pipeline.get("grid_outcome"), ascii_only=True
        ),
        "any_endpoint_support": require_bool(
            pipeline.get("any_endpoint_support")
        ),
        "any_run_support": require_bool(pipeline.get("any_run_support")),
        "q_directional_support_count": require_int(
            pipeline.get("q_directional_support_count"), maximum=24
        ),
        "supporting_run_count": require_int(
            pipeline.get("supporting_run_count"), maximum=12
        ),
        "q_discordant_run_count": require_int(
            pipeline.get("q_discordant_run_count"), maximum=12
        ),
    }


def scientific_projection(
    subplan: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SCIENTIFIC_PROJECTION_SCHEMA,
        "plan_payload_sha256": subplan["plan_payload_sha256"],
        "outer_results": [
            {
                "outer_experiment_index": row["outer_experiment_index"],
                "result_payload_sha256": row["result_payload_sha256"],
                "decision": dict(row["decision"]),
            }
            for row in sorted(
                rows, key=lambda item: item["outer_experiment_index"]
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


def semantic_index_commitment(
    configuration_id: str,
    subplan_payload_sha256: str,
    outer_experiment_count: int,
    outer_id_surface_sha256: str,
    result_commitment_surface_sha256: str,
    scientific_projection_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": SEMANTIC_INDEX_SCHEMA,
        "configuration_id": configuration_id,
        "original_subplan_payload_sha256": subplan_payload_sha256,
        "outer_experiment_count": outer_experiment_count,
        "outer_id_surface_sha256": outer_id_surface_sha256,
        "result_commitment_surface_sha256": (
            result_commitment_surface_sha256
        ),
        "scientific_projection_sha256": scientific_projection_sha256,
    }


def _index_row_matches(
    indexed: Mapping[str, Any], rebuilt: Mapping[str, Any]
) -> bool:
    allowed = set(INDEX_ROW_REQUIRED_FIELDS) | {"operational_metrics"}
    if set(indexed) not in (
        set(INDEX_ROW_REQUIRED_FIELDS),
        allowed,
    ):
        return False
    if type(indexed.get("reused_existing_shard")) is not bool:
        return False
    if "operational_metrics" in indexed:
        if not isinstance(indexed["operational_metrics"], dict):
            return False
        _validate_json_domain(indexed["operational_metrics"])
    return all(indexed.get(key) == value for key, value in rebuilt.items())


def derive_configuration_commitment(
    *,
    configuration_id: str,
    subplan: Mapping[str, Any],
    index_raw: bytes,
    shard_raw_by_relative_path: Mapping[str, bytes],
    result_validator: Callable[[Mapping[str, Any], Mapping[str, Any], int], None],
    phase_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Derive one public commitment row from retained-handle bytes."""

    require_identifier(configuration_id)
    index = require_mapping(strict_json_loads(index_raw, canonical=True))
    if set(index) not in (
        set(INDEX_REQUIRED_FIELDS),
        set(INDEX_REQUIRED_FIELDS) | {"operational_execution_metrics"},
    ):
        _raise("INPUT_SCHEMA_INVALID")
    if index.get("schema_version") != INDEX_SCHEMA:
        _raise("INPUT_SCHEMA_INVALID")
    if (
        index.get("surface_id") != "development"
        or index.get("locked_execution_authorized") is not False
        or index.get("epistemic_status") != "development_shard_index_only"
        or index.get("merge_order") != "ascending_outer_experiment_index"
        or index.get("scientific_projection_schema_version")
        != SCIENTIFIC_PROJECTION_SCHEMA
    ):
        _raise("INPUT_SCHEMA_INVALID")
    if index.get("plan_payload_sha256") != subplan.get("plan_payload_sha256"):
        _raise("INPUT_SCHEMA_INVALID")
    require_int(index.get("worker_count_operational_only"), minimum=1)
    require_bool(index.get("all_outer_indices_present"))
    if index["all_outer_indices_present"] is not True:
        _raise("ZERO_COVERAGE")
    if "operational_execution_metrics" in index:
        if not isinstance(index["operational_execution_metrics"], dict):
            _raise("INPUT_SCHEMA_INVALID")
        _validate_json_domain(index["operational_execution_metrics"])
    verify_self_hash(index, "index_payload_sha256")
    expected_ids_value = subplan.get("outer_experiment_indices")
    if not isinstance(expected_ids_value, list) or not expected_ids_value:
        _raise("ZERO_COVERAGE")
    expected_ids = [
        require_int(value, maximum=(1 << 31) - 1)
        for value in expected_ids_value
    ]
    if expected_ids != sorted(set(expected_ids)):
        _raise("INPUT_SCHEMA_INVALID")
    indexed_rows = index.get("shards")
    if (
        not isinstance(indexed_rows, list)
        or len(indexed_rows) != len(expected_ids)
        or index.get("outer_experiment_count") != len(expected_ids)
    ):
        _raise("ZERO_COVERAGE")
    rebuilt_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    supplied_paths = set(shard_raw_by_relative_path)
    expected_paths = {
        f"shards/outer-{outer_index:06d}.json.gz"
        for outer_index in expected_ids
    }
    if supplied_paths != expected_paths:
        _raise("ZERO_COVERAGE")
    for position, outer_index in enumerate(expected_ids):
        relative = f"shards/outer-{outer_index:06d}.json.gz"
        compressed = shard_raw_by_relative_path[relative]
        shard = strict_gzip_json(compressed)
        require_exact_keys(shard, SHARD_FIELDS)
        if (
            shard.get("schema_version") != SHARD_SCHEMA
            or shard.get("epistemic_status") != "development_outer_shard_only"
            or shard.get("surface_id") != "development"
            or shard.get("locked_execution_authorized") is not False
            or shard.get("plan_payload_sha256")
            != subplan.get("plan_payload_sha256")
            or shard.get("outer_experiment_index") != outer_index
        ):
            _raise("INPUT_SCHEMA_INVALID")
        verify_self_hash(shard, "shard_payload_sha256")
        result = require_mapping(shard.get("result"))
        result_digest = sha256_bytes(canonical_json_bytes(result))
        if shard.get("result_payload_sha256") != result_digest:
            _raise("INPUT_SCHEMA_INVALID")
        validate_result_envelope(
            subplan, result, outer_index=outer_index
        )
        result_validator(subplan, result, outer_index)
        decision = reconstruct_decision(result)
        rebuilt = {
            "outer_experiment_index": outer_index,
            "relative_path": relative,
            "compressed_file_sha256": sha256_bytes(compressed),
            "compressed_bytes": len(compressed),
            "shard_payload_sha256": shard["shard_payload_sha256"],
            "result_payload_sha256": result_digest,
            "decision": decision,
        }
        indexed = require_mapping(indexed_rows[position])
        if not _index_row_matches(indexed, rebuilt):
            _raise("INPUT_SCHEMA_INVALID")
        rebuilt_rows.append(rebuilt)
        result_rows.append(
            {
                "outer_experiment_index": outer_index,
                "result_payload_sha256": result_digest,
                "shard_payload_sha256": shard["shard_payload_sha256"],
            }
        )
    if phase_callback is not None:
        phase_callback("commitment_derivation")
    projection = scientific_projection(subplan, rebuilt_rows)
    projection_digest = sha256_bytes(canonical_json_bytes(projection))
    if index.get("scientific_projection_sha256") != projection_digest:
        _raise("COMMITMENT_MISMATCH")
    outer_digest = sha256_bytes(canonical_json_bytes(expected_ids))
    result_digest = sha256_bytes(canonical_json_bytes(result_rows))
    semantic = semantic_index_commitment(
        configuration_id,
        str(subplan["plan_payload_sha256"]),
        len(expected_ids),
        outer_digest,
        result_digest,
        projection_digest,
    )
    return {
        "configuration_id": configuration_id,
        "outer_experiment_count": len(expected_ids),
        "outer_id_surface_sha256": outer_digest,
        "result_commitment_surface_sha256": result_digest,
        "scientific_projection_sha256": projection_digest,
        "semantic_index_commitment_v0_1_sha256": sha256_bytes(
            canonical_json_bytes(semantic)
        ),
    }



def new_extraction_progress() -> dict[str, Any]:
    """Return the in-memory phase/evidence state for handled failures."""

    return {
        "source_state": "EXTRACTION_EXECUTION_CLAIMED",
        "failure_phase": "lineage_reverification",
        "evidence": {
            "pre_complete_surface_sha256": None,
            "pre_protected_surface_sha256": None,
            "post_complete_surface_sha256": None,
            "post_protected_surface_sha256": None,
            "baseline_commitment_surface_sha256": None,
        },
        "configuration_count_reached": 0,
        "outer_experiment_count_reached": 0,
        "shard_count_reached": 0,
        "index_count_reached": 0,
    }


def update_extraction_progress(
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
    """Advance non-durable execution state without exposing payload values."""

    if progress is None:
        return
    if not progress:
        progress.update(new_extraction_progress())
    if source_state is not None:
        progress["source_state"] = source_state
    if failure_phase is not None:
        progress["failure_phase"] = failure_phase
    if evidence is not None:
        current = progress.get("evidence")
        if not isinstance(current, dict):
            _raise("INTERNAL_SANITIZED_FAILURE")
        current.update(evidence)
    for key, value in (
        ("configuration_count_reached", configuration_count_reached),
        ("outer_experiment_count_reached", outer_experiment_count_reached),
        ("shard_count_reached", shard_count_reached),
        ("index_count_reached", index_count_reached),
    ):
        if value is not None:
            progress[key] = value


def derive_baseline_from_surface(
    plan: Mapping[str, Any],
    original_plan: Mapping[str, Any],
    surface: RetainedProtectedSurface,
    *,
    result_validator: Callable[
        [Mapping[str, Any], Mapping[str, Any], int], None
    ],
    progress: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive all nine rows while retaining every protected handle."""

    root_plan_raw = surface.read("plan.json")
    if root_plan_raw != canonical_json_bytes(original_plan):
        _raise("INPUT_LINEAGE_MISMATCH")
    configurations_value = original_plan.get("configurations")
    if not isinstance(configurations_value, list) or len(configurations_value) != 9:
        _raise("INPUT_SCHEMA_INVALID")
    original_by_id: dict[str, dict[str, Any]] = {}
    for value in configurations_value:
        configuration = require_mapping(value)
        identifier = require_identifier(configuration.get("configuration_id"))
        if identifier in original_by_id:
            _raise("INPUT_SCHEMA_INVALID")
        original_by_id[identifier] = configuration
    commitment_rows: list[dict[str, Any]] = []
    outer_total = shard_total = index_total = 0
    for frozen_value in plan["configuration_surface"]:
        update_extraction_progress(
            progress,
            source_state="EXTRACTION_PRE_MANIFEST_VERIFIED",
            failure_phase="semantic_verification",
        )
        frozen = require_mapping(frozen_value)
        identifier = str(frozen["configuration_id"])
        configuration = original_by_id.get(identifier)
        if configuration is None:
            _raise("INPUT_LINEAGE_MISMATCH")
        if (
            configuration.get("output_relative_path")
            != frozen["output_relative_path"]
            or configuration.get("draw_count") != frozen["draw_count"]
            or configuration.get("regime_id") != frozen["regime_id"]
        ):
            _raise("INPUT_LINEAGE_MISMATCH")
        subplan = require_mapping(configuration.get("subplan"))
        if (
            subplan.get("plan_payload_sha256")
            != frozen["original_subplan_payload_sha256"]
        ):
            _raise("INPUT_LINEAGE_MISMATCH")
        prefix = str(frozen["output_relative_path"])
        plan_relative = f"{prefix}/plan.json"
        if surface.read(plan_relative) != canonical_json_bytes(subplan):
            _raise("INPUT_LINEAGE_MISMATCH")
        index_relative = f"{prefix}/index.json"
        expected_ids = subplan.get("outer_experiment_indices")
        if not isinstance(expected_ids, list):
            _raise("INPUT_SCHEMA_INVALID")
        shard_bytes = {
            f"shards/outer-{require_int(outer_index):06d}.json.gz": (
                surface.read(
                    f"{prefix}/shards/outer-{require_int(outer_index):06d}.json.gz"
                )
            )
            for outer_index in expected_ids
        }
        row = derive_configuration_commitment(
            configuration_id=identifier,
            subplan=subplan,
            index_raw=surface.read(index_relative),
            shard_raw_by_relative_path=shard_bytes,
            result_validator=result_validator,
            phase_callback=lambda phase: update_extraction_progress(
                progress, failure_phase=phase
            ),
        )
        if row["outer_experiment_count"] != frozen["outer_experiment_count"]:
            _raise("ZERO_COVERAGE")
        commitment_rows.append(row)
        outer_total += row["outer_experiment_count"]
        shard_total += len(shard_bytes)
        index_total += 1
        update_extraction_progress(
            progress,
            configuration_count_reached=len(commitment_rows),
            outer_experiment_count_reached=outer_total,
            shard_count_reached=shard_total,
            index_count_reached=index_total,
        )
    update_extraction_progress(
        progress, failure_phase="commitment_derivation"
    )
    commitment_rows.sort(key=lambda row: row["configuration_id"])
    if (
        len(commitment_rows),
        outer_total,
        shard_total,
        index_total,
    ) != (9, 768, 768, 9):
        _raise("ZERO_COVERAGE")
    return {
        "configuration_count": 9,
        "outer_experiment_count": 768,
        "shard_count": 768,
        "index_count": 9,
        "configuration_surface_sha256": CONFIGURATION_SURFACE_SHA256,
        "configuration_commitments": commitment_rows,
        "baseline_commitment_surface_sha256": sha256_bytes(
            canonical_json_bytes(commitment_rows)
        ),
    }


def extract_commitments_after_claim(
    plan: Mapping[str, Any],
    lineage: Mapping[str, Mapping[str, Any]],
    *,
    progress: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The sole extractor entry point that opens protected bytes."""

    update_extraction_progress(
        progress,
        source_state="EXTRACTION_EXECUTION_CLAIMED",
        failure_phase="lineage_reverification",
    )
    manifest = validate_incident_manifest(plan, lineage["incident_manifest"])
    original_plan = lineage["original_plan"]
    identity = _ACTIVE_EXECUTING_CODE_IDENTITY
    if identity is None:
        _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    scientific_sources = {
        "gate12c2_synthetic_lab": "gate12c2_synthetic_lab.py",
        "gate12c2_development_shards": "gate12c2_development_shards.py",
    }
    frozen_lab = identity.module(next(iter(scientific_sources)))
    frozen_shards = identity.module("gate12c2_development_shards")
    if getattr(frozen_shards, "lab", None) is not frozen_lab:
        _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")

    def validate_with_frozen_source(
        subplan: Mapping[str, Any],
        result: Mapping[str, Any],
        outer_index: int,
    ) -> None:
        try:
            verified = frozen_shards._verified_plan(subplan)
            frozen_shards._verify_result_against_plan(
                verified,
                result,
                outer_experiment_index=outer_index,
            )
        except Exception:
            _raise("INPUT_SCHEMA_INVALID")

    update_extraction_progress(
        progress,
        source_state="EXTRACTION_INPUT_LOCKING",
        failure_phase="input_locking",
    )
    with RetainedProtectedSurface(PROTECTED_ROOT, manifest) as surface:
        update_extraction_progress(
            progress,
            source_state="EXTRACTION_INPUT_HANDLES_LOCKED",
            failure_phase="pre_manifest",
        )
        pre = surface.verify_pre_manifest()
        update_extraction_progress(
            progress,
            source_state="EXTRACTION_PRE_MANIFEST_VERIFIED",
            failure_phase="semantic_verification",
            evidence={
                "pre_complete_surface_sha256": pre[
                    "complete_surface_sha256"
                ],
                "pre_protected_surface_sha256": pre[
                    "protected_surface_sha256"
                ],
            },
        )
        derived = derive_baseline_from_surface(
            plan,
            original_plan,
            surface,
            result_validator=validate_with_frozen_source,
            progress=progress,
        )
        update_extraction_progress(
            progress,
            source_state="EXTRACTION_COMMITMENTS_COMPUTED_QUARANTINED",
            failure_phase="post_manifest",
            evidence={
                "baseline_commitment_surface_sha256": derived[
                    "baseline_commitment_surface_sha256"
                ]
            },
        )
        post = surface.verify_post_manifest()
    update_extraction_progress(
        progress,
        source_state="EXTRACTION_POST_MANIFEST_VERIFIED",
        failure_phase="terminal_outcome_reconstruction",
        evidence={
            "post_complete_surface_sha256": post["complete_surface_sha256"],
            "post_protected_surface_sha256": post[
                "protected_surface_sha256"
            ],
        },
    )
    return {
        **derived,
        "pre_complete_surface_sha256": pre["complete_surface_sha256"],
        "pre_protected_surface_sha256": pre["protected_surface_sha256"],
        "post_complete_surface_sha256": post["complete_surface_sha256"],
        "post_protected_surface_sha256": post["protected_surface_sha256"],
        "partial_or_temp_count": 0,
        "unexpected_file_count": 0,
        "file_reparse_point_count": 0,
        "directory_reparse_point_count": 0,
    }


def build_extraction_success_leaf(
    plan: Mapping[str, Any],
    derived: Mapping[str, Any],
    *,
    reviewed_authority_file_sha256: str,
    reviewed_authority_payload_sha256: str,
    preflight_file_sha256: str,
    preflight_payload_sha256: str,
    authorization_file_sha256: str,
    authorization_payload_sha256: str,
    authorization_verdict_file_sha256: str,
    authorization_verdict_payload_sha256: str,
    execution_claim_file_sha256: str,
    execution_claim_payload_sha256: str,
    implementation_source_commit: str,
    git_head_at_protected_read: str,
    git_head_at_terminal: str,
    executing_code_identity_surface_sha256: str,
) -> dict[str, Any]:
    if (
        any(
            re.fullmatch(r"[0-9a-f]{40}", value) is None
            for value in (
                implementation_source_commit,
                git_head_at_protected_read,
                git_head_at_terminal,
            )
        )
        or git_head_at_protected_read != implementation_source_commit
        or git_head_at_terminal != implementation_source_commit
        or not is_sha256(executing_code_identity_surface_sha256)
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    schema = plan["success_receipt"]
    payload = {
        "schema_version": schema["schema_version"],
        "gate_id": GATE_ID,
        "state": schema["state"],
        "verification_status": schema["verification_status"],
        "sequence_ordinal": schema["sequence_ordinal"],
        "original_resource_gate_status": "indeterminate_permanent",
        "replacement_resource_qualification": "not_performed",
        "scientific_values_emitted": False,
        "stability_analysis_authorized": False,
        "locked_execution_authorized": False,
        "real_held_out_execution_authorized": False,
        "N2_open": False,
        "N3_open": False,
        "configuration_count": derived["configuration_count"],
        "outer_experiment_count": derived["outer_experiment_count"],
        "shard_count": derived["shard_count"],
        "index_count": derived["index_count"],
        "partial_or_temp_count": derived["partial_or_temp_count"],
        "unexpected_file_count": derived["unexpected_file_count"],
        "file_reparse_point_count": derived["file_reparse_point_count"],
        "directory_reparse_point_count": derived[
            "directory_reparse_point_count"
        ],
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
        "artifact_path_surface_sha256": artifact_surface_sha256(plan),
        "implementation_source_commit": implementation_source_commit,
        "git_head_at_protected_read": git_head_at_protected_read,
        "git_head_at_terminal": git_head_at_terminal,
        "executing_code_identity_surface_sha256": (
            executing_code_identity_surface_sha256
        ),
        "reviewed_implementation_authority_file_sha256": (
            reviewed_authority_file_sha256
        ),
        "reviewed_implementation_authority_payload_sha256": (
            reviewed_authority_payload_sha256
        ),
        "extraction_preflight_file_sha256": preflight_file_sha256,
        "extraction_preflight_payload_sha256": preflight_payload_sha256,
        "extraction_authorization_file_sha256": authorization_file_sha256,
        "extraction_authorization_payload_sha256": authorization_payload_sha256,
        "extraction_authorization_verdict_file_sha256": (
            authorization_verdict_file_sha256
        ),
        "extraction_authorization_verdict_payload_sha256": (
            authorization_verdict_payload_sha256
        ),
        "extraction_execution_claim_file_sha256": execution_claim_file_sha256,
        "extraction_execution_claim_payload_sha256": (
            execution_claim_payload_sha256
        ),
        "pre_complete_surface_sha256": derived[
            "pre_complete_surface_sha256"
        ],
        "post_complete_surface_sha256": derived[
            "post_complete_surface_sha256"
        ],
        "pre_protected_surface_sha256": derived[
            "pre_protected_surface_sha256"
        ],
        "post_protected_surface_sha256": derived[
            "post_protected_surface_sha256"
        ],
        "configuration_surface_sha256": derived[
            "configuration_surface_sha256"
        ],
        "configuration_commitments": derived["configuration_commitments"],
        "baseline_commitment_surface_sha256": derived[
            "baseline_commitment_surface_sha256"
        ],
    }
    require_exact_keys(
        {**payload, "baseline_receipt_payload_sha256": ""},
        schema["exact_top_level_fields"],
    )
    return add_self_hash(payload, "baseline_receipt_payload_sha256")

def validate_extraction_success_leaf(
    plan: Mapping[str, Any],
    success: Mapping[str, Any],
    *,
    reviewed_authority_file_sha256: str,
    reviewed_authority_payload_sha256: str,
    preflight_file_sha256: str,
    preflight_payload_sha256: str,
    authorization_file_sha256: str,
    authorization_payload_sha256: str,
    authorization_verdict_file_sha256: str,
    authorization_verdict_payload_sha256: str,
    execution_claim_file_sha256: str,
    execution_claim_payload_sha256: str,
    implementation_source_commit: str,
    git_head_at_protected_read: str,
    git_head_at_terminal: str,
    executing_code_identity_surface_sha256: str,
) -> dict[str, Any]:
    schema = plan["success_receipt"]
    supplied = dict(success)
    require_exact_keys(
        supplied, schema["exact_top_level_fields"], code="INPUT_LINEAGE_MISMATCH"
    )
    verify_self_hash(
        supplied, "baseline_receipt_payload_sha256", code="INPUT_LINEAGE_MISMATCH"
    )
    linked_digests = (
        reviewed_authority_file_sha256,
        reviewed_authority_payload_sha256,
        preflight_file_sha256,
        preflight_payload_sha256,
        authorization_file_sha256,
        authorization_payload_sha256,
        authorization_verdict_file_sha256,
        authorization_verdict_payload_sha256,
        execution_claim_file_sha256,
        execution_claim_payload_sha256,
        executing_code_identity_surface_sha256,
    )
    if any(not is_sha256(value) for value in linked_digests):
        _raise("INPUT_LINEAGE_MISMATCH")
    fixed_surface = {
        "configuration_count": 9,
        "outer_experiment_count": 768,
        "shard_count": 768,
        "index_count": 9,
        "partial_or_temp_count": 0,
        "unexpected_file_count": 0,
        "file_reparse_point_count": 0,
        "directory_reparse_point_count": 0,
        "pre_complete_surface_sha256": EXPECTED_COMPLETE_SURFACE_SHA256,
        "post_complete_surface_sha256": EXPECTED_COMPLETE_SURFACE_SHA256,
        "pre_protected_surface_sha256": EXPECTED_PROTECTED_SURFACE_SHA256,
        "post_protected_surface_sha256": EXPECTED_PROTECTED_SURFACE_SHA256,
        "configuration_surface_sha256": CONFIGURATION_SURFACE_SHA256,
    }
    if any(supplied.get(key) != value for key, value in fixed_surface.items()):
        _raise("INPUT_LINEAGE_MISMATCH")
    commitments = supplied.get("configuration_commitments")
    commitment_fields = {
        "configuration_id",
        "outer_experiment_count",
        "outer_id_surface_sha256",
        "result_commitment_surface_sha256",
        "scientific_projection_sha256",
        "semantic_index_commitment_v0_1_sha256",
    }
    if (
        not isinstance(commitments, list)
        or len(commitments) != 9
        or commitments
        != sorted(commitments, key=lambda row: row.get("configuration_id", ""))
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    frozen_by_id = {
        row["configuration_id"]: row for row in plan["configuration_surface"]
    }
    seen: set[str] = set()
    for value in commitments:
        row = require_mapping(value, code="INPUT_LINEAGE_MISMATCH")
        require_exact_keys(row, commitment_fields, code="INPUT_LINEAGE_MISMATCH")
        identifier = row.get("configuration_id")
        if identifier in seen or identifier not in frozen_by_id:
            _raise("INPUT_LINEAGE_MISMATCH")
        seen.add(str(identifier))
        if row.get("outer_experiment_count") != frozen_by_id[identifier][
            "outer_experiment_count"
        ]:
            _raise("INPUT_LINEAGE_MISMATCH")
        for field in commitment_fields - {
            "configuration_id",
            "outer_experiment_count",
        }:
            if not is_sha256(row.get(field)):
                _raise("INPUT_LINEAGE_MISMATCH")
    if (
        seen != set(frozen_by_id)
        or sha256_bytes(canonical_json_bytes(commitments))
        != supplied.get("baseline_commitment_surface_sha256")
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    derived = {
        key: supplied[key]
        for key in (
            *fixed_surface,
            "configuration_commitments",
            "baseline_commitment_surface_sha256",
        )
    }
    expected = build_extraction_success_leaf(
        plan,
        derived,
        reviewed_authority_file_sha256=reviewed_authority_file_sha256,
        reviewed_authority_payload_sha256=reviewed_authority_payload_sha256,
        preflight_file_sha256=preflight_file_sha256,
        preflight_payload_sha256=preflight_payload_sha256,
        authorization_file_sha256=authorization_file_sha256,
        authorization_payload_sha256=authorization_payload_sha256,
        authorization_verdict_file_sha256=authorization_verdict_file_sha256,
        authorization_verdict_payload_sha256=(
            authorization_verdict_payload_sha256
        ),
        execution_claim_file_sha256=execution_claim_file_sha256,
        execution_claim_payload_sha256=execution_claim_payload_sha256,
        implementation_source_commit=implementation_source_commit,
        git_head_at_protected_read=git_head_at_protected_read,
        git_head_at_terminal=git_head_at_terminal,
        executing_code_identity_surface_sha256=(
            executing_code_identity_surface_sha256
        ),
    )
    if supplied != expected:
        _raise("INPUT_LINEAGE_MISMATCH")
    return supplied





def git_commit_parent_lineage(
    repository: Path,
    source_commit: str,
    *,
    code: str = "IMPLEMENTATION_BYTE_IDENTITY_MISMATCH",
) -> tuple[str, ...]:
    if (
        type(source_commit) is not str
        or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
    ):
        _raise(code)
    try:
        completed = subprocess.run(
            [
                "git",
                "rev-list",
                "--parents",
                "-n",
                "1",
                source_commit,
            ],
            cwd=Path(repository),
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="ascii",
        )
    except (OSError, subprocess.SubprocessError):
        _raise(code)
    output = completed.stdout
    if (
        not output.endswith("\n")
        or output.count("\n") != 1
        or "\r" in output
    ):
        _raise(code)
    lineage = tuple(output[:-1].split(" "))
    if any(re.fullmatch(r"[0-9a-f]{40}", value) is None for value in lineage):
        _raise(code)
    return lineage



def git_path_blob_oid(
    repository: Path,
    source_commit: str,
    relative_path: str,
    *,
    code: str = "IMPLEMENTATION_BYTE_IDENTITY_MISMATCH",
) -> str:
    relative = validate_relative_manifest_path(relative_path)
    try:
        completed = subprocess.run(
            ["git", "rev-parse", f"{source_commit}:{relative}"],
            cwd=Path(repository),
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="ascii",
        )
    except (OSError, subprocess.SubprocessError):
        _raise(code)
    output = completed.stdout
    if (
        not output.endswith("\n")
        or output.count("\n") != 1
        or re.fullmatch(r"[0-9a-f]{40,64}", output[:-1]) is None
    ):
        _raise(code)
    return output[:-1]

def require_direct_child_lineage(
    source_commit: str,
    remediation_base_commit: str,
    lineage: Sequence[str],
    *,
    code: str = "IMPLEMENTATION_BYTE_IDENTITY_MISMATCH",
) -> None:
    if tuple(lineage) != (source_commit, remediation_base_commit):
        _raise(code)


def read_schema_receipt(
    path: Path,
    *,
    exact_fields: Sequence[str],
    hash_field: str,
    code: str = "INPUT_LINEAGE_MISMATCH",
) -> tuple[dict[str, Any], str]:
    try:
        raw = Path(path).read_bytes()
    except OSError:
        _raise(code)
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        _raise(code)
    payload = require_mapping(strict_json_loads(raw[:-1], canonical=True), code=code)
    require_exact_keys(payload, exact_fields, code=code)
    verify_self_hash(payload, hash_field, code=code)
    if raw != canonical_receipt_bytes(payload):
        _raise(code)
    return payload, sha256_bytes(raw)


def r2_activation_control(
    plan: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    value = plan.get("r2_activation_control")
    if value is None:
        return None
    return require_mapping(value, code="INPUT_LINEAGE_MISMATCH")


def validate_r2_candidate_manifest(
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    control = r2_activation_control(plan)
    if control is None:
        _raise("INPUT_LINEAGE_MISMATCH")
    contract = require_mapping(
        control.get("candidate_manifest_contract"),
        code="INPUT_LINEAGE_MISMATCH",
    )
    supplied = dict(manifest)
    require_exact_keys(
        supplied,
        contract["exact_top_level_fields"],
        code="IMPLEMENTATION_BYTE_IDENTITY_MISMATCH",
    )
    verify_self_hash(
        supplied,
        "candidate_manifest_payload_sha256",
        code="IMPLEMENTATION_BYTE_IDENTITY_MISMATCH",
    )
    binding = plan["implementation_binding_contract"]
    required = binding["required_values"]
    expected = {
        "schema_version": contract["schema_version"],
        "authority_namespace_id": R2_AUTHORITY_NAMESPACE_ID,
        "state": contract["state"],
        "activation_source_commit": candidate.get("source_commit"),
        "activation_parent_commit": required["remediation_base_commit"],
        "task1_parent_commit": required["remediation_base_parent"],
        "r2_activation_plan_file_sha256": (
            R2_ACTIVATION_PLAN_FILE_SHA256
        ),
        "r2_activation_plan_payload_sha256": (
            R2_ACTIVATION_PLAN_PAYLOAD_SHA256
        ),
        "artifact_path_surface_sha256": artifact_surface_sha256(plan),
        "occupied_v0_9_surface_sha256": (
            R2_OCCUPIED_V0_9_SURFACE_SHA256
        ),
        "review_surface_identity_sha256": (
            REVIEW_SURFACE_IDENTITY_SHA256
        ),
        "implementation_trust_model_sha256": (
            recompute_implementation_trust_model_sha256(plan)
        ),
        "implementation_files": candidate.get("implementation_files"),
        "scientific_dependencies": candidate.get("scientific_dependencies"),
        "clean_restore": candidate.get("clean_restore"),
        "protected_payload_accessed": False,
        "scientific_values_inspected": False,
        "runtime_authorization_issued": False,
    }
    if any(supplied.get(key) != value for key, value in expected.items()):
        _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    return supplied


def validate_r2_clean_restore_receipt(
    plan: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    control = r2_activation_control(plan)
    if control is None:
        _raise("INPUT_LINEAGE_MISMATCH")
    contract = require_mapping(
        control.get("clean_restore_receipt_contract"),
        code="INPUT_LINEAGE_MISMATCH",
    )
    supplied = dict(receipt)
    require_exact_keys(
        supplied,
        contract["exact_top_level_fields"],
        code="IMPLEMENTATION_BYTE_IDENTITY_MISMATCH",
    )
    verify_self_hash(
        supplied,
        "restore_receipt_payload_sha256",
        code="IMPLEMENTATION_BYTE_IDENTITY_MISMATCH",
    )
    restore = require_mapping(
        candidate.get("clean_restore"),
        code="IMPLEMENTATION_BYTE_IDENTITY_MISMATCH",
    )
    binding_required = plan["implementation_binding_contract"][
        "required_values"
    ]
    expected = {
        "schema_version": contract["schema_version"],
        "authority_namespace_id": R2_AUTHORITY_NAMESPACE_ID,
        "state": contract["state"],
        "source_commit": candidate.get("source_commit"),
        "source_parent_commit": binding_required[
            "remediation_base_commit"
        ],
        "task1_parent_commit": binding_required[
            "remediation_base_parent"
        ],
        "bundle_path": restore.get("bundle_path"),
        "bundle_file_sha256": restore.get("bundle_file_sha256"),
        "bundle_size_bytes": restore.get("bundle_size_bytes"),
        "restore_head": candidate.get("source_commit"),
        "restore_worktree_clean": True,
        "git_fsck_full_pass": True,
        "core_autocrlf": False,
        "core_longpaths": True,
        "implementation_rows_match": True,
        "scientific_dependency_rows_match": True,
        "targeted_tests_passed": True,
        "full_suite_passed": True,
        "protected_payload_accessed": False,
        "scientific_values_inspected": False,
        "runtime_artifacts_created": 0,
    }
    if any(supplied.get(key) != value for key, value in expected.items()):
        _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    require_text(
        supplied.get("restore_path"),
        code="IMPLEMENTATION_BYTE_IDENTITY_MISMATCH",
    )
    return supplied


def read_r2_clean_restore_receipt(
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    control = r2_activation_control(plan)
    if control is None:
        _raise("INPUT_LINEAGE_MISMATCH")
    contract = control["clean_restore_receipt_contract"]
    return read_schema_receipt(
        Path(contract["artifact_path"]),
        exact_fields=contract["exact_top_level_fields"],
        hash_field="restore_receipt_payload_sha256",
    )

def validate_r2_fresh_review_evidence(
    plan: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    candidate_file_sha256: str,
) -> dict[str, Any]:
    control = r2_activation_control(plan)
    if control is None:
        _raise("INPUT_LINEAGE_MISMATCH")
    contract = require_mapping(
        control.get("fresh_review_evidence_contract"),
        code="INPUT_LINEAGE_MISMATCH",
    )
    supplied = dict(evidence)
    require_exact_keys(
        supplied,
        contract["exact_top_level_fields"],
        code="INPUT_LINEAGE_MISMATCH",
    )
    verify_self_hash(
        supplied,
        "review_evidence_payload_sha256",
        code="INPUT_LINEAGE_MISMATCH",
    )
    if any(
        supplied.get(key) != value
        for key, value in contract["required_values"].items()
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    restore = require_mapping(
        candidate.get("clean_restore"),
        code="INPUT_LINEAGE_MISMATCH",
    )
    expected = {
        "implementation_source_commit": candidate.get("source_commit"),
        "implementation_candidate_binding_file_sha256": (
            candidate_file_sha256
        ),
        "implementation_candidate_binding_payload_sha256": candidate.get(
            "implementation_candidate_binding_payload_sha256"
        ),
        "candidate_manifest_file_sha256": candidate.get(
            "candidate_manifest_file_sha256"
        ),
        "candidate_manifest_payload_sha256": candidate.get(
            "candidate_manifest_payload_sha256"
        ),
        "r2_activation_plan_file_sha256": (
            R2_ACTIVATION_PLAN_FILE_SHA256
        ),
        "r2_activation_plan_payload_sha256": (
            R2_ACTIVATION_PLAN_PAYLOAD_SHA256
        ),
        "artifact_path_surface_sha256": artifact_surface_sha256(plan),
        "review_surface_identity": review_surface_identity(plan),
        "bundle_file_sha256": restore.get("bundle_file_sha256"),
        "restore_receipt_file_sha256": restore.get(
            "restore_receipt_file_sha256"
        ),
        "restore_receipt_payload_sha256": restore.get(
            "restore_receipt_payload_sha256"
        ),
    }
    if any(supplied.get(key) != value for key, value in expected.items()):
        _raise("INPUT_LINEAGE_MISMATCH")
    for field in (
        "changed_file_manifest_sha256",
        "targeted_test_node_id_sha256",
        "full_suite_node_id_sha256",
    ):
        if not is_sha256(supplied.get(field)):
            _raise("INPUT_LINEAGE_MISMATCH")
    require_int(
        supplied.get("targeted_test_count"),
        minimum=1,
        code="INPUT_LINEAGE_MISMATCH",
    )
    require_int(
        supplied.get("full_suite_test_count"),
        minimum=1,
        code="INPUT_LINEAGE_MISMATCH",
    )
    return supplied


def read_r2_candidate_manifest(
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    control = r2_activation_control(plan)
    if control is None:
        _raise("INPUT_LINEAGE_MISMATCH")
    contract = control["candidate_manifest_contract"]
    return read_schema_receipt(
        Path(contract["artifact_path"]),
        exact_fields=contract["exact_top_level_fields"],
        hash_field="candidate_manifest_payload_sha256",
    )


def read_r2_fresh_review_evidence(
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    control = r2_activation_control(plan)
    if control is None:
        _raise("INPUT_LINEAGE_MISMATCH")
    contract = control["fresh_review_evidence_contract"]
    return read_schema_receipt(
        Path(contract["artifact_path"]),
        exact_fields=contract["exact_top_level_fields"],
        hash_field="review_evidence_payload_sha256",
    )

def r2r1_remediation_control(
    plan: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    value = plan.get("r2r1_remediation_control")
    if value is None:
        return None
    return require_mapping(value, code="INPUT_LINEAGE_MISMATCH")


def r2r2_portability_control(
    plan: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    value = plan.get("r2r2_portability_control")
    if value is None:
        return None
    return require_mapping(value, code="INPUT_LINEAGE_MISMATCH")


def active_remediation_control(
    plan: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    r2r2 = r2r2_portability_control(plan)
    return r2r2 if r2r2 is not None else r2r1_remediation_control(plan)


def active_remediation_identity(
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    control = active_remediation_control(plan)
    if control is None:
        _raise("INPUT_LINEAGE_MISMATCH")
    lineage = require_mapping(
        control.get("parent_lineage"), code="INPUT_LINEAGE_MISMATCH"
    )
    identity = {
        "authority_namespace_id": control["authority_namespace_id"],
        "parent_commit": lineage["remediation_parent"],
        "grandparent_commit": lineage["remediation_grandparent"],
        "static_fields": {
            "r2r1_remediation_plan_file_sha256": (
                R2R1_REMEDIATION_PLAN_FILE_SHA256
            ),
            "r2r1_remediation_plan_payload_sha256": (
                R2R1_REMEDIATION_PLAN_PAYLOAD_SHA256
            ),
            "occupied_r2_surface_sha256": (
                R2R1_OCCUPIED_R2_SURFACE_SHA256
            ),
        },
    }
    if r2r2_portability_control(plan) is not None:
        identity["static_fields"].update(
            {
                "r2r2_portability_plan_file_sha256": (
                    R2R2_PORTABILITY_PLAN_FILE_SHA256
                ),
                "r2r2_portability_plan_payload_sha256": (
                    R2R2_PORTABILITY_PLAN_PAYLOAD_SHA256
                ),
                "occupied_r2r1_surface_sha256": (
                    R2R2_OCCUPIED_R2R1_SURFACE_SHA256
                ),
                "repository_local_artifact_surface_sha256": (
                    R2R2_REPOSITORY_LOCAL_SURFACE_SHA256
                ),
                "upstream_json_framing_surface_sha256": (
                    R2R2_UPSTREAM_FRAMING_SURFACE_SHA256
                ),
            }
        )
    return identity

def _git_object_format(
    repository: Path,
    *,
    code: str = "IMPLEMENTATION_BYTE_IDENTITY_MISMATCH",
) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--show-object-format"],
            cwd=Path(repository),
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="ascii",
        )
    except (OSError, subprocess.SubprocessError):
        _raise(code)
    value = completed.stdout.strip()
    if value not in {"sha1", "sha256"}:
        _raise(code)
    return value


def r2r1_changed_file_manifest(
    plan: Mapping[str, Any],
    repository: Path,
    source_commit: str,
) -> tuple[list[dict[str, str]], str]:
    control = active_remediation_control(plan)
    if control is None:
        _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    allowed = list(control["allowed_changed_paths"])
    try:
        completed = subprocess.run(
            [
                "git",
                "diff-tree",
                "--no-commit-id",
                "--name-only",
                "-r",
                source_commit,
            ],
            cwd=Path(repository),
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.SubprocessError):
        _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    if "\r" in completed.stdout:
        _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    paths = sorted(
        line for line in completed.stdout.split("\n") if line
    )
    if paths != allowed or len(paths) != len(set(paths)):
        _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    object_format = _git_object_format(repository)
    rows: list[dict[str, str]] = []
    for relative in paths:
        relative = validate_relative_manifest_path(relative)
        try:
            raw = (Path(repository) / relative).read_bytes()
        except OSError:
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        blob = git_path_blob_oid(repository, source_commit, relative)
        if git_blob_oid(raw, object_format) != blob:
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        rows.append(
            {
                "file_sha256": sha256_bytes(raw),
                "git_blob_oid": blob,
                "relative_path": relative,
            }
        )
    digest = sha256_bytes(canonical_json_bytes(rows))
    return rows, digest


def validate_r2r1_clean_restore_receipt(
    plan: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    control = active_remediation_control(plan)
    if control is None:
        _raise("INPUT_LINEAGE_MISMATCH")
    contract = control["clean_restore_receipt_contract"]
    supplied = dict(receipt)
    require_exact_keys(
        supplied,
        contract["exact_top_level_fields"],
        code="IMPLEMENTATION_BYTE_IDENTITY_MISMATCH",
    )
    verify_self_hash(
        supplied,
        "restore_receipt_payload_sha256",
        code="IMPLEMENTATION_BYTE_IDENTITY_MISMATCH",
    )
    if any(
        supplied.get(key) != value
        for key, value in contract["required_values"].items()
    ):
        _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    identity = active_remediation_identity(plan)
    source_commit = supplied.get("source_commit")
    if (
        not isinstance(source_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
        or supplied.get("source_parent_commit")
        != identity["parent_commit"]
        or supplied.get("source_grandparent_commit")
        != identity["grandparent_commit"]
        or supplied.get("restore_head") != source_commit
    ):
        _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    for field, expected in identity["static_fields"].items():
        if (
            field in contract["exact_top_level_fields"]
            and supplied.get(field) != expected
        ):
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    coverage = control["review_coverage_identity"]
    for field in (
        "targeted_test_count",
        "targeted_test_node_id_sha256",
        "full_suite_test_count",
        "full_suite_test_node_id_sha256",
    ):
        if supplied.get(field) != coverage[field]:
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    size = require_int(
        supplied.get("bundle_size_bytes"),
        minimum=1,
        code="IMPLEMENTATION_BYTE_IDENTITY_MISMATCH",
    )
    bundle_path = Path(
        require_text(
            supplied.get("bundle_path"),
            code="IMPLEMENTATION_BYTE_IDENTITY_MISMATCH",
        )
    )
    try:
        bundle_raw = bundle_path.read_bytes()
    except OSError:
        _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    if (
        len(bundle_raw) != size
        or sha256_bytes(bundle_raw) != supplied.get("bundle_file_sha256")
    ):
        _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    require_text(
        supplied.get("restore_path"),
        code="IMPLEMENTATION_BYTE_IDENTITY_MISMATCH",
    )
    return supplied


def read_r2r1_clean_restore_receipt(
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    control = active_remediation_control(plan)
    if control is None:
        _raise("INPUT_LINEAGE_MISMATCH")
    contract = control["clean_restore_receipt_contract"]
    receipt, file_hash = read_schema_receipt(
        Path(contract["artifact_path"]),
        exact_fields=contract["exact_top_level_fields"],
        hash_field="restore_receipt_payload_sha256",
    )
    return validate_r2r1_clean_restore_receipt(plan, receipt), file_hash


def validate_r2r1_candidate_selection(
    plan: Mapping[str, Any],
    selection: Mapping[str, Any],
    *,
    repo_root: Path,
    current_head: str | None = None,
) -> dict[str, Any]:
    control = active_remediation_control(plan)
    if control is None:
        _raise("INPUT_LINEAGE_MISMATCH")
    contract = control["candidate_selection_contract"]
    supplied = dict(selection)
    require_exact_keys(
        supplied,
        contract["exact_top_level_fields"],
        code="IMPLEMENTATION_BYTE_IDENTITY_MISMATCH",
    )
    verify_self_hash(
        supplied,
        "candidate_selection_payload_sha256",
        code="IMPLEMENTATION_BYTE_IDENTITY_MISMATCH",
    )
    if any(
        supplied.get(key) != value
        for key, value in contract["required_values"].items()
    ):
        _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    identity = active_remediation_identity(plan)
    source_commit = supplied.get("exact_candidate_commit")
    parent_commit = identity["parent_commit"]
    grandparent_commit = identity["grandparent_commit"]
    if (
        not isinstance(source_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
        or (current_head is not None and source_commit != current_head)
        or supplied.get("exact_parent_commit") != parent_commit
        or supplied.get("exact_grandparent_commit") != grandparent_commit
        or supplied.get("commit_parent_count") != 1
        or supplied.get("parent_parent_count") != 1
    ):
        _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    require_direct_child_lineage(
        source_commit,
        parent_commit,
        git_commit_parent_lineage(repo_root, source_commit),
    )
    require_direct_child_lineage(
        parent_commit,
        grandparent_commit,
        git_commit_parent_lineage(repo_root, parent_commit),
    )
    rows, digest = r2r1_changed_file_manifest(
        plan, repo_root, source_commit
    )
    if (
        supplied.get("changed_path_allowlist")
        != control["allowed_changed_paths"]
        or supplied.get("changed_files") != rows
        or supplied.get("changed_file_manifest_sha256") != digest
    ):
        _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    expected_static = {
        "r2_activation_plan_file_sha256": R2_ACTIVATION_PLAN_FILE_SHA256,
        "r2_activation_plan_payload_sha256": (
            R2_ACTIVATION_PLAN_PAYLOAD_SHA256
        ),
        "artifact_path_surface_sha256": artifact_surface_sha256(plan),
        **identity["static_fields"],
        "review_surface_identity_sha256": REVIEW_SURFACE_IDENTITY_SHA256,
        "implementation_trust_model_sha256": (
            recompute_implementation_trust_model_sha256(plan)
        ),
    }
    coverage = control["review_coverage_identity"]
    for field in (
        "targeted_test_count",
        "targeted_test_node_id_sha256",
        "full_suite_test_count",
        "full_suite_test_node_id_sha256",
    ):
        expected_static[field] = coverage[field]
    if any(
        supplied.get(key) != value for key, value in expected_static.items()
    ):
        _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    receipt, receipt_file_hash = read_r2r1_clean_restore_receipt(plan)
    if (
        supplied.get("bundle_path") != receipt.get("bundle_path")
        or supplied.get("bundle_file_sha256")
        != receipt.get("bundle_file_sha256")
        or supplied.get("bundle_size_bytes")
        != receipt.get("bundle_size_bytes")
        or supplied.get("clean_restore_receipt_file_sha256")
        != receipt_file_hash
        or supplied.get("clean_restore_receipt_payload_sha256")
        != receipt.get("restore_receipt_payload_sha256")
        or receipt.get("source_commit") != source_commit
    ):
        _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    implementation = supplied.get("implementation_files")
    if (
        not isinstance(implementation, list)
        or len(implementation) != 10
        or implementation
        != sorted(implementation, key=lambda row: row.get("role", ""))
    ):
        _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    seen_paths: set[str] = set()
    for value in implementation:
        row = require_mapping(
            value, code="IMPLEMENTATION_BYTE_IDENTITY_MISMATCH"
        )
        require_exact_keys(
            row,
            plan["implementation_binding_contract"][
                "exact_implementation_row_fields"
            ],
            code="IMPLEMENTATION_BYTE_IDENTITY_MISMATCH",
        )
        relative = validate_relative_manifest_path(row.get("relative_path"))
        role = row.get("role")
        if (
            relative in seen_paths
            or IMPLEMENTATION_ROLE_BY_PATH.get(relative) != role
        ):
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        seen_paths.add(relative)
        try:
            raw = (Path(repo_root) / relative).read_bytes()
        except OSError:
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        if (
            sha256_bytes(raw) != row.get("file_sha256")
            or git_path_blob_oid(repo_root, source_commit, relative)
            != row.get("git_blob_oid")
        ):
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    if seen_paths != set(IMPLEMENTATION_PATHS):
        _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    return supplied


def read_r2r1_candidate_selection(
    plan: Mapping[str, Any],
    *,
    repo_root: Path,
    current_head: str | None = None,
) -> tuple[dict[str, Any], str]:
    control = active_remediation_control(plan)
    if control is None:
        _raise("INPUT_LINEAGE_MISMATCH")
    contract = control["candidate_selection_contract"]
    selection, file_hash = read_schema_receipt(
        Path(contract["artifact_path"]),
        exact_fields=contract["exact_top_level_fields"],
        hash_field="candidate_selection_payload_sha256",
    )
    return (
        validate_r2r1_candidate_selection(
            plan,
            selection,
            repo_root=repo_root,
            current_head=current_head,
        ),
        file_hash,
    )


def validate_r2r1_candidate_manifest(
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    selection: Mapping[str, Any],
    selection_file_sha256: str,
) -> dict[str, Any]:
    control = active_remediation_control(plan)
    if control is None:
        _raise("INPUT_LINEAGE_MISMATCH")
    contract = control["candidate_manifest_contract"]
    identity = active_remediation_identity(plan)
    supplied = dict(manifest)
    require_exact_keys(
        supplied,
        contract["exact_top_level_fields"],
        code="IMPLEMENTATION_BYTE_IDENTITY_MISMATCH",
    )
    verify_self_hash(
        supplied,
        "candidate_manifest_payload_sha256",
        code="IMPLEMENTATION_BYTE_IDENTITY_MISMATCH",
    )
    expected = {
        "schema_version": contract["schema_version"],
        "authority_namespace_id": identity["authority_namespace_id"],
        "state": contract["state"],
        "activation_source_commit": selection["exact_candidate_commit"],
        "activation_parent_commit": identity["parent_commit"],
        "task1_parent_commit": identity["grandparent_commit"],
        "r2_activation_plan_file_sha256": R2_ACTIVATION_PLAN_FILE_SHA256,
        "r2_activation_plan_payload_sha256": (
            R2_ACTIVATION_PLAN_PAYLOAD_SHA256
        ),
        "artifact_path_surface_sha256": artifact_surface_sha256(plan),
        **identity["static_fields"],
        "candidate_selection_file_sha256": selection_file_sha256,
        "candidate_selection_payload_sha256": selection[
            "candidate_selection_payload_sha256"
        ],
        "review_surface_identity_sha256": REVIEW_SURFACE_IDENTITY_SHA256,
        "implementation_trust_model_sha256": (
            recompute_implementation_trust_model_sha256(plan)
        ),
        "implementation_files": candidate.get("implementation_files"),
        "scientific_dependencies": candidate.get("scientific_dependencies"),
        "clean_restore": candidate.get("clean_restore"),
        "protected_payload_accessed": False,
        "scientific_values_inspected": False,
        "runtime_authorization_issued": False,
    }
    if any(supplied.get(key) != value for key, value in expected.items()):
        _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    return supplied


def read_r2r1_candidate_manifest(
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    control = active_remediation_control(plan)
    if control is None:
        _raise("INPUT_LINEAGE_MISMATCH")
    contract = control["candidate_manifest_contract"]
    return read_schema_receipt(
        Path(contract["artifact_path"]),
        exact_fields=contract["exact_top_level_fields"],
        hash_field="candidate_manifest_payload_sha256",
    )


def validate_r2r1_review_input_freeze(
    plan: Mapping[str, Any],
    freeze: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    candidate_file_sha256: str,
    selection: Mapping[str, Any],
    selection_file_sha256: str,
) -> dict[str, Any]:
    control = active_remediation_control(plan)
    if control is None:
        _raise("INPUT_LINEAGE_MISMATCH")
    contract = control["review_input_freeze_contract"]
    identity = active_remediation_identity(plan)
    supplied = dict(freeze)
    require_exact_keys(
        supplied,
        contract["exact_top_level_fields"],
        code="INPUT_LINEAGE_MISMATCH",
    )
    verify_self_hash(
        supplied,
        "review_input_freeze_payload_sha256",
        code="INPUT_LINEAGE_MISMATCH",
    )
    if any(
        supplied.get(key) != value
        for key, value in contract["required_values"].items()
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    packet_path = Path(control["fresh_review_packet_path"])
    try:
        packet_raw = packet_path.read_bytes()
    except OSError:
        _raise("INPUT_LINEAGE_MISMATCH")
    manifest, manifest_file_hash = read_r2r1_candidate_manifest(plan)
    restore, restore_file_hash = read_r2r1_clean_restore_receipt(plan)
    coverage = control["review_coverage_identity"]
    expected = {
        "implementation_source_commit": candidate["source_commit"],
        "candidate_selection_file_sha256": selection_file_sha256,
        "candidate_selection_payload_sha256": selection[
            "candidate_selection_payload_sha256"
        ],
        "candidate_manifest_file_sha256": manifest_file_hash,
        "candidate_manifest_payload_sha256": manifest[
            "candidate_manifest_payload_sha256"
        ],
        "implementation_candidate_binding_file_sha256": (
            candidate_file_sha256
        ),
        "implementation_candidate_binding_payload_sha256": candidate[
            "implementation_candidate_binding_payload_sha256"
        ],
        "clean_restore_receipt_file_sha256": restore_file_hash,
        "clean_restore_receipt_payload_sha256": restore[
            "restore_receipt_payload_sha256"
        ],
        "artifact_path_surface_sha256": artifact_surface_sha256(plan),
        **identity["static_fields"],
        "review_packet_path": str(packet_path),
        "review_packet_file_sha256": sha256_bytes(packet_raw),
        "review_packet_size_bytes": len(packet_raw),
        "changed_file_manifest_sha256": selection[
            "changed_file_manifest_sha256"
        ],
        "targeted_test_count": coverage["targeted_test_count"],
        "targeted_test_node_id_sha256": coverage[
            "targeted_test_node_id_sha256"
        ],
        "full_suite_test_count": coverage["full_suite_test_count"],
        "full_suite_test_node_id_sha256": coverage[
            "full_suite_test_node_id_sha256"
        ],
    }
    if any(supplied.get(key) != value for key, value in expected.items()):
        _raise("INPUT_LINEAGE_MISMATCH")
    return supplied


def read_r2r1_review_input_freeze(
    plan: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    candidate_file_sha256: str,
    selection: Mapping[str, Any],
    selection_file_sha256: str,
) -> tuple[dict[str, Any], str]:
    control = active_remediation_control(plan)
    if control is None:
        _raise("INPUT_LINEAGE_MISMATCH")
    contract = control["review_input_freeze_contract"]
    freeze, file_hash = read_schema_receipt(
        Path(contract["artifact_path"]),
        exact_fields=contract["exact_top_level_fields"],
        hash_field="review_input_freeze_payload_sha256",
    )
    return (
        validate_r2r1_review_input_freeze(
            plan,
            freeze,
            candidate=candidate,
            candidate_file_sha256=candidate_file_sha256,
            selection=selection,
            selection_file_sha256=selection_file_sha256,
        ),
        file_hash,
    )


def validate_r2r1_fresh_review_evidence(
    plan: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    candidate_file_sha256: str,
    selection: Mapping[str, Any],
    selection_file_sha256: str,
    freeze: Mapping[str, Any] | None = None,
    freeze_file_sha256: str | None = None,
) -> dict[str, Any]:
    control = active_remediation_control(plan)
    if control is None:
        _raise("INPUT_LINEAGE_MISMATCH")
    contract = control["fresh_review_evidence_contract"]
    identity = active_remediation_identity(plan)
    supplied = dict(evidence)
    require_exact_keys(
        supplied,
        contract["exact_top_level_fields"],
        code="INPUT_LINEAGE_MISMATCH",
    )
    verify_self_hash(
        supplied,
        "review_evidence_payload_sha256",
        code="INPUT_LINEAGE_MISMATCH",
    )
    if any(
        supplied.get(key) != value
        for key, value in contract["required_values"].items()
    ):
        _raise("INPUT_LINEAGE_MISMATCH")
    if freeze is None:
        frozen, frozen_file_hash = read_r2r1_review_input_freeze(
            plan,
            candidate=candidate,
            candidate_file_sha256=candidate_file_sha256,
            selection=selection,
            selection_file_sha256=selection_file_sha256,
        )
    else:
        frozen = validate_r2r1_review_input_freeze(
            plan,
            freeze,
            candidate=candidate,
            candidate_file_sha256=candidate_file_sha256,
            selection=selection,
            selection_file_sha256=selection_file_sha256,
        )
        frozen_file_hash = (
            freeze_file_sha256
            if freeze_file_sha256 is not None
            else sha256_bytes(canonical_receipt_bytes(frozen))
        )
    expected = {
        "implementation_source_commit": candidate["source_commit"],
        "implementation_candidate_binding_file_sha256": (
            candidate_file_sha256
        ),
        "implementation_candidate_binding_payload_sha256": candidate[
            "implementation_candidate_binding_payload_sha256"
        ],
        "candidate_manifest_file_sha256": candidate[
            "candidate_manifest_file_sha256"
        ],
        "candidate_manifest_payload_sha256": candidate[
            "candidate_manifest_payload_sha256"
        ],
        "candidate_selection_file_sha256": selection_file_sha256,
        "candidate_selection_payload_sha256": selection[
            "candidate_selection_payload_sha256"
        ],
        "review_input_freeze_file_sha256": frozen_file_hash,
        "review_input_freeze_payload_sha256": frozen[
            "review_input_freeze_payload_sha256"
        ],
        "artifact_path_surface_sha256": artifact_surface_sha256(plan),
        **identity["static_fields"],
        "review_surface_identity": review_surface_identity(plan),
        "implementation_review_packet_file_sha256": frozen[
            "review_packet_file_sha256"
        ],
        "changed_file_manifest_sha256": frozen[
            "changed_file_manifest_sha256"
        ],
        "targeted_test_count": frozen["targeted_test_count"],
        "targeted_test_node_id_sha256": frozen[
            "targeted_test_node_id_sha256"
        ],
        "full_suite_test_count": frozen["full_suite_test_count"],
        "full_suite_test_node_id_sha256": frozen[
            "full_suite_test_node_id_sha256"
        ],
    }
    if any(supplied.get(key) != value for key, value in expected.items()):
        _raise("INPUT_LINEAGE_MISMATCH")
    return supplied


def read_r2r1_fresh_review_evidence(
    plan: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    candidate_file_sha256: str,
    selection: Mapping[str, Any],
    selection_file_sha256: str,
) -> tuple[dict[str, Any], str]:
    control = active_remediation_control(plan)
    if control is None:
        _raise("INPUT_LINEAGE_MISMATCH")
    contract = control["fresh_review_evidence_contract"]
    evidence, file_hash = read_schema_receipt(
        Path(contract["artifact_path"]),
        exact_fields=contract["exact_top_level_fields"],
        hash_field="review_evidence_payload_sha256",
    )
    return (
        validate_r2r1_fresh_review_evidence(
            plan,
            evidence,
            candidate=candidate,
            candidate_file_sha256=candidate_file_sha256,
            selection=selection,
            selection_file_sha256=selection_file_sha256,
        ),
        file_hash,
    )


def publish_r2r1_control_receipt(
    plan: Mapping[str, Any],
    contract_name: str,
    payload: Mapping[str, Any],
) -> PublicationResult:
    control = active_remediation_control(plan)
    allowed = {
        "candidate_manifest_contract",
        "candidate_selection_contract",
        "clean_restore_receipt_contract",
        "fresh_review_evidence_contract",
        "review_input_freeze_contract",
    }
    if control is None or contract_name not in allowed:
        _raise("UNEXPECTED_ARTIFACT")
    contract = require_mapping(
        control.get(contract_name), code="UNEXPECTED_ARTIFACT"
    )
    exact_fields = contract["exact_top_level_fields"]
    require_exact_keys(payload, exact_fields, code="INPUT_LINEAGE_MISMATCH")
    verify_self_hash(
        payload, exact_fields[-1], code="INPUT_LINEAGE_MISMATCH"
    )
    return atomic_publish_exact(
        Path(contract["artifact_path"]),
        canonical_receipt_bytes(payload),
        pending_path=Path(contract["pending_path"]),
    )


def validate_candidate_binding(
    plan: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    repo_root: Path,
    current_head: str | None = None,
    candidate_manifest: Mapping[str, Any] | None = None,
    candidate_manifest_file_sha256: str | None = None,
    candidate_selection: Mapping[str, Any] | None = None,
    candidate_selection_file_sha256: str | None = None,
) -> dict[str, Any]:
    contract = plan["implementation_binding_contract"]
    supplied = dict(candidate)
    require_exact_keys(
        supplied,
        artifact_exact_fields(plan, "implementation_candidate_binding"),
        code="IMPLEMENTATION_BYTE_IDENTITY_MISMATCH",
    )
    verify_self_hash(
        supplied,
        "implementation_candidate_binding_payload_sha256",
        code="IMPLEMENTATION_BYTE_IDENTITY_MISMATCH",
    )
    validate_review_surface_identity(
        plan,
        supplied.get("review_surface_identity"),
        code="IMPLEMENTATION_BYTE_IDENTITY_MISMATCH",
    )
    if any(
        supplied.get(key) != value
        for key, value in contract["required_values"].items()
    ):
        _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    expected_values = {
        "contract_file_sha256": CONTRACT_FILE_SHA256,
        "plan_file_sha256": PLAN_FILE_SHA256,
        "plan_payload_sha256": PLAN_PAYLOAD_SHA256,
        "formal_design_review_file_sha256": FORMAL_DESIGN_REVIEW_FILE_SHA256,
        "formal_design_review_payload_sha256": (
            FORMAL_DESIGN_REVIEW_PAYLOAD_SHA256
        ),
        "implementation_author_separation_contract_sha256": (
            IMPLEMENTATION_AUTHOR_SEPARATION_SHA256
        ),
        "implementation_trust_model_sha256": (
            recompute_implementation_trust_model_sha256(plan)
        ),
        "artifact_path_surface_sha256": artifact_surface_sha256(plan),
        "authorized_implementation_repository": str(
            AUTHORIZED_IMPLEMENTATION_REPOSITORY
        ),
        "remediation_base_commit": contract["required_values"][
            "remediation_base_commit"
        ],
        "remediation_base_parent": contract["required_values"]["remediation_base_parent"],
        "worktree_clean": True,
        "core_autocrlf": False,
        "core_longpaths": True,
    }
    control = r2_activation_control(plan)
    remediation_control = active_remediation_control(plan)
    if control is not None:
        expected_values.update(
            {
                "r2_activation_plan_file_sha256": (
                    R2_ACTIVATION_PLAN_FILE_SHA256
                ),
                "r2_activation_plan_payload_sha256": (
                    R2_ACTIVATION_PLAN_PAYLOAD_SHA256
                ),
                "occupied_v0_9_surface_sha256": (
                    R2_OCCUPIED_V0_9_SURFACE_SHA256
                ),
            }
        )
    if remediation_control is not None:
        expected_values.update(
            active_remediation_identity(plan)["static_fields"]
        )
    if any(supplied.get(key) != value for key, value in expected_values.items()):
        _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    source_commit = supplied.get("source_commit")
    object_format = supplied.get("git_object_format")
    if (
        not isinstance(source_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
        or object_format not in {"sha1", "sha256"}
    ):
        _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    if current_head is not None and source_commit != current_head:
        _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    selection: dict[str, Any] | None = None
    selection_file_hash: str | None = None
    if remediation_control is not None:
        if candidate_selection is None:
            selection, selection_file_hash = read_r2r1_candidate_selection(
                plan,
                repo_root=repo_root,
                current_head=source_commit,
            )
        else:
            selection = validate_r2r1_candidate_selection(
                plan,
                candidate_selection,
                repo_root=repo_root,
                current_head=source_commit,
            )
            selection_file_hash = (
                candidate_selection_file_sha256
                if candidate_selection_file_sha256 is not None
                else sha256_bytes(canonical_receipt_bytes(selection))
            )
        if (
            source_commit != selection.get("exact_candidate_commit")
            or supplied.get("candidate_selection_file_sha256")
            != selection_file_hash
            or supplied.get("candidate_selection_payload_sha256")
            != selection.get("candidate_selection_payload_sha256")
        ):
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    if remediation_control is not None:
        remediation_identity = active_remediation_identity(plan)
        expected_base = remediation_identity["parent_commit"]
        expected_base_parent = remediation_identity["grandparent_commit"]
    else:
        expected_base = contract["required_values"]["remediation_base_commit"]
        expected_base_parent = contract["required_values"][
            "remediation_base_parent"
        ]
    require_direct_child_lineage(
        source_commit,
        expected_base,
        git_commit_parent_lineage(repo_root, source_commit),
    )
    if control is not None:
        require_direct_child_lineage(
            expected_base,
            expected_base_parent,
            git_commit_parent_lineage(repo_root, expected_base),
        )
        activation_relative = (
            "tools/gate12c2_original_baseline_r2_activation_plan.json"
        )
        try:
            activation_raw = (
                Path(repo_root) / activation_relative
            ).read_bytes()
        except OSError:
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        if (
            sha256_bytes(activation_raw)
            != R2_ACTIVATION_PLAN_FILE_SHA256
            or git_path_blob_oid(
                repo_root,
                source_commit,
                activation_relative,
            )
            != git_blob_oid(activation_raw, str(object_format))
        ):
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    if remediation_control is not None:
        remediation_relative = (
            "tools/gate12c2_original_baseline_r2r1_remediation_plan.json"
        )
        try:
            remediation_raw = (
                Path(repo_root) / remediation_relative
            ).read_bytes()
        except OSError:
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        if (
            sha256_bytes(remediation_raw)
            != R2R1_REMEDIATION_PLAN_FILE_SHA256
            or git_path_blob_oid(
                repo_root,
                source_commit,
                remediation_relative,
            )
            != git_blob_oid(remediation_raw, str(object_format))
        ):
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    if r2r2_portability_control(plan) is not None:
        portability_relative = R2R2_PORTABILITY_PLAN_RELATIVE_PATH
        try:
            portability_raw = (
                Path(repo_root) / portability_relative
            ).read_bytes()
        except OSError:
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        if (
            sha256_bytes(portability_raw)
            != R2R2_PORTABILITY_PLAN_FILE_SHA256
            or git_path_blob_oid(
                repo_root,
                source_commit,
                portability_relative,
            )
            != git_blob_oid(portability_raw, str(object_format))
        ):
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    rows = supplied.get("implementation_files")
    if not isinstance(rows, list) or len(rows) != 10:
        _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    if rows != sorted(rows, key=lambda row: row.get("role", "")):
        _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    seen_roles: set[str] = set()
    seen_paths: set[str] = set()
    for value in rows:
        row = require_mapping(
            value, code="IMPLEMENTATION_BYTE_IDENTITY_MISMATCH"
        )
        require_exact_keys(
            row,
            contract["exact_implementation_row_fields"],
            code="IMPLEMENTATION_BYTE_IDENTITY_MISMATCH",
        )
        role = require_text(
            row["role"],
            ascii_only=True,
            code="IMPLEMENTATION_BYTE_IDENTITY_MISMATCH",
        )
        relative = validate_relative_manifest_path(row["relative_path"])
        if (
            role in seen_roles
            or relative in seen_paths
            or IMPLEMENTATION_ROLE_BY_PATH.get(relative) != role
        ):
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        seen_roles.add(role)
        seen_paths.add(relative)
        if not is_sha256(row["file_sha256"]) or not is_git_oid(
            row["git_blob_oid"]
        ):
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        try:
            raw = (Path(repo_root) / relative).read_bytes()
        except OSError:
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        if (
            sha256_bytes(raw) != row["file_sha256"]
            or git_blob_oid(raw, str(object_format)) != row["git_blob_oid"]
        ):
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    if seen_paths != set(IMPLEMENTATION_PATHS):
        _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    scientific = supplied.get("scientific_dependencies")
    if scientific != contract["scientific_dependencies"]:
        _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    for row in scientific:
        relative = validate_relative_manifest_path(row["relative_path"])
        try:
            raw = (Path(repo_root) / relative).read_bytes()
        except OSError:
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        if (
            sha256_bytes(raw) != row["file_sha256"]
            or git_blob_oid(raw, str(object_format)) != row["git_blob_oid"]
        ):
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    restore = require_mapping(
        supplied.get("clean_restore"),
        code="IMPLEMENTATION_BYTE_IDENTITY_MISMATCH",
    )
    require_exact_keys(
        restore,
        contract["clean_restore_exact_fields"],
        code="IMPLEMENTATION_BYTE_IDENTITY_MISMATCH",
    )
    if any(
        restore.get(key) != value
        for key, value in contract["clean_restore_required_values"].items()
    ):
        _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    for field in (
        "bundle_file_sha256",
        "restore_receipt_file_sha256",
        "restore_receipt_payload_sha256",
    ):
        if not is_sha256(restore.get(field)):
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    bundle_size = require_int(
        restore.get("bundle_size_bytes"),
        minimum=1,
        code="IMPLEMENTATION_BYTE_IDENTITY_MISMATCH",
    )
    bundle_path = Path(
        require_text(
            restore.get("bundle_path"),
            code="IMPLEMENTATION_BYTE_IDENTITY_MISMATCH",
        )
    )
    try:
        bundle_raw = bundle_path.read_bytes()
    except OSError:
        _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    if (
        len(bundle_raw) != bundle_size
        or sha256_bytes(bundle_raw) != restore["bundle_file_sha256"]
    ):
        _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    if restore.get("restore_head") != source_commit:
        _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    if remediation_control is not None:
        if selection is None or selection_file_hash is None:
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        restore_receipt, restore_receipt_file_hash = (
            read_r2r1_clean_restore_receipt(plan)
        )
        if (
            restore.get("restore_receipt_file_sha256")
            != restore_receipt_file_hash
            or restore.get("restore_receipt_payload_sha256")
            != restore_receipt.get("restore_receipt_payload_sha256")
        ):
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        if candidate_manifest is None:
            manifest, manifest_file_hash = read_r2r1_candidate_manifest(plan)
        else:
            manifest = dict(candidate_manifest)
            manifest_file_hash = (
                candidate_manifest_file_sha256
                if candidate_manifest_file_sha256 is not None
                else sha256_bytes(canonical_receipt_bytes(manifest))
            )
        if (
            supplied.get("candidate_manifest_file_sha256")
            != manifest_file_hash
            or supplied.get("candidate_manifest_payload_sha256")
            != manifest.get("candidate_manifest_payload_sha256")
        ):
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        validate_r2r1_candidate_manifest(
            plan,
            manifest,
            candidate=supplied,
            selection=selection,
            selection_file_sha256=selection_file_hash,
        )
    elif control is not None:
        restore_receipt, restore_receipt_file_hash = (
            read_r2_clean_restore_receipt(plan)
        )
        if (
            restore.get("restore_receipt_file_sha256")
            != restore_receipt_file_hash
            or restore.get("restore_receipt_payload_sha256")
            != restore_receipt.get("restore_receipt_payload_sha256")
        ):
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        validate_r2_clean_restore_receipt(
            plan,
            restore_receipt,
            candidate=supplied,
        )
        if candidate_manifest is None:
            manifest, manifest_file_hash = read_r2_candidate_manifest(plan)
        else:
            manifest = dict(candidate_manifest)
            manifest_file_hash = (
                candidate_manifest_file_sha256
                if candidate_manifest_file_sha256 is not None
                else sha256_bytes(canonical_receipt_bytes(manifest))
            )
        if (
            supplied.get("candidate_manifest_file_sha256")
            != manifest_file_hash
            or supplied.get("candidate_manifest_payload_sha256")
            != manifest.get("candidate_manifest_payload_sha256")
        ):
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        validate_r2_candidate_manifest(
            plan,
            manifest,
            candidate=supplied,
        )
    return supplied


def validate_implementation_review(
    plan: Mapping[str, Any],
    review: Mapping[str, Any],
    *,
    candidate_file_sha256: str,
    candidate_payload_sha256: str,
    source_commit: str,
    candidate: Mapping[str, Any],
    review_evidence: Mapping[str, Any] | None = None,
    review_evidence_file_sha256: str | None = None,
    candidate_selection: Mapping[str, Any] | None = None,
    candidate_selection_file_sha256: str | None = None,
    review_input_freeze: Mapping[str, Any] | None = None,
    review_input_freeze_file_sha256: str | None = None,
) -> dict[str, Any]:
    schema = plan["review_receipt_schemas"][
        "fresh_implementation_review_verdict"
    ]
    supplied = dict(review)
    require_exact_keys(
        supplied,
        artifact_exact_fields(plan, "fresh_implementation_review_verdict"),
        code="INPUT_LINEAGE_MISMATCH",
    )
    verify_self_hash(
        supplied,
        "fresh_implementation_review_payload_sha256",
        code="INPUT_LINEAGE_MISMATCH",
    )
    candidate_surface = validate_review_surface_identity(
        plan, candidate.get("review_surface_identity")
    )
    review_surface = validate_review_surface_identity(
        plan, supplied.get("review_surface_identity")
    )
    if review_surface != candidate_surface:
        _raise("INPUT_LINEAGE_MISMATCH")
    required = schema["outcomes"]["pass"]["required_values"]
    if any(supplied.get(key) != value for key, value in required.items()):
        _raise("INPUT_LINEAGE_MISMATCH")
    p0_count = require_int(
        supplied.get("P0_count"), code="INPUT_LINEAGE_MISMATCH"
    )
    p1_count = require_int(
        supplied.get("P1_count"), code="INPUT_LINEAGE_MISMATCH"
    )
    require_int(
        supplied.get("P2_count"), minimum=0, code="INPUT_LINEAGE_MISMATCH"
    )
    if p0_count != 0 or p1_count != 0:
        _raise("INPUT_LINEAGE_MISMATCH")
    restore = require_mapping(
        candidate.get("clean_restore"), code="INPUT_LINEAGE_MISMATCH"
    )
    expected = {
        "implementation_author_separation_contract_sha256": (
            IMPLEMENTATION_AUTHOR_SEPARATION_SHA256
        ),
        "formal_design_review_file_sha256": FORMAL_DESIGN_REVIEW_FILE_SHA256,
        "formal_design_review_payload_sha256": (
            FORMAL_DESIGN_REVIEW_PAYLOAD_SHA256
        ),
        "implementation_trust_model_sha256": (
            recompute_implementation_trust_model_sha256(plan)
        ),
        "implementation_candidate_binding_file_sha256": candidate_file_sha256,
        "implementation_candidate_binding_payload_sha256": (
            candidate_payload_sha256
        ),
        "implementation_source_commit": source_commit,
        "bundle_file_sha256": restore.get("bundle_file_sha256"),
        "restore_receipt_file_sha256": restore.get(
            "restore_receipt_file_sha256"
        ),
        "restore_receipt_payload_sha256": restore.get(
            "restore_receipt_payload_sha256"
        ),
    }
    control = r2_activation_control(plan)
    remediation_control = active_remediation_control(plan)
    if remediation_control is not None:
        if candidate_selection is None:
            selection, selection_file_hash = read_r2r1_candidate_selection(
                plan,
                repo_root=AUTHORIZED_IMPLEMENTATION_REPOSITORY,
                current_head=source_commit,
            )
        else:
            selection = validate_r2r1_candidate_selection(
                plan,
                candidate_selection,
                repo_root=AUTHORIZED_IMPLEMENTATION_REPOSITORY,
                current_head=source_commit,
            )
            selection_file_hash = (
                candidate_selection_file_sha256
                if candidate_selection_file_sha256 is not None
                else sha256_bytes(canonical_receipt_bytes(selection))
            )
        if review_input_freeze is None:
            freeze, freeze_file_hash = read_r2r1_review_input_freeze(
                plan,
                candidate=candidate,
                candidate_file_sha256=candidate_file_sha256,
                selection=selection,
                selection_file_sha256=selection_file_hash,
            )
        else:
            freeze = validate_r2r1_review_input_freeze(
                plan,
                review_input_freeze,
                candidate=candidate,
                candidate_file_sha256=candidate_file_sha256,
                selection=selection,
                selection_file_sha256=selection_file_hash,
            )
            freeze_file_hash = (
                review_input_freeze_file_sha256
                if review_input_freeze_file_sha256 is not None
                else sha256_bytes(canonical_receipt_bytes(freeze))
            )
        if review_evidence is None:
            evidence, evidence_file_hash = read_r2r1_fresh_review_evidence(
                plan,
                candidate=candidate,
                candidate_file_sha256=candidate_file_sha256,
                selection=selection,
                selection_file_sha256=selection_file_hash,
            )
        else:
            evidence = validate_r2r1_fresh_review_evidence(
                plan,
                review_evidence,
                candidate=candidate,
                candidate_file_sha256=candidate_file_sha256,
                selection=selection,
                selection_file_sha256=selection_file_hash,
                freeze=freeze,
                freeze_file_sha256=freeze_file_hash,
            )
            evidence_file_hash = (
                review_evidence_file_sha256
                if review_evidence_file_sha256 is not None
                else sha256_bytes(canonical_receipt_bytes(evidence))
            )
        expected.update(
            {
                "authority_namespace_id": active_remediation_identity(plan)["authority_namespace_id"],
                "contract_file_sha256": CONTRACT_FILE_SHA256,
                "plan_file_sha256": PLAN_FILE_SHA256,
                "plan_payload_sha256": PLAN_PAYLOAD_SHA256,
                "r2_activation_plan_file_sha256": (
                    R2_ACTIVATION_PLAN_FILE_SHA256
                ),
                "r2_activation_plan_payload_sha256": (
                    R2_ACTIVATION_PLAN_PAYLOAD_SHA256
                ),
                **active_remediation_identity(plan)["static_fields"],
                "artifact_path_surface_sha256": (
                    artifact_surface_sha256(plan)
                ),
                "occupied_v0_9_surface_sha256": (
                    R2_OCCUPIED_V0_9_SURFACE_SHA256
                ),

                "candidate_manifest_file_sha256": candidate.get(
                    "candidate_manifest_file_sha256"
                ),
                "candidate_manifest_payload_sha256": candidate.get(
                    "candidate_manifest_payload_sha256"
                ),
                "candidate_selection_file_sha256": selection_file_hash,
                "candidate_selection_payload_sha256": selection[
                    "candidate_selection_payload_sha256"
                ],
                "review_input_freeze_file_sha256": freeze_file_hash,
                "review_input_freeze_payload_sha256": freeze[
                    "review_input_freeze_payload_sha256"
                ],
                "implementation_review_packet_file_sha256": freeze[
                    "review_packet_file_sha256"
                ],
                "review_evidence_file_sha256": evidence_file_hash,
                "review_evidence_payload_sha256": evidence.get(
                    "review_evidence_payload_sha256"
                ),
            }
        )
    elif control is not None:
        if review_evidence is None:
            evidence, evidence_file_hash = read_r2_fresh_review_evidence(plan)
        else:
            evidence = dict(review_evidence)
            evidence_file_hash = (
                review_evidence_file_sha256
                if review_evidence_file_sha256 is not None
                else sha256_bytes(canonical_receipt_bytes(evidence))
            )
        validate_r2_fresh_review_evidence(
            plan,
            evidence,
            candidate=candidate,
            candidate_file_sha256=candidate_file_sha256,
        )
        expected.update(
            {
                "authority_namespace_id": R2_AUTHORITY_NAMESPACE_ID,
                "contract_file_sha256": CONTRACT_FILE_SHA256,
                "plan_file_sha256": PLAN_FILE_SHA256,
                "plan_payload_sha256": PLAN_PAYLOAD_SHA256,
                "r2_activation_plan_file_sha256": (
                    R2_ACTIVATION_PLAN_FILE_SHA256
                ),
                "r2_activation_plan_payload_sha256": (
                    R2_ACTIVATION_PLAN_PAYLOAD_SHA256
                ),
                "artifact_path_surface_sha256": (
                    artifact_surface_sha256(plan)
                ),
                "occupied_v0_9_surface_sha256": (
                    R2_OCCUPIED_V0_9_SURFACE_SHA256
                ),
                "candidate_manifest_file_sha256": candidate.get(
                    "candidate_manifest_file_sha256"
                ),
                "candidate_manifest_payload_sha256": candidate.get(
                    "candidate_manifest_payload_sha256"
                ),
                "review_evidence_file_sha256": evidence_file_hash,
                "review_evidence_payload_sha256": evidence.get(
                    "review_evidence_payload_sha256"
                ),
            }
        )
    if any(supplied.get(key) != value for key, value in expected.items()):
        _raise("INPUT_LINEAGE_MISMATCH")
    if not is_sha256(supplied.get("implementation_review_packet_file_sha256")):
        _raise("INPUT_LINEAGE_MISMATCH")
    parse_utc_ns(supplied.get("reviewed_at_utc"), code="INPUT_LINEAGE_MISMATCH")
    return supplied


def build_reviewed_authority_payload(
    plan: Mapping[str, Any],
    candidate: Mapping[str, Any],
    review: Mapping[str, Any],
    *,
    candidate_file_sha256: str,
    review_file_sha256: str,
) -> dict[str, Any]:
    contract = plan["reviewed_implementation_authority_contract"]
    surface = validate_review_surface_identity(
        plan, candidate.get("review_surface_identity")
    )
    review_surface = validate_review_surface_identity(
        plan, review.get("review_surface_identity")
    )
    if review_surface != surface:
        _raise("INPUT_LINEAGE_MISMATCH")
    payload = {
        "schema_version": contract["schema_version"],
        "authority_id": contract["authority_id_value"],
        "state": contract["state"],
        "authority_derivation_domain": contract["authority_derivation_domain"],
        "implementation_source_commit": candidate["source_commit"],
        "authorized_implementation_repository": str(
            AUTHORIZED_IMPLEMENTATION_REPOSITORY
        ),
        "remediation_base_commit": plan["implementation_binding_contract"][
            "required_values"
        ]["remediation_base_commit"],
        "contract_file_sha256": CONTRACT_FILE_SHA256,
        "plan_file_sha256": PLAN_FILE_SHA256,
        "plan_payload_sha256": PLAN_PAYLOAD_SHA256,
        "implementation_author_separation_contract_sha256": (
            IMPLEMENTATION_AUTHOR_SEPARATION_SHA256
        ),
        "implementation_author_separation_basis": contract[
            "implementation_author_separation_basis"
        ],
        "implementation_context_blindness_machine_authenticated": False,
        "implementation_trust_model_sha256": (
            recompute_implementation_trust_model_sha256(plan)
        ),
        "artifact_path_surface_sha256": artifact_surface_sha256(plan),
        "review_surface_identity": surface,
        "formal_design_review_file_sha256": FORMAL_DESIGN_REVIEW_FILE_SHA256,
        "formal_design_review_payload_sha256": (
            FORMAL_DESIGN_REVIEW_PAYLOAD_SHA256
        ),
        "implementation_candidate_binding_file_sha256": candidate_file_sha256,
        "implementation_candidate_binding_payload_sha256": candidate[
            "implementation_candidate_binding_payload_sha256"
        ],
        "fresh_implementation_review_file_sha256": review_file_sha256,
        "fresh_implementation_review_payload_sha256": review[
            "fresh_implementation_review_payload_sha256"
        ],
        "task_identity_used_as_machine_authority": False,
        "implementation_authorship_machine_verified": False,
        "authority_issuer_identity_required": False,
    }
    control = r2_activation_control(plan)
    remediation_control = active_remediation_control(plan)
    if remediation_control is not None:
        selection, selection_file_hash = read_r2r1_candidate_selection(
            plan,
            repo_root=AUTHORIZED_IMPLEMENTATION_REPOSITORY,
            current_head=candidate.get("source_commit"),
        )
        freeze, freeze_file_hash = read_r2r1_review_input_freeze(
            plan,
            candidate=candidate,
            candidate_file_sha256=candidate_file_sha256,
            selection=selection,
            selection_file_sha256=selection_file_hash,
        )
        evidence, evidence_file_hash = read_r2r1_fresh_review_evidence(
            plan,
            candidate=candidate,
            candidate_file_sha256=candidate_file_sha256,
            selection=selection,
            selection_file_sha256=selection_file_hash,
        )
        if (
            candidate.get("authority_namespace_id")
            != active_remediation_identity(plan)["authority_namespace_id"]
            or review.get("authority_namespace_id")
            != active_remediation_identity(plan)["authority_namespace_id"]
            or review.get("implementation_source_commit")
            != candidate.get("source_commit")
            or review.get("implementation_candidate_binding_file_sha256")
            != candidate_file_sha256
            or review.get(
                "implementation_candidate_binding_payload_sha256"
            )
            != candidate.get(
                "implementation_candidate_binding_payload_sha256"
            )
            or review.get("candidate_selection_file_sha256")
            != candidate.get("candidate_selection_file_sha256")
            or review.get("candidate_selection_payload_sha256")
            != candidate.get("candidate_selection_payload_sha256")
            or candidate.get("source_commit")
            != selection.get("exact_candidate_commit")
            or candidate.get("candidate_selection_file_sha256")
            != selection_file_hash
            or candidate.get("candidate_selection_payload_sha256")
            != selection.get("candidate_selection_payload_sha256")
            or review.get("review_input_freeze_file_sha256")
            != freeze_file_hash
            or review.get("review_input_freeze_payload_sha256")
            != freeze.get("review_input_freeze_payload_sha256")
            or review.get("implementation_review_packet_file_sha256")
            != freeze.get("review_packet_file_sha256")
            or review.get("review_evidence_file_sha256")
            != evidence_file_hash
            or review.get("review_evidence_payload_sha256")
            != evidence.get("review_evidence_payload_sha256")
        ):
            _raise("INPUT_LINEAGE_MISMATCH")
        payload.update(
            {
                "authority_namespace_id": active_remediation_identity(plan)["authority_namespace_id"],
                "r2_activation_plan_file_sha256": (
                    R2_ACTIVATION_PLAN_FILE_SHA256
                ),
                "r2_activation_plan_payload_sha256": (
                    R2_ACTIVATION_PLAN_PAYLOAD_SHA256
                ),
                **active_remediation_identity(plan)["static_fields"],
                "occupied_v0_9_surface_sha256": (
                    R2_OCCUPIED_V0_9_SURFACE_SHA256
                ),

                "candidate_manifest_file_sha256": candidate.get(
                    "candidate_manifest_file_sha256"
                ),
                "candidate_manifest_payload_sha256": candidate.get(
                    "candidate_manifest_payload_sha256"
                ),
                "candidate_selection_file_sha256": candidate.get(
                    "candidate_selection_file_sha256"
                ),
                "candidate_selection_payload_sha256": candidate.get(
                    "candidate_selection_payload_sha256"
                ),
                "review_input_freeze_file_sha256": review.get(
                    "review_input_freeze_file_sha256"
                ),
                "review_input_freeze_payload_sha256": review.get(
                    "review_input_freeze_payload_sha256"
                ),
                "implementation_review_packet_file_sha256": review.get(
                    "implementation_review_packet_file_sha256"
                ),
            }
        )
    elif control is not None:
        if (
            candidate.get("authority_namespace_id")
            != R2_AUTHORITY_NAMESPACE_ID
            or review.get("authority_namespace_id")
            != R2_AUTHORITY_NAMESPACE_ID
            or review.get("implementation_source_commit")
            != candidate.get("source_commit")
            or review.get("implementation_candidate_binding_file_sha256")
            != candidate_file_sha256
            or review.get(
                "implementation_candidate_binding_payload_sha256"
            )
            != candidate.get(
                "implementation_candidate_binding_payload_sha256"
            )
        ):
            _raise("INPUT_LINEAGE_MISMATCH")
        payload.update(
            {
                "authority_namespace_id": R2_AUTHORITY_NAMESPACE_ID,
                "r2_activation_plan_file_sha256": (
                    R2_ACTIVATION_PLAN_FILE_SHA256
                ),
                "r2_activation_plan_payload_sha256": (
                    R2_ACTIVATION_PLAN_PAYLOAD_SHA256
                ),
                "occupied_v0_9_surface_sha256": (
                    R2_OCCUPIED_V0_9_SURFACE_SHA256
                ),
                "candidate_manifest_file_sha256": candidate.get(
                    "candidate_manifest_file_sha256"
                ),
                "candidate_manifest_payload_sha256": candidate.get(
                    "candidate_manifest_payload_sha256"
                ),
            }
        )
    require_exact_keys(
        {**payload, "reviewed_implementation_authority_payload_sha256": ""},
        artifact_exact_fields(plan, "reviewed_implementation_authority"),
        code="INPUT_LINEAGE_MISMATCH",
    )
    return add_self_hash(
        payload, "reviewed_implementation_authority_payload_sha256"
    )


def validate_reviewed_authority(
    plan: Mapping[str, Any],
    authority: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    candidate_file_sha256: str,
    review: Mapping[str, Any],
    review_file_sha256: str,
) -> dict[str, Any]:
    contract = plan["reviewed_implementation_authority_contract"]
    supplied = dict(authority)
    require_exact_keys(
        supplied,
        artifact_exact_fields(plan, "reviewed_implementation_authority"),
        code="INPUT_LINEAGE_MISMATCH",
    )
    verify_self_hash(
        supplied,
        "reviewed_implementation_authority_payload_sha256",
        code="INPUT_LINEAGE_MISMATCH",
    )
    validate_review_surface_identity(
        plan, supplied.get("review_surface_identity")
    )
    expected = build_reviewed_authority_payload(
        plan,
        candidate,
        review,
        candidate_file_sha256=candidate_file_sha256,
        review_file_sha256=review_file_sha256,
    )
    if supplied != expected:
        _raise("INPUT_LINEAGE_MISMATCH")
    prohibited_fields = {
        "issued_at_utc",
        "issuer_id",
        "hostname",
        "username",
        "reviewer_id",
        "task_id",
        "context_id",
    }
    if set(supplied) & prohibited_fields:
        _raise("INPUT_LINEAGE_MISMATCH")
    return supplied



def _control_schema(
    plan: Mapping[str, Any], scope: str, kind: str
) -> Mapping[str, Any]:
    if scope not in {"extraction", "verifier"}:
        _raise("AUTHORIZATION_INVALID")
    key = f"{scope}_{kind}"
    schema = plan["control_receipt_schemas"].get(key)
    if not isinstance(schema, dict):
        _raise("AUTHORIZATION_INVALID")
    return schema


def build_preflight_payload(
    plan: Mapping[str, Any],
    *,
    scope: str,
    preflight_id: str,
    issued_at_utc: str,
    expires_at_utc: str,
    reviewed_authority_file_sha256: str,
    reviewed_authority_payload_sha256: str,
    implementation_source_commit: str,
    extraction_terminal_file_sha256: str | None = None,
    extraction_terminal_payload_sha256: str | None = None,
    baseline_receipt_file_sha256: str | None = None,
    baseline_receipt_payload_sha256: str | None = None,
    now_ns: int | None = None,
) -> dict[str, Any]:
    schema = _control_schema(plan, scope, "preflight")
    require_identifier(preflight_id, code="AUTHORIZATION_INVALID")
    require_fresh_interval(
        issued_at_utc,
        expires_at_utc,
        now_ns=now_ns,
        maximum_age_seconds=schema["maximum_age_seconds"],
    )
    for digest in (
        reviewed_authority_file_sha256,
        reviewed_authority_payload_sha256,
    ):
        if not is_sha256(digest):
            _raise("AUTHORIZATION_INVALID")
    if re.fullmatch(r"[0-9a-f]{40}", implementation_source_commit) is None:
        _raise("AUTHORIZATION_INVALID")
    payload: dict[str, Any] = {
        "schema_version": schema["schema_version"],
        "gate_id": GATE_ID,
        "authorization_scope": scope,
        "preflight_id": preflight_id,
        "state": schema["state"],
        "sequence_ordinal": schema["sequence_ordinal"],
        "issued_at_utc": issued_at_utc,
        "expires_at_utc": expires_at_utc,
        "contract_file_sha256": CONTRACT_FILE_SHA256,
        "plan_file_sha256": PLAN_FILE_SHA256,
        "plan_payload_sha256": PLAN_PAYLOAD_SHA256,
        "reviewed_implementation_authority_file_sha256": (
            reviewed_authority_file_sha256
        ),
        "reviewed_implementation_authority_payload_sha256": (
            reviewed_authority_payload_sha256
        ),
        "authorized_implementation_repository": str(
            AUTHORIZED_IMPLEMENTATION_REPOSITORY
        ),
        "implementation_source_commit": implementation_source_commit,
        "executing_code_identity_status": "verified",
        "artifact_path_surface_sha256": artifact_surface_sha256(plan),
        "artifact_lifecycle_phase": schema[
            "required_artifact_lifecycle_phase"
        ],
        "input_lineage_status": "verified",
        "protected_root_status": "canonical_path_bound_no_payload_read",
        "output_surface_status": "fresh_exact",
        "closed_boundaries_status": "closed",
    }
    if scope == "verifier":
        links = {
            "extraction_terminal_claim_file_sha256": (
                extraction_terminal_file_sha256
            ),
            "extraction_terminal_claim_payload_sha256": (
                extraction_terminal_payload_sha256
            ),
            "baseline_receipt_file_sha256": baseline_receipt_file_sha256,
            "baseline_receipt_payload_sha256": baseline_receipt_payload_sha256,
        }
        if any(not is_sha256(value) for value in links.values()):
            _raise("AUTHORIZATION_INVALID")
        payload.update(links)
    require_exact_keys(
        {**payload, "preflight_payload_sha256": ""},
        schema["exact_top_level_fields"],
        code="AUTHORIZATION_INVALID",
    )
    return add_self_hash(payload, "preflight_payload_sha256")


def validate_preflight_payload(
    plan: Mapping[str, Any],
    preflight: Mapping[str, Any],
    *,
    scope: str,
    reviewed_authority_file_sha256: str,
    reviewed_authority_payload_sha256: str,
    implementation_source_commit: str,
    extraction_terminal_file_sha256: str | None = None,
    extraction_terminal_payload_sha256: str | None = None,
    baseline_receipt_file_sha256: str | None = None,
    baseline_receipt_payload_sha256: str | None = None,
    now_ns: int | None = None,
) -> dict[str, Any]:
    schema = _control_schema(plan, scope, "preflight")
    supplied = dict(preflight)
    require_exact_keys(
        supplied, schema["exact_top_level_fields"], code="AUTHORIZATION_INVALID"
    )
    verify_self_hash(
        supplied, "preflight_payload_sha256", code="AUTHORIZATION_INVALID"
    )
    expected = {
        "schema_version": schema["schema_version"],
        "gate_id": GATE_ID,
        "authorization_scope": scope,
        "state": schema["state"],
        "sequence_ordinal": schema["sequence_ordinal"],
        "contract_file_sha256": CONTRACT_FILE_SHA256,
        "plan_file_sha256": PLAN_FILE_SHA256,
        "plan_payload_sha256": PLAN_PAYLOAD_SHA256,
        "reviewed_implementation_authority_file_sha256": (
            reviewed_authority_file_sha256
        ),
        "reviewed_implementation_authority_payload_sha256": (
            reviewed_authority_payload_sha256
        ),
        "authorized_implementation_repository": str(
            AUTHORIZED_IMPLEMENTATION_REPOSITORY
        ),
        "implementation_source_commit": implementation_source_commit,
        "executing_code_identity_status": "verified",
        "artifact_path_surface_sha256": artifact_surface_sha256(plan),
        "artifact_lifecycle_phase": schema[
            "required_artifact_lifecycle_phase"
        ],
        "input_lineage_status": "verified",
        "protected_root_status": "canonical_path_bound_no_payload_read",
        "output_surface_status": "fresh_exact",
        "closed_boundaries_status": "closed",
    }
    if any(supplied.get(key) != value for key, value in expected.items()):
        _raise("AUTHORIZATION_INVALID")
    if scope == "verifier":
        supplied_links = {
            "extraction_terminal_claim_file_sha256": supplied.get(
                "extraction_terminal_claim_file_sha256"
            ),
            "extraction_terminal_claim_payload_sha256": supplied.get(
                "extraction_terminal_claim_payload_sha256"
            ),
            "baseline_receipt_file_sha256": supplied.get(
                "baseline_receipt_file_sha256"
            ),
            "baseline_receipt_payload_sha256": supplied.get(
                "baseline_receipt_payload_sha256"
            ),
        }
        if any(not is_sha256(value) for value in supplied_links.values()):
            _raise("AUTHORIZATION_INVALID")
        expected_links = {
            "extraction_terminal_claim_file_sha256": (
                extraction_terminal_file_sha256
            ),
            "extraction_terminal_claim_payload_sha256": (
                extraction_terminal_payload_sha256
            ),
            "baseline_receipt_file_sha256": baseline_receipt_file_sha256,
            "baseline_receipt_payload_sha256": baseline_receipt_payload_sha256,
        }
        for field, value in expected_links.items():
            if value is not None and supplied_links[field] != value:
                _raise("AUTHORIZATION_INVALID")
    require_identifier(supplied.get("preflight_id"), code="AUTHORIZATION_INVALID")
    require_fresh_interval(
        supplied.get("issued_at_utc"),
        supplied.get("expires_at_utc"),
        now_ns=now_ns,
        maximum_age_seconds=schema["maximum_age_seconds"],
    )
    return supplied


def build_authorization_payload(
    plan: Mapping[str, Any],
    preflight: Mapping[str, Any],
    *,
    scope: str,
    preflight_file_sha256: str,
    authorization_id: str,
    issued_at_utc: str,
    expires_at_utc: str,
    now_ns: int | None = None,
) -> dict[str, Any]:
    schema = _control_schema(plan, scope, "authorization")
    require_identifier(authorization_id, code="AUTHORIZATION_INVALID")
    if not is_sha256(preflight_file_sha256):
        _raise("AUTHORIZATION_INVALID")
    remaining = require_fresh_interval(
        issued_at_utc,
        expires_at_utc,
        now_ns=now_ns,
        maximum_age_seconds=schema["maximum_age_seconds"],
    )
    del remaining
    preflight_issued_ns = parse_utc_ns(
        preflight.get("issued_at_utc"), code="AUTHORIZATION_INVALID"
    )
    preflight_expires_ns = parse_utc_ns(
        preflight.get("expires_at_utc"), code="AUTHORIZATION_INVALID"
    )
    authorization_issued_ns = parse_utc_ns(
        issued_at_utc, code="AUTHORIZATION_INVALID"
    )
    authorization_expires_ns = parse_utc_ns(
        expires_at_utc, code="AUTHORIZATION_INVALID"
    )
    if (
        authorization_issued_ns < preflight_issued_ns
        or authorization_expires_ns > preflight_expires_ns
    ):
        _raise("AUTHORIZATION_INVALID")
    payload: dict[str, Any] = {
        "schema_version": schema["schema_version"],
        "gate_id": GATE_ID,
        "authorization_scope": scope,
        "authorization_id": authorization_id,
        "state": schema["state"],
        "sequence_ordinal": schema["sequence_ordinal"],
        "issued_at_utc": issued_at_utc,
        "expires_at_utc": expires_at_utc,
        "single_use": True,
        "plan_payload_sha256": PLAN_PAYLOAD_SHA256,
        "reviewed_implementation_authority_file_sha256": preflight[
            "reviewed_implementation_authority_file_sha256"
        ],
        "reviewed_implementation_authority_payload_sha256": preflight[
            "reviewed_implementation_authority_payload_sha256"
        ],
        "preflight_file_sha256": preflight_file_sha256,
        "preflight_payload_sha256": preflight["preflight_payload_sha256"],
        "protected_root_path": str(PROTECTED_ROOT),
        "artifact_path_surface_sha256": artifact_surface_sha256(plan),
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
            payload[field] = preflight[field]
    require_exact_keys(
        {**payload, "authorization_payload_sha256": ""},
        schema["exact_top_level_fields"],
        code="AUTHORIZATION_INVALID",
    )
    return add_self_hash(payload, "authorization_payload_sha256")


def validate_authorization_payload(
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    preflight: Mapping[str, Any],
    *,
    scope: str,
    preflight_file_sha256: str,
    now_ns: int | None = None,
) -> dict[str, Any]:
    schema = _control_schema(plan, scope, "authorization")
    supplied = dict(authorization)
    require_exact_keys(
        supplied, schema["exact_top_level_fields"], code="AUTHORIZATION_INVALID"
    )
    verify_self_hash(
        supplied, "authorization_payload_sha256", code="AUTHORIZATION_INVALID"
    )
    expected = build_authorization_payload(
        plan,
        preflight,
        scope=scope,
        preflight_file_sha256=preflight_file_sha256,
        authorization_id=str(supplied.get("authorization_id", "")),
        issued_at_utc=str(supplied.get("issued_at_utc", "")),
        expires_at_utc=str(supplied.get("expires_at_utc", "")),
        now_ns=now_ns,
    )
    if supplied != expected:
        _raise("AUTHORIZATION_INVALID")
    return supplied


def build_authorization_verdict_payload(
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    preflight: Mapping[str, Any],
    *,
    scope: str,
    verification_id: str,
    verified_at_utc: str,
    outcome_kind: str,
    reason_code: str | None,
    preflight_file_sha256: str,
    authorization_file_sha256: str,
    verifier_relative_path: str,
    verifier_file_sha256: str,
    verifier_git_blob_oid: str,
    now_ns: int | None = None,
) -> dict[str, Any]:
    schema = _control_schema(plan, scope, "authorization_verdict")
    require_identifier(verification_id, code="AUTHORIZATION_INVALID")
    if outcome_kind not in schema["outcome_kind_allowlist"]:
        _raise("AUTHORIZATION_INVALID")
    if reason_code not in schema["reason_code_allowlist"]:
        _raise("AUTHORIZATION_INVALID")
    if (outcome_kind == "pass") != (reason_code is None):
        _raise("AUTHORIZATION_INVALID")
    verified_ns = parse_utc_ns(verified_at_utc, code="AUTHORIZATION_INVALID")
    current = (
        verified_ns
        if now_ns is None
        else require_int(now_ns, minimum=0, code="AUTHORIZATION_INVALID")
    )
    issued_ns = parse_utc_ns(
        authorization.get("issued_at_utc"), code="AUTHORIZATION_INVALID"
    )
    expires_ns = parse_utc_ns(
        authorization.get("expires_at_utc"), code="AUTHORIZATION_INVALID"
    )
    if (
        expires_ns <= issued_ns
        or expires_ns - issued_ns
        > MAXIMUM_FRESHNESS_SECONDS * 1_000_000_000
        or verified_ns > current
    ):
        _raise("AUTHORIZATION_INVALID")
    remaining = expires_ns - verified_ns
    if reason_code == "AUTHORIZATION_STALE":
        if not expires_ns <= verified_ns <= current:
            _raise("AUTHORIZATION_INVALID")
    elif not issued_ns <= verified_ns <= current < expires_ns:
        _raise("AUTHORIZATION_INVALID")
    digests = (
        preflight_file_sha256,
        authorization_file_sha256,
        verifier_file_sha256,
    )
    if any(not is_sha256(value) for value in digests):
        _raise("AUTHORIZATION_INVALID")
    if not is_git_oid(verifier_git_blob_oid):
        _raise("AUTHORIZATION_INVALID")
    payload: dict[str, Any] = {
        "schema_version": schema["schema_version"],
        "gate_id": GATE_ID,
        "authorization_scope": scope,
        "verification_id": verification_id,
        "state": (
            schema["pass_state"]
            if outcome_kind == "pass"
            else schema["reject_state"]
        ),
        "sequence_ordinal": schema["sequence_ordinal"],
        "verified_at_utc": verified_at_utc,
        "outcome_kind": outcome_kind,
        "reason_code": reason_code,
        "plan_payload_sha256": PLAN_PAYLOAD_SHA256,
        "reviewed_implementation_authority_file_sha256": authorization[
            "reviewed_implementation_authority_file_sha256"
        ],
        "reviewed_implementation_authority_payload_sha256": authorization[
            "reviewed_implementation_authority_payload_sha256"
        ],
        "preflight_file_sha256": preflight_file_sha256,
        "preflight_payload_sha256": preflight["preflight_payload_sha256"],
        "authorization_file_sha256": authorization_file_sha256,
        "authorization_payload_sha256": authorization[
            "authorization_payload_sha256"
        ],
        "artifact_path_surface_sha256": artifact_surface_sha256(plan),
        "artifact_lifecycle_phase": schema[
            "required_artifact_lifecycle_phase"
        ],
        "authorization_expires_at_utc": authorization["expires_at_utc"],
        "remaining_freshness_nanoseconds": remaining,
        "protected_root_read": False,
        "authorization_verifier_relative_path": verifier_relative_path,
        "authorization_verifier_file_sha256": verifier_file_sha256,
        "authorization_verifier_git_blob_oid": verifier_git_blob_oid,
    }
    if scope == "verifier":
        for field in (
            "baseline_receipt_file_sha256",
            "baseline_receipt_payload_sha256",
            "extraction_terminal_claim_file_sha256",
            "extraction_terminal_claim_payload_sha256",
        ):
            payload[field] = authorization[field]
    require_exact_keys(
        {**payload, "authorization_verdict_payload_sha256": ""},
        schema["exact_top_level_fields"],
        code="AUTHORIZATION_INVALID",
    )
    return add_self_hash(payload, "authorization_verdict_payload_sha256")

def validate_authorization_verdict_payload(
    plan: Mapping[str, Any],
    verdict: Mapping[str, Any],
    authorization: Mapping[str, Any],
    preflight: Mapping[str, Any],
    *,
    scope: str,
    preflight_file_sha256: str,
    authorization_file_sha256: str,
    verifier_relative_path: str,
    verifier_file_sha256: str,
    verifier_git_blob_oid: str,
    now_ns: int | None = None,
) -> dict[str, Any]:
    schema = _control_schema(plan, scope, "authorization_verdict")
    supplied = dict(verdict)
    require_exact_keys(
        supplied, schema["exact_top_level_fields"], code="AUTHORIZATION_INVALID"
    )
    verify_self_hash(
        supplied,
        "authorization_verdict_payload_sha256",
        code="AUTHORIZATION_INVALID",
    )
    expected = build_authorization_verdict_payload(
        plan,
        authorization,
        preflight,
        scope=scope,
        verification_id=supplied.get("verification_id"),
        verified_at_utc=supplied.get("verified_at_utc"),
        outcome_kind=supplied.get("outcome_kind"),
        reason_code=supplied.get("reason_code"),
        preflight_file_sha256=preflight_file_sha256,
        authorization_file_sha256=authorization_file_sha256,
        verifier_relative_path=verifier_relative_path,
        verifier_file_sha256=verifier_file_sha256,
        verifier_git_blob_oid=verifier_git_blob_oid,
        now_ns=now_ns,
    )
    if supplied != expected:
        _raise("AUTHORIZATION_INVALID")
    return supplied





def build_execution_claim_payload(
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
    preflight: Mapping[str, Any],
    verdict: Mapping[str, Any],
    *,
    scope: str,
    execution_claim_id: str,
    launch_id: str,
    claimed_at_utc: str,
    owner_hostname: str,
    owner_pid: int,
    owner_process_creation_time_utc: str,
    git_head_at_claim: str,
    executing_code_identity_surface_sha256: str,
    preflight_file_sha256: str,
    authorization_file_sha256: str,
    verdict_file_sha256: str,
    now_ns: int | None = None,
) -> dict[str, Any]:
    schema = _control_schema(plan, scope, "execution_claim")
    require_identifier(execution_claim_id, code="AUTHORIZATION_INVALID")
    require_identifier(launch_id, code="AUTHORIZATION_INVALID")
    require_text(owner_hostname, ascii_only=True, code="AUTHORIZATION_INVALID")
    require_int(owner_pid, minimum=1, maximum=(1 << 32) - 1, code="AUTHORIZATION_INVALID")
    creation_ns = parse_utc_ns(
        owner_process_creation_time_utc, code="AUTHORIZATION_INVALID"
    )
    claimed_ns = parse_utc_ns(claimed_at_utc, code="AUTHORIZATION_INVALID")
    verdict_verified_ns = parse_utc_ns(
        verdict.get("verified_at_utc"), code="AUTHORIZATION_INVALID"
    )
    authorization_issued_ns = parse_utc_ns(
        authorization.get("issued_at_utc"), code="AUTHORIZATION_INVALID"
    )
    authorization_expires_ns = parse_utc_ns(
        authorization.get("expires_at_utc"), code="AUTHORIZATION_INVALID"
    )
    current_ns = (
        parse_utc_ns(utc_now_text(), code="AUTHORIZATION_INVALID")
        if now_ns is None
        else require_int(now_ns, minimum=0, code="AUTHORIZATION_INVALID")
    )
    if (
        creation_ns > claimed_ns
        or claimed_ns < verdict_verified_ns
        or not authorization_issued_ns <= claimed_ns < authorization_expires_ns
        or claimed_ns > current_ns
    ):
        _raise("AUTHORIZATION_INVALID")
    require_fresh_interval(
        authorization["issued_at_utc"],
        authorization["expires_at_utc"],
        now_ns=current_ns,
    )
    if (
        verdict.get("outcome_kind") != "pass"
        or verdict.get("reason_code") is not None
        or verdict.get("authorization_payload_sha256")
        != authorization.get("authorization_payload_sha256")
    ):
        _raise("AUTHORIZATION_INVALID")
    for digest in (
        preflight_file_sha256,
        authorization_file_sha256,
        verdict_file_sha256,
        executing_code_identity_surface_sha256,
    ):
        if not is_sha256(digest):
            _raise("AUTHORIZATION_INVALID")
    implementation_source_commit = preflight.get("implementation_source_commit")
    if (
        re.fullmatch(r"[0-9a-f]{40}", str(implementation_source_commit)) is None
        or git_head_at_claim != implementation_source_commit
    ):
        _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    payload: dict[str, Any] = {
        "schema_version": schema["schema_version"],
        "gate_id": GATE_ID,
        "authorization_scope": scope,
        "execution_claim_id": execution_claim_id,
        "launch_id": launch_id,
        "state": schema["state"],
        "sequence_ordinal": schema["sequence_ordinal"],
        "claimed_at_utc": claimed_at_utc,
        "owner_hostname": owner_hostname,
        "owner_pid": owner_pid,
        "owner_process_creation_time_utc": owner_process_creation_time_utc,
        "plan_payload_sha256": PLAN_PAYLOAD_SHA256,
        "reviewed_implementation_authority_file_sha256": authorization[
            "reviewed_implementation_authority_file_sha256"
        ],
        "reviewed_implementation_authority_payload_sha256": authorization[
            "reviewed_implementation_authority_payload_sha256"
        ],
        "implementation_source_commit": implementation_source_commit,
        "git_head_at_claim": git_head_at_claim,
        "executing_code_identity_surface_sha256": (
            executing_code_identity_surface_sha256
        ),
        "preflight_file_sha256": preflight_file_sha256,
        "preflight_payload_sha256": preflight["preflight_payload_sha256"],
        "authorization_file_sha256": authorization_file_sha256,
        "authorization_payload_sha256": authorization[
            "authorization_payload_sha256"
        ],
        "authorization_verdict_file_sha256": verdict_file_sha256,
        "authorization_verdict_payload_sha256": verdict[
            "authorization_verdict_payload_sha256"
        ],
        "artifact_path_surface_sha256": artifact_surface_sha256(plan),
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
            payload[field] = authorization[field]
    require_exact_keys(
        {**payload, "execution_claim_payload_sha256": ""},
        schema["exact_top_level_fields"],
        code="AUTHORIZATION_INVALID",
    )
    return add_self_hash(payload, "execution_claim_payload_sha256")

def validate_execution_claim_payload(
    plan: Mapping[str, Any],
    claim: Mapping[str, Any],
    authorization: Mapping[str, Any],
    preflight: Mapping[str, Any],
    verdict: Mapping[str, Any],
    *,
    scope: str,
    preflight_file_sha256: str,
    authorization_file_sha256: str,
    verdict_file_sha256: str,
    now_ns: int | None = None,
) -> dict[str, Any]:
    schema = _control_schema(plan, scope, "execution_claim")
    supplied = dict(claim)
    require_exact_keys(
        supplied, schema["exact_top_level_fields"], code="AUTHORIZATION_INVALID"
    )
    verify_self_hash(
        supplied, "execution_claim_payload_sha256", code="AUTHORIZATION_INVALID"
    )
    expected = build_execution_claim_payload(
        plan,
        authorization,
        preflight,
        verdict,
        scope=scope,
        execution_claim_id=supplied.get("execution_claim_id"),
        launch_id=supplied.get("launch_id"),
        claimed_at_utc=supplied.get("claimed_at_utc"),
        owner_hostname=supplied.get("owner_hostname"),
        owner_pid=supplied.get("owner_pid"),
        owner_process_creation_time_utc=supplied.get(
            "owner_process_creation_time_utc"
        ),
        git_head_at_claim=supplied.get("git_head_at_claim"),
        executing_code_identity_surface_sha256=supplied.get(
            "executing_code_identity_surface_sha256"
        ),
        preflight_file_sha256=preflight_file_sha256,
        authorization_file_sha256=authorization_file_sha256,
        verdict_file_sha256=verdict_file_sha256,
        now_ns=now_ns,
    )
    if supplied != expected:
        _raise("AUTHORIZATION_INVALID")
    return supplied




def _filetime_to_utc_text(value: int) -> str:
    unix_100ns = value - 116_444_736_000_000_000
    if unix_100ns < 0:
        _raise("AUTHORIZATION_INVALID")
    seconds, remainder = divmod(unix_100ns, 10_000_000)
    moment = datetime.fromtimestamp(seconds, timezone.utc)
    fraction = f"{remainder:07d}".rstrip("0")
    base = moment.strftime("%Y-%m-%dT%H:%M:%S")
    return f"{base}.{fraction}Z" if fraction else f"{base}Z"


def query_process_creation_time_utc(pid: int) -> str:
    if os.name != "nt":
        _raise("AUTHORIZATION_INVALID")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_process = kernel32.OpenProcess
    open_process.argtypes = [
        ctypes.wintypes.DWORD,
        ctypes.wintypes.BOOL,
        ctypes.wintypes.DWORD,
    ]
    open_process.restype = ctypes.wintypes.HANDLE
    handle = open_process(0x1000, False, pid)
    if not handle:
        _raise("AUTHORIZATION_INVALID")
    creation = ctypes.wintypes.FILETIME()
    exit_time = ctypes.wintypes.FILETIME()
    kernel_time = ctypes.wintypes.FILETIME()
    user_time = ctypes.wintypes.FILETIME()
    try:
        get_times = kernel32.GetProcessTimes
        if not get_times(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel_time),
            ctypes.byref(user_time),
        ):
            _raise("AUTHORIZATION_INVALID")
    finally:
        kernel32.CloseHandle(handle)
    combined = (int(creation.dwHighDateTime) << 32) | int(
        creation.dwLowDateTime
    )
    return _filetime_to_utc_text(combined)


def classify_claim_owner(
    claim: Mapping[str, Any],
    *,
    hostname: str | None = None,
    creation_query: Callable[[int], str] = query_process_creation_time_utc,
) -> str:
    current_host = socket.gethostname() if hostname is None else hostname
    if claim.get("owner_hostname") != current_host:
        return "UNKNOWN"
    try:
        pid = require_int(
            claim.get("owner_pid"), minimum=1, code="AUTHORIZATION_INVALID"
        )
        observed = creation_query(pid)
    except Gate12C2OriginalBaselineError:
        return "UNKNOWN"
    except ProcessLookupError:
        return "DEAD"
    except Exception:
        return "UNKNOWN"
    return (
        "ACTIVE"
        if observed == claim.get("owner_process_creation_time_utc")
        else "DEAD"
    )


def build_terminal_payload(
    plan: Mapping[str, Any],
    claim: Mapping[str, Any],
    leaf: Mapping[str, Any],
    *,
    scope: str,
    outcome_kind: str,
    claimed_at_utc: str,
    reviewed_authority_file_sha256: str,
    preflight_file_sha256: str,
    authorization_file_sha256: str,
    verdict_file_sha256: str,
    execution_claim_file_sha256: str,
) -> dict[str, Any]:
    schema = _control_schema(plan, scope, "terminal")
    claim_schema = _control_schema(plan, scope, "execution_claim")
    claim_payload = require_mapping(
        claim, code="TERMINAL_OUTCOME_RECONSTRUCTION_FAILED"
    )
    require_exact_keys(
        claim_payload,
        claim_schema["exact_top_level_fields"],
        code="TERMINAL_OUTCOME_RECONSTRUCTION_FAILED",
    )
    verify_self_hash(
        claim_payload,
        "execution_claim_payload_sha256",
        code="TERMINAL_OUTCOME_RECONSTRUCTION_FAILED",
    )
    if (
        claim_payload.get("authorization_scope") != scope
        or claim_payload.get("state") != claim_schema["state"]
        or claim_payload.get("sequence_ordinal")
        != claim_schema["sequence_ordinal"]
    ):
        _raise("TERMINAL_OUTCOME_RECONSTRUCTION_FAILED")
    if outcome_kind not in schema["outcome_kind_allowlist"]:
        _raise("TERMINAL_CONFLICT")
    leaf_hash_field = (
        "baseline_receipt_payload_sha256"
        if scope == "extraction" and outcome_kind == "success"
        else "verification_receipt_payload_sha256"
        if scope == "verifier" and outcome_kind == "success"
        else "failure_receipt_payload_sha256"
    )
    leaf_payload = dict(leaf)
    leaf_schema = (
        plan["success_receipt"]
        if scope == "extraction" and outcome_kind == "success"
        else plan["verification_receipt"]
        if scope == "verifier" and outcome_kind == "success"
        else plan["extraction_failure_receipt"]
        if scope == "extraction"
        else plan["verifier_failure_receipt"]
    )
    require_exact_keys(
        leaf_payload,
        leaf_schema["exact_top_level_fields"],
        code="TERMINAL_OUTCOME_RECONSTRUCTION_FAILED",
    )
    if leaf_payload.get("schema_version") != leaf_schema["schema_version"]:
        _raise("TERMINAL_OUTCOME_RECONSTRUCTION_FAILED")
    leaf_digest = verify_self_hash(
        leaf_payload, leaf_hash_field, code="TERMINAL_OUTCOME_RECONSTRUCTION_FAILED"
    )
    if parse_utc_ns(claimed_at_utc) < parse_utc_ns(
        claim_payload["claimed_at_utc"]
    ):
        _raise("TERMINAL_CONFLICT")
    linked_file_hashes = {
        "reviewed_implementation_authority_file_sha256": (
            reviewed_authority_file_sha256
        ),
        "preflight_file_sha256": preflight_file_sha256,
        "authorization_file_sha256": authorization_file_sha256,
        "authorization_verdict_file_sha256": verdict_file_sha256,
    }
    if any(
        not is_sha256(value)
        or claim_payload.get(field) != value
        for field, value in linked_file_hashes.items()
    ) or not is_sha256(execution_claim_file_sha256):
        _raise("TERMINAL_OUTCOME_RECONSTRUCTION_FAILED")
    payload: dict[str, Any] = {
        "schema_version": schema["schema_version"],
        "gate_id": GATE_ID,
        "authorization_scope": scope,
        "execution_claim_id": claim_payload["execution_claim_id"],
        "state": (
            f"{scope.upper()}_{outcome_kind.upper()}_TERMINAL_CLAIM_PUBLISHED"
        ),
        "sequence_ordinal": schema["sequence_ordinal"],
        "claimed_at_utc": claimed_at_utc,
        "outcome_kind": outcome_kind,
        "reviewed_implementation_authority_file_sha256": (
            reviewed_authority_file_sha256
        ),
        "reviewed_implementation_authority_payload_sha256": claim_payload[
            "reviewed_implementation_authority_payload_sha256"
        ],
        "preflight_file_sha256": preflight_file_sha256,
        "preflight_payload_sha256": claim_payload["preflight_payload_sha256"],
        "authorization_file_sha256": authorization_file_sha256,
        "authorization_payload_sha256": claim_payload[
            "authorization_payload_sha256"
        ],
        "authorization_verdict_file_sha256": verdict_file_sha256,
        "authorization_verdict_payload_sha256": claim_payload[
            "authorization_verdict_payload_sha256"
        ],
        "execution_claim_file_sha256": execution_claim_file_sha256,
        "execution_claim_payload_sha256": claim_payload[
            "execution_claim_payload_sha256"
        ],
        "leaf_schema_version": leaf_payload["schema_version"],
        "leaf_payload_sha256": leaf_digest,
        "leaf_exact_payload": leaf_payload,
    }
    if scope == "verifier":
        for field in (
            "baseline_receipt_file_sha256",
            "baseline_receipt_payload_sha256",
            "extraction_terminal_claim_file_sha256",
            "extraction_terminal_claim_payload_sha256",
        ):
            payload[field] = claim_payload[field]
    require_exact_keys(
        {**payload, "terminal_claim_payload_sha256": ""},
        schema["exact_top_level_fields"],
        code="TERMINAL_CONFLICT",
    )
    return add_self_hash(payload, "terminal_claim_payload_sha256")

def validate_terminal_payload(
    plan: Mapping[str, Any],
    terminal: Mapping[str, Any],
    claim: Mapping[str, Any],
    leaf: Mapping[str, Any],
    *,
    scope: str,
    reviewed_authority_file_sha256: str,
    preflight_file_sha256: str,
    authorization_file_sha256: str,
    verdict_file_sha256: str,
    execution_claim_file_sha256: str,
) -> dict[str, Any]:
    schema = _control_schema(plan, scope, "terminal")
    supplied = dict(terminal)
    require_exact_keys(
        supplied,
        schema["exact_top_level_fields"],
        code="TERMINAL_OUTCOME_RECONSTRUCTION_FAILED",
    )
    verify_self_hash(
        supplied,
        "terminal_claim_payload_sha256",
        code="TERMINAL_OUTCOME_RECONSTRUCTION_FAILED",
    )
    expected = build_terminal_payload(
        plan,
        claim,
        leaf,
        scope=scope,
        outcome_kind=supplied.get("outcome_kind"),
        claimed_at_utc=supplied.get("claimed_at_utc"),
        reviewed_authority_file_sha256=reviewed_authority_file_sha256,
        preflight_file_sha256=preflight_file_sha256,
        authorization_file_sha256=authorization_file_sha256,
        verdict_file_sha256=verdict_file_sha256,
        execution_claim_file_sha256=execution_claim_file_sha256,
    )
    if supplied != expected:
        _raise("TERMINAL_OUTCOME_RECONSTRUCTION_FAILED")
    return supplied




def matching_failure_row(
    plan: Mapping[str, Any],
    *,
    scope: str,
    source_state: str,
    failure_phase: str,
    failure_code: str,
) -> dict[str, Any]:
    matches = [
        row
        for row in plan["failure_matrix"]
        if row["scope"] == scope
        and row["source_state"] == source_state
        and row["failure_phase"] == failure_phase
        and row["failure_code"] == failure_code
    ]
    if len(matches) != 1:
        _raise("INTERNAL_SANITIZED_FAILURE")
    return dict(matches[0])


def apply_availability_profile(
    plan: Mapping[str, Any],
    *,
    scope: str,
    profile_name: str,
    evidence: Mapping[str, str | None],
) -> dict[str, str | None]:
    profile = plan["failure_evidence_availability_profiles"][scope].get(
        profile_name
    )
    if not isinstance(profile, dict) or profile.get(
        "inherit_exact_terminal_claim_evidence"
    ):
        _raise("INTERNAL_SANITIZED_FAILURE")
    field_map = {
        "pre_complete_surface": "pre_complete_surface_sha256",
        "pre_protected_surface": "pre_protected_surface_sha256",
        "post_complete_surface": "post_complete_surface_sha256",
        "post_protected_surface": "post_protected_surface_sha256",
        "baseline_commitment_surface": "baseline_commitment_surface_sha256",
        "recomputed_baseline_commitment_surface": (
            "recomputed_baseline_commitment_surface_sha256"
        ),
    }
    result: dict[str, str | None] = {}
    for availability_key, receipt_field in field_map.items():
        if availability_key not in profile:
            continue
        available = profile[availability_key]
        value = evidence.get(receipt_field)
        if available:
            if not is_sha256(value):
                _raise("INTERNAL_SANITIZED_FAILURE")
            result[receipt_field] = value
        else:
            if value is not None:
                _raise("INTERNAL_SANITIZED_FAILURE")
            result[receipt_field] = None
    return result
