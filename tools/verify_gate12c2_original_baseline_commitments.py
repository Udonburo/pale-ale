#!/usr/bin/env python3
"""Independently rederive Gate12C-2 v0.9 baseline commitments."""


from __future__ import annotations

import argparse
import calendar
import copy
import ctypes
import ctypes.wintypes
import hashlib
import importlib.util
import json
import math
import os
import re
import socket
import subprocess
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence


_VERIFIER_BOOTSTRAP_RECORD: dict[str, Any] | None = globals().get(
    "_VERIFIER_BOOTSTRAP_RECORD"
)
_RETAINED_SELF_BOOTSTRAPPED = bool(
    globals().get("_RETAINED_SELF_BOOTSTRAPPED", False)
)


PLAN_PATH = Path(
    r"C:\Users\aoika\Documents\Research\pale-ale-local\research-program"
    r"\profile-plans\C2_ORIGINAL_BASELINE_COMMITMENT_GATE_PLAN_v0.9_2026-08-01.json"
)
PLAN_FILE_SHA256 = "ae7979779675fc032b2919a4db0887d60011dcdc510a43deef43fc090e5ef03c"
PLAN_PAYLOAD_SHA256 = "2bfaeb778494772a21975d0de22c6cc5221edbb742787fb1a34db5544eb24621"
CONTRACT_FILE_SHA256 = "794cf22c89375424a3485a2f441ec53fc930f016d470e4cc95778cb2528ae82d"
ARTIFACT_SURFACE_SHA256 = "7018f1a71ca8f41783f6228a7b73d25a28a37ba05462fbe7477f3fc76d1f5e2e"
R2_AUTHORITY_NAMESPACE_ID = "R2_6e92079"
R2_PLAN_PATH = Path(__file__).with_name(
    "gate12c2_original_baseline_r2_activation_plan.json"
)
R2_PLAN_FILE_SHA256 = "ae9a9a7f46660e3ec846767c8cc01ecf7ecd7486a823b1b1491fd0189bf8e02a"
R2_PLAN_PAYLOAD_SHA256 = "bef6f4e29f9fc61b95b40f3a3daa8b332779a4a26f7d04fa6bfac3edd3492c00"
R2_ARTIFACT_SURFACE_SHA256 = "23c1595bc504ae3e500695679cfdb2f060370b3b6bb56652eee9d5aac0637c0b"
R2_OCCUPIED_SURFACE_SHA256 = "2291c234118e033d60faacfc7181c7a5b8ba4890eb0af7cc9a43f61975111438"
R2_TASK1_COMMIT = "6e92079cc962a9748d2c69147186aed2a59da8d0"
R2_TASK1_PARENT = "2e51e0727d456792a474e38d67b1e3ebc605a8aa"
FORMAL_DESIGN_REVIEW_FILE_SHA256 = (
    "5a0ba0d6ad6b5b79df819e73d7ab15831c081ad5e4e44ca6b8195e59bc97cc1e"
)
FORMAL_DESIGN_REVIEW_PAYLOAD_SHA256 = (
    "fafcbb29e998e2e1a7d9c6351e0d2a925e9c652d01dd3236c13e041ac5eac3e2"
)
CONFIGURATION_SURFACE_SHA256 = "a564c25f28e42860f0a1e8f51d4a311b4eae2b771f02dc3f62504547799f19cf"
COMPLETE_SURFACE_SHA256 = "9489e0eb14e33a328167c840a443a80392c022e365bdefd41458c9659aeda6da"
PROTECTED_SURFACE_SHA256 = "a8ef2eb83fbd0517740f5ebbb2c270ba8f4ea37f872b34d137b0447fbb6edc24"
PROTECTED_ROOT = Path(
    r"C:\Users\aoika\Documents\Research\pale-ale-local\throughput"
    r"\C2_DRAW_PROFILE_f9bd14d_2026-07-26"
)
AUTHORIZED_IMPLEMENTATION_REPOSITORY = Path(
    r"C:\Users\aoika\Documents\GitHub\pale-ale"
)
REMEDIATION_BASE_COMMIT = (
    "2e51e0727d456792a474e38d67b1e3ebc605a8aa"
)
GATE_ID = "C2_ORIGINAL_BASELINE_COMMITMENT_GATE_v0.9"
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
IMPLEMENTATION_TRUST_MODEL_SHA256 = (
    "41247cd55b90fb4dc7dfb0d37f0154e7847e32a336fa0fcdb6e1d5bd6b5b944f"
)
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
VERIFIER_SCHEMA_INVARIANT_MANIFEST = (
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


def _trust_model_sha256(plan: Mapping[str, Any]) -> str:
    trust = plan.get("implementation_trust_model_contract")
    if type(trust) is not dict:
        _fail("INPUT_LINEAGE_MISMATCH")
    computed = _digest(trust)
    if (
        computed != IMPLEMENTATION_TRUST_MODEL_SHA256
        or plan.get("implementation_trust_model_sha256") != computed
    ):
        _fail("INPUT_LINEAGE_MISMATCH")
    return computed


def _review_surface_identity(plan: Mapping[str, Any]) -> dict[str, Any]:
    binding = plan.get("implementation_binding_contract")
    if type(binding) is not dict:
        _fail("INPUT_LINEAGE_MISMATCH")
    dependencies = binding.get("scientific_dependencies")
    if type(dependencies) is not list:
        _fail("INPUT_LINEAGE_MISMATCH")
    dependency_digest = _digest(dependencies)
    if dependency_digest != REVIEW_SURFACE_SOURCE_DEPENDENCIES_SHA256:
        _fail("INPUT_LINEAGE_MISMATCH")
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
    computed = _digest(payload)
    if computed != REVIEW_SURFACE_IDENTITY_SHA256:
        _fail("INPUT_LINEAGE_MISMATCH")
    return {
        **payload,
        "review_surface_identity_sha256": computed,
    }


def _validate_review_surface(
    plan: Mapping[str, Any], supplied: object
) -> dict[str, Any]:
    expected = _review_surface_identity(plan)
    if type(supplied) is not dict or set(supplied) != set(expected):
        _fail("INPUT_LINEAGE_MISMATCH")
    if supplied != expected:
        _fail("INPUT_LINEAGE_MISMATCH")
    return supplied


def _artifact_fields(
    plan: Mapping[str, Any], role: str
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
        _fail("INPUT_LINEAGE_MISMATCH")
    result = list(fields)
    if "review_surface_identity" in result:
        _fail("INPUT_LINEAGE_MISMATCH")
    self_hash = result.pop()
    if type(self_hash) is not str or not self_hash.endswith("_payload_sha256"):
        _fail("INPUT_LINEAGE_MISMATCH")
    result.extend(("review_surface_identity", self_hash))
    return tuple(result)


def _check_s2_component_surface(
    endpoint: Mapping[str, Any], expected_count: int
) -> None:
    if type(expected_count) is not int or expected_count < 1:
        _fail("INPUT_SCHEMA_INVALID")
    medians = endpoint.get("component_medians")
    coverage = endpoint.get("component_coverage")
    if type(medians) is not dict or type(coverage) is not dict:
        _fail("INPUT_SCHEMA_INVALID")
    _keys(medians, set(S2_COMPONENT_ARMS))
    _keys(coverage, set(S2_COMPONENT_ARMS))
    for arm in S2_COMPONENT_ARMS:
        arm_medians = medians.get(arm)
        arm_coverage = coverage.get(arm)
        if type(arm_medians) is not dict or type(arm_coverage) is not dict:
            _fail("INPUT_SCHEMA_INVALID")
        _keys(arm_medians, set(S2_COMPONENT_FIELDS))
        _keys(arm_coverage, set(S2_COMPONENT_FIELDS))
        for field_name in S2_COMPONENT_FIELDS:
            median = arm_medians[field_name]
            if median is not None and (
                type(median) is not float
                or not math.isfinite(median)
                or (
                    median == 0.0
                    and math.copysign(1.0, median) == -1.0
                )
            ):
                _fail("INPUT_SCHEMA_INVALID")
            counts = arm_coverage.get(field_name)
            if type(counts) is not dict:
                _fail("INPUT_SCHEMA_INVALID")
            _keys(counts, set(S2_COMPONENT_COUNT_FIELDS))
            checked = {
                name: _integer(counts.get(name))
                for name in S2_COMPONENT_COUNT_FIELDS
            }
            declared_expected = checked["expected_count"]
            defined = checked["defined_count"]
            degenerate = checked["degenerate_count"]
            missing = checked["unexpected_missing_count"]
            nonfinite = checked["nonfinite_count"]
            if declared_expected != expected_count:
                _fail("INPUT_SCHEMA_INVALID")
            if (
                defined
                + degenerate
                + missing
                + nonfinite
                != expected_count
            ):
                _fail("INPUT_SCHEMA_INVALID")
            if missing != 0 or nonfinite != 0:
                _fail("INPUT_SCHEMA_INVALID")
            if field_name in S2_ALWAYS_DEFINED_COMPONENT_FIELDS:
                if (
                    defined != expected_count
                    or degenerate != 0
                    or median is None
                ):
                    _fail("INPUT_SCHEMA_INVALID")
            else:
                if defined == 0:
                    if degenerate != expected_count or median is not None:
                        _fail("INPUT_SCHEMA_INVALID")
                elif median is None or defined + degenerate != expected_count:
                    _fail("INPUT_SCHEMA_INVALID")


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
    _integer(pipeline.get("endpoint_count"), 24)
    _boolean(pipeline.get("partial_or_structured_is_promotional"))
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
            if type(endpoint) is not dict:
                _fail("INPUT_SCHEMA_INVALID")
            if endpoint.get("minimum_log_null_inflation") != threshold:
                _fail("INPUT_SCHEMA_INVALID")
            expected_blocks = _integer(endpoint.get("expected_block_count"))
            if expected_blocks < 1:
                _fail("INPUT_SCHEMA_INVALID")
            _check_s2_component_surface(endpoint, expected_blocks * inner_count)
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
        or index.get("epistemic_status") != "development_shard_index_only"
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
    worker_count = _integer(index.get("worker_count_operational_only"))
    if worker_count < 1:
        _fail("INPUT_SCHEMA_INVALID")
    if "operational_execution_metrics" in index:
        operational = index["operational_execution_metrics"]
        if type(operational) is not dict:
            _fail("INPUT_SCHEMA_INVALID")
        _finite_tree(operational)
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
            or shard.get("epistemic_status")
            != "development_outer_shard_only"
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
        if "operational_metrics" in index_row:
            operational = index_row["operational_metrics"]
            if type(operational) is not dict:
                _fail("INPUT_SCHEMA_INVALID")
            _finite_tree(operational)
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


def independent_windows_ordinal_equal(
    left: str, right: str, api: object
) -> bool:
    compare = getattr(api, "CompareStringOrdinal", None)
    if compare is None:
        _fail("READ_ONLY_HANDLE_FAILED")
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
        _fail("READ_ONLY_HANDLE_FAILED")
    if result == 0:
        _fail("READ_ONLY_HANDLE_FAILED")
    return result == 2


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
        if not independent_windows_ordinal_equal(final, str(expected), self.dll):
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


class _IndependentEntryLoader:
    def __init__(self, origin: str) -> None:
        self.origin = origin

    def get_filename(self, _name: str) -> str:
        return self.origin


class _IndependentRetainedEntryLoader(_IndependentEntryLoader):
    def __init__(self, origin: str, raw: bytes) -> None:
        super().__init__(origin)
        self.raw = raw

    def create_module(self, _spec: object) -> None:
        return None

    def exec_module(self, module: object) -> None:
        exec(
            compile(self.raw, self.origin, "exec", dont_inherit=True),
            module.__dict__,
        )


class IndependentExecutingCodeIdentity:
    """Independent retained identity for the standalone verifier entry code."""

    def __init__(
        self,
        plan: Mapping[str, Any],
        candidate: Mapping[str, Any],
        *,
        entry_path: Path,
        repository_argument: Path,
        entry_module: object,
        module_registry: Mapping[str, object] | None = None,
        authorized_root: Path = AUTHORIZED_IMPLEMENTATION_REPOSITORY,
        api: object | None = None,
        git_head_reader: Callable[[Path], str] | None = None,
        bootstrap_record: Mapping[str, Any] | None = None,
    ) -> None:
        self.plan = dict(plan)
        self.candidate = dict(candidate)
        self.registry = sys.modules if module_registry is None else module_registry
        self.entry_module = entry_module
        self.entry_path = Path(entry_path)
        self.repository_argument = Path(repository_argument)
        self.authorized_root = Path(authorized_root)
        self.source_commit = str(candidate.get("source_commit"))
        self.object_format = str(candidate.get("git_object_format"))
        self.git_head_reader = git_head_reader or self._git_head
        self.bootstrap_record = bootstrap_record
        self.io = VerifierRetainedSurface(
            self.entry_path.parent,
            {"files": [], "directories": []},
            api=api,
        )
        self.root: Path | None = None
        self.root_record: tuple[int, tuple[int, bytes, str, int | None]] | None = None
        self.entry_record: tuple[int, tuple[int, bytes, str, int | None]] | None = None
        self.entry_row: dict[str, Any] | None = None
        self.checkpoints = tuple(
            plan["executing_code_identity_contract"]["checkpoints"]
        )
        self._checkpoint_index = 0
        self._owned_handles: set[int] = set()
        self._closed = False
        try:
            self._initialize()
        except Exception:
            self.close()
            _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")

    @staticmethod
    def _git_head(root: Path) -> str:
        def read_control(path: Path) -> bytes:
            try:
                metadata = os.lstat(path)
                if getattr(metadata, "st_file_attributes", 0) & 0x400:
                    _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
                raw = path.read_bytes()
            except OSError:
                _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
            if len(raw) > 1 << 20:
                _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
            return raw

        try:
            git_directory = root / ".git"
            git_metadata = os.lstat(git_directory)
            if (
                not git_directory.is_dir()
                or getattr(git_metadata, "st_file_attributes", 0) & 0x400
            ):
                _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
            head_raw = read_control(git_directory / "HEAD")
            if (
                not head_raw.endswith(b"\n")
                or head_raw.count(b"\n") != 1
                or b"\r" in head_raw
            ):
                _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
            head_text = head_raw[:-1].decode("ascii", "strict")
        except (OSError, UnicodeDecodeError):
            _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        if re.fullmatch(r"[0-9a-f]{40}", head_text) is not None:
            return head_text
        if not head_text.startswith("ref: refs/"):
            _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        reference = head_text[5:]
        reference_path = PurePosixPath(reference)
        if (
            reference_path.is_absolute()
            or any(part in {"", ".", ".."} for part in reference_path.parts)
            or "\\" in reference
        ):
            _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        loose_path = git_directory.joinpath(*reference_path.parts)
        if loose_path.exists():
            loose_raw = read_control(loose_path)
            if (
                not loose_raw.endswith(b"\n")
                or loose_raw.count(b"\n") != 1
                or b"\r" in loose_raw
            ):
                _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
            try:
                value = loose_raw[:-1].decode("ascii", "strict")
            except UnicodeDecodeError:
                _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        else:
            packed_raw = read_control(git_directory / "packed-refs")
            try:
                packed_lines = packed_raw.decode("ascii", "strict").splitlines()
            except UnicodeDecodeError:
                _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
            matches = [
                line.split(" ", 1)[0]
                for line in packed_lines
                if " " in line and line.split(" ", 1)[1] == reference
            ]
            if len(matches) != 1:
                _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
            value = matches[0]
        if re.fullmatch(r"[0-9a-f]{40}", value) is None:
            _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        return value

    def _same_path(self, left: str | Path, right: str | Path) -> bool:
        return independent_windows_ordinal_equal(
            str(left), str(right), self.io.dll
        )

    def _candidate_paths(self) -> dict[str, dict[str, Any]]:
        rows = self.candidate.get("implementation_files")
        if type(rows) is not list or len(rows) != 10:
            _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        result = {str(row.get("relative_path")): dict(row) for row in rows}
        if len(result) != 10:
            _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        return result

    def _verify_entry_module(self) -> None:
        if self.root is None or self.entry_row is None or self.entry_record is None:
            _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        if self.registry.get("__main__") is not self.entry_module:
            _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        expected = self.root / self.entry_row["relative_path"]
        module_file = getattr(self.entry_module, "__file__", None)
        if type(module_file) is not str or not self._same_path(module_file, expected):
            _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        spec = getattr(self.entry_module, "__spec__", None)
        if spec is None:
            loader = _IndependentEntryLoader(self.entry_record[1][2])
            spec = type("IndependentEntrySpec", (), {})()
            spec.origin = self.entry_record[1][2]
            spec.loader = loader
            self.entry_module.__spec__ = spec
        if (
            type(getattr(spec, "origin", None)) is not str
            or getattr(spec, "loader", None) is None
            or not self._same_path(spec.origin, expected)
        ):
            _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")

    def _scan_aliases(self, rows: Mapping[str, Mapping[str, Any]]) -> None:
        if self.root is None:
            _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        for name, module in tuple(self.registry.items()):
            module_file = getattr(module, "__file__", None)
            if type(module_file) is not str:
                continue
            matches = [
                relative
                for relative in rows
                if self._same_path(module_file, self.root / relative)
            ]
            if matches and (
                len(matches) != 1
                or name != "__main__"
                or module is not self.entry_module
                or matches[0] != self.entry_row["relative_path"]
            ):
                _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")

    def _initialize(self) -> None:
        if (
            re.fullmatch(r"[0-9a-f]{40}", self.source_commit) is None
            or self.object_format not in {"sha1", "sha256"}
        ):
            _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        rows = self._candidate_paths()
        if self.bootstrap_record is None:
            entry_handle = self.io._open_handle(self.entry_path, False)
            self._owned_handles.add(entry_handle)
            entry_metadata = self.io._metadata(
                entry_handle, self.entry_path, False
            )
            retained_root_record = None
        else:
            bootstrap_surface = self.bootstrap_record.get("surface")
            retained_entry = self.bootstrap_record.get("entry_record")
            retained_root_record = self.bootstrap_record.get("root_record")
            if (
                bootstrap_surface is None
                or type(retained_entry) is not tuple
                or len(retained_entry) != 2
                or type(retained_root_record) is not tuple
                or len(retained_root_record) != 2
            ):
                _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
            self.io = bootstrap_surface
            entry_handle = retained_entry[0]
            self._owned_handles.add(entry_handle)
            entry_metadata = self.io._metadata(
                entry_handle, self.entry_path, False
            )
            if entry_metadata != retained_entry[1]:
                _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        self.root = Path(entry_metadata[2]).parent.parent
        if (
            not self._same_path(self.root, self.authorized_root)
            or not self._same_path(self.repository_argument, self.root)
            or self.candidate.get("authorized_implementation_repository")
            != str(AUTHORIZED_IMPLEMENTATION_REPOSITORY)
        ):
            _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        if retained_root_record is None:
            root_handle = self.io._open_handle(self.root, True)
            self._owned_handles.add(root_handle)
            root_metadata = self.io._metadata(root_handle, self.root, True)
        else:
            root_handle = retained_root_record[0]
            self._owned_handles.add(root_handle)
            root_metadata = self.io._metadata(root_handle, self.root, True)
            if root_metadata != retained_root_record[1]:
                _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        self.root_record = (root_handle, root_metadata)
        matches = [
            relative
            for relative in rows
            if self._same_path(entry_metadata[2], self.root / relative)
        ]
        if matches != ["tools/verify_gate12c2_original_baseline_commitments.py"]:
            _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        self.entry_row = rows[matches[0]]
        self.entry_record = (entry_handle, entry_metadata)
        raw = self.io._read(self.entry_record)
        if (
            verifier_sha256(raw) != self.entry_row.get("file_sha256")
            or _git_blob(raw, self.object_format)
            != self.entry_row.get("git_blob_oid")
            or root_metadata[0] != entry_metadata[0]
        ):
            _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        self._verify_entry_module()
        self._scan_aliases(rows)

    def checkpoint(self, checkpoint: str) -> dict[str, str]:
        try:
            if (
                self.root is None
                or self.root_record is None
                or self.entry_record is None
                or self.entry_row is None
                or self._checkpoint_index >= len(self.checkpoints)
                or checkpoint != self.checkpoints[self._checkpoint_index]
            ):
                _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
            if self.io._metadata(
                self.root_record[0], self.root, True
            ) != self.root_record[1]:
                _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
            expected_entry = self.root / self.entry_row["relative_path"]
            if self.io._metadata(
                self.entry_record[0], expected_entry, False
            ) != self.entry_record[1]:
                _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
            raw = self.io._read(self.entry_record)
            if (
                verifier_sha256(raw) != self.entry_row["file_sha256"]
                or _git_blob(raw, self.object_format)
                != self.entry_row["git_blob_oid"]
            ):
                _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
            self._verify_entry_module()
            self._scan_aliases(self._candidate_paths())
            head = self.git_head_reader(self.root)
            if head != self.source_commit:
                _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
            surface = [
                {
                    "module_name": "__main__",
                    "role": self.entry_row["role"],
                    "relative_path": self.entry_row["relative_path"],
                    "file_id_sha256": verifier_sha256(self.entry_record[1][1]),
                    "file_sha256": self.entry_row["file_sha256"],
                    "git_blob_oid": self.entry_row["git_blob_oid"],
                }
            ]
            self._checkpoint_index += 1
            return {
                "git_head": head,
                "executing_code_identity_surface_sha256": _digest(surface),
            }
        except Exception:
            _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        handles: list[int] = list(self._owned_handles)
        if self.entry_record is not None:
            handles.append(self.entry_record[0])
        if self.root_record is not None:
            handles.append(self.root_record[0])
        for handle in dict.fromkeys(handles):
            try:
                self.io.dll.CloseHandle(handle)
            except Exception:
                pass
        self.entry_record = None
        self.root_record = None
        self._owned_handles.clear()



def _independent_load_base_plan() -> dict[str, Any]:
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
        "gate12c2_original_baseline_commitment_gate_plan_v0.9"
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
    _trust_model_sha256(plan)
    _review_surface_identity(plan)
    return plan



def _replace_active_surface(
    value: object,
    old: str,
    new: str,
) -> object:
    if type(value) is dict:
        return {
            key: (
                copy.deepcopy(item)
                if key.startswith("historical_v0_")
                else _replace_active_surface(item, old, new)
            )
            for key, item in value.items()
        }
    if type(value) is list:
        return [_replace_active_surface(item, old, new) for item in value]
    return new if value == old else copy.deepcopy(value)


def _merge_r2_delta(
    target: dict[str, Any],
    delta: Mapping[str, Any],
) -> None:
    for key, value in delta.items():
        if type(value) is dict and type(target.get(key)) is dict:
            _merge_r2_delta(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _legacy_r2_receipt(
    row: Mapping[str, Any],
    hash_field: str,
) -> dict[str, Any]:
    try:
        raw = Path(row["path"]).read_bytes()
    except OSError:
        _fail("INPUT_LINEAGE_MISMATCH")
    if (
        verifier_sha256(raw) != row["file_sha256"]
        or not raw.endswith(b"\n")
        or raw.endswith(b"\n\n")
    ):
        _fail("INPUT_LINEAGE_MISMATCH")
    payload = verifier_json(raw[:-1])
    if (
        raw != verifier_canonical_bytes(payload) + b"\n"
        or _self_hash(payload, hash_field) != row["payload_sha256"]
    ):
        _fail("INPUT_LINEAGE_MISMATCH")
    return payload


def _independent_load_r2_plan(
    base_plan: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        raw = R2_PLAN_PATH.read_bytes()
    except OSError:
        _fail("INPUT_LINEAGE_MISMATCH")
    if (
        verifier_sha256(raw) != R2_PLAN_FILE_SHA256
        or not raw.endswith(b"\n")
        or raw.endswith(b"\n\n")
    ):
        _fail("INPUT_LINEAGE_MISMATCH")
    r2 = verifier_json(raw[:-1])
    if (
        raw != verifier_canonical_bytes(r2) + b"\n"
        or _self_hash(r2, "r2_activation_plan_payload_sha256")
        != R2_PLAN_PAYLOAD_SHA256
        or r2.get("schema_version")
        != "gate12c2_original_baseline_r2_activation_plan_v0.1"
        or r2.get("namespace_id") != R2_AUTHORITY_NAMESPACE_ID
        or r2.get("state") != "R2_CONTROL_LINEAGE_FROZEN"
        or r2.get("artifact_path_surface_sha256")
        != R2_ARTIFACT_SURFACE_SHA256
        or r2.get("occupied_v0_9_surface_sha256")
        != R2_OCCUPIED_SURFACE_SHA256
        or _digest(r2.get("occupied_v0_9"))
        != R2_OCCUPIED_SURFACE_SHA256
    ):
        _fail("INPUT_LINEAGE_MISMATCH")
    expected_lineage = {
        "activation_commit_parent": R2_TASK1_COMMIT,
        "activation_commit_parent_count": 1,
        "task1_commit": R2_TASK1_COMMIT,
        "task1_parent": R2_TASK1_PARENT,
        "task1_parent_count": 1,
    }
    if (
        r2.get("activation_lineage") != expected_lineage
        or r2.get("base_plan")
        != {
            "file_sha256": PLAN_FILE_SHA256,
            "path": str(PLAN_PATH),
            "payload_sha256": PLAN_PAYLOAD_SHA256,
        }
        or r2.get("base_contract", {}).get("file_sha256")
        != CONTRACT_FILE_SHA256
        or r2.get("preserved_identities")
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
    ):
        _fail("INPUT_LINEAGE_MISMATCH")
    rows = r2.get("artifact_path_surface")
    old_rows = {
        row["role"]: row for row in base_plan["artifact_path_surface"]
    }
    if (
        type(rows) is not list
        or len(rows) != 18
        or rows != sorted(rows, key=lambda row: row.get("role", ""))
        or _digest(rows) != R2_ARTIFACT_SURFACE_SHA256
        or {row.get("role") for row in rows} != set(old_rows)
    ):
        _fail("INPUT_LINEAGE_MISMATCH")
    paths: set[str] = set()
    old_paths = {
        path
        for row in old_rows.values()
        for path in (row["final_path"], row["pending_path"])
    }
    for row in rows:
        if (
            type(row) is not dict
            or set(row)
            != {
                "role",
                "final_path",
                "pending_path",
                "publication_mode",
                "lifecycle_scope",
            }
            or type(row["final_path"]) is not str
            or type(row["pending_path"]) is not str
            or row["final_path"] in paths
            or row["pending_path"] in paths
            or row["publication_mode"]
            != "MoveFileExW_nonreplace_write_through"
            or row["lifecycle_scope"]
            != old_rows[row["role"]]["lifecycle_scope"]
        ):
            _fail("INPUT_LINEAGE_MISMATCH")
        paths.update((row["final_path"], row["pending_path"]))
        if row["role"] == "formal_design_review_verdict":
            if row != old_rows[row["role"]]:
                _fail("INPUT_LINEAGE_MISMATCH")
        elif (
            R2_AUTHORITY_NAMESPACE_ID not in Path(row["final_path"]).name
            or row["pending_path"]
            != row["final_path"] + ".pending-" + R2_AUTHORITY_NAMESPACE_ID
            or row["final_path"] in old_paths
            or row["pending_path"] in old_paths
        ):
            _fail("INPUT_LINEAGE_MISMATCH")
    for name in (
        "candidate_manifest_contract",
        "clean_restore_receipt_contract",
        "fresh_review_evidence_contract",
    ):
        contract = r2.get(name)
        if type(contract) is not dict:
            _fail("INPUT_LINEAGE_MISMATCH")
        final_path = contract.get("artifact_path")
        pending_path = contract.get("pending_path")
        if (
            type(final_path) is not str
            or type(pending_path) is not str
            or final_path in paths
            or pending_path in paths
            or R2_AUTHORITY_NAMESPACE_ID not in Path(final_path).name
            or pending_path
            != final_path + ".pending-" + R2_AUTHORITY_NAMESPACE_ID
        ):
            _fail("INPUT_LINEAGE_MISMATCH")
        paths.update((final_path, pending_path))
    binding = r2.get("implementation_binding_contract_overlay")
    review = r2.get("fresh_review_contract_overlay")
    authority = r2.get("reviewed_authority_contract_overlay")
    role_rows = {row["role"]: row for row in rows}
    if (
        type(binding) is not dict
        or type(review) is not dict
        or type(authority) is not dict
        or binding.get("artifact_path")
        != role_rows["implementation_candidate_binding"]["final_path"]
        or review.get("artifact_path")
        != role_rows["fresh_implementation_review_verdict"]["final_path"]
        or authority.get("artifact_path")
        != role_rows["reviewed_implementation_authority"]["final_path"]
        or authority.get("fresh_implementation_review_path")
        != review.get("artifact_path")
        or binding.get("required_values", {}).get(
            "procedural_author_separation_precondition_satisfied"
        )
        is not False
        or binding.get("required_values", {}).get(
            "current_exposed_design_context_authored_final_bytes"
        )
        is not True
        or review.get("outcomes", {})
        .get("pass", {})
        .get("required_values", {})
        .get("procedural_author_separation_precondition_satisfied")
        is not False
        or review.get("outcomes", {})
        .get("pass", {})
        .get("required_values", {})
        .get("current_exposed_design_context_authored_final_bytes")
        is not True
    ):
        _fail("INPUT_LINEAGE_MISMATCH")
    occupied = r2["occupied_v0_9"]
    old_candidate = _legacy_r2_receipt(
        occupied["candidate_binding"],
        "implementation_candidate_binding_payload_sha256",
    )
    old_review = _legacy_r2_receipt(
        occupied["review_verdict"],
        "fresh_implementation_review_payload_sha256",
    )
    if (
        old_candidate.get("source_commit")
        != occupied["candidate_binding"]["source_commit"]
        or old_review.get("implementation_source_commit")
        != occupied["review_verdict"]["source_commit"]
        or old_review.get("state")
        != "FRESH_IMPLEMENTATION_REVIEW_REOPEN"
        or old_review.get("P0_count") != 0
        or old_review.get("P1_count") != 1
        or old_review.get("P2_count") != 0
    ):
        _fail("INPUT_LINEAGE_MISMATCH")
    for role in occupied["required_absent_final_roles"]:
        if Path(old_rows[role]["final_path"]).exists():
            _fail("UNEXPECTED_ARTIFACT")
    for role in occupied["required_absent_pending_roles"]:
        if Path(old_rows[role]["pending_path"]).exists():
            _fail("UNEXPECTED_ARTIFACT")
    return r2


def independent_load_plan() -> dict[str, Any]:
    base = _independent_load_base_plan()
    r2 = _independent_load_r2_plan(base)
    active = _replace_active_surface(
        base,
        ARTIFACT_SURFACE_SHA256,
        R2_ARTIFACT_SURFACE_SHA256,
    )
    if type(active) is not dict:
        _fail("INPUT_LINEAGE_MISMATCH")
    active["artifact_path_surface"] = copy.deepcopy(
        r2["artifact_path_surface"]
    )
    active["artifact_path_surface_sha256"] = R2_ARTIFACT_SURFACE_SHA256
    pending_by_role = {
        row["role"]: row["pending_path"]
        for row in r2["artifact_path_surface"]
    }
    for row in active["artifact_lifecycle_contract"][
        "full_surface_checkpoint_contract"
    ]["pending_injection_tests"]:
        row["injected_pending_path"] = pending_by_role[row["role"]]
    _merge_r2_delta(
        active["implementation_binding_contract"],
        r2["implementation_binding_contract_overlay"],
    )
    _merge_r2_delta(
        active["review_receipt_schemas"][
            "fresh_implementation_review_verdict"
        ],
        r2["fresh_review_contract_overlay"],
    )
    _merge_r2_delta(
        active["reviewed_implementation_authority_contract"],
        r2["reviewed_authority_contract_overlay"],
    )
    active["r2_activation_control"] = {
        "authority_namespace_id": R2_AUTHORITY_NAMESPACE_ID,
        "activation_plan_file_sha256": R2_PLAN_FILE_SHA256,
        "activation_plan_payload_sha256": R2_PLAN_PAYLOAD_SHA256,
        "occupied_v0_9_surface_sha256": R2_OCCUPIED_SURFACE_SHA256,
        "activation_lineage": copy.deepcopy(r2["activation_lineage"]),
        "candidate_manifest_contract": copy.deepcopy(
            r2["candidate_manifest_contract"]
        ),
        "clean_restore_receipt_contract": copy.deepcopy(
            r2["clean_restore_receipt_contract"]
        ),
        "fresh_review_evidence_contract": copy.deepcopy(
            r2["fresh_review_evidence_contract"]
        ),
        "fresh_review_packet_path": r2["fresh_review_packet_path"],
    }
    if (
        _digest(active["artifact_path_surface"])
        != R2_ARTIFACT_SURFACE_SHA256
        or _trust_model_sha256(active) != IMPLEMENTATION_TRUST_MODEL_SHA256
        or _review_surface_identity(active)[
            "review_surface_identity_sha256"
        ]
        != REVIEW_SURFACE_IDENTITY_SHA256
    ):
        _fail("INPUT_LINEAGE_MISMATCH")
    return active


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


def _git_parent_lineage(
    repository: Path,
    source_commit: str,
) -> tuple[str, ...]:
    try:
        completed = subprocess.run(
            (
                "git",
                "rev-list",
                "--parents",
                "-n",
                "1",
                source_commit,
            ),
            cwd=Path(repository),
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="ascii",
        )
    except (OSError, subprocess.SubprocessError):
        _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    lines = completed.stdout.splitlines(keepends=True)
    if len(lines) != 1 or not lines[0].endswith("\n"):
        _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    values = tuple(lines[0][:-1].split())
    if not values or any(
        re.fullmatch(r"[0-9a-f]{40}", value) is None for value in values
    ):
        _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    return values


def _git_path_blob_oid(
    repository: Path,
    source_commit: str,
    relative_path: str,
) -> str:
    try:
        completed = subprocess.run(
            ("git", "rev-parse", f"{source_commit}:{relative_path}"),
            cwd=Path(repository),
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="ascii",
        )
    except (OSError, subprocess.SubprocessError):
        _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    lines = completed.stdout.splitlines()
    if (
        len(lines) != 1
        or re.fullmatch(r"[0-9a-f]{40,64}", lines[0]) is None
    ):
        _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    return lines[0]


def _require_direct_child_lineage(
    source_commit: str,
    lineage: Sequence[str],
    *,
    expected_parent: str = REMEDIATION_BASE_COMMIT,
) -> None:
    if (
        len(lineage) != 2
        or lineage[0] != source_commit
        or lineage[1] != expected_parent
    ):
        _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")


def _independent_reviewed_authority_payload(
    plan: Mapping[str, Any],
    candidate: Mapping[str, Any],
    review: Mapping[str, Any],
    *,
    candidate_file_hash: str,
    review_file_hash: str,
) -> dict[str, Any]:
    binding_contract = plan["implementation_binding_contract"]
    authority_contract = plan["reviewed_implementation_authority_contract"]
    surface = _validate_review_surface(
        plan, candidate.get("review_surface_identity")
    )
    review_surface = _validate_review_surface(
        plan, review.get("review_surface_identity")
    )
    if review_surface != surface:
        _fail("INPUT_LINEAGE_MISMATCH")
    payload = {
        "schema_version": authority_contract["schema_version"],
        "authority_id": authority_contract["authority_id_value"],
        "state": authority_contract["state"],
        "authority_derivation_domain": authority_contract[
            "authority_derivation_domain"
        ],
        "implementation_source_commit": candidate["source_commit"],
        "authorized_implementation_repository": str(
            AUTHORIZED_IMPLEMENTATION_REPOSITORY
        ),
        "remediation_base_commit": binding_contract["required_values"][
            "remediation_base_commit"
        ],
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
        "implementation_trust_model_sha256": _trust_model_sha256(plan),
        "artifact_path_surface_sha256": plan["artifact_path_surface_sha256"],
        "review_surface_identity": surface,
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
    if "r2_activation_control" in plan:
        if (
            candidate.get("authority_namespace_id")
            != R2_AUTHORITY_NAMESPACE_ID
            or review.get("authority_namespace_id")
            != R2_AUTHORITY_NAMESPACE_ID
            or review.get("implementation_source_commit")
            != candidate.get("source_commit")
            or review.get("implementation_candidate_binding_file_sha256")
            != candidate_file_hash
            or review.get(
                "implementation_candidate_binding_payload_sha256"
            )
            != candidate.get(
                "implementation_candidate_binding_payload_sha256"
            )
        ):
            _fail("INPUT_LINEAGE_MISMATCH")
        payload.update(
            {
                "authority_namespace_id": R2_AUTHORITY_NAMESPACE_ID,
                "r2_activation_plan_file_sha256": R2_PLAN_FILE_SHA256,
                "r2_activation_plan_payload_sha256": (
                    R2_PLAN_PAYLOAD_SHA256
                ),
                "occupied_v0_9_surface_sha256": (
                    R2_OCCUPIED_SURFACE_SHA256
                ),
                "candidate_manifest_file_sha256": candidate.get(
                    "candidate_manifest_file_sha256"
                ),
                "candidate_manifest_payload_sha256": candidate.get(
                    "candidate_manifest_payload_sha256"
                ),
            }
        )
    if (
        set(payload) | {"reviewed_implementation_authority_payload_sha256"}
        != set(_artifact_fields(plan, "reviewed_implementation_authority"))
    ):
        _fail("INPUT_LINEAGE_MISMATCH")
    payload["reviewed_implementation_authority_payload_sha256"] = _digest(
        payload
    )
    return payload



def _independent_r2_candidate_manifest(
    plan: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    control = plan["r2_activation_control"]
    contract = control["candidate_manifest_contract"]
    manifest, file_hash = _receipt(
        Path(contract["artifact_path"]),
        contract["exact_top_level_fields"],
        "candidate_manifest_payload_sha256",
    )
    binding_required = plan["implementation_binding_contract"][
        "required_values"
    ]
    expected = {
        "schema_version": contract["schema_version"],
        "authority_namespace_id": R2_AUTHORITY_NAMESPACE_ID,
        "state": contract["state"],
        "activation_source_commit": candidate.get("source_commit"),
        "activation_parent_commit": binding_required[
            "remediation_base_commit"
        ],
        "task1_parent_commit": binding_required[
            "remediation_base_parent"
        ],
        "r2_activation_plan_file_sha256": R2_PLAN_FILE_SHA256,
        "r2_activation_plan_payload_sha256": R2_PLAN_PAYLOAD_SHA256,
        "artifact_path_surface_sha256": (
            plan["artifact_path_surface_sha256"]
        ),
        "occupied_v0_9_surface_sha256": R2_OCCUPIED_SURFACE_SHA256,
        "review_surface_identity_sha256": REVIEW_SURFACE_IDENTITY_SHA256,
        "implementation_trust_model_sha256": _trust_model_sha256(plan),
        "implementation_files": candidate.get("implementation_files"),
        "scientific_dependencies": candidate.get(
            "scientific_dependencies"
        ),
        "clean_restore": candidate.get("clean_restore"),
        "protected_payload_accessed": False,
        "scientific_values_inspected": False,
        "runtime_authorization_issued": False,
    }
    if (
        any(manifest.get(key) != value for key, value in expected.items())
        or candidate.get("candidate_manifest_file_sha256") != file_hash
        or candidate.get("candidate_manifest_payload_sha256")
        != manifest.get("candidate_manifest_payload_sha256")
    ):
        _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    return manifest, file_hash


def _independent_r2_restore_receipt(
    plan: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    control = plan["r2_activation_control"]
    contract = control["clean_restore_receipt_contract"]
    receipt, file_hash = _receipt(
        Path(contract["artifact_path"]),
        contract["exact_top_level_fields"],
        "restore_receipt_payload_sha256",
    )
    restore = candidate.get("clean_restore")
    binding_required = plan["implementation_binding_contract"][
        "required_values"
    ]
    if type(restore) is not dict:
        _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
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
    if (
        any(receipt.get(key) != value for key, value in expected.items())
        or type(receipt.get("restore_path")) is not str
        or not receipt["restore_path"]
        or restore.get("restore_receipt_file_sha256") != file_hash
        or restore.get("restore_receipt_payload_sha256")
        != receipt.get("restore_receipt_payload_sha256")
    ):
        _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    return receipt, file_hash


def _independent_r2_review_evidence(
    plan: Mapping[str, Any],
    candidate: Mapping[str, Any],
    candidate_file_hash: str,
) -> tuple[dict[str, Any], str]:
    control = plan["r2_activation_control"]
    contract = control["fresh_review_evidence_contract"]
    evidence, file_hash = _receipt(
        Path(contract["artifact_path"]),
        contract["exact_top_level_fields"],
        "review_evidence_payload_sha256",
    )
    if any(
        evidence.get(key) != value
        for key, value in contract["required_values"].items()
    ):
        _fail("INPUT_LINEAGE_MISMATCH")
    restore = candidate["clean_restore"]
    expected = {
        "implementation_source_commit": candidate["source_commit"],
        "implementation_candidate_binding_file_sha256": (
            candidate_file_hash
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
        "r2_activation_plan_file_sha256": R2_PLAN_FILE_SHA256,
        "r2_activation_plan_payload_sha256": R2_PLAN_PAYLOAD_SHA256,
        "artifact_path_surface_sha256": (
            plan["artifact_path_surface_sha256"]
        ),
        "review_surface_identity": _review_surface_identity(plan),
        "bundle_file_sha256": restore["bundle_file_sha256"],
        "restore_receipt_file_sha256": restore[
            "restore_receipt_file_sha256"
        ],
        "restore_receipt_payload_sha256": restore[
            "restore_receipt_payload_sha256"
        ],
        "authority_namespace_id": R2_AUTHORITY_NAMESPACE_ID,
        "contract_file_sha256": CONTRACT_FILE_SHA256,
        "plan_file_sha256": PLAN_FILE_SHA256,
        "plan_payload_sha256": PLAN_PAYLOAD_SHA256,
        "r2_activation_plan_file_sha256": R2_PLAN_FILE_SHA256,
        "r2_activation_plan_payload_sha256": R2_PLAN_PAYLOAD_SHA256,
        "artifact_path_surface_sha256": (
            plan["artifact_path_surface_sha256"]
        ),
        "occupied_v0_9_surface_sha256": R2_OCCUPIED_SURFACE_SHA256,
        "candidate_manifest_file_sha256": candidate[
            "candidate_manifest_file_sha256"
        ],
        "candidate_manifest_payload_sha256": candidate[
            "candidate_manifest_payload_sha256"
        ],
        "review_evidence_file_sha256": review_evidence_file_hash,
        "review_evidence_payload_sha256": review_evidence[
            "review_evidence_payload_sha256"
        ],
    }
    if any(evidence.get(key) != value for key, value in expected.items()):
        _fail("INPUT_LINEAGE_MISMATCH")
    for field in (
        "changed_file_manifest_sha256",
        "targeted_test_node_id_sha256",
        "full_suite_node_id_sha256",
    ):
        if (
            type(evidence.get(field)) is not str
            or SHA_RE.fullmatch(evidence[field]) is None
        ):
            _fail("INPUT_LINEAGE_MISMATCH")
    for field in ("targeted_test_count", "full_suite_test_count"):
        if (
            type(evidence.get(field)) is not int
            or evidence[field] < 1
        ):
            _fail("INPUT_LINEAGE_MISMATCH")
    return evidence, file_hash


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
        != CONTRACT_FILE_SHA256
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
        or type(formal.get("P2_count")) is not int
        or formal["P2_count"] != 0
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
        _artifact_fields(plan, "implementation_candidate_binding"),
        "implementation_candidate_binding_payload_sha256",
    )
    fixed_candidate = {
        "schema_version": binding_contract["schema_version"],
        "plan_file_sha256": PLAN_FILE_SHA256,
        "plan_payload_sha256": PLAN_PAYLOAD_SHA256,
        "r2_activation_plan_file_sha256": R2_PLAN_FILE_SHA256,
        "r2_activation_plan_payload_sha256": R2_PLAN_PAYLOAD_SHA256,
        "occupied_v0_9_surface_sha256": R2_OCCUPIED_SURFACE_SHA256,
        "artifact_path_surface_sha256": plan["artifact_path_surface_sha256"],
        "review_surface_identity": _review_surface_identity(plan),
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
            "implementation_trust_model_sha256": _trust_model_sha256(plan),
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
    expected_base = binding_contract["required_values"][
        "remediation_base_commit"
    ]
    expected_base_parent = binding_contract["required_values"][
        "remediation_base_parent"
    ]
    _require_direct_child_lineage(
        source_commit,
        _git_parent_lineage(repository, source_commit),
        expected_parent=expected_base,
    )
    _require_direct_child_lineage(
        expected_base,
        _git_parent_lineage(repository, expected_base),
        expected_parent=expected_base_parent,
    )
    activation_relative = (
        "tools/gate12c2_original_baseline_r2_activation_plan.json"
    )
    try:
        activation_raw = (repository / activation_relative).read_bytes()
    except OSError:
        _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    if (
        verifier_sha256(activation_raw) != R2_PLAN_FILE_SHA256
        or _git_path_blob_oid(
            repository, source_commit, activation_relative
        )
        != _git_blob(activation_raw, object_format)
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
    _independent_r2_restore_receipt(plan, candidate)
    _independent_r2_candidate_manifest(plan, candidate)
    review_schema = plan["review_receipt_schemas"][
        "fresh_implementation_review_verdict"
    ]
    review, review_file_hash = _receipt(
        Path(review_schema["artifact_path"]),
        _artifact_fields(plan, "fresh_implementation_review_verdict"),
        "fresh_implementation_review_payload_sha256",
    )
    review_evidence, review_evidence_file_hash = (
        _independent_r2_review_evidence(
            plan,
            candidate,
            candidate_file_hash,
        )
    )
    candidate_surface = _validate_review_surface(
        plan, candidate.get("review_surface_identity")
    )
    review_surface = _validate_review_surface(
        plan, review.get("review_surface_identity")
    )
    if review_surface != candidate_surface:
        _fail("INPUT_LINEAGE_MISMATCH")
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
        "implementation_trust_model_sha256": _trust_model_sha256(plan),
        "review_surface_identity": candidate_surface,
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
        _artifact_fields(plan, "reviewed_implementation_authority"),
        "reviewed_implementation_authority_payload_sha256",
    )
    expected_authority = _independent_reviewed_authority_payload(
        plan,
        candidate,
        review,
        candidate_file_hash=candidate_file_hash,
        review_file_hash=review_file_hash,
    )
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
    implementation_source_commit: str,
    git_head_at_protected_read: str,
    git_head_at_terminal: str,
    executing_code_identity_surface_sha256: str,
) -> dict[str, Any]:
    if (
        any(
            type(value) is not str
            or re.fullmatch(r"[0-9a-f]{40}", value) is None
            for value in (
                implementation_source_commit,
                git_head_at_protected_read,
                git_head_at_terminal,
            )
        )
        or git_head_at_protected_read != implementation_source_commit
        or git_head_at_terminal != implementation_source_commit
        or type(executing_code_identity_surface_sha256) is not str
        or SHA_RE.fullmatch(executing_code_identity_surface_sha256) is None
    ):
        _fail("INPUT_LINEAGE_MISMATCH")
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
        "artifact_path_surface_sha256": plan["artifact_path_surface_sha256"],
        "implementation_source_commit": implementation_source_commit,
        "git_head_at_protected_read": git_head_at_protected_read,
        "git_head_at_terminal": git_head_at_terminal,
        "executing_code_identity_surface_sha256": (
            executing_code_identity_surface_sha256
        ),
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


def _independent_artifact_schema(
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
    if role not in descriptors:
        _fail("UNEXPECTED_ARTIFACT")
    schema, hash_field = descriptors[role]
    if role in {
        "implementation_candidate_binding",
        "fresh_implementation_review_verdict",
        "reviewed_implementation_authority",
    }:
        schema = {**schema, "exact_top_level_fields": _artifact_fields(plan, role)}
    return schema, hash_field


def _independent_artifact_link_target(
    role: str, prefix: str
) -> str | None:
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


def _independent_artifact_cross_links_valid(
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
        target = _independent_artifact_link_target(role, prefix)
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
        leaf_schema, leaf_hash_field = _independent_artifact_schema(
            plan, target
        )
        leaf = payload.get("leaf_exact_payload")
        if type(leaf) is not dict:
            return False
        _keys(leaf, set(leaf_schema["exact_top_level_fields"]))
        _self_hash(leaf, leaf_hash_field)
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


def independent_observe_artifact_surface(
    plan: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    outcomes = plan["artifact_lifecycle_contract"]["outcome_field_by_role"]
    observations: dict[str, dict[str, Any]] = {}
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
                schema, hash_field = _independent_artifact_schema(plan, role)
                if final.is_symlink() or not final.is_file():
                    _fail("UNEXPECTED_ARTIFACT")
                raw = final.read_bytes()
                if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
                    _fail("UNEXPECTED_ARTIFACT")
                payload = verifier_json(raw[:-1], canonical=True)
                _keys(payload, set(schema["exact_top_level_fields"]))
                _self_hash(payload, hash_field)
                if role in {
                    "implementation_candidate_binding",
                    "fresh_implementation_review_verdict",
                    "reviewed_implementation_authority",
                }:
                    _validate_review_surface(
                        plan, payload.get("review_surface_identity")
                    )
                if raw != verifier_canonical_bytes(payload) + b"\n":
                    _fail("UNEXPECTED_ARTIFACT")
                if role in outcomes:
                    value = payload.get(outcomes[role])
                    outcome = value if type(value) is str else "__invalid__"
                payloads[role] = payload
                file_hashes[role] = verifier_sha256(raw)
                payload_hashes[role] = str(payload[hash_field])
                valid = True
            except Exception:
                outcome = "__invalid__"
                valid = False
        observations[role] = {
            "final_exists": final_exists,
            "pending_exists": os.path.lexists(pending),
            "outcome": outcome,
            "final_valid": valid,
        }
    validity = {
        role: row["final_valid"] for role, row in observations.items()
    }
    changed = True
    while changed:
        changed = False
        for role, payload in payloads.items():
            if validity[role] and not _independent_artifact_cross_links_valid(
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
        if observations[role]["final_exists"] and not valid:
            observations[role]["outcome"] = "__invalid__"
            observations[role]["final_valid"] = False
    return observations


def independent_classify_lifecycle_surface(
    plan: Mapping[str, Any],
    observations: Mapping[str, Mapping[str, Any]],
    *,
    temporal_predicate: str = "not_applicable",
    liveness: str = "not_applicable",
) -> str:
    roles = tuple(plan["artifact_lifecycle_contract"]["roles"])
    if set(observations) != set(roles):
        return "HOLD_new_review"
    if any(row.get("pending_exists") is True for row in observations.values()):
        return "HOLD_new_review"
    if any(
        row.get("final_exists") is True and row.get("final_valid") is not True
        for row in observations.values()
    ):
        return "HOLD_new_review"
    matches: list[str] = []
    for phase in plan["artifact_lifecycle_contract"]["stable_phases"]:
        if any(
            observations[role].get("final_exists") is not True
            for role in phase["must_exist"]
        ):
            continue
        if any(
            observations[role].get("final_exists") is True
            for role in phase["must_be_absent"]
        ):
            continue
        if any(
            (
                observations[role].get("outcome")
                not in {"success", "failure"}
                if expected == "success_or_failure"
                else observations[role].get("outcome") != expected
            )
            for role, expected in phase["required_outcomes"].items()
        ):
            continue
        expected_time = phase["temporal_predicate"]
        if expected_time != "not_applicable" and expected_time != temporal_predicate:
            continue
        expected_live = phase["liveness_predicate"]
        if expected_live == "ACTIVE_exact_owner" and liveness != "ACTIVE":
            continue
        if expected_live == "DEAD_or_UNKNOWN" and liveness not in {"DEAD", "UNKNOWN"}:
            continue
        if expected_live == "not_applicable" and liveness != "not_applicable":
            continue
        matches.append(str(phase["phase"]))
    return matches[0] if len(matches) == 1 else "HOLD_new_review"


def independent_require_full_surface_checkpoint(
    plan: Mapping[str, Any],
    *,
    scope: str,
    checkpoint: str,
    state: str,
    temporal_predicate: str = "not_applicable",
    liveness: str = "not_applicable",
) -> str:
    frozen = plan["artifact_lifecycle_contract"][
        "full_surface_checkpoint_contract"
    ]
    rows = list(frozen["checkpoint_rows"]) + list(
        frozen["failure_publication_checkpoint_contract"][
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
        _fail("UNEXPECTED_ARTIFACT")
    expected = matches[0]
    if any(
        expected[key] != 18
        for key in (
            "role_count_classified",
            "final_path_count_classified",
            "pending_path_count_classified",
        )
    ):
        _fail("UNEXPECTED_ARTIFACT")
    phase = independent_classify_lifecycle_surface(
        plan,
        independent_observe_artifact_surface(plan),
        temporal_predicate=temporal_predicate,
        liveness=liveness,
    )
    if phase != expected["expected_artifact_phase"]:
        _fail("UNEXPECTED_ARTIFACT")
    return phase


def _publish(
    plan: Mapping[str, Any], role: str, payload: Mapping[str, Any]
) -> None:
    row = _artifact_rows(plan).get(role)
    if row is None:
        _fail("OUTPUT_PUBLICATION_FAILED")
    final = Path(row["final_path"])
    pending = Path(row["pending_path"])
    raw = verifier_canonical_bytes(dict(payload)) + b"\n"
    r2_active = "r2_activation_control" in plan
    expected_suffix = (
        ".pending-" + R2_AUTHORITY_NAMESPACE_ID
        if r2_active
        else ".pending-v0.9"
    )
    if (
        final.parent != pending.parent
        or pending.name != final.name + expected_suffix
        or (
            r2_active
            and (
                role == "formal_design_review_verdict"
                or R2_AUTHORITY_NAMESPACE_ID not in final.name
                or R2_AUTHORITY_NAMESPACE_ID not in pending.name
            )
        )
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
        "authorized_implementation_repository": str(
            AUTHORIZED_IMPLEMENTATION_REPOSITORY
        ),
        "implementation_source_commit": authority[
            "implementation_source_commit"
        ],
        "executing_code_identity_status": "verified",
        "artifact_path_surface_sha256": plan["artifact_path_surface_sha256"],
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
        "artifact_path_surface_sha256": plan["artifact_path_surface_sha256"],
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
        "artifact_path_surface_sha256": plan["artifact_path_surface_sha256"],
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
        "artifact_path_surface_sha256": plan["artifact_path_surface_sha256"],
        "artifact_lifecycle_phase": schema[
            "required_artifact_lifecycle_phase"
        ],
        "protected_input_read": False,
        "implementation_source_commit": preflight[
            "implementation_source_commit"
        ],
        "git_head_at_claim": preflight["implementation_source_commit"],
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
    identity_surface = claim.get(
        "executing_code_identity_surface_sha256"
    )
    if type(identity_surface) is not str or SHA_RE.fullmatch(
        identity_surface
    ) is None:
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
        "artifact_path_surface_sha256": plan["artifact_path_surface_sha256"],
        "implementation_source_commit": extraction_claim[
            "implementation_source_commit"
        ],
        "git_head_at_protected_read": extraction_claim[
            "implementation_source_commit"
        ],
        "git_head_at_terminal": extraction_claim[
            "implementation_source_commit"
        ],
        "executing_code_identity_surface_sha256": extraction_claim[
            "executing_code_identity_surface_sha256"
        ],
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
        "authorized_implementation_repository": str(
            AUTHORIZED_IMPLEMENTATION_REPOSITORY
        ),
        "implementation_source_commit": candidate["source_commit"],
        "executing_code_identity_status": "verified",
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
        "artifact_path_surface_sha256": plan["artifact_path_surface_sha256"],
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
        "artifact_path_surface_sha256": plan["artifact_path_surface_sha256"],
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
    implementation_source_commit: str,
    git_head_at_claim: str,
    executing_code_identity_surface_sha256: str,
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
        or re.fullmatch(r"[0-9a-f]{40}", implementation_source_commit) is None
        or git_head_at_claim != implementation_source_commit
        or SHA_RE.fullmatch(executing_code_identity_surface_sha256) is None
        or controls["preflight"].get("implementation_source_commit")
        != implementation_source_commit
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
        "implementation_source_commit": implementation_source_commit,
        "git_head_at_claim": git_head_at_claim,
        "executing_code_identity_surface_sha256": (
            executing_code_identity_surface_sha256
        ),
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
        "artifact_path_surface_sha256": plan["artifact_path_surface_sha256"],
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
        "artifact_path_surface_sha256": plan["artifact_path_surface_sha256"],
        "implementation_source_commit": claim["implementation_source_commit"],
        "git_head_at_protected_read": progress.get(
            "git_head_at_protected_read", claim["git_head_at_claim"]
        ),
        "git_head_at_terminal": progress.get(
            "git_head_at_terminal", claim["git_head_at_claim"]
        ),
        "executing_code_identity_surface_sha256": claim[
            "executing_code_identity_surface_sha256"
        ],
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
    repository_root = Path(repository).absolute()
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
    independent_require_full_surface_checkpoint(
        plan,
        scope="verifier",
        checkpoint="before_execution_claim_publication",
        state="VERIFIER_AUTHORIZATION_VERIFIED_PASS",
        temporal_predicate="verifier_preflight_and_authorization_fresh",
    )
    entry_module = sys.modules.get("__main__")
    if entry_module is None:
        _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    identity = IndependentExecutingCodeIdentity(
        plan,
        candidate,
        entry_path=Path(__file__).absolute(),
        repository_argument=repository_root,
        entry_module=entry_module,
        bootstrap_record=_VERIFIER_BOOTSTRAP_RECORD,
    )
    claim_identity = identity.checkpoint(identity.checkpoints[0])
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
        implementation_source_commit=candidate["source_commit"],
        git_head_at_claim=claim_identity["git_head"],
        executing_code_identity_surface_sha256=claim_identity[
            "executing_code_identity_surface_sha256"
        ],
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
        independent_require_full_surface_checkpoint(
            plan,
            scope="verifier",
            checkpoint="protected_read_entry",
            state="VERIFIER_EXECUTION_CLAIMED",
            liveness="ACTIVE",
        )
        protected_identity = identity.checkpoint(identity.checkpoints[1])
        if (
            protected_identity["executing_code_identity_surface_sha256"]
            != claim["executing_code_identity_surface_sha256"]
        ):
            _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        progress["git_head_at_protected_read"] = protected_identity["git_head"]
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
            implementation_source_commit=claim[
                "implementation_source_commit"
            ],
            git_head_at_protected_read=protected_identity["git_head"],
            git_head_at_terminal=claim["implementation_source_commit"],
            executing_code_identity_surface_sha256=claim[
                "executing_code_identity_surface_sha256"
            ],
        )
        independent_require_full_surface_checkpoint(
            plan,
            scope="verifier",
            checkpoint="before_success_terminal_publication",
            state="VERIFIER_POST_MANIFEST_VERIFIED",
            liveness="ACTIVE",
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
        independent_require_full_surface_checkpoint(
            plan,
            scope="verifier",
            checkpoint=(
                "lineage_or_semantic_failure_before_failure_terminal_publication"
            ),
            state=str(progress["source_state"]),
            liveness="ACTIVE",
        )
        update_verifier_progress(
            progress,
            failure_phase="verifier_failure_terminal_claim_publication",
        )
        terminal_identity = identity.checkpoint(identity.checkpoints[2])
        if (
            terminal_identity["executing_code_identity_surface_sha256"]
            != claim["executing_code_identity_surface_sha256"]
        ):
            _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        progress["git_head_at_terminal"] = terminal_identity["git_head"]
        _publish(plan, "verifier_terminal", failure_terminal)
        update_verifier_progress(
            progress,
            source_state="VERIFIER_FAILURE_TERMINAL_CLAIM_PUBLISHED",
            failure_phase="verifier_leaf_publication",
        )
        independent_require_full_surface_checkpoint(
            plan,
            scope="verifier",
            checkpoint="before_failure_leaf_publication",
            state="VERIFIER_FAILURE_TERMINAL_CLAIM_PUBLISHED",
            liveness="ACTIVE",
        )
        _publish(plan, "verifier_failure", failure)
        independent_require_full_surface_checkpoint(
            plan,
            scope="verifier",
            checkpoint="terminal_failure_verification",
            state="VERIFIER_FAILURE_RECEIPT_PUBLISHED",
        )
        identity.close()
        raise IndependentVerificationError(failure["failure_code"]) from None

    update_verifier_progress(
        progress, failure_phase="verifier_terminal_claim_publication"
    )
    terminal_identity = identity.checkpoint(identity.checkpoints[2])
    if (
        terminal_identity["executing_code_identity_surface_sha256"]
        != claim["executing_code_identity_surface_sha256"]
    ):
        _fail("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
    progress["git_head_at_terminal"] = terminal_identity["git_head"]
    _publish(plan, "verifier_terminal", terminal)
    update_verifier_progress(
        progress,
        source_state="VERIFIER_SUCCESS_TERMINAL_CLAIM_PUBLISHED",
        failure_phase="verifier_leaf_publication",
    )
    independent_require_full_surface_checkpoint(
        plan,
        scope="verifier",
        checkpoint="before_success_leaf_publication",
        state="VERIFIER_SUCCESS_TERMINAL_CLAIM_PUBLISHED",
        liveness="ACTIVE",
    )
    _publish(plan, "verifier_success", leaf)
    independent_require_full_surface_checkpoint(
        plan,
        scope="verifier",
        checkpoint="terminal_success_verification",
        state="VERIFICATION_RECEIPT_PUBLISHED",
    )
    identity.close()
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


def _verifier_bootstrap_fail() -> None:
    sys.stderr.write(ERROR_PREFIX + "IMPLEMENTATION_BYTE_IDENTITY_MISMATCH\n")
    raise SystemExit(2)


def _bootstrap_verifier_entry_from_retained_source() -> None:
    global _RETAINED_SELF_BOOTSTRAPPED, _VERIFIER_BOOTSTRAP_RECORD
    if _RETAINED_SELF_BOOTSTRAPPED:
        return
    entry_path = Path(__file__).absolute()
    surface: VerifierRetainedSurface | None = None
    entry_handle: int | None = None
    root_handle: int | None = None
    try:
        surface = VerifierRetainedSurface(
            entry_path.parent,
            {"files": [], "directories": []},
        )
        entry_handle = surface._open_handle(entry_path, False)
        entry_metadata = surface._metadata(entry_handle, entry_path, False)
        root = Path(entry_metadata[2]).parent.parent
        if not independent_windows_ordinal_equal(
            str(root), str(AUTHORIZED_IMPLEMENTATION_REPOSITORY), surface.dll
        ):
            _verifier_bootstrap_fail()
        root_handle = surface._open_handle(root, True)
        root_metadata = surface._metadata(root_handle, root, True)
        if root_metadata[0] != entry_metadata[0]:
            _verifier_bootstrap_fail()
        entry_record = (entry_handle, entry_metadata)
        raw = surface._read(entry_record)
        module = sys.modules.get("__main__")
        if module is None:
            _verifier_bootstrap_fail()
        loader = _IndependentRetainedEntryLoader(entry_metadata[2], raw)
        spec = importlib.util.spec_from_loader(
            "__main__", loader, origin=entry_metadata[2]
        )
        if spec is None:
            _verifier_bootstrap_fail()
        module.__file__ = entry_metadata[2]
        module.__spec__ = spec
        _VERIFIER_BOOTSTRAP_RECORD = {
            "surface": surface,
            "entry_record": entry_record,
            "root_record": (root_handle, root_metadata),
            "raw": raw,
        }
        _RETAINED_SELF_BOOTSTRAPPED = True
        module.__dict__["_VERIFIER_BOOTSTRAP_RECORD"] = (
            _VERIFIER_BOOTSTRAP_RECORD
        )
        module.__dict__["_RETAINED_SELF_BOOTSTRAPPED"] = True
        exec(
            compile(raw, entry_metadata[2], "exec", dont_inherit=True),
            module.__dict__,
        )
    except SystemExit:
        raise
    except Exception:
        if surface is not None:
            for handle in (entry_handle, root_handle):
                if handle is not None:
                    try:
                        surface.dll.CloseHandle(handle)
                    except Exception:
                        pass
        _verifier_bootstrap_fail()
    _verifier_bootstrap_fail()


if __name__ == "__main__":
    if not _RETAINED_SELF_BOOTSTRAPPED:
        _bootstrap_verifier_entry_from_retained_source()
    raise SystemExit(cli())
