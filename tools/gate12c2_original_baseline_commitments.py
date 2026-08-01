#!/usr/bin/env python3
"""Gate12C-2 v0.8 original-baseline commitment extraction core.

This module deliberately separates control-plane validation from the only
code path that may parse the protected payload.  Importing it performs no
filesystem access.  Protected input is opened only by ``extract_commitments``
after a caller has published and re-read an exact execution claim.
"""
from __future__ import annotations

import calendar
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
import sys
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence


GATE_ID = "C2_ORIGINAL_BASELINE_COMMITMENT_GATE_v0.8"
PLAN_SCHEMA = "gate12c2_original_baseline_commitment_gate_plan_v0.8"
PLAN_PATH = Path(
    r"C:\Users\aoika\Documents\Research\pale-ale-local\research-program"
    r"\profile-plans\C2_ORIGINAL_BASELINE_COMMITMENT_GATE_PLAN_v0.8_2026-08-01.json"
)
PLAN_FILE_SHA256 = "0a433e92c04e92762aa930647a14c465614fcb2bd11836e63851c617a18a338a"
PLAN_PAYLOAD_SHA256 = "5c2cefc27fd36dbf3343e5095ccfc2645f33a5bb4ce4cf596e0c80905df0d1b6"
CONTRACT_PATH = Path(
    r"C:\Users\aoika\Documents\Research\pale-ale-local\research-program"
    r"\contracts\C2_ORIGINAL_BASELINE_COMMITMENT_GATE_CONTRACT_v0.8_2026-08-01.md"
)
CONTRACT_FILE_SHA256 = "80cf61544a492e824865b5b5612b29f341b2052d20e6fb1eef169351b6456b46"
FORMAL_DESIGN_REVIEW_PATH = Path(
    r"C:\Users\aoika\Documents\Research\pale-ale-local\research-program"
    r"\receipts\C2_ORIGINAL_BASELINE_COMMITMENT_GATE_FORMAL_FRESH_DESIGN_REVIEW_VERDICT_v0.8_2026-08-01.json"
)
FORMAL_DESIGN_REVIEW_FILE_SHA256 = (
    "57dd2e8ad95c35614f7af62eb3527d87f050260772201300971972fed2a2c04c"
)
FORMAL_DESIGN_REVIEW_PAYLOAD_SHA256 = (
    "83ec774e8fdefa9e2c8bc5463c27fa68bd29a70f09c389c4c58707db40644c62"
)
ARTIFACT_PATH_SURFACE_SHA256 = (
    "5d622ba240ea5dd93d963cb47e94a5ff5c3756b370adf823d5e2446435eba005"
)
CONFIGURATION_SURFACE_SHA256 = (
    "a564c25f28e42860f0a1e8f51d4a311b4eae2b771f02dc3f62504547799f19cf"
)
IMPLEMENTATION_AUTHOR_SEPARATION_SHA256 = (
    "9876f27cfb0b4a3ab967d9d1421529c412181885f812c462e8b09ca5b563b44f"
)
IMPLEMENTATION_TRUST_MODEL_SHA256 = (
    "0bb5de8494a08f4566561232aef0a14bba31ef69c1d98da69b4f43f4afd0f940"
)
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
    """Validate all machine-checkable frozen v0.8 design invariants."""

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
            or pending_path != final_path + ".pending-v0.8"
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
    trust = require_mapping(
        supplied.get("implementation_trust_model_contract"),
        code="INPUT_LINEAGE_MISMATCH",
    )
    if sha256_bytes(canonical_json_bytes(trust)) != IMPLEMENTATION_TRUST_MODEL_SHA256:
        _raise("INPUT_LINEAGE_MISMATCH")
    if supplied.get("implementation_trust_model_sha256") != IMPLEMENTATION_TRUST_MODEL_SHA256:
        _raise("INPUT_LINEAGE_MISMATCH")
    if tuple(supplied.get("bounded_implementation_scope_after_fresh_design_pass", ())) != IMPLEMENTATION_PATHS:
        _raise("INPUT_LINEAGE_MISMATCH")
    _validate_design_algebras(supplied)
    return supplied


def _validate_design_algebras(plan: Mapping[str, Any]) -> None:
    codes = plan.get("failure_codes")
    if not isinstance(codes, list) or frozenset(codes) != FAILURE_CODES or len(codes) != 21:
        _raise("INPUT_LINEAGE_MISMATCH")
    matrix = plan.get("failure_matrix")
    if not isinstance(matrix, list) or len(matrix) != 92:
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
    require_int(
        payload.get("P2_count"), minimum=0, code="INPUT_LINEAGE_MISMATCH"
    )
    if p0_count != 0 or p1_count != 0:
        _raise("INPUT_LINEAGE_MISMATCH")
    expected = {
        "contract_file_sha256": CONTRACT_FILE_SHA256,
        "plan_file_sha256": PLAN_FILE_SHA256,
        "plan_payload_sha256": PLAN_PAYLOAD_SHA256,
        "implementation_author_separation_contract_sha256": (
            IMPLEMENTATION_AUTHOR_SEPARATION_SHA256
        ),
        "implementation_trust_model_sha256": IMPLEMENTATION_TRUST_MODEL_SHA256,
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
    matches: list[str] = []
    for phase in plan["artifact_lifecycle_contract"]["stable_phases"]:
        if any(not observations[role].final_exists for role in phase["must_exist"]):
            continue
        if any(observations[role].final_exists for role in phase["must_be_absent"]):
            continue
        if any(
            observations[role].outcome != expected
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
        else final.with_name(final.name + ".pending-v0.8")
    )
    if final.parent != pending.parent or pending.name != final.name + ".pending-v0.8":
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
    return atomic_publish_exact(
        Path(row["final_path"]),
        canonical_receipt_bytes(payload),
        pending_path=Path(row["pending_path"]),
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
            result[role] = read_exact_bytes(
                path, str(file_hash), code="INPUT_LINEAGE_MISMATCH"
            )
            continue
        payload_hash = row.get("payload_sha256")
        if not is_sha256(payload_hash):
            _raise("INPUT_LINEAGE_MISMATCH")
        payload = read_frozen_json_artifact(
            path,
            expected_file_sha256=str(file_hash),
            expected_payload_sha256=str(payload_hash),
        )
        if payload.get("schema_version") != row.get("schema_version"):
            _raise("INPUT_LINEAGE_MISMATCH")
        result[role] = payload
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
        compare = getattr(self.api, "CompareStringOrdinal", None)
        if compare is None:
            return left.casefold() == right.casefold()
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
        return int(compare(left, len(left), right, len(right), True)) == 2

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
        for endpoint in result["endpoint_rows"]:
            if (
                not isinstance(endpoint, dict)
                or endpoint.get("minimum_log_null_inflation") != threshold
            ):
                _raise("INPUT_SCHEMA_INVALID")
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
    def load_frozen_module(
        name: str, filename: str, expected_sha256: str
    ) -> Any:
        path = Path(__file__).resolve().with_name(filename)
        try:
            raw = path.read_bytes()
        except OSError:
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        if sha256_bytes(raw) != expected_sha256:
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        existing = sys.modules.get(name)
        if existing is not None:
            existing_path = getattr(existing, "__file__", None)
            if (
                existing_path is None
                or Path(existing_path).resolve() != path
            ):
                _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
            return existing
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            _raise("IMPLEMENTATION_BYTE_IDENTITY_MISMATCH")
        return module

    frozen_lab = load_frozen_module(
        "gate12c2_synthetic_lab",
        "gate12c2_synthetic_lab.py",
        plan["original_input_lineage"][
            "original_synthetic_lab_file_sha256"
        ],
    )
    frozen_shards = load_frozen_module(
        "gate12c2_development_shards",
        "gate12c2_development_shards.py",
        plan["original_input_lineage"][
            "original_development_shards_file_sha256"
        ],
    )
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
) -> dict[str, Any]:
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
        "artifact_path_surface_sha256": ARTIFACT_PATH_SURFACE_SHA256,
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
    )
    if supplied != expected:
        _raise("INPUT_LINEAGE_MISMATCH")
    return supplied





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


def validate_candidate_binding(
    plan: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    repo_root: Path,
    current_head: str | None = None,
) -> dict[str, Any]:
    contract = plan["implementation_binding_contract"]
    supplied = dict(candidate)
    require_exact_keys(
        supplied,
        contract["exact_top_level_fields"],
        code="IMPLEMENTATION_BYTE_IDENTITY_MISMATCH",
    )
    verify_self_hash(
        supplied,
        "implementation_candidate_binding_payload_sha256",
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
        "implementation_trust_model_sha256": IMPLEMENTATION_TRUST_MODEL_SHA256,
        "artifact_path_surface_sha256": ARTIFACT_PATH_SURFACE_SHA256,
        "worktree_clean": True,
        "core_autocrlf": False,
        "core_longpaths": True,
    }
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
    return supplied


def validate_implementation_review(
    plan: Mapping[str, Any],
    review: Mapping[str, Any],
    *,
    candidate_file_sha256: str,
    candidate_payload_sha256: str,
    source_commit: str,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    schema = plan["review_receipt_schemas"][
        "fresh_implementation_review_verdict"
    ]
    supplied = dict(review)
    require_exact_keys(
        supplied, schema["exact_top_level_fields"], code="INPUT_LINEAGE_MISMATCH"
    )
    verify_self_hash(
        supplied,
        "fresh_implementation_review_payload_sha256",
        code="INPUT_LINEAGE_MISMATCH",
    )
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
        "implementation_trust_model_sha256": IMPLEMENTATION_TRUST_MODEL_SHA256,
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
    payload = {
        "schema_version": contract["schema_version"],
        "authority_id": contract["authority_id_value"],
        "state": contract["state"],
        "authority_derivation_domain": contract["authority_derivation_domain"],
        "implementation_source_commit": candidate["source_commit"],
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
        "implementation_trust_model_sha256": IMPLEMENTATION_TRUST_MODEL_SHA256,
        "artifact_path_surface_sha256": ARTIFACT_PATH_SURFACE_SHA256,
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
    require_exact_keys(
        {**payload, "reviewed_implementation_authority_payload_sha256": ""},
        contract["exact_top_level_fields"],
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
        supplied, contract["exact_top_level_fields"], code="INPUT_LINEAGE_MISMATCH"
    )
    verify_self_hash(
        supplied,
        "reviewed_implementation_authority_payload_sha256",
        code="INPUT_LINEAGE_MISMATCH",
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
        "artifact_path_surface_sha256": ARTIFACT_PATH_SURFACE_SHA256,
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
        "artifact_path_surface_sha256": ARTIFACT_PATH_SURFACE_SHA256,
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
        "artifact_path_surface_sha256": ARTIFACT_PATH_SURFACE_SHA256,
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
        "artifact_path_surface_sha256": ARTIFACT_PATH_SURFACE_SHA256,
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
    ):
        if not is_sha256(digest):
            _raise("AUTHORIZATION_INVALID")
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
        "artifact_path_surface_sha256": ARTIFACT_PATH_SURFACE_SHA256,
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
