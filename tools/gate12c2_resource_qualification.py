#!/usr/bin/env python3
"""Fail-closed primitives for Gate12C-2 replacement resource qualification.

This module is implementation-only.  It cannot issue preflight or authorization
receipts, launch the frozen scientific child, extract baseline commitments, or
run the replacement replay.  The bounded surface implemented here is limited to
the reviewed resource wrapper's telemetry, watchdog deadline, Job ownership,
guardian, legacy-terminal classification, and resource-envelope verification.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import hashlib
import json
import math
import os
import queue
import re
import threading
import time
from dataclasses import InitVar, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence


TELEMETRY_SCHEMA = "gate12c2_resource_telemetry_record_v0.4"
PUBLIC_ERROR_CODE = "GATE12C2_RESOURCE_QUALIFICATION_REJECTED"
GENESIS_DIGEST = "0" * 64
MAXIMUM_RECORD_BYTES = 1024
MAXIMUM_RECORD_COUNT = 1_300_000
MAXIMUM_PERIODIC_RECORD_COUNT = 1_296_001
MAXIMUM_TRANSITION_AND_TAIL_RECORD_COUNT = 3_999
TELEMETRY_WORST_CASE_BYTES = 1_331_200_000
QUALIFICATION_OUTPUT_BUDGET_BYTES = 1_615_757_927
TOTAL_WORST_CASE_DISK_BYTES = 5_094_441_575
MINIMUM_PREFLIGHT_FREE_BYTES = 10_188_883_150
MAXIMUM_WALL_SECONDS = 129_600
LAUNCH_EVIDENCE_DEADLINE_NS = 60_000_000_000
MAXIMUM_LIVE_GAP_NS = 1_000_000_000
JOB_HANDLE_CLOSE_ATTEMPTS = 3

EXPECTED_LEGACY_FAILURE_CODE = (
    "GATE12C2_CLOSEOUT_RESTORE_SCRATCH_ROOT_NOT_PROPAGATED"
)
EXPECTED_LEGACY_EXCEPTION_TYPE = "TypeError"
EXPECTED_LEGACY_EXCEPTION_MESSAGE = (
    "expected str, bytes or os.PathLike object, not NoneType"
)
EXPECTED_LEGACY_STDERR_SHA256 = (
    "41092dc60e3873d551e7a7f7141bd79bb4b2e63fec826ac78adc014fb76037b0"
)
EXPECTED_STALE_LOCK_RELATIVE_PATH = ".draw-profile.lock.json"
EXPECTED_STALE_LOCK_FILE_SHA256 = (
    "bfc44653d072c4d02bc9044788581245aad5a5082e1fae2b2ce4a22e8df753f7"
)
EXPECTED_LEGACY_STACK = (
    ("tools/run_gate12c2_draw_profile.py", 80, "<module>"),
    ("tools/run_gate12c2_draw_profile.py", 45, "main"),
    ("tools/gate12c2_draw_profile.py", 4003, "execute_draw_profile"),
    ("tools/gate12c2_draw_profile.py", 3252, "_build_resource_receipt"),
    ("tools/gate12c2_draw_profile.py", 2962, "_verify_control_lineage"),
    ("tools/gate12c2_draw_profile.py", 2523, "_verify_authorization"),
    ("tools/gate12c2_draw_profile.py", 2211, "_verify_preflight"),
    ("tools/gate12c2_draw_profile.py", 1085, "_verify_worker_carry_forward"),
)

RUNTIME_STATES = (
    "PRELAUNCH",
    "CHILD_SUSPENDED",
    "JOB_ASSIGNED",
    "WATCHDOG_READY",
    "REPLAY_RUNNING_LAUNCH_PENDING",
    "REPLAY_RUNNING_LAUNCH_VERIFIED",
    "EXPECTED_LEGACY_CLOSEOUT_OBSERVED",
    "PAYLOAD_VERIFYING",
    "PAYLOAD_VERIFIED",
    "RESOURCE_MONITORING_COMPLETE",
    "RESOURCE_MONITORING_FAILED",
)
NONTERMINAL_STATES = RUNTIME_STATES[:-2]
TERMINAL_STATES = RUNTIME_STATES[-2:]
EVENT_CODES = (
    "WRAPPER_AUTHORIZATION_CONSUMED",
    "JOINT_PRELAUNCH_CLAIM_SEALED",
    "CHILD_CREATED_SUSPENDED",
    "JOB_ASSIGNED",
    "WATCHDOG_SOLE_HANDLE_CONFIRMED",
    "JOINT_PRE_RESUME_RECEIPT_SEALED",
    "CHILD_RESUMED",
    "SCIENTIFIC_AUTHORIZATION_CONSUMPTION_OBSERVED",
    "JOINT_LAUNCH_EVIDENCE_SEALED",
    "JOINT_LAUNCH_EVIDENCE_VERIFIED",
    "PERIODIC_SAMPLE",
    "EXPECTED_LEGACY_CLOSEOUT_FAILURE_OBSERVED",
    "PAYLOAD_VERIFICATION_STARTED",
    "PAYLOAD_VERIFICATION_PASSED",
    "FINAL_RESOURCE_SAMPLE",
    "MONITORING_COMPLETED",
    "FAILURE_DETECTED",
)
SUCCESS_MILESTONES = (
    "WRAPPER_AUTHORIZATION_CONSUMED",
    "JOINT_PRELAUNCH_CLAIM_SEALED",
    "CHILD_CREATED_SUSPENDED",
    "JOB_ASSIGNED",
    "WATCHDOG_SOLE_HANDLE_CONFIRMED",
    "JOINT_PRE_RESUME_RECEIPT_SEALED",
    "CHILD_RESUMED",
    "SCIENTIFIC_AUTHORIZATION_CONSUMPTION_OBSERVED",
    "JOINT_LAUNCH_EVIDENCE_SEALED",
    "JOINT_LAUNCH_EVIDENCE_VERIFIED",
    "EXPECTED_LEGACY_CLOSEOUT_FAILURE_OBSERVED",
    "PAYLOAD_VERIFICATION_STARTED",
    "PAYLOAD_VERIFICATION_PASSED",
    "FINAL_RESOURCE_SAMPLE",
    "MONITORING_COMPLETED",
)

_SUCCESS_TRANSITIONS = (
    ("__START__", "WRAPPER_AUTHORIZATION_CONSUMED", "PRELAUNCH"),
    ("PRELAUNCH", "JOINT_PRELAUNCH_CLAIM_SEALED", "PRELAUNCH"),
    ("PRELAUNCH", "CHILD_CREATED_SUSPENDED", "CHILD_SUSPENDED"),
    ("CHILD_SUSPENDED", "JOB_ASSIGNED", "JOB_ASSIGNED"),
    (
        "JOB_ASSIGNED",
        "WATCHDOG_SOLE_HANDLE_CONFIRMED",
        "WATCHDOG_READY",
    ),
    (
        "WATCHDOG_READY",
        "JOINT_PRE_RESUME_RECEIPT_SEALED",
        "WATCHDOG_READY",
    ),
    (
        "WATCHDOG_READY",
        "CHILD_RESUMED",
        "REPLAY_RUNNING_LAUNCH_PENDING",
    ),
    (
        "REPLAY_RUNNING_LAUNCH_PENDING",
        "SCIENTIFIC_AUTHORIZATION_CONSUMPTION_OBSERVED",
        "REPLAY_RUNNING_LAUNCH_PENDING",
    ),
    (
        "REPLAY_RUNNING_LAUNCH_PENDING",
        "JOINT_LAUNCH_EVIDENCE_SEALED",
        "REPLAY_RUNNING_LAUNCH_PENDING",
    ),
    (
        "REPLAY_RUNNING_LAUNCH_PENDING",
        "JOINT_LAUNCH_EVIDENCE_VERIFIED",
        "REPLAY_RUNNING_LAUNCH_VERIFIED",
    ),
    (
        "REPLAY_RUNNING_LAUNCH_VERIFIED",
        "EXPECTED_LEGACY_CLOSEOUT_FAILURE_OBSERVED",
        "EXPECTED_LEGACY_CLOSEOUT_OBSERVED",
    ),
    (
        "EXPECTED_LEGACY_CLOSEOUT_OBSERVED",
        "PAYLOAD_VERIFICATION_STARTED",
        "PAYLOAD_VERIFYING",
    ),
    (
        "PAYLOAD_VERIFYING",
        "PAYLOAD_VERIFICATION_PASSED",
        "PAYLOAD_VERIFIED",
    ),
    ("PAYLOAD_VERIFIED", "FINAL_RESOURCE_SAMPLE", "PAYLOAD_VERIFIED"),
    (
        "PAYLOAD_VERIFIED",
        "MONITORING_COMPLETED",
        "RESOURCE_MONITORING_COMPLETE",
    ),
)
TRANSITIONS = {
    (source, event): target
    for source, event, target in (
        *_SUCCESS_TRANSITIONS,
        *((state, "PERIODIC_SAMPLE", state) for state in NONTERMINAL_STATES),
        *(
            (state, "FAILURE_DETECTED", "RESOURCE_MONITORING_FAILED")
            for state in NONTERMINAL_STATES
        ),
    )
}

WIRE_KEY_MAP = {
    "schema_version": "v",
    "sequence": "seq",
    "utc_time": "utc",
    "monotonic_ns": "mono_ns",
    "previous_record_sha256": "prev",
    "record_sha256": "sha",
    "state": "state",
    "event_code": "event",
    "watchdog_pid": "wd_pid",
    "watchdog_creation_time_ns": "wd_ct",
    "guardian_pid": "gd_pid",
    "guardian_creation_time_ns": "gd_ct",
    "coordinator_pid": "co_pid",
    "coordinator_creation_time_ns": "co_ct",
    "replay_root_pid": "root_pid",
    "replay_root_creation_time_ns": "root_ct",
    "job_active_process_count": "job_active",
    "job_total_process_count": "job_total",
    "job_terminated_process_count": "job_term",
    "job_current_memory_bytes": "job_mem",
    "job_peak_memory_bytes": "job_peak",
    "sampled_replay_job_rss_bytes": "rss_job",
    "sampled_control_plane_rss_bytes": "rss_ctl",
    "sampled_combined_rss_bytes": "rss_all",
    "available_physical_memory_bytes": "mem_avail",
    "total_physical_memory_bytes": "mem_total",
    "qualification_volume_free_bytes": "disk_free",
    "scheduled_output_file_count": "out_files",
    "scheduled_output_bytes": "out_bytes",
    "partial_or_temp_count": "partials",
}
WIRE_TO_LONG = {wire: long for long, wire in WIRE_KEY_MAP.items()}
LONG_FIELDS_WITHOUT_DIGEST = tuple(
    field for field in WIRE_KEY_MAP if field != "record_sha256"
)
WIRE_FIELDS = frozenset(WIRE_KEY_MAP.values())

_PID_FIELDS = (
    "watchdog_pid",
    "guardian_pid",
    "coordinator_pid",
    "replay_root_pid",
)
_TIME_FIELDS = (
    "monotonic_ns",
    "watchdog_creation_time_ns",
    "guardian_creation_time_ns",
    "coordinator_creation_time_ns",
    "replay_root_creation_time_ns",
)
_PROCESS_COUNT_FIELDS = (
    "job_active_process_count",
    "job_total_process_count",
    "job_terminated_process_count",
)
_BYTE_FIELDS = (
    "job_current_memory_bytes",
    "job_peak_memory_bytes",
    "sampled_replay_job_rss_bytes",
    "sampled_control_plane_rss_bytes",
    "sampled_combined_rss_bytes",
    "available_physical_memory_bytes",
    "total_physical_memory_bytes",
    "qualification_volume_free_bytes",
    "scheduled_output_bytes",
)
_OUTPUT_COUNT_FIELDS = (
    "scheduled_output_file_count",
    "partial_or_temp_count",
)
_UTC_RE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:"
    r"[0-9]{2}:[0-9]{2}\.[0-9]{6}Z"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class Gate12C2ResourceQualificationError(ValueError):
    """Raised when a bounded resource-qualification invariant is violated."""


@dataclass(frozen=True)
class JobHandleOwnershipProof:
    """Mechanically established source-close and watchdog-handle evidence."""

    source_pid: int
    source_creation_time_ns: int
    watchdog_pid: int
    watchdog_creation_time_ns: int
    watchdog_raw_handle: int
    source_handle_closed: bool
    target_handle_noninheritable: bool
    target_handle_valid_job: bool

    def __post_init__(self) -> None:
        integer_fields = (
            self.source_pid,
            self.source_creation_time_ns,
            self.watchdog_pid,
            self.watchdog_creation_time_ns,
            self.watchdog_raw_handle,
        )
        if any(type(value) is not int or value <= 0 for value in integer_fields):
            raise Gate12C2ResourceQualificationError(
                "Job ownership proof contains an invalid identity or handle"
            )
        if (
            self.source_handle_closed is not True
            or self.target_handle_noninheritable is not True
            or self.target_handle_valid_job is not True
        ):
            raise Gate12C2ResourceQualificationError(
                "Job ownership proof is incomplete"
            )


_JOB_TRANSFER_RECEIPT_TOKEN = object()


@dataclass(frozen=True)
class JobHandleTransferReceipt:
    """Launcher evidence emitted only after duplicate-and-close succeeds."""

    source_pid: int
    source_creation_time_ns: int
    watchdog_pid: int
    watchdog_creation_time_ns: int
    watchdog_raw_handle: int
    source_handle_closed: bool
    duplicate_requested_noninheritable: bool
    _token: InitVar[object | None] = None

    def __post_init__(self, _token: object | None) -> None:
        if _token is not _JOB_TRANSFER_RECEIPT_TOKEN:
            raise Gate12C2ResourceQualificationError(
                "direct Job transfer-receipt construction is forbidden"
            )
        integer_fields = (
            self.source_pid,
            self.source_creation_time_ns,
            self.watchdog_pid,
            self.watchdog_creation_time_ns,
            self.watchdog_raw_handle,
        )
        if any(type(value) is not int or value <= 0 for value in integer_fields):
            raise Gate12C2ResourceQualificationError(
                "Job transfer receipt contains an invalid identity or handle"
            )
        if (
            self.source_handle_closed is not True
            or self.duplicate_requested_noninheritable is not True
        ):
            raise Gate12C2ResourceQualificationError(
                "Job transfer receipt does not prove source closure"
            )


class JobKillHandle(Protocol):
    """The watchdog-owned, non-inheritable final Job handle."""

    inheritable: bool
    ownership_proof: JobHandleOwnershipProof

    def close_for_kill(self) -> None:
        """Close the final Job handle, invoking KILL_ON_JOB_CLOSE."""


def canonical_json_bytes(payload: object) -> bytes:
    try:
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError):
        raise Gate12C2ResourceQualificationError(PUBLIC_ERROR_CODE) from None


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _strict_int(value: object, *, label: str, maximum: int) -> int:
    if type(value) is not int or value < 0 or value > maximum:
        raise Gate12C2ResourceQualificationError(
            f"{label} is outside the frozen integer range"
        )
    return int(value)


def _strict_ascii(
    value: object,
    *,
    label: str,
    maximum_bytes: int,
    exact_bytes: int | None = None,
) -> str:
    if not isinstance(value, str):
        raise Gate12C2ResourceQualificationError(f"{label} is invalid")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError:
        raise Gate12C2ResourceQualificationError(f"{label} is invalid") from None
    if len(encoded) > maximum_bytes or (
        exact_bytes is not None and len(encoded) != exact_bytes
    ):
        raise Gate12C2ResourceQualificationError(f"{label} is invalid")
    return value


def _strict_json_loads(payload: bytes) -> dict[str, Any]:
    def reject_duplicate_pairs(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Gate12C2ResourceQualificationError(
                    "telemetry contains duplicate JSON keys"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        del value
        raise Gate12C2ResourceQualificationError(
            "telemetry contains a nonfinite number"
        )

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicate_pairs,
            parse_constant=reject_constant,
        )
    except Gate12C2ResourceQualificationError:
        raise
    except Exception:
        raise Gate12C2ResourceQualificationError(
            "telemetry is not strict UTF-8 JSON"
        ) from None
    if not isinstance(value, dict):
        raise Gate12C2ResourceQualificationError(
            "telemetry record must be a JSON object"
        )
    return value


def _validate_long_record(
    record: Mapping[str, Any],
    *,
    semantic_enum_required: bool,
) -> dict[str, Any]:
    if set(record) != set(LONG_FIELDS_WITHOUT_DIGEST):
        raise Gate12C2ResourceQualificationError(
            "telemetry fields differ from the frozen closed schema"
        )
    result = dict(record)
    if result["schema_version"] != TELEMETRY_SCHEMA:
        raise Gate12C2ResourceQualificationError(
            "telemetry schema version mismatch"
        )
    _strict_int(result["sequence"], label="sequence", maximum=1_299_999)
    _strict_ascii(
        result["utc_time"],
        label="UTC time",
        maximum_bytes=27,
        exact_bytes=27,
    )
    if _UTC_RE.fullmatch(result["utc_time"]) is None:
        raise Gate12C2ResourceQualificationError("UTC time is invalid")
    if not is_sha256(result["previous_record_sha256"]):
        raise Gate12C2ResourceQualificationError(
            "previous telemetry digest is invalid"
        )
    _strict_ascii(
        result["state"], label="telemetry state", maximum_bytes=64
    )
    _strict_ascii(
        result["event_code"], label="telemetry event", maximum_bytes=80
    )
    if semantic_enum_required:
        if result["state"] not in RUNTIME_STATES:
            raise Gate12C2ResourceQualificationError(
                "telemetry state is not frozen"
            )
        if result["event_code"] not in EVENT_CODES:
            raise Gate12C2ResourceQualificationError(
                "telemetry event is not frozen"
            )
    for field in _PID_FIELDS:
        _strict_int(result[field], label=field, maximum=4_294_967_295)
    for field in _TIME_FIELDS:
        _strict_int(
            result[field],
            label=field,
            maximum=9_999_999_999_999_999_999,
        )
    for field in _PROCESS_COUNT_FIELDS:
        _strict_int(result[field], label=field, maximum=999_999)
    for field in _BYTE_FIELDS:
        _strict_int(
            result[field], label=field, maximum=99_999_999_999_999
        )
    for field in _OUTPUT_COUNT_FIELDS:
        _strict_int(result[field], label=field, maximum=999_999_999)
    return result


def encode_telemetry_record(
    record: Mapping[str, Any],
    *,
    semantic_enum_required: bool = True,
) -> tuple[bytes, str]:
    """Encode one exact canonical record and return stored bytes plus digest."""

    validated = _validate_long_record(
        record, semantic_enum_required=semantic_enum_required
    )
    wire_without_digest = {
        WIRE_KEY_MAP[field]: value for field, value in validated.items()
    }
    digest = sha256_bytes(canonical_json_bytes(wire_without_digest))
    stored = dict(wire_without_digest)
    stored[WIRE_KEY_MAP["record_sha256"]] = digest
    encoded = canonical_json_bytes(stored) + b"\n"
    if len(encoded) > MAXIMUM_RECORD_BYTES:
        raise Gate12C2ResourceQualificationError(
            "telemetry record exceeds the frozen byte limit"
        )
    return encoded, digest


def maximum_capacity_fixture_bytes() -> bytes:
    """Reconstruct the frozen 971-byte all-maxima wire-capacity fixture."""

    maxima: dict[str, Any] = {
        "schema_version": TELEMETRY_SCHEMA,
        "sequence": 1_299_999,
        "utc_time": "9999-12-31T23:59:59.999999Z",
        "monotonic_ns": 9_999_999_999_999_999_999,
        "previous_record_sha256": "f" * 64,
        "state": "S" * 64,
        "event_code": "E" * 80,
        "watchdog_pid": 4_294_967_295,
        "watchdog_creation_time_ns": 9_999_999_999_999_999_999,
        "guardian_pid": 4_294_967_295,
        "guardian_creation_time_ns": 9_999_999_999_999_999_999,
        "coordinator_pid": 4_294_967_295,
        "coordinator_creation_time_ns": 9_999_999_999_999_999_999,
        "replay_root_pid": 4_294_967_295,
        "replay_root_creation_time_ns": 9_999_999_999_999_999_999,
        "job_active_process_count": 999_999,
        "job_total_process_count": 999_999,
        "job_terminated_process_count": 999_999,
        "job_current_memory_bytes": 99_999_999_999_999,
        "job_peak_memory_bytes": 99_999_999_999_999,
        "sampled_replay_job_rss_bytes": 99_999_999_999_999,
        "sampled_control_plane_rss_bytes": 99_999_999_999_999,
        "sampled_combined_rss_bytes": 99_999_999_999_999,
        "available_physical_memory_bytes": 99_999_999_999_999,
        "total_physical_memory_bytes": 99_999_999_999_999,
        "qualification_volume_free_bytes": 99_999_999_999_999,
        "scheduled_output_file_count": 999_999_999,
        "scheduled_output_bytes": 99_999_999_999_999,
        "partial_or_temp_count": 999_999_999,
    }
    encoded, _ = encode_telemetry_record(
        maxima, semantic_enum_required=False
    )
    return encoded


@dataclass
class _TelemetryState:
    sequence: int = 0
    previous_digest: str = GENESIS_DIGEST
    previous_state: str = "__START__"
    success_milestone_index: int = 0
    periodic_count: int = 0
    transition_and_tail_count: int = 0
    terminal: bool = False
    previous_monotonic_ns: int | None = None

    def accept(
        self,
        *,
        event_code: str,
        state: str,
        sequence: int,
        previous_digest: str,
        monotonic_ns: int,
        record_digest: str,
    ) -> None:
        if self.terminal:
            raise Gate12C2ResourceQualificationError(
                "telemetry continues after a terminal state"
            )
        if sequence != self.sequence:
            raise Gate12C2ResourceQualificationError(
                "telemetry sequence is not contiguous"
            )
        if previous_digest != self.previous_digest:
            raise Gate12C2ResourceQualificationError(
                "telemetry hash chain is discontinuous"
            )
        if self.previous_monotonic_ns is not None:
            if monotonic_ns < self.previous_monotonic_ns:
                raise Gate12C2ResourceQualificationError(
                    "telemetry monotonic time moved backwards"
                )
            if monotonic_ns - self.previous_monotonic_ns > MAXIMUM_LIVE_GAP_NS:
                raise Gate12C2ResourceQualificationError(
                    "telemetry exceeds the frozen maximum live gap"
                )
        target = TRANSITIONS.get((self.previous_state, event_code))
        if target is None or target != state:
            raise Gate12C2ResourceQualificationError(
                "telemetry state-event transition is not frozen"
            )
        if event_code in SUCCESS_MILESTONES:
            if self.success_milestone_index >= len(SUCCESS_MILESTONES):
                raise Gate12C2ResourceQualificationError(
                    "telemetry success milestone is duplicated"
                )
            expected = SUCCESS_MILESTONES[self.success_milestone_index]
            if event_code != expected:
                raise Gate12C2ResourceQualificationError(
                    "telemetry success milestone is omitted or reordered"
                )
            self.success_milestone_index += 1
            self.transition_and_tail_count += 1
        elif event_code == "PERIODIC_SAMPLE":
            self.periodic_count += 1
        elif event_code == "FAILURE_DETECTED":
            self.transition_and_tail_count += 1
        else:
            raise Gate12C2ResourceQualificationError(
                "telemetry event is not classified"
            )
        if self.periodic_count > MAXIMUM_PERIODIC_RECORD_COUNT:
            raise Gate12C2ResourceQualificationError(
                "periodic telemetry capacity is exhausted"
            )
        if (
            self.transition_and_tail_count
            > MAXIMUM_TRANSITION_AND_TAIL_RECORD_COUNT
        ):
            raise Gate12C2ResourceQualificationError(
                "transition telemetry capacity is exhausted"
            )
        if self.sequence + 1 > MAXIMUM_RECORD_COUNT:
            raise Gate12C2ResourceQualificationError(
                "telemetry record capacity is exhausted"
            )
        self.sequence += 1
        self.previous_digest = record_digest
        self.previous_state = state
        self.previous_monotonic_ns = monotonic_ns
        self.terminal = state in TERMINAL_STATES


def decode_and_verify_telemetry(
    payload: bytes,
    *,
    require_terminal: bool = True,
) -> dict[str, Any]:
    """Strictly verify a complete telemetry JSONL byte stream."""

    if require_terminal is not True:
        raise Gate12C2ResourceQualificationError(
            "partial telemetry verification is forbidden"
        )
    if not payload or not payload.endswith(b"\n"):
        raise Gate12C2ResourceQualificationError(
            "telemetry is empty or lacks the final LF"
        )
    if b"\r" in payload or payload.startswith(b"\xef\xbb\xbf"):
        raise Gate12C2ResourceQualificationError(
            "telemetry contains forbidden CR or BOM bytes"
        )
    lines = payload.splitlines(keepends=True)
    state = _TelemetryState()
    for line in lines:
        if line == b"\n" or not line.endswith(b"\n"):
            raise Gate12C2ResourceQualificationError(
                "telemetry contains a blank or incomplete record"
            )
        if len(line) > MAXIMUM_RECORD_BYTES:
            raise Gate12C2ResourceQualificationError(
                "telemetry record exceeds the frozen byte limit"
            )
        stored = _strict_json_loads(line[:-1])
        if set(stored) != WIRE_FIELDS:
            raise Gate12C2ResourceQualificationError(
                "telemetry wire keys differ from the frozen schema"
            )
        if canonical_json_bytes(stored) + b"\n" != line:
            raise Gate12C2ResourceQualificationError(
                "telemetry record is not canonical JSONL"
            )
        digest = stored.get(WIRE_KEY_MAP["record_sha256"])
        if not is_sha256(digest):
            raise Gate12C2ResourceQualificationError(
                "telemetry record digest is invalid"
            )
        without_digest = dict(stored)
        without_digest.pop(WIRE_KEY_MAP["record_sha256"])
        if sha256_bytes(canonical_json_bytes(without_digest)) != digest:
            raise Gate12C2ResourceQualificationError(
                "telemetry record digest mismatch"
            )
        long_record = {
            WIRE_TO_LONG[key]: value
            for key, value in without_digest.items()
        }
        validated = _validate_long_record(
            long_record, semantic_enum_required=True
        )
        state.accept(
            event_code=str(validated["event_code"]),
            state=str(validated["state"]),
            sequence=int(validated["sequence"]),
            previous_digest=str(validated["previous_record_sha256"]),
            monotonic_ns=int(validated["monotonic_ns"]),
            record_digest=str(digest),
        )
    if not state.terminal:
        raise Gate12C2ResourceQualificationError(
            "telemetry does not end in a frozen terminal state"
        )
    if (
        state.previous_state == "RESOURCE_MONITORING_COMPLETE"
        and state.success_milestone_index != len(SUCCESS_MILESTONES)
    ):
        raise Gate12C2ResourceQualificationError(
            "successful telemetry omits a frozen milestone"
        )
    return {
        "status": "pass",
        "record_count": state.sequence,
        "periodic_record_count": state.periodic_count,
        "transition_and_tail_record_count": state.transition_and_tail_count,
        "terminal_state": state.previous_state,
        "telemetry_file_sha256": sha256_bytes(payload),
        "final_record_sha256": state.previous_digest,
        "scientific_values_emitted": False,
    }


class AppendOnlyTelemetryWriter:
    """OS-exclusive, sticky fail-closed append-only telemetry writer."""

    def __init__(
        self,
        path: Path,
        *,
        fsync: Callable[[int], None] = os.fsync,
    ) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._open_exclusive_handle(self.path)
        self._fsync = fsync
        self._state = _TelemetryState()
        self._closed = False
        self._failed = False
        self._last_durable_offset = 0

    @staticmethod
    def _open_exclusive_handle(path: Path) -> Any:
        if os.name != "nt":
            descriptor = os.open(
                path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
            return os.fdopen(descriptor, "wb", buffering=0)
        import msvcrt

        generic_write = 0x40000000
        create_new = 1
        file_attribute_normal = 0x00000080
        invalid_handle_value = ctypes.c_void_p(-1).value
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateFileW.argtypes = [
            ctypes.c_wchar_p,
            ctypes.wintypes.DWORD,
            ctypes.wintypes.DWORD,
            ctypes.c_void_p,
            ctypes.wintypes.DWORD,
            ctypes.wintypes.DWORD,
            ctypes.wintypes.HANDLE,
        ]
        kernel32.CreateFileW.restype = ctypes.wintypes.HANDLE
        raw = kernel32.CreateFileW(
            str(path),
            generic_write,
            0,
            None,
            create_new,
            file_attribute_normal,
            None,
        )
        if not raw or int(raw) == invalid_handle_value:
            error = ctypes.get_last_error()
            if error in {80, 183}:
                raise FileExistsError(str(path))
            raise Gate12C2ResourceQualificationError(
                "telemetry file could not be opened exclusively"
            )
        try:
            descriptor = msvcrt.open_osfhandle(
                int(raw), os.O_WRONLY | getattr(os, "O_BINARY", 0)
            )
        except Exception:
            kernel32.CloseHandle(raw)
            raise Gate12C2ResourceQualificationError(
                "telemetry file-handle transfer failed"
            ) from None
        return os.fdopen(descriptor, "wb", buffering=0)

    def _close_handle_only(self) -> bool:
        if self._closed:
            return True
        for _ in range(JOB_HANDLE_CLOSE_ATTEMPTS):
            try:
                self._handle.close()
            except Exception:
                continue
            self._closed = True
            return True
        self._closed = bool(getattr(self._handle, "closed", False))
        self._failed = True
        return self._closed

    def _poison(self, start_offset: int, reason: str) -> None:
        self._failed = True
        try:
            self._handle.seek(start_offset)
            self._handle.truncate()
            self._handle.write(b"\x00")
            self._handle.flush()
            os.fsync(self._handle.fileno())
        except Exception:
            pass
        close_verified = self._close_handle_only()
        if not close_verified:
            reason = f"{reason}; telemetry handle close failed"
        raise Gate12C2ResourceQualificationError(reason) from None

    def append(
        self,
        *,
        event_code: str,
        utc_time: str,
        monotonic_ns: int,
        metrics: Mapping[str, Any],
    ) -> str:
        if self._failed:
            raise Gate12C2ResourceQualificationError(
                "telemetry writer is terminally failed"
            )
        if self._closed:
            raise Gate12C2ResourceQualificationError(
                "telemetry writer is closed"
            )
        target = TRANSITIONS.get((self._state.previous_state, event_code))
        if target is None:
            raise Gate12C2ResourceQualificationError(
                "telemetry state-event transition is not frozen"
            )
        automatic = {
            "schema_version": TELEMETRY_SCHEMA,
            "sequence": self._state.sequence,
            "utc_time": utc_time,
            "monotonic_ns": monotonic_ns,
            "previous_record_sha256": self._state.previous_digest,
            "state": target,
            "event_code": event_code,
        }
        overlap = set(automatic) & set(metrics)
        if overlap:
            raise Gate12C2ResourceQualificationError(
                "telemetry metrics overlap state-machine fields"
            )
        record = {**automatic, **dict(metrics)}
        encoded, digest = encode_telemetry_record(record)
        candidate_state = replace(self._state)
        candidate_state.accept(
            event_code=event_code,
            state=target,
            sequence=int(record["sequence"]),
            previous_digest=str(record["previous_record_sha256"]),
            monotonic_ns=int(record["monotonic_ns"]),
            record_digest=digest,
        )
        start_offset = self._handle.tell()
        durable = (
            event_code != "PERIODIC_SAMPLE"
            or candidate_state.sequence % 10 == 0
        )
        try:
            written = self._handle.write(encoded)
            if written != len(encoded):
                raise OSError("short telemetry write")
            self._handle.flush()
            if durable:
                self._fsync(self._handle.fileno())
        except Exception:
            self._poison(start_offset, "telemetry append failed")
        self._state = candidate_state
        if durable:
            self._last_durable_offset = self._handle.tell()
        return digest

    def close(self) -> None:
        if self._failed:
            self._close_handle_only()
            raise Gate12C2ResourceQualificationError(
                "telemetry writer is terminally failed"
            )
        if self._closed:
            return
        if not self._state.terminal:
            self._poison(
                self._last_durable_offset,
                "telemetry writer closed before a terminal state",
            )
        try:
            self._handle.flush()
            self._fsync(self._handle.fileno())
        except Exception:
            self._poison(
                self._last_durable_offset,
                "telemetry close failed",
            )
        self._last_durable_offset = self._handle.tell()
        if not self._close_handle_only():
            raise Gate12C2ResourceQualificationError(
                "telemetry handle close failed"
            )

    def __enter__(self) -> "AppendOnlyTelemetryWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del tb
        if exc_type is not None or exc is not None:
            self._failed = True
            try:
                if not self._closed:
                    self._handle.seek(self._last_durable_offset)
                    self._handle.truncate()
                    self._handle.write(b"\x00")
                    self._handle.flush()
                    os.fsync(self._handle.fileno())
            except Exception:
                pass
            self._close_handle_only()
            return
        self.close()


class LaunchDeadlineWatchdog:
    """Independent strict deadline owner and sole Job-kill actuator."""

    def __init__(
        self,
        job_handle: JobKillHandle,
        *,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        proof = getattr(job_handle, "ownership_proof", None)
        if not isinstance(proof, JobHandleOwnershipProof):
            raise Gate12C2ResourceQualificationError(
                "watchdog Job ownership proof is absent"
            )
        if getattr(job_handle, "inheritable", True):
            raise Gate12C2ResourceQualificationError(
                "watchdog Job handle is inheritable"
            )
        self._job_handle = job_handle
        self._monotonic_ns = monotonic_ns
        self.resume_success_monotonic_ns: int | None = None
        self.ack_monotonic_ns: int | None = None
        self.verified = False
        self.terminated = False
        self.job_handle_close_verified = False
        self.close_attempt_count = 0

    def _clock(self) -> int:
        try:
            value = self._monotonic_ns()
        except Exception:
            self._terminate("monotonic clock failed")
        if type(value) is not int or value < 0:
            self._terminate("monotonic clock is invalid")
        return int(value)

    def _terminate(self, reason: str) -> None:
        self.terminated = True
        last_error = False
        if not self.job_handle_close_verified:
            for _ in range(JOB_HANDLE_CLOSE_ATTEMPTS):
                self.close_attempt_count += 1
                try:
                    self._job_handle.close_for_kill()
                except Exception:
                    last_error = True
                    continue
                self.job_handle_close_verified = True
                last_error = False
                break
        if not self.job_handle_close_verified or last_error:
            raise Gate12C2ResourceQualificationError(
                f"{reason}; Job handle close failed"
            ) from None
        raise Gate12C2ResourceQualificationError(reason) from None

    @property
    def deadline_monotonic_ns(self) -> int:
        if self.resume_success_monotonic_ns is None:
            raise Gate12C2ResourceQualificationError(
                "launch deadline is not armed"
            )
        return self.resume_success_monotonic_ns + LAUNCH_EVIDENCE_DEADLINE_NS

    def resume_and_arm(self, resume: Callable[[], int]) -> int:
        if self.resume_success_monotonic_ns is not None:
            self._terminate("scientific child was resumed more than once")
        try:
            previous_suspend_count = resume()
        except Exception:
            self._terminate("scientific child resume failed")
        if type(previous_suspend_count) is not int or previous_suspend_count != 1:
            self._terminate("scientific child resume result is invalid")
        self.resume_success_monotonic_ns = self._clock()
        return self.resume_success_monotonic_ns

    def enforce_deadline(self) -> None:
        if self.resume_success_monotonic_ns is None:
            self._terminate("launch deadline is not armed")
        if not self.verified and self._clock() >= self.deadline_monotonic_ns:
            self._terminate("launch acknowledgement is missing at deadline")

    def _call_until_deadline(
        self,
        callback: Callable[[], object],
        *,
        failure_reason: str,
        poll_seconds: float,
    ) -> object:
        result_queue: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=1)

        def invoke() -> None:
            try:
                value = callback()
            except Exception:
                result_queue.put(("error", None))
            else:
                result_queue.put(("ok", value))

        try:
            worker = threading.Thread(
                target=invoke,
                name="gate12c2-deadline-callback",
                daemon=True,
            )
            worker.start()
        except Exception:
            self._terminate(f"{failure_reason} thread failed")
        while True:
            try:
                status, value = result_queue.get(timeout=poll_seconds)
            except queue.Empty:
                self.enforce_deadline()
                continue
            except Exception:
                self._terminate(f"{failure_reason} wait failed")
            if status != "ok":
                self._terminate(failure_reason)
            if self._clock() >= self.deadline_monotonic_ns:
                self._terminate("launch acknowledgement missed the deadline")
            return value

    def verify_acknowledgement(
        self,
        acknowledgement: object,
        verifier: Callable[[object], bool],
        *,
        poll_seconds: float = 0.01,
    ) -> int:
        if self.resume_success_monotonic_ns is None:
            self._terminate("acknowledgement preceded scientific resume")
        if self.verified:
            self._terminate("launch acknowledgement is duplicated")
        if acknowledgement is None:
            self._terminate("launch acknowledgement is missing")
        verified = self._call_until_deadline(
            lambda: verifier(acknowledgement),
            failure_reason="launch acknowledgement verifier failed",
            poll_seconds=poll_seconds,
        )
        if verified is not True:
            self._terminate("launch acknowledgement is invalid")
        acknowledgement_time = self._clock()
        if acknowledgement_time >= self.deadline_monotonic_ns:
            self._terminate("launch acknowledgement missed the deadline")
        self.ack_monotonic_ns = acknowledgement_time
        self.verified = True
        return acknowledgement_time

    def run_until_verified(
        self,
        *,
        acknowledgement_supplier: Callable[[], object | None],
        verifier: Callable[[object], bool],
        poll_seconds: float = 0.01,
    ) -> int:
        if self.resume_success_monotonic_ns is None:
            self._terminate("launch deadline is not armed")
        if (
            not isinstance(poll_seconds, (int, float))
            or isinstance(poll_seconds, bool)
            or not math.isfinite(float(poll_seconds))
            or poll_seconds <= 0
            or poll_seconds > 0.1
        ):
            self._terminate("watchdog polling interval is invalid")
        while not self.verified:
            self.enforce_deadline()
            acknowledgement = self._call_until_deadline(
                acknowledgement_supplier,
                failure_reason="launch acknowledgement supplier failed",
                poll_seconds=float(poll_seconds),
            )
            if acknowledgement is not None:
                return self.verify_acknowledgement(
                    acknowledgement,
                    verifier,
                    poll_seconds=float(poll_seconds),
                )
        if self.ack_monotonic_ns is None:
            self._terminate("verified acknowledgement lacks watchdog time")
        return self.ack_monotonic_ns


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    creation_time_ns: int

    def __post_init__(self) -> None:
        _strict_int(self.pid, label="PID", maximum=4_294_967_295)
        _strict_int(
            self.creation_time_ns,
            label="process creation time",
            maximum=9_999_999_999_999_999_999,
        )
        if self.pid == 0:
            raise Gate12C2ResourceQualificationError("PID cannot be zero")


class NoHandleGuardian:
    """Observe closed process identities without owning or reopening the Job."""

    __slots__ = ("_identities",)

    def __init__(self, identities: Sequence[ProcessIdentity]) -> None:
        if not identities:
            raise Gate12C2ResourceQualificationError(
                "guardian identity surface is empty"
            )
        self._identities = tuple(identities)

    @property
    def owns_job_handle(self) -> bool:
        return False

    def record_watchdog_failure(
        self,
        probe: Callable[[ProcessIdentity], str],
    ) -> dict[str, Any]:
        statuses = []
        for identity in self._identities:
            status = probe(identity)
            if status not in {"DEAD", "ACTIVE", "UNKNOWN"}:
                raise Gate12C2ResourceQualificationError(
                    "guardian process probe is invalid"
                )
            statuses.append(status)
        if any(status != "DEAD" for status in statuses):
            raise Gate12C2ResourceQualificationError(
                "guardian cannot close while a process is active or unknown"
            )
        return {
            "status": "watchdog_failure_all_watched_processes_dead",
            "watched_process_count": len(self._identities),
            "continuation_authorized": False,
            "qualification_pass_authorized": False,
            "job_handle_owned": False,
        }


def classify_expected_legacy_closeout(
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify only the exact reviewed post-payload legacy closeout defect."""

    required = {
        "child_exit_code",
        "stdout_bytes",
        "exception_type",
        "exception_message",
        "normalized_project_stack",
        "stderr_sha256",
        "configuration_count",
        "index_count",
        "outer_experiment_count",
        "shard_count",
        "partial_or_temp_count",
        "stale_lock_count",
        "stale_lock_relative_path",
        "stale_lock_file_sha256",
        "stale_lock_manifest_match",
        "unexpected_artifact_count",
        "unexpected_artifact_relative_paths",
        "legacy_execution_evidence_present",
        "legacy_resource_receipt_present",
        "legacy_execution_receipt_present",
        "semantic_commitments_match",
        "telemetry_tail_complete",
    }
    if set(evidence) != required:
        raise Gate12C2ResourceQualificationError(
            "legacy closeout evidence fields differ from the frozen schema"
        )
    if (
        type(evidence["child_exit_code"]) is not int
        or evidence["child_exit_code"] != 1
        or type(evidence["stdout_bytes"]) is not int
        or evidence["stdout_bytes"] != 0
        or evidence["exception_type"] != EXPECTED_LEGACY_EXCEPTION_TYPE
        or evidence["exception_message"] != EXPECTED_LEGACY_EXCEPTION_MESSAGE
        or evidence["stderr_sha256"] != EXPECTED_LEGACY_STDERR_SHA256
    ):
        raise Gate12C2ResourceQualificationError(
            "legacy child terminal differs from the frozen failure"
        )
    stack = evidence["normalized_project_stack"]
    if type(stack) is not list or len(stack) != len(EXPECTED_LEGACY_STACK):
        raise Gate12C2ResourceQualificationError(
            "legacy closeout stack differs from the frozen failure"
        )
    normalized_rows: list[tuple[object, object, object]] = []
    for row in stack:
        if type(row) is not dict or set(row) != {"path", "line", "function"}:
            raise Gate12C2ResourceQualificationError(
                "legacy closeout stack row schema is not exact"
            )
        if (
            not isinstance(row["path"], str)
            or type(row["line"]) is not int
            or not isinstance(row["function"], str)
        ):
            raise Gate12C2ResourceQualificationError(
                "legacy closeout stack row types are invalid"
            )
        normalized_rows.append(
            (row["path"], row["line"], row["function"])
        )
    if tuple(normalized_rows) != EXPECTED_LEGACY_STACK:
        raise Gate12C2ResourceQualificationError(
            "legacy closeout stack differs from the frozen failure"
        )
    exact_counts = {
        "configuration_count": 9,
        "index_count": 9,
        "outer_experiment_count": 768,
        "shard_count": 768,
        "partial_or_temp_count": 0,
        "stale_lock_count": 1,
        "unexpected_artifact_count": 0,
    }
    for field, expected in exact_counts.items():
        if type(evidence[field]) is not int or evidence[field] != expected:
            raise Gate12C2ResourceQualificationError(
                "legacy payload surface is incomplete"
            )
    if (
        evidence["stale_lock_relative_path"]
        != EXPECTED_STALE_LOCK_RELATIVE_PATH
        or evidence["stale_lock_file_sha256"]
        != EXPECTED_STALE_LOCK_FILE_SHA256
        or evidence["stale_lock_manifest_match"] is not True
        or type(evidence["unexpected_artifact_relative_paths"]) is not list
        or evidence["unexpected_artifact_relative_paths"] != []
    ):
        raise Gate12C2ResourceQualificationError(
            "legacy root artifact surface is not exact"
        )
    false_fields = (
        "legacy_execution_evidence_present",
        "legacy_resource_receipt_present",
        "legacy_execution_receipt_present",
    )
    true_fields = (
        "semantic_commitments_match",
        "telemetry_tail_complete",
    )
    if any(evidence[field] is not False for field in false_fields) or any(
        evidence[field] is not True for field in true_fields
    ):
        raise Gate12C2ResourceQualificationError(
            "legacy closeout artifact conditions are not frozen"
        )
    return {
        "status": (
            "REPLAY_PAYLOAD_COMPLETE_WITH_EXPECTED_LEGACY_CLOSEOUT_FAILURE"
        ),
        "failure_code": EXPECTED_LEGACY_FAILURE_CODE,
        "legacy_child_success_claimed": False,
        "payload_count": 768,
        "scientific_values_emitted": False,
    }


def verify_resource_envelope(
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify only frozen resource-envelope scalars; emit no science."""

    required = {
        "physical_ram_bytes",
        "peak_job_memory_bytes",
        "sampled_combined_rss_bytes",
        "sampled_available_physical_memory_bytes",
        "preflight_free_bytes",
        "minimum_observed_free_bytes",
        "qualification_output_bytes",
        "telemetry_bytes",
        "wall_seconds",
        "job_memory_limit_event_count",
        "monitor_error_count",
        "partial_or_temp_count",
    }
    if set(evidence) != required:
        raise Gate12C2ResourceQualificationError(
            "resource evidence fields differ from the frozen schema"
        )
    values = {
        field: _strict_int(
            value,
            label=field,
            maximum=99_999_999_999_999,
        )
        for field, value in evidence.items()
    }
    physical = values["physical_ram_bytes"]
    if physical <= 0:
        raise Gate12C2ResourceQualificationError(
            "physical memory evidence is invalid"
        )
    memory_limit = (3 * physical) // 4
    available_floor = (physical + 9) // 10
    preflight_free = values["preflight_free_bytes"]
    if (
        preflight_free < MINIMUM_PREFLIGHT_FREE_BYTES
        or preflight_free - TOTAL_WORST_CASE_DISK_BYTES
        < (preflight_free + 1) // 2
    ):
        raise Gate12C2ResourceQualificationError(
            "preflight disk evidence fails the frozen gate"
        )
    limits = (
        values["peak_job_memory_bytes"] <= memory_limit,
        values["sampled_combined_rss_bytes"] <= memory_limit,
        values["sampled_available_physical_memory_bytes"]
        >= available_floor,
        values["minimum_observed_free_bytes"]
        >= (preflight_free + 1) // 2,
        values["qualification_output_bytes"]
        <= QUALIFICATION_OUTPUT_BUDGET_BYTES,
        values["telemetry_bytes"] <= TELEMETRY_WORST_CASE_BYTES,
        values["wall_seconds"] <= MAXIMUM_WALL_SECONDS,
        values["job_memory_limit_event_count"] == 0,
        values["monitor_error_count"] == 0,
        values["partial_or_temp_count"] == 0,
    )
    if not all(limits):
        raise Gate12C2ResourceQualificationError(
            "resource evidence fails the frozen envelope"
        )
    return {
        "status": "pass",
        "job_memory_limit_bytes": memory_limit,
        "sampled_available_memory_floor_bytes": available_floor,
        "scientific_values_emitted": False,
        "original_resource_gate_status": "indeterminate",
    }


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", ctypes.wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.wintypes.DWORD),
        ("SchedulingClass", ctypes.wintypes.DWORD),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class WindowsJobApi:
    """Reviewed Win32 adapter; it never launches the scientific child."""

    JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
    DUPLICATE_CLOSE_SOURCE = 0x00000001
    DUPLICATE_SAME_ACCESS = 0x00000002
    HANDLE_FLAG_INHERIT = 0x00000001

    def __init__(self) -> None:
        if os.name != "nt":
            raise Gate12C2ResourceQualificationError(
                "Windows Job Object support is unavailable"
            )
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    def create_unnamed_job(self, *, physical_ram_bytes: int) -> int:
        physical = _strict_int(
            physical_ram_bytes,
            label="physical RAM",
            maximum=99_999_999_999_999,
        )
        if physical <= 0:
            raise Gate12C2ResourceQualificationError(
                "physical RAM is invalid"
            )
        self.kernel32.CreateJobObjectW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_wchar_p,
        ]
        self.kernel32.CreateJobObjectW.restype = ctypes.wintypes.HANDLE
        handle = self.kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise Gate12C2ResourceQualificationError(
                "could not create unnamed Job Object"
            )
        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = (
            self.JOB_OBJECT_LIMIT_JOB_MEMORY
            | self.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        info.JobMemoryLimit = (3 * physical) // 4
        self.kernel32.SetInformationJobObject.argtypes = [
            ctypes.wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.wintypes.DWORD,
        ]
        self.kernel32.SetInformationJobObject.restype = ctypes.wintypes.BOOL
        if not self.kernel32.SetInformationJobObject(
            handle,
            self.JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ):
            self.close_handle(int(handle))
            raise Gate12C2ResourceQualificationError(
                "could not apply frozen Job Object limits"
            )
        return int(handle)

    def assign_process(self, job_handle: int, process_handle: int) -> None:
        self.kernel32.AssignProcessToJobObject.argtypes = [
            ctypes.wintypes.HANDLE,
            ctypes.wintypes.HANDLE,
        ]
        self.kernel32.AssignProcessToJobObject.restype = ctypes.wintypes.BOOL
        if not self.kernel32.AssignProcessToJobObject(
            job_handle, process_handle
        ):
            raise Gate12C2ResourceQualificationError(
                "could not assign process to Job Object"
            )

    def _duplicate_into_process(
        self,
        job_handle: int,
        target_process_handle: int,
    ) -> int:
        self.kernel32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
        self.kernel32.DuplicateHandle.argtypes = [
            ctypes.wintypes.HANDLE,
            ctypes.wintypes.HANDLE,
            ctypes.wintypes.HANDLE,
            ctypes.POINTER(ctypes.wintypes.HANDLE),
            ctypes.wintypes.DWORD,
            ctypes.wintypes.BOOL,
            ctypes.wintypes.DWORD,
        ]
        self.kernel32.DuplicateHandle.restype = ctypes.wintypes.BOOL
        duplicate = ctypes.wintypes.HANDLE()
        if not self.kernel32.DuplicateHandle(
            self.kernel32.GetCurrentProcess(),
            job_handle,
            target_process_handle,
            ctypes.byref(duplicate),
            0,
            False,
            self.DUPLICATE_SAME_ACCESS,
        ):
            raise Gate12C2ResourceQualificationError(
                "could not duplicate Job handle into watchdog"
            )
        return int(duplicate.value)

    def _close_remote_handle(
        self,
        target_process_handle: int,
        remote_handle: int,
    ) -> None:
        self.kernel32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
        self.kernel32.DuplicateHandle.argtypes = [
            ctypes.wintypes.HANDLE,
            ctypes.wintypes.HANDLE,
            ctypes.wintypes.HANDLE,
            ctypes.POINTER(ctypes.wintypes.HANDLE),
            ctypes.wintypes.DWORD,
            ctypes.wintypes.BOOL,
            ctypes.wintypes.DWORD,
        ]
        self.kernel32.DuplicateHandle.restype = ctypes.wintypes.BOOL
        local = ctypes.wintypes.HANDLE()
        if not self.kernel32.DuplicateHandle(
            target_process_handle,
            remote_handle,
            self.kernel32.GetCurrentProcess(),
            ctypes.byref(local),
            0,
            False,
            self.DUPLICATE_CLOSE_SOURCE | self.DUPLICATE_SAME_ACCESS,
        ):
            raise Gate12C2ResourceQualificationError(
                "could not close failed watchdog duplicate"
            )
        self.close_handle(int(local.value))

    def transfer_job_handle_to_watchdog(
        self,
        *,
        source_job_handle: int,
        target_process_handle: int,
        source_identity: ProcessIdentity,
        watchdog_identity: ProcessIdentity,
    ) -> JobHandleTransferReceipt:
        duplicate = self._duplicate_into_process(
            source_job_handle, target_process_handle
        )
        try:
            self.close_handle(source_job_handle)
        except Exception:
            try:
                self._close_remote_handle(
                    target_process_handle, duplicate
                )
            except Exception:
                raise Gate12C2ResourceQualificationError(
                    "source Job handle close and duplicate cleanup failed"
                ) from None
            raise Gate12C2ResourceQualificationError(
                "source Job handle close failed"
            ) from None
        return JobHandleTransferReceipt(
            source_pid=source_identity.pid,
            source_creation_time_ns=source_identity.creation_time_ns,
            watchdog_pid=watchdog_identity.pid,
            watchdog_creation_time_ns=watchdog_identity.creation_time_ns,
            watchdog_raw_handle=duplicate,
            source_handle_closed=True,
            duplicate_requested_noninheritable=True,
            _token=_JOB_TRANSFER_RECEIPT_TOKEN,
        )

    def query_handle_inheritable(self, handle: int) -> bool:
        flags = ctypes.wintypes.DWORD()
        self.kernel32.GetHandleInformation.argtypes = [
            ctypes.wintypes.HANDLE,
            ctypes.POINTER(ctypes.wintypes.DWORD),
        ]
        self.kernel32.GetHandleInformation.restype = ctypes.wintypes.BOOL
        if not self.kernel32.GetHandleInformation(handle, ctypes.byref(flags)):
            raise Gate12C2ResourceQualificationError(
                "could not query watchdog Job handle flags"
            )
        return bool(flags.value & self.HANDLE_FLAG_INHERIT)

    def verify_job_handle(self, handle: int) -> bool:
        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        returned = ctypes.wintypes.DWORD()
        self.kernel32.QueryInformationJobObject.argtypes = [
            ctypes.wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.wintypes.DWORD,
            ctypes.POINTER(ctypes.wintypes.DWORD),
        ]
        self.kernel32.QueryInformationJobObject.restype = ctypes.wintypes.BOOL
        if not self.kernel32.QueryInformationJobObject(
            handle,
            self.JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
            ctypes.byref(returned),
        ):
            raise Gate12C2ResourceQualificationError(
                "watchdog handle is not a verified Job Object"
            )
        required = (
            self.JOB_OBJECT_LIMIT_JOB_MEMORY
            | self.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        return (
            info.BasicLimitInformation.LimitFlags & required
        ) == required

    def close_handle(self, handle: int) -> None:
        self.kernel32.CloseHandle.argtypes = [ctypes.wintypes.HANDLE]
        self.kernel32.CloseHandle.restype = ctypes.wintypes.BOOL
        if not self.kernel32.CloseHandle(handle):
            raise Gate12C2ResourceQualificationError(
                "could not close Job handle"
            )


_WATCHDOG_HANDLE_CONSTRUCTOR_TOKEN = object()


class WatchdogOwnedWindowsJobHandle:
    """Final watchdog handle established from a verified transfer receipt."""

    inheritable = False

    def __init__(
        self,
        raw_handle: int,
        api: WindowsJobApi,
        ownership_proof: JobHandleOwnershipProof,
        *,
        _token: object | None = None,
    ) -> None:
        if _token is not _WATCHDOG_HANDLE_CONSTRUCTOR_TOKEN:
            raise Gate12C2ResourceQualificationError(
                "direct watchdog Job-handle construction is forbidden"
            )
        if (
            type(raw_handle) is not int
            or raw_handle <= 0
            or ownership_proof.watchdog_raw_handle != raw_handle
        ):
            raise Gate12C2ResourceQualificationError(
                "watchdog Job handle is invalid"
            )
        self._raw_handle = raw_handle
        self._api = api
        self.ownership_proof = ownership_proof
        self._closed = False

    @classmethod
    def from_transfer_receipt(
        cls,
        receipt: JobHandleTransferReceipt,
        *,
        api: WindowsJobApi,
        current_identity: ProcessIdentity,
    ) -> "WatchdogOwnedWindowsJobHandle":
        if type(receipt) is not JobHandleTransferReceipt:
            raise Gate12C2ResourceQualificationError(
                "watchdog Job transfer receipt type is invalid"
            )
        if (
            receipt.watchdog_pid != current_identity.pid
            or receipt.watchdog_creation_time_ns
            != current_identity.creation_time_ns
        ):
            raise Gate12C2ResourceQualificationError(
                "Job transfer receipt targets another watchdog identity"
            )
        if api.query_handle_inheritable(receipt.watchdog_raw_handle):
            raise Gate12C2ResourceQualificationError(
                "watchdog Job handle remains inheritable"
            )
        if api.verify_job_handle(receipt.watchdog_raw_handle) is not True:
            raise Gate12C2ResourceQualificationError(
                "watchdog Job handle does not carry the frozen limits"
            )
        proof = JobHandleOwnershipProof(
            source_pid=receipt.source_pid,
            source_creation_time_ns=receipt.source_creation_time_ns,
            watchdog_pid=receipt.watchdog_pid,
            watchdog_creation_time_ns=receipt.watchdog_creation_time_ns,
            watchdog_raw_handle=receipt.watchdog_raw_handle,
            source_handle_closed=receipt.source_handle_closed,
            target_handle_noninheritable=True,
            target_handle_valid_job=True,
        )
        return cls(
            receipt.watchdog_raw_handle,
            api,
            proof,
            _token=_WATCHDOG_HANDLE_CONSTRUCTOR_TOKEN,
        )

    @property
    def sole_owner_verified(self) -> bool:
        return (
            self.ownership_proof.source_handle_closed
            and self.ownership_proof.target_handle_noninheritable
            and self.ownership_proof.target_handle_valid_job
        )

    def close_for_kill(self) -> None:
        if not self._closed:
            self._api.close_handle(self._raw_handle)
            self._closed = True
