#!/usr/bin/env python3
"""Fail-closed primitives for Gate12C-2 replacement resource qualification.

This module is implementation-only.  It cannot issue preflight or authorization
receipts, extract baseline commitments, or run the replacement replay.  It
implements only the reviewed primitives: watchdog-local unnamed Job ownership,
atomic-at-creation suspended-child containment, page-aligned P/S/M/J limits,
pending-only telemetry publication, launch deadline enforcement, no-handle
guardian behavior, legacy-terminal classification, and resource verification.
No function in this module authorizes these primitives to run.
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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


TELEMETRY_SCHEMA = "gate12c2_resource_telemetry_record_v0.5"
TELEMETRY_PUBLICATION_SCHEMA = (
    "gate12c2_resource_telemetry_publication_receipt_v0.7"
)
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
WINDOWS_TO_UNIX_EPOCH_100NS = 116_444_736_000_000_000

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
    "WATCHDOG_SOLE_HANDLE_CONFIRMED",
    "CHILD_CREATED_SUSPENDED",
    "JOB_ASSIGNED",
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
    "WATCHDOG_SOLE_HANDLE_CONFIRMED",
    "CHILD_CREATED_SUSPENDED",
    "JOB_ASSIGNED",
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
    (
        "PRELAUNCH",
        "WATCHDOG_SOLE_HANDLE_CONFIRMED",
        "WATCHDOG_READY",
    ),
    ("WATCHDOG_READY", "CHILD_CREATED_SUSPENDED", "CHILD_SUSPENDED"),
    ("CHILD_SUSPENDED", "JOB_ASSIGNED", "JOB_ASSIGNED"),
    (
        "JOB_ASSIGNED",
        "JOINT_PRE_RESUME_RECEIPT_SEALED",
        "JOB_ASSIGNED",
    ),
    (
        "JOB_ASSIGNED",
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


@dataclass(frozen=True, init=False)
class ResourceMemoryGeometry:
    """Page-aligned memory geometry measured independently by OS processes."""

    physical_ram_bytes: int
    native_page_size_bytes: int
    mathematical_memory_limit_bytes: int
    effective_job_memory_limit_bytes: int
    rounding_delta_bytes: int

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise Gate12C2ResourceQualificationError(
            "direct resource-memory geometry construction is forbidden"
        )

    def _validate(self) -> None:
        physical = _strict_int(
            getattr(self, "physical_ram_bytes", None),
            label="physical RAM",
            maximum=99_999_999_999_999,
        )
        page = _strict_int(
            getattr(self, "native_page_size_bytes", None),
            label="native page size",
            maximum=16_777_216,
        )
        if physical <= 0 or page <= 0 or page & (page - 1):
            raise Gate12C2ResourceQualificationError(
                "resource-memory geometry is invalid"
            )
        declared_mathematical = _strict_int(
            getattr(self, "mathematical_memory_limit_bytes", None),
            label="mathematical memory limit",
            maximum=99_999_999_999_999,
        )
        declared_effective = _strict_int(
            getattr(self, "effective_job_memory_limit_bytes", None),
            label="effective Job memory limit",
            maximum=99_999_999_999_999,
        )
        declared_delta = _strict_int(
            getattr(self, "rounding_delta_bytes", None),
            label="Job memory rounding delta",
            maximum=16_777_216,
        )
        mathematical = (3 * physical) // 4
        effective = (mathematical // page) * page
        if (
            effective <= 0
            or effective > mathematical
            or declared_mathematical != mathematical
            or declared_effective != effective
            or declared_delta != mathematical - effective
        ):
            raise Gate12C2ResourceQualificationError(
                "resource-memory geometry does not match P/S/M/J"
            )


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



def derive_resource_memory_geometry(
    physical_ram_bytes: int,
    native_page_size_bytes: int,
) -> ResourceMemoryGeometry:
    """Derive the frozen P/S/M/J geometry without caller-selected limits."""

    physical = _strict_int(
        physical_ram_bytes,
        label="physical RAM",
        maximum=99_999_999_999_999,
    )
    page = _strict_int(
        native_page_size_bytes,
        label="native page size",
        maximum=16_777_216,
    )
    if physical <= 0 or page <= 0 or page & (page - 1):
        raise Gate12C2ResourceQualificationError(
            "physical RAM and native page size are invalid"
        )
    mathematical = (3 * physical) // 4
    effective = (mathematical // page) * page
    if effective <= 0 or effective > mathematical:
        raise Gate12C2ResourceQualificationError(
            "page-aligned Job memory limit is invalid"
        )
    geometry = object.__new__(ResourceMemoryGeometry)
    values = {
        "physical_ram_bytes": physical,
        "native_page_size_bytes": page,
        "mathematical_memory_limit_bytes": mathematical,
        "effective_job_memory_limit_bytes": effective,
        "rounding_delta_bytes": mathematical - effective,
    }
    for field, value in values.items():
        object.__setattr__(geometry, field, value)
    geometry._validate()
    return geometry


def verify_resource_memory_geometry_match(
    preflight: ResourceMemoryGeometry,
    watchdog: ResourceMemoryGeometry,
) -> dict[str, Any]:
    """Require independently measured preflight and watchdog P/S/M/J equality."""

    if (
        type(preflight) is not ResourceMemoryGeometry
        or type(watchdog) is not ResourceMemoryGeometry
    ):
        raise Gate12C2ResourceQualificationError(
            "resource-memory geometry type is invalid"
        )
    preflight._validate()
    watchdog._validate()
    fields = (
        "physical_ram_bytes",
        "native_page_size_bytes",
        "mathematical_memory_limit_bytes",
        "effective_job_memory_limit_bytes",
        "rounding_delta_bytes",
    )
    if any(
        getattr(preflight, field) != getattr(watchdog, field)
        for field in fields
    ):
        raise Gate12C2ResourceQualificationError(
            "preflight and watchdog P/S/M/J differ"
        )
    return {
        "status": "pass",
        "physical_ram_bytes": watchdog.physical_ram_bytes,
        "native_page_size_bytes": watchdog.native_page_size_bytes,
        "mathematical_memory_limit_bytes": (
            watchdog.mathematical_memory_limit_bytes
        ),
        "effective_job_memory_limit_bytes": (
            watchdog.effective_job_memory_limit_bytes
        ),
        "rounding_delta_bytes": watchdog.rounding_delta_bytes,
        "scientific_values_emitted": False,
    }

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
    try:
        exact_fields = set(record)
    except Exception:
        raise Gate12C2ResourceQualificationError(
            "telemetry record mapping is invalid"
        ) from None
    if exact_fields != set(LONG_FIELDS_WITHOUT_DIGEST):
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


def _accept_telemetry_line(
    line: bytes,
    *,
    state: _TelemetryState,
    first_line: bool,
) -> None:
    if line == b"\n" or not line.endswith(b"\n"):
        raise Gate12C2ResourceQualificationError(
            "telemetry contains a blank or incomplete record"
        )
    if len(line) > MAXIMUM_RECORD_BYTES:
        raise Gate12C2ResourceQualificationError(
            "telemetry record exceeds the frozen byte limit"
        )
    if b"\r" in line or (first_line and line.startswith(b"\xef\xbb\xbf")):
        raise Gate12C2ResourceQualificationError(
            "telemetry contains forbidden CR or BOM bytes"
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
        WIRE_TO_LONG[key]: value for key, value in without_digest.items()
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


def _finalize_telemetry_verification(
    *,
    state: _TelemetryState,
    telemetry_file_sha256: str,
    byte_count: int,
) -> dict[str, Any]:
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
        "telemetry_file_sha256": telemetry_file_sha256,
        "final_record_sha256": state.previous_digest,
        "byte_count": byte_count,
        "scientific_values_emitted": False,
    }


def decode_and_verify_telemetry(
    payload: bytes,
    *,
    require_terminal: bool = True,
) -> dict[str, Any]:
    """Strictly verify an in-memory telemetry JSONL byte stream."""

    if require_terminal is not True:
        raise Gate12C2ResourceQualificationError(
            "partial telemetry verification is forbidden"
        )
    if not payload or not payload.endswith(b"\n"):
        raise Gate12C2ResourceQualificationError(
            "telemetry is empty or lacks the final LF"
        )
    state = _TelemetryState()
    for index, line in enumerate(payload.splitlines(keepends=True)):
        _accept_telemetry_line(
            line, state=state, first_line=index == 0
        )
    return _finalize_telemetry_verification(
        state=state,
        telemetry_file_sha256=sha256_bytes(payload),
        byte_count=len(payload),
    )


def verify_telemetry_file(path: Path) -> dict[str, Any]:
    """Stream-verify telemetry without loading its bounded 1.33 GB surface."""

    selected = Path(path).resolve()
    if not selected.is_file():
        raise Gate12C2ResourceQualificationError(
            "telemetry file is absent"
        )
    state = _TelemetryState()
    hasher = hashlib.sha256()
    byte_count = 0
    line_count = 0
    try:
        with selected.open("rb", buffering=1024 * 1024) as handle:
            while True:
                line = handle.readline(MAXIMUM_RECORD_BYTES + 2)
                if not line:
                    break
                _accept_telemetry_line(
                    line, state=state, first_line=line_count == 0
                )
                hasher.update(line)
                byte_count += len(line)
                line_count += 1
    except Gate12C2ResourceQualificationError:
        raise
    except OSError:
        raise Gate12C2ResourceQualificationError(
            "telemetry file could not be streamed"
        ) from None
    if line_count == 0:
        raise Gate12C2ResourceQualificationError("telemetry is empty")
    return _finalize_telemetry_verification(
        state=state,
        telemetry_file_sha256=hasher.hexdigest(),
        byte_count=byte_count,
    )


def _telemetry_publication_payload(
    *,
    pending_path: Path,
    final_path: Path,
    attempt_identity_sha256: str,
    verified: Mapping[str, Any],
    byte_count: int,
) -> dict[str, Any]:
    terminal_events = {
        "RESOURCE_MONITORING_COMPLETE": "MONITORING_COMPLETED",
        "RESOURCE_MONITORING_FAILED": "FAILURE_DETECTED",
    }
    terminal_state = verified["terminal_state"]
    terminal_event = terminal_events.get(terminal_state)
    if not is_sha256(attempt_identity_sha256) or terminal_event is None:
        raise Gate12C2ResourceQualificationError(
            "telemetry publication identity is invalid"
        )
    payload = {
        "schema_version": TELEMETRY_PUBLICATION_SCHEMA,
        "attempt_identity_sha256": attempt_identity_sha256,
        "pending_path": str(pending_path),
        "final_path": str(final_path),
        "byte_count": byte_count,
        "telemetry_file_sha256": verified["telemetry_file_sha256"],
        "final_record_sha256": verified["final_record_sha256"],
        "record_count": verified["record_count"],
        "terminal_state": terminal_state,
        "terminal_event_code": terminal_event,
        "clean_close_verified": True,
        "publish_api": (
            "MoveFileExW" if os.name == "nt" else "link_then_unlink_test_fallback"
        ),
        "publish_flags": ["MOVEFILE_WRITE_THROUGH"],
        "move_result": "success",
        "replace_existing": False,
        "pending_absent": True,
        "final_present": True,
        "scientific_values_emitted": False,
    }
    payload["payload_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def verify_telemetry_publication(
    *,
    pending_path: Path,
    final_path: Path,
    expected_attempt_identity_sha256: str,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the final non-replacing publication and its closed receipt."""

    pending = Path(pending_path).resolve()
    final = Path(final_path).resolve()
    required = {
        "schema_version",
        "attempt_identity_sha256",
        "pending_path",
        "final_path",
        "byte_count",
        "telemetry_file_sha256",
        "final_record_sha256",
        "record_count",
        "terminal_state",
        "terminal_event_code",
        "clean_close_verified",
        "publish_api",
        "publish_flags",
        "move_result",
        "replace_existing",
        "pending_absent",
        "final_present",
        "scientific_values_emitted",
        "payload_sha256",
    }
    if type(receipt) is not dict or set(receipt) != required:
        raise Gate12C2ResourceQualificationError(
            "telemetry publication receipt schema is invalid"
        )
    payload = dict(receipt)
    declared_digest = payload.pop("payload_sha256")
    if not is_sha256(declared_digest) or sha256_bytes(
        canonical_json_bytes(payload)
    ) != declared_digest:
        raise Gate12C2ResourceQualificationError(
            "telemetry publication receipt digest mismatch"
        )
    if (
        not is_sha256(expected_attempt_identity_sha256)
        or receipt["attempt_identity_sha256"]
        != expected_attempt_identity_sha256
        or type(receipt["byte_count"]) is not int
        or receipt["byte_count"] <= 0
        or type(receipt["record_count"]) is not int
        or receipt["record_count"] <= 0
        or not is_sha256(receipt["telemetry_file_sha256"])
        or not is_sha256(receipt["final_record_sha256"])
        or receipt["terminal_state"] not in TERMINAL_STATES
    ):
        raise Gate12C2ResourceQualificationError(
            "telemetry publication receipt values are invalid"
        )
    expected_api = (
        "MoveFileExW" if os.name == "nt" else "link_then_unlink_test_fallback"
    )
    expected_terminal_event = {
        "RESOURCE_MONITORING_COMPLETE": "MONITORING_COMPLETED",
        "RESOURCE_MONITORING_FAILED": "FAILURE_DETECTED",
    }[receipt["terminal_state"]]
    if (
        receipt["schema_version"] != TELEMETRY_PUBLICATION_SCHEMA
        or receipt["pending_path"] != str(pending)
        or receipt["final_path"] != str(final)
        or receipt["terminal_event_code"] != expected_terminal_event
        or receipt["clean_close_verified"] is not True
        or receipt["publish_api"] != expected_api
        or receipt["publish_flags"] != ["MOVEFILE_WRITE_THROUGH"]
        or receipt["move_result"] != "success"
        or receipt["replace_existing"] is not False
        or receipt["pending_absent"] is not True
        or receipt["final_present"] is not True
        or receipt["scientific_values_emitted"] is not False
    ):
        raise Gate12C2ResourceQualificationError(
            "telemetry publication receipt is not frozen"
        )
    if pending.exists() or not final.is_file():
        raise Gate12C2ResourceQualificationError(
            "published telemetry path surface is invalid"
        )
    verified = verify_telemetry_file(final)
    if (
        type(receipt["byte_count"]) is not int
        or type(receipt["record_count"]) is not int
        or receipt["byte_count"] != verified["byte_count"]
        or receipt["telemetry_file_sha256"]
        != verified["telemetry_file_sha256"]
        or receipt["final_record_sha256"]
        != verified["final_record_sha256"]
        or receipt["record_count"] != verified["record_count"]
        or receipt["terminal_state"] != verified["terminal_state"]
    ):
        raise Gate12C2ResourceQualificationError(
            "published telemetry differs from its receipt"
        )
    return {
        "status": "pass",
        "payload_sha256": str(declared_digest),
        "attempt_identity_sha256": expected_attempt_identity_sha256,
        "telemetry_file_sha256": verified["telemetry_file_sha256"],
        "record_count": verified["record_count"],
        "terminal_state": verified["terminal_state"],
        "terminal_event_code": expected_terminal_event,
        "clean_close_verified": True,
        "move_result": "success",
        "scientific_values_emitted": False,
    }


class AppendOnlyTelemetryWriter:
    """Write only pending telemetry, then publish it without replacement."""

    def __init__(
        self,
        pending_path: Path,
        final_path: Path,
        *,
        attempt_identity_sha256: str,
        fsync: Callable[[int], None] = os.fsync,
    ) -> None:
        self.pending_path = Path(pending_path).resolve()
        self.final_path = Path(final_path).resolve()
        if not is_sha256(attempt_identity_sha256):
            raise Gate12C2ResourceQualificationError(
                "telemetry attempt identity is invalid"
            )
        self.attempt_identity_sha256 = attempt_identity_sha256
        if (
            self.pending_path.name != "telemetry.jsonl.pending"
            or self.final_path.name != "telemetry.jsonl"
            or self.pending_path.parent != self.final_path.parent
            or self.pending_path == self.final_path
        ):
            raise Gate12C2ResourceQualificationError(
                "telemetry pending/final paths are not the frozen pair"
            )
        self.pending_path.parent.mkdir(parents=True, exist_ok=True)
        if self.pending_path.exists() or self.final_path.exists():
            raise FileExistsError("telemetry pending/final path already exists")
        self.path = self.pending_path
        self._handle = self._open_exclusive_handle(self.pending_path)
        self._fsync = fsync
        self._state = _TelemetryState()
        self._closed = False
        self._failed = False
        self._published = False
        self._publication_receipt: dict[str, Any] | None = None

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
                "telemetry pending file could not be opened exclusively"
            )
        try:
            descriptor = msvcrt.open_osfhandle(
                int(raw), os.O_WRONLY | getattr(os, "O_BINARY", 0)
            )
        except Exception:
            kernel32.CloseHandle(raw)
            raise Gate12C2ResourceQualificationError(
                "telemetry pending file-handle transfer failed"
            ) from None
        return os.fdopen(descriptor, "wb", buffering=0)

    @staticmethod
    def _publish_nonreplace(pending: Path, final: Path) -> None:
        if final.exists():
            raise Gate12C2ResourceQualificationError(
                "telemetry final path already exists"
            )
        if os.name != "nt":
            try:
                os.link(pending, final)
                pending.unlink()
            except Exception:
                if final.exists() and pending.exists():
                    try:
                        final.unlink()
                    except Exception:
                        pass
                raise Gate12C2ResourceQualificationError(
                    "telemetry non-replacing publication failed"
                ) from None
            return
        movefile_write_through = 0x00000008
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.MoveFileExW.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
            ctypes.wintypes.DWORD,
        ]
        kernel32.MoveFileExW.restype = ctypes.wintypes.BOOL
        if not kernel32.MoveFileExW(
            str(pending), str(final), movefile_write_through
        ):
            raise Gate12C2ResourceQualificationError(
                "telemetry non-replacing publication failed"
            )

    @staticmethod
    def _quarantine_final_as_pending(pending: Path, final: Path) -> None:
        if pending.exists() or not final.exists():
            return
        try:
            if os.name != "nt":
                os.link(final, pending)
                final.unlink()
                return
            movefile_write_through = 0x00000008
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.MoveFileExW.argtypes = [
                ctypes.c_wchar_p,
                ctypes.c_wchar_p,
                ctypes.wintypes.DWORD,
            ]
            kernel32.MoveFileExW.restype = ctypes.wintypes.BOOL
            kernel32.MoveFileExW(
                str(final), str(pending), movefile_write_through
            )
        except Exception:
            pass

    def _close_pending_handle(self) -> bool:
        if self._closed:
            return True
        try:
            self._handle.close()
        except Exception:
            self._closed = bool(getattr(self._handle, "closed", False))
            self._failed = True
            return False
        self._closed = bool(getattr(self._handle, "closed", False))
        if not self._closed:
            self._failed = True
            return False
        return True

    def _abort(self) -> None:
        self._failed = True
        self._close_pending_handle()

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
        try:
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
            durable = (
                event_code != "PERIODIC_SAMPLE"
                or candidate_state.sequence % 10 == 0
            )
            written = self._handle.write(encoded)
            if written != len(encoded):
                raise OSError("short telemetry write")
            self._handle.flush()
            if durable:
                self._fsync(self._handle.fileno())
        except Exception:
            self._abort()
            raise Gate12C2ResourceQualificationError(
                "telemetry append failed"
            ) from None
        self._state = candidate_state
        return digest

    @property
    def publication_receipt(self) -> dict[str, Any] | None:
        if self._publication_receipt is None:
            return None
        return dict(self._publication_receipt)

    def close(self) -> dict[str, Any]:
        if self._published and self._publication_receipt is not None:
            return dict(self._publication_receipt)
        if self._failed:
            self._close_pending_handle()
            raise Gate12C2ResourceQualificationError(
                "telemetry writer is terminally failed"
            )
        if self._closed:
            raise Gate12C2ResourceQualificationError(
                "closed telemetry pending file was not published"
            )
        if not self._state.terminal:
            self._abort()
            raise Gate12C2ResourceQualificationError(
                "telemetry writer closed before a terminal state"
            )
        try:
            self._handle.flush()
            self._fsync(self._handle.fileno())
        except Exception:
            self._abort()
            raise Gate12C2ResourceQualificationError(
                "telemetry close failed"
            ) from None
        if not self._close_pending_handle():
            raise Gate12C2ResourceQualificationError(
                "telemetry handle close failed"
            )
        try:
            verified = verify_telemetry_file(self.pending_path)
        except Exception:
            self._failed = True
            raise Gate12C2ResourceQualificationError(
                "closed telemetry pending file failed strict verification"
            ) from None
        try:
            self._publish_nonreplace(self.pending_path, self.final_path)
            if self.pending_path.exists() or not self.final_path.is_file():
                raise Gate12C2ResourceQualificationError(
                    "telemetry publication path transition is invalid"
                )
            final_verified = verify_telemetry_file(self.final_path)
            for field in (
                "byte_count",
                "telemetry_file_sha256",
                "final_record_sha256",
                "record_count",
                "terminal_state",
            ):
                if final_verified[field] != verified[field]:
                    raise Gate12C2ResourceQualificationError(
                        "published telemetry bytes changed"
                    )
            receipt = _telemetry_publication_payload(
                pending_path=self.pending_path,
                final_path=self.final_path,
                attempt_identity_sha256=self.attempt_identity_sha256,
                verified=verified,
                byte_count=int(verified["byte_count"]),
            )
            verify_telemetry_publication(
                pending_path=self.pending_path,
                final_path=self.final_path,
                expected_attempt_identity_sha256=(
                    self.attempt_identity_sha256
                ),
                receipt=receipt,
            )
        except Exception:
            self._failed = True
            self._quarantine_final_as_pending(
                self.pending_path, self.final_path
            )
            raise Gate12C2ResourceQualificationError(
                "telemetry publication failed"
            ) from None
        self._published = True
        self._publication_receipt = receipt
        return dict(receipt)

    def __enter__(self) -> "AppendOnlyTelemetryWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del tb
        if exc_type is not None or exc is not None:
            self._abort()
            return
        self.close()


class LaunchDeadlineWatchdog:
    """Independent strict deadline owner and sole Job-kill actuator."""

    def __init__(
        self,
        local_launch: "WatchdogLocalWindowsJobLaunch",
        *,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if type(local_launch) is not WatchdogLocalWindowsJobLaunch:
            raise Gate12C2ResourceQualificationError(
                "watchdog-local launch state type is invalid"
            )
        try:
            local_launch.reverify_for_watchdog()
        except Exception:
            try:
                local_launch.close_for_kill()
            except Exception:
                raise Gate12C2ResourceQualificationError(
                    "watchdog-local launch verification and cleanup failed"
                ) from None
            raise Gate12C2ResourceQualificationError(
                "watchdog-local launch verification failed"
            ) from None
        self._local_launch = local_launch
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
                    self._local_launch.close_for_kill()
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

    def resume_and_arm(self) -> int:
        if self.resume_success_monotonic_ns is not None:
            self._terminate("scientific child was resumed more than once")
        try:
            previous_suspend_count = (
                self._local_launch.resume_suspended_child()
            )
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
        if self.pid == 0 or self.creation_time_ns == 0:
            raise Gate12C2ResourceQualificationError(
                "process identity values cannot be zero"
            )


class NoHandleGuardian:
    """Observe closed process identities without owning or reopening the Job."""

    __slots__ = ("_identities",)

    def __init__(self, identities: Sequence[ProcessIdentity]) -> None:
        try:
            frozen_identities = tuple(identities)
        except Exception:
            raise Gate12C2ResourceQualificationError(
                "guardian identity surface is invalid"
            ) from None
        if (
            not frozen_identities
            or any(
                type(identity) is not ProcessIdentity
                for identity in frozen_identities
            )
            or len(set(frozen_identities)) != len(frozen_identities)
        ):
            raise Gate12C2ResourceQualificationError(
                "guardian identity surface is invalid"
            )
        self._identities = frozen_identities

    @property
    def owns_job_handle(self) -> bool:
        return False

    def record_watchdog_failure(
        self,
        probe: Callable[[ProcessIdentity], str],
    ) -> dict[str, Any]:
        statuses = []
        for identity in self._identities:
            try:
                status = probe(identity)
            except Exception:
                raise Gate12C2ResourceQualificationError(
                    "guardian process probe failed"
                ) from None
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
    try:
        exact_fields = set(evidence)
    except Exception:
        raise Gate12C2ResourceQualificationError(
            "legacy closeout evidence mapping is invalid"
        ) from None
    if exact_fields != required:
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
    """Verify frozen resource scalars with distinct mathematical M and Job J."""

    required = {
        "physical_ram_bytes",
        "native_page_size_bytes",
        "mathematical_memory_limit_bytes",
        "effective_job_memory_limit_bytes",
        "rounding_delta_bytes",
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
    try:
        exact_fields = set(evidence)
    except Exception:
        raise Gate12C2ResourceQualificationError(
            "resource evidence mapping is invalid"
        ) from None
    if exact_fields != required:
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
    geometry = derive_resource_memory_geometry(
        values["physical_ram_bytes"],
        values["native_page_size_bytes"],
    )
    if (
        values["mathematical_memory_limit_bytes"]
        != geometry.mathematical_memory_limit_bytes
        or values["effective_job_memory_limit_bytes"]
        != geometry.effective_job_memory_limit_bytes
        or values["rounding_delta_bytes"] != geometry.rounding_delta_bytes
    ):
        raise Gate12C2ResourceQualificationError(
            "resource evidence P/S/M/J commitment mismatch"
        )
    available_floor = geometry.physical_ram_bytes // 10
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
        values["peak_job_memory_bytes"]
        <= geometry.effective_job_memory_limit_bytes,
        values["sampled_combined_rss_bytes"]
        <= geometry.mathematical_memory_limit_bytes,
        values["sampled_available_physical_memory_bytes"]
        >= available_floor,
        values["minimum_observed_free_bytes"] >= preflight_free // 2,
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
        "physical_ram_bytes": geometry.physical_ram_bytes,
        "native_page_size_bytes": geometry.native_page_size_bytes,
        "mathematical_memory_limit_bytes": (
            geometry.mathematical_memory_limit_bytes
        ),
        "effective_job_memory_limit_bytes": (
            geometry.effective_job_memory_limit_bytes
        ),
        "rounding_delta_bytes": geometry.rounding_delta_bytes,
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


class _JOBOBJECT_BASIC_ACCOUNTING_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", ctypes.c_longlong),
        ("TotalKernelTime", ctypes.c_longlong),
        ("ThisPeriodTotalUserTime", ctypes.c_longlong),
        ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
        ("TotalPageFaultCount", ctypes.wintypes.DWORD),
        ("TotalProcesses", ctypes.wintypes.DWORD),
        ("ActiveProcesses", ctypes.wintypes.DWORD),
        ("TotalTerminatedProcesses", ctypes.wintypes.DWORD),
    ]


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.wintypes.DWORD),
        ("dwMemoryLoad", ctypes.wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class _SYSTEM_INFO_ARCH(ctypes.Structure):
    _fields_ = [
        ("wProcessorArchitecture", ctypes.wintypes.WORD),
        ("wReserved", ctypes.wintypes.WORD),
    ]


class _SYSTEM_INFO_UNION(ctypes.Union):
    _anonymous_ = ("arch",)
    _fields_ = [
        ("dwOemId", ctypes.wintypes.DWORD),
        ("arch", _SYSTEM_INFO_ARCH),
    ]


class _SYSTEM_INFO(ctypes.Structure):
    _anonymous_ = ("identity",)
    _fields_ = [
        ("identity", _SYSTEM_INFO_UNION),
        ("dwPageSize", ctypes.wintypes.DWORD),
        ("lpMinimumApplicationAddress", ctypes.c_void_p),
        ("lpMaximumApplicationAddress", ctypes.c_void_p),
        ("dwActiveProcessorMask", ctypes.c_size_t),
        ("dwNumberOfProcessors", ctypes.wintypes.DWORD),
        ("dwProcessorType", ctypes.wintypes.DWORD),
        ("dwAllocationGranularity", ctypes.wintypes.DWORD),
        ("wProcessorLevel", ctypes.wintypes.WORD),
        ("wProcessorRevision", ctypes.wintypes.WORD),
    ]


class _STARTUPINFOW(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.wintypes.DWORD),
        ("lpReserved", ctypes.c_wchar_p),
        ("lpDesktop", ctypes.c_wchar_p),
        ("lpTitle", ctypes.c_wchar_p),
        ("dwX", ctypes.wintypes.DWORD),
        ("dwY", ctypes.wintypes.DWORD),
        ("dwXSize", ctypes.wintypes.DWORD),
        ("dwYSize", ctypes.wintypes.DWORD),
        ("dwXCountChars", ctypes.wintypes.DWORD),
        ("dwYCountChars", ctypes.wintypes.DWORD),
        ("dwFillAttribute", ctypes.wintypes.DWORD),
        ("dwFlags", ctypes.wintypes.DWORD),
        ("wShowWindow", ctypes.wintypes.WORD),
        ("cbReserved2", ctypes.wintypes.WORD),
        ("lpReserved2", ctypes.POINTER(ctypes.c_ubyte)),
        ("hStdInput", ctypes.wintypes.HANDLE),
        ("hStdOutput", ctypes.wintypes.HANDLE),
        ("hStdError", ctypes.wintypes.HANDLE),
    ]


class _STARTUPINFOEXW(ctypes.Structure):
    _fields_ = [
        ("StartupInfo", _STARTUPINFOW),
        ("lpAttributeList", ctypes.c_void_p),
    ]


class _PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("hProcess", ctypes.wintypes.HANDLE),
        ("hThread", ctypes.wintypes.HANDLE),
        ("dwProcessId", ctypes.wintypes.DWORD),
        ("dwThreadId", ctypes.wintypes.DWORD),
    ]


def _raw_handle_value(value: object) -> int:
    if type(value) is int:
        return value
    raw = getattr(value, "value", None)
    if type(raw) is int:
        return raw
    try:
        converted = int(value)  # type: ignore[arg-type]
    except Exception:
        converted = 0
    return converted


class _JobListAttributeStorage:
    """Exact one-entry PROC_THREAD_ATTRIBUTE_JOB_LIST storage lifecycle."""

    __slots__ = (
        "_api",
        "_attribute_bytes",
        "_buffer",
        "_deleted",
        "_delete_attempted",
        "_job_array",
        "_list_pointer",
    )

    def __init__(self, api: "WindowsJobApi", job_handle: int) -> None:
        self._api = api
        self._attribute_bytes = 0
        self._buffer: Any = None
        self._job_array: Any = None
        self._list_pointer = ctypes.c_void_p()
        self._deleted = False
        self._delete_attempted = False
        self._initialize(job_handle)

    def _initialize(self, job_handle: int) -> None:
        kernel32 = self._api.kernel32
        initialize = kernel32.InitializeProcThreadAttributeList
        initialize.argtypes = [
            ctypes.c_void_p,
            ctypes.wintypes.DWORD,
            ctypes.wintypes.DWORD,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        initialize.restype = ctypes.wintypes.BOOL
        update = kernel32.UpdateProcThreadAttribute
        update.argtypes = [
            ctypes.c_void_p,
            ctypes.wintypes.DWORD,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        update.restype = ctypes.wintypes.BOOL

        required_bytes = ctypes.c_size_t(0)
        ctypes.set_last_error(0)
        first_result = initialize(
            None, 1, 0, ctypes.byref(required_bytes)
        )
        first_error = ctypes.get_last_error()
        if (
            bool(first_result)
            or first_error != self._api.ERROR_INSUFFICIENT_BUFFER
            or required_bytes.value <= 0
        ):
            raise Gate12C2ResourceQualificationError(
                "Job-list attribute sizing protocol failed"
            )
        self._attribute_bytes = int(required_bytes.value)
        self._buffer = (ctypes.c_ubyte * self._attribute_bytes)()
        self._list_pointer = ctypes.cast(self._buffer, ctypes.c_void_p)
        second_bytes = ctypes.c_size_t(self._attribute_bytes)
        if not initialize(
            self._list_pointer, 1, 0, ctypes.byref(second_bytes)
        ):
            raise Gate12C2ResourceQualificationError(
                "Job-list attribute initialization failed"
            )
        if int(second_bytes.value) != self._attribute_bytes:
            self.delete_once()
            raise Gate12C2ResourceQualificationError(
                "Job-list attribute size changed during initialization"
            )
        try:
            self._job_array = (ctypes.wintypes.HANDLE * 1)(
                ctypes.wintypes.HANDLE(job_handle)
            )
            updated = update(
                self._list_pointer,
                0,
                self._api.PROC_THREAD_ATTRIBUTE_JOB_LIST,
                ctypes.cast(self._job_array, ctypes.c_void_p),
                ctypes.sizeof(ctypes.wintypes.HANDLE),
                None,
                None,
            )
        except Exception:
            try:
                self.delete_once()
            except Exception:
                raise Gate12C2ResourceQualificationError(
                    "Job-list process attribute update and cleanup failed"
                ) from None
            raise Gate12C2ResourceQualificationError(
                "Job-list process attribute update failed"
            ) from None
        if not updated:
            self.delete_once()
            raise Gate12C2ResourceQualificationError(
                "Job-list process attribute update failed"
            )

    @property
    def list_pointer(self) -> ctypes.c_void_p:
        if self._delete_attempted:
            raise Gate12C2ResourceQualificationError(
                "Job-list attribute storage is already deleted"
            )
        return self._list_pointer

    @property
    def attribute_bytes(self) -> int:
        return self._attribute_bytes

    @property
    def job_array_value(self) -> int:
        if self._job_array is None:
            return 0
        return _raw_handle_value(self._job_array[0])

    @property
    def delete_attempted(self) -> bool:
        return self._delete_attempted

    @property
    def deleted(self) -> bool:
        return self._deleted

    def delete_once(self) -> None:
        if self._delete_attempted:
            raise Gate12C2ResourceQualificationError(
                "Job-list attribute storage deletion was attempted more than once"
            )
        self._delete_attempted = True
        delete = self._api.kernel32.DeleteProcThreadAttributeList
        delete.argtypes = [ctypes.c_void_p]
        delete.restype = None
        delete(self._list_pointer)
        self._deleted = True


class WindowsJobApi:
    """Reviewed Win32 adapter used only inside the watchdog OS process."""

    JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION = 1
    JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
    HANDLE_FLAG_INHERIT = 0x00000001
    ERROR_INSUFFICIENT_BUFFER = 122
    PROC_THREAD_ATTRIBUTE_JOB_LIST = 0x0002000D
    CREATE_SUSPENDED = 0x00000004
    EXTENDED_STARTUPINFO_PRESENT = 0x00080000
    CREATEPROCESS_FLAGS = CREATE_SUSPENDED | EXTENDED_STARTUPINFO_PRESENT
    INVALID_SUSPEND_COUNT = 0xFFFFFFFF

    def __init__(self) -> None:
        if os.name != "nt":
            raise Gate12C2ResourceQualificationError(
                "Windows Job Object support is unavailable"
            )
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    def measure_resource_geometry(self) -> ResourceMemoryGeometry:
        memory = _MEMORYSTATUSEX()
        memory.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        self.kernel32.GlobalMemoryStatusEx.argtypes = [
            ctypes.POINTER(_MEMORYSTATUSEX)
        ]
        self.kernel32.GlobalMemoryStatusEx.restype = ctypes.wintypes.BOOL
        if not self.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            raise Gate12C2ResourceQualificationError(
                "GlobalMemoryStatusEx failed"
            )
        system = _SYSTEM_INFO()
        self.kernel32.GetNativeSystemInfo.argtypes = [
            ctypes.POINTER(_SYSTEM_INFO)
        ]
        self.kernel32.GetNativeSystemInfo.restype = None
        self.kernel32.GetNativeSystemInfo(ctypes.byref(system))
        return derive_resource_memory_geometry(
            int(memory.ullTotalPhys),
            int(system.dwPageSize),
        )

    def verify_current_process_outside_job(self) -> None:
        self.kernel32.GetCurrentProcess.argtypes = []
        self.kernel32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
        self.kernel32.IsProcessInJob.argtypes = [
            ctypes.wintypes.HANDLE,
            ctypes.wintypes.HANDLE,
            ctypes.POINTER(ctypes.wintypes.BOOL),
        ]
        self.kernel32.IsProcessInJob.restype = ctypes.wintypes.BOOL
        in_job = ctypes.wintypes.BOOL()
        if not self.kernel32.IsProcessInJob(
            self.kernel32.GetCurrentProcess(), None, ctypes.byref(in_job)
        ):
            raise Gate12C2ResourceQualificationError(
                "current-process Job membership is unknown"
            )
        if bool(in_job.value):
            raise Gate12C2ResourceQualificationError(
                "watchdog process is already assigned to a Job"
            )

    def _create_watchdog_local_job(
        self, geometry: ResourceMemoryGeometry
    ) -> int:
        geometry._validate()
        self.kernel32.CreateJobObjectW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_wchar_p,
        ]
        self.kernel32.CreateJobObjectW.restype = ctypes.wintypes.HANDLE
        raw = self.kernel32.CreateJobObjectW(None, None)
        handle = _raw_handle_value(raw)
        if handle <= 0:
            raise Gate12C2ResourceQualificationError(
                "could not create unnamed Job Object"
            )
        try:
            if self.query_handle_inheritable(handle):
                raise Gate12C2ResourceQualificationError(
                    "watchdog-local Job handle is inheritable"
                )
            info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = (
                self.JOB_OBJECT_LIMIT_JOB_MEMORY
                | self.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            )
            info.JobMemoryLimit = geometry.effective_job_memory_limit_bytes
            self.kernel32.SetInformationJobObject.argtypes = [
                ctypes.wintypes.HANDLE,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.wintypes.DWORD,
            ]
            self.kernel32.SetInformationJobObject.restype = (
                ctypes.wintypes.BOOL
            )
            if not self.kernel32.SetInformationJobObject(
                handle,
                self.JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                ctypes.byref(info),
                ctypes.sizeof(info),
            ):
                raise Gate12C2ResourceQualificationError(
                    "could not apply page-aligned Job limits"
                )
            if not self.verify_job_handle(handle, geometry):
                raise Gate12C2ResourceQualificationError(
                    "post-set Job limit query differs from J"
                )
        except Exception:
            self.close_handle_verified(handle)
            raise
        return handle

    def _build_job_list_storage(
        self, job_handle: int
    ) -> _JobListAttributeStorage:
        return _JobListAttributeStorage(self, job_handle)

    def probe_job_list_attribute_support(self) -> dict[str, Any]:
        """Exercise the exact attribute protocol without creating a child."""

        self.verify_current_process_outside_job()
        geometry = self.measure_resource_geometry()
        job_handle = self._create_watchdog_local_job(geometry)
        storage: _JobListAttributeStorage | None = None
        deleted = False
        probe_failed = False
        try:
            storage = self._build_job_list_storage(job_handle)
            if storage.job_array_value != job_handle:
                raise Gate12C2ResourceQualificationError(
                    "support-probe Job-list value changed"
                )
            storage.delete_once()
            deleted = storage.deleted
        except Exception:
            probe_failed = True
        finally:
            if storage is not None and not storage.delete_attempted:
                try:
                    storage.delete_once()
                except Exception:
                    probe_failed = True
                else:
                    deleted = storage.deleted
            self.close_handle_verified(job_handle)
        if probe_failed or not deleted:
            raise Gate12C2ResourceQualificationError(
                "support-probe attribute cleanup was not verified"
            ) from None
        return {
            "status": "pass",
            "entry_count": 1,
            "attribute_bytes": storage.attribute_bytes,
            "attribute_deleted_count": 1,
            "temporary_job_closed": True,
            "scientific_child_created": False,
            "scientific_values_emitted": False,
        }

    def launch_scientific_child_suspended(
        self,
        *,
        preflight_geometry: ResourceMemoryGeometry,
        application_name: str | None,
        command_line: str,
        current_directory: Path,
    ) -> "WatchdogLocalWindowsJobLaunch":
        """Create the scientific root already assigned to the local Job."""

        self.verify_current_process_outside_job()
        watchdog_geometry = self.measure_resource_geometry()
        verify_resource_memory_geometry_match(
            preflight_geometry, watchdog_geometry
        )
        if application_name is not None and (
            not isinstance(application_name, str)
            or not application_name
            or "\x00" in application_name
        ):
            raise Gate12C2ResourceQualificationError(
                "scientific application name is invalid"
            )
        if (
            not isinstance(command_line, str)
            or not command_line
            or "\x00" in command_line
            or len(command_line) >= 32_767
        ):
            raise Gate12C2ResourceQualificationError(
                "scientific command line is invalid"
            )
        cwd = Path(current_directory).resolve()
        if not cwd.is_dir():
            raise Gate12C2ResourceQualificationError(
                "scientific current directory is invalid"
            )

        job_handle = self._create_watchdog_local_job(watchdog_geometry)
        storage: _JobListAttributeStorage | None = None
        process_info = _PROCESS_INFORMATION()
        startup = _STARTUPINFOEXW()
        startup.StartupInfo.cb = ctypes.sizeof(_STARTUPINFOEXW)
        command_buffer = ctypes.create_unicode_buffer(command_line)
        create_succeeded = False
        delete_reached = False
        attribute_storage_unchanged = False
        creation_error = False
        try:
            storage = self._build_job_list_storage(job_handle)
            startup.lpAttributeList = storage.list_pointer
            self.kernel32.CreateProcessW.argtypes = [
                ctypes.c_wchar_p,
                ctypes.c_wchar_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.wintypes.BOOL,
                ctypes.wintypes.DWORD,
                ctypes.c_void_p,
                ctypes.c_wchar_p,
                ctypes.POINTER(_STARTUPINFOEXW),
                ctypes.POINTER(_PROCESS_INFORMATION),
            ]
            self.kernel32.CreateProcessW.restype = ctypes.wintypes.BOOL
            create_succeeded = bool(
                self.kernel32.CreateProcessW(
                    application_name,
                    command_buffer,
                    None,
                    None,
                    False,
                    self.CREATEPROCESS_FLAGS,
                    None,
                    str(cwd),
                    ctypes.byref(startup),
                    ctypes.byref(process_info),
                )
            )
        except Exception:
            creation_error = True
        finally:
            if storage is not None:
                try:
                    attribute_storage_unchanged = (
                        storage.job_array_value == job_handle
                        and _raw_handle_value(startup.lpAttributeList)
                        == _raw_handle_value(storage.list_pointer)
                    )
                    storage.delete_once()
                    delete_reached = attribute_storage_unchanged
                except Exception:
                    delete_reached = False

        process_handle = _raw_handle_value(process_info.hProcess)
        thread_handle = _raw_handle_value(process_info.hThread)
        if creation_error:
            self._close_residual_process_information(
                process_handle, thread_handle
            )
            self.close_handle_verified(job_handle)
            raise Gate12C2ResourceQualificationError(
                "atomic scientific-child launch preparation failed"
            ) from None
        if not create_succeeded:
            self._close_residual_process_information(
                process_handle, thread_handle
            )
            self.close_handle_verified(job_handle)
            raise Gate12C2ResourceQualificationError(
                "CreateProcessW failed for the scientific child"
            )
        if not delete_reached:
            self._close_residual_process_information(
                process_handle, thread_handle
            )
            self.close_handle_verified(job_handle)
            raise Gate12C2ResourceQualificationError(
                "process-attribute deletion was not verified"
            )
        try:
            identity = self._verify_created_child(
                job_handle=job_handle,
                process_handle=process_handle,
                thread_handle=thread_handle,
                declared_pid=int(process_info.dwProcessId),
                declared_thread_id=int(process_info.dwThreadId),
            )
        except Exception:
            self.close_handle_verified(job_handle)
            self._close_residual_process_information(
                process_handle, thread_handle
            )
            raise
        try:
            return WatchdogLocalWindowsJobLaunch._from_verified_creation(
                api=self,
                geometry=watchdog_geometry,
                job_handle=job_handle,
                process_handle=process_handle,
                thread_handle=thread_handle,
                child_identity=identity,
                thread_id=int(process_info.dwThreadId),
                attribute_bytes=storage.attribute_bytes,
            )
        except Exception:
            self.close_handle_verified(job_handle)
            self._close_residual_process_information(
                process_handle, thread_handle
            )
            raise Gate12C2ResourceQualificationError(
                "watchdog-local launch state verification failed"
            ) from None

    def reverify_suspended_child(
        self,
        *,
        job_handle: int,
        process_handle: int,
        thread_handle: int,
        expected_identity: ProcessIdentity,
        expected_thread_id: int,
    ) -> None:
        """Recheck PID, creation time, thread, membership, and suspension."""

        self.kernel32.GetProcessId.argtypes = [ctypes.wintypes.HANDLE]
        self.kernel32.GetProcessId.restype = ctypes.wintypes.DWORD
        self.kernel32.GetThreadId.argtypes = [ctypes.wintypes.HANDLE]
        self.kernel32.GetThreadId.restype = ctypes.wintypes.DWORD
        if (
            int(self.kernel32.GetProcessId(process_handle))
            != expected_identity.pid
            or self._process_creation_time_ns(process_handle)
            != expected_identity.creation_time_ns
            or int(self.kernel32.GetThreadId(thread_handle))
            != expected_thread_id
        ):
            raise Gate12C2ResourceQualificationError(
                "scientific child process identity changed"
            )
        in_job = ctypes.wintypes.BOOL()
        self.kernel32.IsProcessInJob.argtypes = [
            ctypes.wintypes.HANDLE,
            ctypes.wintypes.HANDLE,
            ctypes.POINTER(ctypes.wintypes.BOOL),
        ]
        self.kernel32.IsProcessInJob.restype = ctypes.wintypes.BOOL
        if not self.kernel32.IsProcessInJob(
            process_handle, job_handle, ctypes.byref(in_job)
        ) or not bool(in_job.value):
            raise Gate12C2ResourceQualificationError(
                "scientific child Job membership changed"
            )
        if self._query_job_accounting(job_handle) != (1, 1, 0):
            raise Gate12C2ResourceQualificationError(
                "suspended child Job process surface changed"
            )
        self._verify_thread_remains_suspended(thread_handle)

    def _verify_created_child(
        self,
        *,
        job_handle: int,
        process_handle: int,
        thread_handle: int,
        declared_pid: int,
        declared_thread_id: int,
    ) -> ProcessIdentity:
        if (
            process_handle <= 0
            or thread_handle <= 0
            or declared_pid <= 0
            or declared_thread_id <= 0
        ):
            raise Gate12C2ResourceQualificationError(
                "CreateProcessW returned invalid process information"
            )
        self.kernel32.GetProcessId.argtypes = [ctypes.wintypes.HANDLE]
        self.kernel32.GetProcessId.restype = ctypes.wintypes.DWORD
        self.kernel32.GetThreadId.argtypes = [ctypes.wintypes.HANDLE]
        self.kernel32.GetThreadId.restype = ctypes.wintypes.DWORD
        if int(self.kernel32.GetProcessId(process_handle)) != declared_pid:
            raise Gate12C2ResourceQualificationError(
                "scientific child PID identity mismatch"
            )
        if int(self.kernel32.GetThreadId(thread_handle)) != declared_thread_id:
            raise Gate12C2ResourceQualificationError(
                "scientific child thread identity mismatch"
            )
        self.kernel32.IsProcessInJob.argtypes = [
            ctypes.wintypes.HANDLE,
            ctypes.wintypes.HANDLE,
            ctypes.POINTER(ctypes.wintypes.BOOL),
        ]
        self.kernel32.IsProcessInJob.restype = ctypes.wintypes.BOOL
        in_job = ctypes.wintypes.BOOL()
        if not self.kernel32.IsProcessInJob(
            process_handle, job_handle, ctypes.byref(in_job)
        ) or not bool(in_job.value):
            raise Gate12C2ResourceQualificationError(
                "scientific child is not atomically contained in the Job"
            )
        creation_time_ns = self._process_creation_time_ns(process_handle)
        self._verify_thread_remains_suspended(thread_handle)
        accounting = self._query_job_accounting(job_handle)
        if accounting != (1, 1, 0):
            raise Gate12C2ResourceQualificationError(
                "suspended child Job process surface is not exact"
            )
        return ProcessIdentity(declared_pid, creation_time_ns)

    def _verify_thread_remains_suspended(self, thread_handle: int) -> None:
        self.kernel32.SuspendThread.argtypes = [ctypes.wintypes.HANDLE]
        self.kernel32.SuspendThread.restype = ctypes.wintypes.DWORD
        self.kernel32.ResumeThread.argtypes = [ctypes.wintypes.HANDLE]
        self.kernel32.ResumeThread.restype = ctypes.wintypes.DWORD
        previous = int(self.kernel32.SuspendThread(thread_handle))
        if previous != 1:
            if previous != self.INVALID_SUSPEND_COUNT:
                self.kernel32.ResumeThread(thread_handle)
            raise Gate12C2ResourceQualificationError(
                "scientific primary thread was not exactly suspended"
            )
        restored = int(self.kernel32.ResumeThread(thread_handle))
        if restored != 2:
            raise Gate12C2ResourceQualificationError(
                "scientific primary-thread suspend probe was not restored"
            )

    def _process_creation_time_ns(self, process_handle: int) -> int:
        creation = ctypes.wintypes.FILETIME()
        exit_time = ctypes.wintypes.FILETIME()
        kernel = ctypes.wintypes.FILETIME()
        user = ctypes.wintypes.FILETIME()
        self.kernel32.GetProcessTimes.argtypes = [
            ctypes.wintypes.HANDLE,
            ctypes.POINTER(ctypes.wintypes.FILETIME),
            ctypes.POINTER(ctypes.wintypes.FILETIME),
            ctypes.POINTER(ctypes.wintypes.FILETIME),
            ctypes.POINTER(ctypes.wintypes.FILETIME),
        ]
        self.kernel32.GetProcessTimes.restype = ctypes.wintypes.BOOL
        if not self.kernel32.GetProcessTimes(
            process_handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            raise Gate12C2ResourceQualificationError(
                "scientific process creation time is unavailable"
            )
        ticks = (
            int(creation.dwHighDateTime) << 32
        ) | int(creation.dwLowDateTime)
        if ticks <= WINDOWS_TO_UNIX_EPOCH_100NS:
            raise Gate12C2ResourceQualificationError(
                "scientific process creation time is invalid"
            )
        return (ticks - WINDOWS_TO_UNIX_EPOCH_100NS) * 100

    def _query_job_accounting(self, job_handle: int) -> tuple[int, int, int]:
        info = _JOBOBJECT_BASIC_ACCOUNTING_INFORMATION()
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
            job_handle,
            self.JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
            ctypes.byref(returned),
        ):
            raise Gate12C2ResourceQualificationError(
                "Job accounting query failed"
            )
        return (
            int(info.ActiveProcesses),
            int(info.TotalProcesses),
            int(info.TotalTerminatedProcesses),
        )

    def resume_primary_thread(self, thread_handle: int) -> int:
        self.kernel32.ResumeThread.argtypes = [ctypes.wintypes.HANDLE]
        self.kernel32.ResumeThread.restype = ctypes.wintypes.DWORD
        return int(self.kernel32.ResumeThread(thread_handle))

    def query_handle_inheritable(self, handle: int) -> bool:
        flags = ctypes.wintypes.DWORD()
        self.kernel32.GetHandleInformation.argtypes = [
            ctypes.wintypes.HANDLE,
            ctypes.POINTER(ctypes.wintypes.DWORD),
        ]
        self.kernel32.GetHandleInformation.restype = ctypes.wintypes.BOOL
        if not self.kernel32.GetHandleInformation(handle, ctypes.byref(flags)):
            raise Gate12C2ResourceQualificationError(
                "could not query Job handle flags"
            )
        return bool(flags.value & self.HANDLE_FLAG_INHERIT)

    @classmethod
    def _job_limits_match(
        cls,
        *,
        limit_flags: int,
        job_memory_limit_bytes: int,
        geometry: ResourceMemoryGeometry,
    ) -> bool:
        geometry._validate()
        required = (
            cls.JOB_OBJECT_LIMIT_JOB_MEMORY
            | cls.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        return (
            type(limit_flags) is int
            and limit_flags == required
            and type(job_memory_limit_bytes) is int
            and job_memory_limit_bytes
            == geometry.effective_job_memory_limit_bytes
        )

    def verify_job_handle(
        self, handle: int, geometry: ResourceMemoryGeometry
    ) -> bool:
        if type(geometry) is not ResourceMemoryGeometry:
            raise Gate12C2ResourceQualificationError(
                "Job verification geometry type is invalid"
            )
        geometry._validate()
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
                "handle is not a verified Job Object"
            )
        return self._job_limits_match(
            limit_flags=int(info.BasicLimitInformation.LimitFlags),
            job_memory_limit_bytes=int(info.JobMemoryLimit),
            geometry=geometry,
        )

    def close_handle(self, handle: int) -> None:
        self.kernel32.CloseHandle.argtypes = [ctypes.wintypes.HANDLE]
        self.kernel32.CloseHandle.restype = ctypes.wintypes.BOOL
        if not self.kernel32.CloseHandle(handle):
            raise Gate12C2ResourceQualificationError(
                "could not close Win32 handle"
            )

    def handle_is_closed(self, handle: int) -> bool:
        flags = ctypes.wintypes.DWORD()
        self.kernel32.GetHandleInformation.argtypes = [
            ctypes.wintypes.HANDLE,
            ctypes.POINTER(ctypes.wintypes.DWORD),
        ]
        self.kernel32.GetHandleInformation.restype = ctypes.wintypes.BOOL
        ctypes.set_last_error(0)
        if self.kernel32.GetHandleInformation(handle, ctypes.byref(flags)):
            return False
        error = ctypes.get_last_error()
        if error == 6:
            return True
        raise Gate12C2ResourceQualificationError(
            "could not verify Win32 handle closure"
        )

    def close_handle_verified(self, handle: int) -> bool:
        try:
            self.close_handle(handle)
        except Exception:
            if self.handle_is_closed(handle):
                return True
            raise Gate12C2ResourceQualificationError(
                "could not close and verify Win32 handle"
            ) from None
        if not self.handle_is_closed(handle):
            raise Gate12C2ResourceQualificationError(
                "Win32 handle remained open after close"
            )
        return True

    def _close_residual_process_information(
        self, process_handle: int, thread_handle: int
    ) -> None:
        failures = 0
        for handle in (thread_handle, process_handle):
            if handle > 0:
                try:
                    self.close_handle_verified(handle)
                except Exception:
                    failures += 1
        if failures:
            raise Gate12C2ResourceQualificationError(
                "process-information cleanup failed"
            )


def create_watchdog_local_scientific_launch(
    *,
    preflight_geometry: ResourceMemoryGeometry,
    application_name: str | None,
    command_line: str,
    current_directory: Path,
) -> "WatchdogLocalWindowsJobLaunch":
    """Production boundary: instantiate the real Win32 API in the watchdog."""

    api = WindowsJobApi()
    return api.launch_scientific_child_suspended(
        preflight_geometry=preflight_geometry,
        application_name=application_name,
        command_line=command_line,
        current_directory=current_directory,
    )


class WatchdogLocalWindowsJobLaunch:
    """Watchdog-local state holder; it is never serialized or transferred."""

    __slots__ = (
        "_api",
        "_attribute_bytes",
        "_child_identity",
        "_closed",
        "_geometry",
        "_job_handle",
        "_process_handle",
        "_resumed",
        "_thread_handle",
        "_thread_id",
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise Gate12C2ResourceQualificationError(
            "direct watchdog-local launch construction is forbidden"
        )

    @classmethod
    def _from_verified_creation(
        cls,
        *,
        api: WindowsJobApi,
        geometry: ResourceMemoryGeometry,
        job_handle: int,
        process_handle: int,
        thread_handle: int,
        child_identity: ProcessIdentity,
        thread_id: int,
        attribute_bytes: int,
    ) -> "WatchdogLocalWindowsJobLaunch":
        launch = object.__new__(cls)
        launch._api = api
        launch._geometry = geometry
        launch._job_handle = job_handle
        launch._process_handle = process_handle
        launch._thread_handle = thread_handle
        launch._child_identity = child_identity
        launch._thread_id = thread_id
        launch._attribute_bytes = attribute_bytes
        launch._resumed = False
        launch._closed = False
        launch.reverify_for_watchdog()
        return launch

    @property
    def child_identity(self) -> ProcessIdentity:
        return self._child_identity

    @property
    def resource_geometry(self) -> ResourceMemoryGeometry:
        return self._geometry

    @property
    def attribute_bytes(self) -> int:
        return self._attribute_bytes

    def reverify_for_watchdog(self) -> None:
        if self._closed or type(self._api) is not WindowsJobApi:
            raise Gate12C2ResourceQualificationError(
                "watchdog-local launch state is invalid"
            )
        self._geometry._validate()
        if self._api.query_handle_inheritable(self._job_handle):
            raise Gate12C2ResourceQualificationError(
                "watchdog-local Job handle became inheritable"
            )
        if not self._api.verify_job_handle(
            self._job_handle, self._geometry
        ):
            raise Gate12C2ResourceQualificationError(
                "watchdog-local Job limits changed"
            )
        in_job = ctypes.wintypes.BOOL()
        self._api.kernel32.IsProcessInJob.argtypes = [
            ctypes.wintypes.HANDLE,
            ctypes.wintypes.HANDLE,
            ctypes.POINTER(ctypes.wintypes.BOOL),
        ]
        self._api.kernel32.IsProcessInJob.restype = ctypes.wintypes.BOOL
        if not self._api.kernel32.IsProcessInJob(
            self._process_handle,
            self._job_handle,
            ctypes.byref(in_job),
        ) or not bool(in_job.value):
            raise Gate12C2ResourceQualificationError(
                "scientific child escaped the watchdog-local Job"
            )
        self._api.reverify_suspended_child(
            job_handle=self._job_handle,
            process_handle=self._process_handle,
            thread_handle=self._thread_handle,
            expected_identity=self._child_identity,
            expected_thread_id=self._thread_id,
        )

    def resume_suspended_child(self) -> int:
        if self._closed or self._resumed:
            raise Gate12C2ResourceQualificationError(
                "scientific child cannot be resumed"
            )
        self.reverify_for_watchdog()
        previous = self._api.resume_primary_thread(self._thread_handle)
        if previous != 1:
            self.close_for_kill()
            raise Gate12C2ResourceQualificationError(
                "scientific child resume result is invalid"
            )
        self._resumed = True
        return previous

    def close_for_kill(self) -> None:
        if not self._closed:
            if not self._api.close_handle_verified(self._job_handle):
                raise Gate12C2ResourceQualificationError(
                    "watchdog-local Job close was not verified"
                )
            self._closed = True

    def close_child_handles(self) -> None:
        failures = 0
        for field in ("_thread_handle", "_process_handle"):
            handle = int(getattr(self, field))
            if handle > 0:
                try:
                    self._api.close_handle_verified(handle)
                except Exception:
                    failures += 1
                else:
                    setattr(self, field, 0)
        if failures:
            raise Gate12C2ResourceQualificationError(
                "scientific child handle cleanup failed"
            )
