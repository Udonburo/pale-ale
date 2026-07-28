#!/usr/bin/env python3
"""Fail-closed closeout recovery for one completed Gate12C-2 payload.

This module never recomputes shards, reconstructs missing resource telemetry,
or authorizes scientific interpretation.  It can freeze byte-only incident
evidence and, under a separate single-use authorization, verify and seal the
existing payload without modifying the profile root.  Stale-lock retirement is
a later, separately authorized control-plane operation.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
import hashlib
import json
import os
import re
import socket
import stat
import subprocess
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import gate12c2_development_shards as shards
import gate12c2_draw_profile as profile


INCIDENT_MANIFEST_SCHEMA = "gate12c2_closeout_incident_byte_manifest_v0.1"
FAILURE_RECEIPT_SCHEMA = "gate12c2_closeout_failure_receipt_v0.1"
EXPOSURE_LEDGER_SCHEMA = "gate12c2_reviewer_exposure_ledger_v0.1"
FREEZE_DISCREPANCY_SCHEMA = (
    "gate12c2_incident_freeze_discrepancy_v0.1"
)
AMENDMENT_SCHEMA = "gate12c2_closeout_recovery_amendment_v0.1"
AUTHORIZATION_SCHEMA = "gate12c2_closeout_recovery_authorization_v0.4"
ATTEMPT_SCHEMA = "gate12c2_closeout_recovery_attempt_v0.1"
CONSUMPTION_SCHEMA = "gate12c2_closeout_recovery_consumption_v0.2"
TERMINAL_CLAIM_SCHEMA = "gate12c2_closeout_recovery_terminal_claim_v0.1"
PAYLOAD_SEAL_SCHEMA = "gate12c2_payload_completion_seal_v0.3"
RECOVERY_FAILURE_SCHEMA = "gate12c2_closeout_recovery_failure_v0.3"
AUTHORIZATION_MAX_AGE_SECONDS = 30 * 60
ORIGINAL_SOURCE_COMMIT = "f9bd14d91082b01d6faa628d41e094d78c1337fe"
ORIGINAL_PLAN_PAYLOAD_SHA256 = (
    "60b47d30306982fc605223a223f3b61b52c318abe95d2f4618a284976e45c9c5"
)
INCIDENT_FAILURE_CODE = "GATE12C2_CLOSEOUT_RESTORE_SCRATCH_ROOT_NOT_PROPAGATED"
PUBLIC_ERROR_CODE = "GATE12C2_CLOSEOUT_RECOVERY_REJECTED"
LOCK_NAME = profile.COORDINATOR_LOCK_NAME
REGIME_OUTER_COUNTS = {
    "S0_true_null": 128,
    "S1_known_reverse_shared_node_coupling": 64,
    "S2_null_inflation": 64,
}
DRAW_COUNTS = (255, 511, 1023)
PARTIAL_SUFFIXES = (".tmp", ".partial", ".incomplete")
PROCESS_ACTIVE = "ACTIVE"
PROCESS_DEAD = "DEAD"
PROCESS_UNKNOWN = "UNKNOWN"
FAILURE_PHASES_BY_STATE = {
    "RECOVERY_REJECTED": frozenset({"payload_verification"}),
    "RECOVERY_INTERRUPTED": frozenset(
        {
            "attempt_claimed",
            "consumption_publication",
            "post_consumption_restart",
            "payload_verification",
            "seal_publication",
        }
    ),
    "PAYLOAD_MISMATCH": frozenset({"payload_verification"}),
}
FAILURE_PHASE_CONSUMPTION = {
    "attempt_claimed": "absent",
    "consumption_publication": "either",
    "post_consumption_restart": "present",
    "payload_verification": "present",
    "seal_publication": "present",
}


class Gate12C2CloseoutRecoveryError(ValueError):
    """Raised when closeout recovery crosses a frozen boundary."""


class SanitizedArgumentParser(argparse.ArgumentParser):
    """Raise a value-free domain error instead of printing argparse input."""

    def error(self, message: str) -> None:
        del message
        raise Gate12C2CloseoutRecoveryError(PUBLIC_ERROR_CODE)


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
        raise Gate12C2CloseoutRecoveryError(
            "recovery evidence is not canonical-JSON serializable"
        ) from None


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def read_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_bytes().decode("utf-8"))
    except Exception:
        raise Gate12C2CloseoutRecoveryError(f"could not read {label}") from None
    if not isinstance(payload, dict):
        raise Gate12C2CloseoutRecoveryError(f"{label} must be a JSON object")
    return payload


def verify_self_hash(
    payload: Mapping[str, Any], *, hash_field: str, label: str
) -> str:
    candidate = dict(payload)
    claimed = candidate.pop(hash_field, None)
    if (
        not is_sha256(claimed)
        or sha256_bytes(canonical_json_bytes(candidate)) != claimed
    ):
        raise Gate12C2CloseoutRecoveryError(f"{label} self-hash mismatch")
    return str(claimed)


def write_exclusive_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    encoded = canonical_json_bytes(payload)
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # Publish a fully flushed same-directory temp file without any
            # overwrite path.  os.link is atomic and fails if another process
            # already won the single-use destination.
            os.link(temporary, destination)
        except FileExistsError:
            raise Gate12C2CloseoutRecoveryError(
                "recovery evidence output already exists"
            ) from None
        except OSError:
            raise Gate12C2CloseoutRecoveryError(
                "recovery evidence could not be published exclusively"
            ) from None
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_utc(value: object, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise Gate12C2CloseoutRecoveryError(f"{label} is invalid")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise Gate12C2CloseoutRecoveryError(f"{label} is invalid") from None
    if parsed.tzinfo is None:
        raise Gate12C2CloseoutRecoveryError(f"{label} must include timezone")
    return parsed.astimezone(timezone.utc)



def _strict_int(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise Gate12C2CloseoutRecoveryError(f"{label} must be an integer")
    return int(value)


def _query_process_identity(
    pid: int,
) -> tuple[str, dict[str, Any] | None]:
    """Return liveness plus a PID-reuse-resistant identity, without command lines."""

    if type(pid) is not int or pid <= 0:
        return PROCESS_UNKNOWN, None
    if os.name == "nt":
        process_query_limited_information = 0x1000
        error_invalid_parameter = 87
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [
            ctypes.wintypes.DWORD,
            ctypes.wintypes.BOOL,
            ctypes.wintypes.DWORD,
        ]
        kernel32.OpenProcess.restype = ctypes.wintypes.HANDLE
        kernel32.GetProcessTimes.argtypes = [
            ctypes.wintypes.HANDLE,
            ctypes.POINTER(ctypes.wintypes.FILETIME),
            ctypes.POINTER(ctypes.wintypes.FILETIME),
            ctypes.POINTER(ctypes.wintypes.FILETIME),
            ctypes.POINTER(ctypes.wintypes.FILETIME),
        ]
        kernel32.GetProcessTimes.restype = ctypes.wintypes.BOOL
        kernel32.CloseHandle.argtypes = [ctypes.wintypes.HANDLE]
        kernel32.CloseHandle.restype = ctypes.wintypes.BOOL
        handle = kernel32.OpenProcess(
            process_query_limited_information, False, pid
        )
        if not handle:
            error = ctypes.get_last_error()
            if error == error_invalid_parameter:
                return PROCESS_DEAD, None
            return PROCESS_UNKNOWN, None
        creation = ctypes.wintypes.FILETIME()
        exit_time = ctypes.wintypes.FILETIME()
        kernel = ctypes.wintypes.FILETIME()
        user = ctypes.wintypes.FILETIME()
        try:
            if not kernel32.GetProcessTimes(
                handle,
                ctypes.byref(creation),
                ctypes.byref(exit_time),
                ctypes.byref(kernel),
                ctypes.byref(user),
            ):
                return PROCESS_UNKNOWN, None
        finally:
            kernel32.CloseHandle(handle)
        exit_marker = (int(exit_time.dwHighDateTime) << 32) | int(
            exit_time.dwLowDateTime
        )
        if exit_marker != 0:
            return PROCESS_DEAD, None
        marker = (int(creation.dwHighDateTime) << 32) | int(
            creation.dwLowDateTime
        )
        return (
            PROCESS_ACTIVE,
            {
                "pid": pid,
                "identity_kind": "windows_creation_filetime",
                "start_marker": str(marker),
            },
        )
    proc_stat = Path(f"/proc/{pid}/stat")
    try:
        proc_stat.stat()
    except FileNotFoundError:
        return PROCESS_DEAD, None
    except OSError:
        return PROCESS_UNKNOWN, None
    try:
        raw = proc_stat.read_text(encoding="utf-8")
        close = raw.rfind(")")
        fields = raw[close + 2 :].split()
        marker = fields[19]
    except (OSError, IndexError, ValueError):
        return PROCESS_UNKNOWN, None
    return (
        PROCESS_ACTIVE,
        {
            "pid": pid,
            "identity_kind": "proc_start_ticks",
            "start_marker": marker,
        },
    )


def process_identity(pid: int) -> dict[str, Any] | None:
    """Return a live PID-reuse-resistant process identity, if established."""

    state, identity = _query_process_identity(pid)
    return identity if state == PROCESS_ACTIVE else None


def process_pid_state(pid: int) -> str:
    """Classify a PID as ACTIVE, DEAD, or UNKNOWN without collapsing failures."""

    state, _ = _query_process_identity(pid)
    return state


def process_identity_state(identity: Mapping[str, Any]) -> str:
    """Classify the recorded process identity as ACTIVE, DEAD, or UNKNOWN."""

    try:
        pid = _strict_int(identity.get("pid"), label="process PID")
    except Gate12C2CloseoutRecoveryError:
        return PROCESS_UNKNOWN
    state, current = _query_process_identity(pid)
    if state != PROCESS_ACTIVE:
        return state
    if current is None:
        return PROCESS_UNKNOWN
    return PROCESS_ACTIVE if dict(identity) == current else PROCESS_DEAD




def expected_root_files() -> dict[str, tuple[str, str]]:
    """Return exact relative paths mapped to (plane, file class)."""

    expected: dict[str, tuple[str, str]] = {
        "plan.json": ("protected_payload", "frozen_lineage"),
        "control/preflight.json": ("protected_payload", "frozen_lineage"),
        "control/authorization.json": ("protected_payload", "frozen_lineage"),
        "control/authorization-consumed.json": (
            "protected_payload",
            "frozen_lineage",
        ),
        LOCK_NAME: ("control_plane", "stale_coordinator_lock"),
    }
    for regime, outer_count in REGIME_OUTER_COUNTS.items():
        for draw_count in DRAW_COUNTS:
            prefix = f"runs/{regime}/draw-{draw_count}"
            expected[f"{prefix}/plan.json"] = (
                "protected_payload",
                "frozen_lineage",
            )
            expected[f"{prefix}/index.json"] = (
                "protected_payload",
                "index",
            )
            for outer_index in range(outer_count):
                expected[
                    f"{prefix}/shards/outer-{outer_index:06d}.json.gz"
                ] = (
                    "protected_payload",
                    "shard",
                )
    return expected


def expected_directories() -> set[str]:
    directories = {"control", "runs"}
    for regime in REGIME_OUTER_COUNTS:
        directories.add(f"runs/{regime}")
        for draw_count in DRAW_COUNTS:
            directories.add(f"runs/{regime}/draw-{draw_count}")
            directories.add(
                f"runs/{regime}/draw-{draw_count}/shards"
            )
    return directories


def _reparse_status(path: Path) -> bool:
    try:
        result = path.is_symlink()
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
        return bool(result or attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)
    except OSError:
        raise Gate12C2CloseoutRecoveryError(
            "could not inspect incident path metadata"
        ) from None


def _partial_status(relative_path: str) -> bool:
    lowered = relative_path.lower()
    name = Path(relative_path).name.lower()
    return bool(
        name.startswith(".") and name.endswith(".tmp")
        or any(lowered.endswith(suffix) for suffix in PARTIAL_SUFFIXES)
    )


def scan_root_bytes(root: Path) -> dict[str, Any]:
    """Hash the root as uninterpreted bytes; never parse JSON, gzip, or NPZ."""

    destination = Path(root).resolve()
    if not destination.is_dir():
        raise Gate12C2CloseoutRecoveryError("incident output root is missing")
    expected = expected_root_files()
    expected_dirs = expected_directories()
    actual_files: dict[str, Path] = {}
    directory_rows: list[dict[str, Any]] = []
    for current, directory_names, file_names in os.walk(
        destination, topdown=True, followlinks=False
    ):
        current_path = Path(current)
        for directory_name in sorted(directory_names):
            directory_path = current_path / directory_name
            relative = directory_path.relative_to(destination).as_posix()
            is_reparse = _reparse_status(directory_path)
            directory_rows.append(
                {
                    "canonical_relative_path": relative,
                    "expected": relative in expected_dirs,
                    "unexpected": relative not in expected_dirs,
                    "reparse_point": is_reparse,
                }
            )
            if is_reparse:
                directory_names.remove(directory_name)
        for file_name in sorted(file_names):
            file_path = current_path / file_name
            relative = file_path.relative_to(destination).as_posix()
            if relative in actual_files:
                raise Gate12C2CloseoutRecoveryError(
                    "duplicate canonical incident path"
                )
            actual_files[relative] = file_path
    rows: list[dict[str, Any]] = []
    for relative in sorted(set(expected) | set(actual_files)):
        path = actual_files.get(relative)
        expected_plane, expected_class = expected.get(
            relative, ("unexpected", "unexpected")
        )
        exists = path is not None
        reparse = _reparse_status(path) if path is not None else False
        if exists and (not path.is_file() or reparse):
            size = None
            digest = None
        elif exists:
            size = int(path.stat().st_size)
            digest = sha256_file(path)
        else:
            size = None
            digest = None
        rows.append(
            {
                "canonical_relative_path": relative,
                "file_size_bytes": size,
                "sha256": digest,
                "file_class": expected_class,
                "plane": expected_plane,
                "exists": exists,
                "expected": relative in expected,
                "unexpected": relative not in expected,
                "partial_or_temp": _partial_status(relative),
                "reparse_point": reparse,
            }
        )
    return {
        "root": destination.as_posix(),
        "files": rows,
        "directories": sorted(
            directory_rows, key=lambda row: row["canonical_relative_path"]
        ),
    }


def _surface_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes(canonical_json_bytes(list(rows)))


def _scan_summary(scan: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(scan["files"])
    protected = [row for row in rows if row["plane"] == "protected_payload"]
    control = [row for row in rows if row["plane"] == "control_plane"]
    existing = [row for row in rows if row["exists"]]
    return {
        "expected_file_count": len(expected_root_files()),
        "existing_file_count": len(existing),
        "protected_expected_count": 790,
        "protected_existing_count": sum(row["exists"] for row in protected),
        "shard_existing_count": sum(
            row["exists"] and row["file_class"] == "shard" for row in rows
        ),
        "index_existing_count": sum(
            row["exists"] and row["file_class"] == "index" for row in rows
        ),
        "frozen_lineage_existing_count": sum(
            row["exists"] and row["file_class"] == "frozen_lineage"
            for row in rows
        ),
        "control_existing_count": sum(row["exists"] for row in control),
        "missing_expected_count": sum(
            row["expected"] and not row["exists"] for row in rows
        ),
        "unexpected_file_count": sum(row["unexpected"] for row in rows),
        "partial_or_temp_count": sum(row["partial_or_temp"] for row in rows),
        "reparse_file_count": sum(row["reparse_point"] for row in rows),
        "unexpected_directory_count": sum(
            row["unexpected"] for row in scan["directories"]
        ),
        "reparse_directory_count": sum(
            row["reparse_point"] for row in scan["directories"]
        ),
        "total_existing_bytes": sum(
            int(row["file_size_bytes"] or 0) for row in existing
        ),
        "protected_surface_sha256": _surface_hash(protected),
        "complete_surface_sha256": _surface_hash(rows),
    }



def build_incident_manifest(
    root: Path,
    *,
    incident_id: str,
    observed_at_utc: str,
    original_source_commit: str = ORIGINAL_SOURCE_COMMIT,
    original_plan_payload_sha256: str = ORIGINAL_PLAN_PAYLOAD_SHA256,
) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", incident_id):
        raise Gate12C2CloseoutRecoveryError("incident_id is invalid")
    parse_utc(observed_at_utc, label="incident observation time")
    if (
        not is_git_commit(original_source_commit)
        or not is_sha256(original_plan_payload_sha256)
        or original_source_commit != ORIGINAL_SOURCE_COMMIT
        or original_plan_payload_sha256 != ORIGINAL_PLAN_PAYLOAD_SHA256
    ):
        raise Gate12C2CloseoutRecoveryError("incident provenance is invalid")
    scan = scan_root_bytes(root)
    summary = _scan_summary(scan)
    exact = bool(
        summary["existing_file_count"] == 791
        and summary["protected_existing_count"] == 790
        and summary["shard_existing_count"] == 768
        and summary["index_existing_count"] == 9
        and summary["frozen_lineage_existing_count"] == 13
        and summary["control_existing_count"] == 1
        and summary["missing_expected_count"] == 0
        and summary["unexpected_file_count"] == 0
        and summary["partial_or_temp_count"] == 0
        and summary["reparse_file_count"] == 0
        and summary["unexpected_directory_count"] == 0
        and summary["reparse_directory_count"] == 0
    )
    payload: dict[str, Any] = {
        "schema_version": INCIDENT_MANIFEST_SCHEMA,
        "incident_id": incident_id,
        "epistemic_status": "byte_only_incident_evidence",
        "state": "INCIDENT_FROZEN" if exact else "RECOVERY_REJECTED",
        "observed_at_utc": observed_at_utc,
        "output_root": scan["root"],
        "original_source_commit": original_source_commit,
        "original_plan_payload_sha256": original_plan_payload_sha256,
        "inspection_contract": {
            "json_parsed": False,
            "gzip_parsed": False,
            "npz_parsed": False,
            "scientific_values_inspected": False,
            "recorded_fields": [
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
            ],
        },
        "summary": summary,
        "files": scan["files"],
        "directories": scan["directories"],
        "payload_presence_observed": "768/768" if exact else "incomplete",
        "payload_integrity_status": "pending",
        "index_integrity_status": "pending",
        "original_execution_closeout_status": "failed",
        "original_resource_evidence_status": "missing",
        "resource_gate_status": "indeterminate",
        "scientific_values_emitted": False,
        "stability_analysis_authorized": False,
        "locked_execution_authorized": False,
        "real_held_out_execution_authorized": False,
        "N2_open": False,
        "N3_open": False,
    }
    payload["incident_manifest_payload_sha256"] = sha256_bytes(
        canonical_json_bytes(payload)
    )
    return payload


def verify_incident_manifest(
    manifest: Mapping[str, Any], *, root: Path, require_exact: bool = True
) -> dict[str, Any]:
    supplied = dict(manifest)
    verify_self_hash(
        supplied,
        hash_field="incident_manifest_payload_sha256",
        label="incident manifest",
    )
    rebuilt = build_incident_manifest(
        root,
        incident_id=str(supplied.get("incident_id", "")),
        observed_at_utc=str(supplied.get("observed_at_utc", "")),
        original_source_commit=str(supplied.get("original_source_commit", "")),
        original_plan_payload_sha256=str(
            supplied.get("original_plan_payload_sha256", "")
        ),
    )
    if supplied != rebuilt:
        raise Gate12C2CloseoutRecoveryError(
            "incident manifest differs from the current byte surface"
        )
    if require_exact and supplied.get("state") != "INCIDENT_FROZEN":
        raise Gate12C2CloseoutRecoveryError("incident byte surface is not exact")
    return rebuilt


def _build_failure_receipt_from_observation(
    *,
    incident_manifest_path: Path,
    stdout_log_path: Path,
    stderr_log_path: Path,
    runner_pid: int,
    observed_at_utc: str,
    runner_process_present_observed: bool,
) -> dict[str, Any]:
    manifest = read_mapping(incident_manifest_path, label="incident manifest")
    verify_incident_manifest(
        manifest, root=Path(manifest["output_root"]), require_exact=True
    )
    log_rows = []
    for label, raw_path in (
        ("stdout", stdout_log_path),
        ("stderr", stderr_log_path),
    ):
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise Gate12C2CloseoutRecoveryError("incident log is missing")
        log_rows.append(
            {
                "stream": label,
                "path": path.as_posix(),
                "file_size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
                "content_inspected": False,
            }
        )
    verified_runner_pid = _strict_int(runner_pid, label="runner PID")
    if type(runner_process_present_observed) is not bool:
        raise Gate12C2CloseoutRecoveryError(
            "runner process observation must be boolean"
        )
    runner_process_present = runner_process_present_observed
    payload: dict[str, Any] = {
        "schema_version": FAILURE_RECEIPT_SCHEMA,
        "incident_id": manifest["incident_id"],
        "state": "INCIDENT_FROZEN",
        "observed_at_utc": parse_utc(
            observed_at_utc, label="failure observation time"
        ).isoformat(),
        "failure_code": INCIDENT_FAILURE_CODE,
        "failure_class": "execution_closeout_lineage_defect",
        "runner_pid": verified_runner_pid,
        "runner_process_present": runner_process_present,
        "incident_manifest_path": Path(incident_manifest_path).resolve().as_posix(),
        "incident_manifest_file_sha256": sha256_file(incident_manifest_path),
        "incident_manifest_payload_sha256": manifest[
            "incident_manifest_payload_sha256"
        ],
        "logs": log_rows,
        "protected_payload_mutated": False,
        "stale_lock_retired": False,
        "normal_resume_permitted": False,
        "original_execution_closeout_status": "failed",
        "original_resource_evidence_status": "missing",
        "resource_gate_status": "indeterminate",
        "scientific_values_emitted": False,
        "stability_analysis_authorized": False,
        "locked_execution_authorized": False,
        "real_held_out_execution_authorized": False,
        "N2_open": False,
        "N3_open": False,
    }
    payload["failure_receipt_payload_sha256"] = sha256_bytes(
        canonical_json_bytes(payload)
    )
    return payload


def build_failure_receipt(
    *,
    incident_manifest_path: Path,
    stdout_log_path: Path,
    stderr_log_path: Path,
    runner_pid: int,
    observed_at_utc: str,
) -> dict[str, Any]:
    verified_runner_pid = _strict_int(runner_pid, label="runner PID")
    runner_state = process_pid_state(verified_runner_pid)
    if runner_state == PROCESS_UNKNOWN:
        raise Gate12C2CloseoutRecoveryError(
            "runner process liveness is indeterminate"
        )
    return _build_failure_receipt_from_observation(
        incident_manifest_path=incident_manifest_path,
        stdout_log_path=stdout_log_path,
        stderr_log_path=stderr_log_path,
        runner_pid=verified_runner_pid,
        observed_at_utc=observed_at_utc,
        runner_process_present_observed=runner_state == PROCESS_ACTIVE,
    )


def build_exposure_ledger(
    *, incident_id: str, reviewer_context_id: str, recorded_at_utc: str
) -> dict[str, Any]:
    parse_utc(recorded_at_utc, label="exposure recording time")
    payload: dict[str, Any] = {
        "schema_version": EXPOSURE_LEDGER_SCHEMA,
        "incident_id": incident_id,
        "reviewer_context_id": reviewer_context_id,
        "recorded_at_utc": recorded_at_utc,
        "exposure_scope": "one index tool output included nonoperational fields",
        "scientific_values_interpreted": False,
        "engineering_review_eligibility": "retained",
        "scientific_selector_blinded_eligibility": "lost",
        "draw_selector_blinded_eligibility": "lost",
        "replacement_fresh_review_context_required": True,
    }
    payload["exposure_ledger_payload_sha256"] = sha256_bytes(
        canonical_json_bytes(payload)
    )
    return payload



def is_git_commit(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None


def git_head() -> str:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0 or not is_git_commit(completed.stdout.strip()):
        raise Gate12C2CloseoutRecoveryError("could not establish current commit")
    return completed.stdout.strip()


def git_blob_sha256(commit: str, relative_path: str) -> str:
    if not is_git_commit(commit):
        raise Gate12C2CloseoutRecoveryError("legacy source commit is invalid")
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=str(root),
        capture_output=True,
        timeout=120,
    )
    if completed.returncode != 0:
        raise Gate12C2CloseoutRecoveryError(
            "required legacy provenance blob is unavailable"
        )
    return sha256_bytes(completed.stdout)


def recovery_implementation_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    names = (
        "gate12c2_closeout_recovery.py",
        "freeze_gate12c2_closeout_incident.py",
        "issue_gate12c2_closeout_recovery_authorization.py",
        "run_gate12c2_closeout_recovery.py",
        "verify_gate12c2_closeout_recovery.py",
        "gate12c2_draw_profile.py",
        "gate12c2_development_shards.py",
        "gate12c2_synthetic_lab.py",
    )
    paths = {name: directory / name for name in names}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise Gate12C2CloseoutRecoveryError(
            "recovery implementation surface is incomplete"
        )
    return {name: sha256_file(path) for name, path in sorted(paths.items())}


def _verify_canonical_mapping_file(path: Path, *, label: str) -> dict[str, Any]:
    payload = read_mapping(path, label=label)
    if Path(path).read_bytes() != canonical_json_bytes(payload):
        raise Gate12C2CloseoutRecoveryError(f"{label} is not canonical JSON")
    return payload


def _verify_original_control_lineage(
    plan: Mapping[str, Any], *, output_root: Path
) -> dict[str, Any]:
    control_root = Path(output_root) / "control"
    preflight = _verify_canonical_mapping_file(
        control_root / "preflight.json", label="original preflight"
    )
    authorization = _verify_canonical_mapping_file(
        control_root / "authorization.json", label="original authorization"
    )
    consumption = _verify_canonical_mapping_file(
        control_root / "authorization-consumed.json",
        label="original authorization consumption",
    )
    verify_self_hash(
        preflight,
        hash_field="preflight_receipt_payload_sha256",
        label="original preflight",
    )
    verify_self_hash(
        authorization,
        hash_field="authorization_receipt_payload_sha256",
        label="original authorization",
    )
    verify_self_hash(
        consumption,
        hash_field="consumption_receipt_payload_sha256",
        label="original authorization consumption",
    )
    plan_hash = plan["draw_profile_plan_payload_sha256"]
    if (
        preflight.get("draw_profile_plan_payload_sha256") != plan_hash
        or authorization.get("draw_profile_plan_payload_sha256") != plan_hash
        or consumption.get("draw_profile_plan_payload_sha256") != plan_hash
        or authorization.get("preflight_receipt_payload_sha256")
        != preflight.get("preflight_receipt_payload_sha256")
        or consumption.get("authorization_receipt_payload_sha256")
        != authorization.get("authorization_receipt_payload_sha256")
        or Path(str(authorization.get("output_root", ""))).resolve()
        != Path(output_root).resolve()
        or Path(str(consumption.get("output_root", ""))).resolve()
        != Path(output_root).resolve()
        or authorization.get("single_use") is not True
        or consumption.get("single_use") is not True
        or consumption.get("authorization_status")
        != "consumed_for_this_execution_lineage"
    ):
        raise Gate12C2CloseoutRecoveryError(
            "original control receipts are not one exact lineage"
        )
    return {
        "preflight_receipt_payload_sha256": preflight[
            "preflight_receipt_payload_sha256"
        ],
        "authorization_receipt_payload_sha256": authorization[
            "authorization_receipt_payload_sha256"
        ],
        "consumption_receipt_payload_sha256": consumption[
            "consumption_receipt_payload_sha256"
        ],
    }


def _verify_legacy_lineage_static(
    *,
    output_root: Path,
    archived_plan_path: Path,
    expected_source_commit: str = ORIGINAL_SOURCE_COMMIT,
    expected_plan_payload_sha256: str = ORIGINAL_PLAN_PAYLOAD_SHA256,
) -> dict[str, Any]:
    """Verify immutable legacy evidence without consulting current PID state."""

    archived = _verify_canonical_mapping_file(
        archived_plan_path, label="archived exact plan"
    )
    root_plan_path = Path(output_root).resolve() / "plan.json"
    root_plan = _verify_canonical_mapping_file(
        root_plan_path, label="profile-root exact plan"
    )
    if Path(archived_plan_path).read_bytes() != root_plan_path.read_bytes():
        raise Gate12C2CloseoutRecoveryError(
            "archived and profile-root plan bytes differ"
        )
    verify_self_hash(
        archived,
        hash_field="draw_profile_plan_payload_sha256",
        label="archived exact plan",
    )
    if (
        archived != root_plan
        or archived.get("source_commit") != expected_source_commit
        or archived.get("draw_profile_plan_payload_sha256")
        != expected_plan_payload_sha256
        or not is_git_commit(expected_source_commit)
        or not is_sha256(expected_plan_payload_sha256)
    ):
        raise Gate12C2CloseoutRecoveryError("legacy plan provenance mismatch")
    implementation = archived.get("implementation_sha256")
    if not isinstance(implementation, Mapping) or not implementation:
        raise Gate12C2CloseoutRecoveryError(
            "legacy implementation identity is missing"
        )
    blob_rows = []
    for name, expected_hash in sorted(implementation.items()):
        if not isinstance(name, str) or not is_sha256(expected_hash):
            raise Gate12C2CloseoutRecoveryError(
                "legacy implementation identity is invalid"
            )
        actual = git_blob_sha256(expected_source_commit, f"tools/{name}")
        if actual != expected_hash:
            raise Gate12C2CloseoutRecoveryError(
                "legacy implementation blob identity mismatch"
            )
        blob_rows.append({"name": name, "sha256": actual})
    semantic_dependencies = {
        "gate12c2_development_shards.py": Path(shards.__file__).resolve(),
        "gate12c2_synthetic_lab.py": Path(shards.lab.__file__).resolve(),
    }
    current_semantic_hashes = {
        name: sha256_file(path)
        for name, path in sorted(semantic_dependencies.items())
    }
    for name, current_hash in current_semantic_hashes.items():
        if implementation.get(name) != current_hash:
            raise Gate12C2CloseoutRecoveryError(
                "current semantic verifier dependency is not byte-identical "
                "to legacy verifier"
            )
    control = _verify_original_control_lineage(
        archived, output_root=Path(output_root).resolve()
    )
    lock = _verify_canonical_mapping_file(
        Path(output_root).resolve() / LOCK_NAME,
        label="stale coordinator lock",
    )
    verify_self_hash(lock, hash_field="lock_payload_sha256", label="stale lock")
    if (
        lock.get("plan_payload_sha256") != expected_plan_payload_sha256
        or lock.get("implementation_sha256") != implementation
        or type(lock.get("pid")) is not int
    ):
        raise Gate12C2CloseoutRecoveryError("stale lock provenance is invalid")
    return {
        "original_source_commit": expected_source_commit,
        "original_plan_payload_sha256": expected_plan_payload_sha256,
        "implementation_blobs": blob_rows,
        "current_semantic_verifier_sha256": current_semantic_hashes,
        "control_lineage": control,
        "stale_lock_payload_sha256": lock["lock_payload_sha256"],
        "stale_lock_pid": int(lock["pid"]),
    }


def _compose_legacy_lineage_evidence(
    static: Mapping[str, Any], observation: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(static, Mapping) or not isinstance(observation, Mapping):
        raise Gate12C2CloseoutRecoveryError(
            "legacy stale-lock liveness observation schema mismatch"
        )
    frozen = dict(observation)
    if set(frozen) != {
        "pid",
        "state",
        "observed_at_utc",
        "stale_lock_payload_sha256",
    }:
        raise Gate12C2CloseoutRecoveryError(
            "legacy stale-lock liveness observation schema mismatch"
        )
    parse_utc(
        frozen.get("observed_at_utc"),
        label="legacy stale-lock liveness observation time",
    )
    if (
        _strict_int(frozen.get("pid"), label="legacy stale-lock PID")
        != _strict_int(static.get("stale_lock_pid"), label="static stale-lock PID")
        or frozen.get("state") != PROCESS_DEAD
        or frozen.get("stale_lock_payload_sha256")
        != static.get("stale_lock_payload_sha256")
    ):
        raise Gate12C2CloseoutRecoveryError(
            "legacy stale-lock liveness observation is invalid"
        )
    return {
        **dict(static),
        "stale_lock_liveness_observation": frozen,
        "stale_lock_owner_not_running": True,
    }


def _observe_legacy_lock_dead(
    static: Mapping[str, Any], *, observed_at_utc: str
) -> dict[str, Any]:
    parse_utc(
        observed_at_utc, label="legacy stale-lock liveness observation time"
    )
    pid = _strict_int(static.get("stale_lock_pid"), label="static stale-lock PID")
    if process_pid_state(pid) != PROCESS_DEAD:
        raise Gate12C2CloseoutRecoveryError(
            "stale lock owner liveness is not definitively dead"
        )
    return {
        "pid": pid,
        "state": PROCESS_DEAD,
        "observed_at_utc": observed_at_utc,
        "stale_lock_payload_sha256": static["stale_lock_payload_sha256"],
    }


def verify_legacy_lineage(
    *,
    output_root: Path,
    archived_plan_path: Path,
    expected_source_commit: str = ORIGINAL_SOURCE_COMMIT,
    expected_plan_payload_sha256: str = ORIGINAL_PLAN_PAYLOAD_SHA256,
    observed_at_utc: str | None = None,
) -> dict[str, Any]:
    """Verify immutable legacy evidence and establish current DEAD liveness."""

    static = _verify_legacy_lineage_static(
        output_root=output_root,
        archived_plan_path=archived_plan_path,
        expected_source_commit=expected_source_commit,
        expected_plan_payload_sha256=expected_plan_payload_sha256,
    )
    observation = _observe_legacy_lock_dead(
        static,
        observed_at_utc=observed_at_utc or utc_now().isoformat(),
    )
    return _compose_legacy_lineage_evidence(static, observation)


def build_freeze_discrepancy_receipt(
    *,
    rejected_manifest_path: Path,
    rejected_failure_path: Path,
    rejected_exposure_path: Path,
    replacement_manifest_path: Path,
    replacement_failure_path: Path,
    replacement_exposure_path: Path,
    recorded_at_utc: str,
) -> dict[str, Any]:
    parse_utc(recorded_at_utc, label="freeze discrepancy time")
    rejected_manifest = read_mapping(
        rejected_manifest_path, label="rejected incident manifest"
    )
    verify_self_hash(
        rejected_manifest,
        hash_field="incident_manifest_payload_sha256",
        label="rejected incident manifest",
    )
    if rejected_manifest.get("state") != "RECOVERY_REJECTED":
        raise Gate12C2CloseoutRecoveryError(
            "freeze discrepancy source was not rejected"
        )
    rejected_failure = read_mapping(
        rejected_failure_path, label="rejected-attempt failure receipt"
    )
    rejected_exposure = read_mapping(
        rejected_exposure_path, label="rejected-attempt exposure ledger"
    )
    verify_self_hash(
        rejected_failure,
        hash_field="failure_receipt_payload_sha256",
        label="rejected-attempt failure receipt",
    )
    verify_self_hash(
        rejected_exposure,
        hash_field="exposure_ledger_payload_sha256",
        label="rejected-attempt exposure ledger",
    )
    replacement_manifest = read_mapping(
        replacement_manifest_path, label="replacement incident manifest"
    )
    verify_incident_manifest(
        replacement_manifest,
        root=Path(replacement_manifest["output_root"]),
        require_exact=True,
    )
    replacement_failure = verify_failure_receipt(
        read_mapping(replacement_failure_path, label="replacement failure receipt")
    )
    replacement_exposure = verify_exposure_ledger(
        read_mapping(replacement_exposure_path, label="replacement exposure ledger")
    )
    payload: dict[str, Any] = {
        "schema_version": FREEZE_DISCREPANCY_SCHEMA,
        "recorded_at_utc": recorded_at_utc,
        "discrepancy_code": "INCIDENT_MANIFEST_SHARD_SUBDIRECTORY_OMITTED",
        "rejected_attempt_state": "RECOVERY_REJECTED",
        "rejected_attempt_paths": {
            "manifest": Path(rejected_manifest_path).resolve().as_posix(),
            "failure": Path(rejected_failure_path).resolve().as_posix(),
            "exposure": Path(rejected_exposure_path).resolve().as_posix(),
        },
        "rejected_attempt_file_sha256": {
            "manifest": sha256_file(rejected_manifest_path),
            "failure": sha256_file(rejected_failure_path),
            "exposure": sha256_file(rejected_exposure_path),
        },
        "replacement_paths": {
            "manifest": Path(replacement_manifest_path).resolve().as_posix(),
            "failure": Path(replacement_failure_path).resolve().as_posix(),
            "exposure": Path(replacement_exposure_path).resolve().as_posix(),
        },
        "replacement_file_sha256": {
            "manifest": sha256_file(replacement_manifest_path),
            "failure": sha256_file(replacement_failure_path),
            "exposure": sha256_file(replacement_exposure_path),
        },
        "replacement_payload_sha256": {
            "manifest": replacement_manifest[
                "incident_manifest_payload_sha256"
            ],
            "failure": replacement_failure[
                "failure_receipt_payload_sha256"
            ],
            "exposure": replacement_exposure[
                "exposure_ledger_payload_sha256"
            ],
        },
        "rejected_attempt_superseded": True,
        "profile_root_mutated": False,
        "scientific_values_inspected": False,
        "resource_gate_status": "indeterminate",
        "stability_analysis_authorized": False,
    }
    payload["freeze_discrepancy_payload_sha256"] = sha256_bytes(
        canonical_json_bytes(payload)
    )
    return payload

def verify_failure_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    supplied = dict(receipt)
    verify_self_hash(
        supplied,
        hash_field="failure_receipt_payload_sha256",
        label="failure receipt",
    )
    logs = supplied.get("logs")
    if not isinstance(logs, list) or len(logs) != 2:
        raise Gate12C2CloseoutRecoveryError("failure receipt log surface differs")
    by_stream = {
        row.get("stream"): row for row in logs if isinstance(row, Mapping)
    }
    if set(by_stream) != {"stdout", "stderr"}:
        raise Gate12C2CloseoutRecoveryError("failure receipt log surface differs")
    expected = _build_failure_receipt_from_observation(
        incident_manifest_path=Path(supplied["incident_manifest_path"]),
        stdout_log_path=Path(by_stream["stdout"]["path"]),
        stderr_log_path=Path(by_stream["stderr"]["path"]),
        runner_pid=_strict_int(supplied["runner_pid"], label="runner PID"),
        observed_at_utc=str(supplied["observed_at_utc"]),
        runner_process_present_observed=supplied.get(
            "runner_process_present"
        ),
    )
    if supplied != expected:
        raise Gate12C2CloseoutRecoveryError(
            "failure receipt differs from its exact builder"
        )
    return expected

def verify_exposure_ledger(ledger: Mapping[str, Any]) -> dict[str, Any]:
    supplied = dict(ledger)
    verify_self_hash(
        supplied,
        hash_field="exposure_ledger_payload_sha256",
        label="exposure ledger",
    )
    expected = build_exposure_ledger(
        incident_id=str(supplied["incident_id"]),
        reviewer_context_id=str(supplied["reviewer_context_id"]),
        recorded_at_utc=str(supplied["recorded_at_utc"]),
    )
    if supplied != expected:
        raise Gate12C2CloseoutRecoveryError(
            "exposure ledger differs from its exact builder"
        )
    return expected

def build_recovery_amendment(
    *,
    incident_manifest_path: Path,
    failure_receipt_path: Path,
    exposure_ledger_path: Path,
    amendment_id: str,
    recorded_at_utc: str,
) -> dict[str, Any]:
    manifest = read_mapping(incident_manifest_path, label="incident manifest")
    verify_incident_manifest(
        manifest, root=Path(manifest["output_root"]), require_exact=True
    )
    failure = verify_failure_receipt(
        read_mapping(failure_receipt_path, label="failure receipt")
    )
    exposure = verify_exposure_ledger(
        read_mapping(exposure_ledger_path, label="exposure ledger")
    )
    if not re.fullmatch(r"[A-Za-z0-9._-]+", amendment_id):
        raise Gate12C2CloseoutRecoveryError("amendment_id is invalid")
    parse_utc(recorded_at_utc, label="amendment recording time")
    if not (
        manifest.get("incident_id") == failure.get("incident_id")
        == exposure.get("incident_id")
    ):
        raise Gate12C2CloseoutRecoveryError("incident evidence is not linked")
    payload: dict[str, Any] = {
        "schema_version": AMENDMENT_SCHEMA,
        "amendment_id": amendment_id,
        "recorded_at_utc": recorded_at_utc,
        "epistemic_status": "engineering_closeout_recovery_only",
        "program_amendment": False,
        "study_level_deviation": True,
        "current_source_commit": git_head(),
        "recovery_implementation_sha256": recovery_implementation_hashes(),
        "incident_manifest_path": Path(incident_manifest_path).resolve().as_posix(),
        "incident_manifest_file_sha256": sha256_file(incident_manifest_path),
        "incident_manifest_payload_sha256": manifest[
            "incident_manifest_payload_sha256"
        ],
        "failure_receipt_path": Path(failure_receipt_path).resolve().as_posix(),
        "failure_receipt_file_sha256": sha256_file(failure_receipt_path),
        "failure_receipt_payload_sha256": failure[
            "failure_receipt_payload_sha256"
        ],
        "exposure_ledger_path": Path(exposure_ledger_path).resolve().as_posix(),
        "exposure_ledger_file_sha256": sha256_file(exposure_ledger_path),
        "exposure_ledger_payload_sha256": exposure[
            "exposure_ledger_payload_sha256"
        ],
        "state_model": [
            "INCIDENT_FROZEN",
            "RECOVERY_CODE_REVIEWED",
            "RECOVERY_AUTHORIZED",
            "PAYLOAD_VERIFIED",
            "TERMINAL_OUTCOME_CLAIMED",
            "PAYLOAD_COMPLETION_SEALED",
            "LOCK_RETIRED",
            "RESOURCE_GATE_INDETERMINATE",
        ],
        "failure_states": [
            "RECOVERY_REJECTED",
            "RECOVERY_INTERRUPTED",
            "PAYLOAD_MISMATCH",
        ],
        "protected_payload_contract": {
            "shard_count": 768,
            "index_count": 9,
            "frozen_lineage_count": 13,
            "payload_added": 0,
            "payload_modified": 0,
            "payload_deleted": 0,
            "index_bytes_changed": 0,
        },
        "control_plane_contract": {
            "payload_seal_written_outside_output_root": True,
            "terminal_claim_written_outside_output_root": True,
            "terminal_claim_is_exclusive": True,
            "unknown_liveness_may_finalize": False,
            "lock_retirement_requires_separate_authorization": True,
            "lock_retirement_authorized_by_this_amendment": False,
            "stale_lock_removed_during_payload_seal": False,
        },
        "original_execution_closeout_status": "failed",
        "original_resource_evidence_status": "missing",
        "resource_gate_status": "indeterminate",
        "replacement_resource_qualification": "not_performed",
        "replacement_qualification_cannot_rewrite_original_gate": True,
        "normal_resume_permitted": False,
        "recovery_may_recompute_shards": False,
        "recovery_may_rewrite_indices": False,
        "scientific_values_may_be_emitted": False,
        "stability_analysis_authorized": False,
        "locked_execution_authorized": False,
        "real_held_out_execution_authorized": False,
        "N2_open": False,
        "N3_open": False,
        "next_gate": "fresh_adversarial_review_before_recovery_authorization",
    }
    payload["amendment_payload_sha256"] = sha256_bytes(
        canonical_json_bytes(payload)
    )
    return payload


def verify_recovery_amendment(amendment: Mapping[str, Any]) -> dict[str, Any]:
    supplied = dict(amendment)
    verify_self_hash(
        supplied, hash_field="amendment_payload_sha256", label="recovery amendment"
    )
    expected = build_recovery_amendment(
        incident_manifest_path=Path(supplied["incident_manifest_path"]),
        failure_receipt_path=Path(supplied["failure_receipt_path"]),
        exposure_ledger_path=Path(supplied["exposure_ledger_path"]),
        amendment_id=str(supplied["amendment_id"]),
        recorded_at_utc=str(supplied["recorded_at_utc"]),
    )
    if supplied != expected:
        raise Gate12C2CloseoutRecoveryError(
            "recovery amendment differs from its exact builder"
        )
    return expected


def verify_recovery_review_receipt(
    receipt: Mapping[str, Any], *, amendment: Mapping[str, Any]
) -> dict[str, Any]:
    supplied = dict(receipt)
    verify_self_hash(
        supplied,
        hash_field="review_receipt_payload_sha256",
        label="fresh recovery review receipt",
    )
    required = {
        "schema_version",
        "review_id",
        "reviewer_context_id",
        "reviewed_source_commit",
        "reviewed_implementation_sha256",
        "amendment_payload_sha256",
        "incident_manifest_payload_sha256",
        "review_status",
        "P0_count",
        "P1_count",
        "scientific_values_inspected",
        "recovery_authorization_may_be_issued",
        "lock_retirement_authorized",
        "reviewed_at_utc",
        "review_receipt_payload_sha256",
    }
    if set(supplied) != required:
        raise Gate12C2CloseoutRecoveryError("fresh review schema mismatch")
    parse_utc(supplied["reviewed_at_utc"], label="fresh review time")
    p0_count = _strict_int(supplied["P0_count"], label="P0 count")
    p1_count = _strict_int(supplied["P1_count"], label="P1 count")
    reviewer_context_id = supplied["reviewer_context_id"]
    review_id = supplied["review_id"]
    if (
        not isinstance(reviewer_context_id, str)
        or not re.fullmatch(r"[A-Za-z0-9._-]+", reviewer_context_id)
        or not isinstance(review_id, str)
        or not re.fullmatch(r"[A-Za-z0-9._-]+", review_id)
    ):
        raise Gate12C2CloseoutRecoveryError("fresh review identity is invalid")
    exposure = verify_exposure_ledger(
        read_mapping(
            Path(amendment["exposure_ledger_path"]),
            label="reviewer exposure ledger",
        )
    )
    if (
        sha256_file(Path(amendment["exposure_ledger_path"]))
        != amendment["exposure_ledger_file_sha256"]
        or exposure["exposure_ledger_payload_sha256"]
        != amendment["exposure_ledger_payload_sha256"]
        or reviewer_context_id == exposure["reviewer_context_id"]
        or exposure["replacement_fresh_review_context_required"] is not True
    ):
        raise Gate12C2CloseoutRecoveryError(
            "fresh review context is not independent of recorded exposure"
        )
    if (
        supplied["schema_version"]
        != "gate12c2_closeout_recovery_fresh_review_v0.1"
        or supplied["reviewed_source_commit"] != git_head()
        or supplied["reviewed_implementation_sha256"]
        != recovery_implementation_hashes()
        or supplied["amendment_payload_sha256"]
        != amendment["amendment_payload_sha256"]
        or supplied["incident_manifest_payload_sha256"]
        != amendment["incident_manifest_payload_sha256"]
        or supplied["review_status"] != "pass"
        or p0_count != 0
        or p1_count != 0
        or supplied["scientific_values_inspected"] is not False
        or supplied["recovery_authorization_may_be_issued"] is not True
        or supplied["lock_retirement_authorized"] is not False
    ):
        raise Gate12C2CloseoutRecoveryError("fresh review does not authorize recovery")
    return supplied



def _require_outside_root(path: Path, *, root: Path) -> Path:
    resolved = Path(path).resolve()
    destination = Path(root).resolve()
    try:
        resolved.relative_to(destination)
    except ValueError:
        return resolved
    raise Gate12C2CloseoutRecoveryError(
        "recovery evidence must be written outside the profile root"
    )


def _validate_external_output_path(
    path: Path,
    *,
    root: Path,
    require_fresh: bool,
) -> Path:
    resolved = _require_outside_root(path, root=root)
    current = resolved.parent
    while True:
        if current.exists():
            if not current.is_dir() or _reparse_status(current):
                raise Gate12C2CloseoutRecoveryError(
                    "recovery evidence output ancestry is unsafe"
                )
        if current.parent == current:
            break
        current = current.parent
    if _reparse_status(resolved) if resolved.exists() else False:
        raise Gate12C2CloseoutRecoveryError(
            "recovery evidence output is a reparse point"
        )
    if require_fresh and resolved.exists():
        raise Gate12C2CloseoutRecoveryError(
            "recovery evidence output already exists"
        )
    return resolved

def build_recovery_authorization(
    *,
    amendment_path: Path,
    review_receipt_path: Path,
    incident_manifest_path: Path,
    archived_plan_path: Path,
    output_root: Path,
    authorization_id: str,
    expires_at_utc: str,
    authorization_output: Path,
    attempt_output: Path,
    consumption_output: Path,
    terminal_output: Path,
    seal_output: Path,
    failure_output: Path,
) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", authorization_id):
        raise Gate12C2CloseoutRecoveryError("authorization_id is invalid")
    root = Path(output_root).resolve()
    outputs = {
        "authorization_output": _validate_external_output_path(
            authorization_output, root=root, require_fresh=True
        ),
        "attempt_output": _validate_external_output_path(
            attempt_output, root=root, require_fresh=True
        ),
        "consumption_output": _validate_external_output_path(
            consumption_output, root=root, require_fresh=True
        ),
        "terminal_output": _validate_external_output_path(
            terminal_output, root=root, require_fresh=True
        ),
        "seal_output": _validate_external_output_path(
            seal_output, root=root, require_fresh=True
        ),
        "failure_output": _validate_external_output_path(
            failure_output, root=root, require_fresh=True
        ),
    }
    if len(set(outputs.values())) != 6:
        raise Gate12C2CloseoutRecoveryError(
            "recovery evidence outputs are not fresh and distinct"
        )
    amendment = verify_recovery_amendment(
        read_mapping(amendment_path, label="recovery amendment")
    )
    review = verify_recovery_review_receipt(
        read_mapping(review_receipt_path, label="fresh recovery review receipt"),
        amendment=amendment,
    )
    manifest = verify_incident_manifest(
        read_mapping(incident_manifest_path, label="incident manifest"),
        root=root,
        require_exact=True,
    )
    if (
        manifest["incident_manifest_payload_sha256"]
        != amendment["incident_manifest_payload_sha256"]
    ):
        raise Gate12C2CloseoutRecoveryError("amendment and incident differ")
    issued = utc_now()
    lineage = verify_legacy_lineage(
        output_root=root,
        archived_plan_path=archived_plan_path,
        expected_source_commit=str(manifest["original_source_commit"]),
        expected_plan_payload_sha256=str(manifest["original_plan_payload_sha256"]),
        observed_at_utc=issued.isoformat(),
    )
    expiration = parse_utc(expires_at_utc, label="recovery authorization expiration")
    if expiration <= issued or expiration - issued > timedelta(
        seconds=AUTHORIZATION_MAX_AGE_SECONDS
    ):
        raise Gate12C2CloseoutRecoveryError(
            "recovery authorization expiration is outside the frozen window"
        )
    payload: dict[str, Any] = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "authorization_id": authorization_id,
        "authorization_scope": "payload_verification_and_external_seal_only",
        "authorization_status": "unconsumed",
        "single_use": True,
        "issued_at_utc": issued.isoformat(),
        "expires_at_utc": expiration.isoformat(),
        "maximum_age_seconds": AUTHORIZATION_MAX_AGE_SECONDS,
        "hostname": socket.gethostname(),
        "current_source_commit": git_head(),
        "recovery_implementation_sha256": recovery_implementation_hashes(),
        "original_source_commit": manifest["original_source_commit"],
        "original_plan_payload_sha256": manifest[
            "original_plan_payload_sha256"
        ],
        "legacy_lineage_evidence_sha256": sha256_bytes(
            canonical_json_bytes(lineage)
        ),
        "stale_lock_liveness_observation": lineage[
            "stale_lock_liveness_observation"
        ],
        "output_root": root.as_posix(),
        "archived_plan_path": Path(archived_plan_path).resolve().as_posix(),
        "archived_plan_file_sha256": sha256_file(archived_plan_path),
        "incident_manifest_path": Path(incident_manifest_path).resolve().as_posix(),
        "incident_manifest_file_sha256": sha256_file(incident_manifest_path),
        "incident_manifest_payload_sha256": manifest[
            "incident_manifest_payload_sha256"
        ],
        "amendment_path": Path(amendment_path).resolve().as_posix(),
        "amendment_file_sha256": sha256_file(amendment_path),
        "amendment_payload_sha256": amendment["amendment_payload_sha256"],
        "review_receipt_path": Path(review_receipt_path).resolve().as_posix(),
        "review_receipt_file_sha256": sha256_file(review_receipt_path),
        "review_receipt_payload_sha256": review[
            "review_receipt_payload_sha256"
        ],
        "authorization_output": outputs["authorization_output"].as_posix(),
        "attempt_output": outputs["attempt_output"].as_posix(),
        "consumption_output": outputs["consumption_output"].as_posix(),
        "terminal_output": outputs["terminal_output"].as_posix(),
        "seal_output": outputs["seal_output"].as_posix(),
        "failure_output": outputs["failure_output"].as_posix(),
        "protected_payload_mutation_authorized": False,
        "index_rewrite_authorized": False,
        "stale_lock_retirement_authorized": False,
        "original_resource_evidence_status": "missing",
        "resource_gate_status": "indeterminate",
        "scientific_values_may_be_emitted": False,
        "stability_analysis_authorized": False,
        "locked_execution_authorized": False,
        "real_held_out_execution_authorized": False,
        "N2_open": False,
        "N3_open": False,
    }
    payload["authorization_payload_sha256"] = sha256_bytes(
        canonical_json_bytes(payload)
    )
    return payload


def verify_recovery_authorization(
    authorization: Mapping[str, Any], *, require_current_freshness: bool
) -> dict[str, Any]:
    supplied = dict(authorization)
    verify_self_hash(
        supplied,
        hash_field="authorization_payload_sha256",
        label="closeout recovery authorization",
    )
    required = {
        "schema_version", "authorization_id", "authorization_scope",
        "authorization_status", "single_use", "issued_at_utc",
        "expires_at_utc", "maximum_age_seconds", "hostname",
        "current_source_commit", "recovery_implementation_sha256",
        "original_source_commit", "original_plan_payload_sha256",
        "legacy_lineage_evidence_sha256", "stale_lock_liveness_observation",
        "output_root",
        "archived_plan_path", "archived_plan_file_sha256",
        "incident_manifest_path", "incident_manifest_file_sha256",
        "incident_manifest_payload_sha256", "amendment_path",
        "amendment_file_sha256", "amendment_payload_sha256",
        "review_receipt_path", "review_receipt_file_sha256",
        "review_receipt_payload_sha256", "authorization_output",
        "attempt_output", "consumption_output", "terminal_output",
        "seal_output", "failure_output",
        "protected_payload_mutation_authorized",
        "index_rewrite_authorized", "stale_lock_retirement_authorized",
        "original_resource_evidence_status", "resource_gate_status",
        "scientific_values_may_be_emitted", "stability_analysis_authorized",
        "locked_execution_authorized", "real_held_out_execution_authorized",
        "N2_open", "N3_open", "authorization_payload_sha256",
    }
    if set(supplied) != required or not re.fullmatch(
        r"[A-Za-z0-9._-]+", str(supplied.get("authorization_id", ""))
    ):
        raise Gate12C2CloseoutRecoveryError(
            "closeout recovery authorization schema mismatch"
        )
    issued = parse_utc(supplied.get("issued_at_utc"), label="authorization issue time")
    expiration = parse_utc(
        supplied.get("expires_at_utc"), label="authorization expiration"
    )
    if (
        supplied.get("schema_version") != AUTHORIZATION_SCHEMA
        or supplied.get("authorization_scope")
        != "payload_verification_and_external_seal_only"
        or supplied.get("authorization_status") != "unconsumed"
        or supplied.get("single_use") is not True
        or supplied.get("maximum_age_seconds") != AUTHORIZATION_MAX_AGE_SECONDS
        or expiration <= issued
        or expiration - issued > timedelta(seconds=AUTHORIZATION_MAX_AGE_SECONDS)

        or supplied.get("current_source_commit") != git_head()
        or supplied.get("recovery_implementation_sha256")
        != recovery_implementation_hashes()
        or supplied.get("protected_payload_mutation_authorized") is not False
        or supplied.get("index_rewrite_authorized") is not False
        or supplied.get("stale_lock_retirement_authorized") is not False
        or supplied.get("original_resource_evidence_status") != "missing"
        or supplied.get("resource_gate_status") != "indeterminate"
        or supplied.get("scientific_values_may_be_emitted") is not False
        or supplied.get("stability_analysis_authorized") is not False
        or supplied.get("locked_execution_authorized") is not False
        or supplied.get("real_held_out_execution_authorized") is not False
        or supplied.get("N2_open") is not False
        or supplied.get("N3_open") is not False
    ):
        raise Gate12C2CloseoutRecoveryError(
            "closeout recovery authorization changed a frozen field"
        )
    hostname = supplied.get("hostname")
    if not isinstance(hostname, str) or not hostname:
        raise Gate12C2CloseoutRecoveryError(
            "closeout recovery authorization hostname is invalid"
        )
    if require_current_freshness and (
        not (issued <= utc_now() < expiration)
        or hostname != socket.gethostname()
    ):
        raise Gate12C2CloseoutRecoveryError(
            "closeout recovery authorization is not currently valid"
        )
    amendment = verify_recovery_amendment(
        read_mapping(Path(supplied["amendment_path"]), label="recovery amendment")
    )
    review = verify_recovery_review_receipt(
        read_mapping(
            Path(supplied["review_receipt_path"]),
            label="fresh recovery review receipt",
        ),
        amendment=amendment,
    )
    root = Path(supplied["output_root"]).resolve()
    manifest = verify_incident_manifest(
        read_mapping(
            Path(supplied["incident_manifest_path"]), label="incident manifest"
        ),
        root=root,
        require_exact=True,
    )
    static_lineage = _verify_legacy_lineage_static(
        output_root=root,
        archived_plan_path=Path(supplied["archived_plan_path"]),
        expected_source_commit=str(supplied["original_source_commit"]),
        expected_plan_payload_sha256=str(
            supplied["original_plan_payload_sha256"]
        ),
    )
    observation = supplied.get("stale_lock_liveness_observation")
    if not isinstance(observation, Mapping):
        raise Gate12C2CloseoutRecoveryError(
            "legacy stale-lock liveness observation schema mismatch"
        )
    lineage = _compose_legacy_lineage_evidence(static_lineage, observation)
    if parse_utc(
        observation.get("observed_at_utc"),
        label="legacy stale-lock liveness observation time",
    ) != issued:
        raise Gate12C2CloseoutRecoveryError(
            "legacy stale-lock liveness observation is not bound to issuance"
        )
    if (
        require_current_freshness
        and process_pid_state(
            _strict_int(
                static_lineage.get("stale_lock_pid"),
                label="static stale-lock PID",
            )
        )
        != PROCESS_DEAD
    ):
        raise Gate12C2CloseoutRecoveryError(
            "stale lock owner liveness is not definitively dead"
        )
    checks = {
        "archived_plan_file_sha256": sha256_file(
            Path(supplied["archived_plan_path"])
        ),
        "incident_manifest_file_sha256": sha256_file(
            Path(supplied["incident_manifest_path"])
        ),
        "incident_manifest_payload_sha256": manifest[
            "incident_manifest_payload_sha256"
        ],
        "amendment_file_sha256": sha256_file(Path(supplied["amendment_path"])),
        "amendment_payload_sha256": amendment["amendment_payload_sha256"],
        "review_receipt_file_sha256": sha256_file(
            Path(supplied["review_receipt_path"])
        ),
        "review_receipt_payload_sha256": review[
            "review_receipt_payload_sha256"
        ],
        "legacy_lineage_evidence_sha256": sha256_bytes(
            canonical_json_bytes(lineage)
        ),
    }
    if any(supplied.get(key) != value for key, value in checks.items()):
        raise Gate12C2CloseoutRecoveryError(
            "closeout recovery authorization evidence changed"
        )
    output_paths = [
        _validate_external_output_path(
            Path(supplied[key]), root=root, require_fresh=False
        )
        for key in (
            "authorization_output",
            "attempt_output",
            "consumption_output",
            "terminal_output",
            "seal_output",
            "failure_output",
        )
    ]
    if len(set(output_paths)) != 6:
        raise Gate12C2CloseoutRecoveryError(
            "recovery evidence outputs are not distinct"
        )
    authorization_output = output_paths[0]
    if (
        authorization_output.exists()
        and authorization_output.read_bytes() != canonical_json_bytes(supplied)
    ):
        raise Gate12C2CloseoutRecoveryError(
            "authorization output differs from verified authorization"
        )
    return supplied



def _require_current_recovery_execution_context(
    authorization: Mapping[str, Any]
) -> None:
    """Require current host identity and definitive DEAD stale-lock liveness."""

    if authorization.get("hostname") != socket.gethostname():
        raise Gate12C2CloseoutRecoveryError(
            "closeout recovery execution host differs from authorization"
        )
    observation = authorization.get("stale_lock_liveness_observation")
    if not isinstance(observation, Mapping):
        raise Gate12C2CloseoutRecoveryError(
            "legacy stale-lock liveness observation schema mismatch"
        )
    pid = _strict_int(observation.get("pid"), label="legacy stale-lock PID")
    if process_pid_state(pid) != PROCESS_DEAD:
        raise Gate12C2CloseoutRecoveryError(
            "stale lock owner liveness is not definitively dead"
        )


def build_attempt_receipt(
    authorization: Mapping[str, Any],
    *,
    claimed_at_utc: str,
    attempt_id: str | None = None,
    process_identity_value: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    verified = verify_recovery_authorization(
        authorization, require_current_freshness=False
    )
    claimed = parse_utc(claimed_at_utc, label="recovery attempt claim time")
    issued = parse_utc(verified["issued_at_utc"], label="authorization issue time")
    expiration = parse_utc(
        verified["expires_at_utc"], label="authorization expiration"
    )
    if not (issued <= claimed < expiration):
        raise Gate12C2CloseoutRecoveryError(
            "recovery attempt time is outside the authorization interval"
        )
    identity = (
        dict(process_identity_value)
        if process_identity_value is not None
        else process_identity(os.getpid())
    )
    if identity is None:
        raise Gate12C2CloseoutRecoveryError(
            "recovery attempt process identity is unavailable"
        )
    if (
        set(identity) != {"pid", "identity_kind", "start_marker"}
        or type(identity.get("pid")) is not int
        or not isinstance(identity.get("identity_kind"), str)
        or not isinstance(identity.get("start_marker"), str)
        or not identity["identity_kind"]
        or not identity["start_marker"]
    ):
        raise Gate12C2CloseoutRecoveryError(
            "recovery attempt process identity is invalid"
        )
    claim_id = attempt_id or uuid.uuid4().hex
    if not re.fullmatch(r"[A-Za-z0-9._-]+", claim_id):
        raise Gate12C2CloseoutRecoveryError("recovery attempt ID is invalid")
    payload: dict[str, Any] = {
        "schema_version": ATTEMPT_SCHEMA,
        "attempt_id": claim_id,
        "attempt_status": "claimed_for_payload_verification",
        "claimed_at_utc": claimed_at_utc,
        "authorization_id": verified["authorization_id"],
        "authorization_payload_sha256": verified[
            "authorization_payload_sha256"
        ],
        "hostname": socket.gethostname(),
        "process_identity": identity,
        "output_root": verified["output_root"],
        "single_use": True,
        "stale_lock_retirement_authorized": False,
        "scientific_values_emitted": False,
        "stability_analysis_authorized": False,
    }
    payload["attempt_payload_sha256"] = sha256_bytes(
        canonical_json_bytes(payload)
    )
    return payload


def verify_attempt_receipt(
    receipt: Mapping[str, Any], *, authorization: Mapping[str, Any]
) -> dict[str, Any]:
    supplied = dict(receipt)
    verify_self_hash(
        supplied,
        hash_field="attempt_payload_sha256",
        label="closeout recovery attempt",
    )
    required = {
        "schema_version",
        "attempt_id",
        "attempt_status",
        "claimed_at_utc",
        "authorization_id",
        "authorization_payload_sha256",
        "hostname",
        "process_identity",
        "output_root",
        "single_use",
        "stale_lock_retirement_authorized",
        "scientific_values_emitted",
        "stability_analysis_authorized",
        "attempt_payload_sha256",
    }
    if set(supplied) != required:
        raise Gate12C2CloseoutRecoveryError(
            "closeout recovery attempt schema mismatch"
        )
    expected = build_attempt_receipt(
        authorization,
        claimed_at_utc=str(supplied["claimed_at_utc"]),
        attempt_id=str(supplied["attempt_id"]),
        process_identity_value=supplied["process_identity"],
    )
    if supplied != expected:
        raise Gate12C2CloseoutRecoveryError(
            "closeout recovery attempt differs"
        )
    return expected


def attempt_process_state(attempt: Mapping[str, Any]) -> str:
    if attempt.get("hostname") != socket.gethostname():
        return PROCESS_UNKNOWN
    identity = attempt.get("process_identity")
    if not isinstance(identity, Mapping):
        return PROCESS_UNKNOWN
    return process_identity_state(identity)




def _protected_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in manifest["files"]
        if row["plane"] == "protected_payload"
    ]


def verify_payload_semantics(
    *,
    output_root: Path,
    archived_plan_path: Path,
    incident_manifest: Mapping[str, Any],
    legacy_liveness_observation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Machine-verify payloads while exposing only allowlisted hashes/counts."""

    root = Path(output_root).resolve()
    before = verify_incident_manifest(
        incident_manifest, root=root, require_exact=True
    )
    if legacy_liveness_observation is None:
        lineage = verify_legacy_lineage(
            output_root=root,
            archived_plan_path=archived_plan_path,
            expected_source_commit=str(before["original_source_commit"]),
            expected_plan_payload_sha256=str(
                before["original_plan_payload_sha256"]
            ),
        )
    else:
        static_lineage = _verify_legacy_lineage_static(
            output_root=root,
            archived_plan_path=archived_plan_path,
            expected_source_commit=str(before["original_source_commit"]),
            expected_plan_payload_sha256=str(
                before["original_plan_payload_sha256"]
            ),
        )
        lineage = _compose_legacy_lineage_evidence(
            static_lineage, legacy_liveness_observation
        )
    plan = _verify_canonical_mapping_file(
        archived_plan_path, label="archived exact plan"
    )
    configurations = plan.get("configurations")
    if not isinstance(configurations, list) or len(configurations) != 9:
        raise Gate12C2CloseoutRecoveryError("legacy configuration surface is invalid")
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    try:
        for configuration in configurations:
            if not isinstance(configuration, Mapping):
                raise Gate12C2CloseoutRecoveryError(
                    "legacy configuration surface is invalid"
                )
            configuration_id = str(configuration["configuration_id"])
            relative = str(configuration["output_relative_path"])
            if configuration_id in seen or relative not in expected_directories():
                raise Gate12C2CloseoutRecoveryError(
                    "legacy configuration surface is invalid"
                )
            seen.add(configuration_id)
            subplan = configuration["subplan"]
            output_dir = root / relative
            plan_path = output_dir / "plan.json"
            if plan_path.read_bytes() != canonical_json_bytes(subplan):
                raise Gate12C2CloseoutRecoveryError(
                    "configuration plan bytes changed"
                )
            shards._verified_plan(subplan)
            verification = shards.verify_development_shard_index(
                subplan, output_dir=output_dir
            )
            expected_outer = len(subplan["outer_experiment_indices"])
            if int(verification["outer_experiment_count"]) != expected_outer:
                raise Gate12C2CloseoutRecoveryError(
                    "configuration outer surface is incomplete"
                )
            rows.append(
                {
                    "configuration_id": configuration_id,
                    "outer_experiment_count": expected_outer,
                    "plan_payload_sha256": verification[
                        "plan_payload_sha256"
                    ],
                    "index_payload_sha256": verification[
                        "index_payload_sha256"
                    ],
                    "scientific_projection_sha256": verification[
                        "scientific_projection_sha256"
                    ],
                    "status": "verified",
                }
            )
    except Gate12C2CloseoutRecoveryError:
        raise
    except Exception:
        raise Gate12C2CloseoutRecoveryError(
            "payload semantic verification rejected"
        ) from None
    after_scan = scan_root_bytes(root)
    after_summary = _scan_summary(after_scan)
    after_manifest_rows = after_scan["files"]
    if (
        _protected_rows(before)
        != [
            dict(row)
            for row in after_manifest_rows
            if row["plane"] == "protected_payload"
        ]
        or after_summary["complete_surface_sha256"]
        != before["summary"]["complete_surface_sha256"]
        or after_summary["protected_surface_sha256"]
        != before["summary"]["protected_surface_sha256"]
    ):
        raise Gate12C2CloseoutRecoveryError(
            "payload bytes changed during semantic verification"
        )
    return {
        "schema_version": "gate12c2_payload_semantic_verification_v0.1",
        "status": "verified",
        "configuration_count": 9,
        "outer_experiment_count": 768,
        "shard_count": 768,
        "index_count": 9,
        "configuration_results": sorted(
            rows, key=lambda row: row["configuration_id"]
        ),
        "protected_surface_sha256": before["summary"][
            "protected_surface_sha256"
        ],
        "complete_surface_sha256": before["summary"][
            "complete_surface_sha256"
        ],
        "legacy_lineage_evidence_sha256": sha256_bytes(
            canonical_json_bytes(lineage)
        ),
        "payload_added": 0,
        "payload_modified": 0,
        "payload_deleted": 0,
        "index_bytes_changed": 0,
        "scientific_values_emitted": False,
    }


def build_consumption_receipt(
    authorization: Mapping[str, Any],
    attempt: Mapping[str, Any],
    *,
    consumed_at_utc: str,
    require_current_freshness: bool = True,
) -> dict[str, Any]:
    verified = verify_recovery_authorization(
        authorization, require_current_freshness=require_current_freshness
    )
    verified_attempt = verify_attempt_receipt(
        attempt, authorization=verified
    )
    consumed = parse_utc(consumed_at_utc, label="recovery consumption time")
    issued = parse_utc(verified["issued_at_utc"], label="authorization issue time")
    expiration = parse_utc(
        verified["expires_at_utc"], label="authorization expiration"
    )
    claimed = parse_utc(
        verified_attempt["claimed_at_utc"], label="recovery attempt claim time"
    )
    if not (issued <= claimed <= consumed < expiration):
        raise Gate12C2CloseoutRecoveryError(
            "recovery consumption time is outside the frozen sequence"
        )
    payload: dict[str, Any] = {
        "schema_version": CONSUMPTION_SCHEMA,
        "authorization_payload_sha256": verified[
            "authorization_payload_sha256"
        ],
        "authorization_id": verified["authorization_id"],
        "authorization_scope": verified["authorization_scope"],
        "authorization_status": "consumed_for_payload_verification",
        "attempt_id": verified_attempt["attempt_id"],
        "attempt_payload_sha256": verified_attempt["attempt_payload_sha256"],
        "single_use": True,
        "consumed_at_utc": consumed_at_utc,
        "output_root": verified["output_root"],
        "incident_manifest_payload_sha256": verified[
            "incident_manifest_payload_sha256"
        ],
        "amendment_payload_sha256": verified["amendment_payload_sha256"],
        "stale_lock_retirement_authorized": False,
        "scientific_values_emitted": False,
    }
    payload["consumption_payload_sha256"] = sha256_bytes(
        canonical_json_bytes(payload)
    )
    return payload


def verify_consumption_receipt(
    receipt: Mapping[str, Any],
    *,
    authorization: Mapping[str, Any],
    attempt: Mapping[str, Any],
) -> dict[str, Any]:
    supplied = dict(receipt)
    verify_self_hash(
        supplied,
        hash_field="consumption_payload_sha256",
        label="closeout recovery consumption",
    )
    expected = build_consumption_receipt(
        authorization,
        attempt,
        consumed_at_utc=str(supplied.get("consumed_at_utc", "")),
        require_current_freshness=False,
    )
    if supplied != expected:
        raise Gate12C2CloseoutRecoveryError("recovery consumption differs")
    return expected


def verify_semantic_verification_summary(
    semantic_verification: Mapping[str, Any],
) -> dict[str, Any]:
    supplied = dict(semantic_verification)
    required = {
        "schema_version",
        "status",
        "configuration_count",
        "outer_experiment_count",
        "shard_count",
        "index_count",
        "configuration_results",
        "protected_surface_sha256",
        "complete_surface_sha256",
        "legacy_lineage_evidence_sha256",
        "payload_added",
        "payload_modified",
        "payload_deleted",
        "index_bytes_changed",
        "scientific_values_emitted",
    }
    if set(supplied) != required:
        raise Gate12C2CloseoutRecoveryError(
            "payload semantic verification schema mismatch"
        )
    exact_counts = {
        "configuration_count": 9,
        "outer_experiment_count": 768,
        "shard_count": 768,
        "index_count": 9,
        "payload_added": 0,
        "payload_modified": 0,
        "payload_deleted": 0,
        "index_bytes_changed": 0,
    }
    if any(
        _strict_int(supplied.get(key), label=key) != value
        for key, value in exact_counts.items()
    ):
        raise Gate12C2CloseoutRecoveryError(
            "payload semantic verification counts differ"
        )
    if (
        supplied.get("schema_version")
        != "gate12c2_payload_semantic_verification_v0.1"
        or supplied.get("status") != "verified"
        or supplied.get("scientific_values_emitted") is not False
        or any(
            not is_sha256(supplied.get(key))
            for key in (
                "protected_surface_sha256",
                "complete_surface_sha256",
                "legacy_lineage_evidence_sha256",
            )
        )
    ):
        raise Gate12C2CloseoutRecoveryError(
            "payload semantic verification is not sealable"
        )
    rows = supplied.get("configuration_results")
    if not isinstance(rows, list) or len(rows) != 9:
        raise Gate12C2CloseoutRecoveryError(
            "payload semantic configuration evidence differs"
        )
    seen: set[str] = set()
    outer_total = 0
    row_required = {
        "configuration_id",
        "outer_experiment_count",
        "plan_payload_sha256",
        "index_payload_sha256",
        "scientific_projection_sha256",
        "status",
    }
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != row_required:
            raise Gate12C2CloseoutRecoveryError(
                "payload semantic configuration evidence differs"
            )
        configuration_id = row.get("configuration_id")
        if (
            not isinstance(configuration_id, str)
            or not re.fullmatch(r"[A-Za-z0-9._-]+", configuration_id)
            or configuration_id in seen
            or row.get("status") != "verified"
            or any(
                not is_sha256(row.get(key))
                for key in (
                    "plan_payload_sha256",
                    "index_payload_sha256",
                    "scientific_projection_sha256",
                )
            )
        ):
            raise Gate12C2CloseoutRecoveryError(
                "payload semantic configuration evidence differs"
            )
        seen.add(configuration_id)
        outer_count = _strict_int(
            row.get("outer_experiment_count"),
            label="configuration outer count",
        )
        if outer_count <= 0:
            raise Gate12C2CloseoutRecoveryError(
                "payload semantic configuration evidence differs"
            )
        outer_total += outer_count
    if outer_total != 768:
        raise Gate12C2CloseoutRecoveryError(
            "payload semantic configuration evidence differs"
        )
    return supplied


def build_payload_seal(
    *,
    authorization: Mapping[str, Any],
    attempt: Mapping[str, Any],
    consumption: Mapping[str, Any],
    sealed_at_utc: str,
) -> dict[str, Any]:
    verified_auth = verify_recovery_authorization(
        authorization, require_current_freshness=False
    )
    verified_attempt = verify_attempt_receipt(
        attempt, authorization=verified_auth
    )
    verified_consumption = verify_consumption_receipt(
        consumption,
        authorization=verified_auth,
        attempt=verified_attempt,
    )
    sealed = parse_utc(sealed_at_utc, label="payload seal time")
    consumed = parse_utc(
        verified_consumption["consumed_at_utc"],
        label="recovery consumption time",
    )
    if sealed < consumed:
        raise Gate12C2CloseoutRecoveryError(
            "payload seal time precedes recovery consumption"
        )
    semantic_verification = verify_semantic_verification_summary(
        verify_payload_semantics(
            output_root=Path(verified_auth["output_root"]),
            archived_plan_path=Path(verified_auth["archived_plan_path"]),
            incident_manifest=read_mapping(
                Path(verified_auth["incident_manifest_path"]),
                label="incident manifest",
            ),
            legacy_liveness_observation=verified_auth[
                "stale_lock_liveness_observation"
            ],
        )
    )
    if (
        semantic_verification["legacy_lineage_evidence_sha256"]
        != verified_auth["legacy_lineage_evidence_sha256"]
    ):
        raise Gate12C2CloseoutRecoveryError(
            "payload semantic lineage differs from frozen authorization evidence"
        )
    payload: dict[str, Any] = {
        "schema_version": PAYLOAD_SEAL_SCHEMA,
        "state": "PAYLOAD_COMPLETION_SEALED",
        "sealed_at_utc": sealed_at_utc,
        "authorization_payload_sha256": verified_auth[
            "authorization_payload_sha256"
        ],
        "attempt_id": verified_attempt["attempt_id"],
        "attempt_payload_sha256": verified_attempt["attempt_payload_sha256"],
        "consumption_payload_sha256": verified_consumption[
            "consumption_payload_sha256"
        ],
        "incident_manifest_payload_sha256": verified_auth[
            "incident_manifest_payload_sha256"
        ],
        "amendment_payload_sha256": verified_auth[
            "amendment_payload_sha256"
        ],
        "original_execution_closeout_status": "failed",
        "payload_presence_observed": "768/768",
        "payload_integrity_status": "verified",
        "index_integrity_status": "verified",
        "payload_completion_seal": "verified",
        "recovery_lineage_status": "verified",
        "configuration_count": 9,
        "outer_experiment_count": 768,
        "protected_surface_sha256": semantic_verification[
            "protected_surface_sha256"
        ],
        "complete_surface_sha256": semantic_verification[
            "complete_surface_sha256"
        ],
        "configuration_evidence_sha256": sha256_bytes(
            canonical_json_bytes(
                semantic_verification["configuration_results"]
            )
        ),
        "payload_added": 0,
        "payload_modified": 0,
        "payload_deleted": 0,
        "index_bytes_changed": 0,
        "stale_lock_status": "present_pending_separate_retirement",
        "original_resource_evidence_status": "missing",
        "resource_gate_status": "indeterminate",
        "replacement_resource_qualification": "not_performed",
        "scientific_values_emitted": False,
        "stability_analysis_authorized": False,
        "locked_execution_authorized": False,
        "real_held_out_execution_authorized": False,
        "N2_open": False,
        "N3_open": False,
        "next_gate": "fresh_review_of_payload_seal_and_resource_qualification_design",
    }
    payload["payload_seal_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def build_recovery_failure(
    *,
    authorization: Mapping[str, Any],
    attempt: Mapping[str, Any],
    consumption: Mapping[str, Any] | None,
    failure_state: str,
    failure_phase: str,
    recorded_at_utc: str,
) -> dict[str, Any]:
    if failure_state not in FAILURE_PHASES_BY_STATE:
        raise Gate12C2CloseoutRecoveryError("recovery failure state is invalid")
    if failure_phase not in FAILURE_PHASE_CONSUMPTION:
        raise Gate12C2CloseoutRecoveryError("recovery failure phase is invalid")
    if failure_phase not in FAILURE_PHASES_BY_STATE[failure_state]:
        raise Gate12C2CloseoutRecoveryError(
            "recovery failure state and phase are inconsistent"
        )
    verified_auth = verify_recovery_authorization(
        authorization, require_current_freshness=False
    )
    verified_attempt = verify_attempt_receipt(
        attempt, authorization=verified_auth
    )
    verified_consumption = None
    if consumption is not None:
        verified_consumption = verify_consumption_receipt(
            consumption,
            authorization=verified_auth,
            attempt=verified_attempt,
        )
    consumption_requirement = FAILURE_PHASE_CONSUMPTION[failure_phase]
    consumption_present = verified_consumption is not None
    if (
        (consumption_requirement == "absent" and consumption_present)
        or (consumption_requirement == "present" and not consumption_present)
    ):
        raise Gate12C2CloseoutRecoveryError(
            "recovery failure phase and consumption status are inconsistent"
        )
    recorded = parse_utc(recorded_at_utc, label="recovery failure time")
    earliest = parse_utc(
        (
            verified_consumption["consumed_at_utc"]
            if verified_consumption is not None
            else verified_attempt["claimed_at_utc"]
        ),
        label="recovery failure predecessor time",
    )
    if recorded < earliest:
        raise Gate12C2CloseoutRecoveryError(
            "recovery failure time precedes its predecessor"
        )
    reason_codes = {
        "RECOVERY_REJECTED": "GATE12C2_CLOSEOUT_RECOVERY_REJECTED",
        "RECOVERY_INTERRUPTED": "GATE12C2_CLOSEOUT_RECOVERY_INTERRUPTED",
        "PAYLOAD_MISMATCH": "GATE12C2_CLOSEOUT_RECOVERY_PAYLOAD_MISMATCH",
    }
    payload: dict[str, Any] = {
        "schema_version": RECOVERY_FAILURE_SCHEMA,
        "state": failure_state,
        "failure_phase": failure_phase,
        "recorded_at_utc": recorded_at_utc,
        "authorization_payload_sha256": verified_auth[
            "authorization_payload_sha256"
        ],
        "attempt_id": verified_attempt["attempt_id"],
        "attempt_payload_sha256": verified_attempt["attempt_payload_sha256"],
        "consumption_status": (
            "present_verified" if verified_consumption is not None else "not_created"
        ),
        "consumption_payload_sha256": (
            verified_consumption["consumption_payload_sha256"]
            if verified_consumption is not None
            else None
        ),
        "output_root": verified_auth["output_root"],
        "failure_reason_code": reason_codes[failure_state],
        "raw_exception_emitted": False,
        "protected_payload_mutation_detected": "not_established",
        "stale_lock_retired": False,
        "authorization_reusable": False,
        "new_authorization_required": True,
        "original_resource_evidence_status": "missing",
        "resource_gate_status": "indeterminate",
        "scientific_values_emitted": False,
        "stability_analysis_authorized": False,
    }
    payload["recovery_failure_payload_sha256"] = sha256_bytes(
        canonical_json_bytes(payload)
    )
    return payload


def build_terminal_claim(
    *,
    authorization: Mapping[str, Any],
    attempt: Mapping[str, Any],
    consumption: Mapping[str, Any] | None,
    outcome: Mapping[str, Any],
    terminal_claimed_at_utc: str,
) -> dict[str, Any]:
    verified_auth = verify_recovery_authorization(
        authorization, require_current_freshness=False
    )
    verified_attempt = verify_attempt_receipt(
        attempt, authorization=verified_auth
    )
    verified_consumption = None
    if consumption is not None:
        verified_consumption = verify_consumption_receipt(
            consumption,
            authorization=verified_auth,
            attempt=verified_attempt,
        )
    supplied_outcome = dict(outcome)
    if (
        supplied_outcome.get("schema_version") == PAYLOAD_SEAL_SCHEMA
        and supplied_outcome.get("state") == "PAYLOAD_COMPLETION_SEALED"
    ):
        if verified_consumption is None:
            raise Gate12C2CloseoutRecoveryError(
                "payload seal terminal claim requires consumption"
            )
        verify_self_hash(
            supplied_outcome,
            hash_field="payload_seal_sha256",
            label="payload completion seal",
        )
        terminal_kind = "payload_seal"
        outcome_payload_sha256 = supplied_outcome["payload_seal_sha256"]
        outcome_time = parse_utc(
            supplied_outcome.get("sealed_at_utc"), label="payload seal time"
        )
        outcome_output = Path(verified_auth["seal_output"]).resolve()
        opposite_output = Path(verified_auth["failure_output"]).resolve()
    elif (
        supplied_outcome.get("schema_version") == RECOVERY_FAILURE_SCHEMA
        and supplied_outcome.get("state")
        in {"RECOVERY_REJECTED", "RECOVERY_INTERRUPTED", "PAYLOAD_MISMATCH"}
    ):
        verify_self_hash(
            supplied_outcome,
            hash_field="recovery_failure_payload_sha256",
            label="closeout recovery failure",
        )
        expected_failure = build_recovery_failure(
            authorization=verified_auth,
            attempt=verified_attempt,
            consumption=verified_consumption,
            failure_state=str(supplied_outcome.get("state", "")),
            failure_phase=str(supplied_outcome.get("failure_phase", "")),
            recorded_at_utc=str(supplied_outcome.get("recorded_at_utc", "")),
        )
        if supplied_outcome != expected_failure:
            raise Gate12C2CloseoutRecoveryError(
                "terminal recovery failure differs from exact evidence"
            )
        supplied_outcome = expected_failure
        terminal_kind = "recovery_failure"
        outcome_payload_sha256 = supplied_outcome[
            "recovery_failure_payload_sha256"
        ]
        outcome_time = parse_utc(
            supplied_outcome.get("recorded_at_utc"), label="recovery failure time"
        )
        outcome_output = Path(verified_auth["failure_output"]).resolve()
        opposite_output = Path(verified_auth["seal_output"]).resolve()
    else:
        raise Gate12C2CloseoutRecoveryError(
            "terminal claim outcome is invalid"
        )
    terminal_time = parse_utc(
        terminal_claimed_at_utc, label="terminal claim time"
    )
    if terminal_time < outcome_time:
        raise Gate12C2CloseoutRecoveryError(
            "terminal claim time precedes its outcome"
        )
    payload: dict[str, Any] = {
        "schema_version": TERMINAL_CLAIM_SCHEMA,
        "terminal_kind": terminal_kind,
        "terminal_state": supplied_outcome["state"],
        "terminal_claimed_at_utc": terminal_claimed_at_utc,
        "terminal_output": Path(verified_auth["terminal_output"])
        .resolve()
        .as_posix(),
        "authorization_payload_sha256": verified_auth[
            "authorization_payload_sha256"
        ],
        "attempt_id": verified_attempt["attempt_id"],
        "attempt_payload_sha256": verified_attempt["attempt_payload_sha256"],
        "consumption_status": (
            "present_verified"
            if verified_consumption is not None
            else "not_created"
        ),
        "consumption_payload_sha256": (
            verified_consumption["consumption_payload_sha256"]
            if verified_consumption is not None
            else None
        ),
        "outcome_output": outcome_output.as_posix(),
        "opposite_output": opposite_output.as_posix(),
        "outcome_payload_sha256": outcome_payload_sha256,
        "single_terminal_state": True,
        "original_resource_evidence_status": "missing",
        "resource_gate_status": "indeterminate",
        "scientific_values_emitted": False,
        "stability_analysis_authorized": False,
    }
    payload["terminal_claim_payload_sha256"] = sha256_bytes(
        canonical_json_bytes(payload)
    )
    return payload


def verify_terminal_claim(
    claim: Mapping[str, Any],
    *,
    authorization: Mapping[str, Any],
    attempt: Mapping[str, Any],
    consumption: Mapping[str, Any] | None,
    outcome: Mapping[str, Any],
) -> dict[str, Any]:
    supplied = dict(claim)
    verify_self_hash(
        supplied,
        hash_field="terminal_claim_payload_sha256",
        label="closeout recovery terminal claim",
    )
    expected = build_terminal_claim(
        authorization=authorization,
        attempt=attempt,
        consumption=consumption,
        outcome=outcome,
        terminal_claimed_at_utc=str(
            supplied.get("terminal_claimed_at_utc", "")
        ),
    )
    if supplied != expected:
        raise Gate12C2CloseoutRecoveryError(
            "closeout recovery terminal claim differs"
        )
    return expected


def _verify_published_attempt_and_consumption(
    *,
    authorization: Mapping[str, Any],
    attempt: Mapping[str, Any],
    consumption: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    """Bind terminal publication to canonical physical execution evidence."""

    verified_auth = verify_recovery_authorization(
        authorization, require_current_freshness=False
    )
    actual_attempt = _verify_canonical_mapping_file(
        Path(verified_auth["attempt_output"]),
        label="closeout recovery attempt",
    )
    verified_attempt = verify_attempt_receipt(
        actual_attempt, authorization=verified_auth
    )
    supplied_attempt = verify_attempt_receipt(
        attempt, authorization=verified_auth
    )
    if actual_attempt != supplied_attempt:
        raise Gate12C2CloseoutRecoveryError(
            "recovery terminal attempt evidence differs"
        )
    actual_consumption = _canonical_mapping_if_present(
        Path(verified_auth["consumption_output"]),
        label="closeout recovery consumption",
    )
    if consumption is None:
        if actual_consumption is not None:
            raise Gate12C2CloseoutRecoveryError(
                "recovery terminal omits published consumption evidence"
            )
        verified_consumption = None
    else:
        if actual_consumption is None:
            raise Gate12C2CloseoutRecoveryError(
                "recovery terminal consumption evidence is missing"
            )
        supplied_consumption = verify_consumption_receipt(
            consumption,
            authorization=verified_auth,
            attempt=verified_attempt,
        )
        verified_consumption = verify_consumption_receipt(
            actual_consumption,
            authorization=verified_auth,
            attempt=verified_attempt,
        )
        if actual_consumption != supplied_consumption:
            raise Gate12C2CloseoutRecoveryError(
                "recovery terminal consumption evidence differs"
            )
    return verified_auth, verified_attempt, verified_consumption


def _publish_terminal_outcome(
    *,
    authorization: Mapping[str, Any],
    attempt: Mapping[str, Any],
    consumption: Mapping[str, Any] | None,
    outcome: Mapping[str, Any],
) -> dict[str, Any]:
    verified_auth, verified_attempt, verified_consumption = (
        _verify_published_attempt_and_consumption(
            authorization=authorization,
            attempt=attempt,
            consumption=consumption,
        )
    )
    terminal_path = Path(verified_auth["terminal_output"])
    seal_path = Path(verified_auth["seal_output"])
    failure_path = Path(verified_auth["failure_output"])
    if terminal_path.exists() or seal_path.exists() or failure_path.exists():
        raise Gate12C2CloseoutRecoveryError(
            "recovery authorization already has a terminal state"
        )
    claim = build_terminal_claim(
        authorization=verified_auth,
        attempt=verified_attempt,
        consumption=verified_consumption,
        outcome=outcome,
        terminal_claimed_at_utc=utc_now().isoformat(),
    )
    _publish_exact_or_recover(
        terminal_path, claim, label="closeout recovery terminal claim"
    )
    if claim["terminal_kind"] == "payload_seal":
        if failure_path.exists():
            raise Gate12C2CloseoutRecoveryError(
                "opposite recovery outcome exists"
            )
        _publish_exact_or_recover(
            seal_path, outcome, label="payload completion seal"
        )
    else:
        if seal_path.exists():
            raise Gate12C2CloseoutRecoveryError(
                "opposite recovery outcome exists"
            )
        _publish_exact_or_recover(
            failure_path, outcome, label="closeout recovery failure"
        )
    return claim


def verify_recovery_failure(
    failure: Mapping[str, Any],
    *,
    authorization: Mapping[str, Any],
    attempt: Mapping[str, Any],
    consumption: Mapping[str, Any] | None,
) -> dict[str, Any]:
    supplied = dict(failure)
    verify_self_hash(
        supplied,
        hash_field="recovery_failure_payload_sha256",
        label="closeout recovery failure",
    )
    verified_auth, verified_attempt, verified_consumption = (
        _verify_published_attempt_and_consumption(
            authorization=authorization,
            attempt=attempt,
            consumption=consumption,
        )
    )
    expected = build_recovery_failure(
        authorization=verified_auth,
        attempt=verified_attempt,
        consumption=verified_consumption,
        failure_state=str(supplied.get("state", "")),
        failure_phase=str(supplied.get("failure_phase", "")),
        recorded_at_utc=str(supplied.get("recorded_at_utc", "")),
    )
    if supplied != expected:
        raise Gate12C2CloseoutRecoveryError(
            "closeout recovery failure differs"
        )
    seal_path = Path(verified_auth["seal_output"])
    failure_path = Path(verified_auth["failure_output"])
    terminal_path = Path(verified_auth["terminal_output"])
    if seal_path.exists():
        raise Gate12C2CloseoutRecoveryError(
            "payload seal and recovery failure cannot coexist"
        )
    if (
        not failure_path.is_file()
        or failure_path.read_bytes() != canonical_json_bytes(supplied)
    ):
        raise Gate12C2CloseoutRecoveryError(
            "recovery failure output differs"
        )
    claim = _verify_canonical_mapping_file(
        terminal_path, label="closeout recovery terminal claim"
    )
    verified_claim = verify_terminal_claim(
        claim,
        authorization=verified_auth,
        attempt=verified_attempt,
        consumption=verified_consumption,
        outcome=expected,
    )
    if verified_claim["terminal_kind"] != "recovery_failure":
        raise Gate12C2CloseoutRecoveryError(
            "terminal claim is not a recovery failure"
        )
    return expected


def _canonical_mapping_if_present(path: Path, *, label: str) -> dict[str, Any] | None:
    if not Path(path).exists():
        return None
    return _verify_canonical_mapping_file(path, label=label)


def _publish_exact_or_recover(
    path: Path, payload: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    """Publish evidence, or recover the exact canonical output after an error."""

    try:
        write_exclusive_atomic(path, payload)
    except Exception:
        actual = _canonical_mapping_if_present(path, label=label)
        if actual is None or actual != dict(payload):
            raise
        return actual
    actual = _verify_canonical_mapping_file(path, label=label)
    if actual != dict(payload):
        raise Gate12C2CloseoutRecoveryError(f"{label} output differs")
    return actual


def _load_authorization_for_execution(
    authorization_path: Path, *, require_current_freshness: bool
) -> dict[str, Any]:
    supplied = _verify_canonical_mapping_file(
        authorization_path, label="closeout recovery authorization"
    )
    verified = verify_recovery_authorization(
        supplied, require_current_freshness=require_current_freshness
    )
    if Path(authorization_path).resolve() != Path(
        verified["authorization_output"]
    ).resolve():
        raise Gate12C2CloseoutRecoveryError(
            "authorization path differs from authorized output"
        )
    return verified


def execute_payload_seal(authorization_path: Path) -> dict[str, Any]:
    """Consume one authorization and seal payload externally, without root writes."""

    authorization = _load_authorization_for_execution(
        authorization_path, require_current_freshness=False
    )
    _require_current_recovery_execution_context(authorization)
    attempt_path = Path(authorization["attempt_output"])
    consumption_path = Path(authorization["consumption_output"])
    terminal_path = Path(authorization["terminal_output"])
    seal_path = Path(authorization["seal_output"])
    failure_path = Path(authorization["failure_output"])
    if terminal_path.exists() or seal_path.exists() or failure_path.exists():
        raise Gate12C2CloseoutRecoveryError(
            "recovery authorization has already been terminally claimed"
        )
    existing_attempt = _canonical_mapping_if_present(
        attempt_path, label="closeout recovery attempt"
    )
    existing_consumption = _canonical_mapping_if_present(
        consumption_path, label="closeout recovery consumption"
    )
    if existing_attempt is not None:
        attempt = verify_attempt_receipt(
            existing_attempt, authorization=authorization
        )
        liveness = attempt_process_state(attempt)
        if liveness == PROCESS_ACTIVE:
            raise Gate12C2CloseoutRecoveryError(
                "recovery attempt is already active"
            )
        if liveness == PROCESS_UNKNOWN:
            raise Gate12C2CloseoutRecoveryError(
                "recovery attempt liveness is indeterminate"
            )
        consumption = None
        if existing_consumption is not None:
            consumption = verify_consumption_receipt(
                existing_consumption,
                authorization=authorization,
                attempt=attempt,
            )
        failure = build_recovery_failure(
            authorization=authorization,
            attempt=attempt,
            consumption=consumption,
            failure_state="RECOVERY_INTERRUPTED",
            failure_phase=(
                "post_consumption_restart"
                if consumption is not None
                else "attempt_claimed"
            ),
            recorded_at_utc=utc_now().isoformat(),
        )
        _publish_terminal_outcome(
            authorization=authorization,
            attempt=attempt,
            consumption=consumption,
            outcome=failure,
        )
        raise Gate12C2CloseoutRecoveryError(
            "interrupted recovery attempt was sealed as failed"
        )
    if existing_consumption is not None:
        raise Gate12C2CloseoutRecoveryError(
            "recovery consumption exists without an attempt identity"
        )
    authorization = _load_authorization_for_execution(
        authorization_path, require_current_freshness=True
    )
    attempt = build_attempt_receipt(
        authorization,
        claimed_at_utc=utc_now().isoformat(),
    )
    attempt = _publish_exact_or_recover(
        attempt_path, attempt, label="closeout recovery attempt"
    )
    consumption: dict[str, Any] | None = None

    phase = "consumption_publication"
    try:
        consumption = build_consumption_receipt(
            authorization,
            attempt,
            consumed_at_utc=utc_now().isoformat(),
        )
        consumption = _publish_exact_or_recover(
            consumption_path,
            consumption,
            label="closeout recovery consumption",
        )
        consumption = verify_consumption_receipt(
            consumption,
            authorization=authorization,
            attempt=attempt,
        )
        phase = "payload_verification"
        seal = build_payload_seal(
            authorization=authorization,
            attempt=attempt,
            consumption=consumption,
            sealed_at_utc=utc_now().isoformat(),
        )
        phase = "seal_publication"
        _publish_terminal_outcome(
            authorization=authorization,
            attempt=attempt,
            consumption=consumption,
            outcome=seal,
        )
        return seal
    except Exception as error:
        if terminal_path.exists():
            raise Gate12C2CloseoutRecoveryError(
                "payload closeout recovery terminal state was claimed"
            ) from None
        failure_state = (
            "PAYLOAD_MISMATCH"
            if phase == "payload_verification"
            and isinstance(error, Gate12C2CloseoutRecoveryError)
            else "RECOVERY_INTERRUPTED"
        )
        try:
            physical_consumption = _canonical_mapping_if_present(
                consumption_path,
                label="closeout recovery consumption",
            )
            verified_physical_consumption = None
            if physical_consumption is not None:
                verified_physical_consumption = verify_consumption_receipt(
                    physical_consumption,
                    authorization=authorization,
                    attempt=attempt,
                )
        except Exception:
            raise Gate12C2CloseoutRecoveryError(
                "published recovery consumption could not be verified"
            ) from None
        if not seal_path.exists() and not failure_path.exists():
            failure = build_recovery_failure(
                authorization=authorization,
                attempt=attempt,
                consumption=verified_physical_consumption,
                failure_state=failure_state,
                failure_phase=phase,
                recorded_at_utc=utc_now().isoformat(),
            )
            _publish_terminal_outcome(
                authorization=authorization,
                attempt=attempt,
                consumption=verified_physical_consumption,
                outcome=failure,
            )
        raise Gate12C2CloseoutRecoveryError(
            "payload closeout recovery was rejected"
        ) from None

def verify_payload_seal(
    *, authorization_path: Path, seal_path: Path
) -> dict[str, Any]:
    authorization = _load_authorization_for_execution(
        authorization_path, require_current_freshness=False
    )
    if Path(seal_path).resolve() != Path(authorization["seal_output"]).resolve():
        raise Gate12C2CloseoutRecoveryError("payload seal path differs")
    if Path(authorization["failure_output"]).exists():
        raise Gate12C2CloseoutRecoveryError(
            "payload seal and recovery failure cannot coexist"
        )
    attempt = _verify_canonical_mapping_file(
        Path(authorization["attempt_output"]),
        label="closeout recovery attempt",
    )
    verify_attempt_receipt(attempt, authorization=authorization)
    consumption = _verify_canonical_mapping_file(
        Path(authorization["consumption_output"]),
        label="closeout recovery consumption",
    )
    verify_consumption_receipt(
        consumption,
        authorization=authorization,
        attempt=attempt,
    )
    supplied = _verify_canonical_mapping_file(
        seal_path, label="payload completion seal"
    )
    verify_self_hash(
        supplied, hash_field="payload_seal_sha256", label="payload completion seal"
    )
    expected = build_payload_seal(
        authorization=authorization,
        attempt=attempt,
        consumption=consumption,
        sealed_at_utc=str(supplied.get("sealed_at_utc", "")),
    )
    if supplied != expected:
        raise Gate12C2CloseoutRecoveryError(
            "payload completion seal differs from current evidence"
        )
    claim = _verify_canonical_mapping_file(
        Path(authorization["terminal_output"]),
        label="closeout recovery terminal claim",
    )
    verified_claim = verify_terminal_claim(
        claim,
        authorization=authorization,
        attempt=attempt,
        consumption=consumption,
        outcome=expected,
    )
    if verified_claim["terminal_kind"] != "payload_seal":
        raise Gate12C2CloseoutRecoveryError(
            "terminal claim is not a payload seal"
        )
    if not (Path(authorization["output_root"]) / LOCK_NAME).is_file():
        raise Gate12C2CloseoutRecoveryError(
            "stale lock was retired without separate authorization"
        )
    return expected

def retire_stale_lock(*_: object, **__: object) -> None:
    """Lock retirement is intentionally unavailable in Task A."""

    raise Gate12C2CloseoutRecoveryError(
        "stale-lock retirement remains HOLD pending separate authorization"
    )
