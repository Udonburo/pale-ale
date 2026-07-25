#!/usr/bin/env python3
"""Bounded, development-only throughput profiling for Gate12C-2.

The profiler launches the deterministic shard runner in fresh subprocesses,
monitors process-tree resident memory, and reports execution feasibility only.
It deliberately excludes calibration outcomes, effect direction, FPR, power,
and S2 identification values from its receipt.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
from ctypes import wintypes
from pathlib import Path
from typing import Any, Mapping, Sequence

import gate12c2_development_shards as shards
import gate12c2_synthetic_lab as lab


PROFILE_PLAN_SCHEMA_VERSION = "gate12c2_throughput_profile_plan_v0.1"
PROFILE_RECEIPT_SCHEMA_VERSION = "gate12c2_throughput_profile_receipt_v0.1"
PROFILE_CONFIGURATION_SCHEMA_VERSION = (
    "gate12c2_throughput_configuration_v0.1"
)
PROFILE_PREFLIGHT_SCHEMA_VERSION = (
    "gate12c2_throughput_no_outcome_preflight_v0.1"
)
PROFILE_AUTHORIZATION_SCHEMA_VERSION = (
    "gate12c2_throughput_execution_authorization_v0.1"
)
PROFILE_ID = "gate12c2_bounded_worker_scaling_v0.1"
PROFILE_MASTER_SEED_PREFIX = "gate12c2-development-throughput-v0.1"
PROFILE_MEMORY_SAMPLE_INTERVAL_SECONDS = 0.1


class Gate12C2ThroughputError(ValueError):
    """Raised when a throughput profile crosses its frozen boundary."""


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
        raise Gate12C2ThroughputError(
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
    paths = {
        "gate12c2_synthetic_lab.py": Path(lab.__file__).resolve(),
        "gate12c2_development_shards.py": Path(shards.__file__).resolve(),
        "gate12c2_throughput_profile.py": Path(__file__).resolve(),
        "run_gate12c2_development_shards.py": Path(__file__)
        .with_name("run_gate12c2_development_shards.py")
        .resolve(),
        "run_gate12c2_throughput_profile.py": Path(__file__)
        .with_name("run_gate12c2_throughput_profile.py")
        .resolve(),
    }
    return {
        name: _sha256_file(path)
        for name, path in sorted(paths.items())
    }


def build_bounded_worker_profile_plan(
    *,
    source_commit: str,
    outer_count_per_workload: int = 4,
    inner_valid_draw_count: int = 255,
    worker_counts: Sequence[int] = (1, 2, 4),
) -> dict[str, Any]:
    """Build the frozen first slice of the bounded throughput profile."""

    if not str(source_commit).strip():
        raise Gate12C2ThroughputError("source_commit must be nonempty")
    if outer_count_per_workload <= 0 or inner_valid_draw_count <= 0:
        raise Gate12C2ThroughputError(
            "outer and inner draw counts must be positive"
        )
    workers = tuple(sorted({int(value) for value in worker_counts}))
    if not workers or any(value <= 0 for value in workers):
        raise Gate12C2ThroughputError(
            "worker counts must be positive"
        )

    configurations = []
    regime_rows = (
        ("S0_true_null", None),
        ("S1_known_reverse_shared_node_coupling", 0.25),
        ("S2_null_inflation", None),
    )
    for regime_id, effect_strength in regime_rows:
        workload_id = f"worker-scaling::{regime_id}"
        master_seed = f"{PROFILE_MASTER_SEED_PREFIX}::{workload_id}"
        for worker_count in workers:
            configurations.append(
                {
                    "schema_version": PROFILE_CONFIGURATION_SCHEMA_VERSION,
                    "configuration_id": (
                        f"{regime_id}__w{worker_count}"
                    ),
                    "profile_slice": "worker_scaling",
                    "workload_id": workload_id,
                    "surface_id": "development",
                    "regime_id": regime_id,
                    "master_seed": master_seed,
                    "outer_experiment_indices": list(
                        range(outer_count_per_workload)
                    ),
                    "inner_valid_draw_count": inner_valid_draw_count,
                    "effect_strength": effect_strength,
                    "worker_count": worker_count,
                    "block_count_schedule": (
                        lab._block_count_receipt(
                            lab.reference_block_count_schedule()
                        )
                    ),
                }
            )

    payload: dict[str, Any] = {
        "schema_version": PROFILE_PLAN_SCHEMA_VERSION,
        "profile_id": PROFILE_ID,
        "epistemic_status": "development_throughput_only",
        "surface_id": "development",
        "development_execution_requires_external_authorization": True,
        "locked_execution_authorized": False,
        "real_held_out_execution_authorized": False,
        "N2_open": False,
        "N3_open": False,
        "public_claim": False,
        "source_commit": str(source_commit),
        "implementation_sha256": _implementation_hashes(),
        "thread_environment": dict(
            sorted(shards.SINGLE_THREAD_ENVIRONMENT.items())
        ),
        "memory_sample_interval_seconds": (
            PROFILE_MEMORY_SAMPLE_INTERVAL_SECONDS
        ),
        "configurations": configurations,
        "selection_boundary": {
            "allowed": [
                "scientific_projection_determinism",
                "wall_time",
                "process_CPU_time",
                "process_tree_peak_RSS",
                "valid_draw_throughput",
                "attempts_per_valid_draw",
                "rejection_burden",
                "merge_and_serialization_overhead",
                "output_bytes",
                "worker_scaling_efficiency",
            ],
            "prohibited": [
                "FPR",
                "power",
                "claim_promotion_rate",
                "effect_direction",
                "observed_to_null_ratio",
                "S2_identification_rate",
            ],
        },
        "decision_rules": {
            "determinism": (
                "all worker counts for one workload must reconstruct the "
                "same scientific projection hash"
            ),
            "memory": (
                "process-tree peak RSS must not exceed 0.75 of physical RAM"
            ),
            "disk": (
                "a full candidate projection with safety factor 1.3 must "
                "leave at least half of currently free disk"
            ),
            "worker_selection": (
                "choose the smallest worker count after which the next "
                "candidate improves effective throughput by less than 0.10, "
                "subject to determinism and memory"
            ),
            "outcome": ["GO", "RECONFIGURE", "REDESIGN"],
            "silent_scientific_count_reduction_allowed": False,
        },
        "next_slice_not_yet_authorized": (
            "255/511/1023 draw scaling is instantiated only after this "
            "worker slice closes, using operational criteria only"
        ),
    }
    payload["profile_plan_payload_sha256"] = _sha256_bytes(
        _canonical_json_bytes(payload)
    )
    return payload


def verify_profile_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    supplied = dict(plan)
    payload = dict(supplied)
    claimed = payload.pop("profile_plan_payload_sha256", None)
    actual = _sha256_bytes(_canonical_json_bytes(payload))
    if claimed != actual:
        raise Gate12C2ThroughputError("profile plan hash mismatch")
    if payload.get("schema_version") != PROFILE_PLAN_SCHEMA_VERSION:
        raise Gate12C2ThroughputError("unsupported profile plan schema")
    if payload.get("surface_id") != "development":
        raise Gate12C2ThroughputError(
            "throughput profiling is development-only"
        )
    if payload.get("locked_execution_authorized") is not False:
        raise Gate12C2ThroughputError("locked execution must remain closed")
    if payload.get("real_held_out_execution_authorized") is not False:
        raise Gate12C2ThroughputError(
            "real held-out execution must remain closed"
        )
    if payload.get("N2_open") is not False or payload.get("N3_open") is not False:
        raise Gate12C2ThroughputError("N2 and N3 must remain closed")
    if payload.get("public_claim") is not False:
        raise Gate12C2ThroughputError("public claim must remain closed")
    if payload.get("implementation_sha256") != _implementation_hashes():
        raise Gate12C2ThroughputError(
            "profile implementation hashes no longer match"
        )
    if payload.get("thread_environment") != dict(
        sorted(shards.SINGLE_THREAD_ENVIRONMENT.items())
    ):
        raise Gate12C2ThroughputError(
            "profile thread environment no longer matches"
        )
    configurations = payload.get("configurations")
    if not isinstance(configurations, list) or not configurations:
        raise Gate12C2ThroughputError(
            "profile configurations must be nonempty"
        )
    ids = [str(row.get("configuration_id")) for row in configurations]
    if len(set(ids)) != len(ids):
        raise Gate12C2ThroughputError(
            "profile configuration IDs must be unique"
        )
    for row in configurations:
        if row.get("surface_id") != "development":
            raise Gate12C2ThroughputError(
                "profile configuration opened a non-development surface"
            )
        if row.get("regime_id") not in shards.ALLOWED_REGIMES:
            raise Gate12C2ThroughputError(
                "profile configuration uses an unsupported regime"
            )
        if int(row.get("worker_count", 0)) <= 0:
            raise Gate12C2ThroughputError(
                "profile worker counts must be positive"
            )
        indices = [int(value) for value in row["outer_experiment_indices"]]
        if (
            not indices
            or len(set(indices)) != len(indices)
            or sorted(indices) != list(
                range(min(indices), min(indices) + len(indices))
            )
        ):
            raise Gate12C2ThroughputError(
                "profile outer IDs must be one contiguous unique range"
            )
        expected_schedule = lab._block_count_receipt(
            lab.reference_block_count_schedule()
        )
        if row.get("block_count_schedule") != expected_schedule:
            raise Gate12C2ThroughputError(
                "profile configuration changed the reference block schedule"
            )
        if int(row.get("inner_valid_draw_count", 0)) <= 0:
            raise Gate12C2ThroughputError(
                "profile inner valid draw count must be positive"
            )
    first = configurations[0]
    outer_count = len(first["outer_experiment_indices"])
    inner_count = int(first["inner_valid_draw_count"])
    workers = tuple(
        sorted({int(row["worker_count"]) for row in configurations})
    )
    if any(
        len(row["outer_experiment_indices"]) != outer_count
        or int(row["inner_valid_draw_count"]) != inner_count
        for row in configurations
    ):
        raise Gate12C2ThroughputError(
            "worker profile configurations changed outer or draw counts"
        )
    expected = build_bounded_worker_profile_plan(
        source_commit=str(payload["source_commit"]),
        outer_count_per_workload=outer_count,
        inner_valid_draw_count=inner_count,
        worker_counts=workers,
    )
    if supplied != expected:
        raise Gate12C2ThroughputError(
            "profile plan differs from the complete builder contract"
        )
    return expected


def _profile_output_root(path: Path) -> str:
    return Path(path).resolve().as_posix()


def build_profile_no_outcome_preflight(
    plan: Mapping[str, Any],
    *,
    output_root: Path,
    preflight_id: str,
    checks: Mapping[str, bool],
) -> dict[str, Any]:
    verified = verify_profile_plan(plan)
    normalized_checks = {
        str(key): bool(value) for key, value in checks.items()
    }
    if set(normalized_checks) != set(shards.REQUIRED_PREFLIGHT_CHECKS):
        raise Gate12C2ThroughputError(
            "profile preflight checks differ from the closed allowlist"
        )
    if not all(normalized_checks.values()):
        raise Gate12C2ThroughputError(
            "profile preflight requires every check to pass"
        )
    if not str(preflight_id).strip():
        raise Gate12C2ThroughputError("profile preflight_id must be nonempty")
    payload: dict[str, Any] = {
        "schema_version": PROFILE_PREFLIGHT_SCHEMA_VERSION,
        "preflight_id": str(preflight_id),
        "epistemic_status": "development_profile_preflight_only",
        "surface_id": "development",
        "preflight_status": "pass",
        "development_execution_authorized": False,
        "locked_execution_authorized": False,
        "real_held_out_execution_authorized": False,
        "N2_open": False,
        "N3_open": False,
        "public_claim": False,
        "scientific_outcomes_inspected": False,
        "profile_plan_payload_sha256": verified[
            "profile_plan_payload_sha256"
        ],
        "implementation_sha256": dict(
            verified["implementation_sha256"]
        ),
        "output_root": _profile_output_root(output_root),
        "checks": dict(sorted(normalized_checks.items())),
    }
    payload["preflight_receipt_payload_sha256"] = _sha256_bytes(
        _canonical_json_bytes(payload)
    )
    return payload


def build_profile_execution_authorization(
    plan: Mapping[str, Any],
    preflight_receipt: Mapping[str, Any],
    *,
    output_root: Path,
    authorization_id: str,
) -> dict[str, Any]:
    verified = verify_profile_plan(plan)
    preflight = _verified_profile_preflight(
        verified,
        preflight_receipt,
        output_root=output_root,
    )
    if not str(authorization_id).strip():
        raise Gate12C2ThroughputError(
            "profile authorization_id must be nonempty"
        )
    payload: dict[str, Any] = {
        "schema_version": PROFILE_AUTHORIZATION_SCHEMA_VERSION,
        "authorization_id": str(authorization_id),
        "epistemic_status": "development_profile_authorization_only",
        "surface_id": "development",
        "development_execution_authorized": True,
        "locked_execution_authorized": False,
        "real_held_out_execution_authorized": False,
        "N2_open": False,
        "N3_open": False,
        "public_claim": False,
        "profile_plan_payload_sha256": verified[
            "profile_plan_payload_sha256"
        ],
        "preflight_receipt_payload_sha256": preflight[
            "preflight_receipt_payload_sha256"
        ],
        "implementation_sha256": dict(
            verified["implementation_sha256"]
        ),
        "output_root": _profile_output_root(output_root),
    }
    payload["authorization_receipt_payload_sha256"] = _sha256_bytes(
        _canonical_json_bytes(payload)
    )
    return payload


def _verified_profile_preflight(
    plan: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    output_root: Path,
) -> dict[str, Any]:
    verified = verify_profile_plan(plan)
    if not isinstance(receipt, Mapping):
        raise Gate12C2ThroughputError(
            "profile preflight receipt must be a mapping"
        )
    supplied = dict(receipt)
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
        "profile_plan_payload_sha256",
        "implementation_sha256",
        "output_root",
        "checks",
        "preflight_receipt_payload_sha256",
    }
    try:
        shards._require_exact_keys(
            supplied,
            expected_keys,
            context="profile preflight receipt",
        )
    except shards.Gate12C2ShardError as exc:
        raise Gate12C2ThroughputError(str(exc)) from exc
    claimed = supplied["preflight_receipt_payload_sha256"]
    unhashed = dict(supplied)
    unhashed.pop("preflight_receipt_payload_sha256")
    if claimed != _sha256_bytes(_canonical_json_bytes(unhashed)):
        raise Gate12C2ThroughputError("profile preflight hash mismatch")
    expected_values = {
        "schema_version": PROFILE_PREFLIGHT_SCHEMA_VERSION,
        "epistemic_status": "development_profile_preflight_only",
        "surface_id": "development",
        "preflight_status": "pass",
        "development_execution_authorized": False,
        "locked_execution_authorized": False,
        "real_held_out_execution_authorized": False,
        "N2_open": False,
        "N3_open": False,
        "public_claim": False,
        "scientific_outcomes_inspected": False,
        "profile_plan_payload_sha256": verified[
            "profile_plan_payload_sha256"
        ],
        "implementation_sha256": verified["implementation_sha256"],
        "output_root": _profile_output_root(output_root),
    }
    for key, expected_value in expected_values.items():
        if supplied[key] != expected_value:
            raise Gate12C2ThroughputError(
                f"profile preflight changed frozen field {key!r}"
            )
    checks = supplied["checks"]
    if not isinstance(checks, Mapping):
        raise Gate12C2ThroughputError(
            "profile preflight checks must be a mapping"
        )
    if set(checks) != set(shards.REQUIRED_PREFLIGHT_CHECKS):
        raise Gate12C2ThroughputError(
            "profile preflight check keys differ from the allowlist"
        )
    if any(value is not True for value in checks.values()):
        raise Gate12C2ThroughputError(
            "profile preflight contains a failed check"
        )
    if not str(supplied["preflight_id"]).strip():
        raise Gate12C2ThroughputError(
            "profile preflight_id must be nonempty"
        )
    return supplied


def _verified_profile_authorization(
    plan: Mapping[str, Any],
    preflight_receipt: Mapping[str, Any],
    authorization_receipt: Mapping[str, Any],
    *,
    output_root: Path,
) -> dict[str, Any]:
    verified = verify_profile_plan(plan)
    preflight = _verified_profile_preflight(
        verified,
        preflight_receipt,
        output_root=output_root,
    )
    if not isinstance(authorization_receipt, Mapping):
        raise Gate12C2ThroughputError(
            "profile authorization receipt must be a mapping"
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
        "profile_plan_payload_sha256",
        "preflight_receipt_payload_sha256",
        "implementation_sha256",
        "output_root",
        "authorization_receipt_payload_sha256",
    }
    try:
        shards._require_exact_keys(
            supplied,
            expected_keys,
            context="profile execution authorization",
        )
    except shards.Gate12C2ShardError as exc:
        raise Gate12C2ThroughputError(str(exc)) from exc
    claimed = supplied["authorization_receipt_payload_sha256"]
    unhashed = dict(supplied)
    unhashed.pop("authorization_receipt_payload_sha256")
    if claimed != _sha256_bytes(_canonical_json_bytes(unhashed)):
        raise Gate12C2ThroughputError(
            "profile execution authorization hash mismatch"
        )
    expected_values = {
        "schema_version": PROFILE_AUTHORIZATION_SCHEMA_VERSION,
        "epistemic_status": "development_profile_authorization_only",
        "surface_id": "development",
        "development_execution_authorized": True,
        "locked_execution_authorized": False,
        "real_held_out_execution_authorized": False,
        "N2_open": False,
        "N3_open": False,
        "public_claim": False,
        "profile_plan_payload_sha256": verified[
            "profile_plan_payload_sha256"
        ],
        "preflight_receipt_payload_sha256": preflight[
            "preflight_receipt_payload_sha256"
        ],
        "implementation_sha256": verified["implementation_sha256"],
        "output_root": _profile_output_root(output_root),
    }
    for key, expected_value in expected_values.items():
        if supplied[key] != expected_value:
            raise Gate12C2ThroughputError(
                f"profile authorization changed frozen field {key!r}"
            )
    if not str(supplied["authorization_id"]).strip():
        raise Gate12C2ThroughputError(
            "profile authorization_id must be nonempty"
        )
    return supplied


def _process_table_windows() -> dict[int, int]:
    th32cs_snapprocess = 0x00000002
    max_path = 260
    ulong_ptr = wintypes.WPARAM

    class ProcessEntry32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ulong_ptr),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * max_path),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.argtypes = [
        wintypes.DWORD,
        wintypes.DWORD,
    ]
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Process32FirstW.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessEntry32W),
    ]
    kernel32.Process32FirstW.restype = wintypes.BOOL
    kernel32.Process32NextW.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessEntry32W),
    ]
    kernel32.Process32NextW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    handle = kernel32.CreateToolhelp32Snapshot(th32cs_snapprocess, 0)
    invalid = ctypes.c_void_p(-1).value
    if handle == invalid:
        raise OSError(ctypes.get_last_error(), "process snapshot failed")
    table: dict[int, int] = {}
    entry = ProcessEntry32W()
    entry.dwSize = ctypes.sizeof(entry)
    try:
        if kernel32.Process32FirstW(handle, ctypes.byref(entry)):
            while True:
                table[int(entry.th32ProcessID)] = int(
                    entry.th32ParentProcessID
                )
                if not kernel32.Process32NextW(handle, ctypes.byref(entry)):
                    break
    finally:
        kernel32.CloseHandle(handle)
    return table


def _rss_bytes_windows(pid: int) -> int | None:
    process_query_limited_information = 0x1000
    process_vm_read = 0x0010

    class ProcessMemoryCountersEx(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.OpenProcess.argtypes = [
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
    ]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCountersEx),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(
        process_query_limited_information | process_vm_read,
        False,
        int(pid),
    )
    if not handle:
        return None
    counters = ProcessMemoryCountersEx()
    counters.cb = ctypes.sizeof(counters)
    try:
        if not psapi.GetProcessMemoryInfo(
            handle,
            ctypes.byref(counters),
            counters.cb,
        ):
            return None
        return int(counters.WorkingSetSize)
    finally:
        kernel32.CloseHandle(handle)


def _linux_process_table() -> dict[int, int]:
    table = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / "stat").read_text(encoding="utf-8").split()
            table[int(path.name)] = int(fields[3])
        except (OSError, ValueError, IndexError):
            continue
    return table


def _rss_bytes_linux(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(
            encoding="utf-8"
        ).splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def process_tree_rss_snapshot(root_pid: int) -> dict[str, Any]:
    if os.name == "nt":
        table = _process_table_windows()
        rss_reader = _rss_bytes_windows
    elif sys.platform.startswith("linux"):
        table = _linux_process_table()
        rss_reader = _rss_bytes_linux
    else:
        raise Gate12C2ThroughputError(
            "process-tree RSS monitoring is unsupported on this platform"
        )
    descendants = {int(root_pid)}
    changed = True
    while changed:
        changed = False
        for pid, parent in table.items():
            if parent in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    rss_by_pid = {}
    for pid in sorted(descendants):
        rss = rss_reader(pid)
        if rss is not None:
            rss_by_pid[str(pid)] = int(rss)
    return {
        "root_pid": int(root_pid),
        "observed_pid_count": len(rss_by_pid),
        "rss_bytes": sum(rss_by_pid.values()),
        "rss_bytes_by_pid": rss_by_pid,
    }


class ProcessTreeRssMonitor:
    def __init__(
        self,
        root_pid: int,
        *,
        interval_seconds: float = PROFILE_MEMORY_SAMPLE_INTERVAL_SECONDS,
    ) -> None:
        self.root_pid = int(root_pid)
        self.interval_seconds = float(interval_seconds)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_rss_bytes = 0
        self.peak_process_count = 0
        self.sample_count = 0
        self.monitor_error: str | None = None

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            try:
                row = process_tree_rss_snapshot(self.root_pid)
                self.peak_rss_bytes = max(
                    self.peak_rss_bytes,
                    int(row["rss_bytes"]),
                )
                self.peak_process_count = max(
                    self.peak_process_count,
                    int(row["observed_pid_count"]),
                )
                self.sample_count += 1
            except Exception as exc:  # pragma: no cover - platform failure
                self.monitor_error = f"{type(exc).__name__}: {exc}"
                return
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="gate12c2-process-tree-rss",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 4))
        return {
            "sample_interval_seconds": self.interval_seconds,
            "sample_count": self.sample_count,
            "peak_process_tree_rss_bytes": self.peak_rss_bytes,
            "peak_observed_process_count": self.peak_process_count,
            "monitor_error": self.monitor_error,
        }


def _hardware_receipt() -> dict[str, Any]:
    result: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
    }
    if os.name == "nt":
        command = (
            "$cpu=Get-CimInstance Win32_Processor|Select-Object -First 1 "
            "Name,NumberOfCores,NumberOfLogicalProcessors;"
            "$mem=Get-CimInstance Win32_ComputerSystem|"
            "Select-Object TotalPhysicalMemory;"
            "[pscustomobject]@{CPU=$cpu.Name;"
            "PhysicalCores=$cpu.NumberOfCores;"
            "LogicalCores=$cpu.NumberOfLogicalProcessors;"
            "RAMBytes=$mem.TotalPhysicalMemory}|ConvertTo-Json -Compress"
        )
        try:
            completed = subprocess.run(
                ["powershell", "-NoProfile", "-Command", command],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            result["windows_cim"] = json.loads(completed.stdout)
        except Exception as exc:
            result["windows_cim_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _directory_bytes(path: Path) -> int:
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file()
    )


def _configuration_command(
    configuration: Mapping[str, Any],
    *,
    output_dir: Path,
    plan_path: Path,
    preflight_path: Path,
    authorization_path: Path,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).with_name(
            "run_gate12c2_development_shards.py"
        )),
        "--plan",
        str(plan_path),
        "--preflight-receipt",
        str(preflight_path),
        "--authorization-receipt",
        str(authorization_path),
        "--workers",
        str(configuration["worker_count"]),
        "--output-dir",
        str(output_dir),
    ]


def run_profile_configuration(
    configuration: Mapping[str, Any],
    *,
    output_root: Path,
    profile_authorization: Mapping[str, Any],
) -> dict[str, Any]:
    configuration_id = str(configuration["configuration_id"])
    root = Path(output_root).resolve()
    output_dir = root / configuration_id
    if output_dir.exists() and any(output_dir.iterdir()):
        raise Gate12C2ThroughputError(
            "throughput configurations require a fresh output directory: "
            f"{output_dir}"
        )
    root.mkdir(parents=True, exist_ok=True)
    control_dir = root / ".profile-control" / configuration_id
    if control_dir.exists():
        raise Gate12C2ThroughputError(
            "profile control directory already exists: "
            f"{control_dir}"
        )
    control_dir.mkdir(parents=True, exist_ok=False)
    expected_plan = shards.build_development_shard_plan(
        regime_id=str(configuration["regime_id"]),
        master_seed=str(configuration["master_seed"]),
        outer_experiment_indices=[
            int(value)
            for value in configuration["outer_experiment_indices"]
        ],
        block_count=lab.reference_block_count_schedule(),
        inner_valid_draw_count=int(
            configuration["inner_valid_draw_count"]
        ),
        effect_strength=configuration.get("effect_strength"),
    )
    checks = {key: True for key in shards.REQUIRED_PREFLIGHT_CHECKS}
    parent_authorization_sha256 = str(
        profile_authorization[
            "authorization_receipt_payload_sha256"
        ]
    )
    preflight = shards.build_no_outcome_preflight_receipt(
        expected_plan,
        output_dir=output_dir,
        worker_count=int(configuration["worker_count"]),
        preflight_id=(
            f"profile-derived::{parent_authorization_sha256}::"
            f"{configuration_id}"
        ),
        checks=checks,
    )
    authorization = shards.build_development_execution_authorization(
        expected_plan,
        preflight,
        output_dir=output_dir,
        worker_count=int(configuration["worker_count"]),
        authorization_id=(
            f"profile-derived::{parent_authorization_sha256}::"
            f"{configuration_id}"
        ),
        purpose=f"bounded-throughput-profile::{configuration_id}",
    )
    plan_path = control_dir / "plan.json"
    preflight_path = control_dir / "preflight.json"
    authorization_path = control_dir / "authorization.json"
    shards._atomic_write(
        plan_path,
        shards._canonical_json_bytes(expected_plan),
    )
    shards._atomic_write(
        preflight_path,
        shards._canonical_json_bytes(preflight),
    )
    shards._atomic_write(
        authorization_path,
        shards._canonical_json_bytes(authorization),
    )
    disk_before = shutil.disk_usage(root)
    command = _configuration_command(
        configuration,
        output_dir=output_dir,
        plan_path=plan_path,
        preflight_path=preflight_path,
        authorization_path=authorization_path,
    )
    environment = os.environ.copy()
    environment.update(shards.SINGLE_THREAD_ENVIRONMENT)
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=str(Path(__file__).resolve().parents[1]),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=(
            subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        ),
    )
    monitor = ProcessTreeRssMonitor(process.pid)
    monitor.start()
    stdout, stderr = process.communicate()
    memory = monitor.stop()
    wall_seconds = time.perf_counter() - started
    if memory["monitor_error"] is not None:
        raise Gate12C2ThroughputError(
            f"process-tree memory monitor failed: {memory['monitor_error']}"
        )
    if process.returncode != 0:
        raise Gate12C2ThroughputError(
            f"profile configuration {configuration_id} failed with "
            f"exit {process.returncode}: {stderr[-4000:]}"
        )
    try:
        cli_summary = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise Gate12C2ThroughputError(
            f"profile configuration returned invalid JSON: {exc}"
        ) from exc
    plan = json.loads(
        (output_dir / "plan.json").read_text(encoding="utf-8")
    )
    if plan != expected_plan:
        raise Gate12C2ThroughputError(
            "executed shard plan differs from the frozen profile configuration"
        )
    index = json.loads(
        (output_dir / "index.json").read_text(encoding="utf-8")
    )
    if int(index["worker_count_operational_only"]) != int(
        configuration["worker_count"]
    ):
        raise Gate12C2ThroughputError(
            "executed worker count differs from the profile configuration"
        )
    verification = shards.verify_development_shard_index(
        plan,
        output_dir=output_dir,
    )
    operational_rows = [
        dict(row["operational_metrics"]) for row in index["shards"]
    ]
    if any(row.get("mode") != "execute_new" for row in operational_rows):
        raise Gate12C2ThroughputError(
            "throughput configuration reused an existing shard"
        )
    attempts = sum(
        int(row["endpoint_draw_attempts"]) for row in operational_rows
    )
    acceptances = sum(
        int(row["endpoint_draw_acceptances"]) for row in operational_rows
    )
    rejection_counts: dict[str, int] = defaultdict(int)
    for row in operational_rows:
        for reason, count in row["rejection_reason_counts"].items():
            rejection_counts[str(reason)] += int(count)
    disk_after = shutil.disk_usage(root)
    return {
        "schema_version": PROFILE_CONFIGURATION_SCHEMA_VERSION,
        "configuration_id": configuration_id,
        "profile_slice": str(configuration["profile_slice"]),
        "workload_id": str(configuration["workload_id"]),
        "regime_id": str(configuration["regime_id"]),
        "worker_count": int(configuration["worker_count"]),
        "outer_experiment_count": len(
            configuration["outer_experiment_indices"]
        ),
        "inner_valid_draw_count": int(
            configuration["inner_valid_draw_count"]
        ),
        "wall_seconds": wall_seconds,
        "process_tree_memory": memory,
        "endpoint_draw_attempts": attempts,
        "endpoint_draw_acceptances": acceptances,
        "attempts_per_accepted_draw": (
            attempts / acceptances if acceptances > 0 else None
        ),
        "effective_accepted_draws_per_wall_second": (
            acceptances / wall_seconds if wall_seconds > 0.0 else None
        ),
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "unaccounted_rejection_count": sum(
            int(row["unaccounted_rejection_count"])
            for row in operational_rows
        ),
        "exhausted_incomplete_stream_count": sum(
            int(row["exhausted_incomplete_stream_count"])
            for row in operational_rows
        ),
        "sum_outer_compute_wall_seconds": sum(
            float(row["compute_wall_seconds"]) for row in operational_rows
        ),
        "sum_outer_process_cpu_seconds": sum(
            float(row["compute_cpu_seconds"]) for row in operational_rows
        ),
        "sum_serialization_write_wall_seconds": sum(
            float(row["serialization_write_wall_seconds"])
            for row in operational_rows
        ),
        "shard_phase_wall_seconds": float(
            index["operational_execution_metrics"][
                "shard_phase_wall_seconds"
            ]
        ),
        "merge_validation_before_write_wall_seconds": float(
            index["operational_execution_metrics"][
                "merge_validation_before_write_wall_seconds"
            ]
        ),
        "output_bytes": _directory_bytes(output_dir),
        "compressed_shard_bytes": sum(
            int(row["compressed_bytes"]) for row in index["shards"]
        ),
        "disk_free_bytes_before": int(disk_before.free),
        "disk_free_bytes_after": int(disk_after.free),
        "plan_payload_sha256": str(
            cli_summary["plan_payload_sha256"]
        ),
        "scientific_projection_sha256": str(
            verification["scientific_projection_sha256"]
        ),
        "index_payload_sha256": str(
            verification["index_payload_sha256"]
        ),
        "stdout_payload_sha256": _sha256_bytes(stdout.encode("utf-8")),
        "stderr_payload_sha256": _sha256_bytes(stderr.encode("utf-8")),
        "derived_preflight_receipt_payload_sha256": preflight[
            "preflight_receipt_payload_sha256"
        ],
        "derived_authorization_receipt_payload_sha256": authorization[
            "authorization_receipt_payload_sha256"
        ],
        "parent_profile_authorization_receipt_payload_sha256": (
            parent_authorization_sha256
        ),
        "scientific_outcomes_exposed_in_profile_receipt": False,
    }


def summarize_profile_results(
    results: Sequence[Mapping[str, Any]],
    *,
    physical_ram_bytes: int | None,
) -> dict[str, Any]:
    rows = [dict(row) for row in results]
    by_workload: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_workload[str(row["workload_id"])].append(row)
    workload_rows = []
    determinism_pass = True
    memory_pass = True
    for workload_id, workload in sorted(by_workload.items()):
        ordered = sorted(workload, key=lambda row: int(row["worker_count"]))
        projection_hashes = {
            str(row["scientific_projection_sha256"]) for row in ordered
        }
        workload_determinism = len(projection_hashes) == 1
        determinism_pass &= workload_determinism
        scaling = []
        baseline_throughput = float(
            ordered[0]["effective_accepted_draws_per_wall_second"]
        )
        previous_throughput = None
        for row in ordered:
            throughput = float(
                row["effective_accepted_draws_per_wall_second"]
            )
            peak = int(
                row["process_tree_memory"][
                    "peak_process_tree_rss_bytes"
                ]
            )
            fraction = (
                peak / physical_ram_bytes
                if physical_ram_bytes is not None
                and physical_ram_bytes > 0
                else None
            )
            row_memory_pass = fraction is None or fraction <= 0.75
            memory_pass &= row_memory_pass
            scaling.append(
                {
                    "worker_count": int(row["worker_count"]),
                    "wall_seconds": float(row["wall_seconds"]),
                    "effective_accepted_draws_per_wall_second": throughput,
                    "speedup_vs_smallest_worker_count": (
                        throughput / baseline_throughput
                    ),
                    "parallel_efficiency_vs_smallest_worker_count": (
                        throughput
                        / baseline_throughput
                        * int(ordered[0]["worker_count"])
                        / int(row["worker_count"])
                    ),
                    "marginal_throughput_improvement_vs_previous": (
                        None
                        if previous_throughput is None
                        else throughput / previous_throughput - 1.0
                    ),
                    "peak_process_tree_rss_bytes": peak,
                    "peak_rss_fraction_of_physical_ram": fraction,
                    "memory_gate_pass": row_memory_pass,
                    "output_bytes": int(row["output_bytes"]),
                }
            )
            previous_throughput = throughput
        workload_rows.append(
            {
                "workload_id": workload_id,
                "regime_id": str(ordered[0]["regime_id"]),
                "scientific_projection_determinism_pass": (
                    workload_determinism
                ),
                "scientific_projection_sha256": (
                    next(iter(projection_hashes))
                    if workload_determinism
                    else None
                ),
                "scaling": scaling,
            }
        )
    return {
        "determinism_pass": determinism_pass,
        "memory_gate_pass": memory_pass,
        "workloads": workload_rows,
        "scientific_outcomes_interpreted": False,
    }


def execute_profile_plan(
    plan: Mapping[str, Any],
    *,
    output_root: Path,
    preflight_receipt: Mapping[str, Any] | None = None,
    authorization_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    verified = verify_profile_plan(plan)
    destination = Path(output_root).resolve()
    if preflight_receipt is None or authorization_receipt is None:
        raise Gate12C2ThroughputError(
            "profile execution requires an exact no-outcome preflight and "
            "an explicit profile-bound authorization"
        )
    authorization = _verified_profile_authorization(
        verified,
        preflight_receipt,
        authorization_receipt,
        output_root=destination,
    )
    if destination.exists() and any(destination.iterdir()):
        raise Gate12C2ThroughputError(
            "profile output root must be fresh and empty"
        )
    destination.mkdir(parents=True, exist_ok=True)
    hardware = _hardware_receipt()
    physical_ram = None
    windows_cim = hardware.get("windows_cim")
    if isinstance(windows_cim, Mapping):
        physical_ram = int(windows_cim["RAMBytes"])
    started = time.perf_counter()
    results = [
        run_profile_configuration(
            configuration,
            output_root=destination,
            profile_authorization=authorization,
        )
        for configuration in verified["configurations"]
    ]
    summary = summarize_profile_results(
        results,
        physical_ram_bytes=physical_ram,
    )
    receipt: dict[str, Any] = {
        "schema_version": PROFILE_RECEIPT_SCHEMA_VERSION,
        "profile_id": verified["profile_id"],
        "epistemic_status": "development_throughput_only",
        "surface_id": "development",
        "locked_execution_authorized": False,
        "real_held_out_execution_authorized": False,
        "scientific_calibration_result": None,
        "profile_plan_payload_sha256": verified[
            "profile_plan_payload_sha256"
        ],
        "preflight_receipt_payload_sha256": preflight_receipt[
            "preflight_receipt_payload_sha256"
        ],
        "authorization_receipt_payload_sha256": authorization[
            "authorization_receipt_payload_sha256"
        ],
        "source_commit": verified["source_commit"],
        "implementation_sha256": verified["implementation_sha256"],
        "hardware": hardware,
        "thread_environment": verified["thread_environment"],
        "profile_wall_seconds": time.perf_counter() - started,
        "configuration_results": results,
        "summary": summary,
        "next_authorization": (
            "none; draw-scaling slice requires a separate operational "
            "instantiation after this receipt is reviewed"
        ),
        "N2_open": False,
        "N3_open": False,
    }
    receipt["profile_receipt_payload_sha256"] = _sha256_bytes(
        _canonical_json_bytes(receipt)
    )
    return receipt
