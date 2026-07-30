#!/usr/bin/env python3
from __future__ import annotations

import ctypes
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import gate12c2_resource_qualification as resource


class _KernelFunction:
    def __init__(self, callback):
        self.callback = callback
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.callback(*args)


class _FakeKernel32:
    REQUIRED_FLAGS = (
        resource.WindowsJobApi.JOB_OBJECT_LIMIT_JOB_MEMORY
        | resource.WindowsJobApi.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    )

    def __init__(
        self,
        *,
        expected_limit: int = 12_000_000,
        limit_flags: int | None = None,
        reported_limit: int | None = None,
        fail_watchdog_close_attempts: int = 0,
        noop_watchdog_close: bool = False,
        noop_source_close: bool = False,
    ) -> None:
        self.expected_limit = expected_limit
        self.limit_flags = (
            self.REQUIRED_FLAGS if limit_flags is None else limit_flags
        )
        self.reported_limit = (
            expected_limit if reported_limit is None else reported_limit
        )
        self.fail_watchdog_close_attempts = fail_watchdog_close_attempts
        self.noop_watchdog_close = noop_watchdog_close
        self.noop_source_close = noop_source_close
        self.open_handles = {11}
        self.inheritable_handles: set[int] = set()
        self.close_calls: list[int] = []
        self.watchdog_close_calls = 0
        self.GetCurrentProcess = _KernelFunction(lambda: 999)
        self.DuplicateHandle = _KernelFunction(self._duplicate)
        self.CloseHandle = _KernelFunction(self._close)
        self.GetHandleInformation = _KernelFunction(self._handle_information)
        self.QueryInformationJobObject = _KernelFunction(self._query_job)

    def _duplicate(self, *args):
        source_handle = int(args[1])
        duplicate = args[3]
        options = int(args[6])
        if options & resource.WindowsJobApi.DUPLICATE_CLOSE_SOURCE:
            self.open_handles.discard(source_handle)
            duplicate._obj.value = 124
            self.open_handles.add(124)
        else:
            duplicate._obj.value = 123
            self.open_handles.add(123)
        return True

    def _close(self, handle):
        raw = int(handle)
        self.close_calls.append(raw)
        if raw == 11 and self.noop_source_close:
            return True
        if raw == 123:
            self.watchdog_close_calls += 1
            if self.watchdog_close_calls <= self.fail_watchdog_close_attempts:
                return False
            if self.noop_watchdog_close:
                return True
        if raw not in self.open_handles:
            if hasattr(ctypes, "set_last_error"):
                ctypes.set_last_error(6)
            return False
        self.open_handles.remove(raw)
        return True

    def _handle_information(self, handle, flags):
        raw = int(handle)
        if raw not in self.open_handles:
            if hasattr(ctypes, "set_last_error"):
                ctypes.set_last_error(6)
            return False
        flags._obj.value = (
            resource.WindowsJobApi.HANDLE_FLAG_INHERIT
            if raw in self.inheritable_handles
            else 0
        )
        return True

    def _query_job(self, handle, info_class, info, size, returned):
        del handle, info_class, size
        info._obj.BasicLimitInformation.LimitFlags = self.limit_flags
        info._obj.JobMemoryLimit = self.reported_limit
        returned._obj.value = 1
        return True


def _job_api(kernel: _FakeKernel32) -> resource.WindowsJobApi:
    api = object.__new__(resource.WindowsJobApi)
    api.kernel32 = kernel
    return api


def _make_job_handle(
    *,
    fail_close_attempts: int = 0,
    noop_close: bool = False,
) -> resource.WatchdogOwnedWindowsJobHandle:
    kernel = _FakeKernel32(
        fail_watchdog_close_attempts=fail_close_attempts,
        noop_watchdog_close=noop_close,
    )
    api = _job_api(kernel)
    receipt = api.transfer_job_handle_to_watchdog(
        source_job_handle=11,
        target_process_handle=22,
        source_identity=resource.ProcessIdentity(10, 100),
        watchdog_identity=resource.ProcessIdentity(20, 200),
        expected_job_memory_limit_bytes=kernel.expected_limit,
    )
    handle = resource.WatchdogOwnedWindowsJobHandle.from_transfer_receipt(
        receipt,
        api=api,
        current_identity=resource.ProcessIdentity(20, 200),
    )
    handle._test_kernel = kernel
    return handle


def _JobHandle(
    *, fail_close_attempts: int = 0, noop_close: bool = False
) -> resource.WatchdogOwnedWindowsJobHandle:
    return _make_job_handle(
        fail_close_attempts=fail_close_attempts,
        noop_close=noop_close,
    )


def _metrics() -> dict[str, int]:
    return {
        "watchdog_pid": 101,
        "watchdog_creation_time_ns": 1_000,
        "guardian_pid": 102,
        "guardian_creation_time_ns": 1_001,
        "coordinator_pid": 103,
        "coordinator_creation_time_ns": 1_002,
        "replay_root_pid": 104,
        "replay_root_creation_time_ns": 1_003,
        "job_active_process_count": 4,
        "job_total_process_count": 4,
        "job_terminated_process_count": 0,
        "job_current_memory_bytes": 1_000_000,
        "job_peak_memory_bytes": 2_000_000,
        "sampled_replay_job_rss_bytes": 1_500_000,
        "sampled_control_plane_rss_bytes": 500_000,
        "sampled_combined_rss_bytes": 2_000_000,
        "available_physical_memory_bytes": 8_000_000_000,
        "total_physical_memory_bytes": 16_000_000_000,
        "qualification_volume_free_bytes": 20_000_000_000,
        "scheduled_output_file_count": 1,
        "scheduled_output_bytes": 1_000,
        "partial_or_temp_count": 0,
    }


def _utc(index: int) -> str:
    return f"2026-07-30T00:00:{index:02d}.000000Z"


def _raw_success_stream(*, gap_after_first_ns: int = 1) -> bytes:
    previous_state = "__START__"
    previous_digest = resource.GENESIS_DIGEST
    monotonic = 0
    encoded_records: list[bytes] = []
    for sequence, event_code in enumerate(resource.SUCCESS_MILESTONES):
        state = resource.TRANSITIONS[(previous_state, event_code)]
        record = {
            "schema_version": resource.TELEMETRY_SCHEMA,
            "sequence": sequence,
            "utc_time": _utc(sequence),
            "monotonic_ns": monotonic,
            "previous_record_sha256": previous_digest,
            "state": state,
            "event_code": event_code,
            **_metrics(),
        }
        encoded, digest = resource.encode_telemetry_record(record)
        encoded_records.append(encoded)
        previous_state = state
        previous_digest = digest
        monotonic += (
            gap_after_first_ns
            if sequence == 0
            else 1
        )
    return b"".join(encoded_records)


def _rehash_wire_record(record: dict[str, object]) -> bytes:
    candidate = dict(record)
    candidate.pop("sha", None)
    candidate["sha"] = resource.sha256_bytes(
        resource.canonical_json_bytes(candidate)
    )
    return resource.canonical_json_bytes(candidate) + b"\n"


def _legacy_evidence() -> dict[str, object]:
    return {
        "child_exit_code": 1,
        "stdout_bytes": 0,
        "exception_type": resource.EXPECTED_LEGACY_EXCEPTION_TYPE,
        "exception_message": resource.EXPECTED_LEGACY_EXCEPTION_MESSAGE,
        "normalized_project_stack": [
            {"path": path, "line": line, "function": function}
            for path, line, function in resource.EXPECTED_LEGACY_STACK
        ],
        "stderr_sha256": resource.EXPECTED_LEGACY_STDERR_SHA256,
        "configuration_count": 9,
        "index_count": 9,
        "outer_experiment_count": 768,
        "shard_count": 768,
        "partial_or_temp_count": 0,
        "stale_lock_count": 1,
        "stale_lock_relative_path": (
            resource.EXPECTED_STALE_LOCK_RELATIVE_PATH
        ),
        "stale_lock_file_sha256": (
            resource.EXPECTED_STALE_LOCK_FILE_SHA256
        ),
        "stale_lock_manifest_match": True,
        "unexpected_artifact_count": 0,
        "unexpected_artifact_relative_paths": [],
        "legacy_execution_evidence_present": False,
        "legacy_resource_receipt_present": False,
        "legacy_execution_receipt_present": False,
        "semantic_commitments_match": True,
        "telemetry_tail_complete": True,
    }


class WatchdogAdversarialTest(unittest.TestCase):
    def test_blocking_supplier_is_killed_at_deadline(self) -> None:
        release = threading.Event()
        handle = _JobHandle()
        with mock.patch.object(
            resource, "LAUNCH_EVIDENCE_DEADLINE_NS", 20_000_000
        ):
            watchdog = resource.LaunchDeadlineWatchdog(handle)
            watchdog.resume_and_arm(lambda: 1)
            started = time.monotonic()
            with self.assertRaisesRegex(
                resource.Gate12C2ResourceQualificationError,
                "missing at deadline",
            ):
                watchdog.run_until_verified(
                    acknowledgement_supplier=lambda: release.wait(5),
                    verifier=lambda _: True,
                    poll_seconds=0.001,
                )
            elapsed = time.monotonic() - started
        release.set()
        self.assertLess(elapsed, 0.5)
        self.assertEqual(handle._test_kernel.watchdog_close_calls, 1)
        self.assertTrue(watchdog.job_handle_close_verified)

    def test_blocking_verifier_is_killed_at_deadline(self) -> None:
        release = threading.Event()
        handle = _JobHandle()
        with mock.patch.object(
            resource, "LAUNCH_EVIDENCE_DEADLINE_NS", 20_000_000
        ):
            watchdog = resource.LaunchDeadlineWatchdog(handle)
            watchdog.resume_and_arm(lambda: 1)
            with self.assertRaisesRegex(
                resource.Gate12C2ResourceQualificationError,
                "missing at deadline",
            ):
                watchdog.run_until_verified(
                    acknowledgement_supplier=lambda: {"present": True},
                    verifier=lambda _: release.wait(5),
                    poll_seconds=0.001,
                )
        release.set()
        self.assertEqual(handle._test_kernel.watchdog_close_calls, 1)

    def test_delayed_acknowledgement_is_killed(self) -> None:
        handle = _JobHandle()
        with mock.patch.object(
            resource, "LAUNCH_EVIDENCE_DEADLINE_NS", 10_000_000
        ):
            watchdog = resource.LaunchDeadlineWatchdog(handle)
            watchdog.resume_and_arm(lambda: 1)
            with self.assertRaises(
                resource.Gate12C2ResourceQualificationError
            ):
                watchdog.run_until_verified(
                    acknowledgement_supplier=lambda: (
                        time.sleep(0.03),
                        {"present": True},
                    )[1],
                    verifier=lambda _: True,
                    poll_seconds=0.001,
                )
        self.assertEqual(handle._test_kernel.watchdog_close_calls, 1)

    def test_clock_failure_closes_job(self) -> None:
        calls = 0

        def clock() -> int:
            nonlocal calls
            calls += 1
            if calls == 1:
                return 1
            raise RuntimeError("RAW_CLOCK_SENTINEL")

        handle = _JobHandle()
        watchdog = resource.LaunchDeadlineWatchdog(
            handle, monotonic_ns=clock
        )
        watchdog.resume_and_arm(lambda: 1)
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "^monotonic clock failed$",
        ) as raised:
            watchdog.run_until_verified(
                acknowledgement_supplier=lambda: None,
                verifier=lambda _: True,
            )
        self.assertNotIn("RAW_CLOCK_SENTINEL", str(raised.exception))
        self.assertEqual(handle._test_kernel.watchdog_close_calls, 1)

    def test_close_failure_retries_and_never_claims_verified_close(self) -> None:
        recoverable = _JobHandle(fail_close_attempts=2)
        watchdog = resource.LaunchDeadlineWatchdog(recoverable)
        watchdog.resume_and_arm(lambda: 1)
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "^launch acknowledgement is invalid$",
        ):
            watchdog.run_until_verified(
                acknowledgement_supplier=lambda: {"present": True},
                verifier=lambda _: False,
            )
        self.assertEqual(recoverable._test_kernel.watchdog_close_calls, 3)
        self.assertTrue(watchdog.job_handle_close_verified)

        unrecoverable = _JobHandle(fail_close_attempts=99)
        watchdog = resource.LaunchDeadlineWatchdog(unrecoverable)
        watchdog.resume_and_arm(lambda: 1)
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "Job handle close failed",
        ):
            watchdog.verify_acknowledgement(None, lambda _: True)
        self.assertEqual(unrecoverable._test_kernel.watchdog_close_calls, 3)
        self.assertFalse(watchdog.job_handle_close_verified)
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "Job handle close failed",
        ):
            watchdog.verify_acknowledgement(None, lambda _: True)
        self.assertEqual(unrecoverable._test_kernel.watchdog_close_calls, 6)


class JobOwnershipAdversarialTest(unittest.TestCase):
    @staticmethod
    def _receipt(
        kernel: _FakeKernel32 | None = None,
    ) -> tuple[object, resource.WindowsJobApi, _FakeKernel32]:
        selected = kernel or _FakeKernel32()
        api = _job_api(selected)
        receipt = api.transfer_job_handle_to_watchdog(
            source_job_handle=11,
            target_process_handle=22,
            source_identity=resource.ProcessIdentity(10, 100),
            watchdog_identity=resource.ProcessIdentity(20, 200),
            expected_job_memory_limit_bytes=selected.expected_limit,
        )
        return receipt, api, selected

    def test_transfer_closes_source_and_claim_reverifies_exact_os_handle(
        self,
    ) -> None:
        receipt, api, kernel = self._receipt()
        self.assertNotIn(11, kernel.open_handles)
        self.assertIn(123, kernel.open_handles)
        handle = resource.WatchdogOwnedWindowsJobHandle.from_transfer_receipt(
            receipt,
            api=api,
            current_identity=resource.ProcessIdentity(20, 200),
        )
        self.assertTrue(handle.sole_owner_verified)
        resource.LaunchDeadlineWatchdog(handle, monotonic_ns=lambda: 0)

    def test_public_or_token_style_self_attestation_is_rejected(self) -> None:
        self.assertFalse(
            hasattr(resource, "JobHandleOwnershipProof")
        )
        self.assertFalse(
            hasattr(resource, "_JOB_TRANSFER_RECEIPT_TOKEN")
        )
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "direct Job transfer-receipt construction",
        ):
            resource._JobHandleTransferReceipt()
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "direct watchdog Job-handle construction",
        ):
            resource.WatchdogOwnedWindowsJobHandle()

        self.assertFalse(
            hasattr(
                resource._JobHandleTransferReceipt,
                "_issue_after_verified_os_transfer",
            )
        )

    def test_subclassed_or_monkeypatched_job_api_is_rejected(self) -> None:
        receipt, _, _ = self._receipt()

        class Subclass(resource.WindowsJobApi):
            def __init__(self) -> None:
                self.kernel32 = _FakeKernel32()

        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "reviewed Win32 adapter",
        ):
            resource.WatchdogOwnedWindowsJobHandle.from_transfer_receipt(
                receipt,
                api=Subclass(),
                current_identity=resource.ProcessIdentity(20, 200),
            )

        exact = _job_api(_FakeKernel32())
        exact.verify_job_handle = lambda handle, limit: True
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "method identity",
        ):
            resource.WatchdogOwnedWindowsJobHandle.from_transfer_receipt(
                receipt,
                api=exact,
                current_identity=resource.ProcessIdentity(20, 200),
            )

    def test_rejected_target_identity_inheritance_or_limits_is_cleaned(
        self,
    ) -> None:
        def wrong_identity(kernel: _FakeKernel32) -> resource.ProcessIdentity:
            return resource.ProcessIdentity(21, 200)

        def inheritable(kernel: _FakeKernel32) -> resource.ProcessIdentity:
            kernel.inheritable_handles.add(123)
            return resource.ProcessIdentity(20, 200)

        def wrong_limit(kernel: _FakeKernel32) -> resource.ProcessIdentity:
            kernel.reported_limit -= 1
            return resource.ProcessIdentity(20, 200)

        for mutate in (wrong_identity, inheritable, wrong_limit):
            receipt, api, kernel = self._receipt()
            identity = mutate(kernel)
            with self.subTest(mutate=mutate.__name__):
                with self.assertRaises(
                    resource.Gate12C2ResourceQualificationError
                ):
                    resource.WatchdogOwnedWindowsJobHandle.from_transfer_receipt(
                        receipt,
                        api=api,
                        current_identity=identity,
                    )
                self.assertNotIn(123, kernel.open_handles)

    def test_unverified_source_close_cleans_target_duplicate(self) -> None:
        kernel = _FakeKernel32(noop_source_close=True)
        api = _job_api(kernel)
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "source Job handle close failed",
        ):
            api.transfer_job_handle_to_watchdog(
                source_job_handle=11,
                target_process_handle=22,
                source_identity=resource.ProcessIdentity(10, 100),
                watchdog_identity=resource.ProcessIdentity(20, 200),
                expected_job_memory_limit_bytes=kernel.expected_limit,
            )
        self.assertIn(11, kernel.open_handles)
        self.assertNotIn(123, kernel.open_handles)
        self.assertNotIn(124, kernel.open_handles)

    def test_source_wrong_limits_reject_before_duplicate(self) -> None:
        required = _FakeKernel32.REQUIRED_FLAGS
        for kernel in (
            _FakeKernel32(limit_flags=required | 1),
            _FakeKernel32(reported_limit=1),
        ):
            api = _job_api(kernel)
            with self.subTest(
                flags=kernel.limit_flags, limit=kernel.reported_limit
            ):
                with self.assertRaisesRegex(
                    resource.Gate12C2ResourceQualificationError,
                    "exact frozen limits",
                ):
                    api.transfer_job_handle_to_watchdog(
                        source_job_handle=11,
                        target_process_handle=22,
                        source_identity=resource.ProcessIdentity(10, 100),
                        watchdog_identity=resource.ProcessIdentity(20, 200),
                        expected_job_memory_limit_bytes=kernel.expected_limit,
                    )
                self.assertNotIn(123, kernel.open_handles)

    def test_noop_close_cannot_be_verified_as_job_kill(self) -> None:
        handle = _JobHandle(noop_close=True)
        watchdog = resource.LaunchDeadlineWatchdog(
            handle,
            monotonic_ns=lambda: 10,
        )
        watchdog.resume_and_arm(lambda: 1)
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "Job handle close failed",
        ):
            watchdog.verify_acknowledgement(None, lambda _: True)
        self.assertIn(123, handle._test_kernel.open_handles)
        self.assertEqual(
            handle._test_kernel.watchdog_close_calls,
            resource.JOB_HANDLE_CLOSE_ATTEMPTS,
        )


class TelemetryAdversarialTest(unittest.TestCase):
    def test_decoder_and_writer_reject_live_gap_over_one_second(self) -> None:
        payload = _raw_success_stream(
            gap_after_first_ns=resource.MAXIMUM_LIVE_GAP_NS + 1
        )
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "maximum live gap",
        ):
            resource.decode_and_verify_telemetry(payload)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "telemetry.jsonl"
            writer = resource.AppendOnlyTelemetryWriter(path)
            writer.append(
                event_code=resource.SUCCESS_MILESTONES[0],
                utc_time=_utc(0),
                monotonic_ns=0,
                metrics=_metrics(),
            )
            with self.assertRaisesRegex(
                resource.Gate12C2ResourceQualificationError,
                "maximum live gap",
            ):
                writer.append(
                    event_code=resource.SUCCESS_MILESTONES[1],
                    utc_time=_utc(1),
                    monotonic_ns=resource.MAXIMUM_LIVE_GAP_NS + 1,
                    metrics=_metrics(),
                )
            writer.append(
                event_code="FAILURE_DETECTED",
                utc_time=_utc(2),
                monotonic_ns=1,
                metrics=_metrics(),
            )
            writer.close()

    def test_partial_verification_cannot_return_pass(self) -> None:
        first = _raw_success_stream().splitlines(keepends=True)[0]
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "partial telemetry verification is forbidden",
        ):
            resource.decode_and_verify_telemetry(
                first, require_terminal=False
            )
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "terminal",
        ):
            resource.decode_and_verify_telemetry(first)

    @unittest.skipUnless(os.name == "nt", "Windows share-mode assertion")
    def test_live_file_denies_a_second_raw_writer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "telemetry.jsonl"
            writer = resource.AppendOnlyTelemetryWriter(path)
            with self.assertRaises(OSError):
                path.open("ab")
            writer.append(
                event_code="WRAPPER_AUTHORIZATION_CONSUMED",
                utc_time=_utc(0),
                monotonic_ns=0,
                metrics=_metrics(),
            )
            writer.append(
                event_code="FAILURE_DETECTED",
                utc_time=_utc(1),
                monotonic_ns=1,
                metrics=_metrics(),
            )
            writer.close()

    def test_fsync_failure_is_sticky_and_poisoned_bytes_do_not_pass(
        self,
    ) -> None:
        def failing_fsync(_: int) -> None:
            raise OSError("RAW_FSYNC_SENTINEL")

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "telemetry.jsonl"
            writer = resource.AppendOnlyTelemetryWriter(
                path, fsync=failing_fsync
            )
            with self.assertRaisesRegex(
                resource.Gate12C2ResourceQualificationError,
                "^telemetry append failed$",
            ) as raised:
                writer.append(
                    event_code="WRAPPER_AUTHORIZATION_CONSUMED",
                    utc_time=_utc(0),
                    monotonic_ns=0,
                    metrics=_metrics(),
                )
            self.assertNotIn("RAW_FSYNC_SENTINEL", str(raised.exception))
            with self.assertRaisesRegex(
                resource.Gate12C2ResourceQualificationError,
                "terminally failed",
            ):
                writer.append(
                    event_code="WRAPPER_AUTHORIZATION_CONSUMED",
                    utc_time=_utc(0),
                    monotonic_ns=0,
                    metrics=_metrics(),
                )
            with self.assertRaisesRegex(
                resource.Gate12C2ResourceQualificationError,
                "terminally failed",
            ):
                writer.close()
            payload = path.read_bytes()
        self.assertIn(b"\x00", payload)
        with self.assertRaises(resource.Gate12C2ResourceQualificationError):
            resource.decode_and_verify_telemetry(payload)

    def test_incomplete_close_is_poisoned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "telemetry.jsonl"
            writer = resource.AppendOnlyTelemetryWriter(path)
            writer.append(
                event_code="WRAPPER_AUTHORIZATION_CONSUMED",
                utc_time=_utc(0),
                monotonic_ns=0,
                metrics=_metrics(),
            )
            with self.assertRaisesRegex(
                resource.Gate12C2ResourceQualificationError,
                "before a terminal",
            ):
                writer.close()
            payload = path.read_bytes()
        with self.assertRaises(resource.Gate12C2ResourceQualificationError):
            resource.decode_and_verify_telemetry(payload)

    def test_terminal_close_failure_is_not_silently_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "telemetry.jsonl"
            writer = resource.AppendOnlyTelemetryWriter(path)
            writer.append(
                event_code="WRAPPER_AUTHORIZATION_CONSUMED",
                utc_time=_utc(0),
                monotonic_ns=0,
                metrics=_metrics(),
            )
            writer.append(
                event_code="FAILURE_DETECTED",
                utc_time=_utc(1),
                monotonic_ns=1,
                metrics=_metrics(),
            )
            with mock.patch.object(
                writer, "_close_handle_only", return_value=False
            ):
                with self.assertRaisesRegex(
                    resource.Gate12C2ResourceQualificationError,
                    "telemetry handle close failed",
                ):
                    writer.close()
            writer._handle.close()
            payload = path.read_bytes()
        self.assertIn(b"\x00", payload)
        with self.assertRaises(resource.Gate12C2ResourceQualificationError):
            resource.decode_and_verify_telemetry(payload)

    def test_close_exception_after_os_release_poisoned_artifact(self) -> None:
        class RaisesAfterRelease:
            def __init__(self, wrapped):
                self.wrapped = wrapped

            @property
            def closed(self):
                return self.wrapped.closed

            def __getattr__(self, name):
                return getattr(self.wrapped, name)

            def close(self):
                self.wrapped.close()
                raise OSError("RAW_CLOSE_AFTER_RELEASE_SENTINEL")

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "telemetry.jsonl"
            writer = resource.AppendOnlyTelemetryWriter(path)
            writer.append(
                event_code="WRAPPER_AUTHORIZATION_CONSUMED",
                utc_time=_utc(0),
                monotonic_ns=0,
                metrics=_metrics(),
            )
            writer.append(
                event_code="FAILURE_DETECTED",
                utc_time=_utc(1),
                monotonic_ns=1,
                metrics=_metrics(),
            )
            writer._handle = RaisesAfterRelease(writer._handle)
            with self.assertRaisesRegex(
                resource.Gate12C2ResourceQualificationError,
                "telemetry handle close failed",
            ):
                writer.close()
            payload = path.read_bytes()
        self.assertIn(b"\x00", payload)
        with self.assertRaises(resource.Gate12C2ResourceQualificationError):
            resource.decode_and_verify_telemetry(payload)

    def test_decoder_rejects_strict_type_key_and_nonfinite_attacks(
        self,
    ) -> None:
        payload = _raw_success_stream()
        lines = payload.splitlines(keepends=True)
        first = json.loads(lines[0])
        for field, value in (("seq", True), ("seq", "0")):
            attacked = dict(first)
            attacked[field] = value
            candidate = _rehash_wire_record(attacked) + b"".join(lines[1:])
            with self.subTest(field=field, value=value):
                with self.assertRaises(
                    resource.Gate12C2ResourceQualificationError
                ):
                    resource.decode_and_verify_telemetry(candidate)
        attacked = dict(first)
        attacked["unknown"] = 1
        candidate = _rehash_wire_record(attacked) + b"".join(lines[1:])
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "wire keys",
        ):
            resource.decode_and_verify_telemetry(candidate)
        attacked = dict(first)
        attacked.pop("seq")
        candidate = _rehash_wire_record(attacked) + b"".join(lines[1:])
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "wire keys",
        ):
            resource.decode_and_verify_telemetry(candidate)
        with self.assertRaises(resource.Gate12C2ResourceQualificationError):
            resource.decode_and_verify_telemetry(
                lines[0].replace(b'"seq":0', b'"seq":NaN')
                + b"".join(lines[1:])
            )
        with self.assertRaises(resource.Gate12C2ResourceQualificationError):
            resource.decode_and_verify_telemetry(
                b"\xef\xbb\xbf" + payload
            )
        with self.assertRaises(resource.Gate12C2ResourceQualificationError):
            resource.decode_and_verify_telemetry(
                b"{" + b"x" * resource.MAXIMUM_RECORD_BYTES + b"}\n"
            )


class LegacyClassifierAdversarialTest(unittest.TestCase):
    def test_stack_rows_are_closed_and_exact(self) -> None:
        evidence = _legacy_evidence()
        evidence["normalized_project_stack"][0]["unknown"] = True  # type: ignore[index]
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "row schema",
        ):
            resource.classify_expected_legacy_closeout(evidence)

    def test_stale_lock_and_unexpected_artifact_surface_is_exact(self) -> None:
        mutations = (
            ("stale_lock_relative_path", "other.lock"),
            ("stale_lock_file_sha256", "0" * 64),
            ("stale_lock_manifest_match", False),
            ("unexpected_artifact_count", 1),
            ("unexpected_artifact_relative_paths", ["unexpected.json"]),
        )
        for field, value in mutations:
            evidence = _legacy_evidence()
            evidence[field] = value
            with self.subTest(field=field):
                with self.assertRaises(
                    resource.Gate12C2ResourceQualificationError
                ):
                    resource.classify_expected_legacy_closeout(evidence)


class ResourceEnvelopeBoundaryTest(unittest.TestCase):
    @staticmethod
    def _exact_boundary() -> dict[str, int]:
        physical = 16 * 1024**3
        preflight = resource.MINIMUM_PREFLIGHT_FREE_BYTES
        return {
            "physical_ram_bytes": physical,
            "peak_job_memory_bytes": (3 * physical) // 4,
            "sampled_combined_rss_bytes": (3 * physical) // 4,
            "sampled_available_physical_memory_bytes": (physical + 9) // 10,
            "preflight_free_bytes": preflight,
            "minimum_observed_free_bytes": preflight // 2,
            "qualification_output_bytes": (
                resource.QUALIFICATION_OUTPUT_BUDGET_BYTES
            ),
            "telemetry_bytes": resource.TELEMETRY_WORST_CASE_BYTES,
            "wall_seconds": resource.MAXIMUM_WALL_SECONDS,
            "job_memory_limit_event_count": 0,
            "monitor_error_count": 0,
            "partial_or_temp_count": 0,
        }

    def test_all_exact_resource_boundaries_pass(self) -> None:
        result = resource.verify_resource_envelope(
            self._exact_boundary()
        )
        self.assertEqual(result["status"], "pass")

    def test_odd_disk_floor_uses_the_exact_frozen_floor(self) -> None:
        evidence = self._exact_boundary()
        evidence["preflight_free_bytes"] += 1
        evidence["minimum_observed_free_bytes"] = (
            evidence["preflight_free_bytes"] // 2
        )
        resource.verify_resource_envelope(evidence)
        evidence["minimum_observed_free_bytes"] -= 1
        with self.assertRaises(
            resource.Gate12C2ResourceQualificationError
        ):
            resource.verify_resource_envelope(evidence)

    def test_each_one_unit_resource_breach_rejects(self) -> None:
        base = self._exact_boundary()
        mutations = (
            ("peak_job_memory_bytes", base["peak_job_memory_bytes"] + 1),
            (
                "sampled_combined_rss_bytes",
                base["sampled_combined_rss_bytes"] + 1,
            ),
            (
                "sampled_available_physical_memory_bytes",
                base["sampled_available_physical_memory_bytes"] - 1,
            ),
            ("preflight_free_bytes", resource.MINIMUM_PREFLIGHT_FREE_BYTES - 1),
            (
                "minimum_observed_free_bytes",
                base["minimum_observed_free_bytes"] - 1,
            ),
            (
                "qualification_output_bytes",
                resource.QUALIFICATION_OUTPUT_BUDGET_BYTES + 1,
            ),
            (
                "telemetry_bytes",
                resource.TELEMETRY_WORST_CASE_BYTES + 1,
            ),
            ("wall_seconds", resource.MAXIMUM_WALL_SECONDS + 1),
            ("job_memory_limit_event_count", 1),
            ("monitor_error_count", 1),
            ("partial_or_temp_count", 1),
        )
        for field, value in mutations:
            evidence = dict(base)
            evidence[field] = value
            with self.subTest(field=field):
                with self.assertRaises(
                    resource.Gate12C2ResourceQualificationError
                ):
                    resource.verify_resource_envelope(evidence)

    def test_resource_schema_and_types_are_closed(self) -> None:
        for mutation in ("bool", "string", "extra", "missing"):
            evidence: dict[str, object] = dict(self._exact_boundary())
            if mutation == "bool":
                evidence["wall_seconds"] = True
            elif mutation == "string":
                evidence["wall_seconds"] = str(resource.MAXIMUM_WALL_SECONDS)
            elif mutation == "extra":
                evidence["unknown"] = 0
            else:
                evidence.pop("wall_seconds")
            with self.subTest(mutation=mutation):
                with self.assertRaises(
                    resource.Gate12C2ResourceQualificationError
                ):
                    resource.verify_resource_envelope(evidence)


if __name__ == "__main__":
    unittest.main()
