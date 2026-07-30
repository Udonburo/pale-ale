#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import gate12c2_resource_qualification as resource


def _proof() -> resource.JobHandleOwnershipProof:
    return resource.JobHandleOwnershipProof(
        source_pid=10,
        source_creation_time_ns=100,
        watchdog_pid=20,
        watchdog_creation_time_ns=200,
        watchdog_raw_handle=123,
        source_handle_closed=True,
        target_handle_noninheritable=True,
        target_handle_valid_job=True,
    )


class _JobHandle:
    inheritable = False

    def __init__(self, *, fail_close_attempts: int = 0) -> None:
        self.ownership_proof = _proof()
        self.fail_close_attempts = fail_close_attempts
        self.close_count = 0

    def close_for_kill(self) -> None:
        self.close_count += 1
        if self.close_count <= self.fail_close_attempts:
            raise OSError("RAW_CLOSE_SENTINEL")


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
        self.assertEqual(handle.close_count, 1)
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
        self.assertEqual(handle.close_count, 1)

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
        self.assertEqual(handle.close_count, 1)

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
        self.assertEqual(handle.close_count, 1)

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
        self.assertEqual(recoverable.close_count, 3)
        self.assertTrue(watchdog.job_handle_close_verified)

        unrecoverable = _JobHandle(fail_close_attempts=99)
        watchdog = resource.LaunchDeadlineWatchdog(unrecoverable)
        watchdog.resume_and_arm(lambda: 1)
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "Job handle close failed",
        ):
            watchdog.verify_acknowledgement(None, lambda _: True)
        self.assertEqual(unrecoverable.close_count, 3)
        self.assertFalse(watchdog.job_handle_close_verified)
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "Job handle close failed",
        ):
            watchdog.verify_acknowledgement(None, lambda _: True)
        self.assertEqual(unrecoverable.close_count, 6)


class _TransferApi(resource.WindowsJobApi):
    def __init__(
        self,
        *,
        inheritable: bool = False,
        valid_job: bool = True,
        fail_source_close: bool = False,
    ) -> None:
        self.inheritable_result = inheritable
        self.valid_job_result = valid_job
        self.fail_source_close = fail_source_close
        self.calls: list[tuple[str, int]] = []

    def _duplicate_into_process(
        self, job_handle: int, target_process_handle: int
    ) -> int:
        self.calls.append(("duplicate", job_handle))
        self.target_process_handle = target_process_handle
        return 123

    def close_handle(self, handle: int) -> None:
        self.calls.append(("close", handle))
        if handle == 11 and self.fail_source_close:
            raise resource.Gate12C2ResourceQualificationError(
                "source close failed"
            )

    def _close_remote_handle(
        self, target_process_handle: int, remote_handle: int
    ) -> None:
        self.calls.append(("close_remote", remote_handle))
        self.target_process_handle = target_process_handle

    def query_handle_inheritable(self, handle: int) -> bool:
        self.calls.append(("query_inheritable", handle))
        return self.inheritable_result

    def verify_job_handle(self, handle: int) -> bool:
        self.calls.append(("verify_job", handle))
        return self.valid_job_result


class JobOwnershipAdversarialTest(unittest.TestCase):
    def _receipt(
        self, api: _TransferApi
    ) -> resource.JobHandleTransferReceipt:
        return api.transfer_job_handle_to_watchdog(
            source_job_handle=11,
            target_process_handle=22,
            source_identity=resource.ProcessIdentity(10, 100),
            watchdog_identity=resource.ProcessIdentity(20, 200),
        )

    def test_transfer_closes_source_before_receipt_and_claim_queries_flags(
        self,
    ) -> None:
        api = _TransferApi()
        receipt = self._receipt(api)
        self.assertEqual(
            api.calls[:2], [("duplicate", 11), ("close", 11)]
        )
        handle = resource.WatchdogOwnedWindowsJobHandle.from_transfer_receipt(
            receipt,
            api=api,
            current_identity=resource.ProcessIdentity(20, 200),
        )
        self.assertTrue(handle.sole_owner_verified)
        self.assertIn(("query_inheritable", 123), api.calls)
        self.assertIn(("verify_job", 123), api.calls)

    def test_direct_construction_and_live_source_are_rejected(self) -> None:
        api = _TransferApi()
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "direct Job transfer-receipt construction",
        ):
            resource.JobHandleTransferReceipt(
                source_pid=10,
                source_creation_time_ns=100,
                watchdog_pid=20,
                watchdog_creation_time_ns=200,
                watchdog_raw_handle=123,
                source_handle_closed=True,
                duplicate_requested_noninheritable=True,
            )
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "direct",
        ):
            resource.WatchdogOwnedWindowsJobHandle(
                123,
                api,
                _proof(),
            )
        failed_api = _TransferApi(fail_source_close=True)
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "source Job handle close failed",
        ):
            self._receipt(failed_api)
        self.assertIn(("close_remote", 123), failed_api.calls)

    def test_inheritable_invalid_or_wrong_target_handle_is_rejected(
        self,
    ) -> None:
        receipt = self._receipt(_TransferApi())
        cases = (
            (
                _TransferApi(inheritable=True),
                resource.ProcessIdentity(20, 200),
            ),
            (
                _TransferApi(valid_job=False),
                resource.ProcessIdentity(20, 200),
            ),
            (
                _TransferApi(),
                resource.ProcessIdentity(21, 200),
            ),
        )
        for api, identity in cases:
            with self.subTest(api=api, identity=identity):
                with self.assertRaises(
                    resource.Gate12C2ResourceQualificationError
                ):
                    resource.WatchdogOwnedWindowsJobHandle.from_transfer_receipt(
                        receipt,
                        api=api,
                        current_identity=identity,
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

    def test_odd_disk_floor_uses_the_exact_half_ceiling(self) -> None:
        evidence = self._exact_boundary()
        evidence["preflight_free_bytes"] += 1
        evidence["minimum_observed_free_bytes"] = (
            evidence["preflight_free_bytes"] + 1
        ) // 2
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
