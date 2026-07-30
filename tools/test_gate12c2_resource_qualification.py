#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import gate12c2_resource_qualification as resource


class _FakeJobHandle:
    inheritable = False

    def __init__(self) -> None:
        self.close_count = 0
        self.ownership_proof = resource.JobHandleOwnershipProof(
            source_pid=1,
            source_creation_time_ns=1,
            watchdog_pid=2,
            watchdog_creation_time_ns=2,
            watchdog_raw_handle=3,
            source_handle_closed=True,
            target_handle_noninheritable=True,
            target_handle_valid_job=True,
        )

    def close_for_kill(self) -> None:
        self.close_count += 1


class _UnownedJobHandle(_FakeJobHandle):
    def __init__(self) -> None:
        self.close_count = 0
        self.ownership_proof = None


class _InheritableJobHandle(_FakeJobHandle):
    inheritable = True


class _Clock:
    def __init__(self, *values: int) -> None:
        self.values = list(values)
        self.last = values[-1] if values else 0

    def __call__(self) -> int:
        if self.values:
            self.last = self.values.pop(0)
        return self.last


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


def _success_stream(path: Path, *, periodic: bool = False) -> bytes:
    with resource.AppendOnlyTelemetryWriter(path) as writer:
        monotonic = 10
        for index, event in enumerate(resource.SUCCESS_MILESTONES):
            writer.append(
                event_code=event,
                utc_time=_utc(index),
                monotonic_ns=monotonic,
                metrics=_metrics(),
            )
            monotonic += 1
            if periodic and event == "CHILD_RESUMED":
                writer.append(
                    event_code="PERIODIC_SAMPLE",
                    utc_time=_utc(index),
                    monotonic_ns=monotonic,
                    metrics=_metrics(),
                )
                monotonic += 1
    return path.read_bytes()


class TelemetryContractTest(unittest.TestCase):
    def test_frozen_algebra_is_complete_and_unique(self) -> None:
        self.assertEqual(len(resource.RUNTIME_STATES), 11)
        self.assertEqual(len(set(resource.RUNTIME_STATES)), 11)
        self.assertEqual(len(resource.EVENT_CODES), 17)
        self.assertEqual(len(set(resource.EVENT_CODES)), 17)
        self.assertEqual(len(resource.TRANSITIONS), 33)
        self.assertEqual(len(resource.SUCCESS_MILESTONES), 15)
        for state in resource.NONTERMINAL_STATES:
            self.assertEqual(
                resource.TRANSITIONS[(state, "PERIODIC_SAMPLE")], state
            )
            self.assertEqual(
                resource.TRANSITIONS[(state, "FAILURE_DETECTED")],
                "RESOURCE_MONITORING_FAILED",
            )
        for terminal in resource.TERMINAL_STATES:
            self.assertFalse(
                any(source == terminal for source, _ in resource.TRANSITIONS)
            )

    def test_maximum_capacity_fixture_is_exactly_971_bytes(self) -> None:
        fixture = resource.maximum_capacity_fixture_bytes()
        self.assertEqual(len(fixture), 971)
        self.assertLessEqual(len(fixture), resource.MAXIMUM_RECORD_BYTES)
        self.assertTrue(fixture.endswith(b"\n"))

    def test_complete_success_chain_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "telemetry.jsonl"
            payload = _success_stream(path, periodic=True)
        verified = resource.decode_and_verify_telemetry(payload)
        self.assertEqual(verified["status"], "pass")
        self.assertEqual(
            verified["terminal_state"], "RESOURCE_MONITORING_COMPLETE"
        )
        self.assertEqual(verified["record_count"], 16)
        self.assertEqual(verified["periodic_record_count"], 1)
        self.assertFalse(verified["scientific_values_emitted"])

    def test_failure_terminal_is_allowed_from_each_nonterminal(self) -> None:
        for success_prefix_length in range(1, 10):
            with self.subTest(success_prefix_length=success_prefix_length):
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "telemetry.jsonl"
                    with resource.AppendOnlyTelemetryWriter(path) as writer:
                        monotonic = 10
                        for index, event in enumerate(
                            resource.SUCCESS_MILESTONES[
                                :success_prefix_length
                            ]
                        ):
                            writer.append(
                                event_code=event,
                                utc_time=_utc(index),
                                monotonic_ns=monotonic,
                                metrics=_metrics(),
                            )
                            monotonic += 1
                        writer.append(
                            event_code="FAILURE_DETECTED",
                            utc_time=_utc(success_prefix_length),
                            monotonic_ns=monotonic,
                            metrics=_metrics(),
                        )
                    payload = path.read_bytes()
                verified = resource.decode_and_verify_telemetry(payload)
                self.assertEqual(
                    verified["terminal_state"],
                    "RESOURCE_MONITORING_FAILED",
                )

    def test_writer_rejects_reordered_and_post_terminal_events(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "telemetry.jsonl"
            writer = resource.AppendOnlyTelemetryWriter(path)
            with self.assertRaisesRegex(
                resource.Gate12C2ResourceQualificationError,
                "transition",
            ):
                writer.append(
                    event_code="CHILD_CREATED_SUSPENDED",
                    utc_time=_utc(0),
                    monotonic_ns=1,
                    metrics=_metrics(),
                )
            writer.append(
                event_code="WRAPPER_AUTHORIZATION_CONSUMED",
                utc_time=_utc(0),
                monotonic_ns=1,
                metrics=_metrics(),
            )
            writer.append(
                event_code="FAILURE_DETECTED",
                utc_time=_utc(1),
                monotonic_ns=2,
                metrics=_metrics(),
            )
            with self.assertRaisesRegex(
                resource.Gate12C2ResourceQualificationError,
                "transition",
            ):
                writer.append(
                    event_code="PERIODIC_SAMPLE",
                    utc_time=_utc(2),
                    monotonic_ns=3,
                    metrics=_metrics(),
                )
            writer.close()

    def test_strict_decoder_rejects_digest_canonical_and_key_attacks(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "telemetry.jsonl"
            payload = _success_stream(path)
        lines = payload.splitlines()
        first = json.loads(lines[0])

        tampered = dict(first)
        tampered["sha"] = "0" * 64
        tampered_payload = (
            resource.canonical_json_bytes(tampered)
            + b"\n"
            + b"\n".join(lines[1:])
            + b"\n"
        )
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "digest mismatch",
        ):
            resource.decode_and_verify_telemetry(tampered_payload)

        noncanonical_payload = (
            json.dumps(first, sort_keys=True).encode("utf-8")
            + b"\n"
            + b"\n".join(lines[1:])
            + b"\n"
        )
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "canonical",
        ):
            resource.decode_and_verify_telemetry(noncanonical_payload)

        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "duplicate JSON keys",
        ):
            resource.decode_and_verify_telemetry(
                b'{"seq":0,"seq":0}\n'
            )

    def test_strict_encoder_rejects_bool_numeric_string_and_unknown_enum(
        self,
    ) -> None:
        base = {
            "schema_version": resource.TELEMETRY_SCHEMA,
            "sequence": 0,
            "utc_time": _utc(0),
            "monotonic_ns": 0,
            "previous_record_sha256": resource.GENESIS_DIGEST,
            "state": "PRELAUNCH",
            "event_code": "WRAPPER_AUTHORIZATION_CONSUMED",
            **_metrics(),
        }
        for field, value in (
            ("sequence", True),
            ("sequence", "0"),
            ("event_code", "UNKNOWN"),
        ):
            candidate = dict(base)
            candidate[field] = value
            with self.subTest(field=field, value=value):
                with self.assertRaises(
                    resource.Gate12C2ResourceQualificationError
                ):
                    resource.encode_telemetry_record(candidate)

    def test_writer_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "telemetry.jsonl"
            path.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                resource.AppendOnlyTelemetryWriter(path)


class LaunchDeadlineWatchdogTest(unittest.TestCase):
    def _watchdog(
        self, handle: _FakeJobHandle, *clock_values: int
    ) -> resource.LaunchDeadlineWatchdog:
        return resource.LaunchDeadlineWatchdog(
            handle,
            monotonic_ns=_Clock(*clock_values),
        )

    def test_deadline_minus_one_nanosecond_accepts_in_production_loop(
        self,
    ) -> None:
        handle = _FakeJobHandle()
        start = 1_000
        watchdog = self._watchdog(
            handle,
            start,
            start + 1,
            start + resource.LAUNCH_EVIDENCE_DEADLINE_NS - 1,
        )
        watchdog.resume_and_arm(lambda: 1)
        ack_time = watchdog.run_until_verified(
            acknowledgement_supplier=lambda: {"status": "present"},
            verifier=lambda _: True,
        )
        self.assertEqual(
            ack_time,
            start + resource.LAUNCH_EVIDENCE_DEADLINE_NS - 1,
        )
        self.assertTrue(watchdog.verified)
        self.assertEqual(handle.close_count, 0)

    def test_deadline_exact_rejects_and_closes_sole_job_handle(self) -> None:
        handle = _FakeJobHandle()
        start = 1_000
        watchdog = self._watchdog(
            handle,
            start,
            start + 1,
            start + resource.LAUNCH_EVIDENCE_DEADLINE_NS,
        )
        watchdog.resume_and_arm(lambda: 1)
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "missed the deadline",
        ):
            watchdog.run_until_verified(
                acknowledgement_supplier=lambda: {"status": "present"},
                verifier=lambda _: True,
            )
        self.assertEqual(handle.close_count, 1)

    def test_missing_ack_at_deadline_kills_job(self) -> None:
        handle = _FakeJobHandle()
        start = 50
        watchdog = self._watchdog(
            handle,
            start,
            start + resource.LAUNCH_EVIDENCE_DEADLINE_NS,
        )
        watchdog.resume_and_arm(lambda: 1)
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "missing at deadline",
        ):
            watchdog.run_until_verified(
                acknowledgement_supplier=lambda: None,
                verifier=lambda _: True,
            )
        self.assertEqual(handle.close_count, 1)

    def test_invalid_ack_kills_job(self) -> None:
        handle = _FakeJobHandle()
        watchdog = self._watchdog(handle, 10, 11)
        watchdog.resume_and_arm(lambda: 1)
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "invalid",
        ):
            watchdog.run_until_verified(
                acknowledgement_supplier=lambda: {"bad": True},
                verifier=lambda _: False,
            )
        self.assertEqual(handle.close_count, 1)

    def test_verifier_failure_kills_job_without_raw_exception(self) -> None:
        handle = _FakeJobHandle()
        watchdog = self._watchdog(handle, 10, 11)
        watchdog.resume_and_arm(lambda: 1)

        def fail(_: object) -> bool:
            raise RuntimeError("RAW_ACK_SENTINEL")

        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "^launch acknowledgement verifier failed$",
        ) as raised:
            watchdog.run_until_verified(
                acknowledgement_supplier=lambda: {"status": "present"},
                verifier=fail,
            )
        self.assertNotIn("RAW_ACK_SENTINEL", str(raised.exception))
        self.assertEqual(handle.close_count, 1)

    def test_supplier_failure_and_resume_failure_kill_job(self) -> None:
        handle = _FakeJobHandle()
        watchdog = self._watchdog(handle, 10, 11)
        watchdog.resume_and_arm(lambda: 1)
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "supplier failed",
        ):
            watchdog.run_until_verified(
                acknowledgement_supplier=lambda: (_ for _ in ()).throw(
                    RuntimeError("RAW_SUPPLIER_SENTINEL")
                ),
                verifier=lambda _: True,
            )
        self.assertEqual(handle.close_count, 1)

        second = _FakeJobHandle()
        watchdog = self._watchdog(second, 10)
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "resume result is invalid",
        ):
            watchdog.resume_and_arm(lambda: 0)
        self.assertEqual(second.close_count, 1)

    def test_unowned_or_inheritable_handle_is_rejected(self) -> None:
        for handle in (_UnownedJobHandle(), _InheritableJobHandle()):
            with self.subTest(handle=type(handle).__name__):
                with self.assertRaises(
                    resource.Gate12C2ResourceQualificationError
                ):
                    resource.LaunchDeadlineWatchdog(handle)


class GuardianAndClassifierTest(unittest.TestCase):
    def test_no_handle_guardian_cannot_continue_or_pass(self) -> None:
        guardian = resource.NoHandleGuardian(
            (
                resource.ProcessIdentity(100, 1_000),
                resource.ProcessIdentity(101, 1_001),
            )
        )
        self.assertFalse(guardian.owns_job_handle)
        result = guardian.record_watchdog_failure(lambda _: "DEAD")
        self.assertFalse(result["continuation_authorized"])
        self.assertFalse(result["qualification_pass_authorized"])
        self.assertFalse(result["job_handle_owned"])
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "active or unknown",
        ):
            guardian.record_watchdog_failure(lambda _: "UNKNOWN")

    @staticmethod
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

    def test_exact_legacy_closeout_is_classified_without_success_claim(
        self,
    ) -> None:
        result = resource.classify_expected_legacy_closeout(
            self._legacy_evidence()
        )
        self.assertEqual(
            result["status"],
            "REPLAY_PAYLOAD_COMPLETE_WITH_EXPECTED_LEGACY_CLOSEOUT_FAILURE",
        )
        self.assertFalse(result["legacy_child_success_claimed"])
        self.assertFalse(result["scientific_values_emitted"])

    def test_any_legacy_terminal_or_payload_deviation_rejects(self) -> None:
        for field, value in (
            ("child_exit_code", 0),
            ("stdout_bytes", 1),
            ("exception_type", "ValueError"),
            ("configuration_count", 8),
            ("partial_or_temp_count", 1),
            ("semantic_commitments_match", False),
            ("telemetry_tail_complete", False),
            ("legacy_resource_receipt_present", True),
        ):
            evidence = self._legacy_evidence()
            evidence[field] = value
            with self.subTest(field=field):
                with self.assertRaises(
                    resource.Gate12C2ResourceQualificationError
                ):
                    resource.classify_expected_legacy_closeout(evidence)


class ResourceEnvelopeTest(unittest.TestCase):
    @staticmethod
    def _passing_evidence() -> dict[str, int]:
        physical = 16 * 1024**3
        preflight = 12 * 1024**3
        return {
            "physical_ram_bytes": physical,
            "peak_job_memory_bytes": 8 * 1024**3,
            "sampled_combined_rss_bytes": 9 * 1024**3,
            "sampled_available_physical_memory_bytes": 2 * 1024**3,
            "preflight_free_bytes": preflight,
            "minimum_observed_free_bytes": preflight // 2,
            "qualification_output_bytes": 1_500_000_000,
            "telemetry_bytes": 1_300_000_000,
            "wall_seconds": 120_000,
            "job_memory_limit_event_count": 0,
            "monitor_error_count": 0,
            "partial_or_temp_count": 0,
        }

    def test_resource_envelope_passes_at_frozen_boundaries(self) -> None:
        result = resource.verify_resource_envelope(
            self._passing_evidence()
        )
        self.assertEqual(result["status"], "pass")
        self.assertEqual(
            result["original_resource_gate_status"], "indeterminate"
        )
        self.assertFalse(result["scientific_values_emitted"])

    def test_each_resource_breach_fails_closed(self) -> None:
        base = self._passing_evidence()
        physical = base["physical_ram_bytes"]
        breaches = (
            ("peak_job_memory_bytes", (3 * physical) // 4 + 1),
            ("sampled_combined_rss_bytes", (3 * physical) // 4 + 1),
            (
                "sampled_available_physical_memory_bytes",
                (physical + 9) // 10 - 1,
            ),
            (
                "minimum_observed_free_bytes",
                base["preflight_free_bytes"] // 2 - 1,
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
        for field, value in breaches:
            evidence = dict(base)
            evidence[field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    resource.Gate12C2ResourceQualificationError,
                    "frozen envelope",
                ):
                    resource.verify_resource_envelope(evidence)


class WindowsHandleAbstractionTest(unittest.TestCase):
    def test_watchdog_owned_handle_closes_backend_once(self) -> None:
        class Api:
            def __init__(self) -> None:
                self.closed: list[int] = []

            def query_handle_inheritable(self, handle: int) -> bool:
                self.asserted_handle = handle
                return False

            def verify_job_handle(self, handle: int) -> bool:
                self.asserted_job = handle
                return True

            def close_handle(self, handle: int) -> None:
                self.closed.append(handle)

        api = Api()
        receipt = resource.JobHandleTransferReceipt(
            source_pid=10,
            source_creation_time_ns=100,
            watchdog_pid=20,
            watchdog_creation_time_ns=200,
            watchdog_raw_handle=123,
            source_handle_closed=True,
            duplicate_requested_noninheritable=True,
            _token=resource._JOB_TRANSFER_RECEIPT_TOKEN,
        )
        handle = resource.WatchdogOwnedWindowsJobHandle.from_transfer_receipt(
            receipt,
            api=api,  # type: ignore[arg-type]
            current_identity=resource.ProcessIdentity(20, 200),
        )
        handle.close_for_kill()
        handle.close_for_kill()
        self.assertEqual(api.closed, [123])
        self.assertTrue(handle.sole_owner_verified)
        self.assertFalse(handle.inheritable)


if __name__ == "__main__":
    unittest.main()
