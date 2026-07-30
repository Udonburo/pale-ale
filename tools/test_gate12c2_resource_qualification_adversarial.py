#!/usr/bin/env python3
from __future__ import annotations

import ctypes
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import gate12c2_resource_qualification as resource
from gate12c2_resource_test_support import (
    Clock,
    FakeKernel32,
    TEST_ATTEMPT_ID,
    geometry,
    job_api,
    make_launch,
    telemetry_metrics,
    telemetry_paths,
    utc,
    write_success_telemetry,
)
from test_gate12c2_resource_qualification import (
    GuardianAndClassifierTest,
    ResourceEnvelopeTest,
)


class AtomicLaunchAdversarialTest(unittest.TestCase):
    def _launch(self, kernel: FakeKernel32) -> None:
        job_api(kernel).launch_scientific_child_suspended(
            preflight_geometry=geometry(kernel),
            application_name="python.exe",
            command_line="python.exe -B frozen.py",
            current_directory=Path.cwd(),
        )

    def test_first_attribute_call_must_fail_with_122_and_positive_size(
        self,
    ) -> None:
        cases = (
            FakeKernel32(first_initialize_returns_true=True),
            FakeKernel32(first_initialize_error=5),
            FakeKernel32(first_attribute_bytes=0),
        )
        for kernel in cases:
            with self.subTest(
                returns_true=kernel.first_initialize_returns_true,
                error=kernel.first_initialize_error,
                bytes=kernel.first_attribute_bytes,
            ):
                with self.assertRaisesRegex(
                    resource.Gate12C2ResourceQualificationError,
                    "launch preparation failed",
                ):
                    self._launch(kernel)
                self.assertNotIn("CreateProcessW", kernel.calls)
                self.assertEqual(kernel.open_handles, set())

    def test_second_attribute_initialization_failure_closes_job(
        self,
    ) -> None:
        kernel = FakeKernel32(second_initialize_succeeds=False)
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "launch preparation failed",
        ):
            self._launch(kernel)
        self.assertNotIn("UpdateProcThreadAttribute", kernel.calls)
        self.assertNotIn("CreateProcessW", kernel.calls)
        self.assertEqual(kernel.open_handles, set())

    def test_second_attribute_size_change_is_rejected_after_delete(
        self,
    ) -> None:
        kernel = FakeKernel32(second_attribute_bytes=256)
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "launch preparation failed",
        ):
            self._launch(kernel)
        self.assertEqual(kernel.delete_calls, 1)
        self.assertEqual(kernel.open_handles, set())

    def test_attribute_delete_failure_after_create_kills_everything(
        self,
    ) -> None:
        kernel = FakeKernel32(delete_raises=True)
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "deletion was not verified",
        ):
            self._launch(kernel)
        self.assertIn("CreateProcessW", kernel.calls)
        self.assertEqual(kernel.delete_calls, 1)
        self.assertEqual(kernel.open_handles, set())

    def test_failed_launch_closes_job_first_and_attempts_every_handle(
        self,
    ) -> None:
        cases = (
            (
                "job",
                FakeKernel32(delete_raises=True, fail_close_handles={100}),
                {100},
                "job=unverified,thread=closed,process=closed",
            ),
            (
                "process",
                FakeKernel32(delete_raises=True, fail_close_handles={200}),
                {200},
                "job=closed,thread=closed,process=unverified",
            ),
        )
        for label, kernel, expected_open, expected_summary in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                    resource.Gate12C2ResourceQualificationError,
                    "scientific-launch cleanup incomplete",
                ) as raised:
                    self._launch(kernel)
                self.assertEqual(
                    kernel.close_calls[-3:],
                    [
                        kernel.job_handle,
                        kernel.thread_handle,
                        kernel.process_handle,
                    ],
                )
                self.assertEqual(kernel.open_handles, expected_open)
                self.assertIn(expected_summary, str(raised.exception))
                self.assertNotIn("RAW_", str(raised.exception))

    def test_job_close_exception_does_not_skip_child_handle_cleanup(
        self,
    ) -> None:
        kernel = FakeKernel32(delete_raises=True, fail_close_handles={100})
        with self.assertRaises(
            resource.Gate12C2ResourceQualificationError
        ):
            self._launch(kernel)
        self.assertNotIn(kernel.thread_handle, kernel.open_handles)
        self.assertNotIn(kernel.process_handle, kernel.open_handles)
        self.assertEqual(set(kernel.close_calls[-3:]), {100, 200, 300})

    def test_createprocess_failure_never_accepts_residual_identity(
        self,
    ) -> None:
        kernel = FakeKernel32(
            create_process_succeeds=False,
            residual_process_information_on_failure=True,
        )
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "CreateProcessW failed",
        ):
            self._launch(kernel)
        self.assertEqual(kernel.open_handles, set())
        self.assertEqual(kernel.delete_calls, 1)

    def test_extra_limit_flag_or_wrong_post_set_J_rejects(self) -> None:
        for kernel in (
            FakeKernel32(
                limit_flags=FakeKernel32.REQUIRED_FLAGS | 1
            ),
            FakeKernel32(reported_limit=1),
        ):
            with self.subTest(
                flags=kernel.limit_flags, limit=kernel.reported_limit
            ):
                with self.assertRaisesRegex(
                    resource.Gate12C2ResourceQualificationError,
                    "post-set Job limit",
                ):
                    self._launch(kernel)
                self.assertNotIn("CreateProcessW", kernel.calls)
                self.assertEqual(kernel.open_handles, set())

    def test_inheritable_local_job_is_rejected_before_child(self) -> None:
        kernel = FakeKernel32()
        original = kernel._create_job

        def inheritable_job(security, name):
            handle = original(security, name)
            kernel.inheritable_handles.add(handle)
            return handle

        kernel.CreateJobObjectW.callback = inheritable_job
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "inheritable",
        ):
            self._launch(kernel)
        self.assertNotIn("CreateProcessW", kernel.calls)
        self.assertEqual(kernel.open_handles, set())

    def test_each_child_identity_or_membership_deviation_rejects(self) -> None:
        cases = (
            ("membership", FakeKernel32(child_in_job=False)),
            ("pid", FakeKernel32(process_id_matches=False)),
            ("thread", FakeKernel32(thread_id_matches=False)),
            ("suspend", FakeKernel32(initial_suspend_count=0)),
            ("accounting", FakeKernel32(accounting=(2, 2, 0))),
        )
        for label, kernel in cases:
            with self.subTest(label=label):
                with self.assertRaises(
                    resource.Gate12C2ResourceQualificationError
                ):
                    self._launch(kernel)
                self.assertEqual(kernel.open_handles, set())

    def test_invalid_command_or_cwd_rejects_before_job_creation(self) -> None:
        cases = (
            ("python\x00.exe", "python.exe -B x.py", Path.cwd()),
            ("python.exe", "python.exe\x00 -B x.py", Path.cwd()),
            ("python.exe", "", Path.cwd()),
            ("python.exe", "python.exe -B x.py", Path.cwd() / "missing"),
        )
        for application, command, cwd in cases:
            kernel = FakeKernel32()
            api = job_api(kernel)
            with self.subTest(application=application, command=command):
                with self.assertRaises(
                    resource.Gate12C2ResourceQualificationError
                ):
                    api.launch_scientific_child_suspended(
                        preflight_geometry=geometry(kernel),
                        application_name=application,
                        command_line=command,
                        current_directory=cwd,
                    )
                self.assertNotIn("CreateJobObjectW", kernel.calls)

    def test_support_probe_cleanup_uncertainty_is_not_pass(self) -> None:
        kernel = FakeKernel32(delete_raises=True)
        with self.assertRaises(
            resource.Gate12C2ResourceQualificationError
        ):
            job_api(kernel).probe_job_list_attribute_support()
        self.assertEqual(kernel.delete_calls, 1)
        self.assertNotIn(kernel.job_handle, kernel.open_handles)

    def test_attribute_value_array_and_list_survive_create_until_delete(
        self,
    ) -> None:
        kernel = FakeKernel32()
        launch, _, kernel = make_launch(kernel)
        create_index = kernel.calls.index("CreateProcessW")
        delete_index = kernel.calls.index("DeleteProcThreadAttributeList")
        update_index = kernel.calls.index("UpdateProcThreadAttribute")
        self.assertLess(update_index, create_index)
        self.assertLess(create_index, delete_index)
        self.assertGreater(launch.attribute_bytes, 0)

    def test_direct_local_launch_construction_is_forbidden(self) -> None:
        with self.assertRaises(
            resource.Gate12C2ResourceQualificationError
        ):
            resource.WatchdogLocalWindowsJobLaunch()


    def test_attribute_update_exception_still_deletes_and_closes_job(
        self,
    ) -> None:
        kernel = FakeKernel32()

        def raising_update(*args):
            del args
            raise OSError("RAW_UPDATE_SENTINEL")

        kernel.UpdateProcThreadAttribute.callback = raising_update
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "launch preparation failed",
        ) as raised:
            self._launch(kernel)
        self.assertNotIn("RAW_UPDATE_SENTINEL", str(raised.exception))
        self.assertEqual(kernel.delete_calls, 1)
        self.assertEqual(kernel.open_handles, set())
        self.assertNotIn("CreateProcessW", kernel.calls)

    def test_identity_change_before_deadline_owner_kills_suspended_job(
        self,
    ) -> None:
        launch, _, kernel = make_launch()
        kernel.process_id_matches = False
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "verification failed",
        ):
            resource.LaunchDeadlineWatchdog(
                launch, monotonic_ns=lambda: 1
            )
        self.assertNotIn(kernel.job_handle, kernel.open_handles)


class DeadlineAdversarialTest(unittest.TestCase):
    def test_clock_failure_closes_local_job(self) -> None:
        launch, _, kernel = make_launch()

        def broken_clock():
            raise RuntimeError("RAW_CLOCK_SENTINEL")

        watchdog = resource.LaunchDeadlineWatchdog(
            launch, monotonic_ns=broken_clock
        )
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "^monotonic clock failed$",
        ) as raised:
            watchdog.resume_and_arm()
        self.assertNotIn("RAW_CLOCK_SENTINEL", str(raised.exception))
        self.assertNotIn(kernel.job_handle, kernel.open_handles)

    def test_resume_failure_closes_local_job(self) -> None:
        launch, _, kernel = make_launch()
        watchdog = resource.LaunchDeadlineWatchdog(
            launch, monotonic_ns=lambda: 1
        )
        kernel.suspend_count = 0
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "resume failed",
        ):
            watchdog.resume_and_arm()
        self.assertNotIn(kernel.job_handle, kernel.open_handles)

    def test_supplier_exception_is_sanitized_and_kills_job(self) -> None:
        launch, _, kernel = make_launch()
        watchdog = resource.LaunchDeadlineWatchdog(
            launch, monotonic_ns=Clock(1, 2, 3)
        )
        watchdog.resume_and_arm()

        def broken_supplier():
            raise RuntimeError("RAW_SUPPLIER_SENTINEL")

        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "^launch acknowledgement supplier failed$",
        ) as raised:
            watchdog.run_until_verified(
                acknowledgement_supplier=broken_supplier,
                verifier=lambda _: True,
            )
        self.assertNotIn("RAW_SUPPLIER_SENTINEL", str(raised.exception))
        self.assertNotIn(kernel.job_handle, kernel.open_handles)

    def test_unverifiable_job_close_never_claims_kill(self) -> None:
        kernel = FakeKernel32(noop_close_handles={100})
        launch, _, kernel = make_launch(kernel)
        watchdog = resource.LaunchDeadlineWatchdog(
            launch, monotonic_ns=Clock(1, resource.LAUNCH_EVIDENCE_DEADLINE_NS + 1)
        )
        watchdog.resume_and_arm()
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "Job handle close failed",
        ):
            watchdog.run_until_verified(
                acknowledgement_supplier=lambda: None,
                verifier=lambda _: True,
            )
        self.assertIn(kernel.job_handle, kernel.open_handles)

    def test_queue_wait_is_clamped_to_remaining_deadline(self) -> None:
        launch, _, kernel = make_launch()
        observed_timeouts: list[float] = []

        class NeverCompletesQueue:
            def __init__(self, *args, **kwargs) -> None:
                del args, kwargs

            def put(self, *args, **kwargs) -> None:
                del args, kwargs

            def get(self, *, timeout: float):
                observed_timeouts.append(timeout)
                raise resource.queue.Empty

        watchdog = resource.LaunchDeadlineWatchdog(
            launch,
            monotonic_ns=Clock(0, 1_000_000, 2_000_000, 10_000_000),
        )
        with (
            mock.patch.object(
                resource, "LAUNCH_EVIDENCE_DEADLINE_NS", 10_000_000
            ),
            mock.patch.object(resource.queue, "Queue", NeverCompletesQueue),
        ):
            watchdog.resume_and_arm()
            with self.assertRaisesRegex(
                resource.Gate12C2ResourceQualificationError,
                "missing at deadline",
            ):
                watchdog.run_until_verified(
                    acknowledgement_supplier=lambda: None,
                    verifier=lambda _: True,
                    poll_seconds=0.1,
                )
        self.assertEqual(len(observed_timeouts), 1)
        self.assertAlmostEqual(observed_timeouts[0], 0.008, places=9)
        self.assertNotIn(kernel.job_handle, kernel.open_handles)

    def test_ack_ownership_wait_is_clamped_to_remaining_deadline(self) -> None:
        launch, _, kernel = make_launch()
        observed_timeouts: list[float] = []

        class ContendedLock:
            def acquire(self, *, timeout: float) -> bool:
                observed_timeouts.append(timeout)
                return False

            def release(self) -> None:
                raise AssertionError("unacquired lock must not be released")

        watchdog = resource.LaunchDeadlineWatchdog(
            launch,
            monotonic_ns=Clock(0, 2_000_000),
        )
        watchdog._ack_ownership_lock = ContendedLock()
        with mock.patch.object(
            resource, "LAUNCH_EVIDENCE_DEADLINE_NS", 10_000_000
        ):
            watchdog.resume_and_arm()
            with self.assertRaisesRegex(
                resource.Gate12C2ResourceQualificationError,
                "ownership missed the deadline",
            ):
                watchdog.verify_acknowledgement(
                    {"status": "present"},
                    lambda _: True,
                    poll_seconds=0.1,
                )
        self.assertEqual(len(observed_timeouts), 1)
        self.assertAlmostEqual(observed_timeouts[0], 0.008, places=9)
        self.assertTrue(watchdog.terminated)
        self.assertFalse(watchdog.verified)
        self.assertNotIn(kernel.job_handle, kernel.open_handles)

    def test_concurrent_duplicate_ack_has_one_owner_and_no_final_success(
        self,
    ) -> None:
        launch, _, kernel = make_launch()
        watchdog = resource.LaunchDeadlineWatchdog(launch)
        watchdog.resume_and_arm()
        verifier_entered = threading.Event()
        verifier_release = threading.Event()
        results: list[int] = []
        errors: list[str] = []

        def verifier(_: object) -> bool:
            verifier_entered.set()
            if not verifier_release.wait(timeout=2):
                raise RuntimeError("RAW_ACK_WAIT_SENTINEL")
            return True

        def attempt() -> None:
            try:
                results.append(
                    watchdog.verify_acknowledgement(
                        {"status": "present"}, verifier
                    )
                )
            except resource.Gate12C2ResourceQualificationError as error:
                errors.append(str(error))

        first = threading.Thread(target=attempt, daemon=True)
        second = threading.Thread(target=attempt, daemon=True)
        first.start()
        self.assertTrue(verifier_entered.wait(timeout=2))
        second.start()
        second.join(timeout=2)
        self.assertFalse(second.is_alive())
        verifier_release.set()
        first.join(timeout=2)
        self.assertFalse(first.is_alive())
        self.assertEqual(results, [])
        self.assertEqual(len(errors), 2)
        self.assertTrue(any("duplicated" in error for error in errors))
        self.assertFalse(
            any("RAW_ACK_WAIT_SENTINEL" in error for error in errors)
        )
        self.assertTrue(watchdog.terminated)
        self.assertFalse(watchdog.verified)
        self.assertIsNone(watchdog.ack_monotonic_ns)
        self.assertNotIn(kernel.job_handle, kernel.open_handles)


class TelemetryPublicationAdversarialTest(unittest.TestCase):
    def test_schema_failure_terminally_quarantines_writer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pending, final = telemetry_paths(Path(temporary))
            writer = resource.AppendOnlyTelemetryWriter(
                pending,
                final,
                attempt_identity_sha256=TEST_ATTEMPT_ID,
            )
            with self.assertRaisesRegex(
                resource.Gate12C2ResourceQualificationError,
                "^telemetry append failed$",
            ):
                writer.append(
                    event_code="PERIODIC_SAMPLE",
                    utc_time=utc(0),
                    monotonic_ns=0,
                    metrics=telemetry_metrics(),
                )
            self.assertTrue(pending.exists())
            self.assertFalse(final.exists())
            with self.assertRaisesRegex(
                resource.Gate12C2ResourceQualificationError,
                "terminally failed",
            ):
                writer.append(
                    event_code="WRAPPER_AUTHORIZATION_CONSUMED",
                    utc_time=utc(0),
                    monotonic_ns=0,
                    metrics=telemetry_metrics(),
                )

    def test_publication_receipt_is_bound_to_exact_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _, receipt = write_success_telemetry(directory)
            pending, final = telemetry_paths(directory)
            with self.assertRaises(
                resource.Gate12C2ResourceQualificationError
            ):
                resource.verify_telemetry_publication(
                    pending_path=pending,
                    final_path=final,
                    expected_attempt_identity_sha256="f" * 64,
                    receipt=receipt,
                )

    def test_fsync_failure_leaves_pending_and_never_final(self) -> None:
        def failing_fsync(_: int) -> None:
            raise OSError("RAW_FSYNC_SENTINEL")

        with tempfile.TemporaryDirectory() as temporary:
            pending, final = telemetry_paths(Path(temporary))
            writer = resource.AppendOnlyTelemetryWriter(
                pending,
                final,
                attempt_identity_sha256=TEST_ATTEMPT_ID,
                fsync=failing_fsync,
            )
            with self.assertRaisesRegex(
                resource.Gate12C2ResourceQualificationError,
                "^telemetry append failed$",
            ) as raised:
                writer.append(
                    event_code="WRAPPER_AUTHORIZATION_CONSUMED",
                    utc_time=utc(0),
                    monotonic_ns=0,
                    metrics=telemetry_metrics(),
                )
            self.assertNotIn("RAW_FSYNC_SENTINEL", str(raised.exception))
            self.assertTrue(pending.exists())
            self.assertFalse(final.exists())
            with self.assertRaisesRegex(
                resource.Gate12C2ResourceQualificationError,
                "terminally failed",
            ):
                writer.close()

    def test_close_exception_after_os_release_cannot_publish(self) -> None:
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
                raise OSError("RAW_CLOSE_SENTINEL")

        with tempfile.TemporaryDirectory() as temporary:
            pending, final = telemetry_paths(Path(temporary))
            writer = resource.AppendOnlyTelemetryWriter(
                pending,
                final,
                attempt_identity_sha256=TEST_ATTEMPT_ID,
            )
            writer.append(
                event_code="WRAPPER_AUTHORIZATION_CONSUMED",
                utc_time=utc(0),
                monotonic_ns=0,
                metrics=telemetry_metrics(),
            )
            writer.append(
                event_code="FAILURE_DETECTED",
                utc_time=utc(1),
                monotonic_ns=1,
                metrics=telemetry_metrics(),
            )
            writer._handle = RaisesAfterRelease(writer._handle)
            with self.assertRaisesRegex(
                resource.Gate12C2ResourceQualificationError,
                "handle close failed",
            ) as raised:
                writer.close()
            self.assertNotIn("RAW_CLOSE_SENTINEL", str(raised.exception))
            self.assertTrue(pending.exists())
            self.assertFalse(final.exists())

    def test_move_failure_leaves_pending_and_no_final(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pending, final = telemetry_paths(Path(temporary))
            writer = resource.AppendOnlyTelemetryWriter(
                pending,
                final,
                attempt_identity_sha256=TEST_ATTEMPT_ID,
            )
            writer.append(
                event_code="WRAPPER_AUTHORIZATION_CONSUMED",
                utc_time=utc(0),
                monotonic_ns=0,
                metrics=telemetry_metrics(),
            )
            writer.append(
                event_code="FAILURE_DETECTED",
                utc_time=utc(1),
                monotonic_ns=1,
                metrics=telemetry_metrics(),
            )
            with mock.patch.object(
                resource.AppendOnlyTelemetryWriter,
                "_publish_nonreplace",
                side_effect=OSError("RAW_MOVE_SENTINEL"),
            ):
                with self.assertRaisesRegex(
                    resource.Gate12C2ResourceQualificationError,
                    "^telemetry publication failed$",
                ) as raised:
                    writer.close()
            self.assertNotIn("RAW_MOVE_SENTINEL", str(raised.exception))
            self.assertTrue(pending.exists())
            self.assertFalse(final.exists())

    def test_final_path_race_is_never_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pending, final = telemetry_paths(Path(temporary))
            writer = resource.AppendOnlyTelemetryWriter(
                pending,
                final,
                attempt_identity_sha256=TEST_ATTEMPT_ID,
            )
            writer.append(
                event_code="WRAPPER_AUTHORIZATION_CONSUMED",
                utc_time=utc(0),
                monotonic_ns=0,
                metrics=telemetry_metrics(),
            )
            writer.append(
                event_code="FAILURE_DETECTED",
                utc_time=utc(1),
                monotonic_ns=1,
                metrics=telemetry_metrics(),
            )
            final.write_bytes(b"attacker")
            with self.assertRaisesRegex(
                resource.Gate12C2ResourceQualificationError,
                "publication failed",
            ):
                writer.close()
            self.assertEqual(final.read_bytes(), b"attacker")
            self.assertTrue(pending.exists())

    def test_pending_tamper_before_publication_is_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pending, final = telemetry_paths(Path(temporary))
            writer = resource.AppendOnlyTelemetryWriter(
                pending,
                final,
                attempt_identity_sha256=TEST_ATTEMPT_ID,
            )
            writer.append(
                event_code="WRAPPER_AUTHORIZATION_CONSUMED",
                utc_time=utc(0),
                monotonic_ns=0,
                metrics=telemetry_metrics(),
            )
            writer.append(
                event_code="FAILURE_DETECTED",
                utc_time=utc(1),
                monotonic_ns=1,
                metrics=telemetry_metrics(),
            )
            original_close = writer._close_pending_handle

            def close_and_tamper():
                result = original_close()
                pending.write_bytes(pending.read_bytes() + b"x")
                return result

            with mock.patch.object(
                writer, "_close_pending_handle", side_effect=close_and_tamper
            ):
                with self.assertRaisesRegex(
                    resource.Gate12C2ResourceQualificationError,
                    "strict verification",
                ):
                    writer.close()
            self.assertTrue(pending.exists())
            self.assertFalse(final.exists())

    def test_receipt_from_another_final_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as first_temp:
            first = Path(first_temp)
            _, receipt = write_success_telemetry(first)
            pending, final = telemetry_paths(first)
            with tempfile.TemporaryDirectory() as second_temp:
                _, second_final = telemetry_paths(Path(second_temp))
                second_final.write_bytes(final.read_bytes())
                with self.assertRaisesRegex(
                    resource.Gate12C2ResourceQualificationError,
                    "not frozen",
                ):
                    resource.verify_telemetry_publication(
                        pending_path=Path(second_temp)
                        / "telemetry.jsonl.pending",
                        final_path=second_final,
                        expected_attempt_identity_sha256=TEST_ATTEMPT_ID,
                        receipt=receipt,
                    )

    def test_final_tamper_after_publish_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _, receipt = write_success_telemetry(directory)
            pending, final = telemetry_paths(directory)
            final.write_bytes(final.read_bytes() + b"x")
            with self.assertRaises(
                resource.Gate12C2ResourceQualificationError
            ):
                resource.verify_telemetry_publication(
                    pending_path=pending,
                    final_path=final,
                    expected_attempt_identity_sha256=TEST_ATTEMPT_ID,
                    receipt=receipt,
                )

    def test_pending_artifact_alone_is_never_publication_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pending, final = telemetry_paths(directory)
            writer = resource.AppendOnlyTelemetryWriter(
                pending,
                final,
                attempt_identity_sha256=TEST_ATTEMPT_ID,
            )
            writer.append(
                event_code="WRAPPER_AUTHORIZATION_CONSUMED",
                utc_time=utc(0),
                monotonic_ns=0,
                metrics=telemetry_metrics(),
            )
            writer.append(
                event_code="FAILURE_DETECTED",
                utc_time=utc(1),
                monotonic_ns=1,
                metrics=telemetry_metrics(),
            )
            writer._abort()
            fake = {
                "schema_version": resource.TELEMETRY_PUBLICATION_SCHEMA
            }
            with self.assertRaises(
                resource.Gate12C2ResourceQualificationError
            ):
                resource.verify_telemetry_publication(
                    pending_path=pending,
                    final_path=final,
                    expected_attempt_identity_sha256=TEST_ATTEMPT_ID,
                    receipt=fake,
                )
            self.assertTrue(pending.exists())
            self.assertFalse(final.exists())

    def test_decoder_rejects_gap_unknown_keys_duplicate_and_bom(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payload, _ = write_success_telemetry(Path(temporary))
        lines = payload.splitlines(keepends=True)
        first = json.loads(lines[0])
        attacked = dict(first)
        attacked["unknown"] = 1
        without_sha = dict(attacked)
        without_sha.pop("sha")
        attacked["sha"] = resource.sha256_bytes(
            resource.canonical_json_bytes(without_sha)
        )
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "wire keys",
        ):
            resource.decode_and_verify_telemetry(
                resource.canonical_json_bytes(attacked)
                + b"\n"
                + b"".join(lines[1:])
            )
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "duplicate JSON keys",
        ):
            resource.decode_and_verify_telemetry(b'{"seq":0,"seq":0}\n')
        with self.assertRaises(
            resource.Gate12C2ResourceQualificationError
        ):
            resource.decode_and_verify_telemetry(b"\xef\xbb\xbf" + payload)

    @unittest.skipUnless(os.name == "nt", "Windows share-mode assertion")
    def test_live_pending_denies_second_raw_writer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pending, final = telemetry_paths(Path(temporary))
            writer = resource.AppendOnlyTelemetryWriter(
                pending,
                final,
                attempt_identity_sha256=TEST_ATTEMPT_ID,
            )
            with self.assertRaises(OSError):
                pending.open("ab")
            writer.append(
                event_code="WRAPPER_AUTHORIZATION_CONSUMED",
                utc_time=utc(0),
                monotonic_ns=0,
                metrics=telemetry_metrics(),
            )
            writer.append(
                event_code="FAILURE_DETECTED",
                utc_time=utc(1),
                monotonic_ns=1,
                metrics=telemetry_metrics(),
            )
            writer.close()


class GeometryFailClosedAdversarialTest(unittest.TestCase):
    def test_tampered_derived_geometry_rejects_bool_and_missing_values(
        self,
    ) -> None:
        for field, value in (
            ("rounding_delta_bytes", False),
            ("mathematical_memory_limit_bytes", True),
        ):
            candidate = geometry(FakeKernel32())
            object.__setattr__(candidate, field, value)
            with self.subTest(field=field):
                with self.assertRaises(
                    resource.Gate12C2ResourceQualificationError
                ):
                    candidate._validate()

    def test_process_identity_requires_nonzero_creation_time(self) -> None:
        with self.assertRaises(
            resource.Gate12C2ResourceQualificationError
        ):
            resource.ProcessIdentity(100, 0)


class GuardianFailClosedAdversarialTest(unittest.TestCase):
    def test_guardian_rejects_duplicate_or_untyped_identities(self) -> None:
        identity = resource.ProcessIdentity(100, 1_000)
        for identities in ((identity, identity), (object(),), ()):
            with self.subTest(identities=identities):
                with self.assertRaises(
                    resource.Gate12C2ResourceQualificationError
                ):
                    resource.NoHandleGuardian(identities)

    def test_guardian_sanitizes_probe_failure(self) -> None:
        guardian = resource.NoHandleGuardian(
            (resource.ProcessIdentity(100, 1_000),)
        )

        def fail(_: resource.ProcessIdentity) -> str:
            raise RuntimeError("RAW_GUARDIAN_SENTINEL")

        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "^guardian process probe failed$",
        ) as raised:
            guardian.record_watchdog_failure(fail)
        self.assertNotIn("RAW_GUARDIAN_SENTINEL", str(raised.exception))


class ResourceAndLineageAdversarialTest(unittest.TestCase):
    def test_job_peak_uses_J_but_sampled_combined_uses_M(self) -> None:
        evidence = ResourceEnvelopeTest.passing_evidence()
        self.assertLess(
            evidence["effective_job_memory_limit_bytes"],
            evidence["mathematical_memory_limit_bytes"],
        )
        resource.verify_resource_envelope(evidence)
        attacked = dict(evidence)
        attacked["peak_job_memory_bytes"] = (
            attacked["effective_job_memory_limit_bytes"] + 1
        )
        with self.assertRaises(
            resource.Gate12C2ResourceQualificationError
        ):
            resource.verify_resource_envelope(attacked)
        attacked = dict(evidence)
        attacked["sampled_combined_rss_bytes"] = (
            attacked["mathematical_memory_limit_bytes"] + 1
        )
        with self.assertRaises(
            resource.Gate12C2ResourceQualificationError
        ):
            resource.verify_resource_envelope(attacked)

    def test_available_floor_is_exact_ceil_P_over_ten_for_all_remainders(
        self,
    ) -> None:
        for remainder in range(10):
            with self.subTest(remainder=remainder):
                evidence = ResourceEnvelopeTest.passing_evidence()
                physical = 16_000_000_000 + remainder
                geometry_value = resource.derive_resource_memory_geometry(
                    physical, evidence["native_page_size_bytes"]
                )
                expected_floor = (physical + 9) // 10
                evidence.update(
                    {
                        "physical_ram_bytes": physical,
                        "mathematical_memory_limit_bytes": (
                            geometry_value.mathematical_memory_limit_bytes
                        ),
                        "effective_job_memory_limit_bytes": (
                            geometry_value.effective_job_memory_limit_bytes
                        ),
                        "rounding_delta_bytes": (
                            geometry_value.rounding_delta_bytes
                        ),
                        "peak_job_memory_bytes": (
                            geometry_value.effective_job_memory_limit_bytes
                        ),
                        "sampled_combined_rss_bytes": (
                            geometry_value.mathematical_memory_limit_bytes
                        ),
                        "sampled_available_physical_memory_bytes": expected_floor,
                    }
                )
                verified = resource.verify_resource_envelope(evidence)
                self.assertEqual(
                    verified["sampled_available_memory_floor_bytes"],
                    expected_floor,
                )
                evidence["sampled_available_physical_memory_bytes"] -= 1
                with self.assertRaises(
                    resource.Gate12C2ResourceQualificationError
                ):
                    resource.verify_resource_envelope(evidence)

    def test_legacy_stack_and_root_surface_are_closed(self) -> None:
        base = GuardianAndClassifierTest.legacy_evidence()
        mutations = (
            (
                "normalized_project_stack",
                list(reversed(base["normalized_project_stack"])),
            ),
            ("stale_lock_relative_path", "other.lock"),
            ("stale_lock_file_sha256", "0" * 64),
            ("unexpected_artifact_relative_paths", ["x"]),
        )
        for field, value in mutations:
            evidence = dict(base)
            evidence[field] = value
            if field == "unexpected_artifact_relative_paths":
                evidence["unexpected_artifact_count"] = 1
            with self.subTest(field=field):
                with self.assertRaises(
                    resource.Gate12C2ResourceQualificationError
                ):
                    resource.classify_expected_legacy_closeout(evidence)

    def test_guardian_rejects_active_unknown_or_invalid_probe(self) -> None:
        guardian = resource.NoHandleGuardian(
            (resource.ProcessIdentity(1, 1),)
        )
        for result in ("ACTIVE", "UNKNOWN", "INVALID", None):
            with self.subTest(result=result):
                with self.assertRaises(
                    resource.Gate12C2ResourceQualificationError
                ):
                    guardian.record_watchdog_failure(lambda _: result)


class RealWindowsMeasurementSmokeTest(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows-only OS measurement")
    def test_real_P_and_S_measurement_derives_valid_page_aligned_J(
        self,
    ) -> None:
        measured = resource.WindowsJobApi().measure_resource_geometry()
        measured._validate()
        self.assertGreater(measured.physical_ram_bytes, 0)
        self.assertGreater(measured.native_page_size_bytes, 0)
        self.assertEqual(
            measured.effective_job_memory_limit_bytes
            % measured.native_page_size_bytes,
            0,
        )
        self.assertLessEqual(
            measured.effective_job_memory_limit_bytes,
            measured.mathematical_memory_limit_bytes,
        )


if __name__ == "__main__":
    unittest.main()
