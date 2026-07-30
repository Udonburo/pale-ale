#!/usr/bin/env python3
from __future__ import annotations

import ctypes
import inspect
import json
import tempfile
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


class TelemetryContractTest(unittest.TestCase):
    def test_v05_algebra_is_complete_unique_and_ordered(self) -> None:
        self.assertEqual(resource.TELEMETRY_SCHEMA.endswith("v0.5"), True)
        self.assertEqual(len(resource.RUNTIME_STATES), 11)
        self.assertEqual(len(set(resource.RUNTIME_STATES)), 11)
        self.assertEqual(len(resource.EVENT_CODES), 17)
        self.assertEqual(len(set(resource.EVENT_CODES)), 17)
        self.assertEqual(len(resource.TRANSITIONS), 33)
        self.assertEqual(len(resource.SUCCESS_MILESTONES), 15)
        self.assertEqual(
            resource.SUCCESS_MILESTONES[:6],
            (
                "WRAPPER_AUTHORIZATION_CONSUMED",
                "JOINT_PRELAUNCH_CLAIM_SEALED",
                "WATCHDOG_SOLE_HANDLE_CONFIRMED",
                "CHILD_CREATED_SUSPENDED",
                "JOB_ASSIGNED",
                "JOINT_PRE_RESUME_RECEIPT_SEALED",
            ),
        )
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

    def test_maximum_capacity_fixture_remains_971_bytes(self) -> None:
        fixture = resource.maximum_capacity_fixture_bytes()
        self.assertEqual(len(fixture), 971)
        self.assertLessEqual(len(fixture), resource.MAXIMUM_RECORD_BYTES)
        self.assertTrue(fixture.endswith(b"\n"))

    def test_success_stream_publishes_only_after_clean_close(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            payload, receipt = write_success_telemetry(
                directory, periodic=True
            )
            pending, final = telemetry_paths(directory)
            self.assertFalse(pending.exists())
            self.assertTrue(final.is_file())
            verified = resource.decode_and_verify_telemetry(payload)
            publication = resource.verify_telemetry_publication(
                pending_path=pending,
                final_path=final,
                expected_attempt_identity_sha256=TEST_ATTEMPT_ID,
                receipt=receipt,
            )
        self.assertEqual(
            verified["terminal_state"], "RESOURCE_MONITORING_COMPLETE"
        )
        self.assertEqual(verified["record_count"], 16)
        self.assertEqual(publication["status"], "pass")
        self.assertEqual(
            publication["attempt_identity_sha256"], TEST_ATTEMPT_ID
        )
        self.assertEqual(
            publication["terminal_event_code"], "MONITORING_COMPLETED"
        )
        self.assertTrue(publication["clean_close_verified"])
        self.assertEqual(publication["move_result"], "success")
        self.assertFalse(publication["scientific_values_emitted"])

    def test_failure_terminal_is_publishable_from_each_nonterminal(self) -> None:
        for prefix_length in range(1, 10):
            with self.subTest(prefix_length=prefix_length):
                with tempfile.TemporaryDirectory() as temporary:
                    pending, final = telemetry_paths(Path(temporary))
                    writer = resource.AppendOnlyTelemetryWriter(
                        pending,
                        final,
                        attempt_identity_sha256=TEST_ATTEMPT_ID,
                    )
                    monotonic = 10
                    for index, event in enumerate(
                        resource.SUCCESS_MILESTONES[:prefix_length]
                    ):
                        writer.append(
                            event_code=event,
                            utc_time=utc(index),
                            monotonic_ns=monotonic,
                            metrics=telemetry_metrics(),
                        )
                        monotonic += 1
                    writer.append(
                        event_code="FAILURE_DETECTED",
                        utc_time=utc(prefix_length),
                        monotonic_ns=monotonic,
                        metrics=telemetry_metrics(),
                    )
                    receipt = writer.close()
                    verified = resource.verify_telemetry_publication(
                        pending_path=pending,
                        final_path=final,
                        expected_attempt_identity_sha256=TEST_ATTEMPT_ID,
                        receipt=receipt,
                    )
                self.assertEqual(
                    verified["terminal_state"],
                    "RESOURCE_MONITORING_FAILED",
                )

    def test_writer_requires_exact_attempt_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pending, final = telemetry_paths(Path(temporary))
            for invalid in ("x" * 64, "0" * 63, True, None):
                with self.subTest(invalid=invalid):
                    with self.assertRaises(
                        resource.Gate12C2ResourceQualificationError
                    ):
                        resource.AppendOnlyTelemetryWriter(
                            pending,
                            final,
                            attempt_identity_sha256=invalid,
                        )
            self.assertFalse(pending.exists())
            self.assertFalse(final.exists())

    def test_writer_requires_distinct_frozen_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pending, final = telemetry_paths(directory)
            for first, second in (
                (final, pending),
                (pending, pending),
                (directory / "other.pending", final),
            ):
                with self.subTest(first=first.name, second=second.name):
                    with self.assertRaises(
                        resource.Gate12C2ResourceQualificationError
                    ):
                        resource.AppendOnlyTelemetryWriter(
                            first,
                            second,
                            attempt_identity_sha256=TEST_ATTEMPT_ID,
                        )

    def test_writer_refuses_any_preexisting_pending_or_final(self) -> None:
        for existing in ("pending", "final"):
            with self.subTest(existing=existing):
                with tempfile.TemporaryDirectory() as temporary:
                    pending, final = telemetry_paths(Path(temporary))
                    target = pending if existing == "pending" else final
                    target.write_bytes(b"existing")
                    with self.assertRaises(FileExistsError):
                        resource.AppendOnlyTelemetryWriter(
                            pending,
                            final,
                            attempt_identity_sha256=TEST_ATTEMPT_ID,
                        )

    def test_context_exception_leaves_pending_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pending, final = telemetry_paths(Path(temporary))
            with self.assertRaisesRegex(RuntimeError, "caller failure"):
                with resource.AppendOnlyTelemetryWriter(
                    pending,
                    final,
                    attempt_identity_sha256=TEST_ATTEMPT_ID,
                ) as writer:
                    writer.append(
                        event_code="WRAPPER_AUTHORIZATION_CONSUMED",
                        utc_time=utc(0),
                        monotonic_ns=0,
                        metrics=telemetry_metrics(),
                    )
                    raise RuntimeError("caller failure")
            self.assertTrue(pending.is_file())
            self.assertFalse(final.exists())

    def test_close_before_terminal_never_creates_final(self) -> None:
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
            with self.assertRaisesRegex(
                resource.Gate12C2ResourceQualificationError,
                "before a terminal",
            ):
                writer.close()
            self.assertTrue(pending.is_file())
            self.assertFalse(final.exists())

    def test_publication_receipt_is_closed_and_tamper_evident(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _, receipt = write_success_telemetry(directory)
            pending, final = telemetry_paths(directory)
            attacked = dict(receipt)
            attacked["byte_count"] += 1
            with self.assertRaisesRegex(
                resource.Gate12C2ResourceQualificationError,
                "digest mismatch",
            ):
                resource.verify_telemetry_publication(
                    pending_path=pending,
                    final_path=final,
                    expected_attempt_identity_sha256=TEST_ATTEMPT_ID,
                    receipt=attacked,
                )
            attacked = dict(receipt)
            attacked["unknown"] = 1
            with self.assertRaisesRegex(
                resource.Gate12C2ResourceQualificationError,
                "schema",
            ):
                resource.verify_telemetry_publication(
                    pending_path=pending,
                    final_path=final,
                    expected_attempt_identity_sha256=TEST_ATTEMPT_ID,
                    receipt=attacked,
                )

    def test_strict_decoder_rejects_digest_canonical_and_type_attacks(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payload, _ = write_success_telemetry(Path(temporary))
        lines = payload.splitlines(keepends=True)
        first = json.loads(lines[0])
        attacked = dict(first)
        attacked["sha"] = "0" * 64
        candidate = (
            resource.canonical_json_bytes(attacked)
            + b"\n"
            + b"".join(lines[1:])
        )
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "digest mismatch",
        ):
            resource.decode_and_verify_telemetry(candidate)
        noncanonical = (
            json.dumps(first, sort_keys=True).encode("utf-8")
            + b"\n"
            + b"".join(lines[1:])
        )
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "canonical",
        ):
            resource.decode_and_verify_telemetry(noncanonical)
        with self.assertRaises(
            resource.Gate12C2ResourceQualificationError
        ):
            resource.decode_and_verify_telemetry(
                lines[0].replace(b'"seq":0', b'"seq":true')
                + b"".join(lines[1:])
            )


class MemoryGeometryTest(unittest.TestCase):
    def test_page_aligned_geometry_is_exact(self) -> None:
        physical = 16 * 1024**3 + 123
        page = 4096
        result = resource.derive_resource_memory_geometry(physical, page)
        mathematical = (3 * physical) // 4
        effective = (mathematical // page) * page
        self.assertEqual(result.physical_ram_bytes, physical)
        self.assertEqual(result.native_page_size_bytes, page)
        self.assertEqual(result.mathematical_memory_limit_bytes, mathematical)
        self.assertEqual(result.effective_job_memory_limit_bytes, effective)
        self.assertEqual(
            result.rounding_delta_bytes, mathematical - effective
        )
        self.assertLessEqual(effective, mathematical)

    def test_direct_geometry_construction_is_forbidden(self) -> None:
        with self.assertRaises(
            resource.Gate12C2ResourceQualificationError
        ):
            resource.ResourceMemoryGeometry()

    def test_invalid_physical_or_page_size_rejects(self) -> None:
        for physical, page in (
            (0, 4096),
            (1024, 0),
            (1024, 3000),
            (1024, True),
            ("1024", 4096),
        ):
            with self.subTest(physical=physical, page=page):
                with self.assertRaises(
                    resource.Gate12C2ResourceQualificationError
                ):
                    resource.derive_resource_memory_geometry(
                        physical, page  # type: ignore[arg-type]
                    )

    def test_independent_geometry_match_passes(self) -> None:
        first = resource.derive_resource_memory_geometry(10_000_000, 4096)
        second = resource.derive_resource_memory_geometry(10_000_000, 4096)
        result = resource.verify_resource_memory_geometry_match(
            first, second
        )
        self.assertEqual(result["status"], "pass")
        self.assertEqual(
            result["effective_job_memory_limit_bytes"],
            first.effective_job_memory_limit_bytes,
        )

    def test_independent_geometry_mismatch_rejects(self) -> None:
        first = resource.derive_resource_memory_geometry(10_000_000, 4096)
        second = resource.derive_resource_memory_geometry(10_004_096, 4096)
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "differ",
        ):
            resource.verify_resource_memory_geometry_match(first, second)


class AtomicWindowsLaunchTest(unittest.TestCase):
    def test_frozen_x64_win32_struct_layouts_are_exact(self) -> None:
        self.assertEqual(ctypes.sizeof(ctypes.c_void_p), 8)
        expected = {
            resource._STARTUPINFOW: 104,
            resource._STARTUPINFOEXW: 112,
            resource._PROCESS_INFORMATION: 24,
            resource._SYSTEM_INFO: 48,
            resource._MEMORYSTATUSEX: 64,
            resource._JOBOBJECT_BASIC_LIMIT_INFORMATION: 64,
            resource._JOBOBJECT_EXTENDED_LIMIT_INFORMATION: 144,
            resource._JOBOBJECT_BASIC_ACCOUNTING_INFORMATION: 48,
        }
        for structure, expected_size in expected.items():
            with self.subTest(structure=structure.__name__):
                self.assertEqual(ctypes.sizeof(structure), expected_size)

    def test_support_probe_exercises_attribute_protocol_without_child(
        self,
    ) -> None:
        kernel = FakeKernel32()
        result = job_api(kernel).probe_job_list_attribute_support()
        self.assertEqual(result["status"], "pass")
        self.assertFalse(result["scientific_child_created"])
        self.assertEqual(kernel.delete_calls, 1)
        self.assertEqual(len(kernel.update_calls), 1)
        self.assertNotIn("CreateProcessW", kernel.calls)
        self.assertNotIn(kernel.job_handle, kernel.open_handles)

    def test_atomic_launch_uses_exact_createprocess_abi_and_order(
        self,
    ) -> None:
        launch, _, kernel = make_launch()
        call = kernel.create_process_calls[0]
        update = kernel.update_calls[0]
        self.assertFalse(call["inherit_handles"])
        self.assertEqual(
            call["flags"], resource.WindowsJobApi.CREATEPROCESS_FLAGS
        )
        self.assertEqual(
            call["startup_cb"], ctypes.sizeof(resource._STARTUPINFOEXW)
        )
        self.assertNotEqual(call["attribute_pointer"], 0)
        self.assertEqual(update["flags"], 0)
        self.assertEqual(
            update["attribute"],
            resource.WindowsJobApi.PROC_THREAD_ATTRIBUTE_JOB_LIST,
        )
        self.assertEqual(update["job_value"], kernel.job_handle)
        self.assertEqual(
            update["value_bytes"], ctypes.sizeof(ctypes.wintypes.HANDLE)
        )
        self.assertEqual(update["previous_is_null"], 1)
        self.assertEqual(update["returned_is_null"], 1)
        self.assertLess(
            kernel.calls.index("CreateProcessW"),
            kernel.calls.index("DeleteProcThreadAttributeList"),
        )
        self.assertEqual(kernel.delete_calls, 1)
        self.assertEqual(launch.attribute_bytes, kernel.first_attribute_bytes)
        self.assertEqual(
            launch.child_identity,
            resource.ProcessIdentity(
                kernel.process_id,
                (
                    kernel.creation_ticks
                    - resource.WINDOWS_TO_UNIX_EPOCH_100NS
                )
                * 100,
            ),
        )

    def test_job_limit_is_page_aligned_J_not_mathematical_M(self) -> None:
        kernel = FakeKernel32()
        launch, _, _ = make_launch(kernel)
        memory = launch.resource_geometry
        self.assertEqual(
            kernel.configured_limit,
            memory.effective_job_memory_limit_bytes,
        )
        self.assertEqual(kernel.configured_flags, kernel.REQUIRED_FLAGS)
        self.assertLessEqual(
            kernel.configured_limit,
            memory.mathematical_memory_limit_bytes,
        )

    def test_production_api_has_no_raw_job_transfer_surface(self) -> None:
        signature = inspect.signature(
            resource.WindowsJobApi.launch_scientific_child_suspended
        )
        self.assertNotIn("job_handle", signature.parameters)
        self.assertNotIn("expected_job_memory_limit_bytes", signature.parameters)
        for forbidden in (
            "assign_process",
            "transfer_job_handle_to_watchdog",
            "open_job",
            "duplicate_job_handle",
        ):
            self.assertFalse(hasattr(resource.WindowsJobApi, forbidden))
        source = inspect.getsource(resource.WindowsJobApi)
        self.assertNotIn("AssignProcessToJobObject", source)
        self.assertNotIn("DuplicateHandle", source)
        self.assertNotIn("OpenJobObject", source)

    def test_watchdog_already_in_job_rejects_before_job_creation(self) -> None:
        kernel = FakeKernel32(outside_job=False)
        api = job_api(kernel)
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "already assigned",
        ):
            api.launch_scientific_child_suspended(
                preflight_geometry=geometry(kernel),
                application_name="python.exe",
                command_line="python.exe -B frozen.py",
                current_directory=Path.cwd(),
            )
        self.assertNotIn("CreateJobObjectW", kernel.calls)
        self.assertNotIn("CreateProcessW", kernel.calls)

    def test_preflight_watchdog_geometry_mismatch_rejects_before_child(
        self,
    ) -> None:
        kernel = FakeKernel32()
        api = job_api(kernel)
        wrong = resource.derive_resource_memory_geometry(
            kernel.physical_ram_bytes + kernel.page_size_bytes,
            kernel.page_size_bytes,
        )
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "differ",
        ):
            api.launch_scientific_child_suspended(
                preflight_geometry=wrong,
                application_name="python.exe",
                command_line="python.exe -B frozen.py",
                current_directory=Path.cwd(),
            )
        self.assertNotIn("CreateJobObjectW", kernel.calls)

    def test_createprocess_failure_deletes_list_and_closes_all_handles(
        self,
    ) -> None:
        kernel = FakeKernel32(
            create_process_succeeds=False,
            residual_process_information_on_failure=True,
        )
        api = job_api(kernel)
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "CreateProcessW failed",
        ):
            api.launch_scientific_child_suspended(
                preflight_geometry=geometry(kernel),
                application_name="python.exe",
                command_line="python.exe -B frozen.py",
                current_directory=Path.cwd(),
            )
        self.assertEqual(kernel.delete_calls, 1)
        self.assertEqual(kernel.open_handles, set())

    def test_attribute_update_failure_closes_job_without_child(self) -> None:
        kernel = FakeKernel32(update_succeeds=False)
        api = job_api(kernel)
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "launch preparation failed",
        ):
            api.launch_scientific_child_suspended(
                preflight_geometry=geometry(kernel),
                application_name="python.exe",
                command_line="python.exe -B frozen.py",
                current_directory=Path.cwd(),
            )
        self.assertEqual(kernel.delete_calls, 1)
        self.assertNotIn("CreateProcessW", kernel.calls)
        self.assertEqual(kernel.open_handles, set())

    def test_membership_failure_kills_job_and_closes_child_handles(
        self,
    ) -> None:
        kernel = FakeKernel32(child_in_job=False)
        api = job_api(kernel)
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "not atomically contained",
        ):
            api.launch_scientific_child_suspended(
                preflight_geometry=geometry(kernel),
                application_name="python.exe",
                command_line="python.exe -B frozen.py",
                current_directory=Path.cwd(),
            )
        self.assertEqual(kernel.open_handles, set())

    def test_suspend_probe_restores_one_and_resume_runs_once(self) -> None:
        launch, _, kernel = make_launch()
        self.assertEqual(kernel.suspend_count, 1)
        previous = launch.resume_suspended_child()
        self.assertEqual(previous, 1)
        self.assertEqual(kernel.suspend_count, 0)
        with self.assertRaises(
            resource.Gate12C2ResourceQualificationError
        ):
            launch.resume_suspended_child()


    def test_top_level_production_entry_constructs_real_api_internally(
        self,
    ) -> None:
        signature = inspect.signature(
            resource.create_watchdog_local_scientific_launch
        )
        self.assertEqual(
            set(signature.parameters),
            {
                "preflight_geometry",
                "application_name",
                "command_line",
                "current_directory",
            },
        )
        source = inspect.getsource(
            resource.create_watchdog_local_scientific_launch
        )
        self.assertIn("api = WindowsJobApi()", source)
        self.assertNotIn("job_handle", signature.parameters)
        self.assertNotIn("api", signature.parameters)


class LaunchDeadlineWatchdogTest(unittest.TestCase):
    def test_deadline_minus_one_nanosecond_accepts(self) -> None:
        launch, _, kernel = make_launch()
        start = 1_000
        watchdog = resource.LaunchDeadlineWatchdog(
            launch,
            monotonic_ns=Clock(
                start,
                start + 1,
                start + resource.LAUNCH_EVIDENCE_DEADLINE_NS - 1,
            ),
        )
        watchdog.resume_and_arm()
        ack = watchdog.run_until_verified(
            acknowledgement_supplier=lambda: {"status": "present"},
            verifier=lambda _: True,
        )
        self.assertEqual(
            ack, start + resource.LAUNCH_EVIDENCE_DEADLINE_NS - 1
        )
        self.assertTrue(watchdog.verified)
        self.assertIn(kernel.job_handle, kernel.open_handles)

    def test_deadline_exact_kills_job(self) -> None:
        launch, _, kernel = make_launch()
        start = 1_000
        watchdog = resource.LaunchDeadlineWatchdog(
            launch,
            monotonic_ns=Clock(
                start,
                start + resource.LAUNCH_EVIDENCE_DEADLINE_NS,
            ),
        )
        watchdog.resume_and_arm()
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "missing at deadline",
        ):
            watchdog.run_until_verified(
                acknowledgement_supplier=lambda: None,
                verifier=lambda _: True,
            )
        self.assertNotIn(kernel.job_handle, kernel.open_handles)

    def test_invalid_acknowledgement_kills_job(self) -> None:
        launch, _, kernel = make_launch()
        watchdog = resource.LaunchDeadlineWatchdog(
            launch, monotonic_ns=Clock(10, 11, 12, 13)
        )
        watchdog.resume_and_arm()
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "invalid",
        ):
            watchdog.run_until_verified(
                acknowledgement_supplier=lambda: {"bad": True},
                verifier=lambda _: False,
            )
        self.assertNotIn(kernel.job_handle, kernel.open_handles)

    def test_unreviewed_object_is_not_accepted(self) -> None:
        with self.assertRaisesRegex(
            resource.Gate12C2ResourceQualificationError,
            "type",
        ):
            resource.LaunchDeadlineWatchdog(object())  # type: ignore[arg-type]


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
    def legacy_evidence() -> dict[str, object]:
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

    def test_exact_legacy_failure_is_not_child_success(self) -> None:
        result = resource.classify_expected_legacy_closeout(
            self.legacy_evidence()
        )
        self.assertEqual(
            result["status"],
            "REPLAY_PAYLOAD_COMPLETE_WITH_EXPECTED_LEGACY_CLOSEOUT_FAILURE",
        )
        self.assertFalse(result["legacy_child_success_claimed"])
        self.assertFalse(result["scientific_values_emitted"])

    def test_any_legacy_surface_deviation_rejects(self) -> None:
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
            evidence = self.legacy_evidence()
            evidence[field] = value
            with self.subTest(field=field):
                with self.assertRaises(
                    resource.Gate12C2ResourceQualificationError
                ):
                    resource.classify_expected_legacy_closeout(evidence)


class ResourceEnvelopeTest(unittest.TestCase):
    @staticmethod
    def passing_evidence() -> dict[str, int]:
        physical = 16 * 1024**3 + 123
        page = 4096
        geometry_value = resource.derive_resource_memory_geometry(
            physical, page
        )
        preflight = 12 * 1024**3
        return {
            "physical_ram_bytes": physical,
            "native_page_size_bytes": page,
            "mathematical_memory_limit_bytes": (
                geometry_value.mathematical_memory_limit_bytes
            ),
            "effective_job_memory_limit_bytes": (
                geometry_value.effective_job_memory_limit_bytes
            ),
            "rounding_delta_bytes": geometry_value.rounding_delta_bytes,
            "peak_job_memory_bytes": (
                geometry_value.effective_job_memory_limit_bytes
            ),
            "sampled_combined_rss_bytes": (
                geometry_value.mathematical_memory_limit_bytes
            ),
            "sampled_available_physical_memory_bytes": physical // 10,
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

    def test_exact_boundaries_pass_with_distinct_M_and_J(self) -> None:
        evidence = self.passing_evidence()
        result = resource.verify_resource_envelope(evidence)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(
            result["mathematical_memory_limit_bytes"],
            evidence["mathematical_memory_limit_bytes"],
        )
        self.assertEqual(
            result["effective_job_memory_limit_bytes"],
            evidence["effective_job_memory_limit_bytes"],
        )
        self.assertEqual(
            result["original_resource_gate_status"], "indeterminate"
        )

    def test_each_one_unit_breach_rejects(self) -> None:
        base = self.passing_evidence()
        mutations = (
            (
                "peak_job_memory_bytes",
                base["effective_job_memory_limit_bytes"] + 1,
            ),
            (
                "sampled_combined_rss_bytes",
                base["mathematical_memory_limit_bytes"] + 1,
            ),
            (
                "sampled_available_physical_memory_bytes",
                base["physical_ram_bytes"] // 10 - 1,
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
        for field, value in mutations:
            evidence = dict(base)
            evidence[field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    resource.Gate12C2ResourceQualificationError,
                    "frozen envelope",
                ):
                    resource.verify_resource_envelope(evidence)

    def test_any_P_S_M_J_mismatch_rejects(self) -> None:
        base = self.passing_evidence()
        for field in (
            "mathematical_memory_limit_bytes",
            "effective_job_memory_limit_bytes",
            "rounding_delta_bytes",
        ):
            evidence = dict(base)
            evidence[field] += 1
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    resource.Gate12C2ResourceQualificationError,
                    "commitment mismatch",
                ):
                    resource.verify_resource_envelope(evidence)

    def test_resource_schema_and_types_are_closed(self) -> None:
        for mutation in ("bool", "string", "extra", "missing"):
            evidence: dict[str, object] = dict(self.passing_evidence())
            if mutation == "bool":
                evidence["wall_seconds"] = True
            elif mutation == "string":
                evidence["wall_seconds"] = str(
                    resource.MAXIMUM_WALL_SECONDS
                )
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
