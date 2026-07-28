#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import gate12c2_closeout_recovery as recovery  # noqa: E402
import gate12c2_draw_profile as profile  # noqa: E402


class Gate12C2CloseoutRecoveryTest(unittest.TestCase):
    NOW = "2026-07-28T00:00:00+00:00"

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.root = self.base / "profile"
        self._create_byte_fixture(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _create_byte_fixture(root: Path) -> None:
        for relative in recovery.expected_root_files():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((relative + "\n").encode("utf-8"))

    def _manifest(self) -> dict[str, object]:
        return recovery.build_incident_manifest(
            self.root,
            incident_id="closeout-test",
            observed_at_utc=self.NOW,
        )

    def _write_mapping(self, name: str, payload: dict[str, object]) -> Path:
        path = self.base / name
        path.write_bytes(recovery.canonical_json_bytes(payload))
        return path

    @staticmethod
    def _hashed(payload: dict[str, object], field: str) -> dict[str, object]:
        result = dict(payload)
        result[field] = recovery.sha256_bytes(
            recovery.canonical_json_bytes(result)
        )
        return result

    def test_expected_surface_is_exact(self) -> None:
        expected = recovery.expected_root_files()
        self.assertEqual(len(expected), 791)
        self.assertEqual(sum(row[1] == "shard" for row in expected.values()), 768)
        self.assertEqual(sum(row[1] == "index" for row in expected.values()), 9)
        self.assertEqual(
            sum(row[1] == "frozen_lineage" for row in expected.values()), 13
        )
        self.assertEqual(
            sum(row[0] == "protected_payload" for row in expected.values()), 790
        )

    def test_byte_only_manifest_freezes_exact_surface(self) -> None:
        manifest = self._manifest()
        self.assertEqual(manifest["state"], "INCIDENT_FROZEN")
        self.assertEqual(manifest["payload_presence_observed"], "768/768")
        self.assertEqual(manifest["payload_integrity_status"], "pending")
        self.assertEqual(manifest["resource_gate_status"], "indeterminate")
        self.assertFalse(manifest["stability_analysis_authorized"])
        self.assertEqual(
            recovery.verify_incident_manifest(manifest, root=self.root), manifest
        )
        encoded = json.dumps(manifest, sort_keys=True)
        for prohibited in (
            "median_log_ratio",
            "q_directional_support",
            "scientific_projection",
        ):
            self.assertNotIn(prohibited, encoded)

    def test_byte_manifest_rejects_missing_and_unexpected_files(self) -> None:
        missing = self.root / (
            "runs/S0_true_null/draw-255/shards/outer-000000.json.gz"
        )
        missing.unlink()
        manifest = self._manifest()
        self.assertEqual(manifest["state"], "RECOVERY_REJECTED")
        self.assertEqual(manifest["summary"]["missing_expected_count"], 1)
        missing.write_bytes(b"restored")
        unexpected = self.root / ".receipt.1.tmp"
        unexpected.write_bytes(b"partial")
        manifest = self._manifest()
        self.assertEqual(manifest["state"], "RECOVERY_REJECTED")
        self.assertEqual(manifest["summary"]["unexpected_file_count"], 1)
        self.assertEqual(manifest["summary"]["partial_or_temp_count"], 1)

    def test_manifest_verification_detects_byte_mutation(self) -> None:
        manifest = self._manifest()
        target = self.root / "runs/S2_null_inflation/draw-1023/index.json"
        target.write_bytes(target.read_bytes() + b"mutation")
        with self.assertRaisesRegex(
            recovery.Gate12C2CloseoutRecoveryError,
            "differs from the current byte surface",
        ):
            recovery.verify_incident_manifest(manifest, root=self.root)

    def test_reparse_point_is_not_accepted(self) -> None:
        original = recovery._reparse_status

        def reparse(path: Path) -> bool:
            if path.name == recovery.LOCK_NAME:
                return True
            return original(path)

        with mock.patch.object(recovery, "_reparse_status", side_effect=reparse):
            manifest = self._manifest()
        self.assertEqual(manifest["state"], "RECOVERY_REJECTED")
        self.assertEqual(manifest["summary"]["reparse_file_count"], 1)


    def _incident_packet(self) -> dict[str, Path | dict[str, object]]:
        manifest = self._manifest()
        manifest_path = self._write_mapping("manifest.json", manifest)
        stdout_path = self.base / "stdout.log"
        stderr_path = self.base / "stderr.log"
        stdout_path.write_bytes(b"")
        stderr_path.write_bytes(b"sanitized failure bytes")
        failure = recovery.build_failure_receipt(
            incident_manifest_path=manifest_path,
            stdout_log_path=stdout_path,
            stderr_log_path=stderr_path,
            runner_pid=2147483647,
            observed_at_utc=self.NOW,
        )
        failure_path = self._write_mapping("failure.json", failure)
        exposure = recovery.build_exposure_ledger(
            incident_id="closeout-test",
            reviewer_context_id="exposed-reviewer",
            recorded_at_utc=self.NOW,
        )
        exposure_path = self._write_mapping("exposure.json", exposure)
        return {
            "manifest": manifest,
            "manifest_path": manifest_path,
            "failure": failure,
            "failure_path": failure_path,
            "exposure": exposure,
            "exposure_path": exposure_path,
        }

    def test_failure_receipt_records_bytes_not_log_content(self) -> None:
        packet = self._incident_packet()
        failure = packet["failure"]
        self.assertEqual(
            failure["failure_code"], recovery.INCIDENT_FAILURE_CODE
        )
        self.assertEqual(failure["resource_gate_status"], "indeterminate")
        self.assertFalse(failure["normal_resume_permitted"])
        encoded = json.dumps(failure, sort_keys=True)
        self.assertNotIn("sanitized failure bytes", encoded)
        self.assertTrue(all(not row["content_inspected"] for row in failure["logs"]))

    def test_exposure_ledger_retires_only_blinded_selector_role(self) -> None:
        ledger = recovery.build_exposure_ledger(
            incident_id="closeout-test",
            reviewer_context_id="reviewer-a",
            recorded_at_utc=self.NOW,
        )
        self.assertEqual(ledger["engineering_review_eligibility"], "retained")
        self.assertEqual(
            ledger["scientific_selector_blinded_eligibility"], "lost"
        )
        self.assertEqual(ledger["draw_selector_blinded_eligibility"], "lost")
        self.assertFalse(ledger["scientific_values_interpreted"])

    def test_amendment_keeps_resource_gate_indeterminate(self) -> None:
        packet = self._incident_packet()
        amendment = recovery.build_recovery_amendment(
            incident_manifest_path=packet["manifest_path"],
            failure_receipt_path=packet["failure_path"],
            exposure_ledger_path=packet["exposure_path"],
            amendment_id="closeout-amendment-test",
            recorded_at_utc=self.NOW,
        )
        self.assertEqual(
            recovery.verify_recovery_amendment(amendment), amendment
        )
        self.assertEqual(amendment["resource_gate_status"], "indeterminate")
        self.assertEqual(
            amendment["replacement_resource_qualification"], "not_performed"
        )
        self.assertTrue(
            amendment["replacement_qualification_cannot_rewrite_original_gate"]
        )
        self.assertFalse(
            amendment["control_plane_contract"][
                "lock_retirement_authorized_by_this_amendment"
            ]
        )
        self.assertFalse(amendment["stability_analysis_authorized"])


    def _legacy_fixture(self) -> tuple[Path, str, str]:
        source_commit = "a" * 40
        shard_hash = recovery.sha256_file(Path(recovery.shards.__file__))
        lab_hash = recovery.sha256_file(Path(recovery.shards.lab.__file__))
        plan: dict[str, object] = {
            "source_commit": source_commit,
            "implementation_sha256": {
                "gate12c2_development_shards.py": shard_hash,
                "gate12c2_synthetic_lab.py": lab_hash,
            },
            "configurations": [],
        }
        plan["draw_profile_plan_payload_sha256"] = recovery.sha256_bytes(
            recovery.canonical_json_bytes(plan)
        )
        plan_hash = str(plan["draw_profile_plan_payload_sha256"])
        archived = self._write_mapping("archived-plan.json", plan)
        (self.root / "plan.json").write_bytes(
            recovery.canonical_json_bytes(plan)
        )
        preflight = self._hashed(
            {"draw_profile_plan_payload_sha256": plan_hash},
            "preflight_receipt_payload_sha256",
        )
        authorization = self._hashed(
            {
                "draw_profile_plan_payload_sha256": plan_hash,
                "preflight_receipt_payload_sha256": preflight[
                    "preflight_receipt_payload_sha256"
                ],
                "output_root": self.root.resolve().as_posix(),
                "single_use": True,
            },
            "authorization_receipt_payload_sha256",
        )
        consumption = self._hashed(
            {
                "draw_profile_plan_payload_sha256": plan_hash,
                "authorization_receipt_payload_sha256": authorization[
                    "authorization_receipt_payload_sha256"
                ],
                "output_root": self.root.resolve().as_posix(),
                "single_use": True,
                "authorization_status": "consumed_for_this_execution_lineage",
            },
            "consumption_receipt_payload_sha256",
        )
        (self.root / "control/preflight.json").write_bytes(
            recovery.canonical_json_bytes(preflight)
        )
        (self.root / "control/authorization.json").write_bytes(
            recovery.canonical_json_bytes(authorization)
        )
        (self.root / "control/authorization-consumed.json").write_bytes(
            recovery.canonical_json_bytes(consumption)
        )
        lock = self._hashed(
            {
                "plan_payload_sha256": plan_hash,
                "implementation_sha256": plan["implementation_sha256"],
                "pid": 2147483647,
                "hostname": "test",
            },
            "lock_payload_sha256",
        )
        (self.root / recovery.LOCK_NAME).write_bytes(
            recovery.canonical_json_bytes(lock)
        )
        return archived, source_commit, plan_hash

    def test_legacy_lineage_uses_old_blob_identity(self) -> None:
        archived, source_commit, plan_hash = self._legacy_fixture()
        shard_hash = recovery.sha256_file(Path(recovery.shards.__file__))
        def legacy_blob(_: str, relative: str) -> str:
            return recovery.sha256_file(TOOLS_DIR / Path(relative).name)

        with mock.patch.object(
            recovery, "git_blob_sha256", side_effect=legacy_blob
        ), mock.patch.object(
            recovery,
            "_query_process_identity",
            return_value=(recovery.PROCESS_DEAD, None),
        ), mock.patch.object(
            profile,
            "_pid_is_running",
            side_effect=AssertionError("legacy boolean liveness was used"),
        ):
            evidence = recovery.verify_legacy_lineage(
                output_root=self.root,
                archived_plan_path=archived,
                expected_source_commit=source_commit,
                expected_plan_payload_sha256=plan_hash,
            )
        self.assertEqual(evidence["original_source_commit"], source_commit)
        self.assertTrue(evidence["stale_lock_owner_not_running"])

    def test_legacy_lineage_rejects_indeterminate_lock_owner(self) -> None:
        archived, source_commit, plan_hash = self._legacy_fixture()

        def legacy_blob(_: str, relative: str) -> str:
            return recovery.sha256_file(TOOLS_DIR / Path(relative).name)

        with mock.patch.object(
            recovery, "git_blob_sha256", side_effect=legacy_blob
        ), mock.patch.object(
            recovery,
            "_query_process_identity",
            return_value=(recovery.PROCESS_UNKNOWN, None),
        ) as tristate_probe, mock.patch.object(
            profile,
            "_pid_is_running",
            side_effect=AssertionError("legacy boolean liveness was used"),
        ):
            with self.assertRaisesRegex(
                recovery.Gate12C2CloseoutRecoveryError,
                "not definitively dead",
            ):
                recovery.verify_legacy_lineage(
                    output_root=self.root,
                    archived_plan_path=archived,
                    expected_source_commit=source_commit,
                    expected_plan_payload_sha256=plan_hash,
                )
        tristate_probe.assert_called_once_with(2147483647)

    def test_legacy_lineage_rejects_active_lock_owner(self) -> None:
        archived, source_commit, plan_hash = self._legacy_fixture()

        def legacy_blob(_: str, relative: str) -> str:
            return recovery.sha256_file(TOOLS_DIR / Path(relative).name)

        with mock.patch.object(
            recovery, "git_blob_sha256", side_effect=legacy_blob
        ), mock.patch.object(
            recovery,
            "_query_process_identity",
            return_value=(
                recovery.PROCESS_ACTIVE,
                {
                    "pid": 2147483647,
                    "identity_kind": "fixture",
                    "start_marker": "active",
                },
            ),
        ), mock.patch.object(
            profile,
            "_pid_is_running",
            side_effect=AssertionError("legacy boolean liveness was used"),
        ):
            with self.assertRaisesRegex(
                recovery.Gate12C2CloseoutRecoveryError,
                "not definitively dead",
            ):
                recovery.verify_legacy_lineage(
                    output_root=self.root,
                    archived_plan_path=archived,
                    expected_source_commit=source_commit,
                    expected_plan_payload_sha256=plan_hash,
                )

    def test_legacy_lineage_rejects_changed_shard_verifier(self) -> None:
        archived, source_commit, plan_hash = self._legacy_fixture()
        shard_hash = recovery.sha256_file(Path(recovery.shards.__file__))
        original_sha256_file = recovery.sha256_file

        def changed_current_verifier(path: Path) -> str:
            if Path(path).resolve() == Path(recovery.shards.__file__).resolve():
                return "b" * 64
            return original_sha256_file(path)

        legacy_hashes = {
            "gate12c2_development_shards.py": shard_hash,
            "gate12c2_synthetic_lab.py": original_sha256_file(
                Path(recovery.shards.lab.__file__)
            ),
        }

        def legacy_blob(_: str, relative: str) -> str:
            return legacy_hashes[Path(relative).name]

        with mock.patch.object(
            recovery, "git_blob_sha256", side_effect=legacy_blob
        ), mock.patch.object(
            recovery, "sha256_file", side_effect=changed_current_verifier
        ), mock.patch.object(
            recovery,
            "_query_process_identity",
            return_value=(recovery.PROCESS_DEAD, None),
        ):
            with self.assertRaisesRegex(
                recovery.Gate12C2CloseoutRecoveryError,
                "current semantic verifier dependency",
            ):
                recovery.verify_legacy_lineage(
                    output_root=self.root,
                    archived_plan_path=archived,
                    expected_source_commit=source_commit,
                    expected_plan_payload_sha256=plan_hash,
                )


    def test_semantic_verifier_never_executes_or_rewrites_payload(self) -> None:
        configurations = []
        for regime in recovery.REGIME_OUTER_COUNTS:
            for draw_count in recovery.DRAW_COUNTS:
                relative = f"runs/{regime}/draw-{draw_count}"
                subplan = {
                    "plan_payload_sha256": "a" * 64,
                    "outer_experiment_indices": [0],
                }
                (self.root / relative / "plan.json").write_bytes(
                    recovery.canonical_json_bytes(subplan)
                )
                configurations.append(
                    {
                        "configuration_id": f"{regime}__d{draw_count}",
                        "output_relative_path": relative,
                        "subplan": subplan,
                    }
                )
        manifest = self._manifest()
        archived = self.base / "semantic-plan.json"
        archived.write_bytes(b"not-read-because-mocked")
        plan = {"configurations": configurations}
        result = {
            "outer_experiment_count": 1,
            "plan_payload_sha256": "a" * 64,
            "index_payload_sha256": "b" * 64,
            "scientific_projection_sha256": "c" * 64,
        }
        with mock.patch.object(
            recovery, "verify_legacy_lineage", return_value={"lineage": "ok"}
        ), mock.patch.object(
            recovery,
            "_verify_canonical_mapping_file",
            return_value=plan,
        ), mock.patch.object(
            recovery.shards, "_verified_plan", return_value=subplan
        ), mock.patch.object(
            recovery.shards,
            "verify_development_shard_index",
            return_value=result,
        ), mock.patch.object(
            recovery.shards,
            "execute_development_shard_plan",
        ) as forbidden_execute:
            verification = recovery.verify_payload_semantics(
                output_root=self.root,
                archived_plan_path=archived,
                incident_manifest=manifest,
            )
        forbidden_execute.assert_not_called()
        self.assertEqual(verification["shard_count"], 768)
        self.assertEqual(verification["index_count"], 9)
        self.assertEqual(verification["payload_modified"], 0)
        self.assertFalse(verification["scientific_values_emitted"])

    def _mock_authorization(self) -> tuple[Path, dict[str, object]]:
        authorization_path = self.base / "recovery-auth.json"
        attempt = self.base / "external/attempt.json"
        consumption = self.base / "external/consumption.json"
        terminal = self.base / "external/terminal.json"
        seal = self.base / "external/seal.json"
        failure = self.base / "external/failure.json"
        authorization = {
            "authorization_id": "test-auth",
            "authorization_scope": "payload_verification_and_external_seal_only",
            "authorization_payload_sha256": "a" * 64,
            "issued_at_utc": "2026-07-27T00:00:00+00:00",
            "expires_at_utc": "2026-07-29T00:00:00+00:00",
            "output_root": self.root.resolve().as_posix(),
            "incident_manifest_path": (self.base / "incident.json").as_posix(),
            "archived_plan_path": (self.base / "archived.json").as_posix(),
            "incident_manifest_payload_sha256": "b" * 64,
            "amendment_payload_sha256": "c" * 64,
            "authorization_output": authorization_path.as_posix(),
            "attempt_output": attempt.as_posix(),
            "consumption_output": consumption.as_posix(),
            "terminal_output": terminal.as_posix(),
            "seal_output": seal.as_posix(),
            "failure_output": failure.as_posix(),
        }
        authorization_path.write_bytes(
            recovery.canonical_json_bytes(authorization)
        )
        (self.base / "incident.json").write_text("{}", encoding="utf-8")
        return authorization_path, authorization

    @staticmethod
    def _semantic_stub() -> dict[str, object]:
        rows = []
        for regime, outer_count in recovery.REGIME_OUTER_COUNTS.items():
            for draw_count in recovery.DRAW_COUNTS:
                rows.append(
                    {
                        "configuration_id": f"{regime}__d{draw_count}",
                        "outer_experiment_count": outer_count,
                        "plan_payload_sha256": "a" * 64,
                        "index_payload_sha256": "b" * 64,
                        "scientific_projection_sha256": "c" * 64,
                        "status": "verified",
                    }
                )
        return {
            "schema_version": "gate12c2_payload_semantic_verification_v0.1",
            "status": "verified",
            "configuration_count": 9,
            "outer_experiment_count": 768,
            "shard_count": 768,
            "index_count": 9,
            "configuration_results": rows,
            "protected_surface_sha256": "d" * 64,
            "complete_surface_sha256": "e" * 64,
            "legacy_lineage_evidence_sha256": "f" * 64,
            "payload_added": 0,
            "payload_modified": 0,
            "payload_deleted": 0,
            "index_bytes_changed": 0,
            "scientific_values_emitted": False,
        }

    def test_payload_seal_is_external_and_lock_remains(self) -> None:
        auth_path, authorization = self._mock_authorization()
        lock_before = (self.root / recovery.LOCK_NAME).read_bytes()
        with mock.patch.object(
            recovery,
            "verify_recovery_authorization",
            return_value=authorization,
        ), mock.patch.object(
            recovery,
            "verify_payload_semantics",
            return_value=self._semantic_stub(),
        ):
            seal = recovery.execute_payload_seal(auth_path)
        self.assertEqual(seal["state"], "PAYLOAD_COMPLETION_SEALED")
        self.assertEqual(seal["resource_gate_status"], "indeterminate")
        self.assertFalse(seal["stability_analysis_authorized"])
        self.assertTrue(Path(authorization["consumption_output"]).is_file())
        self.assertTrue(Path(authorization["seal_output"]).is_file())
        self.assertEqual(
            (self.root / recovery.LOCK_NAME).read_bytes(), lock_before
        )

    def test_failed_recovery_consumes_authorization_and_requires_new_one(self) -> None:
        auth_path, authorization = self._mock_authorization()
        with mock.patch.object(
            recovery,
            "verify_recovery_authorization",
            return_value=authorization,
        ), mock.patch.object(
            recovery,
            "verify_payload_semantics",
            side_effect=recovery.Gate12C2CloseoutRecoveryError("sentinel"),
        ):
            with self.assertRaisesRegex(
                recovery.Gate12C2CloseoutRecoveryError, "was rejected"
            ):
                recovery.execute_payload_seal(auth_path)
            self.assertTrue(Path(authorization["consumption_output"]).is_file())
            self.assertTrue(Path(authorization["failure_output"]).is_file())
            failure = recovery.read_mapping(
                Path(authorization["failure_output"]), label="failure"
            )
            self.assertFalse(failure["authorization_reusable"])
            self.assertNotIn("sentinel", json.dumps(failure))
            with self.assertRaisesRegex(
                recovery.Gate12C2CloseoutRecoveryError,
                "already been terminally claimed",
            ):
                recovery.execute_payload_seal(auth_path)

    def test_published_consumption_is_recovered_after_writer_error(self) -> None:
        auth_path, authorization = self._mock_authorization()
        real_writer = recovery.write_exclusive_atomic
        consumption_path = Path(authorization["consumption_output"]).resolve()

        def publish_then_raise(path: Path, payload: dict[str, object]) -> None:
            real_writer(path, payload)
            if Path(path).resolve() == consumption_path:
                raise PermissionError("post-publication cleanup sentinel")

        with mock.patch.object(
            recovery,
            "verify_recovery_authorization",
            return_value=authorization,
        ), mock.patch.object(
            recovery,
            "write_exclusive_atomic",
            side_effect=publish_then_raise,
        ), mock.patch.object(
            recovery,
            "verify_payload_semantics",
            side_effect=recovery.Gate12C2CloseoutRecoveryError("sentinel"),
        ):
            with self.assertRaisesRegex(
                recovery.Gate12C2CloseoutRecoveryError, "was rejected"
            ):
                recovery.execute_payload_seal(auth_path)
        attempt = recovery.read_mapping(
            Path(authorization["attempt_output"]), label="attempt"
        )
        consumption = recovery.read_mapping(
            consumption_path, label="consumption"
        )
        failure = recovery.read_mapping(
            Path(authorization["failure_output"]), label="failure"
        )
        self.assertEqual(failure["consumption_status"], "present_verified")
        with mock.patch.object(
            recovery,
            "verify_recovery_authorization",
            return_value=authorization,
        ):
            self.assertEqual(
                recovery.verify_recovery_failure(
                    failure,
                    authorization=authorization,
                    attempt=attempt,
                    consumption=consumption,
                ),
                failure,
            )
            with self.assertRaisesRegex(
                recovery.Gate12C2CloseoutRecoveryError,
                "omits published consumption",
            ):
                recovery.verify_recovery_failure(
                    failure,
                    authorization=authorization,
                    attempt=attempt,
                    consumption=None,
                )
            other_attempt = recovery.build_attempt_receipt(
                authorization,
                claimed_at_utc=attempt["claimed_at_utc"],
                attempt_id="different-attempt",
                process_identity_value=attempt["process_identity"],
            )
            with self.assertRaisesRegex(
                recovery.Gate12C2CloseoutRecoveryError,
                "attempt evidence differs",
            ):
                recovery.verify_recovery_failure(
                    failure,
                    authorization=authorization,
                    attempt=other_attempt,
                    consumption=consumption,
                )
            other_consumption = recovery.build_consumption_receipt(
                authorization,
                attempt,
                consumed_at_utc=attempt["claimed_at_utc"],
                require_current_freshness=False,
            )
            with self.assertRaisesRegex(
                recovery.Gate12C2CloseoutRecoveryError,
                "consumption evidence differs",
            ):
                recovery.verify_recovery_failure(
                    failure,
                    authorization=authorization,
                    attempt=attempt,
                    consumption=other_consumption,
                )

    def test_lock_retirement_is_unavailable(self) -> None:
        with self.assertRaisesRegex(
            recovery.Gate12C2CloseoutRecoveryError, "remains HOLD"
        ):
            recovery.retire_stale_lock()

    def test_public_cli_sanitizes_recovery_errors(self) -> None:
        sentinel = self.base / "RAW_SCIENTIFIC_DIRECTION_SENTINEL.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(TOOLS_DIR / "run_gate12c2_closeout_recovery.py"),
                "--authorization",
                str(sentinel),
            ],
            cwd=str(TOOLS_DIR.parent),
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(completed.stdout, "")
        self.assertEqual(
            completed.stderr.strip(), recovery.PUBLIC_ERROR_CODE
        )
        self.assertNotIn("RAW_SCIENTIFIC", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)


    def _authorization_packet(
        self,
        *,
        authorization_output: Path | None = None,
        attempt_output: Path | None = None,
        terminal_output: Path | None = None,
    ) -> tuple[dict[str, object], dict[str, object]]:
        packet = self._incident_packet()
        amendment = recovery.build_recovery_amendment(
            incident_manifest_path=packet["manifest_path"],
            failure_receipt_path=packet["failure_path"],
            exposure_ledger_path=packet["exposure_path"],
            amendment_id="authorization-amendment-test",
            recorded_at_utc=self.NOW,
        )
        amendment_path = self._write_mapping("amendment.json", amendment)
        review = {
            "schema_version": "gate12c2_closeout_recovery_fresh_review_v0.1",
            "review_id": "fresh-review",
            "reviewer_context_id": "fresh-reviewer",
            "reviewed_source_commit": recovery.git_head(),
            "reviewed_implementation_sha256": (
                recovery.recovery_implementation_hashes()
            ),
            "amendment_payload_sha256": amendment[
                "amendment_payload_sha256"
            ],
            "incident_manifest_payload_sha256": packet["manifest"][
                "incident_manifest_payload_sha256"
            ],
            "review_status": "pass",
            "P0_count": 0,
            "P1_count": 0,
            "scientific_values_inspected": False,
            "recovery_authorization_may_be_issued": True,
            "lock_retirement_authorized": False,
            "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        review["review_receipt_payload_sha256"] = recovery.sha256_bytes(
            recovery.canonical_json_bytes(review)
        )
        review_path = self._write_mapping("review.json", review)
        archived = self.base / "archived-for-auth.json"
        archived.write_bytes(b"archived-plan-placeholder")
        with mock.patch.object(
            recovery, "verify_legacy_lineage", return_value={"legacy": "ok"}
        ):
            authorization = recovery.build_recovery_authorization(
                amendment_path=amendment_path,
                review_receipt_path=review_path,
                incident_manifest_path=packet["manifest_path"],
                archived_plan_path=archived,
                output_root=self.root,
                authorization_id="recovery-auth-test",
                expires_at_utc=(
                    datetime.now(timezone.utc) + timedelta(minutes=5)
                ).isoformat(),
                authorization_output=(
                    authorization_output
                    or self.base / "auth-output/authorization.json"
                ),
                attempt_output=(
                    attempt_output or self.base / "auth-output/attempt.json"
                ),
                consumption_output=self.base / "auth-output/consumption.json",
                terminal_output=(
                    terminal_output or self.base / "auth-output/terminal.json"
                ),
                seal_output=self.base / "auth-output/seal.json",
                failure_output=self.base / "auth-output/failure.json",
            )
        return authorization, {"legacy": "ok"}

    def test_authorization_is_closed_schema_and_no_lock_retirement(self) -> None:
        authorization, lineage = self._authorization_packet()
        with mock.patch.object(
            recovery, "verify_legacy_lineage", return_value=lineage
        ):
            verified = recovery.verify_recovery_authorization(
                authorization, require_current_freshness=True
            )
        self.assertFalse(verified["stale_lock_retirement_authorized"])
        self.assertEqual(verified["resource_gate_status"], "indeterminate")
        tampered = copy.deepcopy(authorization)
        tampered["scientific_direction"] = "sentinel"
        tampered.pop("authorization_payload_sha256")
        tampered["authorization_payload_sha256"] = recovery.sha256_bytes(
            recovery.canonical_json_bytes(tampered)
        )
        with self.assertRaisesRegex(
            recovery.Gate12C2CloseoutRecoveryError, "schema mismatch"
        ):
            recovery.verify_recovery_authorization(
                tampered, require_current_freshness=True
            )

    def test_authorization_outputs_cannot_be_inside_profile_root(self) -> None:
        with self.assertRaisesRegex(
            recovery.Gate12C2CloseoutRecoveryError, "outside the profile root"
        ):
            recovery._require_outside_root(
                self.root / "seal.json", root=self.root
            )

    def test_terminal_claim_cannot_be_inside_profile_root(self) -> None:
        with self.assertRaisesRegex(
            recovery.Gate12C2CloseoutRecoveryError,
            "outside the profile root",
        ):
            self._authorization_packet(
                terminal_output=self.root / "terminal.json"
            )
    def test_exclusive_writer_never_overwrites_competing_destination(self) -> None:
        destination = self.base / "exclusive/receipt.json"
        competitor = b"competitor-won"
        real_link = recovery.os.link

        def competing_link(source: Path, target: Path) -> None:
            Path(target).write_bytes(competitor)
            real_link(source, target)

        with mock.patch.object(recovery.os, "link", side_effect=competing_link):
            with self.assertRaisesRegex(
                recovery.Gate12C2CloseoutRecoveryError,
                "already exists",
            ):
                recovery.write_exclusive_atomic(destination, {"writer": "loser"})
        self.assertEqual(destination.read_bytes(), competitor)
        self.assertEqual(list(destination.parent.glob("*.tmp")), [])
        self.assertEqual(list(destination.parent.glob(".*.tmp")), [])

    def test_exact_publisher_recovers_after_temp_cleanup_failure(self) -> None:
        destination = self.base / "cleanup-failure/receipt.json"
        payload = {"publication": "canonical"}
        real_unlink = Path.unlink

        def fail_temp_cleanup(path: Path, *args: object, **kwargs: object) -> None:
            if Path(path).suffix == ".tmp":
                raise PermissionError("cleanup sentinel")
            real_unlink(path, *args, **kwargs)

        with mock.patch.object(Path, "unlink", new=fail_temp_cleanup):
            recovered = recovery._publish_exact_or_recover(
                destination, payload, label="cleanup failure receipt"
            )
        self.assertEqual(recovered, payload)
        self.assertEqual(
            destination.read_bytes(), recovery.canonical_json_bytes(payload)
        )
        for temporary in destination.parent.glob(".*.tmp"):
            real_unlink(temporary)

    def test_exclusive_writer_has_one_cross_process_winner(self) -> None:
        destination = self.base / "exclusive-process/receipt.json"
        helper = self.base / "exclusive_writer_helper.py"
        helper.write_text(
            "\n".join(
                [
                    "import sys",
                    "from pathlib import Path",
                    f"sys.path.insert(0, {str(TOOLS_DIR)!r})",
                    "import gate12c2_closeout_recovery as recovery",
                    "try:",
                    "    recovery.write_exclusive_atomic(Path(sys.argv[1]), {'writer': sys.argv[2]})",
                    "    print('won')",
                    "except recovery.Gate12C2CloseoutRecoveryError:",
                    "    print('lost')",
                ]
            ),
            encoding="utf-8",
        )
        processes = [
            subprocess.Popen(
                [sys.executable, str(helper), str(destination), str(index)],
                cwd=str(TOOLS_DIR.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for index in range(4)
        ]
        outcomes = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=30)
            self.assertEqual(process.returncode, 0, stderr)
            outcomes.append(stdout.strip())
        self.assertEqual(outcomes.count("won"), 1)
        self.assertEqual(outcomes.count("lost"), 3)
        payload = recovery.read_mapping(destination, label="winning receipt")
        self.assertIn(payload["writer"], {"0", "1", "2", "3"})

    def test_authorization_receipt_cannot_be_inside_profile_root(self) -> None:
        with self.assertRaisesRegex(
            recovery.Gate12C2CloseoutRecoveryError,
            "outside the profile root",
        ):
            self._authorization_packet(
                authorization_output=self.root / "authorization.json"
            )

    def test_fresh_review_rejects_exposed_context_and_boolean_counts(self) -> None:
        authorization, _ = self._authorization_packet()
        amendment = recovery.read_mapping(
            Path(authorization["amendment_path"]), label="amendment"
        )
        review = recovery.read_mapping(
            Path(authorization["review_receipt_path"]), label="review"
        )
        attacks = (
            ("reviewer_context_id", "exposed-reviewer"),
            ("P0_count", False),
            ("P1_count", False),
        )
        for field, value in attacks:
            with self.subTest(field=field):
                tampered = copy.deepcopy(review)
                tampered[field] = value
                tampered.pop("review_receipt_payload_sha256")
                tampered["review_receipt_payload_sha256"] = (
                    recovery.sha256_bytes(recovery.canonical_json_bytes(tampered))
                )
                with self.assertRaises(recovery.Gate12C2CloseoutRecoveryError):
                    recovery.verify_recovery_review_receipt(
                        tampered, amendment=amendment
                    )

    def test_authorization_rejects_rehashed_resource_status_change(self) -> None:
        authorization, lineage = self._authorization_packet()
        tampered = copy.deepcopy(authorization)
        tampered["original_resource_evidence_status"] = "present"
        tampered.pop("authorization_payload_sha256")
        tampered["authorization_payload_sha256"] = recovery.sha256_bytes(
            recovery.canonical_json_bytes(tampered)
        )
        with mock.patch.object(
            recovery, "verify_legacy_lineage", return_value=lineage
        ):
            with self.assertRaisesRegex(
                recovery.Gate12C2CloseoutRecoveryError,
                "changed a frozen field",
            ):
                recovery.verify_recovery_authorization(
                    tampered, require_current_freshness=True
                )

    def test_post_consumption_restart_seals_interrupted_attempt(self) -> None:
        auth_path, authorization = self._mock_authorization()
        dead_identity = {
            "pid": 2147483647,
            "identity_kind": "windows_creation_filetime",
            "start_marker": "1",
        }
        with mock.patch.object(
            recovery,
            "verify_recovery_authorization",
            return_value=authorization,
        ):
            attempt = recovery.build_attempt_receipt(
                authorization,
                claimed_at_utc=self.NOW,
                attempt_id="interrupted-attempt",
                process_identity_value=dead_identity,
            )
            recovery.write_exclusive_atomic(
                Path(authorization["attempt_output"]), attempt
            )
            consumption = recovery.build_consumption_receipt(
                authorization,
                attempt,
                consumed_at_utc=self.NOW,
                require_current_freshness=False,
            )
            recovery.write_exclusive_atomic(
                Path(authorization["consumption_output"]), consumption
            )
            with mock.patch.object(
                recovery, "attempt_process_state", return_value=recovery.PROCESS_DEAD
            ):
                with self.assertRaisesRegex(
                    recovery.Gate12C2CloseoutRecoveryError,
                    "sealed as failed",
                ):
                    recovery.execute_payload_seal(auth_path)
        failure = recovery.read_mapping(
            Path(authorization["failure_output"]), label="interrupted failure"
        )
        self.assertEqual(failure["state"], "RECOVERY_INTERRUPTED")
        self.assertEqual(failure["failure_phase"], "post_consumption_restart")
        self.assertEqual(failure["consumption_status"], "present_verified")
        self.assertFalse(failure["authorization_reusable"])
        self.assertTrue(failure["new_authorization_required"])

    def test_active_attempt_rejects_competing_recovery_without_failure(self) -> None:
        auth_path, authorization = self._mock_authorization()
        with mock.patch.object(
            recovery,
            "verify_recovery_authorization",
            return_value=authorization,
        ):
            attempt = recovery.build_attempt_receipt(
                authorization,
                claimed_at_utc=self.NOW,
                attempt_id="active-attempt",
                process_identity_value={
                    "pid": 100,
                    "identity_kind": "fixture",
                    "start_marker": "active",
                },
            )
            recovery.write_exclusive_atomic(
                Path(authorization["attempt_output"]), attempt
            )
            with mock.patch.object(
                recovery, "attempt_process_state", return_value=recovery.PROCESS_ACTIVE
            ):
                with self.assertRaisesRegex(
                    recovery.Gate12C2CloseoutRecoveryError,
                    "already active",
                ):
                    recovery.execute_payload_seal(auth_path)
        self.assertFalse(Path(authorization["failure_output"]).exists())
        self.assertFalse(Path(authorization["consumption_output"]).exists())

    def test_unknown_competing_liveness_publishes_no_failure(self) -> None:
        auth_path, authorization = self._mock_authorization()
        semantic_started = threading.Event()
        release_semantic = threading.Event()
        outcomes: dict[str, object] = {}

        def slow_semantics(**_: object) -> dict[str, object]:
            semantic_started.set()
            if not release_semantic.wait(timeout=30):
                raise AssertionError("semantic fixture timed out")
            return self._semantic_stub()

        def run_primary() -> None:
            try:
                outcomes["primary"] = recovery.execute_payload_seal(auth_path)
            except Exception as error:  # pragma: no cover - asserted below
                outcomes["primary_error"] = type(error).__name__

        with mock.patch.object(
            recovery,
            "verify_recovery_authorization",
            return_value=authorization,
        ), mock.patch.object(
            recovery, "verify_payload_semantics", side_effect=slow_semantics
        ):
            primary = threading.Thread(target=run_primary)
            primary.start()
            self.assertTrue(semantic_started.wait(timeout=30))
            with mock.patch.object(
                recovery,
                "attempt_process_state",
                return_value=recovery.PROCESS_UNKNOWN,
            ):
                with self.assertRaisesRegex(
                    recovery.Gate12C2CloseoutRecoveryError,
                    "liveness is indeterminate",
                ):
                    recovery.execute_payload_seal(auth_path)
            self.assertFalse(Path(authorization["terminal_output"]).exists())
            self.assertFalse(Path(authorization["failure_output"]).exists())
            release_semantic.set()
            primary.join(timeout=30)
            self.assertFalse(primary.is_alive())

        self.assertIn("primary", outcomes)
        self.assertNotIn("primary_error", outcomes)
        self.assertTrue(Path(authorization["terminal_output"]).is_file())
        self.assertTrue(Path(authorization["seal_output"]).is_file())
        self.assertFalse(Path(authorization["failure_output"]).exists())
        Path(authorization["failure_output"]).write_bytes(b"{}")
        with mock.patch.object(
            recovery,
            "verify_recovery_authorization",
            return_value=authorization,
        ):
            with self.assertRaisesRegex(
                recovery.Gate12C2CloseoutRecoveryError,
                "cannot coexist",
            ):
                recovery.verify_payload_seal(
                    authorization_path=auth_path,
                    seal_path=Path(authorization["seal_output"]),
                )

    def test_terminal_claim_allows_only_one_competing_outcome(self) -> None:
        auth_path, authorization = self._mock_authorization()
        semantic_started = threading.Event()
        release_semantic = threading.Event()
        outcomes: dict[str, object] = {}

        def slow_semantics(**_: object) -> dict[str, object]:
            semantic_started.set()
            if not release_semantic.wait(timeout=30):
                raise AssertionError("semantic fixture timed out")
            return self._semantic_stub()

        def run_primary() -> None:
            try:
                outcomes["primary"] = recovery.execute_payload_seal(auth_path)
            except Exception as error:
                outcomes["primary_error"] = str(error)

        with mock.patch.object(
            recovery,
            "verify_recovery_authorization",
            return_value=authorization,
        ), mock.patch.object(
            recovery, "verify_payload_semantics", side_effect=slow_semantics
        ):
            primary = threading.Thread(target=run_primary)
            primary.start()
            self.assertTrue(semantic_started.wait(timeout=30))
            with mock.patch.object(
                recovery,
                "attempt_process_state",
                return_value=recovery.PROCESS_DEAD,
            ):
                with self.assertRaisesRegex(
                    recovery.Gate12C2CloseoutRecoveryError,
                    "sealed as failed",
                ):
                    recovery.execute_payload_seal(auth_path)
            release_semantic.set()
            primary.join(timeout=30)
            self.assertFalse(primary.is_alive())

        self.assertIn("primary_error", outcomes)
        self.assertTrue(Path(authorization["terminal_output"]).is_file())
        self.assertTrue(Path(authorization["failure_output"]).is_file())
        self.assertFalse(Path(authorization["seal_output"]).exists())
        attempt = recovery.read_mapping(
            Path(authorization["attempt_output"]), label="attempt"
        )
        consumption = recovery.read_mapping(
            Path(authorization["consumption_output"]), label="consumption"
        )
        failure = recovery.read_mapping(
            Path(authorization["failure_output"]), label="failure"
        )
        with mock.patch.object(
            recovery,
            "verify_recovery_authorization",
            return_value=authorization,
        ):
            self.assertEqual(
                recovery.verify_recovery_failure(
                    failure,
                    authorization=authorization,
                    attempt=attempt,
                    consumption=consumption,
                ),
                failure,
            )
            Path(authorization["seal_output"]).write_bytes(b"{}")
            with self.assertRaisesRegex(
                recovery.Gate12C2CloseoutRecoveryError,
                "cannot coexist",
            ):
                recovery.verify_recovery_failure(
                    failure,
                    authorization=authorization,
                    attempt=attempt,
                    consumption=consumption,
                )

    def test_process_liveness_is_tristate(self) -> None:
        recorded = {
            "pid": 123,
            "identity_kind": "fixture",
            "start_marker": "one",
        }
        with mock.patch.object(
            recovery,
            "_query_process_identity",
            return_value=(recovery.PROCESS_ACTIVE, dict(recorded)),
        ):
            self.assertEqual(
                recovery.process_identity_state(recorded), recovery.PROCESS_ACTIVE
            )
        replacement = dict(recorded)
        replacement["start_marker"] = "two"
        with mock.patch.object(
            recovery,
            "_query_process_identity",
            return_value=(recovery.PROCESS_ACTIVE, replacement),
        ):
            self.assertEqual(
                recovery.process_identity_state(recorded), recovery.PROCESS_DEAD
            )
        with mock.patch.object(
            recovery,
            "_query_process_identity",
            return_value=(recovery.PROCESS_UNKNOWN, None),
        ):
            self.assertEqual(
                recovery.process_identity_state(recorded), recovery.PROCESS_UNKNOWN
            )

    def test_historical_failure_verification_does_not_remeasure_runner(self) -> None:
        manifest = self._manifest()
        manifest_path = self._write_mapping("stable-runner-manifest.json", manifest)
        stdout_path = self.base / "stable-runner-stdout.log"
        stderr_path = self.base / "stable-runner-stderr.log"
        stdout_path.write_bytes(b"")
        stderr_path.write_bytes(b"")
        with mock.patch.object(
            recovery,
            "_query_process_identity",
            return_value=(
                recovery.PROCESS_ACTIVE,
                {
                    "pid": 123,
                    "identity_kind": "fixture",
                    "start_marker": "observed",
                },
            ),
        ):
            failure = recovery.build_failure_receipt(
                incident_manifest_path=manifest_path,
                stdout_log_path=stdout_path,
                stderr_log_path=stderr_path,
                runner_pid=123,
                observed_at_utc=self.NOW,
            )
        with mock.patch.object(
            recovery,
            "_query_process_identity",
            side_effect=AssertionError("historical liveness was remeasured"),
        ):
            self.assertEqual(recovery.verify_failure_receipt(failure), failure)

    def test_receipt_timestamps_must_be_monotone(self) -> None:
        _, authorization = self._mock_authorization()
        with mock.patch.object(
            recovery,
            "verify_recovery_authorization",
            return_value=authorization,
        ):
            with self.assertRaisesRegex(
                recovery.Gate12C2CloseoutRecoveryError,
                "outside the authorization interval",
            ):
                recovery.build_attempt_receipt(
                    authorization,
                    claimed_at_utc="2026-07-26T00:00:00+00:00",
                    process_identity_value={
                        "pid": 1,
                        "identity_kind": "fixture",
                        "start_marker": "one",
                    },
                )
            attempt = recovery.build_attempt_receipt(
                authorization,
                claimed_at_utc=self.NOW,
                process_identity_value={
                    "pid": 1,
                    "identity_kind": "fixture",
                    "start_marker": "one",
                },
            )
            with self.assertRaisesRegex(
                recovery.Gate12C2CloseoutRecoveryError,
                "outside the frozen sequence",
            ):
                recovery.build_consumption_receipt(
                    authorization,
                    attempt,
                    consumed_at_utc="2026-07-27T23:59:59+00:00",
                    require_current_freshness=False,
                )
            consumption = recovery.build_consumption_receipt(
                authorization,
                attempt,
                consumed_at_utc=self.NOW,
                require_current_freshness=False,
            )
            with mock.patch.object(
                recovery,
                "verify_payload_semantics",
                return_value=self._semantic_stub(),
            ):
                with self.assertRaisesRegex(
                    recovery.Gate12C2CloseoutRecoveryError,
                    "precedes recovery consumption",
                ):
                    recovery.build_payload_seal(
                        authorization=authorization,
                        attempt=attempt,
                        consumption=consumption,
                        sealed_at_utc="2026-07-27T23:59:59+00:00",
                    )
            with self.assertRaisesRegex(
                recovery.Gate12C2CloseoutRecoveryError,
                "precedes its predecessor",
            ):
                recovery.build_recovery_failure(
                    authorization=authorization,
                    attempt=attempt,
                    consumption=consumption,
                    failure_state="RECOVERY_INTERRUPTED",
                    failure_phase="payload_verification",
                    recorded_at_utc="2026-07-27T23:59:59+00:00",
                )
    def test_legacy_lineage_rejects_changed_synthetic_lab_dependency(self) -> None:
        archived, source_commit, plan_hash = self._legacy_fixture()
        original_sha256_file = recovery.sha256_file
        legacy_hashes = {
            "gate12c2_development_shards.py": original_sha256_file(
                Path(recovery.shards.__file__)
            ),
            "gate12c2_synthetic_lab.py": original_sha256_file(
                Path(recovery.shards.lab.__file__)
            ),
        }

        def legacy_blob(_: str, relative: str) -> str:
            return legacy_hashes[Path(relative).name]

        def changed_dependency(path: Path) -> str:
            if Path(path).resolve() == Path(recovery.shards.lab.__file__).resolve():
                return "b" * 64
            return original_sha256_file(path)

        with mock.patch.object(
            recovery, "git_blob_sha256", side_effect=legacy_blob
        ), mock.patch.object(
            recovery, "sha256_file", side_effect=changed_dependency
        ), mock.patch.object(
            recovery,
            "_query_process_identity",
            return_value=(recovery.PROCESS_DEAD, None),
        ):
            with self.assertRaisesRegex(
                recovery.Gate12C2CloseoutRecoveryError,
                "current semantic verifier dependency",
            ):
                recovery.verify_legacy_lineage(
                    output_root=self.root,
                    archived_plan_path=archived,
                    expected_source_commit=source_commit,
                    expected_plan_payload_sha256=plan_hash,
                )

    def test_failure_receipt_measures_runner_presence(self) -> None:
        manifest = self._manifest()
        manifest_path = self._write_mapping("runner-manifest.json", manifest)
        stdout_path = self.base / "runner-stdout.log"
        stderr_path = self.base / "runner-stderr.log"
        stdout_path.write_bytes(b"")
        stderr_path.write_bytes(b"")
        with mock.patch.object(
            recovery,
            "_query_process_identity",
            return_value=(
                recovery.PROCESS_ACTIVE,
                {
                    "pid": 123,
                    "identity_kind": "fixture",
                    "start_marker": "present",
                },
            ),
        ):
            failure = recovery.build_failure_receipt(
                incident_manifest_path=manifest_path,
                stdout_log_path=stdout_path,
                stderr_log_path=stderr_path,
                runner_pid=123,
                observed_at_utc=self.NOW,
            )
        self.assertTrue(failure["runner_process_present"])

    def test_failure_receipt_rejects_indeterminate_runner_liveness(self) -> None:
        manifest = self._manifest()
        manifest_path = self._write_mapping("unknown-runner-manifest.json", manifest)
        stdout_path = self.base / "unknown-runner-stdout.log"
        stderr_path = self.base / "unknown-runner-stderr.log"
        stdout_path.write_bytes(b"")
        stderr_path.write_bytes(b"")
        with mock.patch.object(
            recovery,
            "_query_process_identity",
            return_value=(recovery.PROCESS_UNKNOWN, None),
        ):
            with self.assertRaisesRegex(
                recovery.Gate12C2CloseoutRecoveryError,
                "liveness is indeterminate",
            ):
                recovery.build_failure_receipt(
                    incident_manifest_path=manifest_path,
                    stdout_log_path=stdout_path,
                    stderr_log_path=stderr_path,
                    runner_pid=123,
                    observed_at_utc=self.NOW,
                )

    def test_semantic_summary_is_closed_and_strictly_typed(self) -> None:
        valid = self._semantic_stub()
        self.assertEqual(
            recovery.verify_semantic_verification_summary(valid), valid
        )
        attacks = []
        extra = copy.deepcopy(valid)
        extra["scientific_direction"] = "sentinel"
        attacks.append(extra)
        boolean_count = copy.deepcopy(valid)
        boolean_count["configuration_count"] = True
        attacks.append(boolean_count)
        missing_row = copy.deepcopy(valid)
        missing_row["configuration_results"] = missing_row[
            "configuration_results"
        ][:-1]
        attacks.append(missing_row)
        for attack in attacks:
            with self.assertRaises(recovery.Gate12C2CloseoutRecoveryError):
                recovery.verify_semantic_verification_summary(attack)

    def test_all_public_clis_sanitize_malformed_arguments(self) -> None:
        scripts = (
            "freeze_gate12c2_closeout_incident.py",
            "issue_gate12c2_closeout_recovery_authorization.py",
            "run_gate12c2_closeout_recovery.py",
            "verify_gate12c2_closeout_recovery.py",
        )
        for name in scripts:
            with self.subTest(script=name):
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(TOOLS_DIR / name),
                        "--RAW_SCIENTIFIC_DIRECTION_SENTINEL",
                    ],
                    cwd=str(TOOLS_DIR.parent),
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(completed.returncode, 2)
                self.assertEqual(completed.stdout, "")
                self.assertEqual(
                    completed.stderr.strip(), recovery.PUBLIC_ERROR_CODE
                )
                self.assertNotIn("RAW_SCIENTIFIC", completed.stderr)
                self.assertNotIn("usage:", completed.stderr)

if __name__ == "__main__":
    unittest.main()
