#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
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
        plan: dict[str, object] = {
            "source_commit": source_commit,
            "implementation_sha256": {
                "gate12c2_development_shards.py": shard_hash,
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
        with mock.patch.object(
            recovery, "git_blob_sha256", return_value=shard_hash
        ), mock.patch.object(profile, "_pid_is_running", return_value=False):
            evidence = recovery.verify_legacy_lineage(
                output_root=self.root,
                archived_plan_path=archived,
                expected_source_commit=source_commit,
                expected_plan_payload_sha256=plan_hash,
            )
        self.assertEqual(evidence["original_source_commit"], source_commit)
        self.assertTrue(evidence["stale_lock_owner_not_running"])

    def test_legacy_lineage_rejects_changed_shard_verifier(self) -> None:
        archived, source_commit, plan_hash = self._legacy_fixture()
        shard_hash = recovery.sha256_file(Path(recovery.shards.__file__))
        original_sha256_file = recovery.sha256_file

        def changed_current_verifier(path: Path) -> str:
            if Path(path).resolve() == Path(recovery.shards.__file__).resolve():
                return "b" * 64
            return original_sha256_file(path)

        with mock.patch.object(
            recovery, "git_blob_sha256", return_value=shard_hash
        ), mock.patch.object(
            recovery, "sha256_file", side_effect=changed_current_verifier
        ), mock.patch.object(profile, "_pid_is_running", return_value=False):
            with self.assertRaisesRegex(
                recovery.Gate12C2CloseoutRecoveryError,
                "current shard verifier",
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
        consumption = self.base / "external/consumption.json"
        seal = self.base / "external/seal.json"
        failure = self.base / "external/failure.json"
        authorization = {
            "authorization_id": "test-auth",
            "authorization_scope": "payload_verification_and_external_seal_only",
            "authorization_payload_sha256": "a" * 64,
            "output_root": self.root.resolve().as_posix(),
            "incident_manifest_path": (self.base / "incident.json").as_posix(),
            "archived_plan_path": (self.base / "archived.json").as_posix(),
            "incident_manifest_payload_sha256": "b" * 64,
            "amendment_payload_sha256": "c" * 64,
            "consumption_output": consumption.as_posix(),
            "seal_output": seal.as_posix(),
            "failure_output": failure.as_posix(),
        }
        auth_path = self._write_mapping("recovery-auth.json", authorization)
        (self.base / "incident.json").write_text("{}", encoding="utf-8")
        return auth_path, authorization

    @staticmethod
    def _semantic_stub() -> dict[str, object]:
        return {
            "status": "verified",
            "configuration_results": [
                {"configuration_id": "stub", "status": "verified"}
            ],
            "protected_surface_sha256": "d" * 64,
            "complete_surface_sha256": "e" * 64,
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
                "already been consumed",
            ):
                recovery.execute_payload_seal(auth_path)

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


    def _authorization_packet(self) -> tuple[dict[str, object], dict[str, object]]:
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
                consumption_output=self.base / "auth-output/consumption.json",
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

if __name__ == "__main__":
    unittest.main()
