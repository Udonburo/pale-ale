#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import gate12c2_draw_profile as profile  # noqa: E402


class _FakeMonitor:
    def __init__(self, _: int) -> None:
        pass

    def start(self) -> None:
        pass

    def stop(self) -> dict[str, object]:
        return {
            "sample_interval_seconds": 0.1,
            "sample_count": 2,
            "peak_process_tree_rss_bytes": 128 * 1024 * 1024,
            "peak_observed_process_count": 5,
            "monitor_error": None,
        }


class Gate12C2DrawProfileTest(unittest.TestCase):
    def build_plan(self) -> dict[str, object]:
        return profile.build_draw_profile_plan(
            source_commit="test-source-commit"
        )

    @staticmethod
    def _mechanical_checks() -> dict[str, dict[str, object]]:
        return {
            key: {"status": "pass", "evidence": key}
            for key in profile.REQUIRED_PREFLIGHT_CHECKS
        }

    def receipts(
        self,
        plan: dict[str, object],
        output_root: Path,
    ) -> tuple[dict[str, object], dict[str, object]]:
        evidence_root = output_root.parent / "test-evidence"
        evidence_root.mkdir(parents=True, exist_ok=True)
        bundle_path = evidence_root / "recovery.bundle"
        carry_path = evidence_root / "carry.json"
        worker_profile_path = evidence_root / "worker-profile.json"
        bundle_path.write_bytes(b"test-recovery-bundle")
        carry_path.write_bytes(b"test-carry-forward")
        worker_profile_path.write_bytes(b"test-worker-profile")
        preflight = profile._serialize_mechanical_preflight(
            plan,
            output_root=output_root,
            preflight_id="draw-profile-test-preflight",
            checks=self._mechanical_checks(),
            recovery={
                "bundle_path": bundle_path.as_posix(),
                "bundle_file_sha256": profile._sha256_file(bundle_path),
                "bundle_bytes": bundle_path.stat().st_size,
                "git_bundle_verify": "pass",
                "standalone_clone": "pass",
                "explicit_checkout": "pass",
                "restored_head": plan["source_commit"],
                "git_fsck_full": "pass",
                "restored_worktree_clean": True,
                "implementation_blob_identity": "pass",
            },
            worker_carry_forward={
                "path": carry_path.as_posix(),
                "file_sha256": profile._sha256_file(carry_path),
                "payload_sha256": "c" * 64,
            },
            resource_projection={
                "worker_profile_receipt_path": (
                    worker_profile_path.as_posix()
                ),
                "worker_profile_receipt_file_sha256": (
                    profile._sha256_file(worker_profile_path)
                ),
                "worker_profile_receipt_payload_sha256": "e" * 64,
                "projected_output_bytes": 1000,
                "disk_projection_safety_factor": 1.3,
                "projected_output_bytes_with_safety": 1300,
                "disk_free_bytes_at_preflight": 10000,
                "projected_remaining_free_bytes": 8700,
                "minimum_remaining_free_bytes": 5000,
                "disk_gate_pass": True,
                "worker_profile_peak_process_tree_rss_bytes_at_draw_255": 100,
                "projected_peak_process_tree_rss_bytes_at_draw_1023": 402,
                "memory_projection_safety_factor": 1.3,
                "projected_peak_process_tree_rss_bytes_with_safety": 523,
                "physical_ram_bytes_at_preflight": 10000,
                "available_physical_memory_bytes_at_preflight": 9000,
                "maximum_admitted_peak_process_tree_rss_bytes": 7500,
                "memory_headroom_gate_pass": True,
            },
        )
        authorization = profile.build_execution_authorization(
            plan,
            preflight,
            output_root=output_root,
            authorization_id="draw-profile-test-authorization",
            purpose="draw-profile-unit-test",
            expires_at_utc="2100-01-01T00:00:00+00:00",
        )
        return preflight, authorization

    @staticmethod
    def _fake_execute(
        subplan: dict[str, object],
        **_: object,
    ) -> dict[str, object]:
        return {
            "outer_experiment_count": len(
                subplan["outer_experiment_indices"]
            ),
            "all_outer_indices_present": True,
            "plan_payload_sha256": subplan["plan_payload_sha256"],
            "shards": [
                {
                    "operational_metrics": {
                        "mode": "execute_new",
                        "endpoint_draw_attempts": 100,
                        "endpoint_draw_acceptances": 100,
                        "rejection_reason_counts": {},
                        "unaccounted_rejection_count": 0,
                        "exhausted_incomplete_stream_count": 0,
                    }
                }
            ],
        }

    @staticmethod
    def _fake_verify(
        subplan: dict[str, object],
        **_: object,
    ) -> dict[str, object]:
        return {
            "scientific_projection_sha256": (
                subplan["plan_payload_sha256"]
            ),
            "index_payload_sha256": "f" * 64,
        }

    def _execute_mocked(
        self,
        plan: dict[str, object],
        output_root: Path,
    ) -> dict[str, object]:
        preflight, authorization = self.receipts(plan, output_root)
        with mock.patch.object(
            profile.shards,
            "execute_development_shard_plan",
            side_effect=self._fake_execute,
        ), mock.patch.object(
            profile.shards,
            "verify_development_shard_index",
            side_effect=self._fake_verify,
        ), mock.patch.object(
            profile.throughput,
            "ProcessTreeRssMonitor",
            _FakeMonitor,
        ), mock.patch.object(
            profile,
            "_physical_ram_bytes",
            return_value=16 * 1024**3,
        ):
            return profile.execute_draw_profile(
                plan,
                output_root=output_root,
                preflight_receipt=preflight,
                authorization_receipt=authorization,
            )

    def test_plan_exactly_fixes_nine_configurations(self) -> None:
        plan = self.build_plan()
        verified = profile.verify_draw_profile_plan(plan)
        self.assertEqual(len(verified["configurations"]), 9)
        self.assertEqual(verified["worker_count"], 4)
        self.assertEqual(verified["prefix_counts"], [255, 511, 1023])
        self.assertEqual(
            profile._resource_policy(),
            verified["resource_policy"],
        )
        self.assertFalse(verified["locked_execution_authorized"])
        self.assertFalse(verified["real_held_out_execution_authorized"])

    def test_rehashed_boundary_or_layout_change_fails_closed(self) -> None:
        plan = self.build_plan()
        for key, value in (
            ("locked_execution_authorized", True),
            ("real_held_out_execution_authorized", True),
            ("N2_open", True),
            ("N3_open", True),
            ("public_claim", True),
            ("unexpected_permission", False),
        ):
            tampered = dict(plan)
            tampered[key] = value
            tampered.pop("draw_profile_plan_payload_sha256")
            tampered["draw_profile_plan_payload_sha256"] = (
                profile._sha256_bytes(
                    profile._canonical_json_bytes(tampered)
                )
            )
            with self.subTest(key=key):
                with self.assertRaises(
                    profile.Gate12C2DrawProfileError
                ):
                    profile.verify_draw_profile_plan(tampered)

    def test_public_preflight_path_is_mechanical_not_attestational(
        self,
    ) -> None:
        self.assertFalse(
            hasattr(profile, "build_no_outcome_preflight")
        )
        plan = self.build_plan()
        worker_rows = {
            regime["regime_id"]: {
                "output_bytes": 1000,
                "outer_experiment_count": 1,
                "process_tree_memory": {
                    "peak_process_tree_rss_bytes": 100,
                },
            }
            for regime in profile.REGIME_SPECIFICATIONS
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path = root / "plan.json"
            plan_path.write_bytes(profile._canonical_json_bytes(plan))
            with mock.patch.object(
                profile,
                "_verify_prior_worker_profile",
                return_value={
                    "path": (root / "worker.json").as_posix(),
                    "file_sha256": "1" * 64,
                    "payload_sha256": "2" * 64,
                    "worker_4_rows": worker_rows,
                },
            ), mock.patch.object(
                profile,
                "_verify_worker_carry_forward",
                return_value={
                    "path": (root / "carry.json").as_posix(),
                    "file_sha256": "3" * 64,
                    "payload_sha256": "4" * 64,
                },
            ), mock.patch.object(
                profile,
                "_verify_recovery_bundle",
                return_value={
                    "bundle_path": (root / "bundle").as_posix(),
                    "bundle_file_sha256": "5" * 64,
                    "bundle_bytes": 1,
                    "git_bundle_verify": "pass",
                    "standalone_clone": "pass",
                    "explicit_checkout": "pass",
                    "restored_head": plan["source_commit"],
                    "git_fsck_full": "pass",
                    "restored_worktree_clean": True,
                    "implementation_blob_identity": "pass",
                },
            ), mock.patch.object(
                profile,
                "_physical_ram_bytes",
                return_value=10000,
            ), mock.patch.object(
                profile,
                "_available_physical_memory_bytes",
                return_value=9000,
            ):
                receipt = profile.issue_mechanical_preflight(
                    plan_path=plan_path,
                    output_root=root / "profile",
                    preflight_id="mechanical-test",
                    recovery_bundle_path=root / "bundle",
                    worker_profile_receipt_path=root / "worker.json",
                    worker_carry_forward_receipt_path=root / "carry.json",
                    restore_scratch_root=root,
                )
        self.assertEqual(receipt["preflight_issuer"], "mechanical")
        self.assertFalse(receipt["development_execution_authorized"])
        self.assertTrue(
            all(
                row["status"] == "pass"
                for row in receipt["checks"].values()
            )
        )
        self.assertTrue(
            receipt["resource_projection"][
                "memory_headroom_gate_pass"
            ]
        )

    def test_mechanical_preflight_fails_insufficient_memory_headroom(
        self,
    ) -> None:
        plan = self.build_plan()
        worker_rows = {
            regime["regime_id"]: {
                "output_bytes": 1000,
                "outer_experiment_count": 1,
                "process_tree_memory": {
                    "peak_process_tree_rss_bytes": 100,
                },
            }
            for regime in profile.REGIME_SPECIFICATIONS
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path = root / "plan.json"
            plan_path.write_bytes(profile._canonical_json_bytes(plan))
            with mock.patch.object(
                profile,
                "_verify_prior_worker_profile",
                return_value={
                    "path": (root / "worker.json").as_posix(),
                    "file_sha256": "1" * 64,
                    "payload_sha256": "2" * 64,
                    "worker_4_rows": worker_rows,
                },
            ), mock.patch.object(
                profile,
                "_verify_worker_carry_forward",
                return_value={
                    "path": (root / "carry.json").as_posix(),
                    "file_sha256": "3" * 64,
                    "payload_sha256": "4" * 64,
                },
            ), mock.patch.object(
                profile,
                "_physical_ram_bytes",
                return_value=10000,
            ), mock.patch.object(
                profile,
                "_available_physical_memory_bytes",
                return_value=500,
            ):
                with self.assertRaisesRegex(
                    profile.Gate12C2DrawProfileError,
                    "memory-headroom",
                ):
                    profile.issue_mechanical_preflight(
                        plan_path=plan_path,
                        output_root=root / "profile",
                        preflight_id="insufficient-memory-test",
                        recovery_bundle_path=root / "bundle",
                        worker_profile_receipt_path=root / "worker.json",
                        worker_carry_forward_receipt_path=root / "carry.json",
                        restore_scratch_root=root,
                    )

    def test_execution_requires_exact_authorization(self) -> None:
        plan = self.build_plan()
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "profile"
            with self.assertRaises(
                profile.Gate12C2DrawProfileError
            ):
                profile.execute_draw_profile(
                    plan,
                    output_root=output_root,
                )
            preflight, authorization = self.receipts(
                plan,
                output_root,
            )
            changed = dict(authorization)
            changed["output_root"] = "C:/elsewhere"
            changed.pop("authorization_receipt_payload_sha256")
            changed["authorization_receipt_payload_sha256"] = (
                profile._sha256_bytes(
                    profile._canonical_json_bytes(changed)
                )
            )
            with self.assertRaises(
                profile.Gate12C2DrawProfileError
            ):
                profile.execute_draw_profile(
                    plan,
                    output_root=output_root,
                    preflight_receipt=preflight,
                    authorization_receipt=changed,
                )

    def test_mocked_coordinator_emits_exact_resource_chain(self) -> None:
        plan = self.build_plan()
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "profile"
            receipt = self._execute_mocked(plan, output_root)
            resource = json.loads(
                (output_root / profile.RESOURCE_RECEIPT_NAME).read_text(
                    encoding="utf-8"
                )
            )
            gate = profile.verify_resource_evidence_chain(
                plan,
                receipt,
                resource,
            )
            self.assertEqual(gate["eligible_draw_counts"], [255, 511, 1023])
            self.assertFalse(
                (output_root / profile.COORDINATOR_LOCK_NAME).exists()
            )
            self.assertTrue(
                (
                    output_root
                    / "control"
                    / profile.AUTHORIZATION_CONSUMED_NAME
                ).is_file()
            )
        self.assertEqual(receipt["configuration_count"], 9)
        self.assertIsNone(receipt["scientific_calibration_result"])
        self.assertFalse(receipt["scientific_outcomes_exposed"])
        encoded = json.dumps(receipt, allow_nan=False, sort_keys=True)
        for forbidden in (
            '"grid_outcome"',
            '"claim_promotion"',
            '"any_endpoint_support"',
            '"endpoint_identified"',
            '"median_log_ratio"',
        ):
            self.assertNotIn(forbidden, encoded)

    def test_root_stale_partial_fails_before_coordinator_write(self) -> None:
        plan = self.build_plan()
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "profile"
            output_root.mkdir()
            stale = output_root / ".plan.json.999999.tmp"
            stale.write_text("stale", encoding="utf-8")
            preflight, authorization = self.receipts(plan, output_root)
            with self.assertRaisesRegex(
                profile.Gate12C2DrawProfileError,
                "stale transaction artifacts",
            ):
                profile.execute_draw_profile(
                    plan,
                    output_root=output_root,
                    preflight_receipt=preflight,
                    authorization_receipt=authorization,
                )
            self.assertTrue(stale.exists())
            self.assertFalse((output_root / "plan.json").exists())

    def test_nested_partial_and_orphan_lock_fail_closed(self) -> None:
        plan = self.build_plan()
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "profile"
            nested = output_root / "runs" / "x"
            nested.mkdir(parents=True)
            (nested / "shard.partial").write_text("x", encoding="utf-8")
            preflight, authorization = self.receipts(plan, output_root)
            with self.assertRaises(
                profile.Gate12C2DrawProfileError
            ):
                profile.execute_draw_profile(
                    plan,
                    output_root=output_root,
                    preflight_receipt=preflight,
                    authorization_receipt=authorization,
                )

    def test_stale_lock_recovery_is_explicit_and_refuses_partials(self) -> None:
        plan = self.build_plan()
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "profile"
            output_root.mkdir()
            lock = {
                "schema_version": (
                    "gate12c2_draw_profile_coordinator_lock_v0.1"
                ),
                "plan_payload_sha256": plan[
                    "draw_profile_plan_payload_sha256"
                ],
                "implementation_sha256": plan["implementation_sha256"],
                "authorization_receipt_payload_sha256": "a" * 64,
                "pid": 2147483647,
                "hostname": profile.socket.gethostname(),
                "started_at_utc": "2026-07-25T00:00:00+00:00",
            }
            lock["lock_payload_sha256"] = profile._sha256_bytes(
                profile._canonical_json_bytes(lock)
            )
            lock_path = output_root / profile.COORDINATOR_LOCK_NAME
            lock_path.write_bytes(profile._canonical_json_bytes(lock))
            partial = output_root / ".receipt.1.tmp"
            partial.write_text("partial", encoding="utf-8")
            with self.assertRaisesRegex(
                profile.Gate12C2DrawProfileError,
                "partial artifacts remain",
            ):
                profile.recover_stale_coordinator_lock(
                    plan,
                    output_root=output_root,
                    recovery_id="test-recovery",
                    reason="forced-kill unit test",
                )
            self.assertTrue(lock_path.exists())
            partial.unlink()
            with mock.patch.object(
                profile,
                "_pid_is_running",
                return_value=False,
            ):
                receipt = profile.recover_stale_coordinator_lock(
                    plan,
                    output_root=output_root,
                    recovery_id="test-recovery",
                    reason="forced-kill unit test",
                )
            self.assertFalse(lock_path.exists())
            self.assertEqual(
                receipt["prior_owner_pid_not_running"],
                True,
            )
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "profile"
            output_root.mkdir()
            (output_root / profile.COORDINATOR_LOCK_NAME).write_text(
                "{}",
                encoding="utf-8",
            )
            preflight, authorization = self.receipts(plan, output_root)
            with self.assertRaisesRegex(
                profile.Gate12C2DrawProfileError,
                "coordinator lock",
            ):
                profile.execute_draw_profile(
                    plan,
                    output_root=output_root,
                    preflight_receipt=preflight,
                    authorization_receipt=authorization,
                )

    def test_interrupted_final_receipt_boundary_keeps_lock_and_partial(
        self,
    ) -> None:
        plan = self.build_plan()
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "profile"
            preflight, authorization = self.receipts(plan, output_root)
            call_count = 0

            def fake_execute_with_partial(
                subplan: dict[str, object],
                **kwargs: object,
            ) -> dict[str, object]:
                nonlocal call_count
                call_count += 1
                if call_count == 9:
                    (output_root / ".execution-receipt.json.123.tmp").write_text(
                        "partial",
                        encoding="utf-8",
                    )
                return self._fake_execute(subplan, **kwargs)

            with mock.patch.object(
                profile.shards,
                "execute_development_shard_plan",
                side_effect=fake_execute_with_partial,
            ), mock.patch.object(
                profile.shards,
                "verify_development_shard_index",
                side_effect=self._fake_verify,
            ), mock.patch.object(
                profile.throughput,
                "ProcessTreeRssMonitor",
                _FakeMonitor,
            ), mock.patch.object(
                profile,
                "_physical_ram_bytes",
                return_value=16 * 1024**3,
            ):
                with self.assertRaisesRegex(
                    profile.Gate12C2DrawProfileError,
                    "stale transaction artifacts",
                ):
                    profile.execute_draw_profile(
                        plan,
                        output_root=output_root,
                        preflight_receipt=preflight,
                        authorization_receipt=authorization,
                    )
            self.assertTrue(
                (output_root / profile.COORDINATOR_LOCK_NAME).is_file()
            )
            self.assertTrue(
                (
                    output_root
                    / ".execution-receipt.json.123.tmp"
                ).is_file()
            )
            self.assertFalse(
                (output_root / profile.EXECUTION_RECEIPT_NAME).exists()
            )

    def test_resource_receipt_tampering_fails(self) -> None:
        plan = self.build_plan()
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "profile"
            receipt = self._execute_mocked(plan, output_root)
            resource = json.loads(
                (output_root / profile.RESOURCE_RECEIPT_NAME).read_text(
                    encoding="utf-8"
                )
            )
            for mutation in ("nonhex", "rehash_changed_plan"):
                tampered = json.loads(json.dumps(resource))
                if mutation == "nonhex":
                    tampered[
                        "resource_receipt_payload_sha256"
                    ] = "z" * 64
                else:
                    tampered["draw_profile_plan_payload_sha256"] = "0" * 64
                    tampered.pop("resource_receipt_payload_sha256")
                    tampered[
                        "resource_receipt_payload_sha256"
                    ] = profile._sha256_bytes(
                        profile._canonical_json_bytes(tampered)
                    )
                with self.subTest(mutation=mutation):
                    with self.assertRaises(
                        profile.Gate12C2DrawProfileError
                    ):
                        profile.verify_resource_evidence_chain(
                            plan,
                            receipt,
                            tampered,
                        )

    def test_resource_chain_rejects_surface_environment_and_policy_tampering(
        self,
    ) -> None:
        plan = self.build_plan()
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "profile"
            receipt = self._execute_mocked(plan, output_root)
            resource = json.loads(
                (output_root / profile.RESOURCE_RECEIPT_NAME).read_text(
                    encoding="utf-8"
                )
            )
            mutations = {}

            missing_count = json.loads(json.dumps(resource))
            missing_count["draw_count_rows"] = missing_count[
                "draw_count_rows"
            ][1:]
            mutations["missing_draw_count"] = missing_count

            extra_count = json.loads(json.dumps(resource))
            extra_count["eligible_draw_counts"].append(2047)
            mutations["extra_draw_count"] = extra_count

            environment = json.loads(json.dumps(resource))
            environment["execution_evidence"][
                "numerical_environment_sha256"
            ] = "9" * 64
            environment["execution_evidence"].pop(
                "execution_evidence_payload_sha256"
            )
            environment["execution_evidence"][
                "execution_evidence_payload_sha256"
            ] = profile._sha256_bytes(
                profile._canonical_json_bytes(
                    environment["execution_evidence"]
                )
            )
            environment["execution_evidence_payload_sha256"] = (
                environment["execution_evidence"][
                    "execution_evidence_payload_sha256"
                ]
            )
            mutations["environment_mismatch"] = environment

            policy = json.loads(json.dumps(resource))
            policy["resource_policy"][
                "maximum_process_tree_RSS_fraction_of_physical_RAM"
            ] = 1.0
            policy["resource_policy"].pop(
                "resource_policy_payload_sha256"
            )
            policy["resource_policy"][
                "resource_policy_payload_sha256"
            ] = profile._sha256_bytes(
                profile._canonical_json_bytes(policy["resource_policy"])
            )
            mutations["policy_mismatch"] = policy

            for name, tampered in mutations.items():
                tampered.pop("resource_receipt_payload_sha256")
                tampered["resource_receipt_payload_sha256"] = (
                    profile._sha256_bytes(
                        profile._canonical_json_bytes(tampered)
                    )
                )
                with self.subTest(name=name):
                    with self.assertRaises(
                        profile.Gate12C2DrawProfileError
                    ):
                        profile.verify_resource_evidence_chain(
                            plan,
                            receipt,
                            tampered,
                        )

    def test_canonical_json_rejects_nonfinite_values(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaises(
                    profile.Gate12C2DrawProfileError
                ):
                    profile._canonical_json_bytes({"value": value})


if __name__ == "__main__":
    unittest.main()
