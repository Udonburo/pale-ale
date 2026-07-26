#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from datetime import datetime, timedelta, timezone
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
    def _frozen_worker_fixture(
        root: Path,
    ) -> tuple[Path, dict[str, object], dict[str, str]]:
        implementation = {
            "gate12c2_development_shards.py": "1" * 64,
            "gate12c2_synthetic_lab.py": "2" * 64,
            "gate12c2_throughput_profile.py": "3" * 64,
            "run_gate12c2_development_shards.py": "4" * 64,
        }
        rows = []
        for regime in profile.REGIME_SPECIFICATIONS:
            regime_id = str(regime["regime_id"])
            for worker_count in (1, 2, 4):
                rows.append(
                    {
                        "attempts_per_accepted_draw": 1.0,
                        "compressed_shard_bytes": 100,
                        "configuration_id": (
                            f"{regime_id}__w{worker_count}"
                        ),
                        "disk_free_bytes_after": 10_000,
                        "disk_free_bytes_before": 10_100,
                        "effective_accepted_draws_per_wall_second": 1.0,
                        "endpoint_draw_acceptances": 100,
                        "endpoint_draw_attempts": 100,
                        "exhausted_incomplete_stream_count": 0,
                        "index_payload_sha256": "5" * 64,
                        "inner_valid_draw_count": 255,
                        "merge_validation_before_write_wall_seconds": 0.1,
                        "outer_experiment_count": 4,
                        "output_bytes": 1_000 + worker_count,
                        "plan_payload_sha256": "6" * 64,
                        "process_tree_memory": {
                            "monitor_error": None,
                            "peak_observed_process_count": worker_count,
                            "peak_process_tree_rss_bytes": (
                                1_000_000 * worker_count
                            ),
                            "sample_count": 2,
                            "sample_interval_seconds": 0.1,
                        },
                        "profile_slice": "worker_scaling",
                        "regime_id": regime_id,
                        "rejection_reason_counts": {},
                        "schema_version": (
                            "gate12c2_throughput_configuration_v0.1"
                        ),
                        "scientific_outcomes_exposed_in_profile_receipt": (
                            False
                        ),
                        "scientific_projection_sha256": "7" * 64,
                        "shard_phase_wall_seconds": 1.0,
                        "stderr_payload_sha256": "8" * 64,
                        "stdout_payload_sha256": "9" * 64,
                        "sum_outer_compute_wall_seconds": 1.0,
                        "sum_outer_process_cpu_seconds": 1.0,
                        "sum_serialization_write_wall_seconds": 0.1,
                        "unaccounted_rejection_count": 0,
                        "wall_seconds": 1.0,
                        "worker_count": worker_count,
                        "workload_id": f"worker-scaling::{regime_id}",
                    }
                )
        payload: dict[str, object] = {
            "N2_open": False,
            "N3_open": False,
            "configuration_results": rows,
            "epistemic_status": "development_throughput_only",
            "hardware": {},
            "implementation_sha256": implementation,
            "locked_execution_authorized": False,
            "next_authorization": "none",
            "profile_id": "gate12c2_bounded_worker_scaling_v0.1",
            "profile_plan_payload_sha256": "a" * 64,
            "profile_wall_seconds": 1.0,
            "real_held_out_execution_authorized": False,
            "schema_version": (
                "gate12c2_throughput_profile_receipt_v0.1"
            ),
            "scientific_calibration_result": None,
            "source_commit": "a" * 40,
            "summary": {},
            "surface_id": "development",
            "thread_environment": {},
        }
        payload["profile_receipt_payload_sha256"] = (
            profile._sha256_bytes(profile._canonical_json_bytes(payload))
        )
        path = root / "worker-profile.json"
        path.write_bytes(profile._canonical_json_bytes(payload))
        return path, payload, implementation

    @staticmethod
    @contextmanager
    def _mechanical_evidence_context(
        plan: dict[str, object],
        preflight: dict[str, object],
    ):
        resource = preflight["resource_projection"]
        worker_profile = {
            "path": resource["worker_profile_receipt_path"],
            "file_sha256": resource[
                "worker_profile_receipt_file_sha256"
            ],
            "payload_sha256": resource[
                "worker_profile_receipt_payload_sha256"
            ],
            "payload": {"source_commit": "prior-test-commit"},
            "worker_4_rows": {},
        }
        recovery = preflight["recovery_evidence"]
        carry = preflight["worker_carry_forward_evidence"]
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    profile,
                    "_verify_recovery_bundle_file",
                    return_value={
                        "bundle_path": recovery["bundle_path"],
                        "bundle_file_sha256": recovery[
                            "bundle_file_sha256"
                        ],
                        "bundle_bytes": recovery["bundle_bytes"],
                        "git_bundle_verify": "pass",
                        "advertised_source_commit": plan["source_commit"],
                    },
                )
            )
            stack.enter_context(
                mock.patch.object(
                    profile,
                    "_verify_recovery_bundle",
                    return_value=recovery,
                )
            )
            stack.enter_context(
                mock.patch.object(
                    profile,
                    "_verify_prior_worker_profile",
                    return_value=worker_profile,
                )
            )
            stack.enter_context(
                mock.patch.object(
                    profile,
                    "_verify_worker_carry_forward",
                    return_value=carry,
                )
            )
            stack.enter_context(
                mock.patch.object(
                    profile,
                    "_project_output_bytes",
                    return_value=resource["projected_output_bytes"],
                )
            )
            stack.enter_context(
                mock.patch.object(
                    profile,
                    "_project_peak_process_tree_rss_bytes",
                    return_value=resource[
                        "projected_peak_process_tree_rss_bytes_at_draw_1023"
                    ],
                )
            )
            stack.enter_context(
                mock.patch.object(
                    profile,
                    "_physical_ram_bytes",
                    return_value=resource[
                        "physical_ram_bytes_at_preflight"
                    ],
                )
            )
            stack.enter_context(
                mock.patch.object(
                    profile,
                    "_available_physical_memory_bytes",
                    return_value=resource[
                        "available_physical_memory_bytes_at_preflight"
                    ],
                )
            )
            yield

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
        recovery = {
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
            }
        carry = {
                "path": carry_path.as_posix(),
                "file_sha256": profile._sha256_file(carry_path),
                "payload_sha256": "c" * 64,
            }
        resource_projection = {
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
                "physical_ram_bytes_at_preflight": 16 * 1024**3,
                "available_physical_memory_bytes_at_preflight": 15 * 1024**3,
                "maximum_admitted_peak_process_tree_rss_bytes": (
                    int(16 * 1024**3 * 0.75)
                ),
                "memory_headroom_gate_pass": True,
            }
        checks = profile._build_preflight_check_rows(
            plan,
            output_root_evidence_sha256=profile._sha256_bytes(
                profile._canonical_json_bytes(
                    profile._verify_fresh_output_root(output_root)
                )
            ),
            recovery=recovery,
            worker_carry_forward=carry,
            resource_projection=resource_projection,
        )
        preflight = profile._serialize_mechanical_preflight(
            plan,
            output_root=output_root,
            preflight_id="draw-profile-test-preflight",
            checks=checks,
            recovery=recovery,
            worker_carry_forward=carry,
            resource_projection=resource_projection,
        )
        expiration = (
            datetime.now(timezone.utc) + timedelta(minutes=10)
        ).isoformat()
        with self._mechanical_evidence_context(plan, preflight):
            authorization = profile.build_execution_authorization(
                plan,
                preflight,
                output_root=output_root,
                authorization_id="draw-profile-test-authorization",
                purpose="draw-profile-unit-test",
                expires_at_utc=expiration,
                restore_scratch_root=output_root.parent,
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
            "plan_payload_sha256": subplan["plan_payload_sha256"],
            "outer_experiment_count": len(
                subplan["outer_experiment_indices"]
            ),
            "scientific_projection_sha256": (
                subplan["plan_payload_sha256"]
            ),
            "index_payload_sha256": "f" * 64,
        }

    @staticmethod
    def _verify_resource_chain(
        plan: dict[str, object],
        receipt: dict[str, object],
        resource: dict[str, object],
    ) -> dict[str, object]:
        control = resource["control_lineage"]
        return profile.verify_resource_evidence_chain(
            plan,
            receipt,
            resource,
            preflight_receipt=control["preflight_receipt"],
            authorization_receipt=control["authorization_receipt"],
            consumption_receipt=control["consumption_receipt"],
        )

    def _execute_mocked(
        self,
        plan: dict[str, object],
        output_root: Path,
    ) -> dict[str, object]:
        preflight, authorization = self.receipts(plan, output_root)
        with self._mechanical_evidence_context(
            plan,
            preflight,
        ), mock.patch.object(
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
                restore_scratch_root=output_root.parent,
            )

    def _verify_completed_mocked(
        self,
        plan: dict[str, object],
        output_root: Path,
        preflight: dict[str, object],
        authorization: dict[str, object],
        *,
        verification_side_effect=None,
    ) -> dict[str, object]:
        verifier = verification_side_effect or self._fake_verify
        with self._mechanical_evidence_context(
            plan,
            preflight,
        ), mock.patch.object(
            profile.shards,
            "verify_development_shard_index",
            side_effect=verifier,
        ):
            return profile.execute_draw_profile(
                plan,
                output_root=output_root,
                preflight_receipt=preflight,
                authorization_receipt=authorization,
                restore_scratch_root=output_root.parent,
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
            with self._mechanical_evidence_context(
                plan,
                preflight,
            ):
                with self.assertRaises(
                    profile.Gate12C2DrawProfileError
                ):
                    profile.execute_draw_profile(
                        plan,
                        output_root=output_root,
                        preflight_receipt=preflight,
                        authorization_receipt=changed,
                        restore_scratch_root=output_root.parent,
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
            with self._mechanical_evidence_context(
                plan,
                resource["control_lineage"]["preflight_receipt"],
            ):
                gate = self._verify_resource_chain(
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

    def test_rehashed_attestational_preflight_cannot_authorize(self) -> None:
        plan = self.build_plan()
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "profile"
            preflight, _ = self.receipts(plan, output_root)
            with self.assertRaisesRegex(
                profile.Gate12C2DrawProfileError,
                "git bundle verify failed",
            ):
                profile._verify_preflight(
                    plan,
                    preflight,
                    output_root=output_root,
                    restore_scratch_root=output_root.parent,
                )

            extra = json.loads(json.dumps(preflight))
            extra["checks"]["complete_plan_rebuilt"][
                "scientific_direction"
            ] = "favorable"
            extra.pop("preflight_receipt_payload_sha256")
            extra["preflight_receipt_payload_sha256"] = (
                profile._sha256_bytes(
                    profile._canonical_json_bytes(extra)
                )
            )
            with self._mechanical_evidence_context(plan, extra):
                with self.assertRaisesRegex(
                    profile.Gate12C2DrawProfileError,
                    "unexpected=.*scientific_direction",
                ):
                    profile._verify_preflight(
                        plan,
                        extra,
                        output_root=output_root,
                        restore_scratch_root=output_root.parent,
                    )

    def test_preflight_and_authorization_freshness_are_bounded(self) -> None:
        plan = self.build_plan()
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "profile"
            preflight, _ = self.receipts(plan, output_root)
            expired = json.loads(json.dumps(preflight))
            issued = datetime.now(timezone.utc) - timedelta(hours=2)
            expired["issued_at_utc"] = issued.isoformat()
            expired["expires_at_utc"] = (
                issued + timedelta(minutes=30)
            ).isoformat()
            expired.pop("preflight_receipt_payload_sha256")
            expired["preflight_receipt_payload_sha256"] = (
                profile._sha256_bytes(
                    profile._canonical_json_bytes(expired)
                )
            )
            with self._mechanical_evidence_context(plan, expired):
                with self.assertRaisesRegex(
                    profile.Gate12C2DrawProfileError,
                    "not currently valid",
                ):
                    profile._verify_preflight(
                        plan,
                        expired,
                        output_root=output_root,
                        restore_scratch_root=output_root.parent,
                    )
            with self._mechanical_evidence_context(plan, preflight):
                with self.assertRaisesRegex(
                    profile.Gate12C2DrawProfileError,
                    "freshness window",
                ):
                    profile.build_execution_authorization(
                        plan,
                        preflight,
                        output_root=output_root,
                        authorization_id="too-long",
                        purpose="adversarial freshness test",
                        expires_at_utc=(
                            datetime.now(timezone.utc)
                            + timedelta(hours=2)
                        ).isoformat(),
                        restore_scratch_root=output_root.parent,
                    )

    def test_rehashed_preflight_cannot_bypass_fresh_output_root(self) -> None:
        plan = self.build_plan()
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "profile"
            preflight, _ = self.receipts(plan, output_root)
            (output_root / "runs").mkdir(parents=True)
            (output_root / "runs" / "foreign.txt").write_text(
                "same-plan foreign surface",
                encoding="utf-8",
            )
            with self._mechanical_evidence_context(plan, preflight):
                with self.assertRaisesRegex(
                    profile.Gate12C2DrawProfileError,
                    "nonexistent or empty output root",
                ):
                    profile._verify_preflight(
                        plan,
                        preflight,
                        output_root=output_root,
                        restore_scratch_root=output_root.parent,
                    )

    def test_execution_evidence_rejects_duplicate_configuration(self) -> None:
        plan = self.build_plan()
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "profile"
            self._execute_mocked(plan, output_root)
            evidence = json.loads(
                (output_root / profile.EXECUTION_EVIDENCE_NAME).read_text(
                    encoding="utf-8"
                )
            )
            rows = list(evidence["configuration_results"])
            rows.append(dict(rows[0]))
            measurements = evidence["resource_measurements"]
            with self.assertRaisesRegex(
                profile.Gate12C2DrawProfileError,
                "duplicated",
            ):
                profile._build_execution_evidence(
                    plan,
                    configuration_rows=rows,
                    wall_seconds=measurements["wall_seconds"],
                    process_cpu_seconds=measurements[
                        "process_cpu_seconds"
                    ],
                    process_tree_memory=measurements[
                        "process_tree_memory"
                    ],
                    physical_ram_bytes=measurements[
                        "physical_ram_bytes"
                    ],
                    disk_free_bytes_before=measurements[
                        "disk_free_bytes_before"
                    ],
                    disk_free_bytes_after=measurements[
                        "disk_free_bytes_after"
                    ],
                    output_bytes=measurements[
                        "output_bytes_before_resource_receipts"
                    ],
                )

    def test_completed_fast_path_revalidates_every_lineage_file(self) -> None:
        mutations = (
            "plan",
            "preflight",
            "missing_execution_evidence",
            "different_authorization",
            "current_index",
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                plan = self.build_plan()
                with tempfile.TemporaryDirectory() as temporary:
                    output_root = Path(temporary) / "profile"
                    self._execute_mocked(plan, output_root)
                    preflight = json.loads(
                        (
                            output_root / "control" / "preflight.json"
                        ).read_text(encoding="utf-8")
                    )
                    authorization = json.loads(
                        (
                            output_root / "control" / "authorization.json"
                        ).read_text(encoding="utf-8")
                    )
                    verifier = None
                    supplied_authorization = authorization
                    if mutation == "plan":
                        (output_root / "plan.json").write_text(
                            "{}",
                            encoding="utf-8",
                        )
                    elif mutation == "preflight":
                        (
                            output_root / "control" / "preflight.json"
                        ).write_text("{}", encoding="utf-8")
                    elif mutation == "missing_execution_evidence":
                        (
                            output_root / profile.EXECUTION_EVIDENCE_NAME
                        ).unlink()
                    elif mutation == "different_authorization":
                        supplied_authorization = json.loads(
                            json.dumps(authorization)
                        )
                        supplied_authorization["purpose"] = "other lineage"
                        supplied_authorization.pop(
                            "authorization_receipt_payload_sha256"
                        )
                        supplied_authorization[
                            "authorization_receipt_payload_sha256"
                        ] = profile._sha256_bytes(
                            profile._canonical_json_bytes(
                                supplied_authorization
                            )
                        )
                    else:
                        def mismatched_index(
                            subplan: dict[str, object],
                            **kwargs: object,
                        ) -> dict[str, object]:
                            row = self._fake_verify(subplan, **kwargs)
                            row["index_payload_sha256"] = "0" * 64
                            return row

                        verifier = mismatched_index
                    with self.assertRaises(
                        profile.Gate12C2DrawProfileError
                    ):
                        self._verify_completed_mocked(
                            plan,
                            output_root,
                            preflight,
                            supplied_authorization,
                            verification_side_effect=verifier,
                        )

    def test_resource_lineage_binds_available_memory_evidence(self) -> None:
        plan = self.build_plan()
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "profile"
            receipt = self._execute_mocked(plan, output_root)
            resource = json.loads(
                (output_root / profile.RESOURCE_RECEIPT_NAME).read_text(
                    encoding="utf-8"
                )
            )
            actual_control = resource["control_lineage"]
            tampered = json.loads(json.dumps(resource))
            forged_preflight = tampered["control_lineage"][
                "preflight_receipt"
            ]
            forged_preflight["resource_projection"][
                "available_physical_memory_bytes_at_preflight"
            ] += 1
            forged_preflight.pop("preflight_receipt_payload_sha256")
            forged_preflight["preflight_receipt_payload_sha256"] = (
                profile._sha256_bytes(
                    profile._canonical_json_bytes(forged_preflight)
                )
            )
            tampered["control_lineage"][
                "preflight_receipt_payload_sha256"
            ] = forged_preflight["preflight_receipt_payload_sha256"]
            tampered.pop("resource_receipt_payload_sha256")
            tampered["resource_receipt_payload_sha256"] = (
                profile._sha256_bytes(
                    profile._canonical_json_bytes(tampered)
                )
            )
            with self._mechanical_evidence_context(
                plan,
                actual_control["preflight_receipt"],
            ):
                with self.assertRaises(
                    profile.Gate12C2DrawProfileError
                ):
                    profile.verify_resource_evidence_chain(
                        plan,
                        receipt,
                        tampered,
                        preflight_receipt=actual_control[
                            "preflight_receipt"
                        ],
                        authorization_receipt=actual_control[
                            "authorization_receipt"
                        ],
                        consumption_receipt=actual_control[
                            "consumption_receipt"
                        ],
                    )

    def test_root_stale_partial_fails_before_coordinator_write(self) -> None:
        plan = self.build_plan()
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "profile"
            preflight, authorization = self.receipts(plan, output_root)
            output_root.mkdir()
            stale = output_root / ".plan.json.999999.tmp"
            stale.write_text("stale", encoding="utf-8")
            with self.assertRaisesRegex(
                profile.Gate12C2DrawProfileError,
                "stale transaction artifacts",
            ):
                profile.execute_draw_profile(
                    plan,
                    output_root=output_root,
                    preflight_receipt=preflight,
                    authorization_receipt=authorization,
                    restore_scratch_root=output_root.parent,
                )
            self.assertTrue(stale.exists())
            self.assertFalse((output_root / "plan.json").exists())

    def test_nested_partial_and_orphan_lock_fail_closed(self) -> None:
        plan = self.build_plan()
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "profile"
            preflight, authorization = self.receipts(plan, output_root)
            nested = output_root / "runs" / "x"
            nested.mkdir(parents=True)
            (nested / "shard.partial").write_text("x", encoding="utf-8")
            with self.assertRaises(
                profile.Gate12C2DrawProfileError
            ):
                profile.execute_draw_profile(
                    plan,
                    output_root=output_root,
                    preflight_receipt=preflight,
                    authorization_receipt=authorization,
                    restore_scratch_root=output_root.parent,
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
            preflight, authorization = self.receipts(plan, output_root)
            output_root.mkdir()
            (output_root / profile.COORDINATOR_LOCK_NAME).write_text(
                "{}",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                profile.Gate12C2DrawProfileError,
                "coordinator lock",
            ):
                profile.execute_draw_profile(
                    plan,
                    output_root=output_root,
                    preflight_receipt=preflight,
                    authorization_receipt=authorization,
                    restore_scratch_root=output_root.parent,
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

            with self._mechanical_evidence_context(
                plan,
                preflight,
            ), mock.patch.object(
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
                        restore_scratch_root=output_root.parent,
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
                        with self._mechanical_evidence_context(
                            plan,
                            resource["control_lineage"][
                                "preflight_receipt"
                            ],
                        ):
                            self._verify_resource_chain(
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
                        with self._mechanical_evidence_context(
                            plan,
                            resource["control_lineage"][
                                "preflight_receipt"
                            ],
                        ):
                            self._verify_resource_chain(
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

    def test_worker_profile_requires_exact_frozen_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, payload, implementation = self._frozen_worker_fixture(root)

            def blob_hash(
                _: str,
                relative_path: str,
                **__: object,
            ) -> str:
                return implementation[Path(relative_path).name]

            patches = (
                mock.patch.object(
                    profile,
                    "FROZEN_PRIOR_WORKER_PROFILE_FILE_SHA256",
                    profile._sha256_file(path),
                ),
                mock.patch.object(
                    profile,
                    "FROZEN_PRIOR_WORKER_PROFILE_PAYLOAD_SHA256",
                    payload["profile_receipt_payload_sha256"],
                ),
                mock.patch.object(
                    profile,
                    "FROZEN_PRIOR_WORKER_PROFILE_SOURCE_COMMIT",
                    payload["source_commit"],
                ),
                mock.patch.object(
                    profile,
                    "_git_blob_sha256",
                    side_effect=blob_hash,
                ),
            )
            with ExitStack() as stack:
                for patch in patches:
                    stack.enter_context(patch)
                verified = profile._verify_prior_worker_profile(path)
                self.assertEqual(
                    verified["file_sha256"],
                    profile._sha256_file(path),
                )

                forged = json.loads(json.dumps(payload))
                for row in forged["configuration_results"]:
                    if row["worker_count"] == 4:
                        row["output_bytes"] = 1
                        row["process_tree_memory"][
                            "peak_process_tree_rss_bytes"
                        ] = 1
                forged.pop("profile_receipt_payload_sha256")
                forged["profile_receipt_payload_sha256"] = (
                    profile._sha256_bytes(
                        profile._canonical_json_bytes(forged)
                    )
                )
                forged_path = root / "forged-worker-profile.json"
                forged_path.write_bytes(
                    profile._canonical_json_bytes(forged)
                )
                with self.assertRaisesRegex(
                    profile.Gate12C2DrawProfileError,
                    "not the frozen official file",
                ):
                    profile._verify_prior_worker_profile(forged_path)

    def test_worker_carry_requires_reconstructed_smoke_and_git_blobs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, worker_payload, implementation = (
                self._frozen_worker_fixture(root)
            )
            current_implementation = {
                "gate12c2_development_shards.py": "b" * 64,
                "gate12c2_draw_profile.py": "c" * 64,
                "gate12c2_synthetic_lab.py": "d" * 64,
            }
            prior_by_path = {
                "tools/gate12c2_development_shards.py": implementation[
                    "gate12c2_development_shards.py"
                ],
                "tools/gate12c2_draw_profile.py": None,
                "tools/gate12c2_synthetic_lab.py": implementation[
                    "gate12c2_synthetic_lab.py"
                ],
            }
            current_by_path = {
                "tools/gate12c2_development_shards.py": (
                    current_implementation[
                        "gate12c2_development_shards.py"
                    ]
                ),
                "tools/gate12c2_draw_profile.py": current_implementation[
                    "gate12c2_draw_profile.py"
                ],
                "tools/gate12c2_synthetic_lab.py": current_implementation[
                    "gate12c2_synthetic_lab.py"
                ],
            }
            smoke = {
                "worker_counts": [1, 4],
                "regime_projection_commitments": {
                    str(regime["regime_id"]): {
                        worker: {
                            "plan_payload_sha256": "e" * 64,
                            "scientific_projection_sha256": "f" * 64,
                        }
                        for worker in ("1", "4")
                    }
                    for regime in profile.REGIME_SPECIFICATIONS
                },
                "status": "pass",
                "scientific_outcomes_interpreted": False,
            }
            plan = {
                "source_commit": "b" * 40,
                "implementation_sha256": current_implementation,
                "numerical_environment_sha256": "0" * 64,
            }
            comparison = {}
            for critical_path in (
                "tools/gate12c2_development_shards.py",
                "tools/gate12c2_draw_profile.py",
                "tools/gate12c2_synthetic_lab.py",
            ):
                prior = prior_by_path[critical_path]
                current = current_by_path[critical_path]
                comparison[critical_path] = {
                    "prior_sha256": prior,
                    "current_sha256": current,
                    "status": (
                        "new_shared_path_with_bounded_equivalence_smoke"
                        if prior is None
                        else "changed_with_bounded_equivalence_smoke"
                    ),
                }
            carry: dict[str, object] = {
                "schema_version": (
                    profile.WORKER_CARRY_FORWARD_SCHEMA_VERSION
                ),
                "epistemic_status": (
                    "development_worker_selection_carry_forward_only"
                ),
                "surface_id": "development",
                "prior_worker_profile_file_sha256": (
                    profile._sha256_file(path)
                ),
                "prior_worker_profile_payload_sha256": worker_payload[
                    "profile_receipt_payload_sha256"
                ],
                "prior_commit": worker_payload["source_commit"],
                "current_commit": plan["source_commit"],
                "worker_count": 4,
                "current_implementation_sha256": current_implementation,
                "current_numerical_environment_sha256": (
                    plan["numerical_environment_sha256"]
                ),
                "worker_critical_file_comparison": comparison,
                "bounded_equivalence_smoke": smoke,
                "status": "pass",
                "locked_execution_authorized": False,
                "real_held_out_execution_authorized": False,
                "N2_open": False,
                "N3_open": False,
            }
            carry["carry_forward_receipt_payload_sha256"] = (
                profile._sha256_bytes(
                    profile._canonical_json_bytes(carry)
                )
            )
            carry_path = root / "carry.json"
            carry_path.write_bytes(profile._canonical_json_bytes(carry))
            worker_profile = {
                "path": path.as_posix(),
                "file_sha256": profile._sha256_file(path),
                "payload_sha256": worker_payload[
                    "profile_receipt_payload_sha256"
                ],
                "payload": worker_payload,
                "worker_4_rows": {},
            }

            def git_blob(
                commit: str,
                relative_path: str,
                *,
                allow_missing: bool = False,
            ) -> str | None:
                if commit == worker_payload["source_commit"]:
                    value = prior_by_path[relative_path]
                else:
                    value = current_by_path[relative_path]
                if value is None and not allow_missing:
                    raise profile.Gate12C2DrawProfileError(
                        "missing fixture blob"
                    )
                return value

            with mock.patch.object(
                profile,
                "_git_blob_sha256",
                side_effect=git_blob,
            ), mock.patch.object(
                profile,
                "_run_bounded_worker_equivalence_smoke",
                return_value=smoke,
            ):
                profile._verify_worker_carry_forward(
                    carry_path,
                    plan=plan,
                    worker_profile=worker_profile,
                    smoke_scratch_root=root,
                )

                forged = json.loads(json.dumps(carry))
                for commitments in forged[
                    "bounded_equivalence_smoke"
                ]["regime_projection_commitments"].values():
                    for row in commitments.values():
                        row["plan_payload_sha256"] = "0" * 64
                        row["scientific_projection_sha256"] = "0" * 64
                forged.pop("carry_forward_receipt_payload_sha256")
                forged["carry_forward_receipt_payload_sha256"] = (
                    profile._sha256_bytes(
                        profile._canonical_json_bytes(forged)
                    )
                )
                forged_path = root / "forged-carry.json"
                forged_path.write_bytes(
                    profile._canonical_json_bytes(forged)
                )
                with self.assertRaisesRegex(
                    profile.Gate12C2DrawProfileError,
                    "not mechanically reconstructed",
                ):
                    profile._verify_worker_carry_forward(
                        forged_path,
                        plan=plan,
                        worker_profile=worker_profile,
                        smoke_scratch_root=root,
                    )


if __name__ == "__main__":
    unittest.main()
