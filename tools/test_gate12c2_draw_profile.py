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


class Gate12C2DrawProfileTest(unittest.TestCase):
    def build_plan(self) -> dict[str, object]:
        return profile.build_draw_profile_plan(
            source_commit="test-source-commit"
        )

    def receipts(
        self,
        plan: dict[str, object],
        output_root: Path,
    ) -> tuple[dict[str, object], dict[str, object]]:
        checks = {
            key: True for key in profile.REQUIRED_PREFLIGHT_CHECKS
        }
        preflight = profile.build_no_outcome_preflight(
            plan,
            output_root=output_root,
            preflight_id="draw-profile-test-preflight",
            recovery_bundle_sha256="a" * 64,
            checks=checks,
        )
        authorization = profile.build_execution_authorization(
            plan,
            preflight,
            output_root=output_root,
            authorization_id="draw-profile-test-authorization",
            purpose="draw-profile-unit-test",
        )
        return preflight, authorization

    def test_plan_exactly_fixes_nine_configurations(self) -> None:
        plan = self.build_plan()
        verified = profile.verify_draw_profile_plan(plan)
        self.assertEqual(len(verified["configurations"]), 9)
        self.assertEqual(verified["worker_count"], 4)
        self.assertEqual(verified["prefix_counts"], [255, 511, 1023])
        self.assertFalse(verified["locked_execution_authorized"])
        self.assertFalse(verified["real_held_out_execution_authorized"])
        expected_outer_counts = {
            "S0_true_null": 128,
            "S1_known_reverse_shared_node_coupling": 64,
            "S2_null_inflation": 64,
        }
        for configuration in verified["configurations"]:
            subplan = configuration["subplan"]
            self.assertEqual(
                len(subplan["outer_experiment_indices"]),
                expected_outer_counts[configuration["regime_id"]],
            )
            self.assertEqual(
                subplan["inner_valid_draw_count"],
                configuration["draw_count"],
            )

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
        changed = json.loads(json.dumps(plan))
        changed["configurations"][0]["subplan"][
            "inner_valid_draw_count"
        ] = 7
        changed.pop("draw_profile_plan_payload_sha256")
        changed["draw_profile_plan_payload_sha256"] = (
            profile._sha256_bytes(
                profile._canonical_json_bytes(changed)
            )
        )
        with self.assertRaises(profile.Gate12C2DrawProfileError):
            profile.verify_draw_profile_plan(changed)

    def test_preflight_is_non_authorizing_and_bundle_bound(self) -> None:
        plan = self.build_plan()
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "profile"
            preflight, authorization = self.receipts(
                plan,
                output_root,
            )
        self.assertFalse(preflight["development_execution_authorized"])
        self.assertFalse(preflight["scientific_outcomes_inspected"])
        self.assertEqual(preflight["recovery_bundle_sha256"], "a" * 64)
        self.assertTrue(
            authorization["development_execution_authorized"]
        )
        self.assertFalse(
            authorization[
                "scientific_calibration_interpretation_authorized"
            ]
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

    def test_mocked_coordinator_emits_no_scientific_outcomes(self) -> None:
        plan = self.build_plan()

        def fake_execute(
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
                            "unaccounted_rejection_count": 0,
                            "exhausted_incomplete_stream_count": 0,
                        }
                    }
                ],
            }

        def fake_verify(
            subplan: dict[str, object],
            **_: object,
        ) -> dict[str, object]:
            return {
                "scientific_projection_sha256": (
                    subplan["plan_payload_sha256"]
                ),
                "index_payload_sha256": "b" * 64,
            }

        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "profile"
            preflight, authorization = self.receipts(
                plan,
                output_root,
            )
            with mock.patch.object(
                profile.shards,
                "execute_development_shard_plan",
                side_effect=fake_execute,
            ), mock.patch.object(
                profile.shards,
                "verify_development_shard_index",
                side_effect=fake_verify,
            ):
                receipt = profile.execute_draw_profile(
                    plan,
                    output_root=output_root,
                    preflight_receipt=preflight,
                    authorization_receipt=authorization,
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

    def test_canonical_json_rejects_nonfinite_values(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaises(
                    profile.Gate12C2DrawProfileError
                ):
                    profile._canonical_json_bytes({"value": value})


if __name__ == "__main__":
    unittest.main()
