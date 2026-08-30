from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import numpy as np

from analysis.gate13_causal_return.review2 import track_c_review2_validator as review2


class TrackCReview2ValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.review2_root = Path(__file__).resolve().parents[1]

    def _analysis_fixture(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        depth = np.repeat(np.asarray([2.0, 4.0, 6.0, 8.0]), 5)
        phase = np.linspace(-1.5, 1.5, depth.size)
        margin = 0.08 * depth + np.sin(phase * 2.3)
        feature = np.cos(phase * 1.7) + 0.12 * margin
        designs = review2.standardized_analysis_designs(depth, margin, feature)
        feature_z = designs["standardized_representation_feature"]
        outcome = (
            0.30 * designs["nuisance"][:, 1]
            + 0.25 * designs["nuisance"][:, 2]
            + 1.20 * feature_z
            + 0.02 * np.sin(np.arange(depth.size))
        )
        return depth, outcome, designs["nuisance"], feature_z

    def test_source_weighted_scalar_is_gauge_invariant(self) -> None:
        rng = np.random.default_rng(1701)
        activations = rng.normal(size=(41, 9))
        frame, _ = np.linalg.qr(rng.normal(size=(9, 4)))
        source_gauge, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        target_gauge, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        delta = rng.normal(size=(4, 4))

        reference = review2.source_weighted_crossfit_scalar(
            delta,
            activations,
            frame,
        )
        transformed = review2.source_weighted_crossfit_scalar(
            target_gauge.T @ delta @ source_gauge,
            activations,
            frame @ source_gauge,
        )
        self.assertAlmostEqual(reference, transformed, places=12)

    def test_behavior_is_aggregated_once_per_block(self) -> None:
        path_p = np.linspace(-0.2, 1.1, 24)
        path_q = path_p - np.linspace(0.0, 0.46, 24)
        result = review2.block_behavioral_summary(path_p, path_q)
        self.assertAlmostEqual(
            result["rms_equivalent_path_margin_discrepancy"],
            math.sqrt(float(np.mean(np.square(path_p - path_q)))),
        )
        self.assertAlmostEqual(
            result["mean_path_averaged_margin"],
            float(np.mean(0.5 * (path_p + path_q))),
        )
        with self.assertRaises(review2.Review2ValidationError):
            review2.block_behavioral_summary(path_p[:-1], path_q[:-1])

    def test_press_implementation_matches_explicit_lobo_refits(self) -> None:
        _depth, outcome, nuisance, feature = self._analysis_fixture()
        observed = review2.relative_lobo_sse_reduction(
            outcome,
            nuisance,
            feature,
        )

        nuisance_errors = []
        full_errors = []
        full = np.column_stack([nuisance, feature])
        for held_out in range(outcome.size):
            keep = np.arange(outcome.size) != held_out
            nuisance_beta, *_ = np.linalg.lstsq(
                nuisance[keep], outcome[keep], rcond=None
            )
            full_beta, *_ = np.linalg.lstsq(full[keep], outcome[keep], rcond=None)
            nuisance_errors.append(
                float(outcome[held_out] - nuisance[held_out] @ nuisance_beta) ** 2
            )
            full_errors.append(
                float(outcome[held_out] - full[held_out] @ full_beta) ** 2
            )
        expected = 1.0 - sum(full_errors) / sum(nuisance_errors)
        self.assertAlmostEqual(
            observed["relative_held_out_sse_reduction"], expected, places=12
        )
        self.assertGreater(observed["relative_held_out_sse_reduction"], 0.95)

    def test_nuisance_preserving_permutation_is_deterministic(self) -> None:
        depth, outcome, nuisance, feature = self._analysis_fixture()
        first = review2.nuisance_preserving_permutation_test(
            outcome,
            nuisance,
            feature,
            depth,
            permutations=199,
            seed=review2.PERMUTATION_SEED,
        )
        second = review2.nuisance_preserving_permutation_test(
            outcome,
            nuisance,
            feature,
            depth,
            permutations=199,
            seed=review2.PERMUTATION_SEED,
        )
        self.assertEqual(first, second)
        self.assertEqual(
            first["procedure"],
            "FREEDMAN_LANE_NUISANCE_RESIDUAL_PERMUTATION_WITHIN_ROLLOUT_DEPTH",
        )
        self.assertEqual(first["permutations"], 199)
        self.assertGreaterEqual(first["p_value_one_sided"], 1.0 / 200.0)

    def test_frozen_forward_and_cost_forecast(self) -> None:
        forecast = review2.forward_and_cost_forecast(20, 24, 24)
        self.assertEqual(forecast["map_activation_forwards"], 4_800)
        self.assertEqual(forecast["behavior_forwards"], 5_760)
        self.assertEqual(forecast["total_scientific_forwards"], 10_560)
        self.assertAlmostEqual(forecast["empirical_linear_forecast_usd"], 45.22264586)
        self.assertAlmostEqual(forecast["planning_upper_usd"], 56.528307325)
        self.assertLessEqual(
            forecast["planning_upper_usd"], review2.FUTURE_MODAL_BUDGET_USD
        )

    def test_existing_b_downsampling_is_reproducible_when_source_is_present(self) -> None:
        repository_root = Path(__file__).resolve().parents[4]
        source = repository_root / (
            "workstream/local/gate13_causal_return_outputs/checkpoint_panel/retrieved/"
            "qwen3_6_27b/executions/b786d648-8ea6-564b-a1cd-0f797c614a00/"
            "fresh_square_operator"
        )
        if not source.is_dir():
            self.skipTest("ignored existing-B evidence is not present in this checkout")
        result = review2.run_fixed_grid_downsampling(source, replicates=24)
        self.assertEqual(result["model_calls"], 0)
        self.assertFalse(result["track_c_outcome_read"])
        self.assertEqual(
            result["subsample_schedule_sha256"],
            "78569ffac2a3cc7826482cf3e3f51e77a4a71413ac34baab843fadfd17ff44f3",
        )
        self.assertEqual(
            [row["all_layer_gate_pass_count"] for row in result["grid"]],
            [2, 4, 3, 1],
        )
        self.assertAlmostEqual(
            result["full_support_primary_feature"],
            0.00018199685461385876,
            places=18,
        )

    def test_candidate_json_freezes_one_feasible_design(self) -> None:
        lock = json.loads(
            (self.review2_root / "track_c_estimand_lock_candidate.json").read_text(
                encoding="utf-8"
            )
        )
        sensitivity = json.loads(
            (self.review2_root / "track_c_sensitivity_and_cost.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            lock["terminal_review2_state"],
            "REVIEW2_READY_FOR_HUMAN_AUTHORIZATION",
        )
        self.assertEqual(lock["primary_test"]["multiplicity"], 1)
        self.assertEqual(
            sensitivity["proposed_design"]["block_count"], 20
        )
        self.assertEqual(
            sensitivity["proposed_design"]["cloud_samples_per_node_per_half"],
            24,
        )
        self.assertEqual(
            sensitivity["proposed_design"]["behavior_episodes_per_block"], 24
        )
        self.assertFalse(sensitivity["future_modal_budget"]["allocation_created"])
        self.assertEqual(sensitivity["model_calls"], 0)
        self.assertFalse(sensitivity["track_c_outcome_read"])

    def test_complete_candidate_package_passes_fail_closed_validator(self) -> None:
        result = review2.validate_candidate_package(self.review2_root)
        self.assertEqual(result["status"], "PASS", result["errors"])
        self.assertEqual(
            result["terminal_review2_state"],
            "REVIEW2_READY_FOR_HUMAN_AUTHORIZATION",
        )


if __name__ == "__main__":
    unittest.main()
