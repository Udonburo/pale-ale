#!/usr/bin/env python3
"""Tests for Gate8 candidate batch helpers."""

import unittest

import numpy as np

import run_gate8_candidate_batch as batch


class RunGate8CandidateBatchTest(unittest.TestCase):
    def test_fixed_candidate_contract_tracks_granularity(self):
        contract_rows = batch.fixed_candidate_contract_rows()
        self.assertEqual(
            [row["candidate_id"] for row in contract_rows],
            ["F", "gate6f", "gate6h", "gate7c"],
        )
        self.assertEqual(
            [row["label_granularity"] for row in contract_rows],
            ["token", "token", "token", "transition"],
        )
        self.assertEqual(contract_rows[-1]["label_key"], "label_transition")

    def test_quietness_pair_bindings_require_shared_worlds(self):
        benchmark_rows = [
            {"sample_id": "a", "cell_id": "clean_support", "world_type": "genealogy", "answer_target_type": "consistent_answer", "world_id": "w0", "rendering_id": "r0"},
            {"sample_id": "b", "cell_id": "surface_noisy_clean", "world_type": "genealogy", "answer_target_type": "consistent_answer", "world_id": "w0", "rendering_id": "r1"},
            {"sample_id": "c", "cell_id": "clean_support", "world_type": "temporal", "answer_target_type": "consistent_answer", "world_id": "w1", "rendering_id": "r2"},
            {"sample_id": "d", "cell_id": "surface_noisy_clean", "world_type": "temporal", "answer_target_type": "consistent_answer", "world_id": "w1", "rendering_id": "r3"},
        ]
        mapping, pair_rows = batch.quietness_pair_bindings(benchmark_rows)
        self.assertEqual(mapping["a"], "quiet_pair_w0")
        self.assertEqual(mapping["b"], "quiet_pair_w0")
        self.assertEqual(mapping["c"], "quiet_pair_w1")
        self.assertEqual(mapping["d"], "quiet_pair_w1")
        self.assertEqual(pair_rows[0]["world_id"], "w0")
        self.assertEqual(pair_rows[0]["clean_rendering_id"], "r0")
        self.assertEqual(pair_rows[0]["surface_noisy_rendering_id"], "r1")
        self.assertEqual(len(pair_rows), 2)

    def test_quietness_pair_bindings_reject_missing_same_world_partner(self):
        benchmark_rows = [
            {"sample_id": "a", "cell_id": "clean_support", "world_type": "genealogy", "answer_target_type": "consistent_answer", "world_id": "w0", "rendering_id": "r0"},
            {"sample_id": "b", "cell_id": "surface_noisy_clean", "world_type": "genealogy", "answer_target_type": "consistent_answer", "world_id": "w1", "rendering_id": "r1"},
        ]
        with self.assertRaisesRegex(ValueError, "quietness pairing requires shared clean/noisy rows"):
            batch.quietness_pair_bindings(benchmark_rows)

    def test_build_sample_registry_assigns_execution_ids(self):
        benchmark_rows = [
            {
                "sample_id": "gate8_plan_00002",
                "cell_id": "surface_noisy_clean",
                "world_id": "w2",
                "rendering_id": "r2",
                "target_id": "t2",
                "answer_target_type": "consistent_answer",
                "world_ordinal": 1,
                "world_type": "temporal",
                "is_conflict_intended": False,
                "is_surface_noise_only": True,
            },
            {
                "sample_id": "gate8_plan_00001",
                "cell_id": "clean_support",
                "world_id": "w1",
                "rendering_id": "r1",
                "target_id": "t1",
                "answer_target_type": "consistent_answer",
                "world_ordinal": 0,
                "world_type": "temporal",
                "is_conflict_intended": False,
                "is_surface_noise_only": False,
            },
        ]
        with self.assertRaisesRegex(ValueError, "quietness pairing requires shared clean/noisy rows"):
            batch.build_sample_registry(benchmark_rows)

    def test_bridge_metrics_are_zero_for_identity_transition(self):
        current_basis = np.zeros((4, 3), dtype=np.float64)
        next_basis = np.zeros((4, 3), dtype=np.float64)
        current_basis[:, 0] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        next_basis[:, 0] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        singular_values = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        next_coords = np.zeros((3, 3), dtype=np.float64)
        next_coords[0, 0] = 1.0

        metrics = batch.compute_rotation_leakage_bridge_metrics(
            current_basis=current_basis,
            current_singular_values=singular_values,
            current_rank=1,
            next_basis=next_basis,
            next_singular_values=singular_values,
            next_coords_local=next_coords,
            next_rank=1,
        )

        self.assertEqual(metrics["bridge_outcome"], "none")
        self.assertAlmostEqual(float(metrics["rotation_only"]), 0.0, places=10)
        self.assertAlmostEqual(float(metrics["leakage_only"]), 0.0, places=10)
        self.assertAlmostEqual(float(metrics["closure_defect"]), 0.0, places=10)

    def test_bridge_metrics_keep_in_span_anisotropic_loss_out_of_leakage(self):
        current_basis = np.zeros((4, 3), dtype=np.float64)
        next_basis = np.zeros((4, 3), dtype=np.float64)
        current_basis[:, 0] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        current_basis[:, 1] = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        next_basis[:, 0] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        next_basis[:, 1] = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        current_singular_values = np.asarray([1.0, 0.5, 0.0], dtype=np.float64)
        next_singular_values = np.asarray([1.0, 0.5, 0.0], dtype=np.float64)
        next_coords = np.zeros((3, 3), dtype=np.float64)
        next_coords[1, 0] = 1.0

        metrics = batch.compute_rotation_leakage_bridge_metrics(
            current_basis=current_basis,
            current_singular_values=current_singular_values,
            current_rank=2,
            next_basis=next_basis,
            next_singular_values=next_singular_values,
            next_coords_local=next_coords,
            next_rank=2,
        )

        self.assertEqual(metrics["bridge_outcome"], "none")
        self.assertAlmostEqual(float(metrics["rotation_only"]), 0.0, places=10)
        self.assertAlmostEqual(float(metrics["leakage_only"]), 0.0, places=10)
        self.assertAlmostEqual(float(metrics["closure_defect"]), 0.9375, places=10)

    def test_bridge_metrics_mark_orthogonal_escape_as_leakage(self):
        current_basis = np.zeros((4, 3), dtype=np.float64)
        next_basis = np.zeros((4, 3), dtype=np.float64)
        current_basis[:, 0] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        next_basis[:, 0] = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        singular_values = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        next_coords = np.zeros((3, 3), dtype=np.float64)
        next_coords[0, 0] = 1.0

        metrics = batch.compute_rotation_leakage_bridge_metrics(
            current_basis=current_basis,
            current_singular_values=singular_values,
            current_rank=1,
            next_basis=next_basis,
            next_singular_values=singular_values,
            next_coords_local=next_coords,
            next_rank=1,
        )

        self.assertEqual(metrics["bridge_outcome"], "none")
        self.assertAlmostEqual(float(metrics["rotation_only"]), 1.0, places=10)
        self.assertAlmostEqual(float(metrics["leakage_only"]), 1.0, places=10)
        self.assertAlmostEqual(float(metrics["closure_defect"]), 0.0, places=10)

    def test_bridge_cell_aggregate_groups_sample_rows(self):
        per_sample_rows = [
            {
                "cell_id": "clean_support",
                "n_transition_rows_total": 4,
                "n_transition_rows_valid": 4,
                "n_transition_rows_missing": 0,
                "mean_rotation_only": 0.1,
                "p90_rotation_only": 0.2,
                "max_rotation_only": 0.3,
                "mean_leakage_only": 0.01,
                "p90_leakage_only": 0.02,
                "max_leakage_only": 0.03,
                "mean_closure_defect": 0.001,
                "p90_closure_defect": 0.002,
                "max_closure_defect": 0.003,
            },
            {
                "cell_id": "clean_support",
                "n_transition_rows_total": 6,
                "n_transition_rows_valid": 5,
                "n_transition_rows_missing": 1,
                "mean_rotation_only": 0.3,
                "p90_rotation_only": 0.4,
                "max_rotation_only": 0.5,
                "mean_leakage_only": 0.05,
                "p90_leakage_only": 0.06,
                "max_leakage_only": 0.07,
                "mean_closure_defect": 0.004,
                "p90_closure_defect": 0.005,
                "max_closure_defect": 0.006,
            },
        ]

        rows = batch.build_rotation_leakage_by_cell_rows(per_sample_rows)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["cell_id"], "clean_support")
        self.assertEqual(rows[0]["n_transition_rows_total"], 10)
        self.assertEqual(rows[0]["n_transition_rows_valid"], 9)
        self.assertEqual(rows[0]["n_transition_rows_missing"], 1)
        self.assertAlmostEqual(float(rows[0]["mean_sample_mean_rotation_only"]), 0.2, places=10)
        self.assertAlmostEqual(float(rows[0]["mean_sample_mean_leakage_only"]), 0.03, places=10)
        self.assertAlmostEqual(float(rows[0]["mean_sample_mean_closure_defect"]), 0.0025, places=10)

    def test_bridge_report_carries_failure_read(self):
        report = batch.build_rotation_leakage_bridge_report(
            run_id="gate8_bridge_test",
            per_sample_rows=[{"sample_id": 1}],
            by_cell_rows=[
                {
                    "cell_id": "surface_noisy_clean",
                    "n_samples": 1,
                    "n_transition_rows_valid": 10,
                    "mean_sample_mean_rotation_only": 0.6,
                    "mean_sample_mean_leakage_only": 0.3,
                    "mean_sample_mean_closure_defect": 0.5,
                    "mean_sample_p90_rotation_only": 0.7,
                    "mean_sample_p90_leakage_only": 0.4,
                    "mean_sample_p90_closure_defect": 0.61,
                },
                {
                    "cell_id": "direct_contradiction",
                    "n_samples": 1,
                    "n_transition_rows_valid": 10,
                    "mean_sample_mean_rotation_only": 0.54,
                    "mean_sample_mean_leakage_only": 0.25,
                    "mean_sample_mean_closure_defect": 0.51,
                    "mean_sample_p90_rotation_only": 0.64,
                    "mean_sample_p90_leakage_only": 0.39,
                    "mean_sample_p90_closure_defect": 0.59,
                },
                {
                    "cell_id": "distributed_incompatibility",
                    "n_samples": 1,
                    "n_transition_rows_valid": 10,
                    "mean_sample_mean_rotation_only": 0.57,
                    "mean_sample_mean_leakage_only": 0.31,
                    "mean_sample_mean_closure_defect": 0.50,
                    "mean_sample_p90_rotation_only": 0.65,
                    "mean_sample_p90_leakage_only": 0.53,
                    "mean_sample_p90_closure_defect": 0.62,
                },
            ],
        )

        self.assertIn("## Failure Read", report)
        self.assertIn("highest mean is surface_noisy_clean=0.600000", report)
        self.assertIn("lowest mean is direct_contradiction=0.250000", report)
        self.assertIn("highest p90 is distributed_incompatibility=0.620000, runner-up is surface_noisy_clean=0.610000", report)
        self.assertIn("bridge v1 should be read as an explanatory-cut failure", report)


if __name__ == "__main__":
    raise SystemExit(unittest.main())
