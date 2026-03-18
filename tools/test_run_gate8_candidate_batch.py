#!/usr/bin/env python3
"""Tests for Gate8 candidate batch helpers."""

import unittest

import run_gate8_candidate_batch as batch


class RunGate8CandidateBatchTest(unittest.TestCase):
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


if __name__ == "__main__":
    raise SystemExit(unittest.main())
