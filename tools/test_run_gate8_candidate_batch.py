#!/usr/bin/env python3
"""Tests for Gate8 candidate batch helpers."""

import unittest

import run_gate8_candidate_batch as batch


class RunGate8CandidateBatchTest(unittest.TestCase):
    def test_quietness_pair_bindings_are_world_type_stable(self):
        benchmark_rows = [
            {"sample_id": "a", "cell_id": "clean_support", "world_type": "genealogy", "answer_target_type": "consistent_answer", "world_id": "w0"},
            {"sample_id": "b", "cell_id": "clean_support", "world_type": "genealogy", "answer_target_type": "consistent_answer", "world_id": "w1"},
            {"sample_id": "c", "cell_id": "surface_noisy_clean", "world_type": "genealogy", "answer_target_type": "consistent_answer", "world_id": "w2"},
            {"sample_id": "d", "cell_id": "surface_noisy_clean", "world_type": "genealogy", "answer_target_type": "consistent_answer", "world_id": "w3"},
            {"sample_id": "e", "cell_id": "clean_support", "world_type": "temporal", "answer_target_type": "consistent_answer", "world_id": "w4"},
            {"sample_id": "f", "cell_id": "surface_noisy_clean", "world_type": "temporal", "answer_target_type": "consistent_answer", "world_id": "w5"},
        ]
        mapping, pair_rows = batch.quietness_pair_bindings(benchmark_rows)
        self.assertEqual(mapping["a"], "quiet_pair_genealogy_000")
        self.assertEqual(mapping["c"], "quiet_pair_genealogy_000")
        self.assertEqual(mapping["b"], "quiet_pair_genealogy_001")
        self.assertEqual(mapping["d"], "quiet_pair_genealogy_001")
        self.assertEqual(mapping["e"], "quiet_pair_temporal_000")
        self.assertEqual(mapping["f"], "quiet_pair_temporal_000")
        self.assertEqual(len(pair_rows), 3)

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
        registry_rows, quiet_rows = batch.build_sample_registry(benchmark_rows)
        self.assertEqual(registry_rows[0]["execution_sample_id"], 1)
        self.assertEqual(registry_rows[0]["benchmark_sample_id"], "gate8_plan_00001")
        self.assertEqual(registry_rows[1]["execution_sample_id"], 2)
        self.assertEqual(registry_rows[1]["benchmark_sample_id"], "gate8_plan_00002")
        self.assertEqual(registry_rows[0]["quietness_pair_id"], "quiet_pair_temporal_000")
        self.assertEqual(registry_rows[1]["quietness_pair_id"], "quiet_pair_temporal_000")
        self.assertEqual(len(quiet_rows), 1)


if __name__ == "__main__":
    raise SystemExit(unittest.main())
