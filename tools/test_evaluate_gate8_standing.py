#!/usr/bin/env python3
"""Tests for Gate8 standing evaluation."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import evaluate_gate8_standing as evaluator


class EvaluateGate8StandingTest(unittest.TestCase):
    def write_jsonl(self, path: Path, rows):
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    def write_csv(self, path: Path, fieldnames, rows):
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_conflict_and_quietness_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            registry_path = tmp_path / "sample_registry.jsonl"
            token_csv_path = tmp_path / "token.csv"
            out_dir = tmp_path / "out"

            registry_rows = [
                {
                    "execution_sample_id": 1,
                    "benchmark_sample_id": "gate8_plan_00001",
                    "cell_id": "clean_support",
                    "world_id": "w0",
                    "rendering_id": "r0",
                    "target_id": "t0",
                    "answer_target_type": "consistent_answer",
                    "world_ordinal": 0,
                    "world_type": "genealogy",
                    "is_conflict_intended": False,
                    "is_surface_noise_only": False,
                    "quietness_pair_id": "quiet_pair_genealogy_000",
                },
                {
                    "execution_sample_id": 2,
                    "benchmark_sample_id": "gate8_plan_00002",
                    "cell_id": "surface_noisy_clean",
                    "world_id": "w1",
                    "rendering_id": "r1",
                    "target_id": "t1",
                    "answer_target_type": "consistent_answer",
                    "world_ordinal": 1,
                    "world_type": "genealogy",
                    "is_conflict_intended": False,
                    "is_surface_noise_only": True,
                    "quietness_pair_id": "quiet_pair_genealogy_000",
                },
                {
                    "execution_sample_id": 3,
                    "benchmark_sample_id": "gate8_plan_00003",
                    "cell_id": "direct_contradiction",
                    "world_id": "w2",
                    "rendering_id": "r2",
                    "target_id": "t2",
                    "answer_target_type": "consistent_answer",
                    "world_ordinal": 2,
                    "world_type": "temporal",
                    "is_conflict_intended": True,
                    "is_surface_noise_only": False,
                    "quietness_pair_id": "",
                },
                {
                    "execution_sample_id": 4,
                    "benchmark_sample_id": "gate8_plan_00004",
                    "cell_id": "direct_contradiction",
                    "world_id": "w2",
                    "rendering_id": "r2",
                    "target_id": "t3",
                    "answer_target_type": "conflict_following_wrong_answer",
                    "world_ordinal": 2,
                    "world_type": "temporal",
                    "is_conflict_intended": True,
                    "is_surface_noise_only": False,
                    "quietness_pair_id": "",
                },
                {
                    "execution_sample_id": 5,
                    "benchmark_sample_id": "gate8_plan_00005",
                    "cell_id": "distributed_incompatibility",
                    "world_id": "w3",
                    "rendering_id": "r3",
                    "target_id": "t4",
                    "answer_target_type": "consistent_answer",
                    "world_ordinal": 3,
                    "world_type": "reachability",
                    "is_conflict_intended": True,
                    "is_surface_noise_only": False,
                    "quietness_pair_id": "",
                },
                {
                    "execution_sample_id": 6,
                    "benchmark_sample_id": "gate8_plan_00006",
                    "cell_id": "distributed_incompatibility",
                    "world_id": "w3",
                    "rendering_id": "r3",
                    "target_id": "t5",
                    "answer_target_type": "unsupported_bridge_answer",
                    "world_ordinal": 3,
                    "world_type": "reachability",
                    "is_conflict_intended": True,
                    "is_surface_noise_only": False,
                    "quietness_pair_id": "",
                },
            ]
            self.write_jsonl(registry_path, registry_rows)

            fieldnames = ["sample_id", "step", "loop_outcome", "candidate_metric", "label_token"]
            token_rows = [
                {"sample_id": "1", "step": "0", "loop_outcome": "none", "candidate_metric": "0.10", "label_token": "0"},
                {"sample_id": "1", "step": "1", "loop_outcome": "none", "candidate_metric": "0.20", "label_token": "0"},
                {"sample_id": "2", "step": "0", "loop_outcome": "none", "candidate_metric": "0.30", "label_token": "0"},
                {"sample_id": "2", "step": "1", "loop_outcome": "none", "candidate_metric": "0.40", "label_token": "0"},
                {"sample_id": "3", "step": "0", "loop_outcome": "none", "candidate_metric": "0.10", "label_token": "0"},
                {"sample_id": "3", "step": "1", "loop_outcome": "none", "candidate_metric": "0.20", "label_token": "0"},
                {"sample_id": "4", "step": "0", "loop_outcome": "none", "candidate_metric": "0.90", "label_token": "1"},
                {"sample_id": "4", "step": "1", "loop_outcome": "none", "candidate_metric": "0.80", "label_token": "1"},
                {"sample_id": "5", "step": "0", "loop_outcome": "none", "candidate_metric": "0.10", "label_token": "0"},
                {"sample_id": "5", "step": "1", "loop_outcome": "none", "candidate_metric": "0.20", "label_token": "0"},
                {"sample_id": "6", "step": "0", "loop_outcome": "none", "candidate_metric": "0.85", "label_token": "1"},
                {"sample_id": "6", "step": "1", "loop_outcome": "none", "candidate_metric": "0.75", "label_token": "1"},
            ]
            self.write_csv(token_csv_path, fieldnames, token_rows)

            args = [
                "prog",
                "--sample-registry-jsonl",
                str(registry_path),
                "--token-csv",
                str(token_csv_path),
                "--out-dir",
                str(out_dir),
                "--candidate-id",
                "gate6f",
                "--metric-id",
                "candidate_metric",
            ]
            with mock.patch("sys.argv", args):
                self.assertEqual(evaluator.main(), 0)

            with open(out_dir / "conflict_cell_summary.csv", "r", encoding="utf-8", newline="") as handle:
                conflict_rows = list(csv.DictReader(handle))
            direct = next(row for row in conflict_rows if row["cell_id"] == "direct_contradiction")
            self.assertAlmostEqual(float(direct["global_auprc"]), 1.0, places=10)
            self.assertAlmostEqual(float(direct["mean_hit_at_10"]), 2.0, places=10)

            with open(out_dir / "quietness_summary.csv", "r", encoding="utf-8", newline="") as handle:
                quiet_rows = list(csv.DictReader(handle))
            quiet_all = next(row for row in quiet_rows if row["bucket"] == "all")
            self.assertAlmostEqual(float(quiet_all["mean_delta_max"]), 0.2, places=10)
            self.assertAlmostEqual(float(quiet_all["mean_delta_p90"]), 0.2, places=10)
            self.assertAlmostEqual(float(quiet_all["mean_top10_inflation"]), 2.0, places=10)


if __name__ == "__main__":
    raise SystemExit(unittest.main())
