#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import gate12c2_throughput_profile as profile  # noqa: E402


class Gate12C2ThroughputProfileTest(unittest.TestCase):
    def test_plan_is_deterministic_and_keeps_science_closed(self) -> None:
        first = profile.build_bounded_worker_profile_plan(
            source_commit="test-commit",
            outer_count_per_workload=1,
            inner_valid_draw_count=1,
            worker_counts=(2, 1),
        )
        second = profile.build_bounded_worker_profile_plan(
            source_commit="test-commit",
            outer_count_per_workload=1,
            inner_valid_draw_count=1,
            worker_counts=(1, 2),
        )
        self.assertEqual(first, second)
        verified = profile.verify_profile_plan(first)
        self.assertFalse(verified["locked_execution_authorized"])
        self.assertFalse(verified["real_held_out_execution_authorized"])
        self.assertFalse(verified["N2_open"])
        self.assertFalse(verified["N3_open"])
        self.assertEqual(len(verified["configurations"]), 6)
        self.assertIn(
            "FPR",
            verified["selection_boundary"]["prohibited"],
        )

    def test_process_tree_snapshot_includes_current_process(self) -> None:
        snapshot = profile.process_tree_rss_snapshot(os.getpid())
        self.assertGreater(snapshot["rss_bytes"], 0)
        self.assertIn(str(os.getpid()), snapshot["rss_bytes_by_pid"])

    def test_summary_uses_only_operational_scaling_and_hashes(self) -> None:
        common = {
            "profile_slice": "worker_scaling",
            "workload_id": "worker-scaling::S0_true_null",
            "regime_id": "S0_true_null",
            "outer_experiment_count": 1,
            "inner_valid_draw_count": 1,
            "endpoint_draw_attempts": 96,
            "endpoint_draw_acceptances": 96,
            "attempts_per_accepted_draw": 1.0,
            "rejection_reason_counts": {},
            "unaccounted_rejection_count": 0,
            "exhausted_incomplete_stream_count": 0,
            "sum_outer_compute_wall_seconds": 1.0,
            "sum_outer_process_cpu_seconds": 1.0,
            "sum_serialization_write_wall_seconds": 0.1,
            "shard_phase_wall_seconds": 1.1,
            "merge_validation_before_write_wall_seconds": 0.1,
            "output_bytes": 100,
            "compressed_shard_bytes": 80,
            "disk_free_bytes_before": 1000,
            "disk_free_bytes_after": 900,
            "plan_payload_sha256": "a" * 64,
            "scientific_projection_sha256": "b" * 64,
            "index_payload_sha256": "c" * 64,
            "stdout_payload_sha256": "d" * 64,
            "stderr_payload_sha256": "e" * 64,
            "scientific_outcomes_exposed_in_profile_receipt": False,
        }
        rows = [
            {
                **common,
                "configuration_id": "w1",
                "worker_count": 1,
                "wall_seconds": 2.0,
                "effective_accepted_draws_per_wall_second": 48.0,
                "process_tree_memory": {
                    "peak_process_tree_rss_bytes": 100,
                },
            },
            {
                **common,
                "configuration_id": "w2",
                "worker_count": 2,
                "wall_seconds": 1.0,
                "effective_accepted_draws_per_wall_second": 96.0,
                "process_tree_memory": {
                    "peak_process_tree_rss_bytes": 200,
                },
            },
        ]
        summary = profile.summarize_profile_results(
            rows,
            physical_ram_bytes=1000,
        )
        self.assertTrue(summary["determinism_pass"])
        self.assertTrue(summary["memory_gate_pass"])
        self.assertFalse(summary["scientific_outcomes_interpreted"])
        self.assertEqual(
            summary["workloads"][0]["scaling"][1][
                "speedup_vs_smallest_worker_count"
            ],
            2.0,
        )

    def test_tiny_profile_executes_without_scientific_interpretation(
        self,
    ) -> None:
        plan = profile.build_bounded_worker_profile_plan(
            source_commit="test-commit",
            outer_count_per_workload=1,
            inner_valid_draw_count=1,
            worker_counts=(1,),
        )
        with tempfile.TemporaryDirectory() as temporary:
            receipt = profile.execute_profile_plan(
                plan,
                output_root=Path(temporary) / "profile",
            )
        self.assertIsNone(receipt["scientific_calibration_result"])
        self.assertFalse(receipt["locked_execution_authorized"])
        self.assertEqual(len(receipt["configuration_results"]), 3)
        self.assertTrue(receipt["summary"]["determinism_pass"])
        encoded = json.dumps(receipt, sort_keys=True)
        self.assertNotIn('"claim_promotion"', encoded)
        self.assertNotIn('"grid_outcome"', encoded)
        self.assertNotIn('"identification_success"', encoded)


if __name__ == "__main__":
    unittest.main()
