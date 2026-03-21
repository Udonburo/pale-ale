#!/usr/bin/env python3
"""Regression tests for Gate9F conflict-anchor recovery helpers."""

import tempfile
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate9f_conflict_anchor_recovery as gate9f


def make_gate9e_row(
    benchmark_sample_id: str,
    answer_target_type: str,
    *,
    cell_id: str = "distributed_incompatibility",
    dry_run_status: str = "candidate_emitted",
    execution_sample_id: int = 5,
) -> dict:
    return {
        "benchmark_sample_id": benchmark_sample_id,
        "answer_target_type": answer_target_type,
        "cell_id": cell_id,
        "dry_run_status": dry_run_status,
        "execution_sample_id": execution_sample_id,
    }


def make_recovery_row(
    benchmark_sample_id: str,
    *,
    answer_target_type: str = "consistent_answer",
    execution_sample_id: int = 5,
) -> dict:
    return {
        "benchmark_sample_id": benchmark_sample_id,
        "answer_target_type": answer_target_type,
        "execution_sample_id": execution_sample_id,
        "n_steps_written": 9,
        "conflict_anchor_rank": 3,
        "exact_token_match_ratio": 1.0,
    }


class RunGate9FConflictAnchorRecoveryTest(unittest.TestCase):
    def test_select_in_scope_rows_keeps_only_candidate_emitted_focus_rows(self) -> None:
        rows = gate9f.select_in_scope_rows(
            [
                make_gate9e_row("a", "consistent_answer", execution_sample_id=6),
                make_gate9e_row("b", "unsupported_bridge_answer", dry_run_status="blocked"),
                make_gate9e_row("c", "consistent_answer", cell_id="direct_contradiction"),
            ]
        )
        self.assertEqual([row["benchmark_sample_id"] for row in rows], ["a"])

    def test_update_extraction_results_rows_rewrites_sample_dir_and_conflict_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            recovered_dir = Path(temp_dir)
            rows = gate9f.update_extraction_results_rows(
                [
                    {
                        "execution_sample_id": 5,
                        "benchmark_sample_id": "gate8_plan_00005",
                        "sample_dir": "old/path",
                        "conflict_anchor_steps": None,
                        "conflict_anchor_rank": None,
                        "conflict_anchor_exact_token_match_ratio": None,
                    },
                    {
                        "execution_sample_id": 7,
                        "benchmark_sample_id": "gate8_plan_00007",
                        "sample_dir": "old/path",
                        "conflict_anchor_steps": None,
                        "conflict_anchor_rank": None,
                        "conflict_anchor_exact_token_match_ratio": None,
                    },
                ],
                recovered_dir=recovered_dir,
                recovery_rows_by_benchmark={"gate8_plan_00005": make_recovery_row("gate8_plan_00005")},
            )
        self.assertTrue(str(rows[0]["sample_dir"]).endswith("samples/sample_000005"))
        self.assertEqual(rows[0]["conflict_anchor_steps"], 9)
        self.assertEqual(rows[0]["conflict_anchor_rank"], 3)
        self.assertEqual(rows[0]["conflict_anchor_exact_token_match_ratio"], 1.0)
        self.assertTrue(str(rows[1]["sample_dir"]).endswith("samples/sample_000007"))
        self.assertIsNone(rows[1]["conflict_anchor_steps"])

    def test_build_status_payload_keeps_public_judgment_on_gate9d_and_gate9c(self) -> None:
        payload = gate9f.build_status_payload(
            [make_recovery_row("gate8_plan_00005"), make_recovery_row("gate8_plan_00006")],
            gate9c_status={
                "usable_motif_coverage_status": "provisionally_clear",
                "missingness_topology_accounted_status": "clear",
                "operator_admission_status": "denied",
            },
            gate9d_status={
                "coverage_recovery_status": "recovered",
                "frozen_law_recovery_candidate_status": "denied",
            },
        )
        self.assertEqual(payload["materialization_recovery_status"], "materialized")
        self.assertEqual(payload["recovered_row_count"], 2)
        self.assertEqual(payload["gate9d_coverage_recovery_status"], "recovered")
        self.assertEqual(payload["gate9c_usable_motif_coverage_status"], "provisionally_clear")
        self.assertEqual(payload["gate9c_operator_admission_status"], "denied")


if __name__ == "__main__":
    unittest.main()
