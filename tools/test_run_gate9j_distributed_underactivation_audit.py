#!/usr/bin/env python3
"""Regression tests for Gate9J distributed-underactivation helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate9j_distributed_underactivation_audit as gate9j


def make_row(
    cell_id: str,
    answer_target_type: str,
    gap: float,
    answer: float,
    token: float,
    *,
    anchor_kind: str = "support",
) -> dict:
    return {
        "closure_id": f"{cell_id}:{answer_target_type}",
        "benchmark_sample_id": f"{cell_id}:{answer_target_type}",
        "execution_sample_id": 1,
        "cell_id": cell_id,
        "world_id": "w",
        "world_type": "t",
        "answer_target_type": answer_target_type,
        "anchor_kind": anchor_kind,
        "coverage_gap_abs": gap,
        "anchor_answer_coverage": answer,
        "anchor_token_coverage": token,
        "candidate_status": "nontrivial_gap_candidate",
    }


class RunGate9JDistributedUnderactivationAuditTest(unittest.TestCase):
    def test_underactivation_concentrates_on_distributed_consistent_branch(self) -> None:
        rows = [
            make_row("direct_contradiction", "consistent_answer", 0.13, 0.71, 0.58),
            make_row("direct_contradiction", "conflict_following_wrong_answer", 0.14, 0.72, 0.58),
            make_row("distributed_incompatibility", "consistent_answer", 0.05, 0.62, 0.56),
            make_row("distributed_incompatibility", "unsupported_bridge_answer", 0.12, 0.63, 0.51),
        ]
        payload = gate9j.build_status_payload(gate9j.build_registry_rows(rows))
        self.assertEqual(payload["distributed_underactivation_status"], "triggered")
        self.assertEqual(payload["distributed_answer_target_split_status"], "triggered")
        self.assertEqual(payload["distributed_consistent_branch_status"], "underactivated")
        self.assertEqual(payload["direct_baseline_answer_suppression_status"], "triggered")
        self.assertEqual(payload["gap_loss_explained_as_token_only_status"], "denied")
        self.assertEqual(payload["next_named_subblocker"], "distributed_consistent_answer_compression")

    def test_no_target_split_when_distributed_consistent_is_not_below_nonconsistent(self) -> None:
        rows = [
            make_row("direct_contradiction", "consistent_answer", 0.13, 0.71, 0.58),
            make_row("distributed_incompatibility", "consistent_answer", 0.12, 0.64, 0.52),
            make_row("distributed_incompatibility", "unsupported_bridge_answer", 0.10, 0.63, 0.53),
        ]
        payload = gate9j.build_status_payload(gate9j.build_registry_rows(rows))
        self.assertEqual(payload["distributed_answer_target_split_status"], "clear")
        self.assertEqual(payload["distributed_consistent_branch_status"], "clear")

    def test_build_registry_rows_filters_to_support_conflict_cells(self) -> None:
        rows = gate9j.build_registry_rows(
            [
                make_row("direct_contradiction", "consistent_answer", 0.13, 0.71, 0.58),
                make_row("clean_support", "consistent_answer", 0.20, 0.61, 0.41),
                make_row("distributed_incompatibility", "unsupported_bridge_answer", 0.12, 0.63, 0.51),
                make_row("direct_contradiction", "consistent_answer", 0.13, 0.71, 0.58, anchor_kind="conflict"),
            ]
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["branch_kind"], "consistent_answer_branch")
        self.assertEqual(rows[1]["branch_kind"], "nonconsistent_answer_branch")


if __name__ == "__main__":
    unittest.main()
