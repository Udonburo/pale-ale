#!/usr/bin/env python3
"""Regression tests for Gate9D conflict-motif coverage audit helpers."""

import unittest

import run_gate9c_missingness_topology_audit as gate9c
import run_gate9d_conflict_motif_coverage_audit as gate9d


def make_registry_row(
    cell_id: str,
    answer_target_type: str,
    cycle_outcome: str,
    recovery_path_status: str,
    *,
    cleaner_side_risk_status: str = "not_cleaner_side",
) -> dict:
    return {
        "cell_id": cell_id,
        "answer_target_type": answer_target_type,
        "cycle_outcome": cycle_outcome,
        "recovery_path_status": recovery_path_status,
        "cleaner_side_risk_status": cleaner_side_risk_status,
        "benchmark_sample_id": f"{cell_id}:{answer_target_type}",
        "recovery_reason": "test",
    }


class RunGate9DConflictMotifCoverageAuditTest(unittest.TestCase):
    def test_classify_structural_rows_as_not_applicable(self) -> None:
        status, reason = gate9d.classify_recovery_path(
            "missing_conflict_anchor",
            gate9c.STRUCTURAL,
            is_conflict_intended=False,
            has_conflict_chunk_declared=False,
            has_conflict_anchor_materialized=False,
        )
        self.assertEqual(status, gate9d.NOT_APPLICABLE_STRUCTURAL)
        self.assertIn("not_licensed", reason)

    def test_classify_bundle_specific_declared_conflict_as_recoverable_candidate(self) -> None:
        status, reason = gate9d.classify_recovery_path(
            "missing_conflict_anchor",
            gate9c.BUNDLE_SPECIFIC,
            is_conflict_intended=True,
            has_conflict_chunk_declared=True,
            has_conflict_anchor_materialized=False,
        )
        self.assertEqual(status, gate9d.RECOVERABLE_CANDIDATE)
        self.assertIn("declared_conflict_chunk", reason)

    def test_classify_missing_declared_conflict_as_blocked_without_law_change(self) -> None:
        status, reason = gate9d.classify_recovery_path(
            "missing_conflict_anchor",
            gate9c.BUNDLE_SPECIFIC,
            is_conflict_intended=True,
            has_conflict_chunk_declared=False,
            has_conflict_anchor_materialized=False,
        )
        self.assertEqual(status, gate9d.BLOCKED_WITHOUT_LAW_CHANGE)
        self.assertIn("no_declared_conflict_chunk", reason)

    def test_build_status_payload_reports_candidate_without_cleaner_pollution(self) -> None:
        payload = gate9d.build_status_payload(
            [
                make_registry_row(
                    "distributed_incompatibility",
                    "consistent_answer",
                    "missing_conflict_anchor",
                    gate9d.RECOVERABLE_CANDIDATE,
                ),
                make_registry_row(
                    "direct_contradiction",
                    "consistent_answer",
                    "none",
                    gate9d.ALREADY_COVERED,
                ),
                make_registry_row(
                    "clean_support",
                    "consistent_answer",
                    "missing_conflict_anchor",
                    gate9d.NOT_APPLICABLE_STRUCTURAL,
                    cleaner_side_risk_status="clear",
                ),
            ]
        )
        self.assertEqual(payload["coverage_recovery_status"], "not_yet_recovered")
        self.assertEqual(payload["frozen_law_recovery_candidate_status"], "candidate_present")
        self.assertEqual(payload["cleaner_side_pollution_status"], "clear")
        self.assertEqual(payload["implementation_bound_gap_status"], "clear")
        self.assertEqual(payload["law_change_required_status"], "clear")


if __name__ == "__main__":
    unittest.main()
