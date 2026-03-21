#!/usr/bin/env python3
"""Regression tests for Gate9H anchor-coverage-gap redesign helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate9h_anchor_coverage_gap_redesign_audit as gate9h


def make_registry_row(
    cell_id: str,
    anchor_kind: str,
    candidate_status: str,
    gap: float | None,
) -> dict:
    return {
        "cell_id": cell_id,
        "anchor_kind": anchor_kind,
        "candidate_status": candidate_status,
        "coverage_gap_abs": gap,
    }


class RunGate9HAnchorCoverageGapRedesignAuditTest(unittest.TestCase):
    def test_candidate_status_is_nontrivial_when_gap_exceeds_tolerance(self) -> None:
        status, reason = gate9h.candidate_status_for_row(
            {"closure_outcome": "none", "coverage_gap_abs": 0.01}
        )
        self.assertEqual(status, "nontrivial_gap_candidate")
        self.assertIn("exceeds", reason)

    def test_candidate_status_is_missing_when_closure_missing(self) -> None:
        status, reason = gate9h.candidate_status_for_row(
            {"closure_outcome": "missing_conflict_anchor", "coverage_gap_abs": None}
        )
        self.assertEqual(status, "missing_or_insufficient")
        self.assertEqual(reason, "missing_conflict_anchor")

    def test_build_status_payload_names_cleaner_dominance_as_next_blocker(self) -> None:
        payload = gate9h.build_status_payload(
            [
                make_registry_row("clean_support", "support", "nontrivial_gap_candidate", 0.15),
                make_registry_row("surface_noisy_clean", "support", "nontrivial_gap_candidate", 0.16),
                make_registry_row("direct_contradiction", "support", "nontrivial_gap_candidate", 0.10),
                make_registry_row("distributed_incompatibility", "support", "nontrivial_gap_candidate", 0.09),
                make_registry_row("direct_contradiction", "conflict", "nontrivial_gap_candidate", 0.11),
                make_registry_row("distributed_incompatibility", "conflict", "nontrivial_gap_candidate", 0.12),
            ]
        )
        self.assertEqual(payload["redesign_candidate_nontriviality_status"], "provisionally_clear")
        self.assertEqual(payload["support_anchor_cleaner_dominance_status"], "triggered")
        self.assertEqual(payload["conflict_anchor_availability_status"], "clear")
        self.assertEqual(payload["redesign_admission_readiness_status"], "denied")
        self.assertEqual(payload["next_named_blocker"], "cleaner_cell_dominance")


if __name__ == "__main__":
    unittest.main()
