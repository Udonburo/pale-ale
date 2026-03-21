#!/usr/bin/env python3
"""Regression tests for Gate9G anchor-conditioned triviality helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate9g_anchor_conditioned_triviality_audit as gate9g


def make_registry_row(
    triviality_status: str,
    *,
    closure_outcome: str = "none",
) -> dict:
    return {
        "triviality_status": triviality_status,
        "closure_outcome": closure_outcome,
    }


class RunGate9GAnchorConditionedTrivialityAuditTest(unittest.TestCase):
    def test_classify_full_anchor_span_collapse(self) -> None:
        status, reason = gate9g.classify_triviality(
            closure_outcome="none",
            closure_defect=1e-16,
            anchor_rank=3,
            answer_conditioned_rank=3,
            token_conditioned_rank=3,
        )
        self.assertEqual(status, gate9g.FULL_ANCHOR_SPAN_COLLAPSE)
        self.assertIn("saturate", reason)

    def test_classify_nontrivial_signal_candidate(self) -> None:
        status, reason = gate9g.classify_triviality(
            closure_outcome="none",
            closure_defect=0.02,
            anchor_rank=3,
            answer_conditioned_rank=2,
            token_conditioned_rank=2,
        )
        self.assertEqual(status, gate9g.NONTRIVIAL_SIGNAL_CANDIDATE)
        self.assertIn("exceeds", reason)

    def test_build_status_payload_triggers_blocker_when_only_collapse_rows_exist(self) -> None:
        payload = gate9g.build_status_payload(
            [
                make_registry_row(gate9g.FULL_ANCHOR_SPAN_COLLAPSE),
                make_registry_row(gate9g.FULL_ANCHOR_SPAN_COLLAPSE),
                make_registry_row(gate9g.MISSING_OR_INSUFFICIENT, closure_outcome="missing_conflict_anchor"),
            ]
        )
        self.assertEqual(payload["non_trivial_anchor_conditioned_read_status"], "denied")
        self.assertEqual(payload["full_anchor_span_collapse_status"], "triggered")
        self.assertEqual(payload["operator_admission_blocker_status"], "triggered")


if __name__ == "__main__":
    unittest.main()
