#!/usr/bin/env python3
"""Regression tests for Gate10F pre-closeout judgment helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate10f_pre_closeout_judgment as gate10f


def make_gate10e_status(
    three_slice_pattern_status: str = "supported",
    interim_broader_judgment_status: str = "bounded_support",
    pre_closeout_readiness_status: str = "ready",
    operator_admission_still_denied_status: str = "confirmed",
    retroactive_reinterpretation_forbidden_status: str = "confirmed",
    broader_trusted_tree_settlement_status: str = "unearned",
    gate10b_slice_settled_status: str = "preserved",
    gate10c_slice_settled_status: str = "preserved",
    gate10d_slice_settled_status: str = "preserved",
) -> dict:
    return {
        "gate10b_slice_settled_status": gate10b_slice_settled_status,
        "gate10c_slice_settled_status": gate10c_slice_settled_status,
        "gate10d_slice_settled_status": gate10d_slice_settled_status,
        "three_slice_pattern_status": three_slice_pattern_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "retroactive_reinterpretation_forbidden_status": retroactive_reinterpretation_forbidden_status,
        "broader_trusted_tree_settlement_status": broader_trusted_tree_settlement_status,
        "interim_broader_judgment_status": interim_broader_judgment_status,
        "pre_closeout_readiness_status": pre_closeout_readiness_status,
        "next_named_blocker": "",
    }


class RunGate10FPreCloseoutJudgmentTest(unittest.TestCase):
    def test_closeout_supported_when_gate10e_preserves_bounded_support(self) -> None:
        status = gate10f.build_status_payload(make_gate10e_status())

        self.assertEqual(status["bounded_support_preservation_status"], "preserved")
        self.assertEqual(status["pre_closeout_readiness_preservation_status"], "preserved")
        self.assertEqual(status["broader_trusted_tree_settlement_status"], "unearned")
        self.assertEqual(status["overclaim_pressure_status"], "absent")
        self.assertEqual(status["closeout_sentence_support_status"], "supported")
        self.assertEqual(status["closeout_judgment_outcome_status"], "closeout_supported")
        self.assertEqual(status["post_closeout_memory_readiness_status"], "ready")
        self.assertEqual(status["next_named_blocker"], "")

    def test_not_yet_closeable_when_three_slice_pattern_not_preserved(self) -> None:
        gate10e_status = make_gate10e_status(three_slice_pattern_status="not_supported")
        status = gate10f.build_status_payload(gate10e_status)

        self.assertEqual(status["bounded_support_preservation_status"], "preserved")
        self.assertEqual(status["closeout_sentence_support_status"], "not_supported")
        self.assertEqual(status["closeout_judgment_outcome_status"], "not_yet_closeable")
        self.assertEqual(status["next_named_blocker"], "three_slice_pattern_not_preserved")

    def test_not_yet_closeable_when_bounded_support_is_not_preserved(self) -> None:
        gate10e_status = make_gate10e_status(interim_broader_judgment_status="not_yet_supported")
        status = gate10f.build_status_payload(gate10e_status)

        self.assertEqual(status["bounded_support_preservation_status"], "not_preserved")
        self.assertEqual(status["closeout_judgment_outcome_status"], "not_yet_closeable")
        self.assertEqual(status["next_named_blocker"], "bounded_broader_support_not_preserved")

    def test_not_yet_closeable_when_overclaim_pressure_appears(self) -> None:
        gate10e_status = make_gate10e_status(
            broader_trusted_tree_settlement_status="pressure_to_overclaim"
        )
        status = gate10f.build_status_payload(gate10e_status)

        self.assertEqual(status["broader_trusted_tree_settlement_status"], "pressure_to_overclaim")
        self.assertEqual(status["overclaim_pressure_status"], "present")
        self.assertEqual(status["closeout_sentence_support_status"], "not_supported")
        self.assertEqual(
            status["next_named_blocker"],
            "broader_trusted_tree_settlement_overclaim_pressure",
        )

    def test_policy_compare_echoes_single_source_row(self) -> None:
        manifest = {"run_id": "gate10e_run", "code_git_commit": "abc123"}
        registry = gate10f.build_registry(manifest, make_gate10e_status())
        compare = gate10f.build_policy_compare(registry)

        self.assertEqual(len(compare), 1)
        self.assertEqual(compare[0]["source_gate10e_run_id"], "gate10e_run")
        self.assertEqual(compare[0]["three_slice_pattern_status"], "supported")


if __name__ == "__main__":
    unittest.main()