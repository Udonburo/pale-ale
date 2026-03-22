#!/usr/bin/env python3
"""Regression tests for Gate10E interim broader judgment helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate10e_interim_broader_judgment as gate10e


def make_source_run(
    slice_id: str,
    comparison_outcome_status: str = "settled",
    baseline_status: str = "clear",
    operator_status: str = "confirmed",
    retroactive_status: str = "clear",
    broader_non_promotion_status: str = "clear",
    next_named_blocker: str = "",
) -> dict:
    return {
        "slice_id": slice_id,
        "source_dir": f"runs/{slice_id}",
        "source_run_id": f"{slice_id}_run",
        "source_code_git_commit": "abc123",
        "status_payload": {
            "comparison_outcome_status": comparison_outcome_status,
            "forward_basis_baseline_preservation_status": baseline_status,
            "operator_admission_still_denied_status": operator_status,
            "non_retroactive_memory_preservation_status": retroactive_status,
            "broader_tree_settlement_non_promotion_status": broader_non_promotion_status,
            "next_named_blocker": next_named_blocker,
        },
        "slice_settled_status": "preserved"
        if comparison_outcome_status == "settled"
        else "not_preserved",
        "retroactive_guard_status": "confirmed"
        if retroactive_status == "clear"
        else "violated",
        "next_named_blocker": next_named_blocker,
    }


class RunGate10EInterimBroaderJudgmentTest(unittest.TestCase):
    def test_bounded_support_when_all_three_slices_are_preserved(self) -> None:
        source_runs = [
            make_source_run("gate10b"),
            make_source_run("gate10c"),
            make_source_run("gate10d"),
        ]

        status = gate10e.build_status_payload(source_runs)

        self.assertEqual(status["gate10b_slice_settled_status"], "preserved")
        self.assertEqual(status["gate10c_slice_settled_status"], "preserved")
        self.assertEqual(status["gate10d_slice_settled_status"], "preserved")
        self.assertEqual(status["three_slice_pattern_status"], "supported")
        self.assertEqual(status["broader_trusted_tree_settlement_status"], "unearned")
        self.assertEqual(status["interim_broader_judgment_status"], "bounded_support")
        self.assertEqual(status["pre_closeout_readiness_status"], "ready")
        self.assertEqual(status["next_named_blocker"], "")

    def test_not_yet_supported_when_gate10c_slice_is_not_preserved(self) -> None:
        source_runs = [
            make_source_run("gate10b"),
            make_source_run(
                "gate10c",
                comparison_outcome_status="denied",
                next_named_blocker="conflict_side_bridge_degrades",
            ),
            make_source_run("gate10d"),
        ]

        status = gate10e.build_status_payload(source_runs)

        self.assertEqual(status["gate10c_slice_settled_status"], "not_preserved")
        self.assertEqual(status["three_slice_pattern_status"], "not_supported")
        self.assertEqual(status["interim_broader_judgment_status"], "not_yet_supported")
        self.assertEqual(status["pre_closeout_readiness_status"], "not_ready")
        self.assertEqual(status["next_named_blocker"], "conflict_side_bridge_degrades")

    def test_not_yet_supported_when_overclaim_pressure_appears(self) -> None:
        source_runs = [
            make_source_run("gate10b"),
            make_source_run("gate10c"),
            make_source_run(
                "gate10d",
                broader_non_promotion_status="violated",
            ),
        ]

        status = gate10e.build_status_payload(source_runs)

        self.assertEqual(
            status["broader_trusted_tree_settlement_status"],
            "pressure_to_overclaim",
        )
        self.assertEqual(status["interim_broader_judgment_status"], "not_yet_supported")
        self.assertEqual(
            status["next_named_blocker"],
            "broader_trusted_tree_settlement_overclaim_pressure",
        )

    def test_deferred_when_controlling_source_is_incomplete(self) -> None:
        source_runs = [
            make_source_run("gate10b"),
            {**make_source_run("gate10c"), "source_run_id": ""},
            make_source_run("gate10d"),
        ]

        status = gate10e.build_status_payload(source_runs)

        self.assertEqual(status["three_slice_pattern_status"], "deferred")
        self.assertEqual(status["interim_broader_judgment_status"], "deferred")
        self.assertEqual(status["pre_closeout_readiness_status"], "not_ready")
        self.assertEqual(status["next_named_blocker"], "controlling_source_incomplete")

    def test_policy_compare_echoes_slice_rows(self) -> None:
        source_runs = [
            make_source_run("gate10b"),
            make_source_run("gate10c"),
            make_source_run("gate10d"),
        ]
        registry = gate10e.build_registry(source_runs)
        compare = gate10e.build_policy_compare(registry)

        self.assertEqual(len(compare), 3)
        self.assertEqual(compare[0]["slice_id"], "gate10b")
        self.assertEqual(compare[1]["slice_id"], "gate10c")
        self.assertEqual(compare[2]["slice_id"], "gate10d")


if __name__ == "__main__":
    unittest.main()