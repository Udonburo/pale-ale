#!/usr/bin/env python3
"""Regression tests for Gate11B bounded-line insufficiency declarability helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate11b_bounded_line_insufficiency_declarability as gate11b


def make_gate11a_manifest(run_id: str = "gate11a_run") -> dict:
    return {"run_id": run_id, "code_git_commit": "abc123"}


def make_gate11a_status(
    gate10_closeout_preservation_status: str = "preserved",
    broader_trusted_tree_settlement_still_unearned_status: str = "confirmed",
    operator_admission_still_denied_status: str = "confirmed",
    retroactive_reinterpretation_forbidden_status: str = "confirmed",
    named_operator_pressure_case_status: str = "absent",
    admissible_pressure_class_status: str = "none",
    named_operator_pressure_admissibility_status: str = "not_yet_admissible",
    graph_wide_operator_leap_pressure_status: str = "absent",
) -> dict:
    return {
        "gate10_closeout_preservation_status": gate10_closeout_preservation_status,
        "broader_trusted_tree_settlement_still_unearned_status": broader_trusted_tree_settlement_still_unearned_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "retroactive_reinterpretation_forbidden_status": retroactive_reinterpretation_forbidden_status,
        "named_operator_pressure_case_status": named_operator_pressure_case_status,
        "admissible_pressure_class_status": admissible_pressure_class_status,
        "named_operator_pressure_admissibility_status": named_operator_pressure_admissibility_status,
        "graph_wide_operator_leap_pressure_status": graph_wide_operator_leap_pressure_status,
    }


class RunGate11BBoundedLineInsufficiencyDeclarabilityTest(unittest.TestCase):
    def test_absence_defaults_to_not_yet_declarable(self) -> None:
        status = gate11b.build_status_payload(
            make_gate11a_manifest(),
            make_gate11a_status(),
            "Gate11A preserves absence and names no bounded-line insufficiency declaration.",
        )

        self.assertEqual(status["gate10_closeout_preservation_status"], "preserved")
        self.assertEqual(status["gate11a_absence_result_preservation_status"], "preserved")
        self.assertEqual(status["bounded_line_insufficiency_candidate_status"], "absent")
        self.assertEqual(status["bounded_line_insufficiency_class_status"], "none")
        self.assertEqual(status["settlement_inflation_pressure_status"], "absent")
        self.assertEqual(status["bounded_line_insufficiency_declarability_status"], "not_yet_declarable")
        self.assertEqual(status["next_named_blocker"], "no_bounded_line_insufficiency_candidate")

    def test_explicit_single_candidate_can_be_declarable(self) -> None:
        report_text = "\n".join(
            [
                "bounded_line_insufficiency_candidate = current_bounded_line_insufficiency",
                "settlement_inflation_pressure_status = absent",
                "graph_wide_operator_leap_pressure_status = absent",
            ]
        )
        status = gate11b.build_status_payload(
            make_gate11a_manifest(), make_gate11a_status(), report_text
        )

        self.assertEqual(status["bounded_line_insufficiency_candidate_status"], "present")
        self.assertEqual(
            status["bounded_line_insufficiency_class_status"],
            "current_bounded_line_insufficiency",
        )
        self.assertEqual(status["bounded_line_insufficiency_declarability_status"], "declarable")
        self.assertEqual(status["next_named_blocker"], "")

    def test_multiple_candidates_defer_judgment(self) -> None:
        report_text = "\n".join(
            [
                "bounded_line_insufficiency_candidate = current_bounded_line_insufficiency",
                "bounded_line_insufficiency_candidate = nonlocal_reconciliation_pressure",
            ]
        )
        status = gate11b.build_status_payload(
            make_gate11a_manifest(), make_gate11a_status(), report_text
        )

        self.assertEqual(status["bounded_line_insufficiency_candidate_status"], "deferred")
        self.assertEqual(status["bounded_line_insufficiency_class_status"], "deferred")
        self.assertEqual(status["bounded_line_insufficiency_declarability_status"], "deferred")
        self.assertEqual(status["next_named_blocker"], "multiple_bounded_line_insufficiency_candidates")

    def test_settlement_inflation_pressure_denies_case(self) -> None:
        report_text = "\n".join(
            [
                "bounded_line_insufficiency_candidate = current_bounded_line_insufficiency",
                "settlement_inflation_pressure_status = present",
            ]
        )
        status = gate11b.build_status_payload(
            make_gate11a_manifest(), make_gate11a_status(), report_text
        )

        self.assertEqual(status["settlement_inflation_pressure_status"], "present")
        self.assertEqual(status["bounded_line_insufficiency_declarability_status"], "denied")
        self.assertEqual(status["next_named_blocker"], "settlement_inflation_pressure")

    def test_narrative_class_mention_without_marker_stays_absent(self) -> None:
        report_text = (
            "The narrative mentions current_bounded_line_insufficiency as a future class, "
            "but no explicit declaration is present here."
        )
        status = gate11b.build_status_payload(
            make_gate11a_manifest(), make_gate11a_status(), report_text
        )

        self.assertEqual(status["bounded_line_insufficiency_candidate_status"], "absent")
        self.assertEqual(status["bounded_line_insufficiency_class_status"], "none")
        self.assertEqual(status["bounded_line_insufficiency_declarability_status"], "not_yet_declarable")

    def test_policy_compare_echoes_single_source_row(self) -> None:
        status = gate11b.build_status_payload(
            make_gate11a_manifest(),
            make_gate11a_status(),
            "Gate11A preserves absence and names no bounded-line insufficiency declaration.",
        )
        registry = gate11b.build_registry(make_gate11a_manifest(), make_gate11a_status(), status)
        compare = gate11b.build_policy_compare(registry)

        self.assertEqual(len(compare), 1)
        self.assertEqual(compare[0]["source_gate11a_run_id"], "gate11a_run")
        self.assertEqual(compare[0]["bounded_line_insufficiency_candidate_status"], "absent")


if __name__ == "__main__":
    unittest.main()