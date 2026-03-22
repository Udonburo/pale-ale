#!/usr/bin/env python3
"""Regression tests for Gate11A named operator-pressure admissibility helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate11a_named_operator_pressure_admissibility as gate11a


def make_gate10f_manifest(run_id: str = "gate10f_run") -> dict:
    return {"run_id": run_id, "code_git_commit": "abc123"}


def make_gate10f_status(
    closeout_judgment_outcome_status: str = "closeout_supported",
    closeout_sentence_support_status: str = "supported",
    broader_trusted_tree_settlement_status: str = "unearned",
    operator_admission_still_denied_status: str = "confirmed",
    retroactive_reinterpretation_forbidden_status: str = "confirmed",
) -> dict:
    return {
        "closeout_judgment_outcome_status": closeout_judgment_outcome_status,
        "closeout_sentence_support_status": closeout_sentence_support_status,
        "broader_trusted_tree_settlement_status": broader_trusted_tree_settlement_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "retroactive_reinterpretation_forbidden_status": retroactive_reinterpretation_forbidden_status,
    }


class RunGate11ANamedOperatorPressureAdmissibilityTest(unittest.TestCase):
    def test_absence_defaults_to_not_yet_admissible(self) -> None:
        status = gate11a.build_status_payload(
            make_gate10f_manifest(),
            make_gate10f_status(),
            "Gate10F stays bounded and names no operator-pressure case.",
        )

        self.assertEqual(status["gate10_closeout_preservation_status"], "preserved")
        self.assertEqual(status["bounded_closeout_support_preservation_status"], "preserved")
        self.assertEqual(status["named_operator_pressure_case_status"], "absent")
        self.assertEqual(status["admissible_pressure_class_status"], "none")
        self.assertEqual(status["bounded_line_insufficiency_evidence_status"], "absent")
        self.assertEqual(status["named_operator_pressure_admissibility_status"], "not_yet_admissible")
        self.assertEqual(status["next_named_blocker"], "no_named_operator_pressure_case")

    def test_explicit_named_pressure_case_can_be_admissible(self) -> None:
        report_text = "\n".join(
            [
                "named_operator_pressure_case = narrow_reopening_pressure_without_graph_wide_leap",
                "bounded_line_insufficiency_evidence_status = present",
            ]
        )
        status = gate11a.build_status_payload(
            make_gate10f_manifest(), make_gate10f_status(), report_text
        )

        self.assertEqual(status["named_operator_pressure_case_status"], "present")
        self.assertEqual(
            status["admissible_pressure_class_status"],
            "narrow_reopening_pressure_without_graph_wide_leap",
        )
        self.assertEqual(status["bounded_line_insufficiency_evidence_status"], "present")
        self.assertEqual(status["graph_wide_operator_leap_pressure_status"], "absent")
        self.assertEqual(status["named_operator_pressure_admissibility_status"], "admissible")
        self.assertEqual(status["next_named_blocker"], "")

    def test_graph_wide_leap_pressure_denies_case(self) -> None:
        report_text = "\n".join(
            [
                "named_operator_pressure_case = narrow_reopening_pressure_without_graph_wide_leap",
                "bounded_line_insufficiency_evidence_status = present",
                "graph_wide_operator_leap_pressure_status = present",
            ]
        )
        status = gate11a.build_status_payload(
            make_gate10f_manifest(), make_gate10f_status(), report_text
        )

        self.assertEqual(status["graph_wide_operator_leap_pressure_status"], "present")
        self.assertEqual(status["named_operator_pressure_admissibility_status"], "denied")
        self.assertEqual(status["next_named_blocker"], "graph_wide_operator_leap_pressure")

    def test_narrative_class_mention_without_marker_stays_absent(self) -> None:
        report_text = (
            "The narrative mentions tree_choice_instability as an example, "
            "but no explicit named case is declared here."
        )
        status = gate11a.build_status_payload(
            make_gate10f_manifest(), make_gate10f_status(), report_text
        )

        self.assertEqual(status["named_operator_pressure_case_status"], "absent")
        self.assertEqual(status["admissible_pressure_class_status"], "none")
        self.assertEqual(status["bounded_line_insufficiency_evidence_status"], "absent")
        self.assertEqual(status["named_operator_pressure_admissibility_status"], "not_yet_admissible")

    def test_incomplete_source_defers_judgment(self) -> None:
        status = gate11a.build_status_payload(
            make_gate10f_manifest(run_id=""),
            make_gate10f_status(),
            "",
        )

        self.assertEqual(status["named_operator_pressure_case_status"], "deferred")
        self.assertEqual(status["admissible_pressure_class_status"], "deferred")
        self.assertEqual(status["bounded_line_insufficiency_evidence_status"], "deferred")
        self.assertEqual(status["named_operator_pressure_admissibility_status"], "deferred")
        self.assertEqual(status["next_named_blocker"], "controlling_source_incomplete")

    def test_policy_compare_echoes_single_source_row(self) -> None:
        status = gate11a.build_status_payload(
            make_gate10f_manifest(),
            make_gate10f_status(),
            "Gate10F stays bounded and names no operator-pressure case.",
        )
        registry = gate11a.build_registry(make_gate10f_manifest(), make_gate10f_status(), status)
        compare = gate11a.build_policy_compare(registry)

        self.assertEqual(len(compare), 1)
        self.assertEqual(compare[0]["source_gate10f_run_id"], "gate10f_run")
        self.assertEqual(compare[0]["named_operator_pressure_case_status"], "absent")


if __name__ == "__main__":
    unittest.main()
