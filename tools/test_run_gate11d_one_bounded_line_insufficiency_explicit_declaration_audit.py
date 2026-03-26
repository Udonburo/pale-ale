#!/usr/bin/env python3
"""Regression tests for Gate11D one bounded-line insufficiency explicit-declaration helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate11d_one_bounded_line_insufficiency_explicit_declaration_audit as gate11d


def make_gate11c_manifest(run_id: str = "gate11c_run") -> dict:
    return {"run_id": run_id, "code_git_commit": "abc123"}


def make_gate11c_status(
    gate10_closeout_preservation_status: str = "preserved",
    gate11a_absence_result_preservation_status: str = "preserved",
    broader_trusted_tree_settlement_still_unearned_status: str = "confirmed",
    operator_admission_still_denied_status: str = "confirmed",
    retroactive_reinterpretation_forbidden_status: str = "confirmed",
    explicit_marker_shape_status: str = "defined",
    single_candidate_singularity_status: str = "defined",
    bounded_line_insufficiency_evidence_shape_status: str = "defined",
    anti_inflation_boundary_status: str = "defined",
    bounded_line_insufficiency_declaration_surface_status: str = "surface_defined",
) -> dict:
    return {
        "gate10_closeout_preservation_status": gate10_closeout_preservation_status,
        "gate11a_absence_result_preservation_status": gate11a_absence_result_preservation_status,
        "broader_trusted_tree_settlement_still_unearned_status": broader_trusted_tree_settlement_still_unearned_status,
        "operator_admission_still_denied_status": operator_admission_still_denied_status,
        "retroactive_reinterpretation_forbidden_status": retroactive_reinterpretation_forbidden_status,
        "explicit_marker_shape_status": explicit_marker_shape_status,
        "single_candidate_singularity_status": single_candidate_singularity_status,
        "bounded_line_insufficiency_evidence_shape_status": bounded_line_insufficiency_evidence_shape_status,
        "anti_inflation_boundary_status": anti_inflation_boundary_status,
        "bounded_line_insufficiency_declaration_surface_status": bounded_line_insufficiency_declaration_surface_status,
    }


class RunGate11DOneBoundedLineInsufficiencyExplicitDeclarationAuditTest(unittest.TestCase):
    def test_not_yet_declared_when_surface_exists_but_no_declaration_markers(self) -> None:
        status = gate11d.build_status_payload(
            make_gate11c_manifest(),
            make_gate11c_status(),
            "Gate11C fixes the declaration surface but does not declare a candidate.",
            [{"source_gate11b_run_id": "gate11b_run"}],
        )

        self.assertEqual(status["gate11c_declaration_surface_preservation_status"], "preserved")
        self.assertEqual(status["bounded_line_insufficiency_explicit_declaration_marker_status"], "absent")
        self.assertEqual(status["bounded_line_insufficiency_candidate_id_singularity_status"], "absent")
        self.assertEqual(status["bounded_line_insufficiency_class_singularity_status"], "none")
        self.assertEqual(status["bounded_line_host_failure_statement_status"], "absent")
        self.assertEqual(status["one_bounded_line_insufficiency_explicit_declaration_status"], "not_yet_declared")
        self.assertEqual(status["next_named_blocker"], "no_explicit_declaration_marker")

    def test_declared_when_one_full_explicit_declaration_is_present(self) -> None:
        report_text = "\n".join(
            [
                "bounded_line_insufficiency_candidate_declaration_status = declared",
                "bounded_line_insufficiency_candidate_id = candidate_alpha",
                "bounded_line_insufficiency_class_status = current_bounded_line_insufficiency",
                "bounded_line_host_failure_status = explicit",
                "- one bounded-line insufficiency candidate is explicitly declared: candidate_alpha",
                "- the current bounded line cannot honestly host candidate_alpha",
            ]
        )
        registry = [
            {
                "bounded_line_insufficiency_candidate_id": "candidate_alpha",
                "bounded_line_insufficiency_class_status": "current_bounded_line_insufficiency",
            }
        ]
        status = gate11d.build_status_payload(
            make_gate11c_manifest(), make_gate11c_status(), report_text, registry
        )

        self.assertEqual(status["bounded_line_insufficiency_explicit_declaration_marker_status"], "present")
        self.assertEqual(status["bounded_line_insufficiency_candidate_id_singularity_status"], "single")
        self.assertEqual(status["bounded_line_insufficiency_class_singularity_status"], "single")
        self.assertEqual(status["bounded_line_host_failure_statement_status"], "explicit")
        self.assertEqual(status["one_bounded_line_insufficiency_explicit_declaration_status"], "declared")
        self.assertEqual(status["next_named_blocker"], "")

    def test_multiple_candidate_ids_defer(self) -> None:
        report_text = "\n".join(
            [
                "bounded_line_insufficiency_candidate_declaration_status = declared",
                "bounded_line_insufficiency_candidate_id = candidate_alpha",
                "- one bounded-line insufficiency candidate is explicitly declared: candidate_beta",
                "bounded_line_host_failure_status = explicit",
                "- the current bounded line cannot honestly host candidate_alpha",
            ]
        )
        registry = [{"bounded_line_insufficiency_candidate_id": "candidate_alpha"}]
        status = gate11d.build_status_payload(
            make_gate11c_manifest(), make_gate11c_status(), report_text, registry
        )

        self.assertEqual(status["bounded_line_insufficiency_candidate_id_singularity_status"], "multiple")
        self.assertEqual(status["one_bounded_line_insufficiency_explicit_declaration_status"], "deferred")
        self.assertEqual(status["next_named_blocker"], "multiple_candidate_ids")

    def test_multiple_classes_defer(self) -> None:
        report_text = "\n".join(
            [
                "bounded_line_insufficiency_candidate_declaration_status = declared",
                "bounded_line_insufficiency_candidate_id = candidate_alpha",
                "bounded_line_insufficiency_class_status = current_bounded_line_insufficiency",
                "bounded_line_host_failure_status = explicit",
                "- one bounded-line insufficiency candidate is explicitly declared: candidate_alpha",
                "- the current bounded line cannot honestly host candidate_alpha",
            ]
        )
        registry = [
            {
                "bounded_line_insufficiency_candidate_id": "candidate_alpha",
                "bounded_line_insufficiency_class_status": "tree_choice_instability",
            }
        ]
        status = gate11d.build_status_payload(
            make_gate11c_manifest(), make_gate11c_status(), report_text, registry
        )

        self.assertEqual(status["bounded_line_insufficiency_class_singularity_status"], "multiple")
        self.assertEqual(status["one_bounded_line_insufficiency_explicit_declaration_status"], "deferred")
        self.assertEqual(status["next_named_blocker"], "multiple_classes")

    def test_missing_host_failure_statement_keeps_not_yet_declared(self) -> None:
        report_text = "\n".join(
            [
                "bounded_line_insufficiency_candidate_declaration_status = declared",
                "bounded_line_insufficiency_candidate_id = candidate_alpha",
                "bounded_line_insufficiency_class_status = current_bounded_line_insufficiency",
                "- one bounded-line insufficiency candidate is explicitly declared: candidate_alpha",
            ]
        )
        registry = [
            {
                "bounded_line_insufficiency_candidate_id": "candidate_alpha",
                "bounded_line_insufficiency_class_status": "current_bounded_line_insufficiency",
            }
        ]
        status = gate11d.build_status_payload(
            make_gate11c_manifest(), make_gate11c_status(), report_text, registry
        )

        self.assertEqual(status["bounded_line_host_failure_statement_status"], "absent")
        self.assertEqual(status["one_bounded_line_insufficiency_explicit_declaration_status"], "not_yet_declared")
        self.assertEqual(status["next_named_blocker"], "no_explicit_host_failure_statement")

    def test_boundary_break_denies(self) -> None:
        status = gate11d.build_status_payload(
            make_gate11c_manifest(),
            make_gate11c_status(anti_inflation_boundary_status="denied"),
            "Gate11C source is present.",
            [],
        )

        self.assertEqual(status["anti_inflation_boundary_status"], "not_confirmed")
        self.assertEqual(status["one_bounded_line_insufficiency_explicit_declaration_status"], "denied")
        self.assertEqual(status["next_named_blocker"], "anti_inflation_boundary_not_confirmed")


if __name__ == "__main__":
    unittest.main()