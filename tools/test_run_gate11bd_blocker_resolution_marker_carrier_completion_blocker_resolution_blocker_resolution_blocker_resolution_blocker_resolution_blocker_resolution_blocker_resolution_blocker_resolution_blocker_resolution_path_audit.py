#!/usr/bin/env python3
"""Regression tests for Gate11BD blocker-resolution marker carrier-completion blocker-resolution blocker-resolution blocker-resolution blocker-resolution blocker-resolution blocker-resolution blocker_resolution blocker-resolution path helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate11bd_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_path_audit as gate11bd


def make_gate11bc_manifest(run_id: str = "gate11bc_run") -> dict:
    return {"run_id": run_id, "code_git_commit": "c4cf10e5823b815022e0833b93cdbcb1eb95082e"}


def make_gate11bc_status(**overrides: str) -> dict:
    status = {
        "gate10_closeout_preservation_status": "preserved",
        "gate11a_absence_result_preservation_status": "preserved",
        "gate11c_declaration_surface_preservation_status": "preserved",
        "gate11d_not_yet_declared_state_preservation_status": "preserved",
        "gate11e_path_defined_state_preservation_status": "preserved",
        "gate11f_not_yet_admissible_state_preservation_status": "preserved",
        "gate11g_naming_surface_preservation_status": "preserved",
        "gate11h_not_yet_named_state_preservation_status": "preserved",
        "gate11i_path_defined_state_preservation_status": "preserved",
        "gate11j_not_yet_admissible_state_preservation_status": "preserved",
        "gate11k_not_yet_present_state_preservation_status": "preserved",
        "gate11l_path_defined_state_preservation_status": "preserved",
        "gate11m_not_yet_present_state_preservation_status": "preserved",
        "gate11n_residual_named_state_preservation_status": "preserved",
        "gate11o_path_defined_state_preservation_status": "preserved",
        "gate11p_not_yet_completed_state_preservation_status": "preserved",
        "gate11q_surface_defined_state_preservation_status": "preserved",
        "gate11r_not_yet_present_state_preservation_status": "preserved",
        "gate11s_path_defined_state_preservation_status": "preserved",
        "gate11t_not_yet_present_state_preservation_status": "preserved",
        "gate11u_residual_named_state_preservation_status": "preserved",
        "gate11v_path_defined_state_preservation_status": "preserved",
        "gate11w_not_yet_completed_state_preservation_status": "preserved",
        "gate11x_blocker_named_state_preservation_status": "preserved",
        "gate11y_path_defined_state_preservation_status": "preserved",
        "gate11z_not_yet_resolved_state_preservation_status": "preserved",
        "gate11aa_surface_defined_state_preservation_status": "preserved",
        "gate11ab_not_yet_present_state_preservation_status": "preserved",
        "gate11ac_path_defined_state_preservation_status": "preserved",
        "gate11ad_not_yet_present_state_preservation_status": "preserved",
        "gate11ae_residual_named_state_preservation_status": "preserved",
        "gate11af_path_defined_state_preservation_status": "preserved",
        "gate11ag_not_yet_completed_state_preservation_status": "preserved",
        "gate11ah_blocker_named_state_preservation_status": "preserved",
        "gate11ai_path_defined_state_preservation_status": "preserved",
        "gate11aj_not_yet_resolved_state_preservation_status": "preserved",
        "gate11ak_blocker_named_state_preservation_status": "preserved",
        "gate11al_path_defined_state_preservation_status": "preserved",
        "gate11am_not_yet_resolved_state_preservation_status": "preserved",
        "gate11an_blocker_named_state_preservation_status": "preserved",
        "gate11ao_path_defined_state_preservation_status": "preserved",
        "gate11ap_not_yet_resolved_state_preservation_status": "preserved",
        "gate11aq_blocker_named_state_preservation_status": "preserved",
        "gate11ar_path_defined_state_preservation_status": "preserved",
        "gate11as_not_yet_resolved_state_preservation_status": "preserved",
        "gate11at_blocker_named_state_preservation_status": "preserved",
        "gate11au_path_defined_state_preservation_status": "preserved",
        "gate11av_not_yet_resolved_state_preservation_status": "preserved",
        "gate11aw_blocker_named_state_preservation_status": "preserved",
        "gate11ax_path_defined_state_preservation_status": "preserved",
        "gate11ay_not_yet_resolved_state_preservation_status": "preserved",
        "gate11az_blocker_named_state_preservation_status": "preserved",
        "gate11ba_path_defined_state_preservation_status": "preserved",
        "gate11bb_not_yet_resolved_state_preservation_status": "preserved",
        "broader_trusted_tree_settlement_still_unearned_status": "confirmed",
        "operator_admission_still_denied_status": "confirmed",
        "retroactive_reinterpretation_forbidden_status": "confirmed",
        "named_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_condition_preservation_status": "preserved",
        "explicit_blocker_resolution_marker_blocker_status": "named",
        "same_source_blocker_resolution_blocker_status": "named",
        "blocker_resolution_blocker_boundary_status": "confirmed",
        "blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_status": "blocker_named",
        "next_named_blocker": "no_explicit_blocker_resolution_marker",
    }
    status.update(overrides)
    return status


class RunGate11BDBlockerResolutionMarkerCarrierCompletionBlockerResolutionBlockerResolutionBlockerResolutionBlockerResolutionBlockerResolutionBlockerResolutionBlockerResolutionBlockerResolutionPathAuditTest(
    unittest.TestCase
):
    def test_path_defined_for_current_frozen_gate11bc_source(self) -> None:
        status = gate11bd.build_status_payload(
            make_gate11bc_manifest(),
            make_gate11bc_status(),
            "Gate11BC preserves the blocker-named line under the bounded source.",
        )

        self.assertEqual(status["gate11bc_blocker_named_state_preservation_status"], "preserved")
        self.assertEqual(
            status[
                "named_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_condition_preservation_status"
            ],
            "preserved",
        )
        self.assertEqual(status["minimum_same_source_blocker_resolution_rule_status"], "defined")
        self.assertEqual(status["bounded_read_prefix_resolution_requirement_status"], "defined")
        self.assertEqual(
            status[
                "blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_path_status"
            ],
            "path_defined",
        )
        self.assertEqual(status["next_named_blocker"], "")

    def test_not_yet_defined_when_named_blocker_is_not_preserved(self) -> None:
        status = gate11bd.build_status_payload(
            make_gate11bc_manifest(),
            make_gate11bc_status(
                named_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_condition_preservation_status="not_preserved"
            ),
            "Gate11BC no longer preserves the named blocker-resolution marker carrier-completion blocker-resolution blocker-resolution blocker-resolution blocker-resolution blocker-resolution blocker-resolution condition.",
        )

        self.assertEqual(
            status[
                "named_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_condition_preservation_status"
            ],
            "not_preserved",
        )
        self.assertEqual(status["minimum_same_source_blocker_resolution_rule_status"], "not_yet_defined")
        self.assertEqual(
            status[
                "blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_path_status"
            ],
            "not_yet_defined",
        )
        self.assertEqual(status["next_named_blocker"], "named_blocker_not_preserved")

    def test_denied_when_gate11bc_blocker_named_state_is_not_preserved(self) -> None:
        status = gate11bd.build_status_payload(
            make_gate11bc_manifest(),
            make_gate11bc_status(
                blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_status="not_yet_named"
            ),
            "Gate11BC no longer preserves the blocker-named state.",
        )

        self.assertEqual(status["gate11bc_blocker_named_state_preservation_status"], "not_preserved")
        self.assertEqual(
            status[
                "blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_path_status"
            ],
            "denied",
        )
        self.assertEqual(status["next_named_blocker"], "gate11bc_blocker_named_state_not_preserved")

    def test_denied_when_boundary_breaks(self) -> None:
        status = gate11bd.build_status_payload(
            make_gate11bc_manifest(),
            make_gate11bc_status(blocker_resolution_blocker_boundary_status="denied"),
            "Gate11BC breaks the blocker-resolution boundary.",
        )

        self.assertEqual(status["blocker_resolution_boundary_status"], "denied")
        self.assertEqual(status["minimum_same_source_blocker_resolution_rule_status"], "denied")
        self.assertEqual(
            status[
                "blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_path_status"
            ],
            "denied",
        )
        self.assertEqual(status["next_named_blocker"], "blocker_resolution_boundary_not_intact")

    def test_deferred_when_source_is_incomplete(self) -> None:
        status = gate11bd.build_status_payload(make_gate11bc_manifest(run_id=""), make_gate11bc_status(), "")

        self.assertEqual(
            status[
                "named_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_condition_preservation_status"
            ],
            "deferred",
        )
        self.assertEqual(status["minimum_same_source_blocker_resolution_rule_status"], "deferred")
        self.assertEqual(status["bounded_read_prefix_resolution_requirement_status"], "deferred")
        self.assertEqual(
            status[
                "blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_path_status"
            ],
            "deferred",
        )
        self.assertEqual(status["next_named_blocker"], "controlling_source_incomplete")

    def test_not_yet_defined_when_source_does_not_narrow_path_enough(self) -> None:
        status = gate11bd.build_status_payload(
            make_gate11bc_manifest(),
            make_gate11bc_status(next_named_blocker="same_source_blocker_resolution_not_completed"),
            "Gate11BC preserves a blocker, but this slice does not fix a broader path than the frozen line allows.",
        )

        self.assertEqual(
            status[
                "named_blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_condition_preservation_status"
            ],
            "not_preserved",
        )
        self.assertEqual(status["minimum_same_source_blocker_resolution_rule_status"], "not_yet_defined")
        self.assertEqual(status["bounded_read_prefix_resolution_requirement_status"], "not_yet_defined")
        self.assertEqual(
            status[
                "blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_path_status"
            ],
            "not_yet_defined",
        )
        self.assertEqual(status["next_named_blocker"], "named_blocker_not_preserved")

    def test_policy_compare_echoes_single_source_row(self) -> None:
        status = gate11bd.build_status_payload(
            make_gate11bc_manifest(),
            make_gate11bc_status(),
            "Gate11BC preserves the blocker-named line under the bounded source.",
        )
        registry = gate11bd.build_registry(make_gate11bc_manifest(), status)
        compare = gate11bd.build_policy_compare(registry)

        self.assertEqual(len(compare), 1)
        self.assertEqual(compare[0]["source_gate11bc_run_id"], "gate11bc_run")
        self.assertEqual(
            compare[0][
                "blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_path_status"
            ],
            "path_defined",
        )

    def test_prose_alone_does_not_count_as_blocker_resolution_path_definition(self) -> None:
        report_text = (
            "Example: if one explicit blocker-resolution marker, one explicit later_source_id or later_frozen_run_id, "
            "one blocker-resolution marker and only one blocker-resolution marker, and matched status, registry, and read surfaces "
            "were later present, the blocker could be resolved."
        )
        status = gate11bd.build_status_payload(
            make_gate11bc_manifest(),
            make_gate11bc_status(next_named_blocker="same_source_blocker_resolution_not_completed"),
            report_text,
        )

        self.assertEqual(
            status[
                "blocker_resolution_marker_carrier_completion_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_blocker_resolution_path_status"
            ],
            "not_yet_defined",
        )
        self.assertEqual(status["next_named_blocker"], "named_blocker_not_preserved")


if __name__ == "__main__":
    unittest.main()