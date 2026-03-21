#!/usr/bin/env python3
"""Regression tests for Gate9C missingness-topology audit helpers."""

import unittest

import run_gate9c_missingness_topology_audit as gate9c


def make_focus_row(
    cycle_type: str,
    cell_id: str,
    answer_target_type: str,
    cycle_outcome: str,
    *,
    is_conflict_intended: bool,
) -> dict:
    return {
        "cycle_id": f"cycle:{cell_id}:{answer_target_type}:{cycle_type}",
        "cycle_type": cycle_type,
        "execution_sample_id": 1,
        "benchmark_sample_id": f"{cell_id}:{answer_target_type}",
        "cell_id": cell_id,
        "cell_bucket": "conflict_cell" if is_conflict_intended else "cleaner_cell",
        "world_id": "w0",
        "world_type": "genealogy",
        "answer_target_type": answer_target_type,
        "rendering_family_id": "transcript_v1",
        "cycle_outcome": cycle_outcome,
        "holonomy_defect": None,
        "holonomy_trace": None,
        "quietness_pair_id": "",
        "is_conflict_intended": is_conflict_intended,
        "is_surface_noise_only": not is_conflict_intended,
    }


class RunGate9CMissingnessTopologyAuditTest(unittest.TestCase):
    def test_classify_structural_conflict_absence_on_clean_cell(self) -> None:
        rows = [
            make_focus_row(
                "conflict_answer_terminal_token_cycle",
                "clean_support",
                "consistent_answer",
                "missing_conflict_anchor",
                is_conflict_intended=False,
            )
        ]
        missingness_rows = gate9c.build_missingness_rows(rows)
        self.assertEqual(missingness_rows[0]["absence_class"], gate9c.STRUCTURAL)

    def test_classify_taxonomic_when_answer_targets_differ(self) -> None:
        rows = [
            make_focus_row(
                "support_answer_terminal_token_cycle",
                "direct_contradiction",
                "consistent_answer",
                "none",
                is_conflict_intended=True,
            ),
            make_focus_row(
                "support_answer_terminal_token_cycle",
                "direct_contradiction",
                "conflict_following_wrong_answer",
                "missing_support_anchor",
                is_conflict_intended=True,
            ),
        ]
        missingness_rows = gate9c.build_missingness_rows(rows)
        self.assertEqual(missingness_rows[0]["absence_class"], gate9c.TAXONOMIC)

    def test_classify_bundle_specific_when_allowed_but_uninstantiated(self) -> None:
        rows = [
            make_focus_row(
                "conflict_answer_terminal_token_cycle",
                "distributed_incompatibility",
                "consistent_answer",
                "missing_conflict_anchor",
                is_conflict_intended=True,
            )
        ]
        missingness_rows = gate9c.build_missingness_rows(rows)
        self.assertEqual(missingness_rows[0]["absence_class"], gate9c.BUNDLE_SPECIFIC)

    def test_classify_implementation_bound_for_missing_cycle_edge(self) -> None:
        rows = [
            make_focus_row(
                "support_answer_terminal_token_cycle",
                "distributed_incompatibility",
                "consistent_answer",
                "missing_cycle_edge",
                is_conflict_intended=True,
            )
        ]
        missingness_rows = gate9c.build_missingness_rows(rows)
        self.assertEqual(missingness_rows[0]["absence_class"], gate9c.IMPLEMENTATION_BOUND)

    def test_build_admission_slice_denies_unusable_conflict_rows(self) -> None:
        coverage_by_cell_rows = [
            {
                "cell_id": "distributed_incompatibility",
                "cycle_type": "conflict_answer_terminal_token_cycle",
                "coverage_rate": 0.0,
                "usable_status": "not_yet_usable",
            },
            {
                "cell_id": "direct_contradiction",
                "cycle_type": "conflict_answer_terminal_token_cycle",
                "coverage_rate": 1.0,
                "usable_status": "usable",
            },
        ]
        missingness_rows = [
            {"absence_class": gate9c.BUNDLE_SPECIFIC},
            {"absence_class": gate9c.STRUCTURAL},
        ]
        admission_slice = gate9c.build_admission_slice(coverage_by_cell_rows, missingness_rows)
        self.assertEqual(admission_slice["usable_motif_coverage_status"], "denied")
        self.assertEqual(admission_slice["missingness_topology_accounted_status"], "clear")


if __name__ == "__main__":
    unittest.main()
