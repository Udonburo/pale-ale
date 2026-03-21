#!/usr/bin/env python3
"""Regression tests for Gate9B small-cycle holonomy study helpers."""

import unittest

import run_gate9b_small_cycle_holonomy_study as gate9b


def make_cycle_row(
    benchmark_sample_id: str,
    cycle_type: str,
    cell_id: str,
    cycle_outcome: str,
    holonomy_defect,
) -> dict:
    return {
        "cycle_id": f"cycle:{benchmark_sample_id}:{cycle_type}",
        "cycle_type": cycle_type,
        "execution_sample_id": 1,
        "benchmark_sample_id": benchmark_sample_id,
        "cell_id": cell_id,
        "world_id": "w0",
        "world_type": "genealogy",
        "answer_target_type": "consistent_answer",
        "rendering_family_id": "transcript_v1",
        "cycle_outcome": cycle_outcome,
        "holonomy_defect": holonomy_defect,
        "holonomy_trace": None,
        "edge_ids": [],
        "metadata": {},
    }


class RunGate9BSmallCycleHolonomyStudyTest(unittest.TestCase):
    def test_build_quietness_pair_rows_computes_pair_delta(self) -> None:
        focus_rows = [
            {
                **make_cycle_row(
                    "gate8_plan_00001",
                    "support_answer_terminal_token_cycle",
                    "clean_support",
                    "none",
                    0.2,
                ),
                "cell_bucket": "cleaner_cell",
                "quietness_pair_id": "quiet_pair_0",
                "is_conflict_intended": False,
                "is_surface_noise_only": False,
            },
            {
                **make_cycle_row(
                    "gate8_plan_00007",
                    "support_answer_terminal_token_cycle",
                    "surface_noisy_clean",
                    "none",
                    0.35,
                ),
                "cell_bucket": "cleaner_cell",
                "quietness_pair_id": "quiet_pair_0",
                "is_conflict_intended": False,
                "is_surface_noise_only": True,
            },
        ]
        quietness_pair_rows = [
            {
                "quietness_pair_id": "quiet_pair_0",
                "world_id": "w0",
                "world_type": "genealogy",
                "rendering_family_id": "transcript_v1",
                "clean_benchmark_sample_id": "gate8_plan_00001",
                "surface_noisy_benchmark_sample_id": "gate8_plan_00007",
            }
        ]

        pair_rows = gate9b.build_quietness_pair_rows(focus_rows, quietness_pair_rows)
        support_row = next(
            row for row in pair_rows if row["cycle_type"] == "support_answer_terminal_token_cycle"
        )
        self.assertEqual(support_row["pair_outcome"], "none")
        self.assertAlmostEqual(float(support_row["surface_noisy_minus_clean_defect"]), 0.15, places=10)
        self.assertAlmostEqual(float(support_row["abs_quietness_delta"]), 0.15, places=10)

    def test_evaluate_falsifiers_triggers_cleaner_cell_dominance(self) -> None:
        focus_rows = [
            {"cycle_type": "support_answer_terminal_token_cycle", "cell_id": "clean_support", "cycle_outcome": "none", "holonomy_defect": 0.8},
            {"cycle_type": "support_answer_terminal_token_cycle", "cell_id": "surface_noisy_clean", "cycle_outcome": "none", "holonomy_defect": 0.7},
            {"cycle_type": "support_answer_terminal_token_cycle", "cell_id": "direct_contradiction", "cycle_outcome": "none", "holonomy_defect": 0.1},
            {"cycle_type": "support_answer_terminal_token_cycle", "cell_id": "distributed_incompatibility", "cycle_outcome": "none", "holonomy_defect": 0.2},
        ]

        falsifier_rows = gate9b.evaluate_falsifiers(focus_rows)
        status_row = next(
            row
            for row in falsifier_rows
            if row["cycle_type"] == "support_answer_terminal_token_cycle"
            and row["falsifier_id"] == "cleaner_cell_dominance"
        )
        self.assertEqual(status_row["status"], "triggered")

    def test_evaluate_falsifiers_triggers_direct_contradiction_escape(self) -> None:
        focus_rows = [
            {"cycle_type": "support_answer_terminal_token_cycle", "cell_id": "clean_support", "cycle_outcome": "none", "holonomy_defect": 0.2},
            {"cycle_type": "support_answer_terminal_token_cycle", "cell_id": "surface_noisy_clean", "cycle_outcome": "none", "holonomy_defect": 0.3},
            {"cycle_type": "support_answer_terminal_token_cycle", "cell_id": "direct_contradiction", "cycle_outcome": "none", "holonomy_defect": 0.7},
            {"cycle_type": "support_answer_terminal_token_cycle", "cell_id": "distributed_incompatibility", "cycle_outcome": "none", "holonomy_defect": 0.1},
        ]

        falsifier_rows = gate9b.evaluate_falsifiers(focus_rows)
        status_row = next(
            row
            for row in falsifier_rows
            if row["cycle_type"] == "support_answer_terminal_token_cycle"
            and row["falsifier_id"] == "direct_contradiction_escape"
        )
        self.assertEqual(status_row["status"], "triggered")

    def test_evaluate_falsifiers_triggers_missing_anchor_collapse(self) -> None:
        focus_rows = [
            {"cycle_type": "conflict_answer_terminal_token_cycle", "cell_id": "direct_contradiction", "cycle_outcome": "none", "holonomy_defect": 0.2},
            {
                "cycle_type": "conflict_answer_terminal_token_cycle",
                "cell_id": "distributed_incompatibility",
                "cycle_outcome": "missing_conflict_anchor",
                "holonomy_defect": None,
            },
        ]

        falsifier_rows = gate9b.evaluate_falsifiers(focus_rows)
        status_row = next(
            row
            for row in falsifier_rows
            if row["cycle_type"] == "conflict_answer_terminal_token_cycle"
            and row["falsifier_id"] == "missing_anchor_collapse"
        )
        self.assertEqual(status_row["status"], "triggered")


if __name__ == "__main__":
    unittest.main()
