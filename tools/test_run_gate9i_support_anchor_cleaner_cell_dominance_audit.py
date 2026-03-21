#!/usr/bin/env python3
"""Regression tests for Gate9I support-anchor cleaner-cell dominance helpers."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_gate9i_support_anchor_cleaner_cell_dominance_audit as gate9i


def make_row(cell_id: str, gap: float, *, pair_id: str = "", status: str = "nontrivial_gap_candidate") -> dict:
    return {
        "cell_id": cell_id,
        "candidate_status": status,
        "coverage_gap_abs": gap,
        "benchmark_sample_id": f"{cell_id}:{gap}",
        "quietness_pair_id": pair_id,
        "anchor_kind": "support",
    }


class RunGate9ISupportAnchorCleanerCellDominanceAuditTest(unittest.TestCase):
    def test_surface_noisy_corroboration_and_distributed_underactivation(self) -> None:
        registry_rows = [
            make_row("clean_support", 0.14, pair_id="pair_a"),
            make_row("surface_noisy_clean", 0.15, pair_id="pair_a"),
            make_row("direct_contradiction", 0.13),
            make_row("distributed_incompatibility", 0.09),
        ]
        pair_rows = [
            {
                "abs_pair_gap_delta": 0.01,
            }
        ]
        payload = gate9i.build_status_payload(registry_rows, pair_rows)
        self.assertEqual(payload["support_anchor_cleaner_dominance_status"], "triggered")
        self.assertEqual(payload["surface_noisy_corroboration_status"], "corroborated")
        self.assertEqual(payload["distributed_underactivation_status"], "triggered")
        self.assertEqual(payload["dominance_explained_as_quietness_noise_status"], "denied")
        self.assertEqual(payload["next_named_subblocker"], "distributed_underactivation")

    def test_surface_noisy_not_corroborated_when_below_conflict_max(self) -> None:
        registry_rows = [
            make_row("clean_support", 0.14, pair_id="pair_a"),
            make_row("surface_noisy_clean", 0.10, pair_id="pair_a"),
            make_row("direct_contradiction", 0.13),
            make_row("distributed_incompatibility", 0.12),
        ]
        payload = gate9i.build_status_payload(registry_rows, [])
        self.assertEqual(payload["surface_noisy_corroboration_status"], "not_corroborated")
        self.assertEqual(payload["distributed_underactivation_status"], "triggered")

    def test_build_registry_rows_keeps_support_only(self) -> None:
        rows = gate9i.build_registry_rows(
            [
                {
                    "closure_id": "a",
                    "anchor_kind": "support",
                    "execution_sample_id": 1,
                    "benchmark_sample_id": "b1",
                    "cell_id": "clean_support",
                    "world_id": "w",
                    "world_type": "t",
                    "answer_target_type": "consistent_answer",
                    "rendering_family_id": "rf",
                    "candidate_status": "nontrivial_gap_candidate",
                    "coverage_gap_abs": 0.1,
                },
                {
                    "closure_id": "c",
                    "anchor_kind": "conflict",
                    "execution_sample_id": 2,
                    "benchmark_sample_id": "b2",
                    "cell_id": "direct_contradiction",
                    "world_id": "w",
                    "world_type": "t",
                    "answer_target_type": "consistent_answer",
                    "rendering_family_id": "rf",
                    "candidate_status": "nontrivial_gap_candidate",
                    "coverage_gap_abs": 0.1,
                },
            ],
            {"b1": {"quietness_pair_id": "pair_a", "is_surface_noise_only": False, "is_conflict_intended": False}},
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["quietness_pair_id"], "pair_a")


if __name__ == "__main__":
    unittest.main()
