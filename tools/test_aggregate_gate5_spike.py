#!/usr/bin/env python3
"""Regression tests for Gate5 aggregate reporting helpers."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate_gate5_spike import build_cfa_report, build_genealogy_geometry_labels


class AggregateGate5SpikeTests(unittest.TestCase):
    def test_prefix_only_w3_is_empty_when_defect_starts_at_zero(self) -> None:
        self.assertEqual(
            build_genealogy_geometry_labels([1, 1, 0, 0], "prefix_only_w3"),
            [0, 0, 0, 0],
        )

    def test_dual_view_skips_empty_prefix_geometry_samples(self) -> None:
        manifest = {"run_id": "gate5_test", "method_id": "transport_loop_residual_experiment_v1"}
        token_rows = [
            {"sample_id": "1", "step": "0", "label_token": "0", "score_F_loop": "0.10", "rotor_loop_chordal_v1": "0.80"},
            {"sample_id": "1", "step": "1", "label_token": "0", "score_F_loop": "0.20", "rotor_loop_chordal_v1": "0.70"},
            {"sample_id": "1", "step": "2", "label_token": "1", "score_F_loop": "0.95", "rotor_loop_chordal_v1": "0.40"},
            {"sample_id": "1", "step": "3", "label_token": "1", "score_F_loop": "0.85", "rotor_loop_chordal_v1": "0.30"},
            {"sample_id": "2", "step": "0", "label_token": "1", "score_F_loop": "0.90", "rotor_loop_chordal_v1": "0.50"},
            {"sample_id": "2", "step": "1", "label_token": "1", "score_F_loop": "0.80", "rotor_loop_chordal_v1": "0.40"},
            {"sample_id": "2", "step": "2", "label_token": "0", "score_F_loop": "0.10", "rotor_loop_chordal_v1": "0.20"},
        ]
        sample_rows = [
            {
                "sample_id": "1",
                "variant": "frustrated",
                "world_type": "genealogy",
                "auprc_F": "0.90",
                "auprc_rotor_loop_chordal_v1": "0.30",
                "auprc_E": "0.40",
                "delta_auprc_rotor_loop_chordal_v1_vs_F": "-0.60",
                "hit_at_10_F": "2",
                "hit_at_10_rotor_loop_chordal_v1": "2",
            },
            {
                "sample_id": "2",
                "variant": "frustrated",
                "world_type": "genealogy",
                "auprc_F": "0.70",
                "auprc_rotor_loop_chordal_v1": "0.20",
                "auprc_E": "0.30",
                "delta_auprc_rotor_loop_chordal_v1_vs_F": "-0.50",
                "hit_at_10_F": "2",
                "hit_at_10_rotor_loop_chordal_v1": "2",
            },
        ]

        report = build_cfa_report(manifest, token_rows, sample_rows, topk=10)

        self.assertIn("| inside_span | canonical | 2 |", report)
        self.assertIn("| prefix_only_w3 | diagnostic-only | 1 |", report)

    def test_cfa_report_includes_genealogy_dual_view_section(self) -> None:
        manifest = {"run_id": "gate5_test", "method_id": "transport_loop_residual_experiment_v1"}
        token_rows = [
            {"sample_id": "1", "step": "0", "label_token": "0", "score_F_loop": "0.10", "rotor_loop_chordal_v1": "0.80"},
            {"sample_id": "1", "step": "1", "label_token": "0", "score_F_loop": "0.20", "rotor_loop_chordal_v1": "0.70"},
            {"sample_id": "1", "step": "2", "label_token": "1", "score_F_loop": "0.95", "rotor_loop_chordal_v1": "0.40"},
            {"sample_id": "1", "step": "3", "label_token": "1", "score_F_loop": "0.85", "rotor_loop_chordal_v1": "0.30"},
            {"sample_id": "1", "step": "4", "label_token": "0", "score_F_loop": "0.05", "rotor_loop_chordal_v1": "0.10"},
        ]
        sample_rows = [
            {
                "sample_id": "1",
                "variant": "frustrated",
                "world_type": "genealogy",
                "auprc_F": "0.90",
                "auprc_rotor_loop_chordal_v1": "0.30",
                "auprc_E": "0.40",
                "delta_auprc_rotor_loop_chordal_v1_vs_F": "-0.60",
                "hit_at_10_F": "2",
                "hit_at_10_rotor_loop_chordal_v1": "2",
            }
        ]

        report = build_cfa_report(manifest, token_rows, sample_rows, topk=10)

        self.assertIn("## Genealogy Dual-View", report)
        self.assertIn("| inside_span | canonical | 1 |", report)
        self.assertIn("| prefix_only_w3 | diagnostic-only | 1 |", report)
        self.assertIn("Headline genealogy reporting remains canonical `inside_span`.", report)


if __name__ == "__main__":
    unittest.main()
