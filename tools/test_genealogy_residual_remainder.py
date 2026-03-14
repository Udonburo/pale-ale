#!/usr/bin/env python3
"""Minimal regression tests for genealogy residual remainder diagnostics."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_gate5_genealogy_residual_remainder import (
    build_geometry_labels,
    decide,
)


class GenealogyResidualRemainderTests(unittest.TestCase):
    def test_prefix_only_w3_is_empty_when_defect_starts_at_zero(self) -> None:
        self.assertEqual(
            build_geometry_labels([1, 1, 0, 0], "prefix_only_w3"),
            [0, 0, 0, 0],
        )

    def test_little_residual_requires_material_gain(self) -> None:
        inside_summary = {
            "mean_delta_rotor_vs_F": -0.01,
            "still_negative_rate": 0.40,
            "mean_before_to_inside_ratio_rotor": 1.30,
            "mean_inside_to_after_ratio_rotor": 0.45,
        }
        prefix_summary = {
            "mean_delta_rotor_vs_F": 0.001,
            "still_negative_rate": 0.0,
            "mean_before_to_inside_ratio_rotor": 0.80,
            "mean_inside_to_after_ratio_rotor": 0.30,
        }
        decision, _meta = decide(inside_summary, prefix_summary)
        self.assertNotEqual(decision, "little-residual-remainder")


if __name__ == "__main__":
    unittest.main()
