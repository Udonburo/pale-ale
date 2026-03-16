#!/usr/bin/env python3
"""Regression tests for relation-first native local span builders."""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_native_local_span_gate4_input as boundary_builder


class RelationAffineLiftTests(unittest.TestCase):
    def test_sign_unstable_after_first_axis_reports_rank1(self) -> None:
        v = [0.283021, 0.347957, 0.883405, -0.135726]
        splus = [0.540883, 0.577111, 0.521064, -0.320751]
        sminus = [0.438859, -0.149925, -0.291559, 0.836611]

        result = boundary_builder.build_relation_affine_lift_coordinates(v, splus, sminus)

        self.assertEqual(result["boundary_outcome"], "sign_unstable")
        self.assertEqual(result["frame_rank"], 1)
        self.assertEqual(result["basis_sources"], ["d1"])
        self.assertEqual(result["sign_anchor_index_e1"], 0)
        self.assertEqual(result["sign_anchor_index_e2"], 0)

    def test_zero_area_parallel_case_reports_rank1(self) -> None:
        result = boundary_builder.build_relation_affine_lift_coordinates(
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
        )

        self.assertEqual(result["boundary_outcome"], "zero_area")
        self.assertEqual(result["frame_rank"], 1)
        self.assertEqual(result["basis_sources"], ["d1"])

    def test_lift_axis_collapse_branch_reports_partial_rank(self) -> None:
        with mock.patch.object(boundary_builder, "gram_schmidt_rank", return_value=2):
            result = boundary_builder.build_relation_affine_lift_coordinates(
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.5, 1.0, 0.0, 0.0],
            )

        self.assertEqual(result["boundary_outcome"], "lift_axis_collapse")
        self.assertEqual(result["frame_rank"], 2)
        self.assertEqual(result["relation_lift_rank"], 2)
        self.assertEqual(result["basis_sources"], ["d1", "d2_residual", "signed_angle_profile"])

    def test_relation_affine_v1_materializes_with_origin_span_e3(self) -> None:
        result = boundary_builder.build_relation_affine_lift_coordinates_v1(
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        )

        self.assertEqual(result["boundary_outcome"], "materialized_rank3")
        self.assertEqual(result["frame_rank"], 3)
        self.assertEqual(result["basis_sources"], ["d1", "d2_residual", "origin_span_e3"])
        self.assertEqual(result["sign_anchor_index_e3"], 1)
        self.assertAlmostEqual(result["raw_span_lift_center"], 0.5)
        self.assertAlmostEqual(result["coords_v"][2], -0.5)
        self.assertAlmostEqual(result["coords_splus"][2], -0.5)
        self.assertAlmostEqual(result["coords_sminus"][2], 0.5)

    def test_relation_affine_v1_raw_span_axis_collapse_reports_rank2(self) -> None:
        with mock.patch.object(
            boundary_builder,
            "build_origin_span_e3_axis",
            return_value=(None, 0, False, None, 0.0),
        ):
            result = boundary_builder.build_relation_affine_lift_coordinates_v1(
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            )

        self.assertEqual(result["boundary_outcome"], "raw_span_axis_collapse")
        self.assertEqual(result["frame_rank"], 2)
        self.assertEqual(result["basis_sources"], ["d1", "d2_residual"])

    def test_relation_affine_v2_materializes_with_modulation(self) -> None:
        result = boundary_builder.build_relation_affine_lift_coordinates_v2(
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        )

        self.assertEqual(result["boundary_outcome"], "materialized_rank3")
        self.assertEqual(
            result["basis_sources"],
            ["d1", "d2_residual", "signed_angle_profile_origin_span_modulation"],
        )
        self.assertTrue(result["raw_span_axis_available"])
        self.assertAlmostEqual(
            result["raw_span_modulation_alpha"], boundary_builder.RAW_SPAN_MODULATION_ALPHA_V2
        )
        self.assertAlmostEqual(result["coords_v"][2], -0.30618621784789724)
        self.assertAlmostEqual(result["coords_splus"][2], -0.30618621784789724)
        self.assertAlmostEqual(result["coords_sminus"][2], -0.9185586535436919)

    def test_relation_affine_v2_falls_back_to_v0_when_raw_axis_missing(self) -> None:
        with mock.patch.object(
            boundary_builder,
            "build_origin_span_e3_axis",
            return_value=(None, 0, False, None, 0.0),
        ):
            result = boundary_builder.build_relation_affine_lift_coordinates_v2(
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            )

        self.assertEqual(result["boundary_outcome"], "materialized_rank3")
        self.assertEqual(result["basis_sources"], ["d1", "d2_residual", "signed_angle_profile"])
        self.assertFalse(result["raw_span_axis_available"])
        self.assertAlmostEqual(result["raw_span_modulation_alpha"], 0.0)

    def test_raw_span_path_key_distinguishes_modulated_fallback_and_failure(self) -> None:
        self.assertEqual(
            boundary_builder.raw_span_path_key(
                {
                    "coordinate_rule_id": boundary_builder.COORDINATE_RULE_RELATION_AFFINE_LIFT_V2,
                    "boundary_outcome": "materialized_rank3",
                    "raw_span_axis_available": True,
                }
            ),
            "modulated",
        )
        self.assertEqual(
            boundary_builder.raw_span_path_key(
                {
                    "coordinate_rule_id": boundary_builder.COORDINATE_RULE_RELATION_AFFINE_LIFT_V2,
                    "boundary_outcome": "materialized_rank3",
                    "raw_span_axis_available": False,
                }
            ),
            "fallback_materialized",
        )
        self.assertEqual(
            boundary_builder.raw_span_path_key(
                {
                    "coordinate_rule_id": boundary_builder.COORDINATE_RULE_RELATION_AFFINE_LIFT_V2,
                    "boundary_outcome": "sign_unstable",
                    "raw_span_axis_available": False,
                }
            ),
            "axis_unavailable_nonmaterialized",
        )


if __name__ == "__main__":
    unittest.main()
