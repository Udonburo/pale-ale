#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("gate12c2_synthetic_lab.py")
SPEC = importlib.util.spec_from_file_location("gate12c2_synthetic_lab", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not import {MODULE_PATH}")
gate12c2 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate12c2
SPEC.loader.exec_module(gate12c2)


class Gate12C2BatchedKernelTest(unittest.TestCase):
    def assert_diagnostics_equivalent(
        self,
        reference: gate12c2.ResidualDiagnostics,
        actual: gate12c2.ResidualDiagnostics,
        *,
        atol: float = 2.0e-12,
    ) -> None:
        self.assertEqual(actual.q, reference.q)
        for name in (
            "eligibility_status",
            "numerical_status",
            "alignment_status",
            "propagation_left_status",
            "propagation_right_status",
        ):
            self.assertEqual(getattr(actual, name), getattr(reference, name))
        for name in (
            "defect",
            "tail_left",
            "tail_right",
            "propagated_left",
            "propagated_right",
            "alignment",
            "propagation_left",
            "propagation_right",
            "matrix_identity_error",
            "squared_identity_error",
            "relative_gap_left",
            "relative_gap_right",
        ):
            reference_value = getattr(reference, name)
            actual_value = getattr(actual, name)
            if reference_value is None:
                self.assertIsNone(actual_value, name)
            else:
                self.assertIsNotNone(actual_value, name)
                self.assertTrue(
                    np.isclose(
                        float(actual_value),
                        float(reference_value),
                        rtol=2.0e-12,
                        atol=atol,
                    ),
                    msg=(
                        f"{name}: actual={actual_value!r}, "
                        f"reference={reference_value!r}"
                    ),
                )
        np.testing.assert_allclose(
            actual.product_singular_values_left,
            reference.product_singular_values_left,
            rtol=2.0e-12,
            atol=atol,
        )
        np.testing.assert_allclose(
            actual.product_singular_values_right,
            reference.product_singular_values_right,
            rtol=2.0e-12,
            atol=atol,
        )

    def assert_graph_cohort_equivalent(
        self,
        graphs: tuple[gate12c2.SyntheticGraph, ...],
    ) -> None:
        for q in (1, 2):
            batched = gate12c2.batched_graph_residual_diagnostics(
                graphs,
                q=q,
            )
            self.assertEqual(len(batched), len(graphs))
            for index, graph in enumerate(graphs):
                reference = gate12c2.graph_residual_diagnostics(graph, q=q)
                self.assert_diagnostics_equivalent(
                    reference,
                    batched.row(index),
                )

    def test_matches_scalar_reference_across_development_regimes(self) -> None:
        s0 = gate12c2.generate_s0_cohort(
            replicate_count=12,
            master_seed="batched-s0-equivalence",
        )
        s1 = gate12c2.generate_s1_shared_node_coupling_cohort(
            replicate_count=12,
            master_seed="batched-s1-equivalence",
            effect_strength=0.25,
        )
        cohorts = (
            s0,
            s1,
            gate12c2.n1_role_constrained_reassignment(
                s0,
                reassignment_seed="batched-n1-s0-equivalence",
            ),
            gate12c2.n1_role_constrained_reassignment(
                s1,
                reassignment_seed="batched-n1-s1-equivalence",
            ),
            gate12c2.s2_graph_unconstrained_orientation_draw(
                s0,
                orientation_seed="batched-s2-equivalence",
                draw_index=3,
            ),
        )
        for cohort in cohorts:
            with self.subTest(generator_id=cohort[0].generator_id):
                self.assert_graph_cohort_equivalent(cohort)

    def test_preserves_unstable_degenerate_and_full_rank_statuses(self) -> None:
        m0 = np.stack(
            (
                np.eye(2, dtype=np.float64),
                np.zeros((2, 2), dtype=np.float64),
                np.diag([1.0, 0.2]),
            )
        )
        m1 = np.stack(
            (
                np.eye(2, dtype=np.float64),
                np.eye(2, dtype=np.float64),
                np.eye(2, dtype=np.float64),
            )
        )
        m2 = np.stack(
            (
                np.eye(2, dtype=np.float64),
                np.zeros((2, 2), dtype=np.float64),
                np.diag([0.2, 1.0]),
            )
        )
        for q in (1, 2, 3):
            batched = gate12c2.batched_residual_diagnostics(
                m0,
                m1,
                m2,
                q=q,
            )
            for index in range(len(m0)):
                reference = gate12c2.residual_diagnostics(
                    m0[index],
                    m1[index],
                    m2[index],
                    q=q,
                )
                self.assert_diagnostics_equivalent(
                    reference,
                    batched.row(index),
                )

    def test_rejects_malformed_or_nonfinite_batches(self) -> None:
        identity = np.eye(3, dtype=np.float64)
        with self.assertRaises(gate12c2.Gate12C2DevelopmentError):
            gate12c2.batched_residual_diagnostics(
                identity,
                identity,
                identity,
                q=1,
            )
        nonfinite = np.stack((identity, identity)).copy()
        nonfinite[1, 0, 0] = np.nan
        finite = np.stack((identity, identity))
        with self.assertRaises(gate12c2.Gate12C2DevelopmentError):
            gate12c2.batched_residual_diagnostics(
                nonfinite,
                finite,
                finite,
                q=1,
            )

    def test_outer_pipeline_decision_matches_object_reference(self) -> None:
        common = {
            "regime_id": "S1_known_reverse_shared_node_coupling",
            "master_seed": "batched-outer-equivalence",
            "outer_experiment_index": 2,
            "block_count": 4,
            "inner_valid_draw_count": 2,
            "effect_strength": 0.25,
        }
        reference = gate12c2.run_development_outer_experiment(
            **common,
            diagnostic_kernel=(
                gate12c2.OBJECT_REFERENCE_DIAGNOSTIC_KERNEL
            ),
        )
        batched = gate12c2.run_development_outer_experiment(
            **common,
            diagnostic_kernel=gate12c2.BATCHED_DIAGNOSTIC_KERNEL,
        )
        for name in (
            "grid_outcome",
            "claim_promotion",
            "partial_or_structured_is_promotional",
        ):
            self.assertEqual(
                batched["pipeline_decision"][name],
                reference["pipeline_decision"][name],
            )
        for actual, expected in zip(
            batched["pipeline_decision"]["endpoint_rows"],
            reference["pipeline_decision"]["endpoint_rows"],
            strict=True,
        ):
            for name in (
                "endpoint_id",
                "q_directional_support",
                "run_support",
                "q_discordant_run",
                "coverage_complete",
                "informative",
                "directional_raw_p",
                "holm_adjusted_directional_p",
            ):
                self.assertEqual(actual[name], expected[name])
            self.assertTrue(
                np.isclose(
                    actual["median_log_ratio"],
                    expected["median_log_ratio"],
                    rtol=2.0e-12,
                    atol=2.0e-12,
                )
            )
        self.assertEqual(
            len(batched["endpoint_receipts"]),
            len(reference["endpoint_receipts"]),
        )
        for actual_endpoint, reference_endpoint in zip(
            batched["endpoint_receipts"],
            reference["endpoint_receipts"],
            strict=True,
        ):
            self.assertEqual(
                actual_endpoint["endpoint_id"],
                reference_endpoint["endpoint_id"],
            )
            self.assertEqual(
                actual_endpoint["sign_test"],
                reference_endpoint["sign_test"],
            )
            for actual_block, reference_block in zip(
                actual_endpoint["block_rows"],
                reference_endpoint["block_rows"],
                strict=True,
            ):
                self.assertTrue(
                    np.isclose(
                        actual_block["block_log_observed_to_N1_defect"],
                        reference_block[
                            "block_log_observed_to_N1_defect"
                        ],
                        rtol=2.0e-12,
                        atol=2.0e-12,
                    )
                )
                self.assertTrue(
                    np.isclose(
                        actual_block["null_defect_median"],
                        reference_block["null_defect_median"],
                        rtol=2.0e-12,
                        atol=2.0e-12,
                    )
                )

    def test_s2_identification_matches_object_reference(self) -> None:
        common = {
            "master_seed": "batched-s2-outer-equivalence",
            "outer_experiment_index": 1,
            "block_count": 4,
            "inner_valid_draw_count": 1,
        }
        reference = gate12c2.run_development_s2_identification_experiment(
            **common,
            diagnostic_kernel=(
                gate12c2.OBJECT_REFERENCE_DIAGNOSTIC_KERNEL
            ),
        )
        batched = gate12c2.run_development_s2_identification_experiment(
            **common,
            diagnostic_kernel=gate12c2.BATCHED_DIAGNOSTIC_KERNEL,
        )
        for name in (
            "identified_case_count",
            "breadth_pass",
            "identification_success",
        ):
            self.assertEqual(batched[name], reference[name])
        for actual, expected in zip(
            batched["endpoint_rows"],
            reference["endpoint_rows"],
            strict=True,
        ):
            self.assertEqual(actual["endpoint_id"], expected["endpoint_id"])
            self.assertEqual(
                actual["endpoint_identified"],
                expected["endpoint_identified"],
            )
            self.assertEqual(
                actual["inflation_consistent_channels"],
                expected["inflation_consistent_channels"],
            )
            for arm in (
                "observed",
                "N1",
                "graph_unconstrained_stressor",
            ):
                for field_name, expected_value in expected[
                    "component_medians"
                ][arm].items():
                    actual_value = actual["component_medians"][arm][field_name]
                    if expected_value is None:
                        self.assertIsNone(actual_value)
                    else:
                        self.assertTrue(
                            np.isclose(
                                actual_value,
                                expected_value,
                                rtol=2.0e-12,
                                atol=2.0e-12,
                            )
                        )


if __name__ == "__main__":
    unittest.main()
