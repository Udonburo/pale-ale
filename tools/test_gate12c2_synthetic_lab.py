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


class Gate12C2SyntheticLabTest(unittest.TestCase):
    def test_freeze_candidate_specification_closes_selection_freedom(
        self,
    ) -> None:
        specification = gate12c2.c2_freeze_candidate_specification()
        self.assertEqual(
            specification["alternative"],
            "observed_smaller_than_null",
        )
        self.assertEqual(
            specification["promotion_outcomes"],
            ["broad_replicated", "strong_broad"],
        )
        self.assertFalse(
            specification["partial_or_structured_is_promotional"]
        )
        self.assertEqual(
            specification["inner_draw_selection"]["prefix_counts"],
            [255, 511, 1023],
        )
        self.assertEqual(
            specification["inner_draw_selection"]["prefix_basis"],
            "accepted_valid_draw_index",
        )
        self.assertFalse(specification["locked_execution_authorized"])
        self.assertFalse(
            specification["real_held_out_execution_authorized"]
        )
        self.assertFalse(specification["N2_open"])
        self.assertFalse(specification["N3_open"])
        self.assertEqual(
            specification["reference_block_hierarchy"][
                "block_count_by_family"
            ],
            {"family-0": 128, "family-1": 200, "family-2": 128},
        )
        self.assertFalse(
            specification["reference_block_hierarchy"][
                "locked_schedule_frozen"
            ]
        )

    def test_s0_generation_is_deterministic_and_jointly_realizable(self) -> None:
        first = gate12c2.generate_s0_cohort(
            replicate_count=8,
            master_seed="s0-test-seed",
        )
        second = gate12c2.generate_s0_cohort(
            replicate_count=8,
            master_seed="s0-test-seed",
        )
        self.assertEqual(gate12c2.manifests(first), gate12c2.manifests(second))
        for graph in first:
            self.assertEqual(
                gate12c2.check_joint_realizability(graph)["status"], "pass"
            )
            self.assertEqual(
                gate12c2.check_block_gram_realizability(graph)["status"],
                "pass",
            )
            for node in graph.nodes:
                identity = node.frame.T @ node.frame
                np.testing.assert_allclose(
                    identity,
                    np.eye(node.local_rank),
                    atol=1.0e-12,
                    rtol=0.0,
                )

    def test_different_master_seed_changes_graphs(self) -> None:
        first = gate12c2.generate_s0_cohort(
            replicate_count=4,
            master_seed="seed-a",
        )
        second = gate12c2.generate_s0_cohort(
            replicate_count=4,
            master_seed="seed-b",
        )
        first_hashes = [
            node["frame_sha256"]
            for graph in gate12c2.manifests(first)
            for node in graph["nodes"]
        ]
        second_hashes = [
            node["frame_sha256"]
            for graph in gate12c2.manifests(second)
            for node in graph["nodes"]
        ]
        self.assertNotEqual(first_hashes, second_hashes)

    def test_n1_reassigns_within_strata_and_reconstructs_edges(self) -> None:
        observed = gate12c2.generate_s0_cohort(
            replicate_count=12,
            master_seed="n1-observed",
        )
        reassigned = gate12c2.n1_role_constrained_reassignment(
            observed,
            reassignment_seed="n1-reassignment",
        )
        self.assertEqual(
            [graph.replicate_id for graph in observed],
            [graph.replicate_id for graph in reassigned],
        )
        for source, null_graph in zip(observed, reassigned):
            self.assertEqual(
                gate12c2.check_joint_realizability(null_graph)["status"], "pass"
            )
            donors = null_graph.metadata["donor_node_ids"]
            self.assertEqual(set(donors), {node.node_id for node in source.nodes})
            for node in null_graph.nodes:
                donor_graph_id = str(donors[node.node_id]).split("/", 1)[0]
                self.assertNotEqual(donor_graph_id, source.replicate_id)
        source_frames_by_stratum = {}
        null_frames_by_stratum = {}
        for graph in observed:
            for node in graph.nodes:
                source_frames_by_stratum.setdefault(node.stratum, []).append(
                    node.frame.tobytes()
                )
        for graph in reassigned:
            for node in graph.nodes:
                null_frames_by_stratum.setdefault(node.stratum, []).append(
                    node.frame.tobytes()
                )
        self.assertEqual(
            {
                key: sorted(values)
                for key, values in source_frames_by_stratum.items()
            },
            {
                key: sorted(values)
                for key, values in null_frames_by_stratum.items()
            },
        )
        audit = gate12c2.n1_reassignment_audit(observed, reassigned)
        self.assertEqual(audit["status"], "pass")
        self.assertEqual(audit["fixed_point_count"], 0)
        self.assertEqual(audit["same_graph_assignment_count"], 0)
        self.assertEqual(audit["unique_donor_count"], 36)
        self.assertFalse(audit["reused_donor_counts"])
        self.assertFalse(audit["derangement_ineligible_strata"])

    def test_residual_decomposition_identity(self) -> None:
        rng = np.random.Generator(np.random.PCG64(20260724))
        m0 = rng.normal(size=(4, 4))
        m1 = rng.normal(size=(4, 4))
        m2 = rng.normal(size=(4, 4))
        result = gate12c2.residual_diagnostics(m0, m1, m2, q=2)
        self.assertEqual(result.eligibility_status, "eligible")
        self.assertEqual(result.numerical_status, "pass")
        self.assertIsNotNone(result.alignment)
        self.assertLessEqual(result.matrix_identity_error, 1.0e-10)
        self.assertLessEqual(result.squared_identity_error, 1.0e-10)

    def test_full_rank_control_is_zero_and_degenerate_is_explicit(self) -> None:
        rng = np.random.Generator(np.random.PCG64(77))
        matrices = [rng.normal(size=(3, 3)) for _ in range(3)]
        result = gate12c2.residual_diagnostics(*matrices, q=3)
        self.assertEqual(result.eligibility_status, "full_rank_control")
        self.assertEqual(result.numerical_status, "pass")
        self.assertAlmostEqual(result.defect or 0.0, 0.0, places=10)
        self.assertIsNone(result.alignment)
        self.assertTrue(result.alignment_status.startswith("undefined_degenerate"))
        self.assertIsNone(result.propagation_left)
        self.assertIsNone(result.propagation_right)

    def test_unstable_cut_is_not_silently_evaluated(self) -> None:
        identity = np.eye(3)
        result = gate12c2.residual_diagnostics(
            identity,
            identity,
            identity,
            q=1,
        )
        self.assertEqual(result.eligibility_status, "unstable_spectral_cut")
        self.assertEqual(result.numerical_status, "not_evaluated")
        self.assertIsNone(result.defect)

    def test_corrupted_edge_fails_independent_realizability_check(self) -> None:
        graph = gate12c2.generate_s0_cohort(
            replicate_count=2,
            master_seed="corruption-test",
        )[0]
        first_edge = graph.edges[0]
        corrupt_edge = gate12c2.EdgeOverlap(
            edge_id=first_edge.edge_id,
            source_node_id=first_edge.source_node_id,
            target_node_id=first_edge.target_node_id,
            matrix=first_edge.matrix + 0.1,
        )
        corrupt_graph = gate12c2.SyntheticGraph(
            replicate_id=graph.replicate_id,
            regime=graph.regime,
            nodes=graph.nodes,
            edges=(corrupt_edge, *graph.edges[1:]),
            cycle_node_ids=graph.cycle_node_ids,
            generator_id=graph.generator_id,
            seed_receipt=graph.seed_receipt,
        )
        check = gate12c2.check_joint_realizability(corrupt_graph)
        self.assertEqual(check["status"], "fail")
        self.assertEqual(check["failures"][0]["reason"], "overlap_mismatch")
        block_check = gate12c2.check_block_gram_realizability(corrupt_graph)
        self.assertEqual(block_check["status"], "fail")
        self.assertEqual(
            block_check["failures"][0]["reason"],
            "block_overlap_mismatch",
        )

    def test_development_report_refuses_type_i_claim(self) -> None:
        observed = gate12c2.generate_s0_cohort(
            replicate_count=16,
            master_seed="report-observed",
        )
        comparison = gate12c2.n1_role_constrained_reassignment(
            observed,
            reassignment_seed="report-n1",
        )
        report = gate12c2.development_s0_n1_report(
            observed,
            comparison,
            q=1,
        )
        self.assertEqual(report["epistemic_status"], "development_only")
        self.assertEqual(report["independent_unit"], "synthetic_graph_replicate")
        self.assertEqual(report["replicate_count"], 16)
        self.assertEqual(
            report["joint_realizability"]["comparison_pass_count"], 16
        )
        self.assertEqual(
            report["joint_realizability"][
                "comparison_block_gram_pass_count"
            ],
            16,
        )
        self.assertEqual(report["n1_assignment_audit"]["status"], "pass")
        self.assertIn("N1 is the sole primary candidate", report[
            "candidate_selection_policy"
        ])
        self.assertIn(
            "product_singular_values_left",
            report["rows"][0]["observed"],
        )
        self.assertEqual(
            report["type_i_calibration"]["status"],
            "not_estimated_without_frozen_decision_rule",
        )
        self.assertIsNone(
            report["type_i_calibration"]["false_positive_rate"]
        )

    @staticmethod
    def _pipeline_inputs(
        supported: set[tuple[int, int]],
    ) -> tuple[object, ...]:
        rows = []
        for case_order in range(12):
            for q in (1, 2):
                is_supported = (case_order, q) in supported
                rows.append(
                    gate12c2.EndpointDecisionInput(
                        case_id=f"case-{case_order:02d}",
                        case_order=case_order,
                        model=f"model-{case_order % 4}",
                        family=f"family-{case_order // 4}",
                        q=q,
                        coverage_complete=True,
                        informative=True,
                        median_log_ratio=-1.0 if is_supported else 1.0,
                        directional_raw_p=0.001 if is_supported else 1.0,
                    )
                )
        return tuple(rows)

    def test_pipeline_calibration_keeps_type_i_units_separate(self) -> None:
        no_support = gate12c2.complete_pipeline_decision(
            self._pipeline_inputs(set())
        )
        endpoint_only = gate12c2.complete_pipeline_decision(
            self._pipeline_inputs({(0, 1)})
        )
        one_run = gate12c2.complete_pipeline_decision(
            self._pipeline_inputs({(0, 1), (0, 2)})
        )
        self.assertFalse(no_support["any_endpoint_support"])
        self.assertTrue(endpoint_only["any_endpoint_support"])
        self.assertFalse(endpoint_only["any_run_support"])
        self.assertFalse(endpoint_only["claim_promotion"])
        self.assertTrue(one_run["any_run_support"])
        self.assertEqual(one_run["grid_outcome"], "partial_or_structured")
        self.assertFalse(one_run["claim_promotion"])
        self.assertFalse(one_run["partial_or_structured_is_promotional"])

        summary = gate12c2.summarize_outer_calibration(
            (no_support, endpoint_only, one_run)
        )
        self.assertEqual(summary["outer_experiment_count"], 3)
        self.assertEqual(
            summary["epistemic_status"],
            "development_only_contract_v0.2_gates_applied",
        )
        self.assertAlmostEqual(
            summary["family_wise_fpr"]["estimate"],
            2.0 / 3.0,
        )
        self.assertAlmostEqual(
            summary["run_level_fpr"]["estimate"],
            1.0 / 3.0,
        )
        self.assertAlmostEqual(
            summary["claim_promotion_false_rate"]["estimate"],
            0.0,
        )
        self.assertEqual(
            summary["calibration_gate_assessment"]["status"],
            "development_assessment_under_contract_v0.2",
        )

    def test_pipeline_decision_rejects_incomplete_outer_unit(self) -> None:
        with self.assertRaises(gate12c2.Gate12C2DevelopmentError):
            gate12c2.complete_pipeline_decision(
                self._pipeline_inputs(set())[:-1]
            )

    def test_pipeline_direction_is_explicitly_reverse(self) -> None:
        reverse = gate12c2.complete_pipeline_decision(
            self._pipeline_inputs({(0, 1), (0, 2)})
        )
        endpoint = reverse["endpoint_rows"][0]
        self.assertEqual(
            endpoint["alternative"],
            "observed_smaller_than_null",
        )
        self.assertGreater(endpoint["directional_effect"], 0.0)
        self.assertTrue(endpoint["q_directional_support"])

        wrong_direction = list(self._pipeline_inputs(set()))
        wrong_direction[0] = gate12c2.EndpointDecisionInput(
            case_id="case-00",
            case_order=0,
            model="model-0",
            family="family-0",
            q=1,
            coverage_complete=True,
            informative=True,
            median_log_ratio=1.0,
            directional_raw_p=0.001,
        )
        decision = gate12c2.complete_pipeline_decision(wrong_direction)
        self.assertFalse(
            decision["endpoint_rows"][0]["q_directional_support"]
        )

        sign_test = gate12c2.exact_directional_sign_p(
            (-1.0, -2.0, -3.0, 1.0)
        )
        self.assertEqual(sign_test["directional_count"], 3)
        self.assertAlmostEqual(sign_test["directional_raw_p"], 5.0 / 16.0)

    def test_typed_seed_namespace_is_order_and_resume_invariant(self) -> None:
        keys = [
            gate12c2.OuterSeedNamespace(
                surface_id="development",
                null_candidate_id=gate12c2.N1_ID,
                regime_id="S0_true_null",
                effect_strength=None,
                outer_experiment_index=3,
                case_or_endpoint_id="case-02",
                cycle_or_root_id="N1_reassigned_block_cohort",
                draw_attempt_index=index,
            )
            for index in range(4)
        ]
        forward = {
            key.draw_attempt_index: gate12c2.typed_seed_receipt("master", key)
            for key in keys
        }
        reverse = {
            key.draw_attempt_index: gate12c2.typed_seed_receipt("master", key)
            for key in reversed(keys)
        }
        self.assertEqual(forward, reverse)
        self.assertEqual(
            forward[2],
            gate12c2.typed_seed_receipt("master", keys[2]),
        )
        self.assertNotEqual(
            forward[1]["seed_uint64"],
            forward[2]["seed_uint64"],
        )

    def test_valid_draw_prefix_is_based_on_acceptance_not_attempt(self) -> None:
        digest = "a" * 64
        attempts = (
            gate12c2.NullDrawAttempt(
                attempt_index=0,
                accepted=False,
                value=None,
                rejection_reason="unstable_cut",
                accepted_draw_index=None,
                seed_namespace_sha256=digest,
            ),
            gate12c2.NullDrawAttempt(
                attempt_index=1,
                accepted=True,
                value=1.0,
                rejection_reason=None,
                accepted_draw_index=0,
                seed_namespace_sha256=digest,
            ),
            gate12c2.NullDrawAttempt(
                attempt_index=2,
                accepted=False,
                value=None,
                rejection_reason="numerical_failure",
                accepted_draw_index=None,
                seed_namespace_sha256=digest,
            ),
            gate12c2.NullDrawAttempt(
                attempt_index=3,
                accepted=True,
                value=2.0,
                rejection_reason=None,
                accepted_draw_index=1,
                seed_namespace_sha256=digest,
            ),
        )
        stream = gate12c2.accepted_valid_draw_stream(
            attempts,
            required_valid_count=2,
        )
        self.assertTrue(stream["complete"])
        self.assertEqual(stream["accepted_values"], [1.0, 2.0])
        self.assertEqual(stream["final_attempt_index"], 3)
        stability = gate12c2.nested_inner_draw_stability_from_attempts(
            attempts,
            observed_value=0.5,
            prefix_counts=(1, 2),
        )
        self.assertEqual(
            stability["draw_stream_basis"],
            "accepted_valid_draw_index",
        )
        self.assertEqual(stability["attempt_count_to_largest_prefix"], 4)

    def test_graph_derived_outer_experiment_carries_full_hierarchy(self) -> None:
        report = gate12c2.run_development_outer_experiment(
            regime_id="S0_true_null",
            master_seed="outer-unit-test",
            outer_experiment_index=0,
            block_count=6,
            inner_valid_draw_count=3,
            max_draw_attempts=16,
        )
        self.assertEqual(report["surface_id"], "development")
        self.assertFalse(report["locked_execution_authorized"])
        self.assertEqual(len(report["case_receipts"]), 12)
        self.assertEqual(len(report["endpoint_receipts"]), 24)
        self.assertEqual(
            report["pipeline_decision"]["endpoint_count"],
            24,
        )
        self.assertEqual(
            report["dependency_structure"],
            "q1_q2_share_observed_blocks_and_N1_draws_within_case",
        )
        self.assertEqual(
            report["accepted_valid_draw_storage"],
            gate12c2.COMPACT_ACCEPTED_PREFIX_STORAGE_ID,
        )
        self.assertEqual(
            report["accepted_valid_draw_order"],
            "draw_attempt_order_first_required_valid",
        )

    def test_outer_experiment_accepts_case_specific_block_schedule(self) -> None:
        schedule = {
            case["case_id"]: 4 + int(case["family"].split("-")[1])
            for case in gate12c2._outer_case_grid()
        }
        report = gate12c2.run_development_outer_experiment(
            regime_id="S0_true_null",
            master_seed="outer-schedule-unit-test",
            outer_experiment_index=0,
            block_count=schedule,
            inner_valid_draw_count=1,
            max_draw_attempts=8,
        )
        self.assertEqual(
            report["block_count_schedule"]["mode"],
            "case_specific",
        )
        self.assertEqual(
            report["block_count_schedule"]["block_count_by_case"],
            schedule,
        )
        expected = {
            row["case_id"]: row["expected_block_count"]
            for row in report["case_receipts"]
        }
        self.assertEqual(expected, schedule)
        for row in report["endpoint_receipts"]:
            self.assertEqual(
                row["expected_block_count"],
                schedule[row["endpoint_id"].split(":", 1)[0]],
            )

    def test_s2_accepts_case_specific_block_schedule(self) -> None:
        schedule = {
            case["case_id"]: 4 + int(case["family"].split("-")[1])
            for case in gate12c2._outer_case_grid()
        }
        report = gate12c2.run_development_s2_identification_experiment(
            master_seed="s2-schedule-unit-test",
            outer_experiment_index=0,
            block_count=schedule,
            inner_valid_draw_count=1,
            max_draw_attempts=8,
        )
        self.assertEqual(
            report["block_count_schedule"]["block_count_by_case"],
            schedule,
        )
        for row in report["endpoint_rows"]:
            self.assertEqual(
                row["expected_block_count"],
                schedule[row["case_id"]],
            )

    def test_s2_outer_experiment_attributes_only_null_side_change(self) -> None:
        report = gate12c2.run_development_s2_identification_experiment(
            master_seed="s2-outer-unit-test",
            outer_experiment_index=0,
            block_count=4,
            inner_valid_draw_count=2,
            max_draw_attempts=10,
        )
        self.assertFalse(report["observed_process_modified"])
        self.assertFalse(report["locked_execution_authorized"])
        self.assertEqual(len(report["endpoint_rows"]), 24)
        self.assertEqual(len(report["case_rows"]), 12)
        self.assertEqual(
            report["accepted_valid_draw_storage"],
            gate12c2.COMPACT_ACCEPTED_PREFIX_STORAGE_ID,
        )
        for row in report["endpoint_rows"]:
            self.assertFalse(row["observed_process_modified"])
            self.assertIn("observed", row["component_medians"])
            self.assertIn("N1", row["component_medians"])
            self.assertIn(
                "graph_unconstrained_stressor",
                row["component_medians"],
            )
            self.assertEqual(
                set(row["inflation_consistent_channels"]),
                {"x_increased", "y_increased", "c_decreased"},
            )

    def test_residual_mechanism_controls_are_separated(self) -> None:
        controls = gate12c2.development_residual_mechanism_controls()
        self.assertEqual(set(controls), {"tail", "propagation", "alignment"})
        tail_defects = []
        for row in controls["tail"]:
            diagnostic = row.diagnostics
            self.assertEqual(diagnostic.numerical_status, "pass")
            self.assertAlmostEqual(diagnostic.tail_left, row.level)
            self.assertAlmostEqual(diagnostic.tail_right, row.level)
            self.assertAlmostEqual(diagnostic.propagation_left, 1.0)
            self.assertAlmostEqual(diagnostic.propagation_right, 1.0)
            self.assertAlmostEqual(diagnostic.alignment, 0.0)
            self.assertFalse(row.as_dict()["end_to_end_s1_satisfied"])
            tail_defects.append(diagnostic.defect)
        self.assertEqual(tail_defects, sorted(tail_defects))

        propagation_defects = []
        for row in controls["propagation"]:
            diagnostic = row.diagnostics
            self.assertAlmostEqual(diagnostic.tail_left, 0.2)
            self.assertAlmostEqual(diagnostic.tail_right, 0.2)
            self.assertAlmostEqual(
                diagnostic.propagation_left,
                row.level,
            )
            self.assertAlmostEqual(
                diagnostic.propagation_right,
                row.level,
            )
            propagation_defects.append(diagnostic.defect)
        self.assertEqual(propagation_defects, sorted(propagation_defects))

        alignments = []
        alignment_defects = []
        for row in controls["alignment"]:
            diagnostic = row.diagnostics
            self.assertAlmostEqual(diagnostic.propagated_left, 0.2)
            self.assertAlmostEqual(diagnostic.propagated_right, 0.2)
            alignments.append(diagnostic.alignment)
            alignment_defects.append(diagnostic.defect)
        self.assertEqual(alignments, sorted(alignments, reverse=True))
        self.assertEqual(alignment_defects, sorted(alignment_defects))

    def test_s2_orientation_stress_preserves_spectra_but_breaks_graph(
        self,
    ) -> None:
        observed = gate12c2.generate_s0_cohort(
            replicate_count=8,
            master_seed="s2-observed",
        )
        comparison = gate12c2.s2_graph_unconstrained_orientation_draw(
            observed,
            orientation_seed="s2-orientation",
        )
        discrepancy = gate12c2.edge_spectrum_marginal_discrepancy(
            observed,
            comparison,
        )
        self.assertLessEqual(
            discrepancy["maximum_absolute_sorted_difference"],
            1.0e-12,
        )
        self.assertTrue(
            all(
                gate12c2.check_joint_realizability(graph)["status"] == "fail"
                for graph in comparison
            )
        )
        self.assertTrue(
            all(
                gate12c2.check_block_gram_realizability(graph)["status"]
                == "fail"
                for graph in comparison
            )
        )
        report = gate12c2.development_s2_null_inflation_report(
            observed,
            comparison,
            q=1,
        )
        self.assertFalse(report["observed_process_modified"])
        self.assertFalse(report["comparison_is_candidate_null"])
        self.assertEqual(report["valid_pair_count"], 8)
        self.assertEqual(
            report["realizability_failure_count"]["block_gram_checker"],
            8,
        )
        self.assertIn("a_q", report["component_difference_medians"])

    def test_s1_shared_coupling_is_graph_realizable_and_graded(self) -> None:
        n1_medians = []
        for effect_strength in (0.02, 0.05, 0.10, 0.20):
            observed = gate12c2.generate_s1_shared_node_coupling_cohort(
                replicate_count=24,
                master_seed="s1-test",
                effect_strength=effect_strength,
            )
            comparison = gate12c2.n1_role_constrained_reassignment(
                observed,
                reassignment_seed="s1-n1",
            )
            report = gate12c2.development_s1_known_reverse_report(
                observed,
                comparison,
                q=1,
            )
            self.assertEqual(report["informative_pair_count"], 24)
            self.assertGreaterEqual(report["observed_smaller_count"], 22)
            self.assertGreater(
                report["component_medians"]["a_q"]["observed"],
                0.0,
            )
            self.assertEqual(
                report["joint_realizability"]["N1_block_gram_pass_count"],
                24,
            )
            self.assertEqual(
                report["power"]["status"],
                "not_estimated_without_outer_experiments",
            )
            n1_medians.append(report["component_medians"]["a_q"]["N1"])
        self.assertEqual(n1_medians, sorted(n1_medians))

    def test_inner_draw_stability_uses_nested_stream_only(self) -> None:
        rng = np.random.Generator(np.random.PCG64(123))
        draws = rng.lognormal(size=1023)
        first = gate12c2.nested_inner_draw_stability(
            draws,
            observed_value=0.5,
            runtime_seconds_by_prefix={
                255: 1.0,
                511: 2.0,
                1023: 4.0,
            },
        )
        second = gate12c2.nested_inner_draw_stability(
            draws,
            observed_value=0.5,
            runtime_seconds_by_prefix={
                255: 1.0,
                511: 2.0,
                1023: 4.0,
            },
        )
        self.assertEqual(first, second)
        self.assertTrue(first["nested_prefix_contract"])
        self.assertEqual(first["prefix_counts"], [255, 511, 1023])
        self.assertIsNone(first["selected_draw_count"])
        self.assertIn(
            "best_observed_FPR",
            first["selection_basis_prohibited"],
        )


if __name__ == "__main__":
    unittest.main()
