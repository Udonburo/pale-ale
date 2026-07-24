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
            report["type_i_calibration"]["status"],
            "not_estimated_without_frozen_decision_rule",
        )
        self.assertIsNone(
            report["type_i_calibration"]["false_positive_rate"]
        )


if __name__ == "__main__":
    unittest.main()
