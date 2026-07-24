#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


def _load(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


power = _load("process_triage_power_simulation")


class ProcessTriagePowerSimulationTest(unittest.TestCase):
    def clusters(self) -> tuple:
        return tuple(
            power.PowerCluster(
                cluster_id=f"{domain}:{index}",
                domain=domain,
                trajectory_count=5,
            )
            for domain in ("a", "b")
            for index in range(3)
        )

    def test_beta_probabilities_preserve_zero_icc_mean(self) -> None:
        rng = np.random.default_rng(1)
        values = power._beta_probabilities(
            rng,
            mean=0.37,
            intracluster_correlation=0.0,
            size=8,
        )
        np.testing.assert_array_equal(values, np.full(8, 0.37))

    def test_simulated_mechanism_has_requested_marginal_gain(self) -> None:
        baseline_recall = 0.35
        gain = 0.10
        loss = 0.05
        rescue = (
            gain + baseline_recall * loss
        ) / (1.0 - baseline_recall)
        implied = (
            (1.0 - baseline_recall) * rescue
            - baseline_recall * loss
        )
        self.assertAlmostEqual(implied, gain)

    def test_outer_simulation_is_deterministic(self) -> None:
        kwargs = {
            "domain_positive_prevalence": {"a": 0.6, "b": 0.4},
            "baseline_recall": 0.35,
            "true_recall_gain": 0.10,
            "loss_probability": 0.05,
            "intracluster_correlation": 0.05,
            "simulation_count": 100,
            "seed_parts": ("determinism",),
        }
        first = power._simulate_cluster_totals(
            self.clusters(),
            **kwargs,
        )
        second = power._simulate_cluster_totals(
            self.clusters(),
            **kwargs,
        )
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(first[2], second[2])

    def test_cluster_sandwich_detects_positive_difference(self) -> None:
        clusters = self.clusters()
        positives = np.full((len(clusters), 4), 5, dtype=np.int16)
        differences = np.asarray(
            [[1, 1, 2, 2]] * len(clusters),
            dtype=np.int16,
        )
        point, lower = power._cluster_sandwich_lower_bound(
            clusters,
            positives,
            differences,
        )
        self.assertTrue(np.all(point > 0.0))
        self.assertTrue(np.all(lower > 0.0))

    def test_bootstrap_weights_preserve_domain_cluster_counts(self) -> None:
        clusters = self.clusters()
        weights = power._domain_stratified_bootstrap_weights(
            clusters,
            replicate_count=199,
            seed_parts=("test",),
        )
        for domain in ("a", "b"):
            indices = [
                index
                for index, cluster in enumerate(clusters)
                if cluster.domain == domain
            ]
            np.testing.assert_array_equal(
                weights[:, indices].sum(axis=1),
                np.full(199, len(indices)),
            )

    def test_surface_validation_rejects_domain_mismatch(self) -> None:
        with self.assertRaises(power.ProcessTriagePowerError):
            power._validate_surface(
                self.clusters(),
                domain_positive_prevalence={"a": 0.5},
            )

    def test_design_summary_requires_robust_power(self) -> None:
        reports = []
        for gain in power.TRUE_RECALL_GAIN_GRID:
            for baseline_recall in power.BASELINE_RECALL_GRID:
                for icc in power.CLUSTER_ICC_GRID:
                    probability = 0.82 if gain >= 0.15 else 0.50
                    reports.append(
                        {
                            "surface_id": "full_locked_layout",
                            "nominal_true_recall_gain": gain,
                            "baseline_recall": baseline_recall,
                            "intracluster_correlation": icc,
                            "recall_rule_pass_probability": probability,
                            "recall_rule_pass_wilson_95": {
                                "lower": probability - 0.04,
                                "upper": probability + 0.04,
                            },
                        }
                    )
        summary = power._design_sensitivity_summary(reports)
        self.assertEqual(
            summary["surfaces"]["full_locked_layout"][
                "minimum_evaluated_gain_meeting_robust_planning_rule"
            ],
            0.15,
        )


if __name__ == "__main__":
    unittest.main()
