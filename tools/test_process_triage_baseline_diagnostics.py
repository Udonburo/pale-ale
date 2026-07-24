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


triage = _load("process_triage_evaluator")
baseline = _load("process_triage_baseline")
diagnostics = _load("process_triage_baseline_diagnostics")


def record(query_index: int, sample_index: int, labels: tuple[int, ...]) -> dict:
    messages = [{"role": "user", "content": f"task {query_index}"}]
    step_labels = {}
    for index, label in enumerate(labels):
        message_index = len(messages)
        messages.append(
            {
                "role": "assistant",
                "content": f"step {index}",
                "tool_calls": [],
            }
        )
        step_labels[str(message_index)] = label
        messages.append({"role": "tool", "content": "ok"})
    return {
        "total_index": query_index * 10 + sample_index,
        "query_index": query_index,
        "sample_index": sample_index,
        "question": f"task {query_index}",
        "task_description": {"purpose": f"scenario {query_index}"},
        "data_source": "diagnostic_test",
        "messages": messages,
        "step_labels": step_labels,
        "final_label": -1 if -1 in labels else 1,
    }


class ProcessTriageBaselineDiagnosticsTest(unittest.TestCase):
    def test_helmert_contrast_is_centered_and_orthonormal(self) -> None:
        contrast = diagnostics._helmert_contrast(5)
        np.testing.assert_allclose(
            contrast.T @ contrast,
            np.eye(4),
            atol=1.0e-14,
        )
        np.testing.assert_allclose(
            np.ones(5) @ contrast,
            np.zeros(4),
            atol=1.0e-14,
        )

    def test_reduced_categorical_design_is_identifiable(self) -> None:
        trajectories = tuple(
            triage.parse_agent_process_bench_record(
                record(
                    query_index=query_index,
                    sample_index=slot,
                    labels=(1, -1) if query_index % 2 == 0 else (1, 1),
                ),
                domain=domain,
            )
            for domain in ("bfcl", "tau2")
            for query_index in range(4)
            for slot in range(3)
        )
        features = triage.cheap_features(
            triage.build_feature_surface(trajectories)
        )
        encoder = baseline.BaselineEncoder.fit(features)
        _, matrix = encoder.transform(features)
        design, metadata = diagnostics._diagnostic_design(
            encoder,
            matrix,
            numeric_features=(),
            categorical_features=("domain", "source_slot"),
        )
        self.assertEqual(
            np.linalg.matrix_rank(design),
            design.shape[1],
        )
        self.assertEqual(
            metadata["parameterization"],
            "standardized_numeric_plus_orthonormal_centered_helmert",
        )
        self.assertEqual(
            metadata["design_matrix_rank"],
            metadata["design_column_count"],
        )

    def test_diagnostics_reproduce_full_hash_and_keep_surfaces_closed(
        self,
    ) -> None:
        trajectories = tuple(
            triage.parse_agent_process_bench_record(
                record(
                    query_index=query_index,
                    sample_index=slot,
                    labels=(
                        (1, -1)
                        if query_index % 2 == 0
                        else (1, 1)
                    ),
                ),
                domain=domain,
            )
            for domain in ("bfcl", "tau2")
            for query_index in range(4)
            for slot in range(2)
        )
        features = triage.cheap_features(
            triage.build_feature_surface(trajectories)
        )
        cv_manifest = triage.development_cluster_cv_manifest(
            trajectories,
            development_cluster_ids={
                trajectory.group_id for trajectory in trajectories
            },
            fold_count=4,
            seed_id=triage.BASELINE_CV_SEED_ID,
        )
        primary = baseline.fit_development_baseline(
            trajectories,
            features,
            cv_manifest=cv_manifest,
        )
        report = diagnostics.run_baseline_diagnostics(
            trajectories,
            features,
            cv_manifest=cv_manifest,
            group_aliases={},
            expected_full_oof_sha256=primary[
                "selected_oof_score_payload_sha256"
            ],
        )
        self.assertTrue(report["full_baseline_hash_reproduced"])
        self.assertEqual(
            set(report["configuration_reports"]),
            set(diagnostics.CONFIGURATIONS),
        )
        self.assertEqual(
            len(report["leave_one_source_slot_out"]["rows"]),
            2,
        )
        self.assertFalse(report["structural_signal_opened"])
        self.assertFalse(report["prospective_locked_partition_scored"])

    def test_diagnostics_reject_mismatched_full_hash(self) -> None:
        trajectories = tuple(
            triage.parse_agent_process_bench_record(
                record(
                    query_index=query_index,
                    sample_index=slot,
                    labels=(
                        (1, -1)
                        if query_index % 2 == 0
                        else (1, 1)
                    ),
                ),
                domain=domain,
            )
            for domain in ("bfcl", "tau2")
            for query_index in range(4)
            for slot in range(2)
        )
        features = triage.cheap_features(
            triage.build_feature_surface(trajectories)
        )
        cv_manifest = triage.development_cluster_cv_manifest(
            trajectories,
            development_cluster_ids={
                trajectory.group_id for trajectory in trajectories
            },
            fold_count=4,
            seed_id=triage.BASELINE_CV_SEED_ID,
        )
        with self.assertRaises(triage.ProcessTriageDevelopmentError):
            diagnostics.run_baseline_diagnostics(
                trajectories,
                features,
                cv_manifest=cv_manifest,
                group_aliases={},
                expected_full_oof_sha256="0" * 64,
            )


if __name__ == "__main__":
    unittest.main()
