#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


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
_load("process_triage_baseline_diagnostics")
structural = _load("process_triage_structural_signal")
development = _load("process_triage_structural_development")


def record(
    query_index: int,
    sample_index: int,
    labels: tuple[int, ...],
) -> dict:
    messages = [{"role": "user", "content": f"task {query_index}"}]
    step_labels = {}
    for index, label in enumerate(labels):
        message_index = len(messages)
        messages.append(
            {
                "role": "assistant",
                "content": f"step {query_index} {index}",
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
        "data_source": "structural_development_test",
        "messages": messages,
        "step_labels": step_labels,
        "final_label": -1 if -1 in labels else 1,
    }


class ProcessTriageStructuralDevelopmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.trajectories = tuple(
            triage.parse_agent_process_bench_record(
                record(
                    query_index=query_index,
                    sample_index=slot,
                    labels=(
                        (1, -1, 1)
                        if query_index % 2 == 0
                        else (1, 1, 1)
                    ),
                ),
                domain=domain,
            )
            for domain in ("a", "b")
            for query_index in range(4)
            for slot in range(2)
        )
        feature_surface = triage.build_feature_surface(
            self.trajectories
        )
        self.features = triage.cheap_features(feature_surface)
        self.cv = triage.development_cluster_cv_manifest(
            self.trajectories,
            development_cluster_ids={
                trajectory.group_id
                for trajectory in self.trajectories
            },
            fold_count=4,
            seed_id=triage.BASELINE_CV_SEED_ID,
        )
        structural_surface = structural.build_structural_surface(
            self.trajectories
        )
        self.structural_modes = {
            mode: structural.task_anchored_triangle_excess(
                structural_surface,
                mode=mode,
            )
            for mode in (
                "primary",
                "score_order_shuffle",
                "dependency_cycle_randomization",
            )
        }

    def test_label_profile_permutation_preserves_prevalence(self) -> None:
        permuted, receipt = development.permute_first_defect_profiles(
            self.trajectories
        )
        self.assertEqual(
            sum(
                row.first_actionable_row_id is not None
                for row in permuted
            ),
            sum(
                row.first_actionable_row_id is not None
                for row in self.trajectories
            ),
        )
        self.assertEqual(receipt["singleton_stratum_count"], 0)

    def test_development_report_reproduces_sealed_baseline(self) -> None:
        baseline_report = baseline.fit_development_baseline(
            self.trajectories,
            self.features,
            cv_manifest=self.cv,
        )
        report = development.run_structural_development(
            self.trajectories,
            self.features,
            cv_manifest=self.cv,
            group_aliases={},
            structural_modes=self.structural_modes,
        )
        self.assertEqual(
            report["sealed_baseline"][
                "selected_oof_score_payload_sha256"
            ],
            baseline_report["selected_oof_score_payload_sha256"],
        )
        self.assertEqual(
            set(report["frozen_structural_controls"]),
            {
                "score_order_shuffle",
                "dependency_cycle_randomization",
            },
        )
        self.assertEqual(
            set(report["required_ablations"]),
            set(development.ABLATION_CONFIGURATIONS),
        )
        self.assertTrue(
            report["structural_signal_opened_on_development"]
        )
        self.assertFalse(
            report["prospective_locked_partition_scored"]
        )


if __name__ == "__main__":
    unittest.main()
