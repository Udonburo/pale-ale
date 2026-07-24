#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


EVALUATOR_PATH = Path(__file__).with_name("process_triage_evaluator.py")
EVALUATOR_SPEC = importlib.util.spec_from_file_location(
    "process_triage_evaluator",
    EVALUATOR_PATH,
)
if EVALUATOR_SPEC is None or EVALUATOR_SPEC.loader is None:
    raise RuntimeError(f"could not import {EVALUATOR_PATH}")
triage = importlib.util.module_from_spec(EVALUATOR_SPEC)
sys.modules[EVALUATOR_SPEC.name] = triage
EVALUATOR_SPEC.loader.exec_module(triage)

BASELINE_PATH = Path(__file__).with_name("process_triage_baseline.py")
BASELINE_SPEC = importlib.util.spec_from_file_location(
    "process_triage_baseline",
    BASELINE_PATH,
)
if BASELINE_SPEC is None or BASELINE_SPEC.loader is None:
    raise RuntimeError(f"could not import {BASELINE_PATH}")
baseline = importlib.util.module_from_spec(BASELINE_SPEC)
sys.modules[BASELINE_SPEC.name] = baseline
BASELINE_SPEC.loader.exec_module(baseline)


def record(
    *,
    query_index: int,
    sample_index: int,
    labels: tuple[int, ...],
) -> dict:
    messages = [{"role": "user", "content": f"task {query_index}"}]
    step_labels = {}
    for eligible_index, label in enumerate(labels):
        message_index = len(messages)
        messages.append(
            {
                "role": "assistant",
                "content": f"step {eligible_index}",
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
        "data_source": "baseline_synthetic_test",
        "messages": messages,
        "step_labels": step_labels,
        "final_label": -1 if -1 in labels else 1,
    }


def synthetic_surface() -> tuple:
    return tuple(
        triage.parse_agent_process_bench_record(
            record(
                query_index=query_index,
                sample_index=sample_index,
                labels=(1, -1) if sample_index == 0 else (1, 1),
            ),
            domain=domain,
        )
        for domain in ("bfcl", "tau2")
        for query_index in range(4)
        for sample_index in range(2)
    )


class ProcessTriageBaselineTest(unittest.TestCase):
    def test_encoder_uses_frozen_features_and_unknown_category_policy(
        self,
    ) -> None:
        trajectories = synthetic_surface()
        features = triage.cheap_features(
            triage.build_feature_surface(trajectories)
        )
        encoder = baseline.BaselineEncoder.fit(features)
        ordered, matrix = encoder.transform(features)
        self.assertEqual(len(ordered), len(features))
        self.assertEqual(matrix.shape[1], len(encoder.column_names))
        self.assertEqual(encoder.column_names[0], "intercept")
        self.assertTrue(all(value == 1.0 for value in matrix[:, 0]))
        self.assertEqual(
            encoder.as_dict()["unknown_category_policy"],
            "all_zero_indicators",
        )

    def test_development_baseline_is_deterministic_and_oof_complete(
        self,
    ) -> None:
        trajectories = synthetic_surface()
        features = triage.cheap_features(
            triage.build_feature_surface(trajectories)
        )
        development_ids = {row.group_id for row in trajectories}
        cv_manifest = triage.development_cluster_cv_manifest(
            trajectories,
            development_cluster_ids=development_ids,
            fold_count=4,
            seed_id=triage.BASELINE_CV_SEED_ID,
        )
        first = baseline.fit_development_baseline(
            trajectories,
            features,
            cv_manifest=cv_manifest,
        )
        second = baseline.fit_development_baseline(
            tuple(reversed(trajectories)),
            tuple(reversed(features)),
            cv_manifest=cv_manifest,
        )
        self.assertEqual(
            first["selected_regularization_c"],
            second["selected_regularization_c"],
        )
        self.assertEqual(
            first["selected_oof_score_payload_sha256"],
            second["selected_oof_score_payload_sha256"],
        )
        self.assertEqual(first["development_cluster_count"], 8)
        self.assertEqual(first["row_count"], 32)
        self.assertEqual(first["positive_target_row_count"], 8)
        self.assertEqual(len(first["candidate_rows"]), 3)
        self.assertFalse(first["structural_signal_opened"])
        self.assertFalse(first["locked_partition_scored"])
        self.assertFalse(first["locked_evaluation_authorized"])
        self.assertEqual(
            first["refit_model"]["fit_diagnostics"]["status"],
            "converged",
        )

    def test_fit_rejects_nonfrozen_regularization_grid(self) -> None:
        trajectories = synthetic_surface()
        features = triage.cheap_features(
            triage.build_feature_surface(trajectories)
        )
        cv_manifest = triage.development_cluster_cv_manifest(
            trajectories,
            development_cluster_ids={row.group_id for row in trajectories},
            fold_count=4,
            seed_id=triage.BASELINE_CV_SEED_ID,
        )
        with self.assertRaises(triage.ProcessTriageDevelopmentError):
            baseline.fit_development_baseline(
                trajectories,
                features,
                cv_manifest=cv_manifest,
                regularization_grid=(1.0,),
            )

    def test_model_payload_hash_is_stable(self) -> None:
        trajectories = synthetic_surface()
        features = triage.cheap_features(
            triage.build_feature_surface(trajectories)
        )
        targets = {
            step.row_id: int(
                step.row_id == trajectory.first_actionable_row_id
            )
            for trajectory in trajectories
            for step in trajectory.steps
        }
        model = baseline.fit_baseline_model(
            features,
            targets_by_row=targets,
            regularization_c=1.0,
        )
        first = model.as_dict()
        second = model.as_dict()
        self.assertEqual(first, second)
        self.assertEqual(len(first["model_payload_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
