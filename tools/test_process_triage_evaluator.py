#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("process_triage_evaluator.py")
SPEC = importlib.util.spec_from_file_location(
    "process_triage_evaluator", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not import {MODULE_PATH}")
triage = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = triage
SPEC.loader.exec_module(triage)


def record(
    *,
    query_index: int,
    sample_index: int,
    labels: tuple[int, ...],
    tool_error_before: int | None = None,
) -> dict:
    messages = [{"role": "user", "content": "task"}]
    step_labels = {}
    for eligible_index, label in enumerate(labels):
        message_index = len(messages)
        messages.append(
            {
                "role": "assistant",
                "content": "" if eligible_index % 2 == 0 else f"answer {eligible_index}",
                "tool_calls": (
                    [
                        {
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": json.dumps({"q": eligible_index}),
                            },
                        }
                    ]
                    if eligible_index % 2 == 0
                    else []
                ),
            }
        )
        step_labels[str(message_index)] = label
        messages.append(
            {
                "role": "tool",
                "content": (
                    "ERROR timeout"
                    if tool_error_before == eligible_index + 1
                    else "ok"
                ),
            }
        )
    return {
        "total_index": query_index * 10 + sample_index,
        "query_index": query_index,
        "sample_index": sample_index,
        "question": f"task {query_index}",
        "task_description": {"purpose": f"scenario {query_index}"},
        "data_source": "synthetic_test_source",
        "messages": messages,
        "step_labels": step_labels,
        "final_label": -1 if -1 in labels else 1,
    }


class ProcessTriageEvaluatorTest(unittest.TestCase):
    def test_agent_process_bench_mapping_groups_samples_of_same_task(self) -> None:
        first = triage.parse_agent_process_bench_record(
            record(query_index=7, sample_index=0, labels=(1, 0, -1)),
            domain="bfcl",
        )
        second = triage.parse_agent_process_bench_record(
            record(query_index=7, sample_index=1, labels=(1, 1, 1)),
            domain="bfcl",
        )
        self.assertEqual(first.group_id, second.group_id)
        self.assertNotEqual(first.trajectory_id, second.trajectory_id)
        self.assertEqual(first.first_actionable_row_id, first.steps[2].row_id)
        self.assertFalse(first.is_clean)
        self.assertTrue(second.is_clean)

    def test_visible_task_duplicate_across_query_indices_stays_in_one_group(
        self,
    ) -> None:
        first_record = record(
            query_index=7,
            sample_index=0,
            labels=(1, -1),
        )
        second_record = record(
            query_index=9,
            sample_index=0,
            labels=(1, 1),
        )
        first_record["question"] = " Same visible task. "
        second_record["question"] = "Same   visible task."
        first_record["task_description"] = {"purpose": "shared scenario"}
        second_record["task_description"] = {"purpose": "shared scenario"}
        first_record["ground_truth"] = {"hidden_variant": "a"}
        second_record["ground_truth"] = {"hidden_variant": "b"}
        first = triage.parse_agent_process_bench_record(
            first_record,
            domain="tau2",
        )
        second = triage.parse_agent_process_bench_record(
            second_record,
            domain="tau2",
        )
        self.assertEqual(first.group_id, second.group_id)
        self.assertNotEqual(first.trajectory_id, second.trajectory_id)

    def test_loader_and_feature_extraction_are_deterministic(self) -> None:
        rows = [
            record(
                query_index=0,
                sample_index=0,
                labels=(1, 0, -1),
                tool_error_before=1,
            ),
            record(query_index=0, sample_index=1, labels=(1, 1)),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bfcl.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            trajectories = triage.load_agent_process_bench_jsonl(path)
        first = triage.cheap_features(trajectories)
        second = triage.cheap_features(trajectories)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertEqual(first[1].prior_tool_error_count, 1)
        self.assertEqual(first[1].normalized_position, 0.5)

    def test_label_alignment_mismatch_is_rejected(self) -> None:
        malformed = record(
            query_index=0,
            sample_index=0,
            labels=(1, -1),
        )
        malformed["step_labels"].pop(next(iter(malformed["step_labels"])))
        with self.assertRaises(triage.ProcessTriageDevelopmentError):
            triage.parse_agent_process_bench_record(
                malformed,
                domain="gaia_dev",
            )

    def test_global_budget_and_clean_burden(self) -> None:
        trajectories = tuple(
            triage.parse_agent_process_bench_record(row, domain="hotpotqa")
            for row in (
                record(query_index=0, sample_index=0, labels=(1, -1, -1)),
                record(query_index=1, sample_index=0, labels=(1, 1, 1)),
                record(query_index=2, sample_index=0, labels=(1, -1, 1)),
                record(query_index=3, sample_index=0, labels=(1, 1, 1)),
            )
        )
        scores = {
            step.row_id: 0.0
            for trajectory in trajectories
            for step in trajectory.steps
        }
        scores[trajectories[0].steps[1].row_id] = 3.0
        scores[trajectories[1].steps[0].row_id] = 2.0
        scores[trajectories[2].steps[1].row_id] = 1.0
        result = triage.evaluate_global_review_budget(
            trajectories,
            scores=scores,
            budget_fraction=0.25,
        )
        self.assertEqual(result["eligible_row_count"], 12)
        self.assertEqual(result["selected_row_count"], 3)
        self.assertEqual(result["positive_trajectory_count"], 2)
        self.assertEqual(result["clean_trajectory_count"], 2)
        self.assertEqual(result["first_actionable_defect_recall"], 1.0)
        self.assertAlmostEqual(result["clean_row_allocation"], 1.0 / 3.0)
        self.assertEqual(result["clean_trajectory_alert_rate"], 0.5)

    def test_tie_breaking_is_deterministic(self) -> None:
        trajectories = tuple(
            triage.parse_agent_process_bench_record(
                record(query_index=index, sample_index=0, labels=(1,)),
                domain="tau2",
            )
            for index in range(10)
        )
        scores = {
            trajectory.steps[0].row_id: 1.0 for trajectory in trajectories
        }
        first = triage.evaluate_global_review_budget(
            trajectories,
            scores=scores,
            budget_fraction=0.10,
        )
        second = triage.evaluate_global_review_budget(
            tuple(reversed(trajectories)),
            scores=scores,
            budget_fraction=0.10,
        )
        self.assertEqual(first["selected_row_ids"], second["selected_row_ids"])
        self.assertEqual(first["selected_row_count"], 1)

    def test_domain_group_split_has_no_leakage(self) -> None:
        trajectories = []
        for domain in ("bfcl", "tau2"):
            for query_index in range(5):
                for sample_index in range(2):
                    trajectories.append(
                        triage.parse_agent_process_bench_record(
                            record(
                                query_index=query_index,
                                sample_index=sample_index,
                                labels=(1, -1),
                            ),
                            domain=domain,
                        )
                    )
        split = triage.grouped_domain_split(
            trajectories,
            split_seed="split-test",
            development_groups_per_domain=1,
        )
        development = set(split["development_group_ids"])
        locked = set(split["locked_group_ids"])
        self.assertFalse(development & locked)
        self.assertEqual(split["development_group_count"], 2)
        self.assertEqual(split["locked_group_count"], 8)
        self.assertEqual(split["development_trajectory_count"], 4)
        self.assertEqual(split["locked_trajectory_count"], 16)
        repeat = triage.grouped_domain_split(
            tuple(reversed(trajectories)),
            split_seed="split-test",
            development_groups_per_domain=1,
        )
        self.assertEqual(split, repeat)

    def test_admission_summary_counts_groups_not_only_trajectories(self) -> None:
        trajectories = tuple(
            triage.parse_agent_process_bench_record(
                record(
                    query_index=query_index,
                    sample_index=sample_index,
                    labels=(-1,) if sample_index == 0 else (1,),
                ),
                domain="bfcl",
            )
            for query_index in range(3)
            for sample_index in range(2)
        )
        summary = triage.dataset_admission_summary(trajectories)["pooled"]
        self.assertEqual(summary["trajectory_count"], 6)
        self.assertEqual(summary["independent_group_count"], 3)
        self.assertEqual(summary["positive_trajectory_count"], 3)
        self.assertEqual(summary["clean_trajectory_count"], 3)

    def test_score_comparison_remains_development_only(self) -> None:
        trajectories = tuple(
            triage.parse_agent_process_bench_record(
                record(
                    query_index=index,
                    sample_index=0,
                    labels=(1, -1) if index % 2 == 0 else (1, 1),
                ),
                domain="bfcl",
            )
            for index in range(10)
        )
        features = triage.cheap_features(trajectories)
        baseline = triage.position_only_scores(features)
        augmented = dict(baseline)
        result = triage.compare_score_surfaces(
            trajectories,
            baseline_scores=baseline,
            augmented_scores=augmented,
        )
        self.assertEqual(result["epistemic_status"], "development_only")
        self.assertEqual(
            result["uncertainty"]["status"],
            "not_estimated_until_grouped_resampling_rule_is_frozen",
        )
        self.assertEqual(
            result["paired_point_differences"][
                "first_actionable_defect_recall"
            ],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
