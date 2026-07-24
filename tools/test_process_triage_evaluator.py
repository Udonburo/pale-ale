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
        surface = triage.build_feature_surface(trajectories)
        first = triage.cheap_features(surface)
        second = triage.cheap_features(surface)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertEqual(first[1].prior_tool_error_count, 1)
        self.assertEqual(first[1].normalized_position, 0.5)
        self.assertEqual(first[0].source_slot, 0)
        self.assertEqual(
            triage.feature_surface_receipt(surface)["information_horizon"],
            "full_trajectory_retrospective",
        )

    def test_feature_firewall_is_invariant_to_outcome_field_mutations(
        self,
    ) -> None:
        original = record(
            query_index=4,
            sample_index=2,
            labels=(1, 0, -1),
        )
        mutated = json.loads(json.dumps(original))
        mutated["step_labels"] = {
            key: (1 if value != 1 else 0)
            for key, value in mutated["step_labels"].items()
        }
        mutated["ground_truth"] = {"secret": "changed"}
        mutated["final_label"] = 0
        mutated["answer_text"] = "changed final answer"
        first = triage.parse_agent_process_bench_record(
            original,
            domain="bfcl",
        )
        second = triage.parse_agent_process_bench_record(
            mutated,
            domain="bfcl",
        )
        first_surface = triage.build_feature_surface((first,))
        second_surface = triage.build_feature_surface((second,))
        self.assertEqual(
            triage.feature_surface_receipt(first_surface)["sha256"],
            triage.feature_surface_receipt(second_surface)["sha256"],
        )
        self.assertEqual(
            triage.cheap_features(first_surface),
            triage.cheap_features(second_surface),
        )
        with self.assertRaises(triage.ProcessTriageDevelopmentError):
            triage.cheap_features((first,))

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
        development = set(split["development_cluster_ids"])
        locked = set(split["locked_cluster_ids"])
        self.assertFalse(development & locked)
        self.assertEqual(split["development_cluster_count"], 2)
        self.assertEqual(split["locked_cluster_count"], 8)
        self.assertEqual(split["development_trajectory_count"], 4)
        self.assertEqual(split["locked_trajectory_count"], 16)
        repeat = triage.grouped_domain_split(
            tuple(reversed(trajectories)),
            split_seed="split-test",
            development_groups_per_domain=1,
        )
        self.assertEqual(split, repeat)

    def test_protected_cluster_is_forced_to_locked_partition(self) -> None:
        trajectories = tuple(
            triage.parse_agent_process_bench_record(
                record(
                    query_index=query_index,
                    sample_index=sample_index,
                    labels=(1, -1),
                ),
                domain="bfcl",
            )
            for query_index in range(4)
            for sample_index in range(2)
        )
        protected = trajectories[0].group_id
        split = triage.grouped_domain_split(
            trajectories,
            split_seed="protected-test",
            development_groups_per_domain=1,
            protected_locked_group_ids=(protected,),
        )
        self.assertIn(protected, split["locked_cluster_ids"])
        self.assertNotIn(protected, split["development_cluster_ids"])
        self.assertEqual(
            split["protected_locked_cluster_ids"],
            [protected],
        )

    def test_freeze_candidate_specification_is_retrospective_and_bounded(
        self,
    ) -> None:
        specification = triage.process_triage_freeze_candidate_specification()
        self.assertEqual(
            specification["information_horizon"],
            "full_trajectory_retrospective",
        )
        self.assertEqual(
            specification["bootstrap"]["resampling_unit"],
            "visible_task_surface_cluster",
        )
        self.assertFalse(
            specification["bootstrap"]["statistical_independence_claimed"]
        )
        self.assertEqual(specification["bootstrap"]["replicates"], 10_000)
        self.assertEqual(
            specification["baseline"]["regularization_grid"],
            [0.1, 1.0, 10.0],
        )
        self.assertEqual(
            specification["baseline"]["solver"]["algorithm"],
            "deterministic_damped_newton",
        )
        self.assertFalse(
            specification["baseline"]["intercept_penalized"]
        )
        self.assertEqual(
            specification["baseline"]["cross_validation_folds"],
            4,
        )
        self.assertEqual(
            specification["baseline"]["hyperparameter_selection"][
                "primary_metric"
            ],
            "out_of_fold_first_actionable_defect_recall_at_global_"
            "10_percent_row_budget",
        )
        self.assertFalse(specification["structural_signal_opened"])
        self.assertFalse(specification["locked_execution_authorized"])

    def test_development_cv_manifest_is_label_blind_and_balanced(
        self,
    ) -> None:
        original_records = {
            domain: [
                record(
                    query_index=query_index,
                    sample_index=sample_index,
                    labels=(1, -1),
                )
                for query_index in range(4)
                for sample_index in range(2)
            ]
            for domain in ("bfcl", "tau2")
        }
        mutated_records = json.loads(json.dumps(original_records))
        for rows in mutated_records.values():
            for row in rows:
                row["step_labels"] = {
                    key: (0 if value == -1 else -1)
                    for key, value in row["step_labels"].items()
                }
                row["final_label"] = 0
                row["ground_truth"] = {"changed": True}

        def parse(records: dict[str, list[dict]]) -> tuple:
            return tuple(
                triage.parse_agent_process_bench_record(row, domain=domain)
                for domain, rows in sorted(records.items())
                for row in rows
            )

        first = parse(original_records)
        second = parse(mutated_records)
        development_ids = {row.group_id for row in first}
        first_manifest = triage.development_cluster_cv_manifest(
            first,
            development_cluster_ids=development_ids,
            fold_count=2,
            seed_id="cv-label-blind-test",
        )
        second_manifest = triage.development_cluster_cv_manifest(
            second,
            development_cluster_ids=development_ids,
            fold_count=2,
            seed_id="cv-label-blind-test",
        )
        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual(first_manifest["validation_coverage_count"], 8)
        for fold in first_manifest["folds"]:
            self.assertEqual(fold["validation_cluster_count"], 4)
            self.assertEqual(
                fold["validation_domain_counts"],
                {"bfcl": 2, "tau2": 2},
            )

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
        self.assertEqual(summary["leakage_control_cluster_count"], 3)
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
        features = triage.cheap_features(
            triage.build_feature_surface(trajectories)
        )
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
            "not_estimated_until_cluster_resampling_rule_is_frozen",
        )
        self.assertEqual(
            result["paired_point_differences"][
                "first_actionable_defect_recall"
            ],
            0.0,
        )

    def test_source_slot_is_opaque_and_available_as_categorical_baseline(
        self,
    ) -> None:
        trajectories = tuple(
            triage.parse_agent_process_bench_record(
                record(
                    query_index=0,
                    sample_index=slot,
                    labels=(1, -1),
                ),
                domain="bfcl",
            )
            for slot in range(3)
        )
        features = triage.cheap_features(
            triage.build_feature_surface(trajectories)
        )
        scores = triage.source_slot_only_scores(
            features,
            source_slot_weights={0: -1.0, 1: 0.0, 2: 1.0},
        )
        self.assertEqual(scores[trajectories[0].steps[0].row_id], -1.0)
        self.assertEqual(scores[trajectories[2].steps[0].row_id], 1.0)
        summary = triage.dataset_admission_summary(trajectories)
        self.assertEqual(
            summary["named_source_model_mapping_status"],
            triage.SOURCE_MODEL_MAPPING_STATUS,
        )
        self.assertFalse(
            summary["named_source_model_interpretation_authorized"]
        )
        self.assertEqual(
            summary["pooled"]["source_slot_counts"],
            {"0": 1, "1": 1, "2": 1},
        )

    def test_near_duplicate_grouping_is_label_blind_and_deterministic(
        self,
    ) -> None:
        shared = " ".join(f"token{index}" for index in range(100))
        first_record = record(
            query_index=10,
            sample_index=0,
            labels=(-1,),
        )
        second_record = record(
            query_index=11,
            sample_index=0,
            labels=(1,),
        )
        first_record["question"] = shared + " variantA"
        second_record["question"] = shared + " variantB"
        first_record["task_description"] = {"purpose": "same"}
        second_record["task_description"] = {"purpose": "same"}
        trajectories = tuple(
            triage.parse_agent_process_bench_record(row, domain="tau2")
            for row in (first_record, second_record)
        )
        self.assertNotEqual(
            trajectories[0].group_id,
            trajectories[1].group_id,
        )
        manifest = triage.near_duplicate_group_manifest(trajectories)
        repeat = triage.near_duplicate_group_manifest(
            tuple(reversed(trajectories))
        )
        self.assertEqual(manifest, repeat)
        self.assertEqual(manifest["linked_pair_count"], 1)
        self.assertEqual(manifest["component_count"], 1)
        self.assertEqual(manifest["manual_adjudication"], "prohibited")
        aliases = manifest["group_aliases"]
        self.assertEqual(
            aliases[trajectories[0].group_id],
            aliases[trajectories[1].group_id],
        )

        third_record = record(
            query_index=12,
            sample_index=0,
            labels=(1,),
        )
        third_record["question"] = "entirely unrelated short task"
        third_record["task_description"] = {"purpose": "different"}
        third = triage.parse_agent_process_bench_record(
            third_record,
            domain="tau2",
        )
        expanded = (*trajectories, third)
        expanded_manifest = triage.near_duplicate_group_manifest(expanded)
        split = triage.grouped_domain_split(
            expanded,
            split_seed="near-duplicate-split",
            development_groups_per_domain=1,
            group_aliases=expanded_manifest["group_aliases"],
        )
        self.assertEqual(split["development_cluster_count"], 1)
        self.assertEqual(split["locked_cluster_count"], 1)
        selected = triage.subset_by_clusters(
            expanded,
            split["development_cluster_ids"],
            group_aliases=expanded_manifest["group_aliases"],
        )
        self.assertIn(len(selected), {1, 2})

    def test_paired_group_bootstrap_reuses_draws_and_recomputes_budget(
        self,
    ) -> None:
        trajectories = tuple(
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
        features = triage.cheap_features(
            triage.build_feature_surface(trajectories)
        )
        baseline = {feature.row_id: 0.0 for feature in features}
        augmented = dict(baseline)
        for trajectory in trajectories:
            if trajectory.first_actionable_row_id is not None:
                augmented[trajectory.first_actionable_row_id] = 10.0
        first = triage.paired_domain_group_bootstrap(
            trajectories,
            baseline_scores=baseline,
            augmented_scores=augmented,
            replicates=25,
            seed="bootstrap-test",
            budget_fraction=0.25,
        )
        second = triage.paired_domain_group_bootstrap(
            tuple(reversed(trajectories)),
            baseline_scores=baseline,
            augmented_scores=augmented,
            replicates=25,
            seed="bootstrap-test",
            budget_fraction=0.25,
        )
        self.assertEqual(first, second)
        self.assertTrue(first["global_budget_recomputed_each_replicate"])
        self.assertEqual(first["replicates"], 25)
        recall_interval = first["paired_percentile_intervals"][
            "first_actionable_defect_recall"
        ]
        self.assertEqual(recall_interval["defined_replicate_count"], 25)
        self.assertGreaterEqual(recall_interval["percentile_95_lower"], 0.0)
        assessment = triage.assess_operational_success(
            first,
            rule=triage.OperationalSuccessRule(),
        )
        self.assertEqual(
            assessment["rule_status"],
            "candidate_not_frozen",
        )

    def test_largest_group_sensitivity_recomputes_comparison(self) -> None:
        rows = []
        for sample_index in range(3):
            rows.append(
                triage.parse_agent_process_bench_record(
                    record(
                        query_index=0,
                        sample_index=sample_index,
                        labels=(1, -1),
                    ),
                    domain="bfcl",
                )
            )
        rows.append(
            triage.parse_agent_process_bench_record(
                record(query_index=1, sample_index=0, labels=(1, -1)),
                domain="bfcl",
            )
        )
        trajectories = tuple(rows)
        features = triage.cheap_features(
            triage.build_feature_surface(trajectories)
        )
        scores = {feature.row_id: feature.normalized_position for feature in features}
        result = triage.leave_largest_group_out_sensitivity(
            trajectories,
            baseline_scores=scores,
            augmented_scores=scores,
            budget_fraction=0.5,
        )
        self.assertEqual(result["maximum_cluster_trajectory_count"], 3)
        self.assertEqual(result["largest_cluster_count"], 1)
        self.assertEqual(result["rows"][0]["remaining_trajectory_count"], 1)


if __name__ == "__main__":
    unittest.main()
