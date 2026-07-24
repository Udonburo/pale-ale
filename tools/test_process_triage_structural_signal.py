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
structural = _load("process_triage_structural_signal")


def record(labels: tuple[int, ...]) -> dict:
    messages = [{"role": "user", "content": "find alpha"}]
    step_labels = {}
    contents = ("alpha", "alpha beta", "alpha")
    for index, (content, label) in enumerate(zip(contents, labels)):
        message_index = len(messages)
        messages.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [],
            }
        )
        step_labels[str(message_index)] = label
        messages.append({"role": "tool", "content": "ok"})
    return {
        "total_index": 1,
        "query_index": 1,
        "sample_index": 0,
        "question": "find alpha",
        "task_description": {"purpose": "return alpha"},
        "data_source": "unit",
        "messages": messages,
        "step_labels": step_labels,
        "final_label": -1 if -1 in labels else 1,
    }


class ProcessTriageStructuralSignalTest(unittest.TestCase):
    def surface(self, labels: tuple[int, ...] = (1, -1, 1)):
        trajectory = triage.parse_agent_process_bench_record(
            record(labels),
            domain="unit",
        )
        return structural.build_structural_surface((trajectory,))

    def test_surface_hash_is_invariant_to_all_outcomes(self) -> None:
        left = structural.structural_surface_receipt(
            self.surface((1, -1, 1))
        )
        right = structural.structural_surface_receipt(
            self.surface((-1, 1, -1))
        )
        self.assertEqual(left["sha256"], right["sha256"])
        self.assertEqual(
            left["firewall_status"],
            "outcome_fields_absent_by_type",
        )

    def test_primary_score_is_bounded_and_formula_exact(self) -> None:
        rows = structural.task_anchored_triangle_excess(
            self.surface()
        )
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertGreaterEqual(row.score, 0.0)
            self.assertLessEqual(row.score, 1.0)
            self.assertAlmostEqual(
                row.score,
                0.5
                * (
                    row.previous_edge_distance
                    + row.next_edge_distance
                    - row.bypass_distance
                ),
            )

    def test_single_row_score_equals_task_distance(self) -> None:
        trajectory = triage.parse_agent_process_bench_record(
            {
                **record((1,)),
                "messages": [
                    {"role": "user", "content": "find alpha"},
                    {
                        "role": "assistant",
                        "content": "unrelated beta",
                        "tool_calls": [],
                    },
                    {"role": "tool", "content": "ok"},
                ],
                "step_labels": {"1": 1},
            },
            domain="unit",
        )
        surface = structural.build_structural_surface((trajectory,))
        row = structural.task_anchored_triangle_excess(surface)[0]
        self.assertAlmostEqual(
            row.score,
            row.previous_edge_distance,
        )
        self.assertAlmostEqual(
            row.previous_edge_distance,
            row.next_edge_distance,
        )
        self.assertEqual(row.bypass_distance, 0.0)

    def test_controls_are_deterministic_and_cover_same_rows(self) -> None:
        surface = self.surface()
        primary = structural.task_anchored_triangle_excess(surface)
        shuffled_one = structural.task_anchored_triangle_excess(
            surface,
            mode="score_order_shuffle",
        )
        shuffled_two = structural.task_anchored_triangle_excess(
            surface,
            mode="score_order_shuffle",
        )
        randomized_one = structural.task_anchored_triangle_excess(
            surface,
            mode="dependency_cycle_randomization",
        )
        randomized_two = structural.task_anchored_triangle_excess(
            surface,
            mode="dependency_cycle_randomization",
        )
        expected_rows = {row.row_id for row in primary}
        self.assertEqual(
            {row.row_id for row in shuffled_one},
            expected_rows,
        )
        self.assertEqual(
            {row.row_id for row in randomized_one},
            expected_rows,
        )
        self.assertEqual(shuffled_one, shuffled_two)
        self.assertEqual(randomized_one, randomized_two)
        self.assertEqual(
            sorted(row.score for row in shuffled_one),
            sorted(row.score for row in primary),
        )
        score_permutation = (
            structural._deterministic_score_derangement(
                len(primary),
                trajectory_id=surface[0].trajectory_id,
            )
        )
        self.assertTrue(
            all(
                target != source
                for target, source in enumerate(score_permutation)
            )
        )
        dependency_order = structural._dependency_cycle_permutation(
            len(primary),
            trajectory_id=surface[0].trajectory_id,
        )
        self.assertNotEqual(
            dependency_order,
            tuple(range(len(primary) + 1)),
        )
        self.assertNotEqual(
            dependency_order,
            (0, *reversed(range(1, len(primary) + 1))),
        )

    def test_manifest_freezes_one_parameter_free_scalar(self) -> None:
        manifest = structural.structural_family_manifest()
        self.assertEqual(
            manifest["authorized_model_input_columns"],
            ["task_anchored_triangle_excess"],
        )
        self.assertFalse(manifest["learned_parameters"])
        self.assertEqual(manifest["thresholds"], [])
        self.assertEqual(manifest["window_parameters"], [])
        self.assertTrue(manifest["future_context_used"])
        self.assertFalse(manifest["locked_evaluation_authorized"])


if __name__ == "__main__":
    unittest.main()
