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


runner = _load("run_process_triage_power_simulation")
triage = sys.modules["process_triage_evaluator"]


class RunProcessTriagePowerSimulationTest(unittest.TestCase):
    def test_pre_outcome_view_excludes_every_outcome_field(self) -> None:
        record = {
            "question": "same question",
            "task_description": {"purpose": "same task"},
            "data_source": "same source",
            "step_labels": {"1": -1},
            "final_label": -1,
            "ground_truth": "secret",
            "reward": 0.0,
        }
        changed = {
            **record,
            "step_labels": {"1": 1},
            "final_label": 1,
            "ground_truth": "changed",
            "reward": 1.0,
        }
        left = runner._pre_outcome_task_view(record)
        right = runner._pre_outcome_task_view(changed)
        self.assertEqual(left, right)
        self.assertEqual(
            set(left),
            set(runner.PRE_OUTCOME_TASK_FIELDS),
        )
        self.assertEqual(
            triage.agent_process_bench_task_surface_group_id(
                left,
                domain="bfcl",
            ),
            triage.agent_process_bench_task_surface_group_id(
                right,
                domain="bfcl",
            ),
        )


if __name__ == "__main__":
    unittest.main()
