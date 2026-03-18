#!/usr/bin/env python3
"""Tests for Gate8 scale-up runner."""

import unittest
from unittest import mock

import run_gate8_scaleup as scaleup


class RunGate8ScaleupTest(unittest.TestCase):
    def test_command_sequence_includes_execution(self):
        observed = []

        def fake_run(command):
            observed.append(list(command))

        args = [
            "prog",
            "--run-prefix",
            "gate8b_128r",
            "--samples-per-cell",
            "32",
            "--device",
            "cpu",
            "--model-id",
            "Qwen/Qwen2.5-0.5B",
        ]
        with mock.patch("sys.argv", args), mock.patch.object(scaleup, "run_subprocess", fake_run):
            self.assertEqual(scaleup.main(), 0)

        self.assertEqual(len(observed), 3)
        self.assertIn("generate_gate8_semiclosed_conflict.py", observed[0][1])
        self.assertIn("materialize_gate8_semiclosed_conflict.py", observed[1][1])
        self.assertIn("run_gate8_candidate_batch.py", observed[2][1])

    def test_skip_execution_stops_after_materialization(self):
        observed = []

        def fake_run(command):
            observed.append(list(command))

        args = [
            "prog",
            "--run-prefix",
            "gate8b_128r",
            "--samples-per-cell",
            "32",
            "--skip-execution",
        ]
        with mock.patch("sys.argv", args), mock.patch.object(scaleup, "run_subprocess", fake_run):
            self.assertEqual(scaleup.main(), 0)

        self.assertEqual(len(observed), 2)


if __name__ == "__main__":
    raise SystemExit(unittest.main())
