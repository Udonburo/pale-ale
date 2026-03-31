#!/usr/bin/env python3
"""Tests for Gate12A family replay runner."""

import sys
import unittest
from pathlib import Path
from unittest import mock


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import run_gate12a_family_replay as replay


class RunGate12AFamilyReplayTest(unittest.TestCase):
    def assertCommandContains(self, command: list[str], expected_fragment: str) -> None:
        self.assertTrue(any(expected_fragment in part for part in command), msg=f"missing {expected_fragment} in {command}")

    def test_default_origin_label_strips_candidate_execution_suffix(self) -> None:
        observed = []

        def fake_run(command):
            observed.append(list(command))

        args = [
            "prog",
            "--gate8-execution-dir",
            "runs/gate8s_128r_archive_candidate_execution",
        ]
        with mock.patch("sys.argv", args), mock.patch.object(replay, "run_subprocess", fake_run):
            self.assertEqual(replay.main(), 0)

        self.assertEqual(len(observed), 14)
        self.assertIn("run_gate9a_graph_gauge_consumer.py", observed[0][1])
        self.assertCommandContains(observed[0], "gate9a_gate8s_128r_archive_failure_surface")
        self.assertCommandContains(
            observed[-1],
            "gate12a_triangle_reading_packet_balanced_recheck_from_gate12a_upstream_gate8s_128r_archive_gate9k",
        )
        self.assertIn("--balanced-per-band", observed[-1])
        self.assertIn("6", observed[-1])

    def test_explicit_origin_label_and_limit_override_packet_settings(self) -> None:
        observed = []

        def fake_run(command):
            observed.append(list(command))

        args = [
            "prog",
            "--gate8-execution-dir",
            "runs/gate8_candidate_execution",
            "--origin-label",
            "gate8s_archive",
            "--reading-limit",
            "12",
            "--balanced-per-band",
            "4",
        ]
        with mock.patch("sys.argv", args), mock.patch.object(replay, "run_subprocess", fake_run):
            self.assertEqual(replay.main(), 0)

        packet_command = observed[-1]
        self.assertCommandContains(packet_command, "gate12a_triangle_reading_packet_balanced_recheck_from_gate12a_upstream_gate8s_archive_gate9k")
        self.assertIn("--limit", packet_command)
        self.assertIn("12", packet_command)
        self.assertIn("--balanced-per-band", packet_command)
        self.assertIn("4", packet_command)

    def test_zero_balanced_mode_uses_plain_packet_name(self) -> None:
        commands = replay.build_commands(
            gate8_execution_dir=Path("runs/gate8u_128r_archive_candidate_execution"),
            out_root=Path("runs"),
            origin_label="gate8u_128r_archive",
            top_k=5,
            balanced_per_band=0,
            reading_limit=0,
        )

        packet_command = commands[-1][1]
        self.assertCommandContains(
            packet_command,
            "gate12a_triangle_reading_packet_recheck_from_gate12a_upstream_gate8u_128r_archive_gate9k",
        )
        self.assertNotIn("--balanced-per-band", packet_command)
        self.assertNotIn("--limit", packet_command)
        calibration_command = commands[9][1]
        self.assertIn("--top-k", calibration_command)
        self.assertIn("5", calibration_command)


if __name__ == "__main__":
    raise SystemExit(unittest.main())
