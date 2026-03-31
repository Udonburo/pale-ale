#!/usr/bin/env python3
"""Tests for Gate12A cross-model replay harness."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import run_gate12a_cross_model_replay as harness


class RunGate12ACrossModelReplayTest(unittest.TestCase):
    def assertCommandContains(self, command: list[str], expected_fragment: str) -> None:
        self.assertTrue(any(expected_fragment in part for part in command), msg=f"missing {expected_fragment} in {command}")

    def test_default_command_sequence_uses_fixed_family_set(self) -> None:
        commands = harness.build_commands(
            model_id="Qwen/Qwen2.5-0.5B",
            model_label="qwen_qwen2_5_0_5b",
            family_names=["transcript_v1", "briefing_v1", "archive_v1"],
            device="cpu",
            topk=128,
            seed=7,
            gate12a_top_k=3,
            balanced_per_band=6,
            reading_limit=0,
            out_root=Path("runs"),
        )

        self.assertEqual(len(commands), 6)
        self.assertCommandContains(commands[0], "run_gate8_scaleup.py")
        self.assertCommandContains(commands[0], "gate8cm_qwen_qwen2_5_0_5b_transcript_128r")
        self.assertCommandContains(commands[1], "run_gate12a_family_replay.py")
        self.assertCommandContains(commands[1], "runs\\gate8cm_qwen_qwen2_5_0_5b_transcript_128r_candidate_execution")
        self.assertCommandContains(commands[2], "gate8cm_qwen_qwen2_5_0_5b_briefing_200r")
        self.assertCommandContains(commands[4], "gate8cm_qwen_qwen2_5_0_5b_archive_128r")

    def test_model_label_normalization_is_stable(self) -> None:
        self.assertEqual(
            harness.normalize_model_label("mistralai/Mistral-7B-Instruct-v0.3", None),
            "mistralai_mistral_7b_instruct_v0_3",
        )
        self.assertEqual(
            harness.build_summary_run_id("mistral_7b", None),
            "gate12a_cross_model_replay_mistral_7b",
        )

    def test_packet_options_forward_to_family_replay(self) -> None:
        commands = harness.build_commands(
            model_id="meta-llama/Llama-3.2-3B-Instruct",
            model_label="llama_3_2_3b",
            family_names=["archive_v1"],
            device="cpu",
            topk=64,
            seed=9,
            gate12a_top_k=5,
            balanced_per_band=4,
            reading_limit=12,
            out_root=Path("tmp/custom_runs"),
        )

        self.assertEqual(len(commands), 2)
        replay_command = commands[1]
        self.assertIn("--top-k", replay_command)
        self.assertIn("5", replay_command)
        self.assertIn("--balanced-per-band", replay_command)
        self.assertIn("4", replay_command)
        self.assertIn("--reading-limit", replay_command)
        self.assertIn("12", replay_command)
        self.assertIn("--out-root", replay_command)
        self.assertIn("tmp\\custom_runs", replay_command)

    def test_out_root_propagates_to_spawned_gate8_and_replay_commands(self) -> None:
        commands = harness.build_commands(
            model_id="Qwen/Qwen2.5-0.5B",
            model_label="qwen",
            family_names=["transcript_v1"],
            device="cpu",
            topk=128,
            seed=7,
            gate12a_top_k=3,
            balanced_per_band=6,
            reading_limit=0,
            out_root=Path("tmp/custom_root"),
        )

        self.assertEqual(len(commands), 2)
        gate8_command, replay_command = commands
        self.assertIn("--out-root", gate8_command)
        self.assertIn("tmp\\custom_root", gate8_command)
        self.assertIn("--out-root", replay_command)
        self.assertIn("tmp\\custom_root", replay_command)
        self.assertCommandContains(replay_command, "tmp\\custom_root\\gate8cm_qwen_transcript_128r_candidate_execution")

    def test_summarize_only_skips_spawning_replays(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            summary_root = tmp / "runs"
            family_root = summary_root / "gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_transcript_128r_gate9k"
            calibration_root = summary_root / "gate12a_calibration_seed_audit_recheck_from_gate12a_upstream_gate8cm_qwen_transcript_128r_gate9k"
            packet_root = summary_root / "gate12a_triangle_reading_packet_balanced_recheck_from_gate12a_upstream_gate8cm_qwen_transcript_128r_gate9k"
            first_pass_root = summary_root / "gate12a_triangle_phenotype_first_pass_recheck_from_gate12a_upstream_gate8cm_qwen_transcript_128r_gate9k"
            for path in (family_root, calibration_root, packet_root, first_pass_root):
                path.mkdir(parents=True, exist_ok=True)

            (family_root / "gate12a_discrete_connection_status.json").write_text(
                '{"defined_triangle_holonomy_count": 2}\n',
                encoding="utf-8",
            )
            (calibration_root / "gate12a_calibration_seed_audit_status.json").write_text(
                '{"zero_overlap_count": 0, "triangles_with_any_anchor_count": 2, "triangles_with_all_anchor_count": 0}\n',
                encoding="utf-8",
            )
            (calibration_root / "transport_gap_quantiles_by_subregime.csv").write_text(
                "subregime,median\ntrusted_tree,1.0\nresidual_chord,0.5\nanchor_qualified,0.4\nplain,0.9\n",
                encoding="utf-8",
            )
            (first_pass_root / "gate12a_triangle_phenotype_first_pass_status.json").write_text(
                '{"reviewed_tag_counts":[{"reviewed_phenotype_tag":"surface_noise_only","count":2}]}\n',
                encoding="utf-8",
            )

            argv = [
                "prog",
                "--model-id",
                "Qwen/Qwen2.5-0.5B",
                "--model-label",
                "qwen",
                "--families",
                "transcript_v1",
                "--out-root",
                str(summary_root),
                "--summarize-only",
            ]
            with mock.patch("sys.argv", argv), mock.patch.object(harness, "run_subprocess") as mocked_run:
                self.assertEqual(harness.main(), 0)
                mocked_run.assert_not_called()

            summary_csv = summary_root / "gate12a_cross_model_replay_qwen" / "cross_model_family_summary.csv"
            self.assertTrue(summary_csv.exists())
            text = summary_csv.read_text(encoding="utf-8")
            self.assertIn("available", text)
            self.assertIn("surface_noise_only=2", text)


if __name__ == "__main__":
    raise SystemExit(unittest.main())
