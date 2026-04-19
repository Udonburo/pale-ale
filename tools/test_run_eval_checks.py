#!/usr/bin/env python3
"""Tests for the evaluation-factory runner."""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import run_eval_checks as runner


class RunEvalChecksTest(unittest.TestCase):
    def test_required_tiers_are_defined(self) -> None:
        self.assertEqual(
            runner.TIER_VALUES,
            ("cpu-nightly", "l4-smoke", "l4-weekly", "summarize-existing"),
        )

    def test_l4_weekly_keeps_expansion_surfaces_out_of_scope(self) -> None:
        plan = runner.dispatch(runner.Tier.L4_WEEKLY)

        self.assertIn("7B FP32", plan.out_of_scope)
        self.assertIn("protocol-expanding candidates", plan.out_of_scope)
        self.assertIn("quantized candidates", plan.out_of_scope)
        self.assertIn("sidecar candidates", plan.out_of_scope)

    def test_l4_smoke_remains_plan_only(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            self.assertEqual(runner.main(["--tier", "l4-smoke"]), 0)

        text = output.getvalue()
        self.assertIn("tier: l4-smoke", text)
        self.assertIn("planned actions:", text)
        self.assertIn("not implemented yet:", text)
        self.assertIn("0.5B fixed family boundary set", text)

    def test_summarize_existing_parses_materialized_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            summary_dir = repo / "runs" / "gate12a_cross_model_replay_demo"
            summary_dir.mkdir(parents=True)
            (summary_dir / "cross_model_family_summary.csv").write_text(
                (
                    "model_label,model_id,rendering_family,"
                    "zero_overlap_clear,all_defined_triangles_anchor_rich,"
                    "trusted_tree_gt_residual_chord,plain_gt_anchor_qualified,"
                    "extreme_band_first_pass_status\n"
                    "demo,Demo/Model,transcript_v1,True,True,True,True,available\n"
                    "demo,Demo/Model,briefing_v1,True,True,True,True,pending_local_read\n"
                ),
                encoding="utf-8",
            )
            (summary_dir / "manifest.json").write_text(
                (
                    '{"paths": {'
                    '"cross_model_family_summary.csv": '
                    '"runs/gate12a_cross_model_replay_demo/cross_model_family_summary.csv"'
                    '}, "model_id": "Demo/Model", "model_label": "demo"}\n'
                ),
                encoding="utf-8",
            )

            text = runner.render_summarize_existing(repo)

        self.assertIn("tier: summarize-existing", text)
        self.assertIn("gate12a_cross_model_replay_demo", text)
        self.assertIn("model=Demo/Model", text)
        self.assertIn("families=transcript_v1, briefing_v1", text)
        self.assertIn("runs_structural_flags_all_true=2/2", text)
        self.assertIn("runs_first_pass_status=available=1, pending_local_read=1", text)
        self.assertIn("missing families: archive_v1", text)

    def test_summarize_existing_separates_tracked_memos_from_runs_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            memo_path = repo / "workstream" / "215_GATE12A_QWEN_2_5_0_5B_FIXED_FAMILY_SET_REPLICATION_MEMO.md"
            memo_path.parent.mkdir(parents=True, exist_ok=True)
            memo_path.write_text("tracked memo placeholder\n", encoding="utf-8")
            summary_dir = repo / "runs" / "gate12a_cross_model_replay_qwen_qwen2_5_0_5b"
            summary_dir.mkdir(parents=True)
            (summary_dir / "cross_model_family_summary.csv").write_text(
                (
                    "model_label,model_id,rendering_family,"
                    "zero_overlap_clear,all_defined_triangles_anchor_rich,"
                    "trusted_tree_gt_residual_chord,plain_gt_anchor_qualified,"
                    "extreme_band_first_pass_status\n"
                    "qwen_qwen2_5_0_5b,Qwen/Qwen2.5-0.5B,transcript_v1,True,True,True,True,pending_local_read\n"
                ),
                encoding="utf-8",
            )

            text = runner.render_summarize_existing(repo)

        self.assertIn("tracked memo model surfaces:", text)
        self.assertIn("model=Qwen/Qwen2.5-0.5B; memo=215; memo_status=present", text)
        self.assertIn("runs-derived materialized cross-model summaries:", text)
        self.assertIn("runs_first_pass_status=pending_local_read=1", text)

    def test_cpu_nightly_reports_missing_required_files_as_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checks = runner.build_cpu_nightly_checks(Path(tmpdir))

        self.assertTrue(any(check.level == runner.LEVEL_FAIL for check in checks))

    def test_cpu_nightly_accepts_minimal_required_surface_with_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            for relative_path in runner.REQUIRED_CPU_FILES:
                path = repo / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("placeholder\n", encoding="utf-8")
            for memo in runner.EXPECTED_ATLAS_MEMOS:
                path = repo / "workstream" / memo
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("placeholder\n", encoding="utf-8")

            checks = runner.build_cpu_nightly_checks(repo)

        self.assertFalse(any(check.level == runner.LEVEL_FAIL for check in checks))
        self.assertTrue(any(check.level == runner.LEVEL_WARN for check in checks))


if __name__ == "__main__":
    raise SystemExit(unittest.main())
