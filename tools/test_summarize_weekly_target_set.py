"""Tests for the weekly target-set summary helper."""

from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import package_eval_factory_receipt as receipt_helper
import run_eval_checks as runner
import summarize_weekly_target_set as summary_helper


def make_preflight() -> runner.L4SmokePreflight:
    return runner.L4SmokePreflight(
        sys_executable="python",
        python_version="3.11-test",
        cwd=str(runner.REPO_ROOT),
        platform="Linux-test",
        os_name="Linux",
        torch_importable=True,
        torch_version="2.9.1+cu129",
        torch_cuda_available=True,
        torch_cuda_version="12.9",
        gpu_count=1,
        gpu_names=("NVIDIA L4",),
        nvidia_smi_available=True,
        nvidia_smi_path="/usr/bin/nvidia-smi",
        nvidia_smi_summary=("NVIDIA L4, 23034 MiB, 580.126.09",),
        nvidia_smi_error="",
        posture_classification=runner.POSTURE_REMOTE_CUDA_READY,
        preflight_ok=True,
        remediation_hints=(),
        errors=(),
    )


def write_weekly_fixture_run(
    repo: Path,
    target_key: str,
    run_name: str,
    created_at: str,
) -> Path:
    target = runner.l4_weekly_target_for_key(target_key)
    run_dir = repo / "runs" / run_name
    run_dir.mkdir(parents=True)
    preflight = runner.build_l4_weekly_preflight_artifact_payload(
        make_preflight(),
        target,
        "execute",
        created_at=created_at,
    )
    family_results = [
        {
            "family": family,
            "dispatch": "completed",
            "structural_flags_all_true": "True",
            "runs_first_pass_status": "pending_local_read",
        }
        for family in target.families
    ]
    status = {
        "schema_id": runner.L4_WEEKLY_STATUS_SCHEMA_ID,
        "schema_version": runner.ARTIFACT_CONTRACT_VERSION,
        "created_at": created_at,
        "tier": runner.Tier.L4_WEEKLY.value,
        "mode": "execute",
        "target": target.target_key,
        "fixed_target_set": runner.l4_weekly_fixed_target_set(target),
        "model_id": target.model_id,
        "model_label": target.model_label,
        "families": list(target.families),
        "entrypoint": "tools/run_gate12a_cross_model_replay.py",
        "command": ["python", "tools/run_gate12a_cross_model_replay.py"],
        "out_dir": f"runs/{run_name}",
        "returncode": 0,
        "preflight": preflight,
        "downstream_dispatch_summary": runner.build_downstream_dispatch_summary(
            family_results,
            [],
            0,
            expected_family_count=len(target.families),
        ),
        "result": "pass",
        "family_results": family_results,
        "notes": [],
    }
    runner.write_status_artifact(run_dir / runner.L4_WEEKLY_PREFLIGHT_FILENAME, preflight)
    runner.write_status_artifact(run_dir / runner.L4_WEEKLY_STATUS_FILENAME, status)
    (run_dir / receipt_helper.WEEKLY_EXECUTE_LOG_FILENAME).write_text("fixture weekly execute log\n", encoding="utf-8")

    summary_dir = run_dir / target.summary_run_id
    summary_dir.mkdir()
    fieldnames = [
        "model_label",
        "model_id",
        "rendering_family",
        *runner.STRUCTURAL_FLAG_COLUMNS,
        "trusted_tree_median",
        "residual_chord_median",
        "anchor_qualified_median",
        "plain_median",
        "extreme_band_first_pass_status",
    ]
    with (summary_dir / runner.CROSS_MODEL_SUMMARY_FILENAME).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, family in enumerate(target.families, start=1):
            writer.writerow(
                {
                    "model_label": target.model_label,
                    "model_id": target.model_id,
                    "rendering_family": family,
                    "zero_overlap_clear": "True",
                    "all_defined_triangles_anchor_rich": "True",
                    "trusted_tree_gt_residual_chord": "True",
                    "plain_gt_anchor_qualified": "True",
                    "trusted_tree_median": f"1.0{index}",
                    "residual_chord_median": f"0.8{index}",
                    "anchor_qualified_median": f"0.7{index}",
                    "plain_median": f"1.1{index}",
                    "extreme_band_first_pass_status": "pending_local_read",
                }
            )
    return run_dir


def package_weekly_fixture(repo: Path, target_key: str, run_name: str, created_at: str) -> Path:
    run_dir = write_weekly_fixture_run(repo, target_key, run_name, created_at)
    result = receipt_helper.package_receipt(
        run_dir,
        repo / "runs" / runner.RECEIPT_BUNDLES_DIRNAME,
        create_tarball=False,
        created_at=created_at,
        repo_root=repo,
    )
    return result.receipt_root


class WeeklyTargetSetSummaryTest(unittest.TestCase):
    def test_selects_latest_valid_bundle_per_target_and_reports_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            old_qwen2 = package_weekly_fixture(
                repo,
                "qwen2_5_3b",
                "eval_factory_l4_weekly_qwen2_5_3b_vm_fixture_old",
                "2026-04-22T00:00:00Z",
            )
            new_qwen2 = package_weekly_fixture(
                repo,
                "qwen2_5_3b",
                "eval_factory_l4_weekly_qwen2_5_3b_vm_fixture_new",
                "2026-04-23T00:00:00Z",
            )
            package_weekly_fixture(
                repo,
                "llama3_2_3b",
                "eval_factory_l4_weekly_llama3_2_3b_vm_fixture",
                "2026-04-23T01:00:00Z",
            )
            package_weekly_fixture(
                repo,
                "qwen3_4b",
                "eval_factory_l4_weekly_qwen3_4b_vm_fixture",
                "2026-04-23T02:00:00Z",
            )

            summary = summary_helper.summarize_weekly_target_set(repo)
            payload = summary_helper.summary_to_payload(summary)

        self.assertEqual(summary.found_targets, ("qwen2_5_3b", "llama3_2_3b", "qwen3_4b"))
        self.assertEqual(summary.missing_targets, ())
        self.assertEqual(summary.invalid_weekly_receipt_bundle_count, 0)
        qwen2 = next(entry for entry in summary.canonical_entries if entry.target == "qwen2_5_3b")
        self.assertTrue(qwen2.duplicates_present)
        self.assertEqual(qwen2.duplicate_bundle_count, 1)
        self.assertEqual(qwen2.bundle_path, str(new_qwen2.relative_to(repo)).replace("\\", "/"))
        self.assertNotEqual(qwen2.bundle_path, str(old_qwen2.relative_to(repo)).replace("\\", "/"))
        self.assertEqual(len(payload["canonical_bundles"]), 3)

    def test_json_and_text_output_report_missing_and_invalid_weekly_bundles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            package_weekly_fixture(
                repo,
                "qwen2_5_3b",
                "eval_factory_l4_weekly_qwen2_5_3b_vm_fixture",
                "2026-04-22T00:00:00Z",
            )
            bad_bundle = package_weekly_fixture(
                repo,
                "llama3_2_3b",
                "eval_factory_l4_weekly_llama3_2_3b_vm_fixture_bad",
                "2026-04-23T01:00:00Z",
            )
            (bad_bundle / "required_receipt_artifacts.sha256").unlink()

            summary = summary_helper.summarize_weekly_target_set(repo)
            text = summary_helper.render_text_summary(summary)

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = summary_helper.main(["--repo-root", str(repo), "--format", "json"])
            payload = json.loads(buffer.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(summary.found_targets, ("qwen2_5_3b",))
        self.assertEqual(summary.missing_targets, ("llama3_2_3b", "qwen3_4b"))
        self.assertEqual(summary.invalid_weekly_receipt_bundle_count, 1)
        self.assertIn("missing_targets: llama3_2_3b, qwen3_4b", text)
        self.assertIn("invalid_weekly_receipt_bundle_count: 1", text)
        self.assertEqual(payload["found_targets"], ["qwen2_5_3b"])
        self.assertEqual(payload["missing_targets"], ["llama3_2_3b", "qwen3_4b"])
        self.assertEqual(payload["invalid_weekly_receipt_bundle_count"], 1)


if __name__ == "__main__":
    raise SystemExit(unittest.main())
