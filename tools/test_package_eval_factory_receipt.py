#!/usr/bin/env python3
"""Tests for the eval-factory receipt packager."""

from __future__ import annotations

import csv
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import package_eval_factory_receipt as packager
import run_eval_checks as runner


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


def write_fixture_run(repo: Path) -> Path:
    run_dir = repo / "runs" / "eval_factory_l4_smoke_vm_fixture"
    run_dir.mkdir(parents=True)
    created_at = "2026-04-21T00:00:00Z"
    preflight = runner.build_preflight_artifact_payload(make_preflight(), "execute", created_at=created_at)
    family_results = [
        {
            "family": family,
            "dispatch": "completed",
            "structural_flags_all_true": "True",
            "runs_first_pass_status": "pending_local_read",
        }
        for family in runner.FAMILY_SET
    ]
    status = {
        "schema_id": runner.L4_SMOKE_STATUS_SCHEMA_ID,
        "schema_version": runner.ARTIFACT_CONTRACT_VERSION,
        "created_at": created_at,
        "tier": runner.Tier.L4_SMOKE.value,
        "mode": "execute",
        "fixed_target_set": runner.l4_smoke_fixed_target_set(),
        "model_id": runner.L4_SMOKE_CONFIG.model_id,
        "model_label": runner.L4_SMOKE_CONFIG.model_label,
        "families": list(runner.FAMILY_SET),
        "entrypoint": "tools/run_gate12a_cross_model_replay.py",
        "command": ["python", "tools/run_gate12a_cross_model_replay.py"],
        "out_dir": "runs/eval_factory_l4_smoke_vm_fixture",
        "returncode": 0,
        "preflight": preflight,
        "downstream_dispatch_summary": runner.build_downstream_dispatch_summary(family_results, [], 0),
        "result": "pass",
        "family_results": family_results,
        "notes": [],
    }
    runner.write_status_artifact(run_dir / runner.L4_SMOKE_PREFLIGHT_FILENAME, preflight)
    runner.write_status_artifact(run_dir / runner.L4_SMOKE_STATUS_FILENAME, status)
    (run_dir / "eval_factory_l4_smoke_execute.log").write_text("successful fixture run\n", encoding="utf-8")

    summary_dir = run_dir / runner.L4_SMOKE_CONFIG.summary_run_id
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
        for index, family in enumerate(runner.FAMILY_SET, start=1):
            writer.writerow(
                {
                    "model_label": runner.L4_SMOKE_CONFIG.model_label,
                    "model_id": runner.L4_SMOKE_CONFIG.model_id,
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


class PackageEvalFactoryReceiptTest(unittest.TestCase):
    def test_successful_packaging_writes_manifest_checksums_and_tarball(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            run_dir = write_fixture_run(repo)
            result = packager.package_receipt(
                run_dir,
                repo / "runs" / runner.RECEIPT_BUNDLES_DIRNAME,
                created_at="2026-04-21T00:00:00Z",
                repo_root=repo,
            )
            validation = runner.validate_operator_receipt_manifest(repo, result.manifest_path)

            self.assertTrue(result.manifest_path.exists())
            self.assertTrue(result.required_checksums_path.exists())
            self.assertTrue(result.bundle_checksums_path.exists())
            self.assertIsNotNone(result.tarball_path)
            self.assertTrue(result.tarball_path.exists())
            self.assertEqual(validation.status, runner.ARTIFACT_STATUS_VALID)
            with tarfile.open(result.tarball_path, "r:gz") as archive:
                names = archive.getnames()

        self.assertIn("eval_factory_l4_smoke_vm_fixture/eval_factory_l4_smoke_status.json", names)

    def test_missing_required_artifact_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            run_dir = write_fixture_run(repo)
            (run_dir / "eval_factory_l4_smoke_execute.log").unlink()

            with self.assertRaises(packager.ReceiptPackagingError) as raised:
                packager.package_receipt(run_dir, repo / "runs" / runner.RECEIPT_BUNDLES_DIRNAME, repo_root=repo)

        self.assertIn("missing required receipt artifact", str(raised.exception))

    def test_malformed_preflight_fails_before_writing_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            run_dir = write_fixture_run(repo)
            payload = dict(runner.read_json(run_dir / runner.L4_SMOKE_PREFLIGHT_FILENAME))
            del payload["schema_id"]
            runner.write_status_artifact(run_dir / runner.L4_SMOKE_PREFLIGHT_FILENAME, payload)

            with self.assertRaises(packager.ReceiptPackagingError) as raised:
                packager.package_receipt(run_dir, repo / "runs" / runner.RECEIPT_BUNDLES_DIRNAME, repo_root=repo)

        self.assertIn("malformed preflight artifact", str(raised.exception))

    def test_malformed_status_fails_before_writing_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            run_dir = write_fixture_run(repo)
            payload = dict(runner.read_json(run_dir / runner.L4_SMOKE_STATUS_FILENAME))
            del payload["downstream_dispatch_summary"]
            runner.write_status_artifact(run_dir / runner.L4_SMOKE_STATUS_FILENAME, payload)

            with self.assertRaises(packager.ReceiptPackagingError) as raised:
                packager.package_receipt(run_dir, repo / "runs" / runner.RECEIPT_BUNDLES_DIRNAME, repo_root=repo)

        self.assertIn("malformed status artifact", str(raised.exception))

    def test_manifest_stays_operational_and_non_claiming(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            run_dir = write_fixture_run(repo)
            result = packager.package_receipt(
                run_dir,
                repo / "runs" / runner.RECEIPT_BUNDLES_DIRNAME,
                create_tarball=False,
                created_at="2026-04-21T00:00:00Z",
                repo_root=repo,
            )
            manifest = runner.read_json(result.manifest_path)

        self.assertEqual(manifest["schema_id"], runner.OPERATOR_RECEIPT_SCHEMA_ID)
        self.assertEqual(manifest["source_class"], runner.SOURCE_OPERATOR_RECEIPT)
        self.assertTrue(manifest["not_a_checkpoint"])
        self.assertTrue(manifest["not_a_memo_claim"])
        self.assertTrue(manifest["no_new_model_execution_in_packaging"])
        self.assertEqual(manifest["posture_classification"], runner.POSTURE_REMOTE_CUDA_READY)
        self.assertEqual(manifest["execute_result"], "pass")
        self.assertEqual(manifest["family_count"], 3)
        self.assertIn("pending_local_read", manifest["runs_first_pass_status_note"])
        self.assertFalse(manifest["tarball"]["present"])

    def test_inspect_only_does_not_write_bundle_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            run_dir = write_fixture_run(repo)
            out_root = repo / "runs" / runner.RECEIPT_BUNDLES_DIRNAME

            result = packager.package_receipt(
                run_dir,
                out_root,
                inspect_only=True,
                created_at="2026-04-21T00:00:00Z",
                repo_root=repo,
            )

            self.assertTrue(result.inspect_only)
            self.assertFalse(result.receipt_root.exists())


if __name__ == "__main__":
    raise SystemExit(unittest.main())
