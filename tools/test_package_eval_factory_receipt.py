#!/usr/bin/env python3
"""Tests for the eval-factory receipt packager."""

from __future__ import annotations

import csv
import io
import shutil
import sys
import tarfile
import tempfile
import unittest
from contextlib import redirect_stdout
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


def write_weekly_fixture_run(repo: Path) -> Path:
    target = runner.l4_weekly_target_for_key("qwen2_5_3b")
    run_dir = repo / "runs" / "eval_factory_l4_weekly_qwen2_5_3b_vm_fixture"
    run_dir.mkdir(parents=True)
    created_at = "2026-04-22T00:00:00Z"
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
        "out_dir": "runs/eval_factory_l4_weekly_qwen2_5_3b_vm_fixture",
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
    (run_dir / packager.WEEKLY_EXECUTE_LOG_FILENAME).write_text("successful weekly fixture run\n", encoding="utf-8")

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


class PackageEvalFactoryReceiptTest(unittest.TestCase):
    def test_default_export_contains_only_selected_files_and_one_manifest(self) -> None:
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
            self.assertIsNone(result.tarball_path)
            self.assertEqual(validation.status, runner.ARTIFACT_STATUS_VALID)
            files = [path for path in result.receipt_root.rglob("*") if path.is_file()]
            self.assertEqual(len(files), 5)
            self.assertFalse(any(path.suffix == ".sha256" for path in files))
            self.assertFalse(packager.parse_args(["--run-dir", str(run_dir)]).tarball)

    def test_full_run_archive_is_opt_in_and_verified_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            run_dir = write_weekly_fixture_run(repo)
            result = packager.package_receipt(
                run_dir,
                repo / "runs" / runner.RECEIPT_BUNDLES_DIRNAME,
                create_tarball=True,
                created_at="2026-04-22T00:00:00Z",
                repo_root=repo,
            )
            validation = runner.validate_operator_receipt_manifest(repo, result.manifest_path)
            manifest = runner.read_json(result.manifest_path)

            self.assertTrue(result.manifest_path.exists())
            self.assertIsNotNone(result.tarball_path)
            self.assertTrue(result.tarball_path.exists())
            self.assertEqual(validation.status, runner.ARTIFACT_STATUS_VALID)
            with tarfile.open(result.tarball_path, "r:gz") as archive:
                names = archive.getnames()
            self.assertFalse(list(result.receipt_root.rglob("*.sha256")))
            original = result.tarball_path.read_bytes()
            result.tarball_path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
            corrupted = runner.validate_operator_receipt_manifest(repo, result.manifest_path)
            self.assertTrue(any("tarball checksum mismatch" in error for error in corrupted.errors))

        self.assertEqual(manifest["schema_id"], runner.OPERATOR_RECEIPT_L4_WEEKLY_SCHEMA_ID)
        self.assertEqual(manifest["source_class"], runner.SOURCE_OPERATOR_WEEKLY_RECEIPT)
        self.assertEqual(manifest["tier"], runner.Tier.L4_WEEKLY.value)
        self.assertEqual(manifest["target"], "qwen2_5_3b")
        self.assertEqual(manifest["fixed_target_set"]["model_id"], "Qwen/Qwen2.5-3B-Instruct")
        self.assertEqual(manifest["machine_side_structural_family_summary"][0]["runs_first_pass_status"], "pending_local_read")
        self.assertIn("eval_factory_l4_weekly_qwen2_5_3b_vm_fixture/eval_factory_l4_weekly_status.json", names)

    def test_missing_required_artifact_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            run_dir = write_fixture_run(repo)
            (run_dir / "eval_factory_l4_smoke_execute.log").unlink()

            with self.assertRaises(packager.ReceiptPackagingError) as raised:
                packager.package_receipt(run_dir, repo / "runs" / runner.RECEIPT_BUNDLES_DIRNAME, repo_root=repo)

        self.assertIn("missing required receipt artifact", str(raised.exception))

    def test_missing_weekly_required_artifact_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            run_dir = write_weekly_fixture_run(repo)
            (run_dir / packager.WEEKLY_EXECUTE_LOG_FILENAME).unlink()

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

    def test_malformed_weekly_preflight_fails_before_writing_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            run_dir = write_weekly_fixture_run(repo)
            payload = dict(runner.read_json(run_dir / runner.L4_WEEKLY_PREFLIGHT_FILENAME))
            del payload["schema_id"]
            runner.write_status_artifact(run_dir / runner.L4_WEEKLY_PREFLIGHT_FILENAME, payload)

            with self.assertRaises(packager.ReceiptPackagingError) as raised:
                packager.package_receipt(run_dir, repo / "runs" / runner.RECEIPT_BUNDLES_DIRNAME, repo_root=repo)

        self.assertIn("malformed preflight artifact", str(raised.exception))

    def test_malformed_weekly_status_fails_before_writing_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            run_dir = write_weekly_fixture_run(repo)
            payload = dict(runner.read_json(run_dir / runner.L4_WEEKLY_STATUS_FILENAME))
            del payload["downstream_dispatch_summary"]
            runner.write_status_artifact(run_dir / runner.L4_WEEKLY_STATUS_FILENAME, payload)

            with self.assertRaises(packager.ReceiptPackagingError) as raised:
                packager.package_receipt(run_dir, repo / "runs" / runner.RECEIPT_BUNDLES_DIRNAME, repo_root=repo)

        self.assertIn("malformed status artifact", str(raised.exception))

    def test_export_preserves_execution_and_measurement_status(self) -> None:
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
        self.assertEqual(manifest["posture_classification"], runner.POSTURE_REMOTE_CUDA_READY)
        self.assertEqual(manifest["execute_result"], "pass")
        self.assertEqual(manifest["family_count"], 3)
        self.assertEqual(manifest["machine_side_structural_family_summary"][0]["runs_first_pass_status"], "pending_local_read")
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

    def test_weekly_receipt_is_reported_as_distinct_readonly_surface(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            run_dir = write_weekly_fixture_run(repo)
            result = packager.package_receipt(
                run_dir,
                repo / "runs" / runner.RECEIPT_BUNDLES_DIRNAME,
                create_tarball=False,
                created_at="2026-04-22T00:00:00Z",
                repo_root=repo,
            )

            text = runner.render_summarize_existing(repo)
            manifest_exists = result.manifest_path.exists()

        self.assertTrue(manifest_exists)
        self.assertIn("operator/eval-factory l4-weekly receipt bundle surfaces:", text)
        self.assertIn("source_class=operator/eval-factory l4-weekly receipt bundle", text)
        self.assertIn("target=qwen2_5_3b", text)

    def test_export_file_corruption_is_detected_without_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            result = packager.package_receipt(write_fixture_run(repo), repo / "exports", repo_root=repo)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(packager.main(["--verify-export", str(result.manifest_path)]), 0)
            copied_log = result.receipt_root / "required_artifacts/eval_factory_l4_smoke_execute.log"
            original = copied_log.read_bytes()
            copied_log.write_bytes(b"X" + original[1:])
            validation = runner.validate_operator_receipt_manifest(repo, result.manifest_path)
            self.assertEqual(validation.status, runner.ARTIFACT_STATUS_MALFORMED)
            self.assertTrue(any("checksum mismatch" in error for error in validation.errors))
            with redirect_stdout(io.StringIO()):
                self.assertEqual(packager.main(["--verify-export", str(result.manifest_path)]), 1)

    def test_legacy_sidecars_remain_readable_and_checked(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            result = packager.package_receipt(write_fixture_run(repo), repo / "exports", repo_root=repo)
            manifest = runner.read_json(result.manifest_path)
            entries = manifest["required_artifacts"]
            # Historical manifests used repository-relative paths.
            for entry in entries:
                entry["bundled_path"] = (result.receipt_root / entry["bundled_path"]).relative_to(repo).as_posix()
            sums = result.receipt_root / "required_receipt_artifacts.sha256"
            sums.write_text("".join(f"{entry['sha256']}  {entry['bundled_path']}\n" for entry in entries), encoding="utf-8")
            bundle_sums = result.receipt_root / "receipt_bundle_files.sha256"
            bundle_sums.write_text(f"{runner.sha256_file(sums)}  {sums.relative_to(repo).as_posix()}\n", encoding="utf-8")
            manifest["checksums"] = {
                "required_artifacts_sha256": sums.relative_to(repo).as_posix(),
                "bundle_files_sha256": bundle_sums.relative_to(repo).as_posix(),
            }
            packager.write_json(result.manifest_path, manifest)
            self.assertEqual(runner.validate_operator_receipt_manifest(repo, result.manifest_path).status, runner.ARTIFACT_STATUS_VALID)
            sums.write_text("corrupted\n", encoding="utf-8")
            self.assertEqual(runner.validate_operator_receipt_manifest(repo, result.manifest_path).status, runner.ARTIFACT_STATUS_MALFORMED)

    def test_export_cannot_recurse_into_its_source_or_overwrite_a_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            run_dir = write_fixture_run(repo)
            with self.assertRaises(packager.ReceiptPackagingError):
                packager.package_receipt(run_dir, run_dir / "exports", create_tarball=True, repo_root=repo)
            result = packager.package_receipt(run_dir, repo / "exports", created_at="2026-04-21T00:00:00Z", repo_root=repo)
            before = result.manifest_path.read_bytes()
            with self.assertRaises(packager.ReceiptPackagingError):
                packager.package_receipt(run_dir, repo / "exports", created_at="2026-04-21T00:00:00Z", repo_root=repo)
            self.assertEqual(result.manifest_path.read_bytes(), before)

    def test_export_can_be_read_after_transfer_without_the_source_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir) / "sender"
            result = packager.package_receipt(write_fixture_run(repo), repo / "exports", create_tarball=True, repo_root=repo)
            receiver = Path(tmpdir) / "receiver"
            copied = receiver / "renamed-export"
            shutil.copytree(result.receipt_root, copied)
            validation = runner.validate_operator_receipt_manifest(receiver, copied / result.manifest_path.name)
            self.assertEqual(validation.status, runner.ARTIFACT_STATUS_VALID, validation.errors)


if __name__ == "__main__":
    raise SystemExit(unittest.main())
