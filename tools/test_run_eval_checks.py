#!/usr/bin/env python3
"""Tests for the evaluation-factory runner."""

from __future__ import annotations

import io
import csv
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import run_eval_checks as runner
import package_eval_factory_receipt as receipt_helper


class FakeCuda:
    def __init__(self, available: bool, count: int = 0, names: tuple[str, ...] = ()) -> None:
        self.available = available
        self.count = count
        self.names = names

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        return self.count

    def get_device_name(self, index: int) -> str:
        return self.names[index]


def fake_torch(available: bool, count: int = 0, names: tuple[str, ...] = ()) -> types.SimpleNamespace:
    cuda_version = "12.1" if available else None
    return types.SimpleNamespace(
        __version__="2.5.0+cu121",
        version=types.SimpleNamespace(cuda=cuda_version),
        cuda=FakeCuda(available, count, names),
    )


def make_preflight(
    posture: str = runner.POSTURE_REMOTE_CUDA_READY,
    ok: bool = True,
    os_name: str = "Linux",
    torch_importable: bool = True,
    cuda_available: bool | None = True,
    gpu_count: int | None = 1,
    gpu_names: tuple[str, ...] = ("NVIDIA L4",),
) -> runner.L4SmokePreflight:
    return runner.L4SmokePreflight(
        sys_executable="python",
        python_version="3.11-test",
        cwd=str(runner.REPO_ROOT),
        platform="test-platform",
        os_name=os_name,
        torch_importable=torch_importable,
        torch_version="2.5.0+cu121" if torch_importable else "unavailable",
        torch_cuda_available=cuda_available,
        torch_cuda_version="12.1" if cuda_available else "unavailable",
        gpu_count=gpu_count,
        gpu_names=gpu_names,
        nvidia_smi_available=ok,
        nvidia_smi_path="/usr/bin/nvidia-smi" if ok else "",
        nvidia_smi_summary=("NVIDIA L4, 23034 MiB, 550.54",) if ok else (),
        nvidia_smi_error="",
        posture_classification=posture,
        preflight_ok=ok,
        remediation_hints=() if ok else ("Run this lane on the GCP L4 VM instead of local Windows.",),
        errors=() if ok else (f"posture classification is {posture}, expected {runner.POSTURE_REMOTE_CUDA_READY}",),
    )


def make_family_results() -> list[dict[str, str]]:
    return [
        {
            "family": family,
            "dispatch": "completed",
            "structural_flags_all_true": "True",
            "runs_first_pass_status": "pending_local_read",
        }
        for family in runner.L4_SMOKE_CONFIG.families
    ]


def make_status_payload() -> dict[str, object]:
    created_at = "2026-04-20T00:00:00Z"
    family_results = make_family_results()
    downstream_summary = runner.build_downstream_dispatch_summary(family_results, [], 0)
    return {
        "schema_id": runner.L4_SMOKE_STATUS_SCHEMA_ID,
        "schema_version": runner.ARTIFACT_CONTRACT_VERSION,
        "created_at": created_at,
        "tier": runner.Tier.L4_SMOKE.value,
        "mode": "execute",
        "fixed_target_set": runner.l4_smoke_fixed_target_set(),
        "model_id": runner.L4_SMOKE_CONFIG.model_id,
        "model_label": runner.L4_SMOKE_CONFIG.model_label,
        "families": list(runner.L4_SMOKE_CONFIG.families),
        "entrypoint": "tools/run_gate12a_cross_model_replay.py",
        "command": ["python", "tools/run_gate12a_cross_model_replay.py"],
        "out_dir": "tmp/l4_smoke",
        "returncode": 0,
        "preflight": runner.build_preflight_artifact_payload(make_preflight(), "execute", created_at=created_at),
        "downstream_dispatch_summary": downstream_summary,
        "result": downstream_summary["result"],
        "family_results": family_results,
        "notes": [],
    }


def make_weekly_status_payload(
    target: runner.L4WeeklyTarget | None = None,
    preflight: runner.L4SmokePreflight | None = None,
) -> dict[str, object]:
    target = target or runner.l4_weekly_target_for_key("qwen2_5_3b")
    created_at = "2026-04-20T00:00:00Z"
    family_results = [
        {
            "family": family,
            "dispatch": "completed",
            "structural_flags_all_true": "True",
            "runs_first_pass_status": "pending_local_read",
        }
        for family in target.families
    ]
    downstream_summary = runner.build_downstream_dispatch_summary(
        family_results,
        [],
        0,
        expected_family_count=len(target.families),
    )
    preflight = preflight or make_preflight()
    return {
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
        "out_dir": "tmp/l4_weekly",
        "returncode": 0,
        "preflight": runner.build_l4_weekly_preflight_artifact_payload(
            preflight,
            target,
            "execute",
            created_at=created_at,
        ),
        "downstream_dispatch_summary": downstream_summary,
        "result": downstream_summary["result"],
        "family_results": family_results,
        "notes": [],
    }


def write_receipt_fixture(repo: Path) -> Path:
    run_dir = repo / "runs" / "eval_factory_l4_smoke_vm_fixture"
    run_dir.mkdir(parents=True)
    runner.write_status_artifact(
        run_dir / runner.L4_SMOKE_PREFLIGHT_FILENAME,
        runner.build_preflight_artifact_payload(
            make_preflight(),
            "execute",
            created_at="2026-04-20T00:00:00Z",
        ),
    )
    runner.write_status_artifact(run_dir / runner.L4_SMOKE_STATUS_FILENAME, make_status_payload())
    (run_dir / "eval_factory_l4_smoke_execute.log").write_text("fixture log\n", encoding="utf-8")
    summary_dir = run_dir / runner.L4_SMOKE_CONFIG.summary_run_id
    summary_dir.mkdir()
    with (summary_dir / runner.CROSS_MODEL_SUMMARY_FILENAME).open("w", encoding="utf-8", newline="") as handle:
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
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for family in runner.FAMILY_SET:
            writer.writerow(
                {
                    "model_label": runner.L4_SMOKE_CONFIG.model_label,
                    "model_id": runner.L4_SMOKE_CONFIG.model_id,
                    "rendering_family": family,
                    "zero_overlap_clear": "True",
                    "all_defined_triangles_anchor_rich": "True",
                    "trusted_tree_gt_residual_chord": "True",
                    "plain_gt_anchor_qualified": "True",
                    "trusted_tree_median": "1.0",
                    "residual_chord_median": "0.8",
                    "anchor_qualified_median": "0.7",
                    "plain_median": "1.1",
                    "extreme_band_first_pass_status": "pending_local_read",
                }
            )
    result = receipt_helper.package_receipt(
        run_dir,
        repo / "runs" / runner.RECEIPT_BUNDLES_DIRNAME,
        create_tarball=False,
        created_at="2026-04-20T01:00:00Z",
        repo_root=repo,
    )
    return result.manifest_path


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
        self.assertIn("Gate12B promotion", plan.out_of_scope)

    def test_l4_weekly_target_matrix_contents(self) -> None:
        matrix = runner.l4_weekly_target_matrix()

        self.assertEqual(
            [target["target"] for target in matrix],
            ["qwen2_5_3b", "llama3_2_3b", "qwen3_4b"],
        )
        self.assertEqual(
            [target["model_id"] for target in matrix],
            [
                "Qwen/Qwen2.5-3B-Instruct",
                "meta-llama/Llama-3.2-3B-Instruct",
                "Qwen/Qwen3-4B",
            ],
        )
        for target in matrix:
            self.assertEqual(target["families"], ["transcript_v1", "briefing_v1", "archive_v1"])

    def test_l4_weekly_prints_plan_compiler_sections(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            self.assertEqual(runner.main(["--tier", "l4-weekly"]), 0)

        text = output.getvalue()
        self.assertIn("tier: l4-weekly", text)
        self.assertIn("mode: plan-only", text)
        self.assertIn("weekly target matrix:", text)
        self.assertIn("per-model planned families:", text)
        self.assertIn("planned entrypoints:", text)
        self.assertIn("tools/run_gate12a_cross_model_replay.py", text)
        self.assertIn("exclusions:", text)
        self.assertIn("7B FP32", text)
        self.assertIn("protocol-expanding candidates", text)
        self.assertIn("quantized candidates", text)
        self.assertIn("sidecar candidates", text)
        self.assertIn("Gate12B promotion", text)
        self.assertIn("no subprocess execution", text)
        self.assertIn("result: plan-only", text)

    def test_l4_weekly_writes_valid_optional_plan_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "weekly"
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(runner.main(["--tier", "l4-weekly", "--out-dir", str(out_dir)]), 0)

            artifact = out_dir / runner.L4_WEEKLY_PLAN_FILENAME
            artifact_exists = artifact.exists()
            payload = runner.read_json(artifact)
            validation = runner.validate_l4_weekly_plan_artifact(runner.REPO_ROOT, artifact)

        self.assertTrue(artifact_exists)
        self.assertEqual(payload["schema_id"], runner.L4_WEEKLY_PLAN_SCHEMA_ID)
        self.assertEqual(payload["schema_version"], runner.ARTIFACT_CONTRACT_VERSION)
        self.assertEqual(payload["tier"], runner.Tier.L4_WEEKLY.value)
        self.assertEqual(payload["mode"], "plan-only")
        self.assertEqual(payload["weekly_target_matrix"], runner.l4_weekly_target_matrix())
        self.assertEqual(payload["exclusions"], list(runner.L4_WEEKLY_EXCLUSIONS))
        self.assertEqual(payload["result"], "plan-only")
        self.assertEqual(validation.status, runner.ARTIFACT_STATUS_VALID)

    def test_l4_smoke_remains_plan_only(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            self.assertEqual(runner.main(["--tier", "l4-smoke"]), 0)

        text = output.getvalue()
        self.assertIn("tier: l4-smoke", text)
        self.assertIn("mode: dry-run", text)
        self.assertIn("actual entrypoints selected:", text)
        self.assertIn("tools/run_gate12a_cross_model_replay.py", text)
        self.assertIn("not executed; pass --preflight-only", text)
        self.assertIn("pass --execute --out-dir <path>", text)
        self.assertIn("0.5B fixed family boundary set", text)

    def test_execute_preflight_and_target_preconditions(self) -> None:
        missing_out = io.StringIO()
        with redirect_stdout(missing_out):
            self.assertEqual(runner.main(["--tier", "l4-smoke", "--execute"]), 2)
        self.assertIn("--out-dir is required", missing_out.getvalue())

        wrong_tier = io.StringIO()
        with redirect_stdout(wrong_tier):
            self.assertEqual(runner.main(["--tier", "cpu-nightly", "--execute", "--out-dir", "tmp/out"]), 2)
        self.assertIn("--execute is only supported for --tier l4-smoke or l4-weekly", wrong_tier.getvalue())

        wrong_preflight_tier = io.StringIO()
        with redirect_stdout(wrong_preflight_tier):
            self.assertEqual(runner.main(["--tier", "cpu-nightly", "--preflight-only"]), 2)
        self.assertIn("--preflight-only is only supported for --tier l4-smoke or l4-weekly", wrong_preflight_tier.getvalue())

        target_on_wrong_tier = io.StringIO()
        with redirect_stdout(target_on_wrong_tier):
            self.assertEqual(runner.main(["--tier", "l4-smoke", "--target", "qwen2_5_3b"]), 2)
        self.assertIn("--target is only supported for --tier l4-weekly", target_on_wrong_tier.getvalue())

        weekly_missing_target = io.StringIO()
        with redirect_stdout(weekly_missing_target):
            self.assertEqual(runner.main(["--tier", "l4-weekly", "--execute", "--out-dir", "tmp/out"]), 2)
        self.assertIn("--target is required", weekly_missing_target.getvalue())

        weekly_missing_out = io.StringIO()
        with redirect_stdout(weekly_missing_out):
            self.assertEqual(runner.main(["--tier", "l4-weekly", "--execute", "--target", "qwen2_5_3b"]), 2)
        self.assertIn("--out-dir is required for --tier l4-weekly --execute", weekly_missing_out.getvalue())

        conflicting_modes = io.StringIO()
        with redirect_stdout(conflicting_modes):
            self.assertEqual(
                runner.main(["--tier", "l4-smoke", "--preflight-only", "--execute", "--out-dir", "tmp/out"]),
                2,
            )
        self.assertIn("--execute and --preflight-only cannot be used together", conflicting_modes.getvalue())

    def test_l4_smoke_command_uses_committed_cross_model_entrypoint(self) -> None:
        command = runner.build_l4_smoke_command(runner.REPO_ROOT, Path("tmp/l4-smoke"))

        self.assertIn("run_gate12a_cross_model_replay.py", command[1])
        self.assertIn("--model-id", command)
        self.assertIn("Qwen/Qwen2.5-0.5B", command)
        self.assertIn("--families", command)
        families_index = command.index("--families") + 1
        self.assertEqual(command[families_index : families_index + 3], ["transcript_v1", "briefing_v1", "archive_v1"])
        self.assertIn("--device", command)
        self.assertIn("cuda", command)

    def test_l4_weekly_target_enum_routing_and_command(self) -> None:
        target = runner.l4_weekly_target_for_key("qwen3_4b")
        command = runner.build_l4_weekly_command(runner.REPO_ROOT, Path("tmp/l4-weekly"), target)

        self.assertEqual(target.model_id, "Qwen/Qwen3-4B")
        self.assertIn("run_gate12a_cross_model_replay.py", command[1])
        self.assertIn("--model-id", command)
        self.assertIn("Qwen/Qwen3-4B", command)
        self.assertIn("--families", command)
        families_index = command.index("--families") + 1
        self.assertEqual(command[families_index : families_index + 3], ["transcript_v1", "briefing_v1", "archive_v1"])
        self.assertIn("--summary-run-id", command)
        self.assertIn("gate12a_cross_model_replay_qwen_qwen3_4b", command)
        self.assertIn("--device", command)
        self.assertIn("cuda", command)

    def test_l4_smoke_execute_uses_fake_subprocess_and_reports_families(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "l4_smoke"

            def fake_run(command, cwd, capture_output, text, check):
                summary_dir = out_dir / runner.L4_SMOKE_CONFIG.summary_run_id
                summary_dir.mkdir(parents=True)
                summary_path = summary_dir / runner.CROSS_MODEL_SUMMARY_FILENAME
                with summary_path.open("w", encoding="utf-8", newline="") as handle:
                    fieldnames = [
                        "model_id",
                        "rendering_family",
                        *runner.STRUCTURAL_FLAG_COLUMNS,
                        "extreme_band_first_pass_status",
                    ]
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    for family in runner.L4_SMOKE_CONFIG.families:
                        writer.writerow(
                            {
                                "model_id": runner.L4_SMOKE_CONFIG.model_id,
                                "rendering_family": family,
                                "zero_overlap_clear": "True",
                                "all_defined_triangles_anchor_rich": "True",
                                "trusted_tree_gt_residual_chord": "True",
                                "plain_gt_anchor_qualified": "True",
                                "extreme_band_first_pass_status": "pending_local_read",
                            }
                        )
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    runner.run_l4_smoke_execute(
                        runner.REPO_ROOT,
                        out_dir,
                        run_command=fake_run,
                        preflight_provider=lambda repo: make_preflight(),
                    ),
                    0,
                )

            text = output.getvalue()
            self.assertIn("mode: execute", text)
            self.assertIn("model: Qwen/Qwen2.5-0.5B", text)
            self.assertIn("classification: remote_cuda_ready", text)
            self.assertIn("per-family dispatch/result summary:", text)
            self.assertIn("transcript_v1: dispatch=completed; structural_flags_all_true=True", text)
            self.assertIn("result: pass", text)
            self.assertTrue((out_dir / runner.L4_SMOKE_PREFLIGHT_FILENAME).exists())
            self.assertTrue((out_dir / runner.L4_SMOKE_STATUS_FILENAME).exists())
            preflight_validation = runner.validate_eval_factory_preflight_artifact(
                runner.REPO_ROOT,
                out_dir / runner.L4_SMOKE_PREFLIGHT_FILENAME,
            )
            status_validation = runner.validate_eval_factory_status_artifact(
                runner.REPO_ROOT,
                out_dir / runner.L4_SMOKE_STATUS_FILENAME,
            )
            self.assertEqual(preflight_validation.status, runner.ARTIFACT_STATUS_VALID)
            self.assertEqual(status_validation.status, runner.ARTIFACT_STATUS_VALID)

    def test_l4_weekly_execute_uses_fake_subprocess_for_one_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "l4_weekly"
            target = runner.l4_weekly_target_for_key("llama3_2_3b")

            def fake_run(command, cwd, capture_output, text, check):
                summary_dir = out_dir / target.summary_run_id
                summary_dir.mkdir(parents=True)
                summary_path = summary_dir / runner.CROSS_MODEL_SUMMARY_FILENAME
                with summary_path.open("w", encoding="utf-8", newline="") as handle:
                    fieldnames = [
                        "model_id",
                        "rendering_family",
                        *runner.STRUCTURAL_FLAG_COLUMNS,
                        "extreme_band_first_pass_status",
                    ]
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    for family in target.families:
                        writer.writerow(
                            {
                                "model_id": target.model_id,
                                "rendering_family": family,
                                "zero_overlap_clear": "True",
                                "all_defined_triangles_anchor_rich": "True",
                                "trusted_tree_gt_residual_chord": "True",
                                "plain_gt_anchor_qualified": "True",
                                "extreme_band_first_pass_status": "pending_local_read",
                            }
                        )
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    runner.run_l4_weekly_execute(
                        runner.REPO_ROOT,
                        target,
                        out_dir,
                        run_command=fake_run,
                        preflight_provider=lambda repo: make_preflight(),
                    ),
                    0,
                )

            text = output.getvalue()
            self.assertIn("tier: l4-weekly", text)
            self.assertIn("mode: execute", text)
            self.assertIn("target: llama3_2_3b", text)
            self.assertIn("model: meta-llama/Llama-3.2-3B-Instruct", text)
            self.assertIn("classification: remote_cuda_ready", text)
            self.assertIn("per-family dispatch/result summary:", text)
            self.assertIn("archive_v1: dispatch=completed; structural_flags_all_true=True", text)
            self.assertIn("artifact paths:", text)
            self.assertTrue((out_dir / runner.L4_WEEKLY_PREFLIGHT_FILENAME).exists())
            self.assertTrue((out_dir / runner.L4_WEEKLY_STATUS_FILENAME).exists())
            preflight_validation = runner.validate_eval_factory_weekly_preflight_artifact(
                runner.REPO_ROOT,
                out_dir / runner.L4_WEEKLY_PREFLIGHT_FILENAME,
            )
            status_validation = runner.validate_eval_factory_weekly_status_artifact(
                runner.REPO_ROOT,
                out_dir / runner.L4_WEEKLY_STATUS_FILENAME,
            )
            self.assertEqual(preflight_validation.status, runner.ARTIFACT_STATUS_VALID)
            self.assertEqual(status_validation.status, runner.ARTIFACT_STATUS_VALID)
            self.assertEqual(status_validation.target, "llama3_2_3b")

    def test_preflight_classifies_local_windows_no_cuda(self) -> None:
        preflight = runner.collect_l4_smoke_preflight(
            runner.REPO_ROOT,
            torch_loader=lambda: fake_torch(False),
            platform_system=lambda: "Windows",
            platform_string=lambda: "Windows-10-test",
            cwd_getter=lambda: runner.REPO_ROOT,
            which=lambda name: None,
        )

        self.assertEqual(preflight.posture_classification, runner.POSTURE_LOCAL_WINDOWS_NO_CUDA)
        self.assertFalse(preflight.preflight_ok)
        self.assertIn("Run this lane on the GCP L4 VM instead of local Windows.", preflight.remediation_hints)

    def test_preflight_classifies_missing_torch(self) -> None:
        def missing_torch():
            raise ImportError("No module named torch")

        preflight = runner.collect_l4_smoke_preflight(
            runner.REPO_ROOT,
            torch_loader=missing_torch,
            platform_system=lambda: "Linux",
            platform_string=lambda: "Linux-test",
            cwd_getter=lambda: runner.REPO_ROOT,
            which=lambda name: None,
        )

        self.assertEqual(preflight.posture_classification, runner.POSTURE_PYTHON_MISSING_TORCH)
        self.assertFalse(preflight.torch_importable)
        self.assertFalse(preflight.preflight_ok)
        self.assertIn("torch import failed", "\n".join(preflight.errors))

    def test_preflight_classifies_cuda_ready_with_nvidia_smi_summary(self) -> None:
        def fake_smi(command, capture_output, text, check, timeout):
            return subprocess.CompletedProcess(command, 0, stdout="NVIDIA L4, 23034 MiB, 550.54\n", stderr="")

        preflight = runner.collect_l4_smoke_preflight(
            runner.REPO_ROOT,
            torch_loader=lambda: fake_torch(True, 1, ("NVIDIA L4",)),
            platform_system=lambda: "Linux",
            platform_string=lambda: "Linux-test",
            cwd_getter=lambda: runner.REPO_ROOT,
            which=lambda name: "/usr/bin/nvidia-smi",
            run_command=fake_smi,
        )

        self.assertEqual(preflight.posture_classification, runner.POSTURE_REMOTE_CUDA_READY)
        self.assertTrue(preflight.preflight_ok)
        self.assertEqual(preflight.gpu_names, ("NVIDIA L4",))
        self.assertEqual(preflight.nvidia_smi_summary, ("NVIDIA L4, 23034 MiB, 550.54",))

    def test_execute_blocks_on_failed_preflight_without_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "l4_smoke"

            def fake_run(*args, **kwargs):
                raise AssertionError("subprocess should not be invoked when preflight fails")

            failed_preflight = make_preflight(
                posture=runner.POSTURE_LOCAL_WINDOWS_NO_CUDA,
                ok=False,
                os_name="Windows",
                cuda_available=False,
                gpu_count=0,
                gpu_names=(),
            )
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = runner.run_l4_smoke_execute(
                    runner.REPO_ROOT,
                    out_dir,
                    run_command=fake_run,
                    preflight_provider=lambda repo: failed_preflight,
                )

            text = output.getvalue()
            self.assertEqual(exit_code, 1)
            self.assertIn("classification: local_windows_no_cuda", text)
            self.assertIn("downstream subprocess: not invoked", text)
            self.assertIn("Run this lane on the GCP L4 VM instead of local Windows.", text)
            self.assertTrue((out_dir / runner.L4_SMOKE_PREFLIGHT_FILENAME).exists())
            self.assertFalse((out_dir / runner.L4_SMOKE_STATUS_FILENAME).exists())

    def test_l4_weekly_execute_blocks_on_failed_preflight_without_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "l4_weekly"
            target = runner.l4_weekly_target_for_key("qwen2_5_3b")

            def fake_run(*args, **kwargs):
                raise AssertionError("subprocess should not be invoked when weekly preflight fails")

            failed_preflight = make_preflight(
                posture=runner.POSTURE_LOCAL_WINDOWS_NO_CUDA,
                ok=False,
                os_name="Windows",
                cuda_available=False,
                gpu_count=0,
                gpu_names=(),
            )
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = runner.run_l4_weekly_execute(
                    runner.REPO_ROOT,
                    target,
                    out_dir,
                    run_command=fake_run,
                    preflight_provider=lambda repo: failed_preflight,
                )

            text = output.getvalue()
            self.assertEqual(exit_code, 1)
            self.assertIn("tier: l4-weekly", text)
            self.assertIn("target: qwen2_5_3b", text)
            self.assertIn("classification: local_windows_no_cuda", text)
            self.assertIn("downstream subprocess: not invoked", text)
            self.assertTrue((out_dir / runner.L4_WEEKLY_PREFLIGHT_FILENAME).exists())
            self.assertFalse((out_dir / runner.L4_WEEKLY_STATUS_FILENAME).exists())

    def test_preflight_only_writes_artifact_and_stable_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "preflight"
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = runner.run_l4_smoke_preflight_only(
                    runner.REPO_ROOT,
                    out_dir,
                    preflight_provider=lambda repo: make_preflight(),
                )

            text = output.getvalue()
            artifact = out_dir / runner.L4_SMOKE_PREFLIGHT_FILENAME
            payload = runner.read_json(artifact)

        self.assertEqual(exit_code, 0)
        self.assertIn("mode: preflight-only", text)
        self.assertIn("environment diagnostics:", text)
        self.assertIn("posture classification:", text)
        self.assertIn("preflight result:", text)
        self.assertIn("classification: remote_cuda_ready", text)
        self.assertEqual(payload["posture_classification"], runner.POSTURE_REMOTE_CUDA_READY)
        self.assertTrue(payload["preflight_ok"])
        self.assertEqual(payload["schema_id"], runner.L4_SMOKE_PREFLIGHT_SCHEMA_ID)
        self.assertEqual(payload["schema_version"], runner.ARTIFACT_CONTRACT_VERSION)
        self.assertEqual(payload["tier"], runner.Tier.L4_SMOKE.value)
        self.assertEqual(payload["fixed_target_set"], runner.l4_smoke_fixed_target_set())

    def test_l4_weekly_preflight_only_writes_artifact_and_stable_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "weekly_preflight"
            target = runner.l4_weekly_target_for_key("qwen3_4b")
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = runner.run_l4_weekly_preflight_only(
                    runner.REPO_ROOT,
                    target,
                    out_dir,
                    preflight_provider=lambda repo: make_preflight(),
                )

            text = output.getvalue()
            artifact = out_dir / runner.L4_WEEKLY_PREFLIGHT_FILENAME
            payload = runner.read_json(artifact)

        self.assertEqual(exit_code, 0)
        self.assertIn("mode: preflight-only", text)
        self.assertIn("target: qwen3_4b", text)
        self.assertIn("environment diagnostics:", text)
        self.assertIn("posture classification:", text)
        self.assertIn("artifact paths:", text)
        self.assertEqual(payload["schema_id"], runner.L4_WEEKLY_PREFLIGHT_SCHEMA_ID)
        self.assertEqual(payload["tier"], runner.Tier.L4_WEEKLY.value)
        self.assertEqual(payload["target"], "qwen3_4b")
        self.assertEqual(payload["fixed_target_set"], runner.l4_weekly_fixed_target_set(target))

    def test_valid_preflight_artifact_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            artifact = repo / runner.L4_SMOKE_PREFLIGHT_FILENAME
            runner.write_status_artifact(
                artifact,
                runner.build_preflight_artifact_payload(
                    make_preflight(),
                    "preflight-only",
                    created_at="2026-04-20T00:00:00Z",
                ),
            )

            validation = runner.validate_eval_factory_preflight_artifact(repo, artifact)

        self.assertEqual(validation.source_class, runner.SOURCE_EVAL_FACTORY_PREFLIGHT)
        self.assertEqual(validation.status, runner.ARTIFACT_STATUS_VALID)
        self.assertEqual(validation.mode, "preflight-only")
        self.assertEqual(validation.result, "pass")
        self.assertEqual(validation.posture_classification, runner.POSTURE_REMOTE_CUDA_READY)
        self.assertEqual(validation.errors, ())

    def test_malformed_preflight_artifact_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            artifact = repo / runner.L4_SMOKE_PREFLIGHT_FILENAME
            payload = runner.build_preflight_artifact_payload(
                make_preflight(),
                "preflight-only",
                created_at="2026-04-20T00:00:00Z",
            )
            del payload["schema_id"]
            runner.write_status_artifact(artifact, payload)

            validation = runner.validate_eval_factory_preflight_artifact(repo, artifact)

        self.assertEqual(validation.status, runner.ARTIFACT_STATUS_MALFORMED)
        self.assertIn("missing required field: schema_id", validation.errors)

    def test_valid_status_artifact_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            artifact = repo / runner.L4_SMOKE_STATUS_FILENAME
            runner.write_status_artifact(artifact, make_status_payload())

            validation = runner.validate_eval_factory_status_artifact(repo, artifact)

        self.assertEqual(validation.source_class, runner.SOURCE_EVAL_FACTORY_STATUS)
        self.assertEqual(validation.status, runner.ARTIFACT_STATUS_VALID)
        self.assertEqual(validation.mode, "execute")
        self.assertEqual(validation.result, "pass")
        self.assertEqual(validation.downstream_result, "pass")
        self.assertEqual(validation.posture_classification, runner.POSTURE_REMOTE_CUDA_READY)

    def test_malformed_status_artifact_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            artifact = repo / runner.L4_SMOKE_STATUS_FILENAME
            payload = make_status_payload()
            del payload["downstream_dispatch_summary"]
            runner.write_status_artifact(artifact, payload)

            validation = runner.validate_eval_factory_status_artifact(repo, artifact)

        self.assertEqual(validation.status, runner.ARTIFACT_STATUS_MALFORMED)
        self.assertIn("missing required field: downstream_dispatch_summary", validation.errors)

    def test_valid_l4_weekly_artifact_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            target = runner.l4_weekly_target_for_key("qwen2_5_3b")
            preflight_artifact = repo / runner.L4_WEEKLY_PREFLIGHT_FILENAME
            status_artifact = repo / runner.L4_WEEKLY_STATUS_FILENAME
            runner.write_status_artifact(
                preflight_artifact,
                runner.build_l4_weekly_preflight_artifact_payload(
                    make_preflight(),
                    target,
                    "preflight-only",
                    created_at="2026-04-20T00:00:00Z",
                ),
            )
            runner.write_status_artifact(status_artifact, make_weekly_status_payload(target))

            preflight_validation = runner.validate_eval_factory_weekly_preflight_artifact(repo, preflight_artifact)
            status_validation = runner.validate_eval_factory_weekly_status_artifact(repo, status_artifact)

        self.assertEqual(preflight_validation.source_class, runner.SOURCE_EVAL_FACTORY_WEEKLY_PREFLIGHT)
        self.assertEqual(preflight_validation.status, runner.ARTIFACT_STATUS_VALID)
        self.assertEqual(preflight_validation.target, "qwen2_5_3b")
        self.assertEqual(status_validation.source_class, runner.SOURCE_EVAL_FACTORY_WEEKLY_STATUS)
        self.assertEqual(status_validation.status, runner.ARTIFACT_STATUS_VALID)
        self.assertEqual(status_validation.target, "qwen2_5_3b")

    def test_malformed_l4_weekly_status_artifact_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            artifact = repo / runner.L4_WEEKLY_STATUS_FILENAME
            payload = make_weekly_status_payload()
            payload["target"] = "unsupported"
            runner.write_status_artifact(artifact, payload)

            validation = runner.validate_eval_factory_weekly_status_artifact(repo, artifact)

        self.assertEqual(validation.status, runner.ARTIFACT_STATUS_MALFORMED)
        self.assertIn("field target expected one of", "\n".join(validation.errors))

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

    def test_summarize_existing_separates_eval_factory_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            artifact_dir = repo / "local_artifacts"
            artifact_dir.mkdir()
            target = runner.l4_weekly_target_for_key("qwen2_5_3b")
            runner.write_status_artifact(
                artifact_dir / runner.L4_SMOKE_PREFLIGHT_FILENAME,
                runner.build_preflight_artifact_payload(
                    make_preflight(),
                    "preflight-only",
                    created_at="2026-04-20T00:00:00Z",
                ),
            )
            runner.write_status_artifact(artifact_dir / runner.L4_SMOKE_STATUS_FILENAME, make_status_payload())
            runner.write_status_artifact(
                artifact_dir / runner.L4_WEEKLY_PREFLIGHT_FILENAME,
                runner.build_l4_weekly_preflight_artifact_payload(
                    make_preflight(),
                    target,
                    "preflight-only",
                    created_at="2026-04-20T00:00:00Z",
                ),
            )
            runner.write_status_artifact(artifact_dir / runner.L4_WEEKLY_STATUS_FILENAME, make_weekly_status_payload(target))

            text = runner.render_summarize_existing(repo)

        self.assertIn("tracked memo model surfaces:", text)
        self.assertIn("runs-derived materialized cross-model summaries:", text)
        self.assertIn("eval-factory preflight artifact surfaces:", text)
        self.assertIn("source_class=eval-factory preflight artifact", text)
        self.assertIn("artifact_status=valid", text)
        self.assertIn("eval-factory execute/status artifact surfaces:", text)
        self.assertIn("source_class=eval-factory execute/status artifact", text)
        self.assertIn("downstream_result=pass", text)
        self.assertIn("eval-factory l4-weekly preflight artifact surfaces:", text)
        self.assertIn("source_class=eval-factory l4-weekly preflight artifact", text)
        self.assertIn("eval-factory l4-weekly execute/status artifact surfaces:", text)
        self.assertIn("source_class=eval-factory l4-weekly execute/status artifact", text)
        self.assertIn("target=qwen2_5_3b", text)

    def test_summarize_existing_separates_operator_receipt_bundles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            manifest_path = write_receipt_fixture(repo)

            text = runner.render_summarize_existing(repo)
            manifest_exists = manifest_path.exists()

        self.assertTrue(manifest_exists)
        self.assertIn("operator/eval-factory receipt bundle surfaces:", text)
        self.assertIn("source_class=operator/eval-factory receipt bundle", text)
        self.assertIn("artifact_status=valid", text)
        self.assertIn("result=pass", text)
        self.assertIn("posture=remote_cuda_ready", text)
        self.assertIn("family_count=3", text)
        self.assertIn("tarball=absent", text)
        self.assertIn("checksums=present", text)

    def test_cpu_nightly_reports_missing_required_files_as_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checks = runner.build_cpu_nightly_checks(Path(tmpdir))

        self.assertTrue(any(check.level == runner.LEVEL_FAIL for check in checks))

    def test_cpu_nightly_warns_when_optional_artifacts_are_missing(self) -> None:
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

        self.assertTrue(
            any(
                check.level == runner.LEVEL_WARN and check.label == runner.SOURCE_EVAL_FACTORY_PREFLIGHT
                for check in checks
            )
        )
        self.assertTrue(
            any(
                check.level == runner.LEVEL_WARN and check.label == runner.SOURCE_EVAL_FACTORY_STATUS
                for check in checks
            )
        )
        self.assertTrue(
            any(
                check.level == runner.LEVEL_WARN and check.label == runner.SOURCE_OPERATOR_RECEIPT
                for check in checks
            )
        )
        self.assertTrue(
            any(
                check.level == runner.LEVEL_WARN and check.label == runner.SOURCE_EVAL_FACTORY_WEEKLY_PREFLIGHT
                for check in checks
            )
        )
        self.assertTrue(
            any(
                check.level == runner.LEVEL_WARN and check.label == runner.SOURCE_EVAL_FACTORY_WEEKLY_STATUS
                for check in checks
            )
        )
        self.assertFalse(any(check.level == runner.LEVEL_FAIL for check in checks))

    def test_cpu_nightly_fails_on_malformed_present_artifact(self) -> None:
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
            artifact = repo / "local_artifacts" / runner.L4_SMOKE_PREFLIGHT_FILENAME
            artifact.parent.mkdir()
            runner.write_status_artifact(artifact, {"schema_id": "wrong"})

            checks = runner.build_cpu_nightly_checks(repo)

        self.assertTrue(
            any(
                check.level == runner.LEVEL_FAIL and runner.SOURCE_EVAL_FACTORY_PREFLIGHT in check.label
                for check in checks
            )
        )

    def test_cpu_nightly_validates_l4_weekly_artifacts(self) -> None:
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
            target = runner.l4_weekly_target_for_key("qwen3_4b")
            artifact_dir = repo / "weekly"
            artifact_dir.mkdir()
            runner.write_status_artifact(
                artifact_dir / runner.L4_WEEKLY_PREFLIGHT_FILENAME,
                runner.build_l4_weekly_preflight_artifact_payload(
                    make_preflight(),
                    target,
                    "preflight-only",
                    created_at="2026-04-20T00:00:00Z",
                ),
            )
            runner.write_status_artifact(artifact_dir / runner.L4_WEEKLY_STATUS_FILENAME, make_weekly_status_payload(target))

            checks = runner.build_cpu_nightly_checks(repo)

        self.assertTrue(
            any(
                check.level == runner.LEVEL_PASS and runner.SOURCE_EVAL_FACTORY_WEEKLY_PREFLIGHT in check.label
                for check in checks
            )
        )
        self.assertTrue(
            any(
                check.level == runner.LEVEL_PASS and runner.SOURCE_EVAL_FACTORY_WEEKLY_STATUS in check.label
                for check in checks
            )
        )
        self.assertFalse(any(check.level == runner.LEVEL_FAIL for check in checks))

    def test_cpu_nightly_fails_on_malformed_l4_weekly_artifact(self) -> None:
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
            artifact = repo / "weekly" / runner.L4_WEEKLY_STATUS_FILENAME
            artifact.parent.mkdir()
            runner.write_status_artifact(artifact, {"schema_id": "wrong"})

            checks = runner.build_cpu_nightly_checks(repo)

        self.assertTrue(
            any(
                check.level == runner.LEVEL_FAIL and runner.SOURCE_EVAL_FACTORY_WEEKLY_STATUS in check.label
                for check in checks
            )
        )

    def test_cpu_nightly_validates_operator_receipt_bundles(self) -> None:
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
            write_receipt_fixture(repo)

            checks = runner.build_cpu_nightly_checks(repo)

        self.assertTrue(
            any(
                check.level == runner.LEVEL_PASS and runner.SOURCE_OPERATOR_RECEIPT in check.label
                for check in checks
            )
        )
        self.assertFalse(any(check.level == runner.LEVEL_FAIL for check in checks))

    def test_cpu_nightly_fails_on_malformed_receipt_bundle(self) -> None:
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
            manifest = repo / "runs" / runner.RECEIPT_BUNDLES_DIRNAME / "bad" / runner.RECEIPT_MANIFEST_FILENAME
            manifest.parent.mkdir(parents=True)
            runner.write_status_artifact(manifest, {"schema_id": "wrong"})

            checks = runner.build_cpu_nightly_checks(repo)

        self.assertTrue(
            any(
                check.level == runner.LEVEL_FAIL and runner.SOURCE_OPERATOR_RECEIPT in check.label
                for check in checks
            )
        )

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
