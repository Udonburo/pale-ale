#!/usr/bin/env python3
"""Tests for the eval-factory legacy artifact helper."""

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

import manage_eval_factory_artifacts as helper
import run_eval_checks as runner


def make_preflight() -> runner.L4SmokePreflight:
    return runner.L4SmokePreflight(
        sys_executable="python",
        python_version="3.11-test",
        cwd=str(runner.REPO_ROOT),
        platform="Linux-test",
        os_name="Linux",
        torch_importable=True,
        torch_version="2.5.0+cu121",
        torch_cuda_available=True,
        torch_cuda_version="12.1",
        gpu_count=1,
        gpu_names=("NVIDIA L4",),
        nvidia_smi_available=True,
        nvidia_smi_path="/usr/bin/nvidia-smi",
        nvidia_smi_summary=("NVIDIA L4, 23034 MiB, 550.54",),
        nvidia_smi_error="",
        posture_classification=runner.POSTURE_REMOTE_CUDA_READY,
        preflight_ok=True,
        remediation_hints=(),
        errors=(),
    )


def write_fixture_artifacts(root: Path) -> dict[str, Path]:
    valid = root / "current" / runner.L4_SMOKE_PREFLIGHT_FILENAME
    valid.parent.mkdir(parents=True, exist_ok=True)
    runner.write_status_artifact(
        valid,
        runner.build_preflight_artifact_payload(
            make_preflight(),
            "preflight-only",
            created_at="2026-04-20T00:00:00Z",
        ),
    )

    legacy = root / "legacy" / runner.L4_SMOKE_STATUS_FILENAME
    legacy.parent.mkdir(parents=True, exist_ok=True)
    runner.write_status_artifact(
        legacy,
        {
            "tier": "l4-smoke",
            "mode": "execute",
            "returncode": 1,
        },
    )

    malformed = root / "malformed" / runner.L4_WEEKLY_PLAN_FILENAME
    malformed.parent.mkdir(parents=True, exist_ok=True)
    runner.write_status_artifact(
        malformed,
        {
            "schema_id": runner.L4_WEEKLY_PLAN_SCHEMA_ID,
            "schema_version": runner.ARTIFACT_CONTRACT_VERSION,
            "created_at": "2026-04-20T00:00:00Z",
        },
    )

    non_eval = root / "notes.json"
    non_eval.write_text('{"not": "eval-factory"}\n', encoding="utf-8")

    return {
        "valid": valid,
        "legacy": legacy,
        "malformed": malformed,
        "non_eval": non_eval,
    }


class ManageEvalFactoryArtifactsTest(unittest.TestCase):
    def test_dry_run_discovery_and_classification(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = write_fixture_artifacts(root)

            artifacts = helper.inspect_artifacts(root)
            by_path = {artifact.path: artifact for artifact in artifacts}
            non_eval = helper.classify_artifact(root, paths["non_eval"])

        self.assertEqual(len(artifacts), 3)
        self.assertEqual(by_path[paths["valid"]].classification, helper.CLASS_VALID_CURRENT)
        self.assertEqual(by_path[paths["legacy"]].classification, helper.CLASS_LEGACY_UNKNOWN)
        self.assertEqual(by_path[paths["malformed"]].classification, helper.CLASS_MALFORMED_CURRENT)
        self.assertEqual(non_eval.classification, helper.CLASS_NON_EVAL_FACTORY)
        self.assertNotIn(paths["non_eval"], by_path)

    def test_dry_run_cli_does_not_move_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = write_fixture_artifacts(root)
            output = io.StringIO()

            with redirect_stdout(output):
                self.assertEqual(helper.main(["--root", str(root)]), 0)

            text = output.getvalue()

            self.assertTrue(paths["valid"].exists())
            self.assertTrue(paths["legacy"].exists())
            self.assertTrue(paths["malformed"].exists())
            self.assertTrue(paths["non_eval"].exists())

        self.assertIn("root scanned:", text)
        self.assertIn("discovered eval-factory artifacts:", text)
        self.assertIn("classification summary:", text)
        self.assertIn("proposed quarantine actions:", text)
        self.assertIn("dry-run; no files moved", text)
        self.assertIn("valid_current_contract: 1", text)
        self.assertIn("legacy_unknown_schema: 1", text)
        self.assertIn("malformed_current_contract: 1", text)

    def test_quarantine_moves_legacy_and_malformed_but_leaves_valid_and_non_eval(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = write_fixture_artifacts(root)
            output = io.StringIO()

            with redirect_stdout(output):
                self.assertEqual(helper.main(["--root", str(root), "--quarantine"]), 0)

            text = output.getvalue()
            quarantined_legacy = root / helper.DEFAULT_QUARANTINE_DIRNAME / "legacy" / runner.L4_SMOKE_STATUS_FILENAME
            quarantined_malformed = root / helper.DEFAULT_QUARANTINE_DIRNAME / "malformed" / runner.L4_WEEKLY_PLAN_FILENAME

            self.assertTrue(paths["valid"].exists())
            self.assertTrue(paths["non_eval"].exists())
            self.assertFalse(paths["legacy"].exists())
            self.assertFalse(paths["malformed"].exists())
            self.assertTrue(quarantined_legacy.exists())
            self.assertTrue(quarantined_malformed.exists())

        self.assertIn("quarantine actions:", text)
        self.assertIn("moved legacy/eval_factory_l4_smoke_status.json", text)
        self.assertIn("moved malformed/eval_factory_l4_weekly_plan.json", text)
        self.assertIn("result: quarantined", text)

    def test_quarantine_sidecar_manifest_is_optional(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_fixture_artifacts(root)

            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    helper.main(["--root", str(root), "--quarantine", "--write-sidecar-manifest"]),
                    0,
                )

            manifest = root / helper.DEFAULT_QUARANTINE_DIRNAME / "eval_factory_quarantine_manifest.json"
            manifest_exists = manifest.exists()
            payload = runner.read_json(manifest)

        self.assertTrue(manifest_exists)
        self.assertEqual(len(payload["moved"]), 2)


if __name__ == "__main__":
    raise SystemExit(unittest.main())
