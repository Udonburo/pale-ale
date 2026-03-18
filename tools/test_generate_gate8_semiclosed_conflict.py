#!/usr/bin/env python3
"""Regression tests for Gate8 semi-closed conflict scaffold generation."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class Gate8SemiclosedConflictSkeletonTests(unittest.TestCase):
    def test_generator_emits_expected_files_and_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            out_dir = tmp_dir / "gate8_out"
            script = REPO_ROOT / "tools" / "generate_gate8_semiclosed_conflict.py"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--out-dir",
                    str(out_dir),
                    "--run-id",
                    "gate8_test",
                    "--samples-per-cell",
                    "2",
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            conflict_plan = json.loads((out_dir / "conflict_plan.json").read_text(encoding="utf-8"))
            label_contract = json.loads((out_dir / "label_contract.json").read_text(encoding="utf-8"))
            world_plan = json.loads((out_dir / "world_plan.json").read_text(encoding="utf-8"))
            rendering_plan = json.loads((out_dir / "rendering_plan.json").read_text(encoding="utf-8"))
            target_plan = json.loads((out_dir / "target_plan.json").read_text(encoding="utf-8"))
            checksums = json.loads((out_dir / "checksums.json").read_text(encoding="utf-8"))
            sample_rows = (out_dir / "sample_index.jsonl").read_text(encoding="utf-8").strip().splitlines()

            self.assertEqual(manifest["run_id"], "gate8_test")
            self.assertEqual(manifest["generation_stage"], "constitution_scaffold")
            self.assertEqual(manifest["provenance_binding_mode"], "constitution_only_placeholders")
            self.assertNotEqual(manifest["schema_version"], conflict_plan["schema_version"])
            self.assertEqual(manifest["taxonomy_schema_version"], conflict_plan["schema_version"])
            self.assertEqual(manifest["label_contract_version"], label_contract["schema_version"])
            self.assertEqual(manifest["world_plan_schema_version"], world_plan["schema_version"])
            self.assertEqual(manifest["rendering_plan_schema_version"], rendering_plan["schema_version"])
            self.assertEqual(manifest["target_plan_schema_version"], target_plan["schema_version"])
            self.assertEqual(manifest["n_cells_total"], 4)
            self.assertEqual(manifest["n_samples_total"], 8)
            self.assertTrue(manifest["aggregation_ban"])
            self.assertEqual(conflict_plan["samples_per_cell"], 2)
            self.assertEqual(len(conflict_plan["cells"]), 4)
            self.assertEqual(len(label_contract["required_sample_fields"]) >= 8, True)
            self.assertEqual(world_plan["binding_status"], "constitution_only_placeholder")
            self.assertEqual(rendering_plan["binding_status"], "constitution_only_placeholder")
            self.assertEqual(target_plan["binding_status"], "constitution_only_placeholder")
            self.assertEqual(manifest["world_plan_path"], "world_plan.json")
            self.assertEqual(manifest["rendering_plan_path"], "rendering_plan.json")
            self.assertEqual(manifest["target_plan_path"], "target_plan.json")
            self.assertEqual(manifest["world_plan_sha256"], self._sha256(out_dir / "world_plan.json"))
            self.assertEqual(manifest["rendering_plan_sha256"], self._sha256(out_dir / "rendering_plan.json"))
            self.assertEqual(manifest["target_plan_sha256"], self._sha256(out_dir / "target_plan.json"))
            self.assertEqual(checksums["world_plan_json"], self._sha256(out_dir / "world_plan.json"))
            self.assertEqual(checksums["rendering_plan_json"], self._sha256(out_dir / "rendering_plan.json"))
            self.assertEqual(checksums["target_plan_json"], self._sha256(out_dir / "target_plan.json"))
            self.assertEqual(len(sample_rows), 8)

    def test_generator_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            script = REPO_ROOT / "tools" / "generate_gate8_semiclosed_conflict.py"
            out_a = tmp_dir / "out_a"
            out_b = tmp_dir / "out_b"
            for out_dir in (out_a, out_b):
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(script),
                        "--out-dir",
                        str(out_dir),
                        "--run-id",
                        "gate8_test",
                        "--samples-per-cell",
                        "3",
                    ],
                    cwd=str(REPO_ROOT),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

            self.assertEqual(
                (out_a / "manifest.json").read_text(encoding="utf-8"),
                (out_b / "manifest.json").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                (out_a / "conflict_plan.json").read_text(encoding="utf-8"),
                (out_b / "conflict_plan.json").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                (out_a / "label_contract.json").read_text(encoding="utf-8"),
                (out_b / "label_contract.json").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                (out_a / "sample_index.jsonl").read_text(encoding="utf-8"),
                (out_b / "sample_index.jsonl").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                (out_a / "world_plan.json").read_text(encoding="utf-8"),
                (out_b / "world_plan.json").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                (out_a / "rendering_plan.json").read_text(encoding="utf-8"),
                (out_b / "rendering_plan.json").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                (out_a / "target_plan.json").read_text(encoding="utf-8"),
                (out_b / "target_plan.json").read_text(encoding="utf-8"),
            )

    @staticmethod
    def _sha256(path: Path) -> str:
        import hashlib

        return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
