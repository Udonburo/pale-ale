#!/usr/bin/env python3
"""Regression tests for Gate8 semi-closed conflict materialization."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class Gate8SemiclosedConflictMaterializationTests(unittest.TestCase):
    def test_materializer_emits_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            constitution_dir = tmp_dir / "constitution"
            out_dir = tmp_dir / "materialized"
            self._run_scaffold(constitution_dir, samples_per_cell=2)
            self._run_materializer(constitution_dir, out_dir)

            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            sample_index_rows = self._read_jsonl(out_dir / "sample_index.jsonl")
            world_truth_rows = self._read_jsonl(out_dir / "world_truth.jsonl")
            rendering_rows = self._read_jsonl(out_dir / "retrieval_renderings.jsonl")
            target_rows = self._read_jsonl(out_dir / "answer_targets.jsonl")
            benchmark_rows = self._read_jsonl(out_dir / "benchmark_rows.jsonl")

            self.assertEqual(manifest["generation_stage"], "materialized_generation")
            self.assertEqual(manifest["provenance_binding_mode"], "realized_artifacts")
            self.assertEqual(manifest["rendering_family_id"], "archive_v1")
            self.assertEqual(manifest["n_samples_total"], 8)
            self.assertEqual(
                manifest["candidate_granularity_status"], "mixed_candidate_label_granularity_v1"
            )
            self.assertEqual(len(sample_index_rows), 8)
            self.assertEqual(len(world_truth_rows), 4)
            self.assertEqual(len(rendering_rows), 6)
            self.assertEqual(len(target_rows), 8)
            self.assertEqual(len(benchmark_rows), 8)
            self.assertNotEqual(
                manifest["sample_index_sha256"],
                manifest["benchmark_rows_sha256"],
            )
            self.assertEqual(
                manifest["constitution_manifest_sha256"],
                self._sha256(constitution_dir / "manifest.json"),
            )

            direct_bad = next(
                row
                for row in benchmark_rows
                if row["cell_id"] == "direct_contradiction"
                and row["answer_target_type"] == "conflict_following_wrong_answer"
            )
            self.assertGreater(len(direct_bad["label_span_conflict"]), 0)
            self.assertGreater(len(direct_bad["label_span_defect"]), 0)
            self.assertGreater(sum(token["label_token"] for token in direct_bad["label_token"]), 0)

            direct_sample_rows = [
                row for row in sample_index_rows if row["cell_id"] == "direct_contradiction"
            ]
            self.assertEqual(len({row["world_id"] for row in direct_sample_rows}), 1)
            self.assertEqual(len({row["rendering_id"] for row in direct_sample_rows}), 1)
            self.assertEqual({row["rendering_family_id"] for row in direct_sample_rows}, {"archive_v1"})

            noisy_clean = next(row for row in benchmark_rows if row["cell_id"] == "surface_noisy_clean")
            self.assertEqual(noisy_clean["retrieval_conflict_chunk_ids"], [])
            self.assertEqual(noisy_clean["label_span_conflict"], [])
            self.assertEqual(noisy_clean["label_span_defect"], [])

            clean_world_ids = {
                row["world_id"] for row in sample_index_rows if row["cell_id"] == "clean_support"
            }
            noisy_world_ids = {
                row["world_id"] for row in sample_index_rows if row["cell_id"] == "surface_noisy_clean"
            }
            self.assertEqual(clean_world_ids, noisy_world_ids)

    def test_materializer_supports_briefing_rendering_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            constitution_dir = tmp_dir / "constitution"
            out_dir = tmp_dir / "materialized"
            self._run_scaffold(constitution_dir, samples_per_cell=2, rendering_family="briefing_v1")
            self._run_materializer(constitution_dir, out_dir)

            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            rendering_plan = json.loads((out_dir / "rendering_plan.json").read_text(encoding="utf-8"))
            rendering_rows = self._read_jsonl(out_dir / "retrieval_renderings.jsonl")
            benchmark_rows = self._read_jsonl(out_dir / "benchmark_rows.jsonl")

            self.assertEqual(manifest["rendering_family_id"], "briefing_v1")
            self.assertEqual(rendering_plan["rendering_family_id"], "briefing_v1")
            self.assertEqual({row["rendering_family_id"] for row in rendering_rows}, {"briefing_v1"})
            self.assertEqual({row["rendering_family_id"] for row in benchmark_rows}, {"briefing_v1"})
            self.assertTrue(
                str(rendering_rows[0]["prompt"]).startswith("Briefing packets:")
            )
            all_chunk_text = "\n".join(
                str(chunk["text"])
                for row in rendering_rows
                for chunk in row["retrieval_chunks"]
            )
            all_answer_text = "\n".join(str(row["answer_text"]) for row in benchmark_rows)
            self.assertIn("Packet alpha reports:", all_chunk_text)
            self.assertIn("Counter-brief:", all_chunk_text)
            self.assertIn("Given the briefing packets,", all_answer_text)
            self.assertIn("Given the counter-brief,", all_answer_text)

    def test_materializer_supports_transcript_rendering_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            constitution_dir = tmp_dir / "constitution"
            out_dir = tmp_dir / "materialized"
            self._run_scaffold(constitution_dir, samples_per_cell=2, rendering_family="transcript_v1")
            self._run_materializer(constitution_dir, out_dir)

            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            rendering_plan = json.loads((out_dir / "rendering_plan.json").read_text(encoding="utf-8"))
            rendering_rows = self._read_jsonl(out_dir / "retrieval_renderings.jsonl")
            benchmark_rows = self._read_jsonl(out_dir / "benchmark_rows.jsonl")

            self.assertEqual(manifest["rendering_family_id"], "transcript_v1")
            self.assertEqual(rendering_plan["rendering_family_id"], "transcript_v1")
            self.assertEqual({row["rendering_family_id"] for row in rendering_rows}, {"transcript_v1"})
            self.assertEqual({row["rendering_family_id"] for row in benchmark_rows}, {"transcript_v1"})
            self.assertTrue(
                str(rendering_rows[0]["prompt"]).startswith("Transcript excerpts:")
            )
            all_chunk_text = "\n".join(
                str(chunk["text"])
                for row in rendering_rows
                for chunk in row["retrieval_chunks"]
            )
            all_answer_text = "\n".join(str(row["answer_text"]) for row in benchmark_rows)
            self.assertIn("Speaker A:", all_chunk_text)
            self.assertIn("Cross-exam aside:", all_chunk_text)
            self.assertIn("On the transcript record,", all_answer_text)
            self.assertIn("If the cross-exam aside is followed,", all_answer_text)

    def test_materializer_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            constitution_dir = tmp_dir / "constitution"
            out_a = tmp_dir / "out_a"
            out_b = tmp_dir / "out_b"
            self._run_scaffold(constitution_dir, samples_per_cell=2)
            self._run_materializer(constitution_dir, out_a)
            self._run_materializer(constitution_dir, out_b)

            for filename in (
                "manifest.json",
                "world_plan.json",
                "rendering_plan.json",
                "target_plan.json",
                "sample_index.jsonl",
                "world_truth.jsonl",
                "retrieval_renderings.jsonl",
                "answer_targets.jsonl",
                "benchmark_rows.jsonl",
                "checksums.json",
            ):
                self.assertEqual(
                    (out_a / filename).read_text(encoding="utf-8"),
                    (out_b / filename).read_text(encoding="utf-8"),
                )

    def _run_scaffold(
        self,
        out_dir: Path,
        samples_per_cell: int,
        rendering_family: str = "archive_v1",
    ) -> None:
        script = REPO_ROOT / "tools" / "generate_gate8_semiclosed_conflict.py"
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--out-dir",
                str(out_dir),
                "--run-id",
                "gate8_constitution_test",
                "--samples-per-cell",
                str(samples_per_cell),
                "--rendering-family",
                rendering_family,
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def _run_materializer(self, constitution_dir: Path, out_dir: Path) -> None:
        script = REPO_ROOT / "tools" / "materialize_gate8_semiclosed_conflict.py"
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--constitution-dir",
                str(constitution_dir),
                "--out-dir",
                str(out_dir),
                "--run-id",
                "gate8_materialized_test",
                "--seed",
                "11",
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    @staticmethod
    def _sha256(path: Path) -> str:
        import hashlib

        return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
