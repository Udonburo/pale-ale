#!/usr/bin/env python3
"""Regression tests for Gate12B source inspection queue builder."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import build_gate12b_source_inspection_queue as queue


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class Gate12BSourceInspectionQueueTest(unittest.TestCase):
    def test_builds_source_facing_queue_from_candidates_and_text_surface(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gate12b_dir = root / "gate12b"
            text_dir = root / "text_surface"
            out_dir = root / "queue"
            self._write_gate12b_fixture(gate12b_dir)
            self._write_text_surface_fixture(text_dir)

            result = queue.run_gate12b_source_inspection_queue(
                cases=[("fixture_archive", gate12b_dir, text_dir)],
                out_dir=out_dir,
                per_band_limit=1,
            )

            self.assertEqual(result["status"]["case_count"], 1)
            self.assertEqual(result["status"]["queue_row_count"], 2)
            self.assertEqual(result["status"]["flat_queue_count"], 1)
            self.assertEqual(result["status"]["high_tension_queue_count"], 1)
            self.assertEqual(result["status"]["answer_contains_conflict_anchor_text_count"], 1)
            self.assertEqual(result["status"]["answer_contains_support_anchor_text_count"], 1)

            with open(out_dir / queue.DEFAULT_QUEUE_CSV, "r", encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(len(csv_rows), 2)
            self.assertEqual(csv_rows[0]["case_label"], "fixture_archive")
            self.assertIn(csv_rows[0]["candidate_side"], {"flat", "high_tension"})

            jsonl_rows = [
                json.loads(line)
                for line in (out_dir / queue.DEFAULT_QUEUE_JSONL).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual({row["cycle_id"] for row in jsonl_rows}, {"triangle:000001", "triangle:000002"})
            self.assertTrue(any(row["answer_contains_conflict_anchor_text"] for row in jsonl_rows))
            self.assertTrue(any(row["answer_contains_support_anchor_text"] for row in jsonl_rows))

            read_text = (out_dir / queue.DEFAULT_READ).read_text(encoding="utf-8")
            self.assertIn("Gate12B Source Inspection Queue", read_text)
            self.assertIn("Support Anchor", read_text)
            self.assertIn("Conflict Anchor", read_text)

            checksums = json.loads((out_dir / queue.DEFAULT_CHECKSUMS).read_text(encoding="utf-8"))
            self.assertIn(queue.DEFAULT_QUEUE_JSONL, checksums)

    def test_rejects_empty_case_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "at least one"):
                queue.run_gate12b_source_inspection_queue(
                    cases=[],
                    out_dir=Path(tmpdir) / "queue",
                    per_band_limit=1,
                )

    def test_rejects_out_dir_aliasing_gate12b_source_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gate12b_dir = root / "gate12b"
            text_dir = root / "text_surface"
            self._write_gate12b_fixture(gate12b_dir)
            self._write_text_surface_fixture(text_dir)
            original_manifest = json.loads((gate12b_dir / queue.GATE12B_MANIFEST).read_text(encoding="utf-8"))

            with self.assertRaisesRegex(ValueError, "same directory"):
                queue.run_gate12b_source_inspection_queue(
                    cases=[("fixture_archive", gate12b_dir, text_dir)],
                    out_dir=gate12b_dir,
                    per_band_limit=1,
                )

            preserved_manifest = json.loads((gate12b_dir / queue.GATE12B_MANIFEST).read_text(encoding="utf-8"))
            self.assertEqual(preserved_manifest, original_manifest)
            self.assertEqual(preserved_manifest["run_id"], "gate12b_fixture")

    def test_rejects_out_dir_nested_under_text_surface_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gate12b_dir = root / "gate12b"
            text_dir = root / "text_surface"
            nested_out_dir = text_dir / "child_queue"
            self._write_gate12b_fixture(gate12b_dir)
            self._write_text_surface_fixture(text_dir)

            with self.assertRaisesRegex(ValueError, "inside an input artifact"):
                queue.run_gate12b_source_inspection_queue(
                    cases=[("fixture_archive", gate12b_dir, text_dir)],
                    out_dir=nested_out_dir,
                    per_band_limit=1,
                )

            self.assertFalse(nested_out_dir.exists())

    def test_rejects_mismatched_source_gate12a_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gate12b_dir = root / "gate12b"
            text_dir = root / "text_surface"
            out_dir = root / "queue"
            self._write_gate12b_fixture(gate12b_dir)
            self._write_text_surface_fixture(text_dir, source_gate12a_run_id="wrong_gate12a_fixture")

            with self.assertRaisesRegex(ValueError, "source_gate12a_run_id mismatch"):
                queue.run_gate12b_source_inspection_queue(
                    cases=[("fixture_archive", gate12b_dir, text_dir)],
                    out_dir=out_dir,
                    per_band_limit=1,
                )

            self.assertFalse(out_dir.exists())

    def _write_gate12b_fixture(self, gate12b_dir: Path) -> None:
        write_json(
            gate12b_dir / queue.GATE12B_MANIFEST,
            {
                "run_id": "gate12b_fixture",
                "source_gate12a_run_id": "gate12a_fixture",
                "observer_mode_set": "cycle_motif_expansion_v1",
                "top_k": 3,
                "min_observer_support": 3,
                "min_scale_support": 3,
            },
        )
        write_jsonl(
            gate12b_dir / queue.GATE12B_CANDIDATES,
            [
                {
                    "cycle_id": "triangle:000001",
                    "candidate_kind": "flat_observer_scale_stable",
                    "relation_kind_signature": "residual_chord=1|trusted_tree=2",
                    "residual_quantile_band": "flat",
                    "holonomy_residual_fro": 0.1,
                    "residual_percentile": 0.0,
                    "observer_support_count": 3,
                    "scale_support_count": 3,
                    "support_observers": ["all_edges"],
                    "support_scales": ["triangle"],
                    "observer_scope_groups": [],
                },
                {
                    "cycle_id": "triangle:000002",
                    "candidate_kind": "high_tension_observer_scale_stable",
                    "relation_kind_signature": "residual_chord=3",
                    "residual_quantile_band": "high_tension",
                    "holonomy_residual_fro": 2.0,
                    "residual_percentile": 1.0,
                    "observer_support_count": 4,
                    "scale_support_count": 3,
                    "support_observers": ["all_edges"],
                    "support_scales": ["triangle"],
                    "observer_scope_groups": [],
                },
            ],
        )

    def _write_text_surface_fixture(self, text_dir: Path, *, source_gate12a_run_id: str = "gate12a_fixture") -> None:
        write_json(
            text_dir / queue.TEXT_SURFACE_MANIFEST,
            {
                "run_id": "text_surface_fixture",
                "source_gate12a_run_id": source_gate12a_run_id,
            },
        )
        rows = []
        for cycle_id, sample_id, answer, support, conflict in [
            (
                "triangle:000001",
                "sample_001",
                "The support path is warranted.",
                "support path",
                "conflict path",
            ),
            (
                "triangle:000002",
                "sample_002",
                "The conflict path is followed.",
                "support path",
                "conflict path",
            ),
        ]:
            rows.append(
                {
                    "cycle_id": cycle_id,
                    "sample_id": sample_id,
                    "edge_id_path": ["edge:1", "edge:2", "edge:3"],
                    "node_id_path": ["a", "b", "c", "a"],
                    "relation_kind_path": ["residual_chord", "trusted_tree", "trusted_tree"],
                    "anchor_qualified_path": [True, True, False],
                    "compatibility_gap_path_summary": {"min": 0.0, "median": 0.1, "max": 0.2, "mean": 0.1},
                    "prompt_path": "prompt.txt",
                    "answer_path": "answer.txt",
                    "support_anchor_path": "support_anchor.txt",
                    "conflict_anchor_path": "conflict_anchor.txt",
                    "prompt_text": "Question?",
                    "answer_text": answer,
                    "support_anchor_text": support,
                    "conflict_anchor_text": conflict,
                }
            )
        write_jsonl(text_dir / queue.TEXT_SURFACE_JOINED, rows)


if __name__ == "__main__":
    unittest.main()
