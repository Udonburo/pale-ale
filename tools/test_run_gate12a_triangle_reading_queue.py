#!/usr/bin/env python3
"""Regression tests for Gate12A triangle reading queue helper."""

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

import run_gate12a_triangle_reading_queue as reading_queue


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class Gate12ATriangleReadingQueueTest(unittest.TestCase):
    def test_run_triangle_reading_queue_prioritizes_high_tension_then_flat(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            phenotype_dir = self._build_phenotype_prep_fixture(root)
            out_dir = root / "reading_queue"

            result = reading_queue.run_triangle_reading_queue(
                phenotype_prep_dir=phenotype_dir,
                out_dir=out_dir,
            )

            self.assertEqual(result["status"]["queue_row_count"], 4)
            self.assertEqual(result["status"]["high_tension_queue_count"], 2)
            self.assertEqual(result["status"]["flat_queue_count"], 2)

            queue_rows = result["queue_rows"]
            self.assertEqual([row["cycle_id"] for row in queue_rows], ["triangle:003", "triangle:002", "triangle:000", "triangle:001"])
            self.assertEqual([row["provisional_closure_band"] for row in queue_rows], ["high_tension", "high_tension", "flat", "flat"])

            with open(out_dir / reading_queue.DEFAULT_QUEUE, "r", encoding="utf-8", newline="") as handle:
                persisted = list(csv.DictReader(handle))
            self.assertEqual(len(persisted), 4)
            self.assertEqual(persisted[0]["queue_rank"], "1")

            read_text = (out_dir / reading_queue.DEFAULT_READ).read_text(encoding="utf-8")
            self.assertIn("This queue intentionally excludes provisional `tense` rows.", read_text)

    def _build_phenotype_prep_fixture(self, root: Path) -> Path:
        phenotype_dir = root / "phenotype_prep"
        write_json(
            phenotype_dir / "manifest.json",
            {
                "run_id": "phenotype_prep_fixture",
                "code_git_commit": "fixture-commit",
            },
        )
        write_csv(
            phenotype_dir / "triangle_phenotype_tagging_template.csv",
            [
                "cycle_id",
                "sample_id",
                "base_node_id",
                "edge_id_path",
                "anchor_qualified_path",
                "relation_kind_path",
                "compatibility_gap_path_summary",
                "holonomy_residual_fro",
                "residual_percentile",
                "provisional_closure_band",
                "prompt_path",
                "answer_path",
                "support_anchor_path",
                "conflict_anchor_path",
                "phenotype_tag",
                "phenotype_notes",
            ],
            [
                {
                    "cycle_id": "triangle:000",
                    "sample_id": "sample_0",
                    "base_node_id": "sample_0:answer_state",
                    "edge_id_path": "[]",
                    "anchor_qualified_path": "[]",
                    "relation_kind_path": "[]",
                    "compatibility_gap_path_summary": "{}",
                    "holonomy_residual_fro": "0.01",
                    "residual_percentile": "0.0",
                    "provisional_closure_band": "flat",
                    "prompt_path": "p0",
                    "answer_path": "a0",
                    "support_anchor_path": "",
                    "conflict_anchor_path": "",
                    "phenotype_tag": "",
                    "phenotype_notes": "",
                },
                {
                    "cycle_id": "triangle:001",
                    "sample_id": "sample_1",
                    "base_node_id": "sample_1:answer_state",
                    "edge_id_path": "[]",
                    "anchor_qualified_path": "[]",
                    "relation_kind_path": "[]",
                    "compatibility_gap_path_summary": "{}",
                    "holonomy_residual_fro": "0.02",
                    "residual_percentile": "0.1",
                    "provisional_closure_band": "flat",
                    "prompt_path": "p1",
                    "answer_path": "a1",
                    "support_anchor_path": "",
                    "conflict_anchor_path": "",
                    "phenotype_tag": "",
                    "phenotype_notes": "",
                },
                {
                    "cycle_id": "triangle:002",
                    "sample_id": "sample_2",
                    "base_node_id": "sample_2:answer_state",
                    "edge_id_path": "[]",
                    "anchor_qualified_path": "[]",
                    "relation_kind_path": "[]",
                    "compatibility_gap_path_summary": "{}",
                    "holonomy_residual_fro": "0.95",
                    "residual_percentile": "0.8",
                    "provisional_closure_band": "high_tension",
                    "prompt_path": "p2",
                    "answer_path": "a2",
                    "support_anchor_path": "",
                    "conflict_anchor_path": "",
                    "phenotype_tag": "",
                    "phenotype_notes": "",
                },
                {
                    "cycle_id": "triangle:003",
                    "sample_id": "sample_3",
                    "base_node_id": "sample_3:answer_state",
                    "edge_id_path": "[]",
                    "anchor_qualified_path": "[]",
                    "relation_kind_path": "[]",
                    "compatibility_gap_path_summary": "{}",
                    "holonomy_residual_fro": "1.20",
                    "residual_percentile": "1.0",
                    "provisional_closure_band": "high_tension",
                    "prompt_path": "p3",
                    "answer_path": "a3",
                    "support_anchor_path": "",
                    "conflict_anchor_path": "",
                    "phenotype_tag": "",
                    "phenotype_notes": "",
                },
                {
                    "cycle_id": "triangle:004",
                    "sample_id": "sample_4",
                    "base_node_id": "sample_4:answer_state",
                    "edge_id_path": "[]",
                    "anchor_qualified_path": "[]",
                    "relation_kind_path": "[]",
                    "compatibility_gap_path_summary": "{}",
                    "holonomy_residual_fro": "0.50",
                    "residual_percentile": "0.5",
                    "provisional_closure_band": "tense",
                    "prompt_path": "p4",
                    "answer_path": "a4",
                    "support_anchor_path": "",
                    "conflict_anchor_path": "",
                    "phenotype_tag": "",
                    "phenotype_notes": "",
                },
            ],
        )
        return phenotype_dir


if __name__ == "__main__":
    unittest.main()
