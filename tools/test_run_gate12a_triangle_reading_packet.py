#!/usr/bin/env python3
"""Regression tests for Gate12A triangle reading packet helper."""

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

import run_gate12a_triangle_reading_packet as reading_packet


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


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class Gate12ATriangleReadingPacketTest(unittest.TestCase):
    def test_run_triangle_reading_packet_emits_ranked_prompt_answer_packet(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            queue_dir = self._build_queue_fixture(root)
            text_audit_dir = self._build_text_audit_fixture(root)
            out_dir = root / "reading_packet"

            result = reading_packet.run_triangle_reading_packet(
                reading_queue_dir=queue_dir,
                triangle_text_audit_dir=text_audit_dir,
                out_dir=out_dir,
                limit=2,
            )

            self.assertEqual(result["status"]["packet_row_count"], 2)
            self.assertEqual(result["status"]["high_tension_packet_count"], 1)
            self.assertEqual(result["status"]["flat_packet_count"], 1)

            packet_rows = result["packet_rows"]
            self.assertEqual([row["queue_rank"] for row in packet_rows], [1, 2])
            self.assertEqual(packet_rows[0]["prompt_text"], "Prompt one")
            self.assertEqual(packet_rows[0]["answer_text"], "Answer one")
            self.assertEqual(packet_rows[1]["support_anchor_text"], "Support two")

            read_text = (out_dir / reading_packet.DEFAULT_READ).read_text(encoding="utf-8")
            self.assertIn("Queue 1", read_text)
            self.assertIn("Prompt one", read_text)
            self.assertIn("Conflict two", read_text)

    def test_read_csv_normalizes_bom_quoted_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "queue.csv"
            path.write_text(
                '\ufeff"queue_rank",cycle_id,sample_id,provisional_closure_band,holonomy_residual_fro,residual_percentile,prompt_path,answer_path,support_anchor_path,conflict_anchor_path,phenotype_tag,phenotype_notes\n'
                '1,triangle:001,sample_1,high_tension,1.0,1.0,p1,a1,,c1,,\n',
                encoding="utf-8",
                newline="\n",
            )

            rows = reading_packet.read_csv(path)

            self.assertEqual(rows[0]["queue_rank"], "1")
            self.assertEqual(rows[0]["cycle_id"], "triangle:001")

    def _build_queue_fixture(self, root: Path) -> Path:
        queue_dir = root / "queue"
        write_json(queue_dir / "manifest.json", {"run_id": "queue_fixture", "code_git_commit": "fixture-commit"})
        write_csv(
            queue_dir / "triangle_reading_queue.csv",
            [
                "queue_rank",
                "cycle_id",
                "sample_id",
                "provisional_closure_band",
                "holonomy_residual_fro",
                "residual_percentile",
                "prompt_path",
                "answer_path",
                "support_anchor_path",
                "conflict_anchor_path",
                "phenotype_tag",
                "phenotype_notes",
            ],
            [
                {
                    "queue_rank": "1",
                    "cycle_id": "triangle:001",
                    "sample_id": "sample_1",
                    "provisional_closure_band": "high_tension",
                    "holonomy_residual_fro": "1.0",
                    "residual_percentile": "1.0",
                    "prompt_path": "p1",
                    "answer_path": "a1",
                    "support_anchor_path": "",
                    "conflict_anchor_path": "c1",
                    "phenotype_tag": "",
                    "phenotype_notes": "",
                },
                {
                    "queue_rank": "2",
                    "cycle_id": "triangle:002",
                    "sample_id": "sample_2",
                    "provisional_closure_band": "flat",
                    "holonomy_residual_fro": "0.1",
                    "residual_percentile": "0.0",
                    "prompt_path": "p2",
                    "answer_path": "a2",
                    "support_anchor_path": "s2",
                    "conflict_anchor_path": "c2",
                    "phenotype_tag": "",
                    "phenotype_notes": "",
                },
            ],
        )
        return queue_dir

    def _build_text_audit_fixture(self, root: Path) -> Path:
        audit_dir = root / "text_audit"
        write_json(audit_dir / "manifest.json", {"run_id": "text_audit_fixture", "code_git_commit": "fixture-commit"})
        write_jsonl(
            audit_dir / "triangle_text_surface_joined.jsonl",
            [
                {
                    "cycle_id": "triangle:001",
                    "edge_id_path": ["e1", "e2", "e3"],
                    "anchor_qualified_path": [False, True, True],
                    "relation_kind_path": ["residual_chord", "trusted_tree", "trusted_tree"],
                    "compatibility_gap_path_summary": {"min": 0.8, "median": 0.9, "max": 1.0, "mean": 0.9},
                    "prompt_text": "Prompt one",
                    "answer_text": "Answer one",
                    "support_anchor_text": "",
                    "conflict_anchor_text": "Conflict one",
                },
                {
                    "cycle_id": "triangle:002",
                    "edge_id_path": ["e4", "e5", "e6"],
                    "anchor_qualified_path": [True, True, False],
                    "relation_kind_path": ["trusted_tree", "trusted_tree", "residual_chord"],
                    "compatibility_gap_path_summary": {"min": 0.1, "median": 0.2, "max": 0.3, "mean": 0.2},
                    "prompt_text": "Prompt two",
                    "answer_text": "Answer two",
                    "support_anchor_text": "Support two",
                    "conflict_anchor_text": "Conflict two",
                },
            ],
        )
        return audit_dir


if __name__ == "__main__":
    unittest.main()
