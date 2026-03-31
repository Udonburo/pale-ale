#!/usr/bin/env python3
"""Tests for Gate12A triangle phenotype first-pass capture."""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import run_gate12a_triangle_phenotype_first_pass as first_pass


class RunGate12ATrianglePhenotypeFirstPassTest(unittest.TestCase):
    def write_reviewed_csv(self, path: Path) -> None:
        rows = [
            {
                "queue_rank": "1",
                "cycle_id": "triangle:000001",
                "sample_id": "sample_000001",
                "provisional_closure_band": "high_tension",
                "holonomy_residual_fro": "2.0",
                "residual_percentile": "1.0",
                "reviewed_phenotype_tag": "support_fused",
                "reviewed_phenotype_notes": "High residual but still support closure.",
                "prompt_path": "runs/a/prompt.txt",
                "answer_path": "runs/a/answer.txt",
                "support_anchor_path": "runs/a/support_anchor.txt",
                "conflict_anchor_path": "",
            },
            {
                "queue_rank": "2",
                "cycle_id": "triangle:000002",
                "sample_id": "sample_000002",
                "provisional_closure_band": "flat",
                "holonomy_residual_fro": "0.01",
                "residual_percentile": "0.0",
                "reviewed_phenotype_tag": "conflict_respected",
                "reviewed_phenotype_notes": "Flat conflict-respecting row.",
                "prompt_path": "runs/b/prompt.txt",
                "answer_path": "runs/b/answer.txt",
                "support_anchor_path": "runs/b/support_anchor.txt",
                "conflict_anchor_path": "runs/b/conflict_anchor.txt",
            },
        ]
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def write_source_packet_rows(self, path: Path) -> None:
        rows = [
            {
                "queue_rank": 1,
                "cycle_id": "triangle:000001",
                "sample_id": "sample_000001",
                "provisional_closure_band": "high_tension",
            },
            {
                "queue_rank": 2,
                "cycle_id": "triangle:000002",
                "sample_id": "sample_000002",
                "provisional_closure_band": "flat",
            },
        ]
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    def test_main_writes_status_manifest_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            reviewed_csv = tmp / "reviewed.csv"
            source_packet_dir = tmp / "source_packet"
            out_dir = tmp / "first_pass"
            source_packet_dir.mkdir(parents=True)
            self.write_reviewed_csv(reviewed_csv)
            self.write_source_packet_rows(source_packet_dir / "triangle_reading_packet_rows.jsonl")

            argv = [
                "prog",
                "--reviewed-csv",
                str(reviewed_csv),
                "--source-packet-dir",
                str(source_packet_dir),
                "--out-dir",
                str(out_dir),
                "--title-label",
                "Qwen transcript_v1",
            ]
            with mock.patch("sys.argv", argv):
                self.assertEqual(first_pass.main(), 0)

            status = json.loads((out_dir / "gate12a_triangle_phenotype_first_pass_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["packet_row_count"], 2)
            self.assertEqual(status["source_packet_row_count"], 2)
            self.assertEqual(status["high_tension_count"], 1)
            self.assertEqual(status["flat_count"], 1)
            self.assertEqual(status["reviewed_tag_counts"][0]["reviewed_phenotype_tag"], "support_fused")
            self.assertEqual(
                status["reviewed_tag_counts_by_band"],
                [
                    {
                        "provisional_closure_band": "flat",
                        "reviewed_phenotype_tag": "conflict_respected",
                        "count": 1,
                    },
                    {
                        "provisional_closure_band": "high_tension",
                        "reviewed_phenotype_tag": "support_fused",
                        "count": 1,
                    },
                ],
            )

            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["run_id"], "first_pass")
            self.assertEqual(manifest["source_packet_row_count"], 2)
            self.assertIn("triangle_phenotype_first_pass.csv", manifest["paths"])
            self.assertIn("source_packet_rows_sha256", manifest)

            markdown = (out_dir / "gate12a_triangle_phenotype_first_pass.md").read_text(encoding="utf-8")
            self.assertIn("Qwen transcript_v1", markdown)
            self.assertIn("source packet run: `source_packet`", markdown)
            self.assertIn("high_tension` / `support_fused", markdown)

    def test_main_fails_when_reviewed_sheet_does_not_match_packet(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            reviewed_csv = tmp / "reviewed.csv"
            source_packet_dir = tmp / "source_packet"
            out_dir = tmp / "first_pass"
            source_packet_dir.mkdir(parents=True)
            self.write_reviewed_csv(reviewed_csv)
            with open(source_packet_dir / "triangle_reading_packet_rows.jsonl", "w", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "queue_rank": 1,
                    "cycle_id": "triangle:wrong",
                    "sample_id": "sample_000001",
                    "provisional_closure_band": "high_tension",
                }) + "\n")
                handle.write(json.dumps({
                    "queue_rank": 2,
                    "cycle_id": "triangle:000002",
                    "sample_id": "sample_000002",
                    "provisional_closure_band": "flat",
                }) + "\n")

            argv = [
                "prog",
                "--reviewed-csv",
                str(reviewed_csv),
                "--source-packet-dir",
                str(source_packet_dir),
                "--out-dir",
                str(out_dir),
            ]
            with mock.patch("sys.argv", argv):
                with self.assertRaisesRegex(ValueError, "reviewed rows do not match source packet identifiers"):
                    first_pass.main()


if __name__ == "__main__":
    raise SystemExit(unittest.main())
