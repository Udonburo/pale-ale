#!/usr/bin/env python3
"""Regression tests for Gate12A triangle phenotype tag prep."""

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

import run_gate12a_triangle_phenotype_tag_prep as phenotype_prep


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class Gate12ATrianglePhenotypeTagPrepTest(unittest.TestCase):
    def test_run_triangle_phenotype_tag_prep_emits_provisional_bands_and_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audit_dir = self._build_triangle_text_audit_fixture(root)
            out_dir = root / "phenotype_tag_prep"

            result = phenotype_prep.run_triangle_phenotype_tag_prep(
                triangle_text_audit_dir=audit_dir,
                out_dir=out_dir,
                flat_quantile=0.25,
                high_quantile=0.75,
            )

            self.assertEqual(result["status"]["joined_row_count"], 4)
            self.assertEqual(result["status"]["flat_count"], 1)
            self.assertEqual(result["status"]["tense_count"], 2)
            self.assertEqual(result["status"]["high_tension_count"], 1)

            bands = {row["cycle_id"]: row["provisional_closure_band"] for row in result["band_rows"]}
            self.assertEqual(bands["triangle:000"], "flat")
            self.assertEqual(bands["triangle:003"], "high_tension")

            with open(out_dir / phenotype_prep.DEFAULT_TEMPLATE, "r", encoding="utf-8", newline="") as handle:
                template_rows = list(csv.DictReader(handle))
            self.assertEqual(len(template_rows), 4)
            self.assertEqual(template_rows[0]["phenotype_tag"], "")
            self.assertEqual(template_rows[0]["phenotype_notes"], "")

            read_text = (out_dir / phenotype_prep.DEFAULT_READ).read_text(encoding="utf-8")
            self.assertIn("These bands are provisional closure bands only.", read_text)
            self.assertIn("conflict_respected", read_text)

    def _build_triangle_text_audit_fixture(self, root: Path) -> Path:
        audit_dir = root / "triangle_text_audit"
        write_json(
            audit_dir / "manifest.json",
            {
                "run_id": "triangle_text_audit_fixture",
                "code_git_commit": "fixture-commit",
            },
        )
        write_jsonl(
            audit_dir / "triangle_text_surface_joined.jsonl",
            [
                {
                    "cycle_id": "triangle:000",
                    "sample_id": "sample_000001",
                    "base_node_id": "sample_000001:answer_state",
                    "edge_id_path": ["e0", "e1", "e2"],
                    "anchor_qualified_path": [True, True, False],
                    "relation_kind_path": ["trusted_tree", "trusted_tree", "residual_chord"],
                    "compatibility_gap_path_summary": {"min": 0.1, "median": 0.2, "max": 0.3, "mean": 0.2},
                    "holonomy_residual_fro": 0.10,
                    "residual_percentile": 0.0,
                    "prompt_path": "prompt0.txt",
                    "answer_path": "answer0.txt",
                    "support_anchor_path": "support0.txt",
                    "conflict_anchor_path": "",
                },
                {
                    "cycle_id": "triangle:001",
                    "sample_id": "sample_000002",
                    "base_node_id": "sample_000002:answer_state",
                    "edge_id_path": ["e3", "e4", "e5"],
                    "anchor_qualified_path": [False, True, True],
                    "relation_kind_path": ["residual_chord", "trusted_tree", "trusted_tree"],
                    "compatibility_gap_path_summary": {"min": 0.2, "median": 0.4, "max": 0.6, "mean": 0.4},
                    "holonomy_residual_fro": 0.20,
                    "residual_percentile": 0.33,
                    "prompt_path": "prompt1.txt",
                    "answer_path": "answer1.txt",
                    "support_anchor_path": "support1.txt",
                    "conflict_anchor_path": "",
                },
                {
                    "cycle_id": "triangle:002",
                    "sample_id": "sample_000003",
                    "base_node_id": "sample_000003:answer_state",
                    "edge_id_path": ["e6", "e7", "e8"],
                    "anchor_qualified_path": [True, False, True],
                    "relation_kind_path": ["trusted_tree", "residual_chord", "trusted_tree"],
                    "compatibility_gap_path_summary": {"min": 0.3, "median": 0.5, "max": 0.7, "mean": 0.5},
                    "holonomy_residual_fro": 0.80,
                    "residual_percentile": 0.66,
                    "prompt_path": "prompt2.txt",
                    "answer_path": "answer2.txt",
                    "support_anchor_path": "support2.txt",
                    "conflict_anchor_path": "",
                },
                {
                    "cycle_id": "triangle:003",
                    "sample_id": "sample_000004",
                    "base_node_id": "sample_000004:answer_state",
                    "edge_id_path": ["e9", "e10", "e11"],
                    "anchor_qualified_path": [False, True, True],
                    "relation_kind_path": ["residual_chord", "trusted_tree", "trusted_tree"],
                    "compatibility_gap_path_summary": {"min": 0.4, "median": 0.6, "max": 0.9, "mean": 0.633},
                    "holonomy_residual_fro": 1.00,
                    "residual_percentile": 1.0,
                    "prompt_path": "prompt3.txt",
                    "answer_path": "answer3.txt",
                    "support_anchor_path": "support3.txt",
                    "conflict_anchor_path": "",
                },
            ],
        )
        return audit_dir


if __name__ == "__main__":
    unittest.main()
