#!/usr/bin/env python3
"""Regression tests for Gate12B run summarization."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import summarize_gate12b_runs as summary


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        h.update(handle.read())
    return h.hexdigest()


class Gate12BRunSummaryTest(unittest.TestCase):
    def test_summarize_gate12b_runs_emits_csv_json_and_checksum_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "gate12b_run"
            out_dir = root / "summary"
            self._build_gate12b_run(run_dir)

            result = summary.summarize_gate12b_runs([run_dir], out_dir)

            self.assertEqual(result["manifest"]["schema_version"], "gate12b_run_summary_v1")
            self.assertEqual(result["manifest"]["run_count"], 1)
            self.assertEqual(len(result["rows"]), 1)
            row = result["rows"][0]
            self.assertEqual(row["candidate_total"], 3)
            self.assertEqual(row["flat_candidate_count"], 2)
            self.assertEqual(row["high_tension_candidate_count"], 1)
            self.assertEqual(row["dominant_flat_relation_signature"], "residual_chord=1|trusted_tree=2")
            self.assertEqual(row["dominant_flat_relation_signature_count"], 2)
            self.assertEqual(row["dominant_high_tension_relation_signature"], "residual_chord=3")
            self.assertEqual(row["observer_support_distribution"], "3:2|4:1")
            self.assertEqual(row["scale_support_distribution"], "3:3")
            self.assertEqual(row["gauge_unstable_check_count"], 0)
            self.assertEqual(row["checksum_status"], "ok")
            self.assertEqual(row["checksum_mismatch_count"], 0)

            with open(out_dir / summary.DEFAULT_SUMMARY_CSV, "r", encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(len(csv_rows), 1)
            self.assertEqual(csv_rows[0]["candidate_total"], "3")

            summary_json = json.loads((out_dir / summary.DEFAULT_SUMMARY_JSON).read_text(encoding="utf-8"))
            self.assertEqual(summary_json["run_count"], 1)

    def test_summarize_gate12b_runs_rejects_empty_run_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "at least one"):
                summary.summarize_gate12b_runs([], Path(tmpdir) / "summary")

    def _build_gate12b_run(self, run_dir: Path) -> None:
        write_json(
            run_dir / summary.DEFAULT_MANIFEST,
            {
                "run_id": "gate12b_fixture",
                "source_gate12a_run_id": "gate12a_fixture",
                "observer_mode_set": "cycle_motif_expansion_v1",
                "top_k": 1,
                "min_observer_support": 3,
                "min_scale_support": 3,
                "flat_quantile": 0.25,
                "high_quantile": 0.75,
                "builder_script_sha256": "fixture-builder-sha",
                "status": {
                    "gauge_total_check_count": 20,
                    "gauge_unstable_check_count": 0,
                    "gauge_variant_signature_candidate_count": 3,
                },
            },
        )
        write_jsonl(
            run_dir / summary.DEFAULT_CANDIDATES,
            [
                {
                    "candidate_kind": "flat_observer_scale_stable",
                    "relation_kind_signature": "residual_chord=1|trusted_tree=2",
                    "observer_support_count": 3,
                    "scale_support_count": 3,
                },
                {
                    "candidate_kind": "flat_observer_scale_stable",
                    "relation_kind_signature": "residual_chord=1|trusted_tree=2",
                    "observer_support_count": 3,
                    "scale_support_count": 3,
                },
                {
                    "candidate_kind": "high_tension_observer_scale_stable",
                    "relation_kind_signature": "residual_chord=3",
                    "observer_support_count": 4,
                    "scale_support_count": 3,
                },
            ],
        )
        write_json(
            run_dir / summary.DEFAULT_GAUGE_SUMMARY,
            {
                "total_check_count": 20,
                "unstable_check_count": 0,
                "max_residual_delta_abs": 0.0,
            },
        )
        checksums = {
            summary.DEFAULT_MANIFEST: sha256_file(run_dir / summary.DEFAULT_MANIFEST),
            summary.DEFAULT_CANDIDATES: sha256_file(run_dir / summary.DEFAULT_CANDIDATES),
            summary.DEFAULT_GAUGE_SUMMARY: sha256_file(run_dir / summary.DEFAULT_GAUGE_SUMMARY),
        }
        write_json(run_dir / summary.DEFAULT_CHECKSUMS, checksums)


if __name__ == "__main__":
    unittest.main()
