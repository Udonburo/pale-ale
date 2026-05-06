#!/usr/bin/env python3
"""Regression tests for Gate12B source-facing annotation."""

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

import annotate_gate12b_source_inspection_queue as annotate


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class Gate12BSourceAnnotationTest(unittest.TestCase):
    def test_derive_from_queue_writes_annotations_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            queue_jsonl = root / "queue" / "gate12b_source_inspection_queue.jsonl"
            out_dir = root / "annotation"
            write_jsonl(queue_jsonl, self._queue_rows())

            result = annotate.run_gate12b_source_annotation(
                queue_jsonl=queue_jsonl,
                out_dir=out_dir,
                derive_from_queue=True,
                annotator="fixture_annotator",
            )

            self.assertEqual(result["status"]["annotation_row_count"], 4)
            self.assertEqual(
                result["status"]["tag_counts"],
                {
                    "ambiguous": 1,
                    "conflict-following": 1,
                    "non-gluing": 1,
                    "support-following": 1,
                },
            )
            self.assertEqual(result["status"]["high_tension_conflict_following_count"], 1)
            self.assertEqual(result["status"]["flat_support_or_non_gluing_count"], 2)

            with open(out_dir / annotate.DEFAULT_ANNOTATIONS_CSV, "r", encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(len(csv_rows), 4)
            self.assertEqual(csv_rows[0]["annotator"], "fixture_annotator")

            summary_json = json.loads((out_dir / annotate.DEFAULT_SUMMARY_JSON).read_text(encoding="utf-8"))
            self.assertEqual(summary_json["status"]["annotation_row_count"], 4)

            read_text = (out_dir / annotate.DEFAULT_READ).read_text(encoding="utf-8")
            self.assertIn("not answer-quality labels", read_text)

    def test_supplied_annotations_must_match_queue_keys_and_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            queue_jsonl = root / "queue" / "gate12b_source_inspection_queue.jsonl"
            annotations_jsonl = root / "annotations" / "annotations.jsonl"
            out_dir = root / "out"
            write_jsonl(queue_jsonl, self._queue_rows()[:1])
            supplied = [
                {
                    "queue_rank": 1,
                    "case_label": "case_a",
                    "cycle_id": "triangle:000001",
                    "candidate_side": "flat",
                    "relation_kind_signature": "residual_chord=1|trusted_tree=2",
                    "source_facing_tag": "support-following",
                    "evidence_note": "manual read",
                    "annotator": "human",
                }
            ]
            write_jsonl(annotations_jsonl, supplied)

            result = annotate.run_gate12b_source_annotation(
                queue_jsonl=queue_jsonl,
                out_dir=out_dir,
                annotation_jsonl=annotations_jsonl,
            )

            self.assertEqual(result["status"]["tag_counts"], {"support-following": 1})

            supplied[0]["source_facing_tag"] = "bad-tag"
            write_jsonl(annotations_jsonl, supplied)
            with self.assertRaisesRegex(ValueError, "unsupported source_facing_tag"):
                annotate.run_gate12b_source_annotation(
                    queue_jsonl=queue_jsonl,
                    out_dir=root / "bad_out",
                    annotation_jsonl=annotations_jsonl,
                )

    def test_rejects_out_dir_aliasing_queue_artifact_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            queue_dir = root / "queue"
            queue_jsonl = queue_dir / "gate12b_source_inspection_queue.jsonl"
            write_jsonl(queue_jsonl, self._queue_rows()[:1])
            original_text = queue_jsonl.read_text(encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "same directory"):
                annotate.run_gate12b_source_annotation(
                    queue_jsonl=queue_jsonl,
                    out_dir=queue_dir,
                    derive_from_queue=True,
                )

            self.assertEqual(queue_jsonl.read_text(encoding="utf-8"), original_text)

    def test_rejects_missing_annotation_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            queue_jsonl = root / "queue" / "gate12b_source_inspection_queue.jsonl"
            write_jsonl(queue_jsonl, self._queue_rows()[:1])

            with self.assertRaisesRegex(ValueError, "either --annotation-jsonl or --derive-from-queue"):
                annotate.run_gate12b_source_annotation(
                    queue_jsonl=queue_jsonl,
                    out_dir=root / "out",
                )

    def _queue_rows(self) -> list[dict]:
        base = {
            "case_label": "case_a",
            "sample_id": "sample_001",
            "relation_kind_signature": "residual_chord=1|trusted_tree=2",
            "answer_contains_support_anchor_text": False,
            "answer_contains_conflict_anchor_text": False,
        }
        return [
            {
                **base,
                "queue_rank": 1,
                "cycle_id": "triangle:000001",
                "candidate_side": "flat",
                "answer_text": "The support path is warranted.",
                "answer_contains_support_anchor_text": True,
            },
            {
                **base,
                "queue_rank": 2,
                "cycle_id": "triangle:000002",
                "candidate_side": "high_tension",
                "relation_kind_signature": "residual_chord=3",
                "answer_text": "The conflict path is followed.",
                "answer_contains_conflict_anchor_text": True,
            },
            {
                **base,
                "queue_rank": 3,
                "cycle_id": "triangle:000003",
                "candidate_side": "flat",
                "answer_text": "No direct path is warranted across separate ledgers.",
            },
            {
                **base,
                "queue_rank": 4,
                "cycle_id": "triangle:000004",
                "candidate_side": "flat",
                "answer_text": "The row remains difficult to classify.",
            },
        ]


if __name__ == "__main__":
    unittest.main()
