#!/usr/bin/env python3
"""Regression tests for Gate6 native-object seam pair evaluation."""

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import evaluate_gate6_native_object_seam_pairs as evaluator


REPO_ROOT = Path(__file__).resolve().parents[1]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def make_token_rows(
    sample_id: int,
    values: list[tuple[float, float]],
    loop_outcomes: list[str] | None = None,
) -> list[dict[str, object]]:
    outcomes = loop_outcomes or ["none"] * len(values)
    rows: list[dict[str, object]] = []
    for step, ((score_f, score_edge), outcome) in enumerate(zip(values, outcomes)):
        rows.append(
            {
                "sample_id": sample_id,
                "step": step,
                "loop_outcome": outcome,
                evaluator.GUARDRAIL_METRIC: score_f,
                evaluator.PRIMARY_METRIC: score_edge,
            }
        )
    return rows


def make_seam_rows(clean_sample_id: int, perturbed_sample_id: int) -> list[dict[str, object]]:
    return [
        {
            "pair_id": 1,
            "challenge_class": "clean_consistent",
            "sample_id": clean_sample_id,
            "perturbation_family": "control",
        },
        {
            "pair_id": 1,
            "challenge_class": "seam_perturbed_consistent",
            "sample_id": perturbed_sample_id,
            "source_sample_id": clean_sample_id,
            "perturbation_family": "splice",
        },
    ]


class Gate6NativeObjectSeamPairEvaluatorTests(unittest.TestCase):
    def test_clean_equals_perturbed_has_zero_delta(self) -> None:
        token_rows = make_token_rows(1, [(0.2, 0.3), (0.4, 0.5), (0.6, 0.7)])
        token_rows.extend(make_token_rows(2, [(0.2, 0.3), (0.4, 0.5), (0.6, 0.7)]))
        pair_rows = evaluator.build_pair_rows(token_rows, make_seam_rows(1, 2), topk=2)

        self.assertEqual(len(pair_rows), 1)
        pair = pair_rows[0]
        self.assertAlmostEqual(float(pair["delta_max_f_gram"]), 0.0, places=10)
        self.assertAlmostEqual(float(pair["delta_max_edge_plane"]), 0.0, places=10)
        self.assertAlmostEqual(float(pair["delta_p90_f_gram"]), 0.0, places=10)
        self.assertAlmostEqual(float(pair["delta_p90_edge_plane"]), 0.0, places=10)
        self.assertAlmostEqual(float(pair["delta_mean_f_gram"]), 0.0, places=10)
        self.assertAlmostEqual(float(pair["delta_mean_edge_plane"]), 0.0, places=10)

    def test_stronger_perturbed_increases_delta(self) -> None:
        clean_rows = make_token_rows(1, [(0.2, 0.3), (0.4, 0.5), (0.6, 0.7)])
        weak_rows = clean_rows + make_token_rows(2, [(0.3, 0.4), (0.5, 0.6), (0.7, 0.8)])
        strong_rows = clean_rows + make_token_rows(2, [(0.7, 0.8), (0.9, 1.0), (1.1, 1.2)])

        weak_pair = evaluator.build_pair_rows(weak_rows, make_seam_rows(1, 2), topk=2)[0]
        strong_pair = evaluator.build_pair_rows(strong_rows, make_seam_rows(1, 2), topk=2)[0]

        self.assertGreater(
            float(strong_pair["delta_max_f_gram"]),
            float(weak_pair["delta_max_f_gram"]),
        )
        self.assertGreater(
            float(strong_pair["delta_max_edge_plane"]),
            float(weak_pair["delta_max_edge_plane"]),
        )

    def test_partial_loop_missing_rows_are_deterministically_skipped(self) -> None:
        token_rows = make_token_rows(
            1,
            [(0.2, 0.3), (99.0, 99.0)],
            loop_outcomes=["none", "partial_loop_missing"],
        )
        token_rows.extend(
            make_token_rows(
                2,
                [(0.5, 0.8), (999.0, 999.0)],
                loop_outcomes=["none", "partial_loop_missing"],
            )
        )
        pair = evaluator.build_pair_rows(token_rows, make_seam_rows(1, 2), topk=2)[0]

        self.assertAlmostEqual(float(pair["delta_max_f_gram"]), 0.3, places=10)
        self.assertAlmostEqual(float(pair["delta_max_edge_plane"]), 0.5, places=10)
        self.assertAlmostEqual(float(pair["delta_p90_f_gram"]), 0.3, places=10)
        self.assertAlmostEqual(float(pair["delta_p90_edge_plane"]), 0.5, places=10)

    def test_cli_rerun_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            token_csv = tmp_dir / "gate6b_token_telemetry.csv"
            seam_jsonl = tmp_dir / "seam.jsonl"
            out_a = tmp_dir / "out_a"
            out_b = tmp_dir / "out_b"
            script = REPO_ROOT / "tools" / "evaluate_gate6_native_object_seam_pairs.py"

            write_csv(
                token_csv,
                fieldnames=[
                    "sample_id",
                    "step",
                    "loop_outcome",
                    evaluator.GUARDRAIL_METRIC,
                    evaluator.PRIMARY_METRIC,
                ],
                rows=make_token_rows(1, [(0.2, 0.3), (0.4, 0.5), (0.6, 0.7)])
                + make_token_rows(2, [(0.3, 0.4), (0.7, 0.8), (0.9, 1.0)]),
            )
            write_jsonl(seam_jsonl, make_seam_rows(1, 2))

            for out_dir in (out_a, out_b):
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(script),
                        "--token-csv",
                        str(token_csv),
                        "--seam-jsonl",
                        str(seam_jsonl),
                        "--out-dir",
                        str(out_dir),
                        "--run-id",
                        "gate6b_pairs_test",
                        "--topk",
                        "3",
                    ],
                    cwd=str(REPO_ROOT),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

            manifest_a = json.loads((out_a / "manifest.json").read_text(encoding="utf-8"))
            manifest_b = json.loads((out_b / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest_a, manifest_b)
            self.assertEqual(manifest_a["topk"], 3)
            self.assertEqual(
                (out_a / "gate6b_seam_pair_summary.csv").read_text(encoding="utf-8"),
                (out_b / "gate6b_seam_pair_summary.csv").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                (out_a / "gate6b_seam_family_summary.csv").read_text(encoding="utf-8"),
                (out_b / "gate6b_seam_family_summary.csv").read_text(encoding="utf-8"),
            )
            report = (out_a / "gate6b_seam_report.md").read_text(encoding="utf-8")
            self.assertEqual(
                report,
                (out_b / "gate6b_seam_report.md").read_text(encoding="utf-8"),
            )
            self.assertIn(f"mean_delta_max_{evaluator.GUARDRAIL_METRIC}", report)
            self.assertIn(f"mean_delta_max_{evaluator.PRIMARY_METRIC}", report)
            self.assertNotIn("perturbation_overlap", report)
            header = (out_a / "gate6b_seam_pair_summary.csv").read_text(encoding="utf-8").splitlines()[0]
            self.assertNotIn("perturbation_overlap_topk", header)


if __name__ == "__main__":
    unittest.main()
