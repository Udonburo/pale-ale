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
                evaluator.DEFAULT_GUARDRAIL_METRIC: score_f,
                evaluator.DEFAULT_PRIMARY_METRIC: score_edge,
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
        pair_rows = evaluator.build_pair_rows(
            token_rows,
            make_seam_rows(1, 2),
            topk=2,
            guardrail_metric=evaluator.DEFAULT_GUARDRAIL_METRIC,
            primary_metric=evaluator.DEFAULT_PRIMARY_METRIC,
        )

        self.assertEqual(len(pair_rows), 1)
        pair = pair_rows[0]
        self.assertAlmostEqual(float(pair["delta_max_guardrail"]), 0.0, places=10)
        self.assertAlmostEqual(float(pair["delta_max_primary"]), 0.0, places=10)
        self.assertAlmostEqual(float(pair["delta_p90_guardrail"]), 0.0, places=10)
        self.assertAlmostEqual(float(pair["delta_p90_primary"]), 0.0, places=10)
        self.assertAlmostEqual(float(pair["delta_mean_guardrail"]), 0.0, places=10)
        self.assertAlmostEqual(float(pair["delta_mean_primary"]), 0.0, places=10)

    def test_stronger_perturbed_increases_delta(self) -> None:
        clean_rows = make_token_rows(1, [(0.2, 0.3), (0.4, 0.5), (0.6, 0.7)])
        weak_rows = clean_rows + make_token_rows(2, [(0.3, 0.4), (0.5, 0.6), (0.7, 0.8)])
        strong_rows = clean_rows + make_token_rows(2, [(0.7, 0.8), (0.9, 1.0), (1.1, 1.2)])

        weak_pair = evaluator.build_pair_rows(
            weak_rows,
            make_seam_rows(1, 2),
            topk=2,
            guardrail_metric=evaluator.DEFAULT_GUARDRAIL_METRIC,
            primary_metric=evaluator.DEFAULT_PRIMARY_METRIC,
        )[0]
        strong_pair = evaluator.build_pair_rows(
            strong_rows,
            make_seam_rows(1, 2),
            topk=2,
            guardrail_metric=evaluator.DEFAULT_GUARDRAIL_METRIC,
            primary_metric=evaluator.DEFAULT_PRIMARY_METRIC,
        )[0]

        self.assertGreater(
            float(strong_pair["delta_max_guardrail"]),
            float(weak_pair["delta_max_guardrail"]),
        )
        self.assertGreater(
            float(strong_pair["delta_max_primary"]),
            float(weak_pair["delta_max_primary"]),
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
        pair = evaluator.build_pair_rows(
            token_rows,
            make_seam_rows(1, 2),
            topk=2,
            guardrail_metric=evaluator.DEFAULT_GUARDRAIL_METRIC,
            primary_metric=evaluator.DEFAULT_PRIMARY_METRIC,
        )[0]

        self.assertAlmostEqual(float(pair["delta_max_guardrail"]), 0.3, places=10)
        self.assertAlmostEqual(float(pair["delta_max_primary"]), 0.5, places=10)
        self.assertAlmostEqual(float(pair["delta_p90_guardrail"]), 0.3, places=10)
        self.assertAlmostEqual(float(pair["delta_p90_primary"]), 0.5, places=10)

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
                    evaluator.DEFAULT_GUARDRAIL_METRIC,
                    evaluator.DEFAULT_PRIMARY_METRIC,
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
            self.assertIn(f"mean_delta_max_{evaluator.DEFAULT_GUARDRAIL_METRIC}", report)
            self.assertIn(f"mean_delta_max_{evaluator.DEFAULT_PRIMARY_METRIC}", report)
            self.assertNotIn("perturbation_overlap", report)
            header = (out_a / "gate6b_seam_pair_summary.csv").read_text(encoding="utf-8").splitlines()[0]
            self.assertNotIn("perturbation_overlap_topk", header)
            self.assertIn("delta_max_primary", header)
            self.assertIn("delta_max_guardrail", header)
            self.assertNotIn("edge_plane", header)

    def test_cli_accepts_custom_metric_names_and_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            token_csv = tmp_dir / "gate6c_token_telemetry.csv"
            seam_jsonl = tmp_dir / "seam.jsonl"
            out_dir = tmp_dir / "out"
            script = REPO_ROOT / "tools" / "evaluate_gate6_native_object_seam_pairs.py"
            custom_primary = "ray_projector_loop_projective_chordal_v1"
            custom_guardrail = "score_F_gram_loop_v1"

            write_csv(
                token_csv,
                fieldnames=["sample_id", "step", "loop_outcome", custom_guardrail, custom_primary],
                rows=[
                    {
                        "sample_id": 1,
                        "step": 0,
                        "loop_outcome": "none",
                        custom_guardrail: 0.1,
                        custom_primary: 0.2,
                    },
                    {
                        "sample_id": 2,
                        "step": 0,
                        "loop_outcome": "none",
                        custom_guardrail: 0.3,
                        custom_primary: 0.5,
                    },
                ],
            )
            write_jsonl(seam_jsonl, make_seam_rows(1, 2))

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
                    "gate6c_pairs_test",
                    "--primary-metric",
                    custom_primary,
                    "--guardrail-metric",
                    custom_guardrail,
                    "--artifact-prefix",
                    "gate6c_seam",
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["primary_metric_id"], custom_primary)
            self.assertEqual(manifest["guardrail_metric_id"], custom_guardrail)
            self.assertTrue((out_dir / "gate6c_seam_report.md").exists())
            header = (out_dir / "gate6c_seam_pair_summary.csv").read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("delta_max_primary", header)
            self.assertIn("delta_max_guardrail", header)
            self.assertNotIn("edge_plane", header)


if __name__ == "__main__":
    unittest.main()
