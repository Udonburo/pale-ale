#!/usr/bin/env python3
"""Regression tests for Gate7 progression leakage consumer."""

import json
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import run_gate7_progression_leak_consumer as consumer


REPO_ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def make_gate6_dir(root: Path) -> Path:
    gate6_dir = root / "gate6"
    write_json(
        gate6_dir / "manifest.json",
        {
            "run_id": "gate6_unit",
            "method_id": "native_local_span_gate6a_v1",
        },
    )
    write_jsonl(
        gate6_dir / "step_index.jsonl",
        [
            {
                "sample_id": 1,
                "step": 0,
                "token_text": "A",
                "label_token": 0,
                "baseline_logprob": -1.0,
                "baseline_entropy": 2.0,
                "offset_start": 0,
                "offset_end": 1,
                "array_row_index": 0,
                "rank_local": 2,
                "flags_compact": "none",
            },
            {
                "sample_id": 1,
                "step": 1,
                "token_text": "B",
                "label_token": 1,
                "baseline_logprob": -0.5,
                "baseline_entropy": 1.5,
                "offset_start": 1,
                "offset_end": 2,
                "array_row_index": 1,
                "rank_local": 1,
                "flags_compact": "none",
            },
        ],
    )
    d_model = 4
    basis = np.zeros((2, d_model, 3), dtype=np.float64)
    basis[0, :, 0] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    basis[0, :, 1] = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    basis[1, :, 0] = np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    coords_local = np.zeros((2, 3, 3), dtype=np.float64)
    coords_local[0, 0, 0] = 1.0
    coords_local[1, 0, 0] = 1.0
    gram_raw = np.stack([np.eye(3, dtype=np.float64), np.eye(3, dtype=np.float64)], axis=0)
    rank_local = np.asarray([2, 1], dtype=np.int64)
    np.savez(
        gate6_dir / "native_object_arrays.npz",
        basis=basis,
        coords_local=coords_local,
        gram_raw=gram_raw,
        rank_local=rank_local,
    )
    return gate6_dir


class Gate7ProgressionLeakConsumerTests(unittest.TestCase):
    def test_projection_leak_is_zero_inside_span(self) -> None:
        current_basis = np.zeros((4, 3), dtype=np.float64)
        current_basis[:, 0] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        current_basis[:, 1] = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        next_basis = np.zeros((4, 3), dtype=np.float64)
        next_basis[:, 0] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        next_coords = np.zeros((3, 3), dtype=np.float64)
        next_coords[0, 0] = 1.0
        metrics = consumer.compute_progression_metrics(
            current_basis=current_basis,
            current_rank=2,
            current_gram_raw=np.eye(3, dtype=np.float64),
            next_basis=next_basis,
            next_coords_local=next_coords,
            next_rank=1,
        )
        self.assertEqual(metrics["loop_outcome"], "none")
        self.assertAlmostEqual(float(metrics[consumer.PRIMARY_METRIC_ID]), 0.0, places=10)
        self.assertAlmostEqual(float(metrics[consumer.PRIMARY_AUX_METRIC_ID]), 1.0, places=10)

    def test_projection_leak_is_one_for_orthogonal_jump(self) -> None:
        current_basis = np.zeros((4, 3), dtype=np.float64)
        current_basis[:, 0] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        next_basis = np.zeros((4, 3), dtype=np.float64)
        next_basis[:, 0] = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        next_coords = np.zeros((3, 3), dtype=np.float64)
        next_coords[0, 0] = 1.0
        metrics = consumer.compute_progression_metrics(
            current_basis=current_basis,
            current_rank=1,
            current_gram_raw=np.eye(3, dtype=np.float64),
            next_basis=next_basis,
            next_coords_local=next_coords,
            next_rank=1,
        )
        self.assertEqual(metrics["loop_outcome"], "none")
        self.assertAlmostEqual(float(metrics[consumer.PRIMARY_METRIC_ID]), 1.0, places=10)
        self.assertAlmostEqual(float(metrics[consumer.PRIMARY_AUX_METRIC_ID]), 0.0, places=10)

    def test_consumer_rerun_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            gate6_dir = make_gate6_dir(tmp_dir)
            out_a = tmp_dir / "out_a"
            out_b = tmp_dir / "out_b"
            script = REPO_ROOT / "tools" / "run_gate7_progression_leak_consumer.py"

            for out_dir in (out_a, out_b):
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(script),
                        "--gate6-dir",
                        str(gate6_dir),
                        "--out-dir",
                        str(out_dir),
                        "--run-id",
                        "gate7a_test",
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
            self.assertEqual(manifest_a["n_loop_rows_valid"], 1)
            self.assertEqual(manifest_a["n_loop_rows_structural_no_successor"], 1)
            self.assertEqual(manifest_a["n_loop_rows_missing"], 0)
            self.assertEqual(
                (out_a / "gate7a_token_telemetry.csv").read_text(encoding="utf-8"),
                (out_b / "gate7a_token_telemetry.csv").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                (out_a / "gate7a_sample_summary.csv").read_text(encoding="utf-8"),
                (out_b / "gate7a_sample_summary.csv").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                (out_a / "gate7a_aggregate_summary.md").read_text(encoding="utf-8"),
                (out_b / "gate7a_aggregate_summary.md").read_text(encoding="utf-8"),
            )
            with open(out_a / "gate7a_sample_summary.csv", "r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["n_loop_steps_valid"], "1")
            self.assertEqual(rows[0]["n_loop_steps_structural_no_successor"], "1")
            self.assertEqual(rows[0]["n_loop_steps_missing"], "0")


if __name__ == "__main__":
    unittest.main()
