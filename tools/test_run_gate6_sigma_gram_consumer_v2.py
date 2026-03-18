#!/usr/bin/env python3
"""Regression tests for Gate6 sigma-gram v2 consumer."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import run_gate6_sigma_gram_consumer_v2 as consumer


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
                "rank_local": 3,
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
                "rank_local": 3,
                "flags_compact": "none",
            },
        ],
    )
    gram_raw = np.asarray(
        [
            [
                [1.0, 0.4, 0.1],
                [0.4, 1.0, 0.2],
                [0.1, 0.2, 1.0],
            ],
            [
                [1.0, 0.1, 0.0],
                [0.1, 1.0, 0.3],
                [0.0, 0.3, 1.0],
            ],
        ],
        dtype=np.float64,
    )
    singular_values = np.asarray(
        [
            [2.0, 1.0, 0.1],
            [3.0, 1.5, 0.3],
        ],
        dtype=np.float64,
    )
    np.savez(
        gate6_dir / "native_object_arrays.npz",
        gram_raw=gram_raw,
        singular_values=singular_values,
    )
    return gate6_dir


class Gate6SigmaGramConsumerV2Tests(unittest.TestCase):
    def test_invalid_sigma1_returns_missing_primary(self) -> None:
        gram_raw = np.eye(3, dtype=np.float64)
        metrics = consumer.compute_sigma_gap_tailkeep_metrics(
            gram_raw=gram_raw,
            singular_values=np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        )
        self.assertEqual(metrics["loop_outcome"], "invalid_singular_values")
        self.assertIsNotNone(metrics[consumer.BASELINE_METRIC_ID])
        self.assertIsNone(metrics[consumer.PRIMARY_METRIC_ID])
        self.assertIsNone(metrics[consumer.PRIMARY_AUX_METRIC_ID])
        self.assertIsNone(metrics[consumer.PRIMARY_LEAKAGE_METRIC_ID])

    def test_smaller_tail_with_same_gap_increases_primary(self) -> None:
        gram_raw = np.asarray(
            [
                [1.0, 0.3, 0.0],
                [0.3, 1.0, 0.2],
                [0.0, 0.2, 1.0],
            ],
            dtype=np.float64,
        )
        higher_tail = consumer.compute_sigma_gap_tailkeep_metrics(
            gram_raw=gram_raw,
            singular_values=np.asarray([2.0, 1.0, 0.4], dtype=np.float64),
        )
        lower_tail = consumer.compute_sigma_gap_tailkeep_metrics(
            gram_raw=gram_raw,
            singular_values=np.asarray([2.0, 1.0, 0.1], dtype=np.float64),
        )
        self.assertEqual(higher_tail["loop_outcome"], "none")
        self.assertEqual(lower_tail["loop_outcome"], "none")
        self.assertLess(float(higher_tail[consumer.PRIMARY_LEAKAGE_METRIC_ID]), float(lower_tail[consumer.PRIMARY_LEAKAGE_METRIC_ID]))
        self.assertLess(float(higher_tail[consumer.PRIMARY_METRIC_ID]), float(lower_tail[consumer.PRIMARY_METRIC_ID]))

    def test_consumer_rerun_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            gate6_dir = make_gate6_dir(tmp_dir)
            out_a = tmp_dir / "out_a"
            out_b = tmp_dir / "out_b"
            script = REPO_ROOT / "tools" / "run_gate6_sigma_gram_consumer_v2.py"

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
                        "gate6f_test",
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
            self.assertEqual(
                (out_a / "gate6f_token_telemetry.csv").read_text(encoding="utf-8"),
                (out_b / "gate6f_token_telemetry.csv").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                (out_a / "gate6f_sample_summary.csv").read_text(encoding="utf-8"),
                (out_b / "gate6f_sample_summary.csv").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                (out_a / "gate6f_aggregate_summary.md").read_text(encoding="utf-8"),
                (out_b / "gate6f_aggregate_summary.md").read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
