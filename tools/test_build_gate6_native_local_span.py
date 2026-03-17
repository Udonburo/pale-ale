#!/usr/bin/env python3
"""Regression tests for Gate6-A native local span builder."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import build_gate6_native_local_span as gate6_builder


REPO_ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_sample_root(root: Path) -> Path:
    samples_root = root / "samples"
    sample_dir = samples_root / "sample_000000"
    triplets_path = sample_dir / "triplets.ndjson"

    triplets = [
        {
            "step": 0,
            "absolute_pos": 10,
            "answer_char_start": 0,
            "answer_char_end": 1,
            "token_id": 101,
            "token_str": "A",
            "baseline_logprob": -1.25,
            "baseline_entropy": 2.0,
            "V_raw_native": [1.0, 0.0, 0.0, 0.0],
            "Splus_raw_native": [1.0, 1.0, 0.0, 0.0],
            "Sminus_raw_native": [1.0, 0.0, 1.0, 0.0],
        },
        {
            "step": 1,
            "absolute_pos": 11,
            "answer_char_start": 1,
            "answer_char_end": 2,
            "token_id": 102,
            "token_str": "B",
            "baseline_logprob": -0.75,
            "baseline_entropy": 1.5,
            "V_raw_native": [0.0, 1.0, 0.0, 0.0],
            "Splus_raw_native": [0.0, 1.0, 1.0, 0.0],
            "Sminus_raw_native": [0.0, 1.0, 0.0, 1.0],
        },
    ]
    write_jsonl(triplets_path, triplets)
    write_json(
        sample_dir / "meta.json",
        {
            "model_id": "test-model",
            "model_revision": "rev-test",
            "seed": 7,
            "output_ndjson_sha256": sha256_file(triplets_path),
            "exact_token_match_ratio": 1.0,
            "native_raw_schema_id": gate6_builder.RAW_NATIVE_SCHEMA_ID,
        },
    )
    write_json(
        sample_dir / "labels_meta.json",
        {
            "variant": "consistent",
            "world_type": "unit_test",
            "final_alignment_coverage_ratio": 1.0,
        },
    )
    write_jsonl(
        sample_dir / "labels.jsonl",
        [
            {"step": 0, "label": 0, "token_id": 101},
            {"step": 1, "label": 1, "token_id": 102},
        ],
    )
    return samples_root


class Gate6NativeLocalSpanTests(unittest.TestCase):
    def test_sign_tie_break_uses_lowest_index(self) -> None:
        fixed, flipped, anchor_index = gate6_builder.sign_fix_column(
            np.asarray([-0.5, 0.5, 0.0], dtype=np.float64)
        )
        np.testing.assert_allclose(fixed, np.asarray([0.5, -0.5, 0.0], dtype=np.float64))
        self.assertTrue(flipped)
        self.assertEqual(anchor_index, 0)

    def test_rank_drop_to_one_is_preserved(self) -> None:
        local_object = gate6_builder.build_local_object(
            v_raw=[1.0, 0.0, 0.0],
            splus_raw=[1.0, 0.0, 0.0],
            sminus_raw=[1.0, 0.0, 0.0],
        )
        self.assertEqual(local_object["rank_local"], 1)
        self.assertTrue(local_object["flags"]["rank_drop_to_1"])

    def test_reconstruction_matches_normalized_observables(self) -> None:
        local_object = gate6_builder.build_local_object(
            v_raw=[1.0, 0.0, 0.0],
            splus_raw=[1.0, 1.0, 0.0],
            sminus_raw=[1.0, 0.0, 1.0],
        )
        np.testing.assert_allclose(
            local_object["reconstruction_v"],
            local_object["normalized_v"],
            atol=1e-10,
        )
        np.testing.assert_allclose(
            local_object["reconstruction_splus"],
            local_object["normalized_splus"],
            atol=1e-10,
        )
        np.testing.assert_allclose(
            local_object["reconstruction_sminus"],
            local_object["normalized_sminus"],
            atol=1e-10,
        )

    def test_builder_rerun_is_deterministic_and_compat_uses_local8_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            samples_root = make_sample_root(tmp_dir)
            out_a = tmp_dir / "out_a"
            out_b = tmp_dir / "out_b"
            builder_script = REPO_ROOT / "tools" / "build_gate6_native_local_span.py"

            for out_dir in (out_a, out_b):
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(builder_script),
                        "--samples-root",
                        str(samples_root),
                        "--all-samples",
                        "--out-dir",
                        str(out_dir),
                    ],
                    cwd=str(REPO_ROOT),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

            compat_a = json.loads((out_a / "compatibility_input.json").read_text(encoding="utf-8"))
            compat_b = json.loads((out_b / "compatibility_input.json").read_text(encoding="utf-8"))
            self.assertEqual(compat_a, compat_b)

            token_step = compat_a["samples"][0]["token_steps"][0]
            self.assertIn("compat_vectors", token_step)
            self.assertNotIn("V_8d", token_step)
            self.assertEqual(
                sorted(token_step["compat_vectors"].keys()),
                ["Sminus_local8", "Splus_local8", "V_local8"],
            )

            manifest_a = json.loads((out_a / "manifest.json").read_text(encoding="utf-8"))
            manifest_b = json.loads((out_b / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                manifest_a["boundary_outcome_counts"]["materialized_rank3"],
                manifest_a["n_token_steps_total"],
            )
            for field in ("run_id",):
                manifest_a.pop(field, None)
                manifest_b.pop(field, None)
            self.assertEqual(manifest_a, manifest_b)

            with np.load(out_a / "native_object_arrays.npz") as arrays_a, np.load(
                out_b / "native_object_arrays.npz"
            ) as arrays_b:
                for key in arrays_a.files:
                    np.testing.assert_allclose(arrays_a[key], arrays_b[key], atol=0.0)


if __name__ == "__main__":
    unittest.main()
