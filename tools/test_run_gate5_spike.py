#!/usr/bin/env python3
"""Regression tests for run_gate5_spike.py provenance helpers."""

import json
import tempfile
import unittest
from pathlib import Path

import run_gate5_spike as gate5_runner


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


class RunGate5SpikeTests(unittest.TestCase):
    def test_gate6_provenance_sidecar_and_manifest_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            input_path = tmp_dir / "compatibility_input.json"
            boundary_manifest_path = tmp_dir / "manifest.json"
            out_dir = tmp_dir / "gate5_out"
            out_dir.mkdir(parents=True, exist_ok=True)
            gate5_manifest_path = out_dir / "manifest.json"

            compat_payload = {
                "metadata": {
                    "model_id": "test-model",
                    "model_revision": "rev-test",
                    "seed": 7,
                    "perm_r": 2000,
                    "primary_score": "E",
                    "proj_id": "gate6_native_local_span_local8_v1",
                    "splus_def_id": "gate6_local_span_coord_splus_v1",
                    "sminus_def_id": "gate6_local_span_coord_sminus_v1",
                    "script_sha256_extract": "x",
                    "script_sha256_eval": "y",
                    "script_sha256_gate6_builder": "z",
                    "boundary_origin": "gate6_native_local_span_local8_v1",
                    "compatibility_schema_id": "gate6_local8_compat_input_v1",
                    "local_object_method_id": "native_local_span_gate6a_v1",
                    "source_tensor_id": "triality_raw_native_v1",
                },
                "samples": [
                    {
                        "sample_id": 1,
                        "variant": "consistent",
                        "world_type": "unit_test",
                        "exact_token_match_ratio": 1.0,
                        "label_coverage_ratio": 1.0,
                        "triplets_sha256": "a",
                        "labels_sha256": "b",
                        "token_steps": [
                            {
                                "step": 0,
                                "absolute_pos": 10,
                                "answer_char_start": 0,
                                "answer_char_end": 1,
                                "token_id": 101,
                                "token_text": "A",
                                "label_token": 0,
                                "defect_span_id": None,
                                "baseline_logprob": -1.0,
                                "baseline_entropy": 2.0,
                                "compat_vectors": {
                                    "V_local8": [1.0] + [0.0] * 7,
                                    "Splus_local8": [0.0, 1.0] + [0.0] * 6,
                                    "Sminus_local8": [0.0, 0.0, 1.0] + [0.0] * 5,
                                },
                            }
                        ],
                    }
                ],
            }
            write_json(input_path, compat_payload)
            write_json(
                boundary_manifest_path,
                {
                    "schema_version": "gate6_native_local_span_artifacts_v1",
                    "method_id": "native_local_span_gate6a_v1",
                },
            )
            write_json(
                gate5_manifest_path,
                {
                    "run_id": "gate5_test",
                    "proj_id": "gate6_native_local_span_local8_v1",
                },
            )

            payload = json.loads(input_path.read_text(encoding="utf-8"))
            provenance = gate5_runner.build_boundary_provenance(input_path, payload)
            self.assertIsNotNone(provenance)
            updated_manifest = gate5_runner.attach_boundary_provenance(
                out_dir=out_dir,
                manifest_path=gate5_manifest_path,
                manifest=json.loads(gate5_manifest_path.read_text(encoding="utf-8")),
                provenance=provenance,
            )

            sidecar_path = out_dir / gate5_runner.BOUNDARY_PROVENANCE_SIDECAR
            self.assertTrue(sidecar_path.exists())
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            self.assertEqual(sidecar["boundary_origin"], "gate6_native_local_span_local8_v1")
            self.assertEqual(sidecar["compatibility_schema_id"], "gate6_local8_compat_input_v1")
            self.assertEqual(sidecar["canonical_input_path"], str(input_path.resolve()))
            self.assertEqual(
                sidecar["canonical_boundary_manifest_path"],
                str(boundary_manifest_path.resolve()),
            )

            persisted_manifest = json.loads(gate5_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(updated_manifest, persisted_manifest)
            self.assertEqual(
                persisted_manifest["boundary_input_provenance_sidecar"],
                str(sidecar_path.resolve()),
            )
            self.assertEqual(
                persisted_manifest["canonical_boundary_input_path"],
                str(input_path.resolve()),
            )
            self.assertEqual(
                persisted_manifest["canonical_boundary_manifest_path"],
                str(boundary_manifest_path),
            )
            self.assertEqual(
                persisted_manifest["compatibility_schema_id"],
                "gate6_local8_compat_input_v1",
            )


if __name__ == "__main__":
    unittest.main()
