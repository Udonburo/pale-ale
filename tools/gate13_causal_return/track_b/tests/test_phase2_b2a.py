from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.gate13_causal_return.phase2_common import sha256_file
from tools.gate13_causal_return.track_b.phase2_b2a import (
    analytic_reconstruction_tolerances,
    normalized_legacy_scalar,
    primary_scalar_pairs,
    reconstruct_edge_holonomy,
    reconstruction_integrity,
    singular_spectrum_distance,
)
from tools.gate13_causal_return.track_b.source_sufficiency import (
    SOURCE_ARTIFACTS,
    assess_source_sufficiency,
)


class B2aPrimitiveTests(unittest.TestCase):
    def test_reconstruction_identity_and_analytic_tolerance(self) -> None:
        edges = [np.eye(3), np.eye(3), np.eye(3)]
        reconstructed = reconstruct_edge_holonomy(edges, rank_tolerance=1.0e-8)
        result = reconstruction_integrity(reconstructed, np.eye(3), 0.0)
        self.assertEqual(result["status"], "PASS")
        self.assertGreater(analytic_reconstruction_tolerances(3)["matrix_atol_scale"], 0.0)

    def test_rank_deficient_edge_is_not_completed(self) -> None:
        with self.assertRaisesRegex(ValueError, "RANK_DEFICIENT"):
            reconstruct_edge_holonomy(
                [np.eye(2), np.diag([1.0, 0.0]), np.eye(2)],
                rank_tolerance=1.0e-8,
            )

    def test_primary_matching_is_deterministic_one_to_one(self) -> None:
        rows = [
            {"run_id": "r", "rank": 2, "cycle_id": f"c{index}", "legacy_scalar": 0.01 * index}
            for index in range(7)
        ]
        first = primary_scalar_pairs(rows, bin_width=0.05)
        second = primary_scalar_pairs(list(reversed(rows)), bin_width=0.05)
        self.assertEqual(first, second)
        used = [cycle_id for pair in first for cycle_id in pair]
        self.assertEqual(len(used), len(set(used)))

    def test_spectrum_distance_and_scalar_range(self) -> None:
        self.assertEqual(singular_spectrum_distance([1.0, 0.5], [1.0, 0.5]), 0.0)
        self.assertEqual(normalized_legacy_scalar(0.0, 2), 0.0)
        with self.assertRaises(ValueError):
            normalized_legacy_scalar(100.0, 2)


class SourceSufficiencyTests(unittest.TestCase):
    def test_missing_referenced_sample_rows_fail_before_outcome_read(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "gate12a"
            source.mkdir()
            for filename in SOURCE_ARTIFACTS.values():
                (source / filename).write_text("{}\n", encoding="utf-8")
            node_manifest = root / "node_manifest.json"
            node_manifest.write_text(
                '{"source_gate8_execution_dir":"runs/missing","source_local_object_discipline":"gate9a_reconstruction_v1"}\n',
                encoding="utf-8",
            )
            run = {
                "run_id": "r",
                "source_manifest_path": str(source / "manifest.json"),
                "source_node_manifest_path": str(node_manifest),
                "source_node_manifest_sha256": sha256_file(node_manifest),
                "referenced_gate8_sample_source_path": str(root / "missing_gate8"),
            }
            for field, filename in SOURCE_ARTIFACTS.items():
                run[field] = sha256_file(source / filename)
            lock = {
                "source_runs": [run],
                "source_sufficiency": {
                    "minimum_source_sufficient_runs": 1,
                    "deterministic_split_key": "sha256(run_id|node_id|sample_id) parity",
                },
            }
            result = assess_source_sufficiency(lock)
            self.assertEqual(result["status"], "SPLIT_HALF_SOURCE_UNAVAILABLE")
            self.assertEqual(result["B2A"], "NOT_EXECUTED")
            self.assertFalse(result["operator_outcomes_read"])


if __name__ == "__main__":
    unittest.main()
