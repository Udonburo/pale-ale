#!/usr/bin/env python3
"""Regression tests for Gate12B observer-relative coarse-grained closure."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import run_gate12b_observer_relative_coarse_grained_closure as gate12b


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class Gate12BObserverRelativeCoarseGrainedClosureTest(unittest.TestCase):
    def test_run_gate12b_emits_observer_scale_and_gauge_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gate12a_dir = self._build_gate12a_fixture(root)
            out_dir = root / "gate12b_observer_relative"

            result = gate12b.run_observer_relative_coarse_grained_closure(
                gate12a_dir=gate12a_dir,
                out_dir=out_dir,
                flat_quantile=0.25,
                high_quantile=0.75,
                top_k=1,
                min_observer_support=2,
            )

            self.assertEqual(result["status"]["defined_triangle_count"], 4)
            self.assertEqual(result["status"]["flat_count"], 1)
            self.assertEqual(result["status"]["tense_count"], 2)
            self.assertEqual(result["status"]["high_tension_count"], 1)
            self.assertTrue(result["status"]["gauge_arrays_available"])
            self.assertEqual(result["status"]["gauge_unstable_check_count"], 0)
            self.assertGreater(result["status"]["observer_scale_matrix_row_count"], 0)

            matrix_json = json.loads((out_dir / gate12b.DEFAULT_MATRIX_JSON).read_text(encoding="utf-8"))
            self.assertEqual(matrix_json["observer_modes"], list(gate12b.OBSERVER_MODES))
            all_residual_band_rows = [
                row
                for row in matrix_json["matrix_rows"]
                if row["observer"] == "all_edges" and row["scale"] == "residual_quantile_band"
            ]
            self.assertEqual({row["scale_key"] for row in all_residual_band_rows}, {"flat", "tense", "high_tension"})

            candidates = [
                json.loads(line)
                for line in (out_dir / gate12b.DEFAULT_INVARIANT_CANDIDATES).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual({row["candidate_kind"] for row in candidates}, {"flat_observer_scale_stable", "high_tension_observer_scale_stable"})
            self.assertTrue(all(row["observer_support_count"] >= 2 for row in candidates))
            self.assertTrue(all(row["scale_support_count"] >= 2 for row in candidates))
            self.assertTrue(all(row["coarse_scale_support_count"] >= 1 for row in candidates))

            with open(out_dir / gate12b.DEFAULT_GAUGE_MATRIX, "r", encoding="utf-8", newline="") as handle:
                gauge_rows = list(csv.DictReader(handle))
            self.assertTrue(gauge_rows)
            self.assertTrue(all(row["stable"] == "True" for row in gauge_rows))
            self.assertTrue(all(row["gauge_transform"] == gate12b.GAUGE_TRANSFORM_WITH_ARRAYS for row in gauge_rows))

            gauge_candidates = [
                json.loads(line)
                for line in (out_dir / gate12b.DEFAULT_GAUGE_CANDIDATES).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(gauge_candidates), 2)
            self.assertTrue(all(row["candidate_status"] == "gauge_stable_candidate" for row in gauge_candidates))

            manifest = json.loads((out_dir / gate12b.DEFAULT_MANIFEST).read_text(encoding="utf-8"))
            self.assertEqual(manifest["method_id"], gate12b.METHOD_ID)
            self.assertEqual(manifest["secondary_audit_mode"], gate12b.SECONDARY_AUDIT_MODE)
            self.assertIn(gate12b.DEFAULT_GAUGE_SUMMARY, manifest["paths"])

            read_text = (out_dir / gate12b.DEFAULT_READ).read_text(encoding="utf-8")
            self.assertIn("Gate12B Observer-Relative Coarse-Grained Closure", read_text)
            self.assertIn("H1-D", read_text)
            self.assertIn("basis-preserving local reparameterizations", read_text)

            checksums = json.loads((out_dir / gate12b.DEFAULT_CHECKSUMS).read_text(encoding="utf-8"))
            self.assertIn(gate12b.DEFAULT_MATRIX_CSV, checksums)
            self.assertIn(gate12b.DEFAULT_GAUGE_CANDIDATES, checksums)

    def test_registry_only_fallback_does_not_promote_gauge_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gate12a_dir = self._build_gate12a_fixture(root, include_arrays=False)
            out_dir = root / "gate12b_observer_relative_no_arrays"

            result = gate12b.run_observer_relative_coarse_grained_closure(
                gate12a_dir=gate12a_dir,
                out_dir=out_dir,
                flat_quantile=0.25,
                high_quantile=0.75,
                top_k=1,
                min_observer_support=2,
            )

            self.assertFalse(result["status"]["gauge_arrays_available"])
            self.assertEqual(result["status"]["gauge_total_check_count"], 0)
            self.assertEqual(result["status"]["gauge_variant_signature_candidate_count"], 0)
            self.assertEqual(result["gauge_candidates"], [])

            gauge_summary = json.loads((out_dir / gate12b.DEFAULT_GAUGE_SUMMARY).read_text(encoding="utf-8"))
            self.assertFalse(gauge_summary["nontrivial_transform_evaluated"])
            self.assertEqual(gauge_summary["skipped_reason"], "transport_operator_arrays_missing")

            gauge_candidates_text = (out_dir / gate12b.DEFAULT_GAUGE_CANDIDATES).read_text(encoding="utf-8")
            self.assertEqual(gauge_candidates_text, "")

    def test_cycle_motif_observer_mode_set_adds_ordered_relation_views(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gate12a_dir = self._build_gate12a_fixture(root)
            out_dir = root / "gate12b_cycle_motif_expansion"

            result = gate12b.run_observer_relative_coarse_grained_closure(
                gate12a_dir=gate12a_dir,
                out_dir=out_dir,
                flat_quantile=0.25,
                high_quantile=0.75,
                top_k=1,
                min_observer_support=3,
                min_scale_support=2,
                observer_mode_set="cycle_motif_expansion_v1",
            )

            self.assertEqual(result["manifest"]["observer_mode_set"], "cycle_motif_expansion_v1")
            self.assertIn("residual_first_leg", result["manifest"]["observer_modes"])
            self.assertIn("residual_second_leg", result["manifest"]["observer_modes"])
            self.assertIn("residual_third_leg", result["manifest"]["observer_modes"])

            matrix_json = json.loads((out_dir / gate12b.DEFAULT_MATRIX_JSON).read_text(encoding="utf-8"))
            self.assertEqual(matrix_json["observer_mode_set"], "cycle_motif_expansion_v1")
            self.assertIn("residual_first_leg", matrix_json["observer_modes"])

    def test_rejects_out_dir_aliasing_source_gate12a_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gate12a_dir = self._build_gate12a_fixture(root)
            original_manifest = json.loads((gate12a_dir / gate12b.DEFAULT_MANIFEST).read_text(encoding="utf-8"))

            with self.assertRaisesRegex(ValueError, "same directory as gate12a_dir"):
                gate12b.run_observer_relative_coarse_grained_closure(
                    gate12a_dir=gate12a_dir,
                    out_dir=gate12a_dir / ".",
                    flat_quantile=0.25,
                    high_quantile=0.75,
                    top_k=1,
                    min_observer_support=2,
                )

            preserved_manifest = json.loads((gate12a_dir / gate12b.DEFAULT_MANIFEST).read_text(encoding="utf-8"))
            self.assertEqual(preserved_manifest, original_manifest)
            self.assertEqual(preserved_manifest["schema_version"], "gate12a_discrete_connection_v1")

    def test_rejects_out_dir_nested_under_source_gate12a_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gate12a_dir = self._build_gate12a_fixture(root)
            nested_out_dir = gate12a_dir / "gate12b_child"
            original_manifest = json.loads((gate12a_dir / gate12b.DEFAULT_MANIFEST).read_text(encoding="utf-8"))

            with self.assertRaisesRegex(ValueError, "inside gate12a_dir"):
                gate12b.run_observer_relative_coarse_grained_closure(
                    gate12a_dir=gate12a_dir,
                    out_dir=nested_out_dir,
                    flat_quantile=0.25,
                    high_quantile=0.75,
                    top_k=1,
                    min_observer_support=2,
                )

            self.assertFalse(nested_out_dir.exists())
            preserved_manifest = json.loads((gate12a_dir / gate12b.DEFAULT_MANIFEST).read_text(encoding="utf-8"))
            self.assertEqual(preserved_manifest, original_manifest)
            self.assertEqual(preserved_manifest["schema_version"], "gate12a_discrete_connection_v1")

    def _build_gate12a_fixture(self, root: Path, *, include_arrays: bool = True) -> Path:
        gate12a_dir = root / "gate12a_fixture"
        write_json(
            gate12a_dir / gate12b.DEFAULT_MANIFEST,
            {
                "run_id": "gate12a_fixture",
                "schema_version": "gate12a_discrete_connection_v1",
                "code_git_commit": "fixture-commit",
            },
        )

        cycle_rows: list[dict] = []
        holonomy_rows: list[dict] = []
        transport_rows: list[dict] = []
        matrices: list[np.ndarray] = []
        residuals = [0.10, 0.20, 0.80, 1.00]
        relation_patterns = [
            ["residual_chord", "residual_chord", "trusted_tree"],
            ["residual_chord", "trusted_tree", "trusted_tree"],
            ["residual_chord", "trusted_tree", "trusted_tree"],
            ["residual_chord", "residual_chord", "trusted_tree"],
        ]
        anchor_patterns = [
            [True, True, False],
            [True, False, False],
            [False, True, False],
            [True, True, False],
        ]
        for cycle_index, residual in enumerate(residuals):
            nodes = [f"c{cycle_index}:a", f"c{cycle_index}:b", f"c{cycle_index}:c", f"c{cycle_index}:a"]
            edges = [f"e{cycle_index}_{edge_index}" for edge_index in range(3)]
            cycle_rows.append(
                {
                    "cycle_id": f"triangle:{cycle_index:06d}",
                    "base_node_id": nodes[0],
                    "edge_id_path": edges,
                    "node_id_path": nodes,
                    "cycle_length": 3,
                    "cycle_status": "admissible_explicit_triangle",
                }
            )
            holonomy_rows.append(
                {
                    "cycle_id": f"triangle:{cycle_index:06d}",
                    "base_node_id": nodes[0],
                    "holonomy_rank": 2,
                    "holonomy_residual_fro": residual,
                    "holonomy_status": "defined",
                }
            )
            for edge_index, edge_id in enumerate(edges):
                transport_rows.append(
                    {
                        "edge_id": edge_id,
                        "source_node_id": nodes[edge_index],
                        "target_node_id": nodes[edge_index + 1],
                        "relation_kind": relation_patterns[cycle_index][edge_index],
                        "anchor_qualified": anchor_patterns[cycle_index][edge_index],
                        "anchor_relation_id": f"anchor_{cycle_index}_{edge_index}" if anchor_patterns[cycle_index][edge_index] else "",
                        "source_rank": 2,
                        "target_rank": 2,
                        "overlap_rank": 2,
                        "transport_case": "equal_rank_orthogonal",
                        "operator_array_index": len(matrices),
                        "compatibility_gap_fro": 0.0,
                        "transport_level_compatibility_status": "compatible",
                    }
                )
                if edge_index == 0:
                    matrices.append(np.asarray([[1.0 + residual, 0.0], [0.0, 1.0]], dtype=np.float64))
                else:
                    matrices.append(np.eye(2, dtype=np.float64))

        write_jsonl(gate12a_dir / gate12b.DEFAULT_CYCLE_REGISTRY, cycle_rows)
        write_jsonl(gate12a_dir / gate12b.DEFAULT_HOLONOMY_REGISTRY, holonomy_rows)
        write_jsonl(gate12a_dir / gate12b.DEFAULT_TRANSPORT_REGISTRY, transport_rows)
        if include_arrays:
            np.savez(
                gate12a_dir / gate12b.DEFAULT_TRANSPORT_ARRAYS,
                transport_matrix_local=np.asarray(matrices, dtype=np.float64),
            )
        return gate12a_dir


if __name__ == "__main__":
    unittest.main()
