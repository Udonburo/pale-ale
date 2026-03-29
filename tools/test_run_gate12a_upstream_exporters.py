#!/usr/bin/env python3
"""Regression tests for Gate12A upstream exporter helpers."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import run_gate12a_discrete_connection_audit as gate12a
import run_gate12a_node_local_object_export as node_export
import run_gate12a_relation_seed_export as relation_export


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class Gate12AUpstreamExporterTest(unittest.TestCase):
    def test_export_node_family_reconstructs_token_and_anchor_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gate8_dir = self._build_gate8_execution_fixture(root)
            out_dir = root / "node_family_out"

            result = node_export.export_node_family(gate8_dir, out_dir)

            manifest = json.loads((out_dir / node_export.DEFAULT_MANIFEST).read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], node_export.SCHEMA_VERSION)
            self.assertEqual(manifest["n_nodes_total"], 3)

            node_ids = [row["node_id"] for row in result["node_rows"]]
            self.assertEqual(
                node_ids,
                [
                    "sample_000001:answer_state",
                    "sample_000001:support_chunk",
                    "sample_000001:token_state_0000",
                ],
            )
            self.assertTrue(all(row["local_object_status"] == "defined" for row in result["node_rows"]))

            with np.load(out_dir / node_export.DEFAULT_NODE_ARRAYS) as npz_handle:
                basis_factor = np.asarray(npz_handle["basis_factor"], dtype=np.float64)
                rank_active = np.asarray(npz_handle["rank_active"], dtype=np.int64)
            self.assertEqual(basis_factor.shape, (3, 3, 3))
            np.testing.assert_array_equal(rank_active, np.asarray([1, 1, 1], dtype=np.int64))

    def test_export_relation_seed_family_and_audit_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gate8_dir = self._build_gate8_execution_fixture(root)
            gate9a_dir = self._build_gate9a_fixture(root)
            gate9k_dir = self._build_gate9k_fixture(root, gate9a_dir)

            node_family_dir = root / "node_family_out"
            relation_family_dir = root / "relation_family_out"
            audit_out_dir = root / "gate12a_run"

            node_export.export_node_family(gate8_dir, node_family_dir)
            relation_result = relation_export.export_relation_seed_family(gate9k_dir, relation_family_dir)

            relation_rows = relation_result["relation_rows"]
            self.assertEqual(len(relation_rows), 3)
            self.assertEqual(
                [row["relation_kind"] for row in relation_rows],
                ["trusted_tree", "trusted_tree", "residual_chord"],
            )
            self.assertEqual(
                [bool(row["anchor_qualified"]) for row in relation_rows],
                [True, True, False],
            )

            audit_result = gate12a.run_discrete_connection_audit(
                node_artifact_dir=node_family_dir,
                relation_seed_dir=relation_family_dir,
                out_dir=audit_out_dir,
            )

            self.assertEqual(audit_result["status"]["node_count"], 3)
            self.assertEqual(audit_result["status"]["transport_relation_count"], 3)
            self.assertEqual(audit_result["status"]["explicit_triangle_cycle_count"], 1)
            self.assertEqual(audit_result["status"]["defined_triangle_holonomy_count"], 1)
            self.assertEqual(
                audit_result["holonomy_rows"][0]["holonomy_status"],
                "defined",
            )
            self.assertAlmostEqual(
                float(audit_result["holonomy_rows"][0]["holonomy_residual_fro"]),
                0.0,
                places=10,
            )

            read_text = (audit_out_dir / gate12a.DEFAULT_READ).read_text(encoding="utf-8")
            self.assertIn("defined holonomy rows emitted: `1`", read_text)
            self.assertIn("compatibility_gap_fro := ||I_k - Sigma_k||_F", read_text)

    def _build_gate8_execution_fixture(self, root: Path) -> Path:
        gate8_dir = root / "gate8_execution"
        gate6_dir = gate8_dir / "gate6_native"
        sample_dir = gate8_dir / "samples" / "sample_000001"

        write_json(
            gate8_dir / "manifest.json",
            {
                "run_id": "gate8_fixture",
                "code_git_commit": "fixture-gate8",
                "rendering_family_id": "fixture_rendering_v1",
            },
        )
        write_jsonl(
            gate8_dir / "sample_registry.jsonl",
            [
                {
                    "execution_sample_id": 1,
                    "benchmark_sample_id": "bench-1",
                    "cell_id": "cell-alpha",
                    "world_id": "world-1",
                    "world_type": "fixture_world",
                    "answer_target_type": "fixture_answer",
                    "quietness_pair_id": "quiet-1",
                    "rendering_family_id": "fixture_rendering_v1",
                }
            ],
        )
        write_jsonl(
            gate6_dir / "step_index.jsonl",
            [
                {
                    "sample_id": 1,
                    "step": 0,
                    "array_row_index": 0,
                    "token_text": "hello",
                    "label_token": 1,
                    "flags_compact": "none",
                }
            ],
        )

        basis = np.zeros((1, 3, 3), dtype=np.float64)
        basis[0, :, 0] = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        singular_values = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64)
        rank_local = np.asarray([1], dtype=np.int64)
        coords_local = np.zeros((1, 3, 3), dtype=np.float64)
        gate6_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            gate6_dir / "native_object_arrays.npz",
            basis=basis,
            coords_local=coords_local,
            singular_values=singular_values,
            rank_local=rank_local,
        )

        sample_dir.mkdir(parents=True, exist_ok=True)
        triplet_row = {
            "V_raw_native": [1.0, 0.0, 0.0],
            "Splus_raw_native": [1.0, 0.0, 0.0],
            "Sminus_raw_native": [1.0, 0.0, 0.0],
        }
        write_jsonl(sample_dir / "triplets.ndjson", [triplet_row])
        write_jsonl(sample_dir / "support_anchor_triplets.ndjson", [triplet_row])
        return gate8_dir

    def _build_gate9a_fixture(self, root: Path) -> Path:
        gate9a_dir = root / "gate9a_family"
        write_json(
            gate9a_dir / "manifest.json",
            {
                "run_id": "gate9a_fixture",
                "code_git_commit": "fixture-gate9a",
            },
        )
        write_jsonl(
            gate9a_dir / "edge_transport_registry.jsonl",
            [
                {
                    "edge_id": "edge_answer_support",
                    "edge_type": "support_anchor",
                    "source_node_id": "sample_000001:answer_state",
                    "target_node_id": "sample_000001:support_chunk",
                },
                {
                    "edge_id": "edge_support_token",
                    "edge_type": "support_anchor",
                    "source_node_id": "sample_000001:support_chunk",
                    "target_node_id": "sample_000001:token_state_0000",
                },
                {
                    "edge_id": "edge_token_answer",
                    "edge_type": "answer_projection",
                    "source_node_id": "sample_000001:token_state_0000",
                    "target_node_id": "sample_000001:answer_state",
                },
            ],
        )
        return gate9a_dir

    def _build_gate9k_fixture(self, root: Path, gate9a_dir: Path) -> Path:
        gate9k_dir = root / "gate9k_family"
        write_json(
            gate9k_dir / "manifest.json",
            {
                "run_id": "gate9k_fixture",
                "code_git_commit": "fixture-gate9k",
                "source_gate9a_dir": str(gate9a_dir.resolve()),
            },
        )
        write_jsonl(
            gate9k_dir / "trusted_tree_residual_chord_registry.jsonl",
            [
                {
                    "edge_id": "edge_answer_support",
                    "decomposition_role": "trusted_tree_candidate",
                },
                {
                    "edge_id": "edge_support_token",
                    "decomposition_role": "trusted_tree_candidate",
                },
                {
                    "edge_id": "edge_token_answer",
                    "decomposition_role": "residual_chord_candidate",
                },
            ],
        )
        return gate9k_dir


if __name__ == "__main__":
    unittest.main()
