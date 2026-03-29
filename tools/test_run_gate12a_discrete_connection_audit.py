#!/usr/bin/env python3
"""Regression tests for Gate12A discrete-connection audit."""

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


def make_basis(columns: list[list[float]], *, d_model: int, r_max: int) -> np.ndarray:
    basis = np.zeros((d_model, r_max), dtype=np.float64)
    for index, column in enumerate(columns):
        basis[:, index] = np.asarray(column, dtype=np.float64)
    return basis


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class RunGate12ADiscreteConnectionAuditTest(unittest.TestCase):
    def test_compute_transport_identity_equal_rank(self) -> None:
        basis = make_basis([[1.0, 0.0, 0.0]], d_model=3, r_max=2)
        source = gate12a.NodeLocalObject("a", "A", 0, 1, "defined", basis)
        target = gate12a.NodeLocalObject("b", "B", 1, 1, "defined", basis)

        row, matrix, singular_values, active_rank = gate12a.compute_transport_operator(
            source,
            target,
            tau_overlap_sv_min=1.0e-8,
            tau_transport_gap_fro=1.0e-6,
            r_max=2,
        )

        self.assertEqual(row["transport_case"], "equal_rank_orthogonal")
        self.assertEqual(row["transport_level_compatibility_status"], "compatible")
        self.assertEqual(active_rank, 1)
        self.assertAlmostEqual(float(row["compatibility_gap_fro"]), 0.0, places=10)
        self.assertAlmostEqual(float(singular_values[0]), 1.0, places=10)
        np.testing.assert_allclose(matrix[:1, :1], np.eye(1), atol=1.0e-10)

    def test_compute_transport_partial_isometry_for_rank_mismatch(self) -> None:
        source_basis = make_basis(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            d_model=3,
            r_max=2,
        )
        target_basis = make_basis([[1.0, 0.0, 0.0]], d_model=3, r_max=2)
        source = gate12a.NodeLocalObject("src", "SRC", 0, 2, "defined", source_basis)
        target = gate12a.NodeLocalObject("tgt", "TGT", 1, 1, "defined", target_basis)

        row, matrix, singular_values, active_rank = gate12a.compute_transport_operator(
            source,
            target,
            tau_overlap_sv_min=1.0e-8,
            tau_transport_gap_fro=1.0e-6,
            r_max=2,
        )

        self.assertEqual(row["transport_case"], "rank_mismatch_partial_isometry")
        self.assertEqual(row["transport_level_compatibility_status"], "compatible")
        self.assertEqual(active_rank, 1)
        self.assertAlmostEqual(float(singular_values[0]), 1.0, places=10)
        np.testing.assert_allclose(matrix[0, :2], np.asarray([1.0, 0.0]), atol=1.0e-10)

    def test_compute_transport_zero_overlap_is_zero_filled(self) -> None:
        source_basis = make_basis([[1.0, 0.0, 0.0]], d_model=3, r_max=2)
        target_basis = make_basis([[0.0, 1.0, 0.0]], d_model=3, r_max=2)
        source = gate12a.NodeLocalObject("src", "SRC", 0, 1, "defined", source_basis)
        target = gate12a.NodeLocalObject("tgt", "TGT", 1, 1, "defined", target_basis)

        row, matrix, singular_values, active_rank = gate12a.compute_transport_operator(
            source,
            target,
            tau_overlap_sv_min=1.0e-8,
            tau_transport_gap_fro=1.0e-6,
            r_max=2,
        )

        self.assertEqual(row["transport_case"], "undefined_zero_overlap")
        self.assertEqual(row["transport_level_compatibility_status"], "undefined")
        self.assertIsNone(row["compatibility_gap_fro"])
        self.assertEqual(active_rank, 0)
        np.testing.assert_allclose(singular_values, np.zeros((2,), dtype=np.float64), atol=1.0e-10)
        np.testing.assert_allclose(matrix, np.zeros((2, 2), dtype=np.float64), atol=1.0e-10)

    def test_run_discrete_connection_audit_emits_defined_identity_triangle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._build_identity_fixture(Path(tmpdir))
            result = gate12a.run_discrete_connection_audit(
                node_artifact_dir=paths["node_dir"],
                relation_seed_dir=paths["relation_dir"],
                out_dir=paths["out_dir"],
            )

            self.assertEqual(result["status"]["explicit_triangle_cycle_count"], 1)
            self.assertEqual(result["status"]["defined_triangle_holonomy_count"], 1)
            holonomy_row = result["holonomy_rows"][0]
            self.assertEqual(holonomy_row["holonomy_status"], "defined")
            self.assertAlmostEqual(float(holonomy_row["holonomy_residual_fro"]), 0.0, places=10)

            manifest = json.loads((paths["out_dir"] / gate12a.DEFAULT_MANIFEST).read_text(encoding="utf-8"))
            self.assertEqual(manifest["method_id"], gate12a.METHOD_ID)
            self.assertEqual(
                manifest["input_manifest_refs"]["explicit_relation_seed_family"]["relation_seed_mode"],
                gate12a.RELATION_SEED_MODE,
            )

            checksums = json.loads((paths["out_dir"] / gate12a.DEFAULT_CHECKSUMS).read_text(encoding="utf-8"))
            self.assertIn(gate12a.DEFAULT_TRANSPORT_ARRAYS, checksums)
            self.assertIn(gate12a.DEFAULT_HOLONOMY_ARRAYS, checksums)

            read_text = (paths["out_dir"] / gate12a.DEFAULT_READ).read_text(encoding="utf-8")
            self.assertIn("zero-overlap edges emitted: `1`", read_text)
            self.assertIn("defined holonomy rows emitted: `1`", read_text)
            self.assertIn("compatibility_gap_fro := ||I_k - Sigma_k||_F", read_text)
            self.assertIn("defined holonomy rows at or below threshold: `1`", read_text)

            cycle_rows = [
                json.loads(line)
                for line in (paths["out_dir"] / gate12a.DEFAULT_TRIANGLE_REGISTRY).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(cycle_rows[0]["edge_id_path"], ["edge_a", "edge_m", "edge_z"])

    def test_run_discrete_connection_audit_marks_partial_triangle_as_equal_rank_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._build_partial_rank_triangle_fixture(Path(tmpdir))
            result = gate12a.run_discrete_connection_audit(
                node_artifact_dir=paths["node_dir"],
                relation_seed_dir=paths["relation_dir"],
                out_dir=paths["out_dir"],
            )

            self.assertEqual(result["status"]["explicit_triangle_cycle_count"], 1)
            self.assertEqual(result["status"]["defined_triangle_holonomy_count"], 0)
            holonomy_row = result["holonomy_rows"][0]
            self.assertEqual(holonomy_row["holonomy_status"], "equal_rank_required")
            self.assertIsNone(holonomy_row["holonomy_residual_fro"])

    def test_run_discrete_connection_audit_reindexes_sorted_node_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._build_unsorted_node_fixture(Path(tmpdir))
            gate12a.run_discrete_connection_audit(
                node_artifact_dir=paths["node_dir"],
                relation_seed_dir=paths["relation_dir"],
                out_dir=paths["out_dir"],
            )

            output_rows = [
                json.loads(line)
                for line in (paths["out_dir"] / gate12a.DEFAULT_NODE_REGISTRY).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual([row["node_id"] for row in output_rows], ["a", "b", "c"])
            self.assertEqual([row["basis_array_index"] for row in output_rows], [0, 1, 2])

    def test_run_discrete_connection_audit_rejects_nondefined_node_with_positive_rank(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._build_invalid_undefined_node_fixture(Path(tmpdir))
            with self.assertRaisesRegex(ValueError, "non-defined node"):
                gate12a.run_discrete_connection_audit(
                    node_artifact_dir=paths["node_dir"],
                    relation_seed_dir=paths["relation_dir"],
                    out_dir=paths["out_dir"],
                )

    def _build_identity_fixture(self, root: Path) -> dict[str, Path]:
        node_dir = root / "node_family"
        relation_dir = root / "relation_family"
        out_dir = root / "runs" / "gate12a_identity"
        node_dir.mkdir(parents=True)
        relation_dir.mkdir(parents=True)

        d_model = 3
        r_max = 2
        node_rows = [
            {"node_id": "b", "node_label": "Node B", "basis_array_index": 1, "projector_rank": 1, "local_object_status": "defined"},
            {"node_id": "a", "node_label": "Node A", "basis_array_index": 0, "projector_rank": 1, "local_object_status": "defined"},
            {"node_id": "c", "node_label": "Node C", "basis_array_index": 2, "projector_rank": 1, "local_object_status": "defined"},
            {"node_id": "d", "node_label": "Node D", "basis_array_index": 3, "projector_rank": 1, "local_object_status": "defined"},
        ]
        basis_factor = np.asarray(
            [
                make_basis([[1.0, 0.0, 0.0]], d_model=d_model, r_max=r_max),
                make_basis([[1.0, 0.0, 0.0]], d_model=d_model, r_max=r_max),
                make_basis([[1.0, 0.0, 0.0]], d_model=d_model, r_max=r_max),
                make_basis([[0.0, 1.0, 0.0]], d_model=d_model, r_max=r_max),
            ],
            dtype=np.float64,
        )
        rank_active = np.asarray([1, 1, 1, 1], dtype=np.int64)
        write_json(
            node_dir / gate12a.DEFAULT_MANIFEST,
            {
                "run_id": "node_fixture_identity",
                "schema_version": "fixture_v1",
                "code_git_commit": "fixture-node",
            },
        )
        write_jsonl(node_dir / gate12a.DEFAULT_NODE_REGISTRY, node_rows)
        np.savez(node_dir / gate12a.DEFAULT_NODE_ARRAYS, basis_factor=basis_factor, rank_active=rank_active)

        relation_rows = [
            {"edge_id": "edge_z", "source_node_id": "a", "target_node_id": "b", "relation_kind": "trusted_tree", "anchor_qualified": False, "anchor_relation_id": ""},
            {"edge_id": "edge_m", "source_node_id": "b", "target_node_id": "c", "relation_kind": "trusted_tree", "anchor_qualified": False, "anchor_relation_id": ""},
            {"edge_id": "edge_a", "source_node_id": "c", "target_node_id": "a", "relation_kind": "residual_chord", "anchor_qualified": True, "anchor_relation_id": "anchor_ca"},
            {"edge_id": "edge_ad", "source_node_id": "a", "target_node_id": "d", "relation_kind": "residual_chord", "anchor_qualified": False, "anchor_relation_id": ""},
        ]
        write_json(
            relation_dir / gate12a.DEFAULT_MANIFEST,
            {
                "run_id": "relation_fixture_identity",
                "schema_version": "fixture_v1",
                "relation_seed_mode": gate12a.RELATION_SEED_MODE,
            },
        )
        write_jsonl(relation_dir / "explicit_relation_seed_registry.jsonl", relation_rows)
        return {"node_dir": node_dir, "relation_dir": relation_dir, "out_dir": out_dir}

    def _build_partial_rank_triangle_fixture(self, root: Path) -> dict[str, Path]:
        node_dir = root / "node_family"
        relation_dir = root / "relation_family"
        out_dir = root / "runs" / "gate12a_partial_rank"
        node_dir.mkdir(parents=True)
        relation_dir.mkdir(parents=True)

        d_model = 3
        r_max = 2
        node_rows = [
            {"node_id": "a", "node_label": "Node A", "basis_array_index": 0, "projector_rank": 1, "local_object_status": "defined"},
            {"node_id": "b", "node_label": "Node B", "basis_array_index": 1, "projector_rank": 2, "local_object_status": "defined"},
            {"node_id": "c", "node_label": "Node C", "basis_array_index": 2, "projector_rank": 1, "local_object_status": "defined"},
        ]
        basis_factor = np.asarray(
            [
                make_basis([[1.0, 0.0, 0.0]], d_model=d_model, r_max=r_max),
                make_basis([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], d_model=d_model, r_max=r_max),
                make_basis([[1.0, 0.0, 0.0]], d_model=d_model, r_max=r_max),
            ],
            dtype=np.float64,
        )
        rank_active = np.asarray([1, 2, 1], dtype=np.int64)
        write_json(
            node_dir / gate12a.DEFAULT_MANIFEST,
            {
                "run_id": "node_fixture_partial",
                "schema_version": "fixture_v1",
                "code_git_commit": "fixture-node",
            },
        )
        write_jsonl(node_dir / gate12a.DEFAULT_NODE_REGISTRY, node_rows)
        np.savez(node_dir / gate12a.DEFAULT_NODE_ARRAYS, basis_factor=basis_factor, rank_active=rank_active)

        relation_rows = [
            {"edge_id": "edge_ab", "source_node_id": "a", "target_node_id": "b", "relation_kind": "trusted_tree", "anchor_qualified": False, "anchor_relation_id": ""},
            {"edge_id": "edge_bc", "source_node_id": "b", "target_node_id": "c", "relation_kind": "residual_chord", "anchor_qualified": False, "anchor_relation_id": ""},
            {"edge_id": "edge_ca", "source_node_id": "c", "target_node_id": "a", "relation_kind": "trusted_tree", "anchor_qualified": False, "anchor_relation_id": ""},
        ]
        write_json(
            relation_dir / gate12a.DEFAULT_MANIFEST,
            {
                "run_id": "relation_fixture_partial",
                "schema_version": "fixture_v1",
                "relation_seed_mode": gate12a.RELATION_SEED_MODE,
            },
        )
        write_jsonl(relation_dir / "explicit_relation_seed_registry.jsonl", relation_rows)
        return {"node_dir": node_dir, "relation_dir": relation_dir, "out_dir": out_dir}

    def _build_unsorted_node_fixture(self, root: Path) -> dict[str, Path]:
        node_dir = root / "node_family"
        relation_dir = root / "relation_family"
        out_dir = root / "runs" / "gate12a_unsorted_nodes"
        node_dir.mkdir(parents=True)
        relation_dir.mkdir(parents=True)

        basis_factor = np.asarray(
            [
                make_basis([[1.0, 0.0, 0.0]], d_model=3, r_max=2),
                make_basis([[1.0, 0.0, 0.0]], d_model=3, r_max=2),
                make_basis([[1.0, 0.0, 0.0]], d_model=3, r_max=2),
            ],
            dtype=np.float64,
        )
        rank_active = np.asarray([1, 1, 1], dtype=np.int64)
        node_rows = [
            {"node_id": "c", "node_label": "Node C", "basis_array_index": 2, "projector_rank": 1, "local_object_status": "defined"},
            {"node_id": "b", "node_label": "Node B", "basis_array_index": 1, "projector_rank": 1, "local_object_status": "defined"},
            {"node_id": "a", "node_label": "Node A", "basis_array_index": 0, "projector_rank": 1, "local_object_status": "defined"},
        ]
        write_json(
            node_dir / gate12a.DEFAULT_MANIFEST,
            {
                "run_id": "node_fixture_unsorted",
                "schema_version": "fixture_v1",
                "code_git_commit": "fixture-node",
            },
        )
        write_jsonl(node_dir / gate12a.DEFAULT_NODE_REGISTRY, node_rows)
        np.savez(node_dir / gate12a.DEFAULT_NODE_ARRAYS, basis_factor=basis_factor, rank_active=rank_active)

        relation_rows = [
            {"edge_id": "edge_ab", "source_node_id": "a", "target_node_id": "b", "relation_kind": "trusted_tree", "anchor_qualified": False, "anchor_relation_id": ""},
            {"edge_id": "edge_bc", "source_node_id": "b", "target_node_id": "c", "relation_kind": "residual_chord", "anchor_qualified": False, "anchor_relation_id": ""},
            {"edge_id": "edge_ca", "source_node_id": "c", "target_node_id": "a", "relation_kind": "trusted_tree", "anchor_qualified": False, "anchor_relation_id": ""},
        ]
        write_json(
            relation_dir / gate12a.DEFAULT_MANIFEST,
            {
                "run_id": "relation_fixture_unsorted",
                "schema_version": "fixture_v1",
                "relation_seed_mode": gate12a.RELATION_SEED_MODE,
            },
        )
        write_jsonl(relation_dir / "explicit_relation_seed_registry.jsonl", relation_rows)
        return {"node_dir": node_dir, "relation_dir": relation_dir, "out_dir": out_dir}

    def _build_invalid_undefined_node_fixture(self, root: Path) -> dict[str, Path]:
        node_dir = root / "node_family"
        relation_dir = root / "relation_family"
        out_dir = root / "runs" / "gate12a_invalid_undefined"
        node_dir.mkdir(parents=True)
        relation_dir.mkdir(parents=True)

        basis_factor = np.asarray(
            [make_basis([[1.0, 0.0, 0.0]], d_model=3, r_max=2)],
            dtype=np.float64,
        )
        rank_active = np.asarray([1], dtype=np.int64)
        node_rows = [
            {
                "node_id": "a",
                "node_label": "Node A",
                "basis_array_index": 0,
                "projector_rank": 1,
                "local_object_status": "undefined_aux_basis_missing",
            }
        ]
        write_json(
            node_dir / gate12a.DEFAULT_MANIFEST,
            {
                "run_id": "node_fixture_invalid_undefined",
                "schema_version": "fixture_v1",
                "code_git_commit": "fixture-node",
            },
        )
        write_jsonl(node_dir / gate12a.DEFAULT_NODE_REGISTRY, node_rows)
        np.savez(node_dir / gate12a.DEFAULT_NODE_ARRAYS, basis_factor=basis_factor, rank_active=rank_active)

        write_json(
            relation_dir / gate12a.DEFAULT_MANIFEST,
            {
                "run_id": "relation_fixture_invalid_undefined",
                "schema_version": "fixture_v1",
                "relation_seed_mode": gate12a.RELATION_SEED_MODE,
            },
        )
        write_jsonl(relation_dir / "explicit_relation_seed_registry.jsonl", [])
        return {"node_dir": node_dir, "relation_dir": relation_dir, "out_dir": out_dir}


if __name__ == "__main__":
    unittest.main()
