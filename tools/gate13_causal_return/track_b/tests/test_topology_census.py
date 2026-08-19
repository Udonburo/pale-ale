from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.gate13_causal_return.track_b.topology_census import (
    census_source,
    graph_convention_declaration,
)


class TopologyCensusTests(unittest.TestCase):
    def write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    def make_fixture(self, root: Path) -> None:
        manifest = {
            "run_id": "fixture",
            "schema_version": "gate12a_discrete_connection_v1",
            "code_git_commit": "fixture",
            "graph_object_policy": "flat_artifact_only_v1",
            "cycle_mode": "explicit_triangle_only_v1",
            "tau_overlap_sv_min": 1.0e-8,
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        node_rows = [
            {
                "node_id": node,
                "basis_array_index": index,
                "projector_rank": 2,
                "local_object_status": "defined",
            }
            for index, node in enumerate(("A", "B", "C"))
        ]
        self.write_jsonl(root / "node_local_object_registry.jsonl", node_rows)
        basis = np.stack([np.eye(2), np.eye(2), np.eye(2)])
        np.savez(
            root / "node_local_object_arrays.npz",
            basis_factor=basis,
            rank_active=np.asarray([2, 2, 2]),
        )

        edge_specs = [("ab", "A", "B"), ("bc", "B", "C"), ("ca", "C", "A")]
        edge_rows = [
            {
                "edge_id": edge_id,
                "source_node_id": source,
                "target_node_id": target,
                "relation_kind": "residual_chord" if edge_id == "ca" else "trusted_tree",
                "source_rank": 2,
                "target_rank": 2,
                "transport_case": "equal_rank_orthogonal",
                "operator_array_index": index,
            }
            for index, (edge_id, source, target) in enumerate(edge_specs)
        ]
        self.write_jsonl(root / "transport_relation_registry.jsonl", edge_rows)
        np.savez(
            root / "transport_operator_arrays.npz",
            transport_matrix_local=np.stack([np.eye(2)] * 3),
            overlap_singular_values=np.stack([np.ones(2)] * 3),
            active_rank=np.asarray([2, 2, 2]),
        )
        self.write_jsonl(
            root / "explicit_triangle_cycle_registry.jsonl",
            [
                {
                    "cycle_id": "triangle:000000",
                    "base_node_id": "A",
                    "edge_id_path": ["ab", "bc", "ca"],
                    "node_id_path": ["A", "B", "C", "A"],
                    "cycle_length": 3,
                    "cycle_status": "admissible_explicit_triangle",
                }
            ],
        )
        self.write_jsonl(
            root / "triangle_holonomy_registry.jsonl",
            [
                {
                    "cycle_id": "triangle:000000",
                    "base_node_id": "A",
                    "holonomy_status": "defined",
                    "holonomy_residual_fro": 0.0,
                }
            ],
        )

    def test_fixture_census_preserves_convention_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_fixture(root)
            report = census_source(root)
        self.assertEqual(report["node_count"], 3)
        self.assertEqual(report["directed_edge_count"], 3)
        self.assertEqual(report["registered_triangle_count"], 1)
        self.assertEqual(report["registered_full_rank_path_count"], 1)
        self.assertIsNone(report["general_topology"]["beta_1"])

    def test_declaration_does_not_silently_choose_general_cycle_space(self) -> None:
        declaration = graph_convention_declaration()
        self.assertEqual(
            declaration["status"], "PARTIAL_AUTHORITY_REVIEW1_BLOCKER"
        )
        self.assertTrue(
            any(
                item.startswith("beta_1 formula")
                for item in declaration["unresolved"]
            )
        )


if __name__ == "__main__":
    unittest.main()
