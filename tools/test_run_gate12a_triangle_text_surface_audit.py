#!/usr/bin/env python3
"""Regression tests for Gate12A triangle text-surface audit."""

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
import run_gate12a_triangle_text_surface_audit as text_audit


def make_basis(columns: list[list[float]], *, d_model: int, r_max: int) -> np.ndarray:
    basis = np.zeros((d_model, r_max), dtype=np.float64)
    for index, column in enumerate(columns):
        basis[:, index] = np.asarray(column, dtype=np.float64)
    return basis


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class Gate12ATriangleTextSurfaceAuditTest(unittest.TestCase):
    def test_run_triangle_text_surface_audit_emits_joined_rows_and_extremes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gate12a_run_dir = self._build_gate12a_identity_run(root)
            gate8_execution_dir = self._build_gate8_execution_fixture(root)
            out_dir = root / "triangle_text_surface_audit"

            result = text_audit.run_triangle_text_surface_audit(
                gate12a_dir=gate12a_run_dir,
                gate8_execution_dir=gate8_execution_dir,
                out_dir=out_dir,
                top_k=1,
            )

            self.assertEqual(result["status"]["defined_triangle_joined_count"], 1)
            joined = result["joined_rows"][0]
            self.assertEqual(joined["sample_id"], "a")
            self.assertEqual(joined["relation_kind_path"], ["residual_chord", "trusted_tree", "trusted_tree"])
            self.assertEqual(joined["anchor_qualified_path"], [True, False, False])
            self.assertEqual(joined["compatibility_gap_path_summary"]["max"], 0.0)
            self.assertEqual(joined["prompt_text"], "Prompt surface")
            self.assertEqual(joined["answer_text"], "Answer surface")
            self.assertEqual(joined["support_anchor_text"], "Support anchor surface")
            self.assertEqual(joined["conflict_anchor_text"], "")

            extremes_rows = [
                json.loads(line)
                for line in (out_dir / text_audit.DEFAULT_EXTREMES).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(extremes_rows), 2)
            self.assertEqual({row["extreme_kind"] for row in extremes_rows}, {"flattest", "most_distorted"})

            read_text = (out_dir / text_audit.DEFAULT_READ).read_text(encoding="utf-8")
            self.assertIn("Gate12A Triangle Text-Surface Audit", read_text)
            self.assertIn("Prompt surface", read_text)
            self.assertIn("Answer surface", read_text)

    def _build_gate12a_identity_run(self, root: Path) -> Path:
        node_dir = root / "node_family"
        relation_dir = root / "relation_family"
        out_dir = root / "gate12a_identity"
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

        gate12a.run_discrete_connection_audit(
            node_artifact_dir=node_dir,
            relation_seed_dir=relation_dir,
            out_dir=out_dir,
        )
        return out_dir

    def _build_gate8_execution_fixture(self, root: Path) -> Path:
        gate8_dir = root / "gate8_execution"
        sample_dir = gate8_dir / "samples" / "a"
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "prompt.txt").write_text("Prompt surface\n", encoding="utf-8")
        (sample_dir / "answer.txt").write_text("Answer surface\n", encoding="utf-8")
        (sample_dir / "support_anchor.txt").write_text("Support anchor surface\n", encoding="utf-8")
        return gate8_dir


if __name__ == "__main__":
    unittest.main()
