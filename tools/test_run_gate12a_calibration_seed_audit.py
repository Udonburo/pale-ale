#!/usr/bin/env python3
"""Regression tests for Gate12A calibration / seed-audit helper."""

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

import run_gate12a_calibration_seed_audit as calibration
import run_gate12a_discrete_connection_audit as gate12a


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


class Gate12ACalibrationSeedAuditTest(unittest.TestCase):
    def test_run_calibration_seed_audit_emits_anchor_and_extreme_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gate12a_run_dir = self._build_gate12a_identity_run(root)
            out_dir = root / "calibration_seed_audit"

            result = calibration.run_calibration_seed_audit(
                gate12a_dir=gate12a_run_dir,
                out_dir=out_dir,
                top_k=1,
            )

            self.assertEqual(result["status"]["transport_relation_count"], 4)
            self.assertEqual(result["status"]["explicit_triangle_cycle_count"], 1)
            self.assertEqual(result["status"]["defined_triangle_holonomy_count"], 1)
            self.assertEqual(result["status"]["zero_overlap_count"], 1)
            self.assertEqual(result["status"]["triangles_with_any_anchor_count"], 1)
            self.assertEqual(result["status"]["triangles_with_all_anchor_count"], 0)

            quantile_rows = result["transport_quantiles"]
            overall = next(row for row in quantile_rows if row["subregime"] == "overall")
            self.assertEqual(overall["n"], 3)

            flattest_row = result["triangle_extremes"]["flattest"][0]
            self.assertEqual(flattest_row["cycle_id"], "triangle:000000")
            self.assertEqual(flattest_row["anchor_qualified_path"], [True, False, False])

            read_text = (out_dir / calibration.DEFAULT_READ).read_text(encoding="utf-8")
            self.assertIn("Transport Gap Quantiles", read_text)
            self.assertIn("Most Distorted Triangles", read_text)
            self.assertIn("triangle:000000", read_text)

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


if __name__ == "__main__":
    unittest.main()
