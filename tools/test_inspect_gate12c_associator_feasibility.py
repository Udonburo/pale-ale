#!/usr/bin/env python3
"""Regression tests for Gate12C associator-feasibility preflight."""

from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import inspect_gate12c_associator_feasibility as gate12c


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def compute_transport(
    *,
    source_basis: np.ndarray,
    target_basis: np.ndarray,
    rank: int,
    r_max: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    overlap = np.asarray(target_basis[:, :rank].T @ source_basis[:, :rank], dtype=np.float64)
    u_matrix, singular_values, vt_matrix = np.linalg.svd(overlap, full_matrices=False)
    singular_padded = np.zeros((r_max,), dtype=np.float64)
    singular_padded[: singular_values.shape[0]] = singular_values
    active_rank = int(np.sum(singular_values > gate12c.DEFAULT_TAU_OVERLAP_SV_MIN))
    matrix = np.zeros((r_max, r_max), dtype=np.float64)
    if active_rank > 0:
        matrix[:rank, :rank] = u_matrix[:, :active_rank] @ vt_matrix[:active_rank, :]
    return matrix, singular_padded, active_rank


def make_cycle_bases(
    *,
    rank: int,
    cos_values: list[float],
    offset: int,
    d_model: int,
    r_max: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(cos_values) != rank:
        raise ValueError("cos_values length must match rank")
    basis_a = np.zeros((d_model, r_max), dtype=np.float64)
    basis_b = np.zeros((d_model, r_max), dtype=np.float64)
    for index, cos_value in enumerate(cos_values):
        basis_a[offset + index, index] = 1.0
        basis_b[offset + index, index] = float(cos_value)
        basis_b[offset + rank + index, index] = math.sqrt(max(0.0, 1.0 - float(cos_value) ** 2))
    basis_c = np.asarray(basis_a, dtype=np.float64).copy()
    return basis_a, basis_b, basis_c


def snapshot_files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class InspectGate12CAssociatorFeasibilityTest(unittest.TestCase):
    def test_missing_required_artifact_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gate12a_dir = self._build_gate12a_fixture(root, [self._stable_rank2_spec()])
            (gate12a_dir / gate12c.DEFAULT_HOLONOMY_REGISTRY).unlink()

            result = gate12c.run_associator_feasibility_preflight(
                gate12a_dir=gate12a_dir,
                out_dir=root / "out",
                min_eligible_cycles=1,
            )

            self.assertEqual(
                result["preflight"]["contract_feasibility_status"],
                gate12c.CONTRACT_MISSING_ARTIFACT,
            )
            self.assertIn(gate12c.DEFAULT_HOLONOMY_REGISTRY, result["preflight"]["missing_required_artifacts"])
            self.assertTrue((root / "out" / gate12c.DEFAULT_MANIFEST).exists())

    def test_rejects_source_output_directory_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gate12a_dir = self._build_gate12a_fixture(root, [self._stable_rank2_spec()])

            with self.assertRaisesRegex(ValueError, "same directory as gate12a_dir"):
                gate12c.run_associator_feasibility_preflight(
                    gate12a_dir=gate12a_dir,
                    out_dir=gate12a_dir / ".",
                    min_eligible_cycles=1,
                )

    def test_rejects_source_nested_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gate12a_dir = self._build_gate12a_fixture(root, [self._stable_rank2_spec()])
            nested_out_dir = gate12a_dir / "gate12c_child"

            with self.assertRaisesRegex(ValueError, "inside gate12a_dir"):
                gate12c.run_associator_feasibility_preflight(
                    gate12a_dir=gate12a_dir,
                    out_dir=nested_out_dir,
                    min_eligible_cycles=1,
                )
            self.assertFalse(nested_out_dir.exists())

    def test_overlap_reconstruction_from_basis_factor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result = self._run(root, [self._stable_rank2_spec()], min_eligible_cycles=1)
            diagnostics = result["preflight"]["edge_reconstruction_diagnostics"]

            self.assertEqual(diagnostics["reconstructed_edge_count"], 3)
            self.assertEqual(diagnostics["failed_edge_reconstruction_count"], 0)
            self.assertLessEqual(diagnostics["overlap_singular_value_max_abs_error"], 1.0e-12)

    def test_stored_singular_spectrum_consistency(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gate12a_dir = self._build_gate12a_fixture(root, [self._stable_rank2_spec()])
            arrays_path = gate12a_dir / gate12c.DEFAULT_TRANSPORT_ARRAYS
            with np.load(arrays_path) as handle:
                matrices = np.asarray(handle["transport_matrix_local"], dtype=np.float64)
                singular_values = np.asarray(handle["overlap_singular_values"], dtype=np.float64)
                active_rank = np.asarray(handle["active_rank"], dtype=np.int64)
            singular_values[0, 0] += 0.05
            np.savez(
                arrays_path,
                transport_matrix_local=matrices,
                overlap_singular_values=singular_values,
                active_rank=active_rank,
            )

            result = gate12c.run_associator_feasibility_preflight(
                gate12a_dir=gate12a_dir,
                out_dir=root / "out",
                min_eligible_cycles=1,
            )

            self.assertEqual(
                result["preflight"]["contract_feasibility_status"],
                gate12c.CONTRACT_RECONSTRUCTION_MISMATCH,
            )
            self.assertGreater(
                result["preflight"]["edge_reconstruction_diagnostics"]["failed_edge_reconstruction_count"],
                0,
            )

    def test_stored_polar_transport_consistency(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gate12a_dir = self._build_gate12a_fixture(root, [self._stable_rank2_spec()])
            arrays_path = gate12a_dir / gate12c.DEFAULT_TRANSPORT_ARRAYS
            with np.load(arrays_path) as handle:
                matrices = np.asarray(handle["transport_matrix_local"], dtype=np.float64)
                singular_values = np.asarray(handle["overlap_singular_values"], dtype=np.float64)
                active_rank = np.asarray(handle["active_rank"], dtype=np.int64)
            matrices[0, 0, 0] += 0.05
            np.savez(
                arrays_path,
                transport_matrix_local=matrices,
                overlap_singular_values=singular_values,
                active_rank=active_rank,
            )

            result = gate12c.run_associator_feasibility_preflight(
                gate12a_dir=gate12a_dir,
                out_dir=root / "out",
                min_eligible_cycles=1,
            )

            self.assertEqual(
                result["preflight"]["contract_feasibility_status"],
                gate12c.CONTRACT_RECONSTRUCTION_MISMATCH,
            )
            self.assertGreater(
                result["preflight"]["edge_reconstruction_diagnostics"]["transport_reconstruction_max_fro_error"],
                0.0,
            )

    def test_common_rank_cycle_census(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result = self._run(
                root,
                [
                    self._rank1_spec(),
                    self._stable_rank2_spec(),
                    self._stable_rank3_spec(),
                    self._stable_rank4_spec(),
                ],
                min_eligible_cycles=1,
            )
            census = result["preflight"]["cycle_census"]

            self.assertEqual(census["total_gate12a_residual_bearing_explicit_triangle_count"], 4)
            self.assertEqual(census["defined_equal_rank_triangle_count"], 4)
            self.assertEqual(census["common_rank_1_triangle_count"], 1)
            self.assertEqual(census["common_rank_2_triangle_count"], 1)
            self.assertEqual(census["common_rank_3_triangle_count"], 1)
            self.assertEqual(census["common_rank_ge_4_triangle_count"], 1)
            self.assertEqual(census["max_common_equal_rank"], 4)

    def test_rank1_nontrivial_probe_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result = self._run(root, [self._rank1_spec()], min_eligible_cycles=1)

            self.assertEqual(result["preflight"]["cut_census"]["probe_configuration_count"], 0)
            self.assertEqual(result["cut_rows"], [])
            self.assertEqual(
                result["preflight"]["empirical_surface_status"],
                gate12c.EMPIRICAL_FAIL_NO_NONTRIVIAL_EQUAL_RANK_CYCLE,
            )

    def test_rank2_nontrivial_q_enumeration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result = self._run(root, [self._stable_rank2_spec()], min_eligible_cycles=1)

            self.assertEqual(len(result["cut_rows"]), 3)
            self.assertEqual({row["q"] for row in result["cut_rows"]}, {1})

    def test_all_three_cyclic_roots_counted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result = self._run(root, [self._stable_rank2_spec()], min_eligible_cycles=1)

            self.assertEqual(
                {row["evaluation_root_node_id"] for row in result["cut_rows"]},
                {"stable2:a", "stable2:b", "stable2:c"},
            )
            self.assertEqual({row["root_rotation_index"] for row in result["cut_rows"]}, {0, 1, 2})

    def test_lexical_edge_id_order_not_used_as_traversal_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result = self._run(root, [self._stable_rank2_spec()], min_eligible_cycles=1)
            row = result["cycle_rows"][0]

            self.assertEqual(row["edge_id_path"], ["stable2:a_ca", "stable2:m_bc", "stable2:z_ab"])
            self.assertEqual(row["ordered_edge_id_path"], ["stable2:z_ab", "stable2:m_bc", "stable2:a_ca"])
            self.assertNotEqual(row["edge_id_path"], row["ordered_edge_id_path"])

    def test_ordinary_matrix_associativity_null(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result = self._run(root, [self._stable_rank2_spec()], min_eligible_cycles=1)
            ordinary = result["preflight"]["ordinary_associativity_null"]

            self.assertEqual(ordinary["ordinary_associator_failed_count"], 0)
            self.assertLessEqual(ordinary["ordinary_associator_max_fro"], 1.0e-12)

    def test_stable_split_gap_recognition(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result = self._run(root, [self._stable_rank2_spec()], min_eligible_cycles=1)
            cut_census = result["preflight"]["cut_census"]

            self.assertEqual(cut_census["stable_both_inner_cut_count"], 3)
            self.assertEqual(cut_census["eligible_cycle_count_with_at_least_one_stable_q"], 1)
            self.assertTrue(all(row["left_cut_status"] == "stable" for row in result["cut_rows"]))
            self.assertTrue(all(row["right_cut_status"] == "stable" for row in result["cut_rows"]))

    def test_near_degenerate_split_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result = self._run(root, [self._near_rank2_spec()], min_eligible_cycles=1)
            cut_census = result["preflight"]["cut_census"]

            self.assertEqual(cut_census["stable_both_inner_cut_count"], 0)
            self.assertEqual(cut_census["near_degenerate_both_cut_count"], 3)
            self.assertTrue(all(row["left_cut_status"] == "near_degenerate" for row in result["cut_rows"]))
            self.assertTrue(all(row["right_cut_status"] == "near_degenerate" for row in result["cut_rows"]))

    def test_caller_declared_minimum_pass_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result = self._run(root, [self._stable_rank2_spec()], min_eligible_cycles=1)

            self.assertEqual(
                result["preflight"]["empirical_surface_status"],
                gate12c.EMPIRICAL_PASS_DECLARED_MINIMUM,
            )

    def test_caller_declared_minimum_fail_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result = self._run(root, [self._stable_rank2_spec()], min_eligible_cycles=2)

            self.assertEqual(
                result["preflight"]["empirical_surface_status"],
                gate12c.EMPIRICAL_FAIL_BELOW_DECLARED_MINIMUM,
            )

    def test_separate_contract_and_empirical_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result = self._run(root, [self._stable_rank2_spec()], min_eligible_cycles=2)

            self.assertEqual(result["preflight"]["contract_feasibility_status"], gate12c.CONTRACT_PASS)
            self.assertEqual(
                result["preflight"]["empirical_surface_status"],
                gate12c.EMPIRICAL_FAIL_BELOW_DECLARED_MINIMUM,
            )

    def test_deterministic_manifest_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gate12a_dir = self._build_gate12a_fixture(root, [self._stable_rank2_spec()])
            out_dir = root / "out"

            gate12c.run_associator_feasibility_preflight(
                gate12a_dir=gate12a_dir,
                out_dir=out_dir,
                min_eligible_cycles=1,
            )
            first = read_json(out_dir / gate12c.DEFAULT_MANIFEST)
            gate12c.run_associator_feasibility_preflight(
                gate12a_dir=gate12a_dir,
                out_dir=out_dir,
                min_eligible_cycles=1,
            )
            second = read_json(out_dir / gate12c.DEFAULT_MANIFEST)

            self.assertEqual(first, second)

    def test_deterministic_checksum_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gate12a_dir = self._build_gate12a_fixture(root, [self._stable_rank2_spec()])
            out_dir = root / "out"

            gate12c.run_associator_feasibility_preflight(
                gate12a_dir=gate12a_dir,
                out_dir=out_dir,
                min_eligible_cycles=1,
            )
            first = read_json(out_dir / gate12c.DEFAULT_CHECKSUMS)
            gate12c.run_associator_feasibility_preflight(
                gate12a_dir=gate12a_dir,
                out_dir=out_dir,
                min_eligible_cycles=1,
            )
            second = read_json(out_dir / gate12c.DEFAULT_CHECKSUMS)

            self.assertEqual(first, second)

    def test_source_gate12a_artifact_directory_remains_byte_for_byte_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gate12a_dir = self._build_gate12a_fixture(root, [self._stable_rank2_spec()])
            before = snapshot_files(gate12a_dir)

            self._run(root, [self._stable_rank2_spec()], min_eligible_cycles=1, gate12a_dir=gate12a_dir)

            after = snapshot_files(gate12a_dir)
            self.assertEqual(before, after)

    def test_output_files_are_written_with_required_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            out_dir = root / "out"
            result = self._run(root, [self._stable_rank2_spec()], min_eligible_cycles=1, out_dir=out_dir)

            self.assertEqual(result["manifest"]["method_id"], gate12c.METHOD_ID)
            for name in (
                gate12c.DEFAULT_MANIFEST,
                gate12c.DEFAULT_PREFLIGHT_JSON,
                gate12c.DEFAULT_CYCLE_CENSUS,
                gate12c.DEFAULT_CUT_CENSUS,
                gate12c.DEFAULT_READ,
                gate12c.DEFAULT_CHECKSUMS,
            ):
                self.assertTrue((out_dir / name).exists(), name)

            with open(out_dir / gate12c.DEFAULT_CYCLE_CENSUS, "r", encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(len(csv_rows), 1)
            self.assertEqual(len(read_jsonl(out_dir / gate12c.DEFAULT_CUT_CENSUS)), 3)
            read_text = (out_dir / gate12c.DEFAULT_READ).read_text(encoding="utf-8")
            self.assertIn("Gate12C-1 is not implemented.", read_text)
            self.assertIn("does not measure compressed associator behavior.", read_text)

    def _run(
        self,
        root: Path,
        specs: list[dict],
        *,
        min_eligible_cycles: int,
        gate12a_dir: Path | None = None,
        out_dir: Path | None = None,
    ) -> dict:
        source_dir = gate12a_dir or self._build_gate12a_fixture(root, specs)
        return gate12c.run_associator_feasibility_preflight(
            gate12a_dir=source_dir,
            out_dir=out_dir or (root / "out"),
            min_eligible_cycles=min_eligible_cycles,
        )

    def _stable_rank2_spec(self) -> dict:
        return {
            "cycle_id": "triangle:stable2",
            "prefix": "stable2",
            "rank": 2,
            "cos_values": [1.0, 0.5],
            "edge_ids": ["stable2:z_ab", "stable2:m_bc", "stable2:a_ca"],
        }

    def _near_rank2_spec(self) -> dict:
        return {
            "cycle_id": "triangle:near2",
            "prefix": "near2",
            "rank": 2,
            "cos_values": [1.0, 1.0],
            "edge_ids": ["near2:z_ab", "near2:m_bc", "near2:a_ca"],
        }

    def _rank1_spec(self) -> dict:
        return {
            "cycle_id": "triangle:rank1",
            "prefix": "rank1",
            "rank": 1,
            "cos_values": [1.0],
            "edge_ids": ["rank1:z_ab", "rank1:m_bc", "rank1:a_ca"],
        }

    def _stable_rank3_spec(self) -> dict:
        return {
            "cycle_id": "triangle:stable3",
            "prefix": "stable3",
            "rank": 3,
            "cos_values": [1.0, 0.8, 0.4],
            "edge_ids": ["stable3:z_ab", "stable3:m_bc", "stable3:a_ca"],
        }

    def _stable_rank4_spec(self) -> dict:
        return {
            "cycle_id": "triangle:stable4",
            "prefix": "stable4",
            "rank": 4,
            "cos_values": [1.0, 0.7, 0.5, 0.2],
            "edge_ids": ["stable4:z_ab", "stable4:m_bc", "stable4:a_ca"],
        }

    def _build_gate12a_fixture(self, root: Path, specs: list[dict]) -> Path:
        gate12a_dir = root / "gate12a_fixture"
        gate12a_dir.mkdir(parents=True, exist_ok=True)
        r_max = max(int(spec["rank"]) for spec in specs)
        block_width = 2 * r_max + 2
        d_model = block_width * max(1, len(specs))

        node_rows: list[dict] = []
        basis_rows: list[np.ndarray] = []
        rank_rows: list[int] = []
        transport_rows: list[dict] = []
        transport_matrices: list[np.ndarray] = []
        singular_rows: list[np.ndarray] = []
        active_rank_rows: list[int] = []
        cycle_rows: list[dict] = []
        holonomy_rows: list[dict] = []

        for cycle_index, spec in enumerate(specs):
            rank = int(spec["rank"])
            prefix = str(spec["prefix"])
            nodes = [f"{prefix}:a", f"{prefix}:b", f"{prefix}:c"]
            basis_a, basis_b, basis_c = make_cycle_bases(
                rank=rank,
                cos_values=list(spec["cos_values"]),
                offset=cycle_index * block_width,
                d_model=d_model,
                r_max=r_max,
            )
            node_bases = [basis_a, basis_b, basis_c]
            for node_id, basis in zip(nodes, node_bases):
                node_rows.append(
                    {
                        "node_id": node_id,
                        "node_label": node_id,
                        "basis_array_index": len(basis_rows),
                        "projector_rank": rank,
                        "local_object_status": "defined",
                    }
                )
                basis_rows.append(np.asarray(basis, dtype=np.float64))
                rank_rows.append(rank)

            edge_ids = list(spec["edge_ids"])
            relation_kinds = list(spec.get("relation_kinds", ["trusted_tree", "trusted_tree", "residual_chord"]))
            for edge_index, edge_id in enumerate(edge_ids):
                source_index = edge_index
                target_index = (edge_index + 1) % 3
                matrix, singular_values, active_rank = compute_transport(
                    source_basis=node_bases[source_index],
                    target_basis=node_bases[target_index],
                    rank=rank,
                    r_max=r_max,
                )
                transport_rows.append(
                    {
                        "edge_id": edge_id,
                        "source_node_id": nodes[source_index],
                        "target_node_id": nodes[target_index],
                        "relation_kind": relation_kinds[edge_index],
                        "anchor_qualified": edge_index == 2,
                        "anchor_relation_id": f"anchor:{edge_id}" if edge_index == 2 else "",
                        "source_rank": rank,
                        "target_rank": rank,
                        "overlap_rank": active_rank,
                        "transport_case": (
                            "equal_rank_orthogonal"
                            if active_rank == rank
                            else "rank_mismatch_partial_isometry"
                        ),
                        "operator_array_index": len(transport_matrices),
                        "compatibility_gap_fro": float(np.linalg.norm(1.0 - singular_values[:active_rank]))
                        if active_rank > 0
                        else None,
                        "transport_level_compatibility_status": "compatible",
                    }
                )
                transport_matrices.append(matrix)
                singular_rows.append(singular_values)
                active_rank_rows.append(active_rank)

            cycle_rows.append(
                {
                    "cycle_id": str(spec["cycle_id"]),
                    "base_node_id": nodes[0],
                    "edge_id_path": sorted(edge_ids),
                    "node_id_path": nodes + [nodes[0]],
                    "cycle_length": 3,
                    "cycle_status": "admissible_explicit_triangle",
                }
            )
            holonomy_rows.append(
                {
                    "cycle_id": str(spec["cycle_id"]),
                    "base_node_id": nodes[0],
                    "holonomy_rank": rank,
                    "holonomy_residual_fro": 0.0,
                    "holonomy_status": "defined",
                }
            )

        write_json(
            gate12a_dir / gate12c.DEFAULT_MANIFEST,
            {
                "run_id": "gate12a_fixture",
                "schema_version": "gate12a_discrete_connection_v1",
                "code_git_commit": "fixture-commit",
            },
        )
        write_jsonl(gate12a_dir / gate12c.DEFAULT_NODE_REGISTRY, node_rows)
        np.savez(
            gate12a_dir / gate12c.DEFAULT_NODE_ARRAYS,
            basis_factor=np.asarray(basis_rows, dtype=np.float64),
            rank_active=np.asarray(rank_rows, dtype=np.int64),
        )
        write_jsonl(gate12a_dir / gate12c.DEFAULT_TRANSPORT_REGISTRY, transport_rows)
        np.savez(
            gate12a_dir / gate12c.DEFAULT_TRANSPORT_ARRAYS,
            transport_matrix_local=np.asarray(transport_matrices, dtype=np.float64),
            overlap_singular_values=np.asarray(singular_rows, dtype=np.float64),
            active_rank=np.asarray(active_rank_rows, dtype=np.int64),
        )
        write_jsonl(gate12a_dir / gate12c.DEFAULT_CYCLE_REGISTRY, cycle_rows)
        write_jsonl(gate12a_dir / gate12c.DEFAULT_HOLONOMY_REGISTRY, holonomy_rows)
        return gate12a_dir


if __name__ == "__main__":
    unittest.main()
