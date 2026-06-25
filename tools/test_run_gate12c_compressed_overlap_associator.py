#!/usr/bin/env python3
"""Regression tests for Gate12C-1 compressed-overlap associator audit."""

from __future__ import annotations

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

import inspect_gate12c_associator_feasibility as gate12c0
import run_gate12c_compressed_overlap_associator as gate12c


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


def snapshot_files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def sign_normalized_qr(matrix: np.ndarray) -> np.ndarray:
    q_matrix, r_matrix = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r_matrix))
    signs[signs == 0.0] = 1.0
    return np.asarray(q_matrix * signs, dtype=np.float64)


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


class Gate12CCompressedOverlapAssociatorTest(unittest.TestCase):
    def test_ordinary_associativity_null(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._run(Path(tmpdir), [self._positive_rank3_spec()])

            self.assertTrue(
                all(row["ordinary_associator_fro"] <= 1.0e-12 for row in result["registry_rows"])
            )

    def test_q_equals_r_no_compression_null(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._run(Path(tmpdir), [self._positive_rank3_spec()])

            self.assertTrue(
                all(
                    row["no_compression_associator_fro"]
                    <= gate12c.DEFAULT_TAU_NO_COMPRESSION_ASSOCIATOR_FRO
                    for row in result["registry_rows"]
                )
            )

    def test_deterministic_positive_compressed_associator_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._run(Path(tmpdir), [self._positive_rank3_spec()])
            values = [
                row["compressed_overlap_associator_fro"]
                for row in result["registry_rows"]
                if row["compressed_overlap_associator_fro"] is not None
            ]

            self.assertTrue(values)
            self.assertGreater(max(values), 1.0e-8)

    def test_rank2_q_enumeration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._run(Path(tmpdir), [self._positive_rank2_spec()])

            self.assertEqual(len(result["registry_rows"]), 3)
            self.assertEqual({row["compression_rank_q"] for row in result["registry_rows"]}, {1})

    def test_rank3_q_enumeration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._run(Path(tmpdir), [self._positive_rank3_spec()])

            self.assertEqual(len(result["registry_rows"]), 6)
            self.assertEqual({row["compression_rank_q"] for row in result["registry_rows"]}, {1, 2})

    def test_all_three_cyclic_roots_emitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._run(Path(tmpdir), [self._positive_rank2_spec()])

            self.assertEqual({row["root_rotation_index"] for row in result["registry_rows"]}, {0, 1, 2})
            self.assertEqual(
                {row["evaluation_root_node_id"] for row in result["registry_rows"]},
                {
                    "sample_000001:positive2:a",
                    "sample_000001:positive2:b",
                    "sample_000001:positive2:c",
                },
            )

    def test_lexical_edge_id_order_differs_from_traversal_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._run(Path(tmpdir), [self._positive_rank2_spec()])
            row = result["registry_rows"][0]

            self.assertEqual(
                row["ordered_edge_id_path"],
                ["positive2:z_ab", "positive2:m_bc", "positive2:a_ca"],
            )
            self.assertNotEqual(row["ordered_edge_id_path"], sorted(row["ordered_edge_id_path"]))

    def test_near_degenerate_observed_cut_is_not_active(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._run(
                Path(tmpdir),
                [self._near_rank2_spec()],
                requested_draws=1,
                max_attempts=1,
            )

            self.assertTrue(
                all(row["truncation_status"] == gate12c.TRUNCATION_NEAR_BOTH for row in result["registry_rows"])
            )
            self.assertTrue(
                all(row["measurement_status"] == gate12c.MEASUREMENT_NOT_EVALUATED for row in result["registry_rows"])
            )
            self.assertTrue(all(not row["aggregation_eligible"] for row in result["registry_rows"]))

    def test_gauge_operator_covariance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._run(Path(tmpdir), [self._positive_rank3_spec()])
            comparable = [
                row for row in result["registry_rows"] if row["gauge_operator_covariance_fro"] is not None
            ]

            self.assertTrue(comparable)
            self.assertLessEqual(
                max(row["gauge_operator_covariance_fro"] for row in comparable),
                gate12c.DEFAULT_TAU_GAUGE_OPERATOR_COVARIANCE_FRO,
            )

    def test_gauge_scalar_invariance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._run(Path(tmpdir), [self._positive_rank3_spec()])
            comparable = [
                row for row in result["registry_rows"] if row["gauge_scalar_delta_abs"] is not None
            ]

            self.assertTrue(comparable)
            self.assertLessEqual(
                max(row["gauge_scalar_delta_abs"] for row in comparable),
                gate12c.DEFAULT_TAU_GAUGE_SCALAR_DELTA_ABS,
            )

    def test_transformed_cut_status_change_accounting(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._run(Path(tmpdir), [self._near_rank2_spec()])

            self.assertTrue(
                all(not row["gauge_cut_status_preserved"] for row in result["registry_rows"])
            )
            self.assertTrue(
                all(
                    row["gauge_scalar_status"] == gate12c.GAUGE_SCALAR_NOT_EVALUATED
                    for row in result["registry_rows"]
                )
            )

    def test_canonical_json_seed_encoding(self) -> None:
        seed_bytes = gate12c.canonical_orientation_seed_bytes(
            orientation_null_seed="seed",
            cycle_id="cycle",
            edge_id="edge",
            draw_index=0,
            left_or_right_orientation_label="left",
        )

        self.assertEqual(
            seed_bytes,
            b'["gate12c1_orientation_null_v1","seed","cycle","edge",0,"left"]',
        )

    def test_frozen_counter_box_muller_generator_determinism(self) -> None:
        seed_bytes = gate12c.canonical_orientation_seed_bytes(
            orientation_null_seed="seed",
            cycle_id="cycle",
            edge_id="edge",
            draw_index=0,
            left_or_right_orientation_label="left",
        )
        z_matrix = gate12c.normal_matrix_from_seed(seed_bytes, 2)
        q_matrix = gate12c.deterministic_orthogonal_matrix(seed_bytes, 2)

        self.assertEqual(
            gate12c.sha256_counter_stream(bytes.fromhex("00" * 32), 16).hex(),
            "2c34ce1df23b838c5abf2a7f6437cca3",
        )
        np.testing.assert_allclose(
            z_matrix,
            np.asarray(
                [
                    [-0.040611263797589385, 0.24630101071780602],
                    [0.39594733100071083, 0.83470199799367473],
                ]
            ),
            atol=1.0e-15,
        )
        np.testing.assert_allclose(q_matrix.T @ q_matrix, np.eye(2), atol=1.0e-12)

    def test_orientation_null_spectrum_preservation(self) -> None:
        matrix = gate12c.null_edge_matrix(
            singular_values=np.asarray([0.9, 0.4, 0.1]),
            orientation_null_seed="seed",
            cycle_id="cycle",
            edge_id="edge",
            draw_index=0,
            rank=3,
        )

        singular_values = np.linalg.svd(matrix, compute_uv=False)
        np.testing.assert_allclose(singular_values, np.asarray([0.9, 0.4, 0.1]), atol=1.0e-12)

    def test_one_null_triangle_reused_across_roots_and_q(self) -> None:
        call_count = 0
        original = gate12c.null_edge_matrix

        def counting_null_edge_matrix(**kwargs: object) -> np.ndarray:
            nonlocal call_count
            call_count += 1
            return original(**kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                gate12c.null_edge_matrix = counting_null_edge_matrix  # type: ignore[assignment]
                self._run(
                    Path(tmpdir),
                    [self._positive_rank3_spec()],
                    requested_draws=1,
                    max_attempts=1,
                )
            finally:
                gate12c.null_edge_matrix = original  # type: ignore[assignment]

        self.assertEqual(call_count, 3)

    def test_row_order_independent_null_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            specs = [self._positive_rank2_spec(prefix="b_cycle"), self._positive_rank2_spec(prefix="a_cycle")]
            gate12a_a = self._build_gate12a_fixture(root / "a", specs)
            gate12a_b = self._build_gate12a_fixture(root / "b", list(reversed(specs)))

            result_a = self._run_existing(root / "a", gate12a_a, root / "out_a")
            result_b = self._run_existing(root / "b", gate12a_b, root / "out_b")

            self.assertEqual(result_a["registry_rows"], result_b["registry_rows"])

    def test_valid_draw_loop_reaches_requested_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._run(
                Path(tmpdir),
                [self._positive_rank2_spec()],
                requested_draws=2,
                max_attempts=4,
            )

            self.assertTrue(
                all(row["orientation_null_valid_draw_count"] == 2 for row in result["registry_rows"])
            )
            self.assertTrue(
                all(row["orientation_null_status"] == gate12c.ORIENTATION_NULL_COMPLETE for row in result["registry_rows"])
            )

    def test_max_attempt_exhaustion_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._run(
                Path(tmpdir),
                [self._positive_rank2_spec()],
                requested_draws=1,
                max_attempts=1,
                tolerances=gate12c.Tolerances(tau_split_rel=10.0),
            )

            self.assertTrue(
                all(row["orientation_null_attempt_count"] == 1 for row in result["registry_rows"])
            )

    def test_insufficient_valid_draw_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._run(
                Path(tmpdir),
                [self._positive_rank2_spec()],
                requested_draws=1,
                max_attempts=1,
                tolerances=gate12c.Tolerances(tau_split_rel=10.0),
            )

            self.assertTrue(
                all(row["orientation_null_status"] == gate12c.ORIENTATION_NULL_INSUFFICIENT for row in result["registry_rows"])
            )
            self.assertTrue(all(not row["aggregation_eligible"] for row in result["registry_rows"]))

    def test_p_upper_formula(self) -> None:
        summary = gate12c.summarize_null_values(
            observed_value=2.0,
            accumulator=gate12c.NullAccumulator(
                requested_draw_count=3,
                valid_values=[1.0, 2.0, 3.0],
            ),
            epsilon=1.0e-12,
        )

        self.assertEqual(summary["orientation_null_empirical_p_upper"], 0.75)

    def test_mad_and_robust_z_formula(self) -> None:
        summary = gate12c.summarize_null_values(
            observed_value=4.0,
            accumulator=gate12c.NullAccumulator(
                requested_draw_count=3,
                valid_values=[1.0, 2.0, 3.0],
            ),
            epsilon=1.0e-12,
        )

        self.assertEqual(summary["orientation_null_median"], 2.0)
        self.assertEqual(summary["orientation_null_mad"], 1.0)
        self.assertAlmostEqual(
            summary["orientation_null_robust_z"],
            (4.0 - 2.0) / (1.4826 * 1.0 + 1.0e-12),
        )

    def test_scale_degenerate_null_behavior(self) -> None:
        summary = gate12c.summarize_null_values(
            observed_value=1.0,
            accumulator=gate12c.NullAccumulator(
                requested_draw_count=3,
                valid_values=[0.5, 0.5, 0.5],
            ),
            epsilon=1.0e-12,
        )

        self.assertTrue(summary["orientation_null_scale_degenerate"])
        self.assertIsNone(summary["orientation_null_robust_z"])
        self.assertEqual(
            summary["orientation_null_excess_status"],
            gate12c.ORIENTATION_EXCESS_SCALE_DEGENERATE,
        )

    def test_source_block_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._run(Path(tmpdir), [self._positive_rank2_spec()])
            row = result["registry_rows"][0]

            self.assertEqual(row["source_sample_block_id"], "sample_000001")
            self.assertEqual(row["source_block_status"], "single_sample")
            self.assertEqual(
                gate12c.source_block_status(["x:a", "sample_000001:b", "sample_000001:c"]),
                ("mixed_or_undefined", "mixed_or_undefined"),
            )

    def test_absence_of_promotable_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._run(Path(tmpdir), [self._positive_rank2_spec()])

            self.assertTrue(all("promotable" not in row for row in result["registry_rows"]))

    def test_required_compressed_overlap_field_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._run(Path(tmpdir), [self._positive_rank2_spec()])
            row = result["registry_rows"][0]

            self.assertIn("compressed_overlap_closure_left_fro", row)
            self.assertIn("compressed_overlap_closure_right_fro", row)
            self.assertNotIn("left_closure_fro", row)
            self.assertNotIn("right_closure_fro", row)
            self.assertNotIn("holonomy_left", row)
            self.assertNotIn("holonomy_right", row)

    def test_separate_gate12a_holonomy_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._run(Path(tmpdir), [self._positive_rank2_spec()])

            self.assertTrue(
                all(row["gate12a_holonomy_residual_fro"] == 0.25 for row in result["registry_rows"])
            )

    def test_deterministic_registry_and_npz_indexing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            out_dir = root / "out"
            result = self._run(root, [self._positive_rank3_spec()], out_dir=out_dir)
            rows = read_jsonl(out_dir / gate12c.DEFAULT_REGISTRY)

            with np.load(out_dir / gate12c.DEFAULT_ARRAYS) as handle:
                left = np.asarray(handle["compressed_overlap_left_operator"])
                right = np.asarray(handle["compressed_overlap_right_operator"])
                assoc = np.asarray(handle["compressed_overlap_associator_operator"])

            self.assertEqual([row["operator_array_index"] for row in rows], list(range(len(rows))))
            self.assertEqual(left.shape[0], len(rows))
            self.assertEqual(right.shape[0], len(rows))
            self.assertEqual(assoc.shape[0], len(rows))
            self.assertEqual(rows, result["registry_rows"])

    def test_deterministic_manifest_and_checksum_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gate12a_dir = self._build_gate12a_fixture(root, [self._positive_rank2_spec()])
            out_dir = root / "out"

            self._run_existing(root, gate12a_dir, out_dir)
            first_manifest = read_json(out_dir / gate12c.DEFAULT_MANIFEST)
            first_checksums = read_json(out_dir / gate12c.DEFAULT_CHECKSUMS)
            self._run_existing(root, gate12a_dir, out_dir)
            second_manifest = read_json(out_dir / gate12c.DEFAULT_MANIFEST)
            second_checksums = read_json(out_dir / gate12c.DEFAULT_CHECKSUMS)

            self.assertEqual(first_manifest, second_manifest)
            self.assertEqual(first_checksums, second_checksums)

    def test_manifest_uses_claim_boundary_not_validation_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._run(Path(tmpdir), [self._positive_rank2_spec()])
            manifest = result["manifest"]

            self.assertNotIn("boundary", manifest)
            self.assertIn("claim_boundary", manifest)
            self.assertNotIn("implementation_only", manifest["claim_boundary"])
            self.assertNotIn("synthetic_fixture_only_for_tests", manifest["claim_boundary"])
            self.assertFalse(
                manifest["claim_boundary"]["scientific_null_excess_threshold_defined"]
            )
            self.assertFalse(manifest["claim_boundary"]["gate12b_overlay_used"])
            self.assertFalse(
                manifest["claim_boundary"]["rectangular_rank_mismatch_supported"]
            )

    def test_source_directory_immutability(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gate12a_dir = self._build_gate12a_fixture(root, [self._positive_rank2_spec()])
            before = snapshot_files(gate12a_dir)

            self._run_existing(root, gate12a_dir, root / "out")

            after = snapshot_files(gate12a_dir)
            self.assertEqual(before, after)

    def test_valid_no_excess_cli_exit_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gate12a_dir = self._build_gate12a_fixture(root, [self._positive_rank2_spec()])
            exit_code = gate12c.main(
                [
                    "--gate12a-dir",
                    str(gate12a_dir),
                    "--out-dir",
                    str(root / "out"),
                    "--orientation-null-seed",
                    "seed",
                    "--orientation-null-requested-draw-count",
                    "1",
                    "--orientation-null-max-attempt-count",
                    "1",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(read_json(root / "out" / gate12c.DEFAULT_STATUS)["process_status"], "pass")

    def test_contract_failure_cli_exit_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gate12a_dir = self._build_gate12a_fixture(root, [self._positive_rank2_spec()])
            (gate12a_dir / gate12c0.DEFAULT_TRANSPORT_ARRAYS).unlink()
            exit_code = gate12c.main(
                [
                    "--gate12a-dir",
                    str(gate12a_dir),
                    "--out-dir",
                    str(root / "out"),
                    "--orientation-null-seed",
                    "seed",
                    "--orientation-null-requested-draw-count",
                    "1",
                    "--orientation-null-max-attempt-count",
                    "1",
                ]
            )

            self.assertEqual(exit_code, 1)
            self.assertEqual(read_json(root / "out" / gate12c.DEFAULT_STATUS)["process_status"], "fail")

    def test_cli_rejects_source_alias_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gate12a_dir = self._build_gate12a_fixture(root, [self._positive_rank2_spec()])
            before = snapshot_files(gate12a_dir)

            exit_code = gate12c.main(
                [
                    "--gate12a-dir",
                    str(gate12a_dir),
                    "--out-dir",
                    str(gate12a_dir),
                    "--orientation-null-seed",
                    "seed",
                    "--orientation-null-requested-draw-count",
                    "1",
                    "--orientation-null-max-attempt-count",
                    "1",
                ]
            )

            self.assertEqual(exit_code, 1)
            self.assertEqual(before, snapshot_files(gate12a_dir))
            self.assertFalse((gate12a_dir / gate12c.DEFAULT_STATUS).exists())

    def test_cli_rejects_nested_output_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gate12a_dir = self._build_gate12a_fixture(root, [self._positive_rank2_spec()])
            nested_out = gate12a_dir / "nested_gate12c1"
            before = snapshot_files(gate12a_dir)

            exit_code = gate12c.main(
                [
                    "--gate12a-dir",
                    str(gate12a_dir),
                    "--out-dir",
                    str(nested_out),
                    "--orientation-null-seed",
                    "seed",
                    "--orientation-null-requested-draw-count",
                    "1",
                    "--orientation-null-max-attempt-count",
                    "1",
                ]
            )

            self.assertEqual(exit_code, 1)
            self.assertEqual(before, snapshot_files(gate12a_dir))
            self.assertFalse(nested_out.exists())

    def test_defined_holonomy_requires_finite_residual(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gate12a_dir = self._build_gate12a_fixture(root, [self._positive_rank2_spec()])
            holonomy_path = gate12a_dir / gate12c0.DEFAULT_HOLONOMY_REGISTRY
            rows = read_jsonl(holonomy_path)
            del rows[0]["holonomy_residual_fro"]
            write_jsonl(holonomy_path, rows)

            with self.assertRaisesRegex(ValueError, "holonomy_residual_fro"):
                self._run_existing(root, gate12a_dir, root / "out")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gate12a_dir = self._build_gate12a_fixture(root, [self._positive_rank2_spec()])
            holonomy_path = gate12a_dir / gate12c0.DEFAULT_HOLONOMY_REGISTRY
            rows = read_jsonl(holonomy_path)
            rows[0]["holonomy_residual_fro"] = None
            write_jsonl(holonomy_path, rows)

            with self.assertRaisesRegex(gate12c.Gate12CContractError, "non-numeric"):
                self._run_existing(root, gate12a_dir, root / "out")

    def _run(
        self,
        root: Path,
        specs: list[dict],
        *,
        requested_draws: int = 1,
        max_attempts: int = 3,
        out_dir: Path | None = None,
        tolerances: gate12c.Tolerances | None = None,
    ) -> dict:
        gate12a_dir = self._build_gate12a_fixture(root, specs)
        return self._run_existing(
            root,
            gate12a_dir,
            out_dir or (root / "out"),
            requested_draws=requested_draws,
            max_attempts=max_attempts,
            tolerances=tolerances,
        )

    def _run_existing(
        self,
        root: Path,
        gate12a_dir: Path,
        out_dir: Path,
        *,
        requested_draws: int = 1,
        max_attempts: int = 3,
        tolerances: gate12c.Tolerances | None = None,
    ) -> dict:
        return gate12c.run_gate12c_compressed_overlap_associator(
            gate12a_dir=gate12a_dir,
            out_dir=out_dir,
            orientation_null_seed="seed",
            orientation_null_requested_draw_count=requested_draws,
            orientation_null_max_attempt_count=max_attempts,
            tolerances=tolerances or gate12c.Tolerances(),
        )

    def _positive_rank2_spec(self, prefix: str = "positive2") -> dict:
        return {
            "cycle_id": f"triangle:{prefix}",
            "prefix": prefix,
            "rank": 2,
            "kind": "random",
            "seed": 202,
            "sample_id": "sample_000001",
        }

    def _positive_rank3_spec(self, prefix: str = "positive3") -> dict:
        return {
            "cycle_id": f"triangle:{prefix}",
            "prefix": prefix,
            "rank": 3,
            "kind": "random",
            "seed": 303,
            "sample_id": "sample_000001",
        }

    def _near_rank2_spec(self) -> dict:
        return {
            "cycle_id": "triangle:near2",
            "prefix": "near2",
            "rank": 2,
            "kind": "identical",
            "sample_id": "sample_000001",
        }

    def _cycle_bases(
        self,
        *,
        spec: dict,
        offset: int,
        block_width: int,
        d_model: int,
        r_max: int,
    ) -> list[np.ndarray]:
        rank = int(spec["rank"])
        if spec["kind"] == "identical":
            basis = np.zeros((d_model, r_max), dtype=np.float64)
            for col in range(rank):
                basis[offset + col, col] = 1.0
            return [basis.copy(), basis.copy(), basis.copy()]

        bases: list[np.ndarray] = []
        rng = np.random.default_rng(int(spec["seed"]))
        for _node_index in range(3):
            raw = rng.normal(size=(block_width, rank))
            q_matrix = sign_normalized_qr(raw)[:, :rank]
            basis = np.zeros((d_model, r_max), dtype=np.float64)
            basis[offset : offset + block_width, :rank] = q_matrix
            bases.append(basis)
        return bases

    def _build_gate12a_fixture(self, root: Path, specs: list[dict]) -> Path:
        gate12a_dir = root / "gate12a_fixture"
        gate12a_dir.mkdir(parents=True, exist_ok=True)
        r_max = max(int(spec["rank"]) for spec in specs)
        block_width = max(4 * r_max + 5, 12)
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
            sample_id = str(spec.get("sample_id", "sample_000001"))
            nodes = [
                f"{sample_id}:{prefix}:a",
                f"{sample_id}:{prefix}:b",
                f"{sample_id}:{prefix}:c",
            ]
            node_bases = self._cycle_bases(
                spec=spec,
                offset=cycle_index * block_width,
                block_width=block_width,
                d_model=d_model,
                r_max=r_max,
            )
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

            edge_ids = [f"{prefix}:z_ab", f"{prefix}:m_bc", f"{prefix}:a_ca"]
            relation_kinds = ["trusted_tree", "trusted_tree", "residual_chord"]
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
                        "compatibility_gap_fro": float(
                            np.linalg.norm(1.0 - singular_values[:active_rank])
                        )
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
                    "holonomy_residual_fro": 0.25,
                    "holonomy_status": "defined",
                }
            )

        write_json(
            gate12a_dir / gate12c0.DEFAULT_MANIFEST,
            {
                "run_id": "gate12a_fixture",
                "schema_version": "gate12a_discrete_connection_v1",
                "code_git_commit": "fixture-commit",
            },
        )
        write_jsonl(gate12a_dir / gate12c0.DEFAULT_NODE_REGISTRY, node_rows)
        np.savez(
            gate12a_dir / gate12c0.DEFAULT_NODE_ARRAYS,
            basis_factor=np.asarray(basis_rows, dtype=np.float64),
            rank_active=np.asarray(rank_rows, dtype=np.int64),
        )
        write_jsonl(gate12a_dir / gate12c0.DEFAULT_TRANSPORT_REGISTRY, transport_rows)
        np.savez(
            gate12a_dir / gate12c0.DEFAULT_TRANSPORT_ARRAYS,
            transport_matrix_local=np.asarray(transport_matrices, dtype=np.float64),
            overlap_singular_values=np.asarray(singular_rows, dtype=np.float64),
            active_rank=np.asarray(active_rank_rows, dtype=np.int64),
        )
        write_jsonl(gate12a_dir / gate12c0.DEFAULT_CYCLE_REGISTRY, cycle_rows)
        write_jsonl(gate12a_dir / gate12c0.DEFAULT_HOLONOMY_REGISTRY, holonomy_rows)
        return gate12a_dir


if __name__ == "__main__":
    unittest.main()
