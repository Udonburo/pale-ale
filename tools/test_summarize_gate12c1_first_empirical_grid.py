#!/usr/bin/env python3
"""Regression tests for the Gate12C-1 first empirical grid summarizer."""

from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import inspect_gate12c_associator_feasibility as gate12c0
import summarize_gate12c1_first_empirical_grid as summary


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def snapshot_files(root: Path) -> dict[str, bytes]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class Gate12C1FirstEmpiricalGridSummaryTest(unittest.TestCase):
    def test_case_manifest_rejects_missing_substituted_and_noncanonical_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = self._case_manifest_payload(root)

            with self.assertRaisesRegex(summary.Gate12C1SummaryContractError, "exactly 12"):
                summary.validate_case_manifest_payload(
                    {**manifest, "cases": manifest["cases"][:-1]},
                    base_dir=root,
                )

            substituted = json.loads(json.dumps(manifest))
            substituted["cases"][0]["model"] = "not_the_model"
            with self.assertRaisesRegex(summary.Gate12C1SummaryContractError, "model"):
                summary.validate_case_manifest_payload(substituted, base_dir=root)

            swapped = json.loads(json.dumps(manifest))
            swapped["cases"][0], swapped["cases"][1] = swapped["cases"][1], swapped["cases"][0]
            with self.assertRaisesRegex(summary.Gate12C1SummaryContractError, "ordering"):
                summary.validate_case_manifest_payload(swapped, base_dir=root)

    def test_cli_rejects_source_alias_and_nested_output_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_manifest = root / "cases.json"
            payload = self._case_manifest_payload(root)
            write_json(case_manifest, payload)
            source_dir = Path(payload["cases"][0]["source_gate12a_dir"])
            source_dir.mkdir(parents=True)
            before = snapshot_files(source_dir)

            alias_exit = summary.main(
                ["--case-manifest", str(case_manifest), "--out-dir", str(source_dir)]
            )
            nested = source_dir / "summary_child"
            nested_exit = summary.main(
                ["--case-manifest", str(case_manifest), "--out-dir", str(nested)]
            )

            self.assertEqual(alias_exit, 1)
            self.assertEqual(nested_exit, 1)
            self.assertEqual(before, snapshot_files(source_dir))
            self.assertFalse((source_dir / summary.OUTPUT_GRID_SUMMARY).exists())
            self.assertFalse(nested.exists())

    def test_primary_row_eligibility_and_log_ratio_formula(self) -> None:
        row = self._registry_row(
            cycle_id="cycle",
            root=0,
            q=1,
            operator_index=0,
            block_id="sample_000001",
            assoc=4.0,
            null_median=2.0,
            holonomy=0.25,
            edge_gap=0.0,
        )

        metrics = summary.primary_row_metrics(row)

        self.assertTrue(metrics["eligible"])
        self.assertAlmostEqual(
            metrics["log_null_ratio"],
            math.log((4.0 + summary.PRIMARY_EPSILON) / (2.0 + summary.PRIMARY_EPSILON)),
        )

        row["orientation_null_scale_degenerate"] = True
        self.assertFalse(summary.primary_row_metrics(row)["eligible"])

    def test_cycle_q_requires_all_three_roots_and_single_sample_block(self) -> None:
        case = summary.CaseInput(
            spec=summary.CANONICAL_CASES[0],
            source_gate12a_dir=Path("source"),
            gate12c1_run_dir=Path("run"),
        )
        expected = [
            summary.ExpectedCycle(
                case_id="case_01",
                cycle_id="cycle_000001",
                source_sample_block_id="sample_000001",
                source_block_status="single_sample",
                gate12a_holonomy_residual_fro=0.25,
                edge_compatibility_gap_max=0.0,
                cycle_rank=3,
            )
        ]
        rows = [
            self._registry_row(
                cycle_id="cycle_000001",
                root=root,
                q=1,
                operator_index=root,
                block_id="sample_000001",
            )
            for root in (0, 1)
        ]

        scores = summary.build_cycle_q_scores(
            case_inputs=[case],
            expected_cycles_by_case={"case_01": expected},
            rows_by_case={"case_01": rows},
        )

        q1 = [row for row in scores if row["compression_rank_q"] == 1][0]
        self.assertFalse(q1["cycle_q_primary_valid"])
        self.assertIn("missing_or_malformed_roots", q1["coverage_status"])

        rows.append(
            self._registry_row(
                cycle_id="cycle_000001",
                root=2,
                q=1,
                operator_index=2,
                block_id="sample_000001",
            )
        )
        scores = summary.build_cycle_q_scores(
            case_inputs=[case],
            expected_cycles_by_case={"case_01": expected},
            rows_by_case={"case_01": rows},
        )
        q1 = [row for row in scores if row["compression_rank_q"] == 1][0]
        self.assertTrue(q1["cycle_q_primary_valid"])
        self.assertIsNotNone(q1["cycle_q_log_null_ratio"])

    def test_exact_sign_test_and_holm_are_frozen(self) -> None:
        status, raw_p = summary.exact_one_sided_sign_p(positive_count=3, negative_count=1)
        self.assertEqual(status, "informative")
        self.assertEqual(raw_p, 5 / 16)

        endpoint_rows = []
        for case in summary.CANONICAL_CASES:
            for q in (1, 2):
                endpoint_rows.append(
                    {
                        "case_id": case.case_id,
                        "case_order": case.case_order,
                        "compression_rank_q": q,
                        "raw_p": 0.01 if case.case_order == 1 and q == 1 else 1.0,
                    }
                )
        summary.apply_holm(endpoint_rows)

        first = [row for row in endpoint_rows if row["case_id"] == "case_01" and row["compression_rank_q"] == 1][0]
        self.assertEqual(first["holm_sort_position"], 1)
        self.assertAlmostEqual(first["holm_adjusted_p"], 0.24)

    def test_grid_outcome_classifier_precedence(self) -> None:
        endpoints = self._endpoint_rows(supporting_cases=set())
        outcome = summary.classify_grid_outcome(
            execution_status="complete",
            run_q_tests=endpoints,
            case_inputs=self._case_inputs(Path("root")),
        )
        self.assertEqual(outcome["grid_outcome"], "no_directional_support")

        endpoints = self._endpoint_rows(supporting_cases={case.case_id for case in summary.CANONICAL_CASES})
        outcome = summary.classify_grid_outcome(
            execution_status="complete",
            run_q_tests=endpoints,
            case_inputs=self._case_inputs(Path("root")),
        )
        self.assertEqual(outcome["grid_outcome"], "strong_broad")

        endpoints = self._endpoint_rows(supporting_cases={case.case_id for case in summary.CANONICAL_CASES})
        endpoints[0]["cycle_coverage_pass"] = False
        outcome = summary.classify_grid_outcome(
            execution_status="complete",
            run_q_tests=endpoints,
            case_inputs=self._case_inputs(Path("root")),
        )
        self.assertEqual(outcome["grid_outcome"], "coverage_limited")

        endpoints = self._endpoint_rows(
            supporting_cases={case.case_id for case in summary.CANONICAL_CASES[:5]},
            discordant_cases={case.case_id for case in summary.CANONICAL_CASES[:6]},
        )
        outcome = summary.classify_grid_outcome(
            execution_status="complete",
            run_q_tests=endpoints,
            case_inputs=self._case_inputs(Path("root")),
        )
        self.assertEqual(outcome["grid_outcome"], "mixed_q")

    def test_spearman_uses_average_ranks_and_reports_constant_inputs(self) -> None:
        self.assertEqual(summary.average_ranks([1.0, 1.0, 3.0]), [1.5, 1.5, 3.0])
        status, rho, n = summary.spearman_rho([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
        self.assertEqual(status, "defined")
        self.assertEqual(n, 3)
        self.assertAlmostEqual(rho or 0.0, -1.0)

        status, rho, _n = summary.spearman_rho([1.0, 1.0], [2.0, 3.0])
        self.assertEqual(status, "insufficient_or_constant")
        self.assertIsNone(rho)

    def test_full_synthetic_grid_outputs_complete_no_support(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_manifest = self._build_full_synthetic_grid(root, assoc=0.5, null_median=1.0)
            out_dir = root / "summary"

            result = summary.summarize_gate12c1_first_empirical_grid(
                case_manifest_path=case_manifest,
                out_dir=out_dir,
            )

            self.assertEqual(result["grid_summary"]["execution_status"], "complete")
            self.assertEqual(result["grid_summary"]["grid_outcome"], "no_directional_support")
            self.assertEqual(len(result["run_q_tests"]), 24)
            self.assertTrue(all(row["coverage_pass"] for row in result["run_q_tests"]))
            self.assertTrue((out_dir / summary.OUTPUT_CHECKSUMS).exists())
            self.assertFalse(result["manifest"]["claim_boundary"]["gate12b_overlay_used"])
            self._assert_checksums(out_dir)

    def test_gate12b_like_registry_fields_do_not_change_scientific_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_manifest = self._build_full_synthetic_grid(root / "a", assoc=0.5, null_median=1.0)
            case_manifest_b = root / "b" / "cases.json"
            shutil.copytree(root / "a", root / "b")
            for case in summary.CANONICAL_CASES:
                run_dir = root / "b" / "gate12c1" / case.case_id
                registry_path = run_dir / "triangle_associator_registry.jsonl"
                rows = read_jsonl(registry_path)
                for row in rows:
                    row["gate12b_candidate_label"] = "high"
                    row["observer_relative_band"] = "flat"
                write_jsonl(registry_path, rows)
                self._write_runner_checksums(run_dir)

            out_a = root / "out_a"
            out_b = root / "out_b"
            summary.summarize_gate12c1_first_empirical_grid(
                case_manifest_path=case_manifest,
                out_dir=out_a,
            )
            summary.summarize_gate12c1_first_empirical_grid(
                case_manifest_path=case_manifest_b,
                out_dir=out_b,
            )

            for name in (
                summary.OUTPUT_GRID_SUMMARY,
                summary.OUTPUT_CYCLE_Q,
                summary.OUTPUT_BLOCK_Q,
                summary.OUTPUT_RUN_Q,
                summary.OUTPUT_SECONDARY,
            ):
                self.assertEqual((out_a / name).read_bytes(), (out_b / name).read_bytes())

    def test_malformed_registry_duplicate_and_unexpected_cycle_are_contract_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_manifest = self._build_full_synthetic_grid(root, assoc=0.5, null_median=1.0)
            run_dir = root / "gate12c1" / "case_01"
            registry_path = run_dir / "triangle_associator_registry.jsonl"
            rows = read_jsonl(registry_path)
            rows[2]["root_rotation_index"] = rows[0]["root_rotation_index"]
            write_jsonl(registry_path, rows)
            self._write_runner_checksums(run_dir)

            exit_code = summary.main(
                ["--case-manifest", str(case_manifest), "--out-dir", str(root / "summary")]
            )

            self.assertEqual(exit_code, 1)
            failure = read_json(root / "summary" / summary.OUTPUT_GRID_SUMMARY)
            self.assertEqual(failure["execution_status"], "contract_failure")
            self.assertEqual(failure["grid_outcome"], "not_classified")

    def test_runner_manifest_provenance_is_strict(self) -> None:
        case = summary.CaseInput(
            spec=summary.CANONICAL_CASES[0],
            source_gate12a_dir=Path("source"),
            gate12c1_run_dir=Path("run"),
        )
        manifest = self._runner_manifest(case.spec, source_run_id=case.spec.source_gate12a_run_id)
        manifest["orientation_null_seed"] = "changed_after_first_run"

        with self.assertRaisesRegex(summary.Gate12C1SummaryContractError, "orientation_null_seed"):
            summary.validate_runner_manifest(
                case=case,
                runner_manifest=manifest,
                source_manifest={"run_id": case.spec.source_gate12a_run_id},
            )

    def _case_inputs(self, root: Path) -> list[summary.CaseInput]:
        return [
            summary.CaseInput(
                spec=case,
                source_gate12a_dir=root / "gate12a" / case.case_id,
                gate12c1_run_dir=root / "gate12c1" / case.case_id,
            )
            for case in summary.CANONICAL_CASES
        ]

    def _endpoint_rows(
        self,
        *,
        supporting_cases: set[str],
        discordant_cases: set[str] | None = None,
    ) -> list[dict]:
        discordant_cases = discordant_cases or set()
        rows: list[dict] = []
        for case in summary.CANONICAL_CASES:
            for q in (1, 2):
                support = case.case_id in supporting_cases
                rows.append(
                    {
                        "case_id": case.case_id,
                        "case_order": case.case_order,
                        "model": case.model,
                        "family": case.family,
                        "compression_rank_q": q,
                        "cycle_coverage_pass": True,
                        "block_coverage_pass": True,
                        "test_status": "informative",
                        "run_support": support,
                        "q_discordant_run": case.case_id in discordant_cases,
                    }
                )
        return rows

    def _case_manifest_payload(self, root: Path) -> dict:
        return {
            "schema_version": summary.CASE_MANIFEST_SCHEMA_VERSION,
            "plan_id": summary.PLAN_ID,
            "cases": [
                {
                    "case_id": case.case_id,
                    "case_order": case.case_order,
                    "model": case.model,
                    "family": case.family,
                    "source_gate12a_dir": str(root / "gate12a" / case.case_id),
                    "expected_source_gate12a_run_id": case.source_gate12a_run_id,
                    "preflight_eligible_cycle_count": case.preflight_eligible_cycle_count,
                    "gate12c1_run_dir": str(root / "gate12c1" / case.case_id),
                }
                for case in summary.CANONICAL_CASES
            ],
        }

    def _build_full_synthetic_grid(
        self,
        root: Path,
        *,
        assoc: float,
        null_median: float,
    ) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        payload = self._case_manifest_payload(root)
        for case in summary.CANONICAL_CASES:
            source_dir = root / "gate12a" / case.case_id
            run_dir = root / "gate12c1" / case.case_id
            cycle_ids = [
                f"{case.case_id}_cycle_{index:06d}"
                for index in range(1, int(case.preflight_eligible_cycle_count) + 1)
            ]
            self._write_gate12a_source(source_dir, case=case, cycle_ids=cycle_ids)
            self._write_gate12c1_run(
                run_dir,
                case=case,
                cycle_ids=cycle_ids,
                assoc=assoc,
                null_median=null_median,
            )
        case_manifest = root / "cases.json"
        write_json(case_manifest, payload)
        return case_manifest

    def _write_gate12a_source(
        self,
        source_dir: Path,
        *,
        case: summary.CanonicalCase,
        cycle_ids: list[str],
    ) -> None:
        source_dir.mkdir(parents=True, exist_ok=True)
        nodes = [
            f"sample_000001:{case.case_id}:a",
            f"sample_000001:{case.case_id}:b",
            f"sample_000001:{case.case_id}:c",
        ]
        r_max = 3
        basis = np.eye(3, dtype=np.float64)
        basis_rows = np.asarray([basis, basis, basis], dtype=np.float64)
        rank_rows = np.asarray([3, 3, 3], dtype=np.int64)
        edge_ids = [
            f"{case.case_id}:z_ab",
            f"{case.case_id}:m_bc",
            f"{case.case_id}:a_ca",
        ]
        node_rows = [
            {
                "node_id": node_id,
                "basis_array_index": index,
                "projector_rank": 3,
                "local_object_status": "defined",
            }
            for index, node_id in enumerate(nodes)
        ]
        transport_rows = []
        matrices = []
        singular_values = []
        active_ranks = []
        for edge_index, edge_id in enumerate(edge_ids):
            source_index = edge_index
            target_index = (edge_index + 1) % 3
            transport_rows.append(
                {
                    "edge_id": edge_id,
                    "source_node_id": nodes[source_index],
                    "target_node_id": nodes[target_index],
                    "relation_kind": "residual_chord" if edge_index == 2 else "trusted_tree",
                    "source_rank": 3,
                    "target_rank": 3,
                    "overlap_rank": 3,
                    "transport_case": "equal_rank_orthogonal",
                    "operator_array_index": edge_index,
                    "compatibility_gap_fro": 0.0,
                }
            )
            matrices.append(np.eye(r_max, dtype=np.float64))
            singular_values.append(np.ones((r_max,), dtype=np.float64))
            active_ranks.append(3)
        cycle_rows = [
            {
                "cycle_id": cycle_id,
                "base_node_id": nodes[0],
                "edge_id_path": sorted(edge_ids),
                "node_id_path": nodes + [nodes[0]],
                "cycle_length": 3,
                "cycle_status": "admissible_explicit_triangle",
            }
            for cycle_id in cycle_ids
        ]
        holonomy_rows = [
            {
                "cycle_id": cycle_id,
                "base_node_id": nodes[0],
                "holonomy_rank": 3,
                "holonomy_residual_fro": 0.25,
                "holonomy_status": "defined",
            }
            for cycle_id in cycle_ids
        ]
        write_json(
            source_dir / gate12c0.DEFAULT_MANIFEST,
            {
                "run_id": case.source_gate12a_run_id,
                "schema_version": "gate12a_discrete_connection_v1",
                "code_git_commit": "synthetic-source",
            },
        )
        write_jsonl(source_dir / gate12c0.DEFAULT_NODE_REGISTRY, node_rows)
        np.savez(
            source_dir / gate12c0.DEFAULT_NODE_ARRAYS,
            basis_factor=basis_rows,
            rank_active=rank_rows,
        )
        write_jsonl(source_dir / gate12c0.DEFAULT_TRANSPORT_REGISTRY, transport_rows)
        np.savez(
            source_dir / gate12c0.DEFAULT_TRANSPORT_ARRAYS,
            transport_matrix_local=np.asarray(matrices, dtype=np.float64),
            overlap_singular_values=np.asarray(singular_values, dtype=np.float64),
            active_rank=np.asarray(active_ranks, dtype=np.int64),
        )
        write_jsonl(source_dir / gate12c0.DEFAULT_CYCLE_REGISTRY, cycle_rows)
        write_jsonl(source_dir / gate12c0.DEFAULT_HOLONOMY_REGISTRY, holonomy_rows)

    def _write_gate12c1_run(
        self,
        run_dir: Path,
        *,
        case: summary.CanonicalCase,
        cycle_ids: list[str],
        assoc: float,
        null_median: float,
    ) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict] = []
        for cycle_id in cycle_ids:
            for root in (0, 1, 2):
                for q in (1, 2):
                    rows.append(
                        self._registry_row(
                            cycle_id=cycle_id,
                            root=root,
                            q=q,
                            operator_index=len(rows),
                            block_id="sample_000001",
                            assoc=assoc,
                            null_median=null_median,
                            holonomy=0.25,
                            edge_gap=0.0,
                        )
                    )
        write_json(run_dir / "manifest.json", self._runner_manifest(case, source_run_id=case.source_gate12a_run_id))
        write_jsonl(run_dir / "triangle_associator_registry.jsonl", rows)
        array = np.zeros((len(rows), 3, 3), dtype=np.float64)
        np.savez(
            run_dir / "triangle_associator_arrays.npz",
            compressed_overlap_left_operator=array,
            compressed_overlap_right_operator=array,
            compressed_overlap_associator_operator=array,
        )
        write_jsonl(run_dir / "cycle_associator_summary.jsonl", [])
        (run_dir / "compression_sweep_summary.csv").write_text(
            "compression_rank_q,row_count\n1,0\n2,0\n",
            encoding="utf-8",
            newline="\n",
        )
        write_json(run_dir / "gauge_stability_summary.json", {"row_count": len(rows)})
        write_jsonl(run_dir / "spectral_orientation_null_summary.jsonl", [])
        write_json(
            run_dir / "gate12c_status.json",
            {
                "schema_version": summary.EXPECTED_RUNNER_SCHEMA,
                "method_id": summary.EXPECTED_RUNNER_METHOD,
                "process_status": "pass",
            },
        )
        (run_dir / "gate12c_read.md").write_text("# Synthetic Gate12C-1\n", encoding="utf-8")
        self._write_runner_checksums(run_dir)

    def _runner_manifest(self, case: summary.CanonicalCase, *, source_run_id: str) -> dict:
        return {
            "run_id": f"synthetic_gate12c1_{case.case_id}",
            "schema_version": summary.EXPECTED_RUNNER_SCHEMA,
            "method_id": summary.EXPECTED_RUNNER_METHOD,
            "code_git_commit": summary.EXPECTED_RUNNER_COMMIT,
            "builder_script_sha256": summary.EXPECTED_RUNNER_SCRIPT_SHA256,
            "run_mode": summary.EXPECTED_RUN_MODE,
            "orientation_null_mode": summary.EXPECTED_ORIENTATION_NULL_MODE,
            "orientation_null_seed": summary.EXPECTED_ORIENTATION_NULL_SEED,
            "orientation_null_requested_draw_count": summary.EXPECTED_ORIENTATION_NULL_REQUESTED_DRAW_COUNT,
            "orientation_null_max_attempt_count": summary.EXPECTED_ORIENTATION_NULL_MAX_ATTEMPT_COUNT,
            "orientation_null_orthogonal_generator": summary.EXPECTED_ORIENTATION_NULL_GENERATOR,
            "orientation_seed_encoding": summary.EXPECTED_ORIENTATION_SEED_ENCODING,
            "tolerances": dict(summary.EXPECTED_TOLERANCES),
            "source_gate12a_run_id": source_run_id,
            "claim_boundary": {
                "scientific_null_excess_threshold_defined": False,
                "type_iii_claim_authorized": False,
                "gate12b_overlay_used": False,
                "rectangular_rank_mismatch_supported": False,
            },
        }

    def _registry_row(
        self,
        *,
        cycle_id: str,
        root: int,
        q: int,
        operator_index: int,
        block_id: str,
        assoc: float = 2.0,
        null_median: float = 1.0,
        holonomy: float = 0.25,
        edge_gap: float = 0.0,
    ) -> dict:
        return {
            "probe_id": f"probe:{operator_index:06d}",
            "cycle_id": cycle_id,
            "canonical_base_node_id": "node_a",
            "evaluation_root_node_id": f"node_{root}",
            "root_rotation_index": root,
            "ordered_node_id_path": ["node_a", "node_b", "node_c", "node_a"],
            "ordered_edge_id_path": ["edge_ab", "edge_bc", "edge_ca"],
            "ordered_relation_kind_path": ["trusted_tree", "trusted_tree", "residual_chord"],
            "cycle_rank": 3,
            "compression_rank_q": q,
            "left_inner_split_gap_rel": 0.5,
            "right_inner_split_gap_rel": 0.5,
            "left_cut_status": "stable",
            "right_cut_status": "stable",
            "truncation_status": "stable_both_active",
            "ordinary_associator_fro": 0.0,
            "no_compression_associator_fro": 0.0,
            "compressed_overlap_associator_fro": assoc,
            "compressed_overlap_associator_rel": 0.1,
            "compressed_overlap_closure_left_fro": 0.1,
            "compressed_overlap_closure_right_fro": 0.1,
            "compressed_overlap_closure_gap_abs": 0.0,
            "gate12a_holonomy_residual_fro": holonomy,
            "edge_compatibility_gap_max": edge_gap,
            "source_sample_block_id": block_id,
            "source_block_status": "single_sample",
            "measurement_status": "measured",
            "control_status": "pass",
            "aggregation_eligible": True,
            "gauge_operator_covariance_fro": 0.0,
            "gauge_scalar_delta_abs": 0.0,
            "gauge_cut_status_preserved": True,
            "gauge_scalar_status": "pass",
            "orientation_null_status": "complete",
            "orientation_null_excess_status": "descriptive_only",
            "orientation_null_requested_draw_count": summary.EXPECTED_ORIENTATION_NULL_REQUESTED_DRAW_COUNT,
            "orientation_null_valid_draw_count": summary.EXPECTED_ORIENTATION_NULL_REQUESTED_DRAW_COUNT,
            "orientation_null_invalid_cut_count": 0,
            "orientation_null_attempt_count": summary.EXPECTED_ORIENTATION_NULL_REQUESTED_DRAW_COUNT,
            "orientation_null_median": null_median,
            "orientation_null_mad": 0.5,
            "orientation_null_mean": null_median,
            "orientation_null_std": 0.1,
            "orientation_null_empirical_p_upper": 0.5,
            "orientation_null_robust_z": 0.0,
            "orientation_null_scale_degenerate": False,
            "operator_array_index": operator_index,
        }

    def _write_runner_checksums(self, run_dir: Path) -> None:
        write_json(
            run_dir / "checksums.json",
            {
                name: summary.sha256_file(run_dir / name)
                for name in summary.RUNNER_OUTPUT_FILES_FOR_CHECKSUMS
            },
        )

    def _assert_checksums(self, out_dir: Path) -> None:
        checksums = read_json(out_dir / summary.OUTPUT_CHECKSUMS)
        for name, digest in checksums.items():
            self.assertEqual(summary.sha256_file(out_dir / name), digest)


if __name__ == "__main__":
    unittest.main()
