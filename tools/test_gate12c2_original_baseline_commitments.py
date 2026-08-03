#!/usr/bin/env python3
"""Primary tests for the Gate12C-2 v0.9 baseline gate."""


from __future__ import annotations

import ast
import contextlib
import copy
import gzip
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gate12c2_original_baseline_commitments as gate
import verify_gate12c2_original_baseline_commitments as verifier


def synthetic_configuration(
    *,
    operational: bool = False,
    compresslevel: int = 6,
) -> tuple[dict[str, object], bytes, dict[str, bytes]]:
    subplan: dict[str, object] = {
        "schema_version": "gate12c2_development_shard_plan_v0.3",
        "contract_version": "v0.3",
        "regime_id": "S0_true_null",
        "master_seed": "synthetic-only",
        "outer_experiment_indices": [0, 1],
        "block_count_schedule": {"block_count_by_case": {"case": 1}},
        "inner_valid_draw_count": 1,
        "effect_strength": None,
        "max_draw_attempts": 4,
        "minimum_log_null_inflation": 0.25,
        "epsilon": 1e-12,
        "diagnostic_kernel": "synthetic",
        "accepted_valid_draw_storage": "synthetic",
        "outer_experiment_schema": gate.OUTER_EXPERIMENT_SCHEMA,
        "seed_namespace_schema": "synthetic",
        "scientific_execution_parameters": {"finite": 1.0},
        "implementation_sha256": {"a.py": "1" * 64},
        "numerical_environment": {
            "blas_thread_limit": 1,
            "numpy_build": "synthetic",
        },
    }
    subplan["plan_payload_sha256"] = gate.sha256_bytes(
        gate.canonical_json_bytes(subplan)
    )
    shard_bytes: dict[str, bytes] = {}
    index_rows = []
    projection_rows = []
    for outer_id in (0, 1):
        pipeline = {
            "schema_version": "synthetic_pipeline",
            "epistemic_status": "synthetic_only",
            "outer_monte_carlo_unit": "synthetic",
            "alternative": "synthetic",
            "holm_alpha": 0.05,
            "zero_tolerance": 0.0,
            "endpoint_count": 24,
            "q_directional_support_count": outer_id,
            "any_endpoint_support": bool(outer_id),
            "supporting_run_count": outer_id,
            "any_run_support": bool(outer_id),
            "q_discordant_run_count": 0,
            "grid_outcome": (
                "no_directional_support"
                if outer_id == 0
                else "partial_or_structured"
            ),
            "claim_promotion": False,
            "promotion_outcomes": ["broad_replicated", "strong_broad"],
            "partial_or_structured_is_promotional": False,
            "endpoint_rows": [],
            "run_rows": [],
        }
        execution = {
            "plan_payload_sha256": subplan["plan_payload_sha256"],
            "contract_version": subplan["contract_version"],
            "surface_id": "development",
            "locked_execution_authorized": False,
            "real_held_out_execution_authorized": False,
            "N2_open": False,
            "N3_open": False,
            "public_claim": False,
            "regime_id": subplan["regime_id"],
            "outer_experiment_index": outer_id,
            "block_count_schedule": subplan["block_count_schedule"],
            "inner_valid_draw_count": subplan["inner_valid_draw_count"],
            "effect_strength": subplan["effect_strength"],
            "configured_max_draw_attempts": subplan["max_draw_attempts"],
            "resolved_max_draw_attempts": 4,
            "minimum_log_null_inflation": subplan[
                "minimum_log_null_inflation"
            ],
            "epsilon": subplan["epsilon"],
            "diagnostic_kernel": subplan["diagnostic_kernel"],
            "accepted_valid_draw_storage": subplan[
                "accepted_valid_draw_storage"
            ],
            "outer_experiment_schema": subplan[
                "outer_experiment_schema"
            ],
            "seed_namespace_schema": subplan["seed_namespace_schema"],
            "scientific_execution_parameters": subplan[
                "scientific_execution_parameters"
            ],
            "implementation_sha256": subplan["implementation_sha256"],
            "numerical_environment_sha256": gate.sha256_bytes(
                gate.canonical_json_bytes(subplan["numerical_environment"])
            ),
            "master_seed_sha256": gate.sha256_bytes(
                str(subplan["master_seed"]).encode("utf-8")
            ),
            "schema_version": "gate12c2_result_execution_contract_v0.1",
        }
        result = {
            "schema_version": gate.OUTER_EXPERIMENT_SCHEMA,
            "epistemic_status": "development_outer_experiment_only",
            "contract_version": subplan["contract_version"],
            "surface_id": "development",
            "locked_execution_authorized": False,
            "regime_id": subplan["regime_id"],
            "effect_strength": None,
            "outer_experiment_index": outer_id,
            "block_count_schedule": subplan["block_count_schedule"],
            "inner_valid_draw_count": 1,
            "max_draw_attempts": 4,
            "diagnostic_kernel": "synthetic",
            "accepted_valid_draw_storage": "synthetic",
            "accepted_valid_draw_order": (
                "draw_attempt_order_first_required_valid"
            ),
            "dependency_structure": "synthetic",
            "alternative": "synthetic",
            "case_receipts": [],
            "endpoint_receipts": [],
            "pipeline_decision": pipeline,
            "numerical_execution_contract": {
                "blas_thread_limit": 1,
                "thread_environment": {
                    "MKL_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                },
                "active_blas_thread_limit_verified": True,
                "numpy_build": "synthetic",
                "scientific_execution_parameters": {"finite": 1.0},
                "guarantee_scope": (
                    "same_frozen_software_and_numerical_environment"
                ),
                "cross_environment_bitwise_determinism_claimed": False,
            },
            "execution_configuration_contract": execution,
        }
        result_hash = gate.sha256_bytes(gate.canonical_json_bytes(result))
        shard: dict[str, object] = {
            "schema_version": gate.SHARD_SCHEMA,
            "epistemic_status": "development_outer_shard_only",
            "surface_id": "development",
            "locked_execution_authorized": False,
            "plan_payload_sha256": subplan["plan_payload_sha256"],
            "outer_experiment_index": outer_id,
            "result_payload_sha256": result_hash,
            "result": result,
        }
        shard["shard_payload_sha256"] = gate.sha256_bytes(
            gate.canonical_json_bytes(shard)
        )
        compressed = gzip.compress(
            gate.canonical_json_bytes(shard),
            compresslevel=compresslevel,
            mtime=0,
        )
        relative = f"shards/outer-{outer_id:06d}.json.gz"
        shard_bytes[relative] = compressed
        decision = gate.reconstruct_decision(result)
        index_row: dict[str, object] = {
            "outer_experiment_index": outer_id,
            "relative_path": relative,
            "compressed_file_sha256": gate.sha256_bytes(compressed),
            "compressed_bytes": len(compressed),
            "shard_payload_sha256": shard["shard_payload_sha256"],
            "result_payload_sha256": result_hash,
            "reused_existing_shard": False,
            "decision": decision,
        }
        if operational:
            index_row["operational_metrics"] = {
                "synthetic_wall_seconds": 1.0 + outer_id
            }
        index_rows.append(index_row)
        projection_rows.append(
            {
                "outer_experiment_index": outer_id,
                "result_payload_sha256": result_hash,
                "decision": decision,
            }
        )
    projection = gate.scientific_projection(subplan, projection_rows)
    index: dict[str, object] = {
        "schema_version": gate.INDEX_SCHEMA,
        "epistemic_status": "development_shard_index_only",
        "surface_id": "development",
        "locked_execution_authorized": False,
        "plan_payload_sha256": subplan["plan_payload_sha256"],
        "worker_count_operational_only": 1,
        "merge_order": "ascending_outer_experiment_index",
        "outer_experiment_count": 2,
        "all_outer_indices_present": True,
        "shards": index_rows,
        "scientific_projection_schema_version": (
            gate.SCIENTIFIC_PROJECTION_SCHEMA
        ),
        "scientific_projection_sha256": gate.sha256_bytes(
            gate.canonical_json_bytes(projection)
        ),
    }
    if operational:
        index["operational_execution_metrics"] = {"worker_seconds": 2.0}
    index["index_payload_sha256"] = gate.sha256_bytes(
        gate.canonical_json_bytes(index)
    )
    return subplan, gate.canonical_json_bytes(index), shard_bytes


def _synthetic_s2_endpoint() -> dict[str, object]:
    medians = {
        arm: {field: 0.0 for field in gate.S2_COMPONENT_FIELDS}
        for arm in gate.S2_COMPONENT_ARMS
    }
    coverage = {
        arm: {
            field: {
                "expected_count": 1,
                "defined_count": 1,
                "degenerate_count": 0,
                "unexpected_missing_count": 0,
                "nonfinite_count": 0,
            }
            for field in gate.S2_COMPONENT_FIELDS
        }
        for arm in gate.S2_COMPONENT_ARMS
    }
    return {
        "minimum_log_null_inflation": 0.25,
        "expected_block_count": 1,
        "component_medians": medians,
        "component_coverage": coverage,
    }


def synthetic_s2_configuration(
) -> tuple[dict[str, object], bytes, dict[str, bytes]]:
    subplan, index_raw, shard_bytes = synthetic_configuration()
    subplan["regime_id"] = "S2_null_inflation"
    subplan.pop("plan_payload_sha256")
    subplan["plan_payload_sha256"] = gate.sha256_bytes(
        gate.canonical_json_bytes(subplan)
    )
    index = gate.strict_json_loads(index_raw, canonical=True)
    index["plan_payload_sha256"] = subplan["plan_payload_sha256"]
    for row in index["shards"]:
        relative = row["relative_path"]
        shard = gate.strict_json_loads(
            gzip.decompress(shard_bytes[relative]), canonical=True
        )
        result = shard["result"]
        for field in (
            "effect_strength",
            "dependency_structure",
            "alternative",
            "case_receipts",
            "endpoint_receipts",
            "pipeline_decision",
        ):
            result.pop(field)
        result.update(
            {
                "regime_id": "S2_null_inflation",
                "observed_process_modified": False,
                "paired_null_arms": [
                    gate.N1_NULL_ARM_ID,
                    gate.S2_NULL_ARM_ID,
                ],
                "identified_case_count": 0,
                "breadth_pass": False,
                "identification_success": False,
                "endpoint_rows": [
                    _synthetic_s2_endpoint() for _ in range(24)
                ],
                "case_rows": [{} for _ in range(12)],
            }
        )
        execution = result["execution_configuration_contract"]
        execution["plan_payload_sha256"] = subplan["plan_payload_sha256"]
        execution["regime_id"] = "S2_null_inflation"
        shard["plan_payload_sha256"] = subplan["plan_payload_sha256"]
        result_hash = gate.sha256_bytes(gate.canonical_json_bytes(result))
        shard["result_payload_sha256"] = result_hash
        shard.pop("shard_payload_sha256")
        shard["shard_payload_sha256"] = gate.sha256_bytes(
            gate.canonical_json_bytes(shard)
        )
        compressed = gzip.compress(
            gate.canonical_json_bytes(shard), compresslevel=6, mtime=0
        )
        shard_bytes[relative] = compressed
        row.update(
            {
                "compressed_file_sha256": gate.sha256_bytes(compressed),
                "compressed_bytes": len(compressed),
                "shard_payload_sha256": shard["shard_payload_sha256"],
                "result_payload_sha256": result_hash,
                "decision": gate.reconstruct_decision(result),
            }
        )
    projection_rows = [
        {
            "outer_experiment_index": row["outer_experiment_index"],
            "result_payload_sha256": row["result_payload_sha256"],
            "decision": row["decision"],
        }
        for row in index["shards"]
    ]
    index["scientific_projection_sha256"] = gate.sha256_bytes(
        gate.canonical_json_bytes(
            gate.scientific_projection(subplan, projection_rows)
        )
    )
    index.pop("index_payload_sha256")
    index["index_payload_sha256"] = gate.sha256_bytes(
        gate.canonical_json_bytes(index)
    )
    return subplan, gate.canonical_json_bytes(index), shard_bytes


def mutate_s2_configuration(
    mutation: object,
) -> tuple[dict[str, object], bytes, dict[str, bytes]]:
    subplan, index_raw, shard_bytes = synthetic_s2_configuration()
    index = gate.strict_json_loads(index_raw, canonical=True)
    row = index["shards"][0]
    relative = row["relative_path"]
    shard = gate.strict_json_loads(
        gzip.decompress(shard_bytes[relative]), canonical=True
    )
    endpoint = shard["result"]["endpoint_rows"][0]
    mutation(endpoint)
    result = shard["result"]
    result_hash = gate.sha256_bytes(gate.canonical_json_bytes(result))
    shard["result_payload_sha256"] = result_hash
    shard.pop("shard_payload_sha256")
    shard["shard_payload_sha256"] = gate.sha256_bytes(
        gate.canonical_json_bytes(shard)
    )
    compressed = gzip.compress(
        gate.canonical_json_bytes(shard), compresslevel=6, mtime=0
    )
    shard_bytes[relative] = compressed
    row.update(
        {
            "compressed_file_sha256": gate.sha256_bytes(compressed),
            "compressed_bytes": len(compressed),
            "shard_payload_sha256": shard["shard_payload_sha256"],
            "result_payload_sha256": result_hash,
            "decision": gate.reconstruct_decision(result),
        }
    )
    projection_rows = [
        {
            "outer_experiment_index": item["outer_experiment_index"],
            "result_payload_sha256": item["result_payload_sha256"],
            "decision": item["decision"],
        }
        for item in index["shards"]
    ]
    index["scientific_projection_sha256"] = gate.sha256_bytes(
        gate.canonical_json_bytes(
            gate.scientific_projection(subplan, projection_rows)
        )
    )
    index.pop("index_payload_sha256")
    index["index_payload_sha256"] = gate.sha256_bytes(
        gate.canonical_json_bytes(index)
    )
    return subplan, gate.canonical_json_bytes(index), shard_bytes


def derive_both(
    fixture: tuple[dict[str, object], bytes, dict[str, bytes]]
) -> tuple[dict[str, object], dict[str, object]]:
    subplan, index, shards = fixture
    extracted = gate.derive_configuration_commitment(
        configuration_id="S0_true_null__d1",
        subplan=subplan,
        index_raw=index,
        shard_raw_by_relative_path=shards,
        result_validator=lambda _plan, _result, _outer: None,
    )
    verified = verifier.independent_configuration_commitment(
        configuration_id="S0_true_null__d1",
        subplan=subplan,
        index_raw=index,
        shard_raw_by_relative_path=shards,
    )
    return extracted, verified


def synthetic_schema_mutation(
    mutation_id: str,
) -> tuple[dict[str, object], bytes, dict[str, bytes]]:
    subplan, index_raw, shard_bytes = synthetic_configuration(operational=True)
    index = gate.strict_json_loads(index_raw, canonical=True)
    target_path = "shards/outer-000000.json.gz"
    mutated_shard: dict[str, object] | None = None
    if mutation_id == "SCHEMA-INDEX-EPISTEMIC-STATUS-FOREIGN":
        index["epistemic_status"] = "synthetic_sensitive_marker"
    elif mutation_id == "SCHEMA-INDEX-WORKER-COUNT-ZERO":
        index["worker_count_operational_only"] = 0
    elif mutation_id == "SCHEMA-INDEX-OPERATIONAL-EXECUTION-METRICS-NONOBJECT":
        index["operational_execution_metrics"] = "synthetic_sensitive_marker"
    elif mutation_id == "SCHEMA-RESULT-ROW-OPERATIONAL-METRICS-NONOBJECT":
        index["shards"][0]["operational_metrics"] = "synthetic_sensitive_marker"
    else:
        mutated_shard = gate.strict_json_loads(
            gzip.decompress(shard_bytes[target_path]), canonical=True
        )
        if mutation_id == "SCHEMA-SHARD-EPISTEMIC-STATUS-FOREIGN":
            mutated_shard["epistemic_status"] = "synthetic_sensitive_marker"
        elif mutation_id == "SCHEMA-PIPELINE-ENDPOINT-COUNT-NEGATIVE":
            mutated_shard["result"]["pipeline_decision"]["endpoint_count"] = -1
        elif mutation_id == "SCHEMA-PROMOTIONAL-FLAG-STRING":
            mutated_shard["result"]["pipeline_decision"][
                "partial_or_structured_is_promotional"
            ] = "synthetic_sensitive_marker"
        else:
            raise AssertionError(mutation_id)
    if mutated_shard is not None:
        result = mutated_shard["result"]
        result_hash = gate.sha256_bytes(gate.canonical_json_bytes(result))
        mutated_shard["result_payload_sha256"] = result_hash
        mutated_shard.pop("shard_payload_sha256")
        mutated_shard["shard_payload_sha256"] = gate.sha256_bytes(
            gate.canonical_json_bytes(mutated_shard)
        )
        compressed = gzip.compress(
            gate.canonical_json_bytes(mutated_shard), compresslevel=6, mtime=0
        )
        shard_bytes[target_path] = compressed
        row = index["shards"][0]
        row["compressed_file_sha256"] = gate.sha256_bytes(compressed)
        row["compressed_bytes"] = len(compressed)
        row["shard_payload_sha256"] = mutated_shard["shard_payload_sha256"]
        row["result_payload_sha256"] = result_hash
        row["decision"] = gate.reconstruct_decision(result)
        projection_rows = [
            {
                "outer_experiment_index": item["outer_experiment_index"],
                "result_payload_sha256": item["result_payload_sha256"],
                "decision": item["decision"],
            }
            for item in index["shards"]
        ]
        index["scientific_projection_sha256"] = gate.sha256_bytes(
            gate.canonical_json_bytes(
                gate.scientific_projection(subplan, projection_rows)
            )
        )
    index.pop("index_payload_sha256")
    index["index_payload_sha256"] = gate.sha256_bytes(
        gate.canonical_json_bytes(index)
    )
    return subplan, gate.canonical_json_bytes(index), shard_bytes



class CanonicalContractTests(unittest.TestCase):
    def test_canonical_json_exact(self) -> None:
        self.assertEqual(
            gate.canonical_json_bytes({"b": 1, "a": "\u00e9"}),
            b'{"a":"\xc3\xa9","b":1}',
        )
        self.assertEqual(
            gate.canonical_json_bytes({"b": 1, "a": "\u00e9"}),
            verifier.verifier_canonical_bytes({"a": "\u00e9", "b": 1}),
        )

    def test_duplicate_keys_rejected_by_both_paths(self) -> None:
        raw = b'{"a":1,"a":2}'
        with self.assertRaises(gate.Gate12C2OriginalBaselineError) as first:
            gate.strict_json_loads(raw)
        with self.assertRaises(verifier.IndependentVerificationError) as second:
            verifier.verifier_json(raw)
        self.assertEqual(first.exception.code, "DUPLICATE_JSON_KEY")
        self.assertEqual(second.exception.code, "DUPLICATE_JSON_KEY")

    def test_nonfinite_and_noncanonical_json_rejected(self) -> None:
        for raw in (b'{"a":NaN}', b'{"a":Infinity}', b'{ "a":1 }'):
            with self.subTest(raw=raw):
                with self.assertRaises(
                    gate.Gate12C2OriginalBaselineError
                ):
                    gate.strict_json_loads(raw, canonical=True)
                with self.assertRaises(
                    verifier.IndependentVerificationError
                ):
                    verifier.verifier_json(raw)

    def test_boolean_is_not_an_integer(self) -> None:
        for value in (True, False, "1", 1.0):
            with self.subTest(value=value):
                with self.assertRaises(
                    gate.Gate12C2OriginalBaselineError
                ):
                    gate.require_int(value)
                with self.assertRaises(
                    verifier.IndependentVerificationError
                ):
                    verifier._integer(value)

    def test_self_hash_and_receipt_lf_domains(self) -> None:
        payload = gate.add_self_hash({"a": 1}, "payload_sha256")
        self.assertEqual(
            gate.verify_self_hash(payload, "payload_sha256"),
            payload["payload_sha256"],
        )
        raw = gate.canonical_receipt_bytes(payload)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertFalse(raw.endswith(b"\n\n"))
        changed = dict(payload)
        changed["a"] = 2
        with self.assertRaises(gate.Gate12C2OriginalBaselineError):
            gate.verify_self_hash(changed, "payload_sha256")

    def test_single_gzip_member_required(self) -> None:
        value = {"a": 1}
        one = gzip.compress(gate.canonical_json_bytes(value), mtime=0)
        self.assertEqual(gate.strict_gzip_json(one), value)
        self.assertEqual(verifier.verifier_gzip_json(one), value)
        joined = one + one
        with self.assertRaises(gate.Gate12C2OriginalBaselineError):
            gate.strict_gzip_json(joined)
        with self.assertRaises(verifier.IndependentVerificationError):
            verifier.verifier_gzip_json(joined)


class FrozenPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = gate.load_frozen_plan()

    def test_evidence_hashes_and_nested_digests(self) -> None:
        self.assertEqual(
            self.plan["plan_payload_sha256"], gate.PLAN_PAYLOAD_SHA256
        )
        self.assertEqual(
            gate.sha256_bytes(
                gate.canonical_json_bytes(
                    self.plan["artifact_path_surface"]
                )
            ),
            gate.ARTIFACT_PATH_SURFACE_SHA256,
        )
        self.assertEqual(
            gate.sha256_bytes(
                gate.canonical_json_bytes(
                    self.plan["implementation_author_separation_contract"]
                )
            ),
            gate.IMPLEMENTATION_AUTHOR_SEPARATION_SHA256,
        )
        self.assertEqual(
            gate.sha256_bytes(
                gate.canonical_json_bytes(
                    self.plan["implementation_trust_model_contract"]
                )
            ),
            gate.IMPLEMENTATION_TRUST_MODEL_SHA256,
        )
        nested = gate._walk_named_values(
            self.plan, "artifact_path_surface_sha256"
        )
        self.assertTrue(nested)
        self.assertEqual(set(nested), {gate.ARTIFACT_PATH_SURFACE_SHA256})

    def test_formal_design_pass_is_exact(self) -> None:
        receipt = gate.validate_formal_design_pass(self.plan)
        self.assertEqual(receipt["outcome_kind"], "pass")
        self.assertEqual(receipt["P0_count"], 0)
        self.assertEqual(receipt["P1_count"], 0)
        self.assertFalse(receipt["protected_payload_inspected"])
        self.assertFalse(receipt["scientific_values_inspected"])

    def test_state_event_transition_algebra_is_closed(self) -> None:
        model = self.plan["state_model"]
        self.assertEqual(len(model["states"]), len(set(model["states"])))
        self.assertEqual(len(model["events"]), len(set(model["events"])))
        used = {row[1] for row in model["transitions"]}
        self.assertEqual(used, set(model["events"]))
        terminals = set(model["terminal_states"])
        self.assertFalse(
            any(row[0] in terminals for row in model["transitions"])
        )
        for source, event, target in model["transitions"]:
            self.assertEqual(
                gate.transition_state(self.plan, source, event), target
            )

    def test_all_failure_rows_profiles_codes_and_phases_are_closed(self) -> None:
        matrix = self.plan["failure_matrix"]
        self.assertEqual(len(matrix), 94)
        self.assertEqual(
            {row["failure_code"] for row in matrix}, gate.FAILURE_CODES
        )
        self.assertEqual(
            {row["failure_phase"] for row in matrix},
            set(self.plan["failure_phases"]),
        )
        self.assertEqual(
            len({gate.canonical_json_bytes(row) for row in matrix}), 94
        )
        for row in matrix:
            self.assertIn(
                row["availability_profile"],
                self.plan["failure_evidence_availability_profiles"][
                    row["scope"]
                ],
            )

    def test_all_lifecycle_phases_and_594_cells_classify(self) -> None:
        lifecycle = self.plan["artifact_lifecycle_contract"]
        self.assertEqual(len(lifecycle["stable_phases"]), 33)
        self.assertEqual(lifecycle["cell_count"], 594)
        roles = lifecycle["roles"]
        for phase in lifecycle["stable_phases"]:
            observations = {
                role: gate.ArtifactObservation(
                    final_exists=role in phase["must_exist"],
                    outcome=(
                        "success"
                        if phase["required_outcomes"].get(role)
                        == "success_or_failure"
                        else phase["required_outcomes"].get(role)
                    ),
                )
                for role in roles
            }
            temporal = phase["temporal_predicate"]
            live = phase["liveness_predicate"]
            liveness = (
                "ACTIVE"
                if live == "ACTIVE_exact_owner"
                else "DEAD"
                if live == "DEAD_or_UNKNOWN"
                else "not_applicable"
            )
            with self.subTest(phase=phase["phase"]):
                self.assertEqual(
                    gate.classify_lifecycle_surface(
                        self.plan,
                        observations,
                        temporal_predicate=temporal,
                        liveness=liveness,
                    ),
                    phase["phase"],
                )

    def test_artifact_rows_are_exact_and_path_unique(self) -> None:
        rows = self.plan["artifact_path_surface"]
        self.assertEqual(len(rows), 18)
        self.assertEqual(
            [row["role"] for row in rows],
            sorted(row["role"] for row in rows),
        )
        all_paths = [
            path
            for row in rows
            for path in (row["final_path"], row["pending_path"])
        ]
        self.assertEqual(len(all_paths), len(set(all_paths)))
        for row in rows:
            self.assertEqual(
                row["pending_path"], row["final_path"] + ".pending-v0.9"
            )

    def test_exact_configuration_surface_counts(self) -> None:
        configurations = self.plan["configuration_surface"]
        self.assertEqual(len(configurations), 9)
        self.assertEqual(
            sum(row["outer_experiment_count"] for row in configurations),
            768,
        )
        self.assertEqual(
            gate.sha256_bytes(gate.canonical_json_bytes(configurations)),
            gate.CONFIGURATION_SURFACE_SHA256,
        )


class IndependentCommitmentTests(unittest.TestCase):
    def test_complete_invariant_manifests_and_seven_mutations(self) -> None:
        self.assertEqual(
            gate.EXTRACTOR_SCHEMA_INVARIANT_MANIFEST,
            verifier.VERIFIER_SCHEMA_INVARIANT_MANIFEST,
        )
        self.assertEqual(
            len(gate.EXTRACTOR_SCHEMA_INVARIANT_MANIFEST),
            len(set(gate.EXTRACTOR_SCHEMA_INVARIANT_MANIFEST)),
        )
        mutation_ids = (
            "SCHEMA-INDEX-EPISTEMIC-STATUS-FOREIGN",
            "SCHEMA-INDEX-WORKER-COUNT-ZERO",
            "SCHEMA-INDEX-OPERATIONAL-EXECUTION-METRICS-NONOBJECT",
            "SCHEMA-RESULT-ROW-OPERATIONAL-METRICS-NONOBJECT",
            "SCHEMA-SHARD-EPISTEMIC-STATUS-FOREIGN",
            "SCHEMA-PIPELINE-ENDPOINT-COUNT-NEGATIVE",
            "SCHEMA-PROMOTIONAL-FLAG-STRING",
        )
        for mutation_id in mutation_ids:
            subplan, index_raw, shards = synthetic_schema_mutation(mutation_id)
            with self.subTest(mutation_id=mutation_id, path="extractor"):
                with self.assertRaises(
                    gate.Gate12C2OriginalBaselineError
                ) as caught:
                    gate.derive_configuration_commitment(
                        configuration_id="S0_true_null__d1",
                        subplan=subplan,
                        index_raw=index_raw,
                        shard_raw_by_relative_path=shards,
                        result_validator=lambda *_args: None,
                    )
                self.assertEqual(caught.exception.code, "INPUT_SCHEMA_INVALID")
                self.assertNotIn("synthetic_sensitive_marker", str(caught.exception))
            with self.subTest(mutation_id=mutation_id, path="verifier"):
                with self.assertRaises(
                    verifier.IndependentVerificationError
                ) as caught:
                    verifier.independent_configuration_commitment(
                        configuration_id="S0_true_null__d1",
                        subplan=subplan,
                        index_raw=index_raw,
                        shard_raw_by_relative_path=shards,
                    )
                self.assertEqual(caught.exception.code, "INPUT_SCHEMA_INVALID")
                self.assertNotIn("synthetic_sensitive_marker", str(caught.exception))

    def test_independent_paths_rederive_identical_commitments(self) -> None:
        extracted, verified = derive_both(synthetic_configuration())
        self.assertEqual(extracted, verified)
        self.assertEqual(
            set(extracted),
            {
                "configuration_id",
                "outer_experiment_count",
                "outer_id_surface_sha256",
                "result_commitment_surface_sha256",
                "scientific_projection_sha256",
                "semantic_index_commitment_v0_1_sha256",
            },
        )

    def test_outer_id_and_result_domains_are_exact(self) -> None:
        subplan, index, shards = synthetic_configuration()
        extracted, _verified = derive_both((subplan, index, shards))
        self.assertEqual(
            extracted["outer_id_surface_sha256"],
            gate.sha256_bytes(gate.canonical_json_bytes([0, 1])),
        )
        parsed = gate.strict_json_loads(index, canonical=True)
        expected_rows = [
            {
                "outer_experiment_index": row["outer_experiment_index"],
                "result_payload_sha256": row["result_payload_sha256"],
                "shard_payload_sha256": row["shard_payload_sha256"],
            }
            for row in parsed["shards"]
        ]
        self.assertEqual(
            extracted["result_commitment_surface_sha256"],
            gate.sha256_bytes(gate.canonical_json_bytes(expected_rows)),
        )

    def test_operational_fields_do_not_change_commitments(self) -> None:
        plain, _ = derive_both(synthetic_configuration(operational=False))
        operational, _ = derive_both(
            synthetic_configuration(operational=True)
        )
        self.assertEqual(plain, operational)

    def test_recompression_does_not_enter_result_surface(self) -> None:
        base = synthetic_configuration(compresslevel=1)
        alternate = synthetic_configuration(compresslevel=9)
        first, _ = derive_both(base)
        second, _ = derive_both(alternate)
        self.assertEqual(
            first["result_commitment_surface_sha256"],
            second["result_commitment_surface_sha256"],
        )
        self.assertNotEqual(base[1], alternate[1])

    def test_index_and_shard_unknown_fields_are_rejected(self) -> None:
        subplan, index_raw, shards = synthetic_configuration()
        index = gate.strict_json_loads(index_raw, canonical=True)
        index["unexpected"] = False
        index.pop("index_payload_sha256")
        index["index_payload_sha256"] = gate.sha256_bytes(
            gate.canonical_json_bytes(index)
        )
        with self.assertRaises(gate.Gate12C2OriginalBaselineError):
            gate.derive_configuration_commitment(
                configuration_id="S0_true_null__d1",
                subplan=subplan,
                index_raw=gate.canonical_json_bytes(index),
                shard_raw_by_relative_path=shards,
                result_validator=lambda *_args: None,
            )
        with self.assertRaises(verifier.IndependentVerificationError):
            verifier.independent_configuration_commitment(
                configuration_id="S0_true_null__d1",
                subplan=subplan,
                index_raw=gate.canonical_json_bytes(index),
                shard_raw_by_relative_path=shards,
            )

    def test_missing_duplicate_and_zero_coverage_surfaces_fail(self) -> None:
        subplan, index, shards = synthetic_configuration()
        missing = dict(shards)
        missing.pop(next(iter(missing)))
        with self.assertRaises(gate.Gate12C2OriginalBaselineError) as error:
            gate.derive_configuration_commitment(
                configuration_id="S0_true_null__d1",
                subplan=subplan,
                index_raw=index,
                shard_raw_by_relative_path=missing,
                result_validator=lambda *_args: None,
            )
        self.assertEqual(error.exception.code, "ZERO_COVERAGE")

    def test_common_mode_wrong_projection_is_detected_independently(self) -> None:
        fixture = synthetic_configuration()
        _extracted, expected = derive_both(fixture)
        original = gate.scientific_projection

        def wrong(
            subplan: Mapping[str, object],
            rows: Sequence[Mapping[str, object]],
        ) -> dict[str, object]:
            value = original(subplan, rows)
            value["schema_version"] = "wrong"
            return value

        with mock.patch.object(gate, "scientific_projection", wrong):
            with self.assertRaises(
                gate.Gate12C2OriginalBaselineError
            ) as error:
                derive_both(fixture)
        self.assertEqual(error.exception.code, "COMMITMENT_MISMATCH")
        independently_verified = verifier.independent_configuration_commitment(
            configuration_id="S0_true_null__d1",
            subplan=fixture[0],
            index_raw=fixture[1],
            shard_raw_by_relative_path=fixture[2],
        )
        self.assertEqual(independently_verified, expected)

    def test_extractor_and_verifier_have_no_shared_implementation_import(self) -> None:
        source = Path(verifier.__file__).read_text(encoding="utf-8")
        syntax_tree = ast.parse(source)
        imported_modules = {
            alias.name
            for node in ast.walk(syntax_tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module
            for node in ast.walk(syntax_tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        self.assertFalse(
            imported_modules
            & {
                "gate12c2_original_baseline_commitments",
                "gate12c2_development_shards",
                "gate12c2_synthetic_lab",
            }
        )
        self.assertIsNot(
            gate.derive_configuration_commitment,
            verifier.independent_configuration_commitment,
        )


class S2ComponentSchemaProductionTests(unittest.TestCase):
    def test_nested_component_surface_passes_both_production_paths(self) -> None:
        subplan, index_raw, shards = synthetic_s2_configuration()
        extracted = gate.derive_configuration_commitment(
            configuration_id="S2_null_inflation__d1",
            subplan=subplan,
            index_raw=index_raw,
            shard_raw_by_relative_path=shards,
            result_validator=lambda *_args: None,
        )
        verified = verifier.independent_configuration_commitment(
            configuration_id="S2_null_inflation__d1",
            subplan=subplan,
            index_raw=index_raw,
            shard_raw_by_relative_path=shards,
        )
        self.assertEqual(extracted, verified)

    def test_nested_component_mutations_rehashed_end_to_end_are_rejected(
        self,
    ) -> None:
        marker = "synthetic_sensitive_marker"
        mutations = {
            "missing_median_arm": (
                lambda endpoint: endpoint["component_medians"].pop("N1")
            ),
            "extra_median_arm": (
                lambda endpoint: endpoint["component_medians"].__setitem__(
                    marker, {}
                )
            ),
            "wrong_median_arm_type": (
                lambda endpoint: endpoint["component_medians"].__setitem__(
                    "observed", []
                )
            ),
            "missing_median_field": (
                lambda endpoint: endpoint["component_medians"][
                    "observed"
                ].pop("a_q")
            ),
            "extra_median_field": (
                lambda endpoint: endpoint["component_medians"][
                    "observed"
                ].__setitem__(marker, 0.0)
            ),
            "wrong_median_type": (
                lambda endpoint: endpoint["component_medians"][
                    "observed"
                ].__setitem__("a_q", marker)
            ),
            "missing_coverage_arm": (
                lambda endpoint: endpoint["component_coverage"].pop("N1")
            ),
            "extra_coverage_arm": (
                lambda endpoint: endpoint["component_coverage"].__setitem__(
                    marker, {}
                )
            ),
            "missing_coverage_field": (
                lambda endpoint: endpoint["component_coverage"]["N1"].pop(
                    "a_q"
                )
            ),
            "wrong_coverage_field_type": (
                lambda endpoint: endpoint["component_coverage"]["N1"].__setitem__(
                    "a_q", marker
                )
            ),
            "missing_count_field": (
                lambda endpoint: endpoint["component_coverage"]["N1"][
                    "a_q"
                ].pop("defined_count")
            ),
            "extra_count_field": (
                lambda endpoint: endpoint["component_coverage"]["N1"][
                    "a_q"
                ].__setitem__(marker, 0)
            ),
            "bool_as_count": (
                lambda endpoint: endpoint["component_coverage"]["N1"][
                    "a_q"
                ].__setitem__("defined_count", True)
            ),
            "count_total_mismatch": (
                lambda endpoint: endpoint["component_coverage"]["N1"][
                    "a_q"
                ].__setitem__("defined_count", 0)
            ),
            "swapped_median_nesting": (
                lambda endpoint: endpoint.__setitem__(
                    "component_medians",
                    {
                        field: {
                            arm: 0.0 for arm in gate.S2_COMPONENT_ARMS
                        }
                        for field in gate.S2_COMPONENT_FIELDS
                    },
                )
            ),
        }
        paths = (
            (
                gate.derive_configuration_commitment,
                gate.Gate12C2OriginalBaselineError,
                {"result_validator": lambda *_args: None},
            ),
            (
                verifier.independent_configuration_commitment,
                verifier.IndependentVerificationError,
                {},
            ),
        )
        for mutation_id, mutation in mutations.items():
            fixture = mutate_s2_configuration(mutation)
            for function, error_type, extra in paths:
                with self.subTest(
                    mutation_id=mutation_id,
                    path=function.__module__,
                ):
                    with self.assertRaises(error_type) as caught:
                        function(
                            configuration_id="S2_null_inflation__d1",
                            subplan=fixture[0],
                            index_raw=fixture[1],
                            shard_raw_by_relative_path=fixture[2],
                            **extra,
                        )
                    self.assertEqual(
                        caught.exception.code, "INPUT_SCHEMA_INVALID"
                    )
                    self.assertNotIn(marker, str(caught.exception))


    def test_conditionally_defined_all_degenerate_passes_both_paths(
        self,
    ) -> None:
        def make_all_degenerate(endpoint: dict[str, object]) -> None:
            for arm in gate.S2_COMPONENT_ARMS:
                for field in gate.S2_CONDITIONALLY_DEFINED_COMPONENT_FIELDS:
                    endpoint["component_medians"][arm][field] = None
                    endpoint["component_coverage"][arm][field].update(
                        {
                            "defined_count": 0,
                            "degenerate_count": 1,
                        }
                    )

        fixture = mutate_s2_configuration(make_all_degenerate)
        extracted = gate.derive_configuration_commitment(
            configuration_id="S2_null_inflation__d1",
            subplan=fixture[0],
            index_raw=fixture[1],
            shard_raw_by_relative_path=fixture[2],
            result_validator=lambda *_args: None,
        )
        verified = verifier.independent_configuration_commitment(
            configuration_id="S2_null_inflation__d1",
            subplan=fixture[0],
            index_raw=fixture[1],
            shard_raw_by_relative_path=fixture[2],
        )
        self.assertEqual(extracted, verified)

    def test_s2_semantic_mutations_rehashed_end_to_end_are_rejected(
        self,
    ) -> None:
        paths = (
            (
                gate.derive_configuration_commitment,
                gate.Gate12C2OriginalBaselineError,
                {"result_validator": lambda *_args: None},
            ),
            (
                verifier.independent_configuration_commitment,
                verifier.IndependentVerificationError,
                {},
            ),
        )
        attacks_by_family = {
            "always": (
                "negative_zero_median",
                "defined_without_median",
                "always_defined_marked_degenerate",
                "unexpected_missing_nonzero",
                "nonfinite_count_nonzero",
                "bool_as_count",
                "negative_count",
                "count_conservation_failure",
            ),
            "conditional": (
                "negative_zero_median",
                "defined_without_median",
                "median_while_all_degenerate",
                "unexpected_missing_nonzero",
                "nonfinite_count_nonzero",
                "bool_as_count",
                "negative_count",
                "count_conservation_failure",
            ),
        }
        family_fields = {
            "always": gate.S2_ALWAYS_DEFINED_COMPONENT_FIELDS[0],
            "conditional": (
                gate.S2_CONDITIONALLY_DEFINED_COMPONENT_FIELDS[0]
            ),
        }
        for arm in gate.S2_COMPONENT_ARMS:
            for family, attacks in attacks_by_family.items():
                field = family_fields[family]
                for attack in attacks:
                    def mutate(
                        endpoint: dict[str, object],
                        *,
                        selected_arm: str = arm,
                        selected_field: str = field,
                        selected_attack: str = attack,
                    ) -> None:
                        medians = endpoint["component_medians"][selected_arm]
                        counts = endpoint["component_coverage"][selected_arm][
                            selected_field
                        ]
                        if selected_attack == "negative_zero_median":
                            medians[selected_field] = -0.0
                        elif selected_attack == "defined_without_median":
                            medians[selected_field] = None
                        elif selected_attack in {
                            "median_while_all_degenerate",
                            "always_defined_marked_degenerate",
                        }:
                            counts["defined_count"] = 0
                            counts["degenerate_count"] = 1
                            medians[selected_field] = 0.0
                        elif selected_attack == "unexpected_missing_nonzero":
                            counts["defined_count"] = 0
                            counts["degenerate_count"] = 0
                            counts["unexpected_missing_count"] = 1
                            medians[selected_field] = None
                        elif selected_attack == "nonfinite_count_nonzero":
                            counts["defined_count"] = 0
                            counts["degenerate_count"] = 0
                            counts["nonfinite_count"] = 1
                            medians[selected_field] = None
                        elif selected_attack == "bool_as_count":
                            counts["defined_count"] = True
                        elif selected_attack == "negative_count":
                            counts["defined_count"] = 2
                            counts["degenerate_count"] = -1
                        elif selected_attack == "count_conservation_failure":
                            counts["defined_count"] = 0
                            counts["degenerate_count"] = 0
                            medians[selected_field] = None
                        else:
                            raise AssertionError("unknown test mutation")

                    fixture = mutate_s2_configuration(mutate)
                    for function, error_type, extra in paths:
                        stdout = io.StringIO()
                        stderr = io.StringIO()
                        with self.subTest(
                            arm=arm,
                            family=family,
                            attack=attack,
                            path=function.__module__,
                        ):
                            with contextlib.redirect_stdout(
                                stdout
                            ), contextlib.redirect_stderr(stderr):
                                with self.assertRaises(error_type) as caught:
                                    function(
                                        configuration_id=(
                                            "S2_null_inflation__d1"
                                        ),
                                        subplan=fixture[0],
                                        index_raw=fixture[1],
                                        shard_raw_by_relative_path=fixture[2],
                                        **extra,
                                    )
                            self.assertEqual(
                                caught.exception.code,
                                "INPUT_SCHEMA_INVALID",
                            )
                            self.assertEqual(
                                str(caught.exception),
                                "INPUT_SCHEMA_INVALID",
                            )
                            self.assertEqual(stdout.getvalue(), "")
                            self.assertEqual(stderr.getvalue(), "")


class PublicationAndOwnershipTests(unittest.TestCase):
    def test_publication_reconciliation_cases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            final = root / "artifact.json"
            pending = root / "artifact.json.pending-v0.9"
            expected = b"{}\n"
            self.assertEqual(
                gate.classify_publication_after_exception(
                    final, pending, expected
                ).state,
                "no_durable_transition",
            )
            final.write_bytes(expected)
            self.assertEqual(
                gate.classify_publication_after_exception(
                    final, pending, expected
                ).state,
                "published_exact",
            )
            pending.write_bytes(expected)
            self.assertEqual(
                gate.classify_publication_after_exception(
                    final, pending, expected
                ).state,
                "ambiguous_hold_new_review",
            )
            final.write_bytes(b"foreign")
            self.assertEqual(
                gate.classify_publication_after_exception(
                    final, pending, expected
                ).state,
                "ambiguous_hold_new_review",
            )

    def test_owner_liveness_active_dead_unknown_and_foreign(self) -> None:
        claim = {
            "owner_hostname": "host",
            "owner_pid": 7,
            "owner_process_creation_time_utc": "2026-08-01T00:00:00Z",
        }
        self.assertEqual(
            gate.classify_claim_owner(
                claim,
                hostname="host",
                creation_query=lambda _pid: "2026-08-01T00:00:00Z",
            ),
            "ACTIVE",
        )
        self.assertEqual(
            gate.classify_claim_owner(
                claim,
                hostname="host",
                creation_query=lambda _pid: "2026-08-01T00:00:01Z",
            ),
            "DEAD",
        )
        self.assertEqual(
            gate.classify_claim_owner(claim, hostname="foreign"),
            "UNKNOWN",
        )
        self.assertEqual(
            gate.classify_claim_owner(
                claim,
                hostname="host",
                creation_query=lambda _pid: (_ for _ in ()).throw(
                    PermissionError()
                ),
            ),
            "UNKNOWN",
        )

    def test_relative_path_attack_surface_is_closed(self) -> None:
        invalid = (
            "",
            ".",
            "..",
            "../x",
            "x/../y",
            "/absolute",
            r"x\mixed",
            "x:ads",
            "x//y",
            "GLOBALROOT/x",
            "UNC/x",
        )
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(
                    gate.Gate12C2OriginalBaselineError
                ):
                    gate.validate_relative_manifest_path(value)


if __name__ == "__main__":
    unittest.main()
