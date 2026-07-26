#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import gate12c2_draw_profile as profile  # noqa: E402
import gate12c2_draw_stability as stability  # noqa: E402
import run_gate12c2_draw_stability as stability_cli  # noqa: E402


class Gate12C2DrawStabilityTest(unittest.TestCase):
    @staticmethod
    @contextmanager
    def _control_context(resource: dict[str, object]):
        with mock.patch.object(
            profile,
            "_verify_control_lineage",
            return_value=resource["control_lineage"],
        ):
            yield

    @staticmethod
    def _audit(draw_count: int) -> dict[str, object]:
        commitments = {
            str(count): {
                "accepted_sequence_sha256": f"prefix-{count}",
            }
            for count in stability.PREFIX_COUNTS
            if count <= draw_count
        }
        return {
            "accepted_sequence_sha256": f"prefix-{draw_count}",
            "accepted_prefix_commitments": commitments,
        }

    @staticmethod
    def _coverage(
        *,
        conditional_degenerate: bool = False,
    ) -> dict[str, dict[str, dict[str, int]]]:
        result = {}
        for arm in ("observed", "N1", "graph_unconstrained_stressor"):
            result[arm] = {}
            for field in (
                *stability.S2_ALWAYS_DEFINED_FIELDS,
                *stability.S2_CONDITIONALLY_DEFINED_FIELDS,
            ):
                degenerate = (
                    conditional_degenerate
                    and field in stability.S2_CONDITIONALLY_DEFINED_FIELDS
                )
                result[arm][field] = {
                    "expected_count": 10,
                    "defined_count": 0 if degenerate else 10,
                    "degenerate_count": 10 if degenerate else 0,
                    "unexpected_missing_count": 0,
                    "nonfinite_count": 0,
                }
        return result

    @classmethod
    def _result(
        cls,
        regime_id: str,
        draw_count: int,
        *,
        shift: float,
    ) -> dict[str, object]:
        endpoint_rows = []
        endpoint_receipts = []
        for index in range(24):
            endpoint_id = f"case-{index // 2:02d}:q{index % 2 + 1}"
            block = {
                "source_block_id": f"block-{index}",
                "inner_draw_audit": cls._audit(draw_count),
            }
            if regime_id == "S2_null_inflation":
                fields = (
                    *stability.S2_ALWAYS_DEFINED_FIELDS,
                    *stability.S2_CONDITIONALLY_DEFINED_FIELDS,
                )
                endpoint_rows.append(
                    {
                        "endpoint_id": endpoint_id,
                        "endpoint_identified": False,
                        "log_stressor_to_N1_null_defect": 0.2 + shift,
                        "component_medians": {
                            "observed": {
                                field: (
                                    0.2
                                    if field in ("c_q", "p_L_q", "p_R_q")
                                    else 1.0
                                )
                                for field in fields
                            },
                            "N1": {
                                field: (
                                    0.25
                                    if field in ("c_q", "p_L_q", "p_R_q")
                                    else 1.0
                                )
                                for field in fields
                            },
                            "graph_unconstrained_stressor": {
                                field: (
                                    0.1 + shift
                                    if field == "c_q"
                                    else math.exp(0.1 + shift)
                                )
                                for field in fields
                            },
                        },
                        "component_coverage": cls._coverage(),
                        "block_rows": [block],
                    }
                )
            else:
                endpoint_rows.append(
                    {
                        "endpoint_id": endpoint_id,
                        "q_directional_support": False,
                        "median_log_ratio": 0.3 + shift,
                    }
                )
                endpoint_receipts.append(
                    {
                        "endpoint_id": endpoint_id,
                        "block_rows": [block],
                    }
                )
        result: dict[str, object] = {
            "regime_id": regime_id,
            "outer_experiment_index": 0,
            "inner_valid_draw_count": draw_count,
        }
        if regime_id == "S2_null_inflation":
            result["endpoint_rows"] = endpoint_rows
        else:
            result["endpoint_receipts"] = endpoint_receipts
            result["pipeline_decision"] = {
                "endpoint_rows": endpoint_rows,
                "any_endpoint_support": False,
            }
        return result

    @classmethod
    def _inputs(
        cls,
    ) -> dict[str, dict[int, list[dict[str, object]]]]:
        shifts = {255: 0.01, 511: 0.005, 1023: 0.0}
        return {
            regime_id: {
                count: [
                    cls._result(
                        regime_id,
                        count,
                        shift=shifts[count],
                    )
                ]
                for count in stability.PREFIX_COUNTS
            }
            for regime_id in stability.REGIMES
        }

    @staticmethod
    def _resource_chain() -> tuple[
        dict[str, object],
        dict[str, object],
        dict[str, object],
    ]:
        plan = profile.build_draw_profile_plan(
            source_commit="test-source-commit"
        )
        rows = []
        for configuration in plan["configurations"]:
            subplan = configuration["subplan"]
            rows.append(
                {
                    "configuration_id": configuration["configuration_id"],
                    "regime_id": configuration["regime_id"],
                    "draw_count": configuration["draw_count"],
                    "worker_count": profile.WORKER_COUNT,
                    "outer_experiment_count": len(
                        subplan["outer_experiment_indices"]
                    ),
                    "outer_id_surface_sha256": (
                        profile._outer_id_surface_sha256(subplan)
                    ),
                    "all_outer_indices_present": True,
                    "plan_payload_sha256": subplan[
                        "plan_payload_sha256"
                    ],
                    "scientific_projection_sha256": "a" * 64,
                    "index_payload_sha256": "b" * 64,
                    "new_shard_count": 1,
                    "reused_shard_count": 0,
                    "endpoint_draw_attempts": 100,
                    "endpoint_draw_acceptances": 100,
                    "rejection_reason_counts": {},
                    "unaccounted_rejection_count": 0,
                    "exhausted_incomplete_stream_count": 0,
                    "derived_preflight_receipt_payload_sha256": "c" * 64,
                    "derived_authorization_receipt_payload_sha256": (
                        "d" * 64
                    ),
                    "scientific_outcomes_exposed": False,
                }
            )
        evidence = profile._build_execution_evidence(
            plan,
            configuration_rows=rows,
            wall_seconds=10.0,
            process_cpu_seconds=5.0,
            process_tree_memory={
                "sample_interval_seconds": 0.1,
                "sample_count": 3,
                "peak_process_tree_rss_bytes": 128 * 1024 * 1024,
                "peak_observed_process_count": 5,
                "monitor_error": None,
            },
            physical_ram_bytes=16 * 1024**3,
            disk_free_bytes_before=20 * 1024**3,
            disk_free_bytes_after=19 * 1024**3,
            output_bytes=1024**3,
        )
        preflight = {
            "preflight_receipt_payload_sha256": "e" * 64,
        }
        authorization = {
            "authorization_receipt_payload_sha256": "f" * 64,
            "output_root": Path("test-output").resolve().as_posix(),
        }
        consumption = {
            "consumption_receipt_payload_sha256": "1" * 64,
        }
        control = {
            "preflight_receipt": preflight,
            "authorization_receipt": authorization,
            "consumption_receipt": consumption,
            "preflight_receipt_payload_sha256": "e" * 64,
            "authorization_receipt_payload_sha256": "f" * 64,
            "consumption_receipt_payload_sha256": "1" * 64,
        }
        with mock.patch.object(
            profile,
            "_verify_control_lineage",
            return_value=control,
        ):
            resource = profile._build_resource_receipt(
                plan,
                evidence,
                preflight_receipt=preflight,
                authorization_receipt=authorization,
                consumption_receipt=consumption,
            )
        execution: dict[str, object] = {
            "schema_version": profile.RECEIPT_SCHEMA_VERSION,
            "plan_id": profile.PLAN_ID,
            "epistemic_status": "development_draw_profile_execution_only",
            "surface_id": "development",
            "locked_execution_authorized": False,
            "real_held_out_execution_authorized": False,
            "N2_open": False,
            "N3_open": False,
            "public_claim": False,
            "scientific_calibration_result": None,
            "scientific_outcomes_exposed": False,
            "draw_profile_plan_payload_sha256": plan[
                "draw_profile_plan_payload_sha256"
            ],
            "preflight_receipt_payload_sha256": "e" * 64,
            "authorization_receipt_payload_sha256": "f" * 64,
            "authorization_consumption_receipt_payload_sha256": "1" * 64,
            "execution_evidence_payload_sha256": evidence[
                "execution_evidence_payload_sha256"
            ],
            "resource_receipt_payload_sha256": resource[
                "resource_receipt_payload_sha256"
            ],
            "configuration_count": 9,
            "configuration_results": evidence["configuration_results"],
            "next_step": "strict no-outcome draw stability",
        }
        execution["execution_receipt_payload_sha256"] = (
            profile._sha256_bytes(
                profile._canonical_json_bytes(execution)
            )
        )
        return plan, execution, resource

    def _projection(
        self,
        results: dict[str, dict[int, list[dict[str, object]]]],
    ) -> dict[str, object]:
        plan, execution, resource = self._resource_chain()
        with self._control_context(resource):
            return stability.build_no_outcome_projection(
                results,
                draw_profile_plan=plan,
                execution_receipt=execution,
                resource_receipt=resource,
            )

    def test_projection_exposes_only_allowlisted_stability_deltas(
        self,
    ) -> None:
        projection = self._projection(self._inputs())
        self.assertEqual(projection["selected_draw_count"], 255)
        self.assertFalse(projection["scientific_outcomes_exposed"])
        self.assertIsNone(projection["scientific_calibration_result"])
        encoded = json.dumps(projection, allow_nan=False, sort_keys=True)
        for forbidden_key in (
            '"median_log_ratio"',
            '"q_directional_support"',
            '"any_endpoint_support"',
            '"endpoint_identified"',
            '"log_stressor_to_N1_null_defect"',
            '"component_medians"',
        ):
            self.assertNotIn(forbidden_key, encoded)
        self.assertEqual(
            stability.validate_no_outcome_projection(projection),
            projection,
        )

    def test_unknown_nested_projection_key_fails_closed(self) -> None:
        projection = self._projection(self._inputs())
        tampered = copy.deepcopy(projection)
        tampered["candidates"][0]["regimes"][0][
            "raw_outcome"
        ] = False
        with self.assertRaises(
            stability.Gate12C2DrawStabilityError
        ):
            stability.validate_no_outcome_projection(tampered)

    def test_raw_scientific_values_require_exact_json_numbers(self) -> None:
        sentinel = "RAW_SCIENTIFIC_DIRECTION_SENTINEL"
        for value in (True, "0.31", sentinel):
            results = self._inputs()
            results["S0_true_null"][255][0]["pipeline_decision"][
                "endpoint_rows"
            ][0]["median_log_ratio"] = value
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises(
                    stability.Gate12C2DrawStabilityError
                ) as raised:
                    self._projection(results)
                self.assertNotIn(sentinel, str(raised.exception))

    def test_coverage_counts_require_exact_json_integers(self) -> None:
        sentinel = "RAW_COVERAGE_SENTINEL"
        for value in (True, 10.0, "10", sentinel):
            results = self._inputs()
            results["S2_null_inflation"][255][0][
                "endpoint_rows"
            ][0]["component_coverage"]["N1"]["a_q"][
                "expected_count"
            ] = value
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises(
                    stability.Gate12C2DrawStabilityError
                ) as raised:
                    self._projection(results)
                self.assertNotIn(sentinel, str(raised.exception))

    def test_public_cli_emits_only_fixed_error_code(self) -> None:
        sentinel = "RAW_SCIENTIFIC_DIRECTION_SENTINEL"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            output = root / "projection.json"
            manifest.write_text("{}", encoding="utf-8")
            driver = (
                "import sys;"
                f"sys.path.insert(0,{str(TOOLS_DIR)!r});"
                "import run_gate12c2_draw_stability as cli;"
                "cli.stability.analyze_verified_directories="
                "(lambda manifest: cli.stability._finite("
                f"{sentinel!r},context='endpoint primary summary'));"
                f"sys.argv=['run','--manifest',{str(manifest)!r},"
                f"'--output',{str(output)!r}];"
                "raise SystemExit(cli.cli())"
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    driver,
                ],
                cwd=str(TOOLS_DIR.parent),
                capture_output=True,
                text=True,
                timeout=30,
            )
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(completed.stdout, "")
        self.assertEqual(
            completed.stderr.strip(),
            stability_cli.PUBLIC_ERROR_CODE,
        )
        self.assertNotIn(sentinel, completed.stderr)
        self.assertFalse(output.exists())

    def test_all_missing_always_defined_component_fails_closed(self) -> None:
        results = self._inputs()
        for draw_count in stability.PREFIX_COUNTS:
            for endpoint in results["S2_null_inflation"][draw_count][0][
                "endpoint_rows"
            ]:
                for arm in ("N1", "graph_unconstrained_stressor"):
                    endpoint["component_medians"][arm]["a_q"] = None
                    endpoint["component_coverage"][arm]["a_q"] = {
                        "expected_count": 10,
                        "defined_count": 0,
                        "degenerate_count": 0,
                        "unexpected_missing_count": 10,
                        "nonfinite_count": 0,
                    }
        with self.assertRaisesRegex(
            stability.Gate12C2DrawStabilityError,
            "unexpected missing",
        ):
            self._projection(results)

    def test_partial_and_one_sided_missingness_fail_closed(self) -> None:
        for mutation in ("partial", "one_sided"):
            results = self._inputs()
            endpoint = results["S2_null_inflation"][255][0][
                "endpoint_rows"
            ][0]
            endpoint["component_medians"]["N1"]["a_q"] = None
            endpoint["component_coverage"]["N1"]["a_q"] = {
                "expected_count": 10,
                "defined_count": 9,
                "degenerate_count": 0,
                "unexpected_missing_count": 1,
                "nonfinite_count": 0,
            }
            if mutation == "one_sided":
                endpoint["component_medians"][
                    "graph_unconstrained_stressor"
                ]["a_q"] = 1.0
            with self.subTest(mutation=mutation):
                with self.assertRaises(
                    stability.Gate12C2DrawStabilityError
                ):
                    self._projection(results)

    def test_all_degenerate_conditional_field_is_ineligible(self) -> None:
        results = self._inputs()
        for draw_count in stability.PREFIX_COUNTS:
            for endpoint in results["S2_null_inflation"][draw_count][0][
                "endpoint_rows"
            ]:
                for arm in ("observed", "N1", "graph_unconstrained_stressor"):
                    endpoint["component_medians"][arm]["c_q"] = None
                    endpoint["component_coverage"][arm]["c_q"] = {
                        "expected_count": 10,
                        "defined_count": 0,
                        "degenerate_count": 10,
                        "unexpected_missing_count": 0,
                        "nonfinite_count": 0,
                    }
        projection = self._projection(results)
        self.assertIsNone(projection["selected_draw_count"])
        for candidate in projection["candidates"]:
            self.assertFalse(candidate["selection_eligible"])
            s2 = next(
                row
                for row in candidate["regimes"]
                if row["regime_id"] == "S2_null_inflation"
            )
            alignment = next(
                row
                for row in s2["component_stability"]
                if row["field_name"] == "c_q"
            )
            self.assertEqual(alignment["compared_count"], 0)
            self.assertEqual(alignment["degenerate_count"], 24)
            self.assertFalse(alignment["coverage_gate_pass"])

    def test_partially_degenerate_conditional_surface_is_ineligible(
        self,
    ) -> None:
        results = self._inputs()
        for draw_count in stability.PREFIX_COUNTS:
            endpoint = results["S2_null_inflation"][draw_count][0][
                "endpoint_rows"
            ][0]
            for arm in ("observed", "N1", "graph_unconstrained_stressor"):
                endpoint["component_medians"][arm]["c_q"] = None
                endpoint["component_coverage"][arm]["c_q"] = {
                    "expected_count": 10,
                    "defined_count": 0,
                    "degenerate_count": 10,
                    "unexpected_missing_count": 0,
                    "nonfinite_count": 0,
                }
        projection = self._projection(results)
        self.assertIsNone(projection["selected_draw_count"])
        for candidate in projection["candidates"]:
            self.assertFalse(candidate["selection_eligible"])
            s2 = next(
                row
                for row in candidate["regimes"]
                if row["regime_id"] == "S2_null_inflation"
            )
            alignment = next(
                row
                for row in s2["component_stability"]
                if row["field_name"] == "c_q"
            )
            self.assertEqual(alignment["compared_count"], 23)
            self.assertEqual(alignment["degenerate_count"], 1)
            self.assertFalse(alignment["coverage_gate_pass"])

    def test_nan_and_inf_fail_closed(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            results = self._inputs()
            endpoint = results["S2_null_inflation"][255][0][
                "endpoint_rows"
            ][0]
            endpoint["component_medians"]["N1"]["a_q"] = value
            with self.subTest(value=value):
                with self.assertRaises(
                    stability.Gate12C2DrawStabilityError
                ):
                    self._projection(results)

    def test_prefix_mismatch_disqualifies_the_candidate(self) -> None:
        results = self._inputs()
        results["S0_true_null"][255][0]["endpoint_receipts"][0][
            "block_rows"
        ][0]["inner_draw_audit"][
            "accepted_sequence_sha256"
        ] = "different"
        projection = self._projection(results)
        self.assertFalse(
            projection["candidates"][0]["accepted_prefix_gate_pass"]
        )
        self.assertEqual(projection["selected_draw_count"], 511)

    def test_resource_receipt_tamper_cannot_select_draw_count(self) -> None:
        results = self._inputs()
        plan, execution, resource = self._resource_chain()
        tampered = copy.deepcopy(resource)
        tampered["eligible_draw_counts"] = [255]
        tampered.pop("resource_receipt_payload_sha256")
        tampered["resource_receipt_payload_sha256"] = (
            profile._sha256_bytes(
                profile._canonical_json_bytes(tampered)
            )
        )
        with self._control_context(resource):
            with self.assertRaisesRegex(
                stability.Gate12C2DrawStabilityError,
                "resource evidence chain failed",
            ):
                stability.build_no_outcome_projection(
                    results,
                    draw_profile_plan=plan,
                    execution_receipt=execution,
                    resource_receipt=tampered,
                )

    def test_analysis_manifest_rejects_rehashed_permissions(self) -> None:
        roots = {
            regime_id: {
                count: Path(f"root/{regime_id}/{count}")
                for count in stability.PREFIX_COUNTS
            }
            for regime_id in stability.REGIMES
        }
        plan, execution, resource = self._resource_chain()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {
                "plan": root / "plan.json",
                "preflight": root / "preflight.json",
                "authorization": root / "authorization.json",
                "consumption": root / "consumption.json",
                "evidence": root / "execution-evidence.json",
                "execution": root / "execution.json",
                "resource": root / "resource.json",
            }
            for key, payload in (
                ("plan", plan),
                (
                    "preflight",
                    resource["control_lineage"]["preflight_receipt"],
                ),
                (
                    "authorization",
                    resource["control_lineage"]["authorization_receipt"],
                ),
                (
                    "consumption",
                    resource["control_lineage"]["consumption_receipt"],
                ),
                ("evidence", resource["execution_evidence"]),
                ("execution", execution),
                ("resource", resource),
            ):
                paths[key].write_bytes(
                    profile._canonical_json_bytes(payload)
                )
            root_evidence = {
                regime_id: {
                    str(count): {
                        "configuration_id": f"{regime_id}__d{count}",
                        "plan_payload_sha256": "a" * 64,
                        "index_payload_sha256": "b" * 64,
                        "scientific_projection_sha256": "c" * 64,
                        "outer_experiment_count": 1,
                    }
                    for count in stability.PREFIX_COUNTS
                }
                for regime_id in stability.REGIMES
            }
            with self._control_context(resource), mock.patch.object(
                stability,
                "verify_result_root_evidence",
                return_value=root_evidence,
            ):
                manifest = stability.build_analysis_manifest(
                    roots,
                    draw_profile_plan_path=paths["plan"],
                    preflight_receipt_path=paths["preflight"],
                    authorization_receipt_path=paths["authorization"],
                    consumption_receipt_path=paths["consumption"],
                    execution_evidence_path=paths["evidence"],
                    execution_receipt_path=paths["execution"],
                    resource_receipt_path=paths["resource"],
                )
                self.assertEqual(
                    stability.verify_analysis_manifest(manifest),
                    manifest,
                )
            for key in (
                "development_execution_authorized",
                "locked_execution_authorized",
                "real_held_out_execution_authorized",
                "N2_open",
                "N3_open",
                "public_claim",
                "scientific_outcomes_may_be_emitted",
            ):
                tampered = dict(manifest)
                tampered[key] = True
                tampered.pop("analysis_manifest_payload_sha256")
                tampered["analysis_manifest_payload_sha256"] = (
                    stability.shards._sha256_bytes(
                        stability.shards._canonical_json_bytes(tampered)
                    )
                )
                with self.subTest(key=key):
                    with self._control_context(
                        resource
                    ), mock.patch.object(
                        stability,
                        "verify_result_root_evidence",
                        return_value=root_evidence,
                    ):
                        with self.assertRaises(
                            stability.Gate12C2DrawStabilityError
                        ):
                            stability.verify_analysis_manifest(tampered)


if __name__ == "__main__":
    unittest.main()
