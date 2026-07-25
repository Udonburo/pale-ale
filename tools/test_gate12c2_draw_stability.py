#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import math
import sys
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import gate12c2_draw_stability as stability  # noqa: E402


class Gate12C2DrawStabilityTest(unittest.TestCase):
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
                endpoint_rows.append(
                    {
                        "endpoint_id": endpoint_id,
                        "endpoint_identified": False,
                        "log_stressor_to_N1_null_defect": 0.2 + shift,
                        "component_medians": {
                            "N1": {
                                field: (
                                    0.25
                                    if field
                                    == stability.S2_ALIGNMENT_FIELD
                                    else 1.0
                                )
                                for field in (
                                    *stability.S2_MAGNITUDE_FIELDS,
                                    stability.S2_ALIGNMENT_FIELD,
                                )
                            },
                            "graph_unconstrained_stressor": {
                                field: (
                                    0.1 + shift
                                    if field
                                    == stability.S2_ALIGNMENT_FIELD
                                    else math.exp(0.1 + shift)
                                )
                                for field in (
                                    *stability.S2_MAGNITUDE_FIELDS,
                                    stability.S2_ALIGNMENT_FIELD,
                                )
                            },
                        },
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
    ) -> tuple[
        dict[str, dict[int, list[dict[str, object]]]],
        dict[str, dict[int, str]],
    ]:
        shifts = {255: 0.01, 511: 0.005, 1023: 0.0}
        results = {
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
        hashes = {
            regime_id: {
                count: f"{index + count:064x}"[-64:]
                for count in stability.PREFIX_COUNTS
            }
            for index, regime_id in enumerate(stability.REGIMES)
        }
        return results, hashes

    def test_projection_exposes_only_allowlisted_stability_deltas(
        self,
    ) -> None:
        results, hashes = self._inputs()
        projection = stability.build_no_outcome_projection(
            results,
            source_plan_payload_sha256_by_regime_and_draw_count=hashes,
            resource_gate={
                "status": "pass",
                "eligible_draw_counts": [255, 511, 1023],
                "receipt_payload_sha256": "a" * 64,
            },
        )
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
        results, hashes = self._inputs()
        projection = stability.build_no_outcome_projection(
            results,
            source_plan_payload_sha256_by_regime_and_draw_count=hashes,
            resource_gate={
                "status": "not_evaluated",
                "eligible_draw_counts": [],
                "receipt_payload_sha256": None,
            },
        )
        tampered = copy.deepcopy(projection)
        tampered["candidates"][0]["regimes"][0][
            "raw_outcome"
        ] = False
        with self.assertRaises(
            stability.Gate12C2DrawStabilityError
        ):
            stability.validate_no_outcome_projection(tampered)

    def test_one_sided_component_missingness_fails_closed(self) -> None:
        results, hashes = self._inputs()
        results["S2_null_inflation"][255][0]["endpoint_rows"][0][
            "component_medians"
        ]["N1"]["a_q"] = None
        with self.assertRaises(
            stability.Gate12C2DrawStabilityError
        ):
            stability.build_no_outcome_projection(
                results,
                source_plan_payload_sha256_by_regime_and_draw_count=hashes,
                resource_gate={
                    "status": "not_evaluated",
                    "eligible_draw_counts": [],
                    "receipt_payload_sha256": None,
                },
            )

    def test_prefix_mismatch_disqualifies_the_candidate(self) -> None:
        results, hashes = self._inputs()
        results["S0_true_null"][255][0]["endpoint_receipts"][0][
            "block_rows"
        ][0]["inner_draw_audit"][
            "accepted_sequence_sha256"
        ] = "different"
        projection = stability.build_no_outcome_projection(
            results,
            source_plan_payload_sha256_by_regime_and_draw_count=hashes,
            resource_gate={
                "status": "pass",
                "eligible_draw_counts": [255, 511, 1023],
                "receipt_payload_sha256": "b" * 64,
            },
        )
        self.assertFalse(
            projection["candidates"][0]["accepted_prefix_gate_pass"]
        )
        self.assertEqual(projection["selected_draw_count"], 511)

    def test_analysis_manifest_rejects_rehashed_permissions(self) -> None:
        roots = {
            regime_id: {
                count: Path(f"root/{regime_id}/{count}")
                for count in stability.PREFIX_COUNTS
            }
            for regime_id in stability.REGIMES
        }
        manifest = stability.build_analysis_manifest(
            roots,
            resource_gate={
                "status": "not_evaluated",
                "eligible_draw_counts": [],
                "receipt_payload_sha256": None,
            },
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
                with self.assertRaises(
                    stability.Gate12C2DrawStabilityError
                ):
                    stability.verify_analysis_manifest(tampered)


if __name__ == "__main__":
    unittest.main()
