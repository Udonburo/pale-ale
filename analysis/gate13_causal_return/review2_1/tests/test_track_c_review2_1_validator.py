from __future__ import annotations

import copy
import math
import unittest
from pathlib import Path

import numpy as np

from analysis.gate13_causal_return.review2_1 import track_c_review2_1_validator as review21


class TrackCReview21ValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.package_root = Path(__file__).resolve().parents[1]

    @staticmethod
    def analysis_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, review21.AnalysisGeometry]:
        depth = np.repeat(np.asarray(review21.DEPTH_LEVELS, dtype=np.int64), 5)
        position = np.tile(np.arange(5), 4)
        angle = 2.0 * math.pi * position / 5.0 + 0.17 * np.repeat(np.arange(4), 5)
        competence = 0.25 * np.repeat(np.arange(4), 5) + np.cos(angle)
        representation = 4.0 + 0.60 * np.cos(angle) + 0.80 * np.sin(angle)
        geometry = review21.build_analysis_geometry(depth, competence, representation)
        return depth, competence, representation, geometry

    @staticmethod
    def surface_pair(
        pair_id: str = "pair_001",
        block_id: str = "TCB_00",
        episode_id: str = "episode_00",
    ) -> dict[str, object]:
        common = {
            "canonical_template_id": review21.FROZEN_TEMPLATE,
            "canonical_template_sha256": "a" * 64,
            "message_count": 3,
            "operation_count": 2,
            "codebook_token_ids": [31, 32],
            "special_token_count": 4,
            "answer_prefix_utf8_hex": b"answer:".hex(),
            "score_slot": "FINAL_ASSISTANT_TOKEN",
            "non_operation_token_ids": [1, 2],
        }
        return {
            "pair_id": pair_id,
            "block_id": block_id,
            "episode_id": episode_id,
            "path_p": {
                **common,
                "rendered_utf8_hex": b"operation-A operation-B".hex(),
                "token_ids": [1, 10, 20, 2],
                "operation_token_ids": [[10], [20]],
            },
            "path_q": {
                **common,
                "rendered_utf8_hex": b"operation-B operation-A".hex(),
                "token_ids": [1, 20, 10, 2],
                "operation_token_ids": [[20], [10]],
            },
        }

    def campaign_manifest(self) -> dict[str, object]:
        blocks = []
        all_map: list[list[str]] = []
        all_behavior: list[list[str]] = []
        for block_index in range(review21.PLANNED_BLOCKS):
            rollout_depth = review21.DEPTH_LEVELS[block_index // 5]
            map_ids = [f"M_{block_index:02d}_{index:03d}" for index in range(240)]
            behavior_ids = [
                f"E_{block_index:02d}_{index:03d}"
                for index in range(
                    2 * review21.BEHAVIOR_EPISODES_PER_BLOCK * (rollout_depth + 1)
                )
            ]
            all_map.append(map_ids)
            all_behavior.append(behavior_ids)
            blocks.append(
                {
                    "block_id": f"TCB_{block_index:02d}",
                    "rollout_depth": rollout_depth,
                    "template": review21.FROZEN_TEMPLATE,
                    "codebook_id": f"codebook_{block_index:02d}",
                    "codebook_sha256": f"{block_index + 1:064x}",
                    "demonstration_ids": [
                        f"demo_{block_index:02d}_a",
                        f"demo_{block_index:02d}_b",
                    ],
                    "seeds": {
                        "construction": 10_000 + block_index,
                        "map_half_1": 20_000 + block_index,
                        "map_half_2": 30_000 + block_index,
                        "behavior": 40_000 + block_index,
                    },
                    "map_half_ids": [
                        f"TCB_{block_index:02d}_H1",
                        f"TCB_{block_index:02d}_H2",
                    ],
                    "map_case_ids": map_ids,
                    "behavior_case_ids": behavior_ids,
                }
            )
        stage_m_order = [all_map[block][case] for case in range(240) for block in range(20)]
        stage_e_order = []
        for case in range(max(len(values) for values in all_behavior)):
            for block in range(20):
                if case < len(all_behavior[block]):
                    stage_e_order.append(all_behavior[block][case])
        return {
            "authority": {"track_c_authorized": False},
            "model": {
                "repository": review21.MODEL_REPOSITORY,
                "revision": review21.MODEL_REVISION,
                "tokenizer_repository": review21.TOKENIZER_REPOSITORY,
                "tokenizer_revision": review21.TOKENIZER_REVISION,
            },
            "runtime": {
                "image_definition_sha256": review21.RUNTIME_IMAGE_DEFINITION_SHA256,
                "chat_template_sha256": review21.CHAT_TEMPLATE_SHA256,
                "tokenizer_json_sha256": review21.TOKENIZER_JSON_SHA256,
                "dependency_versions": {"python": "3.11.2", "torch": "2.7.1+cu126"},
            },
            "scoring": {
                "score_position": "FINAL_ASSISTANT_TOKEN",
                "correct_is_single_token": True,
                "other_is_single_token": True,
                "correct_token_id": 100,
                "other_token_id": 101,
            },
            "blocks": blocks,
            "execution": {
                "stage_m_order": stage_m_order,
                "stage_e_order": stage_e_order,
                "order_seed": review21.EXECUTION_ORDER_SEED,
                "order_algorithm": "SEEDED_BLOCK_INTERLEAVED_SHUFFLE_V1",
                "accepted_ids_may_be_duplicated_or_replaced": False,
                "exact_resume_missing_ids_only": True,
            },
            "analysis": {
                "permutation_root_seed": review21.SCIENTIFIC_PERMUTATION_SEED,
                "permutations": review21.SCIENTIFIC_PERMUTATIONS,
                "schedule_family_algorithm": "SHA256_ROOT_SEED_AND_ORDERED_QUALIFIED_BLOCK_IDS_V1",
            },
            "path_surface_pairs": [
                self.surface_pair(
                    f"pair_{block_index:02d}_{episode_index:02d}",
                    f"TCB_{block_index:02d}",
                    f"episode_{episode_index:02d}",
                )
                for block_index in range(review21.PLANNED_BLOCKS)
                for episode_index in range(review21.BEHAVIOR_EPISODES_PER_BLOCK)
            ],
        }

    def test_amplitude_observable_retains_energy_components(self) -> None:
        components = {21: [1.0, 4.0], 43: [9.0, 16.0], 62: [25.0, 36.0]}
        result = review21.amplitude_representation_observable(components)
        self.assertAlmostEqual(result["mean_unsquared_energy"], 91.0 / 6.0)
        self.assertAlmostEqual(result["primary_amplitude"], math.sqrt(91.0 / 6.0))
        self.assertEqual(set(result["unsquared_energy_components"]), {"21", "43", "62"})

    def test_crossfit_energy_is_gauge_invariant_in_one_compatible_frame(self) -> None:
        rng = np.random.default_rng(44)
        activations = rng.normal(size=(24, 9))
        frame, _ = np.linalg.qr(rng.normal(size=(9, review21.FRAME_RANK)))
        source_gauge, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        target_gauge, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        delta = rng.normal(size=(4, 4))
        reference = review21.crossfit_return_energy(delta, activations, frame)
        transformed = review21.crossfit_return_energy(
            target_gauge.T @ delta @ source_gauge,
            activations,
            frame @ source_gauge,
        )
        self.assertAlmostEqual(reference, transformed, places=11)

    def test_map_competence_uses_only_exact_map_rows(self) -> None:
        rows = []
        for half in review21.MAP_HALVES:
            for node in review21.EXACT_MAP_NODES:
                for sample in range(review21.MAP_SAMPLES_PER_NODE_PER_HALF):
                    rows.append(
                        {
                            "half_id": half,
                            "sample_id": f"{sample:02d}",
                            "node_id": node,
                            "target_state": 0,
                            "candidate_logits": [2.25, 0.25],
                        }
                    )
        result = review21.map_derived_competence(rows)
        self.assertEqual(result["row_count"], 192)
        self.assertEqual(result["behavior_rows_used"], 0)
        self.assertEqual(result["broken_square_rows_used"], 0)
        self.assertAlmostEqual(result["map_derived_competence"], 2.0)
        broken = copy.deepcopy(rows)
        broken[0]["node_id"] = review21.BROKEN_MAP_NODE
        with self.assertRaises(review21.Review21ValidationError):
            review21.map_derived_competence(broken)

    def test_behavioral_outcome_is_one_block_level_rms(self) -> None:
        path_p = np.linspace(-0.5, 1.0, review21.BEHAVIOR_EPISODES_PER_BLOCK)
        path_q = path_p - np.linspace(0.0, 0.46, review21.BEHAVIOR_EPISODES_PER_BLOCK)
        expected = math.sqrt(float(np.mean(np.square(path_p - path_q))))
        self.assertAlmostEqual(review21.block_behavioral_outcome(path_p, path_q), expected)

    def test_path_surface_contract_records_and_rejects_mismatch(self) -> None:
        passed = review21.validate_path_surface_pair(self.surface_pair())
        self.assertEqual(passed["status"], "PASS")
        self.assertEqual(passed["path_p"]["token_ids"], [1, 10, 20, 2])
        mismatch = self.surface_pair()
        mismatch["path_q"]["token_ids"].append(99)  # type: ignore[index,union-attr]
        failed = review21.validate_path_surface_pair(mismatch)
        self.assertEqual(failed["status"], "FAIL")
        self.assertFalse(failed["checks"]["total_rendered_input_token_count"])

    def test_campaign_freeze_covers_all_cases_and_orders(self) -> None:
        manifest = self.campaign_manifest()
        result = review21.validate_frozen_campaign_manifest(manifest)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["map_case_count"], 4_800)
        self.assertEqual(result["behavior_case_count"], 5_760)
        self.assertFalse(result["track_c_authorized"])
        drifted = copy.deepcopy(manifest)
        drifted["execution"]["stage_m_order"][1] = drifted["execution"]["stage_m_order"][0]  # type: ignore[index]
        with self.assertRaises(review21.Review21ValidationError):
            review21.validate_frozen_campaign_manifest(drifted)

    def test_lobo_linear_operators_match_explicit_fold_refits(self) -> None:
        depth, competence, representation, geometry = self.analysis_fixture()
        outcome = 0.35 * competence + 1.20 * representation + 0.03 * np.sin(np.arange(20))
        fast = review21._outcome_statistics(outcome, geometry)
        slow = review21.brute_force_lobo_statistics(
            outcome=outcome,
            rollout_depth=depth,
            map_competence=competence,
            representation_feature=representation,
        )
        for key in ("sse_nuisance_lobo", "sse_full_lobo", "t_lobo", "beta_r"):
            self.assertAlmostEqual(fast[key], slow[key], places=10)

    def test_directional_primary_rule_cannot_rescue_negative_beta(self) -> None:
        depth, competence, representation, geometry = self.analysis_fixture()
        schedule = review21.generate_stratified_permutation_schedule(
            depth,
            permutations=999,
            seed=55_001,
        )
        positive_outcome = 0.4 * competence + 1.5 * representation + 0.01 * np.sin(np.arange(20))
        positive = review21.run_primary_pipeline(
            outcome=positive_outcome,
            geometry=geometry,
            schedule=schedule,
        )
        self.assertEqual(positive["terminal_state"], "PRIMARY_POSITIVE")
        self.assertGreater(positive["beta_r"], 0.0)
        negative_outcome = 0.4 * competence - 1.5 * representation + 0.01 * np.sin(np.arange(20))
        negative = review21.run_primary_pipeline(
            outcome=negative_outcome,
            geometry=geometry,
            schedule=schedule,
        )
        self.assertEqual(negative["terminal_state"], "WRONG_DIRECTION")
        self.assertLess(negative["beta_r"], 0.0)

    def test_variance_and_qualification_gates_fail_closed(self) -> None:
        depth, competence, representation, geometry = self.analysis_fixture()
        with self.assertRaises(review21.AnalysisTerminal) as outcome_context:
            review21.run_primary_pipeline(
                outcome=np.ones(20),
                geometry=geometry,
                schedule=review21.generate_stratified_permutation_schedule(
                    depth,
                    permutations=19,
                    seed=4,
                ),
            )
        self.assertEqual(outcome_context.exception.state, "NO_OUTCOME_VARIANCE")
        with self.assertRaises(review21.AnalysisTerminal) as representation_context:
            review21.build_analysis_geometry(depth, competence, np.ones(20))
        self.assertEqual(
            representation_context.exception.state,
            "NO_REPRESENTATION_FEATURE_VARIANCE",
        )
        bad_mask = np.ones(20, dtype=bool)
        bad_mask[:2] = False
        with self.assertRaises(review21.AnalysisTerminal) as qualification_context:
            review21.qualified_block_indices(depth, bad_mask)
        self.assertEqual(qualification_context.exception.state, "INSUFFICIENT_DEPTH_STRATUM")

    def test_stage_m_sealed_summary_exposes_only_allowed_qualification_counts(self) -> None:
        depth, competence, representation, _ = self.analysis_fixture()
        result = review21.evaluate_map_stage_predictor_gates(
            block_ids=[f"TCB_{index:02d}" for index in range(20)],
            rollout_depth=depth,
            map_competence=competence,
            representation_feature=representation,
            qualification_mask=np.ones(20, dtype=bool),
            split_half_valid=True,
            frame_rank_valid=True,
            conditioning_valid=True,
            exact_square_reproducibility_valid=True,
            broken_square_sensitivity_valid=True,
            path_surface_valid=True,
            artifact_complete=True,
        )
        self.assertEqual(
            set(result["sealed_public_summary"]),
            {"qualification_state", "qualified_blocks", "depth_counts"},
        )
        self.assertNotIn("representation", result["sealed_public_summary"])

    def test_exact_small_schedule_and_fixed_reproducibility(self) -> None:
        small_depth = np.repeat(np.asarray(review21.DEPTH_LEVELS), 2)
        schedule = review21.enumerate_stratified_permutations(small_depth)
        self.assertEqual(schedule.shape, (15, 8))
        validated = review21.validate_permutation_schedule(
            small_depth,
            schedule,
            expected_count=15,
        )
        self.assertTrue(validated["depth_strata_preserved"])
        reproducibility = review21._run_deterministic_reproducibility_check(
            permutations=99,
        )
        self.assertEqual(reproducibility["status"], "PASS")

    def test_leverage_rule_and_cost_forecast_are_frozen(self) -> None:
        self.assertAlmostEqual(review21.leverage_threshold(5, 20), 0.75)
        self.assertAlmostEqual(review21.leverage_threshold(6, 20), 0.80)
        minimum = review21.forward_and_cost_forecast(qualified_blocks=16)
        maximum = review21.forward_and_cost_forecast(qualified_blocks=20)
        self.assertEqual(minimum["stage_m"]["forwards"], 4_800)
        self.assertEqual(minimum["stage_e"]["forwards"], 4_608)
        self.assertEqual(maximum["stage_e"]["forwards"], 5_760)
        self.assertAlmostEqual(maximum["total"]["expected_usd"], 45.22264586)
        self.assertAlmostEqual(maximum["total"]["contingency_usd"], 56.528307325)

    def test_complete_package_passes_validator(self) -> None:
        repository_root = Path(__file__).resolve().parents[4]
        result = review21.validate_package(repository_root)
        self.assertEqual(result["status"], "PASS")
        self.assertIn(
            result["terminal_review2_1_state"],
            {
                "REVIEW2_1_READY_FOR_HUMAN_AUTHORIZATION",
                "REVIEW2_1_BLOCKED",
            },
        )


if __name__ == "__main__":
    unittest.main()
