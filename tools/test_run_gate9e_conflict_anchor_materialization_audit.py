#!/usr/bin/env python3
"""Regression tests for Gate9E conflict-anchor materialization helpers."""

import unittest

import run_gate9e_conflict_anchor_materialization_audit as gate9e


def make_registry_row(
    answer_target_type: str,
    expected_target: str,
    *,
    dry_run_status: str = "candidate_emitted",
    declaration_stable: bool = True,
    closure_convention_change_required: bool = False,
    has_support_anchor_lane: bool = True,
) -> dict:
    return {
        "benchmark_sample_id": f"sample:{answer_target_type}",
        "answer_target_type": answer_target_type,
        "rendering_id": "render:0",
        "expected_conflict_anchor_target_text": expected_target,
        "dry_run_status": dry_run_status,
        "dry_run_target_path": f"runs/test/{answer_target_type}.txt",
        "declaration_stable": declaration_stable,
        "closure_convention_change_required": closure_convention_change_required,
        "has_support_anchor_lane": has_support_anchor_lane,
        "actual_missing_conflict_anchor_files": [
            "conflict_anchor.txt",
            "conflict_anchor_meta.json",
            "conflict_anchor_triplets.ndjson",
        ],
    }


class RunGate9EConflictAnchorMaterializationAuditTest(unittest.TestCase):
    def test_expected_conflict_anchor_target_for_distributed_incompatibility(self) -> None:
        target, source = gate9e.expected_conflict_anchor_target(
            "distributed_incompatibility",
            {"distributed_block_claim": "block claim"},
        )
        self.assertEqual(target, "block claim")
        self.assertEqual(source, "world_truth.distributed_block_claim")

    def test_summarize_rows_keeps_answer_target_split_count(self) -> None:
        summary_rows = gate9e.summarize_rows(
            [
                make_registry_row("consistent_answer", "same target"),
                make_registry_row("unsupported_bridge_answer", "same target"),
            ]
        )
        self.assertEqual(summary_rows[0]["distinct_expected_target_count"], 1)
        self.assertEqual(summary_rows[1]["distinct_expected_target_count"], 1)

    def test_build_status_payload_stays_clear_when_targets_match(self) -> None:
        payload = gate9e.build_status_payload(
            [
                make_registry_row("consistent_answer", "same target"),
                make_registry_row("unsupported_bridge_answer", "same target"),
            ]
        )
        self.assertEqual(payload["dry_run_candidate_status"], "candidate_emitted")
        self.assertEqual(payload["declaration_stability_status"], "clear")
        self.assertEqual(payload["closure_convention_change_required_status"], "clear")
        self.assertEqual(payload["answer_target_split_status"], "clear")
        self.assertEqual(payload["existing_anchor_lane_ready_status"], "clear")

    def test_build_status_payload_triggers_answer_target_split(self) -> None:
        payload = gate9e.build_status_payload(
            [
                make_registry_row("consistent_answer", "target a"),
                make_registry_row("unsupported_bridge_answer", "target b"),
            ]
        )
        self.assertEqual(payload["answer_target_split_status"], "triggered")


if __name__ == "__main__":
    unittest.main()
