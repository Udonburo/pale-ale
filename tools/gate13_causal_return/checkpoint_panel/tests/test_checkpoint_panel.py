from __future__ import annotations

import unittest
from pathlib import Path

from tools.gate13_causal_return.checkpoint_panel import panel


class FakeTokenizer:
    chat_template = "frozen-template"
    all_special_ids = [999]

    @staticmethod
    def apply_chat_template(messages, *, tokenize, return_dict=False, **kwargs):
        del messages, kwargs
        rendered = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ids = [10, 11, 12]
        if not tokenize:
            return rendered
        value = {"input_ids": ids, "attention_mask": [1, 1, 1]}
        return value if return_dict else ids

    @staticmethod
    def encode(text, add_special_tokens=False):
        del add_special_tokens
        if text == " Apex":
            return [101]
        if text == " Akron":
            return [102]
        if text.endswith(" Apex"):
            return [10, 11, 12, 101]
        if text.endswith(" Akron"):
            return [10, 11, 12, 102]
        raise AssertionError(text)


class CheckpointPanelTests(unittest.TestCase):
    def test_existing_instrument_is_hash_identical(self):
        contract = panel.qualification_contract()
        self.assertEqual(
            contract["qualification_ledger_sha256"],
            panel.BASELINE_QUALIFICATION_LEDGER_SHA256,
        )
        self.assertEqual(contract["forward_counts"]["maximum_conditional_total"], 232)

    def test_normalized_operator_layers_are_deterministic(self):
        self.assertEqual(panel.derive_operator_layers(36), [12, 24, 35])
        self.assertEqual(panel.derive_operator_layers(40), [13, 27, 39])
        self.assertEqual(panel.derive_operator_layers(64), [21, 43, 62])

    def test_execution_and_volume_identities_are_stable_and_distinct(self):
        identities = [panel.execution_identity(key) for key in panel.CHECKPOINTS]
        self.assertEqual(len(set(identities)), 4)
        self.assertEqual(
            len({panel.model_volume_name(key) for key in panel.CHECKPOINTS}),
            4,
        )
        self.assertEqual(
            len({panel.result_volume_name(key) for key in panel.CHECKPOINTS}),
            4,
        )

    def test_score_slot_is_direct_semantic_choice_after_closed_think_boundary(self):
        result = panel.score_slot_record(FakeTokenizer(), "qwen3_14b")
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["active_think_prefix_before_score_slot"])
        self.assertTrue(result["candidate_append_is_direct_assistant_answer"])
        self.assertEqual(result["score_tensor_index"], 2)
        self.assertEqual(result["semantic_answer_token_index"], 3)

    def test_panel_is_finite_and_h100_is_not_a_fallback(self):
        self.assertEqual(tuple(panel.CHECKPOINTS), panel.EXECUTION_ORDER)
        self.assertEqual(len(panel.CHECKPOINTS), 4)
        self.assertNotIn("H100", {spec["fallback_gpu"] for spec in panel.CHECKPOINTS.values()})
        self.assertEqual(panel.PANEL_SPEND_CEILING_USD, 22.0)

    def test_modal_adapter_invokes_frozen_runner_and_never_calls_generate(self):
        root = Path(__file__).resolve().parents[4]
        source = (
            root
            / "tools/gate13_causal_return/modal/modal_checkpoint_transfer_panel.py"
        ).read_text(encoding="utf-8")
        self.assertIn("run_track_a_qualification(journal, SELECTED_INSTRUMENT)", source)
        self.assertIn("compile_track_b_collection_ledger(SELECTED_INSTRUMENT)", source)
        self.assertNotIn("model.generate(", source)
        self.assertNotIn("temperature=", source)
        self.assertIn("retries=0", source)


if __name__ == "__main__":
    unittest.main()
