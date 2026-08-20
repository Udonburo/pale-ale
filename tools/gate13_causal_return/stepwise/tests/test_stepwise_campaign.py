from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.gate13_causal_return.stepwise import compiler
from tools.gate13_causal_return.stepwise.operator_qualification import (
    LAYER_SET,
    qualify_track_b,
)
from tools.gate13_causal_return.stepwise.runner import (
    JsonlJournal,
    run_development_variant,
    run_track_a_qualification,
)


class OracleProbe:
    """Deterministic test double; controls deliberately carry no signal."""

    def __call__(self, prompt, candidates, metadata):
        target = int(metadata["target_state"])
        if metadata["surface"] == "STREAM-A1" and metadata["condition"] != "correct":
            target = 1 - target
        return {
            "predicted_label": candidates[target],
            "candidate_token_ids": [100, 101],
            "candidate_logits": [1.0 if target == 0 else 0.0, 1.0 if target == 1 else 0.0],
            "input_ids_sha256": "0" * 64,
            "attention_mask_sha256": "1" * 64,
        }


class TokenizersAdapter:
    def __init__(self, path: Path):
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_file(str(path))

    def encode(self, text, add_special_tokens=False):
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids, skip_special_tokens=False):
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)


class StepwiseCompilerTests(unittest.TestCase):
    def test_codebook_banks_are_disjoint_and_ledgers_fit_budgets(self):
        partition = compiler.validate_codebook_partition()
        self.assertEqual(partition["status"], "PASS")
        self.assertEqual(partition["total_unique_labels"], 80)
        for variant in compiler.VARIANT_IDS:
            self.assertEqual(
                compiler.compile_development_ledger(variant)["forward_counts"]["total"],
                31,
            )
        qualification = compiler.compile_qualification_ledgers(compiler.VARIANT_IDS[0])
        self.assertEqual(qualification["forward_counts"]["maximum_conditional_total"], 232)
        track_b = compiler.compile_track_b_collection_ledger(compiler.VARIANT_IDS[0])
        self.assertEqual(track_b["forward_count"], 240)
        first = track_b["halves"][0]
        second = track_b["halves"][1]
        self.assertTrue(
            {row["episode_seed"] for row in first["samples"]}.isdisjoint(
                row["episode_seed"] for row in second["samples"]
            )
        )
        self.assertNotEqual(first["codebook_bank"], second["codebook_bank"])

    def test_prompt_contains_only_current_action_and_frozen_context(self):
        codebook = compiler.codebook_bank("development")[0]
        prompt = compiler.render_step_prompt(
            variant_id="compact_table_v1",
            surface="STREAM-A0",
            codebook=codebook,
            current_state=0,
            action=1,
        )
        self.assertIn("Current visible state:", prompt)
        self.assertIn("Next input action:", prompt)
        self.assertNotIn("history", prompt.lower())
        self.assertNotIn("previous input", prompt.lower())
        self.assertNotIn("0/1", prompt)

    def test_exact_external_algebra(self):
        self.assertEqual(compiler.transition(0, 0), 0)
        self.assertEqual(compiler.transition(0, 1), 1)
        self.assertEqual(compiler.transition(1, 0), 1)
        self.assertEqual(compiler.transition(1, 1), 0)
        for state in (0, 1):
            for action in (0, 1):
                self.assertEqual(
                    compiler.transition(1 - state, action),
                    1 - compiler.transition(state, action),
                )

    def test_exact_tokenizer_labels_when_snapshot_is_available(self):
        root = Path(__file__).resolve().parents[4]
        tokenizer_json = root / (
            "workstream/local/gate13_causal_return_outputs/phase2/"
            "modal_track_a_constrained/preflight/tokenizer_snapshot/tokenizer.json"
        )
        if not tokenizer_json.exists():
            self.skipTest("ignored exact tokenizer snapshot is not present")
        report = compiler.validate_exact_tokenizer(TokenizersAdapter(tokenizer_json))
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["label_count"], 80)
        self.assertEqual(len({row["token_id"] for row in report["labels"]}), 80)


class StepwiseRunnerTests(unittest.TestCase):
    def test_development_runner_and_resume_are_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = JsonlJournal(root, OracleProbe())
            result = run_development_variant(first, "compact_table_v1")
            self.assertTrue(result["selection_eligible"])
            self.assertEqual(first.new_forward_count, 31)
            second = JsonlJournal(root, OracleProbe())
            repeated = run_development_variant(second, "compact_table_v1")
            self.assertTrue(repeated["selection_eligible"])
            self.assertEqual(second.new_forward_count, 0)
            self.assertEqual(first.total_response_count, second.total_response_count)

    def test_qualification_conditional_ladder_and_exact_count(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = JsonlJournal(Path(directory), OracleProbe())
            result = run_track_a_qualification(journal, "compact_table_v1")
            self.assertEqual(result["STREAM-A0"]["status"], "PASS")
            self.assertEqual(result["STREAM-A1"]["status"], "PASS")
            self.assertEqual(result["STREAM-A2"]["status"], "PASS")
            self.assertEqual(journal.new_forward_count, 232)

    def test_ambiguous_attempt_is_not_repeated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "forward_attempts.jsonl").write_text(
                json.dumps(
                    {
                        "forward_id": "x",
                        "prompt_sha256": "p",
                        "candidate_labels": [" A", " B"],
                        "metadata_sha256": "m",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            journal = JsonlJournal(root, OracleProbe())
            codebook = compiler.Codebook("x", "x", (" A", " B"), (" C", " D"))
            with self.assertRaisesRegex(RuntimeError, "AMBIGUOUS_FORWARD_ATTEMPT"):
                journal.query(
                    forward_id="x",
                    prompt="p",
                    codebook=codebook,
                    metadata={"target_state": 0, "surface": "STREAM-A0"},
                )


class OperatorQualificationTests(unittest.TestCase):
    @staticmethod
    def _half(seed: int):
        rng = np.random.default_rng(seed)
        samples = 24
        hidden = 12
        latent = rng.normal(size=(samples, 4))
        base, _ = np.linalg.qr(rng.normal(size=(hidden, 4)))
        transforms = {
            "phase0_state0": np.eye(4),
            "phase0_state1": np.diag([1.2, 0.9, 1.1, 0.8]),
            "phase1_state0": np.array(
                [[1.0, 0.1, 0.0, 0.0], [0.0, 1.0, 0.1, 0.0], [0.0, 0.0, 1.0, 0.1], [0.1, 0.0, 0.0, 1.0]]
            ),
        }
        transforms["phase1_state1"] = transforms["phase1_state0"] @ transforms["phase0_state1"]
        rows = {
            name: latent @ transform.T @ base.T
            for name, transform in transforms.items()
        }
        permutation = np.roll(np.arange(samples), 7)
        rows["phase1_state1_broken"] = rows["phase1_state1"][permutation]
        return rows

    def test_split_half_operator_packet_and_broken_control(self):
        first = self._half(11)
        second = self._half(29)
        activations = {
            "half_1": {layer: first for layer in LAYER_SET},
            "half_2": {layer: second for layer in LAYER_SET},
        }
        result = qualify_track_b(activations)
        self.assertEqual(result["status"], "PASS")
        for layer in result["layers"]:
            self.assertIn("P_p", layer["half_1"]["exact_square"]["raw"])
            self.assertIn("S_p", layer["half_1"]["exact_square"]["path_polar"])
            self.assertTrue(layer["broken_square_sensitive"])


if __name__ == "__main__":
    unittest.main()
