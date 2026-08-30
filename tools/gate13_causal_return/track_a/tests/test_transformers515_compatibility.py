from __future__ import annotations

import contextlib
import unittest

import torch

from tools.gate13_causal_return.track_a.phase2_runner import (
    TrackARuntimeError,
    _generate_one,
)


class FakeBatchEncoding(dict):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.requested_device: str | None = None

    def to(self, device: str) -> "FakeBatchEncoding":
        self.requested_device = device
        return self


class FakeTorch:
    @staticmethod
    def inference_mode():
        return contextlib.nullcontext()


class SpyModel:
    def __init__(self, output: torch.Tensor) -> None:
        self.output = output
        self.positional_args: tuple[object, ...] | None = None
        self.keyword_args: dict[str, object] | None = None

    def generate(self, *args, **kwargs):
        self.positional_args = args
        self.keyword_args = kwargs
        return self.output


class FakeTokenizer:
    eos_token_id = 2

    def __init__(self, encoded: FakeBatchEncoding) -> None:
        self.encoded = encoded
        self.template_call: dict[str, object] | None = None
        self.decode_call: dict[str, object] | None = None

    def apply_chat_template(self, messages, **kwargs):
        self.template_call = {"messages": messages, **kwargs}
        return self.encoded

    def decode(self, continuation, **kwargs):
        self.decode_call = {"continuation": continuation.clone(), **kwargs}
        return "decoded"


class Transformers515GenerateCompatibilityTests(unittest.TestCase):
    def _fixture(self):
        input_ids = torch.tensor([[7, 8, 9]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)
        encoded = FakeBatchEncoding(
            {"input_ids": input_ids, "attention_mask": attention_mask}
        )
        tokenizer = FakeTokenizer(encoded)
        model = SpyModel(torch.tensor([[7, 8, 9, 10]], dtype=torch.long))
        return input_ids, attention_mask, encoded, tokenizer, model

    def test_batch_encoding_is_keyword_expanded_without_positional_arguments(self) -> None:
        _, _, encoded, tokenizer, model = self._fixture()
        result = _generate_one(FakeTorch, tokenizer, model, "frozen prompt", 84)
        self.assertEqual(result, "decoded")
        self.assertEqual(encoded.requested_device, "cuda:0")
        self.assertEqual(model.positional_args, ())
        self.assertEqual(
            set(model.keyword_args or {}),
            {"input_ids", "attention_mask", "do_sample", "max_new_tokens", "pad_token_id"},
        )
        self.assertEqual(model.keyword_args["max_new_tokens"], 84)
        self.assertFalse(model.keyword_args["do_sample"])
        self.assertEqual(model.keyword_args["pad_token_id"], tokenizer.eos_token_id)

    def test_tokenization_tensors_reach_generate_identically(self) -> None:
        input_ids, attention_mask, _, tokenizer, model = self._fixture()
        before = {
            "input_ids": input_ids.clone(),
            "attention_mask": attention_mask.clone(),
        }
        _generate_one(FakeTorch, tokenizer, model, "frozen prompt", 84)
        for name, source in (
            ("input_ids", input_ids),
            ("attention_mask", attention_mask),
        ):
            observed = model.keyword_args[name]
            self.assertIs(observed, source)
            self.assertEqual(observed.shape, before[name].shape)
            self.assertEqual(observed.dtype, before[name].dtype)
            self.assertEqual(observed.device, before[name].device)
            self.assertTrue(torch.equal(observed, before[name]))

    def test_model_input_generation_kwarg_collision_fails_before_generate(self) -> None:
        _, _, encoded, tokenizer, model = self._fixture()
        encoded["max_new_tokens"] = torch.tensor([84])
        with self.assertRaisesRegex(TrackARuntimeError, "collide: max_new_tokens"):
            _generate_one(FakeTorch, tokenizer, model, "frozen prompt", 84)
        self.assertIsNone(model.positional_args)
        self.assertIsNone(model.keyword_args)

    def test_prompt_template_and_decode_contract_are_unchanged(self) -> None:
        _, _, _, tokenizer, model = self._fixture()
        _generate_one(FakeTorch, tokenizer, model, "frozen prompt", 84)
        self.assertEqual(
            tokenizer.template_call,
            {
                "messages": [{"role": "user", "content": "frozen prompt"}],
                "tokenize": True,
                "add_generation_prompt": True,
                "enable_thinking": False,
                "return_tensors": "pt",
            },
        )
        self.assertTrue(tokenizer.decode_call["skip_special_tokens"])
        self.assertTrue(
            torch.equal(
                tokenizer.decode_call["continuation"], torch.tensor([10], dtype=torch.long)
            )
        )


if __name__ == "__main__":
    unittest.main()
