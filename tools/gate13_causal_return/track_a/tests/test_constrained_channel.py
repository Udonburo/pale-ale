from __future__ import annotations

import ast
import copy
import inspect
import tempfile
import unittest
from itertools import product
from pathlib import Path

import torch

from tools.gate13_causal_return.phase2_common import read_json, sha256_file, sha256_json
from tools.gate13_causal_return.track_a import constrained_channel
from tools.gate13_causal_return.track_a.compile_constrained_channel import (
    M1_CASE_IDS,
    all_scientific_cases,
    compile_channel_manifest,
    compile_m1_manifest,
    validate_compiled_manifests,
)
from tools.gate13_causal_return.track_a.constrained_channel import (
    ConstrainedChannelError,
    ConstrainedTokenAutomaton,
    PrefixAllowedTokens,
    RegisterSyntax,
    syntax_for_case,
)
from tools.gate13_causal_return.track_a.constrained_runner import (
    ConstrainedGenerator,
    ConstrainedRunnerError,
    _execute_stage,
)
from tools.gate13_causal_return.track_a.parse_phase2_output import (
    Phase2OutputParseError,
    parse_phase2_output,
)
from tools.gate13_causal_return.track_a.parse_register_output import (
    OutputParseError,
    parse_register_output,
)
from tools.gate13_causal_return.track_a.validate_constrained_channel_lock import (
    validate_constrained_channel_lock,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
PHASE2_DIR = REPO_ROOT / "analysis/gate13_causal_return/phase2"


class CharacterTokenizer:
    """Exact additive ASCII tokenizer with one reserved EOS token."""

    eos_token_id = 10_000
    all_special_ids = [eos_token_id]

    class Encoding(dict):
        def to(self, device: str):
            if device != "cuda:0":
                raise AssertionError("runner changed the frozen device")
            return self

    def encode(self, text: str, *, add_special_tokens: bool = False):
        if add_special_tokens:
            raise AssertionError("test tokenizer does not add special tokens")
        return [ord(character) for character in text]

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        if clean_up_tokenization_spaces:
            raise AssertionError("syntax channel must disable tokenizer cleanup")
        pieces = []
        for token_id in token_ids:
            value = int(token_id)
            if value == self.eos_token_id:
                if skip_special_tokens:
                    continue
                pieces.append("<EOS>")
            else:
                pieces.append(chr(value))
        return "".join(pieces)

    def apply_chat_template(self, messages, **kwargs):
        if kwargs != {
            "tokenize": True,
            "add_generation_prompt": True,
            "enable_thinking": False,
            "return_tensors": "pt",
        }:
            raise AssertionError(f"chat template settings drifted: {kwargs!r}")
        if len(messages) != 1 or messages[0]["role"] != "user":
            raise AssertionError("unexpected message structure")
        return self.Encoding(
            {
                "input_ids": torch.tensor([[500, 501]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            }
        )


class FrozenSyntaxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        stages = all_scientific_cases()
        cls.cases = {
            str(case["case_id"]): case for rows in stages.values() for case in rows
        }

    def test_all_fixed_literals_and_whitespace_match_existing_contract(self) -> None:
        case = self.cases["a0-l12-y0-early-r0-S"]
        syntax = syntax_for_case(case)
        text = syntax.render((0,) * syntax.semantic_slot_count)
        self.assertTrue(text.startswith("r0 = 0\nr1 = 0"))
        self.assertTrue(text.endswith("r12 = 0\nanswer = 0"))
        self.assertNotIn("  ", text)
        self.assertFalse(text.endswith("\n"))
        self.assertTrue(syntax.accepts_text(text))
        self.assertFalse(syntax.accepts_text(text + "\n"))
        self.assertFalse(syntax.accepts_text(text.replace("\n", "  \n", 1)))

    def test_all_independent_assignments_have_paths_at_maximum_length(self) -> None:
        syntax = RegisterSyntax(tuple(range(13)))
        automaton = ConstrainedTokenAutomaton(
            tokenizer=CharacterTokenizer(), syntax=syntax
        )
        proof = automaton.prove_all_assignments()
        self.assertEqual(proof["semantic_slot_count"], 14)
        self.assertEqual(proof["assignment_count"], 1 << 14)

    def test_every_semantic_slot_exposes_both_tokens_and_eos_only_at_end(self) -> None:
        syntax = RegisterSyntax((2, 3, 4))
        automaton = ConstrainedTokenAutomaton(
            tokenizer=CharacterTokenizer(), syntax=syntax
        )
        state = automaton.start_state
        branch_count = 0
        for component in automaton.components:
            if component.kind == "literal":
                for token_id in component.token_ids:
                    self.assertNotIn(automaton.eos_token_id, automaton.allowed_token_ids(state))
                    state = automaton.consume(state, token_id)
            else:
                self.assertEqual(
                    set(automaton.allowed_token_ids(state)),
                    {automaton.zero_token_id, automaton.one_token_id},
                )
                self.assertNotIn(automaton.eos_token_id, automaton.allowed_token_ids(state))
                state = automaton.consume(state, automaton.zero_token_id)
                branch_count += 1
        self.assertEqual(branch_count, syntax.semantic_slot_count)
        self.assertEqual(automaton.allowed_token_ids(state), (automaton.eos_token_id,))

    def test_constraint_does_not_filter_transitions_or_answer_mismatch(self) -> None:
        syntax = RegisterSyntax((0, 1, 2, 3, 4))
        automaton = ConstrainedTokenAutomaton(
            tokenizer=CharacterTokenizer(), syntax=syntax
        )
        # Repeated impossible-looking transitions and answer != r4 remain in-language.
        assignment = (0, 0, 0, 0, 0, 1)
        text = automaton.validate_complete_path(automaton.token_path(assignment))
        self.assertEqual(text, syntax.render(assignment))
        self.assertTrue(syntax.accepts_text(text))

    def test_strict_parser_acceptance_and_semantic_rejection_are_unchanged(self) -> None:
        case = self.cases["a0-l04-y0-early-r0-S"]
        syntax = syntax_for_case(case)
        automaton = ConstrainedTokenAutomaton(
            tokenizer=CharacterTokenizer(), syntax=syntax
        )
        # Exhaust every register trajectory while choosing answer == final register.
        for trace in product((0, 1), repeat=5):
            assignment = (*trace, trace[-1])
            text = automaton.validate_complete_path(automaton.token_path(assignment))
            parsed = parse_register_output(case, text)
            self.assertEqual(parsed.trace_prediction, tuple(trace))
        inconsistent = syntax.render((0, 0, 0, 0, 0, 1))
        self.assertTrue(syntax.accepts_text(inconsistent))
        with self.assertRaises(OutputParseError):
            parse_register_output(case, inconsistent)

        n_case = self.cases["a0-l04-y0-early-r0-N"]
        n_syntax = syntax_for_case(n_case)
        valid = n_syntax.render((0, 0, 0, 0))
        parse_phase2_output(n_case, valid)
        with self.assertRaises(Phase2OutputParseError):
            parse_phase2_output(n_case, n_syntax.render((0, 0, 0, 1)))

    def test_constraint_construction_has_no_oracle_or_truth_dependency(self) -> None:
        source = inspect.getsource(constrained_channel)
        tree = ast.parse(source)
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertFalse(any("oracle" in name for name in imported))
        function_source = inspect.getsource(syntax_for_case)
        for forbidden in ("expected_text", "base_trace", "edited_trace", "semantic_answer"):
            self.assertNotIn(forbidden, function_source)

        case = self.cases["a0-l12-y0-early-r0-E"]
        mutated = copy.deepcopy(case)
        mutated["bits"] = [1 - int(value) for value in mutated["bits"]]
        mutated["base_trace"] = list(reversed(mutated["base_trace"]))
        mutated["edited_trace"] = list(reversed(mutated["edited_trace"]))
        mutated["semantic_answer"] ^= 1
        self.assertEqual(syntax_for_case(case), syntax_for_case(mutated))

    def test_prefix_callback_checks_exact_prompt_and_semantic_branches(self) -> None:
        automaton = ConstrainedTokenAutomaton(
            tokenizer=CharacterTokenizer(), syntax=RegisterSyntax((), direct_answer=True)
        )
        prompt = (7, 8, 9)
        callback = PrefixAllowedTokens(automaton=automaton, prompt_token_ids=prompt)
        self.assertEqual(
            set(callback(0, torch.tensor(prompt))),
            {automaton.zero_token_id, automaton.one_token_id},
        )
        self.assertEqual(
            callback(0, torch.tensor((*prompt, automaton.zero_token_id))),
            [automaton.eos_token_id],
        )
        with self.assertRaises(ConstrainedChannelError):
            PrefixAllowedTokens(automaton=automaton, prompt_token_ids=prompt)(
                0, torch.tensor((7, 8, 10))
            )

    def test_generator_uses_transformers_prefix_constraint_and_exact_endpoint(self) -> None:
        class FakeTorch:
            @staticmethod
            def inference_mode():
                import contextlib

                return contextlib.nullcontext()

        class FakeModel:
            def __init__(self) -> None:
                self.kwargs = None

            def generate(self, *args, **kwargs):
                self.kwargs = kwargs
                if args:
                    raise AssertionError("generate received positional arguments")
                sequence = kwargs["input_ids"].clone()
                callback = kwargs["prefix_allowed_tokens_fn"]
                while True:
                    allowed = callback(0, sequence[0])
                    # Choose 1 at the semantic slot, then the uniquely forced EOS.
                    token_id = max(allowed)
                    sequence = torch.cat(
                        (sequence, torch.tensor([[token_id]], dtype=torch.long)),
                        dim=1,
                    )
                    if token_id == CharacterTokenizer.eos_token_id:
                        return sequence

        direct = next(
            case
            for case in all_scientific_cases()["A0"]
            if case["condition"] == "D"
        )
        model = FakeModel()
        generator = ConstrainedGenerator(
            torch=FakeTorch,
            tokenizer=CharacterTokenizer(),
            model=model,
        )
        response, trace = generator(direct)
        self.assertEqual(response, "1")
        self.assertEqual(trace["semantic_slot_count"], 1)
        self.assertFalse(trace["oracle_consulted"])
        self.assertFalse(trace["transition_validity_filtered"])
        self.assertFalse(trace["answer_equality_filtered"])
        self.assertFalse(model.kwargs["do_sample"])
        self.assertEqual(model.kwargs["max_new_tokens"], 2)
        self.assertEqual(
            trace["constrained_max_new_tokens"],
            trace["required_new_token_count"],
        )
        self.assertEqual(model.kwargs["pad_token_id"], CharacterTokenizer.eos_token_id)


class ManifestAndCheckpointTests(unittest.TestCase):
    def test_constrained_derivative_lock_passes_fail_closed_validation(self) -> None:
        report = validate_constrained_channel_lock(
            phase2_dir=PHASE2_DIR,
            require_clean=False,
        )
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["execution_authorized"])
        self.assertEqual(report["prior_forward_count"], 5)
        self.assertEqual(report["maximum_additional_forwards"], 595)

    def test_manifest_preserves_all_case_semantics_and_is_deterministic(self) -> None:
        first = compile_channel_manifest(phase2_dir=PHASE2_DIR)
        second = compile_channel_manifest(phase2_dir=PHASE2_DIR)
        self.assertEqual(first, second)
        self.assertEqual(first["case_counts"], {"A0": 252, "A1": 162, "A2": 90})
        self.assertEqual(first["manifest_sha256"], sha256_json({k: v for k, v in first.items() if k != "manifest_sha256"}))
        self.assertEqual(
            first["source_artifacts"]["phase2_a_lock_sha256"],
            sha256_file(PHASE2_DIR / "phase2_a_lock.json"),
        )
        m1 = compile_m1_manifest(first)
        report = validate_compiled_manifests(first, m1)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual([row["case_id"] for row in m1["cases"]], list(M1_CASE_IDS))
        self.assertEqual(m1["scientific_metric_contribution_count"], 0)
        self.assertEqual(m1["reuse_as_a0_checkpoint_count"], 0)

    def test_resume_skips_completed_cases_and_incomplete_attempt_fails_closed(self) -> None:
        class FakeGenerator:
            def __init__(self) -> None:
                self.calls = 0

            def __call__(self, case):
                self.calls += 1
                return str(case["expected_text"]), {"oracle_consulted": False}

        cases = all_scientific_cases()["A0"][:2]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "m1_development_state.jsonl"
            first = FakeGenerator()
            records, count = _execute_stage(
                stage="M1_DEVELOPMENT_PREFLIGHT",
                cases=cases,
                state_path=path,
                generator=first,  # type: ignore[arg-type]
                forward_count_before_stage=5,
                cumulative_ceiling=600,
            )
            self.assertEqual(len(records), 2)
            self.assertEqual(count, 7)
            self.assertEqual(first.calls, 2)

            resumed = FakeGenerator()
            records, count = _execute_stage(
                stage="M1_DEVELOPMENT_PREFLIGHT",
                cases=cases,
                state_path=path,
                generator=resumed,  # type: ignore[arg-type]
                forward_count_before_stage=5,
                cumulative_ceiling=600,
            )
            self.assertEqual(len(records), 2)
            self.assertEqual(count, 7)
            self.assertEqual(resumed.calls, 0)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "m1_development_state.jsonl"
            attempt = path.with_name(path.stem + "_attempts.jsonl")
            attempt.write_text(
                '{"case_id":"' + str(cases[0]["case_id"]) + '"}\n',
                encoding="utf-8",
            )
            with self.assertRaises(ConstrainedRunnerError):
                _execute_stage(
                    stage="M1_DEVELOPMENT_PREFLIGHT",
                    cases=cases,
                    state_path=path,
                    generator=FakeGenerator(),  # type: ignore[arg-type]
                    forward_count_before_stage=5,
                    cumulative_ceiling=600,
                )

    def test_tracked_generated_manifests_match_compiler(self) -> None:
        channel = read_json(PHASE2_DIR / "track_a_constrained_channel_manifest.json")
        m1 = read_json(
            REPO_ROOT
            / "tools/gate13_causal_return/modal/m1_constrained_preflight_manifest.json"
        )
        self.assertEqual(channel, compile_channel_manifest(phase2_dir=PHASE2_DIR))
        self.assertEqual(m1, compile_m1_manifest(channel))


if __name__ == "__main__":
    unittest.main()
