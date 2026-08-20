"""Bounded Track A runner for the syntax-constrained register channel."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from tools.gate13_causal_return.phase2_common import read_json, sha256_bytes, write_json

from .compile_constrained_channel import (
    M1_CASE_IDS,
    all_scientific_cases,
    validate_compiled_manifests,
)
from .constrained_channel import (
    ConstrainedChannelError,
    ConstrainedTokenAutomaton,
    PrefixAllowedTokens,
    syntax_for_case,
)
from .evaluate_phase2 import evaluate_a0, evaluate_a1, evaluate_a2
from .parse_phase2_output import Phase2OutputParseError, parse_phase2_output
from .parse_register_output import OutputParseError, parse_register_output
from .phase2_runner import (
    TrackARuntimeError,
    _append_record as _base_append_record,
    _load_attempts,
    _load_completed,
    _load_exact_model,
    runtime_probe,
)


class ConstrainedRunnerError(TrackARuntimeError):
    """Fail-closed constrained-channel execution error."""


AppendRecord = Callable[[Path, Mapping[str, Any]], None]


def _tensor_sha256(tensor: Any) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    return sha256_bytes(contiguous.view(-1).numpy().tobytes(order="C"))


class ConstrainedGenerator:
    """Load-independent syntax channel; the model still chooses every bit."""

    def __init__(self, *, torch: Any, tokenizer: Any, model: Any) -> None:
        self.torch = torch
        self.tokenizer = tokenizer
        self.model = model
        self.automata: dict[str, ConstrainedTokenAutomaton] = {}
        self.case_call_count = 0

    def _automaton(self, case: Mapping[str, Any]) -> ConstrainedTokenAutomaton:
        syntax = syntax_for_case(case)
        if syntax.grammar_id not in self.automata:
            self.automata[syntax.grammar_id] = ConstrainedTokenAutomaton(
                tokenizer=self.tokenizer,
                syntax=syntax,
            )
        return self.automata[syntax.grammar_id]

    def __call__(self, case: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        prompt = str(case["prompt"])
        encoded = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        ).to("cuda:0")
        model_inputs = dict(encoded)
        required_inputs = {"input_ids", "attention_mask"}
        if not required_inputs.issubset(model_inputs):
            raise ConstrainedRunnerError("chat-template BatchEncoding lacks required tensors")
        if list(model_inputs["input_ids"].shape)[0] != 1:
            raise ConstrainedRunnerError("constrained channel requires batch size one")

        automaton = self._automaton(case)
        # Output length and EOS position are part of the authorized syntax-only
        # channel.  The exact endpoint is assignment-independent because every
        # semantic branch is one exact tokenizer token.
        max_new_tokens = automaton.required_new_token_count
        prompt_ids = [int(value) for value in model_inputs["input_ids"][0].tolist()]
        prefix_constraint = PrefixAllowedTokens(
            automaton=automaton,
            prompt_token_ids=prompt_ids,
        )
        generation_kwargs: dict[str, Any] = {
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "prefix_allowed_tokens_fn": prefix_constraint,
        }
        collisions = set(model_inputs).intersection(generation_kwargs)
        if collisions:
            raise ConstrainedRunnerError(
                "model inputs and constrained generation kwargs collide: "
                + ", ".join(sorted(collisions))
            )

        with self.torch.inference_mode():
            output = self.model.generate(**model_inputs, **generation_kwargs)
        self.case_call_count += 1
        if len(output.shape) != 2 or int(output.shape[0]) != 1:
            raise ConstrainedRunnerError("generate returned an unexpected output shape")
        prompt_length = len(prompt_ids)
        returned_prefix = [int(value) for value in output[0, :prompt_length].tolist()]
        if returned_prefix != prompt_ids:
            raise ConstrainedRunnerError("decoder-only generate changed the exact prompt prefix")
        continuation_ids = [int(value) for value in output[0, prompt_length:].tolist()]
        try:
            response = automaton.validate_complete_path(continuation_ids)
        except ConstrainedChannelError as exc:
            raise ConstrainedRunnerError(
                f"generated token path violated the syntax channel: {exc}"
            ) from exc
        decoded = self.tokenizer.decode(
            continuation_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if decoded != response:
            raise ConstrainedRunnerError("special-token removal changed constrained content")
        trace = {
            "schema_version": "gate13_track_a_constrained_generation_trace_v1",
            "grammar_id": automaton.syntax.grammar_id,
            "semantic_slot_count": automaton.syntax.semantic_slot_count,
            "all_assignment_count": automaton.syntax.assignment_count,
            "zero_token_id": automaton.zero_token_id,
            "one_token_id": automaton.one_token_id,
            "eos_token_id": automaton.eos_token_id,
            "required_new_token_count": automaton.required_new_token_count,
            "constrained_max_new_tokens": max_new_tokens,
            "constrained_output_length_policy": (
                "exact canonical syntax content token count plus one terminal EOS"
            ),
            "prompt_token_count": prompt_length,
            "prompt_input_ids_sha256": _tensor_sha256(model_inputs["input_ids"]),
            "attention_mask_sha256": _tensor_sha256(model_inputs["attention_mask"]),
            "returned_token_ids": continuation_ids,
            "returned_token_count": len(continuation_ids),
            "decoded_response_sha256": sha256_bytes(response.encode("utf-8")),
            "do_sample": False,
            "oracle_consulted": False,
            "transition_validity_filtered": False,
            "answer_equality_filtered": False,
        }
        return response, trace


def _execute_stage(
    *,
    stage: str,
    cases: Sequence[Mapping[str, Any]],
    state_path: Path,
    generator: ConstrainedGenerator,
    forward_count_before_stage: int,
    cumulative_ceiling: int,
    append_record: AppendRecord = _base_append_record,
) -> tuple[list[dict[str, Any]], int]:
    allowed = {str(case["case_id"]) for case in cases}
    records = _load_completed(state_path, allowed)
    completed = {str(record["case_id"]) for record in records}
    attempt_path = state_path.with_name(state_path.stem + "_attempts.jsonl")
    attempts = _load_attempts(attempt_path, allowed)
    attempted = {str(record["case_id"]) for record in attempts}
    if attempted - completed:
        raise ConstrainedRunnerError(
            "a prior case-level model call has no saved response; retry is forbidden"
        )
    if completed - attempted:
        raise ConstrainedRunnerError("saved response lacks its immutable attempt entry")
    forward_count = int(forward_count_before_stage) + len(attempts)
    for case in cases:
        case_id = str(case["case_id"])
        if case_id in completed:
            continue
        if forward_count >= cumulative_ceiling:
            raise ConstrainedRunnerError("cumulative forward ceiling reached")
        forward_count += 1
        append_record(
            attempt_path,
            {
                "case_id": case_id,
                "stage": stage,
                "case_level_model_forward": forward_count,
                "status": "STARTED_NO_RETRY",
            },
        )
        response, instrument_trace = generator(case)
        record = {
            "case_id": case_id,
            "stage": stage,
            "response": response,
            "instrument_trace": instrument_trace,
            "case_level_model_forward": forward_count,
        }
        append_record(state_path, record)
        records.append(record)
    return records, forward_count


def _parse_response(case: Mapping[str, Any], response: str) -> dict[str, Any]:
    parser: Callable[[Mapping[str, object], str], Any]
    if str(case.get("stage")) == "A0" and str(case.get("condition")) != "N":
        parser = parse_register_output
    else:
        parser = parse_phase2_output
    try:
        parsed = parser(case, response)
        return {
            "status": "PASS",
            "parser": parser.__name__,
            "parsed": dataclasses.asdict(parsed),
            "error": None,
        }
    except (OutputParseError, Phase2OutputParseError) as exc:
        return {
            "status": "REJECT",
            "parser": parser.__name__,
            "parsed": None,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }


def _m1_cases(
    *, cases_by_id: Mapping[str, Mapping[str, Any]], m1_manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    rows = list(m1_manifest.get("cases") or [])
    if [row.get("case_id") for row in rows] != list(M1_CASE_IDS):
        raise ConstrainedRunnerError("M1 development case ordering drifted")
    for binding in rows:
        case_id = str(binding["case_id"])
        case = dict(cases_by_id[case_id])
        if binding.get("contributes_to_a0_a1_a2_metrics") is not False:
            raise ConstrainedRunnerError("M1 development case leaked into metrics")
        if binding.get("reused_as_a0_checkpoint") is not False:
            raise ConstrainedRunnerError("M1 development case leaked into A0 checkpoints")
        syntax = syntax_for_case(case)
        if syntax.grammar_id != binding.get("grammar_id"):
            raise ConstrainedRunnerError("M1 grammar binding mismatch")
        selected.append(case)
    return selected


def run_constrained_track_a(
    *,
    phase2_dir: Path,
    output_dir: Path,
    model_runtime: tuple[Any, Any, Any] | None = None,
    runtime_probe_override: Mapping[str, Any] | None = None,
    append_record: AppendRecord = _base_append_record,
    require_clean: bool = True,
) -> dict[str, Any]:
    from tools.gate13_causal_return.track_a.validate_constrained_channel_lock import (
        validate_constrained_channel_lock,
    )

    validation = validate_constrained_channel_lock(
        phase2_dir=phase2_dir,
        require_clean=require_clean,
    )
    if validation["status"] != "PASS" or not validation["execution_authorized"]:
        raise ConstrainedRunnerError("constrained-channel lock validation did not authorize execution")
    lock = read_json(phase2_dir / "phase2_a_constrained_channel_lock.json")
    parent_lock = read_json(phase2_dir / "phase2_a_lock.json")
    channel_manifest = read_json(phase2_dir / lock["artifacts"]["channel_manifest"]["path"])
    m1_manifest = read_json(phase2_dir / lock["artifacts"]["m1_manifest"]["path"])
    validate_compiled_manifests(channel_manifest, m1_manifest)

    probe = dict(runtime_probe_override) if runtime_probe_override is not None else runtime_probe(parent_lock)
    prior_count = int(lock["forward_accounting"]["prior_consumed_count"])
    cumulative_ceiling = int(lock["forward_accounting"]["cumulative_ceiling"])
    if probe.get("status") != "PASS":
        result = {
            "schema_version": "gate13_track_a_constrained_result_v1",
            "status": "READY_FOR_EXACT_EXTERNAL_RUNTIME",
            "M0": "BLOCKED_EXACT_RUNTIME_UNAVAILABLE",
            "M1": "UNOPENED",
            "TRACK_A_A0": "UNOPENED",
            "TRACK_A_A1": "UNOPENED",
            "TRACK_A_A2": "UNOPENED",
            "model_forward_count": prior_count,
            "fresh_model_forward_count": 0,
            "runtime_probe": probe,
            "activation_extraction_count": 0,
        }
        write_json(output_dir / "track_a_constrained_result.json", result)
        return result

    torch, tokenizer, model = model_runtime or _load_exact_model(parent_lock)
    generator = ConstrainedGenerator(torch=torch, tokenizer=tokenizer, model=model)
    stages = all_scientific_cases()
    cases_by_id = {str(case["case_id"]): case for rows in stages.values() for case in rows}
    m1_cases = _m1_cases(cases_by_id=cases_by_id, m1_manifest=m1_manifest)

    forward_count = prior_count
    m1_records, forward_count = _execute_stage(
        stage="M1_DEVELOPMENT_PREFLIGHT",
        cases=m1_cases,
        state_path=output_dir / "m1_development_state.jsonl",
        generator=generator,
        forward_count_before_stage=forward_count,
        cumulative_ceiling=cumulative_ceiling,
        append_record=append_record,
    )
    m1_parses = [
        {"case_id": record["case_id"], **_parse_response(case, str(record["response"]))}
        for case, record in zip(m1_cases, m1_records)
    ]
    m1_pass = len(m1_records) == len(m1_cases) and all(
        row["status"] == "PASS" for row in m1_parses
    )
    m1_result = {
        "status": "PASS" if m1_pass else "CONSTRAINED_CHANNEL_IMPLEMENTATION_BLOCKER",
        "development_only": True,
        "contributes_to_a0_a1_a2_metrics": False,
        "reused_as_a0_checkpoint": False,
        "case_count": len(m1_records),
        "parses": m1_parses,
    }
    result: dict[str, Any] = {
        "schema_version": "gate13_track_a_constrained_result_v1",
        "status": "MANDATORY_STOP",
        "M0": "PASS",
        "M1": m1_result["status"],
        "M1_result": m1_result,
        "TRACK_A_A0": "UNOPENED",
        "TRACK_A_A1": "UNOPENED",
        "TRACK_A_A2": "UNOPENED",
        "model_forward_count": forward_count,
        "fresh_model_forward_count": forward_count - prior_count,
        "activation_extraction_count": 0,
        "runtime_probe": probe,
    }
    if not m1_pass:
        write_json(output_dir / "track_a_constrained_result.json", result)
        return result

    # M1 is isolated.  A0 begins with a distinct empty checkpoint surface, so
    # the development responses never enter a scientific metric.
    a0_records, forward_count = _execute_stage(
        stage="A0",
        cases=stages["A0"],
        state_path=output_dir / "a0_state.jsonl",
        generator=generator,
        forward_count_before_stage=forward_count,
        cumulative_ceiling=cumulative_ceiling,
        append_record=append_record,
    )
    a0 = evaluate_a0(stages["A0"], a0_records)
    result.update(
        {
            "TRACK_A_A0": a0["status"],
            "A0": a0,
            "model_forward_count": forward_count,
            "fresh_model_forward_count": forward_count - prior_count,
        }
    )
    if a0["status"] == "PASS":
        a1_records, forward_count = _execute_stage(
            stage="A1",
            cases=stages["A1"],
            state_path=output_dir / "a1_state.jsonl",
            generator=generator,
            forward_count_before_stage=forward_count,
            cumulative_ceiling=cumulative_ceiling,
            append_record=append_record,
        )
        a1 = evaluate_a1(stages["A1"], a1_records)
        result.update(
            {
                "TRACK_A_A1": a1["status"],
                "A1": a1,
                "model_forward_count": forward_count,
                "fresh_model_forward_count": forward_count - prior_count,
            }
        )
        if a1["status"] == "PASS":
            a2_records, forward_count = _execute_stage(
                stage="A2",
                cases=stages["A2"],
                state_path=output_dir / "a2_state.jsonl",
                generator=generator,
                forward_count_before_stage=forward_count,
                cumulative_ceiling=cumulative_ceiling,
                append_record=append_record,
            )
            a2 = evaluate_a2(stages["A2"], a2_records)
            result.update(
                {
                    "TRACK_A_A2": a2["status"],
                    "A2": a2,
                    "model_forward_count": forward_count,
                    "fresh_model_forward_count": forward_count - prior_count,
                }
            )
    write_json(output_dir / "track_a_constrained_result.json", result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_constrained_track_a(
        phase2_dir=args.phase2_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
