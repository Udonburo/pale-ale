"""Exact-runtime proof for the syntax-only Track A constrained channel."""

from __future__ import annotations

import ast
import importlib.metadata
import inspect
import platform
from pathlib import Path
from typing import Any, Mapping

from tools.gate13_causal_return.phase2_common import read_json, sha256_bytes, sha256_json
from tools.gate13_causal_return.track_a import constrained_channel
from tools.gate13_causal_return.track_a.compile_constrained_channel import (
    all_scientific_cases,
    legacy_free_generation_max_new_tokens_for_case,
    validate_compiled_manifests,
)
from tools.gate13_causal_return.track_a.constrained_channel import (
    ConstrainedTokenAutomaton,
    PrefixAllowedTokens,
    RegisterSyntax,
    syntax_for_case,
)
from tools.gate13_causal_return.track_a.parse_phase2_output import parse_phase2_output
from tools.gate13_causal_return.track_a.parse_register_output import parse_register_output


MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
EXPECTED_PACKAGES = {
    "python": "3.11.2",
    "torch": "2.7.1+cu126",
    "transformers": "5.15.0",
    "tokenizers": "0.22.2",
}


class ConstrainedValidationError(ValueError):
    """Raised when an exact constrained-channel proof fails."""


def exact_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": importlib.metadata.version("torch"),
        "transformers": importlib.metadata.version("transformers"),
        "tokenizers": importlib.metadata.version("tokenizers"),
    }


def _tensor_identity(tensor: Any) -> dict[str, Any]:
    contiguous = tensor.detach().cpu().contiguous()
    raw = contiguous.view(-1).numpy().tobytes(order="C")
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "content_sha256": sha256_bytes(raw),
    }


def _representative_cases() -> list[Mapping[str, Any]]:
    stages = all_scientific_cases()
    representatives: dict[str, Mapping[str, Any]] = {}
    # Prefer A0 so context proofs remain short while covering every grammar shape.
    for stage in ("A0", "A1", "A2"):
        for case in stages[stage]:
            syntax = syntax_for_case(case)
            representatives.setdefault(syntax.grammar_id, case)
    return [representatives[key] for key in sorted(representatives)]


def _parser_for(case: Mapping[str, Any]):
    if str(case.get("stage")) == "A0" and str(case.get("condition")) != "N":
        return parse_register_output
    return parse_phase2_output


def prove_exact_token_language(*, tokenizer: Any) -> dict[str, Any]:
    stages = all_scientific_cases()
    all_cases = [case for stage in ("A0", "A1", "A2") for case in stages[stage]]
    capacity_rows = []
    legacy_shortfall_rows = []
    for case in all_cases:
        automaton = ConstrainedTokenAutomaton(
            tokenizer=tokenizer,
            syntax=syntax_for_case(case),
        )
        constrained_max = automaton.required_new_token_count
        legacy_max = legacy_free_generation_max_new_tokens_for_case(case)
        capacity_rows.append(
            (
                str(case["case_id"]),
                automaton.required_new_token_count,
                constrained_max,
            )
        )
        if automaton.required_new_token_count > legacy_max:
            legacy_shortfall_rows.append(
                (
                    str(case["case_id"]),
                    automaton.required_new_token_count,
                    legacy_max,
                )
            )

    shape_proofs = []
    total_assignments = 0
    for case in _representative_cases():
        syntax = syntax_for_case(case)
        automaton = ConstrainedTokenAutomaton(tokenizer=tokenizer, syntax=syntax)
        proof = automaton.prove_all_assignments()
        prompt = str(case["prompt"])
        messages = [{"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        )
        prompt_ids = [int(value) for value in encoded["input_ids"][0].tolist()]
        decoded_prompt = tokenizer.decode(
            prompt_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if decoded_prompt != rendered:
            raise ConstrainedValidationError(
                f"chat-template prompt token context mismatch for {case['case_id']}"
            )

        context_checked = 0
        for assignment in syntax.assignments():
            path = automaton.token_path(assignment)
            content = automaton.validate_complete_path(path)
            combined = tokenizer.decode(
                [*prompt_ids, *path[:-1]],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if combined != rendered + content:
                raise ConstrainedValidationError(
                    f"tokenizer context changed constrained output for {case['case_id']}"
                )
            context_checked += 1
        if context_checked != syntax.assignment_count:
            raise ConstrainedValidationError("tokenizer context proof count mismatch")

        # Parser-positive paths exhaust every register trajectory while binding
        # only the test fixture's answer to its final generated register.  The
        # channel itself retains both answer branches, as proved above.
        parser = _parser_for(case)
        parser_positive_count = 0
        if syntax.direct_answer:
            for answer in (0, 1):
                parser(case, syntax.render((answer,)))
                parser_positive_count += 1
        else:
            # Enumerate register slots directly; no transition validity is used.
            from itertools import product

            for register_values in product((0, 1), repeat=len(syntax.expected_steps)):
                assignment = (*register_values, register_values[-1])
                parser(case, syntax.render(assignment))
                parser_positive_count += 1
        inconsistent = (0,) * (syntax.semantic_slot_count - 1) + (1,)
        inconsistent_path = automaton.token_path(inconsistent)
        automaton.validate_complete_path(inconsistent_path)

        shape_proofs.append(
            {
                **proof,
                "representative_case_id": case["case_id"],
                "rendered_prompt_sha256": sha256_bytes(rendered.encode("utf-8")),
                "prompt_input_ids": _tensor_identity(encoded["input_ids"]),
                "attention_mask": _tensor_identity(encoded["attention_mask"]),
                "tokenizer_context_assignment_count": context_checked,
                "strict_parser_positive_assignment_count": parser_positive_count,
                "inconsistent_answer_path_allowed": True,
            }
        )
        total_assignments += int(proof["assignment_count"])

    return {
        "status": "PASS",
        "grammar_shape_count": len(shape_proofs),
        "scientific_case_capacity_count": len(capacity_rows),
        "constrained_output_length_policy": (
            "exact canonical syntax content token count plus one terminal EOS"
        ),
        "legacy_free_generation_shortfall_case_count": len(legacy_shortfall_rows),
        "legacy_free_generation_shortfall_rows_sha256": sha256_json(
            legacy_shortfall_rows
        ),
        "exhaustive_assignment_count": total_assignments,
        "capacity_rows_sha256": sha256_json(capacity_rows),
        "shape_proofs": shape_proofs,
    }


def run_tiny_transformers515_integration(*, tokenizer: Any) -> dict[str, Any]:
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(13)
    syntax = RegisterSyntax((), direct_answer=True)
    automaton = ConstrainedTokenAutomaton(tokenizer=tokenizer, syntax=syntax)
    prompt_ids = [1, 3, 4]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    branch_probe = PrefixAllowedTokens(
        automaton=automaton,
        prompt_token_ids=prompt_ids,
    )
    if set(branch_probe(0, input_ids[0])) != {
        automaton.zero_token_id,
        automaton.one_token_id,
    }:
        raise ConstrainedValidationError("tiny integration lost a semantic branch")
    config = Qwen3Config(
        vocab_size=len(tokenizer),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        bos_token_id=1,
        eos_token_id=automaton.eos_token_id,
        pad_token_id=automaton.eos_token_id,
    )
    model = Qwen3ForCausalLM(config).eval()
    callback = PrefixAllowedTokens(
        automaton=automaton,
        prompt_token_ids=prompt_ids,
    )
    before = {
        "input_ids": _tensor_identity(input_ids),
        "attention_mask": _tensor_identity(attention_mask),
    }
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=2,
            pad_token_id=automaton.eos_token_id,
            prefix_allowed_tokens_fn=callback,
        )
    after = {
        "input_ids": _tensor_identity(input_ids),
        "attention_mask": _tensor_identity(attention_mask),
    }
    if before != after:
        raise ConstrainedValidationError("tiny integration mutated model inputs")
    continuation = [int(value) for value in output[0, len(prompt_ids) :].tolist()]
    response = automaton.validate_complete_path(continuation)
    return {
        "status": "PASS",
        "model": "RANDOMLY_INITIALIZED_TINY_LOCAL_QWEN3",
        "exact_packages": exact_versions(),
        "output_shape": list(output.shape),
        "response": response,
        "returned_token_ids": continuation,
        "input_identity_before": before,
        "input_identity_after": after,
        "scientific_weights_loaded": False,
        "scientific_case_output_generated": False,
        "scientific_forward_count": 0,
    }


def validate_no_oracle_dependency() -> dict[str, Any]:
    source = inspect.getsource(constrained_channel)
    tree = ast.parse(source)
    imports = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    ]
    if any("oracle" in name for name in imports):
        raise ConstrainedValidationError("constraint construction imports an oracle")
    syntax_source = inspect.getsource(constrained_channel.syntax_for_case)
    forbidden = ("expected_text", "base_trace", "edited_trace", "semantic_answer")
    if any(name in syntax_source for name in forbidden):
        raise ConstrainedValidationError("constraint syntax reads a semantic truth field")
    return {
        "status": "PASS",
        "oracle_imports": 0,
        "semantic_truth_fields_read": 0,
        "constraint_source_sha256": sha256_bytes(source.encode("utf-8")),
    }


def run_exact_constrained_channel_validation(
    *, repo_root: Path, model_snapshot: Path
) -> dict[str, Any]:
    observed = exact_versions()
    if observed != EXPECTED_PACKAGES:
        raise ConstrainedValidationError(
            f"exact package environment mismatch: {observed!r}"
        )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_snapshot,
        revision=MODEL_REVISION,
        use_fast=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    phase2_dir = repo_root / "analysis/gate13_causal_return/phase2"
    channel_manifest = read_json(
        phase2_dir / "track_a_constrained_channel_manifest.json"
    )
    m1_manifest = read_json(
        repo_root
        / "tools/gate13_causal_return/modal/m1_constrained_preflight_manifest.json"
    )
    manifest_validation = validate_compiled_manifests(
        channel_manifest, m1_manifest
    )
    token_language = prove_exact_token_language(tokenizer=tokenizer)
    tiny = run_tiny_transformers515_integration(tokenizer=tokenizer)
    dependency = validate_no_oracle_dependency()
    return {
        "schema_version": "gate13_track_a_constrained_exact_validation_v1",
        "status": "PASS",
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "exact_packages": observed,
        "manifest_validation": manifest_validation,
        "token_language_proof": token_language,
        "tiny_transformers515_integration": tiny,
        "no_oracle_dependency": dependency,
        "scientific_weights_loaded": False,
        "scientific_case_output_generated": False,
        "scientific_forward_count": 0,
    }
