"""Compile byte-frozen A1/A2 manifests without loading or running a model."""

from __future__ import annotations

import argparse
import hashlib
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tools.gate13_causal_return.phase2_common import sha256_bytes, sha256_json, write_json

from .compile_register_cases import (
    DEFAULT_SEED as A0_SEED,
    base_cases as a0_base_cases,
    compile_ledger,
)
from .oracle import edited_trace, parity_trace
from .render_register_cases import (
    INTERVENTION_MARKER as A0_INTERVENTION_MARKER,
    OUTPUT_CONTRACT,
    RULE_TEXT,
    bits_text,
    trace_lines,
)


SCHEMA_VERSION = "gate13_phase2_track_a_case_manifest_v1"
SEED = "gate13_phase2_register_cases_v1"
SHOT_COUNTS = (4, 16, 64)
LENGTHS = (4, 8, 12)
EDIT_STRATA = ("early", "middle", "late")
A1_CONTROLS = ("correct", "corrupted", "shuffled")
A2_CONDITIONS = ("base", "edit", "marker_only", "undeclared_corrupt", "filler")
TARGET_COUNT = len(LENGTHS) * 2 * len(EDIT_STRATA)
A0_REVIEW1_FORWARD_COUNT = 216
A0_MARKER_ONLY_FORWARD_COUNT = 36
A0_FORWARD_COUNT = A0_REVIEW1_FORWARD_COUNT + A0_MARKER_ONLY_FORWARD_COUNT
A1_FORWARD_COUNT = TARGET_COUNT * len(SHOT_COUNTS) * len(A1_CONTROLS)
A2_FORWARD_COUNT = TARGET_COUNT * len(A2_CONDITIONS)
TIMING_PREFLIGHT_RESERVE = 48
PROJECTED_FORWARD_MAXIMUM = (
    A0_FORWARD_COUNT + A1_FORWARD_COUNT + A2_FORWARD_COUNT + TIMING_PREFLIGHT_RESERVE
)
FORWARD_CEILING = 600

INTERVENTION_MARKER = (
    "INTERVENTION: overwrite the current register with the displayed value. "
    "Continue from that value; do not repair earlier lines."
)


def _seed_int(label: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{SEED}|{label}".encode()).digest()[:8], "big")


def _all_bits(length: int, label: int) -> list[tuple[int, ...]]:
    rows: list[tuple[int, ...]] = []
    for number in range(1 << length):
        bits = tuple((number >> shift) & 1 for shift in reversed(range(length)))
        if parity_trace(bits)[-1] == label:
            rows.append(bits)
    return rows


def _edit_step(length: int, stratum: str) -> int:
    return {"early": 1, "middle": length // 2, "late": length - 1}[stratum]


def target_ledger() -> list[dict[str, Any]]:
    a0_used = {
        (int(row["length"]), tuple(int(bit) for bit in row["bits"]))
        for row in a0_base_cases(A0_SEED)
    }
    targets: list[dict[str, Any]] = []
    for length in LENGTHS:
        for label in (0, 1):
            fresh_candidates = [
                bits for bits in _all_bits(length, label) if (length, bits) not in a0_used
            ]
            candidates = (
                fresh_candidates
                if len(fresh_candidates) >= len(EDIT_STRATA)
                else _all_bits(length, label)
            )
            random.Random(_seed_int(f"target|{length}|{label}")).shuffle(candidates)
            chosen = candidates[: len(EDIT_STRATA)]
            for stratum, bits in zip(EDIT_STRATA, chosen):
                step = _edit_step(length, stratum)
                targets.append(
                    {
                        "target_id": f"target-l{length:02d}-y{label}-{stratum}",
                        "length": length,
                        "semantic_answer": label,
                        "edit_stratum": stratum,
                        "edit_step": step,
                        "bits": list(bits),
                        "base_trace": list(parity_trace(bits)),
                        "edited_trace": list(edited_trace(bits, step)),
                    }
                )
    return targets


def _corrupted_trace(bits: Sequence[int], index: int) -> tuple[int, ...]:
    base = list(parity_trace(bits))
    flip_step = 1 + (index % (len(bits) - 1))
    for step in range(flip_step, len(base)):
        base[step] ^= 1
    return tuple(base)


def demonstration_bank(targets: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    excluded = {
        tuple(int(bit) for bit in row["bits"])
        for row in targets
        if int(row["length"]) == 8
    }
    excluded.update(
        tuple(int(bit) for bit in row["bits"])
        for row in a0_base_cases(A0_SEED)
        if int(row["length"]) == 8
    )
    by_label: dict[int, list[tuple[int, ...]]] = {}
    for label in (0, 1):
        candidates = [bits for bits in _all_bits(8, label) if bits not in excluded]
        random.Random(_seed_int(f"demos|{label}")).shuffle(candidates)
        by_label[label] = candidates[:32]

    bank: list[dict[str, Any]] = []
    for block in range(16):
        block_bits = [
            by_label[0][2 * block],
            by_label[1][2 * block],
            by_label[0][2 * block + 1],
            by_label[1][2 * block + 1],
        ]
        random.Random(_seed_int(f"block|{block}")).shuffle(block_bits)
        traces = [parity_trace(bits) for bits in block_bits]
        shuffled = traces[1:] + traces[:1]
        for offset, (bits, correct, shuffled_trace) in enumerate(
            zip(block_bits, traces, shuffled)
        ):
            index = 4 * block + offset
            bank.append(
                {
                    "demo_id": f"demo-{index:03d}",
                    "bits": list(bits),
                    "semantic_answer": int(correct[-1]),
                    "correct_trace": list(correct),
                    "corrupted_trace": list(_corrupted_trace(bits, index)),
                    "shuffled_trace": list(shuffled_trace),
                }
            )
    return bank


def _demo_output(row: Mapping[str, Any], control: str) -> tuple[int, ...]:
    key = {
        "correct": "correct_trace",
        "corrupted": "corrupted_trace",
        "shuffled": "shuffled_trace",
    }[control]
    return tuple(int(value) for value in row[key])


def _render_demo(index: int, row: Mapping[str, Any], control: str) -> str:
    trace = _demo_output(row, control)
    return (
        f"Example {index}\n"
        f"Input bits:\n{bits_text(row['bits'])}\n"
        "Output:\n"
        f"{trace_lines(trace)}\n"
        f"answer = {trace[-1]}"
    )


def _render_demo_prefix(
    bank: Sequence[Mapping[str, Any]], *, shots: int, control: str
) -> str:
    return "\n\n".join(
        _render_demo(index + 1, row, control)
        for index, row in enumerate(bank[:shots])
    )


def _a1_prompt(
    target: Mapping[str, Any],
    bank: Sequence[Mapping[str, Any]],
    *,
    shots: int,
    control: str,
) -> str:
    demos = _render_demo_prefix(bank, shots=shots, control=control)
    return (
        "Infer the demonstrated register-trace format from the examples.\n\n"
        f"{demos}\n\n"
        "Target\n"
        f"Input bits:\n{bits_text(target['bits'])}\n\n"
        "Write the complete register trace from r0 through rn, then answer = rn.\n"
        f"{OUTPUT_CONTRACT}\n\nTRACE:"
    )


def _a2_prefix(target: Mapping[str, Any], condition: str) -> str:
    bits = tuple(int(bit) for bit in target["bits"])
    step = int(target["edit_step"])
    base = parity_trace(bits)
    edited = edited_trace(bits, step)
    if condition == "base":
        return "The demonstrated trace is supplied through the current step:\n" + trace_lines(
            base[: step + 1]
        )
    if condition in {"edit", "marker_only"}:
        current = edited[step] if condition == "edit" else base[step]
        return (
            "The demonstrated trace is supplied through the preceding step:\n"
            + trace_lines(base[:step])
            + f"\n\n{INTERVENTION_MARKER}\n"
            + f"r{step} = {current}"
        )
    if condition == "undeclared_corrupt":
        prefix = list(base[:step]) + [edited[step]]
        return (
            "A candidate trace is supplied through the current step. "
            "No intervention is declared:\n"
            + trace_lines(prefix)
        )
    if condition == "filler":
        filler = "\n".join(
            f"z{index} = {'A' if index % 2 == 0 else 'B'}"
            for index in range(step + 1)
        )
        return "A noncomputational ledger is supplied through the current step:\n" + filler
    raise ValueError(f"unknown A2 condition: {condition}")


def _a2_prompt(
    target: Mapping[str, Any],
    bank: Sequence[Mapping[str, Any]],
    *,
    condition: str,
) -> str:
    demos = _render_demo_prefix(bank, shots=64, control="correct")
    step = int(target["edit_step"])
    length = int(target["length"])
    return (
        "Infer the demonstrated register-trace format from the examples.\n\n"
        f"{demos}\n\n"
        "Target\n"
        f"Input bits:\n{bits_text(target['bits'])}\n\n"
        f"{_a2_prefix(target, condition)}\n\n"
        f"Continue with r{step + 1} through r{length}, then answer = r{length}.\n"
        f"{OUTPUT_CONTRACT}\n\nCONTINUE:"
    )


def _expected_a1(target: Mapping[str, Any]) -> str:
    trace = tuple(int(value) for value in target["base_trace"])
    return trace_lines(trace) + f"\nanswer = {trace[-1]}"


def _expected_a2(target: Mapping[str, Any], condition: str) -> str:
    step = int(target["edit_step"])
    trace_key = "edited_trace" if condition == "edit" else "base_trace"
    trace = tuple(int(value) for value in target[trace_key])
    return trace_lines(trace, start=step + 1) + f"\nanswer = {trace[-1]}"


def _a0_marker_only_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for base in a0_base_cases(A0_SEED):
        bits = tuple(int(bit) for bit in base["bits"])
        trace = parity_trace(bits)
        step = int(base["edit_step"])
        prompt = (
            f"{RULE_TEXT}\n\nInput bits:\n{bits_text(bits)}\n"
            f"\nThe base trace was correct through step {step - 1}:\n"
            + trace_lines(trace[:step])
            + f"\n\n{A0_INTERVENTION_MARKER}\n"
            + f"r{step} = {trace[step]}\n\n"
            + f"Continue with r{step + 1} through r{len(bits)}, then answer = r{len(bits)}.\n"
            + OUTPUT_CONTRACT
            + "\n\nCONTINUE:"
        )
        expected = trace_lines(trace, start=step + 1) + f"\nanswer = {trace[-1]}"
        cases.append(
            {
                **base,
                "stage": "A0",
                "case_id": f"{base['base_id']}-N",
                "target_id": base["base_id"],
                "shots": 0,
                "condition": "N",
                "prompt": prompt,
                "expected_text": expected,
                "expected_steps": list(range(step + 1, len(bits) + 1)),
                "matched_to_case_id": f"{base['base_id']}-E",
            }
        )
    return cases


def compile_cases() -> dict[str, Any]:
    targets = target_ledger()
    bank = demonstration_bank(targets)
    a1_cases: list[dict[str, Any]] = []
    for target in targets:
        for shots in SHOT_COUNTS:
            for control in A1_CONTROLS:
                prompt = _a1_prompt(target, bank, shots=shots, control=control)
                expected = _expected_a1(target)
                a1_cases.append(
                    {
                        **target,
                        "stage": "A1",
                        "case_id": f"a1-{target['target_id']}-s{shots:02d}-{control}",
                        "shots": shots,
                        "control": control,
                        "prompt": prompt,
                        "expected_text": expected,
                        "expected_steps": list(range(0, int(target["length"]) + 1)),
                    }
                )

    a2_cases: list[dict[str, Any]] = []
    for target in targets:
        for condition in A2_CONDITIONS:
            prompt = _a2_prompt(target, bank, condition=condition)
            expected = _expected_a2(target, condition)
            a2_cases.append(
                {
                    **target,
                    "stage": "A2",
                    "case_id": f"a2-{target['target_id']}-{condition}",
                    "shots": 64,
                    "condition": condition,
                    "prompt": prompt,
                    "expected_text": expected,
                    "expected_steps": list(
                        range(int(target["edit_step"]) + 1, int(target["length"]) + 1)
                    ),
                }
            )
    return {
        "targets": targets,
        "demonstrations": bank,
        "A0_EXTENSION": _a0_marker_only_cases(),
        "A1": a1_cases,
        "A2": a2_cases,
    }


def _case_binding(case: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "case_id",
        "target_id",
        "stage",
        "length",
        "semantic_answer",
        "edit_stratum",
        "edit_step",
        "shots",
    )
    binding = {field: case[field] for field in fields}
    if case["stage"] == "A1":
        binding["control"] = case["control"]
    else:
        binding["condition"] = case["condition"]
    prompt = str(case["prompt"])
    expected = str(case["expected_text"])
    binding.update(
        {
            "bits": list(case["bits"]),
            "expected_steps": list(case["expected_steps"]),
            "prompt_utf8_bytes": len(prompt.encode("utf-8")),
            "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
            "expected_text_sha256": sha256_bytes(expected.encode("utf-8")),
        }
    )
    return binding


def compile_manifests() -> tuple[dict[str, Any], dict[str, Any]]:
    compiled = compile_cases()
    target_hash = sha256_json(compiled["targets"])
    demo_hash = sha256_json(compiled["demonstrations"])
    shared = {
        "schema_version": SCHEMA_VERSION,
        "compiler_seed": SEED,
        "target_ledger_sha256": target_hash,
        "demonstration_bank_sha256": demo_hash,
        "target_count": len(compiled["targets"]),
        "lengths": list(LENGTHS),
        "edit_strata": list(EDIT_STRATA),
        "prompt_rendering": {
            "chat_roles": ["user"],
            "add_generation_prompt": True,
            "enable_thinking": False,
            "generated_reasoning": False,
            "strict_output": True,
        },
    }
    a1 = {
        **shared,
        "stage": "A1",
        "shot_counts": list(SHOT_COUNTS),
        "controls": list(A1_CONTROLS),
        "case_count": len(compiled["A1"]),
        "cases": [_case_binding(case) for case in compiled["A1"]],
    }
    a2 = {
        **shared,
        "stage": "A2",
        "shot_count": 64,
        "conditions": list(A2_CONDITIONS),
        "intervention_marker": INTERVENTION_MARKER,
        "case_count": len(compiled["A2"]),
        "cases": [_case_binding(case) for case in compiled["A2"]],
    }
    a1["manifest_sha256"] = sha256_json(a1)
    a2["manifest_sha256"] = sha256_json(a2)
    return a1, a2


def compile_a0_extension_manifest() -> dict[str, Any]:
    cases = compile_cases()["A0_EXTENSION"]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "compiler_seed": SEED,
        "stage": "A0_EXTENSION",
        "condition": "marker_only_no_overwrite",
        "matched_review1_condition": "E",
        "case_count": len(cases),
        "intervention_marker": A0_INTERVENTION_MARKER,
        "cases": [
            {
                **_case_binding(case),
                "matched_to_case_id": case["matched_to_case_id"],
            }
            for case in cases
        ],
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    return manifest


def validate_manifests(
    a1: Mapping[str, Any],
    a2: Mapping[str, Any],
    a0_extension: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    expected_a1, expected_a2 = compile_manifests()
    if dict(a1) != expected_a1:
        raise ValueError("A1 manifest differs from deterministic compiler output")
    if dict(a2) != expected_a2:
        raise ValueError("A2 manifest differs from deterministic compiler output")
    expected_a0_extension = compile_a0_extension_manifest()
    if a0_extension is not None and dict(a0_extension) != expected_a0_extension:
        raise ValueError("A0 extension manifest differs from deterministic compiler output")

    compiled = compile_cases()
    case_ids = [case["case_id"] for stage in ("A1", "A2") for case in compiled[stage]]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("duplicate Phase 2 case_id")
    prompts = [str(case["prompt"]) for stage in ("A1", "A2") for case in compiled[stage]]
    if any(" XOR " in prompt or "update rule" in prompt.lower() for prompt in prompts):
        raise ValueError("A1/A2 prompt leaks the explicit XOR rule")

    target_bits = {tuple(row["bits"]) for row in compiled["targets"] if row["length"] == 8}
    demo_bits = {tuple(row["bits"]) for row in compiled["demonstrations"]}
    if target_bits & demo_bits:
        raise ValueError("rendered demonstration and target instances overlap")

    for shots in SHOT_COUNTS:
        counts = Counter(
            int(row["semantic_answer"]) for row in compiled["demonstrations"][:shots]
        )
        if counts != Counter({0: shots // 2, 1: shots // 2}):
            raise ValueError(f"demonstration label imbalance at {shots} shots")

    a1_groups = Counter(
        (int(case["length"]), int(case["semantic_answer"]), str(case["edit_stratum"]))
        for case in compiled["A1"]
    )
    if set(a1_groups.values()) != {len(SHOT_COUNTS) * len(A1_CONTROLS)}:
        raise ValueError("A1 target strata are imbalanced")
    a2_groups = Counter(
        (int(case["length"]), int(case["semantic_answer"]), str(case["edit_stratum"]))
        for case in compiled["A2"]
    )
    if set(a2_groups.values()) != {len(A2_CONDITIONS)}:
        raise ValueError("A2 target strata are imbalanced")

    for case in compiled["A2"]:
        marker_present = INTERVENTION_MARKER in str(case["prompt"])
        if marker_present != (case["condition"] in {"edit", "marker_only"}):
            raise ValueError(f"intervention marker mismatch: {case['case_id']}")
        if case["condition"] == "edit":
            base = tuple(int(value) for value in case["base_trace"])
            edited = tuple(int(value) for value in case["edited_trace"])
            step = int(case["edit_step"])
            if not all(edited[index] == (base[index] ^ 1) for index in range(step, len(base))):
                raise ValueError(f"counterfactual intervention property failed: {case['case_id']}")

    review1_e_by_base = {
        str(row["base_id"]): row
        for row in compile_ledger()["cases"]
        if row["condition"] == "E"
    }
    for case in compiled["A0_EXTENSION"]:
        matched = review1_e_by_base[str(case["base_id"])]
        if len(str(case["prompt"]).encode("utf-8")) != len(
            str(matched["prompt"]).encode("utf-8")
        ):
            raise ValueError(f"A0 marker-only prompt is not byte-length matched: {case['case_id']}")
        if A0_INTERVENTION_MARKER not in str(case["prompt"]):
            raise ValueError(f"A0 marker-only control lacks the frozen marker: {case['case_id']}")

    if PROJECTED_FORWARD_MAXIMUM > FORWARD_CEILING:
        raise ValueError("forward forecast exceeds ceiling")
    return {
        "status": "PASS_MODEL_FREE_PHASE2_MANIFEST_VALIDATION",
        "a1_case_count": len(compiled["A1"]),
        "a2_case_count": len(compiled["A2"]),
        "a0_marker_only_case_count": len(compiled["A0_EXTENSION"]),
        "target_count": len(compiled["targets"]),
        "demonstration_count": len(compiled["demonstrations"]),
        "projected_forward_maximum": PROJECTED_FORWARD_MAXIMUM,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    a1, a2 = compile_manifests()
    a0_extension = compile_a0_extension_manifest()
    validate_manifests(a1, a2, a0_extension)
    write_json(args.out_dir / "track_a_a0_extension_manifest.json", a0_extension)
    write_json(args.out_dir / "track_a_a1_manifest.json", a1)
    write_json(args.out_dir / "track_a_a2_manifest.json", a2)
    print(
        {
            "status": "PASS_MODEL_FREE_PHASE2_COMPILE",
            "a1_cases": a1["case_count"],
            "a2_cases": a2["case_count"],
            "a0_marker_only_cases": a0_extension["case_count"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
