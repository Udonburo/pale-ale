"""Compile the model-free Track A A0 development ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .oracle import edited_trace, parity_trace
from .render_register_cases import expected_text, render_case

SCHEMA_VERSION = "gate13_candidate_register_case_ledger_v0.1.1"
DEFAULT_SEED = "gate13_candidate_register_cases_v0.1.1"
LENGTHS = (4, 8, 12)
CONDITIONS = ("D", "S", "O", "F", "C", "E")
EDIT_STRATA = ("early", "middle", "late")
REPLICATES_PER_LABEL_EDIT = 2

A1_RESERVED_FORWARDS = 3 * 2 * 24
A2_RESERVED_FORWARDS = 24 * 2
TIMING_RESERVED_FORWARDS = 48
TOTAL_FORWARD_CEILING = 600


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def seed_int(seed: str, *, length: int, label: int) -> int:
    payload = f"{seed}|length={length}|label={label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def all_bit_sequences(length: int, label: int) -> List[Tuple[int, ...]]:
    values: List[Tuple[int, ...]] = []
    for number in range(1 << length):
        bits = tuple((number >> shift) & 1 for shift in reversed(range(length)))
        if parity_trace(bits)[-1] == label:
            values.append(bits)
    return values


def selected_sequences(
    *,
    length: int,
    label: int,
    count: int,
    seed: str,
) -> List[Tuple[int, ...]]:
    candidates = all_bit_sequences(length, label)
    rng = random.Random(seed_int(seed, length=length, label=label))
    rng.shuffle(candidates)
    return sorted(candidates[:count])


def edit_steps(length: int) -> Dict[str, int]:
    return {
        "early": 1,
        "middle": length // 2,
        "late": length - 1,
    }


def base_cases(seed: str = DEFAULT_SEED) -> List[Dict[str, Any]]:
    bases: List[Dict[str, Any]] = []
    per_label = len(EDIT_STRATA) * REPLICATES_PER_LABEL_EDIT
    for length in LENGTHS:
        steps = edit_steps(length)
        for label in (0, 1):
            sequences = selected_sequences(
                length=length,
                label=label,
                count=per_label,
                seed=seed,
            )
            cursor = 0
            for edit_stratum in EDIT_STRATA:
                for replicate in range(REPLICATES_PER_LABEL_EDIT):
                    bits = sequences[cursor]
                    cursor += 1
                    base_id = (
                        f"a0-l{length:02d}-y{label}-{edit_stratum}-r{replicate}"
                    )
                    base = parity_trace(bits)
                    edited = edited_trace(bits, steps[edit_stratum])
                    bases.append(
                        {
                            "base_id": base_id,
                            "length": length,
                            "semantic_answer": label,
                            "edit_stratum": edit_stratum,
                            "edit_step": steps[edit_stratum],
                            "replicate": replicate,
                            "bits": list(bits),
                            "base_trace": list(base),
                            "edited_trace": list(edited),
                        }
                    )
    return bases


def compile_ledger(seed: str = DEFAULT_SEED) -> Dict[str, Any]:
    cases: List[Dict[str, Any]] = []
    for base in base_cases(seed):
        for condition in CONDITIONS:
            case = {
                **base,
                "case_id": f"{base['base_id']}-{condition}",
                "condition": condition,
                "split": "development",
                "generated_reasoning": False,
                "model_forward_authorized": False,
            }
            prompt = render_case(case)
            case["prompt"] = prompt
            case["prompt_utf8_bytes"] = len(prompt.encode("utf-8"))
            case["prompt_character_count"] = len(prompt)
            case["expected_text"] = expected_text(
                condition,
                tuple(int(value) for value in case["bits"]),
                int(case["edit_step"]),
            )
            case["expected_final_answer"] = (
                int(case["edited_trace"][-1])
                if condition == "E"
                else int(case["base_trace"][-1])
            )
            case["token_count_status"] = "PENDING_EXACT_TOKENIZER_BINDING"
            cases.append(case)

    a0_forwards = len(cases)
    projected_max = (
        a0_forwards
        + A1_RESERVED_FORWARDS
        + A2_RESERVED_FORWARDS
        + TIMING_RESERVED_FORWARDS
    )
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "authorization": {
            "model_forward": False,
            "activation_extraction": False,
            "track_c": False,
        },
        "design": {
            "lengths": list(LENGTHS),
            "conditions": list(CONDITIONS),
            "edit_strata": list(EDIT_STRATA),
            "replicates_per_label_edit": REPLICATES_PER_LABEL_EDIT,
        },
        "forward_forecast": {
            "a0_compiled": a0_forwards,
            "a1_conditional_reserved": A1_RESERVED_FORWARDS,
            "a2_conditional_reserved": A2_RESERVED_FORWARDS,
            "timing_reserved": TIMING_RESERVED_FORWARDS,
            "projected_maximum": projected_max,
            "ceiling": TOTAL_FORWARD_CEILING,
            "gpu_time_status": "PENDING_RUNTIME_AND_EXACT_TOKEN_BINDING",
        },
        "cases": cases,
    }
    payload["ledger_sha256"] = sha256_json(
        {key: value for key, value in payload.items() if key != "ledger_sha256"}
    )
    return payload


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = compile_ledger(args.seed)
    write_json(args.out, payload)
    print(
        json.dumps(
            {
                "status": "PASS_MODEL_FREE_COMPILE",
                "case_count": len(payload["cases"]),
                "ledger_sha256": payload["ledger_sha256"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

