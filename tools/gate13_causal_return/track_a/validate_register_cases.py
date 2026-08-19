"""Validate the Track A model-free development ledger."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .compile_register_cases import (
    CONDITIONS,
    EDIT_STRATA,
    LENGTHS,
    REPLICATES_PER_LABEL_EDIT,
    SCHEMA_VERSION,
    sha256_json,
)
from .oracle import edited_trace, parity_trace
from .render_register_cases import INTERVENTION_MARKER, expected_text, render_case


class LedgerValidationError(ValueError):
    """Raised when the compiled ledger violates a frozen model-free invariant."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LedgerValidationError(message)


def validate_ledger(payload: Mapping[str, Any]) -> Dict[str, Any]:
    require(payload.get("schema_version") == SCHEMA_VERSION, "schema_version mismatch")
    cases = list(payload.get("cases") or [])
    expected_base_count = len(LENGTHS) * 2 * len(EDIT_STRATA) * REPLICATES_PER_LABEL_EDIT
    expected_case_count = expected_base_count * len(CONDITIONS)
    require(len(cases) == expected_case_count, "unexpected case count")

    expected_hash = sha256_json(
        {key: value for key, value in payload.items() if key != "ledger_sha256"}
    )
    require(payload.get("ledger_sha256") == expected_hash, "ledger hash mismatch")

    case_ids = [str(case.get("case_id") or "") for case in cases]
    require(len(set(case_ids)) == len(case_ids), "duplicate or empty case_id")

    grouped: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    base_bits_by_length: Dict[int, set[tuple[int, ...]]] = defaultdict(set)
    base_balance: Counter[tuple[int, int, str]] = Counter()

    for case in cases:
        base_id = str(case["base_id"])
        grouped[base_id].append(case)
        bits = tuple(int(value) for value in case["bits"])
        base = parity_trace(bits)
        edited = edited_trace(bits, int(case["edit_step"]))
        require(list(base) == list(case["base_trace"]), f"base trace mismatch: {case['case_id']}")
        require(
            list(edited) == list(case["edited_trace"]),
            f"edited trace mismatch: {case['case_id']}",
        )
        require(base[-1] == int(case["semantic_answer"]), "semantic answer mismatch")
        require(render_case(case) == case["prompt"], f"prompt mismatch: {case['case_id']}")
        require(
            expected_text(str(case["condition"]), bits, int(case["edit_step"]))
            == case["expected_text"],
            f"expected text mismatch: {case['case_id']}",
        )
        require(
            len(str(case["prompt"]).encode("utf-8")) == int(case["prompt_utf8_bytes"]),
            "prompt byte count mismatch",
        )
        require(
            len(str(case["prompt"])) == int(case["prompt_character_count"]),
            "prompt character count mismatch",
        )
        condition = str(case["condition"])
        if condition == "E":
            require(INTERVENTION_MARKER in str(case["prompt"]), "E marker missing")
            require(
                int(case["expected_final_answer"]) == edited[-1],
                "E final answer mismatch",
            )
        else:
            require(
                int(case["expected_final_answer"]) == base[-1],
                "base final answer mismatch",
            )
        if condition == "C":
            require(
                INTERVENTION_MARKER not in str(case["prompt"]),
                "corrupted control contains intervention marker",
            )
        require(case.get("model_forward_authorized") is False, "forward authorization leak")
        require(case.get("generated_reasoning") is False, "reasoning authorization leak")

    for base_id, rows in grouped.items():
        require(len(rows) == len(CONDITIONS), f"condition count mismatch: {base_id}")
        require(
            {str(row["condition"]) for row in rows} == set(CONDITIONS),
            f"condition set mismatch: {base_id}",
        )
        representative = rows[0]
        length = int(representative["length"])
        bits = tuple(int(value) for value in representative["bits"])
        base_bits_by_length[length].add(bits)
        base_balance[
            (
                length,
                int(representative["semantic_answer"]),
                str(representative["edit_stratum"]),
            )
        ] += 1

    for length in LENGTHS:
        require(len(base_bits_by_length[length]) == 12, f"duplicate base bits at length {length}")
        for label in (0, 1):
            for edit_stratum in EDIT_STRATA:
                require(
                    base_balance[(length, label, edit_stratum)]
                    == REPLICATES_PER_LABEL_EDIT,
                    f"base balance mismatch: {length}/{label}/{edit_stratum}",
                )

    forecast = dict(payload.get("forward_forecast") or {})
    require(
        int(forecast.get("projected_maximum") or 0)
        <= int(forecast.get("ceiling") or 0),
        "forward forecast exceeds ceiling",
    )

    condition_counts = Counter(str(case["condition"]) for case in cases)
    return {
        "schema_version": "gate13_candidate_register_validation_v0.1.1",
        "status": "PASS_MODEL_FREE_VALIDATION",
        "ledger_sha256": expected_hash,
        "case_count": len(cases),
        "base_case_count": len(grouped),
        "condition_counts": dict(sorted(condition_counts.items())),
        "base_balance": {
            f"length_{length}_y{label}_{edit_stratum}": base_balance[
                (length, label, edit_stratum)
            ]
            for length in LENGTHS
            for label in (0, 1)
            for edit_stratum in EDIT_STRATA
        },
        "unique_bits_by_length": {
            str(length): len(base_bits_by_length[length]) for length in LENGTHS
        },
        "forward_forecast": forecast,
        "exact_token_count_status": "PENDING_EXACT_TOKENIZER_BINDING",
        "gpu_time_status": "PENDING_RUNTIME_AND_EXACT_TOKEN_BINDING",
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = json.loads(args.ledger.read_text(encoding="utf-8"))
    report = validate_ledger(payload)
    write_json(args.out, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

