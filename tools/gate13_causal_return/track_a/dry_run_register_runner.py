"""Exercise resume and counterfactual metrics using synthetic oracle records only."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from .oracle import (
    oracle_counterfactual_exact,
    paired_selectivity_exact,
    parity_trace,
)
from .parse_register_output import parse_register_output
from .validate_register_cases import validate_ledger


class SyntheticRecordError(ValueError):
    """Raised when a synthetic result record violates identity invariants."""


def oracle_record(case: Mapping[str, Any]) -> Dict[str, Any]:
    parsed = parse_register_output(case, str(case["expected_text"]))
    record: Dict[str, Any] = {
        "case_id": str(case["case_id"]),
        "base_id": str(case["base_id"]),
        "condition": parsed.condition,
        "final_prediction": parsed.final_prediction,
        "source": "synthetic_oracle_via_strict_parser_no_model",
    }
    if parsed.trace_prediction is not None:
        record["trace_prediction"] = list(parsed.trace_prediction)
    return record


def completed_ids(records: Iterable[Mapping[str, Any]]) -> set[str]:
    ids = [str(record.get("case_id") or "") for record in records]
    if any(not case_id for case_id in ids):
        raise SyntheticRecordError("record has empty case_id")
    if len(set(ids)) != len(ids):
        raise SyntheticRecordError("duplicate completed case_id")
    return set(ids)


def resume_missing(
    cases: Sequence[Mapping[str, Any]],
    existing_records: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    completed = completed_ids(existing_records)
    case_ids = {str(case["case_id"]) for case in cases}
    if not completed.issubset(case_ids):
        raise SyntheticRecordError("existing records include an unknown case_id")
    return [
        oracle_record(case)
        for case in cases
        if str(case["case_id"]) not in completed
    ]


def evaluate_counterfactuals(
    cases: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    case_by_id = {str(case["case_id"]): case for case in cases}
    record_by_id = {str(record["case_id"]): record for record in records}
    grouped: Dict[str, Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for case_id, case in case_by_id.items():
        grouped[str(case["base_id"])][str(case["condition"])] = record_by_id[case_id]

    oracle_flags: list[bool] = []
    final_flags: list[bool] = []
    pair_flags: list[bool] = []
    ineligible = 0
    for base_id, condition_records in grouped.items():
        base_case = next(
            case for case in cases if str(case["base_id"]) == base_id
        )
        bits = tuple(int(value) for value in base_case["bits"])
        edit_step = int(base_case["edit_step"])
        base_record = condition_records["S"]
        edited_record = condition_records["E"]
        base_prediction = tuple(int(value) for value in base_record["trace_prediction"])
        edited_prediction = tuple(int(value) for value in edited_record["trace_prediction"])
        oracle_flags.append(
            oracle_counterfactual_exact(edited_prediction, bits, edit_step)
        )
        final_flags.append(
            int(edited_record["final_prediction"]) == (parity_trace(bits)[-1] ^ 1)
        )
        pair = paired_selectivity_exact(
            base_prediction,
            edited_prediction,
            bits,
            edit_step,
        )
        if pair is None:
            ineligible += 1
        else:
            pair_flags.append(pair)

    return {
        "oracle_cf_accuracy": sum(oracle_flags) / len(oracle_flags),
        "oracle_final_cf_accuracy": sum(final_flags) / len(final_flags),
        "paired_selectivity": (
            sum(pair_flags) / len(pair_flags) if pair_flags else None
        ),
        "paired_eligible_count": len(pair_flags),
        "paired_ineligible_count": ineligible,
    }


def run_preflight(ledger: Mapping[str, Any]) -> Dict[str, Any]:
    validation = validate_ledger(ledger)
    cases = list(ledger["cases"])
    pending_initial = len(cases)

    partial = [oracle_record(case) for case in cases[::7]]
    partial_ids = completed_ids(partial)
    resumed = resume_missing(cases, partial)
    combined = partial + resumed
    completed = completed_ids(combined)
    if len(completed) != len(cases):
        raise SyntheticRecordError("resume did not complete the ledger")
    second_resume = resume_missing(cases, combined)
    if second_resume:
        raise SyntheticRecordError("full resume was not idempotent")

    duplicate_rejected = False
    try:
        completed_ids(combined + [combined[0]])
    except SyntheticRecordError:
        duplicate_rejected = True
    if not duplicate_rejected:
        raise SyntheticRecordError("duplicate record was not rejected")

    unknown_case_rejected = False
    try:
        resume_missing(cases, [{"case_id": "unknown-case-id"}])
    except SyntheticRecordError:
        unknown_case_rejected = True
    if not unknown_case_rejected:
        raise SyntheticRecordError("unknown case_id was not rejected")

    metrics = evaluate_counterfactuals(cases, combined)
    return {
        "schema_version": "gate13_candidate_register_preflight_v0.1.1",
        "status": "PASS_SYNTHETIC_DRY_RUN",
        "model_forward_count": 0,
        "ledger_sha256": ledger["ledger_sha256"],
        "validation_status": validation["status"],
        "checks": {
            "all_pending_count": pending_initial,
            "partial_completed_count": len(partial_ids),
            "partial_missing_count": len(resumed),
            "full_completed_count": len(completed),
            "second_resume_missing_count": len(second_resume),
            "duplicate_rejected": duplicate_rejected,
            "unknown_case_rejected": unknown_case_rejected,
            "strict_parser_exercised": True,
        },
        "synthetic_oracle_metrics": metrics,
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
    ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
    report = run_preflight(ledger)
    write_json(args.out, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
