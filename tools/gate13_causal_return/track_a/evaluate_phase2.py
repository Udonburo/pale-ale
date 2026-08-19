"""Frozen Track A A0/A1/A2 metrics and conditional progression gates."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from .oracle import edited_trace, paired_selectivity_exact, parity_trace, transition_accuracy
from .parse_phase2_output import Phase2OutputParseError, parse_phase2_output
from .parse_register_output import OutputParseError, parse_register_output


def _record_map(
    cases: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]
) -> dict[str, Mapping[str, Any]]:
    case_ids = {str(case["case_id"]) for case in cases}
    record_ids = [str(record.get("case_id") or "") for record in records]
    if any(not case_id for case_id in record_ids):
        raise ValueError("result record has an empty case_id")
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("duplicate result case_id")
    if set(record_ids) != case_ids:
        missing = sorted(case_ids - set(record_ids))
        extra = sorted(set(record_ids) - case_ids)
        raise ValueError(f"incomplete result set; missing={missing[:3]} extra={extra[:3]}")
    return {str(record["case_id"]): record for record in records}


def _mean(flags: Sequence[bool]) -> float:
    return sum(bool(flag) for flag in flags) / len(flags) if flags else 0.0


def evaluate_a0(
    cases: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    record_by_id = _record_map(cases, records)
    grouped: dict[str, dict[str, tuple[Mapping[str, Any], Any | None]]] = defaultdict(dict)
    malformed = 0
    for case in cases:
        record = record_by_id[str(case["case_id"])]
        try:
            if str(case["condition"]) == "N":
                parsed = parse_phase2_output(case, str(record.get("response") or ""))
            else:
                parsed = parse_register_output(case, str(record.get("response") or ""))
        except (OutputParseError, Phase2OutputParseError):
            parsed = None
            malformed += 1
        grouped[str(case["base_id"])][str(case["condition"])] = (case, parsed)

    structured_steps: list[float] = []
    structured_final: list[bool] = []
    filler_final: list[bool] = []
    corrupted_final: list[bool] = []
    oracle_prefix: list[bool] = []
    oracle_cf: list[bool] = []
    edited_final: list[bool] = []
    marker_only_tail: list[bool] = []
    pair_flags: list[bool] = []
    pair_ineligible = 0
    for rows in grouped.values():
        representative = rows["S"][0]
        bits = tuple(int(bit) for bit in representative["bits"])
        step = int(representative["edit_step"])
        base_oracle = parity_trace(bits)
        edited_oracle = edited_trace(bits, step)

        s_parsed = rows["S"][1]
        if s_parsed is None:
            structured_steps.append(0.0)
            structured_final.append(False)
        else:
            structured_steps.append(transition_accuracy(s_parsed.trace_prediction, bits))
            structured_final.append(s_parsed.final_prediction == base_oracle[-1])

        for condition, target in (("F", filler_final), ("C", corrupted_final)):
            parsed = rows[condition][1]
            target.append(parsed is not None and parsed.final_prediction == base_oracle[-1])

        o_parsed = rows["O"][1]
        oracle_prefix.append(
            o_parsed is not None and tuple(o_parsed.trace_prediction) == base_oracle
        )
        e_parsed = rows["E"][1]
        oracle_cf.append(
            e_parsed is not None
            and tuple(e_parsed.trace_prediction)[step:] == edited_oracle[step:]
        )
        edited_final.append(
            e_parsed is not None and e_parsed.final_prediction == edited_oracle[-1]
        )
        n_parsed = rows["N"][1]
        marker_only_tail.append(
            n_parsed is not None
            and tuple(n_parsed.values) == base_oracle[step + 1 :]
        )
        if s_parsed is None or e_parsed is None:
            pair_ineligible += 1
        else:
            pair = paired_selectivity_exact(
                s_parsed.trace_prediction,
                e_parsed.trace_prediction,
                bits,
                step,
            )
            if pair is None:
                pair_ineligible += 1
            else:
                pair_flags.append(pair)

    metrics = {
        "A_step": sum(structured_steps) / len(structured_steps),
        "A_final": _mean(structured_final),
        "filler_final": _mean(filler_final),
        "corrupted_final": _mean(corrupted_final),
        "oracle_prefix_continuation": _mean(oracle_prefix),
        "A_CF_oracle": _mean(oracle_cf),
        "A_final_CF": _mean(edited_final),
        "marker_only_no_overwrite_accuracy": _mean(marker_only_tail),
        "S_pair": _mean(pair_flags) if pair_flags else None,
        "S_pair_eligible": len(pair_flags),
        "S_pair_ineligible": pair_ineligible,
        "malformed_output_count": malformed,
    }
    gates = {
        "structured_step_accuracy": metrics["A_step"] >= 0.80,
        "structured_final_accuracy": metrics["A_final"] >= 0.80,
        "structured_minus_filler_final": metrics["A_final"] - metrics["filler_final"] >= 0.20,
        "structured_minus_corrupted_final": metrics["A_final"] - metrics["corrupted_final"] >= 0.20,
        "oracle_prefix_continuation": metrics["oracle_prefix_continuation"] >= 0.80,
        "oracle_visible_cf_accuracy": metrics["A_CF_oracle"] >= 0.75,
        "oracle_edited_final_accuracy": metrics["A_final_CF"] >= 0.75,
    }
    return {"status": "PASS" if all(gates.values()) else "FAIL", "metrics": metrics, "gates": gates}


def evaluate_a1(
    cases: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    record_by_id = _record_map(cases, records)
    cell_steps: dict[tuple[int, str], list[float]] = defaultdict(list)
    cell_final: dict[tuple[int, str], list[bool]] = defaultdict(list)
    cell_malformed: dict[tuple[int, str], int] = defaultdict(int)
    for case in cases:
        key = (int(case["shots"]), str(case["control"]))
        record = record_by_id[str(case["case_id"])]
        try:
            parsed = parse_phase2_output(case, str(record.get("response") or ""))
        except Phase2OutputParseError:
            cell_steps[key].append(0.0)
            cell_final[key].append(False)
            cell_malformed[key] += 1
            continue
        bits = tuple(int(bit) for bit in case["bits"])
        cell_steps[key].append(transition_accuracy(parsed.values, bits))
        cell_final[key].append(parsed.final_prediction == parity_trace(bits)[-1])

    cells: dict[str, Any] = {}
    passing_shots: list[int] = []
    for shots in (4, 16, 64):
        for control in ("correct", "corrupted", "shuffled"):
            key = (shots, control)
            cells[f"shots_{shots}_{control}"] = {
                "A_step": sum(cell_steps[key]) / len(cell_steps[key]),
                "A_final": _mean(cell_final[key]),
                "malformed_output_count": cell_malformed[key],
            }
        correct = cells[f"shots_{shots}_correct"]
        corrupted = cells[f"shots_{shots}_corrupted"]
        shuffled = cells[f"shots_{shots}_shuffled"]
        gates = {
            "correct_A_step": correct["A_step"] >= 0.80,
            "correct_A_final": correct["A_final"] >= 0.80,
            "correct_minus_corrupted_final": correct["A_final"] - corrupted["A_final"] >= 0.20,
            "correct_minus_shuffled_final": correct["A_final"] - shuffled["A_final"] >= 0.20,
        }
        cells[f"shots_{shots}_decision"] = {
            "status": "PASS" if all(gates.values()) else "FAIL",
            "gates": gates,
        }
        if all(gates.values()):
            passing_shots.append(shots)

    return {
        "status": "PASS" if passing_shots else "FAIL",
        "formation_shot": min(passing_shots) if passing_shots else None,
        "cells": cells,
    }


def evaluate_a2(
    cases: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    record_by_id = _record_map(cases, records)
    grouped: dict[str, dict[str, tuple[Mapping[str, Any], Any | None]]] = defaultdict(dict)
    malformed = 0
    for case in cases:
        try:
            parsed = parse_phase2_output(
                case, str(record_by_id[str(case["case_id"])].get("response") or "")
            )
        except Phase2OutputParseError:
            parsed = None
            malformed += 1
        grouped[str(case["target_id"])][str(case["condition"])] = (case, parsed)

    base_tail: list[bool] = []
    base_final: list[bool] = []
    edit_tail: list[bool] = []
    edit_final: list[bool] = []
    marker_tail: list[bool] = []
    corrupt_edited: list[bool] = []
    filler_edited: list[bool] = []
    pair_flags: list[bool] = []
    pair_ineligible = 0
    for rows in grouped.values():
        representative = rows["base"][0]
        bits = tuple(int(bit) for bit in representative["bits"])
        step = int(representative["edit_step"])
        base = parity_trace(bits)
        edited = edited_trace(bits, step)
        base_expected = base[step + 1 :]
        edited_expected = edited[step + 1 :]

        def values(condition: str) -> tuple[int, ...] | None:
            parsed = rows[condition][1]
            return None if parsed is None else tuple(parsed.values)

        base_values = values("base")
        edit_values = values("edit")
        marker_values = values("marker_only")
        corrupt_values = values("undeclared_corrupt")
        filler_values = values("filler")
        base_tail.append(base_values == base_expected)
        base_final.append(base_values is not None and base_values[-1] == base[-1])
        edit_tail.append(edit_values == edited_expected)
        edit_final.append(edit_values is not None and edit_values[-1] == edited[-1])
        marker_tail.append(marker_values == base_expected)
        corrupt_edited.append(corrupt_values == edited_expected)
        filler_edited.append(filler_values == edited_expected)

        if base_values != base_expected or edit_values is None:
            pair_ineligible += 1
        else:
            base_full = base[: step + 1] + base_values
            edit_full = edited[: step + 1] + edit_values
            pair = paired_selectivity_exact(base_full, edit_full, bits, step)
            if pair is None:
                pair_ineligible += 1
            else:
                pair_flags.append(pair)

    metrics = {
        "base_oracle_tail_accuracy": _mean(base_tail),
        "base_final_accuracy": _mean(base_final),
        "A_CF_oracle": _mean(edit_tail),
        "A_final_CF": _mean(edit_final),
        "marker_only_base_tail_accuracy": _mean(marker_tail),
        "undeclared_corrupt_edited_oracle_rate": _mean(corrupt_edited),
        "filler_edited_oracle_rate": _mean(filler_edited),
        "S_pair": _mean(pair_flags) if pair_flags else None,
        "S_pair_eligible": len(pair_flags),
        "S_pair_ineligible": pair_ineligible,
        "malformed_output_count": malformed,
    }
    gates = {
        "base_oracle_tail_accuracy": metrics["base_oracle_tail_accuracy"] >= 0.80,
        "base_final_accuracy": metrics["base_final_accuracy"] >= 0.80,
        "oracle_visible_cf_accuracy": metrics["A_CF_oracle"] >= 0.75,
        "oracle_edited_final_accuracy": metrics["A_final_CF"] >= 0.75,
        "marker_only_base_tail_accuracy": metrics["marker_only_base_tail_accuracy"] >= 0.75,
        "undeclared_corrupt_does_not_match_edit": metrics["undeclared_corrupt_edited_oracle_rate"] < 0.50,
        "filler_does_not_match_edit": metrics["filler_edited_oracle_rate"] < 0.50,
    }
    return {"status": "PASS" if all(gates.values()) else "FAIL", "metrics": metrics, "gates": gates}
