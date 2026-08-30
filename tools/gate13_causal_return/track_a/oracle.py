"""Exact phase-indexed XOR-register oracle for Track A."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple


def normalize_bits(bits: Iterable[int]) -> Tuple[int, ...]:
    normalized = tuple(int(bit) for bit in bits)
    if not normalized:
        raise ValueError("bits must be non-empty")
    if any(bit not in (0, 1) for bit in normalized):
        raise ValueError("bits must contain only 0 or 1")
    return normalized


def parity_trace(bits: Iterable[int]) -> Tuple[int, ...]:
    """Return (r_0, ..., r_n) for r_t = r_(t-1) XOR x_t."""
    normalized = normalize_bits(bits)
    trace = [0]
    for bit in normalized:
        trace.append(trace[-1] ^ bit)
    return tuple(trace)


def edited_trace(bits: Iterable[int], edit_step: int) -> Tuple[int, ...]:
    """Return the oracle trace after an external overwrite r_k <- r_k XOR 1."""
    normalized = normalize_bits(bits)
    if edit_step <= 0 or edit_step >= len(normalized):
        raise ValueError("edit_step must satisfy 1 <= edit_step < len(bits)")
    base = parity_trace(normalized)
    result = list(base[:edit_step])
    state = base[edit_step] ^ 1
    result.append(state)
    for step in range(edit_step + 1, len(normalized) + 1):
        state ^= normalized[step - 1]
        result.append(state)
    return tuple(result)


def transition_accuracy(trace: Sequence[int], bits: Iterable[int]) -> float:
    normalized = normalize_bits(bits)
    predicted = tuple(int(value) for value in trace)
    if len(predicted) != len(normalized) + 1:
        raise ValueError("trace length must be len(bits) + 1")
    correct = sum(
        int(predicted[step] == (predicted[step - 1] ^ normalized[step - 1]))
        for step in range(1, len(predicted))
    )
    return correct / len(normalized)


def oracle_counterfactual_exact(
    edited_prediction: Sequence[int],
    bits: Iterable[int],
    edit_step: int,
) -> bool:
    oracle = edited_trace(bits, edit_step)
    predicted = tuple(int(value) for value in edited_prediction)
    return predicted[edit_step:] == oracle[edit_step:]


def paired_selectivity_exact(
    base_prediction: Sequence[int],
    edited_prediction: Sequence[int],
    bits: Iterable[int],
    edit_step: int,
) -> bool | None:
    """Return None when the base tail is not oracle-correct."""
    base = tuple(int(value) for value in base_prediction)
    edited = tuple(int(value) for value in edited_prediction)
    oracle = parity_trace(bits)
    if len(base) != len(oracle) or len(edited) != len(oracle):
        raise ValueError("predicted traces must match oracle length")
    if base[edit_step:] != oracle[edit_step:]:
        return None
    return all(
        edited[index] == (base[index] ^ 1)
        for index in range(edit_step, len(base))
    )

