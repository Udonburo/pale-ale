"""Deterministic, model-free prompt renderers for Track A A0 conditions."""

from __future__ import annotations

from typing import Mapping, Sequence

from .oracle import edited_trace, parity_trace

INTERVENTION_MARKER = (
    "An external intervention now overwrites the current register. "
    "Treat the edited value as authoritative and continue from it; "
    "do not repair the earlier trace."
)

RULE_TEXT = (
    "Use a running binary register. Start with r0 = 0. "
    "For each input bit xt, update rt = r(t-1) XOR xt."
)

OUTPUT_CONTRACT = (
    "Use only the requested register lines and final answer. "
    "Do not add explanations."
)


def bits_text(bits: Sequence[int]) -> str:
    return " ".join(str(int(bit)) for bit in bits)


def trace_lines(trace: Sequence[int], *, start: int = 0) -> str:
    return "\n".join(
        f"r{index} = {int(trace[index])}"
        for index in range(start, len(trace))
    )


def _header(bits: Sequence[int]) -> str:
    return f"{RULE_TEXT}\n\nInput bits:\n{bits_text(bits)}\n"


def render_direct(bits: Sequence[int]) -> str:
    return (
        "Compute the XOR parity of the input bits.\n\n"
        f"Input bits:\n{bits_text(bits)}\n\n"
        "Output exactly one bit, 0 or 1.\n\nANSWER:"
    )


def render_structured(bits: Sequence[int]) -> str:
    return (
        _header(bits)
        + "\nWrite the complete register trace from r0 through rn, then answer = rn.\n"
        + OUTPUT_CONTRACT
        + "\n\nTRACE:"
    )


def render_oracle_prefix(bits: Sequence[int], edit_step: int) -> str:
    trace = parity_trace(bits)
    return (
        _header(bits)
        + f"\nThe correct trace is supplied through step {edit_step}:\n"
        + trace_lines(trace[: edit_step + 1])
        + f"\n\nContinue with r{edit_step + 1} through r{len(bits)}, then answer = r{len(bits)}.\n"
        + OUTPUT_CONTRACT
        + "\n\nCONTINUE:"
    )


def render_filler(bits: Sequence[int]) -> str:
    filler = "\n".join(
        f"z{index} = {'A' if index % 2 == 0 else 'B'}"
        for index in range(len(bits) + 1)
    )
    return (
        "Compute the XOR parity of the input bits. The reference ledger below "
        "is noncomputational filler and contains no register state.\n\n"
        f"Input bits:\n{bits_text(bits)}\n\n"
        f"Reference ledger:\n{filler}\n\n"
        "Output exactly one bit, 0 or 1.\n\nANSWER:"
    )


def render_corrupted(bits: Sequence[int], edit_step: int) -> str:
    trace = list(parity_trace(bits))
    trace[edit_step] ^= 1
    return (
        _header(bits)
        + f"\nA candidate trace is supplied through step {edit_step}. "
        "No external intervention is declared; evaluate it under the stated rule.\n"
        + trace_lines(trace[: edit_step + 1])
        + "\n\nReturn the true final parity as one bit, 0 or 1.\n\nANSWER:"
    )


def render_visible_edit(bits: Sequence[int], edit_step: int) -> str:
    base = parity_trace(bits)
    edited = edited_trace(bits, edit_step)
    prefix = list(base[:edit_step]) + [edited[edit_step]]
    return (
        _header(bits)
        + f"\nThe base trace was correct through step {edit_step - 1}:\n"
        + trace_lines(base[:edit_step])
        + f"\n\n{INTERVENTION_MARKER}\n"
        + f"r{edit_step} = {edited[edit_step]}\n\n"
        + f"Continue with r{edit_step + 1} through r{len(bits)}, then answer = r{len(bits)}.\n"
        + OUTPUT_CONTRACT
        + "\n\nCONTINUE:"
    )


def expected_text(
    condition: str,
    bits: Sequence[int],
    edit_step: int,
) -> str:
    base = parity_trace(bits)
    if condition == "D" or condition == "F" or condition == "C":
        return str(base[-1])
    if condition == "S":
        return trace_lines(base) + f"\nanswer = {base[-1]}"
    if condition == "O":
        return trace_lines(base, start=edit_step + 1) + f"\nanswer = {base[-1]}"
    if condition == "E":
        edited = edited_trace(bits, edit_step)
        return trace_lines(edited, start=edit_step + 1) + f"\nanswer = {edited[-1]}"
    raise ValueError(f"unknown condition: {condition}")


def render_case(case: Mapping[str, object]) -> str:
    bits = tuple(int(value) for value in case["bits"])  # type: ignore[index]
    edit_step = int(case["edit_step"])
    condition = str(case["condition"])
    renderers = {
        "D": lambda: render_direct(bits),
        "S": lambda: render_structured(bits),
        "O": lambda: render_oracle_prefix(bits, edit_step),
        "F": lambda: render_filler(bits),
        "C": lambda: render_corrupted(bits, edit_step),
        "E": lambda: render_visible_edit(bits, edit_step),
    }
    try:
        return renderers[condition]()
    except KeyError as exc:
        raise ValueError(f"unknown condition: {condition}") from exc

