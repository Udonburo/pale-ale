"""Strict parsers for frozen Track A register-output contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Tuple

from .oracle import edited_trace, parity_trace

_REGISTER_LINE = re.compile(r"r(?P<step>0|[1-9][0-9]*) = (?P<value>[01])")
_ANSWER_LINE = re.compile(r"answer = (?P<value>[01])")
_DIRECT_ANSWER = re.compile(r"[01]")


class OutputParseError(ValueError):
    """Raised when a response violates its exact frozen output contract."""


@dataclass(frozen=True)
class ParsedRegisterOutput:
    """Normalized output; direct-answer conditions have no generated trace."""

    condition: str
    final_prediction: int
    trace_prediction: Tuple[int, ...] | None


def _strict_lines(text: str) -> list[str]:
    normalized = text.strip()
    if not normalized:
        raise OutputParseError("output is empty")
    lines = normalized.splitlines()
    if any(line == "" or line != line.strip() for line in lines):
        raise OutputParseError("blank lines or surrounding line whitespace are forbidden")
    return lines


def _parse_trace_contract(
    text: str,
    *,
    expected_steps: range,
) -> tuple[Tuple[int, ...], int]:
    lines = _strict_lines(text)
    steps = list(expected_steps)
    if len(lines) != len(steps) + 1:
        raise OutputParseError("wrong number of output lines")

    values: list[int] = []
    for line, expected_step in zip(lines[:-1], steps):
        match = _REGISTER_LINE.fullmatch(line)
        if match is None:
            raise OutputParseError(f"malformed register line: {line!r}")
        observed_step = int(match.group("step"))
        if observed_step != expected_step:
            raise OutputParseError(
                f"register step {observed_step} is out of order; expected {expected_step}"
            )
        values.append(int(match.group("value")))

    answer_match = _ANSWER_LINE.fullmatch(lines[-1])
    if answer_match is None:
        raise OutputParseError("missing or malformed final answer line")
    answer = int(answer_match.group("value"))
    if not values or answer != values[-1]:
        raise OutputParseError("final answer does not equal the last generated register")
    return tuple(values), answer


def parse_register_output(
    case: Mapping[str, object],
    text: str,
) -> ParsedRegisterOutput:
    """Parse one response under the exact case-specific output contract.

    The parser accepts no prose, labels beyond the declared contract, omitted
    steps, repeated steps, or reordered steps.  O/E continuation outputs are
    reconstructed into full trajectories using the prompt-visible prefix.
    """

    condition = str(case["condition"])
    bits = tuple(int(value) for value in case["bits"])  # type: ignore[index]
    edit_step = int(case["edit_step"])
    length = len(bits)

    if condition in {"D", "F", "C"}:
        normalized = text.strip()
        if _DIRECT_ANSWER.fullmatch(normalized) is None:
            raise OutputParseError("direct-answer output must be exactly one bit")
        return ParsedRegisterOutput(condition, int(normalized), None)

    if condition == "S":
        generated, answer = _parse_trace_contract(
            text,
            expected_steps=range(0, length + 1),
        )
        return ParsedRegisterOutput(condition, answer, generated)

    if condition not in {"O", "E"}:
        raise OutputParseError(f"unknown condition: {condition}")

    generated, answer = _parse_trace_contract(
        text,
        expected_steps=range(edit_step + 1, length + 1),
    )
    prefix = (
        parity_trace(bits)[: edit_step + 1]
        if condition == "O"
        else edited_trace(bits, edit_step)[: edit_step + 1]
    )
    full_trace = tuple(prefix) + generated
    if len(full_trace) != length + 1:
        raise OutputParseError("reconstructed trace has the wrong length")
    return ParsedRegisterOutput(condition, answer, full_trace)
