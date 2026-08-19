"""Strict A1/A2 output parser for prospectively frozen case contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping


_REGISTER = re.compile(r"r(?P<step>0|[1-9][0-9]*) = (?P<value>[01])")
_ANSWER = re.compile(r"answer = (?P<value>[01])")


class Phase2OutputParseError(ValueError):
    """Raised when an A1/A2 response violates its byte-frozen contract."""


@dataclass(frozen=True)
class ParsedPhase2Output:
    values: tuple[int, ...]
    final_prediction: int


def parse_phase2_output(case: Mapping[str, object], text: str) -> ParsedPhase2Output:
    normalized = text.strip()
    if not normalized:
        raise Phase2OutputParseError("output is empty")
    lines = normalized.splitlines()
    if any(not line or line != line.strip() for line in lines):
        raise Phase2OutputParseError("blank lines or surrounding whitespace are forbidden")
    expected_steps = [int(step) for step in case["expected_steps"]]  # type: ignore[index]
    if len(lines) != len(expected_steps) + 1:
        raise Phase2OutputParseError("wrong number of output lines")
    values: list[int] = []
    for line, step in zip(lines[:-1], expected_steps):
        match = _REGISTER.fullmatch(line)
        if match is None:
            raise Phase2OutputParseError(f"malformed register line: {line!r}")
        if int(match.group("step")) != step:
            raise Phase2OutputParseError("register steps are missing, duplicated, or reordered")
        values.append(int(match.group("value")))
    answer_match = _ANSWER.fullmatch(lines[-1])
    if answer_match is None:
        raise Phase2OutputParseError("missing or malformed final answer")
    answer = int(answer_match.group("value"))
    if not values or values[-1] != answer:
        raise Phase2OutputParseError("final answer differs from the final register")
    return ParsedPhase2Output(tuple(values), answer)
