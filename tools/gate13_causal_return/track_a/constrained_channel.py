"""Syntax-only finite-state output channel for Track A register responses.

The channel fixes serialization, not semantic values.  Every register and
answer slot independently exposes both binary token branches.  No oracle,
transition rule, prior slot, or expected answer is consulted here.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable, Mapping, Sequence


class ConstrainedChannelError(ValueError):
    """Raised when a syntax-only channel cannot be constructed or followed."""


@dataclass(frozen=True)
class SyntaxComponent:
    """One fixed literal or one unconstrained binary semantic slot."""

    kind: str
    literal: str = ""

    def __post_init__(self) -> None:
        if self.kind not in {"literal", "bit"}:
            raise ConstrainedChannelError(f"unknown syntax component: {self.kind}")
        if self.kind == "literal" and not self.literal:
            raise ConstrainedChannelError("literal components must be non-empty")
        if self.kind == "bit" and self.literal:
            raise ConstrainedChannelError("bit components cannot carry a literal")


@dataclass(frozen=True)
class RegisterSyntax:
    """Case-shape grammar whose semantic slots are independent binary choices."""

    expected_steps: tuple[int, ...]
    direct_answer: bool = False

    def __post_init__(self) -> None:
        if self.direct_answer and self.expected_steps:
            raise ConstrainedChannelError("direct-answer syntax cannot contain register steps")
        if not self.direct_answer:
            if not self.expected_steps:
                raise ConstrainedChannelError("structured syntax requires register steps")
            if tuple(sorted(set(self.expected_steps))) != self.expected_steps:
                raise ConstrainedChannelError("register steps must be unique and increasing")

    @property
    def components(self) -> tuple[SyntaxComponent, ...]:
        if self.direct_answer:
            return (SyntaxComponent("bit"),)
        components: list[SyntaxComponent] = []
        for index, step in enumerate(self.expected_steps):
            prefix = "" if index == 0 else "\n"
            components.extend(
                (
                    SyntaxComponent("literal", f"{prefix}r{step} = "),
                    SyntaxComponent("bit"),
                )
            )
        components.extend(
            (
                SyntaxComponent("literal", "\nanswer = "),
                SyntaxComponent("bit"),
            )
        )
        return tuple(components)

    @property
    def semantic_slot_count(self) -> int:
        return 1 if self.direct_answer else len(self.expected_steps) + 1

    @property
    def assignment_count(self) -> int:
        return 1 << self.semantic_slot_count

    @property
    def grammar_id(self) -> str:
        if self.direct_answer:
            return "direct-bit"
        return "register-" + "-".join(str(step) for step in self.expected_steps)

    def render(self, bits: Sequence[int]) -> str:
        values = tuple(int(value) for value in bits)
        if len(values) != self.semantic_slot_count or any(value not in (0, 1) for value in values):
            raise ConstrainedChannelError("assignment must provide one binary value per slot")
        iterator = iter(values)
        pieces: list[str] = []
        for component in self.components:
            pieces.append(component.literal if component.kind == "literal" else str(next(iterator)))
        return "".join(pieces)

    def assignments(self) -> Iterable[tuple[int, ...]]:
        return product((0, 1), repeat=self.semantic_slot_count)

    def accepts_text(self, text: str) -> bool:
        position = 0
        for component in self.components:
            if component.kind == "literal":
                if not text.startswith(component.literal, position):
                    return False
                position += len(component.literal)
            else:
                if position >= len(text) or text[position] not in "01":
                    return False
                position += 1
        return position == len(text)


def syntax_for_case(case: Mapping[str, Any]) -> RegisterSyntax:
    """Derive syntax from structural case fields without reading semantic truth."""

    condition = str(case.get("condition") or "")
    if "expected_steps" in case:
        return RegisterSyntax(tuple(int(step) for step in case["expected_steps"]))
    if condition in {"D", "F", "C"}:
        return RegisterSyntax((), direct_answer=True)
    length = int(case["length"])
    if condition == "S":
        return RegisterSyntax(tuple(range(0, length + 1)))
    if condition in {"O", "E", "N"}:
        edit_step = int(case["edit_step"])
        return RegisterSyntax(tuple(range(edit_step + 1, length + 1)))
    raise ConstrainedChannelError(
        f"case lacks a recognized syntax surface: {case.get('case_id')!r}"
    )


@dataclass(frozen=True)
class TokenComponent:
    """One literal token sequence or one independent 0/1 token branch."""

    kind: str
    token_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class TokenState:
    component_index: int
    token_offset: int = 0


class ConstrainedTokenAutomaton:
    """Canonical-token finite-state realization of :class:`RegisterSyntax`."""

    def __init__(self, *, tokenizer: Any, syntax: RegisterSyntax) -> None:
        self.tokenizer = tokenizer
        self.syntax = syntax
        self.eos_token_id = int(tokenizer.eos_token_id)
        if self.eos_token_id < 0:
            raise ConstrainedChannelError("tokenizer lacks a valid EOS token")
        self.zero_token_id = self._single_value_token("0")
        self.one_token_id = self._single_value_token("1")
        if self.zero_token_id == self.one_token_id:
            raise ConstrainedChannelError("0 and 1 map to the same token")

        components: list[TokenComponent] = []
        for component in syntax.components:
            if component.kind == "bit":
                components.append(TokenComponent("bit"))
                continue
            token_ids = tuple(self._encode_literal(component.literal))
            if not token_ids:
                raise ConstrainedChannelError("fixed literal encoded to no tokens")
            components.append(TokenComponent("literal", token_ids))
        self.components = tuple(components)
        self.start_state = TokenState(0, 0)

        # Token count is assignment-independent because each semantic branch is
        # exactly one token under the authority-bound tokenizer.
        self.content_token_count = sum(
            len(component.token_ids) if component.kind == "literal" else 1
            for component in self.components
        )
        self.required_new_token_count = self.content_token_count + 1  # EOS

    def _encode_literal(self, text: str) -> list[int]:
        token_ids = [
            int(value)
            for value in self.tokenizer.encode(text, add_special_tokens=False)
        ]
        decoded = self.tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if decoded != text:
            raise ConstrainedChannelError(
                f"fixed literal tokenizer round trip mismatch: {text!r} -> {decoded!r}"
            )
        special = set(int(value) for value in self.tokenizer.all_special_ids)
        if special.intersection(token_ids):
            raise ConstrainedChannelError("fixed literal unexpectedly uses a special token")
        return token_ids

    def _single_value_token(self, value: str) -> int:
        token_ids = self._encode_literal(value)
        if len(token_ids) != 1:
            raise ConstrainedChannelError(
                f"semantic value {value!r} is not one exact token under the frozen tokenizer"
            )
        return token_ids[0]

    def is_accepting(self, state: TokenState) -> bool:
        return state.component_index == len(self.components) and state.token_offset == 0

    def allowed_token_ids(self, state: TokenState) -> tuple[int, ...]:
        if self.is_accepting(state):
            return (self.eos_token_id,)
        if state.component_index < 0 or state.component_index >= len(self.components):
            raise ConstrainedChannelError("token automaton state is out of range")
        component = self.components[state.component_index]
        if component.kind == "bit":
            if state.token_offset != 0:
                raise ConstrainedChannelError("bit state has a non-zero token offset")
            return (self.zero_token_id, self.one_token_id)
        if state.token_offset < 0 or state.token_offset >= len(component.token_ids):
            raise ConstrainedChannelError("literal token offset is out of range")
        return (component.token_ids[state.token_offset],)

    def consume(self, state: TokenState, token_id: int) -> TokenState:
        allowed = self.allowed_token_ids(state)
        observed = int(token_id)
        if observed not in allowed:
            raise ConstrainedChannelError(
                f"token {observed} is not allowed at state {state}; allowed={allowed}"
            )
        if self.is_accepting(state):
            # EOS is a terminal event, not part of the content-language state.
            return state
        component = self.components[state.component_index]
        if component.kind == "bit" or state.token_offset + 1 == len(component.token_ids):
            return TokenState(state.component_index + 1, 0)
        return TokenState(state.component_index, state.token_offset + 1)

    def state_after(self, content_token_ids: Sequence[int]) -> TokenState:
        state = self.start_state
        for token_id in content_token_ids:
            if int(token_id) == self.eos_token_id:
                raise ConstrainedChannelError("EOS appeared inside constrained content")
            state = self.consume(state, int(token_id))
        return state

    def token_path(self, assignment: Sequence[int], *, include_eos: bool = True) -> tuple[int, ...]:
        assignment_values = tuple(int(value) for value in assignment)
        values = iter(assignment_values)
        if len(assignment_values) != self.syntax.semantic_slot_count:
            raise ConstrainedChannelError("assignment has the wrong semantic slot count")
        tokens: list[int] = []
        for component in self.components:
            if component.kind == "literal":
                tokens.extend(component.token_ids)
            else:
                try:
                    value = next(values)
                except StopIteration as exc:  # pragma: no cover - guarded above
                    raise ConstrainedChannelError("assignment ended early") from exc
                if value not in (0, 1):
                    raise ConstrainedChannelError("semantic slots must be binary")
                tokens.append(self.zero_token_id if value == 0 else self.one_token_id)
        if include_eos:
            tokens.append(self.eos_token_id)
        return tuple(tokens)

    def validate_complete_path(self, token_ids: Sequence[int]) -> str:
        values = tuple(int(value) for value in token_ids)
        if not values or values[-1] != self.eos_token_id:
            raise ConstrainedChannelError("constrained output does not end in EOS")
        if self.eos_token_id in values[:-1]:
            raise ConstrainedChannelError("EOS appeared before the declared endpoint")
        state = self.state_after(values[:-1])
        if not self.is_accepting(state):
            raise ConstrainedChannelError("constrained output ended before the grammar endpoint")
        text = self.tokenizer.decode(
            list(values[:-1]),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if not self.syntax.accepts_text(text):
            raise ConstrainedChannelError("decoded token path is outside the syntax language")
        return text

    def prove_all_assignments(self) -> dict[str, int | str]:
        count = 0
        for assignment in self.syntax.assignments():
            token_path = self.token_path(assignment)
            text = self.validate_complete_path(token_path)
            if text != self.syntax.render(assignment):
                raise ConstrainedChannelError("token path changed the rendered assignment")
            # Exercise the exact transition boundary and both branches, not only
            # final decode equality.
            state = self.start_state
            for token_id in token_path[:-1]:
                state = self.consume(state, token_id)
            if self.allowed_token_ids(state) != (self.eos_token_id,):
                raise ConstrainedChannelError("EOS is not unique at the endpoint")
            count += 1
        if count != self.syntax.assignment_count:
            raise ConstrainedChannelError("assignment proof count mismatch")
        return {
            "grammar_id": self.syntax.grammar_id,
            "semantic_slot_count": self.syntax.semantic_slot_count,
            "assignment_count": count,
            "required_new_token_count": self.required_new_token_count,
        }


class PrefixAllowedTokens:
    """Transformers ``prefix_allowed_tokens_fn`` bound to one exact prompt."""

    def __init__(self, *, automaton: ConstrainedTokenAutomaton, prompt_token_ids: Sequence[int]) -> None:
        self.automaton = automaton
        self.prompt_token_ids = tuple(int(value) for value in prompt_token_ids)
        if not self.prompt_token_ids:
            raise ConstrainedChannelError("prompt token sequence is empty")
        self._prefix_verified = False

    def __call__(self, batch_id: int, sent: Any) -> list[int]:
        if int(batch_id) != 0:
            raise ConstrainedChannelError("constrained channel requires batch size one")
        observed = [int(value) for value in sent.tolist()]
        prompt_length = len(self.prompt_token_ids)
        if len(observed) < prompt_length:
            raise ConstrainedChannelError("generation sequence is shorter than the prompt")
        if not self._prefix_verified:
            if tuple(observed[:prompt_length]) != self.prompt_token_ids:
                raise ConstrainedChannelError("generation prompt prefix identity mismatch")
            self._prefix_verified = True
        state = self.automaton.state_after(observed[prompt_length:])
        return list(self.automaton.allowed_token_ids(state))
