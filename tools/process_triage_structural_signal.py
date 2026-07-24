#!/usr/bin/env python3
"""Bounded retrospective artifact-cycle signal for process triage."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from process_triage_evaluator import (
    TOKEN_PATTERN,
    TriageTrajectory,
)


STRUCTURAL_SURFACE_SCHEMA_VERSION = (
    "pale_ale_structural_surface_v0.1"
)
STRUCTURAL_SIGNAL_SCHEMA_VERSION = (
    "pale_ale_task_anchored_triangle_excess_v0.1"
)
STRUCTURAL_FAMILY_ID = "task_anchored_artifact_detour_v0.1"
ORDER_SHUFFLE_SEED_ID = (
    "pale-ale-task-anchored-detour-order-shuffle-v0.1"
)
DEPENDENCY_RANDOMIZATION_SEED_ID = (
    "pale-ale-task-anchored-detour-dependency-cycle-v0.1"
)
StructuralMode = Literal[
    "primary",
    "score_order_shuffle",
    "dependency_cycle_randomization",
]


class ProcessTriageStructuralError(ValueError):
    """Raised when the bounded structural family cannot be constructed."""


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _seed_from_parts(seed_id: str, *parts: object) -> int:
    digest = hashlib.sha256(
        _canonical_json([seed_id, *parts]).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:16], byteorder="big", signed=False)


def _deterministic_permutation(
    length: int,
    *,
    seed_id: str,
    trajectory_id: str,
) -> tuple[int, ...]:
    keyed = [
        (
            hashlib.sha256(
                _canonical_json(
                    [seed_id, trajectory_id, index]
                ).encode("utf-8")
            ).hexdigest(),
            index,
        )
        for index in range(length)
    ]
    return tuple(index for _, index in sorted(keyed))


def _deterministic_score_derangement(
    length: int,
    *,
    trajectory_id: str,
) -> tuple[int, ...]:
    if length <= 1:
        return tuple(range(length))
    shift = 1 + (
        _seed_from_parts(
            ORDER_SHUFFLE_SEED_ID,
            trajectory_id,
        )
        % (length - 1)
    )
    return tuple((index + shift) % length for index in range(length))


def _dependency_cycle_permutation(
    row_count: int,
    *,
    trajectory_id: str,
) -> tuple[int, ...]:
    """Return task-first node order, avoiding the observed cycle if possible."""
    row_order = list(
        _deterministic_permutation(
            row_count,
            seed_id=DEPENDENCY_RANDOMIZATION_SEED_ID,
            trajectory_id=trajectory_id,
        )
    )
    row_order = [index + 1 for index in row_order]
    primary = list(range(1, row_count + 1))
    reverse = list(reversed(primary))
    if row_count >= 3 and row_order in (primary, reverse):
        row_order = row_order[1:] + row_order[:1]
    return (0, *row_order)


@dataclass(frozen=True)
class StructuralStepSurface:
    row_id: str
    eligible_index: int
    content: str
    tool_names: tuple[str, ...]
    signature: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "eligible_index": self.eligible_index,
            "content": self.content,
            "tool_names": list(self.tool_names),
            "signature": self.signature,
        }


@dataclass(frozen=True)
class StructuralTrajectorySurface:
    trajectory_id: str
    group_id: str
    domain: str
    source_slot: int
    task_surface_text: str
    steps: tuple[StructuralStepSurface, ...]
    schema_version: str = STRUCTURAL_SURFACE_SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "trajectory_id": self.trajectory_id,
            "group_id": self.group_id,
            "domain": self.domain,
            "source_slot": self.source_slot,
            "task_surface_text": self.task_surface_text,
            "steps": [step.as_dict() for step in self.steps],
        }


@dataclass(frozen=True)
class StructuralStepScore:
    row_id: str
    trajectory_id: str
    group_id: str
    domain: str
    source_slot: int
    eligible_index: int
    score: float
    previous_edge_distance: float
    next_edge_distance: float
    bypass_distance: float
    mode: StructuralMode
    schema_version: str = STRUCTURAL_SIGNAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        values = (
            self.score,
            self.previous_edge_distance,
            self.next_edge_distance,
            self.bypass_distance,
        )
        if not all(math.isfinite(value) for value in values):
            raise ProcessTriageStructuralError(
                "structural score contains a non-finite value"
            )
        if not 0.0 <= self.score <= 1.0:
            raise ProcessTriageStructuralError(
                "structural score is outside [0, 1]"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "family_id": STRUCTURAL_FAMILY_ID,
            "mode": self.mode,
            "row_id": self.row_id,
            "trajectory_id": self.trajectory_id,
            "group_id": self.group_id,
            "domain": self.domain,
            "source_slot": self.source_slot,
            "eligible_index": self.eligible_index,
            "task_anchored_triangle_excess": self.score,
            "previous_edge_distance": self.previous_edge_distance,
            "next_edge_distance": self.next_edge_distance,
            "bypass_distance": self.bypass_distance,
        }


def build_structural_surface(
    trajectories: Sequence[TriageTrajectory],
) -> tuple[StructuralTrajectorySurface, ...]:
    """Strip every outcome field before structural feature computation."""
    surfaces = []
    for trajectory in trajectories:
        surfaces.append(
            StructuralTrajectorySurface(
                trajectory_id=trajectory.trajectory_id,
                group_id=trajectory.group_id,
                domain=trajectory.domain,
                source_slot=trajectory.source_slot,
                task_surface_text=trajectory.task_surface_text,
                steps=tuple(
                    StructuralStepSurface(
                        row_id=step.row_id,
                        eligible_index=step.eligible_index,
                        content=step.content,
                        tool_names=step.tool_names,
                        signature=step.signature,
                    )
                    for step in trajectory.steps
                ),
            )
        )
    return tuple(surfaces)


def structural_surface_receipt(
    surfaces: Sequence[StructuralTrajectorySurface],
) -> dict[str, Any]:
    payload = [
        surface.as_dict()
        for surface in sorted(
            surfaces,
            key=lambda value: value.trajectory_id,
        )
    ]
    return {
        "schema_version": STRUCTURAL_SURFACE_SCHEMA_VERSION,
        "firewall_status": "outcome_fields_absent_by_type",
        "information_horizon": "full_trajectory_retrospective",
        "trajectory_count": len(payload),
        "row_count": sum(len(surface.steps) for surface in surfaces),
        "sha256": hashlib.sha256(
            _canonical_json(payload).encode("utf-8")
        ).hexdigest(),
        "forbidden_fields": [
            "native_label",
            "actionable_defect",
            "step_labels",
            "ground_truth",
            "final_label",
            "answer_text",
        ],
    }


def _lexical_tokens(value: object) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        text = value
    else:
        text = _canonical_json(value)
    return {
        f"lex:{token.lower()}"
        for token in TOKEN_PATTERN.findall(text)
    }


def _task_tokens(surface: StructuralTrajectorySurface) -> frozenset[str]:
    tokens = _lexical_tokens(surface.task_surface_text)
    tokens.add("kind:task_anchor")
    return frozenset(tokens)


def _step_tokens(step: StructuralStepSurface) -> frozenset[str]:
    try:
        signature = json.loads(step.signature)
    except json.JSONDecodeError as exc:
        raise ProcessTriageStructuralError(
            f"invalid canonical step signature: {step.row_id}"
        ) from exc
    if not isinstance(signature, dict):
        raise ProcessTriageStructuralError(
            f"step signature is not an object: {step.row_id}"
        )

    tokens = _lexical_tokens(signature.get("content"))
    calls = signature.get("tool_calls")
    if not isinstance(calls, list):
        raise ProcessTriageStructuralError(
            f"step signature tool_calls is not a list: {step.row_id}"
        )
    for call in calls:
        if not isinstance(call, dict):
            raise ProcessTriageStructuralError(
                f"step signature call is not an object: {step.row_id}"
            )
        name = str(call.get("name") or "").strip().lower()
        if name:
            tokens.add(f"tool:{name}")
        tokens.update(_lexical_tokens(call.get("arguments")))
    tokens.add(
        "kind:tool_call" if calls else "kind:assistant_text"
    )
    return frozenset(tokens)


def _jaccard_distance(
    left: frozenset[str],
    right: frozenset[str],
) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    return float(1.0 - len(left & right) / len(union))


def _triangle_score(
    previous: frozenset[str],
    current: frozenset[str],
    following: frozenset[str],
) -> tuple[float, float, float, float]:
    previous_distance = _jaccard_distance(previous, current)
    next_distance = _jaccard_distance(current, following)
    bypass_distance = _jaccard_distance(previous, following)
    raw_excess = (
        previous_distance + next_distance - bypass_distance
    )
    if raw_excess < -1.0e-12:
        raise ProcessTriageStructuralError(
            "Jaccard triangle inequality was violated"
        )
    score = min(1.0, max(0.0, 0.5 * raw_excess))
    return (
        score,
        previous_distance,
        next_distance,
        bypass_distance,
    )


def _primary_cycle_scores(
    surface: StructuralTrajectorySurface,
) -> tuple[StructuralStepScore, ...]:
    if not surface.steps:
        return ()
    task = _task_tokens(surface)
    row_tokens = tuple(_step_tokens(step) for step in surface.steps)
    cycle_tokens = (task, *row_tokens)
    scores = []
    for position, step in enumerate(surface.steps, start=1):
        previous = cycle_tokens[position - 1]
        current = cycle_tokens[position]
        following = (
            cycle_tokens[position + 1]
            if position + 1 < len(cycle_tokens)
            else task
        )
        values = _triangle_score(previous, current, following)
        scores.append(
            StructuralStepScore(
                row_id=step.row_id,
                trajectory_id=surface.trajectory_id,
                group_id=surface.group_id,
                domain=surface.domain,
                source_slot=surface.source_slot,
                eligible_index=step.eligible_index,
                score=values[0],
                previous_edge_distance=values[1],
                next_edge_distance=values[2],
                bypass_distance=values[3],
                mode="primary",
            )
        )
    return tuple(scores)


def _dependency_randomized_scores(
    surface: StructuralTrajectorySurface,
) -> tuple[StructuralStepScore, ...]:
    if not surface.steps:
        return ()
    nodes = (
        _task_tokens(surface),
        *(_step_tokens(step) for step in surface.steps),
    )
    permutation = _dependency_cycle_permutation(
        len(surface.steps),
        trajectory_id=surface.trajectory_id,
    )
    position_by_node = {
        node_index: position
        for position, node_index in enumerate(permutation)
    }
    scores = []
    for row_index, step in enumerate(surface.steps, start=1):
        position = position_by_node[row_index]
        previous_index = permutation[(position - 1) % len(permutation)]
        following_index = permutation[
            (position + 1) % len(permutation)
        ]
        values = _triangle_score(
            nodes[previous_index],
            nodes[row_index],
            nodes[following_index],
        )
        scores.append(
            StructuralStepScore(
                row_id=step.row_id,
                trajectory_id=surface.trajectory_id,
                group_id=surface.group_id,
                domain=surface.domain,
                source_slot=surface.source_slot,
                eligible_index=step.eligible_index,
                score=values[0],
                previous_edge_distance=values[1],
                next_edge_distance=values[2],
                bypass_distance=values[3],
                mode="dependency_cycle_randomization",
            )
        )
    return tuple(scores)


def _order_shuffled_scores(
    surface: StructuralTrajectorySurface,
) -> tuple[StructuralStepScore, ...]:
    primary = _primary_cycle_scores(surface)
    permutation = _deterministic_score_derangement(
        len(primary),
        trajectory_id=surface.trajectory_id,
    )
    return tuple(
        StructuralStepScore(
            row_id=target.row_id,
            trajectory_id=target.trajectory_id,
            group_id=target.group_id,
            domain=target.domain,
            source_slot=target.source_slot,
            eligible_index=target.eligible_index,
            score=source.score,
            previous_edge_distance=source.previous_edge_distance,
            next_edge_distance=source.next_edge_distance,
            bypass_distance=source.bypass_distance,
            mode="score_order_shuffle",
        )
        for target, source in zip(
            primary,
            (primary[index] for index in permutation),
        )
    )


def task_anchored_triangle_excess(
    surfaces: Sequence[StructuralTrajectorySurface],
    *,
    mode: StructuralMode = "primary",
) -> tuple[StructuralStepScore, ...]:
    calculators = {
        "primary": _primary_cycle_scores,
        "score_order_shuffle": _order_shuffled_scores,
        "dependency_cycle_randomization": (
            _dependency_randomized_scores
        ),
    }
    if mode not in calculators:
        raise ProcessTriageStructuralError(
            f"unknown structural mode: {mode}"
        )
    rows = [
        row
        for surface in sorted(
            surfaces,
            key=lambda value: value.trajectory_id,
        )
        for row in calculators[mode](surface)
    ]
    row_ids = [row.row_id for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise ProcessTriageStructuralError(
            "structural output row IDs repeat"
        )
    return tuple(rows)


def structural_family_manifest() -> dict[str, Any]:
    return {
        "schema_version": STRUCTURAL_SIGNAL_SCHEMA_VERSION,
        "epistemic_status": "freeze_candidate_unscored_on_labels",
        "family_id": STRUCTURAL_FAMILY_ID,
        "primary_scalar": "task_anchored_triangle_excess",
        "formula": (
            "0.5 * (d(previous,current) + d(current,next) "
            "- d(previous,next))"
        ),
        "distance": "exact_set_jaccard_distance",
        "node_construction": {
            "task_anchor": (
                "normalized visible task surface with kind:task_anchor"
            ),
            "assistant_artifact": (
                "lowercased Unicode lexical tokens from content and tool "
                "arguments, exact lowercased tool-name token, and artifact "
                "kind token"
            ),
            "cycle": (
                "task_anchor -> assistant rows in recorded order -> "
                "task_anchor"
            ),
        },
        "range": [0.0, 1.0],
        "normalization": "none_beyond_bounded_formula",
        "missing_value_rule": "no_missing_values",
        "empty_trajectory_rule": "emit_no_rows",
        "single_row_rule": "score_equals_task_to_row_jaccard_distance",
        "information_horizon": "full_trajectory_retrospective",
        "future_context_used": True,
        "learned_parameters": False,
        "thresholds": [],
        "window_parameters": [],
        "embedding_model": None,
        "development_candidate_family_count": 1,
        "authorized_model_input_columns": [
            "task_anchored_triangle_excess"
        ],
        "diagnostic_only_not_model_inputs": [
            "previous_edge_distance",
            "next_edge_distance",
            "bypass_distance",
        ],
        "frozen_controls": {
            "score_order_shuffle": {
                "seed_id": ORDER_SHUFFLE_SEED_ID,
                "rule": (
                    "deterministically permute primary score tuples across "
                    "rows within each trajectory"
                ),
            },
            "dependency_cycle_randomization": {
                "seed_id": DEPENDENCY_RANDOMIZATION_SEED_ID,
                "rule": (
                    "deterministically replace the recorded task-row cycle "
                    "with a random Hamiltonian cycle over the same nodes"
                ),
            },
            "label_permutation": (
                "implemented only in the development evaluator with a "
                "separate frozen seed"
            ),
        },
        "tie_breaking": (
            "the existing global evaluator tie-breaker remains unchanged"
        ),
        "claim_boundary": (
            "artifact-level retrospective prioritization only; this scalar "
            "is not a correctness score, hidden-state observable, or online "
            "warning"
        ),
        "retirement_rule": (
            "retire if development gain is negligible, if cheap "
            "length/position/retry/tool-error features explain the gain, "
            "or if frozen controls retain comparable performance"
        ),
        "locked_evaluation_authorized": False,
    }
