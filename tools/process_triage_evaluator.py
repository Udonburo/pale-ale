#!/usr/bin/env python3
"""Development evaluator for retrospective artifact-level process triage.

The evaluator fixes the mechanics that must exist before a structural signal
is developed:

* an AgentProcessBench adapter with visible-task-surface grouping;
* a canonical trajectory/assistant-step schema;
* a typed feature firewall that excludes outcome-bearing fields;
* reproducible cheap baseline features;
* a global eligible-row review budget;
* deterministic tie handling; and
* paired highest-group bootstrap and clean-trajectory review burden.

No structural signal is defined here.  This is development infrastructure, not
a held-out benchmark claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


TRAJECTORY_SCHEMA_VERSION = "pale_ale_process_trajectory_v0.2"
FEATURE_SURFACE_SCHEMA_VERSION = "pale_ale_feature_firewall_v0.1"
FEATURE_SCHEMA_VERSION = "pale_ale_process_cheap_features_v0.2"
EVALUATION_SCHEMA_VERSION = "pale_ale_global_review_budget_v0.2"
BOOTSTRAP_SCHEMA_VERSION = "pale_ale_group_bootstrap_v0.1"
NEAR_DUPLICATE_SCHEMA_VERSION = "pale_ale_task_surface_near_duplicate_v0.1"
AGENT_PROCESS_BENCH_MAPPING_ID = "agent_process_bench_negative_step_v0.1"
AGENT_PROCESS_BENCH_GROUPING_ID = (
    "agent_process_bench_domain_task_surface_sha256_v0.1"
)
GROUP_SPLIT_ID = "sha256_domain_group_order_v0.1"
NEAR_DUPLICATE_METRIC_ID = "unicode_word_set_jaccard_v1"
NEAR_DUPLICATE_THRESHOLD = 0.95
SOURCE_MODEL_MAPPING_STATUS = (
    "unresolved_public_record_exposes_sample_index_only"
)
BOOTSTRAP_RESAMPLING_ID = "sha256_domain_stratified_group_bootstrap_v1"

TOOL_ERROR_PATTERN = re.compile(
    r"\b(error|failed|failure|exception|traceback|timeout|timed out|"
    r"not found|invalid|denied|forbidden|unauthorized|404|500)\b",
    flags=re.IGNORECASE,
)
TOKEN_PATTERN = re.compile(r"[\w]+", flags=re.UNICODE)


class ProcessTriageDevelopmentError(ValueError):
    """Raised when data or evaluation mechanics violate the contract."""


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return _canonical_json(value)


def _tool_calls(message: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return ()
    return tuple(call for call in calls if isinstance(call, Mapping))


def _tool_names(message: Mapping[str, Any]) -> tuple[str, ...]:
    names: list[str] = []
    for call in _tool_calls(message):
        function = call.get("function")
        if isinstance(function, Mapping):
            names.append(str(function.get("name") or ""))
        else:
            names.append(str(call.get("name") or ""))
    return tuple(names)


def _step_signature(message: Mapping[str, Any]) -> str:
    calls = []
    for call in _tool_calls(message):
        function = call.get("function")
        if isinstance(function, Mapping):
            calls.append(
                {
                    "name": str(function.get("name") or ""),
                    "arguments": function.get("arguments"),
                }
            )
        else:
            calls.append(
                {
                    "name": str(call.get("name") or ""),
                    "arguments": call.get("arguments"),
                }
            )
    return _canonical_json(
        {
            "content": " ".join(_text(message.get("content")).split()),
            "tool_calls": calls,
        }
    )


def _artifact_type(message: Mapping[str, Any]) -> str:
    if _tool_calls(message):
        return "tool_call"
    if _text(message.get("content")).strip():
        return "assistant_text"
    return "assistant_empty"


def _normalized_task_surface(record: Mapping[str, Any]) -> dict[str, str]:
    """Return pre-outcome task fields used to block duplicated task surfaces.

    AgentProcessBench contains distinct query indices that can share the same
    visible question and task description.  Grouping only by query index would
    allow those task-family duplicates to cross a development/evaluation split.
    Ground truth, final labels, and step labels are intentionally excluded.
    """

    question = " ".join(_text(record.get("question")).split())
    task_description = " ".join(
        _text(record.get("task_description")).split()
    )
    data_source = str(record.get("data_source") or "")
    return {
        "question": question,
        "task_description": task_description,
        "data_source": data_source,
    }


def _agent_process_bench_group_id(
    record: Mapping[str, Any],
    *,
    domain: str,
) -> str:
    digest = hashlib.sha256(
        _canonical_json(
            [
                AGENT_PROCESS_BENCH_GROUPING_ID,
                domain,
                _normalized_task_surface(record),
            ]
        ).encode("utf-8")
    ).hexdigest()
    return f"{domain}:task-surface:{digest}"


def _task_surface_text(record: Mapping[str, Any]) -> str:
    surface = _normalized_task_surface(record)
    return "\n".join(
        (
            surface["data_source"],
            surface["question"],
            surface["task_description"],
        )
    )


@dataclass(frozen=True)
class TriageStep:
    trajectory_id: str
    group_id: str
    domain: str
    row_id: str
    message_index: int
    eligible_index: int
    artifact_type: str
    content: str
    tool_names: tuple[str, ...]
    signature: str
    native_label: int
    actionable_defect: bool
    preceding_tool_error: bool
    prior_tool_error_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "group_id": self.group_id,
            "domain": self.domain,
            "row_id": self.row_id,
            "message_index": self.message_index,
            "eligible_index": self.eligible_index,
            "artifact_type": self.artifact_type,
            "tool_names": list(self.tool_names),
            "native_label": self.native_label,
            "actionable_defect": self.actionable_defect,
            "preceding_tool_error": self.preceding_tool_error,
            "prior_tool_error_count": self.prior_tool_error_count,
        }


@dataclass(frozen=True)
class TriageTrajectory:
    trajectory_id: str
    group_id: str
    domain: str
    source_record_id: str
    source_slot: int
    task_surface_text: str
    steps: tuple[TriageStep, ...]
    final_label: int | None
    label_mapping_id: str = AGENT_PROCESS_BENCH_MAPPING_ID
    schema_version: str = TRAJECTORY_SCHEMA_VERSION

    @property
    def first_actionable_row_id(self) -> str | None:
        for step in self.steps:
            if step.actionable_defect:
                return step.row_id
        return None

    @property
    def is_clean(self) -> bool:
        return self.first_actionable_row_id is None


@dataclass(frozen=True)
class FeatureStepSurface:
    """A row surface from which all outcome-bearing fields are absent."""

    trajectory_id: str
    group_id: str
    domain: str
    source_slot: int
    row_id: str
    message_index: int
    eligible_index: int
    artifact_type: str
    content: str
    tool_names: tuple[str, ...]
    signature: str
    preceding_tool_error: bool
    prior_tool_error_count: int


@dataclass(frozen=True)
class FeatureTrajectorySurface:
    """Feature-only trajectory passed across the leakage firewall."""

    trajectory_id: str
    group_id: str
    domain: str
    source_slot: int
    steps: tuple[FeatureStepSurface, ...]
    schema_version: str = FEATURE_SURFACE_SCHEMA_VERSION


@dataclass(frozen=True)
class CheapStepFeatures:
    row_id: str
    trajectory_id: str
    group_id: str
    domain: str
    source_slot: int
    normalized_position: float
    eligible_trajectory_length: int
    artifact_type: str
    prior_exact_retry_count: int
    prior_same_tool_count: int
    preceding_tool_error: int
    prior_tool_error_count: int
    lexical_drift_from_previous: float
    content_character_count: int
    schema_version: str = FEATURE_SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "row_id": self.row_id,
            "trajectory_id": self.trajectory_id,
            "group_id": self.group_id,
            "domain": self.domain,
            "source_slot": self.source_slot,
            "normalized_position": self.normalized_position,
            "eligible_trajectory_length": self.eligible_trajectory_length,
            "artifact_type": self.artifact_type,
            "prior_exact_retry_count": self.prior_exact_retry_count,
            "prior_same_tool_count": self.prior_same_tool_count,
            "preceding_tool_error": self.preceding_tool_error,
            "prior_tool_error_count": self.prior_tool_error_count,
            "lexical_drift_from_previous": self.lexical_drift_from_previous,
            "content_character_count": self.content_character_count,
        }


@dataclass(frozen=True)
class OperationalSuccessRule:
    """Candidate operational rule; it is not frozen by defining the type."""

    minimum_point_recall_gain: float = 0.10
    minimum_recall_ci_lower: float = 0.0
    maximum_clean_allocation_ci_upper: float = 0.05
    primary_metric: str = "first_actionable_defect_recall_at_global_row_budget"
    clean_burden_metric: str = "clean_row_allocation"
    status: str = "candidate_not_frozen"


def _tool_error_between(
    messages: Sequence[Mapping[str, Any]],
    start_exclusive: int,
    end_exclusive: int,
) -> bool:
    for message in messages[start_exclusive + 1 : end_exclusive]:
        if str(message.get("role") or "") != "tool":
            continue
        if TOOL_ERROR_PATTERN.search(_text(message.get("content"))):
            return True
    return False


def parse_agent_process_bench_record(
    record: Mapping[str, Any],
    *,
    domain: str,
) -> TriageTrajectory:
    """Map one AgentProcessBench JSON object into the canonical schema."""

    messages_raw = record.get("messages")
    labels_raw = record.get("step_labels")
    if not isinstance(messages_raw, list):
        raise ProcessTriageDevelopmentError("messages must be a list")
    if not isinstance(labels_raw, Mapping):
        raise ProcessTriageDevelopmentError("step_labels must be a mapping")
    messages: list[Mapping[str, Any]] = []
    for index, message in enumerate(messages_raw):
        if not isinstance(message, Mapping):
            raise ProcessTriageDevelopmentError(
                f"message {index} is not an object"
            )
        messages.append(message)

    query_index = record.get("query_index")
    sample_index = record.get("sample_index")
    if query_index is None or sample_index is None:
        raise ProcessTriageDevelopmentError(
            "AgentProcessBench records require query_index and sample_index"
        )
    try:
        source_slot = int(sample_index)
    except (TypeError, ValueError) as exc:
        raise ProcessTriageDevelopmentError(
            "sample_index must be an integer-like opaque source slot"
        ) from exc
    group_id = _agent_process_bench_group_id(record, domain=domain)
    trajectory_id = (
        f"{domain}:query:{query_index}:sample:{sample_index}"
    )
    source_record_id = str(record.get("total_index", trajectory_id))

    assistant_indices = [
        index
        for index, message in enumerate(messages)
        if str(message.get("role") or "") == "assistant"
    ]
    normalized_labels: dict[int, int] = {}
    for raw_index, raw_label in labels_raw.items():
        try:
            message_index = int(raw_index)
            label = int(raw_label)
        except (TypeError, ValueError) as exc:
            raise ProcessTriageDevelopmentError(
                f"invalid step label entry {raw_index!r}: {raw_label!r}"
            ) from exc
        if label not in {-1, 0, 1}:
            raise ProcessTriageDevelopmentError(
                f"step label must be -1, 0, or 1; got {label}"
            )
        normalized_labels[message_index] = label
    if set(assistant_indices) != set(normalized_labels):
        missing = sorted(set(assistant_indices) - set(normalized_labels))
        extra = sorted(set(normalized_labels) - set(assistant_indices))
        raise ProcessTriageDevelopmentError(
            f"assistant/label mismatch; missing={missing}, extra={extra}"
        )

    steps: list[TriageStep] = []
    previous_assistant_index = -1
    cumulative_tool_error_count = 0
    for eligible_index, message_index in enumerate(assistant_indices):
        message = messages[message_index]
        preceding_error = _tool_error_between(
            messages, previous_assistant_index, message_index
        )
        cumulative_tool_error_count += int(preceding_error)
        row_id = f"{trajectory_id}:message:{message_index}"
        label = normalized_labels[message_index]
        steps.append(
            TriageStep(
                trajectory_id=trajectory_id,
                group_id=group_id,
                domain=domain,
                row_id=row_id,
                message_index=message_index,
                eligible_index=eligible_index,
                artifact_type=_artifact_type(message),
                content=_text(message.get("content")),
                tool_names=_tool_names(message),
                signature=_step_signature(message),
                native_label=label,
                actionable_defect=label == -1,
                preceding_tool_error=preceding_error,
                prior_tool_error_count=cumulative_tool_error_count,
            )
        )
        previous_assistant_index = message_index

    final_label_raw = record.get("final_label")
    final_label = None if final_label_raw is None else int(final_label_raw)
    if final_label is not None and final_label not in {-1, 0, 1}:
        raise ProcessTriageDevelopmentError("final_label must be -1, 0, or 1")
    return TriageTrajectory(
        trajectory_id=trajectory_id,
        group_id=group_id,
        domain=domain,
        source_record_id=source_record_id,
        source_slot=source_slot,
        task_surface_text=_task_surface_text(record),
        steps=tuple(steps),
        final_label=final_label,
    )


def load_agent_process_bench_jsonl(
    path: Path,
    *,
    domain: str | None = None,
) -> tuple[TriageTrajectory, ...]:
    resolved_domain = str(domain or path.stem)
    trajectories: list[TriageTrajectory] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProcessTriageDevelopmentError(
                    f"invalid JSON at {path}:{line_number}"
                ) from exc
            if not isinstance(record, Mapping):
                raise ProcessTriageDevelopmentError(
                    f"expected object at {path}:{line_number}"
                )
            trajectories.append(
                parse_agent_process_bench_record(record, domain=resolved_domain)
            )
    return tuple(trajectories)


def _token_set(value: str) -> set[str]:
    return {token.lower() for token in TOKEN_PATTERN.findall(value)}


def _jaccard_drift(left: str, right: str) -> float:
    left_tokens = _token_set(left)
    right_tokens = _token_set(right)
    if not left_tokens and not right_tokens:
        return 0.0
    union = left_tokens | right_tokens
    return float(1.0 - len(left_tokens & right_tokens) / len(union))


def build_feature_surface(
    trajectories: Sequence[TriageTrajectory],
) -> tuple[FeatureTrajectorySurface, ...]:
    """Strip labels, ground truth, and final outcomes before feature work."""

    result: list[FeatureTrajectorySurface] = []
    for trajectory in trajectories:
        result.append(
            FeatureTrajectorySurface(
                trajectory_id=trajectory.trajectory_id,
                group_id=trajectory.group_id,
                domain=trajectory.domain,
                source_slot=trajectory.source_slot,
                steps=tuple(
                    FeatureStepSurface(
                        trajectory_id=step.trajectory_id,
                        group_id=step.group_id,
                        domain=step.domain,
                        source_slot=trajectory.source_slot,
                        row_id=step.row_id,
                        message_index=step.message_index,
                        eligible_index=step.eligible_index,
                        artifact_type=step.artifact_type,
                        content=step.content,
                        tool_names=step.tool_names,
                        signature=step.signature,
                        preceding_tool_error=step.preceding_tool_error,
                        prior_tool_error_count=step.prior_tool_error_count,
                    )
                    for step in trajectory.steps
                ),
            )
        )
    return tuple(result)


def feature_surface_receipt(
    surfaces: Sequence[FeatureTrajectorySurface],
) -> dict[str, Any]:
    """Return an auditable hash of the exact outcome-free feature input."""

    payload = [
        {
            "schema_version": surface.schema_version,
            "trajectory_id": surface.trajectory_id,
            "group_id": surface.group_id,
            "domain": surface.domain,
            "source_slot": surface.source_slot,
            "steps": [
                {
                    "trajectory_id": step.trajectory_id,
                    "group_id": step.group_id,
                    "domain": step.domain,
                    "source_slot": step.source_slot,
                    "row_id": step.row_id,
                    "message_index": step.message_index,
                    "eligible_index": step.eligible_index,
                    "artifact_type": step.artifact_type,
                    "content": step.content,
                    "tool_names": list(step.tool_names),
                    "signature": step.signature,
                    "preceding_tool_error": step.preceding_tool_error,
                    "prior_tool_error_count": step.prior_tool_error_count,
                }
                for step in surface.steps
            ],
        }
        for surface in surfaces
    ]
    encoded = _canonical_json(payload).encode("utf-8")
    return {
        "schema_version": FEATURE_SURFACE_SCHEMA_VERSION,
        "firewall_status": "outcome_fields_absent_by_type",
        "trajectory_count": len(surfaces),
        "row_count": sum(len(surface.steps) for surface in surfaces),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "forbidden_fields": [
            "native_label",
            "actionable_defect",
            "step_labels",
            "ground_truth",
            "final_label",
            "answer_text",
        ],
    }


def cheap_features(
    trajectories: Sequence[FeatureTrajectorySurface],
) -> tuple[CheapStepFeatures, ...]:
    if any(
        not isinstance(trajectory, FeatureTrajectorySurface)
        for trajectory in trajectories
    ):
        raise ProcessTriageDevelopmentError(
            "cheap_features accepts only outcome-free FeatureTrajectorySurface "
            "objects; call build_feature_surface first"
        )
    result: list[CheapStepFeatures] = []
    for trajectory in trajectories:
        signature_counts: Counter[str] = Counter()
        tool_counts: Counter[str] = Counter()
        previous_signature = ""
        length = len(trajectory.steps)
        for index, step in enumerate(trajectory.steps):
            normalized_position = 0.0 if length <= 1 else index / (length - 1)
            prior_exact_retry_count = signature_counts[step.signature]
            prior_same_tool_count = sum(tool_counts[name] for name in step.tool_names)
            result.append(
                CheapStepFeatures(
                    row_id=step.row_id,
                    trajectory_id=trajectory.trajectory_id,
                    group_id=trajectory.group_id,
                    domain=trajectory.domain,
                    source_slot=trajectory.source_slot,
                    normalized_position=float(normalized_position),
                    eligible_trajectory_length=length,
                    artifact_type=step.artifact_type,
                    prior_exact_retry_count=int(prior_exact_retry_count),
                    prior_same_tool_count=int(prior_same_tool_count),
                    preceding_tool_error=int(step.preceding_tool_error),
                    prior_tool_error_count=int(step.prior_tool_error_count),
                    lexical_drift_from_previous=(
                        0.0
                        if index == 0
                        else _jaccard_drift(previous_signature, step.signature)
                    ),
                    content_character_count=len(step.content),
                )
            )
            signature_counts[step.signature] += 1
            for name in step.tool_names:
                tool_counts[name] += 1
            previous_signature = step.signature
    return tuple(result)


def position_only_scores(
    features: Sequence[CheapStepFeatures],
) -> dict[str, float]:
    return {feature.row_id: feature.normalized_position for feature in features}


def linear_development_scores(
    features: Sequence[CheapStepFeatures],
    *,
    weights: Mapping[str, float],
    artifact_type_weights: Mapping[str, float] | None = None,
    source_slot_weights: Mapping[int, float] | None = None,
) -> dict[str, float]:
    """Apply explicit development weights; this function does not fit them."""

    allowed = {
        "normalized_position",
        "eligible_trajectory_length",
        "prior_exact_retry_count",
        "prior_same_tool_count",
        "preceding_tool_error",
        "prior_tool_error_count",
        "lexical_drift_from_previous",
        "content_character_count",
    }
    unknown = set(weights) - allowed
    if unknown:
        raise ProcessTriageDevelopmentError(
            f"unknown cheap-feature weights: {sorted(unknown)}"
        )
    type_weights = dict(artifact_type_weights or {})
    slot_weights = {
        int(slot): float(weight)
        for slot, weight in (source_slot_weights or {}).items()
    }
    scores: dict[str, float] = {}
    for feature in features:
        score = sum(
            float(weight) * float(getattr(feature, name))
            for name, weight in weights.items()
        )
        score += float(type_weights.get(feature.artifact_type, 0.0))
        score += float(slot_weights.get(feature.source_slot, 0.0))
        if not math.isfinite(score):
            raise ProcessTriageDevelopmentError(
                f"non-finite score for row {feature.row_id}"
            )
        scores[feature.row_id] = float(score)
    return scores


def source_slot_only_scores(
    features: Sequence[CheapStepFeatures],
    *,
    source_slot_weights: Mapping[int, float],
) -> dict[str, float]:
    """Score opaque source slots without claiming a named-model mapping."""

    return linear_development_scores(
        features,
        weights={},
        source_slot_weights=source_slot_weights,
    )


def _all_steps(
    trajectories: Sequence[TriageTrajectory],
) -> dict[str, TriageStep]:
    result: dict[str, TriageStep] = {}
    for trajectory in trajectories:
        for step in trajectory.steps:
            if step.row_id in result:
                raise ProcessTriageDevelopmentError(
                    f"duplicate row ID: {step.row_id}"
                )
            result[step.row_id] = step
    return result


def evaluate_global_review_budget(
    trajectories: Sequence[TriageTrajectory],
    *,
    scores: Mapping[str, float],
    budget_fraction: float = 0.10,
) -> dict[str, Any]:
    """Evaluate one score over a single global eligible-row pool."""

    if not 0.0 < budget_fraction <= 1.0:
        raise ProcessTriageDevelopmentError(
            "budget_fraction must be in the interval (0, 1]"
        )
    steps = _all_steps(trajectories)
    if set(scores) != set(steps):
        missing = sorted(set(steps) - set(scores))
        extra = sorted(set(scores) - set(steps))
        raise ProcessTriageDevelopmentError(
            f"score coverage mismatch; missing={missing[:5]}, extra={extra[:5]}"
        )
    for row_id, score in scores.items():
        if not math.isfinite(float(score)):
            raise ProcessTriageDevelopmentError(
                f"non-finite score for row {row_id}"
            )

    budget = int(math.ceil(budget_fraction * len(steps)))
    ranked = sorted(
        steps.values(),
        key=lambda step: (
            -float(scores[step.row_id]),
            step.domain,
            step.group_id,
            step.trajectory_id,
            step.message_index,
            step.row_id,
        ),
    )
    selected = ranked[:budget]
    selected_ids = {step.row_id for step in selected}
    trajectory_by_id = {
        trajectory.trajectory_id: trajectory for trajectory in trajectories
    }

    positive = [
        trajectory
        for trajectory in trajectories
        if trajectory.first_actionable_row_id is not None
    ]
    clean = [trajectory for trajectory in trajectories if trajectory.is_clean]
    first_defects_selected = sum(
        trajectory.first_actionable_row_id in selected_ids
        for trajectory in positive
    )
    clean_selected_rows = sum(
        trajectory_by_id[step.trajectory_id].is_clean for step in selected
    )
    clean_alerted = {
        step.trajectory_id
        for step in selected
        if trajectory_by_id[step.trajectory_id].is_clean
    }
    actionable_selected = sum(step.actionable_defect for step in selected)
    selected_by_domain: dict[str, list[TriageStep]] = defaultdict(list)
    selected_by_group: dict[str, list[TriageStep]] = defaultdict(list)
    trajectories_by_domain: dict[str, list[TriageTrajectory]] = defaultdict(list)
    trajectories_by_group: dict[str, list[TriageTrajectory]] = defaultdict(list)
    for step in selected:
        selected_by_domain[step.domain].append(step)
        selected_by_group[step.group_id].append(step)
    for trajectory in trajectories:
        trajectories_by_domain[trajectory.domain].append(trajectory)
        trajectories_by_group[trajectory.group_id].append(trajectory)

    domain_metrics: dict[str, dict[str, Any]] = {}
    for domain, domain_trajectories in sorted(trajectories_by_domain.items()):
        domain_selected_ids = {
            step.row_id for step in selected_by_domain.get(domain, [])
        }
        domain_positive = [
            trajectory
            for trajectory in domain_trajectories
            if not trajectory.is_clean
        ]
        domain_clean = [
            trajectory for trajectory in domain_trajectories if trajectory.is_clean
        ]
        domain_first_selected = sum(
            trajectory.first_actionable_row_id in domain_selected_ids
            for trajectory in domain_positive
        )
        domain_selected = selected_by_domain.get(domain, [])
        domain_clean_selected = sum(
            trajectory_by_id[step.trajectory_id].is_clean
            for step in domain_selected
        )
        domain_metrics[domain] = {
            "trajectory_count": len(domain_trajectories),
            "positive_trajectory_count": len(domain_positive),
            "clean_trajectory_count": len(domain_clean),
            "selected_row_count": len(domain_selected),
            "first_actionable_defect_recall": (
                domain_first_selected / len(domain_positive)
                if domain_positive
                else None
            ),
            "clean_row_allocation": (
                domain_clean_selected / len(domain_selected)
                if domain_selected
                else None
            ),
        }

    group_metrics: dict[str, dict[str, Any]] = {}
    for group_id, group_trajectories in sorted(trajectories_by_group.items()):
        group_selected_ids = {
            step.row_id for step in selected_by_group.get(group_id, [])
        }
        group_positive = [
            trajectory
            for trajectory in group_trajectories
            if not trajectory.is_clean
        ]
        group_first_selected = sum(
            trajectory.first_actionable_row_id in group_selected_ids
            for trajectory in group_positive
        )
        group_metrics[group_id] = {
            "domain": group_trajectories[0].domain,
            "trajectory_count": len(group_trajectories),
            "positive_trajectory_count": len(group_positive),
            "selected_row_count": len(selected_by_group.get(group_id, [])),
            "first_actionable_defect_recall": (
                group_first_selected / len(group_positive)
                if group_positive
                else None
            ),
        }
    domain_recalls = [
        float(metrics["first_actionable_defect_recall"])
        for metrics in domain_metrics.values()
        if metrics["first_actionable_defect_recall"] is not None
    ]
    group_recalls = [
        float(metrics["first_actionable_defect_recall"])
        for metrics in group_metrics.values()
        if metrics["first_actionable_defect_recall"] is not None
    ]
    flagged_trajectory_count = len(
        {step.trajectory_id for step in selected}
    )

    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "epistemic_status": "development_only",
        "review_unit": "eligible_assistant_message",
        "ranking_scope": "global_evaluation_surface_pool",
        "tie_breaker": (
            "score_desc_domain_group_trajectory_message_index_row_id"
        ),
        "budget_fraction": float(budget_fraction),
        "eligible_row_count": len(steps),
        "selected_row_count": budget,
        "trajectory_count": len(trajectories),
        "independent_group_count": len(
            {trajectory.group_id for trajectory in trajectories}
        ),
        "positive_trajectory_count": len(positive),
        "clean_trajectory_count": len(clean),
        "first_actionable_defect_recall": (
            first_defects_selected / len(positive) if positive else None
        ),
        "first_actionable_defect_selected_count": first_defects_selected,
        "clean_row_allocation": (
            clean_selected_rows / budget if budget else None
        ),
        "clean_selected_row_count": clean_selected_rows,
        "clean_trajectory_alert_rate": (
            len(clean_alerted) / len(clean) if clean else None
        ),
        "clean_alerted_trajectory_count": len(clean_alerted),
        "actionable_row_precision": (
            actionable_selected / budget if budget else None
        ),
        "actionable_selected_row_count": actionable_selected,
        "flagged_trajectory_count": flagged_trajectory_count,
        "selected_rows_per_flagged_trajectory": (
            budget / flagged_trajectory_count
            if flagged_trajectory_count
            else None
        ),
        "domain_macro_first_actionable_defect_recall": (
            sum(domain_recalls) / len(domain_recalls)
            if domain_recalls
            else None
        ),
        "group_macro_first_actionable_defect_recall": (
            sum(group_recalls) / len(group_recalls)
            if group_recalls
            else None
        ),
        "domain_metrics": domain_metrics,
        "group_metrics": group_metrics,
        "selected_row_ids": [step.row_id for step in selected],
    }


def compare_score_surfaces(
    trajectories: Sequence[TriageTrajectory],
    *,
    baseline_scores: Mapping[str, float],
    augmented_scores: Mapping[str, float],
    budget_fraction: float = 0.10,
) -> dict[str, Any]:
    baseline = evaluate_global_review_budget(
        trajectories,
        scores=baseline_scores,
        budget_fraction=budget_fraction,
    )
    augmented = evaluate_global_review_budget(
        trajectories,
        scores=augmented_scores,
        budget_fraction=budget_fraction,
    )

    def difference(field: str) -> float | None:
        left = baseline[field]
        right = augmented[field]
        if left is None or right is None:
            return None
        return float(right - left)

    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "epistemic_status": "development_only",
        "primary_comparison": "full_cheap_baseline_vs_baseline_plus_structural",
        "baseline": baseline,
        "augmented": augmented,
        "paired_point_differences": {
            "first_actionable_defect_recall": difference(
                "first_actionable_defect_recall"
            ),
            "clean_row_allocation": difference("clean_row_allocation"),
            "clean_trajectory_alert_rate": difference(
                "clean_trajectory_alert_rate"
            ),
            "actionable_row_precision": difference(
                "actionable_row_precision"
            ),
            "domain_macro_first_actionable_defect_recall": difference(
                "domain_macro_first_actionable_defect_recall"
            ),
            "group_macro_first_actionable_defect_recall": difference(
                "group_macro_first_actionable_defect_recall"
            ),
        },
        "uncertainty": {
            "status": "not_estimated_until_grouped_resampling_rule_is_frozen",
            "resampling_unit": "highest_independent_group",
        },
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ProcessTriageDevelopmentError(
            "cannot compute a quantile of an empty sequence"
        )
    if not 0.0 <= probability <= 1.0:
        raise ProcessTriageDevelopmentError(
            "quantile probability must lie in [0, 1]"
        )
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return float(
        ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
    )


def _bootstrap_choice(
    *,
    seed: str,
    replicate_index: int,
    domain: str,
    draw_index: int,
    population_size: int,
) -> int:
    digest = hashlib.sha256(
        _canonical_json(
            [
                BOOTSTRAP_RESAMPLING_ID,
                seed,
                replicate_index,
                domain,
                draw_index,
            ]
        ).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % population_size


def _domain_stratified_group_bootstrap_sample(
    trajectories: Sequence[TriageTrajectory],
    *,
    baseline_scores: Mapping[str, float],
    augmented_scores: Mapping[str, float],
    seed: str,
    replicate_index: int,
) -> tuple[
    tuple[TriageTrajectory, ...],
    dict[str, float],
    dict[str, float],
]:
    groups: dict[str, dict[str, list[TriageTrajectory]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for trajectory in trajectories:
        groups[trajectory.domain][trajectory.group_id].append(trajectory)

    cloned: list[TriageTrajectory] = []
    cloned_baseline: dict[str, float] = {}
    cloned_augmented: dict[str, float] = {}
    for domain, by_group in sorted(groups.items()):
        group_ids = sorted(by_group)
        for draw_index in range(len(group_ids)):
            selected_index = _bootstrap_choice(
                seed=seed,
                replicate_index=replicate_index,
                domain=domain,
                draw_index=draw_index,
                population_size=len(group_ids),
            )
            source_group_id = group_ids[selected_index]
            clone_group_id = (
                f"bootstrap:{replicate_index}:{domain}:{draw_index}:"
                f"{source_group_id}"
            )
            for trajectory_index, trajectory in enumerate(
                sorted(
                    by_group[source_group_id],
                    key=lambda row: row.trajectory_id,
                )
            ):
                clone_trajectory_id = (
                    f"{clone_group_id}:trajectory:{trajectory_index}:"
                    f"{trajectory.trajectory_id}"
                )
                clone_steps: list[TriageStep] = []
                for step_index, step in enumerate(trajectory.steps):
                    if (
                        step.row_id not in baseline_scores
                        or step.row_id not in augmented_scores
                    ):
                        raise ProcessTriageDevelopmentError(
                            f"bootstrap score missing for row {step.row_id}"
                        )
                    clone_row_id = (
                        f"{clone_trajectory_id}:row:{step_index}:"
                        f"{step.row_id}"
                    )
                    clone_steps.append(
                        replace(
                            step,
                            trajectory_id=clone_trajectory_id,
                            group_id=clone_group_id,
                            row_id=clone_row_id,
                        )
                    )
                    cloned_baseline[clone_row_id] = float(
                        baseline_scores[step.row_id]
                    )
                    cloned_augmented[clone_row_id] = float(
                        augmented_scores[step.row_id]
                    )
                cloned.append(
                    replace(
                        trajectory,
                        trajectory_id=clone_trajectory_id,
                        group_id=clone_group_id,
                        source_record_id=(
                            f"bootstrap:{replicate_index}:"
                            f"{trajectory.source_record_id}"
                        ),
                        steps=tuple(clone_steps),
                    )
                )
    return tuple(cloned), cloned_baseline, cloned_augmented


def paired_domain_group_bootstrap(
    trajectories: Sequence[TriageTrajectory],
    *,
    baseline_scores: Mapping[str, float],
    augmented_scores: Mapping[str, float],
    replicates: int,
    seed: str,
    budget_fraction: float = 0.10,
) -> dict[str, Any]:
    """Paired, domain-stratified highest-group bootstrap.

    Each replicate samples groups with replacement within domain, preserves all
    trajectories and rows in a sampled group, then recomputes the global
    ranking and review budget for both score surfaces on that same sample.
    """

    if replicates <= 0:
        raise ProcessTriageDevelopmentError(
            "bootstrap replicates must be positive"
        )
    point = compare_score_surfaces(
        trajectories,
        baseline_scores=baseline_scores,
        augmented_scores=augmented_scores,
        budget_fraction=budget_fraction,
    )
    fields = (
        "first_actionable_defect_recall",
        "clean_row_allocation",
        "clean_trajectory_alert_rate",
        "actionable_row_precision",
        "domain_macro_first_actionable_defect_recall",
        "group_macro_first_actionable_defect_recall",
    )
    replicate_differences: dict[str, list[float]] = {
        field: [] for field in fields
    }
    for replicate_index in range(replicates):
        sampled, sampled_baseline, sampled_augmented = (
            _domain_stratified_group_bootstrap_sample(
                trajectories,
                baseline_scores=baseline_scores,
                augmented_scores=augmented_scores,
                seed=seed,
                replicate_index=replicate_index,
            )
        )
        comparison = compare_score_surfaces(
            sampled,
            baseline_scores=sampled_baseline,
            augmented_scores=sampled_augmented,
            budget_fraction=budget_fraction,
        )
        for field in fields:
            value = comparison["paired_point_differences"][field]
            if value is not None:
                replicate_differences[field].append(float(value))

    intervals: dict[str, dict[str, Any]] = {}
    for field, values in replicate_differences.items():
        intervals[field] = {
            "defined_replicate_count": len(values),
            "bootstrap_mean": (
                sum(values) / len(values) if values else None
            ),
            "percentile_95_lower": (
                _quantile(values, 0.025) if values else None
            ),
            "percentile_95_upper": (
                _quantile(values, 0.975) if values else None
            ),
        }
    return {
        "schema_version": BOOTSTRAP_SCHEMA_VERSION,
        "epistemic_status": "development_only",
        "resampling_id": BOOTSTRAP_RESAMPLING_ID,
        "resampling_unit": "highest_independent_task_surface_group",
        "stratification": "domain",
        "pairing": (
            "baseline and augmented scores share every resampled group draw"
        ),
        "global_budget_recomputed_each_replicate": True,
        "replicates": int(replicates),
        "seed_receipt": hashlib.sha256(
            _canonical_json([BOOTSTRAP_RESAMPLING_ID, seed]).encode("utf-8")
        ).hexdigest(),
        "budget_fraction": float(budget_fraction),
        "point_comparison": point,
        "paired_percentile_intervals": intervals,
    }


def assess_operational_success(
    bootstrap_report: Mapping[str, Any],
    *,
    rule: OperationalSuccessRule,
) -> dict[str, Any]:
    """Apply an explicit candidate rule without upgrading epistemic status."""

    if bootstrap_report.get("schema_version") != BOOTSTRAP_SCHEMA_VERSION:
        raise ProcessTriageDevelopmentError(
            "success assessment requires a paired bootstrap report"
        )
    point = bootstrap_report["point_comparison"]["paired_point_differences"]
    intervals = bootstrap_report["paired_percentile_intervals"]
    recall_point = point["first_actionable_defect_recall"]
    recall_lower = intervals["first_actionable_defect_recall"][
        "percentile_95_lower"
    ]
    clean_upper = intervals["clean_row_allocation"][
        "percentile_95_upper"
    ]
    if recall_point is None or recall_lower is None or clean_upper is None:
        passed = False
        status = "not_evaluable"
    else:
        passed = bool(
            recall_point >= rule.minimum_point_recall_gain
            and recall_lower > rule.minimum_recall_ci_lower
            and clean_upper <= rule.maximum_clean_allocation_ci_upper
        )
        status = "pass" if passed else "fail"
    return {
        "epistemic_status": "development_rule_assessment",
        "rule_status": rule.status,
        "assessment_status": status,
        "passed": passed,
        "primary_estimand": (
            "baseline_plus_structural minus baseline_only absolute change in "
            "first-actionable-defect recall at a global eligible-row budget"
        ),
        "clean_burden_estimand": (
            "baseline_plus_structural minus baseline_only absolute change in "
            "the fraction of selected rows belonging to clean trajectories"
        ),
        "criteria": {
            "point_recall_gain_at_least": rule.minimum_point_recall_gain,
            "recall_ci_lower_strictly_above": rule.minimum_recall_ci_lower,
            "clean_allocation_ci_upper_at_most": (
                rule.maximum_clean_allocation_ci_upper
            ),
        },
        "observed": {
            "point_recall_gain": recall_point,
            "recall_ci_lower": recall_lower,
            "clean_allocation_ci_upper": clean_upper,
        },
    }


def leave_largest_group_out_sensitivity(
    trajectories: Sequence[TriageTrajectory],
    *,
    baseline_scores: Mapping[str, float],
    augmented_scores: Mapping[str, float],
    budget_fraction: float = 0.10,
) -> dict[str, Any]:
    """Recompute the comparison after omitting each maximum-size group."""

    group_counts = Counter(trajectory.group_id for trajectory in trajectories)
    if not group_counts:
        raise ProcessTriageDevelopmentError(
            "largest-group sensitivity requires trajectories"
        )
    maximum_count = max(group_counts.values())
    largest_group_ids = sorted(
        group_id
        for group_id, count in group_counts.items()
        if count == maximum_count
    )
    rows: list[dict[str, Any]] = []
    for omitted_group_id in largest_group_ids:
        subset = tuple(
            trajectory
            for trajectory in trajectories
            if trajectory.group_id != omitted_group_id
        )
        row_ids = {
            step.row_id for trajectory in subset for step in trajectory.steps
        }
        comparison = compare_score_surfaces(
            subset,
            baseline_scores={
                row_id: float(baseline_scores[row_id]) for row_id in row_ids
            },
            augmented_scores={
                row_id: float(augmented_scores[row_id]) for row_id in row_ids
            },
            budget_fraction=budget_fraction,
        )
        rows.append(
            {
                "omitted_group_id": omitted_group_id,
                "omitted_trajectory_count": maximum_count,
                "remaining_trajectory_count": len(subset),
                "paired_point_differences": comparison[
                    "paired_point_differences"
                ],
            }
        )
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "epistemic_status": "development_sensitivity",
        "maximum_group_trajectory_count": maximum_count,
        "largest_group_count": len(largest_group_ids),
        "rows": rows,
    }


def near_duplicate_group_manifest(
    trajectories: Sequence[TriageTrajectory],
    *,
    similarity_threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> dict[str, Any]:
    """Cluster visible task surfaces without consulting any outcome label.

    The rule is deliberately simple and reproducible: Unicode word-token sets,
    Jaccard similarity, a fixed threshold, then connected components within
    each domain.  Manual adjudication is not part of this rule.
    """

    if not 0.0 <= similarity_threshold <= 1.0:
        raise ProcessTriageDevelopmentError(
            "similarity_threshold must lie in [0, 1]"
        )
    group_surface: dict[str, str] = {}
    group_domain: dict[str, str] = {}
    for trajectory in trajectories:
        previous_surface = group_surface.setdefault(
            trajectory.group_id,
            trajectory.task_surface_text,
        )
        previous_domain = group_domain.setdefault(
            trajectory.group_id,
            trajectory.domain,
        )
        if previous_surface != trajectory.task_surface_text:
            raise ProcessTriageDevelopmentError(
                f"group has inconsistent task surfaces: {trajectory.group_id}"
            )
        if previous_domain != trajectory.domain:
            raise ProcessTriageDevelopmentError(
                f"group crosses domains: {trajectory.group_id}"
            )

    parent = {group_id: group_id for group_id in group_surface}

    def find(group_id: str) -> str:
        root = group_id
        while parent[root] != root:
            root = parent[root]
        while parent[group_id] != group_id:
            next_group = parent[group_id]
            parent[group_id] = root
            group_id = next_group
        return root

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        smaller, larger = sorted((left_root, right_root))
        parent[larger] = smaller

    candidate_pair_count = 0
    linked_pair_count = 0
    linked_pairs: list[dict[str, Any]] = []
    groups_by_domain: dict[str, list[str]] = defaultdict(list)
    for group_id, domain in group_domain.items():
        groups_by_domain[domain].append(group_id)
    for domain, group_ids in sorted(groups_by_domain.items()):
        ordered = sorted(group_ids)
        token_sets = {
            group_id: _token_set(group_surface[group_id])
            for group_id in ordered
        }
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                candidate_pair_count += 1
                left_tokens = token_sets[left]
                right_tokens = token_sets[right]
                union_tokens = left_tokens | right_tokens
                similarity = (
                    1.0
                    if not union_tokens
                    else len(left_tokens & right_tokens) / len(union_tokens)
                )
                if similarity >= similarity_threshold:
                    linked_pair_count += 1
                    union(left, right)
                    linked_pairs.append(
                        {
                            "domain": domain,
                            "left_group_id": left,
                            "right_group_id": right,
                            "similarity": float(similarity),
                        }
                    )

    components: dict[tuple[str, str], list[str]] = defaultdict(list)
    for group_id, domain in sorted(group_domain.items()):
        components[(domain, find(group_id))].append(group_id)

    aliases: dict[str, str] = {}
    component_rows: list[dict[str, Any]] = []
    for (domain, _), members in sorted(components.items()):
        ordered_members = sorted(members)
        component_digest = hashlib.sha256(
            _canonical_json(
                [
                    NEAR_DUPLICATE_SCHEMA_VERSION,
                    domain,
                    ordered_members,
                ]
            ).encode("utf-8")
        ).hexdigest()
        component_id = f"{domain}:near-task-surface:{component_digest}"
        for member in ordered_members:
            aliases[member] = component_id
        component_rows.append(
            {
                "component_id": component_id,
                "domain": domain,
                "member_group_ids": ordered_members,
                "member_group_count": len(ordered_members),
            }
        )

    return {
        "schema_version": NEAR_DUPLICATE_SCHEMA_VERSION,
        "epistemic_status": "feature_blind_grouping_manifest",
        "normalization_rule": (
            "canonical visible question + task description + data source; "
            "Unicode word tokens lowercased"
        ),
        "metric_id": NEAR_DUPLICATE_METRIC_ID,
        "similarity_threshold": float(similarity_threshold),
        "component_rule": "within_domain_connected_components",
        "exact_grouping_id": AGENT_PROCESS_BENCH_GROUPING_ID,
        "manual_adjudication": "prohibited",
        "input_group_count": len(group_surface),
        "component_count": len(component_rows),
        "candidate_pair_count": candidate_pair_count,
        "linked_pair_count": linked_pair_count,
        "multi_group_component_count": sum(
            row["member_group_count"] > 1 for row in component_rows
        ),
        "group_aliases": dict(sorted(aliases.items())),
        "components": component_rows,
        "linked_pairs": linked_pairs,
    }


def grouped_domain_split(
    trajectories: Sequence[TriageTrajectory],
    *,
    split_seed: str,
    development_groups_per_domain: int,
    group_aliases: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Create a deterministic domain-stratified split at the group level."""

    if development_groups_per_domain <= 0:
        raise ProcessTriageDevelopmentError(
            "development_groups_per_domain must be positive"
        )
    aliases = dict(group_aliases or {})
    groups_by_domain: dict[str, set[str]] = defaultdict(set)
    domain_by_group: dict[str, str] = {}
    for trajectory in trajectories:
        effective_group_id = aliases.get(
            trajectory.group_id,
            trajectory.group_id,
        )
        previous = domain_by_group.setdefault(
            effective_group_id, trajectory.domain
        )
        if previous != trajectory.domain:
            raise ProcessTriageDevelopmentError(
                f"group crosses domains: {effective_group_id}"
            )
        groups_by_domain[trajectory.domain].add(effective_group_id)

    development: list[str] = []
    locked: list[str] = []
    for domain, groups in sorted(groups_by_domain.items()):
        if len(groups) <= development_groups_per_domain:
            raise ProcessTriageDevelopmentError(
                f"domain {domain!r} has too few groups for a locked partition"
            )
        ordered = sorted(
            groups,
            key=lambda group_id: hashlib.sha256(
                _canonical_json(
                    [GROUP_SPLIT_ID, split_seed, domain, group_id]
                ).encode("utf-8")
            ).hexdigest(),
        )
        development.extend(ordered[:development_groups_per_domain])
        locked.extend(ordered[development_groups_per_domain:])

    development_set = set(development)
    locked_set = set(locked)
    if development_set & locked_set:
        raise ProcessTriageDevelopmentError("group split overlap detected")
    return {
        "split_id": GROUP_SPLIT_ID,
        "epistemic_status": "candidate_split_not_yet_frozen",
        "seed_receipt": hashlib.sha256(
            _canonical_json([GROUP_SPLIT_ID, split_seed]).encode("utf-8")
        ).hexdigest(),
        "development_group_ids": sorted(development),
        "locked_group_ids": sorted(locked),
        "development_group_count": len(development),
        "locked_group_count": len(locked),
        "development_trajectory_count": sum(
            aliases.get(trajectory.group_id, trajectory.group_id)
            in development_set
            for trajectory in trajectories
        ),
        "locked_trajectory_count": sum(
            aliases.get(trajectory.group_id, trajectory.group_id)
            in locked_set
            for trajectory in trajectories
        ),
        "group_alias_manifest_applied": bool(aliases),
    }


def dataset_admission_summary(
    trajectories: Sequence[TriageTrajectory],
) -> dict[str, Any]:
    by_domain: dict[str, list[TriageTrajectory]] = defaultdict(list)
    for trajectory in trajectories:
        by_domain[trajectory.domain].append(trajectory)

    def summarize(rows: Sequence[TriageTrajectory]) -> dict[str, Any]:
        return {
            "trajectory_count": len(rows),
            "independent_group_count": len({row.group_id for row in rows}),
            "eligible_row_count": sum(len(row.steps) for row in rows),
            "positive_trajectory_count": sum(not row.is_clean for row in rows),
            "clean_trajectory_count": sum(row.is_clean for row in rows),
            "negative_step_count": sum(
                step.actionable_defect for row in rows for step in row.steps
            ),
            "neutral_step_count": sum(
                step.native_label == 0 for row in rows for step in row.steps
            ),
            "positive_step_count": sum(
                step.native_label == 1 for row in rows for step in row.steps
            ),
            "source_slot_counts": {
                str(slot): count
                for slot, count in sorted(
                    Counter(row.source_slot for row in rows).items()
                )
            },
        }

    return {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "epistemic_status": "schema_and_admission_audit_only",
        "independent_unit": "domain_visible_task_surface_group",
        "grouping_id": AGENT_PROCESS_BENCH_GROUPING_ID,
        "label_mapping_id": AGENT_PROCESS_BENCH_MAPPING_ID,
        "source_policy_field": "sample_index_as_opaque_source_slot",
        "named_source_model_mapping_status": SOURCE_MODEL_MAPPING_STATUS,
        "named_source_model_interpretation_authorized": False,
        "named_leave_one_source_model_out_authorized": False,
        "pooled": summarize(trajectories),
        "domains": {
            domain: summarize(rows) for domain, rows in sorted(by_domain.items())
        },
    }


def subset_by_groups(
    trajectories: Sequence[TriageTrajectory],
    group_ids: Iterable[str],
    *,
    group_aliases: Mapping[str, str] | None = None,
) -> tuple[TriageTrajectory, ...]:
    allowed = set(group_ids)
    aliases = dict(group_aliases or {})
    return tuple(
        trajectory
        for trajectory in trajectories
        if aliases.get(trajectory.group_id, trajectory.group_id) in allowed
    )
