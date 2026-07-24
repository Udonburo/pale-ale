#!/usr/bin/env python3
"""Development evaluator for retrospective artifact-level process triage.

The evaluator fixes the mechanics that must exist before a structural signal
is developed:

* an AgentProcessBench adapter with query-level grouping;
* a canonical trajectory/assistant-step schema;
* reproducible cheap baseline features;
* a global eligible-row review budget;
* deterministic tie handling; and
* first-defect recall plus clean-trajectory review burden.

No structural signal is defined here.  This is development infrastructure, not
a held-out benchmark claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


TRAJECTORY_SCHEMA_VERSION = "pale_ale_process_trajectory_v0.1"
FEATURE_SCHEMA_VERSION = "pale_ale_process_cheap_features_v0.1"
EVALUATION_SCHEMA_VERSION = "pale_ale_global_review_budget_v0.1"
AGENT_PROCESS_BENCH_MAPPING_ID = "agent_process_bench_negative_step_v0.1"
AGENT_PROCESS_BENCH_GROUPING_ID = (
    "agent_process_bench_domain_task_surface_sha256_v0.1"
)
GROUP_SPLIT_ID = "sha256_domain_group_order_v0.1"

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
class CheapStepFeatures:
    row_id: str
    trajectory_id: str
    group_id: str
    domain: str
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


def cheap_features(
    trajectories: Sequence[TriageTrajectory],
) -> tuple[CheapStepFeatures, ...]:
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
    scores: dict[str, float] = {}
    for feature in features:
        score = sum(
            float(weight) * float(getattr(feature, name))
            for name, weight in weights.items()
        )
        score += float(type_weights.get(feature.artifact_type, 0.0))
        if not math.isfinite(score):
            raise ProcessTriageDevelopmentError(
                f"non-finite score for row {feature.row_id}"
            )
        scores[feature.row_id] = float(score)
    return scores


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

    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "epistemic_status": "development_only",
        "review_unit": "eligible_assistant_message",
        "ranking_scope": "global_locked_surface_pool",
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
        },
        "uncertainty": {
            "status": "not_estimated_until_grouped_resampling_rule_is_frozen",
            "resampling_unit": "highest_independent_group",
        },
    }


def grouped_domain_split(
    trajectories: Sequence[TriageTrajectory],
    *,
    split_seed: str,
    development_groups_per_domain: int,
) -> dict[str, Any]:
    """Create a deterministic domain-stratified split at the group level."""

    if development_groups_per_domain <= 0:
        raise ProcessTriageDevelopmentError(
            "development_groups_per_domain must be positive"
        )
    groups_by_domain: dict[str, set[str]] = defaultdict(set)
    domain_by_group: dict[str, str] = {}
    for trajectory in trajectories:
        previous = domain_by_group.setdefault(
            trajectory.group_id, trajectory.domain
        )
        if previous != trajectory.domain:
            raise ProcessTriageDevelopmentError(
                f"group crosses domains: {trajectory.group_id}"
            )
        groups_by_domain[trajectory.domain].add(trajectory.group_id)

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
            trajectory.group_id in development_set for trajectory in trajectories
        ),
        "locked_trajectory_count": sum(
            trajectory.group_id in locked_set for trajectory in trajectories
        ),
    }


def dataset_admission_summary(
    trajectories: Sequence[TriageTrajectory],
) -> dict[str, Any]:
    by_domain: dict[str, list[TriageTrajectory]] = defaultdict(list)
    for trajectory in trajectories:
        by_domain[trajectory.domain].append(trajectory)

    def summarize(rows: Sequence[TriageTrajectory]) -> dict[str, int]:
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
        }

    return {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "epistemic_status": "schema_and_admission_audit_only",
        "independent_unit": "domain_visible_task_surface_group",
        "grouping_id": AGENT_PROCESS_BENCH_GROUPING_ID,
        "label_mapping_id": AGENT_PROCESS_BENCH_MAPPING_ID,
        "pooled": summarize(trajectories),
        "domains": {
            domain: summarize(rows) for domain, rows in sorted(by_domain.items())
        },
    }


def subset_by_groups(
    trajectories: Sequence[TriageTrajectory],
    group_ids: Iterable[str],
) -> tuple[TriageTrajectory, ...]:
    allowed = set(group_ids)
    return tuple(
        trajectory
        for trajectory in trajectories
        if trajectory.group_id in allowed
    )
