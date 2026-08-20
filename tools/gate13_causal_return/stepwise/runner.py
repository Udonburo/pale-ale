"""Resume-safe execution and scoring for the stepwise Track A successor."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .compiler import (
    Codebook,
    StepwiseCompileError,
    codebook_lookup,
    compile_development_ledger,
    compile_qualification_ledgers,
    render_step_prompt,
    sha256_json,
    transition,
)


DEVELOPMENT_THRESHOLDS = {
    "one_step_accuracy_min": 0.90,
    "minimum_transition_cell_accuracy_min": 0.80,
    "correct_minus_strongest_control_min": 0.20,
    "visible_edit_immediate_successor_accuracy_min": 0.85,
    "marker_only_false_change_rate_max": 0.10,
    # A stricter prospective safeguard permitted by the campaign authority.
    "self_fed_rollout_exact_accuracy_min": 0.75,
}

QUALIFICATION_THRESHOLDS = {
    "STREAM-A0": {
        "one_step_accuracy_min": 0.90,
        "minimum_transition_cell_accuracy_min": 0.80,
        "self_fed_rollout_exact_accuracy_min": 0.75,
    },
    "STREAM-A1": {
        "correct_demonstration_accuracy_min": 0.85,
        "correct_minus_strongest_control_min": 0.20,
        "minimum_correct_transition_cell_accuracy_min": 0.75,
        "control_independent_qualification_ceiling": 0.85,
    },
    "STREAM-A2": {
        "edited_immediate_successor_accuracy_min": 0.90,
        "edited_downstream_step_accuracy_min": 0.75,
        "paired_final_state_flip_rate_min": 0.80,
        "marker_only_no_change_accuracy_min": 0.90,
    },
}


class StepwiseRunError(RuntimeError):
    """Fail-closed runner error."""


class Probe(Protocol):
    def __call__(
        self,
        prompt: str,
        candidate_labels: tuple[str, str],
        metadata: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(value), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise StepwiseRunError(f"non-object JSONL row at {path}:{line_number}")
        rows.append(value)
    return rows


@dataclass
class JsonlJournal:
    root: Path
    probe: Probe
    persist: Callable[[], None] = lambda: None

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.attempt_path = self.root / "forward_attempts.jsonl"
        self.response_path = self.root / "forward_responses.jsonl"
        attempts = _load_jsonl(self.attempt_path)
        responses = _load_jsonl(self.response_path)
        attempt_ids = [str(row["forward_id"]) for row in attempts]
        response_ids = [str(row["forward_id"]) for row in responses]
        if len(attempt_ids) != len(set(attempt_ids)):
            raise StepwiseRunError("a forward_id has multiple attempt records")
        if len(response_ids) != len(set(response_ids)):
            raise StepwiseRunError("a forward_id has multiple response records")
        if not set(response_ids).issubset(attempt_ids):
            raise StepwiseRunError("response exists without a prior attempt record")
        self._attempts = {str(row["forward_id"]): row for row in attempts}
        self._responses = {str(row["forward_id"]): row for row in responses}
        self.new_forward_count = 0

    @property
    def total_attempt_count(self) -> int:
        return len(self._attempts)

    @property
    def total_response_count(self) -> int:
        return len(self._responses)

    def query(
        self,
        *,
        forward_id: str,
        prompt: str,
        codebook: Codebook,
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        candidates = tuple(codebook.state_labels)
        binding = {
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "candidate_labels": list(candidates),
            "metadata_sha256": sha256_json(metadata),
        }
        if forward_id in self._responses:
            response = dict(self._responses[forward_id])
            if any(response.get(key) != value for key, value in binding.items()):
                raise StepwiseRunError(f"resume binding mismatch for {forward_id}")
            response["resumed"] = True
            response["actual_forward_this_entry"] = False
            return response
        if forward_id in self._attempts:
            raise StepwiseRunError(
                f"AMBIGUOUS_FORWARD_ATTEMPT_WITHOUT_RESPONSE:{forward_id}"
            )
        attempt = {
            "schema_version": "gate13_stepwise_forward_attempt_v1",
            "forward_id": forward_id,
            **binding,
        }
        _append_jsonl(self.attempt_path, attempt)
        self._attempts[forward_id] = attempt
        self.persist()
        raw = dict(self.probe(prompt, candidates, metadata))
        predicted_label = str(raw.get("predicted_label"))
        if predicted_label not in candidates:
            raise StepwiseRunError(
                f"probe returned a label outside the forced choice for {forward_id}"
            )
        response = {
            "schema_version": "gate13_stepwise_forward_response_v1",
            "forward_id": forward_id,
            **binding,
            **raw,
            "predicted_label": predicted_label,
        }
        _append_jsonl(self.response_path, response)
        self._responses[forward_id] = response
        self.new_forward_count += 1
        self.persist()
        return {**response, "resumed": False, "actual_forward_this_entry": True}


def _mean(values: Sequence[bool | float | int]) -> float:
    if not values:
        raise StepwiseRunError("a declared metric has an empty denominator")
    return sum(float(value) for value in values) / len(values)


def _prediction_state(response: Mapping[str, Any], codebook: Codebook) -> int:
    label = str(response["predicted_label"])
    if label == codebook.state_labels[0]:
        return 0
    if label == codebook.state_labels[1]:
        return 1
    raise StepwiseRunError("response label is outside codebook")


def _query_step(
    journal: JsonlJournal,
    *,
    forward_id: str,
    variant_id: str,
    surface: str,
    codebook: Codebook,
    current_state: int,
    action: int,
    target_state: int,
    condition: str = "correct",
    demo_seed: int = 0,
    marker: bool = False,
    template_flavor: int = 0,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    prompt = render_step_prompt(
        variant_id=variant_id,
        surface=surface,
        codebook=codebook,
        current_state=current_state,
        action=action,
        demonstration_condition=condition,
        demo_seed=demo_seed,
        intervention_marker=marker,
        template_flavor=template_flavor,
    )
    metadata = {
        "surface": surface,
        "condition": condition,
        "codebook_id": codebook.codebook_id,
        "current_visible_state": current_state,
        "action": action,
        "target_state": target_state,
        "full_history_excluded": True,
        **dict(extra or {}),
    }
    response = journal.query(
        forward_id=forward_id,
        prompt=prompt,
        codebook=codebook,
        metadata=metadata,
    )
    predicted = _prediction_state(response, codebook)
    return {
        **metadata,
        "forward_id": forward_id,
        "prompt_sha256": response["prompt_sha256"],
        "predicted_state": predicted,
        "predicted_label": response["predicted_label"],
        "correct": predicted == target_state,
        "probe": {
            key: value
            for key, value in response.items()
            if key
            not in {
                "schema_version",
                "forward_id",
                "prompt_sha256",
                "candidate_labels",
                "metadata_sha256",
                "predicted_label",
            }
        },
    }


def _run_a0(
    journal: JsonlJournal,
    *,
    variant_id: str,
    teacher_rows: Sequence[Mapping[str, Any]],
    rollouts: Sequence[Mapping[str, Any]],
    codebooks: Mapping[str, Codebook],
) -> dict[str, Any]:
    teacher_results = []
    for row in teacher_rows:
        codebook = codebooks[str(row["codebook_id"])]
        teacher_results.append(
            _query_step(
                journal,
                forward_id=str(row["forward_id"]),
                variant_id=variant_id,
                surface="STREAM-A0",
                codebook=codebook,
                current_state=int(row["current_state"]),
                action=int(row["action"]),
                target_state=int(row["target_state"]),
                extra={"transition_cell": row["transition_cell"], "mode": "teacher_forced"},
            )
        )
    rollout_results = []
    rollout_exact = []
    for episode in rollouts:
        codebook = codebooks[str(episode["codebook_id"])]
        visible = int(episode["initial_state"])
        oracle = visible
        episode_rows = []
        for step_index, action_value in enumerate(episode["actions"]):
            action = int(action_value)
            local_target = transition(visible, action)
            oracle = transition(oracle, action)
            result = _query_step(
                journal,
                forward_id=f"{episode['episode_id']}-step-{step_index:02d}",
                variant_id=variant_id,
                surface="STREAM-A0",
                codebook=codebook,
                current_state=visible,
                action=action,
                target_state=oracle,
                extra={
                    "mode": "self_fed",
                    "episode_id": episode["episode_id"],
                    "step_index": step_index,
                    "sequence_length": len(episode["actions"]),
                    "local_transition_target": local_target,
                },
            )
            result["local_transition_correct"] = result["predicted_state"] == local_target
            visible = int(result["predicted_state"])
            episode_rows.append(result)
        rollout_results.extend(episode_rows)
        rollout_exact.append(all(bool(row["correct"]) for row in episode_rows))
    cell_accuracy = {}
    for cell in ("00", "01", "10", "11"):
        values = [row["correct"] for row in teacher_results if row["transition_cell"] == cell]
        cell_accuracy[cell] = {"numerator": sum(values), "denominator": len(values), "accuracy": _mean(values)}
    metrics = {
        "one_step_accuracy": _mean([row["correct"] for row in teacher_results]),
        "one_step_denominator": len(teacher_results),
        "transition_cell_accuracy": cell_accuracy,
        "minimum_transition_cell_accuracy": min(row["accuracy"] for row in cell_accuracy.values()),
        "self_fed_rollout_exact_accuracy": _mean(rollout_exact),
        "self_fed_rollout_denominator": len(rollout_exact),
        "self_fed_step_oracle_accuracy": _mean([row["correct"] for row in rollout_results]),
        "self_fed_step_denominator": len(rollout_results),
        "self_fed_local_transition_accuracy": _mean(
            [row["local_transition_correct"] for row in rollout_results]
        ),
    }
    return {"teacher_forced": teacher_results, "self_fed": rollout_results, "metrics": metrics}


def _run_a1(
    journal: JsonlJournal,
    *,
    variant_id: str,
    rows: Sequence[Mapping[str, Any]],
    codebooks: Mapping[str, Codebook],
) -> dict[str, Any]:
    results = []
    for row in rows:
        codebook = codebooks[str(row["codebook_id"])]
        results.append(
            _query_step(
                journal,
                forward_id=str(row["forward_id"]),
                variant_id=variant_id,
                surface="STREAM-A1",
                codebook=codebook,
                current_state=int(row["current_state"]),
                action=int(row["action"]),
                target_state=int(row["target_state"]),
                condition=str(row["condition"]),
                demo_seed=int(row["demo_seed"]),
                extra={"transition_cell": row["transition_cell"], "mode": "demonstration"},
            )
        )
    by_condition: dict[str, Any] = {}
    for condition in ("correct", "label_shuffled", "corrupted", "format_matched"):
        selected = [row for row in results if row["condition"] == condition]
        by_condition[condition] = {
            "numerator": sum(bool(row["correct"]) for row in selected),
            "denominator": len(selected),
            "accuracy": _mean([row["correct"] for row in selected]),
        }
    correct_cells = {}
    for cell in ("00", "01", "10", "11"):
        selected = [
            row for row in results if row["condition"] == "correct" and row["transition_cell"] == cell
        ]
        if selected:
            correct_cells[cell] = {
                "numerator": sum(bool(row["correct"]) for row in selected),
                "denominator": len(selected),
                "accuracy": _mean([row["correct"] for row in selected]),
            }
    strongest_control = max(
        by_condition[name]["accuracy"]
        for name in ("label_shuffled", "corrupted", "format_matched")
    )
    metrics = {
        "by_condition": by_condition,
        "correct_minus_strongest_control": by_condition["correct"]["accuracy"] - strongest_control,
        "strongest_control_accuracy": strongest_control,
        "correct_transition_cell_accuracy": correct_cells,
        "minimum_correct_transition_cell_accuracy": min(
            row["accuracy"] for row in correct_cells.values()
        ),
    }
    return {"rows": results, "metrics": metrics}


def _run_a2(
    journal: JsonlJournal,
    *,
    variant_id: str,
    pairs: Sequence[Mapping[str, Any]],
    codebooks: Mapping[str, Codebook],
) -> dict[str, Any]:
    pair_results = []
    for pair in pairs:
        codebook = codebooks[str(pair["codebook_id"])]
        pre_target = transition(int(pair["initial_state"]), int(pair["pre_action"]))
        pre = _query_step(
            journal,
            forward_id=f"{pair['pair_id']}-pre-00",
            variant_id=variant_id,
            surface="STREAM-A2",
            codebook=codebook,
            current_state=int(pair["initial_state"]),
            action=int(pair["pre_action"]),
            target_state=pre_target,
            marker=False,
            extra={"pair_id": pair["pair_id"], "branch": "shared_pre", "step_index": 0},
        )
        marker_visible = int(pre["predicted_state"])
        edited_visible = 1 - marker_visible
        marker_oracle = marker_visible
        edited_oracle = edited_visible
        edited_rows = []
        marker_rows = []
        for future_index, action_value in enumerate(pair["future_actions"]):
            action = int(action_value)
            edited_oracle = transition(edited_oracle, action)
            marker_oracle = transition(marker_oracle, action)
            edited = _query_step(
                journal,
                forward_id=f"{pair['pair_id']}-edited-{future_index:02d}",
                variant_id=variant_id,
                surface="STREAM-A2",
                codebook=codebook,
                current_state=edited_visible,
                action=action,
                target_state=edited_oracle,
                marker=True,
                extra={
                    "pair_id": pair["pair_id"],
                    "branch": "edited",
                    "post_edit_step_index": future_index,
                    "authoritative_visible_overwrite": 1 - marker_visible if future_index == 0 else None,
                },
            )
            marker_row = _query_step(
                journal,
                forward_id=f"{pair['pair_id']}-marker-{future_index:02d}",
                variant_id=variant_id,
                surface="STREAM-A2",
                codebook=codebook,
                current_state=marker_visible,
                action=action,
                target_state=marker_oracle,
                marker=True,
                extra={
                    "pair_id": pair["pair_id"],
                    "branch": "marker_only",
                    "post_edit_step_index": future_index,
                },
            )
            edited["counterfactual_marker_oracle"] = marker_oracle
            marker_row["counterfactual_edited_oracle"] = edited_oracle
            marker_row["false_change"] = marker_row["predicted_state"] == edited_oracle
            edited_visible = int(edited["predicted_state"])
            marker_visible = int(marker_row["predicted_state"])
            edited_rows.append(edited)
            marker_rows.append(marker_row)
        pair_results.append(
            {
                "pair_id": pair["pair_id"],
                "shared_pre": pre,
                "edited": edited_rows,
                "marker_only": marker_rows,
                "final_state_flip": edited_rows[-1]["predicted_state"] != marker_rows[-1]["predicted_state"],
            }
        )
    edited_all = [row for pair in pair_results for row in pair["edited"]]
    edited_immediate = [pair["edited"][0] for pair in pair_results]
    edited_downstream = [row for pair in pair_results for row in pair["edited"][1:]]
    marker_all = [row for pair in pair_results for row in pair["marker_only"]]
    metrics = {
        "edited_immediate_successor_accuracy": _mean([row["correct"] for row in edited_immediate]),
        "edited_immediate_denominator": len(edited_immediate),
        "edited_all_post_edit_step_accuracy": _mean([row["correct"] for row in edited_all]),
        "edited_downstream_step_accuracy": (
            _mean([row["correct"] for row in edited_downstream]) if edited_downstream else 1.0
        ),
        "edited_downstream_denominator": len(edited_downstream),
        "paired_final_state_flip_rate": _mean([pair["final_state_flip"] for pair in pair_results]),
        "paired_final_state_flip_denominator": len(pair_results),
        "marker_only_no_change_accuracy": _mean([row["correct"] for row in marker_all]),
        "marker_only_denominator": len(marker_all),
        "marker_only_false_change_rate": _mean([row["false_change"] for row in marker_all]),
    }
    return {"pairs": pair_results, "metrics": metrics}


def _development_pass(a0: Mapping[str, Any], a1: Mapping[str, Any], a2: Mapping[str, Any]) -> bool:
    m0 = a0["metrics"]
    m1 = a1["metrics"]
    m2 = a2["metrics"]
    t = DEVELOPMENT_THRESHOLDS
    return bool(
        m0["one_step_accuracy"] >= t["one_step_accuracy_min"]
        and m0["minimum_transition_cell_accuracy"] >= t["minimum_transition_cell_accuracy_min"]
        and m0["self_fed_rollout_exact_accuracy"] >= t["self_fed_rollout_exact_accuracy_min"]
        and m1["correct_minus_strongest_control"] >= t["correct_minus_strongest_control_min"]
        and m2["edited_immediate_successor_accuracy"]
        >= t["visible_edit_immediate_successor_accuracy_min"]
        and m2["marker_only_false_change_rate"] <= t["marker_only_false_change_rate_max"]
    )


def run_development_variant(journal: JsonlJournal, variant_id: str) -> dict[str, Any]:
    ledger = compile_development_ledger(variant_id)
    codebooks = codebook_lookup("development")
    a0 = _run_a0(
        journal,
        variant_id=variant_id,
        teacher_rows=ledger["teacher_forced"],
        rollouts=[ledger["self_fed_rollout"]],
        codebooks=codebooks,
    )
    a1 = _run_a1(
        journal,
        variant_id=variant_id,
        rows=ledger["stream_a1"],
        codebooks=codebooks,
    )
    a2 = _run_a2(
        journal,
        variant_id=variant_id,
        pairs=ledger["stream_a2"],
        codebooks=codebooks,
    )
    return {
        "schema_version": "gate13_stepwise_development_result_v1",
        "variant_id": variant_id,
        "ledger_sha256": ledger["sha256"],
        "thresholds": DEVELOPMENT_THRESHOLDS,
        "STREAM-A0": a0,
        "STREAM-A1": a1,
        "STREAM-A2": a2,
        "selection_eligible": _development_pass(a0, a1, a2),
        "actual_new_forward_count": journal.new_forward_count,
    }


def _a0_qualification_pass(result: Mapping[str, Any]) -> bool:
    metrics = result["metrics"]
    t = QUALIFICATION_THRESHOLDS["STREAM-A0"]
    return bool(
        metrics["one_step_accuracy"] >= t["one_step_accuracy_min"]
        and metrics["minimum_transition_cell_accuracy"] >= t["minimum_transition_cell_accuracy_min"]
        and metrics["self_fed_rollout_exact_accuracy"] >= t["self_fed_rollout_exact_accuracy_min"]
    )


def _a1_qualification_pass(result: Mapping[str, Any]) -> bool:
    metrics = result["metrics"]
    by_condition = metrics["by_condition"]
    t = QUALIFICATION_THRESHOLDS["STREAM-A1"]
    return bool(
        by_condition["correct"]["accuracy"] >= t["correct_demonstration_accuracy_min"]
        and metrics["correct_minus_strongest_control"] >= t["correct_minus_strongest_control_min"]
        and metrics["minimum_correct_transition_cell_accuracy"]
        >= t["minimum_correct_transition_cell_accuracy_min"]
        and all(
            by_condition[name]["accuracy"] < t["control_independent_qualification_ceiling"]
            for name in ("label_shuffled", "corrupted", "format_matched")
        )
    )


def _a2_qualification_pass(result: Mapping[str, Any]) -> bool:
    metrics = result["metrics"]
    t = QUALIFICATION_THRESHOLDS["STREAM-A2"]
    return bool(
        metrics["edited_immediate_successor_accuracy"]
        >= t["edited_immediate_successor_accuracy_min"]
        and metrics["edited_downstream_step_accuracy"]
        >= t["edited_downstream_step_accuracy_min"]
        and metrics["paired_final_state_flip_rate"] >= t["paired_final_state_flip_rate_min"]
        and metrics["marker_only_no_change_accuracy"] >= t["marker_only_no_change_accuracy_min"]
    )


def run_track_a_qualification(journal: JsonlJournal, variant_id: str) -> dict[str, Any]:
    ledger = compile_qualification_ledgers(variant_id)
    codebooks = codebook_lookup("qualification")
    a0 = _run_a0(
        journal,
        variant_id=variant_id,
        teacher_rows=ledger["teacher_forced"],
        rollouts=ledger["self_fed_rollouts"],
        codebooks=codebooks,
    )
    a0_pass = _a0_qualification_pass(a0)
    result: dict[str, Any] = {
        "schema_version": "gate13_stepwise_track_a_qualification_result_v1",
        "variant_id": variant_id,
        "ledger_sha256": ledger["sha256"],
        "thresholds": QUALIFICATION_THRESHOLDS,
        "STREAM-A0": {**a0, "status": "PASS" if a0_pass else "FAIL"},
        "STREAM-A1": {"status": "UNOPENED_STREAM_A0_FAIL"},
        "STREAM-A2": {"status": "UNOPENED_STREAM_A0_FAIL"},
    }
    if not a0_pass:
        result["terminal_track_a_status"] = "FAIL"
        result["actual_new_forward_count"] = journal.new_forward_count
        return result
    a1 = _run_a1(
        journal,
        variant_id=variant_id,
        rows=ledger["stream_a1"],
        codebooks=codebooks,
    )
    a1_pass = _a1_qualification_pass(a1)
    result["STREAM-A1"] = {**a1, "status": "PASS" if a1_pass else "FAIL"}
    result["STREAM-A2"] = {"status": "UNOPENED_STREAM_A1_FAIL"}
    if not a1_pass:
        result["terminal_track_a_status"] = "FAIL"
        result["actual_new_forward_count"] = journal.new_forward_count
        return result
    a2 = _run_a2(
        journal,
        variant_id=variant_id,
        pairs=ledger["stream_a2"],
        codebooks=codebooks,
    )
    a2_pass = _a2_qualification_pass(a2)
    result["STREAM-A2"] = {**a2, "status": "PASS" if a2_pass else "FAIL"}
    result["terminal_track_a_status"] = "PASS" if a2_pass else "FAIL"
    result["actual_new_forward_count"] = journal.new_forward_count
    return result


def write_result(path: Path, result: Mapping[str, Any]) -> None:
    _atomic_write(path, json.dumps(dict(result), ensure_ascii=False, indent=2, sort_keys=True) + "\n")

