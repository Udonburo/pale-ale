"""Artifact-only, development-only localization of the frozen Track A A0 failure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .oracle import edited_trace, parity_trace
from .parse_phase2_output import parse_phase2_output
from .parse_register_output import parse_register_output

EXECUTION_ID = "e941e509-ab69-4965-85a2-f48a622d89b7"
LEDGER_SHA = "e045b8fa584c7afd74a6d979467f618f86f9664d4c170384cdc2a4c363235c88"
LEDGER_INTERNAL_SHA = "2e746d8d1cc1089e671a75a5f5585334953874e9979552eb6d3ba24ba30387fe"
EXTENSION_SHA = "7f99a653b4a96c4d6fbbf2a61d640c9a3971c45344b52b6bbeaf81b829ebe7f4"
CONDITION_ORDER = ("D", "S", "O", "F", "C", "E", "N")
STRUCTURED = frozenset("SOEN")
ROLES = {
    "D": "direct_final_answer",
    "S": "self_generated_structured_trace",
    "O": "oracle_state_prefix_continuation",
    "F": "length_matched_filler",
    "C": "corrupted_trace",
    "E": "authoritative_visible_edit",
    "N": "marker_only_no_overwrite",
}
REGISTER = re.compile(r"r(?P<step>0|[1-9][0-9]*) = (?P<value>[01])")
ANSWER = re.compile(r"answer = (?P<value>[01])")
PAIR_FIELDS = (
    "baseline_case_id", "edited_case_id", "marker_only_case_id",
    "authoritative_edited_register_value", "generated_edited_state_accepts_overwrite",
    "prefix_before_edit_unchanged_where_required",
    "every_post_edit_state_matches_edited_oracle",
    "final_generated_register_flips_from_baseline", "answer_flips_from_baseline",
    "edited_answer_matches_edited_generated_final_register",
    "edited_answer_matches_edited_oracle", "marker_only_tail_matches_base_oracle",
    "marker_only_answer_matches_base_oracle",
)
CSV_FIELDS = (
    "case_id", "condition", "matched_pair_id", "sequence_length", "grammar_shape",
    "edit_step", "edit_stratum", "control_type", "semantic_label", "bits",
    "oracle_state_sequence", "generated_state_sequence", "model_emitted_state_sequence",
    "authority_supplied_prefix_sequence", "emitted_step_indices", "state_step_provenance",
    "transition_law_consistency_at_every_step",
    "oracle_state_correctness_at_every_step", "first_transition_law_violation",
    "first_oracle_state_divergence", "number_of_transition_law_violations",
    "number_of_oracle_state_errors", "all_transitions_law_consistent",
    "all_generated_states_oracle_correct", "generated_final_register",
    "oracle_final_register", "generated_answer", "answer_equals_generated_final_register",
    "answer_equals_oracle_final_register",
    "generated_final_register_equals_oracle_final_register", "final_answer_correct",
    *PAIR_FIELDS, "raw_output_sha256", "parser",
    "parse_status", "parser_rejection_reason",
)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _json_sha(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha_bytes(raw.encode("utf-8"))


def _ratio(flags: Iterable[bool]) -> dict[str, int | float | None]:
    values = [bool(value) for value in flags]
    n = sum(values)
    d = len(values)
    return {"numerator": n, "denominator": d, "rate": n / d if d else None}


def _diff(left: Mapping[str, Any], right: Mapping[str, Any]) -> float | None:
    return None if left["rate"] is None or right["rate"] is None else left["rate"] - right["rate"]


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value)


def _raw_values(text: str) -> tuple[list[int], list[int], int]:
    lines = text.strip().splitlines()
    if len(lines) == 1 and lines[0] in "01":
        return [], [], int(lines[0])
    steps, values = [], []
    for line in lines[:-1]:
        match = REGISTER.fullmatch(line)
        if not match:
            raise ValueError(f"unexpected register line: {line!r}")
        steps.append(int(match["step"]))
        values.append(int(match["value"]))
    answer = ANSWER.fullmatch(lines[-1]) if lines else None
    if not answer:
        raise ValueError("unexpected answer line")
    return steps, values, int(answer["value"])


def _paths(root: Path) -> dict[str, Path]:
    phase2 = root / "analysis/gate13_causal_return/phase2"
    execution = (root / "workstream/local/gate13_causal_return_outputs/phase2"
                 / "modal_track_a_constrained/volume_snapshot/executions" / EXECUTION_ID)
    return {
        "ledger": root / "workstream/local/gate13_causal_return_outputs/register_case_ledger.json",
        "lock": phase2 / "phase2_a_lock.json",
        "extension": phase2 / "track_a_a0_extension_manifest.json",
        "channel": phase2 / "track_a_constrained_channel_manifest.json",
        "scientific_result": execution / "scientific_result.json",
        "terminal": execution / "terminal_state.json",
        "artifact_manifest": execution / "artifact_manifest.json",
        "execution": execution,
    }


def _authority(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    paths = _paths(root)
    if _sha(paths["ledger"]) != LEDGER_SHA or _sha(paths["extension"]) != EXTENSION_SHA:
        raise ValueError("frozen case-authority file SHA mismatch")
    lock, ledger, extension, channel = map(
        _json, (paths["lock"], paths["ledger"], paths["extension"], paths["channel"])
    )
    if (lock["case_manifests"]["A0_REVIEW1"]["internal_ledger_sha256"] != LEDGER_INTERNAL_SHA
            or lock["case_manifests"]["A0_EXTENSION"]["sha256"] != EXTENSION_SHA):
        raise ValueError("phase2_a_lock A0 binding mismatch")
    for payload, field in ((ledger, "ledger_sha256"), (extension, "manifest_sha256"),
                           (channel, "manifest_sha256")):
        declared = payload[field]
        if declared != _json_sha({key: value for key, value in payload.items() if key != field}):
            raise ValueError(f"internal authority hash mismatch: {field}")
    if ledger["ledger_sha256"] != LEDGER_INTERNAL_SHA:
        raise ValueError("Review-1 internal ledger hash mismatch")

    bindings = {row["case_id"]: row for row in channel["case_bindings"] if row["stage"] == "A0"}
    cases: dict[str, dict[str, Any]] = {}
    by_base: dict[tuple[str, str], dict[str, Any]] = {}
    for source in ledger["cases"]:
        row = dict(source)
        case_id, bits = row["case_id"], tuple(row["bits"])
        if tuple(row["base_trace"]) != parity_trace(bits) or tuple(row["edited_trace"]) != edited_trace(bits, row["edit_step"]):
            raise ValueError(f"ledger oracle mismatch: {case_id}")
        if _json_sha(row) != bindings[case_id]["source_case_sha256"]:
            raise ValueError(f"source case binding mismatch: {case_id}")
        if _sha_bytes(row["prompt"].encode()) != bindings[case_id]["prompt_sha256"]:
            raise ValueError(f"prompt binding mismatch: {case_id}")
        cases[case_id] = row
        by_base[(row["base_id"], row["condition"])] = row
    for source in extension["cases"]:
        row, base_id = dict(source), source["target_id"]
        matched = by_base[(base_id, "E")]
        if row["matched_to_case_id"] != matched["case_id"] or row["bits"] != matched["bits"]:
            raise ValueError(f"marker-only pairing mismatch: {row['case_id']}")
        if row["prompt_sha256"] != bindings[row["case_id"]]["prompt_sha256"]:
            raise ValueError(f"marker-only prompt binding mismatch: {row['case_id']}")
        row.update(base_id=base_id, base_trace=matched["base_trace"],
                   edited_trace=matched["edited_trace"], replicate=matched["replicate"])
        cases[row["case_id"]] = row
    if len(cases) != 252 or set(cases) != set(bindings):
        raise ValueError("A0 authority coverage mismatch")
    return cases, {"paths": paths, "bindings": bindings}


def _rows(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cases, context = _authority(root)
    paths, execution = context["paths"], context["paths"]["execution"]
    scientific, terminal, artifact_manifest = map(
        _json, (paths["scientific_result"], paths["terminal"], paths["artifact_manifest"])
    )
    if (terminal["execution_identity"] != EXECUTION_ID or terminal["terminal_status"] != "A0_FAIL"
            or terminal["MODEL_FORWARD_COUNT"] != 261 or scientific["A0"]["status"] != "FAIL"):
        raise ValueError("immutable terminal state mismatch")
    declared = {item["path"]: item for item in artifact_manifest["artifacts"]}
    rows, case_inventory, raw_inventory = [], [], []

    for case_id in sorted(cases):
        case, binding = cases[case_id], context["bindings"][case_id]
        condition, bits, step = case["condition"], list(case["bits"]), int(case["edit_step"])
        base, edited = list(case["base_trace"]), list(case["edited_trace"])
        oracle = edited if condition == "E" else base
        case_rel, raw_rel = f"cases/a0/{case_id}.json", f"raw/a0/{case_id}.txt"
        case_path, raw_path = execution / case_rel, execution / raw_rel
        for rel, path in ((case_rel, case_path), (raw_rel, raw_path)):
            if rel not in declared or _sha(path) != declared[rel]["sha256"]:
                raise ValueError(f"execution artifact hash mismatch: {rel}")
        record, raw = _json(case_path), raw_path.read_text(encoding="utf-8")
        if (record["case_id"] != case_id or record["parse_status"] != "PASS"
                or record["parse_error"] is not None or _sha_bytes(raw.encode()) != record["raw_output_sha256"]
                or record["instrument_trace"]["grammar_id"] != binding["grammar_id"]
                or record["instrument_trace"]["oracle_consulted"]
                or record["instrument_trace"]["transition_validity_filtered"]):
            raise ValueError(f"case record invariant failed: {case_id}")
        parsed = parse_phase2_output(case, raw) if condition == "N" else parse_register_output(case, raw)
        stored = record["parsed_record"]
        if condition == "N":
            if list(parsed.values) != stored["values"] or parsed.final_prediction != stored["final_prediction"]:
                raise ValueError(f"stored parse mismatch: {case_id}")
        elif ((None if parsed.trace_prediction is None else list(parsed.trace_prediction))
              != stored["trace_prediction"] or parsed.final_prediction != stored["final_prediction"]):
            raise ValueError(f"stored parse mismatch: {case_id}")

        emitted_steps, emitted_values, answer = _raw_values(raw)
        if condition == "S":
            generated, provenance, expected_steps = emitted_values, ["MODEL_EMITTED"] * len(emitted_values), list(range(len(bits) + 1))
        elif condition in STRUCTURED:
            prefix = (edited if condition == "E" else base)[: step + 1]
            generated = prefix + emitted_values
            provenance = ["AUTHORITY_SUPPLIED"] * (step + 1) + ["MODEL_EMITTED"] * len(emitted_values)
            if condition == "E":
                provenance[step] = "AUTHORITY_SUPPLIED_EDITED_OVERWRITE"
            expected_steps = list(range(step + 1, len(bits) + 1))
        else:
            generated, provenance, expected_steps = [], [], []
        if emitted_steps != expected_steps or (condition in STRUCTURED and len(generated) != len(bits) + 1):
            raise ValueError(f"emitted grammar mismatch: {case_id}")

        oracle_record = record["oracle_record"]
        expected_oracle_values = base[step + 1:] if condition == "N" else oracle if condition in STRUCTURED else None
        if condition == "N":
            valid_oracle = oracle_record["values"] == expected_oracle_values and oracle_record["final_prediction"] == base[-1]
        else:
            valid_oracle = (oracle_record["condition"] == condition
                            and oracle_record["trace_prediction"] == expected_oracle_values
                            and oracle_record["final_prediction"] == oracle[-1])
        if not valid_oracle:
            raise ValueError(f"stored oracle mismatch: {case_id}")

        transitions = None if condition not in STRUCTURED else {
            str(index): (None if condition == "E" and index == step
                         else generated[index] == (generated[index - 1] ^ bits[index - 1]))
            for index in range(1, len(generated))
        }
        state_correct = None if condition not in STRUCTURED else {
            str(index): generated[index] == oracle[index] for index in range(len(generated))
        }
        transition_errors = [] if transitions is None else [int(k) for k, v in transitions.items() if v is False]
        state_errors = [] if state_correct is None else [int(k) for k, v in state_correct.items() if not v]
        final_register = generated[-1] if generated else None
        row = {
            "case_id": case_id, "condition": condition, "matched_pair_id": case["base_id"],
            "sequence_length": len(bits), "grammar_shape": binding["grammar_id"],
            "edit_step": step if condition in "OEN" else None, "edit_stratum": case["edit_stratum"],
            "control_type": ROLES[condition], "semantic_label": int(case["semantic_answer"]), "bits": bits,
            "oracle_state_sequence": oracle if condition in STRUCTURED else None,
            "generated_state_sequence": generated if condition in STRUCTURED else None,
            "model_emitted_state_sequence": emitted_values if condition in STRUCTURED else None,
            "authority_supplied_prefix_sequence": generated[: emitted_steps[0]] if condition in STRUCTURED and emitted_steps else None,
            "emitted_step_indices": emitted_steps if condition in STRUCTURED else None,
            "state_step_provenance": provenance if condition in STRUCTURED else None,
            "transition_law_consistency_at_every_step": transitions,
            "oracle_state_correctness_at_every_step": state_correct,
            "first_transition_law_violation": min(transition_errors, default=None),
            "first_oracle_state_divergence": min(state_errors, default=None),
            "number_of_transition_law_violations": len(transition_errors) if transitions is not None else None,
            "number_of_oracle_state_errors": len(state_errors) if state_correct is not None else None,
            "all_transitions_law_consistent": not transition_errors if transitions is not None else None,
            "all_generated_states_oracle_correct": not state_errors if state_correct is not None else None,
            "generated_final_register": final_register, "oracle_final_register": oracle[-1],
            "generated_answer": answer,
            "answer_equals_generated_final_register": answer == final_register if final_register is not None else None,
            "answer_equals_oracle_final_register": answer == oracle[-1],
            "generated_final_register_equals_oracle_final_register": final_register == oracle[-1] if final_register is not None else None,
            "final_answer_correct": answer == oracle[-1],
            **dict.fromkeys(PAIR_FIELDS),
            "raw_output_sha256": record["raw_output_sha256"], "parser": record["parser"],
            "parse_status": record["parse_status"], "parser_rejection_reason": record["parse_error"],
            "_base": base, "_edited": edited,
        }
        rows.append(row)
        case_inventory.append({"path": case_rel, "bytes": case_path.stat().st_size, "sha256": _sha(case_path)})
        raw_inventory.append({"path": raw_rel, "bytes": raw_path.stat().st_size, "sha256": _sha(raw_path)})

    groups = _groups(rows, "matched_pair_id")
    for group_rows in groups.values():
        group = {row["condition"]: row for row in group_rows}
        s, e, n, step = group["S"], group["E"], group["N"], group["E"]["edit_step"]
        pair = {
            "baseline_case_id": s["case_id"], "edited_case_id": e["case_id"],
            "marker_only_case_id": n["case_id"], "authoritative_edited_register_value": e["_edited"][step],
            "generated_edited_state_accepts_overwrite": e["generated_state_sequence"][step + 1] == (e["_edited"][step] ^ e["bits"][step]),
            "prefix_before_edit_unchanged_where_required": e["generated_state_sequence"][:step] == s["_base"][:step],
            "every_post_edit_state_matches_edited_oracle": e["generated_state_sequence"][step:] == e["_edited"][step:],
            "final_generated_register_flips_from_baseline": e["generated_final_register"] != s["generated_final_register"],
            "answer_flips_from_baseline": e["generated_answer"] != s["generated_answer"],
            "edited_answer_matches_edited_generated_final_register": e["generated_answer"] == e["generated_final_register"],
            "edited_answer_matches_edited_oracle": e["generated_answer"] == e["_edited"][-1],
            "marker_only_tail_matches_base_oracle": n["generated_state_sequence"][step + 1:] == n["_base"][step + 1:],
            "marker_only_answer_matches_base_oracle": n["generated_answer"] == n["_base"][-1],
        }
        e.update(pair); n.update(pair)

    essential = [{"role": role, "path": path.relative_to(root).as_posix(),
                  "bytes": path.stat().st_size, "sha256": _sha(path)}
                 for role, path in paths.items() if role != "execution"]
    return rows, {"scientific": scientific, "terminal": terminal, "essential": essential,
                  "case_files": case_inventory, "raw_files": raw_inventory,
                  "artifact_count": artifact_manifest["artifact_count"],
                  "artifact_inventory_sha": artifact_manifest["inventory_sha256"]}


def _groups(rows: Sequence[Mapping[str, Any]], key: str) -> dict[Any, list[Mapping[str, Any]]]:
    result: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        result[row[key]].append(row)
    return result


def _condition_summaries(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for condition in CONDITION_ORDER:
        subset = [row for row in rows if row["condition"] == condition]
        slots = [row["oracle_state_correctness_at_every_step"][str(step)]
                 for row in subset for step in (row["emitted_step_indices"] or [])]
        laws = [row["transition_law_consistency_at_every_step"][str(step)]
                for row in subset for step in (row["emitted_step_indices"] or []) if step > 0
                and row["transition_law_consistency_at_every_step"][str(step)] is not None]
        structured = condition in STRUCTURED
        result[condition] = {
            "control_type": ROLES[condition], "case_count": len(subset),
            "label_balance": dict(sorted(Counter(row["semantic_label"] for row in subset).items())),
            "final_answer_accuracy": _ratio(row["final_answer_correct"] for row in subset),
            "emitted_state_slot_accuracy": _ratio(slots) if structured else None,
            "emitted_trajectory_exact_accuracy": _ratio(row["all_generated_states_oracle_correct"] for row in subset) if structured else None,
            "emitted_transition_law_accuracy": _ratio(laws) if structured else None,
            "answer_follows_generated_final_register": _ratio(row["answer_equals_generated_final_register"] for row in subset) if structured else None,
        }
    return result


def _trajectory(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    structured = [row for row in rows if row["condition"] in STRUCTURED]
    oracle_steps, law_steps = defaultdict(list), defaultdict(list)
    oracle_by_condition, law_by_condition = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    for row in structured:
        for step in row["emitted_step_indices"]:
            flag = row["oracle_state_correctness_at_every_step"][str(step)]
            oracle_steps[step].append(flag); oracle_by_condition[row["condition"]][step].append(flag)
            if step and row["transition_law_consistency_at_every_step"][str(step)] is not None:
                law = row["transition_law_consistency_at_every_step"][str(step)]
                law_steps[step].append(law); law_by_condition[row["condition"]][step].append(law)

    def step_map(source: Mapping[Any, Sequence[bool]]) -> dict[str, Any]:
        return {str(key): _ratio(source[key]) for key in sorted(source)}

    def distribution(field: str) -> dict[str, int]:
        values = Counter("NONE" if row[field] is None else str(row[field]) for row in structured)
        return {key: values[key] for key in sorted(values, key=lambda x: (-1 if x == "NONE" else int(x)))}

    def strata(field: str) -> dict[str, Any]:
        return {str(key): {"case_count": len(group),
                           "trajectory_exact_accuracy": _ratio(row["all_generated_states_oracle_correct"] for row in group),
                           "final_answer_accuracy": _ratio(row["final_answer_correct"] for row in group),
                           "oracle_state_error_count": sum(row["number_of_oracle_state_errors"] for row in group)}
                for key, group in sorted(_groups(structured, field).items(), key=lambda item: str(item[0]))}

    return {
        "scope": "MODEL_EMITTED_STATE_SLOTS_ONLY; supplied prefixes excluded from aggregate denominators",
        "transition_law_accuracy_by_step_position": step_map(law_steps),
        "oracle_state_accuracy_by_step_position": step_map(oracle_steps),
        "transition_law_accuracy_by_step_and_condition": {key: step_map(value) for key, value in sorted(law_by_condition.items())},
        "oracle_state_accuracy_by_step_and_condition": {key: step_map(value) for key, value in sorted(oracle_by_condition.items())},
        "first_oracle_error_position_distribution": distribution("first_oracle_state_divergence"),
        "first_transition_violation_position_distribution": distribution("first_transition_law_violation"),
        "number_of_oracle_errors_distribution": {str(k): v for k, v in sorted(Counter(row["number_of_oracle_state_errors"] for row in structured).items())},
        "accuracy_by_sequence_length": strata("sequence_length"),
        "accuracy_by_grammar_shape": strata("grammar_shape"),
    }


def _summaries(rows: Sequence[Mapping[str, Any]], scientific: Mapping[str, Any]) -> dict[str, Any]:
    conditions, trajectory = _condition_summaries(rows), _trajectory(rows)
    structured = [row for row in rows if row["condition"] in STRUCTURED]
    correct = [row for row in structured if row["all_generated_states_oracle_correct"]]
    wrong = [row for row in structured if not row["all_generated_states_oracle_correct"]]
    final = {
        "scope": "structured S/O/E/N cases only",
        "given_all_oracle_states_correct": {
            "final_answer_correct": _ratio(row["final_answer_correct"] for row in correct),
            "answer_equals_generated_final_register": _ratio(row["answer_equals_generated_final_register"] for row in correct)},
        "given_at_least_one_oracle_state_error": {
            "final_answer_correct": _ratio(row["final_answer_correct"] for row in wrong),
            "answer_equals_generated_final_register": _ratio(row["answer_equals_generated_final_register"] for row in wrong)},
        "answer_equals_oracle_final_register": _ratio(row["answer_equals_oracle_final_register"] for row in structured),
        "generated_final_register_equals_oracle_final_register": _ratio(row["generated_final_register_equals_oracle_final_register"] for row in structured),
    }

    by_pair = {key: {row["condition"]: row for row in group} for key, group in _groups(rows, "matched_pair_id").items()}
    strata, signs = {}, Counter()
    for key, group in sorted(_groups([row for row in rows if row["condition"] in "SO"], "sequence_length").items()):
        for edit, subgroup in sorted(_groups(group, "edit_stratum").items()):
            cell = {c: [row for row in subgroup if row["condition"] == c] for c in "SO"}
            sf, of = (_ratio(row["final_answer_correct"] for row in cell[c]) for c in "SO")
            se, oe = (_ratio(row["all_generated_states_oracle_correct"] for row in cell[c]) for c in "SO")
            delta = _diff(of, sf); signs["positive" if delta > 0 else "negative" if delta < 0 else "zero"] += 1
            strata[f"length={key}|edit_stratum={edit}"] = {
                "S_trajectory_exact": se, "O_continuation_exact": oe, "O_minus_S_trajectory_exact": _diff(oe, se),
                "S_final": sf, "O_final": of, "O_minus_S_final": delta}
    state_use = {
        "S": conditions["S"], "O": conditions["O"],
        "O_minus_S": {name: _diff(conditions["O"][left], conditions["S"][left])
                      for name, left in (("trajectory_exact", "emitted_trajectory_exact_accuracy"),
                                         ("emitted_state_slot_accuracy", "emitted_state_slot_accuracy"),
                                         ("final_answer_accuracy", "final_answer_accuracy"))},
        "paired_discordance": {
            "O_exact_S_not_exact": _ratio(g["O"]["all_generated_states_oracle_correct"] and not g["S"]["all_generated_states_oracle_correct"] for g in by_pair.values()),
            "S_exact_O_not_exact": _ratio(g["S"]["all_generated_states_oracle_correct"] and not g["O"]["all_generated_states_oracle_correct"] for g in by_pair.values())},
        "strata": strata, "stratum_final_difference_signs": dict(sorted(signs.items())),
    }

    pairs = list(by_pair.values())
    def cf_block(groups: Sequence[Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
        return {"pair_count": len(groups),
                "edited_answer_flip_rate": _ratio(g["E"]["answer_flips_from_baseline"] for g in groups),
                "edited_final_register_flip_rate": _ratio(g["E"]["final_generated_register_flips_from_baseline"] for g in groups),
                "edited_post_edit_oracle_trajectory_accuracy": _ratio(g["E"]["every_post_edit_state_matches_edited_oracle"] for g in groups),
                "edited_answer_correctness": _ratio(g["E"]["edited_answer_matches_edited_oracle"] for g in groups)}
    e_flip = _ratio(g["E"]["answer_flips_from_baseline"] for g in pairs)
    n_flip = _ratio(g["N"]["generated_answer"] != g["S"]["generated_answer"] for g in pairs)
    e_tail = _ratio(g["E"]["every_post_edit_state_matches_edited_oracle"] for g in pairs)
    n_base = _ratio(g["N"]["marker_only_tail_matches_base_oracle"] for g in pairs)
    n_edit = _ratio(g["N"]["generated_state_sequence"][g["N"]["edit_step"] + 1:] == g["E"]["_edited"][g["N"]["edit_step"] + 1:] for g in pairs)
    e_final, n_final = (_ratio(g[c]["final_answer_correct"] for g in pairs) for c in "EN")
    counterfactual = {
        "all_pairs": cf_block(pairs),
        "conditioned_on_baseline_trajectory_fully_correct": cf_block([g for g in pairs if g["S"]["all_generated_states_oracle_correct"]]),
        "conditioned_on_baseline_final_answer_correct": cf_block([g for g in pairs if g["S"]["final_answer_correct"]]),
        "conditioned_on_baseline_answer_following_generated_final_register": cf_block([g for g in pairs if g["S"]["answer_equals_generated_final_register"]]),
        "overwrite_successor_acceptance": _ratio(g["E"]["generated_edited_state_accepts_overwrite"] for g in pairs),
        "edited_vs_marker_only": {
            "edited_answer_flip_rate": e_flip, "marker_only_answer_flip_rate": n_flip,
            "edited_minus_marker_answer_flip": _diff(e_flip, n_flip),
            "edited_tail_matches_edited_oracle": e_tail, "marker_tail_matches_base_oracle": n_base,
            "edited_minus_marker_condition_appropriate_tail": _diff(e_tail, n_base),
            "marker_tail_matches_edited_oracle": n_edit, "edited_minus_marker_edited_tail": _diff(e_tail, n_edit),
            "edited_condition_appropriate_final_accuracy": e_final,
            "marker_condition_appropriate_final_accuracy": n_final,
            "edited_minus_marker_condition_appropriate_final": _diff(e_final, n_final)},
        "interpretation_boundary": "r_k is supplied; overwrite acceptance is measured at emitted r_(k+1).",
    }
    controls = {
        "structured_minus_filler_final_accuracy": _diff(conditions["S"]["final_answer_accuracy"], conditions["F"]["final_answer_accuracy"]),
        "structured_minus_corrupted_final_accuracy": _diff(conditions["S"]["final_answer_accuracy"], conditions["C"]["final_answer_accuracy"]),
        "structured_minus_direct_final_accuracy": _diff(conditions["S"]["final_answer_accuracy"], conditions["D"]["final_answer_accuracy"]),
        "predeclared_structured_control_difference_threshold": 0.20,
        "edited_minus_marker_only_effects": counterfactual["edited_vs_marker_only"],
    }
    return {"conditions": conditions, "trajectory": trajectory, "final": final, "state_use": state_use,
            "counterfactual": counterfactual, "controls": controls,
            "representatives": _representatives(rows, by_pair),
            "frozen_crosscheck": _frozen_crosscheck(rows, scientific)}


def _representatives(rows: Sequence[Mapping[str, Any]], pairs: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    structured = [row for row in rows if row["condition"] in STRUCTURED]
    categories = {
        "A_all_oracle_states_correct_wrong_answer": [r for r in structured if r["all_generated_states_oracle_correct"] and not r["final_answer_correct"]],
        "B_oracle_state_errors_answer_follows_generated_final": [r for r in structured if not r["all_generated_states_oracle_correct"] and r["answer_equals_generated_final_register"]],
        "C_visible_edit_accepted_answer_unchanged": [g["E"] for g in pairs.values() if g["E"]["generated_edited_state_accepts_overwrite"] and not g["E"]["answer_flips_from_baseline"]],
        "D_visible_edit_ignored_at_first_successor": [g["E"] for g in pairs.values() if g["E"]["generated_state_sequence"][g["E"]["edit_step"] + 1] == g["S"]["_base"][g["E"]["edit_step"] + 1]],
        "E_early_emitted_steps_correct_late_divergence": [r for r in structured if r["condition"] == "S" and r["first_oracle_state_divergence"] is not None and r["first_oracle_state_divergence"] > r["sequence_length"] // 2 and all(r["oracle_state_correctness_at_every_step"][str(s)] for s in r["emitted_step_indices"] if s < r["first_oracle_state_divergence"])],
        "F_transition_law_consistent_oracle_state_wrong": [r for r in structured if r["all_transitions_law_consistent"] and not r["all_generated_states_oracle_correct"]],
        "G_oracle_prefix_succeeds_self_generation_fails": [g["O"] for g in pairs.values() if g["O"]["all_generated_states_oracle_correct"] and not g["S"]["all_generated_states_oracle_correct"]],
    }
    return {name: {"evidence_count": len(group),
                   "representative_case_ids": [row["case_id"] for row in sorted(group, key=lambda r: r["case_id"])[:3]],
                   "representatives_are_descriptive_not_additional_evidence": True}
            for name, group in categories.items()}


def _frozen_crosscheck(rows: Sequence[Mapping[str, Any]], scientific: Mapping[str, Any]) -> dict[str, Any]:
    pairs = [{row["condition"]: row for row in group} for group in _groups(rows, "matched_pair_id").values()]
    step_rates, pair_flags, ineligible = [], [], 0
    for group in pairs:
        s, e, step = group["S"], group["E"], group["E"]["edit_step"]
        laws = [s["transition_law_consistency_at_every_step"][str(i)] for i in range(1, s["sequence_length"] + 1)]
        step_rates.append(sum(laws) / len(laws))
        if s["generated_state_sequence"][step:] != s["_base"][step:]:
            ineligible += 1
        else:
            pair_flags.append(all(e["generated_state_sequence"][i] == (s["generated_state_sequence"][i] ^ 1) for i in range(step, len(s["_base"]))))
    recomputed = {
        "A_step": sum(step_rates) / len(step_rates),
        "A_final": sum(g["S"]["final_answer_correct"] for g in pairs) / len(pairs),
        "filler_final": sum(g["F"]["final_answer_correct"] for g in pairs) / len(pairs),
        "corrupted_final": sum(g["C"]["final_answer_correct"] for g in pairs) / len(pairs),
        "oracle_prefix_continuation": sum(g["O"]["all_generated_states_oracle_correct"] for g in pairs) / len(pairs),
        "A_CF_oracle": sum(g["E"]["every_post_edit_state_matches_edited_oracle"] for g in pairs) / len(pairs),
        "A_final_CF": sum(g["E"]["final_answer_correct"] for g in pairs) / len(pairs),
        "marker_only_no_overwrite_accuracy": sum(g["N"]["marker_only_tail_matches_base_oracle"] for g in pairs) / len(pairs),
        "S_pair": sum(pair_flags) / len(pair_flags), "S_pair_eligible": len(pair_flags),
        "S_pair_ineligible": ineligible, "malformed_output_count": 0,
    }
    frozen = scientific["A0"]["metrics"]
    for key, value in recomputed.items():
        if (isinstance(value, float) and abs(value - frozen[key]) > 1e-12) or (not isinstance(value, float) and value != frozen[key]):
            raise ValueError(f"frozen metric mismatch: {key}")
    return {"status": "PASS_EXACT_ARTIFACT_REPRODUCTION_NO_GATE_REEVALUATION",
            "frozen_metrics": frozen, "recomputed_metrics": recomputed,
            "note": "A_step retains frozen equal-case weighting; pooled slot diagnostics expose denominators separately."}


def analyze(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    rows, inventory = _rows(root)
    summary = _summaries(rows, inventory["scientific"])
    result = {
        "schema_version": "gate13_track_a_a0_failure_localization_v1",
        "analysis_class": "DEVELOPMENT_ONLY_POSTMORTEM_NO_RESCORING", "execution_identity": EXECUTION_ID,
        "fixed_scientific_state": {"TRACK_A_CONSTRAINED_REGISTER_V1": "CLOSED_A0_FAIL",
            "TRACK_A_GENERAL_SCIENTIFIC_QUESTION": "OPEN", "A0": "FAIL", "A1": "UNOPENED_A0_FAIL",
            "A2": "UNOPENED_A0_FAIL", "A3": "CLOSED", "TRACK_C": "CLOSED",
            "FORMAL_GATE13": "CLOSED", "ACTIVATION_EXTRACTION": "CLOSED",
            "MODEL_FORWARD_COUNT": 261, "REMAINING_FORWARD_CEILING": 339, "NEW_MODEL_FORWARD_COUNT": 0},
        "input_coverage": {"A0_case_count": len(rows), "unique_case_id_count": len({r["case_id"] for r in rows}),
            "matched_pair_count": len({r["matched_pair_id"] for r in rows}),
            "condition_balance": dict(sorted(Counter(r["condition"] for r in rows).items())),
            "sequence_length_balance": dict(sorted(Counter(r["sequence_length"] for r in rows).items())),
            "semantic_label_balance": dict(sorted(Counter(r["semantic_label"] for r in rows).items())),
            "parser_status_balance": dict(sorted(Counter(r["parse_status"] for r in rows).items())),
            "raw_output_count": 252, "case_record_count": 252, "oracle_record_count": 252},
        "authority_gaps": [
            {"code": "AUTHORITY_GAP_N_RENDERED_PROMPT_BYTES", "scope": "marker-only N cases",
             "detail": "The frozen manifest binds prompt SHA and bytes but does not materialize prompt bytes; requested state and pair diagnostics remain covered."},
            {"code": "NOT_BEHAVIORALLY_OBSERVABLE_EDIT_STEP", "scope": "E overwrite r_k",
             "detail": "r_k is supplied, not emitted; acceptance is assessed only at emitted r_(k+1)."}],
        "condition_summaries": summary["conditions"], "trajectory_diagnostics": summary["trajectory"],
        "final_answer_diagnostics": summary["final"], "state_generation_versus_state_use": summary["state_use"],
        "counterfactual_diagnostics": summary["counterfactual"], "controls": summary["controls"],
        "representative_case_inspection": summary["representatives"],
        "frozen_A0_metric_crosscheck": summary["frozen_crosscheck"],
        "failure_classification": {"primary": "AUTOREGRESSIVE_OR_LENGTH_ACCUMULATION",
            "secondary": ["STATE_CONTINUATION_OR_USE_BOTTLENECK", "CONTROL_NONSELECTIVITY"],
            "successor_family": "stepwise / streaming finite-state transducer candidate",
            "successor_action": "RECOMMEND_ONLY_DO_NOT_IMPLEMENT_OR_EXECUTE",
            "rationale": {"length": summary["trajectory"]["accuracy_by_sequence_length"],
                "grammar_shape": summary["trajectory"]["accuracy_by_grammar_shape"],
                "O_minus_S": summary["state_use"]["O_minus_S"],
                "O_minus_S_stratum_signs": summary["state_use"]["stratum_final_difference_signs"],
                "controls": summary["controls"]}},
        "interpretation_constraints": {"predeclared_A0_gate_recomputed_or_changed": False,
            "A0_rescored": False, "missing_information_inferred": False,
            "new_instrument_designed_or_implemented": False, "new_model_forward": False},
    }
    return rows, result, inventory


def _fmt(value: Mapping[str, Any] | None) -> str:
    return "N/A" if not value or not value["denominator"] else f"{value['numerator']}/{value['denominator']} = {value['rate']:.4f}"


def _markdown(result: Mapping[str, Any]) -> str:
    c, t, s, cf, ctl = (result["condition_summaries"], result["trajectory_diagnostics"],
                        result["state_generation_versus_state_use"], result["counterfactual_diagnostics"], result["controls"])
    lines = ["# A0 Failure Localization", "", "Development-only; no rescue, rescoring, or reopening of A0.", "",
        "## Decision", "", "- Primary: `AUTOREGRESSIVE_OR_LENGTH_ACCUMULATION`",
        "- Secondary: `STATE_CONTINUATION_OR_USE_BOTTLENECK`, `CONTROL_NONSELECTIVITY`",
        "- Successor family: `stepwise / streaming finite-state transducer candidate` (recommendation only)",
        "- State: A0 FAIL; A1/A2 UNOPENED; A3/Track C/formal Gate13/activation extraction CLOSED",
        "- Forwards: cumulative 261; remaining 339; this slice 0", "", "## Coverage and boundary", "",
        f"252/252 cases from `{EXECUTION_ID}`: 36 each D/S/O/F/C/E/N, 36 matched identities, 252 raw/parsed/oracle records. Frozen A0 metrics reproduce exactly without gate reevaluation.", "",
        "Behavioral trajectory denominators use emitted slots only. Supplied O/E/N prefixes remain in the CSV with provenance. E's r_k is supplied, so overwrite acceptance is assessed at emitted r_(k+1). N prompt bytes are absent (`AUTHORITY_GAP`), while their frozen SHA/byte count and all needed semantics are present.", "",
        "## D/S/O/F/C/E/N", "", "|Condition|Final|Trajectory exact|State-slot|", "|---|---:|---:|---:|"]
    for condition in CONDITION_ORDER:
        row = c[condition]
        lines.append(f"|{condition}|{_fmt(row['final_answer_accuracy'])}|{_fmt(row['emitted_trajectory_exact_accuracy'])}|{_fmt(row['emitted_state_slot_accuracy'])}|")
    lines += ["", f"D→S final: {ctl['structured_minus_direct_final_accuracy']:+.4f}. S→O exact/final: {s['O_minus_S']['trajectory_exact']:+.4f}/{s['O_minus_S']['final_answer_accuracy']:+.4f}; O remains only {_fmt(c['O']['emitted_trajectory_exact_accuracy'])} exact, with stratum signs {s['stratum_final_difference_signs']}.", "",
        f"S−F and S−C final: {ctl['structured_minus_filler_final_accuracy']:+.4f}, {ctl['structured_minus_corrupted_final_accuracy']:+.4f}; both below frozen +0.20.", "", "## Accumulation and readout", "",
        "|Length|Trajectory exact|Final|", "|---:|---:|---:|"]
    for length, value in sorted(t["accuracy_by_sequence_length"].items(), key=lambda x: int(x[0])):
        lines.append(f"|{length}|{_fmt(value['trajectory_exact_accuracy'])}|{_fmt(value['final_answer_accuracy'])}|")
    shapes = t["accuracy_by_grammar_shape"]
    lines += ["", "Full-trace exactness declines: "
        f"r0..4 {_fmt(shapes['register-0-1-2-3-4']['trajectory_exact_accuracy'])}; "
        f"r0..8 {_fmt(shapes['register-0-1-2-3-4-5-6-7-8']['trajectory_exact_accuracy'])}; "
        f"r0..12 {_fmt(shapes['register-0-1-2-3-4-5-6-7-8-9-10-11-12']['trajectory_exact_accuracy'])}.", "",
        f"Answer follows generated final register for {_fmt(result['final_answer_diagnostics']['given_all_oracle_states_correct']['answer_equals_generated_final_register'])} correct and {_fmt(result['final_answer_diagnostics']['given_at_least_one_oracle_state_error']['answer_equals_generated_final_register'])} errored trajectories; final readout is not localized as primary.", "", "## Counterfactual", "",
        f"All 36 pairs: edit flip {_fmt(cf['all_pairs']['edited_answer_flip_rate'])}; successor acceptance {_fmt(cf['overwrite_successor_acceptance'])}; edited tail exact {_fmt(cf['all_pairs']['edited_post_edit_oracle_trajectory_accuracy'])}; edited answer correct {_fmt(cf['all_pairs']['edited_answer_correctness'])}.",
        f"With fully correct S baseline (n={cf['conditioned_on_baseline_trajectory_fully_correct']['pair_count']}): edit flip {_fmt(cf['conditioned_on_baseline_trajectory_fully_correct']['edited_answer_flip_rate'])}; edited tail exact {_fmt(cf['conditioned_on_baseline_trajectory_fully_correct']['edited_post_edit_oracle_trajectory_accuracy'])}. Edited−marker flip effect {cf['edited_vs_marker_only']['edited_minus_marker_answer_flip']:+.4f}.", "", "## Representative discordances", ""]
    for name, value in result["representative_case_inspection"].items():
        examples = ", ".join(f"`{x}`" for x in value["representative_case_ids"]) or "none"
        lines.append(f"- {name}: n={value['evidence_count']}; {examples}")
    lines += ["", "Examples are descriptive, lexicographic first up to three, and not extra evidence.", "",
              "## Stop", "", "No forward, Modal call, model download, successor implementation, or scientific change occurred. A0 remains FAIL; A1/A2 remain UNOPENED; A3/Track C remain CLOSED.", ""]
    return "\n".join(lines)


def write_outputs(root: Path, output: Path) -> dict[str, str]:
    rows, result, inventory = analyze(root)
    output.mkdir(parents=True, exist_ok=True)
    paths = {"md": output / "A0_FAILURE_LOCALIZATION.md", "json": output / "a0_failure_localization.json",
             "csv": output / "a0_case_table.csv", "inventory": output / "artifact_inventory.json"}
    with paths["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n"); writer.writeheader()
        for row in rows:
            writer.writerow({field: _cell(row.get(field)) for field in CSV_FIELDS})
    paths["json"].write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["md"].write_text(_markdown(result), encoding="utf-8")
    generated = {path.name: {"bytes": path.stat().st_size, "sha256": _sha(path)}
                 for path in (paths["md"], paths["json"], paths["csv"])}
    artifact_inventory = {"schema_version": "gate13_track_a_a0_failure_localization_artifact_inventory_v1",
        "execution_identity": EXECUTION_ID, "scope": "IMMUTABLE_A0_ARTIFACTS_ONLY_NO_MODEL_FORWARD",
        "input_essential_files": inventory["essential"],
        "input_case_artifacts": {"case_record_count": 252, "raw_output_count": 252,
            "all_hashes_verified_against_execution_artifact_manifest": True,
            "case_records": inventory["case_files"], "raw_outputs": inventory["raw_files"]},
        "execution_artifact_manifest": {"declared_artifact_count": inventory["artifact_count"],
            "inventory_sha256": inventory["artifact_inventory_sha"]},
        "generated_outputs": generated, "new_model_forward_count": 0}
    paths["inventory"].write_text(json.dumps(artifact_inventory, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {path.name: _sha(path) for path in paths.values()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("workstream/local/gate13_causal_return_outputs/development/a0_failure_localization"))
    args = parser.parse_args(argv); root = args.repo_root.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    hashes = write_outputs(root, output)
    print(json.dumps({"status": "PASS_DEVELOPMENT_ONLY_A0_FAILURE_LOCALIZATION", "A0_case_count": 252,
                      "execution_identity": EXECUTION_ID, "new_model_forward_count": 0,
                      "output_sha256": hashes}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
