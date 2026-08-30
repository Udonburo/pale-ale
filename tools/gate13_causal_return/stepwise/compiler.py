"""Deterministic ledgers and prompts for the stepwise successor campaign.

The compiler deliberately separates semantic state from rendered labels.  A
model call receives only the current visible label, the next action label, and
the frozen transition context.  It never receives the preceding input history.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


STARTING_COMMIT = "ab047352ff03cb0ce409664470cb633a9ea35ccc"
PRIOR_EXECUTION_ID = "e941e509-ab69-4965-85a2-f48a622d89b7"
PRIOR_CUMULATIVE_FORWARD_COUNT = 261

MODEL_REPOSITORY = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
TOKENIZER_REVISION = MODEL_REVISION

CAMPAIGN_FORWARD_CEILING = 650
CAMPAIGN_SPEND_CEILING_USD = 12.0
DEVELOPMENT_FORWARD_CEILING = 96
DEVELOPMENT_SPEND_CEILING_USD = 3.0
DEVELOPMENT_VARIANT_CEILING = 3
TRACK_A_QUALIFICATION_FORWARD_CEILING = 240
TRACK_B_QUALIFICATION_FORWARD_CEILING = 314

DEVELOPMENT_SEED_RANGE = (410_000, 410_999)
QUALIFICATION_SEED_RANGE = (730_000, 730_999)
TRACK_B_HALF_1_SEED_RANGE = (810_000, 810_999)
TRACK_B_HALF_2_SEED_RANGE = (910_000, 910_999)

VARIANT_IDS = (
    "compact_table_v1",
    "natural_rule_v1",
    "worked_cells_v1",
)

# Each spelling includes the exact leading-space byte sequence scored at the
# assistant's first next-token position.  The exact tokenizer validator binds
# every spelling to one token before any Qwen scientific response is requested.
OPAQUE_LABELS = (
    " Dak", " Zub", " Klo", " Pax",
    " Rex", " Jae", " Ook", " Bez",
    " Dex", " Hok", " Ksi", " Qed",
    " Raq", " Zuk", " Vox", " Ajax",
    " Akron", " Apex", " Baz", " Bij",
    " Byz", " Caj", " Cav", " Dek",
    " Dix", " Dok", " Eck", " Elk",
    " Esk", " Evo", " Ezra", " Fitz",
    " Fuj", " Gaz", " Giz", " Haj",
    " Hak", " Haz", " Hex", " Hij",
    " Ivy", " Jab", " Jad", " Jag",
    " Jed", " Joi", " Jou", " Kab",
    " Kad", " Kag", " Kah", " Kak",
    " Kal", " Kam", " Kan", " Kap",
    " Kas", " Kaw", " Kaz", " Kes",
    " Ket", " Kia", " Kir", " Kis",
    " Kob", " Kod", " Koh", " Kok",
    " Kol", " Kop", " Kor", " Kos",
    " Kot", " Kou", " Kov", " Kra",
    " Kre", " Kro", " Kub", " Kul",
)

BANK_SLICES = {
    "development": (0, 16),
    "qualification": (16, 48),
    "track_b_half_1": (48, 64),
    "track_b_half_2": (64, 80),
}


class StepwiseCompileError(ValueError):
    """Raised when a frozen stepwise ledger cannot be constructed exactly."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _bare(label: str) -> str:
    if not label.startswith(" ") or label.strip() != label[1:]:
        raise StepwiseCompileError(f"opaque label is not canonical: {label!r}")
    return label[1:]


@dataclass(frozen=True)
class Codebook:
    codebook_id: str
    bank: str
    state_labels: tuple[str, str]
    action_labels: tuple[str, str]

    def state(self, value: int) -> str:
        if value not in (0, 1):
            raise StepwiseCompileError("state must be binary")
        return self.state_labels[value]

    def action(self, value: int) -> str:
        if value not in (0, 1):
            raise StepwiseCompileError("action must be identity(0) or flip(1)")
        return self.action_labels[value]

    def as_json(self) -> dict[str, Any]:
        return {
            "codebook_id": self.codebook_id,
            "bank": self.bank,
            "state_labels": list(self.state_labels),
            "action_labels": list(self.action_labels),
        }


def codebook_bank(bank: str) -> tuple[Codebook, ...]:
    if bank not in BANK_SLICES:
        raise StepwiseCompileError(f"unknown codebook bank: {bank}")
    start, stop = BANK_SLICES[bank]
    labels = OPAQUE_LABELS[start:stop]
    if len(labels) % 4:
        raise StepwiseCompileError("codebook bank must contain groups of four labels")
    rows: list[Codebook] = []
    for index in range(0, len(labels), 4):
        group = list(labels[index : index + 4])
        rng = random.Random(_stable_seed("gate13-stepwise-codebook", bank, index // 4))
        rng.shuffle(group)
        rows.append(
            Codebook(
                codebook_id=f"{bank}-cb{index // 4:02d}",
                bank=bank,
                state_labels=(group[0], group[1]),
                action_labels=(group[2], group[3]),
            )
        )
    return tuple(rows)


def validate_codebook_partition() -> dict[str, Any]:
    seen: set[str] = set()
    banks: dict[str, Any] = {}
    for bank in BANK_SLICES:
        rows = codebook_bank(bank)
        labels = [label for row in rows for label in (*row.state_labels, *row.action_labels)]
        overlap = seen.intersection(labels)
        if overlap:
            raise StepwiseCompileError(f"codebook banks overlap: {sorted(overlap)!r}")
        seen.update(labels)
        banks[bank] = {
            "codebook_count": len(rows),
            "labels": labels,
            "sha256": sha256_json([row.as_json() for row in rows]),
        }
    return {"status": "PASS", "banks": banks, "total_unique_labels": len(seen)}


def validate_exact_tokenizer(tokenizer: Any) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for label in OPAQUE_LABELS:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if not isinstance(ids, list):
            ids = list(ids)
        if len(ids) != 1:
            raise StepwiseCompileError(f"opaque label is not one token: {label!r} -> {ids!r}")
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        if decoded != label:
            raise StepwiseCompileError(
                f"opaque label byte round-trip mismatch: {label!r} -> {decoded!r}"
            )
        rows.append({"label": label, "token_id": int(ids[0])})
    return {
        "status": "PASS",
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "label_count": len(rows),
        "labels": rows,
        "label_binding_sha256": sha256_json(rows),
        "partition": validate_codebook_partition(),
    }


def transition(state: int, action: int) -> int:
    if state not in (0, 1) or action not in (0, 1):
        raise StepwiseCompileError("external transition algebra is binary XOR")
    return state ^ action


def _transition_rows(codebook: Codebook) -> list[tuple[str, str, str]]:
    return [
        (
            _bare(codebook.state(state)),
            _bare(codebook.action(action)),
            _bare(codebook.state(transition(state, action))),
        )
        for state in (0, 1)
        for action in (0, 1)
    ]


def _demonstration_rows(
    codebook: Codebook,
    *,
    condition: str,
    demo_seed: int,
) -> list[tuple[str, str, str]]:
    correct = _transition_rows(codebook)
    order_rng = random.Random(_stable_seed("demo-order", demo_seed, codebook.codebook_id))
    order = list(range(len(correct)))
    order_rng.shuffle(order)
    if condition == "correct":
        return [correct[index] for index in order]
    outputs = [row[2] for row in correct]
    if condition == "label_shuffled":
        # A fixed non-identity permutation preserves all surface tokens and row
        # count while breaking the row-to-target association.
        output_order = [2, 0, 3, 1]
        rows = [
            (row[0], row[1], outputs[output_order[index]])
            for index, row in enumerate(correct)
        ]
        return [rows[index] for index in order]
    if condition == "corrupted":
        rows = [
            (row[0], row[1], _bare(codebook.state(1 - transition(state, action))))
            for row, (state, action) in zip(correct, ((0, 0), (0, 1), (1, 0), (1, 1)))
        ]
        return [rows[index] for index in order]
    if condition == "format_matched":
        rng = random.Random(_stable_seed("format-control", demo_seed, codebook.codebook_id))
        balanced = [outputs[0], outputs[1], outputs[1], outputs[0]]
        rng.shuffle(balanced)
        if balanced == outputs:
            balanced = balanced[1:] + balanced[:1]
        rows = [(row[0], row[1], balanced[index]) for index, row in enumerate(correct)]
        return [rows[index] for index in order]
    raise StepwiseCompileError(f"unknown demonstration condition: {condition}")


def _render_rows(rows: Sequence[tuple[str, str, str]], variant_id: str) -> str:
    if variant_id == "compact_table_v1":
        return "\n".join(f"{state} + {action} -> {target}" for state, action, target in rows)
    if variant_id == "natural_rule_v1":
        return "\n".join(
            f"From {state}, applying {action} gives {target}."
            for state, action, target in rows
        )
    if variant_id == "worked_cells_v1":
        return "\n".join(
            f"current={state}; action={action}; next={target}"
            for state, action, target in rows
        )
    raise StepwiseCompileError(f"unknown development variant: {variant_id}")


def render_step_prompt(
    *,
    variant_id: str,
    surface: str,
    codebook: Codebook,
    current_state: int,
    action: int,
    demonstration_condition: str = "correct",
    demo_seed: int = 0,
    intervention_marker: bool = False,
    template_flavor: int = 0,
    phase_index: int | None = None,
    broken_context: bool = False,
) -> str:
    """Render one call without any preceding input symbols or state history."""
    if variant_id not in VARIANT_IDS:
        raise StepwiseCompileError(f"unknown development variant: {variant_id}")
    state_names = ", ".join(_bare(label) for label in codebook.state_labels)
    action_names = ", ".join(_bare(label) for label in codebook.action_labels)
    if surface in {"STREAM-A0", "STREAM-A2"}:
        heading = (
            "Apply the displayed deterministic transition table."
            if template_flavor % 2 == 0
            else "Use this explicit state-update protocol."
        )
        rows = _transition_rows(codebook)
    elif surface == "TRACK-B":
        track_b_headings = (
            "Use these demonstrations as the frozen transition context.",
            "Infer the next state from this fixed demonstration block.",
            "Apply the transition behavior shown in these examples.",
            "Use the example rows to determine the next visible state.",
            "Follow the state-update behavior in this demonstration set.",
            "Determine the next state using only these worked transitions.",
            "Treat these examples as the complete transition context.",
            "Read the demonstrations and return the next visible state.",
        )
        heading = track_b_headings[template_flavor % len(track_b_headings)]
        rows = _demonstration_rows(
            codebook,
            condition="correct",
            demo_seed=demo_seed,
        )
    elif surface == "STREAM-A1":
        heading = (
            "Infer the state update only from the demonstrations."
            if template_flavor % 2 == 0
            else "Use the demonstrations to predict the next state."
        )
        rows = _demonstration_rows(
            codebook,
            condition=demonstration_condition,
            demo_seed=demo_seed,
        )
    else:
        raise StepwiseCompileError(f"unknown surface: {surface}")
    if phase_index not in (None, 0, 1):
        raise StepwiseCompileError("phase_index must be absent, zero, or one")
    if broken_context:
        if surface != "TRACK-B":
            raise StepwiseCompileError("broken context is confined to the Track B positive control")
        first = rows[0]
        replacement = _bare(
            codebook.state(1 - codebook.state_labels.index(" " + first[2]))
        )
        rows = [(first[0], first[1], replacement), *rows[1:]]
    marker = "\nIntervention marker: PRESENT." if intervention_marker else ""
    phase = "" if phase_index is None else f"\nProtocol phase: P{phase_index}."
    return (
        f"{heading}\n"
        f"Allowed state labels: {state_names}.\n"
        f"Allowed action labels: {action_names}.\n"
        f"{_render_rows(rows, variant_id)}{phase}{marker}\n"
        f"Current visible state: {_bare(codebook.state(current_state))}\n"
        f"Next input action: {_bare(codebook.action(action))}\n"
        "Return only the next state label."
    )


def prompt_contract_payload(variant_id: str) -> dict[str, Any]:
    placeholders = []
    sample_codebook = codebook_bank("development")[0]
    for surface in ("STREAM-A0", "STREAM-A1", "STREAM-A2", "TRACK-B"):
        conditions = (
            ("correct", "label_shuffled", "corrupted", "format_matched")
            if surface == "STREAM-A1"
            else ("correct",)
        )
        for condition in conditions:
            for current in (0, 1):
                for action in (0, 1):
                    placeholders.append(
                        {
                            "surface": surface,
                            "condition": condition,
                            "current": current,
                            "action": action,
                            "prompt": render_step_prompt(
                                variant_id=variant_id,
                                surface=surface,
                                codebook=sample_codebook,
                                current_state=current,
                                action=action,
                                demonstration_condition=condition,
                                demo_seed=17,
                                intervention_marker=surface == "STREAM-A2",
                            ),
                        }
                    )
    return {
        "variant_id": variant_id,
        "full_history_excluded": True,
        "model_input_fields": ["frozen transition context", "current visible state", "next input action"],
        "readout": "raw next-token forced-choice logits over both state-label tokens",
        "sampling": False,
        "renderings": placeholders,
        "sha256": sha256_json(placeholders),
    }


def _actions(seed: int, length: int) -> list[int]:
    rng = random.Random(_stable_seed("actions", seed))
    values = [rng.randrange(2) for _ in range(length)]
    if length >= 2 and len(set(values)) == 1:
        values[-1] = 1 - values[-1]
    return values


def compile_development_ledger(variant_id: str) -> dict[str, Any]:
    if variant_id not in VARIANT_IDS:
        raise StepwiseCompileError(f"unknown development variant: {variant_id}")
    codebooks = codebook_bank("development")
    seed = DEVELOPMENT_SEED_RANGE[0] + VARIANT_IDS.index(variant_id) * 100

    teacher = []
    for cb_index, codebook in enumerate(codebooks[:2]):
        for state in (0, 1):
            for action in (0, 1):
                teacher.append(
                    {
                        "forward_id": f"dev-{variant_id}-a0-tf-{cb_index}-{state}{action}",
                        "codebook_id": codebook.codebook_id,
                        "current_state": state,
                        "action": action,
                        "target_state": transition(state, action),
                        "transition_cell": f"{state}{action}",
                    }
                )
    rollout = {
        "episode_id": f"dev-{variant_id}-a0-rollout",
        "codebook_id": codebooks[2].codebook_id,
        "initial_state": seed % 2,
        "actions": _actions(seed + 1, 3),
    }
    a1 = []
    cells = ((0, 0), (0, 1), (1, 1))
    for condition in ("correct", "label_shuffled", "corrupted", "format_matched"):
        for index, (state, action) in enumerate(cells):
            a1.append(
                {
                    "forward_id": f"dev-{variant_id}-a1-{condition}-{index}",
                    "condition": condition,
                    "codebook_id": codebooks[3].codebook_id,
                    "current_state": state,
                    "action": action,
                    "target_state": transition(state, action),
                    "transition_cell": f"{state}{action}",
                    "demo_seed": seed + 10 + index,
                }
            )
    a2 = []
    for pair_index in range(2):
        a2.append(
            {
                "pair_id": f"dev-{variant_id}-a2-pair-{pair_index}",
                "codebook_id": codebooks[pair_index].codebook_id,
                "initial_state": (seed + pair_index) % 2,
                "pre_action": _actions(seed + 20 + pair_index, 1)[0],
                "future_actions": _actions(seed + 30 + pair_index, 2 if pair_index == 0 else 1),
            }
        )
    counts = {
        "teacher_forced": len(teacher),
        "self_fed_rollout": len(rollout["actions"]),
        "stream_a1": len(a1),
        "stream_a2": sum(1 + 2 * len(row["future_actions"]) for row in a2),
    }
    counts["total"] = sum(counts.values())
    if counts["total"] != 31:
        raise StepwiseCompileError(f"development ledger must contain 31 calls, got {counts}")
    return {
        "schema_version": "gate13_stepwise_development_ledger_v1",
        "variant_id": variant_id,
        "seed_range": list(DEVELOPMENT_SEED_RANGE),
        "qualification_seed_range_reserved_unopened": list(QUALIFICATION_SEED_RANGE),
        "teacher_forced": teacher,
        "self_fed_rollout": rollout,
        "stream_a1": a1,
        "stream_a2": a2,
        "forward_counts": counts,
        "sha256": sha256_json({"teacher": teacher, "rollout": rollout, "a1": a1, "a2": a2}),
    }


def compile_qualification_ledgers(variant_id: str) -> dict[str, Any]:
    """Materialize the fresh Track A surface only at simultaneous freeze time."""
    if variant_id not in VARIANT_IDS:
        raise StepwiseCompileError(f"unknown selected variant: {variant_id}")
    codebooks = codebook_bank("qualification")
    seed0 = QUALIFICATION_SEED_RANGE[0]
    teacher = []
    for cb_index, codebook in enumerate(codebooks):
        for state in (0, 1):
            for action in (0, 1):
                teacher.append(
                    {
                        "forward_id": f"qa0-tf-{cb_index:02d}-{state}{action}",
                        "codebook_id": codebook.codebook_id,
                        "current_state": state,
                        "action": action,
                        "target_state": transition(state, action),
                        "transition_cell": f"{state}{action}",
                    }
                )
    rollouts = []
    for index, codebook in enumerate(codebooks):
        length = 4 if index < 4 else 8
        rollouts.append(
            {
                "episode_id": f"qa0-rollout-{index:02d}",
                "codebook_id": codebook.codebook_id,
                "initial_state": (seed0 + index) % 2,
                "actions": _actions(seed0 + 100 + index, length),
                "sequence_length": length,
            }
        )
    a1 = []
    for cb_index, codebook in enumerate(codebooks[:4]):
        for condition in ("correct", "label_shuffled", "corrupted", "format_matched"):
            for state in (0, 1):
                for action in (0, 1):
                    a1.append(
                        {
                            "forward_id": f"qa1-{cb_index:02d}-{condition}-{state}{action}",
                            "condition": condition,
                            "codebook_id": codebook.codebook_id,
                            "current_state": state,
                            "action": action,
                            "target_state": transition(state, action),
                            "transition_cell": f"{state}{action}",
                            "demo_seed": seed0 + 200 + cb_index,
                        }
                    )
    a2 = []
    for index, codebook in enumerate(codebooks):
        a2.append(
            {
                "pair_id": f"qa2-pair-{index:02d}",
                "codebook_id": codebook.codebook_id,
                "initial_state": (seed0 + 300 + index) % 2,
                "pre_action": _actions(seed0 + 400 + index, 1)[0],
                "future_actions": _actions(seed0 + 500 + index, 5),
                "edit_after_pre_steps": 1,
            }
        )
    counts = {
        "STREAM-A0": len(teacher) + sum(len(row["actions"]) for row in rollouts),
        "STREAM-A1": len(a1),
        "STREAM-A2": sum(1 + 2 * len(row["future_actions"]) for row in a2),
    }
    counts["maximum_conditional_total"] = sum(counts.values())
    if counts != {
        "STREAM-A0": 80,
        "STREAM-A1": 64,
        "STREAM-A2": 88,
        "maximum_conditional_total": 232,
    }:
        raise StepwiseCompileError(f"qualification count drift: {counts}")
    if counts["maximum_conditional_total"] > TRACK_A_QUALIFICATION_FORWARD_CEILING:
        raise StepwiseCompileError("Track A qualification ledger exceeds budget")
    payload = {
        "schema_version": "gate13_stepwise_track_a_qualification_ledgers_v1",
        "variant_id": variant_id,
        "seed_range": list(QUALIFICATION_SEED_RANGE),
        "teacher_forced": teacher,
        "self_fed_rollouts": rollouts,
        "stream_a1": a1,
        "stream_a2": a2,
        "forward_counts": counts,
    }
    return {**payload, "sha256": sha256_json(payload)}


def compile_track_b_collection_ledger(variant_id: str) -> dict[str, Any]:
    """Compile independent real-sample halves for the prospective B lock."""
    if variant_id not in VARIANT_IDS:
        raise StepwiseCompileError(f"unknown selected variant: {variant_id}")
    nodes = ("phase0_state0", "phase0_state1", "phase1_state0", "phase1_state1", "phase1_state1_broken")
    halves = []
    for half_id, bank, seed_range in (
        ("half_1", "track_b_half_1", TRACK_B_HALF_1_SEED_RANGE),
        ("half_2", "track_b_half_2", TRACK_B_HALF_2_SEED_RANGE),
    ):
        codebooks = codebook_bank(bank)
        samples = []
        for sample_index in range(24):
            codebook = codebooks[sample_index % len(codebooks)]
            samples.append(
                {
                    "sample_id": f"b-{half_id}-{sample_index:02d}",
                    "episode_seed": seed_range[0] + sample_index,
                    "codebook_id": codebook.codebook_id,
                    "template_id": f"{half_id}-template-{sample_index % 4}",
                    "demonstration_instance_id": f"{half_id}-demo-{sample_index:02d}",
                    "node_ids": list(nodes),
                }
            )
        halves.append(
            {
                "half_id": half_id,
                "codebook_bank": bank,
                "seed_range": list(seed_range),
                "samples": samples,
            }
        )
    count = sum(len(half["samples"]) * len(nodes) for half in halves)
    if count != 240 or count > TRACK_B_QUALIFICATION_FORWARD_CEILING:
        raise StepwiseCompileError(f"Track B collection count drift: {count}")
    payload = {
        "schema_version": "gate13_stepwise_track_b_collection_ledger_v1",
        "variant_id": variant_id,
        "nodes": list(nodes),
        "halves": halves,
        "forward_count": count,
        "independence": {
            "opaque_codebooks": "DISJOINT",
            "templates": "DISJOINT",
            "demonstration_instances": "DISJOINT",
            "episode_seeds": "DISJOINT",
            "bootstrap_primary": False,
        },
    }
    return {**payload, "sha256": sha256_json(payload)}


def iter_all_codebooks() -> Iterable[Codebook]:
    for bank in BANK_SLICES:
        yield from codebook_bank(bank)


def codebook_lookup(bank: str | None = None) -> dict[str, Codebook]:
    rows = codebook_bank(bank) if bank is not None else tuple(iter_all_codebooks())
    return {row.codebook_id: row for row in rows}
