"""Compile deterministic syntax-channel bindings without model or oracle access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.gate13_causal_return.phase2_common import (
    sha256_bytes,
    sha256_file,
    sha256_json,
    write_json,
)

from .compile_phase2_cases import compile_cases
from .compile_register_cases import compile_ledger
from .constrained_channel import syntax_for_case


M1_CASE_IDS = (
    "a0-l12-y0-early-r0-S",
    "a0-l12-y0-early-r0-O",
    "a0-l12-y0-early-r0-E",
    "a0-l12-y0-early-r0-N",
)


def legacy_free_generation_max_new_tokens_for_case(case: Mapping[str, Any]) -> int:
    """Reproduce the terminated free-generation channel's length heuristic."""

    return max(32, len(str(case["expected_text"])) // 2 + 32)


def all_scientific_cases() -> dict[str, list[dict[str, Any]]]:
    compiled = compile_cases()
    return {
        "A0": [dict(row, stage="A0") for row in compile_ledger()["cases"]]
        + [dict(row) for row in compiled["A0_EXTENSION"]],
        "A1": [dict(row, stage="A1") for row in compiled["A1"]],
        "A2": [dict(row, stage="A2") for row in compiled["A2"]],
    }


def _source_case_for_hash(case: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(case)
    # Review-1 A0 predates the explicit stage field.  Preserve its already-bound
    # source bytes instead of normalizing the scientific case.
    if value.get("stage") == "A0" and value.get("condition") != "N":
        value.pop("stage", None)
    return value


def compile_channel_manifest(*, phase2_dir: Path) -> dict[str, Any]:
    stages = all_scientific_cases()
    shape_rows: dict[str, dict[str, Any]] = {}
    case_rows: list[dict[str, Any]] = []
    for stage in ("A0", "A1", "A2"):
        for case in stages[stage]:
            syntax = syntax_for_case(case)
            shape_rows.setdefault(
                syntax.grammar_id,
                {
                    "grammar_id": syntax.grammar_id,
                    "direct_answer": syntax.direct_answer,
                    "expected_steps": list(syntax.expected_steps),
                    "semantic_slot_count": syntax.semantic_slot_count,
                    "assignment_count": syntax.assignment_count,
                    "fixed_components": [
                        component.literal if component.kind == "literal" else "<BIT:0|1>"
                        for component in syntax.components
                    ],
                },
            )
            source_case = _source_case_for_hash(case)
            case_rows.append(
                {
                    "stage": stage,
                    "case_id": str(case["case_id"]),
                    "condition": str(case.get("condition") or ""),
                    "source_case_sha256": sha256_json(source_case),
                    "prompt_sha256": sha256_bytes(str(case["prompt"]).encode("utf-8")),
                    "grammar_id": syntax.grammar_id,
                    "semantic_slot_count": syntax.semantic_slot_count,
                    "legacy_free_generation_max_new_tokens": (
                        legacy_free_generation_max_new_tokens_for_case(case)
                    ),
                    "constrained_max_new_tokens_policy": (
                        "exact canonical syntax content token count plus one terminal EOS"
                    ),
                }
            )
    payload: dict[str, Any] = {
        "schema_version": "gate13_track_a_constrained_channel_manifest_v1",
        "output_channel": "SYNTAX_CONSTRAINED_REGISTER_DFA",
        "semantic_branch_policy": "INDEPENDENT_0_OR_1_AT_EVERY_REGISTER_AND_ANSWER_SLOT",
        "oracle_access": "FORBIDDEN_AND_ABSENT",
        "transition_filtering": "FORBIDDEN_AND_ABSENT",
        "answer_equality_filtering": "FORBIDDEN_AND_ABSENT",
        "eos_policy": "EOS_ONLY_AFTER_FINAL_DECLARED_BIT_SLOT",
        "case_counts": {stage: len(rows) for stage, rows in stages.items()},
        "grammar_shape_count": len(shape_rows),
        "grammar_shapes": [shape_rows[key] for key in sorted(shape_rows)],
        "case_bindings": case_rows,
        "source_artifacts": {
            "phase2_a_lock_sha256": sha256_file(phase2_dir / "phase2_a_lock.json"),
            "a0_extension_manifest_sha256": sha256_file(
                phase2_dir / "track_a_a0_extension_manifest.json"
            ),
            "a1_manifest_sha256": sha256_file(phase2_dir / "track_a_a1_manifest.json"),
            "a2_manifest_sha256": sha256_file(phase2_dir / "track_a_a2_manifest.json"),
        },
    }
    payload["manifest_sha256"] = sha256_json(payload)
    return payload


def compile_m1_manifest(channel_manifest: Mapping[str, Any]) -> dict[str, Any]:
    bindings = {
        str(row["case_id"]): row for row in channel_manifest["case_bindings"]
    }
    cases = []
    for ordinal, case_id in enumerate(M1_CASE_IDS, start=1):
        row = bindings[case_id]
        cases.append(
            {
                "ordinal": ordinal,
                "case_id": case_id,
                "stage": "M1_DEVELOPMENT_PREFLIGHT",
                "source_stage": "A0",
                "condition": row["condition"],
                "source_case_sha256": row["source_case_sha256"],
                "prompt_sha256": row["prompt_sha256"],
                "grammar_id": row["grammar_id"],
                "legacy_free_generation_max_new_tokens": row[
                    "legacy_free_generation_max_new_tokens"
                ],
                "constrained_max_new_tokens_policy": row[
                    "constrained_max_new_tokens_policy"
                ],
                "contributes_to_a0_a1_a2_metrics": False,
                "reused_as_a0_checkpoint": False,
            }
        )
    payload: dict[str, Any] = {
        "schema_version": "gate13_track_a_constrained_m1_manifest_v1",
        "selection_rule": (
            "same frozen S/O/E/N maximum-length surfaces used for prior preflight; "
            "development-only outputs are isolated from every scientific metric"
        ),
        "case_count": len(cases),
        "development_case_level_forward_count": len(cases),
        "scientific_metric_contribution_count": 0,
        "reuse_as_a0_checkpoint_count": 0,
        "cases": cases,
        "channel_manifest_sha256": channel_manifest["manifest_sha256"],
    }
    payload["manifest_sha256"] = sha256_json(payload)
    return payload


def validate_compiled_manifests(
    channel_manifest: Mapping[str, Any], m1_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    if channel_manifest.get("manifest_sha256") != sha256_json(
        {key: value for key, value in channel_manifest.items() if key != "manifest_sha256"}
    ):
        raise ValueError("constrained channel manifest internal SHA mismatch")
    if m1_manifest.get("manifest_sha256") != sha256_json(
        {key: value for key, value in m1_manifest.items() if key != "manifest_sha256"}
    ):
        raise ValueError("constrained M1 manifest internal SHA mismatch")
    expected_counts = {"A0": 252, "A1": 162, "A2": 90}
    if channel_manifest.get("case_counts") != expected_counts:
        raise ValueError("constrained channel scientific case counts drifted")
    rows = list(channel_manifest.get("case_bindings") or [])
    ids = [str(row.get("case_id") or "") for row in rows]
    if len(ids) != sum(expected_counts.values()) or len(ids) != len(set(ids)):
        raise ValueError("constrained channel case bindings are incomplete or duplicated")
    m1_rows = list(m1_manifest.get("cases") or [])
    if [row.get("case_id") for row in m1_rows] != list(M1_CASE_IDS):
        raise ValueError("constrained M1 case selection drifted")
    if any(
        row.get("contributes_to_a0_a1_a2_metrics") is not False
        or row.get("reused_as_a0_checkpoint") is not False
        for row in m1_rows
    ):
        raise ValueError("constrained M1 leaked into scientific metrics or checkpoints")
    return {
        "schema_version": "gate13_track_a_constrained_manifest_validation_v1",
        "status": "PASS",
        "scientific_case_count": len(ids),
        "grammar_shape_count": int(channel_manifest["grammar_shape_count"]),
        "m1_development_case_count": len(m1_rows),
        "m1_scientific_metric_contribution_count": 0,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-dir", type=Path, required=True)
    parser.add_argument("--channel-output", type=Path, required=True)
    parser.add_argument("--m1-output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    channel = compile_channel_manifest(phase2_dir=args.phase2_dir)
    m1 = compile_m1_manifest(channel)
    validation = validate_compiled_manifests(channel, m1)
    write_json(args.channel_output, channel)
    write_json(args.m1_output, m1)
    print(json.dumps(validation, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
