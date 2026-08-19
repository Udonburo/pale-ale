from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.gate13_causal_return.track_a.compile_phase2_cases import (
    A1_CONTROLS,
    FORWARD_CEILING,
    INTERVENTION_MARKER,
    PROJECTED_FORWARD_MAXIMUM,
    compile_a0_extension_manifest,
    compile_cases,
    compile_manifests,
    validate_manifests,
)
from tools.gate13_causal_return.track_a.evaluate_phase2 import evaluate_a1, evaluate_a2
from tools.gate13_causal_return.track_a.parse_phase2_output import (
    Phase2OutputParseError,
    parse_phase2_output,
)
from tools.gate13_causal_return.track_a.phase2_runner import (
    TrackARuntimeError,
    _load_completed,
    model_load_authorized,
)


class Phase2CompilerTests(unittest.TestCase):
    def test_manifests_are_deterministic_balanced_and_within_ceiling(self) -> None:
        first = compile_manifests()
        second = compile_manifests()
        self.assertEqual(first, second)
        report = validate_manifests(*first)
        self.assertEqual(report["status"], "PASS_MODEL_FREE_PHASE2_MANIFEST_VALIDATION")
        self.assertEqual(report["a1_case_count"], 162)
        self.assertEqual(report["a2_case_count"], 90)
        self.assertEqual(report["a0_marker_only_case_count"], 36)
        self.assertEqual(compile_a0_extension_manifest(), compile_a0_extension_manifest())
        self.assertEqual(report["target_count"], 18)
        self.assertLessEqual(PROJECTED_FORWARD_MAXIMUM, FORWARD_CEILING)

    def test_a1_controls_are_byte_length_matched_and_rule_is_not_stated(self) -> None:
        cases = compile_cases()["A1"]
        grouped: dict[tuple[str, int], dict[str, str]] = {}
        for case in cases:
            grouped.setdefault((case["target_id"], case["shots"]), {})[
                case["control"]
            ] = case["prompt"]
        for prompts in grouped.values():
            self.assertEqual(set(prompts), set(A1_CONTROLS))
            self.assertEqual(len({len(value.encode("utf-8")) for value in prompts.values()}), 1)
            for prompt in prompts.values():
                self.assertNotIn(" XOR ", prompt)
                self.assertNotIn("update rule", prompt.lower())

    def test_a2_marker_and_intervention_property_are_frozen(self) -> None:
        cases = compile_cases()["A2"]
        for case in cases:
            has_marker = INTERVENTION_MARKER in case["prompt"]
            self.assertEqual(has_marker, case["condition"] in {"edit", "marker_only"})
            if case["condition"] == "edit":
                step = case["edit_step"]
                self.assertTrue(
                    all(
                        case["edited_trace"][index] == (case["base_trace"][index] ^ 1)
                        for index in range(step, len(case["base_trace"]))
                    )
                )


class Phase2ParserAndMetricsTests(unittest.TestCase):
    def test_all_expected_outputs_parse_and_malformed_output_fails(self) -> None:
        compiled = compile_cases()
        for stage in ("A0_EXTENSION", "A1", "A2"):
            for case in compiled[stage]:
                parsed = parse_phase2_output(case, case["expected_text"])
                self.assertIn(parsed.final_prediction, (0, 1))
        case = compiled["A1"][0]
        with self.assertRaises(Phase2OutputParseError):
            parse_phase2_output(case, case["expected_text"] + "\nextra")
        lines = case["expected_text"].splitlines()
        lines[0], lines[1] = lines[1], lines[0]
        with self.assertRaises(Phase2OutputParseError):
            parse_phase2_output(case, "\n".join(lines))

    def test_oracle_fixture_passes_frozen_a1_and_a2_gates(self) -> None:
        compiled = compile_cases()
        a1_records = []
        for case in compiled["A1"]:
            response = case["expected_text"] if case["control"] == "correct" else "malformed"
            a1_records.append({"case_id": case["case_id"], "response": response})
        a1 = evaluate_a1(compiled["A1"], a1_records)
        self.assertEqual(a1["status"], "PASS")
        self.assertEqual(a1["formation_shot"], 4)

        a2_records = [
            {"case_id": case["case_id"], "response": case["expected_text"]}
            for case in compiled["A2"]
        ]
        a2 = evaluate_a2(compiled["A2"], a2_records)
        self.assertEqual(a2["status"], "PASS")
        self.assertEqual(a2["metrics"]["S_pair"], 1.0)

    def test_resume_state_rejects_duplicate_and_unknown_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "state.jsonl"
            path.write_text(
                json.dumps({"case_id": "known"})
                + "\n"
                + json.dumps({"case_id": "known"})
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(TrackARuntimeError):
                _load_completed(path, {"known"})
            path.write_text(json.dumps({"case_id": "unknown"}) + "\n", encoding="utf-8")
            with self.assertRaises(TrackARuntimeError):
                _load_completed(path, {"known"})

    def test_model_load_requires_runtime_and_dual_authorization(self) -> None:
        self.assertTrue(
            model_load_authorized(
                {"execution_authorized": True}, {"status": "PASS"}, probe_only=False
            )
        )
        self.assertFalse(
            model_load_authorized(
                {"execution_authorized": False}, {"status": "PASS"}, probe_only=False
            )
        )
        self.assertFalse(
            model_load_authorized(
                {"execution_authorized": True},
                {"status": "BLOCKED_EXACT_RUNTIME_UNAVAILABLE"},
                probe_only=False,
            )
        )
        self.assertFalse(
            model_load_authorized(
                {"execution_authorized": True}, {"status": "PASS"}, probe_only=True
            )
        )


if __name__ == "__main__":
    unittest.main()
