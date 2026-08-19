from __future__ import annotations

import copy
import unittest

from tools.gate13_causal_return.track_a.compile_register_cases import compile_ledger
from tools.gate13_causal_return.track_a.dry_run_register_runner import (
    SyntheticRecordError,
    completed_ids,
    run_preflight,
)
from tools.gate13_causal_return.track_a.oracle import (
    edited_trace,
    paired_selectivity_exact,
    parity_trace,
    transition_accuracy,
)
from tools.gate13_causal_return.track_a.parse_register_output import (
    OutputParseError,
    parse_register_output,
)
from tools.gate13_causal_return.track_a.render_register_cases import (
    INTERVENTION_MARKER,
)
from tools.gate13_causal_return.track_a.validate_register_cases import (
    LedgerValidationError,
    validate_ledger,
)


class OracleTests(unittest.TestCase):
    def test_phase_indexed_trace_and_edit_commute(self) -> None:
        bits = (1, 0, 1, 1, 0, 1)
        base = parity_trace(bits)
        edited = edited_trace(bits, 3)
        self.assertEqual(base[:3], edited[:3])
        self.assertTrue(
            all(edited[index] == (base[index] ^ 1) for index in range(3, 7))
        )
        self.assertEqual(transition_accuracy(base, bits), 1.0)
        self.assertEqual(transition_accuracy(edited[3:], bits[3:]), 1.0)

    def test_paired_selectivity_requires_oracle_correct_base(self) -> None:
        bits = (1, 0, 1, 1)
        base = parity_trace(bits)
        edited = edited_trace(bits, 2)
        self.assertTrue(paired_selectivity_exact(base, edited, bits, 2))
        wrong_base = tuple(value ^ 1 for value in base)
        self.assertIsNone(
            paired_selectivity_exact(wrong_base, edited, bits, 2)
        )


class LedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = compile_ledger()

    def test_ledger_passes_all_invariants(self) -> None:
        report = validate_ledger(self.ledger)
        self.assertEqual(report["status"], "PASS_MODEL_FREE_VALIDATION")
        self.assertEqual(report["case_count"], 216)
        self.assertEqual(report["base_case_count"], 36)
        self.assertEqual(report["forward_forecast"]["projected_maximum"], 456)

    def test_visible_edit_marker_is_unique_to_e(self) -> None:
        for case in self.ledger["cases"]:
            present = INTERVENTION_MARKER in case["prompt"]
            self.assertEqual(present, case["condition"] == "E")

    def test_tampered_ledger_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.ledger)
        tampered["cases"][0]["base_trace"][-1] ^= 1
        with self.assertRaises(LedgerValidationError):
            validate_ledger(tampered)


class OutputParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cases = {
            case["condition"]: case
            for case in compile_ledger()["cases"]
            if case["base_id"] == "a0-l04-y0-early-r0"
        }

    def test_all_frozen_condition_outputs_parse(self) -> None:
        for condition, case in self.cases.items():
            with self.subTest(condition=condition):
                parsed = parse_register_output(case, case["expected_text"])
                self.assertEqual(parsed.final_prediction, case["expected_final_answer"])
                if condition in {"S", "O"}:
                    self.assertEqual(parsed.trace_prediction, tuple(case["base_trace"]))
                elif condition == "E":
                    self.assertEqual(parsed.trace_prediction, tuple(case["edited_trace"]))
                else:
                    self.assertIsNone(parsed.trace_prediction)

    def test_direct_answer_rejects_explanatory_prose(self) -> None:
        with self.assertRaises(OutputParseError):
            parse_register_output(self.cases["D"], "The answer is 0")

    def test_structured_answer_rejects_missing_step(self) -> None:
        case = self.cases["S"]
        lines = case["expected_text"].splitlines()
        malformed = "\n".join(lines[:2] + lines[3:])
        with self.assertRaises(OutputParseError):
            parse_register_output(case, malformed)

    def test_continuation_rejects_reordered_steps(self) -> None:
        case = self.cases["O"]
        lines = case["expected_text"].splitlines()
        malformed = "\n".join([lines[1], lines[0], *lines[2:]])
        with self.assertRaises(OutputParseError):
            parse_register_output(case, malformed)

    def test_structured_answer_rejects_final_mismatch(self) -> None:
        case = self.cases["S"]
        wrong_answer = 1 - int(case["expected_final_answer"])
        malformed = case["expected_text"].rsplit("=", 1)[0] + f"= {wrong_answer}"
        with self.assertRaises(OutputParseError):
            parse_register_output(case, malformed)


class DryRunTests(unittest.TestCase):
    def test_synthetic_resume_is_idempotent_and_metrics_are_oracle_based(self) -> None:
        report = run_preflight(compile_ledger())
        self.assertEqual(report["status"], "PASS_SYNTHETIC_DRY_RUN")
        self.assertEqual(report["model_forward_count"], 0)
        self.assertEqual(report["checks"]["second_resume_missing_count"], 0)
        self.assertTrue(report["checks"]["strict_parser_exercised"])
        self.assertTrue(report["checks"]["unknown_case_rejected"])
        self.assertEqual(
            report["synthetic_oracle_metrics"]["oracle_cf_accuracy"], 1.0
        )
        self.assertEqual(
            report["synthetic_oracle_metrics"]["paired_selectivity"], 1.0
        )

    def test_duplicate_completed_id_is_rejected(self) -> None:
        with self.assertRaises(SyntheticRecordError):
            completed_ids([{"case_id": "x"}, {"case_id": "x"}])


if __name__ == "__main__":
    unittest.main()
