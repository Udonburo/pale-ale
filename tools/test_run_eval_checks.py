#!/usr/bin/env python3
"""Tests for the evaluation-factory runner scaffold."""

from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import run_eval_checks as runner


class RunEvalChecksTest(unittest.TestCase):
    def test_required_tiers_are_defined(self) -> None:
        self.assertEqual(
            runner.TIER_VALUES,
            ("cpu-nightly", "l4-smoke", "l4-weekly", "summarize-existing"),
        )

    def test_l4_weekly_keeps_expansion_surfaces_out_of_scope(self) -> None:
        plan = runner.dispatch(runner.Tier.L4_WEEKLY)

        self.assertIn("7B FP32", plan.out_of_scope)
        self.assertIn("protocol-expanding candidates", plan.out_of_scope)
        self.assertIn("quantized candidates", plan.out_of_scope)
        self.assertIn("sidecar candidates", plan.out_of_scope)

    def test_main_prints_dry_run_plan(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            self.assertEqual(runner.main(["--tier", "cpu-nightly"]), 0)

        text = output.getvalue()
        self.assertIn("tier: cpu-nightly", text)
        self.assertIn("expected resource posture:", text)
        self.assertIn("planned actions:", text)
        self.assertIn("not implemented yet:", text)


if __name__ == "__main__":
    raise SystemExit(unittest.main())
