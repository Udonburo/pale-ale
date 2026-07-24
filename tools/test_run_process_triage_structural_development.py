#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


def _load(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load("run_process_triage_structural_development")


class RunProcessTriageStructuralDevelopmentTest(unittest.TestCase):
    def test_reproducibility_projection_excludes_bulk_rows(self) -> None:
        report = {
            "score_payload_sha256": {"primary": "a"},
            "selected_score_payload_sha256": {
                "sealed_baseline": "b",
                "primary_augmented": "c",
            },
            "sealed_baseline": {
                "selected_oof_score_payload_sha256": "b",
            },
            "primary_augmented": {
                "selected_oof_score_payload_sha256": "c",
                "refit_prediction_payload_sha256": "d",
            },
        }
        projection = runner._reproducibility_projection(report)
        self.assertEqual(
            projection["primary_augmented_refit"],
            "d",
        )
        self.assertNotIn("candidate_rows", projection)


if __name__ == "__main__":
    unittest.main()
