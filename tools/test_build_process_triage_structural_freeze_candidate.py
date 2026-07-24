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


builder = _load("build_process_triage_structural_freeze_candidate")


class BuildProcessTriageStructuralFreezeCandidateTest(
    unittest.TestCase
):
    def test_manifest_keeps_locked_surface_closed(self) -> None:
        manifest = builder.structural_family_manifest()
        self.assertEqual(
            manifest["family_id"],
            "task_anchored_artifact_detour_v0.1",
        )
        self.assertEqual(
            manifest["development_candidate_family_count"],
            1,
        )
        self.assertFalse(manifest["learned_parameters"])
        self.assertFalse(manifest["locked_evaluation_authorized"])
        self.assertEqual(
            set(manifest["frozen_controls"]),
            {
                "score_order_shuffle",
                "dependency_cycle_randomization",
                "label_permutation",
            },
        )


if __name__ == "__main__":
    unittest.main()
