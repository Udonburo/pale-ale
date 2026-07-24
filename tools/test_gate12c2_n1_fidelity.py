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
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fidelity = _load("gate12c2_n1_fidelity")


class Gate12C2N1FidelityTest(unittest.TestCase):
    def test_small_profile_is_deterministic_and_development_only(self) -> None:
        arguments = {
            "master_seed": "fidelity-unit-test",
            "block_count": 4,
            "draw_count_per_case": 2,
            "effect_strengths": (0.15,),
        }
        first = fidelity.run_development_n1_fidelity_profile(**arguments)
        second = fidelity.run_development_n1_fidelity_profile(**arguments)
        self.assertEqual(
            first["deterministic_projection_sha256"],
            second["deterministic_projection_sha256"],
        )
        self.assertEqual(first["surface_id"], "development")
        self.assertFalse(
            first["interpretation_boundary"][
                "locked_synthetic_execution_authorized"
            ]
        )
        self.assertFalse(
            first["interpretation_boundary"]["nuisance_threshold_frozen"]
        )
        self.assertEqual(
            set(first["configuration_summaries"]),
            {
                "S0_true_null",
                "S1_known_reverse_shared_node_coupling:effect=0.15",
            },
        )
        for summary in first["configuration_summaries"].values():
            self.assertEqual(summary["profile_count"], 24)
            self.assertTrue(
                all(
                    value == 1.0
                    for value in summary[
                        "hard_constraint_pass_rates"
                    ].values()
                )
            )

    def test_profile_rejects_invalid_development_grid(self) -> None:
        with self.assertRaises(fidelity.Gate12C2FidelityError):
            fidelity.run_development_n1_fidelity_profile(
                master_seed="x",
                block_count=3,
                draw_count_per_case=1,
                effect_strengths=(0.1,),
            )
        with self.assertRaises(fidelity.Gate12C2FidelityError):
            fidelity.run_development_n1_fidelity_profile(
                master_seed="x",
                block_count=4,
                draw_count_per_case=1,
                effect_strengths=(0.1, 0.1),
            )


if __name__ == "__main__":
    unittest.main()
