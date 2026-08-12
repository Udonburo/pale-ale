#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from gate12c2_minimal.experiment import load_spec  # noqa: E402
from gate12c2_minimal.generators import (  # noqa: E402
    edge_spectrum_error,
    generate_s0_cohort,
    independent_edge_reorientation,
    joint_realizability_error,
    n1_reassignment,
)
from gate12c2_minimal.io import write_json_atomic  # noqa: E402
from gate12c2_minimal.metrics import residual_diagnostics  # noqa: E402
from gate12c2_minimal.run import Gate12C2RunError, execute  # noqa: E402
from gate12c2_minimal.validate import (  # noqa: E402
    Gate12C2ValidationError,
    validate_run,
)


SPEC_PATH = TOOLS_DIR / "gate12c2_minimal" / "study.json"


class Gate12C2MinimalTest(unittest.TestCase):
    def test_residual_identities_hold_for_typed_rectangular_cycles(self) -> None:
        generator = np.random.default_rng(20260812)
        for dimensions in ((4, 5, 6), (7, 4, 5), (6, 6, 6)):
            d0, d1, d2 = dimensions
            m0 = generator.normal(size=(d1, d0))
            m1 = generator.normal(size=(d2, d1))
            m2 = generator.normal(size=(d0, d2))
            for q in range(1, min(d0, d1, d2)):
                result = residual_diagnostics(m0, m1, m2, q)
                self.assertTrue(result.numerical_pass)
                self.assertLess(result.matrix_identity_error, 1e-10)
                self.assertLess(result.squared_identity_error, 1e-9)

    def test_n1_is_realizable_and_s2_stressor_preserves_edge_spectra(self) -> None:
        spec, _ = load_spec(SPEC_PATH)
        case = spec["cases"][0]
        cohort = generate_s0_cohort(
            case=case,
            seed_namespace="unit-fresh",
            outer_index=0,
            cohort_size=8,
            frame_noise=0.25,
        )
        n1 = n1_reassignment(
            cohort,
            seed_namespace="unit-fresh",
            case_id=case["case_id"],
            regime="S0",
            outer_index=0,
            draw_index=0,
        )
        self.assertLess(max(joint_realizability_error(graph) for graph in n1), 1e-12)
        stressor = independent_edge_reorientation(
            n1[0].edges,
            seed_namespace="unit-fresh",
            case_id=case["case_id"],
            outer_index=0,
            draw_index=0,
            graph_index=0,
            trial_index=0,
        )
        self.assertLess(edge_spectrum_error(n1[0].edges, stressor), 1e-12)

    def _small_spec(self, directory: Path) -> Path:
        raw = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        value = copy.deepcopy(raw)
        value["study_id"] = "gate12c2-minimal-unit"
        value["seed_namespace"] = "gate12c2-minimal-unit-fresh"
        value["outer_count"] = 1
        value["cohort_size"] = 4
        value["inner_draws"] = 1
        value["stressor_trials"] = 2
        value["smoke_acceptance"]["s1_min_directional_endpoint_fraction"] = 0.0
        value["smoke_acceptance"]["s2_min_inflation_endpoint_fraction"] = 0.0
        path = directory / "study.json"
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        return path

    def test_run_is_deterministic_resumable_and_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec_path = self._small_spec(root)
            first = root / "first"
            second = root / "second"
            execute(spec_path, first)
            execute(spec_path, second)
            self.assertEqual(
                (first / "result.json").read_bytes(),
                (second / "result.json").read_bytes(),
            )
            self.assertEqual(
                (first / "manifest.json").read_bytes(),
                (second / "manifest.json").read_bytes(),
            )
            self.assertEqual(validate_run(spec_path, first)["status"], "pass")
            with self.assertRaises(Gate12C2RunError):
                execute(spec_path, first, resume=True)

            resumed = root / "resumed"
            shutil.copytree(first, resumed)
            (resumed / "manifest.json").unlink()
            (resumed / "result.json").unlink()
            missing = resumed / "shards" / "case-11__S2.json"
            missing.unlink()
            write_json_atomic(
                resumed / "state.json",
                {
                    "schema_version": "gate12c2_minimal_state_v0.1",
                    "study_sha256": load_spec(spec_path)[1],
                    "state": "FAILED",
                    "error": "simulated interruption",
                },
                replace=True,
            )
            execute(spec_path, resumed, resume=True)
            self.assertEqual(validate_run(spec_path, resumed)["status"], "pass")
            self.assertEqual(
                (first / "result.json").read_bytes(),
                (resumed / "result.json").read_bytes(),
            )

    def test_validator_rejects_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec_path = self._small_spec(root)
            output = root / "run"
            execute(spec_path, output)
            shard = output / "shards" / "case-00__S0.json"
            shard.write_bytes(shard.read_bytes() + b" ")
            with self.assertRaises(Gate12C2ValidationError):
                validate_run(spec_path, output)

    def test_resume_rejects_valid_json_scientific_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec_path = self._small_spec(root)
            output = root / "run"
            execute(spec_path, output)
            (output / "manifest.json").unlink()
            (output / "result.json").unlink()
            write_json_atomic(
                output / "state.json",
                {
                    "schema_version": "gate12c2_minimal_state_v0.1",
                    "study_sha256": load_spec(spec_path)[1],
                    "state": "FAILED",
                    "error": "simulated interruption",
                },
                replace=True,
            )
            shard_path = output / "shards" / "case-00__S0.json"
            shard = json.loads(shard_path.read_text(encoding="utf-8"))
            shard["component_rows"][0]["observed"]["a"] += 0.125
            write_json_atomic(shard_path, shard, replace=True)
            with self.assertRaises(Exception):
                execute(spec_path, output, resume=True)


if __name__ == "__main__":
    unittest.main()
