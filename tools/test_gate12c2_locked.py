#!/usr/bin/env python3

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from gate12c2_minimal.io import load_json, sha256_file, write_json_atomic  # noqa: E402
from gate12c2_minimal.locked_run import (  # noqa: E402
    actual_implementation_hashes,
    execute_locked,
)
from gate12c2_minimal.locked_calibration import current_environment  # noqa: E402
from gate12c2_minimal.locked_validate import (  # noqa: E402
    IndependentLockedValidationError,
    validate_locked_run,
)


LOCKED_SPEC = TOOLS_DIR / "gate12c2_minimal" / "locked_calibration.json"


class Gate12C2LockedTest(unittest.TestCase):
    def _spec(self, root: Path) -> Path:
        value = copy.deepcopy(load_json(LOCKED_SPEC))
        value["study_id"] = "gate12c2-locked-unit"
        value["attempt_id"] = "gate12c2-locked-unit-attempt"
        value["seed_namespace"] = "gate12c2-locked-unit-seed"
        value["dataset_count"] = 1
        value["cohort_size"] = 8
        value["inner_draws"] = 3
        value["stability_prefix_draws"] = 1
        value["criteria"]["s0_max_wilson_upper"] = 1.0
        value["criteria"]["s1_primary_min_power"] = 0.0
        value["criteria"]["s1_primary_min_wilson_lower"] = 0.0
        value["criteria"]["s2_min_success_rate"] = 0.0
        value["criteria"]["s2_min_wilson_lower"] = 0.0
        value["criteria"]["nuisance_max_quantile_difference"] = 1.0
        value["criteria"]["nuisance_median_quantile_difference"] = 1.0
        value["criteria"]["stability_min_decision_agreement"] = 0.0
        value["criteria"]["stability_max_p95_effect_shift"] = 10.0
        value["resource_cap"] = {
            "max_wall_seconds": 120,
            "max_output_bytes": 100_000_000,
            "max_dataset_shards": 60,
        }
        value["environment"] = current_environment()
        value["implementation_sha256"] = actual_implementation_hashes()
        path = root / "locked.json"
        write_json_atomic(path, value)
        return path

    def test_locked_run_reaggregates_independently(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = self._spec(root)
            output = root / "run"
            result = execute_locked(spec, output)
            independent = validate_locked_run(spec, output)
            self.assertEqual(result["decision"], independent["decision"])
            self.assertEqual(independent["component_reaggregation"], "independent")
            self.assertEqual(independent["shard_count"], 60)

    def test_resume_rejects_valid_json_component_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = self._spec(root)
            output = root / "run"
            execute_locked(spec, output)
            (output / "analysis.json").unlink()
            (output / "manifest.json").unlink()
            state = load_json(output / "state.json")
            state["state"] = "FAILED"
            state["error"] = "simulated interruption"
            write_json_atomic(output / "state.json", state, replace=True)
            shard_path = output / "shards" / "S0" / "ambient7-separated" / "d0000.json"
            shard = load_json(shard_path)
            shard["null_rows"][0][3][0] += 0.25
            write_json_atomic(shard_path, shard, replace=True)
            with self.assertRaises(Exception):
                execute_locked(spec, output, resume=True)

    def test_independent_validator_rejects_rehashed_scientific_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = self._spec(root)
            output = root / "run"
            execute_locked(spec, output)
            relative = "shards/S0/ambient7-separated/d0000.json"
            shard_path = output / relative
            shard = load_json(shard_path)
            shard["null_rows"][0][3][0] += 0.25
            write_json_atomic(shard_path, shard, replace=True)
            manifest = load_json(output / "manifest.json")
            manifest["files"][relative] = sha256_file(shard_path)
            write_json_atomic(output / "manifest.json", manifest, replace=True)
            with self.assertRaises(IndependentLockedValidationError):
                validate_locked_run(spec, output)


if __name__ == "__main__":
    unittest.main()
