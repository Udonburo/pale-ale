from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.gate12c2_v2_balanced_prototype import prototype
from tools.gate12c2_minimal.io import load_json, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "analysis" / "gate12c2_v2_balanced_prototype" / "prototype_spec.json"
LOCKED_STUDY = Path(
    os.environ.get(
        "GATE12C2_LOCKED_STUDY",
        "__retained_gate12c2_study_unavailable__",
    )
)


class AssignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = prototype.validate_spec(load_json(SPEC_PATH))

    def test_balanced_cycles_have_exact_exposure_and_no_fixed_points(self) -> None:
        assignments = prototype.balanced_assignments(
            size=12,
            draw_count=44,
            seed_namespace=self.spec["seed_namespace"],
            case_id=self.spec["case_ids"][0],
            regime="S0",
            dataset_index=0,
        )
        identity = np.arange(12)
        self.assertFalse(np.any(assignments == identity[None, None, :]))
        for draw in range(44):
            for role in range(3):
                np.testing.assert_array_equal(np.sort(assignments[draw, role]), identity)
        for recipient in range(12):
            for role in range(3):
                counts = [
                    int(np.sum(assignments[:, role, recipient] == donor))
                    for donor in range(12)
                    if donor != recipient
                ]
                self.assertEqual(counts, [4] * 11)
        metrics = prototype.assignment_metrics(assignments)
        self.assertEqual(metrics["max_exposure_cv"], 0.0)
        self.assertEqual(metrics["max_exposure_range"], 0.0)

    def test_assignment_generation_is_deterministic_and_iid_is_not_forced_balanced(self) -> None:
        kwargs = {
            "size": 12,
            "draw_count": 44,
            "seed_namespace": self.spec["seed_namespace"],
            "case_id": self.spec["case_ids"][1],
            "regime": "S1",
            "dataset_index": 3,
        }
        first = prototype.iid_assignments(**kwargs)
        second = prototype.iid_assignments(**kwargs)
        np.testing.assert_array_equal(first, second)
        self.assertGreater(prototype.assignment_metrics(first)["max_exposure_cv"], 0.0)


class PrecisionTests(unittest.TestCase):
    def test_cycle_bootstrap_resamples_whole_cycles(self) -> None:
        indices = prototype.cycle_bootstrap_indices(
            44,
            cycle_size=11,
            seed=17,
            repeats=32,
        )
        self.assertEqual(indices.shape, (32, 44))
        for row in indices:
            for start in range(0, 44, 11):
                block = row[start : start + 11]
                np.testing.assert_array_equal(block - block[0], np.arange(11))
                self.assertEqual(int(block[0]) % 11, 0)

    def test_median_effect_supports_point_and_bootstrap_indices(self) -> None:
        matrix = np.arange(24, dtype=float).reshape(3, 8)
        point = prototype.median_effect(matrix, np.arange(4))
        self.assertEqual(float(point), 9.5)
        boot = prototype.median_effect(matrix, np.asarray([[0, 1, 2, 3], [4, 5, 6, 7]]))
        np.testing.assert_allclose(boot, [9.5, 13.5])


class BoundaryTests(unittest.TestCase):
    def test_output_guard_rejects_locked_subdirectory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            study = root / "study.json"
            study.write_text("{}", encoding="utf-8")
            with self.assertRaises(prototype.PrototypeError):
                prototype.guard_output_location(root / "prototype", study)


@unittest.skipUnless(LOCKED_STUDY.is_file(), "retained locked study is unavailable")
class EndToEndShardTests(unittest.TestCase):
    def test_s0_shard_validates_with_reconstructed_assignments(self) -> None:
        spec = prototype.validate_spec(load_json(SPEC_PATH))
        study = prototype.validate_locked_study(LOCKED_STUDY, spec)
        case = next(case for case in study["cases"] if case["case_id"] == spec["case_ids"][0])
        shard = prototype.run_shard(
            config="S0",
            case=case,
            dataset_index=0,
            spec=spec,
            spec_sha256=sha256_file(SPEC_PATH),
            study=study,
        )
        prototype.validate_shard(
            shard,
            config="S0",
            case_id=case["case_id"],
            dataset=0,
            spec=spec,
            spec_sha256=sha256_file(SPEC_PATH),
        )
        self.assertEqual(
            shard["schedules"]["balanced"]["assignment_metrics"]["max_exposure_cv"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
