from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.gate12c2_v2_statistical_adequacy import audit
from tools.gate12c2_minimal import generators, locked_calibration


RESULT_ROOT = Path(
    os.environ.get(
        "GATE12C2_LOCKED_RESULT_ROOT",
        "__retained_gate12c2_result_unavailable__",
    )
)


@unittest.skipUnless(RESULT_ROOT.is_dir(), "retained locked result is unavailable")
class RetainedEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.study = json.loads((RESULT_ROOT / "study.json").read_text(encoding="utf-8"))
        cls.case_id = str(cls.study["cases"][0]["case_id"])

    def _shard(self, config: str) -> dict:
        path = RESULT_ROOT / "shards" / config / self.case_id / "d0000.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_endpoint_reconstruction_matches_frozen_implementation(self) -> None:
        for config in ("S0", "S2"):
            shard = self._shard(config)
            curves, endpoints, _, _, _ = audit.parse_shard(
                shard,
                epsilon=float(self.study["epsilon"]),
                bootstrap_repeats=64,
            )
            by_q = {int(row["q"]): row for row in endpoints}
            for q in (1, 2):
                for k, effect_name in ((9, "effect_k9"), (15, "effect_k15")):
                    frozen = locked_calibration.endpoint_from_compact(
                        shard,
                        spec=self.study,
                        q=q,
                        draw_limit=k,
                    )
                    self.assertAlmostEqual(by_q[q][effect_name], frozen["median_effect"])
                    curve = next(row for row in curves if row["q"] == q and row["k"] == k)
                    self.assertAlmostEqual(curve["directional_sign_p"], frozen["directional_sign_p"])

    def test_reconstructed_derangements_match_generator(self) -> None:
        namespace = str(self.study["seed_namespace"])
        for draw in (0, 7, 14):
            for role in range(3):
                expected = generators._derangement(
                    12,
                    generators.rng(namespace, self.case_id, "S0", 0, "N1", draw, role),
                )
                actual = audit._derangement(
                    12,
                    np.random.default_rng(
                        audit.stable_seed(namespace, self.case_id, "S0", 0, "N1", draw, role)
                    ),
                )
                np.testing.assert_array_equal(actual, expected)


class AggregationTests(unittest.TestCase):
    def test_nuisance_pool_uses_arm_dimension_without_double_counting(self) -> None:
        pool = {}
        for config in audit.CONFIGS:
            for surface in ("edge", "product"):
                pool[(config, surface, "c01", 0, 0, "observed")] = [0.0, 1.0]
                pool[(config, surface, "c01", 0, 0, "null")] = [0.1, 0.9]
        geometry = pd.DataFrame(
            {
                "config": list(audit.CONFIGS),
                "cross_edge_correlation_max_difference": [0.1] * len(audit.CONFIGS),
            }
        )
        summary = audit.summarize_nuisance(pool, geometry)
        self.assertEqual(len(summary), len(audit.CONFIGS) * 2)
        self.assertTrue((summary["group_count"] == 1).all())

    def test_pooled_geometry_flattens_edge_spectrum_tensor(self) -> None:
        generator = np.random.default_rng(17)
        pool = {}
        for config in audit.CONFIGS:
            for surface in ("edge", "product"):
                feature_shape = (3, 2) if surface == "edge" else (6,)
                pool[(config, surface, "c01", "observed")] = [
                    generator.normal(size=(12, *feature_shape)) for _ in range(48)
                ]
                pool[(config, surface, "c01", "null")] = [
                    generator.normal(size=(180, *feature_shape)) for _ in range(48)
                ]
        by_case, summary = audit.summarize_pooled_geometry(pool, bootstrap_repeats=2)
        self.assertEqual(len(by_case), len(audit.CONFIGS) * 2)
        self.assertEqual(len(summary), len(audit.CONFIGS) * 2)
        self.assertTrue((by_case["dataset_count"] == 48).all())

    def test_output_guard_rejects_locked_subdirectory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(audit.AuditError):
                audit.audit(root, root / "audit", 64)


if __name__ == "__main__":
    unittest.main()
