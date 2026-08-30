from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from tools.gate13_causal_return.track_a.a0_failure_localization import (
    CONDITION_ORDER,
    EXECUTION_ID,
    analyze,
    write_outputs,
)


REPO_ROOT = Path(__file__).resolve().parents[4]


class A0FailureLocalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows, cls.result, _ = analyze(REPO_ROOT)

    def test_frozen_case_id_and_condition_coverage(self) -> None:
        self.assertEqual(len(self.rows), 252)
        self.assertEqual(len({row["case_id"] for row in self.rows}), 252)
        self.assertEqual(
            {condition: sum(row["condition"] == condition for row in self.rows) for condition in CONDITION_ORDER},
            {condition: 36 for condition in CONDITION_ORDER},
        )
        self.assertEqual(self.result["execution_identity"], EXECUTION_ID)
        self.assertEqual(self.result["fixed_scientific_state"]["NEW_MODEL_FORWARD_COUNT"], 0)
        self.assertEqual(self.result["fixed_scientific_state"]["A0"], "FAIL")

    def test_supplied_prefix_and_emitted_suffix_are_not_conflated(self) -> None:
        for row in self.rows:
            if row["condition"] not in {"O", "E", "N"}:
                continue
            step = int(row["edit_step"])
            self.assertEqual(
                row["state_step_provenance"][: step + 1],
                ["AUTHORITY_SUPPLIED"] * step
                + (["AUTHORITY_SUPPLIED_EDITED_OVERWRITE"] if row["condition"] == "E" else ["AUTHORITY_SUPPLIED"]),
            )
            self.assertEqual(row["emitted_step_indices"][0], step + 1)
            if row["condition"] == "E":
                self.assertIsNone(row["transition_law_consistency_at_every_step"][str(step)])

    def test_transition_consistency_is_distinct_from_oracle_correctness(self) -> None:
        discordant = []
        for row in self.rows:
            if row["condition"] not in {"S", "O", "E", "N"}:
                continue
            for step in row["emitted_step_indices"]:
                if step == 0:
                    continue
                if (
                    row["transition_law_consistency_at_every_step"][str(step)] is True
                    and row["oracle_state_correctness_at_every_step"][str(step)] is False
                ):
                    discordant.append((row["case_id"], step))
        self.assertTrue(discordant)

    def test_regeneration_is_byte_deterministic_and_covers_csv_ids(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_dir = Path(first)
            second_dir = Path(second)
            first_hashes = write_outputs(REPO_ROOT, first_dir)
            second_hashes = write_outputs(REPO_ROOT, second_dir)
            self.assertEqual(first_hashes, second_hashes)
            self.assertEqual(
                {path.name for path in first_dir.iterdir()},
                {
                    "A0_FAILURE_LOCALIZATION.md",
                    "a0_failure_localization.json",
                    "a0_case_table.csv",
                    "artifact_inventory.json",
                },
            )
            with (first_dir / "a0_case_table.csv").open(encoding="utf-8", newline="") as handle:
                csv_ids = [row["case_id"] for row in csv.DictReader(handle)]
            expected_ids = sorted(
                path.stem
                for path in (
                    REPO_ROOT
                    / "workstream/local/gate13_causal_return_outputs/phase2/modal_track_a_constrained"
                    / "volume_snapshot/executions"
                    / EXECUTION_ID
                    / "cases/a0"
                ).glob("*.json")
            )
            self.assertEqual(csv_ids, expected_ids)


if __name__ == "__main__":
    unittest.main()
