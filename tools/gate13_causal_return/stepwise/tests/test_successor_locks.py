from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.gate13_causal_return.stepwise.validate_successor_locks import (
    SuccessorLockError,
    validate_successor_locks,
)


class SuccessorLockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[4]
        cls.lock_dir = cls.repo_root / "analysis/gate13_causal_return/successor"

    def test_simultaneous_locks_validate(self):
        result = validate_successor_locks(lock_dir=self.lock_dir)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["maximum_campaign_forward_count"], 534)

    def test_threshold_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            for name in (
                "stepwise_track_a_qualification_lock.json",
                "fresh_operator_qualification_lock.json",
            ):
                (target / name).write_bytes((self.lock_dir / name).read_bytes())
            path = target / "stepwise_track_a_qualification_lock.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["qualification_surface"]["STREAM-A0"]["thresholds"][
                "one_step_accuracy_min"
            ] = 0.1
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(SuccessorLockError, "STREAM-A0 threshold drift"):
                validate_successor_locks(lock_dir=target)


if __name__ == "__main__":
    unittest.main()
