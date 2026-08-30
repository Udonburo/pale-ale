from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.gate13_causal_return.phase2_common import read_json
from tools.gate13_causal_return.modal.modal_track_a_constrained import (
    APP_NAME,
    AUTHORIZATION_FILENAME,
    GPU_MAX_CONTAINERS,
    GPU_RETRIES,
    RESULT_VOLUME_NAME,
    VolumeJournal,
    _terminal_base,
    image_definition_payload,
    image_definition_sha256,
    repo_root_for_module_path,
)
from tools.gate13_causal_return.modal.validate_constrained_modal_execution_authority import (
    CONSTRAINED_LOCK_SHA256,
    CUMULATIVE_FORWARD_CEILING,
    CUMULATIVE_GPU_WALL_CEILING_SECONDS,
    CUMULATIVE_SPEND_CEILING_USD,
    MAXIMUM_ADDITIONAL_FORWARDS,
    MAXIMUM_NEW_GPU_WALL_SECONDS,
    MAXIMUM_NEW_SPEND_USD,
    PRIOR_FORWARD_COUNT,
    PRIOR_GPU_WALL_RESERVATION_SECONDS,
    PRIOR_SPEND_RESERVATION_USD,
)
from tools.gate13_causal_return.track_a.compile_constrained_channel import (
    all_scientific_cases,
)
from tools.gate13_causal_return.track_a.constrained_runner import _base_append_record


REPO_ROOT = Path(__file__).resolve().parents[4]


class FakeVolume:
    def __init__(self) -> None:
        self.commits = 0

    def commit(self) -> None:
        self.commits += 1


class ConstrainedModalAdapterTests(unittest.TestCase):
    def test_repo_root_detection_supports_package_and_remote_alias_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "tools/gate13_causal_return/modal").mkdir(parents=True)
            (root / "analysis/gate13_causal_return/phase2").mkdir(parents=True)
            package_path = (
                root
                / "tools/gate13_causal_return/modal/modal_track_a_constrained.py"
            )
            alias_path = root / "modal_track_a_constrained.py"
            self.assertEqual(repo_root_for_module_path(package_path), root)
            self.assertEqual(repo_root_for_module_path(alias_path), root)

    def test_image_and_resource_limits_are_exact_and_bounded(self) -> None:
        payload = image_definition_payload()
        self.assertEqual(payload, image_definition_payload())
        self.assertEqual(len(image_definition_sha256()), 64)
        self.assertEqual(payload["python"], "3.11.2")
        self.assertIn("torch==2.7.1+cu126", payload["requirements"])
        self.assertIn("transformers==5.15.0", payload["requirements"])
        self.assertIn("tokenizers==0.22.2", payload["requirements"])
        self.assertIn(AUTHORIZATION_FILENAME, payload["excluded_from_image"])
        self.assertEqual(
            payload["entrypoint_alias"],
            {
                "source": "tools/gate13_causal_return/modal/modal_track_a_constrained.py",
                "destination": "/opt/gate13/modal_track_a_constrained.py",
            },
        )
        self.assertEqual(APP_NAME, "gate13-track-a-constrained-v1")
        self.assertEqual(RESULT_VOLUME_NAME, "gate13-track-a-constrained-v1-results")
        self.assertEqual(GPU_RETRIES, 0)
        self.assertEqual(GPU_MAX_CONTAINERS, 1)
        self.assertEqual(
            CONSTRAINED_LOCK_SHA256,
            "2787e1c270405d2dae5cedd5d431500d87a41669175b4ec925c6f49215f0e7dd",
        )
        self.assertEqual(PRIOR_FORWARD_COUNT + MAXIMUM_ADDITIONAL_FORWARDS, CUMULATIVE_FORWARD_CEILING)
        self.assertAlmostEqual(
            PRIOR_SPEND_RESERVATION_USD + MAXIMUM_NEW_SPEND_USD,
            CUMULATIVE_SPEND_CEILING_USD,
        )
        self.assertAlmostEqual(
            PRIOR_GPU_WALL_RESERVATION_SECONDS + MAXIMUM_NEW_GPU_WALL_SECONDS,
            CUMULATIVE_GPU_WALL_CEILING_SECONDS,
        )

    def test_volume_journal_separates_m1_from_a0_and_commits_each_event(self) -> None:
        case = all_scientific_cases()["A0"][0]
        cases = {str(case["case_id"]): case}
        volume = FakeVolume()
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            journal = VolumeJournal(
                output_dir=output,
                cases=cases,
                volume=volume,
                original_append=_base_append_record,
            )
            case_id = str(case["case_id"])
            journal(
                output / "m1_development_state_attempts.jsonl",
                {
                    "case_id": case_id,
                    "stage": "M1_DEVELOPMENT_PREFLIGHT",
                    "case_level_model_forward": 6,
                },
            )
            journal(
                output / "m1_development_state.jsonl",
                {
                    "case_id": case_id,
                    "stage": "M1_DEVELOPMENT_PREFLIGHT",
                    "case_level_model_forward": 6,
                    "response": case["expected_text"],
                    "instrument_trace": {"oracle_consulted": False},
                },
            )
            self.assertEqual(volume.commits, 2)
            self.assertTrue((output / "raw/m1_development" / f"{case_id}.txt").exists())
            self.assertFalse((output / "raw/a0" / f"{case_id}.txt").exists())
            detail = read_json(
                output / "cases/m1_development" / f"{case_id}.json"
            )
            self.assertEqual(detail["parse_status"], "PASS")
            checkpoint = read_json(output / "checkpoints/m1_development.json")
            self.assertEqual(checkpoint["completed_case_ids"], [case_id])

    def test_terminal_state_preserves_closed_surfaces_and_cumulative_count(self) -> None:
        terminal = _terminal_base(
            track_a="A0_FAIL",
            cumulative_forward_count=261,
            fresh_forward_count=256,
            operational_failures=[],
        )
        self.assertEqual(terminal["MODEL_FORWARD_COUNT"], 261)
        self.assertEqual(terminal["FRESH_MODEL_FORWARD_COUNT"], 256)
        self.assertEqual(terminal["REMAINING_FORWARD_CEILING"], 339)
        self.assertEqual(terminal["A3"], "CLOSED")
        self.assertEqual(terminal["TRACK_C"], "CLOSED")
        self.assertEqual(terminal["FORMAL_GATE13"], "CLOSED")
        self.assertEqual(terminal["ACTIVATION_EXTRACTION"], "CLOSED")

    def test_adapter_does_not_download_or_reuse_m1_as_a0(self) -> None:
        adapter = (
            REPO_ROOT
            / "tools/gate13_causal_return/modal/modal_track_a_constrained.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("snapshot_download", adapter)
        self.assertIn('"/opt/gate13/modal_track_a_constrained.py"', adapter)
        runner = (
            REPO_ROOT
            / "tools/gate13_causal_return/track_a/constrained_runner.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"m1_development_state.jsonl"', runner)
        self.assertIn('"a0_state.jsonl"', runner)
        self.assertIn('"contributes_to_a0_a1_a2_metrics": False', runner)
        self.assertNotIn("activation", adapter.lower().replace("activation_extraction", ""))


if __name__ == "__main__":
    unittest.main()
