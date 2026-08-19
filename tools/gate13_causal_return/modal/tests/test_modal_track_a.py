from __future__ import annotations

import copy
import json
import tempfile
import unittest
import uuid
from pathlib import Path

from tools.gate13_causal_return.modal.modal_track_a import (
    BASE_IMAGE_MANIFEST_DIGEST,
    EXPECTED_CLOSED_STATE,
    MODEL_REVISION,
    VolumeCheckpointJournal,
    _all_frozen_cases,
    _validate_m1_cases,
    forecast_after_m1,
    image_definition_payload,
    image_definition_sha256,
    terminal_state,
)
from tools.gate13_causal_return.modal.validate_modal_execution_authority import (
    EXPECTED_CLOSED,
    MODAL_PLAN_SHA256,
    PHASE2_A_LOCK_SHA256,
    STARTING_COMMIT,
    ModalExecutionAuthorityError,
    validate_modal_execution_authority,
)
from tools.gate13_causal_return.modal.validate_modal_runtime import (
    EXPECTED_RUNTIME,
    ModalRuntimeValidationError,
    validate_modal_runtime,
)
from tools.gate13_causal_return.phase2_common import read_json, sha256_file, write_json
from tools.gate13_causal_return.track_a import phase2_runner


REPO_ROOT = Path(__file__).resolve().parents[4]
PHASE2_DIR = REPO_ROOT / "analysis/gate13_causal_return/phase2"
M1_MANIFEST = REPO_ROOT / "tools/gate13_causal_return/modal/m1_preflight_manifest.json"


def valid_probe() -> dict:
    observed = {
        "python": EXPECTED_RUNTIME["python"],
        "torch": EXPECTED_RUNTIME["pytorch"],
        "transformers": EXPECTED_RUNTIME["transformers"],
        "tokenizers": EXPECTED_RUNTIME["tokenizers"],
        "cuda_available": True,
        "cuda_runtime": EXPECTED_RUNTIME["cuda"],
        "nvidia_driver": EXPECTED_RUNTIME["driver"],
        "gpu": EXPECTED_RUNTIME["gpu"],
        "gpu_memory_bytes": 48 * 1024**3,
    }
    return {
        "status": "PASS",
        "observed": observed,
        "checks": {field: True for field in (
            "python", "torch", "transformers", "tokenizers", "cuda_available",
            "cuda_runtime", "driver", "gpu", "gpu_memory"
        )},
    }


def valid_authorization() -> dict:
    return {
        "schema_version": "gate13_track_a_modal_execution_authorization_v1",
        "execution_authorized": True,
        "operational_scope": "TRACK_A_ONLY",
        "adapter_commit": "a" * 40,
        "starting_commit": STARTING_COMMIT,
        "phase2_a_lock_sha256": PHASE2_A_LOCK_SHA256,
        "phase2_a_modal_realization_plan_sha256": MODAL_PLAN_SHA256,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "modal_image_object_id": "im-testexactidentity",
        "modal_image_definition": image_definition_payload(),
        "modal_image_definition_sha256": image_definition_sha256(),
        "base_image_manifest_digest": BASE_IMAGE_MANIFEST_DIGEST,
        "gpu_type": "NVIDIA L40S",
        "maximum_gpu_wall_time_seconds": 36_000,
        "maximum_authorized_modal_spend_usd": 25.0,
        "closed_surfaces": dict(EXPECTED_CLOSED),
        "execution_identity": str(uuid.uuid4()),
        "model_volume": {"name": "model", "object_id": "vo-modeltest"},
        "result_volume": {"name": "result", "object_id": "vo-resulttest"},
        "m1_preflight_manifest_sha256": sha256_file(M1_MANIFEST),
        "m1_case_ids": [row["case_id"] for row in read_json(M1_MANIFEST)["cases"]],
        "modal_automatic_retries": 0,
        "maximum_active_gpu_containers": 1,
    }


class RuntimeValidatorTests(unittest.TestCase):
    def test_exact_injected_runtime_passes_at_forward_zero(self) -> None:
        report = validate_modal_runtime(
            read_json(PHASE2_DIR / "phase2_a_lock.json"),
            probe=valid_probe(),
            extra_versions={
                "huggingface_hub": EXPECTED_RUNTIME["huggingface_hub"],
                "jinja2": EXPECTED_RUNTIME["jinja2"],
                "accelerate": EXPECTED_RUNTIME["accelerate"],
                "safetensors": EXPECTED_RUNTIME["safetensors"],
            },
        )
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["model_forward_count"], 0)
        self.assertFalse(report["model_weights_loaded"])

    def test_runtime_and_static_authority_drift_fail_closed(self) -> None:
        probe = valid_probe()
        probe["observed"]["gpu"] = "NVIDIA A100"
        report = validate_modal_runtime(
            read_json(PHASE2_DIR / "phase2_a_lock.json"),
            probe=probe,
            extra_versions={
                "huggingface_hub": EXPECTED_RUNTIME["huggingface_hub"],
                "jinja2": EXPECTED_RUNTIME["jinja2"],
                "accelerate": EXPECTED_RUNTIME["accelerate"],
                "safetensors": EXPECTED_RUNTIME["safetensors"],
            },
        )
        self.assertEqual(report["status"], "MODAL_RUNTIME_MISMATCH")
        lock = read_json(PHASE2_DIR / "phase2_a_lock.json")
        lock["runtime_binding"]["model_revision"] = "0" * 40
        with self.assertRaises(ModalRuntimeValidationError):
            validate_modal_runtime(lock, probe=valid_probe())


class ExecutionAuthorityTests(unittest.TestCase):
    def _validate(self, auth: dict) -> dict:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "authorization.json"
            write_json(path, auth)
            return validate_modal_execution_authority(
                authorization_path=path,
                phase2_dir=PHASE2_DIR,
                m1_manifest_path=M1_MANIFEST,
                repo_root=REPO_ROOT,
                verify_git=False,
            )

    def test_exact_operational_authority_passes_without_overwriting_dual_lock(self) -> None:
        report = self._validate(valid_authorization())
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["operational_execution_authorized"])
        self.assertFalse(report["frozen_dual_execution_authorized"])

    def test_spend_retry_image_and_closed_surface_drift_are_rejected(self) -> None:
        mutations = []
        auth = valid_authorization()
        auth["maximum_authorized_modal_spend_usd"] = 25.01
        mutations.append(auth)
        auth = valid_authorization()
        auth["modal_automatic_retries"] = 1
        mutations.append(auth)
        auth = valid_authorization()
        auth["modal_image_definition"]["python"] = "3.11.3"
        mutations.append(auth)
        auth = valid_authorization()
        auth["closed_surfaces"]["TRACK_C"] = "OPEN"
        mutations.append(auth)
        for value in mutations:
            with self.assertRaises((ValueError, ModalExecutionAuthorityError)):
                self._validate(value)


class AdapterSemanticsTests(unittest.TestCase):
    def test_m1_is_only_four_existing_a0_cases_and_adds_no_forwards(self) -> None:
        stages, cases = _all_frozen_cases()
        self.assertEqual({key: len(value) for key, value in stages.items()}, {
            "A0": 252, "A1": 162, "A2": 90
        })
        manifest = read_json(M1_MANIFEST)
        selected = _validate_m1_cases(manifest, cases)
        self.assertEqual(len(selected), 4)
        self.assertEqual(manifest["scientific_case_additions"], 0)
        self.assertEqual(manifest["scientific_forward_additions"], 0)
        self.assertTrue(all(case["stage"] == "A0" for case in selected))

    def test_forecast_and_terminal_state_are_fail_closed(self) -> None:
        passing = forecast_after_m1(
            elapsed_before_m1_seconds=120,
            m1_elapsed_seconds=40,
            m1_case_count=4,
            completed_forward_count=4,
            forward_ceiling=600,
        )
        self.assertEqual(passing["status"], "PASS")
        blocked = forecast_after_m1(
            elapsed_before_m1_seconds=120,
            m1_elapsed_seconds=400,
            m1_case_count=4,
            completed_forward_count=4,
            forward_ceiling=600,
        )
        self.assertEqual(blocked["status"], "BLOCK")
        state = terminal_state(track_a="MODAL_RUNTIME_MISMATCH")
        self.assertEqual(state["MODEL_FORWARD_COUNT"], 0)
        self.assertEqual(state["FINAL_STOP"], "MANDATORY_STOP")
        for field, expected in EXPECTED_CLOSED_STATE.items():
            self.assertEqual(state[field], expected)

    def test_case_checkpoint_commits_attempt_then_raw_parsed_and_oracle(self) -> None:
        class FakeVolume:
            def __init__(self) -> None:
                self.commits = 0

            def commit(self) -> None:
                self.commits += 1

        stages, cases = _all_frozen_cases()
        case = next(row for row in stages["A0"] if row["condition"] == "S")
        case_id = case["case_id"]
        volume = FakeVolume()
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            journal = VolumeCheckpointJournal(
                output_dir=output,
                cases=cases,
                volume=volume,
                original_append=phase2_runner._append_record,
            )
            journal(
                output / "a0_state_attempts.jsonl",
                {
                    "case_id": case_id,
                    "stage": "A0",
                    "case_level_model_forward": 1,
                    "status": "STARTED_NO_RETRY",
                },
            )
            journal(
                output / "a0_state.jsonl",
                {
                    "case_id": case_id,
                    "stage": "A0",
                    "case_level_model_forward": 1,
                    "response": case["expected_text"],
                },
            )
            self.assertEqual(volume.commits, 2)
            self.assertEqual((output / "raw" / f"{case_id}.txt").read_text(), case["expected_text"])
            detail = read_json(output / "cases" / f"{case_id}.json")
            self.assertEqual(detail["parse_status"], "PASS")
            self.assertIsNotNone(detail["parsed_record"])
            self.assertIsNotNone(detail["oracle_record"])
            resumed = VolumeCheckpointJournal(
                output_dir=output,
                cases=cases,
                volume=volume,
                original_append=phase2_runner._append_record,
            )
            self.assertEqual(resumed.completed_ids, {case_id})

    def test_image_definition_is_canonical_and_exactly_pinned(self) -> None:
        first = image_definition_payload()
        second = image_definition_payload()
        self.assertEqual(first, second)
        self.assertEqual(len(image_definition_sha256()), 64)
        self.assertTrue(all("==" in requirement for requirement in first["requirements"]))


if __name__ == "__main__":
    unittest.main()
