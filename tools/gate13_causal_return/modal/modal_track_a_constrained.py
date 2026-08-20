"""Modal realization of the bounded Track A syntax-constrained channel."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

try:
    import modal
except ModuleNotFoundError:  # pragma: no cover - local model-free test path
    modal = None  # type: ignore[assignment]
else:
    if not all(hasattr(modal, field) for field in ("Image", "App", "Volume")):
        modal = None  # type: ignore[assignment]

from tools.gate13_causal_return.phase2_common import (
    read_json,
    sha256_bytes,
    sha256_file,
    sha256_json,
)
from tools.gate13_causal_return.modal.modal_track_a import (
    BASE_IMAGE,
    BASE_IMAGE_LINUX_AMD64_DIGEST,
    BASE_IMAGE_MANIFEST_DIGEST,
    CPU_RATE_USD_PER_CORE_SECOND,
    GPU_CPU_CORES,
    GPU_MEMORY_MIB,
    GPU_TOTAL_RATE_USD_PER_SECOND,
    HF_CACHE,
    HF_HOME,
    MEMORY_RATE_USD_PER_GIB_SECOND,
    MODEL_ACQUISITION_REPORT,
    MODEL_MOUNT,
    MODEL_REPOSITORY,
    MODEL_VOLUME_NAME,
    PYTORCH_INDEX_URL,
    REMOTE_ROOT,
    RESULT_MOUNT,
    RUNTIME_REQUIREMENTS,
    TOKENIZER_REVISION,
    _artifact_manifest,
    _atomic_write_json,
    _atomic_write_text,
    _parse_case_record,
    _verify_model_report,
)
from tools.gate13_causal_return.modal.validate_constrained_modal_execution_authority import (
    CUMULATIVE_FORWARD_CEILING,
    CUMULATIVE_GPU_WALL_CEILING_SECONDS,
    CUMULATIVE_SPEND_CEILING_USD,
    MAXIMUM_ADDITIONAL_FORWARDS,
    MAXIMUM_NEW_GPU_WALL_SECONDS,
    MODEL_REVISION,
    MODEL_VOLUME_OBJECT_ID,
    PRIOR_FORWARD_COUNT,
    PRIOR_GPU_WALL_RESERVATION_SECONDS,
    PRIOR_SPEND_RESERVATION_USD,
    RESULT_VOLUME_NAME,
    validate_constrained_modal_execution_authority,
    validate_constrained_modal_execution_authority_payload,
)
from tools.gate13_causal_return.modal.validate_modal_runtime import (
    validate_modal_runtime,
)


APP_NAME = "gate13-track-a-constrained-v1"
AUTHORIZATION_FILENAME = "phase2_a_constrained_modal_execution_authorization.json"
REMOTE_PHASE2_DIR = REMOTE_ROOT / "analysis/gate13_causal_return/phase2"
GPU_REQUEST = "L40S"
GPU_EXACT_NAME = "NVIDIA L40S"
GPU_RETRIES = 0
GPU_MAX_CONTAINERS = 1
GPU_TIMEOUT_SECONDS = int(MAXIMUM_NEW_GPU_WALL_SECONDS)


class ConstrainedModalError(RuntimeError):
    """Fail-closed operational error for the constrained variant."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def local_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def image_definition_payload() -> dict[str, Any]:
    return {
        "schema_version": "gate13_track_a_constrained_modal_image_definition_v1",
        "base_image": BASE_IMAGE,
        "base_image_manifest_digest": BASE_IMAGE_MANIFEST_DIGEST,
        "base_image_linux_amd64_digest": BASE_IMAGE_LINUX_AMD64_DIGEST,
        "python": "3.11.2",
        "pip": "26.2.1",
        "requirements": list(RUNTIME_REQUIREMENTS),
        "pytorch_extra_index_url": PYTORCH_INDEX_URL,
        "copied_roots": [
            "tools/gate13_causal_return",
            "analysis/gate13_causal_return/phase2",
        ],
        "entrypoint_alias": {
            "source": "tools/gate13_causal_return/modal/modal_track_a_constrained.py",
            "destination": "/opt/gate13/modal_track_a_constrained.py",
        },
        "excluded_from_image": [
            "**/__pycache__/**",
            "**/*.pyc",
            "phase2_a_modal_execution_authorization.json",
            "phase2_a_modal_execution_authorization_v2.json",
            AUTHORIZATION_FILENAME,
        ],
        "output_channel": "CANONICAL_TOKEN_FINITE_STATE_PREFIX_CONSTRAINT",
    }


def image_definition_sha256() -> str:
    return sha256_json(image_definition_payload())


def _authorization_from_text(
    text: str, expected_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    actual = sha256_bytes(text.encode("utf-8"))
    if actual != expected_sha256:
        raise ConstrainedModalError("authorization byte SHA-256 mismatch")
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ConstrainedModalError("authorization root must be an object")
    validation = validate_constrained_modal_execution_authority_payload(
        auth=value,
        phase2_dir=Path(REMOTE_PHASE2_DIR.as_posix()),
        repo_root=Path(REMOTE_ROOT.as_posix()),
        verify_git=False,
    )
    return value, validation


def _claim_execution(output_dir: Path, auth: Mapping[str, Any], volume: Any) -> dict[str, Any]:
    path = output_dir / "execution_claim.json"
    expected = {
        "execution_identity": auth["execution_identity"],
        "implementation_commit": auth["implementation_commit"],
        "authorization_sha256": sha256_json(auth),
    }
    if path.exists():
        claim = read_json(path)
        for field, value in expected.items():
            if claim.get(field) != value:
                raise ConstrainedModalError("execution identity is claimed by different authority")
        claim["container_start_count"] = int(claim.get("container_start_count") or 0) + 1
        claim.setdefault("operational_restarts", []).append(
            {"observed_at": utc_now(), "reason": "UNCHANGED_CODE_RESUME_ENTRY"}
        )
    else:
        claim = {
            "schema_version": "gate13_track_a_constrained_execution_claim_v1",
            **expected,
            "claimed_at": utc_now(),
            "container_start_count": 1,
            "operational_restarts": [],
        }
    _atomic_write_json(path, claim)
    volume.commit()
    return claim


def _terminal_base(
    *,
    track_a: str,
    cumulative_forward_count: int,
    fresh_forward_count: int,
    operational_failures: list[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "gate13_track_a_constrained_modal_terminal_v1",
        "terminal_status": track_a,
        "TRACK_A": track_a,
        "TRACK_A_FREE_GENERATION_CHANNEL": "TERMINATED_INSTRUMENT_CHANNEL_INADEQUATE",
        "TRACK_A_SCIENTIFIC_QUESTION": "OPEN",
        "TRACK_A_CONSTRAINED_REGISTER_CHANNEL": "AUTHORIZED_FOR_BOUNDED_REDESIGN_AND_EXECUTION",
        "MODEL_FORWARD_COUNT": cumulative_forward_count,
        "FRESH_MODEL_FORWARD_COUNT": fresh_forward_count,
        "REMAINING_FORWARD_CEILING": CUMULATIVE_FORWARD_CEILING
        - cumulative_forward_count,
        "A0": "UNOPENED",
        "A1": "UNOPENED",
        "A2": "UNOPENED",
        "B2A_HISTORICAL_12RUN": "TERMINATED_SUBSTRATE_INADEQUATE",
        "TRACK_B_SCIENTIFIC_QUESTION": "OPEN",
        "B2B_FRESH_SUBSTRATE": "RESERVED_NOT_AUTHORIZED",
        "A3": "CLOSED",
        "TRACK_C": "CLOSED",
        "FORMAL_GATE13": "CLOSED",
        "ACTIVATION_EXTRACTION": "CLOSED",
        "operational_failures": list(operational_failures),
        "mandatory_stop": True,
    }


def _persist_terminal(*, output_dir: Path, terminal: Mapping[str, Any], volume: Any) -> dict[str, Any]:
    _atomic_write_json(output_dir / "terminal_state.json", terminal)
    manifest = _artifact_manifest(output_dir)
    _atomic_write_json(output_dir / "artifact_manifest.json", manifest)
    volume.commit()
    return {**dict(terminal), "artifact_manifest": manifest}


def _all_cases() -> dict[str, dict[str, Any]]:
    from tools.gate13_causal_return.track_a.compile_constrained_channel import (
        all_scientific_cases,
    )

    stages = all_scientific_cases()
    return {str(case["case_id"]): case for rows in stages.values() for case in rows}


class VolumeJournal:
    """Persist each attempt and response to the fresh result Volume atomically."""

    def __init__(
        self,
        *,
        output_dir: Path,
        cases: Mapping[str, Mapping[str, Any]],
        volume: Any,
        original_append: Any,
    ) -> None:
        self.output_dir = output_dir
        self.cases = cases
        self.volume = volume
        self.original_append = original_append

    @staticmethod
    def _stage_slug(stage: str) -> str:
        return stage.lower().replace("_development_preflight", "_development")

    def __call__(self, path: Path, record: Mapping[str, Any]) -> None:
        self.original_append(path, record)
        case_id = str(record.get("case_id") or "")
        if case_id not in self.cases:
            raise ConstrainedModalError("checkpoint references an unknown case")
        stage = str(record.get("stage") or "")
        slug = self._stage_slug(stage)
        is_attempt = path.name.endswith("_attempts.jsonl")
        event = {
            "recorded_at": utc_now(),
            "event": "ATTEMPT_COMMITTED_BEFORE_FORWARD"
            if is_attempt
            else "RESPONSE_COMMITTED",
            "case_id": case_id,
            "stage": stage,
            "cumulative_case_level_forward": int(
                record.get("case_level_model_forward") or 0
            ),
        }
        _atomic_write_json(self.output_dir / "execution_state.json", event)
        if not is_attempt:
            response = str(record.get("response") or "")
            _atomic_write_text(
                self.output_dir / "raw" / slug / f"{case_id}.txt",
                response,
            )
            detail = {
                "schema_version": "gate13_track_a_constrained_case_record_v1",
                **event,
                "raw_output_sha256": sha256_bytes(response.encode("utf-8")),
                "instrument_trace": record.get("instrument_trace"),
                **_parse_case_record(self.cases[case_id], response),
            }
            _atomic_write_json(
                self.output_dir / "cases" / slug / f"{case_id}.json",
                detail,
            )
            state_path = self.output_dir / "checkpoints" / f"{slug}.json"
            existing = read_json(state_path) if state_path.exists() else {
                "completed_case_ids": []
            }
            completed = set(existing.get("completed_case_ids") or [])
            completed.add(case_id)
            _atomic_write_json(
                state_path,
                {
                    "schema_version": "gate13_track_a_constrained_checkpoint_v1",
                    "stage": stage,
                    "completed_case_count": len(completed),
                    "completed_case_ids": sorted(completed),
                    "latest_cumulative_forward": event[
                        "cumulative_case_level_forward"
                    ],
                },
            )
        self.volume.commit()


if modal is not None:
    _repo_root = local_repo_root()
    _image = (
        modal.Image.from_registry(BASE_IMAGE, add_python=None)
        .run_commands(
            "python -m pip install --no-cache-dir --upgrade "
            "pip==26.2.1 setuptools==80.9.0 wheel==0.46.3"
        )
        .pip_install(
            *RUNTIME_REQUIREMENTS,
            extra_index_url=PYTORCH_INDEX_URL,
            extra_options="--no-cache-dir",
        )
        .add_local_dir(
            _repo_root / "tools/gate13_causal_return",
            "/opt/gate13/tools/gate13_causal_return",
            copy=True,
            ignore=["**/__pycache__/**", "**/*.pyc"],
        )
        .add_local_dir(
            _repo_root / "analysis/gate13_causal_return/phase2",
            "/opt/gate13/analysis/gate13_causal_return/phase2",
            copy=True,
            ignore=[
                "phase2_a_modal_execution_authorization.json",
                "phase2_a_modal_execution_authorization_v2.json",
                AUTHORIZATION_FILENAME,
                "**/__pycache__/**",
                "**/*.pyc",
            ],
        )
        .add_local_file(
            _repo_root
            / "tools/gate13_causal_return/modal/modal_track_a_constrained.py",
            "/opt/gate13/modal_track_a_constrained.py",
            copy=True,
        )
        .workdir(REMOTE_ROOT)
        .env({"PYTHONPATH": str(REMOTE_ROOT), "PYTHONUNBUFFERED": "1"})
    )
    app = modal.App(APP_NAME)
    model_volume = modal.Volume.from_name(
        MODEL_VOLUME_NAME,
        create_if_missing=False,
        version=2,
    )
    result_volume = modal.Volume.from_name(
        RESULT_VOLUME_NAME,
        create_if_missing=True,
        version=2,
    )

    @app.function(
        image=_image,
        volumes={str(MODEL_MOUNT): model_volume},
        cpu=2.0,
        memory=8192,
        retries=0,
        timeout=7200,
        max_containers=1,
        single_use_containers=True,
        include_source=False,
        block_network=True,
        name="verify_existing_exact_model_cpu_only",
    )
    def verify_existing_exact_model_cpu_only(
        authorization_text: str, authorization_sha256: str
    ) -> dict[str, Any]:
        auth, validation = _authorization_from_text(
            authorization_text, authorization_sha256
        )
        if auth["model_volume"]["name"] != MODEL_VOLUME_NAME:
            raise ConstrainedModalError("authorized model Volume mismatch")
        report_path = Path(MODEL_ACQUISITION_REPORT.as_posix())
        verified = _verify_model_report(read_json(report_path), rehash=True)
        return {
            "status": "PASS_REUSED_VERIFIED_EXACT_SNAPSHOT_NO_DOWNLOAD",
            "model_volume_name": MODEL_VOLUME_NAME,
            "model_volume_object_id": MODEL_VOLUME_OBJECT_ID,
            "identity": verified,
            "authority_validation": validation,
            "download_attempted": False,
            "gpu_allocated": False,
        }

    @app.function(
        image=_image,
        volumes={str(MODEL_MOUNT): model_volume},
        env={
            "HF_HOME": str(HF_HOME),
            "HF_HUB_CACHE": str(HF_CACHE),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        cpu=4.0,
        memory=16384,
        retries=0,
        timeout=7200,
        max_containers=1,
        single_use_containers=True,
        include_source=False,
        block_network=True,
        name="validate_exact_constrained_channel_cpu_only",
    )
    def validate_exact_constrained_channel_cpu_only() -> dict[str, Any]:
        from tools.gate13_causal_return.modal.validate_constrained_channel import (
            run_exact_constrained_channel_validation,
        )

        report_path = Path(MODEL_ACQUISITION_REPORT.as_posix())
        verified = _verify_model_report(read_json(report_path), rehash=True)
        validation = run_exact_constrained_channel_validation(
            repo_root=Path(REMOTE_ROOT.as_posix()),
            model_snapshot=Path(str(verified["snapshot_directory"])),
        )
        return {
            **validation,
            "model_volume_name": MODEL_VOLUME_NAME,
            "model_volume_object_id": MODEL_VOLUME_OBJECT_ID,
            "model_volume_identity": verified,
            "gpu_allocated": False,
            "model_download_attempted": False,
        }

    @app.function(
        image=_image,
        volumes={
            str(MODEL_MOUNT): model_volume,
            str(RESULT_MOUNT): result_volume,
        },
        env={
            "HF_HOME": str(HF_HOME),
            "HF_HUB_CACHE": str(HF_CACHE),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        gpu=GPU_REQUEST,
        cpu=GPU_CPU_CORES,
        memory=GPU_MEMORY_MIB,
        retries=GPU_RETRIES,
        timeout=GPU_TIMEOUT_SECONDS,
        max_containers=GPU_MAX_CONTAINERS,
        single_use_containers=True,
        include_source=False,
        block_network=True,
        name="execute_constrained_track_a",
    )
    def execute_constrained_track_a(
        authorization_text: str, authorization_sha256: str
    ) -> dict[str, Any]:
        from tools.gate13_causal_return.track_a import constrained_runner
        from tools.gate13_causal_return.track_a import phase2_runner

        gpu_started = time.monotonic()
        auth, authority_validation = _authorization_from_text(
            authorization_text, authorization_sha256
        )
        if auth["result_volume"]["name"] != RESULT_VOLUME_NAME:
            raise ConstrainedModalError("authorized result Volume mismatch")
        output_dir = (
            Path(RESULT_MOUNT.as_posix())
            / "executions"
            / str(auth["execution_identity"])
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        if (output_dir / "terminal_state.json").exists():
            return {
                **read_json(output_dir / "terminal_state.json"),
                "idempotent_terminal_retrieval": True,
            }
        claim = _claim_execution(output_dir, auth, result_volume)
        operational_failures = list(claim.get("operational_restarts") or [])
        parent_lock = read_json(Path(REMOTE_PHASE2_DIR.as_posix()) / "phase2_a_lock.json")
        try:
            try:
                model_report_path = Path(MODEL_ACQUISITION_REPORT.as_posix())
                model_report = read_json(model_report_path)
                model_identity = _verify_model_report(model_report, rehash=True)
            except (OSError, ValueError, RuntimeError) as exc:
                mismatch = {
                    "status": "MODAL_RUNTIME_MISMATCH",
                    "reason": "IMMUTABLE_MODEL_VOLUME_IDENTITY_MISMATCH",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "model_forward_count": 0,
                }
                _atomic_write_json(output_dir / "m0_runtime_report.json", mismatch)
                state = _terminal_base(
                    track_a="MODAL_RUNTIME_MISMATCH",
                    cumulative_forward_count=PRIOR_FORWARD_COUNT,
                    fresh_forward_count=0,
                    operational_failures=operational_failures,
                )
                state.update({"M0": "MODAL_RUNTIME_MISMATCH", "M1": "UNOPENED"})
                return _persist_terminal(
                    output_dir=output_dir, terminal=state, volume=result_volume
                )
            shutil.copy2(model_report_path, output_dir / "model_acquisition_report.json")
            m0 = validate_modal_runtime(parent_lock)
            m0["model_volume_identity"] = model_identity
            m0["authority_validation"] = authority_validation
            m0["offline_execution"] = {
                "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
                "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
                "block_network": True,
                "local_files_only": True,
            }
            if m0["status"] == "PASS":
                from transformers import AutoTokenizer

                tokenizer_probe = AutoTokenizer.from_pretrained(
                    MODEL_REPOSITORY,
                    revision=TOKENIZER_REVISION,
                    use_fast=True,
                    local_files_only=True,
                    trust_remote_code=False,
                )
                template_sha = sha256_bytes(
                    str(tokenizer_probe.chat_template).encode("utf-8")
                )
                tokenizer_pass = bool(tokenizer_probe.is_fast) and template_sha == parent_lock[
                    "runtime_binding"
                ]["chat_template_sha256"]
                m0["checks"]["tokenizer_and_chat_template"] = tokenizer_pass
                m0["tokenizer_forward_zero"] = {
                    "revision": TOKENIZER_REVISION,
                    "use_fast": bool(tokenizer_probe.is_fast),
                    "chat_template_sha256": template_sha,
                    "enable_thinking": False,
                    "add_generation_prompt": True,
                }
                if not tokenizer_pass:
                    m0["status"] = "MODAL_RUNTIME_MISMATCH"
            _atomic_write_json(output_dir / "m0_runtime_report.json", m0)
            result_volume.commit()
            if m0["status"] != "PASS":
                state = _terminal_base(
                    track_a="MODAL_RUNTIME_MISMATCH",
                    cumulative_forward_count=PRIOR_FORWARD_COUNT,
                    fresh_forward_count=0,
                    operational_failures=operational_failures,
                )
                state.update({"M0": "MODAL_RUNTIME_MISMATCH", "M1": "UNOPENED"})
                return _persist_terminal(
                    output_dir=output_dir, terminal=state, volume=result_volume
                )

            import torch

            torch.cuda.reset_peak_memory_stats()
            load_started = time.monotonic()
            torch_module, tokenizer, model = phase2_runner._load_exact_model(parent_lock)
            model_load_seconds = time.monotonic() - load_started
            cases = _all_cases()
            journal = VolumeJournal(
                output_dir=output_dir,
                cases=cases,
                volume=result_volume,
                original_append=constrained_runner._base_append_record,
            )
            scientific = constrained_runner.run_constrained_track_a(
                phase2_dir=Path(REMOTE_PHASE2_DIR.as_posix()),
                output_dir=output_dir,
                model_runtime=(torch_module, tokenizer, model),
                runtime_probe_override=m0,
                append_record=journal,
                require_clean=False,
            )
            _atomic_write_json(output_dir / "scientific_result.json", scientific)
            result_volume.commit()
            cumulative = int(scientific["model_forward_count"])
            fresh = int(scientific["fresh_model_forward_count"])
            if cumulative > CUMULATIVE_FORWARD_CEILING or fresh > MAXIMUM_ADDITIONAL_FORWARDS:
                raise ConstrainedModalError("scientific runner exceeded forward authority")
            if scientific["M1"] != "PASS":
                track_state = "CONSTRAINED_CHANNEL_IMPLEMENTATION_BLOCKER"
            elif scientific["TRACK_A_A0"] == "FAIL":
                track_state = "A0_FAIL"
            elif scientific["TRACK_A_A1"] == "FAIL":
                track_state = "A1_FAIL"
            elif scientific["TRACK_A_A2"] == "PASS":
                track_state = "A2_PASS"
            elif scientific["TRACK_A_A2"] == "FAIL":
                track_state = "A2_FAIL"
            else:
                raise ConstrainedModalError("runner returned a non-terminal ladder state")
            terminal = _terminal_base(
                track_a=track_state,
                cumulative_forward_count=cumulative,
                fresh_forward_count=fresh,
                operational_failures=operational_failures,
            )
            terminal.update(
                {
                    "execution_identity": auth["execution_identity"],
                    "M0": "PASS",
                    "M1": scientific["M1"],
                    "A0": "UNOPENED"
                    if scientific["TRACK_A_A0"] == "UNOPENED"
                    else f"A0_{scientific['TRACK_A_A0']}",
                    "A1": "UNOPENED"
                    if scientific["TRACK_A_A1"] == "UNOPENED"
                    else f"A1_{scientific['TRACK_A_A1']}",
                    "A2": "UNOPENED"
                    if scientific["TRACK_A_A2"] == "UNOPENED"
                    else f"A2_{scientific['TRACK_A_A2']}",
                    "model_load_seconds": model_load_seconds,
                    "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
                }
            )
            gpu_elapsed = time.monotonic() - gpu_started
            estimated_new = gpu_elapsed * GPU_TOTAL_RATE_USD_PER_SECOND
            estimated_cumulative = PRIOR_SPEND_RESERVATION_USD + estimated_new
            cumulative_wall = PRIOR_GPU_WALL_RESERVATION_SECONDS + gpu_elapsed
            if estimated_cumulative > CUMULATIVE_SPEND_CEILING_USD:
                raise ConstrainedModalError("estimated cumulative Modal usage exceeded ceiling")
            if cumulative_wall > CUMULATIVE_GPU_WALL_CEILING_SECONDS:
                raise ConstrainedModalError("cumulative GPU wall exceeded ceiling")
            terminal.update(
                {
                    "gpu_elapsed_seconds": gpu_elapsed,
                    "estimated_new_modal_gpu_function_usage_usd": estimated_new,
                    "estimated_cumulative_modal_usage_usd": estimated_cumulative,
                    "cumulative_gpu_wall_seconds": cumulative_wall,
                    "provider_actual_usage_status": "PENDING_PROVIDER_BILLING_OBSERVATION",
                }
            )
            return _persist_terminal(
                output_dir=output_dir,
                terminal=terminal,
                volume=result_volume,
            )
        except Exception as exc:
            attempt_paths = list(output_dir.rglob("*_attempts.jsonl"))
            response_paths = [
                path
                for path in output_dir.rglob("*_state.jsonl")
                if not path.name.endswith("_attempts.jsonl")
            ]
            fresh_attempts = sum(
                1
                for path in attempt_paths
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            responses = sum(
                1
                for path in response_paths
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            failure = {
                "observed_at": utc_now(),
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "fresh_attempt_count": fresh_attempts,
                "persisted_response_count": responses,
                "post_response_change_forbidden": responses > 0,
            }
            operational_failures.append(failure)
            _atomic_write_json(output_dir / "execution_blocker.json", failure)
            track_state = (
                "SCIENTIFIC_RUNNER_BLOCKER"
                if responses > 0
                else "CONSTRAINED_CHANNEL_IMPLEMENTATION_BLOCKER"
            )
            terminal = _terminal_base(
                track_a=track_state,
                cumulative_forward_count=PRIOR_FORWARD_COUNT + fresh_attempts,
                fresh_forward_count=fresh_attempts,
                operational_failures=operational_failures,
            )
            terminal.update(
                {
                    "execution_identity": auth["execution_identity"],
                    "M0": "PASS"
                    if (output_dir / "m0_runtime_report.json").exists()
                    else "UNOPENED",
                    "M1": "BLOCKED",
                }
            )
            return _persist_terminal(
                output_dir=output_dir,
                terminal=terminal,
                volume=result_volume,
            )

    @app.local_entrypoint()
    def prepare_resources() -> None:
        _image.hydrate()
        model_volume.hydrate()
        result_volume.hydrate()
        print(
            json.dumps(
                {
                    "schema_version": "gate13_track_a_constrained_resource_identity_v1",
                    "app_name": APP_NAME,
                    "modal_image_object_id": _image.object_id,
                    "modal_image_definition": image_definition_payload(),
                    "modal_image_definition_sha256": image_definition_sha256(),
                    "model_volume": {
                        "name": MODEL_VOLUME_NAME,
                        "object_id": model_volume.object_id,
                    },
                    "result_volume": {
                        "name": RESULT_VOLUME_NAME,
                        "object_id": result_volume.object_id,
                    },
                    "gpu_allocated": False,
                    "model_downloaded": False,
                },
                sort_keys=True,
            )
        )

    @app.local_entrypoint()
    def run_exact_validation(control_output: str) -> None:
        validation = validate_exact_constrained_channel_cpu_only.remote()
        control = {
            "schema_version": "gate13_track_a_constrained_exact_control_v1",
            "app_name": APP_NAME,
            "modal_image_object_id": _image.object_id,
            "modal_image_definition": image_definition_payload(),
            "modal_image_definition_sha256": image_definition_sha256(),
            "model_volume": {
                "name": MODEL_VOLUME_NAME,
                "object_id": model_volume.object_id,
            },
            "result_volume": {
                "name": RESULT_VOLUME_NAME,
                "object_id": result_volume.object_id,
            },
            "validation": validation,
        }
        _atomic_write_json(Path(control_output).resolve(), control)
        print(json.dumps(control, sort_keys=True))

    @app.local_entrypoint()
    def run_authorized(authorization: str, control_output: str) -> None:
        authorization_path = Path(authorization).resolve()
        auth_text = authorization_path.read_text(encoding="utf-8")
        auth_sha = sha256_bytes(auth_text.encode("utf-8"))
        validation = validate_constrained_modal_execution_authority(
            authorization_path=authorization_path,
            phase2_dir=local_repo_root()
            / "analysis/gate13_causal_return/phase2",
            repo_root=local_repo_root(),
            verify_git=True,
        )
        acquisition = verify_existing_exact_model_cpu_only.remote(
            auth_text, auth_sha
        )
        execution = execute_constrained_track_a.remote(auth_text, auth_sha)
        control = {
            "schema_version": "gate13_track_a_constrained_modal_control_v1",
            "authorization_sha256": auth_sha,
            "local_authority_validation": validation,
            "model_acquisition": acquisition,
            "execution": execution,
        }
        _atomic_write_json(Path(control_output).resolve(), control)
        print(json.dumps(control, sort_keys=True))

else:  # pragma: no cover
    app = None
    model_volume = None
    result_volume = None
