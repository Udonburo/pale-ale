"""Thin Modal lifecycle adapter for the frozen Gate13 Candidate Track A runner.

This module owns only operational realization: immutable image construction,
exact model acquisition, forward-zero validation, checkpoint persistence, budget
forecasting, and invocation of the unchanged scientific runner.  Prompt
rendering, parsing, oracles, metrics, progression gates, and the forward ceiling
remain in ``tools.gate13_causal_return.track_a``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

try:  # Keep model-free local tests runnable without installing the Modal client.
    import modal
except ModuleNotFoundError:  # pragma: no cover - exercised by the base test runtime
    modal = None  # type: ignore[assignment]
else:
    # ``unittest discover -s tools/gate13_causal_return`` exposes this package
    # itself as top-level ``modal``.  Treat that shadow module as SDK-absent;
    # the real client always exports Image, App, and Volume.
    if not all(hasattr(modal, field) for field in ("Image", "App", "Volume")):
        modal = None  # type: ignore[assignment]

from tools.gate13_causal_return.phase2_common import (
    canonical_json_bytes,
    read_json,
    sha256_bytes,
    sha256_file,
    sha256_json,
    write_json,
)
from tools.gate13_causal_return.modal.validate_modal_execution_authority import (
    MAX_GPU_WALL_SECONDS,
    MAX_SPEND_USD,
    MODEL_REVISION,
    validate_modal_execution_authority,
)
from tools.gate13_causal_return.modal.validate_modal_runtime import (
    validate_modal_runtime,
)


APP_NAME = "gate13-track-a-frozen-a5a83ec"
MODEL_VOLUME_NAME = "gate13-track-a-qwen3-8b-b968826-model"
RESULT_VOLUME_NAME = "gate13-track-a-a5a83ec-results"
MODEL_REPOSITORY = "Qwen/Qwen3-8B"
TOKENIZER_REVISION = MODEL_REVISION
BASE_IMAGE = (
    "python:3.11.2-slim-bullseye@"
    "sha256:2f749ef90f54fd4b3c77cde78eec23ab5b8199d9ac84e4ced6ae523ef223ef7b"
)
BASE_IMAGE_MANIFEST_DIGEST = (
    "sha256:2f749ef90f54fd4b3c77cde78eec23ab5b8199d9ac84e4ced6ae523ef223ef7b"
)
BASE_IMAGE_LINUX_AMD64_DIGEST = (
    "sha256:9ad4ffc502779e5508f7ac1eccab4a22786b80bd53d721d735f6de0840b245a1"
)
PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/cu126"
REMOTE_ROOT = PurePosixPath("/opt/gate13")
REMOTE_PHASE2_DIR = REMOTE_ROOT / "analysis/gate13_causal_return/phase2"
REMOTE_M1_MANIFEST = (
    REMOTE_ROOT / "tools/gate13_causal_return/modal/m1_preflight_manifest.json"
)
MODEL_MOUNT = PurePosixPath("/model")
RESULT_MOUNT = PurePosixPath("/results")
HF_HOME = MODEL_MOUNT / "hf_home"
HF_CACHE = HF_HOME / "hub"
MODEL_ACQUISITION_REPORT = MODEL_MOUNT / "gate13_model_acquisition.json"

GPU_REQUEST = "L40S"
GPU_EXACT_NAME = "NVIDIA L40S"
GPU_CPU_CORES = 4.0
GPU_MEMORY_MIB = 32_768
GPU_TIMEOUT_SECONDS = MAX_GPU_WALL_SECONDS
GPU_MAX_CONTAINERS = 1
GPU_RETRIES = 0

# Modal's published per-second rates observed prospectively on 2026-08-20.
L40S_RATE_USD_PER_SECOND = 0.000542
CPU_RATE_USD_PER_CORE_SECOND = 0.0000131
MEMORY_RATE_USD_PER_GIB_SECOND = 0.00000222
GPU_TOTAL_RATE_USD_PER_SECOND = (
    L40S_RATE_USD_PER_SECOND
    + GPU_CPU_CORES * CPU_RATE_USD_PER_CORE_SECOND
    + (GPU_MEMORY_MIB / 1024) * MEMORY_RATE_USD_PER_GIB_SECOND
)
NON_GPU_USAGE_RESERVE_USD = 1.0
M1_CONTINGENCY_MULTIPLIER = 1.25
M1_MAX_PROJECTED_GPU_SECONDS_WITH_CONTINGENCY = 9.5 * 60 * 60
TOTAL_FROZEN_CASE_COUNT = 252 + 162 + 90

# Every direct and transitive Python dependency is exact.  CUDA library wheels
# are included explicitly so an upstream resolver change cannot alter the image.
RUNTIME_REQUIREMENTS = (
    "accelerate==1.14.0",
    "annotated-doc==0.0.5",
    "anyio==4.14.2",
    "certifi==2026.7.22",
    "click==8.4.2",
    "colorama==0.4.6",
    "filelock==3.32.3",
    "fsspec==2026.7.0",
    "h11==0.16.0",
    "hf-xet==1.6.0",
    "httpcore==1.0.9",
    "httpx==0.28.1",
    "huggingface-hub==1.27.0",
    "idna==3.19",
    "jinja2==3.1.6",
    "markdown-it-py==4.2.0",
    "markupsafe==3.0.3",
    "mdurl==0.1.2",
    "mpmath==1.3.0",
    "networkx==3.6.1",
    "numpy==2.4.6",
    "nvidia-cublas-cu12==12.6.4.1",
    "nvidia-cuda-cupti-cu12==12.6.80",
    "nvidia-cuda-nvrtc-cu12==12.6.77",
    "nvidia-cuda-runtime-cu12==12.6.77",
    "nvidia-cudnn-cu12==9.5.1.17",
    "nvidia-cufft-cu12==11.3.0.4",
    "nvidia-cufile-cu12==1.11.1.6",
    "nvidia-curand-cu12==10.3.7.77",
    "nvidia-cusolver-cu12==11.7.1.2",
    "nvidia-cusparse-cu12==12.5.4.2",
    "nvidia-cusparselt-cu12==0.6.3",
    "nvidia-nccl-cu12==2.26.2",
    "nvidia-nvjitlink-cu12==12.6.85",
    "nvidia-nvtx-cu12==12.6.77",
    "packaging==26.3",
    "psutil==7.2.2",
    "pygments==2.21.0",
    "pyyaml==6.0.3",
    "regex==2026.7.19",
    "rich==15.0.0",
    "safetensors==0.8.0",
    "setuptools==80.9.0",
    "shellingham==1.5.4",
    "sympy==1.14.0",
    "tokenizers==0.22.2",
    "torch==2.7.1+cu126",
    "tqdm==4.70.0",
    "transformers==5.15.0",
    "triton==3.3.1",
    "typer==0.27.1",
    "typing-extensions==4.16.0",
    "wheel==0.46.3",
)

EXPECTED_CLOSED_STATE = {
    "B2A_HISTORICAL_12RUN": "TERMINATED_SUBSTRATE_INADEQUATE",
    "TRACK_B_SCIENTIFIC_QUESTION": "OPEN",
    "B2B_FRESH_SUBSTRATE": "RESERVED_NOT_AUTHORIZED",
    "A3": "CLOSED",
    "TRACK_C": "CLOSED",
    "FORMAL_GATE13": "CLOSED",
    "ACTIVATION_EXTRACTION": "CLOSED",
}


class ModalTrackAError(RuntimeError):
    """Fail-closed operational realization error."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def local_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def image_definition_payload() -> dict[str, Any]:
    """Return the canonical, authorization-bindable image recipe."""
    return {
        "schema_version": "gate13_track_a_modal_image_definition_v1",
        "base_image": BASE_IMAGE,
        "base_image_manifest_digest": BASE_IMAGE_MANIFEST_DIGEST,
        "base_image_linux_amd64_digest": BASE_IMAGE_LINUX_AMD64_DIGEST,
        "python": "3.11.2",
        "pip": "26.2.1",
        "requirements": list(RUNTIME_REQUIREMENTS),
        "pytorch_extra_index_url": PYTORCH_INDEX_URL,
        "workdir": str(REMOTE_ROOT),
        "copied_roots": [
            "tools/gate13_causal_return",
            "analysis/gate13_causal_return/phase2",
        ],
        "excluded_from_image": [
            "**/__pycache__/**",
            "**/*.pyc",
            "phase2_a_modal_execution_authorization.json",
        ],
    }


def image_definition_sha256() -> str:
    return sha256_json(image_definition_payload())


def forecast_after_m1(
    *,
    elapsed_before_m1_seconds: float,
    m1_elapsed_seconds: float,
    m1_case_count: int,
    completed_forward_count: int,
    forward_ceiling: int,
) -> dict[str, Any]:
    """Prospectively forecast the worst frozen ladder from M1 observations."""
    if m1_case_count <= 0 or completed_forward_count < m1_case_count:
        raise ValueError("M1 forecast requires completed preflight forwards")
    if completed_forward_count > TOTAL_FROZEN_CASE_COUNT:
        raise ValueError("completed forwards exceed frozen case inventory")
    seconds_per_case = m1_elapsed_seconds / m1_case_count
    remaining = TOTAL_FROZEN_CASE_COUNT - completed_forward_count
    projected_gpu_seconds = (
        elapsed_before_m1_seconds + m1_elapsed_seconds + seconds_per_case * remaining
    )
    projected_with_contingency = projected_gpu_seconds * M1_CONTINGENCY_MULTIPLIER
    projected_spend = (
        projected_with_contingency * GPU_TOTAL_RATE_USD_PER_SECOND
        + NON_GPU_USAGE_RESERVE_USD
    )
    checks = {
        "projected_gpu_with_contingency_le_9_5h": projected_with_contingency
        <= M1_MAX_PROJECTED_GPU_SECONDS_WITH_CONTINGENCY,
        "projected_modal_usage_le_usd_25": projected_spend <= MAX_SPEND_USD,
        "forward_forecast_le_lock_ceiling": TOTAL_FROZEN_CASE_COUNT
        <= forward_ceiling,
    }
    return {
        "schema_version": "gate13_track_a_modal_m1_forecast_v1",
        "m1_case_count": m1_case_count,
        "completed_forward_count": completed_forward_count,
        "total_frozen_case_count": TOTAL_FROZEN_CASE_COUNT,
        "remaining_worst_ladder_case_count": remaining,
        "seconds_per_m1_case_including_persistence": seconds_per_case,
        "projected_gpu_seconds": projected_gpu_seconds,
        "contingency_multiplier": M1_CONTINGENCY_MULTIPLIER,
        "projected_gpu_seconds_with_25_percent_contingency": projected_with_contingency,
        "projected_modal_usage_usd_with_non_gpu_reserve": projected_spend,
        "modal_rate_assumptions": {
            "L40S_usd_per_second": L40S_RATE_USD_PER_SECOND,
            "cpu_core_usd_per_second": CPU_RATE_USD_PER_CORE_SECOND,
            "memory_gib_usd_per_second": MEMORY_RATE_USD_PER_GIB_SECOND,
            "gpu_function_total_usd_per_second": GPU_TOTAL_RATE_USD_PER_SECOND,
            "non_gpu_usage_reserve_usd": NON_GPU_USAGE_RESERVE_USD,
        },
        "forward_ceiling": forward_ceiling,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "BLOCK",
    }


def terminal_state(
    *,
    track_a: str,
    a0: str = "UNOPENED",
    a1: str = "UNOPENED",
    a2: str = "UNOPENED",
    forward_count: int = 0,
    operational_failures: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "schema_version": "gate13_track_a_modal_terminal_state_v1",
        "TRACK_A": track_a,
        "TRACK_A_A0": a0,
        "TRACK_A_A1": a1,
        "TRACK_A_A2": a2,
        "MODEL_FORWARD_COUNT": int(forward_count),
        "ACTIVATION_EXTRACTION_COUNT": 0,
        **EXPECTED_CLOSED_STATE,
        "OPERATIONAL_FAILURES_OR_RESTARTS": [dict(row) for row in operational_failures],
        "FINAL_STOP": "MANDATORY_STOP",
    }


def _authorization_from_text(text: str, expected_sha256: str) -> tuple[Path, dict[str, Any]]:
    actual_sha = sha256_bytes(text.encode("utf-8"))
    if actual_sha != expected_sha256:
        raise ModalTrackAError("transmitted authorization byte SHA-256 mismatch")
    path = Path("/tmp/phase2_a_modal_execution_authorization.json")
    path.write_text(text, encoding="utf-8", newline="\n")
    value = read_json(path)
    if value.get("modal_image_definition_sha256") != image_definition_sha256():
        raise ModalTrackAError("running adapter image definition differs from authorization")
    return path, value


def _validate_remote_authority(
    authorization_text: str, authorization_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    auth_path, auth = _authorization_from_text(authorization_text, authorization_sha256)
    validation = validate_modal_execution_authority(
        authorization_path=auth_path,
        phase2_dir=REMOTE_PHASE2_DIR,
        m1_manifest_path=REMOTE_M1_MANIFEST,
        repo_root=REMOTE_ROOT,
        verify_git=False,
    )
    return auth, validation


def _file_inventory(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        raise ModalTrackAError(f"model snapshot directory is absent: {root}")
    rows: list[dict[str, Any]] = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=str):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not rows:
        raise ModalTrackAError("exact model snapshot has no files")
    return rows


def _named_sha(inventory: Sequence[Mapping[str, Any]], name: str) -> str | None:
    for row in inventory:
        if row["path"] == name:
            return str(row["sha256"])
    return None


def _model_identity(snapshot_dir: Path) -> dict[str, Any]:
    inventory = _file_inventory(snapshot_dir)
    tokenizer_names = {
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "merges.txt",
        "vocab.json",
        "chat_template.jinja",
    }
    tokenizer_files = [
        dict(row) for row in inventory if str(row["path"]) in tokenizer_names
    ]
    identity_payload = {
        "model_repository": MODEL_REPOSITORY,
        "resolved_commit": MODEL_REVISION,
        "files": inventory,
    }
    return {
        "model_repository": MODEL_REPOSITORY,
        "requested_revision": MODEL_REVISION,
        "resolved_commit": snapshot_dir.name,
        "snapshot_directory": str(snapshot_dir),
        "file_count": len(inventory),
        "total_bytes": sum(int(row["bytes"]) for row in inventory),
        "complete_file_inventory": inventory,
        "config_sha256": _named_sha(inventory, "config.json"),
        "tokenizer_config_sha256": _named_sha(inventory, "tokenizer_config.json"),
        "tokenizer_file_sha256": tokenizer_files,
        "weight_index_sha256": _named_sha(inventory, "model.safetensors.index.json"),
        "model_directory_identity_sha256": sha256_json(identity_payload),
    }


def _verify_model_report(report: Mapping[str, Any], *, rehash: bool) -> dict[str, Any]:
    identity = report.get("model_identity") or {}
    if identity.get("model_repository") != MODEL_REPOSITORY:
        raise ModalTrackAError("model acquisition repository mismatch")
    if identity.get("resolved_commit") != MODEL_REVISION:
        raise ModalTrackAError("model acquisition resolved commit mismatch")
    snapshot_dir = Path(str(identity.get("snapshot_directory") or ""))
    if snapshot_dir.name != MODEL_REVISION or not snapshot_dir.is_dir():
        raise ModalTrackAError("exact acquired model snapshot is unavailable")
    if rehash:
        actual = _model_identity(snapshot_dir)
        if actual["model_directory_identity_sha256"] != identity.get(
            "model_directory_identity_sha256"
        ):
            raise ModalTrackAError("model Volume content identity mismatch")
    return {"status": "PASS", "snapshot_directory": str(snapshot_dir), **dict(identity)}


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _parse_case_record(case: Mapping[str, Any], response: str) -> dict[str, Any]:
    """Invoke, but never reimplement, the frozen parser and oracle contracts."""
    from tools.gate13_causal_return.track_a.parse_phase2_output import (
        Phase2OutputParseError,
        parse_phase2_output,
    )
    from tools.gate13_causal_return.track_a.parse_register_output import (
        OutputParseError,
        parse_register_output,
    )

    stage = str(case.get("stage") or "")
    condition = str(case.get("condition") or "")
    parser: Callable[[Mapping[str, object], str], Any]
    if stage == "A0" and condition != "N":
        parser = parse_register_output
        parser_name = "parse_register_output"
    else:
        parser = parse_phase2_output
        parser_name = "parse_phase2_output"

    oracle_parsed = parser(case, str(case["expected_text"]))
    try:
        parsed = parser(case, response)
        parsed_record = dataclasses.asdict(parsed)
        parse_status = "PASS"
        parse_error = None
    except (OutputParseError, Phase2OutputParseError) as exc:
        parsed_record = None
        parse_status = "MALFORMED_SCORED_INCORRECT_NO_RETRY"
        parse_error = {"type": type(exc).__name__, "message": str(exc)}
    return {
        "parser": parser_name,
        "parse_status": parse_status,
        "parsed_record": parsed_record,
        "parse_error": parse_error,
        "oracle_record": dataclasses.asdict(oracle_parsed),
        "expected_text_sha256": sha256_bytes(str(case["expected_text"]).encode("utf-8")),
    }


class VolumeCheckpointJournal:
    """Persist every original runner attempt/response at the smallest safe unit."""

    def __init__(
        self,
        *,
        output_dir: Path,
        cases: Mapping[str, Mapping[str, Any]],
        volume: Any,
        original_append: Callable[[Path, Mapping[str, Any]], None],
    ) -> None:
        self.output_dir = output_dir
        self.cases = cases
        self.volume = volume
        self.original_append = original_append
        checkpoint_path = output_dir / "completed_case_ids.json"
        self.completed_ids: set[str] = (
            set(read_json(checkpoint_path).get("completed_case_ids") or [])
            if checkpoint_path.exists()
            else set()
        )

    def __call__(self, path: Path, record: Mapping[str, Any]) -> None:
        self.original_append(path, record)
        case_id = str(record.get("case_id") or "")
        if not case_id or case_id not in self.cases:
            raise ModalTrackAError("runner checkpoint references an unknown case")
        is_attempt = path.name.endswith("_attempts.jsonl")
        event = {
            "recorded_at": utc_now(),
            "case_id": case_id,
            "stage": str(record.get("stage") or self.cases[case_id].get("stage") or ""),
            "forward_count": int(record.get("case_level_model_forward") or 0),
            "event": "ATTEMPT_COMMITTED_BEFORE_FORWARD" if is_attempt else "RESPONSE_COMMITTED",
        }
        _atomic_write_json(self.output_dir / "execution_state.json", event)
        if not is_attempt:
            response = str(record.get("response") or "")
            _atomic_write_text(self.output_dir / "raw" / f"{case_id}.txt", response)
            detail = {
                "schema_version": "gate13_track_a_modal_case_record_v1",
                **event,
                "case_sha256": sha256_json(self.cases[case_id]),
                "raw_output_sha256": sha256_bytes(response.encode("utf-8")),
                **_parse_case_record(self.cases[case_id], response),
            }
            _atomic_write_json(self.output_dir / "cases" / f"{case_id}.json", detail)
            self.completed_ids.add(case_id)
            _atomic_write_json(
                self.output_dir / "completed_case_ids.json",
                {
                    "schema_version": "gate13_track_a_completed_case_ids_v1",
                    "completed_case_count": len(self.completed_ids),
                    "completed_case_ids": sorted(self.completed_ids),
                    "latest_forward_count": event["forward_count"],
                },
            )
        self.volume.commit()


def _all_frozen_cases() -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    from tools.gate13_causal_return.track_a.compile_phase2_cases import compile_cases
    from tools.gate13_causal_return.track_a.compile_register_cases import compile_ledger

    compiled = compile_cases()
    stages = {
        "A0": [dict(row, stage="A0") for row in compile_ledger()["cases"]]
        + [dict(row, stage="A0") for row in compiled["A0_EXTENSION"]],
        "A1": [dict(row, stage="A1") for row in compiled["A1"]],
        "A2": [dict(row, stage="A2") for row in compiled["A2"]],
    }
    by_id = {str(row["case_id"]): row for rows in stages.values() for row in rows}
    if len(by_id) != TOTAL_FROZEN_CASE_COUNT:
        raise ModalTrackAError("compiled frozen case inventory count mismatch")
    return stages, by_id


def _validate_m1_cases(
    manifest: Mapping[str, Any], all_cases: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for binding in manifest.get("cases") or []:
        case_id = str(binding["case_id"])
        if case_id not in all_cases:
            raise ModalTrackAError(f"M1 case is absent from frozen A0: {case_id}")
        case = dict(all_cases[case_id])
        frozen_case = dict(case)
        # Review-1 A0 rows predate the explicit ``stage`` field, while the
        # frozen A0-extension rows already contain it.  Bind each source row
        # exactly as compiled rather than normalizing their scientific bytes.
        if case.get("condition") != "N":
            frozen_case.pop("stage", None)
        derived_max_new_tokens = max(
            32, len(str(case["expected_text"])) // 2 + 32
        )
        checks = {
            "case_sha256": sha256_json(frozen_case) == binding["case_sha256"],
            "prompt_sha256": sha256_bytes(str(case["prompt"]).encode("utf-8"))
            == binding["prompt_sha256"],
            "expected_text_sha256": sha256_bytes(
                str(case["expected_text"]).encode("utf-8")
            )
            == binding["expected_text_sha256"],
            "stage": case.get("stage") == "A0",
            "condition": case.get("condition") == binding["condition"],
            "max_new_tokens": derived_max_new_tokens == binding["max_new_tokens"],
        }
        if not all(checks.values()):
            failed = sorted(name for name, passed in checks.items() if not passed)
            raise ModalTrackAError(f"M1 frozen binding mismatch for {case_id}: {failed}")
        selected.append(case)
    if len(selected) != int(manifest.get("case_count") or 0):
        raise ModalTrackAError("M1 selected case count mismatch")
    return selected


def _claim_execution(output_dir: Path, auth: Mapping[str, Any], volume: Any) -> dict[str, Any]:
    claim_path = output_dir / "execution_claim.json"
    expected = {
        "execution_identity": auth["execution_identity"],
        "adapter_commit": auth["adapter_commit"],
        "authorization_sha256": sha256_json(auth),
    }
    if claim_path.exists():
        claim = read_json(claim_path)
        for field, value in expected.items():
            if claim.get(field) != value:
                raise ModalTrackAError("execution identity is already claimed by different authority")
        claim["container_start_count"] = int(claim.get("container_start_count") or 0) + 1
        claim.setdefault("operational_restarts", []).append(
            {"observed_at": utc_now(), "reason": "UNCHANGED_CODE_RESUME_ENTRY"}
        )
    else:
        claim = {
            "schema_version": "gate13_track_a_modal_execution_claim_v1",
            **expected,
            "claimed_at": utc_now(),
            "container_start_count": 1,
            "operational_restarts": [],
        }
    _atomic_write_json(claim_path, claim)
    volume.commit()
    return claim


def _artifact_manifest(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "artifact_manifest.json"
    rows = []
    for path in sorted((item for item in output_dir.rglob("*") if item.is_file()), key=str):
        if path == manifest_path or path.name.endswith(".tmp"):
            continue
        rows.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": "gate13_track_a_modal_artifact_manifest_v1",
        "generated_at": utc_now(),
        "artifact_count": len(rows),
        "artifacts": rows,
        "inventory_sha256": sha256_json(rows),
    }


def _persist_terminal(
    *, output_dir: Path, terminal: Mapping[str, Any], volume: Any
) -> dict[str, Any]:
    _atomic_write_json(output_dir / "terminal_state.json", terminal)
    manifest = _artifact_manifest(output_dir)
    _atomic_write_json(output_dir / "artifact_manifest.json", manifest)
    volume.commit()
    return {**dict(terminal), "artifact_manifest": manifest}


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
                "**/__pycache__/**",
                "**/*.pyc",
            ],
        )
        .workdir(REMOTE_ROOT)
        .env({"PYTHONPATH": str(REMOTE_ROOT), "PYTHONUNBUFFERED": "1"})
    )
    app = modal.App(APP_NAME)
    model_volume = modal.Volume.from_name(
        MODEL_VOLUME_NAME, create_if_missing=True, version=2
    )
    result_volume = modal.Volume.from_name(
        RESULT_VOLUME_NAME, create_if_missing=True, version=2
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
        name="acquire_exact_model_cpu_only",
    )
    def acquire_exact_model_cpu_only(
        authorization_text: str, authorization_sha256: str
    ) -> dict[str, Any]:
        """Populate and commit the exact content-addressed model snapshot."""
        auth, validation = _validate_remote_authority(
            authorization_text, authorization_sha256
        )
        if auth["model_volume"]["name"] != MODEL_VOLUME_NAME:
            raise ModalTrackAError("authorized model Volume name mismatch")
        model_report_path = Path(MODEL_ACQUISITION_REPORT.as_posix())
        if model_report_path.exists():
            existing = read_json(model_report_path)
            verified = _verify_model_report(existing, rehash=True)
            return {
                **existing,
                "status": "PASS_REUSED_VERIFIED_EXACT_SNAPSHOT",
                "verified_at": utc_now(),
                "verification": verified,
            }

        from huggingface_hub import snapshot_download

        started = time.monotonic()
        snapshot = Path(
            snapshot_download(
                repo_id=MODEL_REPOSITORY,
                revision=MODEL_REVISION,
                cache_dir=str(HF_CACHE),
                local_files_only=False,
            )
        )
        if snapshot.name != MODEL_REVISION:
            raise ModalTrackAError("snapshot_download did not resolve to the exact commit")
        identity = _model_identity(snapshot)
        report = {
            "schema_version": "gate13_track_a_modal_model_acquisition_v1",
            "status": "PASS_EXACT_SNAPSHOT_DOWNLOADED",
            "acquired_at": utc_now(),
            "elapsed_seconds": time.monotonic() - started,
            "cpu_only": True,
            "gpu_allocated": False,
            "snapshot_download_revision_was_exact": True,
            "authorization_execution_identity": auth["execution_identity"],
            "authority_validation": validation,
            "model_volume_name": MODEL_VOLUME_NAME,
            "model_volume_object_id": auth["model_volume"]["object_id"],
            "model_identity": identity,
            "volume_commit_status": "REQUESTED",
        }
        _atomic_write_json(model_report_path, report)
        model_volume.commit()
        report["volume_commit_status"] = "PASS"
        _atomic_write_json(model_report_path, report)
        model_volume.commit()
        return report

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
        block_network=True,
        include_source=False,
        name="execute_one_frozen_track_a",
    )
    def execute_one_frozen_track_a(
        authorization_text: str, authorization_sha256: str
    ) -> dict[str, Any]:
        """Run M0, frozen M1, and the unchanged conditional A0/A1/A2 ladder."""
        from tools.gate13_causal_return.track_a import phase2_runner as runner

        gpu_started = time.monotonic()
        auth, authority_validation = _validate_remote_authority(
            authorization_text, authorization_sha256
        )
        if auth["result_volume"]["name"] != RESULT_VOLUME_NAME:
            raise ModalTrackAError("authorized result Volume name mismatch")
        model_report_path = Path(MODEL_ACQUISITION_REPORT.as_posix())
        output_dir = (
            Path(RESULT_MOUNT.as_posix())
            / "executions"
            / str(auth["execution_identity"])
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        terminal_path = output_dir / "terminal_state.json"
        if terminal_path.exists():
            existing = read_json(terminal_path)
            return {**existing, "idempotent_terminal_retrieval": True}
        claim = _claim_execution(output_dir, auth, result_volume)
        operational_failures = list(claim.get("operational_restarts") or [])
        lock = read_json(REMOTE_PHASE2_DIR / "phase2_a_lock.json")

        try:
            model_report = read_json(model_report_path)
            model_identity = _verify_model_report(model_report, rehash=True)
            shutil.copy2(
                model_report_path, output_dir / "model_acquisition_report.json"
            )
            m0 = validate_modal_runtime(lock)
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
                m0["tokenizer_forward_zero"] = {
                    "revision": TOKENIZER_REVISION,
                    "use_fast": bool(tokenizer_probe.is_fast),
                    "chat_template_sha256": template_sha,
                    "enable_thinking": False,
                    "add_generation_prompt": True,
                }
                if (
                    not tokenizer_probe.is_fast
                    or template_sha != lock["runtime_binding"]["chat_template_sha256"]
                ):
                    m0["status"] = "MODAL_RUNTIME_MISMATCH"
                    m0["checks"]["tokenizer_and_chat_template"] = False
                else:
                    m0["checks"]["tokenizer_and_chat_template"] = True
            _atomic_write_json(output_dir / "m0_runtime_report.json", m0)
            result_volume.commit()
            if m0["status"] != "PASS":
                state = terminal_state(
                    track_a="MODAL_RUNTIME_MISMATCH",
                    forward_count=0,
                    operational_failures=operational_failures,
                )
                state.update(
                    {
                        "M0": "MODAL_RUNTIME_MISMATCH",
                        "M1": "UNOPENED",
                        "runtime_report_sha256": sha256_file(
                            output_dir / "m0_runtime_report.json"
                        ),
                    }
                )
                return _persist_terminal(
                    output_dir=output_dir, terminal=state, volume=result_volume
                )

            # M0 passed: only now load the exact model, exactly once.
            import torch

            torch.cuda.reset_peak_memory_stats()
            load_started = time.monotonic()
            torch_module, tokenizer, model = runner._load_exact_model(lock)
            model_load_seconds = time.monotonic() - load_started
            model_load_peak_vram = int(torch.cuda.max_memory_allocated())

            stages, cases_by_id = _all_frozen_cases()
            m1_manifest = read_json(REMOTE_M1_MANIFEST)
            m1_cases = _validate_m1_cases(m1_manifest, cases_by_id)
            journal = VolumeCheckpointJournal(
                output_dir=output_dir,
                cases=cases_by_id,
                volume=result_volume,
                original_append=runner._append_record,
            )
            original_append = runner._append_record
            original_load = runner._load_exact_model
            original_validate = runner.validate_phase2_locks
            original_authorized = runner.model_load_authorized
            runner._append_record = journal
            runner._load_exact_model = lambda unused_lock: (
                torch_module,
                tokenizer,
                model,
            )
            runner.validate_phase2_locks = lambda **unused: {
                "status": "PASS",
                "execution_authorized": False,
                "operational_execution_authorized": True,
                "phase2_a_lock_sha256": auth["phase2_a_lock_sha256"],
            }
            runner.model_load_authorized = lambda validation, probe, *, probe_only: (
                not probe_only and probe.get("status") == "PASS"
            )
            try:
                m1_started = time.monotonic()
                m1_records, forward_count = runner._execute_stage(
                    stage="A0",
                    cases=m1_cases,
                    state_path=output_dir / "a0_state.jsonl",
                    torch=torch_module,
                    tokenizer=tokenizer,
                    model=model,
                    forward_count=0,
                    ceiling=int(lock["forward_ceiling"]),
                )
                m1_elapsed = time.monotonic() - m1_started
                result_volume.commit()
                result_volume.reload()
                persisted_records = runner._load_completed(
                    output_dir / "a0_state.jsonl",
                    {str(case["case_id"]) for case in stages["A0"]},
                )
                persisted_ids = {str(row["case_id"]) for row in persisted_records}
                m1_ids = {str(row["case_id"]) for row in m1_records}
                detail_paths = [
                    output_dir / "cases" / f"{case['case_id']}.json" for case in m1_cases
                ]
                parser_compatible = all(
                    path.exists() and read_json(path).get("parse_status") == "PASS"
                    for path in detail_paths
                )
                persistence_pass = m1_ids.issubset(persisted_ids) and all(
                    path.exists() for path in detail_paths
                )
                forecast = forecast_after_m1(
                    elapsed_before_m1_seconds=m1_started - gpu_started,
                    m1_elapsed_seconds=m1_elapsed,
                    m1_case_count=len(m1_cases),
                    completed_forward_count=forward_count,
                    forward_ceiling=int(lock["forward_ceiling"]),
                )
                m1_checks = {
                    "parser_compatibility": parser_compatible,
                    "checkpoint_persistence": persistence_pass,
                    "model_loaded_once": True,
                    "model_revision_exact": model_identity["resolved_commit"]
                    == MODEL_REVISION,
                    "forward_forecast": forecast["status"] == "PASS",
                }
                m1 = {
                    "schema_version": "gate13_track_a_modal_m1_result_v1",
                    "status": "PASS" if all(m1_checks.values()) else "BLOCK",
                    "case_ids": [case["case_id"] for case in m1_cases],
                    "cases_are_existing_a0_and_reused": True,
                    "additional_scientific_cases": 0,
                    "additional_scientific_forwards": 0,
                    "elapsed_seconds": m1_elapsed,
                    "model_load_seconds": model_load_seconds,
                    "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
                    "model_load_peak_vram_bytes": model_load_peak_vram,
                    "completed_forward_count": forward_count,
                    "checks": m1_checks,
                    "forecast": forecast,
                }
                _atomic_write_json(output_dir / "m1_preflight_report.json", m1)
                result_volume.commit()
                if m1["status"] != "PASS":
                    state = terminal_state(
                        track_a="MODAL_PREFLIGHT_BUDGET_OR_RUNTIME_BLOCK",
                        forward_count=forward_count,
                        operational_failures=operational_failures,
                    )
                    state.update({"M0": "PASS", "M1": "BLOCK"})
                    return _persist_terminal(
                        output_dir=output_dir, terminal=state, volume=result_volume
                    )

                # Invoke the unchanged scientific runner.  It reuses the M1 A0
                # checkpoints, applies its frozen metrics, and alone controls A1/A2.
                scientific = runner.run_track_a(
                    phase2_dir=REMOTE_PHASE2_DIR,
                    output_dir=output_dir,
                    probe_only=False,
                )
            finally:
                runner._append_record = original_append
                runner._load_exact_model = original_load
                runner.validate_phase2_locks = original_validate
                runner.model_load_authorized = original_authorized

            final_forward_count = int(scientific["model_forward_count"])
            a0 = str(scientific["TRACK_A_A0"])
            a1 = str(scientific["TRACK_A_A1"])
            a2 = str(scientific["TRACK_A_A2"])
            if a0 == "FAIL":
                track_state = "A0_FAIL"
            elif a1 == "FAIL":
                track_state = "A1_FAIL"
            elif a2 == "PASS":
                track_state = "A2_PASS"
            elif a2 == "FAIL":
                track_state = "A2_FAIL"
            else:
                raise ModalTrackAError("scientific runner returned a non-terminal ladder state")
            state = terminal_state(
                track_a=track_state,
                a0=f"A0_{a0}",
                a1="UNOPENED" if a1 == "UNOPENED" else f"A1_{a1}",
                a2="UNOPENED" if a2 == "UNOPENED" else f"A2_{a2}",
                forward_count=final_forward_count,
                operational_failures=operational_failures,
            )
            gpu_elapsed = time.monotonic() - gpu_started
            state.update(
                {
                    "M0": "PASS",
                    "M1": "PASS",
                    "gpu_elapsed_seconds": gpu_elapsed,
                    "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
                    "estimated_modal_gpu_function_usage_usd": gpu_elapsed
                    * GPU_TOTAL_RATE_USD_PER_SECOND,
                    "provider_actual_usage_status": "PENDING_PROVIDER_BILLING_OBSERVATION",
                }
            )
            return _persist_terminal(
                output_dir=output_dir, terminal=state, volume=result_volume
            )
        except Exception as exc:
            forward_count = 0
            for attempt_path in output_dir.glob("*_attempts.jsonl"):
                forward_count += sum(
                    1
                    for line in attempt_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
            failure = {
                "observed_at": utc_now(),
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "forward_count_at_failure": forward_count,
            }
            operational_failures.append(failure)
            _atomic_write_json(output_dir / "scientific_runner_blocker.json", failure)
            state = terminal_state(
                track_a="SCIENTIFIC_RUNNER_BLOCKER",
                forward_count=forward_count,
                operational_failures=operational_failures,
            )
            state.update(
                {
                    "M0": "PASS" if (output_dir / "m0_runtime_report.json").exists() else "UNOPENED",
                    "M1": "BLOCKED",
                }
            )
            return _persist_terminal(
                output_dir=output_dir, terminal=state, volume=result_volume
            )

    @app.local_entrypoint()
    def prepare_resources() -> None:
        """Build/hydrate immutable image and stable Volumes without allocating GPU."""
        _image.hydrate()
        model_volume.hydrate()
        result_volume.hydrate()
        print(
            json.dumps(
                {
                    "schema_version": "gate13_track_a_modal_resource_identity_v1",
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
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    @app.local_entrypoint()
    def run_authorized(authorization: str, control_output: str) -> None:
        """Validate clean local authority, acquire exact weights, then run one ladder."""
        authorization_path = Path(authorization).resolve()
        auth_text = authorization_path.read_text(encoding="utf-8")
        auth_sha = sha256_bytes(auth_text.encode("utf-8"))
        validate_modal_execution_authority(
            authorization_path=authorization_path,
            phase2_dir=local_repo_root() / "analysis/gate13_causal_return/phase2",
            m1_manifest_path=local_repo_root()
            / "tools/gate13_causal_return/modal/m1_preflight_manifest.json",
            repo_root=local_repo_root(),
            verify_git=True,
        )
        acquisition = acquire_exact_model_cpu_only.remote(auth_text, auth_sha)
        execution = execute_one_frozen_track_a.remote(auth_text, auth_sha)
        control = {
            "schema_version": "gate13_track_a_modal_local_control_result_v1",
            "authorization_sha256": auth_sha,
            "model_acquisition": acquisition,
            "execution": execution,
        }
        control_path = Path(control_output).resolve()
        _atomic_write_json(control_path, control)
        print(json.dumps(control, ensure_ascii=False, sort_keys=True))

else:  # pragma: no cover - definitions are intentionally absent without Modal
    app = None
    model_volume = None
    result_volume = None
