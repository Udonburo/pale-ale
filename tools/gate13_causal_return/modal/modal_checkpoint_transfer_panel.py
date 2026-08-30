"""Thin Modal adapter for the frozen Gate13 checkpoint-transfer panel.

The adapter owns acquisition, runtime/score-slot preflight, persistence, and
invocation of the unchanged stepwise runner.  It does not define prompts,
cases, metrics, thresholds, controls, or progression logic.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

try:
    import modal
except ModuleNotFoundError:  # pragma: no cover - local tests do not require Modal
    modal = None  # type: ignore[assignment]
else:
    if not all(hasattr(modal, name) for name in ("App", "Image", "Volume")):
        modal = None  # type: ignore[assignment]

from tools.gate13_causal_return.checkpoint_panel.panel import (
    BASE_IMAGE,
    CHECKPOINTS,
    MAX_FRESH_OPERATOR_FORWARD_COUNT,
    MAX_TRACK_A_FORWARD_COUNT_PER_CHECKPOINT,
    PANEL_SPEND_CEILING_USD,
    SELECTED_INSTRUMENT,
    derive_operator_layers,
    image_definition_sha256,
    model_volume_name,
    result_volume_name,
    sha256_file,
)
from tools.gate13_causal_return.modal.modal_track_a import (
    PYTORCH_INDEX_URL,
    RUNTIME_REQUIREMENTS,
)
from tools.gate13_causal_return.stepwise.compiler import (
    codebook_lookup,
    compile_track_b_collection_ledger,
    render_step_prompt,
    sha256_json,
    transition,
)
from tools.gate13_causal_return.stepwise.operator_qualification import qualify_track_b
from tools.gate13_causal_return.stepwise.runner import (
    JsonlJournal,
    run_track_a_qualification,
    write_result,
)


APP_NAME = "gate13-checkpoint-transfer-panel"
REMOTE_ROOT = PurePosixPath("/opt/gate13")
REMOTE_LOCK_DIR = REMOTE_ROOT / "analysis/gate13_causal_return/checkpoint_panel"
AUTHORIZATION_FILE = "panel_execution_authorization.json"
LOCAL_RUNTIME_WHEEL = (
    "workstream/local/gate13_causal_return_outputs/checkpoint_panel/runtime/"
    "transformers-5.15.1-py3-none-any.whl"
)
REMOTE_RUNTIME_WHEEL = "/opt/wheels/transformers-5.15.1-py3-none-any.whl"

MODEL_ROOT = PurePosixPath("/models")
RESULT_ROOT = PurePosixPath("/results")
MODEL_REPORT_NAME = "model_acquisition.json"

CPU_RATE_USD_PER_CORE_SECOND = 0.0000131
MEMORY_RATE_USD_PER_GIB_SECOND = 0.00000222
GPU_RATE_USD_PER_SECOND = {
    "L40S": 0.000542,
    "A100-80GB": 0.000694,
    "H200": 0.001261,
}
ACQUISITION_CPU_CORES = 8.0
ACQUISITION_MEMORY_MIB = 32_768
GPU_CPU_CORES = 8.0
GPU_MEMORY_MIB = 65_536
ACQUISITION_TIMEOUT_SECONDS = 10_800
PREFLIGHT_TIMEOUT_SECONDS = 7_200
QUALIFICATION_TIMEOUT_SECONDS = 14_400
OPERATOR_TIMEOUT_SECONDS = 14_400


class CheckpointPanelModalError(RuntimeError):
    """Fail-closed operational error for the frozen panel."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_root() -> Path:
    remote = Path(REMOTE_ROOT.as_posix())
    if (remote / "tools/gate13_causal_return").is_dir():
        return remote
    return Path(__file__).resolve().parents[3]


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(value), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tensor_sha256(tensor: Any) -> str:
    value = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _file_manifest(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
        if path.name == "artifact_manifest.json":
            continue
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": "gate13_checkpoint_panel_artifact_manifest_v1",
        "file_count": len(rows),
        "files": rows,
        "manifest_payload_sha256": sha256_json(rows),
    }


def _runtime_rate(gpu: str) -> float:
    return (
        GPU_RATE_USD_PER_SECOND[gpu]
        + GPU_CPU_CORES * CPU_RATE_USD_PER_CORE_SECOND
        + (GPU_MEMORY_MIB / 1024) * MEMORY_RATE_USD_PER_GIB_SECOND
    )


def _acquisition_rate() -> float:
    return (
        ACQUISITION_CPU_CORES * CPU_RATE_USD_PER_CORE_SECOND
        + (ACQUISITION_MEMORY_MIB / 1024) * MEMORY_RATE_USD_PER_GIB_SECOND
    )


def _authorization(authorization_text: str, authorization_sha256: str) -> dict[str, Any]:
    if _sha256_text(authorization_text) != authorization_sha256:
        raise CheckpointPanelModalError("authorization byte SHA-256 mismatch")
    value = json.loads(authorization_text)
    if not value.get("execution_authorized"):
        raise CheckpointPanelModalError("panel execution is not authorized")
    for filename, expected in value["tracked_bindings"].items():
        actual = sha256_file(Path(REMOTE_LOCK_DIR.as_posix()) / filename)
        if actual != expected:
            raise CheckpointPanelModalError(f"authority/hash mismatch: {filename}")
    return value


def _panel_lock() -> dict[str, Any]:
    return _load_json(Path(REMOTE_LOCK_DIR.as_posix()) / "checkpoint_transfer_panel_lock.json")


def _tokenizer_report() -> dict[str, Any]:
    return _load_json(Path(REMOTE_LOCK_DIR.as_posix()) / "panel_tokenizer_codebook_report.json")


def _runtime_manifest() -> dict[str, Any]:
    return _load_json(Path(REMOTE_LOCK_DIR.as_posix()) / "panel_g_common_runtime.json")


def _operator_reservations() -> dict[str, Any]:
    return _load_json(Path(REMOTE_LOCK_DIR.as_posix()) / "fresh_square_operator_reservations.json")


def _registry_row(checkpoint_key: str) -> dict[str, Any]:
    for row in _panel_lock()["official_model_registry"]:
        if row["checkpoint_key"] == checkpoint_key:
            return dict(row)
    raise CheckpointPanelModalError(f"checkpoint is absent from panel lock: {checkpoint_key}")


def _model_mount(checkpoint_key: str) -> Path:
    return Path((MODEL_ROOT / checkpoint_key).as_posix())


def _result_mount(checkpoint_key: str) -> Path:
    return Path((RESULT_ROOT / checkpoint_key).as_posix())


def _execution_root(checkpoint_key: str, authorization: Mapping[str, Any]) -> Path:
    identity = str(authorization["checkpoint_executions"][checkpoint_key]["execution_identity"])
    return _result_mount(checkpoint_key) / "executions" / identity


def _claim(
    checkpoint_key: str,
    authorization: Mapping[str, Any],
    authorization_sha256: str,
    model_volume: Any,
    result_volume: Any,
) -> dict[str, Any]:
    root = _execution_root(checkpoint_key, authorization)
    path = root / "execution_claim.json"
    entry = authorization["checkpoint_executions"][checkpoint_key]
    expected = {
        "checkpoint_key": checkpoint_key,
        "execution_identity": entry["execution_identity"],
        "authorization_sha256": authorization_sha256,
        "repo_id": entry["repo_id"],
        "revision": entry["revision"],
        "model_volume_name": entry["model_volume_name"],
        "model_volume_object_id": model_volume.object_id,
        "result_volume_name": entry["result_volume_name"],
        "result_volume_object_id": result_volume.object_id,
    }
    if path.exists():
        value = _load_json(path)
        if any(value.get(key) != item for key, item in expected.items()):
            raise CheckpointPanelModalError("execution identity was previously bound differently")
    else:
        value = {
            "schema_version": "gate13_checkpoint_panel_execution_claim_v1",
            **expected,
            "claimed_at": utc_now(),
            "stages": [],
        }
    _atomic_json(path, value)
    result_volume.commit()
    return value


def validate_mounted_volume_binding(
    checkpoint_key: str,
    authorization: Mapping[str, Any],
    acquisition_report: Mapping[str, Any],
    model_volume_object_id: str,
    result_volume_object_id: str,
) -> dict[str, str]:
    """Validate mounted Volumes without relying on Modal's nullable name field."""
    entry = authorization["checkpoint_executions"][checkpoint_key]
    expected_model_name = model_volume_name(checkpoint_key)
    expected_result_name = result_volume_name(checkpoint_key)
    if entry["model_volume_name"] != expected_model_name:
        raise CheckpointPanelModalError("authorized model Volume name mismatch")
    if entry["result_volume_name"] != expected_result_name:
        raise CheckpointPanelModalError("authorized result Volume name mismatch")
    if acquisition_report.get("model_volume_name") != expected_model_name:
        raise CheckpointPanelModalError("acquired model Volume name mismatch")
    if acquisition_report.get("model_volume_object_id") != model_volume_object_id:
        raise CheckpointPanelModalError("mounted model Volume object mismatch")
    if not str(model_volume_object_id).startswith("vo-"):
        raise CheckpointPanelModalError("invalid model Volume object identity")
    if not str(result_volume_object_id).startswith("vo-"):
        raise CheckpointPanelModalError("invalid result Volume object identity")
    if model_volume_object_id == result_volume_object_id:
        raise CheckpointPanelModalError("model and result Volumes are not distinct")
    return {
        "model_volume_name": expected_model_name,
        "model_volume_object_id": model_volume_object_id,
        "result_volume_name": expected_result_name,
        "result_volume_object_id": result_volume_object_id,
    }


def _mark_stage(
    checkpoint_key: str,
    authorization: Mapping[str, Any],
    stage: str,
    result_volume: Any,
) -> None:
    path = _execution_root(checkpoint_key, authorization) / "execution_claim.json"
    value = _load_json(path)
    if stage not in value["stages"]:
        value["stages"].append(stage)
    _atomic_json(path, value)
    result_volume.commit()


def _verify_snapshot(checkpoint_key: str, *, full_file_check: bool = True) -> dict[str, Any]:
    mount = _model_mount(checkpoint_key)
    report_path = mount / MODEL_REPORT_NAME
    if not report_path.exists():
        raise CheckpointPanelModalError("model acquisition report is unavailable")
    report = _load_json(report_path)
    expected = _registry_row(checkpoint_key)
    spec = CHECKPOINTS[checkpoint_key]
    if report.get("repo_id") != spec["repo_id"] or report.get("resolved_revision") != spec["revision"]:
        raise CheckpointPanelModalError("model acquisition repository/revision mismatch")
    if report.get("model_directory_identity_sha256") != expected["model_directory_identity_sha256"]:
        raise CheckpointPanelModalError("model directory identity mismatch")
    snapshot = Path(str(report["snapshot_directory"]))
    if snapshot.name != spec["revision"] or not snapshot.is_dir():
        raise CheckpointPanelModalError("immutable snapshot directory is unavailable")
    if full_file_check:
        for row in expected["file_inventory"]:
            path = snapshot / str(row["path"])
            if not path.is_file() or path.stat().st_size != int(row["bytes"]):
                raise CheckpointPanelModalError(f"model inventory mismatch: {row['path']}")
        for name, expected_sha in expected["asset_sha256"].items():
            if expected_sha is None:
                continue
            path = snapshot / name
            if not path.is_file() or sha256_file(path) != expected_sha:
                raise CheckpointPanelModalError(f"model asset SHA mismatch: {name}")
    return {**report, "verification_status": "PASS"}


def _acquire(
    checkpoint_key: str,
    authorization: Mapping[str, Any],
    model_volume: Any,
) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    mount = _model_mount(checkpoint_key)
    report_path = mount / MODEL_REPORT_NAME
    if report_path.exists():
        return {**_verify_snapshot(checkpoint_key), "idempotent_retrieval": True}
    started = time.monotonic()
    expected = _registry_row(checkpoint_key)
    spec = CHECKPOINTS[checkpoint_key]
    cache = mount / "hf_home" / "hub"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        {
            "HF_HOME": str(mount / "hf_home"),
            "HF_HUB_CACHE": str(cache),
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        }
    )
    snapshot = Path(
        snapshot_download(
            repo_id=str(spec["repo_id"]),
            revision=str(spec["revision"]),
            cache_dir=str(cache),
            max_workers=8,
        )
    )
    if snapshot.name != spec["revision"]:
        raise CheckpointPanelModalError("snapshot_download did not resolve the frozen commit")
    actual_files = []
    for row in expected["file_inventory"]:
        path = snapshot / str(row["path"])
        if not path.is_file() or path.stat().st_size != int(row["bytes"]):
            raise CheckpointPanelModalError(f"acquired inventory mismatch: {row['path']}")
        actual_files.append(
            {
                "path": row["path"],
                "bytes": path.stat().st_size,
                "content_identity": row.get("lfs_sha256") or row.get("git_blob_id"),
            }
        )
    if sha256_json(actual_files) != expected["model_directory_identity_sha256"]:
        raise CheckpointPanelModalError("acquired model directory identity mismatch")
    actual_assets = {}
    for name, expected_sha in expected["asset_sha256"].items():
        if expected_sha is None:
            continue
        actual = sha256_file(snapshot / name)
        if actual != expected_sha:
            raise CheckpointPanelModalError(f"acquired asset SHA mismatch: {name}")
        actual_assets[name] = actual
    elapsed = time.monotonic() - started
    report = {
        "schema_version": "gate13_checkpoint_panel_model_acquisition_v1",
        "checkpoint_key": checkpoint_key,
        "repo_id": spec["repo_id"],
        "resolved_revision": spec["revision"],
        "snapshot_directory": str(snapshot),
        "model_volume_name": model_volume_name(checkpoint_key),
        "model_volume_object_id": model_volume.object_id,
        "file_count": len(actual_files),
        "total_file_bytes": sum(int(row["bytes"]) for row in actual_files),
        "complete_file_inventory": actual_files,
        "asset_sha256": actual_assets,
        "weight_shard_identities": [
            {
                "path": row["path"],
                "bytes": row["bytes"],
                "lfs_sha256": row["lfs_sha256"],
            }
            for row in expected["file_inventory"]
            if str(row["path"]).endswith(".safetensors")
        ],
        "model_directory_identity_sha256": expected["model_directory_identity_sha256"],
        "cpu_acquisition_elapsed_seconds": elapsed,
        "estimated_cpu_acquisition_cost_usd": elapsed * _acquisition_rate(),
        "gpu_allocated": False,
        "scientific_model_forward_count": 0,
        "volume_immutable_after_acquisition": True,
        "acquired_at": utc_now(),
    }
    _atomic_json(report_path, report)
    model_volume.commit()
    return report


def _gpu_identity() -> dict[str, Any]:
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) != 1:
        raise CheckpointPanelModalError("exactly one GPU is required")
    name, driver, memory = [part.strip() for part in lines[0].split(",")]
    return {"name": name, "driver": driver, "memory_total_mib": int(memory)}


def _m0(checkpoint_key: str, requested_gpu: str) -> dict[str, Any]:
    import accelerate
    import safetensors
    import tokenizers
    import torch
    import transformers

    runtime = _runtime_manifest()["Panel_S" if checkpoint_key == "qwen3_14b" else "Panel_G"]
    gpu = _gpu_identity()
    mismatches = []
    expected_name = {
        "L40S": "NVIDIA L40S",
        "A100-80GB": "A100",
        "H200": "H200",
    }[requested_gpu]
    if expected_name not in gpu["name"]:
        mismatches.append(f"gpu:{gpu['name']}")
    if requested_gpu == "A100-80GB" and gpu["memory_total_mib"] < 80_000:
        mismatches.append(f"gpu_memory:{gpu['memory_total_mib']}")
    expected = {
        "driver": "580.95.05",
        "python": "3.11.2",
        "torch": "2.7.1+cu126",
        "cuda": "12.6",
        "transformers": "5.15.0" if checkpoint_key == "qwen3_14b" else "5.15.1",
        "tokenizers": "0.22.2",
        "safetensors": "0.8.0",
        "accelerate": "1.14.0",
    }
    observed = {
        "driver": gpu["driver"],
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "transformers": str(transformers.__version__),
        "tokenizers": str(tokenizers.__version__),
        "safetensors": str(safetensors.__version__),
        "accelerate": str(accelerate.__version__),
    }
    if checkpoint_key != "qwen3_14b":
        import PIL
        import torchvision

        expected.update({"torchvision": "0.22.1+cu126", "Pillow": "12.1.1"})
        observed.update({"torchvision": str(torchvision.__version__), "Pillow": str(PIL.__version__)})
    for field, value in expected.items():
        if observed.get(field) != value:
            mismatches.append(f"{field}:{observed.get(field)}!=${value}".replace("$", ""))
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        mismatches.append("cuda_bfloat16_unavailable")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        mismatches.append("offline_environment")
    return {
        "schema_version": "gate13_checkpoint_panel_m0_v1",
        "checkpoint_key": checkpoint_key,
        "status": "PASS" if not mismatches else "RUNTIME_MISMATCH",
        "requested_gpu": requested_gpu,
        "gpu": gpu,
        "expected": expected,
        "observed": observed,
        "mismatches": mismatches,
        "dtype": "bfloat16",
        "quantization": False,
        "batch": 1,
        "generation": False,
        "sampling": False,
        "maximum_active_gpu_containers": 1,
        "automatic_retries": 0,
        "runtime_lock": runtime,
        "scientific_model_forward_count": 0,
    }


def _load_model(checkpoint_key: str) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    import torch

    report = _verify_snapshot(checkpoint_key)
    snapshot = str(report["snapshot_directory"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    if checkpoint_key == "qwen3_14b":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        endpoint = AutoTokenizer.from_pretrained(
            snapshot,
            use_fast=True,
            local_files_only=True,
            trust_remote_code=False,
        )
        tokenizer = endpoint
        model = AutoModelForCausalLM.from_pretrained(
            snapshot,
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            local_files_only=True,
            trust_remote_code=False,
            attn_implementation="sdpa",
        )
    else:
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        endpoint = AutoProcessor.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
        )
        if endpoint.__class__.__name__ != "Qwen3VLProcessor":
            raise CheckpointPanelModalError("official Qwen3VLProcessor did not resolve")
        tokenizer = endpoint.tokenizer
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            snapshot,
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            local_files_only=True,
            trust_remote_code=False,
            attn_implementation="sdpa",
        )
    model.eval()
    if model.__class__.__name__ != CHECKPOINTS[checkpoint_key]["model_class"]:
        raise CheckpointPanelModalError("frozen model class did not resolve")
    if endpoint.__class__.__name__ != CHECKPOINTS[checkpoint_key]["processor_class"]:
        raise CheckpointPanelModalError("frozen tokenizer/processor class did not resolve")
    if getattr(model, "is_quantized", False) or getattr(model.config, "quantization_config", None):
        raise CheckpointPanelModalError("quantized model realization is forbidden")
    if model.dtype != torch.bfloat16:
        raise CheckpointPanelModalError("model realization is not BF16")
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    memory = {
        "model_load_elapsed_seconds": elapsed,
        "allocated_mib": torch.cuda.memory_allocated() / 2**20,
        "reserved_mib": torch.cuda.memory_reserved() / 2**20,
        "model_load_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "model_load_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "free_mib_after_load": torch.cuda.mem_get_info()[0] / 2**20,
    }
    return torch, endpoint, tokenizer, model, memory


def _prepare_inputs(endpoint: Any, checkpoint_key: str, prompt: str) -> tuple[str, dict[str, Any]]:
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": False,
        "preserve_thinking": False,
        "return_tensors": "pt",
        "return_dict": True,
    }
    messages = [{"role": "user", "content": prompt}]
    rendered = endpoint.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
        preserve_thinking=False,
    )
    encoded = endpoint.apply_chat_template(messages, **kwargs)
    values = dict(encoded)
    forbidden = {
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
    }
    if any(key in values and values[key] is not None for key in forbidden):
        raise CheckpointPanelModalError("text-only processor emitted a vision/video tensor")
    inputs = {
        key: value.to("cuda:0")
        for key, value in values.items()
        if hasattr(value, "to") and key not in forbidden
    }
    if "input_ids" not in inputs or "attention_mask" not in inputs:
        raise CheckpointPanelModalError("official template did not yield input_ids and attention_mask")
    if set(inputs) - {"input_ids", "attention_mask", "mm_token_type_ids", "position_ids"}:
        raise CheckpointPanelModalError(f"unexpected text-only processor fields: {sorted(inputs)}")
    return str(rendered), inputs


def _score_slot_validation(
    checkpoint_key: str,
    endpoint: Any,
    tokenizer: Any,
    prompt: str,
    candidates: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    rendered, inputs = _prepare_inputs(endpoint, checkpoint_key, prompt)
    expected = _tokenizer_report()["checkpoints"][checkpoint_key]["score_slot"]
    candidate_ids = []
    for label in candidates:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1 or int(ids[0]) in tokenizer.all_special_ids:
            raise CheckpointPanelModalError("candidate ceased to be one non-special token")
        appended = tokenizer.encode(rendered + label, add_special_tokens=False)
        if [int(value) for value in appended] != inputs["input_ids"][0].tolist() + [int(ids[0])]:
            raise CheckpointPanelModalError("candidate is not direct at the semantic score slot")
        candidate_ids.append(int(ids[0]))
    observed = {
        "canonical_message_content_sha256": _sha256_text(prompt),
        "rendered_prompt_sha256": _sha256_text(rendered),
        "assistant_generation_prompt_suffix": rendered[-80:],
        "input_ids_sha256": _tensor_sha256(inputs["input_ids"]),
        "attention_mask_sha256": _tensor_sha256(inputs["attention_mask"]),
        "input_length": int(inputs["input_ids"].shape[-1]),
        "score_tensor_index": int(inputs["input_ids"].shape[-1]) - 1,
        "semantic_answer_token_index": int(inputs["input_ids"].shape[-1]),
        "candidate_labels": [
            {"label": label, "token_id": token_id}
            for label, token_id in zip(candidates, candidate_ids)
        ],
        "active_think_prefix_before_score_slot": rendered.rfind("<think>") > rendered.rfind("</think>"),
        "empty_closed_think_protocol_before_slot": rendered.endswith("<think>\n\n</think>\n\n"),
        "vision_or_video_placeholder": any(
            marker in rendered for marker in ("<|image_pad|>", "<|video_pad|>", "<|vision_start|>")
        ),
        "tool_call_prefix": "<tool_call>" in rendered,
    }
    for field in (
        "canonical_message_content_sha256",
        "rendered_prompt_sha256",
        "assistant_generation_prompt_suffix",
        "input_ids_sha256",
        "attention_mask_sha256",
        "input_length",
        "score_tensor_index",
        "semantic_answer_token_index",
        "candidate_labels",
        "active_think_prefix_before_score_slot",
        "empty_closed_think_protocol_before_slot",
        "vision_or_video_placeholder",
        "tool_call_prefix",
    ):
        if observed[field] != expected[field]:
            raise CheckpointPanelModalError(f"SCORE_SLOT_INCOMPATIBLE:{field}")
    return observed, inputs, candidate_ids


class PanelProbe:
    def __init__(self, torch_module: Any, endpoint: Any, tokenizer: Any, model: Any):
        self.torch = torch_module
        self.endpoint = endpoint
        self.tokenizer = tokenizer
        self.model = model

    def _query(self, prompt: str, candidates: Sequence[str], *, hidden: bool) -> tuple[dict[str, Any], Any]:
        _rendered, inputs = _prepare_inputs(self.endpoint, "runtime", prompt)
        token_ids = []
        for label in candidates:
            ids = self.tokenizer.encode(label, add_special_tokens=False)
            if len(ids) != 1 or int(ids[0]) in self.tokenizer.all_special_ids:
                raise CheckpointPanelModalError("forced-choice label ceased to be one non-special token")
            token_ids.append(int(ids[0]))
        self.torch.cuda.synchronize()
        started = time.monotonic()
        with self.torch.inference_mode():
            output = self.model(
                **inputs,
                use_cache=False,
                output_hidden_states=hidden,
                return_dict=True,
            )
        self.torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        logits = output.logits[0, -1, token_ids].float().detach().cpu().tolist()
        if logits[0] == logits[1]:
            raise CheckpointPanelModalError("exact tie in forced-choice logits")
        predicted = 0 if logits[0] > logits[1] else 1
        return (
            {
                "predicted_label": candidates[predicted],
                "candidate_token_ids": token_ids,
                "candidate_logits": [float(value) for value in logits],
                "logit_margin_predicted_minus_alternative": float(abs(logits[0] - logits[1])),
                "input_ids_sha256": _tensor_sha256(inputs["input_ids"]),
                "attention_mask_sha256": _tensor_sha256(inputs["attention_mask"]),
                "input_token_count": int(inputs["input_ids"].shape[-1]),
                "score_tensor_index": int(inputs["input_ids"].shape[-1]) - 1,
                "forward_elapsed_seconds": elapsed,
                "readout": "RAW_NEXT_TOKEN_FORCED_CHOICE_LOGITS",
                "generate_called": False,
                "sampling": False,
            },
            output,
        )

    def __call__(self, prompt: str, candidates: tuple[str, str], metadata: Mapping[str, Any]) -> Mapping[str, Any]:
        del metadata
        response, _output = self._query(prompt, candidates, hidden=False)
        return response

    def activation_call(
        self,
        prompt: str,
        candidates: tuple[str, str],
        layer_set: Sequence[int],
    ) -> tuple[dict[str, Any], dict[int, Any]]:
        response, output = self._query(prompt, candidates, hidden=True)
        if output.hidden_states is None:
            raise CheckpointPanelModalError("frozen hidden-state tap is unavailable")
        maximum = max(int(layer) for layer in layer_set)
        if len(output.hidden_states) <= maximum:
            raise CheckpointPanelModalError("frozen hidden-state index exceeds model depth")
        activations = {
            int(layer): output.hidden_states[int(layer)][0, -1].float().detach().cpu().numpy()
            for layer in layer_set
        }
        response["readout"] = "RAW_NEXT_TOKEN_FORCED_CHOICE_LOGITS_WITH_FROZEN_HIDDEN_STATE_TAP"
        return response, activations


def _preflight_core(
    checkpoint_key: str,
    requested_gpu: str,
    authorization_text: str,
    authorization_sha256: str,
    model_volume: Any,
    result_volume: Any,
) -> dict[str, Any]:
    authorization = _authorization(authorization_text, authorization_sha256)
    acquisition_report = _load_json(_model_mount(checkpoint_key) / MODEL_REPORT_NAME)
    validate_mounted_volume_binding(
        checkpoint_key,
        authorization,
        acquisition_report,
        model_volume.object_id,
        result_volume.object_id,
    )
    _claim(
        checkpoint_key,
        authorization,
        authorization_sha256,
        model_volume,
        result_volume,
    )
    root = _execution_root(checkpoint_key, authorization)
    report_path = root / f"m1_preflight_{requested_gpu}.json"
    if report_path.exists():
        return {**_load_json(report_path), "idempotent_retrieval": True}
    started = time.monotonic()
    model_forward_count = 0
    try:
        if checkpoint_key == "qwen3_14b":
            expected_image_sha = image_definition_sha256("Panel S")
        else:
            wheel_sha = _runtime_manifest()["Panel_G"]["transformers_wheel_sha256"]
            if sha256_file(Path(REMOTE_RUNTIME_WHEEL)) != wheel_sha:
                raise CheckpointPanelModalError("Panel G Transformers wheel SHA mismatch")
            expected_image_sha = image_definition_sha256("Panel G", wheel_sha)
        if authorization["image_definitions"][
            "Panel_S" if checkpoint_key == "qwen3_14b" else "Panel_G"
        ]["sha256"] != expected_image_sha:
            raise CheckpointPanelModalError("authorized image definition mismatch")
        m0 = _m0(checkpoint_key, requested_gpu)
        _atomic_json(root / f"m0_runtime_{requested_gpu}.json", m0)
        result_volume.commit()
        if m0["status"] != "PASS":
            raise CheckpointPanelModalError("RUNTIME_MISMATCH")
        torch_module, endpoint, tokenizer, model, memory = _load_model(checkpoint_key)
        runtime_key = "Panel_S" if checkpoint_key == "qwen3_14b" else "Panel_G"
        minimum_margin = int(_runtime_manifest()[runtime_key]["minimum_free_vram_margin_mib"])
        if memory["free_mib_after_load"] < minimum_margin:
            raise CheckpointPanelModalError("PREFLIGHT_FROZEN_VRAM_MARGIN_FAILURE")
        from tools.gate13_causal_return.checkpoint_panel.panel import _first_transfer_prompt

        prompt, candidates = _first_transfer_prompt()
        slot, inputs, candidate_ids = _score_slot_validation(
            checkpoint_key, endpoint, tokenizer, prompt, candidates
        )
        torch_module.cuda.reset_peak_memory_stats()
        torch_module.cuda.synchronize()
        forward_started = time.monotonic()
        with torch_module.inference_mode():
            output = model(
                **inputs,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
        torch_module.cuda.synchronize()
        forward_elapsed = time.monotonic() - forward_started
        model_forward_count = 1
        scores = output.logits[0, -1, candidate_ids].float().detach().cpu().tolist()
        if len(scores) != 2 or not all(float(value) == float(value) for value in scores):
            raise CheckpointPanelModalError("M1 forced-choice score tensor is invalid")
        total_elapsed = time.monotonic() - started
        rate = _runtime_rate(requested_gpu)
        report = {
            "schema_version": "gate13_checkpoint_panel_m1_preflight_v1",
            "status": "PASS",
            "checkpoint_key": checkpoint_key,
            "execution_identity": authorization["checkpoint_executions"][checkpoint_key]["execution_identity"],
            "requested_gpu": requested_gpu,
            "model_volume_object_id": model_volume.object_id,
            "result_volume_object_id": result_volume.object_id,
            "model_identity": _verify_snapshot(checkpoint_key, full_file_check=False),
            "score_slot": slot,
            "model_class": model.__class__.__name__,
            "processor_class": endpoint.__class__.__name__,
            "model_load": memory,
            "minimum_free_vram_margin_mib": minimum_margin,
            "steady_state_peak_allocated_mib": torch_module.cuda.max_memory_allocated() / 2**20,
            "steady_state_peak_reserved_mib": torch_module.cuda.max_memory_reserved() / 2**20,
            "forced_choice_forward_elapsed_seconds": forward_elapsed,
            "m1_candidate_logits": [float(value) for value in scores],
            "m1_selection_or_threshold_use": False,
            "m1_qualification_ledger_member": False,
            "model_forward_count": model_forward_count,
            "scientific_qualification_forward_count": 0,
            "gpu_elapsed_seconds": total_elapsed,
            "estimated_modal_usage_usd": total_elapsed * rate,
            "maximum_track_a_projection_seconds": memory["model_load_elapsed_seconds"]
            + forward_elapsed * MAX_TRACK_A_FORWARD_COUNT_PER_CHECKPOINT,
            "maximum_track_a_projection_usd": (
                memory["model_load_elapsed_seconds"]
                + forward_elapsed * MAX_TRACK_A_FORWARD_COUNT_PER_CHECKPOINT
            )
            * rate,
            "completed_at": utc_now(),
        }
        _atomic_json(report_path, report)
        _mark_stage(checkpoint_key, authorization, f"M1_PASS_{requested_gpu}", result_volume)
        _atomic_json(root / "artifact_manifest.json", _file_manifest(root))
        result_volume.commit()
        return report
    except Exception as exc:
        text = str(exc)
        if "SCORE_SLOT_INCOMPATIBLE" in text:
            status = "SCORE_SLOT_INCOMPATIBLE"
        elif "CUDA out of memory" in text or "VRAM_MARGIN" in text:
            status = "PREFLIGHT_OOM_OR_FROZEN_MARGIN_BLOCK"
        elif "RUNTIME_MISMATCH" in text:
            status = "RUNTIME_MISMATCH"
        else:
            status = "M1_PREFLIGHT_IMPLEMENTATION_OR_INFRASTRUCTURE_BLOCK"
        blocker = {
            "schema_version": "gate13_checkpoint_panel_m1_preflight_v1",
            "status": status,
            "checkpoint_key": checkpoint_key,
            "requested_gpu": requested_gpu,
            "error_type": type(exc).__name__,
            "error": text,
            "traceback": traceback.format_exc(),
            "model_forward_count": model_forward_count,
            "scientific_qualification_forward_count": 0,
            "gpu_elapsed_seconds": time.monotonic() - started,
            "estimated_modal_usage_usd": (time.monotonic() - started) * _runtime_rate(requested_gpu),
            "completed_at": utc_now(),
        }
        _atomic_json(report_path, blocker)
        _mark_stage(checkpoint_key, authorization, f"M1_{status}_{requested_gpu}", result_volume)
        _atomic_json(root / "artifact_manifest.json", _file_manifest(root))
        result_volume.commit()
        return blocker


def _qualification_core(
    checkpoint_key: str,
    requested_gpu: str,
    authorization_text: str,
    authorization_sha256: str,
    budget_gate_text: str,
    model_volume: Any,
    result_volume: Any,
) -> dict[str, Any]:
    authorization = _authorization(authorization_text, authorization_sha256)
    acquisition_report = _load_json(_model_mount(checkpoint_key) / MODEL_REPORT_NAME)
    validate_mounted_volume_binding(
        checkpoint_key,
        authorization,
        acquisition_report,
        model_volume.object_id,
        result_volume.object_id,
    )
    _claim(
        checkpoint_key,
        authorization,
        authorization_sha256,
        model_volume,
        result_volume,
    )
    root = _execution_root(checkpoint_key, authorization)
    terminal_path = root / "track_a_terminal_state.json"
    if terminal_path.exists():
        return {**_load_json(terminal_path), "idempotent_retrieval": True}
    budget_gate = json.loads(budget_gate_text)
    if not budget_gate.get("panel_execution_permitted"):
        raise CheckpointPanelModalError("panel budget gate is not open")
    if float(budget_gate["projected_with_25pct_contingency_usd"]) > PANEL_SPEND_CEILING_USD:
        raise CheckpointPanelModalError("panel budget forecast exceeds authorization")
    preflight = root / f"m1_preflight_{requested_gpu}.json"
    if not preflight.exists() or _load_json(preflight).get("status") != "PASS":
        raise CheckpointPanelModalError("matching M1 preflight is not PASS")
    started = time.monotonic()
    journal: JsonlJournal | None = None
    try:
        m0 = _m0(checkpoint_key, requested_gpu)
        _atomic_json(root / f"qualification_m0_runtime_{requested_gpu}.json", m0)
        result_volume.commit()
        if m0["status"] != "PASS":
            raise CheckpointPanelModalError("qualification runtime mismatch")
        torch_module, endpoint, tokenizer, model, memory = _load_model(checkpoint_key)
        probe = PanelProbe(torch_module, endpoint, tokenizer, model)
        journal = JsonlJournal(root / "track_a_ledger", probe, result_volume.commit)
        result = run_track_a_qualification(journal, SELECTED_INSTRUMENT)
        write_result(root / "track_a_qualification_result.json", result)
        if journal.total_attempt_count > MAX_TRACK_A_FORWARD_COUNT_PER_CHECKPOINT:
            raise CheckpointPanelModalError("per-checkpoint Track A forward count exceeded")
        elapsed = time.monotonic() - started
        terminal = {
            "schema_version": "gate13_checkpoint_panel_track_a_terminal_v1",
            "status": "COMPLETE",
            "checkpoint_key": checkpoint_key,
            "execution_identity": authorization["checkpoint_executions"][checkpoint_key]["execution_identity"],
            "gpu": requested_gpu,
            "STREAM-A0": result["STREAM-A0"]["status"],
            "STREAM-A1": result["STREAM-A1"]["status"],
            "STREAM-A2": result["STREAM-A2"]["status"],
            "terminal_track_a_status": result["terminal_track_a_status"],
            "actual_model_forward_count": journal.total_attempt_count,
            "actual_model_response_count": journal.total_response_count,
            "new_forward_count": journal.new_forward_count,
            "gpu_elapsed_seconds": elapsed,
            "estimated_modal_usage_usd": elapsed * _runtime_rate(requested_gpu),
            "model_load": memory,
            "A3": "CLOSED",
            "TRACK_C": "CLOSED",
            "FORMAL_GATE13": "CLOSED",
            "completed_at": utc_now(),
        }
        _atomic_json(terminal_path, terminal)
        _mark_stage(checkpoint_key, authorization, "TRACK_A_TERMINAL", result_volume)
        _atomic_json(root / "artifact_manifest.json", _file_manifest(root))
        result_volume.commit()
        return terminal
    except Exception as exc:
        attempts = 0 if journal is None else journal.total_attempt_count
        responses = 0 if journal is None else journal.total_response_count
        blocker = {
            "schema_version": "gate13_checkpoint_panel_track_a_terminal_v1",
            "status": "SCIENTIFIC_RUNNER_BLOCK" if attempts else "CHECKPOINT_INFRASTRUCTURE_BLOCK",
            "checkpoint_key": checkpoint_key,
            "gpu": requested_gpu,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "actual_model_forward_count": attempts,
            "actual_model_response_count": responses,
            "rerun_authorized": False,
            "gpu_elapsed_seconds": time.monotonic() - started,
            "estimated_modal_usage_usd": (time.monotonic() - started) * _runtime_rate(requested_gpu),
            "A3": "CLOSED",
            "TRACK_C": "CLOSED",
            "completed_at": utc_now(),
        }
        _atomic_json(terminal_path, blocker)
        _mark_stage(checkpoint_key, authorization, blocker["status"], result_volume)
        _atomic_json(root / "artifact_manifest.json", _file_manifest(root))
        result_volume.commit()
        return blocker


def _collect_operator(
    checkpoint_key: str,
    probe: PanelProbe,
    root: Path,
    result_volume: Any,
) -> tuple[dict[str, Any], int]:
    import numpy as np

    ledger = compile_track_b_collection_ledger(SELECTED_INSTRUMENT)
    codebooks = codebook_lookup()
    layers = derive_operator_layers(int(CHECKPOINTS[checkpoint_key]["layers"]))
    attempt_path = root / "activation_attempts.jsonl"
    response_path = root / "activation_responses.jsonl"
    attempt_rows = _load_jsonl(attempt_path)
    response_rows = _load_jsonl(response_path)
    attempts = {row["forward_id"]: row for row in attempt_rows}
    responses = {row["forward_id"]: row for row in response_rows}
    if len(attempts) != len(attempt_rows) or len(responses) != len(response_rows):
        raise CheckpointPanelModalError("duplicate fresh operator forward identity")
    activations: dict[str, dict[int, dict[str, list[Any]]]] = {
        half: {layer: {node: [] for node in ledger["nodes"]} for layer in layers}
        for half in ("half_1", "half_2")
    }
    node_values = {
        "phase0_state0": (0, 0, False),
        "phase0_state1": (0, 1, False),
        "phase1_state0": (1, 0, False),
        "phase1_state1": (1, 1, False),
        "phase1_state1_broken": (1, 1, True),
    }
    new_count = 0
    for half in ledger["halves"]:
        half_id = str(half["half_id"])
        template_offset = 0 if half_id == "half_1" else 4
        for sample_index, sample in enumerate(half["samples"]):
            codebook = codebooks[str(sample["codebook_id"])]
            action = int(sample["episode_seed"]) % 2
            for node in sample["node_ids"]:
                phase, state, broken = node_values[str(node)]
                forward_id = f"{sample['sample_id']}-{node}"
                prompt = render_step_prompt(
                    variant_id=SELECTED_INSTRUMENT,
                    surface="TRACK-B",
                    codebook=codebook,
                    current_state=state,
                    action=action,
                    demonstration_condition="correct",
                    demo_seed=int(sample["episode_seed"]),
                    template_flavor=template_offset + sample_index % 4,
                    phase_index=phase,
                    broken_context=broken,
                )
                binding = {
                    "forward_id": forward_id,
                    "half_id": half_id,
                    "sample_id": sample["sample_id"],
                    "node_id": node,
                    "codebook_id": codebook.codebook_id,
                    "episode_seed": sample["episode_seed"],
                    "template_id": sample["template_id"],
                    "demonstration_instance_id": sample["demonstration_instance_id"],
                    "prompt_sha256": _sha256_text(prompt),
                    "candidate_labels": list(codebook.state_labels),
                    "target_state": transition(state, action),
                }
                if forward_id not in responses:
                    if forward_id in attempts:
                        raise CheckpointPanelModalError(f"AMBIGUOUS_OPERATOR_FORWARD:{forward_id}")
                    _append_jsonl(attempt_path, binding)
                    attempts[forward_id] = binding
                    result_volume.commit()
                    metadata, vectors = probe.activation_call(prompt, codebook.state_labels, layers)
                    artifact = root / "activations" / half_id / f"{forward_id}.npz"
                    artifact.parent.mkdir(parents=True, exist_ok=True)
                    temporary = artifact.with_suffix(".npz.tmp")
                    with temporary.open("wb") as handle:
                        np.savez_compressed(handle, **{f"layer_{layer}": vector for layer, vector in vectors.items()})
                    os.replace(temporary, artifact)
                    response = {
                        **binding,
                        **metadata,
                        "activation_artifact": artifact.relative_to(root).as_posix(),
                        "activation_artifact_sha256": sha256_file(artifact),
                    }
                    _append_jsonl(response_path, response)
                    responses[forward_id] = response
                    new_count += 1
                    result_volume.commit()
                response = responses[forward_id]
                if any(response.get(key) != value for key, value in binding.items()):
                    raise CheckpointPanelModalError(f"operator resume binding mismatch:{forward_id}")
                artifact = root / str(response["activation_artifact"])
                if sha256_file(artifact) != response["activation_artifact_sha256"]:
                    raise CheckpointPanelModalError(f"operator activation SHA mismatch:{forward_id}")
                with np.load(artifact) as stored:
                    for layer in layers:
                        activations[half_id][layer][str(node)].append(
                            stored[f"layer_{layer}"].astype(np.float64)
                        )
    if len(responses) > MAX_FRESH_OPERATOR_FORWARD_COUNT:
        raise CheckpointPanelModalError("fresh operator forward count exceeded")
    arrays = {
        half: {
            layer: {node: np.stack(values, axis=0) for node, values in nodes.items()}
            for layer, nodes in by_layer.items()
        }
        for half, by_layer in activations.items()
    }
    result = qualify_track_b(arrays, layer_set=layers)
    result["collection_ledger_sha256"] = ledger["sha256"]
    result["actual_new_forward_count"] = new_count
    result["total_activation_forward_count"] = len(responses)
    result["checkpoint_key"] = checkpoint_key
    return result, new_count


def _operator_core(
    checkpoint_key: str,
    requested_gpu: str,
    authorization_text: str,
    authorization_sha256: str,
    panel_results_text: str,
    model_volume: Any,
    result_volume: Any,
) -> dict[str, Any]:
    authorization = _authorization(authorization_text, authorization_sha256)
    acquisition_report = _load_json(_model_mount(checkpoint_key) / MODEL_REPORT_NAME)
    validate_mounted_volume_binding(
        checkpoint_key,
        authorization,
        acquisition_report,
        model_volume.object_id,
        result_volume.object_id,
    )
    _claim(
        checkpoint_key,
        authorization,
        authorization_sha256,
        model_volume,
        result_volume,
    )
    panel_results = json.loads(panel_results_text)
    if panel_results.get("selected_operator_checkpoint") != checkpoint_key:
        raise CheckpointPanelModalError("fresh operator selection does not match frozen priority gate")
    root = _execution_root(checkpoint_key, authorization) / "fresh_square_operator"
    terminal_path = root / "terminal_state.json"
    if terminal_path.exists():
        return {**_load_json(terminal_path), "idempotent_retrieval": True}
    started = time.monotonic()
    new_count = 0
    try:
        m0 = _m0(checkpoint_key, requested_gpu)
        _atomic_json(root / "m0_runtime.json", m0)
        result_volume.commit()
        if m0["status"] != "PASS":
            raise CheckpointPanelModalError("operator runtime mismatch")
        torch_module, endpoint, tokenizer, model, memory = _load_model(checkpoint_key)
        probe = PanelProbe(torch_module, endpoint, tokenizer, model)
        result, new_count = _collect_operator(checkpoint_key, probe, root, result_volume)
        write_result(root / "qualification_result.json", result)
        elapsed = time.monotonic() - started
        terminal = {
            "schema_version": "gate13_checkpoint_panel_fresh_square_terminal_v1",
            "status": result["status"],
            "checkpoint_key": checkpoint_key,
            "gpu": requested_gpu,
            "actual_model_forward_count": int(result["total_activation_forward_count"]),
            "new_model_forward_count": new_count,
            "gpu_elapsed_seconds": elapsed,
            "estimated_modal_usage_usd": elapsed * _runtime_rate(requested_gpu),
            "model_load": memory,
            "A3": "CLOSED",
            "TRACK_C": "CLOSED",
            "FORMAL_GATE13": "CLOSED",
            "completed_at": utc_now(),
        }
        _atomic_json(terminal_path, terminal)
        _atomic_json(root / "artifact_manifest.json", _file_manifest(root))
        _mark_stage(checkpoint_key, authorization, "FRESH_SQUARE_OPERATOR_TERMINAL", result_volume)
        result_volume.commit()
        return terminal
    except Exception as exc:
        blocker = {
            "schema_version": "gate13_checkpoint_panel_fresh_square_terminal_v1",
            "status": "FRESH_SQUARE_OPERATOR_BLOCK",
            "checkpoint_key": checkpoint_key,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "new_model_forward_count": new_count,
            "rerun_authorized": False,
            "gpu_elapsed_seconds": time.monotonic() - started,
            "estimated_modal_usage_usd": (time.monotonic() - started) * _runtime_rate(requested_gpu),
            "TRACK_C": "CLOSED",
            "completed_at": utc_now(),
        }
        _atomic_json(terminal_path, blocker)
        _atomic_json(root / "artifact_manifest.json", _file_manifest(root))
        _mark_stage(checkpoint_key, authorization, "FRESH_SQUARE_OPERATOR_BLOCK", result_volume)
        result_volume.commit()
        return blocker


def _local_control(path: str, value: Mapping[str, Any]) -> None:
    target = Path(path).resolve()
    _atomic_json(target, value)
    print(json.dumps(dict(value), ensure_ascii=False, sort_keys=True))


if modal is not None:
    local_root = _repo_root()
    wheel_path = local_root / LOCAL_RUNTIME_WHEEL
    panel_s_image = (
        modal.Image.from_registry(BASE_IMAGE, add_python=None)
        .run_commands("python -m pip install --no-cache-dir --upgrade pip==26.2.1 setuptools==80.9.0 wheel==0.46.3")
        .pip_install(
            *RUNTIME_REQUIREMENTS,
            extra_index_url=PYTORCH_INDEX_URL,
            extra_options="--no-cache-dir",
        )
        .add_local_dir(
            local_root / "tools/gate13_causal_return",
            "/opt/gate13/tools/gate13_causal_return",
            copy=True,
            ignore=["**/__pycache__/**", "**/*.pyc"],
        )
        .add_local_dir(
            local_root / "analysis/gate13_causal_return/checkpoint_panel",
            "/opt/gate13/analysis/gate13_causal_return/checkpoint_panel",
            copy=True,
            ignore=[AUTHORIZATION_FILE, "**/__pycache__/**", "**/*.pyc"],
        )
        .workdir(REMOTE_ROOT)
        .env({"PYTHONPATH": str(REMOTE_ROOT), "PYTHONUNBUFFERED": "1"})
    )
    panel_g_requirements = tuple(
        requirement
        for requirement in RUNTIME_REQUIREMENTS
        if not requirement.startswith("transformers==")
    ) + ("pillow==12.1.1", "torchvision==0.22.1+cu126")
    panel_g_image = (
        modal.Image.from_registry(BASE_IMAGE, add_python=None)
        .run_commands("python -m pip install --no-cache-dir --upgrade pip==26.2.1 setuptools==80.9.0 wheel==0.46.3")
        .pip_install(
            *panel_g_requirements,
            extra_index_url=PYTORCH_INDEX_URL,
            extra_options="--no-cache-dir",
        )
        .add_local_file(wheel_path, REMOTE_RUNTIME_WHEEL, copy=True)
        .run_commands(f"python -m pip install --no-cache-dir --no-deps {REMOTE_RUNTIME_WHEEL}")
        .add_local_dir(
            local_root / "tools/gate13_causal_return",
            "/opt/gate13/tools/gate13_causal_return",
            copy=True,
            ignore=["**/__pycache__/**", "**/*.pyc"],
        )
        .add_local_dir(
            local_root / "analysis/gate13_causal_return/checkpoint_panel",
            "/opt/gate13/analysis/gate13_causal_return/checkpoint_panel",
            copy=True,
            ignore=[AUTHORIZATION_FILE, "**/__pycache__/**", "**/*.pyc"],
        )
        .workdir(REMOTE_ROOT)
        .env({"PYTHONPATH": str(REMOTE_ROOT), "PYTHONUNBUFFERED": "1"})
    )
    app = modal.App(APP_NAME)
    model_volumes = {
        key: modal.Volume.from_name(model_volume_name(key), create_if_missing=True, version=2)
        for key in CHECKPOINTS
    }
    result_volumes = {
        key: modal.Volume.from_name(result_volume_name(key), create_if_missing=True, version=2)
        for key in CHECKPOINTS
    }
    model_mounts = {str(MODEL_ROOT / key): volume for key, volume in model_volumes.items()}
    result_mounts = {str(RESULT_ROOT / key): volume for key, volume in result_volumes.items()}
    common_env = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }

    @app.function(
        image=panel_g_image,
        volumes=model_mounts,
        cpu=ACQUISITION_CPU_CORES,
        memory=ACQUISITION_MEMORY_MIB,
        retries=0,
        timeout=ACQUISITION_TIMEOUT_SECONDS,
        max_containers=1,
        single_use_containers=True,
        include_source=False,
        name="acquire_exact_checkpoint_model",
    )
    def acquire_exact_checkpoint_model(
        checkpoint_key: str,
        authorization_text: str,
        authorization_sha256: str,
    ) -> dict[str, Any]:
        authorization = _authorization(authorization_text, authorization_sha256)
        if checkpoint_key not in CHECKPOINTS:
            raise CheckpointPanelModalError("unknown acquisition checkpoint")
        entry = authorization["checkpoint_executions"][checkpoint_key]
        if entry["model_volume_name"] != model_volume_name(checkpoint_key):
            raise CheckpointPanelModalError("authorized model Volume name mismatch")
        return _acquire(checkpoint_key, authorization, model_volumes[checkpoint_key])

    @app.function(
        image=panel_s_image,
        volumes={**model_mounts, **result_mounts},
        env=common_env,
        gpu="L40S",
        cpu=GPU_CPU_CORES,
        memory=GPU_MEMORY_MIB,
        retries=0,
        timeout=PREFLIGHT_TIMEOUT_SECONDS,
        max_containers=1,
        single_use_containers=True,
        include_source=False,
        block_network=True,
        name="preflight_panel_s_l40s",
    )
    def preflight_panel_s_l40s(authorization_text: str, authorization_sha256: str) -> dict[str, Any]:
        key = "qwen3_14b"
        return _preflight_core(
            key,
            "L40S",
            authorization_text,
            authorization_sha256,
            model_volumes[key],
            result_volumes[key],
        )

    @app.function(
        image=panel_g_image,
        volumes={**model_mounts, **result_mounts},
        env=common_env,
        gpu="A100-80GB",
        cpu=GPU_CPU_CORES,
        memory=GPU_MEMORY_MIB,
        retries=0,
        timeout=PREFLIGHT_TIMEOUT_SECONDS,
        max_containers=1,
        single_use_containers=True,
        include_source=False,
        block_network=True,
        name="preflight_panel_g_a100_80gb",
    )
    def preflight_panel_g_a100_80gb(
        checkpoint_key: str,
        authorization_text: str,
        authorization_sha256: str,
    ) -> dict[str, Any]:
        if checkpoint_key == "qwen3_14b":
            raise CheckpointPanelModalError("Panel S checkpoint cannot use Panel G runtime")
        return _preflight_core(
            checkpoint_key,
            "A100-80GB",
            authorization_text,
            authorization_sha256,
            model_volumes[checkpoint_key],
            result_volumes[checkpoint_key],
        )

    @app.function(
        image=panel_g_image,
        volumes={**model_mounts, **result_mounts},
        env=common_env,
        gpu="H200",
        cpu=GPU_CPU_CORES,
        memory=GPU_MEMORY_MIB,
        retries=0,
        timeout=PREFLIGHT_TIMEOUT_SECONDS,
        max_containers=1,
        single_use_containers=True,
        include_source=False,
        block_network=True,
        name="preflight_panel_g_h200_fallback",
    )
    def preflight_panel_g_h200_fallback(
        checkpoint_key: str,
        authorization_text: str,
        authorization_sha256: str,
    ) -> dict[str, Any]:
        return _preflight_core(
            checkpoint_key,
            "H200",
            authorization_text,
            authorization_sha256,
            model_volumes[checkpoint_key],
            result_volumes[checkpoint_key],
        )

    @app.function(
        image=panel_s_image,
        volumes={**model_mounts, **result_mounts},
        env=common_env,
        gpu="L40S",
        cpu=GPU_CPU_CORES,
        memory=GPU_MEMORY_MIB,
        retries=0,
        timeout=QUALIFICATION_TIMEOUT_SECONDS,
        max_containers=1,
        single_use_containers=True,
        include_source=False,
        block_network=True,
        name="qualify_panel_s_l40s",
    )
    def qualify_panel_s_l40s(
        authorization_text: str,
        authorization_sha256: str,
        budget_gate_text: str,
    ) -> dict[str, Any]:
        key = "qwen3_14b"
        return _qualification_core(
            key,
            "L40S",
            authorization_text,
            authorization_sha256,
            budget_gate_text,
            model_volumes[key],
            result_volumes[key],
        )

    @app.function(
        image=panel_g_image,
        volumes={**model_mounts, **result_mounts},
        env=common_env,
        gpu="A100-80GB",
        cpu=GPU_CPU_CORES,
        memory=GPU_MEMORY_MIB,
        retries=0,
        timeout=QUALIFICATION_TIMEOUT_SECONDS,
        max_containers=1,
        single_use_containers=True,
        include_source=False,
        block_network=True,
        name="qualify_panel_g_a100_80gb",
    )
    def qualify_panel_g_a100_80gb(
        checkpoint_key: str,
        authorization_text: str,
        authorization_sha256: str,
        budget_gate_text: str,
    ) -> dict[str, Any]:
        return _qualification_core(
            checkpoint_key,
            "A100-80GB",
            authorization_text,
            authorization_sha256,
            budget_gate_text,
            model_volumes[checkpoint_key],
            result_volumes[checkpoint_key],
        )

    @app.function(
        image=panel_g_image,
        volumes={**model_mounts, **result_mounts},
        env=common_env,
        gpu="H200",
        cpu=GPU_CPU_CORES,
        memory=GPU_MEMORY_MIB,
        retries=0,
        timeout=QUALIFICATION_TIMEOUT_SECONDS,
        max_containers=1,
        single_use_containers=True,
        include_source=False,
        block_network=True,
        name="qualify_panel_g_h200_fallback",
    )
    def qualify_panel_g_h200_fallback(
        checkpoint_key: str,
        authorization_text: str,
        authorization_sha256: str,
        budget_gate_text: str,
    ) -> dict[str, Any]:
        return _qualification_core(
            checkpoint_key,
            "H200",
            authorization_text,
            authorization_sha256,
            budget_gate_text,
            model_volumes[checkpoint_key],
            result_volumes[checkpoint_key],
        )

    @app.function(
        image=panel_s_image,
        volumes={**model_mounts, **result_mounts},
        env=common_env,
        gpu="L40S",
        cpu=GPU_CPU_CORES,
        memory=GPU_MEMORY_MIB,
        retries=0,
        timeout=OPERATOR_TIMEOUT_SECONDS,
        max_containers=1,
        single_use_containers=True,
        include_source=False,
        block_network=True,
        name="operator_panel_s_l40s",
    )
    def operator_panel_s_l40s(
        authorization_text: str,
        authorization_sha256: str,
        panel_results_text: str,
    ) -> dict[str, Any]:
        key = "qwen3_14b"
        return _operator_core(
            key,
            "L40S",
            authorization_text,
            authorization_sha256,
            panel_results_text,
            model_volumes[key],
            result_volumes[key],
        )

    @app.function(
        image=panel_g_image,
        volumes={**model_mounts, **result_mounts},
        env=common_env,
        gpu="A100-80GB",
        cpu=GPU_CPU_CORES,
        memory=GPU_MEMORY_MIB,
        retries=0,
        timeout=OPERATOR_TIMEOUT_SECONDS,
        max_containers=1,
        single_use_containers=True,
        include_source=False,
        block_network=True,
        name="operator_panel_g_a100_80gb",
    )
    def operator_panel_g_a100_80gb(
        checkpoint_key: str,
        authorization_text: str,
        authorization_sha256: str,
        panel_results_text: str,
    ) -> dict[str, Any]:
        return _operator_core(
            checkpoint_key,
            "A100-80GB",
            authorization_text,
            authorization_sha256,
            panel_results_text,
            model_volumes[checkpoint_key],
            result_volumes[checkpoint_key],
        )

    @app.function(
        image=panel_g_image,
        volumes={**model_mounts, **result_mounts},
        env=common_env,
        gpu="H200",
        cpu=GPU_CPU_CORES,
        memory=GPU_MEMORY_MIB,
        retries=0,
        timeout=OPERATOR_TIMEOUT_SECONDS,
        max_containers=1,
        single_use_containers=True,
        include_source=False,
        block_network=True,
        name="operator_panel_g_h200_fallback",
    )
    def operator_panel_g_h200_fallback(
        checkpoint_key: str,
        authorization_text: str,
        authorization_sha256: str,
        panel_results_text: str,
    ) -> dict[str, Any]:
        return _operator_core(
            checkpoint_key,
            "H200",
            authorization_text,
            authorization_sha256,
            panel_results_text,
            model_volumes[checkpoint_key],
            result_volumes[checkpoint_key],
        )

    def _auth_from_path(path: str) -> tuple[str, str]:
        text = Path(path).resolve().read_text(encoding="utf-8")
        return text, _sha256_text(text)

    @app.local_entrypoint()
    def acquire(checkpoint_key: str, authorization: str, control_output: str) -> None:
        if checkpoint_key not in CHECKPOINTS:
            raise CheckpointPanelModalError("unknown checkpoint key")
        panel_g_image.hydrate()
        model_volumes[checkpoint_key].hydrate()
        text, digest = _auth_from_path(authorization)
        result = acquire_exact_checkpoint_model.remote(checkpoint_key, text, digest)
        _local_control(
            control_output,
            {
                "schema_version": "gate13_checkpoint_panel_acquisition_control_v1",
                "checkpoint_key": checkpoint_key,
                "authorization_sha256": digest,
                "modal_image_object_id": panel_g_image.object_id,
                "model_volume_name": model_volumes[checkpoint_key].name,
                "model_volume_object_id": model_volumes[checkpoint_key].object_id,
                "result": result,
            },
        )

    @app.local_entrypoint()
    def preflight(
        checkpoint_key: str,
        authorization: str,
        control_output: str,
        gpu: str = "primary",
    ) -> None:
        text, digest = _auth_from_path(authorization)
        model_volumes[checkpoint_key].hydrate()
        result_volumes[checkpoint_key].hydrate()
        if checkpoint_key == "qwen3_14b":
            if gpu != "primary":
                raise CheckpointPanelModalError("Panel S has no GPU fallback")
            panel_s_image.hydrate()
            result = preflight_panel_s_l40s.remote(text, digest)
            image_id = panel_s_image.object_id
        else:
            panel_g_image.hydrate()
            if gpu == "primary":
                result = preflight_panel_g_a100_80gb.remote(checkpoint_key, text, digest)
            elif gpu == "fallback":
                result = preflight_panel_g_h200_fallback.remote(checkpoint_key, text, digest)
            else:
                raise CheckpointPanelModalError("gpu must be primary or fallback")
            image_id = panel_g_image.object_id
        _local_control(
            control_output,
            {
                "schema_version": "gate13_checkpoint_panel_preflight_control_v1",
                "checkpoint_key": checkpoint_key,
                "authorization_sha256": digest,
                "modal_image_object_id": image_id,
                "model_volume_object_id": model_volumes[checkpoint_key].object_id,
                "result_volume_object_id": result_volumes[checkpoint_key].object_id,
                "result": result,
            },
        )

    @app.local_entrypoint()
    def qualify(
        checkpoint_key: str,
        authorization: str,
        budget_gate: str,
        control_output: str,
        gpu: str = "primary",
    ) -> None:
        text, digest = _auth_from_path(authorization)
        gate_text = Path(budget_gate).resolve().read_text(encoding="utf-8")
        if checkpoint_key == "qwen3_14b":
            result = qualify_panel_s_l40s.remote(text, digest, gate_text)
            image_id = panel_s_image.object_id
        elif gpu == "primary":
            result = qualify_panel_g_a100_80gb.remote(checkpoint_key, text, digest, gate_text)
            image_id = panel_g_image.object_id
        elif gpu == "fallback":
            result = qualify_panel_g_h200_fallback.remote(checkpoint_key, text, digest, gate_text)
            image_id = panel_g_image.object_id
        else:
            raise CheckpointPanelModalError("gpu must be primary or fallback")
        _local_control(
            control_output,
            {
                "schema_version": "gate13_checkpoint_panel_qualification_control_v1",
                "checkpoint_key": checkpoint_key,
                "authorization_sha256": digest,
                "modal_image_object_id": image_id,
                "result": result,
            },
        )

    @app.local_entrypoint()
    def operator(
        checkpoint_key: str,
        authorization: str,
        panel_results: str,
        control_output: str,
        gpu: str = "primary",
    ) -> None:
        text, digest = _auth_from_path(authorization)
        results_text = Path(panel_results).resolve().read_text(encoding="utf-8")
        if checkpoint_key == "qwen3_14b":
            result = operator_panel_s_l40s.remote(text, digest, results_text)
            image_id = panel_s_image.object_id
        elif gpu == "primary":
            result = operator_panel_g_a100_80gb.remote(checkpoint_key, text, digest, results_text)
            image_id = panel_g_image.object_id
        elif gpu == "fallback":
            result = operator_panel_g_h200_fallback.remote(checkpoint_key, text, digest, results_text)
            image_id = panel_g_image.object_id
        else:
            raise CheckpointPanelModalError("gpu must be primary or fallback")
        _local_control(
            control_output,
            {
                "schema_version": "gate13_checkpoint_panel_operator_control_v1",
                "checkpoint_key": checkpoint_key,
                "authorization_sha256": digest,
                "modal_image_object_id": image_id,
                "result": result,
            },
        )

else:  # pragma: no cover
    app = None
    panel_s_image = None
    panel_g_image = None
