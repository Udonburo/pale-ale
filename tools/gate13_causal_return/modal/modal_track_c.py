"""Thin Modal realization adapter for frozen Review 2.1 Track C.

The adapter owns exact model acquisition, runtime realization, atomic case
persistence, and invocation of the frozen compiler/analysis.  It does not
define scientific prompts, cases, estimands, gates, or thresholds.
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
except ModuleNotFoundError:  # pragma: no cover - targeted local tests are model-free
    modal = None  # type: ignore[assignment]
else:
    if not all(hasattr(modal, name) for name in ("App", "Image", "Volume")):
        modal = None  # type: ignore[assignment]

from analysis.gate13_causal_return.review2_1.track_c_review2_1_validator import (
    AnalysisTerminal,
    FROZEN_LAYERS,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    validate_frozen_campaign_manifest,
)
from tools.gate13_causal_return.checkpoint_panel.panel import BASE_IMAGE
from tools.gate13_causal_return.modal.modal_track_a import PYTORCH_INDEX_URL, RUNTIME_REQUIREMENTS
from tools.gate13_causal_return.track_c.campaign import (
    AtomicCaseStore,
    TrackCCampaignError,
    analyze_behavior_and_primary,
    analyze_map_block,
    atomic_json,
    canonical_sha256,
    codebook_from_block,
    evaluate_map_campaign,
    json_ready,
    ledger_indexes,
    load_json,
    official_text_chat_render,
    render_behavior_case,
    render_map_case,
    sha256_file,
    sha256_text,
    synthetic_preflight,
)


APP_NAME = "gate13-track-c-review2-1"
EXECUTION_ID = "bf41b049-f04b-442e-b0bd-05c8adbd4944"
MODEL_VOLUME_NAME = "gate13-track-c-qwen3-6-27b-6a9e13bd-model"
RESULT_VOLUME_NAME = "gate13-track-c-review2-1-bf41b049-results"
REMOTE_ROOT = PurePosixPath("/opt/gate13")
MODEL_MOUNT = Path("/models/qwen3_6_27b")
RESULT_MOUNT = Path("/results")
EXECUTION_DIR = RESULT_MOUNT / "executions" / EXECUTION_ID
CAMPAIGN_DIR = Path("/opt/gate13/analysis/gate13_causal_return/track_c_execution")
MANIFEST_PATH = CAMPAIGN_DIR / "track_c_campaign_manifest.json"
LEDGER_PATH = CAMPAIGN_DIR / "track_c_execution_ledger.json"
PLAN_PATH = CAMPAIGN_DIR / "track_c_campaign_plan.json"
MODEL_REPORT_PATH = MODEL_MOUNT / "model_acquisition.json"
LOCAL_RUNTIME_WHEEL = (
    "workstream/local/gate13_causal_return_outputs/checkpoint_panel/runtime/"
    "transformers-5.15.1-py3-none-any.whl"
)
REMOTE_RUNTIME_WHEEL = "/opt/wheels/transformers-5.15.1-py3-none-any.whl"
ACQUISITION_TIMEOUT_SECONDS = 10_800
STAGE_M_TIMEOUT_SECONDS = 28_800
STAGE_E_TIMEOUT_SECONDS = 43_200
CPU_CORES = 8.0
MEMORY_MIB = 65_536
GPU_RATE_USD_PER_HOUR = 2.50
CPU_RATE_USD_PER_CORE_HOUR = 0.0473
MEMORY_RATE_USD_PER_GIB_HOUR = 0.008
CAMPAIGN_SPEND_CEILING_USD = 65.0


class TrackCModalError(RuntimeError):
    """Fail-closed operational error for the frozen Track C execution."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_root() -> Path:
    remote = Path(REMOTE_ROOT.as_posix())
    if (remote / "tools/gate13_causal_return").is_dir():
        return remote
    return Path(__file__).resolve().parents[3]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _estimated_gpu_container_cost(elapsed_seconds: float) -> float:
    hourly = (
        GPU_RATE_USD_PER_HOUR
        + CPU_CORES * CPU_RATE_USD_PER_CORE_HOUR
        + (MEMORY_MIB / 1024.0) * MEMORY_RATE_USD_PER_GIB_HOUR
    )
    return elapsed_seconds * hourly / 3600.0


def _authorization(text: str, claimed_sha256: str) -> dict[str, Any]:
    actual = _sha256_bytes(text.encode("utf-8"))
    if actual != claimed_sha256:
        raise TrackCModalError("execution authorization byte hash mismatch")
    value = json.loads(text)
    if not isinstance(value, dict):
        raise TrackCModalError("execution authorization root is not an object")
    required = {
        "execution_authorized": True,
        "execution_id": EXECUTION_ID,
        "model_volume_name": MODEL_VOLUME_NAME,
        "result_volume_name": RESULT_VOLUME_NAME,
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "gpu": "A100-80GB",
        "dtype": "bfloat16",
        "quantization": False,
        "automatic_retries": 0,
        "maximum_active_gpu_containers": 1,
        "absolute_campaign_ceiling_usd": CAMPAIGN_SPEND_CEILING_USD,
        "a3": "CLOSED",
        "formal_gate13": "CLOSED",
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise TrackCModalError(f"execution authorization mismatch: {key}")
    for filename, path in (
        ("track_c_campaign_plan.json", PLAN_PATH),
        ("track_c_campaign_manifest.json", MANIFEST_PATH),
        ("track_c_execution_ledger.json", LEDGER_PATH),
    ):
        expected = value.get("frozen_artifacts_sha256", {}).get(filename)
        if not isinstance(expected, str) or sha256_file(path) != expected:
            raise TrackCModalError(f"frozen execution artifact mismatch: {filename}")
    for group in (
        "review2_1_files_sha256",
        "historical_authority_files_sha256",
        "panel_files_sha256",
        "runner_sources_sha256",
    ):
        bindings = value.get(group)
        if not isinstance(bindings, Mapping) or not bindings:
            raise TrackCModalError(f"authorization hash group missing: {group}")
        for relative, expected in bindings.items():
            path = Path(REMOTE_ROOT.as_posix()) / str(relative)
            if not path.is_file() or sha256_file(path) != expected:
                raise TrackCModalError(f"authority/source hash mismatch: {relative}")
    if value.get("model_volume_object_id") != model_volume.object_id:
        raise TrackCModalError("model Volume identity mismatch")
    if value.get("result_volume_object_id") != result_volume.object_id:
        raise TrackCModalError("result Volume identity mismatch")
    return value


def _claim(authorization: Mapping[str, Any], authorization_sha256: str) -> None:
    path = EXECUTION_DIR / "execution_claim.json"
    value = {
        "execution_id": EXECUTION_ID,
        "authorization_sha256": authorization_sha256,
        "model_volume_name": MODEL_VOLUME_NAME,
        "model_volume_object_id": model_volume.object_id,
        "result_volume_name": RESULT_VOLUME_NAME,
        "result_volume_object_id": result_volume.object_id,
    }
    if path.exists():
        if load_json(path) != value:
            raise TrackCModalError("execution identity is already claimed by different authority")
        return
    atomic_json(path, value)
    result_volume.commit()


def _expected_registry() -> tuple[dict[str, Any], dict[str, Any]]:
    from tools.gate13_causal_return.checkpoint_panel.panel import CHECKPOINTS
    from tools.gate13_causal_return.modal.modal_checkpoint_transfer_panel import _registry_row

    return dict(CHECKPOINTS["qwen3_6_27b"]), dict(_registry_row("qwen3_6_27b"))


def _verify_snapshot() -> dict[str, Any]:
    if not MODEL_REPORT_PATH.is_file():
        raise TrackCModalError("exact model acquisition report is unavailable")
    report = load_json(MODEL_REPORT_PATH)
    spec, expected = _expected_registry()
    if report.get("repo_id") != spec["repo_id"] or report.get("resolved_revision") != spec["revision"]:
        raise TrackCModalError("model repository/revision mismatch")
    if report.get("model_directory_identity_sha256") != expected["model_directory_identity_sha256"]:
        raise TrackCModalError("model directory identity mismatch")
    snapshot = Path(str(report["snapshot_directory"]))
    if snapshot.name != MODEL_REVISION or not snapshot.is_dir():
        raise TrackCModalError("immutable model snapshot is unavailable")
    for row in expected["file_inventory"]:
        path = snapshot / str(row["path"])
        if not path.is_file() or path.stat().st_size != int(row["bytes"]):
            raise TrackCModalError(f"model inventory mismatch: {row['path']}")
    for name, expected_sha in expected["asset_sha256"].items():
        if expected_sha is not None and sha256_file(snapshot / name) != expected_sha:
            raise TrackCModalError(f"model asset hash mismatch: {name}")
    return {**report, "verification_status": "PASS"}


def _acquire_model() -> dict[str, Any]:
    from huggingface_hub import snapshot_download
    from tools.gate13_causal_return.stepwise.compiler import sha256_json

    if MODEL_REPORT_PATH.exists():
        return {**_verify_snapshot(), "idempotent_retrieval": True}
    spec, expected = _expected_registry()
    started = time.monotonic()
    cache = MODEL_MOUNT / "hf_home" / "hub"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        {
            "HF_HOME": str(MODEL_MOUNT / "hf_home"),
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
    if snapshot.name != MODEL_REVISION:
        raise TrackCModalError("snapshot_download did not resolve the frozen revision")
    inventory = []
    for row in expected["file_inventory"]:
        path = snapshot / str(row["path"])
        if not path.is_file() or path.stat().st_size != int(row["bytes"]):
            raise TrackCModalError(f"acquired inventory mismatch: {row['path']}")
        inventory.append(
            {
                "path": row["path"],
                "bytes": path.stat().st_size,
                "content_identity": row.get("lfs_sha256") or row.get("git_blob_id"),
            }
        )
    if sha256_json(inventory) != expected["model_directory_identity_sha256"]:
        raise TrackCModalError("acquired model directory identity mismatch")
    assets = {}
    for name, expected_sha in expected["asset_sha256"].items():
        if expected_sha is None:
            continue
        actual = sha256_file(snapshot / name)
        if actual != expected_sha:
            raise TrackCModalError(f"acquired asset hash mismatch: {name}")
        assets[name] = actual
    elapsed = time.monotonic() - started
    report = {
        "schema_version": "gate13_track_c_exact_model_acquisition_v1",
        "repo_id": spec["repo_id"],
        "resolved_revision": spec["revision"],
        "snapshot_directory": str(snapshot),
        "model_volume_name": MODEL_VOLUME_NAME,
        "model_volume_object_id": model_volume.object_id,
        "file_count": len(inventory),
        "total_file_bytes": sum(int(row["bytes"]) for row in inventory),
        "complete_file_inventory": inventory,
        "asset_sha256": assets,
        "weight_shard_identities": [
            {"path": row["path"], "bytes": row["bytes"], "lfs_sha256": row.get("lfs_sha256")}
            for row in expected["file_inventory"]
            if str(row["path"]).endswith(".safetensors")
        ],
        "model_directory_identity_sha256": expected["model_directory_identity_sha256"],
        "cpu_acquisition_elapsed_seconds": elapsed,
        "gpu_allocated": False,
        "scientific_model_forward_count": 0,
        "volume_immutable_after_acquisition": True,
        "acquired_at": utc_now(),
    }
    atomic_json(MODEL_REPORT_PATH, report)
    model_volume.commit()
    return report


def _m0_runtime() -> dict[str, Any]:
    from tools.gate13_causal_return.modal.modal_checkpoint_transfer_panel import _m0

    report = _m0("qwen3_6_27b", "A100-80GB")
    report["scientific_model_forward_count"] = 0
    return report


def _load_probe() -> tuple[Any, Any, Any, Any, Mapping[str, Any]]:
    from tools.gate13_causal_return.modal.modal_checkpoint_transfer_panel import PanelProbe, _load_model

    torch, endpoint, tokenizer, model, memory = _load_model("qwen3_6_27b")
    return torch, endpoint, tokenizer, PanelProbe(torch, endpoint, tokenizer, model), memory


def _block_index(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(block["block_id"]): dict(block) for block in manifest["blocks"]}


def _validate_prompt_runtime(
    *,
    endpoint: Any,
    tokenizer: Any,
    prompt: str,
    candidates: Sequence[str],
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    from tools.gate13_causal_return.modal.modal_checkpoint_transfer_panel import _prepare_inputs

    rendered, inputs = _prepare_inputs(endpoint, "runtime", prompt)
    if rendered != official_text_chat_render(prompt):
        raise TrackCModalError("official chat renderer drift")
    observed = {
        "canonical_message_sha256": sha256_text(prompt),
        "rendered_prompt_sha256": sha256_text(rendered),
        "input_ids_sha256": canonical_sha256(inputs["input_ids"][0].detach().cpu().tolist()),
        "input_token_count": int(inputs["input_ids"].shape[-1]),
    }
    if observed != dict(binding):
        raise TrackCModalError("frozen prompt/token binding mismatch")
    candidate_rows = []
    for label in candidates:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1 or int(ids[0]) in tokenizer.all_special_ids:
            raise TrackCModalError("candidate label is not one non-special token")
        appended = tokenizer.encode(rendered + label, add_special_tokens=False)
        if list(map(int, appended)) != inputs["input_ids"][0].detach().cpu().tolist() + [int(ids[0])]:
            raise TrackCModalError("candidate is not direct at the first semantic answer slot")
        candidate_rows.append({"label": label, "token_id": int(ids[0])})
    return {**observed, "candidate_labels": candidate_rows, "status": "PASS"}


def _activation_loader(root: Path):
    def load(row: Mapping[str, Any]) -> dict[int, Any]:
        import numpy as np

        path = root / str(row["activation_artifact"])
        if not path.is_file() or sha256_file(path) != row["activation_artifact_sha256"]:
            raise TrackCModalError(f"activation artifact mismatch: {row['case_id']}")
        with np.load(path) as stored:
            return {int(layer): stored[f"layer_{layer}"].copy() for layer in FROZEN_LAYERS}

    return load


def _stage_m_core(authorization_text: str, authorization_sha256: str) -> dict[str, Any]:
    import numpy as np

    authorization = _authorization(authorization_text, authorization_sha256)
    _claim(authorization, authorization_sha256)
    manifest = load_json(MANIFEST_PATH)
    ledger = load_json(LEDGER_PATH)
    validate_frozen_campaign_manifest(manifest)
    model_free = synthetic_preflight(manifest, ledger)
    atomic_json(EXECUTION_DIR / "model_free_runner_preflight.json", model_free)
    result_volume.commit()
    existing_terminal = EXECUTION_DIR / "stage_m_terminal.json"
    if existing_terminal.exists():
        return {**load_json(existing_terminal), "idempotent_retrieval": True}
    m0 = _m0_runtime()
    atomic_json(EXECUTION_DIR / "m0_runtime.json", m0)
    result_volume.commit()
    if m0["status"] != "PASS":
        terminal = {
            "terminal_state": "TRACK_C_PREFLIGHT_BLOCKED",
            "m0": m0,
            "stage_m_scientific_forwards": 0,
            "created_at": utc_now(),
        }
        atomic_json(existing_terminal, terminal)
        result_volume.commit()
        return terminal
    started = time.monotonic()
    torch, endpoint, tokenizer, probe, memory = _load_probe()
    map_index, _behavior_index = ledger_indexes(ledger)
    blocks = _block_index(manifest)
    representative = []
    for block in manifest["blocks"]:
        case = map_index[str(block["map_case_ids"][0])]
        prompt = render_map_case(case, block)
        representative.append(
            {
                "block_id": block["block_id"],
                **_validate_prompt_runtime(
                    endpoint=endpoint,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    candidates=case["candidate_labels"],
                    binding=case["prompt_binding"],
                ),
            }
        )
    atomic_json(
        EXECUTION_DIR / "m1_forward0_score_slot_preflight.json",
        {
            "status": "PASS",
            "model_load": memory,
            "representative_block_count": len(representative),
            "representative_bindings": representative,
            "scientific_model_forward_count": 0,
        },
    )
    result_volume.commit()
    store = AtomicCaseStore(EXECUTION_DIR, "stage_m")
    accepted_before = store.accepted_ids()
    new_forwards = 0
    try:
        for ordinal, case_id in enumerate(manifest["execution"]["stage_m_order"], start=1):
            case = map_index[str(case_id)]
            existing = store.accepted(str(case_id))
            if existing is not None:
                continue
            block = blocks[str(case["block_id"])]
            prompt = render_map_case(case, block)
            expected_binding = dict(case["prompt_binding"])
            if sha256_text(prompt) != expected_binding["canonical_message_sha256"]:
                raise TrackCModalError(f"map prompt drift: {case_id}")
            binding = {
                "case_id": case_id,
                "frozen_case_sha256": canonical_sha256(case),
                "prompt_binding": expected_binding,
                "authorization_sha256": authorization_sha256,
            }
            store.record_attempt(str(case_id), binding)
            result_volume.commit()
            meta, vectors = probe.activation_call(
                prompt,
                tuple(map(str, case["candidate_labels"])),
                FROZEN_LAYERS,
            )
            new_forwards += 1
            artifact = EXECUTION_DIR / "stage_m" / "activations" / f"{case_id}.npz"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            temporary = artifact.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    **{f"layer_{layer}": vectors[int(layer)] for layer in FROZEN_LAYERS},
                )
            os.replace(temporary, artifact)
            response = {
                **case,
                **meta,
                "case_id": case_id,
                "call_role": "MAP_ACTIVATION_AND_FORCED_CHOICE",
                "activation_artifact": artifact.relative_to(EXECUTION_DIR).as_posix(),
                "activation_artifact_sha256": sha256_file(artifact),
                "accepted_at": utc_now(),
            }
            store.accept(str(case_id), response)
            result_volume.commit()
            if ordinal % 50 == 0:
                progress = {
                    "stage": "M",
                    "accepted": len(store.accepted_ids()),
                    "required": 4_800,
                    "new_forwards_this_invocation": new_forwards,
                    "elapsed_seconds": time.monotonic() - started,
                }
                atomic_json(EXECUTION_DIR / "stage_m_progress.json", progress)
                result_volume.commit()
                print(json.dumps(progress, sort_keys=True), flush=True)
        response_rows = [
            store.accepted(case_id) for case_id in manifest["execution"]["stage_m_order"]
        ]
        if any(row is None for row in response_rows):
            raise TrackCModalError("Stage M accepted-ID ledger is incomplete")
        block_results = [
            analyze_map_block(
                block=block,
                response_rows=response_rows,  # type: ignore[arg-type]
                activation_loader=_activation_loader(EXECUTION_DIR),
            )
            for block in manifest["blocks"]
        ]
        private = {
            "schema_version": "gate13_track_c_sealed_map_result_v1",
            "block_results": block_results,
            "stage_m_forwards": 4_800,
            "manifest_sha256": sha256_file(MANIFEST_PATH),
            "ledger_sha256": sha256_file(LEDGER_PATH),
        }
        atomic_json(EXECUTION_DIR / "sealed_map_private_result.json", private)
        try:
            qualification = evaluate_map_campaign(
                manifest=manifest,
                block_results=block_results,
                artifact_complete=len(store.accepted_ids()) == 4_800,
            )
        except AnalysisTerminal as exc:
            elapsed = time.monotonic() - started
            terminal = {
                "terminal_state": "MAP_COMPLETE_NOT_QUALIFIED",
                "track_c_terminal_close": True,
                "scientific_cause": exc.state,
                "scientific_details": exc.details,
                "stage_m_scientific_forwards": 4_800,
                "new_forwards_this_invocation": new_forwards,
                "elapsed_seconds": elapsed,
                "estimated_modal_cost_usd": _estimated_gpu_container_cost(elapsed),
                "private_map_result_sha256": sha256_file(
                    EXECUTION_DIR / "sealed_map_private_result.json"
                ),
                "created_at": utc_now(),
            }
            atomic_json(existing_terminal, terminal)
            result_volume.commit()
            return terminal
        elapsed = time.monotonic() - started
        summary = {
            "terminal_state": "MAP_COMPLETE_AND_QUALIFIED",
            "qualification_state": "MAP_COMPLETE_AND_QUALIFIED",
            "qualified_count_by_depth": qualification["sealed_public_summary"]["depth_counts"],
            "qualified_blocks": qualification["sealed_public_summary"]["qualified_blocks"],
            "stage_m_scientific_forwards": 4_800,
            "new_forwards_this_invocation": new_forwards,
            "elapsed_seconds": elapsed,
            "estimated_modal_cost_usd": _estimated_gpu_container_cost(elapsed),
            "private_map_result_sha256": sha256_file(
                EXECUTION_DIR / "sealed_map_private_result.json"
            ),
            "artifact_hashes": {
                "manifest": sha256_file(MANIFEST_PATH),
                "ledger": sha256_file(LEDGER_PATH),
            },
            "created_at": utc_now(),
        }
        atomic_json(existing_terminal, summary)
        result_volume.commit()
        return summary
    except Exception as exc:
        elapsed = time.monotonic() - started
        terminal = {
            "terminal_state": "TRACK_C_OPERATIONAL_ABORT",
            "stage": "M",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "accepted_forwards": len(store.accepted_ids()),
            "new_forwards_this_invocation": new_forwards,
            "accepted_before_invocation": len(accepted_before),
            "elapsed_seconds": elapsed,
            "estimated_modal_cost_usd": _estimated_gpu_container_cost(elapsed),
            "created_at": utc_now(),
        }
        atomic_json(existing_terminal, terminal)
        result_volume.commit()
        return terminal


def _stage_e_core(
    authorization_text: str,
    authorization_sha256: str,
    billing_gate_text: str,
) -> dict[str, Any]:
    authorization = _authorization(authorization_text, authorization_sha256)
    _claim(authorization, authorization_sha256)
    billing_gate = json.loads(billing_gate_text)
    if billing_gate.get("stage_e_authorized_by_balance_check") is not True:
        raise TrackCModalError("Stage E billing gate is not open")
    if float(billing_gate.get("cumulative_forecast_usd", 1e9)) > CAMPAIGN_SPEND_CEILING_USD:
        raise TrackCModalError("Stage E cumulative forecast exceeds campaign ceiling")
    stage_m = load_json(EXECUTION_DIR / "stage_m_terminal.json")
    if stage_m.get("terminal_state") != "MAP_COMPLETE_AND_QUALIFIED":
        raise TrackCModalError("Stage E requires a qualified sealed Stage M")
    terminal_path = EXECUTION_DIR / "stage_e_terminal.json"
    if terminal_path.exists():
        return {**load_json(terminal_path), "idempotent_retrieval": True}
    manifest = load_json(MANIFEST_PATH)
    ledger = load_json(LEDGER_PATH)
    _map_index, behavior_index = ledger_indexes(ledger)
    blocks = _block_index(manifest)
    private_map = load_json(EXECUTION_DIR / "sealed_map_private_result.json")
    map_results = private_map["block_results"]
    qualified = {
        str(row["block_id"]) for row in map_results if bool(row["qualified"])
    }
    required_order = [
        str(case_id)
        for case_id in manifest["execution"]["stage_e_order"]
        if str(behavior_index[str(case_id)]["block_id"]) in qualified
    ]
    started = time.monotonic()
    torch, endpoint, tokenizer, probe, memory = _load_probe()
    del endpoint, tokenizer, memory
    store = AtomicCaseStore(EXECUTION_DIR, "stage_e")
    new_forwards = 0
    try:
        for ordinal, case_id in enumerate(required_order, start=1):
            case = behavior_index[case_id]
            if store.accepted(case_id) is not None:
                continue
            block = blocks[str(case["block_id"])]
            call_index = int(case["call_index"])
            if call_index == 0:
                current_state = int(case["oracle_state_before"])
            else:
                prefix = f"tc-e-{case['episode_id']}-{case['path_id']}-"
                previous_id = (
                    f"{prefix}s{call_index - 1:02d}"
                    if call_index - 1 < int(block["rollout_depth"])
                    else f"{prefix}probe"
                )
                previous = store.accepted(previous_id)
                if previous is None:
                    raise TrackCModalError(f"self-fed predecessor missing: {case_id}")
                current_state = int(previous["predicted_state"])
            prompt = render_behavior_case(case, block, current_state=current_state)
            binding = case["prompt_variants"][str(current_state)]
            if sha256_text(prompt) != binding["canonical_message_sha256"]:
                raise TrackCModalError(f"behavior prompt drift: {case_id}")
            attempt = {
                "case_id": case_id,
                "frozen_case_sha256": canonical_sha256(case),
                "selected_current_state": current_state,
                "selected_prompt_binding": binding,
                "authorization_sha256": authorization_sha256,
            }
            store.record_attempt(case_id, attempt)
            result_volume.commit()
            meta = dict(probe(prompt, tuple(map(str, case["candidate_labels"])), case))
            predicted_state = 0 if meta["candidate_logits"][0] > meta["candidate_logits"][1] else 1
            response = {
                **case,
                **meta,
                "case_id": case_id,
                "selected_current_state": current_state,
                "predicted_state": predicted_state,
                "accepted_at": utc_now(),
            }
            store.accept(case_id, response)
            result_volume.commit()
            new_forwards += 1
            if ordinal % 50 == 0:
                progress = {
                    "stage": "E",
                    "accepted": len(store.accepted_ids()),
                    "required": len(required_order),
                    "new_forwards_this_invocation": new_forwards,
                    "elapsed_seconds": time.monotonic() - started,
                }
                atomic_json(EXECUTION_DIR / "stage_e_progress.json", progress)
                result_volume.commit()
                print(json.dumps(progress, sort_keys=True), flush=True)
        rows = [store.accepted(case_id) for case_id in required_order]
        if any(row is None for row in rows):
            raise TrackCModalError("Stage E accepted-ID ledger is incomplete")
        analysis_attempt = EXECUTION_DIR / "primary_analysis_attempt.json"
        if analysis_attempt.exists() and not (EXECUTION_DIR / "primary_result.json").exists():
            raise TrackCModalError("primary analysis has an ambiguous prior attempt")
        atomic_json(
            analysis_attempt,
            {
                "attempted_at": utc_now(),
                "qualified_block_ids": sorted(qualified),
                "behavior_response_manifest_sha256": canonical_sha256(rows),
            },
        )
        result_volume.commit()
        try:
            primary = analyze_behavior_and_primary(
                manifest=manifest,
                map_block_results=map_results,
                behavior_responses=rows,  # type: ignore[arg-type]
            )
            scientific_terminal = primary["primary"]["terminal_state"]
        except AnalysisTerminal as exc:
            primary = {
                "schema_version": "gate13_track_c_primary_result_v1",
                "primary": {
                    "terminal_state": exc.state,
                    "details": exc.details,
                },
            }
            scientific_terminal = exc.state
        atomic_json(EXECUTION_DIR / "primary_result.json", primary)
        elapsed = time.monotonic() - started
        terminal = {
            "terminal_state": (
                "TRACK_C_COMPLETE_PRIMARY_POSITIVE"
                if scientific_terminal == "PRIMARY_POSITIVE"
                else "TRACK_C_COMPLETE_PRIMARY_NONPOSITIVE"
            ),
            "scientific_terminal": scientific_terminal,
            "stage_e_scientific_forwards": len(required_order),
            "new_forwards_this_invocation": new_forwards,
            "elapsed_seconds": elapsed,
            "estimated_modal_cost_usd": _estimated_gpu_container_cost(elapsed),
            "primary_result_sha256": sha256_file(EXECUTION_DIR / "primary_result.json"),
            "a3": "CLOSED",
            "formal_gate13": "CLOSED",
            "created_at": utc_now(),
        }
        atomic_json(terminal_path, terminal)
        result_volume.commit()
        return terminal
    except Exception as exc:
        elapsed = time.monotonic() - started
        terminal = {
            "terminal_state": "TRACK_C_OPERATIONAL_ABORT",
            "stage": "E",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "accepted_forwards": len(store.accepted_ids()),
            "new_forwards_this_invocation": new_forwards,
            "elapsed_seconds": elapsed,
            "estimated_modal_cost_usd": _estimated_gpu_container_cost(elapsed),
            "created_at": utc_now(),
        }
        atomic_json(terminal_path, terminal)
        result_volume.commit()
        return terminal


if modal is not None:
    local_root = _repo_root()
    wheel_path = local_root / LOCAL_RUNTIME_WHEEL
    requirements = tuple(
        requirement for requirement in RUNTIME_REQUIREMENTS if not requirement.startswith("transformers==")
    ) + ("pillow==12.1.1", "torchvision==0.22.1+cu126")
    track_c_image = (
        modal.Image.from_registry(BASE_IMAGE, add_python=None)
        .run_commands(
            "python -m pip install --no-cache-dir --upgrade "
            "pip==26.2.1 setuptools==80.9.0 wheel==0.46.3"
        )
        .pip_install(
            *requirements,
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
            local_root / "analysis/gate13_causal_return",
            "/opt/gate13/analysis/gate13_causal_return",
            copy=True,
            ignore=["**/__pycache__/**", "**/*.pyc"],
        )
        .workdir(REMOTE_ROOT)
        .env({"PYTHONPATH": str(REMOTE_ROOT), "PYTHONUNBUFFERED": "1"})
    )
    app = modal.App(APP_NAME)
    model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True, version=2)
    result_volume = modal.Volume.from_name(RESULT_VOLUME_NAME, create_if_missing=True, version=2)
    offline_env = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }

    @app.function(
        image=track_c_image,
        volumes={str(MODEL_MOUNT): model_volume},
        cpu=CPU_CORES,
        memory=32_768,
        retries=0,
        timeout=ACQUISITION_TIMEOUT_SECONDS,
        max_containers=1,
        single_use_containers=True,
        include_source=False,
        name="acquire_exact_qwen3_6_27b",
    )
    def acquire_exact_qwen3_6_27b(
        authorization_text: str,
        authorization_sha256: str,
    ) -> dict[str, Any]:
        _authorization(authorization_text, authorization_sha256)
        return _acquire_model()

    @app.function(
        image=track_c_image,
        volumes={str(MODEL_MOUNT): model_volume, str(RESULT_MOUNT): result_volume},
        env=offline_env,
        gpu="A100-80GB",
        cpu=CPU_CORES,
        memory=MEMORY_MIB,
        retries=0,
        timeout=STAGE_M_TIMEOUT_SECONDS,
        max_containers=1,
        single_use_containers=True,
        include_source=False,
        block_network=True,
        name="run_frozen_stage_m",
    )
    def run_frozen_stage_m(
        authorization_text: str,
        authorization_sha256: str,
    ) -> dict[str, Any]:
        return _stage_m_core(authorization_text, authorization_sha256)

    @app.function(
        image=track_c_image,
        volumes={str(MODEL_MOUNT): model_volume, str(RESULT_MOUNT): result_volume},
        env=offline_env,
        gpu="A100-80GB",
        cpu=CPU_CORES,
        memory=MEMORY_MIB,
        retries=0,
        timeout=STAGE_E_TIMEOUT_SECONDS,
        max_containers=1,
        single_use_containers=True,
        include_source=False,
        block_network=True,
        name="run_frozen_stage_e",
    )
    def run_frozen_stage_e(
        authorization_text: str,
        authorization_sha256: str,
        billing_gate_text: str,
    ) -> dict[str, Any]:
        return _stage_e_core(authorization_text, authorization_sha256, billing_gate_text)
else:  # pragma: no cover
    app = None
    model_volume = None
    result_volume = None
