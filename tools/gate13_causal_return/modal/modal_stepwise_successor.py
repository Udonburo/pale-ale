"""Modal execution adapter for the autonomous Gate13 stepwise campaign."""

from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

try:
    import modal
except ModuleNotFoundError:  # pragma: no cover - local tests do not require Modal
    modal = None  # type: ignore[assignment]
else:
    if not all(hasattr(modal, name) for name in ("App", "Image", "Volume")):
        modal = None  # type: ignore[assignment]

from tools.gate13_causal_return.modal.modal_track_a import (
    BASE_IMAGE,
    BASE_IMAGE_LINUX_AMD64_DIGEST,
    BASE_IMAGE_MANIFEST_DIGEST,
    GPU_CPU_CORES,
    GPU_MEMORY_MIB,
    GPU_TOTAL_RATE_USD_PER_SECOND,
    HF_CACHE,
    HF_HOME,
    MODEL_ACQUISITION_REPORT,
    MODEL_MOUNT,
    MODEL_VOLUME_NAME,
    PYTORCH_INDEX_URL,
    REMOTE_ROOT,
    RUNTIME_REQUIREMENTS,
    _verify_model_report,
)
from tools.gate13_causal_return.modal.validate_constrained_modal_execution_authority import (
    MODEL_VOLUME_OBJECT_ID,
)
from tools.gate13_causal_return.modal.validate_modal_runtime import validate_modal_runtime
from tools.gate13_causal_return.phase2_common import read_json, sha256_bytes, sha256_file
from tools.gate13_causal_return.stepwise.compiler import (
    CAMPAIGN_FORWARD_CEILING,
    CAMPAIGN_SPEND_CEILING_USD,
    DEVELOPMENT_FORWARD_CEILING,
    DEVELOPMENT_SPEND_CEILING_USD,
    MODEL_REVISION,
    PRIOR_CUMULATIVE_FORWARD_COUNT,
    PRIOR_EXECUTION_ID,
    STARTING_COMMIT,
    TOKENIZER_REVISION,
    VARIANT_IDS,
    codebook_lookup,
    compile_track_b_collection_ledger,
    render_step_prompt,
    sha256_json,
    transition,
    validate_exact_tokenizer,
)
from tools.gate13_causal_return.stepwise.operator_qualification import (
    LAYER_SET,
    qualify_track_b,
)
from tools.gate13_causal_return.stepwise.runner import (
    JsonlJournal,
    run_development_variant,
    run_track_a_qualification,
    write_result,
)


APP_NAME = "gate13-stepwise-successor"
MODEL_REPOSITORY = "Qwen/Qwen3-8B"
DEVELOPMENT_VOLUME_NAME = "gate13-stepwise-successor-development"
QUALIFICATION_VOLUME_NAME = "gate13-stepwise-successor-qualification"
DEVELOPMENT_MOUNT = PurePosixPath("/development-results")
QUALIFICATION_MOUNT = PurePosixPath("/qualification-results")
REMOTE_PHASE2_DIR = REMOTE_ROOT / "analysis/gate13_causal_return/phase2"
REMOTE_SUCCESSOR_DIR = REMOTE_ROOT / "analysis/gate13_causal_return/successor"
AUTHORIZATION_FILENAME = "campaign_execution_authorization.json"
GPU_REQUEST = "L40S"
DEVELOPMENT_TIMEOUT_SECONDS = 3600
QUALIFICATION_TIMEOUT_SECONDS = 10_800


class StepwiseModalError(RuntimeError):
    """Fail-closed operational error for the successor adapter."""


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


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
        "schema_version": "gate13_stepwise_artifact_manifest_v1",
        "file_count": len(rows),
        "files": rows,
        "manifest_payload_sha256": sha256_json(rows),
    }


def image_definition_payload() -> dict[str, Any]:
    return {
        "schema_version": "gate13_stepwise_modal_image_definition_v1",
        "base_image": BASE_IMAGE,
        "base_image_manifest_digest": BASE_IMAGE_MANIFEST_DIGEST,
        "base_image_linux_amd64_digest": BASE_IMAGE_LINUX_AMD64_DIGEST,
        "python": "3.11.2",
        "requirements": list(RUNTIME_REQUIREMENTS),
        "pytorch_extra_index_url": PYTORCH_INDEX_URL,
        "gpu": GPU_REQUEST,
        "copied_roots": [
            "tools/gate13_causal_return",
            "analysis/gate13_causal_return/phase2",
            "analysis/gate13_causal_return/successor",
        ],
        "entrypoint_alias": {
            "source": "tools/gate13_causal_return/modal/modal_stepwise_successor.py",
            "destination": "/opt/gate13/modal_stepwise_successor.py"
        },
        "authorization_excluded": AUTHORIZATION_FILENAME,
        "automatic_retries": 0,
        "maximum_active_gpu_containers": 1,
    }


def image_definition_sha256() -> str:
    return sha256_json(image_definition_payload())


def _tensor_sha256(tensor: Any) -> str:
    value = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


class TorchForcedChoiceProbe:
    def __init__(self, torch_module: Any, tokenizer: Any, model: Any):
        self.torch = torch_module
        self.tokenizer = tokenizer
        self.model = model

    def _inputs(self, prompt: str) -> dict[str, Any]:
        encoded = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        )
        if isinstance(encoded, self.torch.Tensor):
            encoded = {
                "input_ids": encoded,
                "attention_mask": self.torch.ones_like(encoded),
            }
        inputs = {
            key: value.to("cuda:0")
            for key, value in dict(encoded).items()
            if key in {"input_ids", "attention_mask"}
        }
        if set(inputs) != {"input_ids", "attention_mask"}:
            raise StepwiseModalError("chat rendering did not yield exact input_ids and attention_mask")
        return inputs

    def __call__(self, prompt: str, candidates: tuple[str, str], metadata: Mapping[str, Any]) -> Mapping[str, Any]:
        inputs = self._inputs(prompt)
        token_ids = []
        for label in candidates:
            ids = self.tokenizer.encode(label, add_special_tokens=False)
            if len(ids) != 1:
                raise StepwiseModalError("forced-choice label ceased to be one token")
            token_ids.append(int(ids[0]))
        started = time.monotonic()
        with self.torch.inference_mode():
            output = self.model(
                **inputs,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
        elapsed = time.monotonic() - started
        logits = output.logits[0, -1, token_ids].float().detach().cpu().tolist()
        if logits[0] == logits[1]:
            raise StepwiseModalError("exact tie in frozen forced-choice logits")
        prediction = 0 if logits[0] > logits[1] else 1
        return {
            "predicted_label": candidates[prediction],
            "candidate_token_ids": token_ids,
            "candidate_logits": [float(value) for value in logits],
            "logit_margin_predicted_minus_alternative": float(abs(logits[0] - logits[1])),
            "input_ids_sha256": _tensor_sha256(inputs["input_ids"]),
            "attention_mask_sha256": _tensor_sha256(inputs["attention_mask"]),
            "input_token_count": int(inputs["input_ids"].shape[-1]),
            "forward_elapsed_seconds": elapsed,
            "readout": "RAW_NEXT_TOKEN_FORCED_CHOICE_LOGITS",
            "generate_called": False,
            "sampling": False,
        }

    def activation_call(
        self,
        prompt: str,
        candidates: tuple[str, str],
    ) -> tuple[dict[str, Any], dict[int, Any]]:
        inputs = self._inputs(prompt)
        token_ids = []
        for label in candidates:
            ids = self.tokenizer.encode(label, add_special_tokens=False)
            if len(ids) != 1:
                raise StepwiseModalError("activation-call choice label is not one token")
            token_ids.append(int(ids[0]))
        started = time.monotonic()
        with self.torch.inference_mode():
            output = self.model(
                **inputs,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        elapsed = time.monotonic() - started
        logits = output.logits[0, -1, token_ids].float().detach().cpu().tolist()
        if logits[0] == logits[1]:
            raise StepwiseModalError("exact tie in activation-call forced-choice logits")
        prediction = 0 if logits[0] > logits[1] else 1
        activations = {
            layer: output.hidden_states[layer][0, -1].float().detach().cpu().numpy()
            for layer in LAYER_SET
        }
        metadata = {
            "predicted_label": candidates[prediction],
            "candidate_token_ids": token_ids,
            "candidate_logits": [float(value) for value in logits],
            "input_ids_sha256": _tensor_sha256(inputs["input_ids"]),
            "attention_mask_sha256": _tensor_sha256(inputs["attention_mask"]),
            "input_token_count": int(inputs["input_ids"].shape[-1]),
            "forward_elapsed_seconds": elapsed,
            "readout": "RAW_NEXT_TOKEN_FORCED_CHOICE_LOGITS_WITH_FROZEN_HIDDEN_STATE_TAP",
        }
        return metadata, activations


def _load_runtime() -> tuple[Any, Any, Any, dict[str, Any]]:
    parent_lock = read_json(Path(REMOTE_PHASE2_DIR.as_posix()) / "phase2_a_lock.json")
    m0 = validate_modal_runtime(parent_lock)
    model_identity = _verify_model_report(read_json(Path(MODEL_ACQUISITION_REPORT.as_posix())), rehash=True)
    m0["model_volume_identity"] = model_identity
    if m0["status"] != "PASS":
        raise StepwiseModalError("M0_RUNTIME_MISMATCH")
    from tools.gate13_causal_return.track_a.phase2_runner import _load_exact_model

    torch_module, tokenizer, model = _load_exact_model(parent_lock)
    token_validation = validate_exact_tokenizer(tokenizer)
    m0["stepwise_tokenizer_validation"] = token_validation
    m0["offline_execution"] = {
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
        "local_files_only": True,
        "network_blocked": True,
    }
    return torch_module, tokenizer, model, m0


def _claim(output_dir: Path, identity: str, binding: Mapping[str, Any], volume: Any) -> dict[str, Any]:
    path = output_dir / "execution_claim.json"
    expected = {"execution_identity": identity, **dict(binding)}
    if path.exists():
        existing = read_json(path)
        if any(existing.get(key) != value for key, value in expected.items()):
            raise StepwiseModalError("execution identity is already bound differently")
        existing["container_start_count"] = int(existing.get("container_start_count", 1)) + 1
        existing.setdefault("resume_entries", []).append(utc_now())
        claim = existing
    else:
        claim = {
            "schema_version": "gate13_stepwise_execution_claim_v1",
            **expected,
            "claimed_at": utc_now(),
            "container_start_count": 1,
            "resume_entries": [],
        }
    _atomic_json(path, claim)
    volume.commit()
    return claim


def _development_table(result: Mapping[str, Any]) -> dict[str, Any]:
    a0 = result["STREAM-A0"]["metrics"]
    a1 = result["STREAM-A1"]["metrics"]
    a2 = result["STREAM-A2"]["metrics"]
    return {
        "variant_id": result["variant_id"],
        "selection_eligible": result["selection_eligible"],
        "one_step_accuracy": a0["one_step_accuracy"],
        "minimum_transition_cell_accuracy": a0["minimum_transition_cell_accuracy"],
        "self_fed_rollout_exact_accuracy": a0["self_fed_rollout_exact_accuracy"],
        "correct_minus_strongest_control": a1["correct_minus_strongest_control"],
        "visible_edit_immediate_successor_accuracy": a2["edited_immediate_successor_accuracy"],
        "marker_only_false_change_rate": a2["marker_only_false_change_rate"],
    }


def _collect_track_b(
    *,
    probe: TorchForcedChoiceProbe,
    variant_id: str,
    output_dir: Path,
    volume: Any,
) -> tuple[dict[str, Any], int]:
    import numpy as np

    ledger = compile_track_b_collection_ledger(variant_id)
    codebooks = codebook_lookup()
    attempt_path = output_dir / "track_b_activation_attempts.jsonl"
    response_path = output_dir / "track_b_activation_responses.jsonl"
    attempts = {row["forward_id"]: row for row in _load_jsonl(attempt_path)}
    responses = {row["forward_id"]: row for row in _load_jsonl(response_path)}
    if len(attempts) != len(_load_jsonl(attempt_path)) or len(responses) != len(_load_jsonl(response_path)):
        raise StepwiseModalError("duplicate Track B activation forward identity")
    activations: dict[str, dict[int, dict[str, list[Any]]]] = {
        half: {layer: {node: [] for node in ledger["nodes"]} for layer in LAYER_SET}
        for half in ("half_1", "half_2")
    }
    new_count = 0
    node_values = {
        "phase0_state0": (0, 0, False),
        "phase0_state1": (0, 1, False),
        "phase1_state0": (1, 0, False),
        "phase1_state1": (1, 1, False),
        "phase1_state1_broken": (1, 1, True),
    }
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
                    variant_id=variant_id,
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
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "candidate_labels": list(codebook.state_labels),
                    "target_state": transition(state, action),
                }
                if forward_id not in responses:
                    if forward_id in attempts:
                        raise StepwiseModalError(f"AMBIGUOUS_TRACK_B_FORWARD:{forward_id}")
                    _append_jsonl(attempt_path, binding)
                    attempts[forward_id] = binding
                    volume.commit()
                    meta, vectors = probe.activation_call(prompt, codebook.state_labels)
                    artifact = output_dir / "activations" / half_id / f"{forward_id}.npz"
                    artifact.parent.mkdir(parents=True, exist_ok=True)
                    temporary = artifact.with_suffix(".npz.tmp")
                    with temporary.open("wb") as handle:
                        np.savez_compressed(handle, **{f"layer_{layer}": vector for layer, vector in vectors.items()})
                    os.replace(temporary, artifact)
                    response = {
                        **binding,
                        **meta,
                        "activation_artifact": artifact.relative_to(output_dir).as_posix(),
                        "activation_artifact_sha256": sha256_file(artifact),
                    }
                    _append_jsonl(response_path, response)
                    responses[forward_id] = response
                    new_count += 1
                    volume.commit()
                response = responses[forward_id]
                if any(response.get(key) != value for key, value in binding.items()):
                    raise StepwiseModalError(f"Track B resume binding mismatch: {forward_id}")
                artifact = output_dir / str(response["activation_artifact"])
                if sha256_file(artifact) != response["activation_artifact_sha256"]:
                    raise StepwiseModalError(f"Track B activation artifact hash mismatch: {forward_id}")
                with np.load(artifact) as stored:
                    for layer in LAYER_SET:
                        activations[half_id][layer][str(node)].append(stored[f"layer_{layer}"].astype(np.float64))
    arrays = {
        half: {
            layer: {node: np.stack(rows, axis=0) for node, rows in nodes.items()}
            for layer, nodes in layers.items()
        }
        for half, layers in activations.items()
    }
    result = qualify_track_b(arrays)
    result["collection_ledger_sha256"] = ledger["sha256"]
    result["actual_new_forward_count"] = new_count
    result["total_activation_forward_count"] = len(responses)
    return result, new_count


if modal is not None:
    local_root = _repo_root()
    image = (
        modal.Image.from_registry(BASE_IMAGE, add_python=None)
        .run_commands(
            "python -m pip install --no-cache-dir --upgrade pip==26.2.1 setuptools==80.9.0 wheel==0.46.3"
        )
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
            local_root / "analysis/gate13_causal_return/phase2",
            "/opt/gate13/analysis/gate13_causal_return/phase2",
            copy=True,
            ignore=["**/__pycache__/**", "**/*.pyc"],
        )
        .add_local_dir(
            local_root / "analysis/gate13_causal_return/successor",
            "/opt/gate13/analysis/gate13_causal_return/successor",
            copy=True,
            ignore=[AUTHORIZATION_FILENAME, "**/__pycache__/**", "**/*.pyc"],
        )
        .add_local_file(
            local_root / "tools/gate13_causal_return/modal/modal_stepwise_successor.py",
            "/opt/gate13/modal_stepwise_successor.py",
            copy=True,
        )
        .workdir(REMOTE_ROOT)
        .env({"PYTHONPATH": str(REMOTE_ROOT), "PYTHONUNBUFFERED": "1"})
    )
    app = modal.App(APP_NAME)
    model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=False, version=2)
    development_volume = modal.Volume.from_name(DEVELOPMENT_VOLUME_NAME, create_if_missing=True, version=2)
    qualification_volume = modal.Volume.from_name(QUALIFICATION_VOLUME_NAME, create_if_missing=True, version=2)

    COMMON_ENV = {
        "HF_HOME": str(HF_HOME),
        "HF_HUB_CACHE": str(HF_CACHE),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }

    @app.function(
        image=image,
        volumes={str(MODEL_MOUNT): model_volume},
        env=COMMON_ENV,
        cpu=4.0,
        memory=16_384,
        retries=0,
        timeout=1800,
        max_containers=1,
        single_use_containers=True,
        include_source=False,
        block_network=True,
        name="validate_exact_stepwise_cpu",
    )
    def validate_exact_stepwise_cpu() -> dict[str, Any]:
        import platform

        import tokenizers
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config

        identity = _verify_model_report(
            read_json(Path(MODEL_ACQUISITION_REPORT.as_posix())), rehash=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_REPOSITORY,
            revision=TOKENIZER_REVISION,
            use_fast=True,
            local_files_only=True,
            trust_remote_code=False,
        )
        token_validation = validate_exact_tokenizer(tokenizer)
        codebook = next(iter(codebook_lookup("development").values()))
        prompt = render_step_prompt(
            variant_id=VARIANT_IDS[0],
            surface="STREAM-A0",
            codebook=codebook,
            current_state=0,
            action=1,
        )
        config = Qwen3Config(
            vocab_size=len(tokenizer),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=1024,
        )
        model = AutoModelForCausalLM.from_config(config).eval()
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        )
        model_inputs = {
            key: value.cpu()
            for key, value in dict(encoded).items()
            if key in {"input_ids", "attention_mask"}
        }
        candidate_ids = [
            tokenizer.encode(label, add_special_tokens=False)[0]
            for label in codebook.state_labels
        ]
        with torch.inference_mode():
            output = model(
                **model_inputs,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        logits = output.logits[0, -1, candidate_ids]
        if logits.shape != (2,) or len(output.hidden_states) != 2:
            raise StepwiseModalError("exact-package forced-choice integration failed")
        return {
            "schema_version": "gate13_stepwise_exact_cpu_validation_v1",
            "status": "PASS",
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda),
            "transformers": str(transformers.__version__),
            "tokenizers": str(tokenizers.__version__),
            "model_revision": MODEL_REVISION,
            "tokenizer_revision": TOKENIZER_REVISION,
            "model_volume_identity": identity,
            "token_validation": token_validation,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "candidate_token_ids": candidate_ids,
            "input_ids_sha256": _tensor_sha256(model_inputs["input_ids"]),
            "attention_mask_sha256": _tensor_sha256(model_inputs["attention_mask"]),
            "synthetic_random_model_forward_count": 1,
            "scientific_model_forward_count": 0,
            "scientific_weights_loaded": False,
            "gpu_allocated": False,
        }

    @app.function(
        image=image,
        volumes={str(MODEL_MOUNT): model_volume, str(DEVELOPMENT_MOUNT): development_volume},
        env=COMMON_ENV,
        gpu=GPU_REQUEST,
        cpu=GPU_CPU_CORES,
        memory=GPU_MEMORY_MIB,
        retries=0,
        timeout=DEVELOPMENT_TIMEOUT_SECONDS,
        max_containers=1,
        single_use_containers=True,
        include_source=False,
        block_network=True,
        name="run_stepwise_development",
    )
    def run_stepwise_development(development_identity: str, implementation_commit: str) -> dict[str, Any]:
        started = time.monotonic()
        output_dir = Path(DEVELOPMENT_MOUNT.as_posix()) / "executions" / development_identity
        output_dir.mkdir(parents=True, exist_ok=True)
        if (output_dir / "terminal_state.json").exists():
            return {**read_json(output_dir / "terminal_state.json"), "idempotent_retrieval": True}
        _claim(
            output_dir,
            development_identity,
            {
                "implementation_commit": implementation_commit,
                "starting_commit": STARTING_COMMIT,
                "prior_execution_identity": PRIOR_EXECUTION_ID,
                "prior_cumulative_forward_count": PRIOR_CUMULATIVE_FORWARD_COUNT,
            },
            development_volume,
        )
        try:
            torch_module, tokenizer, model, m0 = _load_runtime()
            _atomic_json(output_dir / "m0_runtime_report.json", m0)
            development_volume.commit()
            probe = TorchForcedChoiceProbe(torch_module, tokenizer, model)
            journal = JsonlJournal(output_dir / "development_ledger", probe, development_volume.commit)
            table = []
            selected = None
            for variant_id in VARIANT_IDS:
                if journal.total_attempt_count >= DEVELOPMENT_FORWARD_CEILING:
                    break
                result = run_development_variant(journal, variant_id)
                write_result(output_dir / f"{variant_id}_result.json", result)
                table.append(_development_table(result))
                _atomic_json(
                    output_dir / "development_result_table.json",
                    {"schema_version": "gate13_stepwise_development_table_v1", "rows": table},
                )
                with (output_dir / "development_change_log.md").open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(
                        f"- {utc_now()} — {variant_id}: "
                        f"{'SELECTED' if result['selection_eligible'] else 'REJECTED'}; "
                        f"cumulative forwards={journal.total_attempt_count}.\n"
                    )
                development_volume.commit()
                if result["selection_eligible"]:
                    selected = variant_id
                    break
            if journal.total_attempt_count > DEVELOPMENT_FORWARD_CEILING:
                raise StepwiseModalError("development forward ceiling exceeded")
            elapsed = time.monotonic() - started
            estimated = elapsed * GPU_TOTAL_RATE_USD_PER_SECOND
            if estimated > DEVELOPMENT_SPEND_CEILING_USD:
                raise StepwiseModalError("development spend ceiling exceeded")
            terminal = {
                "schema_version": "gate13_stepwise_development_terminal_v1",
                "status": "CANDIDATE_SELECTED" if selected else "NO_QUALIFIED_STEPWISE_SUBSTRATE",
                "selected_variant": selected,
                "variants_attempted": [row["variant_id"] for row in table],
                "development_forward_count": journal.total_attempt_count,
                "gpu_elapsed_seconds": elapsed,
                "estimated_modal_usage_usd": estimated,
                "old_candidate_unchanged": True,
                "qualification_data_opened": False,
            }
            _atomic_json(output_dir / "terminal_state.json", terminal)
            _atomic_json(output_dir / "artifact_manifest.json", _file_manifest(output_dir))
            development_volume.commit()
            return terminal
        except Exception as exc:
            blocker = {
                "schema_version": "gate13_stepwise_development_blocker_v1",
                "status": "DEVELOPMENT_IMPLEMENTATION_OR_INFRASTRUCTURE_BLOCKER",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "observed_at": utc_now(),
            }
            _atomic_json(output_dir / "blocker.json", blocker)
            _atomic_json(output_dir / "artifact_manifest.json", _file_manifest(output_dir))
            development_volume.commit()
            return blocker

    @app.function(
        image=image,
        volumes={str(MODEL_MOUNT): model_volume, str(QUALIFICATION_MOUNT): qualification_volume},
        env=COMMON_ENV,
        gpu=GPU_REQUEST,
        cpu=GPU_CPU_CORES,
        memory=GPU_MEMORY_MIB,
        retries=0,
        timeout=QUALIFICATION_TIMEOUT_SECONDS,
        max_containers=1,
        single_use_containers=True,
        include_source=False,
        block_network=True,
        name="run_stepwise_qualification",
    )
    def run_stepwise_qualification(authorization_text: str, authorization_sha256: str) -> dict[str, Any]:
        if sha256_bytes(authorization_text.encode("utf-8")) != authorization_sha256:
            raise StepwiseModalError("authorization byte SHA-256 mismatch")
        auth = json.loads(authorization_text)
        if not auth.get("execution_authorized"):
            raise StepwiseModalError("qualification execution is not authorized")
        execution_identity = str(auth["execution_identity"])
        output_dir = Path(QUALIFICATION_MOUNT.as_posix()) / "executions" / execution_identity
        output_dir.mkdir(parents=True, exist_ok=True)
        if (output_dir / "terminal_state.json").exists():
            return {**read_json(output_dir / "terminal_state.json"), "idempotent_retrieval": True}
        _claim(
            output_dir,
            execution_identity,
            {
                "authorization_sha256": authorization_sha256,
                "qualification_commit": auth["qualification_commit"],
                "track_a_lock_sha256": auth["track_a_lock_sha256"],
                "track_b_lock_sha256": auth["track_b_lock_sha256"],
            },
            qualification_volume,
        )
        started = time.monotonic()
        try:
            for filename, field in (
                ("stepwise_track_a_qualification_lock.json", "track_a_lock_sha256"),
                ("fresh_operator_qualification_lock.json", "track_b_lock_sha256"),
            ):
                actual = sha256_file(Path(REMOTE_SUCCESSOR_DIR.as_posix()) / filename)
                if actual != auth[field]:
                    raise StepwiseModalError(f"authority/hash mismatch: {filename}")
            if auth["model_volume"]["object_id"] != MODEL_VOLUME_OBJECT_ID:
                raise StepwiseModalError("model Volume object identity mismatch")
            torch_module, tokenizer, model, m0 = _load_runtime()
            _atomic_json(output_dir / "m0_runtime_report.json", m0)
            qualification_volume.commit()
            probe = TorchForcedChoiceProbe(torch_module, tokenizer, model)
            journal = JsonlJournal(output_dir / "track_a_ledger", probe, qualification_volume.commit)
            track_a = run_track_a_qualification(journal, str(auth["selected_variant"]))
            write_result(output_dir / "track_a_qualification_result.json", track_a)
            qualification_volume.commit()
            prior_dev_forwards = int(auth["development_accounting"]["forward_count"])
            if prior_dev_forwards + journal.total_attempt_count > CAMPAIGN_FORWARD_CEILING:
                raise StepwiseModalError("campaign forward ceiling exceeded during Track A")
            if track_a["terminal_track_a_status"] != "PASS":
                terminal_code = "T1_NO_QUALIFIED_STEPWISE_SUBSTRATE"
                track_b: Mapping[str, Any] = {"status": "UNOPENED_TRACK_A_FAIL"}
                track_b_new = 0
            else:
                track_b, track_b_new = _collect_track_b(
                    probe=probe,
                    variant_id=str(auth["selected_variant"]),
                    output_dir=output_dir,
                    volume=qualification_volume,
                )
                write_result(output_dir / "track_b_qualification_result.json", track_b)
                qualification_volume.commit()
                terminal_code = (
                    "T3_TRACK_A_AND_TRACK_B_QUALIFIED_READY_FOR_TRACK_C_DESIGN"
                    if track_b["status"] == "PASS"
                    else "T2_TRACK_A_STEPWISE_PASS_TRACK_B_FAIL"
                )
            qualification_forward_count = journal.total_attempt_count + int(
                track_b.get("total_activation_forward_count", 0)
            )
            campaign_forward_count = prior_dev_forwards + qualification_forward_count
            if campaign_forward_count > CAMPAIGN_FORWARD_CEILING:
                raise StepwiseModalError("campaign forward ceiling exceeded")
            elapsed = time.monotonic() - started
            estimated_qualification = elapsed * GPU_TOTAL_RATE_USD_PER_SECOND
            estimated_total = float(auth["development_accounting"]["estimated_modal_usage_usd"]) + estimated_qualification
            if estimated_total > CAMPAIGN_SPEND_CEILING_USD:
                raise StepwiseModalError("campaign Modal spend ceiling exceeded")
            terminal = {
                "schema_version": "gate13_stepwise_campaign_terminal_v1",
                "terminal_state": terminal_code,
                "execution_identity": execution_identity,
                "selected_variant": auth["selected_variant"],
                "development_forward_count": prior_dev_forwards,
                "qualification_forward_count": qualification_forward_count,
                "new_campaign_forward_count": campaign_forward_count,
                "prior_closed_candidate_forward_count": PRIOR_CUMULATIVE_FORWARD_COUNT,
                "qualification_gpu_elapsed_seconds": elapsed,
                "estimated_qualification_modal_usage_usd": estimated_qualification,
                "estimated_campaign_modal_usage_usd": estimated_total,
                "STREAM-A0": track_a["STREAM-A0"]["status"],
                "STREAM-A1": track_a["STREAM-A1"]["status"],
                "STREAM-A2": track_a["STREAM-A2"]["status"],
                "TRACK_B": track_b["status"],
                "TRACK_A_CONSTRAINED_REGISTER_V1": "CLOSED_A0_FAIL",
                "A3": "CLOSED",
                "TRACK_C": "CLOSED",
                "FORMAL_GATE13": "CLOSED",
                "ACTIVATION_EXTRACTION_OUTSIDE_FRESH_TRACK_B": "CLOSED",
                "mandatory_stop": True,
            }
            _atomic_json(output_dir / "terminal_state.json", terminal)
            _atomic_json(output_dir / "artifact_manifest.json", _file_manifest(output_dir))
            qualification_volume.commit()
            return terminal
        except Exception as exc:
            blocker = {
                "schema_version": "gate13_stepwise_campaign_terminal_v1",
                "terminal_state": "T4_TERMINAL_INFRASTRUCTURE_BLOCKER",
                "execution_identity": execution_identity,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "observed_at": utc_now(),
                "TRACK_C": "CLOSED",
                "mandatory_stop": True,
            }
            _atomic_json(output_dir / "terminal_state.json", blocker)
            _atomic_json(output_dir / "artifact_manifest.json", _file_manifest(output_dir))
            qualification_volume.commit()
            return blocker

    @app.local_entrypoint()
    def prepare_resources(mode: str = "development") -> None:
        if mode not in {"development", "qualification"}:
            raise StepwiseModalError("mode must be development or qualification")
        image.hydrate()
        model_volume.hydrate()
        volume = development_volume if mode == "development" else qualification_volume
        volume.hydrate()
        print(
            json.dumps(
                {
                    "schema_version": "gate13_stepwise_resource_identity_v1",
                    "mode": mode,
                    "app_name": APP_NAME,
                    "modal_image_object_id": image.object_id,
                    "modal_image_definition_sha256": image_definition_sha256(),
                    "model_volume": {"name": MODEL_VOLUME_NAME, "object_id": model_volume.object_id},
                    "result_volume": {"name": volume.name, "object_id": volume.object_id},
                    "gpu_allocated": False,
                },
                sort_keys=True,
            )
        )

    @app.local_entrypoint()
    def run_exact_validation(control_output: str) -> None:
        validation = validate_exact_stepwise_cpu.remote()
        control = {
            "schema_version": "gate13_stepwise_exact_cpu_control_v1",
            "modal_image_object_id": image.object_id,
            "modal_image_definition_sha256": image_definition_sha256(),
            "model_volume": {"name": MODEL_VOLUME_NAME, "object_id": model_volume.object_id},
            "validation": validation,
        }
        _atomic_json(Path(control_output).resolve(), control)
        print(json.dumps(control, sort_keys=True))

    @app.local_entrypoint()
    def run_development(development_identity: str, implementation_commit: str, control_output: str) -> None:
        result = run_stepwise_development.remote(development_identity, implementation_commit)
        control = {
            "schema_version": "gate13_stepwise_development_control_v1",
            "modal_image_object_id": image.object_id,
            "model_volume": {"name": MODEL_VOLUME_NAME, "object_id": model_volume.object_id},
            "result_volume": {"name": DEVELOPMENT_VOLUME_NAME, "object_id": development_volume.object_id},
            "result": result,
        }
        _atomic_json(Path(control_output).resolve(), control)
        print(json.dumps(control, sort_keys=True))

    @app.local_entrypoint()
    def run_qualification(authorization: str, control_output: str) -> None:
        path = Path(authorization).resolve()
        text = path.read_text(encoding="utf-8")
        digest = sha256_bytes(text.encode("utf-8"))
        result = run_stepwise_qualification.remote(text, digest)
        control = {
            "schema_version": "gate13_stepwise_qualification_control_v1",
            "authorization_sha256": digest,
            "modal_image_object_id": image.object_id,
            "model_volume": {"name": MODEL_VOLUME_NAME, "object_id": model_volume.object_id},
            "result_volume": {"name": QUALIFICATION_VOLUME_NAME, "object_id": qualification_volume.object_id},
            "result": result,
        }
        _atomic_json(Path(control_output).resolve(), control)
        print(json.dumps(control, sort_keys=True))

else:  # pragma: no cover
    app = None
    image = None
    model_volume = None
    development_volume = None
    qualification_volume = None
