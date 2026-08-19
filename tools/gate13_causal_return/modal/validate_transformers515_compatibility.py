"""Forward-zero and tiny-model checks for the bounded Transformers 5.15 correction."""

from __future__ import annotations

import importlib.metadata
import platform
from pathlib import Path
from typing import Any, Mapping

from tools.gate13_causal_return.modal.modal_track_a import (
    MODEL_REVISION,
    _all_frozen_cases,
    _validate_m1_cases,
)
from tools.gate13_causal_return.phase2_common import (
    read_json,
    sha256_bytes,
    sha256_json,
)


EXPECTED_EXACT_PACKAGES = {
    "python": "3.11.2",
    "torch": "2.7.1+cu126",
    "transformers": "5.15.0",
    "tokenizers": "0.22.2",
}
FIRST_M1_CASE_ID = "a0-l12-y0-early-r0-S"


class Transformers515CompatibilityError(RuntimeError):
    """Fail-closed compatibility regression error."""


def _tensor_identity(tensor: Any) -> dict[str, Any]:
    contiguous = tensor.detach().cpu().contiguous()
    raw = contiguous.view(-1).numpy().tobytes(order="C")
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "content_sha256": sha256_bytes(raw),
    }


def exact_package_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": importlib.metadata.version("torch"),
        "transformers": importlib.metadata.version("transformers"),
        "tokenizers": importlib.metadata.version("tokenizers"),
    }


def run_tiny_qwen3_generate_integration() -> dict[str, Any]:
    """Exercise GenerationMixin with keyword-expanded synthetic BatchEncoding."""
    observed = exact_package_versions()
    if observed != EXPECTED_EXACT_PACKAGES:
        raise Transformers515CompatibilityError(
            f"exact package regression environment mismatch: {observed!r}"
        )

    import torch
    from transformers import BatchEncoding, Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(13)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = Qwen3ForCausalLM(config).eval()
    batch = BatchEncoding(
        {
            "input_ids": torch.tensor([[1, 3, 4]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }
    )
    before = {key: _tensor_identity(value) for key, value in batch.items()}
    with torch.inference_mode():
        output = model.generate(
            **batch,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=0,
        )
    after = {key: _tensor_identity(value) for key, value in batch.items()}
    if before != after:
        raise Transformers515CompatibilityError("synthetic BatchEncoding tensors mutated")
    if list(output.shape) != [1, 4]:
        raise Transformers515CompatibilityError(
            f"tiny Qwen3 generate returned unexpected shape: {list(output.shape)!r}"
        )
    return {
        "schema_version": "gate13_transformers515_tiny_qwen3_integration_v1",
        "status": "PASS",
        "exact_packages": observed,
        "model": "RANDOMLY_INITIALIZED_TINY_LOCAL_QWEN3",
        "scientific_weights_loaded": False,
        "scientific_case_output_generated": False,
        "scientific_forward_count": 0,
        "input_tensor_identity_before": before,
        "input_tensor_identity_after": after,
        "output_shape": list(output.shape),
    }


def prepare_first_m1_case_forward_zero(
    *, repo_root: Path, model_snapshot: Path
) -> dict[str, Any]:
    """Prepare the first frozen M1 call on CPU and stop before model.generate."""
    from transformers import AutoTokenizer

    manifest_path = (
        repo_root / "tools/gate13_causal_return/modal/m1_preflight_manifest.json"
    )
    manifest = read_json(manifest_path)
    stages, cases_by_id = _all_frozen_cases()
    selected = _validate_m1_cases(manifest, cases_by_id)
    case = selected[0]
    if case["case_id"] != FIRST_M1_CASE_ID:
        raise Transformers515CompatibilityError("first frozen M1 case identity drifted")

    binding = manifest["cases"][0]
    prompt = str(case["prompt"])
    raw_prompt_sha = sha256_bytes(prompt.encode("utf-8"))
    if raw_prompt_sha != binding["prompt_sha256"]:
        raise Transformers515CompatibilityError("first M1 raw prompt SHA drifted")
    max_new_tokens = max(32, len(str(case["expected_text"])) // 2 + 32)
    if max_new_tokens != int(binding["max_new_tokens"]):
        raise Transformers515CompatibilityError("first M1 max_new_tokens drifted")

    tokenizer = AutoTokenizer.from_pretrained(
        model_snapshot,
        revision=MODEL_REVISION,
        use_fast=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    messages = [{"role": "user", "content": prompt}]
    rendered_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    ).to("cpu")
    model_inputs = dict(encoded)
    generation_kwargs = {
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    collisions = set(model_inputs).intersection(generation_kwargs)
    if collisions:
        raise Transformers515CompatibilityError(
            "model inputs and frozen generation kwargs collide: "
            + ", ".join(sorted(collisions))
        )
    required_inputs = {"input_ids", "attention_mask"}
    if not required_inputs.issubset(model_inputs):
        raise Transformers515CompatibilityError("prepared input mapping lacks required tensors")

    return {
        "schema_version": "gate13_first_m1_forward_zero_preparation_v1",
        "status": "PASS_FORWARD_ZERO",
        "case_id": FIRST_M1_CASE_ID,
        "model_forward_count": 0,
        "raw_prompt_sha256": raw_prompt_sha,
        "raw_prompt_comparison": {
            "source": "m1_preflight_manifest.json cases[0].prompt_sha256",
            "expected": binding["prompt_sha256"],
            "status": "MATCH",
        },
        "rendered_prompt_sha256": sha256_bytes(rendered_prompt.encode("utf-8")),
        "input_ids": _tensor_identity(model_inputs["input_ids"]),
        "attention_mask": _tensor_identity(model_inputs["attention_mask"]),
        "generation_kwargs": generation_kwargs,
        "generation_kwargs_sha256": sha256_json(generation_kwargs),
        "comparison_authority": {
            "rendered_prompt_sha256": "AUTHORITY_GAP_NO_PRIOR_OR_FROZEN_HASH",
            "input_ids_content_sha256": "AUTHORITY_GAP_NO_PRIOR_OR_FROZEN_HASH",
            "attention_mask_content_sha256": "AUTHORITY_GAP_NO_PRIOR_OR_FROZEN_HASH",
            "generation_kwargs_sha256": "AUTHORITY_GAP_NO_PRIOR_OR_FROZEN_HASH",
        },
        "generate_invoked": False,
        "scientific_weights_loaded": False,
        "scientific_case_output_generated": False,
        "scientific_forward_count": 0,
        "frozen_a0_case_count": len(stages["A0"]),
    }


def run_exact_compatibility_regressions(
    *, repo_root: Path, model_snapshot: Path
) -> dict[str, Any]:
    integration = run_tiny_qwen3_generate_integration()
    frozen = prepare_first_m1_case_forward_zero(
        repo_root=repo_root, model_snapshot=model_snapshot
    )
    return {
        "schema_version": "gate13_transformers515_compatibility_regression_v1",
        "status": "PASS",
        "tiny_qwen3_integration": integration,
        "first_m1_forward_zero": frozen,
        "scientific_forward_count": 0,
    }
