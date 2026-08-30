"""Forward-zero validation of only the runtime fields frozen by Track A authority."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.gate13_causal_return.phase2_common import read_json
from tools.gate13_causal_return.track_a.phase2_runner import runtime_probe


EXPECTED_RUNTIME = {
    "python": "3.11.2",
    "pytorch": "2.7.1+cu126",
    "transformers": "5.15.0",
    "tokenizers": "0.22.2",
    "huggingface_hub": "1.27.0",
    "jinja2": "3.1.6",
    "accelerate": "1.14.0",
    "safetensors": "0.8.0",
    "cuda": "12.6",
    "driver": "580.95.05",
    "gpu": "NVIDIA L40S",
    "dtype": "bfloat16",
    "quantization": False,
}
EXPECTED_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
EXPECTED_HISTORICAL_READOUT = "raw next-token forced-choice logits; generate=false"
EXPECTED_GATE13_READOUT = (
    "deterministic greedy structured generation required by the register-trace instrument"
)
EXTRA_DISTRIBUTIONS = {
    "huggingface_hub": "huggingface-hub",
    "jinja2": "jinja2",
    "accelerate": "accelerate",
    "safetensors": "safetensors",
}


class ModalRuntimeValidationError(ValueError):
    """Raised when the frozen Modal runtime cannot be realized exactly."""


def _extra_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for field, distribution in EXTRA_DISTRIBUTIONS.items():
        try:
            versions[field] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[field] = "UNAVAILABLE"
    return versions


def validate_static_runtime_authority(lock: Mapping[str, Any]) -> dict[str, Any]:
    runtime = lock["runtime_binding"]
    prompt = lock["prompt_and_scoring"]
    checks: dict[str, bool] = {
        f"runtime.{field}": runtime.get(field) == expected
        for field, expected in EXPECTED_RUNTIME.items()
    }
    checks.update(
        {
            "runtime.model_repository": runtime.get("model_repository") == "Qwen/Qwen3-8B",
            "runtime.model_revision": runtime.get("model_revision") == EXPECTED_REVISION,
            "runtime.tokenizer_revision": runtime.get("tokenizer_revision") == EXPECTED_REVISION,
            "runtime.tokenizer_use_fast": runtime.get("tokenizer_use_fast") is True,
            "runtime.chat_template_sha256": runtime.get("chat_template_sha256")
            == "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
            "mode.enable_thinking": runtime.get("official_non_thinking_mode", {}).get(
                "enable_thinking"
            )
            is False,
            "mode.add_generation_prompt": runtime.get("official_non_thinking_mode", {}).get(
                "add_generation_prompt"
            )
            is True,
            "historical_readout": prompt.get("historical_local_mapping_readout_rechecked")
            == EXPECTED_HISTORICAL_READOUT,
            "gate13_task_readout": prompt.get("gate13_task_readout") == EXPECTED_GATE13_READOUT,
            "gate13_do_sample": prompt.get("do_sample") is False,
            "activation_extraction": prompt.get("activation_extraction") is False,
        }
    )
    failed = sorted(field for field, passed in checks.items() if not passed)
    if failed:
        raise ModalRuntimeValidationError(
            "frozen runtime authority mismatch: " + ", ".join(failed)
        )
    return {
        "model_repository": runtime["model_repository"],
        "model_revision": runtime["model_revision"],
        "tokenizer_revision": runtime["tokenizer_revision"],
        "historical_readout": prompt["historical_local_mapping_readout_rechecked"],
        "gate13_task_readout": prompt["gate13_task_readout"],
        "checks": checks,
    }


def validate_modal_runtime(
    lock: Mapping[str, Any],
    *,
    probe: Mapping[str, Any] | None = None,
    extra_versions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return a fail-closed M0 report without loading model weights or forwarding."""
    authority = validate_static_runtime_authority(lock)
    actual_probe = dict(probe or runtime_probe(lock))
    observed = dict(actual_probe.get("observed") or {})
    observed_extra = dict(extra_versions or _extra_versions())
    observed.update(observed_extra)
    checks = dict(actual_probe.get("checks") or {})
    checks.update(
        {
            "gpu_exact": observed.get("gpu") == EXPECTED_RUNTIME["gpu"],
            "huggingface_hub": observed.get("huggingface_hub")
            == EXPECTED_RUNTIME["huggingface_hub"],
            "jinja2": observed.get("jinja2") == EXPECTED_RUNTIME["jinja2"],
            "accelerate": observed.get("accelerate") == EXPECTED_RUNTIME["accelerate"],
            "safetensors": observed.get("safetensors") == EXPECTED_RUNTIME["safetensors"],
        }
    )
    passed = actual_probe.get("status") == "PASS" and all(checks.values())
    return {
        "schema_version": "gate13_track_a_modal_m0_runtime_v1",
        "status": "PASS" if passed else "MODAL_RUNTIME_MISMATCH",
        "model_forward_count": 0,
        "model_weights_loaded": False,
        "observed": observed,
        "checks": checks,
        "authority": authority,
        "not_validated_post_hoc": ["OS", "glibc", "cuDNN", "NCCL"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = validate_modal_runtime(read_json(args.lock))
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

