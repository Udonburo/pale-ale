"""Fail-closed validation for the Track A constrained-channel derivative lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.gate13_causal_return.phase2_common import (
    git_status_porcelain,
    read_json,
    require_fields,
    sha256_file,
)

from .compile_constrained_channel import (
    compile_channel_manifest,
    compile_m1_manifest,
    validate_compiled_manifests,
)


PARENT_LOCK_SHA256 = "9c4b94b5199c3d355e8707798ba9bc1797aa2d690762b226f57bec63742215fa"
DECISION_SHA256 = "636b194088c52bfd2547f885b24c0ab504d2c07c1c9f88ddc604aba6565ccbd2"
CHANNEL_MANIFEST_SHA256 = "d51460e337d10067ae763d0af2b88ab02464978a8ed7126ed6659204092b947f"
M1_MANIFEST_SHA256 = "e4a7a544c7ee9caec6d1eced59be005c09acaba9d11db39c37fe3d279b958990"

UNCHANGED_SOURCE_HASHES = {
    "compile_register_cases.py": "e858c344dfb6f45cad11bc26fc390b99b1c5ab42a12fd193143a7c5598706b84",
    "compile_phase2_cases.py": "141d0547949f7aae268f39c548d8c6e19239bda6fda2f32ce3d13990e41b7ebb",
    "render_register_cases.py": "71f18094608135b97899969c5ae8795ccd7bda8ff04ee391f5b7d9d6a1333ee7",
    "parse_register_output.py": "d2d826c3d433c2a3ef00aca6bda9a51a8d0f6e5388a5dea282b0c10091ea8686",
    "parse_phase2_output.py": "87852e4c40d9207a2fc4aba575b0234d2f66cd42f1a181b48a05111877fec726",
    "oracle.py": "586480644c29a3e55fb2959e27e624bd3a50a82e91eb5043f7448cf56795553b",
    "evaluate_phase2.py": "fec29862931b8088770534c7af24237075afc85c92d787e8f282d237475aee60",
}
UNCHANGED_MANIFEST_HASHES = {
    "track_a_a0_extension_manifest.json": "7f99a653b4a96c4d6fbbf2a61d640c9a3971c45344b52b6bbeaf81b829ebe7f4",
    "track_a_a1_manifest.json": "eae8f1abd328c2df27aa1fb80a9b1f0d98683546149964f4d6825da29a93b400",
    "track_a_a2_manifest.json": "70a011e9e7629e5ad5dc3adff31c5304bc8791396220c04b96f1827a1dd440b5",
}
EXPECTED_STATE = {
    "TRACK_A_FREE_GENERATION_CHANNEL": "TERMINATED_INSTRUMENT_CHANNEL_INADEQUATE",
    "TRACK_A_SCIENTIFIC_QUESTION": "OPEN",
    "TRACK_A_CONSTRAINED_REGISTER_CHANNEL": "AUTHORIZED_FOR_BOUNDED_REDESIGN_AND_EXECUTION",
    "A0": "UNOPENED",
    "A1": "UNOPENED",
    "A2": "UNOPENED",
}
EXPECTED_CLOSED = {
    "B2A_HISTORICAL_12RUN": "TERMINATED_SUBSTRATE_INADEQUATE",
    "TRACK_B_SCIENTIFIC_QUESTION": "OPEN",
    "B2B_FRESH_SUBSTRATE": "RESERVED_NOT_AUTHORIZED",
    "B2B_EXECUTION": "NOT_AUTHORIZED",
    "A3": "CLOSED",
    "TRACK_C": "CLOSED",
    "FORMAL_GATE13": "CLOSED",
    "ACTIVATION_EXTRACTION": "CLOSED",
    "PUBLIC_README_OR_GATE_MAP_EDITS": "FORBIDDEN",
}


class ConstrainedLockError(ValueError):
    """Raised when constrained-channel authority is ambiguous or drifted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConstrainedLockError(message)


def _validate_runtime(parent: Mapping[str, Any], lock: Mapping[str, Any]) -> None:
    parent_runtime = parent["runtime_binding"]
    constrained = lock["runtime_binding"]
    expected = {
        "model_repository": parent_runtime["model_repository"],
        "model_revision": parent_runtime["model_revision"],
        "tokenizer_revision": parent_runtime["tokenizer_revision"],
        "chat_template_sha256": parent_runtime["chat_template_sha256"],
        "enable_thinking": parent_runtime["official_non_thinking_mode"]["enable_thinking"],
        "add_generation_prompt": parent_runtime["official_non_thinking_mode"]["add_generation_prompt"],
        "python": parent_runtime["python"],
        "pytorch": parent_runtime["pytorch"],
        "transformers": parent_runtime["transformers"],
        "tokenizers": parent_runtime["tokenizers"],
        "cuda": parent_runtime["cuda"],
        "driver": parent_runtime["driver"],
        "gpu": parent_runtime["gpu"],
        "dtype": parent_runtime["dtype"],
        "quantization": parent_runtime["quantization"],
        "local_files_only": parent_runtime["local_files_only"],
        "trust_remote_code": parent_runtime["trust_remote_code"],
    }
    _require(constrained == expected, "constrained runtime differs from parent authority")


def validate_constrained_channel_lock(
    *, phase2_dir: Path, require_clean: bool = True
) -> dict[str, Any]:
    phase2_dir = phase2_dir.resolve()
    repo_root = phase2_dir.parents[2]
    lock_path = phase2_dir / "phase2_a_constrained_channel_lock.json"
    parent_path = phase2_dir / "phase2_a_lock.json"
    lock = read_json(lock_path)
    parent = read_json(parent_path)
    require_fields(
        lock,
        (
            "schema_version",
            "authority_status",
            "state",
            "parent_authority",
            "decision_record",
            "immutable_prior_executions",
            "sole_changed_surface",
            "constrained_decoding",
            "unchanged_scientific_surfaces",
            "artifacts",
            "mechanical_proof_requirement",
            "runtime_binding",
            "forward_accounting",
            "resource_accounting",
            "scientific_interpretation_boundary",
            "closed_state",
        ),
        context="constrained-channel lock",
    )
    _require(
        lock["schema_version"] == "gate13_phase2_a_constrained_channel_lock_v1",
        "constrained-channel schema mismatch",
    )
    _require(
        lock["authority_status"] == "AUTHORIZED_FOR_BOUNDED_REDESIGN_AND_EXECUTION",
        "constrained-channel execution is not authorized",
    )
    _require(sha256_file(parent_path) == PARENT_LOCK_SHA256, "parent Track A lock drifted")
    _require(
        lock["parent_authority"]["parent_lock_sha256"] == PARENT_LOCK_SHA256,
        "parent lock binding mismatch",
    )
    decision_path = phase2_dir / str(lock["decision_record"]["path"])
    _require(sha256_file(decision_path) == DECISION_SHA256, "decision record drifted")
    _require(lock["decision_record"]["sha256"] == DECISION_SHA256, "decision SHA binding mismatch")
    for field, expected in EXPECTED_STATE.items():
        _require(lock["state"].get(field) == expected, f"state mismatch: {field}")

    channel = lock["constrained_decoding"]
    _require(channel["method"] == "CANONICAL_TOKEN_FINITE_STATE_PREFIX_CONSTRAINT", "channel method mismatch")
    _require(channel["transformers_interface"] == "prefix_allowed_tokens_fn", "Transformers constraint interface mismatch")
    _require(channel["semantic_slot_policy"].startswith("both exact tokenizer tokens for 0 and 1"), "both semantic branches are not fixed")
    _require(
        channel["output_length_policy"]
        == "exact canonical syntax content token count plus one terminal EOS",
        "constrained endpoint length policy mismatch",
    )
    for forbidden in (
        "oracle_access",
        "xor_transition_filtering",
        "answer_equals_final_register_filtering",
        "internal_consistency_filtering",
        "truth_conditioning",
        "complete_output_reranking",
    ):
        _require(channel.get(forbidden) is False, f"forbidden constraint leakage: {forbidden}")

    source_dir = repo_root / "tools/gate13_causal_return/track_a"
    for name, expected in UNCHANGED_SOURCE_HASHES.items():
        _require(sha256_file(source_dir / name) == expected, f"unchanged source drifted: {name}")
        _require(
            lock["unchanged_scientific_surfaces"]["source_hashes"].get(name) == expected,
            f"unchanged source lock binding drifted: {name}",
        )
    for name, expected in UNCHANGED_MANIFEST_HASHES.items():
        _require(sha256_file(phase2_dir / name) == expected, f"unchanged manifest drifted: {name}")
        _require(
            lock["unchanged_scientific_surfaces"]["manifest_hashes"].get(name) == expected,
            f"unchanged manifest lock binding drifted: {name}",
        )

    channel_path = phase2_dir / lock["artifacts"]["channel_manifest"]["path"]
    m1_path = repo_root / lock["artifacts"]["m1_manifest"]["repository_path"]
    _require(sha256_file(channel_path) == CHANNEL_MANIFEST_SHA256, "channel manifest file drifted")
    _require(sha256_file(m1_path) == M1_MANIFEST_SHA256, "M1 manifest file drifted")
    _require(lock["artifacts"]["channel_manifest"]["sha256"] == CHANNEL_MANIFEST_SHA256, "channel manifest SHA binding mismatch")
    _require(lock["artifacts"]["m1_manifest"]["sha256"] == M1_MANIFEST_SHA256, "M1 manifest SHA binding mismatch")
    channel_manifest = read_json(channel_path)
    m1_manifest = read_json(m1_path)
    validate_compiled_manifests(channel_manifest, m1_manifest)
    _require(channel_manifest == compile_channel_manifest(phase2_dir=phase2_dir), "channel manifest is not deterministic")
    _require(m1_manifest == compile_m1_manifest(channel_manifest), "M1 manifest is not deterministic")

    _validate_runtime(parent, lock)
    forward = lock["forward_accounting"]
    _require(int(forward["cumulative_ceiling"]) == 600, "cumulative forward ceiling mismatch")
    _require(int(forward["prior_consumed_count"]) == 5, "prior forward count mismatch")
    _require(int(forward["maximum_additional_count"]) == 595, "remaining forward count mismatch")
    _require(int(forward["prior_consumed_count"]) + int(forward["maximum_additional_count"]) == 600, "forward arithmetic mismatch")
    expected_new = int(forward["m1_development_forecast"]) + int(forward["A0_forecast"]) + int(forward["A1_forecast"]) + int(forward["A2_forecast"])
    _require(expected_new == int(forward["new_variant_maximum_if_all_stages_open"]) == 508, "new variant forward forecast mismatch")
    _require(int(forward["cumulative_maximum_if_all_stages_open"]) == 513, "cumulative forecast mismatch")
    resource = lock["resource_accounting"]
    _require(float(resource["cumulative_modal_spend_ceiling_usd"]) == 25.0, "spend ceiling mismatch")
    _require(resource["modal_automatic_retries"] == 0, "automatic retries are not zero")
    _require(resource["maximum_active_gpu_containers"] == 1, "GPU container maximum mismatch")
    for field, expected in EXPECTED_CLOSED.items():
        _require(lock["closed_state"].get(field) == expected, f"closed state mismatch: {field}")
    if require_clean:
        _require(git_status_porcelain(cwd=repo_root) == "", "Git state is not clean")
    return {
        "schema_version": "gate13_track_a_constrained_lock_validation_v1",
        "status": "PASS",
        "execution_authorized": True,
        "lock_sha256": sha256_file(lock_path),
        "parent_lock_sha256": PARENT_LOCK_SHA256,
        "channel_manifest_sha256": CHANNEL_MANIFEST_SHA256,
        "m1_manifest_sha256": M1_MANIFEST_SHA256,
        "prior_forward_count": 5,
        "maximum_additional_forwards": 595,
        "closed_state": dict(lock["closed_state"]),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-dir", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = validate_constrained_channel_lock(
            phase2_dir=args.phase2_dir,
            require_clean=not args.allow_dirty,
        )
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
