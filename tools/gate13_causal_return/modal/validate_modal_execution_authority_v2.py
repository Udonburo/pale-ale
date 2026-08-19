"""Validate the one-time post-blocker Track A Modal execution authority."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.gate13_causal_return.modal.validate_modal_execution_authority import (
    B2B_FRESH_SUBSTRATE_RESERVATION_SHA256,
    MODAL_PLAN_SHA256,
    PHASE2_A_LOCK_SHA256,
    STARTING_COMMIT,
    TRACK_B_POST_PHASE2_DECISION_SHA256,
    _validate_m1_manifest,
)
from tools.gate13_causal_return.phase2_common import (
    REPO_ROOT,
    git_status_porcelain,
    read_json,
    require_fields,
    require_sha256,
    sha256_file,
    sha256_json,
)
from tools.gate13_causal_return.validate_phase2_locks import validate_phase2_locks


PRIOR_EXECUTION_IDENTITY = "aca4b3c1-de70-4d75-a715-16d351f3f6da"
PRIOR_TERMINAL_STATUS = "SCIENTIFIC_RUNNER_BLOCKER"
PRIOR_ADAPTER_COMMIT = "88caf2c4706ad0e93c0baca108a9c5963430020d"
PRIOR_AUTHORIZATION_COMMIT = "0928fd4278570f40d106a50fe0785997afdf98bb"
PRIOR_AUTHORIZATION_SHA256 = (
    "e4aca50fccc54a8acf29aee77b90f8bb0a4bdff9c59fa2bff97040b1e01f6e9a"
)
PRIOR_BLOCKER_SHA256 = (
    "99e818622ff92f3955ff5cd1654551c80502accdd4644059ba5ae408880a0b0f"
)
CORRECTION_COMMIT = "84d6161154eaf0d317514d487f4f16ac98c56e6a"
OLD_RUNNER_SHA256 = "2c1401ac0077aafe3d4e14a8636c2c24ef6d8ecc8f52d6abe8ee072d5961d9f8"
NEW_RUNNER_SHA256 = "a41dfbdba970afdbf91fc3d7bfd1d13b1acfd2d875e10b69218019fcac08e9c4"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
MODEL_VOLUME_NAME = "gate13-track-a-qwen3-8b-b968826-model"
MODEL_VOLUME_OBJECT_ID = "vo-qcBxnmomu5avCRoacF7eKW"
RESULT_VOLUME_NAME = "gate13-track-a-transformers515-v2-results"
FROZEN_PROGRAM_FORWARD_CEILING = 600
PRIOR_FORWARD_COUNT = 1
NEW_MAXIMUM_FORWARDS = 599
CUMULATIVE_MODAL_SPEND_CEILING_USD = 25.0
PRIOR_MODAL_SPEND_USD = 0.05812777684
NEW_MODAL_SPEND_CEILING_USD = CUMULATIVE_MODAL_SPEND_CEILING_USD - PRIOR_MODAL_SPEND_USD
CUMULATIVE_GPU_WALL_CEILING_SECONDS = 36_000.0
PRIOR_GPU_WALL_SECONDS = 82.257
NEW_GPU_WALL_CEILING_SECONDS = CUMULATIVE_GPU_WALL_CEILING_SECONDS - PRIOR_GPU_WALL_SECONDS

EXPECTED_CASE_MANIFEST_HASHES = {
    "m1_preflight_manifest_sha256": "03920024a4b173cbd08426c23b38ab0fb97a8a14334a1898e1eeeff65cf15c8c",
    "a0_extension_manifest_sha256": "7f99a653b4a96c4d6fbbf2a61d640c9a3971c45344b52b6bbeaf81b829ebe7f4",
    "a1_manifest_sha256": "eae8f1abd328c2df27aa1fb80a9b1f0d98683546149964f4d6825da29a93b400",
    "a2_manifest_sha256": "70a011e9e7629e5ad5dc3adff31c5304bc8791396220c04b96f1827a1dd440b5",
}
EXPECTED_MODEL_IDENTITY = {
    "file_count": 15,
    "total_bytes": 16_397_461_266,
    "model_directory_identity_sha256": "935acddbddad11307d52408e0e0125d4fe328ed2ab9e3af2d4bc18299f5ece14",
    "config_sha256": "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
    "tokenizer_config_sha256": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "tokenizer_json_sha256": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "weight_index_sha256": "f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc",
}
EXPECTED_CLOSED = {
    "B2A_HISTORICAL_12RUN": "TERMINATED_SUBSTRATE_INADEQUATE",
    "TRACK_B_SCIENTIFIC_QUESTION": "OPEN",
    "B2B_FRESH_SUBSTRATE": "RESERVED_NOT_AUTHORIZED",
    "A3": "CLOSED",
    "TRACK_C": "CLOSED",
    "FORMAL_GATE13": "CLOSED",
    "ACTIVATION_EXTRACTION": "CLOSED",
    "B2B_EXECUTION": "NOT_AUTHORIZED",
    "B2A_REPAIR": "FORBIDDEN",
    "PUBLIC_README_OR_GATE_MAP_EDITS": "FORBIDDEN",
}


class ModalExecutionAuthorityV2Error(ValueError):
    """Raised on any v2 operational authority ambiguity or drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ModalExecutionAuthorityV2Error(message)


def _commit_exists(commit: str, repo_root: Path) -> bool:
    return (
        subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=repo_root,
            capture_output=True,
        ).returncode
        == 0
    )


def _is_ancestor(ancestor: str, descendant: str, repo_root: Path) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=repo_root,
            capture_output=True,
        ).returncode
        == 0
    )


def _validate_git_binding(auth: Mapping[str, Any], repo_root: Path) -> None:
    runner_commit = str(auth["new_runner_commit"])
    adapter_commit = str(auth["adapter_commit"])
    for commit in (STARTING_COMMIT, runner_commit, adapter_commit, PRIOR_AUTHORIZATION_COMMIT):
        _require(_commit_exists(commit, repo_root), f"bound commit is absent: {commit}")
    _require(runner_commit == CORRECTION_COMMIT, "runner correction commit mismatch")
    _require(_is_ancestor(STARTING_COMMIT, runner_commit, repo_root), "runner commit ancestry mismatch")
    _require(_is_ancestor(runner_commit, adapter_commit, repo_root), "adapter predates runner correction")

    runner_path = "tools/gate13_causal_return/track_a/phase2_runner.py"
    _require(
        subprocess.run(
            ["git", "diff", "--quiet", runner_commit, "HEAD", "--", runner_path],
            cwd=repo_root,
        ).returncode
        == 0,
        "scientific runner changed after correction commit",
    )
    _require(
        subprocess.run(
            [
                "git",
                "diff",
                "--quiet",
                PRIOR_AUTHORIZATION_COMMIT,
                "HEAD",
                "--",
                "tools/gate13_causal_return/track_a",
                ":(exclude)tools/gate13_causal_return/track_a/phase2_runner.py",
                ":(exclude)tools/gate13_causal_return/track_a/tests",
            ],
            cwd=repo_root,
        ).returncode
        == 0,
        "scientific Track A surface outside the authorized call site changed",
    )
    _require(
        subprocess.run(
            [
                "git",
                "diff",
                "--quiet",
                adapter_commit,
                "HEAD",
                "--",
                "tools/gate13_causal_return/modal",
            ],
            cwd=repo_root,
        ).returncode
        == 0,
        "v2 adapter changed after its bound commit",
    )
    _require(git_status_porcelain(cwd=repo_root) == "", "Git state is not clean")


def _validate_close(value: object, expected: float, field: str) -> None:
    _require(math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-9), f"{field} mismatch")


def validate_modal_execution_authority_v2(
    *,
    authorization_path: Path,
    phase2_dir: Path,
    m1_manifest_path: Path,
    repo_root: Path = REPO_ROOT,
    verify_git: bool = True,
) -> dict[str, Any]:
    auth = read_json(authorization_path)
    require_fields(
        auth,
        (
            "schema_version",
            "execution_authorized",
            "operational_scope",
            "starting_commit",
            "new_runner_commit",
            "adapter_commit",
            "scientific_runner_sha256",
            "compatibility_correction_sha256",
            "phase2_a_lock_sha256",
            "phase2_a_modal_realization_plan_sha256",
            "case_manifest_hashes",
            "model_revision",
            "tokenizer_revision",
            "modal_image_object_id",
            "modal_image_definition",
            "modal_image_definition_sha256",
            "base_image_manifest_digest",
            "gpu_type",
            "forward_accounting",
            "spend_accounting",
            "gpu_wall_accounting",
            "prior_execution",
            "closed_surfaces",
            "execution_identity",
            "model_volume",
            "result_volume",
            "m1_case_ids",
            "modal_automatic_retries",
            "maximum_active_gpu_containers",
        ),
        context="Modal v2 execution authorization",
    )
    _require(
        auth["schema_version"] == "gate13_track_a_modal_execution_authorization_v2",
        "v2 authorization schema mismatch",
    )
    _require(auth["execution_authorized"] is True, "execution_authorized is not true")
    _require(auth["operational_scope"] == "TRACK_A_ONLY", "operational scope mismatch")
    _require(auth["starting_commit"] == STARTING_COMMIT, "starting commit mismatch")
    _require(auth["new_runner_commit"] == CORRECTION_COMMIT, "runner commit mismatch")
    _require(auth["scientific_runner_sha256"] == NEW_RUNNER_SHA256, "runner SHA mismatch")
    _require(sha256_file(repo_root / "tools/gate13_causal_return/track_a/phase2_runner.py") == NEW_RUNNER_SHA256, "runner file drifted")
    correction_path = phase2_dir / "track_a_transformers515_compatibility_correction.json"
    _require(sha256_file(correction_path) == auth["compatibility_correction_sha256"], "correction record SHA mismatch")
    correction = read_json(correction_path)
    _require(correction.get("SCIENTIFIC_RESULT") == "NONE", "correction record scientific result mismatch")
    _require(correction.get("CORRECTION_CLASS") == "INTERFACE_COMPATIBILITY_ONLY", "correction class mismatch")
    tests = correction.get("regression_tests") or {}
    _require(tests.get("exact_package_tiny_qwen3_generation_mixin") == "PASS_EXACT_CPU_MODAL", "exact integration did not pass")
    _require(tests.get("first_m1_forward_zero_preparation") == "PASS_FORWARD_ZERO_WITH_AUTHORITY_GAPS_REPORTED", "frozen case preparation did not pass")

    _require(auth["phase2_a_lock_sha256"] == PHASE2_A_LOCK_SHA256, "A lock SHA mismatch")
    _require(auth["phase2_a_modal_realization_plan_sha256"] == MODAL_PLAN_SHA256, "Modal plan SHA mismatch")
    _require(auth["case_manifest_hashes"] == EXPECTED_CASE_MANIFEST_HASHES, "case manifest binding mismatch")
    actual_case_hashes = {
        "m1_preflight_manifest_sha256": sha256_file(m1_manifest_path),
        "a0_extension_manifest_sha256": sha256_file(phase2_dir / "track_a_a0_extension_manifest.json"),
        "a1_manifest_sha256": sha256_file(phase2_dir / "track_a_a1_manifest.json"),
        "a2_manifest_sha256": sha256_file(phase2_dir / "track_a_a2_manifest.json"),
    }
    _require(actual_case_hashes == EXPECTED_CASE_MANIFEST_HASHES, "case manifest file drift")
    _require(sha256_file(phase2_dir / "phase2_a_modal_realization_plan.json") == MODAL_PLAN_SHA256, "Modal plan file drift")
    _require(sha256_file(phase2_dir / "TRACK_B_POST_PHASE2_DECISION.md") == TRACK_B_POST_PHASE2_DECISION_SHA256, "Track B decision drift")
    _require(sha256_file(phase2_dir / "b2b_fresh_substrate_reservation.json") == B2B_FRESH_SUBSTRATE_RESERVATION_SHA256, "B2b reservation drift")

    prior = auth["prior_execution"]
    expected_prior = {
        "execution_identity": PRIOR_EXECUTION_IDENTITY,
        "terminal_status": PRIOR_TERMINAL_STATUS,
        "adapter_commit": PRIOR_ADAPTER_COMMIT,
        "authorization_commit": PRIOR_AUTHORIZATION_COMMIT,
        "authorization_sha256": PRIOR_AUTHORIZATION_SHA256,
        "blocker_artifact_sha256": PRIOR_BLOCKER_SHA256,
        "model_forward_count": PRIOR_FORWARD_COUNT,
        "model_response_count": 0,
        "a0": "UNOPENED",
        "a1": "UNOPENED",
        "a2": "UNOPENED",
    }
    _require(prior == expected_prior, "prior immutable execution binding mismatch")

    forward = auth["forward_accounting"]
    _require(int(forward["frozen_program_ceiling"]) == FROZEN_PROGRAM_FORWARD_CEILING, "forward ceiling mismatch")
    _require(int(forward["prior_consumed_count"]) == PRIOR_FORWARD_COUNT, "prior forward count mismatch")
    _require(int(forward["new_maximum_forwards"]) == NEW_MAXIMUM_FORWARDS, "new forward limit mismatch")
    _require(int(forward["prior_consumed_count"]) + int(forward["new_maximum_forwards"]) == FROZEN_PROGRAM_FORWARD_CEILING, "cumulative forward arithmetic mismatch")
    spend = auth["spend_accounting"]
    _validate_close(spend["cumulative_ceiling_usd"], CUMULATIVE_MODAL_SPEND_CEILING_USD, "cumulative spend ceiling")
    _validate_close(spend["prior_measured_usage_usd"], PRIOR_MODAL_SPEND_USD, "prior spend")
    _validate_close(spend["new_maximum_usage_usd"], NEW_MODAL_SPEND_CEILING_USD, "new spend limit")
    wall = auth["gpu_wall_accounting"]
    _validate_close(wall["cumulative_ceiling_seconds"], CUMULATIVE_GPU_WALL_CEILING_SECONDS, "cumulative GPU wall ceiling")
    _validate_close(wall["prior_l40s_seconds"], PRIOR_GPU_WALL_SECONDS, "prior GPU wall")
    _validate_close(wall["new_maximum_l40s_seconds"], NEW_GPU_WALL_CEILING_SECONDS, "new GPU wall limit")

    _require(auth["model_revision"] == MODEL_REVISION, "model revision mismatch")
    _require(auth["tokenizer_revision"] == MODEL_REVISION, "tokenizer revision mismatch")
    _require(auth["model_volume"]["name"] == MODEL_VOLUME_NAME, "model Volume name mismatch")
    _require(auth["model_volume"]["object_id"] == MODEL_VOLUME_OBJECT_ID, "model Volume object mismatch")
    _require(auth["model_volume"]["identity"] == EXPECTED_MODEL_IDENTITY, "model identity binding mismatch")
    _require(auth["result_volume"]["name"] == RESULT_VOLUME_NAME, "result Volume name mismatch")
    _require(str(auth["result_volume"].get("object_id") or "").startswith("vo-"), "result Volume object ID invalid")
    _require(auth["result_volume"]["name"] != MODEL_VOLUME_NAME, "model/result Volume alias")

    _require(str(auth["modal_image_object_id"]).startswith("im-"), "Modal Image ID invalid")
    require_sha256(auth["modal_image_definition_sha256"], field="modal_image_definition_sha256")
    _require(sha256_json(auth["modal_image_definition"]) == auth["modal_image_definition_sha256"], "image definition SHA mismatch")
    image_definition = auth["modal_image_definition"]
    _require(image_definition.get("schema_version") == "gate13_track_a_modal_image_definition_v1", "image definition schema mismatch")
    _require(image_definition.get("python") == "3.11.2", "image Python mismatch")
    requirements = set(image_definition.get("requirements") or [])
    for requirement in ("torch==2.7.1+cu126", "transformers==5.15.0", "tokenizers==0.22.2"):
        _require(requirement in requirements, f"image requirement mismatch: {requirement}")
    digest = str(auth["base_image_manifest_digest"])
    _require(image_definition.get("base_image_manifest_digest") == digest, "base image digest mismatch")
    _require(auth["gpu_type"] == "NVIDIA L40S", "GPU type mismatch")
    _require(auth["modal_automatic_retries"] == 0, "Modal automatic retries are not zero")
    _require(auth["maximum_active_gpu_containers"] == 1, "GPU container limit mismatch")
    for field, expected in EXPECTED_CLOSED.items():
        _require(auth["closed_surfaces"].get(field) == expected, f"closed surface mismatch: {field}")
    try:
        uuid.UUID(str(auth["execution_identity"]))
    except ValueError as exc:
        raise ModalExecutionAuthorityV2Error("execution identity is not a UUID") from exc
    _require(str(auth["execution_identity"]) != PRIOR_EXECUTION_IDENTITY, "fresh execution identity reused prior ID")

    lock_validation = validate_phase2_locks(phase2_dir=phase2_dir, require_clean=False, verify_git=False)
    _require(lock_validation["status"] == "PASS", "frozen lock validation failed")
    _require(lock_validation["phase2_a_lock_sha256"] == PHASE2_A_LOCK_SHA256, "frozen lock file drifted")
    m1 = _validate_m1_manifest(m1_manifest_path)
    _require([row["case_id"] for row in m1["cases"]] == auth["m1_case_ids"], "M1 case IDs mismatch")
    _require(auth["m1_case_ids"][0] == "a0-l12-y0-early-r0-S", "first M1 case mismatch")

    if verify_git:
        _validate_git_binding(auth, repo_root)

    return {
        "schema_version": "gate13_track_a_modal_execution_authority_validation_v2",
        "status": "PASS",
        "operational_execution_authorized": True,
        "frozen_dual_execution_authorized": lock_validation["execution_authorized"],
        "new_runner_commit": auth["new_runner_commit"],
        "adapter_commit": auth["adapter_commit"],
        "execution_identity": auth["execution_identity"],
        "modal_image_object_id": auth["modal_image_object_id"],
        "new_maximum_forwards": NEW_MAXIMUM_FORWARDS,
        "prior_forward_count": PRIOR_FORWARD_COUNT,
        "m1_case_ids": auth["m1_case_ids"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--phase2-dir", type=Path, required=True)
    parser.add_argument("--m1-manifest", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = validate_modal_execution_authority_v2(
            authorization_path=args.authorization,
            phase2_dir=args.phase2_dir,
            m1_manifest_path=args.m1_manifest,
        )
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
