"""Validate one fresh Modal execution of the constrained Track A channel."""

from __future__ import annotations

import json
import math
import subprocess
import uuid
from pathlib import Path
from typing import Any, Mapping

from tools.gate13_causal_return.phase2_common import (
    git_status_porcelain,
    read_json,
    require_fields,
    require_sha256,
    sha256_file,
    sha256_json,
)
from tools.gate13_causal_return.track_a.validate_constrained_channel_lock import (
    validate_constrained_channel_lock,
)


CONSTRAINED_LOCK_SHA256 = "413213494e091d6e49826580557f0c1f5415cb738514bd9471f8e1a92e17640e"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
MODEL_VOLUME_NAME = "gate13-track-a-qwen3-8b-b968826-model"
MODEL_VOLUME_OBJECT_ID = "vo-qcBxnmomu5avCRoacF7eKW"
RESULT_VOLUME_NAME = "gate13-track-a-constrained-v1-results"
PRIOR_FORWARD_COUNT = 5
MAXIMUM_ADDITIONAL_FORWARDS = 595
CUMULATIVE_FORWARD_CEILING = 600
CUMULATIVE_SPEND_CEILING_USD = 25.0
PRIOR_SPEND_RESERVATION_USD = 0.12
MAXIMUM_NEW_SPEND_USD = 24.88
CUMULATIVE_GPU_WALL_CEILING_SECONDS = 36_000.0
PRIOR_GPU_WALL_RESERVATION_SECONDS = 143.93
MAXIMUM_NEW_GPU_WALL_SECONDS = 35_856.07
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
    "B2B_EXECUTION": "NOT_AUTHORIZED",
    "A3": "CLOSED",
    "TRACK_C": "CLOSED",
    "FORMAL_GATE13": "CLOSED",
    "ACTIVATION_EXTRACTION": "CLOSED",
    "PUBLIC_README_OR_GATE_MAP_EDITS": "FORBIDDEN",
}
BOUND_PATHS = (
    "analysis/gate13_causal_return/phase2/TRACK_A_CONSTRAINED_CHANNEL_DECISION.md",
    "analysis/gate13_causal_return/phase2/phase2_a_constrained_channel_lock.json",
    "analysis/gate13_causal_return/phase2/track_a_constrained_channel_manifest.json",
    "tools/gate13_causal_return/modal/m1_constrained_preflight_manifest.json",
    "tools/gate13_causal_return/modal/modal_track_a_constrained.py",
    "tools/gate13_causal_return/modal/validate_constrained_channel.py",
    "tools/gate13_causal_return/modal/validate_constrained_modal_execution_authority.py",
    "tools/gate13_causal_return/track_a/compile_constrained_channel.py",
    "tools/gate13_causal_return/track_a/constrained_channel.py",
    "tools/gate13_causal_return/track_a/constrained_runner.py",
    "tools/gate13_causal_return/track_a/validate_constrained_channel_lock.py",
)


class ConstrainedModalAuthorityError(ValueError):
    """Raised when constrained Modal execution authority is ambiguous."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConstrainedModalAuthorityError(message)


def _close(value: object, expected: float, field: str) -> None:
    _require(
        math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-9),
        f"{field} mismatch",
    )


def _commit_exists(commit: str, repo_root: Path) -> bool:
    return (
        subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=repo_root,
            capture_output=True,
        ).returncode
        == 0
    )


def validate_constrained_modal_execution_authority_payload(
    *,
    auth: Mapping[str, Any],
    phase2_dir: Path,
    repo_root: Path,
    verify_git: bool,
) -> dict[str, Any]:
    require_fields(
        auth,
        (
            "schema_version",
            "execution_authorized",
            "operational_scope",
            "implementation_commit",
            "bound_file_hashes",
            "constrained_channel_lock_sha256",
            "exact_validation",
            "model_revision",
            "tokenizer_revision",
            "modal_image_object_id",
            "modal_image_definition",
            "modal_image_definition_sha256",
            "gpu_type",
            "execution_identity",
            "model_volume",
            "result_volume",
            "forward_accounting",
            "spend_accounting",
            "gpu_wall_accounting",
            "m1_case_ids",
            "m1_metric_contribution_count",
            "modal_automatic_retries",
            "maximum_active_gpu_containers",
            "closed_surfaces",
        ),
        context="constrained Modal execution authority",
    )
    _require(
        auth["schema_version"]
        == "gate13_track_a_constrained_modal_execution_authorization_v1",
        "authorization schema mismatch",
    )
    _require(auth["execution_authorized"] is True, "execution is not authorized")
    _require(auth["operational_scope"] == "TRACK_A_CONSTRAINED_CHANNEL_ONLY", "scope mismatch")
    _require(
        auth["constrained_channel_lock_sha256"] == CONSTRAINED_LOCK_SHA256,
        "constrained lock binding mismatch",
    )
    lock_path = phase2_dir / "phase2_a_constrained_channel_lock.json"
    _require(sha256_file(lock_path) == CONSTRAINED_LOCK_SHA256, "constrained lock drifted")
    lock_validation = validate_constrained_channel_lock(
        phase2_dir=phase2_dir,
        require_clean=False,
    )
    _require(lock_validation["status"] == "PASS", "constrained lock validation failed")

    bound = auth["bound_file_hashes"]
    for relative in BOUND_PATHS:
        _require(relative in bound, f"bound file hash absent: {relative}")
        _require(
            sha256_file(repo_root / relative) == bound[relative],
            f"bound file drifted: {relative}",
        )
    exact = auth["exact_validation"]
    _require(exact.get("status") == "PASS", "exact constrained validation did not pass")
    require_sha256(exact.get("receipt_sha256"), field="exact validation receipt SHA")
    if verify_git:
        receipt_path = repo_root / str(exact["local_receipt_path"])
        _require(receipt_path.exists(), "exact validation receipt is absent")
        _require(
            sha256_file(receipt_path) == exact["receipt_sha256"],
            "exact validation receipt drifted",
        )

    _require(auth["model_revision"] == MODEL_REVISION, "model revision mismatch")
    _require(auth["tokenizer_revision"] == MODEL_REVISION, "tokenizer revision mismatch")
    model_volume = auth["model_volume"]
    _require(model_volume["name"] == MODEL_VOLUME_NAME, "model Volume name mismatch")
    _require(model_volume["object_id"] == MODEL_VOLUME_OBJECT_ID, "model Volume object mismatch")
    _require(model_volume["identity"] == EXPECTED_MODEL_IDENTITY, "model Volume identity mismatch")
    result_volume = auth["result_volume"]
    _require(result_volume["name"] == RESULT_VOLUME_NAME, "result Volume name mismatch")
    _require(str(result_volume.get("object_id") or "").startswith("vo-"), "result Volume object ID invalid")
    _require(result_volume["name"] != model_volume["name"], "model/result Volume alias")

    _require(str(auth["modal_image_object_id"]).startswith("im-"), "Modal Image ID invalid")
    require_sha256(auth["modal_image_definition_sha256"], field="image definition SHA")
    _require(
        sha256_json(auth["modal_image_definition"])
        == auth["modal_image_definition_sha256"],
        "image definition hash mismatch",
    )
    image = auth["modal_image_definition"]
    _require(image.get("python") == "3.11.2", "image Python mismatch")
    requirements = set(image.get("requirements") or [])
    for requirement in (
        "torch==2.7.1+cu126",
        "transformers==5.15.0",
        "tokenizers==0.22.2",
    ):
        _require(requirement in requirements, f"image requirement absent: {requirement}")
    _require(auth["gpu_type"] == "NVIDIA L40S", "GPU mismatch")

    forward = auth["forward_accounting"]
    _require(int(forward["cumulative_ceiling"]) == CUMULATIVE_FORWARD_CEILING, "forward ceiling mismatch")
    _require(int(forward["prior_consumed_count"]) == PRIOR_FORWARD_COUNT, "prior forward count mismatch")
    _require(int(forward["maximum_additional_count"]) == MAXIMUM_ADDITIONAL_FORWARDS, "additional forward limit mismatch")
    _require(int(forward["prior_consumed_count"]) + int(forward["maximum_additional_count"]) == CUMULATIVE_FORWARD_CEILING, "forward arithmetic mismatch")
    spend = auth["spend_accounting"]
    _close(spend["cumulative_ceiling_usd"], CUMULATIVE_SPEND_CEILING_USD, "spend ceiling")
    _close(spend["prior_usage_reservation_usd"], PRIOR_SPEND_RESERVATION_USD, "prior spend reservation")
    _close(spend["maximum_new_usage_usd"], MAXIMUM_NEW_SPEND_USD, "new spend limit")
    wall = auth["gpu_wall_accounting"]
    _close(wall["cumulative_ceiling_seconds"], CUMULATIVE_GPU_WALL_CEILING_SECONDS, "GPU wall ceiling")
    _close(wall["prior_reservation_seconds"], PRIOR_GPU_WALL_RESERVATION_SECONDS, "prior GPU wall reservation")
    _close(wall["maximum_new_seconds"], MAXIMUM_NEW_GPU_WALL_SECONDS, "new GPU wall limit")

    _require(auth["m1_case_ids"] == [
        "a0-l12-y0-early-r0-S",
        "a0-l12-y0-early-r0-O",
        "a0-l12-y0-early-r0-E",
        "a0-l12-y0-early-r0-N",
    ], "M1 case identity mismatch")
    _require(auth["m1_metric_contribution_count"] == 0, "M1 contributes to metrics")
    _require(auth["modal_automatic_retries"] == 0, "automatic retries are not zero")
    _require(auth["maximum_active_gpu_containers"] == 1, "GPU container maximum mismatch")
    for field, expected in EXPECTED_CLOSED.items():
        _require(auth["closed_surfaces"].get(field) == expected, f"closed surface mismatch: {field}")
    try:
        uuid.UUID(str(auth["execution_identity"]))
    except ValueError as exc:
        raise ConstrainedModalAuthorityError("execution identity is not a UUID") from exc

    if verify_git:
        commit = str(auth["implementation_commit"])
        _require(_commit_exists(commit, repo_root), "implementation commit is absent")
        _require(
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
                cwd=repo_root,
                capture_output=True,
            ).returncode
            == 0,
            "implementation commit is not an ancestor of HEAD",
        )
        for relative in BOUND_PATHS:
            _require(
                subprocess.run(
                    ["git", "diff", "--quiet", commit, "HEAD", "--", relative],
                    cwd=repo_root,
                ).returncode
                == 0,
                f"bound implementation changed after commit: {relative}",
            )
        _require(git_status_porcelain(cwd=repo_root) == "", "Git state is not clean")
    return {
        "schema_version": "gate13_track_a_constrained_modal_authority_validation_v1",
        "status": "PASS",
        "execution_authorized": True,
        "execution_identity": auth["execution_identity"],
        "implementation_commit": auth["implementation_commit"],
        "modal_image_object_id": auth["modal_image_object_id"],
        "result_volume": dict(auth["result_volume"]),
        "prior_forward_count": PRIOR_FORWARD_COUNT,
        "maximum_additional_forwards": MAXIMUM_ADDITIONAL_FORWARDS,
    }


def validate_constrained_modal_execution_authority(
    *, authorization_path: Path, phase2_dir: Path, repo_root: Path, verify_git: bool = True
) -> dict[str, Any]:
    return validate_constrained_modal_execution_authority_payload(
        auth=read_json(authorization_path),
        phase2_dir=phase2_dir,
        repo_root=repo_root,
        verify_git=verify_git,
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--phase2-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = validate_constrained_modal_execution_authority(
            authorization_path=args.authorization,
            phase2_dir=args.phase2_dir,
            repo_root=args.repo_root,
        )
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
