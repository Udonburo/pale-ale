"""Validate the one-time operational override for frozen Track A only."""

from __future__ import annotations

import argparse
import json
import subprocess
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.gate13_causal_return.phase2_common import (
    REPO_ROOT,
    git_head,
    git_status_porcelain,
    read_json,
    require_fields,
    require_sha256,
    sha256_file,
    sha256_json,
)
from tools.gate13_causal_return.validate_phase2_locks import validate_phase2_locks


STARTING_COMMIT = "a5a83ec5456da9f964d09130a9c7ffabf6545cfa"
PHASE2_A_LOCK_SHA256 = "9c4b94b5199c3d355e8707798ba9bc1797aa2d690762b226f57bec63742215fa"
MODAL_PLAN_SHA256 = "0875fe973806dabb9776053eb818b8d295d25b6162c6a7c0d4296af0188b30df"
TRACK_B_POST_PHASE2_DECISION_SHA256 = (
    "4829ae17d3f86fa39d34fd5a4695d2724bef6fc1e2693c7dda1178d035015b12"
)
B2B_FRESH_SUBSTRATE_RESERVATION_SHA256 = (
    "9412db5406b5c91419cc434c6d493f2d6b6188acccaa7e72059fbe6623270035"
)
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
MAX_SPEND_USD = 25.0
MAX_GPU_WALL_SECONDS = 36_000
EXPECTED_CLOSED = {
    "A3": "CLOSED",
    "TRACK_C": "CLOSED",
    "FORMAL_GATE13": "CLOSED",
    "ACTIVATION_EXTRACTION": "CLOSED",
    "B2B_EXECUTION": "NOT_AUTHORIZED",
    "B2A_REPAIR": "FORBIDDEN",
}


class ModalExecutionAuthorityError(ValueError):
    """Raised on any operational authorization ambiguity or drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ModalExecutionAuthorityError(message)


def _commit_exists(commit: str, repo_root: Path) -> bool:
    return (
        subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=repo_root,
            capture_output=True,
        ).returncode
        == 0
    )


def _validate_git_binding(auth: Mapping[str, Any], repo_root: Path) -> None:
    adapter_commit = str(auth["adapter_commit"])
    _require(_commit_exists(STARTING_COMMIT, repo_root), "starting commit is absent")
    _require(_commit_exists(adapter_commit, repo_root), "adapter commit is absent")
    _require(
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", STARTING_COMMIT, adapter_commit],
            cwd=repo_root,
        ).returncode
        == 0,
        "adapter commit is not descended from the authorized starting commit",
    )
    scientific_paths = [
        "tools/gate13_causal_return",
        ":(exclude)tools/gate13_causal_return/modal",
        "analysis/gate13_causal_return/phase2/track_a_a0_extension_manifest.json",
        "analysis/gate13_causal_return/phase2/track_a_a1_manifest.json",
        "analysis/gate13_causal_return/phase2/track_a_a2_manifest.json",
    ]
    code_commit = "259884d5dc146877bb95428c987697a17a6fbd22"
    _require(
        subprocess.run(
            ["git", "diff", "--quiet", code_commit, adapter_commit, "--", *scientific_paths],
            cwd=repo_root,
        ).returncode
        == 0,
        "frozen scientific runner or case manifests changed",
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
        "adapter changed after its bound commit",
    )
    _require(git_status_porcelain(cwd=repo_root) == "", "Git state is not clean")


def _validate_m1_manifest(path: Path) -> dict[str, Any]:
    manifest = read_json(path)
    expected_internal = str(manifest.get("manifest_sha256") or "")
    without_self = dict(manifest)
    without_self.pop("manifest_sha256", None)
    _require(sha256_json(without_self) == expected_internal, "M1 internal manifest SHA mismatch")
    rows = list(manifest.get("cases") or [])
    _require(len(rows) == 4 == int(manifest.get("case_count") or 0), "M1 case count mismatch")
    ids = [str(row.get("case_id") or "") for row in rows]
    _require(len(ids) == len(set(ids)) and all(ids), "M1 case IDs are empty or duplicated")
    _require(all(row.get("stage") == "A0" for row in rows), "M1 contains a non-A0 case")
    _require(all(row.get("reuse_in_a0") is True for row in rows), "M1 forward would be additive")
    _require(manifest.get("scientific_case_additions") == 0, "M1 adds scientific cases")
    _require(manifest.get("scientific_forward_additions") == 0, "M1 adds forwards")
    for row in rows:
        for field in ("case_sha256", "prompt_sha256", "expected_text_sha256"):
            require_sha256(row[field], field=f"M1.{row['case_id']}.{field}")
    return manifest


def validate_modal_execution_authority(
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
            "adapter_commit",
            "starting_commit",
            "phase2_a_lock_sha256",
            "phase2_a_modal_realization_plan_sha256",
            "model_revision",
            "tokenizer_revision",
            "modal_image_object_id",
            "modal_image_definition",
            "modal_image_definition_sha256",
            "base_image_manifest_digest",
            "gpu_type",
            "maximum_gpu_wall_time_seconds",
            "maximum_authorized_modal_spend_usd",
            "closed_surfaces",
            "execution_identity",
            "model_volume",
            "result_volume",
            "m1_preflight_manifest_sha256",
            "modal_automatic_retries",
            "maximum_active_gpu_containers",
        ),
        context="Modal execution authorization",
    )
    _require(auth["execution_authorized"] is True, "execution_authorized is not true")
    _require(auth["operational_scope"] == "TRACK_A_ONLY", "operational scope is not Track A only")
    _require(auth["starting_commit"] == STARTING_COMMIT, "starting commit mismatch")
    _require(auth["phase2_a_lock_sha256"] == PHASE2_A_LOCK_SHA256, "A lock SHA mismatch")
    _require(
        auth["phase2_a_modal_realization_plan_sha256"] == MODAL_PLAN_SHA256,
        "Modal realization plan SHA mismatch",
    )
    _require(auth["model_revision"] == MODEL_REVISION, "model revision mismatch")
    _require(auth["tokenizer_revision"] == MODEL_REVISION, "tokenizer revision mismatch")
    _require(str(auth["modal_image_object_id"]).startswith("im-"), "Modal Image ID is invalid")
    require_sha256(auth["modal_image_definition_sha256"], field="modal_image_definition_sha256")
    _require(
        sha256_json(auth["modal_image_definition"])
        == auth["modal_image_definition_sha256"],
        "Modal Image definition SHA mismatch",
    )
    image_definition = auth["modal_image_definition"]
    _require(
        image_definition.get("schema_version")
        == "gate13_track_a_modal_image_definition_v1",
        "Modal Image definition schema mismatch",
    )
    _require(image_definition.get("python") == "3.11.2", "image Python mismatch")
    _require(image_definition.get("pip") == "26.2.1", "image pip mismatch")
    image_requirements = set(image_definition.get("requirements") or [])
    for requirement in (
        "torch==2.7.1+cu126",
        "transformers==5.15.0",
        "tokenizers==0.22.2",
        "huggingface-hub==1.27.0",
        "accelerate==1.14.0",
        "safetensors==0.8.0",
    ):
        _require(requirement in image_requirements, f"image requirement mismatch: {requirement}")
    digest = str(auth["base_image_manifest_digest"])
    _require(digest.startswith("sha256:") and len(digest) == 71, "base image digest is invalid")
    _require(
        image_definition.get("base_image_manifest_digest") == digest,
        "base image definition/authorization mismatch",
    )
    _require(auth["gpu_type"] == "NVIDIA L40S", "GPU type mismatch")
    _require(
        int(auth["maximum_gpu_wall_time_seconds"]) == MAX_GPU_WALL_SECONDS,
        "GPU wall-time ceiling mismatch",
    )
    _require(
        float(auth["maximum_authorized_modal_spend_usd"]) == MAX_SPEND_USD,
        "Modal spend ceiling mismatch",
    )
    _require(auth["modal_automatic_retries"] == 0, "Modal retries are not zero")
    _require(auth["maximum_active_gpu_containers"] == 1, "GPU container limit is not one")
    _require(auth["model_volume"]["name"] != auth["result_volume"]["name"], "volumes alias")
    _require(str(auth["model_volume"].get("object_id") or "").startswith("vo-"), "model Volume ID invalid")
    _require(str(auth["result_volume"].get("object_id") or "").startswith("vo-"), "result Volume ID invalid")
    for field, expected in EXPECTED_CLOSED.items():
        _require(auth["closed_surfaces"].get(field) == expected, f"closed surface mismatch: {field}")
    try:
        uuid.UUID(str(auth["execution_identity"]))
    except ValueError as exc:
        raise ModalExecutionAuthorityError("execution identity is not a UUID") from exc

    lock_validation = validate_phase2_locks(
        phase2_dir=phase2_dir, require_clean=False, verify_git=False
    )
    _require(lock_validation["status"] == "PASS", "frozen lock document validation failed")
    _require(
        lock_validation["phase2_a_lock_sha256"] == PHASE2_A_LOCK_SHA256,
        "frozen lock file drifted",
    )
    plan_path = phase2_dir / "phase2_a_modal_realization_plan.json"
    _require(sha256_file(plan_path) == MODAL_PLAN_SHA256, "Modal plan file drifted")
    _require(
        sha256_file(phase2_dir / "TRACK_B_POST_PHASE2_DECISION.md")
        == TRACK_B_POST_PHASE2_DECISION_SHA256,
        "post-Phase-2 Track B decision drifted",
    )
    _require(
        sha256_file(phase2_dir / "b2b_fresh_substrate_reservation.json")
        == B2B_FRESH_SUBSTRATE_RESERVATION_SHA256,
        "fresh B2b reservation drifted",
    )
    m1_manifest = _validate_m1_manifest(m1_manifest_path)
    _require(
        sha256_file(m1_manifest_path) == auth["m1_preflight_manifest_sha256"],
        "M1 file SHA mismatch",
    )
    _require(
        [row["case_id"] for row in m1_manifest["cases"]] == auth.get("m1_case_ids"),
        "M1 case binding mismatch",
    )
    if verify_git:
        _validate_git_binding(auth, repo_root)
        _require(git_head(cwd=repo_root) != STARTING_COMMIT, "authorization commit is absent")

    return {
        "schema_version": "gate13_track_a_modal_execution_authority_validation_v1",
        "status": "PASS",
        "operational_execution_authorized": True,
        "frozen_dual_execution_authorized": lock_validation["execution_authorized"],
        "adapter_commit": auth["adapter_commit"],
        "execution_identity": auth["execution_identity"],
        "modal_image_object_id": auth["modal_image_object_id"],
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
        result = validate_modal_execution_authority(
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
