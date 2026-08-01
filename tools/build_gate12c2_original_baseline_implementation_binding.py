#!/usr/bin/env python3
"""Build the Gate12C-2 v0.8 candidate implementation binding."""

from __future__ import annotations


import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import gate12c2_original_baseline_commitments as gate


IMPLEMENTATION_ROLES = {
    "tools/gate12c2_original_baseline_commitments.py": "extraction_core",
    "tools/build_gate12c2_original_baseline_implementation_binding.py": (
        "implementation_binding_builder"
    ),
    "tools/build_gate12c2_original_baseline_reviewed_authority.py": (
        "reviewed_authority_builder"
    ),
    "tools/issue_gate12c2_original_baseline_preflight.py": "preflight_issuer",
    "tools/issue_gate12c2_original_baseline_authorization.py": (
        "authorization_issuer"
    ),
    "tools/run_gate12c2_original_baseline_extraction.py": "extraction_runner",
    "tools/verify_gate12c2_original_baseline_authorization.py": (
        "authorization_verifier"
    ),
    "tools/verify_gate12c2_original_baseline_commitments.py": (
        "independent_verifier"
    ),
    "tools/test_gate12c2_original_baseline_commitments.py": "primary_tests",
    "tools/test_gate12c2_original_baseline_commitments_adversarial.py": (
        "adversarial_tests"
    ),
}


def _git(repository: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.SubprocessError):
        raise gate.Gate12C2OriginalBaselineError(
            "IMPLEMENTATION_BYTE_IDENTITY_MISMATCH"
        ) from None
    return completed.stdout.strip()


def _git_bool(repository: Path, key: str, *, unset: bool) -> bool:
    try:
        value = _git(repository, "config", "--bool", "--get", key)
    except gate.Gate12C2OriginalBaselineError:
        return unset
    if value not in {"true", "false"}:
        raise gate.Gate12C2OriginalBaselineError(
            "IMPLEMENTATION_BYTE_IDENTITY_MISMATCH"
        )
    return value == "true"


def _tree_blob(
    repository: Path, head: str, relative_path: str, object_format: str
) -> tuple[str, str]:
    output = _git(repository, "ls-tree", head, "--", relative_path)
    fields = output.split(None, 3)
    if (
        len(fields) != 4
        or fields[1] != "blob"
        or fields[3] != relative_path
        or not gate.is_git_oid(fields[2])
    ):
        raise gate.Gate12C2OriginalBaselineError(
            "IMPLEMENTATION_BYTE_IDENTITY_MISMATCH"
        )
    raw = (repository / relative_path).read_bytes()
    computed = gate.git_blob_oid(raw, object_format)
    if computed != fields[2]:
        raise gate.Gate12C2OriginalBaselineError(
            "IMPLEMENTATION_BYTE_IDENTITY_MISMATCH"
        )
    return gate.sha256_bytes(raw), computed


def validate_clean_restore(
    clean_restore: Mapping[str, Any], *, source_commit: str
) -> dict[str, Any]:
    supplied = dict(clean_restore)
    exact = {
        "bundle_path",
        "bundle_file_sha256",
        "bundle_size_bytes",
        "restore_receipt_file_sha256",
        "restore_receipt_payload_sha256",
        "restore_head",
        "restore_worktree_clean",
        "git_fsck_full_pass",
        "core_autocrlf",
        "core_longpaths",
        "implementation_rows_match",
        "scientific_dependency_rows_match",
    }
    gate.require_exact_keys(
        supplied,
        exact,
        code="IMPLEMENTATION_BYTE_IDENTITY_MISMATCH",
    )
    bundle = Path(
        gate.require_text(
            supplied["bundle_path"],
            code="IMPLEMENTATION_BYTE_IDENTITY_MISMATCH",
        )
    )
    try:
        bundle_raw = bundle.read_bytes()
    except OSError:
        raise gate.Gate12C2OriginalBaselineError(
            "IMPLEMENTATION_BYTE_IDENTITY_MISMATCH"
        ) from None
    bundle_size = gate.require_int(
        supplied["bundle_size_bytes"],
        minimum=1,
        code="IMPLEMENTATION_BYTE_IDENTITY_MISMATCH",
    )
    if (
        gate.sha256_bytes(bundle_raw) != supplied["bundle_file_sha256"]
        or len(bundle_raw) != bundle_size
        or supplied["restore_head"] != source_commit
    ):
        raise gate.Gate12C2OriginalBaselineError(
            "IMPLEMENTATION_BYTE_IDENTITY_MISMATCH"
        )
    required = {
        "restore_worktree_clean": True,
        "git_fsck_full_pass": True,
        "core_autocrlf": False,
        "core_longpaths": True,
        "implementation_rows_match": True,
        "scientific_dependency_rows_match": True,
    }
    if any(supplied.get(key) is not value for key, value in required.items()):
        raise gate.Gate12C2OriginalBaselineError(
            "IMPLEMENTATION_BYTE_IDENTITY_MISMATCH"
        )
    for field in (
        "bundle_file_sha256",
        "restore_receipt_file_sha256",
        "restore_receipt_payload_sha256",
    ):
        if not gate.is_sha256(supplied[field]):
            raise gate.Gate12C2OriginalBaselineError(
                "IMPLEMENTATION_BYTE_IDENTITY_MISMATCH"
            )
    return supplied


def build_candidate_binding(
    repository: Path,
    clean_restore: Mapping[str, Any],
    *,
    procedural_author_separation_precondition_satisfied: bool,
    current_exposed_design_context_authored_final_bytes: bool,
) -> dict[str, Any]:
    if (
        procedural_author_separation_precondition_satisfied is not True
        or current_exposed_design_context_authored_final_bytes is not False
    ):
        raise gate.Gate12C2OriginalBaselineError(
            "IMPLEMENTATION_BYTE_IDENTITY_MISMATCH"
        )
    root = Path(repository).resolve()
    plan = gate.load_frozen_plan()
    gate.read_exact_bytes(
        gate.CONTRACT_PATH,
        gate.CONTRACT_FILE_SHA256,
        code="INPUT_LINEAGE_MISMATCH",
    )
    gate.validate_formal_design_pass(plan)
    source_commit = _git(root, "rev-parse", "HEAD")
    if len(source_commit) != 40:
        raise gate.Gate12C2OriginalBaselineError(
            "IMPLEMENTATION_BYTE_IDENTITY_MISMATCH"
        )
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise gate.Gate12C2OriginalBaselineError(
            "IMPLEMENTATION_BYTE_IDENTITY_MISMATCH"
        )
    object_format = _git(root, "rev-parse", "--show-object-format")
    if object_format not in {"sha1", "sha256"}:
        raise gate.Gate12C2OriginalBaselineError(
            "IMPLEMENTATION_BYTE_IDENTITY_MISMATCH"
        )
    autocrlf = _git_bool(root, "core.autocrlf", unset=False)
    longpaths = _git_bool(root, "core.longpaths", unset=False)
    if autocrlf is not False or longpaths is not True:
        raise gate.Gate12C2OriginalBaselineError(
            "IMPLEMENTATION_BYTE_IDENTITY_MISMATCH"
        )
    implementation_rows = []
    for relative in gate.IMPLEMENTATION_PATHS:
        file_hash, blob = _tree_blob(
            root, source_commit, relative, object_format
        )
        implementation_rows.append(
            {
                "role": IMPLEMENTATION_ROLES[relative],
                "relative_path": relative,
                "file_sha256": file_hash,
                "git_blob_oid": blob,
            }
        )
    implementation_rows.sort(key=lambda row: row["role"])
    scientific_rows = []
    for frozen in plan["implementation_binding_contract"][
        "scientific_dependencies"
    ]:
        row = dict(frozen)
        current_hash, current_blob = _tree_blob(
            root, source_commit, row["relative_path"], object_format
        )
        frozen_blob = _git(
            root,
            "rev-parse",
            f"{row['source_commit']}:{row['relative_path']}",
        )
        if (
            current_hash != row["file_sha256"]
            or current_blob != row["git_blob_oid"]
            or frozen_blob != row["git_blob_oid"]
        ):
            raise gate.Gate12C2OriginalBaselineError(
                "IMPLEMENTATION_BYTE_IDENTITY_MISMATCH"
            )
        scientific_rows.append(row)
    restore = validate_clean_restore(
        clean_restore, source_commit=source_commit
    )
    payload = {
        "schema_version": (
            "gate12c2_original_baseline_implementation_candidate_binding_v0.8"
        ),
        "binding_id": (
            "C2_ORIGINAL_BASELINE_COMMITMENT_GATE_IMPLEMENTATION_CANDIDATE_BINDING_v0.8"
        ),
        "source_commit": source_commit,
        "git_object_format": object_format,
        "core_autocrlf": False,
        "core_longpaths": True,
        "worktree_clean": True,
        "contract_file_sha256": gate.CONTRACT_FILE_SHA256,
        "plan_file_sha256": gate.PLAN_FILE_SHA256,
        "plan_payload_sha256": gate.PLAN_PAYLOAD_SHA256,
        "formal_design_review_file_sha256": (
            gate.FORMAL_DESIGN_REVIEW_FILE_SHA256
        ),
        "formal_design_review_payload_sha256": (
            gate.FORMAL_DESIGN_REVIEW_PAYLOAD_SHA256
        ),
        "implementation_author_separation_contract_sha256": (
            gate.IMPLEMENTATION_AUTHOR_SEPARATION_SHA256
        ),
        "procedural_author_separation_precondition_satisfied": True,
        "implementation_context_blindness_machine_authenticated": False,
        "current_exposed_design_context_authored_final_bytes": False,
        "implementation_trust_model_sha256": (
            gate.IMPLEMENTATION_TRUST_MODEL_SHA256
        ),
        "artifact_path_surface_sha256": (
            gate.ARTIFACT_PATH_SURFACE_SHA256
        ),
        "implementation_files": implementation_rows,
        "scientific_dependencies": scientific_rows,
        "clean_restore": restore,
        "task_identity_used_as_machine_authority": False,
        "implementation_authorship_machine_verified": False,
        "protected_payload_access_required_for_implementation": False,
    }
    result = gate.add_self_hash(
        payload, "implementation_candidate_binding_payload_sha256"
    )
    return gate.validate_candidate_binding(
        plan,
        result,
        repo_root=root,
        current_head=source_commit,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--bundle-path", type=Path, required=True)
    parser.add_argument("--bundle-file-sha256", required=True)
    parser.add_argument("--bundle-size-bytes", type=int, required=True)
    parser.add_argument("--restore-receipt-file-sha256", required=True)
    parser.add_argument("--restore-receipt-payload-sha256", required=True)
    parser.add_argument("--restore-head", required=True)
    parser.add_argument(
        "--confirm-fresh-unexposed-no-history-implementation-task",
        action="store_true",
    )
    parser.add_argument(
        "--confirm-exposed-design-context-authored-no-final-byte",
        action="store_true",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    restore = {
        "bundle_path": str(args.bundle_path.resolve()),
        "bundle_file_sha256": args.bundle_file_sha256,
        "bundle_size_bytes": args.bundle_size_bytes,
        "restore_receipt_file_sha256": args.restore_receipt_file_sha256,
        "restore_receipt_payload_sha256": (
            args.restore_receipt_payload_sha256
        ),
        "restore_head": args.restore_head,
        "restore_worktree_clean": True,
        "git_fsck_full_pass": True,
        "core_autocrlf": False,
        "core_longpaths": True,
        "implementation_rows_match": True,
        "scientific_dependency_rows_match": True,
    }
    payload = build_candidate_binding(
        args.repository,
        restore,
        procedural_author_separation_precondition_satisfied=(
            args.confirm_fresh_unexposed_no_history_implementation_task
        ),
        current_exposed_design_context_authored_final_bytes=not (
            args.confirm_exposed_design_context_authored_no_final_byte
        ),
    )
    plan = gate.load_frozen_plan()
    gate.publish_role(plan, "implementation_candidate_binding", payload)
    print(
        json.dumps(
            {
                "state": "IMPLEMENTATION_CANDIDATE_BOUND",
                "implementation_candidate_binding_payload_sha256": payload[
                    "implementation_candidate_binding_payload_sha256"
                ],
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


def cli(argv: list[str] | None = None) -> int:
    try:
        return main(argv)
    except SystemExit:
        raise
    except Exception:
        print(
            "gate12c2-original-baseline-binding:ERROR:"
            "INTERNAL_SANITIZED_FAILURE",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
