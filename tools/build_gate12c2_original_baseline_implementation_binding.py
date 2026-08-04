#!/usr/bin/env python3
"""Build the Gate12C-2 v0.9 candidate implementation binding."""

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


def _require_direct_child_lineage(
    source_commit: str,
    lineage: list[str],
    *,
    expected_parent: str = gate.REMEDIATION_BASE_COMMIT,
) -> None:
    if lineage != [source_commit, expected_parent]:
        raise gate.Gate12C2OriginalBaselineError(
            "IMPLEMENTATION_BYTE_IDENTITY_MISMATCH"
        )


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


def build_candidate_selection(
    repository: Path,
    clean_restore: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(repository).resolve()
    plan = gate.load_active_plan(repository_root=root)
    control = gate.active_remediation_control(plan)
    if control is None:
        raise gate.Gate12C2OriginalBaselineError("INPUT_LINEAGE_MISMATCH")
    identity = gate.active_remediation_identity(plan)
    source_commit = _git(root, "rev-parse", "HEAD")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise gate.Gate12C2OriginalBaselineError(
            "IMPLEMENTATION_BYTE_IDENTITY_MISMATCH"
        )
    object_format = _git(root, "rev-parse", "--show-object-format")
    if (
        object_format not in {"sha1", "sha256"}
        or _git_bool(root, "core.autocrlf", unset=False) is not False
        or _git_bool(root, "core.longpaths", unset=False) is not True
    ):
        raise gate.Gate12C2OriginalBaselineError(
            "IMPLEMENTATION_BYTE_IDENTITY_MISMATCH"
        )
    _require_direct_child_lineage(
        source_commit,
        _git(root, "rev-list", "--parents", "-n", "1", source_commit).split(),
        expected_parent=identity["parent_commit"],
    )
    _require_direct_child_lineage(
        identity["parent_commit"],
        _git(
            root,
            "rev-list",
            "--parents",
            "-n",
            "1",
            identity["parent_commit"],
        ).split(),
        expected_parent=identity["grandparent_commit"],
    )
    restore = validate_clean_restore(clean_restore, source_commit=source_commit)
    receipt, receipt_file_hash = gate.read_r2r1_clean_restore_receipt(plan)
    if (
        restore["restore_receipt_file_sha256"] != receipt_file_hash
        or restore["restore_receipt_payload_sha256"]
        != receipt["restore_receipt_payload_sha256"]
        or receipt["source_commit"] != source_commit
    ):
        raise gate.Gate12C2OriginalBaselineError(
            "IMPLEMENTATION_BYTE_IDENTITY_MISMATCH"
        )
    changed_rows, changed_digest = gate.r2r1_changed_file_manifest(
        plan, root, source_commit
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
    coverage = control["review_coverage_identity"]
    contract = control["candidate_selection_contract"]
    payload = {
        "schema_version": contract["schema_version"],
        "authority_namespace_id": identity["authority_namespace_id"],
        "state": contract["state"],
        "exact_candidate_commit": source_commit,
        "exact_parent_commit": identity["parent_commit"],
        "exact_grandparent_commit": identity["grandparent_commit"],
        "commit_parent_count": 1,
        "parent_parent_count": 1,
        "git_object_format": object_format,
        "changed_path_allowlist": control["allowed_changed_paths"],
        "changed_files": changed_rows,
        "changed_file_manifest_sha256": changed_digest,
        "implementation_files": implementation_rows,
        "r2_activation_plan_file_sha256": (
            gate.R2_ACTIVATION_PLAN_FILE_SHA256
        ),
        "r2_activation_plan_payload_sha256": (
            gate.R2_ACTIVATION_PLAN_PAYLOAD_SHA256
        ),
        "artifact_path_surface_sha256": gate.artifact_surface_sha256(plan),
        **identity["static_fields"],
        "review_surface_identity_sha256": (
            gate.REVIEW_SURFACE_IDENTITY_SHA256
        ),
        "implementation_trust_model_sha256": (
            gate.recompute_implementation_trust_model_sha256(plan)
        ),
        "bundle_path": restore["bundle_path"],
        "bundle_file_sha256": restore["bundle_file_sha256"],
        "bundle_size_bytes": restore["bundle_size_bytes"],
        "clean_restore_receipt_file_sha256": receipt_file_hash,
        "clean_restore_receipt_payload_sha256": receipt[
            "restore_receipt_payload_sha256"
        ],
        "targeted_test_count": coverage["targeted_test_count"],
        "targeted_test_node_id_sha256": coverage[
            "targeted_test_node_id_sha256"
        ],
        "full_suite_test_count": coverage["full_suite_test_count"],
        "full_suite_test_node_id_sha256": coverage[
            "full_suite_test_node_id_sha256"
        ],
        "protected_payload_accessed": False,
        "scientific_values_inspected": False,
        "runtime_authorization_issued": False,
    }
    selection = gate.add_self_hash(
        payload, "candidate_selection_payload_sha256"
    )
    return gate.validate_r2r1_candidate_selection(
        plan,
        selection,
        repo_root=root,
        current_head=source_commit,
    )


def build_candidate_package(
    repository: Path,
    clean_restore: Mapping[str, Any],
    *,
    procedural_author_separation_precondition_satisfied: bool,
    current_exposed_design_context_authored_final_bytes: bool,
    candidate_selection: Mapping[str, Any] | None = None,
    candidate_selection_file_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(repository).resolve()
    plan = gate.load_active_plan(repository_root=root)
    contract = plan["implementation_binding_contract"]
    required_values = contract["required_values"]
    if (
        procedural_author_separation_precondition_satisfied
        is not required_values[
            "procedural_author_separation_precondition_satisfied"
        ]
        or current_exposed_design_context_authored_final_bytes
        is not required_values[
            "current_exposed_design_context_authored_final_bytes"
        ]
    ):
        raise gate.Gate12C2OriginalBaselineError(
            "IMPLEMENTATION_BYTE_IDENTITY_MISMATCH"
        )
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
    expected_base = required_values["remediation_base_commit"]
    expected_base_parent = required_values["remediation_base_parent"]
    source_lineage = _git(
        root, "rev-list", "--parents", "-n", "1", source_commit
    ).split()
    _require_direct_child_lineage(
        source_commit,
        source_lineage,
        expected_parent=expected_base,
    )
    base_lineage = _git(
        root,
        "rev-list",
        "--parents",
        "-n",
        "1",
        expected_base,
    ).split()
    _require_direct_child_lineage(
        expected_base,
        base_lineage,
        expected_parent=expected_base_parent,
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
    identity = gate.active_remediation_identity(plan)
    r2_control = gate.r2_activation_control(plan)
    remediation_control = gate.active_remediation_control(plan)
    if r2_control is None:
        raise gate.Gate12C2OriginalBaselineError(
            "INPUT_LINEAGE_MISMATCH"
        )
    activation_relative = gate.require_text(
        gate.load_r2_activation_plan(
            base_plan=gate.load_frozen_plan(),
            check_legacy_occupancy=True,
        )["activation_plan_relative_path"],
        code="INPUT_LINEAGE_MISMATCH",
    )
    activation_hash, _activation_blob = _tree_blob(
        root,
        source_commit,
        activation_relative,
        object_format,
    )
    if activation_hash != gate.R2_ACTIVATION_PLAN_FILE_SHA256:
        raise gate.Gate12C2OriginalBaselineError(
            "IMPLEMENTATION_BYTE_IDENTITY_MISMATCH"
        )
    if remediation_control is not None:
        remediation_relative = (
            "tools/gate12c2_original_baseline_r2r1_remediation_plan.json"
        )
        remediation_hash, _remediation_blob = _tree_blob(
            root, source_commit, remediation_relative, object_format
        )
        if remediation_hash != gate.R2R1_REMEDIATION_PLAN_FILE_SHA256:
            raise gate.Gate12C2OriginalBaselineError(
                "IMPLEMENTATION_BYTE_IDENTITY_MISMATCH"
            )
    if gate.r2r2_portability_control(plan) is not None:
        portability_hash, _portability_blob = _tree_blob(
            root,
            source_commit,
            gate.R2R2_PORTABILITY_PLAN_RELATIVE_PATH,
            object_format,
        )
        if portability_hash != gate.R2R2_PORTABILITY_PLAN_FILE_SHA256:
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
    for frozen in contract["scientific_dependencies"]:
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
    selection: dict[str, Any] | None = None
    selection_file_hash: str | None = None
    if remediation_control is not None:
        if candidate_selection is None:
            selection, selection_file_hash = gate.read_r2r1_candidate_selection(
                plan,
                repo_root=root,
                current_head=source_commit,
            )
        else:
            selection = gate.validate_r2r1_candidate_selection(
                plan,
                candidate_selection,
                repo_root=root,
                current_head=source_commit,
            )
            selection_file_hash = (
                candidate_selection_file_sha256
                if candidate_selection_file_sha256 is not None
                else gate.sha256_bytes(gate.canonical_receipt_bytes(selection))
            )
        if (
            selection["implementation_files"] != implementation_rows
            or selection["bundle_file_sha256"]
            != restore["bundle_file_sha256"]
            or selection["clean_restore_receipt_file_sha256"]
            != restore["restore_receipt_file_sha256"]
        ):
            raise gate.Gate12C2OriginalBaselineError(
                "IMPLEMENTATION_BYTE_IDENTITY_MISMATCH"
            )
    payload = {
        "schema_version": contract["schema_version"],
        "binding_id": required_values["binding_id"],
        "authority_namespace_id": (
            identity["authority_namespace_id"]
            if remediation_control is not None
            else gate.R2_AUTHORITY_NAMESPACE_ID
        ),
        "source_commit": source_commit,
        "authorized_implementation_repository": str(
            gate.AUTHORIZED_IMPLEMENTATION_REPOSITORY
        ),
        "remediation_base_commit": expected_base,
        "remediation_base_parent": expected_base_parent,
        "git_object_format": object_format,
        "core_autocrlf": False,
        "core_longpaths": True,
        "worktree_clean": True,
        "contract_file_sha256": gate.CONTRACT_FILE_SHA256,
        "plan_file_sha256": gate.PLAN_FILE_SHA256,
        "plan_payload_sha256": gate.PLAN_PAYLOAD_SHA256,
        "r2_activation_plan_file_sha256": (
            gate.R2_ACTIVATION_PLAN_FILE_SHA256
        ),
        "r2_activation_plan_payload_sha256": (
            gate.R2_ACTIVATION_PLAN_PAYLOAD_SHA256
        ),
        "formal_design_review_file_sha256": (
            gate.FORMAL_DESIGN_REVIEW_FILE_SHA256
        ),
        "formal_design_review_payload_sha256": (
            gate.FORMAL_DESIGN_REVIEW_PAYLOAD_SHA256
        ),
        "occupied_v0_9_surface_sha256": (
            gate.R2_OCCUPIED_V0_9_SURFACE_SHA256
        ),
        "implementation_author_separation_contract_sha256": (
            gate.IMPLEMENTATION_AUTHOR_SEPARATION_SHA256
        ),
        "procedural_author_separation_precondition_satisfied": False,
        "implementation_context_blindness_machine_authenticated": False,
        "current_exposed_design_context_authored_final_bytes": True,
        "implementation_trust_model_sha256": (
            gate.recompute_implementation_trust_model_sha256(plan)
        ),
        "artifact_path_surface_sha256": (
            gate.artifact_surface_sha256(plan)
        ),
        "review_surface_identity": gate.review_surface_identity(plan),
        "implementation_files": implementation_rows,
        "scientific_dependencies": scientific_rows,
        "clean_restore": restore,
        "task_identity_used_as_machine_authority": False,
        "implementation_authorship_machine_verified": False,
        "protected_payload_access_required_for_implementation": False,
    }
    if remediation_control is not None:
        if selection is None or selection_file_hash is None:
            raise gate.Gate12C2OriginalBaselineError(
                "IMPLEMENTATION_BYTE_IDENTITY_MISMATCH"
            )
        payload.update(
            {
                **identity["static_fields"],
                "candidate_selection_file_sha256": selection_file_hash,
                "candidate_selection_payload_sha256": selection[
                    "candidate_selection_payload_sha256"
                ],
            }
        )
        manifest_contract = remediation_control["candidate_manifest_contract"]
    else:
        manifest_contract = r2_control["candidate_manifest_contract"]
    manifest_payload = {
        "schema_version": manifest_contract["schema_version"],
        "authority_namespace_id": (
            identity["authority_namespace_id"]
            if remediation_control is not None
            else gate.R2_AUTHORITY_NAMESPACE_ID
        ),
        "state": manifest_contract["state"],
        "activation_source_commit": source_commit,
        "activation_parent_commit": expected_base,
        "task1_parent_commit": expected_base_parent,
        "r2_activation_plan_file_sha256": (
            gate.R2_ACTIVATION_PLAN_FILE_SHA256
        ),
        "r2_activation_plan_payload_sha256": (
            gate.R2_ACTIVATION_PLAN_PAYLOAD_SHA256
        ),
        "artifact_path_surface_sha256": (
            gate.artifact_surface_sha256(plan)
        ),
        "review_surface_identity_sha256": (
            gate.REVIEW_SURFACE_IDENTITY_SHA256
        ),
        "implementation_trust_model_sha256": (
            gate.recompute_implementation_trust_model_sha256(plan)
        ),
        "implementation_files": implementation_rows,
        "scientific_dependencies": scientific_rows,
        "clean_restore": restore,
        "protected_payload_accessed": False,
        "scientific_values_inspected": False,
        "runtime_authorization_issued": False,
    }
    if remediation_control is not None:
        manifest_payload.update(
            {
                **identity["static_fields"],
                "candidate_selection_file_sha256": selection_file_hash,
                "candidate_selection_payload_sha256": selection[
                    "candidate_selection_payload_sha256"
                ],
            }
        )
    else:
        manifest_payload["occupied_v0_9_surface_sha256"] = (
            gate.R2_OCCUPIED_V0_9_SURFACE_SHA256
        )
    manifest = gate.add_self_hash(
        manifest_payload,
        "candidate_manifest_payload_sha256",
    )
    gate.require_exact_keys(
        manifest,
        manifest_contract["exact_top_level_fields"],
        code="IMPLEMENTATION_BYTE_IDENTITY_MISMATCH",
    )
    manifest_file_hash = gate.sha256_bytes(
        gate.canonical_receipt_bytes(manifest)
    )
    payload.update(
        {
            "candidate_manifest_file_sha256": manifest_file_hash,
            "candidate_manifest_payload_sha256": manifest[
                "candidate_manifest_payload_sha256"
            ],
        }
    )
    candidate = gate.add_self_hash(
        payload, "implementation_candidate_binding_payload_sha256"
    )
    validated = gate.validate_candidate_binding(
        plan,
        candidate,
        repo_root=root,
        current_head=source_commit,
        candidate_manifest=manifest,
        candidate_manifest_file_sha256=manifest_file_hash,
        candidate_selection=selection,
        candidate_selection_file_sha256=selection_file_hash,
    )
    return manifest, validated


def build_candidate_binding(
    repository: Path,
    clean_restore: Mapping[str, Any],
    *,
    procedural_author_separation_precondition_satisfied: bool,
    current_exposed_design_context_authored_final_bytes: bool,
) -> dict[str, Any]:
    _manifest, candidate = build_candidate_package(
        repository,
        clean_restore,
        procedural_author_separation_precondition_satisfied=(
            procedural_author_separation_precondition_satisfied
        ),
        current_exposed_design_context_authored_final_bytes=(
            current_exposed_design_context_authored_final_bytes
        ),
    )
    return candidate

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--bundle-path", type=Path, required=True)
    parser.add_argument("--bundle-file-sha256", required=True)
    parser.add_argument("--bundle-size-bytes", type=int, required=True)
    parser.add_argument("--restore-receipt-file-sha256", required=True)
    parser.add_argument("--restore-receipt-payload-sha256", required=True)
    parser.add_argument("--restore-head", required=True)
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
    plan = gate.load_active_plan(repository_root=args.repository)
    selection = build_candidate_selection(args.repository, restore)
    selection_file_hash = gate.sha256_bytes(
        gate.canonical_receipt_bytes(selection)
    )
    gate.publish_r2r1_control_receipt(
        plan,
        "candidate_selection_contract",
        selection,
    )
    manifest, payload = build_candidate_package(
        args.repository,
        restore,
        procedural_author_separation_precondition_satisfied=False,
        current_exposed_design_context_authored_final_bytes=True,
        candidate_selection=selection,
        candidate_selection_file_sha256=selection_file_hash,
    )
    gate.publish_r2r1_control_receipt(
        plan,
        "candidate_manifest_contract",
        manifest,
    )
    gate.publish_role(plan, "implementation_candidate_binding", payload)
    print(
        json.dumps(
            {
                "state": "IMPLEMENTATION_CANDIDATE_BOUND",
                "candidate_selection_payload_sha256": selection[
                    "candidate_selection_payload_sha256"
                ],
                "candidate_manifest_payload_sha256": manifest[
                    "candidate_manifest_payload_sha256"
                ],
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
