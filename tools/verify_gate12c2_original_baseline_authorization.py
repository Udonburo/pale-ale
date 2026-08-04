#!/usr/bin/env python3
"""Independently verify a Gate12C-2 v0.9 authorization."""


from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import gate12c2_original_baseline_commitments as gate
import issue_gate12c2_original_baseline_preflight as preflight_issuer


RELATIVE_PATH = "tools/verify_gate12c2_original_baseline_authorization.py"


def verify_authorization(
    repository: Path,
    *,
    scope: str,
    verification_id: str,
    verified_at_utc: str,
    now_ns: int | None = None,
) -> dict[str, Any]:
    root = Path(repository).resolve()
    plan = gate.load_active_plan(repository_root=root)
    if scope not in {"extraction", "verifier"}:
        raise gate.Gate12C2OriginalBaselineError("AUTHORIZATION_INVALID")
    binding_schema = plan["implementation_binding_contract"]
    candidate, _candidate_file_hash = gate.read_schema_receipt(
        Path(binding_schema["artifact_path"]),
        exact_fields=gate.artifact_exact_fields(
            plan, "implementation_candidate_binding"
        ),
        hash_field="implementation_candidate_binding_payload_sha256",
        code="IMPLEMENTATION_BYTE_IDENTITY_MISMATCH",
    )
    implementation_rows = candidate.get("implementation_files")
    matching_rows = (
        [
            row
            for row in implementation_rows
            if isinstance(row, dict)
            and row.get("relative_path") == RELATIVE_PATH
        ]
        if isinstance(implementation_rows, list)
        else []
    )
    try:
        raw_self = (root / RELATIVE_PATH).read_bytes()
    except OSError:
        raise gate.Gate12C2OriginalBaselineError(
            "IMPLEMENTATION_BYTE_IDENTITY_MISMATCH"
        ) from None
    verifier_file_sha256 = gate.sha256_bytes(raw_self)
    verifier_git_blob_oid = gate.git_blob_oid(
        raw_self, candidate.get("git_object_format")
    )
    rows = gate.artifact_rows_by_role(plan)
    preflight_schema = plan["control_receipt_schemas"][f"{scope}_preflight"]
    preflight, preflight_file_hash = gate.read_schema_receipt(
        Path(rows[f"{scope}_preflight"]["final_path"]),
        exact_fields=preflight_schema["exact_top_level_fields"],
        hash_field="preflight_payload_sha256",
        code="AUTHORIZATION_INVALID",
    )
    authorization_schema = plan["control_receipt_schemas"][
        f"{scope}_authorization"
    ]
    authorization, authorization_file_hash = gate.read_schema_receipt(
        Path(rows[f"{scope}_authorization"]["final_path"]),
        exact_fields=authorization_schema["exact_top_level_fields"],
        hash_field="authorization_payload_sha256",
        code="AUTHORIZATION_INVALID",
    )
    current_ns = (
        gate.parse_utc_ns(gate.utc_now_text(), code="AUTHORIZATION_INVALID")
        if now_ns is None
        else gate.require_int(now_ns, code="AUTHORIZATION_INVALID")
    )
    reason_code: str | None = None
    try:
        gate.read_exact_bytes(
            gate.CONTRACT_PATH,
            gate.CONTRACT_FILE_SHA256,
            code="INPUT_LINEAGE_MISMATCH",
        )
        gate.validate_formal_design_pass(plan)
        gate.validate_upstream_authority(plan)
        gate.validate_original_input_lineage(plan)
        authority, authority_file_hash, reviewed_candidate = (
            preflight_issuer.load_reviewed_chain(plan, root)
        )
        if reviewed_candidate != candidate:
            raise gate.Gate12C2OriginalBaselineError(
                "IMPLEMENTATION_BYTE_IDENTITY_MISMATCH"
            )
        link_kwargs: dict[str, str] = {}
        if scope == "verifier":
            success, success_file, terminal, terminal_file = (
                preflight_issuer._load_verified_extraction(
                    plan,
                    authority,
                    authority_file_hash,
                    reviewed_candidate,
                )
            )
            link_kwargs = {
                "extraction_terminal_file_sha256": terminal_file,
                "extraction_terminal_payload_sha256": terminal[
                    "terminal_claim_payload_sha256"
                ],
                "baseline_receipt_file_sha256": success_file,
                "baseline_receipt_payload_sha256": success[
                    "baseline_receipt_payload_sha256"
                ],
            }
        if len(matching_rows) != 1:
            raise gate.Gate12C2OriginalBaselineError(
                "IMPLEMENTATION_BYTE_IDENTITY_MISMATCH"
            )
        implementation_row = matching_rows[0]
        if (
            verifier_file_sha256 != implementation_row.get("file_sha256")
            or verifier_git_blob_oid != implementation_row.get("git_blob_oid")
        ):
            raise gate.Gate12C2OriginalBaselineError(
                "IMPLEMENTATION_BYTE_IDENTITY_MISMATCH"
            )
        gate.validate_preflight_payload(
            plan,
            preflight,
            scope=scope,
            reviewed_authority_file_sha256=authority_file_hash,
            reviewed_authority_payload_sha256=authority[
                "reviewed_implementation_authority_payload_sha256"
            ],
            implementation_source_commit=preflight[
                "implementation_source_commit"
            ],
            now_ns=current_ns,
            **link_kwargs,
        )
        expires_ns = gate.parse_utc_ns(
            authorization.get("expires_at_utc"), code="AUTHORIZATION_INVALID"
        )
        if current_ns >= expires_ns:
            raise gate.Gate12C2OriginalBaselineError("AUTHORIZATION_INVALID")
        gate.validate_authorization_payload(
            plan,
            authorization,
            preflight,
            scope=scope,
            preflight_file_sha256=preflight_file_hash,
            now_ns=current_ns,
        )
        observations = preflight_issuer._observe_surface(plan)
        phase = gate.classify_lifecycle_surface(
            plan,
            observations,
            temporal_predicate=f"{scope}_preflight_and_authorization_fresh",
        )
        if phase != f"{scope}_authorization_issued_unverified_fresh":
            raise gate.Gate12C2OriginalBaselineError(
                "OUTPUT_SURFACE_MISMATCH"
            )
    except gate.Gate12C2OriginalBaselineError as exc:
        verdict_schema = plan["control_receipt_schemas"][
            f"{scope}_authorization_verdict"
        ]
        reason_code = (
            exc.code
            if exc.code in verdict_schema["reason_code_allowlist"]
            else "AUTHORIZATION_INVALID"
        )
        try:
            expires_ns = gate.parse_utc_ns(
                authorization.get("expires_at_utc"),
                code="AUTHORIZATION_INVALID",
            )
        except gate.Gate12C2OriginalBaselineError:
            raise exc from None
        if current_ns >= expires_ns:
            reason_code = "AUTHORIZATION_STALE"
    return gate.build_authorization_verdict_payload(
        plan,
        authorization,
        preflight,
        scope=scope,
        verification_id=verification_id,
        verified_at_utc=verified_at_utc,
        outcome_kind="pass" if reason_code is None else "reject",
        reason_code=reason_code,
        preflight_file_sha256=preflight_file_hash,
        authorization_file_sha256=authorization_file_hash,
        verifier_relative_path=RELATIVE_PATH,
        verifier_file_sha256=verifier_file_sha256,
        verifier_git_blob_oid=verifier_git_blob_oid,
        now_ns=current_ns,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--scope", choices=("extraction", "verifier"), required=True)
    parser.add_argument("--verification-id", required=True)
    parser.add_argument("--verified-at-utc", required=True)
    args = parser.parse_args(argv)
    verdict = verify_authorization(
        args.repository,
        scope=args.scope,
        verification_id=args.verification_id,
        verified_at_utc=args.verified_at_utc,
    )
    plan = gate.load_active_plan(repository_root=args.repository)
    gate.publish_role(plan, f"{args.scope}_authorization_verdict", verdict)
    if verdict["outcome_kind"] == "reject":
        print(
            gate.AUTHORIZATION_ERROR_PREFIX + str(verdict["reason_code"]),
            file=sys.stderr,
        )
        return 2
    print(gate.AUTHORIZATION_VERIFICATION_PASS_LINE)
    return 0


def cli(argv: list[str] | None = None) -> int:
    try:
        return main(argv)
    except SystemExit:
        raise
    except gate.Gate12C2OriginalBaselineError as exc:
        print(gate.AUTHORIZATION_ERROR_PREFIX + exc.code, file=sys.stderr)
        return 2
    except Exception:
        print(
            gate.AUTHORIZATION_ERROR_PREFIX + "INTERNAL_SANITIZED_FAILURE",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
