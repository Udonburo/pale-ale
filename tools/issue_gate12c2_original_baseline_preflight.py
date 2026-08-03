#!/usr/bin/env python3
"""Issue a Gate12C-2 v0.9 mechanical preflight."""


from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import gate12c2_original_baseline_commitments as gate


def _git_head(repository: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        raise gate.Gate12C2OriginalBaselineError(
            "IMPLEMENTATION_BYTE_IDENTITY_MISMATCH"
        ) from None


def load_reviewed_chain(
    plan: dict[str, Any], repository: Path
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    binding_schema = plan["implementation_binding_contract"]
    candidate, candidate_file_hash = gate.read_schema_receipt(
        Path(binding_schema["artifact_path"]),
        exact_fields=gate.artifact_exact_fields(
            plan, "implementation_candidate_binding"
        ),
        hash_field="implementation_candidate_binding_payload_sha256",
    )
    gate.validate_candidate_binding(
        plan,
        candidate,
        repo_root=repository,
        current_head=_git_head(repository),
    )
    review_schema = plan["review_receipt_schemas"][
        "fresh_implementation_review_verdict"
    ]
    review, review_file_hash = gate.read_schema_receipt(
        Path(review_schema["artifact_path"]),
        exact_fields=gate.artifact_exact_fields(
            plan, "fresh_implementation_review_verdict"
        ),
        hash_field="fresh_implementation_review_payload_sha256",
    )
    gate.validate_implementation_review(
        plan,
        review,
        candidate_file_sha256=candidate_file_hash,
        candidate_payload_sha256=candidate[
            "implementation_candidate_binding_payload_sha256"
        ],
        source_commit=candidate["source_commit"],
        candidate=candidate,
    )
    authority_schema = plan["reviewed_implementation_authority_contract"]
    authority, authority_file_hash = gate.read_schema_receipt(
        Path(authority_schema["artifact_path"]),
        exact_fields=gate.artifact_exact_fields(
            plan, "reviewed_implementation_authority"
        ),
        hash_field="reviewed_implementation_authority_payload_sha256",
    )
    gate.validate_reviewed_authority(
        plan,
        authority,
        candidate=candidate,
        candidate_file_sha256=candidate_file_hash,
        review=review,
        review_file_sha256=review_file_hash,
    )
    return authority, authority_file_hash, candidate


def _observe_surface(
    plan: dict[str, Any],
) -> dict[str, gate.ArtifactObservation]:
    return gate.observe_artifact_surface(plan)


def _load_verified_extraction(
    plan: dict[str, Any],
    authority: dict[str, Any],
    authority_file_hash: str,
    candidate: dict[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    rows = gate.artifact_rows_by_role(plan)
    verifier_rows = [
        row
        for row in candidate["implementation_files"]
        if row["relative_path"]
        == "tools/verify_gate12c2_original_baseline_authorization.py"
    ]
    if len(verifier_rows) != 1:
        raise gate.Gate12C2OriginalBaselineError(
            "IMPLEMENTATION_BYTE_IDENTITY_MISMATCH"
        )
    verifier_row = verifier_rows[0]
    preflight_schema = plan["control_receipt_schemas"]["extraction_preflight"]
    preflight, preflight_file_hash = gate.read_schema_receipt(
        Path(rows["extraction_preflight"]["final_path"]),
        exact_fields=preflight_schema["exact_top_level_fields"],
        hash_field="preflight_payload_sha256",
        code="INPUT_LINEAGE_MISMATCH",
    )
    gate.validate_preflight_payload(
        plan,
        preflight,
        scope="extraction",
        reviewed_authority_file_sha256=authority_file_hash,
        reviewed_authority_payload_sha256=authority[
            "reviewed_implementation_authority_payload_sha256"
        ],
        implementation_source_commit=candidate["source_commit"],
        now_ns=gate.parse_utc_ns(
            preflight["issued_at_utc"], code="INPUT_LINEAGE_MISMATCH"
        ),
    )
    authorization_schema = plan["control_receipt_schemas"][
        "extraction_authorization"
    ]
    authorization, authorization_file_hash = gate.read_schema_receipt(
        Path(rows["extraction_authorization"]["final_path"]),
        exact_fields=authorization_schema["exact_top_level_fields"],
        hash_field="authorization_payload_sha256",
        code="INPUT_LINEAGE_MISMATCH",
    )
    gate.validate_authorization_payload(
        plan,
        authorization,
        preflight,
        scope="extraction",
        preflight_file_sha256=preflight_file_hash,
        now_ns=gate.parse_utc_ns(
            authorization["issued_at_utc"], code="INPUT_LINEAGE_MISMATCH"
        ),
    )
    verdict_schema = plan["control_receipt_schemas"][
        "extraction_authorization_verdict"
    ]
    verdict, verdict_file_hash = gate.read_schema_receipt(
        Path(rows["extraction_authorization_verdict"]["final_path"]),
        exact_fields=verdict_schema["exact_top_level_fields"],
        hash_field="authorization_verdict_payload_sha256",
        code="INPUT_LINEAGE_MISMATCH",
    )
    gate.validate_authorization_verdict_payload(
        plan,
        verdict,
        authorization,
        preflight,
        scope="extraction",
        preflight_file_sha256=preflight_file_hash,
        authorization_file_sha256=authorization_file_hash,
        verifier_relative_path=verifier_row["relative_path"],
        verifier_file_sha256=verifier_row["file_sha256"],
        verifier_git_blob_oid=verifier_row["git_blob_oid"],
        now_ns=gate.parse_utc_ns(
            verdict["verified_at_utc"], code="INPUT_LINEAGE_MISMATCH"
        ),
    )
    claim_schema = plan["control_receipt_schemas"]["extraction_execution_claim"]
    claim, claim_file_hash = gate.read_schema_receipt(
        Path(rows["extraction_execution_claim"]["final_path"]),
        exact_fields=claim_schema["exact_top_level_fields"],
        hash_field="execution_claim_payload_sha256",
        code="INPUT_LINEAGE_MISMATCH",
    )
    gate.validate_execution_claim_payload(
        plan,
        claim,
        authorization,
        preflight,
        verdict,
        scope="extraction",
        preflight_file_sha256=preflight_file_hash,
        authorization_file_sha256=authorization_file_hash,
        verdict_file_sha256=verdict_file_hash,
        now_ns=gate.parse_utc_ns(
            claim["claimed_at_utc"], code="INPUT_LINEAGE_MISMATCH"
        ),
    )
    success_schema = plan["success_receipt"]
    success, success_file_hash = gate.read_schema_receipt(
        Path(rows["extraction_success"]["final_path"]),
        exact_fields=success_schema["exact_top_level_fields"],
        hash_field="baseline_receipt_payload_sha256",
        code="INPUT_LINEAGE_MISMATCH",
    )
    gate.validate_extraction_success_leaf(
        plan,
        success,
        reviewed_authority_file_sha256=authority_file_hash,
        reviewed_authority_payload_sha256=authority[
            "reviewed_implementation_authority_payload_sha256"
        ],
        preflight_file_sha256=preflight_file_hash,
        preflight_payload_sha256=preflight["preflight_payload_sha256"],
        authorization_file_sha256=authorization_file_hash,
        authorization_payload_sha256=authorization[
            "authorization_payload_sha256"
        ],
        authorization_verdict_file_sha256=verdict_file_hash,
        authorization_verdict_payload_sha256=verdict[
            "authorization_verdict_payload_sha256"
        ],
        execution_claim_file_sha256=claim_file_hash,
        execution_claim_payload_sha256=claim["execution_claim_payload_sha256"],
        implementation_source_commit=claim["implementation_source_commit"],
        git_head_at_protected_read=claim["implementation_source_commit"],
        git_head_at_terminal=claim["implementation_source_commit"],
        executing_code_identity_surface_sha256=claim[
            "executing_code_identity_surface_sha256"
        ],
    )
    terminal_schema = plan["control_receipt_schemas"]["extraction_terminal"]
    terminal, terminal_file_hash = gate.read_schema_receipt(
        Path(rows["extraction_terminal"]["final_path"]),
        exact_fields=terminal_schema["exact_top_level_fields"],
        hash_field="terminal_claim_payload_sha256",
        code="INPUT_LINEAGE_MISMATCH",
    )
    gate.validate_terminal_payload(
        plan,
        terminal,
        claim,
        success,
        scope="extraction",
        reviewed_authority_file_sha256=authority_file_hash,
        preflight_file_sha256=preflight_file_hash,
        authorization_file_sha256=authorization_file_hash,
        verdict_file_sha256=verdict_file_hash,
        execution_claim_file_sha256=claim_file_hash,
    )
    return success, success_file_hash, terminal, terminal_file_hash


def issue_preflight(
    repository: Path,
    *,
    scope: str,
    preflight_id: str,
    issued_at_utc: str,
    expires_at_utc: str,
    now_ns: int | None = None,
) -> dict[str, Any]:
    root = Path(repository).resolve()
    plan = gate.load_active_plan()
    if scope not in {"extraction", "verifier"}:
        raise gate.Gate12C2OriginalBaselineError("AUTHORIZATION_INVALID")
    gate.read_exact_bytes(
        gate.CONTRACT_PATH,
        gate.CONTRACT_FILE_SHA256,
        code="INPUT_LINEAGE_MISMATCH",
    )
    gate.validate_formal_design_pass(plan)
    gate.validate_upstream_authority(plan)
    gate.validate_original_input_lineage(plan)
    authority, authority_file_hash, candidate = load_reviewed_chain(plan, root)
    observations = _observe_surface(plan)
    kwargs: dict[str, Any] = {}
    required_phase = "reviewed_implementation_authority_published"
    if scope == "verifier":
        success, success_file, terminal, terminal_file = (
            _load_verified_extraction(plan, authority, authority_file_hash, candidate)
        )
        kwargs = {
            "baseline_receipt_file_sha256": success_file,
            "baseline_receipt_payload_sha256": success[
                "baseline_receipt_payload_sha256"
            ],
            "extraction_terminal_file_sha256": terminal_file,
            "extraction_terminal_payload_sha256": terminal[
                "terminal_claim_payload_sha256"
            ],
        }
        required_phase = "extraction_success_complete"
    phase = gate.classify_lifecycle_surface(plan, observations)
    if phase != required_phase:
        raise gate.Gate12C2OriginalBaselineError("OUTPUT_PUBLICATION_FAILED")
    return gate.build_preflight_payload(
        plan,
        scope=scope,
        preflight_id=preflight_id,
        issued_at_utc=issued_at_utc,
        expires_at_utc=expires_at_utc,
        reviewed_authority_file_sha256=authority_file_hash,
        reviewed_authority_payload_sha256=authority[
            "reviewed_implementation_authority_payload_sha256"
        ],
        implementation_source_commit=candidate["source_commit"],
        now_ns=now_ns,
        **kwargs,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--scope", choices=("extraction", "verifier"), required=True)
    parser.add_argument("--preflight-id", required=True)
    parser.add_argument("--issued-at-utc", required=True)
    parser.add_argument("--expires-at-utc", required=True)
    args = parser.parse_args(argv)
    plan = gate.load_active_plan()
    payload = issue_preflight(
        args.repository,
        scope=args.scope,
        preflight_id=args.preflight_id,
        issued_at_utc=args.issued_at_utc,
        expires_at_utc=args.expires_at_utc,
    )
    gate.publish_role(plan, f"{args.scope}_preflight", payload)
    print(
        json.dumps(
            {
                "state": payload["state"],
                "preflight_payload_sha256": payload[
                    "preflight_payload_sha256"
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
            "gate12c2-original-baseline-preflight:ERROR:"
            "INTERNAL_SANITIZED_FAILURE",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
