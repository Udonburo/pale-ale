#!/usr/bin/env python3
"""Issue a Gate12C-2 v0.9 single-use authorization."""


from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import gate12c2_original_baseline_commitments as gate
import issue_gate12c2_original_baseline_preflight as preflight_issuer


def issue_authorization(
    repository: Path,
    *,
    scope: str,
    authorization_id: str,
    issued_at_utc: str,
    expires_at_utc: str,
    now_ns: int | None = None,
) -> dict[str, Any]:
    root = Path(repository).resolve()
    plan = gate.load_active_plan(repository_root=root)
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
    authority, authority_file_hash, candidate = (
        preflight_issuer.load_reviewed_chain(
            plan, Path(repository).resolve()
        )
    )
    rows = gate.artifact_rows_by_role(plan)
    schema = plan["control_receipt_schemas"][f"{scope}_preflight"]
    preflight, preflight_file_hash = gate.read_schema_receipt(
        Path(rows[f"{scope}_preflight"]["final_path"]),
        exact_fields=schema["exact_top_level_fields"],
        hash_field="preflight_payload_sha256",
        code="AUTHORIZATION_INVALID",
    )
    link_kwargs: dict[str, str] = {}
    if scope == "verifier":
        success, success_file, terminal, terminal_file = (
            preflight_issuer._load_verified_extraction(
                plan, authority, authority_file_hash, candidate
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
    gate.validate_preflight_payload(
        plan,
        preflight,
        scope=scope,
        reviewed_authority_file_sha256=authority_file_hash,
        reviewed_authority_payload_sha256=authority[
            "reviewed_implementation_authority_payload_sha256"
        ],
        implementation_source_commit=candidate["source_commit"],
        now_ns=now_ns,
        **link_kwargs,
    )
    observations = preflight_issuer._observe_surface(plan)
    phase = gate.classify_lifecycle_surface(
        plan,
        observations,
        temporal_predicate=f"{scope}_preflight_fresh",
    )
    if phase != f"{scope}_preflight_passed_fresh":
        raise gate.Gate12C2OriginalBaselineError("OUTPUT_PUBLICATION_FAILED")
    return gate.build_authorization_payload(
        plan,
        preflight,
        scope=scope,
        preflight_file_sha256=preflight_file_hash,
        authorization_id=authorization_id,
        issued_at_utc=issued_at_utc,
        expires_at_utc=expires_at_utc,
        now_ns=now_ns,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--scope", choices=("extraction", "verifier"), required=True)
    parser.add_argument("--authorization-id", required=True)
    parser.add_argument("--issued-at-utc", required=True)
    parser.add_argument("--expires-at-utc", required=True)
    args = parser.parse_args(argv)
    payload = issue_authorization(
        args.repository,
        scope=args.scope,
        authorization_id=args.authorization_id,
        issued_at_utc=args.issued_at_utc,
        expires_at_utc=args.expires_at_utc,
    )
    plan = gate.load_active_plan(repository_root=args.repository)
    gate.publish_role(plan, f"{args.scope}_authorization", payload)
    print(
        json.dumps(
            {
                "state": payload["state"],
                "authorization_payload_sha256": payload[
                    "authorization_payload_sha256"
                ],
                "single_use": True,
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
            "gate12c2-original-baseline-authorization:ERROR:"
            "INTERNAL_SANITIZED_FAILURE",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
