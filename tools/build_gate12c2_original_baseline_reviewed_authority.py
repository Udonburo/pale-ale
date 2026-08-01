#!/usr/bin/env python3
"""Build the deterministic Gate12C-2 v0.8 reviewed authority."""


from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import gate12c2_original_baseline_commitments as gate


def _head(repository: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
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


def build_authority(repository: Path) -> dict[str, object]:
    root = Path(repository).resolve()
    plan = gate.load_frozen_plan()
    gate.read_exact_bytes(
        gate.CONTRACT_PATH,
        gate.CONTRACT_FILE_SHA256,
        code="INPUT_LINEAGE_MISMATCH",
    )
    gate.validate_formal_design_pass(plan)
    contract = plan["implementation_binding_contract"]
    candidate_path = Path(contract["artifact_path"])
    candidate, candidate_file_hash = gate.read_schema_receipt(
        candidate_path,
        exact_fields=contract["exact_top_level_fields"],
        hash_field="implementation_candidate_binding_payload_sha256",
    )
    gate.validate_candidate_binding(
        plan,
        candidate,
        repo_root=root,
        current_head=_head(root),
    )
    review_schema = plan["review_receipt_schemas"][
        "fresh_implementation_review_verdict"
    ]
    review_path = Path(
        plan["reviewed_implementation_authority_contract"][
            "fresh_implementation_review_path"
        ]
    )
    review, review_file_hash = gate.read_schema_receipt(
        review_path,
        exact_fields=review_schema["exact_top_level_fields"],
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
    authority = gate.build_reviewed_authority_payload(
        plan,
        candidate,
        review,
        candidate_file_sha256=candidate_file_hash,
        review_file_sha256=review_file_hash,
    )
    return gate.validate_reviewed_authority(
        plan,
        authority,
        candidate=candidate,
        candidate_file_sha256=candidate_file_hash,
        review=review,
        review_file_sha256=review_file_hash,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args(argv)
    authority = build_authority(args.repository)
    plan = gate.load_frozen_plan()
    gate.publish_role(plan, "reviewed_implementation_authority", authority)
    print(
        json.dumps(
            {
                "state": authority["state"],
                "reviewed_implementation_authority_payload_sha256": authority[
                    "reviewed_implementation_authority_payload_sha256"
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
            "gate12c2-original-baseline-authority:ERROR:"
            "INTERNAL_SANITIZED_FAILURE",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
