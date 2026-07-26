#!/usr/bin/env python3
"""Bounded worker-1/worker-4 carry-forward smoke for Gate12C-2."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import gate12c2_draw_profile as profile
import gate12c2_development_shards as shards


CRITICAL_PATHS = (
    "tools/gate12c2_synthetic_lab.py",
    "tools/gate12c2_development_shards.py",
    "tools/gate12c2_draw_profile.py",
)


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(Path(__file__).resolve().parents[1]),
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return completed.stdout.strip()


def run_carry_forward(
    *,
    prior_commit: str,
    current_commit: str,
    worker_profile_receipt_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    if _git("rev-parse", "HEAD") != current_commit:
        raise profile.Gate12C2DrawProfileError(
            "worker carry-forward requires the exact checked-out current commit"
        )
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise profile.Gate12C2DrawProfileError(
            "worker carry-forward requires a clean worktree"
        )
    worker_profile = profile._verify_prior_worker_profile(
        worker_profile_receipt_path
    )
    destination = Path(output_root).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise profile.Gate12C2DrawProfileError(
            "worker carry-forward output root must be empty"
        )
    current_plan = profile.build_draw_profile_plan(
        source_commit=current_commit
    )
    smoke = profile._run_bounded_worker_equivalence_smoke(
        current_plan,
        output_root=destination,
    )
    comparison = {}
    for path in CRITICAL_PATHS:
        prior_hash = profile._git_blob_sha256(
            prior_commit,
            path,
            allow_missing=True,
        )
        current_hash = profile._git_blob_sha256(current_commit, path)
        comparison[path] = {
            "prior_sha256": prior_hash,
            "current_sha256": current_hash,
            "status": (
                "unchanged"
                if prior_hash is not None and prior_hash == current_hash
                else (
                    "changed_with_bounded_equivalence_smoke"
                    if prior_hash is not None
                    else "new_shared_path_with_bounded_equivalence_smoke"
                )
            ),
        }
    payload: dict[str, Any] = {
        "schema_version": profile.WORKER_CARRY_FORWARD_SCHEMA_VERSION,
        "epistemic_status": (
            "development_worker_selection_carry_forward_only"
        ),
        "surface_id": "development",
        "prior_worker_profile_file_sha256": worker_profile["file_sha256"],
        "prior_worker_profile_payload_sha256": worker_profile[
            "payload_sha256"
        ],
        "prior_commit": prior_commit,
        "current_commit": current_commit,
        "worker_count": profile.WORKER_COUNT,
        "current_implementation_sha256": current_plan[
            "implementation_sha256"
        ],
        "current_numerical_environment_sha256": current_plan[
            "numerical_environment_sha256"
        ],
        "worker_critical_file_comparison": comparison,
        "bounded_equivalence_smoke": smoke,
        "status": "pass",
        "locked_execution_authorized": False,
        "real_held_out_execution_authorized": False,
        "N2_open": False,
        "N3_open": False,
    }
    payload["carry_forward_receipt_payload_sha256"] = (
        profile._sha256_bytes(profile._canonical_json_bytes(payload))
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-commit", required=True)
    parser.add_argument("--current-commit", required=True)
    parser.add_argument(
        "--worker-profile-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args()
    receipt = run_carry_forward(
        prior_commit=args.prior_commit,
        current_commit=args.current_commit,
        worker_profile_receipt_path=args.worker_profile_receipt,
        output_root=args.output_root,
    )
    shards._atomic_write(
        args.receipt_output.resolve(),
        shards._canonical_json_bytes(receipt),
    )
    print(
        json.dumps(
            {
                "receipt_output": str(args.receipt_output.resolve()),
                "carry_forward_receipt_payload_sha256": receipt[
                    "carry_forward_receipt_payload_sha256"
                ],
                "status": "pass",
                "scientific_outcomes_interpreted": False,
            },
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
