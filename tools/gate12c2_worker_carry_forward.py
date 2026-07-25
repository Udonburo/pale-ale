#!/usr/bin/env python3
"""Bounded worker-1/worker-4 carry-forward smoke for Gate12C-2."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import gate12c2_development_shards as shards
import gate12c2_draw_profile as profile
import gate12c2_synthetic_lab as lab


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


def _git_blob_sha256(commit: str, path: str) -> str | None:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        timeout=120,
    )
    if completed.returncode != 0:
        return None
    return hashlib.sha256(completed.stdout).hexdigest()


def _run_one(
    *,
    regime_id: str,
    worker_count: int,
    output_root: Path,
) -> dict[str, Any]:
    plan = shards.build_development_shard_plan(
        regime_id=regime_id,
        master_seed=(
            "gate12c2-worker-carry-forward-v0.1::"
            f"{regime_id}"
        ),
        outer_experiment_indices=[0],
        block_count=4,
        inner_valid_draw_count=1,
        effect_strength=(
            0.25
            if regime_id
            == "S1_known_reverse_shared_node_coupling"
            else None
        ),
    )
    run_root = output_root / regime_id / f"worker-{worker_count}"
    checks = {key: True for key in shards.REQUIRED_PREFLIGHT_CHECKS}
    preflight = shards.build_no_outcome_preflight_receipt(
        plan,
        output_dir=run_root,
        worker_count=worker_count,
        preflight_id=f"worker-carry-forward::{regime_id}::{worker_count}",
        checks=checks,
    )
    authorization = shards.build_development_execution_authorization(
        plan,
        preflight,
        output_dir=run_root,
        worker_count=worker_count,
        authorization_id=(
            f"worker-carry-forward::{regime_id}::{worker_count}"
        ),
        purpose="bounded-worker-equivalence-smoke",
    )
    shards.execute_development_shard_plan(
        plan,
        output_dir=run_root,
        worker_count=worker_count,
        preflight_receipt=preflight,
        authorization_receipt=authorization,
    )
    verification = shards.verify_development_shard_index(
        plan,
        output_dir=run_root,
    )
    return {
        "regime_id": regime_id,
        "worker_count": worker_count,
        "plan_payload_sha256": plan["plan_payload_sha256"],
        "scientific_projection_sha256": verification[
            "scientific_projection_sha256"
        ],
        "index_payload_sha256": verification["index_payload_sha256"],
    }


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
    destination.mkdir(parents=True, exist_ok=True)
    rows = [
        _run_one(
            regime_id=str(regime["regime_id"]),
            worker_count=worker_count,
            output_root=destination,
        )
        for regime in profile.REGIME_SPECIFICATIONS
        for worker_count in (1, 4)
    ]
    projections = {}
    for regime in profile.REGIME_SPECIFICATIONS:
        regime_id = str(regime["regime_id"])
        by_worker = {
            str(row["worker_count"]): row["scientific_projection_sha256"]
            for row in rows
            if row["regime_id"] == regime_id
        }
        if set(by_worker) != {"1", "4"} or (
            by_worker["1"] != by_worker["4"]
        ):
            raise profile.Gate12C2DrawProfileError(
                f"worker carry-forward projections differ for {regime_id}"
            )
        projections[regime_id] = by_worker
    current_plan = profile.build_draw_profile_plan(
        source_commit=current_commit
    )
    comparison = {}
    for path in CRITICAL_PATHS:
        prior_hash = _git_blob_sha256(prior_commit, path)
        current_hash = _git_blob_sha256(current_commit, path)
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
        "bounded_equivalence_smoke": {
            "worker_counts": [1, 4],
            "regime_projection_sha256": projections,
            "status": "pass",
            "scientific_outcomes_interpreted": False,
        },
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
