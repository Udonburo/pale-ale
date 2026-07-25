#!/usr/bin/env python3
"""Issue one mechanically verified Gate12C-2 draw-profile preflight receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gate12c2_development_shards as shards
import gate12c2_draw_profile as profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight-id", required=True)
    parser.add_argument("--recovery-bundle", type=Path, required=True)
    parser.add_argument(
        "--worker-profile-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--worker-carry-forward-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--restore-scratch-root",
        type=Path,
        required=True,
    )
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args()

    receipt = profile.issue_mechanical_preflight(
        plan_path=args.plan,
        output_root=args.output_root,
        preflight_id=args.preflight_id,
        recovery_bundle_path=args.recovery_bundle,
        worker_profile_receipt_path=args.worker_profile_receipt,
        worker_carry_forward_receipt_path=(
            args.worker_carry_forward_receipt
        ),
        restore_scratch_root=args.restore_scratch_root,
    )
    shards._atomic_write(
        args.receipt_output.resolve(),
        shards._canonical_json_bytes(receipt),
    )
    print(
        json.dumps(
            {
                "receipt_output": str(
                    args.receipt_output.resolve()
                ),
                "preflight_receipt_payload_sha256": receipt[
                    "preflight_receipt_payload_sha256"
                ],
                "development_execution_authorized": False,
                "locked_execution_authorized": False,
                "scientific_outcomes_inspected": False,
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
