#!/usr/bin/env python3
"""Record an explicit Gate12C-2 coordinator stale-lock recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gate12c2_draw_profile as profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--recovery-id", required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()

    receipt = profile.recover_stale_coordinator_lock(
        profile._read_json_mapping(
            args.plan,
            label="exact draw-profile plan",
        ),
        output_root=args.output_root,
        recovery_id=args.recovery_id,
        reason=args.reason,
    )
    print(
        json.dumps(
            {
                "recovery_receipt_payload_sha256": receipt[
                    "recovery_receipt_payload_sha256"
                ],
                "partial_artifact_count": 0,
                "prior_owner_pid_not_running": True,
                "locked_execution_authorized": False,
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
