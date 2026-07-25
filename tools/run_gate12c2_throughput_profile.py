#!/usr/bin/env python3
"""Build or execute the bounded Gate12C-2 development throughput profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gate12c2_throughput_profile as profile


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build-plan", action="store_true")
    mode.add_argument("--execute-plan", type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--plan-out", type=Path)
    parser.add_argument("--outer-count", type=int, default=4)
    parser.add_argument("--inner-valid-draw-count", type=int, default=255)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--receipt-out", type=Path)
    parser.add_argument("--preflight-receipt", type=Path)
    parser.add_argument("--authorization-receipt", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.build_plan:
        if not args.source_commit or args.plan_out is None:
            raise SystemExit(
                "--build-plan requires --source-commit and --plan-out"
            )
        plan = profile.build_bounded_worker_profile_plan(
            source_commit=args.source_commit,
            outer_count_per_workload=args.outer_count,
            inner_valid_draw_count=args.inner_valid_draw_count,
            worker_counts=args.workers,
        )
        _write_json(args.plan_out, plan)
        print(
            json.dumps(
                {
                    "plan_out": str(args.plan_out.resolve()),
                    "profile_plan_payload_sha256": plan[
                        "profile_plan_payload_sha256"
                    ],
                    "configuration_count": len(plan["configurations"]),
                    "locked_execution_authorized": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if (
        args.output_root is None
        or args.receipt_out is None
        or args.preflight_receipt is None
        or args.authorization_receipt is None
    ):
        raise SystemExit(
            "--execute-plan requires --output-root, --receipt-out, "
            "--preflight-receipt, and --authorization-receipt"
        )
    plan = json.loads(
        args.execute_plan.read_text(encoding="utf-8")
    )
    receipt = profile.execute_profile_plan(
        plan,
        output_root=args.output_root,
        preflight_receipt=json.loads(
            args.preflight_receipt.read_text(encoding="utf-8")
        ),
        authorization_receipt=json.loads(
            args.authorization_receipt.read_text(encoding="utf-8")
        ),
    )
    _write_json(args.receipt_out, receipt)
    print(
        json.dumps(
            {
                "receipt_out": str(args.receipt_out.resolve()),
                "profile_receipt_payload_sha256": receipt[
                    "profile_receipt_payload_sha256"
                ],
                "determinism_pass": receipt["summary"][
                    "determinism_pass"
                ],
                "memory_gate_pass": receipt["summary"][
                    "memory_gate_pass"
                ],
                "locked_execution_authorized": False,
                "scientific_calibration_result": None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
