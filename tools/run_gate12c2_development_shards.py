#!/usr/bin/env python3
"""Execute an exactly authorized Gate12C-2 development shard plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gate12c2_development_shards import (
    Gate12C2ShardError,
    execute_development_shard_plan,
)


def _read_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise Gate12C2ShardError(
            f"could not read {label} JSON {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise Gate12C2ShardError(f"{label} must contain a JSON object")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--preflight-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--authorization-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers <= 0:
        raise SystemExit("workers must be positive")
    plan = _read_json_mapping(args.plan, label="plan")
    preflight = _read_json_mapping(
        args.preflight_receipt,
        label="preflight receipt",
    )
    authorization = _read_json_mapping(
        args.authorization_receipt,
        label="authorization receipt",
    )
    result = execute_development_shard_plan(
        plan,
        output_dir=args.output_dir,
        worker_count=args.workers,
        preflight_receipt=preflight,
        authorization_receipt=authorization,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "plan_payload_sha256": result["plan_payload_sha256"],
                "scientific_projection_sha256": result[
                    "scientific_projection_sha256"
                ],
                "outer_experiment_count": result[
                    "outer_experiment_count"
                ],
                "all_outer_indices_present": result[
                    "all_outer_indices_present"
                ],
                "locked_execution_authorized": result[
                    "locked_execution_authorized"
                ],
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
