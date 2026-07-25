#!/usr/bin/env python3
"""Issue one exact, single-use Gate12C-2 draw-profile authorization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gate12c2_development_shards as shards
import gate12c2_draw_profile as profile


def _read(path: Path, *, label: str) -> dict[str, object]:
    payload = profile._read_json_mapping(path, label=label)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--preflight-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--authorization-id", required=True)
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--expires-at-utc", required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args()

    authorization = profile.build_execution_authorization(
        _read(args.plan, label="exact draw-profile plan"),
        _read(
            args.preflight_receipt,
            label="mechanical preflight receipt",
        ),
        output_root=args.output_root,
        authorization_id=args.authorization_id,
        purpose=args.purpose,
        expires_at_utc=args.expires_at_utc,
    )
    shards._atomic_write(
        args.receipt_output.resolve(),
        shards._canonical_json_bytes(authorization),
    )
    print(
        json.dumps(
            {
                "receipt_output": str(
                    args.receipt_output.resolve()
                ),
                "authorization_receipt_payload_sha256": authorization[
                    "authorization_receipt_payload_sha256"
                ],
                "single_use": True,
                "authorization_status": "unconsumed",
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
