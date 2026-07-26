#!/usr/bin/env python3
"""Execute the exact authorized Gate12C-2 development draw profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gate12c2_draw_profile as profile


def _read_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise profile.Gate12C2DrawProfileError(
            f"could not read {label} {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise profile.Gate12C2DrawProfileError(
            f"{label} must contain a JSON object"
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--preflight-receipt", type=Path, required=True)
    parser.add_argument(
        "--authorization-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--restore-scratch-root",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    receipt = profile.execute_draw_profile(
        _read_mapping(args.plan, label="draw profile plan"),
        output_root=args.output_root,
        preflight_receipt=_read_mapping(
            args.preflight_receipt,
            label="no-outcome preflight",
        ),
        authorization_receipt=_read_mapping(
            args.authorization_receipt,
            label="execution authorization",
        ),
        restore_scratch_root=args.restore_scratch_root,
    )
    print(
        json.dumps(
            {
                "output_root": str(args.output_root.resolve()),
                "execution_receipt_payload_sha256": receipt[
                    "execution_receipt_payload_sha256"
                ],
                "configuration_count": receipt["configuration_count"],
                "scientific_calibration_result": None,
                "scientific_outcomes_exposed": False,
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
