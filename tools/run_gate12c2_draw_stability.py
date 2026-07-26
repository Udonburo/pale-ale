#!/usr/bin/env python3
"""Emit only the authorized no-outcome Gate12C-2 draw-stability projection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import gate12c2_development_shards as shards
import gate12c2_draw_stability as stability


PUBLIC_ERROR_CODE = "GATE12C2_DRAW_STABILITY_INPUT_REJECTED"


def _read_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        raise stability.Gate12C2DrawStabilityError(
            f"could not read {label}"
        ) from None
    if not isinstance(payload, dict):
        raise stability.Gate12C2DrawStabilityError(
            f"{label} must contain a JSON object"
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = _read_mapping(args.manifest, label="analysis manifest")
    projection = stability.analyze_verified_directories(manifest)
    shards._atomic_write(
        args.output.resolve(),
        shards._canonical_json_bytes(projection),
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "projection_payload_sha256": projection[
                    "projection_payload_sha256"
                ],
                "selected_draw_count": projection[
                    "selected_draw_count"
                ],
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


def cli() -> int:
    try:
        return main()
    except Exception:
        print(PUBLIC_ERROR_CODE, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
