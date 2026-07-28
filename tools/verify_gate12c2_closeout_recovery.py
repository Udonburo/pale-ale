#!/usr/bin/env python3
"""Reverify a Gate12C-2 payload completion seal without interpretation."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import gate12c2_closeout_recovery as recovery


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--seal", type=Path, required=True)
    args = parser.parse_args()
    seal = recovery.verify_payload_seal(
        authorization_path=args.authorization, seal_path=args.seal
    )
    print(json.dumps({
        "state": seal["state"],
        "payload_seal_sha256": seal["payload_seal_sha256"],
        "resource_gate_status": seal["resource_gate_status"],
        "scientific_values_emitted": False,
        "stability_analysis_authorized": False,
    }, sort_keys=True))
    return 0


def cli() -> int:
    try:
        return main()
    except Exception:
        print(recovery.PUBLIC_ERROR_CODE, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
