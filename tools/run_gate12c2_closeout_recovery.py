#!/usr/bin/env python3
"""Execute an authorized read-only payload verification and external seal."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import gate12c2_closeout_recovery as recovery


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args()
    seal = recovery.execute_payload_seal(args.authorization)
    print(json.dumps({
        "state": seal["state"],
        "payload_seal_sha256": seal["payload_seal_sha256"],
        "payload_integrity_status": seal["payload_integrity_status"],
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
