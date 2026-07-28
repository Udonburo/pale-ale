#!/usr/bin/env python3
"""Freeze byte-only Gate12C-2 closeout incident evidence."""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import gate12c2_closeout_recovery as recovery


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--incident-id", required=True)
    parser.add_argument("--runner-pid", type=int, required=True)
    parser.add_argument("--stdout-log", type=Path, required=True)
    parser.add_argument("--stderr-log", type=Path, required=True)
    parser.add_argument("--reviewer-context-id", required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--failure-output", type=Path, required=True)
    parser.add_argument("--exposure-output", type=Path, required=True)
    args = parser.parse_args()
    now = datetime.now(timezone.utc).isoformat()
    manifest = recovery.build_incident_manifest(
        args.output_root, incident_id=args.incident_id, observed_at_utc=now
    )
    recovery.write_exclusive_atomic(args.manifest_output, manifest)
    if manifest["state"] != "INCIDENT_FROZEN":
        raise recovery.Gate12C2CloseoutRecoveryError(
            "incident byte surface is not exact"
        )
    failure = recovery.build_failure_receipt(
        incident_manifest_path=args.manifest_output,
        stdout_log_path=args.stdout_log,
        stderr_log_path=args.stderr_log,
        runner_pid=args.runner_pid,
        observed_at_utc=now,
    )
    recovery.write_exclusive_atomic(args.failure_output, failure)
    exposure = recovery.build_exposure_ledger(
        incident_id=args.incident_id,
        reviewer_context_id=args.reviewer_context_id,
        recorded_at_utc=now,
    )
    recovery.write_exclusive_atomic(args.exposure_output, exposure)
    print(json.dumps({
        "state": manifest["state"],
        "file_count": manifest["summary"]["existing_file_count"],
        "shard_count": manifest["summary"]["shard_existing_count"],
        "index_count": manifest["summary"]["index_existing_count"],
        "incident_manifest_payload_sha256": manifest[
            "incident_manifest_payload_sha256"
        ],
        "scientific_values_inspected": False,
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
