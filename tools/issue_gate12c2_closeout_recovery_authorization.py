#!/usr/bin/env python3
"""Issue one exact payload-seal recovery authorization."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import gate12c2_closeout_recovery as recovery


def main() -> int:
    parser = recovery.SanitizedArgumentParser(description=__doc__)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--review-receipt", type=Path, required=True)
    parser.add_argument("--incident-manifest", type=Path, required=True)
    parser.add_argument("--archived-plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--authorization-id", required=True)
    parser.add_argument("--expires-at-utc", required=True)
    parser.add_argument("--attempt-output", type=Path, required=True)
    parser.add_argument("--consumption-output", type=Path, required=True)
    parser.add_argument("--seal-output", type=Path, required=True)
    parser.add_argument("--failure-output", type=Path, required=True)
    parser.add_argument("--authorization-output", type=Path, required=True)
    args = parser.parse_args()
    authorization = recovery.build_recovery_authorization(
        amendment_path=args.amendment,
        review_receipt_path=args.review_receipt,
        incident_manifest_path=args.incident_manifest,
        archived_plan_path=args.archived_plan,
        output_root=args.output_root,
        authorization_id=args.authorization_id,
        expires_at_utc=args.expires_at_utc,
        authorization_output=args.authorization_output,
        attempt_output=args.attempt_output,
        consumption_output=args.consumption_output,
        seal_output=args.seal_output,
        failure_output=args.failure_output,
    )
    recovery.write_exclusive_atomic(args.authorization_output, authorization)
    print(json.dumps({
        "authorization_payload_sha256": authorization["authorization_payload_sha256"],
        "expires_at_utc": authorization["expires_at_utc"],
        "stale_lock_retirement_authorized": False,
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
