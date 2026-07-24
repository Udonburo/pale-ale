#!/usr/bin/env python3
"""Write one development-only Gate12C-2 N1 fidelity receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from gate12c2_n1_fidelity import (
    DEFAULT_S1_EFFECT_STRENGTHS,
    run_development_n1_fidelity_profile,
)


RECEIPT_SCHEMA_VERSION = "gate12c2_n1_fidelity_receipt_v0.1"
BOUNDARY_SCHEMA_VERSION = "gate12c2_development_freeze_receipt_v0.2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _load_boundary_receipt(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != BOUNDARY_SCHEMA_VERSION:
        raise RuntimeError("unexpected C2 development-boundary schema")
    claimed = payload.get("receipt_payload_sha256")
    projection = dict(payload)
    projection.pop("receipt_payload_sha256", None)
    computed = hashlib.sha256(
        _canonical_json(projection).encode("utf-8")
    ).hexdigest()
    if claimed != computed:
        raise RuntimeError("C2 development-boundary payload hash mismatch")
    authorization = payload.get("authorization", {})
    if any(
        bool(authorization.get(field_name))
        for field_name in (
            "locked_synthetic_execution",
            "real_held_out_execution",
            "N2_implementation",
            "N3_implementation",
            "public_claim",
        )
    ):
        raise RuntimeError(
            "development fidelity requires every downstream authorization closed"
        )
    return payload


def build_receipt(
    *,
    boundary_receipt_path: Path,
    repository_commit: str,
    master_seed: str,
    block_count: int,
    draw_count_per_case: int,
    effect_strengths: tuple[float, ...],
) -> dict:
    boundary = _load_boundary_receipt(boundary_receipt_path)
    profile = run_development_n1_fidelity_profile(
        master_seed=master_seed,
        block_count=block_count,
        draw_count_per_case=draw_count_per_case,
        effect_strengths=effect_strengths,
    )
    module_path = Path(__file__).with_name(
        "gate12c2_n1_fidelity.py"
    )
    synthetic_lab_path = Path(__file__).with_name(
        "gate12c2_synthetic_lab.py"
    )
    runner_path = Path(__file__).resolve()
    payload = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "epistemic_status": (
            "development_nuisance_exploration_not_threshold_freeze"
        ),
        "repository_commit": repository_commit,
        "input_receipt": {
            "development_boundary": {
                "path": str(boundary_receipt_path.resolve()),
                "file_sha256": _sha256(boundary_receipt_path),
                "payload_sha256": boundary[
                    "receipt_payload_sha256"
                ],
                "repository_commit": boundary[
                    "repository_commit"
                ],
            }
        },
        "profile": profile,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "reference_dtype": "float64",
        },
        "implementation_sha256": {
            synthetic_lab_path.name: _sha256(synthetic_lab_path),
            module_path.name: _sha256(module_path),
            runner_path.name: _sha256(runner_path),
        },
        "authorization": {
            "freeze_nuisance_threshold_from_this_receipt_alone": False,
            "locked_synthetic_execution": False,
            "real_held_out_execution": False,
            "N2_implementation": False,
            "N3_implementation": False,
            "public_claim": False,
        },
    }
    payload["receipt_payload_sha256"] = hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--development-boundary-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument("--repository-commit", required=True)
    parser.add_argument("--master-seed", required=True)
    parser.add_argument("--block-count", type=int, default=64)
    parser.add_argument("--draw-count-per-case", type=int, default=16)
    parser.add_argument(
        "--effect-strength",
        type=float,
        action="append",
        dest="effect_strengths",
    )
    args = parser.parse_args()
    strengths = tuple(
        DEFAULT_S1_EFFECT_STRENGTHS
        if args.effect_strengths is None
        else args.effect_strengths
    )
    receipt = build_receipt(
        boundary_receipt_path=(
            args.development_boundary_receipt.resolve()
        ),
        repository_commit=str(args.repository_commit),
        master_seed=str(args.master_seed),
        block_count=int(args.block_count),
        draw_count_per_case=int(args.draw_count_per_case),
        effect_strengths=strengths,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "receipt_payload_sha256": receipt[
                    "receipt_payload_sha256"
                ],
                "deterministic_projection_sha256": receipt["profile"][
                    "deterministic_projection_sha256"
                ],
                "elapsed_seconds": receipt["profile"][
                    "elapsed_seconds"
                ],
                "locked_synthetic_execution": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
