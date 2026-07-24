#!/usr/bin/env python3
"""Run confound diagnostics on the sealed process-triage development surface."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from process_triage_baseline_diagnostics import run_baseline_diagnostics
from process_triage_evaluator import build_feature_surface, cheap_features
from run_process_triage_baseline_development import (
    _load_boundary_receipt,
    _load_development_partition,
)


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


def _load_baseline_receipt(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version")
        != "pale_ale_process_triage_baseline_receipt_v0.1"
    ):
        raise RuntimeError("unexpected baseline development receipt schema")
    exposure = payload.get("exposure_transition", {})
    if exposure.get("structural_signal_opened") is not False:
        raise RuntimeError("baseline receipt unexpectedly opened structural work")
    if exposure.get("prospective_locked_partition_scored") is not False:
        raise RuntimeError("baseline receipt unexpectedly scored locked data")
    return payload


def build_receipt(
    *,
    dataset_root: Path,
    boundary_receipt_path: Path,
    baseline_receipt_path: Path,
    repository_commit: str,
) -> dict[str, Any]:
    boundary = _load_boundary_receipt(boundary_receipt_path)
    baseline_receipt = _load_baseline_receipt(baseline_receipt_path)
    if (
        baseline_receipt["boundary_receipt"]["file_sha256"]
        != _sha256(boundary_receipt_path)
    ):
        raise RuntimeError("baseline receipt points to another boundary file")
    trajectories, admission = _load_development_partition(
        dataset_root=dataset_root,
        boundary=boundary,
    )
    features = cheap_features(build_feature_surface(trajectories))
    baseline_report = baseline_receipt["baseline_report"]
    diagnostics = run_baseline_diagnostics(
        trajectories,
        features,
        cv_manifest=boundary["development_cv_manifest"],
        group_aliases=boundary["near_duplicate_manifest"][
            "group_aliases"
        ],
        expected_full_oof_sha256=baseline_report[
            "selected_oof_score_payload_sha256"
        ],
    )

    module_path = Path(__file__).with_name(
        "process_triage_baseline_diagnostics.py"
    )
    runner_path = Path(__file__).resolve()
    payload: dict[str, Any] = {
        "schema_version": (
            "pale_ale_process_triage_baseline_diagnostic_receipt_v0.1"
        ),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "epistemic_status": "development_baseline_diagnostics_only",
        "repository_commit": repository_commit,
        "boundary_receipt": {
            "path": str(boundary_receipt_path.resolve()),
            "file_sha256": _sha256(boundary_receipt_path),
            "payload_sha256": boundary["receipt_payload_sha256"],
        },
        "baseline_receipt": {
            "path": str(baseline_receipt_path.resolve()),
            "file_sha256": _sha256(baseline_receipt_path),
            "payload_sha256": baseline_receipt["receipt_payload_sha256"],
            "repository_commit": baseline_receipt["repository_commit"],
        },
        "dataset_admission": admission,
        "diagnostics": diagnostics,
        "authorization": {
            "structural_signal": False,
            "prospective_locked_partition_scoring": False,
            "locked_evaluation": False,
            "public_claim": False,
        },
        "implementation_sha256": {
            module_path.name: _sha256(module_path),
            runner_path.name: _sha256(runner_path),
        },
    }
    payload["receipt_payload_sha256"] = hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--boundary-receipt", type=Path, required=True)
    parser.add_argument("--baseline-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-commit", required=True)
    args = parser.parse_args()

    receipt = build_receipt(
        dataset_root=args.dataset_root.resolve(),
        boundary_receipt_path=args.boundary_receipt.resolve(),
        baseline_receipt_path=args.baseline_receipt.resolve(),
        repository_commit=str(args.repository_commit),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    diagnostics = receipt["diagnostics"]
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "receipt_payload_sha256": receipt[
                    "receipt_payload_sha256"
                ],
                "full_baseline_hash_reproduced": diagnostics[
                    "full_baseline_hash_reproduced"
                ],
                "configuration_count": len(
                    diagnostics["configuration_reports"]
                ),
                "structural_signal_opened": False,
                "prospective_locked_partition_scored": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
