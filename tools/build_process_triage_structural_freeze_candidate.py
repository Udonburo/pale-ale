#!/usr/bin/env python3
"""Build the pre-execution freeze receipt for one structural family."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from process_triage_structural_signal import (
    structural_family_manifest,
)
from run_process_triage_baseline_development import (
    _load_boundary_receipt,
)


RECEIPT_SCHEMA_VERSION = (
    "pale_ale_process_triage_structural_freeze_v0.1"
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


def _load_receipt(
    path: Path,
    *,
    expected_schema: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != expected_schema:
        raise RuntimeError(f"unexpected receipt schema: {path}")
    authorization = payload.get("authorization", {})
    if authorization.get("locked_evaluation") is not False:
        raise RuntimeError(
            f"input receipt unexpectedly authorizes locked work: {path}"
        )
    return payload


def build_receipt(
    *,
    boundary_receipt_path: Path,
    diagnostic_receipt_path: Path,
    power_receipt_path: Path,
    repository_commit: str,
) -> dict[str, Any]:
    boundary = _load_boundary_receipt(boundary_receipt_path)
    diagnostics = _load_receipt(
        diagnostic_receipt_path,
        expected_schema=(
            "pale_ale_process_triage_baseline_diagnostic_receipt_v0.1"
        ),
    )
    power = _load_receipt(
        power_receipt_path,
        expected_schema=(
            "pale_ale_process_triage_power_receipt_v0.1"
        ),
    )
    if (
        diagnostics["boundary_receipt"]["file_sha256"]
        != _sha256(boundary_receipt_path)
        or power["boundary_receipt"]["file_sha256"]
        != _sha256(boundary_receipt_path)
    ):
        raise RuntimeError(
            "structural freeze inputs do not share the sealed boundary"
        )
    if not diagnostics["diagnostics"][
        "full_baseline_hash_reproduced"
    ]:
        raise RuntimeError("sealed full baseline hash did not reproduce")
    if power["simulation"]["interpretation_boundary"][
        "structural_signal_opened"
    ]:
        raise RuntimeError("power receipt unexpectedly opened the signal")

    manifest = structural_family_manifest()
    if manifest["development_candidate_family_count"] != 1:
        raise RuntimeError("exactly one structural family must be frozen")
    if manifest["learned_parameters"]:
        raise RuntimeError(
            "the selected bounded family unexpectedly learns parameters"
        )
    if manifest["locked_evaluation_authorized"]:
        raise RuntimeError(
            "the structural family manifest opens locked evaluation"
        )

    module_path = Path(__file__).with_name(
        "process_triage_structural_signal.py"
    )
    test_path = Path(__file__).with_name(
        "test_process_triage_structural_signal.py"
    )
    builder_path = Path(__file__).resolve()
    payload: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "epistemic_status": (
            "structural_family_frozen_development_execution_unopened"
        ),
        "repository_commit": repository_commit,
        "governing_contract": (
            "PROCESS_TRIAGE_CONTRACT_v0.2.md"
        ),
        "input_receipts": {
            "boundary": {
                "path": str(boundary_receipt_path.resolve()),
                "file_sha256": _sha256(boundary_receipt_path),
                "payload_sha256": boundary[
                    "receipt_payload_sha256"
                ],
            },
            "baseline_diagnostics": {
                "path": str(diagnostic_receipt_path.resolve()),
                "file_sha256": _sha256(diagnostic_receipt_path),
                "payload_sha256": diagnostics[
                    "receipt_payload_sha256"
                ],
            },
            "power_simulation": {
                "path": str(power_receipt_path.resolve()),
                "file_sha256": _sha256(power_receipt_path),
                "payload_sha256": power[
                    "receipt_payload_sha256"
                ],
            },
        },
        "manual_pre_freeze_schema_inspection": {
            "purpose": (
                "confirm artifact/tool-call shapes without viewing labels"
            ),
            "feature_surface_only": True,
            "outcomes_viewed": False,
            "trajectory_ids": [
                "bfcl:query:9:sample:0",
                "gaia_dev:query:2:sample:0",
                "hotpotqa:query:0:sample:0",
                "tau2:query:1:sample:0",
            ],
            "fields_viewed": [
                "eligible_index",
                "artifact_type",
                "tool_names",
                "content_prefix",
                "canonical_step_signature_prefix",
            ],
        },
        "structural_family_manifest": manifest,
        "development_degrees_of_freedom": {
            "candidate_family_count": 1,
            "primary_scalar_count": 1,
            "learned_structural_coefficients": 0,
            "structural_hyperparameter_grid": [],
            "window_choices": [],
            "threshold_choices": [],
            "embedding_choices": [],
            "normalization_choices": [],
            "allowed_model_tuning": (
                "existing baseline C grid and frozen four-fold "
                "development CV only"
            ),
            "post_result_signal_replacement": False,
        },
        "authorization": {
            "compute_primary_signal_on_development_surface": True,
            "compute_frozen_controls_on_development_surface": True,
            "fit_baseline_plus_primary_on_development_surface": True,
            "inspect_prospective_locked_outcomes": False,
            "score_prospective_locked_partition": False,
            "locked_evaluation": False,
            "public_claim": False,
        },
        "required_development_readout": {
            "primary": (
                "baseline plus task-anchored triangle excess versus "
                "sealed full cheap baseline"
            ),
            "standalone": "task-anchored triangle excess alone",
            "controls": [
                "score_order_shuffle",
                "dependency_cycle_randomization",
                "label_permutation",
            ],
            "required_ablations": [
                "remove_trajectory_length",
                "remove_normalized_position",
                "remove_retry_and_tool_error_features",
            ],
            "guardrails": [
                "clean_row_allocation",
                "clean_trajectory_alert_rate",
            ],
        },
        "implementation_sha256": {
            module_path.name: _sha256(module_path),
            test_path.name: _sha256(test_path),
            builder_path.name: _sha256(builder_path),
        },
    }
    payload["receipt_payload_sha256"] = hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--boundary-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--diagnostic-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--power-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-commit", required=True)
    args = parser.parse_args()

    receipt = build_receipt(
        boundary_receipt_path=args.boundary_receipt.resolve(),
        diagnostic_receipt_path=args.diagnostic_receipt.resolve(),
        power_receipt_path=args.power_receipt.resolve(),
        repository_commit=str(args.repository_commit),
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
                "family_id": receipt["structural_family_manifest"][
                    "family_id"
                ],
                "receipt_payload_sha256": receipt[
                    "receipt_payload_sha256"
                ],
                "development_execution_authorized": receipt[
                    "authorization"
                ][
                    "compute_primary_signal_on_development_surface"
                ],
                "locked_evaluation": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
