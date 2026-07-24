#!/usr/bin/env python3
"""Run the frozen structural family on development clusters only."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from process_triage_evaluator import (
    build_feature_surface,
    cheap_features,
)
from process_triage_structural_development import (
    run_structural_development,
)
from process_triage_structural_signal import (
    build_structural_surface,
    structural_surface_receipt,
    task_anchored_triangle_excess,
)
from run_process_triage_baseline_development import (
    _load_boundary_receipt,
    _load_development_partition,
)


RECEIPT_SCHEMA_VERSION = (
    "pale_ale_process_triage_structural_development_receipt_v0.1"
)
STRUCTURAL_FREEZE_SCHEMA_VERSION = (
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


def _load_json(path: Path, *, schema: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != schema:
        raise RuntimeError(f"unexpected receipt schema: {path}")
    return payload


def _score_distribution(rows: tuple) -> dict[str, Any]:
    values = np.asarray(
        [float(row.score) for row in rows],
        dtype=np.float64,
    )
    if values.size == 0:
        raise RuntimeError("structural development emitted no rows")
    return {
        "row_count": int(values.size),
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "maximum": float(np.max(values)),
        "zero_count": int(np.sum(values == 0.0)),
        "unique_value_count": int(len(np.unique(values))),
    }


def _reproducibility_projection(
    report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "score_payload_sha256": report["score_payload_sha256"],
        "selected_score_payload_sha256": report[
            "selected_score_payload_sha256"
        ],
        "sealed_baseline_oof": report["sealed_baseline"][
            "selected_oof_score_payload_sha256"
        ],
        "primary_augmented_oof": report["primary_augmented"][
            "selected_oof_score_payload_sha256"
        ],
        "primary_augmented_refit": report["primary_augmented"][
            "refit_prediction_payload_sha256"
        ],
    }


def build_receipt(
    *,
    dataset_root: Path,
    boundary_receipt_path: Path,
    baseline_receipt_path: Path,
    structural_freeze_receipt_path: Path,
    repository_commit: str,
) -> dict[str, Any]:
    boundary = _load_boundary_receipt(boundary_receipt_path)
    baseline_receipt = _load_json(
        baseline_receipt_path,
        schema="pale_ale_process_triage_baseline_receipt_v0.1",
    )
    structural_freeze = _load_json(
        structural_freeze_receipt_path,
        schema=STRUCTURAL_FREEZE_SCHEMA_VERSION,
    )
    if (
        structural_freeze["input_receipts"]["boundary"][
            "file_sha256"
        ]
        != _sha256(boundary_receipt_path)
    ):
        raise RuntimeError("structural freeze uses another boundary")
    if not structural_freeze["authorization"][
        "compute_primary_signal_on_development_surface"
    ]:
        raise RuntimeError(
            "structural freeze does not authorize development execution"
        )
    if structural_freeze["authorization"]["locked_evaluation"]:
        raise RuntimeError(
            "structural freeze unexpectedly authorizes locked execution"
        )

    structural_module_path = Path(__file__).with_name(
        "process_triage_structural_signal.py"
    )
    frozen_module_hash = structural_freeze["implementation_sha256"][
        structural_module_path.name
    ]
    if _sha256(structural_module_path) != frozen_module_hash:
        raise RuntimeError(
            "frozen structural implementation changed before execution"
        )

    trajectories, admission = _load_development_partition(
        dataset_root=dataset_root,
        boundary=boundary,
    )
    cheap = cheap_features(build_feature_surface(trajectories))
    structural_surface = build_structural_surface(trajectories)
    surface_receipt = structural_surface_receipt(
        structural_surface
    )
    modes = {
        mode: task_anchored_triangle_excess(
            structural_surface,
            mode=mode,
        )
        for mode in (
            "primary",
            "score_order_shuffle",
            "dependency_cycle_randomization",
        )
    }
    report = run_structural_development(
        trajectories,
        cheap,
        cv_manifest=boundary["development_cv_manifest"],
        group_aliases=boundary["near_duplicate_manifest"][
            "group_aliases"
        ],
        structural_modes=modes,
    )
    if (
        report["sealed_baseline"][
            "selected_oof_score_payload_sha256"
        ]
        != baseline_receipt["baseline_report"][
            "selected_oof_score_payload_sha256"
        ]
    ):
        raise RuntimeError(
            "structural runner did not reproduce the sealed baseline"
        )
    repeated = run_structural_development(
        trajectories,
        cheap,
        cv_manifest=boundary["development_cv_manifest"],
        group_aliases=boundary["near_duplicate_manifest"][
            "group_aliases"
        ],
        structural_modes=modes,
    )
    first_projection = _reproducibility_projection(report)
    second_projection = _reproducibility_projection(repeated)
    if first_projection != second_projection:
        raise RuntimeError(
            "structural development did not reproduce exactly"
        )

    development_module_path = Path(__file__).with_name(
        "process_triage_structural_development.py"
    )
    runner_path = Path(__file__).resolve()
    payload: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "epistemic_status": (
            "development_structural_result_not_confirmatory"
        ),
        "repository_commit": repository_commit,
        "input_receipts": {
            "boundary": {
                "path": str(boundary_receipt_path.resolve()),
                "file_sha256": _sha256(boundary_receipt_path),
                "payload_sha256": boundary[
                    "receipt_payload_sha256"
                ],
            },
            "baseline": {
                "path": str(baseline_receipt_path.resolve()),
                "file_sha256": _sha256(baseline_receipt_path),
                "payload_sha256": baseline_receipt[
                    "receipt_payload_sha256"
                ],
            },
            "structural_freeze": {
                "path": str(
                    structural_freeze_receipt_path.resolve()
                ),
                "file_sha256": _sha256(
                    structural_freeze_receipt_path
                ),
                "payload_sha256": structural_freeze[
                    "receipt_payload_sha256"
                ],
            },
        },
        "dataset_admission": admission,
        "exposure_transition": {
            "development_structural_signal_computed": True,
            "development_feature_label_association_computed": True,
            "prospective_locked_labelled_records_parsed": False,
            "prospective_locked_partition_scored": False,
            "locked_evaluation_authorized": False,
        },
        "structural_surface_receipt": surface_receipt,
        "structural_score_distributions": {
            mode: _score_distribution(rows)
            for mode, rows in modes.items()
        },
        "development_report": report,
        "repeat_execution": {
            "exact_projection_match": True,
            "projection": first_projection,
        },
        "interpretation_boundary": {
            "development_only": True,
            "confirmatory_result": False,
            "locked_generalization_estimated": False,
            "public_claim_authorized": False,
            "note": (
                "Development results decide whether this single frozen "
                "family is worth carrying to a later no-peek preflight. "
                "They do not estimate held-out value."
            ),
        },
        "authorization": {
            "inspect_prospective_locked_outcomes": False,
            "score_prospective_locked_partition": False,
            "locked_evaluation": False,
            "public_claim": False,
        },
        "implementation_sha256": {
            structural_module_path.name: _sha256(
                structural_module_path
            ),
            development_module_path.name: _sha256(
                development_module_path
            ),
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
    parser.add_argument(
        "--boundary-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--baseline-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--structural-freeze-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-commit", required=True)
    args = parser.parse_args()

    receipt = build_receipt(
        dataset_root=args.dataset_root.resolve(),
        boundary_receipt_path=args.boundary_receipt.resolve(),
        baseline_receipt_path=args.baseline_receipt.resolve(),
        structural_freeze_receipt_path=(
            args.structural_freeze_receipt.resolve()
        ),
        repository_commit=str(args.repository_commit),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = receipt["development_report"]
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "receipt_payload_sha256": receipt[
                    "receipt_payload_sha256"
                ],
                "baseline_recall": report["sealed_baseline"][
                    "selected_oof_evaluation"
                ]["first_actionable_defect_recall"],
                "augmented_recall": report["primary_augmented"][
                    "selected_oof_evaluation"
                ]["first_actionable_defect_recall"],
                "development_recall_increment": report[
                    "primary_increment_vs_sealed_baseline"
                ]["first_actionable_defect_recall"],
                "repeat_execution_match": True,
                "prospective_locked_partition_scored": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
