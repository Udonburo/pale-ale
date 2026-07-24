#!/usr/bin/env python3
"""Fit the frozen cheap baseline on AgentProcessBench development clusters.

The runner reads the sealed partition receipt, admits full labels only for the
48 development clusters, and leaves the 141 prospective locked clusters
unparsed as typed labelled trajectories. It fits no structural signal and
does not score the prospective locked partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from build_process_triage_freeze_candidate import (
    DATASET_COMMIT,
    EXPECTED_DATA_HASHES,
)
from process_triage_baseline import fit_development_baseline
from process_triage_evaluator import (
    agent_process_bench_task_surface_group_id,
    build_feature_surface,
    cheap_features,
    parse_agent_process_bench_record,
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


def _load_boundary_receipt(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version")
        != "pale_ale_process_triage_freeze_receipt_v0.2"
    ):
        raise RuntimeError("unexpected process-triage boundary schema")
    if (
        payload.get("authorization", {}).get("locked_evaluation")
        is not False
    ):
        raise RuntimeError(
            "boundary receipt unexpectedly authorizes locked evaluation"
        )
    if (
        payload.get("exposure_boundary", {}).get(
            "feature_label_association_computed"
        )
        is not False
    ):
        raise RuntimeError(
            "boundary receipt is not the pre-association freeze candidate"
        )
    return payload


def _load_development_partition(
    *,
    dataset_root: Path,
    boundary: dict[str, Any],
) -> tuple[tuple, dict[str, Any]]:
    files = {
        name: dataset_root / name for name in sorted(EXPECTED_DATA_HASHES)
    }
    actual_hashes = {name: _sha256(path) for name, path in files.items()}
    if actual_hashes != EXPECTED_DATA_HASHES:
        raise RuntimeError(
            "dataset snapshot hash mismatch before baseline development"
        )

    aliases = {
        str(key): str(value)
        for key, value in boundary["near_duplicate_manifest"][
            "group_aliases"
        ].items()
    }
    development_ids = {
        str(value)
        for value in boundary["split"]["development_cluster_ids"]
    }
    locked_ids = {
        str(value) for value in boundary["split"]["locked_cluster_ids"]
    }
    if development_ids & locked_ids:
        raise RuntimeError("boundary receipt contains partition overlap")

    development = []
    membership_counts = {
        "development_record_count": 0,
        "prospective_locked_record_count": 0,
        "unknown_record_count": 0,
    }
    for name, path in files.items():
        domain = Path(name).stem
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                exact_group_id = (
                    agent_process_bench_task_surface_group_id(
                        record,
                        domain=domain,
                    )
                )
                cluster_id = aliases.get(exact_group_id, exact_group_id)
                if cluster_id in development_ids:
                    development.append(
                        parse_agent_process_bench_record(
                            record,
                            domain=domain,
                        )
                    )
                    membership_counts["development_record_count"] += 1
                elif cluster_id in locked_ids:
                    membership_counts[
                        "prospective_locked_record_count"
                    ] += 1
                else:
                    membership_counts["unknown_record_count"] += 1
    if membership_counts != {
        "development_record_count": 240,
        "prospective_locked_record_count": 760,
        "unknown_record_count": 0,
    }:
        raise RuntimeError(
            "dataset membership no longer matches the sealed 240/760 "
            f"partition: {membership_counts!r}"
        )
    return tuple(development), {
        "dataset_commit": DATASET_COMMIT,
        "file_sha256": actual_hashes,
        **membership_counts,
        "development_labelled_records_parsed": True,
        "prospective_locked_labelled_records_parsed_to_typed_trajectories": (
            False
        ),
        "prospective_locked_records_scored": False,
    }


def build_receipt(
    *,
    dataset_root: Path,
    boundary_receipt_path: Path,
    repository_commit: str,
) -> dict[str, Any]:
    if not repository_commit.strip():
        raise ValueError("repository_commit must be nonempty")
    boundary = _load_boundary_receipt(boundary_receipt_path)
    trajectories, admission = _load_development_partition(
        dataset_root=dataset_root,
        boundary=boundary,
    )
    aliases = boundary["near_duplicate_manifest"]["group_aliases"]
    features = cheap_features(build_feature_surface(trajectories))
    baseline_report = fit_development_baseline(
        trajectories,
        features,
        cv_manifest=boundary["development_cv_manifest"],
        group_aliases=aliases,
    )

    evaluator_path = Path(__file__).with_name(
        "process_triage_evaluator.py"
    )
    baseline_path = Path(__file__).with_name("process_triage_baseline.py")
    runner_path = Path(__file__).resolve()
    boundary_file_hash = _sha256(boundary_receipt_path)
    payload: dict[str, Any] = {
        "schema_version": "pale_ale_process_triage_baseline_receipt_v0.1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "epistemic_status": "development_baseline_only",
        "repository_commit": repository_commit,
        "boundary_receipt": {
            "path": str(boundary_receipt_path.resolve()),
            "file_sha256": boundary_file_hash,
            "payload_sha256": boundary["receipt_payload_sha256"],
            "repository_commit": boundary["repository_commit"],
        },
        "exposure_transition": {
            "development_feature_label_association_computed": True,
            "development_baseline_performance_computed": True,
            "structural_signal_opened": False,
            "prospective_locked_feature_label_association_computed": False,
            "prospective_locked_partition_scored": False,
            "locked_evaluation_authorized": False,
            "public_claim_authorized": False,
        },
        "dataset_admission": admission,
        "baseline_report": baseline_report,
        "implementation_sha256": {
            evaluator_path.name: _sha256(evaluator_path),
            baseline_path.name: _sha256(baseline_path),
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-commit", required=True)
    args = parser.parse_args()

    receipt = build_receipt(
        dataset_root=args.dataset_root.resolve(),
        boundary_receipt_path=args.boundary_receipt.resolve(),
        repository_commit=str(args.repository_commit),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = receipt["baseline_report"]
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "receipt_payload_sha256": receipt[
                    "receipt_payload_sha256"
                ],
                "development_cluster_count": report[
                    "development_cluster_count"
                ],
                "selected_regularization_c": report[
                    "selected_regularization_c"
                ],
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
