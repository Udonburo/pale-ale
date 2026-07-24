#!/usr/bin/env python3
"""Run outcome-association-blind power planning on the sealed cluster layout."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from build_process_triage_freeze_candidate import (
    DATASET_COMMIT,
    EXPECTED_DATA_HASHES,
)
from process_triage_evaluator import (
    agent_process_bench_task_surface_group_id,
)
from process_triage_power_simulation import (
    PowerCluster,
    run_power_simulation,
)
from run_process_triage_baseline_development import (
    _load_boundary_receipt,
)


RECEIPT_SCHEMA_VERSION = (
    "pale_ale_process_triage_power_receipt_v0.1"
)
PRE_OUTCOME_TASK_FIELDS = (
    "question",
    "task_description",
    "data_source",
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


def _pre_outcome_task_view(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy only fields authorized for cluster membership recovery."""
    return {
        field: record.get(field) for field in PRE_OUTCOME_TASK_FIELDS
    }


def _load_locked_cluster_surface(
    *,
    dataset_root: Path,
    boundary: Mapping[str, Any],
) -> tuple[tuple[PowerCluster, ...], dict[str, Any]]:
    files = {
        name: dataset_root / name for name in sorted(EXPECTED_DATA_HASHES)
    }
    actual_hashes = {name: _sha256(path) for name, path in files.items()}
    if actual_hashes != EXPECTED_DATA_HASHES:
        raise RuntimeError(
            "dataset snapshot hash mismatch before power planning"
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
    cluster_counts: Counter[tuple[str, str]] = Counter()
    membership = Counter()
    for name, path in files.items():
        domain = Path(name).stem
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw_record = json.loads(line)
                task_view = _pre_outcome_task_view(raw_record)
                exact_group_id = (
                    agent_process_bench_task_surface_group_id(
                        task_view,
                        domain=domain,
                    )
                )
                cluster_id = aliases.get(exact_group_id, exact_group_id)
                if cluster_id in locked_ids:
                    cluster_counts[(cluster_id, domain)] += 1
                    membership["prospective_locked"] += 1
                elif cluster_id in development_ids:
                    membership["development"] += 1
                else:
                    membership["unknown"] += 1

    clusters = tuple(
        PowerCluster(
            cluster_id=cluster_id,
            domain=domain,
            trajectory_count=count,
        )
        for (cluster_id, domain), count in sorted(
            cluster_counts.items()
        )
    )
    expected_membership = {
        "development": int(
            boundary["split"]["development_trajectory_count"]
        ),
        "prospective_locked": int(
            boundary["split"]["locked_trajectory_count"]
        ),
        "unknown": 0,
    }
    actual_membership = {
        key: int(membership[key]) for key in expected_membership
    }
    if actual_membership != expected_membership:
        raise RuntimeError(
            "outcome-free membership recovery disagrees with the seal: "
            f"{actual_membership!r}"
        )
    if len(clusters) != int(boundary["split"]["locked_cluster_count"]):
        raise RuntimeError(
            "locked cluster count disagrees with the sealed split"
        )
    maximum = max(cluster.trajectory_count for cluster in clusters)
    if maximum != int(
        boundary["split"]["maximum_cluster_trajectory_count"]
    ):
        raise RuntimeError(
            "maximum cluster size disagrees with the sealed split"
        )
    return clusters, {
        "dataset_commit": DATASET_COMMIT,
        "file_sha256": actual_hashes,
        "fields_copied_from_raw_record": list(
            PRE_OUTCOME_TASK_FIELDS
        ),
        "raw_snapshot_co_locates_outcome_fields": True,
        "forbidden_outcome_fields_accessed": [],
        "outcome_fields_copied_or_dereferenced": False,
        "typed_labelled_trajectories_constructed": False,
        "human_locked_outcome_inspection": False,
        "membership_counts": actual_membership,
        "locked_cluster_count": len(clusters),
        "locked_trajectory_count": sum(
            cluster.trajectory_count for cluster in clusters
        ),
    }


def build_receipt(
    *,
    dataset_root: Path,
    boundary_receipt_path: Path,
    repository_commit: str,
    simulation_count: int,
    bootstrap_validation_simulations: int,
    bootstrap_validation_replicates: int,
) -> dict[str, Any]:
    boundary = _load_boundary_receipt(boundary_receipt_path)
    clusters, admission = _load_locked_cluster_surface(
        dataset_root=dataset_root,
        boundary=boundary,
    )
    domain_positive_prevalence = {
        domain: (
            float(summary["positive_trajectory_count"])
            / float(summary["trajectory_count"])
        )
        for domain, summary in sorted(
            boundary["admission_summary"]["domains"].items()
        )
    }
    simulation = run_power_simulation(
        clusters,
        domain_positive_prevalence=domain_positive_prevalence,
        simulation_count=simulation_count,
        bootstrap_validation_simulations=(
            bootstrap_validation_simulations
        ),
        bootstrap_validation_replicates=(
            bootstrap_validation_replicates
        ),
    )

    module_path = Path(__file__).with_name(
        "process_triage_power_simulation.py"
    )
    runner_path = Path(__file__).resolve()
    payload: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "epistemic_status": "design_power_simulation_only",
        "repository_commit": repository_commit,
        "boundary_receipt": {
            "path": str(boundary_receipt_path.resolve()),
            "file_sha256": _sha256(boundary_receipt_path),
            "payload_sha256": boundary["receipt_payload_sha256"],
        },
        "dataset_admission": admission,
        "marginal_information_source": {
            "source": "sealed_pre_association_boundary_receipt",
            "individual_outcomes": False,
            "feature_label_associations": False,
        },
        "simulation": simulation,
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
    parser.add_argument(
        "--boundary-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-commit", required=True)
    parser.add_argument(
        "--simulation-count",
        type=int,
        default=5_000,
    )
    parser.add_argument(
        "--bootstrap-validation-simulations",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--bootstrap-validation-replicates",
        type=int,
        default=999,
    )
    args = parser.parse_args()

    receipt = build_receipt(
        dataset_root=args.dataset_root.resolve(),
        boundary_receipt_path=args.boundary_receipt.resolve(),
        repository_commit=str(args.repository_commit),
        simulation_count=args.simulation_count,
        bootstrap_validation_simulations=(
            args.bootstrap_validation_simulations
        ),
        bootstrap_validation_replicates=(
            args.bootstrap_validation_replicates
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    simulation = receipt["simulation"]
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "receipt_payload_sha256": receipt[
                    "receipt_payload_sha256"
                ],
                "report_count": len(simulation["reports"]),
                "proxy_validation_within_0_05": simulation[
                    "bootstrap_validation"
                ]["planning_proxy_agreement_within_0_05"],
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
