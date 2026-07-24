#!/usr/bin/env python3
"""Build the local-only AgentProcessBench split/freeze candidate receipt.

This command is label-marginal aware but does not compute a feature-label
association, fit a baseline, define a structural signal, or execute a locked
evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from process_triage_evaluator import (
    agent_process_bench_freeze_candidate_split,
    build_feature_surface,
    dataset_admission_summary,
    development_cluster_cv_manifest,
    feature_surface_receipt,
    load_agent_process_bench_jsonl,
    near_duplicate_group_manifest,
    process_triage_freeze_candidate_specification,
)


DATASET_COMMIT = "0a42606b178a8c69d40c5765dc05c342f921e578"
EXPECTED_DATA_HASHES = {
    "bfcl.jsonl": (
        "6aee0b71eff7feb872c6b54d962f8831b56f7bebf770cba9cb657f219afb6fe5"
    ),
    "gaia_dev.jsonl": (
        "f7b75c668fc1e6ad943e8f6d93a21a9a8f5f076841e1230e1ed2fc6d05ce8192"
    ),
    "hotpotqa.jsonl": (
        "160eef2ded872d8dc6ddf4cee5752295fda691b6a9e99b5bafc2d484e6309c57"
    ),
    "tau2.jsonl": (
        "6f22818ff88822512767fe735f56e7b78bed4b9aaf26959144feb92f127e1e95"
    ),
}


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


def build_receipt(
    *,
    dataset_root: Path,
    split_seed: str,
    repository_commit: str,
) -> dict[str, Any]:
    if not repository_commit.strip():
        raise ValueError("repository_commit must be nonempty")
    files = {
        name: dataset_root / name for name in sorted(EXPECTED_DATA_HASHES)
    }
    actual_hashes = {name: _sha256(path) for name, path in files.items()}
    if actual_hashes != EXPECTED_DATA_HASHES:
        raise RuntimeError(
            "dataset snapshot hash mismatch: "
            f"expected={EXPECTED_DATA_HASHES!r} actual={actual_hashes!r}"
        )

    trajectories = tuple(
        trajectory
        for name, path in files.items()
        for trajectory in load_agent_process_bench_jsonl(
            path,
            domain=Path(name).stem,
        )
    )
    feature_receipt = feature_surface_receipt(
        build_feature_surface(trajectories)
    )
    duplicate_manifest = near_duplicate_group_manifest(trajectories)
    split = agent_process_bench_freeze_candidate_split(
        trajectories,
        split_seed=split_seed,
        group_aliases=duplicate_manifest["group_aliases"],
    )
    development_cv = development_cluster_cv_manifest(
        trajectories,
        development_cluster_ids=split["development_cluster_ids"],
        group_aliases=duplicate_manifest["group_aliases"],
    )
    specification = process_triage_freeze_candidate_specification()
    evaluator_path = Path(__file__).with_name("process_triage_evaluator.py")
    builder_path = Path(__file__).resolve()
    payload: dict[str, Any] = {
        "schema_version": "pale_ale_process_triage_freeze_receipt_v0.2",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "epistemic_status": (
            "freeze_candidate_feature_label_association_unopened"
        ),
        "authorization": {
            "development_feature_label_association": False,
            "structural_signal": False,
            "locked_evaluation": False,
            "public_claim": False,
        },
        "repository_commit": repository_commit,
        "dataset_snapshot": {
            "github_repository": "https://github.com/RUCBM/AgentProcessBench",
            "hugging_face_dataset": (
                "https://huggingface.co/datasets/LulaCola/AgentProcessBench"
            ),
            "commit": DATASET_COMMIT,
            "license": "MIT",
            "file_sha256": actual_hashes,
        },
        "exposure_boundary": {
            "label_marginals_known": True,
            "individual_parser_sanity_labels_previously_viewed": True,
            "feature_label_association_computed": False,
            "baseline_performance_computed": False,
            "structural_signal_opened": False,
            "locked_split_individual_outcomes_viewed": False,
        },
        "admission_summary": dataset_admission_summary(trajectories),
        "feature_firewall": feature_receipt,
        "near_duplicate_manifest": duplicate_manifest,
        "split_seed": split_seed,
        "split": split,
        "development_cv_manifest": development_cv,
        "evaluation_specification": specification,
        "implementation_sha256": {
            str(evaluator_path.name): _sha256(evaluator_path),
            str(builder_path.name): _sha256(builder_path),
        },
    }
    payload["receipt_payload_sha256"] = hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-seed", required=True)
    parser.add_argument("--repository-commit", required=True)
    args = parser.parse_args()

    receipt = build_receipt(
        dataset_root=args.dataset_root.resolve(),
        split_seed=str(args.split_seed),
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
                "receipt_payload_sha256": receipt[
                    "receipt_payload_sha256"
                ],
                "development_cluster_count": receipt["split"][
                    "development_cluster_count"
                ],
                "locked_cluster_count": receipt["split"][
                    "locked_cluster_count"
                ],
                "feature_label_association_computed": False,
                "locked_evaluation_authorized": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
