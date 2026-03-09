#!/usr/bin/env python3
"""Run Gate4 full-batch ingestion and artifact-only sufficiency diagnostics."""

import argparse
import datetime as dt
import subprocess
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build full Gate4 artifacts from existing CFA sample dirs and run "
            "artifact-only negative stability diagnostics."
        )
    )
    parser.add_argument(
        "--samples-root",
        default="runs/cfa_batch_primaryE/samples",
    )
    parser.add_argument(
        "--cfa-jsonl",
        default="data/cfa/cfa_v1.jsonl",
    )
    parser.add_argument(
        "--spec-path",
        default="SPEC.internal.draft.md",
    )
    parser.add_argument(
        "--out-dir",
        default="runs/gate4_artifact_sufficiency",
    )
    parser.add_argument(
        "--dataset-revision-id",
        default="cfa_v1_full200_gate4_v1",
    )
    parser.add_argument(
        "--run-id",
        default="gate4_artifact_sufficiency",
    )
    parser.add_argument(
        "--evaluation-mode-id",
        default="supervised_v1",
    )
    parser.add_argument(
        "--out-report",
        default=(
            f"attestations/triality/gate4_validation/"
            f"{dt.date.today().isoformat()}_gate4_negative_stability_from_artifacts.txt"
        ),
    )
    parser.add_argument("--top-samples", type=int, default=10)
    parser.add_argument("--top-transitions", type=int, default=5)
    return parser.parse_args()


def run_command(cmd: Sequence[str]) -> None:
    completed = subprocess.run(
        list(cmd),
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed rc={completed.returncode}: {' '.join(cmd)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    if completed.stdout.strip():
        print(completed.stdout.strip())


def main() -> int:
    args = parse_args()
    batch_cmd = [
        "python",
        "tools/run_gate4_batch_ingestion.py",
        "--samples-root",
        args.samples_root,
        "--all-samples",
        "--cfa-jsonl",
        args.cfa_jsonl,
        "--spec-path",
        args.spec_path,
        "--out-dir",
        args.out_dir,
        "--run-id",
        args.run_id,
        "--dataset-revision-id",
        args.dataset_revision_id,
        "--evaluation-mode-id",
        args.evaluation_mode_id,
    ]
    run_command(batch_cmd)

    gate4_out_dir = (REPO_ROOT / args.out_dir / "gate4_out").as_posix()
    report_cmd = [
        "python",
        "tools/check_gate4_negative_stability.py",
        "--gate4-out-dir",
        gate4_out_dir,
        "--out",
        args.out_report,
        "--top-samples",
        str(args.top_samples),
        "--top-transitions",
        str(args.top_transitions),
    ]
    run_command(report_cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
