#!/usr/bin/env python3
"""One-shot Gate8 scale-up runner."""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full Gate8 scale-up pipeline: constitution scaffold, materialization, "
            "and fixed-candidate execution."
        )
    )
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--samples-per-cell", type=int, required=True)
    parser.add_argument("--model-id", help="Optional explicit HF model id.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rendering-family", default="archive_v1")
    parser.add_argument(
        "--skip-execution",
        action="store_true",
        help="Only generate scaffold + materialized benchmark, do not run candidate execution.",
    )
    return parser.parse_args()


def run_subprocess(command: Sequence[str]) -> None:
    completed = subprocess.run(
        list(command),
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed rc={completed.returncode}: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def main() -> int:
    args = parse_args()
    constitution_dir = Path("runs") / f"{args.run_prefix}_constitution"
    benchmark_dir = Path("runs") / f"{args.run_prefix}_benchmark"
    execution_dir = Path("runs") / f"{args.run_prefix}_candidate_execution"

    run_subprocess(
        [
            sys.executable,
            str((REPO_ROOT / "tools" / "generate_gate8_semiclosed_conflict.py").resolve()),
            "--out-dir",
            str(constitution_dir),
            "--run-id",
            constitution_dir.name,
            "--samples-per-cell",
            str(args.samples_per_cell),
            "--rendering-family",
            str(args.rendering_family),
        ]
    )
    run_subprocess(
        [
            sys.executable,
            str((REPO_ROOT / "tools" / "materialize_gate8_semiclosed_conflict.py").resolve()),
            "--constitution-dir",
            str(constitution_dir),
            "--out-dir",
            str(benchmark_dir),
            "--run-id",
            benchmark_dir.name,
            "--seed",
            str(args.seed),
        ]
    )
    if not args.skip_execution:
        command = [
            sys.executable,
            str((REPO_ROOT / "tools" / "run_gate8_candidate_batch.py").resolve()),
            "--benchmark-dir",
            str(benchmark_dir),
            "--out-dir",
            str(execution_dir),
            "--device",
            args.device,
            "--topk",
            str(args.topk),
            "--seed",
            str(args.seed),
        ]
        if args.model_id:
            command.extend(["--model-id", args.model_id])
        run_subprocess(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
