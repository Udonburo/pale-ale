#!/usr/bin/env python3
"""Execute or resume a deterministic Gate12C-2 development shard plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gate12c2_synthetic_lab as lab
from gate12c2_development_shards import (
    build_development_shard_plan,
    execute_development_shard_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regime",
        required=True,
        choices=(
            "S0_true_null",
            "S1_known_reverse_shared_node_coupling",
            "S2_null_inflation",
        ),
    )
    parser.add_argument("--master-seed", required=True)
    parser.add_argument("--outer-start", type=int, default=0)
    parser.add_argument("--outer-count", type=int, required=True)
    parser.add_argument("--inner-valid-draw-count", type=int, required=True)
    parser.add_argument("--effect-strength", type=float)
    parser.add_argument("--max-draw-attempts", type=int)
    parser.add_argument(
        "--minimum-log-null-inflation",
        type=float,
        default=lab.S2_MIN_LOG_NULL_INFLATION,
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.outer_start < 0 or args.outer_count <= 0:
        raise SystemExit("outer-start must be nonnegative and outer-count positive")
    indices = range(
        args.outer_start,
        args.outer_start + args.outer_count,
    )
    plan = build_development_shard_plan(
        regime_id=args.regime,
        master_seed=args.master_seed,
        outer_experiment_indices=indices,
        block_count=lab.reference_block_count_schedule(),
        inner_valid_draw_count=args.inner_valid_draw_count,
        effect_strength=args.effect_strength,
        max_draw_attempts=args.max_draw_attempts,
        minimum_log_null_inflation=args.minimum_log_null_inflation,
    )
    result = execute_development_shard_plan(
        plan,
        output_dir=args.output_dir,
        worker_count=args.workers,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "plan_payload_sha256": result["plan_payload_sha256"],
                "scientific_projection_sha256": result[
                    "scientific_projection_sha256"
                ],
                "outer_experiment_count": result[
                    "outer_experiment_count"
                ],
                "all_outer_indices_present": result[
                    "all_outer_indices_present"
                ],
                "locked_execution_authorized": result[
                    "locked_execution_authorized"
                ],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
