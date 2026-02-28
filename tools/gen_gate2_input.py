#!/usr/bin/env python3
"""Generate synthetic Gate2 JSON v1 inputs for smoke testing."""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Gate2 trajectories (smooth or kink)."
    )
    parser.add_argument("--mode", choices=("smooth", "kink"), required=True)
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--kink-at", type=int, default=12)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.n_samples <= 0:
        parser.error("--n-samples must be > 0")
    if args.n_steps <= 0:
        parser.error("--n-steps must be > 0")
    if args.noise < 0.0:
        parser.error("--noise must be >= 0")
    if args.mode == "kink" and not (0 <= args.kink_at < args.n_steps):
        parser.error("--kink-at must satisfy 0 <= kink-at < n-steps")
    return args


def normalize(vec: List[float]) -> List[float]:
    norm_sq = 0.0
    for value in vec:
        if not math.isfinite(value):
            raise ValueError("non-finite value encountered before normalization")
        norm_sq += value * value
    norm = math.sqrt(norm_sq)
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("invalid norm during normalization")
    return [value / norm for value in vec]


def smooth_step(step_idx: int, n_steps: int, rng: random.Random, noise: float) -> List[float]:
    angle = (2.0 * math.pi * float(step_idx)) / float(n_steps)
    vec = [math.cos(angle), math.sin(angle)]
    vec.extend(rng.gauss(0.0, noise) for _ in range(6))
    return normalize(vec)


def sample_trajectory(
    mode: str,
    n_steps: int,
    seed: int,
    sample_id: int,
    noise: float,
    kink_at: int,
) -> List[List[float]]:
    rng = random.Random(seed + sample_id)
    steps: List[List[float]] = []
    for step_idx in range(n_steps):
        vec = smooth_step(step_idx, n_steps, rng, noise)
        if mode == "kink" and step_idx == kink_at:
            vec = [-value for value in vec]
        steps.append(vec)
    return steps


def validate_payload(payload: Dict) -> None:
    if not isinstance(payload.get("run_id"), str):
        raise ValueError("run_id must be a string")
    if not isinstance(payload.get("samples"), list):
        raise ValueError("samples must be a list")
    for sample in payload["samples"]:
        sample_id = sample.get("sample_id")
        if not isinstance(sample_id, int):
            raise ValueError(f"sample_id must be int, got: {type(sample_id)}")
        ans_vec8 = sample.get("ans_vec8")
        if not isinstance(ans_vec8, list):
            raise ValueError(f"ans_vec8 must be list for sample {sample_id}")
        for row_idx, row in enumerate(ans_vec8):
            if not isinstance(row, list) or len(row) != 8:
                raise ValueError(
                    f"ans_vec8[{row_idx}] must be an array of exactly 8 floats "
                    f"(sample_id={sample_id})"
                )
            for col_idx, value in enumerate(row):
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise ValueError(
                        f"non-finite/non-numeric value at sample_id={sample_id}, "
                        f"row={row_idx}, col={col_idx}"
                    )


def main() -> int:
    args = parse_args()
    samples = []
    for sample_id in range(args.n_samples):
        ans_vec8 = sample_trajectory(
            mode=args.mode,
            n_steps=args.n_steps,
            seed=args.seed,
            sample_id=sample_id,
            noise=args.noise,
            kink_at=args.kink_at,
        )
        samples.append(
            {
                "sample_id": sample_id,
                "ans_vec8": ans_vec8,
                "sample_label": None,
                "answer_length": args.n_steps,
            }
        )

    payload = {
        "run_id": args.run_id,
        "explicitly_unrelated_sample_ids": [],
        "samples": samples,
    }
    validate_payload(payload)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text + "\n")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
