"""Locked B2a primitives and fail-closed conditional execution entry point."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tools.gate13_causal_return.phase2_common import read_json, write_json
from tools.gate13_causal_return.validate_phase2_locks import validate_phase2_locks

from .operator_core import QUALIFIED, polar_record
from .source_sufficiency import assess_source_sufficiency


def analytic_reconstruction_tolerances(rank: int) -> dict[str, float]:
    """Float64/SVD/three-product tolerance fixed without observing outcomes."""
    if int(rank) <= 0:
        raise ValueError("rank must be positive")
    epsilon = float(np.finfo(np.float64).eps)
    factor = 512.0 * epsilon * max(1, int(rank))
    return {
        "matrix_atol_scale": factor,
        "matrix_rtol": factor,
        "scalar_atol_scale": 2.0 * factor,
        "scalar_rtol": 2.0 * factor,
    }


def reconstruct_edge_holonomy(
    raw_edges_in_traversal_order: Sequence[np.ndarray],
    *,
    rank_tolerance: float,
) -> np.ndarray:
    if len(raw_edges_in_traversal_order) != 3:
        raise ValueError("an explicit triangle must contain exactly three ordered edges")
    factors: list[np.ndarray] = []
    for edge in raw_edges_in_traversal_order:
        record = polar_record(
            np.asarray(edge, dtype=np.float64),
            rank_tolerance=rank_tolerance,
            condition_ceiling=float("inf"),
        )
        if record["status"] != QUALIFIED:
            raise ValueError("RANK_DEFICIENT")
        factors.append(np.asarray(record["O"], dtype=np.float64))
    return factors[2] @ factors[1] @ factors[0]


def reconstruction_integrity(
    reconstructed: np.ndarray,
    stored: np.ndarray,
    stored_legacy_scalar: float,
) -> dict[str, Any]:
    actual = np.asarray(reconstructed, dtype=np.float64)
    expected = np.asarray(stored, dtype=np.float64)
    if actual.shape != expected.shape or actual.ndim != 2 or actual.shape[0] != actual.shape[1]:
        raise ValueError("holonomy matrices must have the same square shape")
    rank = actual.shape[0]
    tolerance = analytic_reconstruction_tolerances(rank)
    matrix_scale = max(1.0, float(np.linalg.norm(actual, ord="fro")), float(np.linalg.norm(expected, ord="fro")))
    matrix_error = float(np.linalg.norm(actual - expected, ord="fro"))
    matrix_limit = tolerance["matrix_atol_scale"] * matrix_scale + tolerance["matrix_rtol"] * matrix_scale
    scalar = float(np.linalg.norm(actual - np.eye(rank), ord="fro"))
    scalar_scale = max(1.0, math.sqrt(rank), abs(scalar), abs(float(stored_legacy_scalar)))
    scalar_error = abs(scalar - float(stored_legacy_scalar))
    scalar_limit = tolerance["scalar_atol_scale"] * scalar_scale + tolerance["scalar_rtol"] * scalar_scale
    return {
        "status": "PASS" if matrix_error <= matrix_limit and scalar_error <= scalar_limit else "RECONSTRUCTION_FAIL",
        "matrix_error_fro": matrix_error,
        "matrix_limit": matrix_limit,
        "legacy_scalar_reconstructed": scalar,
        "legacy_scalar_error": scalar_error,
        "legacy_scalar_limit": scalar_limit,
    }


def normalized_legacy_scalar(value: float, rank: int) -> float:
    if rank <= 0:
        raise ValueError("rank must be positive")
    normalized = float(value) / (2.0 * math.sqrt(rank))
    if normalized < -1.0e-12 or normalized > 1.0 + 1.0e-12:
        raise ValueError("legacy scalar lies outside the theoretical orthogonal range")
    return min(1.0, max(0.0, normalized))


def primary_scalar_pairs(
    rows: Sequence[Mapping[str, Any]], *, bin_width: float
) -> list[tuple[str, str]]:
    """Stable adjacent one-to-one matching within source-run/rank/fixed bins."""
    if not 0.0 < float(bin_width) <= 1.0:
        raise ValueError("bin_width must be in (0, 1]")
    groups: dict[tuple[str, int, int], list[tuple[float, str]]] = {}
    for row in rows:
        run_id = str(row["run_id"])
        rank = int(row["rank"])
        normalized = normalized_legacy_scalar(float(row["legacy_scalar"]), rank)
        bin_index = min(int(normalized / bin_width), int(math.ceil(1.0 / bin_width)) - 1)
        groups.setdefault((run_id, rank, bin_index), []).append(
            (normalized, str(row["cycle_id"]))
        )
    pairs: list[tuple[str, str]] = []
    for key in sorted(groups):
        ordered = sorted(groups[key], key=lambda item: (item[0], item[1]))
        pairs.extend(
            (ordered[index][1], ordered[index + 1][1])
            for index in range(0, len(ordered) - 1, 2)
        )
    flat = [cycle_id for pair in pairs for cycle_id in pair]
    if len(flat) != len(set(flat)):
        raise ValueError("primary matcher reused a triangle")
    return pairs


def singular_spectrum_distance(first: Sequence[float], second: Sequence[float]) -> float:
    left = np.sort(np.asarray(first, dtype=np.float64))[::-1]
    right = np.sort(np.asarray(second, dtype=np.float64))[::-1]
    if left.shape != right.shape or left.ndim != 1 or left.size == 0:
        raise ValueError("singular spectra must have the same nonzero length")
    return float(np.linalg.norm(left - right) / math.sqrt(left.size))


def run_b2a(*, phase2_dir: Path, output_dir: Path) -> dict[str, Any]:
    validation = validate_phase2_locks(phase2_dir=phase2_dir, require_clean=True)
    if validation["status"] != "PASS":
        raise RuntimeError("dual-lock validation did not pass")
    lock = read_json(phase2_dir / "phase2_b2a_lock.json")
    sufficiency = assess_source_sufficiency(lock)
    if sufficiency["status"] != "PASS":
        result = {
            "schema_version": "gate13_phase2_b2a_result_v1",
            "status": "B2A_SPLIT_HALF_SOURCE_UNAVAILABLE",
            "B2A_SOURCE_SUFFICIENCY": "SPLIT_HALF_SOURCE_UNAVAILABLE",
            "B2A_RECONSTRUCTION": "UNOPENED",
            "B2A_STABILITY": "UNOPENED",
            "B2A_SCALAR_SHADOW": "UNOPENED",
            "B2A": "NOT_EXECUTED",
            "operator_outcomes_read": False,
            "semantic_labels_read": False,
            "track_a_outcomes_read": False,
            "source_sufficiency": sufficiency,
        }
        write_json(output_dir / "b2a_result.json", result)
        return result
    raise RuntimeError(
        "source sufficiency unexpectedly passed for a lock frozen as unavailable; "
        "stop for fresh review before reading operator outcomes"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_b2a(phase2_dir=args.phase2_dir, output_dir=args.output_dir)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
