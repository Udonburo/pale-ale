#!/usr/bin/env python3
"""One bounded Gate12C-2 v2 balanced-null development prototype.

This program compares the retained iid N1 idea with a balanced derangement
schedule on fresh synthetic development data. It cannot alter the consumed
locked result, authorize another locked suite, or open real held-out data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from tools.gate12c2_minimal.generators import (
    Graph,
    edge_singular_values,
    edge_spectrum_error,
    generate_s0_cohort,
    generate_s1_cohort,
    graph_digest,
    independent_edge_reorientation,
    joint_realizability_error,
    rng,
)
from tools.gate12c2_minimal.io import (
    load_json,
    sha256_file,
    write_bytes_atomic,
    write_json_atomic,
)
from tools.gate12c2_minimal.metrics import ResidualDiagnostics, residual_diagnostics


SPEC_SCHEMA = "gate12c2_v2_balanced_prototype_spec_v0.1"
SHARD_SCHEMA = "gate12c2_v2_balanced_prototype_shard_v0.1"
ANALYSIS_SCHEMA = "gate12c2_v2_balanced_prototype_analysis_v0.1"
STATE_SCHEMA = "gate12c2_v2_balanced_prototype_state_v0.1"
CONFIGS = ("S0", "S1_PRIMARY", "S2")
SCHEDULES = ("iid", "balanced")
SURFACES = {
    "edge": slice(0, 9),
    "product": slice(9, 15),
    "gram": slice(15, 24),
}
FEATURE_NAMES = tuple(
    [f"edge_{edge}_{singular}" for edge in range(3) for singular in range(3)]
    + [f"product_{side}_{singular}" for side in ("left", "right") for singular in range(3)]
    + [f"block_gram_eigen_{index}" for index in range(9)]
)
QUANTILES = np.linspace(0.1, 0.9, 9)
ASSIGNMENT_METRIC_FIELDS = {
    "pair_collision_rate",
    "triple_collision_rate",
    "mean_exposure_cv",
    "max_exposure_cv",
    "max_exposure_range",
}
GEOMETRY_METRIC_FIELDS = {
    "max_quantile_difference",
    "median_quantile_difference",
    "cross_feature_correlation_max_difference",
    "cross_feature_correlation_frobenius",
}


class PrototypeError(RuntimeError):
    """Raised when the bounded prototype surface is inconsistent."""


def stable_seed(*parts: object) -> int:
    encoded = "\x1f".join(str(value) for value in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:16], "big")


def validate_spec(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PrototypeError("prototype spec must be an object")
    required = {
        "schema_version",
        "study_id",
        "epistemic_status",
        "seed_namespace",
        "source_locked_study_sha256",
        "configs",
        "case_ids",
        "dataset_count",
        "cohort_size",
        "draw_count",
        "checkpoints",
        "schedules",
        "q_values",
        "bootstrap_repeats",
        "decision_margins",
        "prototype_criteria",
        "resource_cap",
    }
    if set(value) != required or value.get("schema_version") != SPEC_SCHEMA:
        raise PrototypeError("prototype spec fields or schema differ")
    if value["epistemic_status"] != "post_locked_synthetic_development_only":
        raise PrototypeError("prototype must remain development-only")
    if value["configs"] != list(CONFIGS) or value["schedules"] != list(SCHEDULES):
        raise PrototypeError("prototype configs or schedules differ")
    if value["q_values"] != [1, 2]:
        raise PrototypeError("prototype requires q=[1,2]")
    for name in ("study_id", "seed_namespace", "source_locked_study_sha256"):
        if not isinstance(value[name], str) or not value[name]:
            raise PrototypeError(f"{name} must be nonempty text")
    source_hash = value["source_locked_study_sha256"]
    if len(source_hash) != 64 or any(character not in "0123456789abcdef" for character in source_hash):
        raise PrototypeError("source locked study hash differs")
    for name, minimum in (
        ("dataset_count", 2),
        ("cohort_size", 4),
        ("draw_count", 1),
        ("bootstrap_repeats", 64),
    ):
        if isinstance(value[name], bool) or not isinstance(value[name], int) or value[name] < minimum:
            raise PrototypeError(f"{name} differs")
    if value["cohort_size"] != 12 or value["draw_count"] % 11 != 0:
        raise PrototypeError("balanced schedule requires 12 graphs and complete 11-draw cycles")
    if value["checkpoints"] != [11, 22, 44] or value["draw_count"] != 44:
        raise PrototypeError("prototype checkpoints differ")
    case_ids = value["case_ids"]
    if (
        not isinstance(case_ids, list)
        or len(case_ids) != 6
        or len(set(case_ids)) != 6
        or any(not isinstance(case_id, str) or not case_id for case_id in case_ids)
    ):
        raise PrototypeError("prototype must declare six unique cases")
    expected_margin_fields = {
        "S0_equivalence_half_width",
        "S1_upper_limit",
        "S2_lower_limit",
    }
    if set(value["decision_margins"]) != expected_margin_fields:
        raise PrototypeError("decision margin fields differ")
    if any(not math.isfinite(float(item)) for item in value["decision_margins"].values()):
        raise PrototypeError("decision margins must be finite")
    if (
        float(value["decision_margins"]["S0_equivalence_half_width"]) <= 0.0
        or float(value["decision_margins"]["S1_upper_limit"]) != 0.0
        or float(value["decision_margins"]["S2_lower_limit"]) <= 0.0
    ):
        raise PrototypeError("decision margins differ")
    criteria_fields = {
        "half_draw_count_for_reproducibility",
        "max_balanced_to_iid_median_disjoint_error_ratio",
        "min_configs_with_reproducibility_gain",
        "min_k44_decision_clear_fraction",
        "max_s0_balanced_minus_iid_quantile_difference",
        "max_s0_balanced_minus_iid_correlation_difference",
        "max_absolute_pair_collision_rate_difference",
        "max_absolute_triple_collision_rate_difference",
        "max_balanced_exposure_cv",
    }
    if set(value["prototype_criteria"]) != criteria_fields:
        raise PrototypeError("prototype criteria fields differ")
    criteria = value["prototype_criteria"]
    if int(criteria["half_draw_count_for_reproducibility"]) * 2 != value["draw_count"]:
        raise PrototypeError("reproducibility halves must exhaust the draw surface")
    if int(criteria["half_draw_count_for_reproducibility"]) % (value["cohort_size"] - 1):
        raise PrototypeError("reproducibility halves must contain complete balanced cycles")
    bounded_zero_one = (
        "max_balanced_to_iid_median_disjoint_error_ratio",
        "min_k44_decision_clear_fraction",
        "max_s0_balanced_minus_iid_quantile_difference",
        "max_s0_balanced_minus_iid_correlation_difference",
        "max_absolute_pair_collision_rate_difference",
        "max_absolute_triple_collision_rate_difference",
        "max_balanced_exposure_cv",
    )
    if any(
        not math.isfinite(float(criteria[name])) or not 0.0 <= float(criteria[name]) <= 1.0
        for name in bounded_zero_one
    ):
        raise PrototypeError("prototype numeric criteria must be finite and in [0,1]")
    min_configs = criteria["min_configs_with_reproducibility_gain"]
    if isinstance(min_configs, bool) or not isinstance(min_configs, int) or not 1 <= min_configs <= len(CONFIGS):
        raise PrototypeError("prototype reproducibility config count differs")
    caps = value["resource_cap"]
    if set(caps) != {"max_wall_seconds", "max_output_bytes", "max_shards"}:
        raise PrototypeError("resource cap fields differ")
    expected_shards = len(CONFIGS) * value["dataset_count"] * len(case_ids)
    if (
        isinstance(caps["max_shards"], bool)
        or not isinstance(caps["max_shards"], int)
        or caps["max_shards"] != expected_shards
    ):
        raise PrototypeError("resource shard cap is not exact")
    for name in ("max_wall_seconds", "max_output_bytes"):
        if isinstance(caps[name], bool) or not isinstance(caps[name], int) or caps[name] <= 0:
            raise PrototypeError(f"resource cap {name} differs")
    return json.loads(json.dumps(value))


def load_spec(path: Path) -> tuple[dict[str, Any], str]:
    return validate_spec(load_json(path)), sha256_file(path)


def validate_locked_study(path: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    if sha256_file(path) != spec["source_locked_study_sha256"]:
        raise PrototypeError("source locked study hash differs")
    study = load_json(path)
    if (
        study.get("schema_version") != "gate12c2_locked_calibration_spec_v0.1"
        or study.get("cohort_size") != spec["cohort_size"]
        or study.get("q_values") != spec["q_values"]
    ):
        raise PrototypeError("source locked study surface differs")
    cases = {case["case_id"]: case for case in study["cases"]}
    if any(case_id not in cases for case_id in spec["case_ids"]):
        raise PrototypeError("prototype case missing from source study")
    return study


def random_derangement(size: int, generator: np.random.Generator) -> np.ndarray:
    identity = np.arange(size)
    for _ in range(128):
        candidate = generator.permutation(size)
        if np.all(candidate != identity):
            return candidate
    shift = int(generator.integers(1, size))
    return np.roll(identity, shift)


def iid_assignments(
    *,
    size: int,
    draw_count: int,
    seed_namespace: str,
    case_id: str,
    regime: str,
    dataset_index: int,
) -> np.ndarray:
    assignments = np.empty((draw_count, 3, size), dtype=int)
    for draw in range(draw_count):
        for role in range(3):
            assignments[draw, role] = random_derangement(
                size,
                rng(
                    seed_namespace,
                    case_id,
                    regime,
                    dataset_index,
                    "iid",
                    draw,
                    role,
                ),
            )
    return assignments


def balanced_assignments(
    *,
    size: int,
    draw_count: int,
    seed_namespace: str,
    case_id: str,
    regime: str,
    dataset_index: int,
) -> np.ndarray:
    """Build randomized 1-factorization cycles with exact donor exposure."""

    cycle_size = size - 1
    if draw_count % cycle_size:
        raise PrototypeError("balanced draws must contain complete cycles")
    assignments = np.empty((draw_count, 3, size), dtype=int)
    for cycle in range(draw_count // cycle_size):
        for role in range(3):
            generator = rng(
                seed_namespace,
                case_id,
                regime,
                dataset_index,
                "balanced",
                cycle,
                role,
            )
            vertex_order = generator.permutation(size)
            shift_order = generator.permutation(np.arange(1, size))
            for within, shift in enumerate(shift_order):
                mapping = np.empty(size, dtype=int)
                for position, recipient in enumerate(vertex_order):
                    mapping[int(recipient)] = int(vertex_order[(position + int(shift)) % size])
                assignments[cycle * cycle_size + within, role] = mapping
    return assignments


def assignment_metrics(assignments: np.ndarray) -> dict[str, float]:
    draw_count, roles, size = assignments.shape
    if roles != 3:
        raise PrototypeError("assignment role count differs")
    identity = np.arange(size)
    if np.any(assignments == identity[None, None, :]):
        raise PrototypeError("assignment contains a fixed point")
    recipient_tuples = np.transpose(assignments, (0, 2, 1))
    unique_counts = np.apply_along_axis(lambda row: len(set(row.tolist())), 2, recipient_tuples)
    exposure_cvs: list[float] = []
    exposure_ranges: list[float] = []
    for recipient in range(size):
        eligible = [donor for donor in range(size) if donor != recipient]
        for role in range(3):
            counts = np.asarray(
                [np.sum(assignments[:, role, recipient] == donor) for donor in eligible],
                dtype=float,
            )
            exposure_cvs.append(float(np.std(counts) / np.mean(counts)))
            exposure_ranges.append(float(np.max(counts) - np.min(counts)))
    return {
        "pair_collision_rate": float(np.mean(unique_counts < 3)),
        "triple_collision_rate": float(np.mean(unique_counts == 1)),
        "mean_exposure_cv": float(np.mean(exposure_cvs)),
        "max_exposure_cv": float(np.max(exposure_cvs)),
        "max_exposure_range": float(np.max(exposure_ranges)),
    }


def reassignment(
    graphs: Sequence[Graph],
    assignments: np.ndarray,
    draw_index: int,
    schedule: str,
) -> tuple[Graph, ...]:
    result: list[Graph] = []
    for recipient, graph in enumerate(graphs):
        frames = tuple(
            graphs[int(assignments[draw_index, role, recipient])].frames[role]
            for role in range(3)
        )
        result.append(
            Graph.from_frames(
                replicate_id=f"{graph.replicate_id}:N1:{schedule}:d{draw_index:03d}",
                regime=f"{graph.regime}_N1_{schedule}",
                frames=frames,
            )
        )
    return tuple(result)


def diagnostic(
    graph_or_edges: Graph | Sequence[np.ndarray],
    q: int,
    study: Mapping[str, Any],
) -> ResidualDiagnostics:
    edges = graph_or_edges.edges if isinstance(graph_or_edges, Graph) else graph_or_edges
    return residual_diagnostics(
        edges[0],
        edges[1],
        edges[2],
        q,
        spectral_gap_tolerance=float(study["spectral_gap_tolerance"]),
        numerical_tolerance=float(study["numerical_tolerance"]),
    )


def feature_vector(graph: Graph, q1: ResidualDiagnostics) -> np.ndarray:
    edge = np.asarray(edge_singular_values(graph.edges), dtype=float).reshape(-1)
    product = np.asarray(
        [*q1.product_singular_values_left, *q1.product_singular_values_right],
        dtype=float,
    )
    stacked = np.concatenate(graph.frames, axis=1)
    gram = stacked.T @ stacked
    eigenvalues = np.linalg.eigvalsh(gram)[::-1]
    value = np.concatenate([edge, product, eigenvalues])
    if value.shape != (24,) or not np.isfinite(value).all():
        raise PrototypeError("geometry feature vector differs")
    return value


def correlation_matrix(value: np.ndarray) -> np.ndarray:
    result = np.corrcoef(value, rowvar=False)
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def geometry_evidence(observed: np.ndarray, null: np.ndarray) -> dict[str, Any]:
    observed_quantiles = np.quantile(observed, QUANTILES, axis=0)
    null_quantiles = np.quantile(null, QUANTILES, axis=0)
    observed_corr = correlation_matrix(observed)
    null_corr = correlation_matrix(null)
    rows: dict[str, Any] = {}
    for surface, selection in SURFACES.items():
        qdiff = np.abs(observed_quantiles[:, selection] - null_quantiles[:, selection])
        cdiff = observed_corr[selection, selection] - null_corr[selection, selection]
        mask = ~np.eye(cdiff.shape[0], dtype=bool)
        rows[surface] = {
            "max_quantile_difference": float(np.max(qdiff)),
            "median_quantile_difference": float(np.median(qdiff)),
            "cross_feature_correlation_max_difference": float(np.max(np.abs(cdiff[mask]))),
            "cross_feature_correlation_frobenius": float(np.linalg.norm(cdiff, ord="fro")),
        }
    return {
        "feature_names": list(FEATURE_NAMES),
        "quantiles": QUANTILES.tolist(),
        "observed_quantiles": observed_quantiles.tolist(),
        "null_quantiles": null_quantiles.tolist(),
        "observed_correlation": observed_corr.tolist(),
        "null_correlation": null_corr.tolist(),
        "metrics": rows,
    }


def generate_observed(
    config: str,
    case: Mapping[str, Any],
    dataset_index: int,
    spec: Mapping[str, Any],
    study: Mapping[str, Any],
) -> tuple[Graph, ...]:
    seed_namespace = str(spec["seed_namespace"])
    cohort_size = int(spec["cohort_size"])
    if config == "S1_PRIMARY":
        return generate_s1_cohort(
            case=case,
            seed_namespace=seed_namespace,
            outer_index=dataset_index,
            cohort_size=cohort_size,
            effect_strength=float(study["s1_effects"]["primary"]),
            observed_mismatch=0.01,
        )
    regime = "S2" if config == "S2" else "S0"
    noise_name = "s2_frame_noise" if config == "S2" else "s0_frame_noise"
    return generate_s0_cohort(
        case=case,
        seed_namespace=seed_namespace,
        outer_index=dataset_index,
        cohort_size=cohort_size,
        frame_noise=float(case[noise_name]),
        regime=regime,
    )


def run_shard(
    *,
    config: str,
    case: Mapping[str, Any],
    dataset_index: int,
    spec: Mapping[str, Any],
    spec_sha256: str,
    study: Mapping[str, Any],
) -> dict[str, Any]:
    regime = "S1" if config == "S1_PRIMARY" else config
    draw_count = int(spec["draw_count"])
    cohort_size = int(spec["cohort_size"])
    epsilon = float(study["epsilon"])
    observed = generate_observed(config, case, dataset_index, spec, study)
    observed_digest = graph_digest(observed)
    observed_diagnostics = {
        (graph_index, q): diagnostic(graph, q, study)
        for graph_index, graph in enumerate(observed)
        for q in spec["q_values"]
    }
    if not all(value.eligible and value.numerical_pass for value in observed_diagnostics.values()):
        raise PrototypeError("observed diagnostic is ineligible or numerically invalid")
    observed_features = np.asarray(
        [feature_vector(graph, observed_diagnostics[(index, 1)]) for index, graph in enumerate(observed)],
        dtype=float,
    )
    schedule_results: dict[str, Any] = {}
    max_realizability_error = 0.0
    max_stressor_spectrum_error = 0.0

    for schedule in SCHEDULES:
        assignment = (
            iid_assignments(
                size=cohort_size,
                draw_count=draw_count,
                seed_namespace=spec["seed_namespace"],
                case_id=case["case_id"],
                regime=regime,
                dataset_index=dataset_index,
            )
            if schedule == "iid"
            else balanced_assignments(
                size=cohort_size,
                draw_count=draw_count,
                seed_namespace=spec["seed_namespace"],
                case_id=case["case_id"],
                regime=regime,
                dataset_index=dataset_index,
            )
        )
        effects = {
            str(q): np.empty((cohort_size, draw_count), dtype=float)
            for q in spec["q_values"]
        }
        null_features: list[np.ndarray] = []
        numerical_failures = 0
        ineligible = 0
        for draw in range(draw_count):
            null_graphs = reassignment(observed, assignment, draw, schedule)
            max_realizability_error = max(
                max_realizability_error,
                max(joint_realizability_error(graph) for graph in null_graphs),
            )
            for graph_index, null_graph in enumerate(null_graphs):
                baseline = {
                    q: diagnostic(null_graph, q, study) for q in spec["q_values"]
                }
                null_features.append(feature_vector(null_graph, baseline[1]))
                stress: dict[int, ResidualDiagnostics] | None = None
                if config == "S2":
                    stress_edges = independent_edge_reorientation(
                        null_graph.edges,
                        # Common random rotations reduce noise in the paired
                        # schedule comparison without coupling either null.
                        seed_namespace=f"{spec['seed_namespace']}:shared-stressor",
                        case_id=case["case_id"],
                        outer_index=dataset_index,
                        draw_index=draw,
                        graph_index=graph_index,
                        trial_index=0,
                    )
                    max_stressor_spectrum_error = max(
                        max_stressor_spectrum_error,
                        edge_spectrum_error(null_graph.edges, stress_edges),
                    )
                    stress = {q: diagnostic(stress_edges, q, study) for q in spec["q_values"]}
                for q in spec["q_values"]:
                    numerator = stress[q] if stress is not None else observed_diagnostics[(graph_index, q)]
                    numerical_failures += int(not baseline[q].numerical_pass or not numerator.numerical_pass)
                    ineligible += int(not baseline[q].eligible or not numerator.eligible)
                    effects[str(q)][graph_index, draw] = (
                        math.log(numerator.a + epsilon) - math.log(baseline[q].a + epsilon)
                    )
        if numerical_failures or ineligible:
            raise PrototypeError("prototype diagnostic coverage differs")
        schedule_results[schedule] = {
            "assignment_metrics": assignment_metrics(assignment),
            "effects": {q: value.tolist() for q, value in effects.items()},
            "geometry": geometry_evidence(observed_features, np.asarray(null_features, dtype=float)),
        }

    if graph_digest(observed) != observed_digest:
        raise PrototypeError("observed cohort mutated")
    return {
        "schema_version": SHARD_SCHEMA,
        "study_id": spec["study_id"],
        "spec_sha256": spec_sha256,
        "config": config,
        "case": dict(case),
        "dataset_index": dataset_index,
        "observed_cohort_sha256": observed_digest,
        "observed_features": observed_features.tolist(),
        "schedules": schedule_results,
        "controls": {
            "observed_unchanged": True,
            "max_n1_realizability_error": max_realizability_error,
            "max_stressor_edge_spectrum_error": (
                max_stressor_spectrum_error if config == "S2" else None
            ),
        },
    }


def expected_shards(spec: Mapping[str, Any]) -> list[tuple[str, str, int]]:
    return [
        (config, case_id, dataset)
        for config in CONFIGS
        for dataset in range(int(spec["dataset_count"]))
        for case_id in spec["case_ids"]
    ]


def shard_path(root: Path, config: str, case_id: str, dataset: int) -> Path:
    return root / "shards" / config / case_id / f"d{dataset:04d}.json"


def validate_shard(
    value: Mapping[str, Any],
    *,
    config: str,
    case_id: str,
    dataset: int,
    spec: Mapping[str, Any],
    spec_sha256: str,
) -> None:
    required = {
        "schema_version",
        "study_id",
        "spec_sha256",
        "config",
        "case",
        "dataset_index",
        "observed_cohort_sha256",
        "observed_features",
        "schedules",
        "controls",
    }
    if set(value) != required:
        raise PrototypeError("shard fields differ")
    if (
        value["schema_version"] != SHARD_SCHEMA
        or value["study_id"] != spec["study_id"]
        or value["spec_sha256"] != spec_sha256
        or value["config"] != config
        or value["case"]["case_id"] != case_id
        or int(value["dataset_index"]) != dataset
    ):
        raise PrototypeError("shard identity differs")
    digest = value["observed_cohort_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise PrototypeError("shard observed digest differs")
    observed = np.asarray(value["observed_features"], dtype=float)
    if observed.shape != (spec["cohort_size"], len(FEATURE_NAMES)) or not np.isfinite(observed).all():
        raise PrototypeError("shard observed feature surface differs")
    if set(value["schedules"]) != set(SCHEDULES):
        raise PrototypeError("shard schedule surface differs")
    for schedule in SCHEDULES:
        row = value["schedules"][schedule]
        if set(row) != {"assignment_metrics", "effects", "geometry"}:
            raise PrototypeError("shard schedule fields differ")
        metrics = row["assignment_metrics"]
        if set(metrics) != ASSIGNMENT_METRIC_FIELDS or any(
            not math.isfinite(float(item)) or float(item) < 0.0 for item in metrics.values()
        ):
            raise PrototypeError("shard assignment metrics differ")
        regime = "S1" if config == "S1_PRIMARY" else config
        reconstructed = (
            iid_assignments(
                size=int(spec["cohort_size"]),
                draw_count=int(spec["draw_count"]),
                seed_namespace=str(spec["seed_namespace"]),
                case_id=case_id,
                regime=regime,
                dataset_index=dataset,
            )
            if schedule == "iid"
            else balanced_assignments(
                size=int(spec["cohort_size"]),
                draw_count=int(spec["draw_count"]),
                seed_namespace=str(spec["seed_namespace"]),
                case_id=case_id,
                regime=regime,
                dataset_index=dataset,
            )
        )
        expected_metrics = assignment_metrics(reconstructed)
        if any(
            not math.isclose(float(metrics[name]), expected_metrics[name], rel_tol=0.0, abs_tol=1e-15)
            for name in ASSIGNMENT_METRIC_FIELDS
        ):
            raise PrototypeError("shard assignment reconstruction differs")
        for q in spec["q_values"]:
            matrix = np.asarray(row["effects"][str(q)], dtype=float)
            if matrix.shape != (spec["cohort_size"], spec["draw_count"]) or not np.isfinite(matrix).all():
                raise PrototypeError("shard effect surface differs")
        if set(row["effects"]) != {str(q) for q in spec["q_values"]}:
            raise PrototypeError("shard effect keys differ")
        geometry = row["geometry"]
        if set(geometry) != {
            "feature_names",
            "quantiles",
            "observed_quantiles",
            "null_quantiles",
            "observed_correlation",
            "null_correlation",
            "metrics",
        }:
            raise PrototypeError("shard geometry fields differ")
        if geometry["feature_names"] != list(FEATURE_NAMES) or set(geometry["metrics"]) != set(SURFACES):
            raise PrototypeError("shard geometry surface differs")
        if not np.allclose(np.asarray(geometry["quantiles"], dtype=float), QUANTILES, rtol=0.0, atol=0.0):
            raise PrototypeError("shard geometry quantiles differ")
        observed_quantiles = np.asarray(geometry["observed_quantiles"], dtype=float)
        null_quantiles = np.asarray(geometry["null_quantiles"], dtype=float)
        observed_correlation = np.asarray(geometry["observed_correlation"], dtype=float)
        null_correlation = np.asarray(geometry["null_correlation"], dtype=float)
        for matrix, shape in (
            (observed_quantiles, (len(QUANTILES), len(FEATURE_NAMES))),
            (null_quantiles, (len(QUANTILES), len(FEATURE_NAMES))),
            (observed_correlation, (len(FEATURE_NAMES), len(FEATURE_NAMES))),
            (null_correlation, (len(FEATURE_NAMES), len(FEATURE_NAMES))),
        ):
            if matrix.shape != shape or not np.isfinite(matrix).all():
                raise PrototypeError("shard geometry array differs")
        if not np.allclose(
            observed_quantiles,
            np.quantile(observed, QUANTILES, axis=0),
            rtol=0.0,
            atol=1e-12,
        ) or not np.allclose(
            observed_correlation,
            correlation_matrix(observed),
            rtol=0.0,
            atol=1e-12,
        ):
            raise PrototypeError("shard observed geometry reconstruction differs")
        for surface, selection in SURFACES.items():
            surface_metrics = geometry["metrics"][surface]
            if set(surface_metrics) != GEOMETRY_METRIC_FIELDS or any(
                not math.isfinite(float(item)) or float(item) < 0.0
                for item in surface_metrics.values()
            ):
                raise PrototypeError("shard geometry metrics differ")
            qdiff = np.abs(observed_quantiles[:, selection] - null_quantiles[:, selection])
            cdiff = observed_correlation[selection, selection] - null_correlation[selection, selection]
            mask = ~np.eye(cdiff.shape[0], dtype=bool)
            expected_geometry = {
                "max_quantile_difference": float(np.max(qdiff)),
                "median_quantile_difference": float(np.median(qdiff)),
                "cross_feature_correlation_max_difference": float(np.max(np.abs(cdiff[mask]))),
                "cross_feature_correlation_frobenius": float(np.linalg.norm(cdiff, ord="fro")),
            }
            if any(
                not math.isclose(
                    float(surface_metrics[name]), expected_geometry[name], rel_tol=0.0, abs_tol=1e-12
                )
                for name in GEOMETRY_METRIC_FIELDS
            ):
                raise PrototypeError("shard geometry metric reconstruction differs")
    controls = value["controls"]
    if set(controls) != {
        "observed_unchanged",
        "max_n1_realizability_error",
        "max_stressor_edge_spectrum_error",
    }:
        raise PrototypeError("shard control fields differ")
    if controls["observed_unchanged"] is not True or not 0.0 <= float(
        controls["max_n1_realizability_error"]
    ) <= 1e-10:
        raise PrototypeError("shard realizability control differs")
    if config == "S2":
        stressor_error = float(controls["max_stressor_edge_spectrum_error"])
        if not 0.0 <= stressor_error <= 1e-10:
            raise PrototypeError("shard stressor spectrum control differs")
    if config != "S2" and controls["max_stressor_edge_spectrum_error"] is not None:
        raise PrototypeError("non-S2 shard has a stressor control")


def median_effect(matrix: np.ndarray, indices: np.ndarray) -> np.ndarray:
    selected = matrix[:, indices]
    return np.median(np.median(selected, axis=-1), axis=0)


def cycle_bootstrap_indices(
    k: int,
    *,
    cycle_size: int,
    seed: int,
    repeats: int,
) -> np.ndarray:
    """Resample complete randomized assignment cycles, never individual draws."""

    if k % cycle_size or k < 2 * cycle_size:
        raise PrototypeError("cycle bootstrap requires at least two complete cycles")
    block_count = k // cycle_size
    generator = np.random.default_rng(seed)
    blocks = generator.integers(0, block_count, size=(repeats, block_count))
    offsets = np.arange(cycle_size)
    return (blocks[:, :, None] * cycle_size + offsets[None, None, :]).reshape(repeats, k)


def cycle_block_mcse(
    matrix: np.ndarray,
    k: int,
    *,
    cycle_size: int,
    seed: int,
    repeats: int,
) -> float:
    if k < 2 * cycle_size:
        return float("nan")
    indices = cycle_bootstrap_indices(
        k,
        cycle_size=cycle_size,
        seed=seed,
        repeats=repeats,
    )
    estimates = median_effect(matrix[:, :k], indices)
    return float(np.std(estimates, ddof=1))


def bootstrap_interval(values: Sequence[float], seed: int, repeats: int = 5000) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(array), size=(repeats, len(array)))
    estimates = np.median(array[indices], axis=1)
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(lower), float(upper)


def paired_median_ratio_interval(
    numerator: Sequence[float],
    denominator: Sequence[float],
    seed: int,
    repeats: int = 5000,
) -> tuple[float, float]:
    numerator_array = np.asarray(numerator, dtype=float)
    denominator_array = np.asarray(denominator, dtype=float)
    if numerator_array.shape != denominator_array.shape or not numerator_array.size:
        raise PrototypeError("paired median ratio surface differs")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(numerator_array), size=(repeats, len(numerator_array)))
    numerator_medians = np.median(numerator_array[indices], axis=1)
    denominator_medians = np.maximum(
        np.median(denominator_array[indices], axis=1),
        np.finfo(float).eps,
    )
    lower, upper = np.quantile(numerator_medians / denominator_medians, [0.025, 0.975])
    return float(lower), float(upper)


def decision_clear(config: str, effect: float, mcse: float, margins: Mapping[str, Any]) -> bool:
    radius = 1.96 * mcse
    if config == "S0":
        return bool(abs(effect) + radius <= float(margins["S0_equivalence_half_width"]))
    if config == "S1_PRIMARY":
        return bool(effect + radius < float(margins["S1_upper_limit"]))
    return bool(effect - radius > float(margins["S2_lower_limit"]))


def analyze(
    root: Path,
    spec: Mapping[str, Any],
    spec_sha256: str,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    repeats = int(spec["bootstrap_repeats"])
    cycle_size = int(spec["cohort_size"]) - 1
    half_draw_count = int(spec["prototype_criteria"]["half_draw_count_for_reproducibility"])
    endpoint_rows: list[dict[str, Any]] = []
    dataset_matrices: dict[tuple[str, str, int], list[tuple[str, int, np.ndarray]]] = defaultdict(list)
    geometry_rows: list[dict[str, Any]] = []
    donor_rows: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []

    for config, case_id, dataset in expected_shards(spec):
        value = load_json(shard_path(root, config, case_id, dataset))
        validate_shard(
            value,
            config=config,
            case_id=case_id,
            dataset=dataset,
            spec=spec,
            spec_sha256=spec_sha256,
        )
        controls.append(value["controls"])
        for schedule in SCHEDULES:
            schedule_row = value["schedules"][schedule]
            donor_rows.append(
                {
                    "config": config,
                    "schedule": schedule,
                    "dataset_index": dataset,
                    "case_id": case_id,
                    **schedule_row["assignment_metrics"],
                }
            )
            for surface, metrics in schedule_row["geometry"]["metrics"].items():
                geometry_rows.append(
                    {
                        "config": config,
                        "schedule": schedule,
                        "dataset_index": dataset,
                        "case_id": case_id,
                        "surface": surface,
                        **metrics,
                    }
                )
            for q in spec["q_values"]:
                matrix = np.asarray(schedule_row["effects"][str(q)], dtype=float)
                dataset_matrices[(config, schedule, dataset)].append((case_id, q, matrix))
                first_half_effect = float(
                    median_effect(matrix, np.arange(half_draw_count))
                )
                second_half_effect = float(
                    median_effect(
                        matrix,
                        np.arange(half_draw_count, 2 * half_draw_count),
                    )
                )
                for k in spec["checkpoints"]:
                    point = float(median_effect(matrix[:, :k], np.arange(k)))
                    mcse = cycle_block_mcse(
                        matrix,
                        k,
                        cycle_size=cycle_size,
                        seed=stable_seed(
                            "endpoint-cycle-mcse",
                            config,
                            schedule,
                            dataset,
                            case_id,
                            q,
                            k,
                        ),
                        repeats=repeats,
                    )
                    endpoint_rows.append(
                        {
                            "config": config,
                            "schedule": schedule,
                            "dataset_index": dataset,
                            "case_id": case_id,
                            "q": q,
                            "k": k,
                            "effect": point,
                            "mcse": mcse,
                            "first_half_effect": first_half_effect,
                            "second_half_effect": second_half_effect,
                            "disjoint_half_absolute_error": abs(
                                first_half_effect - second_half_effect
                            ),
                            "mcse_method": (
                                "complete_cycle_bootstrap"
                                if math.isfinite(mcse)
                                else "unavailable_single_cycle"
                            ),
                        }
                    )

    dataset_rows: list[dict[str, Any]] = []
    reproducibility_rows: list[dict[str, Any]] = []
    for (config, schedule, dataset), matrices in sorted(dataset_matrices.items()):
        first_half_points = [
            float(median_effect(matrix, np.arange(half_draw_count)))
            for _case, _q, matrix in matrices
        ]
        second_half_points = [
            float(
                median_effect(
                    matrix,
                    np.arange(half_draw_count, 2 * half_draw_count),
                )
            )
            for _case, _q, matrix in matrices
        ]
        first_half_family = float(np.median(first_half_points))
        second_half_family = float(np.median(second_half_points))
        reproducibility_rows.append(
            {
                "config": config,
                "schedule": schedule,
                "dataset_index": dataset,
                "half_draw_count": half_draw_count,
                "first_half_family_effect": first_half_family,
                "second_half_family_effect": second_half_family,
                "disjoint_half_absolute_error": abs(
                    first_half_family - second_half_family
                ),
            }
        )
        for k in spec["checkpoints"]:
            endpoint_points = [
                float(median_effect(matrix[:, :k], np.arange(k)))
                for _case, _q, matrix in matrices
            ]
            point = float(np.median(endpoint_points))
            if k >= 2 * cycle_size:
                indices_by_case = {
                    case_id: cycle_bootstrap_indices(
                        k,
                        cycle_size=cycle_size,
                        seed=stable_seed(
                            "dataset-cycle-mcse",
                            config,
                            schedule,
                            dataset,
                            case_id,
                            k,
                        ),
                        repeats=repeats,
                    )
                    for case_id, _q, _matrix in matrices
                }
                bootstrap_endpoints = np.vstack(
                    [
                        median_effect(matrix[:, :k], indices_by_case[case_id])
                        for case_id, _q, matrix in matrices
                    ]
                )
                bootstrap_family = np.median(bootstrap_endpoints, axis=0)
                mcse = float(np.std(bootstrap_family, ddof=1))
            else:
                mcse = float("nan")
            dataset_rows.append(
                {
                    "config": config,
                    "schedule": schedule,
                    "dataset_index": dataset,
                    "k": k,
                    "family_effect": point,
                    "family_mcse": mcse,
                    "decision_clear": (
                        float(
                            decision_clear(
                                config,
                                point,
                                mcse,
                                spec["decision_margins"],
                            )
                        )
                        if math.isfinite(mcse)
                        else float("nan")
                    ),
                    "mcse_method": (
                        "case-independent_complete-cycle_bootstrap"
                        if math.isfinite(mcse)
                        else "unavailable_single_cycle"
                    ),
                }
            )

    endpoints = pd.DataFrame(endpoint_rows)
    datasets = pd.DataFrame(dataset_rows)
    reproducibility_long = pd.DataFrame(reproducibility_rows)
    geometry = pd.DataFrame(geometry_rows)
    donors = pd.DataFrame(donor_rows)

    paired = datasets.pivot(
        index=["config", "dataset_index", "k"],
        columns="schedule",
        values=["family_effect", "family_mcse", "decision_clear"],
    ).reset_index()
    paired.columns = [
        "_".join(str(item) for item in column if str(item)) if isinstance(column, tuple) else str(column)
        for column in paired.columns
    ]
    paired["balanced_to_iid_mcse_ratio"] = paired["family_mcse_balanced"] / paired[
        "family_mcse_iid"
    ].clip(lower=np.finfo(float).eps)
    paired["absolute_schedule_effect_difference"] = (
        paired["family_effect_balanced"] - paired["family_effect_iid"]
    ).abs()

    reproducibility = reproducibility_long.pivot(
        index=["config", "dataset_index", "half_draw_count"],
        columns="schedule",
        values=[
            "first_half_family_effect",
            "second_half_family_effect",
            "disjoint_half_absolute_error",
        ],
    ).reset_index()
    reproducibility.columns = [
        "_".join(str(item) for item in column if str(item))
        if isinstance(column, tuple)
        else str(column)
        for column in reproducibility.columns
    ]

    reproducibility_summary_rows: list[dict[str, Any]] = []
    for config, rows in reproducibility.groupby("config"):
        iid_errors = rows["disjoint_half_absolute_error_iid"].to_numpy(dtype=float)
        balanced_errors = rows["disjoint_half_absolute_error_balanced"].to_numpy(dtype=float)
        iid_median = float(np.median(iid_errors))
        balanced_median = float(np.median(balanced_errors))
        ratio = balanced_median / max(iid_median, np.finfo(float).eps)
        lower, upper = paired_median_ratio_interval(
            balanced_errors,
            iid_errors,
            stable_seed("disjoint-error-ratio", config),
        )
        reproducibility_summary_rows.append(
            {
                "config": config,
                "dataset_count": len(rows),
                "median_iid_disjoint_half_absolute_error": iid_median,
                "median_balanced_disjoint_half_absolute_error": balanced_median,
                "balanced_to_iid_median_disjoint_error_ratio": ratio,
                "ratio_bootstrap_lower": lower,
                "ratio_bootstrap_upper": upper,
                "fraction_balanced_disjoint_error_lower": float(
                    np.mean(balanced_errors < iid_errors)
                ),
            }
        )
    reproducibility_summary = pd.DataFrame(reproducibility_summary_rows)

    ratio_rows: list[dict[str, Any]] = []
    for (config, k), rows in paired.groupby(["config", "k"]):
        rows = rows.loc[np.isfinite(rows["balanced_to_iid_mcse_ratio"])]
        if rows.empty:
            continue
        lower, upper = bootstrap_interval(
            rows["balanced_to_iid_mcse_ratio"],
            stable_seed("ratio-interval", config, k),
        )
        ratio_rows.append(
            {
                "config": config,
                "k": k,
                "dataset_count": len(rows),
                "median_balanced_to_iid_mcse_ratio": float(
                    rows["balanced_to_iid_mcse_ratio"].median()
                ),
                "median_ratio_bootstrap_lower": lower,
                "median_ratio_bootstrap_upper": upper,
                "fraction_balanced_mcse_lower": float(
                    np.mean(rows["balanced_to_iid_mcse_ratio"] < 1.0)
                ),
                "median_absolute_schedule_effect_difference": float(
                    rows["absolute_schedule_effect_difference"].median()
                ),
            }
        )
    ratio_summary = pd.DataFrame(ratio_rows)

    precision_datasets = datasets.loc[np.isfinite(datasets["family_mcse"])]
    decision_summary = (
        precision_datasets.groupby(["config", "schedule", "k"])
        .agg(
            dataset_count=("dataset_index", "size"),
            median_family_effect=("family_effect", "median"),
            median_family_mcse=("family_mcse", "median"),
            p90_family_mcse=("family_mcse", lambda value: value.quantile(0.90)),
            decision_clear_fraction=("decision_clear", "mean"),
        )
        .reset_index()
    )
    precision_endpoints = endpoints.loc[np.isfinite(endpoints["mcse"])]
    endpoint_summary = (
        precision_endpoints.groupby(["config", "schedule", "k"])
        .agg(
            endpoint_count=("mcse", "size"),
            median_endpoint_mcse=("mcse", "median"),
            p95_endpoint_mcse=("mcse", lambda value: value.quantile(0.95)),
            fraction_1_96_mcse_within_old_0_30=("mcse", lambda value: float(np.mean(1.96 * value <= 0.30))),
        )
        .reset_index()
    )

    geometry_paired = geometry.pivot(
        index=["config", "dataset_index", "case_id", "surface"],
        columns="schedule",
        values=[
            "max_quantile_difference",
            "cross_feature_correlation_max_difference",
        ],
    ).reset_index()
    geometry_paired.columns = [
        "_".join(str(item) for item in column if str(item)) if isinstance(column, tuple) else str(column)
        for column in geometry_paired.columns
    ]
    geometry_paired["balanced_minus_iid_quantile_difference"] = (
        geometry_paired["max_quantile_difference_balanced"]
        - geometry_paired["max_quantile_difference_iid"]
    )
    geometry_paired["balanced_minus_iid_correlation_difference"] = (
        geometry_paired["cross_feature_correlation_max_difference_balanced"]
        - geometry_paired["cross_feature_correlation_max_difference_iid"]
    )
    geometry_summary = (
        geometry_paired.groupby(["config", "surface"])
        .agg(
            shard_count=("case_id", "size"),
            median_balanced_minus_iid_quantile_difference=(
                "balanced_minus_iid_quantile_difference",
                "median",
            ),
            p90_balanced_minus_iid_quantile_difference=(
                "balanced_minus_iid_quantile_difference",
                lambda value: value.quantile(0.90),
            ),
            median_balanced_minus_iid_correlation_difference=(
                "balanced_minus_iid_correlation_difference",
                "median",
            ),
            p90_balanced_minus_iid_correlation_difference=(
                "balanced_minus_iid_correlation_difference",
                lambda value: value.quantile(0.90),
            ),
        )
        .reset_index()
    )
    donor_summary = (
        donors.groupby(["config", "schedule"])
        .agg(
            shard_count=("case_id", "size"),
            mean_pair_collision_rate=("pair_collision_rate", "mean"),
            mean_triple_collision_rate=("triple_collision_rate", "mean"),
            mean_exposure_cv=("mean_exposure_cv", "mean"),
            max_exposure_cv=("max_exposure_cv", "max"),
            max_exposure_range=("max_exposure_range", "max"),
        )
        .reset_index()
    )
    donor_paired = donor_summary.pivot(
        index="config",
        columns="schedule",
        values=[
            "mean_pair_collision_rate",
            "mean_triple_collision_rate",
            "mean_exposure_cv",
            "max_exposure_cv",
            "max_exposure_range",
        ],
    ).reset_index()
    donor_paired.columns = [
        "_".join(str(item) for item in column if str(item))
        if isinstance(column, tuple)
        else str(column)
        for column in donor_paired.columns
    ]
    donor_paired["absolute_pair_collision_rate_difference"] = (
        donor_paired["mean_pair_collision_rate_balanced"]
        - donor_paired["mean_pair_collision_rate_iid"]
    ).abs()
    donor_paired["absolute_triple_collision_rate_difference"] = (
        donor_paired["mean_triple_collision_rate_balanced"]
        - donor_paired["mean_triple_collision_rate_iid"]
    ).abs()

    criteria = spec["prototype_criteria"]
    if set(reproducibility_summary["config"]) != set(CONFIGS):
        raise PrototypeError("reproducibility summary config surface differs")
    reproducibility_configs = int(
        np.sum(
            reproducibility_summary[
                "balanced_to_iid_median_disjoint_error_ratio"
            ]
            <= float(criteria["max_balanced_to_iid_median_disjoint_error_ratio"])
        )
    )
    reproducibility_pass = reproducibility_configs >= int(
        criteria["min_configs_with_reproducibility_gain"]
    )
    precision_rows = decision_summary.loc[
        (decision_summary["schedule"] == "balanced") & (decision_summary["k"] == 44)
    ]
    if set(precision_rows["config"]) != set(CONFIGS):
        raise PrototypeError("decision precision config surface differs")
    precision_pass = bool(
        np.all(
            precision_rows["decision_clear_fraction"]
            >= float(criteria["min_k44_decision_clear_fraction"])
        )
    )
    s0_geometry = geometry_summary.loc[geometry_summary["config"] == "S0"]
    if set(s0_geometry["surface"]) != set(SURFACES):
        raise PrototypeError("S0 geometry surface differs")
    geometry_pass = bool(
        np.all(
            s0_geometry["median_balanced_minus_iid_quantile_difference"]
            <= float(criteria["max_s0_balanced_minus_iid_quantile_difference"])
        )
        and np.all(
            s0_geometry["median_balanced_minus_iid_correlation_difference"]
            <= float(criteria["max_s0_balanced_minus_iid_correlation_difference"])
        )
    )
    balanced_donors = donor_summary.loc[donor_summary["schedule"] == "balanced"]
    if set(balanced_donors["config"]) != set(CONFIGS) or set(donor_paired["config"]) != set(CONFIGS):
        raise PrototypeError("donor summary config surface differs")
    balance_pass = bool(
        balanced_donors["max_exposure_cv"].max()
        <= float(criteria["max_balanced_exposure_cv"])
    )
    collision_pass = bool(
        np.all(
            donor_paired["absolute_pair_collision_rate_difference"]
            <= float(criteria["max_absolute_pair_collision_rate_difference"])
        )
        and np.all(
            donor_paired["absolute_triple_collision_rate_difference"]
            <= float(criteria["max_absolute_triple_collision_rate_difference"])
        )
    )
    gate_results = {
        "independent_disjoint_half_reproducibility_gain": reproducibility_pass,
        "k44_dataset_decision_precision": precision_pass,
        "s0_spectral_geometry_nonworsening": geometry_pass,
        "donor_collision_fidelity": collision_pass,
        "exact_balanced_donor_exposure": balance_pass,
    }
    advance = all(gate_results.values())
    max_realizability = max(float(row["max_n1_realizability_error"]) for row in controls)
    max_spectrum = max(
        float(row["max_stressor_edge_spectrum_error"])
        for row in controls
        if row["max_stressor_edge_spectrum_error"] is not None
    )
    analysis = {
        "schema_version": ANALYSIS_SCHEMA,
        "study_id": spec["study_id"],
        "spec_sha256": spec_sha256,
        "epistemic_status": spec["epistemic_status"],
        "historical_locked_decision_unchanged": True,
        "real_held_out_authorized": False,
        "replacement_locked_suite_authorized": False,
        "gate_results": gate_results,
        "decision": (
            "ADVANCE_TO_V2_SPECIFICATION"
            if advance
            else "STOP_N1_OR_REDESIGN_WITHOUT_NEW_LOCKED_SUITE"
        ),
        "configs_with_disjoint_half_reproducibility_gain": reproducibility_configs,
        "controls": {
            "max_n1_realizability_error": max_realizability,
            "max_stressor_edge_spectrum_error": max_spectrum,
        },
        "row_counts": {
            "endpoint_precision.csv": len(endpoints),
            "dataset_precision.csv": len(datasets),
            "paired_schedule_comparison.csv": len(paired),
            "disjoint_half_reproducibility.csv": len(reproducibility),
            "disjoint_half_reproducibility_summary.csv": len(reproducibility_summary),
            "mcse_ratio_summary.csv": len(ratio_summary),
            "decision_precision_summary.csv": len(decision_summary),
            "endpoint_precision_summary.csv": len(endpoint_summary),
            "geometry_by_shard.csv": len(geometry),
            "geometry_paired.csv": len(geometry_paired),
            "geometry_summary.csv": len(geometry_summary),
            "donor_by_shard.csv": len(donors),
            "donor_summary.csv": len(donor_summary),
            "donor_paired.csv": len(donor_paired),
        },
        "limitations": [
            "This is a fresh synthetic development prototype, not a locked calibration suite.",
            "Only six representative cases and twelve independent datasets are used.",
            "Balanced 1-factorization changes the finite assignment design and must pass geometry checks before qualification.",
            "The primary precision comparison uses independent 22-draw halves; cycle-block MCSE is secondary because draws within a balanced cycle are dependent.",
            "The S0 equivalence half-width is a development target anchored to the frozen S2 minimum effect, not a real-data estimand.",
            "No result here can revise RETIRE_OR_DEMOTE or authorize real held-out evaluation.",
        ],
    }
    tables = {
        "endpoint_precision.csv": endpoints,
        "dataset_precision.csv": datasets,
        "paired_schedule_comparison.csv": paired,
        "disjoint_half_reproducibility.csv": reproducibility,
        "disjoint_half_reproducibility_summary.csv": reproducibility_summary,
        "mcse_ratio_summary.csv": ratio_summary,
        "decision_precision_summary.csv": decision_summary,
        "endpoint_precision_summary.csv": endpoint_summary,
        "geometry_by_shard.csv": geometry,
        "geometry_paired.csv": geometry_paired,
        "geometry_summary.csv": geometry_summary,
        "donor_by_shard.csv": donors,
        "donor_summary.csv": donor_summary,
        "donor_paired.csv": donor_paired,
    }
    return analysis, tables


def output_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def guard_output_location(output: Path, locked_study_path: Path) -> None:
    output_resolved = output.resolve()
    locked_root = locked_study_path.resolve().parent
    try:
        output_resolved.relative_to(locked_root)
    except ValueError:
        return
    raise PrototypeError("prototype output must be outside the frozen locked result root")


def implementation_hashes(repo_root: Path) -> dict[str, str]:
    paths = [
        repo_root / "analysis" / "gate12c2_v2_balanced_prototype" / "prototype.py",
        repo_root / "analysis" / "gate12c2_v2_balanced_prototype" / "prototype_spec.json",
        repo_root / "tools" / "gate12c2_minimal" / "generators.py",
        repo_root / "tools" / "gate12c2_minimal" / "metrics.py",
    ]
    return {str(path.relative_to(repo_root)).replace("\\", "/"): sha256_file(path) for path in paths}


def run(spec_path: Path, locked_study_path: Path, output: Path, repo_root: Path) -> dict[str, Any]:
    spec, spec_sha256 = load_spec(spec_path)
    study = validate_locked_study(locked_study_path, spec)
    guard_output_location(output, locked_study_path)
    cases = {case["case_id"]: case for case in study["cases"]}
    if output.exists() and not (output / "state.json").is_file():
        raise PrototypeError("existing output lacks resumable state")
    output.mkdir(parents=True, exist_ok=True)
    retained_spec = output / "prototype_spec.json"
    if retained_spec.exists():
        if sha256_file(retained_spec) != spec_sha256:
            raise PrototypeError("retained prototype spec differs")
    else:
        write_bytes_atomic(retained_spec, spec_path.read_bytes())
    metadata = {
        "schema_version": "gate12c2_v2_balanced_prototype_run_metadata_v0.1",
        "study_id": spec["study_id"],
        "spec_sha256": spec_sha256,
        "implementation_sha256": implementation_hashes(repo_root),
        "source_locked_study_path": str(locked_study_path.resolve()),
        "source_locked_study_sha256": sha256_file(locked_study_path),
    }
    metadata_path = output / "run_metadata.json"
    if metadata_path.exists():
        if load_json(metadata_path) != metadata:
            raise PrototypeError("run metadata differs")
    else:
        write_json_atomic(metadata_path, metadata)
    state_path = output / "state.json"
    expected = expected_shards(spec)
    expected_ids = {
        f"{config}__{case_id}__d{dataset:04d}"
        for config, case_id, dataset in expected
    }
    if state_path.exists():
        state = load_json(state_path)
        if state.get("schema_version") != STATE_SCHEMA or state.get("spec_sha256") != spec_sha256:
            raise PrototypeError("resume state differs")
        if state.get("status") == "COMPLETE":
            raise PrototypeError("prototype is complete; use --validate-only")
        if state.get("status") != "RUNNING":
            raise PrototypeError("prototype state is terminal and cannot resume")
        elapsed = float(state.get("cumulative_wall_seconds", 0.0))
        completed = set(state.get("completed_shards", []))
        if not completed <= expected_ids:
            raise PrototypeError("resume state contains an unexpected shard")
    else:
        elapsed = 0.0
        completed: set[str] = set()
        state = {
            "schema_version": STATE_SCHEMA,
            "study_id": spec["study_id"],
            "spec_sha256": spec_sha256,
            "status": "RUNNING",
            "completed_shards": [],
            "cumulative_wall_seconds": 0.0,
        }
        write_json_atomic(state_path, state)
    invocation_start = time.perf_counter()

    def stop_for_cap(message: str) -> None:
        state.update(
            {
                "status": "CAP_REACHED",
                "completed_shards": sorted(completed),
                "cumulative_wall_seconds": elapsed + time.perf_counter() - invocation_start,
                "output_bytes": output_bytes(output),
                "terminal_reason": message,
            }
        )
        write_json_atomic(state_path, state, replace=True)
        raise PrototypeError(message)

    for index, (config, case_id, dataset) in enumerate(expected, start=1):
        shard_id = f"{config}__{case_id}__d{dataset:04d}"
        path = shard_path(output, config, case_id, dataset)
        if path.exists():
            value = load_json(path)
            validate_shard(
                value,
                config=config,
                case_id=case_id,
                dataset=dataset,
                spec=spec,
                spec_sha256=spec_sha256,
            )
            completed.add(shard_id)
            continue
        current_elapsed = elapsed + time.perf_counter() - invocation_start
        if current_elapsed >= float(spec["resource_cap"]["max_wall_seconds"]):
            stop_for_cap("prototype wall-time cap reached before completion")
        value = run_shard(
            config=config,
            case=cases[case_id],
            dataset_index=dataset,
            spec=spec,
            spec_sha256=spec_sha256,
            study=study,
        )
        validate_shard(
            value,
            config=config,
            case_id=case_id,
            dataset=dataset,
            spec=spec,
            spec_sha256=spec_sha256,
        )
        write_json_atomic(path, value)
        completed.add(shard_id)
        state.update(
            {
                "status": "RUNNING",
                "completed_shards": sorted(completed),
                "cumulative_wall_seconds": elapsed + time.perf_counter() - invocation_start,
                "last_completed": shard_id,
                "progress": f"{index}/{len(expected)}",
            }
        )
        write_json_atomic(state_path, state, replace=True)
        if elapsed + time.perf_counter() - invocation_start >= float(
            spec["resource_cap"]["max_wall_seconds"]
        ):
            stop_for_cap("prototype wall-time cap reached before completion")
        if output_bytes(output) > int(spec["resource_cap"]["max_output_bytes"]):
            stop_for_cap("prototype output cap exceeded")
    if completed != expected_ids:
        raise PrototypeError("prototype shard surface is incomplete")
    analysis, tables = analyze(output, spec, spec_sha256)
    for name, frame in tables.items():
        write_bytes_atomic(
            output / name,
            frame.to_csv(index=False).encode("utf-8"),
            replace=True,
        )
    write_json_atomic(output / "analysis.json", analysis, replace=True)
    if output_bytes(output) > int(spec["resource_cap"]["max_output_bytes"]):
        stop_for_cap("prototype output cap exceeded during analysis")
    files = {
        str(path.relative_to(output)).replace("\\", "/"): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "state.json"}
    }
    manifest = {
        "schema_version": "gate12c2_v2_balanced_prototype_manifest_v0.1",
        "study_id": spec["study_id"],
        "spec_sha256": spec_sha256,
        "analysis_sha256": files["analysis.json"],
        "files": files,
    }
    write_json_atomic(output / "manifest.json", manifest, replace=True)
    if elapsed + time.perf_counter() - invocation_start >= float(
        spec["resource_cap"]["max_wall_seconds"]
    ):
        stop_for_cap("prototype wall-time cap reached during analysis")
    state.update(
        {
            "status": "COMPLETE",
            "completed_shards": sorted(completed),
            "cumulative_wall_seconds": elapsed + time.perf_counter() - invocation_start,
            "analysis_sha256": files["analysis.json"],
            "manifest_sha256": sha256_file(output / "manifest.json"),
            "output_bytes": output_bytes(output),
        }
    )
    write_json_atomic(state_path, state, replace=True)
    if output_bytes(output) > int(spec["resource_cap"]["max_output_bytes"]):
        stop_for_cap("prototype output cap exceeded at completion")
    return {"analysis": analysis, "state": state, "manifest": manifest}


def validate_output(spec_path: Path, output: Path, repo_root: Path) -> dict[str, Any]:
    spec, spec_sha256 = load_spec(spec_path)
    state = load_json(output / "state.json")
    manifest = load_json(output / "manifest.json")
    analysis = load_json(output / "analysis.json")
    metadata = load_json(output / "run_metadata.json")
    expected = expected_shards(spec)
    expected_ids = {
        f"{config}__{case_id}__d{dataset:04d}"
        for config, case_id, dataset in expected
    }
    if (
        state.get("schema_version") != STATE_SCHEMA
        or state.get("status") != "COMPLETE"
        or set(state.get("completed_shards", [])) != expected_ids
    ):
        raise PrototypeError("prototype output is incomplete")
    if (
        manifest.get("schema_version") != "gate12c2_v2_balanced_prototype_manifest_v0.1"
        or analysis.get("schema_version") != ANALYSIS_SCHEMA
        or manifest.get("study_id") != spec["study_id"]
        or analysis.get("study_id") != spec["study_id"]
        or manifest.get("spec_sha256") != spec_sha256
        or analysis.get("spec_sha256") != spec_sha256
    ):
        raise PrototypeError("prototype output spec differs")
    if (
        analysis.get("historical_locked_decision_unchanged") is not True
        or analysis.get("real_held_out_authorized") is not False
        or analysis.get("replacement_locked_suite_authorized") is not False
        or analysis.get("decision")
        not in {
            "ADVANCE_TO_V2_SPECIFICATION",
            "STOP_N1_OR_REDESIGN_WITHOUT_NEW_LOCKED_SUITE",
        }
    ):
        raise PrototypeError("prototype epistemic boundary differs")
    if (
        metadata.get("schema_version")
        != "gate12c2_v2_balanced_prototype_run_metadata_v0.1"
        or metadata.get("study_id") != spec["study_id"]
        or metadata.get("spec_sha256") != spec_sha256
        or metadata.get("source_locked_study_sha256") != spec["source_locked_study_sha256"]
        or metadata.get("implementation_sha256") != implementation_hashes(repo_root)
    ):
        raise PrototypeError("prototype run metadata differs")
    locked_study_path = Path(str(metadata["source_locked_study_path"]))
    guard_output_location(output, locked_study_path)
    validate_locked_study(locked_study_path, spec)
    for relative, expected_hash in manifest["files"].items():
        path = (output / relative).resolve()
        try:
            path.relative_to(output.resolve())
        except ValueError as exc:
            raise PrototypeError("manifest path escapes output") from exc
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise PrototypeError(f"prototype manifest hash differs: {relative}")
    actual_files = {
        str(path.relative_to(output)).replace("\\", "/")
        for path in output.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "state.json"}
    }
    if actual_files != set(manifest["files"]):
        raise PrototypeError("prototype manifest file surface differs")
    if sha256_file(output / "analysis.json") != manifest["analysis_sha256"]:
        raise PrototypeError("prototype analysis hash differs")
    if (
        state.get("analysis_sha256") != manifest["analysis_sha256"]
        or state.get("manifest_sha256") != sha256_file(output / "manifest.json")
    ):
        raise PrototypeError("prototype terminal state hash differs")
    for config, case_id, dataset in expected:
        validate_shard(
            load_json(shard_path(output, config, case_id, dataset)),
            config=config,
            case_id=case_id,
            dataset=dataset,
            spec=spec,
            spec_sha256=spec_sha256,
        )
    for name, expected_rows in analysis["row_counts"].items():
        if len(pd.read_csv(output / name)) != int(expected_rows):
            raise PrototypeError(f"prototype table row count differs: {name}")
    return {
        "status": "pass",
        "decision": analysis["decision"],
        "analysis_sha256": manifest["analysis_sha256"],
        "manifest_sha256": sha256_file(output / "manifest.json"),
        "file_count": len(manifest["files"]),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--locked-study", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.validate_only:
        result = validate_output(
            args.spec.resolve(),
            args.output.resolve(),
            args.repo_root.resolve(),
        )
    else:
        if args.locked_study is None:
            raise PrototypeError("--locked-study is required for execution")
        result = run(
            args.spec.resolve(),
            args.locked_study.resolve(),
            args.output.resolve(),
            args.repo_root.resolve(),
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
