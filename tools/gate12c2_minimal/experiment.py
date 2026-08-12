"""One small, complete 24-endpoint synthetic development experiment."""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .generators import (
    Graph,
    edge_singular_values,
    edge_spectrum_error,
    gauge_transform,
    generate_s0_cohort,
    generate_s1_cohort,
    graph_digest,
    independent_edge_reorientation,
    joint_realizability_error,
    n1_reassignment,
)
from .io import canonical_json_bytes, load_json, sha256_file
from .metrics import ResidualDiagnostics, residual_diagnostics


STUDY_SCHEMA = "gate12c2_minimal_study_v0.1"
SHARD_SCHEMA = "gate12c2_minimal_shard_v0.1"
RESULT_SCHEMA = "gate12c2_minimal_result_v0.1"
COMPONENT_SCHEMA = "gate12c2_minimal_component_row_v0.1"
REGIMES = ("S0", "S1", "S2")


class Gate12C2ExperimentError(ValueError):
    """Raised when a study or shard violates the minimal contract."""


def _integer(value: object, label: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Gate12C2ExperimentError(f"{label} must be an integer >= {minimum}")
    return value


def _finite(value: object, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Gate12C2ExperimentError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise Gate12C2ExperimentError(f"{label} is outside its allowed range")
    return result


def validate_spec(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise Gate12C2ExperimentError("study specification must be an object")
    required = {
        "schema_version",
        "study_id",
        "epistemic_status",
        "seed_namespace",
        "alternative",
        "q_values",
        "epsilon",
        "spectral_gap_tolerance",
        "numerical_tolerance",
        "outer_count",
        "cohort_size",
        "inner_draws",
        "stressor_trials",
        "s1",
        "smoke_acceptance",
        "cases",
    }
    if set(value) != required:
        raise Gate12C2ExperimentError("study specification fields differ")
    if value["schema_version"] != STUDY_SCHEMA:
        raise Gate12C2ExperimentError("unsupported study schema")
    if value["epistemic_status"] != "synthetic_development_smoke_only":
        raise Gate12C2ExperimentError("study must remain development-only")
    if value["alternative"] != "observed_smaller_than_null":
        raise Gate12C2ExperimentError("unexpected directional alternative")
    for label in ("study_id", "seed_namespace"):
        if not isinstance(value[label], str) or not value[label]:
            raise Gate12C2ExperimentError(f"{label} must be nonempty text")
    if value["q_values"] != [1, 2]:
        raise Gate12C2ExperimentError("the minimal study requires q=[1,2]")

    _finite(value["epsilon"], "epsilon", minimum=0.0)
    _finite(
        value["spectral_gap_tolerance"],
        "spectral_gap_tolerance",
        minimum=0.0,
    )
    _finite(value["numerical_tolerance"], "numerical_tolerance", minimum=0.0)
    _integer(value["outer_count"], "outer_count", 1)
    _integer(value["cohort_size"], "cohort_size", 4)
    _integer(value["inner_draws"], "inner_draws", 1)
    _integer(value["stressor_trials"], "stressor_trials", 1)

    s1 = value["s1"]
    if not isinstance(s1, dict) or set(s1) != {
        "effect_strength",
        "observed_mismatch",
    }:
        raise Gate12C2ExperimentError("invalid S1 specification")
    _finite(s1["effect_strength"], "s1.effect_strength", minimum=0.0)
    _finite(s1["observed_mismatch"], "s1.observed_mismatch", minimum=0.0)

    acceptance = value["smoke_acceptance"]
    acceptance_fields = {
        "holm_alpha",
        "zero_tolerance",
        "s0_max_supported_endpoints",
        "s1_min_directional_endpoint_fraction",
        "s2_min_log_inflation",
        "s2_min_inflation_endpoint_fraction",
        "s2_min_channel_fraction",
    }
    if not isinstance(acceptance, dict) or set(acceptance) != acceptance_fields:
        raise Gate12C2ExperimentError("invalid smoke acceptance specification")
    _finite(acceptance["holm_alpha"], "holm_alpha", minimum=0.0)
    _finite(acceptance["zero_tolerance"], "zero_tolerance", minimum=0.0)
    _integer(
        acceptance["s0_max_supported_endpoints"],
        "s0_max_supported_endpoints",
        0,
    )
    for label in (
        "s1_min_directional_endpoint_fraction",
        "s2_min_log_inflation",
        "s2_min_inflation_endpoint_fraction",
        "s2_min_channel_fraction",
    ):
        _finite(acceptance[label], label, minimum=0.0)

    cases = value["cases"]
    if not isinstance(cases, list) or len(cases) != 12:
        raise Gate12C2ExperimentError("study must declare exactly twelve cases")
    case_ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) != {
            "case_id",
            "model",
            "family",
            "ambient_dim",
            "local_rank",
            "s0_frame_noise",
            "s2_frame_noise",
        }:
            raise Gate12C2ExperimentError("invalid case specification")
        for label in ("case_id", "model", "family"):
            if not isinstance(case[label], str) or not case[label]:
                raise Gate12C2ExperimentError(f"case {label} must be text")
        if case["case_id"] in case_ids:
            raise Gate12C2ExperimentError("duplicate case_id")
        case_ids.add(case["case_id"])
        ambient = _integer(case["ambient_dim"], "ambient_dim", 4)
        local = _integer(case["local_rank"], "local_rank", 3)
        if ambient <= local:
            raise Gate12C2ExperimentError("ambient_dim must exceed local_rank")
        _finite(case["s0_frame_noise"], "s0_frame_noise", minimum=0.0)
        _finite(case["s2_frame_noise"], "s2_frame_noise", minimum=0.0)
    return copy.deepcopy(value)


def load_spec(path: Path) -> tuple[dict[str, Any], str]:
    return validate_spec(load_json(path)), sha256_file(path)


def expected_shard_ids(spec: Mapping[str, Any]) -> list[str]:
    return [
        f"{case['case_id']}__{regime}"
        for case in spec["cases"]
        for regime in REGIMES
    ]


def _diagnostic(
    edges: Sequence[np.ndarray], q: int, spec: Mapping[str, Any]
) -> ResidualDiagnostics:
    return residual_diagnostics(
        edges[0],
        edges[1],
        edges[2],
        q,
        spectral_gap_tolerance=float(spec["spectral_gap_tolerance"]),
        numerical_tolerance=float(spec["numerical_tolerance"]),
    )


def _sign_p(values: Sequence[float], *, negative: bool, tolerance: float) -> float:
    signs = [value for value in values if abs(value) > tolerance]
    if not signs:
        return 1.0
    directional = sum(value < 0.0 if negative else value > 0.0 for value in signs)
    numerator = sum(
        math.comb(len(signs), count) for count in range(directional, len(signs) + 1)
    )
    return float(numerator / (2 ** len(signs)))


def _channel_moved(
    baseline: ResidualDiagnostics,
    stressor: ResidualDiagnostics,
    tolerance: float,
) -> bool:
    if stressor.x > baseline.x + tolerance or stressor.y > baseline.y + tolerance:
        return True
    return bool(
        baseline.c is not None
        and stressor.c is not None
        and stressor.c < baseline.c - tolerance
    )


def _component_fields(diagnostic: ResidualDiagnostics) -> dict[str, Any]:
    return {
        "a": diagnostic.a,
        "u": diagnostic.u,
        "v": diagnostic.v,
        "x": diagnostic.x,
        "y": diagnostic.y,
        "c": diagnostic.c,
        "p_left": diagnostic.p_left,
        "p_right": diagnostic.p_right,
        "relative_gap_left": diagnostic.relative_gap_left,
        "relative_gap_right": diagnostic.relative_gap_right,
        "product_singular_values_left": list(
            diagnostic.product_singular_values_left
        ),
        "product_singular_values_right": list(
            diagnostic.product_singular_values_right
        ),
        "eligible": diagnostic.eligible,
        "numerical_pass": diagnostic.numerical_pass,
        "matrix_identity_error": diagnostic.matrix_identity_error,
        "squared_identity_error": diagnostic.squared_identity_error,
    }


def _component_row(
    *,
    case_id: str,
    regime: str,
    outer_index: int,
    graph_index: int,
    draw_index: int,
    q: int,
    observed: ResidualDiagnostics,
    observed_gauge: ResidualDiagnostics,
    null: ResidualDiagnostics,
    edge_spectra_observed: Sequence[Sequence[float]],
    edge_spectra_null: Sequence[Sequence[float]],
    epsilon: float,
    gauge_defect_error: float,
    gauge_component_error: float,
    stressor: ResidualDiagnostics | None = None,
    edge_spectra_stressor: Sequence[Sequence[float]] | None = None,
    channel_moved: bool | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": COMPONENT_SCHEMA,
        "case_id": case_id,
        "regime": regime,
        "dataset_id": f"{case_id}:{regime}:d{outer_index:04d}",
        "outer_index": int(outer_index),
        "graph_index": int(graph_index),
        "draw_index": int(draw_index),
        "q": int(q),
        "observed": _component_fields(observed),
        "observed_gauge": _component_fields(observed_gauge),
        "null": _component_fields(null),
        "stressor": None if stressor is None else _component_fields(stressor),
        "edge_singular_values_observed": [
            list(values) for values in edge_spectra_observed
        ],
        "edge_singular_values_null": [
            list(values) for values in edge_spectra_null
        ],
        "edge_singular_values_stressor": (
            None
            if edge_spectra_stressor is None
            else [list(values) for values in edge_spectra_stressor]
        ),
        "observed_to_null_log_defect": (
            math.log(observed.a + epsilon) - math.log(null.a + epsilon)
        ),
        "stressor_to_null_log_defect": (
            None
            if stressor is None
            else math.log(stressor.a + epsilon) - math.log(null.a + epsilon)
        ),
        "gauge_defect_error": float(gauge_defect_error),
        "gauge_component_max_error": float(gauge_component_error),
        "inflation_channel_moved": channel_moved,
    }


def _optional_error(left: float | None, right: float | None) -> float:
    if left is None and right is None:
        return 0.0
    if left is None or right is None:
        return math.inf
    return abs(left - right)


def _gauge_errors(
    original: ResidualDiagnostics, transformed: ResidualDiagnostics
) -> tuple[float, float]:
    defect_error = abs(original.a - transformed.a)
    component_error = max(
        abs(original.u - transformed.u),
        abs(original.v - transformed.v),
        abs(original.x - transformed.x),
        abs(original.y - transformed.y),
        _optional_error(original.c, transformed.c),
        _optional_error(original.p_left, transformed.p_left),
        _optional_error(original.p_right, transformed.p_right),
    )
    return float(defect_error), float(component_error)


def _endpoint_summary(
    *,
    q: int,
    values_by_block: Mapping[str, Sequence[float]],
    numerical_failures: int,
    ineligible: int,
    realizability_error: float,
    zero_tolerance: float,
    negative: bool,
    channel_by_block: Mapping[str, Sequence[bool]] | None = None,
) -> dict[str, Any]:
    block_values = [
        float(np.median(values_by_block[key])) for key in sorted(values_by_block)
    ]
    channel_fraction = None
    if channel_by_block is not None:
        block_channels = [
            float(np.mean(channel_by_block[key])) for key in sorted(channel_by_block)
        ]
        channel_fraction = float(np.mean(block_channels))
    coverage_complete = bool(
        block_values
        and numerical_failures == 0
        and ineligible == 0
        and realizability_error <= 1e-10
    )
    return {
        "q": int(q),
        "coverage_complete": coverage_complete,
        "block_count": len(block_values),
        "median_effect": float(np.median(block_values)),
        "directional_fraction": float(
            np.mean(
                [
                    value < -zero_tolerance if negative else value > zero_tolerance
                    for value in block_values
                ]
            )
        ),
        "directional_sign_p": _sign_p(
            block_values, negative=negative, tolerance=zero_tolerance
        ),
        "channel_fraction": channel_fraction,
        "numerical_failure_count": int(numerical_failures),
        "ineligible_count": int(ineligible),
    }


def run_dataset(
    spec: Mapping[str, Any],
    spec_sha256: str,
    case: Mapping[str, Any],
    regime: str,
    dataset_index: int,
) -> dict[str, Any]:
    if regime not in REGIMES:
        raise Gate12C2ExperimentError(f"unsupported regime: {regime}")
    epsilon = float(spec["epsilon"])
    zero_tolerance = float(spec["smoke_acceptance"]["zero_tolerance"])
    q_values = [int(q) for q in spec["q_values"]]
    values: dict[int, dict[str, list[float]]] = {
        q: defaultdict(list) for q in q_values
    }
    channels: dict[int, dict[str, list[bool]]] = {
        q: defaultdict(list) for q in q_values
    }
    numerical_failures = {q: 0 for q in q_values}
    ineligible = {q: 0 for q in q_values}
    max_realizability_error = 0.0
    max_spectrum_error = 0.0
    max_gauge_defect_error = 0.0
    max_gauge_component_error = 0.0
    observed_digests: list[str] = []
    component_rows: list[dict[str, Any]] = []

    for outer_index in (dataset_index,):
        if regime == "S0":
            observed = generate_s0_cohort(
                case=case,
                seed_namespace=spec["seed_namespace"],
                outer_index=outer_index,
                cohort_size=int(spec["cohort_size"]),
                frame_noise=float(case["s0_frame_noise"]),
                regime="S0",
            )
        elif regime == "S1":
            observed = generate_s1_cohort(
                case=case,
                seed_namespace=spec["seed_namespace"],
                outer_index=outer_index,
                cohort_size=int(spec["cohort_size"]),
                effect_strength=float(spec["s1"]["effect_strength"]),
                observed_mismatch=float(spec["s1"]["observed_mismatch"]),
            )
        else:
            observed = generate_s0_cohort(
                case=case,
                seed_namespace=spec["seed_namespace"],
                outer_index=outer_index,
                cohort_size=int(spec["cohort_size"]),
                frame_noise=float(case["s2_frame_noise"]),
                regime="S2",
            )
        observed_digest = graph_digest(observed)
        observed_digests.append(observed_digest)
        observed_diagnostics = {
            (graph_index, q): _diagnostic(graph.edges, q, spec)
            for graph_index, graph in enumerate(observed)
            for q in q_values
        }
        observed_gauge = tuple(
            gauge_transform(
                graph,
                seed_namespace=spec["seed_namespace"],
                case_id=case["case_id"],
                dataset_index=outer_index,
                graph_index=graph_index,
            )
            for graph_index, graph in enumerate(observed)
        )
        observed_gauge_diagnostics = {
            (graph_index, q): _diagnostic(graph.edges, q, spec)
            for graph_index, graph in enumerate(observed_gauge)
            for q in q_values
        }
        gauge_errors = {
            (graph_index, q): _gauge_errors(
                observed_diagnostics[(graph_index, q)],
                observed_gauge_diagnostics[(graph_index, q)],
            )
            for graph_index in range(len(observed))
            for q in q_values
        }
        max_gauge_defect_error = max(
            max_gauge_defect_error,
            max(value[0] for value in gauge_errors.values()),
        )
        max_gauge_component_error = max(
            max_gauge_component_error,
            max(value[1] for value in gauge_errors.values()),
        )

        for draw_index in range(int(spec["inner_draws"])):
            null_graphs = n1_reassignment(
                observed,
                seed_namespace=spec["seed_namespace"],
                case_id=case["case_id"],
                regime=regime,
                outer_index=outer_index,
                draw_index=draw_index,
            )
            max_realizability_error = max(
                max_realizability_error,
                max(joint_realizability_error(graph) for graph in null_graphs),
            )
            for graph_index, null_graph in enumerate(null_graphs):
                block_id = f"o{outer_index:03d}:r{graph_index:03d}"
                if regime in {"S0", "S1"}:
                    for q in q_values:
                        observed_diag = observed_diagnostics[(graph_index, q)]
                        null_diag = _diagnostic(null_graph.edges, q, spec)
                        numerical_failures[q] += int(
                            not observed_diag.numerical_pass or not null_diag.numerical_pass
                        )
                        ineligible[q] += int(
                            not observed_diag.eligible or not null_diag.eligible
                        )
                        values[q][block_id].append(
                            math.log(observed_diag.a + epsilon)
                            - math.log(null_diag.a + epsilon)
                        )
                        component_rows.append(
                            _component_row(
                                case_id=case["case_id"],
                                regime=regime,
                                outer_index=outer_index,
                                graph_index=graph_index,
                                draw_index=draw_index,
                                q=q,
                                observed=observed_diag,
                                observed_gauge=observed_gauge_diagnostics[
                                    (graph_index, q)
                                ],
                                null=null_diag,
                                edge_spectra_observed=edge_singular_values(
                                    observed[graph_index].edges
                                ),
                                edge_spectra_null=edge_singular_values(
                                    null_graph.edges
                                ),
                                epsilon=epsilon,
                                gauge_defect_error=gauge_errors[(graph_index, q)][0],
                                gauge_component_error=gauge_errors[(graph_index, q)][1],
                            )
                        )
                    continue

                baseline = {q: _diagnostic(null_graph.edges, q, spec) for q in q_values}
                candidates: list[
                    tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray], dict[int, ResidualDiagnostics]]
                ] = []
                for trial_index in range(int(spec["stressor_trials"])):
                    stress_edges = independent_edge_reorientation(
                        null_graph.edges,
                        seed_namespace=spec["seed_namespace"],
                        case_id=case["case_id"],
                        outer_index=outer_index,
                        draw_index=draw_index,
                        graph_index=graph_index,
                        trial_index=trial_index,
                    )
                    stress_diagnostics = {
                        q: _diagnostic(stress_edges, q, spec) for q in q_values
                    }
                    score = sum(
                        math.log(stress_diagnostics[q].a + epsilon) for q in q_values
                    )
                    candidates.append((score, stress_edges, stress_diagnostics))
                _, stress_edges, stress = max(candidates, key=lambda row: row[0])
                max_spectrum_error = max(
                    max_spectrum_error,
                    edge_spectrum_error(null_graph.edges, stress_edges),
                )
                for q in q_values:
                    numerical_failures[q] += int(
                        not baseline[q].numerical_pass or not stress[q].numerical_pass
                    )
                    ineligible[q] += int(
                        not baseline[q].eligible or not stress[q].eligible
                    )
                    values[q][block_id].append(
                        math.log(stress[q].a + epsilon)
                        - math.log(baseline[q].a + epsilon)
                    )
                    channels[q][block_id].append(
                        _channel_moved(baseline[q], stress[q], zero_tolerance)
                    )
                    component_rows.append(
                        _component_row(
                            case_id=case["case_id"],
                            regime=regime,
                            outer_index=outer_index,
                            graph_index=graph_index,
                            draw_index=draw_index,
                            q=q,
                            observed=observed_diagnostics[(graph_index, q)],
                            observed_gauge=observed_gauge_diagnostics[
                                (graph_index, q)
                            ],
                            null=baseline[q],
                            stressor=stress[q],
                            edge_spectra_observed=edge_singular_values(
                                observed[graph_index].edges
                            ),
                            edge_spectra_null=edge_singular_values(null_graph.edges),
                            edge_spectra_stressor=edge_singular_values(stress_edges),
                            epsilon=epsilon,
                            gauge_defect_error=gauge_errors[(graph_index, q)][0],
                            gauge_component_error=gauge_errors[(graph_index, q)][1],
                            channel_moved=channels[q][block_id][-1],
                        )
                    )
        if graph_digest(observed) != observed_digest:
            raise Gate12C2ExperimentError("observed arm mutated during null generation")

    endpoints = [
        _endpoint_summary(
            q=q,
            values_by_block=values[q],
            numerical_failures=numerical_failures[q],
            ineligible=ineligible[q],
            realizability_error=max_realizability_error,
            zero_tolerance=zero_tolerance,
            negative=regime in {"S0", "S1"},
            channel_by_block=channels[q] if regime == "S2" else None,
        )
        for q in q_values
    ]
    return {
        "schema_version": SHARD_SCHEMA,
        "study_id": spec["study_id"],
        "study_sha256": spec_sha256,
        "shard_id": f"{case['case_id']}__{regime}__d{dataset_index:04d}",
        "case": dict(case),
        "regime": regime,
        "dataset_index": int(dataset_index),
        "alternative": spec["alternative"],
        "observed_cohort_sha256": observed_digests,
        "observed_arm_unchanged": True,
        "n1_realizability_max_error": float(max_realizability_error),
        "gauge_defect_max_error": float(max_gauge_defect_error),
        "gauge_component_max_error": float(max_gauge_component_error),
        "stressor_edge_spectrum_max_error": (
            float(max_spectrum_error) if regime == "S2" else None
        ),
        "endpoint_count": len(endpoints),
        "endpoints": endpoints,
        "component_row_count": len(component_rows),
        "component_rows": component_rows,
    }


def run_shard(
    spec: Mapping[str, Any],
    spec_sha256: str,
    case: Mapping[str, Any],
    regime: str,
) -> dict[str, Any]:
    """Compatibility wrapper for the original smoke's multi-dataset shard."""

    datasets = [
        run_dataset(spec, spec_sha256, case, regime, dataset_index)
        for dataset_index in range(int(spec["outer_count"]))
    ]
    if len(datasets) == 1:
        shard = copy.deepcopy(datasets[0])
        shard["shard_id"] = f"{case['case_id']}__{regime}"
        return shard

    component_rows = [
        row for dataset in datasets for row in dataset["component_rows"]
    ]
    values: dict[int, dict[str, list[float]]] = {
        int(q): defaultdict(list) for q in spec["q_values"]
    }
    channels: dict[int, dict[str, list[bool]]] = {
        int(q): defaultdict(list) for q in spec["q_values"]
    }
    for row in component_rows:
        q = int(row["q"])
        block_id = f"o{row['outer_index']:03d}:r{row['graph_index']:03d}"
        value = (
            row["stressor_to_null_log_defect"]
            if regime == "S2"
            else row["observed_to_null_log_defect"]
        )
        values[q][block_id].append(float(value))
        if regime == "S2":
            channels[q][block_id].append(bool(row["inflation_channel_moved"]))
    endpoints = [
        _endpoint_summary(
            q=int(q),
            values_by_block=values[int(q)],
            numerical_failures=sum(
                int(
                    not row["observed"]["numerical_pass"]
                    or not row["null"]["numerical_pass"]
                    or (
                        row["stressor"] is not None
                        and not row["stressor"]["numerical_pass"]
                    )
                )
                for row in component_rows
                if int(row["q"]) == int(q)
            ),
            ineligible=sum(
                int(
                    not row["observed"]["eligible"]
                    or not row["null"]["eligible"]
                    or (
                        row["stressor"] is not None
                        and not row["stressor"]["eligible"]
                    )
                )
                for row in component_rows
                if int(row["q"]) == int(q)
            ),
            realizability_error=max(
                float(dataset["n1_realizability_max_error"])
                for dataset in datasets
            ),
            zero_tolerance=float(spec["smoke_acceptance"]["zero_tolerance"]),
            negative=regime in {"S0", "S1"},
            channel_by_block=channels[int(q)] if regime == "S2" else None,
        )
        for q in spec["q_values"]
    ]
    return {
        "schema_version": SHARD_SCHEMA,
        "study_id": spec["study_id"],
        "study_sha256": spec_sha256,
        "shard_id": f"{case['case_id']}__{regime}",
        "case": dict(case),
        "regime": regime,
        "alternative": spec["alternative"],
        "observed_cohort_sha256": [
            digest
            for dataset in datasets
            for digest in dataset["observed_cohort_sha256"]
        ],
        "observed_arm_unchanged": True,
        "n1_realizability_max_error": max(
            float(dataset["n1_realizability_max_error"]) for dataset in datasets
        ),
        "gauge_defect_max_error": max(
            float(dataset["gauge_defect_max_error"]) for dataset in datasets
        ),
        "gauge_component_max_error": max(
            float(dataset["gauge_component_max_error"]) for dataset in datasets
        ),
        "stressor_edge_spectrum_max_error": (
            max(
                float(dataset["stressor_edge_spectrum_max_error"])
                for dataset in datasets
            )
            if regime == "S2"
            else None
        ),
        "endpoint_count": len(endpoints),
        "endpoints": endpoints,
        "component_row_count": len(component_rows),
        "component_rows": component_rows,
    }


def validate_shard(
    shard: object,
    *,
    spec: Mapping[str, Any],
    spec_sha256: str,
    shard_id: str,
) -> dict[str, Any]:
    if not isinstance(shard, dict):
        raise Gate12C2ExperimentError("shard must be an object")
    if shard.get("schema_version") != SHARD_SCHEMA:
        raise Gate12C2ExperimentError("invalid shard schema")
    if shard.get("study_id") != spec["study_id"]:
        raise Gate12C2ExperimentError("shard study mismatch")
    if shard.get("study_sha256") != spec_sha256:
        raise Gate12C2ExperimentError("shard study hash mismatch")
    if shard.get("shard_id") != shard_id:
        raise Gate12C2ExperimentError("shard identifier mismatch")
    if shard.get("endpoint_count") != 2 or not isinstance(
        shard.get("endpoints"), list
    ):
        raise Gate12C2ExperimentError("shard endpoint surface is incomplete")
    if {row.get("q") for row in shard["endpoints"]} != {1, 2}:
        raise Gate12C2ExperimentError("shard q surface is incomplete")
    expected_rows = (
        int(spec["outer_count"])
        * int(spec["cohort_size"])
        * int(spec["inner_draws"])
        * len(spec["q_values"])
    )
    if shard.get("component_row_count") != expected_rows:
        raise Gate12C2ExperimentError("shard component count differs")
    _validate_component_surface(shard, spec=spec)
    return copy.deepcopy(shard)


def validate_dataset_shard(
    shard: object,
    *,
    spec: Mapping[str, Any],
    spec_sha256: str,
    shard_id: str,
) -> dict[str, Any]:
    if not isinstance(shard, dict):
        raise Gate12C2ExperimentError("dataset shard must be an object")
    if shard.get("schema_version") != SHARD_SCHEMA:
        raise Gate12C2ExperimentError("invalid dataset shard schema")
    if shard.get("study_id") != spec["study_id"]:
        raise Gate12C2ExperimentError("dataset shard study mismatch")
    if shard.get("study_sha256") != spec_sha256:
        raise Gate12C2ExperimentError("dataset shard study hash mismatch")
    if shard.get("shard_id") != shard_id:
        raise Gate12C2ExperimentError("dataset shard identifier mismatch")
    if not isinstance(shard.get("dataset_index"), int):
        raise Gate12C2ExperimentError("dataset shard index is missing")
    expected_rows = (
        int(spec["cohort_size"])
        * int(spec["inner_draws"])
        * len(spec["q_values"])
    )
    if shard.get("component_row_count") != expected_rows:
        raise Gate12C2ExperimentError("dataset component surface is incomplete")
    rows = shard.get("component_rows")
    if not isinstance(rows, list) or len(rows) != expected_rows:
        raise Gate12C2ExperimentError("dataset component rows are incomplete")
    expected_keys = {
        (
            int(shard["dataset_index"]),
            graph_index,
            draw_index,
            int(q),
        )
        for graph_index in range(int(spec["cohort_size"]))
        for draw_index in range(int(spec["inner_draws"]))
        for q in spec["q_values"]
    }
    actual_keys = {
        (
            row.get("outer_index"),
            row.get("graph_index"),
            row.get("draw_index"),
            row.get("q"),
        )
        for row in rows
        if isinstance(row, dict)
    }
    if actual_keys != expected_keys:
        raise Gate12C2ExperimentError("dataset component row keys differ")
    if shard.get("endpoint_count") != len(spec["q_values"]):
        raise Gate12C2ExperimentError("dataset endpoint count differs")
    _validate_component_surface(shard, spec=spec)
    return copy.deepcopy(shard)


def _validate_component_surface(
    shard: Mapping[str, Any], *, spec: Mapping[str, Any]
) -> None:
    rows = shard.get("component_rows")
    if not isinstance(rows, list) or not rows:
        raise Gate12C2ExperimentError("component rows are absent")
    regime = str(shard["regime"])
    q_values = {int(value) for value in spec["q_values"]}
    expected_component_fields = {
        "a",
        "u",
        "v",
        "x",
        "y",
        "c",
        "p_left",
        "p_right",
        "relative_gap_left",
        "relative_gap_right",
        "product_singular_values_left",
        "product_singular_values_right",
        "eligible",
        "numerical_pass",
        "matrix_identity_error",
        "squared_identity_error",
    }
    values: dict[int, dict[str, list[float]]] = {
        q: defaultdict(list) for q in q_values
    }
    channels: dict[int, dict[str, list[bool]]] = {
        q: defaultdict(list) for q in q_values
    }
    numerical_failures = {q: 0 for q in q_values}
    ineligible = {q: 0 for q in q_values}
    row_keys: set[tuple[int, int, int, int]] = set()
    max_gauge_defect = 0.0
    max_gauge_component = 0.0
    epsilon = float(spec["epsilon"])
    zero_tolerance = float(spec["smoke_acceptance"]["zero_tolerance"])

    for row in rows:
        if not isinstance(row, dict) or row.get("schema_version") != COMPONENT_SCHEMA:
            raise Gate12C2ExperimentError("invalid component row schema")
        q = int(row.get("q"))
        if q not in q_values:
            raise Gate12C2ExperimentError("unexpected q in component row")
        key = (
            int(row.get("outer_index")),
            int(row.get("graph_index")),
            int(row.get("draw_index")),
            q,
        )
        if key in row_keys:
            raise Gate12C2ExperimentError("duplicate component row")
        row_keys.add(key)
        if row.get("case_id") != shard["case"]["case_id"]:
            raise Gate12C2ExperimentError("component case differs")
        if row.get("regime") != regime:
            raise Gate12C2ExperimentError("component regime differs")

        arms: list[Mapping[str, Any]] = []
        for arm_name in ("observed", "observed_gauge", "null"):
            arm = row.get(arm_name)
            if not isinstance(arm, dict) or set(arm) != expected_component_fields:
                raise Gate12C2ExperimentError(f"invalid {arm_name} component")
            arms.append(arm)
        stressor = row.get("stressor")
        if regime == "S2":
            if not isinstance(stressor, dict) or set(stressor) != expected_component_fields:
                raise Gate12C2ExperimentError("S2 stressor component is missing")
            arms.append(stressor)
        elif stressor is not None:
            raise Gate12C2ExperimentError("non-S2 component has a stressor")

        for arm in arms:
            numeric = [
                float(arm[name])
                for name in ("a", "u", "v", "x", "y")
            ]
            if not all(math.isfinite(value) and value >= 0.0 for value in numeric):
                raise Gate12C2ExperimentError("invalid component magnitude")
            c = arm["c"]
            if c is not None and (not math.isfinite(float(c)) or abs(float(c)) > 1.0):
                raise Gate12C2ExperimentError("invalid component alignment")
            if c is not None:
                squared_rhs = (
                    float(arm["x"]) ** 2
                    + float(arm["y"]) ** 2
                    - 2.0 * float(arm["x"]) * float(arm["y"]) * float(c)
                )
                if abs(float(arm["a"]) ** 2 - squared_rhs) > float(
                    spec["numerical_tolerance"]
                ) * max(1.0, float(arm["a"]) ** 2, abs(squared_rhs)):
                    raise Gate12C2ExperimentError("component residual identity fails")
            if arm["eligible"] is not True or arm["numerical_pass"] is not True:
                pass
            for spectrum_name in (
                "product_singular_values_left",
                "product_singular_values_right",
            ):
                spectrum = arm[spectrum_name]
                if (
                    not isinstance(spectrum, list)
                    or not spectrum
                    or any(
                        not math.isfinite(float(value)) or float(value) < 0.0
                        for value in spectrum
                    )
                ):
                    raise Gate12C2ExperimentError("invalid product spectrum")

        observed = arms[0]
        observed_gauge = arms[1]
        null = arms[2]
        observed_to_null = math.log(float(observed["a"]) + epsilon) - math.log(
            float(null["a"]) + epsilon
        )
        if not math.isclose(
            float(row["observed_to_null_log_defect"]),
            observed_to_null,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise Gate12C2ExperimentError("observed/null log defect differs")

        block_id = f"o{key[0]:03d}:r{key[1]:03d}"
        if regime == "S2":
            assert isinstance(stressor, dict)
            stress_to_null = math.log(float(stressor["a"]) + epsilon) - math.log(
                float(null["a"]) + epsilon
            )
            if not math.isclose(
                float(row["stressor_to_null_log_defect"]),
                stress_to_null,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise Gate12C2ExperimentError("stressor/null log defect differs")
            channel = bool(
                float(stressor["x"]) > float(null["x"]) + zero_tolerance
                or float(stressor["y"]) > float(null["y"]) + zero_tolerance
                or (
                    stressor["c"] is not None
                    and null["c"] is not None
                    and float(stressor["c"]) < float(null["c"]) - zero_tolerance
                )
            )
            if row["inflation_channel_moved"] is not channel:
                raise Gate12C2ExperimentError("inflation channel flag differs")
            values[q][block_id].append(stress_to_null)
            channels[q][block_id].append(channel)
        else:
            if row["stressor_to_null_log_defect"] is not None:
                raise Gate12C2ExperimentError("unexpected stressor log defect")
            values[q][block_id].append(observed_to_null)

        gauge_defect, gauge_component = _gauge_errors(
            ResidualDiagnostics(
                q=q,
                eligibility="eligible" if observed["eligible"] else "persisted",
                **observed,
            ),
            ResidualDiagnostics(
                q=q,
                eligibility=(
                    "eligible" if observed_gauge["eligible"] else "persisted"
                ),
                **observed_gauge,
            ),
        )
        if not math.isclose(
            gauge_defect,
            float(row["gauge_defect_error"]),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ) or not math.isclose(
            gauge_component,
            float(row["gauge_component_max_error"]),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise Gate12C2ExperimentError("gauge component evidence differs")

        scientific_arms = [observed, null]
        if regime == "S2":
            assert isinstance(stressor, dict)
            scientific_arms.append(stressor)
        numerical_failures[q] += int(
            any(arm["numerical_pass"] is not True for arm in scientific_arms)
        )
        ineligible[q] += int(
            any(arm["eligible"] is not True for arm in scientific_arms)
        )
        max_gauge_defect = max(max_gauge_defect, gauge_defect)
        max_gauge_component = max(
            max_gauge_component, gauge_component
        )

    recomputed = [
        _endpoint_summary(
            q=q,
            values_by_block=values[q],
            numerical_failures=numerical_failures[q],
            ineligible=ineligible[q],
            realizability_error=float(shard["n1_realizability_max_error"]),
            zero_tolerance=zero_tolerance,
            negative=regime in {"S0", "S1"},
            channel_by_block=channels[q] if regime == "S2" else None,
        )
        for q in sorted(q_values)
    ]
    if canonical_json_bytes(recomputed) != canonical_json_bytes(shard["endpoints"]):
        raise Gate12C2ExperimentError("component aggregate differs from endpoints")
    if not math.isclose(
        max_gauge_defect,
        float(shard["gauge_defect_max_error"]),
        rel_tol=1e-12,
        abs_tol=1e-15,
    ) or not math.isclose(
        max_gauge_component,
        float(shard["gauge_component_max_error"]),
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise Gate12C2ExperimentError("gauge maximum differs")


def _holm(endpoints: list[dict[str, Any]]) -> None:
    ordered = sorted(
        range(len(endpoints)),
        key=lambda index: (
            float(endpoints[index]["directional_sign_p"]),
            endpoints[index]["case_id"],
            int(endpoints[index]["q"]),
        ),
    )
    running = 0.0
    count = len(endpoints)
    for position, index in enumerate(ordered):
        adjusted = min(
            1.0,
            float(endpoints[index]["directional_sign_p"]) * (count - position),
        )
        running = max(running, adjusted)
        endpoints[index]["holm_adjusted_p"] = float(running)


def summarize_shards(
    spec: Mapping[str, Any],
    spec_sha256: str,
    shards: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected = set(expected_shard_ids(spec))
    if {str(shard["shard_id"]) for shard in shards} != expected:
        raise Gate12C2ExperimentError("aggregate shard set is incomplete")
    by_regime: dict[str, list[dict[str, Any]]] = {regime: [] for regime in REGIMES}
    controls_pass = True
    for shard in sorted(shards, key=lambda row: str(row["shard_id"])):
        controls_pass = bool(
            controls_pass
            and shard["observed_arm_unchanged"] is True
            and float(shard["n1_realizability_max_error"]) <= 1e-10
            and (
                shard["regime"] != "S2"
                or float(shard["stressor_edge_spectrum_max_error"]) <= 1e-10
            )
        )
        for row in shard["endpoints"]:
            endpoint = copy.deepcopy(row)
            endpoint.update(
                {
                    "case_id": shard["case"]["case_id"],
                    "model": shard["case"]["model"],
                    "family": shard["case"]["family"],
                }
            )
            by_regime[str(shard["regime"])].append(endpoint)

    acceptance = spec["smoke_acceptance"]
    regime_results: dict[str, Any] = {}
    for regime in REGIMES:
        endpoints = sorted(
            by_regime[regime], key=lambda row: (row["case_id"], row["q"])
        )
        if len(endpoints) != 24:
            raise Gate12C2ExperimentError(f"{regime} does not have 24 endpoints")
        _holm(endpoints)
        if regime in {"S0", "S1"}:
            for endpoint in endpoints:
                endpoint["holm_directional_support"] = bool(
                    endpoint["coverage_complete"]
                    and endpoint["median_effect"]
                    < -float(acceptance["zero_tolerance"])
                    and endpoint["holm_adjusted_p"]
                    < float(acceptance["holm_alpha"])
                )
        else:
            for endpoint in endpoints:
                endpoint["injected_inflation_support"] = bool(
                    endpoint["coverage_complete"]
                    and endpoint["median_effect"]
                    > float(acceptance["s2_min_log_inflation"])
                    and endpoint["channel_fraction"]
                    >= float(acceptance["s2_min_channel_fraction"])
                )
        regime_results[regime] = {
            "endpoint_count": 24,
            "coverage_complete": all(row["coverage_complete"] for row in endpoints),
            "endpoints": endpoints,
        }

    s0_supported = sum(
        row["holm_directional_support"] for row in regime_results["S0"]["endpoints"]
    )
    s1_directional_fraction = float(
        np.mean(
            [
                row["coverage_complete"]
                and row["median_effect"] < -float(acceptance["zero_tolerance"])
                for row in regime_results["S1"]["endpoints"]
            ]
        )
    )
    s2_inflation_fraction = float(
        np.mean(
            [
                row["injected_inflation_support"]
                for row in regime_results["S2"]["endpoints"]
            ]
        )
    )
    criteria = {
        "controls_and_identities": bool(
            controls_pass
            and all(
                regime_results[regime]["coverage_complete"] for regime in REGIMES
            )
        ),
        "s0_no_false_support": bool(
            s0_supported <= int(acceptance["s0_max_supported_endpoints"])
        ),
        "s1_known_direction": bool(
            s1_directional_fraction
            >= float(acceptance["s1_min_directional_endpoint_fraction"])
        ),
        "s2_injected_inflation": bool(
            s2_inflation_fraction
            >= float(acceptance["s2_min_inflation_endpoint_fraction"])
        ),
    }
    smoke_pass = all(criteria.values())
    return {
        "schema_version": RESULT_SCHEMA,
        "study_id": spec["study_id"],
        "study_sha256": spec_sha256,
        "epistemic_status": "synthetic_development_smoke_only",
        "decision": "SMOKE_PASS" if smoke_pass else "REPAIR_ONCE",
        "smoke_pass": smoke_pass,
        "locked_ready": False,
        "scientific_claim_authorized": False,
        "legacy_payload_accessed": False,
        "shard_count": len(shards),
        "endpoint_count_per_regime": 24,
        "criteria": criteria,
        "diagnostic_summary": {
            "s0_holm_supported_endpoint_count": int(s0_supported),
            "s1_directional_endpoint_fraction": s1_directional_fraction,
            "s2_inflation_endpoint_fraction": s2_inflation_fraction,
        },
        "regimes": regime_results,
    }
