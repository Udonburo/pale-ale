#!/usr/bin/env python3
"""Outcome-association-blind power planning for process triage."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


POWER_SCHEMA_VERSION = "pale_ale_process_triage_power_v0.1"
POWER_SEED_ID = "pale-ale-agent-process-bench-power-v0.1"
BASELINE_RECALL_GRID = (0.20, 0.35, 0.50)
TRUE_RECALL_GAIN_GRID = (0.10, 0.15, 0.20)
CLUSTER_ICC_GRID = (0.00, 0.05, 0.15)
LOSS_PROBABILITY_GRID = (0.00, 0.05, 0.10)
POINT_GAIN_THRESHOLD = 0.10
NORMAL_QUANTILE_95 = 1.959963984540054


class ProcessTriagePowerError(ValueError):
    """Raised when a power-planning input violates the frozen boundary."""


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _seed_from_parts(*parts: object) -> int:
    digest = hashlib.sha256(
        _canonical_json([POWER_SEED_ID, *parts]).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:16], byteorder="big", signed=False)


@dataclass(frozen=True, order=True)
class PowerCluster:
    cluster_id: str
    domain: str
    trajectory_count: int

    def __post_init__(self) -> None:
        if not self.cluster_id:
            raise ProcessTriagePowerError("cluster ID must be non-empty")
        if not self.domain:
            raise ProcessTriagePowerError("domain must be non-empty")
        if self.trajectory_count < 1:
            raise ProcessTriagePowerError(
                "cluster trajectory count must be positive"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "domain": self.domain,
            "trajectory_count": self.trajectory_count,
        }


def _validate_surface(
    clusters: Sequence[PowerCluster],
    *,
    domain_positive_prevalence: Mapping[str, float],
) -> tuple[PowerCluster, ...]:
    ordered = tuple(sorted(clusters))
    if not ordered:
        raise ProcessTriagePowerError("power surface must not be empty")
    ids = [cluster.cluster_id for cluster in ordered]
    if len(ids) != len(set(ids)):
        raise ProcessTriagePowerError("power surface cluster IDs repeat")
    domains = {cluster.domain for cluster in ordered}
    if domains != set(domain_positive_prevalence):
        raise ProcessTriagePowerError(
            "prevalence domains do not match the power surface"
        )
    for domain, prevalence in domain_positive_prevalence.items():
        if not 0.0 < float(prevalence) < 1.0:
            raise ProcessTriagePowerError(
                f"invalid positive prevalence for {domain}"
            )
        if sum(cluster.domain == domain for cluster in ordered) < 2:
            raise ProcessTriagePowerError(
                f"domain needs at least two clusters: {domain}"
            )
    return ordered


def _beta_probabilities(
    rng: np.random.Generator,
    *,
    mean: float,
    intracluster_correlation: float,
    size: int,
) -> np.ndarray:
    if not 0.0 <= mean <= 1.0:
        raise ProcessTriagePowerError("Bernoulli mean is outside [0, 1]")
    if not 0.0 <= intracluster_correlation < 1.0:
        raise ProcessTriagePowerError("ICC is outside [0, 1)")
    if mean in (0.0, 1.0) or intracluster_correlation == 0.0:
        return np.full(size, mean, dtype=np.float64)
    concentration = (1.0 / intracluster_correlation) - 1.0
    return rng.beta(
        mean * concentration,
        (1.0 - mean) * concentration,
        size=size,
    )


def _wilson_interval(
    successes: int,
    trials: int,
) -> tuple[float, float]:
    if trials < 1:
        raise ProcessTriagePowerError(
            "Wilson interval requires at least one trial"
        )
    proportion = successes / trials
    z = NORMAL_QUANTILE_95
    denominator = 1.0 + (z * z / trials)
    center = (
        proportion + (z * z / (2.0 * trials))
    ) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _simulate_cluster_totals(
    clusters: Sequence[PowerCluster],
    *,
    domain_positive_prevalence: Mapping[str, float],
    baseline_recall: float,
    true_recall_gain: float,
    loss_probability: float,
    intracluster_correlation: float,
    simulation_count: int,
    seed_parts: Sequence[object],
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if not 0.0 < baseline_recall < 1.0:
        raise ProcessTriagePowerError(
            "baseline recall must be strictly between zero and one"
        )
    if true_recall_gain <= 0.0:
        raise ProcessTriagePowerError(
            "true recall gain must be strictly positive"
        )
    if not 0.0 <= loss_probability < 1.0:
        raise ProcessTriagePowerError(
            "loss probability is outside [0, 1)"
        )
    rescue_probability = (
        true_recall_gain + baseline_recall * loss_probability
    ) / (1.0 - baseline_recall)
    if not 0.0 <= rescue_probability <= 1.0:
        raise ProcessTriagePowerError(
            "scenario implies an invalid rescue probability"
        )
    rng = np.random.default_rng(_seed_from_parts(*seed_parts))
    cluster_count = len(clusters)
    positives = np.zeros(
        (cluster_count, simulation_count),
        dtype=np.int16,
    )
    differences = np.zeros_like(positives)
    for cluster_index, cluster in enumerate(clusters):
        prevalence = _beta_probabilities(
            rng,
            mean=float(domain_positive_prevalence[cluster.domain]),
            intracluster_correlation=intracluster_correlation,
            size=simulation_count,
        )
        positive_count = rng.binomial(
            cluster.trajectory_count,
            prevalence,
        )
        baseline_probability = _beta_probabilities(
            rng,
            mean=baseline_recall,
            intracluster_correlation=intracluster_correlation,
            size=simulation_count,
        )
        baseline_hits = rng.binomial(
            positive_count,
            baseline_probability,
        )
        rescue = _beta_probabilities(
            rng,
            mean=rescue_probability,
            intracluster_correlation=intracluster_correlation,
            size=simulation_count,
        )
        loss = _beta_probabilities(
            rng,
            mean=loss_probability,
            intracluster_correlation=intracluster_correlation,
            size=simulation_count,
        )
        rescued = rng.binomial(positive_count - baseline_hits, rescue)
        lost = rng.binomial(baseline_hits, loss)
        positives[cluster_index] = positive_count
        differences[cluster_index] = rescued - lost
    return positives, differences, {
        "loss_probability": loss_probability,
        "rescue_probability": rescue_probability,
    }


def _cluster_sandwich_lower_bound(
    clusters: Sequence[PowerCluster],
    positives: np.ndarray,
    differences: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    total_positives = positives.sum(axis=0, dtype=np.int64)
    total_difference = differences.sum(axis=0, dtype=np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        point = total_difference / total_positives
    variance_numerator = np.zeros(point.shape, dtype=np.float64)
    domains = sorted({cluster.domain for cluster in clusters})
    for domain in domains:
        indices = [
            index
            for index, cluster in enumerate(clusters)
            if cluster.domain == domain
        ]
        influence = (
            differences[indices].astype(np.float64)
            - point[None, :] * positives[indices]
        )
        influence -= influence.mean(axis=0, keepdims=True)
        variance_numerator += np.square(influence).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        standard_error = np.sqrt(variance_numerator) / total_positives
    lower = point - NORMAL_QUANTILE_95 * standard_error
    return point, lower


def _domain_stratified_bootstrap_weights(
    clusters: Sequence[PowerCluster],
    *,
    replicate_count: int,
    seed_parts: Sequence[object],
) -> np.ndarray:
    rng = np.random.default_rng(_seed_from_parts(*seed_parts))
    weights = np.zeros(
        (replicate_count, len(clusters)),
        dtype=np.int16,
    )
    for domain in sorted({cluster.domain for cluster in clusters}):
        indices = np.asarray(
            [
                index
                for index, cluster in enumerate(clusters)
                if cluster.domain == domain
            ],
            dtype=np.int64,
        )
        draws = rng.integers(
            0,
            len(indices),
            size=(replicate_count, len(indices)),
        )
        for local_index, global_index in enumerate(indices):
            weights[:, global_index] = np.sum(
                draws == local_index,
                axis=1,
            )
    return weights


def _percentile_bootstrap_lower_bound(
    weights: np.ndarray,
    positives: np.ndarray,
    differences: np.ndarray,
) -> np.ndarray:
    bootstrap_positives = (
        weights.astype(np.float64) @ positives.astype(np.float64)
    )
    bootstrap_differences = (
        weights.astype(np.float64) @ differences.astype(np.float64)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        bootstrap_delta = bootstrap_differences / bootstrap_positives
    return np.nanquantile(
        bootstrap_delta,
        0.025,
        axis=0,
        method="linear",
    )


def _power_summary(
    point: np.ndarray,
    lower: np.ndarray,
) -> dict[str, Any]:
    valid = np.isfinite(point) & np.isfinite(lower)
    valid_count = int(valid.sum())
    if valid_count < 1:
        raise ProcessTriagePowerError(
            "power simulation produced no valid outer experiment"
        )
    point_pass = point[valid] >= POINT_GAIN_THRESHOLD
    lower_pass = lower[valid] > 0.0
    joint_pass = point_pass & lower_pass
    success_count = int(joint_pass.sum())
    interval = _wilson_interval(success_count, valid_count)
    return {
        "valid_outer_simulations": valid_count,
        "invalid_outer_simulations": int(len(point) - valid_count),
        "mean_estimated_recall_gain": float(np.mean(point[valid])),
        "median_estimated_recall_gain": float(
            np.median(point[valid])
        ),
        "point_gain_threshold_pass_probability": float(
            np.mean(point_pass)
        ),
        "lower_bound_above_zero_probability": float(
            np.mean(lower_pass)
        ),
        "recall_rule_pass_probability": success_count / valid_count,
        "recall_rule_pass_wilson_95": {
            "lower": interval[0],
            "upper": interval[1],
        },
    }


def _surface_variants(
    clusters: Sequence[PowerCluster],
) -> dict[str, tuple[PowerCluster, ...]]:
    maximum = max(cluster.trajectory_count for cluster in clusters)
    largest = [
        cluster
        for cluster in clusters
        if cluster.trajectory_count == maximum
    ]
    surfaces = {"full_locked_layout": tuple(clusters)}
    for cluster in largest:
        surfaces[
            f"leave_largest_out:{cluster.cluster_id}"
        ] = tuple(
            candidate
            for candidate in clusters
            if candidate.cluster_id != cluster.cluster_id
        )
    return surfaces


def _design_sensitivity_summary(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_surface: dict[str, dict[str, Any]] = {}
    for surface_id in sorted(
        {str(row["surface_id"]) for row in reports}
    ):
        surface_rows = [
            row
            for row in reports
            if str(row["surface_id"]) == surface_id
        ]
        gain_rows = []
        for true_gain in TRUE_RECALL_GAIN_GRID:
            rows = [
                row
                for row in surface_rows
                if float(row["nominal_true_recall_gain"])
                == true_gain
            ]
            probabilities = [
                float(row["recall_rule_pass_probability"])
                for row in rows
            ]
            lower_bounds = [
                float(row["recall_rule_pass_wilson_95"]["lower"])
                for row in rows
            ]
            gain_rows.append(
                {
                    "nominal_true_recall_gain": true_gain,
                    "scenario_count": len(rows),
                    "minimum_pass_probability": min(probabilities),
                    "median_pass_probability": float(
                        np.median(probabilities)
                    ),
                    "maximum_pass_probability": max(probabilities),
                    "minimum_wilson_95_lower": min(lower_bounds),
                    "all_point_power_at_least_0_80": (
                        min(probabilities) >= 0.80
                    ),
                    "all_wilson_lower_at_least_0_75": (
                        min(lower_bounds) >= 0.75
                    ),
                }
            )
        adequate = [
            row["nominal_true_recall_gain"]
            for row in gain_rows
            if row["all_point_power_at_least_0_80"]
            and row["all_wilson_lower_at_least_0_75"]
        ]
        by_surface[surface_id] = {
            "gain_rows": gain_rows,
            "minimum_evaluated_gain_meeting_robust_planning_rule": (
                min(adequate) if adequate else None
            ),
        }
    return {
        "robust_planning_rule": (
            "all baseline-recall and ICC scenarios have point power "
            "at least 0.80 and Wilson lower bound at least 0.75"
        ),
        "surfaces": by_surface,
        "boundary_gain_interpretation": (
            "At a true gain exactly equal to the required observed "
            "+0.10 threshold, the point-estimate condition alone has "
            "an asymptotic pass probability near one half. Failure to "
            "reach 80% at that boundary is not by itself evidence that "
            "the cluster layout is too small."
        ),
    }


def run_power_simulation(
    clusters: Sequence[PowerCluster],
    *,
    domain_positive_prevalence: Mapping[str, float],
    simulation_count: int = 5_000,
    bootstrap_validation_simulations: int = 500,
    bootstrap_validation_replicates: int = 999,
) -> dict[str, Any]:
    """Run the frozen planning grid without feature-label associations."""
    ordered = _validate_surface(
        clusters,
        domain_positive_prevalence=domain_positive_prevalence,
    )
    if simulation_count < 1_000:
        raise ProcessTriagePowerError(
            "planning grid requires at least 1,000 outer simulations"
        )
    if not 100 <= bootstrap_validation_simulations <= simulation_count:
        raise ProcessTriagePowerError(
            "invalid bootstrap-validation outer count"
        )
    if bootstrap_validation_replicates < 199:
        raise ProcessTriagePowerError(
            "bootstrap validation requires at least 199 replicates"
        )

    reports = []
    bootstrap_validation = []
    surfaces = _surface_variants(ordered)
    for surface_id, surface in surfaces.items():
        weights = _domain_stratified_bootstrap_weights(
            surface,
            replicate_count=bootstrap_validation_replicates,
            seed_parts=("bootstrap_weights", surface_id),
        )
        for baseline_recall in BASELINE_RECALL_GRID:
            for true_gain in TRUE_RECALL_GAIN_GRID:
                for icc in CLUSTER_ICC_GRID:
                    for loss_probability in LOSS_PROBABILITY_GRID:
                        scenario_id = (
                            f"baseline={baseline_recall:.2f};"
                            f"gain={true_gain:.2f};icc={icc:.2f};"
                            f"loss={loss_probability:.2f}"
                        )
                        positives, differences, mechanism = (
                            _simulate_cluster_totals(
                                surface,
                                domain_positive_prevalence=(
                                    domain_positive_prevalence
                                ),
                                baseline_recall=baseline_recall,
                                true_recall_gain=true_gain,
                                loss_probability=loss_probability,
                                intracluster_correlation=icc,
                                simulation_count=simulation_count,
                                seed_parts=(
                                    "outer",
                                    surface_id,
                                    scenario_id,
                                ),
                            )
                        )
                        point, lower = _cluster_sandwich_lower_bound(
                            surface,
                            positives,
                            differences,
                        )
                        summary = _power_summary(point, lower)
                        reports.append(
                            {
                                "surface_id": surface_id,
                                "scenario_id": scenario_id,
                                "cluster_count": len(surface),
                                "trajectory_count": sum(
                                    cluster.trajectory_count
                                    for cluster in surface
                                ),
                                "baseline_recall": baseline_recall,
                                "nominal_true_recall_gain": true_gain,
                                "intracluster_correlation": icc,
                                **mechanism,
                                "planning_interval": (
                                    "domain_stratified_cluster_sandwich_"
                                    "normal_proxy"
                                ),
                                **summary,
                            }
                        )

                        is_validation_anchor = (
                            baseline_recall == 0.35
                            and icc in (0.00, 0.15)
                            and loss_probability == 0.10
                        )
                        if is_validation_anchor:
                            validation_count = (
                                bootstrap_validation_simulations
                            )
                            bootstrap_lower = (
                                _percentile_bootstrap_lower_bound(
                                    weights,
                                    positives[:, :validation_count],
                                    differences[:, :validation_count],
                                )
                            )
                            normal_validation = _power_summary(
                                point[:validation_count],
                                lower[:validation_count],
                            )
                            percentile_validation = _power_summary(
                                point[:validation_count],
                                bootstrap_lower,
                            )
                            bootstrap_validation.append(
                                {
                                    "surface_id": surface_id,
                                    "scenario_id": scenario_id,
                                    "outer_simulations": (
                                        validation_count
                                    ),
                                    "bootstrap_replicates": (
                                        bootstrap_validation_replicates
                                    ),
                                    "normal_proxy_pass_probability": (
                                        normal_validation[
                                            "recall_rule_pass_probability"
                                        ]
                                    ),
                                    "percentile_bootstrap_pass_probability": (
                                        percentile_validation[
                                            "recall_rule_pass_probability"
                                        ]
                                    ),
                                    "absolute_probability_difference": abs(
                                        normal_validation[
                                            "recall_rule_pass_probability"
                                        ]
                                        - percentile_validation[
                                            "recall_rule_pass_probability"
                                        ]
                                    ),
                                }
                            )

    maximum_validation_difference = max(
        row["absolute_probability_difference"]
        for row in bootstrap_validation
    )
    sensitivity_summary = _design_sensitivity_summary(reports)
    surface_payload = [
        cluster.as_dict() for cluster in ordered
    ]
    return {
        "schema_version": POWER_SCHEMA_VERSION,
        "epistemic_status": "design_power_simulation_only",
        "information_used": {
            "locked_cluster_ids_domains_and_sizes": True,
            "frozen_domain_label_marginals": True,
            "individual_locked_outcomes": False,
            "feature_label_associations": False,
            "development_baseline_performance": False,
            "structural_signal": False,
        },
        "seed_id": POWER_SEED_ID,
        "simulation_count_per_scenario": simulation_count,
        "bootstrap_validation": {
            "outer_simulations_per_anchor": (
                bootstrap_validation_simulations
            ),
            "replicates_per_anchor": (
                bootstrap_validation_replicates
            ),
            "anchor_reports": bootstrap_validation,
            "maximum_absolute_probability_difference": (
                maximum_validation_difference
            ),
            "planning_proxy_agreement_within_0_05": (
                maximum_validation_difference <= 0.05
            ),
        },
        "scenario_grid": {
            "baseline_recall": list(BASELINE_RECALL_GRID),
            "nominal_true_recall_gain": list(
                TRUE_RECALL_GAIN_GRID
            ),
            "intracluster_correlation": list(CLUSTER_ICC_GRID),
            "loss_probability": list(LOSS_PROBABILITY_GRID),
            "point_gain_threshold": POINT_GAIN_THRESHOLD,
        },
        "locked_layout": {
            "cluster_count": len(ordered),
            "trajectory_count": sum(
                cluster.trajectory_count for cluster in ordered
            ),
            "domain_cluster_count": {
                domain: sum(
                    cluster.domain == domain for cluster in ordered
                )
                for domain in sorted(domain_positive_prevalence)
            },
            "domain_trajectory_count": {
                domain: sum(
                    cluster.trajectory_count
                    for cluster in ordered
                    if cluster.domain == domain
                )
                for domain in sorted(domain_positive_prevalence)
            },
            "maximum_cluster_trajectory_count": max(
                cluster.trajectory_count for cluster in ordered
            ),
            "cluster_surface_sha256": hashlib.sha256(
                _canonical_json(surface_payload).encode("utf-8")
            ).hexdigest(),
        },
        "domain_positive_prevalence": {
            domain: float(domain_positive_prevalence[domain])
            for domain in sorted(domain_positive_prevalence)
        },
        "reports": reports,
        "design_sensitivity_summary": sensitivity_summary,
        "interpretation_boundary": {
            "recall_component_only": True,
            "clean_burden_guardrail_powered": False,
            "confirmatory_result": False,
            "locked_evaluation_authorized": False,
            "structural_signal_opened": False,
            "note": (
                "The point-estimate threshold makes pass probability near "
                "one half when the true gain is exactly +0.10, even as "
                "sampling error shrinks. Higher-gain scenarios are included "
                "to characterize design sensitivity, not to move the "
                "predeclared practical threshold."
            ),
        },
    }
