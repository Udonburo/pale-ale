#!/usr/bin/env python3
"""Read-only statistical adequacy audit of the consumed Gate12C-2 N1 suite.

This script never writes inside the locked result directory.  It independently
reconstructs bounded-draw endpoint effects from the persisted component rows and
diagnoses three distinct questions:

1. Does finite null-draw Monte Carlo error explain the 9-to-15-draw shift?
2. Does N1 retain edge marginals while changing product/cross-edge geometry?
3. What changes when the independent synthetic dataset is the analysis grain?

The historical locked decision is an input fact and is never recomputed as a
replacement promotion decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


CONFIGS = ("S0", "S1_LOW", "S1_PRIMARY", "S1_HIGH", "S2")
DIAGNOSTIC_CONFIGS = {"S0", "S1_PRIMARY", "S2"}
FIELDS = (
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
)
COMPONENTS = ("a", "u", "v", "x", "y", "c", "p_left", "p_right")
LOG_COMPONENTS = {"a", "u", "v", "x", "y", "p_left", "p_right"}
PALETTE = {
    "S0": "#315A7D",
    "S1_LOW": "#C79531",
    "S1_PRIMARY": "#D06B32",
    "S1_HIGH": "#9B506C",
    "S2": "#697A3A",
}
QUANTILES = np.linspace(0.1, 0.9, 9)


class AuditError(RuntimeError):
    """Raised when the retained locked evidence is incomplete or inconsistent."""


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    encoded = "\x1f".join(str(value) for value in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:16], "big")


def arm(values: Sequence[Any]) -> dict[str, Any]:
    if len(values) != len(FIELDS):
        raise AuditError("diagnostic arm width differs")
    return dict(zip(FIELDS, values))


def safe_log_ratio(left: Any, right: Any, epsilon: float) -> float:
    if left is None or right is None:
        return math.nan
    return math.log(float(left) + epsilon) - math.log(float(right) + epsilon)


def component_contrast(
    numerator: Mapping[str, Any],
    denominator: Mapping[str, Any],
    component: str,
    epsilon: float,
) -> float:
    if component in LOG_COMPONENTS:
        return safe_log_ratio(numerator[component], denominator[component], epsilon)
    left, right = numerator[component], denominator[component]
    if left is None or right is None:
        return math.nan
    return float(left) - float(right)


def median_of_medians(values: np.ndarray, draws: Sequence[int]) -> float:
    selected = values[:, np.asarray(draws, dtype=int)]
    with np.errstate(all="ignore"):
        return float(np.nanmedian(np.nanmedian(selected, axis=1)))


def sign_tail_probability(values: Sequence[float], *, negative: bool, zero: float) -> float:
    informative = [float(value) for value in values if abs(float(value)) > zero]
    if not informative:
        return 1.0
    hits = sum(value < 0.0 if negative else value > 0.0 for value in informative)
    return sum(
        math.comb(len(informative), count)
        for count in range(hits, len(informative) + 1)
    ) / (2 ** len(informative))


def block_bootstrap_se(values: np.ndarray, draw_count: int, seed: int, repeats: int) -> float:
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, draw_count, size=(repeats, draw_count))
    sampled = values[:, indices]
    with np.errstate(all="ignore"):
        estimates = np.nanmedian(np.nanmedian(sampled, axis=2), axis=0)
    return float(np.nanstd(estimates, ddof=1))


def bimodality_bic_delta(values: np.ndarray) -> float:
    """Descriptive one-vs-two contiguous Gaussian split score for n=15.

    Positive values favor two clusters. This is a screening statistic, not a
    formal multimodality test.
    """

    ordered = np.sort(np.asarray(values, dtype=float))
    ordered = ordered[np.isfinite(ordered)]
    count = len(ordered)
    if count < 8:
        return math.nan
    floor = np.finfo(float).tiny
    one_rss = float(np.sum((ordered - np.mean(ordered)) ** 2))
    bic_one = count * math.log(max(one_rss / count, floor)) + 2 * math.log(count)
    best_two = math.inf
    for split in range(3, count - 2):
        left, right = ordered[:split], ordered[split:]
        left_rss = float(np.sum((left - np.mean(left)) ** 2))
        right_rss = float(np.sum((right - np.mean(right)) ** 2))
        pooled = max((left_rss + right_rss) / count, floor)
        bic_two = count * math.log(pooled) + 5 * math.log(count)
        best_two = min(best_two, bic_two)
    return float(bic_one - best_two)


def distribution_diagnostics(draw_statistics: np.ndarray) -> dict[str, float]:
    values = np.asarray(draw_statistics, dtype=float)
    median = float(np.median(values))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    iqr = float(q75 - q25)
    mad = float(np.median(np.abs(values - median)))
    robust_scale = max(1.4826 * mad, np.finfo(float).eps)
    return {
        "draw_median": median,
        "draw_iqr": iqr,
        "draw_mad": mad,
        "draw_skew": float(stats.skew(values, bias=False)),
        "draw_excess_kurtosis": float(stats.kurtosis(values, fisher=True, bias=False)),
        "draw_max_robust_z": float(np.max(np.abs(values - median)) / robust_scale),
        "bimodality_bic_delta": bimodality_bic_delta(values),
    }


def normal_reference_diagnostics(
    *, sample_count: int = 50_000, bic_sample_count: int = 10_000
) -> dict[str, float | int]:
    """Calibrate n=15 descriptive screens against a fixed Gaussian reference."""

    generator = np.random.default_rng(stable_seed("audit-normal-reference", 15))
    samples = generator.normal(size=(sample_count, 15))
    skew = np.abs(stats.skew(samples, axis=1, bias=False))
    kurtosis = stats.kurtosis(samples, axis=1, fisher=True, bias=False)
    medians = np.median(samples, axis=1)
    mad = np.median(np.abs(samples - medians[:, None]), axis=1)
    robust_z = np.max(np.abs(samples - medians[:, None]), axis=1) / np.maximum(
        1.4826 * mad, np.finfo(float).eps
    )
    bic = np.asarray(
        [bimodality_bic_delta(row) for row in samples[:bic_sample_count]],
        dtype=float,
    )
    return {
        "distribution": "standard_normal",
        "draw_count": 15,
        "sample_count": sample_count,
        "bic_sample_count": bic_sample_count,
        "absolute_skew_p95": float(np.quantile(skew, 0.95)),
        "excess_kurtosis_p95": float(np.quantile(kurtosis, 0.95)),
        "max_robust_z_p95": float(np.quantile(robust_z, 0.95)),
        "bimodality_bic_delta_p95": float(np.quantile(bic, 0.95)),
        "bimodality_bic_delta_gt6_rate": float(np.mean(bic > 6.0)),
    }


def ks_statistic(left: Sequence[float], right: Sequence[float]) -> float:
    return float(stats.ks_2samp(left, right, method="auto").statistic)


def quantile_max_difference(left: Sequence[float], right: Sequence[float]) -> float:
    return float(
        np.max(
            np.abs(
                np.quantile(np.asarray(left, dtype=float), QUANTILES)
                - np.quantile(np.asarray(right, dtype=float), QUANTILES)
            )
        )
    )


def correlation_difference(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    left_corr = np.corrcoef(left, rowvar=False)
    right_corr = np.corrcoef(right, rowvar=False)
    difference = np.nan_to_num(left_corr - right_corr, nan=0.0)
    mask = ~np.eye(difference.shape[0], dtype=bool)
    return float(np.max(np.abs(difference[mask]))), float(np.linalg.norm(difference, ord="fro"))


def verify_source(result_root: Path) -> dict[str, Any]:
    manifest = load_json(result_root / "manifest.json")
    state = load_json(result_root / "state.json")
    analysis = load_json(result_root / "analysis.json")
    if state.get("state") != "COMPLETE" or state.get("completed_shard_count") != 2880:
        raise AuditError("locked source is not complete")
    if analysis.get("decision") != "RETIRE_OR_DEMOTE" or analysis.get("locked_pass") is not False:
        raise AuditError("historical locked decision differs")
    files = manifest.get("files")
    if not isinstance(files, dict) or len(files) != 2882:
        raise AuditError("manifest file surface differs")
    for relative, expected in files.items():
        path = (result_root / relative).resolve()
        try:
            path.relative_to(result_root.resolve())
        except ValueError as exc:
            raise AuditError(f"manifest path escapes source root: {relative}") from exc
        if not path.is_file() or sha256_file(path) != expected:
            raise AuditError(f"manifest hash differs: {relative}")
    return {
        "source_root": str(result_root.resolve()),
        "study_sha256": manifest["study_sha256"],
        "analysis_sha256": files["analysis.json"],
        "manifest_sha256": sha256_file(result_root / "manifest.json"),
        "manifest_file_count": len(files),
        "locked_decision": analysis["decision"],
        "locked_stability": analysis["stability"],
    }


def _derangement(size: int, generator: np.random.Generator) -> np.ndarray:
    identity = np.arange(size)
    for _ in range(128):
        candidate = generator.permutation(size)
        if np.all(candidate != identity):
            return candidate
    shift = int(generator.integers(1, size))
    return np.roll(identity, shift)


def donor_metrics(
    *,
    seed_namespace: str,
    case_id: str,
    regime: str,
    dataset_index: int,
    cohort_size: int,
    draw_count: int,
) -> dict[str, float]:
    assignments = np.empty((draw_count, 3, cohort_size), dtype=int)
    for draw in range(draw_count):
        for role in range(3):
            generator = np.random.default_rng(
                stable_seed(
                    seed_namespace,
                    case_id,
                    regime,
                    dataset_index,
                    "N1",
                    draw,
                    role,
                )
            )
            assignments[draw, role] = _derangement(cohort_size, generator)

    recipient_tuples = np.transpose(assignments, (0, 2, 1))
    unique_counts = np.apply_along_axis(lambda row: len(set(row.tolist())), 2, recipient_tuples)
    pair_collision = float(np.mean(unique_counts < 3))
    triple_collision = float(np.mean(unique_counts == 1))
    repeated_tuple_rates: list[float] = []
    exposure_cvs: list[float] = []
    exposure_ranges: list[float] = []
    for recipient in range(cohort_size):
        tuples = [tuple(row) for row in recipient_tuples[:, recipient, :]]
        repeated_tuple_rates.append(1.0 - len(set(tuples)) / draw_count)
        eligible = [donor for donor in range(cohort_size) if donor != recipient]
        for role in range(3):
            counts = np.array(
                [np.sum(assignments[:, role, recipient] == donor) for donor in eligible],
                dtype=float,
            )
            exposure_cvs.append(float(np.std(counts) / np.mean(counts)))
            exposure_ranges.append(float(np.max(counts) - np.min(counts)))
    return {
        "pair_collision_rate": pair_collision,
        "triple_collision_rate": triple_collision,
        "mean_repeated_tuple_rate": float(np.mean(repeated_tuple_rates)),
        "mean_exposure_cv": float(np.mean(exposure_cvs)),
        "max_exposure_range": float(np.max(exposure_ranges)),
    }


def parse_shard(
    shard: Mapping[str, Any],
    *,
    epsilon: float,
    bootstrap_repeats: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, list[np.ndarray]],
]:
    config = str(shard["config_id"])
    dataset = int(shard["dataset_index"])
    case_id = str(shard["case"]["case_id"])
    observed = {
        (int(row[0]), int(row[1])): arm(row[2]) for row in shard["observed_rows"]
    }
    null = {
        (int(row[0]), int(row[1]), int(row[2])): arm(row[3])
        for row in shard["null_rows"]
    }
    stress = {
        (int(row[0]), int(row[1]), int(row[2])): (arm(row[3]), bool(row[5]))
        for row in shard["stressor_rows"]
    }
    cohort_size = 12
    draw_count = 15
    curve_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []

    for q in (1, 2):
        matrices: dict[str, np.ndarray] = {
            component: np.full((cohort_size, draw_count), np.nan, dtype=float)
            for component in COMPONENTS
        }
        channels = np.zeros((cohort_size, draw_count), dtype=float)
        for graph in range(cohort_size):
            for draw in range(draw_count):
                baseline = null[(graph, draw, q)]
                if config == "S2":
                    numerator, channel = stress[(graph, draw, q)]
                    channels[graph, draw] = float(channel)
                else:
                    numerator = observed[(graph, q)]
                for component in COMPONENTS:
                    matrices[component][graph, draw] = component_contrast(
                        numerator, baseline, component, epsilon
                    )

        a_matrix = matrices["a"]
        reference = median_of_medians(a_matrix, range(draw_count))
        for k in range(1, draw_count + 1):
            effect = median_of_medians(a_matrix, range(k))
            block_values = np.nanmedian(a_matrix[:, :k], axis=1)
            channel_fraction = (
                float(np.mean(np.mean(channels[:, :k], axis=1)))
                if config == "S2"
                else math.nan
            )
            curve_rows.append(
                {
                    "config": config,
                    "dataset_index": dataset,
                    "case_id": case_id,
                    "q": q,
                    "k": k,
                    "effect": effect,
                    "reference_effect_k15": reference,
                    "signed_error_to_k15": effect - reference,
                    "absolute_error_to_k15": abs(effect - reference),
                    "directional_sign_p": sign_tail_probability(
                        block_values,
                        negative=config != "S2",
                        zero=1e-6,
                    ),
                    "channel_fraction": channel_fraction,
                }
            )

        effect9 = median_of_medians(a_matrix, range(9))
        effect15 = reference
        suffix6 = median_of_medians(a_matrix, range(9, 15))
        odd = median_of_medians(a_matrix, range(0, 15, 2))
        even = median_of_medians(a_matrix, range(1, 15, 2))
        draw_statistics = np.nanmedian(a_matrix, axis=0)
        endpoint = {
            "config": config,
            "dataset_index": dataset,
            "case_id": case_id,
            "q": q,
            "effect_k9": effect9,
            "effect_k15": effect15,
            "signed_shift_9_to_15": effect15 - effect9,
            "absolute_shift_9_to_15": abs(effect15 - effect9),
            "effect_suffix6": suffix6,
            "absolute_prefix9_suffix6": abs(effect9 - suffix6),
            "absolute_odd_even": abs(odd - even),
            "mcse_k9": block_bootstrap_se(
                a_matrix[:, :9],
                9,
                stable_seed("audit-bootstrap", config, dataset, case_id, q, 9),
                bootstrap_repeats,
            ),
            "mcse_k15": block_bootstrap_se(
                a_matrix,
                15,
                stable_seed("audit-bootstrap", config, dataset, case_id, q, 15),
                bootstrap_repeats,
            ),
        }
        endpoint["shift_over_mcse_k15"] = endpoint["absolute_shift_9_to_15"] / max(
            endpoint["mcse_k15"], np.finfo(float).eps
        )
        endpoint.update(distribution_diagnostics(draw_statistics))
        endpoint_rows.append(endpoint)

        for component, matrix in matrices.items():
            component_rows.append(
                {
                    "config": config,
                    "dataset_index": dataset,
                    "case_id": case_id,
                    "q": q,
                    "component": component,
                    "effect_k9": median_of_medians(matrix, range(9)),
                    "effect_k15": median_of_medians(matrix, range(15)),
                }
            )

    observed_q1 = [row for row in shard["observed_rows"] if int(row[1]) == 1]
    null_q1 = [row for row in shard["null_rows"] if int(row[2]) == 1]
    observed_edge = np.asarray([row[4] for row in observed_q1], dtype=float)
    null_edge = np.asarray([row[4] for row in null_q1], dtype=float)
    observed_product = np.asarray(
        [
            [*arm(row[2])["product_singular_values_left"], *arm(row[2])["product_singular_values_right"]]
            for row in observed_q1
        ],
        dtype=float,
    )
    null_product = np.asarray(
        [
            [*arm(row[3])["product_singular_values_left"], *arm(row[3])["product_singular_values_right"]]
            for row in null_q1
        ],
        dtype=float,
    )
    edge_qdiffs: list[float] = []
    edge_ks: list[float] = []
    for edge_index in range(3):
        for singular_index in range(observed_edge.shape[2]):
            left = observed_edge[:, edge_index, singular_index]
            right = null_edge[:, edge_index, singular_index]
            edge_qdiffs.append(quantile_max_difference(left, right))
            edge_ks.append(ks_statistic(left, right))
    product_qdiffs: list[float] = []
    product_ks: list[float] = []
    for index in range(observed_product.shape[1]):
        left = observed_product[:, index]
        right = null_product[:, index]
        product_qdiffs.append(quantile_max_difference(left, right))
        product_ks.append(ks_statistic(left, right))
    observed_cross = np.stack(
        [
            np.sum(observed_edge, axis=2),
            np.sqrt(np.sum(observed_edge**2, axis=2)),
            observed_edge[:, :, 0],
        ],
        axis=2,
    ).reshape(len(observed_edge), -1)
    null_cross = np.stack(
        [
            np.sum(null_edge, axis=2),
            np.sqrt(np.sum(null_edge**2, axis=2)),
            null_edge[:, :, 0],
        ],
        axis=2,
    ).reshape(len(null_edge), -1)
    corr_max, corr_fro = correlation_difference(observed_cross, null_cross)
    geometry_row = {
        "config": config,
        "dataset_index": dataset,
        "case_id": case_id,
        "edge_quantile_max_difference": max(edge_qdiffs),
        "edge_ks_max": max(edge_ks),
        "product_quantile_max_difference": max(product_qdiffs),
        "product_ks_max": max(product_ks),
        "cross_edge_correlation_max_difference": corr_max,
        "cross_edge_correlation_frobenius": corr_fro,
    }
    nuisance_values = {
        "observed_edge": [observed_edge],
        "null_edge": [null_edge],
        "observed_product": [observed_product],
        "null_product": [null_product],
    }
    return curve_rows, endpoint_rows, component_rows, geometry_row, nuisance_values


def holm_support(rows: pd.DataFrame, alpha: float = 0.05) -> pd.Series:
    ordered = rows.sort_values(["directional_sign_p", "case_id", "q"]).copy()
    adjusted: dict[int, float] = {}
    running = 0.0
    total = len(ordered)
    for position, (index, row) in enumerate(ordered.iterrows()):
        running = max(running, min(1.0, float(row["directional_sign_p"]) * (total - position)))
        adjusted[index] = running
    return pd.Series(
        {
            index: bool(rows.loc[index, "effect"] < -1e-6 and value < alpha)
            for index, value in adjusted.items()
        }
    )


def decision_convergence(curves: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    decisions: list[dict[str, Any]] = []
    for (config, dataset, k), rows in curves.groupby(["config", "dataset_index", "k"]):
        if config != "S2":
            support = holm_support(rows)
            supported = rows.copy()
            supported["supported"] = support.reindex(rows.index).fillna(False).astype(bool)
            case_hits = int(supported.groupby("case_id")["supported"].all().sum())
            decisions.append(
                {
                    "config": config,
                    "dataset_index": dataset,
                    "k": k,
                    "family_median_effect": float(rows["effect"].median()),
                    "any_endpoint_support": bool(supported["supported"].any()),
                    "supported_endpoint_count": int(supported["supported"].sum()),
                    "supported_case_count": case_hits,
                    "promotion_or_identification": bool(case_hits >= 10),
                }
            )
        else:
            supported = (rows["effect"] > 0.0) & (rows["channel_fraction"] >= 0.5)
            success = bool(rows["effect"].median() > 0.05 and supported.mean() >= 0.5)
            decisions.append(
                {
                    "config": config,
                    "dataset_index": dataset,
                    "k": k,
                    "family_median_effect": float(rows["effect"].median()),
                    "any_endpoint_support": math.nan,
                    "supported_endpoint_count": int(supported.sum()),
                    "supported_case_count": math.nan,
                    "promotion_or_identification": success,
                }
            )
    decision_frame = pd.DataFrame(decisions)
    summary_rows: list[dict[str, Any]] = []
    for (config, k), rows in decision_frame.groupby(["config", "k"]):
        summary_rows.append(
            {
                "config": config,
                "k": k,
                "dataset_count": len(rows),
                "promotion_or_identification_rate": float(rows["promotion_or_identification"].mean()),
                "any_endpoint_support_rate": (
                    float(rows["any_endpoint_support"].mean()) if config == "S0" else math.nan
                ),
                "family_median_effect": float(rows["family_median_effect"].median()),
            }
        )
    return decision_frame, pd.DataFrame(summary_rows)


def bootstrap_median_interval(values: Sequence[float], seed: int, repeats: int = 5000) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(array), size=(repeats, len(array)))
    estimates = np.median(array[indices], axis=1)
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def auc_probability(lower: Sequence[float], higher: Sequence[float]) -> float:
    left = np.asarray(lower, dtype=float)[:, None]
    right = np.asarray(higher, dtype=float)[None, :]
    return float(np.mean(left < right) + 0.5 * np.mean(left == right))


def summarize_dataset_statistics(decisions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    final = decisions.loc[decisions["k"] == 15].copy()
    summaries: list[dict[str, Any]] = []
    for config, rows in final.groupby("config"):
        lower, upper = bootstrap_median_interval(
            rows["family_median_effect"], stable_seed("dataset-median", config)
        )
        summaries.append(
            {
                "config": config,
                "dataset_count": len(rows),
                "median_family_effect": float(rows["family_median_effect"].median()),
                "median_effect_bootstrap_lower": lower,
                "median_effect_bootstrap_upper": upper,
                "success_rate": float(rows["promotion_or_identification"].mean()),
            }
        )
    values = {
        config: final.loc[final["config"] == config, "family_median_effect"].to_numpy()
        for config in CONFIGS
    }
    separation = {
        "probability_s1_primary_lower_than_s0": auc_probability(
            values["S1_PRIMARY"], values["S0"]
        ),
        "probability_s2_higher_than_s0": auc_probability(values["S0"], values["S2"]),
    }
    return pd.DataFrame(summaries), separation


def summarize_components(components: pd.DataFrame, endpoints: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    components = components.copy()
    components["signed_shift_9_to_15"] = components["effect_k15"] - components["effect_k9"]
    components["absolute_shift_9_to_15"] = components["signed_shift_9_to_15"].abs()
    grouped = components.groupby(["config", "component"])["effect_k15"]
    components["robust_scale"] = (
        grouped.transform(lambda value: value.quantile(0.75))
        - grouped.transform(lambda value: value.quantile(0.25))
    ).clip(lower=1e-12)
    components["normalized_absolute_shift"] = (
        components["absolute_shift_9_to_15"] / components["robust_scale"]
    )
    a_shift = endpoints.set_index(["config", "dataset_index", "case_id", "q"])[
        "absolute_shift_9_to_15"
    ]
    summaries: list[dict[str, Any]] = []
    for (config, component), rows in components.groupby(["config", "component"]):
        keys = pd.MultiIndex.from_frame(rows[["config", "dataset_index", "case_id", "q"]])
        target = a_shift.reindex(keys).to_numpy()
        correlation = stats.spearmanr(
            target,
            rows["absolute_shift_9_to_15"].to_numpy(),
            nan_policy="omit",
        ).statistic
        summaries.append(
            {
                "config": config,
                "component": component,
                "median_absolute_shift": float(rows["absolute_shift_9_to_15"].median()),
                "p95_absolute_shift": float(rows["absolute_shift_9_to_15"].quantile(0.95)),
                "max_absolute_shift": float(rows["absolute_shift_9_to_15"].max()),
                "p95_normalized_absolute_shift": float(rows["normalized_absolute_shift"].quantile(0.95)),
                "spearman_with_a_absolute_shift": float(correlation),
            }
        )

    wide = components.pivot_table(
        index=["config", "dataset_index", "case_id", "q"],
        columns="component",
        values="normalized_absolute_shift",
    ).reset_index()
    wide["tail_score"] = wide[["u", "v"]].max(axis=1)
    wide["propagation_score"] = wide[["x", "y", "p_left", "p_right"]].max(axis=1)
    wide["cancellation_score"] = wide[["c"]].max(axis=1)
    wide["dominant_channel"] = wide[
        ["tail_score", "propagation_score", "cancellation_score"]
    ].idxmax(axis=1).str.replace("_score", "", regex=False)
    wide = wide.merge(
        endpoints[
            ["config", "dataset_index", "case_id", "q", "absolute_shift_9_to_15"]
        ],
        on=["config", "dataset_index", "case_id", "q"],
        how="left",
        validate="one_to_one",
    )
    return pd.DataFrame(summaries), wide


def summarize_nuisance(
    nuisance_pool: Mapping[tuple[str, str, str, int, int, str], list[float]],
    geometry: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for config in CONFIGS:
        for surface in ("edge", "product"):
            differences: list[float] = []
            ks_values: list[float] = []
            keys = sorted(
                {
                    (case, first, second)
                    for (
                        candidate,
                        candidate_surface,
                        case,
                        first,
                        second,
                        _arm,
                    ) in nuisance_pool
                    if candidate == config and candidate_surface == surface
                }
            )
            for case, first, second in keys:
                observed = nuisance_pool[(config, surface, case, first, second, "observed")]
                null = nuisance_pool[(config, surface, case, first, second, "null")]
                differences.append(quantile_max_difference(observed, null))
                ks_values.append(ks_statistic(observed, null))
            rows.append(
                {
                    "config": config,
                    "surface": surface,
                    "group_count": len(differences),
                    "max_quantile_difference": max(differences),
                    "median_quantile_difference": float(np.median(differences)),
                    "max_ks": max(ks_values),
                    "median_ks": float(np.median(ks_values)),
                    "median_shard_cross_edge_corr_difference": (
                        float(
                            geometry.loc[
                                geometry["config"] == config,
                                "cross_edge_correlation_max_difference",
                            ].median()
                        )
                        if surface == "edge"
                        else math.nan
                    ),
                    "p95_shard_cross_edge_corr_difference": (
                        float(
                            geometry.loc[
                                geometry["config"] == config,
                                "cross_edge_correlation_max_difference",
                            ].quantile(0.95)
                        )
                        if surface == "edge"
                        else math.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_pooled_geometry(
    geometry_pool: Mapping[tuple[str, str, str, str], list[np.ndarray]],
    *,
    bootstrap_repeats: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare persisted spectral geometry after pooling independent datasets.

    Each bootstrap resamples the 48 paired synthetic datasets, retaining the
    observed cohort and all of its N1 draws together. The result remains a
    spectral-summary diagnostic rather than a test of unpersisted raw frames.
    """

    rows: list[dict[str, Any]] = []
    for config in CONFIGS:
        cases = sorted(
            {
                case
                for candidate, _surface, case, _arm in geometry_pool
                if candidate == config
            }
        )
        for surface in ("edge", "product"):
            for case in cases:
                observed_blocks = [
                    np.asarray(block, dtype=float).reshape(len(block), -1)
                    for block in geometry_pool[(config, surface, case, "observed")]
                ]
                null_blocks = [
                    np.asarray(block, dtype=float).reshape(len(block), -1)
                    for block in geometry_pool[(config, surface, case, "null")]
                ]
                if len(observed_blocks) != len(null_blocks) or len(observed_blocks) != 48:
                    raise AuditError("pooled geometry dataset pairing differs")
                observed = np.concatenate(observed_blocks, axis=0)
                null = np.concatenate(null_blocks, axis=0)
                quantile_differences = [
                    quantile_max_difference(observed[:, index], null[:, index])
                    for index in range(observed.shape[1])
                ]
                ks_values = [
                    ks_statistic(observed[:, index], null[:, index])
                    for index in range(observed.shape[1])
                ]
                corr_max, corr_fro = correlation_difference(observed, null)
                generator = np.random.default_rng(
                    stable_seed("pooled-geometry-bootstrap", config, surface, case)
                )
                bootstrap_corr_max: list[float] = []
                for _ in range(bootstrap_repeats):
                    indices = generator.integers(0, len(observed_blocks), size=len(observed_blocks))
                    sampled_observed = np.concatenate(
                        [observed_blocks[index] for index in indices], axis=0
                    )
                    sampled_null = np.concatenate(
                        [null_blocks[index] for index in indices], axis=0
                    )
                    value, _ = correlation_difference(sampled_observed, sampled_null)
                    bootstrap_corr_max.append(value)
                lower, upper = np.quantile(bootstrap_corr_max, [0.025, 0.975])
                rows.append(
                    {
                        "config": config,
                        "surface": surface,
                        "case_id": case,
                        "dataset_count": len(observed_blocks),
                        "observed_row_count": len(observed),
                        "null_row_count": len(null),
                        "max_quantile_difference": max(quantile_differences),
                        "median_quantile_difference": float(np.median(quantile_differences)),
                        "max_ks": max(ks_values),
                        "median_ks": float(np.median(ks_values)),
                        "cross_feature_correlation_max_difference": corr_max,
                        "cross_feature_correlation_frobenius": corr_fro,
                        "cross_feature_correlation_max_bootstrap_lower": float(lower),
                        "cross_feature_correlation_max_bootstrap_upper": float(upper),
                    }
                )
    by_case = pd.DataFrame(rows)
    summary = (
        by_case.groupby(["config", "surface"])
        .agg(
            case_count=("case_id", "size"),
            max_quantile_difference=("max_quantile_difference", "max"),
            median_quantile_difference=("median_quantile_difference", "median"),
            max_ks=("max_ks", "max"),
            median_ks=("median_ks", "median"),
            median_cross_feature_correlation_max_difference=(
                "cross_feature_correlation_max_difference",
                "median",
            ),
            max_cross_feature_correlation_max_difference=(
                "cross_feature_correlation_max_difference",
                "max",
            ),
            max_cross_feature_correlation_bootstrap_upper=(
                "cross_feature_correlation_max_bootstrap_upper",
                "max",
            ),
        )
        .reset_index()
    )
    return by_case, summary


def correlation_summary(
    endpoints: pd.DataFrame,
    geometry: pd.DataFrame,
    donors: pd.DataFrame,
) -> pd.DataFrame:
    merged = endpoints.merge(
        geometry,
        on=["config", "dataset_index", "case_id"],
        how="left",
        validate="many_to_one",
    ).merge(
        donors,
        on=["config", "dataset_index", "case_id"],
        how="left",
        validate="many_to_one",
    )
    predictors = (
        "edge_quantile_max_difference",
        "product_quantile_max_difference",
        "cross_edge_correlation_max_difference",
        "pair_collision_rate",
        "mean_exposure_cv",
        "max_exposure_range",
        "draw_excess_kurtosis",
        "draw_max_robust_z",
        "bimodality_bic_delta",
        "mcse_k15",
    )
    rows: list[dict[str, Any]] = []
    subsets = [
        *( (config, merged.loc[merged["config"] == config]) for config in CONFIGS ),
        (
            "DIAGNOSTIC",
            merged.loc[merged["config"].isin(DIAGNOSTIC_CONFIGS)],
        ),
        ("ALL", merged),
    ]
    for config, subset in subsets:
        for predictor in predictors:
            correlation = stats.spearmanr(
                subset["absolute_shift_9_to_15"],
                subset[predictor],
                nan_policy="omit",
            )
            rows.append(
                {
                    "config": config,
                    "predictor": predictor,
                    "spearman_rho": float(correlation.statistic),
                    "p_value_descriptive": float(correlation.pvalue),
                    "endpoint_count": int(subset[["absolute_shift_9_to_15", predictor]].dropna().shape[0]),
                }
            )
    return pd.DataFrame(rows)


def save_charts(
    output: Path,
    curves: pd.DataFrame,
    component_summary: pd.DataFrame,
    endpoints: pd.DataFrame,
    geometry: pd.DataFrame,
    decisions: pd.DataFrame,
) -> list[str]:
    figure_root = output / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "text.color": "#222222",
            "font.size": 10,
            "axes.grid": True,
            "grid.color": "#E4E7EB",
            "grid.linewidth": 0.7,
        }
    )
    paths: list[str] = []

    convergence = (
        curves.groupby(["config", "k"])["absolute_error_to_k15"]
        .agg(median="median", p95=lambda value: value.quantile(0.95))
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for config in CONFIGS:
        rows = convergence.loc[convergence["config"] == config]
        axes[0].plot(rows["k"], rows["median"], marker="o", ms=3, label=config, color=PALETTE[config])
        axes[1].plot(rows["k"], rows["p95"], marker="o", ms=3, label=config, color=PALETTE[config])
    axes[0].set(title="Median absolute error to k=15", xlabel="Null draws k", ylabel="Absolute log-effect error")
    axes[1].set(title="95th-percentile absolute error to k=15", xlabel="Null draws k", ylabel="Absolute log-effect error")
    axes[0].set_xticks(range(1, 16, 2))
    axes[1].set_xticks(range(1, 16, 2))
    axes[1].legend(frameon=False, ncol=2)
    path = figure_root / "convergence_to_k15.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path.relative_to(output).as_posix())

    heat = component_summary.pivot(index="component", columns="config", values="p95_normalized_absolute_shift").reindex(
        index=COMPONENTS, columns=CONFIGS
    )
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    image = ax.imshow(heat.to_numpy(), cmap="YlOrBr", aspect="auto")
    ax.set_xticks(range(len(CONFIGS)), CONFIGS, rotation=25, ha="right")
    ax.set_yticks(range(len(COMPONENTS)), COMPONENTS)
    ax.set_title("Component stability: p95 shift normalized by component IQR")
    for row in range(heat.shape[0]):
        for column in range(heat.shape[1]):
            ax.text(column, row, f"{heat.iloc[row, column]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Normalized absolute shift")
    path = figure_root / "component_stability_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path.relative_to(output).as_posix())

    merged = endpoints.merge(
        geometry,
        on=["config", "dataset_index", "case_id"],
        how="left",
        validate="many_to_one",
    )
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    for config in CONFIGS:
        rows = merged.loc[merged["config"] == config]
        ax.scatter(
            rows["product_quantile_max_difference"],
            rows["absolute_shift_9_to_15"],
            s=12,
            alpha=0.35,
            color=PALETTE[config],
            label=config,
        )
    ax.set(
        title="Endpoint instability versus product-spectrum mismatch",
        xlabel="Per-shard maximum product-spectrum quantile difference",
        ylabel="Absolute 9-to-15 effect shift",
    )
    ax.legend(frameon=False, ncol=2)
    path = figure_root / "geometry_vs_instability.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path.relative_to(output).as_posix())

    final = decisions.loc[decisions["k"] == 15]
    fig, ax = plt.subplots(figsize=(9.5, 5.0), constrained_layout=True)
    arrays = [final.loc[final["config"] == config, "family_median_effect"] for config in CONFIGS]
    box = ax.boxplot(arrays, tick_labels=CONFIGS, patch_artist=True, showfliers=True)
    for patch, config in zip(box["boxes"], CONFIGS):
        patch.set_facecolor(PALETTE[config])
        patch.set_alpha(0.7)
    ax.axhline(0.0, color="#333333", linewidth=1.0)
    ax.set(
        title="Dataset-level family median effects at k=15",
        xlabel="Synthetic configuration (48 independent datasets each)",
        ylabel="Family median log effect",
    )
    path = figure_root / "dataset_level_effects.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path.relative_to(output).as_posix())
    return paths


def audit(result_root: Path, output: Path, bootstrap_repeats: int) -> dict[str, Any]:
    if output.resolve() == result_root.resolve() or result_root.resolve() in output.resolve().parents:
        raise AuditError("audit output must not be inside the locked result root")
    source = verify_source(result_root)
    output.mkdir(parents=True, exist_ok=False)
    study = load_json(result_root / "study.json")
    epsilon = float(study["epsilon"])
    cases = [str(case["case_id"]) for case in study["cases"]]

    curve_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    donor_rows: list[dict[str, Any]] = []
    nuisance_pool: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    geometry_pool: dict[tuple[str, str, str, str], list[np.ndarray]] = defaultdict(list)

    for config in CONFIGS:
        regime = "S1" if config.startswith("S1_") else config
        for dataset in range(int(study["dataset_count"])):
            for case_id in cases:
                path = result_root / "shards" / config / case_id / f"d{dataset:04d}.json"
                shard = load_json(path)
                curves, endpoints, components, geometry, nuisance = parse_shard(
                    shard,
                    epsilon=epsilon,
                    bootstrap_repeats=bootstrap_repeats,
                )
                curve_rows.extend(curves)
                endpoint_rows.extend(endpoints)
                component_rows.extend(components)
                geometry_rows.append(geometry)
                donor_rows.append(
                    {
                        "config": config,
                        "dataset_index": dataset,
                        "case_id": case_id,
                        **donor_metrics(
                            seed_namespace=study["seed_namespace"],
                            case_id=case_id,
                            regime=regime,
                            dataset_index=dataset,
                            cohort_size=int(study["cohort_size"]),
                            draw_count=int(study["inner_draws"]),
                        ),
                    }
                )

                observed_edge = nuisance["observed_edge"][0]
                null_edge = nuisance["null_edge"][0]
                geometry_pool[(config, "edge", case_id, "observed")].append(observed_edge)
                geometry_pool[(config, "edge", case_id, "null")].append(null_edge)
                for edge_index in range(3):
                    for singular_index in range(observed_edge.shape[2]):
                        nuisance_pool[(config, "edge", case_id, edge_index, singular_index, "observed")].extend(
                            observed_edge[:, edge_index, singular_index].tolist()
                        )
                        nuisance_pool[(config, "edge", case_id, edge_index, singular_index, "null")].extend(
                            null_edge[:, edge_index, singular_index].tolist()
                        )
                observed_product = nuisance["observed_product"][0]
                null_product = nuisance["null_product"][0]
                geometry_pool[(config, "product", case_id, "observed")].append(observed_product)
                geometry_pool[(config, "product", case_id, "null")].append(null_product)
                rank = observed_product.shape[1] // 2
                for side_index in range(2):
                    for singular_index in range(rank):
                        column = side_index * rank + singular_index
                        nuisance_pool[(config, "product", case_id, side_index, singular_index, "observed")].extend(
                            observed_product[:, column].tolist()
                        )
                        nuisance_pool[(config, "product", case_id, side_index, singular_index, "null")].extend(
                            null_product[:, column].tolist()
                        )

    curves = pd.DataFrame(curve_rows)
    endpoints = pd.DataFrame(endpoint_rows)
    components = pd.DataFrame(component_rows)
    geometry = pd.DataFrame(geometry_rows)
    donors = pd.DataFrame(donor_rows)
    decisions, decision_summary = decision_convergence(curves)
    dataset_summary, separation = summarize_dataset_statistics(decisions)
    component_summary, attribution = summarize_components(components, endpoints)
    nuisance_summary = summarize_nuisance(nuisance_pool, geometry)
    pooled_geometry, pooled_geometry_summary = summarize_pooled_geometry(
        geometry_pool,
        bootstrap_repeats=bootstrap_repeats,
    )
    correlations = correlation_summary(endpoints, geometry, donors)
    normal_reference = normal_reference_diagnostics()

    convergence_summary = (
        curves.groupby(["config", "k"])["absolute_error_to_k15"]
        .agg(
            median_absolute_error="median",
            p90_absolute_error=lambda value: value.quantile(0.90),
            p95_absolute_error=lambda value: value.quantile(0.95),
        )
        .reset_index()
    )
    endpoint_summary = (
        endpoints.groupby("config")
        .agg(
            endpoint_count=("absolute_shift_9_to_15", "size"),
            median_shift=("absolute_shift_9_to_15", "median"),
            p95_shift=("absolute_shift_9_to_15", lambda value: value.quantile(0.95)),
            p95_prefix_suffix=("absolute_prefix9_suffix6", lambda value: value.quantile(0.95)),
            p95_odd_even=("absolute_odd_even", lambda value: value.quantile(0.95)),
            median_mcse_k15=("mcse_k15", "median"),
            p95_mcse_k15=("mcse_k15", lambda value: value.quantile(0.95)),
            p95_shift_over_mcse=("shift_over_mcse_k15", lambda value: value.quantile(0.95)),
            fraction_shift_within_1_96_mcse=(
                "shift_over_mcse_k15",
                lambda value: float(np.mean(value <= 1.96)),
            ),
            fraction_1_96_mcse_within_locked_tolerance=(
                "mcse_k15",
                lambda value: float(np.mean(1.96 * value <= 0.30)),
            ),
            fraction_absolute_skew_above_normal_p95=(
                "draw_skew",
                lambda value: float(
                    np.mean(np.abs(value) > normal_reference["absolute_skew_p95"])
                ),
            ),
            fraction_kurtosis_above_normal_p95=(
                "draw_excess_kurtosis",
                lambda value: float(
                    np.mean(value > normal_reference["excess_kurtosis_p95"])
                ),
            ),
            fraction_robust_z_above_normal_p95=(
                "draw_max_robust_z",
                lambda value: float(
                    np.mean(value > normal_reference["max_robust_z_p95"])
                ),
            ),
            fraction_bimodality_delta_above_normal_p95=(
                "bimodality_bic_delta",
                lambda value: float(
                    np.mean(value > normal_reference["bimodality_bic_delta_p95"])
                ),
            ),
            p95_max_robust_z=("draw_max_robust_z", lambda value: value.quantile(0.95)),
        )
        .reset_index()
    )
    endpoint_case_q_summary = (
        endpoints.groupby(["config", "case_id", "q"])
        .agg(
            endpoint_count=("absolute_shift_9_to_15", "size"),
            median_shift=("absolute_shift_9_to_15", "median"),
            p90_shift=("absolute_shift_9_to_15", lambda value: value.quantile(0.90)),
            p95_shift=("absolute_shift_9_to_15", lambda value: value.quantile(0.95)),
            max_shift=("absolute_shift_9_to_15", "max"),
            median_mcse_k15=("mcse_k15", "median"),
            p95_shift_over_mcse=("shift_over_mcse_k15", lambda value: value.quantile(0.95)),
        )
        .reset_index()
    )
    donor_summary = (
        donors.groupby("config")
        .agg(
            shard_count=("case_id", "size"),
            pair_collision_rate=("pair_collision_rate", "mean"),
            triple_collision_rate=("triple_collision_rate", "mean"),
            mean_repeated_tuple_rate=("mean_repeated_tuple_rate", "mean"),
            mean_exposure_cv=("mean_exposure_cv", "mean"),
            p95_exposure_cv=("mean_exposure_cv", lambda value: value.quantile(0.95)),
            max_exposure_range=("max_exposure_range", "max"),
        )
        .reset_index()
    )
    eligible_donors = int(study["cohort_size"]) - 1
    draw_count = int(study["inner_draws"])
    tuple_count = eligible_donors**3
    donor_summary["independent_pair_collision_expectation"] = 1.0 - (
        eligible_donors * (eligible_donors - 1) * (eligible_donors - 2)
    ) / (eligible_donors**3)
    donor_summary["independent_triple_collision_expectation"] = 1.0 / (
        eligible_donors**2
    )
    donor_summary["independent_repeated_tuple_rate_expectation"] = 1.0 - (
        tuple_count * (1.0 - (1.0 - 1.0 / tuple_count) ** draw_count) / draw_count
    )
    donor_summary["multinomial_exposure_cv_approximation"] = math.sqrt(
        (eligible_donors - 1) / draw_count
    )

    tables = {
        "convergence_curves.csv": curves,
        "convergence_summary.csv": convergence_summary,
        "endpoint_audit.csv": endpoints,
        "endpoint_summary.csv": endpoint_summary,
        "endpoint_case_q_summary.csv": endpoint_case_q_summary,
        "component_endpoints.csv": components,
        "component_summary.csv": component_summary,
        "attribution_channels.csv": attribution,
        "decision_convergence.csv": decisions,
        "decision_convergence_summary.csv": decision_summary,
        "dataset_level_summary.csv": dataset_summary,
        "geometry_by_shard.csv": geometry,
        "nuisance_summary.csv": nuisance_summary,
        "pooled_geometry_by_case.csv": pooled_geometry,
        "pooled_geometry_summary.csv": pooled_geometry_summary,
        "donor_by_shard.csv": donors,
        "donor_summary.csv": donor_summary,
        "driver_correlations.csv": correlations,
    }
    for name, frame in tables.items():
        frame.to_csv(output / name, index=False)
    (output / "normal_reference.json").write_text(
        json.dumps(normal_reference, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    figures = save_charts(output, curves, component_summary, endpoints, geometry, decisions)
    summary = {
        "schema_version": "gate12c2_v2_statistical_adequacy_audit_v0.1",
        "epistemic_status": "post_locked_development_audit",
        "historical_locked_decision_unchanged": True,
        "source": source,
        "audit_parameters": {
            "draw_counts": list(range(1, 16)),
            "nested_comparison": [9, 15],
            "disjoint_comparison": ["prefix_1_9", "suffix_10_15"],
            "block_bootstrap_repeats": bootstrap_repeats,
            "bootstrap_unit": "draw_index_shared_across_all_12_graphs",
            "inference_unit_for_descriptive_reconstruction": "independent_synthetic_dataset",
        },
        "row_counts": {name: int(len(frame)) for name, frame in tables.items()},
        "dataset_level_separation": separation,
        "normal_reference_diagnostics": normal_reference,
        "locked_failure_reproduction": {
            "diagnostic_configs": sorted(DIAGNOSTIC_CONFIGS),
            "p95_absolute_shift_9_to_15": float(
                endpoints.loc[endpoints["config"].isin(DIAGNOSTIC_CONFIGS), "absolute_shift_9_to_15"].quantile(0.95)
            ),
        },
        "figures": figures,
        "limitations": [
            "k=15 remains a finite reference rather than a known infinite-draw target.",
            "Block bootstrap treats draw indices as exchangeable conditional on the fixed cohort.",
            "Multimodality BIC is a descriptive screen with only 15 draw statistics.",
            "Cross-edge geometry uses persisted spectral summaries; raw frames and matrices were not persisted.",
            "All v2 findings are development evidence and cannot revise the consumed locked decision.",
        ],
    }
    (output / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="retained locked result root")
    parser.add_argument("--output", type=Path, required=True, help="new audit output directory")
    parser.add_argument("--bootstrap-repeats", type=int, default=256)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.bootstrap_repeats < 64:
        raise AuditError("bootstrap repeats must be at least 64")
    summary = audit(args.input.resolve(), args.output.resolve(), args.bootstrap_repeats)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
