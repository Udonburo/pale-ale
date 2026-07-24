#!/usr/bin/env python3
"""Deterministic development-only cheap baseline for process triage.

The implementation matches the contract-v0.2 freeze candidate:

* first-actionable-defect rows are the only positive row targets;
* numeric cheap features are standardized on each training surface;
* domain, artifact type, and opaque source slot are one-hot encoded;
* an unweighted FP64 L2 logistic model is fit by damped Newton iteration;
* C is selected from the frozen grid using deterministic grouped OOF scores;
* the selected model is refit on all declared development clusters.

This module has no structural signal and no locked-evaluation entry point.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from process_triage_evaluator import (
    BASELINE_CV_FOLD_COUNT,
    BASELINE_CV_ID,
    BASELINE_CV_SEED_ID,
    FREEZE_CANDIDATE_SCHEMA_VERSION,
    BaselineFreezeCandidate,
    CheapStepFeatures,
    ProcessTriageDevelopmentError,
    TriageTrajectory,
    evaluate_global_review_budget,
    position_only_scores,
)


BASELINE_MODEL_SCHEMA_VERSION = "pale_ale_l2_baseline_model_v0.1"
BASELINE_DEVELOPMENT_SCHEMA_VERSION = (
    "pale_ale_l2_baseline_development_v0.1"
)
REFERENCE_DTYPE = "float64"
NUMERIC_FEATURE_NAMES = (
    "normalized_position",
    "eligible_trajectory_length",
    "prior_exact_retry_count",
    "prior_same_tool_count",
    "preceding_tool_error",
    "prior_tool_error_count",
    "lexical_drift_from_previous",
    "content_character_count",
)
CATEGORICAL_FEATURE_NAMES = (
    "domain",
    "artifact_type",
    "source_slot",
)
MAXIMUM_ITERATIONS = 100
GRADIENT_INFINITY_TOLERANCE = 1.0e-8
MINIMUM_LINE_SEARCH_STEP = 2.0**-20
ARMIJO_CONSTANT = 1.0e-4


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _category_value(feature: CheapStepFeatures, name: str) -> str:
    if name == "domain":
        return feature.domain
    if name == "artifact_type":
        return feature.artifact_type
    if name == "source_slot":
        return str(int(feature.source_slot))
    raise ProcessTriageDevelopmentError(
        f"unknown categorical feature: {name!r}"
    )


def _effective_cluster_id(
    original_group_id: str,
    aliases: Mapping[str, str],
) -> str:
    return str(aliases.get(original_group_id, original_group_id))


def _validate_feature_rows(
    features: Sequence[CheapStepFeatures],
) -> tuple[CheapStepFeatures, ...]:
    ordered = tuple(sorted(features, key=lambda row: row.row_id))
    if not ordered:
        raise ProcessTriageDevelopmentError(
            "baseline fitting requires at least one cheap-feature row"
        )
    row_ids = [row.row_id for row in ordered]
    if len(row_ids) != len(set(row_ids)):
        raise ProcessTriageDevelopmentError(
            "baseline feature rows contain duplicate row IDs"
        )
    for row in ordered:
        for name in NUMERIC_FEATURE_NAMES:
            value = float(getattr(row, name))
            if not math.isfinite(value):
                raise ProcessTriageDevelopmentError(
                    f"non-finite {name} for row {row.row_id}"
                )
    return ordered


@dataclass(frozen=True)
class BaselineEncoder:
    numeric_means: tuple[float, ...]
    numeric_scales: tuple[float, ...]
    category_levels: tuple[tuple[str, ...], ...]
    column_names: tuple[str, ...]
    schema_version: str = BASELINE_MODEL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if len(self.numeric_means) != len(NUMERIC_FEATURE_NAMES):
            raise ProcessTriageDevelopmentError(
                "numeric mean count does not match frozen feature schema"
            )
        if len(self.numeric_scales) != len(NUMERIC_FEATURE_NAMES):
            raise ProcessTriageDevelopmentError(
                "numeric scale count does not match frozen feature schema"
            )
        if len(self.category_levels) != len(CATEGORICAL_FEATURE_NAMES):
            raise ProcessTriageDevelopmentError(
                "category level count does not match frozen feature schema"
            )
        if any(
            not math.isfinite(scale) or scale <= 0.0
            for scale in self.numeric_scales
        ):
            raise ProcessTriageDevelopmentError(
                "numeric scales must be finite and positive"
            )

    @classmethod
    def fit(
        cls,
        features: Sequence[CheapStepFeatures],
    ) -> "BaselineEncoder":
        ordered = _validate_feature_rows(features)
        numeric = np.asarray(
            [
                [float(getattr(row, name)) for name in NUMERIC_FEATURE_NAMES]
                for row in ordered
            ],
            dtype=np.float64,
        )
        means = np.mean(numeric, axis=0, dtype=np.float64)
        scales = np.std(numeric, axis=0, dtype=np.float64)
        scales = np.where(scales > 0.0, scales, 1.0)
        levels = tuple(
            tuple(
                sorted(
                    {
                        _category_value(row, name)
                        for row in ordered
                    }
                )
            )
            for name in CATEGORICAL_FEATURE_NAMES
        )
        columns = ["intercept"]
        columns.extend(f"numeric:{name}" for name in NUMERIC_FEATURE_NAMES)
        for name, values in zip(CATEGORICAL_FEATURE_NAMES, levels):
            columns.extend(
                f"categorical:{name}={value}" for value in values
            )
        return cls(
            numeric_means=tuple(float(value) for value in means),
            numeric_scales=tuple(float(value) for value in scales),
            category_levels=levels,
            column_names=tuple(columns),
        )

    def transform(
        self,
        features: Sequence[CheapStepFeatures],
    ) -> tuple[tuple[CheapStepFeatures, ...], np.ndarray]:
        ordered = _validate_feature_rows(features)
        row_count = len(ordered)
        matrix = np.zeros(
            (row_count, len(self.column_names)),
            dtype=np.float64,
        )
        matrix[:, 0] = 1.0
        numeric = np.asarray(
            [
                [float(getattr(row, name)) for name in NUMERIC_FEATURE_NAMES]
                for row in ordered
            ],
            dtype=np.float64,
        )
        means = np.asarray(self.numeric_means, dtype=np.float64)
        scales = np.asarray(self.numeric_scales, dtype=np.float64)
        matrix[:, 1 : 1 + len(NUMERIC_FEATURE_NAMES)] = (
            numeric - means
        ) / scales
        column_index = 1 + len(NUMERIC_FEATURE_NAMES)
        for name, levels in zip(
            CATEGORICAL_FEATURE_NAMES,
            self.category_levels,
        ):
            level_to_offset = {
                level: offset for offset, level in enumerate(levels)
            }
            for row_index, row in enumerate(ordered):
                value = _category_value(row, name)
                offset = level_to_offset.get(value)
                if offset is not None:
                    matrix[row_index, column_index + offset] = 1.0
            column_index += len(levels)
        if matrix.shape[1] != column_index:
            raise ProcessTriageDevelopmentError(
                "baseline design column assembly mismatch"
            )
        return ordered, matrix

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "numeric_feature_names": list(NUMERIC_FEATURE_NAMES),
            "numeric_means": list(self.numeric_means),
            "numeric_scales": list(self.numeric_scales),
            "zero_variance_numeric_scale": 1.0,
            "categorical_feature_names": list(CATEGORICAL_FEATURE_NAMES),
            "category_levels": {
                name: list(levels)
                for name, levels in zip(
                    CATEGORICAL_FEATURE_NAMES,
                    self.category_levels,
                )
            },
            "unknown_category_policy": "all_zero_indicators",
            "column_names": list(self.column_names),
        }


def _sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    nonnegative = values >= 0.0
    result[nonnegative] = 1.0 / (
        1.0 + np.exp(-values[nonnegative])
    )
    exponentiated = np.exp(values[~nonnegative])
    result[~nonnegative] = exponentiated / (1.0 + exponentiated)
    return result


def _objective(
    matrix: np.ndarray,
    targets: np.ndarray,
    coefficients: np.ndarray,
    *,
    regularization_c: float,
) -> float:
    logits = matrix @ coefficients
    loss = np.sum(
        np.logaddexp(0.0, logits) - targets * logits,
        dtype=np.float64,
    )
    penalty = float(
        np.dot(coefficients[1:], coefficients[1:])
        / (2.0 * regularization_c)
    )
    return float(loss + penalty)


def _fit_coefficients(
    matrix: np.ndarray,
    targets: np.ndarray,
    *,
    regularization_c: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if (
        matrix.ndim != 2
        or targets.ndim != 1
        or matrix.shape[0] != targets.shape[0]
    ):
        raise ProcessTriageDevelopmentError(
            "baseline design and target shapes are incompatible"
        )
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ProcessTriageDevelopmentError(
            "baseline design matrix must be nonempty"
        )
    if not math.isfinite(regularization_c) or regularization_c <= 0.0:
        raise ProcessTriageDevelopmentError(
            "regularization C must be finite and positive"
        )
    unique_targets = set(float(value) for value in targets)
    if not unique_targets <= {0.0, 1.0} or len(unique_targets) != 2:
        raise ProcessTriageDevelopmentError(
            "baseline training requires both binary target classes"
        )

    coefficients = np.zeros(matrix.shape[1], dtype=np.float64)
    current_objective = _objective(
        matrix,
        targets,
        coefficients,
        regularization_c=regularization_c,
    )
    last_step = None
    for iteration in range(MAXIMUM_ITERATIONS + 1):
        logits = matrix @ coefficients
        probabilities = _sigmoid(logits)
        gradient = matrix.T @ (probabilities - targets)
        gradient[1:] += coefficients[1:] / regularization_c
        gradient_infinity = float(np.max(np.abs(gradient)))
        if gradient_infinity <= GRADIENT_INFINITY_TOLERANCE:
            return coefficients, {
                "status": "converged",
                "iterations": iteration,
                "objective": current_objective,
                "gradient_infinity_norm": gradient_infinity,
                "last_line_search_step": last_step,
                "maximum_iterations": MAXIMUM_ITERATIONS,
                "gradient_infinity_tolerance": (
                    GRADIENT_INFINITY_TOLERANCE
                ),
                "minimum_line_search_step": MINIMUM_LINE_SEARCH_STEP,
                "reference_dtype": REFERENCE_DTYPE,
            }
        if iteration == MAXIMUM_ITERATIONS:
            break

        variances = probabilities * (1.0 - probabilities)
        hessian = matrix.T @ (matrix * variances[:, None])
        regularization = np.zeros(matrix.shape[1], dtype=np.float64)
        regularization[1:] = 1.0 / regularization_c
        hessian += np.diag(regularization)
        hessian += np.eye(matrix.shape[1], dtype=np.float64) * 1.0e-12
        try:
            direction = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as exc:
            raise ProcessTriageDevelopmentError(
                "baseline Newton system is singular"
            ) from exc
        directional_decrease = float(np.dot(gradient, direction))
        if not math.isfinite(directional_decrease) or directional_decrease <= 0:
            raise ProcessTriageDevelopmentError(
                "baseline Newton direction is not a descent direction"
            )

        step = 1.0
        accepted = False
        while step >= MINIMUM_LINE_SEARCH_STEP:
            candidate = coefficients - step * direction
            candidate_objective = _objective(
                matrix,
                targets,
                candidate,
                regularization_c=regularization_c,
            )
            if candidate_objective <= (
                current_objective
                - ARMIJO_CONSTANT * step * directional_decrease
            ):
                coefficients = candidate
                current_objective = candidate_objective
                last_step = step
                accepted = True
                break
            step *= 0.5
        if not accepted:
            raise ProcessTriageDevelopmentError(
                "baseline Newton line search did not find an acceptable step"
            )
    raise ProcessTriageDevelopmentError(
        "baseline Newton solver did not converge within the frozen limit"
    )


@dataclass(frozen=True)
class FittedBaselineModel:
    encoder: BaselineEncoder
    coefficients: tuple[float, ...]
    regularization_c: float
    fit_diagnostics: Mapping[str, Any]
    schema_version: str = BASELINE_MODEL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if len(self.coefficients) != len(self.encoder.column_names):
            raise ProcessTriageDevelopmentError(
                "coefficient count does not match encoded columns"
            )

    def predict_scores(
        self,
        features: Sequence[CheapStepFeatures],
    ) -> dict[str, float]:
        ordered, matrix = self.encoder.transform(features)
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        probabilities = _sigmoid(matrix @ coefficients)
        return {
            row.row_id: float(probability)
            for row, probability in zip(ordered, probabilities)
        }

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "epistemic_status": "development_baseline_only",
            "model_class": "L2_regularized_logistic_regression",
            "objective": (
                "sum_binary_log_loss_plus_nonintercept_L2_over_2C"
            ),
            "class_weight": "none",
            "intercept_penalized": False,
            "regularization_c": float(self.regularization_c),
            "encoder": self.encoder.as_dict(),
            "coefficients": list(self.coefficients),
            "fit_diagnostics": dict(self.fit_diagnostics),
            "structural_signal_included": False,
            "locked_evaluation_authorized": False,
        }
        payload["model_payload_sha256"] = hashlib.sha256(
            _canonical_json(payload).encode("utf-8")
        ).hexdigest()
        return payload


def _targets_by_row(
    trajectories: Sequence[TriageTrajectory],
) -> dict[str, int]:
    targets: dict[str, int] = {}
    for trajectory in trajectories:
        first = trajectory.first_actionable_row_id
        for step in trajectory.steps:
            if step.row_id in targets:
                raise ProcessTriageDevelopmentError(
                    f"duplicate target row ID: {step.row_id}"
                )
            targets[step.row_id] = int(step.row_id == first)
    return targets


def fit_baseline_model(
    features: Sequence[CheapStepFeatures],
    *,
    targets_by_row: Mapping[str, int],
    regularization_c: float,
) -> FittedBaselineModel:
    encoder = BaselineEncoder.fit(features)
    ordered, matrix = encoder.transform(features)
    row_ids = {row.row_id for row in ordered}
    if row_ids != set(targets_by_row):
        raise ProcessTriageDevelopmentError(
            "baseline target coverage does not match feature rows"
        )
    targets = np.asarray(
        [float(targets_by_row[row.row_id]) for row in ordered],
        dtype=np.float64,
    )
    coefficients, diagnostics = _fit_coefficients(
        matrix,
        targets,
        regularization_c=regularization_c,
    )
    return FittedBaselineModel(
        encoder=encoder,
        coefficients=tuple(float(value) for value in coefficients),
        regularization_c=float(regularization_c),
        fit_diagnostics=diagnostics,
    )


def _binary_log_loss(
    scores: Mapping[str, float],
    targets: Mapping[str, int],
) -> float:
    if set(scores) != set(targets):
        raise ProcessTriageDevelopmentError(
            "log-loss score and target coverage mismatch"
        )
    losses = []
    for row_id, target in targets.items():
        probability = min(max(float(scores[row_id]), 1.0e-15), 1.0 - 1.0e-15)
        losses.append(
            -float(target) * math.log(probability)
            - (1.0 - float(target)) * math.log(1.0 - probability)
        )
    return float(sum(losses) / len(losses))


def fit_development_baseline(
    trajectories: Sequence[TriageTrajectory],
    features: Sequence[CheapStepFeatures],
    *,
    cv_manifest: Mapping[str, Any],
    group_aliases: Mapping[str, str] | None = None,
    regularization_grid: Sequence[float] = (0.1, 1.0, 10.0),
) -> dict[str, Any]:
    """Select and refit the frozen cheap baseline on development only."""

    if (
        cv_manifest.get("assignment_id") != BASELINE_CV_ID
        or cv_manifest.get("seed_id") != BASELINE_CV_SEED_ID
        or int(cv_manifest.get("fold_count", -1)) != BASELINE_CV_FOLD_COUNT
    ):
        raise ProcessTriageDevelopmentError(
            "baseline fitting requires the frozen four-fold CV manifest"
        )
    if cv_manifest.get("label_or_outcome_fields_used") != []:
        raise ProcessTriageDevelopmentError(
            "baseline CV manifest must be label and outcome blind"
        )
    aliases = dict(group_aliases or {})
    ordered_features = _validate_feature_rows(features)
    feature_ids = {row.row_id for row in ordered_features}
    target_by_row = _targets_by_row(trajectories)
    if feature_ids != set(target_by_row):
        raise ProcessTriageDevelopmentError(
            "development feature and trajectory row coverage mismatch"
        )
    declared_clusters = {
        str(cluster_id)
        for fold in cv_manifest["folds"]
        for cluster_id in fold["validation_cluster_ids"]
    }
    trajectory_clusters = {
        _effective_cluster_id(trajectory.group_id, aliases)
        for trajectory in trajectories
    }
    if trajectory_clusters != declared_clusters:
        raise ProcessTriageDevelopmentError(
            "baseline trajectories do not equal the declared development "
            "cluster surface"
        )
    feature_cluster = {
        row.row_id: _effective_cluster_id(row.group_id, aliases)
        for row in ordered_features
    }
    trajectory_by_cluster: dict[str, list[TriageTrajectory]] = {}
    for trajectory in trajectories:
        cluster_id = _effective_cluster_id(trajectory.group_id, aliases)
        trajectory_by_cluster.setdefault(cluster_id, []).append(trajectory)

    grid = tuple(float(value) for value in regularization_grid)
    if grid != tuple(BaselineFreezeCandidate().regularization_grid):
        raise ProcessTriageDevelopmentError(
            "regularization grid differs from the frozen candidate"
        )
    candidate_rows: list[dict[str, Any]] = []
    candidate_oof_scores: dict[float, dict[str, float]] = {}
    for regularization_c in grid:
        oof_scores: dict[str, float] = {}
        fold_receipts: list[dict[str, Any]] = []
        for fold in sorted(
            cv_manifest["folds"],
            key=lambda row: int(row["fold_index"]),
        ):
            training_clusters = {
                str(value) for value in fold["training_cluster_ids"]
            }
            validation_clusters = {
                str(value) for value in fold["validation_cluster_ids"]
            }
            training_features = tuple(
                row
                for row in ordered_features
                if feature_cluster[row.row_id] in training_clusters
            )
            validation_features = tuple(
                row
                for row in ordered_features
                if feature_cluster[row.row_id] in validation_clusters
            )
            training_ids = {row.row_id for row in training_features}
            training_targets = {
                row_id: target_by_row[row_id] for row_id in training_ids
            }
            model = fit_baseline_model(
                training_features,
                targets_by_row=training_targets,
                regularization_c=regularization_c,
            )
            fold_scores = model.predict_scores(validation_features)
            overlap = set(oof_scores) & set(fold_scores)
            if overlap:
                raise ProcessTriageDevelopmentError(
                    "OOF rows appear in more than one validation fold"
                )
            oof_scores.update(fold_scores)
            fold_receipts.append(
                {
                    "fold_index": int(fold["fold_index"]),
                    "training_cluster_count": len(training_clusters),
                    "validation_cluster_count": len(validation_clusters),
                    "training_row_count": len(training_features),
                    "validation_row_count": len(validation_features),
                    "positive_training_row_count": sum(
                        training_targets.values()
                    ),
                    "fit_diagnostics": dict(model.fit_diagnostics),
                }
            )
        if set(oof_scores) != feature_ids:
            raise ProcessTriageDevelopmentError(
                "OOF score surface does not cover every development row once"
            )
        evaluation = evaluate_global_review_budget(
            trajectories,
            scores=oof_scores,
            budget_fraction=0.10,
        )
        candidate_rows.append(
            {
                "regularization_c": regularization_c,
                "first_actionable_defect_recall": evaluation[
                    "first_actionable_defect_recall"
                ],
                "clean_row_allocation": evaluation[
                    "clean_row_allocation"
                ],
                "clean_trajectory_alert_rate": evaluation[
                    "clean_trajectory_alert_rate"
                ],
                "row_log_loss": _binary_log_loss(
                    oof_scores,
                    target_by_row,
                ),
                "selected_row_count": evaluation["selected_row_count"],
                "fold_receipts": fold_receipts,
            }
        )
        candidate_oof_scores[regularization_c] = oof_scores

    def selection_key(row: Mapping[str, Any]) -> tuple[float, float, float]:
        recall = row["first_actionable_defect_recall"]
        clean = row["clean_row_allocation"]
        if recall is None or clean is None:
            raise ProcessTriageDevelopmentError(
                "baseline selection metrics are undefined"
            )
        return (
            -float(recall),
            float(clean),
            float(row["regularization_c"]),
        )

    selected_row = min(candidate_rows, key=selection_key)
    selected_c = float(selected_row["regularization_c"])
    final_model = fit_baseline_model(
        ordered_features,
        targets_by_row=target_by_row,
        regularization_c=selected_c,
    )
    position_evaluation = evaluate_global_review_budget(
        trajectories,
        scores=position_only_scores(ordered_features),
        budget_fraction=0.10,
    )
    selected_oof = candidate_oof_scores[selected_c]
    selected_oof_evaluation = evaluate_global_review_budget(
        trajectories,
        scores=selected_oof,
        budget_fraction=0.10,
    )
    model_payload = final_model.as_dict()
    oof_score_payload_sha256 = hashlib.sha256(
        _canonical_json(
            {
                row_id: selected_oof[row_id]
                for row_id in sorted(selected_oof)
            }
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": BASELINE_DEVELOPMENT_SCHEMA_VERSION,
        "epistemic_status": "development_baseline_only",
        "evaluation_surface": "sealed_development_clusters_only",
        "development_cluster_count": len(declared_clusters),
        "trajectory_count": len(trajectories),
        "row_count": len(ordered_features),
        "positive_target_row_count": sum(target_by_row.values()),
        "cv_assignment_id": BASELINE_CV_ID,
        "cv_seed_id": BASELINE_CV_SEED_ID,
        "cv_fold_count": BASELINE_CV_FOLD_COUNT,
        "selection_rule": {
            "primary": (
                "maximize_out_of_fold_first_actionable_defect_recall_at_"
                "global_10_percent_row_budget"
            ),
            "tie_breakers": [
                "lower_out_of_fold_clean_row_allocation",
                "smaller_regularization_C",
            ],
        },
        "candidate_rows": candidate_rows,
        "selected_regularization_c": selected_c,
        "selected_oof_evaluation": selected_oof_evaluation,
        "position_only_oof_surface_diagnostic": position_evaluation,
        "selected_oof_score_payload_sha256": oof_score_payload_sha256,
        "refit_model": model_payload,
        "structural_signal_opened": False,
        "locked_partition_scored": False,
        "locked_evaluation_authorized": False,
        "public_claim_authorized": False,
    }
