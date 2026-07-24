#!/usr/bin/env python3
"""Development-only confound and dependence diagnostics for the cheap baseline."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

import process_triage_baseline as baseline
from process_triage_evaluator import (
    BASELINE_CV_ID,
    BASELINE_CV_SEED_ID,
    BaselineFreezeCandidate,
    CheapStepFeatures,
    ProcessTriageDevelopmentError,
    TriageTrajectory,
    evaluate_global_review_budget,
)


DIAGNOSTIC_SCHEMA_VERSION = "pale_ale_baseline_diagnostics_v0.1"
CONFIGURATIONS = {
    "source_slot_only": {
        "numeric": (),
        "categorical": ("source_slot",),
    },
    "domain_source_slot": {
        "numeric": (),
        "categorical": ("domain", "source_slot"),
    },
    "position_length": {
        "numeric": (
            "normalized_position",
            "eligible_trajectory_length",
        ),
        "categorical": (),
    },
    "full_cheap": {
        "numeric": baseline.NUMERIC_FEATURE_NAMES,
        "categorical": baseline.CATEGORICAL_FEATURE_NAMES,
    },
}


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _effective_cluster(
    group_id: str,
    aliases: Mapping[str, str],
) -> str:
    return str(aliases.get(group_id, group_id))


def _helmert_contrast(level_count: int) -> np.ndarray:
    """Return a deterministic orthonormal basis for the centered level space."""
    if level_count < 1:
        raise ProcessTriageDevelopmentError(
            "categorical diagnostics require at least one fitted level"
        )
    if level_count == 1:
        return np.empty((1, 0), dtype=np.float64)
    contrast = np.zeros(
        (level_count, level_count - 1),
        dtype=np.float64,
    )
    for column in range(level_count - 1):
        denominator = np.sqrt(
            float((column + 1) * (column + 2))
        )
        contrast[: column + 1, column] = 1.0 / denominator
        contrast[column + 1, column] = (
            -float(column + 1) / denominator
        )
    return contrast


def _diagnostic_design(
    encoder: baseline.BaselineEncoder,
    matrix: np.ndarray,
    *,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build an identifiable reduced design without changing the sealed model."""
    numeric = set(numeric_features)
    categorical = set(categorical_features)
    if not numeric <= set(baseline.NUMERIC_FEATURE_NAMES):
        raise ProcessTriageDevelopmentError(
            "diagnostic requested an unknown numeric feature"
        )
    if not categorical <= set(baseline.CATEGORICAL_FEATURE_NAMES):
        raise ProcessTriageDevelopmentError(
            "diagnostic requested an unknown categorical feature"
        )

    blocks = [matrix[:, [0]]]
    parameter_names = ["intercept"]
    for name in numeric_features:
        column_name = f"numeric:{name}"
        try:
            index = encoder.column_names.index(column_name)
        except ValueError as exc:
            raise ProcessTriageDevelopmentError(
                f"diagnostic numeric column is unavailable: {name}"
            ) from exc
        blocks.append(matrix[:, [index]])
        parameter_names.append(column_name)

    categorical_levels: dict[str, list[str]] = {}
    for name in categorical_features:
        prefix = f"categorical:{name}="
        indices = [
            index
            for index, column_name in enumerate(encoder.column_names)
            if column_name.startswith(prefix)
        ]
        if not indices:
            raise ProcessTriageDevelopmentError(
                f"diagnostic categorical column is unavailable: {name}"
            )
        levels = [
            encoder.column_names[index].removeprefix(prefix)
            for index in indices
        ]
        categorical_levels[name] = levels
        contrast = _helmert_contrast(len(indices))
        if contrast.shape[1]:
            blocks.append(matrix[:, indices] @ contrast)
            parameter_names.extend(
                f"contrast:{name}:{column}"
                for column in range(contrast.shape[1])
            )

    design = np.concatenate(blocks, axis=1)
    if not np.all(np.isfinite(design)):
        raise ProcessTriageDevelopmentError(
            "diagnostic design contains non-finite values"
        )
    return design, {
        "parameterization": (
            "standardized_numeric_plus_orthonormal_centered_helmert"
        ),
        "parameter_names": parameter_names,
        "categorical_levels": categorical_levels,
        "design_column_count": int(design.shape[1]),
        "design_matrix_rank": int(np.linalg.matrix_rank(design)),
    }


def _fit_and_score(
    training_features: Sequence[CheapStepFeatures],
    validation_features: Sequence[CheapStepFeatures],
    *,
    targets_by_row: Mapping[str, int],
    regularization_c: float,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    reproduce_sealed_full_parameterization: bool = True,
) -> tuple[dict[str, float], dict[str, Any]]:
    if (
        reproduce_sealed_full_parameterization
        and tuple(numeric_features) == baseline.NUMERIC_FEATURE_NAMES
        and tuple(categorical_features)
        == baseline.CATEGORICAL_FEATURE_NAMES
    ):
        training_ids = {row.row_id for row in training_features}
        model = baseline.fit_baseline_model(
            training_features,
            targets_by_row={
                row_id: int(targets_by_row[row_id])
                for row_id in training_ids
            },
            regularization_c=regularization_c,
        )
        return (
            model.predict_scores(validation_features),
            {
                **dict(model.fit_diagnostics),
                "active_column_count": len(
                    model.encoder.column_names
                ),
                "active_numeric_features": list(numeric_features),
                "active_categorical_features": list(
                    categorical_features
                ),
            },
        )
    encoder = baseline.BaselineEncoder.fit(training_features)
    ordered_training, training_matrix = encoder.transform(training_features)
    ordered_validation, validation_matrix = encoder.transform(
        validation_features
    )
    training_design, design_diagnostics = _diagnostic_design(
        encoder,
        training_matrix,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    validation_design, validation_design_diagnostics = (
        _diagnostic_design(
            encoder,
            validation_matrix,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )
    )
    if (
        design_diagnostics["parameter_names"]
        != validation_design_diagnostics["parameter_names"]
    ):
        raise ProcessTriageDevelopmentError(
            "training and validation diagnostic designs disagree"
        )
    training_targets = np.asarray(
        [float(targets_by_row[row.row_id]) for row in ordered_training],
        dtype=np.float64,
    )
    coefficients, diagnostics = baseline._fit_coefficients(
        training_design,
        training_targets,
        regularization_c=regularization_c,
    )
    probabilities = baseline._sigmoid(
        validation_design @ coefficients
    )
    return (
        {
            row.row_id: float(probability)
            for row, probability in zip(
                ordered_validation,
                probabilities,
            )
        },
        {
            **diagnostics,
            "active_column_count": int(training_design.shape[1]),
            "active_numeric_features": list(numeric_features),
            "active_categorical_features": list(categorical_features),
            "diagnostic_design": design_diagnostics,
        },
    )


def _configured_oof(
    trajectories: Sequence[TriageTrajectory],
    features: Sequence[CheapStepFeatures],
    *,
    cv_manifest: Mapping[str, Any],
    group_aliases: Mapping[str, str],
    configuration_id: str,
) -> tuple[dict[str, Any], dict[str, float]]:
    if configuration_id not in CONFIGURATIONS:
        raise ProcessTriageDevelopmentError(
            f"unknown baseline diagnostic configuration: {configuration_id}"
        )
    if (
        cv_manifest.get("assignment_id") != BASELINE_CV_ID
        or cv_manifest.get("seed_id") != BASELINE_CV_SEED_ID
    ):
        raise ProcessTriageDevelopmentError(
            "diagnostics require the frozen baseline CV manifest"
        )
    aliases = dict(group_aliases)
    ordered_features = baseline._validate_feature_rows(features)
    target_by_row = baseline._targets_by_row(trajectories)
    feature_cluster = {
        row.row_id: _effective_cluster(row.group_id, aliases)
        for row in ordered_features
    }
    configuration = CONFIGURATIONS[configuration_id]
    candidate_rows = []
    candidate_scores: dict[float, dict[str, float]] = {}
    for regularization_c in BaselineFreezeCandidate().regularization_grid:
        oof_scores: dict[str, float] = {}
        folds = []
        for fold in sorted(
            cv_manifest["folds"],
            key=lambda row: int(row["fold_index"]),
        ):
            training_clusters = set(fold["training_cluster_ids"])
            validation_clusters = set(fold["validation_cluster_ids"])
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
            fold_scores, fit_diagnostics = _fit_and_score(
                training_features,
                validation_features,
                targets_by_row=target_by_row,
                regularization_c=float(regularization_c),
                numeric_features=configuration["numeric"],
                categorical_features=configuration["categorical"],
            )
            if set(oof_scores) & set(fold_scores):
                raise ProcessTriageDevelopmentError(
                    "diagnostic OOF rows overlap across folds"
                )
            oof_scores.update(fold_scores)
            folds.append(
                {
                    "fold_index": int(fold["fold_index"]),
                    "fit_diagnostics": fit_diagnostics,
                }
            )
        if set(oof_scores) != set(target_by_row):
            raise ProcessTriageDevelopmentError(
                "diagnostic OOF score coverage is incomplete"
            )
        evaluation = evaluate_global_review_budget(
            trajectories,
            scores=oof_scores,
            budget_fraction=0.10,
        )
        candidate_rows.append(
            {
                "regularization_c": float(regularization_c),
                "first_actionable_defect_recall": evaluation[
                    "first_actionable_defect_recall"
                ],
                "clean_row_allocation": evaluation[
                    "clean_row_allocation"
                ],
                "clean_trajectory_alert_rate": evaluation[
                    "clean_trajectory_alert_rate"
                ],
                "row_log_loss": baseline._binary_log_loss(
                    oof_scores,
                    target_by_row,
                ),
                "folds": folds,
            }
        )
        candidate_scores[float(regularization_c)] = oof_scores

    selected = min(
        candidate_rows,
        key=lambda row: (
            -float(row["first_actionable_defect_recall"]),
            float(row["clean_row_allocation"]),
            float(row["regularization_c"]),
        ),
    )
    selected_c = float(selected["regularization_c"])
    scores = candidate_scores[selected_c]
    evaluation = evaluate_global_review_budget(
        trajectories,
        scores=scores,
        budget_fraction=0.10,
    )
    score_hash = hashlib.sha256(
        _canonical_json(
            {row_id: scores[row_id] for row_id in sorted(scores)}
        ).encode("utf-8")
    ).hexdigest()
    return (
        {
            "configuration_id": configuration_id,
            "active_numeric_features": list(configuration["numeric"]),
            "active_categorical_features": list(
                configuration["categorical"]
            ),
            "candidate_rows": candidate_rows,
            "selected_regularization_c": selected_c,
            "selected_oof_evaluation": evaluation,
            "selected_oof_score_payload_sha256": score_hash,
        },
        scores,
    )


def _leave_one_source_slot_out(
    trajectories: Sequence[TriageTrajectory],
    features: Sequence[CheapStepFeatures],
    *,
    selected_regularization_c: float,
) -> dict[str, Any]:
    ordered_features = baseline._validate_feature_rows(features)
    targets = baseline._targets_by_row(trajectories)
    slots = sorted({int(row.source_slot) for row in ordered_features})
    pooled_scores: dict[str, float] = {}
    rows = []
    for slot in slots:
        training = tuple(
            row for row in ordered_features if int(row.source_slot) != slot
        )
        validation = tuple(
            row for row in ordered_features if int(row.source_slot) == slot
        )
        scores, diagnostics = _fit_and_score(
            training,
            validation,
            targets_by_row=targets,
            regularization_c=selected_regularization_c,
            numeric_features=baseline.NUMERIC_FEATURE_NAMES,
            categorical_features=baseline.CATEGORICAL_FEATURE_NAMES,
            reproduce_sealed_full_parameterization=False,
        )
        pooled_scores.update(scores)
        slot_trajectories = tuple(
            trajectory
            for trajectory in trajectories
            if int(trajectory.source_slot) == slot
        )
        rows.append(
            {
                "omitted_source_slot": slot,
                "training_row_count": len(training),
                "validation_row_count": len(validation),
                "fit_diagnostics": diagnostics,
                "evaluation": evaluate_global_review_budget(
                    slot_trajectories,
                    scores=scores,
                    budget_fraction=0.10,
                ),
            }
        )
    if set(pooled_scores) != set(targets):
        raise ProcessTriageDevelopmentError(
            "leave-one-source-slot scores do not cover development rows"
        )
    return {
        "named_source_model_interpretation_authorized": False,
        "source_slot_treated_as": "opaque_categorical_nuisance",
        "selected_regularization_c": selected_regularization_c,
        "rows": rows,
        "pooled_evaluation": evaluate_global_review_budget(
            trajectories,
            scores=pooled_scores,
            budget_fraction=0.10,
        ),
        "pooled_score_payload_sha256": hashlib.sha256(
            _canonical_json(
                {
                    row_id: pooled_scores[row_id]
                    for row_id in sorted(pooled_scores)
                }
            ).encode("utf-8")
        ).hexdigest(),
    }


def _largest_cluster_sensitivity(
    trajectories: Sequence[TriageTrajectory],
    *,
    scores: Mapping[str, float],
    group_aliases: Mapping[str, str],
) -> dict[str, Any]:
    aliases = dict(group_aliases)
    counts = Counter(
        _effective_cluster(trajectory.group_id, aliases)
        for trajectory in trajectories
    )
    maximum = max(counts.values())
    largest = sorted(
        cluster_id
        for cluster_id, count in counts.items()
        if count == maximum
    )
    rows = []
    for omitted in largest:
        subset = tuple(
            trajectory
            for trajectory in trajectories
            if _effective_cluster(trajectory.group_id, aliases) != omitted
        )
        row_ids = {
            step.row_id for trajectory in subset for step in trajectory.steps
        }
        rows.append(
            {
                "omitted_cluster_id": omitted,
                "omitted_trajectory_count": maximum,
                "evaluation": evaluate_global_review_budget(
                    subset,
                    scores={
                        row_id: float(scores[row_id]) for row_id in row_ids
                    },
                    budget_fraction=0.10,
                ),
            }
        )
    return {
        "maximum_cluster_trajectory_count": maximum,
        "largest_cluster_count": len(largest),
        "rows": rows,
    }


def run_baseline_diagnostics(
    trajectories: Sequence[TriageTrajectory],
    features: Sequence[CheapStepFeatures],
    *,
    cv_manifest: Mapping[str, Any],
    group_aliases: Mapping[str, str],
    expected_full_oof_sha256: str,
) -> dict[str, Any]:
    configuration_reports = {}
    full_scores: dict[str, float] | None = None
    for configuration_id in CONFIGURATIONS:
        report, scores = _configured_oof(
            trajectories,
            features,
            cv_manifest=cv_manifest,
            group_aliases=group_aliases,
            configuration_id=configuration_id,
        )
        configuration_reports[configuration_id] = report
        if configuration_id == "full_cheap":
            full_scores = scores
    assert full_scores is not None
    actual_full_hash = configuration_reports["full_cheap"][
        "selected_oof_score_payload_sha256"
    ]
    if actual_full_hash != expected_full_oof_sha256:
        raise ProcessTriageDevelopmentError(
            "full cheap baseline no longer reproduces the sealed OOF hash"
        )
    selected_c = float(
        configuration_reports["full_cheap"][
            "selected_regularization_c"
        ]
    )
    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "epistemic_status": "development_baseline_diagnostics_only",
        "configuration_reports": configuration_reports,
        "full_baseline_hash_reproduced": True,
        "leave_one_source_slot_out": _leave_one_source_slot_out(
            trajectories,
            features,
            selected_regularization_c=selected_c,
        ),
        "largest_development_cluster_sensitivity": (
            _largest_cluster_sensitivity(
                trajectories,
                scores=full_scores,
                group_aliases=group_aliases,
            )
        ),
        "structural_signal_opened": False,
        "prospective_locked_partition_scored": False,
        "public_claim_authorized": False,
    }
