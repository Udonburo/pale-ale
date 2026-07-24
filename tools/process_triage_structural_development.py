#!/usr/bin/env python3
"""Development-only evaluation for the frozen structural triage signal."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np

import process_triage_baseline as baseline
from process_triage_baseline_diagnostics import _diagnostic_design
from process_triage_evaluator import (
    BASELINE_CV_FOLD_COUNT,
    BASELINE_CV_ID,
    BASELINE_CV_SEED_ID,
    BaselineFreezeCandidate,
    CheapStepFeatures,
    ProcessTriageDevelopmentError,
    TriageTrajectory,
    evaluate_global_review_budget,
)
from process_triage_structural_signal import (
    StructuralStepScore,
)


STRUCTURAL_DEVELOPMENT_SCHEMA_VERSION = (
    "pale_ale_structural_development_v0.1"
)
LABEL_PERMUTATION_SEED_ID = (
    "pale-ale-process-triage-label-profile-permutation-v0.1"
)
PRIMARY_STRUCTURAL_COLUMN = "task_anchored_triangle_excess"
ABLATION_CONFIGURATIONS = {
    "remove_trajectory_length": {
        "numeric": tuple(
            name
            for name in baseline.NUMERIC_FEATURE_NAMES
            if name != "eligible_trajectory_length"
        ),
        "categorical": baseline.CATEGORICAL_FEATURE_NAMES,
    },
    "remove_normalized_position": {
        "numeric": tuple(
            name
            for name in baseline.NUMERIC_FEATURE_NAMES
            if name != "normalized_position"
        ),
        "categorical": baseline.CATEGORICAL_FEATURE_NAMES,
    },
    "remove_retry_and_tool_error_features": {
        "numeric": tuple(
            name
            for name in baseline.NUMERIC_FEATURE_NAMES
            if name
            not in {
                "prior_exact_retry_count",
                "prior_same_tool_count",
                "preceding_tool_error",
                "prior_tool_error_count",
            }
        ),
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


def _score_map(
    rows: Sequence[StructuralStepScore],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for row in rows:
        if row.row_id in result:
            raise ProcessTriageDevelopmentError(
                f"duplicate structural row: {row.row_id}"
            )
        score = float(row.score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ProcessTriageDevelopmentError(
                f"invalid structural score: {row.row_id}"
            )
        result[row.row_id] = score
    return result


def _validate_cv(
    cv_manifest: Mapping[str, Any],
) -> None:
    if (
        cv_manifest.get("assignment_id") != BASELINE_CV_ID
        or cv_manifest.get("seed_id") != BASELINE_CV_SEED_ID
        or int(cv_manifest.get("fold_count", -1))
        != BASELINE_CV_FOLD_COUNT
        or cv_manifest.get("label_or_outcome_fields_used") != []
    ):
        raise ProcessTriageDevelopmentError(
            "structural development requires the frozen baseline CV"
        )


def _base_design(
    encoder: baseline.BaselineEncoder,
    matrix: np.ndarray,
    *,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    sealed_full_parameterization: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    is_full = (
        tuple(numeric_features) == baseline.NUMERIC_FEATURE_NAMES
        and tuple(categorical_features)
        == baseline.CATEGORICAL_FEATURE_NAMES
    )
    if is_full and sealed_full_parameterization:
        return matrix, {
            "parameterization": "sealed_full_one_hot",
            "design_column_count": int(matrix.shape[1]),
            "column_names": list(encoder.column_names),
        }
    return _diagnostic_design(
        encoder,
        matrix,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )


def _fit_and_score(
    training_features: Sequence[CheapStepFeatures],
    validation_features: Sequence[CheapStepFeatures],
    *,
    targets_by_row: Mapping[str, int],
    structural_by_row: Mapping[str, float] | None,
    regularization_c: float,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    sealed_full_parameterization: bool,
) -> tuple[dict[str, float], dict[str, Any]]:
    encoder = baseline.BaselineEncoder.fit(training_features)
    ordered_training, raw_training = encoder.transform(
        training_features
    )
    ordered_validation, raw_validation = encoder.transform(
        validation_features
    )
    training_design, design_metadata = _base_design(
        encoder,
        raw_training,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        sealed_full_parameterization=sealed_full_parameterization,
    )
    validation_design, validation_metadata = _base_design(
        encoder,
        raw_validation,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        sealed_full_parameterization=sealed_full_parameterization,
    )
    if (
        design_metadata["design_column_count"]
        != validation_metadata["design_column_count"]
    ):
        raise ProcessTriageDevelopmentError(
            "training and validation base designs disagree"
        )

    structural_metadata: dict[str, Any] = {
        "included": structural_by_row is not None
    }
    if structural_by_row is not None:
        row_coverage = {
            row.row_id for row in (*ordered_training, *ordered_validation)
        }
        if not row_coverage <= set(structural_by_row):
            raise ProcessTriageDevelopmentError(
                "structural score coverage is incomplete"
            )
        training_structural = np.asarray(
            [
                float(structural_by_row[row.row_id])
                for row in ordered_training
            ],
            dtype=np.float64,
        )
        validation_structural = np.asarray(
            [
                float(structural_by_row[row.row_id])
                for row in ordered_validation
            ],
            dtype=np.float64,
        )
        mean = float(np.mean(training_structural))
        scale = float(np.std(training_structural))
        if scale <= 0.0:
            scale = 1.0
        training_design = np.column_stack(
            (
                training_design,
                (training_structural - mean) / scale,
            )
        )
        validation_design = np.column_stack(
            (
                validation_design,
                (validation_structural - mean) / scale,
            )
        )
        structural_metadata.update(
            {
                "column": PRIMARY_STRUCTURAL_COLUMN,
                "training_mean": mean,
                "training_scale": scale,
                "zero_variance_scale": 1.0,
            }
        )

    training_targets = np.asarray(
        [
            float(targets_by_row[row.row_id])
            for row in ordered_training
        ],
        dtype=np.float64,
    )
    coefficients, fit_diagnostics = baseline._fit_coefficients(
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
            **fit_diagnostics,
            "base_design": design_metadata,
            "structural": structural_metadata,
            "final_design_column_count": int(
                training_design.shape[1]
            ),
        },
    )


def fit_development_configuration(
    trajectories: Sequence[TriageTrajectory],
    features: Sequence[CheapStepFeatures],
    *,
    cv_manifest: Mapping[str, Any],
    group_aliases: Mapping[str, str],
    configuration_id: str,
    structural_by_row: Mapping[str, float] | None,
    numeric_features: Sequence[str] = baseline.NUMERIC_FEATURE_NAMES,
    categorical_features: Sequence[
        str
    ] = baseline.CATEGORICAL_FEATURE_NAMES,
    sealed_full_parameterization: bool = True,
) -> tuple[dict[str, Any], dict[str, float]]:
    _validate_cv(cv_manifest)
    aliases = dict(group_aliases)
    ordered_features = baseline._validate_feature_rows(features)
    targets = baseline._targets_by_row(trajectories)
    if {row.row_id for row in ordered_features} != set(targets):
        raise ProcessTriageDevelopmentError(
            "structural development row coverage mismatch"
        )
    if (
        structural_by_row is not None
        and set(structural_by_row) != set(targets)
    ):
        raise ProcessTriageDevelopmentError(
            "structural scores do not cover the development rows exactly"
        )
    feature_cluster = {
        row.row_id: _effective_cluster(row.group_id, aliases)
        for row in ordered_features
    }
    grid = tuple(
        float(value)
        for value in BaselineFreezeCandidate().regularization_grid
    )
    candidates = []
    scores_by_c: dict[float, dict[str, float]] = {}
    for regularization_c in grid:
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
            fold_scores, diagnostics = _fit_and_score(
                training_features,
                validation_features,
                targets_by_row=targets,
                structural_by_row=structural_by_row,
                regularization_c=regularization_c,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                sealed_full_parameterization=(
                    sealed_full_parameterization
                ),
            )
            if set(oof_scores) & set(fold_scores):
                raise ProcessTriageDevelopmentError(
                    "structural OOF rows overlap"
                )
            oof_scores.update(fold_scores)
            folds.append(
                {
                    "fold_index": int(fold["fold_index"]),
                    "fit_diagnostics": diagnostics,
                }
            )
        if set(oof_scores) != set(targets):
            raise ProcessTriageDevelopmentError(
                "structural OOF coverage is incomplete"
            )
        evaluation = evaluate_global_review_budget(
            trajectories,
            scores=oof_scores,
            budget_fraction=0.10,
        )
        candidates.append(
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
                "row_log_loss": baseline._binary_log_loss(
                    oof_scores,
                    targets,
                ),
                "folds": folds,
            }
        )
        scores_by_c[regularization_c] = oof_scores

    selected = min(
        candidates,
        key=lambda row: (
            -float(row["first_actionable_defect_recall"]),
            float(row["clean_row_allocation"]),
            float(row["regularization_c"]),
        ),
    )
    selected_c = float(selected["regularization_c"])
    selected_scores = scores_by_c[selected_c]
    evaluation = evaluate_global_review_budget(
        trajectories,
        scores=selected_scores,
        budget_fraction=0.10,
    )
    all_scores, refit_diagnostics = _fit_and_score(
        ordered_features,
        ordered_features,
        targets_by_row=targets,
        structural_by_row=structural_by_row,
        regularization_c=selected_c,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        sealed_full_parameterization=sealed_full_parameterization,
    )
    score_hash = hashlib.sha256(
        _canonical_json(
            {
                row_id: selected_scores[row_id]
                for row_id in sorted(selected_scores)
            }
        ).encode("utf-8")
    ).hexdigest()
    refit_prediction_hash = hashlib.sha256(
        _canonical_json(
            {row_id: all_scores[row_id] for row_id in sorted(all_scores)}
        ).encode("utf-8")
    ).hexdigest()
    return (
        {
            "schema_version": STRUCTURAL_DEVELOPMENT_SCHEMA_VERSION,
            "epistemic_status": "development_only",
            "configuration_id": configuration_id,
            "structural_included": structural_by_row is not None,
            "active_numeric_features": list(numeric_features),
            "active_categorical_features": list(
                categorical_features
            ),
            "sealed_full_parameterization": (
                sealed_full_parameterization
            ),
            "candidate_rows": candidates,
            "selected_regularization_c": selected_c,
            "selected_oof_evaluation": evaluation,
            "selected_oof_score_payload_sha256": score_hash,
            "refit_prediction_payload_sha256": (
                refit_prediction_hash
            ),
            "refit_diagnostics": refit_diagnostics,
            "locked_partition_scored": False,
            "public_claim_authorized": False,
        },
        selected_scores,
    )


def _profile_permutation_shift(
    length: int,
    *,
    domain: str,
    eligible_length: int,
) -> int:
    if length <= 1:
        return 0
    digest = hashlib.sha256(
        _canonical_json(
            [
                LABEL_PERMUTATION_SEED_ID,
                domain,
                eligible_length,
            ]
        ).encode("utf-8")
    ).digest()
    return 1 + (
        int.from_bytes(digest[:8], byteorder="big", signed=False)
        % (length - 1)
    )


def permute_first_defect_profiles(
    trajectories: Sequence[TriageTrajectory],
) -> tuple[tuple[TriageTrajectory, ...], dict[str, Any]]:
    strata: dict[tuple[str, int], list[TriageTrajectory]] = defaultdict(list)
    for trajectory in trajectories:
        strata[(trajectory.domain, len(trajectory.steps))].append(
            trajectory
        )
    result = []
    singleton_count = 0
    for (domain, eligible_length), members in sorted(strata.items()):
        ordered = sorted(members, key=lambda row: row.trajectory_id)
        profiles = [
            next(
                (
                    step.eligible_index
                    for step in trajectory.steps
                    if step.actionable_defect
                ),
                None,
            )
            for trajectory in ordered
        ]
        shift = _profile_permutation_shift(
            len(ordered),
            domain=domain,
            eligible_length=eligible_length,
        )
        singleton_count += int(len(ordered) == 1)
        for target_index, trajectory in enumerate(ordered):
            profile = profiles[(target_index + shift) % len(ordered)]
            steps = tuple(
                replace(
                    step,
                    actionable_defect=(
                        profile is not None
                        and step.eligible_index == profile
                    ),
                    native_label=(
                        -1
                        if profile is not None
                        and step.eligible_index == profile
                        else 1
                    ),
                )
                for step in trajectory.steps
            )
            result.append(
                replace(
                    trajectory,
                    steps=steps,
                    final_label=None,
                )
            )
    ordered_result = tuple(
        sorted(result, key=lambda row: row.trajectory_id)
    )
    original_positive = sum(
        trajectory.first_actionable_row_id is not None
        for trajectory in trajectories
    )
    permuted_positive = sum(
        trajectory.first_actionable_row_id is not None
        for trajectory in ordered_result
    )
    if original_positive != permuted_positive:
        raise ProcessTriageDevelopmentError(
            "label profile permutation changed positive prevalence"
        )
    return ordered_result, {
        "seed_id": LABEL_PERMUTATION_SEED_ID,
        "stratification": "domain_x_eligible_trajectory_length",
        "permutation": "deterministic_nonzero_cyclic_shift",
        "stratum_count": len(strata),
        "singleton_stratum_count": singleton_count,
        "trajectory_count": len(ordered_result),
        "positive_trajectory_count": permuted_positive,
    }


def _increment(
    augmented: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, float]:
    augmented_evaluation = augmented["selected_oof_evaluation"]
    reference_evaluation = reference["selected_oof_evaluation"]
    return {
        "first_actionable_defect_recall": (
            float(
                augmented_evaluation[
                    "first_actionable_defect_recall"
                ]
            )
            - float(
                reference_evaluation[
                    "first_actionable_defect_recall"
                ]
            )
        ),
        "clean_row_allocation": (
            float(augmented_evaluation["clean_row_allocation"])
            - float(reference_evaluation["clean_row_allocation"])
        ),
        "clean_trajectory_alert_rate": (
            float(
                augmented_evaluation[
                    "clean_trajectory_alert_rate"
                ]
            )
            - float(
                reference_evaluation[
                    "clean_trajectory_alert_rate"
                ]
            )
        ),
    }


def run_structural_development(
    trajectories: Sequence[TriageTrajectory],
    features: Sequence[CheapStepFeatures],
    *,
    cv_manifest: Mapping[str, Any],
    group_aliases: Mapping[str, str],
    structural_modes: Mapping[
        str, Sequence[StructuralStepScore]
    ],
) -> dict[str, Any]:
    required_modes = {
        "primary",
        "score_order_shuffle",
        "dependency_cycle_randomization",
    }
    if set(structural_modes) != required_modes:
        raise ProcessTriageDevelopmentError(
            "structural development modes differ from the freeze"
        )
    score_maps = {
        mode: _score_map(rows)
        for mode, rows in structural_modes.items()
    }
    target_rows = set(baseline._targets_by_row(trajectories))
    if any(set(scores) != target_rows for scores in score_maps.values()):
        raise ProcessTriageDevelopmentError(
            "one structural mode has incomplete row coverage"
        )

    sealed_baseline, sealed_baseline_scores = (
        fit_development_configuration(
            trajectories,
            features,
            cv_manifest=cv_manifest,
            group_aliases=group_aliases,
            configuration_id="sealed_full_cheap_baseline",
            structural_by_row=None,
        )
    )
    primary, primary_scores = fit_development_configuration(
        trajectories,
        features,
        cv_manifest=cv_manifest,
        group_aliases=group_aliases,
        configuration_id="full_cheap_plus_primary_structural",
        structural_by_row=score_maps["primary"],
    )
    controls = {}
    for mode in (
        "score_order_shuffle",
        "dependency_cycle_randomization",
    ):
        report, _ = fit_development_configuration(
            trajectories,
            features,
            cv_manifest=cv_manifest,
            group_aliases=group_aliases,
            configuration_id=f"full_cheap_plus_{mode}",
            structural_by_row=score_maps[mode],
        )
        controls[mode] = {
            "report": report,
            "increment_vs_sealed_baseline": _increment(
                report,
                sealed_baseline,
            ),
        }

    structural_alone = evaluate_global_review_budget(
        trajectories,
        scores=score_maps["primary"],
        budget_fraction=0.10,
    )
    ablations = {}
    for ablation_id, configuration in ABLATION_CONFIGURATIONS.items():
        reference, _ = fit_development_configuration(
            trajectories,
            features,
            cv_manifest=cv_manifest,
            group_aliases=group_aliases,
            configuration_id=f"{ablation_id}_cheap_reference",
            structural_by_row=None,
            numeric_features=configuration["numeric"],
            categorical_features=configuration["categorical"],
            sealed_full_parameterization=False,
        )
        augmented, _ = fit_development_configuration(
            trajectories,
            features,
            cv_manifest=cv_manifest,
            group_aliases=group_aliases,
            configuration_id=f"{ablation_id}_plus_structural",
            structural_by_row=score_maps["primary"],
            numeric_features=configuration["numeric"],
            categorical_features=configuration["categorical"],
            sealed_full_parameterization=False,
        )
        ablations[ablation_id] = {
            "reference": reference,
            "augmented": augmented,
            "increment": _increment(augmented, reference),
        }

    permuted_trajectories, permutation_receipt = (
        permute_first_defect_profiles(trajectories)
    )
    permuted_reference, _ = fit_development_configuration(
        permuted_trajectories,
        features,
        cv_manifest=cv_manifest,
        group_aliases=group_aliases,
        configuration_id="label_permutation_cheap_reference",
        structural_by_row=None,
    )
    permuted_augmented, _ = fit_development_configuration(
        permuted_trajectories,
        features,
        cv_manifest=cv_manifest,
        group_aliases=group_aliases,
        configuration_id="label_permutation_plus_structural",
        structural_by_row=score_maps["primary"],
    )

    return {
        "schema_version": STRUCTURAL_DEVELOPMENT_SCHEMA_VERSION,
        "epistemic_status": "development_only_not_confirmatory",
        "structural_family_id": (
            "task_anchored_artifact_detour_v0.1"
        ),
        "sealed_baseline": sealed_baseline,
        "primary_augmented": primary,
        "primary_increment_vs_sealed_baseline": _increment(
            primary,
            sealed_baseline,
        ),
        "structural_alone_evaluation": structural_alone,
        "frozen_structural_controls": controls,
        "required_ablations": ablations,
        "label_permutation_control": {
            "permutation_receipt": permutation_receipt,
            "reference": permuted_reference,
            "augmented": permuted_augmented,
            "increment": _increment(
                permuted_augmented,
                permuted_reference,
            ),
        },
        "score_payload_sha256": {
            mode: hashlib.sha256(
                _canonical_json(
                    {
                        row_id: scores[row_id]
                        for row_id in sorted(scores)
                    }
                ).encode("utf-8")
            ).hexdigest()
            for mode, scores in score_maps.items()
        },
        "selected_score_payload_sha256": {
            "sealed_baseline": hashlib.sha256(
                _canonical_json(
                    {
                        row_id: sealed_baseline_scores[row_id]
                        for row_id in sorted(sealed_baseline_scores)
                    }
                ).encode("utf-8")
            ).hexdigest(),
            "primary_augmented": hashlib.sha256(
                _canonical_json(
                    {
                        row_id: primary_scores[row_id]
                        for row_id in sorted(primary_scores)
                    }
                ).encode("utf-8")
            ).hexdigest(),
        },
        "structural_signal_opened_on_development": True,
        "prospective_locked_partition_scored": False,
        "locked_evaluation_authorized": False,
        "public_claim_authorized": False,
    }
