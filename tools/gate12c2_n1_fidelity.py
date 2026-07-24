#!/usr/bin/env python3
"""Development-only nuisance-fidelity profiling for the Gate12C-2 N1 null.

This module characterizes what role-constrained node-frame reassignment keeps
and changes before any nuisance threshold or locked synthetic suite is frozen.
It cannot authorize calibration, N2, locked execution, or a scientific claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

import gate12c2_synthetic_lab as lab


PROFILE_SCHEMA_VERSION = "gate12c2_n1_fidelity_profile_v0.1"
DEFAULT_S1_EFFECT_STRENGTHS = (0.05, 0.15, 0.25, 0.40)


class Gate12C2FidelityError(ValueError):
    """Raised when a development-only fidelity profile is malformed."""


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _payload_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _distribution_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise Gate12C2FidelityError(
            "fidelity summaries require finite nonempty vectors"
        )
    absolute = np.abs(array)
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "maximum": float(np.max(array)),
        "p95_absolute": float(np.quantile(absolute, 0.95)),
        "maximum_absolute": float(np.max(absolute)),
    }


def _configuration_grid(
    effect_strengths: Sequence[float],
) -> tuple[tuple[str, float | None], ...]:
    strengths = tuple(float(value) for value in effect_strengths)
    if not strengths:
        raise Gate12C2FidelityError(
            "at least one S1 effect strength is required"
        )
    if (
        any(not math.isfinite(value) or value <= 0.0 or value > 0.5 for value in strengths)
        or len(set(strengths)) != len(strengths)
    ):
        raise Gate12C2FidelityError(
            "S1 effect strengths must be unique finite values in (0, 0.5]"
        )
    return (
        ("S0_true_null", None),
        *(
            ("S1_known_reverse_shared_node_coupling", value)
            for value in strengths
        ),
    )


def _configuration_id(
    regime_id: str,
    effect_strength: float | None,
) -> str:
    if effect_strength is None:
        return regime_id
    return f"{regime_id}:effect={effect_strength:.8g}"


def _eligibility_rate(
    graphs: Sequence[lab.SyntheticGraph],
    *,
    q: int,
) -> float:
    return sum(
        lab.graph_residual_diagnostics(graph, q=q).defect is not None
        for graph in graphs
    ) / len(graphs)


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise Gate12C2FidelityError("one fidelity configuration has no rows")
    exact_fields = (
        "assignment_audit_pass",
        "all_direct_realizability_pass",
        "all_block_gram_realizability_pass",
        "zero_fixed_points",
        "zero_same_graph_assignments",
        "zero_crossed_strata",
        "zero_reused_or_unused_donors",
    )
    return {
        "profile_count": len(rows),
        "case_count": len({str(row["case_id"]) for row in rows}),
        "draw_count_per_case": int(
            len(rows)
            / len({str(row["case_id"]) for row in rows})
        ),
        "hard_constraint_pass_rates": {
            field_name: float(
                sum(bool(row[field_name]) for row in rows) / len(rows)
            )
            for field_name in exact_fields
        },
        "mean_absolute_sorted_edge_spectrum_difference": (
            _distribution_summary(
                [
                    float(row["mean_absolute_sorted_spectrum_difference"])
                    for row in rows
                ]
            )
        ),
        "mean_absolute_sorted_edge_spectrum_difference_relative_to_observed_mean": (
            _distribution_summary(
                [
                    float(row["mean_absolute_sorted_spectrum_difference_relative"])
                    for row in rows
                ]
            )
        ),
        "maximum_absolute_sorted_edge_spectrum_difference": (
            _distribution_summary(
                [
                    float(row["maximum_absolute_sorted_spectrum_difference"])
                    for row in rows
                ]
            )
        ),
        "edge_spectrum_mean_shift_relative_to_observed_mean": (
            _distribution_summary(
                [
                    float(row["edge_spectrum_mean_shift_relative"])
                    for row in rows
                ]
            )
        ),
        "stable_cut_eligibility_rate_shift": {
            f"q{q}": _distribution_summary(
                [
                    float(row[f"q{q}_eligibility_rate_shift"])
                    for row in rows
                ]
            )
            for q in (1, 2)
        },
        "row_payload_sha256": _payload_sha256(list(rows)),
    }


def run_development_n1_fidelity_profile(
    *,
    master_seed: str,
    block_count: int,
    draw_count_per_case: int,
    effect_strengths: Sequence[float] = DEFAULT_S1_EFFECT_STRENGTHS,
) -> dict[str, Any]:
    """Profile N1 on S0 and graded S1 development cohorts only."""

    if not master_seed:
        raise Gate12C2FidelityError("master_seed must be nonempty")
    if block_count < 4:
        raise Gate12C2FidelityError("block_count must be at least four")
    if draw_count_per_case <= 0:
        raise Gate12C2FidelityError(
            "draw_count_per_case must be positive"
        )
    configurations = _configuration_grid(effect_strengths)
    started = time.perf_counter()
    configuration_rows: dict[str, list[dict[str, Any]]] = {}
    seed_namespace_hashes: list[str] = []
    for regime_id, effect_strength in configurations:
        configuration_id = _configuration_id(regime_id, effect_strength)
        rows: list[dict[str, Any]] = []
        for case in lab._outer_case_grid():
            observed, observed_seed_receipt = lab._outer_observed_cohort(
                regime_id=regime_id,
                master_seed=master_seed,
                outer_experiment_index=0,
                case=case,
                block_count=block_count,
                effect_strength=effect_strength,
            )
            observed_manifest_sha256 = _payload_sha256(
                lab.manifests(observed)
            )
            observed_eligibility = {
                q: _eligibility_rate(observed, q=q) for q in (1, 2)
            }
            for draw_index in range(draw_count_per_case):
                namespace = lab.OuterSeedNamespace(
                    surface_id="development",
                    null_candidate_id=lab.N1_ID,
                    regime_id=regime_id,
                    effect_strength=effect_strength,
                    outer_experiment_index=0,
                    case_or_endpoint_id=str(case["case_id"]),
                    cycle_or_root_id="n1_nuisance_fidelity_profile",
                    draw_attempt_index=draw_index,
                )
                seed_receipt = lab.typed_seed_receipt(
                    master_seed,
                    namespace,
                )
                seed_namespace_hashes.append(
                    str(seed_receipt["namespace_sha256"])
                )
                reassigned = lab.n1_role_constrained_reassignment(
                    observed,
                    reassignment_seed=str(
                        seed_receipt["seed_receipt_sha256"]
                    ),
                )
                audit = lab.n1_reassignment_audit(
                    observed,
                    reassigned,
                )
                direct_pass_count = sum(
                    lab.check_joint_realizability(graph)["status"] == "pass"
                    for graph in reassigned
                )
                gram_pass_count = sum(
                    lab.check_block_gram_realizability(graph)["status"]
                    == "pass"
                    for graph in reassigned
                )
                spectrum = lab.edge_spectrum_marginal_discrepancy(
                    observed,
                    reassigned,
                )
                observed_mean = max(
                    float(spectrum["observed_mean"]),
                    np.finfo(np.float64).tiny,
                )
                reassigned_eligibility = {
                    q: _eligibility_rate(reassigned, q=q)
                    for q in (1, 2)
                }
                rows.append(
                    {
                        "case_id": str(case["case_id"]),
                        "draw_index": draw_index,
                        "observed_seed_namespace_sha256": (
                            observed_seed_receipt["namespace_sha256"]
                        ),
                        "observed_manifest_sha256": (
                            observed_manifest_sha256
                        ),
                        "n1_seed_namespace_sha256": (
                            seed_receipt["namespace_sha256"]
                        ),
                        "assignment_audit_pass": (
                            audit["status"] == "pass"
                        ),
                        "all_direct_realizability_pass": (
                            direct_pass_count == block_count
                        ),
                        "all_block_gram_realizability_pass": (
                            gram_pass_count == block_count
                        ),
                        "zero_fixed_points": (
                            audit["fixed_point_count"] == 0
                        ),
                        "zero_same_graph_assignments": (
                            audit["same_graph_assignment_count"] == 0
                        ),
                        "zero_crossed_strata": (
                            audit["crossed_stratum_count"] == 0
                        ),
                        "zero_reused_or_unused_donors": (
                            not audit["unused_donor_references"]
                            and not audit["reused_donor_counts"]
                        ),
                        "stratum_count": int(audit["stratum_count"]),
                        "stratum_size_min": int(
                            audit["stratum_size_min"]
                        ),
                        "stratum_size_max": int(
                            audit["stratum_size_max"]
                        ),
                        "mean_absolute_sorted_spectrum_difference": (
                            spectrum[
                                "mean_absolute_sorted_difference"
                            ]
                        ),
                        "mean_absolute_sorted_spectrum_difference_relative": (
                            float(
                                spectrum[
                                    "mean_absolute_sorted_difference"
                                ]
                            )
                            / observed_mean
                        ),
                        "maximum_absolute_sorted_spectrum_difference": (
                            spectrum[
                                "maximum_absolute_sorted_difference"
                            ]
                        ),
                        "edge_spectrum_mean_shift_relative": (
                            float(spectrum["mean_shift"]) / observed_mean
                        ),
                        "q1_eligibility_rate_shift": (
                            reassigned_eligibility[1]
                            - observed_eligibility[1]
                        ),
                        "q2_eligibility_rate_shift": (
                            reassigned_eligibility[2]
                            - observed_eligibility[2]
                        ),
                    }
                )
        configuration_rows[configuration_id] = rows

    summaries = {
        configuration_id: _summarize_rows(rows)
        for configuration_id, rows in configuration_rows.items()
    }
    deterministic_projection = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "master_seed_receipt_sha256": lab._seed_receipt(master_seed),
        "block_count": block_count,
        "draw_count_per_case": draw_count_per_case,
        "s1_effect_strengths": [
            float(value) for value in effect_strengths
        ],
        "case_grid": [dict(case) for case in lab._outer_case_grid()],
        "configuration_summaries": summaries,
        "seed_namespace_sequence_sha256": _payload_sha256(
            seed_namespace_hashes
        ),
    }
    return {
        **deterministic_projection,
        "epistemic_status": (
            "development_nuisance_exploration_not_threshold_freeze"
        ),
        "surface_id": "development",
        "null_candidate_id": lab.N1_ID,
        "elapsed_seconds": float(time.perf_counter() - started),
        "interpretation_boundary": {
            "calibration_gate_decision_authorized": False,
            "nuisance_threshold_frozen": False,
            "locked_synthetic_execution_authorized": False,
            "real_held_out_execution_authorized": False,
            "N2_open": False,
            "N3_open": False,
            "maximum_sorted_spectrum_difference_is_diagnostic_only": True,
        },
        "deterministic_projection_sha256": _payload_sha256(
            deterministic_projection
        ),
    }
