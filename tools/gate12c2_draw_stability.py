#!/usr/bin/env python3
"""Outcome-blind stability projection for Gate12C-2 nested draw counts.

This module may read development results in order to compare deterministic
accepted-valid prefixes, but its public return value contains only absolute
shifts, agreement rates, counts, hashes, and gate statuses.  It never emits a
raw decision, rate, direction, p-value, endpoint summary, or S2 component
value.
"""

from __future__ import annotations

import math
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import gate12c2_development_shards as shards
import gate12c2_draw_profile as profile
import gate12c2_synthetic_lab as lab


PROJECTION_SCHEMA_VERSION = "gate12c2_no_outcome_draw_stability_v0.2"
ANALYSIS_MANIFEST_SCHEMA_VERSION = (
    "gate12c2_no_outcome_draw_stability_manifest_v0.2"
)
PREFIX_COUNTS = (255, 511, 1023)
REFERENCE_DRAW_COUNT = 1023
MINIMUM_ENDPOINT_DECISION_AGREEMENT = 0.99
MAXIMUM_ABSOLUTE_SUMMARY_SHIFT = 0.05
MAXIMUM_ABSOLUTE_S0_FAMILYWISE_SHIFT = 0.01
S2_MAGNITUDE_FIELDS = (
    "a_q",
    "u_q",
    "v_q",
    "x_q",
    "y_q",
    "p_L_q",
    "p_R_q",
)
S2_ALIGNMENT_FIELD = "c_q"
S2_ALWAYS_DEFINED_FIELDS = (
    "a_q",
    "u_q",
    "v_q",
    "x_q",
    "y_q",
)
S2_CONDITIONALLY_DEFINED_FIELDS = (
    "c_q",
    "p_L_q",
    "p_R_q",
)
S2_COMPONENT_FIELDS = (
    *S2_MAGNITUDE_FIELDS,
    S2_ALIGNMENT_FIELD,
)
REGIMES = (
    "S0_true_null",
    "S1_known_reverse_shared_node_coupling",
    "S2_null_inflation",
)


class Gate12C2DrawStabilityError(ValueError):
    """Raised when stability inputs or the no-outcome projection are unsafe."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_hashes() -> dict[str, str]:
    return {
        "gate12c2_draw_stability.py": _sha256_file(
            Path(__file__).resolve()
        ),
        "gate12c2_development_shards.py": _sha256_file(
            Path(shards.__file__).resolve()
        ),
        "gate12c2_draw_profile.py": _sha256_file(
            Path(profile.__file__).resolve()
        ),
        "gate12c2_synthetic_lab.py": _sha256_file(
            Path(lab.__file__).resolve()
        ),
        "run_gate12c2_draw_stability.py": _sha256_file(
            Path(__file__).with_name(
                "run_gate12c2_draw_stability.py"
            ).resolve()
        ),
    }


def _require_exact_keys(
    payload: Mapping[str, Any],
    expected: set[str],
    *,
    context: str,
) -> None:
    actual = set(payload)
    if actual != expected:
        raise Gate12C2DrawStabilityError(
            f"{context} keys differ from the closed allowlist: "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _finite(value: Any, *, context: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise Gate12C2DrawStabilityError(
            f"{context} must be finite"
        )
    return numeric


def _component_arm_summary(
    endpoint: Mapping[str, Any],
    field_name: str,
) -> dict[str, tuple[str, float | None]]:
    components = endpoint.get("component_medians")
    coverage = endpoint.get("component_coverage")
    if not isinstance(components, Mapping):
        raise Gate12C2DrawStabilityError(
            "S2 endpoint is missing component_medians"
        )
    if not isinstance(coverage, Mapping):
        raise Gate12C2DrawStabilityError(
            "S2 endpoint is missing component_coverage"
        )
    arms = ("observed", "N1", "graph_unconstrained_stressor")
    if set(components) != set(arms) or set(coverage) != set(arms):
        raise Gate12C2DrawStabilityError(
            "S2 endpoint changed the frozen component arm surface"
        )
    result: dict[str, tuple[str, float | None]] = {}
    count_keys = {
        "expected_count",
        "defined_count",
        "degenerate_count",
        "unexpected_missing_count",
        "nonfinite_count",
    }
    for arm in arms:
        arm_components = components[arm]
        arm_coverage = coverage[arm]
        if (
            not isinstance(arm_components, Mapping)
            or not isinstance(arm_coverage, Mapping)
            or set(arm_components)
            != set(S2_COMPONENT_FIELDS)
            or set(arm_coverage) != set(arm_components)
        ):
            raise Gate12C2DrawStabilityError(
                f"S2 component surface is incomplete for arm {arm!r}"
            )
        counts = arm_coverage[field_name]
        if not isinstance(counts, Mapping) or set(counts) != count_keys:
            raise Gate12C2DrawStabilityError(
                f"S2 coverage counts are incomplete for {arm}/{field_name}"
            )
        normalized = {key: int(counts[key]) for key in count_keys}
        if any(value < 0 for value in normalized.values()):
            raise Gate12C2DrawStabilityError(
                f"S2 coverage counts are negative for {arm}/{field_name}"
            )
        expected = normalized["expected_count"]
        accounted = sum(
            normalized[key]
            for key in (
                "defined_count",
                "degenerate_count",
                "unexpected_missing_count",
                "nonfinite_count",
            )
        )
        if expected <= 0 or accounted != expected:
            raise Gate12C2DrawStabilityError(
                f"S2 coverage does not conserve expected observations for "
                f"{arm}/{field_name}"
            )
        if (
            normalized["unexpected_missing_count"] > 0
            or normalized["nonfinite_count"] > 0
        ):
            raise Gate12C2DrawStabilityError(
                f"S2 component has unexpected missing/nonfinite coverage: "
                f"{arm}/{field_name}"
            )
        value = arm_components[field_name]
        if field_name in S2_ALWAYS_DEFINED_FIELDS:
            if (
                normalized["defined_count"] != expected
                or normalized["degenerate_count"] != 0
                or value is None
            ):
                raise Gate12C2DrawStabilityError(
                    f"always-defined S2 component lacks complete coverage: "
                    f"{arm}/{field_name}"
                )
            result[arm] = (
                "defined",
                _finite(value, context=f"{arm} {field_name}"),
            )
        else:
            if normalized["defined_count"] == 0:
                if (
                    normalized["degenerate_count"] != expected
                    or value is not None
                ):
                    raise Gate12C2DrawStabilityError(
                        "conditional S2 component has inconsistent "
                        f"all-degenerate coverage: {arm}/{field_name}"
                    )
                result[arm] = ("degenerate", None)
            else:
                if value is None:
                    raise Gate12C2DrawStabilityError(
                        "conditional S2 component has defined observations "
                        f"but no summary: {arm}/{field_name}"
                    )
                result[arm] = (
                    "defined",
                    _finite(value, context=f"{arm} {field_name}"),
                )
    return result


def _transformed_s2_component(
    endpoint: Mapping[str, Any],
    field_name: str,
) -> tuple[float | None, str]:
    summaries = _component_arm_summary(endpoint, field_name)
    left_status, left_value = summaries["N1"]
    right_status, right_value = summaries[
        "graph_unconstrained_stressor"
    ]
    if left_status != right_status:
        raise Gate12C2DrawStabilityError(
            f"S2 component {field_name!r} has one-sided degeneracy"
        )
    if left_status == "degenerate":
        return None, "degenerate"
    if left_value is None or right_value is None:
        raise Gate12C2DrawStabilityError(
            f"S2 component {field_name!r} has unexpected missingness"
        )
    if field_name in S2_MAGNITUDE_FIELDS:
        if left_value < 0.0 or right_value < 0.0:
            raise Gate12C2DrawStabilityError(
                f"S2 magnitude component {field_name!r} is negative"
            )
        epsilon = float(lab.DEFAULT_LOG_EPSILON)
        return (
            math.log(right_value + epsilon)
            - math.log(left_value + epsilon),
            "defined",
        )
    if field_name == S2_ALIGNMENT_FIELD:
        return right_value - left_value, "defined"
    raise Gate12C2DrawStabilityError(
        f"unsupported S2 component field: {field_name!r}"
    )


def _keyed_results(
    results: Sequence[Mapping[str, Any]],
    *,
    regime_id: str,
    draw_count: int,
) -> dict[int, Mapping[str, Any]]:
    keyed: dict[int, Mapping[str, Any]] = {}
    for result in results:
        if not isinstance(result, Mapping):
            raise Gate12C2DrawStabilityError(
                "each outer result must be a mapping"
            )
        if result.get("regime_id") != regime_id:
            raise Gate12C2DrawStabilityError(
                "outer result regime differs from its stability slot"
            )
        if int(result.get("inner_valid_draw_count", -1)) != draw_count:
            raise Gate12C2DrawStabilityError(
                "outer result draw count differs from its stability slot"
            )
        outer_index = int(result.get("outer_experiment_index", -1))
        if outer_index < 0 or outer_index in keyed:
            raise Gate12C2DrawStabilityError(
                "outer result indices must be unique and nonnegative"
            )
        keyed[outer_index] = result
    if not keyed:
        raise Gate12C2DrawStabilityError(
            "each regime/draw slot must contain outer results"
        )
    return keyed


def _endpoint_map(
    result: Mapping[str, Any],
    *,
    regime_id: str,
) -> dict[str, Mapping[str, Any]]:
    if regime_id == "S2_null_inflation":
        rows = result.get("endpoint_rows")
    else:
        decision = result.get("pipeline_decision")
        if not isinstance(decision, Mapping):
            raise Gate12C2DrawStabilityError(
                "S0/S1 result is missing pipeline_decision"
            )
        rows = decision.get("endpoint_rows")
    if not isinstance(rows, list) or len(rows) != 24:
        raise Gate12C2DrawStabilityError(
            "each outer result must contain exactly 24 endpoints"
        )
    keyed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise Gate12C2DrawStabilityError(
                "endpoint rows must be mappings"
            )
        endpoint_id = str(row.get("endpoint_id", ""))
        if not endpoint_id or endpoint_id in keyed:
            raise Gate12C2DrawStabilityError(
                "endpoint IDs must be nonempty and unique"
            )
        keyed[endpoint_id] = row
    return keyed


def _audit_map(
    result: Mapping[str, Any],
    *,
    regime_id: str,
) -> dict[tuple[str, str], Mapping[str, Any]]:
    rows = (
        result.get("endpoint_rows")
        if regime_id == "S2_null_inflation"
        else result.get("endpoint_receipts")
    )
    if not isinstance(rows, list) or len(rows) != 24:
        raise Gate12C2DrawStabilityError(
            "result is missing 24 endpoint audit receipts"
        )
    keyed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for endpoint in rows:
        if not isinstance(endpoint, Mapping):
            raise Gate12C2DrawStabilityError(
                "endpoint audit receipt must be a mapping"
            )
        endpoint_id = str(endpoint.get("endpoint_id", ""))
        blocks = endpoint.get("block_rows")
        if not endpoint_id or not isinstance(blocks, list):
            raise Gate12C2DrawStabilityError(
                "endpoint audit receipt is incomplete"
            )
        for block in blocks:
            if not isinstance(block, Mapping):
                raise Gate12C2DrawStabilityError(
                    "block audit receipt must be a mapping"
                )
            block_id = str(block.get("source_block_id", ""))
            audit = block.get("inner_draw_audit")
            key = (endpoint_id, block_id)
            if not block_id or key in keyed or not isinstance(audit, Mapping):
                raise Gate12C2DrawStabilityError(
                    "block audit IDs must be complete and unique"
                )
            keyed[key] = audit
    return keyed


def _compare_prefixes(
    candidate_results: Mapping[int, Mapping[str, Any]],
    reference_results: Mapping[int, Mapping[str, Any]],
    *,
    regime_id: str,
    draw_count: int,
) -> tuple[int, int]:
    matches = 0
    mismatches = 0
    for outer_index in sorted(reference_results):
        candidate_audits = _audit_map(
            candidate_results[outer_index],
            regime_id=regime_id,
        )
        reference_audits = _audit_map(
            reference_results[outer_index],
            regime_id=regime_id,
        )
        if set(candidate_audits) != set(reference_audits):
            raise Gate12C2DrawStabilityError(
                "candidate and reference block audit surfaces differ"
            )
        for key in sorted(reference_audits):
            candidate = candidate_audits[key]
            reference = reference_audits[key]
            commitments = reference.get("accepted_prefix_commitments")
            if not isinstance(commitments, Mapping):
                raise Gate12C2DrawStabilityError(
                    "reference audit is missing prefix commitments"
                )
            commitment = commitments.get(str(draw_count))
            if not isinstance(commitment, Mapping):
                raise Gate12C2DrawStabilityError(
                    "reference audit is missing the requested prefix"
                )
            candidate_hash = candidate.get("accepted_sequence_sha256")
            expected_hash = commitment.get("accepted_sequence_sha256")
            if candidate_hash == expected_hash:
                matches += 1
            else:
                mismatches += 1
    return matches, mismatches


def _decision_and_primary(
    endpoint: Mapping[str, Any],
    *,
    regime_id: str,
) -> tuple[bool, float]:
    if regime_id == "S2_null_inflation":
        decision = endpoint.get("endpoint_identified")
        primary = endpoint.get("log_stressor_to_N1_null_defect")
    else:
        decision = endpoint.get("q_directional_support")
        primary = endpoint.get("median_log_ratio")
    if not isinstance(decision, bool):
        raise Gate12C2DrawStabilityError(
            "endpoint decision must be boolean"
        )
    if primary is None:
        raise Gate12C2DrawStabilityError(
            "always-defined endpoint primary summary is missing"
        )
    return decision, _finite(
        primary,
        context="endpoint primary summary",
    )


def _compare_regime(
    candidate_results: Mapping[int, Mapping[str, Any]],
    reference_results: Mapping[int, Mapping[str, Any]],
    *,
    regime_id: str,
) -> dict[str, Any]:
    agreement_count = 0
    comparison_count = 0
    primary_shifts: list[float] = []
    component_shifts = (
        {
            field: []
            for field in S2_COMPONENT_FIELDS
        }
        if regime_id == "S2_null_inflation"
        else {}
    )
    component_degenerate = {
        field: 0 for field in component_shifts
    }
    for outer_index in sorted(reference_results):
        candidate_endpoints = _endpoint_map(
            candidate_results[outer_index],
            regime_id=regime_id,
        )
        reference_endpoints = _endpoint_map(
            reference_results[outer_index],
            regime_id=regime_id,
        )
        if set(candidate_endpoints) != set(reference_endpoints):
            raise Gate12C2DrawStabilityError(
                "candidate and reference endpoint surfaces differ"
            )
        for endpoint_id in sorted(reference_endpoints):
            candidate = candidate_endpoints[endpoint_id]
            reference = reference_endpoints[endpoint_id]
            candidate_decision, candidate_primary = _decision_and_primary(
                candidate,
                regime_id=regime_id,
            )
            reference_decision, reference_primary = _decision_and_primary(
                reference,
                regime_id=regime_id,
            )
            comparison_count += 1
            agreement_count += int(
                candidate_decision == reference_decision
            )
            primary_shifts.append(
                abs(candidate_primary - reference_primary)
            )
            if regime_id == "S2_null_inflation":
                for field_name in component_shifts:
                    candidate_value, candidate_status = (
                        _transformed_s2_component(
                            candidate,
                            field_name,
                        )
                    )
                    reference_value, reference_status = (
                        _transformed_s2_component(
                            reference,
                            field_name,
                        )
                    )
                    if candidate_status != reference_status:
                        raise Gate12C2DrawStabilityError(
                            "candidate/reference component definition "
                            f"changed for {regime_id}/{outer_index}/"
                            f"{endpoint_id}/{field_name}"
                        )
                    if candidate_status == "degenerate":
                        component_degenerate[field_name] += 1
                        continue
                    if candidate_value is None or reference_value is None:
                        raise Gate12C2DrawStabilityError(
                            "defined component comparison is unexpectedly "
                            f"missing for {field_name}"
                        )
                    component_shifts[field_name].append(
                        abs(
                            _finite(
                                candidate_value,
                                context=(
                                    f"{regime_id}/{outer_index}/"
                                    f"{endpoint_id}/{field_name} candidate"
                                ),
                            )
                            - _finite(
                                reference_value,
                                context=(
                                    f"{regime_id}/{outer_index}/"
                                    f"{endpoint_id}/{field_name} reference"
                                ),
                            )
                        )
                    )
    if comparison_count == 0:
        raise Gate12C2DrawStabilityError(
            "regime comparison contains no endpoints"
        )
    component_rows = []
    field_coverage_gate_pass = True
    for field_name in S2_COMPONENT_FIELDS:
        if regime_id != "S2_null_inflation":
            continue
        values = component_shifts[field_name]
        degenerate_count = component_degenerate[field_name]
        coverage_pass = bool(
            (
                field_name in S2_ALWAYS_DEFINED_FIELDS
                and len(values) == comparison_count
                and degenerate_count == 0
            )
            or (
                field_name in S2_CONDITIONALLY_DEFINED_FIELDS
                and len(values) == comparison_count
                and degenerate_count == 0
            )
        )
        field_coverage_gate_pass &= coverage_pass
        component_rows.append(
            {
                "field_name": field_name,
                "transform": (
                    "log_stressor_minus_log_N1"
                    if field_name in S2_MAGNITUDE_FIELDS
                    else "stressor_minus_N1"
                ),
                "maximum_absolute_shift": max(values) if values else None,
                "expected_count": comparison_count,
                "compared_count": len(values),
                "jointly_missing_count": 0,
                "candidate_only_missing_count": 0,
                "reference_only_missing_count": 0,
                "degenerate_count": degenerate_count,
                "unexpected_missing_count": 0,
                "nonfinite_count": 0,
                "coverage_gate_pass": coverage_pass,
            }
        )
    return {
        "regime_id": regime_id,
        "endpoint_decision_agreement": (
            agreement_count / comparison_count
        ),
        "endpoint_comparison_count": comparison_count,
        "maximum_absolute_primary_summary_shift": (
            max(primary_shifts) if primary_shifts else None
        ),
        "primary_summary_expected_count": comparison_count,
        "primary_summary_compared_count": len(primary_shifts),
        "primary_summary_jointly_missing_count": 0,
        "primary_summary_candidate_only_missing_count": 0,
        "primary_summary_reference_only_missing_count": 0,
        "primary_summary_degenerate_count": 0,
        "primary_summary_unexpected_missing_count": 0,
        "primary_summary_nonfinite_count": 0,
        "field_coverage_gate_pass": field_coverage_gate_pass,
        "component_stability": component_rows,
    }


def _s0_familywise_absolute_shift(
    candidate_results: Mapping[int, Mapping[str, Any]],
    reference_results: Mapping[int, Mapping[str, Any]],
) -> float:
    candidate_count = 0
    reference_count = 0
    total = len(reference_results)
    for outer_index in sorted(reference_results):
        candidate_decision = candidate_results[outer_index].get(
            "pipeline_decision"
        )
        reference_decision = reference_results[outer_index].get(
            "pipeline_decision"
        )
        if not isinstance(candidate_decision, Mapping) or not isinstance(
            reference_decision,
            Mapping,
        ):
            raise Gate12C2DrawStabilityError(
                "S0 result is missing pipeline_decision"
            )
        candidate_value = candidate_decision.get("any_endpoint_support")
        reference_value = reference_decision.get("any_endpoint_support")
        if not isinstance(candidate_value, bool) or not isinstance(
            reference_value,
            bool,
        ):
            raise Gate12C2DrawStabilityError(
                "S0 family-wise event must be boolean"
            )
        candidate_count += int(candidate_value)
        reference_count += int(reference_value)
    return abs(candidate_count / total - reference_count / total)


def _validate_result_surface(
    result_sets: Mapping[str, Mapping[int, Sequence[Mapping[str, Any]]]],
) -> dict[str, dict[int, dict[int, Mapping[str, Any]]]]:
    if set(result_sets) != set(REGIMES):
        raise Gate12C2DrawStabilityError(
            "result regimes differ from the frozen three-regime surface"
        )
    normalized: dict[str, dict[int, dict[int, Mapping[str, Any]]]] = {}
    for regime_id in REGIMES:
        by_count = result_sets[regime_id]
        if set(by_count) != set(PREFIX_COUNTS):
            raise Gate12C2DrawStabilityError(
                f"{regime_id} draw counts differ from the frozen prefixes"
            )
        normalized[regime_id] = {
            draw_count: _keyed_results(
                by_count[draw_count],
                regime_id=regime_id,
                draw_count=draw_count,
            )
            for draw_count in PREFIX_COUNTS
        }
        reference_indices = set(
            normalized[regime_id][REFERENCE_DRAW_COUNT]
        )
        if any(
            set(normalized[regime_id][count]) != reference_indices
            for count in PREFIX_COUNTS
        ):
            raise Gate12C2DrawStabilityError(
                f"{regime_id} outer experiment sets differ by draw count"
            )
    return normalized


def build_no_outcome_projection(
    result_sets: Mapping[str, Mapping[int, Sequence[Mapping[str, Any]]]],
    *,
    draw_profile_plan: Mapping[str, Any],
    execution_receipt: Mapping[str, Any],
    resource_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a strict projection containing stability deltas only."""

    normalized = _validate_result_surface(result_sets)
    try:
        resources = profile.verify_resource_evidence_chain(
            draw_profile_plan,
            execution_receipt,
            resource_receipt,
        )
    except profile.Gate12C2DrawProfileError as exc:
        raise Gate12C2DrawStabilityError(
            f"resource evidence chain failed: {exc}"
        ) from exc
    source_from_receipt = resources[
        "source_plan_payload_sha256_by_regime_and_draw_count"
    ]
    if set(source_from_receipt) != set(REGIMES):
        raise Gate12C2DrawStabilityError(
            "resource evidence changed the frozen regime surface"
        )
    source_hashes: dict[str, dict[str, str]] = {}
    for regime_id in REGIMES:
        values = source_from_receipt[regime_id]
        if set(values) != {str(value) for value in PREFIX_COUNTS}:
            raise Gate12C2DrawStabilityError(
                "source plan hash draw counts differ from the frozen prefixes"
            )
        source_hashes[regime_id] = {}
        for draw_count in PREFIX_COUNTS:
            value = values[str(draw_count)]
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in value
                )
            ):
                raise Gate12C2DrawStabilityError(
                    "source plan hash must be a verified SHA-256"
                )
            source_hashes[regime_id][str(draw_count)] = value

    candidates = []
    for draw_count in PREFIX_COUNTS:
        regime_rows = []
        prefix_matches = 0
        prefix_mismatches = 0
        for regime_id in REGIMES:
            candidate_results = normalized[regime_id][draw_count]
            reference_results = normalized[regime_id][
                REFERENCE_DRAW_COUNT
            ]
            matches, mismatches = _compare_prefixes(
                candidate_results,
                reference_results,
                regime_id=regime_id,
                draw_count=draw_count,
            )
            prefix_matches += matches
            prefix_mismatches += mismatches
            regime_rows.append(
                _compare_regime(
                    candidate_results,
                    reference_results,
                    regime_id=regime_id,
                )
            )
        comparison_count = sum(
            int(row["endpoint_comparison_count"]) for row in regime_rows
        )
        weighted_agreement = sum(
            float(row["endpoint_decision_agreement"])
            * int(row["endpoint_comparison_count"])
            for row in regime_rows
        ) / comparison_count
        primary_shifts = [
            float(row["maximum_absolute_primary_summary_shift"])
            for row in regime_rows
            if row["maximum_absolute_primary_summary_shift"] is not None
        ]
        component_shifts = [
            float(component["maximum_absolute_shift"])
            for row in regime_rows
            for component in row["component_stability"]
            if component["maximum_absolute_shift"] is not None
        ]
        s0_shift = _s0_familywise_absolute_shift(
            normalized["S0_true_null"][draw_count],
            normalized["S0_true_null"][REFERENCE_DRAW_COUNT],
        )
        prefix_pass = prefix_mismatches == 0 and prefix_matches > 0
        stability_pass = bool(
            prefix_pass
            and all(
                float(row["endpoint_decision_agreement"])
                >= MINIMUM_ENDPOINT_DECISION_AGREEMENT
                for row in regime_rows
            )
            and all(
                int(row["primary_summary_compared_count"])
                == int(row["primary_summary_expected_count"])
                and bool(row["field_coverage_gate_pass"])
                for row in regime_rows
            )
            and primary_shifts
            and max(primary_shifts) <= MAXIMUM_ABSOLUTE_SUMMARY_SHIFT
            and component_shifts
            and max(component_shifts) <= MAXIMUM_ABSOLUTE_SUMMARY_SHIFT
            and s0_shift <= MAXIMUM_ABSOLUTE_S0_FAMILYWISE_SHIFT
        )
        resource_pass = bool(
            resources["status"] == "pass"
            and draw_count in resources["eligible_draw_counts"]
        )
        candidates.append(
            {
                "draw_count": draw_count,
                "accepted_prefix_match_count": prefix_matches,
                "accepted_prefix_mismatch_count": prefix_mismatches,
                "accepted_prefix_gate_pass": prefix_pass,
                "endpoint_decision_agreement_overall": weighted_agreement,
                "endpoint_decision_comparison_count": comparison_count,
                "maximum_absolute_primary_summary_shift": (
                    max(primary_shifts) if primary_shifts else None
                ),
                "S0_family_wise_false_promotion_absolute_shift": s0_shift,
                "regimes": regime_rows,
                "stability_gate_pass": stability_pass,
                "resource_gate_pass": resource_pass,
                "selection_eligible": bool(
                    stability_pass and resource_pass
                ),
            }
        )
    eligible = [
        int(row["draw_count"])
        for row in candidates
        if row["selection_eligible"]
    ]
    payload: dict[str, Any] = {
        "schema_version": PROJECTION_SCHEMA_VERSION,
        "epistemic_status": "development_draw_stability_only",
        "surface_id": "development",
        "locked_execution_authorized": False,
        "real_held_out_execution_authorized": False,
        "N2_open": False,
        "N3_open": False,
        "public_claim": False,
        "scientific_calibration_result": None,
        "scientific_outcomes_exposed": False,
        "prefix_counts": list(PREFIX_COUNTS),
        "reference_draw_count": REFERENCE_DRAW_COUNT,
        "selection_thresholds": {
            "minimum_endpoint_decision_agreement": (
                MINIMUM_ENDPOINT_DECISION_AGREEMENT
            ),
            "maximum_absolute_primary_summary_shift": (
                MAXIMUM_ABSOLUTE_SUMMARY_SHIFT
            ),
            "maximum_absolute_S2_component_shift": (
                MAXIMUM_ABSOLUTE_SUMMARY_SHIFT
            ),
            "maximum_absolute_S0_family_wise_shift": (
                MAXIMUM_ABSOLUTE_S0_FAMILYWISE_SHIFT
            ),
        },
        "S2_component_definition": {
            "magnitude_fields": list(S2_MAGNITUDE_FIELDS),
            "magnitude_transform": "log_stressor_minus_log_N1",
            "alignment_field": S2_ALIGNMENT_FIELD,
            "alignment_transform": "stressor_minus_N1",
            "epsilon": float(lab.DEFAULT_LOG_EPSILON),
            "missing_rule": (
                "joint_missingness_is_not_agreement;always_defined_fields_"
                "require_complete_coverage;conditional_degeneracy_is_"
                "counted_explicitly;zero_comparable_field_is_ineligible;"
                "one_sided_missing_or_nonfinite_is_hard_failure"
            ),
            "aggregation_rule": (
                "maximum_absolute_candidate_minus_1023_shift_over_all_"
                "matched_outer_experiment_and_endpoint_summaries"
            ),
        },
        "source_plan_payload_sha256_by_regime_and_draw_count": (
            source_hashes
        ),
        "resource_gate": {
            "status": resources["status"],
            "eligible_draw_counts": resources["eligible_draw_counts"],
            "draw_profile_plan_payload_sha256": (
                draw_profile_plan["draw_profile_plan_payload_sha256"]
            ),
            "execution_evidence_payload_sha256": resources[
                "execution_evidence_payload_sha256"
            ],
            "resource_receipt_payload_sha256": resources[
                "resource_receipt_payload_sha256"
            ],
        },
        "candidates": candidates,
        "selected_draw_count": min(eligible) if eligible else None,
        "selection_basis_allowed": [
            "accepted_prefix_identity",
            "decision_stability",
            "absolute_summary_stability",
            "resource_feasibility",
        ],
        "selection_basis_prohibited": [
            "best_observed_FPR",
            "best_observed_power",
            "most_favorable_direction",
            "raw_S2_identification_rate",
        ],
    }
    validate_no_outcome_projection(payload, require_hash=False)
    payload["projection_payload_sha256"] = shards._sha256_bytes(
        shards._canonical_json_bytes(payload)
    )
    validate_no_outcome_projection(payload, require_hash=True)
    return payload


def validate_no_outcome_projection(
    projection: Mapping[str, Any],
    *,
    require_hash: bool = True,
    include_manifest_hash: bool = False,
) -> dict[str, Any]:
    """Reject every unknown key at every nested projection level."""

    supplied = dict(projection)
    top_keys = {
        "schema_version",
        "epistemic_status",
        "surface_id",
        "locked_execution_authorized",
        "real_held_out_execution_authorized",
        "N2_open",
        "N3_open",
        "public_claim",
        "scientific_calibration_result",
        "scientific_outcomes_exposed",
        "prefix_counts",
        "reference_draw_count",
        "selection_thresholds",
        "S2_component_definition",
        "source_plan_payload_sha256_by_regime_and_draw_count",
        "resource_gate",
        "candidates",
        "selected_draw_count",
        "selection_basis_allowed",
        "selection_basis_prohibited",
    }
    if require_hash:
        top_keys.add("projection_payload_sha256")
    if include_manifest_hash:
        top_keys.add("analysis_manifest_payload_sha256")
    _require_exact_keys(supplied, top_keys, context="stability projection")
    frozen_values = {
        "schema_version": PROJECTION_SCHEMA_VERSION,
        "epistemic_status": "development_draw_stability_only",
        "surface_id": "development",
        "locked_execution_authorized": False,
        "real_held_out_execution_authorized": False,
        "N2_open": False,
        "N3_open": False,
        "public_claim": False,
        "scientific_calibration_result": None,
        "scientific_outcomes_exposed": False,
        "prefix_counts": list(PREFIX_COUNTS),
        "reference_draw_count": REFERENCE_DRAW_COUNT,
    }
    for key, expected in frozen_values.items():
        if supplied[key] != expected:
            raise Gate12C2DrawStabilityError(
                f"stability projection changed frozen field {key!r}"
            )
    _require_exact_keys(
        supplied["selection_thresholds"],
        {
            "minimum_endpoint_decision_agreement",
            "maximum_absolute_primary_summary_shift",
            "maximum_absolute_S2_component_shift",
            "maximum_absolute_S0_family_wise_shift",
        },
        context="selection thresholds",
    )
    _require_exact_keys(
        supplied["S2_component_definition"],
        {
            "magnitude_fields",
            "magnitude_transform",
            "alignment_field",
            "alignment_transform",
            "epsilon",
            "missing_rule",
            "aggregation_rule",
        },
        context="S2 component definition",
    )
    _require_exact_keys(
        supplied["resource_gate"],
        {
            "status",
            "eligible_draw_counts",
            "draw_profile_plan_payload_sha256",
            "execution_evidence_payload_sha256",
            "resource_receipt_payload_sha256",
        },
        context="projected resource gate",
    )
    resource_gate = supplied["resource_gate"]
    if (
        resource_gate["status"] not in {"pass", "fail"}
        or any(
            int(value) not in PREFIX_COUNTS
            for value in resource_gate["eligible_draw_counts"]
        )
        or not all(
            isinstance(resource_gate[key], str)
            and len(resource_gate[key]) == 64
            and all(
                character in "0123456789abcdef"
                for character in resource_gate[key]
            )
            for key in (
                "draw_profile_plan_payload_sha256",
                "execution_evidence_payload_sha256",
                "resource_receipt_payload_sha256",
            )
        )
    ):
        raise Gate12C2DrawStabilityError(
            "projected resource gate is not exact-evidence-bound"
        )
    if set(
        supplied[
            "source_plan_payload_sha256_by_regime_and_draw_count"
        ]
    ) != set(REGIMES):
        raise Gate12C2DrawStabilityError(
            "projected source hashes changed the regime surface"
        )
    for regime_id, hashes in supplied[
        "source_plan_payload_sha256_by_regime_and_draw_count"
    ].items():
        if set(hashes) != {str(value) for value in PREFIX_COUNTS}:
            raise Gate12C2DrawStabilityError(
                f"projected source hashes changed {regime_id} draw keys"
            )
    candidates = supplied["candidates"]
    if not isinstance(candidates, list) or len(candidates) != len(
        PREFIX_COUNTS
    ):
        raise Gate12C2DrawStabilityError(
            "projection must contain exactly three draw candidates"
        )
    candidate_keys = {
        "draw_count",
        "accepted_prefix_match_count",
        "accepted_prefix_mismatch_count",
        "accepted_prefix_gate_pass",
        "endpoint_decision_agreement_overall",
        "endpoint_decision_comparison_count",
        "maximum_absolute_primary_summary_shift",
        "S0_family_wise_false_promotion_absolute_shift",
        "regimes",
        "stability_gate_pass",
        "resource_gate_pass",
        "selection_eligible",
    }
    regime_keys = {
        "regime_id",
        "endpoint_decision_agreement",
        "endpoint_comparison_count",
        "maximum_absolute_primary_summary_shift",
        "primary_summary_expected_count",
        "primary_summary_compared_count",
        "primary_summary_jointly_missing_count",
        "primary_summary_candidate_only_missing_count",
        "primary_summary_reference_only_missing_count",
        "primary_summary_degenerate_count",
        "primary_summary_unexpected_missing_count",
        "primary_summary_nonfinite_count",
        "field_coverage_gate_pass",
        "component_stability",
    }
    component_keys = {
        "field_name",
        "transform",
        "maximum_absolute_shift",
        "expected_count",
        "compared_count",
        "jointly_missing_count",
        "candidate_only_missing_count",
        "reference_only_missing_count",
        "degenerate_count",
        "unexpected_missing_count",
        "nonfinite_count",
        "coverage_gate_pass",
    }
    for candidate in candidates:
        _require_exact_keys(
            candidate,
            candidate_keys,
            context="draw candidate projection",
        )
        regimes = candidate["regimes"]
        if (
            not isinstance(regimes, list)
            or [row.get("regime_id") for row in regimes]
            != list(REGIMES)
        ):
            raise Gate12C2DrawStabilityError(
                "candidate regime rows changed the frozen order"
            )
        for regime in regimes:
            _require_exact_keys(
                regime,
                regime_keys,
                context="regime stability projection",
            )
            primary_counts = [
                int(regime[key])
                for key in (
                    "primary_summary_compared_count",
                    "primary_summary_jointly_missing_count",
                    "primary_summary_candidate_only_missing_count",
                    "primary_summary_reference_only_missing_count",
                    "primary_summary_degenerate_count",
                    "primary_summary_unexpected_missing_count",
                    "primary_summary_nonfinite_count",
                )
            ]
            if (
                any(value < 0 for value in primary_counts)
                or sum(primary_counts)
                != int(regime["primary_summary_expected_count"])
                or int(regime["primary_summary_compared_count"])
                != int(regime["primary_summary_expected_count"])
            ):
                raise Gate12C2DrawStabilityError(
                    "primary-summary coverage accounting is incomplete"
                )
            components = regime["component_stability"]
            if not isinstance(components, list):
                raise Gate12C2DrawStabilityError(
                    "component stability must be a list"
                )
            for component in components:
                _require_exact_keys(
                    component,
                    component_keys,
                    context="component stability projection",
                )
                counts = [
                    int(component[key])
                    for key in (
                        "compared_count",
                        "jointly_missing_count",
                        "candidate_only_missing_count",
                        "reference_only_missing_count",
                        "degenerate_count",
                        "unexpected_missing_count",
                        "nonfinite_count",
                    )
                ]
                if (
                    any(value < 0 for value in counts)
                    or sum(counts) != int(component["expected_count"])
                    or (
                        component["coverage_gate_pass"]
                        and int(component["compared_count"]) <= 0
                    )
                ):
                    raise Gate12C2DrawStabilityError(
                        "component coverage accounting is inconsistent"
                    )
    if require_hash:
        claimed = supplied["projection_payload_sha256"]
        unhashed = dict(supplied)
        unhashed.pop("projection_payload_sha256")
        actual = shards._sha256_bytes(
            shards._canonical_json_bytes(unhashed)
        )
        if claimed != actual:
            raise Gate12C2DrawStabilityError(
                "stability projection hash mismatch"
            )
    shards._canonical_json_bytes(supplied)
    return supplied


def load_verified_result_sets(
    output_roots: Mapping[str, Mapping[int, Path]],
) -> tuple[
    dict[str, dict[int, list[Mapping[str, Any]]]],
    dict[str, dict[int, str]],
]:
    """Load exact verified development shard sets without returning outcomes."""

    if set(output_roots) != set(REGIMES):
        raise Gate12C2DrawStabilityError(
            "output roots differ from the frozen regime surface"
        )
    result_sets: dict[
        str, dict[int, list[Mapping[str, Any]]]
    ] = {}
    plan_hashes: dict[str, dict[int, str]] = {}
    for regime_id in REGIMES:
        by_count = output_roots[regime_id]
        if set(by_count) != set(PREFIX_COUNTS):
            raise Gate12C2DrawStabilityError(
                "output roots differ from the frozen draw counts"
            )
        result_sets[regime_id] = {}
        plan_hashes[regime_id] = {}
        reference_contract: dict[str, Any] | None = None
        for draw_count in PREFIX_COUNTS:
            root = Path(by_count[draw_count]).resolve()
            plan_path = root / "plan.json"
            if not plan_path.is_file():
                raise Gate12C2DrawStabilityError(
                    f"missing development plan: {plan_path}"
                )
            try:
                import json

                plan = json.loads(plan_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise Gate12C2DrawStabilityError(
                    f"could not read development plan {plan_path}: {exc}"
                ) from exc
            verified_plan = shards._verified_plan(plan)
            if verified_plan["regime_id"] != regime_id:
                raise Gate12C2DrawStabilityError(
                    "development plan occupies the wrong regime slot"
                )
            if int(verified_plan["inner_valid_draw_count"]) != draw_count:
                raise Gate12C2DrawStabilityError(
                    "development plan occupies the wrong draw-count slot"
                )
            shards.verify_development_shard_index(
                verified_plan,
                output_dir=root,
            )
            contract = dict(verified_plan)
            contract.pop("plan_payload_sha256")
            contract.pop("inner_valid_draw_count")
            contract["resolved_max_draw_attempts"] = None
            contract.pop("max_draw_attempts")
            if reference_contract is None:
                reference_contract = contract
            elif contract != reference_contract:
                raise Gate12C2DrawStabilityError(
                    "draw-count plans differ outside admitted count fields"
                )
            results = []
            for outer_index in verified_plan["outer_experiment_indices"]:
                shard_path = shards._shard_path(root, int(outer_index))
                payload = shards._read_shard(shard_path)
                shards._verify_result_against_plan(
                    verified_plan,
                    payload["result"],
                    outer_experiment_index=int(outer_index),
                )
                results.append(payload["result"])
            result_sets[regime_id][draw_count] = results
            plan_hashes[regime_id][draw_count] = verified_plan[
                "plan_payload_sha256"
            ]
    return result_sets, plan_hashes


def build_analysis_manifest(
    output_roots: Mapping[str, Mapping[int, Path]],
    *,
    draw_profile_plan_path: Path,
    execution_receipt_path: Path,
    resource_receipt_path: Path,
) -> dict[str, Any]:
    """Build a read-only, path-bound analysis manifest."""

    if set(output_roots) != set(REGIMES):
        raise Gate12C2DrawStabilityError(
            "analysis roots differ from the frozen regime surface"
        )
    normalized_roots = {}
    for regime_id in REGIMES:
        by_count = output_roots[regime_id]
        if set(by_count) != set(PREFIX_COUNTS):
            raise Gate12C2DrawStabilityError(
                "analysis roots differ from the frozen draw counts"
            )
        normalized_roots[regime_id] = {
            str(draw_count): Path(by_count[draw_count]).resolve().as_posix()
            for draw_count in PREFIX_COUNTS
        }
    evidence_paths = {
        "draw_profile_plan": Path(draw_profile_plan_path).resolve(),
        "execution_receipt": Path(execution_receipt_path).resolve(),
        "resource_receipt": Path(resource_receipt_path).resolve(),
    }
    try:
        import json

        draw_profile_plan = json.loads(
            evidence_paths["draw_profile_plan"].read_text(
                encoding="utf-8"
            )
        )
        execution_receipt = json.loads(
            evidence_paths["execution_receipt"].read_text(
                encoding="utf-8"
            )
        )
        resource_receipt = json.loads(
            evidence_paths["resource_receipt"].read_text(
                encoding="utf-8"
            )
        )
        resources = profile.verify_resource_evidence_chain(
            draw_profile_plan,
            execution_receipt,
            resource_receipt,
        )
    except Exception as exc:
        raise Gate12C2DrawStabilityError(
            f"analysis resource evidence is invalid: {exc}"
        ) from exc
    resource_evidence = {
        "draw_profile_plan_path": evidence_paths[
            "draw_profile_plan"
        ].as_posix(),
        "draw_profile_plan_file_sha256": _sha256_file(
            evidence_paths["draw_profile_plan"]
        ),
        "draw_profile_plan_payload_sha256": draw_profile_plan[
            "draw_profile_plan_payload_sha256"
        ],
        "execution_receipt_path": evidence_paths[
            "execution_receipt"
        ].as_posix(),
        "execution_receipt_file_sha256": _sha256_file(
            evidence_paths["execution_receipt"]
        ),
        "execution_receipt_payload_sha256": execution_receipt[
            "execution_receipt_payload_sha256"
        ],
        "resource_receipt_path": evidence_paths[
            "resource_receipt"
        ].as_posix(),
        "resource_receipt_file_sha256": _sha256_file(
            evidence_paths["resource_receipt"]
        ),
        "resource_receipt_payload_sha256": resources[
            "resource_receipt_payload_sha256"
        ],
    }
    payload: dict[str, Any] = {
        "schema_version": ANALYSIS_MANIFEST_SCHEMA_VERSION,
        "epistemic_status": "development_draw_stability_analysis_only",
        "surface_id": "development",
        "read_only_analysis": True,
        "development_execution_authorized": False,
        "locked_execution_authorized": False,
        "real_held_out_execution_authorized": False,
        "N2_open": False,
        "N3_open": False,
        "public_claim": False,
        "scientific_outcomes_may_be_emitted": False,
        "output_roots": normalized_roots,
        "resource_evidence": resource_evidence,
        "implementation_sha256": _implementation_hashes(),
    }
    payload["analysis_manifest_payload_sha256"] = shards._sha256_bytes(
        shards._canonical_json_bytes(payload)
    )
    return payload


def verify_analysis_manifest(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild the complete manifest so rehashed permissions cannot open."""

    if not isinstance(manifest, Mapping):
        raise Gate12C2DrawStabilityError(
            "analysis manifest must be a mapping"
        )
    supplied = dict(manifest)
    _require_exact_keys(
        supplied,
        {
            "schema_version",
            "epistemic_status",
            "surface_id",
            "read_only_analysis",
            "development_execution_authorized",
            "locked_execution_authorized",
            "real_held_out_execution_authorized",
            "N2_open",
            "N3_open",
            "public_claim",
            "scientific_outcomes_may_be_emitted",
            "output_roots",
            "resource_evidence",
            "implementation_sha256",
            "analysis_manifest_payload_sha256",
        },
        context="draw stability analysis manifest",
    )
    claimed = supplied["analysis_manifest_payload_sha256"]
    unhashed = dict(supplied)
    unhashed.pop("analysis_manifest_payload_sha256")
    if claimed != shards._sha256_bytes(
        shards._canonical_json_bytes(unhashed)
    ):
        raise Gate12C2DrawStabilityError(
            "draw stability analysis manifest hash mismatch"
        )
    roots = supplied["output_roots"]
    if not isinstance(roots, Mapping) or set(roots) != set(REGIMES):
        raise Gate12C2DrawStabilityError(
            "analysis manifest changed the regime surface"
        )
    resource_evidence = supplied["resource_evidence"]
    if not isinstance(resource_evidence, Mapping):
        raise Gate12C2DrawStabilityError(
            "analysis manifest resource evidence must be a mapping"
        )
    _require_exact_keys(
        resource_evidence,
        {
            "draw_profile_plan_path",
            "draw_profile_plan_file_sha256",
            "draw_profile_plan_payload_sha256",
            "execution_receipt_path",
            "execution_receipt_file_sha256",
            "execution_receipt_payload_sha256",
            "resource_receipt_path",
            "resource_receipt_file_sha256",
            "resource_receipt_payload_sha256",
        },
        context="analysis manifest resource evidence",
    )
    for key in (
        "draw_profile_plan_file_sha256",
        "draw_profile_plan_payload_sha256",
        "execution_receipt_file_sha256",
        "execution_receipt_payload_sha256",
        "resource_receipt_file_sha256",
        "resource_receipt_payload_sha256",
    ):
        value = resource_evidence[key]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(
                character not in "0123456789abcdef"
                for character in value
            )
        ):
            raise Gate12C2DrawStabilityError(
                f"analysis manifest has an invalid evidence digest: {key}"
            )
    typed_roots: dict[str, dict[int, Path]] = {}
    for regime_id in REGIMES:
        by_count = roots[regime_id]
        if not isinstance(by_count, Mapping) or set(by_count) != {
            str(value) for value in PREFIX_COUNTS
        }:
            raise Gate12C2DrawStabilityError(
                "analysis manifest changed the draw-count surface"
            )
        typed_roots[regime_id] = {
            draw_count: Path(str(by_count[str(draw_count)]))
            for draw_count in PREFIX_COUNTS
        }
    expected = build_analysis_manifest(
        typed_roots,
        draw_profile_plan_path=Path(
            supplied["resource_evidence"]["draw_profile_plan_path"]
        ),
        execution_receipt_path=Path(
            supplied["resource_evidence"]["execution_receipt_path"]
        ),
        resource_receipt_path=Path(
            supplied["resource_evidence"]["resource_receipt_path"]
        ),
    )
    if supplied != expected:
        raise Gate12C2DrawStabilityError(
            "analysis manifest differs from the complete builder contract"
        )
    return expected


def analyze_verified_directories(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Return only the strict no-outcome projection for an exact manifest."""

    verified = verify_analysis_manifest(manifest)
    roots = {
        regime_id: {
            draw_count: Path(
                verified["output_roots"][regime_id][str(draw_count)]
            )
            for draw_count in PREFIX_COUNTS
        }
        for regime_id in REGIMES
    }
    results, plan_hashes = load_verified_result_sets(roots)
    evidence = verified["resource_evidence"]
    try:
        import json

        draw_profile_plan = json.loads(
            Path(evidence["draw_profile_plan_path"]).read_text(
                encoding="utf-8"
            )
        )
        execution_receipt = json.loads(
            Path(evidence["execution_receipt_path"]).read_text(
                encoding="utf-8"
            )
        )
        resource_receipt = json.loads(
            Path(evidence["resource_receipt_path"]).read_text(
                encoding="utf-8"
            )
        )
        resources = profile.verify_resource_evidence_chain(
            draw_profile_plan,
            execution_receipt,
            resource_receipt,
        )
    except Exception as exc:
        raise Gate12C2DrawStabilityError(
            f"analysis resource evidence failed at execution: {exc}"
        ) from exc
    expected_plan_hashes = {
        regime_id: {
            draw_count: resources[
                "source_plan_payload_sha256_by_regime_and_draw_count"
            ][regime_id][str(draw_count)]
            for draw_count in PREFIX_COUNTS
        }
        for regime_id in REGIMES
    }
    if plan_hashes != expected_plan_hashes:
        raise Gate12C2DrawStabilityError(
            "verified result roots do not match the resource evidence chain"
        )
    projection = build_no_outcome_projection(
        results,
        draw_profile_plan=draw_profile_plan,
        execution_receipt=execution_receipt,
        resource_receipt=resource_receipt,
    )
    projection["analysis_manifest_payload_sha256"] = verified[
        "analysis_manifest_payload_sha256"
    ]
    unhashed = dict(projection)
    unhashed.pop("projection_payload_sha256")
    projection["projection_payload_sha256"] = shards._sha256_bytes(
        shards._canonical_json_bytes(unhashed)
    )
    return validate_no_outcome_projection(
        projection,
        require_hash=True,
        include_manifest_hash=True,
    )
