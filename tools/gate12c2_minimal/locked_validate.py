"""Independent component-level validator for the locked synthetic suite.

This module intentionally does not import ``experiment``, ``locked_calibration``
or ``locked_run``. It reconstructs endpoint and gate decisions from the compact
component rows by a separate implementation.
"""

from __future__ import annotations

import argparse
import math
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .io import canonical_json_bytes, load_json, sha256_file


CONFIGS = ("S0", "S1_LOW", "S1_PRIMARY", "S1_HIGH", "S2")
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
Z95 = 1.6448536269514722


class IndependentLockedValidationError(ValueError):
    """Raised when independent reconstruction fails."""


def _environment() -> dict[str, str]:
    import numpy

    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": numpy.__version__,
        "platform_system": platform.system(),
        "machine": platform.machine(),
    }


def _arm(
    values: object,
    *,
    numerical_tolerance: float,
    spectral_gap_tolerance: float,
    rank: int,
    q: int,
) -> dict[str, Any]:
    if not isinstance(values, list) or len(values) != len(FIELDS):
        raise IndependentLockedValidationError("diagnostic vector width differs")
    result = dict(zip(FIELDS, values))
    magnitudes = [float(result[name]) for name in ("a", "u", "v", "x", "y")]
    if any(not math.isfinite(value) or value < 0.0 for value in magnitudes):
        raise IndependentLockedValidationError("diagnostic magnitude differs")
    if result["c"] is not None:
        c = float(result["c"])
        rhs = (
            float(result["x"]) ** 2
            + float(result["y"]) ** 2
            - 2.0 * float(result["x"]) * float(result["y"]) * c
        )
        if abs(float(result["a"]) ** 2 - rhs) > numerical_tolerance * max(
            1.0, float(result["a"]) ** 2, abs(rhs)
        ):
            raise IndependentLockedValidationError("residual identity differs")
        if not math.isfinite(c) or abs(c) > 1.0:
            raise IndependentLockedValidationError("diagnostic alignment differs")
    for numerator, denominator, name in (
        ("x", "u", "p_left"),
        ("y", "v", "p_right"),
    ):
        expected = (
            None
            if float(result[denominator]) <= 1e-12
            else float(result[numerator]) / float(result[denominator])
        )
        actual = result[name]
        if expected is None:
            if actual is not None:
                raise IndependentLockedValidationError("propagation ratio differs")
        elif actual is None or not math.isclose(
            float(actual), expected, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise IndependentLockedValidationError("propagation ratio differs")
    gaps = [
        float(result["relative_gap_left"]),
        float(result["relative_gap_right"]),
    ]
    if any(not math.isfinite(gap) or gap < -1e-15 for gap in gaps):
        raise IndependentLockedValidationError("spectral gap differs")
    expected_eligible = all(gap > spectral_gap_tolerance for gap in gaps)
    if not isinstance(result["eligible"], bool) or result["eligible"] != expected_eligible:
        raise IndependentLockedValidationError("eligibility flag differs")
    if not isinstance(result["numerical_pass"], bool):
        raise IndependentLockedValidationError("numerical flag differs")
    for name in ("product_singular_values_left", "product_singular_values_right"):
        spectrum = result[name]
        if not isinstance(spectrum, list) or len(spectrum) != rank:
            raise IndependentLockedValidationError("product spectrum rank differs")
        numeric = [float(item) for item in spectrum]
        if (
            any(not math.isfinite(item) or item < 0.0 for item in numeric)
            or numeric != sorted(numeric, reverse=True)
        ):
            raise IndependentLockedValidationError("product spectrum differs")
    errors = [
        float(result["matrix_identity_error"]),
        float(result["squared_identity_error"]),
    ]
    if any(not math.isfinite(error) or error < 0.0 for error in errors):
        raise IndependentLockedValidationError("identity error differs")
    expected_numerical = bool(
        errors[0]
        <= numerical_tolerance
        * max(1.0, float(result["a"]), float(result["x"]), float(result["y"]))
        and errors[1]
        <= numerical_tolerance
        * max(
            1.0,
            float(result["a"]) ** 2,
            float(result["x"]) ** 2 + float(result["y"]) ** 2,
        )
    )
    if result["numerical_pass"] != expected_numerical or not 1 <= q < rank:
        raise IndependentLockedValidationError("numerical diagnostic differs")
    return result


def _spectra(value: object, rank: int) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != 3:
        raise IndependentLockedValidationError("edge spectra differ")
    result: list[list[float]] = []
    for spectrum in value:
        if not isinstance(spectrum, list) or len(spectrum) != rank:
            raise IndependentLockedValidationError("edge spectrum rank differs")
        row = [float(item) for item in spectrum]
        if (
            any(not math.isfinite(item) or item < 0.0 for item in row)
            or row != sorted(row, reverse=True)
        ):
            raise IndependentLockedValidationError("edge spectrum order differs")
        result.append(row)
    return result


def _tail_probability(values: Sequence[float], *, negative: bool, zero: float) -> float:
    votes = [value for value in values if abs(value) > zero]
    if not votes:
        return 1.0
    hits = sum(value < 0.0 if negative else value > 0.0 for value in votes)
    return sum(math.comb(len(votes), k) for k in range(hits, len(votes) + 1)) / (
        2 ** len(votes)
    )


def _endpoint(
    shard: Mapping[str, Any], spec: Mapping[str, Any], q: int, draws: int
) -> dict[str, Any]:
    numerical_tolerance = float(spec["numerical_tolerance"])
    spectral_gap_tolerance = float(spec["spectral_gap_tolerance"])
    epsilon = float(spec["epsilon"])
    zero = float(spec["criteria"]["zero_tolerance"])
    observed: dict[tuple[int, int], dict[str, Any]] = {}
    null: dict[tuple[int, int, int], dict[str, Any]] = {}
    stress: dict[tuple[int, int, int], tuple[dict[str, Any], bool]] = {}
    for row in shard["observed_rows"]:
        if not isinstance(row, list) or len(row) != 7:
            raise IndependentLockedValidationError("observed row width differs")
        row_q = int(row[1])
        rank = int(shard["case"]["local_rank"])
        seen = _arm(
            row[2],
            numerical_tolerance=numerical_tolerance,
            spectral_gap_tolerance=spectral_gap_tolerance,
            rank=rank,
            q=row_q,
        )
        gauge = _arm(
            row[3],
            numerical_tolerance=numerical_tolerance,
            spectral_gap_tolerance=spectral_gap_tolerance,
            rank=rank,
            q=row_q,
        )
        observed[(int(row[0]), row_q)] = seen
        _spectra(row[4], rank)
        gauge_defect = abs(float(seen["a"]) - float(gauge["a"]))
        gauge_component = max(
            abs(float(seen[name]) - float(gauge[name]))
            for name in ("u", "v", "x", "y")
        )
        for name in ("c", "p_left", "p_right"):
            left, right = seen[name], gauge[name]
            difference = (
                0.0
                if left is None and right is None
                else math.inf
                if left is None or right is None
                else abs(float(left) - float(right))
            )
            gauge_component = max(gauge_component, difference)
        if not math.isclose(gauge_defect, float(row[5]), rel_tol=1e-12, abs_tol=1e-15):
            raise IndependentLockedValidationError("gauge defect evidence differs")
        if not math.isclose(gauge_component, float(row[6]), rel_tol=1e-12, abs_tol=1e-15):
            raise IndependentLockedValidationError("gauge component evidence differs")
    for row in shard["null_rows"]:
        if not isinstance(row, list) or len(row) != 5:
            raise IndependentLockedValidationError("null row width differs")
        if int(row[1]) < draws:
            null[(int(row[0]), int(row[1]), int(row[2]))] = _arm(
                row[3],
                numerical_tolerance=numerical_tolerance,
                spectral_gap_tolerance=spectral_gap_tolerance,
                rank=int(shard["case"]["local_rank"]),
                q=int(row[2]),
            )
        _spectra(row[4], int(shard["case"]["local_rank"]))
    for row in shard["stressor_rows"]:
        if not isinstance(row, list) or len(row) != 6:
            raise IndependentLockedValidationError("stressor row width differs")
        if int(row[1]) < draws:
            stress[(int(row[0]), int(row[1]), int(row[2]))] = (
                _arm(
                    row[3],
                    numerical_tolerance=numerical_tolerance,
                    spectral_gap_tolerance=spectral_gap_tolerance,
                    rank=int(shard["case"]["local_rank"]),
                    q=int(row[2]),
                ),
                bool(row[5]),
            )
        after = _spectra(row[4], int(shard["case"]["local_rank"]))
        matching = next(
            candidate[4]
            for candidate in shard["null_rows"]
            if candidate[0:3] == row[0:3]
        )
        before = _spectra(matching, int(shard["case"]["local_rank"]))
        if not np.allclose(before, after, rtol=1e-12, atol=1e-12):
            raise IndependentLockedValidationError("S2 spectrum preservation fails")

    block_values: list[float] = []
    block_channels: list[float] = []
    coverage = True
    for graph in range(int(spec["cohort_size"])):
        draw_values: list[float] = []
        channel_values: list[bool] = []
        for draw in range(draws):
            base = null[(graph, draw, q)]
            coverage = coverage and bool(base["eligible"] and base["numerical_pass"])
            if shard["config_id"] == "S2":
                changed, channel = stress[(graph, draw, q)]
                coverage = coverage and bool(
                    changed["eligible"] and changed["numerical_pass"]
                )
                recomputed_channel = bool(
                    float(changed["x"]) > float(base["x"]) + zero
                    or float(changed["y"]) > float(base["y"]) + zero
                    or (
                        changed["c"] is not None
                        and base["c"] is not None
                        and float(changed["c"]) < float(base["c"]) - zero
                    )
                )
                if channel != recomputed_channel:
                    raise IndependentLockedValidationError("S2 channel flag differs")
                draw_values.append(
                    math.log(float(changed["a"]) + epsilon)
                    - math.log(float(base["a"]) + epsilon)
                )
                channel_values.append(channel)
            else:
                seen = observed[(graph, q)]
                coverage = coverage and bool(seen["eligible"] and seen["numerical_pass"])
                draw_values.append(
                    math.log(float(seen["a"]) + epsilon)
                    - math.log(float(base["a"]) + epsilon)
                )
        block_values.append(float(np.median(draw_values)))
        if channel_values:
            block_channels.append(float(np.mean(channel_values)))
    return {
        "case_id": shard["case"]["case_id"],
        "q": q,
        "median_effect": float(np.median(block_values)),
        "directional_sign_p": _tail_probability(
            block_values, negative=shard["config_id"] != "S2", zero=zero
        ),
        "channel_fraction": (
            None if not block_channels else float(np.mean(block_channels))
        ),
        "coverage_complete": bool(coverage),
    }


def _dataset(
    shards: Sequence[Mapping[str, Any]], spec: Mapping[str, Any], draws: int
) -> dict[str, Any]:
    config = str(shards[0]["config_id"])
    endpoints = [
        _endpoint(shard, spec, int(q), draws)
        for shard in shards
        for q in spec["q_values"]
    ]
    criteria = spec["criteria"]
    if config != "S2":
        order = sorted(
            range(len(endpoints)),
            key=lambda i: (
                endpoints[i]["directional_sign_p"],
                endpoints[i]["case_id"],
                endpoints[i]["q"],
            ),
        )
        running = 0.0
        for position, index in enumerate(order):
            running = max(
                running,
                min(
                    1.0,
                    endpoints[index]["directional_sign_p"]
                    * (len(endpoints) - position),
                ),
            )
            endpoints[index]["supported"] = bool(
                endpoints[index]["coverage_complete"]
                and endpoints[index]["median_effect"]
                < -float(criteria["zero_tolerance"])
                and running < float(criteria["holm_alpha"])
            )
        cases: dict[str, list[bool]] = defaultdict(list)
        for endpoint in endpoints:
            cases[endpoint["case_id"]].append(endpoint["supported"])
        case_hits = sum(len(rows) == 2 and all(rows) for rows in cases.values())
        return {
            "config_id": config,
            "dataset_index": int(shards[0]["dataset_index"]),
            "any_endpoint_support": any(row["supported"] for row in endpoints),
            "supported_endpoint_count": sum(row["supported"] for row in endpoints),
            "supported_case_count": int(case_hits),
            "promotion": bool(case_hits >= int(criteria["promotion_min_case_count"])),
            "identification_success": None,
            "endpoints": endpoints,
        }
    for endpoint in endpoints:
        endpoint["supported"] = bool(
            endpoint["coverage_complete"]
            and endpoint["median_effect"] > 0.0
            and endpoint["channel_fraction"]
            >= float(criteria["s2_min_channel_fraction"])
        )
    hits = sum(row["supported"] for row in endpoints)
    family_median = float(np.median([row["median_effect"] for row in endpoints]))
    return {
        "config_id": config,
        "dataset_index": int(shards[0]["dataset_index"]),
        "any_endpoint_support": None,
        "supported_endpoint_count": int(hits),
        "supported_case_count": None,
        "promotion": None,
        "identification_success": bool(
            family_median > float(criteria["s2_min_log_inflation"])
            and
            hits / len(endpoints) >= float(criteria["s2_min_endpoint_fraction"])
        ),
        "family_median_effect": family_median,
        "endpoints": endpoints,
    }


def _wilson(hits: int, total: int) -> list[float]:
    p = hits / total
    z2 = Z95 * Z95
    denominator = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denominator
    radius = Z95 * math.sqrt(
        p * (1.0 - p) / total + z2 / (4.0 * total * total)
    ) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _reconstruct_analysis(
    spec: Mapping[str, Any], spec_hash: str, shards: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for shard in shards:
        grouped[(shard["config_id"], int(shard["dataset_index"]))].append(shard)
    full = [
        _dataset(grouped[key], spec, int(spec["inner_draws"]))
        for key in sorted(grouped)
    ]
    prefix = {
        key: _dataset(grouped[key], spec, int(spec["stability_prefix_draws"]))
        for key in sorted(grouped)
    }
    summaries: dict[str, Any] = {}
    for config in CONFIGS:
        rows = [row for row in full if row["config_id"] == config]
        total = len(rows)
        if config == "S0":
            any_hits = sum(row["any_endpoint_support"] for row in rows)
            promotion_hits = sum(row["promotion"] for row in rows)
            summaries[config] = {
                "dataset_count": total,
                "any_endpoint_rate": any_hits / total,
                "any_endpoint_wilson_95": _wilson(any_hits, total),
                "promotion_rate": promotion_hits / total,
                "promotion_wilson_95": _wilson(promotion_hits, total),
            }
        elif config.startswith("S1_"):
            hits = sum(row["promotion"] for row in rows)
            summaries[config] = {
                "dataset_count": total,
                "promotion_power": hits / total,
                "promotion_wilson_95": _wilson(hits, total),
            }
        else:
            hits = sum(row["identification_success"] for row in rows)
            summaries[config] = {
                "dataset_count": total,
                "identification_rate": hits / total,
                "identification_wilson_95": _wilson(hits, total),
            }

    shifts: list[float] = []
    agreements: list[bool] = []
    for row in full:
        if row["config_id"] not in {"S0", "S1_PRIMARY", "S2"}:
            continue
        shorter = prefix[(row["config_id"], row["dataset_index"])]
        left = {(e["case_id"], e["q"]): e for e in row["endpoints"]}
        right = {(e["case_id"], e["q"]): e for e in shorter["endpoints"]}
        for key in left:
            shifts.append(abs(left[key]["median_effect"] - right[key]["median_effect"]))
            agreements.append(bool(left[key]["supported"]) == bool(right[key]["supported"]))
    stability = {
        "comparison_count": len(shifts),
        "decision_agreement": float(np.mean(agreements)),
        "p95_effect_shift": float(np.quantile(shifts, 0.95)),
        "max_effect_shift": float(max(shifts)),
    }

    observed: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    null: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for shard in shards:
        if shard["config_id"] != "S0":
            continue
        case_id = shard["case"]["case_id"]
        for row in shard["observed_rows"]:
            if int(row[1]) == 1:
                for edge, spectrum in enumerate(row[4]):
                    for index, value in enumerate(spectrum):
                        observed[(case_id, edge, index)].append(float(value))
        for row in shard["null_rows"]:
            if int(row[2]) == 1:
                for edge, spectrum in enumerate(row[4]):
                    for index, value in enumerate(spectrum):
                        null[(case_id, edge, index)].append(float(value))
    quantiles = np.linspace(0.1, 0.9, 9)
    differences = [
        float(
            np.max(
                np.abs(
                    np.quantile(observed[key], quantiles)
                    - np.quantile(null[key], quantiles)
                )
            )
        )
        for key in sorted(observed)
    ]
    nuisance = {
        "group_count": len(differences),
        "max_quantile_difference": float(max(differences)),
        "median_quantile_difference": float(np.median(differences)),
    }
    recomputed_gauge_defects = [
        float(row[5])
        for shard in shards
        for row in shard["observed_rows"]
    ]
    recomputed_gauge_components = [
        float(row[6])
        for shard in shards
        for row in shard["observed_rows"]
    ]
    recomputed_stressor_spectrum_errors = [
        float(
            np.max(
                np.abs(
                    np.asarray(null_row[4], dtype=np.float64)
                    - np.asarray(stressor_row[4], dtype=np.float64)
                )
            )
        )
        for shard in shards
        if shard["config_id"] == "S2"
        for stressor_row in shard["stressor_rows"]
        for null_row in shard["null_rows"]
        if null_row[0:3] == stressor_row[0:3]
    ]
    controls = {
        "all_observed_unchanged": all(row["observed_arm_unchanged"] is True for row in shards),
        "max_n1_realizability_error": max(float(row["n1_realizability_max_error"]) for row in shards),
        "max_gauge_defect_error": max(recomputed_gauge_defects),
        "max_gauge_component_error": max(recomputed_gauge_components),
        "max_stressor_edge_spectrum_error": max(recomputed_stressor_spectrum_errors),
    }
    c = spec["criteria"]
    gates = {
        "mechanical_controls": bool(
            controls["all_observed_unchanged"]
            and controls["max_n1_realizability_error"] <= 1e-10
            and controls["max_gauge_defect_error"] <= c["gauge_max_error"]
            and controls["max_gauge_component_error"] <= c["gauge_max_error"]
            and controls["max_stressor_edge_spectrum_error"] <= 1e-10
        ),
        "s0_familywise_safety": bool(
            summaries["S0"]["any_endpoint_rate"] <= c["s0_max_point_rate"]
            and summaries["S0"]["any_endpoint_wilson_95"][1]
            <= c["s0_max_wilson_upper"]
        ),
        "s0_promotion_safety": bool(
            summaries["S0"]["promotion_rate"] <= c["s0_max_point_rate"]
            and summaries["S0"]["promotion_wilson_95"][1]
            <= c["s0_max_wilson_upper"]
        ),
        "s1_primary_power": bool(
            summaries["S1_PRIMARY"]["promotion_power"] >= c["s1_primary_min_power"]
            and summaries["S1_PRIMARY"]["promotion_wilson_95"][0]
            >= c["s1_primary_min_wilson_lower"]
        ),
        "s2_identification": bool(
            summaries["S2"]["identification_rate"] >= c["s2_min_success_rate"]
            and summaries["S2"]["identification_wilson_95"][0]
            >= c["s2_min_wilson_lower"]
        ),
        "nuisance_fidelity": bool(
            nuisance["max_quantile_difference"] <= c["nuisance_max_quantile_difference"]
            and nuisance["median_quantile_difference"]
            <= c["nuisance_median_quantile_difference"]
        ),
        "inner_draw_stability": bool(
            stability["decision_agreement"] >= c["stability_min_decision_agreement"]
            and stability["p95_effect_shift"] <= c["stability_max_p95_effect_shift"]
        ),
    }
    passed = all(gates.values())
    return {
        "schema_version": "gate12c2_locked_calibration_analysis_v0.1",
        "study_id": spec["study_id"],
        "study_sha256": spec_hash,
        "attempt_id": spec["attempt_id"],
        "epistemic_status": "locked_synthetic_calibration",
        "decision": "LOCKED_PASS" if passed else "RETIRE_OR_DEMOTE",
        "locked_pass": passed,
        "real_held_out_authorized": False,
        "scientific_claim_authorized": False,
        "dataset_is_inference_unit": True,
        "shard_count": len(shards),
        "gate_results": gates,
        "controls": controls,
        "nuisance": nuisance,
        "stability": stability,
        "configurations": summaries,
        "dataset_decisions": [
            {key: value for key, value in row.items() if key != "endpoints"}
            for row in full
        ],
    }


def validate_locked_run(spec_path: Path, output: Path) -> dict[str, Any]:
    spec = load_json(spec_path)
    if not isinstance(spec, dict):
        raise IndependentLockedValidationError("locked spec is not an object")
    if (
        spec.get("schema_version") != "gate12c2_locked_calibration_spec_v0.1"
        or spec.get("epistemic_status") != "locked_synthetic_calibration_once"
        or spec.get("q_values") != [1, 2]
        or spec.get("stressor_trials") != 1
        or not isinstance(spec.get("cases"), list)
        or len(spec["cases"]) != 12
    ):
        raise IndependentLockedValidationError("locked spec surface differs")
    spec_hash = sha256_file(spec_path)
    output = output.resolve()
    state = load_json(output / "state.json")
    manifest = load_json(output / "manifest.json")
    if (
        not isinstance(state, dict)
        or state.get("state") != "COMPLETE"
        or state.get("study_id") != spec.get("study_id")
        or state.get("study_sha256") != spec_hash
        or state.get("attempt_id") != spec.get("attempt_id")
        or state.get("completed_shard_count")
        != spec.get("resource_cap", {}).get("max_dataset_shards")
    ):
        raise IndependentLockedValidationError("locked run is not complete")
    elapsed = float(state.get("elapsed_wall_seconds", math.inf))
    if not math.isfinite(elapsed) or not 0.0 <= elapsed <= float(
        spec["resource_cap"]["max_wall_seconds"]
    ):
        raise IndependentLockedValidationError("locked wall-time accounting differs")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != (
        "gate12c2_locked_calibration_manifest_v0.1"
    ):
        raise IndependentLockedValidationError("locked manifest differs")
    if (
        manifest.get("study_id") != spec.get("study_id")
        or manifest.get("study_sha256") != spec_hash
        or manifest.get("attempt_id") != spec.get("attempt_id")
        or manifest.get("dataset_is_inference_unit") is not True
        or manifest.get("stressor_selection_count") != 1
    ):
        raise IndependentLockedValidationError("locked manifest study differs")
    if manifest.get("environment") != _environment() or spec.get("environment") != _environment():
        raise IndependentLockedValidationError("locked environment differs")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise IndependentLockedValidationError("locked file map is absent")
    implementation_names = {
        "io.py",
        "metrics.py",
        "generators.py",
        "experiment.py",
        "locked_calibration.py",
        "locked_run.py",
        "locked_validate.py",
    }
    frozen_implementation = spec.get("implementation_sha256")
    if (
        not isinstance(frozen_implementation, dict)
        or set(frozen_implementation) != implementation_names
        or manifest.get("implementation_sha256") != frozen_implementation
    ):
        raise IndependentLockedValidationError("locked implementation map differs")
    package = Path(__file__).resolve().parent
    for name in sorted(implementation_names):
        if sha256_file(package / name) != frozen_implementation[name]:
            raise IndependentLockedValidationError(
                f"locked implementation hash differs: {name}"
            )
    for relative, digest in files.items():
        path = output / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise IndependentLockedValidationError(f"locked file hash differs: {relative}")

    expected = [
        (config, case["case_id"], dataset)
        for config in CONFIGS
        for dataset in range(int(spec["dataset_count"]))
        for case in spec["cases"]
    ]
    expected_files = {
        "study.json",
        "analysis.json",
        *(
            f"shards/{config}/{case_id}/d{dataset:04d}.json"
            for config, case_id, dataset in expected
        ),
    }
    if set(files) != expected_files or manifest.get("shard_count") != len(expected):
        raise IndependentLockedValidationError("locked manifest file surface differs")
    if sha256_file(output / "study.json") != spec_hash:
        raise IndependentLockedValidationError("stored locked study differs")
    expected_cases = {case["case_id"]: case for case in spec["cases"]}
    shards: list[Mapping[str, Any]] = []
    for config, case_id, dataset in expected:
        relative = f"shards/{config}/{case_id}/d{dataset:04d}.json"
        shard = load_json(output / relative)
        if (
            not isinstance(shard, dict)
            or shard.get("schema_version") != "gate12c2_locked_dataset_shard_v0.1"
            or shard.get("config_id") != config
            or shard.get("dataset_index") != dataset
            or shard.get("study_id") != spec.get("study_id")
            or shard.get("study_sha256") != spec_hash
            or shard.get("attempt_id") != spec.get("attempt_id")
            or shard.get("case") != expected_cases[case_id]
            or shard.get("arm_fields") != list(FIELDS)
        ):
            raise IndependentLockedValidationError(f"locked shard identity differs: {relative}")
        observed_rows = shard.get("observed_rows")
        null_rows = shard.get("null_rows")
        stressor_rows = shard.get("stressor_rows")
        cohort = int(spec["cohort_size"])
        draws = int(spec["inner_draws"])
        if (
            not isinstance(observed_rows, list)
            or len(observed_rows) != cohort * 2
            or not isinstance(null_rows, list)
            or len(null_rows) != cohort * draws * 2
            or not isinstance(stressor_rows, list)
            or len(stressor_rows) != (cohort * draws * 2 if config == "S2" else 0)
        ):
            raise IndependentLockedValidationError(
                f"locked shard component surface differs: {relative}"
            )
        observed_keys = {(int(row[0]), int(row[1])) for row in observed_rows}
        null_keys = {
            (int(row[0]), int(row[1]), int(row[2])) for row in null_rows
        }
        stressor_keys = {
            (int(row[0]), int(row[1]), int(row[2])) for row in stressor_rows
        }
        expected_observed_keys = {
            (graph, q) for graph in range(cohort) for q in (1, 2)
        }
        expected_draw_keys = {
            (graph, draw, q)
            for graph in range(cohort)
            for draw in range(draws)
            for q in (1, 2)
        }
        if (
            observed_keys != expected_observed_keys
            or null_keys != expected_draw_keys
            or stressor_keys != (expected_draw_keys if config == "S2" else set())
        ):
            raise IndependentLockedValidationError(
                f"locked shard component keys differ: {relative}"
            )
        shards.append(shard)
    recomputed = _reconstruct_analysis(spec, spec_hash, shards)
    stored = load_json(output / "analysis.json")
    if canonical_json_bytes(recomputed) != canonical_json_bytes(stored):
        raise IndependentLockedValidationError("independent locked analysis differs")
    return {
        "status": "pass",
        "decision": recomputed["decision"],
        "locked_pass": recomputed["locked_pass"],
        "study_sha256": spec_hash,
        "analysis_sha256": sha256_file(output / "analysis.json"),
        "manifest_sha256": sha256_file(output / "manifest.json"),
        "shard_count": len(shards),
        "component_reaggregation": "independent",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        sys.stdout.buffer.write(
            canonical_json_bytes(validate_locked_run(args.spec, args.output))
        )
        return 0
    except Exception as exc:
        sys.stderr.write(f"gate12c2 independent locked validation failed: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
