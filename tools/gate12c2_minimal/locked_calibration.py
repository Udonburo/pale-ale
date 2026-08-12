"""Scientific analysis for one bounded locked synthetic calibration."""

from __future__ import annotations

import copy
import math
import platform
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .experiment import load_spec as load_smoke_spec
from .io import canonical_json_bytes, load_json, sha256_file


LOCKED_SPEC_SCHEMA = "gate12c2_locked_calibration_spec_v0.1"
LOCKED_SHARD_SCHEMA = "gate12c2_locked_dataset_shard_v0.1"
LOCKED_ANALYSIS_SCHEMA = "gate12c2_locked_calibration_analysis_v0.1"
CONFIGURATIONS = ("S0", "S1_LOW", "S1_PRIMARY", "S1_HIGH", "S2")
ARM_FIELDS = (
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
Z_ONE_SIDED_95 = 1.6448536269514722


class LockedCalibrationError(ValueError):
    """Raised when the locked design or one dataset shard is invalid."""


def current_environment() -> dict[str, str]:
    import numpy

    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": numpy.__version__,
        "platform_system": platform.system(),
        "machine": platform.machine(),
    }


def _require_int(value: object, label: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise LockedCalibrationError(f"{label} must be an integer >= {minimum}")
    return value


def _require_float(value: object, label: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LockedCalibrationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise LockedCalibrationError(f"{label} is outside its range")
    return result


def validate_locked_spec(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LockedCalibrationError("locked specification must be an object")
    expected = {
        "schema_version",
        "study_id",
        "attempt_id",
        "epistemic_status",
        "seed_namespace",
        "alternative",
        "q_values",
        "epsilon",
        "spectral_gap_tolerance",
        "numerical_tolerance",
        "dataset_count",
        "cohort_size",
        "inner_draws",
        "stability_prefix_draws",
        "stressor_trials",
        "s1_effects",
        "cases",
        "criteria",
        "resource_cap",
        "environment",
        "implementation_sha256",
    }
    if set(value) != expected:
        raise LockedCalibrationError("locked specification fields differ")
    if value["schema_version"] != LOCKED_SPEC_SCHEMA:
        raise LockedCalibrationError("unsupported locked specification")
    if value["epistemic_status"] != "locked_synthetic_calibration_once":
        raise LockedCalibrationError("locked epistemic status differs")
    if value["alternative"] != "observed_smaller_than_null":
        raise LockedCalibrationError("locked alternative differs")
    if value["q_values"] != [1, 2]:
        raise LockedCalibrationError("locked q values must be [1, 2]")
    for label in ("study_id", "attempt_id", "seed_namespace"):
        if not isinstance(value[label], str) or not value[label]:
            raise LockedCalibrationError(f"{label} must be nonempty text")
    _require_float(value["epsilon"], "epsilon")
    _require_float(value["spectral_gap_tolerance"], "spectral_gap_tolerance")
    _require_float(value["numerical_tolerance"], "numerical_tolerance")
    _require_int(value["dataset_count"], "dataset_count", 1)
    _require_int(value["cohort_size"], "cohort_size", 8)
    draws = _require_int(value["inner_draws"], "inner_draws", 3)
    prefix = _require_int(
        value["stability_prefix_draws"], "stability_prefix_draws", 1
    )
    if prefix >= draws:
        raise LockedCalibrationError("stability prefix must be shorter than full")
    if value["stressor_trials"] != 1:
        raise LockedCalibrationError("locked S2 permits exactly one stressor draw")

    effects = value["s1_effects"]
    if not isinstance(effects, dict) or set(effects) != {"low", "primary", "high"}:
        raise LockedCalibrationError("S1 effect grid differs")
    low, primary, high = (
        _require_float(effects[name], f"s1_effects.{name}")
        for name in ("low", "primary", "high")
    )
    if not 0.0 < low < primary < high <= 0.5:
        raise LockedCalibrationError("S1 effects must be strictly increasing")

    cases = value["cases"]
    case_fields = {
        "case_id",
        "model",
        "family",
        "ambient_dim",
        "local_rank",
        "s0_frame_noise",
        "s2_frame_noise",
        "role_center_blend",
        "s1_input_blend",
        "s1_effect_multiplier",
    }
    if not isinstance(cases, list) or len(cases) != 12:
        raise LockedCalibrationError("locked suite requires twelve cases")
    ids: set[str] = set()
    mechanisms: set[tuple[float, float, float]] = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) != case_fields:
            raise LockedCalibrationError("locked case fields differ")
        case_id = case["case_id"]
        if not isinstance(case_id, str) or not case_id or case_id in ids:
            raise LockedCalibrationError("case IDs must be unique text")
        ids.add(case_id)
        for label in ("model", "family"):
            if not isinstance(case[label], str) or not case[label]:
                raise LockedCalibrationError(f"case {label} must be text")
        ambient = _require_int(case["ambient_dim"], "ambient_dim", 4)
        local = _require_int(case["local_rank"], "local_rank", 3)
        if ambient <= local:
            raise LockedCalibrationError("ambient dimension must exceed rank")
        for label in (
            "s0_frame_noise",
            "s2_frame_noise",
            "role_center_blend",
            "s1_input_blend",
            "s1_effect_multiplier",
        ):
            _require_float(case[label], label)
        if float(case["s0_frame_noise"]) <= 0.0 or float(case["s2_frame_noise"]) <= 0.0:
            raise LockedCalibrationError("case frame noise must be positive")
        if not 0.0 <= float(case["role_center_blend"]) < 1.0:
            raise LockedCalibrationError("role center blend differs")
        if not 0.0 <= float(case["s1_input_blend"]) < 1.0:
            raise LockedCalibrationError("S1 input blend differs")
        if float(case["s1_effect_multiplier"]) <= 0.0:
            raise LockedCalibrationError("S1 effect multiplier must be positive")
        mechanisms.add(
            (
                float(case["role_center_blend"]),
                float(case["s1_input_blend"]),
                float(case["s1_effect_multiplier"]),
            )
        )
    if len(mechanisms) < 4:
        raise LockedCalibrationError("case labels do not encode enough mechanisms")

    criteria = value["criteria"]
    criteria_fields = {
        "holm_alpha",
        "zero_tolerance",
        "s0_max_point_rate",
        "s0_max_wilson_upper",
        "s1_primary_min_power",
        "s1_primary_min_wilson_lower",
        "s2_min_success_rate",
        "s2_min_wilson_lower",
        "s2_min_log_inflation",
        "s2_min_endpoint_fraction",
        "s2_min_channel_fraction",
        "nuisance_max_quantile_difference",
        "nuisance_median_quantile_difference",
        "gauge_max_error",
        "stability_min_decision_agreement",
        "stability_max_p95_effect_shift",
        "promotion_min_case_count",
    }
    if not isinstance(criteria, dict) or set(criteria) != criteria_fields:
        raise LockedCalibrationError("locked criteria fields differ")
    for label in criteria_fields - {"promotion_min_case_count"}:
        _require_float(criteria[label], f"criteria.{label}")
    bounded_criteria = {
        "holm_alpha",
        "s0_max_point_rate",
        "s0_max_wilson_upper",
        "s1_primary_min_power",
        "s1_primary_min_wilson_lower",
        "s2_min_success_rate",
        "s2_min_wilson_lower",
        "s2_min_endpoint_fraction",
        "s2_min_channel_fraction",
        "stability_min_decision_agreement",
    }
    if any(float(criteria[label]) > 1.0 for label in bounded_criteria):
        raise LockedCalibrationError("probability criterion exceeds one")
    if float(criteria["holm_alpha"]) <= 0.0:
        raise LockedCalibrationError("Holm alpha must be positive")
    promotion_minimum = _require_int(
        criteria["promotion_min_case_count"], "promotion_min_case_count", 1
    )
    if promotion_minimum > len(cases):
        raise LockedCalibrationError("promotion case count exceeds suite")

    cap = value["resource_cap"]
    if not isinstance(cap, dict) or set(cap) != {
        "max_wall_seconds",
        "max_output_bytes",
        "max_dataset_shards",
    }:
        raise LockedCalibrationError("resource cap differs")
    _require_float(cap["max_wall_seconds"], "max_wall_seconds", 1.0)
    _require_int(cap["max_output_bytes"], "max_output_bytes", 1)
    expected_shards = len(CONFIGURATIONS) * len(cases) * int(value["dataset_count"])
    if cap["max_dataset_shards"] != expected_shards:
        raise LockedCalibrationError("resource cap shard count is not exact")

    environment = value["environment"]
    if not isinstance(environment, dict) or environment != current_environment():
        raise LockedCalibrationError("runtime environment differs from frozen spec")
    hashes = value["implementation_sha256"]
    if not isinstance(hashes, dict) or not hashes:
        raise LockedCalibrationError("implementation hash set is absent")
    if any(
        not isinstance(name, str)
        or not isinstance(digest, str)
        or len(digest) != 64
        for name, digest in hashes.items()
    ):
        raise LockedCalibrationError("implementation hash set is malformed")
    return copy.deepcopy(value)


def load_locked_spec(path: Path) -> tuple[dict[str, Any], str]:
    return validate_locked_spec(load_json(path)), sha256_file(path)


def generation_spec(spec: Mapping[str, Any], config_id: str) -> dict[str, Any]:
    effect_name = {
        "S1_LOW": "low",
        "S1_PRIMARY": "primary",
        "S1_HIGH": "high",
    }.get(config_id, "primary")
    return {
        "study_id": spec["study_id"],
        "seed_namespace": spec["seed_namespace"],
        "q_values": list(spec["q_values"]),
        "epsilon": spec["epsilon"],
        "spectral_gap_tolerance": spec["spectral_gap_tolerance"],
        "numerical_tolerance": spec["numerical_tolerance"],
        "outer_count": spec["dataset_count"],
        "cohort_size": spec["cohort_size"],
        "inner_draws": spec["inner_draws"],
        "stressor_trials": spec["stressor_trials"],
        "s1": {
            "effect_strength": spec["s1_effects"][effect_name],
            "observed_mismatch": 0.01,
        },
        "smoke_acceptance": {
            "zero_tolerance": spec["criteria"]["zero_tolerance"],
        },
        "alternative": spec["alternative"],
    }


def expected_dataset_shards(spec: Mapping[str, Any]) -> list[tuple[str, str, int]]:
    return [
        (config_id, case["case_id"], dataset_index)
        for config_id in CONFIGURATIONS
        for dataset_index in range(int(spec["dataset_count"]))
        for case in spec["cases"]
    ]


def shard_id(config_id: str, case_id: str, dataset_index: int) -> str:
    return f"{config_id}__{case_id}__d{dataset_index:04d}"


def shard_relative_path(config_id: str, case_id: str, dataset_index: int) -> str:
    return f"shards/{config_id}/{case_id}/d{dataset_index:04d}.json"


def _pack_arm(value: Mapping[str, Any]) -> list[Any]:
    return [copy.deepcopy(value[name]) for name in ARM_FIELDS]


def _unpack_arm(value: object) -> dict[str, Any]:
    if not isinstance(value, list) or len(value) != len(ARM_FIELDS):
        raise LockedCalibrationError("compact diagnostic arm differs")
    return dict(zip(ARM_FIELDS, copy.deepcopy(value)))


def compact_dataset_shard(
    dataset: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
    spec_sha256: str,
    config_id: str,
) -> dict[str, Any]:
    case = dataset["case"]
    dataset_index = int(dataset["dataset_index"])
    observed: dict[tuple[int, int], list[Any]] = {}
    null_rows: list[list[Any]] = []
    stressor_rows: list[list[Any]] = []
    for row in dataset["component_rows"]:
        graph_index = int(row["graph_index"])
        draw_index = int(row["draw_index"])
        q = int(row["q"])
        observed_row = [
            graph_index,
            q,
            _pack_arm(row["observed"]),
            _pack_arm(row["observed_gauge"]),
            row["edge_singular_values_observed"],
            row["gauge_defect_error"],
            row["gauge_component_max_error"],
        ]
        key = (graph_index, q)
        if key in observed and canonical_json_bytes(observed[key]) != canonical_json_bytes(
            observed_row
        ):
            raise LockedCalibrationError("observed component changes across draws")
        observed[key] = observed_row
        null_rows.append(
            [
                graph_index,
                draw_index,
                q,
                _pack_arm(row["null"]),
                row["edge_singular_values_null"],
            ]
        )
        if config_id == "S2":
            stressor_rows.append(
                [
                    graph_index,
                    draw_index,
                    q,
                    _pack_arm(row["stressor"]),
                    row["edge_singular_values_stressor"],
                    row["inflation_channel_moved"],
                ]
            )
    result = {
        "schema_version": LOCKED_SHARD_SCHEMA,
        "study_id": spec["study_id"],
        "study_sha256": spec_sha256,
        "attempt_id": spec["attempt_id"],
        "shard_id": shard_id(config_id, case["case_id"], dataset_index),
        "config_id": config_id,
        "generation_regime": "S1" if config_id.startswith("S1_") else config_id,
        "effect_strength": (
            generation_spec(spec, config_id)["s1"]["effect_strength"]
            if config_id.startswith("S1_")
            else None
        ),
        "case": dict(case),
        "dataset_index": dataset_index,
        "observed_cohort_sha256": dataset["observed_cohort_sha256"][0],
        "observed_arm_unchanged": dataset["observed_arm_unchanged"],
        "n1_realizability_max_error": dataset["n1_realizability_max_error"],
        "gauge_defect_max_error": dataset["gauge_defect_max_error"],
        "gauge_component_max_error": dataset["gauge_component_max_error"],
        "stressor_edge_spectrum_max_error": dataset[
            "stressor_edge_spectrum_max_error"
        ],
        "arm_fields": list(ARM_FIELDS),
        "observed_rows": [observed[key] for key in sorted(observed)],
        "null_rows": sorted(null_rows, key=lambda row: (row[0], row[1], row[2])),
        "stressor_rows": sorted(
            stressor_rows, key=lambda row: (row[0], row[1], row[2])
        ),
    }
    validate_compact_shard(
        result,
        spec=spec,
        spec_sha256=spec_sha256,
        config_id=config_id,
        case_id=case["case_id"],
        dataset_index=dataset_index,
    )
    return result


def _validate_arm(
    value: object,
    spec: Mapping[str, Any],
    *,
    q: int,
    local_rank: int,
) -> dict[str, Any]:
    arm = _unpack_arm(value)
    for name in ("a", "u", "v", "x", "y"):
        numeric = float(arm[name])
        if not math.isfinite(numeric) or numeric < 0.0:
            raise LockedCalibrationError("diagnostic magnitude differs")
    c = arm["c"]
    if c is not None:
        c = float(c)
        if not math.isfinite(c) or abs(c) > 1.0:
            raise LockedCalibrationError("diagnostic alignment differs")
        rhs = (
            float(arm["x"]) ** 2
            + float(arm["y"]) ** 2
            - 2.0 * float(arm["x"]) * float(arm["y"]) * c
        )
        tolerance = float(spec["numerical_tolerance"]) * max(
            1.0, float(arm["a"]) ** 2, abs(rhs)
        )
        if abs(float(arm["a"]) ** 2 - rhs) > tolerance:
            raise LockedCalibrationError("diagnostic residual identity differs")
    degeneracy_tolerance = 1e-12
    for numerator, denominator, name in (
        ("x", "u", "p_left"),
        ("y", "v", "p_right"),
    ):
        expected = (
            None
            if float(arm[denominator]) <= degeneracy_tolerance
            else float(arm[numerator]) / float(arm[denominator])
        )
        actual = arm[name]
        if expected is None:
            if actual is not None:
                raise LockedCalibrationError("diagnostic propagation ratio differs")
        elif actual is None or not math.isclose(
            float(actual), expected, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise LockedCalibrationError("diagnostic propagation ratio differs")
    gaps = [
        float(arm["relative_gap_left"]),
        float(arm["relative_gap_right"]),
    ]
    if any(not math.isfinite(gap) or gap < -1e-15 for gap in gaps):
        raise LockedCalibrationError("diagnostic spectral gap differs")
    expected_eligible = all(
        gap > float(spec["spectral_gap_tolerance"]) for gap in gaps
    )
    if not isinstance(arm["eligible"], bool) or arm["eligible"] != expected_eligible:
        raise LockedCalibrationError("diagnostic eligibility differs")
    if not isinstance(arm["numerical_pass"], bool):
        raise LockedCalibrationError("diagnostic status differs")
    for name in (
        "product_singular_values_left",
        "product_singular_values_right",
    ):
        spectrum = arm[name]
        if not isinstance(spectrum, list) or len(spectrum) != local_rank:
            raise LockedCalibrationError("diagnostic product spectrum rank differs")
        numeric = [float(item) for item in spectrum]
        if (
            any(not math.isfinite(item) or item < 0.0 for item in numeric)
            or numeric != sorted(numeric, reverse=True)
        ):
            raise LockedCalibrationError("diagnostic product spectrum differs")
    errors = [
        float(arm["matrix_identity_error"]),
        float(arm["squared_identity_error"]),
    ]
    if any(not math.isfinite(error) or error < 0.0 for error in errors):
        raise LockedCalibrationError("diagnostic identity error differs")
    numerical_tolerance = float(spec["numerical_tolerance"])
    expected_numerical = bool(
        errors[0]
        <= numerical_tolerance
        * max(1.0, float(arm["a"]), float(arm["x"]), float(arm["y"]))
        and errors[1]
        <= numerical_tolerance
        * max(
            1.0,
            float(arm["a"]) ** 2,
            float(arm["x"]) ** 2 + float(arm["y"]) ** 2,
        )
    )
    if arm["numerical_pass"] != expected_numerical:
        raise LockedCalibrationError("diagnostic numerical status differs")
    if not 1 <= q < local_rank:
        raise LockedCalibrationError("diagnostic q differs")
    return arm


def _validate_edge_spectra(value: object, local_rank: int) -> None:
    if not isinstance(value, list) or len(value) != 3:
        raise LockedCalibrationError("edge spectrum surface differs")
    for spectrum in value:
        if not isinstance(spectrum, list) or len(spectrum) != local_rank:
            raise LockedCalibrationError("edge spectrum rank differs")
        numbers = [float(item) for item in spectrum]
        if any(not math.isfinite(item) or item < 0.0 for item in numbers):
            raise LockedCalibrationError("edge spectrum value differs")
        if numbers != sorted(numbers, reverse=True):
            raise LockedCalibrationError("edge spectrum order differs")


def validate_compact_shard(
    value: object,
    *,
    spec: Mapping[str, Any],
    spec_sha256: str,
    config_id: str,
    case_id: str,
    dataset_index: int,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LockedCalibrationError("locked shard must be an object")
    expected_root = {
        "schema_version",
        "study_id",
        "study_sha256",
        "attempt_id",
        "shard_id",
        "config_id",
        "generation_regime",
        "effect_strength",
        "case",
        "dataset_index",
        "observed_cohort_sha256",
        "observed_arm_unchanged",
        "n1_realizability_max_error",
        "gauge_defect_max_error",
        "gauge_component_max_error",
        "stressor_edge_spectrum_max_error",
        "arm_fields",
        "observed_rows",
        "null_rows",
        "stressor_rows",
    }
    if set(value) != expected_root:
        raise LockedCalibrationError("locked shard fields differ")
    if (
        value["schema_version"] != LOCKED_SHARD_SCHEMA
        or value["study_id"] != spec["study_id"]
        or value["study_sha256"] != spec_sha256
        or value["attempt_id"] != spec["attempt_id"]
        or value["config_id"] != config_id
        or value["dataset_index"] != dataset_index
        or value["shard_id"] != shard_id(config_id, case_id, dataset_index)
        or value["arm_fields"] != list(ARM_FIELDS)
    ):
        raise LockedCalibrationError("locked shard identity differs")
    cases = {case["case_id"]: case for case in spec["cases"]}
    if case_id not in cases or value["case"] != cases[case_id]:
        raise LockedCalibrationError("locked shard case differs")
    if value["observed_arm_unchanged"] is not True:
        raise LockedCalibrationError("observed arm mutation marker differs")
    local_rank = int(cases[case_id]["local_rank"])
    cohort = int(spec["cohort_size"])
    draws = int(spec["inner_draws"])
    q_values = [int(q) for q in spec["q_values"]]
    observed_rows = value["observed_rows"]
    null_rows = value["null_rows"]
    stressor_rows = value["stressor_rows"]
    if not isinstance(observed_rows, list) or len(observed_rows) != cohort * len(
        q_values
    ):
        raise LockedCalibrationError("observed compact surface differs")
    if not isinstance(null_rows, list) or len(null_rows) != cohort * draws * len(
        q_values
    ):
        raise LockedCalibrationError("null compact surface differs")
    expected_stressor = cohort * draws * len(q_values) if config_id == "S2" else 0
    if not isinstance(stressor_rows, list) or len(stressor_rows) != expected_stressor:
        raise LockedCalibrationError("stressor compact surface differs")

    observed_keys: set[tuple[int, int]] = set()
    for row in observed_rows:
        if not isinstance(row, list) or len(row) != 7:
            raise LockedCalibrationError("observed compact row differs")
        graph, q = int(row[0]), int(row[1])
        observed_keys.add((graph, q))
        observed_arm = _validate_arm(row[2], spec, q=q, local_rank=local_rank)
        gauge_arm = _validate_arm(row[3], spec, q=q, local_rank=local_rank)
        _validate_edge_spectra(row[4], local_rank)
        gauge_defect = abs(float(observed_arm["a"]) - float(gauge_arm["a"]))
        gauge_component = max(
            abs(float(observed_arm[name]) - float(gauge_arm[name]))
            for name in ("u", "v", "x", "y")
        )
        optional_errors = []
        for name in ("c", "p_left", "p_right"):
            left, right = observed_arm[name], gauge_arm[name]
            optional_errors.append(
                0.0
                if left is None and right is None
                else math.inf
                if left is None or right is None
                else abs(float(left) - float(right))
            )
        gauge_component = max(gauge_component, *optional_errors)
        if not math.isclose(gauge_defect, float(row[5]), rel_tol=1e-12, abs_tol=1e-15):
            raise LockedCalibrationError("gauge defect evidence differs")
        if not math.isclose(gauge_component, float(row[6]), rel_tol=1e-12, abs_tol=1e-15):
            raise LockedCalibrationError("gauge component evidence differs")
        if any(not math.isfinite(float(item)) or float(item) < 0.0 for item in row[5:]):
            raise LockedCalibrationError("gauge evidence differs")
    if observed_keys != {
        (graph, q) for graph in range(cohort) for q in q_values
    }:
        raise LockedCalibrationError("observed compact keys differ")

    null_keys: set[tuple[int, int, int]] = set()
    null_spectra: dict[tuple[int, int, int], Any] = {}
    for row in null_rows:
        if not isinstance(row, list) or len(row) != 5:
            raise LockedCalibrationError("null compact row differs")
        key = (int(row[0]), int(row[1]), int(row[2]))
        null_keys.add(key)
        _validate_arm(row[3], spec, q=key[2], local_rank=local_rank)
        _validate_edge_spectra(row[4], local_rank)
        null_spectra[key] = row[4]
    expected_keys = {
        (graph, draw, q)
        for graph in range(cohort)
        for draw in range(draws)
        for q in q_values
    }
    if null_keys != expected_keys:
        raise LockedCalibrationError("null compact keys differ")

    stressor_keys: set[tuple[int, int, int]] = set()
    for row in stressor_rows:
        if not isinstance(row, list) or len(row) != 6:
            raise LockedCalibrationError("stressor compact row differs")
        key = (int(row[0]), int(row[1]), int(row[2]))
        stressor_keys.add(key)
        _validate_arm(row[3], spec, q=key[2], local_rank=local_rank)
        _validate_edge_spectra(row[4], local_rank)
        if row[5] not in {True, False}:
            raise LockedCalibrationError("stressor channel flag differs")
        for before, after in zip(null_spectra[key], row[4]):
            if not np.allclose(before, after, rtol=1e-12, atol=1e-12):
                raise LockedCalibrationError("stressor edge spectrum differs")
    if stressor_keys != (expected_keys if config_id == "S2" else set()):
        raise LockedCalibrationError("stressor compact keys differ")
    return copy.deepcopy(value)


def _sign_p(values: Sequence[float], negative: bool, tolerance: float) -> float:
    informative = [value for value in values if abs(value) > tolerance]
    if not informative:
        return 1.0
    directional = sum(value < 0.0 if negative else value > 0.0 for value in informative)
    return float(
        sum(
            math.comb(len(informative), count)
            for count in range(directional, len(informative) + 1)
        )
        / (2 ** len(informative))
    )


def endpoint_from_compact(
    shard: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
    q: int,
    draw_limit: int,
) -> dict[str, Any]:
    observed = {
        (int(row[0]), int(row[1])): _unpack_arm(row[2])
        for row in shard["observed_rows"]
    }
    null = {
        (int(row[0]), int(row[1]), int(row[2])): _unpack_arm(row[3])
        for row in shard["null_rows"]
        if int(row[1]) < draw_limit
    }
    stressor = {
        (int(row[0]), int(row[1]), int(row[2])): (
            _unpack_arm(row[3]), bool(row[5])
        )
        for row in shard["stressor_rows"]
        if int(row[1]) < draw_limit
    }
    epsilon = float(spec["epsilon"])
    tolerance = float(spec["criteria"]["zero_tolerance"])
    block_effects: list[float] = []
    block_channels: list[float] = []
    for graph in range(int(spec["cohort_size"])):
        effects: list[float] = []
        channels: list[bool] = []
        for draw in range(draw_limit):
            base = null[(graph, draw, q)]
            if shard["config_id"] == "S2":
                stress, channel = stressor[(graph, draw, q)]
                effects.append(
                    math.log(float(stress["a"]) + epsilon)
                    - math.log(float(base["a"]) + epsilon)
                )
                channels.append(channel)
            else:
                obs = observed[(graph, q)]
                effects.append(
                    math.log(float(obs["a"]) + epsilon)
                    - math.log(float(base["a"]) + epsilon)
                )
        block_effects.append(float(np.median(effects)))
        if channels:
            block_channels.append(float(np.mean(channels)))
    negative = shard["config_id"] != "S2"
    return {
        "case_id": shard["case"]["case_id"],
        "q": int(q),
        "median_effect": float(np.median(block_effects)),
        "directional_sign_p": _sign_p(block_effects, negative, tolerance),
        "channel_fraction": (
            None if not block_channels else float(np.mean(block_channels))
        ),
        "coverage_complete": bool(
            all(arm["eligible"] and arm["numerical_pass"] for arm in observed.values())
            and all(arm["eligible"] and arm["numerical_pass"] for arm in null.values())
            and all(
                arm[0]["eligible"] and arm[0]["numerical_pass"]
                for arm in stressor.values()
            )
        ),
    }


def _holm(endpoints: list[dict[str, Any]], alpha: float, tolerance: float) -> None:
    ordered = sorted(
        range(len(endpoints)),
        key=lambda index: (
            float(endpoints[index]["directional_sign_p"]),
            endpoints[index]["case_id"],
            int(endpoints[index]["q"]),
        ),
    )
    running = 0.0
    for position, index in enumerate(ordered):
        running = max(
            running,
            min(
                1.0,
                float(endpoints[index]["directional_sign_p"])
                * (len(endpoints) - position),
            ),
        )
        endpoints[index]["holm_adjusted_p"] = float(running)
        endpoints[index]["supported"] = bool(
            endpoints[index]["coverage_complete"]
            and endpoints[index]["median_effect"] < -tolerance
            and running < alpha
        )


def dataset_decision(
    shards: Sequence[Mapping[str, Any]],
    *,
    spec: Mapping[str, Any],
    draw_limit: int,
) -> dict[str, Any]:
    if len(shards) != 12 or len({row["config_id"] for row in shards}) != 1:
        raise LockedCalibrationError("dataset decision requires twelve one-config shards")
    config_id = str(shards[0]["config_id"])
    endpoints = [
        endpoint_from_compact(shard, spec=spec, q=q, draw_limit=draw_limit)
        for shard in shards
        for q in spec["q_values"]
    ]
    criteria = spec["criteria"]
    if config_id != "S2":
        _holm(
            endpoints,
            float(criteria["holm_alpha"]),
            float(criteria["zero_tolerance"]),
        )
        by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for endpoint in endpoints:
            by_case[endpoint["case_id"]].append(endpoint)
        supported_cases = sum(
            len(rows) == 2 and all(row["supported"] for row in rows)
            for rows in by_case.values()
        )
        promotion = supported_cases >= int(criteria["promotion_min_case_count"])
        return {
            "config_id": config_id,
            "dataset_index": int(shards[0]["dataset_index"]),
            "any_endpoint_support": any(row["supported"] for row in endpoints),
            "supported_endpoint_count": sum(row["supported"] for row in endpoints),
            "supported_case_count": int(supported_cases),
            "promotion": bool(promotion),
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
    supported_count = sum(row["supported"] for row in endpoints)
    family_median = float(
        np.median([float(row["median_effect"]) for row in endpoints])
    )
    return {
        "config_id": config_id,
        "dataset_index": int(shards[0]["dataset_index"]),
        "any_endpoint_support": None,
        "supported_endpoint_count": int(supported_count),
        "supported_case_count": None,
        "promotion": None,
        "identification_success": bool(
            family_median > float(criteria["s2_min_log_inflation"])
            and
            supported_count / len(endpoints)
            >= float(criteria["s2_min_endpoint_fraction"])
        ),
        "family_median_effect": family_median,
        "endpoints": endpoints,
    }


def wilson_interval(successes: int, count: int) -> tuple[float, float]:
    if count <= 0 or not 0 <= successes <= count:
        raise LockedCalibrationError("invalid Wilson inputs")
    proportion = successes / count
    z2 = Z_ONE_SIDED_95**2
    denominator = 1.0 + z2 / count
    center = (proportion + z2 / (2.0 * count)) / denominator
    radius = Z_ONE_SIDED_95 * math.sqrt(
        proportion * (1.0 - proportion) / count + z2 / (4.0 * count * count)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _nuisance_summary(shards: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    observed: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    null: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for shard in shards:
        if shard["config_id"] != "S0":
            continue
        case_id = str(shard["case"]["case_id"])
        for row in shard["observed_rows"]:
            if int(row[1]) != 1:
                continue
            for edge_index, spectrum in enumerate(row[4]):
                for singular_index, value in enumerate(spectrum):
                    observed[(case_id, edge_index, singular_index)].append(float(value))
        for row in shard["null_rows"]:
            if int(row[2]) != 1:
                continue
            for edge_index, spectrum in enumerate(row[4]):
                for singular_index, value in enumerate(spectrum):
                    null[(case_id, edge_index, singular_index)].append(float(value))
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
    return {
        "group_count": len(differences),
        "max_quantile_difference": float(max(differences)),
        "median_quantile_difference": float(np.median(differences)),
    }


def analyze_locked_shards(
    spec: Mapping[str, Any],
    spec_sha256: str,
    shards: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected = expected_dataset_shards(spec)
    if len(shards) != len(expected):
        raise LockedCalibrationError("locked shard count differs")
    by_dataset: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for shard in shards:
        by_dataset[(str(shard["config_id"]), int(shard["dataset_index"]))].append(
            shard
        )
    full: list[dict[str, Any]] = []
    prefix: list[dict[str, Any]] = []
    for key in sorted(by_dataset):
        rows = by_dataset[key]
        full.append(
            dataset_decision(rows, spec=spec, draw_limit=int(spec["inner_draws"]))
        )
        prefix.append(
            dataset_decision(
                rows,
                spec=spec,
                draw_limit=int(spec["stability_prefix_draws"]),
            )
        )

    config_summary: dict[str, Any] = {}
    for config_id in CONFIGURATIONS:
        rows = [row for row in full if row["config_id"] == config_id]
        count = len(rows)
        if config_id == "S0":
            any_successes = sum(row["any_endpoint_support"] for row in rows)
            promotion_successes = sum(row["promotion"] for row in rows)
            any_interval = wilson_interval(any_successes, count)
            promotion_interval = wilson_interval(promotion_successes, count)
            config_summary[config_id] = {
                "dataset_count": count,
                "any_endpoint_rate": any_successes / count,
                "any_endpoint_wilson_95": list(any_interval),
                "promotion_rate": promotion_successes / count,
                "promotion_wilson_95": list(promotion_interval),
            }
        elif config_id.startswith("S1_"):
            successes = sum(row["promotion"] for row in rows)
            interval = wilson_interval(successes, count)
            config_summary[config_id] = {
                "dataset_count": count,
                "promotion_power": successes / count,
                "promotion_wilson_95": list(interval),
            }
        else:
            successes = sum(row["identification_success"] for row in rows)
            interval = wilson_interval(successes, count)
            config_summary[config_id] = {
                "dataset_count": count,
                "identification_rate": successes / count,
                "identification_wilson_95": list(interval),
            }

    prefix_by_key = {
        (row["config_id"], row["dataset_index"]): row for row in prefix
    }
    shifts: list[float] = []
    agreements: list[bool] = []
    for full_row in full:
        if full_row["config_id"] not in {"S0", "S1_PRIMARY", "S2"}:
            continue
        prefix_row = prefix_by_key[(full_row["config_id"], full_row["dataset_index"])]
        full_endpoints = {
            (row["case_id"], row["q"]): row for row in full_row["endpoints"]
        }
        prefix_endpoints = {
            (row["case_id"], row["q"]): row for row in prefix_row["endpoints"]
        }
        for key in full_endpoints:
            shifts.append(
                abs(
                    float(full_endpoints[key]["median_effect"])
                    - float(prefix_endpoints[key]["median_effect"])
                )
            )
            agreements.append(
                bool(full_endpoints[key]["supported"])
                == bool(prefix_endpoints[key]["supported"])
            )
    stability = {
        "comparison_count": len(shifts),
        "decision_agreement": float(np.mean(agreements)),
        "p95_effect_shift": float(np.quantile(shifts, 0.95)),
        "max_effect_shift": float(max(shifts)),
    }
    nuisance = _nuisance_summary(shards)
    controls = {
        "all_observed_unchanged": all(
            shard["observed_arm_unchanged"] is True for shard in shards
        ),
        "max_n1_realizability_error": max(
            float(shard["n1_realizability_max_error"]) for shard in shards
        ),
        "max_gauge_defect_error": max(
            float(shard["gauge_defect_max_error"]) for shard in shards
        ),
        "max_gauge_component_error": max(
            float(shard["gauge_component_max_error"]) for shard in shards
        ),
        "max_stressor_edge_spectrum_error": max(
            float(shard["stressor_edge_spectrum_max_error"])
            for shard in shards
            if shard["config_id"] == "S2"
        ),
    }
    criteria = spec["criteria"]
    gate_results = {
        "mechanical_controls": bool(
            controls["all_observed_unchanged"]
            and controls["max_n1_realizability_error"] <= 1e-10
            and controls["max_gauge_defect_error"] <= criteria["gauge_max_error"]
            and controls["max_gauge_component_error"] <= criteria["gauge_max_error"]
            and controls["max_stressor_edge_spectrum_error"] <= 1e-10
        ),
        "s0_familywise_safety": bool(
            config_summary["S0"]["any_endpoint_rate"]
            <= criteria["s0_max_point_rate"]
            and config_summary["S0"]["any_endpoint_wilson_95"][1]
            <= criteria["s0_max_wilson_upper"]
        ),
        "s0_promotion_safety": bool(
            config_summary["S0"]["promotion_rate"]
            <= criteria["s0_max_point_rate"]
            and config_summary["S0"]["promotion_wilson_95"][1]
            <= criteria["s0_max_wilson_upper"]
        ),
        "s1_primary_power": bool(
            config_summary["S1_PRIMARY"]["promotion_power"]
            >= criteria["s1_primary_min_power"]
            and config_summary["S1_PRIMARY"]["promotion_wilson_95"][0]
            >= criteria["s1_primary_min_wilson_lower"]
        ),
        "s2_identification": bool(
            config_summary["S2"]["identification_rate"]
            >= criteria["s2_min_success_rate"]
            and config_summary["S2"]["identification_wilson_95"][0]
            >= criteria["s2_min_wilson_lower"]
        ),
        "nuisance_fidelity": bool(
            nuisance["max_quantile_difference"]
            <= criteria["nuisance_max_quantile_difference"]
            and nuisance["median_quantile_difference"]
            <= criteria["nuisance_median_quantile_difference"]
        ),
        "inner_draw_stability": bool(
            stability["decision_agreement"]
            >= criteria["stability_min_decision_agreement"]
            and stability["p95_effect_shift"]
            <= criteria["stability_max_p95_effect_shift"]
        ),
    }
    passed = all(gate_results.values())
    return {
        "schema_version": LOCKED_ANALYSIS_SCHEMA,
        "study_id": spec["study_id"],
        "study_sha256": spec_sha256,
        "attempt_id": spec["attempt_id"],
        "epistemic_status": "locked_synthetic_calibration",
        "decision": "LOCKED_PASS" if passed else "RETIRE_OR_DEMOTE",
        "locked_pass": passed,
        "real_held_out_authorized": False,
        "scientific_claim_authorized": False,
        "dataset_is_inference_unit": True,
        "shard_count": len(shards),
        "gate_results": gate_results,
        "controls": controls,
        "nuisance": nuisance,
        "stability": stability,
        "configurations": config_summary,
        "dataset_decisions": [
            {
                key: value
                for key, value in row.items()
                if key != "endpoints"
            }
            for row in full
        ],
    }
