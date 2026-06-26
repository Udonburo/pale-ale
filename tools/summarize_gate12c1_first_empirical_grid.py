#!/usr/bin/env python3
"""Summarize the frozen Gate12C-1 first empirical twelve-case grid."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

import inspect_gate12c_associator_feasibility as gate12c0


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()

SCHEMA_VERSION = "gate12c1_first_empirical_grid_summary_v1"
METHOD_ID = "gate12c1_first_empirical_grid_summary_v1"
PLAN_ID = "gate12c1_first_empirical_execution_plan_v1"
CASE_MANIFEST_SCHEMA_VERSION = "gate12c1_first_empirical_case_manifest_v1"

EXPECTED_RUNNER_COMMIT = "8d5613bffe5b6c91d0956c812404072eb76e98c6"
EXPECTED_RUNNER_SCRIPT_SHA256 = (
    "b363fd874a0538dc548853e97e8ec17c0eb84be5658f6e2f01f60d2a12789c3e"
)
EXPECTED_RUNNER_SCHEMA = "gate12c_compressed_overlap_associator_v1"
EXPECTED_RUNNER_METHOD = "gate12c_compressed_overlap_associator_v1"
EXPECTED_RUN_MODE = "gate12a_residual_bearing_explicit_triangle_equal_rank_alpha_v1"
EXPECTED_ORIENTATION_NULL_MODE = "cycle_shared_spectrum_preserving_operator_null_v1"
EXPECTED_ORIENTATION_NULL_GENERATOR = "sha256_counter_box_muller_qr_sign_normalized_v1"
EXPECTED_ORIENTATION_SEED_ENCODING = "canonical_json_utf8_no_insignificant_whitespace_v1"
EXPECTED_ORIENTATION_NULL_SEED = "gate12c1_first_empirical_orientation_null_v1"
EXPECTED_ORIENTATION_NULL_REQUESTED_DRAW_COUNT = 255
EXPECTED_ORIENTATION_NULL_MAX_ATTEMPT_COUNT = 1024

PRIMARY_ZERO_TOLERANCE = 1.0e-12
PRIMARY_EPSILON = 1.0e-12
COVERAGE_THRESHOLD = 0.90
HOLM_ALPHA = 0.05
HOLM_TEST_COUNT = 24

EXPECTED_TOLERANCES: Dict[str, float] = {
    "tau_overlap_sv_min": 1.0e-8,
    "tau_overlap_singular_value_abs_error": 1.0e-8,
    "tau_transport_reconstruction_fro": 1.0e-8,
    "tau_ordinary_associator_fro": 1.0e-10,
    "tau_no_compression_associator_fro": 1.0e-10,
    "tau_split_rel": 1.0e-3,
    "tau_gauge_operator_covariance_fro": 1.0e-8,
    "tau_gauge_scalar_delta_abs": 1.0e-10,
    "epsilon": 1.0e-12,
}

REQUIRED_GATE12C1_FILES = (
    "manifest.json",
    "triangle_associator_registry.jsonl",
    "triangle_associator_arrays.npz",
    "cycle_associator_summary.jsonl",
    "compression_sweep_summary.csv",
    "gauge_stability_summary.json",
    "spectral_orientation_null_summary.jsonl",
    "gate12c_status.json",
    "gate12c_read.md",
    "checksums.json",
)

RUNNER_OUTPUT_FILES_FOR_CHECKSUMS = tuple(
    name for name in REQUIRED_GATE12C1_FILES if name != "checksums.json"
)

OUTPUT_MANIFEST = "manifest.json"
OUTPUT_CASE_INVENTORY = "gate12c1_case_inventory.csv"
OUTPUT_CYCLE_Q = "gate12c1_cycle_q_scores.jsonl"
OUTPUT_BLOCK_Q = "gate12c1_block_q_scores.jsonl"
OUTPUT_RUN_Q = "gate12c1_run_q_tests.csv"
OUTPUT_GRID_SUMMARY = "gate12c1_grid_summary.json"
OUTPUT_SECONDARY = "gate12c1_secondary_telemetry.json"
OUTPUT_READ = "gate12c1_summary_read.md"
OUTPUT_CHECKSUMS = "checksums.json"

OUTPUT_FILES = (
    OUTPUT_MANIFEST,
    OUTPUT_CASE_INVENTORY,
    OUTPUT_CYCLE_Q,
    OUTPUT_BLOCK_Q,
    OUTPUT_RUN_Q,
    OUTPUT_GRID_SUMMARY,
    OUTPUT_SECONDARY,
    OUTPUT_READ,
)

RUNNER_REGISTRY_REQUIRED_FIELDS = (
    "probe_id",
    "cycle_id",
    "canonical_base_node_id",
    "evaluation_root_node_id",
    "root_rotation_index",
    "ordered_node_id_path",
    "ordered_edge_id_path",
    "ordered_relation_kind_path",
    "cycle_rank",
    "compression_rank_q",
    "left_inner_split_gap_rel",
    "right_inner_split_gap_rel",
    "left_cut_status",
    "right_cut_status",
    "truncation_status",
    "ordinary_associator_fro",
    "no_compression_associator_fro",
    "compressed_overlap_associator_fro",
    "compressed_overlap_associator_rel",
    "compressed_overlap_closure_left_fro",
    "compressed_overlap_closure_right_fro",
    "compressed_overlap_closure_gap_abs",
    "gate12a_holonomy_residual_fro",
    "edge_compatibility_gap_max",
    "source_sample_block_id",
    "source_block_status",
    "measurement_status",
    "control_status",
    "aggregation_eligible",
    "gauge_operator_covariance_fro",
    "gauge_scalar_delta_abs",
    "gauge_cut_status_preserved",
    "gauge_scalar_status",
    "orientation_null_status",
    "orientation_null_excess_status",
    "orientation_null_requested_draw_count",
    "orientation_null_valid_draw_count",
    "orientation_null_invalid_cut_count",
    "orientation_null_attempt_count",
    "orientation_null_median",
    "orientation_null_mad",
    "orientation_null_mean",
    "orientation_null_std",
    "orientation_null_empirical_p_upper",
    "orientation_null_robust_z",
    "orientation_null_scale_degenerate",
    "operator_array_index",
)

CASE_INVENTORY_FIELDNAMES = (
    "case_id",
    "case_order",
    "model",
    "family",
    "source_gate12a_run_id",
    "gate12c1_run_id",
    "preflight_eligible_cycle_count",
    "derived_eligible_cycle_count",
    "expected_single_sample_block_count",
    "mixed_or_undefined_expected_cycle_count",
    "source_gate12a_checksum_status",
    "gate12c1_output_checksum_status",
    "source_gate12a_immutability_status",
    "gate12c1_output_immutability_status",
)

RUN_Q_FIELDNAMES = (
    "case_id",
    "case_order",
    "model",
    "family",
    "compression_rank_q",
    "expected_cycle_count",
    "represented_cycle_count",
    "cycle_coverage_ratio",
    "cycle_coverage_pass",
    "expected_block_count",
    "represented_block_count",
    "block_coverage_ratio",
    "block_coverage_pass",
    "mixed_or_undefined_expected_cycle_count",
    "coverage_pass",
    "run_q_median",
    "positive_block_count",
    "negative_block_count",
    "tie_block_count",
    "test_status",
    "raw_p",
    "holm_adjusted_p",
    "holm_sort_position",
    "q_support",
    "run_support",
    "q_discordant_run",
)


@dataclass(frozen=True)
class CanonicalCase:
    case_id: str
    case_order: int
    model: str
    family: str
    source_gate12a_dir: str
    source_gate12a_run_id: str
    preflight_eligible_cycle_count: int


@dataclass(frozen=True)
class CaseInput:
    spec: CanonicalCase
    source_gate12a_dir: Path
    gate12c1_run_dir: Path


@dataclass(frozen=True)
class ExpectedCycle:
    case_id: str
    cycle_id: str
    source_sample_block_id: str
    source_block_status: str
    gate12a_holonomy_residual_fro: float
    edge_compatibility_gap_max: float
    cycle_rank: int


class Gate12C1SummaryContractError(RuntimeError):
    """Raised when summary inputs violate the frozen Gate12C-1 plan."""


CANONICAL_CASES: Tuple[CanonicalCase, ...] = (
    CanonicalCase(
        "case_01",
        1,
        "qwen_qwen2_5_0_5b",
        "transcript_128r",
        "runs/gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_0_5b_transcript_128r_gate9k",
        "gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_0_5b_transcript_128r_gate9k",
        320,
    ),
    CanonicalCase(
        "case_02",
        2,
        "qwen_qwen2_5_0_5b",
        "briefing_200r",
        "runs/gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_0_5b_briefing_200r_gate9k",
        "gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_0_5b_briefing_200r_gate9k",
        500,
    ),
    CanonicalCase(
        "case_03",
        3,
        "qwen_qwen2_5_0_5b",
        "archive_128r",
        "runs/gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_0_5b_archive_128r_gate9k",
        "gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_0_5b_archive_128r_gate9k",
        320,
    ),
    CanonicalCase(
        "case_04",
        4,
        "qwen_qwen2_5_3b_instruct",
        "transcript_128r",
        "runs/gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_3b_instruct_transcript_128r_gate9k",
        "gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_3b_instruct_transcript_128r_gate9k",
        320,
    ),
    CanonicalCase(
        "case_05",
        5,
        "qwen_qwen2_5_3b_instruct",
        "briefing_200r",
        "runs/gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_3b_instruct_briefing_200r_gate9k",
        "gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_3b_instruct_briefing_200r_gate9k",
        500,
    ),
    CanonicalCase(
        "case_06",
        6,
        "qwen_qwen2_5_3b_instruct",
        "archive_128r",
        "runs/gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_3b_instruct_archive_128r_gate9k",
        "gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_3b_instruct_archive_128r_gate9k",
        320,
    ),
    CanonicalCase(
        "case_07",
        7,
        "meta_llama_llama_3_2_3b_instruct",
        "transcript_128r",
        "runs/gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_meta_llama_llama_3_2_3b_instruct_transcript_128r_gate9k",
        "gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_meta_llama_llama_3_2_3b_instruct_transcript_128r_gate9k",
        320,
    ),
    CanonicalCase(
        "case_08",
        8,
        "meta_llama_llama_3_2_3b_instruct",
        "briefing_200r",
        "runs/gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_meta_llama_llama_3_2_3b_instruct_briefing_200r_gate9k",
        "gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_meta_llama_llama_3_2_3b_instruct_briefing_200r_gate9k",
        500,
    ),
    CanonicalCase(
        "case_09",
        9,
        "meta_llama_llama_3_2_3b_instruct",
        "archive_128r",
        "runs/gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_meta_llama_llama_3_2_3b_instruct_archive_128r_gate9k",
        "gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_meta_llama_llama_3_2_3b_instruct_archive_128r_gate9k",
        320,
    ),
    CanonicalCase(
        "case_10",
        10,
        "qwen_qwen3_4b",
        "transcript_128r",
        "runs/gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen3_4b_transcript_128r_gate9k",
        "gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen3_4b_transcript_128r_gate9k",
        320,
    ),
    CanonicalCase(
        "case_11",
        11,
        "qwen_qwen3_4b",
        "briefing_200r",
        "runs/gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen3_4b_briefing_200r_gate9k",
        "gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen3_4b_briefing_200r_gate9k",
        500,
    ),
    CanonicalCase(
        "case_12",
        12,
        "qwen_qwen3_4b",
        "archive_128r",
        "runs/gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen3_4b_archive_128r_gate9k",
        "gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen3_4b_archive_128r_gate9k",
        320,
    ),
)

CANONICAL_BY_ID = {case.case_id: case for case in CANONICAL_CASES}
CANONICAL_BY_ORDER = {case.case_order: case for case in CANONICAL_CASES}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the frozen Gate12C-1 first empirical twelve-case grid."
    )
    parser.add_argument("--case-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args(argv)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(dict(payload)), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(json_ready(dict(row)), ensure_ascii=False, allow_nan=False)
                + "\n"
            )


def csv_value(value: Any) -> Any:
    ready = json_ready(value)
    if isinstance(ready, (dict, list, tuple)):
        return json.dumps(ready, ensure_ascii=False, sort_keys=True)
    if ready is None:
        return ""
    return ready


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: csv_value(row.get(name)) for name in fieldnames})


def write_text(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8", newline="\n")


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return json_ready(float(value))
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON outputs must not contain NaN or infinity")
        return value
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def repo_relative_or_posix(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def path_is_relative_to(*, child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def resolve_local_path(raw_path: str, *, base_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return base_dir / path


def is_finite_number(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def finite_float(value: Any, *, context: str) -> float:
    if not is_finite_number(value):
        raise Gate12C1SummaryContractError(f"{context} must be a finite number")
    return float(value)


def optional_finite_float(value: Any) -> float | None:
    if value is None:
        return None
    if not is_finite_number(value):
        return None
    return float(value)


def median_or_none(values: Sequence[float]) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return None
    return float(statistics.median(clean))


def quantiles_or_none(values: Sequence[float]) -> Dict[str, float | None]:
    clean = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=np.float64)
    if clean.size == 0:
        return {"min": None, "p25": None, "median": None, "p75": None, "max": None}
    return {
        "min": float(np.min(clean)),
        "p25": float(np.quantile(clean, 0.25)),
        "median": float(np.quantile(clean, 0.50)),
        "p75": float(np.quantile(clean, 0.75)),
        "max": float(np.max(clean)),
    }


def sign_with_tolerance(value: Any) -> int | None:
    if not is_finite_number(value):
        return None
    score = float(value)
    if score > PRIMARY_ZERO_TOLERANCE:
        return 1
    if score < -PRIMARY_ZERO_TOLERANCE:
        return -1
    return 0


def source_block_status(node_ids: Sequence[str]) -> Tuple[str, str]:
    sample_ids: List[str] = []
    for node_id in node_ids:
        match = re.match(r"^(sample_\d{6})(?:$|[:/_-])", str(node_id))
        if not match:
            return "mixed_or_undefined", "mixed_or_undefined"
        sample_ids.append(match.group(1))
    if len(set(sample_ids)) == 1:
        return sample_ids[0], "single_sample"
    return "mixed_or_undefined", "mixed_or_undefined"


def edge_compatibility_gap_max(edges: Sequence[Mapping[str, Any]]) -> float:
    values: List[float] = []
    for edge in edges:
        raw = edge.get("compatibility_gap_fro")
        if raw is not None:
            values.append(finite_float(raw, context="compatibility_gap_fro"))
    return float(max(values) if values else 0.0)


def validate_case_manifest_payload(manifest: Mapping[str, Any], *, base_dir: Path) -> List[CaseInput]:
    if manifest.get("schema_version") != CASE_MANIFEST_SCHEMA_VERSION:
        raise Gate12C1SummaryContractError(
            f"case manifest schema_version must be {CASE_MANIFEST_SCHEMA_VERSION}"
        )
    if manifest.get("plan_id") != PLAN_ID:
        raise Gate12C1SummaryContractError(f"case manifest plan_id must be {PLAN_ID}")
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise Gate12C1SummaryContractError("case manifest cases must be a list")
    if len(cases) != len(CANONICAL_CASES):
        raise Gate12C1SummaryContractError("case manifest must contain exactly 12 cases")

    seen_ids: set[str] = set()
    seen_orders: set[int] = set()
    inputs: List[CaseInput] = []
    for index, row in enumerate(cases):
        if not isinstance(row, Mapping):
            raise Gate12C1SummaryContractError(f"case row {index + 1} must be an object")
        case_id = str(row.get("case_id") or "")
        if case_id in seen_ids:
            raise Gate12C1SummaryContractError(f"duplicate case_id in manifest: {case_id}")
        seen_ids.add(case_id)
        try:
            case_order = int(row.get("case_order"))
        except (TypeError, ValueError) as exc:
            raise Gate12C1SummaryContractError(
                f"case {case_id or index + 1} has invalid case_order"
            ) from exc
        if case_order in seen_orders:
            raise Gate12C1SummaryContractError(f"duplicate case_order in manifest: {case_order}")
        seen_orders.add(case_order)

        expected = CANONICAL_CASES[index]
        if case_id != expected.case_id or case_order != expected.case_order:
            raise Gate12C1SummaryContractError(
                "case manifest ordering must exactly match case_01 through case_12"
            )
        canonical = CANONICAL_BY_ID.get(case_id)
        if canonical is None or CANONICAL_BY_ORDER.get(case_order) != canonical:
            raise Gate12C1SummaryContractError(f"case {case_id} is not canonical")
        if str(row.get("model") or "") != canonical.model:
            raise Gate12C1SummaryContractError(f"case {case_id} model does not match canonical grid")
        if str(row.get("family") or "") != canonical.family:
            raise Gate12C1SummaryContractError(f"case {case_id} family does not match canonical grid")
        if str(row.get("expected_source_gate12a_run_id") or "") != canonical.source_gate12a_run_id:
            raise Gate12C1SummaryContractError(
                f"case {case_id} expected_source_gate12a_run_id does not match canonical grid"
            )
        if int(row.get("preflight_eligible_cycle_count", -1)) != int(
            canonical.preflight_eligible_cycle_count
        ):
            raise Gate12C1SummaryContractError(
                f"case {case_id} preflight_eligible_cycle_count does not match canonical grid"
            )
        for key in ("source_gate12a_dir", "gate12c1_run_dir"):
            if not isinstance(row.get(key), str) or not str(row.get(key)):
                raise Gate12C1SummaryContractError(f"case {case_id} missing {key}")

        inputs.append(
            CaseInput(
                spec=canonical,
                source_gate12a_dir=resolve_local_path(
                    str(row["source_gate12a_dir"]), base_dir=base_dir
                ),
                gate12c1_run_dir=resolve_local_path(
                    str(row["gate12c1_run_dir"]), base_dir=base_dir
                ),
            )
        )
    return inputs


def validate_summary_output_directory(*, out_dir: Path, case_inputs: Sequence[CaseInput]) -> None:
    target = Path(out_dir).resolve(strict=False)
    for case in case_inputs:
        for label, input_dir in (
            ("Gate12A source", case.source_gate12a_dir),
            ("Gate12C-1 run", case.gate12c1_run_dir),
        ):
            source = Path(input_dir).resolve(strict=False)
            if target == source:
                raise Gate12C1SummaryContractError(
                    f"summary out_dir must not alias {label} directory for {case.spec.case_id}"
                )
            if path_is_relative_to(child=target, parent=source):
                raise Gate12C1SummaryContractError(
                    f"summary out_dir must not be nested under {label} directory for {case.spec.case_id}"
                )
            if path_is_relative_to(child=source, parent=target):
                raise Gate12C1SummaryContractError(
                    f"summary out_dir must not contain {label} directory for {case.spec.case_id}"
                )
        if case.source_gate12a_dir.resolve(strict=False) == case.gate12c1_run_dir.resolve(
            strict=False
        ):
            raise Gate12C1SummaryContractError(
                f"case {case.spec.case_id} source_gate12a_dir and gate12c1_run_dir must differ"
            )


def load_case_manifest_and_validate_paths(
    case_manifest_path: Path,
    out_dir: Path,
) -> Tuple[Dict[str, Any], List[CaseInput]]:
    manifest = read_json(case_manifest_path)
    case_inputs = validate_case_manifest_payload(manifest, base_dir=case_manifest_path.parent)
    validate_summary_output_directory(out_dir=out_dir, case_inputs=case_inputs)
    return manifest, case_inputs


def verify_run_checksums(run_dir: Path) -> None:
    missing = [name for name in REQUIRED_GATE12C1_FILES if not (run_dir / name).exists()]
    if missing:
        raise Gate12C1SummaryContractError(
            f"Gate12C-1 run {run_dir} missing required files: {missing}"
        )
    checksums = read_json(run_dir / "checksums.json")
    if not isinstance(checksums, Mapping):
        raise Gate12C1SummaryContractError(f"{run_dir / 'checksums.json'} must be an object")
    for name in RUNNER_OUTPUT_FILES_FOR_CHECKSUMS:
        if name not in checksums:
            raise Gate12C1SummaryContractError(f"checksums.json missing {name} in {run_dir}")
        actual = sha256_file(run_dir / name)
        if str(checksums[name]) != actual:
            raise Gate12C1SummaryContractError(
                f"checksum mismatch for {name} in {run_dir}: expected {checksums[name]} got {actual}"
            )


def hash_required_files(root_dir: Path, required_files: Sequence[str]) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for name in required_files:
        path = root_dir / name
        if not path.exists():
            raise Gate12C1SummaryContractError(f"required input file missing: {path}")
        hashes[name] = sha256_file(path)
    return hashes


def verify_source_gate12a_checksums(source_dir: Path) -> None:
    checksums_path = source_dir / "checksums.json"
    if not checksums_path.exists():
        raise Gate12C1SummaryContractError(
            f"Gate12A source {source_dir} missing required checksums.json"
        )
    checksums = read_json(checksums_path)
    if not isinstance(checksums, Mapping):
        raise Gate12C1SummaryContractError(f"{checksums_path} must be an object")
    for name in gate12c0.REQUIRED_FILES:
        if name not in checksums:
            raise Gate12C1SummaryContractError(
                f"Gate12A source checksums.json missing {name} in {source_dir}"
            )
        actual = sha256_file(source_dir / name)
        if str(checksums[name]) != actual:
            raise Gate12C1SummaryContractError(
                f"Gate12A source checksum mismatch for {name} in {source_dir}: "
                f"expected {checksums[name]} got {actual}"
            )


def snapshot_input_hashes(case_inputs: Sequence[CaseInput]) -> Dict[str, Dict[str, Dict[str, str]]]:
    snapshots: Dict[str, Dict[str, Dict[str, str]]] = {}
    for case in case_inputs:
        snapshots[case.spec.case_id] = {
            "source_gate12a": hash_required_files(
                case.source_gate12a_dir,
                gate12c0.REQUIRED_FILES,
            ),
            "gate12c1_output": hash_required_files(
                case.gate12c1_run_dir,
                REQUIRED_GATE12C1_FILES,
            ),
        }
    return snapshots


def verify_input_immutability(
    *,
    case_inputs: Sequence[CaseInput],
    before: Mapping[str, Mapping[str, Mapping[str, str]]],
) -> None:
    after = snapshot_input_hashes(case_inputs)
    if dict(before) != dict(after):
        raise Gate12C1SummaryContractError(
            "Gate12A source or Gate12C-1 output input files changed during summary"
        )


def require_equal(
    actual: Any,
    expected: Any,
    *,
    context: str,
) -> None:
    if actual != expected:
        raise Gate12C1SummaryContractError(
            f"{context} must be {expected!r}, got {actual!r}"
        )


def validate_runner_manifest(
    *,
    case: CaseInput,
    runner_manifest: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> None:
    require_equal(
        runner_manifest.get("schema_version"),
        EXPECTED_RUNNER_SCHEMA,
        context=f"{case.spec.case_id} runner schema_version",
    )
    require_equal(
        runner_manifest.get("method_id"),
        EXPECTED_RUNNER_METHOD,
        context=f"{case.spec.case_id} runner method_id",
    )
    require_equal(
        runner_manifest.get("code_git_commit"),
        EXPECTED_RUNNER_COMMIT,
        context=f"{case.spec.case_id} runner code_git_commit",
    )
    require_equal(
        runner_manifest.get("builder_script_sha256"),
        EXPECTED_RUNNER_SCRIPT_SHA256,
        context=f"{case.spec.case_id} runner builder_script_sha256",
    )
    require_equal(
        runner_manifest.get("run_mode"),
        EXPECTED_RUN_MODE,
        context=f"{case.spec.case_id} runner run_mode",
    )
    require_equal(
        runner_manifest.get("orientation_null_mode"),
        EXPECTED_ORIENTATION_NULL_MODE,
        context=f"{case.spec.case_id} orientation_null_mode",
    )
    require_equal(
        runner_manifest.get("orientation_null_orthogonal_generator"),
        EXPECTED_ORIENTATION_NULL_GENERATOR,
        context=f"{case.spec.case_id} orientation_null_orthogonal_generator",
    )
    require_equal(
        runner_manifest.get("orientation_seed_encoding"),
        EXPECTED_ORIENTATION_SEED_ENCODING,
        context=f"{case.spec.case_id} orientation_seed_encoding",
    )
    require_equal(
        runner_manifest.get("orientation_null_seed"),
        EXPECTED_ORIENTATION_NULL_SEED,
        context=f"{case.spec.case_id} orientation_null_seed",
    )
    require_equal(
        runner_manifest.get("orientation_null_requested_draw_count"),
        EXPECTED_ORIENTATION_NULL_REQUESTED_DRAW_COUNT,
        context=f"{case.spec.case_id} orientation_null_requested_draw_count",
    )
    require_equal(
        runner_manifest.get("orientation_null_max_attempt_count"),
        EXPECTED_ORIENTATION_NULL_MAX_ATTEMPT_COUNT,
        context=f"{case.spec.case_id} orientation_null_max_attempt_count",
    )
    require_equal(
        dict(runner_manifest.get("tolerances") or {}),
        EXPECTED_TOLERANCES,
        context=f"{case.spec.case_id} runner tolerances",
    )
    require_equal(
        runner_manifest.get("source_gate12a_run_id"),
        case.spec.source_gate12a_run_id,
        context=f"{case.spec.case_id} source_gate12a_run_id",
    )
    require_equal(
        source_manifest.get("run_id"),
        case.spec.source_gate12a_run_id,
        context=f"{case.spec.case_id} source manifest run_id",
    )

    claim_boundary = runner_manifest.get("claim_boundary")
    if not isinstance(claim_boundary, Mapping):
        raise Gate12C1SummaryContractError(
            f"{case.spec.case_id} runner manifest missing claim_boundary"
        )
    for key in (
        "gate12b_overlay_used",
        "type_iii_claim_authorized",
        "scientific_null_excess_threshold_defined",
        "rectangular_rank_mismatch_supported",
    ):
        if claim_boundary.get(key) is not False:
            raise Gate12C1SummaryContractError(
                f"{case.spec.case_id} claim_boundary {key} must be false"
            )


def validate_run_status(*, case: CaseInput, status: Mapping[str, Any]) -> None:
    require_equal(
        status.get("schema_version"),
        EXPECTED_RUNNER_SCHEMA,
        context=f"{case.spec.case_id} status schema_version",
    )
    require_equal(
        status.get("method_id"),
        EXPECTED_RUNNER_METHOD,
        context=f"{case.spec.case_id} status method_id",
    )
    require_equal(
        status.get("process_status"),
        "pass",
        context=f"{case.spec.case_id} process_status",
    )


def derive_expected_cycles(case: CaseInput) -> Tuple[List[ExpectedCycle], Dict[str, ExpectedCycle]]:
    artifacts = gate12c0.load_gate12a_artifacts(case.source_gate12a_dir)
    edge_reconstructions, edge_diagnostics = gate12c0.reconstruct_edges(
        artifacts=artifacts,
        tau_overlap_sv_min=EXPECTED_TOLERANCES["tau_overlap_sv_min"],
        tau_overlap_sv_abs_error=EXPECTED_TOLERANCES[
            "tau_overlap_singular_value_abs_error"
        ],
        tau_transport_reconstruction_fro=EXPECTED_TOLERANCES[
            "tau_transport_reconstruction_fro"
        ],
    )
    if int(edge_diagnostics["failed_edge_reconstruction_count"]) > 0:
        raise Gate12C1SummaryContractError(
            f"{case.spec.case_id} Gate12A edge reconstruction validation failed"
        )

    expected_cycles: List[ExpectedCycle] = []
    for cycle in sorted(artifacts.cycle_rows, key=lambda row: str(row.get("cycle_id") or "")):
        gate12c0.require_keys(
            cycle,
            ("cycle_id", "base_node_id", "edge_id_path", "node_id_path"),
            "explicit_triangle_cycle_registry row",
        )
        cycle_id = str(cycle["cycle_id"])
        ordered_edges = gate12c0.reconstruct_ordered_edges(cycle=cycle, edge_map=artifacts.edge_map)
        relation_path = [str(edge["relation_kind"]) for edge in ordered_edges]
        if sum(1 for kind in relation_path if kind == "residual_chord") <= 0:
            continue
        holonomy = artifacts.holonomy_map.get(cycle_id, {})
        if str(holonomy.get("holonomy_status") or "missing") != "defined":
            continue
        gate12c0.require_keys(
            holonomy,
            ("holonomy_residual_fro",),
            "triangle_holonomy_registry defined row",
        )
        holonomy_residual = finite_float(
            holonomy.get("holonomy_residual_fro"),
            context=f"{case.spec.case_id} cycle {cycle_id} holonomy_residual_fro",
        )
        node_ids = [str(node_id) for node_id in list(cycle["node_id_path"])[:3]]
        if any(node_id not in artifacts.node_map for node_id in node_ids):
            continue
        node_ranks = [int(artifacts.node_map[node_id].projector_rank) for node_id in node_ids]
        common_rank = node_ranks[0] if len(set(node_ranks)) == 1 else 0
        transport_cases = [str(edge["transport_case"]) for edge in ordered_edges]
        if common_rank < 2 or any(item != "equal_rank_orthogonal" for item in transport_cases):
            continue
        if common_rank != 3:
            raise Gate12C1SummaryContractError(
                f"{case.spec.case_id} cycle {cycle_id} has canonical-grid rank {common_rank}; expected 3"
            )
        ordered_edge_ids = [str(edge["edge_id"]) for edge in ordered_edges]
        for edge_id in ordered_edge_ids:
            reconstruction = edge_reconstructions[edge_id]
            if reconstruction.overlap_matrix is None:
                raise Gate12C1SummaryContractError(
                    f"{case.spec.case_id} eligible cycle {cycle_id} has undefined overlap {edge_id}"
                )
            matrix = np.asarray(reconstruction.overlap_matrix, dtype=np.float64)
            if matrix.shape != (common_rank, common_rank):
                raise Gate12C1SummaryContractError(
                    f"{case.spec.case_id} eligible cycle {cycle_id} edge {edge_id} shape mismatch"
                )
        block_id, block_status = source_block_status(node_ids)
        expected_cycles.append(
            ExpectedCycle(
                case_id=case.spec.case_id,
                cycle_id=cycle_id,
                source_sample_block_id=block_id,
                source_block_status=block_status,
                gate12a_holonomy_residual_fro=holonomy_residual,
                edge_compatibility_gap_max=edge_compatibility_gap_max(ordered_edges),
                cycle_rank=common_rank,
            )
        )

    if len(expected_cycles) != int(case.spec.preflight_eligible_cycle_count):
        raise Gate12C1SummaryContractError(
            f"{case.spec.case_id} derived {len(expected_cycles)} eligible cycles; "
            f"expected {case.spec.preflight_eligible_cycle_count}"
        )
    by_id: Dict[str, ExpectedCycle] = {}
    for cycle in expected_cycles:
        if cycle.cycle_id in by_id:
            raise Gate12C1SummaryContractError(
                f"{case.spec.case_id} duplicate eligible cycle_id: {cycle.cycle_id}"
            )
        by_id[cycle.cycle_id] = cycle
    return expected_cycles, by_id


def validate_npz_arrays(run_dir: Path, *, expected_row_count: int) -> None:
    with np.load(run_dir / "triangle_associator_arrays.npz") as handle:
        missing = [
            key
            for key in (
                "compressed_overlap_left_operator",
                "compressed_overlap_right_operator",
                "compressed_overlap_associator_operator",
            )
            if key not in handle.files
        ]
        if missing:
            raise Gate12C1SummaryContractError(
                f"{run_dir / 'triangle_associator_arrays.npz'} missing arrays: {missing}"
            )
        for key in (
            "compressed_overlap_left_operator",
            "compressed_overlap_right_operator",
            "compressed_overlap_associator_operator",
        ):
            array = np.asarray(handle[key])
            if array.ndim < 1 or int(array.shape[0]) != int(expected_row_count):
                raise Gate12C1SummaryContractError(
                    f"{run_dir} array {key} first dimension must match registry row count"
                )


def validate_and_load_registry(
    *,
    case: CaseInput,
    expected_by_cycle: Mapping[str, ExpectedCycle],
) -> List[Dict[str, Any]]:
    rows = read_jsonl(case.gate12c1_run_dir / "triangle_associator_registry.jsonl")
    validate_npz_arrays(case.gate12c1_run_dir, expected_row_count=len(rows))

    seen_keys: set[Tuple[str, int, int]] = set()
    seen_indices: set[int] = set()
    block_by_cycle: Dict[str, Tuple[str, str]] = {}
    holonomy_by_cycle: Dict[str, float] = {}
    edge_gap_by_cycle_q: Dict[Tuple[str, int], float] = {}
    filtered_rows: List[Dict[str, Any]] = []

    for expected_index, raw_row in enumerate(rows):
        gate12c0.require_keys(
            raw_row,
            RUNNER_REGISTRY_REQUIRED_FIELDS,
            f"{case.spec.case_id} triangle_associator_registry row",
        )
        row = {key: raw_row[key] for key in RUNNER_REGISTRY_REQUIRED_FIELDS}
        cycle_id = str(row["cycle_id"])
        if cycle_id not in expected_by_cycle:
            raise Gate12C1SummaryContractError(
                f"{case.spec.case_id} registry contains unexpected cycle_id {cycle_id}"
            )
        expected_cycle = expected_by_cycle[cycle_id]
        root = int(row["root_rotation_index"])
        q = int(row["compression_rank_q"])
        if root not in (0, 1, 2):
            raise Gate12C1SummaryContractError(
                f"{case.spec.case_id} cycle {cycle_id} invalid root_rotation_index {root}"
            )
        if q not in (1, 2):
            raise Gate12C1SummaryContractError(
                f"{case.spec.case_id} cycle {cycle_id} invalid q {q}"
            )
        if int(row["cycle_rank"]) != 3:
            raise Gate12C1SummaryContractError(
                f"{case.spec.case_id} cycle {cycle_id} cycle_rank must be 3"
            )
        key = (cycle_id, root, q)
        if key in seen_keys:
            raise Gate12C1SummaryContractError(
                f"{case.spec.case_id} duplicate registry key {key}"
            )
        seen_keys.add(key)
        operator_index = int(row["operator_array_index"])
        if operator_index != expected_index:
            raise Gate12C1SummaryContractError(
                f"{case.spec.case_id} operator_array_index must equal row order"
            )
        if operator_index in seen_indices:
            raise Gate12C1SummaryContractError(
                f"{case.spec.case_id} duplicate operator_array_index {operator_index}"
            )
        seen_indices.add(operator_index)

        block_pair = (str(row["source_sample_block_id"]), str(row["source_block_status"]))
        if block_pair != (
            expected_cycle.source_sample_block_id,
            expected_cycle.source_block_status,
        ):
            raise Gate12C1SummaryContractError(
                f"{case.spec.case_id} cycle {cycle_id} source block provenance mismatch"
            )
        previous_block = block_by_cycle.setdefault(cycle_id, block_pair)
        if previous_block != block_pair:
            raise Gate12C1SummaryContractError(
                f"{case.spec.case_id} cycle {cycle_id} has inconsistent source block"
            )

        holonomy = finite_float(
            row["gate12a_holonomy_residual_fro"],
            context=f"{case.spec.case_id} cycle {cycle_id} gate12a_holonomy_residual_fro",
        )
        if holonomy != expected_cycle.gate12a_holonomy_residual_fro:
            raise Gate12C1SummaryContractError(
                f"{case.spec.case_id} cycle {cycle_id} holonomy residual mismatch"
            )
        previous_holonomy = holonomy_by_cycle.setdefault(cycle_id, holonomy)
        if previous_holonomy != holonomy:
            raise Gate12C1SummaryContractError(
                f"{case.spec.case_id} cycle {cycle_id} has inconsistent holonomy residual"
            )

        edge_gap = finite_float(
            row["edge_compatibility_gap_max"],
            context=f"{case.spec.case_id} cycle {cycle_id} edge_compatibility_gap_max",
        )
        if edge_gap != expected_cycle.edge_compatibility_gap_max:
            raise Gate12C1SummaryContractError(
                f"{case.spec.case_id} cycle {cycle_id} edge compatibility mismatch"
            )
        gap_key = (cycle_id, q)
        previous_gap = edge_gap_by_cycle_q.setdefault(gap_key, edge_gap)
        if previous_gap != edge_gap:
            raise Gate12C1SummaryContractError(
                f"{case.spec.case_id} cycle {cycle_id} q {q} has inconsistent edge gap"
            )
        if int(row["orientation_null_requested_draw_count"]) != (
            EXPECTED_ORIENTATION_NULL_REQUESTED_DRAW_COUNT
        ):
            raise Gate12C1SummaryContractError(
                f"{case.spec.case_id} registry row has wrong orientation_null_requested_draw_count"
            )
        row["_case_id"] = case.spec.case_id
        row["_case_order"] = case.spec.case_order
        row["_model"] = case.spec.model
        row["_family"] = case.spec.family
        filtered_rows.append(row)

    if seen_indices != set(range(len(rows))):
        raise Gate12C1SummaryContractError(
            f"{case.spec.case_id} operator_array_index values must be contiguous"
        )
    return filtered_rows


def primary_row_metrics(row: Mapping[str, Any]) -> Dict[str, Any]:
    assoc = optional_finite_float(row.get("compressed_overlap_associator_fro"))
    null_median = optional_finite_float(row.get("orientation_null_median"))
    holonomy = optional_finite_float(row.get("gate12a_holonomy_residual_fro"))
    edge_gap = optional_finite_float(row.get("edge_compatibility_gap_max"))
    log_ratio = None
    if assoc is not None and null_median is not None:
        log_ratio = float(math.log((assoc + PRIMARY_EPSILON) / (null_median + PRIMARY_EPSILON)))
    eligible = bool(
        row.get("aggregation_eligible") is True
        and row.get("orientation_null_status") == "complete"
        and row.get("orientation_null_scale_degenerate") is False
        and null_median is not None
        and null_median > PRIMARY_EPSILON
        and assoc is not None
        and holonomy is not None
        and edge_gap is not None
        and log_ratio is not None
        and math.isfinite(log_ratio)
    )
    return {
        "assoc": assoc,
        "assoc_rel": optional_finite_float(row.get("compressed_overlap_associator_rel")),
        "null_median": null_median,
        "log_null_ratio": log_ratio,
        "robust_z": optional_finite_float(row.get("orientation_null_robust_z")),
        "empirical_p_upper": optional_finite_float(row.get("orientation_null_empirical_p_upper")),
        "holonomy": holonomy,
        "edge_gap": edge_gap,
        "eligible": eligible,
        "scale_degenerate": row.get("orientation_null_scale_degenerate") is True,
        "incomplete_null": row.get("orientation_null_status") != "complete",
    }


def build_cycle_q_scores(
    *,
    case_inputs: Sequence[CaseInput],
    expected_cycles_by_case: Mapping[str, List[ExpectedCycle]],
    rows_by_case: Mapping[str, List[Mapping[str, Any]]],
) -> List[Dict[str, Any]]:
    rows_by_case_cycle_q: Dict[Tuple[str, str, int], List[Mapping[str, Any]]] = {}
    for case_id, rows in rows_by_case.items():
        for row in rows:
            key = (case_id, str(row["cycle_id"]), int(row["compression_rank_q"]))
            rows_by_case_cycle_q.setdefault(key, []).append(row)

    output: List[Dict[str, Any]] = []
    for case in case_inputs:
        expected_cycles = expected_cycles_by_case[case.spec.case_id]
        for cycle in expected_cycles:
            for q in (1, 2):
                root_rows = rows_by_case_cycle_q.get((case.spec.case_id, cycle.cycle_id, q), [])
                metrics = [primary_row_metrics(row) for row in root_rows]
                root_set = {int(row["root_rotation_index"]) for row in root_rows}
                valid = (
                    len(root_rows) == 3
                    and root_set == {0, 1, 2}
                    and all(metric["eligible"] for metric in metrics)
                    and cycle.source_block_status == "single_sample"
                )
                status_reasons: List[str] = []
                if len(root_rows) != 3 or root_set != {0, 1, 2}:
                    status_reasons.append("missing_or_malformed_roots")
                if root_rows and not all(metric["eligible"] for metric in metrics):
                    status_reasons.append("primary_ineligible_root")
                if cycle.source_block_status != "single_sample":
                    status_reasons.append("mixed_or_undefined_block")
                log_values = [
                    float(metric["log_null_ratio"])
                    for metric in metrics
                    if metric["log_null_ratio"] is not None
                ]
                robust_z_values = [
                    float(metric["robust_z"]) for metric in metrics if metric["robust_z"] is not None
                ]
                assoc_rel_values = [
                    float(metric["assoc_rel"]) for metric in metrics if metric["assoc_rel"] is not None
                ]
                cycle_score = median_or_none(log_values) if valid else None
                output.append(
                    {
                        "case_id": case.spec.case_id,
                        "case_order": case.spec.case_order,
                        "model": case.spec.model,
                        "family": case.spec.family,
                        "cycle_id": cycle.cycle_id,
                        "compression_rank_q": q,
                        "source_sample_block_id": cycle.source_sample_block_id,
                        "source_block_status": cycle.source_block_status,
                        "expected_root_count": 3,
                        "observed_root_count": len(root_rows),
                        "valid_root_count": int(sum(1 for metric in metrics if metric["eligible"])),
                        "cycle_q_primary_valid": bool(valid),
                        "coverage_status": "valid" if valid else "|".join(status_reasons or ["invalid"]),
                        "cycle_q_log_null_ratio": cycle_score,
                        "cycle_q_robust_z_median": median_or_none(robust_z_values)
                        if valid
                        else None,
                        "cycle_q_associator_rel_median": median_or_none(assoc_rel_values)
                        if valid
                        else None,
                        "cycle_q_root_spread": float(max(log_values) - min(log_values))
                        if valid and len(log_values) == 3
                        else None,
                        "gate12a_holonomy_residual_fro": cycle.gate12a_holonomy_residual_fro,
                        "edge_compatibility_gap_max": cycle.edge_compatibility_gap_max,
                    }
                )
    return output


def build_block_q_scores(
    *,
    case_inputs: Sequence[CaseInput],
    expected_cycles_by_case: Mapping[str, List[ExpectedCycle]],
    cycle_q_scores: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    valid_cycle_rows: Dict[Tuple[str, str, int], List[Mapping[str, Any]]] = {}
    for row in cycle_q_scores:
        if row.get("cycle_q_primary_valid") is True:
            key = (
                str(row["case_id"]),
                str(row["source_sample_block_id"]),
                int(row["compression_rank_q"]),
            )
            valid_cycle_rows.setdefault(key, []).append(row)

    output: List[Dict[str, Any]] = []
    for case in case_inputs:
        block_to_expected_cycles: Dict[str, List[ExpectedCycle]] = {}
        for cycle in expected_cycles_by_case[case.spec.case_id]:
            if cycle.source_block_status == "single_sample":
                block_to_expected_cycles.setdefault(cycle.source_sample_block_id, []).append(cycle)
        for block_id in sorted(block_to_expected_cycles):
            for q in (1, 2):
                valid_rows = valid_cycle_rows.get((case.spec.case_id, block_id, q), [])
                scores = [
                    float(row["cycle_q_log_null_ratio"])
                    for row in valid_rows
                    if is_finite_number(row.get("cycle_q_log_null_ratio"))
                ]
                output.append(
                    {
                        "case_id": case.spec.case_id,
                        "case_order": case.spec.case_order,
                        "model": case.spec.model,
                        "family": case.spec.family,
                        "source_sample_block_id": block_id,
                        "compression_rank_q": q,
                        "expected_cycle_count_in_block": len(block_to_expected_cycles[block_id]),
                        "valid_cycle_count_in_block": len(scores),
                        "block_q_score": median_or_none(scores),
                    }
                )
    return output


def exact_one_sided_sign_p(*, positive_count: int, negative_count: int) -> Tuple[str, float]:
    n = int(positive_count) + int(negative_count)
    if n == 0:
        return "non_informative", 1.0
    numerator = sum(math.comb(n, k) for k in range(int(positive_count), n + 1))
    denominator = 2**n
    return "informative", float(numerator / denominator)


def build_run_q_tests(
    *,
    case_inputs: Sequence[CaseInput],
    expected_cycles_by_case: Mapping[str, List[ExpectedCycle]],
    cycle_q_scores: Sequence[Mapping[str, Any]],
    block_q_scores: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    cycle_rows_by_case_q: Dict[Tuple[str, int], List[Mapping[str, Any]]] = {}
    for row in cycle_q_scores:
        cycle_rows_by_case_q.setdefault(
            (str(row["case_id"]), int(row["compression_rank_q"])),
            [],
        ).append(row)
    block_rows_by_case_q: Dict[Tuple[str, int], List[Mapping[str, Any]]] = {}
    for row in block_q_scores:
        block_rows_by_case_q.setdefault(
            (str(row["case_id"]), int(row["compression_rank_q"])),
            [],
        ).append(row)

    endpoint_rows: List[Dict[str, Any]] = []
    for case in case_inputs:
        expected_cycles = expected_cycles_by_case[case.spec.case_id]
        expected_blocks = sorted(
            {
                cycle.source_sample_block_id
                for cycle in expected_cycles
                if cycle.source_block_status == "single_sample"
            }
        )
        mixed_count = sum(
            1 for cycle in expected_cycles if cycle.source_block_status != "single_sample"
        )
        for q in (1, 2):
            cycle_rows = cycle_rows_by_case_q.get((case.spec.case_id, q), [])
            valid_cycle_count = sum(
                1 for row in cycle_rows if row.get("cycle_q_primary_valid") is True
            )
            cycle_ratio = (
                float(valid_cycle_count / case.spec.preflight_eligible_cycle_count)
                if case.spec.preflight_eligible_cycle_count > 0
                else 0.0
            )
            block_rows = block_rows_by_case_q.get((case.spec.case_id, q), [])
            block_scores = [
                float(row["block_q_score"])
                for row in block_rows
                if is_finite_number(row.get("block_q_score"))
            ]
            represented_blocks = [
                str(row["source_sample_block_id"])
                for row in block_rows
                if is_finite_number(row.get("block_q_score"))
            ]
            block_ratio = (
                float(len(set(represented_blocks)) / len(expected_blocks))
                if expected_blocks
                else 0.0
            )
            signs = [sign_with_tolerance(score) for score in block_scores]
            positive_count = sum(1 for sign in signs if sign == 1)
            negative_count = sum(1 for sign in signs if sign == -1)
            tie_count = sum(1 for sign in signs if sign == 0)
            test_status, raw_p = exact_one_sided_sign_p(
                positive_count=positive_count,
                negative_count=negative_count,
            )
            cycle_coverage_pass = bool(cycle_ratio >= COVERAGE_THRESHOLD)
            block_coverage_pass = bool(expected_blocks and block_ratio >= COVERAGE_THRESHOLD)
            coverage_pass = bool(
                cycle_coverage_pass and block_coverage_pass and mixed_count == 0
            )
            if not coverage_pass:
                raw_p = 1.0
            endpoint_rows.append(
                {
                    "case_id": case.spec.case_id,
                    "case_order": case.spec.case_order,
                    "model": case.spec.model,
                    "family": case.spec.family,
                    "compression_rank_q": q,
                    "expected_cycle_count": int(case.spec.preflight_eligible_cycle_count),
                    "represented_cycle_count": int(valid_cycle_count),
                    "cycle_coverage_ratio": cycle_ratio,
                    "cycle_coverage_pass": cycle_coverage_pass,
                    "expected_block_count": int(len(expected_blocks)),
                    "represented_block_count": int(len(set(represented_blocks))),
                    "block_coverage_ratio": block_ratio,
                    "block_coverage_pass": block_coverage_pass,
                    "mixed_or_undefined_expected_cycle_count": int(mixed_count),
                    "coverage_pass": coverage_pass,
                    "run_q_median": median_or_none(block_scores),
                    "positive_block_count": int(positive_count),
                    "negative_block_count": int(negative_count),
                    "tie_block_count": int(tie_count),
                    "test_status": test_status,
                    "raw_p": raw_p,
                    "holm_adjusted_p": None,
                    "holm_sort_position": None,
                    "q_support": False,
                    "run_support": False,
                    "q_discordant_run": None,
                }
            )

    apply_holm(endpoint_rows)
    run_by_case: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for row in endpoint_rows:
        q = int(row["compression_rank_q"])
        row["q_support"] = bool(
            row["coverage_pass"] is True
            and row["test_status"] == "informative"
            and is_finite_number(row.get("run_q_median"))
            and float(row["run_q_median"]) > PRIMARY_ZERO_TOLERANCE
            and is_finite_number(row.get("holm_adjusted_p"))
            and float(row["holm_adjusted_p"]) < HOLM_ALPHA
        )
        run_by_case.setdefault(str(row["case_id"]), {})[q] = row
    for case_id, by_q in run_by_case.items():
        q1 = by_q[1]
        q2 = by_q[2]
        run_support = bool(q1["q_support"] and q2["q_support"])
        sign_q1 = sign_with_tolerance(q1.get("run_q_median"))
        sign_q2 = sign_with_tolerance(q2.get("run_q_median"))
        q_discordant = None
        if sign_q1 is not None and sign_q2 is not None:
            q_discordant = bool(
                bool(q1["q_support"]) != bool(q2["q_support"]) or sign_q1 != sign_q2
            )
        for row in (q1, q2):
            row["run_support"] = run_support
            row["q_discordant_run"] = q_discordant
    return endpoint_rows


def apply_holm(endpoint_rows: List[Dict[str, Any]]) -> None:
    if len(endpoint_rows) != HOLM_TEST_COUNT:
        raise Gate12C1SummaryContractError("Holm correction requires exactly 24 endpoints")
    ordered = sorted(
        enumerate(endpoint_rows),
        key=lambda pair: (
            float(pair[1]["raw_p"]),
            int(pair[1]["case_order"]),
            int(pair[1]["compression_rank_q"]),
        ),
    )
    running = 0.0
    for position, (original_index, row) in enumerate(ordered, start=1):
        multiplier = HOLM_TEST_COUNT - position + 1
        candidate = min(1.0, float(row["raw_p"]) * multiplier)
        running = max(running, candidate)
        endpoint_rows[original_index]["holm_adjusted_p"] = float(running)
        endpoint_rows[original_index]["holm_sort_position"] = int(position)


def classify_grid_outcome(
    *,
    execution_status: str,
    run_q_tests: Sequence[Mapping[str, Any]],
    case_inputs: Sequence[CaseInput],
) -> Dict[str, Any]:
    if execution_status != "complete":
        return {
            "execution_status": execution_status,
            "grid_outcome": "not_classified",
            "supporting_run_count": 0,
            "q_discordant_run_count": 0,
        }
    if len(run_q_tests) != HOLM_TEST_COUNT:
        raise Gate12C1SummaryContractError("grid outcome requires exactly 24 run/q rows")
    if any(
        row.get("cycle_coverage_pass") is not True
        or row.get("block_coverage_pass") is not True
        or row.get("test_status") == "non_informative"
        for row in run_q_tests
    ):
        return summarize_outcome(
            run_q_tests=run_q_tests,
            case_inputs=case_inputs,
            grid_outcome="coverage_limited",
        )
    run_rows = run_level_rows(run_q_tests)
    q_discordant_count = sum(1 for row in run_rows if row["q_discordant_run"] is True)
    if q_discordant_count >= 6:
        return summarize_outcome(
            run_q_tests=run_q_tests,
            case_inputs=case_inputs,
            grid_outcome="mixed_q",
        )
    support_count = sum(1 for row in run_rows if row["run_support"] is True)
    if support_count == 12:
        outcome = "strong_broad"
    elif support_count >= 10 and breadth_constraints_pass(run_rows):
        outcome = "broad_replicated"
    elif support_count == 0:
        outcome = "no_directional_support"
    else:
        outcome = "partial_or_structured"
    return summarize_outcome(run_q_tests=run_q_tests, case_inputs=case_inputs, grid_outcome=outcome)


def run_level_rows(run_q_tests: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_case: Dict[str, List[Mapping[str, Any]]] = {}
    for row in run_q_tests:
        by_case.setdefault(str(row["case_id"]), []).append(row)
    output: List[Dict[str, Any]] = []
    for case_id in sorted(by_case):
        rows = sorted(by_case[case_id], key=lambda row: int(row["compression_rank_q"]))
        if len(rows) != 2:
            raise Gate12C1SummaryContractError(f"run {case_id} missing q endpoints")
        output.append(
            {
                "case_id": case_id,
                "case_order": int(rows[0]["case_order"]),
                "model": str(rows[0]["model"]),
                "family": str(rows[0]["family"]),
                "run_support": bool(rows[0].get("run_support")),
                "q_discordant_run": rows[0].get("q_discordant_run"),
            }
        )
    return sorted(output, key=lambda row: int(row["case_order"]))


def breadth_constraints_pass(run_rows: Sequence[Mapping[str, Any]]) -> bool:
    supporting = [row for row in run_rows if row.get("run_support") is True]
    family_counts: Dict[str, int] = {}
    model_counts: Dict[str, int] = {}
    for row in supporting:
        family_counts[str(row["family"])] = family_counts.get(str(row["family"]), 0) + 1
        model_counts[str(row["model"])] = model_counts.get(str(row["model"]), 0) + 1
    families = sorted({case.family for case in CANONICAL_CASES})
    models = sorted({case.model for case in CANONICAL_CASES})
    return bool(
        all(family_counts.get(family, 0) >= 3 for family in families)
        and all(model_counts.get(model, 0) >= 2 for model in models)
    )


def summarize_outcome(
    *,
    run_q_tests: Sequence[Mapping[str, Any]],
    case_inputs: Sequence[CaseInput],
    grid_outcome: str,
) -> Dict[str, Any]:
    run_rows = run_level_rows(run_q_tests)
    return {
        "execution_status": "complete",
        "grid_outcome": grid_outcome,
        "supporting_run_count": int(sum(1 for row in run_rows if row["run_support"] is True)),
        "q_discordant_run_count": int(
            sum(1 for row in run_rows if row["q_discordant_run"] is True)
        ),
        "run_support": run_rows,
        "coverage_limited_endpoint_count": int(
            sum(
                1
                for row in run_q_tests
                if row.get("cycle_coverage_pass") is not True
                or row.get("block_coverage_pass") is not True
                or row.get("test_status") == "non_informative"
            )
        ),
        "canonical_case_count": int(len(case_inputs)),
    }


def average_ranks(values: Sequence[float]) -> List[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for ordered_index in range(index, end):
            original_index = ordered[ordered_index][0]
            ranks[original_index] = float(average_rank)
        index = end
    return ranks


def spearman_rho(x_values: Sequence[float], y_values: Sequence[float]) -> Tuple[str, float | None, int]:
    pairs = [
        (float(x), float(y))
        for x, y in zip(x_values, y_values)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(pairs) < 2:
        return "insufficient_or_constant", None, len(pairs)
    x = [pair[0] for pair in pairs]
    y = [pair[1] for pair in pairs]
    if len(set(x)) < 2 or len(set(y)) < 2:
        return "insufficient_or_constant", None, len(pairs)
    rank_x = np.asarray(average_ranks(x), dtype=np.float64)
    rank_y = np.asarray(average_ranks(y), dtype=np.float64)
    centered_x = rank_x - float(np.mean(rank_x))
    centered_y = rank_y - float(np.mean(rank_y))
    denominator = float(np.linalg.norm(centered_x) * np.linalg.norm(centered_y))
    if denominator == 0.0:
        return "insufficient_or_constant", None, len(pairs)
    return "defined", float(np.dot(centered_x, centered_y) / denominator), len(pairs)


def build_secondary_telemetry(
    *,
    case_inputs: Sequence[CaseInput],
    expected_cycles_by_case: Mapping[str, List[ExpectedCycle]],
    rows_by_case: Mapping[str, List[Mapping[str, Any]]],
    cycle_q_scores: Sequence[Mapping[str, Any]],
    block_q_scores: Sequence[Mapping[str, Any]],
    run_q_tests: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    cycle_by_case_q: Dict[Tuple[str, int], List[Mapping[str, Any]]] = {}
    for row in cycle_q_scores:
        if row.get("cycle_q_primary_valid") is True:
            cycle_by_case_q.setdefault(
                (str(row["case_id"]), int(row["compression_rank_q"])),
                [],
            ).append(row)

    telemetry_rows: List[Dict[str, Any]] = []
    for case in case_inputs:
        raw_rows = rows_by_case.get(case.spec.case_id, [])
        for q in (1, 2):
            registry_rows = [
                row for row in raw_rows if int(row["compression_rank_q"]) == q
            ]
            registry_metrics = [primary_row_metrics(row) for row in registry_rows]
            cycle_rows = cycle_by_case_q.get((case.spec.case_id, q), [])
            robust_by_block: Dict[str, List[float]] = {}
            for row in cycle_rows:
                if is_finite_number(row.get("cycle_q_robust_z_median")):
                    robust_by_block.setdefault(
                        str(row["source_sample_block_id"]),
                        [],
                    ).append(float(row["cycle_q_robust_z_median"]))
            robust_block_medians = [
                float(statistics.median(values))
                for values in robust_by_block.values()
                if values
            ]
            rel_values = [
                float(row["cycle_q_associator_rel_median"])
                for row in cycle_rows
                if is_finite_number(row.get("cycle_q_associator_rel_median"))
            ]
            root_spread_values = [
                float(row["cycle_q_root_spread"])
                for row in cycle_rows
                if is_finite_number(row.get("cycle_q_root_spread"))
            ]
            p_values = [
                float(metric["empirical_p_upper"])
                for metric in registry_metrics
                if metric["empirical_p_upper"] is not None
            ]
            scale_count = sum(1 for metric in registry_metrics if metric["scale_degenerate"])
            incomplete_count = sum(1 for metric in registry_metrics if metric["incomplete_null"])
            telemetry_rows.append(
                {
                    "case_id": case.spec.case_id,
                    "case_order": case.spec.case_order,
                    "model": case.spec.model,
                    "family": case.spec.family,
                    "compression_rank_q": q,
                    "hierarchical_block_median_robust_z": median_or_none(
                        robust_block_medians
                    ),
                    "empirical_p_upper_quantiles": quantiles_or_none(p_values),
                    "compressed_associator_rel_median": median_or_none(rel_values),
                    "cycle_q_root_spread_median": median_or_none(root_spread_values),
                    "scale_degenerate_row_count": int(scale_count),
                    "scale_degenerate_row_rate": float(scale_count / len(registry_rows))
                    if registry_rows
                    else None,
                    "incomplete_null_row_count": int(incomplete_count),
                    "incomplete_null_row_rate": float(incomplete_count / len(registry_rows))
                    if registry_rows
                    else None,
                }
            )

    low_holonomy_rows = build_low_holonomy_surface(
        case_inputs=case_inputs,
        expected_cycles_by_case=expected_cycles_by_case,
        cycle_q_scores=cycle_q_scores,
    )
    spearman_rows = build_spearman_rows(case_inputs=case_inputs, cycle_q_scores=cycle_q_scores)
    q_differences = build_q_difference_rows(run_q_tests=run_q_tests)
    return {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "secondary_checks_are_descriptive": True,
        "multiple_testing_boundary": {
            "row_level_discovery_claims_authorized": False,
            "holm_corrected_endpoint_count": HOLM_TEST_COUNT,
            "secondary_checks_holm_corrected": False,
        },
        "run_q_secondary_telemetry": telemetry_rows,
        "q1_vs_q2_difference": q_differences,
        "low_holonomy_secondary_surface": low_holonomy_rows,
        "spearman_correlations": spearman_rows,
    }


def build_low_holonomy_surface(
    *,
    case_inputs: Sequence[CaseInput],
    expected_cycles_by_case: Mapping[str, List[ExpectedCycle]],
    cycle_q_scores: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    valid_by_case_cycle_q: Dict[Tuple[str, str, int], Mapping[str, Any]] = {}
    for row in cycle_q_scores:
        if row.get("cycle_q_primary_valid") is True:
            valid_by_case_cycle_q[
                (str(row["case_id"]), str(row["cycle_id"]), int(row["compression_rank_q"]))
            ] = row
    output: List[Dict[str, Any]] = []
    for case in case_inputs:
        expected = sorted(
            expected_cycles_by_case[case.spec.case_id],
            key=lambda cycle: (cycle.gate12a_holonomy_residual_fro, cycle.cycle_id),
        )
        selected = expected[: len(expected) // 4]
        selected_ids = {cycle.cycle_id for cycle in selected}
        selected_expected_blocks = {
            cycle.source_sample_block_id
            for cycle in selected
            if cycle.source_block_status == "single_sample"
        }
        selected_mixed_count = sum(
            1 for cycle in selected if cycle.source_block_status != "single_sample"
        )
        for q in (1, 2):
            rows = [
                valid_by_case_cycle_q[(case.spec.case_id, cycle_id, q)]
                for cycle_id in sorted(selected_ids)
                if (case.spec.case_id, cycle_id, q) in valid_by_case_cycle_q
            ]
            block_values: Dict[str, List[float]] = {}
            for row in rows:
                block_values.setdefault(str(row["source_sample_block_id"]), []).append(
                    float(row["cycle_q_log_null_ratio"])
                )
            block_scores = [
                float(statistics.median(values)) for values in block_values.values() if values
            ]
            selected_cycle_coverage_ratio = (
                float(len(rows) / len(selected)) if selected else 0.0
            )
            selected_block_coverage_ratio = (
                float(len(block_values) / len(selected_expected_blocks))
                if selected_expected_blocks
                else 0.0
            )
            output.append(
                {
                    "case_id": case.spec.case_id,
                    "case_order": case.spec.case_order,
                    "compression_rank_q": q,
                    "selected_expected_cycle_count": int(len(selected)),
                    "selected_valid_cycle_count": int(len(rows)),
                    "selected_cycle_coverage_ratio": selected_cycle_coverage_ratio,
                    "selected_expected_block_count": int(len(selected_expected_blocks)),
                    "selected_represented_block_count": int(len(block_values)),
                    "selected_block_coverage_ratio": selected_block_coverage_ratio,
                    "selected_mixed_or_undefined_count": int(selected_mixed_count),
                    "low_holonomy_run_q_median": median_or_none(block_scores),
                }
            )
    return output


def build_spearman_rows(
    *,
    case_inputs: Sequence[CaseInput],
    cycle_q_scores: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    by_case_q: Dict[Tuple[str, int], List[Mapping[str, Any]]] = {}
    for row in cycle_q_scores:
        if row.get("cycle_q_primary_valid") is True:
            by_case_q.setdefault((str(row["case_id"]), int(row["compression_rank_q"])), []).append(
                row
            )
    output: List[Dict[str, Any]] = []
    for case in case_inputs:
        for q in (1, 2):
            rows = by_case_q.get((case.spec.case_id, q), [])
            effects = [
                float(row["cycle_q_log_null_ratio"])
                for row in rows
                if is_finite_number(row.get("cycle_q_log_null_ratio"))
            ]
            holonomy = [
                float(row["gate12a_holonomy_residual_fro"])
                for row in rows
                if is_finite_number(row.get("cycle_q_log_null_ratio"))
            ]
            edge_gap = [
                float(row["edge_compatibility_gap_max"])
                for row in rows
                if is_finite_number(row.get("cycle_q_log_null_ratio"))
            ]
            for predictor, y_values in (
                ("gate12a_holonomy_residual_fro", holonomy),
                ("edge_compatibility_gap_max", edge_gap),
            ):
                status, rho, n = spearman_rho(effects, y_values)
                output.append(
                    {
                        "case_id": case.spec.case_id,
                        "case_order": case.spec.case_order,
                        "compression_rank_q": q,
                        "predictor": predictor,
                        "spearman_status": status,
                        "spearman_rho": rho,
                        "cycle_q_count": int(n),
                    }
                )
    return output


def build_q_difference_rows(run_q_tests: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_case: Dict[str, Dict[int, Mapping[str, Any]]] = {}
    for row in run_q_tests:
        by_case.setdefault(str(row["case_id"]), {})[int(row["compression_rank_q"])] = row
    output: List[Dict[str, Any]] = []
    for case_id in sorted(by_case):
        q1 = by_case[case_id].get(1)
        q2 = by_case[case_id].get(2)
        diff = None
        if q1 and q2 and is_finite_number(q1.get("run_q_median")) and is_finite_number(
            q2.get("run_q_median")
        ):
            diff = float(q2["run_q_median"]) - float(q1["run_q_median"])
        source = q1 or q2 or {}
        output.append(
            {
                "case_id": case_id,
                "case_order": int(source.get("case_order", 0)),
                "run_q2_minus_q1_median": diff,
            }
        )
    return sorted(output, key=lambda row: int(row["case_order"]))


def build_case_inventory_rows(
    *,
    case_inputs: Sequence[CaseInput],
    expected_cycles_by_case: Mapping[str, List[ExpectedCycle]],
    runner_manifests: Mapping[str, Mapping[str, Any]],
    integrity_status_by_case: Mapping[str, Mapping[str, str]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for case in case_inputs:
        expected_cycles = expected_cycles_by_case[case.spec.case_id]
        integrity = integrity_status_by_case[case.spec.case_id]
        expected_blocks = {
            cycle.source_sample_block_id
            for cycle in expected_cycles
            if cycle.source_block_status == "single_sample"
        }
        mixed_count = sum(
            1 for cycle in expected_cycles if cycle.source_block_status != "single_sample"
        )
        rows.append(
            {
                "case_id": case.spec.case_id,
                "case_order": case.spec.case_order,
                "model": case.spec.model,
                "family": case.spec.family,
                "source_gate12a_run_id": case.spec.source_gate12a_run_id,
                "gate12c1_run_id": str(runner_manifests[case.spec.case_id].get("run_id") or ""),
                "preflight_eligible_cycle_count": case.spec.preflight_eligible_cycle_count,
                "derived_eligible_cycle_count": len(expected_cycles),
                "expected_single_sample_block_count": len(expected_blocks),
                "mixed_or_undefined_expected_cycle_count": mixed_count,
                "source_gate12a_checksum_status": integrity["source_gate12a_checksum_status"],
                "gate12c1_output_checksum_status": integrity["gate12c1_output_checksum_status"],
                "source_gate12a_immutability_status": integrity[
                    "source_gate12a_immutability_status"
                ],
                "gate12c1_output_immutability_status": integrity[
                    "gate12c1_output_immutability_status"
                ],
            }
        )
    return rows


def build_summary_manifest(
    *,
    case_manifest_path: Path,
    case_manifest_sha256: str,
    case_inputs: Sequence[CaseInput],
    runner_manifests: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "plan_id": PLAN_ID,
        "builder_git_commit": current_git_commit(),
        "builder_script_sha256": sha256_file(SCRIPT_PATH),
        "case_manifest_path": repo_relative_or_posix(case_manifest_path),
        "case_manifest_sha256": case_manifest_sha256,
        "primary_zero_tolerance": PRIMARY_ZERO_TOLERANCE,
        "primary_epsilon": PRIMARY_EPSILON,
        "coverage_threshold": COVERAGE_THRESHOLD,
        "holm_alpha": HOLM_ALPHA,
        "holm_endpoint_count": HOLM_TEST_COUNT,
        "runner_expected_provenance": {
            "post_merge_main_commit": EXPECTED_RUNNER_COMMIT,
            "runner_script_sha256": EXPECTED_RUNNER_SCRIPT_SHA256,
            "schema_version": EXPECTED_RUNNER_SCHEMA,
            "method_id": EXPECTED_RUNNER_METHOD,
            "run_mode": EXPECTED_RUN_MODE,
            "orientation_null_seed": EXPECTED_ORIENTATION_NULL_SEED,
            "orientation_null_requested_draw_count": EXPECTED_ORIENTATION_NULL_REQUESTED_DRAW_COUNT,
            "orientation_null_max_attempt_count": EXPECTED_ORIENTATION_NULL_MAX_ATTEMPT_COUNT,
            "tolerances": EXPECTED_TOLERANCES,
        },
        "case_run_ids": [
            {
                "case_id": case.spec.case_id,
                "source_gate12a_run_id": case.spec.source_gate12a_run_id,
                "gate12c1_run_id": str(runner_manifests[case.spec.case_id].get("run_id") or ""),
                "source_gate12a_dir": repo_relative_or_posix(case.source_gate12a_dir),
                "gate12c1_run_dir": repo_relative_or_posix(case.gate12c1_run_dir),
            }
            for case in case_inputs
        ],
        "claim_boundary": {
            "gate12b_overlay_used": False,
            "type_iii_claim_authorized": False,
            "row_level_discovery_claims_authorized": False,
            "scientific_null_excess_threshold_defined_by_runner": False,
            "rectangular_rank_mismatch_supported": False,
            "gate12b_overlay_forbidden_until_result_memo": True,
        },
    }


def build_grid_summary_payload(
    *,
    outcome: Mapping[str, Any],
    run_q_tests: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "plan_id": PLAN_ID,
        "execution_status": outcome["execution_status"],
        "grid_outcome": outcome["grid_outcome"],
        "primary_zero_tolerance": PRIMARY_ZERO_TOLERANCE,
        "primary_epsilon": PRIMARY_EPSILON,
        "coverage_threshold": COVERAGE_THRESHOLD,
        "holm_alpha": HOLM_ALPHA,
        "holm_endpoint_count": HOLM_TEST_COUNT,
        "supporting_run_count": outcome["supporting_run_count"],
        "q_discordant_run_count": outcome["q_discordant_run_count"],
        "coverage_limited_endpoint_count": outcome.get("coverage_limited_endpoint_count"),
        "run_support": outcome.get("run_support", []),
        "run_q_tests": list(run_q_tests),
        "multiple_testing_boundary": {
            "row_level_discovery_claims_authorized": False,
            "holm_corrected_endpoint_count": HOLM_TEST_COUNT,
            "secondary_checks_holm_corrected": False,
        },
    }


def build_readme(*, outcome: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Gate12C-1 First Empirical Grid Summary",
            "",
            "This summary applies the frozen synthetic-validated Gate12C-1 first empirical grid plan.",
            "",
            "It does not run Gate12C-1, inspect Gate12B overlays, authorize row-level discovery claims, or alter runner settings.",
            "",
            "## Status",
            "",
            f"- execution_status: `{outcome['execution_status']}`",
            f"- grid_outcome: `{outcome['grid_outcome']}`",
            f"- supporting_run_count: `{outcome['supporting_run_count']}`",
            f"- q_discordant_run_count: `{outcome['q_discordant_run_count']}`",
            "",
            "Only the 24 predeclared run/q sign tests receive Holm correction.",
        ]
    ) + "\n"


def write_checksums(out_dir: Path) -> None:
    checksums = {name: sha256_file(out_dir / name) for name in OUTPUT_FILES}
    write_json(out_dir / OUTPUT_CHECKSUMS, checksums)


def write_failure_diagnostic(out_dir: Path, exc: BaseException) -> None:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            out_dir / OUTPUT_GRID_SUMMARY,
            {
                "schema_version": SCHEMA_VERSION,
                "method_id": METHOD_ID,
                "plan_id": PLAN_ID,
                "execution_status": "contract_failure",
                "grid_outcome": "not_classified",
                "failure_type": exc.__class__.__name__,
                "failure_message": str(exc),
            },
        )
    except Exception:
        return


def summarize_gate12c1_first_empirical_grid(
    *,
    case_manifest_path: Path,
    out_dir: Path,
) -> Dict[str, Any]:
    case_manifest_path = Path(case_manifest_path)
    out_dir = Path(out_dir)
    case_manifest, case_inputs = load_case_manifest_and_validate_paths(
        case_manifest_path,
        out_dir,
    )
    case_manifest_sha = sha256_file(case_manifest_path)

    runner_manifests: Dict[str, Dict[str, Any]] = {}
    expected_cycles_by_case: Dict[str, List[ExpectedCycle]] = {}
    expected_by_cycle_by_case: Dict[str, Dict[str, ExpectedCycle]] = {}
    rows_by_case: Dict[str, List[Mapping[str, Any]]] = {}
    integrity_status_by_case: Dict[str, Dict[str, str]] = {}

    for case in case_inputs:
        verify_source_gate12a_checksums(case.source_gate12a_dir)
        verify_run_checksums(case.gate12c1_run_dir)
        integrity_status_by_case[case.spec.case_id] = {
            "source_gate12a_checksum_status": "pass",
            "gate12c1_output_checksum_status": "pass",
            "source_gate12a_immutability_status": "pass",
            "gate12c1_output_immutability_status": "pass",
        }
    input_hashes_before = snapshot_input_hashes(case_inputs)

    for case in case_inputs:
        runner_manifest = read_json(case.gate12c1_run_dir / "manifest.json")
        run_status = read_json(case.gate12c1_run_dir / "gate12c_status.json")
        source_manifest = read_json(case.source_gate12a_dir / gate12c0.DEFAULT_MANIFEST)
        validate_runner_manifest(
            case=case,
            runner_manifest=runner_manifest,
            source_manifest=source_manifest,
        )
        validate_run_status(case=case, status=run_status)
        expected_cycles, expected_by_cycle = derive_expected_cycles(case)
        runner_rows = validate_and_load_registry(
            case=case,
            expected_by_cycle=expected_by_cycle,
        )
        runner_manifests[case.spec.case_id] = runner_manifest
        expected_cycles_by_case[case.spec.case_id] = expected_cycles
        expected_by_cycle_by_case[case.spec.case_id] = expected_by_cycle
        rows_by_case[case.spec.case_id] = runner_rows

    cycle_q_scores = build_cycle_q_scores(
        case_inputs=case_inputs,
        expected_cycles_by_case=expected_cycles_by_case,
        rows_by_case=rows_by_case,
    )
    block_q_scores = build_block_q_scores(
        case_inputs=case_inputs,
        expected_cycles_by_case=expected_cycles_by_case,
        cycle_q_scores=cycle_q_scores,
    )
    run_q_tests = build_run_q_tests(
        case_inputs=case_inputs,
        expected_cycles_by_case=expected_cycles_by_case,
        cycle_q_scores=cycle_q_scores,
        block_q_scores=block_q_scores,
    )
    outcome = classify_grid_outcome(
        execution_status="complete",
        run_q_tests=run_q_tests,
        case_inputs=case_inputs,
    )
    secondary = build_secondary_telemetry(
        case_inputs=case_inputs,
        expected_cycles_by_case=expected_cycles_by_case,
        rows_by_case=rows_by_case,
        cycle_q_scores=cycle_q_scores,
        block_q_scores=block_q_scores,
        run_q_tests=run_q_tests,
    )
    case_inventory = build_case_inventory_rows(
        case_inputs=case_inputs,
        expected_cycles_by_case=expected_cycles_by_case,
        runner_manifests=runner_manifests,
        integrity_status_by_case=integrity_status_by_case,
    )
    summary_manifest = build_summary_manifest(
        case_manifest_path=case_manifest_path,
        case_manifest_sha256=case_manifest_sha,
        case_inputs=case_inputs,
        runner_manifests=runner_manifests,
    )
    grid_summary = build_grid_summary_payload(outcome=outcome, run_q_tests=run_q_tests)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / OUTPUT_MANIFEST, summary_manifest)
    write_csv(out_dir / OUTPUT_CASE_INVENTORY, CASE_INVENTORY_FIELDNAMES, case_inventory)
    write_jsonl(out_dir / OUTPUT_CYCLE_Q, cycle_q_scores)
    write_jsonl(out_dir / OUTPUT_BLOCK_Q, block_q_scores)
    write_csv(out_dir / OUTPUT_RUN_Q, RUN_Q_FIELDNAMES, run_q_tests)
    write_json(out_dir / OUTPUT_GRID_SUMMARY, grid_summary)
    write_json(out_dir / OUTPUT_SECONDARY, secondary)
    write_text(out_dir / OUTPUT_READ, build_readme(outcome=outcome))
    write_checksums(out_dir)
    verify_input_immutability(case_inputs=case_inputs, before=input_hashes_before)

    return {
        "manifest": summary_manifest,
        "case_manifest": case_manifest,
        "case_inventory": case_inventory,
        "cycle_q_scores": cycle_q_scores,
        "block_q_scores": block_q_scores,
        "run_q_tests": run_q_tests,
        "grid_summary": grid_summary,
        "secondary_telemetry": secondary,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    case_manifest_path = Path(args.case_manifest)
    out_dir = Path(args.out_dir)
    try:
        load_case_manifest_and_validate_paths(case_manifest_path, out_dir)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    try:
        summarize_gate12c1_first_empirical_grid(
            case_manifest_path=case_manifest_path,
            out_dir=out_dir,
        )
    except Exception as exc:
        write_failure_diagnostic(out_dir, exc)
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
