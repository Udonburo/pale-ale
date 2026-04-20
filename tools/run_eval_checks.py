#!/usr/bin/env python3
"""Plan or run read-only evaluation-factory checks by standing tier.

This entrypoint does not introduce analytical methods or change Gate12A
doctrine. Model/GPU execution is only allowed for the explicit
``--tier l4-smoke --execute`` lane.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]


class Tier(str, Enum):
    CPU_NIGHTLY = "cpu-nightly"
    L4_SMOKE = "l4-smoke"
    L4_WEEKLY = "l4-weekly"
    SUMMARIZE_EXISTING = "summarize-existing"


@dataclass(frozen=True)
class TierPlan:
    tier: Tier
    intent: str
    resource_posture: str
    planned_actions: tuple[str, ...]
    out_of_scope: tuple[str, ...]
    not_implemented_yet: tuple[str, ...]


@dataclass(frozen=True)
class CheckResult:
    level: str
    label: str
    detail: str = ""


@dataclass(frozen=True)
class CrossModelSummary:
    run_id: str
    status: str
    path: str
    model_label: str
    model_id: str
    families: tuple[str, ...]
    row_count: int
    structural_flags_all_true: str
    first_pass_statuses: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class L4SmokeConfig:
    model_id: str
    model_label: str
    families: tuple[str, ...]
    device: str
    topk: int
    seed: int
    gate12a_top_k: int
    balanced_per_band: int
    reading_limit: int
    summary_run_id: str


@dataclass(frozen=True)
class L4WeeklyTarget:
    model_id: str
    model_label: str
    families: tuple[str, ...]


@dataclass(frozen=True)
class L4SmokePreflight:
    sys_executable: str
    python_version: str
    cwd: str
    platform: str
    os_name: str
    torch_importable: bool
    torch_version: str
    torch_cuda_available: bool | None
    torch_cuda_version: str
    gpu_count: int | None
    gpu_names: tuple[str, ...]
    nvidia_smi_available: bool
    nvidia_smi_path: str
    nvidia_smi_summary: tuple[str, ...]
    nvidia_smi_error: str
    posture_classification: str
    preflight_ok: bool
    remediation_hints: tuple[str, ...]
    errors: tuple[str, ...]


@dataclass(frozen=True)
class EvalFactoryArtifactValidation:
    source_class: str
    artifact_kind: str
    path: str
    status: str
    schema_id: str
    mode: str
    result: str
    posture_classification: str
    downstream_result: str
    errors: tuple[str, ...]


@dataclass(frozen=True)
class TrackedMemoSurface:
    model_label: str
    model_id: str
    memo_id: str
    memo_file: str
    tracked_scope: str


TIER_VALUES = tuple(tier.value for tier in Tier)
LEVEL_PASS = "PASS"
LEVEL_WARN = "WARN"
LEVEL_FAIL = "FAIL"

FAMILY_SET = ("transcript_v1", "briefing_v1", "archive_v1")

L4_SMOKE_BOUNDARY = "0.5B fixed family boundary set: transcript_v1, briefing_v1, archive_v1"
L4_SMOKE_CONFIG = L4SmokeConfig(
    model_id="Qwen/Qwen2.5-0.5B",
    model_label="qwen_qwen2_5_0_5b",
    families=FAMILY_SET,
    device="cuda",
    topk=128,
    seed=7,
    gate12a_top_k=3,
    balanced_per_band=6,
    reading_limit=0,
    summary_run_id="gate12a_cross_model_replay_qwen_qwen2_5_0_5b",
)
L4_SMOKE_STATUS_FILENAME = "eval_factory_l4_smoke_status.json"
L4_SMOKE_PREFLIGHT_FILENAME = "eval_factory_l4_smoke_preflight.json"
L4_WEEKLY_PLAN_FILENAME = "eval_factory_l4_weekly_plan.json"
ARTIFACT_CONTRACT_VERSION = 1
L4_SMOKE_PREFLIGHT_SCHEMA_ID = "pale-ale.eval_factory.l4_smoke.preflight.v1"
L4_SMOKE_STATUS_SCHEMA_ID = "pale-ale.eval_factory.l4_smoke.status.v1"
L4_WEEKLY_PLAN_SCHEMA_ID = "pale-ale.eval_factory.l4_weekly.plan.v1"
SOURCE_EVAL_FACTORY_PREFLIGHT = "eval-factory preflight artifact"
SOURCE_EVAL_FACTORY_STATUS = "eval-factory execute/status artifact"
ARTIFACT_STATUS_VALID = "valid"
ARTIFACT_STATUS_MALFORMED = "malformed"
ARTIFACT_STATUS_MISSING = "missing"
ARTIFACT_DISCOVERY_EXCLUDED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "node_modules",
    "target",
    "venv",
}

POSTURE_REMOTE_CUDA_READY = "remote_cuda_ready"
POSTURE_LOCAL_WINDOWS_NO_CUDA = "local_windows_no_cuda"
POSTURE_PYTHON_MISSING_TORCH = "python_missing_torch"
POSTURE_CUDA_UNAVAILABLE = "cuda_unavailable"
POSTURE_UNKNOWN = "unknown_posture"
POSTURE_CLASSIFICATIONS = (
    POSTURE_REMOTE_CUDA_READY,
    POSTURE_LOCAL_WINDOWS_NO_CUDA,
    POSTURE_PYTHON_MISSING_TORCH,
    POSTURE_CUDA_UNAVAILABLE,
    POSTURE_UNKNOWN,
)
RESULT_VALUES = ("pass", "fail")
PREFLIGHT_ARTIFACT_MODES = ("preflight-only", "execute")

L4_WEEKLY_SURFACES = (
    "current 3B/4B dense-transformer family-set surfaces under the frozen Gate12A observable contract"
)

# 7B FP32 is not in the L4 mainline. Keep it out of l4-weekly until there is a
# separate resource posture and claim surface for it.
L4_WEEKLY_EXCLUDES_7B_FP32 = "7B FP32"

# Protocol-expanding, quantized, and sidecar candidates are not l4-weekly work.
L4_WEEKLY_EXCLUDED_CANDIDATES = (
    "protocol-expanding candidates",
    "quantized candidates",
    "sidecar candidates",
)
L4_WEEKLY_EXCLUSIONS = (
    L4_WEEKLY_EXCLUDES_7B_FP32,
    *L4_WEEKLY_EXCLUDED_CANDIDATES,
    "Gate12B promotion",
)
L4_WEEKLY_TARGETS = (
    L4WeeklyTarget(
        model_id="Qwen/Qwen2.5-3B-Instruct",
        model_label="qwen_qwen2_5_3b_instruct",
        families=FAMILY_SET,
    ),
    L4WeeklyTarget(
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        model_label="meta_llama_llama_3_2_3b_instruct",
        families=FAMILY_SET,
    ),
    L4WeeklyTarget(
        model_id="Qwen/Qwen3-4B",
        model_label="qwen_qwen3_4b",
        families=FAMILY_SET,
    ),
)

EXPECTED_ATLAS_MEMOS = (
    "200_GATE12A_TRANSCRIPT_V1_ANCHOR_RICH_CLOSURE_TENSION_REPLICATION_MEMO.md",
    "201_GATE12A_BRIEFING_V1_ANCHOR_RICH_CLOSURE_TENSION_EMPIRICAL_MEMO.md",
    "202_GATE12A_TRANSCRIPT_V1_VS_BRIEFING_V1_ANCHOR_RICH_CLOSURE_TENSION_COMPARISON_MEMO.md",
    "206_GATE12A_LLAMA_3_2_1B_INSTRUCT_FIXED_FAMILY_SET_REPLICATION_MEMO.md",
    "207_GATE12A_QWEN_2_5_1_5B_INSTRUCT_TRANSCRIPT_V1_GPU_IMPORT_REPLICATION_MEMO.md",
    "210_GATE12A_QWEN_2_5_3B_INSTRUCT_FIXED_FAMILY_SET_REPLICATION_MEMO.md",
    "211_GATE12A_LLAMA_3_2_3B_INSTRUCT_FIXED_FAMILY_SET_REPLICATION_MEMO.md",
    "212_GATE12A_QWEN3_4B_FIXED_FAMILY_SET_REPLICATION_MEMO.md",
    "214_GATE12A_FROZEN_PROTOCOL_EXCLUSION_AND_NON_TRANSFORMER_SIDECAR_MEMO.md",
    "215_GATE12A_QWEN_2_5_0_5B_FIXED_FAMILY_SET_REPLICATION_MEMO.md",
)

EXPECTED_SUMMARY_RUNS = (
    "gate12a_cross_model_replay_qwen_qwen2_5_0_5b",
    "gate12a_cross_model_replay_qwen_qwen2_5_3b_instruct",
    "gate12a_cross_model_replay_meta_llama_llama_3_2_3b_instruct",
    "gate12a_cross_model_replay_qwen_qwen3_4b",
)

TRACKED_MEMO_SURFACES = (
    TrackedMemoSurface(
        model_label="qwen_qwen2_5_3b_instruct",
        model_id="Qwen/Qwen2.5-3B-Instruct",
        memo_id="210",
        memo_file="210_GATE12A_QWEN_2_5_3B_INSTRUCT_FIXED_FAMILY_SET_REPLICATION_MEMO.md",
        tracked_scope="mainline dense-transformer fixed-family-set memo",
    ),
    TrackedMemoSurface(
        model_label="meta_llama_llama_3_2_3b_instruct",
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        memo_id="211",
        memo_file="211_GATE12A_LLAMA_3_2_3B_INSTRUCT_FIXED_FAMILY_SET_REPLICATION_MEMO.md",
        tracked_scope="mainline dense-transformer fixed-family-set memo",
    ),
    TrackedMemoSurface(
        model_label="qwen_qwen3_4b",
        model_id="Qwen/Qwen3-4B",
        memo_id="212",
        memo_file="212_GATE12A_QWEN3_4B_FIXED_FAMILY_SET_REPLICATION_MEMO.md",
        tracked_scope="mainline dense-transformer fixed-family-set memo",
    ),
    TrackedMemoSurface(
        model_label="qwen_qwen2_5_0_5b",
        model_id="Qwen/Qwen2.5-0.5B",
        memo_id="215",
        memo_file="215_GATE12A_QWEN_2_5_0_5B_FIXED_FAMILY_SET_REPLICATION_MEMO.md",
        tracked_scope="post-checkpoint lower-bound family-set memo",
    ),
    TrackedMemoSurface(
        model_label="meta_llama_llama_3_2_1b_instruct",
        model_id="meta-llama/Llama-3.2-1B-Instruct",
        memo_id="206",
        memo_file="206_GATE12A_LLAMA_3_2_1B_INSTRUCT_FIXED_FAMILY_SET_REPLICATION_MEMO.md",
        tracked_scope="post-checkpoint fixed-family-set memo",
    ),
    TrackedMemoSurface(
        model_label="qwen_qwen2_5_1_5b_instruct",
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        memo_id="207",
        memo_file="207_GATE12A_QWEN_2_5_1_5B_INSTRUCT_TRANSCRIPT_V1_GPU_IMPORT_REPLICATION_MEMO.md",
        tracked_scope="post-checkpoint transcript-only follow-up memo",
    ),
)

REQUIRED_CPU_FILES = (
    "README.md",
    "pyproject.toml",
    "CITATION.cff",
    "docs/gate12a_evidence_atlas.md",
    "docs/reproduce_gate12a.md",
    "workstream/README.md",
    "zenodo-release/CHECKSUMS-SHA256.txt",
    "tools/run_eval_checks.py",
    "tools/test_run_eval_checks.py",
    "tools/run_gate12a_cross_model_replay.py",
    "tools/run_gate12a_family_replay.py",
)

CROSS_MODEL_SUMMARY_FILENAME = "cross_model_family_summary.csv"
MANIFEST_FILENAME = "manifest.json"
STRUCTURAL_FLAG_COLUMNS = (
    "zero_overlap_clear",
    "all_defined_triangles_anchor_rich",
    "trusted_tree_gt_residual_chord",
    "plain_gt_anchor_qualified",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run read-only evaluation-factory checks or print a plan for one standing tier. "
            "GPU/model execution is only available for --tier l4-smoke --execute."
        )
    )
    parser.add_argument(
        "--tier",
        required=True,
        choices=TIER_VALUES,
        help="Standing evaluation tier to plan or run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help=(
            "Compatibility flag. Default behavior is non-executing; read-only tiers inspect existing "
            "local files without running jobs."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the l4-smoke lane. Only supported with --tier l4-smoke and an explicit --out-dir.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run only the l4-smoke environment/GPU posture preflight. Does not invoke model execution.",
    )
    parser.add_argument(
        "--out-dir",
        help=(
            "Required output root for --tier l4-smoke --execute; optional artifact root for "
            "--tier l4-smoke --preflight-only and --tier l4-weekly. Generated files are not committed by this tool."
        ),
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def plan_cpu_nightly() -> TierPlan:
    return TierPlan(
        tier=Tier.CPU_NIGHTLY,
        intent="fast local checks; manifest / file / structure validation; no GPU expectation",
        resource_posture="CPU-only; cheap; suitable for local or nightly validation",
        planned_actions=(
            "verify expected repository file surfaces and key docs/tools",
            "verify the eval-factory script surface is intact",
            "perform shallow manifest/path sanity checks for already-materialized summaries",
        ),
        out_of_scope=(
            "GPU invocation",
            "model execution",
            "Gate12A math or doctrine changes",
        ),
        not_implemented_yet=(
            "CI scheduling integration",
            "artifact publication",
            "model dispatch",
        ),
    )


def plan_l4_smoke() -> TierPlan:
    return TierPlan(
        tier=Tier.L4_SMOKE,
        intent="small standing smoke lane aligned with the 0.5B boundary set",
        resource_posture="single L4 posture; explicitly cheap and repeatable",
        planned_actions=(
            f"dry-run or execute checks for {L4_SMOKE_BOUNDARY}",
            "use the committed Gate12A cross-model replay entrypoint for the 0.5B fixed family set",
            "require --execute and an explicit --out-dir before any model/GPU execution",
        ),
        out_of_scope=(
            "full dense-transformer family-set expansion",
            "7B FP32",
            *L4_WEEKLY_EXCLUDED_CANDIDATES,
            "new analytical methods",
        ),
        not_implemented_yet=(
            "automated scheduling",
            "l4-weekly execution",
        ),
    )


def plan_l4_weekly() -> TierPlan:
    return TierPlan(
        tier=Tier.L4_WEEKLY,
        intent="mainline standing lane aligned with the current 3B/4B dense-transformer family-set surfaces",
        resource_posture="single L4 posture for planned weekly work; excludes 7B FP32",
        planned_actions=(
            f"plan checks for {L4_WEEKLY_SURFACES}",
            "confirm the frozen Gate12A observable surface before any future dispatch",
            "compile a structured plan for the current 3B/4B dense-transformer mainline only",
        ),
        out_of_scope=L4_WEEKLY_EXCLUSIONS,
        not_implemented_yet=(
            "3B/4B weekly dispatch",
            "L4 runtime budget enforcement",
            "standing execution summary publication",
        ),
    )


def plan_summarize_existing() -> TierPlan:
    return TierPlan(
        tier=Tier.SUMMARIZE_EXISTING,
        intent="parse existing artifacts / manifests / summaries; no new model execution",
        resource_posture="CPU-only; read-only artifact parsing; no GPU invocation",
        planned_actions=(
            "discover existing memo-facing and materialized summary surfaces",
            "parse already-materialized cross-model summaries and manifests",
            "emit a compact rollup for what currently exists",
        ),
        out_of_scope=(
            "new model execution",
            "GPU invocation",
            "new claim or release surface",
        ),
        not_implemented_yet=(),
    )


def dispatch(tier: Tier) -> TierPlan:
    if tier == Tier.CPU_NIGHTLY:
        return plan_cpu_nightly()
    if tier == Tier.L4_SMOKE:
        return plan_l4_smoke()
    if tier == Tier.L4_WEEKLY:
        return plan_l4_weekly()
    if tier == Tier.SUMMARIZE_EXISTING:
        return plan_summarize_existing()
    raise ValueError(f"unsupported tier: {tier}")


def render_plan(plan: TierPlan) -> str:
    lines = [
        f"tier: {plan.tier.value}",
        f"intent: {plan.intent}",
        f"expected resource posture: {plan.resource_posture}",
        "planned actions:",
    ]
    lines.extend(f"  - {action}" for action in plan.planned_actions)
    lines.append("out of scope:")
    lines.extend(f"  - {item}" for item in plan.out_of_scope)
    lines.append("not implemented yet:")
    if plan.not_implemented_yet:
        lines.extend(f"  - {item}" for item in plan.not_implemented_yet)
    else:
        lines.append("  - none")
    return "\n".join(lines)


def repo_relative(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        return rows, tuple(reader.fieldnames or ())


def bool_cell(value: str | None) -> bool:
    return str(value).strip().lower() == "true"


def ordered_families(families: set[str]) -> tuple[str, ...]:
    known = [family for family in FAMILY_SET if family in families]
    extra = sorted(family for family in families if family not in FAMILY_SET)
    return tuple(known + extra)


def format_counter(counter: Counter[str]) -> str:
    if not counter:
        return "none"
    return ", ".join(f"{key}={counter[key]}" for key in sorted(counter))


def utc_created_at() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def l4_smoke_fixed_target_set() -> dict[str, Any]:
    return {
        "boundary": L4_SMOKE_BOUNDARY,
        "model_id": L4_SMOKE_CONFIG.model_id,
        "model_label": L4_SMOKE_CONFIG.model_label,
        "families": list(L4_SMOKE_CONFIG.families),
        "device": L4_SMOKE_CONFIG.device,
    }


def l4_weekly_target_matrix() -> list[dict[str, Any]]:
    return [
        {
            "model_id": target.model_id,
            "model_label": target.model_label,
            "families": list(target.families),
        }
        for target in L4_WEEKLY_TARGETS
    ]


def result_from_bool(ok: bool) -> str:
    return "pass" if ok else "fail"


def discover_summary_dirs(repo_root: Path) -> tuple[Path, ...]:
    runs_root = repo_root / "runs"
    if not runs_root.exists():
        return ()
    return tuple(
        sorted(
            (
                path
                for path in runs_root.iterdir()
                if path.is_dir() and path.name.startswith("gate12a_cross_model_replay_")
            ),
            key=lambda item: item.name,
        )
    )


def discover_files_by_name(repo_root: Path, filename: str) -> tuple[Path, ...]:
    if not repo_root.exists():
        return ()
    matches: list[Path] = []
    for current_root, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [name for name in dirnames if name not in ARTIFACT_DISCOVERY_EXCLUDED_DIRS]
        if filename in filenames:
            matches.append(Path(current_root) / filename)
    return tuple(sorted(matches, key=lambda path: repo_relative(repo_root, path)))


def discover_eval_factory_preflight_artifacts(repo_root: Path) -> tuple[Path, ...]:
    return discover_files_by_name(repo_root, L4_SMOKE_PREFLIGHT_FILENAME)


def discover_eval_factory_status_artifacts(repo_root: Path) -> tuple[Path, ...]:
    return discover_files_by_name(repo_root, L4_SMOKE_STATUS_FILENAME)


def count_shallow_gate12a_run_dirs(repo_root: Path) -> int:
    runs_root = repo_root / "runs"
    if not runs_root.exists():
        return 0
    return sum(1 for path in runs_root.iterdir() if path.is_dir() and path.name.startswith("gate12a_"))


def parse_cross_model_summary(repo_root: Path, run_dir: Path) -> CrossModelSummary:
    csv_path = run_dir / CROSS_MODEL_SUMMARY_FILENAME
    manifest_path = run_dir / MANIFEST_FILENAME
    notes: list[str] = []
    manifest: Mapping[str, Any] = {}

    if manifest_path.exists():
        try:
            manifest = read_json(manifest_path)
        except (OSError, json.JSONDecodeError) as exc:
            notes.append(f"manifest unreadable: {exc}")
    else:
        notes.append("manifest missing")

    manifest_model_id = str(manifest.get("model_id", "")).strip()
    manifest_model_label = str(manifest.get("model_label", "")).strip()

    if not csv_path.exists():
        return CrossModelSummary(
            run_id=run_dir.name,
            status="missing-summary",
            path=repo_relative(repo_root, run_dir),
            model_label=manifest_model_label or "unknown",
            model_id=manifest_model_id or "unknown",
            families=(),
            row_count=0,
            structural_flags_all_true="n/a",
            first_pass_statuses="none",
            notes=tuple(notes + [f"{CROSS_MODEL_SUMMARY_FILENAME} missing"]),
        )

    try:
        rows, fieldnames = read_csv_rows(csv_path)
    except (OSError, csv.Error, UnicodeDecodeError) as exc:
        return CrossModelSummary(
            run_id=run_dir.name,
            status="malformed-summary",
            path=repo_relative(repo_root, csv_path),
            model_label=manifest_model_label or "unknown",
            model_id=manifest_model_id or "unknown",
            families=(),
            row_count=0,
            structural_flags_all_true="n/a",
            first_pass_statuses="none",
            notes=tuple(notes + [f"summary unreadable: {exc}"]),
        )

    missing_columns = [
        column
        for column in ("model_label", "model_id", "rendering_family", *STRUCTURAL_FLAG_COLUMNS)
        if column not in fieldnames
    ]
    if missing_columns:
        notes.append("missing columns: " + ", ".join(missing_columns))

    families = ordered_families({row.get("rendering_family", "").strip() for row in rows if row.get("rendering_family")})
    missing_families = tuple(family for family in FAMILY_SET if family not in families)
    if missing_families:
        notes.append("missing families: " + ", ".join(missing_families))

    structural_total = len(rows)
    structural_clear = 0
    if all(column in fieldnames for column in STRUCTURAL_FLAG_COLUMNS):
        for row in rows:
            if all(bool_cell(row.get(column)) for column in STRUCTURAL_FLAG_COLUMNS):
                structural_clear += 1
        structural_flags = f"{structural_clear}/{structural_total}"
    else:
        structural_flags = "unavailable"

    first_pass_counter = Counter(
        (row.get("extreme_band_first_pass_status") or "unreported").strip() or "unreported"
        for row in rows
    )
    model_label = next((row.get("model_label", "").strip() for row in rows if row.get("model_label")), "")
    model_id = next((row.get("model_id", "").strip() for row in rows if row.get("model_id")), "")

    return CrossModelSummary(
        run_id=run_dir.name,
        status="available",
        path=repo_relative(repo_root, csv_path),
        model_label=model_label or manifest_model_label or "unknown",
        model_id=model_id or manifest_model_id or "unknown",
        families=families,
        row_count=len(rows),
        structural_flags_all_true=structural_flags,
        first_pass_statuses=format_counter(first_pass_counter),
        notes=tuple(notes),
    )


def validate_manifest_paths(repo_root: Path, manifest_path: Path) -> list[CheckResult]:
    if not manifest_path.exists():
        return [CheckResult(LEVEL_WARN, repo_relative(repo_root, manifest_path), "manifest missing")]
    try:
        manifest = read_json(manifest_path)
    except (OSError, json.JSONDecodeError) as exc:
        return [CheckResult(LEVEL_FAIL, repo_relative(repo_root, manifest_path), f"manifest unreadable: {exc}")]

    path_map = manifest.get("paths")
    if not isinstance(path_map, dict):
        return [CheckResult(LEVEL_WARN, repo_relative(repo_root, manifest_path), "manifest has no paths map")]

    results: list[CheckResult] = []
    for label, relative_path in sorted(path_map.items()):
        if not isinstance(relative_path, str) or not relative_path:
            results.append(CheckResult(LEVEL_FAIL, repo_relative(repo_root, manifest_path), f"invalid path for {label}"))
            continue
        target = repo_root / relative_path
        if target.exists():
            results.append(CheckResult(LEVEL_PASS, f"manifest path {label}", relative_path))
        else:
            results.append(CheckResult(LEVEL_FAIL, f"manifest path {label}", f"missing {relative_path}"))
    return results


def append_missing(errors: list[str], field: str) -> None:
    errors.append(f"missing required field: {field}")


def require_str(
    payload: Mapping[str, Any],
    key: str,
    errors: list[str],
    prefix: str = "",
    allow_empty: bool = False,
) -> str | None:
    field = f"{prefix}.{key}" if prefix else key
    if key not in payload:
        append_missing(errors, field)
        return None
    value = payload[key]
    if not isinstance(value, str) or (not allow_empty and not value):
        errors.append(f"field {field} expected non-empty string")
        return None
    return value


def require_literal(payload: Mapping[str, Any], key: str, expected: Any, errors: list[str], prefix: str = "") -> None:
    field = f"{prefix}.{key}" if prefix else key
    if key not in payload:
        append_missing(errors, field)
        return
    if payload[key] != expected:
        errors.append(f"field {field} expected {expected!r}, got {payload[key]!r}")


def require_int(payload: Mapping[str, Any], key: str, errors: list[str], prefix: str = "", expected: int | None = None) -> int | None:
    field = f"{prefix}.{key}" if prefix else key
    if key not in payload:
        append_missing(errors, field)
        return None
    value = payload[key]
    if type(value) is not int:
        errors.append(f"field {field} expected integer")
        return None
    if expected is not None and value != expected:
        errors.append(f"field {field} expected {expected!r}, got {value!r}")
    return value


def require_optional_int(payload: Mapping[str, Any], key: str, errors: list[str], prefix: str = "") -> int | None:
    field = f"{prefix}.{key}" if prefix else key
    if key not in payload:
        append_missing(errors, field)
        return None
    value = payload[key]
    if value is None:
        return None
    if type(value) is not int:
        errors.append(f"field {field} expected integer or null")
        return None
    return value


def require_bool(payload: Mapping[str, Any], key: str, errors: list[str], prefix: str = "") -> bool | None:
    field = f"{prefix}.{key}" if prefix else key
    if key not in payload:
        append_missing(errors, field)
        return None
    value = payload[key]
    if type(value) is not bool:
        errors.append(f"field {field} expected boolean")
        return None
    return value


def require_optional_bool(payload: Mapping[str, Any], key: str, errors: list[str], prefix: str = "") -> bool | None:
    field = f"{prefix}.{key}" if prefix else key
    if key not in payload:
        append_missing(errors, field)
        return None
    value = payload[key]
    if value is None:
        return None
    if type(value) is not bool:
        errors.append(f"field {field} expected boolean or null")
        return None
    return value


def require_string_list(payload: Mapping[str, Any], key: str, errors: list[str], prefix: str = "") -> list[str] | None:
    field = f"{prefix}.{key}" if prefix else key
    if key not in payload:
        append_missing(errors, field)
        return None
    value = payload[key]
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        errors.append(f"field {field} expected list of strings")
        return None
    return value


def require_mapping(payload: Mapping[str, Any], key: str, errors: list[str], prefix: str = "") -> Mapping[str, Any] | None:
    field = f"{prefix}.{key}" if prefix else key
    if key not in payload:
        append_missing(errors, field)
        return None
    value = payload[key]
    if not isinstance(value, dict):
        errors.append(f"field {field} expected object")
        return None
    return value


def validate_created_at(payload: Mapping[str, Any], errors: list[str], prefix: str = "") -> None:
    value = require_str(payload, "created_at", errors, prefix)
    if value is None:
        return
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        field = f"{prefix}.created_at" if prefix else "created_at"
        errors.append(f"field {field} expected ISO-8601 timestamp")


def validate_fixed_target_set(payload: Mapping[str, Any], errors: list[str], prefix: str = "") -> None:
    fixed_target = require_mapping(payload, "fixed_target_set", errors, prefix)
    if fixed_target is None:
        return
    field_prefix = f"{prefix}.fixed_target_set" if prefix else "fixed_target_set"
    expected = l4_smoke_fixed_target_set()
    for key in ("boundary", "model_id", "model_label", "device"):
        require_literal(fixed_target, key, expected[key], errors, field_prefix)
    families = require_string_list(fixed_target, "families", errors, field_prefix)
    if families is not None and families != expected["families"]:
        errors.append(f"field {field_prefix}.families expected {expected['families']!r}, got {families!r}")


def validate_preflight_posture_fields(payload: Mapping[str, Any], errors: list[str], prefix: str = "") -> None:
    require_str(payload, "sys_executable", errors, prefix)
    require_str(payload, "python_version", errors, prefix)
    require_str(payload, "cwd", errors, prefix)
    require_str(payload, "platform", errors, prefix)
    require_str(payload, "os_name", errors, prefix)
    require_bool(payload, "torch_importable", errors, prefix)
    require_str(payload, "torch_version", errors, prefix)
    require_optional_bool(payload, "torch_cuda_available", errors, prefix)
    require_str(payload, "torch_cuda_version", errors, prefix)
    require_optional_int(payload, "gpu_count", errors, prefix)
    require_string_list(payload, "gpu_names", errors, prefix)
    require_bool(payload, "nvidia_smi_available", errors, prefix)
    require_str(payload, "nvidia_smi_path", errors, prefix, allow_empty=True)
    require_string_list(payload, "nvidia_smi_summary", errors, prefix)
    require_str(payload, "nvidia_smi_error", errors, prefix, allow_empty=True)
    posture = require_str(payload, "posture_classification", errors, prefix)
    if posture is not None and posture not in POSTURE_CLASSIFICATIONS:
        field = f"{prefix}.posture_classification" if prefix else "posture_classification"
        errors.append(f"field {field} expected one of {', '.join(POSTURE_CLASSIFICATIONS)}, got {posture!r}")
    require_bool(payload, "preflight_ok", errors, prefix)
    require_string_list(payload, "remediation_hints", errors, prefix)
    require_string_list(payload, "errors", errors, prefix)


def validate_preflight_artifact_payload(payload: Any, prefix: str = "") -> tuple[str, ...]:
    if not isinstance(payload, dict):
        return (f"{prefix or 'artifact'} expected object",)
    errors: list[str] = []
    require_literal(payload, "schema_id", L4_SMOKE_PREFLIGHT_SCHEMA_ID, errors, prefix)
    require_int(payload, "schema_version", errors, prefix, expected=ARTIFACT_CONTRACT_VERSION)
    validate_created_at(payload, errors, prefix)
    require_literal(payload, "tier", Tier.L4_SMOKE.value, errors, prefix)
    mode = require_str(payload, "mode", errors, prefix)
    if mode is not None and mode not in PREFLIGHT_ARTIFACT_MODES:
        field = f"{prefix}.mode" if prefix else "mode"
        errors.append(f"field {field} expected one of {', '.join(PREFLIGHT_ARTIFACT_MODES)}, got {mode!r}")
    validate_fixed_target_set(payload, errors, prefix)
    validate_preflight_posture_fields(payload, errors, prefix)
    result = require_str(payload, "result", errors, prefix)
    if result is not None and result not in RESULT_VALUES:
        field = f"{prefix}.result" if prefix else "result"
        errors.append(f"field {field} expected one of {', '.join(RESULT_VALUES)}, got {result!r}")
    preflight_ok = payload.get("preflight_ok")
    if type(preflight_ok) is bool and result in RESULT_VALUES and result != result_from_bool(preflight_ok):
        field = f"{prefix}.result" if prefix else "result"
        errors.append(f"field {field} does not match preflight_ok={preflight_ok!r}")
    return tuple(errors)


def validate_downstream_dispatch_summary(payload: Mapping[str, Any], errors: list[str], prefix: str) -> None:
    summary = require_mapping(payload, "downstream_dispatch_summary", errors, prefix)
    if summary is None:
        return
    field_prefix = f"{prefix}.downstream_dispatch_summary" if prefix else "downstream_dispatch_summary"
    require_int(summary, "subprocess_returncode", errors, field_prefix)
    require_int(summary, "families_expected", errors, field_prefix, expected=len(L4_SMOKE_CONFIG.families))
    require_int(summary, "families_reported", errors, field_prefix)
    require_int(summary, "fail", errors, field_prefix)
    result = require_str(summary, "result", errors, field_prefix)
    if result is not None and result not in RESULT_VALUES:
        errors.append(f"field {field_prefix}.result expected one of {', '.join(RESULT_VALUES)}, got {result!r}")


def validate_status_artifact_payload(payload: Any, prefix: str = "") -> tuple[str, ...]:
    if not isinstance(payload, dict):
        return (f"{prefix or 'artifact'} expected object",)
    errors: list[str] = []
    require_literal(payload, "schema_id", L4_SMOKE_STATUS_SCHEMA_ID, errors, prefix)
    require_int(payload, "schema_version", errors, prefix, expected=ARTIFACT_CONTRACT_VERSION)
    validate_created_at(payload, errors, prefix)
    require_literal(payload, "tier", Tier.L4_SMOKE.value, errors, prefix)
    require_literal(payload, "mode", "execute", errors, prefix)
    validate_fixed_target_set(payload, errors, prefix)
    require_str(payload, "entrypoint", errors, prefix)
    require_string_list(payload, "command", errors, prefix)
    require_str(payload, "out_dir", errors, prefix)
    require_int(payload, "returncode", errors, prefix)
    result = require_str(payload, "result", errors, prefix)
    if result is not None and result not in RESULT_VALUES:
        field = f"{prefix}.result" if prefix else "result"
        errors.append(f"field {field} expected one of {', '.join(RESULT_VALUES)}, got {result!r}")
    preflight = require_mapping(payload, "preflight", errors, prefix)
    if preflight is not None:
        errors.extend(validate_preflight_artifact_payload(preflight, f"{prefix}.preflight" if prefix else "preflight"))
    validate_downstream_dispatch_summary(payload, errors, prefix)
    family_results = payload.get("family_results")
    field = f"{prefix}.family_results" if prefix else "family_results"
    if "family_results" not in payload:
        append_missing(errors, field)
    elif not isinstance(family_results, list) or any(not isinstance(item, dict) for item in family_results):
        errors.append(f"field {field} expected list of objects")
    require_string_list(payload, "notes", errors, prefix)
    downstream = payload.get("downstream_dispatch_summary")
    if isinstance(downstream, dict) and result in RESULT_VALUES and downstream.get("result") in RESULT_VALUES and result != downstream.get("result"):
        field = f"{prefix}.result" if prefix else "result"
        errors.append(f"field {field} does not match downstream_dispatch_summary.result={downstream.get('result')!r}")
    return tuple(errors)


def validate_l4_weekly_target_matrix(payload: Mapping[str, Any], errors: list[str], prefix: str = "") -> None:
    field = f"{prefix}.weekly_target_matrix" if prefix else "weekly_target_matrix"
    if "weekly_target_matrix" not in payload:
        append_missing(errors, field)
        return
    target_matrix = payload["weekly_target_matrix"]
    if not isinstance(target_matrix, list) or any(not isinstance(item, dict) for item in target_matrix):
        errors.append(f"field {field} expected list of objects")
        return
    expected = l4_weekly_target_matrix()
    if target_matrix != expected:
        errors.append(f"field {field} expected current 3B/4B dense-transformer mainline matrix")


def validate_l4_weekly_plan_artifact_payload(payload: Any, prefix: str = "") -> tuple[str, ...]:
    if not isinstance(payload, dict):
        return (f"{prefix or 'artifact'} expected object",)
    errors: list[str] = []
    require_literal(payload, "schema_id", L4_WEEKLY_PLAN_SCHEMA_ID, errors, prefix)
    require_int(payload, "schema_version", errors, prefix, expected=ARTIFACT_CONTRACT_VERSION)
    validate_created_at(payload, errors, prefix)
    require_literal(payload, "tier", Tier.L4_WEEKLY.value, errors, prefix)
    require_literal(payload, "mode", "plan-only", errors, prefix)
    require_str(payload, "resource_posture", errors, prefix)
    validate_l4_weekly_target_matrix(payload, errors, prefix)
    entrypoints = require_string_list(payload, "planned_entrypoints", errors, prefix)
    expected_entrypoints = [
        "tools/run_gate12a_cross_model_replay.py",
        "tools/run_gate8_scaleup.py",
        "tools/run_gate12a_family_replay.py",
    ]
    if entrypoints is not None and entrypoints != expected_entrypoints:
        field = f"{prefix}.planned_entrypoints" if prefix else "planned_entrypoints"
        errors.append(f"field {field} expected {expected_entrypoints!r}, got {entrypoints!r}")
    exclusions = require_string_list(payload, "exclusions", errors, prefix)
    if exclusions is not None and exclusions != list(L4_WEEKLY_EXCLUSIONS):
        field = f"{prefix}.exclusions" if prefix else "exclusions"
        errors.append(f"field {field} expected {list(L4_WEEKLY_EXCLUSIONS)!r}, got {exclusions!r}")
    result = require_str(payload, "result", errors, prefix)
    if result is not None and result != "plan-only":
        field = f"{prefix}.result" if prefix else "result"
        errors.append(f"field {field} expected 'plan-only', got {result!r}")
    return tuple(errors)


def validation_summary(payload: Mapping[str, Any], artifact_kind: str) -> tuple[str, str, str, str]:
    schema_id = str(payload.get("schema_id", ""))
    mode = str(payload.get("mode", ""))
    result = str(payload.get("result", ""))
    posture = ""
    downstream_result = ""
    if artifact_kind == "preflight":
        posture = str(payload.get("posture_classification", ""))
    else:
        preflight = payload.get("preflight")
        if isinstance(preflight, dict):
            posture = str(preflight.get("posture_classification", ""))
        downstream = payload.get("downstream_dispatch_summary")
        if isinstance(downstream, dict):
            downstream_result = str(downstream.get("result", ""))
    return schema_id, mode, result, posture or "n/a", downstream_result or "n/a"


def validate_eval_factory_artifact_file(repo_root: Path, path: Path, artifact_kind: str) -> EvalFactoryArtifactValidation:
    source_class = SOURCE_EVAL_FACTORY_PREFLIGHT if artifact_kind == "preflight" else SOURCE_EVAL_FACTORY_STATUS
    relative_path = repo_relative(repo_root, path)
    if not path.exists():
        return EvalFactoryArtifactValidation(
            source_class=source_class,
            artifact_kind=artifact_kind,
            path=relative_path,
            status=ARTIFACT_STATUS_MISSING,
            schema_id="",
            mode="",
            result="",
            posture_classification="",
            downstream_result="",
            errors=(f"missing artifact: {relative_path}",),
        )
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return EvalFactoryArtifactValidation(
            source_class=source_class,
            artifact_kind=artifact_kind,
            path=relative_path,
            status=ARTIFACT_STATUS_MALFORMED,
            schema_id="",
            mode="",
            result="",
            posture_classification="",
            downstream_result="",
            errors=(f"artifact unreadable: {exc}",),
        )

    errors = (
        validate_preflight_artifact_payload(payload)
        if artifact_kind == "preflight"
        else validate_status_artifact_payload(payload)
    )
    schema_id, mode, result, posture, downstream_result = validation_summary(payload, artifact_kind) if isinstance(payload, dict) else ("", "", "", "", "")
    return EvalFactoryArtifactValidation(
        source_class=source_class,
        artifact_kind=artifact_kind,
        path=relative_path,
        status=ARTIFACT_STATUS_VALID if not errors else ARTIFACT_STATUS_MALFORMED,
        schema_id=schema_id,
        mode=mode,
        result=result,
        posture_classification=posture,
        downstream_result=downstream_result,
        errors=errors,
    )


def validate_eval_factory_preflight_artifact(repo_root: Path, path: Path) -> EvalFactoryArtifactValidation:
    return validate_eval_factory_artifact_file(repo_root, path, "preflight")


def validate_eval_factory_status_artifact(repo_root: Path, path: Path) -> EvalFactoryArtifactValidation:
    return validate_eval_factory_artifact_file(repo_root, path, "status")


def discover_and_validate_eval_factory_artifacts(repo_root: Path) -> tuple[tuple[EvalFactoryArtifactValidation, ...], tuple[EvalFactoryArtifactValidation, ...]]:
    preflight_results = tuple(
        validate_eval_factory_preflight_artifact(repo_root, path)
        for path in discover_eval_factory_preflight_artifacts(repo_root)
    )
    status_results = tuple(
        validate_eval_factory_status_artifact(repo_root, path)
        for path in discover_eval_factory_status_artifacts(repo_root)
    )
    return preflight_results, status_results


def artifact_check_detail(result: EvalFactoryArtifactValidation) -> str:
    if result.status == ARTIFACT_STATUS_VALID:
        parts = [
            f"path={result.path}",
            f"schema={result.schema_id}",
            f"mode={result.mode}",
            f"result={result.result}",
            f"posture={result.posture_classification}",
        ]
        if result.artifact_kind == "status":
            parts.append(f"downstream_result={result.downstream_result}")
        return "; ".join(parts)
    return (
        f"path={result.path}; errors="
        + " | ".join(result.errors)
        + "; helper=python tools/manage_eval_factory_artifacts.py --root runs"
    )


def append_eval_factory_artifact_checks(checks: list[CheckResult], repo_root: Path) -> None:
    preflight_results, status_results = discover_and_validate_eval_factory_artifacts(repo_root)

    if not preflight_results:
        checks.append(CheckResult(LEVEL_WARN, SOURCE_EVAL_FACTORY_PREFLIGHT, f"optional missing: {L4_SMOKE_PREFLIGHT_FILENAME}"))
    else:
        for result in preflight_results:
            level = LEVEL_PASS if result.status == ARTIFACT_STATUS_VALID else LEVEL_FAIL
            checks.append(CheckResult(level, f"{SOURCE_EVAL_FACTORY_PREFLIGHT} {result.path}", artifact_check_detail(result)))

    if not status_results:
        checks.append(CheckResult(LEVEL_WARN, SOURCE_EVAL_FACTORY_STATUS, f"optional missing: {L4_SMOKE_STATUS_FILENAME}"))
    else:
        for result in status_results:
            level = LEVEL_PASS if result.status == ARTIFACT_STATUS_VALID else LEVEL_FAIL
            checks.append(CheckResult(level, f"{SOURCE_EVAL_FACTORY_STATUS} {result.path}", artifact_check_detail(result)))


def build_cpu_nightly_checks(repo_root: Path) -> list[CheckResult]:
    checks: list[CheckResult] = []

    for relative_path in REQUIRED_CPU_FILES:
        path = repo_root / relative_path
        level = LEVEL_PASS if path.exists() else LEVEL_FAIL
        detail = "present" if path.exists() else "missing"
        checks.append(CheckResult(level, relative_path, detail))

    missing_memos = [memo for memo in EXPECTED_ATLAS_MEMOS if not (repo_root / "workstream" / memo).exists()]
    if missing_memos:
        checks.append(CheckResult(LEVEL_FAIL, "tracked Gate12A atlas memos", "missing " + ", ".join(missing_memos)))
    else:
        checks.append(CheckResult(LEVEL_PASS, "tracked Gate12A atlas memos", f"{len(EXPECTED_ATLAS_MEMOS)}/{len(EXPECTED_ATLAS_MEMOS)} present"))

    checks.append(
        CheckResult(
            LEVEL_PASS if TIER_VALUES == ("cpu-nightly", "l4-smoke", "l4-weekly", "summarize-existing") else LEVEL_FAIL,
            "eval-factory tier surface",
            ", ".join(TIER_VALUES),
        )
    )

    weekly_plan = plan_l4_weekly()
    required_exclusions = L4_WEEKLY_EXCLUSIONS
    missing_exclusions = [item for item in required_exclusions if item not in weekly_plan.out_of_scope]
    if missing_exclusions:
        checks.append(CheckResult(LEVEL_FAIL, "l4-weekly exclusions", "missing " + ", ".join(missing_exclusions)))
    else:
        checks.append(CheckResult(LEVEL_PASS, "l4-weekly exclusions", ", ".join(required_exclusions)))

    append_eval_factory_artifact_checks(checks, repo_root)

    summary_dirs = discover_summary_dirs(repo_root)
    if not (repo_root / "runs").exists():
        checks.append(CheckResult(LEVEL_WARN, "runs/", "missing; skipping materialized summary checks"))
        return checks
    if not summary_dirs:
        checks.append(CheckResult(LEVEL_WARN, "materialized cross-model summaries", "none found"))
        return checks

    discovered_names = {path.name for path in summary_dirs}
    missing_expected_summaries = [name for name in EXPECTED_SUMMARY_RUNS if name not in discovered_names]
    if missing_expected_summaries:
        checks.append(
            CheckResult(
                LEVEL_WARN,
                "expected cross-model summary dirs",
                "missing " + ", ".join(missing_expected_summaries),
            )
        )
    else:
        checks.append(CheckResult(LEVEL_PASS, "expected cross-model summary dirs", f"{len(EXPECTED_SUMMARY_RUNS)}/{len(EXPECTED_SUMMARY_RUNS)} present"))

    for summary_dir in summary_dirs:
        summary = parse_cross_model_summary(repo_root, summary_dir)
        if summary.status == "available":
            checks.append(CheckResult(LEVEL_PASS, f"{summary.run_id} summary", f"rows={summary.row_count}; families={', '.join(summary.families) or 'none'}"))
        elif summary.status == "missing-summary":
            checks.append(CheckResult(LEVEL_WARN, f"{summary.run_id} summary", "; ".join(summary.notes)))
        else:
            checks.append(CheckResult(LEVEL_FAIL, f"{summary.run_id} summary", "; ".join(summary.notes)))

        for note in summary.notes:
            level = LEVEL_WARN if "missing families:" in note or "manifest missing" in note else LEVEL_FAIL
            checks.append(CheckResult(level, f"{summary.run_id} note", note))

        checks.extend(validate_manifest_paths(repo_root, summary_dir / MANIFEST_FILENAME))

    return checks


def render_check_report(plan: TierPlan, checks: Sequence[CheckResult]) -> str:
    counts = Counter(check.level for check in checks)
    lines = [
        f"tier: {plan.tier.value}",
        f"intent: {plan.intent}",
        f"expected resource posture: {plan.resource_posture}",
        "status summary:",
        f"  pass: {counts[LEVEL_PASS]}",
        f"  warn: {counts[LEVEL_WARN]}",
        f"  fail: {counts[LEVEL_FAIL]}",
        "checks:",
    ]
    for check in checks:
        detail = f" - {check.detail}" if check.detail else ""
        lines.append(f"  [{check.level}] {check.label}{detail}")
    if counts[LEVEL_FAIL]:
        lines.append("result: fail")
    elif counts[LEVEL_WARN]:
        lines.append("result: pass-with-warnings")
    else:
        lines.append("result: pass")
    return "\n".join(lines)


def load_torch_module() -> Any:
    return importlib.import_module("torch")


def classify_l4_smoke_posture(
    os_name: str,
    torch_importable: bool,
    torch_cuda_available: bool | None,
    gpu_count: int | None,
) -> str:
    normalized_os = os_name.lower()
    if not torch_importable:
        return POSTURE_PYTHON_MISSING_TORCH
    if normalized_os == "windows" and not torch_cuda_available:
        return POSTURE_LOCAL_WINDOWS_NO_CUDA
    if normalized_os == "linux" and torch_cuda_available and (gpu_count or 0) > 0:
        return POSTURE_REMOTE_CUDA_READY
    if torch_cuda_available is False:
        return POSTURE_CUDA_UNAVAILABLE
    return POSTURE_UNKNOWN


def remediation_hints_for_posture(posture: str) -> tuple[str, ...]:
    common = (
        'Check `nvidia-smi` on the target machine.',
        'Check `python -c "import torch; print(torch.cuda.is_available())"` in the same environment.',
    )
    if posture == POSTURE_REMOTE_CUDA_READY:
        return ()
    if posture == POSTURE_LOCAL_WINDOWS_NO_CUDA:
        return (
            "Run this lane on the GCP L4 VM instead of local Windows.",
            *common,
            "Confirm the VM Python interpreter is the one used by `tools/run_eval_checks.py`.",
        )
    if posture == POSTURE_PYTHON_MISSING_TORCH:
        return (
            "Use the GCP L4 VM environment with a CUDA-capable PyTorch installation.",
            "Install or activate the Python environment that provides `torch` before executing the lane.",
            *common,
        )
    if posture == POSTURE_CUDA_UNAVAILABLE:
        return (
            "CUDA was requested for l4-smoke but is not available to this Python interpreter.",
            *common,
            "Confirm NVIDIA drivers, CUDA runtime visibility, and the PyTorch build on the VM.",
        )
    return (
        "Posture is mixed or unknown; verify that this is the intended GCP L4 VM environment.",
        *common,
    )


def query_nvidia_smi(
    nvidia_smi_path: str,
    run_command: Callable[..., subprocess.CompletedProcess[str]],
) -> tuple[tuple[str, ...], str]:
    try:
        completed = run_command(
            [
                nvidia_smi_path,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return (), str(exc)
    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    if completed.returncode != 0:
        return (), stderr or f"nvidia-smi exited with rc={completed.returncode}"
    if not stdout:
        return (), ""
    return tuple(line.strip() for line in stdout.splitlines() if line.strip()), ""


def collect_l4_smoke_preflight(
    repo_root: Path,
    torch_loader: Callable[[], Any] = load_torch_module,
    platform_system: Callable[[], str] = platform.system,
    platform_string: Callable[[], str] = platform.platform,
    cwd_getter: Callable[[], Path] = Path.cwd,
    which: Callable[[str], str | None] = shutil.which,
    run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> L4SmokePreflight:
    errors: list[str] = []
    os_name = platform_system()
    torch_importable = False
    torch_version = "unavailable"
    torch_cuda_available: bool | None = None
    torch_cuda_version = "unavailable"
    gpu_count: int | None = None
    gpu_names: tuple[str, ...] = ()

    try:
        torch_module = torch_loader()
        torch_importable = True
        torch_version = str(getattr(torch_module, "__version__", "unknown"))
        torch_version_obj = getattr(torch_module, "version", None)
        torch_cuda_version = str(getattr(torch_version_obj, "cuda", "unavailable") or "unavailable")
        cuda_obj = getattr(torch_module, "cuda", None)
        if cuda_obj is None:
            errors.append("torch.cuda is not present")
            torch_cuda_available = False
            gpu_count = 0
        else:
            torch_cuda_available = bool(cuda_obj.is_available())
            try:
                gpu_count = int(cuda_obj.device_count()) if torch_cuda_available else 0
            except (AttributeError, TypeError, ValueError) as exc:
                gpu_count = None
                errors.append(f"torch.cuda.device_count unavailable: {exc}")
            names: list[str] = []
            if torch_cuda_available and gpu_count:
                for index in range(gpu_count):
                    try:
                        names.append(str(cuda_obj.get_device_name(index)))
                    except (AttributeError, RuntimeError, TypeError) as exc:
                        names.append(f"unavailable:{exc}")
            gpu_names = tuple(names)
    except ImportError as exc:
        errors.append(f"torch import failed: {exc}")
    except Exception as exc:  # pragma: no cover - defensive against broken torch installs
        errors.append(f"torch inspection failed: {exc}")

    nvidia_smi_path = which("nvidia-smi") or ""
    nvidia_smi_available = bool(nvidia_smi_path)
    nvidia_smi_summary: tuple[str, ...] = ()
    nvidia_smi_error = ""
    if nvidia_smi_available:
        nvidia_smi_summary, nvidia_smi_error = query_nvidia_smi(nvidia_smi_path, run_command)

    posture = classify_l4_smoke_posture(
        os_name=os_name,
        torch_importable=torch_importable,
        torch_cuda_available=torch_cuda_available,
        gpu_count=gpu_count,
    )
    preflight_ok = posture == POSTURE_REMOTE_CUDA_READY
    if not preflight_ok:
        errors.append(f"posture classification is {posture}, expected {POSTURE_REMOTE_CUDA_READY}")

    return L4SmokePreflight(
        sys_executable=sys.executable,
        python_version=sys.version.replace("\n", " "),
        cwd=str(cwd_getter()),
        platform=platform_string(),
        os_name=os_name,
        torch_importable=torch_importable,
        torch_version=torch_version,
        torch_cuda_available=torch_cuda_available,
        torch_cuda_version=torch_cuda_version,
        gpu_count=gpu_count,
        gpu_names=gpu_names,
        nvidia_smi_available=nvidia_smi_available,
        nvidia_smi_path=nvidia_smi_path,
        nvidia_smi_summary=nvidia_smi_summary,
        nvidia_smi_error=nvidia_smi_error,
        posture_classification=posture,
        preflight_ok=preflight_ok,
        remediation_hints=remediation_hints_for_posture(posture),
        errors=tuple(errors),
    )


def preflight_to_dict(preflight: L4SmokePreflight) -> dict[str, Any]:
    return {
        "sys_executable": preflight.sys_executable,
        "python_version": preflight.python_version,
        "cwd": preflight.cwd,
        "platform": preflight.platform,
        "os_name": preflight.os_name,
        "torch_importable": preflight.torch_importable,
        "torch_version": preflight.torch_version,
        "torch_cuda_available": preflight.torch_cuda_available,
        "torch_cuda_version": preflight.torch_cuda_version,
        "gpu_count": preflight.gpu_count,
        "gpu_names": list(preflight.gpu_names),
        "nvidia_smi_available": preflight.nvidia_smi_available,
        "nvidia_smi_path": preflight.nvidia_smi_path,
        "nvidia_smi_summary": list(preflight.nvidia_smi_summary),
        "nvidia_smi_error": preflight.nvidia_smi_error,
        "posture_classification": preflight.posture_classification,
        "preflight_ok": preflight.preflight_ok,
        "remediation_hints": list(preflight.remediation_hints),
        "errors": list(preflight.errors),
    }


def build_preflight_artifact_payload(
    preflight: L4SmokePreflight,
    mode: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_id": L4_SMOKE_PREFLIGHT_SCHEMA_ID,
        "schema_version": ARTIFACT_CONTRACT_VERSION,
        "created_at": created_at or utc_created_at(),
        "tier": Tier.L4_SMOKE.value,
        "mode": mode,
        "fixed_target_set": l4_smoke_fixed_target_set(),
        **preflight_to_dict(preflight),
        "result": result_from_bool(preflight.preflight_ok),
    }


def preflight_artifact_path(out_dir: Path) -> Path:
    return out_dir / L4_SMOKE_PREFLIGHT_FILENAME


def write_preflight_artifact(out_dir: Path, preflight: L4SmokePreflight, mode: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_status_artifact(preflight_artifact_path(out_dir), build_preflight_artifact_payload(preflight, mode))


def render_l4_smoke_preflight(preflight: L4SmokePreflight) -> str:
    lines = [
        "environment diagnostics:",
        f"  sys.executable: {preflight.sys_executable}",
        f"  python_version: {preflight.python_version}",
        f"  cwd: {preflight.cwd}",
        f"  platform: {preflight.platform}",
        f"  os_name: {preflight.os_name}",
        f"  torch_importable: {preflight.torch_importable}",
        f"  torch.__version__: {preflight.torch_version}",
        f"  torch.cuda.is_available(): {preflight.torch_cuda_available}",
        f"  torch.version.cuda: {preflight.torch_cuda_version}",
        f"  gpu_count: {preflight.gpu_count}",
        f"  gpu_names: {', '.join(preflight.gpu_names) if preflight.gpu_names else 'none'}",
        f"  nvidia-smi available: {preflight.nvidia_smi_available}",
        f"  nvidia-smi path: {preflight.nvidia_smi_path or 'none'}",
        "  nvidia-smi summary:",
    ]
    if preflight.nvidia_smi_summary:
        lines.extend(f"    - {row}" for row in preflight.nvidia_smi_summary)
    else:
        lines.append("    - none")
    if preflight.nvidia_smi_error:
        lines.append(f"  nvidia-smi error: {preflight.nvidia_smi_error}")
    lines.extend(
        [
            "posture classification:",
            f"  classification: {preflight.posture_classification}",
            "preflight result:",
            f"  result: {'pass' if preflight.preflight_ok else 'fail'}",
        ]
    )
    if preflight.errors:
        lines.append("  errors:")
        lines.extend(f"    - {error}" for error in preflight.errors)
    lines.append("remediation hints:")
    if preflight.remediation_hints:
        lines.extend(f"  - {hint}" for hint in preflight.remediation_hints)
    else:
        lines.append("  - none")
    return "\n".join(lines)


def l4_smoke_entrypoints(repo_root: Path) -> tuple[Path, ...]:
    return (
        repo_root / "tools" / "run_gate12a_cross_model_replay.py",
        repo_root / "tools" / "run_gate8_scaleup.py",
        repo_root / "tools" / "run_gate12a_family_replay.py",
    )


def l4_weekly_entrypoints(repo_root: Path) -> tuple[Path, ...]:
    return (
        repo_root / "tools" / "run_gate12a_cross_model_replay.py",
        repo_root / "tools" / "run_gate8_scaleup.py",
        repo_root / "tools" / "run_gate12a_family_replay.py",
    )


def build_l4_weekly_plan_payload(repo_root: Path, created_at: str | None = None) -> dict[str, Any]:
    return {
        "schema_id": L4_WEEKLY_PLAN_SCHEMA_ID,
        "schema_version": ARTIFACT_CONTRACT_VERSION,
        "created_at": created_at or utc_created_at(),
        "tier": Tier.L4_WEEKLY.value,
        "mode": "plan-only",
        "resource_posture": plan_l4_weekly().resource_posture,
        "weekly_target_matrix": l4_weekly_target_matrix(),
        "planned_entrypoints": [repo_relative(repo_root, path) for path in l4_weekly_entrypoints(repo_root)],
        "exclusions": list(L4_WEEKLY_EXCLUSIONS),
        "result": "plan-only",
    }


def l4_weekly_plan_artifact_path(out_dir: Path) -> Path:
    return out_dir / L4_WEEKLY_PLAN_FILENAME


def write_l4_weekly_plan_artifact(repo_root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_status_artifact(l4_weekly_plan_artifact_path(out_dir), build_l4_weekly_plan_payload(repo_root))


def validate_l4_weekly_plan_artifact(repo_root: Path, path: Path) -> EvalFactoryArtifactValidation:
    relative_path = repo_relative(repo_root, path)
    if not path.exists():
        return EvalFactoryArtifactValidation(
            source_class="eval-factory weekly plan artifact",
            artifact_kind="weekly-plan",
            path=relative_path,
            status=ARTIFACT_STATUS_MISSING,
            schema_id="",
            mode="",
            result="",
            posture_classification="n/a",
            downstream_result="n/a",
            errors=(f"missing artifact: {relative_path}",),
        )
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return EvalFactoryArtifactValidation(
            source_class="eval-factory weekly plan artifact",
            artifact_kind="weekly-plan",
            path=relative_path,
            status=ARTIFACT_STATUS_MALFORMED,
            schema_id="",
            mode="",
            result="",
            posture_classification="n/a",
            downstream_result="n/a",
            errors=(f"artifact unreadable: {exc}",),
        )
    errors = validate_l4_weekly_plan_artifact_payload(payload)
    schema_id = str(payload.get("schema_id", "")) if isinstance(payload, dict) else ""
    mode = str(payload.get("mode", "")) if isinstance(payload, dict) else ""
    result = str(payload.get("result", "")) if isinstance(payload, dict) else ""
    return EvalFactoryArtifactValidation(
        source_class="eval-factory weekly plan artifact",
        artifact_kind="weekly-plan",
        path=relative_path,
        status=ARTIFACT_STATUS_VALID if not errors else ARTIFACT_STATUS_MALFORMED,
        schema_id=schema_id,
        mode=mode,
        result=result,
        posture_classification="n/a",
        downstream_result="n/a",
        errors=errors,
    )


def render_l4_weekly_plan(repo_root: Path, out_dir: Path | None = None) -> str:
    plan = plan_l4_weekly()
    payload = build_l4_weekly_plan_payload(repo_root, created_at="<runtime>")
    lines = [
        f"tier: {Tier.L4_WEEKLY.value}",
        "mode: plan-only",
        f"intent: {plan.intent}",
        f"expected resource posture: {plan.resource_posture}",
        "weekly target matrix:",
    ]
    for target in payload["weekly_target_matrix"]:
        lines.append(
            "  - "
            f"model={target['model_id']}; model_label={target['model_label']}; "
            f"families={', '.join(target['families'])}"
        )
    lines.append("per-model planned families:")
    for target in payload["weekly_target_matrix"]:
        lines.append(f"  - {target['model_id']}: " + ", ".join(target["families"]))
    lines.append("planned entrypoints:")
    lines.extend(f"  - {entrypoint}" for entrypoint in payload["planned_entrypoints"])
    lines.append("exclusions:")
    lines.extend(f"  - {exclusion}" for exclusion in payload["exclusions"])
    lines.append("artifact:")
    if out_dir is None:
        lines.append(f"  - not written; pass --out-dir <path> to write {L4_WEEKLY_PLAN_FILENAME}")
    else:
        lines.append(f"  - {l4_weekly_plan_artifact_path(out_dir)}")
    lines.extend(
        [
            "execution:",
            "  - no subprocess execution",
            "  - no GPU/model execution",
            "final result:",
            "  result: plan-only",
        ]
    )
    return "\n".join(lines)


def run_l4_weekly(repo_root: Path, out_dir: Path | None) -> int:
    if out_dir is not None:
        write_l4_weekly_plan_artifact(repo_root, out_dir)
    print(render_l4_weekly_plan(repo_root, out_dir))
    return 0


def build_l4_smoke_command(repo_root: Path, out_dir: Path) -> list[str]:
    entrypoint = repo_root / "tools" / "run_gate12a_cross_model_replay.py"
    return [
        sys.executable,
        str(entrypoint.resolve()),
        "--model-id",
        L4_SMOKE_CONFIG.model_id,
        "--model-label",
        L4_SMOKE_CONFIG.model_label,
        "--families",
        *L4_SMOKE_CONFIG.families,
        "--device",
        L4_SMOKE_CONFIG.device,
        "--topk",
        str(L4_SMOKE_CONFIG.topk),
        "--seed",
        str(L4_SMOKE_CONFIG.seed),
        "--gate12a-top-k",
        str(L4_SMOKE_CONFIG.gate12a_top_k),
        "--balanced-per-band",
        str(L4_SMOKE_CONFIG.balanced_per_band),
        "--reading-limit",
        str(L4_SMOKE_CONFIG.reading_limit),
        "--out-root",
        str(out_dir),
        "--summary-run-id",
        L4_SMOKE_CONFIG.summary_run_id,
    ]


def render_l4_smoke_header(repo_root: Path, out_dir: Path | None, mode: str) -> str:
    out_dir_text = str(out_dir) if out_dir is not None else "<required for --execute>"
    command = build_l4_smoke_command(repo_root, out_dir or Path("<out-dir>"))
    lines = [
        f"tier: {Tier.L4_SMOKE.value}",
        f"mode: {mode}",
        "fixed target set:",
        f"  boundary: {L4_SMOKE_BOUNDARY}",
        f"  model: {L4_SMOKE_CONFIG.model_id}",
        "  families: " + ", ".join(L4_SMOKE_CONFIG.families),
        f"  device: {L4_SMOKE_CONFIG.device}",
        "actual entrypoints selected:",
    ]
    lines.extend(f"  - {repo_relative(repo_root, path)}" for path in l4_smoke_entrypoints(repo_root))
    lines.extend(
        [
            f"out-dir: {out_dir_text}",
            "planned command:",
            "  " + " ".join(command),
            "out of scope:",
            "  - 1B / 1.5B / 3B / 4B execution",
            "  - 7B FP32",
        ]
    )
    lines.extend(f"  - {item}" for item in L4_WEEKLY_EXCLUDED_CANDIDATES)
    return "\n".join(lines)


def render_l4_smoke_dry_run(repo_root: Path) -> str:
    return "\n".join(
        [
            render_l4_smoke_header(repo_root, None, "dry-run"),
            "dispatch:",
            "  - not executed; pass --preflight-only for GPU posture diagnostics",
            "  - pass --execute --out-dir <path> to run the committed l4-smoke lane after preflight",
            "final summary:",
            "  result: dry-run",
        ]
    )


def validate_l4_smoke_execute_preconditions(repo_root: Path, out_dir: Path | None) -> list[str]:
    errors: list[str] = []
    if out_dir is None:
        errors.append("--out-dir is required for --tier l4-smoke --execute")
    elif out_dir.exists() and not out_dir.is_dir():
        errors.append(f"--out-dir points to a file, not a directory: {out_dir}")

    missing_entrypoints = [repo_relative(repo_root, path) for path in l4_smoke_entrypoints(repo_root) if not path.exists()]
    if missing_entrypoints:
        errors.append("missing committed entrypoint(s): " + ", ".join(missing_entrypoints))
    return errors


def status_artifact_path(out_dir: Path) -> Path:
    return out_dir / L4_SMOKE_STATUS_FILENAME


def write_status_artifact(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def parse_l4_smoke_family_results(repo_root: Path, out_dir: Path) -> tuple[list[dict[str, str]], list[str]]:
    summary_dir = out_dir / L4_SMOKE_CONFIG.summary_run_id
    summary_path = summary_dir / CROSS_MODEL_SUMMARY_FILENAME
    if not summary_path.exists():
        return [], [f"missing summary: {repo_relative(repo_root, summary_path)}"]
    try:
        rows, fieldnames = read_csv_rows(summary_path)
    except (OSError, csv.Error, UnicodeDecodeError) as exc:
        return [], [f"summary unreadable: {exc}"]

    notes: list[str] = []
    missing_columns = [
        column
        for column in ("rendering_family", "model_id", *STRUCTURAL_FLAG_COLUMNS, "extreme_band_first_pass_status")
        if column not in fieldnames
    ]
    if missing_columns:
        notes.append("missing columns: " + ", ".join(missing_columns))

    by_family = {str(row.get("rendering_family", "")).strip(): row for row in rows}
    results: list[dict[str, str]] = []
    for family in L4_SMOKE_CONFIG.families:
        row = by_family.get(family)
        if row is None:
            results.append(
                {
                    "family": family,
                    "dispatch": "missing-summary-row",
                    "structural_flags_all_true": "n/a",
                    "runs_first_pass_status": "n/a",
                }
            )
            notes.append(f"missing summary row for {family}")
            continue
        structural_all_true = all(bool_cell(row.get(column)) for column in STRUCTURAL_FLAG_COLUMNS)
        results.append(
            {
                "family": family,
                "dispatch": "completed",
                "structural_flags_all_true": str(structural_all_true),
                "runs_first_pass_status": str(row.get("extreme_band_first_pass_status", "unreported") or "unreported"),
            }
        )
    return results, notes


def build_downstream_dispatch_summary(
    family_results: Sequence[Mapping[str, str]],
    notes: Sequence[str],
    completed_returncode: int,
) -> dict[str, Any]:
    fail_count = sum(
        1
        for row in family_results
        if row.get("dispatch") != "completed" or row.get("structural_flags_all_true") != "True"
    )
    families_reported = len(family_results)
    families_expected = len(L4_SMOKE_CONFIG.families)
    result = "pass" if completed_returncode == 0 and fail_count == 0 and not notes and families_reported == families_expected else "fail"
    return {
        "subprocess_returncode": int(completed_returncode),
        "families_expected": families_expected,
        "families_reported": families_reported,
        "fail": fail_count,
        "result": result,
    }


def render_l4_smoke_results(family_results: Sequence[Mapping[str, str]], notes: Sequence[str], completed_returncode: int) -> str:
    downstream_summary = build_downstream_dispatch_summary(family_results, notes, completed_returncode)
    lines = [
        "execution dispatch/result summary:",
        "per-family dispatch/result summary:",
    ]
    for row in family_results:
        lines.append(
            "  - "
            f"{row.get('family')}: dispatch={row.get('dispatch')}; "
            f"structural_flags_all_true={row.get('structural_flags_all_true')}; "
            f"runs_first_pass_status={row.get('runs_first_pass_status')}"
        )
    if not family_results:
        lines.append("  - none")
    if notes:
        lines.append("notes:")
        lines.extend(f"  - {note}" for note in notes)
    lines.extend(
        [
            "final pass/fail summary:",
            f"  subprocess_returncode: {downstream_summary['subprocess_returncode']}",
            f"  families_expected: {downstream_summary['families_expected']}",
            f"  families_reported: {downstream_summary['families_reported']}",
            f"  fail: {downstream_summary['fail']}",
            f"  result: {downstream_summary['result']}",
        ]
    )
    return "\n".join(lines)


def render_l4_smoke_precondition_failure(repo_root: Path, out_dir: Path | None, errors: Sequence[str]) -> str:
    return "\n".join(
        [
            render_l4_smoke_header(repo_root, out_dir, "execute"),
            "precondition errors:",
            *(f"  - {error}" for error in errors),
            "final pass/fail summary:",
            "  result: fail",
        ]
    )


def render_l4_smoke_preflight_blocked(repo_root: Path, out_dir: Path, preflight: L4SmokePreflight) -> str:
    return "\n".join(
        [
            render_l4_smoke_header(repo_root, out_dir, "execute"),
            render_l4_smoke_preflight(preflight),
            "execution dispatch/result summary:",
            "  downstream subprocess: not invoked",
            "final pass/fail summary:",
            "  result: fail",
        ]
    )


def run_l4_smoke_preflight_only(
    repo_root: Path,
    out_dir: Path | None,
    preflight_provider: Callable[[Path], L4SmokePreflight] = collect_l4_smoke_preflight,
) -> int:
    preflight = preflight_provider(repo_root)
    if out_dir is not None:
        write_preflight_artifact(out_dir, preflight, "preflight-only")
    print(
        "\n".join(
            [
                render_l4_smoke_header(repo_root, out_dir, "preflight-only"),
                render_l4_smoke_preflight(preflight),
            ]
        )
    )
    return 0 if preflight.preflight_ok else 1


def run_l4_smoke_execute(
    repo_root: Path,
    out_dir: Path | None,
    run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    preflight_provider: Callable[[Path], L4SmokePreflight] = collect_l4_smoke_preflight,
) -> int:
    errors = validate_l4_smoke_execute_preconditions(repo_root, out_dir)
    if errors:
        print(render_l4_smoke_precondition_failure(repo_root, out_dir, errors))
        return 2

    assert out_dir is not None
    out_dir.mkdir(parents=True, exist_ok=True)
    preflight = preflight_provider(repo_root)
    write_preflight_artifact(out_dir, preflight, "execute")
    if not preflight.preflight_ok:
        print(render_l4_smoke_preflight_blocked(repo_root, out_dir, preflight))
        return 1

    command = build_l4_smoke_command(repo_root, out_dir)
    print(render_l4_smoke_header(repo_root, out_dir, "execute"))
    print(render_l4_smoke_preflight(preflight))
    completed = run_command(
        command,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )

    family_results, notes = parse_l4_smoke_family_results(repo_root, out_dir) if completed.returncode == 0 else ([], [])
    if completed.returncode != 0:
        if completed.stdout:
            notes.append("subprocess stdout: " + completed.stdout.strip()[-1000:])
        if completed.stderr:
            notes.append("subprocess stderr: " + completed.stderr.strip()[-1000:])
    print(render_l4_smoke_results(family_results, notes, int(completed.returncode)))

    created_at = utc_created_at()
    downstream_summary = build_downstream_dispatch_summary(family_results, notes, int(completed.returncode))
    status_payload = {
        "schema_id": L4_SMOKE_STATUS_SCHEMA_ID,
        "schema_version": ARTIFACT_CONTRACT_VERSION,
        "created_at": created_at,
        "tier": Tier.L4_SMOKE.value,
        "mode": "execute",
        "fixed_target_set": l4_smoke_fixed_target_set(),
        "model_id": L4_SMOKE_CONFIG.model_id,
        "model_label": L4_SMOKE_CONFIG.model_label,
        "families": list(L4_SMOKE_CONFIG.families),
        "entrypoint": repo_relative(repo_root, repo_root / "tools" / "run_gate12a_cross_model_replay.py"),
        "command": command,
        "out_dir": str(out_dir),
        "returncode": int(completed.returncode),
        "preflight": build_preflight_artifact_payload(preflight, "execute", created_at=created_at),
        "downstream_dispatch_summary": downstream_summary,
        "result": downstream_summary["result"],
        "family_results": list(family_results),
        "notes": list(notes),
    }
    write_status_artifact(status_artifact_path(out_dir), status_payload)
    return 0 if downstream_summary["result"] == "pass" and family_results else 1


def render_summarize_existing(repo_root: Path) -> str:
    plan = plan_summarize_existing()
    workstream_root = repo_root / "workstream"
    present_memos = [memo for memo in EXPECTED_ATLAS_MEMOS if (workstream_root / memo).exists()]
    missing_memos = [memo for memo in EXPECTED_ATLAS_MEMOS if memo not in present_memos]
    summary_dirs = discover_summary_dirs(repo_root)
    summaries = [parse_cross_model_summary(repo_root, path) for path in summary_dirs]
    preflight_artifacts, status_artifacts = discover_and_validate_eval_factory_artifacts(repo_root)
    discovered_summary_names = {summary.run_id for summary in summaries}
    missing_expected_summaries = [name for name in EXPECTED_SUMMARY_RUNS if name not in discovered_summary_names]
    docs = (
        "docs/gate12a_evidence_atlas.md",
        "docs/reproduce_gate12a.md",
        "workstream/README.md",
        "zenodo-release/CHECKSUMS-SHA256.txt",
    )

    lines = [
        f"tier: {plan.tier.value}",
        f"intent: {plan.intent}",
        f"expected resource posture: {plan.resource_posture}",
        "memo-facing surfaces:",
        f"  tracked atlas memos present: {len(present_memos)}/{len(EXPECTED_ATLAS_MEMOS)}",
    ]
    for doc in docs:
        status = "present" if (repo_root / doc).exists() else "missing"
        lines.append(f"  {doc}: {status}")
    if missing_memos:
        lines.append("  missing tracked memos: " + ", ".join(missing_memos))
    else:
        lines.append("  missing tracked memos: none")

    lines.append("tracked memo model surfaces:")
    for surface in TRACKED_MEMO_SURFACES:
        memo_path = workstream_root / surface.memo_file
        memo_status = "present" if memo_path.exists() else "missing"
        matching_run_summary = next((summary.run_id for summary in summaries if summary.model_label == surface.model_label), "none")
        lines.append(
            "  - "
            f"model={surface.model_id}; memo={surface.memo_id}; memo_status={memo_status}; "
            f"tracked_scope={surface.tracked_scope}; matching_runs_summary={matching_run_summary}"
        )

    lines.extend(
        [
            "runs-derived materialized cross-model summaries:",
            f"  discovered: {len(summaries)}",
        ]
    )
    if summaries:
        for summary in summaries:
            families = ", ".join(summary.families) if summary.families else "none"
            notes = "; ".join(summary.notes) if summary.notes else "none"
            lines.append(
                "  - "
                f"{summary.run_id}: runs_status={summary.status}; model={summary.model_id}; "
                f"families={families}; rows={summary.row_count}; "
                f"runs_structural_flags_all_true={summary.structural_flags_all_true}; "
                f"runs_first_pass_status={summary.first_pass_statuses}; notes={notes}"
            )
    else:
        lines.append("  - none")

    lines.extend(
        [
            "eval-factory preflight artifact surfaces:",
            f"  discovered: {len(preflight_artifacts)}",
        ]
    )
    if preflight_artifacts:
        for artifact in preflight_artifacts:
            if artifact.status == ARTIFACT_STATUS_VALID:
                lines.append(
                    "  - "
                    f"source_class={artifact.source_class}; path={artifact.path}; artifact_status={artifact.status}; "
                    f"schema={artifact.schema_id}; mode={artifact.mode}; result={artifact.result}; "
                    f"posture={artifact.posture_classification}"
                )
            else:
                lines.append(
                    "  - "
                    f"source_class={artifact.source_class}; path={artifact.path}; artifact_status={artifact.status}; "
                    "errors=" + " | ".join(artifact.errors)
                )
    else:
        lines.append("  - none")

    lines.extend(
        [
            "eval-factory execute/status artifact surfaces:",
            f"  discovered: {len(status_artifacts)}",
        ]
    )
    if status_artifacts:
        for artifact in status_artifacts:
            if artifact.status == ARTIFACT_STATUS_VALID:
                lines.append(
                    "  - "
                    f"source_class={artifact.source_class}; path={artifact.path}; artifact_status={artifact.status}; "
                    f"schema={artifact.schema_id}; mode={artifact.mode}; result={artifact.result}; "
                    f"posture={artifact.posture_classification}; downstream_result={artifact.downstream_result}"
                )
            else:
                lines.append(
                    "  - "
                    f"source_class={artifact.source_class}; path={artifact.path}; artifact_status={artifact.status}; "
                    "errors=" + " | ".join(artifact.errors)
                )
    else:
        lines.append("  - none")

    lines.extend(
        [
            "artifact/path notes:",
            f"  runs/ present: {'yes' if (repo_root / 'runs').exists() else 'no'}",
            f"  shallow gate12a run dirs: {count_shallow_gate12a_run_dirs(repo_root)}",
        ]
    )
    if missing_expected_summaries:
        lines.append("  missing expected summary dirs: " + ", ".join(missing_expected_summaries))
    else:
        lines.append("  missing expected summary dirs: none")
    lines.append("result: read-only summary complete")
    return "\n".join(lines)


def run_cpu_nightly(repo_root: Path) -> int:
    checks = build_cpu_nightly_checks(repo_root)
    print(render_check_report(plan_cpu_nightly(), checks))
    return 1 if any(check.level == LEVEL_FAIL for check in checks) else 0


def run_summarize_existing(repo_root: Path) -> int:
    print(render_summarize_existing(repo_root))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    tier = Tier(args.tier)
    if args.execute and tier != Tier.L4_SMOKE:
        print(f"error: --execute is only supported for --tier {Tier.L4_SMOKE.value}")
        return 2
    if args.preflight_only and tier != Tier.L4_SMOKE:
        print(f"error: --preflight-only is only supported for --tier {Tier.L4_SMOKE.value}")
        return 2
    if args.execute and args.preflight_only:
        print("error: --execute and --preflight-only cannot be used together")
        return 2
    if tier == Tier.CPU_NIGHTLY:
        return run_cpu_nightly(REPO_ROOT)
    if tier == Tier.SUMMARIZE_EXISTING:
        return run_summarize_existing(REPO_ROOT)
    if tier == Tier.L4_SMOKE:
        if args.preflight_only:
            return run_l4_smoke_preflight_only(REPO_ROOT, Path(args.out_dir) if args.out_dir else None)
        if args.execute:
            return run_l4_smoke_execute(REPO_ROOT, Path(args.out_dir) if args.out_dir else None)
        print(render_l4_smoke_dry_run(REPO_ROOT))
        return 0
    if tier == Tier.L4_WEEKLY:
        return run_l4_weekly(REPO_ROOT, Path(args.out_dir) if args.out_dir else None)

    plan = dispatch(tier)
    print(render_plan(plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
