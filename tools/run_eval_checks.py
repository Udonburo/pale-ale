#!/usr/bin/env python3
"""Plan or run read-only evaluation-factory checks by standing tier.

This entrypoint does not introduce analytical methods, change Gate12A doctrine,
or invoke model/GPU execution.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


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


TIER_VALUES = tuple(tier.value for tier in Tier)
LEVEL_PASS = "PASS"
LEVEL_WARN = "WARN"
LEVEL_FAIL = "FAIL"

FAMILY_SET = ("transcript_v1", "briefing_v1", "archive_v1")

L4_SMOKE_BOUNDARY = "0.5B fixed family boundary set: transcript_v1, briefing_v1, archive_v1"

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
            "This entrypoint never invokes GPU/model execution."
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
            "Compatibility flag. L4 tiers are plan-only; read-only tiers inspect existing local files "
            "without running jobs."
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
            f"plan checks for {L4_SMOKE_BOUNDARY}",
            "confirm expected fixed-family inputs before any future dispatch",
            "reserve only a narrow smoke-lane path for later cheap model execution wiring",
        ),
        out_of_scope=(
            "full dense-transformer family-set expansion",
            "7B FP32",
            "new analytical methods",
        ),
        not_implemented_yet=(
            "0.5B smoke dispatch",
            "artifact completeness checks for the smoke lane",
            "result summarization handoff",
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
            "keep weekly planning to repeatable dense-transformer family-set surfaces only",
        ),
        out_of_scope=(
            L4_WEEKLY_EXCLUDES_7B_FP32,
            *L4_WEEKLY_EXCLUDED_CANDIDATES,
            "Gate12B promotion",
        ),
        not_implemented_yet=(
            "3B/4B weekly dispatch",
            "L4 runtime budget enforcement",
            "standing summary publication",
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
    required_exclusions = (L4_WEEKLY_EXCLUDES_7B_FP32, *L4_WEEKLY_EXCLUDED_CANDIDATES)
    missing_exclusions = [item for item in required_exclusions if item not in weekly_plan.out_of_scope]
    if missing_exclusions:
        checks.append(CheckResult(LEVEL_FAIL, "l4-weekly exclusions", "missing " + ", ".join(missing_exclusions)))
    else:
        checks.append(CheckResult(LEVEL_PASS, "l4-weekly exclusions", ", ".join(required_exclusions)))

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


def render_summarize_existing(repo_root: Path) -> str:
    plan = plan_summarize_existing()
    workstream_root = repo_root / "workstream"
    present_memos = [memo for memo in EXPECTED_ATLAS_MEMOS if (workstream_root / memo).exists()]
    missing_memos = [memo for memo in EXPECTED_ATLAS_MEMOS if memo not in present_memos]
    summary_dirs = discover_summary_dirs(repo_root)
    summaries = [parse_cross_model_summary(repo_root, path) for path in summary_dirs]
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

    lines.extend(
        [
            "materialized cross-model summaries:",
            f"  discovered: {len(summaries)}",
        ]
    )
    if summaries:
        for summary in summaries:
            families = ", ".join(summary.families) if summary.families else "none"
            notes = "; ".join(summary.notes) if summary.notes else "none"
            lines.append(
                "  - "
                f"{summary.run_id}: status={summary.status}; model={summary.model_id}; "
                f"families={families}; rows={summary.row_count}; "
                f"structural_flags_all_true={summary.structural_flags_all_true}; "
                f"first_pass={summary.first_pass_statuses}; notes={notes}"
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
    if tier == Tier.CPU_NIGHTLY:
        return run_cpu_nightly(REPO_ROOT)
    if tier == Tier.SUMMARIZE_EXISTING:
        return run_summarize_existing(REPO_ROOT)

    plan = dispatch(tier)
    print(render_plan(plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
