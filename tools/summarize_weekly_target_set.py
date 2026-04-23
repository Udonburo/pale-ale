"""Summarize the current bounded weekly target set from weekly receipt bundles.

This helper is strictly read-only. It reports operator/status surfaces from
canonical valid l4-weekly receipt bundles and does not create new claim,
checkpoint, or memo surfaces.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import run_eval_checks as runner


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class WeeklyReceiptCandidate:
    target: str
    schema: str
    result: str
    posture: str
    family_count: int | None
    manifest_path: str
    bundle_path: str
    checksums_present: bool
    tarball_present: bool
    created_at: str


@dataclass(frozen=True)
class WeeklyTargetSetEntry:
    target: str
    schema: str
    result: str
    posture: str
    family_count: int | None
    bundle_path: str
    checksums_present: bool
    tarball_present: bool
    duplicates_present: bool
    duplicate_bundle_count: int


@dataclass(frozen=True)
class WeeklyTargetSetSummary:
    expected_targets: tuple[str, ...]
    found_targets: tuple[str, ...]
    missing_targets: tuple[str, ...]
    invalid_weekly_receipt_bundle_count: int
    canonical_entries: tuple[WeeklyTargetSetEntry, ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the current bounded weekly target set from canonical "
            "valid l4-weekly receipt bundles."
        )
    )
    parser.add_argument(
        "--repo-root",
        default=str(REPO_ROOT),
        help="Repository root to scan. Defaults to the current pale-ale repo root.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format. Defaults to human-readable text.",
    )
    return parser


def expected_weekly_targets() -> tuple[str, ...]:
    return tuple(target.target_key for target in runner.L4_WEEKLY_TARGETS)


def manifest_path_to_bundle_path(manifest_path: str) -> str:
    return str(Path(manifest_path).parent).replace("\\", "/")


def build_weekly_receipt_candidates(repo_root: Path) -> tuple[WeeklyReceiptCandidate, ...]:
    candidates: list[WeeklyReceiptCandidate] = []
    for validation in runner.discover_and_validate_operator_receipts(repo_root):
        if validation.source_class != runner.SOURCE_OPERATOR_WEEKLY_RECEIPT:
            continue
        if validation.status != runner.ARTIFACT_STATUS_VALID:
            continue
        manifest_path = repo_root / Path(validation.path)
        payload = runner.read_json(manifest_path)
        created_at = str(payload.get("created_at", "")) if isinstance(payload, Mapping) else ""
        candidates.append(
            WeeklyReceiptCandidate(
                target=validation.target,
                schema=validation.schema_id,
                result=validation.result,
                posture=validation.posture_classification,
                family_count=validation.family_count,
                manifest_path=validation.path,
                bundle_path=manifest_path_to_bundle_path(validation.path),
                checksums_present=validation.checksum_present,
                tarball_present=validation.tarball_present,
                created_at=created_at,
            )
        )
    return tuple(candidates)


def select_canonical_weekly_entries(
    candidates: Sequence[WeeklyReceiptCandidate],
    expected_targets_override: Sequence[str] | None = None,
) -> WeeklyTargetSetSummary:
    expected = tuple(expected_targets_override or expected_weekly_targets())

    grouped: dict[str, list[WeeklyReceiptCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.target].append(candidate)

    entries: list[WeeklyTargetSetEntry] = []
    found_targets: list[str] = []
    for target in expected:
        target_candidates = grouped.get(target, [])
        if not target_candidates:
            continue
        selected = max(target_candidates, key=lambda candidate: (candidate.created_at, candidate.bundle_path))
        found_targets.append(target)
        entries.append(
            WeeklyTargetSetEntry(
                target=selected.target,
                schema=selected.schema,
                result=selected.result,
                posture=selected.posture,
                family_count=selected.family_count,
                bundle_path=selected.bundle_path,
                checksums_present=selected.checksums_present,
                tarball_present=selected.tarball_present,
                duplicates_present=len(target_candidates) > 1,
                duplicate_bundle_count=max(len(target_candidates) - 1, 0),
            )
        )

    missing_targets = tuple(target for target in expected if target not in found_targets)
    return WeeklyTargetSetSummary(
        expected_targets=expected,
        found_targets=tuple(found_targets),
        missing_targets=missing_targets,
        invalid_weekly_receipt_bundle_count=0,
        canonical_entries=tuple(entries),
    )


def summarize_weekly_target_set(repo_root: Path) -> WeeklyTargetSetSummary:
    validations = runner.discover_and_validate_operator_receipts(repo_root)
    invalid_weekly_count = sum(
        1
        for validation in validations
        if validation.source_class == runner.SOURCE_OPERATOR_WEEKLY_RECEIPT
        and validation.status != runner.ARTIFACT_STATUS_VALID
    )
    candidates = build_weekly_receipt_candidates(repo_root)
    summary = select_canonical_weekly_entries(candidates, expected_weekly_targets())
    return WeeklyTargetSetSummary(
        expected_targets=summary.expected_targets,
        found_targets=summary.found_targets,
        missing_targets=summary.missing_targets,
        invalid_weekly_receipt_bundle_count=invalid_weekly_count,
        canonical_entries=summary.canonical_entries,
    )


def summary_to_payload(summary: WeeklyTargetSetSummary) -> dict[str, Any]:
    return {
        "expected_targets": list(summary.expected_targets),
        "found_targets": list(summary.found_targets),
        "missing_targets": list(summary.missing_targets),
        "invalid_weekly_receipt_bundle_count": summary.invalid_weekly_receipt_bundle_count,
        "canonical_bundles": [asdict(entry) for entry in summary.canonical_entries],
    }


def render_text_summary(summary: WeeklyTargetSetSummary) -> str:
    lines = [
        "weekly target-set summary:",
        "  intent: operator/status summary from canonical valid l4-weekly receipt bundles",
        f"  expected_targets: {', '.join(summary.expected_targets)}",
        f"  found_targets: {len(summary.found_targets)}/{len(summary.expected_targets)}",
        "  missing_targets: none" if not summary.missing_targets else f"  missing_targets: {', '.join(summary.missing_targets)}",
        f"  invalid_weekly_receipt_bundle_count: {summary.invalid_weekly_receipt_bundle_count}",
        "canonical bundles:",
    ]
    if not summary.canonical_entries:
        lines.append("  - none")
        return "\n".join(lines)
    for entry in summary.canonical_entries:
        lines.append(
            "  - "
            + "; ".join(
                [
                    f"target={entry.target}",
                    f"schema={entry.schema}",
                    f"result={entry.result}",
                    f"posture={entry.posture}",
                    f"family_count={entry.family_count if entry.family_count is not None else 'n/a'}",
                    f"checksums={'present' if entry.checksums_present else 'absent'}",
                    f"tarball={'present' if entry.tarball_present else 'absent'}",
                    f"duplicates={'yes' if entry.duplicates_present else 'no'}",
                    f"duplicate_bundle_count={entry.duplicate_bundle_count}",
                    f"bundle_path={entry.bundle_path}",
                ]
            )
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    summary = summarize_weekly_target_set(repo_root)
    if args.format == "json":
        print(json.dumps(summary_to_payload(summary), indent=2))
    else:
        print(render_text_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
