#!/usr/bin/env python3
"""Inspect and quarantine local eval-factory artifact files.

This helper is intentionally operational only. It does not execute models,
rewrite analytical outputs, or promote any artifact into evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import run_eval_checks as eval_checks


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "runs"
DEFAULT_QUARANTINE_DIRNAME = "_quarantine"

CLASS_VALID_CURRENT = "valid_current_contract"
CLASS_MALFORMED_CURRENT = "malformed_current_contract"
CLASS_LEGACY_UNKNOWN = "legacy_unknown_schema"
CLASS_NON_EVAL_FACTORY = "non_eval_factory_file"
QUARANTINE_CLASSES = (CLASS_MALFORMED_CURRENT, CLASS_LEGACY_UNKNOWN)

ARTIFACT_KIND_BY_FILENAME = {
    eval_checks.L4_SMOKE_PREFLIGHT_FILENAME: "l4_smoke_preflight",
    eval_checks.L4_SMOKE_STATUS_FILENAME: "l4_smoke_status",
    eval_checks.L4_WEEKLY_PLAN_FILENAME: "l4_weekly_plan",
    eval_checks.L4_WEEKLY_PREFLIGHT_FILENAME: "l4_weekly_preflight",
    eval_checks.L4_WEEKLY_STATUS_FILENAME: "l4_weekly_status",
}
EXPECTED_SCHEMA_BY_KIND = {
    "l4_smoke_preflight": eval_checks.L4_SMOKE_PREFLIGHT_SCHEMA_ID,
    "l4_smoke_status": eval_checks.L4_SMOKE_STATUS_SCHEMA_ID,
    "l4_weekly_plan": eval_checks.L4_WEEKLY_PLAN_SCHEMA_ID,
    "l4_weekly_preflight": eval_checks.L4_WEEKLY_PREFLIGHT_SCHEMA_ID,
    "l4_weekly_status": eval_checks.L4_WEEKLY_STATUS_SCHEMA_ID,
}


@dataclass(frozen=True)
class ManagedArtifact:
    path: Path
    relative_path: str
    artifact_kind: str
    classification: str
    schema_id: str
    errors: tuple[str, ...]
    quarantine_target: Path | None = None


@dataclass(frozen=True)
class QuarantineMove:
    source: Path
    target: Path
    classification: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def repo_relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect local eval-factory artifact files and optionally quarantine "
            "legacy or malformed artifacts. Default mode is dry-run."
        )
    )
    parser.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help="Root to scan. Defaults to the repository runs/ directory.",
    )
    parser.add_argument(
        "--quarantine",
        action="store_true",
        help="Move malformed/legacy eval-factory artifacts into a quarantine directory under --root.",
    )
    parser.add_argument(
        "--quarantine-dir",
        default=DEFAULT_QUARANTINE_DIRNAME,
        help="Quarantine directory name under --root. Defaults to _quarantine.",
    )
    parser.add_argument(
        "--write-sidecar-manifest",
        action="store_true",
        help="When used with --quarantine, write a small manifest describing moved artifacts.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def iter_files(root: Path, quarantine_dirname: str = DEFAULT_QUARANTINE_DIRNAME) -> tuple[Path, ...]:
    if not root.exists():
        return ()
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            relative_parts = path.relative_to(root).parts
        except ValueError:
            relative_parts = path.parts
        if quarantine_dirname in relative_parts:
            continue
        files.append(path)
    return tuple(sorted(files, key=lambda item: repo_relative(root, item)))


def discover_eval_factory_artifacts(root: Path, quarantine_dirname: str = DEFAULT_QUARANTINE_DIRNAME) -> tuple[Path, ...]:
    if not root.exists():
        return ()
    names = set(ARTIFACT_KIND_BY_FILENAME)
    matches: list[Path] = []
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name != quarantine_dirname]
        matched_names = names.intersection(filenames)
        for filename in matched_names:
            matches.append(Path(current_root) / filename)
    return tuple(sorted(matches, key=lambda path: repo_relative(root, path)))


def validation_errors_for_kind(root: Path, path: Path, artifact_kind: str) -> tuple[str, ...]:
    if artifact_kind == "l4_smoke_preflight":
        return eval_checks.validate_eval_factory_preflight_artifact(root, path).errors
    if artifact_kind == "l4_smoke_status":
        return eval_checks.validate_eval_factory_status_artifact(root, path).errors
    if artifact_kind == "l4_weekly_plan":
        return eval_checks.validate_l4_weekly_plan_artifact(root, path).errors
    if artifact_kind == "l4_weekly_preflight":
        return eval_checks.validate_eval_factory_weekly_preflight_artifact(root, path).errors
    if artifact_kind == "l4_weekly_status":
        return eval_checks.validate_eval_factory_weekly_status_artifact(root, path).errors
    return (f"unsupported eval-factory artifact kind: {artifact_kind}",)


def classify_artifact(root: Path, path: Path, quarantine_dirname: str = DEFAULT_QUARANTINE_DIRNAME) -> ManagedArtifact:
    relative_path = repo_relative(root, path)
    artifact_kind = ARTIFACT_KIND_BY_FILENAME.get(path.name, "non_eval_factory")
    if artifact_kind == "non_eval_factory":
        return ManagedArtifact(
            path=path,
            relative_path=relative_path,
            artifact_kind=artifact_kind,
            classification=CLASS_NON_EVAL_FACTORY,
            schema_id="",
            errors=(),
        )

    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return ManagedArtifact(
            path=path,
            relative_path=relative_path,
            artifact_kind=artifact_kind,
            classification=CLASS_LEGACY_UNKNOWN,
            schema_id="",
            errors=(f"artifact unreadable: {exc}",),
            quarantine_target=quarantine_target_for(root, path, quarantine_dirname),
        )

    schema_id = str(payload.get("schema_id", "")) if isinstance(payload, dict) else ""
    expected_schema = EXPECTED_SCHEMA_BY_KIND[artifact_kind]
    if schema_id != expected_schema:
        detail = "missing schema_id" if not schema_id else f"unexpected schema_id: {schema_id}"
        return ManagedArtifact(
            path=path,
            relative_path=relative_path,
            artifact_kind=artifact_kind,
            classification=CLASS_LEGACY_UNKNOWN,
            schema_id=schema_id,
            errors=(detail,),
            quarantine_target=quarantine_target_for(root, path, quarantine_dirname),
        )

    errors = validation_errors_for_kind(root, path, artifact_kind)
    classification = CLASS_VALID_CURRENT if not errors else CLASS_MALFORMED_CURRENT
    quarantine_target = quarantine_target_for(root, path, quarantine_dirname) if errors else None
    return ManagedArtifact(
        path=path,
        relative_path=relative_path,
        artifact_kind=artifact_kind,
        classification=classification,
        schema_id=schema_id,
        errors=errors,
        quarantine_target=quarantine_target,
    )


def inspect_artifacts(root: Path, quarantine_dirname: str = DEFAULT_QUARANTINE_DIRNAME) -> tuple[ManagedArtifact, ...]:
    return tuple(
        classify_artifact(root, path, quarantine_dirname)
        for path in discover_eval_factory_artifacts(root, quarantine_dirname)
    )


def quarantine_target_for(root: Path, source: Path, quarantine_dirname: str) -> Path:
    try:
        relative = source.resolve().relative_to(root.resolve())
    except ValueError:
        relative = Path(source.name)
    return root / quarantine_dirname / relative


def unique_target(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 1000):
        candidate = path.with_name(f"{path.stem}.{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not allocate quarantine target for {path}")


def quarantine_artifacts(artifacts: Sequence[ManagedArtifact]) -> tuple[QuarantineMove, ...]:
    moves: list[QuarantineMove] = []
    for artifact in artifacts:
        if artifact.classification not in QUARANTINE_CLASSES or artifact.quarantine_target is None:
            continue
        target = unique_target(artifact.quarantine_target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(artifact.path), str(target))
        moves.append(QuarantineMove(artifact.path, target, artifact.classification))
    return tuple(moves)


def write_sidecar_manifest(root: Path, moves: Sequence[QuarantineMove], quarantine_dirname: str) -> Path:
    manifest_path = unique_target(root / quarantine_dirname / "eval_factory_quarantine_manifest.json")
    payload = {
        "created_at": utc_now(),
        "root": str(root),
        "moved": [
            {
                "source": repo_relative(root, move.source),
                "target": repo_relative(root, move.target),
                "classification": move.classification,
            }
            for move in moves
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return manifest_path


def render_report(
    root: Path,
    artifacts: Sequence[ManagedArtifact],
    quarantine: bool,
    moves: Sequence[QuarantineMove] = (),
    manifest_path: Path | None = None,
) -> str:
    counts = Counter(artifact.classification for artifact in artifacts)
    quarantine_candidates = [artifact for artifact in artifacts if artifact.classification in QUARANTINE_CLASSES]
    lines = [
        f"root scanned: {root}",
        "discovered eval-factory artifacts:",
        f"  count: {len(artifacts)}",
    ]
    if artifacts:
        for artifact in artifacts:
            error_text = "none" if not artifact.errors else " | ".join(artifact.errors)
            lines.append(
                "  - "
                f"path={artifact.relative_path}; kind={artifact.artifact_kind}; "
                f"classification={artifact.classification}; schema_id={artifact.schema_id or 'none'}; "
                f"errors={error_text}"
            )
    else:
        lines.append("  - none")

    lines.extend(
        [
            "classification summary:",
            f"  {CLASS_VALID_CURRENT}: {counts[CLASS_VALID_CURRENT]}",
            f"  {CLASS_MALFORMED_CURRENT}: {counts[CLASS_MALFORMED_CURRENT]}",
            f"  {CLASS_LEGACY_UNKNOWN}: {counts[CLASS_LEGACY_UNKNOWN]}",
            f"  {CLASS_NON_EVAL_FACTORY}: {counts[CLASS_NON_EVAL_FACTORY]}",
            "proposed quarantine actions:",
        ]
    )
    if quarantine_candidates:
        for artifact in quarantine_candidates:
            target = artifact.quarantine_target or Path("<unavailable>")
            lines.append(f"  - move {artifact.relative_path} -> {repo_relative(root, target)}")
    else:
        lines.append("  - none")

    lines.append("quarantine actions:")
    if quarantine:
        if moves:
            for move in moves:
                lines.append(f"  - moved {repo_relative(root, move.source)} -> {repo_relative(root, move.target)}")
        else:
            lines.append("  - none")
    else:
        lines.append("  - dry-run; no files moved")

    lines.append("sidecar manifest:")
    if manifest_path is None:
        lines.append("  - none")
    else:
        lines.append(f"  - {repo_relative(root, manifest_path)}")

    lines.extend(
        [
            "final result:",
            f"  result: {'quarantined' if quarantine else 'dry-run'}",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root)
    artifacts = inspect_artifacts(root, args.quarantine_dir)
    moves: tuple[QuarantineMove, ...] = ()
    manifest_path: Path | None = None
    if args.quarantine:
        moves = quarantine_artifacts(artifacts)
        if args.write_sidecar_manifest:
            manifest_path = write_sidecar_manifest(root, moves, args.quarantine_dir)
    print(render_report(root, artifacts, quarantine=args.quarantine, moves=moves, manifest_path=manifest_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
