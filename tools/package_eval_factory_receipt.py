#!/usr/bin/env python3
"""Package a successful eval-factory run as an operator receipt.

This helper is operational only. It validates existing eval-factory artifacts,
copies a compact receipt surface, and optionally packs the full run directory.
It does not run models, change Gate12A math, or create checkpoint claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import run_eval_checks as runner


@dataclass(frozen=True)
class ReceiptArtifactSpec:
    role: str
    source_relative_path: str
    bundled_relative_path: str


@dataclass(frozen=True)
class ReceiptRunProfile:
    tier: str
    source_class: str
    manifest_schema_id: str
    target: str
    fixed_target_set: Mapping[str, Any]
    preflight_filename: str
    status_filename: str
    execute_log_filename: str
    summary_run_id: str
    required_artifacts: tuple[ReceiptArtifactSpec, ...]
    preflight_validator: Any
    status_validator: Any


@dataclass(frozen=True)
class ReceiptBundleResult:
    run_dir: Path
    receipt_root: Path
    manifest_path: Path
    required_checksums_path: Path
    bundle_checksums_path: Path
    tarball_path: Path | None
    tarball_checksum_path: Path | None
    inspect_only: bool
    tier: str
    target: str
    source_class: str
    family_count: int
    result: str


class ReceiptPackagingError(RuntimeError):
    """Raised when a run directory cannot be packaged as a receipt."""


WEEKLY_EXECUTE_LOG_FILENAME = "eval_factory_l4_weekly_execute_cli.log"


def build_required_artifacts(preflight_filename: str, status_filename: str, execute_log_filename: str, summary_run_id: str) -> tuple[ReceiptArtifactSpec, ...]:
    return (
        ReceiptArtifactSpec(
            "preflight",
            preflight_filename,
            f"required_artifacts/{preflight_filename}",
        ),
        ReceiptArtifactSpec(
            "status",
            status_filename,
            f"required_artifacts/{status_filename}",
        ),
        ReceiptArtifactSpec(
            "execute_log",
            execute_log_filename,
            f"required_artifacts/{execute_log_filename}",
        ),
        ReceiptArtifactSpec(
            "cross_model_family_summary",
            f"{summary_run_id}/{runner.CROSS_MODEL_SUMMARY_FILENAME}",
            f"required_artifacts/{runner.CROSS_MODEL_SUMMARY_FILENAME}",
        ),
    )


SMOKE_REQUIRED_RECEIPT_ARTIFACTS = build_required_artifacts(
    runner.L4_SMOKE_PREFLIGHT_FILENAME,
    runner.L4_SMOKE_STATUS_FILENAME,
    "eval_factory_l4_smoke_execute.log",
    runner.L4_SMOKE_CONFIG.summary_run_id,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Package a successful eval-factory l4-smoke or l4-weekly run as a durable operator receipt."
    )
    parser.add_argument("--run-dir", required=True, help="Successful eval-factory run directory.")
    parser.add_argument(
        "--out-root",
        default=f"runs/{runner.RECEIPT_BUNDLES_DIRNAME}",
        help="Receipt bundle root. Defaults to runs/receipt_bundles.",
    )
    parser.add_argument("--inspect-only", action="store_true", help="Validate and print the plan without writing files.")
    parser.add_argument("--no-tarball", action="store_true", help="Skip full run tar.gz creation.")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def utc_created_at() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def timestamp_slug(created_at: str) -> str:
    parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    return parsed.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def repo_relative(path: Path, repo_root: Path = runner.REPO_ROOT) -> str:
    return runner.repo_relative(repo_root, path)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_sha256sum(path: Path, entries: Sequence[tuple[str, Path]], repo_root: Path = runner.REPO_ROOT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{digest}  {repo_relative(target, repo_root)}" for digest, target in entries]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def detect_receipt_tier(run_dir: Path) -> str:
    smoke_present = (run_dir / runner.L4_SMOKE_PREFLIGHT_FILENAME).is_file() or (run_dir / runner.L4_SMOKE_STATUS_FILENAME).is_file()
    weekly_present = (run_dir / runner.L4_WEEKLY_PREFLIGHT_FILENAME).is_file() or (run_dir / runner.L4_WEEKLY_STATUS_FILENAME).is_file()
    if smoke_present and weekly_present:
        raise ReceiptPackagingError("run directory contains both l4-smoke and l4-weekly receipt artifacts")
    if weekly_present:
        return runner.Tier.L4_WEEKLY.value
    if smoke_present:
        return runner.Tier.L4_SMOKE.value
    raise ReceiptPackagingError("run directory does not contain known eval-factory receipt artifacts")


def build_smoke_profile() -> ReceiptRunProfile:
    return ReceiptRunProfile(
        tier=runner.Tier.L4_SMOKE.value,
        source_class=runner.SOURCE_OPERATOR_RECEIPT,
        manifest_schema_id=runner.OPERATOR_RECEIPT_SCHEMA_ID,
        target="n/a",
        fixed_target_set=runner.l4_smoke_fixed_target_set(),
        preflight_filename=runner.L4_SMOKE_PREFLIGHT_FILENAME,
        status_filename=runner.L4_SMOKE_STATUS_FILENAME,
        execute_log_filename="eval_factory_l4_smoke_execute.log",
        summary_run_id=runner.L4_SMOKE_CONFIG.summary_run_id,
        required_artifacts=SMOKE_REQUIRED_RECEIPT_ARTIFACTS,
        preflight_validator=runner.validate_preflight_artifact_payload,
        status_validator=runner.validate_status_artifact_payload,
    )


def build_weekly_profile(status: Mapping[str, Any]) -> ReceiptRunProfile:
    target_key = status.get("target")
    if not isinstance(target_key, str) or not target_key:
        raise ReceiptPackagingError("weekly status artifact is missing target")
    try:
        target = runner.l4_weekly_target_for_key(target_key)
    except ValueError as exc:
        raise ReceiptPackagingError(str(exc)) from exc
    return ReceiptRunProfile(
        tier=runner.Tier.L4_WEEKLY.value,
        source_class=runner.SOURCE_OPERATOR_WEEKLY_RECEIPT,
        manifest_schema_id=runner.OPERATOR_RECEIPT_L4_WEEKLY_SCHEMA_ID,
        target=target.target_key,
        fixed_target_set=runner.l4_weekly_fixed_target_set(target),
        preflight_filename=runner.L4_WEEKLY_PREFLIGHT_FILENAME,
        status_filename=runner.L4_WEEKLY_STATUS_FILENAME,
        execute_log_filename=WEEKLY_EXECUTE_LOG_FILENAME,
        summary_run_id=target.summary_run_id,
        required_artifacts=build_required_artifacts(
            runner.L4_WEEKLY_PREFLIGHT_FILENAME,
            runner.L4_WEEKLY_STATUS_FILENAME,
            WEEKLY_EXECUTE_LOG_FILENAME,
            target.summary_run_id,
        ),
        preflight_validator=runner.validate_l4_weekly_preflight_artifact_payload,
        status_validator=runner.validate_l4_weekly_status_artifact_payload,
    )


def required_artifact_paths(run_dir: Path, profile: ReceiptRunProfile) -> dict[str, Path]:
    return {spec.role: run_dir / spec.source_relative_path for spec in profile.required_artifacts}


def read_summary_rows(summary_path: Path) -> list[dict[str, str]]:
    try:
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except (OSError, csv.Error, UnicodeDecodeError) as exc:
        raise ReceiptPackagingError(f"summary unreadable: {summary_path}: {exc}") from exc


def validate_source_run(run_dir: Path) -> tuple[ReceiptRunProfile, Mapping[str, Any], Mapping[str, Any], list[dict[str, str]]]:
    if not run_dir.is_dir():
        raise ReceiptPackagingError(f"run directory missing: {run_dir}")

    tier = detect_receipt_tier(run_dir)
    if tier == runner.Tier.L4_SMOKE.value:
        profile = build_smoke_profile()
    else:
        status_path = run_dir / runner.L4_WEEKLY_STATUS_FILENAME
        if not status_path.is_file():
            raise ReceiptPackagingError(f"missing required receipt artifact(s): {status_path}")
        try:
            status_probe = runner.read_json(status_path)
        except (OSError, json.JSONDecodeError) as exc:
            raise ReceiptPackagingError(f"weekly status artifact unreadable: {exc}") from exc
        profile = build_weekly_profile(status_probe)

    paths = required_artifact_paths(run_dir, profile)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ReceiptPackagingError("missing required receipt artifact(s): " + ", ".join(missing))

    try:
        preflight = runner.read_json(paths["preflight"])
        status = runner.read_json(paths["status"])
    except (OSError, json.JSONDecodeError) as exc:
        raise ReceiptPackagingError(f"preflight/status artifact unreadable: {exc}") from exc

    preflight_errors = profile.preflight_validator(preflight)
    status_errors = profile.status_validator(status)
    if preflight_errors:
        raise ReceiptPackagingError("malformed preflight artifact: " + " | ".join(preflight_errors))
    if status_errors:
        raise ReceiptPackagingError("malformed status artifact: " + " | ".join(status_errors))
    if preflight.get("result") != "pass" or preflight.get("posture_classification") != runner.POSTURE_REMOTE_CUDA_READY:
        raise ReceiptPackagingError("preflight artifact is not a passing remote_cuda_ready posture")
    downstream = status.get("downstream_dispatch_summary")
    if status.get("result") != "pass" or not isinstance(downstream, dict) or downstream.get("result") != "pass":
        raise ReceiptPackagingError("status artifact does not describe a successful downstream dispatch")

    rows = read_summary_rows(paths["cross_model_family_summary"])
    families = [row.get("rendering_family", "") for row in rows]
    if families != list(runner.FAMILY_SET):
        raise ReceiptPackagingError(f"summary families expected {list(runner.FAMILY_SET)!r}, got {families!r}")
    return profile, preflight, status, rows


def family_summary(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "rendering_family": row.get("rendering_family", ""),
            "zero_overlap_clear": row.get("zero_overlap_clear", ""),
            "all_defined_triangles_anchor_rich": row.get("all_defined_triangles_anchor_rich", ""),
            "trusted_tree_gt_residual_chord": row.get("trusted_tree_gt_residual_chord", ""),
            "plain_gt_anchor_qualified": row.get("plain_gt_anchor_qualified", ""),
            "trusted_tree_median": row.get("trusted_tree_median", ""),
            "residual_chord_median": row.get("residual_chord_median", ""),
            "plain_median": row.get("plain_median", ""),
            "anchor_qualified_median": row.get("anchor_qualified_median", ""),
            "runs_first_pass_status": row.get("extreme_band_first_pass_status", ""),
        }
        for row in rows
    ]


def build_receipt_manifest(
    profile: ReceiptRunProfile,
    run_dir: Path,
    receipt_root: Path,
    preflight: Mapping[str, Any],
    status: Mapping[str, Any],
    rows: Sequence[Mapping[str, str]],
    required_entries: Sequence[dict[str, Any]],
    required_checksums_path: Path,
    bundle_checksums_path: Path,
    tarball_path: Path | None,
    tarball_checksum_path: Path | None,
    created_at: str,
    repo_root: Path = runner.REPO_ROOT,
) -> dict[str, Any]:
    tarball_digest = runner.sha256_file(tarball_path) if tarball_path is not None and tarball_path.exists() else ""
    tarball_size = tarball_path.stat().st_size if tarball_path is not None and tarball_path.exists() else None
    return {
        "schema_id": profile.manifest_schema_id,
        "schema_version": runner.ARTIFACT_CONTRACT_VERSION,
        "source_class": profile.source_class,
        "created_at": created_at,
        "source_run_path": repo_relative(run_dir, repo_root),
        "source_run_absolute_path": str(run_dir.resolve()),
        "bundle_path": repo_relative(receipt_root, repo_root),
        "tier": profile.tier,
        "mode": "execute",
        "target": profile.target,
        "fixed_target_set": profile.fixed_target_set,
        "posture_classification": preflight.get("posture_classification"),
        "preflight_result": preflight.get("result"),
        "execute_result": status.get("result"),
        "downstream_dispatch_summary": status.get("downstream_dispatch_summary"),
        "family_count": len(rows),
        "families": [row.get("rendering_family", "") for row in rows],
        "downstream_summary_path": repo_relative(required_artifact_paths(run_dir, profile)["cross_model_family_summary"], repo_root),
        "runs_first_pass_status_note": (
            "runs_first_pass_status is pending_local_read for this receipt; "
            "no phenotype interpretation is added here."
        ),
        "not_a_checkpoint": True,
        "not_a_memo_claim": True,
        "no_new_model_execution_in_packaging": True,
        "checksums": {
            "required_artifacts_sha256": repo_relative(required_checksums_path, repo_root),
            "bundle_files_sha256": repo_relative(bundle_checksums_path, repo_root),
        },
        "tarball": {
            "present": tarball_path is not None,
            "path": repo_relative(tarball_path, repo_root) if tarball_path is not None else "",
            "sha256_path": repo_relative(tarball_checksum_path, repo_root) if tarball_checksum_path is not None else "",
            "sha256": tarball_digest,
            "size_bytes": tarball_size,
        },
        "required_artifacts": list(required_entries),
        "machine_side_structural_family_summary": family_summary(rows),
    }


def package_receipt(
    run_dir: Path,
    out_root: Path,
    *,
    create_tarball: bool = True,
    inspect_only: bool = False,
    created_at: str | None = None,
    repo_root: Path = runner.REPO_ROOT,
) -> ReceiptBundleResult:
    run_dir = run_dir.resolve()
    out_root = out_root.resolve()
    created_at = created_at or utc_created_at()
    profile, preflight, status, rows = validate_source_run(run_dir)
    receipt_root = out_root / f"{run_dir.name}_receipt_{timestamp_slug(created_at)}"
    required_checksums_path = receipt_root / runner.RECEIPT_REQUIRED_ARTIFACT_CHECKSUMS_FILENAME
    bundle_checksums_path = receipt_root / runner.RECEIPT_BUNDLE_CHECKSUMS_FILENAME
    manifest_path = receipt_root / runner.RECEIPT_MANIFEST_FILENAME
    tarball_path = receipt_root / f"{run_dir.name}.tar.gz" if create_tarball else None
    tarball_checksum_path = receipt_root / f"{run_dir.name}.tar.gz.sha256" if create_tarball else None

    if inspect_only:
        return ReceiptBundleResult(
            run_dir=run_dir,
            receipt_root=receipt_root,
            manifest_path=manifest_path,
            required_checksums_path=required_checksums_path,
            bundle_checksums_path=bundle_checksums_path,
            tarball_path=tarball_path,
            tarball_checksum_path=tarball_checksum_path,
            inspect_only=True,
            tier=profile.tier,
            target=profile.target,
            source_class=profile.source_class,
            family_count=len(rows),
            result=str(status.get("result")),
        )

    receipt_root.mkdir(parents=True, exist_ok=True)
    required_entries: list[dict[str, Any]] = []
    required_checksum_entries: list[tuple[str, Path]] = []
    for spec in profile.required_artifacts:
        source = run_dir / spec.source_relative_path
        bundled = receipt_root / spec.bundled_relative_path
        bundled.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, bundled)
        digest = runner.sha256_file(bundled)
        required_entries.append(
            {
                "role": spec.role,
                "source_path": repo_relative(source, repo_root),
                "bundled_path": repo_relative(bundled, repo_root),
                "size_bytes": bundled.stat().st_size,
                "sha256": digest,
            }
        )
        required_checksum_entries.append((digest, bundled))
    write_sha256sum(required_checksums_path, required_checksum_entries, repo_root)

    if create_tarball:
        assert tarball_path is not None
        with tarfile.open(tarball_path, "w:gz") as archive:
            archive.add(run_dir, arcname=run_dir.name)
        assert tarball_checksum_path is not None
        write_sha256sum(tarball_checksum_path, [(runner.sha256_file(tarball_path), tarball_path)], repo_root)

    manifest = build_receipt_manifest(
        profile,
        run_dir,
        receipt_root,
        preflight,
        status,
        rows,
        required_entries,
        required_checksums_path,
        bundle_checksums_path,
        tarball_path,
        tarball_checksum_path,
        created_at,
        repo_root,
    )
    write_json(manifest_path, manifest)

    bundle_entries = [(runner.sha256_file(manifest_path), manifest_path), (runner.sha256_file(required_checksums_path), required_checksums_path)]
    bundle_entries.extend(required_checksum_entries)
    if create_tarball:
        assert tarball_path is not None
        assert tarball_checksum_path is not None
        bundle_entries.append((runner.sha256_file(tarball_path), tarball_path))
        bundle_entries.append((runner.sha256_file(tarball_checksum_path), tarball_checksum_path))
    write_sha256sum(bundle_checksums_path, bundle_entries, repo_root)

    validation = runner.validate_operator_receipt_manifest(repo_root, manifest_path)
    if validation.status != runner.ARTIFACT_STATUS_VALID:
        raise ReceiptPackagingError("packaged receipt failed validation: " + " | ".join(validation.errors))

    return ReceiptBundleResult(
        run_dir=run_dir,
        receipt_root=receipt_root,
        manifest_path=manifest_path,
        required_checksums_path=required_checksums_path,
        bundle_checksums_path=bundle_checksums_path,
        tarball_path=tarball_path,
        tarball_checksum_path=tarball_checksum_path,
        inspect_only=False,
        tier=profile.tier,
        target=profile.target,
        source_class=profile.source_class,
        family_count=len(rows),
        result=str(status.get("result")),
    )


def render_result(result: ReceiptBundleResult) -> str:
    mode = "inspect-only" if result.inspect_only else "package"
    lines = [
        "eval-factory receipt packager:",
        f"  mode: {mode}",
        f"  tier: {result.tier}",
        f"  target: {result.target}",
        f"  source_class: {result.source_class}",
        f"  source run: {repo_relative(result.run_dir)}",
        f"  receipt root: {repo_relative(result.receipt_root)}",
        f"  result: {result.result}",
        f"  family_count: {result.family_count}",
        "artifacts:",
        f"  manifest: {repo_relative(result.manifest_path)}",
        f"  required checksums: {repo_relative(result.required_checksums_path)}",
        f"  bundle checksums: {repo_relative(result.bundle_checksums_path)}",
    ]
    if result.tarball_path is not None:
        lines.append(f"  tarball: {repo_relative(result.tarball_path)}")
        lines.append(f"  tarball checksum: {repo_relative(result.tarball_checksum_path or result.tarball_path.with_suffix(result.tarball_path.suffix + '.sha256'))}")
    else:
        lines.append("  tarball: skipped")
    lines.append("notes:")
    lines.append("  - operator receipt only; not a checkpoint or memo claim")
    lines.append("  - no model execution is launched by this helper")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = package_receipt(
            Path(args.run_dir),
            Path(args.out_root),
            create_tarball=not args.no_tarball,
            inspect_only=args.inspect_only,
        )
    except ReceiptPackagingError as exc:
        print(f"error: {exc}")
        return 1
    print(render_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
