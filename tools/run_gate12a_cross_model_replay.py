#!/usr/bin/env python3
"""Run a fixed-family Gate8 -> Gate12A replay harness for one model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import run_gate12a_family_replay as family_replay


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
SCRIPT_PATH = Path(__file__).resolve()

DEFAULT_FAMILIES = ("transcript_v1", "briefing_v1", "archive_v1")
DEFAULT_STATUS = "cross_model_family_summary.csv"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"


@dataclass(frozen=True)
class FamilyConfig:
    rendering_family: str
    family_token: str
    samples_per_cell: int


FAMILY_CONFIGS: Dict[str, FamilyConfig] = {
    "transcript_v1": FamilyConfig("transcript_v1", "transcript", 32),
    "briefing_v1": FamilyConfig("briefing_v1", "briefing", 50),
    "archive_v1": FamilyConfig("archive_v1", "archive", 32),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed transcript_v1 / briefing_v1 / archive_v1 family set through "
            "Gate8 materialization and unchanged Gate12A replay for one model."
        )
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-label", help="Stable short label for run ids. Defaults to a normalized model id.")
    parser.add_argument("--families", nargs="+", default=list(DEFAULT_FAMILIES), choices=sorted(FAMILY_CONFIGS))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gate12a-top-k", type=int, default=3)
    parser.add_argument("--balanced-per-band", type=int, default=6)
    parser.add_argument("--reading-limit", type=int, default=0)
    parser.add_argument("--out-root", default="runs")
    parser.add_argument("--summary-run-id", help="Optional explicit summary run id under the out root.")
    return parser.parse_args()


def normalize_model_label(model_id: str, explicit_label: str | None) -> str:
    if explicit_label:
        return explicit_label
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", model_id.strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "model"


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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_run_prefix(model_label: str, config: FamilyConfig) -> str:
    total_rows = config.samples_per_cell * 4
    return f"gate8cm_{model_label}_{config.family_token}_{total_rows}r"


def build_summary_run_id(model_label: str, explicit_run_id: str | None) -> str:
    if explicit_run_id:
        return explicit_run_id
    return f"gate12a_cross_model_replay_{model_label}"


def build_commands(
    model_id: str,
    model_label: str,
    family_names: Sequence[str],
    device: str,
    topk: int,
    seed: int,
    gate12a_top_k: int,
    balanced_per_band: int,
    reading_limit: int,
) -> List[List[str]]:
    python_exe = sys.executable
    commands: List[List[str]] = []
    for family_name in family_names:
        config = FAMILY_CONFIGS[family_name]
        run_prefix = build_run_prefix(model_label, config)
        gate8_execution_dir = Path("runs") / f"{run_prefix}_candidate_execution"
        commands.append(
            [
                python_exe,
                str((TOOLS_DIR / "run_gate8_scaleup.py").resolve()),
                "--run-prefix",
                run_prefix,
                "--samples-per-cell",
                str(config.samples_per_cell),
                "--device",
                device,
                "--topk",
                str(topk),
                "--seed",
                str(seed),
                "--rendering-family",
                config.rendering_family,
                "--model-id",
                model_id,
            ]
        )
        replay_command = [
            python_exe,
            str((TOOLS_DIR / "run_gate12a_family_replay.py").resolve()),
            "--gate8-execution-dir",
            str(gate8_execution_dir),
            "--top-k",
            str(gate12a_top_k),
        ]
        if balanced_per_band > 0:
            replay_command.extend(["--balanced-per-band", str(balanced_per_band)])
        if reading_limit > 0:
            replay_command.extend(["--reading-limit", str(reading_limit)])
        commands.append(replay_command)
    return commands


def run_subprocess(command: Sequence[str]) -> None:
    completed = subprocess.run(
        list(command),
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed rc={completed.returncode}: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def summarize_first_pass(replay_dirs: Mapping[str, Path]) -> tuple[str, str]:
    first_pass_dir = replay_dirs["out_root"] / f"gate12a_triangle_phenotype_first_pass_recheck_from_gate12a_upstream_{replay_dirs['upstream_label']}"
    status_path = first_pass_dir / "gate12a_triangle_phenotype_first_pass_status.json"
    if not status_path.exists():
        return "pending_local_read", ""
    status = read_json(status_path)
    reviewed_counts = status.get("reviewed_tag_counts") or []
    detail = ";".join(
        f"{row.get('reviewed_phenotype_tag')}={row.get('count')}"
        for row in reviewed_counts
    )
    return "available", detail


def build_family_summary_rows(
    out_root: Path,
    model_id: str,
    model_label: str,
    family_names: Sequence[str],
    balanced_per_band: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for family_name in family_names:
        config = FAMILY_CONFIGS[family_name]
        run_prefix = build_run_prefix(model_label, config)
        origin_label = run_prefix
        replay_dirs = family_replay.build_run_dirs(out_root=out_root, origin_label=origin_label, balanced_per_band=balanced_per_band)
        replay_dirs["out_root"] = out_root
        replay_dirs["upstream_label"] = Path(replay_dirs["gate12a"]).name.removeprefix("gate12a_discrete_connection_recheck_from_gate12a_upstream_")

        gate12a_status = read_json(replay_dirs["gate12a"] / "gate12a_discrete_connection_status.json")
        calibration_status = read_json(replay_dirs["calibration"] / "gate12a_calibration_seed_audit_status.json")
        quantile_rows = list(csv.DictReader((replay_dirs["calibration"] / "transport_gap_quantiles_by_subregime.csv").open("r", encoding="utf-8")))
        quantiles_by_subregime = {str(row["subregime"]): row for row in quantile_rows}
        first_pass_status, first_pass_detail = summarize_first_pass(replay_dirs)

        trusted_tree_median = float(quantiles_by_subregime["trusted_tree"]["median"])
        residual_chord_median = float(quantiles_by_subregime["residual_chord"]["median"])
        anchor_qualified_median = float(quantiles_by_subregime["anchor_qualified"]["median"])
        plain_median = float(quantiles_by_subregime["plain"]["median"])
        defined_triangle_count = int(gate12a_status["defined_triangle_holonomy_count"])
        any_anchor_count = int(calibration_status["triangles_with_any_anchor_count"])

        rows.append(
            {
                "model_label": model_label,
                "model_id": model_id,
                "rendering_family": family_name,
                "run_prefix": run_prefix,
                "samples_per_cell": config.samples_per_cell,
                "gate8_execution_dir": repo_relative_or_posix(out_root / f"{run_prefix}_candidate_execution"),
                "gate12a_dir": repo_relative_or_posix(replay_dirs["gate12a"]),
                "reading_packet_dir": repo_relative_or_posix(replay_dirs["reading_packet"]),
                "zero_overlap_clear": int(calibration_status["zero_overlap_count"]) == 0,
                "all_defined_triangles_anchor_rich": any_anchor_count == defined_triangle_count and int(calibration_status["triangles_with_all_anchor_count"]) == 0,
                "trusted_tree_median": trusted_tree_median,
                "residual_chord_median": residual_chord_median,
                "anchor_qualified_median": anchor_qualified_median,
                "plain_median": plain_median,
                "trusted_tree_gt_residual_chord": trusted_tree_median > residual_chord_median,
                "plain_gt_anchor_qualified": plain_median > anchor_qualified_median,
                "extreme_band_first_pass_status": first_pass_status,
                "extreme_band_first_pass_detail": first_pass_detail,
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    model_label = normalize_model_label(args.model_id, args.model_label)
    out_root = Path(args.out_root)
    summary_run_id = build_summary_run_id(model_label, args.summary_run_id)
    summary_dir = out_root / summary_run_id
    summary_dir.mkdir(parents=True, exist_ok=True)

    commands = build_commands(
        model_id=args.model_id,
        model_label=model_label,
        family_names=args.families,
        device=args.device,
        topk=args.topk,
        seed=args.seed,
        gate12a_top_k=args.gate12a_top_k,
        balanced_per_band=args.balanced_per_band,
        reading_limit=args.reading_limit,
    )
    for command in commands:
        run_subprocess(command)

    summary_rows = build_family_summary_rows(
        out_root=out_root,
        model_id=args.model_id,
        model_label=model_label,
        family_names=args.families,
        balanced_per_band=args.balanced_per_band,
    )

    manifest_path = summary_dir / DEFAULT_MANIFEST
    summary_path = summary_dir / DEFAULT_STATUS
    checksums_path = summary_dir / DEFAULT_CHECKSUMS

    fieldnames = [
        "model_label",
        "model_id",
        "rendering_family",
        "run_prefix",
        "samples_per_cell",
        "gate8_execution_dir",
        "gate12a_dir",
        "reading_packet_dir",
        "zero_overlap_clear",
        "all_defined_triangles_anchor_rich",
        "trusted_tree_median",
        "residual_chord_median",
        "anchor_qualified_median",
        "plain_median",
        "trusted_tree_gt_residual_chord",
        "plain_gt_anchor_qualified",
        "extreme_band_first_pass_status",
        "extreme_band_first_pass_detail",
    ]
    write_csv(summary_path, fieldnames, summary_rows)

    manifest = {
        "run_id": summary_run_id,
        "schema_version": "gate12a_cross_model_replay_v1",
        "method_id": "gate12a_cross_model_replay_v1",
        "code_git_commit": current_git_commit(),
        "builder_script_sha256": sha256_file(SCRIPT_PATH),
        "model_id": args.model_id,
        "model_label": model_label,
        "family_set": list(args.families),
        "balanced_per_band": args.balanced_per_band,
        "reading_limit": args.reading_limit,
        "paths": {
            DEFAULT_STATUS: repo_relative_or_posix(summary_path),
        },
    }
    write_json(manifest_path, manifest)
    write_json(
        checksums_path,
        {
            DEFAULT_MANIFEST: sha256_file(manifest_path),
            DEFAULT_STATUS: sha256_file(summary_path),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
