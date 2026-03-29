#!/usr/bin/env python3
"""Build a priority reading queue from Gate12A phenotype prep artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()

SCHEMA_VERSION = "gate12a_triangle_reading_queue_v1"
METHOD_ID = "gate12a_triangle_reading_queue_v1"

DEFAULT_MANIFEST = "manifest.json"
DEFAULT_STATUS = "gate12a_triangle_reading_queue_status.json"
DEFAULT_POLICY_COMPARE = "gate12a_triangle_reading_queue_policy_compare.csv"
DEFAULT_QUEUE = "triangle_reading_queue.csv"
DEFAULT_READ = "gate12a_triangle_reading_queue.md"
DEFAULT_CHECKSUMS = "checksums.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read one Gate12A phenotype-prep template and emit a prioritized human-reading queue."
        )
    )
    parser.add_argument("--phenotype-prep-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
            writer.writerow({name: row.get(name) for name in fieldnames})


def write_text(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8", newline="\n")


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


def queue_priority(band: str, percentile: float) -> tuple[int, float]:
    if band == "high_tension":
        return (0, -percentile)
    if band == "flat":
        return (1, percentile)
    return (2, percentile)


def build_queue_rows(template_rows: Sequence[Mapping[str, str]]) -> List[Dict[str, Any]]:
    prioritized = [
        row for row in template_rows if str(row.get("provisional_closure_band") or "") in {"flat", "high_tension"}
    ]
    prioritized.sort(
        key=lambda row: (
            *queue_priority(
                str(row.get("provisional_closure_band") or ""),
                float(row.get("residual_percentile") or 0.0),
            ),
            str(row.get("cycle_id") or ""),
        )
    )

    queue_rows: List[Dict[str, Any]] = []
    for index, row in enumerate(prioritized, start=1):
        queue_rows.append(
            {
                "queue_rank": index,
                "cycle_id": str(row["cycle_id"]),
                "sample_id": str(row["sample_id"]),
                "provisional_closure_band": str(row["provisional_closure_band"]),
                "holonomy_residual_fro": float(row["holonomy_residual_fro"]),
                "residual_percentile": float(row["residual_percentile"]),
                "prompt_path": str(row["prompt_path"]),
                "answer_path": str(row["answer_path"]),
                "support_anchor_path": str(row.get("support_anchor_path") or ""),
                "conflict_anchor_path": str(row.get("conflict_anchor_path") or ""),
                "phenotype_tag": str(row.get("phenotype_tag") or ""),
                "phenotype_notes": str(row.get("phenotype_notes") or ""),
            }
        )
    return queue_rows


def build_readme(queue_rows: Sequence[Mapping[str, Any]], source_manifest: Mapping[str, Any]) -> str:
    high = sum(1 for row in queue_rows if row["provisional_closure_band"] == "high_tension")
    flat = sum(1 for row in queue_rows if row["provisional_closure_band"] == "flat")
    lines = [
        "# Gate12A Triangle Reading Queue",
        "",
        f"- source phenotype-prep run: `{source_manifest.get('run_id')}`",
        f"- source phenotype-prep code commit: `{source_manifest.get('code_git_commit')}`",
        f"- prioritized rows: `{len(queue_rows)}`",
        f"- high-tension rows in queue: `{high}`",
        f"- flat rows in queue: `{flat}`",
        "",
        "This queue intentionally excludes provisional `tense` rows.",
        "Read `high_tension` and `flat` first before widening the tag surface.",
        "",
    ]
    return "\n".join(lines)


def run_triangle_reading_queue(
    *,
    phenotype_prep_dir: Path,
    out_dir: Path,
) -> Dict[str, Any]:
    phenotype_prep_dir = Path(phenotype_prep_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = read_json(phenotype_prep_dir / "manifest.json")
    template_rows = read_csv(phenotype_prep_dir / "triangle_phenotype_tagging_template.csv")
    queue_rows = build_queue_rows(template_rows)

    manifest_path = out_dir / DEFAULT_MANIFEST
    status_path = out_dir / DEFAULT_STATUS
    policy_compare_path = out_dir / DEFAULT_POLICY_COMPARE
    queue_path = out_dir / DEFAULT_QUEUE
    read_path = out_dir / DEFAULT_READ
    checksums_path = out_dir / DEFAULT_CHECKSUMS

    status_payload = {
        "queue_row_count": len(queue_rows),
        "high_tension_queue_count": sum(1 for row in queue_rows if row["provisional_closure_band"] == "high_tension"),
        "flat_queue_count": sum(1 for row in queue_rows if row["provisional_closure_band"] == "flat"),
    }
    write_json(status_path, status_payload)
    write_csv(
        policy_compare_path,
        ("run_id", "queue_row_count", "high_tension_queue_count", "flat_queue_count"),
        [{"run_id": out_dir.name, **status_payload}],
    )
    write_csv(
        queue_path,
        (
            "queue_rank",
            "cycle_id",
            "sample_id",
            "provisional_closure_band",
            "holonomy_residual_fro",
            "residual_percentile",
            "prompt_path",
            "answer_path",
            "support_anchor_path",
            "conflict_anchor_path",
            "phenotype_tag",
            "phenotype_notes",
        ),
        queue_rows,
    )
    write_text(read_path, build_readme(queue_rows, source_manifest))

    manifest = {
        "run_id": out_dir.name,
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "code_git_commit": current_git_commit(),
        "builder_script_sha256": sha256_file(SCRIPT_PATH),
        "source_phenotype_prep_manifest_path": repo_relative_or_posix(phenotype_prep_dir / "manifest.json"),
        "source_phenotype_prep_run_id": str(source_manifest.get("run_id") or ""),
        "source_phenotype_prep_code_git_commit": str(source_manifest.get("code_git_commit") or ""),
        "paths": {
            DEFAULT_STATUS: repo_relative_or_posix(status_path),
            DEFAULT_POLICY_COMPARE: repo_relative_or_posix(policy_compare_path),
            DEFAULT_QUEUE: repo_relative_or_posix(queue_path),
            DEFAULT_READ: repo_relative_or_posix(read_path),
        },
    }
    write_json(manifest_path, manifest)
    write_json(
        checksums_path,
        {
            DEFAULT_MANIFEST: sha256_file(manifest_path),
            DEFAULT_STATUS: sha256_file(status_path),
            DEFAULT_POLICY_COMPARE: sha256_file(policy_compare_path),
            DEFAULT_QUEUE: sha256_file(queue_path),
            DEFAULT_READ: sha256_file(read_path),
        },
    )
    return {"manifest": manifest, "status": status_payload, "queue_rows": queue_rows}


def main() -> int:
    args = parse_args()
    run_triangle_reading_queue(
        phenotype_prep_dir=Path(args.phenotype_prep_dir),
        out_dir=Path(args.out_dir),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
