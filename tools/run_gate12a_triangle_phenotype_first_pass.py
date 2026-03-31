#!/usr/bin/env python3
"""Record a local first-pass phenotype read from a reviewed packet CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()

DEFAULT_MANIFEST = "manifest.json"
DEFAULT_STATUS = "gate12a_triangle_phenotype_first_pass_status.json"
DEFAULT_CSV = "triangle_phenotype_first_pass.csv"
DEFAULT_MD = "gate12a_triangle_phenotype_first_pass.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture a local first-pass phenotype read from a reviewed Gate12A triangle "
            "packet CSV without changing the machine-side observable surface."
        )
    )
    parser.add_argument("--reviewed-csv", required=True)
    parser.add_argument("--source-packet-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--title-label", default="")
    return parser.parse_args()


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


def read_rows(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def build_status(rows: List[Mapping[str, Any]]) -> Dict[str, Any]:
    packet_row_count = len(rows)
    band_counts = Counter(str(row.get("provisional_closure_band", "")) for row in rows)
    reviewed_counts = Counter(str(row.get("reviewed_phenotype_tag", "")).strip() for row in rows)
    if "" in reviewed_counts:
        del reviewed_counts[""]
    return {
        "packet_row_count": packet_row_count,
        "high_tension_count": int(band_counts.get("high_tension", 0)),
        "flat_count": int(band_counts.get("flat", 0)),
        "reviewed_tag_counts": [
            {"reviewed_phenotype_tag": tag, "count": count}
            for tag, count in reviewed_counts.most_common()
        ],
    }


def build_markdown(source_packet_dir: Path, title_label: str, status: Mapping[str, Any]) -> str:
    label = f" ({title_label})" if title_label else ""
    lines = [
        f"# Gate12A Triangle Phenotype First Pass{label}",
        "",
        f"- source packet run: `{source_packet_dir.name}`",
        f"- reviewed rows: `{status['packet_row_count']}`",
        f"- code git commit: `{current_git_commit()}`",
        "",
        "## Band Counts",
        f"- `high_tension`: `{status['high_tension_count']}`",
        f"- `flat`: `{status['flat_count']}`",
        "",
        "## Reviewed Tag Counts",
    ]
    for row in status["reviewed_tag_counts"]:
        lines.append(f"- `{row['reviewed_phenotype_tag']}`: `{row['count']}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    reviewed_csv = Path(args.reviewed_csv)
    source_packet_dir = Path(args.source_packet_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(reviewed_csv)
    status = build_status(rows)

    csv_path = out_dir / DEFAULT_CSV
    status_path = out_dir / DEFAULT_STATUS
    manifest_path = out_dir / DEFAULT_MANIFEST
    markdown_path = out_dir / DEFAULT_MD

    write_csv(csv_path, rows[0].keys() if rows else [], rows)
    write_json(status_path, status)
    markdown_path.write_text(
        build_markdown(source_packet_dir=source_packet_dir, title_label=args.title_label, status=status),
        encoding="utf-8",
        newline="\n",
    )
    write_json(
        manifest_path,
        {
            "run_id": out_dir.name,
            "schema_version": "gate12a_triangle_phenotype_first_pass_v1",
            "method_id": "gate12a_triangle_phenotype_first_pass_v1",
            "code_git_commit": current_git_commit(),
            "builder_script_sha256": sha256_file(SCRIPT_PATH),
            "source_packet_dir": repo_relative_or_posix(source_packet_dir),
            "reviewed_csv": repo_relative_or_posix(reviewed_csv),
            "paths": {
                DEFAULT_STATUS: repo_relative_or_posix(status_path),
                DEFAULT_CSV: repo_relative_or_posix(csv_path),
                DEFAULT_MD: repo_relative_or_posix(markdown_path),
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
