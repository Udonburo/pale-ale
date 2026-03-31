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
from typing import Any, Dict, Iterable, List, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()

DEFAULT_MANIFEST = "manifest.json"
DEFAULT_STATUS = "gate12a_triangle_phenotype_first_pass_status.json"
DEFAULT_CSV = "triangle_phenotype_first_pass.csv"
DEFAULT_MD = "gate12a_triangle_phenotype_first_pass.md"
DEFAULT_PACKET_ROWS = "triangle_reading_packet_rows.jsonl"
IDENTIFIER_FIELDS = ("queue_rank", "cycle_id", "sample_id", "provisional_closure_band")


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


def read_packet_rows(source_packet_dir: Path) -> List[Dict[str, Any]]:
    packet_rows_path = source_packet_dir / DEFAULT_PACKET_ROWS
    rows: List[Dict[str, Any]] = []
    with open(packet_rows_path, "r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


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


def identifier_tuple(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")).strip() for field in IDENTIFIER_FIELDS)


def sorted_identifier_tuples(rows: Sequence[Mapping[str, Any]]) -> List[tuple[str, ...]]:
    return sorted((identifier_tuple(row) for row in rows), key=lambda item: item[0])


def validate_reviewed_rows_against_packet(
    reviewed_rows: Sequence[Mapping[str, Any]],
    packet_rows: Sequence[Mapping[str, Any]],
) -> None:
    if len(reviewed_rows) != len(packet_rows):
        raise ValueError(
            f"reviewed row count {len(reviewed_rows)} does not match packet row count {len(packet_rows)}"
        )
    reviewed_identifiers = sorted_identifier_tuples(reviewed_rows)
    packet_identifiers = sorted_identifier_tuples(packet_rows)
    if reviewed_identifiers != packet_identifiers:
        raise ValueError("reviewed rows do not match source packet identifiers")


def validate_reviewed_rows_are_complete(reviewed_rows: Sequence[Mapping[str, Any]]) -> None:
    missing_identifiers = [
        identifier_tuple(row)
        for row in reviewed_rows
        if not str(row.get("reviewed_phenotype_tag", "")).strip()
    ]
    if missing_identifiers:
        raise ValueError(
            "reviewed rows are missing reviewed_phenotype_tag values for "
            f"{len(missing_identifiers)} row(s)"
        )


def build_status(rows: List[Mapping[str, Any]], source_packet_row_count: int) -> Dict[str, Any]:
    packet_row_count = len(rows)
    band_counts = Counter(str(row.get("provisional_closure_band", "")).strip() for row in rows)
    reviewed_counts = Counter(str(row.get("reviewed_phenotype_tag", "")).strip() for row in rows)
    if "" in reviewed_counts:
        del reviewed_counts[""]
    reviewed_counts_by_band = Counter(
        (
            str(row.get("provisional_closure_band", "")).strip(),
            str(row.get("reviewed_phenotype_tag", "")).strip(),
        )
        for row in rows
        if str(row.get("reviewed_phenotype_tag", "")).strip()
    )
    return {
        "packet_row_count": packet_row_count,
        "source_packet_row_count": source_packet_row_count,
        "validated_identifier_fields": list(IDENTIFIER_FIELDS),
        "high_tension_count": int(band_counts.get("high_tension", 0)),
        "flat_count": int(band_counts.get("flat", 0)),
        "reviewed_tag_counts": [
            {"reviewed_phenotype_tag": tag, "count": count}
            for tag, count in reviewed_counts.most_common()
        ],
        "reviewed_tag_counts_by_band": [
            {
                "provisional_closure_band": band,
                "reviewed_phenotype_tag": tag,
                "count": count,
            }
            for (band, tag), count in sorted(
                reviewed_counts_by_band.items(),
                key=lambda item: (item[0][0], -item[1], item[0][1]),
            )
        ],
    }


def build_markdown(source_packet_dir: Path, title_label: str, status: Mapping[str, Any]) -> str:
    label = f" ({title_label})" if title_label else ""
    lines = [
        f"# Gate12A Triangle Phenotype First Pass{label}",
        "",
        f"- source packet run: `{source_packet_dir.name}`",
        f"- reviewed rows: `{status['packet_row_count']}`",
        f"- validated against source rows: `{status['source_packet_row_count']}`",
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
    lines.extend(["", "## Reviewed Tag Counts By Band"])
    for row in status["reviewed_tag_counts_by_band"]:
        lines.append(
            f"- `{row['provisional_closure_band']}` / `{row['reviewed_phenotype_tag']}`: `{row['count']}`"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    reviewed_csv = Path(args.reviewed_csv)
    source_packet_dir = Path(args.source_packet_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(reviewed_csv)
    packet_rows = read_packet_rows(source_packet_dir)
    validate_reviewed_rows_against_packet(rows, packet_rows)
    validate_reviewed_rows_are_complete(rows)
    status = build_status(rows, source_packet_row_count=len(packet_rows))

    csv_path = out_dir / DEFAULT_CSV
    status_path = out_dir / DEFAULT_STATUS
    manifest_path = out_dir / DEFAULT_MANIFEST
    markdown_path = out_dir / DEFAULT_MD
    source_packet_rows_path = source_packet_dir / DEFAULT_PACKET_ROWS

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
            "source_packet_rows_sha256": sha256_file(source_packet_rows_path),
            "source_packet_row_count": len(packet_rows),
            "validated_identifier_fields": list(IDENTIFIER_FIELDS),
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
