#!/usr/bin/env python3
"""Build a human-readable packet from the Gate12A triangle reading queue."""

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

SCHEMA_VERSION = "gate12a_triangle_reading_packet_v1"
METHOD_ID = "gate12a_triangle_reading_packet_v1"

DEFAULT_MANIFEST = "manifest.json"
DEFAULT_STATUS = "gate12a_triangle_reading_packet_status.json"
DEFAULT_POLICY_COMPARE = "gate12a_triangle_reading_packet_policy_compare.csv"
DEFAULT_PACKET_ROWS = "triangle_reading_packet_rows.jsonl"
DEFAULT_READ = "gate12a_triangle_reading_packet.md"
DEFAULT_CHECKSUMS = "checksums.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read one Gate12A reading queue plus the corresponding triangle text-surface audit "
            "and emit a rank-ordered human review packet."
        )
    )
    parser.add_argument("--reading-queue-dir", required=True)
    parser.add_argument("--triangle-text-audit-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--balanced-per-band", type=int, default=0)
    return parser.parse_args()


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


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames:
            reader.fieldnames = [normalize_csv_key(name) for name in reader.fieldnames]
        rows: List[Dict[str, str]] = []
        for raw_row in reader:
            rows.append({normalize_csv_key(key): value for key, value in raw_row.items() if key is not None})
        return rows


def normalize_csv_key(name: str) -> str:
    cleaned = name.replace("\ufeff", "").strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] == '"':
        cleaned = cleaned[1:-1]
    return cleaned


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n")


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


def build_packet_rows(
    queue_rows: Sequence[Mapping[str, str]],
    joined_rows: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    balanced_per_band: int,
) -> List[Dict[str, Any]]:
    joined_map = {str(row["cycle_id"]): row for row in joined_rows}
    selected_queue_rows: List[Mapping[str, str]]
    if balanced_per_band > 0:
        high_rows = [row for row in queue_rows if str(row["provisional_closure_band"]) == "high_tension"]
        flat_rows = [row for row in queue_rows if str(row["provisional_closure_band"]) == "flat"]
        selected_queue_rows = high_rows[:balanced_per_band] + flat_rows[:balanced_per_band]
    else:
        selected_queue_rows = list(queue_rows)

    packet_rows: List[Dict[str, Any]] = []
    for queue_row in selected_queue_rows:
        cycle_id = str(queue_row["cycle_id"])
        joined = joined_map[cycle_id]
        packet_rows.append(
            {
                "queue_rank": int(queue_row["queue_rank"]),
                "cycle_id": cycle_id,
                "sample_id": str(queue_row["sample_id"]),
                "provisional_closure_band": str(queue_row["provisional_closure_band"]),
                "holonomy_residual_fro": float(queue_row["holonomy_residual_fro"]),
                "residual_percentile": float(queue_row["residual_percentile"]),
                "edge_id_path": joined["edge_id_path"],
                "anchor_qualified_path": joined["anchor_qualified_path"],
                "relation_kind_path": joined["relation_kind_path"],
                "compatibility_gap_path_summary": joined["compatibility_gap_path_summary"],
                "prompt_path": str(queue_row["prompt_path"]),
                "answer_path": str(queue_row["answer_path"]),
                "support_anchor_path": str(queue_row.get("support_anchor_path") or ""),
                "conflict_anchor_path": str(queue_row.get("conflict_anchor_path") or ""),
                "prompt_text": str(joined.get("prompt_text") or ""),
                "answer_text": str(joined.get("answer_text") or ""),
                "support_anchor_text": str(joined.get("support_anchor_text") or ""),
                "conflict_anchor_text": str(joined.get("conflict_anchor_text") or ""),
                "phenotype_tag": str(queue_row.get("phenotype_tag") or ""),
                "phenotype_notes": str(queue_row.get("phenotype_notes") or ""),
            }
        )
    packet_rows.sort(key=lambda row: int(row["queue_rank"]))
    if balanced_per_band <= 0 and limit > 0:
        return packet_rows[:limit]
    return packet_rows


def build_readme(packet_rows: Sequence[Mapping[str, Any]], queue_manifest: Mapping[str, Any]) -> str:
    lines = [
        "# Gate12A Triangle Reading Packet",
        "",
        f"- source reading queue run: `{queue_manifest.get('run_id')}`",
        f"- source reading queue code commit: `{queue_manifest.get('code_git_commit')}`",
        f"- packet rows: `{len(packet_rows)}`",
        "",
    ]
    for row in packet_rows:
        lines.extend(
            [
                f"## Queue {int(row['queue_rank'])}: `{row['cycle_id']}`",
                "",
                f"- sample: `{row['sample_id']}`",
                f"- band: `{row['provisional_closure_band']}`",
                f"- residual: `{float(row['holonomy_residual_fro']):.6f}`",
                f"- percentile: `{float(row['residual_percentile']):.3f}`",
                f"- relation kinds: `{row['relation_kind_path']}`",
                f"- anchor-qualified path: `{row['anchor_qualified_path']}`",
                f"- compatibility summary: `{row['compatibility_gap_path_summary']}`",
                f"- suggested phenotype tag: `{row['phenotype_tag']}`",
                f"- notes: `{row['phenotype_notes']}`",
                "",
                "### Prompt",
                "```text",
                str(row["prompt_text"]),
                "```",
                "",
                "### Answer",
                "```text",
                str(row["answer_text"]),
                "```",
            ]
        )
        if row["support_anchor_text"]:
            lines.extend(["", "### Support Anchor", "```text", str(row["support_anchor_text"]), "```"])
        if row["conflict_anchor_text"]:
            lines.extend(["", "### Conflict Anchor", "```text", str(row["conflict_anchor_text"]), "```"])
        lines.append("")
    return "\n".join(lines)


def build_selection_metadata(
    packet_rows: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    balanced_per_band: int,
) -> Dict[str, Any]:
    selection_mode = "balanced_per_band" if balanced_per_band > 0 else "queue_prefix"
    return {
        "selection_mode": selection_mode,
        "queue_limit": int(limit),
        "balanced_per_band": int(balanced_per_band),
        "selected_high_tension_count": sum(
            1 for row in packet_rows if str(row["provisional_closure_band"]) == "high_tension"
        ),
        "selected_flat_count": sum(
            1 for row in packet_rows if str(row["provisional_closure_band"]) == "flat"
        ),
    }


def run_triangle_reading_packet(
    *,
    reading_queue_dir: Path,
    triangle_text_audit_dir: Path,
    out_dir: Path,
    limit: int,
    balanced_per_band: int,
) -> Dict[str, Any]:
    reading_queue_dir = Path(reading_queue_dir)
    triangle_text_audit_dir = Path(triangle_text_audit_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    queue_manifest = read_json(reading_queue_dir / "manifest.json")
    queue_rows = read_csv(reading_queue_dir / "triangle_reading_queue.csv")
    joined_rows = read_jsonl(triangle_text_audit_dir / "triangle_text_surface_joined.jsonl")
    packet_rows = build_packet_rows(
        queue_rows,
        joined_rows,
        limit=int(limit),
        balanced_per_band=int(balanced_per_band),
    )
    selection_metadata = build_selection_metadata(
        packet_rows,
        limit=int(limit),
        balanced_per_band=int(balanced_per_band),
    )

    manifest_path = out_dir / DEFAULT_MANIFEST
    status_path = out_dir / DEFAULT_STATUS
    policy_compare_path = out_dir / DEFAULT_POLICY_COMPARE
    packet_rows_path = out_dir / DEFAULT_PACKET_ROWS
    read_path = out_dir / DEFAULT_READ
    checksums_path = out_dir / DEFAULT_CHECKSUMS

    status_payload = {
        "packet_row_count": len(packet_rows),
        "high_tension_packet_count": sum(1 for row in packet_rows if row["provisional_closure_band"] == "high_tension"),
        "flat_packet_count": sum(1 for row in packet_rows if row["provisional_closure_band"] == "flat"),
        **selection_metadata,
    }
    write_json(status_path, status_payload)
    write_csv(
        policy_compare_path,
        (
            "run_id",
            "packet_row_count",
            "high_tension_packet_count",
            "flat_packet_count",
            "selection_mode",
            "queue_limit",
            "balanced_per_band",
            "selected_high_tension_count",
            "selected_flat_count",
        ),
        [{"run_id": out_dir.name, **status_payload}],
    )
    write_jsonl(packet_rows_path, packet_rows)
    write_text(read_path, build_readme(packet_rows, queue_manifest))

    manifest = {
        "run_id": out_dir.name,
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "code_git_commit": current_git_commit(),
        "builder_script_sha256": sha256_file(SCRIPT_PATH),
        "source_reading_queue_manifest_path": repo_relative_or_posix(reading_queue_dir / "manifest.json"),
        "source_reading_queue_run_id": str(queue_manifest.get("run_id") or ""),
        "source_reading_queue_code_git_commit": str(queue_manifest.get("code_git_commit") or ""),
        "source_triangle_text_audit_manifest_path": repo_relative_or_posix(triangle_text_audit_dir / "manifest.json"),
        "packet_selection": selection_metadata,
        "paths": {
            DEFAULT_STATUS: repo_relative_or_posix(status_path),
            DEFAULT_POLICY_COMPARE: repo_relative_or_posix(policy_compare_path),
            DEFAULT_PACKET_ROWS: repo_relative_or_posix(packet_rows_path),
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
            DEFAULT_PACKET_ROWS: sha256_file(packet_rows_path),
            DEFAULT_READ: sha256_file(read_path),
        },
    )
    return {"manifest": manifest, "status": status_payload, "packet_rows": packet_rows}


def main() -> int:
    args = parse_args()
    run_triangle_reading_packet(
        reading_queue_dir=Path(args.reading_queue_dir),
        triangle_text_audit_dir=Path(args.triangle_text_audit_dir),
        out_dir=Path(args.out_dir),
        limit=int(args.limit),
        balanced_per_band=int(args.balanced_per_band),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
