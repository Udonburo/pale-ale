#!/usr/bin/env python3
"""Join Gate12A triangle holonomy rows with recovered text surfaces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()

SCHEMA_VERSION = "gate12a_triangle_text_surface_audit_v1"
METHOD_ID = "gate12a_triangle_text_surface_audit_v1"

DEFAULT_MANIFEST = "manifest.json"
DEFAULT_STATUS = "gate12a_triangle_text_surface_audit_status.json"
DEFAULT_POLICY_COMPARE = "gate12a_triangle_text_surface_audit_policy_compare.csv"
DEFAULT_JOINED = "triangle_text_surface_joined.jsonl"
DEFAULT_EXTREMES = "triangle_text_surface_extremes.jsonl"
DEFAULT_READ = "gate12a_triangle_text_surface_audit.md"
DEFAULT_CHECKSUMS = "checksums.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read one Gate12A discrete-connection run plus its recovered Gate8 execution "
            "surface and emit joined triangle-level text-surface audit rows."
        )
    )
    parser.add_argument("--gate12a-dir", required=True)
    parser.add_argument("--gate8-execution-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-k", type=int, default=3)
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


def read_optional_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def sample_id_from_node_id(node_id: str) -> str:
    return str(node_id).split(":", 1)[0]


def summarize_path(values: Sequence[float]) -> Dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "max": None, "mean": None}
    xs = [float(value) for value in values]
    return {
        "min": min(xs),
        "median": statistics.median(xs),
        "max": max(xs),
        "mean": statistics.fmean(xs),
    }


def build_joined_rows(
    *,
    cycle_rows: Sequence[Mapping[str, Any]],
    holonomy_rows: Sequence[Mapping[str, Any]],
    transport_rows: Sequence[Mapping[str, Any]],
    gate8_execution_dir: Path,
) -> List[Dict[str, Any]]:
    cycle_map = {str(row["cycle_id"]): row for row in cycle_rows}
    edge_map = {str(row["edge_id"]): row for row in transport_rows}

    joined_rows: List[Dict[str, Any]] = []
    for holonomy_row in holonomy_rows:
        if str(holonomy_row.get("holonomy_status") or "") != "defined":
            continue
        cycle = cycle_map[str(holonomy_row["cycle_id"])]
        sample_id = sample_id_from_node_id(str(holonomy_row["base_node_id"]))
        sample_dir = gate8_execution_dir / "samples" / sample_id
        edge_path = [str(edge_id) for edge_id in cycle["edge_id_path"]]
        relation_kind_path = [str(edge_map[edge_id]["relation_kind"]) for edge_id in edge_path]
        anchor_qualified_path = [bool(edge_map[edge_id]["anchor_qualified"]) for edge_id in edge_path]
        compatibility_gap_path = [float(edge_map[edge_id]["compatibility_gap_fro"]) for edge_id in edge_path]

        prompt_path = sample_dir / "prompt.txt"
        answer_path = sample_dir / "answer.txt"
        support_anchor_path = sample_dir / "support_anchor.txt"
        conflict_anchor_path = sample_dir / "conflict_anchor.txt"

        joined_rows.append(
            {
                "cycle_id": str(holonomy_row["cycle_id"]),
                "sample_id": sample_id,
                "base_node_id": str(holonomy_row["base_node_id"]),
                "edge_id_path": edge_path,
                "node_id_path": [str(node_id) for node_id in cycle["node_id_path"]],
                "anchor_qualified_path": anchor_qualified_path,
                "relation_kind_path": relation_kind_path,
                "compatibility_gap_path": compatibility_gap_path,
                "compatibility_gap_path_summary": summarize_path(compatibility_gap_path),
                "holonomy_residual_fro": float(holonomy_row["holonomy_residual_fro"]),
                "prompt_path": repo_relative_or_posix(prompt_path),
                "answer_path": repo_relative_or_posix(answer_path),
                "support_anchor_path": repo_relative_or_posix(support_anchor_path) if support_anchor_path.exists() else "",
                "conflict_anchor_path": repo_relative_or_posix(conflict_anchor_path) if conflict_anchor_path.exists() else "",
                "prompt_text": read_optional_text(prompt_path),
                "answer_text": read_optional_text(answer_path),
                "support_anchor_text": read_optional_text(support_anchor_path),
                "conflict_anchor_text": read_optional_text(conflict_anchor_path),
            }
        )

    joined_rows.sort(key=lambda row: (float(row["holonomy_residual_fro"]), str(row["cycle_id"])))
    total = len(joined_rows)
    for index, row in enumerate(joined_rows):
        row["residual_rank_index"] = index
        row["residual_percentile"] = (index / (total - 1)) if total > 1 else 0.0
    return joined_rows


def build_readme(
    *,
    source_manifest: Mapping[str, Any],
    joined_rows: Sequence[Mapping[str, Any]],
    flattest_rows: Sequence[Mapping[str, Any]],
    distorted_rows: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# Gate12A Triangle Text-Surface Audit",
        "",
        f"- source Gate12A run: `{source_manifest.get('run_id')}`",
        f"- source Gate12A code commit: `{source_manifest.get('code_git_commit')}`",
        f"- joined defined triangles: `{len(joined_rows)}`",
        "",
        "The current surface is triangle-level and empirical only.",
        "This artifact does not claim `high holonomy residual = bad answer`.",
        "",
        "## Flattest Triangles",
        "",
    ]
    for row in flattest_rows:
        lines.extend(
            [
                f"- `{row['cycle_id']}` residual `{float(row['holonomy_residual_fro']):.6f}` percentile `{float(row['residual_percentile']):.3f}`",
                f"  - sample: `{row['sample_id']}`",
                f"  - relation kinds: `{row['relation_kind_path']}`",
                f"  - anchor-qualified path: `{row['anchor_qualified_path']}`",
                f"  - compatibility summary: `{row['compatibility_gap_path_summary']}`",
                "  - prompt:",
                "```text",
                str(row["prompt_text"]),
                "```",
                "  - answer:",
                "```text",
                str(row["answer_text"]),
                "```",
            ]
        )
        if row["support_anchor_text"]:
            lines.extend(["  - support anchor:", "```text", str(row["support_anchor_text"]), "```"])
        if row["conflict_anchor_text"]:
            lines.extend(["  - conflict anchor:", "```text", str(row["conflict_anchor_text"]), "```"])
    lines.extend(["", "## Most Distorted Triangles", ""])
    for row in distorted_rows:
        lines.extend(
            [
                f"- `{row['cycle_id']}` residual `{float(row['holonomy_residual_fro']):.6f}` percentile `{float(row['residual_percentile']):.3f}`",
                f"  - sample: `{row['sample_id']}`",
                f"  - relation kinds: `{row['relation_kind_path']}`",
                f"  - anchor-qualified path: `{row['anchor_qualified_path']}`",
                f"  - compatibility summary: `{row['compatibility_gap_path_summary']}`",
                "  - prompt:",
                "```text",
                str(row["prompt_text"]),
                "```",
                "  - answer:",
                "```text",
                str(row["answer_text"]),
                "```",
            ]
        )
        if row["support_anchor_text"]:
            lines.extend(["  - support anchor:", "```text", str(row["support_anchor_text"]), "```"])
        if row["conflict_anchor_text"]:
            lines.extend(["  - conflict anchor:", "```text", str(row["conflict_anchor_text"]), "```"])
    lines.append("")
    return "\n".join(lines)


def run_triangle_text_surface_audit(
    *,
    gate12a_dir: Path,
    gate8_execution_dir: Path,
    out_dir: Path,
    top_k: int,
) -> Dict[str, Any]:
    gate12a_dir = Path(gate12a_dir)
    gate8_execution_dir = Path(gate8_execution_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = read_json(gate12a_dir / "manifest.json")
    cycle_rows = read_jsonl(gate12a_dir / "explicit_triangle_cycle_registry.jsonl")
    holonomy_rows = read_jsonl(gate12a_dir / "triangle_holonomy_registry.jsonl")
    transport_rows = read_jsonl(gate12a_dir / "transport_relation_registry.jsonl")

    joined_rows = build_joined_rows(
        cycle_rows=cycle_rows,
        holonomy_rows=holonomy_rows,
        transport_rows=transport_rows,
        gate8_execution_dir=gate8_execution_dir,
    )
    k = max(1, int(top_k))
    flattest_rows = joined_rows[:k]
    distorted_rows = list(reversed(joined_rows[-k:]))

    manifest_path = out_dir / DEFAULT_MANIFEST
    status_path = out_dir / DEFAULT_STATUS
    policy_compare_path = out_dir / DEFAULT_POLICY_COMPARE
    joined_path = out_dir / DEFAULT_JOINED
    extremes_path = out_dir / DEFAULT_EXTREMES
    read_path = out_dir / DEFAULT_READ
    checksums_path = out_dir / DEFAULT_CHECKSUMS

    status_payload = {
        "defined_triangle_joined_count": len(joined_rows),
        "flattest_row_count": len(flattest_rows),
        "most_distorted_row_count": len(distorted_rows),
    }
    write_json(status_path, status_payload)
    write_csv(
        policy_compare_path,
        ("run_id", "defined_triangle_joined_count", "flattest_row_count", "most_distorted_row_count"),
        [{"run_id": out_dir.name, **status_payload}],
    )
    write_jsonl(joined_path, joined_rows)
    write_jsonl(
        extremes_path,
        [{"extreme_kind": "flattest", **row} for row in flattest_rows]
        + [{"extreme_kind": "most_distorted", **row} for row in distorted_rows],
    )
    write_text(
        read_path,
        build_readme(
            source_manifest=source_manifest,
            joined_rows=joined_rows,
            flattest_rows=flattest_rows,
            distorted_rows=distorted_rows,
        ),
    )

    manifest = {
        "run_id": out_dir.name,
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "code_git_commit": current_git_commit(),
        "builder_script_sha256": sha256_file(SCRIPT_PATH),
        "source_gate12a_manifest_path": repo_relative_or_posix(gate12a_dir / "manifest.json"),
        "source_gate12a_run_id": str(source_manifest.get("run_id") or ""),
        "source_gate12a_code_git_commit": str(source_manifest.get("code_git_commit") or ""),
        "source_gate8_execution_dir": repo_relative_or_posix(gate8_execution_dir),
        "paths": {
            DEFAULT_STATUS: repo_relative_or_posix(status_path),
            DEFAULT_POLICY_COMPARE: repo_relative_or_posix(policy_compare_path),
            DEFAULT_JOINED: repo_relative_or_posix(joined_path),
            DEFAULT_EXTREMES: repo_relative_or_posix(extremes_path),
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
            DEFAULT_JOINED: sha256_file(joined_path),
            DEFAULT_EXTREMES: sha256_file(extremes_path),
            DEFAULT_READ: sha256_file(read_path),
        },
    )
    return {
        "manifest": manifest,
        "status": status_payload,
        "joined_rows": joined_rows,
        "flattest_rows": flattest_rows,
        "distorted_rows": distorted_rows,
    }


def main() -> int:
    args = parse_args()
    run_triangle_text_surface_audit(
        gate12a_dir=Path(args.gate12a_dir),
        gate8_execution_dir=Path(args.gate8_execution_dir),
        out_dir=Path(args.out_dir),
        top_k=int(args.top_k),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
