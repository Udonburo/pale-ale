#!/usr/bin/env python3
"""Summarize Gate12A real-run calibration and seed-audit surfaces."""

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

SCHEMA_VERSION = "gate12a_calibration_seed_audit_v1"
METHOD_ID = "gate12a_calibration_seed_audit_v1"

DEFAULT_MANIFEST = "manifest.json"
DEFAULT_STATUS = "gate12a_calibration_seed_audit_status.json"
DEFAULT_POLICY_COMPARE = "gate12a_calibration_seed_audit_policy_compare.csv"
DEFAULT_READ = "gate12a_calibration_seed_audit.md"
DEFAULT_SUBREGIME_QUANTILES = "transport_gap_quantiles_by_subregime.csv"
DEFAULT_TRIANGLE_EXTREMES = "triangle_holonomy_extremes.jsonl"
DEFAULT_CHECKSUMS = "checksums.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read one committed Gate12A discrete-connection run and emit calibration / "
            "seed-audit summaries without reopening the transport court."
        )
    )
    parser.add_argument("--gate12a-dir", required=True)
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


def quantile(values: Sequence[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(float(v) for v in values)
    index = (len(xs) - 1) * p
    lo = int(index)
    hi = min(lo + 1, len(xs) - 1)
    frac = index - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def summarize_distribution(values: Sequence[float]) -> Dict[str, Any]:
    xs = sorted(float(v) for v in values)
    if not xs:
        return {
            "n": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "max": None,
        }
    return {
        "n": len(xs),
        "min": xs[0],
        "p25": quantile(xs, 0.25),
        "median": quantile(xs, 0.50),
        "p75": quantile(xs, 0.75),
        "p90": quantile(xs, 0.90),
        "max": xs[-1],
    }


def build_transport_quantile_rows(transport_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups = [
        ("overall", lambda row: row.get("compatibility_gap_fro") is not None),
        ("trusted_tree", lambda row: row.get("relation_kind") == "trusted_tree" and row.get("compatibility_gap_fro") is not None),
        ("residual_chord", lambda row: row.get("relation_kind") == "residual_chord" and row.get("compatibility_gap_fro") is not None),
        ("anchor_qualified", lambda row: bool(row.get("anchor_qualified")) and row.get("compatibility_gap_fro") is not None),
        ("plain", lambda row: (not bool(row.get("anchor_qualified"))) and row.get("compatibility_gap_fro") is not None),
    ]
    rows: List[Dict[str, Any]] = []
    for label, predicate in groups:
        values = [float(row["compatibility_gap_fro"]) for row in transport_rows if predicate(row)]
        rows.append({"subregime": label, **summarize_distribution(values)})
    return rows


def build_triangle_extremes(
    *,
    cycle_rows: Sequence[Mapping[str, Any]],
    holonomy_rows: Sequence[Mapping[str, Any]],
    transport_rows: Sequence[Mapping[str, Any]],
    top_k: int,
) -> Dict[str, Any]:
    cycle_map = {str(row["cycle_id"]): row for row in cycle_rows}
    edge_map = {str(row["edge_id"]): row for row in transport_rows}

    defined_rows = []
    for row in holonomy_rows:
        if str(row.get("holonomy_status") or "") != "defined":
            continue
        cycle = cycle_map[str(row["cycle_id"])]
        edge_path = [str(edge_id) for edge_id in cycle["edge_id_path"]]
        anchor_flags = [bool(edge_map[edge_id]["anchor_qualified"]) for edge_id in edge_path]
        defined_rows.append(
            {
                "cycle_id": str(row["cycle_id"]),
                "base_node_id": str(row["base_node_id"]),
                "holonomy_residual_fro": float(row["holonomy_residual_fro"]),
                "node_id_path": [str(node_id) for node_id in cycle["node_id_path"]],
                "edge_id_path": edge_path,
                "anchor_qualified_path": anchor_flags,
            }
        )

    defined_rows.sort(key=lambda row: (float(row["holonomy_residual_fro"]), str(row["cycle_id"])))
    return {
        "defined_rows": defined_rows,
        "flattest": defined_rows[:top_k],
        "most_distorted": list(reversed(defined_rows[-top_k:])),
        "triangles_with_any_anchor": sum(1 for row in defined_rows if any(bool(flag) for flag in row["anchor_qualified_path"])),
        "triangles_with_all_anchor": sum(1 for row in defined_rows if all(bool(flag) for flag in row["anchor_qualified_path"])),
    }


def build_readme(
    *,
    source_manifest: Mapping[str, Any],
    status_payload: Mapping[str, Any],
    transport_quantiles: Sequence[Mapping[str, Any]],
    holonomy_distribution: Mapping[str, Any],
    flattest_rows: Sequence[Mapping[str, Any]],
    distorted_rows: Sequence[Mapping[str, Any]],
) -> str:
    quantile_rows = [
        "| subregime | n | min | p25 | median | p75 | p90 | max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in transport_quantiles:
        quantile_rows.append(
            "| {subregime} | {n} | {min:.6f} | {p25:.6f} | {median:.6f} | {p75:.6f} | {p90:.6f} | {max:.6f} |".format(
                subregime=row["subregime"],
                n=int(row["n"]),
                min=float(row["min"] or 0.0),
                p25=float(row["p25"] or 0.0),
                median=float(row["median"] or 0.0),
                p75=float(row["p75"] or 0.0),
                p90=float(row["p90"] or 0.0),
                max=float(row["max"] or 0.0),
            )
        )

    lines = [
        "# Gate12A Calibration / Seed Audit",
        "",
        f"- source Gate12A run: `{source_manifest.get('run_id')}`",
        f"- source Gate12A code commit: `{source_manifest.get('code_git_commit')}`",
        f"- transport relations: `{int(status_payload['transport_relation_count'])}`",
        f"- explicit triangle cycles: `{int(status_payload['explicit_triangle_cycle_count'])}`",
        f"- defined triangle holonomies: `{int(status_payload['defined_triangle_holonomy_count'])}`",
        f"- defined triangles with any anchor-qualified edge: `{int(status_payload['triangles_with_any_anchor_count'])}`",
        f"- defined triangles with all-anchor edge paths: `{int(status_payload['triangles_with_all_anchor_count'])}`",
        "",
        "## Transport Gap Quantiles",
        "",
        *quantile_rows,
        "",
        "## Holonomy Residual Distribution",
        "",
        "- `holonomy_residual_fro` is read only on `defined` triangles",
        "- distribution summary:",
        f"  - n = `{int(holonomy_distribution['n'])}`",
        f"  - min = `{float(holonomy_distribution['min'] or 0.0):.6f}`",
        f"  - p25 = `{float(holonomy_distribution['p25'] or 0.0):.6f}`",
        f"  - median = `{float(holonomy_distribution['median'] or 0.0):.6f}`",
        f"  - p75 = `{float(holonomy_distribution['p75'] or 0.0):.6f}`",
        f"  - p90 = `{float(holonomy_distribution['p90'] or 0.0):.6f}`",
        f"  - max = `{float(holonomy_distribution['max'] or 0.0):.6f}`",
        "",
        "## Flattest Triangles",
        "",
    ]
    for row in flattest_rows:
        lines.extend(
            [
                f"- `{row['cycle_id']}` residual `{float(row['holonomy_residual_fro']):.6f}`",
                f"  - base node: `{row['base_node_id']}`",
                f"  - edge path: `{row['edge_id_path']}`",
                f"  - node path: `{row['node_id_path']}`",
                f"  - anchor-qualified path: `{row['anchor_qualified_path']}`",
            ]
        )
    lines.extend(["", "## Most Distorted Triangles", ""])
    for row in distorted_rows:
        lines.extend(
            [
                f"- `{row['cycle_id']}` residual `{float(row['holonomy_residual_fro']):.6f}`",
                f"  - base node: `{row['base_node_id']}`",
                f"  - edge path: `{row['edge_id_path']}`",
                f"  - node path: `{row['node_id_path']}`",
                f"  - anchor-qualified path: `{row['anchor_qualified_path']}`",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def run_calibration_seed_audit(
    *,
    gate12a_dir: Path,
    out_dir: Path,
    top_k: int,
) -> Dict[str, Any]:
    gate12a_dir = Path(gate12a_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = read_json(gate12a_dir / "manifest.json")
    transport_rows = read_jsonl(gate12a_dir / "transport_relation_registry.jsonl")
    cycle_rows = read_jsonl(gate12a_dir / "explicit_triangle_cycle_registry.jsonl")
    holonomy_rows = read_jsonl(gate12a_dir / "triangle_holonomy_registry.jsonl")

    transport_quantiles = build_transport_quantile_rows(transport_rows)
    triangle_extremes = build_triangle_extremes(
        cycle_rows=cycle_rows,
        holonomy_rows=holonomy_rows,
        transport_rows=transport_rows,
        top_k=max(1, int(top_k)),
    )
    holonomy_distribution = summarize_distribution(
        [float(row["holonomy_residual_fro"]) for row in triangle_extremes["defined_rows"]]
    )

    zero_overlap_count = sum(1 for row in transport_rows if str(row.get("transport_case") or "") == "undefined_zero_overlap")
    compatible_count = sum(1 for row in transport_rows if str(row.get("transport_level_compatibility_status") or "") == "compatible")
    incompatible_count = sum(1 for row in transport_rows if str(row.get("transport_level_compatibility_status") or "") == "incompatible")

    status_payload = {
        "transport_relation_count": len(transport_rows),
        "explicit_triangle_cycle_count": len(cycle_rows),
        "defined_triangle_holonomy_count": int(holonomy_distribution["n"]),
        "zero_overlap_count": zero_overlap_count,
        "compatible_transport_count": compatible_count,
        "incompatible_transport_count": incompatible_count,
        "triangles_with_any_anchor_count": int(triangle_extremes["triangles_with_any_anchor"]),
        "triangles_with_all_anchor_count": int(triangle_extremes["triangles_with_all_anchor"]),
    }

    manifest_path = out_dir / DEFAULT_MANIFEST
    status_path = out_dir / DEFAULT_STATUS
    policy_compare_path = out_dir / DEFAULT_POLICY_COMPARE
    subregime_quantiles_path = out_dir / DEFAULT_SUBREGIME_QUANTILES
    triangle_extremes_path = out_dir / DEFAULT_TRIANGLE_EXTREMES
    read_path = out_dir / DEFAULT_READ
    checksums_path = out_dir / DEFAULT_CHECKSUMS

    write_json(status_path, status_payload)
    write_csv(
        policy_compare_path,
        (
            "run_id",
            "transport_relation_count",
            "explicit_triangle_cycle_count",
            "defined_triangle_holonomy_count",
            "zero_overlap_count",
            "compatible_transport_count",
            "incompatible_transport_count",
            "triangles_with_any_anchor_count",
            "triangles_with_all_anchor_count",
        ),
        [{"run_id": out_dir.name, **status_payload}],
    )
    write_csv(
        subregime_quantiles_path,
        ("subregime", "n", "min", "p25", "median", "p75", "p90", "max"),
        transport_quantiles,
    )
    write_jsonl(
        triangle_extremes_path,
        [
            {"extreme_kind": "flattest", **row} for row in triangle_extremes["flattest"]
        ]
        + [
            {"extreme_kind": "most_distorted", **row} for row in triangle_extremes["most_distorted"]
        ],
    )
    write_text(
        read_path,
        build_readme(
            source_manifest=source_manifest,
            status_payload=status_payload,
            transport_quantiles=transport_quantiles,
            holonomy_distribution=holonomy_distribution,
            flattest_rows=triangle_extremes["flattest"],
            distorted_rows=triangle_extremes["most_distorted"],
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
        "paths": {
            DEFAULT_STATUS: repo_relative_or_posix(status_path),
            DEFAULT_POLICY_COMPARE: repo_relative_or_posix(policy_compare_path),
            DEFAULT_SUBREGIME_QUANTILES: repo_relative_or_posix(subregime_quantiles_path),
            DEFAULT_TRIANGLE_EXTREMES: repo_relative_or_posix(triangle_extremes_path),
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
            DEFAULT_SUBREGIME_QUANTILES: sha256_file(subregime_quantiles_path),
            DEFAULT_TRIANGLE_EXTREMES: sha256_file(triangle_extremes_path),
            DEFAULT_READ: sha256_file(read_path),
        },
    )
    return {
        "manifest": manifest,
        "status": status_payload,
        "transport_quantiles": transport_quantiles,
        "triangle_extremes": triangle_extremes,
    }


def main() -> int:
    args = parse_args()
    run_calibration_seed_audit(
        gate12a_dir=Path(args.gate12a_dir),
        out_dir=Path(args.out_dir),
        top_k=int(args.top_k),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
