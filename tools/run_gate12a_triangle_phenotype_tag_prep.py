#!/usr/bin/env python3
"""Prepare provisional closure bands and a human-tagging template for Gate12A triangles."""

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

SCHEMA_VERSION = "gate12a_triangle_phenotype_tag_prep_v1"
METHOD_ID = "gate12a_triangle_phenotype_tag_prep_v1"

DEFAULT_MANIFEST = "manifest.json"
DEFAULT_STATUS = "gate12a_triangle_phenotype_tag_prep_status.json"
DEFAULT_POLICY_COMPARE = "gate12a_triangle_phenotype_tag_prep_policy_compare.csv"
DEFAULT_BAND_REGISTRY = "triangle_phenotype_band_registry.jsonl"
DEFAULT_TEMPLATE = "triangle_phenotype_tagging_template.csv"
DEFAULT_READ = "gate12a_triangle_phenotype_tag_prep.md"
DEFAULT_CHECKSUMS = "checksums.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read one Gate12A triangle text-surface audit and emit provisional closure bands "
            "plus a human phenotype-tagging template."
        )
    )
    parser.add_argument("--triangle-text-audit-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--flat-quantile", type=float, default=0.25)
    parser.add_argument("--high-quantile", type=float, default=0.75)
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


def quantile(values: Sequence[float], p: float) -> float:
    xs = sorted(float(value) for value in values)
    index = (len(xs) - 1) * p
    lo = int(index)
    hi = min(lo + 1, len(xs) - 1)
    frac = index - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def build_band_rows(
    joined_rows: Sequence[Mapping[str, Any]],
    *,
    flat_quantile: float,
    high_quantile: float,
) -> Dict[str, Any]:
    residuals = [float(row["holonomy_residual_fro"]) for row in joined_rows]
    if not residuals:
        raise ValueError("joined audit rows are empty")
    flat_cut = quantile(residuals, flat_quantile)
    high_cut = quantile(residuals, high_quantile)

    band_rows: List[Dict[str, Any]] = []
    counts = {"flat": 0, "tense": 0, "high_tension": 0}
    for row in joined_rows:
        residual = float(row["holonomy_residual_fro"])
        if residual <= flat_cut:
            band = "flat"
        elif residual > high_cut:
            band = "high_tension"
        else:
            band = "tense"
        counts[band] += 1
        band_rows.append(
            {
                "cycle_id": str(row["cycle_id"]),
                "sample_id": str(row["sample_id"]),
                "base_node_id": str(row["base_node_id"]),
                "edge_id_path": row["edge_id_path"],
                "anchor_qualified_path": row["anchor_qualified_path"],
                "relation_kind_path": row["relation_kind_path"],
                "compatibility_gap_path_summary": row["compatibility_gap_path_summary"],
                "holonomy_residual_fro": residual,
                "residual_percentile": float(row["residual_percentile"]),
                "prompt_path": str(row["prompt_path"]),
                "answer_path": str(row["answer_path"]),
                "support_anchor_path": str(row.get("support_anchor_path") or ""),
                "conflict_anchor_path": str(row.get("conflict_anchor_path") or ""),
                "provisional_closure_band": band,
            }
        )
    return {
        "band_rows": band_rows,
        "counts": counts,
        "flat_cut": flat_cut,
        "high_cut": high_cut,
    }


def build_template_rows(band_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    template_rows: List[Dict[str, Any]] = []
    for row in band_rows:
        template_rows.append(
            {
                "cycle_id": row["cycle_id"],
                "sample_id": row["sample_id"],
                "base_node_id": row["base_node_id"],
                "edge_id_path": json.dumps(row["edge_id_path"], ensure_ascii=False),
                "anchor_qualified_path": json.dumps(row["anchor_qualified_path"], ensure_ascii=False),
                "relation_kind_path": json.dumps(row["relation_kind_path"], ensure_ascii=False),
                "compatibility_gap_path_summary": json.dumps(row["compatibility_gap_path_summary"], ensure_ascii=False),
                "holonomy_residual_fro": row["holonomy_residual_fro"],
                "residual_percentile": row["residual_percentile"],
                "provisional_closure_band": row["provisional_closure_band"],
                "prompt_path": row["prompt_path"],
                "answer_path": row["answer_path"],
                "support_anchor_path": row["support_anchor_path"],
                "conflict_anchor_path": row["conflict_anchor_path"],
                "phenotype_tag": "",
                "phenotype_notes": "",
            }
        )
    return template_rows


def build_readme(
    *,
    source_manifest: Mapping[str, Any],
    status_payload: Mapping[str, Any],
) -> str:
    return "\n".join(
        [
            "# Gate12A Triangle Phenotype Tag Prep",
            "",
            f"- source text-surface audit run: `{source_manifest.get('run_id')}`",
            f"- source text-surface audit code commit: `{source_manifest.get('code_git_commit')}`",
            f"- joined rows: `{int(status_payload['joined_row_count'])}`",
            f"- provisional flat cut: `{float(status_payload['flat_cut']):.6f}`",
            f"- provisional high-tension cut: `{float(status_payload['high_cut']):.6f}`",
            f"- flat rows: `{int(status_payload['flat_count'])}`",
            f"- tense rows: `{int(status_payload['tense_count'])}`",
            f"- high-tension rows: `{int(status_payload['high_tension_count'])}`",
            "",
            "These bands are provisional closure bands only.",
            "They are not correctness labels and they are not doctrine.",
            "Human tagging should precede any stronger threshold claim.",
            "",
            "Recommended phenotype tags:",
            "",
            "- `conflict_respected`",
            "- `support_fused`",
            "- `compensatory_closure`",
            "- `anchor_overreach`",
            "- `surface_noise_only`",
            "- `needs_manual_read`",
            "",
        ]
    )


def run_triangle_phenotype_tag_prep(
    *,
    triangle_text_audit_dir: Path,
    out_dir: Path,
    flat_quantile: float,
    high_quantile: float,
) -> Dict[str, Any]:
    triangle_text_audit_dir = Path(triangle_text_audit_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = read_json(triangle_text_audit_dir / "manifest.json")
    joined_rows = read_jsonl(triangle_text_audit_dir / "triangle_text_surface_joined.jsonl")

    band_result = build_band_rows(
        joined_rows,
        flat_quantile=float(flat_quantile),
        high_quantile=float(high_quantile),
    )
    band_rows = band_result["band_rows"]
    template_rows = build_template_rows(band_rows)

    status_payload = {
        "joined_row_count": len(joined_rows),
        "flat_cut": float(band_result["flat_cut"]),
        "high_cut": float(band_result["high_cut"]),
        "flat_count": int(band_result["counts"]["flat"]),
        "tense_count": int(band_result["counts"]["tense"]),
        "high_tension_count": int(band_result["counts"]["high_tension"]),
    }

    manifest_path = out_dir / DEFAULT_MANIFEST
    status_path = out_dir / DEFAULT_STATUS
    policy_compare_path = out_dir / DEFAULT_POLICY_COMPARE
    band_registry_path = out_dir / DEFAULT_BAND_REGISTRY
    template_path = out_dir / DEFAULT_TEMPLATE
    read_path = out_dir / DEFAULT_READ
    checksums_path = out_dir / DEFAULT_CHECKSUMS

    write_json(status_path, status_payload)
    write_csv(
        policy_compare_path,
        ("run_id", "joined_row_count", "flat_cut", "high_cut", "flat_count", "tense_count", "high_tension_count"),
        [{"run_id": out_dir.name, **status_payload}],
    )
    write_jsonl(band_registry_path, band_rows)
    write_csv(
        template_path,
        (
            "cycle_id",
            "sample_id",
            "base_node_id",
            "edge_id_path",
            "anchor_qualified_path",
            "relation_kind_path",
            "compatibility_gap_path_summary",
            "holonomy_residual_fro",
            "residual_percentile",
            "provisional_closure_band",
            "prompt_path",
            "answer_path",
            "support_anchor_path",
            "conflict_anchor_path",
            "phenotype_tag",
            "phenotype_notes",
        ),
        template_rows,
    )
    write_text(read_path, build_readme(source_manifest=source_manifest, status_payload=status_payload))

    manifest = {
        "run_id": out_dir.name,
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "code_git_commit": current_git_commit(),
        "builder_script_sha256": sha256_file(SCRIPT_PATH),
        "source_triangle_text_audit_manifest_path": repo_relative_or_posix(triangle_text_audit_dir / "manifest.json"),
        "source_triangle_text_audit_run_id": str(source_manifest.get("run_id") or ""),
        "source_triangle_text_audit_code_git_commit": str(source_manifest.get("code_git_commit") or ""),
        "flat_quantile": float(flat_quantile),
        "high_quantile": float(high_quantile),
        "paths": {
            DEFAULT_STATUS: repo_relative_or_posix(status_path),
            DEFAULT_POLICY_COMPARE: repo_relative_or_posix(policy_compare_path),
            DEFAULT_BAND_REGISTRY: repo_relative_or_posix(band_registry_path),
            DEFAULT_TEMPLATE: repo_relative_or_posix(template_path),
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
            DEFAULT_BAND_REGISTRY: sha256_file(band_registry_path),
            DEFAULT_TEMPLATE: sha256_file(template_path),
            DEFAULT_READ: sha256_file(read_path),
        },
    )
    return {
        "manifest": manifest,
        "status": status_payload,
        "band_rows": band_rows,
    }


def main() -> int:
    args = parse_args()
    run_triangle_phenotype_tag_prep(
        triangle_text_audit_dir=Path(args.triangle_text_audit_dir),
        out_dir=Path(args.out_dir),
        flat_quantile=float(args.flat_quantile),
        high_quantile=float(args.high_quantile),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
