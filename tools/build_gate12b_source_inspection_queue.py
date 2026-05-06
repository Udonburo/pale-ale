#!/usr/bin/env python3
"""Build a source-facing inspection queue from Gate12B candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()

SCHEMA_VERSION = "gate12b_source_inspection_queue_v1"
METHOD_ID = "gate12b_source_inspection_queue_v1"

DEFAULT_MANIFEST = "manifest.json"
DEFAULT_STATUS = "gate12b_source_inspection_queue_status.json"
DEFAULT_QUEUE_CSV = "gate12b_source_inspection_queue.csv"
DEFAULT_QUEUE_JSONL = "gate12b_source_inspection_queue.jsonl"
DEFAULT_READ = "gate12b_source_inspection_queue.md"
DEFAULT_CHECKSUMS = "checksums.json"

GATE12B_CANDIDATES = "invariant_signature_candidates.jsonl"
GATE12B_MANIFEST = "manifest.json"
TEXT_SURFACE_JOINED = "triangle_text_surface_joined.jsonl"
TEXT_SURFACE_MANIFEST = "manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read Gate12B candidate runs plus matching Gate12A triangle text-surface "
            "audits and emit a source-facing inspection queue."
        )
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--case",
        action="append",
        nargs=3,
        metavar=("LABEL", "GATE12B_DIR", "TRIANGLE_TEXT_AUDIT_DIR"),
        default=[],
        help="One inspection case. Repeat for each model/family surface.",
    )
    parser.add_argument("--per-band-limit", type=int, default=2)
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n")


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: csv_value(row.get(name)) for name in fieldnames})


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


def validate_output_dir_boundary(out_dir: Path, input_dirs: Sequence[Path]) -> None:
    resolved_out_dir = Path(out_dir).resolve()
    for input_dir in input_dirs:
        resolved_input_dir = Path(input_dir).resolve()
        if resolved_out_dir == resolved_input_dir:
            raise ValueError(
                "Gate12B source inspection out_dir must not be the same directory "
                f"as an input artifact directory: {repo_relative_or_posix(input_dir)}"
            )
        if resolved_input_dir in resolved_out_dir.parents:
            raise ValueError(
                "Gate12B source inspection out_dir must not be inside an input "
                f"artifact directory: {repo_relative_or_posix(input_dir)}"
            )


def validate_matching_source_gate12a_run(
    *,
    case_label: str,
    gate12b_manifest: Mapping[str, Any],
    text_manifest: Mapping[str, Any],
) -> str:
    gate12b_source = str(gate12b_manifest.get("source_gate12a_run_id") or "")
    text_source = str(text_manifest.get("source_gate12a_run_id") or "")
    if not gate12b_source:
        raise ValueError(f"{case_label} Gate12B manifest is missing source_gate12a_run_id")
    if not text_source:
        raise ValueError(f"{case_label} triangle text audit manifest is missing source_gate12a_run_id")
    if gate12b_source != text_source:
        raise ValueError(
            f"{case_label} source_gate12a_run_id mismatch: "
            f"Gate12B={gate12b_source!r} text_audit={text_source!r}"
        )
    return gate12b_source


def candidate_side(candidate_kind: str) -> str:
    if str(candidate_kind).startswith("flat_"):
        return "flat"
    if str(candidate_kind).startswith("high_tension_"):
        return "high_tension"
    return str(candidate_kind)


def normalized_contains(text: str, needle: str) -> bool:
    text_norm = " ".join(str(text).lower().split())
    needle_norm = " ".join(str(needle).lower().split())
    return bool(needle_norm and needle_norm in text_norm)


def candidate_sort_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    side = candidate_side(str(row.get("candidate_kind") or ""))
    residual = float(row.get("holonomy_residual_fro") or 0.0)
    if side == "flat":
        residual_key = residual
    elif side == "high_tension":
        residual_key = -residual
    else:
        residual_key = residual
    return (
        side,
        str(row.get("relation_kind_signature") or ""),
        residual_key,
        str(row.get("cycle_id") or ""),
    )


def select_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    per_band_limit: int,
) -> List[Mapping[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for row in candidates:
        side = candidate_side(str(row.get("candidate_kind") or ""))
        relation_signature = str(row.get("relation_kind_signature") or "")
        if side not in {"flat", "high_tension"}:
            continue
        grouped.setdefault((side, relation_signature), []).append(row)

    selected: List[Mapping[str, Any]] = []
    limit = max(1, int(per_band_limit))
    for key in sorted(grouped):
        selected.extend(sorted(grouped[key], key=candidate_sort_key)[:limit])
    return sorted(selected, key=candidate_sort_key)


def build_case_rows(
    *,
    case_label: str,
    gate12b_dir: Path,
    triangle_text_audit_dir: Path,
    per_band_limit: int,
) -> List[Dict[str, Any]]:
    gate12b_manifest = read_json(gate12b_dir / GATE12B_MANIFEST)
    text_manifest = read_json(triangle_text_audit_dir / TEXT_SURFACE_MANIFEST)
    source_gate12a_run_id = validate_matching_source_gate12a_run(
        case_label=str(case_label),
        gate12b_manifest=gate12b_manifest,
        text_manifest=text_manifest,
    )
    candidates = read_jsonl(gate12b_dir / GATE12B_CANDIDATES)
    joined_rows = read_jsonl(triangle_text_audit_dir / TEXT_SURFACE_JOINED)
    joined_by_cycle = {str(row["cycle_id"]): row for row in joined_rows}

    rows: List[Dict[str, Any]] = []
    for candidate in select_candidates(candidates, per_band_limit=per_band_limit):
        cycle_id = str(candidate["cycle_id"])
        if cycle_id not in joined_by_cycle:
            raise ValueError(f"{case_label} candidate references missing text-surface cycle_id: {cycle_id}")
        joined = joined_by_cycle[cycle_id]
        answer_text = str(joined.get("answer_text") or "")
        support_anchor_text = str(joined.get("support_anchor_text") or "")
        conflict_anchor_text = str(joined.get("conflict_anchor_text") or "")
        rows.append(
            {
                "case_label": str(case_label),
                "gate12b_run_id": str(gate12b_manifest.get("run_id") or gate12b_dir.name),
                "gate12b_run_dir": repo_relative_or_posix(gate12b_dir),
                "gate12b_observer_mode_set": str(gate12b_manifest.get("observer_mode_set") or ""),
                "gate12b_top_k": int(gate12b_manifest.get("top_k") or 0),
                "gate12b_min_observer_support": int(gate12b_manifest.get("min_observer_support") or 0),
                "gate12b_min_scale_support": int(gate12b_manifest.get("min_scale_support") or 0),
                "source_gate12a_run_id": source_gate12a_run_id,
                "triangle_text_audit_run_id": str(text_manifest.get("run_id") or triangle_text_audit_dir.name),
                "cycle_id": cycle_id,
                "sample_id": str(joined.get("sample_id") or ""),
                "candidate_side": candidate_side(str(candidate.get("candidate_kind") or "")),
                "candidate_kind": str(candidate.get("candidate_kind") or ""),
                "relation_kind_signature": str(candidate.get("relation_kind_signature") or ""),
                "residual_quantile_band": str(candidate.get("residual_quantile_band") or ""),
                "holonomy_residual_fro": float(candidate.get("holonomy_residual_fro") or 0.0),
                "residual_percentile": float(candidate.get("residual_percentile") or 0.0),
                "observer_support_count": int(candidate.get("observer_support_count") or 0),
                "scale_support_count": int(candidate.get("scale_support_count") or 0),
                "support_observers": candidate.get("support_observers") or [],
                "support_scales": candidate.get("support_scales") or [],
                "observer_scope_groups": candidate.get("observer_scope_groups") or [],
                "edge_id_path": joined.get("edge_id_path") or [],
                "node_id_path": joined.get("node_id_path") or [],
                "relation_kind_path": joined.get("relation_kind_path") or [],
                "anchor_qualified_path": joined.get("anchor_qualified_path") or [],
                "compatibility_gap_path_summary": joined.get("compatibility_gap_path_summary") or {},
                "prompt_path": str(joined.get("prompt_path") or ""),
                "answer_path": str(joined.get("answer_path") or ""),
                "support_anchor_path": str(joined.get("support_anchor_path") or ""),
                "conflict_anchor_path": str(joined.get("conflict_anchor_path") or ""),
                "prompt_text": str(joined.get("prompt_text") or ""),
                "answer_text": answer_text,
                "support_anchor_text": support_anchor_text,
                "conflict_anchor_text": conflict_anchor_text,
                "answer_contains_support_anchor_text": normalized_contains(answer_text, support_anchor_text),
                "answer_contains_conflict_anchor_text": normalized_contains(answer_text, conflict_anchor_text),
            }
        )
    return rows


def with_queue_ranks(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        ranked.append({"queue_rank": index, **dict(row)})
    return ranked


def build_status(rows: Sequence[Mapping[str, Any]], case_count: int, per_band_limit: int) -> Dict[str, Any]:
    flat = sum(1 for row in rows if row["candidate_side"] == "flat")
    high = sum(1 for row in rows if row["candidate_side"] == "high_tension")
    conflict_hits = sum(1 for row in rows if bool(row["answer_contains_conflict_anchor_text"]))
    support_hits = sum(1 for row in rows if bool(row["answer_contains_support_anchor_text"]))
    return {
        "case_count": int(case_count),
        "queue_row_count": len(rows),
        "per_band_limit": int(per_band_limit),
        "flat_queue_count": flat,
        "high_tension_queue_count": high,
        "answer_contains_support_anchor_text_count": support_hits,
        "answer_contains_conflict_anchor_text_count": conflict_hits,
    }


QUEUE_CSV_FIELDS = (
    "queue_rank",
    "case_label",
    "gate12b_run_id",
    "gate12b_observer_mode_set",
    "gate12b_top_k",
    "gate12b_min_observer_support",
    "gate12b_min_scale_support",
    "source_gate12a_run_id",
    "cycle_id",
    "sample_id",
    "candidate_side",
    "relation_kind_signature",
    "residual_quantile_band",
    "holonomy_residual_fro",
    "residual_percentile",
    "observer_support_count",
    "scale_support_count",
    "support_observers",
    "support_scales",
    "relation_kind_path",
    "anchor_qualified_path",
    "compatibility_gap_path_summary",
    "prompt_path",
    "answer_path",
    "support_anchor_path",
    "conflict_anchor_path",
    "answer_contains_support_anchor_text",
    "answer_contains_conflict_anchor_text",
)


def build_readme(rows: Sequence[Mapping[str, Any]], status: Mapping[str, Any]) -> str:
    lines = [
        "# Gate12B Source Inspection Queue",
        "",
        f"- cases: `{status['case_count']}`",
        f"- queue rows: `{status['queue_row_count']}`",
        f"- per-band limit: `{status['per_band_limit']}`",
        f"- flat rows: `{status['flat_queue_count']}`",
        f"- high-tension rows: `{status['high_tension_queue_count']}`",
        "",
        "This queue is source-facing only.",
        "It does not convert a Gate12B candidate into an answer-quality label.",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## Queue {int(row['queue_rank'])}: `{row['case_label']}` `{row['cycle_id']}`",
                "",
                f"- side: `{row['candidate_side']}`",
                f"- relation signature: `{row['relation_kind_signature']}`",
                f"- residual: `{float(row['holonomy_residual_fro']):.6f}`",
                f"- percentile: `{float(row['residual_percentile']):.3f}`",
                f"- observer support: `{row['observer_support_count']}`",
                f"- scale support: `{row['scale_support_count']}`",
                f"- sample: `{row['sample_id']}`",
                f"- relation path: `{row['relation_kind_path']}`",
                f"- anchor-qualified path: `{row['anchor_qualified_path']}`",
                f"- compatibility summary: `{row['compatibility_gap_path_summary']}`",
                f"- answer contains support anchor text: `{row['answer_contains_support_anchor_text']}`",
                f"- answer contains conflict anchor text: `{row['answer_contains_conflict_anchor_text']}`",
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


def run_gate12b_source_inspection_queue(
    *,
    cases: Sequence[Tuple[str, Path, Path]],
    out_dir: Path,
    per_band_limit: int,
) -> Dict[str, Any]:
    if not cases:
        raise ValueError("at least one --case is required")
    normalized_cases = [(str(label), Path(gate12b_dir), Path(text_audit_dir)) for label, gate12b_dir, text_audit_dir in cases]
    out_dir = Path(out_dir)
    input_dirs = [input_dir for _, gate12b_dir, text_audit_dir in normalized_cases for input_dir in (gate12b_dir, text_audit_dir)]
    validate_output_dir_boundary(out_dir, input_dirs)

    selected_rows: List[Dict[str, Any]] = []
    case_manifests: List[Dict[str, str]] = []
    for label, gate12b_dir, text_audit_dir in normalized_cases:
        selected_rows.extend(
            build_case_rows(
                case_label=str(label),
                gate12b_dir=gate12b_dir,
                triangle_text_audit_dir=text_audit_dir,
                per_band_limit=int(per_band_limit),
            )
        )
        case_manifests.append(
            {
                "case_label": str(label),
                "gate12b_manifest_path": repo_relative_or_posix(gate12b_dir / GATE12B_MANIFEST),
                "triangle_text_audit_manifest_path": repo_relative_or_posix(text_audit_dir / TEXT_SURFACE_MANIFEST),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    rows = with_queue_ranks(
        sorted(
            selected_rows,
            key=lambda row: (
                str(row["case_label"]),
                str(row["candidate_side"]),
                str(row["relation_kind_signature"]),
                float(row["holonomy_residual_fro"]) if row["candidate_side"] == "flat" else -float(row["holonomy_residual_fro"]),
                str(row["cycle_id"]),
            ),
        )
    )
    status = build_status(rows, case_count=len(cases), per_band_limit=int(per_band_limit))

    manifest_path = out_dir / DEFAULT_MANIFEST
    status_path = out_dir / DEFAULT_STATUS
    queue_csv_path = out_dir / DEFAULT_QUEUE_CSV
    queue_jsonl_path = out_dir / DEFAULT_QUEUE_JSONL
    read_path = out_dir / DEFAULT_READ
    checksums_path = out_dir / DEFAULT_CHECKSUMS

    write_json(status_path, status)
    write_csv(queue_csv_path, QUEUE_CSV_FIELDS, rows)
    write_jsonl(queue_jsonl_path, rows)
    write_text(read_path, build_readme(rows, status))

    manifest = {
        "run_id": out_dir.name,
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "code_git_commit": current_git_commit(),
        "builder_script_sha256": sha256_file(SCRIPT_PATH),
        "per_band_limit": int(per_band_limit),
        "cases": case_manifests,
        "paths": {
            DEFAULT_STATUS: repo_relative_or_posix(status_path),
            DEFAULT_QUEUE_CSV: repo_relative_or_posix(queue_csv_path),
            DEFAULT_QUEUE_JSONL: repo_relative_or_posix(queue_jsonl_path),
            DEFAULT_READ: repo_relative_or_posix(read_path),
        },
        "status": status,
    }
    write_json(manifest_path, manifest)
    write_json(
        checksums_path,
        {
            DEFAULT_MANIFEST: sha256_file(manifest_path),
            DEFAULT_STATUS: sha256_file(status_path),
            DEFAULT_QUEUE_CSV: sha256_file(queue_csv_path),
            DEFAULT_QUEUE_JSONL: sha256_file(queue_jsonl_path),
            DEFAULT_READ: sha256_file(read_path),
        },
    )
    return {"manifest": manifest, "status": status, "queue_rows": rows}


def main() -> int:
    args = parse_args()
    cases = [(label, Path(gate12b_dir), Path(text_audit_dir)) for label, gate12b_dir, text_audit_dir in args.case]
    run_gate12b_source_inspection_queue(
        cases=cases,
        out_dir=Path(args.out_dir),
        per_band_limit=int(args.per_band_limit),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
