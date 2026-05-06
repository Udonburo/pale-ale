#!/usr/bin/env python3
"""Annotate Gate12B source inspection queue rows with source-facing tags."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()

SCHEMA_VERSION = "gate12b_source_annotation_v1"
METHOD_ID = "gate12b_source_annotation_v1"

DEFAULT_ANNOTATIONS_JSONL = "gate12b_source_annotations.jsonl"
DEFAULT_ANNOTATIONS_CSV = "gate12b_source_annotations.csv"
DEFAULT_SUMMARY_JSON = "gate12b_source_annotation_summary.json"
DEFAULT_SUMMARY_CSV = "gate12b_source_annotation_summary.csv"
DEFAULT_READ = "gate12b_source_annotation.md"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"

ALLOWED_SOURCE_FACING_TAGS = (
    "support-following",
    "conflict-following",
    "non-gluing",
    "ambiguous",
)
DERIVED_ANNOTATION_MODE = "exact_anchor_and_non_gluing_phrase_v1"
SUPPLIED_ANNOTATION_MODE = "supplied_source_annotation_v1"

QUEUE_REQUIRED_FIELDS = (
    "queue_rank",
    "case_label",
    "cycle_id",
    "candidate_side",
    "relation_kind_signature",
    "answer_text",
    "answer_contains_support_anchor_text",
    "answer_contains_conflict_anchor_text",
)
ANNOTATION_REQUIRED_FIELDS = (
    "queue_rank",
    "case_label",
    "cycle_id",
    "candidate_side",
    "relation_kind_signature",
    "source_facing_tag",
    "evidence_note",
    "annotator",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a Gate12B source inspection queue and emit/validate source-facing annotations."
        )
    )
    parser.add_argument("--queue-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--annotation-jsonl", default="")
    parser.add_argument("--derive-from-queue", action="store_true")
    parser.add_argument("--annotator", default="codex_source_surface_v1")
    return parser.parse_args()


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


def validate_output_dir_boundary(out_dir: Path, input_paths: Sequence[Path]) -> None:
    resolved_out_dir = Path(out_dir).resolve()
    input_dirs = [Path(path).resolve().parent for path in input_paths]
    for input_dir in input_dirs:
        if resolved_out_dir == input_dir:
            raise ValueError(
                "Gate12B source annotation out_dir must not be the same directory "
                f"as an input artifact directory: {repo_relative_or_posix(input_dir)}"
            )
        if input_dir in resolved_out_dir.parents:
            raise ValueError(
                "Gate12B source annotation out_dir must not be inside an input "
                f"artifact directory: {repo_relative_or_posix(input_dir)}"
            )


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def row_key(row: Mapping[str, Any]) -> Tuple[int, str, str]:
    return (int(row["queue_rank"]), str(row["case_label"]), str(row["cycle_id"]))


def require_fields(row: Mapping[str, Any], fields: Sequence[str], context: str) -> None:
    missing = [field for field in fields if field not in row]
    if missing:
        raise ValueError(f"{context} missing required fields: {', '.join(missing)}")


def normalize_text(text: Any) -> str:
    return " ".join(str(text).lower().split())


def has_non_gluing_phrase(answer_text: str) -> bool:
    answer = normalize_text(answer_text)
    phrases = (
        "no direct",
        "not warranted",
        "cannot be glued",
        "cannot be inferred",
        "cannot infer",
        "separate ledgers",
        "cannot support",
        "does not warrant",
        "does not establish",
    )
    return any(phrase in answer for phrase in phrases)


def derive_annotation(row: Mapping[str, Any], annotator: str) -> Dict[str, Any]:
    require_fields(row, QUEUE_REQUIRED_FIELDS, "queue row")
    if bool_value(row.get("answer_contains_conflict_anchor_text")):
        tag = "conflict-following"
        evidence_note = "answer contains the conflict-anchor text exactly"
    elif bool_value(row.get("answer_contains_support_anchor_text")):
        tag = "support-following"
        evidence_note = "answer contains the support-anchor text exactly"
    elif has_non_gluing_phrase(str(row.get("answer_text") or "")):
        tag = "non-gluing"
        evidence_note = "answer declines or blocks a direct source merge"
    else:
        tag = "ambiguous"
        evidence_note = "no exact anchor text or non-gluing phrase rule matched"

    return {
        "queue_rank": int(row["queue_rank"]),
        "case_label": str(row["case_label"]),
        "cycle_id": str(row["cycle_id"]),
        "sample_id": str(row.get("sample_id") or ""),
        "candidate_side": str(row["candidate_side"]),
        "relation_kind_signature": str(row["relation_kind_signature"]),
        "source_facing_tag": tag,
        "evidence_note": evidence_note,
        "annotator": str(annotator),
        "annotation_mode": DERIVED_ANNOTATION_MODE,
        "answer_contains_support_anchor_text": bool_value(row.get("answer_contains_support_anchor_text")),
        "answer_contains_conflict_anchor_text": bool_value(row.get("answer_contains_conflict_anchor_text")),
    }


def load_supplied_annotations(path: Path) -> List[Dict[str, Any]]:
    rows = read_jsonl(path)
    for row in rows:
        require_fields(row, ANNOTATION_REQUIRED_FIELDS, "annotation row")
    return rows


def validate_annotations(
    *,
    queue_rows: Sequence[Mapping[str, Any]],
    annotation_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    queue_by_key = {row_key(row): row for row in queue_rows}
    annotation_by_key = {row_key(row): row for row in annotation_rows}
    if len(queue_by_key) != len(queue_rows):
        raise ValueError("queue rows contain duplicate queue_rank/case_label/cycle_id keys")
    if len(annotation_by_key) != len(annotation_rows):
        raise ValueError("annotation rows contain duplicate queue_rank/case_label/cycle_id keys")
    missing = sorted(set(queue_by_key) - set(annotation_by_key))
    extra = sorted(set(annotation_by_key) - set(queue_by_key))
    if missing:
        raise ValueError(f"annotation rows missing queue keys: {missing[:3]}")
    if extra:
        raise ValueError(f"annotation rows include unknown queue keys: {extra[:3]}")

    validated: List[Dict[str, Any]] = []
    for key in sorted(queue_by_key):
        queue_row = queue_by_key[key]
        annotation = annotation_by_key[key]
        tag = str(annotation["source_facing_tag"])
        if tag not in ALLOWED_SOURCE_FACING_TAGS:
            raise ValueError(f"unsupported source_facing_tag for {key}: {tag}")
        if str(annotation["candidate_side"]) != str(queue_row["candidate_side"]):
            raise ValueError(f"candidate_side mismatch for {key}")
        if str(annotation["relation_kind_signature"]) != str(queue_row["relation_kind_signature"]):
            raise ValueError(f"relation_kind_signature mismatch for {key}")

        validated.append(
            {
                "queue_rank": int(annotation["queue_rank"]),
                "case_label": str(annotation["case_label"]),
                "cycle_id": str(annotation["cycle_id"]),
                "sample_id": str(queue_row.get("sample_id") or annotation.get("sample_id") or ""),
                "candidate_side": str(annotation["candidate_side"]),
                "relation_kind_signature": str(annotation["relation_kind_signature"]),
                "source_facing_tag": tag,
                "evidence_note": str(annotation.get("evidence_note") or ""),
                "annotator": str(annotation.get("annotator") or ""),
                "annotation_mode": str(annotation.get("annotation_mode") or SUPPLIED_ANNOTATION_MODE),
                "answer_contains_support_anchor_text": bool_value(queue_row.get("answer_contains_support_anchor_text")),
                "answer_contains_conflict_anchor_text": bool_value(queue_row.get("answer_contains_conflict_anchor_text")),
            }
        )
    return validated


def build_summary_rows(annotation_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Counter[Tuple[str, str, str]] = Counter()
    for row in annotation_rows:
        grouped[
            (
                str(row["candidate_side"]),
                str(row["relation_kind_signature"]),
                str(row["source_facing_tag"]),
            )
        ] += 1
    return [
        {
            "candidate_side": candidate_side,
            "relation_kind_signature": relation_kind_signature,
            "source_facing_tag": source_facing_tag,
            "count": count,
        }
        for (candidate_side, relation_kind_signature, source_facing_tag), count in sorted(grouped.items())
    ]


def build_status(annotation_rows: Sequence[Mapping[str, Any]], summary_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    tag_counts = Counter(str(row["source_facing_tag"]) for row in annotation_rows)
    side_counts = Counter(str(row["candidate_side"]) for row in annotation_rows)
    high_conflict = sum(
        1
        for row in annotation_rows
        if row["candidate_side"] == "high_tension" and row["source_facing_tag"] == "conflict-following"
    )
    flat_support_or_non_gluing = sum(
        1
        for row in annotation_rows
        if row["candidate_side"] == "flat" and row["source_facing_tag"] in {"support-following", "non-gluing"}
    )
    return {
        "annotation_row_count": len(annotation_rows),
        "summary_row_count": len(summary_rows),
        "tag_counts": dict(sorted(tag_counts.items())),
        "candidate_side_counts": dict(sorted(side_counts.items())),
        "high_tension_conflict_following_count": high_conflict,
        "flat_support_or_non_gluing_count": flat_support_or_non_gluing,
    }


ANNOTATION_FIELDS = (
    "queue_rank",
    "case_label",
    "cycle_id",
    "sample_id",
    "candidate_side",
    "relation_kind_signature",
    "source_facing_tag",
    "evidence_note",
    "annotator",
    "annotation_mode",
    "answer_contains_support_anchor_text",
    "answer_contains_conflict_anchor_text",
)

SUMMARY_FIELDS = (
    "candidate_side",
    "relation_kind_signature",
    "source_facing_tag",
    "count",
)


def build_readme(
    *,
    status: Mapping[str, Any],
    summary_rows: Sequence[Mapping[str, Any]],
    annotation_mode: str,
) -> str:
    lines = [
        "# Gate12B Source-Facing Annotation",
        "",
        f"- annotation mode: `{annotation_mode}`",
        f"- rows: `{status['annotation_row_count']}`",
        f"- high-tension conflict-following rows: `{status['high_tension_conflict_following_count']}`",
        f"- flat support-or-non-gluing rows: `{status['flat_support_or_non_gluing_count']}`",
        "",
        "These tags describe the source-facing read of the queued row.",
        "They are not answer-quality labels.",
        "",
        "## Summary",
        "",
        "| candidate side | relation signature | source-facing tag | count |",
        "| --- | --- | --- | ---: |",
    ]
    for row in summary_rows:
        relation_signature = str(row["relation_kind_signature"]).replace("|", "\\|")
        lines.append(
            f"| `{row['candidate_side']}` | `{relation_signature}` | `{row['source_facing_tag']}` | {int(row['count'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_gate12b_source_annotation(
    *,
    queue_jsonl: Path,
    out_dir: Path,
    annotation_jsonl: Path | None = None,
    derive_from_queue: bool = False,
    annotator: str = "codex_source_surface_v1",
) -> Dict[str, Any]:
    queue_jsonl = Path(queue_jsonl)
    out_dir = Path(out_dir)
    input_paths = [queue_jsonl]
    if annotation_jsonl is not None:
        input_paths.append(Path(annotation_jsonl))
    validate_output_dir_boundary(out_dir, input_paths)

    queue_rows = read_jsonl(queue_jsonl)
    if not queue_rows:
        raise ValueError("queue_jsonl contains no rows")
    for row in queue_rows:
        require_fields(row, QUEUE_REQUIRED_FIELDS, "queue row")

    if annotation_jsonl is not None and derive_from_queue:
        raise ValueError("use either --annotation-jsonl or --derive-from-queue, not both")
    if annotation_jsonl is None and not derive_from_queue:
        raise ValueError("either --annotation-jsonl or --derive-from-queue is required")

    if derive_from_queue:
        annotation_mode = DERIVED_ANNOTATION_MODE
        annotation_rows = [derive_annotation(row, annotator=annotator) for row in queue_rows]
    else:
        annotation_mode = SUPPLIED_ANNOTATION_MODE
        annotation_rows = load_supplied_annotations(Path(annotation_jsonl))

    validated_rows = validate_annotations(queue_rows=queue_rows, annotation_rows=annotation_rows)
    summary_rows = build_summary_rows(validated_rows)
    status = build_status(validated_rows, summary_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    annotations_jsonl_path = out_dir / DEFAULT_ANNOTATIONS_JSONL
    annotations_csv_path = out_dir / DEFAULT_ANNOTATIONS_CSV
    summary_json_path = out_dir / DEFAULT_SUMMARY_JSON
    summary_csv_path = out_dir / DEFAULT_SUMMARY_CSV
    read_path = out_dir / DEFAULT_READ
    manifest_path = out_dir / DEFAULT_MANIFEST
    checksums_path = out_dir / DEFAULT_CHECKSUMS

    write_jsonl(annotations_jsonl_path, validated_rows)
    write_csv(annotations_csv_path, ANNOTATION_FIELDS, validated_rows)
    write_json(summary_json_path, {"status": status, "summary_rows": summary_rows})
    write_csv(summary_csv_path, SUMMARY_FIELDS, summary_rows)
    write_text(read_path, build_readme(status=status, summary_rows=summary_rows, annotation_mode=annotation_mode))

    manifest = {
        "run_id": out_dir.name,
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "code_git_commit": current_git_commit(),
        "builder_script_sha256": sha256_file(SCRIPT_PATH),
        "annotation_mode": annotation_mode,
        "source_queue_jsonl": repo_relative_or_posix(queue_jsonl),
        "source_annotation_jsonl": repo_relative_or_posix(annotation_jsonl) if annotation_jsonl else "",
        "allowed_source_facing_tags": list(ALLOWED_SOURCE_FACING_TAGS),
        "paths": {
            DEFAULT_ANNOTATIONS_JSONL: repo_relative_or_posix(annotations_jsonl_path),
            DEFAULT_ANNOTATIONS_CSV: repo_relative_or_posix(annotations_csv_path),
            DEFAULT_SUMMARY_JSON: repo_relative_or_posix(summary_json_path),
            DEFAULT_SUMMARY_CSV: repo_relative_or_posix(summary_csv_path),
            DEFAULT_READ: repo_relative_or_posix(read_path),
        },
        "status": status,
    }
    write_json(manifest_path, manifest)
    write_json(
        checksums_path,
        {
            DEFAULT_MANIFEST: sha256_file(manifest_path),
            DEFAULT_ANNOTATIONS_JSONL: sha256_file(annotations_jsonl_path),
            DEFAULT_ANNOTATIONS_CSV: sha256_file(annotations_csv_path),
            DEFAULT_SUMMARY_JSON: sha256_file(summary_json_path),
            DEFAULT_SUMMARY_CSV: sha256_file(summary_csv_path),
            DEFAULT_READ: sha256_file(read_path),
        },
    )
    return {"manifest": manifest, "status": status, "annotations": validated_rows, "summary_rows": summary_rows}


def main() -> int:
    args = parse_args()
    annotation_jsonl = Path(args.annotation_jsonl) if args.annotation_jsonl else None
    run_gate12b_source_annotation(
        queue_jsonl=Path(args.queue_jsonl),
        out_dir=Path(args.out_dir),
        annotation_jsonl=annotation_jsonl,
        derive_from_queue=bool(args.derive_from_queue),
        annotator=str(args.annotator),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
