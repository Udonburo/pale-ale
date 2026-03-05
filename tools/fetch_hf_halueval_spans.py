#!/usr/bin/env python3
"""Fetch halueval-spans from HuggingFace and write Gate2-ready JSONL fields."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

REPO_ID = "llm-semantic-router/halueval-spans"
SPLIT = "train"
DEFAULT_OUT = "data/realdata/halueval_spans_train.jsonl"
DEFAULT_PROVENANCE = "data/realdata/hf_halueval_provenance.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch HF halueval-spans and convert to local JSONL."
    )
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--revision", default="main")
    return parser.parse_args()


def now_iso_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def resolve_revision_sha(repo_id: str, revision: str) -> Dict[str, Any]:
    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        return {
            "revision_resolved_sha": None,
            "revision_resolved_note": f"huggingface_hub unavailable: {exc}",
        }

    try:
        info = HfApi().dataset_info(repo_id=repo_id, revision=revision)
        sha = getattr(info, "sha", None)
        return {
            "revision_resolved_sha": sha if isinstance(sha, str) and sha else None,
            "revision_resolved_note": (
                None if isinstance(sha, str) and sha else "dataset_info returned empty sha"
            ),
        }
    except Exception as exc:
        return {
            "revision_resolved_sha": None,
            "revision_resolved_note": f"failed to resolve sha: {exc}",
        }


def select_answer_field(columns: Iterable[str], first_row: Dict[str, Any]) -> str:
    preferred = ("answer", "response", "output", "generated_answer", "model_answer")
    for key in preferred:
        if key in columns and isinstance(first_row.get(key), str):
            return key
    for key in columns:
        if isinstance(first_row.get(key), str):
            return key
    raise ValueError(f"could not find string answer field in columns={list(columns)}")


def select_labels_field(columns: Iterable[str], first_row: Dict[str, Any]) -> Optional[str]:
    preferred = ("labels", "spans", "annotations")
    for key in preferred:
        if key in columns:
            return key
    for key in columns:
        value = first_row.get(key)
        if isinstance(value, list):
            return key
    return None


def derive_label(row: Dict[str, Any], labels_field: Optional[str]) -> Optional[int]:
    if labels_field is None:
        return None
    value = row.get(labels_field)
    if isinstance(value, list):
        return 1 if len(value) > 0 else 0
    if value is None:
        return 0
    # Keep null if schema is unexpected.
    return None


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text + "\n")


def main() -> int:
    args = parse_args()
    if args.n <= 0:
        raise ValueError("--n must be > 0")

    try:
        import datasets
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required. Install with: python -m pip install datasets"
        ) from exc

    ds = load_dataset(REPO_ID, split=SPLIT, revision=args.revision)
    if len(ds) == 0:
        raise ValueError("dataset is empty")

    columns = list(ds.column_names)
    first_row = ds[0]
    if not isinstance(first_row, dict):
        raise ValueError("dataset row is not a mapping")

    answer_field = select_answer_field(columns, first_row)
    labels_field = select_labels_field(columns, first_row)

    out_rows = []
    # Deterministic selection: take first N rows in dataset order.
    for idx, row in enumerate(ds):
        answer = row.get(answer_field)
        if not isinstance(answer, str) or not answer.strip():
            continue
        out_rows.append(
            {
                "sample_id": int(idx),
                "answer": answer,
                "label": derive_label(row, labels_field),
            }
        )
        if len(out_rows) >= args.n:
            break

    if not out_rows:
        raise ValueError("no rows with non-empty answer were found")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    sha_info = resolve_revision_sha(REPO_ID, args.revision)
    provenance = {
        "repo_id": REPO_ID,
        "split": SPLIT,
        "revision_requested": args.revision,
        "revision_resolved_sha": sha_info["revision_resolved_sha"],
        "revision_resolved_note": sha_info["revision_resolved_note"],
        "n_rows_written": len(out_rows),
        "timestamp": now_iso_utc(),
        "datasets_version": getattr(datasets, "__version__", None),
        "columns": columns,
        "answer_field": answer_field,
        "labels_field": labels_field,
        "seed": args.seed,
        "selection_strategy": "first_n_dataset_order",
        "output_jsonl": str(out_path.as_posix()),
    }
    write_json(Path(DEFAULT_PROVENANCE), provenance)

    print(f"Fetched dataset: {REPO_ID} split={SPLIT} revision={args.revision}")
    print(f"Answer field: {answer_field}")
    print(f"Labels field: {labels_field}")
    print(f"Wrote rows: {len(out_rows)} -> {out_path}")
    print(f"Wrote provenance: {DEFAULT_PROVENANCE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
