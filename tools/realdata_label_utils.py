#!/usr/bin/env python3
"""Shared helpers for label handling and label-count stats on realdata JSONL."""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def normalize_label(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"invalid boolean label: {value!r}")
    if isinstance(value, int) and value in (0, 1):
        return int(value)
    raise ValueError(f"label must be 0/1/null, got: {value!r}")


def count_labels_records(records: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    count0 = 0
    count1 = 0
    countnull = 0
    total = 0
    for row in records:
        total += 1
        label = normalize_label(row.get("label"))
        if label is None:
            countnull += 1
        elif label == 0:
            count0 += 1
        else:
            count1 += 1
    return {
        "total_rows": total,
        "count0": count0,
        "count1": count1,
        "countnull": countnull,
    }


def count_labels_jsonl(path: Path) -> Dict[str, int]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object row at {path}:{line_no}")
            records.append(row)
    return count_labels_records(records)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Count label distribution in JSONL.")
    parser.add_argument("input_jsonl")
    args = parser.parse_args()
    stats = count_labels_jsonl(Path(args.input_jsonl))
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))
