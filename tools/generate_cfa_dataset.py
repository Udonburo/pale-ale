#!/usr/bin/env python3
"""Generate a deterministic Constraint Frustration Arena (CFA) dataset."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


RELATION_TYPES = ("genealogy", "temporal", "reachability")
ENTITY_POOL = (
    "Aster",
    "Beryl",
    "Cedar",
    "Dover",
    "Ember",
    "Frost",
    "Grove",
    "Haven",
    "Ivory",
    "Jasper",
    "Kappa",
    "Lumen",
    "Mirth",
    "Noble",
    "Opalx",
    "Prism",
    "Quill",
    "Riven",
    "Solis",
    "Thorn",
    "Umber",
    "Vivid",
    "Waltz",
    "Xenon",
    "Yarrow",
    "Zorin",
)


@dataclass(frozen=True)
class WorldInstance:
    world_type: str
    prompt: str
    answer_consistent: str
    answer_frustrated: str
    defect_span_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate closed-world consistent/frustrated text pairs with deterministic "
            "defect span labels."
        )
    )
    parser.add_argument("--out", default="data/cfa/cfa_v1.jsonl")
    parser.add_argument("--meta-out", default="data/cfa/cfa_v1_meta.json")
    parser.add_argument("--n-worlds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--world-types",
        default="genealogy,temporal,reachability",
        help="Comma-separated subset of: genealogy, temporal, reachability",
    )
    parser.add_argument("--emit-consistent", type=int, choices=(0, 1), default=1)
    parser.add_argument("--emit-frustrated", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()

    if args.n_worlds <= 0:
        parser.error("--n-worlds must be > 0")
    if args.emit_consistent == 0 and args.emit_frustrated == 0:
        parser.error("At least one of --emit-consistent or --emit-frustrated must be 1")
    return args


def build_entities(seed: int, world_id: int) -> Tuple[str, str, str]:
    n = len(ENTITY_POOL)
    i0 = (seed + world_id * 3) % n
    i1 = (seed + world_id * 3 + 7) % n
    i2 = (seed + world_id * 3 + 13) % n
    return ENTITY_POOL[i0], ENTITY_POOL[i1], ENTITY_POOL[i2]


def make_genealogy(a: str, b: str, c: str) -> WorldInstance:
    prompt = (
        "Closed world facts:\n"
        f"1) {a} is the parent of {b}.\n"
        f"2) {b} is the parent of {c}.\n"
        "Question: Which ancestor relation follows?\n"
        "Answer in one short paragraph."
    )
    claim_ok = f"{a} is an ancestor of {c}"
    claim_bad = f"{c} is an ancestor of {a}"
    suffix = "This follows by transitivity of the parent relation."
    return WorldInstance(
        world_type="genealogy",
        prompt=prompt,
        answer_consistent=f"Given these facts, {claim_ok}. {suffix}",
        answer_frustrated=f"Given these facts, {claim_bad}. {suffix}",
        defect_span_text=claim_bad,
    )


def make_temporal(a: str, b: str, c: str) -> WorldInstance:
    prompt = (
        "Closed world facts:\n"
        f"1) Event {a} happened before event {b}.\n"
        f"2) Event {b} happened before event {c}.\n"
        "Question: Which order relation follows?\n"
        "Answer in one short paragraph."
    )
    claim_ok = f"event {a} happened before event {c}"
    claim_bad = f"event {c} happened before event {a}"
    suffix = "This follows by transitivity of the before relation."
    return WorldInstance(
        world_type="temporal",
        prompt=prompt,
        answer_consistent=f"Given these facts, {claim_ok}. {suffix}",
        answer_frustrated=f"Given these facts, {claim_bad}. {suffix}",
        defect_span_text=claim_bad,
    )


def make_reachability(a: str, b: str, c: str) -> WorldInstance:
    prompt = (
        "Closed world facts:\n"
        f"1) There is a directed edge from {a} to {b}.\n"
        f"2) There is a directed edge from {b} to {c}.\n"
        "Question: Which path relation follows?\n"
        "Answer in one short paragraph."
    )
    claim_ok = f"a directed path exists from {a} to {c}"
    claim_bad = f"a directed path exists from {c} to {a}"
    suffix = "This follows by transitivity of path reachability."
    return WorldInstance(
        world_type="reachability",
        prompt=prompt,
        answer_consistent=f"Given these facts, {claim_ok}. {suffix}",
        answer_frustrated=f"Given these facts, {claim_bad}. {suffix}",
        defect_span_text=claim_bad,
    )


def build_world(world_type: str, a: str, b: str, c: str) -> WorldInstance:
    if world_type == "genealogy":
        return make_genealogy(a, b, c)
    if world_type == "temporal":
        return make_temporal(a, b, c)
    if world_type == "reachability":
        return make_reachability(a, b, c)
    raise ValueError(f"Unsupported world_type: {world_type}")


def find_span_or_fail(text: str, needle: str) -> Dict[str, object]:
    start = text.find(needle)
    if start < 0:
        raise ValueError(f"Could not find defect span text in answer: {needle!r}")
    end = start + len(needle)
    return {
        "start": start,
        "end": end,
        "label": "frustration_defect_v1",
        "text": needle,
    }


def parse_world_types(raw: str) -> Sequence[str]:
    out: List[str] = []
    for token in raw.split(","):
        t = token.strip().lower()
        if not t:
            continue
        if t not in RELATION_TYPES:
            raise ValueError(
                f"Unknown world type: {t}. Allowed: {', '.join(RELATION_TYPES)}"
            )
        out.append(t)
    if not out:
        raise ValueError("No valid world types selected")
    return out


def write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def write_json(path: Path, obj: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(payload + "\n")


def main() -> int:
    args = parse_args()
    world_types = parse_world_types(args.world_types)

    rows: List[Dict[str, object]] = []
    sample_id = 0
    world_counts = {t: 0 for t in RELATION_TYPES}
    variant_counts = {"consistent": 0, "frustrated": 0}

    for world_id in range(args.n_worlds):
        world_type = world_types[world_id % len(world_types)]
        a, b, c = build_entities(seed=args.seed, world_id=world_id)
        w = build_world(world_type, a, b, c)
        world_counts[w.world_type] += 1

        pair_ids: Dict[str, int] = {}

        if args.emit_consistent == 1:
            row = {
                "sample_id": int(sample_id),
                "world_id": int(world_id),
                "world_type": w.world_type,
                "variant": "consistent",
                "has_defect": 0,
                "prompt": w.prompt,
                "answer": w.answer_consistent,
                "defect_spans": [],
            }
            rows.append(row)
            pair_ids["consistent"] = sample_id
            sample_id += 1
            variant_counts["consistent"] += 1

        if args.emit_frustrated == 1:
            span = find_span_or_fail(w.answer_frustrated, w.defect_span_text)
            row = {
                "sample_id": int(sample_id),
                "world_id": int(world_id),
                "world_type": w.world_type,
                "variant": "frustrated",
                "has_defect": 1,
                "prompt": w.prompt,
                "answer": w.answer_frustrated,
                "defect_spans": [span],
            }
            rows.append(row)
            pair_ids["frustrated"] = sample_id
            sample_id += 1
            variant_counts["frustrated"] += 1

        if "consistent" in pair_ids and "frustrated" in pair_ids:
            rows[-1]["contrast_sample_id"] = pair_ids["consistent"]
            rows[-2]["contrast_sample_id"] = pair_ids["frustrated"]

    out_path = Path(args.out)
    meta_path = Path(args.meta_out)
    write_jsonl(out_path, rows)

    meta = {
        "dataset_id": "cfa_v1",
        "n_worlds": int(args.n_worlds),
        "n_rows": len(rows),
        "seed": int(args.seed),
        "world_types_selected": list(world_types),
        "emit_consistent": int(args.emit_consistent),
        "emit_frustrated": int(args.emit_frustrated),
        "world_type_counts": world_counts,
        "variant_counts": variant_counts,
        "out_jsonl": out_path.as_posix(),
    }
    write_json(meta_path, meta)

    print(f"Wrote dataset: {out_path.as_posix()}")
    print(f"Wrote meta: {meta_path.as_posix()}")
    print(f"Rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
