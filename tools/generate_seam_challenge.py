#!/usr/bin/env python3
"""Generate Seam Challenge Set v0 with deterministic clean/perturbed pairs."""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

GENERATOR_ID = "seam_challenge_v0"
JSONL_SCHEMA_ID = "seam_challenge_jsonl_v1"
META_SCHEMA_ID = "seam_challenge_meta_v1"
FAMILIES = (
    "punctuation_injection",
    "casing_perturbation",
    "spacing_perturbation",
    "harmless_fragmentation_trigger",
)

BASE_FACTS: Sequence[Dict[str, str]] = (
    {
        "topic": "loop closure",
        "prompt": "State the note exactly.",
        "answer": "The loop closure check stays stable during the dry run.",
    },
    {
        "topic": "token level",
        "prompt": "Repeat the report sentence exactly.",
        "answer": "The token level audit remains calm after the first pass.",
    },
    {
        "topic": "state space",
        "prompt": "Copy the field report sentence.",
        "answer": "The state space update remains local to the morning sample.",
    },
    {
        "topic": "cross check",
        "prompt": "Return the record sentence verbatim.",
        "answer": "The cross check note matches the archived answer.",
    },
    {
        "topic": "signal path",
        "prompt": "Emit the operator note exactly.",
        "answer": "The signal path record stays readable after the routine update.",
    },
    {
        "topic": "sample pair",
        "prompt": "Print the sample memo exactly.",
        "answer": "The sample pair summary remains consistent across the final review.",
    },
    {
        "topic": "local window",
        "prompt": "Provide the local memo exactly.",
        "answer": "The local window check remains quiet near the tagged segment.",
    },
    {
        "topic": "data flow",
        "prompt": "Copy the audit line exactly.",
        "answer": "The data flow report stays clear during the handoff step.",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic Seam Challenge Set v0 JSONL + meta JSON."
    )
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    parser.add_argument("--meta-out", required=True, help="Output metadata JSON path.")
    parser.add_argument("--n-pairs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def locate_once(text: str, needle: str) -> Tuple[int, int]:
    start = text.index(needle)
    return start, start + len(needle)


def punctuation_injection(text: str, rng: random.Random) -> Tuple[str, List[Dict[str, Any]]]:
    markers = ["dry run", "first pass", "morning sample", "final review", "handoff step"]
    candidates = [marker for marker in markers if marker in text]
    target = rng.choice(candidates) if candidates else text.split()[2]
    start, end = locate_once(text, target)
    out = text[:start] + "(" + target + ")" + text[end:]
    return out, [
        {
            "kind": "punctuation_wrap",
            "original": target,
            "replacement": f"({target})",
            "start": start,
            "end": start + len(target) + 2,
        }
    ]


def casing_perturbation(text: str, rng: random.Random) -> Tuple[str, List[Dict[str, Any]]]:
    words = text.split()
    idx = 0 if len(words) < 3 else rng.randrange(1, min(len(words), 4))
    original = words[idx]
    transformed = original.upper()
    words[idx] = transformed
    out = " ".join(words)
    start, end = locate_once(out, transformed)
    return out, [
        {
            "kind": "uppercase_token",
            "original": original,
            "replacement": transformed,
            "start": start,
            "end": end,
        }
    ]


def spacing_perturbation(text: str, rng: random.Random) -> Tuple[str, List[Dict[str, Any]]]:
    words = text.split(" ")
    gap_idx = rng.randrange(1, max(2, len(words) - 1))
    prefix = " ".join(words[:gap_idx])
    suffix = " ".join(words[gap_idx:])
    out = prefix + "  " + suffix
    start = len(prefix)
    return out, [
        {
            "kind": "double_space",
            "original": " ",
            "replacement": "  ",
            "start": start,
            "end": start + 2,
        }
    ]


def harmless_fragmentation_trigger(
    text: str, rng: random.Random
) -> Tuple[str, List[Dict[str, Any]]]:
    phrases = [
        "loop closure",
        "token level",
        "state space",
        "cross check",
        "signal path",
        "sample pair",
        "local window",
        "data flow",
    ]
    candidates = [phrase for phrase in phrases if phrase in text]
    if not candidates:
        raise ValueError(f"no fragmentation candidate in {text!r}")
    phrase = rng.choice(candidates)
    start, end = locate_once(text, phrase)
    replacement = phrase.replace(" ", "-")
    out = text[:start] + replacement + text[end:]
    return out, [
        {
            "kind": "hyphenate_phrase",
            "original": phrase,
            "replacement": replacement,
            "start": start,
            "end": start + len(replacement),
        }
    ]


def apply_perturbation(
    family: str, text: str, rng: random.Random
) -> Tuple[str, List[Dict[str, Any]]]:
    if family == "punctuation_injection":
        return punctuation_injection(text, rng)
    if family == "casing_perturbation":
        return casing_perturbation(text, rng)
    if family == "spacing_perturbation":
        return spacing_perturbation(text, rng)
    if family == "harmless_fragmentation_trigger":
        return harmless_fragmentation_trigger(text, rng)
    raise ValueError(f"unsupported family: {family}")


def build_pair_rows(pair_id: int, fact: Dict[str, str], family: str, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random((seed * 1000003) + pair_id)
    clean_sample_id = pair_id * 2
    perturbed_sample_id = clean_sample_id + 1
    perturbed_answer, spans = apply_perturbation(family, fact["answer"], rng)
    clean = {
        "sample_id": clean_sample_id,
        "pair_id": pair_id,
        "source_sample_id": None,
        "challenge_class": "clean_consistent",
        "variant": "consistent",
        "world_type": GENERATOR_ID,
        "contrast_sample_id": perturbed_sample_id,
        "prompt": fact["prompt"],
        "answer": fact["answer"],
        "defect_spans": [],
        "perturbation_family": "none",
        "perturbation_spans": [],
        "seed": seed,
        "topic": fact["topic"],
    }
    perturbed = {
        "sample_id": perturbed_sample_id,
        "pair_id": pair_id,
        "source_sample_id": clean_sample_id,
        "challenge_class": "seam_perturbed_consistent",
        "variant": "consistent",
        "world_type": GENERATOR_ID,
        "contrast_sample_id": clean_sample_id,
        "prompt": fact["prompt"],
        "answer": perturbed_answer,
        "defect_spans": [],
        "perturbation_family": family,
        "perturbation_spans": spans,
        "seed": seed,
        "topic": fact["topic"],
    }
    return [clean, perturbed]


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n")


def main() -> int:
    args = parse_args()
    if args.n_pairs <= 0:
        raise SystemExit("--n-pairs must be > 0")

    rows: List[Dict[str, Any]] = []
    family_counts = {family: 0 for family in FAMILIES}
    for pair_id in range(args.n_pairs):
        fact = BASE_FACTS[pair_id % len(BASE_FACTS)]
        family = FAMILIES[pair_id % len(FAMILIES)]
        family_counts[family] += 1
        rows.extend(build_pair_rows(pair_id=pair_id, fact=fact, family=family, seed=args.seed))

    meta = {
        "generator_id": GENERATOR_ID,
        "jsonl_schema_id": JSONL_SCHEMA_ID,
        "meta_schema_id": META_SCHEMA_ID,
        "seed": args.seed,
        "n_pairs": args.n_pairs,
        "n_rows": len(rows),
        "families": list(FAMILIES),
        "family_counts": family_counts,
        "class_counts": {
            "clean_consistent": args.n_pairs,
            "seam_perturbed_consistent": args.n_pairs,
        },
        "notes": [
            "Deterministic paired seam challenge set.",
            "Synonym substitution is intentionally excluded in v0.",
        ],
    }

    write_jsonl(Path(args.out), rows)
    write_json(Path(args.meta_out), meta)
    print(Path(args.out).as_posix())
    print(Path(args.meta_out).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
