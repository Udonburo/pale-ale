#!/usr/bin/env python3
"""Create deterministic token-step labels from CFA defect spans."""

import argparse
import difflib
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create step labels for a CFA sample.")
    parser.add_argument("--cfa-jsonl", default="data/cfa/cfa_v1.jsonl")
    parser.add_argument("--sample-id", type=int, required=True)
    parser.add_argument("--triplets-ndjson", required=True)
    parser.add_argument("--tokenizer-model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--answer-file")
    parser.add_argument("--out", default="runs/cfa/labels_step.jsonl")
    parser.add_argument("--min-coverage", type=float, default=0.95)
    parser.add_argument("--fail-below-coverage", action="store_true")
    parser.add_argument(
        "--tokenizer-local-files-only",
        type=int,
        choices=(0, 1),
        default=0,
        help="When using tokenizer fallback, require local cache only.",
    )
    return parser.parse_args()


def load_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                raise ValueError(f"non-object row at {path}:{line_no}")
            rows.append(obj)
    return rows


def load_tokenizer(tokenizer_model: str, local_files_only: bool) -> Any:
    try:
        from transformers import AutoTokenizer  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "transformers is required for tokenizer fallback mode. "
            "Install with: python -m pip install transformers"
        ) from exc
    return AutoTokenizer.from_pretrained(
        tokenizer_model,
        use_fast=True,
        local_files_only=local_files_only,
    )


def find_sample(path: Path, sample_id: int) -> Dict[str, Any]:
    for row in load_jsonl_rows(path):
        if int(row.get("sample_id", -1)) == sample_id:
            return row
    raise ValueError(f"sample_id={sample_id} not found in {path}")


def load_triplets(path: Path) -> List[Dict[str, Any]]:
    rows = load_jsonl_rows(path)
    if not rows:
        raise ValueError(f"empty triplets NDJSON: {path}")
    rows.sort(key=lambda r: int(r.get("step", 0)))
    return rows


def decode_tokens_with_spans(tokenizer: Any, token_ids: Sequence[int]) -> Tuple[str, List[Tuple[int, int]]]:
    text_parts: List[str] = []
    spans: List[Tuple[int, int]] = []
    pos = 0
    for token_id in token_ids:
        piece = tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        text_parts.append(piece)
        start = pos
        pos += len(piece)
        spans.append((start, pos))
    return "".join(text_parts), spans


def normalize_spans(spans_raw: Any, answer_len: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    if not isinstance(spans_raw, list):
        return out
    for item in spans_raw:
        if not isinstance(item, dict):
            continue
        if "start" not in item or "end" not in item:
            continue
        try:
            s = int(item["start"])
            e = int(item["end"])
        except Exception:
            continue
        if 0 <= s < e <= answer_len:
            out.append((s, e))
    return sorted(set(out))


def positive_token_indices(
    token_spans: Sequence[Tuple[int, int]], defect_spans: Sequence[Tuple[int, int]]
) -> List[int]:
    idxs: List[int] = []
    for i, (ts, te) in enumerate(token_spans):
        if te <= ts:
            continue
        for ds, de in defect_spans:
            if ts >= ds and te <= de:
                idxs.append(i)
                break
    return idxs


def map_answer_tokens_to_generated(
    answer_token_ids: Sequence[int],
    generated_token_ids: Sequence[int],
    positive_answer_idxs: Sequence[int],
) -> Dict[str, Any]:
    matcher = difflib.SequenceMatcher(
        a=list(answer_token_ids), b=list(generated_token_ids), autojunk=False
    )
    a_to_b: Dict[int, int] = {}
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            continue
        for off in range(i2 - i1):
            a_to_b[i1 + off] = j1 + off

    labels = [0] * len(generated_token_ids)
    mapped = 0
    for a_idx in positive_answer_idxs:
        b_idx = a_to_b.get(int(a_idx))
        if b_idx is not None and 0 <= b_idx < len(labels):
            labels[b_idx] = 1
            mapped += 1

    total = len(positive_answer_idxs)
    coverage = (mapped / float(total)) if total > 0 else 1.0
    return {
        "labels": labels,
        "coverage": coverage,
        "mapped_positive_tokens": mapped,
        "total_positive_tokens": total,
        "equal_blocks": sum(1 for t, *_ in matcher.get_opcodes() if t == "equal"),
        "mode": "token_id_sequence_alignment_v1",
    }


def map_using_triplet_char_offsets(
    triplets: Sequence[Dict[str, Any]], defect_spans: Sequence[Tuple[int, int]]
) -> Dict[str, Any]:
    labels: List[int] = [0] * len(triplets)
    positive_triplet_idxs: List[int] = []
    for i, row in enumerate(triplets):
        if "answer_char_start" not in row or "answer_char_end" not in row:
            raise ValueError("triplet row missing answer_char_start/end")
        ts = int(row["answer_char_start"])
        te = int(row["answer_char_end"])
        if te <= ts:
            continue
        for ds, de in defect_spans:
            if ts >= ds and te <= de:
                labels[i] = 1
                positive_triplet_idxs.append(i)
                break

    total = len(positive_triplet_idxs)
    return {
        "labels": labels,
        "coverage": 1.0,
        "mapped_positive_tokens": total,
        "total_positive_tokens": total,
        "equal_blocks": len(triplets),
        "mode": "triplet_char_offsets_v1",
    }


def write_labels_jsonl(path: Path, labels: Sequence[int], token_ids: Sequence[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for step, (label, token_id) in enumerate(zip(labels, token_ids)):
            row = {"step": int(step), "label": int(label), "token_id": int(token_id)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_meta_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(payload, ensure_ascii=False, indent=2)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(txt + "\n")


def main() -> int:
    args = parse_args()
    cfa_path = Path(args.cfa_jsonl)
    triplets_path = Path(args.triplets_ndjson)
    out_path = Path(args.out)

    sample = find_sample(cfa_path, sample_id=args.sample_id)
    triplets = load_triplets(triplets_path)
    generated_token_ids = [int(r["token_id"]) for r in triplets]

    answer = (
        Path(args.answer_file).read_text(encoding="utf-8")
        if args.answer_file
        else str(sample.get("answer", ""))
    )
    if not answer:
        raise ValueError("empty answer text")

    defect_spans = normalize_spans(sample.get("defect_spans", []), answer_len=len(answer))

    use_direct_offsets = all(
        ("answer_char_start" in row and "answer_char_end" in row) for row in triplets
    )

    if use_direct_offsets:
        mapped = map_using_triplet_char_offsets(triplets, defect_spans)
        answer_token_ids: List[int] = []
        positive_answer_idxs: List[int] = []
    else:
        tokenizer = load_tokenizer(
            tokenizer_model=args.tokenizer_model,
            local_files_only=bool(args.tokenizer_local_files_only),
        )
        answer_token_ids = tokenizer.encode(answer, add_special_tokens=False)
        _, answer_token_spans = decode_tokens_with_spans(tokenizer, answer_token_ids)
        positive_answer_idxs = positive_token_indices(answer_token_spans, defect_spans)
        mapped = map_answer_tokens_to_generated(
            answer_token_ids=answer_token_ids,
            generated_token_ids=generated_token_ids,
            positive_answer_idxs=positive_answer_idxs,
        )

    labels = mapped["labels"]
    coverage = float(mapped["coverage"])
    if args.fail_below_coverage and coverage < float(args.min_coverage):
        raise RuntimeError(
            f"Coverage below threshold: coverage={coverage:.6f} < min_coverage={args.min_coverage:.6f}"
        )

    write_labels_jsonl(out_path, labels=labels, token_ids=generated_token_ids)
    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    meta = {
        "label_source": "cfa_defect_spans_v1",
        "cfa_jsonl": cfa_path.as_posix(),
        "sample_id": int(args.sample_id),
        "variant": sample.get("variant"),
        "world_type": sample.get("world_type"),
        "has_defect": int(sample.get("has_defect", 0)),
        "triplets_path": triplets_path.as_posix(),
        "tokenizer_model": args.tokenizer_model,
        "tokenizer_local_files_only": bool(args.tokenizer_local_files_only),
        "label_mapping_mode": str(mapped.get("mode")),
        "n_triplet_steps": len(generated_token_ids),
        "n_answer_tokens": len(answer_token_ids),
        "n_defect_spans": len(defect_spans),
        "answer_positive_token_count": len(positive_answer_idxs),
        "mapped_positive_tokens": int(mapped["mapped_positive_tokens"]),
        "total_positive_tokens": int(mapped["total_positive_tokens"]),
        "equal_blocks": int(mapped["equal_blocks"]),
        "final_alignment_coverage_ratio": coverage,
        "min_coverage_threshold": float(args.min_coverage),
        "fail_below_coverage": bool(args.fail_below_coverage),
        "final_positive_steps": int(sum(1 for x in labels if x == 1)),
        "final_negative_steps": int(sum(1 for x in labels if x == 0)),
        "labels_out": out_path.as_posix(),
    }
    write_meta_json(meta_path, meta)

    print(f"Wrote labels: {out_path.as_posix()}")
    print(f"Wrote meta: {meta_path.as_posix()}")
    print(f"Coverage: {coverage:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
