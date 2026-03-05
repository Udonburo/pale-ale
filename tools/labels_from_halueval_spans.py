#!/usr/bin/env python3
"""Build step labels from halueval-spans with alignment verification and low-coverage fallback."""

import argparse
import difflib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from transformers import AutoTokenizer
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "transformers is required. Install with: python -m pip install transformers"
    ) from exc

REPO_ID = "llm-semantic-router/halueval-spans"
SPLIT = "train"
REVISION = "main"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create step labels from halueval spans with verification and fallback."
    )
    parser.add_argument("--input-jsonl", help="Optional local JSONL containing sample_id/answer/labels.")
    parser.add_argument("--sample-id", type=int, required=True)
    parser.add_argument("--triplets-ndjson", required=True)
    parser.add_argument("--tokenizer-model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--answer-file")
    parser.add_argument("--answer")
    parser.add_argument("--out", default="runs/triality_smoke/labels_step.jsonl")
    parser.add_argument("--min-coverage", type=float, default=0.35)
    parser.add_argument(
        "--no-improve-low-coverage",
        action="store_true",
        help="Disable fallback strategy when alignment coverage is below threshold.",
    )
    return parser.parse_args()


def load_triplets(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"non-object row at {path}:{line_no}")
            rows.append(row)
    rows = sorted(rows, key=lambda r: int(r.get("step", 0)))
    if not rows:
        raise ValueError(f"triplets file is empty: {path}")
    return rows


def load_row_from_jsonl(path: Path, sample_id: int) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                continue
            if int(row.get("sample_id", -1)) == sample_id:
                return row
    raise ValueError(f"sample_id={sample_id} not found in {path}")


def load_row_from_hf(sample_id: int) -> Dict[str, Any]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required for HF loading. Install with: python -m pip install datasets"
        ) from exc
    ds = load_dataset(REPO_ID, split=SPLIT, revision=REVISION)
    if sample_id < 0 or sample_id >= len(ds):
        raise ValueError(f"sample_id={sample_id} out of range for {REPO_ID}/{SPLIT}")
    row = ds[int(sample_id)]
    if not isinstance(row, dict):
        raise ValueError("dataset row is not an object")
    row["sample_id"] = int(sample_id)
    return row


def choose_answer_text(args: argparse.Namespace, row: Dict[str, Any]) -> str:
    if args.answer is not None:
        return str(args.answer)
    if args.answer_file is not None:
        return Path(args.answer_file).read_text(encoding="utf-8")
    for key in ("answer", "response", "output", "generated_answer", "model_answer"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    raise ValueError("could not find answer text; provide --answer or --answer-file")


def extract_span_payload(row: Dict[str, Any]) -> Any:
    for key in ("labels", "spans", "annotations"):
        if key in row:
            return row[key]
    return None


def find_all_occurrences(text: str, needle: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    if not needle:
        return out
    start = 0
    while True:
        idx = text.find(needle, start)
        if idx < 0:
            break
        out.append((idx, idx + len(needle)))
        start = idx + 1
    return out


def parse_single_span(item: Any, answer: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    if isinstance(item, dict):
        for k0, k1 in (
            ("start", "end"),
            ("answer_start", "answer_end"),
            ("offset_start", "offset_end"),
            ("char_start", "char_end"),
        ):
            if k0 in item and k1 in item:
                try:
                    s = int(item[k0])
                    e = int(item[k1])
                except Exception:
                    continue
                if 0 <= s < e <= len(answer):
                    out.append((s, e))
                return out
        for txt_key in ("text", "span", "label_text"):
            if txt_key in item and isinstance(item[txt_key], str):
                out.extend(find_all_occurrences(answer, item[txt_key]))
                if out:
                    return out
        return out
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        try:
            s = int(item[0])
            e = int(item[1])
        except Exception:
            return out
        if 0 <= s < e <= len(answer):
            out.append((s, e))
        return out
    if isinstance(item, str):
        out.extend(find_all_occurrences(answer, item))
        return out
    return out


def normalize_spans(payload: Any, answer: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    if payload is None:
        return out
    if isinstance(payload, list):
        for item in payload:
            out.extend(parse_single_span(item, answer))
    else:
        out.extend(parse_single_span(payload, answer))
    return sorted(set(out))


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


def conservative_labels_for_token_spans(
    token_spans: Sequence[Tuple[int, int]],
    positive_char_spans: Sequence[Tuple[int, int]],
) -> List[int]:
    labels = [0] * len(token_spans)
    for i, (ts, te) in enumerate(token_spans):
        if te <= ts:
            continue
        for ps, pe in positive_char_spans:
            if ts >= ps and te <= pe:
                labels[i] = 1
                break
    return labels


def merge_intervals(intervals: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    sorted_intervals = sorted(intervals)
    merged: List[Tuple[int, int]] = []
    cur_s, cur_e = sorted_intervals[0]
    for s, e in sorted_intervals[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def build_answer_positive_token_set(
    answer_token_spans: Sequence[Tuple[int, int]],
    answer_spans: Sequence[Tuple[int, int]],
) -> List[int]:
    idxs: List[int] = []
    for i, (ts, te) in enumerate(answer_token_spans):
        if te <= ts:
            continue
        for s, e in answer_spans:
            if ts >= s and te <= e:
                idxs.append(i)
                break
    return idxs


def project_by_token_sequence_alignment(
    answer_token_ids: Sequence[int],
    generated_token_ids: Sequence[int],
    answer_positive_token_idxs: Sequence[int],
) -> Dict[str, Any]:
    matcher = difflib.SequenceMatcher(
        a=list(answer_token_ids), b=list(generated_token_ids), autojunk=False
    )
    a_to_b: Dict[int, int] = {}
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            continue
        for offset in range(i2 - i1):
            a_to_b[i1 + offset] = j1 + offset

    out_labels = [0] * len(generated_token_ids)
    mapped_pos = 0
    for a_idx in answer_positive_token_idxs:
        b_idx = a_to_b.get(int(a_idx))
        if b_idx is not None and 0 <= b_idx < len(out_labels):
            out_labels[b_idx] = 1
            mapped_pos += 1

    total_pos = len(answer_positive_token_idxs)
    coverage = (mapped_pos / float(total_pos)) if total_pos > 0 else 0.0
    return {
        "labels": out_labels,
        "coverage": coverage,
        "mapped_positive_tokens": mapped_pos,
        "total_positive_tokens": total_pos,
        "equal_blocks": sum(1 for t, *_ in matcher.get_opcodes() if t == "equal"),
    }


def project_answer_spans_to_generated_char_intervals(
    answer_text: str,
    generated_text: str,
    answer_spans: Sequence[Tuple[int, int]],
) -> Dict[str, Any]:
    matcher = difflib.SequenceMatcher(a=answer_text, b=generated_text, autojunk=False)
    a_to_b_char: Dict[int, int] = {}
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            continue
        for offset in range(i2 - i1):
            a_to_b_char[i1 + offset] = j1 + offset

    mapped_intervals: List[Tuple[int, int]] = []
    total_span_chars = 0
    mapped_span_chars = 0
    for s, e in answer_spans:
        total_span_chars += max(0, e - s)
        mapped_chars = [a_to_b_char[i] for i in range(s, e) if i in a_to_b_char]
        mapped_span_chars += len(mapped_chars)
        if not mapped_chars:
            continue
        mapped_chars.sort()
        run_start = mapped_chars[0]
        prev = mapped_chars[0]
        for pos in mapped_chars[1:]:
            if pos == prev + 1:
                prev = pos
                continue
            mapped_intervals.append((run_start, prev + 1))
            run_start = pos
            prev = pos
        mapped_intervals.append((run_start, prev + 1))

    coverage = (mapped_span_chars / float(total_span_chars)) if total_span_chars > 0 else 0.0
    return {
        "intervals": merge_intervals(mapped_intervals),
        "coverage": coverage,
        "mapped_span_chars": mapped_span_chars,
        "total_span_chars": total_span_chars,
    }


def span_text_exact_match_intervals(
    answer_text: str,
    generated_text: str,
    answer_spans: Sequence[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    matches: List[Tuple[int, int]] = []
    for s, e in answer_spans:
        snippet = answer_text[s:e].strip()
        if len(snippet) < 3:
            continue
        matches.extend(find_all_occurrences(generated_text, snippet))
    return merge_intervals(matches)


def write_step_labels(path: Path, labels: Sequence[int], token_ids: Sequence[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for step, (label, token_id) in enumerate(zip(labels, token_ids)):
            row = {"step": int(step), "label": int(label), "token_id": int(token_id)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_meta(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text + "\n")


def main() -> int:
    args = parse_args()
    improve_low_coverage = not bool(args.no_improve_low_coverage)

    triplets_rows = load_triplets(Path(args.triplets_ndjson))
    generated_token_ids = [int(r["token_id"]) for r in triplets_rows]

    if args.input_jsonl:
        row = load_row_from_jsonl(Path(args.input_jsonl), sample_id=args.sample_id)
        source = "jsonl"
    else:
        row = load_row_from_hf(sample_id=args.sample_id)
        source = "hf"

    answer_text = choose_answer_text(args, row)
    span_payload = extract_span_payload(row)
    answer_spans = normalize_spans(span_payload, answer=answer_text)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model, use_fast=True)
    answer_token_ids = tokenizer.encode(answer_text, add_special_tokens=False)
    _, answer_token_spans = decode_tokens_with_spans(tokenizer, answer_token_ids)
    generated_text, generated_token_spans = decode_tokens_with_spans(tokenizer, generated_token_ids)

    answer_positive_token_idxs = build_answer_positive_token_set(answer_token_spans, answer_spans)

    # Stage 1: token-sequence alignment
    stage1 = project_by_token_sequence_alignment(
        answer_token_ids=answer_token_ids,
        generated_token_ids=generated_token_ids,
        answer_positive_token_idxs=answer_positive_token_idxs,
    )
    final_labels = list(stage1["labels"])
    final_coverage = float(stage1["coverage"])
    mode_used = "token_sequence_equal_blocks_v1"

    # Stage 2 fallback: char alignment + exact span-text matches
    stage2 = None
    if improve_low_coverage and final_coverage < float(args.min_coverage):
        stage2 = project_answer_spans_to_generated_char_intervals(
            answer_text=answer_text,
            generated_text=generated_text,
            answer_spans=answer_spans,
        )
        exact_intervals = span_text_exact_match_intervals(
            answer_text=answer_text,
            generated_text=generated_text,
            answer_spans=answer_spans,
        )
        fallback_intervals = merge_intervals(stage2["intervals"] + exact_intervals)
        fallback_labels = conservative_labels_for_token_spans(
            token_spans=generated_token_spans,
            positive_char_spans=fallback_intervals,
        )
        fallback_coverage = float(stage2["coverage"])

        # Improve if fallback has strictly better coverage or more positives
        if (fallback_coverage > final_coverage) or (
            sum(fallback_labels) > sum(final_labels)
        ):
            final_labels = fallback_labels
            final_coverage = max(final_coverage, fallback_coverage)
            mode_used = "char_alignment_plus_exact_span_match_v2"

    out_path = Path(args.out)
    write_step_labels(out_path, final_labels, generated_token_ids)
    meta_path = out_path.with_name(out_path.stem + "_meta.json")

    coverage_answer_chars = sum(max(0, e - s) for s, e in answer_spans)
    meta = {
        "source": source,
        "sample_id": int(args.sample_id),
        "tokenizer_model": args.tokenizer_model,
        "triplets_path": str(Path(args.triplets_ndjson).as_posix()),
        "labels_out": str(out_path.as_posix()),
        "labels_mode_used": mode_used,
        "min_coverage_threshold": float(args.min_coverage),
        "improve_low_coverage_enabled": improve_low_coverage,
        "n_triplet_steps": len(generated_token_ids),
        "n_answer_tokens": len(answer_token_ids),
        "n_answer_spans": len(answer_spans),
        "answer_positive_token_count": len(answer_positive_token_idxs),
        "stage1_coverage_token_alignment": float(stage1["coverage"]),
        "stage1_mapped_positive_tokens": int(stage1["mapped_positive_tokens"]),
        "stage1_total_positive_tokens": int(stage1["total_positive_tokens"]),
        "stage1_equal_blocks": int(stage1["equal_blocks"]),
        "final_alignment_coverage_ratio": float(final_coverage),
        "final_positive_steps": int(sum(1 for x in final_labels if x == 1)),
        "final_negative_steps": int(sum(1 for x in final_labels if x == 0)),
        "coverage_answer_positive_chars": int(coverage_answer_chars),
        "span_field_present": span_payload is not None,
        "note": (
            "Primary method uses token-sequence equal-block mapping. "
            "If low coverage, fallback uses answer->generated char alignment and exact span-text match."
        ),
    }
    if stage2 is not None:
        meta["stage2_char_alignment_coverage"] = float(stage2["coverage"])
        meta["stage2_mapped_span_chars"] = int(stage2["mapped_span_chars"])
        meta["stage2_total_span_chars"] = int(stage2["total_span_chars"])

    write_meta(meta_path, meta)
    print(f"Wrote labels: {out_path}")
    print(f"Wrote meta: {meta_path}")
    print(
        "Coverage final="
        f"{meta['final_alignment_coverage_ratio']:.6f} "
        f"(stage1={meta['stage1_coverage_token_alignment']:.6f}, "
        f"threshold={meta['min_coverage_threshold']:.6f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

