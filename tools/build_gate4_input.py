#!/usr/bin/env python3
"""Build Gate4RunInputV1 JSON from existing triplets/labels sample directories."""

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_PROJ_ID = "fwht_pad_pow2_take8_v1"
DEFAULT_SPLUS_DEF_ID = "attn_lastlayer_weighted_hidden_v1"
DEFAULT_SMINUS_DEF_ID = "lm_head_row_expectation_topk128_v1"
SAMPLE_DIR_RE = re.compile(r"^sample_(\d{6,})$")


def normalize_meta_value(meta: Dict[str, Any], key: str) -> Any:
    if key == "proj_id":
        return meta.get(key) or DEFAULT_PROJ_ID
    if key == "splus_def_id":
        return meta.get(key) or DEFAULT_SPLUS_DEF_ID
    if key == "sminus_def_id":
        return meta.get(key) or DEFAULT_SMINUS_DEF_ID
    return meta.get(key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pack sample triplets/labels outputs into Gate4RunInputV1."
    )
    parser.add_argument("--samples-root", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--sample-ids", nargs="+", type=int)
    selection.add_argument("--sample-id-file")
    selection.add_argument("--all-samples", action="store_true")
    parser.add_argument(
        "--variant",
        choices=("consistent", "frustrated", "unknown"),
        help="Optional variant filter applied after sample discovery.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip the first N selected samples after filtering.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Keep at most N samples after filtering and offset.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--selection-manifest-out")
    parser.add_argument("--perm-r", type=int, default=2000)
    parser.add_argument("--primary-score", default="E")
    parser.add_argument("--script-extract", default="tools/extract_triality_triplets.py")
    parser.add_argument("--script-eval", default="tools/eval_triality_token.py")
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_json(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object: {path}")
    return obj


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
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


def parse_label01(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("bool is not a valid label")
    if isinstance(value, int):
        if value in (0, 1):
            return value
        raise ValueError(f"invalid integer label {value}")
    raw = str(value).strip()
    if raw in ("0", "1"):
        return int(raw)
    raise ValueError(f"invalid label value {value!r}")


def load_step_labels(path: Path, n_steps: int) -> List[int]:
    labels = [0] * n_steps
    seen = [False] * n_steps
    rows = load_jsonl(path)
    for row in rows:
        if "step" not in row:
            raise ValueError(f"unsupported label row without step in {path}")
        step = int(row["step"])
        if step < 0 or step >= n_steps:
            raise ValueError(f"label step out of range in {path}: {step}")
        labels[step] = parse_label01(row.get("label"))
        seen[step] = True
    missing = [idx for idx, flag in enumerate(seen) if not flag]
    if missing:
        raise ValueError(f"missing labels for steps {missing[:8]} in {path}")
    return labels


def normalize_sample_ids(sample_ids: Iterable[int]) -> List[int]:
    normalized = sorted(int(sample_id) for sample_id in sample_ids)
    if not normalized:
        raise ValueError("sample_ids must be non-empty")
    deduped: List[int] = []
    seen = set()
    for sample_id in normalized:
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id in selection: {sample_id}")
        seen.add(sample_id)
        deduped.append(sample_id)
    return deduped


def parse_sample_id_file(path: Path) -> List[int]:
    sample_ids: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            try:
                sample_ids.append(int(raw))
            except ValueError as err:
                raise ValueError(
                    f"invalid sample_id at {path}:{line_no}: {raw!r}"
                ) from err
    return normalize_sample_ids(sample_ids)


def sample_dir_name(sample_id: int) -> str:
    return f"sample_{sample_id:06d}"


def discover_sample_dirs(samples_root: Path) -> List[Tuple[int, Path]]:
    discovered: List[Tuple[int, Path]] = []
    for child in samples_root.iterdir():
        if not child.is_dir():
            continue
        match = SAMPLE_DIR_RE.match(child.name)
        if not match:
            continue
        discovered.append((int(match.group(1)), child))
    discovered.sort(key=lambda item: item[0])
    if not discovered:
        raise ValueError(f"no sample_* directories found under {samples_root}")
    return discovered


def load_variant_for_sample(sample_dir: Path) -> str:
    labels_meta = load_json(sample_dir / "labels_meta.json")
    return str(labels_meta.get("variant") or "unknown")


def resolve_selected_sample_ids(
    samples_root: Path,
    sample_ids: Optional[Sequence[int]] = None,
    sample_id_file: Optional[Path] = None,
    all_samples: bool = False,
    variant: Optional[str] = None,
    offset: int = 0,
    limit: Optional[int] = None,
) -> List[int]:
    if offset < 0:
        raise ValueError("offset must be >= 0")
    if limit is not None and limit < 0:
        raise ValueError("limit must be >= 0")

    if sample_ids is not None:
        selected = normalize_sample_ids(sample_ids)
    elif sample_id_file is not None:
        selected = parse_sample_id_file(sample_id_file)
    elif all_samples:
        selected = [sample_id for sample_id, _ in discover_sample_dirs(samples_root)]
    else:
        raise ValueError("one of sample_ids, sample_id_file, or all_samples is required")

    sample_dir_map = {sample_id: path for sample_id, path in discover_sample_dirs(samples_root)}
    missing_dirs = [sample_id for sample_id in selected if sample_id not in sample_dir_map]
    if missing_dirs:
        raise ValueError(
            f"missing sample directories under {samples_root}: {missing_dirs[:8]}"
        )

    if variant is not None:
        filtered: List[int] = []
        for sample_id in selected:
            current_variant = load_variant_for_sample(sample_dir_map[sample_id])
            if current_variant == variant:
                filtered.append(sample_id)
        selected = filtered

    selected = selected[offset:]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("selection is empty after applying filters")
    return selected


def build_selection_manifest(
    samples_root: Path,
    sample_ids: Sequence[int],
    variant_filter: Optional[str],
    offset: int,
    limit: Optional[int],
    out_path: Path,
) -> Dict[str, Any]:
    variant_counts: Dict[str, int] = {}
    for sample_id in sample_ids:
        variant = load_variant_for_sample(samples_root / sample_dir_name(sample_id))
        variant_counts[variant] = variant_counts.get(variant, 0) + 1
    return {
        "samples_root": samples_root.as_posix(),
        "selection_mode": "batch_packer_v1",
        "variant_filter": variant_filter,
        "offset": offset,
        "limit": limit,
        "n_samples": len(sample_ids),
        "sample_ids": [int(sample_id) for sample_id in sample_ids],
        "variant_counts": variant_counts,
        "out_path": out_path.as_posix(),
    }


def build_sample_payload(sample_dir: Path, sample_id: int) -> Dict[str, Any]:
    triplets_path = sample_dir / "triplets.ndjson"
    meta_path = sample_dir / "meta.json"
    labels_path = sample_dir / "labels.jsonl"
    labels_meta_path = sample_dir / "labels_meta.json"

    triplets = load_jsonl(triplets_path)
    if not triplets:
        raise ValueError(f"empty triplets: {triplets_path}")
    triplets.sort(key=lambda row: int(row["step"]))
    expected_steps = list(range(len(triplets)))
    actual_steps = [int(row["step"]) for row in triplets]
    if actual_steps != expected_steps:
        raise ValueError(
            f"triplets steps for sample {sample_id} are not contiguous 0..N-1: {actual_steps[:16]}"
        )

    meta = load_json(meta_path)
    labels_meta = load_json(labels_meta_path)
    labels = load_step_labels(labels_path, n_steps=len(triplets))
    triplets_file_sha256 = sha256_file(triplets_path)
    meta_triplets_sha256 = str(meta.get("output_ndjson_sha256") or "")
    if triplets_file_sha256 != meta_triplets_sha256:
        raise ValueError(
            f"triplets SHA mismatch for sample {sample_id}: "
            f"file={triplets_file_sha256} meta={meta_triplets_sha256}"
        )

    token_steps: List[Dict[str, Any]] = []
    for idx, row in enumerate(triplets):
        token_steps.append(
            {
                "step": int(row["step"]),
                "absolute_pos": int(row["absolute_pos"]),
                "answer_char_start": row.get("answer_char_start"),
                "answer_char_end": row.get("answer_char_end"),
                "token_id": int(row["token_id"]),
                "token_str": str(row["token_str"]),
                "label_token": int(labels[idx]),
                "defect_span_id": None,
                "V_8d": [float(x) for x in row["V_8d"]],
                "Splus_8d": [float(x) for x in row["Splus_8d"]],
                "Sminus_8d": [float(x) for x in row["Sminus_8d"]],
                "baseline_logprob": float(row["baseline_logprob"]),
                "baseline_entropy": float(row["baseline_entropy"]),
            }
        )

    return {
        "sample_id": int(sample_id),
        "variant": str(labels_meta.get("variant") or "unknown"),
        "world_type": labels_meta.get("world_type"),
        "exact_token_match_ratio": float(meta["exact_token_match_ratio"]),
        "label_coverage_ratio": float(labels_meta["final_alignment_coverage_ratio"]),
        "triplets_sha256": triplets_file_sha256,
        "labels_sha256": sha256_file(labels_path),
        "token_steps": token_steps,
    }


def build_metadata(
    first_meta: Dict[str, Any],
    script_extract: Path,
    script_eval: Path,
    perm_r: int,
    primary_score: str,
) -> Dict[str, Any]:
    return {
        "model_id": str(first_meta["model_id"]),
        "model_revision": str(first_meta.get("model_revision") or ""),
        "seed": int(first_meta["seed"]),
        "perm_r": int(perm_r),
        "primary_score": str(primary_score),
        "proj_id": str(normalize_meta_value(first_meta, "proj_id")),
        "splus_def_id": str(normalize_meta_value(first_meta, "splus_def_id")),
        "sminus_def_id": str(normalize_meta_value(first_meta, "sminus_def_id")),
        "script_sha256_extract": sha256_file(script_extract),
        "script_sha256_eval": sha256_file(script_eval),
    }


def validate_homogeneous_metadata(
    sample_dirs: Sequence[Path],
    keys: Sequence[str],
) -> Dict[str, Any]:
    first_meta = load_json(sample_dirs[0] / "meta.json")
    first_view = {key: normalize_meta_value(first_meta, key) for key in keys}
    for sample_dir in sample_dirs[1:]:
        current_meta = load_json(sample_dir / "meta.json")
        current_view = {key: normalize_meta_value(current_meta, key) for key in keys}
        if current_view != first_view:
            raise ValueError(
                f"sample metadata mismatch between {sample_dirs[0] / 'meta.json'} and "
                f"{sample_dir / 'meta.json'} for keys={list(keys)}: "
                f"expected={first_view!r} actual={current_view!r}"
            )
    return first_meta


def pack_gate4_input(
    samples_root: Path,
    sample_ids: Sequence[int],
    script_extract: Path,
    script_eval: Path,
    perm_r: int,
    primary_score: str,
) -> Dict[str, Any]:
    normalized_sample_ids = normalize_sample_ids(sample_ids)
    sample_dirs = [samples_root / sample_dir_name(sample_id) for sample_id in normalized_sample_ids]
    first_meta = validate_homogeneous_metadata(
        sample_dirs,
        keys=(
            "model_id",
            "model_revision",
            "seed",
            "proj_id",
            "splus_def_id",
            "sminus_def_id",
        ),
    )
    return {
        "metadata": build_metadata(
            first_meta=first_meta,
            script_extract=script_extract,
            script_eval=script_eval,
            perm_r=perm_r,
            primary_score=primary_score,
        ),
        "samples": [
            build_sample_payload(path, sample_id)
            for path, sample_id in zip(sample_dirs, normalized_sample_ids)
        ],
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text + "\n")


def main() -> int:
    args = parse_args()
    samples_root = Path(args.samples_root)
    selected_ids = resolve_selected_sample_ids(
        samples_root=samples_root,
        sample_ids=args.sample_ids,
        sample_id_file=Path(args.sample_id_file) if args.sample_id_file else None,
        all_samples=bool(args.all_samples),
        variant=args.variant,
        offset=args.offset,
        limit=args.limit,
    )
    payload = pack_gate4_input(
        samples_root=samples_root,
        sample_ids=selected_ids,
        script_extract=Path(args.script_extract),
        script_eval=Path(args.script_eval),
        perm_r=args.perm_r,
        primary_score=args.primary_score,
    )
    out_path = Path(args.out)
    write_json(out_path, payload)

    if args.selection_manifest_out:
        selection_manifest = build_selection_manifest(
            samples_root=samples_root,
            sample_ids=selected_ids,
            variant_filter=args.variant,
            offset=args.offset,
            limit=args.limit,
            out_path=out_path,
        )
        write_json(Path(args.selection_manifest_out), selection_manifest)

    print(f"out={out_path.as_posix()}")
    print(f"n_samples={len(payload['samples'])}")
    print(f"sample_ids={','.join(str(sample_id) for sample_id in selected_ids)}")
    print(f"input_sha256={sha256_file(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
