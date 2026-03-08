#!/usr/bin/env python3
"""Build Gate4RunInputV1 JSON from existing triplets/labels sample directories."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pack sample triplets/labels outputs into Gate4RunInputV1."
    )
    parser.add_argument("--samples-root", required=True)
    parser.add_argument("--sample-ids", nargs="+", type=int, required=True)
    parser.add_argument("--out", required=True)
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
        "proj_id": str(first_meta["proj_id"]),
        "splus_def_id": str(first_meta["splus_def_id"]),
        "sminus_def_id": str(first_meta["sminus_def_id"]),
        "script_sha256_extract": sha256_file(script_extract),
        "script_sha256_eval": sha256_file(script_eval),
    }


def validate_homogeneous_metadata(
    sample_dirs: Sequence[Path],
    keys: Sequence[str],
) -> Dict[str, Any]:
    first_meta = load_json(sample_dirs[0] / "meta.json")
    first_view = {key: first_meta.get(key) for key in keys}
    for sample_dir in sample_dirs[1:]:
        current_meta = load_json(sample_dir / "meta.json")
        current_view = {key: current_meta.get(key) for key in keys}
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
    if not sample_ids:
        raise ValueError("sample_ids must be non-empty")
    sample_dirs = [samples_root / f"sample_{sample_id:06d}" for sample_id in sample_ids]
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
            for path, sample_id in zip(sample_dirs, sample_ids)
        ],
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text + "\n")


def main() -> int:
    args = parse_args()
    payload = pack_gate4_input(
        samples_root=Path(args.samples_root),
        sample_ids=args.sample_ids,
        script_extract=Path(args.script_extract),
        script_eval=Path(args.script_eval),
        perm_r=args.perm_r,
        primary_score=args.primary_score,
    )
    out_path = Path(args.out)
    write_json(out_path, payload)
    print(f"out={out_path.as_posix()}")
    print(f"n_samples={len(payload['samples'])}")
    print(f"input_sha256={sha256_file(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
