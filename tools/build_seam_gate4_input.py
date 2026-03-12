#!/usr/bin/env python3
"""Build Gate4RunInputV1 from Seam Challenge rows via teacher forcing extraction."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import build_gate4_input as packer
import extract_triality_triplets as extractor


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = "runs/seam_gate5"
DEFAULT_GATE4_INPUT = "gate4_input.json"
DEFAULT_SELECTION_MANIFEST = "seam_selection_manifest.json"
DEFAULT_BUILD_MANIFEST = "seam_build_manifest.json"
EXPECTED_PAIR_CLASSES = ("clean_consistent", "seam_perturbed_consistent")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract deterministic Seam sample dirs and pack them into Gate4RunInputV1."
        )
    )
    parser.add_argument("--seam-jsonl", required=True)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--model-id", help="Optional explicit HF model id.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--emit-native-raw",
        action="store_true",
        help="Preserve raw native V/Splus/Sminus vectors in triplets.ndjson for boundary experiments.",
    )
    parser.add_argument("--perm-r", type=int, default=2000)
    parser.add_argument("--primary-score", default="E")
    parser.add_argument("--sample-ids", nargs="+", type=int)
    parser.add_argument("--sample-id-min", type=int)
    parser.add_argument("--sample-id-max", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--allow-incomplete-pairs",
        action="store_true",
        help="Allow subset selections that do not preserve full clean/perturbed pairs.",
    )
    parser.add_argument("--script-extract", default="tools/extract_triality_triplets.py")
    parser.add_argument("--script-eval", default="tools/eval_triality_token.py")
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
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


def repo_relative_or_posix(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    sample_id = int(row["sample_id"])
    prompt = str(row.get("prompt", ""))
    answer = str(row.get("answer", ""))
    if not prompt:
        raise ValueError(f"sample_id={sample_id} has empty prompt")
    if not answer:
        raise ValueError(f"sample_id={sample_id} has empty answer")
    return {
        "sample_id": sample_id,
        "prompt": prompt,
        "answer": answer,
        "variant": str(row.get("variant") or "consistent"),
        "world_type": row.get("world_type"),
        "challenge_class": row.get("challenge_class"),
        "pair_id": row.get("pair_id"),
        "source_sample_id": row.get("source_sample_id"),
        "contrast_sample_id": row.get("contrast_sample_id"),
        "perturbation_family": row.get("perturbation_family"),
        "perturbation_spans": row.get("perturbation_spans", []),
        "topic": row.get("topic"),
        "seed": row.get("seed"),
    }


def select_rows(
    rows: Sequence[Dict[str, Any]],
    sample_ids: Optional[Sequence[int]],
    sample_id_min: Optional[int],
    sample_id_max: Optional[int],
    max_samples: Optional[int],
    allow_incomplete_pairs: bool,
) -> List[Dict[str, Any]]:
    wanted = set(int(sample_id) for sample_id in sample_ids) if sample_ids else None
    out: List[Dict[str, Any]] = []
    seen = set()
    for raw in rows:
        row = normalize_row(raw)
        sample_id = int(row["sample_id"])
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id in seam JSONL: {sample_id}")
        seen.add(sample_id)
        if wanted is not None and sample_id not in wanted:
            continue
        if sample_id_min is not None and sample_id < sample_id_min:
            continue
        if sample_id_max is not None and sample_id > sample_id_max:
            continue
        out.append(row)
    out.sort(key=lambda row: int(row["sample_id"]))
    if wanted is not None:
        missing = sorted(wanted - {int(row["sample_id"]) for row in out})
        if missing:
            raise ValueError(f"requested sample_ids missing from seam JSONL: {missing}")
    if max_samples is not None:
        out = out[:max_samples]
    if not out:
        raise ValueError("no seam rows selected after filters")
    if not allow_incomplete_pairs:
        validate_complete_pairs(out)
    return out


def validate_complete_pairs(rows: Sequence[Dict[str, Any]]) -> None:
    pair_to_classes: Dict[int, set[str]] = {}
    pair_to_sample_ids: Dict[int, List[int]] = {}
    for row in rows:
        pair_id_raw = row.get("pair_id")
        if pair_id_raw is None:
            raise ValueError(
                f"selected seam row missing pair_id: sample_id={row.get('sample_id')}"
            )
        pair_id = int(pair_id_raw)
        challenge_class = str(row.get("challenge_class") or "")
        pair_to_classes.setdefault(pair_id, set()).add(challenge_class)
        pair_to_sample_ids.setdefault(pair_id, []).append(int(row["sample_id"]))

    expected = set(EXPECTED_PAIR_CLASSES)
    incomplete: List[str] = []
    for pair_id in sorted(pair_to_classes):
        actual = pair_to_classes[pair_id]
        if actual != expected:
            incomplete.append(
                f"pair_id={pair_id} sample_ids={sorted(pair_to_sample_ids[pair_id])} "
                f"classes={sorted(actual)} expected={sorted(expected)}"
            )
    if incomplete:
        preview = "; ".join(incomplete[:8])
        raise ValueError(
            "selected seam subset breaks paired quietness contract; "
            "use full clean/perturbed pairs or pass --allow-incomplete-pairs. "
            f"bad_pairs={preview}"
        )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_zero_labels(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            payload = {
                "step": int(row["step"]),
                "label": 0,
                "token_id": int(row["token_id"]),
            }
            f.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


def build_labels_meta(
    seam_jsonl: Path,
    sample_dir: Path,
    row: Dict[str, Any],
    n_steps: int,
) -> Dict[str, Any]:
    labels_path = sample_dir / "labels.jsonl"
    triplets_path = sample_dir / "triplets.ndjson"
    return {
        "label_source": "seam_consistent_zero_labels_v1",
        "seam_jsonl": repo_relative_or_posix(seam_jsonl),
        "sample_id": int(row["sample_id"]),
        "variant": row["variant"],
        "world_type": row["world_type"],
        "challenge_class": row["challenge_class"],
        "pair_id": row["pair_id"],
        "source_sample_id": row["source_sample_id"],
        "contrast_sample_id": row["contrast_sample_id"],
        "topic": row["topic"],
        "perturbation_family": row["perturbation_family"],
        "perturbation_spans": row["perturbation_spans"],
        "triplets_path": repo_relative_or_posix(triplets_path),
        "label_mapping_mode": "all_zero_consistent_v1",
        "n_triplet_steps": n_steps,
        "n_defect_spans": 0,
        "mapped_positive_tokens": 0,
        "total_positive_tokens": 0,
        "equal_blocks": n_steps,
        "final_alignment_coverage_ratio": 1.0,
        "min_coverage_threshold": 1.0,
        "fail_below_coverage": False,
        "final_positive_steps": 0,
        "final_negative_steps": n_steps,
        "labels_out": repo_relative_or_posix(labels_path),
    }


def build_sample_manifest(
    row: Dict[str, Any],
    sample_dir: Path,
    n_steps: int,
) -> Dict[str, Any]:
    return {
        "mode": "seam_sample_teacher_forcing_v1",
        "sample_id": int(row["sample_id"]),
        "pair_id": row["pair_id"],
        "source_sample_id": row["source_sample_id"],
        "contrast_sample_id": row["contrast_sample_id"],
        "challenge_class": row["challenge_class"],
        "variant": row["variant"],
        "world_type": row["world_type"],
        "topic": row["topic"],
        "perturbation_family": row["perturbation_family"],
        "n_steps": n_steps,
        "sample_dir": repo_relative_or_posix(sample_dir),
    }


def main() -> int:
    args = parse_args()
    seam_jsonl = (REPO_ROOT / args.seam_jsonl).resolve()
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    samples_root = out_dir / "samples"
    gate4_input_path = out_dir / DEFAULT_GATE4_INPUT
    selection_manifest_path = out_dir / DEFAULT_SELECTION_MANIFEST
    build_manifest_path = out_dir / DEFAULT_BUILD_MANIFEST

    seam_rows = select_rows(
        rows=read_jsonl(seam_jsonl),
        sample_ids=args.sample_ids,
        sample_id_min=args.sample_id_min,
        sample_id_max=args.sample_id_max,
        max_samples=args.max_samples,
        allow_incomplete_pairs=bool(args.allow_incomplete_pairs),
    )

    extractor.configure_reproducibility(args.seed, deterministic=True)
    device = extractor.resolve_device(args.device)
    model_id, tokenizer, model, model_revision = extractor.load_first_available_model(
        model_candidates=extractor.build_model_candidates(args.model_id),
        device=device,
    )

    sample_ids: List[int] = []
    challenge_class_counts: Dict[str, int] = {}
    family_counts: Dict[str, int] = {}
    for row in seam_rows:
        sample_id = int(row["sample_id"])
        sample_dir = samples_root / packer.sample_dir_name(sample_id)
        prompt_path = sample_dir / "prompt.txt"
        answer_path = sample_dir / "answer.txt"
        triplets_path = sample_dir / "triplets.ndjson"
        labels_path = sample_dir / "labels.jsonl"
        labels_meta_path = sample_dir / "labels_meta.json"
        sample_manifest_path = sample_dir / "sample_manifest.json"

        write_text(prompt_path, str(row["prompt"]))
        write_text(answer_path, str(row["answer"]))

        result_rows, result_meta = extractor.run_teacher_forcing_extraction(
            prompt=str(row["prompt"]),
            target_answer=str(row["answer"]),
            model=model,
            tokenizer=tokenizer,
            device=device,
            topk=args.topk,
            emit_native_raw=bool(args.emit_native_raw),
        )
        effective_topk = int(result_meta["topk_effective"])
        mode_details = result_meta["mode_details"]
        ndjson_sha256 = extractor.write_ndjson(triplets_path, result_rows)
        meta_payload = {
            "model_id": model_id,
            "model_revision": model_revision,
            "transformers_version": extractor.transformers.__version__,
            "torch_version": extractor.torch.__version__,
            "seed": args.seed,
            "topk_requested": args.topk,
            "topk_effective": effective_topk,
            "max_new_tokens": 128,
            "proj_id": extractor.PROJ_ID,
            "splus_def_id": extractor.SPLUS_DEF_ID,
            "sminus_def_id": extractor.SMINUS_DEF_ID_TEMPLATE.format(topk=effective_topk),
            "native_raw_emitted": bool(args.emit_native_raw),
            "native_raw_schema_id": (
                extractor.RAW_NATIVE_SCHEMA_ID if args.emit_native_raw else None
            ),
            "prompt_sha256": extractor.sha256_bytes(str(row["prompt"]).encode("utf-8")),
            "target_answer_sha256": extractor.sha256_bytes(str(row["answer"]).encode("utf-8")),
            "output_ndjson_sha256": ndjson_sha256,
            "output_ndjson_path": repo_relative_or_posix(triplets_path),
            "device": str(device),
            "dtype": "float32",
            "deterministic_requested": True,
            "n_steps_written": len(result_rows),
            "extraction_mode": mode_details["mode"],
            "alignment_method": mode_details.get("alignment_method"),
            "target_token_count_expected": mode_details.get("target_token_count_expected"),
            "target_token_count_extracted": mode_details.get("target_token_count_extracted"),
            "exact_token_match_ratio": mode_details.get("exact_token_match_ratio"),
            "bos_prepended_for_teacher_forcing": mode_details.get(
                "bos_prepended_for_teacher_forcing"
            ),
            "answer_char_start": mode_details.get("answer_char_start"),
            "target_token_indices_count": mode_details.get("target_token_indices_count"),
            "target_only_token_count": mode_details.get("target_only_token_count"),
            "boundary_merge_token_delta": mode_details.get("boundary_merge_token_delta"),
        }
        extractor.write_meta_json(sample_dir / "meta.json", meta_payload)
        write_zero_labels(labels_path, result_rows)
        write_json(
            labels_meta_path,
            build_labels_meta(
                seam_jsonl=seam_jsonl,
                sample_dir=sample_dir,
                row=row,
                n_steps=len(result_rows),
            ),
        )
        write_json(
            sample_manifest_path,
            build_sample_manifest(row=row, sample_dir=sample_dir, n_steps=len(result_rows)),
        )

        sample_ids.append(sample_id)
        challenge_class = str(row["challenge_class"] or "unknown")
        family = str(row["perturbation_family"] or "unknown")
        challenge_class_counts[challenge_class] = challenge_class_counts.get(challenge_class, 0) + 1
        family_counts[family] = family_counts.get(family, 0) + 1

    payload = packer.pack_gate4_input(
        samples_root=samples_root,
        sample_ids=sample_ids,
        script_extract=REPO_ROOT / args.script_extract,
        script_eval=REPO_ROOT / args.script_eval,
        perm_r=args.perm_r,
        primary_score=args.primary_score,
    )
    packer.write_json(gate4_input_path, payload)
    selection_manifest = {
        "selection_mode": "seam_gate4_builder_v1",
        "seam_jsonl": repo_relative_or_posix(seam_jsonl),
        "n_samples": len(sample_ids),
        "sample_ids": [int(sample_id) for sample_id in sample_ids],
        "allow_incomplete_pairs": bool(args.allow_incomplete_pairs),
        "challenge_class_counts": challenge_class_counts,
        "family_counts": family_counts,
        "out_path": repo_relative_or_posix(gate4_input_path),
    }
    packer.write_json(selection_manifest_path, selection_manifest)
    build_manifest = {
        "mode": "seam_gate4_builder_v1",
        "seam_jsonl": repo_relative_or_posix(seam_jsonl),
        "out_dir": repo_relative_or_posix(out_dir),
        "samples_root": repo_relative_or_posix(samples_root),
        "gate4_input_json": repo_relative_or_posix(gate4_input_path),
        "selection_manifest_json": repo_relative_or_posix(selection_manifest_path),
        "n_samples": len(sample_ids),
        "sample_ids": [int(sample_id) for sample_id in sample_ids],
        "allow_incomplete_pairs": bool(args.allow_incomplete_pairs),
        "model_id": model_id,
        "model_revision": model_revision,
        "seed": args.seed,
        "topk_requested": args.topk,
        "emit_native_raw": bool(args.emit_native_raw),
        "device": str(device),
    }
    write_json(build_manifest_path, build_manifest)

    print(f"gate4_input_json={repo_relative_or_posix(gate4_input_path)}")
    print(f"selection_manifest_json={repo_relative_or_posix(selection_manifest_path)}")
    print(f"build_manifest_json={repo_relative_or_posix(build_manifest_path)}")
    print(f"n_samples={len(sample_ids)}")
    print(f"sample_ids={','.join(str(sample_id) for sample_id in sample_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
