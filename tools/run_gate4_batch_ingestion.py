#!/usr/bin/env python3
"""One-shot batch ingestion: pack Gate4 input from sample dirs and run Gate4."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Sequence

import build_gate4_input as packer


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Gate4 batch input from existing sample dirs and run pale-ale gate4."
    )
    parser.add_argument("--samples-root", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--sample-ids", nargs="+", type=int)
    selection.add_argument("--sample-id-file")
    selection.add_argument("--all-samples", action="store_true")
    parser.add_argument("--variant", choices=("consistent", "frustrated", "unknown"))
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cfa-jsonl", default="data/cfa/cfa_v1.jsonl")
    parser.add_argument("--spec-path", default="SPEC.internal.draft.md")
    parser.add_argument("--out-dir", default="runs/gate4_batch_ingestion")
    parser.add_argument("--run-id", default="gate4_batch_ingestion")
    parser.add_argument("--dataset-revision-id", default="cfa_v1_batch_ingestion_v1")
    parser.add_argument("--evaluation-mode-id", default="supervised_v1")
    parser.add_argument("--perm-r", type=int, default=2000)
    parser.add_argument("--primary-score", default="E")
    parser.add_argument("--script-extract", default="tools/extract_triality_triplets.py")
    parser.add_argument("--script-eval", default="tools/eval_triality_token.py")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_command(cmd: Sequence[str], cwd: Path) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        list(cmd),
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed rc={completed.returncode}: {' '.join(cmd)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


def ensure_cli_built() -> Path:
    run_command(["cargo", "build", "-p", "pale-ale-cli"], cwd=REPO_ROOT)
    for candidate in (
        REPO_ROOT / "target" / "debug" / "pale-ale.exe",
        REPO_ROOT / "target" / "debug" / "pale-ale",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("pale-ale CLI binary not found after cargo build")


def compute_gate4_identity_hashes(
    cli_path: Path,
    cfa_jsonl: Path,
    sample_ids: Sequence[int],
    spec_path: Path,
) -> Dict[str, str]:
    cmd = [
        str(cli_path),
        "gate4",
        "hash-identity",
        "--cfa-jsonl",
        str(cfa_jsonl),
        "--sample-ids",
        *[str(sample_id) for sample_id in sample_ids],
        "--spec-path",
        str(spec_path),
        "--json",
    ]
    payload = json.loads(run_command(cmd, cwd=REPO_ROOT).stdout)
    data = payload.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("gate4 hash-identity returned missing data object")
    out: Dict[str, str] = {}
    for key in (
        "dataset_hash_blake3",
        "spec_hash_raw_blake3",
        "spec_hash_blake3",
    ):
        value = data.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise RuntimeError(f"gate4 hash-identity returned invalid {key}: {value!r}")
        out[key] = value
    return out


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text + "\n")


def load_cfa_row_map(path: Path) -> Dict[int, Dict[str, Any]]:
    rows = packer.load_jsonl(path)
    row_map: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        sample_id = int(row["sample_id"])
        if sample_id in row_map:
            raise ValueError(f"duplicate sample_id in CFA JSONL {path}: {sample_id}")
        row_map[sample_id] = row
    return row_map


def resolve_reference_path(reference: str, *, base_dir: Path) -> Path:
    candidate = Path(reference)
    if candidate.is_absolute():
        return candidate.resolve()
    return (base_dir / candidate).resolve()


def verify_sample_dir_provenance(
    *,
    samples_root: Path,
    cfa_jsonl: Path,
    sample_ids: Sequence[int],
) -> Dict[str, object]:
    cfa_row_map = load_cfa_row_map(cfa_jsonl)
    cfa_jsonl_resolved = cfa_jsonl.resolve()
    for sample_id in sample_ids:
        sample_dir = samples_root / packer.sample_dir_name(sample_id)
        row = cfa_row_map.get(sample_id)
        if row is None:
            raise ValueError(f"selected sample_id not found in CFA JSONL: {sample_id}")

        meta = packer.load_json(sample_dir / "meta.json")
        labels_meta = packer.load_json(sample_dir / "labels_meta.json")
        prompt_path = sample_dir / "prompt.txt"
        answer_path = sample_dir / "answer.txt"
        prompt_text = prompt_path.read_text(encoding="utf-8")
        answer_text = answer_path.read_text(encoding="utf-8")

        expected_prompt = str(row.get("prompt", ""))
        expected_answer = str(row.get("answer", ""))
        if prompt_text != expected_prompt:
            raise ValueError(
                f"prompt.txt mismatch for sample {sample_id}: {prompt_path}"
            )
        if answer_text != expected_answer:
            raise ValueError(
                f"answer.txt mismatch for sample {sample_id}: {answer_path}"
            )

        expected_prompt_sha = sha256_text(expected_prompt)
        expected_answer_sha = sha256_text(expected_answer)
        if str(meta.get("prompt_sha256") or "") != expected_prompt_sha:
            raise ValueError(
                f"prompt_sha256 mismatch for sample {sample_id}: "
                f"meta={meta.get('prompt_sha256')!r} expected={expected_prompt_sha}"
            )
        if str(meta.get("target_answer_sha256") or "") != expected_answer_sha:
            raise ValueError(
                f"target_answer_sha256 mismatch for sample {sample_id}: "
                f"meta={meta.get('target_answer_sha256')!r} expected={expected_answer_sha}"
            )

        if int(labels_meta.get("sample_id")) != sample_id:
            raise ValueError(
                f"labels_meta sample_id mismatch for sample {sample_id}: "
                f"{labels_meta.get('sample_id')!r}"
            )
        expected_variant = str(row.get("variant") or "unknown")
        actual_variant = str(labels_meta.get("variant") or "unknown")
        if actual_variant != expected_variant:
            raise ValueError(
                f"labels_meta variant mismatch for sample {sample_id}: "
                f"{actual_variant!r} != {expected_variant!r}"
            )

        expected_world_type = row.get("world_type")
        actual_world_type = labels_meta.get("world_type")
        if actual_world_type != expected_world_type:
            raise ValueError(
                f"labels_meta world_type mismatch for sample {sample_id}: "
                f"{actual_world_type!r} != {expected_world_type!r}"
            )

        labels_cfa_jsonl = labels_meta.get("cfa_jsonl")
        if not isinstance(labels_cfa_jsonl, str) or not labels_cfa_jsonl:
            raise ValueError(f"labels_meta cfa_jsonl missing for sample {sample_id}")
        resolved_labels_cfa_jsonl = resolve_reference_path(
            labels_cfa_jsonl, base_dir=REPO_ROOT
        )
        if resolved_labels_cfa_jsonl != cfa_jsonl_resolved:
            raise ValueError(
                f"labels_meta cfa_jsonl mismatch for sample {sample_id}: "
                f"{resolved_labels_cfa_jsonl} != {cfa_jsonl_resolved}"
            )

    return {
        "provenance_cross_check": "passed",
        "cfa_jsonl": cfa_jsonl.as_posix(),
        "n_checked_samples": len(sample_ids),
    }


def main() -> int:
    args = parse_args()
    samples_root = REPO_ROOT / args.samples_root
    cfa_jsonl = REPO_ROOT / args.cfa_jsonl
    spec_path = REPO_ROOT / args.spec_path
    out_dir = REPO_ROOT / args.out_dir
    gate4_input_path = out_dir / "gate4_input.json"
    selection_manifest_path = out_dir / "batch_selection_manifest.json"
    wrapper_manifest_path = out_dir / "batch_run_manifest.json"
    gate4_out_dir = out_dir / "gate4_out"

    selected_ids = packer.resolve_selected_sample_ids(
        samples_root=samples_root,
        sample_ids=args.sample_ids,
        sample_id_file=Path(args.sample_id_file) if args.sample_id_file else None,
        all_samples=bool(args.all_samples),
        variant=args.variant,
        offset=args.offset,
        limit=args.limit,
    )
    payload = packer.pack_gate4_input(
        samples_root=samples_root,
        sample_ids=selected_ids,
        script_extract=REPO_ROOT / args.script_extract,
        script_eval=REPO_ROOT / args.script_eval,
        perm_r=args.perm_r,
        primary_score=args.primary_score,
    )
    provenance_check = verify_sample_dir_provenance(
        samples_root=samples_root,
        cfa_jsonl=cfa_jsonl,
        sample_ids=selected_ids,
    )
    packer.write_json(gate4_input_path, payload)
    selection_manifest = packer.build_selection_manifest(
        samples_root=samples_root,
        sample_ids=selected_ids,
        variant_filter=args.variant,
        offset=args.offset,
        limit=args.limit,
        out_path=gate4_input_path,
    )
    write_json(selection_manifest_path, selection_manifest)

    cli_path = ensure_cli_built()
    identity_hashes = compute_gate4_identity_hashes(
        cli_path=cli_path,
        cfa_jsonl=cfa_jsonl,
        sample_ids=selected_ids,
        spec_path=spec_path,
    )

    gate4_cmd = [
        str(cli_path),
        "gate4",
        "run",
        "--input",
        str(gate4_input_path),
        "--out",
        str(gate4_out_dir),
        "--run-id",
        args.run_id,
        "--dataset-revision-id",
        args.dataset_revision_id,
        "--dataset-hash-blake3",
        identity_hashes["dataset_hash_blake3"],
        "--spec-hash-raw-blake3",
        identity_hashes["spec_hash_raw_blake3"],
        "--spec-hash-blake3",
        identity_hashes["spec_hash_blake3"],
        "--evaluation-mode-id",
        args.evaluation_mode_id,
    ]
    run_command(gate4_cmd, cwd=REPO_ROOT)

    wrapper_manifest = {
        "mode": "gate4_batch_ingestion_v1",
        "samples_root": samples_root.as_posix(),
        "cfa_jsonl": cfa_jsonl.as_posix(),
        "spec_path": spec_path.as_posix(),
        "run_id": args.run_id,
        "dataset_revision_id": args.dataset_revision_id,
        "evaluation_mode_id": args.evaluation_mode_id,
        "selection_manifest_json": selection_manifest_path.as_posix(),
        "gate4_input_json": gate4_input_path.as_posix(),
        "gate4_out_dir": gate4_out_dir.as_posix(),
        "n_samples": len(selected_ids),
        "sample_ids": [int(sample_id) for sample_id in selected_ids],
        "provenance_cross_check": provenance_check["provenance_cross_check"],
        "n_checked_samples": provenance_check["n_checked_samples"],
        "dataset_hash_blake3": identity_hashes["dataset_hash_blake3"],
        "spec_hash_raw_blake3": identity_hashes["spec_hash_raw_blake3"],
        "spec_hash_blake3": identity_hashes["spec_hash_blake3"],
        "input_json_sha256": sha256_file(gate4_input_path),
        "gate4_manifest_json": (gate4_out_dir / "manifest.json").as_posix(),
        "token_features_csv": (gate4_out_dir / "gate4_token_features.csv").as_posix(),
        "sample_summary_csv": (gate4_out_dir / "gate4_sample_summary.csv").as_posix(),
        "run_summary_csv": (gate4_out_dir / "gate4_run_summary.csv").as_posix(),
    }
    write_json(wrapper_manifest_path, wrapper_manifest)

    print(f"gate4_input_json={gate4_input_path.as_posix()}")
    print(f"batch_selection_manifest_json={selection_manifest_path.as_posix()}")
    print(f"batch_run_manifest_json={wrapper_manifest_path.as_posix()}")
    print(f"gate4_out_dir={gate4_out_dir.as_posix()}")
    print(f"n_samples={len(selected_ids)}")
    print(f"sample_ids={','.join(str(sample_id) for sample_id in selected_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
