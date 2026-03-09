#!/usr/bin/env python3
"""Run small CFA Gate4 validation smoke: extract -> labels -> pack -> gate4 -> parity."""

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import build_gate4_input as packer
import extract_triality_triplets as extractor
import labels_from_cfa_spans as cfa_labels


REPO_ROOT = Path(__file__).resolve().parents[1]
MIN_TOKEN_MATCH = 0.98
MIN_COVERAGE = 0.30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gate4 parity/smoke validation on small CFA subset.")
    parser.add_argument("--cfa-jsonl", default="data/cfa/cfa_v1.jsonl")
    parser.add_argument("--out-dir", default="runs/gate4_validation_smoke")
    parser.add_argument("--attest-dir", default="attestations/triality/gate4_validation")
    parser.add_argument("--model-id", help="Optional model override")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--n-consistent", type=int, default=2)
    parser.add_argument("--n-frustrated", type=int, default=2)
    parser.add_argument("--run-id", default="gate4_validation_smoke")
    parser.add_argument("--perm-r", type=int, default=2000)
    parser.add_argument("--primary-score", default="E")
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def load_cfa_rows(path: Path) -> List[Dict[str, Any]]:
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
    rows.sort(key=lambda row: int(row["sample_id"]))
    return rows


def select_samples(rows: Sequence[Dict[str, Any]], n_consistent: int, n_frustrated: int) -> List[Dict[str, Any]]:
    consistent = [row for row in rows if str(row.get("variant")) == "consistent"][:n_consistent]
    frustrated = [row for row in rows if str(row.get("variant")) == "frustrated"][:n_frustrated]
    if len(consistent) < n_consistent or len(frustrated) < n_frustrated:
        raise ValueError(
            f"insufficient CFA rows: consistent={len(consistent)}/{n_consistent}, "
            f"frustrated={len(frustrated)}/{n_frustrated}"
        )
    selected = consistent + frustrated
    selected.sort(key=lambda row: int(row["sample_id"]))
    return selected


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


def compare_bytes(path_a: Path, path_b: Path) -> bool:
    return path_a.read_bytes() == path_b.read_bytes()


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
    completed = run_command(cmd, cwd=REPO_ROOT)
    payload = json.loads(completed.stdout)
    data = payload.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("gate4 hash-identity returned missing data object")
    expected_keys = [
        "dataset_hash_blake3",
        "spec_hash_raw_blake3",
        "spec_hash_blake3",
    ]
    out: Dict[str, str] = {}
    for key in expected_keys:
        value = data.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise RuntimeError(f"gate4 hash-identity returned invalid {key}: {value!r}")
        out[key] = value
    return out


def write_report(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line.rstrip("\n") + "\n")


def parse_key_value_report(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            out[key] = value
    return out


def generate_sample_outputs(
    selected: Sequence[Dict[str, Any]],
    cfa_path: Path,
    samples_root: Path,
    model_id: str,
    model_revision: str,
    tokenizer: Any,
    model: Any,
    device: Any,
    seed: int,
    topk: int,
) -> List[int]:
    sample_ids: List[int] = []
    for row in selected:
        sample_id = int(row["sample_id"])
        sample_ids.append(sample_id)
        sample_dir = samples_root / f"sample_{sample_id:06d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        prompt = str(row.get("prompt", ""))
        answer = str(row.get("answer", ""))
        if not prompt or not answer:
            raise ValueError(f"sample {sample_id} has empty prompt/answer")

        write_text(sample_dir / "prompt.txt", prompt)
        write_text(sample_dir / "answer.txt", answer)

        triplet_rows, triplet_meta = extractor.run_teacher_forcing_extraction(
            prompt=prompt,
            target_answer=answer,
            model=model,
            tokenizer=tokenizer,
            device=device,
            topk=topk,
        )
        mode_details = triplet_meta["mode_details"]
        ratio = float(mode_details.get("exact_token_match_ratio") or 0.0)
        if ratio < MIN_TOKEN_MATCH:
            raise RuntimeError(
                f"sample {sample_id} exact_token_match_ratio={ratio:.6f} < {MIN_TOKEN_MATCH:.2f}"
            )

        triplets_path = sample_dir / "triplets.ndjson"
        meta_path = sample_dir / "meta.json"
        labels_path = sample_dir / "labels.jsonl"
        labels_meta_path = sample_dir / "labels_meta.json"

        ndjson_sha = extractor.write_ndjson(triplets_path, triplet_rows)
        extractor.write_meta_json(
            meta_path,
            {
                "model_id": model_id,
                "model_revision": model_revision,
                "seed": seed,
                "topk_requested": topk,
                "topk_effective": int(triplet_meta["topk_effective"]),
                "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
                "target_answer_sha256": sha256_bytes(answer.encode("utf-8")),
                "output_ndjson_sha256": ndjson_sha,
                "output_ndjson_path": triplets_path.as_posix(),
                "device": str(device),
                "deterministic_requested": True,
                "n_steps_written": len(triplet_rows),
                "extraction_mode": mode_details.get("mode"),
                "alignment_method": mode_details.get("alignment_method"),
                "target_token_count_expected": mode_details.get("target_token_count_expected"),
                "target_token_count_extracted": mode_details.get("target_token_count_extracted"),
                "exact_token_match_ratio": mode_details.get("exact_token_match_ratio"),
                "target_token_indices_count": mode_details.get("target_token_indices_count"),
                "target_only_token_count": mode_details.get("target_only_token_count"),
                "boundary_merge_token_delta": mode_details.get("boundary_merge_token_delta"),
                "bos_prepended_for_teacher_forcing": mode_details.get(
                    "bos_prepended_for_teacher_forcing"
                ),
                "proj_id": extractor.PROJ_ID,
                "splus_def_id": extractor.SPLUS_DEF_ID,
                "sminus_def_id": extractor.SMINUS_DEF_ID_TEMPLATE.format(
                    topk=int(triplet_meta["topk_effective"])
                ),
            },
        )

        defect_spans = cfa_labels.normalize_spans(row.get("defect_spans", []), answer_len=len(answer))
        mapped = cfa_labels.map_using_triplet_char_offsets(triplet_rows, defect_spans)
        coverage = float(mapped["coverage"])
        if coverage < MIN_COVERAGE:
            raise RuntimeError(f"sample {sample_id} coverage={coverage:.6f} < {MIN_COVERAGE:.2f}")

        labels = mapped["labels"]
        token_ids = [int(r["token_id"]) for r in triplet_rows]
        cfa_labels.write_labels_jsonl(labels_path, labels=labels, token_ids=token_ids)
        labels_meta = {
            "label_source": "cfa_defect_spans_v1",
            "cfa_jsonl": cfa_path.as_posix(),
            "sample_id": sample_id,
            "variant": row.get("variant"),
            "world_type": row.get("world_type"),
            "triplets_path": triplets_path.as_posix(),
            "label_mapping_mode": mapped.get("mode"),
            "n_triplet_steps": len(token_ids),
            "n_defect_spans": len(defect_spans),
            "mapped_positive_tokens": int(mapped["mapped_positive_tokens"]),
            "total_positive_tokens": int(mapped["total_positive_tokens"]),
            "equal_blocks": int(mapped["equal_blocks"]),
            "final_alignment_coverage_ratio": coverage,
            "min_coverage_threshold": float(MIN_COVERAGE),
            "fail_below_coverage": True,
            "final_positive_steps": int(sum(1 for x in labels if x == 1)),
            "final_negative_steps": int(sum(1 for x in labels if x == 0)),
            "labels_out": labels_path.as_posix(),
        }
        cfa_labels.write_meta_json(labels_meta_path, labels_meta)

    return sample_ids


def main() -> int:
    args = parse_args()
    cfa_path = REPO_ROOT / args.cfa_jsonl
    out_dir = REPO_ROOT / args.out_dir
    attest_dir = REPO_ROOT / args.attest_dir
    samples_root = out_dir / "samples"
    gate4_input_path = out_dir / "gate4_input.json"
    gate4_out_a = out_dir / "gate4_out_a"
    gate4_out_b = out_dir / "gate4_out_b"
    date_stamp = dt.date.today().isoformat()
    parity_report = attest_dir / f"{date_stamp}_gate4_parity_smoke.txt"

    rows = load_cfa_rows(cfa_path)
    selected = select_samples(rows, args.n_consistent, args.n_frustrated)
    dataset_revision_id = "cfa_v1_small_smoke_v1"
    spec_path = REPO_ROOT / "SPEC.internal.draft.md"

    extractor.configure_reproducibility(args.seed, deterministic=True)
    device = extractor.resolve_device(args.device)
    model_id, tokenizer, model, model_revision = extractor.load_first_available_model(
        extractor.build_model_candidates(args.model_id),
        device,
    )
    model_revision = str(model_revision or "")

    sample_ids = generate_sample_outputs(
        selected=selected,
        cfa_path=cfa_path,
        samples_root=samples_root,
        model_id=model_id,
        model_revision=model_revision,
        tokenizer=tokenizer,
        model=model,
        device=device,
        seed=args.seed,
        topk=args.topk,
    )

    pack_payload = packer.pack_gate4_input(
        samples_root=samples_root,
        sample_ids=sample_ids,
        script_extract=REPO_ROOT / "tools" / "extract_triality_triplets.py",
        script_eval=REPO_ROOT / "tools" / "eval_triality_token.py",
        perm_r=args.perm_r,
        primary_score=args.primary_score,
    )
    packer.write_json(gate4_input_path, pack_payload)

    cli_path = ensure_cli_built()
    identity_hashes = compute_gate4_identity_hashes(
        cli_path=cli_path,
        cfa_jsonl=cfa_path,
        sample_ids=sample_ids,
        spec_path=spec_path,
    )

    gate4_cmd_base = [
        str(cli_path),
        "gate4",
        "run",
        "--input",
        str(gate4_input_path),
        "--run-id",
        args.run_id,
        "--dataset-revision-id",
        dataset_revision_id,
        "--dataset-hash-blake3",
        identity_hashes["dataset_hash_blake3"],
        "--spec-hash-raw-blake3",
        identity_hashes["spec_hash_raw_blake3"],
        "--spec-hash-blake3",
        identity_hashes["spec_hash_blake3"],
        "--evaluation-mode-id",
        "supervised_v1",
    ]
    run_command(gate4_cmd_base + ["--out", str(gate4_out_a)], cwd=REPO_ROOT)
    run_command(gate4_cmd_base + ["--out", str(gate4_out_b)], cwd=REPO_ROOT)

    parity_cmd = [
        sys.executable,
        str((REPO_ROOT / "tools" / "validate_gate4_parity.py").resolve()),
        "--input-json",
        str(gate4_input_path),
        "--token-features-csv",
        str(gate4_out_a / "gate4_token_features.csv"),
        "--sample-summary-csv",
        str(gate4_out_a / "gate4_sample_summary.csv"),
        "--run-summary-csv",
        str(gate4_out_a / "gate4_run_summary.csv"),
        "--manifest-json",
        str(gate4_out_a / "manifest.json"),
        "--expected-dataset-hash-blake3",
        identity_hashes["dataset_hash_blake3"],
        "--expected-spec-hash-raw-blake3",
        identity_hashes["spec_hash_raw_blake3"],
        "--expected-spec-hash-blake3",
        identity_hashes["spec_hash_blake3"],
        "--out",
        str(parity_report),
    ]
    run_command(parity_cmd, cwd=REPO_ROOT)
    parity_report_fields = parse_key_value_report(parity_report)

    deterministic_manifest = compare_bytes(gate4_out_a / "manifest.json", gate4_out_b / "manifest.json")
    deterministic_tokens = compare_bytes(
        gate4_out_a / "gate4_token_features.csv", gate4_out_b / "gate4_token_features.csv"
    )
    deterministic_summary = compare_bytes(
        gate4_out_a / "gate4_sample_summary.csv", gate4_out_b / "gate4_sample_summary.csv"
    )
    deterministic_run_summary = compare_bytes(
        gate4_out_a / "gate4_run_summary.csv", gate4_out_b / "gate4_run_summary.csv"
    )

    determinism_ok = (
        deterministic_manifest
        and deterministic_tokens
        and deterministic_summary
        and deterministic_run_summary
    )
    provenance_verdict = parity_report_fields.get("provenance_verdict", "")
    parity_verdict = parity_report_fields.get("parity_verdict", "")
    if parity_verdict != "PASS":
        final_verdict = "FAIL"
    elif not determinism_ok:
        final_verdict = "FAIL"
    elif provenance_verdict == "PASS":
        final_verdict = "PASS"
    elif provenance_verdict in {"PLACEHOLDER_IDENTITY", "UNSPECIFIED_EXPECTATION"}:
        final_verdict = "PASS_PARITY_ONLY"
    else:
        final_verdict = "FAIL"

    extra_lines = [
        f"cfa_jsonl={cfa_path.as_posix()}",
        f"spec_path={spec_path.as_posix()}",
        f"sample_ids={','.join(str(x) for x in sample_ids)}",
        f"model_id={model_id}",
        f"model_revision={model_revision}",
        f"device={device}",
        f"run_id={args.run_id}",
        f"dataset_revision_id={dataset_revision_id}",
        f"dataset_hash_blake3={identity_hashes['dataset_hash_blake3']}",
        f"spec_hash_raw_blake3={identity_hashes['spec_hash_raw_blake3']}",
        f"spec_hash_blake3={identity_hashes['spec_hash_blake3']}",
        f"gate4_input_json={gate4_input_path.as_posix()}",
        f"gate4_out_a={gate4_out_a.as_posix()}",
        f"gate4_out_b={gate4_out_b.as_posix()}",
        f"deterministic_manifest={int(deterministic_manifest)}",
        f"deterministic_token_features={int(deterministic_tokens)}",
        f"deterministic_sample_summary={int(deterministic_summary)}",
        f"deterministic_run_summary={int(deterministic_run_summary)}",
        "determinism_verdict=PASS"
        if determinism_ok
        else "determinism_verdict=FAIL",
        f"verdict={final_verdict}",
    ]
    with open(parity_report, "a", encoding="utf-8", newline="\n") as f:
        f.write("\n")
        for line in extra_lines:
            f.write(line + "\n")

    if not determinism_ok:
        raise RuntimeError(
            "Gate4 deterministic rerun drift detected: "
            f"manifest={deterministic_manifest} token_features={deterministic_tokens} "
            f"sample_summary={deterministic_summary} run_summary={deterministic_run_summary}"
        )

    if parity_verdict != "PASS":
        raise RuntimeError(
            f"unexpected parity_verdict in report: {parity_verdict}"
        )

    if final_verdict == "FAIL":
        raise RuntimeError(
            f"Gate4 validation smoke failed: provenance_verdict={provenance_verdict} "
            f"determinism_ok={determinism_ok}"
        )

    print(parity_report.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
