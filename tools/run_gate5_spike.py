#!/usr/bin/env python3
"""Run Gate5 on a Gate4RunInputV1 payload and emit local reports."""

import argparse
import datetime as dt
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
ZERO64 = "0" * 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gate5 spike on gate4_input.json.")
    parser.add_argument("--input", required=True, help="Path to gate4_input.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-id", default="gate5_spike")
    parser.add_argument("--dataset-revision-id", required=True)
    parser.add_argument("--evaluation-mode-id", default="supervised_v1")
    parser.add_argument("--spec-path", default="gate5_min_scope_ssot.md")
    parser.add_argument("--dataset-hash-blake3", default="")
    parser.add_argument("--cfa-jsonl", default="")
    parser.add_argument("--seam-jsonl", default="")
    parser.add_argument("--cli-path", default="")
    parser.add_argument("--aggregate-out", default="")
    parser.add_argument("--attestation-out", default="")
    return parser.parse_args()


def ensure_cli(cli_path_arg: str) -> Path:
    if cli_path_arg:
        return Path(cli_path_arg)
    subprocess.run(
        ["cargo", "build", "-q", "-p", "pale-ale-cli", "--bin", "pale-ale"],
        cwd=str(REPO_ROOT),
        check=True,
    )
    exe = "pale-ale.exe" if sys.platform.startswith("win") else "pale-ale"
    return REPO_ROOT / "target" / "debug" / exe


def run_command(cmd: Sequence[str]) -> str:
    completed = subprocess.run(
        list(cmd),
        cwd=str(REPO_ROOT),
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
    return completed.stdout


def load_gate4_input_sample_ids(path: Path) -> List[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [int(sample["sample_id"]) for sample in payload["samples"]]


def compute_spec_hashes_via_synthetic_identity(
    cli_path: Path, spec_path: Path, sample_ids: Sequence[int]
) -> Dict[str, str]:
    synthetic_rows = [
        {
            "sample_id": int(sample_id),
            "variant": "unknown",
            "world_type": None,
            "contrast_sample_id": None,
            "prompt": "",
            "answer": "",
            "defect_spans": [],
        }
        for sample_id in sample_ids
    ]
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", suffix=".jsonl", delete=False
    ) as handle:
        temp_path = Path(handle.name)
        for row in synthetic_rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")

    try:
        stdout = run_command(
            [
                str(cli_path),
                "gate4",
                "hash-identity",
                "--cfa-jsonl",
                str(temp_path),
                "--sample-ids",
                *[str(sample_id) for sample_id in sample_ids],
                "--spec-path",
                str(spec_path.resolve()),
                "--json",
            ]
        )
        payload = json.loads(stdout)
        data = payload["data"]
        return {
            "spec_hash_raw_blake3": str(data["spec_hash_raw_blake3"]),
            "spec_hash_blake3": str(data["spec_hash_blake3"]),
            "hash_source": "gate4_hash_identity_synthetic_dataset_v1",
        }
    finally:
        temp_path.unlink(missing_ok=True)


def compute_identity(
    cli_path: Path,
    spec_path: Path,
    input_path: Path,
    dataset_revision_id: str,
    dataset_hash_blake3: str,
    cfa_jsonl: str,
) -> Dict[str, str]:
    sample_ids = load_gate4_input_sample_ids(input_path)
    if cfa_jsonl:
        stdout = run_command(
            [
                str(cli_path),
                "gate4",
                "hash-identity",
                "--cfa-jsonl",
                str((REPO_ROOT / cfa_jsonl).resolve()),
                "--sample-ids",
                *[str(sample_id) for sample_id in sample_ids],
                "--spec-path",
                str(spec_path.resolve()),
                "--json",
            ]
        )
        payload = json.loads(stdout)
        data = payload["data"]
        return {
            "dataset_revision_id": dataset_revision_id,
            "dataset_hash_blake3": str(data["dataset_hash_blake3"]),
            "spec_hash_raw_blake3": str(data["spec_hash_raw_blake3"]),
            "spec_hash_blake3": str(data["spec_hash_blake3"]),
            "identity_source": "gate4_hash_identity",
        }

    spec_hashes = compute_spec_hashes_via_synthetic_identity(cli_path, spec_path, sample_ids)
    return {
        "dataset_revision_id": dataset_revision_id,
        "dataset_hash_blake3": dataset_hash_blake3 or ZERO64,
        "spec_hash_raw_blake3": spec_hashes["spec_hash_raw_blake3"],
        "spec_hash_blake3": spec_hashes["spec_hash_blake3"],
        "identity_source": "local_fallback_real_spec_hashes_dataset_placeholder",
    }


def write_attestation_report(path: Path, manifest: Dict[str, Any], identity: Dict[str, str]) -> None:
    lines = [
        f"date={dt.date.today().isoformat()}",
        f"run_id={manifest.get('run_id', '')}",
        f"method_id={manifest.get('method_id', '')}",
        f"spec_version={manifest.get('spec_version', '')}",
        f"dataset_revision_id={identity['dataset_revision_id']}",
        f"dataset_hash_blake3={identity['dataset_hash_blake3']}",
        f"spec_hash_raw_blake3={identity['spec_hash_raw_blake3']}",
        f"spec_hash_blake3={identity['spec_hash_blake3']}",
        f"identity_source={identity['identity_source']}",
        f"evaluation_mode_id={manifest.get('evaluation_mode_id', '')}",
        f"n_samples_total={manifest.get('n_samples_total', '')}",
        f"n_token_rows_total={manifest.get('n_token_rows_total', '')}",
        f"n_loop_rows_valid={manifest.get('n_loop_rows_valid', '')}",
        f"n_loop_rows_missing={manifest.get('n_loop_rows_missing', '')}",
        "attestation_status=local_deterministic_identity_recorded",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    args = parse_args()
    input_path = (REPO_ROOT / args.input).resolve()
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    spec_path = (REPO_ROOT / args.spec_path).resolve()
    cli_path = ensure_cli(args.cli_path)
    identity = compute_identity(
        cli_path=cli_path,
        spec_path=spec_path,
        input_path=input_path,
        dataset_revision_id=args.dataset_revision_id,
        dataset_hash_blake3=args.dataset_hash_blake3,
        cfa_jsonl=args.cfa_jsonl,
    )

    run_command(
        [
            str(cli_path),
            "gate5",
            "run",
            "--input",
            str(input_path),
            "--out",
            str(out_dir),
            "--run-id",
            args.run_id,
            "--dataset-revision-id",
            identity["dataset_revision_id"],
            "--dataset-hash-blake3",
            identity["dataset_hash_blake3"],
            "--spec-hash-raw-blake3",
            identity["spec_hash_raw_blake3"],
            "--spec-hash-blake3",
            identity["spec_hash_blake3"],
            "--evaluation-mode-id",
            args.evaluation_mode_id,
        ]
    )

    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    attestation_out = (
        Path(args.attestation_out).resolve()
        if args.attestation_out
        else out_dir / "gate5_attestation_report.txt"
    )
    write_attestation_report(attestation_out, manifest, identity)

    aggregate_out = (
        Path(args.aggregate_out).resolve()
        if args.aggregate_out
        else out_dir / "gate5_aggregate_report.md"
    )
    aggregate_cmd = [
        sys.executable,
        str((REPO_ROOT / "tools" / "aggregate_gate5_spike.py").resolve()),
        "--gate5-out-dir",
        str(out_dir),
        "--out",
        str(aggregate_out),
    ]
    if args.cfa_jsonl:
        aggregate_cmd.extend(["--cfa-jsonl", str((REPO_ROOT / args.cfa_jsonl).resolve())])
    if args.seam_jsonl:
        aggregate_cmd.extend(["--seam-jsonl", str((REPO_ROOT / args.seam_jsonl).resolve())])
        aggregate_cmd.extend(["--surface", "seam"])
    run_command(aggregate_cmd)

    print(str(out_dir))
    print(str(attestation_out))
    print(str(aggregate_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
