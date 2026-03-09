#!/usr/bin/env python3
"""Run model-free Gate4 fixture parity and determinism checks."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Gate4 parity and deterministic rerun checks from a committed, "
            "model-free fixture."
        )
    )
    parser.add_argument(
        "--fixture-input",
        default="fixtures/gate4/core/gate4_input.json",
    )
    parser.add_argument(
        "--fixture-cfa-jsonl",
        default="fixtures/gate4/core/cfa_subset.txt",
    )
    parser.add_argument(
        "--spec-path",
        default="SPEC.internal.draft.md",
    )
    parser.add_argument(
        "--run-id",
        default="gate4_fixture_parity",
    )
    parser.add_argument(
        "--dataset-revision-id",
        default="cfa_v1_fixture_core_v1",
    )
    parser.add_argument(
        "--evaluation-mode-id",
        default="supervised_v1",
    )
    parser.add_argument(
        "--out-report",
        help="Optional report path. If omitted, a temporary report is used.",
    )
    return parser.parse_args()


def run_command(cmd: Sequence[str]) -> subprocess.CompletedProcess:
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
    return completed


def ensure_cli_built() -> Path:
    run_command(["cargo", "build", "-p", "pale-ale-cli"])
    for candidate in (
        REPO_ROOT / "target" / "debug" / "pale-ale.exe",
        REPO_ROOT / "target" / "debug" / "pale-ale",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("pale-ale CLI binary not found after cargo build")


def compare_bytes(path_a: Path, path_b: Path) -> bool:
    return path_a.read_bytes() == path_b.read_bytes()


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
    completed = run_command(cmd)
    payload = json.loads(completed.stdout)
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
            raise RuntimeError(f"invalid {key} from gate4 hash-identity: {value!r}")
        out[key] = value
    return out


def main() -> int:
    args = parse_args()
    fixture_input = REPO_ROOT / args.fixture_input
    fixture_cfa = REPO_ROOT / args.fixture_cfa_jsonl
    spec_path = REPO_ROOT / args.spec_path
    payload = json.loads(fixture_input.read_text(encoding="utf-8"))
    sample_ids: List[int] = [int(sample["sample_id"]) for sample in payload["samples"]]
    cli_path = ensure_cli_built()
    identity_hashes = compute_gate4_identity_hashes(
        cli_path=cli_path,
        cfa_jsonl=fixture_cfa,
        sample_ids=sample_ids,
        spec_path=spec_path,
    )

    with tempfile.TemporaryDirectory(prefix="gate4_fixture_parity_") as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        out_a = temp_dir / "gate4_out_a"
        out_b = temp_dir / "gate4_out_b"
        report_path = (
            Path(args.out_report)
            if args.out_report
            else temp_dir / "gate4_fixture_parity_report.txt"
        )

        gate4_cmd_base = [
            str(cli_path),
            "gate4",
            "run",
            "--input",
            str(fixture_input),
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
        run_command(gate4_cmd_base + ["--out", str(out_a)])
        run_command(gate4_cmd_base + ["--out", str(out_b)])

        parity_cmd = [
            sys.executable,
            str((REPO_ROOT / "tools" / "validate_gate4_parity.py").resolve()),
            "--input-json",
            str(fixture_input),
            "--token-features-csv",
            str(out_a / "gate4_token_features.csv"),
            "--sample-summary-csv",
            str(out_a / "gate4_sample_summary.csv"),
            "--run-summary-csv",
            str(out_a / "gate4_run_summary.csv"),
            "--manifest-json",
            str(out_a / "manifest.json"),
            "--expected-dataset-hash-blake3",
            identity_hashes["dataset_hash_blake3"],
            "--expected-spec-hash-raw-blake3",
            identity_hashes["spec_hash_raw_blake3"],
            "--expected-spec-hash-blake3",
            identity_hashes["spec_hash_blake3"],
            "--out",
            str(report_path),
        ]
        run_command(parity_cmd)
        report_fields = parse_key_value_report(report_path)

        deterministic_manifest = compare_bytes(out_a / "manifest.json", out_b / "manifest.json")
        deterministic_tokens = compare_bytes(
            out_a / "gate4_token_features.csv",
            out_b / "gate4_token_features.csv",
        )
        deterministic_summary = compare_bytes(
            out_a / "gate4_sample_summary.csv",
            out_b / "gate4_sample_summary.csv",
        )
        deterministic_run_summary = compare_bytes(
            out_a / "gate4_run_summary.csv",
            out_b / "gate4_run_summary.csv",
        )
        determinism_ok = (
            deterministic_manifest
            and deterministic_tokens
            and deterministic_summary
            and deterministic_run_summary
        )
        parity_verdict = report_fields.get("parity_verdict", "")
        provenance_verdict = report_fields.get("provenance_verdict", "")
        final_verdict = (
            "PASS"
            if parity_verdict == "PASS" and provenance_verdict == "PASS" and determinism_ok
            else "FAIL"
        )

        extra_lines = [
            f"fixture_input={fixture_input.as_posix()}",
            f"fixture_cfa_jsonl={fixture_cfa.as_posix()}",
            f"spec_path={spec_path.as_posix()}",
            f"sample_ids={','.join(str(x) for x in sample_ids)}",
            f"deterministic_manifest={int(deterministic_manifest)}",
            f"deterministic_token_features={int(deterministic_tokens)}",
            f"deterministic_sample_summary={int(deterministic_summary)}",
            f"deterministic_run_summary={int(deterministic_run_summary)}",
            "determinism_verdict=PASS" if determinism_ok else "determinism_verdict=FAIL",
            f"verdict={final_verdict}",
        ]
        with open(report_path, "a", encoding="utf-8", newline="\n") as f:
            f.write("\n")
            for line in extra_lines:
                f.write(line + "\n")

        print(report_path.read_text(encoding="utf-8"), end="")

        if final_verdict != "PASS":
            raise RuntimeError(
                f"fixture parity failed: parity_verdict={parity_verdict} "
                f"provenance_verdict={provenance_verdict} determinism_ok={determinism_ok}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
