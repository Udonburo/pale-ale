#!/usr/bin/env python3
"""Run Gate4 parity/smoke validation on the representative CFA case-study set."""

import argparse
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import build_gate4_input as packer
import run_gate4_validation_smoke as smoke


REPO_ROOT = Path(__file__).resolve().parents[1]
ZERO64 = "0" * 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Gate4 parity/smoke validation on the representative CFA set."
    )
    parser.add_argument(
        "--representative-summary",
        default="attestations/triality/case_study/representative_set_summary.md",
    )
    parser.add_argument("--samples-root", default="runs/cfa_batch_primaryE/samples")
    parser.add_argument("--out-dir", default="runs/gate4_representative_smoke")
    parser.add_argument("--attest-dir", default="attestations/triality/gate4_validation")
    parser.add_argument("--run-id", default="gate4_representative_smoke")
    parser.add_argument("--dataset-revision-id", default="cfa_v1_representative_smoke")
    parser.add_argument("--dataset-hash-blake3", default=ZERO64)
    parser.add_argument("--spec-hash-raw-blake3", default=ZERO64)
    parser.add_argument("--spec-hash-blake3", default=ZERO64)
    parser.add_argument("--perm-r", type=int, default=2000)
    parser.add_argument("--primary-score", default="E")
    parser.add_argument("--script-extract", default="tools/extract_triality_triplets.py")
    parser.add_argument("--script-eval", default="tools/eval_triality_token.py")
    return parser.parse_args()


def parse_representative_pairs(path: Path) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    pipe_line = re.compile(r"^\|\s*(top|median|bottom)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            match = pipe_line.match(line.strip())
            if not match:
                continue
            sample_id = int(match.group(2))
            contrast_id = int(match.group(3))
            pairs.append((sample_id, contrast_id))
    if not pairs:
        raise ValueError(f"no representative pairs parsed from {path}")
    return pairs


def flatten_pair_sample_ids(pairs: Sequence[Tuple[int, int]]) -> List[int]:
    ordered: List[int] = []
    seen = set()
    for sample_id, contrast_id in pairs:
        for current in (contrast_id, sample_id):
            if current not in seen:
                seen.add(current)
                ordered.append(current)
    return ordered


def main() -> int:
    args = parse_args()
    summary_path = REPO_ROOT / args.representative_summary
    samples_root = REPO_ROOT / args.samples_root
    out_dir = REPO_ROOT / args.out_dir
    attest_dir = REPO_ROOT / args.attest_dir
    gate4_input_path = out_dir / "gate4_input.json"
    gate4_out_a = out_dir / "gate4_out_a"
    gate4_out_b = out_dir / "gate4_out_b"
    date_stamp = dt.date.today().isoformat()
    parity_report = attest_dir / f"{date_stamp}_gate4_representative_smoke.txt"

    pairs = parse_representative_pairs(summary_path)
    sample_ids = flatten_pair_sample_ids(pairs)

    payload = packer.pack_gate4_input(
        samples_root=samples_root,
        sample_ids=sample_ids,
        script_extract=REPO_ROOT / args.script_extract,
        script_eval=REPO_ROOT / args.script_eval,
        perm_r=args.perm_r,
        primary_score=args.primary_score,
    )
    packer.write_json(gate4_input_path, payload)

    cli_path = smoke.ensure_cli_built()
    gate4_cmd_base = [
        str(cli_path),
        "gate4",
        "run",
        "--input",
        str(gate4_input_path),
        "--run-id",
        args.run_id,
        "--dataset-revision-id",
        args.dataset_revision_id,
        "--dataset-hash-blake3",
        args.dataset_hash_blake3,
        "--spec-hash-raw-blake3",
        args.spec_hash_raw_blake3,
        "--spec-hash-blake3",
        args.spec_hash_blake3,
        "--evaluation-mode-id",
        "supervised_v1",
    ]
    smoke.run_command(gate4_cmd_base + ["--out", str(gate4_out_a)], cwd=REPO_ROOT)
    smoke.run_command(gate4_cmd_base + ["--out", str(gate4_out_b)], cwd=REPO_ROOT)

    parity_cmd = [
        sys.executable,
        str((REPO_ROOT / "tools" / "validate_gate4_parity.py").resolve()),
        "--input-json",
        str(gate4_input_path),
        "--token-features-csv",
        str(gate4_out_a / "gate4_token_features.csv"),
        "--sample-summary-csv",
        str(gate4_out_a / "gate4_sample_summary.csv"),
        "--manifest-json",
        str(gate4_out_a / "manifest.json"),
        "--expected-dataset-hash-blake3",
        args.dataset_hash_blake3,
        "--expected-spec-hash-raw-blake3",
        args.spec_hash_raw_blake3,
        "--expected-spec-hash-blake3",
        args.spec_hash_blake3,
        "--out",
        str(parity_report),
    ]
    smoke.run_command(parity_cmd, cwd=REPO_ROOT)
    parity_report_fields = smoke.parse_key_value_report(parity_report)

    deterministic_manifest = smoke.compare_bytes(
        gate4_out_a / "manifest.json", gate4_out_b / "manifest.json"
    )
    deterministic_tokens = smoke.compare_bytes(
        gate4_out_a / "gate4_token_features.csv",
        gate4_out_b / "gate4_token_features.csv",
    )
    deterministic_summary = smoke.compare_bytes(
        gate4_out_a / "gate4_sample_summary.csv",
        gate4_out_b / "gate4_sample_summary.csv",
    )
    determinism_ok = deterministic_manifest and deterministic_tokens and deterministic_summary

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
        f"representative_summary={summary_path.as_posix()}",
        f"pair_count={len(pairs)}",
        f"sample_count={len(sample_ids)}",
        f"sample_ids={','.join(str(x) for x in sample_ids)}",
        "pairs="
        + ";".join(f"{sample_id}:{contrast_id}" for sample_id, contrast_id in pairs),
        f"run_id={args.run_id}",
        f"gate4_input_json={gate4_input_path.as_posix()}",
        f"gate4_out_a={gate4_out_a.as_posix()}",
        f"gate4_out_b={gate4_out_b.as_posix()}",
        f"deterministic_manifest={int(deterministic_manifest)}",
        f"deterministic_token_features={int(deterministic_tokens)}",
        f"deterministic_sample_summary={int(deterministic_summary)}",
        "determinism_verdict=PASS" if determinism_ok else "determinism_verdict=FAIL",
        f"verdict={final_verdict}",
    ]
    with open(parity_report, "a", encoding="utf-8", newline="\n") as f:
        f.write("\n")
        for line in extra_lines:
            f.write(line + "\n")

    if not determinism_ok:
        raise RuntimeError(
            "Gate4 representative deterministic rerun drift detected: "
            f"manifest={deterministic_manifest} token_features={deterministic_tokens} "
            f"sample_summary={deterministic_summary}"
        )
    if parity_verdict != "PASS":
        raise RuntimeError(f"unexpected parity_verdict in report: {parity_verdict}")
    if final_verdict == "FAIL":
        raise RuntimeError(
            f"Gate4 representative smoke failed: provenance_verdict={provenance_verdict} "
            f"determinism_ok={determinism_ok}"
        )

    print(parity_report.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
