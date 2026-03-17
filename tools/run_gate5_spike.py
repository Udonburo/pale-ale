#!/usr/bin/env python3
"""Run Gate5 on a Gate4RunInputV1 payload and emit local reports."""

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
ZERO64 = "0" * 64
BOUNDARY_PROVENANCE_SIDECAR = "gate5_boundary_input_provenance.json"


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
    parser.add_argument("--seam-pair-summary-out", default="")
    parser.add_argument("--seam-family-summary-out", default="")
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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_gate4_input_sample_ids(path: Path) -> List[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [int(sample["sample_id"]) for sample in payload["samples"]]


def is_gate6_compat_payload(payload: Dict[str, Any]) -> bool:
    samples = payload.get("samples", [])
    if not samples:
        return False
    token_steps = samples[0].get("token_steps", [])
    if not token_steps:
        return False
    return "compat_vectors" in token_steps[0]


def build_gate4_adapter_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    metadata = payload["metadata"]
    samples_out: List[Dict[str, Any]] = []
    for sample in payload["samples"]:
        token_steps_out: List[Dict[str, Any]] = []
        for step in sample["token_steps"]:
            compat_vectors = step["compat_vectors"]
            token_steps_out.append(
                {
                    "step": int(step["step"]),
                    "absolute_pos": int(step["absolute_pos"]),
                    "answer_char_start": step.get("answer_char_start"),
                    "answer_char_end": step.get("answer_char_end"),
                    "token_id": int(step["token_id"]),
                    "token_str": str(step["token_text"]),
                    "label_token": int(step["label_token"]),
                    "defect_span_id": step.get("defect_span_id"),
                    "V_8d": [float(x) for x in compat_vectors["V_local8"]],
                    "Splus_8d": [float(x) for x in compat_vectors["Splus_local8"]],
                    "Sminus_8d": [float(x) for x in compat_vectors["Sminus_local8"]],
                    "baseline_logprob": float(step["baseline_logprob"]),
                    "baseline_entropy": float(step["baseline_entropy"]),
                }
            )
        samples_out.append(
            {
                "sample_id": int(sample["sample_id"]),
                "variant": str(sample.get("variant") or "unknown"),
                "world_type": sample.get("world_type"),
                "exact_token_match_ratio": float(sample["exact_token_match_ratio"]),
                "label_coverage_ratio": float(sample["label_coverage_ratio"]),
                "triplets_sha256": str(sample["triplets_sha256"]),
                "labels_sha256": str(sample["labels_sha256"]),
                "token_steps": token_steps_out,
            }
        )
    return {
        "metadata": {
            "model_id": str(metadata["model_id"]),
            "model_revision": str(metadata.get("model_revision") or ""),
            "seed": int(metadata["seed"]),
            "perm_r": int(metadata.get("perm_r") or 0),
            "primary_score": str(metadata.get("primary_score") or ""),
            "proj_id": str(metadata["proj_id"]),
            "splus_def_id": str(metadata["splus_def_id"]),
            "sminus_def_id": str(metadata["sminus_def_id"]),
            "script_sha256_extract": str(metadata["script_sha256_extract"]),
            "script_sha256_eval": str(metadata.get("script_sha256_eval") or ""),
        },
        "samples": samples_out,
    }


def build_boundary_provenance(input_path: Path, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not is_gate6_compat_payload(payload):
        return None

    metadata = payload["metadata"]
    boundary_manifest_path = input_path.with_name("manifest.json")
    provenance = {
        "adapter_mode": "gate6_compat_to_gate4_cli_v1",
        "boundary_origin": str(metadata.get("boundary_origin") or ""),
        "compatibility_schema_id": str(metadata.get("compatibility_schema_id") or ""),
        "local_object_method_id": str(metadata.get("local_object_method_id") or ""),
        "source_tensor_id": str(metadata.get("source_tensor_id") or ""),
        "canonical_input_path": str(input_path),
        "canonical_input_sha256": sha256_file(input_path),
        "canonical_boundary_manifest_path": "",
        "canonical_boundary_manifest_sha256": "",
    }
    if boundary_manifest_path.exists():
        provenance["canonical_boundary_manifest_path"] = str(boundary_manifest_path)
        provenance["canonical_boundary_manifest_sha256"] = sha256_file(boundary_manifest_path)
    return provenance


def materialize_gate5_input(input_path: Path) -> Tuple[Path, Optional[Path], Optional[Dict[str, Any]]]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    provenance = build_boundary_provenance(input_path, payload)
    if not is_gate6_compat_payload(payload):
        return (input_path, None, provenance)

    adapter_payload = build_gate4_adapter_payload(payload)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", suffix=".json", delete=False
    ) as handle:
        temp_path = Path(handle.name)
        handle.write(json.dumps(adapter_payload, ensure_ascii=False, allow_nan=False))
        handle.write("\n")
    return (temp_path, temp_path, provenance)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def attach_boundary_provenance(
    out_dir: Path,
    manifest_path: Path,
    manifest: Dict[str, Any],
    provenance: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if provenance is None:
        return manifest

    sidecar_path = out_dir / BOUNDARY_PROVENANCE_SIDECAR
    sidecar_payload = dict(provenance)
    sidecar_payload["canonical_input_path"] = str(Path(provenance["canonical_input_path"]).resolve())
    if provenance.get("canonical_boundary_manifest_path"):
        sidecar_payload["canonical_boundary_manifest_path"] = str(
            Path(provenance["canonical_boundary_manifest_path"]).resolve()
        )
    write_json(sidecar_path, sidecar_payload)

    updated = dict(manifest)
    updated["boundary_origin"] = str(provenance.get("boundary_origin") or "")
    updated["compatibility_schema_id"] = str(provenance.get("compatibility_schema_id") or "")
    updated["local_object_method_id"] = str(provenance.get("local_object_method_id") or "")
    updated["source_tensor_id"] = str(provenance.get("source_tensor_id") or "")
    updated["canonical_boundary_input_path"] = str(Path(provenance["canonical_input_path"]).resolve())
    updated["canonical_boundary_input_sha256"] = str(provenance.get("canonical_input_sha256") or "")
    updated["canonical_boundary_manifest_path"] = str(
        provenance.get("canonical_boundary_manifest_path") or ""
    )
    updated["canonical_boundary_manifest_sha256"] = str(
        provenance.get("canonical_boundary_manifest_sha256") or ""
    )
    updated["boundary_input_provenance_sidecar"] = str(sidecar_path.resolve())
    updated["input_adapter_mode"] = str(provenance.get("adapter_mode") or "")
    write_json(manifest_path, updated)
    return updated


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
    gate5_input_path, temp_input_path, boundary_provenance = materialize_gate5_input(input_path)
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

    try:
        run_command(
            [
                str(cli_path),
                "gate5",
                "run",
                "--input",
                str(gate5_input_path),
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
        manifest = attach_boundary_provenance(out_dir, manifest_path, manifest, boundary_provenance)

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
            if args.seam_pair_summary_out:
                aggregate_cmd.extend(
                    ["--seam-pair-summary-out", str(Path(args.seam_pair_summary_out).resolve())]
                )
            if args.seam_family_summary_out:
                aggregate_cmd.extend(
                    ["--seam-family-summary-out", str(Path(args.seam_family_summary_out).resolve())]
                )
        run_command(aggregate_cmd)
    finally:
        if temp_input_path is not None:
            temp_input_path.unlink(missing_ok=True)

    print(str(out_dir))
    print(str(attestation_out))
    print(str(aggregate_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
