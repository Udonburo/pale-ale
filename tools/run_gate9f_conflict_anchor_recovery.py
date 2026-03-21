#!/usr/bin/env python3
"""Run a narrow Gate9F actual conflict-anchor recovery on Gate9E outputs."""

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import extract_triality_triplets as extractor
import run_gate9a_graph_gauge_consumer as gate9a
import run_gate9c_missingness_topology_audit as gate9c
import run_gate9d_conflict_motif_coverage_audit as gate9d
import run_gate9e_conflict_anchor_materialization_audit as gate9e


REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "gate9f_conflict_anchor_recovery_v1"
METHOD_ID = "gate9f_conflict_anchor_recovery_v1"
RECOVERED_GATE8_DIRNAME = "recovered_gate8_execution"
PREREQ_GATE9A_DIRNAME = "gate9a_recovered_from_gate9f"
PREREQ_GATE9B_DIRNAME = "gate9b_recovered_from_gate9f"
PREREQ_GATE9C_DIRNAME = "gate9c_recovered_from_gate9f"
PREREQ_GATE9D_DIRNAME = "gate9d_recovered_from_gate9f"
DEFAULT_REGISTRY = "conflict_anchor_recovery_registry.jsonl"
DEFAULT_STATUS = "recovery_status.json"
DEFAULT_REPORT = "gate9f_conflict_anchor_recovery_read.md"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the distributed_incompatibility conflict-anchor branch on the "
            "existing artifact lane, then rerun Gate9D and Gate9C slices only."
        )
    )
    parser.add_argument("--gate9e-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_source_execution_bundle(source_dir: Path, recovered_dir: Path) -> None:
    if recovered_dir.exists():
        shutil.rmtree(recovered_dir)
    recovered_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("sample_registry.jsonl", "quietness_pairs.jsonl", "extraction_results.jsonl"):
        shutil.copy2(source_dir / filename, recovered_dir / filename)
    shutil.copytree(source_dir / "gate6_native", recovered_dir / "gate6_native")
    shutil.copytree(source_dir / "samples", recovered_dir / "samples")


def select_in_scope_rows(gate9e_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out_rows: List[Dict[str, Any]] = []
    for row in gate9e_rows:
        if str(row.get("cell_id") or "") != gate9e.FOCUS_CELL_ID:
            continue
        if str(row.get("dry_run_status") or "") != "candidate_emitted":
            continue
        out_rows.append(dict(row))
    out_rows.sort(key=lambda row: int(row["execution_sample_id"]))
    return out_rows


def load_source_context(
    source_gate9e_dir: Path,
) -> Tuple[
    Dict[str, Any],
    Path,
    Dict[str, Any],
    Path,
    Dict[str, Any],
    Path,
    Dict[str, Any],
    Path,
    Dict[str, Any],
]:
    source_gate9e_manifest = gate9a.read_json(source_gate9e_dir / gate9a.DEFAULT_MANIFEST)
    source_gate9d_dir = REPO_ROOT / str(source_gate9e_manifest["source_gate9d_dir"])
    source_gate9d_manifest = gate9a.read_json(source_gate9d_dir / gate9a.DEFAULT_MANIFEST)
    source_gate9c_dir = REPO_ROOT / str(source_gate9d_manifest["source_gate9c_dir"])
    (
        source_gate9b_dir,
        source_gate9b_manifest,
        _source_gate9a_dir,
        _source_gate9a_manifest,
        source_gate8_dir,
        source_gate9c_manifest,
    ) = gate9d.derive_source_dirs(source_gate9c_dir)
    source_gate8_manifest = gate9a.read_json(source_gate8_dir / gate9a.DEFAULT_MANIFEST)
    return (
        source_gate9e_manifest,
        source_gate9d_dir,
        source_gate9d_manifest,
        source_gate9c_dir,
        source_gate9c_manifest,
        source_gate9b_dir,
        source_gate9b_manifest,
        source_gate8_dir,
        source_gate8_manifest,
    )


def update_extraction_results_rows(
    extraction_rows: Sequence[Dict[str, Any]],
    recovered_dir: Path,
    recovery_rows_by_benchmark: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out_rows: List[Dict[str, Any]] = []
    for row in extraction_rows:
        new_row = dict(row)
        execution_sample_id = int(new_row["execution_sample_id"])
        sample_dir = recovered_dir / "samples" / f"sample_{execution_sample_id:06d}"
        new_row["sample_dir"] = gate9a.repo_relative_or_posix(sample_dir)
        recovery_row = recovery_rows_by_benchmark.get(str(new_row["benchmark_sample_id"]))
        if recovery_row is not None:
            new_row["conflict_anchor_steps"] = int(recovery_row["n_steps_written"])
            new_row["conflict_anchor_rank"] = int(recovery_row["conflict_anchor_rank"])
            new_row["conflict_anchor_exact_token_match_ratio"] = float(
                recovery_row["exact_token_match_ratio"]
            )
        out_rows.append(new_row)
    return out_rows


def write_recovered_manifest(
    recovered_dir: Path,
    *,
    source_gate8_dir: Path,
    source_gate8_manifest: Dict[str, Any],
    recovery_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    manifest_path = recovered_dir / DEFAULT_MANIFEST
    checksums_path = recovered_dir / DEFAULT_CHECKSUMS
    manifest = {
        "schema_version": "gate8_candidate_batch_recovered_conflict_anchor_v1",
        "method_id": METHOD_ID,
        "run_id": recovered_dir.name,
        "code_git_commit": gate9a.current_git_commit(),
        "benchmark_dir": str(source_gate8_manifest.get("benchmark_dir") or ""),
        "rendering_family_id": str(source_gate8_manifest.get("rendering_family_id") or ""),
        "model_id": str(source_gate8_manifest.get("model_id") or ""),
        "model_revision": str(source_gate8_manifest.get("model_revision") or ""),
        "device": str(source_gate8_manifest.get("device") or ""),
        "topk_requested": int(source_gate8_manifest.get("topk_requested") or 0),
        "seed": int(source_gate8_manifest.get("seed") or 0),
        "quietness_pairing_rule": str(source_gate8_manifest.get("quietness_pairing_rule") or ""),
        "n_samples_total": int(source_gate8_manifest.get("n_samples_total") or 0),
        "recovery_parent_execution_dir": gate9a.repo_relative_or_posix(source_gate8_dir),
        "recovery_parent_run_id": str(source_gate8_manifest.get("run_id") or ""),
        "recovery_parent_code_git_commit": str(source_gate8_manifest.get("code_git_commit") or ""),
        "recovery_scope_cell_id": gate9e.FOCUS_CELL_ID,
        "recovery_focus_cycle_type": gate9e.FOCUS_CYCLE_TYPE,
        "recovery_target_source_field": "world_truth.distributed_block_claim",
        "recovery_rows": [
            {
                "benchmark_sample_id": str(row["benchmark_sample_id"]),
                "answer_target_type": str(row["answer_target_type"]),
                "execution_sample_id": int(row["execution_sample_id"]),
            }
            for row in recovery_rows
        ],
        "sample_registry_path": gate9a.repo_relative_or_posix(recovered_dir / "sample_registry.jsonl"),
        "quietness_pairs_path": gate9a.repo_relative_or_posix(recovered_dir / "quietness_pairs.jsonl"),
        "extraction_results_path": gate9a.repo_relative_or_posix(recovered_dir / "extraction_results.jsonl"),
        "paths": {
            "samples": gate9a.repo_relative_or_posix(recovered_dir / "samples"),
            "gate6_native": gate9a.repo_relative_or_posix(recovered_dir / "gate6_native"),
        },
    }
    gate9a.write_json(manifest_path, manifest)
    gate9a.write_json(
        checksums_path,
        {
            "manifest.json": sha256_file(manifest_path),
            "sample_registry.jsonl": sha256_file(recovered_dir / "sample_registry.jsonl"),
            "quietness_pairs.jsonl": sha256_file(recovered_dir / "quietness_pairs.jsonl"),
            "extraction_results.jsonl": sha256_file(recovered_dir / "extraction_results.jsonl"),
        },
    )
    return manifest


def materialize_conflict_anchor_rows(
    recovered_dir: Path,
    *,
    source_gate8_manifest: Dict[str, Any],
    registry_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    model_id = str(source_gate8_manifest.get("model_id") or "")
    model_revision = str(source_gate8_manifest.get("model_revision") or "")
    device_arg = str(source_gate8_manifest.get("device") or "cpu")
    topk = int(source_gate8_manifest.get("topk_requested") or 0)
    seed = int(source_gate8_manifest.get("seed") or 0)
    if not model_id:
        raise ValueError("source Gate8 manifest is missing model_id")
    if topk <= 0:
        raise ValueError("source Gate8 manifest is missing a valid topk_requested")

    extractor.configure_reproducibility(seed=seed, deterministic=True)
    device = extractor.resolve_device(device_arg)
    _loaded_model_id, tokenizer, model, resolved_revision = extractor.load_first_available_model(
        [model_id],
        device,
    )
    effective_revision = resolved_revision or model_revision

    recovery_rows: List[Dict[str, Any]] = []
    for registry_row in registry_rows:
        execution_sample_id = int(registry_row["execution_sample_id"])
        benchmark_sample_id = str(registry_row["benchmark_sample_id"])
        answer_target_type = str(registry_row["answer_target_type"])
        world_id = str(registry_row["world_id"])
        target_text = str(registry_row["expected_conflict_anchor_target_text"])
        target_source_field = str(registry_row["expected_conflict_anchor_target_source_field"])
        if not target_text:
            raise ValueError(f"missing expected target text for {benchmark_sample_id}")

        sample_dir = recovered_dir / "samples" / f"sample_{execution_sample_id:06d}"
        prompt = (sample_dir / "prompt.txt").read_text(encoding="utf-8")
        conflict_anchor_path = sample_dir / "conflict_anchor.txt"
        conflict_anchor_triplets_path = sample_dir / "conflict_anchor_triplets.ndjson"
        conflict_anchor_meta_path = sample_dir / "conflict_anchor_meta.json"

        gate9a.write_text(conflict_anchor_path, target_text)
        triplet_rows, triplet_meta = extractor.run_teacher_forcing_extraction(
            prompt=prompt,
            target_answer=target_text,
            model=model,
            tokenizer=tokenizer,
            device=device,
            topk=topk,
            emit_native_raw=True,
        )
        ndjson_sha = extractor.write_ndjson(conflict_anchor_triplets_path, triplet_rows)
        mode_details = triplet_meta["mode_details"]
        extractor.write_meta_json(
            conflict_anchor_meta_path,
            {
                "model_id": model_id,
                "model_revision": effective_revision,
                "seed": int(seed),
                "topk_requested": int(topk),
                "topk_effective": int(triplet_meta["topk_effective"]),
                "native_raw_emitted": True,
                "native_raw_schema_id": extractor.RAW_NATIVE_SCHEMA_ID,
                "prompt_sha256": extractor.sha256_bytes(prompt.encode("utf-8")),
                "target_answer_sha256": extractor.sha256_bytes(target_text.encode("utf-8")),
                "output_ndjson_sha256": ndjson_sha,
                "output_ndjson_path": gate9a.repo_relative_or_posix(conflict_anchor_triplets_path),
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
                "benchmark_sample_id": benchmark_sample_id,
                "world_id": world_id,
                "distributed_block_claim": target_text,
                "target_text_source_field": target_source_field,
            },
        )
        conflict_anchor_object = gate9a.build_anchor_object(triplet_rows)
        recovery_rows.append(
            {
                "execution_sample_id": execution_sample_id,
                "benchmark_sample_id": benchmark_sample_id,
                "answer_target_type": answer_target_type,
                "world_id": world_id,
                "target_text_source_field": target_source_field,
                "target_text": target_text,
                "conflict_anchor_txt_path": gate9a.repo_relative_or_posix(conflict_anchor_path),
                "conflict_anchor_triplets_path": gate9a.repo_relative_or_posix(
                    conflict_anchor_triplets_path
                ),
                "conflict_anchor_meta_path": gate9a.repo_relative_or_posix(conflict_anchor_meta_path),
                "conflict_anchor_triplets_sha256": ndjson_sha,
                "n_steps_written": len(triplet_rows),
                "conflict_anchor_rank": int(conflict_anchor_object["rank_local"]),
                "exact_token_match_ratio": float(mode_details.get("exact_token_match_ratio") or 0.0),
            }
        )
    return recovery_rows


def run_python_tool(script_relative_path: str, *, args: Sequence[str]) -> None:
    command = [sys.executable, str(REPO_ROOT / script_relative_path), *args]
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{script_relative_path} failed with exit code {completed.returncode}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )


def rerun_downstream_layers(out_dir: Path, recovered_gate8_dir: Path) -> Dict[str, Path]:
    gate9a_dir = out_dir / PREREQ_GATE9A_DIRNAME
    gate9b_dir = out_dir / PREREQ_GATE9B_DIRNAME
    gate9c_dir = out_dir / PREREQ_GATE9C_DIRNAME
    gate9d_dir = out_dir / PREREQ_GATE9D_DIRNAME
    for path in (gate9a_dir, gate9b_dir, gate9c_dir, gate9d_dir):
        if path.exists():
            shutil.rmtree(path)

    run_python_tool(
        "tools/run_gate9a_graph_gauge_consumer.py",
        args=["--gate8-execution-dir", str(recovered_gate8_dir), "--out-dir", str(gate9a_dir)],
    )
    run_python_tool(
        "tools/run_gate9b_small_cycle_holonomy_study.py",
        args=["--gate9a-dir", str(gate9a_dir), "--out-dir", str(gate9b_dir)],
    )
    run_python_tool(
        "tools/run_gate9c_missingness_topology_audit.py",
        args=["--gate9b-dir", str(gate9b_dir), "--out-dir", str(gate9c_dir)],
    )
    run_python_tool(
        "tools/run_gate9d_conflict_motif_coverage_audit.py",
        args=["--gate9c-dir", str(gate9c_dir), "--out-dir", str(gate9d_dir)],
    )
    return {
        "gate9a": gate9a_dir,
        "gate9b": gate9b_dir,
        "gate9c": gate9c_dir,
        "gate9d": gate9d_dir,
    }


def build_status_payload(
    recovery_rows: Sequence[Dict[str, Any]],
    *,
    gate9c_status: Dict[str, Any],
    gate9d_status: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "focus_cell_id": gate9e.FOCUS_CELL_ID,
        "focus_cycle_type": gate9e.FOCUS_CYCLE_TYPE,
        "materialization_recovery_status": "materialized",
        "recovered_row_count": len(recovery_rows),
        "recovery_target_source_field": "world_truth.distributed_block_claim",
        "post_recovery_judgment_scope": "gate9d_and_gate9c_only",
        "prerequisite_rerun_status": "completed",
        "gate9d_coverage_recovery_status": str(gate9d_status.get("coverage_recovery_status") or ""),
        "gate9d_frozen_law_recovery_candidate_status": str(
            gate9d_status.get("frozen_law_recovery_candidate_status") or ""
        ),
        "gate9c_usable_motif_coverage_status": str(
            gate9c_status.get("usable_motif_coverage_status") or ""
        ),
        "gate9c_missingness_topology_accounted_status": str(
            gate9c_status.get("missingness_topology_accounted_status") or ""
        ),
        "gate9c_operator_admission_status": str(
            gate9c_status.get("operator_admission_status") or ""
        ),
        "recovered_rows": [
            {
                "benchmark_sample_id": str(row["benchmark_sample_id"]),
                "answer_target_type": str(row["answer_target_type"]),
                "execution_sample_id": int(row["execution_sample_id"]),
            }
            for row in recovery_rows
        ],
    }


def build_report(
    run_id: str,
    *,
    source_gate9e_manifest: Dict[str, Any],
    recovery_rows: Sequence[Dict[str, Any]],
    status_payload: Dict[str, Any],
    rerun_dirs: Dict[str, Path],
) -> str:
    lines = [
        "# Gate9F Conflict-Anchor Recovery Read",
        "",
        f"run_id: {run_id}",
        f"source_gate9e_run_id: {source_gate9e_manifest.get('run_id', '')}",
        f"source_gate9e_code_git_commit: {source_gate9e_manifest.get('code_git_commit', '')}",
        "",
        "## Discipline",
        "",
        "- recovery scope stays on `distributed_incompatibility` rows only",
        "- recovery stays on the existing `conflict_anchor` artifact lane",
        "- no new closure convention is introduced",
        "- no new cycle motif is introduced",
        "- Gate9A and Gate9B reruns are prerequisite only; public judgment stays on Gate9D and Gate9C",
        "",
        "## Materialized Rows",
        "",
    ]
    for row in recovery_rows:
        lines.append(
            "- "
            + f"{row['benchmark_sample_id']} / {row['answer_target_type']}: "
            + f"source={row['target_text_source_field']}, "
            + f"rank={row['conflict_anchor_rank']}, "
            + f"exact_token_match_ratio={row['exact_token_match_ratio']:.6f}"
        )

    lines.extend(
        [
            "",
            "## Post-Recovery Status",
            "",
            f"- gate9d_coverage_recovery_status: `{status_payload['gate9d_coverage_recovery_status']}`",
            f"- gate9d_frozen_law_recovery_candidate_status: `{status_payload['gate9d_frozen_law_recovery_candidate_status']}`",
            f"- gate9c_usable_motif_coverage_status: `{status_payload['gate9c_usable_motif_coverage_status']}`",
            f"- gate9c_missingness_topology_accounted_status: `{status_payload['gate9c_missingness_topology_accounted_status']}`",
            f"- gate9c_operator_admission_status: `{status_payload['gate9c_operator_admission_status']}`",
            "",
            "## Rerun Paths",
            "",
            f"- recovered_gate8_execution: `{gate9a.repo_relative_or_posix(rerun_dirs['recovered_gate8'])}`",
            f"- gate9a_prereq: `{gate9a.repo_relative_or_posix(rerun_dirs['gate9a'])}`",
            f"- gate9b_prereq: `{gate9a.repo_relative_or_posix(rerun_dirs['gate9b'])}`",
            f"- gate9c_judgment: `{gate9a.repo_relative_or_posix(rerun_dirs['gate9c'])}`",
            f"- gate9d_judgment: `{gate9a.repo_relative_or_posix(rerun_dirs['gate9d'])}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    source_gate9e_dir = Path(args.gate9e_dir)
    out_dir = Path(args.out_dir)
    run_id = out_dir.name

    (
        source_gate9e_manifest,
        _source_gate9d_dir,
        source_gate9d_manifest,
        _source_gate9c_dir,
        _source_gate9c_manifest,
        _source_gate9b_dir,
        _source_gate9b_manifest,
        source_gate8_dir,
        source_gate8_manifest,
    ) = load_source_context(source_gate9e_dir)
    gate9e_rows = gate9a.read_jsonl(source_gate9e_dir / gate9e.DEFAULT_REGISTRY)
    in_scope_rows = select_in_scope_rows(gate9e_rows)
    if not in_scope_rows:
        raise ValueError("no in-scope Gate9E candidate rows found for actual recovery")

    recovered_gate8_dir = out_dir / RECOVERED_GATE8_DIRNAME
    copy_source_execution_bundle(source_gate8_dir, recovered_gate8_dir)
    recovery_rows = materialize_conflict_anchor_rows(
        recovered_dir=recovered_gate8_dir,
        source_gate8_manifest=source_gate8_manifest,
        registry_rows=in_scope_rows,
    )
    recovery_rows_by_benchmark = {
        str(row["benchmark_sample_id"]): row for row in recovery_rows
    }
    extraction_rows = gate9a.read_jsonl(recovered_gate8_dir / "extraction_results.jsonl")
    gate9a.write_jsonl(
        recovered_gate8_dir / "extraction_results.jsonl",
        update_extraction_results_rows(
            extraction_rows,
            recovered_dir=recovered_gate8_dir,
            recovery_rows_by_benchmark=recovery_rows_by_benchmark,
        ),
    )
    recovered_manifest = write_recovered_manifest(
        recovered_dir=recovered_gate8_dir,
        source_gate8_dir=source_gate8_dir,
        source_gate8_manifest=source_gate8_manifest,
        recovery_rows=recovery_rows,
    )

    rerun_dirs = rerun_downstream_layers(out_dir=out_dir, recovered_gate8_dir=recovered_gate8_dir)
    rerun_dirs["recovered_gate8"] = recovered_gate8_dir
    gate9c_status = gate9a.read_json(rerun_dirs["gate9c"] / gate9c.DEFAULT_ADMISSION_STATUS)
    gate9d_status = gate9a.read_json(rerun_dirs["gate9d"] / gate9d.DEFAULT_STATUS)
    status_payload = build_status_payload(
        recovery_rows,
        gate9c_status=gate9c_status,
        gate9d_status=gate9d_status,
    )

    registry_path = out_dir / DEFAULT_REGISTRY
    status_path = out_dir / DEFAULT_STATUS
    report_path = out_dir / DEFAULT_REPORT
    manifest_path = out_dir / DEFAULT_MANIFEST
    checksums_path = out_dir / DEFAULT_CHECKSUMS

    gate9a.write_jsonl(registry_path, recovery_rows)
    gate9a.write_json(status_path, status_payload)
    gate9a.write_text(
        report_path,
        build_report(
            run_id=run_id,
            source_gate9e_manifest=source_gate9e_manifest,
            recovery_rows=recovery_rows,
            status_payload=status_payload,
            rerun_dirs=rerun_dirs,
        ),
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "run_id": run_id,
        "code_git_commit": gate9a.current_git_commit(),
        "source_gate9e_dir": gate9a.repo_relative_or_posix(source_gate9e_dir),
        "source_gate9e_run_id": str(source_gate9e_manifest.get("run_id") or ""),
        "source_gate9e_code_git_commit": str(source_gate9e_manifest.get("code_git_commit") or ""),
        "source_gate9d_run_id": str(source_gate9d_manifest.get("run_id") or ""),
        "recovered_gate8_execution_dir": gate9a.repo_relative_or_posix(recovered_gate8_dir),
        "recovered_gate8_execution_run_id": str(recovered_manifest.get("run_id") or ""),
        "focus_cell_id": gate9e.FOCUS_CELL_ID,
        "focus_cycle_type": gate9e.FOCUS_CYCLE_TYPE,
        "paths": {
            DEFAULT_REGISTRY: gate9a.repo_relative_or_posix(registry_path),
            DEFAULT_STATUS: gate9a.repo_relative_or_posix(status_path),
            DEFAULT_REPORT: gate9a.repo_relative_or_posix(report_path),
            "recovered_gate8_execution": gate9a.repo_relative_or_posix(recovered_gate8_dir),
            PREREQ_GATE9A_DIRNAME: gate9a.repo_relative_or_posix(rerun_dirs["gate9a"]),
            PREREQ_GATE9B_DIRNAME: gate9a.repo_relative_or_posix(rerun_dirs["gate9b"]),
            PREREQ_GATE9C_DIRNAME: gate9a.repo_relative_or_posix(rerun_dirs["gate9c"]),
            PREREQ_GATE9D_DIRNAME: gate9a.repo_relative_or_posix(rerun_dirs["gate9d"]),
        },
    }
    gate9a.write_json(manifest_path, manifest)
    gate9a.write_json(
        checksums_path,
        {
            DEFAULT_MANIFEST: sha256_file(manifest_path),
            DEFAULT_REGISTRY: sha256_file(registry_path),
            DEFAULT_STATUS: sha256_file(status_path),
            DEFAULT_REPORT: sha256_file(report_path),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
