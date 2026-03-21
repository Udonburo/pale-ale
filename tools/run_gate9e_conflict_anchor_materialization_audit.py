#!/usr/bin/env python3
"""Run a Gate9E conflict-anchor materialization audit on Gate9D outputs."""

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import run_gate9a_graph_gauge_consumer as gate9a
import run_gate9d_conflict_motif_coverage_audit as gate9d


REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "gate9e_conflict_anchor_materialization_audit_v1"
METHOD_ID = "gate9e_conflict_anchor_materialization_audit_v1"
FOCUS_CELL_ID = "distributed_incompatibility"
FOCUS_CYCLE_TYPE = gate9d.FOCUS_CYCLE_TYPE

DEFAULT_REGISTRY = "conflict_anchor_materialization_registry.jsonl"
DEFAULT_SUMMARY = "conflict_anchor_materialization_by_answer_target.csv"
DEFAULT_STATUS = "materialization_status.json"
DEFAULT_REPORT = "gate9e_conflict_anchor_materialization_read.md"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"
DEFAULT_DRY_RUN_DIRNAME = "dry_run_targets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Gate9E dry-run audit for the distributed_incompatibility "
            "conflict-anchor materialization gap without reopening the frozen law."
        )
    )
    parser.add_argument("--gate9d-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def derive_source_context(
    source_gate9d_dir: Path,
) -> Tuple[Dict[str, Any], Path, Dict[str, Any], Path, Dict[str, Any], Path, Dict[str, Any], Path]:
    gate9d_manifest = gate9a.read_json(source_gate9d_dir / gate9a.DEFAULT_MANIFEST)
    source_gate9c_dir = REPO_ROOT / str(gate9d_manifest["source_gate9c_dir"])
    (
        _source_gate9b_dir,
        _source_gate9b_manifest,
        _source_gate9a_dir,
        source_gate9a_manifest,
        source_gate8_dir,
        source_gate9c_manifest,
    ) = gate9d.derive_source_dirs(source_gate9c_dir)
    source_gate8_manifest = gate9a.read_json(source_gate8_dir / gate9a.DEFAULT_MANIFEST)
    benchmark_dir = REPO_ROOT / str(source_gate8_manifest["benchmark_dir"])
    return (
        gate9d_manifest,
        source_gate9c_dir,
        source_gate9c_manifest,
        source_gate8_dir,
        source_gate8_manifest,
        benchmark_dir,
        source_gate9a_manifest,
        REPO_ROOT / DEFAULT_DRY_RUN_DIRNAME,
    )


def expected_conflict_anchor_target(cell_id: str, world_truth_row: Dict[str, Any]) -> Tuple[str, str]:
    if cell_id == "direct_contradiction":
        return str(world_truth_row.get("wrong_claim") or "").strip(), "world_truth.wrong_claim"
    if cell_id == "distributed_incompatibility":
        return (
            str(world_truth_row.get("distributed_block_claim") or "").strip(),
            "world_truth.distributed_block_claim",
        )
    return "", ""


def sample_dir_for_execution_id(source_gate8_dir: Path, execution_sample_id: int) -> Path:
    return source_gate8_dir / "samples" / f"sample_{execution_sample_id:06d}"


def dry_run_filename(benchmark_sample_id: str) -> str:
    return f"{benchmark_sample_id}__conflict_anchor.txt"


def write_dry_run_target(dry_run_dir: Path, benchmark_sample_id: str, target_text: str) -> Tuple[str, str]:
    dry_run_dir.mkdir(parents=True, exist_ok=True)
    target_path = dry_run_dir / dry_run_filename(benchmark_sample_id)
    gate9a.write_text(target_path, target_text + "\n" if not target_text.endswith("\n") else target_text)
    return gate9a.repo_relative_or_posix(target_path), hashlib.sha256(target_path.read_bytes()).hexdigest()


def build_registry_rows(
    gate9d_rows: Sequence[Dict[str, Any]],
    source_gate8_dir: Path,
    benchmark_rows_by_sample_id: Dict[str, Dict[str, Any]],
    rendering_rows_by_rendering_id: Dict[str, Dict[str, Any]],
    world_truth_by_world_id: Dict[str, Dict[str, Any]],
    out_dir: Path,
) -> List[Dict[str, Any]]:
    dry_run_dir = out_dir / DEFAULT_DRY_RUN_DIRNAME
    registry_rows: List[Dict[str, Any]] = []
    for row in gate9d_rows:
        benchmark_sample_id = str(row["benchmark_sample_id"])
        execution_sample_id = int(row["execution_sample_id"])
        benchmark_row = benchmark_rows_by_sample_id[benchmark_sample_id]
        rendering_row = rendering_rows_by_rendering_id[str(benchmark_row["rendering_id"])]
        world_truth_row = world_truth_by_world_id[str(benchmark_row["world_id"])]
        sample_dir = sample_dir_for_execution_id(source_gate8_dir, execution_sample_id)

        declared_conflict_chunk_ids = list(benchmark_row.get("retrieval_conflict_chunk_ids") or [])
        chunk_lookup = {
            str(chunk["chunk_id"]): chunk for chunk in list(rendering_row.get("retrieval_chunks") or [])
        }
        declared_conflict_chunks = [
            chunk_lookup[chunk_id] for chunk_id in declared_conflict_chunk_ids if chunk_id in chunk_lookup
        ]
        expected_target_text, expected_target_source_field = expected_conflict_anchor_target(
            str(row["cell_id"]),
            world_truth_row,
        )

        conflict_anchor_txt = sample_dir / "conflict_anchor.txt"
        conflict_anchor_meta = sample_dir / "conflict_anchor_meta.json"
        conflict_anchor_triplets = sample_dir / "conflict_anchor_triplets.ndjson"
        support_anchor_txt = sample_dir / "support_anchor.txt"
        support_anchor_meta = sample_dir / "support_anchor_meta.json"
        support_anchor_triplets = sample_dir / "support_anchor_triplets.ndjson"

        support_anchor_text = (
            support_anchor_txt.read_text(encoding="utf-8").strip() if support_anchor_txt.exists() else ""
        )
        support_lane_ready = (
            support_anchor_txt.exists()
            and support_anchor_meta.exists()
            and support_anchor_triplets.exists()
            and support_anchor_text == str(world_truth_row.get("support_claim") or "").strip()
        )
        declared_conflict_chunk_texts = [str(chunk.get("text") or "") for chunk in declared_conflict_chunks]
        declared_conflict_chunk_roles = [str(chunk.get("role") or "") for chunk in declared_conflict_chunks]
        declared_conflict_chunk_contains_expected_target = any(
            expected_target_text and expected_target_text in text for text in declared_conflict_chunk_texts
        )

        missing_files = [
            name
            for name, path in (
                ("conflict_anchor.txt", conflict_anchor_txt),
                ("conflict_anchor_meta.json", conflict_anchor_meta),
                ("conflict_anchor_triplets.ndjson", conflict_anchor_triplets),
            )
            if not path.exists()
        ]

        declaration_stable = bool(declared_conflict_chunk_ids) and len(declared_conflict_chunk_ids) == len(declared_conflict_chunks)
        closure_convention_change_required = not declared_conflict_chunk_contains_expected_target
        dry_run_status = "blocked"
        dry_run_target_path = ""
        dry_run_target_sha256 = ""
        if declaration_stable and expected_target_text and not closure_convention_change_required:
            dry_run_status = "candidate_emitted"
            dry_run_target_path, dry_run_target_sha256 = write_dry_run_target(
                dry_run_dir=dry_run_dir,
                benchmark_sample_id=benchmark_sample_id,
                target_text=expected_target_text,
            )

        registry_rows.append(
            {
                "execution_sample_id": execution_sample_id,
                "benchmark_sample_id": benchmark_sample_id,
                "cell_id": str(row["cell_id"]),
                "answer_target_type": str(row["answer_target_type"]),
                "world_id": str(benchmark_row["world_id"]),
                "rendering_id": str(benchmark_row["rendering_id"]),
                "source_recovery_path_status": str(row["recovery_path_status"]),
                "declared_conflict_chunk_ids": declared_conflict_chunk_ids,
                "declared_conflict_chunk_roles": declared_conflict_chunk_roles,
                "declared_conflict_chunk_texts": declared_conflict_chunk_texts,
                "declared_conflict_chunk_count": len(declared_conflict_chunk_ids),
                "expected_conflict_anchor_target_text": expected_target_text,
                "expected_conflict_anchor_target_source_field": expected_target_source_field,
                "declared_conflict_chunk_contains_expected_target": declared_conflict_chunk_contains_expected_target,
                "has_conflict_anchor_txt": conflict_anchor_txt.exists(),
                "has_conflict_anchor_meta": conflict_anchor_meta.exists(),
                "has_conflict_anchor_triplets": conflict_anchor_triplets.exists(),
                "actual_missing_conflict_anchor_files": missing_files,
                "has_support_anchor_lane": support_lane_ready,
                "dry_run_status": dry_run_status,
                "dry_run_target_path": dry_run_target_path,
                "dry_run_target_sha256": dry_run_target_sha256,
                "declaration_stable": declaration_stable,
                "closure_convention_change_required": closure_convention_change_required,
            }
        )
    return registry_rows


def summarize_rows(registry_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in registry_rows:
        grouped[str(row["answer_target_type"])].append(row)

    out_rows: List[Dict[str, Any]] = []
    for answer_target_type in sorted(grouped):
        rows = grouped[answer_target_type]
        status_counter = Counter(str(row["dry_run_status"]) for row in rows)
        out_rows.append(
            {
                "answer_target_type": answer_target_type,
                "n_rows": len(rows),
                "candidate_emitted_count": int(status_counter["candidate_emitted"]),
                "blocked_count": int(status_counter["blocked"]),
                "all_declared_chunks_stable": all(bool(row["declaration_stable"]) for row in rows),
                "all_support_anchor_lane_ready": all(bool(row["has_support_anchor_lane"]) for row in rows),
                "any_closure_convention_change_required": any(
                    bool(row["closure_convention_change_required"]) for row in rows
                ),
                "distinct_expected_target_count": len(
                    {str(row["expected_conflict_anchor_target_text"]) for row in rows}
                ),
            }
        )
    return out_rows


def build_status_payload(registry_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    candidate_rows = [row for row in registry_rows if str(row["dry_run_status"]) == "candidate_emitted"]
    declaration_unstable_rows = [row for row in registry_rows if not bool(row["declaration_stable"])]
    closure_change_rows = [row for row in registry_rows if bool(row["closure_convention_change_required"])]
    support_lane_missing_rows = [row for row in registry_rows if not bool(row["has_support_anchor_lane"])]

    expected_targets_by_rendering: Dict[str, set] = defaultdict(set)
    for row in registry_rows:
        expected_targets_by_rendering[str(row["rendering_id"])].add(
            str(row["expected_conflict_anchor_target_text"])
        )
    answer_target_split_rows = [
        {"rendering_id": rendering_id, "n_targets": len(targets)}
        for rendering_id, targets in expected_targets_by_rendering.items()
        if len(targets) > 1
    ]

    return {
        "focus_cell_id": FOCUS_CELL_ID,
        "focus_cycle_type": FOCUS_CYCLE_TYPE,
        "materialization_recovery_status": "dry_run_only",
        "dry_run_candidate_status": "candidate_emitted" if candidate_rows else "denied",
        "cleaner_side_spill_status": "clear",
        "declaration_stability_status": "triggered" if declaration_unstable_rows else "clear",
        "closure_convention_change_required_status": "triggered" if closure_change_rows else "clear",
        "answer_target_split_status": "triggered" if answer_target_split_rows else "clear",
        "existing_anchor_lane_ready_status": "clear" if not support_lane_missing_rows else "triggered",
        "cycle_coverage_recovery_status": "not_yet_rerun",
        "candidate_rows": [
            {
                "benchmark_sample_id": str(row["benchmark_sample_id"]),
                "answer_target_type": str(row["answer_target_type"]),
                "dry_run_target_path": str(row["dry_run_target_path"]),
            }
            for row in candidate_rows
        ],
    }


def build_report(
    run_id: str,
    source_gate9d_manifest: Dict[str, Any],
    summary_rows: Sequence[Dict[str, Any]],
    registry_rows: Sequence[Dict[str, Any]],
    status_payload: Dict[str, Any],
) -> str:
    lines = [
        "# Gate9E Conflict-Anchor Materialization Read",
        "",
        f"run_id: {run_id}",
        f"source_gate9d_run_id: {source_gate9d_manifest.get('run_id', '')}",
        f"source_gate9d_code_git_commit: {source_gate9d_manifest.get('code_git_commit', '')}",
        "",
        "## Discipline",
        "",
        "- scope stays on the distributed_incompatibility conflict-anchor gap",
        "- frozen law remains unchanged",
        "- operator remains closed",
        "- this is a dry-run artifact-lane audit only",
        "",
        "## Summary By Answer Target",
        "",
        "| answer_target_type | n_rows | candidate_emitted | blocked | all_declared_chunks_stable | all_support_anchor_lane_ready | any_closure_change_required | distinct_expected_target_count |",
        "|---|---:|---:|---:|---|---|---|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["answer_target_type"]),
                    str(row["n_rows"]),
                    str(row["candidate_emitted_count"]),
                    str(row["blocked_count"]),
                    str(row["all_declared_chunks_stable"]),
                    str(row["all_support_anchor_lane_ready"]),
                    str(row["any_closure_convention_change_required"]),
                    str(row["distinct_expected_target_count"]),
                ]
            )
            + " |"
        )

    lines.extend(["", "## In-Scope Rows", ""])
    for row in registry_rows:
        lines.append(
            "- "
            + f"{row['benchmark_sample_id']} / {row['answer_target_type']}: "
            + f"expected_source={row['expected_conflict_anchor_target_source_field']}, "
            + f"dry_run_status={row['dry_run_status']}, "
            + f"missing_files={','.join(row['actual_missing_conflict_anchor_files']) or 'none'}"
        )

    lines.extend(
        [
            "",
            "## Status",
            "",
            f"- materialization_recovery_status: `{status_payload['materialization_recovery_status']}`",
            f"- dry_run_candidate_status: `{status_payload['dry_run_candidate_status']}`",
            f"- cleaner_side_spill_status: `{status_payload['cleaner_side_spill_status']}`",
            f"- declaration_stability_status: `{status_payload['declaration_stability_status']}`",
            f"- closure_convention_change_required_status: `{status_payload['closure_convention_change_required_status']}`",
            f"- answer_target_split_status: `{status_payload['answer_target_split_status']}`",
            f"- existing_anchor_lane_ready_status: `{status_payload['existing_anchor_lane_ready_status']}`",
            f"- cycle_coverage_recovery_status: `{status_payload['cycle_coverage_recovery_status']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()
    source_gate9d_dir = Path(args.gate9d_dir)
    out_dir = Path(args.out_dir)
    run_id = out_dir.name

    (
        source_gate9d_manifest,
        _source_gate9c_dir,
        _source_gate9c_manifest,
        source_gate8_dir,
        source_gate8_manifest,
        benchmark_dir,
        _source_gate9a_manifest,
        _unused,
    ) = derive_source_context(source_gate9d_dir)

    gate9d_rows = gate9a.read_jsonl(source_gate9d_dir / gate9d.DEFAULT_REGISTRY)
    benchmark_rows = gate9a.read_jsonl(benchmark_dir / "benchmark_rows.jsonl")
    rendering_rows = gate9a.read_jsonl(benchmark_dir / "retrieval_renderings.jsonl")
    world_truth_rows = gate9a.read_jsonl(benchmark_dir / "world_truth.jsonl")

    focus_rows = [
        row
        for row in gate9d_rows
        if str(row["cell_id"]) == FOCUS_CELL_ID
        and str(row["recovery_path_status"]) == gate9d.RECOVERABLE_CANDIDATE
    ]
    benchmark_rows_by_sample_id = {str(row["sample_id"]): row for row in benchmark_rows}
    rendering_rows_by_rendering_id = {str(row["rendering_id"]): row for row in rendering_rows}
    world_truth_by_world_id = {str(row["world_id"]): row for row in world_truth_rows}

    registry_rows = build_registry_rows(
        gate9d_rows=focus_rows,
        source_gate8_dir=source_gate8_dir,
        benchmark_rows_by_sample_id=benchmark_rows_by_sample_id,
        rendering_rows_by_rendering_id=rendering_rows_by_rendering_id,
        world_truth_by_world_id=world_truth_by_world_id,
        out_dir=out_dir,
    )
    summary_rows = summarize_rows(registry_rows)
    status_payload = build_status_payload(registry_rows)

    registry_path = out_dir / DEFAULT_REGISTRY
    summary_path = out_dir / DEFAULT_SUMMARY
    status_path = out_dir / DEFAULT_STATUS
    report_path = out_dir / DEFAULT_REPORT
    manifest_path = out_dir / DEFAULT_MANIFEST
    checksums_path = out_dir / DEFAULT_CHECKSUMS

    gate9a.write_jsonl(registry_path, registry_rows)
    gate9a.write_csv(
        summary_path,
        (
            "answer_target_type",
            "n_rows",
            "candidate_emitted_count",
            "blocked_count",
            "all_declared_chunks_stable",
            "all_support_anchor_lane_ready",
            "any_closure_convention_change_required",
            "distinct_expected_target_count",
        ),
        summary_rows,
    )
    gate9a.write_json(status_path, status_payload)
    gate9a.write_text(
        report_path,
        build_report(
            run_id=run_id,
            source_gate9d_manifest=source_gate9d_manifest,
            summary_rows=summary_rows,
            registry_rows=registry_rows,
            status_payload=status_payload,
        ),
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "run_id": run_id,
        "code_git_commit": gate9a.current_git_commit(),
        "source_gate9d_dir": gate9a.repo_relative_or_posix(source_gate9d_dir),
        "source_gate9d_run_id": str(source_gate9d_manifest.get("run_id") or ""),
        "source_gate9d_code_git_commit": str(source_gate9d_manifest.get("code_git_commit") or ""),
        "source_gate8_run_id": str(source_gate8_manifest.get("run_id") or ""),
        "focus_cell_id": FOCUS_CELL_ID,
        "focus_cycle_type": FOCUS_CYCLE_TYPE,
        "paths": {
            DEFAULT_REGISTRY: gate9a.repo_relative_or_posix(registry_path),
            DEFAULT_SUMMARY: gate9a.repo_relative_or_posix(summary_path),
            DEFAULT_STATUS: gate9a.repo_relative_or_posix(status_path),
            DEFAULT_REPORT: gate9a.repo_relative_or_posix(report_path),
            DEFAULT_DRY_RUN_DIRNAME: gate9a.repo_relative_or_posix(out_dir / DEFAULT_DRY_RUN_DIRNAME),
        },
    }
    gate9a.write_json(manifest_path, manifest)

    checksums = {
        DEFAULT_MANIFEST: sha256_file(manifest_path),
        DEFAULT_REGISTRY: sha256_file(registry_path),
        DEFAULT_SUMMARY: sha256_file(summary_path),
        DEFAULT_STATUS: sha256_file(status_path),
        DEFAULT_REPORT: sha256_file(report_path),
    }
    dry_run_dir = out_dir / DEFAULT_DRY_RUN_DIRNAME
    if dry_run_dir.exists():
        for path in sorted(dry_run_dir.glob("*.txt")):
            checksums[gate9a.repo_relative_or_posix(path)] = sha256_file(path)
    gate9a.write_json(checksums_path, checksums)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
