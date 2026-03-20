#!/usr/bin/env python3
"""Materialize a deterministic Gate8 semi-closed conflict batch from a constitution scaffold."""

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "gate8_semiclosed_conflict_batch_v1"
WORLD_PLAN_SCHEMA_VERSION = "gate8_world_plan_v1"
RENDERING_PLAN_SCHEMA_VERSION = "gate8_rendering_plan_v1"
TARGET_PLAN_SCHEMA_VERSION = "gate8_target_plan_v1"
METHOD_ID = "gate8_semiclosed_conflict_batch_v1"
GENERATION_STAGE = "materialized_generation"

DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CONFLICT_PLAN = "conflict_plan.json"
DEFAULT_LABEL_CONTRACT = "label_contract.json"
DEFAULT_WORLD_PLAN = "world_plan.json"
DEFAULT_RENDERING_PLAN = "rendering_plan.json"
DEFAULT_TARGET_PLAN = "target_plan.json"
DEFAULT_SAMPLE_INDEX = "sample_index.jsonl"
DEFAULT_WORLD_TRUTH = "world_truth.jsonl"
DEFAULT_RENDERINGS = "retrieval_renderings.jsonl"
DEFAULT_ANSWER_TARGETS = "answer_targets.jsonl"
DEFAULT_BENCHMARK_ROWS = "benchmark_rows.jsonl"
DEFAULT_CHECKSUMS = "checksums.json"

ENTITY_POOL = (
    "Aster",
    "Beryl",
    "Cedar",
    "Dover",
    "Ember",
    "Frost",
    "Grove",
    "Haven",
    "Ivory",
    "Jasper",
    "Kappa",
    "Lumen",
    "Mirth",
    "Noble",
    "Opalx",
    "Prism",
    "Quill",
    "Riven",
    "Solis",
    "Thorn",
    "Umber",
    "Vivid",
    "Waltz",
    "Xenon",
    "Yarrow",
    "Zorin",
)

RELATION_TYPES = ("genealogy", "temporal", "reachability")


@dataclass(frozen=True)
class RelationSpec:
    world_type: str
    question: str
    fact_1: str
    fact_2: str
    support_claim: str
    wrong_claim: str
    distributed_block_claim: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a deterministic Gate8 semi-closed conflict batch from a constitution "
            "scaffold without changing candidate or evaluator semantics."
        )
    )
    parser.add_argument("--constitution-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def repo_relative_or_posix(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_git_commit() -> str:
    head_path = REPO_ROOT / ".git" / "HEAD"
    if not head_path.exists():
        return ""
    head = head_path.read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    ref_path = REPO_ROOT / ".git" / head[5:]
    if ref_path.exists():
        return ref_path.read_text(encoding="utf-8").strip()
    return ""


def build_entities(seed: int, ordinal: int) -> Tuple[str, str, str]:
    n = len(ENTITY_POOL)
    i0 = (seed + ordinal * 3) % n
    i1 = (seed + ordinal * 3 + 7) % n
    i2 = (seed + ordinal * 3 + 13) % n
    return ENTITY_POOL[i0], ENTITY_POOL[i1], ENTITY_POOL[i2]


def build_relation_spec(world_type: str, a: str, b: str, c: str) -> RelationSpec:
    if world_type == "genealogy":
        return RelationSpec(
            world_type=world_type,
            question="Question: Which family conclusion is warranted by the retrieved notes?",
            fact_1=f"{a} is the parent of {b}.",
            fact_2=f"{b} is the parent of {c}.",
            support_claim=f"{a} is an ancestor of {c}",
            wrong_claim=f"{c} is an ancestor of {a}",
            distributed_block_claim=f"no direct ancestor relation between {a} and {c} is warranted across separate ledgers",
        )
    if world_type == "temporal":
        return RelationSpec(
            world_type=world_type,
            question="Question: Which order conclusion is warranted by the retrieved notes?",
            fact_1=f"Event {a} happened before event {b}.",
            fact_2=f"Event {b} happened before event {c}.",
            support_claim=f"event {a} happened before event {c}",
            wrong_claim=f"event {c} happened before event {a}",
            distributed_block_claim=f"no direct order between event {a} and event {c} is warranted across separate ledgers",
        )
    if world_type == "reachability":
        return RelationSpec(
            world_type=world_type,
            question="Question: Which path conclusion is warranted by the retrieved notes?",
            fact_1=f"There is a directed edge from {a} to {b}.",
            fact_2=f"There is a directed edge from {b} to {c}.",
            support_claim=f"a directed path exists from {a} to {c}",
            wrong_claim=f"a directed path exists from {c} to {a}",
            distributed_block_claim=f"no direct path conclusion from {a} to {c} is warranted across separate ledgers",
        )
    raise ValueError(f"Unsupported world_type: {world_type}")


def find_span_or_fail(text: str, needle: str, label: str) -> Dict[str, Any]:
    start = text.find(needle)
    if start < 0:
        raise ValueError(f"Could not find span text: {needle!r}")
    end = start + len(needle)
    return {"start": start, "end": end, "text": needle, "label": label}


def tokenize_with_defect_labels(text: str, defect_spans: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tokens: List[Dict[str, Any]] = []
    for index, match in enumerate(re.finditer(r"\S+", text)):
        start, end = match.span()
        label_token = int(
            any(start < int(span["end"]) and end > int(span["start"]) for span in defect_spans)
        )
        tokens.append(
            {
                "token_index": index,
                "text": match.group(0),
                "start": start,
                "end": end,
                "label_token": label_token,
            }
        )
    return tokens


def build_clean_support_rendering(spec: RelationSpec) -> Tuple[List[Dict[str, str]], List[int], List[int]]:
    chunks = [
        {"role": "support", "text": f"Archive note: {spec.fact_1}"},
        {"role": "support", "text": f"Archive note: {spec.fact_2}"},
        {"role": "support", "text": f"Summary sheet: therefore {spec.support_claim}."},
    ]
    return chunks, [0, 1, 2], []


def build_clean_support_rendering_briefing(spec: RelationSpec) -> Tuple[List[Dict[str, str]], List[int], List[int]]:
    chunks = [
        {"role": "support", "text": f"Packet alpha reports: {spec.fact_1}"},
        {"role": "support", "text": f"Packet beta reports: {spec.fact_2}"},
        {"role": "support", "text": f"Briefing merge: the supported conclusion is {spec.support_claim}."},
    ]
    return chunks, [0, 1, 2], []


def build_direct_contradiction_rendering(spec: RelationSpec) -> Tuple[List[Dict[str, str]], List[int], List[int]]:
    chunks = [
        {"role": "support", "text": f"Archive note: {spec.fact_1}"},
        {"role": "support", "text": f"Archive note: {spec.fact_2}"},
        {"role": "conflict", "text": f"Disputed memo: {spec.wrong_claim}."},
    ]
    return chunks, [0, 1], [2]


def build_direct_contradiction_rendering_briefing(
    spec: RelationSpec,
) -> Tuple[List[Dict[str, str]], List[int], List[int]]:
    chunks = [
        {"role": "conflict", "text": f"Counter-brief: {spec.wrong_claim}."},
        {"role": "support", "text": f"Packet alpha reports: {spec.fact_1}"},
        {"role": "support", "text": f"Packet beta reports: {spec.fact_2}"},
    ]
    return chunks, [1, 2], [0]


def build_distributed_incompatibility_rendering(spec: RelationSpec) -> Tuple[List[Dict[str, str]], List[int], List[int]]:
    chunks = [
        {"role": "support", "text": f"Ledger North: {spec.fact_1}"},
        {"role": "support", "text": f"Ledger South: {spec.fact_2}"},
        {
            "role": "conflict",
            "text": (
                "Registry rule: cross-ledger transitivity is invalid; "
                f"{spec.distributed_block_claim}."
            ),
        },
    ]
    return chunks, [0, 1], [2]


def build_distributed_incompatibility_rendering_briefing(
    spec: RelationSpec,
) -> Tuple[List[Dict[str, str]], List[int], List[int]]:
    chunks = [
        {"role": "support", "text": f"Packet alpha reports: {spec.fact_1}"},
        {
            "role": "conflict",
            "text": (
                "Integrator warning: independent packets must not be transitively fused; "
                f"{spec.distributed_block_claim}."
            ),
        },
        {"role": "support", "text": f"Packet beta reports: {spec.fact_2}"},
    ]
    return chunks, [0, 2], [1]


def build_surface_noisy_clean_rendering(spec: RelationSpec) -> Tuple[List[Dict[str, str]], List[int], List[int]]:
    chunks = [
        {"role": "support", "text": f"Field note // relation record // {spec.fact_1}"},
        {"role": "support", "text": f"Reordered copy :: {spec.fact_2}"},
        {
            "role": "noise",
            "text": (
                "Catalog stamp: formatting drift only; semantics unchanged; "
                f"the notes still support {spec.support_claim}."
            ),
        },
    ]
    return chunks, [0, 1, 2], []


def build_surface_noisy_clean_rendering_briefing(
    spec: RelationSpec,
) -> Tuple[List[Dict[str, str]], List[int], List[int]]:
    chunks = [
        {"role": "noise", "text": "Transmission note: headers drifted during packet relay; content should be normalized."},
        {"role": "support", "text": f"Packet alpha reports: {spec.fact_1}"},
        {
            "role": "support",
            "text": f"Packet beta plus cleanup note: {spec.fact_2}; the supported conclusion remains {spec.support_claim}.",
        },
    ]
    return chunks, [1, 2], []


def build_rendering(
    cell_id: str,
    spec: RelationSpec,
    rendering_family_id: str,
) -> Tuple[List[Dict[str, str]], List[int], List[int]]:
    if rendering_family_id == "archive_v1":
        if cell_id == "clean_support":
            return build_clean_support_rendering(spec)
        if cell_id == "direct_contradiction":
            return build_direct_contradiction_rendering(spec)
        if cell_id == "distributed_incompatibility":
            return build_distributed_incompatibility_rendering(spec)
        if cell_id == "surface_noisy_clean":
            return build_surface_noisy_clean_rendering(spec)
    elif rendering_family_id == "briefing_v1":
        if cell_id == "clean_support":
            return build_clean_support_rendering_briefing(spec)
        if cell_id == "direct_contradiction":
            return build_direct_contradiction_rendering_briefing(spec)
        if cell_id == "distributed_incompatibility":
            return build_distributed_incompatibility_rendering_briefing(spec)
        if cell_id == "surface_noisy_clean":
            return build_surface_noisy_clean_rendering_briefing(spec)
    else:
        raise ValueError(f"Unsupported rendering_family_id: {rendering_family_id}")
    raise ValueError(f"Unsupported cell_id: {cell_id}")


def build_answer_payload(
    cell_id: str,
    answer_target_type: str,
    spec: RelationSpec,
    rendering_family_id: str,
) -> Dict[str, Any]:
    if rendering_family_id == "archive_v1":
        if cell_id in ("clean_support", "surface_noisy_clean"):
            answer_text = (
                f"Given the retrieved notes, {spec.support_claim}. "
                "The relation remains intact despite surface variation."
            )
            support_spans = [find_span_or_fail(answer_text, spec.support_claim, "gate8_support_span_v1")]
            conflict_spans: List[Dict[str, Any]] = []
            defect_spans: List[Dict[str, Any]] = []
        elif cell_id == "direct_contradiction" and answer_target_type == "consistent_answer":
            answer_text = (
                f"Given the aligned archive notes, {spec.support_claim}. "
                "The disputed memo is not adopted."
            )
            support_spans = [find_span_or_fail(answer_text, spec.support_claim, "gate8_support_span_v1")]
            conflict_spans = []
            defect_spans = []
        elif cell_id == "direct_contradiction" and answer_target_type == "conflict_following_wrong_answer":
            answer_text = (
                f"Given the disputed memo, {spec.wrong_claim}. "
                "This follows the conflicting note instead of the aligned archive."
            )
            support_spans = []
            conflict_spans = [find_span_or_fail(answer_text, spec.wrong_claim, "gate8_conflict_span_v1")]
            defect_spans = [find_span_or_fail(answer_text, spec.wrong_claim, "gate8_defect_span_v1")]
        elif cell_id == "distributed_incompatibility" and answer_target_type == "consistent_answer":
            answer_text = (
                f"Given the ledger split, {spec.distributed_block_claim}. "
                "The notes cannot be glued into one transitive conclusion."
            )
            support_spans = [
                find_span_or_fail(answer_text, spec.distributed_block_claim, "gate8_support_span_v1")
            ]
            conflict_spans = []
            defect_spans = []
        elif cell_id == "distributed_incompatibility" and answer_target_type == "unsupported_bridge_answer":
            answer_text = (
                f"Given the two ledgers, {spec.support_claim}. "
                "This answer wrongly glues separate records into one bridge."
            )
            support_spans = []
            conflict_spans = [find_span_or_fail(answer_text, spec.support_claim, "gate8_conflict_span_v1")]
            defect_spans = [find_span_or_fail(answer_text, spec.support_claim, "gate8_defect_span_v1")]
        else:
            raise ValueError(f"Unsupported cell/answer target combination: {cell_id}/{answer_target_type}")
    elif rendering_family_id == "briefing_v1":
        if cell_id in ("clean_support", "surface_noisy_clean"):
            answer_text = (
                f"Given the briefing packets, {spec.support_claim}. "
                "The relation remains intact despite presentation drift."
            )
            support_spans = [find_span_or_fail(answer_text, spec.support_claim, "gate8_support_span_v1")]
            conflict_spans = []
            defect_spans = []
        elif cell_id == "direct_contradiction" and answer_target_type == "consistent_answer":
            answer_text = (
                f"Given the aligned briefing packets, {spec.support_claim}. "
                "The counter-brief is not adopted."
            )
            support_spans = [find_span_or_fail(answer_text, spec.support_claim, "gate8_support_span_v1")]
            conflict_spans = []
            defect_spans = []
        elif cell_id == "direct_contradiction" and answer_target_type == "conflict_following_wrong_answer":
            answer_text = (
                f"Given the counter-brief, {spec.wrong_claim}. "
                "This follows the conflicting packet instead of the aligned briefing packets."
            )
            support_spans = []
            conflict_spans = [find_span_or_fail(answer_text, spec.wrong_claim, "gate8_conflict_span_v1")]
            defect_spans = [find_span_or_fail(answer_text, spec.wrong_claim, "gate8_defect_span_v1")]
        elif cell_id == "distributed_incompatibility" and answer_target_type == "consistent_answer":
            answer_text = (
                f"Given the packet split, {spec.distributed_block_claim}. "
                "The packets cannot be fused into one transitive conclusion."
            )
            support_spans = [
                find_span_or_fail(answer_text, spec.distributed_block_claim, "gate8_support_span_v1")
            ]
            conflict_spans = []
            defect_spans = []
        elif cell_id == "distributed_incompatibility" and answer_target_type == "unsupported_bridge_answer":
            answer_text = (
                f"Given the two packets, {spec.support_claim}. "
                "This answer wrongly fuses separate packets into one bridge."
            )
            support_spans = []
            conflict_spans = [find_span_or_fail(answer_text, spec.support_claim, "gate8_conflict_span_v1")]
            defect_spans = [find_span_or_fail(answer_text, spec.support_claim, "gate8_defect_span_v1")]
        else:
            raise ValueError(f"Unsupported cell/answer target combination: {cell_id}/{answer_target_type}")
    else:
        raise ValueError(f"Unsupported rendering_family_id: {rendering_family_id}")

    return {
        "answer_text": answer_text,
        "label_span_support": support_spans,
        "label_span_conflict": conflict_spans,
        "label_span_defect": defect_spans,
        "label_token": tokenize_with_defect_labels(answer_text, defect_spans),
    }


def build_prompt(chunks: Sequence[Dict[str, Any]], question: str, rendering_family_id: str) -> str:
    if rendering_family_id == "archive_v1":
        lines = ["Retrieved notes:"]
        for idx, chunk in enumerate(chunks, start=1):
            lines.append(f"[{idx}] {chunk['text']}")
        lines.append(question)
        lines.append("Answer in one short paragraph.")
        return "\n".join(lines)
    if rendering_family_id == "briefing_v1":
        lines = ["Briefing packets:"]
        for idx, chunk in enumerate(chunks, start=1):
            lines.append(f"{idx}. ({str(chunk['role']).upper()}) {chunk['text']}")
        lines.append(question.replace("Question:", "Task:"))
        lines.append("Return the most warranted conclusion in one short paragraph.")
        return "\n".join(lines)
    raise ValueError(f"Unsupported rendering_family_id: {rendering_family_id}")


def build_world_plan(world_truth_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    world_type_counts: Dict[str, int] = {}
    for row in world_truth_rows:
        world_type = str(row["world_type"])
        world_type_counts[world_type] = world_type_counts.get(world_type, 0) + 1
    return {
        "schema_version": WORLD_PLAN_SCHEMA_VERSION,
        "stage": GENERATION_STAGE,
        "binding_status": "materialized",
        "world_types": list(RELATION_TYPES),
        "n_worlds_total": len(world_truth_rows),
        "world_type_counts": world_type_counts,
        "world_construction_rule": "deterministic relation triples keyed by stable world_id/world_ordinal",
    }


def build_rendering_plan(rendering_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rendering_family_ids = sorted(
        {str(row.get("rendering_family_id") or "archive_v1") for row in rendering_rows}
    )
    if len(rendering_family_ids) != 1:
        raise ValueError("Gate8 materialization expects exactly one rendering family per benchmark")
    rendering_family_id = rendering_family_ids[0]
    return {
        "schema_version": RENDERING_PLAN_SCHEMA_VERSION,
        "stage": GENERATION_STAGE,
        "binding_status": "materialized",
        "rendering_family_id": rendering_family_id,
        "n_renderings_total": len(rendering_rows),
        "cell_rendering_rules": {
            "clean_support": "support-only family-specific retrieval packaging",
            "direct_contradiction": "support plus explicit contradiction under family-specific ordering",
            "distributed_incompatibility": "support plus cross-record caveat under family-specific packaging",
            "surface_noisy_clean": "surface-noisy but non-conflicting family-specific packaging",
        },
    }


def build_target_plan(target_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    target_types = sorted({str(row["answer_target_type"]) for row in target_rows})
    return {
        "schema_version": TARGET_PLAN_SCHEMA_VERSION,
        "stage": GENERATION_STAGE,
        "binding_status": "materialized",
        "n_targets_total": len(target_rows),
        "answer_target_types": target_types,
        "target_rule": "deterministic answer templates with explicit span labels",
    }


def build_manifest(
    run_id: str,
    constitution_dir: Path,
    conflict_plan: Dict[str, Any],
    label_contract: Dict[str, Any],
    world_plan_path: Path,
    rendering_plan_path: Path,
    target_plan_path: Path,
    sample_index_path: Path,
    world_truth_path: Path,
    renderings_path: Path,
    targets_path: Path,
    rows_path: Path,
    n_samples_total: int,
) -> Dict[str, Any]:
    constitution_manifest_path = constitution_dir / DEFAULT_MANIFEST
    return {
        "run_id": run_id,
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "generation_stage": GENERATION_STAGE,
        "provenance_binding_mode": "realized_artifacts",
        "samples_per_cell": int(conflict_plan["samples_per_cell"]),
        "rendering_family_id": str(conflict_plan.get("rendering_family_id") or "archive_v1"),
        "n_cells_total": len(conflict_plan["cells"]),
        "n_samples_total": n_samples_total,
        "candidate_set": conflict_plan["candidate_set"],
        "candidate_granularity_status": conflict_plan["candidate_granularity_status"],
        "candidate_granularity_note": conflict_plan["candidate_granularity_note"],
        "headline_metrics": conflict_plan["headline_metrics"],
        "aggregation_ban": True,
        "semi_closed_layers": label_contract["layer_separation"],
        "taxonomy_schema_version": conflict_plan["schema_version"],
        "label_contract_version": label_contract["schema_version"],
        "world_plan_schema_version": WORLD_PLAN_SCHEMA_VERSION,
        "rendering_plan_schema_version": RENDERING_PLAN_SCHEMA_VERSION,
        "target_plan_schema_version": TARGET_PLAN_SCHEMA_VERSION,
        "code_git_commit": current_git_commit(),
        "generator_script_path": repo_relative_or_posix(Path(__file__)),
        "generator_script_sha256": sha256_file(Path(__file__)),
        "constitution_manifest_path": repo_relative_or_posix(constitution_manifest_path),
        "constitution_manifest_sha256": sha256_file(constitution_manifest_path),
        "world_plan_path": world_plan_path.name,
        "world_plan_sha256": sha256_file(world_plan_path),
        "rendering_plan_path": rendering_plan_path.name,
        "rendering_plan_sha256": sha256_file(rendering_plan_path),
        "target_plan_path": target_plan_path.name,
        "target_plan_sha256": sha256_file(target_plan_path),
        "sample_index_path": sample_index_path.name,
        "sample_index_sha256": sha256_file(sample_index_path),
        "world_truth_path": world_truth_path.name,
        "world_truth_sha256": sha256_file(world_truth_path),
        "retrieval_renderings_path": renderings_path.name,
        "retrieval_renderings_sha256": sha256_file(renderings_path),
        "answer_targets_path": targets_path.name,
        "answer_targets_sha256": sha256_file(targets_path),
        "benchmark_rows_path": rows_path.name,
        "benchmark_rows_sha256": sha256_file(rows_path),
    }


def build_checksums(entries: Sequence[Tuple[str, Path]]) -> Dict[str, str]:
    return {name: sha256_file(path) for name, path in entries}


def materialize_rows(
    sample_rows: Sequence[Dict[str, Any]],
    seed: int,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    world_truth_by_id: Dict[str, Dict[str, Any]] = {}
    rendering_by_id: Dict[str, Dict[str, Any]] = {}
    target_rows: List[Dict[str, Any]] = []
    sample_index_rows: List[Dict[str, Any]] = []
    benchmark_rows: List[Dict[str, Any]] = []

    for sample in sample_rows:
        cell_id = str(sample["cell_id"])
        rendering_family_id = str(sample.get("rendering_family_id") or "archive_v1")
        world_type = str(sample["world_type"])
        world_ordinal = int(sample["world_ordinal"])
        a, b, c = build_entities(seed=seed, ordinal=world_ordinal)
        spec = build_relation_spec(world_type, a, b, c)
        chunks, support_chunk_indexes, conflict_chunk_indexes = build_rendering(
            cell_id,
            spec,
            rendering_family_id,
        )
        chunk_ids = [f"{sample['sample_id']}_chunk_{idx:02d}" for idx in range(len(chunks))]
        prompt = build_prompt(chunks, spec.question, rendering_family_id)
        answer_payload = build_answer_payload(
            cell_id,
            str(sample["answer_target_type"]),
            spec,
            rendering_family_id,
        )

        world_truth_row = {
            "world_id": sample["world_id"],
            "world_ordinal": world_ordinal,
            "world_type": spec.world_type,
            "entity_a": a,
            "entity_b": b,
            "entity_c": c,
            "fact_1": spec.fact_1,
            "fact_2": spec.fact_2,
            "support_claim": spec.support_claim,
            "wrong_claim": spec.wrong_claim,
            "distributed_block_claim": spec.distributed_block_claim,
            "question": spec.question,
        }
        existing_world = world_truth_by_id.get(str(sample["world_id"]))
        if existing_world is None:
            world_truth_by_id[str(sample["world_id"])] = world_truth_row
        elif existing_world != world_truth_row:
            raise ValueError(f"Inconsistent world construction for world_id={sample['world_id']}")

        rendering_row = {
            "rendering_id": sample["rendering_id"],
            "rendering_family_id": rendering_family_id,
            "world_id": sample["world_id"],
            "cell_id": cell_id,
            "retrieval_chunks": [
                {"chunk_id": f"{sample['rendering_id']}_chunk_{idx:02d}", "role": chunk["role"], "text": chunk["text"]}
                for idx, chunk in enumerate(chunks)
            ],
            "retrieval_support_chunk_ids": [
                f"{sample['rendering_id']}_chunk_{idx:02d}" for idx in support_chunk_indexes
            ],
            "retrieval_conflict_chunk_ids": [
                f"{sample['rendering_id']}_chunk_{idx:02d}" for idx in conflict_chunk_indexes
            ],
            "prompt": prompt,
        }
        existing_rendering = rendering_by_id.get(str(sample["rendering_id"]))
        if existing_rendering is None:
            rendering_by_id[str(sample["rendering_id"])] = rendering_row
        elif existing_rendering != rendering_row:
            raise ValueError(
                f"Inconsistent rendering construction for rendering_id={sample['rendering_id']}"
            )

        target_rows.append(
            {
                "target_id": sample["target_id"],
                "world_id": sample["world_id"],
                "rendering_id": sample["rendering_id"],
                "rendering_family_id": rendering_family_id,
                "answer_target_type": sample["answer_target_type"],
                "answer_text": answer_payload["answer_text"],
                "label_span_support": answer_payload["label_span_support"],
                "label_span_conflict": answer_payload["label_span_conflict"],
                "label_span_defect": answer_payload["label_span_defect"],
                "label_token": answer_payload["label_token"],
            }
        )

        sample_index_rows.append(
            {
                "sample_id": sample["sample_id"],
                "cell_id": cell_id,
                "rendering_family_id": rendering_family_id,
                "world_id": sample["world_id"],
                "rendering_id": sample["rendering_id"],
                "target_id": sample["target_id"],
                "answer_target_type": sample["answer_target_type"],
                "is_conflict_intended": sample["is_conflict_intended"],
                "is_surface_noise_only": sample["is_surface_noise_only"],
                "world_ordinal": world_ordinal,
                "world_type": world_type,
                "retrieval_chunk_ids": [
                    f"{sample['rendering_id']}_chunk_{idx:02d}" for idx in range(len(chunks))
                ],
                "retrieval_conflict_chunk_ids": [
                    f"{sample['rendering_id']}_chunk_{idx:02d}" for idx in conflict_chunk_indexes
                ],
                "retrieval_support_chunk_ids": [
                    f"{sample['rendering_id']}_chunk_{idx:02d}" for idx in support_chunk_indexes
                ],
                "status": "materialized",
            }
        )

        benchmark_rows.append(
            {
                "sample_id": sample["sample_id"],
                "cell_id": cell_id,
                "rendering_family_id": rendering_family_id,
                "world_id": sample["world_id"],
                "rendering_id": sample["rendering_id"],
                "target_id": sample["target_id"],
                "answer_target_type": sample["answer_target_type"],
                "is_conflict_intended": sample["is_conflict_intended"],
                "is_surface_noise_only": sample["is_surface_noise_only"],
                "world_ordinal": world_ordinal,
                "world_type": world_type,
                "retrieval_chunk_ids": [
                    f"{sample['rendering_id']}_chunk_{idx:02d}" for idx in range(len(chunks))
                ],
                "retrieval_conflict_chunk_ids": [
                    f"{sample['rendering_id']}_chunk_{idx:02d}" for idx in conflict_chunk_indexes
                ],
                "retrieval_support_chunk_ids": [
                    f"{sample['rendering_id']}_chunk_{idx:02d}" for idx in support_chunk_indexes
                ],
                "prompt": prompt,
                "answer_text": answer_payload["answer_text"],
                "label_token": answer_payload["label_token"],
                "label_span_support": answer_payload["label_span_support"],
                "label_span_conflict": answer_payload["label_span_conflict"],
                "label_span_defect": answer_payload["label_span_defect"],
                "status": "materialized",
            }
        )

    world_truth_rows = sorted(world_truth_by_id.values(), key=lambda row: str(row["world_id"]))
    rendering_rows = sorted(rendering_by_id.values(), key=lambda row: str(row["rendering_id"]))
    target_rows = sorted(target_rows, key=lambda row: str(row["target_id"]))
    sample_index_rows = sorted(sample_index_rows, key=lambda row: str(row["sample_id"]))
    benchmark_rows = sorted(benchmark_rows, key=lambda row: str(row["sample_id"]))
    return world_truth_rows, rendering_rows, target_rows, sample_index_rows, benchmark_rows


def main() -> int:
    args = parse_args()
    constitution_dir = (REPO_ROOT / args.constitution_dir).resolve()
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    run_id = args.run_id or out_dir.name

    conflict_plan = read_json(constitution_dir / DEFAULT_CONFLICT_PLAN)
    label_contract = read_json(constitution_dir / DEFAULT_LABEL_CONTRACT)
    sample_rows = sorted(
        read_jsonl(constitution_dir / DEFAULT_SAMPLE_INDEX),
        key=lambda row: str(row["sample_id"]),
    )

    world_truth_rows, rendering_rows, target_rows, sample_index_rows, benchmark_rows = materialize_rows(
        sample_rows,
        seed=int(args.seed),
    )

    manifest_path = out_dir / DEFAULT_MANIFEST
    conflict_plan_path = out_dir / DEFAULT_CONFLICT_PLAN
    label_contract_path = out_dir / DEFAULT_LABEL_CONTRACT
    world_plan_path = out_dir / DEFAULT_WORLD_PLAN
    rendering_plan_path = out_dir / DEFAULT_RENDERING_PLAN
    target_plan_path = out_dir / DEFAULT_TARGET_PLAN
    sample_index_path = out_dir / DEFAULT_SAMPLE_INDEX
    world_truth_path = out_dir / DEFAULT_WORLD_TRUTH
    renderings_path = out_dir / DEFAULT_RENDERINGS
    targets_path = out_dir / DEFAULT_ANSWER_TARGETS
    rows_path = out_dir / DEFAULT_BENCHMARK_ROWS
    checksums_path = out_dir / DEFAULT_CHECKSUMS

    write_json(conflict_plan_path, conflict_plan)
    write_json(label_contract_path, label_contract)
    write_json(world_plan_path, build_world_plan(world_truth_rows))
    write_json(rendering_plan_path, build_rendering_plan(rendering_rows))
    write_json(target_plan_path, build_target_plan(target_rows))
    write_jsonl(sample_index_path, sample_index_rows)
    write_jsonl(world_truth_path, world_truth_rows)
    write_jsonl(renderings_path, rendering_rows)
    write_jsonl(targets_path, target_rows)
    write_jsonl(rows_path, benchmark_rows)

    manifest = build_manifest(
        run_id,
        constitution_dir,
        conflict_plan,
        label_contract,
        world_plan_path,
        rendering_plan_path,
        target_plan_path,
        sample_index_path,
        world_truth_path,
        renderings_path,
        targets_path,
        rows_path,
        len(benchmark_rows),
    )
    write_json(manifest_path, manifest)
    write_json(
        checksums_path,
        build_checksums(
            (
                ("manifest_json", manifest_path),
                ("conflict_plan_json", conflict_plan_path),
                ("label_contract_json", label_contract_path),
                ("world_plan_json", world_plan_path),
                ("rendering_plan_json", rendering_plan_path),
                ("target_plan_json", target_plan_path),
                ("sample_index_jsonl", sample_index_path),
                ("world_truth_jsonl", world_truth_path),
                ("retrieval_renderings_jsonl", renderings_path),
                ("answer_targets_jsonl", targets_path),
                ("benchmark_rows_jsonl", rows_path),
            )
        ),
    )

    print(f"manifest_json={repo_relative_or_posix(manifest_path)}")
    print(f"world_truth_jsonl={repo_relative_or_posix(world_truth_path)}")
    print(f"retrieval_renderings_jsonl={repo_relative_or_posix(renderings_path)}")
    print(f"answer_targets_jsonl={repo_relative_or_posix(targets_path)}")
    print(f"benchmark_rows_jsonl={repo_relative_or_posix(rows_path)}")
    print(f"checksums_json={repo_relative_or_posix(checksums_path)}")
    print(f"n_samples_total={len(benchmark_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
