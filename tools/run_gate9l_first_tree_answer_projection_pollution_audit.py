#!/usr/bin/env python3
"""Run a Gate9L first-tree answer-projection pollution audit on Gate9K outputs."""

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import run_gate9a_graph_gauge_consumer as gate9a
import run_gate9k_trusted_tree_residual_chord_logging as gate9k


REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "gate9l_first_tree_answer_projection_pollution_audit_v1"
METHOD_ID = "gate9l_first_tree_answer_projection_pollution_audit_v1"

DEFAULT_TREE_REGISTRY = "first_tree_edge_registry.jsonl"
DEFAULT_RESIDUAL_REGISTRY = "first_tree_residual_pollution_registry.jsonl"
DEFAULT_CELL_SUMMARY = "first_tree_residual_pollution_by_cell.csv"
DEFAULT_STATUS = "first_tree_residual_pollution_status.json"
DEFAULT_REPORT = "gate9l_first_tree_answer_projection_pollution_read.md"
DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CHECKSUMS = "checksums.json"

CLEANER_CELLS = {"clean_support", "surface_noisy_clean"}
CONFLICT_CELLS = {"direct_contradiction", "distributed_incompatibility"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Gate9L first-tree answer-projection pollution audit over the "
            "Gate9K trusted-tree logging bundle."
        )
    )
    parser.add_argument("--gate9k-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive_source_context(
    source_gate9k_dir: Path,
) -> Tuple[Dict[str, Any], Path, Dict[str, Any], Path]:
    source_gate9k_manifest = gate9a.read_json(source_gate9k_dir / gate9a.DEFAULT_MANIFEST)
    source_gate9j_dir = REPO_ROOT / str(source_gate9k_manifest["source_gate9j_dir"])
    source_gate9j_manifest = gate9a.read_json(source_gate9j_dir / gate9a.DEFAULT_MANIFEST)
    source_gate9a_dir = REPO_ROOT / str(source_gate9k_manifest["source_gate9a_dir"])
    return source_gate9k_manifest, source_gate9j_dir, source_gate9j_manifest, source_gate9a_dir


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


class UnionFind:
    def __init__(self, items: Sequence[str]) -> None:
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item: str) -> str:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: str, right: str) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return True

    def component_count(self) -> int:
        return len({self.find(item) for item in self.parent})


def canonical_pair(left: str, right: str) -> Tuple[str, str]:
    return (left, right) if left <= right else (right, left)


def build_first_tree_and_residual_rows(
    node_rows: Sequence[Dict[str, Any]],
    edge_rows: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    node_ids_by_sample: Dict[int, List[str]] = defaultdict(list)
    for row in node_rows:
        node_ids_by_sample[int(row["execution_sample_id"])].append(str(row["node_id"]))

    trusted_candidates_by_sample: Dict[int, Dict[Tuple[str, str], List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    residual_candidates: List[Dict[str, Any]] = []
    for row in edge_rows:
        edge_type = str(row["edge_type"])
        sample_id = int(row["execution_sample_id"])
        if edge_type in gate9k.TRUSTED_EDGE_TYPES:
            trusted_candidates_by_sample[sample_id][canonical_pair(str(row["source_node_id"]), str(row["target_node_id"]))].append(row)
        elif edge_type in gate9k.RESIDUAL_CHORD_EDGE_TYPES:
            residual_candidates.append(row)

    tree_rows: List[Dict[str, Any]] = []
    residual_rows: List[Dict[str, Any]] = []

    for sample_id, node_ids in sorted(node_ids_by_sample.items()):
        uf = UnionFind(node_ids)
        ordered_pairs = []
        for pair, rows in trusted_candidates_by_sample[sample_id].items():
            edge_types = {str(row["edge_type"]) for row in rows}
            priority = 0 if "temporal_transition" in edge_types else 1
            representative = sorted(rows, key=lambda row: str(row["edge_id"]))[0]
            ordered_pairs.append((priority, pair, representative))
        for _priority, pair, representative in sorted(ordered_pairs, key=lambda item: (item[0], item[1][0], item[1][1], str(item[2]["edge_id"]))):
            selected = uf.union(pair[0], pair[1])
            tree_rows.append(
                {
                    "execution_sample_id": sample_id,
                    "benchmark_sample_id": str(representative["benchmark_sample_id"]),
                    "cell_id": str(representative["cell_id"]),
                    "world_id": str(representative["world_id"]),
                    "answer_target_type": str(representative["answer_target_type"]),
                    "tree_edge_pair_id": f"{pair[0]}<->{pair[1]}",
                    "representative_edge_id": str(representative["edge_id"]),
                    "representative_edge_type": str(representative["edge_type"]),
                    "tree_edge_selected": selected,
                    "source_node_id": pair[0],
                    "target_node_id": pair[1],
                }
            )
        component_count = uf.component_count()
        for row in residual_candidates:
            if int(row["execution_sample_id"]) != sample_id:
                continue
            cell_id = str(row["cell_id"])
            residual_rows.append(
                {
                    "edge_id": str(row["edge_id"]),
                    "edge_type": str(row["edge_type"]),
                    "execution_sample_id": sample_id,
                    "benchmark_sample_id": str(row["benchmark_sample_id"]),
                    "cell_id": cell_id,
                    "world_id": str(row["world_id"]),
                    "world_type": str(row["world_type"]),
                    "answer_target_type": str(row["answer_target_type"]),
                    "edge_transport_defect": row.get("edge_transport_defect"),
                    "sample_component_count": component_count,
                    "cell_class": (
                        "cleaner" if cell_id in CLEANER_CELLS else "conflict" if cell_id in CONFLICT_CELLS else "other"
                    ),
                }
            )
    return tree_rows, residual_rows


def summarize_residual_by_cell(residual_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in residual_rows:
        grouped[(str(row["cell_id"]), str(row["edge_type"]))].append(row)

    out_rows: List[Dict[str, Any]] = []
    for cell_edge in sorted(grouped):
        rows = grouped[cell_edge]
        defects = [
            float(row["edge_transport_defect"])
            for row in rows
            if row["edge_transport_defect"] not in (None, "")
        ]
        component_counts = sorted({int(row["sample_component_count"]) for row in rows})
        out_rows.append(
            {
                "cell_id": cell_edge[0],
                "edge_type": cell_edge[1],
                "n_edges": len(rows),
                "mean_edge_transport_defect": mean_or_none(defects),
                "sample_component_counts": ",".join(str(value) for value in component_counts),
            }
        )
    return out_rows


def build_status_payload(tree_rows: Sequence[Dict[str, Any]], residual_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    selected_tree_edges = [row for row in tree_rows if bool(row["tree_edge_selected"])]
    skipped_tree_edges = [row for row in tree_rows if not bool(row["tree_edge_selected"])]

    cleaner_answer_projection = [
        row for row in residual_rows if row["cell_class"] == "cleaner" and row["edge_type"] == "answer_projection"
    ]
    cleaner_conflict_anchor = [
        row for row in residual_rows if row["cell_class"] == "cleaner" and row["edge_type"] == "conflict_anchor"
    ]
    conflict_answer_projection = [
        row for row in residual_rows if row["cell_class"] == "conflict" and row["edge_type"] == "answer_projection"
    ]
    conflict_conflict_anchor = [
        row for row in residual_rows if row["cell_class"] == "conflict" and row["edge_type"] == "conflict_anchor"
    ]

    cleaner_answer_projection_residual_pollution_status = (
        "triggered" if cleaner_answer_projection and not cleaner_conflict_anchor else "clear"
    )
    conflict_residual_chord_bridge_status = (
        "clear" if conflict_answer_projection and conflict_conflict_anchor else "denied"
    )
    return {
        "trusted_forest_build_status": "built" if selected_tree_edges else "denied",
        "trusted_forest_cycle_free_status": "clear" if not skipped_tree_edges else "triggered",
        "tree_choice_dependence_status": "not_yet_executed",
        "cleaner_answer_projection_residual_pollution_status": cleaner_answer_projection_residual_pollution_status,
        "residual_cleaner_pollution_source_status": (
            "answer_projection_only" if cleaner_answer_projection and not cleaner_conflict_anchor else "mixed_or_absent"
        ),
        "conflict_residual_chord_bridge_status": conflict_residual_chord_bridge_status,
        "trusted_tree_selected_edge_count": len(selected_tree_edges),
        "trusted_tree_skipped_edge_count": len(skipped_tree_edges),
        "cleaner_residual_answer_projection_edge_count": len(cleaner_answer_projection),
        "cleaner_residual_conflict_anchor_edge_count": len(cleaner_conflict_anchor),
        "conflict_residual_answer_projection_edge_count": len(conflict_answer_projection),
        "conflict_residual_conflict_anchor_edge_count": len(conflict_conflict_anchor),
        "bypass_readiness_status": (
            "denied" if cleaner_answer_projection_residual_pollution_status == "triggered" else "not_yet_denied"
        ),
        "next_named_blocker": (
            "cleaner_answer_projection_residual_pollution"
            if cleaner_answer_projection_residual_pollution_status == "triggered"
            else ""
        ),
    }


def build_report(
    run_id: str,
    source_gate9k_manifest: Dict[str, Any],
    cell_summary_rows: Sequence[Dict[str, Any]],
    status_payload: Dict[str, Any],
) -> str:
    lines = [
        "# Gate9L First-Tree Answer-Projection Pollution Read",
        "",
        f"run_id: {run_id}",
        f"source_gate9k_run_id: {source_gate9k_manifest.get('run_id', '')}",
        f"source_gate9k_code_git_commit: {source_gate9k_manifest.get('code_git_commit', '')}",
        "",
        "## Discipline",
        "",
        "- one deterministic first forest only",
        "- no tree-choice sensitivity yet",
        "- residual set remains policy-defined, not score-shaped",
        "- operator admission remains denied",
        "",
        "## Residual Summary By Cell",
        "",
        "| cell_id | edge_type | n_edges | mean_edge_transport_defect | sample_component_counts |",
        "|---|---|---:|---:|---|",
    ]
    for row in cell_summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["cell_id"]),
                    str(row["edge_type"]),
                    str(row["n_edges"]),
                    "" if row["mean_edge_transport_defect"] in (None, "") else f"{float(row['mean_edge_transport_defect']):.6f}",
                    str(row["sample_component_counts"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Status",
            "",
            f"- trusted_forest_build_status: `{status_payload['trusted_forest_build_status']}`",
            f"- trusted_forest_cycle_free_status: `{status_payload['trusted_forest_cycle_free_status']}`",
            f"- tree_choice_dependence_status: `{status_payload['tree_choice_dependence_status']}`",
            f"- cleaner_answer_projection_residual_pollution_status: `{status_payload['cleaner_answer_projection_residual_pollution_status']}`",
            f"- residual_cleaner_pollution_source_status: `{status_payload['residual_cleaner_pollution_source_status']}`",
            f"- conflict_residual_chord_bridge_status: `{status_payload['conflict_residual_chord_bridge_status']}`",
            f"- bypass_readiness_status: `{status_payload['bypass_readiness_status']}`",
        ]
    )
    if status_payload.get("next_named_blocker"):
        lines.extend(["", "## Next Blocker", "", f"- `{status_payload['next_named_blocker']}`"])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    source_gate9k_dir = Path(args.gate9k_dir)
    out_dir = Path(args.out_dir)
    run_id = out_dir.name

    source_gate9k_manifest, _source_gate9j_dir, _source_gate9j_manifest, source_gate9a_dir = derive_source_context(
        source_gate9k_dir
    )
    node_rows = gate9a.read_jsonl(source_gate9a_dir / gate9a.DEFAULT_NODE_REGISTRY)
    edge_rows = gate9a.read_jsonl(source_gate9a_dir / gate9a.DEFAULT_EDGE_REGISTRY)

    tree_rows, residual_rows = build_first_tree_and_residual_rows(node_rows, edge_rows)
    cell_summary_rows = summarize_residual_by_cell(residual_rows)
    status_payload = build_status_payload(tree_rows, residual_rows)

    tree_registry_path = out_dir / DEFAULT_TREE_REGISTRY
    residual_registry_path = out_dir / DEFAULT_RESIDUAL_REGISTRY
    cell_summary_path = out_dir / DEFAULT_CELL_SUMMARY
    status_path = out_dir / DEFAULT_STATUS
    report_path = out_dir / DEFAULT_REPORT
    manifest_path = out_dir / DEFAULT_MANIFEST
    checksums_path = out_dir / DEFAULT_CHECKSUMS

    gate9a.write_jsonl(tree_registry_path, tree_rows)
    gate9a.write_jsonl(residual_registry_path, residual_rows)
    gate9a.write_csv(
        cell_summary_path,
        ("cell_id", "edge_type", "n_edges", "mean_edge_transport_defect", "sample_component_counts"),
        cell_summary_rows,
    )
    gate9a.write_json(status_path, status_payload)
    gate9a.write_text(
        report_path,
        build_report(
            run_id=run_id,
            source_gate9k_manifest=source_gate9k_manifest,
            cell_summary_rows=cell_summary_rows,
            status_payload=status_payload,
        ),
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "run_id": run_id,
        "code_git_commit": gate9a.current_git_commit(),
        "source_gate9k_dir": gate9a.repo_relative_or_posix(source_gate9k_dir),
        "source_gate9k_run_id": str(source_gate9k_manifest.get("run_id") or ""),
        "source_gate9k_code_git_commit": str(source_gate9k_manifest.get("code_git_commit") or ""),
        "source_gate9a_dir": gate9a.repo_relative_or_posix(source_gate9a_dir),
        "paths": {
            DEFAULT_TREE_REGISTRY: gate9a.repo_relative_or_posix(tree_registry_path),
            DEFAULT_RESIDUAL_REGISTRY: gate9a.repo_relative_or_posix(residual_registry_path),
            DEFAULT_CELL_SUMMARY: gate9a.repo_relative_or_posix(cell_summary_path),
            DEFAULT_STATUS: gate9a.repo_relative_or_posix(status_path),
            DEFAULT_REPORT: gate9a.repo_relative_or_posix(report_path),
        },
    }
    gate9a.write_json(manifest_path, manifest)
    gate9a.write_json(
        checksums_path,
        {
            DEFAULT_MANIFEST: sha256_file(manifest_path),
            DEFAULT_TREE_REGISTRY: sha256_file(tree_registry_path),
            DEFAULT_RESIDUAL_REGISTRY: sha256_file(residual_registry_path),
            DEFAULT_CELL_SUMMARY: sha256_file(cell_summary_path),
            DEFAULT_STATUS: sha256_file(status_path),
            DEFAULT_REPORT: sha256_file(report_path),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
