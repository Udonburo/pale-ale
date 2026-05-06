#!/usr/bin/env python3
"""Run Gate12B observer-relative coarse-grained closure as a secondary audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()

SCHEMA_VERSION = "gate12b_observer_relative_coarse_grained_closure_v1"
METHOD_ID = "gate12b_observer_relative_coarse_grained_closure_v1"
SECONDARY_AUDIT_MODE = "read_only_gate12a_artifacts_v1"
PRIMITIVE_MODE = "observer_x_scale_x_admissible_gauge_transform_v1"
GAUGE_LANGUAGE_BOUNDARY = "basis_preserving_local_reparameterization_v1"

DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CYCLE_REGISTRY = "explicit_triangle_cycle_registry.jsonl"
DEFAULT_HOLONOMY_REGISTRY = "triangle_holonomy_registry.jsonl"
DEFAULT_TRANSPORT_REGISTRY = "transport_relation_registry.jsonl"
DEFAULT_TRANSPORT_ARRAYS = "transport_operator_arrays.npz"

DEFAULT_MATRIX_CSV = "observer_scale_closure_matrix.csv"
DEFAULT_MATRIX_JSON = "observer_scale_closure_matrix.json"
DEFAULT_INVARIANT_CANDIDATES = "invariant_signature_candidates.jsonl"
DEFAULT_GAUGE_MATRIX = "gauge_stability_matrix.csv"
DEFAULT_GAUGE_SUMMARY = "gauge_stability_summary.json"
DEFAULT_GAUGE_CANDIDATES = "gauge_variant_signature_candidates.jsonl"
DEFAULT_READ = "gate12b_observer_relative_coarse_grained_closure.md"
DEFAULT_CHECKSUMS = "checksums.json"

CORE_OBSERVER_MODES = (
    "all_edges",
    "anchor_qualified",
    "residual_chord_heavy",
    "relation_kind_conditioned",
)
OBSERVER_MODES = CORE_OBSERVER_MODES
CYCLE_MOTIF_EXPANSION_OBSERVER_MODES = CORE_OBSERVER_MODES + (
    "residual_first_leg",
    "residual_second_leg",
    "residual_third_leg",
)
OBSERVER_MODE_SETS = {
    "core_v1": CORE_OBSERVER_MODES,
    "cycle_motif_expansion_v1": CYCLE_MOTIF_EXPANSION_OBSERVER_MODES,
}

SCALE_MODES = (
    "triangle",
    "relation_kind_band",
    "anchor_policy_band",
    "residual_quantile_band",
)

GAUGE_TRANSFORM_WITH_ARRAYS = "basis_coordinate_reversal_v1"
GAUGE_TRANSFORM_REGISTRY_ONLY = "registry_identity_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read one Gate12A discrete-connection run and emit a Gate12B "
            "observer x scale x admissible-gauge-transform secondary audit."
        )
    )
    parser.add_argument("--gate12a-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--flat-quantile", type=float, default=0.25)
    parser.add_argument("--high-quantile", type=float, default=0.75)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-observer-support", type=int, default=2)
    parser.add_argument("--min-scale-support", type=int, default=2)
    parser.add_argument(
        "--observer-mode-set",
        choices=sorted(OBSERVER_MODE_SETS),
        default="core_v1",
    )
    parser.add_argument("--tau-gauge-residual-delta", type=float, default=1.0e-8)
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n")


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: csv_value(row.get(name)) for name in fieldnames})


def write_text(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8", newline="\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def repo_relative_or_posix(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def require_keys(row: Mapping[str, Any], keys: Sequence[str], context: str) -> None:
    missing = [key for key in keys if key not in row]
    if missing:
        raise ValueError(f"{context} is missing required keys: {missing}")


def quantile(values: Sequence[float], p: float) -> float:
    if not values:
        raise ValueError("cannot compute quantile over an empty sequence")
    xs = sorted(float(value) for value in values)
    index = (len(xs) - 1) * float(p)
    lo = int(index)
    hi = min(lo + 1, len(xs) - 1)
    frac = index - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def summarize_floats(values: Sequence[float]) -> Dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "mean": None, "max": None}
    xs = [float(value) for value in values]
    return {
        "min": min(xs),
        "median": statistics.median(xs),
        "mean": statistics.fmean(xs),
        "max": max(xs),
    }


def classify_band(residual: float | None, *, flat_cut: float, high_cut: float) -> str:
    if residual is None:
        return "undefined"
    value = float(residual)
    if value <= flat_cut:
        return "flat"
    if value > high_cut:
        return "high_tension"
    return "tense"


def relation_kind_signature(relation_kind_counts: Mapping[str, int]) -> str:
    parts = []
    for key in sorted(relation_kind_counts):
        count = int(relation_kind_counts[key])
        if count > 0:
            parts.append(f"{key}={count}")
    return "|".join(parts) if parts else "none"


def rank_pattern_for_cycle(
    *,
    cycle: Mapping[str, Any],
    ordered_edges: Sequence[Mapping[str, Any]],
) -> str:
    node_ranks: Dict[str, int] = {}
    for edge in ordered_edges:
        node_ranks[str(edge["source_node_id"])] = int(edge["source_rank"])
        node_ranks[str(edge["target_node_id"])] = int(edge["target_rank"])
    ranks = [node_ranks.get(str(node_id), 0) for node_id in list(cycle["node_id_path"])[:3]]
    return ">".join(str(rank) for rank in ranks)


def observer_modes_for_set(observer_mode_set: str) -> Tuple[str, ...]:
    try:
        return tuple(OBSERVER_MODE_SETS[str(observer_mode_set)])
    except KeyError as exc:
        raise ValueError(f"unsupported observer_mode_set: {observer_mode_set}") from exc


def reconstruct_ordered_edges(
    *,
    cycle: Mapping[str, Any],
    edge_map: Mapping[str, Mapping[str, Any]],
) -> List[Mapping[str, Any]]:
    cycle_edge_ids = {str(edge_id) for edge_id in cycle["edge_id_path"]}
    node_path = [str(node_id) for node_id in cycle["node_id_path"]]
    ordered_edges: List[Mapping[str, Any]] = []
    for source_node_id, target_node_id in zip(node_path[:3], node_path[1:4]):
        matches = [
            edge
            for edge_id, edge in edge_map.items()
            if edge_id in cycle_edge_ids
            and str(edge["source_node_id"]) == source_node_id
            and str(edge["target_node_id"]) == target_node_id
        ]
        if len(matches) != 1:
            raise ValueError(
                "triangle cycle cannot be reconstructed from edge_id_path "
                f"for cycle {cycle['cycle_id']}"
            )
        ordered_edges.append(matches[0])
    return ordered_edges


def load_gate12a_rows(gate12a_dir: Path) -> Dict[str, Any]:
    gate12a_dir = Path(gate12a_dir)
    manifest = read_json(gate12a_dir / DEFAULT_MANIFEST)
    cycle_rows = read_jsonl(gate12a_dir / DEFAULT_CYCLE_REGISTRY)
    holonomy_rows = read_jsonl(gate12a_dir / DEFAULT_HOLONOMY_REGISTRY)
    transport_rows = read_jsonl(gate12a_dir / DEFAULT_TRANSPORT_REGISTRY)

    cycle_map: Dict[str, Mapping[str, Any]] = {}
    for row in cycle_rows:
        require_keys(row, ("cycle_id", "base_node_id", "edge_id_path", "node_id_path"), "cycle row")
        cycle_map[str(row["cycle_id"])] = row

    edge_map: Dict[str, Mapping[str, Any]] = {}
    for row in transport_rows:
        require_keys(
            row,
            (
                "edge_id",
                "source_node_id",
                "target_node_id",
                "relation_kind",
                "anchor_qualified",
                "source_rank",
                "target_rank",
                "transport_case",
                "operator_array_index",
                "transport_level_compatibility_status",
            ),
            "transport row",
        )
        edge_map[str(row["edge_id"])] = row

    joined_rows: List[Dict[str, Any]] = []
    for row in holonomy_rows:
        require_keys(row, ("cycle_id", "base_node_id", "holonomy_status"), "holonomy row")
        cycle_id = str(row["cycle_id"])
        if cycle_id not in cycle_map:
            raise ValueError(f"holonomy row references unknown cycle_id: {cycle_id}")
        cycle = cycle_map[cycle_id]
        ordered_edges = reconstruct_ordered_edges(cycle=cycle, edge_map=edge_map)
        relation_counts: Dict[str, int] = {"trusted_tree": 0, "residual_chord": 0}
        ordered_relation_kind_path = [str(edge["relation_kind"]) for edge in ordered_edges]
        anchor_count = 0
        compatibility_statuses: Dict[str, int] = {}
        transport_cases: Dict[str, int] = {}
        for edge in ordered_edges:
            relation_kind = str(edge["relation_kind"])
            relation_counts[relation_kind] = relation_counts.get(relation_kind, 0) + 1
            if boolish(edge["anchor_qualified"]):
                anchor_count += 1
            compatibility_status = str(edge["transport_level_compatibility_status"])
            compatibility_statuses[compatibility_status] = compatibility_statuses.get(compatibility_status, 0) + 1
            transport_case = str(edge["transport_case"])
            transport_cases[transport_case] = transport_cases.get(transport_case, 0) + 1

        holonomy_status = str(row["holonomy_status"])
        raw_residual = row.get("holonomy_residual_fro")
        residual = float(raw_residual) if holonomy_status == "defined" and raw_residual is not None else None
        joined_rows.append(
            {
                "cycle_id": cycle_id,
                "base_node_id": str(row["base_node_id"]),
                "node_id_path": [str(node_id) for node_id in cycle["node_id_path"]],
                "edge_id_path": [str(edge_id) for edge_id in cycle["edge_id_path"]],
                "ordered_edge_id_path": [str(edge["edge_id"]) for edge in ordered_edges],
                "ordered_relation_kind_path": ordered_relation_kind_path,
                "holonomy_status": holonomy_status,
                "holonomy_rank": int(row.get("holonomy_rank") or 0),
                "holonomy_residual_fro": residual,
                "anchor_qualified_count": int(anchor_count),
                "anchor_policy_band": "anchor_qualified" if anchor_count > 0 else "plain",
                "relation_kind_counts": relation_counts,
                "relation_kind_signature": relation_kind_signature(relation_counts),
                "residual_chord_count": int(relation_counts.get("residual_chord", 0)),
                "trusted_tree_count": int(relation_counts.get("trusted_tree", 0)),
                "rank_pattern": rank_pattern_for_cycle(cycle=cycle, ordered_edges=ordered_edges),
                "compatibility_status_counts": compatibility_statuses,
                "transport_case_counts": transport_cases,
            }
        )

    defined_residuals = [
        float(row["holonomy_residual_fro"])
        for row in joined_rows
        if row["holonomy_status"] == "defined" and row["holonomy_residual_fro"] is not None
    ]
    if not defined_residuals:
        raise ValueError("Gate12B secondary audit requires at least one defined Gate12A holonomy row")

    return {
        "manifest": manifest,
        "cycle_rows": cycle_rows,
        "holonomy_rows": holonomy_rows,
        "transport_rows": transport_rows,
        "edge_map": edge_map,
        "cycle_map": cycle_map,
        "joined_rows": joined_rows,
    }


def finalize_bands(
    joined_rows: Sequence[Dict[str, Any]],
    *,
    flat_quantile: float,
    high_quantile: float,
) -> Dict[str, Any]:
    defined_rows = [
        row
        for row in joined_rows
        if row["holonomy_status"] == "defined" and row["holonomy_residual_fro"] is not None
    ]
    residuals = [float(row["holonomy_residual_fro"]) for row in defined_rows]
    flat_cut = quantile(residuals, flat_quantile)
    high_cut = quantile(residuals, high_quantile)

    sorted_defined = sorted(defined_rows, key=lambda row: (float(row["holonomy_residual_fro"]), str(row["cycle_id"])))
    denominator = max(len(sorted_defined) - 1, 1)
    percentiles = {
        str(row["cycle_id"]): (index / denominator if len(sorted_defined) > 1 else 0.0)
        for index, row in enumerate(sorted_defined)
    }

    counts = {"flat": 0, "tense": 0, "high_tension": 0, "undefined": 0}
    for row in joined_rows:
        band = classify_band(
            row["holonomy_residual_fro"],
            flat_cut=float(flat_cut),
            high_cut=float(high_cut),
        )
        row["residual_quantile_band"] = band
        row["residual_percentile"] = percentiles.get(str(row["cycle_id"]))
        counts[band] += 1
    return {
        "flat_cut": float(flat_cut),
        "high_cut": float(high_cut),
        "band_counts": counts,
    }


def observer_predicates() -> Dict[str, Callable[[Mapping[str, Any]], bool]]:
    return {
        "all_edges": lambda row: True,
        "anchor_qualified": lambda row: int(row["anchor_qualified_count"]) > 0,
        "residual_chord_heavy": lambda row: int(row["residual_chord_count"]) >= 2,
        "relation_kind_conditioned": lambda row: int(row["residual_chord_count"]) > 0
        and int(row["trusted_tree_count"]) > 0,
        "residual_first_leg": lambda row: list(row.get("ordered_relation_kind_path") or ["", "", ""])[0]
        == "residual_chord",
        "residual_second_leg": lambda row: list(row.get("ordered_relation_kind_path") or ["", "", ""])[1]
        == "residual_chord",
        "residual_third_leg": lambda row: list(row.get("ordered_relation_kind_path") or ["", "", ""])[2]
        == "residual_chord",
    }


def scale_key(row: Mapping[str, Any], scale: str) -> str:
    if scale == "triangle":
        return str(row["cycle_id"])
    if scale == "relation_kind_band":
        return str(row["relation_kind_signature"])
    if scale == "anchor_policy_band":
        return str(row["anchor_policy_band"])
    if scale == "residual_quantile_band":
        return str(row["residual_quantile_band"])
    raise ValueError(f"unsupported scale mode: {scale}")


def summarize_group(
    *,
    observer: str,
    scale: str,
    key: str,
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    residuals = [
        float(row["holonomy_residual_fro"])
        for row in rows
        if row["holonomy_status"] == "defined" and row["holonomy_residual_fro"] is not None
    ]
    band_counts = {
        "flat": sum(1 for row in rows if row["residual_quantile_band"] == "flat"),
        "tense": sum(1 for row in rows if row["residual_quantile_band"] == "tense"),
        "high_tension": sum(1 for row in rows if row["residual_quantile_band"] == "high_tension"),
        "undefined": sum(1 for row in rows if row["residual_quantile_band"] == "undefined"),
    }
    defined_count = len(residuals)
    dominant_band = "none"
    if defined_count > 0:
        dominant_band = max(
            ("flat", "tense", "high_tension"),
            key=lambda band: (int(band_counts[band]), {"flat": 0, "tense": 1, "high_tension": 2}[band]),
        )
    summary = summarize_floats(residuals)
    return {
        "observer": observer,
        "scale": scale,
        "scale_key": key,
        "included_cycle_count": len(rows),
        "defined_cycle_count": defined_count,
        "undefined_cycle_count": len(rows) - defined_count,
        "residual_min": summary["min"],
        "residual_median": summary["median"],
        "residual_mean": summary["mean"],
        "residual_max": summary["max"],
        "flat_count": band_counts["flat"],
        "tense_count": band_counts["tense"],
        "high_tension_count": band_counts["high_tension"],
        "undefined_count": band_counts["undefined"],
        "dominant_closure_band": dominant_band,
        "anchor_qualified_cycle_count": sum(1 for row in rows if int(row["anchor_qualified_count"]) > 0),
        "residual_chord_cycle_count": sum(1 for row in rows if int(row["residual_chord_count"]) > 0),
        "rank_patterns": sorted({str(row["rank_pattern"]) for row in rows}),
        "relation_kind_signatures": sorted({str(row["relation_kind_signature"]) for row in rows}),
    }


def build_observer_scale_matrix(
    joined_rows: Sequence[Mapping[str, Any]],
    *,
    observer_modes: Sequence[str],
) -> List[Dict[str, Any]]:
    predicates = observer_predicates()
    matrix_rows: List[Dict[str, Any]] = []
    for observer in observer_modes:
        observer_rows = [row for row in joined_rows if predicates[observer](row)]
        for scale in SCALE_MODES:
            grouped: Dict[str, List[Mapping[str, Any]]] = {}
            for row in observer_rows:
                grouped.setdefault(scale_key(row, scale), []).append(row)
            for key in sorted(grouped):
                matrix_rows.append(
                    summarize_group(
                        observer=observer,
                        scale=scale,
                        key=key,
                        rows=sorted(grouped[key], key=lambda item: str(item["cycle_id"])),
                    )
                )
    return matrix_rows


def topk_cycle_hits(
    rows: Sequence[Mapping[str, Any]],
    *,
    top_k: int,
) -> Tuple[set[str], set[str]]:
    defined = [
        row
        for row in rows
        if row["holonomy_status"] == "defined" and row["holonomy_residual_fro"] is not None
    ]
    ordered = sorted(defined, key=lambda row: (float(row["holonomy_residual_fro"]), str(row["cycle_id"])))
    k = max(0, min(int(top_k), len(ordered)))
    flat_hits = {str(row["cycle_id"]) for row in ordered[:k]}
    high_hits = {str(row["cycle_id"]) for row in ordered[-k:]} if k > 0 else set()
    return flat_hits, high_hits


def build_observer_views(
    joined_rows: Sequence[Mapping[str, Any]],
    *,
    observer_modes: Sequence[str],
) -> Dict[str, List[Mapping[str, Any]]]:
    predicates = observer_predicates()
    return {
        observer: [row for row in joined_rows if predicates[observer](row)]
        for observer in observer_modes
    }


def build_observer_scope_maps(
    observer_views: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    observer_modes: Sequence[str],
) -> Dict[str, Any]:
    scope_key_to_observers: Dict[Tuple[str, ...], List[str]] = {}
    observer_order = {observer: index for index, observer in enumerate(observer_modes)}
    for observer in observer_modes:
        scope_key = tuple(sorted(str(row["cycle_id"]) for row in observer_views.get(observer, [])))
        scope_key_to_observers.setdefault(scope_key, []).append(observer)

    observer_to_scope_id: Dict[str, str] = {}
    scope_id_to_observers: Dict[str, List[str]] = {}
    for index, scope_key in enumerate(
        sorted(
            scope_key_to_observers,
            key=lambda key: min(observer_order[observer] for observer in scope_key_to_observers[key]),
        )
    ):
        scope_id = f"observer_scope:{index:03d}"
        observers = sorted(scope_key_to_observers[scope_key], key=lambda observer: observer_order[observer])
        scope_id_to_observers[scope_id] = observers
        for observer in observers:
            observer_to_scope_id[observer] = scope_id
    return {
        "observer_to_scope_id": observer_to_scope_id,
        "scope_id_to_observers": scope_id_to_observers,
    }


def scale_extreme_cycle_hits(
    rows: Sequence[Mapping[str, Any]],
    *,
    scale: str,
    top_k: int,
) -> Tuple[set[str], set[str]]:
    if scale == "triangle":
        return topk_cycle_hits(rows, top_k=top_k)

    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        if row["holonomy_status"] == "defined" and row["holonomy_residual_fro"] is not None:
            grouped.setdefault(scale_key(row, scale), []).append(row)
    if len(grouped) < 2:
        return set(), set()

    summaries: List[Tuple[float, str, List[Mapping[str, Any]]]] = []
    for key, group_rows in grouped.items():
        residuals = [float(row["holonomy_residual_fro"]) for row in group_rows]
        summaries.append((statistics.median(residuals), key, group_rows))
    summaries.sort(key=lambda item: (item[0], item[1]))

    flat_groups = summaries[:1]
    high_groups = summaries[-1:]
    flat_hits = {
        str(row["cycle_id"])
        for _median, _key, group_rows in flat_groups
        for row in group_rows
        if row["residual_quantile_band"] == "flat"
    }
    high_hits = {
        str(row["cycle_id"])
        for _median, _key, group_rows in high_groups
        for row in group_rows
        if row["residual_quantile_band"] == "high_tension"
    }
    return flat_hits, high_hits


def unique_support_entries(entries: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[Tuple[str, str]] = set()
    unique: List[Dict[str, Any]] = []
    for entry in entries:
        key = (str(entry["observer_scope_id"]), str(entry["scale"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(dict(entry))
    return unique


def build_invariant_candidates(
    joined_rows: Sequence[Mapping[str, Any]],
    *,
    top_k: int,
    min_observer_support: int,
    min_scale_support: int,
    observer_modes: Sequence[str],
) -> List[Dict[str, Any]]:
    observer_views = build_observer_views(joined_rows, observer_modes=observer_modes)
    scope_maps = build_observer_scope_maps(observer_views, observer_modes=observer_modes)
    observer_to_scope_id = scope_maps["observer_to_scope_id"]
    scope_id_to_observers = scope_maps["scope_id_to_observers"]
    flat_support: Dict[str, List[Dict[str, Any]]] = {}
    high_support: Dict[str, List[Dict[str, Any]]] = {}
    membership: Dict[str, List[str]] = {str(row["cycle_id"]): [] for row in joined_rows}

    observer_order = {observer: index for index, observer in enumerate(observer_modes)}
    for observer in observer_modes:
        observer_rows = observer_views[observer]
        for row in observer_rows:
            membership[str(row["cycle_id"])].append(observer)
        observer_scope_id = str(observer_to_scope_id[observer])
        for scale in SCALE_MODES:
            flat_hits, high_hits = scale_extreme_cycle_hits(observer_rows, scale=scale, top_k=top_k)
            for cycle_id in flat_hits:
                flat_support.setdefault(cycle_id, []).append(
                    {
                        "observer": observer,
                        "observer_scope_id": observer_scope_id,
                        "scale": scale,
                    }
                )
            for cycle_id in high_hits:
                high_support.setdefault(cycle_id, []).append(
                    {
                        "observer": observer,
                        "observer_scope_id": observer_scope_id,
                        "scale": scale,
                    }
                )

    row_map = {str(row["cycle_id"]): row for row in joined_rows}
    candidates: List[Dict[str, Any]] = []
    for cycle_id in sorted(row_map):
        row = row_map[cycle_id]
        for candidate_kind, support_map in (
            ("flat_observer_scale_stable", flat_support),
            ("high_tension_observer_scale_stable", high_support),
        ):
            support_entries = sorted(
                unique_support_entries(support_map.get(cycle_id, [])),
                key=lambda entry: (str(entry["observer_scope_id"]), str(entry["scale"])),
            )
            observer_scope_ids = sorted({str(entry["observer_scope_id"]) for entry in support_entries})
            scale_modes = sorted({str(entry["scale"]) for entry in support_entries})
            coarse_scale_modes = sorted(scale for scale in scale_modes if scale != "triangle")
            if len(observer_scope_ids) < int(min_observer_support):
                continue
            if len(scale_modes) < int(min_scale_support):
                continue
            if not coarse_scale_modes:
                continue
            support_observers = sorted(
                {observer for scope_id in observer_scope_ids for observer in scope_id_to_observers[scope_id]},
                key=lambda observer: observer_order[observer],
            )
            candidates.append(
                {
                    "cycle_id": cycle_id,
                    "candidate_kind": candidate_kind,
                    "candidate_status": "candidate",
                    "support_observers": support_observers,
                    "support_entries": support_entries,
                    "observer_memberships": sorted(membership.get(cycle_id, [])),
                    "observer_support_count": len(observer_scope_ids),
                    "observer_view_support_count": len(support_observers),
                    "observer_scope_groups": [
                        {
                            "observer_scope_id": scope_id,
                            "observers": scope_id_to_observers[scope_id],
                        }
                        for scope_id in observer_scope_ids
                    ],
                    "scale_support_count": len(scale_modes),
                    "coarse_scale_support_count": len(coarse_scale_modes),
                    "support_scales": scale_modes,
                    "support_coarse_scales": coarse_scale_modes,
                    "holonomy_residual_fro": row["holonomy_residual_fro"],
                    "residual_quantile_band": row["residual_quantile_band"],
                    "residual_percentile": row["residual_percentile"],
                    "scale_keys": {scale: scale_key(row, scale) for scale in SCALE_MODES},
                    "relation_kind_signature": row["relation_kind_signature"],
                    "anchor_policy_band": row["anchor_policy_band"],
                    "rank_pattern": row["rank_pattern"],
                }
            )
    return candidates


def load_optional_transport_matrices(gate12a_dir: Path) -> Tuple[bool, np.ndarray | None]:
    arrays_path = Path(gate12a_dir) / DEFAULT_TRANSPORT_ARRAYS
    if not arrays_path.exists():
        return False, None
    with np.load(arrays_path) as handle:
        matrices = np.asarray(handle["transport_matrix_local"], dtype=np.float64)
    if matrices.ndim != 3:
        raise ValueError("transport_matrix_local must have shape [E, r_max, r_max]")
    return True, matrices


def path_is_relative_to(*, child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_output_directory(*, gate12a_dir: Path, out_dir: Path) -> None:
    source_dir = Path(gate12a_dir).resolve(strict=False)
    target_dir = Path(out_dir).resolve(strict=False)
    if target_dir == source_dir:
        raise ValueError(
            "Gate12B out_dir must not be the same directory as gate12a_dir; "
            "the Gate12A source artifact directory is read-only input."
        )
    if path_is_relative_to(child=target_dir, parent=source_dir):
        raise ValueError(
            "Gate12B out_dir must not be inside gate12a_dir; choose a separate "
            "output directory so the Gate12A source artifact directory remains read-only input."
        )

    gate12a_marker_files = (
        DEFAULT_CYCLE_REGISTRY,
        DEFAULT_HOLONOMY_REGISTRY,
        DEFAULT_TRANSPORT_REGISTRY,
    )
    if not target_dir.exists() or not all((target_dir / name).exists() for name in gate12a_marker_files):
        return

    manifest_path = target_dir / DEFAULT_MANIFEST
    if not manifest_path.exists():
        return
    manifest = read_json(manifest_path)
    schema_version = str(manifest.get("schema_version") or "")
    method_id = str(manifest.get("method_id") or "")
    if schema_version.startswith("gate12a_") or method_id.startswith("gate12a_"):
        raise ValueError(
            "Gate12B out_dir must not point at an existing Gate12A artifact directory; "
            "choose a separate output directory."
        )


def reversal_matrix(rank: int) -> np.ndarray:
    if rank <= 0:
        return np.zeros((0, 0), dtype=np.float64)
    return np.eye(rank, dtype=np.float64)[::-1]


def transformed_cycle_residual(
    *,
    row: Mapping[str, Any],
    cycle_map: Mapping[str, Mapping[str, Any]],
    edge_map: Mapping[str, Mapping[str, Any]],
    transport_matrices: np.ndarray,
) -> float | None:
    if str(row["holonomy_status"]) != "defined":
        return None
    rank = int(row["holonomy_rank"])
    if rank <= 0:
        return None
    cycle = cycle_map[str(row["cycle_id"])]
    ordered_edges = reconstruct_ordered_edges(cycle=cycle, edge_map=edge_map)
    local_maps: List[np.ndarray] = []
    for edge in ordered_edges:
        operator_index = int(edge["operator_array_index"])
        matrix = np.asarray(transport_matrices[operator_index], dtype=np.float64)[:rank, :rank]
        q_source = reversal_matrix(rank)
        q_target = reversal_matrix(rank)
        local_maps.append(q_target.T @ matrix @ q_source)
    product = local_maps[2] @ local_maps[1] @ local_maps[0]
    return float(np.linalg.norm(product - np.eye(rank, dtype=np.float64), ord="fro"))


def build_gauge_stability(
    *,
    joined_rows: Sequence[Mapping[str, Any]],
    gate12a_payload: Mapping[str, Any],
    gate12a_dir: Path,
    flat_cut: float,
    high_cut: float,
    tau_gauge_residual_delta: float,
    observer_modes: Sequence[str],
) -> Dict[str, Any]:
    arrays_available, transport_matrices = load_optional_transport_matrices(gate12a_dir)
    gauge_transform = GAUGE_TRANSFORM_WITH_ARRAYS if arrays_available else GAUGE_TRANSFORM_REGISTRY_ONLY
    predicates = observer_predicates()
    cycle_map = gate12a_payload["cycle_map"]
    edge_map = gate12a_payload["edge_map"]

    if not arrays_available:
        return {
            "rows": [],
            "summary": {
                "arrays_available": False,
                "nontrivial_transform_evaluated": False,
                "gauge_transform": gauge_transform,
                "gauge_language_boundary": GAUGE_LANGUAGE_BOUNDARY,
                "total_check_count": 0,
                "stable_check_count": 0,
                "unstable_check_count": 0,
                "band_stable_count": 0,
                "max_residual_delta_abs": None,
                "tau_gauge_residual_delta": float(tau_gauge_residual_delta),
                "skipped_reason": "transport_operator_arrays_missing",
            },
        }

    assert transport_matrices is not None
    residual_after_transform: Dict[str, float | None] = {}
    for row in joined_rows:
        cycle_id = str(row["cycle_id"])
        residual_after_transform[cycle_id] = transformed_cycle_residual(
            row=row,
            cycle_map=cycle_map,
            edge_map=edge_map,
            transport_matrices=transport_matrices,
        )

    stability_rows: List[Dict[str, Any]] = []
    for observer in observer_modes:
        observer_rows = [row for row in joined_rows if predicates[observer](row)]
        for scale in SCALE_MODES:
            for row in observer_rows:
                if row["holonomy_status"] != "defined" or row["holonomy_residual_fro"] is None:
                    continue
                pre_residual = float(row["holonomy_residual_fro"])
                post_residual = residual_after_transform[str(row["cycle_id"])]
                if post_residual is None:
                    continue
                post_band = classify_band(post_residual, flat_cut=flat_cut, high_cut=high_cut)
                residual_delta_abs = abs(float(post_residual) - pre_residual)
                band_stable = str(row["residual_quantile_band"]) == post_band
                status_stable = residual_delta_abs <= float(tau_gauge_residual_delta)
                stability_rows.append(
                    {
                        "cycle_id": str(row["cycle_id"]),
                        "observer": observer,
                        "scale": scale,
                        "scale_key": scale_key(row, scale),
                        "gauge_transform": gauge_transform,
                        "pre_holonomy_status": str(row["holonomy_status"]),
                        "post_holonomy_status": str(row["holonomy_status"]),
                        "pre_band": str(row["residual_quantile_band"]),
                        "post_band": post_band,
                        "pre_residual_fro": pre_residual,
                        "post_residual_fro": float(post_residual),
                        "residual_delta_abs": residual_delta_abs,
                        "band_stable": band_stable,
                        "stable": bool(band_stable and status_stable),
                    }
                )

    deltas = [float(row["residual_delta_abs"]) for row in stability_rows]
    stable_count = sum(1 for row in stability_rows if bool(row["stable"]))
    summary = {
        "arrays_available": arrays_available,
        "nontrivial_transform_evaluated": True,
        "gauge_transform": gauge_transform,
        "gauge_language_boundary": GAUGE_LANGUAGE_BOUNDARY,
        "total_check_count": len(stability_rows),
        "stable_check_count": stable_count,
        "unstable_check_count": len(stability_rows) - stable_count,
        "band_stable_count": sum(1 for row in stability_rows if bool(row["band_stable"])),
        "max_residual_delta_abs": max(deltas) if deltas else None,
        "tau_gauge_residual_delta": float(tau_gauge_residual_delta),
    }
    return {
        "rows": stability_rows,
        "summary": summary,
    }


def build_gauge_variant_candidates(
    *,
    invariant_candidates: Sequence[Mapping[str, Any]],
    gauge_rows: Sequence[Mapping[str, Any]],
    gauge_summary: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    if not bool(gauge_summary.get("nontrivial_transform_evaluated")):
        return []

    gauge_by_cycle: Dict[str, List[Mapping[str, Any]]] = {}
    for row in gauge_rows:
        gauge_by_cycle.setdefault(str(row["cycle_id"]), []).append(row)

    candidates: List[Dict[str, Any]] = []
    for candidate in invariant_candidates:
        cycle_id = str(candidate["cycle_id"])
        rows = gauge_by_cycle.get(cycle_id, [])
        if not rows:
            continue
        stable_rows = [row for row in rows if bool(row["stable"])]
        if len(stable_rows) != len(rows):
            continue
        candidates.append(
            {
                "cycle_id": cycle_id,
                "candidate_kind": candidate["candidate_kind"],
                "candidate_status": "gauge_stable_candidate",
                "gauge_transform": rows[0]["gauge_transform"],
                "stable_check_count": len(stable_rows),
                "residual_delta_max": max(float(row["residual_delta_abs"]) for row in rows),
                "support_observers": candidate["support_observers"],
                "scale_keys": candidate["scale_keys"],
                "residual_quantile_band": candidate["residual_quantile_band"],
                "holonomy_residual_fro": candidate["holonomy_residual_fro"],
            }
        )
    return candidates


def build_readme(
    *,
    source_manifest: Mapping[str, Any],
    status_payload: Mapping[str, Any],
    invariant_candidate_count: int,
    gauge_candidate_count: int,
    observer_mode_set: str,
) -> str:
    return "\n".join(
        [
            "# Gate12B Observer-Relative Coarse-Grained Closure",
            "",
            f"- source Gate12A run: `{source_manifest.get('run_id', '')}`",
            f"- source Gate12A code commit: `{source_manifest.get('code_git_commit', '')}`",
            f"- secondary audit mode: `{SECONDARY_AUDIT_MODE}`",
            f"- primitive: `{PRIMITIVE_MODE}`",
            f"- observer mode set: `{observer_mode_set}`",
            f"- gauge language boundary: `{GAUGE_LANGUAGE_BOUNDARY}`",
            "",
            "This is a read-only secondary audit over existing Gate12A artifacts.",
            "It does not modify Gate12A semantics, thresholds, classifications, or artifact schemas.",
            "",
            "## Hypotheses",
            "",
            "- H1: Closure-defect magnitudes are observer-relative, but a subset of defect signatures remains stable under coarse-graining.",
            "- H1-A: Changing the observer changes the defect distribution.",
            "- H1-B: Coarse-graining reduces local variance but preserves conflict-aligned ordering.",
            "- H1-C: Invariant signature candidates survive across multiple observer x scale views.",
            "- H1-D: Projector-level closure signatures remain stable under admissible basis-preserving local reparameterizations.",
            "",
            "## Current Run",
            "",
            f"- input triangle rows: `{status_payload['input_triangle_count']}`",
            f"- defined triangle rows: `{status_payload['defined_triangle_count']}`",
            f"- observer-scale matrix rows: `{status_payload['observer_scale_matrix_row_count']}`",
            f"- invariant signature candidates: `{invariant_candidate_count}`",
            f"- gauge stability checks: `{status_payload['gauge_total_check_count']}`",
            f"- gauge-stable candidate rows: `{gauge_candidate_count}`",
            f"- flat cut: `{float(status_payload['flat_cut']):.6f}`",
            f"- high-tension cut: `{float(status_payload['high_cut']):.6f}`",
            "",
            "Invariant signature candidates require independent observer-scope support and support across multiple scale modes.",
            "Observer views with identical cycle membership are reported together but counted as one observer scope.",
            "Gauge-stable candidates are emitted only when a nontrivial array-level reparameterization was evaluated.",
            "",
            "## Reading Boundary",
            "",
            "- residual bands remain structural closure bands, not correctness labels",
            "- high residual is not treated as automatic failure",
            "- the output is not collapsed into one scalar score",
            "- admissible gauge transforms are limited to basis-preserving local reparameterization",
            "- external physics terminology remains outside this code surface",
            "",
        ]
    )


def build_checksums(out_dir: Path, included_files: Sequence[str]) -> Dict[str, str]:
    return {name: sha256_file(out_dir / name) for name in included_files}


MATRIX_FIELDNAMES = (
    "observer",
    "scale",
    "scale_key",
    "included_cycle_count",
    "defined_cycle_count",
    "undefined_cycle_count",
    "residual_min",
    "residual_median",
    "residual_mean",
    "residual_max",
    "flat_count",
    "tense_count",
    "high_tension_count",
    "undefined_count",
    "dominant_closure_band",
    "anchor_qualified_cycle_count",
    "residual_chord_cycle_count",
    "rank_patterns",
    "relation_kind_signatures",
)

GAUGE_FIELDNAMES = (
    "cycle_id",
    "observer",
    "scale",
    "scale_key",
    "gauge_transform",
    "pre_holonomy_status",
    "post_holonomy_status",
    "pre_band",
    "post_band",
    "pre_residual_fro",
    "post_residual_fro",
    "residual_delta_abs",
    "band_stable",
    "stable",
)


def run_observer_relative_coarse_grained_closure(
    *,
    gate12a_dir: Path,
    out_dir: Path,
    flat_quantile: float = 0.25,
    high_quantile: float = 0.75,
    top_k: int = 3,
    min_observer_support: int = 2,
    min_scale_support: int = 2,
    observer_mode_set: str = "core_v1",
    tau_gauge_residual_delta: float = 1.0e-8,
) -> Dict[str, Any]:
    gate12a_dir = Path(gate12a_dir)
    out_dir = Path(out_dir)
    validate_output_directory(gate12a_dir=gate12a_dir, out_dir=out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    observer_modes = observer_modes_for_set(observer_mode_set)

    gate12a_payload = load_gate12a_rows(gate12a_dir)
    joined_rows = gate12a_payload["joined_rows"]
    band_result = finalize_bands(
        joined_rows,
        flat_quantile=float(flat_quantile),
        high_quantile=float(high_quantile),
    )
    matrix_rows = build_observer_scale_matrix(joined_rows, observer_modes=observer_modes)
    invariant_candidates = build_invariant_candidates(
        joined_rows,
        top_k=int(top_k),
        min_observer_support=int(min_observer_support),
        min_scale_support=int(min_scale_support),
        observer_modes=observer_modes,
    )
    gauge_result = build_gauge_stability(
        joined_rows=joined_rows,
        gate12a_payload=gate12a_payload,
        gate12a_dir=gate12a_dir,
        flat_cut=float(band_result["flat_cut"]),
        high_cut=float(band_result["high_cut"]),
        tau_gauge_residual_delta=float(tau_gauge_residual_delta),
        observer_modes=observer_modes,
    )
    gauge_rows = gauge_result["rows"]
    gauge_summary = gauge_result["summary"]
    gauge_candidates = build_gauge_variant_candidates(
        invariant_candidates=invariant_candidates,
        gauge_rows=gauge_rows,
        gauge_summary=gauge_summary,
    )

    status_payload = {
        "input_triangle_count": len(joined_rows),
        "defined_triangle_count": sum(1 for row in joined_rows if row["holonomy_status"] == "defined"),
        "flat_cut": float(band_result["flat_cut"]),
        "high_cut": float(band_result["high_cut"]),
        "flat_count": int(band_result["band_counts"]["flat"]),
        "tense_count": int(band_result["band_counts"]["tense"]),
        "high_tension_count": int(band_result["band_counts"]["high_tension"]),
        "undefined_count": int(band_result["band_counts"]["undefined"]),
        "observer_scale_matrix_row_count": len(matrix_rows),
        "invariant_signature_candidate_count": len(invariant_candidates),
        "gauge_total_check_count": int(gauge_summary["total_check_count"]),
        "gauge_stable_check_count": int(gauge_summary["stable_check_count"]),
        "gauge_unstable_check_count": int(gauge_summary["unstable_check_count"]),
        "gauge_variant_signature_candidate_count": len(gauge_candidates),
        "gauge_arrays_available": bool(gauge_summary["arrays_available"]),
        "gauge_transform": str(gauge_summary["gauge_transform"]),
    }

    manifest_path = out_dir / DEFAULT_MANIFEST
    matrix_csv_path = out_dir / DEFAULT_MATRIX_CSV
    matrix_json_path = out_dir / DEFAULT_MATRIX_JSON
    invariant_candidates_path = out_dir / DEFAULT_INVARIANT_CANDIDATES
    gauge_matrix_path = out_dir / DEFAULT_GAUGE_MATRIX
    gauge_summary_path = out_dir / DEFAULT_GAUGE_SUMMARY
    gauge_candidates_path = out_dir / DEFAULT_GAUGE_CANDIDATES
    read_path = out_dir / DEFAULT_READ
    checksums_path = out_dir / DEFAULT_CHECKSUMS

    write_csv(matrix_csv_path, MATRIX_FIELDNAMES, matrix_rows)
    write_json(
        matrix_json_path,
        {
            "observer_mode_set": str(observer_mode_set),
            "observer_modes": list(observer_modes),
            "scale_modes": list(SCALE_MODES),
            "flat_cut": float(band_result["flat_cut"]),
            "high_cut": float(band_result["high_cut"]),
            "matrix_rows": matrix_rows,
        },
    )
    write_jsonl(invariant_candidates_path, invariant_candidates)
    write_csv(gauge_matrix_path, GAUGE_FIELDNAMES, gauge_rows)
    write_json(gauge_summary_path, gauge_summary)
    write_jsonl(gauge_candidates_path, gauge_candidates)
    write_text(
        read_path,
        build_readme(
            source_manifest=gate12a_payload["manifest"],
            status_payload=status_payload,
            invariant_candidate_count=len(invariant_candidates),
            gauge_candidate_count=len(gauge_candidates),
            observer_mode_set=str(observer_mode_set),
        ),
    )

    manifest = {
        "run_id": out_dir.name,
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "code_git_commit": current_git_commit(),
        "builder_script_sha256": sha256_file(SCRIPT_PATH),
        "secondary_audit_mode": SECONDARY_AUDIT_MODE,
        "primitive_mode": PRIMITIVE_MODE,
        "gauge_language_boundary": GAUGE_LANGUAGE_BOUNDARY,
        "observer_mode_set": str(observer_mode_set),
        "source_gate12a_manifest_path": repo_relative_or_posix(gate12a_dir / DEFAULT_MANIFEST),
        "source_gate12a_run_id": str(gate12a_payload["manifest"].get("run_id") or ""),
        "source_gate12a_code_git_commit": str(gate12a_payload["manifest"].get("code_git_commit") or ""),
        "observer_modes": list(observer_modes),
        "scale_modes": list(SCALE_MODES),
        "gauge_transform_modes": [str(gauge_summary["gauge_transform"])],
        "flat_quantile": float(flat_quantile),
        "high_quantile": float(high_quantile),
        "top_k": int(top_k),
        "min_observer_support": int(min_observer_support),
        "min_scale_support": int(min_scale_support),
        "tau_gauge_residual_delta": float(tau_gauge_residual_delta),
        "paths": {
            DEFAULT_MATRIX_CSV: repo_relative_or_posix(matrix_csv_path),
            DEFAULT_MATRIX_JSON: repo_relative_or_posix(matrix_json_path),
            DEFAULT_INVARIANT_CANDIDATES: repo_relative_or_posix(invariant_candidates_path),
            DEFAULT_GAUGE_MATRIX: repo_relative_or_posix(gauge_matrix_path),
            DEFAULT_GAUGE_SUMMARY: repo_relative_or_posix(gauge_summary_path),
            DEFAULT_GAUGE_CANDIDATES: repo_relative_or_posix(gauge_candidates_path),
            DEFAULT_READ: repo_relative_or_posix(read_path),
        },
        "status": status_payload,
    }
    write_json(manifest_path, manifest)
    included_files = (
        DEFAULT_MANIFEST,
        DEFAULT_MATRIX_CSV,
        DEFAULT_MATRIX_JSON,
        DEFAULT_INVARIANT_CANDIDATES,
        DEFAULT_GAUGE_MATRIX,
        DEFAULT_GAUGE_SUMMARY,
        DEFAULT_GAUGE_CANDIDATES,
        DEFAULT_READ,
    )
    write_json(checksums_path, build_checksums(out_dir, included_files))

    return {
        "manifest": manifest,
        "status": status_payload,
        "matrix_rows": matrix_rows,
        "invariant_candidates": invariant_candidates,
        "gauge_rows": gauge_rows,
        "gauge_summary": gauge_summary,
        "gauge_candidates": gauge_candidates,
    }


def main() -> int:
    args = parse_args()
    run_observer_relative_coarse_grained_closure(
        gate12a_dir=Path(args.gate12a_dir),
        out_dir=Path(args.out_dir),
        flat_quantile=float(args.flat_quantile),
        high_quantile=float(args.high_quantile),
        top_k=int(args.top_k),
        min_observer_support=int(args.min_observer_support),
        min_scale_support=int(args.min_scale_support),
        observer_mode_set=str(args.observer_mode_set),
        tau_gauge_residual_delta=float(args.tau_gauge_residual_delta),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
