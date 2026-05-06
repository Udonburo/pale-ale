#!/usr/bin/env python3
"""Summarize Gate12B observer-relative closure run directories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
GATE12B_RUNNER = REPO_ROOT / "tools" / "run_gate12b_observer_relative_coarse_grained_closure.py"

DEFAULT_MANIFEST = "manifest.json"
DEFAULT_CANDIDATES = "invariant_signature_candidates.jsonl"
DEFAULT_GAUGE_SUMMARY = "gauge_stability_summary.json"
DEFAULT_CHECKSUMS = "checksums.json"

DEFAULT_SUMMARY_CSV = "gate12b_run_summary.csv"
DEFAULT_SUMMARY_JSON = "gate12b_run_summary.json"
DEFAULT_SUMMARY_MANIFEST = "manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize existing Gate12B run directories.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-dir", action="append", default=[], help="Gate12B run directory. Repeatable.")
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if raw:
            rows.append(json.loads(raw))
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def candidate_band(candidate_kind: str) -> str:
    if candidate_kind.startswith("flat_"):
        return "flat"
    if candidate_kind.startswith("high_tension_"):
        return "high_tension"
    return candidate_kind


def format_counter(counter: Counter[Any]) -> str:
    return "|".join(f"{key}:{counter[key]}" for key in sorted(counter, key=str))


def dominant_counter(counter: Counter[str]) -> Tuple[str, int]:
    if not counter:
        return "", 0
    signature, count = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0]
    return signature, count


def checksum_status(run_dir: Path) -> Tuple[str, int, int]:
    checksums_path = run_dir / DEFAULT_CHECKSUMS
    if not checksums_path.exists():
        return "missing", 0, 0
    checksums = read_json(checksums_path)
    checked = 0
    mismatches = 0
    for name, expected in checksums.items():
        path = run_dir / str(name)
        if not path.exists():
            mismatches += 1
            continue
        checked += 1
        if sha256_file(path) != expected:
            mismatches += 1
    return ("ok" if mismatches == 0 else "mismatch"), checked, mismatches


def summarize_run(run_dir: Path, current_runner_sha256: str) -> Dict[str, Any]:
    manifest = read_json(run_dir / DEFAULT_MANIFEST)
    candidates = read_jsonl(run_dir / DEFAULT_CANDIDATES)
    gauge_summary = read_json(run_dir / DEFAULT_GAUGE_SUMMARY)
    status = manifest.get("status", {})

    by_band = Counter(candidate_band(str(row.get("candidate_kind", ""))) for row in candidates)
    support_observer = Counter(int(row.get("observer_support_count", 0)) for row in candidates)
    support_scale = Counter(int(row.get("scale_support_count", 0)) for row in candidates)
    relation_by_band: Dict[str, Counter[str]] = {
        "flat": Counter(),
        "high_tension": Counter(),
    }
    for row in candidates:
        band = candidate_band(str(row.get("candidate_kind", "")))
        relation = str(row.get("relation_kind_signature", ""))
        if band in relation_by_band and relation:
            relation_by_band[band][relation] += 1

    flat_signature, flat_signature_count = dominant_counter(relation_by_band["flat"])
    high_signature, high_signature_count = dominant_counter(relation_by_band["high_tension"])
    checksum_state, checksum_checked, checksum_mismatches = checksum_status(run_dir)
    builder_sha = str(manifest.get("builder_script_sha256", ""))

    return {
        "run_dir": str(run_dir.as_posix()),
        "run_id": str(manifest.get("run_id", run_dir.name)),
        "source_gate12a_run_id": str(manifest.get("source_gate12a_run_id", "")),
        "observer_mode_set": str(manifest.get("observer_mode_set", "")),
        "top_k": int(manifest.get("top_k", 0)),
        "min_observer_support": int(manifest.get("min_observer_support", 0)),
        "min_scale_support": int(manifest.get("min_scale_support", 0)),
        "flat_quantile": manifest.get("flat_quantile"),
        "high_quantile": manifest.get("high_quantile"),
        "candidate_total": len(candidates),
        "flat_candidate_count": by_band.get("flat", 0),
        "high_tension_candidate_count": by_band.get("high_tension", 0),
        "dominant_flat_relation_signature": flat_signature,
        "dominant_flat_relation_signature_count": flat_signature_count,
        "dominant_high_tension_relation_signature": high_signature,
        "dominant_high_tension_relation_signature_count": high_signature_count,
        "flat_relation_signature_counts": dict(sorted(relation_by_band["flat"].items())),
        "high_tension_relation_signature_counts": dict(sorted(relation_by_band["high_tension"].items())),
        "observer_support_distribution": format_counter(support_observer),
        "scale_support_distribution": format_counter(support_scale),
        "gauge_total_check_count": int(gauge_summary.get("total_check_count", status.get("gauge_total_check_count", 0))),
        "gauge_unstable_check_count": int(
            gauge_summary.get("unstable_check_count", status.get("gauge_unstable_check_count", 0))
        ),
        "gauge_variant_signature_candidate_count": int(status.get("gauge_variant_signature_candidate_count", 0)),
        "gauge_max_residual_delta_abs": gauge_summary.get("max_residual_delta_abs"),
        "builder_script_sha256": builder_sha,
        "builder_script_sha256_matches_current": bool(builder_sha and builder_sha == current_runner_sha256),
        "checksum_status": checksum_state,
        "checksum_checked_count": checksum_checked,
        "checksum_mismatch_count": checksum_mismatches,
    }


SUMMARY_FIELDS = (
    "run_dir",
    "run_id",
    "source_gate12a_run_id",
    "observer_mode_set",
    "top_k",
    "min_observer_support",
    "min_scale_support",
    "flat_quantile",
    "high_quantile",
    "candidate_total",
    "flat_candidate_count",
    "high_tension_candidate_count",
    "dominant_flat_relation_signature",
    "dominant_flat_relation_signature_count",
    "dominant_high_tension_relation_signature",
    "dominant_high_tension_relation_signature_count",
    "flat_relation_signature_counts",
    "high_tension_relation_signature_counts",
    "observer_support_distribution",
    "scale_support_distribution",
    "gauge_total_check_count",
    "gauge_unstable_check_count",
    "gauge_variant_signature_candidate_count",
    "gauge_max_residual_delta_abs",
    "builder_script_sha256",
    "builder_script_sha256_matches_current",
    "checksum_status",
    "checksum_checked_count",
    "checksum_mismatch_count",
)


def summarize_gate12b_runs(run_dirs: Sequence[Path], out_dir: Path) -> Dict[str, Any]:
    resolved_run_dirs = [Path(run_dir).resolve() for run_dir in run_dirs]
    if not resolved_run_dirs:
        raise ValueError("at least one --run-dir is required")
    current_runner_sha256 = sha256_file(GATE12B_RUNNER) if GATE12B_RUNNER.exists() else ""
    rows = [summarize_run(run_dir, current_runner_sha256) for run_dir in resolved_run_dirs]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / DEFAULT_SUMMARY_CSV
    summary_json = out_dir / DEFAULT_SUMMARY_JSON
    manifest_path = out_dir / DEFAULT_SUMMARY_MANIFEST

    write_csv(summary_csv, SUMMARY_FIELDS, rows)
    write_json(summary_json, {"run_count": len(rows), "rows": rows})
    manifest = {
        "schema_version": "gate12b_run_summary_v1",
        "run_count": len(rows),
        "current_gate12b_runner_sha256": current_runner_sha256,
        "paths": {
            DEFAULT_SUMMARY_CSV: str(summary_csv.as_posix()),
            DEFAULT_SUMMARY_JSON: str(summary_json.as_posix()),
        },
    }
    write_json(manifest_path, manifest)
    return {"manifest": manifest, "rows": rows, "out_dir": str(out_dir.as_posix())}


def main() -> None:
    args = parse_args()
    result = summarize_gate12b_runs([Path(path) for path in args.run_dir], Path(args.out_dir))
    print(f"run_count={result['manifest']['run_count']}")
    print(f"summary_csv={Path(result['out_dir']) / DEFAULT_SUMMARY_CSV}")
    print(f"summary_json={Path(result['out_dir']) / DEFAULT_SUMMARY_JSON}")


if __name__ == "__main__":
    main()
