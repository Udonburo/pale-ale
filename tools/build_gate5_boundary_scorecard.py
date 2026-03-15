#!/usr/bin/env python3
"""Build a fixed Gate5 boundary standing scorecard from aggregate reports."""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
METRIC_LINE_RE = re.compile(r"^- ([A-Za-z0-9_]+): ([^\r\n]+)$")
FWHT_BASELINE_PROJ_ID = "fwht_pad_pow2_take8_v1"

SURFACE_TITLES = {
    "cfa": "CFA",
    "seam": "Seam Challenge v0",
}

SURFACE_COLUMNS = {
    "cfa": [
        ("global_auprc_F", "global_auprc_F"),
        ("global_auprc_rotor", "global_auprc_rotor_loop_chordal_v1"),
        ("mean_sample_auprc_F", "mean_sample_auprc_F"),
        ("mean_sample_auprc_rotor", "mean_sample_auprc_rotor_loop_chordal_v1"),
        ("mean_hit@10_F", "mean_sample_hit_at_10_F"),
        ("mean_hit@10_rotor", "mean_sample_hit_at_10_rotor"),
        ("mean_first_hit_F", "mean_first_hit_distance_F"),
        ("mean_first_hit_rotor", "mean_first_hit_distance_rotor"),
    ],
    "seam": [
        ("mean_delta_max_F", "mean_delta_max_F"),
        ("mean_delta_max_rotor", "mean_delta_max_rotor"),
        ("mean_delta_p90_F", "mean_delta_p90_F"),
        ("mean_delta_p90_rotor", "mean_delta_p90_rotor"),
        ("mean_iqr_delta_max_F", "mean_iqr_normalized_delta_max_F"),
        ("mean_iqr_delta_max_rotor", "mean_iqr_normalized_delta_max_rotor"),
        ("mean_top10_inflation_F", "mean_top10_inflation_F_vs_clean_p90"),
        ("mean_top10_inflation_rotor", "mean_top10_inflation_rotor_vs_clean_p90"),
    ],
}

MANIFEST_MATCH_FIELDS = (
    "spec_version",
    "spec_hash_blake3",
    "method_id",
    "evaluation_mode_id",
    "model_id",
    "model_revision",
    "seed",
    "perm_r",
    "splus_def_id",
    "sminus_def_id",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a fixed boundary standing scorecard.")
    parser.add_argument("--surface", choices=("cfa", "seam"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv-out", default="")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help=(
            "Run spec in 'label=<name>;gate5_out=<dir>;input=<gate4_input.json>"
            "[;boundary_manifest=<manifest.json>]' form."
        ),
    )
    return parser.parse_args()


def resolve_repo_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_sample_ids(input_path: Path) -> List[int]:
    payload = read_json(input_path)
    samples = payload.get("samples", [])
    return [int(sample["sample_id"]) for sample in samples]


def sample_id_digest(sample_ids: Sequence[int]) -> str:
    encoded = json.dumps(list(sample_ids), separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_report_surface(report_text: str) -> Optional[str]:
    for line in report_text.splitlines():
        if line.startswith("Surface: "):
            return line[len("Surface: ") :].strip()
    return None


def parse_report_metrics(report_text: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for line in report_text.splitlines():
        match = METRIC_LINE_RE.match(line.strip())
        if not match:
            continue
        key = match.group(1)
        raw_value = match.group(2).strip()
        try:
            metrics[key] = float(raw_value)
        except ValueError:
            continue
    return metrics


def render_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


def render_int(value: Optional[int]) -> str:
    if value is None:
        return ""
    return str(int(value))


def parse_run_spec(raw: str) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for chunk in raw.split(";"):
        item = chunk.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"invalid --run item {item!r}; expected key=value")
        key, value = item.split("=", 1)
        parsed[key.strip()] = value.strip()
    required = ("label", "gate5_out", "input")
    missing = [key for key in required if not parsed.get(key)]
    if missing:
        raise ValueError(f"--run is missing required keys: {', '.join(missing)}")
    return parsed


def encode_for_compare(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True)


def manifest_mismatches(run_rows: Sequence[Dict[str, Any]]) -> List[str]:
    mismatches: List[str] = []
    for field in MANIFEST_MATCH_FIELDS:
        encoded_values = {
            encode_for_compare(row["manifest"].get(field))
            for row in run_rows
        }
        if len(encoded_values) > 1:
            mismatches.append(field)
    return mismatches


def boundary_candidate_rows(run_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in run_rows if row["proj_id"] != FWHT_BASELINE_PROJ_ID]


def missing_boundary_manifest_labels(run_rows: Sequence[Dict[str, Any]]) -> List[str]:
    return [
        str(row["label"])
        for row in boundary_candidate_rows(run_rows)
        if row.get("boundary_manifest") is None
    ]


def native_root_mismatches(run_rows: Sequence[Dict[str, Any]]) -> List[str]:
    native_roots = [
        str(row["boundary_manifest"].get("samples_root"))
        for row in boundary_candidate_rows(run_rows)
        if row.get("boundary_manifest") is not None
    ]
    if len(native_roots) < 2:
        return []
    return sorted(set(native_roots)) if len(set(native_roots)) > 1 else []


def boundary_sample_id_mismatches(run_rows: Sequence[Dict[str, Any]]) -> List[str]:
    mismatches: List[str] = []
    for row in boundary_candidate_rows(run_rows):
        boundary_manifest = row.get("boundary_manifest")
        if boundary_manifest is None:
            continue
        boundary_ids = [int(value) for value in boundary_manifest.get("sample_ids", [])]
        if boundary_ids != row["sample_ids"]:
            mismatches.append(str(row["label"]))
    return mismatches


def loop_row_coverage_match(run_rows: Sequence[Dict[str, Any]]) -> bool:
    token_totals = {int(row["n_token_rows_total"]) for row in run_rows}
    valid_counts = {int(row["n_loop_rows_valid"]) for row in run_rows}
    missing_counts = [int(row["n_loop_rows_missing"]) for row in run_rows]
    return len(token_totals) == 1 and len(valid_counts) == 1 and all(value == 0 for value in missing_counts)


def render_loop_row_coverage_detail(run_rows: Sequence[Dict[str, Any]]) -> str:
    return ", ".join(
        (
            f"{row['label']}(token_total={row['n_token_rows_total']},"
            f"loop_valid={row['n_loop_rows_valid']},missing={row['n_loop_rows_missing']})"
        )
        for row in run_rows
    )


def build_run_row(spec: Dict[str, str], surface: str) -> Dict[str, Any]:
    gate5_out = resolve_repo_path(spec["gate5_out"])
    input_path = resolve_repo_path(spec["input"])
    report_path = gate5_out / "gate5_aggregate_report.md"
    manifest_path = gate5_out / "manifest.json"
    report_text = report_path.read_text(encoding="utf-8")
    report_surface = parse_report_surface(report_text)
    expected_surface = SURFACE_TITLES[surface]
    if report_surface != expected_surface:
        raise ValueError(
            f"{report_path} surface mismatch: expected {expected_surface!r}, got {report_surface!r}"
        )

    manifest = read_json(manifest_path)
    sample_ids = load_sample_ids(input_path)
    boundary_manifest = None
    if spec.get("boundary_manifest"):
        boundary_manifest = read_json(resolve_repo_path(spec["boundary_manifest"]))

    metrics = parse_report_metrics(report_text)
    row: Dict[str, Any] = {
        "label": spec["label"],
        "gate5_out": gate5_out,
        "input_path": input_path,
        "manifest": manifest,
        "metrics": metrics,
        "sample_ids": sample_ids,
        "sample_id_digest": sample_id_digest(sample_ids),
        "boundary_manifest": boundary_manifest,
    }
    row["proj_id"] = str(manifest.get("proj_id", ""))
    row["n_samples_total"] = int(manifest.get("n_samples_total", len(sample_ids)))
    row["n_token_rows_total"] = int(manifest.get("n_token_rows_total", 0))
    row["n_loop_rows_valid"] = int(manifest.get("n_loop_rows_valid", 0))
    row["n_loop_rows_missing"] = int(manifest.get("n_loop_rows_missing", 0))
    row["boundary_materialized_rank3"] = None
    row["boundary_sign_unstable"] = None
    if boundary_manifest is not None:
        outcome_counts = boundary_manifest.get("boundary_outcome_counts", {})
        row["boundary_materialized_rank3"] = int(outcome_counts.get("materialized_rank3", 0))
        row["boundary_sign_unstable"] = int(outcome_counts.get("sign_unstable", 0))
    for header, metric_key in SURFACE_COLUMNS[surface]:
        row[header] = metrics.get(metric_key)
    return row


def build_markdown(surface: str, run_rows: Sequence[Dict[str, Any]]) -> str:
    baseline_ids = run_rows[0]["sample_ids"]
    exact_sample_match = all(row["sample_ids"] == baseline_ids for row in run_rows[1:])
    manifest_field_mismatches = manifest_mismatches(run_rows)
    missing_boundary_manifests = missing_boundary_manifest_labels(run_rows)
    if missing_boundary_manifests:
        raise ValueError(
            "boundary_manifest is required for non-baseline runs: "
            + ", ".join(missing_boundary_manifests)
        )
    boundary_id_mismatches = boundary_sample_id_mismatches(run_rows)
    native_root_conflicts = native_root_mismatches(run_rows)
    coverage_match = loop_row_coverage_match(run_rows)
    common_digest = run_rows[0]["sample_id_digest"] if exact_sample_match else "mismatch"

    headers = [
        "run",
        "proj_id",
        "n_samples",
        "n_loop_missing",
        "boundary_rank3",
        "boundary_sign_unstable",
    ] + [header for header, _metric_key in SURFACE_COLUMNS[surface]]

    lines = [
        "# Gate5 Boundary Standing Scorecard",
        "",
        f"Surface: {SURFACE_TITLES[surface]}",
        "",
        "## Comparability Checks",
        "",
        f"- exact_sample_ids_match: {'PASS' if exact_sample_match else 'FAIL'} "
        f"(n={len(run_rows[0]['sample_ids'])}, sample_id_sha256={common_digest})",
        f"- gate5_fixed_fields_match: {'PASS' if not manifest_field_mismatches else 'FAIL'} "
        f"(fields={','.join(MANIFEST_MATCH_FIELDS)})",
        f"- loop_row_coverage_match: {'PASS' if coverage_match else 'FAIL'}",
    ]
    if manifest_field_mismatches:
        lines.append(f"- gate5_fixed_fields_mismatch_detail: {', '.join(manifest_field_mismatches)}")
    if not coverage_match:
        lines.append(f"- loop_row_coverage_detail: {render_loop_row_coverage_detail(run_rows)}")
    lines.append(
        f"- boundary_manifest_sample_ids_match: {'PASS' if not boundary_id_mismatches else 'FAIL'}"
    )
    if boundary_id_mismatches:
        lines.append(
            f"- boundary_manifest_sample_ids_mismatch_runs: {', '.join(boundary_id_mismatches)}"
        )
    if native_root_conflicts:
        lines.append("- native_samples_root_match: FAIL")
        lines.append(f"- native_samples_root_values: {', '.join(native_root_conflicts)}")
    else:
        native_roots = [
            str(row["boundary_manifest"].get("samples_root"))
            for row in boundary_candidate_rows(run_rows)
            if row.get("boundary_manifest") is not None
        ]
        if native_roots:
            lines.append(f"- native_samples_root_match: PASS ({native_roots[0]})")
        else:
            lines.append("- native_samples_root_match: N/A")

    lines.extend(
        [
            "",
            "## Headline Table",
            "",
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
    )

    for row in run_rows:
        cells = [
            str(row["label"]),
            str(row["proj_id"]),
            render_int(row["n_samples_total"]),
            render_int(row["n_loop_rows_missing"]),
            render_int(row["boundary_materialized_rank3"]),
            render_int(row["boundary_sign_unstable"]),
        ]
        for header, _metric_key in SURFACE_COLUMNS[surface]:
            cells.append(render_float(row.get(header)))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_csv_rows(surface: str, run_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in run_rows:
        encoded = {
            "run": row["label"],
            "proj_id": row["proj_id"],
            "n_samples": row["n_samples_total"],
            "n_token_rows_total": row["n_token_rows_total"],
            "n_loop_valid": row["n_loop_rows_valid"],
            "n_loop_missing": row["n_loop_rows_missing"],
            "sample_id_sha256": row["sample_id_digest"],
            "boundary_rank3": row["boundary_materialized_rank3"],
            "boundary_sign_unstable": row["boundary_sign_unstable"],
        }
        for header, _metric_key in SURFACE_COLUMNS[surface]:
            encoded[header] = row.get(header)
        rows.append(encoded)
    return rows


def main() -> int:
    args = parse_args()
    run_rows = [build_run_row(parse_run_spec(raw), surface=args.surface) for raw in args.run]

    out_path = resolve_repo_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_markdown(args.surface, run_rows), encoding="utf-8", newline="\n")

    if args.csv_out:
        csv_path = resolve_repo_path(args.csv_out)
        csv_rows = build_csv_rows(args.surface, run_rows)
        fieldnames = list(csv_rows[0].keys()) if csv_rows else []
        write_csv(csv_path, fieldnames, csv_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
