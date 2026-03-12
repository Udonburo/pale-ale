#!/usr/bin/env python3
"""Diagnose where a boundary candidate goes flat before or inside Gate5."""

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run matched baseline/candidate boundary diagnostics under the fixed Gate5 comparator."
        )
    )
    parser.add_argument("--baseline-input", required=True)
    parser.add_argument("--candidate-input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--candidate-boundary-steps", default="")
    parser.add_argument("--baseline-boundary-steps", default="")
    parser.add_argument("--cli-path", default="")
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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object: {path}")
    return obj


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                raise ValueError(f"non-object row at {path}:{line_no}")
            rows.append(obj)
    return rows


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8", newline="\n")


def parse_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    raw = str(value).strip()
    if raw == "":
        return None
    return float(raw)


def l2_norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def dot_abs_clamped(left: Sequence[float], right: Sequence[float]) -> float:
    inner = sum(float(a) * float(b) for a, b in zip(left, right))
    return min(1.0, abs(inner))


def chordal(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(max(0.0, 2.0 * (1.0 - dot_abs_clamped(left, right))))


def centroid(values: Sequence[Sequence[float]]) -> List[float]:
    width = len(values[0])
    out = [0.0] * width
    for value in values:
        for idx, entry in enumerate(value):
            out[idx] += float(entry)
    inv = 1.0 / float(len(values))
    return [entry * inv for entry in out]


def centered_energy(values: Sequence[Sequence[float]], center: Sequence[float]) -> float:
    return sum(
        l2_norm([float(a) - float(b) for a, b in zip(value, center)]) for value in values
    ) / float(len(values))


def load_boundary_steps(path: Optional[Path]) -> Dict[Tuple[int, int], Dict[str, Any]]:
    if path is None:
        return {}
    rows = load_jsonl(path)
    return {(int(row["sample_id"]), int(row["step"])): row for row in rows}


def build_boundary_rows(
    gate4_input_path: Path, boundary_steps: Mapping[Tuple[int, int], Dict[str, Any]]
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    payload = load_json(gate4_input_path)
    out: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for sample in payload["samples"]:
        sample_id = int(sample["sample_id"])
        for step in sample["token_steps"]:
            step_id = int(step["step"])
            key = (sample_id, step_id)
            v = [float(x) for x in step["V_8d"]]
            splus = [float(x) for x in step["Splus_8d"]]
            sminus = [float(x) for x in step["Sminus_8d"]]
            center = centroid([v, splus, sminus])
            row = {
                "sample_id": sample_id,
                "step": step_id,
                "token_text": str(step["token_str"]),
                "label_token": int(step["label_token"]),
                "emitted_norm_v": l2_norm(v),
                "emitted_norm_splus": l2_norm(splus),
                "emitted_norm_sminus": l2_norm(sminus),
                "emitted_dot_v_splus": dot_abs_clamped(v, splus),
                "emitted_dot_splus_sminus": dot_abs_clamped(splus, sminus),
                "emitted_dot_sminus_v": dot_abs_clamped(sminus, v),
                "emitted_chordal_v_splus": chordal(v, splus),
                "emitted_chordal_splus_sminus": chordal(splus, sminus),
                "emitted_chordal_sminus_v": chordal(sminus, v),
                "emitted_coord_centroid_norm": l2_norm(center),
                "emitted_centered_energy": centered_energy([v, splus, sminus], center),
                "frame_rank": None,
                "boundary_outcome": None,
                "projected_norm_v": None,
                "projected_norm_splus": None,
                "projected_norm_sminus": None,
                "raw_triplet_centroid_norm": None,
                "boundary_id": None,
                "coordinate_rule_id": None,
            }
            if key in boundary_steps:
                extra = boundary_steps[key]
                row.update(
                    {
                        "frame_rank": extra.get("frame_rank"),
                        "boundary_outcome": extra.get("boundary_outcome"),
                        "projected_norm_v": extra.get("projected_norm_v"),
                        "projected_norm_splus": extra.get("projected_norm_splus"),
                        "projected_norm_sminus": extra.get("projected_norm_sminus"),
                        "raw_triplet_centroid_norm": extra.get("raw_triplet_centroid_norm"),
                        "boundary_id": extra.get("boundary_id"),
                        "coordinate_rule_id": extra.get("coordinate_rule_id"),
                    }
                )
            out[key] = row
    return out


def run_gate5_diagnose(cli_path: Path, gate4_input_path: Path, out_csv_path: Path) -> Path:
    run_command(
        [
            str(cli_path),
            "gate5",
            "diagnose",
            "--input",
            str(gate4_input_path.resolve()),
            "--out",
            str(out_csv_path.resolve()),
        ]
    )
    return out_csv_path


def load_diag_rows(path: Path) -> Dict[Tuple[int, int], Dict[str, Any]]:
    out: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for row in read_csv_rows(path):
        key = (int(row["sample_id"]), int(row["step"]))
        converted: Dict[str, Any] = {
            "sample_id": key[0],
            "step": key[1],
            "absolute_pos": int(row["absolute_pos"]),
            "token_id": int(row["token_id"]),
            "token_text": row["token_text"],
            "label_token": int(row["label_token"]),
        }
        for key_name, value in row.items():
            if key_name in converted:
                continue
            if (
                key_name.startswith("norm_status_")
                or "outcome" in key_name
                or key_name == "token_text"
            ):
                converted[key_name] = value
            else:
                converted[key_name] = parse_optional_float(value)
        out[key] = converted
    return out


def safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    items = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not items:
        return None
    return sum(items) / float(len(items))


def quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("quantile requires non-empty input")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def iqr(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return quantile(ordered, 0.75) - quantile(ordered, 0.25)


def build_sample_transport_summary(
    diag_rows: Mapping[Tuple[int, int], Dict[str, Any]]
) -> List[Dict[str, Any]]:
    sample_to_values: Dict[int, List[float]] = {}
    for (sample_id, _), row in diag_rows.items():
        value = row.get("rotor_loop_chordal_v1")
        if value is not None:
            sample_to_values.setdefault(sample_id, []).append(float(value))

    out: List[Dict[str, Any]] = []
    for sample_id in sorted(sample_to_values):
        values = sample_to_values[sample_id]
        unique_count = len({round(value, 15) for value in values})
        zero_iqr = iqr(values) <= 1e-15
        adjacent_ties = sum(
            1 for left, right in zip(values, values[1:]) if abs(left - right) <= 1e-15
        )
        tie_rate = adjacent_ties / float(max(1, len(values) - 1))
        out.append(
            {
                "sample_id": sample_id,
                "n_loop_values": len(values),
                "loop_unique_value_count": unique_count,
                "loop_zero_iqr": int(zero_iqr),
                "loop_iqr": iqr(values),
                "loop_adjacent_tie_rate": tie_rate,
            }
        )
    return out


def join_rows(
    baseline_boundary: Mapping[Tuple[int, int], Dict[str, Any]],
    baseline_diag: Mapping[Tuple[int, int], Dict[str, Any]],
    candidate_boundary: Mapping[Tuple[int, int], Dict[str, Any]],
    candidate_diag: Mapping[Tuple[int, int], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    keys = sorted(set(baseline_diag) & set(candidate_diag))
    rows: List[Dict[str, Any]] = []
    for key in keys:
        left_diag = baseline_diag[key]
        right_diag = candidate_diag[key]
        left_boundary = baseline_boundary.get(key, {})
        right_boundary = candidate_boundary.get(key, {})
        row: Dict[str, Any] = {
            "sample_id": key[0],
            "step": key[1],
            "token_text": left_diag["token_text"],
            "label_token": left_diag["label_token"],
        }
        for prefix, source in (
            ("baseline_boundary", left_boundary),
            ("baseline_diag", left_diag),
            ("candidate_boundary", right_boundary),
            ("candidate_diag", right_diag),
        ):
            for field, value in source.items():
                if field in ("sample_id", "step", "token_text", "label_token"):
                    continue
                row[f"{prefix}_{field}"] = value
        rows.append(row)
    return rows


def format_optional(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{value:.6f}"


def build_report(
    matched_rows: Sequence[Dict[str, Any]],
    baseline_sample_summary: Sequence[Dict[str, Any]],
    candidate_sample_summary: Sequence[Dict[str, Any]],
) -> str:
    baseline_zero_iqr = sum(int(row["loop_zero_iqr"]) for row in baseline_sample_summary)
    candidate_zero_iqr = sum(int(row["loop_zero_iqr"]) for row in candidate_sample_summary)
    baseline_unique_mean = safe_mean(
        float(row["loop_unique_value_count"]) for row in baseline_sample_summary
    )
    candidate_unique_mean = safe_mean(
        float(row["loop_unique_value_count"]) for row in candidate_sample_summary
    )
    baseline_tie_rate_mean = safe_mean(
        float(row["loop_adjacent_tie_rate"]) for row in baseline_sample_summary
    )
    candidate_tie_rate_mean = safe_mean(
        float(row["loop_adjacent_tie_rate"]) for row in candidate_sample_summary
    )
    baseline_loop_mean = safe_mean(
        parse_optional_float(row.get("baseline_diag_rotor_loop_chordal_v1")) for row in matched_rows
    )
    candidate_loop_mean = safe_mean(
        parse_optional_float(row.get("candidate_diag_rotor_loop_chordal_v1")) for row in matched_rows
    )
    baseline_edge_mean = safe_mean(
        parse_optional_float(row.get("baseline_diag_edge_chordal_r1_v_to_splus")) for row in matched_rows
    )
    candidate_edge_mean = safe_mean(
        parse_optional_float(row.get("candidate_diag_edge_chordal_r1_v_to_splus")) for row in matched_rows
    )
    lines = [
        "# Boundary Dead-Zone Diagnostic",
        "",
        "## Coverage",
        "",
        f"- matched_token_steps: {len(matched_rows)}",
        f"- baseline_sample_count: {len(baseline_sample_summary)}",
        f"- candidate_sample_count: {len(candidate_sample_summary)}",
        "",
        "## Transport Flatness",
        "",
        f"- baseline_zero_iqr_sample_count: {baseline_zero_iqr}",
        f"- candidate_zero_iqr_sample_count: {candidate_zero_iqr}",
        f"- baseline_mean_loop_unique_value_count: {format_optional(baseline_unique_mean)}",
        f"- candidate_mean_loop_unique_value_count: {format_optional(candidate_unique_mean)}",
        f"- baseline_mean_loop_adjacent_tie_rate: {format_optional(baseline_tie_rate_mean)}",
        f"- candidate_mean_loop_adjacent_tie_rate: {format_optional(candidate_tie_rate_mean)}",
        "",
        "## Comparator / Transport Means",
        "",
        f"- baseline_mean_loop_residual: {format_optional(baseline_loop_mean)}",
        f"- candidate_mean_loop_residual: {format_optional(candidate_loop_mean)}",
        f"- baseline_mean_edge_r1_identity_gap: {format_optional(baseline_edge_mean)}",
        f"- candidate_mean_edge_r1_identity_gap: {format_optional(candidate_edge_mean)}",
        "",
        "## Promotion Checks",
        "",
        f"- zero_iqr_reduced: {str(candidate_zero_iqr < baseline_zero_iqr).lower()}",
        f"- unique_value_count_increased: {str((candidate_unique_mean or 0.0) > (baseline_unique_mean or 0.0)).lower()}",
        f"- tie_rate_reduced: {str((candidate_tie_rate_mean or 1.0) < (baseline_tie_rate_mean or 1.0)).lower()}",
        "",
        "## Note",
        "",
        "- This report only diagnoses dead-zone behavior under a fixed comparator.",
        "- Seam quietness and CFA localization remain separate promotion checks.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    cli_path = ensure_cli(args.cli_path)
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    ensure_dir(out_dir)

    baseline_diag_csv = out_dir / "baseline_gate5_diagnostic.csv"
    candidate_diag_csv = out_dir / "candidate_gate5_diagnostic.csv"
    run_gate5_diagnose(cli_path, (REPO_ROOT / args.baseline_input).resolve(), baseline_diag_csv)
    run_gate5_diagnose(cli_path, (REPO_ROOT / args.candidate_input).resolve(), candidate_diag_csv)

    baseline_boundary = build_boundary_rows(
        (REPO_ROOT / args.baseline_input).resolve(),
        load_boundary_steps((REPO_ROOT / args.baseline_boundary_steps).resolve())
        if args.baseline_boundary_steps
        else {},
    )
    candidate_boundary = build_boundary_rows(
        (REPO_ROOT / args.candidate_input).resolve(),
        load_boundary_steps((REPO_ROOT / args.candidate_boundary_steps).resolve())
        if args.candidate_boundary_steps
        else {},
    )
    baseline_diag = load_diag_rows(baseline_diag_csv)
    candidate_diag = load_diag_rows(candidate_diag_csv)

    matched_rows = join_rows(
        baseline_boundary=baseline_boundary,
        baseline_diag=baseline_diag,
        candidate_boundary=candidate_boundary,
        candidate_diag=candidate_diag,
    )
    matched_csv = out_dir / "matched_boundary_diagnostics.csv"
    if matched_rows:
        write_csv(matched_csv, matched_rows, fieldnames=list(matched_rows[0].keys()))
    else:
        write_csv(matched_csv, [], fieldnames=["sample_id", "step", "token_text", "label_token"])

    baseline_sample_summary = build_sample_transport_summary(baseline_diag)
    candidate_sample_summary = build_sample_transport_summary(candidate_diag)
    write_csv(
        out_dir / "baseline_transport_sample_summary.csv",
        baseline_sample_summary,
        fieldnames=[
            "sample_id",
            "n_loop_values",
            "loop_unique_value_count",
            "loop_zero_iqr",
            "loop_iqr",
            "loop_adjacent_tie_rate",
        ],
    )
    write_csv(
        out_dir / "candidate_transport_sample_summary.csv",
        candidate_sample_summary,
        fieldnames=[
            "sample_id",
            "n_loop_values",
            "loop_unique_value_count",
            "loop_zero_iqr",
            "loop_iqr",
            "loop_adjacent_tie_rate",
        ],
    )
    report = build_report(matched_rows, baseline_sample_summary, candidate_sample_summary)
    report_path = out_dir / "boundary_dead_zone_report.md"
    write_text(report_path, report)

    print(f"baseline_gate5_diagnostic_csv={baseline_diag_csv.as_posix()}")
    print(f"candidate_gate5_diagnostic_csv={candidate_diag_csv.as_posix()}")
    print(f"matched_boundary_diagnostics_csv={matched_csv.as_posix()}")
    print(f"report_md={report_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
