#!/usr/bin/env python3
"""Build a fixed genealogy supervision-policy summary from existing diagnostics."""

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a stable genealogy policy summary from an existing "
            "genealogy label-geometry diagnostic run."
        )
    )
    parser.add_argument("--label-geometry-out-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            encoded: Dict[str, Any] = {}
            for field in fieldnames:
                value = row.get(field)
                if isinstance(value, float):
                    encoded[field] = f"{value:.17e}"
                elif value is None:
                    encoded[field] = ""
                else:
                    encoded[field] = value
            writer.writerow(encoded)


def parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return float(raw)
    except Exception:
        return None


def render_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def main() -> None:
    args = parse_args()
    label_geometry_out_dir = (REPO_ROOT / Path(args.label_geometry_out_dir)).resolve()
    out_dir = (REPO_ROOT / Path(args.out_dir)).resolve()

    world_path = label_geometry_out_dir / "gate5_genealogy_label_geometry_world_summary.csv"
    decision_path = label_geometry_out_dir / "gate5_genealogy_label_geometry_decision.md"

    rows = read_csv(world_path)
    by_key = {(row["geometry_id"], row["world_type"]): row for row in rows}

    canonical = by_key[("inside_span", "genealogy")]
    candidate_ids = [
        "prefix_only_w3",
        "prefix_only_w1",
        "start_neighborhood_w1",
        "start_neighborhood_w3",
        "onset_only",
    ]

    summary_rows: List[Dict[str, Any]] = []
    canonical_delta = parse_float(canonical.get("mean_delta_rotor_vs_F"))
    for geometry_id in ["inside_span"] + candidate_ids:
        genealogy = by_key.get((geometry_id, "genealogy"))
        temporal = by_key.get((geometry_id, "temporal"))
        reachability = by_key.get((geometry_id, "reachability"))
        row = {
            "geometry_id": geometry_id,
            "is_canonical": 1 if geometry_id == "inside_span" else 0,
            "is_diagnostic_only": 0 if geometry_id == "inside_span" else 1,
            "genealogy_mean_delta_rotor_vs_F": parse_float(genealogy.get("mean_delta_rotor_vs_F")) if genealogy else None,
            "genealogy_rotor_before_rate": parse_float(genealogy.get("rotor_first_hit_before_rate")) if genealogy else None,
            "temporal_mean_delta_rotor_vs_F": parse_float(temporal.get("mean_delta_rotor_vs_F")) if temporal else None,
            "reachability_mean_delta_rotor_vs_F": parse_float(reachability.get("mean_delta_rotor_vs_F")) if reachability else None,
        }
        if row["genealogy_mean_delta_rotor_vs_F"] is not None and canonical_delta is not None:
            row["genealogy_gain_vs_inside_span"] = row["genealogy_mean_delta_rotor_vs_F"] - canonical_delta
        else:
            row["genealogy_gain_vs_inside_span"] = None
        summary_rows.append(row)

    fieldnames = [
        "geometry_id",
        "is_canonical",
        "is_diagnostic_only",
        "genealogy_mean_delta_rotor_vs_F",
        "genealogy_gain_vs_inside_span",
        "genealogy_rotor_before_rate",
        "temporal_mean_delta_rotor_vs_F",
        "reachability_mean_delta_rotor_vs_F",
    ]
    write_csv(out_dir / "genealogy_supervision_policy_summary.csv", fieldnames, summary_rows)

    decision_text = decision_path.read_text(encoding="utf-8")
    report_lines = [
        "# Genealogy Supervision Policy Summary",
        "",
        "## Policy Position",
        "",
        "- canonical genealogy evaluation remains `inside_span`",
        "- diagnostic geometries remain supplementary only",
        "- canonical and diagnostic views must not be merged",
        "",
        "## Fixed Summary Table",
        "",
        "| geometry | canonical | diagnostic_only | genealogy_delta | genealogy_gain_vs_inside_span | genealogy_before_rate | temporal_delta | reachability_delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        report_lines.append(
            f"| {row['geometry_id']} | {row['is_canonical']} | {row['is_diagnostic_only']} | "
            f"{render_float(row['genealogy_mean_delta_rotor_vs_F'])} | "
            f"{render_float(row['genealogy_gain_vs_inside_span'])} | "
            f"{render_float(row['genealogy_rotor_before_rate'])} | "
            f"{render_float(row['temporal_mean_delta_rotor_vs_F'])} | "
            f"{render_float(row['reachability_mean_delta_rotor_vs_F'])} |"
        )
    report_lines.extend(
        [
            "",
            "## Decision Snapshot",
            "",
            decision_text.strip(),
            "",
            "## Reading Rule",
            "",
            "- Use `inside_span` for canonical benchmark reporting.",
            "- Use `prefix_only_w3` and related geometries only as diagnostic interpretation layers.",
            "- Do not mix diagnostic geometry into canonical aggregate scores.",
        ]
    )
    (out_dir / "genealogy_supervision_policy_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    decision_lines = [
        "# Genealogy Supervision Policy Decision",
        "",
        "- canonical_genealogy_geometry: `inside_span`",
        "- diagnostic_geometry_view_enabled: `true`",
        "- preferred_diagnostic_geometry: `prefix_only_w3`",
        "- policy_status: `canonical-fixed-diagnostic-supplement`",
        "",
        "Canonical CFA labels remain unchanged. Diagnostic geometries exist to interpret genealogy-specific rotor behavior and must remain separate from canonical evaluation.",
    ]
    (out_dir / "genealogy_supervision_policy_decision.md").write_text(
        "\n".join(decision_lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
