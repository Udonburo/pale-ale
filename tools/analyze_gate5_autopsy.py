#!/usr/bin/env python3
"""Analyze Gate5 failure modes on existing token/sample telemetry artifacts."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Gate5 autopsy report from an existing gate5_out directory. "
            "Focuses on rotor-win case studies and signed timing / span-relative displacement."
        )
    )
    parser.add_argument("--gate5-out-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sample-ids", nargs="+", type=int)
    parser.add_argument("--top-positive-rotor-wins", type=int, default=5)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--percentile", type=float, default=0.90)
    parser.add_argument("--variant", default="frustrated")
    parser.add_argument("--world-type")
    parser.add_argument("--output-prefix", default="gate5_autopsy")
    return parser.parse_args()


def parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        out = float(raw)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def parse_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    return int(raw)


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


def percentile_nearest_rank(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    arr = sorted(values)
    rank = int(math.ceil(q * len(arr))) - 1
    rank = max(0, min(rank, len(arr) - 1))
    return arr[rank]


def mean(values: Iterable[float]) -> Optional[float]:
    arr = list(values)
    if not arr:
        return None
    return sum(arr) / float(len(arr))


def zone_for_step(step: Optional[int], defect_start: Optional[int], defect_end: Optional[int]) -> str:
    if step is None or defect_start is None or defect_end is None:
        return ""
    if step < defect_start:
        return "before"
    if step > defect_end:
        return "after"
    return "inside"


def first_hit_distance(rows: Sequence[Dict[str, str]], score_col: str, percentile: float) -> Dict[str, Any]:
    labels = [1 if row.get("label_token") == "1" else 0 for row in rows]
    scores = [parse_float(row.get(score_col)) for row in rows]
    defect_steps = [idx for idx, lab in enumerate(labels) if lab == 1]
    defect_start = defect_steps[0] if defect_steps else None
    defect_end = defect_steps[-1] if defect_steps else None
    valid_scores = [score for score in scores if score is not None]
    threshold = percentile_nearest_rank(valid_scores, percentile)
    first_hit = next(
        (idx for idx, score in enumerate(scores) if threshold is not None and score is not None and score >= threshold),
        None,
    )
    first_hit_after_defect = None
    if threshold is not None and defect_start is not None:
        for idx in range(defect_start, len(scores)):
            score = scores[idx]
            if score is not None and score >= threshold:
                first_hit_after_defect = idx
                break
    signed_distance = None
    if defect_start is not None and first_hit is not None:
        signed_distance = int(first_hit) - int(defect_start)
    after_defect_distance = None
    if defect_start is not None and first_hit_after_defect is not None:
        after_defect_distance = int(first_hit_after_defect) - int(defect_start)
    return {
        "threshold": threshold,
        "defect_start": defect_start,
        "defect_end": defect_end,
        "first_hit_step": first_hit,
        "first_hit_distance_signed": signed_distance,
        "first_hit_after_defect_step": first_hit_after_defect,
        "first_hit_after_defect_distance": after_defect_distance,
        "first_hit_zone": zone_for_step(first_hit, defect_start, defect_end),
    }


def top_ranked(rows: Sequence[Dict[str, str]], score_col: str, topk: int) -> List[Tuple[int, float]]:
    ranked: List[Tuple[int, float]] = []
    for idx, row in enumerate(rows):
        score = parse_float(row.get(score_col))
        if score is not None:
            ranked.append((idx, score))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked[:topk]


def build_case_row(
    sample_summary: Dict[str, str],
    token_rows: Sequence[Dict[str, str]],
    topk: int,
    percentile: float,
) -> Dict[str, Any]:
    labels = [1 if row.get("label_token") == "1" else 0 for row in token_rows]
    defect_steps = [idx for idx, label in enumerate(labels) if label == 1]
    defect_start = defect_steps[0] if defect_steps else None
    defect_end = defect_steps[-1] if defect_steps else None
    top_f = top_ranked(token_rows, "score_F_loop", topk=max(1, topk))
    top_rotor = top_ranked(token_rows, "rotor_loop_chordal_v1", topk=max(1, topk))
    top1_f_step = top_f[0][0] if top_f else None
    top1_rotor_step = top_rotor[0][0] if top_rotor else None
    first_hit_f = first_hit_distance(token_rows, "score_F_loop", percentile)
    first_hit_rotor = first_hit_distance(token_rows, "rotor_loop_chordal_v1", percentile)

    return {
        "sample_id": int(sample_summary["sample_id"]),
        "variant": sample_summary.get("variant", ""),
        "world_type": sample_summary.get("world_type", ""),
        "positive_token_count": parse_int(sample_summary.get("positive_token_count")),
        "delta_auprc_rotor_vs_F": parse_float(
            sample_summary.get("delta_auprc_rotor_loop_chordal_v1_vs_F")
        ),
        "auprc_F": parse_float(sample_summary.get("auprc_F")),
        "auprc_rotor": parse_float(sample_summary.get("auprc_rotor_loop_chordal_v1")),
        "hit_at_10_F": parse_int(sample_summary.get("hit_at_10_F")),
        "hit_at_10_rotor": parse_int(sample_summary.get("hit_at_10_rotor_loop_chordal_v1")),
        "defect_start_step": defect_start,
        "defect_end_step": defect_end,
        "top1_f_step": top1_f_step,
        "top1_rotor_step": top1_rotor_step,
        "top1_f_zone": zone_for_step(top1_f_step, defect_start, defect_end),
        "top1_rotor_zone": zone_for_step(top1_rotor_step, defect_start, defect_end),
        "top1_f_distance_signed": None if defect_start is None or top1_f_step is None else int(top1_f_step) - int(defect_start),
        "top1_rotor_distance_signed": None if defect_start is None or top1_rotor_step is None else int(top1_rotor_step) - int(defect_start),
        "first_hit_f_step": first_hit_f["first_hit_step"],
        "first_hit_rotor_step": first_hit_rotor["first_hit_step"],
        "first_hit_f_distance_signed": first_hit_f["first_hit_distance_signed"],
        "first_hit_rotor_distance_signed": first_hit_rotor["first_hit_distance_signed"],
        "first_hit_f_after_defect_distance": first_hit_f["first_hit_after_defect_distance"],
        "first_hit_rotor_after_defect_distance": first_hit_rotor["first_hit_after_defect_distance"],
        "first_hit_f_zone": first_hit_f["first_hit_zone"],
        "first_hit_rotor_zone": first_hit_rotor["first_hit_zone"],
        "rotor_topk_positive_hits": sum(labels[idx] for idx, _ in top_rotor),
        "f_topk_positive_hits": sum(labels[idx] for idx, _ in top_f),
    }


def build_top_token_rows(
    case_row: Dict[str, Any],
    token_rows: Sequence[Dict[str, str]],
    topk: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for metric_name, score_col in (
        ("F", "score_F_loop"),
        ("rotor", "rotor_loop_chordal_v1"),
    ):
        for rank, (step_idx, score) in enumerate(top_ranked(token_rows, score_col, topk=topk), start=1):
            row = token_rows[step_idx]
            out.append(
                {
                    "sample_id": case_row["sample_id"],
                    "variant": case_row["variant"],
                    "world_type": case_row["world_type"],
                    "metric": metric_name,
                    "rank": rank,
                    "step": step_idx,
                    "absolute_pos": parse_int(row.get("absolute_pos")),
                    "token_text": row.get("token_text", ""),
                    "label_token": parse_int(row.get("label_token")),
                    "score": score,
                }
            )
    return out


def summarize_population(
    case_rows: Sequence[Dict[str, Any]],
    key_name: str,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        grouped[str(row[key_name])].append(row)
    out: List[Dict[str, Any]] = []
    for key in sorted(grouped):
        rows = grouped[key]
        out.append(
            {
                key_name: key,
                "n_samples": len(rows),
                "mean_delta_auprc_rotor_vs_F": mean(
                    row["delta_auprc_rotor_vs_F"]
                    for row in rows
                    if row["delta_auprc_rotor_vs_F"] is not None
                ),
                "mean_top1_distance_F": mean(
                    row["top1_f_distance_signed"]
                    for row in rows
                    if row["top1_f_distance_signed"] is not None
                ),
                "mean_top1_distance_rotor": mean(
                    row["top1_rotor_distance_signed"]
                    for row in rows
                    if row["top1_rotor_distance_signed"] is not None
                ),
                "mean_first_hit_distance_F": mean(
                    row["first_hit_f_distance_signed"]
                    for row in rows
                    if row["first_hit_f_distance_signed"] is not None
                ),
                "mean_first_hit_distance_rotor": mean(
                    row["first_hit_rotor_distance_signed"]
                    for row in rows
                    if row["first_hit_rotor_distance_signed"] is not None
                ),
                "mean_first_hit_after_defect_distance_F": mean(
                    row["first_hit_f_after_defect_distance"]
                    for row in rows
                    if row["first_hit_f_after_defect_distance"] is not None
                ),
                "mean_first_hit_after_defect_distance_rotor": mean(
                    row["first_hit_rotor_after_defect_distance"]
                    for row in rows
                    if row["first_hit_rotor_after_defect_distance"] is not None
                ),
                "rotor_top1_before_rate": mean(
                    1.0 if row["top1_rotor_zone"] == "before" else 0.0
                    for row in rows
                    if row["top1_rotor_zone"]
                ),
                "f_top1_before_rate": mean(
                    1.0 if row["top1_f_zone"] == "before" else 0.0
                    for row in rows
                    if row["top1_f_zone"]
                ),
                "rotor_first_hit_before_rate": mean(
                    1.0 if row["first_hit_rotor_zone"] == "before" else 0.0
                    for row in rows
                    if row["first_hit_rotor_zone"]
                ),
                "f_first_hit_before_rate": mean(
                    1.0 if row["first_hit_f_zone"] == "before" else 0.0
                    for row in rows
                    if row["first_hit_f_zone"]
                ),
                "rotor_top1_after_rate": mean(
                    1.0 if row["top1_rotor_zone"] == "after" else 0.0
                    for row in rows
                    if row["top1_rotor_zone"]
                ),
                "f_top1_after_rate": mean(
                    1.0 if row["top1_f_zone"] == "after" else 0.0
                    for row in rows
                    if row["top1_f_zone"]
                ),
                "rotor_first_hit_after_rate": mean(
                    1.0 if row["first_hit_rotor_zone"] == "after" else 0.0
                    for row in rows
                    if row["first_hit_rotor_zone"]
                ),
                "f_first_hit_after_rate": mean(
                    1.0 if row["first_hit_f_zone"] == "after" else 0.0
                    for row in rows
                    if row["first_hit_f_zone"]
                ),
            }
        )
    return out


def render_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def write_report(
    out_path: Path,
    manifest: Dict[str, Any],
    selected_case_rows: Sequence[Dict[str, Any]],
    frustrated_summary: Sequence[Dict[str, Any]],
    variant: str,
    world_type: Optional[str],
) -> None:
    lines = [
        "# Gate5 Failure-Mode Autopsy",
        "",
        f"Run ID: {manifest.get('run_id', '')}",
        f"Method ID: {manifest.get('method_id', '')}",
        "",
        "## Focus",
        "",
        "- Current FWHT baseline only",
        f"- Variant filter: {variant}",
        f"- World-type filter: {world_type or 'all'}",
        "- Rotor-win case studies on the filtered population",
        "- Signed timing / span-relative displacement",
        "",
        "## Frustrated Population Summary",
        "",
        "| world_type | n | mean_delta_rotor_vs_F | mean_top1_F | mean_top1_rotor | mean_first_hit_F | mean_first_hit_rotor | mean_first_after_F | mean_first_after_rotor | rotor_first_hit_before | rotor_first_hit_after |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in frustrated_summary:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["world_type"]),
                    str(row["n_samples"]),
                    render_float(row["mean_delta_auprc_rotor_vs_F"]),
                    render_float(row["mean_top1_distance_F"]),
                    render_float(row["mean_top1_distance_rotor"]),
                    render_float(row["mean_first_hit_distance_F"]),
                    render_float(row["mean_first_hit_distance_rotor"]),
                    render_float(row["mean_first_hit_after_defect_distance_F"]),
                    render_float(row["mean_first_hit_after_defect_distance_rotor"]),
                    render_float(row["rotor_first_hit_before_rate"]),
                    render_float(row["rotor_first_hit_after_rate"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Selected Case Studies", ""])
    for row in selected_case_rows:
        lines.extend(
            [
                f"### sample_id={row['sample_id']} variant={row['variant']} world_type={row['world_type']}",
                "",
                f"- delta_auprc_rotor_vs_F: {render_float(row['delta_auprc_rotor_vs_F'])}",
                f"- auprc_F: {render_float(row['auprc_F'])}",
                f"- auprc_rotor: {render_float(row['auprc_rotor'])}",
                f"- defect_span: {row['defect_start_step']}..{row['defect_end_step']}",
                f"- top1_F: step={row['top1_f_step']} zone={row['top1_f_zone']} signed_distance={row['top1_f_distance_signed']}",
                f"- top1_rotor: step={row['top1_rotor_step']} zone={row['top1_rotor_zone']} signed_distance={row['top1_rotor_distance_signed']}",
                f"- first_hit_F: step={row['first_hit_f_step']} zone={row['first_hit_f_zone']} signed_distance={row['first_hit_f_distance_signed']}",
                f"- first_hit_rotor: step={row['first_hit_rotor_step']} zone={row['first_hit_rotor_zone']} signed_distance={row['first_hit_rotor_distance_signed']}",
                f"- hit_at_10_F: {row['hit_at_10_F']}",
                f"- hit_at_10_rotor: {row['hit_at_10_rotor']}",
                "",
            ]
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    gate5_out_dir = (REPO_ROOT / args.gate5_out_dir).resolve()
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    token_csv = gate5_out_dir / "gate5_token_telemetry.csv"
    sample_csv = gate5_out_dir / "gate5_sample_summary.csv"
    manifest_json = gate5_out_dir / "manifest.json"

    token_rows = read_csv(token_csv)
    sample_rows = read_csv(sample_csv)
    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))

    token_grouped: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    for row in token_rows:
        token_grouped[int(row["sample_id"])].append(row)
    for rows in token_grouped.values():
        rows.sort(key=lambda row: int(row["step"]))

    frustrated_rows = [
        row
        for row in sample_rows
        if row.get("variant") == args.variant
        and (args.world_type is None or row.get("world_type") == args.world_type)
        and parse_float(row.get("delta_auprc_rotor_loop_chordal_v1_vs_F")) is not None
    ]
    if not frustrated_rows:
        raise ValueError(
            "autopsy population is empty for the requested variant/world_type filter"
        )
    frustrated_rows.sort(
        key=lambda row: (-float(parse_float(row.get("delta_auprc_rotor_loop_chordal_v1_vs_F"))), int(row["sample_id"]))
    )
    frustrated_ids = {int(row["sample_id"]) for row in frustrated_rows}
    if args.sample_ids:
        selected_ids = [int(sample_id) for sample_id in args.sample_ids]
        missing_ids = [sample_id for sample_id in selected_ids if sample_id not in frustrated_ids]
        if missing_ids:
            raise ValueError(
                "requested sample_ids are not available in the frustrated autopsy population: "
                + ",".join(str(sample_id) for sample_id in missing_ids)
            )
    else:
        selected_ids = [int(row["sample_id"]) for row in frustrated_rows[: args.top_positive_rotor_wins]]

    selected_case_rows: List[Dict[str, Any]] = []
    top_token_rows: List[Dict[str, Any]] = []
    frustrated_case_rows: List[Dict[str, Any]] = []
    for row in frustrated_rows:
        sample_id = int(row["sample_id"])
        case_row = build_case_row(
            sample_summary=row,
            token_rows=token_grouped[sample_id],
            topk=args.topk,
            percentile=args.percentile,
        )
        frustrated_case_rows.append(case_row)
        if sample_id in selected_ids:
            selected_case_rows.append(case_row)
            top_token_rows.extend(
                build_top_token_rows(
                    case_row=case_row,
                    token_rows=token_grouped[sample_id],
                    topk=args.topk,
                )
            )

    if not selected_case_rows:
        raise ValueError(
            "autopsy selected zero cases; provide valid frustrated sample ids or omit --sample-ids"
        )

    selected_case_rows.sort(key=lambda row: selected_ids.index(int(row["sample_id"])))
    frustrated_summary = summarize_population(frustrated_case_rows, "world_type")

    out_dir.mkdir(parents=True, exist_ok=True)
    selected_cases_csv = out_dir / f"{args.output_prefix}_selected_cases.csv"
    selected_tokens_csv = out_dir / f"{args.output_prefix}_top_tokens.csv"
    frustrated_summary_csv = out_dir / f"{args.output_prefix}_world_summary.csv"
    report_md = out_dir / f"{args.output_prefix}_report.md"

    write_csv(
        selected_cases_csv,
        fieldnames=[
            "sample_id",
            "variant",
            "world_type",
            "positive_token_count",
            "delta_auprc_rotor_vs_F",
            "auprc_F",
            "auprc_rotor",
            "hit_at_10_F",
            "hit_at_10_rotor",
            "defect_start_step",
            "defect_end_step",
            "top1_f_step",
            "top1_rotor_step",
            "top1_f_zone",
            "top1_rotor_zone",
            "top1_f_distance_signed",
            "top1_rotor_distance_signed",
            "first_hit_f_step",
            "first_hit_rotor_step",
            "first_hit_f_distance_signed",
            "first_hit_rotor_distance_signed",
            "first_hit_f_after_defect_distance",
            "first_hit_rotor_after_defect_distance",
            "first_hit_f_zone",
            "first_hit_rotor_zone",
            "f_topk_positive_hits",
            "rotor_topk_positive_hits",
        ],
        rows=selected_case_rows,
    )
    write_csv(
        selected_tokens_csv,
        fieldnames=[
            "sample_id",
            "variant",
            "world_type",
            "metric",
            "rank",
            "step",
            "absolute_pos",
            "token_text",
            "label_token",
            "score",
        ],
        rows=top_token_rows,
    )
    write_csv(
        frustrated_summary_csv,
        fieldnames=[
            "world_type",
            "n_samples",
            "mean_delta_auprc_rotor_vs_F",
            "mean_top1_distance_F",
            "mean_top1_distance_rotor",
            "mean_first_hit_distance_F",
            "mean_first_hit_distance_rotor",
            "mean_first_hit_after_defect_distance_F",
            "mean_first_hit_after_defect_distance_rotor",
            "rotor_top1_before_rate",
            "f_top1_before_rate",
            "rotor_first_hit_before_rate",
            "f_first_hit_before_rate",
            "rotor_top1_after_rate",
            "f_top1_after_rate",
            "rotor_first_hit_after_rate",
            "f_first_hit_after_rate",
        ],
        rows=frustrated_summary,
    )
    write_report(
        report_md,
        manifest,
        selected_case_rows,
        frustrated_summary,
        variant=args.variant,
        world_type=args.world_type,
    )

    print(f"selected_cases_csv={selected_cases_csv.as_posix()}")
    print(f"selected_top_tokens_csv={selected_tokens_csv.as_posix()}")
    print(f"frustrated_world_summary_csv={frustrated_summary_csv.as_posix()}")
    print(f"report_md={report_md.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
