#!/usr/bin/env python3
"""CFA pair case-study visualizer for Triality Primary=E."""

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import eval_triality_token as evaltok
import extract_triality_triplets as extractor
import labels_from_cfa_spans as cfa_labels


PRIMARY_SCORE = "E"
DEFAULT_MIN_COVERAGE = 0.30
MIN_TOKEN_MATCH = 0.98


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a CFA pair case-study visualization (sample vs contrast)."
    )
    parser.add_argument("--cfa-jsonl", default="data/cfa/cfa_v1.jsonl")
    parser.add_argument("--sample-id", type=int, default=127)
    parser.add_argument("--contrast-sample-id", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--perm-R",
        type=int,
        default=0,
        help="Recorded for provenance. Not used in this visualization workflow.",
    )
    parser.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE)
    parser.add_argument("--out-root", default="runs/cfa_case_study")
    parser.add_argument("--attest-root", default="attestations/triality/case_study")
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def fmt(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{float(value):.17e}"


def safe_token_text(token: str) -> str:
    return (
        str(token)
        .replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def load_cfa_rows(path: Path) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"non-object row at {path}:{line_no}")
            sid = int(row.get("sample_id", -1))
            if sid < 0:
                continue
            out[sid] = row
    return out


def zscores(values: Sequence[Optional[float]]) -> List[Optional[float]]:
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return [None] * len(values)
    mu = sum(valid) / float(len(valid))
    var = sum((x - mu) * (x - mu) for x in valid) / float(len(valid))
    sigma = math.sqrt(var)
    if sigma <= 1e-15:
        return [0.0 if v is not None else None for v in values]
    out: List[Optional[float]] = []
    for v in values:
        if v is None:
            out.append(None)
        else:
            out.append((float(v) - mu) / sigma)
    return out


def contiguous_segments(labels: Sequence[int]) -> List[Tuple[int, int]]:
    segments: List[Tuple[int, int]] = []
    in_seg = False
    start = 0
    for i, value in enumerate(labels):
        if value == 1 and not in_seg:
            in_seg = True
            start = i
        elif value != 1 and in_seg:
            segments.append((start, i))
            in_seg = False
    if in_seg:
        segments.append((start, len(labels)))
    return segments


def transition_labels(step_labels: Sequence[int]) -> List[int]:
    out: List[int] = []
    for t in range(max(0, len(step_labels) - 1)):
        y0 = int(step_labels[t])
        y1 = int(step_labels[t + 1])
        if evaltok.TRANSITION_LABEL_MODE == "next":
            out.append(y1)
        else:
            out.append(max(y0, y1))
    return out


def compute_scores(
    triplets: Sequence[Dict[str, Any]], step_labels: Sequence[int]
) -> Dict[str, Any]:
    n_steps = len(triplets)
    if n_steps != len(step_labels):
        raise ValueError("triplets/labels length mismatch")

    score_a = [-float(row["baseline_logprob"]) for row in triplets]
    score_b = [float(row["baseline_entropy"]) for row in triplets]

    score_e_trans: List[float] = []
    for t in range(max(0, n_steps - 1)):
        v_t = triplets[t]["V_8d"]
        sm_t = triplets[t]["Sminus_8d"]
        v_tp1 = triplets[t + 1]["V_8d"]
        score_e_trans.append(evaltok.d_proj(v_t, sm_t) + evaltok.d_proj(sm_t, v_tp1))

    labels_trans = transition_labels(step_labels)

    auprc_a = evaltok.average_precision(step_labels, score_a)
    auprc_b = evaltok.average_precision(step_labels, score_b)
    auprc_e = evaltok.average_precision(labels_trans, score_e_trans)

    best_name = None
    best_auprc = None
    for name, value in [("A", auprc_a), ("B", auprc_b)]:
        if value is None:
            continue
        if best_auprc is None or value > best_auprc:
            best_auprc = value
            best_name = name

    delta = None
    if auprc_e is not None and best_auprc is not None:
        delta = auprc_e - best_auprc

    score_e_token: List[Optional[float]] = [None] * n_steps
    for i, val in enumerate(score_e_trans):
        score_e_token[i] = val

    z_a = zscores([float(x) for x in score_a])
    z_b = zscores([float(x) for x in score_b])
    z_e = zscores(score_e_token)

    rank_e: List[Optional[int]] = [None] * n_steps
    indexed = [(i, v) for i, v in enumerate(score_e_token) if v is not None]
    indexed.sort(key=lambda t: (-float(t[1]), int(t[0])))
    for rnk, (idx, _) in enumerate(indexed, start=1):
        rank_e[idx] = rnk

    return {
        "score_a": score_a,
        "score_b": score_b,
        "score_e_token": score_e_token,
        "score_e_trans": score_e_trans,
        "labels_trans": labels_trans,
        "z_a": z_a,
        "z_b": z_b,
        "z_e": z_e,
        "rank_e": rank_e,
        "auprc_a": auprc_a,
        "auprc_b": auprc_b,
        "auprc_e": auprc_e,
        "best_baseline_name": best_name,
        "best_baseline_auprc": best_auprc,
        "delta_e_vs_best": delta,
    }


def write_token_table_csv(
    out_path: Path,
    triplets: Sequence[Dict[str, Any]],
    step_labels: Sequence[int],
    scores: Dict[str, Any],
) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "absolute_pos",
                "token_text",
                "label",
                "is_defect_token",
                "score_A_logprob",
                "score_B_entropy",
                "score_E",
                "z_A",
                "z_B",
                "z_E",
                "rank_E_desc",
                "in_defect_span",
            ]
        )
        for step, row in enumerate(triplets):
            label = int(step_labels[step])
            writer.writerow(
                [
                    int(step),
                    int(row.get("absolute_pos", step)),
                    safe_token_text(str(row.get("token_str", ""))),
                    label,
                    label,
                    fmt(scores["score_a"][step]),
                    fmt(scores["score_b"][step]),
                    fmt(scores["score_e_token"][step]),
                    fmt(scores["z_a"][step]),
                    fmt(scores["z_b"][step]),
                    fmt(scores["z_e"][step]),
                    "" if scores["rank_e"][step] is None else int(scores["rank_e"][step]),
                    label,
                ]
            )
    return sha256_file(out_path)


def write_pair_overlay_csv(
    out_path: Path,
    fr_triplets: Sequence[Dict[str, Any]],
    co_triplets: Sequence[Dict[str, Any]],
    fr_labels: Sequence[int],
    co_labels: Sequence[int],
    fr_scores: Dict[str, Any],
    co_scores: Dict[str, Any],
) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = min(len(fr_triplets), len(co_triplets))
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "token_frustrated",
                "token_consistent",
                "label_frustrated",
                "label_consistent",
                "E_frustrated",
                "E_consistent",
                "A_frustrated",
                "A_consistent",
                "delta_E",
                "delta_A",
            ]
        )
        for step in range(n):
            e_fr = fr_scores["score_e_token"][step] if step < len(fr_scores["score_e_token"]) else None
            e_co = co_scores["score_e_token"][step] if step < len(co_scores["score_e_token"]) else None
            d_e = None
            if e_fr is not None and e_co is not None:
                d_e = float(e_fr) - float(e_co)

            a_fr = fr_scores["score_a"][step]
            a_co = co_scores["score_a"][step]
            d_a = float(a_fr) - float(a_co)

            writer.writerow(
                [
                    int(step),
                    safe_token_text(str(fr_triplets[step].get("token_str", ""))),
                    safe_token_text(str(co_triplets[step].get("token_str", ""))),
                    int(fr_labels[step]),
                    int(co_labels[step]),
                    fmt(e_fr),
                    fmt(e_co),
                    fmt(a_fr),
                    fmt(a_co),
                    fmt(d_e),
                    fmt(d_a),
                ]
            )
    return sha256_file(out_path)


def sparse_xticks(ax: Any, tokens: Sequence[str], max_ticks: int = 24) -> None:
    n = len(tokens)
    if n <= 0:
        return
    step = max(1, n // max_ticks)
    ticks = list(range(0, n, step))
    labels = []
    for i in ticks:
        raw = safe_token_text(tokens[i])
        labels.append(raw if len(raw) <= 14 else raw[:11] + "...")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)


def shade_segments(ax: Any, segments: Sequence[Tuple[int, int]], color: str = "#f4c2c2") -> None:
    for start, end in segments:
        ax.axvspan(start - 0.5, end - 0.5, color=color, alpha=0.35, linewidth=0)


def plot_single_case(
    out_path: Path,
    sample_id: int,
    sample_row: Dict[str, Any],
    triplets: Sequence[Dict[str, Any]],
    labels: Sequence[int],
    scores: Dict[str, Any],
    exact_match_ratio: float,
    coverage: float,
) -> str:
    tokens = [str(r.get("token_str", "")) for r in triplets]
    x_token = list(range(len(triplets)))
    x_e = list(range(len(scores["score_e_trans"])))
    segs = contiguous_segments(labels)

    best = scores["best_baseline_name"] or "A"
    best_trace = scores["score_a"] if best == "A" else scores["score_b"]
    best_label = "A(-logprob)" if best == "A" else "B(entropy)"

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(x_e, scores["score_e_trans"], label="E (V+Sminus)", color="#d62728", linewidth=1.8)
    ax.plot(x_token, best_trace, label=f"Best baseline {best_label}", color="#1f77b4", linewidth=1.3)
    shade_segments(ax, segs)
    sparse_xticks(ax, tokens)
    ax.set_xlabel("token step")
    ax.set_ylabel("score")
    ax.grid(alpha=0.25, linestyle="--")
    title = (
        f"Case sample={sample_id} world={sample_row.get('world_type')} variant={sample_row.get('variant')} "
        f"exact_match={exact_match_ratio:.3f} coverage={coverage:.3f}"
    )
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return sha256_file(out_path)


def plot_pair_compare(
    out_path: Path,
    sample_id: int,
    contrast_id: int,
    fr_triplets: Sequence[Dict[str, Any]],
    co_triplets: Sequence[Dict[str, Any]],
    fr_labels: Sequence[int],
    fr_scores: Dict[str, Any],
    co_scores: Dict[str, Any],
) -> str:
    tokens_fr = [str(r.get("token_str", "")) for r in fr_triplets]
    segs_fr = contiguous_segments(fr_labels)

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=False)

    # Top: E traces
    x_ef = list(range(len(fr_scores["score_e_trans"])))
    x_ec = list(range(len(co_scores["score_e_trans"])))
    axes[0].plot(x_ef, fr_scores["score_e_trans"], label=f"E frustrated {sample_id}", color="#d62728")
    axes[0].plot(x_ec, co_scores["score_e_trans"], label=f"E consistent {contrast_id}", color="#2ca02c")
    shade_segments(axes[0], segs_fr)
    axes[0].set_title("E score traces")
    axes[0].grid(alpha=0.25, linestyle="--")
    axes[0].legend(loc="upper right")

    # Middle: baseline traces
    fr_best = fr_scores["best_baseline_name"] or "A"
    co_best = co_scores["best_baseline_name"] or "A"
    fr_trace = fr_scores["score_a"] if fr_best == "A" else fr_scores["score_b"]
    co_trace = co_scores["score_a"] if co_best == "A" else co_scores["score_b"]
    axes[1].plot(list(range(len(fr_trace))), fr_trace, label=f"baseline(frustrated,{fr_best})", color="#1f77b4")
    axes[1].plot(list(range(len(co_trace))), co_trace, label=f"baseline(consistent,{co_best})", color="#9467bd")
    shade_segments(axes[1], segs_fr)
    axes[1].set_title("Baseline traces")
    axes[1].grid(alpha=0.25, linestyle="--")
    axes[1].legend(loc="upper right")

    # Bottom: delta E
    n = min(len(fr_scores["score_e_trans"]), len(co_scores["score_e_trans"]))
    delta_e = [fr_scores["score_e_trans"][i] - co_scores["score_e_trans"][i] for i in range(n)]
    axes[2].plot(list(range(n)), delta_e, label="delta_E (frustrated-consistent)", color="#ff7f0e")
    shade_segments(axes[2], segs_fr)
    sparse_xticks(axes[2], tokens_fr)
    axes[2].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axes[2].set_title("Delta E")
    axes[2].set_xlabel("token step")
    axes[2].grid(alpha=0.25, linestyle="--")
    axes[2].legend(loc="upper right")

    fig.suptitle(f"CFA pair compare: frustrated={sample_id} vs consistent={contrast_id}")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return sha256_file(out_path)


def top_k_e_tokens(
    triplets: Sequence[Dict[str, Any]],
    labels_trans: Sequence[int],
    score_e_trans: Sequence[float],
    k: int = 10,
) -> List[Dict[str, Any]]:
    rows = []
    for step, score in enumerate(score_e_trans):
        tok = safe_token_text(str(triplets[step].get("token_str", "")))
        rows.append(
            {
                "step": step,
                "token_text": tok,
                "score_e": float(score),
                "transition_label": int(labels_trans[step]) if step < len(labels_trans) else 0,
            }
        )
    rows.sort(key=lambda r: (-r["score_e"], r["step"]))
    return rows[:k]


def write_case_summary_md(
    out_path: Path,
    sample_id: int,
    contrast_id: int,
    fr_row: Dict[str, Any],
    co_row: Dict[str, Any],
    fr_scores: Dict[str, Any],
    co_scores: Dict[str, Any],
    top10: Sequence[Dict[str, Any]],
    overlap_count: int,
    overlap_k: int,
    token_table_fr: Path,
    token_table_co: Path,
    overlay_csv: Path,
    plot_fr: Path,
    plot_pair: Path,
) -> str:
    lines: List[str] = []
    lines.append(f"# CFA Case Study: sample {sample_id} vs {contrast_id}")
    lines.append("")
    lines.append("## Pair")
    lines.append(f"- Frustrated sample_id: `{sample_id}`")
    lines.append(f"- Consistent contrast_sample_id: `{contrast_id}`")
    lines.append(f"- World type: `{fr_row.get('world_type')}`")
    lines.append("")
    lines.append("## CFA Snippets")
    lines.append("### Frustrated Prompt")
    lines.append("```text")
    lines.append(str(fr_row.get("prompt", "")))
    lines.append("```")
    lines.append("### Frustrated Answer")
    lines.append("```text")
    lines.append(str(fr_row.get("answer", "")))
    lines.append("```")
    lines.append(f"- Defect spans: `{json.dumps(fr_row.get('defect_spans', []), ensure_ascii=False)}`")
    lines.append("")
    lines.append("### Consistent Answer")
    lines.append("```text")
    lines.append(str(co_row.get("answer", "")))
    lines.append("```")
    lines.append("")
    lines.append("## AUPRC Summary")
    lines.append("| Sample | AUPRC_A | AUPRC_B | AUPRC_E | Best Baseline | Delta_AUPRC(E-best) |")
    lines.append("|---|---:|---:|---:|---|---:|")
    lines.append(
        f"| {sample_id} (frustrated) | {fmt(fr_scores['auprc_a'])} | {fmt(fr_scores['auprc_b'])} | "
        f"{fmt(fr_scores['auprc_e'])} | {fr_scores['best_baseline_name']} | {fmt(fr_scores['delta_e_vs_best'])} |"
    )
    lines.append(
        f"| {contrast_id} (consistent) | {fmt(co_scores['auprc_a'])} | {fmt(co_scores['auprc_b'])} | "
        f"{fmt(co_scores['auprc_e'])} | {co_scores['best_baseline_name']} | {fmt(co_scores['delta_e_vs_best'])} |"
    )
    lines.append("")
    lines.append("## Top 10 Tokens by E (Frustrated)")
    lines.append("| Rank | Step | Token | E score | Transition label |")
    lines.append("|---:|---:|---|---:|---:|")
    for rank, row in enumerate(top10, start=1):
        lines.append(
            f"| {rank} | {row['step']} | `{row['token_text']}` | {fmt(row['score_e'])} | {row['transition_label']} |"
        )
    lines.append("")
    lines.append(
        f"- Overlap top-{overlap_k} with defect-transition labels: `{overlap_count}/{overlap_k}`"
    )
    lines.append("")
    lines.append("## Output Files")
    lines.append(f"- token_table frustrated: `{token_table_fr.as_posix()}`")
    lines.append(f"- token_table consistent: `{token_table_co.as_posix()}`")
    lines.append(f"- pair overlay CSV: `{overlay_csv.as_posix()}`")
    lines.append(f"- plot frustrated: `{plot_fr.as_posix()}`")
    lines.append(f"- plot pair: `{plot_pair.as_posix()}`")
    lines.append("")
    lines.append("## Interpretation Notes (fill in)")
    lines.append("- E spikes coincide with defect span transitions:")
    lines.append("- Baseline behavior near defect span:")
    lines.append("- Consistent vs frustrated divergence pattern:")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return sha256_file(out_path)


def extract_and_label_sample(
    sample_row: Dict[str, Any],
    sample_id: int,
    model: Any,
    tokenizer: Any,
    device: Any,
    topk: int,
    min_coverage: float,
    out_dir: Path,
) -> Dict[str, Any]:
    prompt = str(sample_row.get("prompt", ""))
    answer = str(sample_row.get("answer", ""))
    if not prompt or not answer:
        raise ValueError(f"sample_id={sample_id} has empty prompt or answer")

    triplets, ext_meta = extractor.run_teacher_forcing_extraction(
        prompt=prompt,
        target_answer=answer,
        model=model,
        tokenizer=tokenizer,
        device=device,
        topk=topk,
    )
    mode_details = ext_meta["mode_details"]
    ratio = float(mode_details.get("exact_token_match_ratio") or 0.0)
    if ratio < MIN_TOKEN_MATCH:
        raise RuntimeError(
            f"sample_id={sample_id} exact_token_match_ratio={ratio:.6f} < {MIN_TOKEN_MATCH:.2f}"
        )

    triplets_path = out_dir / f"triplets_{sample_id}.ndjson"
    triplets_sha = extractor.write_ndjson(triplets_path, triplets)

    defect_spans = cfa_labels.normalize_spans(sample_row.get("defect_spans", []), len(answer))
    mapped = cfa_labels.map_using_triplet_char_offsets(triplets, defect_spans)
    coverage = float(mapped["coverage"])
    if coverage < min_coverage:
        raise RuntimeError(
            f"sample_id={sample_id} coverage={coverage:.6f} < min_coverage={min_coverage:.6f}"
        )
    labels = [int(x) for x in mapped["labels"]]
    token_ids = [int(r["token_id"]) for r in triplets]

    labels_path = out_dir / f"labels_{sample_id}.jsonl"
    cfa_labels.write_labels_jsonl(labels_path, labels=labels, token_ids=token_ids)
    labels_sha = sha256_file(labels_path)

    labels_meta_path = out_dir / f"labels_{sample_id}_meta.json"
    labels_meta = {
        "sample_id": sample_id,
        "variant": sample_row.get("variant"),
        "world_type": sample_row.get("world_type"),
        "triplets_path": triplets_path.as_posix(),
        "label_mapping_mode": mapped.get("mode"),
        "final_alignment_coverage_ratio": coverage,
        "mapped_positive_tokens": int(mapped["mapped_positive_tokens"]),
        "total_positive_tokens": int(mapped["total_positive_tokens"]),
        "final_positive_steps": int(sum(1 for x in labels if x == 1)),
        "final_negative_steps": int(sum(1 for x in labels if x == 0)),
        "labels_out": labels_path.as_posix(),
    }
    cfa_labels.write_meta_json(labels_meta_path, labels_meta)

    score_pack = compute_scores(triplets=triplets, step_labels=labels)

    return {
        "sample_id": sample_id,
        "prompt": prompt,
        "answer": answer,
        "triplets": triplets,
        "labels": labels,
        "scores": score_pack,
        "exact_token_match_ratio": ratio,
        "coverage": coverage,
        "triplets_path": triplets_path,
        "labels_path": labels_path,
        "labels_meta_path": labels_meta_path,
        "triplets_sha256": triplets_sha,
        "labels_sha256": labels_sha,
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "answer_sha256": sha256_bytes(answer.encode("utf-8")),
        "labels_meta": labels_meta,
        "extract_mode_details": mode_details,
    }


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(str(repo_root))

    cfa_path = Path(args.cfa_jsonl)
    rows = load_cfa_rows(cfa_path)
    if args.sample_id not in rows:
        raise ValueError(f"sample_id={args.sample_id} not found in {cfa_path}")

    sample_row = rows[args.sample_id]
    contrast_id = args.contrast_sample_id
    if contrast_id is None:
        if "contrast_sample_id" not in sample_row:
            raise ValueError("contrast_sample_id not provided and not found in sample row")
        contrast_id = int(sample_row["contrast_sample_id"])
    if contrast_id not in rows:
        raise ValueError(f"contrast_sample_id={contrast_id} not found in {cfa_path}")
    contrast_row = rows[contrast_id]

    pair_id = f"sample{args.sample_id}_vs_{contrast_id}"
    run_dir = Path(args.out_root) / pair_id
    att_dir = Path(args.attest_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    att_dir.mkdir(parents=True, exist_ok=True)

    extractor.configure_reproducibility(args.seed, deterministic=True)
    device = extractor.resolve_device(args.device)
    candidates = extractor.build_model_candidates(args.model_id)
    model_id, tokenizer, model, model_revision = extractor.load_first_available_model(
        model_candidates=candidates,
        device=device,
    )

    sample_pack = extract_and_label_sample(
        sample_row=sample_row,
        sample_id=args.sample_id,
        model=model,
        tokenizer=tokenizer,
        device=device,
        topk=args.topk,
        min_coverage=args.min_coverage,
        out_dir=run_dir,
    )
    contrast_pack = extract_and_label_sample(
        sample_row=contrast_row,
        sample_id=contrast_id,
        model=model,
        tokenizer=tokenizer,
        device=device,
        topk=args.topk,
        min_coverage=args.min_coverage,
        out_dir=run_dir,
    )

    token_table_sample = run_dir / f"token_table_{args.sample_id}.csv"
    token_table_contrast = run_dir / f"token_table_{contrast_id}.csv"
    overlay_csv = run_dir / f"pair_overlay_{args.sample_id}_vs_{contrast_id}.csv"
    plot_case = run_dir / f"plot_case_{args.sample_id}.png"
    plot_pair = run_dir / f"plot_pair_compare_{args.sample_id}_{contrast_id}.png"
    summary_md = att_dir / f"case_summary_{args.sample_id}_{contrast_id}.md"

    sha_token_sample = write_token_table_csv(
        token_table_sample,
        sample_pack["triplets"],
        sample_pack["labels"],
        sample_pack["scores"],
    )
    sha_token_contrast = write_token_table_csv(
        token_table_contrast,
        contrast_pack["triplets"],
        contrast_pack["labels"],
        contrast_pack["scores"],
    )
    sha_overlay = write_pair_overlay_csv(
        overlay_csv,
        sample_pack["triplets"],
        contrast_pack["triplets"],
        sample_pack["labels"],
        contrast_pack["labels"],
        sample_pack["scores"],
        contrast_pack["scores"],
    )
    sha_plot_case = plot_single_case(
        plot_case,
        args.sample_id,
        sample_row,
        sample_pack["triplets"],
        sample_pack["labels"],
        sample_pack["scores"],
        sample_pack["exact_token_match_ratio"],
        sample_pack["coverage"],
    )
    sha_plot_pair = plot_pair_compare(
        plot_pair,
        args.sample_id,
        contrast_id,
        sample_pack["triplets"],
        contrast_pack["triplets"],
        sample_pack["labels"],
        sample_pack["scores"],
        contrast_pack["scores"],
    )

    top10 = top_k_e_tokens(
        sample_pack["triplets"],
        sample_pack["scores"]["labels_trans"],
        sample_pack["scores"]["score_e_trans"],
        k=10,
    )
    overlap = sum(1 for row in top10 if int(row["transition_label"]) == 1)
    sha_summary = write_case_summary_md(
        summary_md,
        args.sample_id,
        contrast_id,
        sample_row,
        contrast_row,
        sample_pack["scores"],
        contrast_pack["scores"],
        top10,
        overlap_count=overlap,
        overlap_k=10,
        token_table_fr=token_table_sample,
        token_table_co=token_table_contrast,
        overlay_csv=overlay_csv,
        plot_fr=plot_case,
        plot_pair=plot_pair,
    )

    meta_path = run_dir / f"case_meta_{args.sample_id}_{contrast_id}.json"
    meta = {
        "date": dt.date.today().isoformat(),
        "primary_score": PRIMARY_SCORE,
        "seed": args.seed,
        "perm_R": args.perm_R,
        "model_id": model_id,
        "model_revision": model_revision,
        "device": str(device),
        "pair_id": pair_id,
        "sample_id": args.sample_id,
        "contrast_sample_id": contrast_id,
        "cfa_jsonl": cfa_path.as_posix(),
        "outputs": {
            "token_table_sample": token_table_sample.as_posix(),
            "token_table_contrast": token_table_contrast.as_posix(),
            "pair_overlay_csv": overlay_csv.as_posix(),
            "plot_case_png": plot_case.as_posix(),
            "plot_pair_png": plot_pair.as_posix(),
            "summary_md": summary_md.as_posix(),
        },
        "sha256": {
            "triplets_sample": sample_pack["triplets_sha256"],
            "triplets_contrast": contrast_pack["triplets_sha256"],
            "labels_sample": sample_pack["labels_sha256"],
            "labels_contrast": contrast_pack["labels_sha256"],
            "token_table_sample": sha_token_sample,
            "token_table_contrast": sha_token_contrast,
            "pair_overlay_csv": sha_overlay,
            "plot_case_png": sha_plot_case,
            "plot_pair_png": sha_plot_pair,
            "summary_md": sha_summary,
        },
        "sample_meta": {
            str(args.sample_id): {
                "variant": sample_row.get("variant"),
                "world_type": sample_row.get("world_type"),
                "exact_token_match_ratio": sample_pack["exact_token_match_ratio"],
                "label_coverage": sample_pack["coverage"],
                "prompt_sha256": sample_pack["prompt_sha256"],
                "answer_sha256": sample_pack["answer_sha256"],
                "auprc_A": sample_pack["scores"]["auprc_a"],
                "auprc_B": sample_pack["scores"]["auprc_b"],
                "auprc_E": sample_pack["scores"]["auprc_e"],
                "best_baseline": sample_pack["scores"]["best_baseline_name"],
                "delta_E_minus_best": sample_pack["scores"]["delta_e_vs_best"],
                "triplets_path": sample_pack["triplets_path"].as_posix(),
                "labels_path": sample_pack["labels_path"].as_posix(),
                "labels_meta_path": sample_pack["labels_meta_path"].as_posix(),
            },
            str(contrast_id): {
                "variant": contrast_row.get("variant"),
                "world_type": contrast_row.get("world_type"),
                "exact_token_match_ratio": contrast_pack["exact_token_match_ratio"],
                "label_coverage": contrast_pack["coverage"],
                "prompt_sha256": contrast_pack["prompt_sha256"],
                "answer_sha256": contrast_pack["answer_sha256"],
                "auprc_A": contrast_pack["scores"]["auprc_a"],
                "auprc_B": contrast_pack["scores"]["auprc_b"],
                "auprc_E": contrast_pack["scores"]["auprc_e"],
                "best_baseline": contrast_pack["scores"]["best_baseline_name"],
                "delta_E_minus_best": contrast_pack["scores"]["delta_e_vs_best"],
                "triplets_path": contrast_pack["triplets_path"].as_posix(),
                "labels_path": contrast_pack["labels_path"].as_posix(),
                "labels_meta_path": contrast_pack["labels_meta_path"].as_posix(),
            },
        },
        "top10_E_overlap_in_defect_transitions": {
            "k": 10,
            "overlap_count": overlap,
        },
    }
    with open(meta_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    print(f"pair_id={pair_id}")
    print(f"sample_id={args.sample_id} contrast_sample_id={contrast_id}")
    print(f"run_dir={run_dir.as_posix()}")
    print(f"summary_md={summary_md.as_posix()}")
    print(f"meta_json={meta_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
