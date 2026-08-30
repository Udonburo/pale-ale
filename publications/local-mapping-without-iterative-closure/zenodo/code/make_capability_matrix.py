#!/usr/bin/env python3
"""Render the public capability-boundary matrix from its frozen JSON data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle


COLUMNS = [
    ("bit_echo", "Bit\necho"),
    ("one_edge", "One-edge\nrelation"),
    ("two_input_xor", "Two-input\nXOR mapping"),
    ("parity_8", "Length-8\nparity"),
    ("ordered_cycle", "Ordered\ncycle"),
    ("shuffled_cycle", "Shuffled\ncycle"),
    ("global_b", "Global\nobstruction"),
]

PALETTE = {
    "pass": "#2f855a",
    "formation": "#2f855a",
    "signal": "#d69e2e",
    "negative": "#c0564a",
    "unopened": "#d8dde5",
}


def state_key(value: str) -> str:
    if value.startswith("FORMATION PASS"):
        return "formation"
    if value == "PASS":
        return "pass"
    if value == "SCORE SIGNAL":
        return "signal"
    if value in {"NO SIGNAL", "NO FORMATION"}:
        return "negative"
    if value == "UNOPENED":
        return "unopened"
    raise ValueError(f"Unknown state: {value}")


def cell_label(value: str, row: dict[str, object], field: str) -> str:
    if value == "FORMATION PASS" and field == "two_input_xor":
        point = int(row["formation_point"])
        joint_passes = list(row["joint_formation_pass_shots"])
        if joint_passes == [point]:
            return f"FORMATION\nAT {point} SHOTS"
        return f"FORMATION\nFROM {point} SHOTS"
    return {
        "PASS": "PASS",
        "SCORE SIGNAL": "SIGNAL",
        "NO SIGNAL": "NO\nSIGNAL",
        "NO FORMATION": "NO\nFORMATION",
        "UNOPENED": "UNOPENED",
    }[value]


def text_color(key: str) -> str:
    return "#1f2937" if key == "unopened" else "#ffffff"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = payload["rows"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titleweight": "bold",
    })
    fig, ax = plt.subplots(figsize=(15.5, 8.8), dpi=180)
    fig.patch.set_facecolor("#f7f8fa")
    ax.set_facecolor("#f7f8fa")

    nrows = len(rows)
    ncols = len(COLUMNS)
    for row_index, row in enumerate(rows):
        y = nrows - 1 - row_index
        for col_index, (field, _) in enumerate(COLUMNS):
            value = row[field]
            key = state_key(value)
            ax.add_patch(Rectangle(
                (col_index, y), 1, 1,
                facecolor=PALETTE[key], edgecolor="#ffffff", linewidth=2.0,
            ))
            ax.text(
                col_index + 0.5, y + 0.5, cell_label(value, row, field),
                ha="center", va="center", color=text_color(key),
                fontsize=8.4, fontweight="bold", linespacing=1.05,
            )

    # Separate the zero-shot and ICL sections without creating another visual grammar.
    ax.plot([0, ncols], [2, 2], color="#111827", linewidth=3.0, clip_on=False)
    ax.text(-0.08, 5.92, "ZERO-SHOT", ha="right", va="top", fontsize=9.5,
            color="#4b5563", fontweight="bold")
    ax.text(-0.08, 1.92, "INPUT-OUTPUT-ONLY ICL", ha="right", va="top", fontsize=9.5,
            color="#4b5563", fontweight="bold")

    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    ax.set_xticks([index + 0.5 for index in range(ncols)])
    ax.set_xticklabels([label for _, label in COLUMNS], fontsize=10.2, fontweight="bold")
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=12)
    ax.set_yticks([nrows - 0.5 - index for index in range(nrows)])
    ax.set_yticklabels([row["condition"] for row in rows], fontsize=10.3)
    ax.tick_params(axis="y", length=0, pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.suptitle(
        "Context-dependent two-input mapping did not extend to iterative parity",
        x=0.16, y=0.975, ha="left", fontsize=18, fontweight="bold", color="#111827",
    )
    fig.text(
        0.16, 0.935,
        "Prospectively frozen capability states for Qwen3 0.6B-8B. Unopened cells are conditional stops, not negative measurements.",
        ha="left", va="top", fontsize=10.5, color="#4b5563",
    )

    legend = [
        Patch(facecolor=PALETTE["pass"], label="Behavior pass / surface formation"),
        Patch(facecolor=PALETTE["signal"], label="Score signal only"),
        Patch(facecolor=PALETTE["negative"], label="No detected signal / formation"),
        Patch(facecolor=PALETTE["unopened"], label="Unopened by rule"),
    ]
    fig.legend(
        handles=legend, loc="lower center", ncol=4, frameon=False,
        bbox_to_anchor=(0.58, 0.045), fontsize=9.5,
    )
    fig.text(
        0.16, 0.032,
        "Primary boundary: joint P2 formation began at 16 shots in 4B and occurred at 4 shots in 8B; no P3 cell met the predeclared score-signal criterion through 64 demonstrations.",
        ha="left", va="bottom", fontsize=9.2, color="#374151",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.34, right=0.985, top=0.82, bottom=0.17)
    fig.savefig(args.output, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
