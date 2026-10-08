"""Render README artwork and a chart from published Table 2; run no experiments.

Requires matplotlib. From the repository root:
    python docs/assets/readme/render_assets.py
Optional PNG previews go to --preview-dir, outside the published companion.
"""

from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path as DrawPath
from matplotlib.ticker import FixedLocator, FixedFormatter


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TABLE = ROOT / "papers/exact-state-ranked-array-rqmc/tables/primal_dual_times.md"
PALETTES = {
    "light": dict(bg="#f7f5ef", ink="#20363c", muted="#586b70", grid="#dddeda",
                  amber="#a56221", teal="#267b77", slate="#526ea0", faint="#e9e8e1"),
    "dark": dict(bg="#142126", ink="#edf0e9", muted="#bac8c7", grid="#35464a",
                 amber="#ecb476", teal="#77c7b6", slate="#9ab5e8", faint="#24363b"),
}


def save(fig, name, theme, preview_dir):
    svg_path = HERE / f"{name}-{theme}.svg"
    fig.savefig(svg_path, metadata={
        "Date": None,
        "Creator": "pale-ale README asset renderer",
        "Description": (
            "Illustrative repository artwork; no measured data."
            if name == "header" else
            "Published Table 2: median within-seed stream/direct timing ratios."
        ),
    })
    # Keep regenerated vector assets free of platform-specific line endings
    # and the SVG backend's trailing spaces.
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8", newline="\n",
    )
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_dir / f"{name}-{theme}.png", dpi=140)
    plt.close(fig)


def header(theme, preview_dir):
    p = PALETTES[theme]
    fig = plt.figure(figsize=(12.8, 3.25), facecolor=p["bg"])
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.plot((0.055, 0.105), (0.86, 0.86), color=p["amber"], lw=3)
    ax.text(0.12, 0.86, "INDEPENDENT RESEARCH  /  AOI KAWASAKI",
            fontsize=10, color=p["muted"], va="center", weight="medium")
    ax.text(0.05, 0.48, "pale-ale", fontsize=68, family="DejaVu Serif",
            color=p["ink"], weight="bold", va="center")
    ax.text(0.056, 0.24, "Structure. Computation. Evidence.", fontsize=20,
            color=p["ink"], va="center")
    ax.text(0.056, 0.12, "Exact simulation  /  Learned systems", fontsize=11.5,
            color=p["muted"], va="center")

    # Decorative aggregation motif. It depicts no empirical observations.
    colors = (p["amber"], p["teal"], p["slate"])
    for group, color in enumerate(colors):
        center = 0.72 - group * 0.22
        for row in range(4):
            y = center + (row - 1.5) * 0.034
            vertices = [(0.68, y), (0.735, y), (0.74, center), (0.79, center)]
            ax.add_patch(PathPatch(DrawPath(vertices, [1, 4, 4, 4]),
                                   fill=False, lw=1.0, edgecolor=color, alpha=0.65))
            # Markers keep a circular footprint on the wide canvas.
            ax.plot(0.675, y, marker="o", markersize=4.6, color=color, mec=p["bg"], mew=0.7)
        ax.add_patch(FancyBboxPatch((0.795, center - 0.055), 0.048, 0.11,
                                    boxstyle="round,pad=0.004,rounding_size=0.012",
                                    lw=1.3, edgecolor=color, facecolor=p["bg"]))
        for offset in (-0.02, 0, 0.02):
            ax.plot((0.806, 0.832), (center + offset, center + offset), color=color, lw=1.1)
        ax.plot((0.852, 0.9), (center, center), color=color, lw=1.2)
        ax.plot(0.907, center, marker="o", markersize=9, color=p["bg"], mec=color, mew=1.7)
    ax.plot((0.645, 0.935), (0.11, 0.11), color=p["grid"], lw=0.9)
    ax.text(0.645, 0.055, "REPRESENT  /  PRESERVE  /  REPLAY", fontsize=8.5,
            color=p["muted"], va="center")
    save(fig, "header", theme, preview_dir)


def observations():
    rows = []
    for line in TABLE.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[0] not in ("R63", "R255", "T"):
            continue
        w, m = map(int, cells[1].split(","))
        rows.append(dict(model=cells[0], w=w, m=m, ratio=float(cells[3])))
    if len(rows) != 17:
        raise ValueError("Expected the published 17-cell Table 2.")
    return rows


def crossover(theme, preview_dir, rows):
    p = PALETTES[theme]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8), facecolor=p["bg"])
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.32, top=0.77, wspace=0.16)
    fig.text(0.075, 0.93, "When counting beats rank enumeration", fontsize=20,
             color=p["ink"], weight="semibold")
    fig.text(0.075, 0.875, "Saved paired timing ratios  ·  8 seeds per condition  ·  1 host",
             fontsize=11, color=p["muted"])
    specs = [
        ("Machine repair", [("R63", 30, "Capacity 63 · 30-bit", p["amber"], "o", "-"),
                            ("R63", 52, "Capacity 63 · 52-bit", p["teal"], "s", "--"),
                            ("R255", 52, "Capacity 255 · 52-bit", p["slate"], "D", "-")]),
        ("Tandem queue", [("T", 30, "30-bit input", p["amber"], "o", "-"),
                           ("T", 52, "52-bit input", p["teal"], "s", "--")]),
    ]
    for idx, (ax, (title, series)) in enumerate(zip(axes, specs)):
        ax.set_facecolor(p["bg"])
        ax.set(yscale="log", ylim=(0.65, 85), xlim=(7.6, 20.4))
        ax.axhspan(0.65, 1, facecolor=p["faint"], zorder=0)
        ax.set_xticks([8, 12, 16, 20], ["256", "4,096", "65,536", "1,048,576"])
        ticks = [1, 2, 5, 10, 20, 50]
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
        ax.minorticks_off()
        ax.grid(axis="y", color=p["grid"], lw=0.75, zorder=0)
        ax.axhline(1, color=p["muted"], lw=1.2, zorder=1)
        ax.tick_params(colors=p["muted"], labelsize=10, length=0, pad=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("bottom", "left"):
            ax.spines[side].set_color(p["grid"])
        ax.set_title(title, loc="left", color=p["ink"], fontsize=14, pad=15, weight="semibold")
        ax.set_xlabel("Population size N (log scale)", fontsize=11, color=p["muted"], labelpad=12)
        if idx == 0:
            ax.set_ylabel("Stream time / direct-basis time", fontsize=11,
                          color=p["muted"], labelpad=10)
        for model, w, label, color, marker, linestyle in series:
            selected = sorted((r for r in rows if r["model"] == model and r["w"] == w),
                              key=lambda r: r["m"])
            ax.plot([r["m"] for r in selected], [r["ratio"] for r in selected],
                    color=color, marker=marker, linestyle=linestyle, label=label,
                    lw=2.0, markersize=6, markeredgecolor=p["bg"], markeredgewidth=0.8)
        ax.legend(loc="upper left", bbox_to_anchor=(-0.01, -0.24), frameon=False,
                  fontsize=9.5, labelcolor=p["ink"], ncol=1, handlelength=2.4)
    fig.text(0.075, 0.025,
             "Above 1: direct basis faster. Below 1: streaming faster. Lines join evaluated populations.",
             fontsize=10.5, color=p["muted"])
    save(fig, "crossover", theme, preview_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "pale-ale-readme-20261008",
                         "svg.fonttype": "path"})
    rows = observations()
    for theme in PALETTES:
        header(theme, args.preview_dir)
        crossover(theme, args.preview_dir, rows)
    print("Rendered light/dark headers and all 17 published Table 2 ratios; no timings collected.")


if __name__ == "__main__":
    main()
