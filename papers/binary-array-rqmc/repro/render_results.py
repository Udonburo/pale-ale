"""Regenerate two manuscript figures and Tables 1-4 from saved inputs only."""
from pathlib import Path
from collections import Counter
import json
import math
import statistics

ROOT = Path(__file__).resolve().parent
LABELS = {"LIBRARY": "Library array", "DIRECT": "Direct binary array",
          "MAINTAINED": "Order-maintained array", "CRN": "CRN", "ANTITHETIC": "Antithetic"}
METHODS = ("DIRECT", "MAINTAINED", "CRN", "ANTITHETIC")
COLORS = {"DIRECT": "#386CB0", "MAINTAINED": "#087F5B", "CRN": "#D07417", "ANTITHETIC": "#9461A2"}


def load_data():
    return json.loads((ROOT / "data/saved_results.json").read_text(encoding="utf-8"))


def geometric(values):
    return math.exp(statistics.mean(map(math.log, values)))


def target_rows(data, condition):
    return {r["method"]: r for r in data["targets"] if r["condition"] == condition}


def crossover(candidate, reference, setup_key="all_acquired_setup"):
    difference = reference["wall"] - candidate["wall"]
    if difference <= 0:
        return None
    return max(1, math.floor((candidate[setup_key] - reference[setup_key]) / difference) + 1)


def validate_data(data):
    # Validate the exported grain and accounting, not the unseen raw experiments.
    counts = Counter((r["batch"], r["method"], r["condition"], r["n"]) for r in data["profiles"])
    expected = {"projection": {"LIBRARY", "DIRECT"}, "ordering": {"DIRECT", "MAINTAINED"}}
    assert len(counts) == 64
    for (batch, method, condition, n), count in counts.items():
        assert method in expected[batch] and n == 4096
        assert count == (8 if batch == "projection" else 3)
    for row in data["profiles"]:
        assert all(math.isfinite(v) and v >= 0 for v in row["times"].values())
    for prefix, methods in (("projection", ("LIBRARY", "DIRECT")), ("ordering", ("DIRECT", "MAINTAINED"))):
        rows = data[prefix + "_timing"]
        keys = [(r["condition"], r["n"], r["method"]) for r in rows]
        assert len(rows) == len(set(keys)) == 64
        for n in (512, 4096):
            means = {m: geometric([r["wall"] for r in rows if r["n"] == n and r["method"] == m]) for m in methods}
            saved = next(r for r in data[prefix + "_summary"] if r["n"] == n)
            ratio = means[methods[1]] / means[methods[0]]
            wanted = saved["DIRECT"]["wall_over_old_array"] if prefix == "projection" else saved["wall_ratio"]
            assert math.isclose(ratio, wanted, rel_tol=1e-12)
    assert len(data["targets"]) == 8
    for condition in ("case_07_distant", "case_08_distant"):
        rows = target_rows(data, condition)
        assert set(rows) == set(METHODS)
        for r in rows.values():
            incremental = r["costs"]["new_variance_wall"] + r["costs"]["new_timing_wall"] + r["cold_overhead"]
            setup = incremental + r["costs"]["historical_variance_wall"]
            assert math.isclose(incremental, r["incremental_setup"], abs_tol=1e-12)
            assert math.isclose(setup, r["all_acquired_setup"], abs_tol=1e-12)
            assert math.isclose(setup + r["wall"], r["historical_inclusive_first"], abs_tol=1e-12)
            assert math.isclose(setup/1000 + r["wall"], r["historical_inclusive_amortized_1000"], abs_tol=1e-12)
            assert 0.6/(r["n"]*r["k"]) <= r["risk"] <= r["risk_hi"] < r["target"]
            assert r["timing_requests"] == 24 and r["validation_requests"] == 128
    quadratic = target_rows(data, "case_08_distant")
    assert crossover(quadratic["MAINTAINED"], quadratic["CRN"]) == 2799
    assert crossover(quadratic["MAINTAINED"], quadratic["CRN"], "incremental_setup") == 476


def write_tables(data):
    lines = ["# Tables regenerated from saved numerical inputs", "",
             "No new observations. Interval endpoints are retained from the original analyses;",
             "aggregate saved means do not suffice to rerun their paired-round bootstraps.", "",
             "## Table 1 — direct binary generation", "",
             "| N | Library ms | Direct ms | Reduction % | Ratio interval |",
             "|---:|---:|---:|---:|---|"]
    for r in data["projection_summary"]:
        old, new = r["LIBRARY"]["wall"], r["DIRECT"]["wall"]
        lines.append(f'| {r["n"]:,} | {1000*old:.2f} | {1000*new:.2f} | {100*(1-new/old):.1f} | [{r["wall_ratio_lo"]:.4f}, {r["wall_ratio_hi"]:.4f}] |')
    lines += ["", "## Table 2 — order maintenance", "",
              "| N | Wall ratio | Ratio interval | Reduction % | CPU ratio |",
              "|---:|---:|---|---:|---:|"]
    for r in data["ordering_summary"]:
        lines.append(f'| {r["n"]:,} | {r["wall_ratio"]:.4f} | [{r["wall_ratio_lo"]:.4f}, {r["wall_ratio_hi"]:.4f}] | {100*(1-r["wall_ratio"]):.1f} | {r["cpu_ratio"]:.4f} |')
    lines += ["", "## Table 3 — selected sampling configurations", "",
              "| Payoff | Method | N × k | Estimated MSE | 95% upper | Warm ms |",
              "|---|---|---:|---:|---:|---:|"]
    for condition, payoff in (("case_07_distant", "One-sided"), ("case_08_distant", "Quadratic")):
        rows = target_rows(data, condition)
        for m in METHODS:
            r = rows[m]
            lines.append(f'| {payoff} | {LABELS[m]} | {r["n"]} × {r["k"]} | {r["risk"]:.5g} | {r["risk_hi"]:.5g} | {1000*r["wall"]:.2f} |')
    lines += ["", "## Table 4 — acquisition-inclusive accounting", "",
              "| Payoff | Method | First evaluation s | At R=1000, ms/evaluation |",
              "|---|---|---:|---:|"]
    for condition, payoff in (("case_07_distant", "One-sided"), ("case_08_distant", "Quadratic")):
        for m, r in target_rows(data, condition).items():
            lines.append(f'| {payoff} | {LABELS[m]} | {r["all_acquired_setup"]+r["wall"]:.3f} | {1000*(r["all_acquired_setup"]/1000+r["wall"]):.2f} |')
    lines += ["", "Cost convention: recorded mask-initialization cost is included for the baseline implementations.",
              "This is not an intrinsic lower bound for aligned-innovation CRN.",
              "First-use and reuse figures are component sums, not new timed end-to-end experiments.", ""]
    (ROOT / "TABLES.md").write_text("\n".join(lines), encoding="utf-8")


def figures(data):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.axisbelow": True, "svg.fonttype": "none", "svg.hashsalt": "binary-array-rqmc"})
    out = ROOT.parent / "figures"
    out.mkdir(exist_ok=True)
    components = [("Net construction", ("net_generate",), "#386CB0"),
                  ("Point sort", ("point_sort",), "#76B7D5"),
                  ("Binary projection", ("projection",), "#E6B450"),
                  ("State order", ("state_sort", "state_order"), "#9461A2"),
                  ("Transition + stopping", ("transition",), "#087F5B"),
                  ("Setup / payoff / noise", ("setup", "payoff", "noise"), "#B6BDC5")]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.7), sharey=True)
    for ax, batch, methods, title in zip(
            axs, ("projection", "ordering"), (("LIBRARY", "DIRECT"), ("DIRECT", "MAINTAINED")),
            ("A  Direct binary generation\n11 September", "B  Exact order maintenance\n12 September")):
        totals = [0., 0.]
        for label, keys, color in components:
            heights = [1000*statistics.mean(sum(r["times"].get(k, 0.) for k in keys)
                       for r in data["profiles"] if r["batch"] == batch and r["method"] == method)
                       for method in methods]
            ax.bar(range(2), heights, bottom=totals, width=.58, color=color, label=label)
            totals = [x+y for x,y in zip(totals, heights)]
        for x, value in enumerate(totals):
            ax.text(x, value+1.4, f"{value:.1f} ms", ha="center", fontsize=10)
        ax.set_xticks(range(2), ["Library" if m == "LIBRARY" else "Direct binary" if m == "DIRECT" else "Order-maintained" for m in methods])
        ax.set_title(title, loc="left", fontsize=11, pad=12)
        ax.grid(axis="y", color="#DDE1E4", linewidth=.6)
        ax.set_xlim(-.7, 1.7)
    axs[0].set_ylabel("Sum of instrumented component times (ms)")
    axs[0].set_ylim(0, max(p.get_height()+p.get_y() for a in axs for p in a.patches)*1.15)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(.5,.02))
    fig.subplots_adjust(left=.08, right=.99, top=.83, bottom=.23, wspace=.15)
    fig.suptitle("The two rewrites remove different work   |   N = 4,096", x=.08, ha="left", y=.985, fontsize=13)
    for ext in ("png", "svg"):
        fig.savefig(out / ("component-costs."+ext), dpi=180, metadata={"Date": None} if ext=="svg" else None)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
    counts = np.geomspace(1, 100000, 500)
    for ax, condition, title in zip(axs, ("case_07_distant", "case_08_distant"),
                                    ("A  One-sided payoff   |   target MSE 0.001", "B  Quadratic payoff   |   target MSE 0.01")):
        rows = target_rows(data, condition)
        for m in METHODS:
            r = rows[m]
            ax.plot(counts, 1000*(r["all_acquired_setup"]/counts+r["wall"]), label=LABELS[m], color=COLORS[m], linewidth=2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(1,100000); ax.set_ylim(8,100000)
        ax.set_xticks([1,10,100,1000,10000,100000])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p: f"{v/1000:.0f}k" if v >= 1000 else f"{v:.0f}"))
        ax.set_yticks([10,100,1000,10000,100000])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p: f"{v:,.0f}"))
        ax.grid(which="major", color="#DDE1E4", linewidth=.6)
        ax.set_xlabel("Reuse count R (estimator evaluations)")
        ax.set_title(title, loc="left", fontsize=10.5, pad=12)
    axs[0].set_ylabel("Amortized cost per estimator evaluation (ms)")
    q = target_rows(data, "case_08_distant")
    cross = crossover(q["MAINTAINED"], q["CRN"])
    ycross = 1000*(q["MAINTAINED"]["all_acquired_setup"]/cross+q["MAINTAINED"]["wall"])
    axs[1].plot(cross,ycross,"o",color=COLORS["MAINTAINED"],ms=5)
    axs[1].annotate("Overtakes measured CRN\nat R = 2,799", xy=(cross,ycross), xytext=(350,350),
                    fontsize=9, ha="left", arrowprops={"arrowstyle":"->","color":"#555555","linewidth":.8})
    handles, labels=axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(.5,.025))
    fig.subplots_adjust(left=.09, right=.965, top=.83, bottom=.23, wspace=.15)
    fig.suptitle("Warm speed does not determine first-use cost", x=.09, ha="left", y=.985, fontsize=13)
    for ext in ("png","svg"):
        fig.savefig(out / ("reuse-costs."+ext), dpi=180, metadata={"Date": None} if ext=="svg" else None)
    plt.close(fig)


def main():
    data=load_data()
    validate_data(data)
    write_tables(data)
    figures(data)
    print("Saved-input checks passed; regenerated Tables 1-4 and two figures. No simulation was run.")


if __name__ == "__main__":
    main()
