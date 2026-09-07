"""Extract/check report aggregates; never execute a model or an official reducer.

Without --source-root, --check uses only the accompanying report data and image.
With --source-root, it also reconstructs aggregates from retained local exports.
Without --check, print the reconstructed JSON to stdout. No files are written.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
EXPECTED = {
    "crd01_result": "0ab0a113d300afe3470ca59c5016dac96116f1889c8896b67b53b74d861376b3",
    "crd02_result": "b1972c84ea147c4f7653ce9de6cea46b9517980b7a68ffaa196a36ed50a7a8af",
    "crd02_receipt": "8b5213596b2385c07289e9bafd87aaa99bbeb429f60d52d19d78481a2a8f41ef",
    "protocol": "c07a4d68d2c8f52fc5f1513f731b96ceb55e650f1165a247606751c68fa470af",
}
ARM_FIELDS = (
    "first_8_batch_mean_loss", "last_8_batch_mean_loss", "pre_action_error",
    "post_action_error", "post_decoder_error", "post_rho", "post_alias_error",
    "descriptive_counts",
)
LENGTHS = (0, 1, 3, 4, 5, 8, 9, 12, 16, 17, 32)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def extract(root):
    sources = {}

    def read(name, relative, expected=None, parse=True):
        payload = (root / relative).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if expected:
            require(digest == expected, f"Source hash mismatch: {name}")
        sources[name] = {"retained_relative_path": relative, "sha256": digest,
                         "bytes": len(payload)}
        return json.loads(payload) if parse else payload

    r1 = read("crd01_result", "crd_01_official_execution_v0.1/official_run/RESULT.local.json",
              EXPECTED["crd01_result"])
    base = "crd_02_official_run_v0.1_continuation_v2/review_export/"
    result = read("crd02_result", base + "COMPLETE_RESULT_TABLE.json", EXPECTED["crd02_result"])
    receipt = read("crd02_receipt", base + "FINAL_RECEIPT.json", EXPECTED["crd02_receipt"])
    read("protocol", "CRD_02_PROSPECTIVE_PROTOCOL_v0.2.local.md", EXPECTED["protocol"], False)
    require(receipt["result_table_sha256"] == sources["crd02_result"]["sha256"], "Result binding")
    metrics = read("primary_metrics", base + "critical_parents/primary_metric_table_sha256.bytes",
                   receipt["parent_hashes"]["primary_metric_table_sha256"])
    summary = read("postmortem_summary", "crd_02_postmortem/summary.json")
    read("postmortem_plotter", "crd_02_postmortem/postmortem.py", parse=False)
    read("training_figure", "crd_02_postmortem/training_losses.png", parse=False)
    for summary_key, source_key in (("result", "crd02_result"), ("receipt", "crd02_receipt"),
                                    ("protocol", "protocol")):
        require(summary["sources"][summary_key]["sha256"] == sources[source_key]["sha256"],
                f"Postmortem input mismatch: {summary_key}")

    rows = metrics["records"]
    require(len(rows) == metrics["record_count"] == 242, "Primary record count")
    require(len({r["ticket_token"] for r in rows}) == 242, "Duplicate primary ticket")
    counts = {key: 0 for key in ("K_A", "K_F", "K_V", "K_B", "K_C_descriptive_only")}
    for row in rows:
        require(sorted(map(int, row["behavior_error_by_length"])) == list(LENGTHS), "Length set")
        for key, count in (("positive_contrasts", 24), ("scalar_nulls", 52), ("signed_residuals", 152)):
            require(len(row[key]) == count and all(math.isfinite(v) for v in row[key]), key)
        a = row["post_action_error"] <= .05 and row["decoder_error"] <= .05 and (
            row["pre_action_error"] - row["post_action_error"] >= .20)
        f = all(math.isfinite(v) and 0 <= v <= .25 for v in row["behavior_error_by_length"].values())
        v = (all(.030 < x <= 1 for x in row["positive_contrasts"])
             and all(0 <= x < .010 for x in row["scalar_nulls"])
             and all(abs(x) < .010 for x in row["signed_residuals"]))
        for flag, value in (("A", a), ("F", f), ("V", v), ("B", a and f and v)):
            require(row[flag] == value, f"Stored {flag} differs from fixed rule")
            counts["K_" + flag] += int(value)
        counts["K_C_descriptive_only"] += int(row["post_action_error"] <= .00625 and row["decoder_error"] <= .05)
    require(all(result[key] == value for key, value in counts.items()), "Result count mismatch")
    require(counts == summary["official_counts"], "Postmortem count mismatch")
    diagnostics = result["diagnostic_continuous_records"]
    require(len(diagnostics) == 242 and {r["ticket_token"] for r in diagnostics}
            == {r["ticket_token"] for r in rows}, "Diagnostic pairing")
    positive = [v for r in rows for v in r["positive_contrasts"]]
    scalar = [v for r in rows for v in r["scalar_nulls"]]
    signed = [v for r in rows for v in r["signed_residuals"]]
    return {
        "scope": "Selected report aggregates, not a complete experimental reproduction package",
        "source_files": sources,
        "crd01": {key: r1[key] for key in (
            "result_status", "claim_ceiling", "seed_count", "endpoint_count", "critical_count",
            "minimum_endpoint_success_count", "endpoints_passing_population_rule",
            "minimum_positive_endpoint_by_seed", "maximum_scalar_null_by_seed",
            "maximum_signed_abs_by_seed", "interpretation_boundaries")},
        "crd02": {
            "ticket_count": len(rows), "terminal": result["primary_terminal_decision"],
            "counts": counts, "diagnostic_local_count": sum(r["R_RI"] for r in diagnostics),
            "diagnostic_behavior_count": sum(r["F_RI"] for r in diagnostics),
            "arms": {arm: {key: stats[key] for key in ARM_FIELDS}
                     for arm, stats in summary["arms"].items()},
            "lengths": summary["lengths"], "assay_signal": summary["assay_signal"],
            "postmortem_identity_checks": summary["identity_checks"],
            "endpoint_recheck": {
                "positive_count": len(positive), "positive_max": max(positive),
                "positive_above_floor": sum(v > .030 for v in positive),
                "scalar_count": len(scalar), "scalar_abs_max": max(map(abs, scalar)),
                "signed_count": len(signed), "signed_abs_max": max(map(abs, signed)),
            },
        },
    }


def check_snapshot(data):
    require(data["crd01"]["seed_count"] == data["crd02"]["ticket_count"] == 242, "Study sizes")
    require(data["crd01"]["endpoints_passing_population_rule"] == 228, "CRD-01 result")
    require(all(x == 0 for x in data["crd02"]["counts"].values()), "CRD-02 result")
    require(data["crd02"]["terminal"] == "CRD_02_NO_LOCAL_ACTION_ACQUISITION_DETECTED", "Terminal")
    image_hash = hashlib.sha256((HERE / "figures/training_losses.png").read_bytes()).hexdigest()
    require(image_hash == data["source_files"]["training_figure"]["sha256"], "Figure copy mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, help="Retained workstream/local equivalent")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        snapshot = json.loads((HERE / "report_data.json").read_bytes())
        check_snapshot(snapshot)
        if args.source_root:
            require(snapshot == extract(args.source_root), "Retained exports differ from report snapshot")
        print("PASS: " + ("retained-source extraction and report snapshot" if args.source_root
                          else "report snapshot consistency and figure identity (no raw-source audit)"))
    else:
        if not args.source_root:
            parser.error("--source-root is required for extraction")
        print(json.dumps(extract(args.source_root), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
