#!/usr/bin/env python3
"""Validate Rust Gate4 artifacts against Python-computed expectations."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import eval_triality_token as evaltok


FLOAT_TOL = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Gate4 parity against Python formulas.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--token-features-csv", required=True)
    parser.add_argument("--sample-summary-csv", required=True)
    parser.add_argument("--run-summary-csv", required=True)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--expected-dataset-hash-blake3")
    parser.add_argument("--expected-spec-hash-raw-blake3")
    parser.add_argument("--expected-spec-hash-blake3")
    parser.add_argument("--out")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object: {path}")
    return obj


def load_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"missing CSV header: {path}")
        return list(reader)


def parse_opt_float(text: Any) -> Optional[float]:
    raw = str(text).strip()
    if not raw:
        return None
    out = float(raw)
    if not math.isfinite(out):
        raise ValueError(f"non-finite float in CSV: {text!r}")
    return out


def compare_opt_float(left: Optional[float], right: Optional[float], label: str) -> None:
    if left is None or right is None:
        if left != right:
            raise AssertionError(f"{label}: expected {left!r}, got {right!r}")
        return
    if abs(left - right) > FLOAT_TOL:
        raise AssertionError(f"{label}: expected {left:.17e}, got {right:.17e}")


def hit_at_k(labels: Sequence[int], scores: Sequence[float], k: int) -> int:
    indexed = list(range(len(scores)))
    indexed.sort(key=lambda i: (-float(scores[i]), i))
    return sum(int(labels[idx]) for idx in indexed[:k])


def compute_expected_rows(sample: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    token_steps = sorted(sample["token_steps"], key=lambda row: int(row["step"]))
    n = len(token_steps)
    labels_token = [int(row["label_token"]) for row in token_steps]
    score_a = [-float(row["baseline_logprob"]) for row in token_steps]
    score_b = [float(row["baseline_entropy"]) for row in token_steps]
    score_f: List[float] = []
    labels_transition: List[int] = []
    score_c: List[float] = []
    score_d: List[float] = []
    score_e: List[float] = []

    for row in token_steps:
        score_f.append(
            evaltok.d_proj(row["V_8d"], row["Splus_8d"])
            + evaltok.d_proj(row["Splus_8d"], row["Sminus_8d"])
            + evaltok.d_proj(row["Sminus_8d"], row["V_8d"])
        )

    for idx in range(max(0, n - 1)):
        current = token_steps[idx]
        nxt = token_steps[idx + 1]
        labels_transition.append(max(int(current["label_token"]), int(nxt["label_token"])))
        score_c.append(evaltok.d_proj(current["V_8d"], nxt["V_8d"]))
        score_d.append(
            evaltok.d_proj(current["V_8d"], current["Splus_8d"])
            + evaltok.d_proj(current["Splus_8d"], nxt["V_8d"])
        )
        score_e.append(
            evaltok.d_proj(current["V_8d"], current["Sminus_8d"])
            + evaltok.d_proj(current["Sminus_8d"], nxt["V_8d"])
        )

    auprc_a = evaltok.average_precision(labels_token, score_a)
    auprc_b = evaltok.average_precision(labels_token, score_b)
    if auprc_a is not None and auprc_b is not None:
        if auprc_a >= auprc_b:
            best_baseline_auprc = auprc_a
            best_baseline_name = "A"
        else:
            best_baseline_auprc = auprc_b
            best_baseline_name = "B"
    elif auprc_a is not None:
        best_baseline_auprc = auprc_a
        best_baseline_name = "A"
    elif auprc_b is not None:
        best_baseline_auprc = auprc_b
        best_baseline_name = "B"
    else:
        best_baseline_auprc = None
        best_baseline_name = "none"

    auprc_c = evaltok.average_precision(labels_transition, score_c)
    auprc_d = evaltok.average_precision(labels_transition, score_d)
    auprc_e = evaltok.average_precision(labels_transition, score_e)
    auprc_f = evaltok.average_precision(labels_token, score_f)
    delta = None if (auprc_e is None or best_baseline_auprc is None) else auprc_e - best_baseline_auprc

    expected_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(token_steps):
        if idx + 1 < n:
            missing_reason = "none"
            label_transition = labels_transition[idx]
            row_c = score_c[idx]
            row_d = score_d[idx]
            row_e = score_e[idx]
        else:
            missing_reason = "final_step_no_successor"
            label_transition = 0
            row_c = None
            row_d = None
            row_e = None
        expected_rows.append(
            {
                "sample_id": int(sample["sample_id"]),
                "variant": str(sample.get("variant", "unknown")),
                "world_type": "" if sample.get("world_type") is None else str(sample.get("world_type")),
                "step": int(row["step"]),
                "absolute_pos": int(row["absolute_pos"]),
                "token_id": int(row["token_id"]),
                "token_text": str(row["token_str"]),
                "answer_char_start": row.get("answer_char_start"),
                "answer_char_end": row.get("answer_char_end"),
                "label_token": int(row["label_token"]),
                "label_transition": int(label_transition),
                "defect_span_id": "" if row.get("defect_span_id") in (None, "") else str(row["defect_span_id"]),
                "label_coverage_ratio": float(sample["label_coverage_ratio"]),
                "exact_token_match_ratio": float(sample["exact_token_match_ratio"]),
                "transition_missing_reason": missing_reason,
                "score_a": float(score_a[idx]),
                "score_b": float(score_b[idx]),
                "score_c": row_c,
                "score_d": row_d,
                "score_e": row_e,
                "score_f": float(score_f[idx]),
            }
        )

    summary = {
        "sample_id": int(sample["sample_id"]),
        "variant": str(sample.get("variant", "unknown")),
        "world_type": "" if sample.get("world_type") is None else str(sample.get("world_type")),
        "n_token_steps": n,
        "n_transition_steps": len(labels_transition),
        "positive_token_count": sum(1 for x in labels_token if x == 1),
        "positive_transition_count": sum(1 for x in labels_transition if x == 1),
        "label_coverage_ratio": float(sample["label_coverage_ratio"]),
        "exact_token_match_ratio": float(sample["exact_token_match_ratio"]),
        "triplets_sha256": str(sample["triplets_sha256"]),
        "labels_sha256": str(sample["labels_sha256"]),
        "auprc_a": auprc_a,
        "auprc_b": auprc_b,
        "auprc_c": auprc_c,
        "auprc_d": auprc_d,
        "auprc_e": auprc_e,
        "auprc_f": auprc_f,
        "best_baseline_name": best_baseline_name,
        "delta_auprc_e_vs_best_baseline": delta,
        "hit_at_10_e": hit_at_k(labels_transition, score_e, 10),
    }
    return expected_rows, summary


def validate_token_rows(
    run_id: str,
    input_samples: Sequence[Dict[str, Any]],
    token_csv_rows: Sequence[Dict[str, str]],
) -> Dict[str, Any]:
    expected_rows: List[Dict[str, Any]] = []
    for sample in sorted(input_samples, key=lambda row: int(row["sample_id"])):
        sample_rows, _ = compute_expected_rows(sample)
        expected_rows.extend(sample_rows)

    if len(expected_rows) != len(token_csv_rows):
        raise AssertionError(
            f"token row count mismatch: expected {len(expected_rows)}, got {len(token_csv_rows)}"
        )

    for idx, (expected, actual) in enumerate(zip(expected_rows, token_csv_rows)):
        label_prefix = f"token_row[{idx}] sample={expected['sample_id']} step={expected['step']}"
        if actual["run_id"] != run_id:
            raise AssertionError(f"{label_prefix}: run_id mismatch")
        for key, csv_key in [
            ("sample_id", "sample_id"),
            ("variant", "variant"),
            ("world_type", "world_type"),
            ("step", "step"),
            ("absolute_pos", "absolute_pos"),
            ("token_id", "token_id"),
            ("token_text", "token_text"),
            ("label_token", "label_token"),
            ("label_transition", "label_transition"),
            ("defect_span_id", "defect_span_id"),
            ("transition_missing_reason", "transition_missing_reason"),
        ]:
            expected_value = "" if expected[key] is None else str(expected[key])
            if actual[csv_key] != expected_value:
                raise AssertionError(
                    f"{label_prefix}: {csv_key} mismatch: expected {expected_value!r}, got {actual[csv_key]!r}"
                )

        for key in ["answer_char_start", "answer_char_end"]:
            expected_value = expected[key]
            actual_value = actual[key]
            if expected_value is None:
                if actual_value != "":
                    raise AssertionError(f"{label_prefix}: {key} expected empty, got {actual_value!r}")
            elif actual_value != str(expected_value):
                raise AssertionError(
                    f"{label_prefix}: {key} mismatch: expected {expected_value}, got {actual_value!r}"
                )

        compare_opt_float(expected["label_coverage_ratio"], parse_opt_float(actual["label_coverage_ratio"]), f"{label_prefix} label_coverage_ratio")
        compare_opt_float(expected["exact_token_match_ratio"], parse_opt_float(actual["exact_token_match_ratio"]), f"{label_prefix} exact_token_match_ratio")
        compare_opt_float(expected["score_a"], parse_opt_float(actual["score_A_logprob"]), f"{label_prefix} score_A")
        compare_opt_float(expected["score_b"], parse_opt_float(actual["score_B_entropy"]), f"{label_prefix} score_B")
        compare_opt_float(expected["score_c"], parse_opt_float(actual["score_C_v_curvature"]), f"{label_prefix} score_C")
        compare_opt_float(expected["score_d"], parse_opt_float(actual["score_D_v_splus_vnext"]), f"{label_prefix} score_D")
        compare_opt_float(expected["score_e"], parse_opt_float(actual["score_E_v_sminus_vnext"]), f"{label_prefix} score_E")
        compare_opt_float(expected["score_f"], parse_opt_float(actual["score_F_loop"]), f"{label_prefix} score_F")

    return {"token_rows_checked": len(expected_rows)}


def validate_sample_rows(
    run_id: str,
    input_samples: Sequence[Dict[str, Any]],
    summary_csv_rows: Sequence[Dict[str, str]],
) -> Dict[str, Any]:
    expected = [
        compute_expected_rows(sample)[1]
        for sample in sorted(input_samples, key=lambda row: int(row["sample_id"]))
    ]
    if len(expected) != len(summary_csv_rows):
        raise AssertionError(
            f"sample summary count mismatch: expected {len(expected)}, got {len(summary_csv_rows)}"
        )

    for idx, (exp, act) in enumerate(zip(expected, summary_csv_rows)):
        label_prefix = f"sample_row[{idx}] sample={exp['sample_id']}"
        if act["run_id"] != run_id:
            raise AssertionError(f"{label_prefix}: run_id mismatch")
        for key in [
            "sample_id",
            "variant",
            "world_type",
            "n_token_steps",
            "n_transition_steps",
            "positive_token_count",
            "positive_transition_count",
            "triplets_sha256",
            "labels_sha256",
            "best_baseline_name",
            "hit_at_10_e",
        ]:
            csv_key = "hit_at_10_E" if key == "hit_at_10_e" else key
            expected_value = str(exp[key])
            if act[csv_key] != expected_value:
                raise AssertionError(
                    f"{label_prefix}: {csv_key} mismatch: expected {expected_value!r}, got {act[csv_key]!r}"
                )

        compare_opt_float(exp["label_coverage_ratio"], parse_opt_float(act["label_coverage_ratio"]), f"{label_prefix} label_coverage_ratio")
        compare_opt_float(exp["exact_token_match_ratio"], parse_opt_float(act["exact_token_match_ratio"]), f"{label_prefix} exact_token_match_ratio")
        compare_opt_float(exp["auprc_a"], parse_opt_float(act["auprc_A"]), f"{label_prefix} auprc_A")
        compare_opt_float(exp["auprc_b"], parse_opt_float(act["auprc_B"]), f"{label_prefix} auprc_B")
        compare_opt_float(exp["auprc_c"], parse_opt_float(act["auprc_C"]), f"{label_prefix} auprc_C")
        compare_opt_float(exp["auprc_d"], parse_opt_float(act["auprc_D"]), f"{label_prefix} auprc_D")
        compare_opt_float(exp["auprc_e"], parse_opt_float(act["auprc_E"]), f"{label_prefix} auprc_E")
        compare_opt_float(exp["auprc_f"], parse_opt_float(act["auprc_F"]), f"{label_prefix} auprc_F")
        compare_opt_float(
            exp["delta_auprc_e_vs_best_baseline"],
            parse_opt_float(act["delta_auprc_E_vs_best_baseline"]),
            f"{label_prefix} delta_auprc_E_vs_best_baseline",
        )

    return {"sample_rows_checked": len(expected)}


def validate_manifest(
    input_json_path: Path,
    token_csv_path: Path,
    summary_csv_path: Path,
    run_summary_csv_path: Path,
    manifest_path: Path,
    expected_identity: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    manifest = load_json(manifest_path)
    checks = {
        "input_json_sha256": sha256_file(input_json_path),
        "token_features_sha256": sha256_file(token_csv_path),
        "sample_summary_sha256": sha256_file(summary_csv_path),
        "run_summary_sha256": sha256_file(run_summary_csv_path),
    }
    for key, expected in checks.items():
        actual = str(manifest.get(key) or "")
        if actual != expected:
            raise AssertionError(f"manifest {key} mismatch: expected {expected}, got {actual}")
    identity_keys = [
        "dataset_hash_blake3",
        "spec_hash_raw_blake3",
        "spec_hash_blake3",
    ]
    placeholder_zero64 = "0" * 64
    identity_values = {key: str(manifest.get(key) or "") for key in identity_keys}
    identity_expected_values = {key: expected_identity.get(key) for key in identity_keys}
    if any(value is None for value in identity_expected_values.values()):
        provenance_verdict = "UNSPECIFIED_EXPECTATION"
    elif any(identity_values[key] != identity_expected_values[key] for key in identity_keys):
        provenance_verdict = "FAIL"
    elif any(identity_values[key] == placeholder_zero64 for key in identity_keys):
        provenance_verdict = "PLACEHOLDER_IDENTITY"
    else:
        provenance_verdict = "PASS"
    return {
        "manifest_sha_checks": sorted(checks.keys()),
        "provenance_verdict": provenance_verdict,
        "identity_values": identity_values,
        "identity_expected_values": identity_expected_values,
    }


def write_report(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line.rstrip("\n") + "\n")


def main() -> int:
    args = parse_args()
    input_json_path = Path(args.input_json)
    token_csv_path = Path(args.token_features_csv)
    summary_csv_path = Path(args.sample_summary_csv)
    run_summary_csv_path = Path(args.run_summary_csv)
    manifest_path = Path(args.manifest_json)

    payload = load_json(input_json_path)
    input_samples = list(payload["samples"])
    metadata = payload["metadata"]
    run_id = load_json(manifest_path)["run_id"]

    token_rows = load_csv_dicts(token_csv_path)
    summary_rows = load_csv_dicts(summary_csv_path)
    expected_identity = {
        "dataset_hash_blake3": args.expected_dataset_hash_blake3,
        "spec_hash_raw_blake3": args.expected_spec_hash_raw_blake3,
        "spec_hash_blake3": args.expected_spec_hash_blake3,
    }

    token_result = validate_token_rows(run_id, input_samples, token_rows)
    sample_result = validate_sample_rows(run_id, input_samples, summary_rows)
    manifest_result = validate_manifest(
        input_json_path,
        token_csv_path,
        summary_csv_path,
        run_summary_csv_path,
        manifest_path,
        expected_identity,
    )
    report_lines = [
        f"input_json={input_json_path.as_posix()}",
        f"manifest_json={manifest_path.as_posix()}",
        f"token_features_csv={token_csv_path.as_posix()}",
        f"sample_summary_csv={summary_csv_path.as_posix()}",
        f"run_summary_csv={run_summary_csv_path.as_posix()}",
        f"run_id={run_id}",
        f"model_id={metadata['model_id']}",
        f"model_revision={metadata['model_revision']}",
        f"seed={metadata['seed']}",
        f"primary_score={metadata.get('primary_score')}",
        f"perm_r={metadata.get('perm_r')}",
        f"token_rows_checked={token_result['token_rows_checked']}",
        f"sample_rows_checked={sample_result['sample_rows_checked']}",
        "manifest_sha_checks=" + ",".join(manifest_result["manifest_sha_checks"]),
        f"provenance_verdict={manifest_result['provenance_verdict']}",
        "manifest_identity="
        + ",".join(f"{k}:{v}" for k, v in sorted(manifest_result["identity_values"].items())),
        "manifest_identity_expected="
        + ",".join(
            f"{k}:{'' if v is None else v}"
            for k, v in sorted(manifest_result["identity_expected_values"].items())
        ),
        "parity_verdict=PASS",
    ]

    if args.out:
        write_report(Path(args.out), report_lines)

    print("\n".join(report_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
