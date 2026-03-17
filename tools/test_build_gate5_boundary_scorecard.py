#!/usr/bin/env python3
"""Regression test for build_gate5_boundary_scorecard.py."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools" / "build_gate5_boundary_scorecard.py"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8", newline="\n")


def build_gate5_run(
    root: Path,
    label: str,
    surface_title: str,
    proj_id: str,
    input_payload: object,
    metrics_body: str,
    *,
    n_token_rows_total: int = 2,
    n_loop_rows_valid: int = 2,
    n_loop_rows_missing: int = 0,
) -> Path:
    gate5_out = root / label / "gate5_out"
    manifest = {
        "spec_version": "v0.1.0-ssot.draft.0",
        "spec_hash_blake3": "abc",
        "method_id": "transport_loop_residual_experiment_v1",
        "evaluation_mode_id": "supervised_v1",
        "model_id": "Qwen/Qwen2.5-1.5B",
        "model_revision": "rev",
        "seed": 7,
        "perm_r": 2000,
        "splus_def_id": "splus",
        "sminus_def_id": "sminus",
        "proj_id": proj_id,
        "n_samples_total": 2,
        "n_token_rows_total": n_token_rows_total,
        "n_loop_rows_valid": n_loop_rows_valid,
        "n_loop_rows_missing": n_loop_rows_missing,
    }
    write_json(gate5_out / "manifest.json", manifest)
    report = "\n".join(
        [
            "# Gate5 Aggregate Report",
            "",
            f"Surface: {surface_title}",
            "",
            metrics_body.strip(),
            "",
        ]
    )
    write_text(gate5_out / "gate5_aggregate_report.md", report)
    write_json(root / label / "gate4_input.json", input_payload)
    return gate5_out


def run_scorecard(tmp: Path, run_args: list[str]) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--surface",
        "cfa",
        "--out",
        str(tmp / "scorecard.md"),
        "--csv-out",
        str(tmp / "scorecard.csv"),
        *run_args,
    ]
    return subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        input_payload = {"samples": [{"sample_id": 1}, {"sample_id": 3}]}

        fwht_out = build_gate5_run(
            tmp,
            "fwht",
            "CFA",
            "fwht_pad_pow2_take8_v1",
            input_payload,
            "\n".join(
                [
                    "- global_auprc_F: 0.181275",
                    "- global_auprc_rotor_loop_chordal_v1: 0.165703",
                    "- mean_sample_auprc_F: 0.419987",
                    "- mean_sample_auprc_rotor_loop_chordal_v1: 0.396926",
                    "- mean_sample_hit_at_10_F: 3.750000",
                    "- mean_sample_hit_at_10_rotor: 3.180000",
                    "- mean_first_hit_distance_F: 1.620000",
                    "- mean_first_hit_distance_rotor: -1.900000",
                ]
            ),
        )
        origin_out = build_gate5_run(
            tmp,
            "origin",
            "CFA",
            "origin_span_projection_v2",
            input_payload,
            "\n".join(
                [
                    "- global_auprc_F: 0.170000",
                    "- global_auprc_rotor_loop_chordal_v1: 0.150000",
                    "- mean_sample_auprc_F: 0.400000",
                    "- mean_sample_auprc_rotor_loop_chordal_v1: 0.380000",
                    "- mean_sample_hit_at_10_F: 3.500000",
                    "- mean_sample_hit_at_10_rotor: 3.000000",
                    "- mean_first_hit_distance_F: 1.000000",
                    "- mean_first_hit_distance_rotor: -1.000000",
                ]
            ),
        )
        relation_out = build_gate5_run(
            tmp,
            "relation",
            "CFA",
            "local_relation_affine_lift_v2",
            input_payload,
            "\n".join(
                [
                    "- global_auprc_F: 0.122385",
                    "- global_auprc_rotor_loop_chordal_v1: 0.118193",
                    "- mean_sample_auprc_F: 0.311803",
                    "- mean_sample_auprc_rotor_loop_chordal_v1: 0.308614",
                    "- mean_sample_hit_at_10_F: 2.000000",
                    "- mean_sample_hit_at_10_rotor: 2.210000",
                    "- mean_first_hit_distance_F: -4.550000",
                    "- mean_first_hit_distance_rotor: -5.940000",
                ]
            ),
        )

        boundary_manifest = {
            "samples_root": "runs/cfa_batch_primaryE_native_raw/samples",
            "sample_ids": [1, 3],
            "boundary_outcome_counts": {
                "materialized_rank3": 2,
                "sign_unstable": 0,
            },
            "raw_span_path_counts": {
                "modulated": 1,
                "fallback_materialized": 1,
            },
        }
        write_json(tmp / "origin" / "native_local_span_build_manifest.json", boundary_manifest)
        write_json(tmp / "relation" / "native_local_span_build_manifest.json", boundary_manifest)

        completed = run_scorecard(
            tmp,
            [
                "--run",
                f"label=fwht;gate5_out={fwht_out};input={tmp / 'fwht' / 'gate4_input.json'}",
                "--run",
                (
                    f"label=origin_v2;gate5_out={origin_out};input={tmp / 'origin' / 'gate4_input.json'};"
                    f"boundary_manifest={tmp / 'origin' / 'native_local_span_build_manifest.json'}"
                ),
                "--run",
                (
                    f"label=relation_affine_v2;gate5_out={relation_out};input={tmp / 'relation' / 'gate4_input.json'};"
                    f"boundary_manifest={tmp / 'relation' / 'native_local_span_build_manifest.json'}"
                ),
            ],
        )
        if completed.returncode != 0:
            raise SystemExit(
                f"scorecard command failed rc={completed.returncode}\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )

        scorecard = (tmp / "scorecard.md").read_text(encoding="utf-8")
        csv_body = (tmp / "scorecard.csv").read_text(encoding="utf-8")
        assert "exact_sample_ids_match: PASS" in scorecard
        assert "gate5_fixed_fields_match: PASS" in scorecard
        assert "loop_row_coverage_match: PASS" in scorecard
        assert "native_samples_root_match: PASS (runs/cfa_batch_primaryE_native_raw/samples)" in scorecard
        assert "| relation_affine_v2 | local_relation_affine_lift_v2 | 2 | 0 | 2 | 0 | 0 | 1 | 1 | 0.122385 |" in scorecard
        assert "run,proj_id,n_samples,n_token_rows_total,n_loop_valid,n_loop_missing,sample_id_sha256,boundary_rank3,boundary_sign_unstable,boundary_raw_span_axis_collapse,boundary_modulated_rows,boundary_fallback_rows" in csv_body
        assert "origin_v2,origin_span_projection_v2,2,2,2,0," in csv_body

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        input_payload = {"samples": [{"sample_id": 1}, {"sample_id": 3}]}

        fwht_out = build_gate5_run(
            tmp,
            "fwht",
            "CFA",
            "fwht_pad_pow2_take8_v1",
            input_payload,
            "- global_auprc_F: 0.181275",
        )
        origin_out = build_gate5_run(
            tmp,
            "origin",
            "CFA",
            "origin_span_projection_v2",
            input_payload,
            "- global_auprc_F: 0.170000",
        )
        relation_out = build_gate5_run(
            tmp,
            "relation",
            "CFA",
            "local_relation_affine_lift_v0",
            input_payload,
            "- global_auprc_F: 0.122385",
            n_token_rows_total=2,
            n_loop_rows_valid=1,
            n_loop_rows_missing=1,
        )

        boundary_manifest = {
            "samples_root": "runs/cfa_batch_primaryE_native_raw/samples",
            "sample_ids": [1, 3],
            "boundary_outcome_counts": {
                "materialized_rank3": 2,
                "sign_unstable": 0,
            },
            "raw_span_axis_available_counts": {
                "True": 1,
                "False": 1,
            },
        }
        write_json(tmp / "origin" / "native_local_span_build_manifest.json", boundary_manifest)
        write_json(tmp / "relation" / "native_local_span_build_manifest.json", boundary_manifest)

        completed = run_scorecard(
            tmp,
            [
                "--run",
                f"label=fwht;gate5_out={fwht_out};input={tmp / 'fwht' / 'gate4_input.json'}",
                "--run",
                (
                    f"label=origin_v2;gate5_out={origin_out};input={tmp / 'origin' / 'gate4_input.json'};"
                    f"boundary_manifest={tmp / 'origin' / 'native_local_span_build_manifest.json'}"
                ),
                "--run",
                (
                    f"label=relation_affine_v0;gate5_out={relation_out};input={tmp / 'relation' / 'gate4_input.json'};"
                    f"boundary_manifest={tmp / 'relation' / 'native_local_span_build_manifest.json'}"
                ),
            ],
        )
        if completed.returncode != 0:
            raise SystemExit(
                f"scorecard command failed rc={completed.returncode}\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )

        scorecard = (tmp / "scorecard.md").read_text(encoding="utf-8")
        assert "loop_row_coverage_match: FAIL" in scorecard
        assert "relation_affine_v0(token_total=2,loop_valid=1,missing=1)" in scorecard

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        input_payload = {"samples": [{"sample_id": 1}, {"sample_id": 3}]}

        fwht_out = build_gate5_run(
            tmp,
            "fwht",
            "CFA",
            "fwht_pad_pow2_take8_v1",
            input_payload,
            "- global_auprc_F: 0.181275",
        )
        origin_out = build_gate5_run(
            tmp,
            "origin",
            "CFA",
            "origin_span_projection_v2",
            input_payload,
            "- global_auprc_F: 0.170000",
        )

        completed = run_scorecard(
            tmp,
            [
                "--run",
                f"label=fwht;gate5_out={fwht_out};input={tmp / 'fwht' / 'gate4_input.json'}",
                "--run",
                f"label=origin_v2;gate5_out={origin_out};input={tmp / 'origin' / 'gate4_input.json'}",
            ],
        )
        assert completed.returncode != 0
        assert "boundary_manifest is required for non-baseline runs: origin_v2" in completed.stderr

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        input_payload = {"samples": [{"sample_id": 1}, {"sample_id": 3}]}

        fwht_out = build_gate5_run(
            tmp,
            "fwht",
            "CFA",
            "fwht_pad_pow2_take8_v1",
            input_payload,
            "- global_auprc_F: 0.181275",
        )
        gate6_out = build_gate5_run(
            tmp,
            "gate6",
            "CFA",
            "gate6_native_local_span_local8_v1",
            input_payload,
            "- global_auprc_F: 0.137833",
        )
        gate6_gate5_manifest_path = gate6_out / "manifest.json"
        gate6_gate5_manifest = json.loads(gate6_gate5_manifest_path.read_text(encoding="utf-8"))
        gate6_gate5_manifest["splus_def_id"] = "gate6_local_span_coord_splus_v1"
        gate6_gate5_manifest["sminus_def_id"] = "gate6_local_span_coord_sminus_v1"
        gate6_gate5_manifest["boundary_origin"] = "gate6_native_local_span_local8_v1"
        gate6_gate5_manifest["compatibility_schema_id"] = "gate6_local8_compat_input_v1"
        write_json(gate6_gate5_manifest_path, gate6_gate5_manifest)
        gate6_manifest = {
            "samples_root": "runs/cfa_batch_primaryE_native_raw/samples",
            "sample_ids": [1, 3],
            "rank_local_counts": {
                "3": 2,
            },
        }
        write_json(tmp / "gate6" / "manifest.json", gate6_manifest)

        completed = run_scorecard(
            tmp,
            [
                "--run",
                f"label=fwht;gate5_out={fwht_out};input={tmp / 'fwht' / 'gate4_input.json'}",
                "--run",
                (
                    f"label=gate6a_v0;gate5_out={gate6_out};input={tmp / 'gate6' / 'gate4_input.json'};"
                    f"boundary_manifest={tmp / 'gate6' / 'manifest.json'}"
                ),
            ],
        )
        if completed.returncode != 0:
            raise SystemExit(
                f"scorecard command failed rc={completed.returncode}\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )

        scorecard = (tmp / "scorecard.md").read_text(encoding="utf-8")
        assert "gate5_fixed_fields_match: PASS" in scorecard
        assert "boundary_identity_drift: YES" in scorecard
        assert "boundary_identity_drift_detail:" in scorecard
        assert "| gate6a_v0 | gate6_native_local_span_local8_v1 | 2 | 0 | 2 | 0 | 0 |  |  | 0.137833 |" in scorecard
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
