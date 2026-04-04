#!/usr/bin/env python3
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import run_gate8_candidate_finalize as finalize


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def write_csv(path: Path, fieldnames, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class Gate8CandidateFinalizeTests(unittest.TestCase):
    def test_finalize_writes_manifest_summary_and_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            benchmark_dir = tmp / "benchmark"
            execution_dir = tmp / "execution"

            write_json(
                benchmark_dir / "manifest.json",
                {
                    "rendering_family_id": "transcript_v1",
                    "candidate_set": [],
                    "candidate_granularity_status": "mixed_candidate_label_granularity_v1",
                    "candidate_granularity_note": "fixture",
                },
            )
            write_jsonl(benchmark_dir / "benchmark_rows.jsonl", [{"sample_id": "a"}])
            write_jsonl(
                execution_dir / "sample_registry.jsonl",
                [
                    {
                        "execution_sample_id": 1,
                        "benchmark_sample_id": "a",
                        "cell_id": "clean_support",
                        "world_type": "genealogy",
                        "answer_target_type": "consistent_answer",
                        "world_id": "w0",
                        "rendering_id": "r0",
                        "rendering_family_id": "transcript_v1",
                        "quietness_pair_id": "quiet_pair_000",
                    }
                ],
            )
            write_jsonl(execution_dir / "quietness_pairs.jsonl", [{"quietness_pair_id": "quiet_pair_000"}])
            write_jsonl(
                execution_dir / "extraction_results.jsonl",
                [
                    {
                        "execution_sample_id": 1,
                        "benchmark_sample_id": "a",
                        "cell_id": "clean_support",
                        "rendering_family_id": "transcript_v1",
                        "sample_dir": str(execution_dir / "samples" / "sample_000001"),
                    }
                ],
            )

            for rel in (
                "samples/sample_000001",
                "gate6_native",
                "gate6f",
                "gate6h",
                "gate7c",
                "diagnostics",
            ):
                (execution_dir / rel).mkdir(parents=True, exist_ok=True)

            for rel in (
                "diagnostics/rotation_leakage_per_sample.csv",
                "diagnostics/rotation_leakage_by_cell.csv",
                "diagnostics/rotation_leakage_bridge_report.md",
                "diagnostics/support_closure_per_sample.csv",
                "diagnostics/support_closure_by_cell.csv",
                "diagnostics/support_closure_bridge_report.md",
                "diagnostics/direct_contradiction_dual_anchor_per_sample.csv",
                "diagnostics/direct_contradiction_dual_anchor_by_answer_target.csv",
                "diagnostics/direct_contradiction_dual_anchor_report.md",
            ):
                path = execution_dir / rel
                path.write_text("fixture\n", encoding="utf-8")

            write_csv(
                execution_dir / "gate6f" / "gate6f_token_telemetry.csv",
                ["sample_id", "step", "label_token", "score_F_gram_loop_v1", "sigma_gap_tailkeep_weighted_gram_loop_v2"],
                [{"sample_id": "1", "step": "0", "label_token": "1", "score_F_gram_loop_v1": "0.1", "sigma_gap_tailkeep_weighted_gram_loop_v2": "0.2"}],
            )
            write_csv(
                execution_dir / "gate6h" / "gate6h_token_telemetry.csv",
                ["sample_id", "step", "label_token", "sigma_sqrtgap_tailkeep_object_v2"],
                [{"sample_id": "1", "step": "0", "label_token": "1", "sigma_sqrtgap_tailkeep_object_v2": "0.3"}],
            )
            write_csv(
                execution_dir / "gate7c" / "gate7c_token_telemetry.csv",
                ["sample_id", "step", "label_transition", "progression_anisotropic_closure_v3"],
                [{"sample_id": "1", "step": "0", "label_transition": "1", "progression_anisotropic_closure_v3": "0.4"}],
            )

            def fake_run_subprocess(command):
                out_dir = Path(command[command.index("--out-dir") + 1])
                out_dir.mkdir(parents=True, exist_ok=True)
                write_json(out_dir / "manifest.json", {"run_id": out_dir.name})
                write_csv(
                    out_dir / "standing_summary.csv",
                    [
                        "candidate_id",
                        "label_key",
                        "label_granularity",
                        "metric_id",
                        "direct_global_auprc",
                        "direct_mean_sample_auprc",
                        "direct_mean_hit_at_10",
                        "direct_mean_first_hit_distance",
                        "distributed_global_auprc",
                        "distributed_mean_sample_auprc",
                        "distributed_mean_hit_at_10",
                        "distributed_mean_first_hit_distance",
                        "quiet_mean_delta_max",
                        "quiet_mean_delta_p90",
                        "quiet_mean_iqr_normalized_delta_max",
                        "quiet_mean_top10_inflation",
                    ],
                    [
                        {
                            "candidate_id": out_dir.name,
                            "label_key": "label_token",
                            "label_granularity": "token",
                            "metric_id": "fixture_metric",
                            "direct_global_auprc": "0.1",
                            "direct_mean_sample_auprc": "0.1",
                            "direct_mean_hit_at_10": "1.0",
                            "direct_mean_first_hit_distance": "0.0",
                            "distributed_global_auprc": "0.2",
                            "distributed_mean_sample_auprc": "0.2",
                            "distributed_mean_hit_at_10": "1.0",
                            "distributed_mean_first_hit_distance": "0.0",
                            "quiet_mean_delta_max": "0.0",
                            "quiet_mean_delta_p90": "0.0",
                            "quiet_mean_iqr_normalized_delta_max": "0.0",
                            "quiet_mean_top10_inflation": "0.0",
                        }
                    ],
                )

            summary_rows = [
                {
                    "candidate_id": "F",
                    "label_key": "label_token",
                    "label_granularity": "token",
                    "metric_id": "score_F_gram_loop_v1",
                    "direct_global_auprc": 0.1,
                    "direct_mean_sample_auprc": 0.1,
                    "direct_mean_hit_at_10": 1.0,
                    "direct_mean_first_hit_distance": 0.0,
                    "distributed_global_auprc": 0.2,
                    "distributed_mean_sample_auprc": 0.2,
                    "distributed_mean_hit_at_10": 1.0,
                    "distributed_mean_first_hit_distance": 0.0,
                    "quiet_mean_delta_max": 0.0,
                    "quiet_mean_delta_p90": 0.0,
                    "quiet_mean_iqr_normalized_delta_max": 0.0,
                    "quiet_mean_top10_inflation": 0.0,
                }
            ]

            with mock.patch.object(finalize, "run_subprocess", side_effect=fake_run_subprocess), mock.patch.object(
                finalize.batch, "build_candidate_summary", return_value=summary_rows
            ), mock.patch.object(
                finalize.batch, "build_standing_report", return_value="fixture report\n"
            ), mock.patch.object(
                finalize.batch, "validate_benchmark_manifest", return_value=None
            ):
                rc = finalize.finalize_candidate_execution(
                    benchmark_dir=benchmark_dir,
                    execution_dir=execution_dir,
                    model_id="Qwen/Qwen2.5-3B-Instruct",
                    model_revision="fixture-rev",
                    device="cuda",
                    topk=128,
                    seed=7,
                )

            self.assertEqual(rc, 0)
            manifest = json.loads((execution_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["rendering_family_id"], "transcript_v1")
            self.assertEqual(manifest["model_id"], "Qwen/Qwen2.5-3B-Instruct")
            self.assertTrue((execution_dir / "candidate_summary.csv").exists())
            self.assertTrue((execution_dir / "gate8a_standing_summary.md").exists())
            checksums = json.loads((execution_dir / "checksums.json").read_text(encoding="utf-8"))
            self.assertIn("manifest.json", checksums)
            self.assertIn("candidate_summary.csv", checksums)


if __name__ == "__main__":
    unittest.main()
