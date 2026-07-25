#!/usr/bin/env python3

from __future__ import annotations

import gzip
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import gate12c2_development_shards as shards  # noqa: E402
import gate12c2_synthetic_lab as lab  # noqa: E402


class Gate12C2DevelopmentShardsTest(unittest.TestCase):
    def build_plan(
        self,
        *,
        indices: tuple[int, ...] = (0, 1),
    ) -> dict[str, object]:
        return shards.build_development_shard_plan(
            regime_id="S0_true_null",
            master_seed="development-shard-test",
            outer_experiment_indices=indices,
            block_count=4,
            inner_valid_draw_count=1,
        )

    def execution_receipts(
        self,
        plan: dict[str, object],
        output_dir: Path,
        worker_count: int,
    ) -> tuple[dict[str, object], dict[str, object]]:
        checks = {
            key: True for key in shards.REQUIRED_PREFLIGHT_CHECKS
        }
        preflight = shards.build_no_outcome_preflight_receipt(
            plan,
            output_dir=output_dir,
            worker_count=worker_count,
            preflight_id="unit-test-preflight",
            checks=checks,
        )
        authorization = (
            shards.build_development_execution_authorization(
                plan,
                preflight,
                output_dir=output_dir,
                worker_count=worker_count,
                authorization_id="unit-test-authorization",
                purpose="development-unit-test",
            )
        )
        return preflight, authorization

    def execute(
        self,
        plan: dict[str, object],
        *,
        output_dir: Path,
        worker_count: int,
    ) -> dict[str, object]:
        preflight, authorization = self.execution_receipts(
            plan,
            output_dir,
            worker_count,
        )
        return shards.execute_development_shard_plan(
            plan,
            output_dir=output_dir,
            worker_count=worker_count,
            preflight_receipt=preflight,
            authorization_receipt=authorization,
        )

    def test_plan_is_sorted_hashed_and_development_only(self) -> None:
        first = self.build_plan(indices=(3, 1, 2))
        second = self.build_plan(indices=(2, 3, 1))
        self.assertEqual(first, second)
        self.assertEqual(first["outer_experiment_indices"], [1, 2, 3])
        self.assertEqual(first["surface_id"], "development")
        self.assertFalse(first["locked_execution_authorized"])
        self.assertFalse(first["real_held_out_execution_authorized"])
        self.assertFalse(first["N2_open"])
        self.assertFalse(first["N3_open"])
        self.assertNotIn("worker_count", first)
        self.assertEqual(
            first["accepted_valid_draw_storage"],
            lab.COMPACT_ACCEPTED_PREFIX_STORAGE_ID,
        )
        self.assertEqual(
            first["numerical_environment"]["blas_thread_limit"],
            1,
        )
        self.assertEqual(
            first["numerical_environment"]["thread_environment"],
            shards.SINGLE_THREAD_ENVIRONMENT,
        )

    def test_execution_requires_exact_external_authorization(self) -> None:
        plan = self.build_plan(indices=(0,))
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "sharded"
            with self.assertRaises(shards.Gate12C2ShardError):
                shards.execute_development_shard_plan(
                    plan,
                    output_dir=output_dir,
                    worker_count=1,
                )
            preflight, authorization = self.execution_receipts(
                plan,
                output_dir,
                1,
            )
            changed = dict(authorization)
            changed["worker_count"] = 2
            changed.pop("authorization_receipt_payload_sha256")
            changed["authorization_receipt_payload_sha256"] = (
                shards._sha256_bytes(
                    shards._canonical_json_bytes(changed)
                )
            )
            with self.assertRaises(shards.Gate12C2ShardError):
                shards.execute_development_shard_plan(
                    plan,
                    output_dir=output_dir,
                    worker_count=1,
                    preflight_receipt=preflight,
                    authorization_receipt=changed,
                )

    def test_sequential_execution_resumes_without_scientific_change(
        self,
    ) -> None:
        plan = self.build_plan()
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "sharded"
            first = self.execute(
                plan,
                output_dir=output_dir,
                worker_count=1,
            )
            self.assertEqual(first["outer_experiment_count"], 2)
            self.assertTrue(first["all_outer_indices_present"])
            self.assertFalse(first["locked_execution_authorized"])
            self.assertEqual(
                [row["outer_experiment_index"] for row in first["shards"]],
                [0, 1],
            )
            self.assertTrue(
                all(
                    not row["reused_existing_shard"]
                    for row in first["shards"]
                )
            )
            file_hashes = [
                row["compressed_file_sha256"] for row in first["shards"]
            ]
            second = self.execute(
                plan,
                output_dir=output_dir,
                worker_count=1,
            )
            self.assertTrue(
                all(
                    row["reused_existing_shard"]
                    for row in second["shards"]
                )
            )
            self.assertEqual(
                second["scientific_projection_sha256"],
                first["scientific_projection_sha256"],
            )
            self.assertEqual(
                [row["compressed_file_sha256"] for row in second["shards"]],
                file_hashes,
            )
            for row in second["shards"]:
                path = output_dir / row["relative_path"]
                payload = json.loads(
                    gzip.decompress(path.read_bytes()).decode("utf-8")
                )
                self.assertEqual(payload["surface_id"], "development")
                self.assertFalse(payload["locked_execution_authorized"])
                self.assertEqual(
                    payload["result"]["numerical_execution_contract"][
                        "blas_thread_limit"
                    ],
                    1,
                )
            verification = shards.verify_development_shard_index(
                plan,
                output_dir=output_dir,
            )
            self.assertEqual(verification["status"], "pass")
            self.assertEqual(
                verification["scientific_projection_sha256"],
                first["scientific_projection_sha256"],
            )

    def test_existing_plan_or_shard_mismatch_is_rejected(self) -> None:
        plan = self.build_plan(indices=(0,))
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "sharded"
            result = self.execute(
                plan,
                output_dir=output_dir,
                worker_count=1,
            )
            changed_plan = shards.build_development_shard_plan(
                regime_id="S0_true_null",
                master_seed="different-seed",
                outer_experiment_indices=(0,),
                block_count=4,
                inner_valid_draw_count=1,
            )
            with self.assertRaises(shards.Gate12C2ShardError):
                self.execute(
                    changed_plan,
                    output_dir=output_dir,
                    worker_count=1,
                )
            shard_path = output_dir / result["shards"][0]["relative_path"]
            corrupted = bytearray(shard_path.read_bytes())
            corrupted[-1] ^= 1
            shard_path.write_bytes(corrupted)
            with self.assertRaises(shards.Gate12C2ShardError):
                self.execute(
                    plan,
                    output_dir=output_dir,
                    worker_count=1,
                )

    def test_merge_order_and_worker_count_do_not_change_science(self) -> None:
        # Simulate a long-lived parent that loaded an additional BLAS consumer.
        import scipy.linalg  # noqa: F401

        plan = self.build_plan(indices=(0, 1))
        with tempfile.TemporaryDirectory() as first_temporary:
            first_dir = Path(first_temporary) / "workers-1"
            first = self.execute(
                plan,
                output_dir=first_dir,
                worker_count=1,
            )
            paths = [
                first_dir / row["relative_path"]
                for row in first["shards"]
            ]
            forward = shards.verify_development_shard_set(
                plan,
                output_dir=first_dir,
                candidate_paths=paths,
            )
            reverse = shards.verify_development_shard_set(
                plan,
                output_dir=first_dir,
                candidate_paths=list(reversed(paths)),
            )
            self.assertEqual(
                forward["scientific_projection_sha256"],
                reverse["scientific_projection_sha256"],
            )
            with self.assertRaises(shards.Gate12C2ShardError):
                shards.verify_development_shard_set(
                    plan,
                    output_dir=first_dir,
                    candidate_paths=(paths[0], paths[0]),
                )

            with tempfile.TemporaryDirectory() as second_temporary:
                second_dir = Path(second_temporary) / "workers-2"
                second = self.execute(
                    plan,
                    output_dir=second_dir,
                    worker_count=2,
                )
                self.assertEqual(
                    first["scientific_projection_sha256"],
                    second["scientific_projection_sha256"],
                )
                self.assertEqual(
                    {
                        row["outer_experiment_index"]: row[
                            "result_payload_sha256"
                        ]
                        for row in first["shards"]
                    },
                    {
                        row["outer_experiment_index"]: row[
                            "result_payload_sha256"
                        ]
                        for row in second["shards"]
                    },
                )

    def test_missing_unexpected_and_partial_artifacts_fail_closed(self) -> None:
        plan = self.build_plan(indices=(0, 1))
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "sharded"
            result = self.execute(
                plan,
                output_dir=output_dir,
                worker_count=1,
            )
            first_path = output_dir / result["shards"][0]["relative_path"]
            first_bytes = first_path.read_bytes()
            first_path.unlink()
            with self.assertRaises(shards.Gate12C2ShardError):
                shards.verify_development_shard_set(
                    plan,
                    output_dir=output_dir,
                )
            first_path.write_bytes(first_bytes)

            unexpected = (
                output_dir / "shards" / "outer-999999.json.gz"
            )
            shutil.copyfile(first_path, unexpected)
            with self.assertRaises(shards.Gate12C2ShardError):
                shards.verify_development_shard_set(
                    plan,
                    output_dir=output_dir,
                )
            unexpected.unlink()

            partial = (
                output_dir
                / "shards"
                / ".outer-000000.json.gz.999999.tmp"
            )
            partial.write_bytes(b"partial")
            with self.assertRaises(shards.Gate12C2ShardError):
                self.execute(
                    plan,
                    output_dir=output_dir,
                    worker_count=1,
                )
            partial.unlink()
            resumed = self.execute(
                plan,
                output_dir=output_dir,
                worker_count=1,
            )
            self.assertEqual(
                resumed["scientific_projection_sha256"],
                result["scientific_projection_sha256"],
            )

    def test_mixed_plan_shard_and_corrupt_index_fail_closed(self) -> None:
        plan = self.build_plan(indices=(0,))
        other = shards.build_development_shard_plan(
            regime_id="S0_true_null",
            master_seed="other-development-shard-test",
            outer_experiment_indices=(0,),
            block_count=4,
            inner_valid_draw_count=1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "primary"
            other_dir = root / "other"
            result = self.execute(
                plan,
                output_dir=output_dir,
                worker_count=1,
            )
            other_result = self.execute(
                other,
                output_dir=other_dir,
                worker_count=1,
            )
            primary_path = (
                output_dir / result["shards"][0]["relative_path"]
            )
            primary_bytes = primary_path.read_bytes()
            other_path = (
                other_dir / other_result["shards"][0]["relative_path"]
            )
            shutil.copyfile(other_path, primary_path)
            with self.assertRaises(shards.Gate12C2ShardError):
                shards.verify_development_shard_set(
                    plan,
                    output_dir=output_dir,
                )
            primary_path.write_bytes(primary_bytes)

            index_path = output_dir / "index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            index["scientific_projection_sha256"] = "0" * 64
            index_path.write_text(
                json.dumps(index, sort_keys=True),
                encoding="utf-8",
            )
            with self.assertRaises(shards.Gate12C2ShardError):
                self.execute(
                    plan,
                    output_dir=output_dir,
                    worker_count=1,
                )

    def test_tampered_implementation_contract_is_rejected(self) -> None:
        plan = self.build_plan(indices=(0,))
        tampered = dict(plan)
        tampered["implementation_sha256"] = dict(
            tampered["implementation_sha256"]
        )
        tampered["implementation_sha256"][
            "gate12c2_synthetic_lab.py"
        ] = "0" * 64
        tampered.pop("plan_payload_sha256")
        tampered["plan_payload_sha256"] = shards._sha256_bytes(
            shards._canonical_json_bytes(tampered)
        )
        with self.assertRaises(shards.Gate12C2ShardError):
            shards._verified_plan(tampered)

    def test_rehashed_open_or_extended_plan_is_rejected(self) -> None:
        plan = self.build_plan(indices=(0,))
        for key, value in (
            ("locked_execution_authorized", True),
            ("real_held_out_execution_authorized", True),
            ("N2_open", True),
            ("N3_open", True),
            ("public_claim", True),
            ("unexpected_permission", False),
        ):
            tampered = dict(plan)
            tampered[key] = value
            tampered.pop("plan_payload_sha256")
            tampered["plan_payload_sha256"] = shards._sha256_bytes(
                shards._canonical_json_bytes(tampered)
            )
            with self.subTest(key=key):
                with self.assertRaises(shards.Gate12C2ShardError):
                    shards._verified_plan(tampered)

    def test_self_consistent_result_configuration_mismatch_is_rejected(
        self,
    ) -> None:
        plan = self.build_plan(indices=(0,))
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "sharded"
            result = self.execute(
                plan,
                output_dir=output_dir,
                worker_count=1,
            )
            shard_path = output_dir / result["shards"][0]["relative_path"]
            payload = json.loads(
                gzip.decompress(shard_path.read_bytes()).decode("utf-8")
            )
            payload["result"]["inner_valid_draw_count"] = 2
            payload["result"]["execution_configuration_contract"][
                "inner_valid_draw_count"
            ] = 2
            payload["result_payload_sha256"] = shards._sha256_bytes(
                shards._canonical_json_bytes(payload["result"])
            )
            payload.pop("shard_payload_sha256")
            payload["shard_payload_sha256"] = shards._sha256_bytes(
                shards._canonical_json_bytes(payload)
            )
            shard_path.write_bytes(
                gzip.compress(
                    shards._canonical_json_bytes(payload),
                    compresslevel=6,
                    mtime=0,
                )
            )
            with self.assertRaises(shards.Gate12C2ShardError):
                shards.verify_development_shard_set(
                    plan,
                    output_dir=output_dir,
                )

    def test_worker_change_resume_reuses_identical_science(self) -> None:
        plan = self.build_plan(indices=(0, 1))
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "sharded"
            first = self.execute(
                plan,
                output_dir=output_dir,
                worker_count=1,
            )
            second = self.execute(
                plan,
                output_dir=output_dir,
                worker_count=2,
            )
            self.assertEqual(
                first["scientific_projection_sha256"],
                second["scientific_projection_sha256"],
            )
            self.assertTrue(
                all(row["reused_existing_shard"] for row in second["shards"])
            )

    def test_merge_interruption_resumes_from_complete_shards(self) -> None:
        plan = self.build_plan(indices=(0, 1))
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "sharded"
            original_atomic_write = shards._atomic_write

            def fail_index(path: Path, payload: bytes) -> None:
                if path.name == "index.json":
                    raise OSError("injected merge interruption")
                original_atomic_write(path, payload)

            preflight, authorization = self.execution_receipts(
                plan,
                output_dir,
                1,
            )
            with mock.patch.object(
                shards,
                "_atomic_write",
                side_effect=fail_index,
            ):
                with self.assertRaises(OSError):
                    shards.execute_development_shard_plan(
                        plan,
                        output_dir=output_dir,
                        worker_count=1,
                        preflight_receipt=preflight,
                        authorization_receipt=authorization,
                    )
            self.assertEqual(
                len(list((output_dir / "shards").glob("*.json.gz"))),
                2,
            )
            resumed = self.execute(
                plan,
                output_dir=output_dir,
                worker_count=1,
            )
            self.assertTrue(
                all(
                    row["reused_existing_shard"]
                    for row in resumed["shards"]
                )
            )

    def test_disk_write_failure_leaves_no_committed_partial(self) -> None:
        plan = self.build_plan(indices=(0,))
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "sharded"
            preflight, authorization = self.execution_receipts(
                plan,
                output_dir,
                1,
            )
            with mock.patch.object(
                shards,
                "_atomic_write",
                side_effect=OSError("injected disk failure"),
            ):
                with self.assertRaises(OSError):
                    shards.execute_development_shard_plan(
                        plan,
                        output_dir=output_dir,
                        worker_count=1,
                        preflight_receipt=preflight,
                        authorization_receipt=authorization,
                    )
            self.assertFalse((output_dir / "plan.json").exists())

    def test_canonical_json_rejects_nonfinite_values(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaises(shards.Gate12C2ShardError):
                    shards._canonical_json_bytes({"value": value})
                with self.assertRaises(lab.Gate12C2DevelopmentError):
                    lab._canonical_json_bytes({"value": value})

    def test_plan_rejects_locked_aliases_and_invalid_regimes(self) -> None:
        with self.assertRaises(shards.Gate12C2ShardError):
            shards.build_development_shard_plan(
                regime_id="locked_synthetic",
                master_seed="forbidden",
                outer_experiment_indices=(0,),
                block_count=4,
                inner_valid_draw_count=1,
            )
        with self.assertRaises(shards.Gate12C2ShardError):
            shards.build_development_shard_plan(
                regime_id="S1_known_reverse_shared_node_coupling",
                master_seed="missing-effect",
                outer_experiment_indices=(0,),
                block_count=4,
                inner_valid_draw_count=1,
            )


if __name__ == "__main__":
    unittest.main()
