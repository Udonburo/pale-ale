from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from tools.gate13_causal_return.phase2_common import sha256_file
from tools.gate13_causal_return.validate_phase2_locks import (
    EXPECTED_CLOSED,
    Phase2LockValidationError,
    _validate_documents,
    validate_phase2_locks,
)


H = "a" * 64
C = "b" * 40


def valid_documents() -> tuple[dict, dict, dict]:
    a = {
        "schema_version": "a",
        "review1_snapshot_commit": C,
        "phase2_code_commit": C,
        "handoff_sha256": H,
        "review1_report_sha256": H,
        "runtime_binding": {"model_repository": "Qwen/Qwen3-8B"},
        "case_manifests": {
            "A1": {"path": "a1.json", "sha256": H, "case_count": 1},
            "A2": {"path": "a2.json", "sha256": H, "case_count": 1},
        },
        "metrics": {},
        "gates": {},
        "forward_ceiling": 600,
        "closed_surfaces": dict(EXPECTED_CLOSED),
    }
    run = {
        "run_id": "r",
        "source_manifest_path": "x/manifest.json",
        "schema_version": "gate12a_discrete_connection_v1",
        "method_id": "gate12a_discrete_connection_v1",
        "cycle_mode": "explicit_triangle_only_v1",
        "holonomy_mode": "triangle_equal_rank_orthogonal_fro_residual_v1",
        "source_node_manifest_path": "x/node-manifest.json",
        "referenced_gate8_sample_source_path": "x/gate8",
        "manifest_sha256": H,
        "node_registry_sha256": H,
        "node_artifact_sha256": H,
        "edge_artifact_sha256": H,
        "triangle_registry_sha256": H,
        "holonomy_registry_sha256": H,
        "operator_array_sha256": H,
        "holonomy_array_sha256": H,
        "source_node_manifest_sha256": H,
    }
    b = {
        "schema_version": "b",
        "review1_snapshot_commit": C,
        "phase2_code_commit": C,
        "handoff_sha256": H,
        "review1_report_sha256": H,
        "scope": {
            "cycle_family": "explicit_triangle_only_v1",
            "general_cycles": "out_of_scope",
            "general_cycle_enumeration_enabled": False,
            "beta_1": "not_computed",
            "fundamental_cycles": "not_constructed",
            "loop_independence": "not_defined",
        },
        "legacy_scalar": {
            "artifact": "triangle_holonomy_registry.jsonl",
            "field": "holonomy_residual_fro",
            "method_id": "gate12a_discrete_connection_v1",
            "cycle_mode": "explicit_triangle_only_v1",
            "holonomy_mode": "triangle_equal_rank_orthogonal_fro_residual_v1",
        },
        "source_runs": [run],
        "source_sufficiency": {
            "status": "SPLIT_HALF_SOURCE_UNAVAILABLE",
            "retained_underlying_sample_rows": False,
            "deterministic_split_key": "fixed",
            "frame_reconstruction_provenance": "fixed",
            "same_rank_rule": "fixed",
            "same_local_object_rule": "fixed",
            "node_wise_procrustes_alignment": "fixed",
            "minimum_source_sufficient_runs": 1,
        },
        "closed_surfaces": dict(EXPECTED_CLOSED),
    }
    d = {
        "schema_version": "d",
        "review1_snapshot_commit": C,
        "phase2_code_commit": C,
        "phase2_a_lock_sha256": H,
        "phase2_b2a_lock_sha256": H,
        "handoff_sha256": H,
        "review1_report_sha256": H,
        "track_a_forward_ceiling": 600,
        "track_b2a_source_runs": ["r"],
        "closed_surfaces": dict(EXPECTED_CLOSED),
        "authorization_timestamp": "2026-08-19T00:00:00+09:00",
        "scientific_outputs_observed_before_lock": False,
        "phase2_code_paths": ["tools/gate13_causal_return"],
        "A3_enabled": False,
        "track_c_enabled": False,
    }
    return a, b, d


class LockDocumentTests(unittest.TestCase):
    def test_valid_documents_pass(self) -> None:
        _validate_documents(*valid_documents())

    def test_required_rejections(self) -> None:
        mutations = []
        a, b, d = valid_documents()
        a["forward_ceiling"] = 601
        mutations.append((a, b, d))
        a, b, d = valid_documents()
        del a["case_manifests"]["A2"]
        mutations.append((a, b, d))
        a, b, d = valid_documents()
        del b["source_runs"][0]["operator_array_sha256"]
        mutations.append((a, b, d))
        a, b, d = valid_documents()
        b["scope"]["general_cycle_enumeration_enabled"] = True
        mutations.append((a, b, d))
        a, b, d = valid_documents()
        b["legacy_scalar"]["field"] = "wrong"
        mutations.append((a, b, d))
        a, b, d = valid_documents()
        del b["source_sufficiency"]["deterministic_split_key"]
        mutations.append((a, b, d))
        a, b, d = valid_documents()
        d["track_c_enabled"] = True
        mutations.append((a, b, d))
        for documents in mutations:
            with self.assertRaises((ValueError, Phase2LockValidationError)):
                _validate_documents(*documents)

    def test_file_sha_mismatch_is_rejected_without_git_probe(self) -> None:
        a, b, d = valid_documents()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "a1.json").write_text("{}\n", encoding="utf-8")
            (root / "a2.json").write_text("{}\n", encoding="utf-8")
            a["case_manifests"]["A1"]["sha256"] = sha256_file(root / "a1.json")
            a["case_manifests"]["A2"]["sha256"] = sha256_file(root / "a2.json")
            (root / "phase2_a_lock.json").write_text(json.dumps(a), encoding="utf-8")
            (root / "phase2_b2a_lock.json").write_text(json.dumps(b), encoding="utf-8")
            d["phase2_a_lock_sha256"] = "0" * 64
            d["phase2_b2a_lock_sha256"] = sha256_file(root / "phase2_b2a_lock.json")
            (root / "phase2_dual_authorization.json").write_text(json.dumps(d), encoding="utf-8")
            with self.assertRaisesRegex(Phase2LockValidationError, "phase2_a_lock SHA mismatch"):
                validate_phase2_locks(phase2_dir=root, require_clean=False, verify_git=False)


if __name__ == "__main__":
    unittest.main()
