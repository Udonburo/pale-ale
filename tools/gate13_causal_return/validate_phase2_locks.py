"""Validate Gate13 candidate Phase 2 dual locks before any scientific read."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.gate13_causal_return.phase2_common import (
    REPO_ROOT,
    git_status_porcelain,
    read_json,
    require_fields,
    require_sha256,
    sha256_file,
)


A_LOCK = "phase2_a_lock.json"
B_LOCK = "phase2_b2a_lock.json"
DUAL = "phase2_dual_authorization.json"
EXPECTED_CLOSED = {
    "FORMAL_GATE13_OPENING": "CLOSED",
    "A3": "CLOSED",
    "TRACK_C": "CLOSED",
    "ACTIVATION_EXTRACTION": "CLOSED",
    "HIDDEN_STATE_INTERVENTION": "CLOSED",
    "ALIGNMENT_SEARCH": "CLOSED",
    "AMBER_INTEGRATION": "CLOSED",
    "GENERAL_CYCLE_SPACE": "UNRESOLVED",
    "PUBLIC_README_OR_GATE_MAP_EDITS": "FORBIDDEN",
}


class Phase2LockValidationError(ValueError):
    """Raised on any authorization ambiguity or drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Phase2LockValidationError(message)


def _validate_closed(value: Mapping[str, Any], *, context: str) -> None:
    for field, expected in EXPECTED_CLOSED.items():
        _require(value.get(field) == expected, f"{context} closure mismatch: {field}")


def _validate_documents(
    a_lock: Mapping[str, Any],
    b_lock: Mapping[str, Any],
    dual: Mapping[str, Any],
) -> None:
    require_fields(
        a_lock,
        (
            "schema_version",
            "review1_snapshot_commit",
            "phase2_code_commit",
            "handoff_sha256",
            "review1_report_sha256",
            "runtime_binding",
            "case_manifests",
            "metrics",
            "gates",
            "forward_ceiling",
            "closed_surfaces",
        ),
        context="phase2_a_lock",
    )
    require_fields(
        b_lock,
        (
            "schema_version",
            "review1_snapshot_commit",
            "phase2_code_commit",
            "handoff_sha256",
            "review1_report_sha256",
            "scope",
            "legacy_scalar",
            "source_runs",
            "source_sufficiency",
            "closed_surfaces",
        ),
        context="phase2_b2a_lock",
    )
    require_fields(
        dual,
        (
            "schema_version",
            "review1_snapshot_commit",
            "phase2_code_commit",
            "phase2_a_lock_sha256",
            "phase2_b2a_lock_sha256",
            "handoff_sha256",
            "review1_report_sha256",
            "track_a_forward_ceiling",
            "track_b2a_source_runs",
            "closed_surfaces",
            "authorization_timestamp",
            "scientific_outputs_observed_before_lock",
            "phase2_code_paths",
        ),
        context="phase2_dual_authorization",
    )
    for document, name in ((a_lock, "A lock"), (b_lock, "B2a lock"), (dual, "dual")):
        require_sha256(document["handoff_sha256"], field=f"{name}.handoff_sha256")
        require_sha256(
            document["review1_report_sha256"], field=f"{name}.review1_report_sha256"
        )

    for field in ("review1_snapshot_commit", "phase2_code_commit", "handoff_sha256", "review1_report_sha256"):
        values = {str(document[field]) for document in (a_lock, b_lock, dual)}
        _require(len(values) == 1, f"cross-lock mismatch: {field}")

    _require(int(a_lock["forward_ceiling"]) <= 600, "forward ceiling exceeds 600")
    _require(int(dual["track_a_forward_ceiling"]) <= 600, "dual forward ceiling exceeds 600")
    _require(
        int(a_lock["forward_ceiling"]) == int(dual["track_a_forward_ceiling"]),
        "forward ceiling cross-binding mismatch",
    )
    manifests = a_lock["case_manifests"]
    _require(isinstance(manifests, Mapping), "case_manifests must be an object")
    for stage in ("A0_EXTENSION", "A1", "A2"):
        _require(stage in manifests, f"missing {stage} manifest")
        require_fields(manifests[stage], ("path", "sha256", "case_count"), context=stage)
        require_sha256(manifests[stage]["sha256"], field=f"{stage}.sha256")

    scope = b_lock["scope"]
    _require(scope.get("cycle_family") == "explicit_triangle_only_v1", "B2a cycle family mismatch")
    _require(scope.get("general_cycles") == "out_of_scope", "general-cycle flag enabled")
    _require(scope.get("general_cycle_enumeration_enabled") is False, "general-cycle enumeration enabled")
    _require(scope.get("beta_1") == "not_computed", "beta_1 must not be computed")
    _require(scope.get("fundamental_cycles") == "not_constructed", "fundamental cycles enabled")
    _require(scope.get("loop_independence") == "not_defined", "loop independence enabled")

    scalar = b_lock["legacy_scalar"]
    expected_scalar = {
        "artifact": "triangle_holonomy_registry.jsonl",
        "field": "holonomy_residual_fro",
        "method_id": "gate12a_discrete_connection_v1",
        "cycle_mode": "explicit_triangle_only_v1",
        "holonomy_mode": "triangle_equal_rank_orthogonal_fro_residual_v1",
    }
    for field, expected in expected_scalar.items():
        _require(scalar.get(field) == expected, f"legacy scalar mismatch: {field}")

    sufficiency = b_lock["source_sufficiency"]
    require_fields(
        sufficiency,
        (
            "status",
            "retained_underlying_sample_rows",
            "deterministic_split_key",
            "frame_reconstruction_provenance",
            "same_rank_rule",
            "same_local_object_rule",
            "node_wise_procrustes_alignment",
            "minimum_source_sufficient_runs",
        ),
        context="source_sufficiency",
    )
    _require(
        sufficiency["status"] in {"PASS", "SPLIT_HALF_SOURCE_UNAVAILABLE"},
        "invalid split-half source-sufficiency declaration",
    )
    runs = list(b_lock["source_runs"])
    _require(runs, "B2a source run list is empty")
    _require(
        len(runs) == len({str(run.get("run_id") or "") for run in runs}),
        "duplicate or empty B2a run_id",
    )
    required_run_hashes = (
        "manifest_sha256",
        "node_registry_sha256",
        "node_artifact_sha256",
        "edge_artifact_sha256",
        "triangle_registry_sha256",
        "holonomy_registry_sha256",
        "operator_array_sha256",
        "holonomy_array_sha256",
        "source_node_manifest_sha256",
    )
    for run in runs:
        require_fields(
            run,
            (
                "run_id",
                "source_manifest_path",
                "schema_version",
                "method_id",
                "cycle_mode",
                "holonomy_mode",
                "source_node_manifest_path",
                "referenced_gate8_sample_source_path",
                *required_run_hashes,
            ),
            context=f"source run {run.get('run_id')}",
        )
        for field in required_run_hashes:
            require_sha256(run[field], field=f"{run['run_id']}.{field}")
        _require(run["schema_version"] == "gate12a_discrete_connection_v1", "source schema mismatch")
        _require(run["method_id"] == "gate12a_discrete_connection_v1", "source method mismatch")
        _require(run["cycle_mode"] == "explicit_triangle_only_v1", "source cycle mode mismatch")
        _require(
            run["holonomy_mode"] == "triangle_equal_rank_orthogonal_fro_residual_v1",
            "source holonomy mode mismatch",
        )
    _require(
        sorted(str(run["run_id"]) for run in runs)
        == sorted(str(run_id) for run_id in dual["track_b2a_source_runs"]),
        "dual source-run binding mismatch",
    )

    for document, name in ((a_lock, "A lock"), (b_lock, "B2a lock"), (dual, "dual")):
        _validate_closed(document["closed_surfaces"], context=name)
    _require(
        dual["scientific_outputs_observed_before_lock"] is False,
        "scientific output was observed before dual lock",
    )
    _require(dual.get("A3_enabled") is False, "A3 enabled")
    _require(dual.get("track_c_enabled") is False, "Track C enabled")


def _git_commit_exists(commit: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    return result.returncode == 0


def validate_phase2_locks(
    *, phase2_dir: Path, require_clean: bool = True, verify_git: bool = True
) -> dict[str, Any]:
    phase2_dir = Path(phase2_dir)
    a_path = phase2_dir / A_LOCK
    b_path = phase2_dir / B_LOCK
    dual_path = phase2_dir / DUAL
    for path in (a_path, b_path, dual_path):
        if not path.is_file():
            raise Phase2LockValidationError(f"missing lock file: {path}")
    a_lock = read_json(a_path)
    b_lock = read_json(b_path)
    dual = read_json(dual_path)
    _validate_documents(a_lock, b_lock, dual)
    _require(
        sha256_file(a_path) == dual["phase2_a_lock_sha256"],
        "phase2_a_lock SHA mismatch",
    )
    _require(
        sha256_file(b_path) == dual["phase2_b2a_lock_sha256"],
        "phase2_b2a_lock SHA mismatch",
    )
    for stage in ("A0_EXTENSION", "A1", "A2"):
        binding = a_lock["case_manifests"][stage]
        path = phase2_dir / str(binding["path"])
        _require(path.is_file(), f"missing {stage} manifest file")
        _require(sha256_file(path) == binding["sha256"], f"{stage} manifest SHA mismatch")

    if verify_git:
        code_commit = str(dual["phase2_code_commit"])
        snapshot_commit = str(dual["review1_snapshot_commit"])
        _require(_git_commit_exists(code_commit), "phase2_code_commit does not exist")
        _require(_git_commit_exists(snapshot_commit), "review1_snapshot_commit does not exist")
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", snapshot_commit, code_commit],
            cwd=REPO_ROOT,
        )
        _require(ancestor.returncode == 0, "Review 1 snapshot is not an ancestor of Phase 2 code")
        code_paths = [str(path) for path in dual["phase2_code_paths"]]
        _require(bool(code_paths), "phase2_code_paths is empty")
        changed = subprocess.run(
            ["git", "diff", "--quiet", code_commit, "--", *code_paths],
            cwd=REPO_ROOT,
        )
        _require(changed.returncode == 0, "mutable or uncommitted Phase 2 runner/manifests")
        if require_clean:
            _require(git_status_porcelain() == "", "dirty Git state")
    return {
        "schema_version": "gate13_phase2_dual_lock_validation_v1",
        "status": "PASS",
        "phase2_a_lock_sha256": sha256_file(a_path),
        "phase2_b2a_lock_sha256": sha256_file(b_path),
        "phase2_dual_authorization_sha256": sha256_file(dual_path),
        "execution_authorized": bool(dual.get("execution_authorized")),
        "scientific_outputs_observed_before_lock": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = validate_phase2_locks(phase2_dir=args.phase2_dir, require_clean=True)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
