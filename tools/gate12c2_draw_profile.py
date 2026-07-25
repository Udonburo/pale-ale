#!/usr/bin/env python3
"""Exact, authorization-gated coordinator for the Gate12C-2 draw profile.

The coordinator admits one frozen development layout only:

* S0: 128 complete outer experiments;
* S1: 64 complete outer experiments at the representative 0.25 effect;
* S2: 64 complete outer experiments;
* accepted-valid draw counts 255, 511, and 1023; and
* four single-threaded BLAS workers.

It does not expose scientific outcomes.  The separate draw-stability analyzer
is the only admitted human-facing projection over completed result shards.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import gate12c2_development_shards as shards
import gate12c2_synthetic_lab as lab
import gate12c2_throughput_profile as throughput


PLAN_SCHEMA_VERSION = "gate12c2_draw_profile_plan_v0.1"
PREFLIGHT_SCHEMA_VERSION = "gate12c2_draw_profile_preflight_v0.2"
AUTHORIZATION_SCHEMA_VERSION = (
    "gate12c2_draw_profile_execution_authorization_v0.2"
)
RECEIPT_SCHEMA_VERSION = "gate12c2_draw_profile_execution_receipt_v0.2"
EXECUTION_EVIDENCE_SCHEMA_VERSION = (
    "gate12c2_draw_profile_execution_evidence_v0.1"
)
RESOURCE_RECEIPT_SCHEMA_VERSION = (
    "gate12c2_draw_profile_resource_receipt_v0.1"
)
RESOURCE_POLICY_SCHEMA_VERSION = (
    "gate12c2_draw_profile_resource_policy_v0.1"
)
MECHANICAL_PREFLIGHT_SCHEMA_VERSION = (
    "gate12c2_draw_profile_mechanical_preflight_v0.1"
)
RESTORE_REHEARSAL_SCHEMA_VERSION = (
    "gate12c2_draw_profile_restore_rehearsal_v0.1"
)
WORKER_CARRY_FORWARD_SCHEMA_VERSION = (
    "gate12c2_worker_profile_carry_forward_v0.1"
)
PLAN_ID = "gate12c2-development-accepted-valid-draw-scaling-v0.2"
WORKER_COUNT = 4
PREFIX_COUNTS = (255, 511, 1023)
REFERENCE_DRAW_COUNT = 1023
COORDINATOR_LOCK_NAME = ".draw-profile.lock.json"
AUTHORIZATION_CONSUMED_NAME = "authorization-consumed.json"
EXECUTION_EVIDENCE_NAME = "execution-evidence.json"
RESOURCE_RECEIPT_NAME = "resource-receipt.json"
EXECUTION_RECEIPT_NAME = "execution-receipt.json"
RESOURCE_DISK_SAFETY_FACTOR = 1.3
RESOURCE_MINIMUM_REMAINING_FRACTION_OF_PRERUN_FREE_DISK = 0.5
RESOURCE_MAXIMUM_RSS_FRACTION_OF_PHYSICAL_RAM = 0.75
S2_AMENDMENT_PAYLOAD_SHA256 = (
    "c9ceff435b6e653c02362dc3e50abf0bfc64f1fd80af38819ef412638cbecb8c"
)
REGIME_SPECIFICATIONS = (
    {
        "regime_id": "S0_true_null",
        "master_seed": (
            "gate12c2-development-draw-scaling-v0.1::S0_true_null"
        ),
        "outer_count": 128,
        "effect_strength": None,
    },
    {
        "regime_id": "S1_known_reverse_shared_node_coupling",
        "master_seed": (
            "gate12c2-development-draw-scaling-v0.1::"
            "S1_known_reverse_shared_node_coupling"
        ),
        "outer_count": 64,
        "effect_strength": 0.25,
    },
    {
        "regime_id": "S2_null_inflation",
        "master_seed": (
            "gate12c2-development-draw-scaling-v0.1::S2_null_inflation"
        ),
        "outer_count": 64,
        "effect_strength": None,
    },
)
REQUIRED_PREFLIGHT_CHECKS = (
    "complete_plan_rebuilt",
    "implementation_hashes_verified",
    "numerical_environment_verified",
    "all_nine_subplans_verified",
    "outer_id_surfaces_verified",
    "accepted_prefix_namespaces_verified",
    "S2_amendment_verified",
    "strict_no_outcome_analyzer_verified",
    "output_root_verified",
    "disk_gate_verified",
    "standalone_recovery_bundle_verified",
    "short_path_restore_rehearsal_verified",
    "worker_profile_carry_forward_verified",
    "profile_root_transaction_boundary_verified",
    "no_active_competing_execution_verified",
    "no_scientific_outcomes_inspected",
    "locked_and_held_out_boundaries_verified",
)


class Gate12C2DrawProfileError(ValueError):
    """Raised when the exact draw profile crosses a frozen boundary."""


def _canonical_json_bytes(payload: object) -> bytes:
    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise Gate12C2DrawProfileError(
            f"canonical JSON requires finite, serializable values: {exc}"
        ) from exc
    return encoded.encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_hashes() -> dict[str, str]:
    paths = {
        "gate12c2_draw_profile.py": Path(__file__).resolve(),
        "gate12c2_development_shards.py": Path(shards.__file__).resolve(),
        "gate12c2_draw_stability.py": Path(__file__)
        .with_name("gate12c2_draw_stability.py")
        .resolve(),
        "gate12c2_synthetic_lab.py": Path(lab.__file__).resolve(),
        "run_gate12c2_draw_profile.py": Path(__file__)
        .with_name("run_gate12c2_draw_profile.py")
        .resolve(),
        "issue_gate12c2_draw_profile_preflight.py": Path(__file__)
        .with_name("issue_gate12c2_draw_profile_preflight.py")
        .resolve(),
        "issue_gate12c2_draw_profile_authorization.py": Path(__file__)
        .with_name("issue_gate12c2_draw_profile_authorization.py")
        .resolve(),
        "recover_gate12c2_draw_profile.py": Path(__file__)
        .with_name("recover_gate12c2_draw_profile.py")
        .resolve(),
        "gate12c2_worker_carry_forward.py": Path(__file__)
        .with_name("gate12c2_worker_carry_forward.py")
        .resolve(),
    }
    return {
        name: _sha256_file(path)
        for name, path in sorted(paths.items())
    }


def _require_exact_keys(
    payload: Mapping[str, Any],
    expected: set[str],
    *,
    context: str,
) -> None:
    actual = set(payload)
    if actual != expected:
        raise Gate12C2DrawProfileError(
            f"{context} keys differ from the frozen schema: "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _output_root(path: Path) -> str:
    return Path(path).resolve().as_posix()


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _resource_policy() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": RESOURCE_POLICY_SCHEMA_VERSION,
        "worker_count": WORKER_COUNT,
        "draw_counts": list(PREFIX_COUNTS),
        "regimes": [
            str(row["regime_id"]) for row in REGIME_SPECIFICATIONS
        ],
        "maximum_process_tree_RSS_fraction_of_physical_RAM": (
            RESOURCE_MAXIMUM_RSS_FRACTION_OF_PHYSICAL_RAM
        ),
        "disk_projection_safety_factor": RESOURCE_DISK_SAFETY_FACTOR,
        "minimum_remaining_fraction_of_prerun_free_disk": (
            RESOURCE_MINIMUM_REMAINING_FRACTION_OF_PRERUN_FREE_DISK
        ),
        "require_memory_monitor_success": True,
        "require_zero_unaccounted_rejections": True,
        "require_zero_exhausted_incomplete_streams": True,
        "require_complete_outer_ID_surface": True,
        "eligibility_derivation": (
            "a draw count is eligible only when the global memory and disk "
            "gates pass and every regime configuration at that draw count "
            "is complete, exact-plan-bound, and has zero unaccounted "
            "rejections and zero exhausted incomplete streams"
        ),
    }
    payload["resource_policy_payload_sha256"] = _sha256_bytes(
        _canonical_json_bytes(payload)
    )
    return payload


def _plan_numerical_environment_sha256(
    configurations: list[Mapping[str, Any]],
) -> str:
    values = {
        _sha256_bytes(
            _canonical_json_bytes(
                configuration["subplan"]["numerical_environment"]
            )
        )
        for configuration in configurations
    }
    if len(values) != 1:
        raise Gate12C2DrawProfileError(
            "draw-profile subplans do not share one numerical environment"
        )
    return values.pop()


def build_draw_profile_plan(*, source_commit: str) -> dict[str, Any]:
    """Build the one admitted post-hardening draw-profile plan."""

    if not str(source_commit).strip():
        raise Gate12C2DrawProfileError("source_commit must be nonempty")
    configurations = []
    for regime in REGIME_SPECIFICATIONS:
        for draw_count in PREFIX_COUNTS:
            subplan = shards.build_development_shard_plan(
                regime_id=str(regime["regime_id"]),
                master_seed=str(regime["master_seed"]),
                outer_experiment_indices=range(
                    int(regime["outer_count"])
                ),
                block_count=lab.reference_block_count_schedule(),
                inner_valid_draw_count=draw_count,
                effect_strength=regime["effect_strength"],
            )
            configurations.append(
                {
                    "configuration_id": (
                        f"{regime['regime_id']}__d{draw_count}"
                    ),
                    "regime_id": str(regime["regime_id"]),
                    "draw_count": draw_count,
                    "worker_count": WORKER_COUNT,
                    "output_relative_path": (
                        f"runs/{regime['regime_id']}/draw-{draw_count}"
                    ),
                    "subplan": subplan,
                }
            )
    payload: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "plan_id": PLAN_ID,
        "epistemic_status": "development_draw_profile_plan_only",
        "surface_id": "development",
        "development_execution_requires_external_authorization": True,
        "locked_execution_authorized": False,
        "real_held_out_execution_authorized": False,
        "N2_open": False,
        "N3_open": False,
        "public_claim": False,
        "scientific_calibration_interpretation_authorized": False,
        "source_commit": str(source_commit),
        "worker_count": WORKER_COUNT,
        "prefix_counts": list(PREFIX_COUNTS),
        "reference_draw_count": REFERENCE_DRAW_COUNT,
        "S2_amendment_payload_sha256": S2_AMENDMENT_PAYLOAD_SHA256,
        "standalone_off_device_recovery_required": True,
        "implementation_sha256": _implementation_hashes(),
        "numerical_environment_sha256": (
            _plan_numerical_environment_sha256(configurations)
        ),
        "resource_policy": _resource_policy(),
        "thread_environment": dict(
            sorted(shards.SINGLE_THREAD_ENVIRONMENT.items())
        ),
        "configurations": configurations,
        "selection_boundary": {
            "allowed": [
                "accepted_prefix_identity",
                "endpoint_decision_agreement",
                "absolute_primary_summary_shift",
                "absolute_S2_component_shift",
                "absolute_S0_family_wise_shift",
                "runtime",
                "memory",
                "disk",
                "rejection_burden",
                "resume_integrity",
            ],
            "prohibited": [
                "best_observed_FPR",
                "best_observed_power",
                "most_favorable_direction",
                "raw_claim_promotion_rate",
                "raw_S2_identification_rate",
            ],
        },
        "scientific_result": None,
    }
    payload["draw_profile_plan_payload_sha256"] = _sha256_bytes(
        _canonical_json_bytes(payload)
    )
    return payload


def verify_draw_profile_plan(
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild and exact-compare the complete frozen coordinator plan."""

    if not isinstance(plan, Mapping):
        raise Gate12C2DrawProfileError("draw profile plan must be a mapping")
    supplied = dict(plan)
    claimed = supplied.get("draw_profile_plan_payload_sha256")
    unhashed = dict(supplied)
    unhashed.pop("draw_profile_plan_payload_sha256", None)
    if claimed != _sha256_bytes(_canonical_json_bytes(unhashed)):
        raise Gate12C2DrawProfileError("draw profile plan hash mismatch")
    try:
        expected = build_draw_profile_plan(
            source_commit=str(unhashed["source_commit"])
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, Gate12C2DrawProfileError):
            raise
        raise Gate12C2DrawProfileError(
            f"draw profile plan cannot be reconstructed: {exc}"
        ) from exc
    if supplied != expected:
        raise Gate12C2DrawProfileError(
            "draw profile plan differs from the complete builder contract"
        )
    for configuration in expected["configurations"]:
        shards._verified_plan(configuration["subplan"])
    return expected


def _read_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    resolved = Path(path).resolve()
    try:
        raw = resolved.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise Gate12C2DrawProfileError(
            f"could not read {label} {resolved}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise Gate12C2DrawProfileError(
            f"{label} must contain one JSON object"
        )
    return payload


def _verify_self_hash(
    payload: Mapping[str, Any],
    *,
    hash_field: str,
    label: str,
) -> str:
    supplied = dict(payload)
    claimed = supplied.pop(hash_field, None)
    if not _is_sha256(claimed):
        raise Gate12C2DrawProfileError(
            f"{label} has an invalid {hash_field}"
        )
    actual = _sha256_bytes(_canonical_json_bytes(supplied))
    if claimed != actual:
        raise Gate12C2DrawProfileError(
            f"{label} {hash_field} does not match its payload"
        )
    return str(claimed)


def _partial_artifacts(root: Path) -> list[str]:
    if not root.exists():
        return []
    matches = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        lowered = path.name.lower()
        if (
            lowered.endswith(".tmp")
            or ".tmp." in lowered
            or lowered.endswith(".partial")
            or lowered.endswith(".incomplete")
        ):
            matches.append(path.relative_to(root).as_posix())
    return sorted(matches)


def _verify_fresh_output_root(output_root: Path) -> dict[str, Any]:
    root = Path(output_root).resolve()
    parent = root.parent
    if not parent.is_dir():
        raise Gate12C2DrawProfileError(
            "draw-profile output parent must already exist"
        )
    partials = _partial_artifacts(root)
    if partials:
        raise Gate12C2DrawProfileError(
            f"draw-profile output root contains partial artifacts: {partials}"
        )
    if root.exists() and any(root.iterdir()):
        raise Gate12C2DrawProfileError(
            "mechanical preflight requires a nonexistent or empty output root"
        )
    if (root / COORDINATOR_LOCK_NAME).exists():
        raise Gate12C2DrawProfileError(
            "draw-profile output root has an active or orphan lock"
        )
    probe = parent / (
        f".gate12c2-preflight-write-probe-{os.getpid()}-"
        f"{time.time_ns()}.tmp"
    )
    try:
        with probe.open("xb") as handle:
            handle.write(b"gate12c2-preflight-probe")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise Gate12C2DrawProfileError(
            f"draw-profile output parent is not atomically writable: {exc}"
        ) from exc
    finally:
        if probe.exists():
            probe.unlink()
    return {
        "output_root": root.as_posix(),
        "output_root_exists": root.exists(),
        "output_root_empty": not root.exists() or not any(root.iterdir()),
        "partial_artifact_count": 0,
        "coordinator_lock_present": False,
        "atomic_write_probe": "pass",
    }


def _verify_prior_worker_profile(
    receipt_path: Path,
) -> dict[str, Any]:
    path = Path(receipt_path).resolve()
    payload = _read_json_mapping(path, label="worker profile receipt")
    payload_hash = _verify_self_hash(
        payload,
        hash_field="profile_receipt_payload_sha256",
        label="worker profile receipt",
    )
    rows = payload.get("configuration_results")
    if not isinstance(rows, list):
        raise Gate12C2DrawProfileError(
            "worker profile receipt lacks configuration results"
        )
    admitted = {}
    for regime in REGIME_SPECIFICATIONS:
        regime_id = str(regime["regime_id"])
        matching = [
            row
            for row in rows
            if isinstance(row, Mapping)
            and row.get("regime_id") == regime_id
            and int(row.get("worker_count", -1)) == WORKER_COUNT
            and int(row.get("inner_valid_draw_count", -1))
            == PREFIX_COUNTS[0]
        ]
        if len(matching) != 1:
            raise Gate12C2DrawProfileError(
                "worker profile must contain one worker-4 draw-255 row "
                f"for {regime_id}"
            )
        row = dict(matching[0])
        if (
            int(row.get("outer_experiment_count", 0)) <= 0
            or int(row.get("output_bytes", 0)) <= 0
            or int(
                row.get("process_tree_memory", {}).get(
                    "peak_process_tree_rss_bytes",
                    0,
                )
            )
            <= 0
            or row.get("process_tree_memory", {}).get("monitor_error")
            is not None
            or int(row.get("unaccounted_rejection_count", -1)) != 0
            or int(row.get("exhausted_incomplete_stream_count", -1))
            != 0
        ):
            raise Gate12C2DrawProfileError(
                f"worker profile row is not resource-admissible: {regime_id}"
            )
        admitted[regime_id] = row
    return {
        "path": path.as_posix(),
        "file_sha256": _sha256_file(path),
        "payload_sha256": payload_hash,
        "payload": payload,
        "worker_4_rows": admitted,
    }


def _verify_worker_carry_forward(
    receipt_path: Path,
    *,
    plan: Mapping[str, Any],
    worker_profile: Mapping[str, Any],
) -> dict[str, Any]:
    path = Path(receipt_path).resolve()
    payload = _read_json_mapping(path, label="worker carry-forward receipt")
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "epistemic_status",
            "surface_id",
            "prior_worker_profile_file_sha256",
            "prior_worker_profile_payload_sha256",
            "prior_commit",
            "current_commit",
            "worker_count",
            "current_implementation_sha256",
            "current_numerical_environment_sha256",
            "worker_critical_file_comparison",
            "bounded_equivalence_smoke",
            "status",
            "locked_execution_authorized",
            "real_held_out_execution_authorized",
            "N2_open",
            "N3_open",
            "carry_forward_receipt_payload_sha256",
        },
        context="worker carry-forward receipt",
    )
    payload_hash = _verify_self_hash(
        payload,
        hash_field="carry_forward_receipt_payload_sha256",
        label="worker carry-forward receipt",
    )
    expected = {
        "schema_version": WORKER_CARRY_FORWARD_SCHEMA_VERSION,
        "epistemic_status": "development_worker_selection_carry_forward_only",
        "surface_id": "development",
        "prior_worker_profile_file_sha256": worker_profile["file_sha256"],
        "prior_worker_profile_payload_sha256": worker_profile[
            "payload_sha256"
        ],
        "prior_commit": worker_profile["payload"]["source_commit"],
        "current_commit": plan["source_commit"],
        "worker_count": WORKER_COUNT,
        "current_implementation_sha256": plan["implementation_sha256"],
        "current_numerical_environment_sha256": plan[
            "numerical_environment_sha256"
        ],
        "status": "pass",
        "locked_execution_authorized": False,
        "real_held_out_execution_authorized": False,
        "N2_open": False,
        "N3_open": False,
    }
    for key, value in expected.items():
        if payload[key] != value:
            raise Gate12C2DrawProfileError(
                f"worker carry-forward changed frozen field {key!r}"
            )
    comparison = payload["worker_critical_file_comparison"]
    required_comparison_paths = {
        "tools/gate12c2_synthetic_lab.py",
        "tools/gate12c2_development_shards.py",
        "tools/gate12c2_draw_profile.py",
    }
    if (
        not isinstance(comparison, Mapping)
        or set(comparison) != required_comparison_paths
    ):
        raise Gate12C2DrawProfileError(
            "worker carry-forward lacks critical-file comparison"
        )
    current_hash_by_path = {
        "tools/gate12c2_synthetic_lab.py": plan[
            "implementation_sha256"
        ]["gate12c2_synthetic_lab.py"],
        "tools/gate12c2_development_shards.py": plan[
            "implementation_sha256"
        ]["gate12c2_development_shards.py"],
        "tools/gate12c2_draw_profile.py": plan[
            "implementation_sha256"
        ]["gate12c2_draw_profile.py"],
    }
    for critical_path, row in comparison.items():
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {"prior_sha256", "current_sha256", "status"}
            or row["current_sha256"]
            != current_hash_by_path[critical_path]
            or (
                row["prior_sha256"] is not None
                and not _is_sha256(row["prior_sha256"])
            )
            or row["status"]
            not in {
                "unchanged",
                "changed_with_bounded_equivalence_smoke",
                "new_shared_path_with_bounded_equivalence_smoke",
            }
        ):
            raise Gate12C2DrawProfileError(
                f"worker carry-forward critical-file row is invalid: "
                f"{critical_path}"
            )
    smoke = payload["bounded_equivalence_smoke"]
    if not isinstance(smoke, Mapping):
        raise Gate12C2DrawProfileError(
            "worker carry-forward lacks bounded equivalence smoke"
        )
    _require_exact_keys(
        smoke,
        {
            "worker_counts",
            "regime_projection_sha256",
            "status",
            "scientific_outcomes_interpreted",
        },
        context="worker carry-forward bounded smoke",
    )
    if smoke["worker_counts"] != [1, 4] or smoke["status"] != "pass":
        raise Gate12C2DrawProfileError(
            "worker carry-forward smoke did not pass worker 1/4"
        )
    if smoke["scientific_outcomes_interpreted"] is not False:
        raise Gate12C2DrawProfileError(
            "worker carry-forward smoke inspected scientific outcomes"
        )
    projection_rows = smoke["regime_projection_sha256"]
    if not isinstance(projection_rows, Mapping) or set(projection_rows) != {
        str(row["regime_id"]) for row in REGIME_SPECIFICATIONS
    }:
        raise Gate12C2DrawProfileError(
            "worker carry-forward smoke changed the regime surface"
        )
    for regime_id, hashes in projection_rows.items():
        if (
            not isinstance(hashes, Mapping)
            or set(hashes) != {"1", "4"}
            or not all(_is_sha256(value) for value in hashes.values())
            or hashes["1"] != hashes["4"]
        ):
            raise Gate12C2DrawProfileError(
                "worker 1/4 scientific projections differ for "
                f"{regime_id}"
            )
    return {
        "path": path.as_posix(),
        "file_sha256": _sha256_file(path),
        "payload_sha256": payload_hash,
    }


def _project_output_bytes(
    plan: Mapping[str, Any],
    worker_profile: Mapping[str, Any],
) -> int:
    total = 0.0
    rows = worker_profile["worker_4_rows"]
    for regime in REGIME_SPECIFICATIONS:
        regime_id = str(regime["regime_id"])
        baseline = rows[regime_id]
        bytes_per_outer_at_255 = (
            int(baseline["output_bytes"])
            / int(baseline["outer_experiment_count"])
        )
        outer_count = int(regime["outer_count"])
        total += bytes_per_outer_at_255 * outer_count * sum(
            draw_count / PREFIX_COUNTS[0]
            for draw_count in PREFIX_COUNTS
        )
    return int(math.ceil(total))


def _verify_recovery_bundle(
    bundle_path: Path,
    *,
    expected_commit: str,
    implementation_sha256: Mapping[str, str],
    scratch_root: Path,
) -> dict[str, Any]:
    bundle = Path(bundle_path).resolve()
    if not bundle.is_file():
        raise Gate12C2DrawProfileError(
            f"recovery bundle does not exist: {bundle}"
        )
    scratch_parent = Path(scratch_root).resolve()
    if not scratch_parent.is_dir():
        raise Gate12C2DrawProfileError(
            "restore scratch root must already exist"
        )
    verify = subprocess.run(
        ["git", "bundle", "verify", str(bundle)],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if verify.returncode != 0:
        raise Gate12C2DrawProfileError(
            f"git bundle verify failed: {verify.stderr[-2000:]}"
        )
    with tempfile.TemporaryDirectory(
        prefix="g12c2-restore-",
        dir=str(scratch_parent),
    ) as temporary:
        checkout = Path(temporary) / "r"
        clone = subprocess.run(
            ["git", "clone", "--no-checkout", str(bundle), str(checkout)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if clone.returncode != 0:
            raise Gate12C2DrawProfileError(
                f"standalone bundle clone failed: {clone.stderr[-2000:]}"
            )
        checkout_result = subprocess.run(
            [
                "git",
                "-C",
                str(checkout),
                "checkout",
                "--detach",
                str(expected_commit),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if checkout_result.returncode != 0:
            raise Gate12C2DrawProfileError(
                "standalone bundle requires the exact source commit: "
                f"{checkout_result.stderr[-2000:]}"
            )
        fsck = subprocess.run(
            ["git", "-C", str(checkout), "fsck", "--full"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if fsck.returncode != 0:
            raise Gate12C2DrawProfileError(
                f"restored bundle failed git fsck: {fsck.stderr[-2000:]}"
            )
        restored_head = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        if restored_head != expected_commit:
            raise Gate12C2DrawProfileError(
                "restored bundle HEAD differs from exact source commit"
            )
        status = subprocess.run(
            ["git", "-C", str(checkout), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        if status.strip():
            raise Gate12C2DrawProfileError(
                "standalone restored worktree is not clean"
            )
        for name, expected_hash in implementation_sha256.items():
            restored_path = checkout / "tools" / name
            if not restored_path.is_file() or (
                _sha256_file(restored_path) != expected_hash
            ):
                raise Gate12C2DrawProfileError(
                    f"restored implementation blob differs: {name}"
                )
    return {
        "bundle_path": bundle.as_posix(),
        "bundle_file_sha256": _sha256_file(bundle),
        "bundle_bytes": bundle.stat().st_size,
        "git_bundle_verify": "pass",
        "standalone_clone": "pass",
        "explicit_checkout": "pass",
        "restored_head": expected_commit,
        "git_fsck_full": "pass",
        "restored_worktree_clean": True,
        "implementation_blob_identity": "pass",
    }


def _serialize_mechanical_preflight(
    plan: Mapping[str, Any],
    *,
    output_root: Path,
    preflight_id: str,
    checks: Mapping[str, Mapping[str, Any]],
    recovery: Mapping[str, Any],
    worker_carry_forward: Mapping[str, Any],
    resource_projection: Mapping[str, Any],
) -> dict[str, Any]:
    verified = verify_draw_profile_plan(plan)
    if not str(preflight_id).strip():
        raise Gate12C2DrawProfileError("preflight_id must be nonempty")
    if set(checks) != set(REQUIRED_PREFLIGHT_CHECKS):
        raise Gate12C2DrawProfileError(
            "mechanical preflight evidence differs from the frozen allowlist"
        )
    if any(row.get("status") != "pass" for row in checks.values()):
        raise Gate12C2DrawProfileError(
            "every mechanically derived preflight check must pass"
        )
    payload: dict[str, Any] = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "preflight_id": str(preflight_id),
        "epistemic_status": "development_draw_profile_preflight_only",
        "surface_id": "development",
        "preflight_status": "pass",
        "preflight_issuer": "mechanical",
        "development_execution_authorized": False,
        "locked_execution_authorized": False,
        "real_held_out_execution_authorized": False,
        "N2_open": False,
        "N3_open": False,
        "public_claim": False,
        "scientific_outcomes_inspected": False,
        "draw_profile_plan_payload_sha256": verified[
            "draw_profile_plan_payload_sha256"
        ],
        "implementation_sha256": dict(
            verified["implementation_sha256"]
        ),
        "numerical_environment_sha256": verified[
            "numerical_environment_sha256"
        ],
        "resource_policy_payload_sha256": verified["resource_policy"][
            "resource_policy_payload_sha256"
        ],
        "output_root": _output_root(output_root),
        "worker_count": WORKER_COUNT,
        "recovery_evidence": dict(recovery),
        "worker_carry_forward_evidence": dict(worker_carry_forward),
        "resource_projection": dict(resource_projection),
        "checks": {
            key: dict(value) for key, value in sorted(checks.items())
        },
    }
    payload["preflight_receipt_payload_sha256"] = _sha256_bytes(
        _canonical_json_bytes(payload)
    )
    return payload


def issue_mechanical_preflight(
    *,
    plan_path: Path,
    output_root: Path,
    preflight_id: str,
    recovery_bundle_path: Path,
    worker_profile_receipt_path: Path,
    worker_carry_forward_receipt_path: Path,
    restore_scratch_root: Path,
) -> dict[str, Any]:
    """Perform the preflight checks and issue a non-authorizing receipt.

    This function accepts paths to evidence, never caller-supplied pass/fail
    booleans. It is intentionally not invoked by the plan builder.
    """

    plan_file = Path(plan_path).resolve()
    plan = _read_json_mapping(plan_file, label="exact draw-profile plan")
    if plan_file.read_bytes() != _canonical_json_bytes(plan):
        raise Gate12C2DrawProfileError(
            "exact draw-profile plan file is not canonical JSON"
        )
    verified = verify_draw_profile_plan(plan)
    root_evidence = _verify_fresh_output_root(output_root)
    current_environment = shards._numerical_environment_receipt()
    current_environment_hash = _sha256_bytes(
        _canonical_json_bytes(current_environment)
    )
    if current_environment_hash != verified[
        "numerical_environment_sha256"
    ]:
        raise Gate12C2DrawProfileError(
            "current numerical environment differs from the exact plan"
        )
    worker_profile = _verify_prior_worker_profile(
        worker_profile_receipt_path
    )
    carry_forward = _verify_worker_carry_forward(
        worker_carry_forward_receipt_path,
        plan=verified,
        worker_profile=worker_profile,
    )
    projected_output_bytes = _project_output_bytes(
        verified,
        worker_profile,
    )
    disk = shutil.disk_usage(Path(output_root).resolve().parent)
    projected_with_safety = int(
        math.ceil(
            projected_output_bytes * RESOURCE_DISK_SAFETY_FACTOR
        )
    )
    projected_remaining = int(disk.free) - projected_with_safety
    disk_pass = bool(
        projected_remaining
        >= int(
            disk.free
            * RESOURCE_MINIMUM_REMAINING_FRACTION_OF_PRERUN_FREE_DISK
        )
    )
    if not disk_pass:
        raise Gate12C2DrawProfileError(
            "exact draw-profile disk projection failed the frozen gate"
        )
    recovery = _verify_recovery_bundle(
        recovery_bundle_path,
        expected_commit=str(verified["source_commit"]),
        implementation_sha256=verified["implementation_sha256"],
        scratch_root=restore_scratch_root,
    )
    resource_projection = {
        "worker_profile_receipt_path": worker_profile["path"],
        "worker_profile_receipt_file_sha256": worker_profile[
            "file_sha256"
        ],
        "worker_profile_receipt_payload_sha256": worker_profile[
            "payload_sha256"
        ],
        "projected_output_bytes": projected_output_bytes,
        "disk_projection_safety_factor": RESOURCE_DISK_SAFETY_FACTOR,
        "projected_output_bytes_with_safety": projected_with_safety,
        "disk_free_bytes_at_preflight": int(disk.free),
        "projected_remaining_free_bytes": projected_remaining,
        "minimum_remaining_free_bytes": int(
            disk.free
            * RESOURCE_MINIMUM_REMAINING_FRACTION_OF_PRERUN_FREE_DISK
        ),
        "disk_gate_pass": disk_pass,
    }
    evidence_rows = {
        "complete_plan_rebuilt": {
            "status": "pass",
            "evidence_sha256": verified[
                "draw_profile_plan_payload_sha256"
            ],
        },
        "implementation_hashes_verified": {
            "status": "pass",
            "evidence_sha256": _sha256_bytes(
                _canonical_json_bytes(verified["implementation_sha256"])
            ),
        },
        "numerical_environment_verified": {
            "status": "pass",
            "evidence_sha256": current_environment_hash,
        },
        "all_nine_subplans_verified": {
            "status": "pass",
            "configuration_count": len(verified["configurations"]),
        },
        "outer_id_surfaces_verified": {
            "status": "pass",
            "evidence_sha256": _sha256_bytes(
                _canonical_json_bytes(
                    [
                        row["subplan"]["outer_experiment_indices"]
                        for row in verified["configurations"]
                    ]
                )
            ),
        },
        "accepted_prefix_namespaces_verified": {
            "status": "pass",
            "draw_counts": list(PREFIX_COUNTS),
        },
        "S2_amendment_verified": {
            "status": "pass",
            "evidence_sha256": verified[
                "S2_amendment_payload_sha256"
            ],
        },
        "strict_no_outcome_analyzer_verified": {
            "status": "pass",
            "evidence_sha256": verified["implementation_sha256"][
                "gate12c2_draw_stability.py"
            ],
        },
        "output_root_verified": {
            "status": "pass",
            "evidence_sha256": _sha256_bytes(
                _canonical_json_bytes(root_evidence)
            ),
        },
        "disk_gate_verified": {
            "status": "pass",
            "evidence_sha256": _sha256_bytes(
                _canonical_json_bytes(resource_projection)
            ),
        },
        "standalone_recovery_bundle_verified": {
            "status": "pass",
            "evidence_sha256": recovery["bundle_file_sha256"],
        },
        "short_path_restore_rehearsal_verified": {
            "status": "pass",
            "restored_head": recovery["restored_head"],
        },
        "worker_profile_carry_forward_verified": {
            "status": "pass",
            "evidence_sha256": carry_forward["payload_sha256"],
        },
        "profile_root_transaction_boundary_verified": {
            "status": "pass",
            "partial_artifact_count": 0,
        },
        "no_active_competing_execution_verified": {
            "status": "pass",
            "coordinator_lock_present": False,
        },
        "no_scientific_outcomes_inspected": {
            "status": "pass",
            "scientific_outcomes_inspected": False,
        },
        "locked_and_held_out_boundaries_verified": {
            "status": "pass",
            "locked_execution_authorized": False,
            "real_held_out_execution_authorized": False,
            "N2_open": False,
            "N3_open": False,
        },
    }
    return _serialize_mechanical_preflight(
        verified,
        output_root=output_root,
        preflight_id=preflight_id,
        checks=evidence_rows,
        recovery=recovery,
        worker_carry_forward=carry_forward,
        resource_projection=resource_projection,
    )


def _verify_preflight(
    plan: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    output_root: Path,
) -> dict[str, Any]:
    verified = verify_draw_profile_plan(plan)
    if not isinstance(receipt, Mapping):
        raise Gate12C2DrawProfileError(
            "draw profile preflight must be a mapping"
        )
    supplied = dict(receipt)
    _require_exact_keys(
        supplied,
        {
            "schema_version",
            "preflight_id",
            "epistemic_status",
            "surface_id",
            "preflight_status",
            "preflight_issuer",
            "development_execution_authorized",
            "locked_execution_authorized",
            "real_held_out_execution_authorized",
            "N2_open",
            "N3_open",
            "public_claim",
            "scientific_outcomes_inspected",
            "draw_profile_plan_payload_sha256",
            "implementation_sha256",
            "numerical_environment_sha256",
            "resource_policy_payload_sha256",
            "output_root",
            "worker_count",
            "recovery_evidence",
            "worker_carry_forward_evidence",
            "resource_projection",
            "checks",
            "preflight_receipt_payload_sha256",
        },
        context="draw profile preflight",
    )
    claimed = supplied["preflight_receipt_payload_sha256"]
    unhashed = dict(supplied)
    unhashed.pop("preflight_receipt_payload_sha256")
    if claimed != _sha256_bytes(_canonical_json_bytes(unhashed)):
        raise Gate12C2DrawProfileError(
            "draw profile preflight hash mismatch"
        )
    expected_values = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "epistemic_status": "development_draw_profile_preflight_only",
        "surface_id": "development",
        "preflight_status": "pass",
        "preflight_issuer": "mechanical",
        "development_execution_authorized": False,
        "locked_execution_authorized": False,
        "real_held_out_execution_authorized": False,
        "N2_open": False,
        "N3_open": False,
        "public_claim": False,
        "scientific_outcomes_inspected": False,
        "draw_profile_plan_payload_sha256": verified[
            "draw_profile_plan_payload_sha256"
        ],
        "implementation_sha256": verified["implementation_sha256"],
        "numerical_environment_sha256": verified[
            "numerical_environment_sha256"
        ],
        "resource_policy_payload_sha256": verified["resource_policy"][
            "resource_policy_payload_sha256"
        ],
        "output_root": _output_root(output_root),
        "worker_count": WORKER_COUNT,
    }
    for key, expected in expected_values.items():
        if supplied[key] != expected:
            raise Gate12C2DrawProfileError(
                f"draw profile preflight changed frozen field {key!r}"
            )
    if not str(supplied["preflight_id"]).strip():
        raise Gate12C2DrawProfileError("preflight_id must be nonempty")
    recovery = supplied["recovery_evidence"]
    if (
        not isinstance(recovery, Mapping)
        or not _is_sha256(recovery.get("bundle_file_sha256"))
        or recovery.get("restored_head") != verified["source_commit"]
        or recovery.get("git_bundle_verify") != "pass"
        or recovery.get("standalone_clone") != "pass"
        or recovery.get("explicit_checkout") != "pass"
        or recovery.get("git_fsck_full") != "pass"
        or recovery.get("restored_worktree_clean") is not True
        or recovery.get("implementation_blob_identity") != "pass"
    ):
        raise Gate12C2DrawProfileError(
            "draw profile preflight has invalid recovery evidence"
        )
    bundle_path = Path(str(recovery["bundle_path"]))
    if (
        not bundle_path.is_file()
        or _sha256_file(bundle_path)
        != recovery["bundle_file_sha256"]
    ):
        raise Gate12C2DrawProfileError(
            "draw profile preflight recovery bundle changed or disappeared"
        )
    carry = supplied["worker_carry_forward_evidence"]
    if (
        not isinstance(carry, Mapping)
        or not _is_sha256(carry.get("file_sha256"))
        or not _is_sha256(carry.get("payload_sha256"))
    ):
        raise Gate12C2DrawProfileError(
            "draw profile preflight has invalid worker carry-forward evidence"
        )
    carry_path = Path(str(carry["path"]))
    if (
        not carry_path.is_file()
        or _sha256_file(carry_path) != carry["file_sha256"]
    ):
        raise Gate12C2DrawProfileError(
            "draw profile worker carry-forward evidence changed or disappeared"
        )
    resource_projection = supplied["resource_projection"]
    if (
        not isinstance(resource_projection, Mapping)
        or resource_projection.get("disk_gate_pass") is not True
        or int(resource_projection.get("projected_output_bytes", 0)) <= 0
        or int(
            resource_projection.get(
                "projected_remaining_free_bytes",
                -1,
            )
        )
        < int(resource_projection.get("minimum_remaining_free_bytes", 0))
    ):
        raise Gate12C2DrawProfileError(
            "draw profile preflight has invalid resource projection"
        )
    worker_profile_path = Path(
        str(resource_projection["worker_profile_receipt_path"])
    )
    if (
        not worker_profile_path.is_file()
        or _sha256_file(worker_profile_path)
        != resource_projection[
            "worker_profile_receipt_file_sha256"
        ]
    ):
        raise Gate12C2DrawProfileError(
            "draw profile worker profile evidence changed or disappeared"
        )
    checks = supplied["checks"]
    if not isinstance(checks, Mapping):
        raise Gate12C2DrawProfileError(
            "draw profile preflight checks must be a mapping"
        )
    _require_exact_keys(
        checks,
        set(REQUIRED_PREFLIGHT_CHECKS),
        context="draw profile preflight checks",
    )
    for check_name, check in checks.items():
        if not isinstance(check, Mapping) or check.get("status") != "pass":
            raise Gate12C2DrawProfileError(
                "draw profile preflight contains a failed mechanical check: "
                f"{check_name}"
            )
    return supplied


def build_execution_authorization(
    plan: Mapping[str, Any],
    preflight_receipt: Mapping[str, Any],
    *,
    output_root: Path,
    authorization_id: str,
    purpose: str,
    expires_at_utc: str,
) -> dict[str, Any]:
    """Explicitly authorize exactly one plan and output root."""

    verified = verify_draw_profile_plan(plan)
    preflight = _verify_preflight(
        verified,
        preflight_receipt,
        output_root=output_root,
    )
    if not str(authorization_id).strip() or not str(purpose).strip():
        raise Gate12C2DrawProfileError(
            "authorization_id and purpose must be nonempty"
        )
    try:
        expiration = datetime.fromisoformat(
            str(expires_at_utc).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise Gate12C2DrawProfileError(
            "authorization expiration must be ISO-8601"
        ) from exc
    if (
        expiration.tzinfo is None
        or expiration <= datetime.now(timezone.utc)
    ):
        raise Gate12C2DrawProfileError(
            "authorization expiration must be in the future"
        )
    payload: dict[str, Any] = {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "authorization_id": str(authorization_id),
        "epistemic_status": (
            "development_draw_profile_execution_authorization_only"
        ),
        "surface_id": "development",
        "development_execution_authorized": True,
        "locked_execution_authorized": False,
        "real_held_out_execution_authorized": False,
        "N2_open": False,
        "N3_open": False,
        "public_claim": False,
        "scientific_calibration_interpretation_authorized": False,
        "draw_profile_plan_payload_sha256": verified[
            "draw_profile_plan_payload_sha256"
        ],
        "preflight_receipt_payload_sha256": preflight[
            "preflight_receipt_payload_sha256"
        ],
        "implementation_sha256": dict(
            verified["implementation_sha256"]
        ),
        "numerical_environment_sha256": verified[
            "numerical_environment_sha256"
        ],
        "resource_policy_payload_sha256": verified["resource_policy"][
            "resource_policy_payload_sha256"
        ],
        "output_root": _output_root(output_root),
        "worker_count": WORKER_COUNT,
        "purpose": str(purpose),
        "single_use": True,
        "authorization_status": "unconsumed",
        "expires_at_utc": expiration.astimezone(timezone.utc).isoformat(),
    }
    payload["authorization_receipt_payload_sha256"] = _sha256_bytes(
        _canonical_json_bytes(payload)
    )
    return payload


def _verify_authorization(
    plan: Mapping[str, Any],
    preflight_receipt: Mapping[str, Any],
    authorization_receipt: Mapping[str, Any],
    *,
    output_root: Path,
) -> dict[str, Any]:
    verified = verify_draw_profile_plan(plan)
    preflight = _verify_preflight(
        verified,
        preflight_receipt,
        output_root=output_root,
    )
    if not isinstance(authorization_receipt, Mapping):
        raise Gate12C2DrawProfileError(
            "draw profile authorization must be a mapping"
        )
    supplied = dict(authorization_receipt)
    _require_exact_keys(
        supplied,
        {
            "schema_version",
            "authorization_id",
            "epistemic_status",
            "surface_id",
            "development_execution_authorized",
            "locked_execution_authorized",
            "real_held_out_execution_authorized",
            "N2_open",
            "N3_open",
            "public_claim",
            "scientific_calibration_interpretation_authorized",
            "draw_profile_plan_payload_sha256",
            "preflight_receipt_payload_sha256",
            "implementation_sha256",
            "numerical_environment_sha256",
            "resource_policy_payload_sha256",
            "output_root",
            "worker_count",
            "purpose",
            "single_use",
            "authorization_status",
            "expires_at_utc",
            "authorization_receipt_payload_sha256",
        },
        context="draw profile authorization",
    )
    claimed = supplied["authorization_receipt_payload_sha256"]
    unhashed = dict(supplied)
    unhashed.pop("authorization_receipt_payload_sha256")
    if claimed != _sha256_bytes(_canonical_json_bytes(unhashed)):
        raise Gate12C2DrawProfileError(
            "draw profile authorization hash mismatch"
        )
    expected_values = {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "epistemic_status": (
            "development_draw_profile_execution_authorization_only"
        ),
        "surface_id": "development",
        "development_execution_authorized": True,
        "locked_execution_authorized": False,
        "real_held_out_execution_authorized": False,
        "N2_open": False,
        "N3_open": False,
        "public_claim": False,
        "scientific_calibration_interpretation_authorized": False,
        "draw_profile_plan_payload_sha256": verified[
            "draw_profile_plan_payload_sha256"
        ],
        "preflight_receipt_payload_sha256": preflight[
            "preflight_receipt_payload_sha256"
        ],
        "implementation_sha256": verified["implementation_sha256"],
        "numerical_environment_sha256": verified[
            "numerical_environment_sha256"
        ],
        "resource_policy_payload_sha256": verified["resource_policy"][
            "resource_policy_payload_sha256"
        ],
        "output_root": _output_root(output_root),
        "worker_count": WORKER_COUNT,
        "single_use": True,
        "authorization_status": "unconsumed",
    }
    for key, expected in expected_values.items():
        if supplied[key] != expected:
            raise Gate12C2DrawProfileError(
                f"draw profile authorization changed frozen field {key!r}"
            )
    if not str(supplied["authorization_id"]).strip():
        raise Gate12C2DrawProfileError(
            "authorization_id must be nonempty"
        )
    if not str(supplied["purpose"]).strip():
        raise Gate12C2DrawProfileError(
            "authorization purpose must be nonempty"
        )
    try:
        expiration = datetime.fromisoformat(
            str(supplied["expires_at_utc"]).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise Gate12C2DrawProfileError(
            "authorization expiration must be ISO-8601"
        ) from exc
    if (
        expiration.tzinfo is None
        or expiration <= datetime.now(timezone.utc)
    ):
        raise Gate12C2DrawProfileError(
            "draw profile authorization has expired"
        )
    return supplied


def _write_or_verify(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = _canonical_json_bytes(payload)
    if path.exists():
        if path.read_bytes() != encoded:
            raise Gate12C2DrawProfileError(
                f"existing coordinator artifact differs: {path}"
            )
        return
    shards._atomic_write(path, encoded)


def _scan_profile_root(
    root: Path,
    *,
    active_lock: Mapping[str, Any] | None,
) -> None:
    destination = Path(root).resolve()
    if not destination.exists():
        return
    partials = _partial_artifacts(destination)
    if partials:
        raise Gate12C2DrawProfileError(
            f"profile root contains stale transaction artifacts: {partials}"
        )
    allowed_root_files = {
        "plan.json",
        EXECUTION_EVIDENCE_NAME,
        RESOURCE_RECEIPT_NAME,
        EXECUTION_RECEIPT_NAME,
        COORDINATOR_LOCK_NAME,
    }
    allowed_root_directories = {"control", "runs"}
    unknown = [
        path.name
        for path in destination.iterdir()
        if (
            path.is_file() and path.name not in allowed_root_files
        )
        or (
            path.is_dir() and path.name not in allowed_root_directories
        )
    ]
    if unknown:
        raise Gate12C2DrawProfileError(
            f"profile root contains unknown coordinator artifacts: "
            f"{sorted(unknown)}"
        )
    control = destination / "control"
    if control.exists():
        unknown_control = [
            path.name
            for path in control.iterdir()
            if (
                path.is_dir()
                or (
                    path.is_file()
                    and path.name
                    not in {
                        "preflight.json",
                        "authorization.json",
                        AUTHORIZATION_CONSUMED_NAME,
                    }
                    and not path.name.startswith("recovery-")
                )
            )
        ]
        if unknown_control:
            raise Gate12C2DrawProfileError(
                "profile control root contains unknown artifacts: "
                f"{sorted(unknown_control)}"
            )
    lock_path = destination / COORDINATOR_LOCK_NAME
    if active_lock is None:
        if lock_path.exists():
            raise Gate12C2DrawProfileError(
                "profile root contains an active or orphan coordinator lock"
            )
    else:
        if (
            not lock_path.is_file()
            or lock_path.read_bytes()
            != _canonical_json_bytes(active_lock)
        ):
            raise Gate12C2DrawProfileError(
                "coordinator ownership lock changed during execution"
            )


def _acquire_coordinator_lock(
    root: Path,
    *,
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    destination = Path(root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    lock_payload: dict[str, Any] = {
        "schema_version": "gate12c2_draw_profile_coordinator_lock_v0.1",
        "plan_payload_sha256": plan[
            "draw_profile_plan_payload_sha256"
        ],
        "implementation_sha256": plan["implementation_sha256"],
        "authorization_receipt_payload_sha256": authorization[
            "authorization_receipt_payload_sha256"
        ],
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    lock_payload["lock_payload_sha256"] = _sha256_bytes(
        _canonical_json_bytes(lock_payload)
    )
    lock_path = destination / COORDINATOR_LOCK_NAME
    encoded = _canonical_json_bytes(lock_payload)
    try:
        descriptor = os.open(
            lock_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        raise Gate12C2DrawProfileError(
            "a coordinator lock already exists"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        if lock_path.exists():
            lock_path.unlink()
        raise
    return lock_payload


def _release_coordinator_lock(
    root: Path,
    *,
    active_lock: Mapping[str, Any],
) -> None:
    lock_path = Path(root).resolve() / COORDINATOR_LOCK_NAME
    if (
        not lock_path.is_file()
        or lock_path.read_bytes() != _canonical_json_bytes(active_lock)
    ):
        raise Gate12C2DrawProfileError(
            "cannot release a changed coordinator ownership lock"
        )
    lock_path.unlink()


def _pid_is_running(pid: int) -> bool:
    if int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, PermissionError, OverflowError, ValueError):
        return False
    return True


def recover_stale_coordinator_lock(
    plan: Mapping[str, Any],
    *,
    output_root: Path,
    recovery_id: str,
    reason: str,
) -> dict[str, Any]:
    """Record an explicit stale-lock recovery; never deletes partials."""

    verified = verify_draw_profile_plan(plan)
    destination = Path(output_root).resolve()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", str(recovery_id)):
        raise Gate12C2DrawProfileError(
            "recovery_id may contain only letters, digits, dot, dash, underscore"
        )
    if not str(reason).strip():
        raise Gate12C2DrawProfileError(
            "coordinator recovery requires a nonempty reason"
        )
    partials = _partial_artifacts(destination)
    if partials:
        raise Gate12C2DrawProfileError(
            "coordinator recovery cannot remove a lock while partial "
            f"artifacts remain: {partials}"
        )
    lock_path = destination / COORDINATOR_LOCK_NAME
    lock = _read_json_mapping(
        lock_path,
        label="coordinator ownership lock",
    )
    _verify_self_hash(
        lock,
        hash_field="lock_payload_sha256",
        label="coordinator ownership lock",
    )
    if (
        lock.get("plan_payload_sha256")
        != verified["draw_profile_plan_payload_sha256"]
        or lock.get("implementation_sha256")
        != verified["implementation_sha256"]
        or lock.get("hostname") != socket.gethostname()
    ):
        raise Gate12C2DrawProfileError(
            "coordinator lock does not belong to this exact plan and host"
        )
    if _pid_is_running(int(lock.get("pid", -1))):
        raise Gate12C2DrawProfileError(
            "coordinator lock owner is still running"
        )
    payload: dict[str, Any] = {
        "schema_version": (
            "gate12c2_draw_profile_coordinator_recovery_v0.1"
        ),
        "recovery_id": str(recovery_id),
        "epistemic_status": "development_transaction_recovery_only",
        "surface_id": "development",
        "draw_profile_plan_payload_sha256": verified[
            "draw_profile_plan_payload_sha256"
        ],
        "recovered_lock_payload_sha256": lock["lock_payload_sha256"],
        "reason": str(reason),
        "partial_artifact_count": 0,
        "prior_owner_pid_not_running": True,
        "locked_execution_authorized": False,
        "real_held_out_execution_authorized": False,
        "N2_open": False,
        "N3_open": False,
        "scientific_outcomes_inspected": False,
    }
    payload["recovery_receipt_payload_sha256"] = _sha256_bytes(
        _canonical_json_bytes(payload)
    )
    receipt_path = (
        destination / "control" / f"recovery-{recovery_id}.json"
    )
    _write_or_verify(receipt_path, payload)
    if lock_path.read_bytes() != _canonical_json_bytes(lock):
        raise Gate12C2DrawProfileError(
            "coordinator lock changed before explicit recovery commit"
        )
    lock_path.unlink()
    return payload


def _authorization_consumption_receipt(
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": (
            "gate12c2_draw_profile_authorization_consumption_v0.1"
        ),
        "authorization_receipt_payload_sha256": authorization[
            "authorization_receipt_payload_sha256"
        ],
        "draw_profile_plan_payload_sha256": plan[
            "draw_profile_plan_payload_sha256"
        ],
        "output_root": authorization["output_root"],
        "single_use": True,
        "authorization_status": "consumed_for_this_execution_lineage",
    }
    payload["consumption_receipt_payload_sha256"] = _sha256_bytes(
        _canonical_json_bytes(payload)
    )
    return payload


def _physical_ram_bytes() -> int:
    hardware = throughput._hardware_receipt()
    windows = hardware.get("windows_cim")
    if isinstance(windows, Mapping):
        value = int(windows.get("RAMBytes", 0))
        if value > 0:
            return value
    if hasattr(os, "sysconf"):
        try:
            pages = int(os.sysconf("SC_PHYS_PAGES"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            if pages > 0 and page_size > 0:
                return pages * page_size
        except (OSError, ValueError):
            pass
    raise Gate12C2DrawProfileError(
        "physical RAM could not be measured mechanically"
    )


def _directory_bytes(path: Path) -> int:
    return sum(
        item.stat().st_size
        for item in Path(path).rglob("*")
        if item.is_file()
    )


def _outer_id_surface_sha256(subplan: Mapping[str, Any]) -> str:
    return _sha256_bytes(
        _canonical_json_bytes(subplan["outer_experiment_indices"])
    )


def _build_execution_evidence(
    plan: Mapping[str, Any],
    *,
    configuration_rows: list[Mapping[str, Any]],
    wall_seconds: float,
    process_cpu_seconds: float,
    process_tree_memory: Mapping[str, Any],
    physical_ram_bytes: int,
    disk_free_bytes_before: int,
    disk_free_bytes_after: int,
    output_bytes: int,
) -> dict[str, Any]:
    verified = verify_draw_profile_plan(plan)
    if (
        not math.isfinite(float(wall_seconds))
        or float(wall_seconds) <= 0.0
        or not math.isfinite(float(process_cpu_seconds))
        or float(process_cpu_seconds) < 0.0
        or int(physical_ram_bytes) <= 0
        or int(disk_free_bytes_before) <= 0
        or int(disk_free_bytes_after) < 0
        or int(output_bytes) <= 0
    ):
        raise Gate12C2DrawProfileError(
            "execution resource measurements are invalid"
        )
    expected_configurations = {
        str(row["configuration_id"]): row
        for row in verified["configurations"]
    }
    rows = [dict(row) for row in configuration_rows]
    if {str(row.get("configuration_id")) for row in rows} != set(
        expected_configurations
    ):
        raise Gate12C2DrawProfileError(
            "execution evidence changed the exact configuration surface"
        )
    expected_row_keys = {
        "configuration_id",
        "regime_id",
        "draw_count",
        "worker_count",
        "outer_experiment_count",
        "outer_id_surface_sha256",
        "all_outer_indices_present",
        "plan_payload_sha256",
        "scientific_projection_sha256",
        "index_payload_sha256",
        "new_shard_count",
        "reused_shard_count",
        "endpoint_draw_attempts",
        "endpoint_draw_acceptances",
        "rejection_reason_counts",
        "unaccounted_rejection_count",
        "exhausted_incomplete_stream_count",
        "derived_preflight_receipt_payload_sha256",
        "derived_authorization_receipt_payload_sha256",
        "scientific_outcomes_exposed",
    }
    for row in rows:
        _require_exact_keys(
            row,
            expected_row_keys,
            context="draw-profile configuration evidence",
        )
        expected = expected_configurations[str(row["configuration_id"])]
        if (
            row["regime_id"] != expected["regime_id"]
            or int(row["draw_count"]) != int(expected["draw_count"])
            or int(row["worker_count"]) != WORKER_COUNT
            or row["plan_payload_sha256"]
            != expected["subplan"]["plan_payload_sha256"]
            or row["outer_id_surface_sha256"]
            != _outer_id_surface_sha256(expected["subplan"])
            or int(row["outer_experiment_count"])
            != len(expected["subplan"]["outer_experiment_indices"])
            or row["scientific_outcomes_exposed"] is not False
            or not _is_sha256(row["scientific_projection_sha256"])
            or not _is_sha256(row["index_payload_sha256"])
            or not _is_sha256(
                row["derived_preflight_receipt_payload_sha256"]
            )
            or not _is_sha256(
                row["derived_authorization_receipt_payload_sha256"]
            )
        ):
            raise Gate12C2DrawProfileError(
                "execution evidence differs from an exact subplan"
            )
    payload: dict[str, Any] = {
        "schema_version": EXECUTION_EVIDENCE_SCHEMA_VERSION,
        "epistemic_status": "development_execution_resource_evidence_only",
        "surface_id": "development",
        "draw_profile_plan_payload_sha256": verified[
            "draw_profile_plan_payload_sha256"
        ],
        "implementation_sha256": verified["implementation_sha256"],
        "numerical_environment_sha256": verified[
            "numerical_environment_sha256"
        ],
        "resource_policy_payload_sha256": verified["resource_policy"][
            "resource_policy_payload_sha256"
        ],
        "worker_count": WORKER_COUNT,
        "configuration_count": len(rows),
        "configuration_results": sorted(
            rows,
            key=lambda row: str(row["configuration_id"]),
        ),
        "resource_measurements": {
            "wall_seconds": float(wall_seconds),
            "process_cpu_seconds": float(process_cpu_seconds),
            "process_tree_memory": dict(process_tree_memory),
            "physical_ram_bytes": int(physical_ram_bytes),
            "disk_free_bytes_before": int(disk_free_bytes_before),
            "disk_free_bytes_after": int(disk_free_bytes_after),
            "output_bytes_before_resource_receipts": int(output_bytes),
        },
        "locked_execution_authorized": False,
        "real_held_out_execution_authorized": False,
        "N2_open": False,
        "N3_open": False,
        "scientific_outcomes_exposed": False,
    }
    payload["execution_evidence_payload_sha256"] = _sha256_bytes(
        _canonical_json_bytes(payload)
    )
    return payload


def _verify_execution_evidence(
    plan: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    supplied = dict(evidence)
    _verify_self_hash(
        supplied,
        hash_field="execution_evidence_payload_sha256",
        label="draw-profile execution evidence",
    )
    measurements = supplied.get("resource_measurements")
    rows = supplied.get("configuration_results")
    if not isinstance(measurements, Mapping) or not isinstance(rows, list):
        raise Gate12C2DrawProfileError(
            "draw-profile execution evidence is incomplete"
        )
    expected = _build_execution_evidence(
        plan,
        configuration_rows=rows,
        wall_seconds=float(measurements["wall_seconds"]),
        process_cpu_seconds=float(measurements["process_cpu_seconds"]),
        process_tree_memory=measurements["process_tree_memory"],
        physical_ram_bytes=int(measurements["physical_ram_bytes"]),
        disk_free_bytes_before=int(
            measurements["disk_free_bytes_before"]
        ),
        disk_free_bytes_after=int(
            measurements["disk_free_bytes_after"]
        ),
        output_bytes=int(
            measurements["output_bytes_before_resource_receipts"]
        ),
    )
    if supplied != expected:
        raise Gate12C2DrawProfileError(
            "draw-profile execution evidence differs from its exact builder"
        )
    return expected


def _build_resource_receipt(
    plan: Mapping[str, Any],
    execution_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    verified = verify_draw_profile_plan(plan)
    evidence = _verify_execution_evidence(
        verified,
        execution_evidence,
    )
    measurements = evidence["resource_measurements"]
    memory = measurements["process_tree_memory"]
    memory_pass = bool(
        memory.get("monitor_error") is None
        and int(memory.get("sample_count", 0)) > 0
        and int(memory.get("peak_process_tree_rss_bytes", 0)) > 0
        and (
            int(memory["peak_process_tree_rss_bytes"])
            / int(measurements["physical_ram_bytes"])
        )
        <= RESOURCE_MAXIMUM_RSS_FRACTION_OF_PHYSICAL_RAM
    )
    projected_disk_cost = int(
        math.ceil(
            int(measurements["output_bytes_before_resource_receipts"])
            * RESOURCE_DISK_SAFETY_FACTOR
        )
    )
    projected_remaining = (
        int(measurements["disk_free_bytes_before"])
        - projected_disk_cost
    )
    minimum_remaining = int(
        int(measurements["disk_free_bytes_before"])
        * RESOURCE_MINIMUM_REMAINING_FRACTION_OF_PRERUN_FREE_DISK
    )
    disk_pass = projected_remaining >= minimum_remaining
    by_id = {
        str(row["configuration_id"]): row
        for row in evidence["configuration_results"]
    }
    draw_rows = []
    eligible_draw_counts = []
    for draw_count in PREFIX_COUNTS:
        expected = [
            row
            for row in verified["configurations"]
            if int(row["draw_count"]) == draw_count
        ]
        checks = []
        for configuration in expected:
            row = by_id[str(configuration["configuration_id"])]
            configuration_pass = bool(
                row["all_outer_indices_present"] is True
                and int(row["unaccounted_rejection_count"]) == 0
                and int(row["exhausted_incomplete_stream_count"]) == 0
                and row["plan_payload_sha256"]
                == configuration["subplan"]["plan_payload_sha256"]
                and row["outer_id_surface_sha256"]
                == _outer_id_surface_sha256(configuration["subplan"])
                and _is_sha256(row["index_payload_sha256"])
                and _is_sha256(row["scientific_projection_sha256"])
            )
            checks.append(
                {
                    "configuration_id": configuration[
                        "configuration_id"
                    ],
                    "status": "pass" if configuration_pass else "fail",
                }
            )
        eligible = bool(
            memory_pass
            and disk_pass
            and all(row["status"] == "pass" for row in checks)
        )
        if eligible:
            eligible_draw_counts.append(draw_count)
        draw_rows.append(
            {
                "draw_count": draw_count,
                "configuration_checks": checks,
                "resource_eligible": eligible,
            }
        )
    payload: dict[str, Any] = {
        "schema_version": RESOURCE_RECEIPT_SCHEMA_VERSION,
        "epistemic_status": "development_resource_gate_only",
        "surface_id": "development",
        "draw_profile_plan_payload_sha256": verified[
            "draw_profile_plan_payload_sha256"
        ],
        "execution_evidence_payload_sha256": evidence[
            "execution_evidence_payload_sha256"
        ],
        "execution_evidence": evidence,
        "implementation_sha256": verified["implementation_sha256"],
        "numerical_environment_sha256": verified[
            "numerical_environment_sha256"
        ],
        "resource_policy": verified["resource_policy"],
        "worker_count": WORKER_COUNT,
        "memory_gate": {
            "peak_process_tree_rss_bytes": int(
                memory.get("peak_process_tree_rss_bytes", 0)
            ),
            "physical_ram_bytes": int(
                measurements["physical_ram_bytes"]
            ),
            "maximum_fraction": (
                RESOURCE_MAXIMUM_RSS_FRACTION_OF_PHYSICAL_RAM
            ),
            "monitor_sample_count": int(memory.get("sample_count", 0)),
            "monitor_error": memory.get("monitor_error"),
            "status": "pass" if memory_pass else "fail",
        },
        "disk_gate": {
            "disk_free_bytes_before": int(
                measurements["disk_free_bytes_before"]
            ),
            "measured_output_bytes": int(
                measurements["output_bytes_before_resource_receipts"]
            ),
            "safety_factor": RESOURCE_DISK_SAFETY_FACTOR,
            "projected_disk_cost_bytes": projected_disk_cost,
            "projected_remaining_free_bytes": projected_remaining,
            "minimum_remaining_free_bytes": minimum_remaining,
            "status": "pass" if disk_pass else "fail",
        },
        "draw_count_rows": draw_rows,
        "eligible_draw_counts": eligible_draw_counts,
        "status": "pass" if eligible_draw_counts else "fail",
        "locked_execution_authorized": False,
        "real_held_out_execution_authorized": False,
        "N2_open": False,
        "N3_open": False,
        "scientific_outcomes_exposed": False,
    }
    payload["resource_receipt_payload_sha256"] = _sha256_bytes(
        _canonical_json_bytes(payload)
    )
    return payload


def verify_resource_evidence_chain(
    plan: Mapping[str, Any],
    execution_receipt: Mapping[str, Any],
    resource_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    verified = verify_draw_profile_plan(plan)
    supplied_resource = dict(resource_receipt)
    _verify_self_hash(
        supplied_resource,
        hash_field="resource_receipt_payload_sha256",
        label="draw-profile resource receipt",
    )
    expected_resource = _build_resource_receipt(
        verified,
        supplied_resource.get("execution_evidence", {}),
    )
    if supplied_resource != expected_resource:
        raise Gate12C2DrawProfileError(
            "resource receipt differs from mechanically derived evidence"
        )
    supplied_execution = dict(execution_receipt)
    _verify_self_hash(
        supplied_execution,
        hash_field="execution_receipt_payload_sha256",
        label="draw-profile execution receipt",
    )
    _require_exact_keys(
        supplied_execution,
        {
            "schema_version",
            "plan_id",
            "epistemic_status",
            "surface_id",
            "locked_execution_authorized",
            "real_held_out_execution_authorized",
            "N2_open",
            "N3_open",
            "public_claim",
            "scientific_calibration_result",
            "scientific_outcomes_exposed",
            "draw_profile_plan_payload_sha256",
            "preflight_receipt_payload_sha256",
            "authorization_receipt_payload_sha256",
            "authorization_consumption_receipt_payload_sha256",
            "execution_evidence_payload_sha256",
            "resource_receipt_payload_sha256",
            "configuration_count",
            "configuration_results",
            "next_step",
            "execution_receipt_payload_sha256",
        },
        context="draw-profile execution receipt",
    )
    expected_links = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "plan_id": PLAN_ID,
        "epistemic_status": "development_draw_profile_execution_only",
        "surface_id": "development",
        "locked_execution_authorized": False,
        "real_held_out_execution_authorized": False,
        "N2_open": False,
        "N3_open": False,
        "public_claim": False,
        "scientific_calibration_result": None,
        "scientific_outcomes_exposed": False,
        "draw_profile_plan_payload_sha256": verified[
            "draw_profile_plan_payload_sha256"
        ],
        "execution_evidence_payload_sha256": supplied_resource[
            "execution_evidence_payload_sha256"
        ],
        "resource_receipt_payload_sha256": supplied_resource[
            "resource_receipt_payload_sha256"
        ],
        "configuration_count": len(verified["configurations"]),
    }
    for key, value in expected_links.items():
        if supplied_execution[key] != value:
            raise Gate12C2DrawProfileError(
                f"execution/resource chain changed frozen field {key!r}"
            )
    for key in (
        "preflight_receipt_payload_sha256",
        "authorization_receipt_payload_sha256",
        "authorization_consumption_receipt_payload_sha256",
    ):
        if not _is_sha256(supplied_execution[key]):
            raise Gate12C2DrawProfileError(
                f"execution receipt has an invalid evidence digest: {key}"
            )
    if supplied_execution["configuration_results"] != supplied_resource[
        "execution_evidence"
    ]["configuration_results"]:
        raise Gate12C2DrawProfileError(
            "execution receipt and resource evidence configuration rows differ"
        )
    return {
        "status": supplied_resource["status"],
        "eligible_draw_counts": list(
            supplied_resource["eligible_draw_counts"]
        ),
        "resource_receipt_payload_sha256": supplied_resource[
            "resource_receipt_payload_sha256"
        ],
        "execution_evidence_payload_sha256": supplied_resource[
            "execution_evidence_payload_sha256"
        ],
        "source_plan_payload_sha256_by_regime_and_draw_count": {
            str(configuration["regime_id"]): {
                str(draw_count): next(
                    str(row["subplan"]["plan_payload_sha256"])
                    for row in verified["configurations"]
                    if row["regime_id"] == configuration["regime_id"]
                    and int(row["draw_count"]) == draw_count
                )
                for draw_count in PREFIX_COUNTS
            }
            for configuration in REGIME_SPECIFICATIONS
        },
    }


def execute_draw_profile(
    plan: Mapping[str, Any],
    *,
    output_root: Path,
    preflight_receipt: Mapping[str, Any] | None = None,
    authorization_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute or resume the exact plan without exposing raw outcomes."""

    verified = verify_draw_profile_plan(plan)
    destination = Path(output_root).resolve()
    if preflight_receipt is None or authorization_receipt is None:
        raise Gate12C2DrawProfileError(
            "draw profile execution requires exact preflight and "
            "authorization receipts"
        )
    authorization = _verify_authorization(
        verified,
        preflight_receipt,
        authorization_receipt,
        output_root=destination,
    )
    _scan_profile_root(destination, active_lock=None)
    completed_receipt = destination / EXECUTION_RECEIPT_NAME
    if completed_receipt.is_file():
        existing_execution = _read_json_mapping(
            completed_receipt,
            label="completed execution receipt",
        )
        existing_resource = _read_json_mapping(
            destination / RESOURCE_RECEIPT_NAME,
            label="completed resource receipt",
        )
        verify_resource_evidence_chain(
            verified,
            existing_execution,
            existing_resource,
        )
        return existing_execution

    active_lock = _acquire_coordinator_lock(
        destination,
        plan=verified,
        authorization=authorization,
    )
    monitor: throughput.ProcessTreeRssMonitor | None = None
    try:
        _scan_profile_root(destination, active_lock=active_lock)
        _write_or_verify(destination / "plan.json", verified)
        control_root = destination / "control"
        _write_or_verify(
            control_root / "preflight.json",
            dict(preflight_receipt),
        )
        _write_or_verify(
            control_root / "authorization.json",
            authorization,
        )
        consumption = _authorization_consumption_receipt(
            verified,
            authorization,
        )
        _write_or_verify(
            control_root / AUTHORIZATION_CONSUMED_NAME,
            consumption,
        )
        _scan_profile_root(destination, active_lock=active_lock)

        rows = []
        checks = {key: True for key in shards.REQUIRED_PREFLIGHT_CHECKS}
        disk_before = shutil.disk_usage(destination)
        physical_ram = _physical_ram_bytes()
        started_wall = time.perf_counter()
        started_cpu = time.process_time()
        monitor = throughput.ProcessTreeRssMonitor(os.getpid())
        monitor.start()
        for configuration in verified["configurations"]:
            configuration_id = str(configuration["configuration_id"])
            subplan = shards._verified_plan(configuration["subplan"])
            run_root = (
                destination / str(configuration["output_relative_path"])
            ).resolve()
            if not run_root.is_relative_to(destination):
                raise Gate12C2DrawProfileError(
                    "configuration output escaped the profile root"
                )
            derived_preflight = shards.build_no_outcome_preflight_receipt(
                subplan,
                output_dir=run_root,
                worker_count=WORKER_COUNT,
                preflight_id=(
                    "draw-profile-derived::"
                    f"{authorization['authorization_id']}::"
                    f"{configuration_id}"
                ),
                checks=checks,
            )
            derived_authorization = (
                shards.build_development_execution_authorization(
                    subplan,
                    derived_preflight,
                    output_dir=run_root,
                    worker_count=WORKER_COUNT,
                    authorization_id=(
                        "draw-profile-derived::"
                        f"{authorization['authorization_id']}::"
                        f"{configuration_id}"
                    ),
                    purpose=f"bounded-draw-profile::{configuration_id}",
                )
            )
            index = shards.execute_development_shard_plan(
                subplan,
                output_dir=run_root,
                worker_count=WORKER_COUNT,
                preflight_receipt=derived_preflight,
                authorization_receipt=derived_authorization,
            )
            verification = shards.verify_development_shard_index(
                subplan,
                output_dir=run_root,
            )
            operational = [
                row.get("operational_metrics", {})
                for row in index["shards"]
            ]
            rejection_counts: dict[str, int] = {}
            for operational_row in operational:
                for reason, count in operational_row.get(
                    "rejection_reason_counts",
                    {},
                ).items():
                    rejection_counts[str(reason)] = (
                        rejection_counts.get(str(reason), 0)
                        + int(count)
                    )
            rows.append(
                {
                    "configuration_id": configuration_id,
                    "regime_id": str(configuration["regime_id"]),
                    "draw_count": int(configuration["draw_count"]),
                    "worker_count": WORKER_COUNT,
                    "outer_experiment_count": int(
                        index["outer_experiment_count"]
                    ),
                    "outer_id_surface_sha256": (
                        _outer_id_surface_sha256(subplan)
                    ),
                    "all_outer_indices_present": bool(
                        index["all_outer_indices_present"]
                    ),
                    "plan_payload_sha256": str(
                        index["plan_payload_sha256"]
                    ),
                    "scientific_projection_sha256": str(
                        verification["scientific_projection_sha256"]
                    ),
                    "index_payload_sha256": str(
                        verification["index_payload_sha256"]
                    ),
                    "new_shard_count": sum(
                        row.get("mode") == "execute_new"
                        for row in operational
                    ),
                    "reused_shard_count": sum(
                        row.get("mode") == "verify_existing"
                        for row in operational
                    ),
                    "endpoint_draw_attempts": sum(
                        int(row.get("endpoint_draw_attempts", 0))
                        for row in operational
                    ),
                    "endpoint_draw_acceptances": sum(
                        int(row.get("endpoint_draw_acceptances", 0))
                        for row in operational
                    ),
                    "rejection_reason_counts": dict(
                        sorted(rejection_counts.items())
                    ),
                    "unaccounted_rejection_count": sum(
                        int(
                            row.get(
                                "unaccounted_rejection_count",
                                0,
                            )
                        )
                        for row in operational
                    ),
                    "exhausted_incomplete_stream_count": sum(
                        int(
                            row.get(
                                "exhausted_incomplete_stream_count",
                                0,
                            )
                        )
                        for row in operational
                    ),
                    "derived_preflight_receipt_payload_sha256": (
                        derived_preflight[
                            "preflight_receipt_payload_sha256"
                        ]
                    ),
                    "derived_authorization_receipt_payload_sha256": (
                        derived_authorization[
                            "authorization_receipt_payload_sha256"
                        ]
                    ),
                    "scientific_outcomes_exposed": False,
                }
            )
        wall_seconds = time.perf_counter() - started_wall
        cpu_seconds = time.process_time() - started_cpu
        memory = monitor.stop()
        monitor = None
        if any(
            not row["all_outer_indices_present"]
            or row["unaccounted_rejection_count"] != 0
            or row["exhausted_incomplete_stream_count"] != 0
            for row in rows
        ):
            raise Gate12C2DrawProfileError(
                "one or more draw-profile configurations failed completeness"
            )
        disk_after = shutil.disk_usage(destination)
        output_bytes = _directory_bytes(destination)
        execution_evidence = _build_execution_evidence(
            verified,
            configuration_rows=rows,
            wall_seconds=wall_seconds,
            process_cpu_seconds=cpu_seconds,
            process_tree_memory=memory,
            physical_ram_bytes=physical_ram,
            disk_free_bytes_before=int(disk_before.free),
            disk_free_bytes_after=int(disk_after.free),
            output_bytes=output_bytes,
        )
        resource_receipt = _build_resource_receipt(
            verified,
            execution_evidence,
        )
        _write_or_verify(
            destination / EXECUTION_EVIDENCE_NAME,
            execution_evidence,
        )
        _write_or_verify(
            destination / RESOURCE_RECEIPT_NAME,
            resource_receipt,
        )
        _scan_profile_root(destination, active_lock=active_lock)
        receipt: dict[str, Any] = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "plan_id": PLAN_ID,
            "epistemic_status": "development_draw_profile_execution_only",
            "surface_id": "development",
            "locked_execution_authorized": False,
            "real_held_out_execution_authorized": False,
            "N2_open": False,
            "N3_open": False,
            "public_claim": False,
            "scientific_calibration_result": None,
            "scientific_outcomes_exposed": False,
            "draw_profile_plan_payload_sha256": verified[
                "draw_profile_plan_payload_sha256"
            ],
            "preflight_receipt_payload_sha256": preflight_receipt[
                "preflight_receipt_payload_sha256"
            ],
            "authorization_receipt_payload_sha256": authorization[
                "authorization_receipt_payload_sha256"
            ],
            "authorization_consumption_receipt_payload_sha256": (
                consumption["consumption_receipt_payload_sha256"]
            ),
            "execution_evidence_payload_sha256": execution_evidence[
                "execution_evidence_payload_sha256"
            ],
            "resource_receipt_payload_sha256": resource_receipt[
                "resource_receipt_payload_sha256"
            ],
            "configuration_count": len(rows),
            "configuration_results": execution_evidence[
                "configuration_results"
            ],
            "next_step": (
                "build one exact read-only analysis manifest and emit only "
                "the strict no-outcome draw-stability projection"
            ),
        }
        receipt["execution_receipt_payload_sha256"] = _sha256_bytes(
            _canonical_json_bytes(receipt)
        )
        _scan_profile_root(destination, active_lock=active_lock)
        _write_or_verify(
            destination / EXECUTION_RECEIPT_NAME,
            receipt,
        )
        verify_resource_evidence_chain(
            verified,
            receipt,
            resource_receipt,
        )
        _release_coordinator_lock(
            destination,
            active_lock=active_lock,
        )
        _scan_profile_root(destination, active_lock=None)
        return receipt
    finally:
        if monitor is not None:
            monitor.stop()
