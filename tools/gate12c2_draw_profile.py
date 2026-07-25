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
from pathlib import Path
from typing import Any, Mapping

import gate12c2_development_shards as shards
import gate12c2_draw_stability as stability
import gate12c2_synthetic_lab as lab


PLAN_SCHEMA_VERSION = "gate12c2_draw_profile_plan_v0.1"
PREFLIGHT_SCHEMA_VERSION = "gate12c2_draw_profile_preflight_v0.1"
AUTHORIZATION_SCHEMA_VERSION = (
    "gate12c2_draw_profile_execution_authorization_v0.1"
)
RECEIPT_SCHEMA_VERSION = "gate12c2_draw_profile_execution_receipt_v0.1"
PLAN_ID = "gate12c2-development-accepted-valid-draw-scaling-v0.2"
WORKER_COUNT = 4
S2_AMENDMENT_PAYLOAD_SHA256 = (
    "d6163a5e7979971e7e2623e4b1a4e66ec692cbf5f1b546b6b0e812b59a3549b8"
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
        "gate12c2_draw_stability.py": Path(stability.__file__).resolve(),
        "gate12c2_synthetic_lab.py": Path(lab.__file__).resolve(),
        "run_gate12c2_draw_profile.py": Path(__file__)
        .with_name("run_gate12c2_draw_profile.py")
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


def build_draw_profile_plan(*, source_commit: str) -> dict[str, Any]:
    """Build the one admitted post-hardening draw-profile plan."""

    if not str(source_commit).strip():
        raise Gate12C2DrawProfileError("source_commit must be nonempty")
    configurations = []
    for regime in REGIME_SPECIFICATIONS:
        for draw_count in stability.PREFIX_COUNTS:
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
        "prefix_counts": list(stability.PREFIX_COUNTS),
        "reference_draw_count": stability.REFERENCE_DRAW_COUNT,
        "S2_amendment_payload_sha256": S2_AMENDMENT_PAYLOAD_SHA256,
        "standalone_off_device_recovery_required": True,
        "implementation_sha256": _implementation_hashes(),
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


def build_no_outcome_preflight(
    plan: Mapping[str, Any],
    *,
    output_root: Path,
    preflight_id: str,
    recovery_bundle_sha256: str,
    checks: Mapping[str, bool],
) -> dict[str, Any]:
    """Build an outcome-blind receipt; this does not authorize execution."""

    verified = verify_draw_profile_plan(plan)
    normalized_checks = {
        str(key): bool(value) for key, value in checks.items()
    }
    if set(normalized_checks) != set(REQUIRED_PREFLIGHT_CHECKS):
        raise Gate12C2DrawProfileError(
            "draw profile preflight checks differ from the frozen allowlist"
        )
    if not all(normalized_checks.values()):
        raise Gate12C2DrawProfileError(
            "every draw profile preflight check must pass"
        )
    if not str(preflight_id).strip():
        raise Gate12C2DrawProfileError("preflight_id must be nonempty")
    if (
        not isinstance(recovery_bundle_sha256, str)
        or len(recovery_bundle_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in recovery_bundle_sha256.lower()
        )
    ):
        raise Gate12C2DrawProfileError(
            "preflight requires a lowercase-or-uppercase SHA-256 recovery "
            "bundle identity"
        )
    payload: dict[str, Any] = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "preflight_id": str(preflight_id),
        "epistemic_status": "development_draw_profile_preflight_only",
        "surface_id": "development",
        "preflight_status": "pass",
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
        "output_root": _output_root(output_root),
        "worker_count": WORKER_COUNT,
        "recovery_bundle_sha256": recovery_bundle_sha256.lower(),
        "checks": dict(sorted(normalized_checks.items())),
    }
    payload["preflight_receipt_payload_sha256"] = _sha256_bytes(
        _canonical_json_bytes(payload)
    )
    return payload


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
            "development_execution_authorized",
            "locked_execution_authorized",
            "real_held_out_execution_authorized",
            "N2_open",
            "N3_open",
            "public_claim",
            "scientific_outcomes_inspected",
            "draw_profile_plan_payload_sha256",
            "implementation_sha256",
            "output_root",
            "worker_count",
            "recovery_bundle_sha256",
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
    recovery_hash = supplied["recovery_bundle_sha256"]
    if (
        not isinstance(recovery_hash, str)
        or len(recovery_hash) != 64
        or any(
            character not in "0123456789abcdef"
            for character in recovery_hash
        )
    ):
        raise Gate12C2DrawProfileError(
            "draw profile preflight has an invalid recovery SHA-256"
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
    if any(value is not True for value in checks.values()):
        raise Gate12C2DrawProfileError(
            "draw profile preflight contains a failed check"
        )
    return supplied


def build_execution_authorization(
    plan: Mapping[str, Any],
    preflight_receipt: Mapping[str, Any],
    *,
    output_root: Path,
    authorization_id: str,
    purpose: str,
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
        "output_root": _output_root(output_root),
        "worker_count": WORKER_COUNT,
        "purpose": str(purpose),
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
            "output_root",
            "worker_count",
            "purpose",
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
        "output_root": _output_root(output_root),
        "worker_count": WORKER_COUNT,
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
    destination.mkdir(parents=True, exist_ok=True)
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

    rows = []
    checks = {key: True for key in shards.REQUIRED_PREFLIGHT_CHECKS}
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
                f"draw-profile-derived::{authorization['authorization_id']}"
                f"::{configuration_id}"
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
        rows.append(
            {
                "configuration_id": configuration_id,
                "regime_id": str(configuration["regime_id"]),
                "draw_count": int(configuration["draw_count"]),
                "worker_count": WORKER_COUNT,
                "outer_experiment_count": int(
                    index["outer_experiment_count"]
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
                "unaccounted_rejection_count": sum(
                    int(row.get("unaccounted_rejection_count", 0))
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
    if any(
        not row["all_outer_indices_present"]
        or row["unaccounted_rejection_count"] != 0
        or row["exhausted_incomplete_stream_count"] != 0
        for row in rows
    ):
        raise Gate12C2DrawProfileError(
            "one or more draw-profile configurations failed completeness"
        )
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
        "configuration_count": len(rows),
        "configuration_results": rows,
        "next_step": (
            "build one exact read-only analysis manifest and emit only the "
            "strict gate12c2_no_outcome_draw_stability_v0.1 projection"
        ),
    }
    receipt["execution_receipt_payload_sha256"] = _sha256_bytes(
        _canonical_json_bytes(receipt)
    )
    _write_or_verify(destination / "execution-receipt.json", receipt)
    return receipt
