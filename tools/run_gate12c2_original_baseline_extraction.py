#!/usr/bin/env python3
"""Run an authorized Gate12C-2 v0.8 baseline extraction."""


from __future__ import annotations

import argparse
import importlib.util
import os
import socket
import sys
from pathlib import Path
from typing import Any

def _load_local_module(name: str, filename: str) -> Any:
    path = Path(__file__).resolve().with_name(filename)
    existing = sys.modules.get(name)
    if existing is not None:
        existing_path = getattr(existing, "__file__", None)
        if existing_path is not None and Path(existing_path).resolve() == path:
            return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("local module loader unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


gate = _load_local_module(
    "gate12c2_original_baseline_commitments",
    "gate12c2_original_baseline_commitments.py",
)
def _load_reviewed_chain(
    plan: dict[str, Any], repository: Path
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    binding_schema = plan["implementation_binding_contract"]
    candidate, candidate_file_hash = gate.read_schema_receipt(
        Path(binding_schema["artifact_path"]),
        exact_fields=binding_schema["exact_top_level_fields"],
        hash_field="implementation_candidate_binding_payload_sha256",
    )
    gate.validate_candidate_binding(
        plan, candidate, repo_root=repository, current_head=None
    )
    review_schema = plan["review_receipt_schemas"][
        "fresh_implementation_review_verdict"
    ]
    review, review_file_hash = gate.read_schema_receipt(
        Path(review_schema["artifact_path"]),
        exact_fields=review_schema["exact_top_level_fields"],
        hash_field="fresh_implementation_review_payload_sha256",
    )
    gate.validate_implementation_review(
        plan,
        review,
        candidate_file_sha256=candidate_file_hash,
        candidate_payload_sha256=candidate[
            "implementation_candidate_binding_payload_sha256"
        ],
        source_commit=candidate["source_commit"],
        candidate=candidate,
    )
    authority_schema = plan["reviewed_implementation_authority_contract"]
    authority, authority_file_hash = gate.read_schema_receipt(
        Path(authority_schema["artifact_path"]),
        exact_fields=authority_schema["exact_top_level_fields"],
        hash_field="reviewed_implementation_authority_payload_sha256",
    )
    gate.validate_reviewed_authority(
        plan,
        authority,
        candidate=candidate,
        candidate_file_sha256=candidate_file_hash,
        review=review,
        review_file_sha256=review_file_hash,
    )
    return authority, authority_file_hash, candidate


def _observe_surface(
    plan: dict[str, Any],
) -> dict[str, gate.ArtifactObservation]:
    outcome_fields = plan["artifact_lifecycle_contract"][
        "outcome_field_by_role"
    ]
    observations: dict[str, gate.ArtifactObservation] = {}
    for row in plan["artifact_path_surface"]:
        role = row["role"]
        final = Path(row["final_path"])
        pending = Path(row["pending_path"])
        outcome = None
        if final.is_file() and role in outcome_fields:
            try:
                raw = final.read_bytes()
                if raw.endswith(b"\n"):
                    payload = gate.require_mapping(
                        gate.strict_json_loads(raw[:-1], canonical=True)
                    )
                    outcome = payload.get(outcome_fields[role])
            except Exception:
                outcome = "__invalid__"
        observations[role] = gate.ArtifactObservation(
            final_exists=final.is_file(),
            pending_exists=pending.exists(),
            outcome=outcome,
        )
    return observations


def _load_controls(
    plan: dict[str, Any],
    repository: Path,
    *,
    now_ns: int | None,
) -> tuple[
    dict[str, Any],
    str,
    dict[str, Any],
    str,
    dict[str, Any],
    str,
    dict[str, Any],
    str,
]:
    authority, authority_file_hash, candidate = _load_reviewed_chain(plan, repository)
    rows = gate.artifact_rows_by_role(plan)
    preflight_schema = plan["control_receipt_schemas"]["extraction_preflight"]
    preflight, preflight_file_hash = gate.read_schema_receipt(
        Path(rows["extraction_preflight"]["final_path"]),
        exact_fields=preflight_schema["exact_top_level_fields"],
        hash_field="preflight_payload_sha256",
        code="AUTHORIZATION_INVALID",
    )
    gate.validate_preflight_payload(
        plan,
        preflight,
        scope="extraction",
        reviewed_authority_file_sha256=authority_file_hash,
        reviewed_authority_payload_sha256=authority[
            "reviewed_implementation_authority_payload_sha256"
        ],
        now_ns=now_ns,
    )
    authorization_schema = plan["control_receipt_schemas"][
        "extraction_authorization"
    ]
    authorization, authorization_file_hash = gate.read_schema_receipt(
        Path(rows["extraction_authorization"]["final_path"]),
        exact_fields=authorization_schema["exact_top_level_fields"],
        hash_field="authorization_payload_sha256",
        code="AUTHORIZATION_INVALID",
    )
    gate.validate_authorization_payload(
        plan,
        authorization,
        preflight,
        scope="extraction",
        preflight_file_sha256=preflight_file_hash,
        now_ns=now_ns,
    )
    verdict_schema = plan["control_receipt_schemas"][
        "extraction_authorization_verdict"
    ]
    verdict, verdict_file_hash = gate.read_schema_receipt(
        Path(rows["extraction_authorization_verdict"]["final_path"]),
        exact_fields=verdict_schema["exact_top_level_fields"],
        hash_field="authorization_verdict_payload_sha256",
        code="AUTHORIZATION_INVALID",
    )
    verifier_rows = [
        row
        for row in candidate["implementation_files"]
        if row["relative_path"]
        == "tools/verify_gate12c2_original_baseline_authorization.py"
    ]
    if len(verifier_rows) != 1:
        raise gate.Gate12C2OriginalBaselineError(
            "IMPLEMENTATION_BYTE_IDENTITY_MISMATCH"
        )
    verifier_row = verifier_rows[0]
    gate.validate_authorization_verdict_payload(
        plan,
        verdict,
        authorization,
        preflight,
        scope="extraction",
        preflight_file_sha256=preflight_file_hash,
        authorization_file_sha256=authorization_file_hash,
        verifier_relative_path=verifier_row["relative_path"],
        verifier_file_sha256=verifier_row["file_sha256"],
        verifier_git_blob_oid=verifier_row["git_blob_oid"],
        now_ns=now_ns,
    )
    if verdict["outcome_kind"] != "pass":
        raise gate.Gate12C2OriginalBaselineError("AUTHORIZATION_INVALID")
    return (
        authority,
        authority_file_hash,
        preflight,
        preflight_file_hash,
        authorization,
        authorization_file_hash,
        verdict,
        verdict_file_hash,
    )


def _runtime_isolated() -> None:
    if (
        not sys.flags.isolated
        or not sys.dont_write_bytecode
        or os.environ.get("PYTHONPATH")
    ):
        raise gate.Gate12C2OriginalBaselineError("AUTHORIZATION_INVALID")


def _failure_leaf(
    plan: dict[str, Any],
    claim: dict[str, Any],
    *,
    failure_code: str,
    occurred_at_utc: str,
    preflight_file_hash: str,
    authorization_file_hash: str,
    verdict_file_hash: str,
    claim_file_hash: str,
    progress: dict[str, Any],
) -> dict[str, Any]:
    occurred_ns = gate.parse_utc_ns(
        occurred_at_utc, code="INTERNAL_SANITIZED_FAILURE"
    )
    claim_ns = gate.parse_utc_ns(
        claim.get("claimed_at_utc"), code="INTERNAL_SANITIZED_FAILURE"
    )
    if occurred_ns < claim_ns:
        raise gate.Gate12C2OriginalBaselineError("INTERNAL_SANITIZED_FAILURE")
    source_state = str(progress.get("source_state", ""))
    phase = str(progress.get("failure_phase", ""))
    selected_code = (
        failure_code
        if failure_code in gate.FAILURE_CODES
        else "INTERNAL_SANITIZED_FAILURE"
    )
    try:
        row = gate.matching_failure_row(
            plan,
            scope="extraction",
            source_state=source_state,
            failure_phase=phase,
            failure_code=selected_code,
        )
    except gate.Gate12C2OriginalBaselineError:
        selected_code = "INTERNAL_SANITIZED_FAILURE"
        row = gate.matching_failure_row(
            plan,
            scope="extraction",
            source_state=source_state,
            failure_phase=phase,
            failure_code=selected_code,
        )
    if row["failure_receipt_allowed"] is not True:
        raise gate.Gate12C2OriginalBaselineError(
            "INTERNAL_SANITIZED_FAILURE"
        )
    evidence_value = progress.get("evidence")
    evidence = (
        dict(evidence_value) if isinstance(evidence_value, dict) else {}
    )
    availability = gate.apply_availability_profile(
        plan,
        scope="extraction",
        profile_name=row["availability_profile"],
        evidence=evidence,
    )
    counts = {
        "configuration_count_reached": gate.require_int(
            progress.get("configuration_count_reached", 0), maximum=9
        ),
        "outer_experiment_count_reached": gate.require_int(
            progress.get("outer_experiment_count_reached", 0), maximum=768
        ),
        "shard_count_reached": gate.require_int(
            progress.get("shard_count_reached", 0), maximum=768
        ),
        "index_count_reached": gate.require_int(
            progress.get("index_count_reached", 0), maximum=9
        ),
    }
    schema = plan["extraction_failure_receipt"]
    payload = {
        "schema_version": schema["schema_version"],
        "gate_id": gate.GATE_ID,
        "authorization_scope": "extraction",
        "execution_claim_id": claim["execution_claim_id"],
        "state": "EXTRACTION_FAILURE_RECEIPT_PUBLISHED",
        "failure_code": selected_code,
        "failure_phase": phase,
        "source_state": source_state,
        "sequence_ordinal": 60,
        "occurred_at_utc": occurred_at_utc,
        **counts,
        "original_plan_payload_sha256": plan["original_input_lineage"][
            "original_plan_payload_sha256"
        ],
        "incident_manifest_payload_sha256": plan["original_input_lineage"][
            "incident_manifest_payload_sha256"
        ],
        "payload_seal_sha256": plan["original_input_lineage"][
            "payload_seal_payload_sha256"
        ],
        "artifact_path_surface_sha256": gate.ARTIFACT_PATH_SURFACE_SHA256,
        "reviewed_implementation_authority_file_sha256": claim[
            "reviewed_implementation_authority_file_sha256"
        ],
        "reviewed_implementation_authority_payload_sha256": claim[
            "reviewed_implementation_authority_payload_sha256"
        ],
        "preflight_file_sha256": preflight_file_hash,
        "preflight_payload_sha256": claim["preflight_payload_sha256"],
        "authorization_file_sha256": authorization_file_hash,
        "authorization_payload_sha256": claim[
            "authorization_payload_sha256"
        ],
        "authorization_verdict_file_sha256": verdict_file_hash,
        "authorization_verdict_payload_sha256": claim[
            "authorization_verdict_payload_sha256"
        ],
        "execution_claim_file_sha256": claim_file_hash,
        "execution_claim_payload_sha256": claim[
            "execution_claim_payload_sha256"
        ],
        "evidence_availability": row["availability_profile"],
        **availability,
        "scientific_values_emitted": False,
        "baseline_commitments_published": schema["baseline_commitments_published"],
    }
    gate.require_exact_keys(
        {**payload, "failure_receipt_payload_sha256": ""},
        schema["exact_top_level_fields"],
    )
    return gate.add_self_hash(payload, "failure_receipt_payload_sha256")


def _reverify_claimed_lineage(
    plan: dict[str, Any],
    lineage: dict[str, dict[str, Any]],
    controls: tuple[
        dict[str, Any],
        str,
        dict[str, Any],
        str,
        dict[str, Any],
        str,
        dict[str, Any],
        str,
    ],
    repository: Path,
    *,
    now_ns: int | None,
) -> dict[str, dict[str, Any]]:
    repeated_plan = gate.load_frozen_plan()
    gate.read_exact_bytes(
        gate.CONTRACT_PATH,
        gate.CONTRACT_FILE_SHA256,
        code="INPUT_LINEAGE_MISMATCH",
    )
    gate.validate_formal_design_pass(repeated_plan)
    gate.validate_upstream_authority(repeated_plan)
    repeated_lineage = gate.validate_original_input_lineage(repeated_plan)
    repeated_controls = _load_controls(
        repeated_plan, repository, now_ns=now_ns
    )
    if (
        repeated_plan != plan
        or repeated_lineage != lineage
        or repeated_controls != controls
    ):
        raise gate.Gate12C2OriginalBaselineError("INPUT_LINEAGE_MISMATCH")
    observations = _observe_surface(repeated_plan)
    phase = gate.classify_lifecycle_surface(
        repeated_plan, observations, liveness="ACTIVE"
    )
    if phase != "extraction_execution_claimed_owner_active":
        raise gate.Gate12C2OriginalBaselineError("AUTHORIZATION_INVALID")
    return repeated_lineage


def execute(
    repository: Path,
    *,
    execution_claim_id: str,
    launch_id: str,
    claimed_at_utc: str,
    now_ns: int | None = None,
) -> dict[str, Any]:
    _runtime_isolated()
    root = Path(repository).resolve()
    plan = gate.load_frozen_plan()
    gate.read_exact_bytes(
        gate.CONTRACT_PATH,
        gate.CONTRACT_FILE_SHA256,
        code="INPUT_LINEAGE_MISMATCH",
    )
    gate.validate_formal_design_pass(plan)
    gate.validate_upstream_authority(plan)
    lineage = gate.validate_original_input_lineage(plan)
    controls = _load_controls(plan, root, now_ns=now_ns)
    (
        authority,
        authority_file_hash,
        preflight,
        preflight_file_hash,
        authorization,
        authorization_file_hash,
        verdict,
        verdict_file_hash,
    ) = controls
    observations = _observe_surface(plan)
    phase = gate.classify_lifecycle_surface(
        plan,
        observations,
        temporal_predicate="extraction_preflight_and_authorization_fresh",
    )
    if phase != "extraction_authorization_verdict_pass_fresh":
        raise gate.Gate12C2OriginalBaselineError("AUTHORIZATION_INVALID")
    pid = os.getpid()
    creation = gate.query_process_creation_time_utc(pid)
    claim = gate.build_execution_claim_payload(
        plan,
        authorization,
        preflight,
        verdict,
        scope="extraction",
        execution_claim_id=execution_claim_id,
        launch_id=launch_id,
        claimed_at_utc=claimed_at_utc,
        owner_hostname=socket.gethostname(),
        owner_pid=pid,
        owner_process_creation_time_utc=creation,
        preflight_file_sha256=preflight_file_hash,
        authorization_file_sha256=authorization_file_hash,
        verdict_file_sha256=verdict_file_hash,
        now_ns=now_ns,
    )
    gate.publish_role(plan, "extraction_execution_claim", claim)
    claim_path = Path(
        gate.artifact_rows_by_role(plan)["extraction_execution_claim"][
            "final_path"
        ]
    )
    published_claim, claim_file_hash = gate.read_schema_receipt(
        claim_path,
        exact_fields=plan["control_receipt_schemas"][
            "extraction_execution_claim"
        ]["exact_top_level_fields"],
        hash_field="execution_claim_payload_sha256",
        code="CONCURRENT_EXECUTION",
    )
    if (
        published_claim != claim
        or gate.classify_claim_owner(published_claim) != "ACTIVE"
    ):
        raise gate.Gate12C2OriginalBaselineError("CONCURRENT_EXECUTION")

    progress = gate.new_extraction_progress()
    try:
        lineage = _reverify_claimed_lineage(
            plan,
            lineage,
            controls,
            root,
            now_ns=now_ns,
        )
        derived = gate.extract_commitments_after_claim(
            plan, lineage, progress=progress
        )
        leaf = gate.build_extraction_success_leaf(
            plan,
            derived,
            reviewed_authority_file_sha256=authority_file_hash,
            reviewed_authority_payload_sha256=authority[
                "reviewed_implementation_authority_payload_sha256"
            ],
            preflight_file_sha256=preflight_file_hash,
            preflight_payload_sha256=preflight["preflight_payload_sha256"],
            authorization_file_sha256=authorization_file_hash,
            authorization_payload_sha256=authorization[
                "authorization_payload_sha256"
            ],
            authorization_verdict_file_sha256=verdict_file_hash,
            authorization_verdict_payload_sha256=verdict[
                "authorization_verdict_payload_sha256"
            ],
            execution_claim_file_sha256=claim_file_hash,
            execution_claim_payload_sha256=claim[
                "execution_claim_payload_sha256"
            ],
        )
        terminal = gate.build_terminal_payload(
            plan,
            claim,
            leaf,
            scope="extraction",
            outcome_kind="success",
            claimed_at_utc=gate.utc_now_text(),
            reviewed_authority_file_sha256=authority_file_hash,
            preflight_file_sha256=preflight_file_hash,
            authorization_file_sha256=authorization_file_hash,
            verdict_file_sha256=verdict_file_hash,
            execution_claim_file_sha256=claim_file_hash,
        )
    except Exception as raw_error:
        error_code = (
            raw_error.code
            if isinstance(raw_error, gate.Gate12C2OriginalBaselineError)
            else "INTERNAL_SANITIZED_FAILURE"
        )
        failure_time = gate.utc_now_text()
        failure = _failure_leaf(
            plan,
            claim,
            failure_code=error_code,
            occurred_at_utc=failure_time,
            preflight_file_hash=preflight_file_hash,
            authorization_file_hash=authorization_file_hash,
            verdict_file_hash=verdict_file_hash,
            claim_file_hash=claim_file_hash,
            progress=progress,
        )
        failure_terminal = gate.build_terminal_payload(
            plan,
            claim,
            failure,
            scope="extraction",
            outcome_kind="failure",
            claimed_at_utc=failure_time,
            reviewed_authority_file_sha256=authority_file_hash,
            preflight_file_sha256=preflight_file_hash,
            authorization_file_sha256=authorization_file_hash,
            verdict_file_sha256=verdict_file_hash,
            execution_claim_file_sha256=claim_file_hash,
        )
        gate.update_extraction_progress(
            progress,
            failure_phase="failure_terminal_claim_publication",
        )
        gate.publish_role(plan, "extraction_terminal", failure_terminal)
        gate.update_extraction_progress(
            progress,
            source_state="EXTRACTION_FAILURE_TERMINAL_CLAIM_PUBLISHED",
            failure_phase="leaf_publication",
        )
        gate.publish_role(plan, "extraction_failure", failure)
        raise gate.Gate12C2OriginalBaselineError(
            failure["failure_code"]
        ) from None

    gate.update_extraction_progress(
        progress, failure_phase="terminal_claim_publication"
    )
    gate.publish_role(plan, "extraction_terminal", terminal)
    gate.update_extraction_progress(
        progress,
        source_state="EXTRACTION_SUCCESS_TERMINAL_CLAIM_PUBLISHED",
        failure_phase="leaf_publication",
    )
    gate.publish_role(plan, "extraction_success", leaf)
    return leaf


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--execution-claim-id", required=True)
    parser.add_argument("--launch-id", required=True)
    parser.add_argument("--claimed-at-utc", required=True)
    args = parser.parse_args(argv)
    execute(
        args.repository,
        execution_claim_id=args.execution_claim_id,
        launch_id=args.launch_id,
        claimed_at_utc=args.claimed_at_utc,
    )
    print(gate.EXTRACTION_PASS_LINE)
    return 0


def cli(argv: list[str] | None = None) -> int:
    try:
        return main(argv)
    except SystemExit:
        raise
    except gate.Gate12C2OriginalBaselineError as exc:
        print(gate.EXTRACTION_ERROR_PREFIX + exc.code, file=sys.stderr)
        return 2
    except Exception:
        print(
            gate.EXTRACTION_ERROR_PREFIX + "INTERNAL_SANITIZED_FAILURE",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
