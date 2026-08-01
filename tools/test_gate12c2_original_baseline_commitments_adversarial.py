#!/usr/bin/env python3
"""Adversarial tests for the Gate12C-2 v0.8 baseline gate."""

from __future__ import annotations

import ast
import contextlib
import copy
import gzip
import inspect
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from unittest import mock

import build_gate12c2_original_baseline_implementation_binding as binding_builder
import gate12c2_original_baseline_commitments as gate
import test_gate12c2_original_baseline_commitments as primary
import run_gate12c2_original_baseline_extraction as extraction_runner
import verify_gate12c2_original_baseline_commitments as independent
import verify_gate12c2_original_baseline_authorization as authorization_verifier


REPOSITORY = Path(__file__).resolve().parents[1]
DIGEST = "d" * 64
OTHER_DIGEST = "e" * 64
SOURCE_COMMIT = "a" * 40


def rehash_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    payload.pop("plan_payload_sha256", None)
    return gate.add_self_hash(payload, "plan_payload_sha256")


def nested_key_paths(
    value: object,
    key_name: str,
    *,
    path: tuple[object, ...] = (),
    active: bool = True,
) -> Iterator[tuple[object, ...]]:
    if isinstance(value, dict):
        for key, item in value.items():
            child_active = active and not str(key).startswith("historical_v0_")
            if child_active and key == key_name:
                yield path + (key,)
            yield from nested_key_paths(
                item,
                key_name,
                path=path + (key,),
                active=child_active,
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from nested_key_paths(
                item,
                key_name,
                path=path + (index,),
                active=active,
            )


def set_path(value: object, path: Sequence[object], replacement: object) -> None:
    cursor: Any = value
    for component in path[:-1]:
        cursor = cursor[component]
    cursor[path[-1]] = replacement


def observations_for_phase(
    plan: Mapping[str, Any], phase: Mapping[str, Any]
) -> dict[str, gate.ArtifactObservation]:
    required = phase["required_outcomes"]
    return {
        role: gate.ArtifactObservation(
            final_exists=role in phase["must_exist"],
            pending_exists=False,
            outcome=required.get(role),
        )
        for role in plan["artifact_lifecycle_contract"]["roles"]
    }


def control_chain(
    plan: Mapping[str, Any], scope: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    preflight_arguments: dict[str, Any] = {}
    if scope == "verifier":
        preflight_arguments = {
            "extraction_terminal_file_sha256": DIGEST,
            "extraction_terminal_payload_sha256": OTHER_DIGEST,
            "baseline_receipt_file_sha256": "1" * 64,
            "baseline_receipt_payload_sha256": "2" * 64,
        }
    preflight = gate.build_preflight_payload(
        plan,
        scope=scope,
        preflight_id=f"{scope}-preflight",
        issued_at_utc="2026-08-01T00:00:00Z",
        expires_at_utc="2026-08-01T00:20:00Z",
        reviewed_authority_file_sha256="3" * 64,
        reviewed_authority_payload_sha256="4" * 64,
        now_ns=gate.parse_utc_ns("2026-08-01T00:01:00Z"),
        **preflight_arguments,
    )
    authorization = gate.build_authorization_payload(
        plan,
        preflight,
        scope=scope,
        preflight_file_sha256="5" * 64,
        authorization_id=f"{scope}-authorization",
        issued_at_utc="2026-08-01T00:02:00Z",
        expires_at_utc="2026-08-01T00:19:00Z",
        now_ns=gate.parse_utc_ns("2026-08-01T00:03:00Z"),
    )
    verdict = gate.build_authorization_verdict_payload(
        plan,
        authorization,
        preflight,
        scope=scope,
        verification_id=f"{scope}-verification",
        verified_at_utc="2026-08-01T00:04:00Z",
        outcome_kind="pass",
        reason_code=None,
        preflight_file_sha256="5" * 64,
        authorization_file_sha256="6" * 64,
        verifier_relative_path=(
            "tools/verify_gate12c2_original_baseline_authorization.py"
        ),
        verifier_file_sha256="7" * 64,
        verifier_git_blob_oid="8" * 40,
        now_ns=gate.parse_utc_ns("2026-08-01T00:04:00Z"),
    )
    claim = gate.build_execution_claim_payload(
        plan,
        authorization,
        preflight,
        verdict,
        scope=scope,
        execution_claim_id=f"{scope}-claim",
        launch_id=f"{scope}-launch",
        claimed_at_utc="2026-08-01T00:05:00Z",
        owner_hostname="test-host",
        owner_pid=1234,
        owner_process_creation_time_utc="2026-08-01T00:00:30Z",
        preflight_file_sha256="5" * 64,
        authorization_file_sha256="6" * 64,
        verdict_file_sha256="9" * 64,
        now_ns=gate.parse_utc_ns("2026-08-01T00:05:00Z"),
    )
    return preflight, authorization, verdict, claim


class FrozenDesignMutationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = gate.load_frozen_plan()

    def test_every_active_nested_surface_digest_is_recursive_and_enforced(self) -> None:
        paths = list(
            nested_key_paths(self.plan, "artifact_path_surface_sha256")
        )
        self.assertGreater(len(paths), 1)
        self.assertTrue(
            all(
                self._value_at(self.plan, path)
                == gate.ARTIFACT_PATH_SURFACE_SHA256
                for path in paths
            )
        )
        for path in paths:
            if path == ("artifact_path_surface_sha256",):
                continue
            with self.subTest(path=path):
                altered = copy.deepcopy(self.plan)
                set_path(altered, path, OTHER_DIGEST)
                altered = rehash_plan(altered)
                with mock.patch.object(
                    gate, "PLAN_PAYLOAD_SHA256", altered["plan_payload_sha256"]
                ):
                    with self.assertRaises(
                        gate.Gate12C2OriginalBaselineError
                    ):
                        gate.validate_frozen_plan(altered)

    @staticmethod
    def _value_at(value: object, path: Sequence[object]) -> object:
        cursor: Any = value
        for component in path:
            cursor = cursor[component]
        return cursor

    def test_artifact_surface_change_cannot_reuse_frozen_digest(self) -> None:
        altered = copy.deepcopy(self.plan)
        row = altered["artifact_path_surface"][0]
        row["final_path"] += ".changed"
        row["pending_path"] = row["final_path"] + ".pending-v0.8"
        altered = rehash_plan(altered)
        with mock.patch.object(
            gate, "PLAN_PAYLOAD_SHA256", altered["plan_payload_sha256"]
        ):
            with self.assertRaises(gate.Gate12C2OriginalBaselineError):
                gate.validate_frozen_plan(altered)

    def test_state_event_transition_mutations_fail_closed(self) -> None:
        mutations = []
        duplicate = copy.deepcopy(self.plan)
        duplicate["state_model"]["transitions"].append(
            copy.deepcopy(duplicate["state_model"]["transitions"][0])
        )
        mutations.append(duplicate)
        terminal_edge = copy.deepcopy(self.plan)
        terminal_state = terminal_edge["state_model"]["terminal_states"][0]
        event = terminal_edge["state_model"]["events"][0]
        terminal_edge["state_model"]["transitions"].append(
            [terminal_state, event, terminal_state]
        )
        mutations.append(terminal_edge)
        unknown = copy.deepcopy(self.plan)
        unknown["state_model"]["transitions"][0][1] = "undeclared_event"
        mutations.append(unknown)
        for index, value in enumerate(mutations):
            with self.subTest(index=index):
                altered = rehash_plan(value)
                with mock.patch.object(
                    gate, "PLAN_PAYLOAD_SHA256", altered["plan_payload_sha256"]
                ):
                    with self.assertRaises(
                        gate.Gate12C2OriginalBaselineError
                    ):
                        gate.validate_frozen_plan(altered)

    def test_every_terminal_expiry_reject_and_blocked_phase_is_exact(self) -> None:
        selected = [
            phase
            for phase in self.plan["artifact_lifecycle_contract"]["stable_phases"]
            if any(
                marker in phase["phase"]
                for marker in (
                    "terminal",
                    "expired_blocked",
                    "verdict_reject",
                    "not_active_blocked",
                    "failure_complete",
                    "success_complete",
                )
            )
        ]
        self.assertEqual(len(selected), 18)
        for phase in selected:
            with self.subTest(phase=phase["phase"]):
                observed = observations_for_phase(self.plan, phase)
                self.assertEqual(
                    gate.classify_lifecycle_surface(
                        self.plan,
                        observed,
                        temporal_predicate=phase["temporal_predicate"],
                        liveness=(
                            "ACTIVE"
                            if phase["liveness_predicate"]
                            == "ACTIVE_exact_owner"
                            else "DEAD"
                            if phase["liveness_predicate"] == "DEAD_or_UNKNOWN"
                            else "not_applicable"
                        ),
                    ),
                    phase["phase"],
                )

    def test_owner_death_blocks_each_post_claim_active_phase(self) -> None:
        active = [

            phase
            for phase in self.plan["artifact_lifecycle_contract"]["stable_phases"]
            if phase["liveness_predicate"] == "ACTIVE_exact_owner"
        ]
        self.assertEqual(len(active), 4)
        for phase in active:
            observed = observations_for_phase(self.plan, phase)
            with self.subTest(phase=phase["phase"]):
                self.assertEqual(
                    gate.classify_lifecycle_surface(
                        self.plan,
                        observed,
                        temporal_predicate=phase["temporal_predicate"],
                        liveness="DEAD",
                    ),
                    phase["phase"].replace(
                        "owner_active", "owner_not_active_blocked"
                    ),
                )






class FailureMatrixAdversarialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = gate.load_frozen_plan()

    def test_every_failure_row_is_unique_and_exactly_addressable(self) -> None:
        rows = self.plan["failure_matrix"]
        self.assertEqual(len(rows), 92)
        self.assertEqual(
            len({gate.canonical_json_bytes(row) for row in rows}), 92
        )
        for row in rows:
            with self.subTest(
                scope=row["scope"],
                phase=row["failure_phase"],
                code=row["failure_code"],
            ):
                self.assertEqual(
                    gate.matching_failure_row(
                        self.plan,
                        scope=row["scope"],
                        source_state=row["source_state"],
                        failure_phase=row["failure_phase"],
                        failure_code=row["failure_code"],
                    ),
                    row,
                )

    def test_every_noninherit_availability_profile_enforces_nulls(self) -> None:
        field_map = {
            "pre_complete_surface": "pre_complete_surface_sha256",
            "pre_protected_surface": "pre_protected_surface_sha256",
            "post_complete_surface": "post_complete_surface_sha256",
            "post_protected_surface": "post_protected_surface_sha256",
            "baseline_commitment_surface": "baseline_commitment_surface_sha256",
            "recomputed_baseline_commitment_surface": (
                "recomputed_baseline_commitment_surface_sha256"
            ),
        }
        checked = 0
        for scope in ("extraction", "verifier"):
            profiles = self.plan["failure_evidence_availability_profiles"][scope]
            for name, profile in profiles.items():
                if profile.get("inherit_exact_terminal_claim_evidence"):
                    with self.assertRaises(gate.Gate12C2OriginalBaselineError):
                        gate.apply_availability_profile(
                            self.plan,
                            scope=scope,
                            profile_name=name,
                            evidence={},
                        )
                    continue
                checked += 1
                evidence = {
                    output: DIGEST if profile.get(source) else None
                    for source, output in field_map.items()
                }
                selected = gate.apply_availability_profile(
                    self.plan,
                    scope=scope,
                    profile_name=name,
                    evidence=evidence,
                )
                self.assertEqual(
                    selected,
                    {
                        output: DIGEST if available else None
                        for source, available in profile.items()
                        if source in field_map
                        for output in (field_map[source],)
                    },
                )
                for source, available in profile.items():
                    if source not in field_map:
                        continue
                    broken = dict(evidence)
                    broken[field_map[source]] = None if available else DIGEST
                    with self.subTest(scope=scope, profile=name, field=source):
                        with self.assertRaises(
                            gate.Gate12C2OriginalBaselineError
                        ):
                            gate.apply_availability_profile(
                                self.plan,
                                scope=scope,
                                profile_name=name,
                                evidence=broken,
                            )
        self.assertEqual(checked, 8)

    def _failure_fixture(self, row: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        profile = self.plan["failure_evidence_availability_profiles"][
            row["scope"]
        ][row["availability_profile"]]
        field_map = {
            "pre_complete_surface": "pre_complete_surface_sha256",
            "pre_protected_surface": "pre_protected_surface_sha256",
            "post_complete_surface": "post_complete_surface_sha256",
            "post_protected_surface": "post_protected_surface_sha256",
            "baseline_commitment_surface": "baseline_commitment_surface_sha256",
            "recomputed_baseline_commitment_surface": (
                "recomputed_baseline_commitment_surface_sha256"
            ),
        }
        evidence = {
            receipt_field: DIGEST if profile.get(profile_field) is True else None
            for profile_field, receipt_field in field_map.items()
            if profile_field in profile
        }
        progress = {
            "source_state": row["source_state"],
            "failure_phase": row["failure_phase"],
            "evidence": evidence,
            "configuration_count_reached": 0,
            "outer_experiment_count_reached": 0,
            "shard_count_reached": 0,
            "index_count_reached": 0,
        }
        claim = {
            "execution_claim_id": "failure-test",
            "claimed_at_utc": "2026-08-01T00:00:00Z",
            "reviewed_implementation_authority_file_sha256": DIGEST,
            "reviewed_implementation_authority_payload_sha256": DIGEST,
            "preflight_payload_sha256": DIGEST,
            "authorization_payload_sha256": DIGEST,
            "authorization_verdict_payload_sha256": DIGEST,
            "execution_claim_payload_sha256": DIGEST,
        }
        controls = {
            "baseline_file_hash": DIGEST,
            "baseline": {"baseline_receipt_payload_sha256": DIGEST},
            "extraction_terminal_file_hash": DIGEST,
            "extraction_terminal": {"terminal_claim_payload_sha256": DIGEST},
            "preflight_file_hash": DIGEST,
            "authorization_file_hash": DIGEST,
            "verdict_file_hash": DIGEST,
        }
        return progress, claim, controls

    def test_all_60_allowed_failure_rows_build_exact_receipts(self) -> None:
        rows = [
            row
            for row in self.plan["failure_matrix"]
            if row["failure_receipt_allowed"] is True
        ]
        self.assertEqual(len(rows), 60)
        for row in rows:
            progress, claim, controls = self._failure_fixture(row)
            with self.subTest(
                scope=row["scope"],
                phase=row["failure_phase"],
                code=row["failure_code"],
            ):
                if row["scope"] == "extraction":
                    receipt = extraction_runner._failure_leaf(
                        self.plan,
                        claim,
                        failure_code=row["failure_code"],
                        occurred_at_utc="2026-08-01T00:00:00Z",
                        preflight_file_hash=DIGEST,
                        authorization_file_hash=DIGEST,
                        verdict_file_hash=DIGEST,
                        claim_file_hash=DIGEST,
                        progress=progress,
                    )
                    schema = self.plan["extraction_failure_receipt"]
                    gate.verify_self_hash(
                        receipt, "failure_receipt_payload_sha256"
                    )
                else:
                    receipt = independent._verifier_failure(
                        self.plan,
                        controls,
                        claim,
                        claim_file_hash=DIGEST,
                        code=row["failure_code"],
                        occurred_at_utc="2026-08-01T00:00:00Z",
                        progress=progress,
                    )
                    schema = self.plan["verifier_failure_receipt"]
                    independent._self_hash(
                        receipt, "failure_receipt_payload_sha256"
                    )
                self.assertEqual(set(receipt), set(schema["exact_top_level_fields"]))
                self.assertEqual(receipt["failure_code"], row["failure_code"])
                self.assertEqual(receipt["failure_phase"], row["failure_phase"])
                self.assertEqual(receipt["source_state"], row["source_state"])
                self.assertEqual(
                    receipt["evidence_availability"],
                    row["availability_profile"],
                )

    def test_all_32_nonreceipt_failure_rows_cannot_invent_a_leaf(self) -> None:
        rows = [
            row
            for row in self.plan["failure_matrix"]
            if row["failure_receipt_allowed"] is False
        ]
        self.assertEqual(len(rows), 32)
        for row in rows:
            progress, claim, controls = self._failure_fixture(row)
            with self.subTest(
                scope=row["scope"],
                phase=row["failure_phase"],
                code=row["failure_code"],
            ):
                if row["scope"] == "extraction":
                    with self.assertRaises(gate.Gate12C2OriginalBaselineError):
                        extraction_runner._failure_leaf(
                            self.plan,
                            claim,
                            failure_code=row["failure_code"],
                            occurred_at_utc="2026-08-01T00:00:00Z",
                            preflight_file_hash=DIGEST,
                            authorization_file_hash=DIGEST,
                            verdict_file_hash=DIGEST,
                            claim_file_hash=DIGEST,
                            progress=progress,
                        )
                else:
                    with self.assertRaises(
                        independent.IndependentVerificationError
                    ):
                        independent._verifier_failure(
                            self.plan,
                            controls,
                            claim,
                            claim_file_hash=DIGEST,
                            code=row["failure_code"],
                            occurred_at_utc="2026-08-01T00:00:00Z",
                            progress=progress,
                        )

    def test_failure_receipts_reject_preclaim_timestamps(self) -> None:
        rows_by_scope = {
            scope: next(
                row
                for row in self.plan["failure_matrix"]
                if row["scope"] == scope
                and row["failure_receipt_allowed"] is True
            )
            for scope in ("extraction", "verifier")
        }
        for scope, row in rows_by_scope.items():
            progress, claim, controls = self._failure_fixture(row)
            claim["claimed_at_utc"] = "2026-08-01T00:01:00Z"
            with self.subTest(scope=scope):
                if scope == "extraction":
                    with self.assertRaises(
                        gate.Gate12C2OriginalBaselineError
                    ):
                        extraction_runner._failure_leaf(
                            self.plan,
                            claim,
                            failure_code=row["failure_code"],
                            occurred_at_utc="2026-08-01T00:00:00Z",
                            preflight_file_hash=DIGEST,
                            authorization_file_hash=DIGEST,
                            verdict_file_hash=DIGEST,
                            claim_file_hash=DIGEST,
                            progress=progress,
                        )
                else:
                    with self.assertRaises(
                        independent.IndependentVerificationError
                    ):
                        independent._verifier_failure(
                            self.plan,
                            controls,
                            claim,
                            claim_file_hash=DIGEST,
                            code=row["failure_code"],
                            occurred_at_utc="2026-08-01T00:00:00Z",
                            progress=progress,
                        )

    def test_all_codes_and_phases_include_symmetric_lineage_reverification(self) -> None:
        rows = self.plan["failure_matrix"]
        self.assertEqual(
            {row["failure_code"] for row in rows}, gate.FAILURE_CODES
        )
        self.assertEqual(
            {row["failure_phase"] for row in rows},
            set(self.plan["failure_phases"]),
        )
        expected = {
            (
                "extraction",
                "EXTRACTION_EXECUTION_CLAIMED",
                "lineage_reverification",
                code,
            )
            for code in (
                "INPUT_LINEAGE_MISMATCH",
                "IMPLEMENTATION_BYTE_IDENTITY_MISMATCH",
            )
        } | {
            (
                "verifier",
                "VERIFIER_EXECUTION_CLAIMED",
                "verifier_lineage_reverification",
                code,
            )
            for code in (
                "INPUT_LINEAGE_MISMATCH",
                "IMPLEMENTATION_BYTE_IDENTITY_MISMATCH",
            )
        }
        actual = {
            (
                row["scope"],
                row["source_state"],
                row["failure_phase"],
                row["failure_code"],
            )
            for row in rows
        }
        self.assertTrue(expected <= actual)


def candidate_for_temp_repository(
    plan: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    rows = []
    for relative in gate.IMPLEMENTATION_PATHS:
        raw = ("bounded-source:" + relative).encode("ascii")
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        rows.append(
            {
                "role": binding_builder.IMPLEMENTATION_ROLES[relative],
                "relative_path": relative,
                "file_sha256": gate.sha256_bytes(raw),
                "git_blob_oid": gate.git_blob_oid(raw),
            }
        )
    rows.sort(key=lambda row: row["role"])
    scientific = copy.deepcopy(
        plan["implementation_binding_contract"]["scientific_dependencies"]
    )
    for row in scientific:
        source = REPOSITORY / row["relative_path"]
        target = root / row["relative_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    bundle = root / "clean.bundle"
    bundle.write_bytes(b"offline-clean-bundle")
    clean_restore = {
        "bundle_path": str(bundle),
        "bundle_file_sha256": gate.sha256_bytes(bundle.read_bytes()),
        "bundle_size_bytes": bundle.stat().st_size,
        "restore_receipt_file_sha256": "b" * 64,
        "restore_receipt_payload_sha256": "c" * 64,
        "restore_head": SOURCE_COMMIT,
        "restore_worktree_clean": True,
        "git_fsck_full_pass": True,
        "core_autocrlf": False,
        "core_longpaths": True,
        "implementation_rows_match": True,
        "scientific_dependency_rows_match": True,
    }
    payload = {
        "schema_version": (
            "gate12c2_original_baseline_implementation_candidate_binding_v0.8"
        ),
        "binding_id": (
            "C2_ORIGINAL_BASELINE_COMMITMENT_GATE_IMPLEMENTATION_CANDIDATE_BINDING_v0.8"
        ),
        "source_commit": SOURCE_COMMIT,
        "git_object_format": "sha1",
        "core_autocrlf": False,
        "core_longpaths": True,
        "worktree_clean": True,
        "contract_file_sha256": gate.CONTRACT_FILE_SHA256,
        "plan_file_sha256": gate.PLAN_FILE_SHA256,
        "plan_payload_sha256": gate.PLAN_PAYLOAD_SHA256,
        "formal_design_review_file_sha256": (
            gate.FORMAL_DESIGN_REVIEW_FILE_SHA256
        ),
        "formal_design_review_payload_sha256": (
            gate.FORMAL_DESIGN_REVIEW_PAYLOAD_SHA256
        ),
        "implementation_author_separation_contract_sha256": (
            gate.IMPLEMENTATION_AUTHOR_SEPARATION_SHA256
        ),
        "procedural_author_separation_precondition_satisfied": True,
        "implementation_context_blindness_machine_authenticated": False,
        "current_exposed_design_context_authored_final_bytes": False,
        "implementation_trust_model_sha256": (
            gate.IMPLEMENTATION_TRUST_MODEL_SHA256
        ),
        "artifact_path_surface_sha256": gate.ARTIFACT_PATH_SURFACE_SHA256,
        "implementation_files": rows,
        "scientific_dependencies": scientific,
        "clean_restore": clean_restore,
        "task_identity_used_as_machine_authority": False,
        "implementation_authorship_machine_verified": False,
        "protected_payload_access_required_for_implementation": False,
    }
    return gate.add_self_hash(
        payload, "implementation_candidate_binding_payload_sha256"
    )


def pass_review(
    plan: Mapping[str, Any],
    candidate: Mapping[str, Any],
    candidate_file_sha256: str,
) -> dict[str, Any]:
    schema = plan["review_receipt_schemas"][
        "fresh_implementation_review_verdict"
    ]
    review = dict(schema["outcomes"]["pass"]["required_values"])
    restore = candidate["clean_restore"]
    review.update(
        {
            "reviewed_at_utc": "2026-08-01T00:00:00Z",
            "implementation_author_separation_contract_sha256": (
                gate.IMPLEMENTATION_AUTHOR_SEPARATION_SHA256
            ),
            "formal_design_review_file_sha256": (
                gate.FORMAL_DESIGN_REVIEW_FILE_SHA256
            ),
            "formal_design_review_payload_sha256": (
                gate.FORMAL_DESIGN_REVIEW_PAYLOAD_SHA256
            ),
            "implementation_trust_model_sha256": (
                gate.IMPLEMENTATION_TRUST_MODEL_SHA256
            ),
            "implementation_candidate_binding_file_sha256": (
                candidate_file_sha256
            ),
            "implementation_candidate_binding_payload_sha256": candidate[
                "implementation_candidate_binding_payload_sha256"
            ],
            "implementation_source_commit": candidate["source_commit"],
            "implementation_review_packet_file_sha256": "f" * 64,
            "bundle_file_sha256": restore["bundle_file_sha256"],
            "restore_receipt_file_sha256": restore[
                "restore_receipt_file_sha256"
            ],
            "restore_receipt_payload_sha256": restore[
                "restore_receipt_payload_sha256"
            ],
            "P0_count": 0,
            "P1_count": 0,
            "P2_count": 0,
        }
    )
    self_field = "fresh_implementation_review_payload_sha256"
    self_keys = set(review) | {self_field}
    if self_keys != set(schema["exact_top_level_fields"]):
        raise AssertionError("review fixture does not cover exact schema")
    return gate.add_self_hash(review, self_field)


class AuthorityAndAuthorizationAdversarialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = gate.load_frozen_plan()

    def test_control_links_timestamps_and_integer_domains_fail_closed(self) -> None:
        preflight, authorization, verdict, _claim = control_chain(
            self.plan, "verifier"
        )
        now = gate.parse_utc_ns("2026-08-01T00:01:00Z")
        gate.validate_preflight_payload(
            self.plan,
            preflight,
            scope="verifier",
            reviewed_authority_file_sha256="3" * 64,
            reviewed_authority_payload_sha256="4" * 64,
            extraction_terminal_file_sha256=DIGEST,
            extraction_terminal_payload_sha256=OTHER_DIGEST,
            baseline_receipt_file_sha256="1" * 64,
            baseline_receipt_payload_sha256="2" * 64,
            now_ns=now,
        )
        altered = dict(preflight)
        altered["baseline_receipt_file_sha256"] = "0" * 64
        altered.pop("preflight_payload_sha256")
        altered = gate.add_self_hash(altered, "preflight_payload_sha256")
        with self.assertRaises(gate.Gate12C2OriginalBaselineError):
            gate.validate_preflight_payload(
                self.plan,
                altered,
                scope="verifier",
                reviewed_authority_file_sha256="3" * 64,
                reviewed_authority_payload_sha256="4" * 64,
                extraction_terminal_file_sha256=DIGEST,
                extraction_terminal_payload_sha256=OTHER_DIGEST,
                baseline_receipt_file_sha256="1" * 64,
                baseline_receipt_payload_sha256="2" * 64,
                now_ns=now,
            )
        with self.assertRaises(gate.Gate12C2OriginalBaselineError):
            gate.require_fresh_interval(
                "2026-08-01T00:00:00Z",
                "2026-08-01T00:20:00Z",
                now_ns=True,
            )
        with self.assertRaises(gate.Gate12C2OriginalBaselineError):
            gate.build_authorization_payload(
                self.plan,
                preflight,
                scope="verifier",
                preflight_file_sha256="5" * 64,
                authorization_id="out-of-order-auth",
                issued_at_utc="2026-07-31T23:59:00Z",
                expires_at_utc="2026-08-01T00:10:00Z",
                now_ns=gate.parse_utc_ns("2026-08-01T00:03:00Z"),
            )
        with self.assertRaises(gate.Gate12C2OriginalBaselineError):
            gate.build_authorization_verdict_payload(
                self.plan,
                authorization,
                preflight,
                scope="verifier",
                verification_id="out-of-order-verdict",
                verified_at_utc="2026-08-01T00:01:00Z",
                outcome_kind="pass",
                reason_code=None,
                preflight_file_sha256="5" * 64,
                authorization_file_sha256="6" * 64,
                verifier_relative_path=(
                    "tools/verify_gate12c2_original_baseline_authorization.py"
                ),
                verifier_file_sha256="7" * 64,
                verifier_git_blob_oid="8" * 40,
                now_ns=gate.parse_utc_ns("2026-08-01T00:03:00Z"),
            )
        with self.assertRaises(gate.Gate12C2OriginalBaselineError):
            gate.build_execution_claim_payload(
                self.plan,
                authorization,
                preflight,
                verdict,
                scope="verifier",
                execution_claim_id="out-of-order-claim",
                launch_id="launch",
                claimed_at_utc="2026-08-01T00:03:00Z",
                owner_hostname="test-host",
                owner_pid=1234,
                owner_process_creation_time_utc="2026-08-01T00:00:30Z",
                preflight_file_sha256="5" * 64,
                authorization_file_sha256="6" * 64,
                verdict_file_sha256="9" * 64,
                now_ns=gate.parse_utc_ns("2026-08-01T00:05:00Z"),
            )

    def test_candidate_roles_and_review_counts_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = candidate_for_temp_repository(self.plan, root)
            altered = copy.deepcopy(candidate)
            first_role = altered["implementation_files"][0]["role"]
            altered["implementation_files"][0]["role"] = altered[
                "implementation_files"
            ][1]["role"]
            altered["implementation_files"][1]["role"] = first_role
            altered["implementation_files"].sort(key=lambda row: row["role"])
            altered.pop("implementation_candidate_binding_payload_sha256")
            altered = gate.add_self_hash(
                altered, "implementation_candidate_binding_payload_sha256"
            )
            with self.assertRaises(gate.Gate12C2OriginalBaselineError):
                gate.validate_candidate_binding(
                    self.plan,
                    altered,
                    repo_root=root,
                    current_head=SOURCE_COMMIT,
                )
            candidate_file_hash = gate.sha256_bytes(
                gate.canonical_receipt_bytes(candidate)
            )
            review = pass_review(self.plan, candidate, candidate_file_hash)
            review["P2_count"] = -1
            review.pop("fresh_implementation_review_payload_sha256")
            review = gate.add_self_hash(
                review, "fresh_implementation_review_payload_sha256"
            )
            with self.assertRaises(gate.Gate12C2OriginalBaselineError):
                gate.validate_implementation_review(
                    self.plan,
                    review,
                    candidate_file_sha256=candidate_file_hash,
                    candidate_payload_sha256=candidate[
                        "implementation_candidate_binding_payload_sha256"
                    ],
                    source_commit=SOURCE_COMMIT,
                    candidate=candidate,
                )

    def test_source_and_clean_restore_bytes_are_bound_without_protected_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = candidate_for_temp_repository(self.plan, root)
            self.assertEqual(
                gate.validate_candidate_binding(
                    self.plan,
                    candidate,
                    repo_root=root,
                    current_head=SOURCE_COMMIT,
                ),
                candidate,
            )
            self.assertEqual(
                binding_builder.validate_clean_restore(
                    candidate["clean_restore"], source_commit=SOURCE_COMMIT
                ),
                candidate["clean_restore"],
            )
            first = root / gate.IMPLEMENTATION_PATHS[0]
            first.write_bytes(first.read_bytes() + b"changed")
            with self.assertRaises(gate.Gate12C2OriginalBaselineError):
                gate.validate_candidate_binding(
                    self.plan,
                    candidate,
                    repo_root=root,
                    current_head=SOURCE_COMMIT,
                )
            first.write_bytes(
                ("bounded-source:" + gate.IMPLEMENTATION_PATHS[0]).encode(
                    "ascii"
                )
            )
            bundle = Path(candidate["clean_restore"]["bundle_path"])
            bundle.write_bytes(b"changed-bundle")
            with self.assertRaises(gate.Gate12C2OriginalBaselineError):
                gate.validate_candidate_binding(
                    self.plan,
                    candidate,
                    repo_root=root,
                    current_head=SOURCE_COMMIT,
                )
            with self.assertRaises(gate.Gate12C2OriginalBaselineError):
                binding_builder.validate_clean_restore(
                    candidate["clean_restore"], source_commit=SOURCE_COMMIT
                )

    def test_operator_confirmation_and_exposed_authorship_are_blocking(self) -> None:
        for confirmed, exposed in ((False, False), (True, True), (False, True)):
            with self.subTest(confirmed=confirmed, exposed=exposed):
                with self.assertRaises(gate.Gate12C2OriginalBaselineError):
                    binding_builder.build_candidate_binding(
                        REPOSITORY,
                        {},
                        procedural_author_separation_precondition_satisfied=(
                            confirmed
                        ),
                        current_exposed_design_context_authored_final_bytes=(
                            exposed
                        ),
                    )

    def test_deterministic_authority_order_and_mutations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = candidate_for_temp_repository(self.plan, root)
            candidate_file_hash = gate.sha256_bytes(
                gate.canonical_receipt_bytes(candidate)
            )
            review = pass_review(self.plan, candidate, candidate_file_hash)
            gate.validate_implementation_review(
                self.plan,
                review,
                candidate_file_sha256=candidate_file_hash,
                candidate_payload_sha256=candidate[
                    "implementation_candidate_binding_payload_sha256"
                ],
                source_commit=SOURCE_COMMIT,
                candidate=candidate,
            )
            review_file_hash = gate.sha256_bytes(
                gate.canonical_receipt_bytes(review)
            )
            first = gate.build_reviewed_authority_payload(
                self.plan,
                candidate,
                review,
                candidate_file_sha256=candidate_file_hash,
                review_file_sha256=review_file_hash,
            )
            second = gate.build_reviewed_authority_payload(
                self.plan,
                candidate,
                review,
                candidate_file_sha256=candidate_file_hash,
                review_file_sha256=review_file_hash,
            )
            self.assertEqual(
                gate.canonical_receipt_bytes(first),
                gate.canonical_receipt_bytes(second),
            )
            gate.validate_reviewed_authority(
                self.plan,
                first,
                candidate=candidate,
                candidate_file_sha256=candidate_file_hash,
                review=review,
                review_file_sha256=review_file_hash,
            )
            for field in (
                "formal_design_review_payload_sha256",
                "implementation_author_separation_contract_sha256",
                "implementation_trust_model_sha256",
                "artifact_path_surface_sha256",
                "implementation_candidate_binding_payload_sha256",
                "fresh_implementation_review_payload_sha256",
                "implementation_source_commit",
            ):
                altered = dict(first)
                altered[field] = (
                    "0" * 40 if field == "implementation_source_commit" else "0" * 64
                )
                altered = gate.add_self_hash(
                    {
                        key: value
                        for key, value in altered.items()
                        if key
                        != "reviewed_implementation_authority_payload_sha256"
                    },
                    "reviewed_implementation_authority_payload_sha256",
                )
                with self.subTest(field=field):
                    with self.assertRaises(
                        gate.Gate12C2OriginalBaselineError
                    ):
                        gate.validate_reviewed_authority(
                            self.plan,
                            altered,
                            candidate=candidate,
                            candidate_file_sha256=candidate_file_hash,
                            review=review,
                            review_file_sha256=review_file_hash,
                        )

    def test_union_verdict_competition_and_rejection_permanence(self) -> None:
        for scope in ("extraction", "verifier"):
            preflight, authorization, passed, _claim = control_chain(
                self.plan, scope
            )
            rejected = gate.build_authorization_verdict_payload(
                self.plan,
                authorization,
                preflight,
                scope=scope,
                verification_id=f"{scope}-rejected",

                verified_at_utc="2026-08-01T00:04:00Z",
                outcome_kind="reject",
                reason_code="AUTHORIZATION_INVALID",
                preflight_file_sha256="5" * 64,
                authorization_file_sha256="6" * 64,
                verifier_relative_path=(
                    "tools/verify_gate12c2_original_baseline_authorization.py"
                ),
                verifier_file_sha256="7" * 64,
                verifier_git_blob_oid="8" * 40,
                now_ns=gate.parse_utc_ns("2026-08-01T00:04:00Z"),
            )
            self.assertEqual(passed["outcome_kind"], "pass")
            self.assertEqual(rejected["outcome_kind"], "reject")
            stale = gate.build_authorization_verdict_payload(
                self.plan,
                authorization,
                preflight,
                scope=scope,
                verification_id=f"{scope}-stale",
                verified_at_utc="2026-08-01T00:20:00Z",
                outcome_kind="reject",
                reason_code="AUTHORIZATION_STALE",
                preflight_file_sha256="5" * 64,
                authorization_file_sha256="6" * 64,
                verifier_relative_path=(
                    "tools/verify_gate12c2_original_baseline_authorization.py"
                ),
                verifier_file_sha256="7" * 64,
                verifier_git_blob_oid="8" * 40,
                now_ns=gate.parse_utc_ns("2026-08-01T00:20:00Z"),
            )
            self.assertEqual(stale["outcome_kind"], "reject")
            self.assertEqual(stale["reason_code"], "AUTHORIZATION_STALE")
            self.assertLess(stale["remaining_freshness_nanoseconds"], 0)
            self.assertNotEqual(
                gate.canonical_receipt_bytes(passed),
                gate.canonical_receipt_bytes(rejected),
            )
            phase_name = f"{scope}_authorization_verdict_reject"
            phase = next(
                item
                for item in self.plan["artifact_lifecycle_contract"][
                    "stable_phases"
                ]
                if item["phase"] == phase_name
            )
            observed = observations_for_phase(self.plan, phase)
            self.assertEqual(
                gate.classify_lifecycle_surface(self.plan, observed), phase_name
            )
            observed[f"{scope}_execution_claim"] = gate.ArtifactObservation(True)
            self.assertEqual(
                gate.classify_lifecycle_surface(self.plan, observed),
                "HOLD_new_review",
            )
        preflight, authorization, _passed, _claim = control_chain(
            self.plan, "extraction"
        )
        for outcome, reason in (
            ("pass", "AUTHORIZATION_INVALID"),
            ("reject", None),
        ):
            with self.subTest(outcome=outcome, reason=reason):
                with self.assertRaises(gate.Gate12C2OriginalBaselineError):
                    gate.build_authorization_verdict_payload(
                        self.plan,
                        authorization,
                        preflight,
                        scope="extraction",
                        verification_id="bad-union",
                        verified_at_utc="2026-08-01T00:04:00Z",
                        outcome_kind=outcome,
                        reason_code=reason,
                        preflight_file_sha256="5" * 64,
                        authorization_file_sha256="6" * 64,
                        verifier_relative_path="tools/verifier.py",
                        verifier_file_sha256="7" * 64,
                        verifier_git_blob_oid="8" * 40,
                    )


    def test_authorization_verifier_returns_pass_reject_and_stale_union(self) -> None:
        preflight, authorization, _verdict, _claim = control_chain(
            self.plan, "extraction"
        )
        source = Path(authorization_verifier.__file__).read_bytes()
        candidate = {
            "git_object_format": "sha1",
            "implementation_files": [
                {
                    "relative_path": authorization_verifier.RELATIVE_PATH,
                    "file_sha256": gate.sha256_bytes(source),
                    "git_blob_oid": gate.git_blob_oid(source),
                }
            ],
        }
        authority = {
            "reviewed_implementation_authority_payload_sha256": "4" * 64
        }

        def read_receipt(
            _path: Path, *, hash_field: str, **_kwargs: Any
        ) -> tuple[dict[str, Any], str]:
            if hash_field == (
                "implementation_candidate_binding_payload_sha256"
            ):
                return candidate, "0" * 64
            if hash_field == "preflight_payload_sha256":
                return preflight, "5" * 64
            if hash_field == "authorization_payload_sha256":
                return authorization, "6" * 64
            raise AssertionError(hash_field)

        scenarios = (
            (
                "pass",
                gate.parse_utc_ns("2026-08-01T00:04:00Z"),
                None,
                "pass",
                None,
            ),
            (
                "lineage-reject",
                gate.parse_utc_ns("2026-08-01T00:04:00Z"),
                gate.Gate12C2OriginalBaselineError(
                    "INPUT_LINEAGE_MISMATCH"
                ),
                "reject",
                "INPUT_LINEAGE_MISMATCH",
            ),
            (
                "stale-reject",
                gate.parse_utc_ns("2026-08-01T00:19:00Z"),
                None,
                "reject",
                "AUTHORIZATION_STALE",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            verifier_path = repository / authorization_verifier.RELATIVE_PATH
            verifier_path.parent.mkdir(parents=True)
            verifier_path.write_bytes(source)
            for name, current_ns, lineage_error, outcome, reason in scenarios:
                with self.subTest(name=name):
                    with contextlib.ExitStack() as stack:
                        stack.enter_context(
                            mock.patch.object(
                                authorization_verifier.gate,
                                "load_frozen_plan",
                                return_value=self.plan,
                            )
                        )
                        stack.enter_context(
                            mock.patch.object(
                                authorization_verifier.gate,
                                "read_schema_receipt",
                                side_effect=read_receipt,
                            )
                        )
                        stack.enter_context(
                            mock.patch.object(
                                authorization_verifier.gate,
                                "read_exact_bytes",
                                return_value=b"contract",
                            )
                        )
                        stack.enter_context(
                            mock.patch.object(
                                authorization_verifier.gate,
                                "validate_formal_design_pass",
                            )
                        )
                        stack.enter_context(
                            mock.patch.object(
                                authorization_verifier.gate,
                                "validate_upstream_authority",
                                side_effect=lineage_error,
                            )
                        )
                        stack.enter_context(
                            mock.patch.object(
                                authorization_verifier.gate,
                                "validate_original_input_lineage",
                            )
                        )
                        stack.enter_context(
                            mock.patch.object(
                                authorization_verifier.preflight_issuer,
                                "load_reviewed_chain",
                                return_value=(authority, "3" * 64, candidate),
                            )
                        )
                        stack.enter_context(
                            mock.patch.object(
                                authorization_verifier.preflight_issuer,
                                "_observe_surface",
                                return_value={},
                            )
                        )
                        stack.enter_context(
                            mock.patch.object(
                                authorization_verifier.gate,
                                "classify_lifecycle_surface",
                                return_value=(
                                    "extraction_authorization_issued_"
                                    "unverified_fresh"
                                ),
                            )
                        )
                        result = authorization_verifier.verify_authorization(
                            repository,
                            scope="extraction",
                            verification_id=f"test-{name}",
                            verified_at_utc=(
                                "2026-08-01T00:19:00Z"
                                if name == "stale-reject"
                                else "2026-08-01T00:04:00Z"
                            ),
                            now_ns=current_ns,
                        )
                    self.assertEqual(result["outcome_kind"], outcome)
                    self.assertEqual(result["reason_code"], reason)
                    gate.verify_self_hash(
                        result, "authorization_verdict_payload_sha256"
                    )


    def test_extra_fields_and_changed_predecessors_reject_authority(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = candidate_for_temp_repository(self.plan, root)
            candidate_file_hash = gate.sha256_bytes(
                gate.canonical_receipt_bytes(candidate)
            )
            review = pass_review(self.plan, candidate, candidate_file_hash)
            review_file_hash = gate.sha256_bytes(
                gate.canonical_receipt_bytes(review)
            )
            authority = gate.build_reviewed_authority_payload(
                self.plan,
                candidate,
                review,
                candidate_file_sha256=candidate_file_hash,
                review_file_sha256=review_file_hash,
            )

            injected_candidate = dict(candidate)
            injected_candidate["issuer_id"] = "untrusted"
            injected_candidate = gate.add_self_hash(
                {
                    key: value
                    for key, value in injected_candidate.items()
                    if key
                    != "implementation_candidate_binding_payload_sha256"
                },
                "implementation_candidate_binding_payload_sha256",
            )
            with self.assertRaises(gate.Gate12C2OriginalBaselineError):
                gate.validate_candidate_binding(
                    self.plan,
                    injected_candidate,
                    repo_root=root,
                    current_head=SOURCE_COMMIT,
                )

            injected_review = dict(review)
            injected_review["reviewer_id"] = "untrusted"
            injected_review = gate.add_self_hash(
                {
                    key: value
                    for key, value in injected_review.items()
                    if key != "fresh_implementation_review_payload_sha256"
                },
                "fresh_implementation_review_payload_sha256",
            )
            with self.assertRaises(gate.Gate12C2OriginalBaselineError):
                gate.validate_implementation_review(
                    self.plan,
                    injected_review,
                    candidate_file_sha256=candidate_file_hash,
                    candidate_payload_sha256=candidate[
                        "implementation_candidate_binding_payload_sha256"
                    ],
                    source_commit=SOURCE_COMMIT,
                    candidate=candidate,
                )

            injected_authority = dict(authority)
            injected_authority["issued_at_utc"] = "2026-08-01T00:00:00Z"
            injected_authority = gate.add_self_hash(
                {
                    key: value
                    for key, value in injected_authority.items()
                    if key
                    != "reviewed_implementation_authority_payload_sha256"
                },
                "reviewed_implementation_authority_payload_sha256",
            )
            with self.assertRaises(gate.Gate12C2OriginalBaselineError):
                gate.validate_reviewed_authority(
                    self.plan,
                    injected_authority,
                    candidate=candidate,
                    candidate_file_sha256=candidate_file_hash,
                    review=review,
                    review_file_sha256=review_file_hash,
                )

            changed_candidate = dict(candidate)
            changed_candidate[
                "implementation_candidate_binding_payload_sha256"
            ] = "0" * 64
            with self.assertRaises(gate.Gate12C2OriginalBaselineError):
                gate.validate_reviewed_authority(
                    self.plan,
                    authority,
                    candidate=changed_candidate,
                    candidate_file_sha256=candidate_file_hash,
                    review=review,
                    review_file_sha256=review_file_hash,
                )
            changed_review = dict(review)
            changed_review["fresh_implementation_review_payload_sha256"] = (
                "0" * 64
            )
            with self.assertRaises(gate.Gate12C2OriginalBaselineError):
                gate.validate_reviewed_authority(
                    self.plan,
                    authority,
                    candidate=candidate,
                    candidate_file_sha256=candidate_file_hash,
                    review=changed_review,
                    review_file_sha256=review_file_hash,
                )

    def test_arbitrary_process_rebuild_is_byte_identical_and_issuer_independent(self) -> None:
        candidate = {
            "source_commit": SOURCE_COMMIT,
            "implementation_candidate_binding_payload_sha256": DIGEST,
            "issuer_id": "ignored-by-derivation",
        }
        review = {
            "fresh_implementation_review_payload_sha256": OTHER_DIGEST,
            "reviewer_id": "ignored-by-derivation",
        }
        expected = gate.canonical_receipt_bytes(
            gate.build_reviewed_authority_payload(
                self.plan,
                candidate,
                review,
                candidate_file_sha256="1" * 64,
                review_file_sha256="2" * 64,
            )
        )
        program = (
            "import json,sys;"
            "sys.path.insert(0,sys.argv[1]);"
            "import gate12c2_original_baseline_commitments as g;"
            "x=json.load(sys.stdin);p=g.load_frozen_plan();"
            "a=g.build_reviewed_authority_payload(p,x['candidate'],x['review'],"
            "candidate_file_sha256=x['candidate_file'],"
            "review_file_sha256=x['review_file']);"
            "sys.stdout.buffer.write(g.canonical_receipt_bytes(a))"
        )
        request = json.dumps(
            {
                "candidate": candidate,
                "review": review,
                "candidate_file": "1" * 64,
                "review_file": "2" * 64,
            }
        ).encode("utf-8")
        outputs = []
        for _ in range(2):
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    "-c",
                    program,
                    str(REPOSITORY / "tools"),
                ],
                input=request,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stderr, b"")
            outputs.append(completed.stdout)
        self.assertEqual(outputs, [expected, expected])

    def test_provenance_and_capability_audit_fields_are_absent(self) -> None:
        binding_fields = set(
            self.plan["implementation_binding_contract"][
                "exact_top_level_fields"
            ]
        )
        review_fields = set(
            self.plan["review_receipt_schemas"][
                "fresh_implementation_review_verdict"
            ]["exact_top_level_fields"]
        )
        authority_fields = set(
            self.plan["reviewed_implementation_authority_contract"][
                "exact_top_level_fields"
            ]
        )
        identity_fragments = (
            "task_id",
            "context_id",
            "origin",
            "fork",
            "history",
            "reviewer_id",
            "issuer_id",
            "owner_pid",
            "hostname",
            "username",
        )
        for fields in (binding_fields, review_fields, authority_fields):
            self.assertFalse(
                any(
                    fragment in field
                    for field in fields
                    if field
                    not in {
                        "task_identity_used_as_machine_authority",
                        "authority_issuer_identity_required",
                    }
                    for fragment in identity_fragments
                )
            )
        self.assertNotIn("issued_at_utc", authority_fields)
        self.assertNotIn("reviewed_at_utc", authority_fields)
        active_schemas = {
            "binding": self.plan["implementation_binding_contract"],
            "review": self.plan["review_receipt_schemas"][
                "fresh_implementation_review_verdict"
            ],
            "authority": self.plan[
                "reviewed_implementation_authority_contract"
            ],
            "controls": self.plan["control_receipt_schemas"],
            "success": self.plan["success_receipt"],
            "extraction_failure": self.plan["extraction_failure_receipt"],
            "verification": self.plan["verification_receipt"],
            "verifier_failure": self.plan["verifier_failure_receipt"],
        }
        for name, value in active_schemas.items():
            with self.subTest(schema=name):
                self.assertNotIn(
                    "capability_audit",
                    gate.canonical_json_bytes(value).decode("utf-8").lower(),
                )


def terminal_leaf(
    plan: Mapping[str, Any], scope: str, outcome: str
) -> dict[str, Any]:
    schema = (
        plan["success_receipt"]
        if scope == "extraction" and outcome == "success"
        else plan["verification_receipt"]
        if scope == "verifier" and outcome == "success"
        else plan["extraction_failure_receipt"]
        if scope == "extraction"
        else plan["verifier_failure_receipt"]
    )
    hash_field = (
        "baseline_receipt_payload_sha256"
        if scope == "extraction" and outcome == "success"
        else "verification_receipt_payload_sha256"
        if scope == "verifier" and outcome == "success"
        else "failure_receipt_payload_sha256"
    )
    payload = {
        field: None
        for field in schema["exact_top_level_fields"]
        if field != hash_field
    }
    payload["schema_version"] = schema["schema_version"]
    return gate.add_self_hash(payload, hash_field)


class TerminalAndPublicationAdversarialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = gate.load_frozen_plan()

    def _terminal(
        self,
        scope: str,
        leaf: Mapping[str, Any],
        outcome: str,
        claim: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        _preflight, _authorization, _verdict, built_claim = control_chain(
            self.plan, scope
        )
        selected_claim = built_claim if claim is None else claim
        return gate.build_terminal_payload(
            self.plan,
            selected_claim,
            leaf,
            scope=scope,
            outcome_kind=outcome,
            claimed_at_utc="2026-08-01T00:06:00Z",
            reviewed_authority_file_sha256="3" * 64,
            preflight_file_sha256="5" * 64,
            authorization_file_sha256="6" * 64,
            verdict_file_sha256="9" * 64,
            execution_claim_file_sha256="a" * 64,
        )

    def test_success_and_failure_compete_for_one_terminal_with_exact_leaf(self) -> None:
        for scope in ("extraction", "verifier"):
            success_leaf = terminal_leaf(self.plan, scope, "success")
            failure_leaf = terminal_leaf(self.plan, scope, "failure")
            success = self._terminal(scope, success_leaf, "success")
            failure = self._terminal(scope, failure_leaf, "failure")
            self.assertEqual(success["leaf_exact_payload"], success_leaf)
            self.assertEqual(failure["leaf_exact_payload"], failure_leaf)
            self.assertNotEqual(
                success["terminal_claim_payload_sha256"],
                failure["terminal_claim_payload_sha256"],
            )
            self.assertNotIn("terminal_claim_payload_sha256", success_leaf)

    def test_opposite_modified_and_orphan_leaves_are_rejected(self) -> None:
        success_leaf = terminal_leaf(self.plan, "extraction", "success")
        failure_leaf = terminal_leaf(self.plan, "extraction", "failure")
        with self.assertRaises(gate.Gate12C2OriginalBaselineError):
            self._terminal("extraction", failure_leaf, "success")
        modified = dict(success_leaf)
        modified["unexpected"] = "altered"
        with self.assertRaises(gate.Gate12C2OriginalBaselineError):
            self._terminal("extraction", modified, "success")
        with self.assertRaises(gate.Gate12C2OriginalBaselineError):
            self._terminal("extraction", success_leaf, "success", claim={})

    def test_two_process_atomic_claim_race_has_one_winner(self) -> None:
        program = (
            "import sys;from pathlib import Path;"
            "sys.path.insert(0,sys.argv[1]);"
            "import gate12c2_original_baseline_commitments as g;"
            "p=Path(sys.argv[2]);b=sys.argv[3].encode();"
            "\ntry:\n g.atomic_publish_exact(p,b);print('PUBLISHED')"
            "\nexcept g.Gate12C2OriginalBaselineError as e:"
            "\n print('ERROR:'+e.code);raise SystemExit(3)"
        )
        with tempfile.TemporaryDirectory() as directory:
            final = Path(directory) / "claim.json"
            commands = [
                [
                    sys.executable,
                    "-I",
                    "-B",
                    "-c",
                    program,
                    str(REPOSITORY / "tools"),
                    str(final),
                    payload,
                ]
                for payload in ("claim-A", "claim-B")
            ]
            processes = [
                subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                for command in commands
            ]
            results = [process.communicate(timeout=30) for process in processes]
            codes = sorted(process.returncode for process in processes)
            self.assertEqual(codes, [0, 3], results)
            self.assertIn(final.read_bytes(), {b"claim-A", b"claim-B"})
            self.assertFalse(
                final.with_name(final.name + ".pending-v0.8").exists()
            )

    def test_pid_reuse_owner_death_and_foreign_host_fail_closed(self) -> None:
        claim = {
            "owner_hostname": "host-A",
            "owner_pid": 4321,
            "owner_process_creation_time_utc": "2026-08-01T00:00:00Z",
        }
        self.assertEqual(
            gate.classify_claim_owner(
                claim,
                hostname="host-A",
                creation_query=lambda _pid: "2026-08-01T00:00:01Z",
            ),
            "DEAD",
        )
        self.assertEqual(
            gate.classify_claim_owner(
                claim,
                hostname="host-A",
                creation_query=lambda _pid: (_ for _ in ()).throw(
                    ProcessLookupError()
                ),
            ),
            "DEAD",
        )
        self.assertEqual(
            gate.classify_claim_owner(claim, hostname="host-B"), "UNKNOWN"
        )



def mutate_first_result(
    subplan: Mapping[str, Any],
    index: bytes,
    shards: Mapping[str, bytes],
    mutation: Any,
) -> tuple[bytes, dict[str, bytes]]:
    changed_index = copy.deepcopy(
        gate.require_mapping(gate.strict_json_loads(index, canonical=True))
    )
    changed_shards = dict(shards)
    row = changed_index["shards"][0]
    relative = row["relative_path"]
    shard = gate.strict_gzip_json(changed_shards[relative])
    result = shard["result"]
    mutation(result)
    shard["result_payload_sha256"] = gate.sha256_bytes(
        gate.canonical_json_bytes(result)
    )
    shard = gate.add_self_hash(
        {
            key: value
            for key, value in shard.items()
            if key != "shard_payload_sha256"
        },
        "shard_payload_sha256",
    )
    compressed = gzip.compress(
        gate.canonical_json_bytes(shard), compresslevel=6, mtime=0
    )
    changed_shards[relative] = compressed
    row.update(
        {
            "compressed_file_sha256": gate.sha256_bytes(compressed),
            "compressed_bytes": len(compressed),
            "shard_payload_sha256": shard["shard_payload_sha256"],
            "result_payload_sha256": shard["result_payload_sha256"],
            "decision": gate.reconstruct_decision(result),
        }
    )
    changed_index["scientific_projection_sha256"] = gate.sha256_bytes(
        gate.canonical_json_bytes(
            gate.scientific_projection(subplan, changed_index["shards"])
        )
    )
    changed_index = gate.add_self_hash(
        {
            key: value
            for key, value in changed_index.items()
            if key != "index_payload_sha256"
        },
        "index_payload_sha256",
    )
    return gate.canonical_json_bytes(changed_index), changed_shards


class IndependentVerifierAdversarialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = gate.load_frozen_plan()

    def test_independent_control_chain_is_exact_and_chronological(self) -> None:
        preflight, authorization, verdict, claim = control_chain(
            self.plan, "verifier"
        )
        authority = {
            "reviewed_implementation_authority_payload_sha256": "4" * 64
        }
        verifier_row = {
            "relative_path": (
                "tools/verify_gate12c2_original_baseline_authorization.py"
            ),
            "file_sha256": "7" * 64,
            "git_blob_oid": "8" * 40,
        }
        links = {
            "extraction_terminal_claim_file_sha256": DIGEST,
            "extraction_terminal_claim_payload_sha256": OTHER_DIGEST,
            "baseline_receipt_file_sha256": "1" * 64,
            "baseline_receipt_payload_sha256": "2" * 64,
        }
        independent._validate_control_preflight(
            self.plan,
            preflight,
            scope="verifier",
            authority=authority,
            authority_file_hash="3" * 64,
            now_ns=gate.parse_utc_ns("2026-08-01T00:01:00Z"),
            linked_receipts=links,
        )
        independent._validate_control_authorization(
            self.plan,
            authorization,
            preflight,
            scope="verifier",
            preflight_file_hash="5" * 64,
            now_ns=gate.parse_utc_ns("2026-08-01T00:03:00Z"),
        )
        independent._validate_control_verdict(
            self.plan,
            verdict,
            authorization,
            preflight,
            scope="verifier",
            preflight_file_hash="5" * 64,
            authorization_file_hash="6" * 64,
            verifier_row=verifier_row,
            now_ns=gate.parse_utc_ns("2026-08-01T00:04:00Z"),
        )
        independent._validate_control_claim(
            self.plan,
            claim,
            authorization,
            preflight,
            verdict,
            scope="verifier",
            preflight_file_hash="5" * 64,
            authorization_file_hash="6" * 64,
            verdict_file_hash="9" * 64,
        )
        controls = {
            "baseline_file_hash": "1" * 64,
            "baseline": {"baseline_receipt_payload_sha256": "2" * 64},
            "extraction_terminal_file_hash": DIGEST,
            "extraction_terminal": {
                "terminal_claim_payload_sha256": OTHER_DIGEST
            },
            "preflight_file_hash": "5" * 64,
            "preflight": preflight,
            "authorization_file_hash": "6" * 64,
            "authorization": authorization,
            "verdict_file_hash": "9" * 64,
            "verdict": verdict,
        }
        rederived = {
            "baseline_commitment_surface_sha256": "a" * 64,
            "pre_complete_surface_sha256": "b" * 64,
            "post_complete_surface_sha256": "c" * 64,
            "pre_protected_surface_sha256": "d" * 64,
            "post_protected_surface_sha256": "e" * 64,
        }
        leaf = independent.independent_verification_receipt(
            self.plan,
            rederived,
            authority_file_sha256="3" * 64,
            authority_payload_sha256="4" * 64,
            baseline_file_sha256="1" * 64,
            baseline_payload_sha256="2" * 64,
            extraction_terminal_file_sha256=DIGEST,
            extraction_terminal_payload_sha256=OTHER_DIGEST,
            preflight_file_sha256="5" * 64,
            preflight_payload_sha256=preflight["preflight_payload_sha256"],
            authorization_file_sha256="6" * 64,
            authorization_payload_sha256=authorization[
                "authorization_payload_sha256"
            ],
            verdict_file_sha256="9" * 64,
            verdict_payload_sha256=verdict[
                "authorization_verdict_payload_sha256"
            ],
            claim_file_sha256="a" * 64,
            claim_payload_sha256=claim["execution_claim_payload_sha256"],
        )
        terminal = independent._verifier_terminal(
            self.plan,
            controls,
            claim,
            claim_file_hash="a" * 64,
            leaf=leaf,
            outcome="success",
            claimed_at_utc="2026-08-01T00:06:00Z",
        )
        self.assertEqual(terminal["outcome_kind"], "success")
        independent._self_hash(terminal, "terminal_claim_payload_sha256")
        with self.assertRaises(independent.IndependentVerificationError):
            independent._verifier_terminal(
                self.plan,
                controls,
                claim,
                claim_file_hash="a" * 64,
                leaf=leaf,
                outcome="unknown",
                claimed_at_utc="2026-08-01T00:06:00Z",
            )
        altered = dict(verdict)
        altered["sequence_ordinal"] = 126
        with self.assertRaises(independent.IndependentVerificationError):
            independent._validate_control_verdict(
                self.plan,
                altered,
                authorization,
                preflight,
                scope="verifier",
                preflight_file_hash="5" * 64,
                authorization_file_hash="6" * 64,
                verifier_row=verifier_row,
                now_ns=gate.parse_utc_ns("2026-08-01T00:04:00Z"),
            )

    def test_unknown_result_and_pipeline_fields_fail_both_paths(self) -> None:
        for mutation in (
            lambda result: result.__setitem__("unknown_result_field", 1),
            lambda result: result["pipeline_decision"].__setitem__(
                "unknown_pipeline_field", False
            ),
            lambda result: result["execution_configuration_contract"].__setitem__(
                "caller_supplied_authority", True
            ),
            lambda result: result["numerical_execution_contract"].__setitem__(
                "blas_thread_limit", 2
            ),
            lambda result: result["numerical_execution_contract"].__setitem__(
                "unknown_numerical_field", False
            ),
        ):
            subplan, index, shards = primary.synthetic_configuration()
            changed_index, changed_shards = mutate_first_result(
                subplan, index, shards, mutation
            )
            with self.subTest(mutation=repr(mutation)):
                with self.assertRaises(gate.Gate12C2OriginalBaselineError):
                    gate.derive_configuration_commitment(
                        configuration_id="synthetic_configuration",
                        subplan=subplan,
                        index_raw=changed_index,
                        shard_raw_by_relative_path=changed_shards,
                        result_validator=lambda *_args: None,
                    )
                with self.assertRaises(independent.IndependentVerificationError):
                    independent.independent_configuration_commitment(
                        configuration_id="synthetic_configuration",
                        subplan=subplan,
                        index_raw=changed_index,
                        shard_raw_by_relative_path=changed_shards,
                    )

    def test_verifier_has_no_shared_or_scientific_imports(self) -> None:
        verifier_source = Path(independent.__file__).read_text(encoding="utf-8")
        tree = ast.parse(verifier_source)
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        forbidden = {
            "gate12c2_original_baseline_commitments",
            "gate12c2_development_shards",
            "gate12c2_synthetic_lab",
        }
        self.assertTrue(forbidden.isdisjoint(imported))
        core_source = Path(gate.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import gate12c2_development_shards", core_source)
        self.assertNotIn("import gate12c2_synthetic_lab", core_source)
        extraction_source = inspect.getsource(gate.extract_commitments_after_claim)
        for filename in (
            "gate12c2_development_shards.py",
            "gate12c2_synthetic_lab.py",
        ):
            self.assertIn(filename, extraction_source)
        self.assertLess(
            extraction_source.index("sha256_bytes(raw) != expected_sha256"),
            extraction_source.index("spec.loader.exec_module(module)"),
        )
        self.assertNotIn("subprocess", core_source)
        self.assertNotIn("subprocess", verifier_source)
        runner_source = Path(extraction_runner.__file__).read_text(encoding="utf-8")
        self.assertNotIn("subprocess", runner_source)

    def test_retained_handle_implementations_encode_all_frozen_controls(self) -> None:
        self.assertEqual(gate.EXPECTED_FILE_COUNT, 791)
        self.assertEqual(gate.EXPECTED_DIRECTORY_COUNT, 23)
        self.assertEqual(
            self.plan["anti_toctou"]["total_retained_input_handle_count"],
            815,
        )
        sources = (
            inspect.getsource(gate.RetainedProtectedSurface),
            inspect.getsource(independent.VerifierRetainedSurface),
        )
        required_tokens = (
            "CreateFileW",
            "GetFileInformationByHandleEx",
            "GetFinalPathNameByHandleW",
            "ReadFile",
            "0x00200000",
            "0x08000000",
        )
        for source in sources:
            with self.subTest(source=gate.sha256_bytes(source.encode())):
                for token in required_tokens:
                    self.assertIn(token, source)
                self.assertNotIn("write_bytes", source)
                self.assertNotIn("read_bytes", source)
        self.assertIn("FILE_SHARE_READ", sources[0])
        self.assertIn("str(path), desired, 0x1", sources[1])
        self.assertIn("len(self.files) != EXPECTED_FILE_COUNT", sources[0])
        self.assertIn("len(self.file_records) != 791", sources[1])

    def test_digest_only_outputs_and_cli_failures_do_not_leak_values(self) -> None:
        extracted, verified = primary.derive_both(primary.synthetic_configuration())
        for result in (extracted, verified):
            value = gate.canonical_json_bytes(result).decode("utf-8")
            self.assertNotIn("scientific_values", value)
            self.assertNotIn("case_receipts", value)
            self.assertNotIn("endpoint_receipts", value)
        cases = (
            (
                extraction_runner,
                gate.EXTRACTION_ERROR_PREFIX,
                gate.Gate12C2OriginalBaselineError("INPUT_SCHEMA_INVALID"),
            ),
            (
                authorization_verifier,
                gate.AUTHORIZATION_ERROR_PREFIX,
                RuntimeError("protected-secret"),
            ),
            (
                independent,
                independent.ERROR_PREFIX,
                RuntimeError("protected-secret"),
            ),
        )
        for module, prefix, error in cases:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with self.subTest(module=module.__name__):
                with mock.patch.object(module, "main", side_effect=error):
                    with contextlib.redirect_stdout(stdout):
                        with contextlib.redirect_stderr(stderr):
                            self.assertEqual(module.cli([]), 2)
                self.assertEqual(stdout.getvalue(), "")
                self.assertTrue(stderr.getvalue().startswith(prefix))
                self.assertNotIn("protected-secret", stderr.getvalue())
                self.assertNotIn("Traceback", stderr.getvalue())

    def test_common_mode_wrong_digest_is_rejected_by_independent_comparison(self) -> None:
        extracted, verified = primary.derive_both(primary.synthetic_configuration())
        wrong = copy.deepcopy(extracted)
        wrong["scientific_projection_sha256"] = "0" * 64
        self.assertNotEqual(wrong, verified)
        self.assertNotEqual(
            wrong["scientific_projection_sha256"],
            verified["scientific_projection_sha256"],
        )


class LiteralLineageAndSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = gate.load_frozen_plan()

    def test_literal_upstream_paths_schemas_and_hash_domains_are_closed(self) -> None:
        rows = self.plan["upstream_authority"]["artifact_rows"]
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows, sorted(rows, key=lambda row: row["role"]))
        self.assertEqual(len({row["path"] for row in rows}), 4)
        for row in rows:
            self.assertTrue(Path(row["path"]).is_absolute())
            self.assertTrue(gate.is_sha256(row["file_sha256"]))
            if row["format"] == "canonical_self_hashed_JSON":
                self.assertTrue(gate.is_sha256(row["payload_sha256"]))
                self.assertTrue(row["self_hash_field"].endswith("sha256"))
                self.assertTrue(row["schema_version"])
            else:
                self.assertIsNone(row["payload_sha256"])
                self.assertIsNone(row["self_hash_field"])
                self.assertIsNone(row["schema_version"])
        lineage = self.plan["original_input_lineage"]
        for key, value in lineage.items():
            if key.endswith("_path"):
                self.assertTrue(Path(value).is_absolute())
                self.assertIn(key[:-5] + "_file_sha256", lineage)
            elif key.endswith("_file_sha256") or key.endswith(
                "_payload_sha256"
            ):
                self.assertTrue(gate.is_sha256(value))
        frozen_dependencies = {
            row["relative_path"]: row["file_sha256"]
            for row in self.plan["implementation_binding_contract"][
                "scientific_dependencies"
            ]
        }
        self.assertEqual(
            lineage["original_development_shards_file_sha256"],
            frozen_dependencies["tools/gate12c2_development_shards.py"],
        )
        self.assertEqual(
            lineage["original_synthetic_lab_file_sha256"],
            frozen_dependencies["tools/gate12c2_synthetic_lab.py"],
        )

    def test_all_candidate_review_authority_fields_are_closed_and_noncyclic(self) -> None:
        candidate = set(
            self.plan["implementation_binding_contract"][
                "exact_top_level_fields"
            ]
        )
        review = set(
            self.plan["review_receipt_schemas"][
                "fresh_implementation_review_verdict"
            ]["exact_top_level_fields"]
        )
        authority_contract = self.plan[
            "reviewed_implementation_authority_contract"
        ]
        authority = set(authority_contract["exact_top_level_fields"])
        self.assertIn(
            "implementation_candidate_binding_payload_sha256", candidate
        )
        self.assertIn(
            "implementation_candidate_binding_payload_sha256", review
        )
        self.assertIn(
            "fresh_implementation_review_payload_sha256", authority
        )
        self.assertNotIn(
            "reviewed_implementation_authority_payload_sha256", candidate
        )
        self.assertNotIn(
            "reviewed_implementation_authority_payload_sha256", review
        )
        self.assertEqual(
            authority_contract["non_circular_construction_order"],
            [
                "candidate_binding",
                "fresh_implementation_review_verdict_pass",
                "deterministic_reviewed_implementation_authority",
            ],
        )

    def test_procedural_separation_is_never_cryptographic_task_authority(self) -> None:
        binding_required = self.plan["implementation_binding_contract"][
            "required_values"
        ]
        authority_required = self.plan[
            "reviewed_implementation_authority_contract"
        ]["required_values"]
        self.assertIs(
            binding_required["task_identity_used_as_machine_authority"], False
        )
        self.assertIs(
            binding_required["implementation_authorship_machine_verified"],
            False,
        )
        self.assertIs(
            binding_required[
                "implementation_context_blindness_machine_authenticated"
            ],
            False,
        )
        self.assertIs(
            authority_required["task_identity_used_as_machine_authority"],
            False,
        )
        self.assertIs(
            authority_required["authority_issuer_identity_required"], False
        )


if __name__ == "__main__":
    unittest.main()
