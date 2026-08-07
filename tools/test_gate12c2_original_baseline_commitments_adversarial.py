#!/usr/bin/env python3
"""Adversarial tests for the Gate12C-2 v0.9 baseline gate."""

from __future__ import annotations

import ast
import contextlib
import copy
import ctypes
import gzip
import inspect
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from unittest import mock

import build_gate12c2_original_baseline_implementation_binding as binding_builder
import gate12c2_original_baseline_commitments as gate
import issue_gate12c2_original_baseline_preflight as preflight_issuer
import test_gate12c2_original_baseline_commitments as primary
import run_gate12c2_original_baseline_extraction as extraction_runner
import verify_gate12c2_original_baseline_commitments as independent
import verify_gate12c2_original_baseline_authorization as authorization_verifier


REPOSITORY = Path(__file__).resolve().parents[1]
DIGEST = "d" * 64
OTHER_DIGEST = "e" * 64
SOURCE_COMMIT = "a" * 40

V011_REVIEW_PLAN = Path(
    "C:/Users/aoika/Documents/Research/pale-ale-local/research-program/"
    "profile-plans/C2_ORIGINAL_BASELINE_COMMITMENT_GATE_PLAN_v0.11_2026-08-02.json"
)
V011_REVIEW_PLAN_FILE_SHA256 = (
    "8a36d78e36bf20a162a6903c5968c15838f1df4da20451aca35638731c24049c"
)


def rehash_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    payload.pop("plan_payload_sha256", None)
    return gate.add_self_hash(payload, "plan_payload_sha256")


def independently_rederive_review_surface() -> dict[str, object]:
    raw = V011_REVIEW_PLAN.read_bytes()
    if gate.sha256_bytes(raw) != V011_REVIEW_PLAN_FILE_SHA256:
        raise AssertionError("v0.11 review-plan byte identity mismatch")
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise AssertionError("v0.11 review-plan LF domain mismatch")
    source = gate.strict_json_loads(raw[:-1], canonical=True)
    gate.verify_self_hash(source, "plan_payload_sha256")
    compatibility = []
    catalogs = source["source_bound_leaf_schema_contract"][
        "canonical_leaf_surface"
    ]["catalogs"]
    for catalog in catalogs:
        for record in catalog["records"]:
            if record["field_path"].startswith(
                ("component_medians.", "component_coverage.")
            ):
                continue
            compatibility.append(
                {"schema_id": catalog["catalog_id"], **record}
            )
    schema_id = "input.source_bound_s2_endpoint_rows"
    for arm in gate.S2_COMPONENT_ARMS:
        for field in gate.S2_COMPONENT_FIELDS:
            compatibility.append(
                {
                    "schema_id": schema_id,
                    "field_path": f"component_medians.{arm}.{field}",
                    "json_type": "null_or_number",
                    "presence": "required",
                }
            )
            for count_field in gate.S2_COMPONENT_COUNT_FIELDS:
                compatibility.append(
                    {
                        "schema_id": schema_id,
                        "field_path": (
                            f"component_coverage.{arm}.{field}."
                            f"{count_field}"
                        ),
                        "json_type": "integer",
                        "presence": "required",
                    }
                )
    compatibility.sort(
        key=lambda row: (row["schema_id"], row["field_path"])
    )
    normative = []
    for row in source["normative_schema_contract"]["rows"]:
        field = row["field_name"]
        if field.startswith(("component_medians.", "component_coverage.")):
            prefix, remainder = field.split(".", 1)
            for arm in gate.S2_COMPONENT_ARMS:
                copied = copy.deepcopy(row)
                copied["field_name"] = f"{prefix}.{arm}.{remainder}"
                copied["row_id"] = (
                    f"{copied['schema_id']}::{copied['field_name']}"
                )
                normative.append(copied)
        else:
            normative.append(copy.deepcopy(row))
    normative.sort(key=lambda row: row["row_id"])
    dimensions = source["mutation_coverage_contract"][
        "mutation_dimensions"
    ]
    if tuple(dimensions) != gate.REVIEW_SURFACE_MUTATION_DIMENSIONS:
        raise AssertionError("mutation dimension order mismatch")
    applicability = []
    required = []
    for row in normative:
        for dimension in dimensions:
            is_required = bool(
                row["mutation_dimension_applicability"][dimension]
            )
            applicability.append(
                {
                    "row_id": row["row_id"],
                    "dimension": dimension,
                    "required": is_required,
                }
            )
            if is_required:
                required.append(
                    {"row_id": row["row_id"], "dimension": dimension}
                )
    return {
        "compatibility": compatibility,
        "normative": normative,
        "applicability": applicability,
        "required": required,
    }


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
            outcome=(
                "success"
                if required.get(role) == "success_or_failure"
                else required.get(role)
            ),
        )
        for role in plan["artifact_lifecycle_contract"]["roles"]
    }


def synthetic_lifecycle_plan(
    plan: Mapping[str, Any], phase_name: str, root: Path
) -> dict[str, Any]:
    copied = copy.deepcopy(dict(plan))
    phase = next(
        row
        for row in copied["artifact_lifecycle_contract"]["stable_phases"]
        if row["phase"] == phase_name
    )
    required_outcomes = phase["required_outcomes"]
    root.mkdir(parents=True, exist_ok=True)
    for row in copied["artifact_path_surface"]:
        role = row["role"]
        final = root / f"{role}.json"
        pending = root / f"{role}.json.pending-v0.9"
        row["final_path"] = str(final)
        row["pending_path"] = str(pending)
    order = (
        "formal_design_review_verdict",
        "implementation_candidate_binding",
        "fresh_implementation_review_verdict",
        "reviewed_implementation_authority",
        "extraction_preflight",
        "extraction_authorization",
        "extraction_authorization_verdict",
        "extraction_execution_claim",
        "extraction_success",
        "extraction_failure",
        "extraction_terminal",
        "verifier_preflight",
        "verifier_authorization",
        "verifier_authorization_verdict",
        "verifier_execution_claim",
        "verifier_success",
        "verifier_failure",
        "verifier_terminal",
    )
    payloads: dict[str, dict[str, Any]] = {}
    outcome_fields = copied["artifact_lifecycle_contract"][
        "outcome_field_by_role"
    ]
    for role in order:
        schema, hash_field = gate._artifact_schema_descriptor(copied, role)
        payload = {
            field: None
            for field in schema["exact_top_level_fields"]
            if field != hash_field
        }
        if role in gate.REVIEW_SURFACE_BOUND_ROLES:
            payload["review_surface_identity"] = (
                gate.review_surface_identity(copied)
            )
        if role in required_outcomes:
            payload[outcome_fields[role]] = required_outcomes[role]
        elif role in outcome_fields:
            payload[outcome_fields[role]] = (
                "success"
                if role in {"extraction_terminal", "verifier_terminal"}
                else "pass"
            )
        for field in tuple(payload):
            suffix = "_file_sha256"
            if not field.endswith(suffix):
                continue
            prefix = field[: -len(suffix)]
            target = gate._artifact_link_target(role, prefix)
            if target is None:
                continue
            target_payload = payloads[target]
            _target_schema, target_hash_field = (
                gate._artifact_schema_descriptor(copied, target)
            )
            payload[field] = gate.sha256_bytes(
                gate.canonical_receipt_bytes(target_payload)
            )
            payload[prefix + "_payload_sha256"] = target_payload[
                target_hash_field
            ]
        if role in {"extraction_terminal", "verifier_terminal"}:
            scope = (
                "verifier" if role == "verifier_terminal" else "extraction"
            )
            outcome = payload["outcome_kind"]
            if outcome not in {"success", "failure"}:
                outcome = "success"
                payload["outcome_kind"] = outcome
            target = f"{scope}_{outcome}"
            target_payload = payloads[target]
            _target_schema, target_hash_field = (
                gate._artifact_schema_descriptor(copied, target)
            )
            payload["leaf_schema_version"] = target_payload[
                "schema_version"
            ]
            payload["leaf_payload_sha256"] = target_payload[
                target_hash_field
            ]
            payload["leaf_exact_payload"] = copy.deepcopy(target_payload)
        payload = gate.add_self_hash(payload, hash_field)
        payloads[role] = payload
    rows = {
        row["role"]: row for row in copied["artifact_path_surface"]
    }
    for role in phase["must_exist"]:
        Path(rows[role]["final_path"]).write_bytes(
            gate.canonical_receipt_bytes(payloads[role])
        )
    return copied


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
        implementation_source_commit=SOURCE_COMMIT,
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
        git_head_at_claim=SOURCE_COMMIT,
        executing_code_identity_surface_sha256="a" * 64,
        preflight_file_sha256="5" * 64,
        authorization_file_sha256="6" * 64,
        verdict_file_sha256="9" * 64,
        now_ns=gate.parse_utc_ns("2026-08-01T00:05:00Z"),
    )
    return preflight, authorization, verdict, claim


class FrozenDesignMutationTests(unittest.TestCase):
    def test_windows_ordinal_edges_and_exact_call_shape(self) -> None:
        api = ctypes.WinDLL("kernel32", use_last_error=True)
        cases = (
            (r"C:\PALE-ALE\TOOLS", r"c:\pale-ale\tools", True),
            ("C:\\pale-ale\\stra\u00dfe", r"C:\pale-ale\STRASSE", False),
            ("C:\\pale-ale\\\ufb00", r"C:\pale-ale\ff", False),
            ("C:\\pale-ale\\\u0130", "C:\\pale-ale\\i\u0307", False),
        )
        for left, right, expected in cases:
            with self.subTest(left=left, right=right):
                self.assertEqual(left.casefold(), right.casefold())
                self.assertIs(gate.windows_ordinal_equal(left, right, api), expected)
                self.assertIs(
                    independent.independent_windows_ordinal_equal(
                        left, right, api
                    ),
                    expected,
                )

        class Compare:
            def __init__(self) -> None:
                self.calls: list[tuple[object, ...]] = []

            def __call__(self, *args: object) -> int:
                self.calls.append(args)
                return 2

        class Api:
            def __init__(self) -> None:
                self.CompareStringOrdinal = Compare()

        for function in (
            gate.windows_ordinal_equal,
            independent.independent_windows_ordinal_equal,
        ):
            fake = Api()
            self.assertTrue(function("A", "a", fake))
            self.assertEqual(fake.CompareStringOrdinal.calls, [("A", -1, "a", -1, True)])
        with self.assertRaises(gate.Gate12C2OriginalBaselineError):
            gate.windows_ordinal_equal("A", "a", object())
        with self.assertRaises(independent.IndependentVerificationError):
            independent.independent_windows_ordinal_equal("A", "a", object())

    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = gate.load_frozen_plan()

    def test_all_12_final_and_15_failure_checkpoints_classify_18_by_18(self) -> None:
        frozen = self.plan["artifact_lifecycle_contract"][
            "full_surface_checkpoint_contract"
        ]
        checkpoint_rows = list(frozen["checkpoint_rows"])
        failure_rows = list(
            frozen["failure_publication_checkpoint_contract"][
                "failure_checkpoint_rows"
            ]
        )
        self.assertEqual((len(checkpoint_rows), len(failure_rows)), (12, 15))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for ordinal, row in enumerate(checkpoint_rows + failure_rows):
                phase = next(
                    item
                    for item in self.plan["artifact_lifecycle_contract"][
                        "stable_phases"
                    ]
                    if item["phase"] == row["expected_artifact_phase"]
                )
                plan = synthetic_lifecycle_plan(
                    self.plan, phase["phase"], root / str(ordinal)
                )
                temporal = phase["temporal_predicate"]
                liveness = (
                    "ACTIVE"
                    if phase["liveness_predicate"] == "ACTIVE_exact_owner"
                    else "not_applicable"
                )
                function = (
                    gate.require_full_surface_checkpoint
                    if row["scope"] == "extraction"
                    else independent.independent_require_full_surface_checkpoint
                )
                with self.subTest(
                    scope=row["scope"],
                    checkpoint=row["checkpoint"],
                    state=row["expected_state"],
                ):
                    self.assertEqual(
                        function(
                            plan,
                            scope=row["scope"],
                            checkpoint=row["checkpoint"],
                            state=row["expected_state"],
                            temporal_predicate=temporal,
                            liveness=liveness,
                        ),
                        row["expected_artifact_phase"],
                    )

    def test_cross_link_tamper_blocks_both_production_observers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan = synthetic_lifecycle_plan(
                self.plan,
                "extraction_execution_claimed_owner_active",
                Path(temporary),
            )
            rows = {
                row["role"]: row for row in plan["artifact_path_surface"]
            }
            claim_path = Path(
                rows["extraction_execution_claim"]["final_path"]
            )
            payload = gate.require_mapping(
                gate.strict_json_loads(
                    claim_path.read_bytes()[:-1], canonical=True
                )
            )
            _schema, hash_field = gate._artifact_schema_descriptor(
                plan, "extraction_execution_claim"
            )
            payload.pop(hash_field)
            payload["preflight_file_sha256"] = "f" * 64
            payload = gate.add_self_hash(payload, hash_field)
            claim_path.write_bytes(gate.canonical_receipt_bytes(payload))

            core = gate.observe_artifact_surface(plan)
            standalone = independent.independent_observe_artifact_surface(plan)
            self.assertFalse(
                core["extraction_execution_claim"].final_valid
            )
            self.assertIs(
                standalone["extraction_execution_claim"]["final_valid"],
                False,
            )
            self.assertEqual(
                gate.classify_lifecycle_surface(
                    plan, core, liveness="ACTIVE"
                ),
                "HOLD_new_review",
            )
            self.assertEqual(
                independent.independent_classify_lifecycle_surface(
                    plan, standalone, liveness="ACTIVE"
                ),
                "HOLD_new_review",
            )

    def test_all_18_pending_injections_block_both_production_clis(self) -> None:
        roles = tuple(self.plan["artifact_lifecycle_contract"]["roles"])
        self.assertEqual(len(roles), 18)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            extraction_plan = synthetic_lifecycle_plan(
                self.plan,
                "extraction_authorization_verdict_pass_fresh",
                root / "extraction",
            )
            verifier_plan = synthetic_lifecycle_plan(
                self.plan,
                "verifier_authorization_verdict_pass_fresh",
                root / "verifier",
            )
            for scope, plan in (
                ("extraction", extraction_plan),
                ("verifier", verifier_plan),
            ):
                rows = {row["role"]: row for row in plan["artifact_path_surface"]}
                checked = 0
                for role in roles:
                    pending = Path(rows[role]["pending_path"])
                    pending.write_bytes(b"synthetic-pending-only")
                    stdout = io.StringIO()
                    stderr = io.StringIO()
                    if scope == "extraction":
                        with contextlib.ExitStack() as stack:
                            stack.enter_context(
                                mock.patch.object(
                                    extraction_runner, "_runtime_isolated"
                                )
                            )
                            stack.enter_context(
                                mock.patch.object(
                                    extraction_runner.gate,
                                    "load_active_plan",
                                    return_value=plan,
                                )
                            )
                            for name in (
                                "read_exact_bytes",
                                "validate_formal_design_pass",
                                "validate_upstream_authority",
                            ):
                                stack.enter_context(
                                    mock.patch.object(extraction_runner.gate, name)
                                )
                            stack.enter_context(
                                mock.patch.object(
                                    extraction_runner.gate,
                                    "validate_original_input_lineage",
                                    return_value={},
                                )
                            )
                            stack.enter_context(
                                mock.patch.object(
                                    extraction_runner,
                                    "_load_controls",
                                    return_value=(
                                        {}, DIGEST, {}, DIGEST,
                                        {}, DIGEST, {}, DIGEST, {},
                                    ),
                                )
                            )
                            publish = stack.enter_context(
                                mock.patch.object(
                                    extraction_runner.gate, "publish_role"
                                )
                            )
                            protected = stack.enter_context(
                                mock.patch.object(
                                    extraction_runner.gate,
                                    "extract_commitments_after_claim",
                                )
                            )
                            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                                result = extraction_runner.cli(
                                    [
                                        "--repository", str(root),
                                        "--execution-claim-id", "synthetic",
                                        "--launch-id", "synthetic",
                                        "--claimed-at-utc", "2026-08-01T00:00:00Z",
                                    ]
                                )
                    else:
                        with contextlib.ExitStack() as stack:
                            stack.enter_context(
                                mock.patch.object(
                                    independent.sys,
                                    "flags",
                                    mock.Mock(
                                        isolated=True,
                                        dont_write_bytecode=True,
                                    ),
                                )
                            )
                            stack.enter_context(
                                mock.patch.object(
                                    independent.sys,
                                    "dont_write_bytecode",
                                    True,
                                )
                            )
                            stack.enter_context(
                                mock.patch.dict(
                                    independent.os.environ,
                                    {"PYTHONPATH": ""},
                                )
                            )
                            stack.enter_context(
                                mock.patch.object(
                                    independent,
                                    "independent_load_plan",
                                    return_value=plan,
                                )
                            )
                            stack.enter_context(
                                mock.patch.object(
                                    independent,
                                    "independent_lineage",
                                    return_value=({}, {}),
                                )
                            )
                            stack.enter_context(
                                mock.patch.object(
                                    independent,
                                    "independent_runtime_lineage",
                                    return_value=({}, DIGEST, {}),
                                )
                            )
                            stack.enter_context(
                                mock.patch.object(
                                    independent,
                                    "_load_verifier_controls",
                                    return_value={},
                                )
                            )
                            publish = stack.enter_context(
                                mock.patch.object(independent, "_publish")
                            )
                            protected = stack.enter_context(
                                mock.patch.object(
                                    independent, "independent_rederive"
                                )
                            )
                            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                                result = independent.cli(
                                    [
                                        "--repository", str(root),
                                        "--execution-claim-id", "synthetic",
                                        "--launch-id", "synthetic",
                                        "--claimed-at-utc", "2026-08-01T00:00:00Z",
                                    ]
                                )
                    with self.subTest(scope=scope, role=role):
                        self.assertEqual(result, 2)
                        self.assertEqual(stdout.getvalue(), "")
                        self.assertIn("UNEXPECTED_ARTIFACT", stderr.getvalue())
                        publish.assert_not_called()
                        protected.assert_not_called()
                    pending.unlink()
                    checked += 1
                self.assertEqual(checked, 18)

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
        row["pending_path"] = row["final_path"] + ".pending-v0.9"
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
        self.assertEqual(len(rows), 94)
        self.assertEqual(
            len({gate.canonical_json_bytes(row) for row in rows}), 94
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
            "implementation_source_commit": "1" * 40,
            "git_head_at_claim": "1" * 40,
            "executing_code_identity_surface_sha256": DIGEST,
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

    def test_all_62_allowed_failure_rows_build_exact_receipts(self) -> None:
        rows = [
            row
            for row in self.plan["failure_matrix"]
            if row["failure_receipt_allowed"] is True
        ]
        self.assertEqual(len(rows), 62)
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
            "gate12c2_original_baseline_implementation_candidate_binding_v0.9"
        ),
        "binding_id": (
            "C2_ORIGINAL_BASELINE_COMMITMENT_GATE_IMPLEMENTATION_CANDIDATE_BINDING_v0.9"
        ),
        "source_commit": SOURCE_COMMIT,
        "authorized_implementation_repository": str(
            gate.AUTHORIZED_IMPLEMENTATION_REPOSITORY
        ),
        "remediation_base_commit": gate.REMEDIATION_BASE_COMMIT,
        "remediation_base_parent": gate.REMEDIATION_BASE_PARENT,
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
            gate.recompute_implementation_trust_model_sha256(plan)
        ),
        "artifact_path_surface_sha256": gate.ARTIFACT_PATH_SURFACE_SHA256,
        "review_surface_identity": gate.review_surface_identity(plan),
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


def synthetic_identity_candidate(
    plan: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    implementation_rows = []
    for row in plan["executing_code_identity_contract"][
        "loaded_gate_module_allowlist"
    ]:
        relative = row["relative_path"]
        raw = (
            "SYNTHETIC_IDENTITY_ROLE = " + repr(row["role"]) + "\n"
        ).encode("utf-8")
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        implementation_rows.append(
            {
                "role": row["role"],
                "relative_path": relative,
                "file_sha256": gate.sha256_bytes(raw),
                "git_blob_oid": gate.git_blob_oid(raw),
            }
        )
    scientific_rows = []
    scientific_sources = {
        "tools/gate12c2_synthetic_lab.py": (
            b'SYNTHETIC_IDENTITY_MARKER = "lab"\n'
        ),
        "tools/gate12c2_development_shards.py": (
            b"import gate12c2_synthetic_lab as lab\n"
            b"SYNTHETIC_IDENTITY_MARKER = lab.SYNTHETIC_IDENTITY_MARKER\n"
        ),
    }
    for frozen in plan["implementation_binding_contract"][
        "scientific_dependencies"
    ]:
        relative = frozen["relative_path"]
        raw = scientific_sources[relative]
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        scientific_rows.append(
            {
                "role": frozen["role"],
                "relative_path": relative,
                "file_sha256": gate.sha256_bytes(raw),
                "git_blob_oid": gate.git_blob_oid(raw),
                "source_commit": SOURCE_COMMIT,
            }
        )
    return {
        "source_commit": SOURCE_COMMIT,
        "git_object_format": "sha1",
        "authorized_implementation_repository": str(
            gate.AUTHORIZED_IMPLEMENTATION_REPOSITORY
        ),
        "implementation_files": implementation_rows,
        "scientific_dependencies": scientific_rows,
    }


def synthetic_loaded_module(name: str, path: Path) -> types.ModuleType:
    raw = path.read_bytes()
    module = types.ModuleType(name)
    loader = types.SimpleNamespace(origin=str(path), synthetic=True)
    module.__file__ = str(path)
    module.__spec__ = types.SimpleNamespace(
        origin=str(path), loader=loader
    )
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


class ExecutingCodeIdentityAdversarialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = gate.load_frozen_plan()

    def _identity(
        self,
        family: str,
        root: Path,
        *,
        repository_argument: Path | None = None,
        authorized_root: Path | None = None,
        head_values: Sequence[str] = (SOURCE_COMMIT,) * 3,
    ) -> tuple[object, dict[str, object]]:
        candidate = synthetic_identity_candidate(self.plan, root)
        values = iter(head_values)

        def read_head(_root: Path) -> str:
            return next(values)

        if family == "extractor":
            entry_path = (
                root / "tools/run_gate12c2_original_baseline_extraction.py"
            )
            core_path = (
                root / "tools/gate12c2_original_baseline_commitments.py"
            )
            entry = synthetic_loaded_module("__main__", entry_path)
            core = synthetic_loaded_module(
                "gate12c2_original_baseline_commitments", core_path
            )
            registry: dict[str, object] = {
                "__main__": entry,
                "gate12c2_original_baseline_commitments": core,
            }
            identity = gate.ExecutingCodeIdentity(
                self.plan,
                candidate,
                entry_path=entry_path,
                repository_argument=(
                    root
                    if repository_argument is None
                    else repository_argument
                ),
                loaded_modules=dict(registry),
                module_registry=registry,
                authorized_root=(
                    root if authorized_root is None else authorized_root
                ),
                git_head_reader=read_head,
            )
            return identity, registry
        entry_path = (
            root / "tools/verify_gate12c2_original_baseline_commitments.py"
        )
        entry = synthetic_loaded_module("__main__", entry_path)
        registry = {"__main__": entry}
        identity = independent.IndependentExecutingCodeIdentity(
            self.plan,
            candidate,
            entry_path=entry_path,
            repository_argument=(
                root if repository_argument is None else repository_argument
            ),
            entry_module=entry,
            module_registry=registry,
            authorized_root=(
                root if authorized_root is None else authorized_root
            ),
            git_head_reader=read_head,
        )
        return identity, registry

    def test_frozen_identity_attack_corpus_on_both_production_paths(self) -> None:
        expected = set(
            self.plan["executing_code_identity_contract"][
                "substitution_tests"
            ]
        )
        for family in ("extractor", "verifier"):
            error = (
                gate.Gate12C2OriginalBaselineError
                if family == "extractor"
                else independent.IndependentVerificationError
            )
            covered: set[str] = set()

            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                with self.assertRaises(error):
                    self._identity(
                        family,
                        root,
                        repository_argument=root / "reviewed-checkout-R",
                    )
                covered.add(
                    "checkout_S_execution_with_authorized_checkout_R_argument_rejected"
                )

            with tempfile.TemporaryDirectory() as temporary:
                identity, _registry = self._identity(
                    family,
                    Path(temporary),
                    head_values=("b" * 40,),
                )
                try:
                    with self.assertRaises(error):
                        identity.checkpoint(identity.checkpoints[0])
                finally:
                    identity.close()
                covered.add("stale_compatible_checkout_rejected")

            with tempfile.TemporaryDirectory() as temporary:
                identity, _registry = self._identity(
                    family, Path(temporary)
                )
                try:
                    if family == "extractor":
                        record = identity.root_record
                        assert record is not None
                        metadata = record.identity
                        record.identity = gate.HandleIdentity(
                            metadata.volume_serial,
                            metadata.file_id,
                            str(Path(temporary) / "moved"),
                            metadata.size,
                        )
                    else:
                        record = identity.root_record
                        assert record is not None
                        metadata = record[1]
                        identity.root_record = (
                            record[0],
                            (
                                metadata[0],
                                metadata[1],
                                str(Path(temporary) / "moved"),
                                metadata[3],
                            ),
                        )
                    with self.assertRaises(error):
                        identity.checkpoint(identity.checkpoints[0])
                finally:
                    identity.close()
                covered.add("moved_or_reparse_root_rejected")

            with tempfile.TemporaryDirectory() as temporary:
                identity, registry = self._identity(
                    family, Path(temporary)
                )
                try:
                    registry["__main__"] = types.ModuleType("__main__")
                    with self.assertRaises(error):
                        identity.checkpoint(identity.checkpoints[0])
                finally:
                    identity.close()
                covered.add("loaded_module_substitution_rejected")

            with tempfile.TemporaryDirectory() as temporary:
                identity, registry = self._identity(
                    family, Path(temporary)
                )
                try:
                    registry["synthetic_alias"] = registry["__main__"]
                    with self.assertRaises(error):
                        identity.checkpoint(identity.checkpoints[0])
                finally:
                    identity.close()
                covered.add("duplicate_or_alias_import_rejected")

            with tempfile.TemporaryDirectory() as temporary:
                identity, _registry = self._identity(
                    family, Path(temporary)
                )
                method_name = (
                    "_read_all" if family == "extractor" else "_read"
                )
                original_read = getattr(identity.io, method_name)

                def mutated_read(*args: object, **kwargs: object) -> bytes:
                    return original_read(*args, **kwargs) + b"#mutation"

                try:
                    with mock.patch.object(
                        identity.io,
                        method_name,
                        side_effect=mutated_read,
                    ), self.assertRaises(error):
                        identity.checkpoint(identity.checkpoints[0])
                finally:
                    identity.close()
                covered.add("loaded_source_byte_mutation_rejected")

            drift_cases = (
                (
                    "git_HEAD_drift_before_claim_rejected",
                    ("b" * 40,),
                    0,
                ),
                (
                    "git_HEAD_drift_before_protected_read_rejected",
                    (SOURCE_COMMIT, "b" * 40),
                    1,
                ),
                (
                    "git_HEAD_drift_before_terminal_publication_rejected",
                    (SOURCE_COMMIT, SOURCE_COMMIT, "b" * 40),
                    2,
                ),
            )
            for attack_id, heads, failing_index in drift_cases:
                with tempfile.TemporaryDirectory() as temporary:
                    identity, _registry = self._identity(
                        family, Path(temporary), head_values=heads
                    )
                    try:
                        for index in range(failing_index):
                            identity.checkpoint(identity.checkpoints[index])
                        with self.assertRaises(error):
                            identity.checkpoint(
                                identity.checkpoints[failing_index]
                            )
                    finally:
                        identity.close()
                covered.add(attack_id)

            with self.subTest(family=family):
                self.assertEqual(covered, expected)

    def test_production_entries_bootstrap_before_unauthorized_checkout_use(
        self,
    ) -> None:
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        cases = (
            (
                REPOSITORY
                / "tools/run_gate12c2_original_baseline_extraction.py",
                "gate12c2-original-baseline:ERROR:"
                "IMPLEMENTATION_BYTE_IDENTITY_MISMATCH\n",
            ),
            (
                REPOSITORY
                / "tools/verify_gate12c2_original_baseline_commitments.py",
                "gate12c2-original-baseline-verification:ERROR:"
                "IMPLEMENTATION_BYTE_IDENTITY_MISMATCH\n",
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            unauthorized_root = Path(temporary) / "checkout"
            unauthorized_tools = unauthorized_root / "tools"
            unauthorized_tools.mkdir(parents=True)
            for source, expected_stderr in cases:
                entry = unauthorized_tools / source.name
                entry.write_bytes(source.read_bytes())
                completed = subprocess.run(
                    [sys.executable, "-I", "-B", str(entry), "--help"],
                    cwd=unauthorized_root,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    timeout=30,
                    check=False,
                )
                with self.subTest(entry=entry.name):
                    self.assertEqual(completed.returncode, 2)
                    self.assertEqual(completed.stdout, "")
                    self.assertEqual(completed.stderr, expected_stderr)
                    self.assertNotIn("Traceback", completed.stderr)

    def test_scientific_dependencies_load_from_retained_synthetic_bytes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = synthetic_identity_candidate(self.plan, root)
            entry_path = (
                root / "tools/run_gate12c2_original_baseline_extraction.py"
            )
            core_path = (
                root / "tools/gate12c2_original_baseline_commitments.py"
            )
            entry = synthetic_loaded_module("__main__", entry_path)
            core = synthetic_loaded_module(
                "gate12c2_original_baseline_commitments", core_path
            )
            names = (
                "__main__",
                "gate12c2_original_baseline_commitments",
                "gate12c2_synthetic_lab",
                "gate12c2_development_shards",
            )
            missing = object()
            saved = {name: sys.modules.get(name, missing) for name in names}
            identity: gate.ExecutingCodeIdentity | None = None
            try:
                sys.modules["__main__"] = entry
                sys.modules["gate12c2_original_baseline_commitments"] = core
                sys.modules.pop("gate12c2_synthetic_lab", None)
                sys.modules.pop("gate12c2_development_shards", None)
                identity = gate.ExecutingCodeIdentity(
                    self.plan,
                    candidate,
                    entry_path=entry_path,
                    repository_argument=root,
                    loaded_modules={
                        "__main__": entry,
                        "gate12c2_original_baseline_commitments": core,
                    },
                    module_registry=sys.modules,
                    authorized_root=root,
                    git_head_reader=lambda _root: SOURCE_COMMIT,
                )
                identity.load_scientific_dependencies()
                lab = identity.module("gate12c2_synthetic_lab")
                shards = identity.module("gate12c2_development_shards")
                self.assertIs(shards.lab, lab)
                result = identity.checkpoint(identity.checkpoints[0])
                self.assertEqual(result["git_head"], SOURCE_COMMIT)
                self.assertEqual(len(identity.sources), 4)
            finally:
                if identity is not None:
                    identity.close()
                for name, module in saved.items():
                    if module is missing:
                        sys.modules.pop(name, None)
                    else:
                        sys.modules[name] = module


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
                gate.recompute_implementation_trust_model_sha256(plan)
            ),
            "review_surface_identity": candidate["review_surface_identity"],
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
    if self_keys != set(
        gate.artifact_exact_fields(plan, "fresh_implementation_review_verdict")
    ):
        raise AssertionError("review fixture does not cover exact schema")
    return gate.add_self_hash(review, self_field)


class ReviewSurfaceAndTrustModelAdversarialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = gate.load_frozen_plan()

    def test_complete_review_surface_is_independently_rederived(self) -> None:
        surfaces = independently_rederive_review_surface()
        expected = gate.review_surface_identity(self.plan)
        independent_expected = independent._review_surface_identity(
            self.plan
        )
        self.assertEqual(independent_expected, expected)
        checks = (
            (
                "compatibility",
                662,
                "compatibility_surface_sha256",
            ),
            ("normative", 841, "normative_surface_sha256"),
            (
                "applicability",
                13456,
                "mutation_applicability_surface_sha256",
            ),
            (
                "required",
                6487,
                "required_mutation_surface_sha256",
            ),
        )
        for name, count, digest_field in checks:
            rows = surfaces[name]
            with self.subTest(surface=name):
                self.assertEqual(len(rows), count)
                self.assertEqual(
                    gate.sha256_bytes(gate.canonical_json_bytes(rows)),
                    expected[digest_field],
                )
        compatibility = surfaces["compatibility"]
        normative = surfaces["normative"]
        self.assertEqual(
            compatibility,
            sorted(
                compatibility,
                key=lambda row: (row["schema_id"], row["field_path"]),
            ),
        )
        self.assertEqual(
            normative,
            sorted(normative, key=lambda row: row["row_id"]),
        )
        self.assertEqual(
            expected["review_surface_identity_sha256"],
            gate.REVIEW_SURFACE_IDENTITY_SHA256,
        )

    def test_boolean_only_or_incomplete_review_cannot_reach_authority(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = candidate_for_temp_repository(self.plan, root)
            candidate_file_hash = gate.sha256_bytes(
                gate.canonical_receipt_bytes(candidate)
            )
            review = pass_review(
                self.plan, candidate, candidate_file_hash
            )
            old_boolean_only = dict(review)
            old_boolean_only.pop(
                "fresh_implementation_review_payload_sha256"
            )
            old_boolean_only.pop("review_surface_identity")
            old_boolean_only = gate.add_self_hash(
                old_boolean_only,
                "fresh_implementation_review_payload_sha256",
            )
            with self.assertRaises(
                gate.Gate12C2OriginalBaselineError
            ):
                gate.validate_implementation_review(
                    self.plan,
                    old_boolean_only,
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
            authority = gate.build_reviewed_authority_payload(
                self.plan,
                candidate,
                review,
                candidate_file_sha256=candidate_file_hash,
                review_file_sha256=review_file_hash,
            )
            self.assertEqual(
                candidate["review_surface_identity"],
                review["review_surface_identity"],
            )
            self.assertEqual(
                review["review_surface_identity"],
                authority["review_surface_identity"],
            )
            altered_surface = copy.deepcopy(
                candidate["review_surface_identity"]
            )
            altered_surface["compatibility_row_count"] = 144
            altered_surface.pop("review_surface_identity_sha256")
            altered_surface[
                "review_surface_identity_sha256"
            ] = gate.sha256_bytes(
                gate.canonical_json_bytes(altered_surface)
            )
            altered_candidate = dict(candidate)
            altered_candidate["review_surface_identity"] = altered_surface
            altered_candidate.pop(
                "implementation_candidate_binding_payload_sha256"
            )
            altered_candidate = gate.add_self_hash(
                altered_candidate,
                "implementation_candidate_binding_payload_sha256",
            )
            with self.assertRaises(
                gate.Gate12C2OriginalBaselineError
            ):
                gate.validate_candidate_binding(
                    self.plan,
                    altered_candidate,
                    repo_root=root,
                    current_head=SOURCE_COMMIT,
                )
            altered_review = dict(review)
            altered_review["review_surface_identity"] = altered_surface
            altered_review.pop(
                "fresh_implementation_review_payload_sha256"
            )
            altered_review = gate.add_self_hash(
                altered_review,
                "fresh_implementation_review_payload_sha256",
            )
            with self.assertRaises(
                gate.Gate12C2OriginalBaselineError
            ):
                gate.validate_implementation_review(
                    self.plan,
                    altered_review,
                    candidate_file_sha256=candidate_file_hash,
                    candidate_payload_sha256=candidate[
                        "implementation_candidate_binding_payload_sha256"
                    ],
                    source_commit=SOURCE_COMMIT,
                    candidate=candidate,
                )
            altered_authority = dict(authority)
            altered_authority["review_surface_identity"] = altered_surface
            altered_authority.pop(
                "reviewed_implementation_authority_payload_sha256"
            )
            altered_authority = gate.add_self_hash(
                altered_authority,
                "reviewed_implementation_authority_payload_sha256",
            )
            with self.assertRaises(
                gate.Gate12C2OriginalBaselineError
            ):
                gate.validate_reviewed_authority(
                    self.plan,
                    altered_authority,
                    candidate=candidate,
                    candidate_file_sha256=candidate_file_hash,
                    review=review,
                    review_file_sha256=review_file_hash,
                )

    def test_trust_digest_is_recomputed_and_never_copied_through(self) -> None:
        retained_digest = copy.deepcopy(self.plan)
        retained_digest["implementation_trust_model_contract"][
            "synthetic_sensitive_marker"
        ] = False
        replaced_digest = copy.deepcopy(retained_digest)
        replaced_digest["implementation_trust_model_sha256"] = (
            gate.sha256_bytes(
                gate.canonical_json_bytes(
                    replaced_digest[
                        "implementation_trust_model_contract"
                    ]
                )
            )
        )
        for altered in (retained_digest, replaced_digest):
            with self.subTest(
                digest=altered["implementation_trust_model_sha256"]
            ):
                with self.assertRaises(
                    gate.Gate12C2OriginalBaselineError
                ):
                    gate.recompute_implementation_trust_model_sha256(
                        altered
                    )
                with self.assertRaises(
                    independent.IndependentVerificationError
                ):
                    independent._trust_model_sha256(altered)
        wrong_digest = replaced_digest[
            "implementation_trust_model_sha256"
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = candidate_for_temp_repository(self.plan, root)
            candidate_file_hash = gate.sha256_bytes(
                gate.canonical_receipt_bytes(candidate)
            )
            review = pass_review(
                self.plan, candidate, candidate_file_hash
            )
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
            altered_candidate = dict(candidate)
            altered_candidate["implementation_trust_model_sha256"] = (
                wrong_digest
            )
            altered_candidate.pop(
                "implementation_candidate_binding_payload_sha256"
            )
            altered_candidate = gate.add_self_hash(
                altered_candidate,
                "implementation_candidate_binding_payload_sha256",
            )
            with self.assertRaises(
                gate.Gate12C2OriginalBaselineError
            ):
                gate.validate_candidate_binding(
                    self.plan,
                    altered_candidate,
                    repo_root=root,
                    current_head=SOURCE_COMMIT,
                )
            altered_review = dict(review)
            altered_review["implementation_trust_model_sha256"] = (
                wrong_digest
            )
            altered_review.pop(
                "fresh_implementation_review_payload_sha256"
            )
            altered_review = gate.add_self_hash(
                altered_review,
                "fresh_implementation_review_payload_sha256",
            )
            with self.assertRaises(
                gate.Gate12C2OriginalBaselineError
            ):
                gate.validate_implementation_review(
                    self.plan,
                    altered_review,
                    candidate_file_sha256=candidate_file_hash,
                    candidate_payload_sha256=candidate[
                        "implementation_candidate_binding_payload_sha256"
                    ],
                    source_commit=SOURCE_COMMIT,
                    candidate=candidate,
                )
            altered_authority = dict(authority)
            altered_authority["implementation_trust_model_sha256"] = (
                wrong_digest
            )
            altered_authority.pop(
                "reviewed_implementation_authority_payload_sha256"
            )
            altered_authority = gate.add_self_hash(
                altered_authority,
                "reviewed_implementation_authority_payload_sha256",
            )
            with self.assertRaises(
                gate.Gate12C2OriginalBaselineError
            ):
                gate.validate_reviewed_authority(
                    self.plan,
                    altered_authority,
                    candidate=candidate,
                    candidate_file_sha256=candidate_file_hash,
                    review=review,
                    review_file_sha256=review_file_hash,
                )
            with self.assertRaises(
                gate.Gate12C2OriginalBaselineError
            ):
                gate.build_reviewed_authority_payload(
                    replaced_digest,
                    candidate,
                    review,
                    candidate_file_sha256=candidate_file_hash,
                    review_file_sha256=review_file_hash,
                )
            with self.assertRaises(
                independent.IndependentVerificationError
            ):
                independent._independent_reviewed_authority_payload(
                    replaced_digest,
                    candidate,
                    review,
                    candidate_file_hash=candidate_file_hash,
                    review_file_hash=review_file_hash,
                )


class CandidateLineageAdversarialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = gate.load_frozen_plan()

    @staticmethod
    def _lineage_checks(
        lineage: Sequence[str],
    ) -> tuple[tuple[str, object, type[BaseException]], ...]:
        return (
            (
                "builder",
                lambda: binding_builder._require_direct_child_lineage(
                    SOURCE_COMMIT, list(lineage)
                ),
                gate.Gate12C2OriginalBaselineError,
            ),
            (
                "core",
                lambda: gate.require_direct_child_lineage(
                    SOURCE_COMMIT,
                    gate.REMEDIATION_BASE_COMMIT,
                    lineage,
                ),
                gate.Gate12C2OriginalBaselineError,
            ),
            (
                "independent_verifier",
                lambda: independent._require_direct_child_lineage(
                    SOURCE_COMMIT, lineage
                ),
                independent.IndependentVerificationError,
            ),
        )

    def test_direct_child_is_the_only_accepted_parent_shape(self) -> None:
        direct = (SOURCE_COMMIT, gate.REMEDIATION_BASE_COMMIT)
        for name, check, _error_type in self._lineage_checks(direct):
            with self.subTest(path=name, shape="direct_child"):
                check()

        invalid = {
            "grandchild": (SOURCE_COMMIT, "b" * 40),
            "merge_commit": (
                SOURCE_COMMIT,
                gate.REMEDIATION_BASE_COMMIT,
                "c" * 40,
            ),
            "zero_parent": (SOURCE_COMMIT,),
            "unrelated_commit": ("d" * 40, gate.REMEDIATION_BASE_COMMIT),
        }
        for shape, lineage in invalid.items():
            for name, check, error_type in self._lineage_checks(lineage):
                with self.subTest(path=name, shape=shape):
                    with self.assertRaises(error_type) as caught:
                        check()
                    self.assertEqual(
                        caught.exception.code,
                        "IMPLEMENTATION_BYTE_IDENTITY_MISMATCH",
                    )

    def test_all_lineage_consumers_read_the_frozen_base_parent(self) -> None:
        self.assertEqual(
            gate.git_commit_parent_lineage(
                REPOSITORY, gate.REMEDIATION_BASE_COMMIT
            ),
            (
                gate.REMEDIATION_BASE_COMMIT,
                gate.REMEDIATION_BASE_PARENT,
            ),
        )

        self.assertEqual(
            tuple(
                binding_builder._git(
                    REPOSITORY,
                    "rev-list",
                    "--parents",
                    "-n",
                    "1",
                    gate.REMEDIATION_BASE_COMMIT,
                ).split()
            ),
            (
                gate.REMEDIATION_BASE_COMMIT,
                gate.REMEDIATION_BASE_PARENT,
            ),
        )
        self.assertEqual(
            independent._git_parent_lineage(
                REPOSITORY, gate.REMEDIATION_BASE_COMMIT
            ),
            (
                gate.REMEDIATION_BASE_COMMIT,
                gate.REMEDIATION_BASE_PARENT,
            ),
        )

    def test_candidate_source_swap_and_restore_head_mismatch_reject(
        self,
    ) -> None:
        direct = (SOURCE_COMMIT, gate.REMEDIATION_BASE_COMMIT)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = candidate_for_temp_repository(self.plan, root)
            with mock.patch.object(
                gate,
                "git_commit_parent_lineage",
                return_value=direct,
            ):
                self.assertEqual(
                    gate.validate_candidate_binding(
                        self.plan,
                        candidate,
                        repo_root=root,
                        current_head=SOURCE_COMMIT,
                    ),
                    candidate,
                )

            swapped = copy.deepcopy(candidate)
            swapped["source_commit"] = "b" * 40
            swapped.pop(
                "implementation_candidate_binding_payload_sha256"
            )
            swapped = gate.add_self_hash(
                swapped,
                "implementation_candidate_binding_payload_sha256",
            )
            with mock.patch.object(
                gate,
                "git_commit_parent_lineage",
                return_value=direct,
            ):
                with self.assertRaises(
                    gate.Gate12C2OriginalBaselineError
                ):
                    gate.validate_candidate_binding(
                        self.plan,
                        swapped,
                        repo_root=root,
                        current_head="b" * 40,
                    )

            restore_mismatch = copy.deepcopy(candidate)
            restore_mismatch["clean_restore"]["restore_head"] = "c" * 40
            restore_mismatch.pop(
                "implementation_candidate_binding_payload_sha256"
            )
            restore_mismatch = gate.add_self_hash(
                restore_mismatch,
                "implementation_candidate_binding_payload_sha256",
            )
            with mock.patch.object(
                gate,
                "git_commit_parent_lineage",
                return_value=direct,
            ):
                with self.assertRaises(
                    gate.Gate12C2OriginalBaselineError
                ):
                    gate.validate_candidate_binding(
                        self.plan,
                        restore_mismatch,
                        repo_root=root,
                        current_head=SOURCE_COMMIT,
                    )

    def test_downstream_authority_rehash_cannot_change_source_commit(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = candidate_for_temp_repository(self.plan, root)
            candidate_file_hash = gate.sha256_bytes(
                gate.canonical_receipt_bytes(candidate)
            )
            review = pass_review(
                self.plan,
                candidate,
                candidate_file_hash,
            )
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
            authority["implementation_source_commit"] = "b" * 40
            authority.pop(
                "reviewed_implementation_authority_payload_sha256"
            )
            authority = gate.add_self_hash(
                authority,
                "reviewed_implementation_authority_payload_sha256",
            )
            with self.assertRaises(
                gate.Gate12C2OriginalBaselineError
            ):
                gate.validate_reviewed_authority(
                    self.plan,
                    authority,
                    candidate=candidate,
                    candidate_file_sha256=candidate_file_hash,
                    review=review,
                    review_file_sha256=review_file_hash,
                )


class AuthorityAndAuthorizationAdversarialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = gate.load_frozen_plan()

    def setUp(self) -> None:
        lineage = (SOURCE_COMMIT, gate.REMEDIATION_BASE_COMMIT)
        patcher = mock.patch.object(
            gate,
            "git_commit_parent_lineage",
            return_value=lineage,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

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
            implementation_source_commit=SOURCE_COMMIT,
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
                implementation_source_commit=SOURCE_COMMIT,
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
                git_head_at_claim=SOURCE_COMMIT,
                executing_code_identity_surface_sha256="a" * 64,
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
            "reviewed_implementation_authority_payload_sha256": "4" * 64,
            "implementation_source_commit": SOURCE_COMMIT,
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
                                "load_active_plan",
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
            "review_surface_identity": gate.review_surface_identity(self.plan),
            "issuer_id": "ignored-by-derivation",
        }
        review = {
            "fresh_implementation_review_payload_sha256": OTHER_DIGEST,
            "review_surface_identity": candidate["review_surface_identity"],
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
            gate.artifact_exact_fields(
                self.plan, "implementation_candidate_binding"
            )
        )
        review_fields = set(
            gate.artifact_exact_fields(
                self.plan, "fresh_implementation_review_verdict"
            )
        )
        authority_fields = set(
            gate.artifact_exact_fields(
                self.plan, "reviewed_implementation_authority"
            )
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
                final.with_name(final.name + ".pending-v0.9").exists()
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
            "reviewed_implementation_authority_payload_sha256": "4" * 64,
            "implementation_source_commit": SOURCE_COMMIT,
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
        for field, replacement in (
            ("authorized_implementation_repository", "C:\\substituted"),
            ("implementation_source_commit", "b" * 40),
            ("executing_code_identity_status", "unverified"),
        ):
            altered = dict(preflight)
            altered[field] = replacement
            with self.subTest(preflight_field=field), self.assertRaises(
                independent.IndependentVerificationError
            ):
                independent._validate_control_preflight(
                    self.plan,
                    altered,
                    scope="verifier",
                    authority=authority,
                    authority_file_hash="3" * 64,
                    now_ns=gate.parse_utc_ns(
                        "2026-08-01T00:01:00Z"
                    ),
                    linked_receipts=links,
                )
        for field, replacement in (
            ("implementation_source_commit", "b" * 40),
            ("git_head_at_claim", "b" * 40),
            ("executing_code_identity_surface_sha256", "not-a-digest"),
        ):
            altered = dict(claim)
            altered[field] = replacement
            with self.subTest(claim_field=field), self.assertRaises(
                independent.IndependentVerificationError
            ):
                independent._validate_control_claim(
                    self.plan,
                    altered,
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
            implementation_source_commit=claim[
                "implementation_source_commit"
            ],
            git_head_at_protected_read=claim["git_head_at_claim"],
            git_head_at_terminal=claim["git_head_at_claim"],
            executing_code_identity_surface_sha256=claim[
                "executing_code_identity_surface_sha256"
            ],
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
        extraction_source = inspect.getsource(
            gate.ExecutingCodeIdentity.load_scientific_dependencies
        )
        for filename in (
            "gate12c2_development_shards.py",
            "gate12c2_synthetic_lab.py",
        ):
            self.assertIn(filename, extraction_source)
        self.assertIn(
            "sha256_bytes(raw)",
            inspect.getsource(gate.ExecutingCodeIdentity._open_source),
        )
        self.assertLess(
            extraction_source.index(
                "retained, raw = self._open_source(relative)"
            ),
            extraction_source.index("loader.exec_module(module)"),
        )
        lineage_sources = (
            inspect.getsource(gate.git_commit_parent_lineage),
            inspect.getsource(independent._git_parent_lineage),
        )
        for source in lineage_sources:
            self.assertIn("subprocess.run", source)
            self.assertIn('"rev-list"', source)
            self.assertIn('"--parents"', source)
        protected_read_sources = (
            inspect.getsource(gate.extract_commitments_after_claim),
            inspect.getsource(independent.independent_rederive),
        )
        for source in protected_read_sources:
            self.assertNotIn("subprocess", source)
        runner_source = Path(extraction_runner.__file__).read_text(
            encoding="utf-8"
        )
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
        self.assertEqual(
            independent.CONTRACT_FILE_SHA256,
            gate.CONTRACT_FILE_SHA256,
        )
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
            gate.artifact_exact_fields(
                self.plan, "implementation_candidate_binding"
            )
        )
        review = set(
            gate.artifact_exact_fields(
                self.plan, "fresh_implementation_review_verdict"
            )
        )
        authority_contract = self.plan[
            "reviewed_implementation_authority_contract"
        ]
        authority = set(
            gate.artifact_exact_fields(
                self.plan, "reviewed_implementation_authority"
            )
        )
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
        candidate_fixture = {
            "source_commit": SOURCE_COMMIT,
            "implementation_candidate_binding_payload_sha256": DIGEST,
            "review_surface_identity": gate.review_surface_identity(self.plan),
        }
        review_fixture = {
            "fresh_implementation_review_payload_sha256": OTHER_DIGEST,
            "review_surface_identity": candidate_fixture[
                "review_surface_identity"
            ],
        }
        core_authority = gate.build_reviewed_authority_payload(
            self.plan,
            candidate_fixture,
            review_fixture,
            candidate_file_sha256="1" * 64,
            review_file_sha256="2" * 64,
        )
        independent_authority = (
            independent._independent_reviewed_authority_payload(
                self.plan,
                candidate_fixture,
                review_fixture,
                candidate_file_hash="1" * 64,
                review_file_hash="2" * 64,
            )
        )
        self.assertEqual(independent_authority, core_authority)
        self.assertEqual(set(independent_authority), authority)

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



class R2ActivationLaneAdversarialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base_plan = gate.load_frozen_plan()
        cls.r2_plan = gate.load_r2_activation_plan(
            base_plan=cls.base_plan,
            check_legacy_occupancy=True,
        )
        cls.plan = gate.build_r2_active_plan(
            cls.base_plan,
            cls.r2_plan,
        )

    @staticmethod
    def _rehash_r2(value: Mapping[str, Any]) -> dict[str, Any]:
        payload = copy.deepcopy(dict(value))
        payload.pop("r2_activation_plan_payload_sha256", None)
        return gate.add_self_hash(
            payload,
            "r2_activation_plan_payload_sha256",
        )

    def _surface_mutation(
        self,
        mutate: Any,
    ) -> tuple[dict[str, Any], str]:
        payload = copy.deepcopy(self.r2_plan)
        mutate(payload["artifact_path_surface"])
        surface_hash = gate.sha256_bytes(
            gate.canonical_json_bytes(payload["artifact_path_surface"])
        )
        payload["artifact_path_surface_sha256"] = surface_hash
        payload["implementation_binding_contract_overlay"][
            "artifact_path_surface_sha256"
        ] = surface_hash
        payload["reviewed_authority_contract_overlay"][
            "artifact_path_surface_sha256"
        ] = surface_hash
        return self._rehash_r2(payload), surface_hash

    def test_old_v09_occupancy_and_r2_surface_coexist(self) -> None:
        rows = self.plan["artifact_path_surface"]
        finals = [row["final_path"] for row in rows]
        pending = [row["pending_path"] for row in rows]
        self.assertEqual(len(finals), 18)
        self.assertEqual(len(set(finals + pending)), 36)
        occupied = self.r2_plan["occupied_v0_9"]
        self.assertNotIn(
            occupied["candidate_binding"]["path"],
            finals,
        )
        self.assertNotIn(
            occupied["review_verdict"]["path"],
            finals,
        )
        self.assertEqual(
            gate.artifact_surface_sha256(self.plan),
            gate.R2_ARTIFACT_PATH_SURFACE_SHA256,
        )
        self.assertEqual(
            independent._independent_load_r2_active()[
                "artifact_path_surface_sha256"
            ],
            gate.R2_ARTIFACT_PATH_SURFACE_SHA256,
        )

    def test_phase_a_plan_load_never_reads_protected_root(self) -> None:
        original = Path.read_bytes
        protected = str(gate.PROTECTED_ROOT).casefold()

        def guarded(path: Path) -> bytes:
            if str(path).casefold().startswith(protected):
                raise AssertionError("protected root read during Phase A")
            return original(path)

        with mock.patch.object(Path, "read_bytes", guarded):
            core = gate.load_active_plan()
            independently = independent.independent_load_plan()
        self.assertEqual(
            core["artifact_path_surface_sha256"],
            independently["artifact_path_surface_sha256"],
        )

    def test_old_namespace_publication_is_rejected_before_io(self) -> None:
        with mock.patch.object(gate, "atomic_publish_exact") as publish:
            with self.assertRaises(gate.Gate12C2OriginalBaselineError):
                gate.publish_role(
                    self.plan,
                    "formal_design_review_verdict",
                    {},
                )
            tampered = copy.deepcopy(self.plan)
            occupied = self.r2_plan["occupied_v0_9"]
            row = next(
                item
                for item in tampered["artifact_path_surface"]
                if item["role"] == "implementation_candidate_binding"
            )
            row["final_path"] = occupied["candidate_binding"]["path"]
            row["pending_path"] = row["final_path"] + ".pending-v0.9"
            with self.assertRaises(gate.Gate12C2OriginalBaselineError):
                gate.publish_role(
                    tampered,
                    "implementation_candidate_binding",
                    {},
                )
            publish.assert_not_called()

    def test_unknown_duplicate_and_colliding_roles_are_rejected(self) -> None:
        def duplicate(rows: list[dict[str, Any]]) -> None:
            rows[1]["role"] = rows[0]["role"]

        def unknown(rows: list[dict[str, Any]]) -> None:
            rows[1]["role"] = "unknown_r2_role"

        def collision(rows: list[dict[str, Any]]) -> None:
            rows[1]["final_path"] = rows[0]["pending_path"]

        for mutation in (duplicate, unknown, collision):
            with self.subTest(mutation=mutation.__name__):
                payload, surface_hash = self._surface_mutation(mutation)
                with (
                    mock.patch.object(
                        gate,
                        "R2_ARTIFACT_PATH_SURFACE_SHA256",
                        surface_hash,
                    ),
                    mock.patch.object(
                        gate,
                        "R2_ACTIVATION_PLAN_PAYLOAD_SHA256",
                        payload[
                            "r2_activation_plan_payload_sha256"
                        ],
                    ),
                    self.assertRaises(
                        gate.Gate12C2OriginalBaselineError
                    ),
                ):
                    gate.validate_r2_activation_plan(
                        self.base_plan,
                        payload,
                        check_legacy_occupancy=False,
                    )

    def test_review_surface_shrink_is_rejected(self) -> None:
        surface = gate.review_surface_identity(self.plan)
        surface["compatibility_row_count"] -= 1
        surface["review_surface_identity_sha256"] = gate.sha256_bytes(
            gate.canonical_json_bytes(
                {
                    key: value
                    for key, value in surface.items()
                    if key != "review_surface_identity_sha256"
                }
            )
        )
        with self.assertRaises(gate.Gate12C2OriginalBaselineError):
            gate.validate_review_surface_identity(self.plan, surface)

        r2 = copy.deepcopy(self.r2_plan)
        r2["preserved_identities"]["required_mutation_count"] -= 1
        r2 = self._rehash_r2(r2)
        with (
            mock.patch.object(
                gate,
                "R2_ACTIVATION_PLAN_PAYLOAD_SHA256",
                r2["r2_activation_plan_payload_sha256"],
            ),
            self.assertRaises(gate.Gate12C2OriginalBaselineError),
        ):
            gate.validate_r2_activation_plan(
                self.base_plan,
                r2,
                check_legacy_occupancy=False,
            )

    def test_boolean_only_pass_cannot_reach_review_validation(self) -> None:
        schema = self.plan["review_receipt_schemas"][
            "fresh_implementation_review_verdict"
        ]
        fields = gate.artifact_exact_fields(
            self.plan,
            "fresh_implementation_review_verdict",
        )
        review = {field: None for field in fields}
        review.update(schema["outcomes"]["pass"]["required_values"])
        review.update(
            {
                "reviewed_at_utc": "2026-08-03T00:00:00Z",
                "reviewer_context_id": "fresh-r2-reviewer",
                "review_surface_identity": gate.review_surface_identity(
                    self.plan
                ),
                "P0_count": 0,
                "P1_count": 0,
                "P2_count": 0,
                "implementation_candidate_binding_file_sha256": DIGEST,
                "implementation_candidate_binding_payload_sha256": (
                    OTHER_DIGEST
                ),
                "implementation_source_commit": SOURCE_COMMIT,
                "implementation_review_packet_file_sha256": DIGEST,
                "bundle_file_sha256": DIGEST,
                "restore_receipt_file_sha256": DIGEST,
                "restore_receipt_payload_sha256": DIGEST,
                "implementation_author_separation_contract_sha256": (
                    gate.IMPLEMENTATION_AUTHOR_SEPARATION_SHA256
                ),
                "implementation_trust_model_sha256": (
                    gate.IMPLEMENTATION_TRUST_MODEL_SHA256
                ),
                "formal_design_review_file_sha256": (
                    gate.FORMAL_DESIGN_REVIEW_FILE_SHA256
                ),
                "formal_design_review_payload_sha256": (
                    gate.FORMAL_DESIGN_REVIEW_PAYLOAD_SHA256
                ),
                "contract_file_sha256": gate.CONTRACT_FILE_SHA256,
                "plan_file_sha256": gate.PLAN_FILE_SHA256,
                "plan_payload_sha256": gate.PLAN_PAYLOAD_SHA256,
                "r2_activation_plan_file_sha256": (
                    gate.R2_ACTIVATION_PLAN_FILE_SHA256
                ),
                "r2_activation_plan_payload_sha256": (
                    gate.R2_ACTIVATION_PLAN_PAYLOAD_SHA256
                ),
                "artifact_path_surface_sha256": (
                    gate.R2_ARTIFACT_PATH_SURFACE_SHA256
                ),
                "occupied_v0_9_surface_sha256": (
                    gate.R2_OCCUPIED_V0_9_SURFACE_SHA256
                ),
                "candidate_manifest_file_sha256": DIGEST,
                "candidate_manifest_payload_sha256": OTHER_DIGEST,
                "review_evidence_file_sha256": DIGEST,
                "review_evidence_payload_sha256": OTHER_DIGEST,
            }
        )
        review.pop(
            "fresh_implementation_review_payload_sha256",
            None,
        )
        review = gate.add_self_hash(
            review,
            "fresh_implementation_review_payload_sha256",
        )
        candidate = {
            "source_commit": SOURCE_COMMIT,
            "review_surface_identity": gate.review_surface_identity(
                self.plan
            ),
            "clean_restore": {
                "bundle_file_sha256": DIGEST,
                "restore_receipt_file_sha256": DIGEST,
                "restore_receipt_payload_sha256": DIGEST,
            },
            "candidate_manifest_file_sha256": DIGEST,
            "candidate_manifest_payload_sha256": OTHER_DIGEST,
        }
        with (
            mock.patch.object(
                gate,
                "read_r2_fresh_review_evidence",
                side_effect=gate.Gate12C2OriginalBaselineError(
                    "R2_REVIEW_EVIDENCE_INVALID"
                ),
            ),
            self.assertRaises(gate.Gate12C2OriginalBaselineError),
        ):
            gate.validate_implementation_review(
                self.plan,
                review,
                candidate_file_sha256=DIGEST,
                candidate_payload_sha256=OTHER_DIGEST,
                source_commit=SOURCE_COMMIT,
                candidate=candidate,
            )

    def test_old_candidate_and_old_reopen_cannot_mix_into_r2(self) -> None:
        occupied = self.r2_plan["occupied_v0_9"]
        old_candidate = gate.read_canonical_receipt(
            Path(occupied["candidate_binding"]["path"]),
            expected_file_sha256=occupied["candidate_binding"][
                "file_sha256"
            ],
            hash_field="implementation_candidate_binding_payload_sha256",
            expected_payload_sha256=occupied["candidate_binding"][
                "payload_sha256"
            ],
        )
        with self.assertRaises(gate.Gate12C2OriginalBaselineError):
            gate.validate_candidate_binding(
                self.plan,
                old_candidate,
                repo_root=REPOSITORY,
            )
        old_review = gate.read_canonical_receipt(
            Path(occupied["review_verdict"]["path"]),
            expected_file_sha256=occupied["review_verdict"][
                "file_sha256"
            ],
            hash_field="fresh_implementation_review_payload_sha256",
            expected_payload_sha256=occupied["review_verdict"][
                "payload_sha256"
            ],
        )
        with self.assertRaises(gate.Gate12C2OriginalBaselineError):
            gate.require_exact_keys(
                old_review,
                gate.artifact_exact_fields(
                    self.plan,
                    "fresh_implementation_review_verdict",
                ),
                code="INPUT_LINEAGE_MISMATCH",
            )

    def test_candidate_review_authority_rehash_mixes_are_rejected(self) -> None:
        surface = gate.review_surface_identity(self.plan)
        candidate = {
            "source_commit": SOURCE_COMMIT,
            "authority_namespace_id": gate.R2_AUTHORITY_NAMESPACE_ID,
            "review_surface_identity": surface,
            "implementation_candidate_binding_payload_sha256": DIGEST,
            "candidate_manifest_file_sha256": DIGEST,
            "candidate_manifest_payload_sha256": OTHER_DIGEST,
        }
        review = {
            "authority_namespace_id": gate.R2_AUTHORITY_NAMESPACE_ID,
            "implementation_source_commit": SOURCE_COMMIT,
            "implementation_candidate_binding_file_sha256": DIGEST,
            "implementation_candidate_binding_payload_sha256": DIGEST,
            "review_surface_identity": surface,
            "fresh_implementation_review_payload_sha256": OTHER_DIGEST,
        }
        authority = gate.build_reviewed_authority_payload(
            self.plan,
            candidate,
            review,
            candidate_file_sha256=DIGEST,
            review_file_sha256=OTHER_DIGEST,
        )
        independent_authority = (
            independent._independent_reviewed_authority_payload(
                independent._independent_load_r2_active(),
                candidate,
                review,
                candidate_file_hash=DIGEST,
                review_file_hash=OTHER_DIGEST,
            )
        )
        self.assertEqual(independent_authority, authority)
        tampered_authority = dict(authority)
        tampered_authority["candidate_manifest_file_sha256"] = (
            "a" * 64
        )
        tampered_authority.pop(
            "reviewed_implementation_authority_payload_sha256"
        )
        tampered_authority = gate.add_self_hash(
            tampered_authority,
            "reviewed_implementation_authority_payload_sha256",
        )
        with self.assertRaises(gate.Gate12C2OriginalBaselineError):
            gate.validate_reviewed_authority(
                self.plan,
                tampered_authority,
                candidate=candidate,
                candidate_file_sha256=DIGEST,
                review=review,
                review_file_sha256=OTHER_DIGEST,
            )
        mixed_review = dict(review)
        mixed_review["authority_namespace_id"] = "v0.9"
        with self.assertRaises(gate.Gate12C2OriginalBaselineError):
            gate.build_reviewed_authority_payload(
                self.plan,
                candidate,
                mixed_review,
                candidate_file_sha256=DIGEST,
                review_file_sha256=OTHER_DIGEST,
            )
        with self.assertRaises(
            independent.IndependentVerificationError
        ):
            independent._independent_reviewed_authority_payload(
                independent._independent_load_r2_active(),
                candidate,
                mixed_review,
                candidate_file_hash=DIGEST,
                review_file_hash=OTHER_DIGEST,
            )
        mixed_candidate = dict(candidate)
        mixed_candidate[
            "implementation_candidate_binding_payload_sha256"
        ] = "a" * 64
        with self.assertRaises(gate.Gate12C2OriginalBaselineError):
            gate.build_reviewed_authority_payload(
                self.plan,
                mixed_candidate,
                review,
                candidate_file_sha256=DIGEST,
                review_file_sha256=OTHER_DIGEST,
            )
        with self.assertRaises(
            independent.IndependentVerificationError
        ):
            independent._independent_reviewed_authority_payload(
                independent._independent_load_r2_active(),
                mixed_candidate,
                review,
                candidate_file_hash=DIGEST,
                review_file_hash=OTHER_DIGEST,
            )

    def test_exact_two_commit_lineage_rejects_other_shapes(self) -> None:
        source = "1" * 40
        base = gate.R2_TASK1_COMMIT
        parent = gate.R2_TASK1_PARENT
        gate.require_direct_child_lineage(
            source,
            base,
            (source, base),
        )
        independent._require_direct_child_lineage(
            source,
            (source, base),
            expected_parent=base,
        )
        for lineage in (
            (source,),
            (source, base, parent),
            (source, parent),
            (source, "f" * 40),
        ):
            with self.subTest(lineage=lineage):
                with self.assertRaises(
                    gate.Gate12C2OriginalBaselineError
                ):
                    gate.require_direct_child_lineage(
                        source,
                        base,
                        lineage,
                    )
                with self.assertRaises(
                    independent.IndependentVerificationError
                ):
                    independent._require_direct_child_lineage(
                        source,
                        lineage,
                        expected_parent=base,
                    )


class R2R1P1RemediationAdversarialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        r2_active = gate.load_r2_active_plan()
        remediation = gate.load_r2r1_remediation_plan(
            repository_root=REPOSITORY,
            r2_active_plan=r2_active,
            check_r2_occupancy=True,
        )
        cls.plan = gate.build_r2r1_active_plan(
            r2_active,
            remediation,
            repository_root=REPOSITORY,
        )
        independent_r2 = independent._independent_load_r2_active()
        independent_overlay = independent._independent_load_r2r1_plan(
            independent_r2,
            repository_root=REPOSITORY,
        )
        cls.independent_plan = (
            independent._independent_build_r2r1_active(
                independent_r2, independent_overlay
            )
        )
        cls.control = cls.plan["r2r1_remediation_control"]
        cls.source_commit = "b" * 40
        cls.sibling_commit = "c" * 40
        cls.raw = b"r2r1-implementation-byte\n"
        cls.blob = gate.git_blob_oid(cls.raw, "sha1")
        cls.plan_raw = gate.R2R1_REMEDIATION_PLAN_PATH.read_bytes()
        role_by_path = gate.IMPLEMENTATION_ROLE_BY_PATH
        cls.implementation_rows = sorted(
            [
                {
                    "role": role_by_path[path],
                    "relative_path": path,
                    "file_sha256": gate.sha256_bytes(cls.raw),
                    "git_blob_oid": cls.blob,
                }
                for path in gate.IMPLEMENTATION_PATHS
            ],
            key=lambda row: row["role"],
        )
        cls.changed_rows = [
            {
                "relative_path": path,
                "file_sha256": gate.sha256_bytes(
                    ("changed:" + path).encode("utf-8")
                ),
                "git_blob_oid": gate.sha256_bytes(
                    ("blob:" + path).encode("utf-8")
                )[:40],
            }
            for path in cls.control["allowed_changed_paths"]
        ]
        cls.changed_digest = gate.sha256_bytes(
            gate.canonical_json_bytes(cls.changed_rows)
        )
        cls.restore = cls._make_restore(cls.source_commit)
        cls.selection = cls._make_selection(cls.source_commit)

    @classmethod
    def _make_restore(cls, source_commit: str) -> dict[str, Any]:
        contract = cls.control["clean_restore_receipt_contract"]
        coverage = cls.control["review_coverage_identity"]
        payload = {field: None for field in contract["exact_top_level_fields"]}
        payload.update(contract["required_values"])
        payload.update(
            {
                "source_commit": source_commit,
                "source_parent_commit": gate.R2R1_PARENT_COMMIT,
                "source_grandparent_commit": gate.R2R1_GRANDPARENT_COMMIT,
                "bundle_path": "C:\\tmp\\r2r1-test.bundle",
                "bundle_file_sha256": "1" * 64,
                "bundle_size_bytes": 4,
                "restore_path": "C:\\tmp\\r2r1-test-restore",
                "restore_head": source_commit,
                "targeted_test_count": coverage["targeted_test_count"],
                "targeted_test_node_id_sha256": coverage[
                    "targeted_test_node_id_sha256"
                ],
                "full_suite_test_count": coverage["full_suite_test_count"],
                "full_suite_test_node_id_sha256": coverage[
                    "full_suite_test_node_id_sha256"
                ],
            }
        )
        payload.pop("restore_receipt_payload_sha256", None)
        return gate.add_self_hash(payload, "restore_receipt_payload_sha256")

    @classmethod
    def _make_selection(cls, source_commit: str) -> dict[str, Any]:
        contract = cls.control["candidate_selection_contract"]
        coverage = cls.control["review_coverage_identity"]
        payload = {field: None for field in contract["exact_top_level_fields"]}
        payload.update(contract["required_values"])
        payload.update(
            {
                "exact_candidate_commit": source_commit,
                "git_object_format": "sha1",
                "changed_path_allowlist": cls.control[
                    "allowed_changed_paths"
                ],
                "changed_files": cls.changed_rows,
                "changed_file_manifest_sha256": cls.changed_digest,
                "implementation_files": cls.implementation_rows,
                "r2_activation_plan_file_sha256": (
                    gate.R2_ACTIVATION_PLAN_FILE_SHA256
                ),
                "r2_activation_plan_payload_sha256": (
                    gate.R2_ACTIVATION_PLAN_PAYLOAD_SHA256
                ),
                "r2r1_remediation_plan_file_sha256": (
                    gate.R2R1_REMEDIATION_PLAN_FILE_SHA256
                ),
                "r2r1_remediation_plan_payload_sha256": (
                    gate.R2R1_REMEDIATION_PLAN_PAYLOAD_SHA256
                ),
                "artifact_path_surface_sha256": (
                    gate.R2R1_ARTIFACT_PATH_SURFACE_SHA256
                ),
                "occupied_r2_surface_sha256": (
                    gate.R2R1_OCCUPIED_R2_SURFACE_SHA256
                ),
                "review_surface_identity_sha256": (
                    gate.REVIEW_SURFACE_IDENTITY_SHA256
                ),
                "implementation_trust_model_sha256": (
                    gate.IMPLEMENTATION_TRUST_MODEL_SHA256
                ),
                "bundle_path": cls.restore["bundle_path"],
                "bundle_file_sha256": cls.restore["bundle_file_sha256"],
                "bundle_size_bytes": cls.restore["bundle_size_bytes"],
                "clean_restore_receipt_file_sha256": DIGEST,
                "clean_restore_receipt_payload_sha256": cls.restore[
                    "restore_receipt_payload_sha256"
                ],
                "targeted_test_count": coverage["targeted_test_count"],
                "targeted_test_node_id_sha256": coverage[
                    "targeted_test_node_id_sha256"
                ],
                "full_suite_test_count": coverage["full_suite_test_count"],
                "full_suite_test_node_id_sha256": coverage[
                    "full_suite_test_node_id_sha256"
                ],
            }
        )
        payload.pop("candidate_selection_payload_sha256", None)
        return gate.add_self_hash(payload, "candidate_selection_payload_sha256")

    @staticmethod
    def _rehash(value: Mapping[str, Any], field: str) -> dict[str, Any]:
        payload = copy.deepcopy(dict(value))
        payload.pop(field, None)
        return gate.add_self_hash(payload, field)

    def _lineage(self, commit: str) -> tuple[str, ...]:
        if commit == self.selection["exact_candidate_commit"]:
            return (commit, gate.R2R1_PARENT_COMMIT)
        if commit == self.sibling_commit:
            return (commit, gate.R2R1_PARENT_COMMIT)
        if commit == gate.R2R1_PARENT_COMMIT:
            return (commit, gate.R2R1_GRANDPARENT_COMMIT)
        return (commit,)

    @contextlib.contextmanager
    def _core_selection_mocks(self) -> Iterator[None]:
        with (
            mock.patch.object(
                gate,
                "git_commit_parent_lineage",
                side_effect=lambda _repo, commit: self._lineage(commit),
            ),
            mock.patch.object(
                gate,
                "r2r1_changed_file_manifest",
                return_value=(self.changed_rows, self.changed_digest),
            ),
            mock.patch.object(
                gate,
                "read_r2r1_clean_restore_receipt",
                return_value=(self.restore, DIGEST),
            ),
            mock.patch.object(Path, "read_bytes", return_value=self.raw),
            mock.patch.object(
                gate, "git_path_blob_oid", return_value=self.blob
            ),
        ):
            yield

    @contextlib.contextmanager
    def _independent_selection_mocks(
        self, selection: Mapping[str, Any]
    ) -> Iterator[None]:
        def read_bytes(path: Path) -> bytes:
            if path.name == gate.R2R1_REMEDIATION_PLAN_PATH.name:
                return self.plan_raw
            return self.raw

        def blob_oid(
            _repository: Path, _commit: str, relative: str
        ) -> str:
            if relative.endswith(
                "gate12c2_original_baseline_r2r1_remediation_plan.json"
            ):
                return independent._git_blob(self.plan_raw, "sha1")
            return self.blob

        with (
            mock.patch.object(
                independent,
                "_receipt",
                return_value=(dict(selection), DIGEST),
            ),
            mock.patch.object(
                independent,
                "_git_parent_lineage",
                side_effect=lambda _repo, commit: self._lineage(commit),
            ),
            mock.patch.object(
                independent,
                "_independent_r2r1_changed_manifest",
                return_value=(self.changed_rows, self.changed_digest),
            ),
            mock.patch.object(
                independent,
                "_independent_git_object_format",
                return_value="sha1",
            ),
            mock.patch.object(
                independent,
                "_independent_r2r1_restore_receipt",
                return_value=(self.restore, DIGEST),
            ),
            mock.patch.object(
                Path, "read_bytes", autospec=True, side_effect=read_bytes
            ),
            mock.patch.object(
                independent, "_git_path_blob_oid", side_effect=blob_oid
            ),
        ):
            yield

    def _candidate_review_freeze_evidence(
        self,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        selection = self.selection
        candidate = {
            "source_commit": selection["exact_candidate_commit"],
            "authority_namespace_id": gate.R2R1_AUTHORITY_NAMESPACE_ID,
            "review_surface_identity": gate.review_surface_identity(self.plan),
            "implementation_candidate_binding_payload_sha256": "2" * 64,
            "candidate_manifest_file_sha256": "3" * 64,
            "candidate_manifest_payload_sha256": "4" * 64,
            "candidate_selection_file_sha256": DIGEST,
            "candidate_selection_payload_sha256": selection[
                "candidate_selection_payload_sha256"
            ],
            "r2r1_remediation_plan_file_sha256": (
                gate.R2R1_REMEDIATION_PLAN_FILE_SHA256
            ),
            "r2r1_remediation_plan_payload_sha256": (
                gate.R2R1_REMEDIATION_PLAN_PAYLOAD_SHA256
            ),
            "occupied_r2_surface_sha256": (
                gate.R2R1_OCCUPIED_R2_SURFACE_SHA256
            ),
            "clean_restore": {
                "bundle_file_sha256": self.restore["bundle_file_sha256"],
                "restore_receipt_file_sha256": DIGEST,
                "restore_receipt_payload_sha256": self.restore[
                    "restore_receipt_payload_sha256"
                ],
            },
        }
        packet = b"r2r1-review-packet\n"
        freeze_contract = self.control["review_input_freeze_contract"]
        coverage = self.control["review_coverage_identity"]
        freeze = {
            field: None for field in freeze_contract["exact_top_level_fields"]
        }
        freeze.update(freeze_contract["required_values"])
        freeze.update(
            {
                "implementation_source_commit": candidate["source_commit"],
                "candidate_selection_file_sha256": DIGEST,
                "candidate_selection_payload_sha256": selection[
                    "candidate_selection_payload_sha256"
                ],
                "candidate_manifest_file_sha256": candidate[
                    "candidate_manifest_file_sha256"
                ],
                "candidate_manifest_payload_sha256": candidate[
                    "candidate_manifest_payload_sha256"
                ],
                "implementation_candidate_binding_file_sha256": DIGEST,
                "implementation_candidate_binding_payload_sha256": candidate[
                    "implementation_candidate_binding_payload_sha256"
                ],
                "clean_restore_receipt_file_sha256": DIGEST,
                "clean_restore_receipt_payload_sha256": self.restore[
                    "restore_receipt_payload_sha256"
                ],
                "r2r1_remediation_plan_file_sha256": (
                    gate.R2R1_REMEDIATION_PLAN_FILE_SHA256
                ),
                "r2r1_remediation_plan_payload_sha256": (
                    gate.R2R1_REMEDIATION_PLAN_PAYLOAD_SHA256
                ),
                "artifact_path_surface_sha256": (
                    gate.R2R1_ARTIFACT_PATH_SURFACE_SHA256
                ),
                "occupied_r2_surface_sha256": (
                    gate.R2R1_OCCUPIED_R2_SURFACE_SHA256
                ),
                "review_packet_path": self.control[
                    "fresh_review_packet_path"
                ],
                "review_packet_file_sha256": gate.sha256_bytes(packet),
                "review_packet_size_bytes": len(packet),
                "changed_file_manifest_sha256": self.changed_digest,
                "targeted_test_count": coverage["targeted_test_count"],
                "targeted_test_node_id_sha256": coverage[
                    "targeted_test_node_id_sha256"
                ],
                "full_suite_test_count": coverage["full_suite_test_count"],
                "full_suite_test_node_id_sha256": coverage[
                    "full_suite_test_node_id_sha256"
                ],
            }
        )
        freeze.pop("review_input_freeze_payload_sha256", None)
        freeze = gate.add_self_hash(
            freeze, "review_input_freeze_payload_sha256"
        )
        evidence_contract = self.control["fresh_review_evidence_contract"]
        evidence = {
            field: None for field in evidence_contract["exact_top_level_fields"]
        }
        evidence.update(evidence_contract["required_values"])
        evidence.update(
            {
                "implementation_source_commit": candidate["source_commit"],
                "implementation_candidate_binding_file_sha256": DIGEST,
                "implementation_candidate_binding_payload_sha256": candidate[
                    "implementation_candidate_binding_payload_sha256"
                ],
                "candidate_manifest_file_sha256": candidate[
                    "candidate_manifest_file_sha256"
                ],
                "candidate_manifest_payload_sha256": candidate[
                    "candidate_manifest_payload_sha256"
                ],
                "candidate_selection_file_sha256": DIGEST,
                "candidate_selection_payload_sha256": selection[
                    "candidate_selection_payload_sha256"
                ],
                "review_input_freeze_file_sha256": OTHER_DIGEST,
                "review_input_freeze_payload_sha256": freeze[
                    "review_input_freeze_payload_sha256"
                ],
                "r2r1_remediation_plan_file_sha256": (
                    gate.R2R1_REMEDIATION_PLAN_FILE_SHA256
                ),
                "r2r1_remediation_plan_payload_sha256": (
                    gate.R2R1_REMEDIATION_PLAN_PAYLOAD_SHA256
                ),
                "artifact_path_surface_sha256": (
                    gate.R2R1_ARTIFACT_PATH_SURFACE_SHA256
                ),
                "occupied_r2_surface_sha256": (
                    gate.R2R1_OCCUPIED_R2_SURFACE_SHA256
                ),
                "review_surface_identity": gate.review_surface_identity(
                    self.plan
                ),
                "implementation_review_packet_file_sha256": freeze[
                    "review_packet_file_sha256"
                ],
                "changed_file_manifest_sha256": self.changed_digest,
                "targeted_test_count": coverage["targeted_test_count"],
                "targeted_test_node_id_sha256": coverage[
                    "targeted_test_node_id_sha256"
                ],
                "full_suite_test_count": coverage["full_suite_test_count"],
                "full_suite_test_node_id_sha256": coverage[
                    "full_suite_test_node_id_sha256"
                ],
            }
        )
        evidence.pop("review_evidence_payload_sha256", None)
        evidence = gate.add_self_hash(evidence, "review_evidence_payload_sha256")
        review_schema = self.plan["review_receipt_schemas"][
            "fresh_implementation_review_verdict"
        ]
        review = {
            field: None
            for field in gate.artifact_exact_fields(
                self.plan, "fresh_implementation_review_verdict"
            )
        }
        review.update(review_schema["outcomes"]["pass"]["required_values"])
        review.update(
            {
                "reviewed_at_utc": "2026-08-03T00:00:00Z",
                "implementation_author_separation_contract_sha256": (
                    gate.IMPLEMENTATION_AUTHOR_SEPARATION_SHA256
                ),
                "contract_file_sha256": gate.CONTRACT_FILE_SHA256,
                "plan_file_sha256": gate.PLAN_FILE_SHA256,
                "plan_payload_sha256": gate.PLAN_PAYLOAD_SHA256,
                "formal_design_review_file_sha256": (
                    gate.FORMAL_DESIGN_REVIEW_FILE_SHA256
                ),
                "formal_design_review_payload_sha256": (
                    gate.FORMAL_DESIGN_REVIEW_PAYLOAD_SHA256
                ),
                "implementation_trust_model_sha256": (
                    gate.IMPLEMENTATION_TRUST_MODEL_SHA256
                ),
                "implementation_candidate_binding_file_sha256": DIGEST,
                "implementation_candidate_binding_payload_sha256": candidate[
                    "implementation_candidate_binding_payload_sha256"
                ],
                "implementation_source_commit": candidate["source_commit"],
                "implementation_review_packet_file_sha256": freeze[
                    "review_packet_file_sha256"
                ],
                "bundle_file_sha256": self.restore["bundle_file_sha256"],
                "restore_receipt_file_sha256": DIGEST,
                "restore_receipt_payload_sha256": self.restore[
                    "restore_receipt_payload_sha256"
                ],
                "P0_count": 0,
                "P1_count": 0,
                "P2_count": 0,
                "review_surface_identity": gate.review_surface_identity(
                    self.plan
                ),
                "r2_activation_plan_file_sha256": (
                    gate.R2_ACTIVATION_PLAN_FILE_SHA256
                ),
                "r2_activation_plan_payload_sha256": (
                    gate.R2_ACTIVATION_PLAN_PAYLOAD_SHA256
                ),
                "artifact_path_surface_sha256": (
                    gate.R2R1_ARTIFACT_PATH_SURFACE_SHA256
                ),
                "occupied_v0_9_surface_sha256": (
                    gate.R2_OCCUPIED_V0_9_SURFACE_SHA256
                ),
                "candidate_manifest_file_sha256": candidate[
                    "candidate_manifest_file_sha256"
                ],
                "candidate_manifest_payload_sha256": candidate[
                    "candidate_manifest_payload_sha256"
                ],
                "review_evidence_file_sha256": "5" * 64,
                "review_evidence_payload_sha256": evidence[
                    "review_evidence_payload_sha256"
                ],
                "r2r1_remediation_plan_file_sha256": (
                    gate.R2R1_REMEDIATION_PLAN_FILE_SHA256
                ),
                "r2r1_remediation_plan_payload_sha256": (
                    gate.R2R1_REMEDIATION_PLAN_PAYLOAD_SHA256
                ),
                "occupied_r2_surface_sha256": (
                    gate.R2R1_OCCUPIED_R2_SURFACE_SHA256
                ),
                "candidate_selection_file_sha256": DIGEST,
                "candidate_selection_payload_sha256": selection[
                    "candidate_selection_payload_sha256"
                ],
                "review_input_freeze_file_sha256": OTHER_DIGEST,
                "review_input_freeze_payload_sha256": freeze[
                    "review_input_freeze_payload_sha256"
                ],
            }
        )
        review.pop("fresh_implementation_review_payload_sha256", None)
        review = gate.add_self_hash(
            review, "fresh_implementation_review_payload_sha256"
        )
        return candidate, review, freeze, evidence

    def test_exact_candidate_sibling_commit_is_rejected(self) -> None:
        with self._core_selection_mocks():
            accepted = gate.validate_r2r1_candidate_selection(
                self.plan,
                self.selection,
                repo_root=REPOSITORY,
                current_head=self.selection["exact_candidate_commit"],
            )
            self.assertEqual(
                accepted["exact_candidate_commit"],
                self.selection["exact_candidate_commit"],
            )
            with self.assertRaises(gate.Gate12C2OriginalBaselineError):
                gate.validate_r2r1_candidate_selection(
                    self.plan,
                    self.selection,
                    repo_root=REPOSITORY,
                    current_head=self.sibling_commit,
                )
        with self._independent_selection_mocks(self.selection):
            independently, _file_hash = (
                independent._independent_r2r1_candidate_selection(
                    self.independent_plan, REPOSITORY
                )
            )
        self.assertEqual(
            independently["exact_candidate_commit"],
            self.selection["exact_candidate_commit"],
        )

    def test_sibling_with_recomputed_downstream_hashes_is_rejected(self) -> None:
        candidate, review, freeze, evidence = (
            self._candidate_review_freeze_evidence()
        )
        candidate["source_commit"] = self.sibling_commit
        review["implementation_source_commit"] = self.sibling_commit
        with (
            mock.patch.object(
                gate,
                "read_r2r1_candidate_selection",
                return_value=(self.selection, DIGEST),
            ),
            mock.patch.object(
                gate,
                "read_r2r1_review_input_freeze",
                return_value=(freeze, OTHER_DIGEST),
            ),
            mock.patch.object(
                gate,
                "read_r2r1_fresh_review_evidence",
                return_value=(evidence, "5" * 64),
            ),
            self.assertRaises(gate.Gate12C2OriginalBaselineError),
        ):
            gate.build_reviewed_authority_payload(
                self.plan,
                candidate,
                review,
                candidate_file_sha256=DIGEST,
                review_file_sha256=OTHER_DIGEST,
            )
        with (
            mock.patch.object(
                independent,
                "_independent_r2r1_candidate_selection",
                return_value=(self.selection, DIGEST),
            ),
            mock.patch.object(
                independent,
                "_independent_r2r1_review_input_freeze",
                return_value=(freeze, OTHER_DIGEST),
            ),
            mock.patch.object(
                independent,
                "_independent_r2r1_review_evidence",
                return_value=(evidence, "5" * 64),
            ),
            self.assertRaises(independent.IndependentVerificationError),
        ):
            independent._independent_reviewed_authority_payload(
                self.independent_plan,
                candidate,
                review,
                candidate_file_hash=DIGEST,
                review_file_hash=OTHER_DIGEST,
            )

    def test_selection_commit_mutation_and_self_rehash_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.selection)
        mutated["exact_candidate_commit"] = self.sibling_commit
        mutated = self._rehash(mutated, "candidate_selection_payload_sha256")
        with self._core_selection_mocks():
            with self.assertRaises(gate.Gate12C2OriginalBaselineError):
                gate.validate_r2r1_candidate_selection(
                    self.plan,
                    mutated,
                    repo_root=REPOSITORY,
                    current_head=self.selection["exact_candidate_commit"],
                )

    def test_selection_and_candidate_mixed_lineage_is_rejected(self) -> None:
        candidate, review, freeze, evidence = (
            self._candidate_review_freeze_evidence()
        )
        candidate["candidate_selection_payload_sha256"] = "7" * 64
        review["candidate_selection_payload_sha256"] = "7" * 64
        with (
            mock.patch.object(
                gate,
                "read_r2r1_candidate_selection",
                return_value=(self.selection, DIGEST),
            ),
            mock.patch.object(
                gate,
                "read_r2r1_review_input_freeze",
                return_value=(freeze, OTHER_DIGEST),
            ),
            mock.patch.object(
                gate,
                "read_r2r1_fresh_review_evidence",
                return_value=(evidence, "5" * 64),
            ),
            self.assertRaises(gate.Gate12C2OriginalBaselineError),
        ):
            gate.build_reviewed_authority_payload(
                self.plan,
                candidate,
                review,
                candidate_file_sha256=DIGEST,
                review_file_sha256=OTHER_DIGEST,
            )

    def test_targeted_count_one_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.selection)
        mutated["targeted_test_count"] = 1
        mutated = self._rehash(mutated, "candidate_selection_payload_sha256")
        with self._core_selection_mocks():
            with self.assertRaises(gate.Gate12C2OriginalBaselineError):
                gate.validate_r2r1_candidate_selection(
                    self.plan, mutated, repo_root=REPOSITORY
                )
        with self._independent_selection_mocks(mutated):
            with self.assertRaises(independent.IndependentVerificationError):
                independent._independent_r2r1_candidate_selection(
                    self.independent_plan, REPOSITORY
                )

    def test_full_suite_count_one_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.selection)
        mutated["full_suite_test_count"] = 1
        mutated = self._rehash(mutated, "candidate_selection_payload_sha256")
        with self._core_selection_mocks():
            with self.assertRaises(gate.Gate12C2OriginalBaselineError):
                gate.validate_r2r1_candidate_selection(
                    self.plan, mutated, repo_root=REPOSITORY
                )
        with self._independent_selection_mocks(mutated):
            with self.assertRaises(independent.IndependentVerificationError):
                independent._independent_r2r1_candidate_selection(
                    self.independent_plan, REPOSITORY
                )

    def test_random_and_all_zero_node_digests_are_rejected(self) -> None:
        for field in (
            "targeted_test_node_id_sha256",
            "full_suite_test_node_id_sha256",
        ):
            for value in ("0" * 64, "9" * 64):
                with self.subTest(field=field, value=value[:1]):
                    mutated = copy.deepcopy(self.selection)
                    mutated[field] = value
                    mutated = self._rehash(
                        mutated, "candidate_selection_payload_sha256"
                    )
                    with self._core_selection_mocks():
                        with self.assertRaises(
                            gate.Gate12C2OriginalBaselineError
                        ):
                            gate.validate_r2r1_candidate_selection(
                                self.plan, mutated, repo_root=REPOSITORY
                            )
                    with self._independent_selection_mocks(mutated):
                        with self.assertRaises(
                            independent.IndependentVerificationError
                        ):
                            independent._independent_r2r1_candidate_selection(
                                self.independent_plan, REPOSITORY
                            )

    def test_changed_file_manifest_substitution_is_rejected(self) -> None:
        mutated = copy.deepcopy(self.selection)
        mutated["changed_file_manifest_sha256"] = "8" * 64
        mutated = self._rehash(mutated, "candidate_selection_payload_sha256")
        with self._core_selection_mocks():
            with self.assertRaises(gate.Gate12C2OriginalBaselineError):
                gate.validate_r2r1_candidate_selection(
                    self.plan, mutated, repo_root=REPOSITORY
                )
        with self._independent_selection_mocks(mutated):
            with self.assertRaises(independent.IndependentVerificationError):
                independent._independent_r2r1_candidate_selection(
                    self.independent_plan, REPOSITORY
                )

    def test_packet_hash_substitution_and_downstream_rehash_is_rejected(self) -> None:
        candidate, _review, freeze, _evidence = (
            self._candidate_review_freeze_evidence()
        )
        mutated = copy.deepcopy(freeze)
        mutated["review_packet_file_sha256"] = "d" * 64
        mutated = self._rehash(mutated, "review_input_freeze_payload_sha256")
        manifest = {
            "candidate_manifest_payload_sha256": candidate[
                "candidate_manifest_payload_sha256"
            ]
        }
        restore = {
            "restore_receipt_payload_sha256": self.restore[
                "restore_receipt_payload_sha256"
            ]
        }
        with (
            mock.patch.object(Path, "read_bytes", return_value=b"packet\n"),
            mock.patch.object(
                gate,
                "read_r2r1_candidate_manifest",
                return_value=(manifest, candidate["candidate_manifest_file_sha256"]),
            ),
            mock.patch.object(
                gate,
                "read_r2r1_clean_restore_receipt",
                return_value=(restore, DIGEST),
            ),
            self.assertRaises(gate.Gate12C2OriginalBaselineError),
        ):
            gate.validate_r2r1_review_input_freeze(
                self.plan,
                mutated,
                candidate=candidate,
                candidate_file_sha256=DIGEST,
                selection=self.selection,
                selection_file_sha256=DIGEST,
            )
        with (
            mock.patch.object(
                independent, "_receipt", return_value=(mutated, DIGEST)
            ),
            mock.patch.object(Path, "read_bytes", return_value=b"packet\n"),
            mock.patch.object(
                independent,
                "_independent_r2r1_candidate_manifest",
                return_value=(
                    manifest,
                    candidate["candidate_manifest_file_sha256"],
                ),
            ),
            mock.patch.object(
                independent,
                "_independent_r2r1_restore_receipt",
                return_value=(restore, DIGEST),
            ),
            self.assertRaises(independent.IndependentVerificationError),
        ):
            independent._independent_r2r1_review_input_freeze(
                self.independent_plan,
                candidate,
                DIGEST,
                self.selection,
                DIGEST,
            )

    def test_packet_one_byte_mutation_is_rejected(self) -> None:
        candidate, _review, freeze, _evidence = (
            self._candidate_review_freeze_evidence()
        )
        manifest = {
            "candidate_manifest_payload_sha256": candidate[
                "candidate_manifest_payload_sha256"
            ]
        }
        restore = {
            "restore_receipt_payload_sha256": self.restore[
                "restore_receipt_payload_sha256"
            ]
        }
        with (
            mock.patch.object(
                Path, "read_bytes", return_value=b"r2r1-review-packeu\n"
            ),
            mock.patch.object(
                gate,
                "read_r2r1_candidate_manifest",
                return_value=(manifest, candidate["candidate_manifest_file_sha256"]),
            ),
            mock.patch.object(
                gate,
                "read_r2r1_clean_restore_receipt",
                return_value=(restore, DIGEST),
            ),
            self.assertRaises(gate.Gate12C2OriginalBaselineError),
        ):
            gate.validate_r2r1_review_input_freeze(
                self.plan,
                freeze,
                candidate=candidate,
                candidate_file_sha256=DIGEST,
                selection=self.selection,
                selection_file_sha256=DIGEST,
            )
        with (
            mock.patch.object(
                independent, "_receipt", return_value=(freeze, DIGEST)
            ),
            mock.patch.object(
                Path, "read_bytes", return_value=b"r2r1-review-packeu\n"
            ),
            mock.patch.object(
                independent,
                "_independent_r2r1_candidate_manifest",
                return_value=(
                    manifest,
                    candidate["candidate_manifest_file_sha256"],
                ),
            ),
            mock.patch.object(
                independent,
                "_independent_r2r1_restore_receipt",
                return_value=(restore, DIGEST),
            ),
            self.assertRaises(independent.IndependentVerificationError),
        ):
            independent._independent_r2r1_review_input_freeze(
                self.independent_plan,
                candidate,
                DIGEST,
                self.selection,
                DIGEST,
            )

    def test_valid_evidence_independent_verification_succeeds(self) -> None:
        candidate, review, freeze, evidence = (
            self._candidate_review_freeze_evidence()
        )
        with mock.patch.object(
            independent,
            "_receipt",
            return_value=(evidence, "5" * 64),
        ):
            supplied, file_hash = independent._independent_r2r1_review_evidence(
                self.independent_plan,
                candidate,
                DIGEST,
                self.selection,
                DIGEST,
                freeze,
                OTHER_DIGEST,
            )
        self.assertEqual(supplied, evidence)
        self.assertEqual(file_hash, "5" * 64)
        historical_plan = independent._independent_load_r2_active()
        historical_contract = historical_plan["r2_activation_control"][
            "fresh_review_evidence_contract"
        ]
        historical_candidate = {
            "source_commit": "a" * 40,
            "implementation_candidate_binding_payload_sha256": "1" * 64,
            "candidate_manifest_file_sha256": "2" * 64,
            "candidate_manifest_payload_sha256": "3" * 64,
            "review_surface_identity": independent._review_surface_identity(
                historical_plan
            ),
            "clean_restore": {
                "bundle_file_sha256": "4" * 64,
                "restore_receipt_file_sha256": "5" * 64,
                "restore_receipt_payload_sha256": "6" * 64,
            },
        }
        historical_evidence = {
            field: None
            for field in historical_contract["exact_top_level_fields"]
        }
        historical_evidence.update(historical_contract["required_values"])
        historical_evidence.update(
            {
                "implementation_source_commit": "a" * 40,
                "implementation_candidate_binding_file_sha256": DIGEST,
                "implementation_candidate_binding_payload_sha256": "1" * 64,
                "candidate_manifest_file_sha256": "2" * 64,
                "candidate_manifest_payload_sha256": "3" * 64,
                "r2_activation_plan_file_sha256": independent.R2_PLAN_FILE_SHA256,
                "r2_activation_plan_payload_sha256": (
                    independent.R2_PLAN_PAYLOAD_SHA256
                ),
                "artifact_path_surface_sha256": (
                    independent.R2_ARTIFACT_SURFACE_SHA256
                ),
                "review_surface_identity": (
                    independent._review_surface_identity(historical_plan)
                ),
                "bundle_file_sha256": "4" * 64,
                "restore_receipt_file_sha256": "5" * 64,
                "restore_receipt_payload_sha256": "6" * 64,
                "changed_file_manifest_sha256": "7" * 64,
                "targeted_test_count": 2,
                "targeted_test_node_id_sha256": "8" * 64,
                "full_suite_test_count": 2,
                "full_suite_node_id_sha256": "9" * 64,
            }
        )
        historical_evidence.pop("review_evidence_payload_sha256", None)
        historical_evidence = gate.add_self_hash(
            historical_evidence, "review_evidence_payload_sha256"
        )
        with mock.patch.object(
            independent,
            "_receipt",
            return_value=(historical_evidence, DIGEST),
        ):
            historical_validated, _ = independent._independent_r2_review_evidence(
                historical_plan, historical_candidate, DIGEST
            )
        self.assertEqual(historical_validated, historical_evidence)
        with (
            mock.patch.object(
                gate,
                "validate_r2r1_candidate_selection",
                return_value=self.selection,
            ),
            mock.patch.object(
                gate,
                "validate_r2r1_review_input_freeze",
                return_value=freeze,
            ),
            mock.patch.object(
                gate,
                "validate_r2r1_fresh_review_evidence",
                return_value=evidence,
            ),
        ):
            validated_review = gate.validate_implementation_review(
                self.plan,
                review,
                candidate_file_sha256=DIGEST,
                candidate_payload_sha256=candidate[
                    "implementation_candidate_binding_payload_sha256"
                ],
                source_commit=candidate["source_commit"],
                candidate=candidate,
                review_evidence=evidence,
                review_evidence_file_sha256="5" * 64,
                candidate_selection=self.selection,
                candidate_selection_file_sha256=DIGEST,
                review_input_freeze=freeze,
                review_input_freeze_file_sha256=OTHER_DIGEST,
            )
        self.assertEqual(validated_review, review)
        with (
            mock.patch.object(
                gate,
                "read_r2r1_candidate_selection",
                return_value=(self.selection, DIGEST),
            ),
            mock.patch.object(
                gate,
                "read_r2r1_review_input_freeze",
                return_value=(freeze, OTHER_DIGEST),
            ),
            mock.patch.object(
                gate,
                "read_r2r1_fresh_review_evidence",
                return_value=(evidence, "5" * 64),
            ),
        ):
            core_authority = gate.build_reviewed_authority_payload(
                self.plan,
                candidate,
                review,
                candidate_file_sha256=DIGEST,
                review_file_sha256=OTHER_DIGEST,
            )
        with (
            mock.patch.object(
                independent,
                "_independent_r2r1_candidate_selection",
                return_value=(self.selection, DIGEST),
            ),
            mock.patch.object(
                independent,
                "_independent_r2r1_review_input_freeze",
                return_value=(freeze, OTHER_DIGEST),
            ),
            mock.patch.object(
                independent,
                "_independent_r2r1_review_evidence",
                return_value=(evidence, "5" * 64),
            ),
        ):
            independent_authority = (
                independent._independent_reviewed_authority_payload(
                    self.independent_plan,
                    candidate,
                    review,
                    candidate_file_hash=DIGEST,
                    review_file_hash=OTHER_DIGEST,
                )
            )
        self.assertEqual(independent_authority, core_authority)

    def test_old_v09_r2_and_r2r1_namespace_mix_is_rejected(self) -> None:
        candidate, review, freeze, evidence = (
            self._candidate_review_freeze_evidence()
        )
        review["authority_namespace_id"] = gate.R2_AUTHORITY_NAMESPACE_ID
        with (
            mock.patch.object(
                gate,
                "read_r2r1_candidate_selection",
                return_value=(self.selection, DIGEST),
            ),
            mock.patch.object(
                gate,
                "read_r2r1_review_input_freeze",
                return_value=(freeze, OTHER_DIGEST),
            ),
            mock.patch.object(
                gate,
                "read_r2r1_fresh_review_evidence",
                return_value=(evidence, "5" * 64),
            ),
            self.assertRaises(gate.Gate12C2OriginalBaselineError),
        ):
            gate.build_reviewed_authority_payload(
                self.plan,
                candidate,
                review,
                candidate_file_sha256=DIGEST,
                review_file_sha256=OTHER_DIGEST,
            )

    def test_phase_a_plan_load_has_zero_protected_root_reads(self) -> None:
        original = Path.read_bytes
        protected = str(gate.PROTECTED_ROOT).casefold()

        def guarded(path: Path) -> bytes:
            if str(path).casefold().startswith(protected):
                raise AssertionError("protected root read during R2R1 Phase A")
            return original(path)

        with mock.patch.object(Path, "read_bytes", guarded):
            core_plan = gate.load_active_plan()
            verifier_plan = independent.independent_load_plan()
        self.assertEqual(
            core_plan["artifact_path_surface_sha256"],
            verifier_plan["artifact_path_surface_sha256"],
        )


class R2R2PortabilityAndFramingAdversarialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = gate.load_active_plan(repository_root=REPOSITORY)
        cls.independent_plan = independent.independent_load_plan(
            repository_root=REPOSITORY
        )

    @staticmethod
    def _copy_repository_plans(root: Path) -> None:
        target = root / "tools"
        target.mkdir(parents=True)
        for relative in (
            gate.R2_ACTIVATION_PLAN_RELATIVE_PATH,
            gate.R2R1_REMEDIATION_PLAN_RELATIVE_PATH,
            gate.R2R2_PORTABILITY_PLAN_RELATIVE_PATH,
            gate.R2R5_RUNNER_RELATIVE_PATH,
        ):
            shutil.copyfile(REPOSITORY / relative, root / relative)

    @staticmethod
    def _framed_payload(
        domain: str, *, value: int = 1
    ) -> tuple[dict[str, Any], bytes, str]:
        suffix = (
            b""
            if domain == gate.FROZEN_JSON_WITHOUT_LF
            else b"\n"
        )
        unhashed = {"schema_version": "r2r2_test_v0.1", "value": value}
        payload_hash = gate.sha256_bytes(
            gate.canonical_json_bytes(unhashed) + suffix
        )
        payload = {
            **unhashed,
            "payload_sha256": payload_hash,
        }
        return (
            payload,
            gate.canonical_json_bytes(payload) + suffix,
            payload_hash,
        )

    def _assert_core_and_independent_framing(
        self, raw: bytes, domain: str, payload_hash: str
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            path.write_bytes(raw)
            file_hash = gate.sha256_bytes(raw)
            core = gate.read_declared_frozen_json_artifact(
                path,
                expected_file_sha256=file_hash,
                expected_payload_sha256=payload_hash,
                payload_hash_domain=domain,
                self_hash_field="payload_sha256",
                expected_schema_version="r2r2_test_v0.1",
            )
            row = {
                "file_sha256": file_hash,
                "format": "canonical_self_hashed_JSON",
                "path": str(path),
                "payload_hash_domain": domain,
                "payload_sha256": payload_hash,
                "role": "test",
                "schema_version": "r2r2_test_v0.1",
                "self_hash_field": "payload_sha256",
            }
            independently = independent._independent_upstream_artifact(row)
            self.assertEqual(core, independently)

    def _assert_framing_rejected(
        self,
        raw: bytes,
        domain: str,
        payload_hash: str,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            path.write_bytes(raw)
            file_hash = gate.sha256_bytes(raw)
            with self.assertRaises(gate.Gate12C2OriginalBaselineError):
                gate.read_declared_frozen_json_artifact(
                    path,
                    expected_file_sha256=file_hash,
                    expected_payload_sha256=payload_hash,
                    payload_hash_domain=domain,
                    self_hash_field="payload_sha256",
                    expected_schema_version="r2r2_test_v0.1",
                )
            row = {
                "file_sha256": file_hash,
                "format": "canonical_self_hashed_JSON",
                "path": str(path),
                "payload_hash_domain": domain,
                "payload_sha256": payload_hash,
                "role": "test",
                "schema_version": "r2r2_test_v0.1",
                "self_hash_field": "payload_sha256",
            }
            with self.assertRaises(
                independent.IndependentVerificationError
            ):
                independent._independent_upstream_artifact(row)

    def test_source_checkout_and_two_restore_locations_are_identical(
        self,
    ) -> None:
        self.assertEqual(
            gate.canonical_json_bytes(self.plan),
            independent.verifier_canonical_bytes(self.independent_plan),
        )
        materialized: list[bytes] = []
        independent_materialized: list[bytes] = []
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            for directory in (first, second):
                root = Path(directory)
                self._copy_repository_plans(root)

                def core_blob(
                    _repo: Path, _commit: str, relative: str, **_kwargs: Any
                ) -> str:
                    return {
                        gate.R2_ACTIVATION_PLAN_RELATIVE_PATH: (
                            gate.R2_ACTIVATION_PLAN_BASE_BLOB_OID
                        ),
                        gate.R2R1_REMEDIATION_PLAN_RELATIVE_PATH: (
                            gate.R2R1_REMEDIATION_PLAN_BASE_BLOB_OID
                        ),
                        gate.R2R5_RUNNER_RELATIVE_PATH: (
                            "ca49cd0850202e718d8f028a8d74b3cb0bb64c15"
                        ),
                    }[relative]

                def verifier_blob(
                    _repo: Path, _commit: str, relative: str
                ) -> str:
                    return {
                        independent.R2_PLAN_RELATIVE_PATH: (
                            independent.R2_PLAN_BASE_BLOB_OID
                        ),
                        independent.R2R1_PLAN_RELATIVE_PATH: (
                            independent.R2R1_PLAN_BASE_BLOB_OID
                        ),
                        independent.R2R5_RUNNER_RELATIVE_PATH: (
                            "ca49cd0850202e718d8f028a8d74b3cb0bb64c15"
                        ),
                    }[relative]

                with mock.patch.object(
                    gate, "git_path_blob_oid", side_effect=core_blob
                ):
                    reconstructed = gate.load_active_plan(
                        repository_root=root
                    )
                with mock.patch.object(
                    independent,
                    "_git_path_blob_oid",
                    side_effect=verifier_blob,
                ):
                    independently = independent.independent_load_plan(
                        repository_root=root
                    )
                materialized.append(gate.canonical_json_bytes(reconstructed))
                independent_materialized.append(
                    independent.verifier_canonical_bytes(independently)
                )
        self.assertEqual(materialized[0], materialized[1])
        self.assertEqual(
            independent_materialized[0], independent_materialized[1]
        )
        self.assertEqual(materialized, independent_materialized)
        row = {
            item["role"]: item
            for item in self.plan["r2r2_portability_control"][
                "repository_local_artifacts"
            ]
        }["r2_activation_plan"]
        self.assertEqual(
            row["historical_declared_path"],
            str(gate.R2_ACTIVATION_PLAN_HISTORICAL_DECLARED_PATH),
        )
        self.assertNotIn(first, materialized[0].decode("utf-8"))
        self.assertNotIn(second, materialized[1].decode("utf-8"))

    def test_explicit_root_ignores_import_checkout_r2_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._copy_repository_plans(root)
            blocked = {
                str(gate.R2_ACTIVATION_PLAN_PATH).casefold(),
                str(independent.R2_PLAN_PATH).casefold(),
            }
            original = Path.read_bytes

            def guarded(path: Path) -> bytes:
                if str(path).casefold() in blocked:
                    raise AssertionError(
                        "import-checkout R2 activation plan read"
                    )
                return original(path)

            def core_blob(
                _repo: Path, _commit: str, relative: str, **_kwargs: Any
            ) -> str:
                return {
                    gate.R2_ACTIVATION_PLAN_RELATIVE_PATH: (
                        gate.R2_ACTIVATION_PLAN_BASE_BLOB_OID
                    ),
                    gate.R2R1_REMEDIATION_PLAN_RELATIVE_PATH: (
                        gate.R2R1_REMEDIATION_PLAN_BASE_BLOB_OID
                    ),
                    gate.R2R5_RUNNER_RELATIVE_PATH: (
                        "ca49cd0850202e718d8f028a8d74b3cb0bb64c15"
                    ),
                }[relative]

            def verifier_blob(
                _repo: Path, _commit: str, relative: str
            ) -> str:
                return {
                    independent.R2_PLAN_RELATIVE_PATH: (
                        independent.R2_PLAN_BASE_BLOB_OID
                    ),
                    independent.R2R1_PLAN_RELATIVE_PATH: (
                        independent.R2R1_PLAN_BASE_BLOB_OID
                    ),
                    independent.R2R5_RUNNER_RELATIVE_PATH: (
                        "ca49cd0850202e718d8f028a8d74b3cb0bb64c15"
                    ),
                }[relative]

            with mock.patch.object(Path, "read_bytes", guarded):
                with mock.patch.object(
                    gate, "git_path_blob_oid", side_effect=core_blob
                ):
                    core_plan = gate.load_active_plan(
                        repository_root=root
                    )
                with mock.patch.object(
                    independent,
                    "_git_path_blob_oid",
                    side_effect=verifier_blob,
                ):
                    independent_plan = independent.independent_load_plan(
                        repository_root=root
                    )
        self.assertEqual(
            gate.canonical_json_bytes(core_plan),
            independent.verifier_canonical_bytes(independent_plan),
        )

    def test_all_r2r2_roles_use_isolated_namespace_paths(self) -> None:
        rows = self.plan["artifact_path_surface"]
        current_paths = {
            value
            for row in rows
            for value in (row["final_path"], row["pending_path"])
        }
        self.assertEqual(len(rows), 18)
        self.assertEqual(len(current_paths), 36)
        for row in rows:
            self.assertIn(
                gate.R2R2_AUTHORITY_NAMESPACE_ID,
                Path(row["final_path"]).name,
            )
            self.assertEqual(
                row["pending_path"],
                row["final_path"]
                + ".pending-"
                + gate.R2R2_AUTHORITY_NAMESPACE_ID,
            )
        formal = next(
            row
            for row in rows
            if row["role"] == "formal_design_review_verdict"
        )
        self.assertNotEqual(
            formal["final_path"], str(gate.FORMAL_DESIGN_REVIEW_PATH)
        )
        r2 = gate.load_r2_active_plan(repository_root=REPOSITORY)
        r2r1 = gate.build_r2r1_active_plan(
            r2,
            gate.load_r2r1_remediation_plan(
                repository_root=REPOSITORY, r2_active_plan=r2
            ),
            repository_root=REPOSITORY,
        )
        for historical in (gate.load_frozen_plan(), r2, r2r1):
            historical_paths = {
                value
                for row in historical["artifact_path_surface"]
                for value in (row["final_path"], row["pending_path"])
            }
            self.assertTrue(current_paths.isdisjoint(historical_paths))
        with mock.patch.object(gate, "atomic_publish_exact") as publish:
            with self.assertRaises(gate.Gate12C2OriginalBaselineError):
                gate.publish_role(
                    self.plan, "formal_design_review_verdict", {}
                )
            publish.assert_not_called()

    def test_repository_path_attacks_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "tools" / "bound.json"
            target.parent.mkdir()
            target.write_bytes(b"{}\n")
            file_hash = gate.sha256_bytes(target.read_bytes())
            historical = Path(r"C:\historical\bound.json")
            arguments = {
                "historical_declared_path": str(historical),
                "expected_historical_declared_path": historical,
                "canonical_repository_relative_path": "tools/bound.json",
                "expected_file_sha256": file_hash,
                "bound_commit": "a" * 40,
                "expected_git_blob_oid": "b" * 40,
            }
            with mock.patch.object(
                gate, "git_path_blob_oid", return_value="b" * 40
            ):
                self.assertEqual(
                    gate.validate_repository_local_artifact(
                        root, **arguments
                    ),
                    b"{}\n",
                )
            for override in (
                str(target),
                "../bound.json",
                "tools/../bound.json",
                "C:/bound.json",
            ):
                mutated = dict(arguments)
                mutated["canonical_repository_relative_path"] = override
                with (
                    mock.patch.object(
                        gate, "git_path_blob_oid", return_value="b" * 40
                    ),
                    self.assertRaises(
                        gate.Gate12C2OriginalBaselineError
                    ),
                ):
                    gate.validate_repository_local_artifact(
                        root, **mutated
                    )
            with (
                mock.patch.object(
                    gate, "git_path_blob_oid", return_value="c" * 40
                ),
                self.assertRaises(gate.Gate12C2OriginalBaselineError),
            ):
                gate.validate_repository_local_artifact(root, **arguments)
            original_lstat = gate.os.lstat

            def reparse(path: Path) -> Any:
                value = original_lstat(path)
                if Path(path).name == "tools":
                    return types.SimpleNamespace(
                        st_file_attributes=0x400
                    )
                return value

            with (
                mock.patch.object(gate.os, "lstat", side_effect=reparse),
                mock.patch.object(
                    gate, "git_path_blob_oid", return_value="b" * 40
                ),
                self.assertRaises(gate.Gate12C2OriginalBaselineError),
            ):
                gate.validate_repository_local_artifact(root, **arguments)

    def test_exact_no_lf_and_single_lf_domains_pass(self) -> None:
        for domain, value in (
            (gate.FROZEN_JSON_WITHOUT_LF, 1),
            (gate.FROZEN_JSON_WITHOUT_LF, 2),
            (gate.FROZEN_JSON_WITH_SINGLE_LF, 3),
        ):
            with self.subTest(domain=domain, value=value):
                _payload, raw, payload_hash = self._framed_payload(
                    domain, value=value
                )
                self._assert_core_and_independent_framing(
                    raw, domain, payload_hash
                )

    def test_framing_and_canonicalization_attacks_are_rejected(self) -> None:
        payload, no_lf, no_lf_hash = self._framed_payload(
            gate.FROZEN_JSON_WITHOUT_LF
        )
        _single_payload, single_lf, single_hash = self._framed_payload(
            gate.FROZEN_JSON_WITH_SINGLE_LF
        )
        reordered = (
            b'{"value":1,"schema_version":"r2r2_test_v0.1",'
            b'"payload_sha256":"'
            + no_lf_hash.encode("ascii")
            + b'"}'
        )
        duplicate = (
            b'{"payload_sha256":"'
            + no_lf_hash.encode("ascii")
            + b'","schema_version":"r2r2_test_v0.1",'
            b'"value":1,"value":1}'
        )
        mismatched = dict(payload)
        mismatched["payload_sha256"] = "f" * 64
        cases = (
            (
                "added_lf",
                no_lf + b"\n",
                gate.FROZEN_JSON_WITHOUT_LF,
                no_lf_hash,
            ),
            (
                "missing_lf",
                single_lf[:-1],
                gate.FROZEN_JSON_WITH_SINGLE_LF,
                single_hash,
            ),
            (
                "crlf",
                single_lf[:-1] + b"\r\n",
                gate.FROZEN_JSON_WITH_SINGLE_LF,
                single_hash,
            ),
            (
                "double_lf",
                single_lf + b"\n",
                gate.FROZEN_JSON_WITH_SINGLE_LF,
                single_hash,
            ),
            (
                "reordered",
                reordered,
                gate.FROZEN_JSON_WITHOUT_LF,
                no_lf_hash,
            ),
            (
                "duplicate",
                duplicate,
                gate.FROZEN_JSON_WITHOUT_LF,
                no_lf_hash,
            ),
            (
                "self_hash_mismatch",
                gate.canonical_json_bytes(mismatched),
                gate.FROZEN_JSON_WITHOUT_LF,
                no_lf_hash,
            ),
        )
        for name, raw, domain, payload_hash in cases:
            with self.subTest(name=name):
                self._assert_framing_rejected(
                    raw, domain, payload_hash
                )
        with self.assertRaises(gate.Gate12C2OriginalBaselineError):
            gate.read_declared_frozen_json_artifact(
                Path(__file__),
                expected_file_sha256=gate.sha256_bytes(
                    Path(__file__).read_bytes()
                ),
                expected_payload_sha256=no_lf_hash,
                payload_hash_domain="unknown",
                self_hash_field="payload_sha256",
                expected_schema_version="r2r2_test_v0.1",
            )

    def test_old_authority_is_not_r2r2_authority(self) -> None:
        old = self.plan["r2r2_portability_control"]
        occupied_plan = gate.load_r2r2_portability_plan(
            repository_root=REPOSITORY,
            r2r1_active_plan=gate.build_r2r1_active_plan(
                gate.load_r2_active_plan(repository_root=REPOSITORY),
                gate.load_r2r1_remediation_plan(
                    repository_root=REPOSITORY
                ),
                repository_root=REPOSITORY,
            ),
        )
        authority_row = occupied_plan["occupied_r2r1"][
            "reviewed_authority"
        ]
        with self.assertRaises(gate.Gate12C2OriginalBaselineError):
            gate.read_schema_receipt(
                Path(authority_row["path"]),
                exact_fields=gate.artifact_exact_fields(
                    self.plan, "reviewed_implementation_authority"
                ),
                hash_field=(
                    "reviewed_implementation_authority_payload_sha256"
                ),
            )
        self.assertEqual(
            old["authority_namespace_id"],
            gate.R2R2_AUTHORITY_NAMESPACE_ID,
        )

    def test_core_independent_lineage_has_no_protected_read(self) -> None:
        original = Path.read_bytes
        protected = str(gate.PROTECTED_ROOT).casefold()

        def guarded(path: Path) -> bytes:
            if str(path).casefold().startswith(protected):
                raise AssertionError("protected root read during R2R2 Phase A")
            return original(path)

        with mock.patch.object(Path, "read_bytes", guarded):
            core_plan = gate.load_active_plan(repository_root=REPOSITORY)
            independent_plan = independent.independent_load_plan(
                repository_root=REPOSITORY
            )
        self.assertEqual(
            gate.canonical_json_bytes(core_plan),
            independent.verifier_canonical_bytes(independent_plan),
        )

    def test_core_and_independent_runtime_lineage_accept_exact_valid_fixture(
        self,
    ) -> None:
        plan = self.plan
        control = gate.active_remediation_control(plan)
        identity = gate.active_remediation_identity(plan)
        binding = plan["implementation_binding_contract"]
        source_commit = "d" * 40
        parent_commit = identity["parent_commit"]
        grandparent_commit = identity["grandparent_commit"]
        object_format = "sha1"
        implementation_rows = []
        for relative in gate.IMPLEMENTATION_PATHS:
            role = gate.IMPLEMENTATION_ROLE_BY_PATH[relative]
            raw = (REPOSITORY / relative).read_bytes()
            implementation_rows.append(
                {
                    "role": role,
                    "relative_path": relative,
                    "file_sha256": gate.sha256_bytes(raw),
                    "git_blob_oid": gate.git_blob_oid(raw, object_format),
                }
            )
        implementation_rows.sort(key=lambda row: row["role"])
        changed_rows = []
        for relative in control["allowed_changed_paths"]:
            raw = (REPOSITORY / relative).read_bytes()
            changed_rows.append(
                {
                    "relative_path": relative,
                    "file_sha256": gate.sha256_bytes(raw),
                    "git_blob_oid": gate.git_blob_oid(raw, object_format),
                }
            )
        changed_rows.sort(key=lambda row: row["relative_path"])
        changed_digest = gate.sha256_bytes(
            gate.canonical_json_bytes(changed_rows)
        )
        coverage = control["review_coverage_identity"]
        candidate_file_hash = "1" * 64
        manifest_file_hash = "2" * 64
        manifest_payload_hash = "3" * 64
        selection_file_hash = "4" * 64
        freeze_file_hash = "5" * 64
        evidence_file_hash = "6" * 64
        review_file_hash = "7" * 64
        restore_file_hash = "8" * 64
        packet_hash = "9" * 64
        with tempfile.TemporaryDirectory() as directory:
            bundle_path = Path(directory) / "r2r2-synthetic.bundle"
            bundle_raw = b"r2r2-synthetic-bundle\n"
            bundle_path.write_bytes(bundle_raw)
            restore_contract = control["clean_restore_receipt_contract"]
            restore_receipt = {
                field: None
                for field in restore_contract["exact_top_level_fields"]
            }
            restore_receipt.update(restore_contract["required_values"])
            restore_receipt.update(
                {
                    "source_commit": source_commit,
                    "source_parent_commit": parent_commit,
                    "source_grandparent_commit": grandparent_commit,
                    "bundle_path": str(bundle_path),
                    "bundle_file_sha256": gate.sha256_bytes(bundle_raw),
                    "bundle_size_bytes": len(bundle_raw),
                    "restore_path": str(Path(directory) / "restore"),
                    "restore_head": source_commit,
                    "targeted_test_count": coverage["targeted_test_count"],
                    "targeted_test_node_id_sha256": coverage[
                        "targeted_test_node_id_sha256"
                    ],
                    "full_suite_test_count": coverage[
                        "full_suite_test_count"
                    ],
                    "full_suite_test_node_id_sha256": coverage[
                        "full_suite_test_node_id_sha256"
                    ],
                    **identity["static_fields"],
                }
            )
            restore_receipt.pop("restore_receipt_payload_sha256", None)
            restore_receipt = gate.add_self_hash(
                restore_receipt, "restore_receipt_payload_sha256"
            )
            selection_contract = control["candidate_selection_contract"]
            selection = {
                field: None
                for field in selection_contract["exact_top_level_fields"]
            }
            selection.update(selection_contract["required_values"])
            selection.update(
                {
                    "exact_candidate_commit": source_commit,
                    "git_object_format": object_format,
                    "changed_path_allowlist": control[
                        "allowed_changed_paths"
                    ],
                    "changed_files": changed_rows,
                    "changed_file_manifest_sha256": changed_digest,
                    "implementation_files": implementation_rows,
                    "r2_activation_plan_file_sha256": (
                        gate.R2_ACTIVATION_PLAN_FILE_SHA256
                    ),
                    "r2_activation_plan_payload_sha256": (
                        gate.R2_ACTIVATION_PLAN_PAYLOAD_SHA256
                    ),
                    "artifact_path_surface_sha256": (
                        gate.artifact_surface_sha256(plan)
                    ),
                    "review_surface_identity_sha256": (
                        gate.REVIEW_SURFACE_IDENTITY_SHA256
                    ),
                    "implementation_trust_model_sha256": (
                        gate.recompute_implementation_trust_model_sha256(plan)
                    ),
                    "bundle_path": str(bundle_path),
                    "bundle_file_sha256": gate.sha256_bytes(bundle_raw),
                    "bundle_size_bytes": len(bundle_raw),
                    "clean_restore_receipt_file_sha256": restore_file_hash,
                    "clean_restore_receipt_payload_sha256": restore_receipt[
                        "restore_receipt_payload_sha256"
                    ],
                    "targeted_test_count": coverage[
                        "targeted_test_count"
                    ],
                    "targeted_test_node_id_sha256": coverage[
                        "targeted_test_node_id_sha256"
                    ],
                    "full_suite_test_count": coverage[
                        "full_suite_test_count"
                    ],
                    "full_suite_test_node_id_sha256": coverage[
                        "full_suite_test_node_id_sha256"
                    ],
                    **identity["static_fields"],
                }
            )
            selection.pop("candidate_selection_payload_sha256", None)
            selection = gate.add_self_hash(
                selection, "candidate_selection_payload_sha256"
            )
            clean_restore = {
                "bundle_path": str(bundle_path),
                "bundle_file_sha256": gate.sha256_bytes(bundle_raw),
                "bundle_size_bytes": len(bundle_raw),
                "restore_receipt_file_sha256": restore_file_hash,
                "restore_receipt_payload_sha256": restore_receipt[
                    "restore_receipt_payload_sha256"
                ],
                "restore_head": source_commit,
                "restore_worktree_clean": True,
                "git_fsck_full_pass": True,
                "core_autocrlf": False,
                "core_longpaths": True,
                "implementation_rows_match": True,
                "scientific_dependency_rows_match": True,
            }
            candidate = {
                field: None for field in binding["exact_top_level_fields"]
            }
            candidate.update(binding["required_values"])
            candidate.update(
                {
                    "source_commit": source_commit,
                    "git_object_format": object_format,
                    "core_autocrlf": False,
                    "core_longpaths": True,
                    "worktree_clean": True,
                    "contract_file_sha256": plan["contract_file_sha256"],
                    "plan_file_sha256": gate.PLAN_FILE_SHA256,
                    "plan_payload_sha256": gate.PLAN_PAYLOAD_SHA256,
                    "r2_activation_plan_file_sha256": (
                        gate.R2_ACTIVATION_PLAN_FILE_SHA256
                    ),
                    "r2_activation_plan_payload_sha256": (
                        gate.R2_ACTIVATION_PLAN_PAYLOAD_SHA256
                    ),
                    "formal_design_review_file_sha256": (
                        gate.FORMAL_DESIGN_REVIEW_FILE_SHA256
                    ),
                    "formal_design_review_payload_sha256": (
                        gate.FORMAL_DESIGN_REVIEW_PAYLOAD_SHA256
                    ),
                    "occupied_v0_9_surface_sha256": (
                        gate.R2_OCCUPIED_V0_9_SURFACE_SHA256
                    ),
                    "candidate_manifest_file_sha256": manifest_file_hash,
                    "candidate_manifest_payload_sha256": (
                        manifest_payload_hash
                    ),
                    "implementation_author_separation_contract_sha256": (
                        gate.IMPLEMENTATION_AUTHOR_SEPARATION_SHA256
                    ),
                    "implementation_trust_model_sha256": (
                        gate.recompute_implementation_trust_model_sha256(plan)
                    ),
                    "artifact_path_surface_sha256": (
                        gate.artifact_surface_sha256(plan)
                    ),
                    "review_surface_identity": (
                        gate.review_surface_identity(plan)
                    ),
                    "implementation_files": implementation_rows,
                    "scientific_dependencies": binding[
                        "scientific_dependencies"
                    ],
                    "clean_restore": clean_restore,
                    "candidate_selection_file_sha256": selection_file_hash,
                    "candidate_selection_payload_sha256": selection[
                        "candidate_selection_payload_sha256"
                    ],
                    **identity["static_fields"],
                }
            )
            candidate.pop(
                "implementation_candidate_binding_payload_sha256", None
            )
            candidate = gate.add_self_hash(
                candidate,
                "implementation_candidate_binding_payload_sha256",
            )
            freeze_contract = control["review_input_freeze_contract"]
            freeze = {
                field: None
                for field in freeze_contract["exact_top_level_fields"]
            }
            freeze.update(freeze_contract["required_values"])
            freeze.update(
                {
                    "implementation_source_commit": source_commit,
                    "candidate_selection_file_sha256": selection_file_hash,
                    "candidate_selection_payload_sha256": selection[
                        "candidate_selection_payload_sha256"
                    ],
                    "candidate_manifest_file_sha256": manifest_file_hash,
                    "candidate_manifest_payload_sha256": (
                        manifest_payload_hash
                    ),
                    "implementation_candidate_binding_file_sha256": (
                        candidate_file_hash
                    ),
                    "implementation_candidate_binding_payload_sha256": (
                        candidate[
                            "implementation_candidate_binding_payload_sha256"
                        ]
                    ),
                    "clean_restore_receipt_file_sha256": restore_file_hash,
                    "clean_restore_receipt_payload_sha256": restore_receipt[
                        "restore_receipt_payload_sha256"
                    ],
                    "artifact_path_surface_sha256": (
                        gate.artifact_surface_sha256(plan)
                    ),
                    "review_packet_path": control[
                        "fresh_review_packet_path"
                    ],
                    "review_packet_file_sha256": packet_hash,
                    "review_packet_size_bytes": 1,
                    "changed_file_manifest_sha256": changed_digest,
                    "targeted_test_count": coverage[
                        "targeted_test_count"
                    ],
                    "targeted_test_node_id_sha256": coverage[
                        "targeted_test_node_id_sha256"
                    ],
                    "full_suite_test_count": coverage[
                        "full_suite_test_count"
                    ],
                    "full_suite_test_node_id_sha256": coverage[
                        "full_suite_test_node_id_sha256"
                    ],
                    **identity["static_fields"],
                }
            )
            freeze.pop("review_input_freeze_payload_sha256", None)
            freeze = gate.add_self_hash(
                freeze, "review_input_freeze_payload_sha256"
            )
            evidence_contract = control["fresh_review_evidence_contract"]
            evidence = {
                field: None
                for field in evidence_contract["exact_top_level_fields"]
            }
            evidence.update(evidence_contract["required_values"])
            evidence.update(
                {
                    "implementation_source_commit": source_commit,
                    "implementation_candidate_binding_file_sha256": (
                        candidate_file_hash
                    ),
                    "implementation_candidate_binding_payload_sha256": (
                        candidate[
                            "implementation_candidate_binding_payload_sha256"
                        ]
                    ),
                    "candidate_manifest_file_sha256": manifest_file_hash,
                    "candidate_manifest_payload_sha256": (
                        manifest_payload_hash
                    ),
                    "candidate_selection_file_sha256": selection_file_hash,
                    "candidate_selection_payload_sha256": selection[
                        "candidate_selection_payload_sha256"
                    ],
                    "review_input_freeze_file_sha256": freeze_file_hash,
                    "review_input_freeze_payload_sha256": freeze[
                        "review_input_freeze_payload_sha256"
                    ],
                    "artifact_path_surface_sha256": (
                        gate.artifact_surface_sha256(plan)
                    ),
                    "review_surface_identity": (
                        gate.review_surface_identity(plan)
                    ),
                    "implementation_review_packet_file_sha256": packet_hash,
                    "changed_file_manifest_sha256": changed_digest,
                    "targeted_test_count": coverage[
                        "targeted_test_count"
                    ],
                    "targeted_test_node_id_sha256": coverage[
                        "targeted_test_node_id_sha256"
                    ],
                    "full_suite_test_count": coverage[
                        "full_suite_test_count"
                    ],
                    "full_suite_test_node_id_sha256": coverage[
                        "full_suite_test_node_id_sha256"
                    ],
                    **identity["static_fields"],
                }
            )
            evidence.pop("review_evidence_payload_sha256", None)
            evidence = gate.add_self_hash(
                evidence, "review_evidence_payload_sha256"
            )
            review_schema = plan["review_receipt_schemas"][
                "fresh_implementation_review_verdict"
            ]
            review = {
                field: None
                for field in review_schema["exact_top_level_fields"]
            }
            review.update(
                review_schema["outcomes"]["pass"]["required_values"]
            )
            review.update(
                {
                    "reviewed_at_utc": "2026-08-04T00:00:00Z",
                    "implementation_author_separation_contract_sha256": (
                        gate.IMPLEMENTATION_AUTHOR_SEPARATION_SHA256
                    ),
                    "contract_file_sha256": plan["contract_file_sha256"],
                    "plan_file_sha256": gate.PLAN_FILE_SHA256,
                    "plan_payload_sha256": gate.PLAN_PAYLOAD_SHA256,
                    "r2_activation_plan_file_sha256": (
                        gate.R2_ACTIVATION_PLAN_FILE_SHA256
                    ),
                    "r2_activation_plan_payload_sha256": (
                        gate.R2_ACTIVATION_PLAN_PAYLOAD_SHA256
                    ),
                    "artifact_path_surface_sha256": (
                        gate.artifact_surface_sha256(plan)
                    ),
                    "occupied_v0_9_surface_sha256": (
                        gate.R2_OCCUPIED_V0_9_SURFACE_SHA256
                    ),
                    "candidate_manifest_file_sha256": manifest_file_hash,
                    "candidate_manifest_payload_sha256": (
                        manifest_payload_hash
                    ),
                    "formal_design_review_file_sha256": (
                        gate.FORMAL_DESIGN_REVIEW_FILE_SHA256
                    ),
                    "formal_design_review_payload_sha256": (
                        gate.FORMAL_DESIGN_REVIEW_PAYLOAD_SHA256
                    ),
                    "implementation_trust_model_sha256": (
                        gate.recompute_implementation_trust_model_sha256(plan)
                    ),
                    "implementation_candidate_binding_file_sha256": (
                        candidate_file_hash
                    ),
                    "implementation_candidate_binding_payload_sha256": (
                        candidate[
                            "implementation_candidate_binding_payload_sha256"
                        ]
                    ),
                    "implementation_source_commit": source_commit,
                    "implementation_review_packet_file_sha256": packet_hash,
                    "bundle_file_sha256": gate.sha256_bytes(bundle_raw),
                    "restore_receipt_file_sha256": restore_file_hash,
                    "restore_receipt_payload_sha256": restore_receipt[
                        "restore_receipt_payload_sha256"
                    ],
                    "P0_count": 0,
                    "P1_count": 0,
                    "P2_count": 0,
                    "review_surface_identity": (
                        gate.review_surface_identity(plan)
                    ),
                    "review_evidence_file_sha256": evidence_file_hash,
                    "review_evidence_payload_sha256": evidence[
                        "review_evidence_payload_sha256"
                    ],
                    "candidate_selection_file_sha256": selection_file_hash,
                    "candidate_selection_payload_sha256": selection[
                        "candidate_selection_payload_sha256"
                    ],
                    "review_input_freeze_file_sha256": freeze_file_hash,
                    "review_input_freeze_payload_sha256": freeze[
                        "review_input_freeze_payload_sha256"
                    ],
                    **identity["static_fields"],
                }
            )
            review.pop("fresh_implementation_review_payload_sha256", None)
            review = gate.add_self_hash(
                review, "fresh_implementation_review_payload_sha256"
            )
            with (
                mock.patch.object(
                    gate,
                    "read_r2r1_candidate_selection",
                    return_value=(selection, selection_file_hash),
                ),
                mock.patch.object(
                    gate,
                    "read_r2r1_review_input_freeze",
                    return_value=(freeze, freeze_file_hash),
                ),
                mock.patch.object(
                    gate,
                    "read_r2r1_fresh_review_evidence",
                    return_value=(evidence, evidence_file_hash),
                ),
            ):
                authority = gate.build_reviewed_authority_payload(
                    plan,
                    candidate,
                    review,
                    candidate_file_sha256=candidate_file_hash,
                    review_file_sha256=review_file_hash,
                )
            authority_file_hash = gate.sha256_bytes(
                gate.canonical_receipt_bytes(authority)
            )
            formal_schema = plan["review_receipt_schemas"][
                "formal_design_review_verdict"
            ]
            formal, formal_file_hash = independent._receipt(
                gate.FORMAL_DESIGN_REVIEW_PATH,
                formal_schema["exact_top_level_fields"],
                "formal_design_review_payload_sha256",
            )
            receipt_rows = {
                str(Path(binding["formal_design_review_path"])): (
                    formal,
                    formal_file_hash,
                ),
                str(Path(binding["artifact_path"])): (
                    candidate,
                    candidate_file_hash,
                ),
                str(Path(review_schema["artifact_path"])): (
                    review,
                    review_file_hash,
                ),
                str(
                    Path(
                        plan["reviewed_implementation_authority_contract"][
                            "artifact_path"
                        ]
                    )
                ): (authority, authority_file_hash),
            }

            def receipt_reader(
                path: Path,
                _fields: Any,
                _hash_field: str,
            ) -> tuple[dict[str, Any], str]:
                try:
                    value, file_hash = receipt_rows[str(path)]
                except KeyError as exc:
                    raise AssertionError(f"unexpected receipt: {path}") from exc
                return copy.deepcopy(value), file_hash

            def lineage(_repository: Path, commit: str) -> tuple[str, ...]:
                if commit == source_commit:
                    return (source_commit, parent_commit)
                if commit == parent_commit:
                    return (parent_commit, grandparent_commit)
                return (commit,)

            def blob_oid(
                repository: Path, _commit: str, relative: str
            ) -> str:
                return independent._git_blob(
                    (repository / relative).read_bytes(), object_format
                )

            original_read = Path.read_bytes
            protected = str(gate.PROTECTED_ROOT).casefold()

            def guarded_read(path: Path) -> bytes:
                if str(path).casefold().startswith(protected):
                    raise AssertionError("protected root read in R2R2 lineage")
                return original_read(path)

            with (
                mock.patch.object(
                    independent, "_receipt", side_effect=receipt_reader
                ),
                mock.patch.object(
                    independent,
                    "_independent_upstream_artifact",
                    return_value={},
                ),
                mock.patch.object(
                    independent, "_git_parent_lineage", side_effect=lineage
                ),
                mock.patch.object(
                    independent, "_git_path_blob_oid", side_effect=blob_oid
                ),
                mock.patch.object(
                    independent,
                    "_independent_r2r1_candidate_selection",
                    return_value=(selection, selection_file_hash),
                ),
                mock.patch.object(
                    independent,
                    "_independent_r2r1_restore_receipt",
                    return_value=(restore_receipt, restore_file_hash),
                ),
                mock.patch.object(
                    independent,
                    "_independent_r2r1_candidate_manifest",
                    return_value=({}, manifest_file_hash),
                ),
                mock.patch.object(
                    independent,
                    "_independent_r2r1_review_input_freeze",
                    return_value=(freeze, freeze_file_hash),
                ),
                mock.patch.object(
                    independent,
                    "_independent_r2r1_review_evidence",
                    return_value=(evidence, evidence_file_hash),
                ),
                mock.patch.object(Path, "read_bytes", guarded_read),
            ):
                rebuilt, rebuilt_file_hash, rebuilt_candidate = (
                    independent.independent_runtime_lineage(
                        self.independent_plan, REPOSITORY
                    )
                )
            self.assertEqual(rebuilt, authority)
            self.assertEqual(rebuilt_file_hash, authority_file_hash)
            self.assertEqual(rebuilt_candidate, candidate)

class R2R4OriginalInputLineageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = gate.load_active_plan(repository_root=REPOSITORY)
        cls.independent_plan = independent.independent_load_plan(
            repository_root=REPOSITORY
        )

    def _assert_readers_reject(
        self, raw: bytes, row: Mapping[str, Any]
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            path.write_bytes(raw)
            attacked = dict(row)
            attacked["path"] = str(path)
            attacked["file_sha256"] = gate.sha256_bytes(raw)
            with self.assertRaises(gate.Gate12C2OriginalBaselineError):
                gate.read_declared_frozen_json_artifact(
                    path,
                    expected_file_sha256=attacked["file_sha256"],
                    expected_payload_sha256=attacked["payload_sha256"],
                    payload_hash_domain=attacked["payload_hash_domain"],
                    self_hash_field=attacked["self_hash_field"],
                    expected_schema_version=attacked["schema_version"],
                )
            with self.assertRaises(
                independent.IndependentVerificationError
            ):
                independent._independent_original_input_artifact(attacked)

    def _assert_lineage_rejected(self, plan: Mapping[str, Any]) -> None:
        with self.assertRaises(gate.Gate12C2OriginalBaselineError):
            gate.validate_original_input_lineage(plan)
        with self.assertRaises(
            independent.IndependentVerificationError
        ):
            independent.independent_lineage(plan)

    def test_exact_five_row_surface_and_real_lineage_pass(self) -> None:
        expected_roles = [
            "original_plan",
            "incident_manifest",
            "payload_seal",
            "payload_seal_verification",
            "formal_payload_closeout",
        ]
        core_rows = gate.original_input_framing_surface(self.plan)
        independent_rows = (
            independent._independent_original_input_framing_surface(
                self.independent_plan
            )
        )
        self.assertEqual([row["role"] for row in core_rows], expected_roles)
        self.assertEqual(core_rows, independent_rows)
        self.assertEqual(
            gate.sha256_bytes(gate.canonical_json_bytes(core_rows)),
            gate.R2R4_ORIGINAL_INPUT_FRAMING_SURFACE_SHA256,
        )
        original_read = Path.read_bytes
        protected = str(gate.PROTECTED_ROOT).casefold()

        def guarded(path: Path) -> bytes:
            if str(path).casefold().startswith(protected):
                raise AssertionError("protected root read during original lineage")
            return original_read(path)

        with mock.patch.object(Path, "read_bytes", guarded):
            core = gate.validate_original_input_lineage(self.plan)
            independently = independent.independent_lineage(
                self.independent_plan
            )
        self.assertEqual(set(core), set(expected_roles))
        self.assertEqual(
            core["original_plan"]["draw_profile_plan_payload_sha256"],
            self.plan["original_input_lineage"][
                "original_plan_payload_sha256"
            ],
        )
        self.assertEqual(
            independently[0]["draw_profile_plan_payload_sha256"],
            self.plan["original_input_lineage"][
                "original_plan_payload_sha256"
            ],
        )
        core_source = inspect.getsource(gate.validate_original_input_lineage)
        independent_source = inspect.getsource(independent.independent_lineage)
        self.assertNotIn('original_plan.get("plan_payload_sha256")', core_source)
        self.assertNotIn('original.get("plan_payload_sha256")', independent_source)

    def test_each_artifact_rejects_framing_and_identity_attacks(self) -> None:
        rows = gate.original_input_framing_surface(self.plan)
        for row in rows:
            raw = Path(row["path"]).read_bytes()
            payload = gate.strict_json_loads(raw, canonical=True)
            reordered = json.dumps(
                dict(reversed(list(payload.items()))),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
            duplicate = b'{"__duplicate__":0,"__duplicate__":0,' + raw[1:]
            wrong_schema = dict(payload)
            wrong_schema["schema_version"] = "wrong_schema"
            cases = {
                "added_lf": (raw + b"\n", {}),
                "crlf": (raw + b"\r\n", {}),
                "double_lf": (raw + b"\n\n", {}),
                "truncated": (raw[:-1], {}),
                "reordered": (reordered, {}),
                "duplicate": (duplicate, {}),
                "wrong_schema": (
                    gate.canonical_json_bytes(wrong_schema),
                    {},
                ),
                "wrong_self_hash_field": (
                    raw,
                    {"self_hash_field": "plan_payload_sha256"},
                ),
                "wrong_payload_hash": (
                    raw,
                    {"payload_sha256": "f" * 64},
                ),
                "unknown_domain": (
                    raw,
                    {"payload_hash_domain": "unknown"},
                ),
            }
            for name, (attacked_raw, updates) in cases.items():
                with self.subTest(role=row["role"], attack=name):
                    attacked_row = dict(row)
                    attacked_row.update(updates)
                    self._assert_readers_reject(attacked_raw, attacked_row)

    def test_surface_attacks_and_full_downstream_rehash_are_rejected(self) -> None:
        mutations = {}

        def missing(rows: list[dict[str, Any]]) -> None:
            rows.pop()

        def extra(rows: list[dict[str, Any]]) -> None:
            row = dict(rows[-1])
            row["role"] = "unexpected_role"
            rows.append(row)

        def duplicate(rows: list[dict[str, Any]]) -> None:
            rows[-1] = dict(rows[0])

        def substitution(rows: list[dict[str, Any]]) -> None:
            rows[0], rows[1] = rows[1], rows[0]

        def wrong_domain(rows: list[dict[str, Any]]) -> None:
            rows[0]["payload_hash_domain"] = "unknown"

        def wrong_field(rows: list[dict[str, Any]]) -> None:
            rows[0]["self_hash_field"] = "plan_payload_sha256"

        mutations.update(
            {
                "missing": missing,
                "extra": extra,
                "duplicate": duplicate,
                "role_row_substitution": substitution,
                "unknown_domain": wrong_domain,
                "original_plan_wrong_field": wrong_field,
            }
        )
        for name, mutate in mutations.items():
            with self.subTest(attack=name):
                plan = copy.deepcopy(self.plan)
                rows = plan["r2r2_portability_control"][
                    "original_input_json_framing"
                ]
                mutate(rows)
                plan["r2r2_portability_control"][
                    "original_input_json_framing_surface_sha256"
                ] = gate.sha256_bytes(gate.canonical_json_bytes(rows))
                self._assert_lineage_rejected(plan)

        specs = {
            role: (path_key, file_key, payload_key, self_hash_field)
            for (
                role,
                path_key,
                file_key,
                payload_key,
                _schema,
                self_hash_field,
            ) in gate.ORIGINAL_INPUT_FRAMING_SPECIFICATIONS
        }
        for original_row in gate.original_input_framing_surface(self.plan):
            role = original_row["role"]
            with self.subTest(attack="downstream_rehash", role=role):
                payload = gate.strict_json_loads(
                    Path(original_row["path"]).read_bytes(), canonical=True
                )
                self_hash_field = original_row["self_hash_field"]
                payload = dict(payload)
                payload.pop(self_hash_field)
                payload["r2r4_adversarial_probe"] = True
                payload_hash = gate.sha256_bytes(
                    gate.canonical_json_bytes(payload)
                )
                payload[self_hash_field] = payload_hash
                attacked_raw = gate.canonical_json_bytes(payload)
                with tempfile.TemporaryDirectory() as directory:
                    attacked_path = Path(directory) / f"{role}.json"
                    attacked_path.write_bytes(attacked_raw)
                    plan = copy.deepcopy(self.plan)
                    path_key, file_key, payload_key, _field = specs[role]
                    lineage = plan["original_input_lineage"]
                    lineage[path_key] = str(attacked_path)
                    lineage[file_key] = gate.sha256_bytes(attacked_raw)
                    lineage[payload_key] = payload_hash
                    rows = plan["r2r2_portability_control"][
                        "original_input_json_framing"
                    ]
                    for row in rows:
                        if row["role"] == role:
                            row["path"] = str(attacked_path)
                            row["file_sha256"] = gate.sha256_bytes(attacked_raw)
                            row["payload_sha256"] = payload_hash
                    plan["r2r2_portability_control"][
                        "original_input_json_framing_surface_sha256"
                    ] = gate.sha256_bytes(gate.canonical_json_bytes(rows))
                    self._assert_lineage_rejected(plan)

    def test_real_issue_preflight_reaches_pass_without_publication(self) -> None:
        plan = self.plan
        runtime_roles = {
            row["role"]: row
            for row in plan["artifact_path_surface"]
            if row["lifecycle_scope"] in {"extraction", "verifier"}
        }
        paths = [
            Path(row[key])
            for row in runtime_roles.values()
            for key in ("final_path", "pending_path")
        ]
        self.assertTrue(all(not path.exists() for path in paths))
        authority_fixture = {
            "reviewed_implementation_authority_payload_sha256": "a" * 64
        }
        candidate_fixture = {
            "source_commit": "b" * 40,
        }
        original_read = Path.read_bytes
        protected = str(gate.PROTECTED_ROOT).casefold()

        def guarded(path: Path) -> bytes:
            if str(path).casefold().startswith(protected):
                raise AssertionError("protected root read during preflight")
            return original_read(path)

        issued = "2026-08-07T00:00:00Z"
        expires = "2026-08-07T00:10:00Z"
        with (
            mock.patch.object(Path, "read_bytes", guarded),
            mock.patch.object(
                gate, "validate_formal_design_pass", return_value={}
            ),
            mock.patch.object(
                gate, "validate_upstream_authority", return_value={}
            ),
            mock.patch.object(
                preflight_issuer,
                "load_reviewed_chain",
                return_value=(
                    authority_fixture,
                    "c" * 64,
                    candidate_fixture,
                ),
            ),
            mock.patch.object(
                preflight_issuer, "_observe_surface", return_value={}
            ),
            mock.patch.object(
                gate,
                "classify_lifecycle_surface",
                return_value="reviewed_implementation_authority_published",
            ),
        ):
            payload = preflight_issuer.issue_preflight(
                REPOSITORY,
                scope="extraction",
                preflight_id="r2r4-no-publication-regression",
                issued_at_utc=issued,
                expires_at_utc=expires,
                now_ns=gate.parse_utc_ns(issued),
            )
        self.assertEqual(payload["state"], "EXTRACTION_PREFLIGHT_PASS")
        self.assertEqual(
            payload["protected_root_status"],
            "canonical_path_bound_no_payload_read",
        )
        self.assertNotIn("scientific_values_emitted", payload)
        self.assertTrue(all(not path.exists() for path in paths))


class R2R5LaunchRetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = gate.load_active_plan(repository_root=REPOSITORY)
        cls.independent_plan = independent.independent_load_plan(
            repository_root=REPOSITORY
        )
        cls.control = cls.plan["r2r2_portability_control"]
        cls.historical = cls.control["historical_r2r4_preclaim_stop"]
        cls.current_by_role = {
            row["role"]: row for row in cls.plan["artifact_path_surface"]
        }
        cls.historical_by_role = {
            row["role"]: row
            for row in cls.historical["artifact_path_surface"]
        }

    def _runner_command(self, *flags: str) -> list[str]:
        contract = self.control["extraction_launch_contract"]
        return [
            contract["python_executable_path"],
            *flags,
            contract["argv_prefix"][3],
            "--repository",
            str(REPOSITORY),
            "--execution-claim-id",
            "r2r5-focused-preclaim",
            "--launch-id",
            "r2r5-focused-preclaim",
            "--claimed-at-utc",
            "2026-08-07T00:00:00Z",
        ]

    def _run_runner(
        self, *flags: str, pythonpath: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        if pythonpath is None:
            environment.pop("PYTHONPATH", None)
        else:
            environment["PYTHONPATH"] = pythonpath
        return subprocess.run(
            self._runner_command(*flags),
            cwd=REPOSITORY,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            timeout=30,
            check=False,
        )

    def _assert_no_r2r5_runtime(self) -> None:
        runtime = {
            "extraction_execution_claim",
            "extraction_failure",
            "extraction_success",
            "extraction_terminal",
            "verifier_authorization",
            "verifier_authorization_verdict",
            "verifier_execution_claim",
            "verifier_failure",
            "verifier_preflight",
            "verifier_success",
            "verifier_terminal",
        }
        for role in runtime:
            row = self.current_by_role[role]
            self.assertFalse(Path(row["final_path"]).exists(), role)
            self.assertFalse(Path(row["pending_path"]).exists(), role)

    def test_exact_launch_contract_and_isolation_guard_order(self) -> None:
        core = gate.validate_r2r5_extraction_launch_contract(
            self.control["extraction_launch_contract"],
            repository_root=REPOSITORY,
        )
        standalone = independent._independent_validate_r2r5_launch_contract(
            self.independent_plan["r2r2_portability_control"][
                "extraction_launch_contract"
            ],
            repository_root=REPOSITORY,
        )
        self.assertEqual(core, standalone)
        self.assertEqual(
            gate.sha256_bytes(gate.canonical_json_bytes(core)),
            gate.R2R5_EXTRACTION_LAUNCH_CONTRACT_SHA256,
        )
        self.assertEqual(core["argv_prefix"][1:3], ["-I", "-B"])
        self.assertEqual(core["cwd"], str(REPOSITORY))
        self.assertEqual(core["stdin"], "DEVNULL")
        self.assertEqual(
            gate.sha256_bytes(
                (REPOSITORY / core["runner_relative_path"]).read_bytes()
            ),
            gate.R2R5_RUNNER_FILE_SHA256,
        )
        source = inspect.getsource(extraction_runner.execute)
        self.assertLess(source.index("_runtime_isolated()"), source.index("load_active_plan"))
        self.assertLess(source.index("_runtime_isolated()"), source.index("publish_role"))

    def test_nonisolated_and_pythonpath_reject_before_claim(self) -> None:
        self._assert_no_r2r5_runtime()
        nonisolated = self._run_runner("-B")
        self.assertEqual(nonisolated.returncode, 2)
        self.assertEqual(nonisolated.stdout, "")
        self.assertEqual(
            nonisolated.stderr,
            "gate12c2-original-baseline:ERROR:AUTHORIZATION_INVALID\n",
        )
        with_pythonpath = self._run_runner(
            "-I", "-B", pythonpath=str(REPOSITORY)
        )
        self.assertEqual(with_pythonpath.returncode, 2)
        self.assertEqual(with_pythonpath.stdout, "")
        self.assertEqual(
            with_pythonpath.stderr,
            "gate12c2-original-baseline:ERROR:AUTHORIZATION_INVALID\n",
        )
        self._assert_no_r2r5_runtime()

    def test_isolated_launch_passes_guard_and_stops_on_absent_controls(self) -> None:
        self._assert_no_r2r5_runtime()
        isolated = self._run_runner("-I", "-B")
        self.assertEqual(isolated.returncode, 2)
        self.assertEqual(isolated.stdout, "")
        self.assertEqual(
            isolated.stderr,
            "gate12c2-original-baseline:ERROR:INPUT_LINEAGE_MISMATCH\n",
        )
        self.assertNotIn("AUTHORIZATION_INVALID", isolated.stderr)
        self._assert_no_r2r5_runtime()

    def test_historical_controls_are_exact_retired_and_runtime_absent(self) -> None:
        core = gate.validate_r2r5_historical_preclaim_stop(
            self.historical, self.plan["artifact_path_surface"]
        )
        standalone = independent._independent_validate_r2r5_preclaim_stop(
            self.independent_plan["r2r2_portability_control"][
                "historical_r2r4_preclaim_stop"
            ],
            self.independent_plan["artifact_path_surface"],
        )
        self.assertEqual(core, standalone)
        self.assertEqual(
            core["execution_status"],
            "HISTORICALLY_VALID_CONTROLS_OPERATIONALLY_RETIRED_AFTER_"
            "PRECLAIM_LAUNCH_CONTRACT_MISMATCH",
        )
        self.assertEqual(core["launch_failure_code"], "AUTHORIZATION_INVALID")
        for row in core["occupied_controls"]:
            raw = Path(row["path"]).read_bytes()
            self.assertEqual(gate.sha256_bytes(raw), row["file_sha256"])
        self._assert_no_r2r5_runtime()

    def test_tampered_historical_control_and_present_leaf_are_rejected(self) -> None:
        target = Path(self.historical["occupied_controls"][0]["path"])
        original_read = Path.read_bytes

        def tampered_read(path: Path) -> bytes:
            raw = original_read(path)
            return raw + b" " if path == target else raw

        with mock.patch.object(Path, "read_bytes", tampered_read):
            with self.assertRaises(gate.Gate12C2OriginalBaselineError):
                gate.validate_r2r5_historical_preclaim_stop(
                    self.historical, self.plan["artifact_path_surface"]
                )
            with self.assertRaises(independent.IndependentVerificationError):
                independent._independent_validate_r2r5_preclaim_stop(
                    self.historical,
                    self.independent_plan["artifact_path_surface"],
                )

        claim = Path(
            self.historical_by_role["extraction_execution_claim"]["final_path"]
        )
        original_exists = Path.exists

        def injected_exists(path: Path) -> bool:
            return path == claim or original_exists(path)

        with mock.patch.object(Path, "exists", injected_exists):
            with self.assertRaises(gate.Gate12C2OriginalBaselineError):
                gate.validate_r2r5_historical_preclaim_stop(
                    self.historical, self.plan["artifact_path_surface"]
                )
            with self.assertRaises(independent.IndependentVerificationError):
                independent._independent_validate_r2r5_preclaim_stop(
                    self.historical,
                    self.independent_plan["artifact_path_surface"],
                )

    def test_r2r5_surface_is_disjoint_from_every_historical_path(self) -> None:
        current = {
            path
            for row in self.plan["artifact_path_surface"]
            for path in (row["final_path"], row["pending_path"])
        }
        historical = {
            path
            for row in self.historical["artifact_path_surface"]
            for path in (row["final_path"], row["pending_path"])
        }
        self.assertEqual(len(current), 36)
        self.assertEqual(len(historical), 36)
        self.assertFalse(current & historical)
        self.assertTrue(
            all(gate.R2R2_AUTHORITY_NAMESPACE_ID in Path(path).name for path in current)
        )

    def test_phase_a_plan_load_reads_zero_protected_bytes(self) -> None:
        original_read = Path.read_bytes
        protected = str(gate.PROTECTED_ROOT).casefold()
        reads: list[str] = []

        def guarded(path: Path) -> bytes:
            normalized = str(path).casefold()
            if normalized.startswith(protected):
                raise AssertionError("protected root read during R2R5 Phase A")
            reads.append(normalized)
            return original_read(path)

        with mock.patch.object(Path, "read_bytes", guarded):
            core = gate.load_active_plan(repository_root=REPOSITORY)
            standalone = independent.independent_load_plan(
                repository_root=REPOSITORY
            )
        self.assertEqual(
            core["r2r2_portability_control"]["authority_namespace_id"],
            "R2R5_20260807",
        )
        self.assertEqual(
            standalone["r2r2_portability_control"]["authority_namespace_id"],
            "R2R5_20260807",
        )
        self.assertTrue(reads)
        self.assertFalse(any(path.startswith(protected) for path in reads))

    def test_launch_contract_and_historical_surface_rehash_attacks_fail(self) -> None:
        overlay = gate.load_r2r2_portability_plan(
            repository_root=REPOSITORY,
            check_r2r1_occupancy=True,
        )
        attacks = []
        launch = copy.deepcopy(overlay)
        launch["extraction_launch_contract"]["argv_prefix"][1] = "-B"
        launch["extraction_launch_contract_sha256"] = gate.sha256_bytes(
            gate.canonical_json_bytes(launch["extraction_launch_contract"])
        )
        launch = gate.add_self_hash(
            {
                key: value
                for key, value in launch.items()
                if key != "r2r2_portability_plan_payload_sha256"
            },
            "r2r2_portability_plan_payload_sha256",
        )
        attacks.append(launch)
        historical = copy.deepcopy(overlay)
        historical["historical_r2r4_preclaim_stop"][
            "required_absent_final_roles"
        ].pop()
        historical["historical_r2r4_preclaim_surface_sha256"] = (
            gate.sha256_bytes(
                gate.canonical_json_bytes(
                    historical["historical_r2r4_preclaim_stop"]
                )
            )
        )
        historical = gate.add_self_hash(
            {
                key: value
                for key, value in historical.items()
                if key != "r2r2_portability_plan_payload_sha256"
            },
            "r2r2_portability_plan_payload_sha256",
        )
        attacks.append(historical)
        r2r1 = gate.build_r2r1_active_plan(
            gate.load_r2_active_plan(repository_root=REPOSITORY),
            gate.load_r2r1_remediation_plan(
                repository_root=REPOSITORY,
                check_r2_occupancy=True,
            ),
            repository_root=REPOSITORY,
        )
        for attacked in attacks:
            with self.subTest(attack=attacked["state"]):
                with self.assertRaises(gate.Gate12C2OriginalBaselineError):
                    gate.validate_r2r2_portability_plan(
                        r2r1,
                        attacked,
                        repository_root=REPOSITORY,
                        check_r2r1_occupancy=True,
                    )

if __name__ == "__main__":
    unittest.main()
