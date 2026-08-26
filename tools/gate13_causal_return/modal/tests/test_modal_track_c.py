from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from tools.gate13_causal_return.modal import modal_track_c


SOURCE = Path(modal_track_c.__file__)


def test_modal_track_c_is_bound_to_one_frozen_execution() -> None:
    assert modal_track_c.EXECUTION_ID == "bf41b049-f04b-442e-b0bd-05c8adbd4944"
    assert modal_track_c.MODEL_VOLUME_NAME.startswith("gate13-track-c-qwen3-6-27b-")
    assert modal_track_c.RESULT_VOLUME_NAME.endswith("bf41b049-results")
    assert modal_track_c.CAMPAIGN_SPEND_CEILING_USD == 65.0
    assert (
        modal_track_c.MODAL_DEPLOYMENT_MODULE
        == "tools.gate13_causal_return.modal.modal_track_c"
    )


def test_adapter_has_no_prompt_or_scientific_threshold_literals() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "tools.gate13_causal_return.track_c.campaign"
        for alias in node.names
    }
    assert {"render_map_case", "render_behavior_case", "evaluate_map_campaign"} <= imported
    text = SOURCE.read_text(encoding="utf-8")
    assert "Return only the next state label" not in text
    assert "99999" not in text
    assert "phase0_state0" not in text


def test_modal_execution_policy_is_single_container_zero_retry() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    assert text.count("retries=0") == 3
    assert text.count("max_containers=1") == 3
    assert 'gpu="A100-80GB"' in text
    assert "block_network=True" in text


def test_forward0_failures_have_a_persisted_preflight_terminal() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    stage_m = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_stage_m_core"
    )
    guarded_load = [
        node
        for node in ast.walk(stage_m)
        if isinstance(node, ast.Try)
        and any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "_load_probe"
            for statement in node.body
            for call in ast.walk(statement)
        )
    ]
    assert len(guarded_load) == 1
    handler_text = ast.unparse(guarded_load[0].handlers[0])
    assert "TRACK_C_PREFLIGHT_BLOCKED" in handler_text
    assert "stage_m_scientific_forwards': 0" in handler_text


def test_authorization_validator_fails_closed_on_core_bindings() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    for field in (
        "execution_authorized",
        "execution_id",
        "model_volume_object_id",
        "result_volume_object_id",
        "absolute_campaign_ceiling_usd",
        "formal_gate13",
    ):
        assert field in text


def test_committed_authorization_verifies_all_local_hashes(monkeypatch) -> None:
    class FakeVolume:
        def __init__(self, object_id: str):
            self.object_id = object_id

    monkeypatch.setattr(
        modal_track_c,
        "model_volume",
        FakeVolume("vo-LJ2HR3cdLcr4k287kokPym"),
    )
    monkeypatch.setattr(
        modal_track_c,
        "result_volume",
        FakeVolume("vo-um6ngAeKYhggXfHD5OKsLS"),
    )
    authorization_path = (
        SOURCE.parents[3]
        / "analysis/gate13_causal_return/track_c_execution/track_c_execution_authorization.json"
    )
    text = authorization_path.read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    result = modal_track_c._authorization(text, digest)
    assert result["execution_authorized"] is True
