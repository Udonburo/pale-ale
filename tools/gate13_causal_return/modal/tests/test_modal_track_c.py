from __future__ import annotations

import ast
from pathlib import Path

from tools.gate13_causal_return.modal import modal_track_c


SOURCE = Path(modal_track_c.__file__)


def test_modal_track_c_is_bound_to_one_frozen_execution() -> None:
    assert modal_track_c.EXECUTION_ID == "bf41b049-f04b-442e-b0bd-05c8adbd4944"
    assert modal_track_c.MODEL_VOLUME_NAME.startswith("gate13-track-c-qwen3-6-27b-")
    assert modal_track_c.RESULT_VOLUME_NAME.endswith("bf41b049-results")
    assert modal_track_c.CAMPAIGN_SPEND_CEILING_USD == 65.0


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
