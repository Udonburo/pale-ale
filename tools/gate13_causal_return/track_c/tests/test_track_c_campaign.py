from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest

from analysis.gate13_causal_return.review2_1.track_c_review2_1_validator import (
    validate_frozen_campaign_manifest,
)
from tools.gate13_causal_return.track_c.campaign import (
    AtomicCaseStore,
    ExactTokenizer,
    analyze_map_block,
    compile_campaign,
    json_ready,
    ledger_indexes,
    official_text_chat_render,
    synthetic_preflight,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
TOKENIZER = (
    REPO_ROOT
    / "workstream/local/gate13_causal_return_outputs/checkpoint_panel/assets/qwen3_6-27b/tokenizer.json"
)


@pytest.fixture(scope="module")
def compiled() -> dict[str, dict]:
    return compile_campaign(TOKENIZER)


def test_complete_frozen_surface(compiled: dict[str, dict]) -> None:
    result = validate_frozen_campaign_manifest(compiled["manifest"])
    assert result["status"] == "PASS"
    assert result["block_count"] == 20
    assert result["map_case_count"] == 4_800
    assert result["behavior_case_count"] == 5_760
    assert result["path_surface_validation"]["pair_count"] == 480


def test_case_ids_and_dynamic_prompt_variants_are_complete(compiled: dict[str, dict]) -> None:
    map_rows, behavior_rows = ledger_indexes(compiled["ledger"])
    assert len(map_rows) == 4_800
    assert len(behavior_rows) == 5_760
    assert not set(map_rows) & set(behavior_rows)
    for row in behavior_rows.values():
        assert set(row["prompt_variants"]) == {"0", "1"}
        for binding in row["prompt_variants"].values():
            assert len(binding["canonical_message_sha256"]) == 64
            assert len(binding["rendered_prompt_sha256"]) == 64
            assert binding["input_token_count"] > 0


def test_stage_e_order_preserves_each_self_fed_path(compiled: dict[str, dict]) -> None:
    manifest = compiled["manifest"]
    ownership = {
        case_id: block["block_id"]
        for block in manifest["blocks"]
        for case_id in block["behavior_case_ids"]
    }
    for block in manifest["blocks"]:
        observed = [
            case_id
            for case_id in manifest["execution"]["stage_e_order"]
            if ownership[case_id] == block["block_id"]
        ]
        assert observed == block["behavior_case_ids"]


def test_all_opaque_labels_are_one_non_special_token() -> None:
    tokenizer = ExactTokenizer.from_file(TOKENIZER)
    raw = json.loads(TOKENIZER.read_text(encoding="utf-8"))
    assert len(raw["model"]["vocab"]) > 100_000
    compiled = compile_campaign(TOKENIZER)
    bindings = compiled["manifest"]["scoring"]["all_label_bindings"]
    assert len(bindings) == 64
    assert len({row["token_id"] for row in bindings}) == 64
    assert all(row["token_id"] not in tokenizer.special_ids for row in bindings)


def test_path_surface_changes_only_operation_order(compiled: dict[str, dict]) -> None:
    results = compiled["validation"]["path_surface_validation"]["pair_results"]
    assert len(results) == 480
    assert all(result["status"] == "PASS" for result in results)
    assert all(result["checks"]["operation_order_is_the_only_intended_difference"] for result in results)
    assert all(result["checks"]["non_operation_token_sequence"] for result in results)


def test_exact_resume_skips_accepted_and_rejects_binding_drift(tmp_path: Path) -> None:
    store = AtomicCaseStore(tmp_path, "stage_m")
    binding = {"case_id": "tc-m-test", "prompt_sha256": "a" * 64}
    store.record_attempt("tc-m-test", binding)
    store.record_attempt("tc-m-test", binding)
    response = {"case_id": "tc-m-test", "candidate_logits": [1.0, 0.0]}
    store.accept("tc-m-test", response)
    store.accept("tc-m-test", response)
    assert store.accepted("tc-m-test") == response
    assert store.accepted_ids() == {"tc-m-test"}
    with pytest.raises(Exception, match="attempt binding drift"):
        store.record_attempt("tc-m-test", {"case_id": "tc-m-test", "prompt_sha256": "b" * 64})


def test_model_free_pipeline_and_determinism(compiled: dict[str, dict]) -> None:
    first = compile_campaign(TOKENIZER)
    second = compile_campaign(TOKENIZER)
    assert first == second == compiled
    result = synthetic_preflight(compiled["manifest"], compiled["ledger"])
    assert result["status"] == "PASS"
    assert result["scientific_model_forward_count"] == 0


def test_chat_renderer_has_frozen_nonthinking_score_slot() -> None:
    rendered = official_text_chat_render("Return only Dak.")
    assert rendered.endswith("<think>\n\n</think>\n\n")
    assert rendered.rfind("<think>") < rendered.rfind("</think>")
    assert "<|image_pad|>" not in rendered


def test_map_block_analysis_accepts_complete_synthetic_packet(compiled: dict[str, dict]) -> None:
    block = compiled["manifest"]["blocks"][0]
    map_index, _ = ledger_indexes(compiled["ledger"])
    rows = []
    latent: dict[tuple[str, str], np.ndarray] = {}
    node_order = {
        "phase0_state0": 0,
        "phase0_state1": 1,
        "phase1_state0": 2,
        "phase1_state1": 3,
        "phase1_state1_broken": 4,
    }
    for case_id in block["map_case_ids"]:
        row = dict(map_index[case_id])
        target = int(row["target_state"])
        row["candidate_logits"] = [2.0 if target == 0 else -1.0, 2.0 if target == 1 else -1.0]
        rows.append(row)
        key = (row["half_id"], row["sample_id"])
        if key not in latent:
            seed = int.from_bytes(
                hashlib.sha256(row["sample_id"].encode("utf-8")).digest()[:8],
                "little",
            )
            latent[key] = np.random.default_rng(seed).normal(size=10)

    def activation_loader(row: dict) -> dict[int, np.ndarray]:
        base = latent[(row["half_id"], row["sample_id"])]
        shift = node_order[row["node_id"]]
        rolled = np.roll(base, shift)
        if row["node_id"].endswith("_broken"):
            rolled = rolled * np.linspace(0.4, 1.8, rolled.size)
        return {layer: rolled + layer * 1.0e-4 for layer in (21, 43, 62)}

    result = analyze_map_block(
        block=block,
        response_rows=rows,
        activation_loader=activation_loader,
    )
    assert result["block_id"] == block["block_id"]
    assert len(result["layers"]) == 3
    json.dumps(json_ready(result), sort_keys=True)
