"""Fail-closed validator for the simultaneous stepwise A/B freeze."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from .compiler import (
    CAMPAIGN_FORWARD_CEILING,
    CAMPAIGN_SPEND_CEILING_USD,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    QUALIFICATION_SEED_RANGE,
    TOKENIZER_REVISION,
    codebook_bank,
    compile_qualification_ledgers,
    compile_track_b_collection_ledger,
    prompt_contract_payload,
    sha256_json,
    validate_codebook_partition,
)
from .operator_qualification import operator_lock_payload
from .runner import DEVELOPMENT_THRESHOLDS, QUALIFICATION_THRESHOLDS


class SuccessorLockError(ValueError):
    """Raised when any lock binding drifts."""


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SuccessorLockError(f"{path.name} must contain an object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SuccessorLockError(message)


def _git_clean(repo_root: Path) -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return not result.stdout.strip()


def validate_successor_locks(
    *,
    lock_dir: Path,
    repo_root: Path | None = None,
    require_clean: bool = False,
) -> dict[str, Any]:
    a_path = lock_dir / "stepwise_track_a_qualification_lock.json"
    b_path = lock_dir / "fresh_operator_qualification_lock.json"
    a_lock = _read(a_path)
    b_lock = _read(b_path)
    _require(
        a_lock["campaign_binding_id"] == b_lock["campaign_binding_id"],
        "A/B campaign binding differs",
    )
    _require(a_lock["counterpart_lock"] == b_path.name, "A counterpart drift")
    _require(b_lock["counterpart_lock"] == a_path.name, "B counterpart drift")
    variant = str(a_lock["development_selection"]["selected_variant"])
    _require(variant == b_lock["collection_surface"]["same_stepwise_substrate_variant"], "A/B variant drift")

    a_ledger = compile_qualification_ledgers(variant)
    b_ledger = compile_track_b_collection_ledger(variant)
    partition = validate_codebook_partition()
    _require(
        a_lock["qualification_surface"]["ledger_sha256"] == a_ledger["sha256"],
        "Track A ledger hash drift",
    )
    _require(
        b_lock["collection_surface"]["collection_ledger_sha256"] == b_ledger["sha256"],
        "Track B collection ledger hash drift",
    )
    _require(
        a_lock["instrument"]["prompt_contract_sha256"]
        == prompt_contract_payload(variant)["sha256"],
        "prompt contract drift",
    )
    _require(
        a_lock["instrument"]["qualification_codebook_bank_sha256"]
        == partition["banks"]["qualification"]["sha256"],
        "qualification codebook drift",
    )
    _require(
        b_lock["independent_halves"]["half_1_codebook_bank_sha256"]
        == partition["banks"]["track_b_half_1"]["sha256"],
        "Track B half 1 codebook drift",
    )
    _require(
        b_lock["independent_halves"]["half_2_codebook_bank_sha256"]
        == partition["banks"]["track_b_half_2"]["sha256"],
        "Track B half 2 codebook drift",
    )
    _require(
        set(partition["banks"]["track_b_half_1"]["labels"]).isdisjoint(
            partition["banks"]["track_b_half_2"]["labels"]
        ),
        "Track B codebook halves overlap",
    )
    _require(
        a_lock["qualification_surface"]["seed_range"] == list(QUALIFICATION_SEED_RANGE),
        "qualification seed range drift",
    )
    _require(
        a_lock["development_selection"]["selection_thresholds"] == DEVELOPMENT_THRESHOLDS,
        "development selection threshold drift",
    )
    for stage, thresholds in QUALIFICATION_THRESHOLDS.items():
        stored = a_lock["qualification_surface"][stage]["thresholds"]
        if stage == "STREAM-A1":
            stored = dict(stored)
            stored["control_independent_qualification_ceiling"] = stored.pop(
                "each_control_independent_qualification_ceiling_exclusive"
            )
        _require(stored == thresholds, f"{stage} threshold drift")

    operator = operator_lock_payload()
    observation = b_lock["observation_surface"]
    for key in (
        "layer_set",
        "token_position",
        "activation_representation",
        "frame_estimator",
        "frame_rank",
        "frame_relative_singular_tolerance",
        "edge_estimator",
        "edge_ridge_relative",
        "edge_rank_tolerance",
        "edge_condition_ceiling",
        "minimum_node_support_per_half",
    ):
        _require(observation[key] == operator[key], f"operator field drift: {key}")
    _require(
        b_lock["qualification"]["split_half_singular_floor_max"]
        == operator["split_half_singular_floor_max"],
        "split-half ceiling drift",
    )
    _require(
        b_lock["qualification"]["minimum_qualified_layer_count"]
        == operator["minimum_qualified_layer_count"],
        "minimum layer count drift",
    )

    for lock in (a_lock, b_lock):
        runtime = lock["runtime_binding"]
        _require(runtime["model_repository"] == MODEL_REPOSITORY, "model repository drift")
        _require(runtime["model_revision"] == MODEL_REVISION, "model revision drift")
        _require(runtime["tokenizer_revision"] == TOKENIZER_REVISION, "tokenizer revision drift")
        _require(runtime["gpu"] == "NVIDIA L40S", "GPU drift")
        _require(runtime["dtype"] == "bfloat16", "dtype drift")
        _require(runtime["quantization"] is False, "quantization drift")

    total_max = (
        int(a_lock["budget"]["development_forwards_consumed"])
        + int(a_lock["qualification_surface"]["maximum_conditional_forward_count"])
        + int(b_lock["collection_surface"]["forward_count"])
    )
    _require(total_max == 534, "maximum campaign ledger count drift")
    _require(total_max <= CAMPAIGN_FORWARD_CEILING, "campaign forward budget exceeded")
    _require(
        float(a_lock["budget"]["campaign_incremental_modal_spend_ceiling_usd"])
        == CAMPAIGN_SPEND_CEILING_USD,
        "campaign spend ceiling drift",
    )
    _require(a_lock["closed_surfaces"]["TRACK_C"] == "CLOSED", "Track C opened in A lock")
    _require(b_lock["closed_surfaces"]["TRACK_C_execution"] == "FORBIDDEN", "Track C opened in B lock")
    _require(b_lock["forbidden_primary_claim"] == "generic nonzero representation holonomy", "novelty boundary drift")
    _require(b_lock["activation_collection_gate"] == "STREAM-A2_PASS_ONLY", "B gate drift")

    clean = None
    if require_clean:
        if repo_root is None:
            raise SuccessorLockError("repo_root required for clean validation")
        clean = _git_clean(repo_root)
        _require(clean, "Git status is not clean")
    return {
        "schema_version": "gate13_stepwise_successor_lock_validation_v1",
        "status": "PASS",
        "campaign_binding_id": a_lock["campaign_binding_id"],
        "selected_variant": variant,
        "track_a_lock_sha256": __import__("hashlib").sha256(a_path.read_bytes()).hexdigest(),
        "track_b_lock_sha256": __import__("hashlib").sha256(b_path.read_bytes()).hexdigest(),
        "track_a_ledger_sha256": a_ledger["sha256"],
        "track_b_collection_ledger_sha256": b_ledger["sha256"],
        "maximum_campaign_forward_count": total_max,
        "git_clean": clean,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--require-clean", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = validate_successor_locks(
        lock_dir=args.lock_dir,
        repo_root=args.repo_root,
        require_clean=args.require_clean,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
