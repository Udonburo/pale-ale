#!/usr/bin/env python3
"""Build a local-only Gate12C-2 development freeze-candidate receipt.

The command exercises one deliberately tiny, development-only outer unit for
S0, S1, and S2.  It verifies that the graph-derived hierarchy is executable
and records runtime and implementation provenance.  The resulting receipt is
not a calibration estimate, does not open locked seeds, and cannot authorize
N2 or a real held-out execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from gate12c2_synthetic_lab import (
    c2_freeze_candidate_specification,
    run_development_outer_experiment,
    run_development_s2_identification_experiment,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _run_timed(call: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    started = time.perf_counter()
    result = call()
    elapsed = time.perf_counter() - started
    return {
        "elapsed_seconds": elapsed,
        "result": result,
    }


def _outer_smoke_summary(
    timed_result: dict[str, Any],
) -> dict[str, Any]:
    result = timed_result["result"]
    endpoint_receipts = result["endpoint_receipts"]
    return {
        "elapsed_seconds": timed_result["elapsed_seconds"],
        "case_count": len(result["case_receipts"]),
        "endpoint_count": len(endpoint_receipts),
        "coverage_complete_endpoint_count": sum(
            bool(row["coverage_complete"]) for row in endpoint_receipts
        ),
        "result_payload_sha256": hashlib.sha256(
            _canonical_json(result).encode("utf-8")
        ).hexdigest(),
        "locked_execution_authorized": bool(
            result["locked_execution_authorized"]
        ),
        "performance_interpretation_authorized": False,
    }


def _s2_smoke_summary(
    timed_result: dict[str, Any],
) -> dict[str, Any]:
    result = timed_result["result"]
    endpoint_rows = result["endpoint_rows"]
    return {
        "elapsed_seconds": timed_result["elapsed_seconds"],
        "case_count": len(result["case_rows"]),
        "endpoint_count": len(endpoint_rows),
        "coverage_complete_endpoint_count": sum(
            bool(row["coverage_complete"]) for row in endpoint_rows
        ),
        "observed_process_unmodified": all(
            not bool(row["observed_process_modified"])
            for row in endpoint_rows
        ),
        "result_payload_sha256": hashlib.sha256(
            _canonical_json(result).encode("utf-8")
        ).hexdigest(),
        "locked_execution_authorized": bool(
            result["locked_execution_authorized"]
        ),
        "performance_interpretation_authorized": False,
    }


def build_receipt(
    *,
    repository_commit: str,
    master_seed: str,
) -> dict[str, Any]:
    if not repository_commit.strip():
        raise ValueError("repository_commit must be nonempty")
    if not master_seed.strip():
        raise ValueError("master_seed must be nonempty")

    smoke_configuration = {
        "surface_id": "development",
        "outer_experiment_count_per_regime": 1,
        "S0": {
            "block_count_per_case": 6,
            "inner_valid_draw_count": 3,
        },
        "S1": {
            "block_count_per_case": 6,
            "inner_valid_draw_count": 3,
            "effect_strength": 0.25,
        },
        "S2": {
            "block_count_per_case": 4,
            "inner_valid_draw_count": 2,
        },
        "purpose": "runtime_and_end_to_end_schema_smoke_only",
        "calibration_gate_decision_authorized": False,
    }
    s0 = _run_timed(
        lambda: run_development_outer_experiment(
            regime_id="S0_true_null",
            master_seed=master_seed,
            outer_experiment_index=0,
            block_count=6,
            inner_valid_draw_count=3,
            max_draw_attempts=16,
        )
    )
    s1 = _run_timed(
        lambda: run_development_outer_experiment(
            regime_id="S1_known_reverse_shared_node_coupling",
            master_seed=master_seed,
            outer_experiment_index=0,
            block_count=6,
            inner_valid_draw_count=3,
            effect_strength=0.25,
            max_draw_attempts=16,
        )
    )
    s2 = _run_timed(
        lambda: run_development_s2_identification_experiment(
            master_seed=master_seed,
            outer_experiment_index=0,
            block_count=4,
            inner_valid_draw_count=2,
            max_draw_attempts=12,
        )
    )

    laboratory_path = Path(__file__).with_name(
        "gate12c2_synthetic_lab.py"
    )
    builder_path = Path(__file__).resolve()
    payload: dict[str, Any] = {
        "schema_version": "gate12c2_development_freeze_receipt_v0.2",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "epistemic_status": (
            "development_freeze_candidate_runtime_smoke_not_calibration"
        ),
        "authorization": {
            "locked_synthetic_execution": False,
            "real_held_out_execution": False,
            "N2_implementation": False,
            "N3_implementation": False,
            "public_claim": False,
        },
        "repository_commit": repository_commit,
        "master_seed_id": master_seed,
        "master_seed_receipt_sha256": hashlib.sha256(
            master_seed.encode("utf-8")
        ).hexdigest(),
        "decision_specification": c2_freeze_candidate_specification(),
        "smoke_configuration": smoke_configuration,
        "smoke_results": {
            "S0": _outer_smoke_summary(s0),
            "S1": _outer_smoke_summary(s1),
            "S2": _s2_smoke_summary(s2),
        },
        "explicitly_pending": [
            "nuisance_fidelity_thresholds",
            "runtime_and_storage_limits_at_candidate_draw_counts",
            "primary_S1_effect_strength",
            "outer_experiment_counts",
            "accepted_valid_255_511_1023_runtime_study",
            "locked_seed_manifest",
            "mechanical_no_peek_preflight",
        ],
        "implementation_sha256": {
            laboratory_path.name: _sha256(laboratory_path),
            builder_path.name: _sha256(builder_path),
        },
    }
    payload["receipt_payload_sha256"] = hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-commit", required=True)
    parser.add_argument("--master-seed", required=True)
    args = parser.parse_args()

    receipt = build_receipt(
        repository_commit=str(args.repository_commit),
        master_seed=str(args.master_seed),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "receipt_payload_sha256": receipt[
                    "receipt_payload_sha256"
                ],
                "epistemic_status": receipt["epistemic_status"],
                "locked_synthetic_execution_authorized": False,
                "real_held_out_execution_authorized": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
