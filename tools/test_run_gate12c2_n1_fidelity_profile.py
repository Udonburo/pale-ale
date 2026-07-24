#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


def _load(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load("run_gate12c2_n1_fidelity_profile")


class RunGate12C2N1FidelityProfileTest(unittest.TestCase):
    def test_receipt_keeps_all_locked_authorizations_closed(self) -> None:
        boundary = {
            "schema_version": runner.BOUNDARY_SCHEMA_VERSION,
            "repository_commit": "boundary-unit-test",
            "authorization": {
                "locked_synthetic_execution": False,
                "real_held_out_execution": False,
                "N2_implementation": False,
                "N3_implementation": False,
                "public_claim": False,
            },
        }
        boundary["receipt_payload_sha256"] = hashlib.sha256(
            runner._canonical_json(boundary).encode("utf-8")
        ).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            boundary_path = Path(directory) / "boundary.json"
            boundary_path.write_text(
                json.dumps(boundary),
                encoding="utf-8",
            )
            receipt = runner.build_receipt(
                boundary_receipt_path=boundary_path,
                repository_commit="unit-test",
                master_seed="runner-unit-test",
                block_count=4,
                draw_count_per_case=1,
                effect_strengths=(0.15,),
            )
        self.assertFalse(
            receipt["authorization"]["locked_synthetic_execution"]
        )
        self.assertFalse(
            receipt["authorization"]["real_held_out_execution"]
        )
        self.assertFalse(receipt["authorization"]["N2_implementation"])
        self.assertFalse(receipt["authorization"]["public_claim"])
        self.assertIn("receipt_payload_sha256", receipt)


if __name__ == "__main__":
    unittest.main()
