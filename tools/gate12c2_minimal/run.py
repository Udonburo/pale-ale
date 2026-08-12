"""Run or resume the minimal Gate12C-2 synthetic development smoke."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .experiment import (
    REGIMES,
    expected_shard_ids,
    load_spec,
    run_shard,
    summarize_shards,
    validate_shard,
)
from .io import (
    canonical_json_bytes,
    load_json,
    sha256_file,
    write_bytes_atomic,
    write_json_atomic,
)


STATE_SCHEMA = "gate12c2_minimal_state_v0.1"
MANIFEST_SCHEMA = "gate12c2_minimal_manifest_v0.1"
IMPLEMENTATION_FILES = (
    "__init__.py",
    "io.py",
    "metrics.py",
    "generators.py",
    "experiment.py",
    "run.py",
    "validate.py",
)


class Gate12C2RunError(RuntimeError):
    """Raised when a run cannot safely start or resume."""


def implementation_hashes() -> dict[str, str]:
    package_root = Path(__file__).resolve().parent
    return {
        name: sha256_file(package_root / name) for name in IMPLEMENTATION_FILES
    }


def _state(study_sha256: str, state: str, error: str | None = None) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA,
        "study_sha256": study_sha256,
        "state": state,
        "error": error,
    }


def execute(spec_path: Path, output: Path, *, resume: bool = False) -> dict[str, Any]:
    spec, spec_sha256 = load_spec(spec_path)
    output = output.resolve()
    state_path = output / "state.json"
    initialized = False
    if output.exists():
        if not resume:
            raise Gate12C2RunError(f"output already exists: {output}")
        stored_study = output / "study.json"
        if not stored_study.is_file() or sha256_file(stored_study) != spec_sha256:
            raise Gate12C2RunError("resume study bytes do not match")
        state = load_json(state_path)
        if not isinstance(state, dict) or state.get("state") == "COMPLETE":
            raise Gate12C2RunError("completed run cannot be resumed")
        if state.get("state") not in {"RUNNING", "FAILED"}:
            raise Gate12C2RunError("run is not resumable")
        initialized = True
    else:
        output.mkdir(parents=True)
        write_bytes_atomic(output / "study.json", spec_path.read_bytes())
        write_json_atomic(state_path, _state(spec_sha256, "FROZEN"))
        initialized = True

    write_json_atomic(state_path, _state(spec_sha256, "RUNNING"), replace=True)
    try:
        shard_root = output / "shards"
        shards: list[dict[str, Any]] = []
        case_by_id = {case["case_id"]: case for case in spec["cases"]}
        for shard_id in expected_shard_ids(spec):
            case_id, regime = shard_id.split("__", 1)
            shard_path = shard_root / f"{shard_id}.json"
            if shard_path.exists():
                if not resume:
                    raise Gate12C2RunError(f"unexpected existing shard: {shard_id}")
                shard = validate_shard(
                    load_json(shard_path),
                    spec=spec,
                    spec_sha256=spec_sha256,
                    shard_id=shard_id,
                )
            else:
                shard = run_shard(
                    spec, spec_sha256, case_by_id[case_id], regime
                )
                write_json_atomic(shard_path, shard)
            shards.append(shard)

        result = summarize_shards(spec, spec_sha256, shards)
        result_path = output / "result.json"
        write_json_atomic(result_path, result, replace=resume and result_path.exists())
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "study_id": spec["study_id"],
            "study_sha256": spec_sha256,
            "shard_count": len(shards),
            "expected_shard_ids": expected_shard_ids(spec),
            "implementation": implementation_hashes(),
            "files": {
                "study.json": spec_sha256,
                "result.json": sha256_file(result_path),
                **{
                    f"shards/{shard_id}.json": sha256_file(
                        shard_root / f"{shard_id}.json"
                    )
                    for shard_id in expected_shard_ids(spec)
                },
            },
        }
        manifest_path = output / "manifest.json"
        write_json_atomic(
            manifest_path, manifest, replace=resume and manifest_path.exists()
        )
        write_json_atomic(state_path, _state(spec_sha256, "COMPLETE"), replace=True)
        return {
            "decision": result["decision"],
            "smoke_pass": result["smoke_pass"],
            "output": str(output),
            "manifest_sha256": sha256_file(manifest_path),
        }
    except Exception as exc:
        if initialized:
            write_json_atomic(
                state_path,
                _state(spec_sha256, "FAILED", f"{type(exc).__name__}: {exc}"),
                replace=True,
            )
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        sys.stdout.buffer.write(
            canonical_json_bytes(execute(args.spec, args.output, resume=args.resume))
        )
        return 0
    except Exception as exc:
        sys.stderr.write(f"gate12c2-minimal run failed: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
