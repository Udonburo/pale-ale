"""Execute or resume the one bounded Gate12C-2 locked synthetic suite."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from .experiment import run_dataset
from .io import canonical_json_bytes, load_json, sha256_file, write_bytes_atomic, write_json_atomic
from .locked_calibration import (
    LOCKED_ANALYSIS_SCHEMA,
    analyze_locked_shards,
    compact_dataset_shard,
    current_environment,
    expected_dataset_shards,
    generation_spec,
    load_locked_spec,
    shard_id,
    shard_relative_path,
    validate_compact_shard,
)


LOCKED_STATE_SCHEMA = "gate12c2_locked_calibration_state_v0.1"
LOCKED_MANIFEST_SCHEMA = "gate12c2_locked_calibration_manifest_v0.1"


class LockedRunError(RuntimeError):
    """Raised when a locked attempt cannot safely start or resume."""


def package_root() -> Path:
    return Path(__file__).resolve().parent


def actual_implementation_hashes(names: list[str] | None = None) -> dict[str, str]:
    root = package_root()
    selected = names or [
        "io.py",
        "metrics.py",
        "generators.py",
        "experiment.py",
        "locked_calibration.py",
        "locked_run.py",
        "locked_validate.py",
    ]
    return {name: sha256_file(root / name) for name in selected}


def verify_frozen_implementation(spec: dict[str, Any]) -> None:
    expected = spec["implementation_sha256"]
    actual = actual_implementation_hashes(sorted(expected))
    if actual != expected:
        differences = sorted(
            name for name in expected if actual.get(name) != expected.get(name)
        )
        raise LockedRunError(f"implementation differs: {differences}")


def _state(
    spec: dict[str, Any],
    spec_sha256: str,
    state: str,
    error: str | None,
    *,
    elapsed_wall_seconds: float,
    completed_shard_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": LOCKED_STATE_SCHEMA,
        "study_id": spec["study_id"],
        "study_sha256": spec_sha256,
        "attempt_id": spec["attempt_id"],
        "state": state,
        "error": error,
        "elapsed_wall_seconds": float(elapsed_wall_seconds),
        "completed_shard_count": int(completed_shard_count),
    }


def _output_bytes(output: Path) -> int:
    return sum(path.stat().st_size for path in output.rglob("*") if path.is_file())


def execute_locked(spec_path: Path, output: Path, *, resume: bool = False) -> dict[str, Any]:
    spec, spec_sha256 = load_locked_spec(spec_path)
    verify_frozen_implementation(spec)
    output = output.resolve()
    state_path = output / "state.json"
    if output.exists():
        if not resume:
            raise LockedRunError(f"locked output already exists: {output}")
        stored_spec = output / "study.json"
        if not stored_spec.is_file() or sha256_file(stored_spec) != spec_sha256:
            raise LockedRunError("resume study bytes differ")
        state = load_json(state_path)
        if not isinstance(state, dict) or state.get("state") not in {"RUNNING", "FAILED"}:
            raise LockedRunError("locked attempt is not resumable")
        previous_elapsed = float(state.get("elapsed_wall_seconds", -1.0))
        previous_completed = int(state.get("completed_shard_count", -1))
        if (
            not 0.0 <= previous_elapsed <= float(spec["resource_cap"]["max_wall_seconds"])
            or not 0 <= previous_completed <= int(spec["resource_cap"]["max_dataset_shards"])
        ):
            raise LockedRunError("resume resource accounting differs")
    else:
        output.mkdir(parents=True)
        write_bytes_atomic(output / "study.json", spec_path.read_bytes())
        previous_elapsed = 0.0
        previous_completed = 0
        write_json_atomic(
            state_path,
            _state(
                spec,
                spec_sha256,
                "FROZEN",
                None,
                elapsed_wall_seconds=0.0,
                completed_shard_count=0,
            ),
        )

    started = time.monotonic()
    write_json_atomic(
        state_path,
        _state(
            spec,
            spec_sha256,
            "RUNNING",
            None,
            elapsed_wall_seconds=previous_elapsed,
            completed_shard_count=previous_completed,
        ),
        replace=True,
    )
    output_bytes = _output_bytes(output)
    shards: list[dict[str, Any]] = []
    try:
        cases = {case["case_id"]: case for case in spec["cases"]}
        cap = spec["resource_cap"]
        for config_id, case_id, dataset_index in expected_dataset_shards(spec):
            relative = shard_relative_path(config_id, case_id, dataset_index)
            path = output / relative
            if path.exists():
                if not resume:
                    raise LockedRunError(f"unexpected locked shard: {relative}")
                shard = validate_compact_shard(
                    load_json(path),
                    spec=spec,
                    spec_sha256=spec_sha256,
                    config_id=config_id,
                    case_id=case_id,
                    dataset_index=dataset_index,
                )
            else:
                generation = generation_spec(spec, config_id)
                regime = "S1" if config_id.startswith("S1_") else config_id
                dataset = run_dataset(
                    generation,
                    spec_sha256,
                    cases[case_id],
                    regime,
                    dataset_index,
                )
                shard = compact_dataset_shard(
                    dataset,
                    spec=spec,
                    spec_sha256=spec_sha256,
                    config_id=config_id,
                )
                write_json_atomic(path, shard)
                output_bytes += path.stat().st_size
            shards.append(shard)
            elapsed = previous_elapsed + time.monotonic() - started
            if elapsed > float(cap["max_wall_seconds"]):
                raise LockedRunError("locked wall-time cap exceeded")
            if output_bytes > int(cap["max_output_bytes"]):
                raise LockedRunError("locked output-byte cap exceeded")
            previous_state_bytes = state_path.stat().st_size
            write_json_atomic(
                state_path,
                _state(
                    spec,
                    spec_sha256,
                    "RUNNING",
                    None,
                    elapsed_wall_seconds=elapsed,
                    completed_shard_count=len(shards),
                ),
                replace=True,
            )
            output_bytes += state_path.stat().st_size - previous_state_bytes
            if output_bytes > int(cap["max_output_bytes"]):
                raise LockedRunError("locked state-byte cap exceeded")

        analysis = analyze_locked_shards(spec, spec_sha256, shards)
        if analysis.get("schema_version") != LOCKED_ANALYSIS_SCHEMA:
            raise LockedRunError("primary analysis schema differs")
        analysis_path = output / "analysis.json"
        write_json_atomic(analysis_path, analysis)
        output_bytes += analysis_path.stat().st_size
        file_hashes = {
            "study.json": spec_sha256,
            "analysis.json": sha256_file(analysis_path),
        }
        for config_id, case_id, dataset_index in expected_dataset_shards(spec):
            relative = shard_relative_path(config_id, case_id, dataset_index)
            file_hashes[relative] = sha256_file(output / relative)
        manifest = {
            "schema_version": LOCKED_MANIFEST_SCHEMA,
            "study_id": spec["study_id"],
            "study_sha256": spec_sha256,
            "attempt_id": spec["attempt_id"],
            "environment": current_environment(),
            "implementation_sha256": actual_implementation_hashes(
                sorted(spec["implementation_sha256"])
            ),
            "dataset_is_inference_unit": True,
            "stressor_selection_count": 1,
            "shard_count": len(shards),
            "output_bytes_before_manifest": output_bytes,
            "files": file_hashes,
        }
        manifest_path = output / "manifest.json"
        write_json_atomic(manifest_path, manifest)
        output_bytes += manifest_path.stat().st_size
        if output_bytes > int(cap["max_output_bytes"]):
            raise LockedRunError("locked final output-byte cap exceeded")
        final_elapsed = previous_elapsed + time.monotonic() - started
        if final_elapsed > float(cap["max_wall_seconds"]):
            raise LockedRunError("locked final wall-time cap exceeded")
        write_json_atomic(
            state_path,
            _state(
                spec,
                spec_sha256,
                "COMPLETE",
                None,
                elapsed_wall_seconds=final_elapsed,
                completed_shard_count=len(shards),
            ),
            replace=True,
        )
        if _output_bytes(output) > int(cap["max_output_bytes"]):
            raise LockedRunError("locked completed output-byte cap exceeded")
        return {
            "decision": analysis["decision"],
            "locked_pass": analysis["locked_pass"],
            "study_sha256": spec_sha256,
            "analysis_sha256": sha256_file(analysis_path),
            "manifest_sha256": sha256_file(manifest_path),
            "shard_count": len(shards),
            "output": str(output),
        }
    except Exception as exc:
        elapsed = previous_elapsed + time.monotonic() - started
        write_json_atomic(
            state_path,
            _state(
                spec,
                spec_sha256,
                "FAILED",
                f"{type(exc).__name__}: {exc}",
                elapsed_wall_seconds=elapsed,
                completed_shard_count=len(shards),
            ),
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
            canonical_json_bytes(
                execute_locked(args.spec, args.output, resume=args.resume)
            )
        )
        return 0
    except Exception as exc:
        sys.stderr.write(f"gate12c2 locked run failed: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
