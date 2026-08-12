"""Validate one completed minimal Gate12C-2 run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .experiment import (
    expected_shard_ids,
    load_spec,
    summarize_shards,
    validate_shard,
)
from .io import canonical_json_bytes, load_json, sha256_file
from .run import MANIFEST_SCHEMA, STATE_SCHEMA, implementation_hashes


class Gate12C2ValidationError(ValueError):
    """Raised when a completed run does not reproduce from its shards."""


def validate_run(spec_path: Path, output: Path) -> dict[str, Any]:
    spec, spec_sha256 = load_spec(spec_path)
    output = output.resolve()
    manifest = load_json(output / "manifest.json")
    state = load_json(output / "state.json")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise Gate12C2ValidationError("invalid manifest")
    if not isinstance(state, dict) or state.get("schema_version") != STATE_SCHEMA:
        raise Gate12C2ValidationError("invalid state")
    if state.get("state") != "COMPLETE" or state.get("study_sha256") != spec_sha256:
        raise Gate12C2ValidationError("run is not complete for this study")
    if manifest.get("study_sha256") != spec_sha256:
        raise Gate12C2ValidationError("manifest study hash mismatch")
    if manifest.get("implementation") != implementation_hashes():
        raise Gate12C2ValidationError("implementation hash mismatch")
    expected_ids = expected_shard_ids(spec)
    if manifest.get("expected_shard_ids") != expected_ids:
        raise Gate12C2ValidationError("manifest shard set mismatch")

    expected_files = {
        "study.json",
        "result.json",
        *(f"shards/{shard_id}.json" for shard_id in expected_ids),
    }
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != expected_files:
        raise Gate12C2ValidationError("manifest file surface mismatch")
    for relative, expected_hash in files.items():
        path = output / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise Gate12C2ValidationError(f"file hash mismatch: {relative}")
    if sha256_file(output / "study.json") != spec_sha256:
        raise Gate12C2ValidationError("stored study bytes differ")

    shards = [
        validate_shard(
            load_json(output / "shards" / f"{shard_id}.json"),
            spec=spec,
            spec_sha256=spec_sha256,
            shard_id=shard_id,
        )
        for shard_id in expected_ids
    ]
    recomputed = summarize_shards(spec, spec_sha256, shards)
    stored_result = load_json(output / "result.json")
    if canonical_json_bytes(recomputed) != canonical_json_bytes(stored_result):
        raise Gate12C2ValidationError("aggregate result does not reproduce")
    return {
        "status": "pass",
        "decision": recomputed["decision"],
        "study_sha256": spec_sha256,
        "manifest_sha256": sha256_file(output / "manifest.json"),
        "shard_count": len(shards),
        "legacy_payload_accessed": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        sys.stdout.buffer.write(canonical_json_bytes(validate_run(args.spec, args.output)))
        return 0
    except Exception as exc:
        sys.stderr.write(f"gate12c2-minimal validation failed: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
