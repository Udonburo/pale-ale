"""Shared fail-closed utilities for the Gate13 candidate Phase 2 locks."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, value: object) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def git_output(args: Sequence[str], *, cwd: Path = REPO_ROOT) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def git_head(*, cwd: Path = REPO_ROOT) -> str:
    return git_output(["rev-parse", "HEAD"], cwd=cwd)


def git_status_porcelain(*, cwd: Path = REPO_ROOT) -> str:
    return git_output(["status", "--porcelain", "--untracked-files=all"], cwd=cwd)


def git_path_blob_sha(commit: str, path: str, *, cwd: Path = REPO_ROOT) -> str:
    return git_output(["rev-parse", f"{commit}:{path}"], cwd=cwd)


def require_sha256(value: object, *, field: str) -> str:
    normalized = str(value or "").lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return normalized


def require_fields(value: Mapping[str, Any], fields: Sequence[str], *, context: str) -> None:
    missing = [field for field in fields if field not in value or value[field] in (None, "")]
    if missing:
        raise ValueError(f"{context} missing required fields: {', '.join(missing)}")
