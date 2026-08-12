"""Small deterministic JSON and hashing helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


class Gate12C2IOError(ValueError):
    """Raised when a persisted run artifact is malformed."""


def _reject_constant(value: str) -> None:
    raise Gate12C2IOError(f"non-finite JSON constant: {value}")


def _closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Gate12C2IOError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def load_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_closed_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Gate12C2IOError(f"cannot read JSON artifact {path}: {exc}") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    try:
        return sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise Gate12C2IOError(f"cannot hash {path}: {exc}") from exc


def write_bytes_atomic(path: Path, value: bytes, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_bytes(value)
        if path.exists() and not replace:
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_json_atomic(path: Path, value: object, *, replace: bool = False) -> None:
    write_bytes_atomic(path, canonical_json_bytes(value), replace=replace)


def require_canonical_json(path: Path) -> Any:
    value = load_json(path)
    if path.read_bytes() != canonical_json_bytes(value):
        raise Gate12C2IOError(f"JSON artifact is not canonical: {path}")
    return value

