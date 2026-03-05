#!/usr/bin/env python3
"""Build and verify E8 root vectors used by E2 vec8 soft snap."""

import datetime as _dt
import hashlib
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

NORM_TOL = 1e-12
ROOT_A_ABS = 1.0 / math.sqrt(2.0)
ROOT_B_ABS = 0.5 / math.sqrt(2.0)


def _norm(vec: Sequence[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in vec))


def _root_line(vec: Sequence[float]) -> str:
    return " ".join(f"{float(x):.17e}" for x in vec)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_e8_roots() -> List[List[float]]:
    roots: List[List[float]] = []

    # Group A: 112 roots, two non-zero coordinates +/-1/sqrt(2).
    for i in range(8):
        for j in range(i + 1, 8):
            for sign_i in (-1.0, 1.0):
                for sign_j in (-1.0, 1.0):
                    vec = [0.0] * 8
                    vec[i] = sign_i * ROOT_A_ABS
                    vec[j] = sign_j * ROOT_A_ABS
                    roots.append(vec)

    # Group B: 128 roots, all coordinates +/- (1/2)/sqrt(2), even negatives.
    for mask in range(1 << 8):
        neg_count = 0
        vec = [0.0] * 8
        for i in range(8):
            is_neg = ((mask >> i) & 1) == 1
            if is_neg:
                neg_count += 1
            vec[i] = (-ROOT_B_ABS) if is_neg else ROOT_B_ABS
        if (neg_count % 2) == 0:
            roots.append(vec)

    return roots


def _classify_root(vec: Sequence[float], tol: float = 1e-12) -> str:
    abs_vals = [abs(float(x)) for x in vec]
    non_zero = [v for v in abs_vals if v > tol]

    if len(non_zero) == 2 and all(abs(v - ROOT_A_ABS) <= tol for v in non_zero):
        return "A"
    if len(non_zero) == 8 and all(abs(v - ROOT_B_ABS) <= tol for v in non_zero):
        return "B"
    return "?"


def verify_e8_roots(roots: Sequence[Sequence[float]]) -> Dict[str, object]:
    if len(roots) != 240:
        raise ValueError(f"E8 roots count must be 240, got {len(roots)}")

    group_a = 0
    group_b = 0
    max_norm_error = 0.0
    lines: List[str] = []

    for idx, root in enumerate(roots):
        if len(root) != 8:
            raise ValueError(f"root[{idx}] must have length 8")
        if not all(math.isfinite(float(v)) for v in root):
            raise ValueError(f"root[{idx}] contains non-finite values")

        n = _norm(root)
        norm_error = abs(n - 1.0)
        max_norm_error = max(max_norm_error, norm_error)
        if norm_error > NORM_TOL:
            raise ValueError(f"root[{idx}] norm error too large: {norm_error}")

        cls = _classify_root(root)
        if cls == "A":
            group_a += 1
        elif cls == "B":
            group_b += 1
        else:
            raise ValueError(f"root[{idx}] does not match E8 pattern")

        lines.append(_root_line(root))

    if group_a != 112 or group_b != 128:
        raise ValueError(f"unexpected group counts: A={group_a}, B={group_b}")

    all_lines_text = "\n".join(lines) + "\n"
    roots_hash_sha256 = _sha256_text(all_lines_text)
    unique_count = len(set(lines))
    if unique_count != 240:
        raise ValueError(f"duplicate roots detected: unique_count={unique_count}")

    sample_indices = [0, 1, 119, 120, 239]
    sample_hashes = {str(i): _sha256_text(lines[i] + "\n") for i in sample_indices}

    return {
        "root_count": 240,
        "group_a_count": group_a,
        "group_b_count": group_b,
        "max_norm_error": max_norm_error,
        "roots_hash_sha256": roots_hash_sha256,
        "sample_root_hashes_sha256": sample_hashes,
    }


def write_verification_record(
    roots: Sequence[Sequence[float]], out_path: Path
) -> Tuple[Path, Dict[str, object], str]:
    info = verify_e8_roots(roots)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append(f"date={_dt.date.today().isoformat()}")
    lines.append(f"root_count={info['root_count']}")
    lines.append(f"group_a_count={info['group_a_count']}")
    lines.append(f"group_b_count={info['group_b_count']}")
    lines.append(f"max_norm_error={info['max_norm_error']:.17e}")
    lines.append(f"roots_hash_sha256={info['roots_hash_sha256']}")
    lines.append("sample_root_hashes_sha256:")
    sample_hashes = info["sample_root_hashes_sha256"]
    for key in sorted(sample_hashes.keys(), key=lambda x: int(x)):
        lines.append(f"  root_{key}={sample_hashes[key]}")
    text = "\n".join(lines) + "\n"

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)

    record_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return out_path, info, record_sha256

