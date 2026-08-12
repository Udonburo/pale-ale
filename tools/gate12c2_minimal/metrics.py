"""Compression-composition residual diagnostics from Annex A."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


class Gate12C2MetricError(ValueError):
    """Raised when a matrix triple or diagnostic is invalid."""


@dataclass(frozen=True)
class ResidualDiagnostics:
    q: int
    eligible: bool
    eligibility: str
    numerical_pass: bool
    a: float
    u: float
    v: float
    x: float
    y: float
    c: float | None
    p_left: float | None
    p_right: float | None
    relative_gap_left: float
    relative_gap_right: float
    product_singular_values_left: tuple[float, ...]
    product_singular_values_right: tuple[float, ...]
    matrix_identity_error: float
    squared_identity_error: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _matrix(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise Gate12C2MetricError(f"{name} must be a finite matrix")
    return result


def _truncated_reconstruction(
    value: np.ndarray,
    q: int,
    *,
    spectral_gap_tolerance: float,
) -> tuple[np.ndarray, float, bool]:
    rows, columns = value.shape
    limit = min(rows, columns)
    if not 1 <= q < limit:
        raise Gate12C2MetricError(f"q must satisfy 1 <= q < {limit}")
    left, singular, right = np.linalg.svd(value, full_matrices=False)
    scale = max(float(singular[0]), np.finfo(np.float64).tiny)
    relative_gap = float((singular[q - 1] - singular[q]) / scale)
    reconstruction = (left[:, :q] * singular[:q]) @ right[:q, :]
    return reconstruction, relative_gap, relative_gap > spectral_gap_tolerance


def residual_diagnostics(
    m0: np.ndarray,
    m1: np.ndarray,
    m2: np.ndarray,
    q: int,
    *,
    spectral_gap_tolerance: float = 1e-10,
    numerical_tolerance: float = 1e-9,
    degeneracy_tolerance: float = 1e-12,
) -> ResidualDiagnostics:
    """Evaluate ``a,u,v,x,y,c,p_L,p_R`` and both exact identities.

    Matrix shapes follow a typed cycle: ``m0: V0 -> V1``, ``m1: V1 ->
    V2``, and ``m2: V2 -> V0``.
    """

    if not math.isfinite(spectral_gap_tolerance) or spectral_gap_tolerance < 0:
        raise Gate12C2MetricError("spectral gap tolerance must be nonnegative")
    if not math.isfinite(numerical_tolerance) or numerical_tolerance <= 0:
        raise Gate12C2MetricError("numerical tolerance must be positive")
    if not math.isfinite(degeneracy_tolerance) or degeneracy_tolerance < 0:
        raise Gate12C2MetricError("degeneracy tolerance must be nonnegative")

    m0_array = _matrix(m0, "m0")
    m1_array = _matrix(m1, "m1")
    m2_array = _matrix(m2, "m2")
    if (
        m1_array.shape[1] != m0_array.shape[0]
        or m2_array.shape[1] != m1_array.shape[0]
        or m2_array.shape[0] != m0_array.shape[1]
    ):
        raise Gate12C2MetricError("matrix triple is not a composable cycle")

    product_left = m2_array @ m1_array
    product_right = m1_array @ m0_array
    product_singular_values_left = tuple(
        float(value)
        for value in np.linalg.svd(
            product_left, full_matrices=False, compute_uv=False
        )
    )
    product_singular_values_right = tuple(
        float(value)
        for value in np.linalg.svd(
            product_right, full_matrices=False, compute_uv=False
        )
    )
    q_left, gap_left, eligible_left = _truncated_reconstruction(
        product_left, q, spectral_gap_tolerance=spectral_gap_tolerance
    )
    q_right, gap_right, eligible_right = _truncated_reconstruction(
        product_right, q, spectral_gap_tolerance=spectral_gap_tolerance
    )

    residual_left = product_left - q_left
    residual_right = product_right - q_right
    propagated_left = residual_left @ m0_array
    propagated_right = m2_array @ residual_right
    defect_matrix = q_left @ m0_array - m2_array @ q_right
    decomposition = propagated_right - propagated_left

    a = float(np.linalg.norm(defect_matrix, ord="fro"))
    u = float(np.linalg.norm(residual_left, ord="fro"))
    v = float(np.linalg.norm(residual_right, ord="fro"))
    x = float(np.linalg.norm(propagated_left, ord="fro"))
    y = float(np.linalg.norm(propagated_right, ord="fro"))
    inner = float(np.real(np.vdot(propagated_left, propagated_right)))
    c = None if x <= degeneracy_tolerance or y <= degeneracy_tolerance else max(
        -1.0, min(1.0, inner / (x * y))
    )
    p_left = None if u <= degeneracy_tolerance else x / u
    p_right = None if v <= degeneracy_tolerance else y / v

    matrix_error = float(np.linalg.norm(defect_matrix - decomposition, ord="fro"))
    squared_rhs = x * x + y * y - 2.0 * inner
    squared_error = float(abs(a * a - squared_rhs))
    matrix_scale = max(1.0, a, x, y)
    squared_scale = max(1.0, a * a, x * x + y * y)
    finite = all(
        math.isfinite(value)
        for value in (a, u, v, x, y, matrix_error, squared_error)
    )
    numerical_pass = bool(
        finite
        and matrix_error <= numerical_tolerance * matrix_scale
        and squared_error <= numerical_tolerance * squared_scale
    )
    eligible = bool(eligible_left and eligible_right)
    if eligible:
        eligibility = "eligible"
    elif not eligible_left and not eligible_right:
        eligibility = "unstable_cut_both"
    elif not eligible_left:
        eligibility = "unstable_cut_left"
    else:
        eligibility = "unstable_cut_right"

    return ResidualDiagnostics(
        q=int(q),
        eligible=eligible,
        eligibility=eligibility,
        numerical_pass=numerical_pass,
        a=a,
        u=u,
        v=v,
        x=x,
        y=y,
        c=c,
        p_left=p_left,
        p_right=p_right,
        relative_gap_left=gap_left,
        relative_gap_right=gap_right,
        product_singular_values_left=product_singular_values_left,
        product_singular_values_right=product_singular_values_right,
        matrix_identity_error=matrix_error,
        squared_identity_error=squared_error,
    )
