"""Fresh graph-realizable S0/S1 generators and N1/S2 transformations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


class Gate12C2GeneratorError(ValueError):
    """Raised when a synthetic graph construction is invalid."""


def rng(seed_namespace: str, *parts: object) -> np.random.Generator:
    encoded = "\x1f".join(str(value) for value in (seed_namespace, *parts)).encode(
        "utf-8"
    )
    seed = int.from_bytes(hashlib.sha256(encoded).digest()[:16], "big")
    return np.random.default_rng(seed)


def orthonormalize(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < array.shape[1]:
        raise Gate12C2GeneratorError("frame must have rows >= columns")
    frame, upper = np.linalg.qr(array, mode="reduced")
    signs = np.where(np.diag(upper) < 0.0, -1.0, 1.0)
    return np.asarray(frame * signs, dtype=np.float64)


def edges_from_frames(
    frames: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(frames) != 3:
        raise Gate12C2GeneratorError("a synthetic cycle requires three frames")
    arrays = tuple(np.asarray(frame, dtype=np.float64) for frame in frames)
    if len({array.shape for array in arrays}) != 1:
        raise Gate12C2GeneratorError("all frames must share one shape")
    return (
        arrays[1].T @ arrays[0],
        arrays[2].T @ arrays[1],
        arrays[0].T @ arrays[2],
    )


def edge_singular_values(
    edges: Sequence[np.ndarray],
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    if len(edges) != 3:
        raise Gate12C2GeneratorError("a cycle requires exactly three edges")
    return tuple(
        tuple(
            float(value)
            for value in np.linalg.svd(
                np.asarray(edge, dtype=np.float64),
                full_matrices=False,
                compute_uv=False,
            )
        )
        for edge in edges
    )  # type: ignore[return-value]


@dataclass(frozen=True)
class Graph:
    replicate_id: str
    regime: str
    frames: tuple[np.ndarray, np.ndarray, np.ndarray]
    edges: tuple[np.ndarray, np.ndarray, np.ndarray]

    @classmethod
    def from_frames(
        cls,
        *,
        replicate_id: str,
        regime: str,
        frames: Sequence[np.ndarray],
    ) -> "Graph":
        copied = tuple(np.asarray(frame, dtype=np.float64).copy() for frame in frames)
        if len(copied) != 3:
            raise Gate12C2GeneratorError("a graph requires exactly three frames")
        return cls(
            replicate_id=replicate_id,
            regime=regime,
            frames=copied,  # type: ignore[arg-type]
            edges=edges_from_frames(copied),
        )


def graph_digest(graphs: Sequence[Graph]) -> str:
    digest = hashlib.sha256()
    for graph in graphs:
        digest.update(graph.replicate_id.encode("utf-8"))
        digest.update(b"\0")
        for array in (*graph.frames, *graph.edges):
            canonical = np.ascontiguousarray(array, dtype="<f8")
            digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
            digest.update(canonical.tobytes())
    return digest.hexdigest()


def generate_s0_cohort(
    *,
    case: Mapping[str, object],
    seed_namespace: str,
    outer_index: int,
    cohort_size: int,
    frame_noise: float,
    regime: str = "S0",
) -> tuple[Graph, ...]:
    ambient_dim = int(case["ambient_dim"])
    local_rank = int(case["local_rank"])
    if cohort_size < 4 or ambient_dim <= local_rank or local_rank < 3:
        raise Gate12C2GeneratorError("invalid S0 dimensions or cohort size")
    center_blend = float(case.get("role_center_blend", 0.0))
    if not 0.0 <= center_blend < 1.0:
        raise Gate12C2GeneratorError("role_center_blend must lie in [0, 1)")
    shared_center = orthonormalize(
        rng(
            seed_namespace,
            case["case_id"],
            regime,
            outer_index,
            "shared_center",
        ).normal(size=(ambient_dim, local_rank))
    )
    centers = tuple(
        orthonormalize(
            (center_blend**0.5) * shared_center
            + ((1.0 - center_blend) ** 0.5)
            * rng(
                seed_namespace,
                case["case_id"],
                regime,
                outer_index,
                "center",
                role,
            ).normal(size=(ambient_dim, local_rank))
        )
        for role in range(3)
    )
    graphs: list[Graph] = []
    for replicate_index in range(cohort_size):
        frames = tuple(
            orthonormalize(
                centers[role]
                + frame_noise
                * rng(
                    seed_namespace,
                    case["case_id"],
                    regime,
                    outer_index,
                    replicate_index,
                    role,
                ).normal(size=(ambient_dim, local_rank))
            )
            for role in range(3)
        )
        graphs.append(
            Graph.from_frames(
                replicate_id=(
                    f"{case['case_id']}:{regime}:o{outer_index:03d}:"
                    f"r{replicate_index:03d}"
                ),
                regime=regime,
                frames=frames,
            )
        )
    return tuple(graphs)


def generate_s1_cohort(
    *,
    case: Mapping[str, object],
    seed_namespace: str,
    outer_index: int,
    cohort_size: int,
    effect_strength: float,
    observed_mismatch: float,
) -> tuple[Graph, ...]:
    ambient_dim = int(case["ambient_dim"])
    local_rank = int(case["local_rank"])
    if cohort_size < 4 or ambient_dim <= local_rank or local_rank < 3:
        raise Gate12C2GeneratorError("invalid S1 dimensions or cohort size")
    prefix = (case["case_id"], "S1", outer_index)
    shared_center = orthonormalize(
        rng(seed_namespace, *prefix, "center").normal(
            size=(ambient_dim, local_rank)
        )
    )
    input_blend = float(case.get("s1_input_blend", 0.0))
    effect_multiplier = float(case.get("s1_effect_multiplier", 1.0))
    if not 0.0 <= input_blend < 1.0 or effect_multiplier <= 0.0:
        raise Gate12C2GeneratorError("invalid S1 geometry parameters")
    input_frame = orthonormalize(
        (input_blend**0.5) * shared_center
        + ((1.0 - input_blend) ** 0.5)
        * rng(seed_namespace, *prefix, "input").normal(
            size=(ambient_dim, local_rank)
        )
    )

    raw_tangent = rng(seed_namespace, *prefix, "tangent").normal(
        size=(ambient_dim, local_rank)
    )
    tangent = raw_tangent - shared_center @ (shared_center.T @ raw_tangent)
    tangent *= np.sqrt(local_rank) / np.linalg.norm(tangent, ord="fro")
    raw_offset = rng(seed_namespace, *prefix, "offset").normal(
        size=(ambient_dim, local_rank)
    )
    offset = raw_offset - shared_center @ (shared_center.T @ raw_offset)
    offset *= np.sqrt(local_rank) / np.linalg.norm(offset, ord="fro")

    graphs: list[Graph] = []
    for replicate_index, position in enumerate(
        np.linspace(-1.0, 1.0, cohort_size)
    ):
        bridge = orthonormalize(
            shared_center
            + effect_strength
            * effect_multiplier
            * float(position)
            * tangent
        )
        output = orthonormalize(bridge + observed_mismatch * offset)
        graphs.append(
            Graph.from_frames(
                replicate_id=(
                    f"{case['case_id']}:S1:o{outer_index:03d}:"
                    f"r{replicate_index:03d}"
                ),
                regime="S1",
                frames=(input_frame, bridge, output),
            )
        )
    return tuple(graphs)


def _derangement(size: int, generator: np.random.Generator) -> np.ndarray:
    identity = np.arange(size)
    for _ in range(128):
        candidate = generator.permutation(size)
        if np.all(candidate != identity):
            return candidate
    shift = int(generator.integers(1, size))
    return np.roll(identity, shift)


def n1_reassignment(
    graphs: Sequence[Graph],
    *,
    seed_namespace: str,
    case_id: str,
    regime: str,
    outer_index: int,
    draw_index: int,
) -> tuple[Graph, ...]:
    if len(graphs) < 4:
        raise Gate12C2GeneratorError("N1 requires at least four graphs")
    donors = tuple(
        _derangement(
            len(graphs),
            rng(
                seed_namespace,
                case_id,
                regime,
                outer_index,
                "N1",
                draw_index,
                role,
            ),
        )
        for role in range(3)
    )
    result: list[Graph] = []
    for graph_index, graph in enumerate(graphs):
        frames = tuple(
            graphs[int(donors[role][graph_index])].frames[role]
            for role in range(3)
        )
        result.append(
            Graph.from_frames(
                replicate_id=f"{graph.replicate_id}:N1:d{draw_index:03d}",
                regime=f"{regime}_N1",
                frames=frames,
            )
        )
    return tuple(result)


def joint_realizability_error(graph: Graph) -> float:
    reconstructed = edges_from_frames(graph.frames)
    return max(
        float(np.linalg.norm(expected - actual, ord="fro"))
        for expected, actual in zip(reconstructed, graph.edges)
    )


def gauge_transform(
    graph: Graph,
    *,
    seed_namespace: str,
    case_id: str,
    dataset_index: int,
    graph_index: int,
) -> Graph:
    transformed: list[np.ndarray] = []
    for role, frame in enumerate(graph.frames):
        local_rank = frame.shape[1]
        gauge = orthonormalize(
            rng(
                seed_namespace,
                case_id,
                "gauge",
                dataset_index,
                graph_index,
                role,
            ).normal(size=(local_rank, local_rank))
        )
        transformed.append(frame @ gauge)
    return Graph.from_frames(
        replicate_id=f"{graph.replicate_id}:gauge",
        regime=f"{graph.regime}_gauge",
        frames=transformed,
    )


def independent_edge_reorientation(
    edges: Sequence[np.ndarray],
    *,
    seed_namespace: str,
    case_id: str,
    outer_index: int,
    draw_index: int,
    graph_index: int,
    trial_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result: list[np.ndarray] = []
    for edge_index, edge in enumerate(edges):
        singular = np.linalg.svd(edge, full_matrices=False, compute_uv=False)
        rank = len(singular)
        left = orthonormalize(
            rng(
                seed_namespace,
                case_id,
                "S2",
                outer_index,
                draw_index,
                graph_index,
                trial_index,
                edge_index,
                "left",
            ).normal(size=(rank, rank))
        )
        right = orthonormalize(
            rng(
                seed_namespace,
                case_id,
                "S2",
                outer_index,
                draw_index,
                graph_index,
                trial_index,
                edge_index,
                "right",
            ).normal(size=(rank, rank))
        )
        result.append(left @ np.diag(singular) @ right.T)
    return tuple(result)  # type: ignore[return-value]


def edge_spectrum_error(
    before: Sequence[np.ndarray], after: Sequence[np.ndarray]
) -> float:
    return max(
        float(
            np.max(
                np.abs(
                    np.linalg.svd(left, full_matrices=False, compute_uv=False)
                    - np.linalg.svd(right, full_matrices=False, compute_uv=False)
                )
            )
        )
        for left, right in zip(before, after)
    )
